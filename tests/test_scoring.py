import math

import pytest
from pydantic import ValidationError

from churn_radar.scoring import (
    Direction,
    MetricSpec,
    ProbabilityModel,
    ScoringConfig,
    Tier,
    TierThresholds,
    build_queue,
    churn_probability,
    goodness,
    score_batch,
    score_customer,
    tier_for,
)
from tests.conftest import make_customer


def spec(direction: Direction, lower: float = 0, upper: float = 10) -> MetricSpec:
    return MetricSpec(name="m", direction=direction, lower=lower, upper=upper, weight=1)


class TestNormalization:
    def test_higher_is_better_scales_to_unit_interval(self) -> None:
        s = spec(Direction.HIGHER_IS_BETTER)
        assert [goodness(v, s) for v in (0, 5, 10)] == [0.0, 0.5, 1.0]

    def test_higher_is_worse_is_flipped(self) -> None:
        s = spec(Direction.HIGHER_IS_WORSE)
        assert [goodness(v, s) for v in (0, 2.5, 10)] == [1.0, 0.75, 0.0]

    def test_values_outside_bounds_are_clipped(self) -> None:
        s = spec(Direction.HIGHER_IS_BETTER, lower=-0.5, upper=0.5)
        assert goodness(-3, s) == 0.0
        assert goodness(9, s) == 1.0


class TestConfigValidation:
    def test_upper_must_exceed_lower(self) -> None:
        with pytest.raises(ValidationError, match="upper bound"):
            spec(Direction.HIGHER_IS_BETTER, lower=5, upper=5)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MetricSpec(name="m", direction="higher_is_worse", lower=0, upper=1, weight=-0.1)

    def test_all_zero_weights_rejected(self) -> None:
        zero = MetricSpec(name="m", direction="higher_is_worse", lower=0, upper=1, weight=0)
        with pytest.raises(ValidationError, match="positive weight"):
            ScoringConfig(metrics=[zero])

    def test_duplicate_names_rejected(self, config: ScoringConfig) -> None:
        with pytest.raises(ValidationError, match="unique"):
            ScoringConfig(metrics=[config.metrics[0], config.metrics[0]])

    def test_thresholds_must_descend(self) -> None:
        with pytest.raises(ValidationError, match="healthy > watch > at_risk"):
            TierThresholds(healthy=50, watch=60, at_risk=30)

    def test_weights_are_normalized(self, two_metric_config: ScoringConfig) -> None:
        assert two_metric_config.normalized_weights() == {"usage": 0.75, "tickets": 0.25}


@pytest.mark.parametrize(
    ("health", "tier"),
    [
        (100, Tier.HEALTHY),
        (75, Tier.HEALTHY),
        (74.99, Tier.WATCH),
        (55, Tier.WATCH),
        (54.9, Tier.AT_RISK),
        (35, Tier.AT_RISK),
        (34.99, Tier.CRITICAL),
        (0, Tier.CRITICAL),
    ],
)
def test_tiers_use_inclusive_lower_bounds(health: float, tier: Tier) -> None:
    assert tier_for(health, TierThresholds()) is tier


class TestScore:
    def test_weighted_average_by_hand(self, two_metric_config: ScoringConfig) -> None:
        # usage 40 -> goodness 0.4 (weight .75); tickets 5 -> goodness 0.5 (weight .25)
        result = score_customer(make_customer("a", 1000, usage=40, tickets=5), two_metric_config)
        assert result.health_score == pytest.approx(100 * (0.75 * 0.4 + 0.25 * 0.5))
        assert result.churn_probability == pytest.approx(1 - 0.425)
        assert result.revenue_at_risk == pytest.approx(575.0)
        assert result.tier is Tier.AT_RISK

    def test_factors_ranked_by_points_lost_and_sum_to_gap(
        self, two_metric_config: ScoringConfig
    ) -> None:
        result = score_customer(make_customer("a", 10, usage=80, tickets=10), two_metric_config)
        # usage loses 0.75 * 0.2 * 100 = 15 points; tickets loses 0.25 * 1 * 100 = 25 points
        assert [f.metric for f in result.top_factors] == ["tickets", "usage"]
        assert [f.points_lost for f in result.top_factors] == [25.0, 15.0]
        assert sum(f.points_lost for f in result.top_factors) == pytest.approx(
            100 - result.health_score
        )

    def test_perfect_customer_has_no_factors(self, two_metric_config: ScoringConfig) -> None:
        result = score_customer(make_customer("a", 10, usage=100, tickets=0), two_metric_config)
        assert (result.health_score, result.tier, result.top_factors) == (100, Tier.HEALTHY, [])

    def test_top_factors_limit(self, config: ScoringConfig) -> None:
        worst = {
            "days_since_last_login": 60,
            "tickets_last_30d": 10,
            "usage_trend": -0.5,
            "nps": 0,
            "payment_delays": 3,
            "seats_utilization": 0,
        }
        result = score_customer(make_customer("a", 10, **worst), config)
        assert result.health_score == 0
        assert result.tier is Tier.CRITICAL
        assert len(result.top_factors) == config.top_factors == 3

    def test_missing_metric_is_reported_and_weights_renormalized(
        self, two_metric_config: ScoringConfig
    ) -> None:
        result = score_customer(make_customer("a", 10, usage=60, tickets=None), two_metric_config)
        assert result.missing_metrics == ["tickets"]
        assert result.health_score == pytest.approx(60)

    def test_no_scorable_metric_raises(self, two_metric_config: ScoringConfig) -> None:
        with pytest.raises(ValueError, match="no weighted metric"):
            score_customer(make_customer("a", 10), two_metric_config)


def test_probability_model_is_monotone_sigmoid() -> None:
    model = ProbabilityModel(intercept=2.0, slope=4.0)
    assert churn_probability(50, model) == pytest.approx(0.5)
    p90, p50, p10 = (churn_probability(h, model) for h in (90, 50, 10))
    assert p90 < p50 < p10
    assert math.isclose(churn_probability(30, None), 0.7)


def test_queue_orders_by_revenue_at_risk_then_health(two_metric_config: ScoringConfig) -> None:
    customers = [
        make_customer("small-bad", 100, usage=0, tickets=10),  # risk 100 * 1.0  = 100
        make_customer("big-ok", 1000, usage=80, tickets=2),  # risk 1000 * 0.2 = 200
        make_customer("tie-worse", 400, usage=60, tickets=8),  # health 50, risk 200
        make_customer("tie-better", 1000, usage=90, tickets=5),  # health 80, risk 200
        make_customer("safe", 5000, usage=100, tickets=0),  # risk 0
    ]
    queue = build_queue(score_batch(customers, two_metric_config))
    assert [s.customer_id for s in queue] == [
        "tie-worse",
        "big-ok",
        "tie-better",
        "small-bad",
        "safe",
    ]
    assert len(build_queue(score_batch(customers, two_metric_config), limit=2)) == 2
