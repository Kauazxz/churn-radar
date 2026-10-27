"""Pure scoring core: normalization, health score, risk tiers, factors and the follow-up queue.

Nothing in this module knows about HTTP or tenants, so it can be tested and reused on its own.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetricName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class Direction(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    HIGHER_IS_WORSE = "higher_is_worse"


class Tier(StrEnum):
    HEALTHY = "healthy"
    WATCH = "watch"
    AT_RISK = "at_risk"
    CRITICAL = "critical"


class MetricSpec(BaseModel):
    """One input signal: how to read it, where to clip it and how much it matters."""

    model_config = ConfigDict(extra="forbid")

    name: MetricName
    direction: Direction
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    weight: float = Field(ge=0, allow_inf_nan=False)
    description: str = ""

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.upper <= self.lower:
            raise ValueError(f"{self.name}: upper bound must be greater than lower bound")
        return self


class TierThresholds(BaseModel):
    """Minimum health score for each tier; anything below `at_risk` is critical."""

    model_config = ConfigDict(extra="forbid")

    healthy: float = Field(default=75, gt=0, lt=100)
    watch: float = Field(default=55, gt=0, lt=100)
    at_risk: float = Field(default=35, gt=0, lt=100)

    @model_validator(mode="after")
    def _descending(self) -> Self:
        if not self.healthy > self.watch > self.at_risk:
            raise ValueError("thresholds must satisfy healthy > watch > at_risk")
        return self


class ProbabilityModel(BaseModel):
    """Logistic link fitted by calibration: p = sigmoid(intercept - slope * health / 100)."""

    model_config = ConfigDict(extra="forbid")

    intercept: float = Field(allow_inf_nan=False)
    slope: float = Field(ge=0, allow_inf_nan=False)


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricSpec] = Field(min_length=1, max_length=50)
    thresholds: TierThresholds = TierThresholds()
    top_factors: int = Field(default=3, ge=1, le=10)
    probability_model: ProbabilityModel | None = None

    @model_validator(mode="after")
    def _weights(self) -> Self:
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("metric names must be unique")
        if sum(m.weight for m in self.metrics) <= 0:
            raise ValueError("at least one metric needs a positive weight")
        return self

    def normalized_weights(self) -> dict[str, float]:
        total = sum(m.weight for m in self.metrics)
        return {m.name: m.weight / total for m in self.metrics}


def default_config() -> ScoringConfig:
    """A hand-tuned starting point. `POST /calibrate` can replace the weights with fitted ones."""
    hib, hiw = Direction.HIGHER_IS_BETTER, Direction.HIGHER_IS_WORSE
    specs = [
        ("days_since_last_login", hiw, 0, 60, 0.25, "Days since any user of the account logged in"),
        ("tickets_last_30d", hiw, 0, 10, 0.10, "Support tickets opened in the last 30 days"),
        ("usage_trend", hib, -0.5, 0.5, 0.25, "Usage change vs. previous 30 days (-0.2 = -20%)"),
        ("nps", hib, 0, 10, 0.15, "Latest NPS answer (0-10)"),
        ("payment_delays", hiw, 0, 3, 0.15, "Late invoices in the last 6 months"),
        ("seats_utilization", hib, 0, 1, 0.10, "Active seats / purchased seats"),
    ]
    return ScoringConfig(
        metrics=[
            MetricSpec(name=n, direction=d, lower=lo, upper=hi, weight=w, description=desc)
            for n, d, lo, hi, w, desc in specs
        ]
    )


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=128)
    name: str = Field(default="", max_length=256)
    mrr: float = Field(ge=0, allow_inf_nan=False)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class Factor(BaseModel):
    metric: str
    value: float
    goodness: float = Field(description="Normalized value in [0, 1], where 1 is the best case")
    weight: float = Field(description="Weight after normalizing the present metrics to sum 1")
    points_lost: float = Field(description="Health points this metric removed from 100")


class CustomerScore(BaseModel):
    customer_id: str
    name: str
    mrr: float
    health_score: float
    tier: Tier
    churn_probability: float
    revenue_at_risk: float
    top_factors: list[Factor]
    missing_metrics: list[str]


def goodness(value: float, spec: MetricSpec) -> float:
    """Clip to the bounds, scale to [0, 1] and flip so that 1 always means 'healthy'."""
    clipped = min(max(value, spec.lower), spec.upper)
    scaled = (clipped - spec.lower) / (spec.upper - spec.lower)
    return scaled if spec.direction is Direction.HIGHER_IS_BETTER else 1.0 - scaled


def tier_for(health: float, thresholds: TierThresholds) -> Tier:
    if health >= thresholds.healthy:
        return Tier.HEALTHY
    if health >= thresholds.watch:
        return Tier.WATCH
    if health >= thresholds.at_risk:
        return Tier.AT_RISK
    return Tier.CRITICAL


def churn_probability(health: float, model: ProbabilityModel | None) -> float:
    """Without calibration this is only a linear proxy (1 - health/100), not a probability."""
    if model is None:
        return 1.0 - health / 100.0
    z = model.intercept - model.slope * health / 100.0
    return 1.0 / (1.0 + math.exp(-z))


def score_customer(customer: Customer, config: ScoringConfig) -> CustomerScore:
    present: list[tuple[MetricSpec, float]] = []
    missing: list[str] = []
    for spec in config.metrics:
        raw = customer.metrics.get(spec.name)
        if raw is None or not math.isfinite(raw):
            missing.append(spec.name)
        elif spec.weight > 0:
            present.append((spec, raw))
    total_weight = sum(spec.weight for spec, _ in present)
    if total_weight <= 0:
        raise ValueError(f"customer {customer.customer_id!r} has no weighted metric to score")

    factors: list[Factor] = []
    health = 0.0
    for spec, raw in present:
        w = spec.weight / total_weight
        g = goodness(raw, spec)
        health += 100.0 * w * g
        factors.append(
            Factor(
                metric=spec.name,
                value=raw,
                goodness=round(g, 4),
                weight=round(w, 4),
                points_lost=round(100.0 * w * (1.0 - g), 2),
            )
        )
    # Biggest loss first; the metric name breaks ties so the output is deterministic.
    factors.sort(key=lambda f: (-f.points_lost, f.metric))
    top = [f for f in factors if f.points_lost > 0][: config.top_factors]

    p = churn_probability(health, config.probability_model)
    return CustomerScore(
        customer_id=customer.customer_id,
        name=customer.name,
        mrr=customer.mrr,
        health_score=round(health, 2),
        tier=tier_for(health, config.thresholds),
        churn_probability=round(p, 4),
        revenue_at_risk=round(customer.mrr * p, 2),
        top_factors=top,
        missing_metrics=missing,
    )


def score_batch(customers: list[Customer], config: ScoringConfig) -> list[CustomerScore]:
    return [score_customer(c, config) for c in customers]


def build_queue(scores: list[CustomerScore], limit: int | None = None) -> list[CustomerScore]:
    """Who to call first: most revenue at risk, then lowest health, then id for stability."""
    ordered = sorted(scores, key=lambda s: (-s.revenue_at_risk, s.health_score, s.customer_id))
    return ordered if limit is None else ordered[:limit]


def summarize(scores: list[CustomerScore]) -> dict[str, object]:
    by_tier = {t.value: 0 for t in Tier}
    for s in scores:
        by_tier[s.tier.value] += 1
    return {
        "customers": len(scores),
        "by_tier": by_tier,
        "total_mrr": round(sum(s.mrr for s in scores), 2),
        "total_revenue_at_risk": round(sum(s.revenue_at_risk for s in scores), 2),
    }
