import pytest

from churn_radar.calibration import CalibrationError, calibrate
from churn_radar.csv_io import read_customers
from churn_radar.scoring import Customer, Direction, ScoringConfig

Labeled = tuple[list[Customer], list[bool]]


@pytest.fixture
def labeled(sample_csv: str, config: ScoringConfig) -> Labeled:
    return read_customers(sample_csv, [m.name for m in config.metrics], with_label=True)


def test_calibration_beats_uniform_weights_on_sample(
    labeled: Labeled, config: ScoringConfig
) -> None:
    customers, churned = labeled
    result = calibrate(customers, churned, config, seed=42)
    assert result.auc.calibrated > result.auc.uniform_weights
    assert result.auc.calibrated > 0.8
    assert sum(result.weights.values()) == pytest.approx(1, abs=1e-3)
    assert result.n_train + result.n_test == 200
    # The fitted config must be usable as-is.
    assert result.config.probability_model is not None
    assert {m.name: m.weight for m in result.config.metrics} == result.weights


def test_calibration_is_deterministic(labeled: Labeled, config: ScoringConfig) -> None:
    customers, churned = labeled
    first = calibrate(customers, churned, config, seed=7)
    second = calibrate(customers, churned, config, seed=7)
    assert first.model_dump() == second.model_dump()


def test_wrong_direction_gets_zero_weight_and_warning(
    labeled: Labeled, config: ScoringConfig
) -> None:
    customers, churned = labeled
    flipped = config.model_copy(deep=True)
    login = next(m for m in flipped.metrics if m.name == "days_since_last_login")
    login.direction = Direction.HIGHER_IS_BETTER
    result = calibrate(customers, churned, flipped)
    assert result.weights["days_since_last_login"] == 0
    assert any(w.startswith("days_since_last_login") for w in result.warnings)


def test_too_few_examples_rejected(labeled: Labeled, config: ScoringConfig) -> None:
    customers, churned = labeled
    with pytest.raises(CalibrationError, match="at least"):
        calibrate(customers[:15], churned[:15], config)
