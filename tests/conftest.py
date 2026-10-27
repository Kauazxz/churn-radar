from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn_radar.api import create_app
from churn_radar.scoring import (
    Customer,
    Direction,
    MetricSpec,
    ScoringConfig,
    default_config,
)

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "examples" / "customers.csv"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(max_tenants=5))


@pytest.fixture
def sample_csv() -> str:
    return SAMPLE_CSV.read_text(encoding="utf-8")


@pytest.fixture
def config() -> ScoringConfig:
    return default_config()


@pytest.fixture
def two_metric_config() -> ScoringConfig:
    """Small config with round numbers so expected scores can be computed by hand."""
    return ScoringConfig(
        metrics=[
            MetricSpec(
                name="usage", direction=Direction.HIGHER_IS_BETTER, lower=0, upper=100, weight=3
            ),
            MetricSpec(
                name="tickets", direction=Direction.HIGHER_IS_WORSE, lower=0, upper=10, weight=1
            ),
        ]
    )


def make_customer(cid: str, mrr: float, **metrics: float | None) -> Customer:
    return Customer(customer_id=cid, mrr=mrr, metrics=metrics)
