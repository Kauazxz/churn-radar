import csv
import io

import pytest
from fastapi.testclient import TestClient

from churn_radar.csv_io import CsvError, read_customers, write_customers
from churn_radar.scoring import ScoringConfig
from tests.conftest import make_customer


def test_customers_round_trip(config: ScoringConfig) -> None:
    names = [m.name for m in config.metrics]
    customers = [
        make_customer("A1", 199.9, days_since_last_login=3, usage_trend=-0.125, nps=None),
        make_customer("B2", 0.0, payment_delays=2, seats_utilization=0.333),
    ]
    customers[0].name = 'Acme, "the" Co'
    text = write_customers(customers, names, labels=[True, False])
    parsed, labels = read_customers(text, names, with_label=True)
    assert labels == [True, False]
    for original, back in zip(customers, parsed, strict=True):
        assert back.model_dump(exclude={"metrics"}) == original.model_dump(exclude={"metrics"})
        assert {k: v for k, v in back.metrics.items() if v is not None} == {
            k: v for k, v in original.metrics.items() if v is not None
        }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("customer_id,nps\nA,5\n", "missing required column"),
        ("customer_id,mrr,foo\nA,5,1\n", "no metric column"),
        ("customer_id,mrr,nps\nA,10,abc\n", "row 2: column 'nps' is not a number"),
        ("customer_id,mrr,nps\nA,10,5\nB,-1,5\n", "row 3: mrr"),
        ("customer_id,mrr,nps\n", "no rows"),
    ],
)
def test_bad_csv_reports_row(text: str, message: str, config: ScoringConfig) -> None:
    with pytest.raises(CsvError, match=message):
        read_customers(text, [m.name for m in config.metrics])


def test_api_csv_download_matches_json(client: TestClient, sample_csv: str) -> None:
    upload = {"file": ("customers.csv", sample_csv, "text/csv")}
    as_json = client.post("/score/csv", files=upload)
    as_csv = client.post("/score/csv", params={"format": "csv"}, files=upload)
    assert as_json.status_code == as_csv.status_code == 200
    assert as_csv.headers["content-type"].startswith("text/csv")

    rows = list(csv.DictReader(io.StringIO(as_csv.text)))
    results = as_json.json()["results"]
    assert len(rows) == len(results) == 200
    for row, result in zip(rows, results, strict=True):
        assert row["customer_id"] == result["customer_id"]
        assert float(row["health_score"]) == result["health_score"]
        assert row["tier"] == result["tier"]
        assert float(row["revenue_at_risk"]) == result["revenue_at_risk"]
        assert row["top_factors"].split(";")[0].split(":")[0] == result["top_factors"][0]["metric"]


def test_api_rejects_bad_csv(client: TestClient) -> None:
    response = client.post("/score/csv", files={"file": ("x.csv", "customer_id\nA\n", "text/csv")})
    assert response.status_code == 422
    assert "mrr" in response.json()["detail"]
