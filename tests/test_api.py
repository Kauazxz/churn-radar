from typing import Any

from fastapi.testclient import TestClient

CUSTOMERS: list[dict[str, Any]] = [
    {"customer_id": "low", "mrr": 100, "metrics": {"days_since_last_login": 50, "nps": 2}},
    {"customer_id": "big", "mrr": 9000, "metrics": {"days_since_last_login": 20, "nps": 6}},
    {"customer_id": "fine", "mrr": 500, "metrics": {"days_since_last_login": 0, "nps": 10}},
]


def tenant(name: str) -> dict[str, str]:
    return {"X-Tenant-Id": name}


def test_healthz_and_openapi(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/score", "/score/csv", "/config", "/calibrate", "/queue"} <= set(paths)


def test_score_returns_results_in_input_order_and_summary(client: TestClient) -> None:
    body = client.post("/score", json={"customers": CUSTOMERS}).json()
    assert [r["customer_id"] for r in body["results"]] == ["low", "big", "fine"]
    assert body["summary"]["customers"] == 3
    assert sum(body["summary"]["by_tier"].values()) == 3
    assert body["results"][0]["missing_metrics"]  # only 2 of 6 metrics were sent


def test_queue_is_top_n_by_revenue_at_risk(client: TestClient) -> None:
    assert client.get("/queue").json() == {"scored_at": None, "total": 0, "items": []}
    client.post("/score", json={"customers": CUSTOMERS})
    queue = client.get("/queue", params={"limit": 2}).json()
    assert queue["total"] == 3
    assert [i["customer_id"] for i in queue["items"]] == ["big", "low"]


def test_put_config_validates_weights(client: TestClient) -> None:
    config = client.get("/config").json()["config"]
    for metric in config["metrics"]:
        metric["weight"] = 0
    assert client.put("/config", json=config).status_code == 422

    config["metrics"][0]["weight"] = 2
    config["metrics"][1]["weight"] = 6
    body = client.put("/config", json=config).json()
    assert body["normalized_weights"][config["metrics"][1]["name"]] == 0.75


def test_unknown_config_fields_rejected(client: TestClient) -> None:
    config = client.get("/config").json()["config"]
    config["metrics"][0]["wieght"] = 1
    assert client.put("/config", json=config).status_code == 422


def test_tenants_are_isolated(client: TestClient) -> None:
    config = client.get("/config", headers=tenant("acme")).json()["config"]
    config["top_factors"] = 1
    assert client.put("/config", json=config, headers=tenant("acme")).status_code == 200
    client.post("/score", json={"customers": CUSTOMERS}, headers=tenant("acme"))

    assert client.get("/config", headers=tenant("globex")).json()["config"]["top_factors"] == 3
    assert client.get("/queue", headers=tenant("globex")).json()["total"] == 0
    acme_queue = client.get("/queue", headers=tenant("acme")).json()
    assert acme_queue["total"] == 3
    assert all(len(i["top_factors"]) <= 1 for i in acme_queue["items"])


def test_tenant_header_is_validated_and_capped(client: TestClient) -> None:
    assert client.get("/config", headers=tenant("not valid!")).status_code == 422
    for i in range(5):  # the test app allows 5 tenants
        assert client.get("/config", headers=tenant(f"t{i}")).status_code == 200
    assert client.get("/config", headers=tenant("t5")).status_code == 429


def test_calibrate_csv_and_apply(client: TestClient, sample_csv: str) -> None:
    upload = {"file": ("customers.csv", sample_csv, "text/csv")}
    dry = client.post("/calibrate/csv", files=upload).json()
    assert dry["applied"] is False
    assert client.get("/config").json()["config"]["probability_model"] is None

    applied = client.post("/calibrate/csv", params={"apply": "true"}, files=upload).json()
    assert applied["auc"]["calibrated"] > applied["auc"]["uniform_weights"]
    config = client.get("/config").json()["config"]
    assert {m["name"]: m["weight"] for m in config["metrics"]} == applied["weights"]
    assert config["probability_model"] is not None


def test_calibrate_json_needs_both_classes(client: TestClient) -> None:
    rows = [{**c, "churned": False} for c in CUSTOMERS]
    response = client.post("/calibrate", json={"customers": rows})
    assert response.status_code == 422
    assert "at least" in response.json()["detail"]
