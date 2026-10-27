"""HTTP layer: validation, tenants and formats. The math lives in `scoring` and `calibration`."""

import os
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from churn_radar import __version__
from churn_radar.calibration import CalibrationError, CalibrationResult, calibrate
from churn_radar.csv_io import CsvError, read_customers, write_scores
from churn_radar.scoring import (
    Customer,
    CustomerScore,
    ScoringConfig,
    build_queue,
    score_batch,
    summarize,
)
from churn_radar.state import TenantLimitError, TenantStore

MAX_BATCH = 10_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class ScoreRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class ScoreResponse(BaseModel):
    scored_at: datetime
    summary: dict[str, object]
    results: list[CustomerScore]


class ConfigResponse(BaseModel):
    config: ScoringConfig
    normalized_weights: dict[str, float]


class LabeledCustomer(Customer):
    churned: bool


class CalibrateRequest(BaseModel):
    customers: list[LabeledCustomer] = Field(min_length=1, max_length=MAX_BATCH)
    test_size: float = Field(default=0.3, ge=0.1, le=0.5)
    seed: int = 42
    apply: bool = False


class QueueResponse(BaseModel):
    scored_at: datetime | None
    total: int
    items: list[CustomerScore]


TenantHeader = Annotated[
    str,
    Header(alias="X-Tenant-Id", pattern=r"^[A-Za-z0-9_.-]{1,64}$", description="Tenant key"),
]


def tenant(request: Request, x_tenant_id: TenantHeader = "default") -> str:
    try:
        request.app.state.store.get(x_tenant_id)
    except TenantLimitError as exc:
        raise HTTPException(429, str(exc)) from exc
    return x_tenant_id


Tenant = Annotated[str, Depends(tenant)]


def store(request: Request) -> TenantStore:
    result: TenantStore = request.app.state.store
    return result


Store = Annotated[TenantStore, Depends(store)]


def create_app(max_tenants: int | None = None) -> FastAPI:
    app = FastAPI(
        title="churn-radar",
        version=__version__,
        summary="Customer health scoring and churn-risk prioritization API.",
    )
    limit = max_tenants or int(os.getenv("CHURN_RADAR_MAX_TENANTS", "1000"))
    app.state.store = TenantStore(max_tenants=limit)

    async def read_upload(file: UploadFile) -> str:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES} bytes")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(422, "CSV must be UTF-8 encoded") from exc

    def run_scoring(db: TenantStore, tenant_id: str, customers: list[Customer]) -> ScoreResponse:
        if len(customers) > MAX_BATCH:
            raise HTTPException(413, f"at most {MAX_BATCH} customers per batch")
        try:
            scores = score_batch(customers, db.get(tenant_id).config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        now = datetime.now(UTC)
        db.update(tenant_id, last_scores=scores, scored_at=now)
        return ScoreResponse(scored_at=now, summary=summarize(scores), results=scores)

    def run_calibration(
        db: TenantStore,
        tenant_id: str,
        customers: list[Customer],
        labels: list[bool],
        test_size: float,
        seed: int,
        apply: bool,
    ) -> CalibrationResult:
        try:
            result = calibrate(
                customers, labels, db.get(tenant_id).config, test_size=test_size, seed=seed
            )
        except CalibrationError as exc:
            raise HTTPException(422, str(exc)) from exc
        if apply:
            db.update(tenant_id, config=result.config)
            result.applied = True
        return result

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/score", tags=["scoring"])
    def score(body: ScoreRequest, tenant_id: Tenant, db: Store) -> ScoreResponse:
        return run_scoring(db, tenant_id, body.customers)

    @app.post(
        "/score/csv",
        tags=["scoring"],
        response_model=ScoreResponse,
        responses={200: {"content": {"text/csv": {}}}},
    )
    async def score_csv(
        tenant_id: Tenant,
        db: Store,
        file: Annotated[UploadFile, File(description="CSV with customer_id, name, mrr, metrics")],
        output: Annotated[Literal["json", "csv"], Query(alias="format")] = "json",
    ) -> ScoreResponse | Response:
        text = await read_upload(file)
        names = [m.name for m in db.get(tenant_id).config.metrics]
        try:
            customers, _ = read_customers(text, names)
        except CsvError as exc:
            raise HTTPException(422, str(exc)) from exc
        result = run_scoring(db, tenant_id, customers)
        if output == "csv":
            return Response(
                write_scores(result.results),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="scores.csv"'},
            )
        return result

    @app.get("/config", tags=["config"])
    def get_config(tenant_id: Tenant, db: Store) -> ConfigResponse:
        config = db.get(tenant_id).config
        return ConfigResponse(config=config, normalized_weights=config.normalized_weights())

    @app.put("/config", tags=["config"])
    def put_config(body: ScoringConfig, tenant_id: Tenant, db: Store) -> ConfigResponse:
        db.update(tenant_id, config=body)
        return ConfigResponse(config=body, normalized_weights=body.normalized_weights())

    @app.post("/calibrate", tags=["calibration"])
    def calibrate_json(body: CalibrateRequest, tenant_id: Tenant, db: Store) -> CalibrationResult:
        customers = [
            Customer.model_validate(c.model_dump(exclude={"churned"})) for c in body.customers
        ]
        labels = [c.churned for c in body.customers]
        return run_calibration(
            db, tenant_id, customers, labels, body.test_size, body.seed, body.apply
        )

    @app.post("/calibrate/csv", tags=["calibration"])
    async def calibrate_csv(
        tenant_id: Tenant,
        db: Store,
        file: Annotated[UploadFile, File(description="Same columns as /score/csv plus churned")],
        test_size: Annotated[float, Query(ge=0.1, le=0.5)] = 0.3,
        seed: int = 42,
        apply: bool = False,
    ) -> CalibrationResult:
        text = await read_upload(file)
        names = [m.name for m in db.get(tenant_id).config.metrics]
        try:
            customers, labels = read_customers(text, names, with_label=True)
        except CsvError as exc:
            raise HTTPException(422, str(exc)) from exc
        return run_calibration(db, tenant_id, customers, labels, test_size, seed, apply)

    @app.get("/queue", tags=["scoring"])
    def queue(
        tenant_id: Tenant, db: Store, limit: Annotated[int, Query(ge=1, le=1000)] = 20
    ) -> QueueResponse:
        state = db.get(tenant_id)
        return QueueResponse(
            scored_at=state.scored_at,
            total=len(state.last_scores),
            items=build_queue(state.last_scores, limit),
        )

    return app


app = create_app()


def serve() -> None:
    """Console entry point: `churn-radar` (HOST/PORT from the environment)."""
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
