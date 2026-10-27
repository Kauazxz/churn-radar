"""Fit metric weights from historical outcomes with a logistic regression on normalized metrics.

Every feature is the metric's `goodness` in [0, 1] (1 = healthy), so coefficients are on the same
scale and a negative coefficient means "more of this good thing, less churn". The risk weight of a
metric is `max(0, -coef)`; metrics whose coefficient has the wrong sign get weight 0 and a warning,
because the health score is a weighted average of goodness and cannot express "good is bad".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from churn_radar.scoring import (
    Customer,
    MetricSpec,
    ProbabilityModel,
    ScoringConfig,
    goodness,
)

MIN_PER_CLASS = 10


class CalibrationError(ValueError):
    pass


class HoldoutAuc(BaseModel):
    uniform_weights: float = Field(description="Every metric weighted equally")
    current_config: float = Field(description="Weights in the config before calibration")
    calibrated: float = Field(description="Weights fitted on the training split")


class CalibrationResult(BaseModel):
    weights: dict[str, float]
    coefficients: dict[str, float]
    intercept: float
    auc: HoldoutAuc
    n_train: int
    n_test: int
    n_dropped_incomplete: int
    churn_rate: float
    warnings: list[str]
    config: ScoringConfig
    applied: bool = False


def _matrix(
    customers: list[Customer], specs: list[MetricSpec]
) -> tuple[NDArray[np.float64], list[int]]:
    rows: list[list[float]] = []
    kept: list[int] = []
    for i, c in enumerate(customers):
        values = [c.metrics.get(s.name) for s in specs]
        if all(v is not None and np.isfinite(v) for v in values):
            rows.append([goodness(float(v), s) for v, s in zip(values, specs, strict=True)])  # type: ignore[arg-type]
            kept.append(i)
    return np.asarray(rows, dtype=float).reshape(len(rows), len(specs)), kept


def _auc(y: NDArray[np.int_], x: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    # Lower health means higher risk, so the risk score is the negated weighted goodness.
    return round(float(roc_auc_score(y, -(x @ weights))), 4)


def calibrate(
    customers: list[Customer],
    churned: list[bool],
    config: ScoringConfig,
    *,
    test_size: float = 0.3,
    seed: int = 42,
    regularization_c: float = 1.0,
) -> CalibrationResult:
    if len(customers) != len(churned):
        raise CalibrationError("customers and churned labels must have the same length")
    specs = config.metrics
    x, kept = _matrix(customers, specs)
    y = np.asarray([int(churned[i]) for i in kept], dtype=int)
    positives = int(y.sum())
    if min(positives, len(y) - positives) < MIN_PER_CLASS:
        raise CalibrationError(
            f"need at least {MIN_PER_CLASS} churned and {MIN_PER_CLASS} retained customers with "
            f"every metric filled; got {positives} churned of {len(y)} complete rows"
        )

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = LogisticRegression(C=regularization_c, max_iter=1000, random_state=seed)
    model.fit(x_tr, y_tr)
    coef = model.coef_[0]

    warnings: list[str] = []
    raw = np.maximum(0.0, -coef)
    for spec, c in zip(specs, coef, strict=True):
        if c > 0:
            warnings.append(
                f"{spec.name}: once the other metrics are accounted for, a better value went with "
                "more churn in this data; weight set to 0 (check direction, bounds, collinearity)"
            )
    if raw.sum() <= 0:
        raise CalibrationError("no metric predicts churn in its configured direction")

    fitted = raw / raw.sum()
    uniform = np.full(len(specs), 1.0 / len(specs))
    current = np.asarray([m.weight for m in specs]) / sum(m.weight for m in specs)

    weights = {s.name: round(float(w), 4) for s, w in zip(specs, fitted, strict=True)}
    new_config = config.model_copy(
        update={
            "metrics": [s.model_copy(update={"weight": weights[s.name]}) for s in specs],
            # Exact when no coefficient was clipped: sum(coef * g) = -slope * health / 100.
            "probability_model": ProbabilityModel(
                intercept=round(float(model.intercept_[0]), 4), slope=round(float(raw.sum()), 4)
            ),
        }
    )
    return CalibrationResult(
        weights=weights,
        coefficients={s.name: round(float(c), 4) for s, c in zip(specs, coef, strict=True)},
        intercept=round(float(model.intercept_[0]), 4),
        auc=HoldoutAuc(
            uniform_weights=_auc(y_te, x_te, uniform),
            current_config=_auc(y_te, x_te, current),
            calibrated=_auc(y_te, x_te, fitted),
        ),
        n_train=len(y_tr),
        n_test=len(y_te),
        n_dropped_incomplete=len(customers) - len(kept),
        churn_rate=round(positives / len(y), 4),
        warnings=warnings,
        config=ScoringConfig.model_validate(new_config.model_dump()),
    )
