"""CSV in and out. Columns: customer_id, name, mrr, one column per metric, optional churned."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

from pydantic import ValidationError

from churn_radar.scoring import Customer, CustomerScore

ID_COLUMNS = ("customer_id", "name", "mrr")
TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


class CsvError(ValueError):
    """Raised with a row number so the API can say exactly what is wrong."""


def _number(text: str, column: str, row: int) -> float | None:
    text = text.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        raise CsvError(f"row {row}: column {column!r} is not a number: {text!r}") from None


def read_customers(
    text: str, metric_names: Iterable[str], *, with_label: bool = False
) -> tuple[list[Customer], list[bool]]:
    """Parse customers (and the `churned` label when `with_label`). Unknown columns are ignored."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    header = set(reader.fieldnames or [])
    required = {"customer_id", "mrr"} | ({"churned"} if with_label else set())
    if missing := sorted(required - header):
        raise CsvError(f"missing required column(s): {', '.join(missing)}")
    metric_columns = [m for m in metric_names if m in header]
    if not metric_columns:
        raise CsvError("no metric column from the scoring config was found in the header")

    customers: list[Customer] = []
    labels: list[bool] = []
    for row_no, row in enumerate(reader, start=2):  # row 1 is the header
        mrr = _number(row.get("mrr") or "", "mrr", row_no)
        if mrr is None:
            raise CsvError(f"row {row_no}: mrr is required")
        try:
            customers.append(
                Customer(
                    customer_id=(row.get("customer_id") or "").strip(),
                    name=(row.get("name") or "").strip(),
                    mrr=mrr,
                    metrics={m: _number(row.get(m) or "", m, row_no) for m in metric_columns},
                )
            )
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(p) for p in first["loc"])
            raise CsvError(f"row {row_no}: {field}: {first['msg']}") from None
        if with_label:
            flag = (row.get("churned") or "").strip().lower()
            if flag not in TRUE_VALUES | FALSE_VALUES:
                raise CsvError(f"row {row_no}: churned must be 0/1 or true/false, got {flag!r}")
            labels.append(flag in TRUE_VALUES)
    if not customers:
        raise CsvError("the CSV has a header but no rows")
    return customers, labels


def write_customers(
    customers: list[Customer], metric_names: list[str], labels: list[bool] | None = None
) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([*ID_COLUMNS, *metric_names, *(["churned"] if labels is not None else [])])
    for i, c in enumerate(customers):
        values = ["" if c.metrics.get(m) is None else repr(c.metrics[m]) for m in metric_names]
        extra = [int(labels[i])] if labels is not None else []
        writer.writerow([c.customer_id, c.name, repr(c.mrr), *values, *extra])
    return out.getvalue()


def write_scores(scores: list[CustomerScore]) -> str:
    """Flat export; factors become `metric:points_lost` joined by `;`."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        [
            *ID_COLUMNS,
            "health_score",
            "tier",
            "churn_probability",
            "revenue_at_risk",
            "top_factors",
            "missing_metrics",
        ]
    )
    for s in scores:
        factors = ";".join(f"{f.metric}:{f.points_lost}" for f in s.top_factors)
        writer.writerow(
            [
                s.customer_id,
                s.name,
                s.mrr,
                s.health_score,
                s.tier.value,
                s.churn_probability,
                s.revenue_at_risk,
                factors,
                ";".join(s.missing_metrics),
            ]
        )
    return out.getvalue()
