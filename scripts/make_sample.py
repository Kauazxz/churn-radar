"""Generate examples/customers.csv: synthetic B2B SaaS accounts with a churned label.

The data is made up. In this toy world churn is driven mostly by falling usage, inactivity and
late payments; NPS is a moderate signal and tickets/seat utilization are weak, which gives the
calibration endpoint something real to learn. Fixed seed, so the file is reproducible:

    python scripts/make_sample.py [--rows 200] [--seed 7] [--out examples/customers.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from churn_radar.csv_io import write_customers
from churn_radar.scoring import Customer, default_config

ADJECTIVES = ["Blue", "Swift", "Bright", "Northern", "Quiet", "Bold", "Silver", "Green", "Prime"]
NOUNS = ["Labs", "Logistics", "Foods", "Health", "Studio", "Retail", "Systems", "Media", "Energy"]


def generate(rows: int, seed: int) -> tuple[list[Customer], list[bool]]:
    rng = np.random.default_rng(seed)
    engagement = rng.normal(0, 1, rows)  # hidden "how much they rely on the product"
    finance = rng.normal(0, 1, rows)  # hidden "how healthy their own business is"

    login = np.clip(np.round(np.exp(2.0 - 0.9 * engagement + rng.normal(0, 0.35, rows))), 0, 90)
    usage = np.clip(0.22 * engagement + rng.normal(0, 0.08, rows), -0.8, 0.8)
    delays = rng.poisson(np.exp(-0.4 - 1.0 * finance))
    nps = np.clip(np.round(7 + 1.0 * engagement + rng.normal(0, 2.2, rows)), 0, 10)
    tickets = rng.poisson(np.exp(0.9 - 0.1 * engagement))
    seats = np.clip(0.65 + 0.03 * engagement + rng.normal(0, 0.22, rows), 0.05, 1.0)
    mrr = np.round(np.exp(rng.normal(6.3, 0.8, rows)), 2)

    logit = -1.2 - 2.2 * engagement - 1.3 * finance
    churned = rng.random(rows) < 1 / (1 + np.exp(-logit))

    customers = [
        Customer(
            customer_id=f"C{i + 1:04d}",
            name=f"{ADJECTIVES[i % len(ADJECTIVES)]} {NOUNS[(i * 7) % len(NOUNS)]} {i + 1}",
            mrr=float(mrr[i]),
            metrics={
                "days_since_last_login": float(login[i]),
                "tickets_last_30d": float(tickets[i]),
                "usage_trend": round(float(usage[i]), 3),
                "nps": float(nps[i]),
                "payment_delays": float(delays[i]),
                "seats_utilization": round(float(seats[i]), 3),
            },
        )
        for i in range(rows)
    ]
    return customers, [bool(c) for c in churned]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("examples/customers.csv"))
    args = parser.parse_args()

    customers, churned = generate(args.rows, args.seed)
    names = [m.name for m in default_config().metrics]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(write_customers(customers, names, churned), encoding="utf-8")
    print(f"wrote {len(customers)} rows to {args.out} ({sum(churned)} churned)")


if __name__ == "__main__":
    main()
