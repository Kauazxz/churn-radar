"""In-memory, per-tenant state. Deliberately simple: one process, no persistence.

Limits (see README): state is lost on restart, is not shared between worker processes, and the
number of tenants is capped so a stream of random `X-Tenant-Id` values cannot exhaust memory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from datetime import datetime

from churn_radar.scoring import CustomerScore, ScoringConfig, default_config


class TenantLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class TenantState:
    config: ScoringConfig = field(default_factory=default_config)
    last_scores: list[CustomerScore] = field(default_factory=list)
    scored_at: datetime | None = None


class TenantStore:
    def __init__(self, max_tenants: int) -> None:
        self._max = max_tenants
        self._tenants: dict[str, TenantState] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str) -> TenantState:
        with self._lock:
            state = self._tenants.get(tenant_id)
            if state is None:
                if len(self._tenants) >= self._max:
                    raise TenantLimitError(f"tenant limit reached ({self._max})")
                state = self._tenants[tenant_id] = TenantState()
            return state

    def update(self, tenant_id: str, **changes: object) -> TenantState:
        """Swap in a new immutable snapshot, so a reader never sees a half-written batch."""
        with self._lock:
            state = self._tenants[tenant_id] = replace(self._tenants[tenant_id], **changes)  # type: ignore[arg-type]
            return state
