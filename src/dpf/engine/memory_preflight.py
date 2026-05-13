"""Launch-time memory preflight for solver runs."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from math import prod
from typing import Any


@dataclass(frozen=True)
class MemoryPreflightResult:
    """Projected memory demand and launch decision."""

    projected_bytes: int
    available_bytes: int
    limit_bytes: int
    limit_fraction: float
    required_fraction: float
    passed: bool
    override: bool
    reason: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def system_available_memory_bytes() -> int:
    """Return available system memory, falling back to total memory when needed."""

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except Exception:
            pass
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except Exception:
            pass

    return 0


def estimate_run_memory_bytes(config: Any, backend: str | None = None) -> int:
    """Estimate resident memory needed for state, solver workspaces, and diagnostics."""

    resolved_backend = backend or config.fluid.backend
    cells = int(prod(int(n) for n in config.grid_shape))
    precision = getattr(config.fluid, "precision", "float32")
    dtype_bytes = 8 if precision == "float64" or resolved_backend in {"python", "athena"} else 4

    # Engine state: rho, pressure, Te, Ti, psi, velocity(3), B(3), plus optional e_electron.
    component_count = 11 + (1 if getattr(config.fluid, "two_temperature", False) else 0)

    workspace_multiplier = {
        "python": 8.0,
        "athena": 6.0,
        "athenak": 6.0,
        "hybrid": 8.0,
        "metal": 10.0,
        "mlx": 10.0,
        "auto": 8.0,
    }.get(resolved_backend, 8.0)

    field_snapshot_multiplier = 1.0
    if getattr(config.diagnostics, "field_output_interval", 0) > 0:
        field_snapshot_multiplier += 1.0

    return int(cells * component_count * dtype_bytes * workspace_multiplier * field_snapshot_multiplier)


def run_memory_preflight(
    config: Any,
    backend: str,
    *,
    available_bytes: int | None = None,
) -> MemoryPreflightResult:
    """Compute memory preflight and fail closed unless override is enabled."""

    projected = estimate_run_memory_bytes(config, backend)
    available = int(available_bytes if available_bytes is not None else system_available_memory_bytes())
    limit_fraction = float(getattr(config.diagnostics, "memory_limit_fraction", 0.70))
    limit = int(max(0, available) * limit_fraction)
    override = bool(getattr(config.diagnostics, "allow_memory_overcommit", False))
    if available <= 0:
        return MemoryPreflightResult(
            projected_bytes=projected,
            available_bytes=available,
            limit_bytes=0,
            limit_fraction=limit_fraction,
            required_fraction=0.0,
            passed=True,
            override=override,
            reason="available memory unavailable; preflight recorded but not enforced",
        )

    required_fraction = projected / max(available, 1)
    passed = projected <= limit or override
    reason = "projected memory within launch limit"
    if projected > limit and override:
        reason = "projected memory exceeds launch limit; explicit overcommit enabled"
    elif projected > limit:
        reason = "projected memory exceeds launch limit"

    return MemoryPreflightResult(
        projected_bytes=projected,
        available_bytes=available,
        limit_bytes=limit,
        limit_fraction=limit_fraction,
        required_fraction=required_fraction,
        passed=passed,
        override=override,
        reason=reason,
    )
