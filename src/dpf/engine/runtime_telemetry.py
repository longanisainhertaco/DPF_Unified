"""Runtime telemetry sampling for solver executions."""

from __future__ import annotations

import os
import resource
from dataclasses import asdict, dataclass


def process_rss_bytes() -> int | None:
    """Return current resident set size when available."""

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None

    # macOS reports bytes; Linux reports KiB. Treat very small values as KiB.
    return peak if peak > 10_000_000 else peak * 1024


def backend_memory_bytes(backend: str) -> tuple[int | None, int | None, bool, str]:
    """Return backend active/peak memory telemetry where the backend exposes it."""

    if backend != "mlx":
        return None, None, False, "backend memory telemetry is not exposed for this backend"

    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        active = int(mx.metal.get_active_memory())
        peak = int(mx.metal.get_peak_memory())
        return active, peak, True, "MLX metal telemetry available"
    except Exception as exc:
        return None, None, False, f"MLX metal telemetry unavailable: {type(exc).__name__}"


@dataclass
class RuntimeMemoryTelemetry:
    """Peak runtime memory sampled during a solver execution."""

    backend: str
    telemetry_supported: bool = False
    process_start_rss_bytes: int | None = None
    process_end_rss_bytes: int | None = None
    process_peak_rss_bytes: int | None = None
    backend_memory_supported: bool = False
    backend_active_bytes: int | None = None
    backend_peak_bytes: int | None = None
    sample_count: int = 0
    reason: str = "not sampled"

    @classmethod
    def start(cls, backend: str) -> RuntimeMemoryTelemetry:
        telemetry = cls(backend=backend)
        telemetry.sample()
        telemetry.process_start_rss_bytes = telemetry.process_end_rss_bytes
        return telemetry

    def sample(self) -> None:
        rss = process_rss_bytes()
        if rss is not None:
            self.telemetry_supported = True
            self.process_end_rss_bytes = rss
            if self.process_peak_rss_bytes is None or rss > self.process_peak_rss_bytes:
                self.process_peak_rss_bytes = rss
            self.reason = "process RSS telemetry available"
        elif not self.telemetry_supported:
            self.reason = "process RSS telemetry unavailable"

        active, peak, supported, backend_reason = backend_memory_bytes(self.backend)
        self.backend_memory_supported = supported
        if active is not None:
            self.backend_active_bytes = active
        if peak is not None:
            self.backend_peak_bytes = peak
        if supported:
            self.reason = f"{self.reason}; {backend_reason}"
        self.sample_count += 1

    def finish(self) -> RuntimeMemoryTelemetry:
        self.sample()
        return self

    def to_dict(self) -> dict[str, int | str | bool | None]:
        return asdict(self)
