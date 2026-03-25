"""Backend resolution and solver factory for DPF simulation engine.

Resolves backend names to canonical forms and checks availability.
Separated from core.py to isolate backend-specific lazy imports.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_backend(requested: str) -> str:
    """Resolve the requested backend to an actual backend name.

    Args:
        requested: ``"python"``, ``"athena"``, ``"athenak"``,
            ``"metal"``, ``"mlx"``, ``"hybrid"``, or ``"auto"``.

    Returns:
        Canonical backend name string.

    Raises:
        RuntimeError: If an explicit backend was requested but is
            not available.
        ValueError: If the backend name is unrecognized.
    """
    if requested == "python":
        return "python"

    if requested == "hybrid":
        return "hybrid"

    if requested == "metal":
        from dpf.metal.metal_solver import MetalMHDSolver
        if not MetalMHDSolver.is_available():
            raise RuntimeError(
                "Metal GPU backend requested but PyTorch MPS is not available. "
                "Requires Apple Silicon with PyTorch >= 2.0.\n"
                "Or use backend='python' or backend='auto'."
            )
        return "metal"

    if requested == "mlx":
        from dpf.metal.mlx_solver import MLXMHDSolver
        if not MLXMHDSolver.is_available():
            raise RuntimeError(
                "MLX Metal backend requested but MLX is not available or "
                "the default device is not a Metal GPU.\n"
                "Install MLX:  pip install mlx\n"
                "Requires macOS 13.3+ on Apple Silicon.\n"
                "Or use backend='python' or backend='auto'."
            )
        return "mlx"

    if requested == "athenak":
        from dpf.athenak_wrapper import is_available as athenak_available
        if not athenak_available():
            raise RuntimeError(
                "AthenaK backend requested but binary not found. Build with:\n"
                "  bash scripts/setup_athenak.sh\n"
                "  bash scripts/build_athenak.sh\n"
                "Or use backend='python' or backend='auto'."
            )
        return "athenak"

    if requested == "athena":
        from dpf.athena_wrapper import is_available
        if not is_available():
            raise RuntimeError(
                "Athena++ backend requested but _athena_core extension "
                "is not compiled.  Build with:\n"
                "  cd src/dpf/athena_wrapper/cpp && mkdir -p build && cd build\n"
                "  cmake .. -DATHENA_ROOT=../../external/athena && make -j8\n"
                "Or use backend='python' or backend='auto'."
            )
        return "athena"

    if requested == "auto":
        try:
            from dpf.athena_wrapper import is_available
            if is_available():
                logger.info("Auto-selected Athena++ backend (primary)")
                return "athena"
        except ImportError:
            pass
        try:
            from dpf.metal.metal_solver import MetalMHDSolver
            if MetalMHDSolver.is_available():
                logger.info("Auto-selected Metal backend (GPU)")
                return "metal"
        except ImportError:
            pass
        try:
            from dpf.athenak_wrapper import is_available as athenak_available
            if athenak_available():
                logger.info("Auto-selected AthenaK backend")
                return "athenak"
        except ImportError:
            pass
        logger.info("Auto-selected Python backend (no C++ backends available)")
        return "python"

    raise ValueError(f"Unknown backend: {requested!r}")


def engine_tier(backend: str) -> str:
    """Return the engine tier based on backend.

    Returns:
        ``"production"`` for conservative backends (Athena++, Metal, MLX),
        ``"teaching"`` for the Python backend (non-conservative dp/dt).
    """
    if backend in ("athena", "metal", "mlx", "hybrid"):
        return "production"
    return "teaching"
