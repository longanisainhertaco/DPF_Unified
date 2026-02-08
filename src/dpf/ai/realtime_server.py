"""FastAPI router for AI/ML inference endpoints.

Provides REST and WebSocket endpoints for WALRUS surrogate prediction,
parameter sweeps, inverse design, and confidence estimation.
Include in main app via: app.include_router(ai_router).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from dpf.ai import HAS_TORCH

logger = logging.getLogger(__name__)

ai_router = APIRouter(prefix="/api/ai", tags=["ai"])

# Module-level state — populated by load_surrogate() or serve-ai CLI
_surrogate: Any | None = None
_ensemble: Any | None = None


def load_surrogate(checkpoint_path: str, device: str = "cpu") -> None:
    """Load a WALRUS surrogate model."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available — cannot load surrogate")
    global _surrogate
    from dpf.ai.surrogate import DPFSurrogate

    _surrogate = DPFSurrogate(checkpoint_path, device=device)
    logger.info("Loaded surrogate from %s on %s", checkpoint_path, device)


def load_ensemble(checkpoint_paths: list[str], device: str = "cpu") -> None:
    """Load ensemble of WALRUS models."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available — cannot load ensemble")
    global _ensemble
    from dpf.ai.confidence import EnsemblePredictor

    _ensemble = EnsemblePredictor(checkpoint_paths, device=device)
    logger.info("Loaded ensemble of %d models", _ensemble.n_models)


def _require_surrogate() -> Any:
    if _surrogate is None:
        raise HTTPException(status_code=503, detail="No surrogate model loaded")
    return _surrogate


def _require_ensemble() -> Any:
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="No ensemble model loaded")
    return _ensemble


def _arrays_to_lists(state: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert NumPy arrays to nested lists for JSON serialization."""
    result = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = _arrays_to_lists(value)
        elif isinstance(value, list):
            result[key] = [
                _arrays_to_lists(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


def _lists_to_arrays(state: dict[str, Any]) -> dict[str, Any]:
    """Convert nested lists back to NumPy arrays."""
    result = {}
    for key, value in state.items():
        if isinstance(value, list):
            # Try to convert to array if it's a numeric list
            try:
                result[key] = np.array(value)
            except (ValueError, TypeError):
                # Not a numeric array — keep as list or recurse if dict list
                if value and isinstance(value[0], dict):
                    result[key] = [_lists_to_arrays(item) for item in value]
                else:
                    result[key] = value
        elif isinstance(value, dict):
            result[key] = _lists_to_arrays(value)
        else:
            result[key] = value
    return result


# ── Status ───────────────────────────────────────────────────


@ai_router.get("/status")
async def ai_status() -> dict[str, Any]:
    """Check AI subsystem status and model availability."""
    device = "none"
    if _surrogate is not None:
        device = getattr(_surrogate, "device", "unknown")

    ensemble_size = 0
    if _ensemble is not None:
        ensemble_size = getattr(_ensemble, "n_models", 0)

    return {
        "torch_available": HAS_TORCH,
        "model_loaded": _surrogate is not None,
        "device": device,
        "ensemble_size": ensemble_size,
    }


# ── Surrogate Prediction ─────────────────────────────────────


@ai_router.post("/predict")
async def ai_predict(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Predict next simulation step from history.

    Args:
        history: List of state dicts with keys {rho, velocity, pressure, B, Te, Ti, psi}

    Returns:
        {
            "predicted_state": {...},  # Next state dict with arrays as nested lists
            "inference_time_ms": float
        }
    """
    surrogate = _require_surrogate()

    # Convert to arrays
    history_arrays = [_lists_to_arrays(state) for state in history]

    # Run prediction in thread pool to avoid blocking event loop
    start = time.perf_counter()
    predicted = await asyncio.to_thread(surrogate.predict_next_step, history_arrays)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Convert back to lists for JSON
    predicted_lists = _arrays_to_lists(predicted)

    return {
        "predicted_state": predicted_lists,
        "inference_time_ms": elapsed_ms,
    }


@ai_router.post("/rollout")
async def ai_rollout(history: list[dict[str, Any]], n_steps: int = 10) -> dict[str, Any]:
    """Autoregressive rollout for multiple steps.

    Args:
        history: Initial history states
        n_steps: Number of steps to predict forward

    Returns:
        {
            "trajectory": [...],  # List of n_steps predicted states
            "n_steps": int,
            "total_inference_time_ms": float
        }
    """
    surrogate = _require_surrogate()

    if n_steps <= 0:
        raise HTTPException(status_code=422, detail="n_steps must be positive")
    if n_steps > 1000:
        raise HTTPException(
            status_code=422, detail="n_steps too large (max 1000) — use /sweep for parameter exploration"
        )

    history_arrays = [_lists_to_arrays(state) for state in history]

    start = time.perf_counter()
    trajectory = await asyncio.to_thread(surrogate.rollout, history_arrays, n_steps)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    trajectory_lists = [_arrays_to_lists(state) for state in trajectory]

    return {
        "trajectory": trajectory_lists,
        "n_steps": len(trajectory_lists),
        "total_inference_time_ms": elapsed_ms,
    }


# ── Parameter Sweep ──────────────────────────────────────────


@ai_router.post("/sweep")
async def ai_sweep(configs: list[dict[str, Any]], n_steps: int = 100) -> dict[str, Any]:
    """Run surrogate-accelerated parameter sweep.

    Args:
        configs: List of config dicts (with keys like V0, C, rho0, ...)
        n_steps: Number of steps to simulate per config

    Returns:
        {
            "results": [...],  # List of sweep results per config
            "n_configs": int
        }
    """
    surrogate = _require_surrogate()

    if not configs:
        raise HTTPException(status_code=422, detail="configs list cannot be empty")
    if n_steps <= 0:
        raise HTTPException(status_code=422, detail="n_steps must be positive")

    # Lazy import
    from dpf.ai.parameter_sweep import parameter_sweep

    results = await asyncio.to_thread(parameter_sweep, surrogate, configs, n_steps)

    # Convert results to JSON-serializable format
    results_lists = [_arrays_to_lists(r) for r in results]

    return {
        "results": results_lists,
        "n_configs": len(results_lists),
    }


# ── Inverse Design ───────────────────────────────────────────


@ai_router.post("/inverse")
async def ai_inverse(
    targets: dict[str, float],
    constraints: dict[str, float] | None = None,
    method: str = "bayesian",
    n_trials: int = 100,
) -> dict[str, Any]:
    """Inverse design to find config that achieves target observables.

    Args:
        targets: Dict of target values, e.g. {"max_Te": 5e3, "max_rho": 1e20}
        constraints: Optional dict of parameter bounds
        method: Optimization method ("bayesian", "gradient", "genetic")
        n_trials: Number of optimization trials

    Returns:
        {
            "best_config": {...},  # Config dict that best matches targets
            "predicted_outcomes": {...},  # Predicted observables for best config
            "loss": float,  # Optimization loss (lower is better)
            "n_trials": int
        }
    """
    surrogate = _require_surrogate()

    if not targets:
        raise HTTPException(status_code=422, detail="targets dict cannot be empty")
    if n_trials <= 0 or n_trials > 10000:
        raise HTTPException(status_code=422, detail="n_trials must be in [1, 10000]")

    # Lazy import
    from dpf.ai.inverse_design import InverseDesigner

    # Default parameter ranges (voltage, capacitance, initial density)
    default_ranges = {
        "V0": (1e3, 50e3),  # 1-50 kV
        "C": (1e-6, 100e-6),  # 1-100 μF
        "rho0": (1e17, 1e20),  # Initial density
    }
    if constraints:
        default_ranges.update(constraints)

    designer = InverseDesigner(surrogate, parameter_ranges=default_ranges)

    result = await asyncio.to_thread(designer.optimize, targets, method=method, n_trials=n_trials)

    return _arrays_to_lists(result)


# ── Ensemble Confidence ──────────────────────────────────────


@ai_router.post("/confidence")
async def ai_confidence(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Predict with uncertainty estimation using ensemble.

    Args:
        history: List of state dicts

    Returns:
        {
            "predicted_state": {...},  # Ensemble mean prediction
            "confidence": {...},  # Per-variable standard deviations
            "ood_score": float,  # Out-of-distribution score (higher = less confident)
            "inference_time_ms": float
        }
    """
    ensemble = _require_ensemble()

    history_arrays = [_lists_to_arrays(state) for state in history]

    start = time.perf_counter()
    result = await asyncio.to_thread(ensemble.predict, history_arrays)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # result is dict with keys: mean, std, ood_score
    result_lists = _arrays_to_lists(result)
    result_lists["inference_time_ms"] = elapsed_ms

    # Rename for clarity
    return {
        "predicted_state": result_lists.get("mean", {}),
        "confidence": result_lists.get("std", {}),
        "ood_score": result_lists.get("ood_score", 0.0),
        "inference_time_ms": elapsed_ms,
    }


# ── WebSocket Streaming ──────────────────────────────────────


@ai_router.websocket("/ws/stream")
async def ai_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming AI predictions.

    Full implementation deferred to Unity integration phase.
    Currently accepts connections and sends status messages.
    """
    await websocket.accept()
    logger.info("AI WebSocket client connected")

    try:
        # Send initial status
        status = await ai_status()
        await websocket.send_json({"type": "status", "data": status})

        # Keep connection alive and listen for commands
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)

                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("type") == "status":
                    status = await ai_status()
                    await websocket.send_json({"type": "status", "data": status})
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown message type: {msg.get('type')}"}
                    )
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})

    except WebSocketDisconnect:
        logger.info("AI WebSocket client disconnected")
    except Exception:
        logger.exception("AI WebSocket error")
