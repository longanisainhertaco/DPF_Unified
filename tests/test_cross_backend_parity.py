"""Cross-backend I(t) NRMSE parity test.

Per AUDIT_BRIEF.md:72-78 DoD (2026-03-19):
    "Every backend labeled 'MHD' must couple plasma state to circuit via Lp
    feedback. Cross-backend parity test: two MHD backends on the same problem
    must agree on I(t) NRMSE < 0.10. Backends that are Lee-model-equivalent
    must be labeled 'Lee' not 'MHD.'"

This gate test compares two claimed-MHD backends (Python cylindrical
CylindricalMHDSolver and MLX MLXMHDSolver) on the same PF-1000 problem. It
measures the range-normalized RMSE between their I(t) time-series, sampled
on a shared time grid via linear interpolation of the MLX trace onto the
cylindrical trace's sample times.

Acceptance:
    - NRMSE < 0.10  (DoD)     : backends agree as MHD solvers
    - NRMSE > 1e-6 (anti-dup) : backends are genuinely independent (not
                                  silently falling back to the same path)

If the test fails, that is VALUABLE DATA: the two "MHD" backends do not
agree at the DoD level, which means at least one is mislabeled or the
circuit/Lp coupling in one of them is broken. Report the actual NRMSE
rather than hacking the assertion to pass.

NRMSE convention (matches src/dpf/validation/suite.py:228 nrmse_range):
    NRMSE = sqrt(mean((I_a - I_b)**2)) / (max(I_ref) - min(I_ref))
"""

from __future__ import annotations

import numpy as np
import pytest

# MLX is optional — skip the whole module if unavailable.
_mx = pytest.importorskip("mlx.core", reason="MLX backend required for parity test")

# Torch is imported lazily by the Metal path; we do NOT require it here, since
# the parity compares Python-cylindrical vs MLX. Document this so that the
# test does not regress when torch is absent (see CI breakage learning:
# commit 37c0f1d "fix(ci): gate torch import in metal backend").

from dpf.config import SimulationConfig  # noqa: E402
from dpf.engine import SimulationEngine  # noqa: E402
from dpf.metal.mlx_solver import MLXMHDSolver  # noqa: E402
from dpf.presets import get_preset  # noqa: E402

# Canonical nrmse_range from suite.py (copied inline to keep the test
# independent of validation-facade drift).

_MAX_STEPS = 300         # Enough to see circuit ramp; keeps wall-time < 90 s.
_COARSE_GRID = (48, 1, 96)  # 48x1x96 cylindrical — run time ~30-60 s per backend.
_COARSE_SIM_TIME = 3e-6    # 3 us: covers early ramp and a fraction of peak build.


def _nrmse_range(a: np.ndarray, b: np.ndarray, ref: np.ndarray) -> float:
    """Range-normalized RMSE (NRMSE = RMSE / (max(ref)-min(ref))).

    Matches :func:`dpf.validation.suite.nrmse_range` verbatim.
    Returns ``inf`` if either array is empty and ``sqrt(MSE)`` if the
    reference range collapses to zero.
    """
    if len(a) == 0 or len(b) == 0 or len(ref) == 0:
        return float("inf")
    mse = float(np.mean((a - b) ** 2))
    ref_range = float(np.max(ref) - np.min(ref))
    if ref_range <= 0.0:
        return float(np.sqrt(mse))
    return float(np.sqrt(mse) / ref_range)


def _pf1000_preset_trimmed(backend: str) -> dict:
    """Return a PF-1000 preset trimmed for a fast parity run.

    The preset is mutated in place with:
      - ``fluid.backend`` set to the requested backend.
      - Coarse grid and short sim_time for CI-friendly runtime.
      - Radiation / collision disabled so the MHD step is the only
        source of inter-backend difference.
      - Kinetic / PIC / surrogate sub-systems disabled for the same
        reason.
    """
    preset = get_preset("pf1000")

    # Mandatory: pick backend under test.
    preset["fluid"] = {
        "backend": backend,
        "gamma": 5.0 / 3.0,
        "cfl": 0.3,
    }
    if backend == "mlx":
        # MLX-specific schema that is known to work with SimulationEngine
        # (see tests/test_mlx_pf1000.py:73-83).
        preset["fluid"].update({
            "riemann_solver": "hll",
            "reconstruction": "plm",
            "time_integrator": "ssp_rk2",
            "precision": "float32",
            "use_ct": False,
        })

    # Coarse grid / short horizon so the test fits in the slow-marker
    # budget (<90 s combined for both backends).
    preset["grid_shape"] = list(_COARSE_GRID)
    preset["sim_time"] = _COARSE_SIM_TIME

    # Turn off radiation + collision + impurity line radiation so that
    # both backends differ ONLY in their MHD step.
    preset["radiation"] = {
        "bremsstrahlung_enabled": False,
        "line_radiation_enabled": False,
        "fld_enabled": False,
    }
    if "collision" in preset:
        preset["collision"] = {"enabled": False}
    else:
        preset["collision"] = {"enabled": False}

    return preset


def _run_and_capture_current(backend: str, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Run SimulationEngine with the requested backend, capture per-step I(t).

    Returns
    -------
    times_s : ndarray, shape (n,)
        Time at end of each step [s].
    currents_A : ndarray, shape (n,)
        Absolute circuit current at end of each step [A].
    """
    preset = _pf1000_preset_trimmed(backend)
    config = SimulationConfig(**preset)
    engine = SimulationEngine(config)

    times: list[float] = []
    currents: list[float] = []

    for _ in range(max_steps):
        result = engine.step(_max_steps=max_steps)
        times.append(float(engine.time))
        currents.append(float(abs(engine.circuit.current)))
        if result.finished:
            break

    return np.asarray(times, dtype=np.float64), np.asarray(currents, dtype=np.float64)


@pytest.mark.slow
def test_cross_backend_current_nrmse_parity() -> None:
    """Two MHD backends must agree on I(t) NRMSE < 0.10 (AUDIT_BRIEF DoD).

    Backends under test:
      A = 'python' + geometry.type='cylindrical'  ->  CylindricalMHDSolver
      B = 'mlx'                                    ->  MLXMHDSolver

    Both are claimed to be MHD solvers (not Lee-model) and must therefore
    satisfy the DoD NRMSE < 0.10 parity bar. The test intentionally does
    NOT include energy / yield metrics; the DoD gate is I(t) alone.
    """
    if not MLXMHDSolver.is_available():
        pytest.skip("MLX backend unavailable (requires Apple Silicon + mlx)")

    t_a, I_a = _run_and_capture_current("python", max_steps=_MAX_STEPS)
    t_b, I_b = _run_and_capture_current("mlx", max_steps=_MAX_STEPS)

    # Guard against degenerate traces — either side producing zero steps
    # or zero current is an engine bug, not a parity failure.
    assert len(t_a) > 10, f"Python-cylindrical: only {len(t_a)} steps captured"
    assert len(t_b) > 10, f"MLX: only {len(t_b)} steps captured"
    assert np.max(I_a) > 0.0, "Python-cylindrical: zero current (no plasma coupling?)"
    assert np.max(I_b) > 0.0, "MLX: zero current (no plasma coupling?)"

    # Resample B onto A's time grid in the overlap window.
    t_lo = max(float(t_a[0]), float(t_b[0]))
    t_hi = min(float(t_a[-1]), float(t_b[-1]))
    if t_hi <= t_lo:
        pytest.fail(
            f"No time overlap between backends: "
            f"python t=[{t_a[0]:.3e}, {t_a[-1]:.3e}], "
            f"mlx t=[{t_b[0]:.3e}, {t_b[-1]:.3e}]"
        )
    mask_a = (t_a >= t_lo) & (t_a <= t_hi)
    t_common = t_a[mask_a]
    I_a_common = I_a[mask_a]
    I_b_common = np.interp(t_common, t_b, I_b)

    # DoD metric — use Python-cylindrical as the reference for the range
    # normalization (arbitrary but documented choice).
    nrmse = _nrmse_range(I_a_common, I_b_common, ref=I_a_common)

    # --- Anti-trivial check: backends must not be bit-identical -----------
    #
    # If NRMSE == 0 exactly, one backend is silently dispatching to the
    # other (or both are reading the same Lee cache). That is a severe
    # infrastructure bug, not a parity pass.
    assert nrmse > 1e-6, (
        f"Backends produced bit-identical I(t) (NRMSE={nrmse:.3e}). "
        "One backend is likely falling back to the other, or both are "
        "reading the same cached Lee trajectory. Investigate engine dispatch."
    )

    # --- DoD gate ---------------------------------------------------------
    #
    # Report the actual value in the assertion message so a failing run
    # yields immediately-actionable physics data.
    assert nrmse < 0.10, (
        f"Cross-backend I(t) NRMSE = {nrmse:.4f} >= 0.10 (AUDIT_BRIEF DoD). "
        f"Backends: python-cylindrical vs mlx. "
        f"Captured steps: python={len(t_a)}, mlx={len(t_b)}. "
        f"Peak currents: python={np.max(I_a):.3e} A, mlx={np.max(I_b):.3e} A. "
        "At least one of these backends is not actually solving MHD, or its "
        "Lp feedback is broken. Do NOT hack the tolerance -- investigate."
    )
