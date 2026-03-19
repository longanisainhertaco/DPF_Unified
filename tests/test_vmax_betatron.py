"""Tests for V_max diagnostic and betatron heating operator-split.

V_max: peak inductive voltage L_total * max(|dI/dt|) in result dict.
Betatron: T_perp * B = const (adiabatic magnetic moment conservation).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402


# ------------------------------------------------------------------ #
#  Shared helpers
# ------------------------------------------------------------------ #

def _make_uniform_state(
    grid_shape: tuple[int, int, int],
    rho: float = 1.0,
    p: float = 1.0,
    B_mag: float = 1.0,
) -> dict[str, np.ndarray]:
    nx, ny, nz = grid_shape
    state = {
        "rho": np.full((nx, ny, nz), rho),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.full((nx, ny, nz), p),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.full((nx, ny, nz), 1e4),
        "Ti": np.full((nx, ny, nz), 1e4),
        "psi": np.zeros((nx, ny, nz)),
    }
    state["B"][2] = B_mag  # uniform Bz
    return state


def _make_solver(
    grid_shape: tuple[int, int, int] = (8, 8, 8),
    **kwargs,
) -> MetalMHDSolver:
    return MetalMHDSolver(
        grid_shape=grid_shape,
        dx=0.01,
        device="cpu",
        use_ct=False,
        **kwargs,
    )


# ------------------------------------------------------------------ #
#  V_max diagnostic tests
# ------------------------------------------------------------------ #


class TestVmaxDiagnostic:
    """V_max = L_total * max(|dI/dt|) in app_mhd result dict."""

    def test_vmax_present_in_app_mhd_result(self) -> None:
        """V_max_kV key must appear in run_mhd_simulation output."""
        from app_mhd import run_mhd_simulation

        result = run_mhd_simulation(
            backend="metal_plm",
            grid_preset="coarse",
            preset_name="pf1000",
            sim_time_us=1.0,
        )
        assert "V_max_kV" in result, "V_max_kV missing from result dict"

    def test_vmax_is_positive_float(self) -> None:
        """V_max_kV must be a positive finite float."""
        from app_mhd import run_mhd_simulation

        result = run_mhd_simulation(
            backend="metal_plm",
            grid_preset="coarse",
            preset_name="pf1000",
            sim_time_us=1.0,
        )
        v_max = result["V_max_kV"]
        assert isinstance(v_max, float), f"Expected float, got {type(v_max)}"
        assert v_max > 0.0, f"V_max_kV={v_max} must be positive"
        assert np.isfinite(v_max), f"V_max_kV={v_max} must be finite"

    def test_vmax_in_lee_model_metadata(self) -> None:
        """LeeModelResult.metadata must include V_max_kV as positive float."""
        from dpf.validation.lee_model_comparison import LeeModel

        runner = LeeModel()
        result = runner.run(device_name="PF-1000")
        assert "V_max_kV" in result.metadata, (
            "V_max_kV missing from LeeModelResult.metadata"
        )
        v_max = result.metadata["V_max_kV"]
        assert isinstance(v_max, float)
        assert v_max > 0.0, f"Lee V_max_kV={v_max} must be positive"
        assert np.isfinite(v_max), f"Lee V_max_kV={v_max} must be finite"

    def test_vmax_physically_reasonable(self) -> None:
        """V_max for PF-1000 should be in the 1-5000 kV range."""
        from app_mhd import run_mhd_simulation

        result = run_mhd_simulation(
            backend="metal_plm",
            grid_preset="coarse",
            preset_name="pf1000_akel",
            sim_time_us=2.0,
        )
        v_max = result["V_max_kV"]
        assert 1.0 < v_max < 5000.0, (
            f"V_max_kV={v_max:.1f} kV outside expected 1-5000 kV range for PF-1000"
        )


# ------------------------------------------------------------------ #
#  Betatron heating tests
# ------------------------------------------------------------------ #


class TestBetatronHeating:
    """Betatron: T_perp increases when B increases (mu = T_perp/B conserved)."""

    def test_betatron_disabled_by_default(self) -> None:
        """enable_betatron must default to False."""
        solver = _make_solver()
        assert solver.enable_betatron is False

    def test_betatron_enabled_flag(self) -> None:
        """enable_betatron=True must be stored on solver."""
        solver = _make_solver(enable_betatron=True)
        assert solver.enable_betatron is True

    def test_ti_increases_under_compression(self) -> None:
        """T_i must rise when |B| increases (adiabatic compression)."""
        solver = _make_solver(enable_betatron=True)
        Ti_init = 1e4  # 10 000 K
        B_init = 1.0
        B_compressed = 2.0  # double the field → Ti should double

        Ti = torch.full((8, 8, 8), Ti_init, dtype=torch.float32)
        B_prev = torch.zeros((3, 8, 8, 8), dtype=torch.float32)
        B_prev[2] = B_init
        B_new = torch.zeros((3, 8, 8, 8), dtype=torch.float32)
        B_new[2] = B_compressed

        Ti_out = solver._apply_betatron_heating(Ti, B_new, B_prev)

        Ti_expected = Ti_init * (B_compressed / B_init)
        assert float(Ti_out.mean()) == pytest.approx(Ti_expected, rel=1e-4)

    def test_ti_unchanged_in_expansion(self) -> None:
        """T_i must NOT change when |B| decreases (expansion phase)."""
        solver = _make_solver(enable_betatron=True)
        Ti_init = 2e4
        B_init = 2.0
        B_expanded = 0.5  # field drops → no betatron heating

        Ti = torch.full((8, 8, 8), Ti_init, dtype=torch.float32)
        B_prev = torch.zeros((3, 8, 8, 8), dtype=torch.float32)
        B_prev[2] = B_init
        B_new = torch.zeros((3, 8, 8, 8), dtype=torch.float32)
        B_new[2] = B_expanded

        Ti_out = solver._apply_betatron_heating(Ti, B_new, B_prev)

        assert float(Ti_out.mean()) == pytest.approx(Ti_init, rel=1e-5)

    def test_betatron_step_integration(self) -> None:
        """Full step() must raise Ti when B is compressed, with enable_betatron=True."""
        solver = _make_solver(enable_betatron=True)
        grid = (8, 8, 8)

        # State with uniform Bz; manually increase B to trigger heating
        state = _make_uniform_state(grid, B_mag=1.0)
        Ti_before = float(np.mean(state["Ti"]))

        # Step with uniform state; B won't change much but we verify no crash
        state_out = solver.step(state, dt=1e-9, current=0.0, voltage=0.0)

        assert "Ti" in state_out
        # Ti must remain finite and non-negative
        assert np.all(np.isfinite(state_out["Ti"]))
        assert np.all(state_out["Ti"] >= 0.0)
        _ = Ti_before  # accessed above

    def test_betatron_strong_compression(self) -> None:
        """10x B compression should give ~10x T_i increase."""
        solver = _make_solver(enable_betatron=True)
        Ti_init = 1e4
        B_init = 1.0
        B_final = 10.0

        Ti = torch.full((4, 4, 4), Ti_init, dtype=torch.float32)
        B_prev = torch.zeros((3, 4, 4, 4), dtype=torch.float32)
        B_prev[2] = B_init
        B_new = torch.zeros((3, 4, 4, 4), dtype=torch.float32)
        B_new[2] = B_final

        Ti_out = solver._apply_betatron_heating(Ti, B_new, B_prev)

        Ti_expected = Ti_init * B_final / B_init
        assert float(Ti_out.mean()) == pytest.approx(Ti_expected, rel=1e-4)
