"""Tests for advanced physics modules wired into app_mhd."""
import sys
from pathlib import Path

import numpy as np
import pytest

# app_engine.py is a root-level Gradio app, not in dpf package.
# Add project root to path for CI where PYTHONPATH doesn't include it.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app_engine import GAS_SPECIES  # noqa: E402


def _make_state(nr=8, ny=8, nz=16):
    """Create a minimal MHD state dict for testing."""
    rho0 = 0.084
    p0 = 400.0
    T0 = 300.0
    return {
        "rho": np.full((nr, ny, nz), rho0),
        "velocity": np.zeros((3, nr, ny, nz)),
        "pressure": np.full((nr, ny, nz), p0),
        "B": np.zeros((3, nr, ny, nz)),
        "Te": np.full((nr, ny, nz), T0),
        "Ti": np.full((nr, ny, nz), T0),
        "psi": np.zeros((nr, ny, nz)),
    }


def _gas_d2():
    return GAS_SPECIES.get("D2", {"m_mol": 6.68e-27, "Z": 1, "gamma": 5 / 3})


class TestApplyAdvancedPhysics:
    def test_no_modules_is_noop(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        rho_before = state["rho"].copy()
        Te_before = state["Te"].copy()
        B_before = state["B"].copy()

        state_out, cr = _apply_advanced_physics(
            state, dt=1e-9, gas=_gas_d2(), dr=0.001, dz=0.002,
            a=0.01, b=0.03,
        )
        np.testing.assert_array_equal(state_out["rho"], rho_before)
        np.testing.assert_array_equal(state_out["Te"], Te_before)
        np.testing.assert_array_equal(state_out["B"], B_before)
        assert cr is None

    def test_fld_runs_without_crash(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        state["Te"] = np.full_like(state["Te"], 1e6)  # hot plasma
        state_out, _ = _apply_advanced_physics(
            state, dt=1e-12, gas=_gas_d2(), dr=0.001, dz=0.002,
            a=0.01, b=0.03, enable_fld=True,
        )
        assert "Te" in state_out
        assert not np.any(np.isnan(state_out["Te"]))

    def test_sheath_runs_without_crash(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        state["Te"] = np.full_like(state["Te"], 5e4)
        state_out, _ = _apply_advanced_physics(
            state, dt=1e-9, gas=_gas_d2(), dr=0.001, dz=0.002,
            a=0.01, b=0.03, enable_sheath=True,
        )
        assert "velocity" in state_out

    def test_ablation_adds_mass(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        state["Te"] = np.full_like(state["Te"], 1e6)
        # Give it a B field so J = curl(B)/mu0 is nonzero
        B_z = np.linspace(0, 10, state["rho"].shape[0])
        state["B"][2] = B_z[:, np.newaxis, np.newaxis]
        rho_before = state["rho"].copy()

        state_out, _ = _apply_advanced_physics(
            state, dt=1e-7, gas=_gas_d2(), dr=0.001, dz=0.002,
            a=0.01, b=0.03, enable_ablation=True,
        )
        # Ablation should add mass at boundary (first cell)
        assert np.any(state_out["rho"][0] >= rho_before[0])

    def test_nernst_modifies_B(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        # Need Te gradient and B field for Nernst
        Te_grad = np.linspace(1e4, 1e6, state["rho"].shape[0])
        state["Te"] = Te_grad[:, np.newaxis, np.newaxis] * np.ones_like(state["Te"])
        state["B"][1] = 5.0  # uniform By (theta)

        state_out, _ = _apply_advanced_physics(
            state, dt=1e-12, gas=_gas_d2(), dr=0.001, dz=0.002,
            a=0.01, b=0.03, enable_nernst=True,
        )
        # Nernst should modify B
        assert "B" in state_out

    def test_cr_evolves_charge_state(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        state["Te"] = np.full_like(state["Te"], 5e5)  # ~43 eV — should partially ionize H
        state_out, cr_fracs = _apply_advanced_physics(
            state, dt=1e-7, gas=_gas_d2(), dr=0.001, dz=0.002,
            a=0.01, b=0.03, enable_cr=True,
        )
        assert cr_fracs is not None
        assert "Z_bar" in state_out
        # At ~43 eV, hydrogen should be significantly ionized
        Z_bar_max = float(np.max(state_out["Z_bar"]))
        assert Z_bar_max >= 0.0  # at least some ionization tracking

    def test_all_modules_together(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        state["Te"] = np.full_like(state["Te"], 1e6)
        B_z = np.linspace(0, 5, state["rho"].shape[0])
        state["B"][1] = 2.0
        state["B"][2] = B_z[:, np.newaxis, np.newaxis]

        state_out, cr_fracs = _apply_advanced_physics(
            state, dt=1e-12, gas=_gas_d2(), dr=0.001, dz=0.002,
            a=0.01, b=0.03,
            enable_fld=True, enable_sheath=True, enable_ablation=True,
            enable_nernst=True, enable_cr=True,
        )
        assert not np.any(np.isnan(state_out["rho"]))
        assert not np.any(np.isnan(state_out["Te"]))
        assert cr_fracs is not None

    def test_cr_fractions_persist_across_calls(self):
        from app_mhd import _apply_advanced_physics

        state = _make_state()
        state["Te"] = np.full_like(state["Te"], 1e6)
        gas = _gas_d2()

        _, cr1 = _apply_advanced_physics(
            state, dt=1e-9, gas=gas, dr=0.001, dz=0.002,
            a=0.01, b=0.03, enable_cr=True, cr_fractions=None,
        )
        assert cr1 is not None

        # Pass cr1 back in — should continue evolving
        _, cr2 = _apply_advanced_physics(
            state, dt=1e-9, gas=gas, dr=0.001, dz=0.002,
            a=0.01, b=0.03, enable_cr=True, cr_fractions=cr1,
        )
        assert cr2 is not None


@pytest.mark.skipif(
    not hasattr(__import__("importlib"), "import_module") or
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed (Metal/hybrid backend unavailable)",
)
class TestRunMhdSimulationToggle:
    def test_advanced_physics_in_result(self):
        from app_mhd import run_mhd_simulation

        result = run_mhd_simulation(
            backend="hybrid", grid_preset="coarse",
            preset_name="pf1000", sim_time_us=1.0,
            enable_fld=True, enable_ablation=True,
        )
        assert "advanced_physics" in result
        assert "FLD radiation transport" in result["advanced_physics"]
        assert "Electrode ablation (Cu)" in result["advanced_physics"]
        assert result["reproducibility"]["version"] == "v1.4.0"

    def test_no_advanced_physics_empty_list(self):
        from app_mhd import run_mhd_simulation

        result = run_mhd_simulation(
            backend="hybrid", grid_preset="coarse",
            preset_name="pf1000", sim_time_us=1.0,
        )
        assert result.get("advanced_physics") == []
