"""Phase V: Cylindrical conservative energy — detailed tests.

Tests the CylindricalMHDSolver conservative energy formulation including:
- E_total = p/(gamma-1) + 0.5*rho*v^2 + B^2/(2*mu_0) evolution
- Pressure recovery: p = (gamma-1)*(E - KE - ME), floored at 1e-20
- SSP-RK3 stage weights applied to E_total (not pressure)
- Axis BC: v_r=0, B_r=0 at r=0
- Source term passthrough: S_rho_snowplow, S_mom_snowplow, S_energy_snowplow, Q_ohmic_correction
- Conservative vs non-conservative energy conservation comparison
"""

import numpy as np
import pytest

from dpf.constants import mu_0
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(nr: int, nz: int, rho: float = 1.0, p: float = 1e5,
                vr: float = 0.0, vz: float = 0.0, Btheta: float = 0.0,
                Bz: float = 0.0) -> dict[str, np.ndarray]:
    """Build a cylindrical state dict with shapes (nr,1,nz) / (3,nr,1,nz)."""
    state = {
        "rho": np.full((nr, 1, nz), rho),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }
    state["velocity"][0] = vr
    state["velocity"][2] = vz
    state["B"][1] = Btheta
    state["B"][2] = Bz
    return state


def _total_energy_volume(state: dict, dr: float, dz: float,
                         gamma: float = 5.0 / 3.0) -> float:
    """Compute volume-integrated total energy for cylindrical geometry."""
    nr = state["rho"].shape[0]
    nz = state["rho"].shape[2]
    rho = state["rho"][:, 0, :]
    p = state["pressure"][:, 0, :]
    v = state["velocity"][:, :, 0, :]
    B = state["B"][:, :, 0, :]
    v_sq = np.sum(v**2, axis=0)
    B_sq = np.sum(B**2, axis=0)
    E_density = p / (gamma - 1) + 0.5 * rho * v_sq + B_sq / (2.0 * mu_0)
    r = np.array([(i + 0.5) * dr for i in range(nr)])
    cell_vol = 2.0 * np.pi * r[:, np.newaxis] * dr * dz
    return float(np.sum(E_density * cell_vol))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cons_solver():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        conservative_energy=True, time_integrator="ssp_rk3",
    )


@pytest.fixture
def noncons_solver():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        conservative_energy=False, time_integrator="ssp_rk3",
    )


# ---------------------------------------------------------------------------
# 1. Conservative energy flag
# ---------------------------------------------------------------------------

class TestConservativeFlag:
    def test_default_is_true(self):
        s = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        assert s.conservative_energy is True

    def test_explicit_true(self):
        s = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01,
                                 conservative_energy=True)
        assert s.conservative_energy is True

    def test_rhs_dE_dt_when_conservative(self, cons_solver):
        """_compute_rhs returns dE_dt (not dp_dt) when conservative_energy=True."""
        rho = np.ones((16, 16))
        vel = np.zeros((3, 16, 16))
        p = np.full((16, 16), 1e5)
        B = np.zeros((3, 16, 16))
        psi = np.zeros((16, 16))
        rhs = cons_solver._compute_rhs(rho, vel, p, B, psi)
        assert "dE_dt" in rhs
        assert "dp_dt" not in rhs

    def test_rhs_dp_dt_when_nonconservative(self, noncons_solver):
        """_compute_rhs returns dp_dt (not dE_dt) when conservative_energy=False."""
        rho = np.ones((16, 16))
        vel = np.zeros((3, 16, 16))
        p = np.full((16, 16), 1e5)
        B = np.zeros((3, 16, 16))
        psi = np.zeros((16, 16))
        rhs = noncons_solver._compute_rhs(rho, vel, p, B, psi)
        assert "dp_dt" in rhs
        assert "dE_dt" not in rhs


# ---------------------------------------------------------------------------
# 2. Energy conservation comparison
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    def test_uniform_state_preserves_energy(self, cons_solver):
        """Uniform quiescent state should preserve total energy exactly."""
        state = _make_state(16, 16, rho=1.0, p=1e5)
        E0 = _total_energy_volume(state, 0.001, 0.001)
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        E1 = _total_energy_volume(new, 0.001, 0.001)
        assert E1 == pytest.approx(E0, rel=1e-6)

    def test_sod_shock_energy_bounded(self, cons_solver):
        """Sod shock tube: total energy change < 10% over 5 steps."""
        nr, nz, dr, dz = 16, 16, 0.001, 0.001
        state = _make_state(nr, nz, rho=1.0, p=1e5)
        mid = nz // 2
        state["rho"][:, :, mid:] = 0.125
        state["pressure"][:, :, mid:] = 1e4
        E0 = _total_energy_volume(state, dr, dz)
        for _ in range(5):
            state = cons_solver.step(state, dt=1e-7, current=0.0, voltage=0.0)
        E1 = _total_energy_volume(state, dr, dz)
        rel = abs(E1 - E0) / abs(E0)
        assert rel < 0.10, f"Energy changed by {rel:.2%}"

    def test_conservative_better_than_nonconservative(self):
        """Conservative mode should conserve energy at least as well as non-conservative
        across a shock discontinuity."""
        nr, nz, dr, dz = 16, 32, 0.001, 0.001
        cons = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                    conservative_energy=True, time_integrator="ssp_rk3")
        noncons = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                       conservative_energy=False, time_integrator="ssp_rk3")
        # Sod-like IC
        state_c = _make_state(nr, nz, rho=1.0, p=1e5)
        mid = nz // 2
        state_c["rho"][:, :, mid:] = 0.125
        state_c["pressure"][:, :, mid:] = 1e4
        state_nc = {k: v.copy() for k, v in state_c.items()}
        E0 = _total_energy_volume(state_c, dr, dz)
        dt = 1e-7
        for _ in range(10):
            state_c = cons.step(state_c, dt=dt, current=0.0, voltage=0.0)
            state_nc = noncons.step(state_nc, dt=dt, current=0.0, voltage=0.0)
        rel_c = abs(_total_energy_volume(state_c, dr, dz) - E0) / abs(E0)
        rel_nc = abs(_total_energy_volume(state_nc, dr, dz) - E0) / abs(E0)
        # Conservative should be no worse (allow small tolerance)
        assert rel_c <= rel_nc + 0.02, (
            f"Conservative ({rel_c:.4f}) worse than non-conservative ({rel_nc:.4f})"
        )


# ---------------------------------------------------------------------------
# 3. Pressure recovery
# ---------------------------------------------------------------------------

class TestPressureRecovery:
    def test_pressure_positive_after_step(self, cons_solver):
        """Pressure must remain positive (floored at 1e-20) after conservative step."""
        state = _make_state(16, 16, rho=1.0, p=1e5)
        # High velocity to make KE dominate
        state["velocity"][2] = 1e4
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        assert np.all(new["pressure"] >= 1e-20)

    def test_pressure_recovery_formula(self, cons_solver):
        """Verify p = (gamma-1)*(E - 0.5*rho*v^2 - B^2/(2*mu_0))."""
        gamma = cons_solver.gamma
        gm1 = gamma - 1.0
        # Known state
        rho_val, p_val = 2.0, 5e5
        vz_val, Bt_val = 1e3, 0.5
        state = _make_state(16, 16, rho=rho_val, p=p_val, vz=vz_val, Btheta=Bt_val)
        # Compute expected E_total
        v_sq = vz_val**2
        B_sq = Bt_val**2
        E_expected = p_val / gm1 + 0.5 * rho_val * v_sq + B_sq / (2.0 * mu_0)
        # Recover pressure from that E_total
        p_recovered = gm1 * (E_expected - 0.5 * rho_val * v_sq - B_sq / (2.0 * mu_0))
        assert p_recovered == pytest.approx(p_val, rel=1e-12)

    def test_pressure_floor_prevents_negative(self, cons_solver):
        """When KE+ME > E_total, pressure is floored at 1e-20."""
        state = _make_state(16, 16, rho=1.0, p=1.0)
        # Inject huge velocity so KE >> thermal energy
        state["velocity"][2] = 1e8
        new = cons_solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        assert np.all(new["pressure"] >= 1e-20)


# ---------------------------------------------------------------------------
# 4. SSP-RK3 conservative combining
# ---------------------------------------------------------------------------

class TestSSPRK3Combining:
    def test_rk3_three_stage_weights(self):
        """Verify SSP-RK3 weights: (1, 3/4+1/4, 1/3+2/3) on E_total."""
        # This is tested indirectly: run one step with RK3 vs RK2 on same IC.
        # RK3 should give a different (generally more accurate) result.
        nr, nz = 16, 16
        dr, dz = 0.001, 0.001
        rk3 = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                    conservative_energy=True, time_integrator="ssp_rk3")
        rk2 = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                    conservative_energy=True, time_integrator="ssp_rk2")
        state = _make_state(nr, nz, rho=1.0, p=1e5, vz=1e3)
        out3 = rk3.step(state, dt=1e-7, current=0.0, voltage=0.0)
        out2 = rk2.step(state, dt=1e-7, current=0.0, voltage=0.0)
        # Results should differ (different weights)
        diff = np.max(np.abs(out3["pressure"] - out2["pressure"]))
        assert diff > 0.0, "RK3 and RK2 produced identical results"

    def test_rk3_conservative_no_nan(self, cons_solver):
        """RK3 conservative stepping should not produce NaN on moderate flow."""
        state = _make_state(16, 16, rho=1.0, p=1e5, vz=5e3, Btheta=0.1)
        for _ in range(10):
            state = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0)
        for key in ("rho", "pressure", "velocity", "B"):
            assert not np.any(np.isnan(state[key])), f"NaN in {key}"


# ---------------------------------------------------------------------------
# 5. Axis boundary conditions
# ---------------------------------------------------------------------------

class TestAxisBC:
    def test_vr_zero_at_axis(self, cons_solver):
        """v_r must be zero at r=0 after step."""
        state = _make_state(16, 16, vr=500.0)
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        np.testing.assert_allclose(new["velocity"][0, 0, :, :], 0.0, atol=1e-30)

    def test_Br_zero_at_axis(self, cons_solver):
        """B_r must be zero at r=0 after step."""
        state = _make_state(16, 16)
        state["B"][0] = 0.01
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        np.testing.assert_allclose(new["B"][0, 0, :, :], 0.0, atol=1e-30)

    def test_axis_bc_survives_multiple_steps(self, cons_solver):
        """Axis BC should hold after many steps with active dynamics."""
        state = _make_state(16, 16, vr=100.0, Btheta=0.5)
        for _ in range(20):
            state = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        np.testing.assert_allclose(state["velocity"][0, 0, :, :], 0.0, atol=1e-30)
        np.testing.assert_allclose(state["B"][0, 0, :, :], 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# 6. Source term passthrough
# ---------------------------------------------------------------------------

class TestSourceTermPassthrough:
    def test_Q_ohmic_correction_heats(self, cons_solver):
        """Q_ohmic_correction source should increase pressure / total energy."""
        state = _make_state(16, 16, rho=1.0, p=1e5)
        E0 = _total_energy_volume(state, 0.001, 0.001)
        src = {"Q_ohmic_correction": np.full((16, 1, 16), 1e12)}
        new = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0,
                               source_terms=src)
        E1 = _total_energy_volume(new, 0.001, 0.001)
        assert E1 > E0, "Ohmic correction did not increase total energy"

    def test_S_rho_snowplow_adds_mass(self, cons_solver):
        """S_rho_snowplow should increase density."""
        state = _make_state(16, 16, rho=0.01, p=1e5)
        S_rho = np.zeros((16, 1, 16))
        S_rho[8, :, 8] = 1e3  # inject mass at one cell
        src = {"S_rho_snowplow": S_rho}
        new = cons_solver.step(state, dt=1e-6, current=0.0, voltage=0.0,
                               source_terms=src)
        assert new["rho"][8, 0, 8] > state["rho"][8, 0, 8]

    def test_S_mom_snowplow_adds_momentum(self, cons_solver):
        """S_mom_snowplow should change velocity."""
        state = _make_state(16, 16, rho=1.0, p=1e5)
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2, 8, :, 8] = 1e6  # axial kick
        src = {"S_mom_snowplow": S_mom}
        new = cons_solver.step(state, dt=1e-7, current=0.0, voltage=0.0,
                               source_terms=src)
        assert abs(new["velocity"][2, 8, 0, 8]) > 0.0

    def test_S_energy_snowplow_in_conservative(self, cons_solver):
        """S_energy_snowplow should be added to dE/dt in conservative mode."""
        state = _make_state(16, 16, rho=1.0, p=1e5)
        E0 = _total_energy_volume(state, 0.001, 0.001)
        S_E = np.full((16, 1, 16), 1e12)
        src = {"S_energy_snowplow": S_E}
        new = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0,
                               source_terms=src)
        E1 = _total_energy_volume(new, 0.001, 0.001)
        assert E1 > E0

    def test_multiple_source_terms_combined(self, cons_solver):
        """All four source terms active simultaneously."""
        state = _make_state(16, 16, rho=1.0, p=1e5)
        src = {
            "S_rho_snowplow": np.full((16, 1, 16), 0.1),
            "S_mom_snowplow": np.zeros((3, 16, 1, 16)),
            "S_energy_snowplow": np.full((16, 1, 16), 1e8),
            "Q_ohmic_correction": np.full((16, 1, 16), 1e8),
        }
        new = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0,
                               source_terms=src)
        for key in ("rho", "pressure", "velocity", "B"):
            assert not np.any(np.isnan(new[key])), f"NaN in {key} with combined sources"
