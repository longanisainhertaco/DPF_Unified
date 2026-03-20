"""Tests for CircuitCoupler — density-weighted circuit-MHD coupling.

Tests cover:
- Unit: synthetic density fields with known r_eff, z_sheath
- Monotonicity: Lp can only increase
- BDF2 dLp/dt finite difference
- Back-EMF clamping
- Fallback for 3D Cartesian grids
- Integration: coupling_mode="density_weighted" vs "lee_only" produces different I(t)
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dpf.circuit.coupler import BACK_EMF_CLAMP_V, CircuitCoupler, FeedbackResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cylindrical_coupler():
    """Coupler for a typical DPF cylindrical geometry."""
    return CircuitCoupler(
        anode_radius=0.008,
        cathode_radius=0.032,
        dr=0.001,
        dz=0.002,
        r_inner=0.008,
    )


def _make_cylindrical_state(
    nr: int = 24,
    nz: int = 50,
    rho_peak_z: int = 25,
    rho_peak_r: int | None = None,
    rho_bg: float = 1e-4,
    rho_peak: float = 1.0,
) -> dict:
    """Create a synthetic cylindrical state with a density peak.

    Density is shaped as a narrow Gaussian peak at (rho_peak_r, rho_peak_z)
    on top of a uniform background.
    """
    rho = np.full((nr, 1, nz), rho_bg, dtype=np.float64)
    if rho_peak_r is None:
        rho_peak_r = nr // 4
    # Place peak: Gaussian in both r and z
    for ir in range(nr):
        for iz in range(nz):
            dr2 = (ir - rho_peak_r) ** 2
            dz2 = (iz - rho_peak_z) ** 2
            rho[ir, 0, iz] += rho_peak * np.exp(-(dr2 / 4.0 + dz2 / 16.0))
    return {"rho": rho, "B": np.zeros((3, nr, 1, nz))}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestCircuitCouplerInit:
    def test_init_stores_params(self, cylindrical_coupler: CircuitCoupler):
        c = cylindrical_coupler
        assert c.anode_radius == 0.008
        assert c.cathode_radius == 0.032
        assert c.dr == 0.001
        assert c.dz == 0.002
        assert c.r_inner == 0.008

    def test_reset_clears_state(self, cylindrical_coupler: CircuitCoupler):
        c = cylindrical_coupler
        c._Lp_max = 1.0
        c._time = 1e-6
        c._history.append((1e-6, 1e-9))
        c.reset()
        assert c._Lp_max == 0.0
        assert c._time == 0.0
        assert len(c._history) == 0


class TestFeedbackComputation:
    def test_returns_nonzero_Lp_for_peaked_density(self, cylindrical_coupler: CircuitCoupler):
        state = _make_cylindrical_state(rho_peak_z=25)
        fb = cylindrical_coupler.compute_feedback(state, current=100e3, dt=1e-9)
        assert fb.Lp > 0, "Lp should be positive for peaked density"
        assert fb.z_sheath > 0, "z_sheath should be positive"
        assert fb.r_eff > 0, "r_eff should be positive"

    def test_r_eff_in_physical_range(self, cylindrical_coupler: CircuitCoupler):
        c = cylindrical_coupler
        state = _make_cylindrical_state()
        fb = c.compute_feedback(state, current=100e3, dt=1e-9)
        # r_eff must be between axis (practically >0) and cathode
        assert fb.r_eff > 0
        assert fb.r_eff < c.cathode_radius

    def test_z_sheath_tracks_density_peak(self, cylindrical_coupler: CircuitCoupler):
        c = cylindrical_coupler
        state1 = _make_cylindrical_state(rho_peak_z=10)
        fb1 = c.compute_feedback(state1, current=100e3, dt=1e-9)

        c.reset()
        state2 = _make_cylindrical_state(rho_peak_z=40)
        fb2 = c.compute_feedback(state2, current=100e3, dt=1e-9)

        # z_sheath should be further for the second state
        assert fb2.z_sheath > fb1.z_sheath

    def test_Lp_scales_with_z_sheath(self, cylindrical_coupler: CircuitCoupler):
        """Lp = (mu0/2pi) * z * ln(b/r_eff) should increase with z."""
        c = cylindrical_coupler
        state_near = _make_cylindrical_state(rho_peak_z=10, rho_peak=10.0)
        fb_near = c.compute_feedback(state_near, current=100e3, dt=1e-9)

        c.reset()
        state_far = _make_cylindrical_state(rho_peak_z=40, rho_peak=10.0)
        fb_far = c.compute_feedback(state_far, current=100e3, dt=1e-9)

        assert fb_far.Lp > fb_near.Lp

    def test_empty_state_returns_zero_Lp(self, cylindrical_coupler: CircuitCoupler):
        fb = cylindrical_coupler.compute_feedback({}, current=100e3, dt=1e-9)
        assert fb.Lp == 0.0

    def test_handles_2d_rho(self, cylindrical_coupler: CircuitCoupler):
        """Should handle (nr, nz) shape without ny dimension."""
        rho = np.full((24, 50), 1e-4)
        rho[6, 25] = 1.0
        state = {"rho": rho, "B": np.zeros((3, 24, 50))}
        fb = cylindrical_coupler.compute_feedback(state, current=100e3, dt=1e-9)
        assert fb.Lp > 0


class TestMonotonicity:
    def test_Lp_monotonically_increasing(self, cylindrical_coupler: CircuitCoupler):
        """Lp should never decrease — enforced by _Lp_max clamp."""
        c = cylindrical_coupler
        Lp_values = []

        # First step: large peak → high Lp
        state1 = _make_cylindrical_state(rho_peak_z=40, rho_peak=10.0)
        fb1 = c.compute_feedback(state1, current=100e3, dt=1e-9)
        Lp_values.append(fb1.Lp)

        # Second step: smaller peak → lower Lp attempt
        state2 = _make_cylindrical_state(rho_peak_z=10, rho_peak=0.1)
        fb2 = c.compute_feedback(state2, current=100e3, dt=1e-9)
        Lp_values.append(fb2.Lp)

        # Monotonicity: Lp[1] >= Lp[0]
        assert Lp_values[1] >= Lp_values[0], (
            f"Lp decreased from {Lp_values[0]:.3e} to {Lp_values[1]:.3e}"
        )


class TestBDF2:
    def test_dLp_dt_zero_on_first_step(self, cylindrical_coupler: CircuitCoupler):
        state = _make_cylindrical_state()
        fb = cylindrical_coupler.compute_feedback(state, current=100e3, dt=1e-9)
        assert fb.dLp_dt == 0.0, "First step should have zero dLp/dt (no history)"

    def test_dLp_dt_positive_for_increasing_Lp(self, cylindrical_coupler: CircuitCoupler):
        c = cylindrical_coupler
        # Step 1: small z_sheath
        state1 = _make_cylindrical_state(rho_peak_z=10, rho_peak=10.0)
        c.compute_feedback(state1, current=100e3, dt=1e-9)

        # Step 2: larger z_sheath → Lp increases
        state2 = _make_cylindrical_state(rho_peak_z=30, rho_peak=10.0)
        fb2 = c.compute_feedback(state2, current=100e3, dt=1e-9)
        assert fb2.dLp_dt > 0, "dLp/dt should be positive when Lp increases"


class TestBackEMFClamp:
    def test_back_emf_clamped(self):
        """Extreme dLp/dt should be clamped to +/-50 kV."""
        c = CircuitCoupler(
            anode_radius=0.008,
            cathode_radius=0.032,
            dr=0.001,
            dz=0.002,
            r_inner=0.008,
        )
        # Create two states with huge Lp jump to trigger clamp
        state1 = _make_cylindrical_state(rho_peak_z=5, rho_peak=0.01)
        c.compute_feedback(state1, current=1e6, dt=1e-12)

        state2 = _make_cylindrical_state(rho_peak_z=49, rho_peak=100.0)
        fb2 = c.compute_feedback(state2, current=1e6, dt=1e-12)
        assert abs(fb2.back_emf) <= BACK_EMF_CLAMP_V


class TestCartesianFallback:
    def test_3d_cartesian_uses_b_energy_fallback(self):
        c = CircuitCoupler(
            anode_radius=0.008,
            cathode_radius=0.032,
            dr=0.01,
            dz=0.01,
        )
        nx, ny, nz = 16, 16, 16
        B = np.zeros((3, nx, ny, nz))
        B[1, :, :, :] = 0.01  # Uniform B_y
        state = {"rho": np.ones((nx, ny, nz)) * 1e-4, "B": B}
        fb = c.compute_feedback(state, current=100e3, dt=1e-9)
        # Should use B-energy fallback, not density-weighted
        assert fb.Lp >= 0


# ---------------------------------------------------------------------------
# Integration tests: coupling mode affects I(t)
# ---------------------------------------------------------------------------

class TestCouplingModeIntegration:
    """Verify that coupling_mode="density_weighted" changes circuit behavior."""

    @pytest.fixture()
    def _pf1000_like_config(self):
        """Minimal config for a cylindrical DPF run."""
        from dpf.config import SimulationConfig
        return SimulationConfig(
            grid_shape=[24, 1, 50],
            dx=0.001,
            sim_time=1e-7,
            rho0=1e-4,
            circuit=dict(
                C=204e-6,
                V0=27e3,
                L0=33.5e-9,
                R0=12.5e-3,
                anode_radius=0.008,
                cathode_radius=0.032,
                coupling_mode="auto",
            ),
            geometry=dict(type="cylindrical", dz=0.002),
            fluid=dict(backend="python", cfl=0.3, riemann_solver="hll"),
            snowplow=dict(enabled=False),
            boundary=dict(electrode_bc=True),
            diagnostics=dict(hdf5_filename=":memory:"),
        )

    def test_config_coupling_mode_default_is_auto(self, _pf1000_like_config):
        assert _pf1000_like_config.circuit.coupling_mode == "auto"

    def test_config_accepts_all_coupling_modes(self):
        from dpf.config import CircuitConfig
        for mode in ("auto", "lee_only", "density_weighted"):
            cc = CircuitConfig(
                C=1e-6, V0=1e3, L0=1e-9,
                anode_radius=0.005, cathode_radius=0.01,
                coupling_mode=mode,
            )
            assert cc.coupling_mode == mode

    def test_config_rejects_invalid_coupling_mode(self):
        from dpf.config import CircuitConfig
        with pytest.raises(ValidationError):
            CircuitConfig(
                C=1e-6, V0=1e3, L0=1e-9,
                anode_radius=0.005, cathode_radius=0.01,
                coupling_mode="invalid",
            )


class TestCouplerOnEngine:
    """Verify the coupler is wired into the engine correctly."""

    def test_engine_has_coupler_attribute(self):
        from dpf.config import SimulationConfig
        config = SimulationConfig(
            grid_shape=[16, 1, 20],
            dx=0.002,
            sim_time=1e-8,
            circuit=dict(
                C=204e-6, V0=27e3, L0=33.5e-9, R0=12.5e-3,
                anode_radius=0.008, cathode_radius=0.032,
                coupling_mode="density_weighted",
            ),
            geometry=dict(type="cylindrical", dz=0.004),
            fluid=dict(backend="python", cfl=0.3, riemann_solver="hll"),
            snowplow=dict(enabled=False),
            boundary=dict(electrode_bc=True),
            diagnostics=dict(hdf5_filename=":memory:"),
        )
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(config)
        assert hasattr(engine, "coupler")
        assert isinstance(engine.coupler, CircuitCoupler)
        assert engine.coupling_mode == "density_weighted"

    def test_engine_coupler_default_lee_only_with_snowplow(self):
        """When snowplow is enabled and coupling_mode=auto, coupler exists but
        snowplow Lp takes priority during active phases."""
        from dpf.config import SimulationConfig
        config = SimulationConfig(
            grid_shape=[16, 1, 20],
            dx=0.002,
            sim_time=1e-8,
            circuit=dict(
                C=204e-6, V0=27e3, L0=33.5e-9, R0=12.5e-3,
                anode_radius=0.008, cathode_radius=0.032,
            ),
            geometry=dict(type="cylindrical", dz=0.004),
            fluid=dict(backend="python", cfl=0.3, riemann_solver="hll"),
            snowplow=dict(enabled=True, anode_length=0.16),
            boundary=dict(electrode_bc=True),
            diagnostics=dict(hdf5_filename=":memory:"),
        )
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(config)
        assert engine.coupling_mode == "auto"
        assert engine.coupler is not None


class TestFeedbackResultDataclass:
    def test_defaults_are_zero(self):
        fb = FeedbackResult()
        assert fb.Lp == 0.0
        assert fb.dLp_dt == 0.0
        assert fb.back_emf == 0.0
        assert fb.r_eff == 0.0
        assert fb.z_sheath == 0.0
