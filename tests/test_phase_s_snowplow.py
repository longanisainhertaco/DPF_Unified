"""Phase S: Snowplow dynamics tests.

Tests for the SnowplowModel (Lee model Phase 2 — axial rundown) and its
integration into the simulation engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import mu_0, pi
from dpf.fluid.snowplow import SnowplowModel

# ---------------------------------------------------------------------------
# PF-1000-like parameters for realistic testing
# ---------------------------------------------------------------------------
PF1000_A = 0.0575       # anode radius [m]
PF1000_B = 0.08         # cathode radius [m]
PF1000_RHO0 = 4e-4      # fill density [kg/m^3]
PF1000_L_ANODE = 0.16   # anode length [m]
PF1000_FM = 0.3          # mass fraction
PF1000_P_FILL = 400.0   # fill pressure [Pa]


@pytest.fixture
def snowplow():
    """Create a PF-1000-like snowplow model."""
    return SnowplowModel(
        anode_radius=PF1000_A,
        cathode_radius=PF1000_B,
        fill_density=PF1000_RHO0,
        anode_length=PF1000_L_ANODE,
        mass_fraction=PF1000_FM,
        fill_pressure_Pa=PF1000_P_FILL,
    )


class TestSnowplowInstantiation:
    """Test SnowplowModel initialization."""

    def test_basic_instantiation(self, snowplow):
        assert snowplow.a == PF1000_A
        assert snowplow.b == PF1000_B
        assert snowplow.rho0 == PF1000_RHO0
        assert snowplow.L_anode == PF1000_L_ANODE
        assert snowplow.f_m == PF1000_FM
        assert snowplow.p_fill == PF1000_P_FILL

    def test_initial_position_nonzero(self, snowplow):
        """Sheath starts at a small nonzero position."""
        assert snowplow.z > 0.0
        assert snowplow.z < snowplow.L_anode

    def test_initial_velocity_zero(self, snowplow):
        assert snowplow.v == 0.0

    def test_phase_is_rundown(self, snowplow):
        assert snowplow.phase == "rundown"
        assert not snowplow.rundown_complete

    def test_geometric_constants(self, snowplow):
        """Check derived geometric constants."""
        ln_ba = np.log(PF1000_B / PF1000_A)
        A_ann = pi * (PF1000_B**2 - PF1000_A**2)
        assert snowplow.ln_ba == pytest.approx(ln_ba, rel=1e-10)
        assert snowplow.A_annular == pytest.approx(A_ann, rel=1e-10)

    def test_force_coefficient(self, snowplow):
        """F_coeff = (mu_0 / 4pi) * ln(b/a)."""
        expected = (mu_0 / (4.0 * pi)) * np.log(PF1000_B / PF1000_A)
        assert snowplow.F_coeff == pytest.approx(expected, rel=1e-10)


class TestSnowplowProperties:
    """Test computed properties."""

    def test_swept_mass(self, snowplow):
        """m = rho0 * A * z * f_m."""
        m = PF1000_RHO0 * snowplow.A_annular * snowplow.z * PF1000_FM
        assert snowplow.swept_mass == pytest.approx(m, rel=1e-10)

    def test_plasma_inductance(self, snowplow):
        """L = (mu_0 / 4pi) * ln(b/a) * z."""
        L = snowplow.L_coeff * snowplow.z
        assert snowplow.plasma_inductance == pytest.approx(L, rel=1e-10)

    def test_sheath_position_property(self, snowplow):
        assert snowplow.sheath_position == snowplow.z

    def test_sheath_velocity_property(self, snowplow):
        assert snowplow.sheath_velocity == snowplow.v


class TestSnowplowStep:
    """Test the snowplow step function."""

    def test_sheath_advances_with_current(self, snowplow):
        """Sheath position increases when current flows."""
        z0 = snowplow.z
        dt = 1e-8
        current = 1e6  # 1 MA typical for PF-1000

        result = snowplow.step(dt, current)

        assert snowplow.z > z0
        assert snowplow.v > 0.0
        assert result["z_sheath"] == snowplow.z
        assert result["v_sheath"] == snowplow.v

    def test_zero_current_no_acceleration(self, snowplow):
        """No magnetic force when I=0; sheath stays put (pressure may slow it)."""
        z0 = snowplow.z
        snowplow.step(1e-8, current=0.0)
        # With zero current, F_mag = 0 and only back-pressure acts.
        # Sheath should not advance (v was 0, pressure is decelerating)
        assert snowplow.z == pytest.approx(z0, rel=1e-6)

    def test_mass_conservation(self, snowplow):
        """Swept mass = rho0 * A * z * f_m at all times."""
        dt = 1e-8
        for _ in range(100):
            snowplow.step(dt, current=5e5)
        expected = PF1000_RHO0 * snowplow.A_annular * snowplow.z * PF1000_FM
        assert snowplow.swept_mass == pytest.approx(expected, rel=1e-10)

    def test_inductance_consistency(self, snowplow):
        """L_plasma = L_coeff * z at all times."""
        dt = 1e-8
        for _ in range(50):
            result = snowplow.step(dt, current=5e5)
        expected_L = snowplow.L_coeff * snowplow.z
        assert result["L_plasma"] == pytest.approx(expected_L, rel=1e-10)
        assert snowplow.plasma_inductance == pytest.approx(expected_L, rel=1e-10)

    def test_dL_dt_positive(self, snowplow):
        """dL/dt should be positive when sheath advances."""
        dt = 1e-8
        # Give it a few steps to build velocity
        for _ in range(10):
            snowplow.step(dt, current=1e6)
        result = snowplow.step(dt, current=1e6)
        assert result["dL_dt"] > 0.0

    def test_dL_dt_equals_Lcoeff_times_v(self, snowplow):
        """dL/dt = L_coeff * v (analytically exact)."""
        dt = 1e-8
        for _ in range(20):
            result = snowplow.step(dt, current=1e6)
        expected = snowplow.L_coeff * snowplow.v
        assert result["dL_dt"] == pytest.approx(expected, rel=1e-10)

    def test_magnetic_force_scales_with_I_squared(self, snowplow):
        """F_mag = F_coeff * I^2."""
        dt = 1e-8
        r1 = snowplow.step(dt, current=1e5)
        sp2 = SnowplowModel(
            anode_radius=PF1000_A,
            cathode_radius=PF1000_B,
            fill_density=PF1000_RHO0,
            anode_length=PF1000_L_ANODE,
            mass_fraction=PF1000_FM,
            fill_pressure_Pa=PF1000_P_FILL,
        )
        r2 = sp2.step(dt, current=2e5)
        # F_mag scales as I^2, so 4× ratio
        assert r2["F_magnetic"] == pytest.approx(4.0 * r1["F_magnetic"], rel=1e-10)

    def test_pressure_force_opposes_motion(self, snowplow):
        """Pressure force acts against sheath motion."""
        result = snowplow.step(1e-8, current=1e6)
        assert result["F_pressure"] > 0.0
        assert result["F_magnetic"] > result["F_pressure"]  # Must overcome pressure


class TestSnowplowRundownCompletion:
    """Test rundown termination when sheath reaches end of anode."""

    def test_rundown_completes(self, snowplow):
        """Sheath eventually reaches the end of the anode."""
        dt = 1e-8
        for _ in range(100_000):
            snowplow.step(dt, current=1e6)
            if snowplow.rundown_complete:
                break
        assert snowplow.rundown_complete
        assert snowplow.z == snowplow.L_anode
        assert snowplow.phase == "radial"

    def test_radial_phase_after_rundown(self, snowplow):
        """After axial rundown, radial phase begins with growing L_plasma."""
        dt = 1e-8
        # Drive to rundown completion
        for _ in range(100_000):
            snowplow.step(dt, current=1e6)
            if snowplow.rundown_complete:
                break
        z_final = snowplow.z
        L_at_rundown = snowplow.plasma_inductance

        # First radial step: shock moves inward, L grows
        result = snowplow.step(dt, current=1e6)
        assert snowplow.z == z_final  # Axial position frozen
        assert snowplow.phase in ("radial", "pinch")
        assert snowplow.plasma_inductance > L_at_rundown  # L grows during radial
        assert result["r_shock"] < snowplow.b  # Shock has moved inward

    def test_pinch_reached(self, snowplow):
        """Radial shock eventually reaches pinch (r_min)."""
        dt = 1e-8
        # Drive through axial + radial
        for _ in range(200_000):
            snowplow.step(dt, current=1e6)
            if snowplow.pinch_complete:
                break
        assert snowplow.pinch_complete
        assert snowplow.phase == "pinch"
        assert snowplow.r_shock <= snowplow.r_pinch_min
        # After pinch, state is frozen
        L_pinch = snowplow.plasma_inductance
        snowplow.step(dt, current=1e6)
        assert snowplow.plasma_inductance == L_pinch

    def test_sheath_does_not_overshoot(self, snowplow):
        """Position never exceeds anode length."""
        dt = 1e-7  # Large timestep to force potential overshoot
        for _ in range(10_000):
            snowplow.step(dt, current=2e6)
            assert snowplow.z <= snowplow.L_anode


class TestSnowplowPhysics:
    """Tests verifying physical correctness."""

    def test_force_formula_matches_lee(self):
        """Verify force uses mu_0/(4pi) * ln(b/a) * (f_c * I)^2."""
        a, b = 0.01, 0.02
        f_c = 0.7
        sp = SnowplowModel(
            anode_radius=a, cathode_radius=b, fill_density=1e-4,
            anode_length=0.1, mass_fraction=0.3, fill_pressure_Pa=0.0,
            current_fraction=f_c,
        )
        current = 1e6
        # Expected force: (mu_0/4pi) * ln(b/a) * (f_c * I)^2
        F_expected = (mu_0 / (4 * pi)) * np.log(b / a) * (f_c * current)**2
        result = sp.step(1e-10, current)
        assert result["F_magnetic"] == pytest.approx(F_expected, rel=1e-10)

    def test_inductance_formula_coaxial(self):
        """L_plasma = (mu_0/2pi) * ln(b/a) * z."""
        a, b = 0.01, 0.02
        sp = SnowplowModel(
            anode_radius=a, cathode_radius=b, fill_density=1e-4,
            anode_length=0.1, mass_fraction=0.3,
        )
        z = 0.05
        sp.z = z
        L_expected = (mu_0 / (2 * pi)) * np.log(b / a) * z
        assert sp.plasma_inductance == pytest.approx(L_expected, rel=1e-10)

    def test_pf1000_timing_order_of_magnitude(self, snowplow):
        """PF-1000 sheath should reach end in ~microseconds."""
        dt = 1e-9  # 1 ns steps
        n_steps = 0
        max_steps = 10_000_000
        for _ in range(max_steps):
            snowplow.step(dt, current=1.5e6)  # 1.5 MA typical
            n_steps += 1
            if snowplow.rundown_complete:
                break
        t_rundown = n_steps * dt
        # PF-1000 rundown typically takes ~3-6 microseconds
        assert 0.5e-6 < t_rundown < 20e-6, f"Rundown took {t_rundown:.2e} s"

    def test_velocity_verlet_second_order(self):
        """Verify the integrator is second-order by refinement."""
        def run_snowplow(dt, n_steps):
            sp = SnowplowModel(
                anode_radius=0.01, cathode_radius=0.02, fill_density=1e-4,
                anode_length=1.0, mass_fraction=0.3, fill_pressure_Pa=0.0,
            )
            for _ in range(n_steps):
                sp.step(dt, current=1e5)
            return sp.z

        # Run with dt and dt/2
        dt1 = 1e-7
        n1 = 1000
        z1 = run_snowplow(dt1, n1)
        z2 = run_snowplow(dt1 / 2, n1 * 2)
        z4 = run_snowplow(dt1 / 4, n1 * 4)

        # Richardson extrapolation: error should scale as dt^2
        err1 = abs(z2 - z1)
        err2 = abs(z4 - z2)
        if err2 > 1e-15:
            order = np.log2(err1 / err2)
            # Should be approximately 2 for second-order
            assert order > 1.5, f"Convergence order {order:.2f}, expected ~2"


class TestSnowplowConfigIntegration:
    """Test snowplow configuration in SimulationConfig."""

    def test_snowplow_config_defaults(self):
        from dpf.config import SnowplowConfig
        cfg = SnowplowConfig()
        assert cfg.enabled is True
        assert cfg.mass_fraction == 0.3
        assert cfg.fill_pressure_Pa == 400.0
        assert cfg.anode_length == 0.16

    def test_snowplow_in_simulation_config(self):
        from dpf.config import SimulationConfig
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": True, "mass_fraction": 0.2},
        )
        assert config.snowplow.enabled is True
        assert config.snowplow.mass_fraction == 0.2

    def test_snowplow_disabled(self):
        from dpf.config import SimulationConfig
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": False},
        )
        assert config.snowplow.enabled is False
