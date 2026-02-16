"""Phase T tests: Snowplow validation and DPF physics consistency.

Tests for:
1. Snowplow L_coeff = 2 * F_coeff fix verification
2. Current fraction f_c in force calculation
3. Radial compression phase dynamics
4. Radial phase physics formulas
5. PF-1000 I_peak consistency between suite.py and experimental.py
6. Lee model docstring correctness (mu_0/(4pi), not mu_0/2)
7. Coupled snowplow+circuit integration

References:
    Lee, S. & Saw, S.H., Phys. Plasmas 21, 072501 (2014).
    Miyoshi, T. & Kusano, K., JCP 208, 315-344 (2005).
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.config import SimulationConfig, SnowplowConfig
from dpf.constants import mu_0, pi
from dpf.fluid.snowplow import SnowplowModel
from dpf.validation.experimental import PF1000_DATA
from dpf.validation.suite import DEVICE_REGISTRY

# =====================================================================
# Helper: create a standard PF-1000-like snowplow model for testing
# =====================================================================

def _pf1000_snowplow(
    current_fraction: float = 0.7,
    mass_fraction: float = 0.3,
    radial_mass_fraction: float | None = None,
) -> SnowplowModel:
    """Create a PF-1000-like SnowplowModel with known parameters."""
    # Fill density from ideal gas law: p = n*k_B*T, rho = n*m_d
    # ~4e-4 kg/m^3 for 3.5 Torr D2 at 300 K
    return SnowplowModel(
        anode_radius=0.0575,
        cathode_radius=0.08,
        fill_density=4e-4,
        anode_length=0.16,
        mass_fraction=mass_fraction,
        fill_pressure_Pa=400.0,
        current_fraction=current_fraction,
        radial_mass_fraction=radial_mass_fraction,
    )


def _make_radial_snowplow(
    r_shock: float = 0.04,
    vr: float = -1e4,
    **kwargs,
) -> SnowplowModel:
    """Create a SnowplowModel already positioned in the radial phase.

    When the shock starts at r=b (cathode), swept mass is zero and any force
    causes immediate pinch in a single timestep. This helper places the shock
    at a reasonable intermediate radius with nonzero swept mass so that
    radial-phase dynamics can be tested over multiple steps.

    Args:
        r_shock: Initial shock radius [m]. Must be between a and b.
        vr: Initial radial velocity [m/s] (negative = inward).
        **kwargs: Forwarded to _pf1000_snowplow().
    """
    sp = _pf1000_snowplow(**kwargs)
    sp.phase = "radial"
    sp._rundown_complete = True
    sp.z = sp.L_anode
    sp._L_axial_frozen = sp.L_coeff * sp.L_anode
    sp.r_shock = r_shock
    sp.vr = vr
    return sp


# =====================================================================
# 1. Snowplow L_coeff = 2 * F_coeff fix verification
# =====================================================================

class TestLCoeffFix:
    """Verify L_coeff = 2 * F_coeff (the factor-of-2 relationship)."""

    def test_L_coeff_equals_2_F_coeff(self) -> None:
        """L_coeff must be exactly 2 * F_coeff.

        Physics: F = dW/dz = (1/2) * I^2 * dL/dz
        F_coeff = mu_0/(4*pi) * ln(b/a) and L_coeff = mu_0/(2*pi) * ln(b/a).
        """
        sp = _pf1000_snowplow()
        assert sp.L_coeff == pytest.approx(2.0 * sp.F_coeff, rel=1e-14)

    def test_F_coeff_formula(self) -> None:
        """F_coeff = mu_0 / (4*pi) * ln(b/a)."""
        sp = _pf1000_snowplow()
        expected = (mu_0 / (4.0 * pi)) * np.log(sp.b / sp.a)
        assert sp.F_coeff == pytest.approx(expected, rel=1e-14)

    def test_L_coeff_formula(self) -> None:
        """L_coeff = mu_0 / (2*pi) * ln(b/a)."""
        sp = _pf1000_snowplow()
        expected = (mu_0 / (2.0 * pi)) * np.log(sp.b / sp.a)
        assert sp.L_coeff == pytest.approx(expected, rel=1e-14)

    def test_plasma_inductance_axial(self) -> None:
        """During rundown, L_plasma = L_coeff * z = (mu_0/2pi)*ln(b/a)*z."""
        sp = _pf1000_snowplow()
        # Model starts at z ~ 1e-4 m
        z = sp.z
        expected_L = sp.L_coeff * z
        assert sp.plasma_inductance == pytest.approx(expected_L, rel=1e-10)


# =====================================================================
# 2. Current fraction f_c verification
# =====================================================================

class TestCurrentFraction:
    """Verify that force uses (f_c * I)^2 with f_c=0.7 default."""

    def test_default_current_fraction(self) -> None:
        """Default current fraction should be 0.7."""
        sp = _pf1000_snowplow()
        assert sp.f_c == pytest.approx(0.7, abs=1e-15)

    def test_force_uses_fc_squared(self) -> None:
        """Axial magnetic force must use (f_c * I)^2, not I^2.

        F_mag = F_coeff * (f_c * I)^2
        """
        fc = 0.7
        sp = _pf1000_snowplow(current_fraction=fc)
        I_test = 500e3  # 500 kA
        dt = 1e-9  # very small dt to minimize dynamics

        result = sp.step(dt, I_test)
        F_mag = result["F_magnetic"]

        # Expected: F_coeff * (f_c * I)^2
        expected = sp.F_coeff * (fc * I_test) ** 2
        assert F_mag == pytest.approx(expected, rel=1e-10)

    def test_fc_affects_force_magnitude(self) -> None:
        """Reducing f_c should reduce force by f_c^2 ratio."""
        I_test = 500e3
        dt = 1e-9

        sp1 = _pf1000_snowplow(current_fraction=0.7)
        result1 = sp1.step(dt, I_test)

        sp2 = _pf1000_snowplow(current_fraction=0.5)
        result2 = sp2.step(dt, I_test)

        # Force ratio should be (0.5/0.7)^2
        ratio = result2["F_magnetic"] / result1["F_magnetic"]
        expected_ratio = (0.5 / 0.7) ** 2
        assert ratio == pytest.approx(expected_ratio, rel=1e-10)

    def test_fc_one_recovers_full_current(self) -> None:
        """With f_c=1, force should use the full current."""
        sp = _pf1000_snowplow(current_fraction=1.0)
        I_test = 300e3
        dt = 1e-9

        result = sp.step(dt, I_test)
        expected = sp.F_coeff * I_test**2
        assert result["F_magnetic"] == pytest.approx(expected, rel=1e-10)


# =====================================================================
# 3. Radial compression phase tests
# =====================================================================

def _run_to_radial(sp: SnowplowModel, I_current: float = 1.0e6) -> None:
    """Step the snowplow through axial rundown until it enters the radial phase."""
    dt = 1e-8
    max_steps = 100_000
    for _ in range(max_steps):
        sp.step(dt, I_current)
        if sp.phase == "radial":
            return
    raise RuntimeError("Snowplow did not reach radial phase within max_steps")


class TestRadialCompression:
    """Tests for radial compression phase behavior."""

    def test_radial_starts_after_rundown(self) -> None:
        """Radial phase must start only after the sheath reaches the anode end."""
        sp = _pf1000_snowplow()
        assert sp.phase == "rundown"
        assert not sp.rundown_complete

        _run_to_radial(sp)
        assert sp.phase == "radial"
        assert sp.rundown_complete
        assert sp.z == pytest.approx(sp.L_anode, rel=1e-10)

    def test_shock_radius_decreases_during_radial(self) -> None:
        """Shock radius must decrease (compress inward) during radial phase.

        Uses _make_radial_snowplow to start with nonzero swept mass
        (r_shock < b), since M_slug=0 at r=b causes instant pinch.
        """
        sp = _make_radial_snowplow(r_shock=0.06, vr=-1e4)

        r_initial = sp.shock_radius
        assert r_initial < sp.b, "Shock should be inside cathode"

        # Take several radial steps with moderate current
        dt = 1e-10
        for _ in range(500):
            sp.step(dt, 500e3)

        assert sp.shock_radius < r_initial, (
            f"Shock radius should decrease: {sp.shock_radius} < {r_initial}"
        )

    def test_L_plasma_grows_during_radial(self) -> None:
        """Total L_plasma should increase as the shock compresses inward.

        L = L_axial + (mu_0/2pi)*z_f*ln(b/r_s) grows as r_s decreases.
        """
        sp = _make_radial_snowplow(r_shock=0.06, vr=-1e4)
        L_at_start = sp.plasma_inductance

        dt = 1e-10
        for _ in range(500):
            sp.step(dt, 500e3)

        L_after = sp.plasma_inductance
        assert L_after > L_at_start, (
            f"L_plasma should grow during compression: {L_after} > {L_at_start}"
        )

    def test_dL_dt_positive_during_radial(self) -> None:
        """dL/dt should be positive during radial phase (vr < 0 -> dL/dt > 0).

        dL/dt = -(mu_0/2pi)*z_f*vr/r_s, and vr < 0 (inward), so dL/dt > 0.
        """
        sp = _make_radial_snowplow(r_shock=0.04, vr=-1e4)

        dt = 1e-10
        result = sp.step(dt, 500e3)

        assert result["dL_dt"] > 0, (
            f"dL/dt should be positive during inward compression: {result['dL_dt']}"
        )
        assert result["vr_shock"] <= 0, (
            f"Radial velocity should be non-positive (inward): {result['vr_shock']}"
        )

    def test_radial_swept_mass_grows(self) -> None:
        """Radial swept mass grows as the shock moves inward.

        M_slug = f_mr * rho0 * pi * (b^2 - r_s^2) * z_f
        As r_s decreases, (b^2 - r_s^2) increases, so mass grows.
        """
        sp = _make_radial_snowplow(r_shock=0.06, vr=-1e4)
        m_initial = sp.radial_swept_mass
        assert m_initial > 0, "Swept mass should be nonzero at r < b"

        dt = 1e-10
        for _ in range(500):
            sp.step(dt, 500e3)

        m_after = sp.radial_swept_mass
        assert m_after > m_initial, (
            f"Radial swept mass should grow: {m_after} > {m_initial}"
        )

    def test_pinch_detection(self) -> None:
        """Pinch detected when r_shock <= 0.1 * a."""
        sp = _make_radial_snowplow(r_shock=0.02, vr=-5e4)

        # Run radial phase until pinch with small dt
        dt = 1e-10
        max_steps = 500_000
        pinched = False
        for _ in range(max_steps):
            sp.step(dt, 500e3)
            if sp.pinch_complete:
                pinched = True
                break

        assert pinched, "Snowplow should reach pinch"
        assert sp.phase == "pinch"
        assert sp.shock_radius <= 0.1 * sp.a + 1e-10

    def test_frozen_state_after_pinch(self) -> None:
        """After pinch, state should be frozen (dL/dt=0, F=0)."""
        sp = _make_radial_snowplow(r_shock=0.02, vr=-5e4)

        # Run to pinch
        dt = 1e-10
        for _ in range(500_000):
            sp.step(dt, 500e3)
            if sp.pinch_complete:
                break

        assert sp.pinch_complete

        # Take additional steps -- state should be frozen
        r_at_pinch = sp.shock_radius
        L_at_pinch = sp.plasma_inductance
        result = sp.step(dt, 500e3)

        assert result["dL_dt"] == pytest.approx(0.0, abs=1e-20)
        assert result["F_magnetic"] == pytest.approx(0.0, abs=1e-20)
        assert result["F_pressure"] == pytest.approx(0.0, abs=1e-20)
        assert sp.shock_radius == pytest.approx(r_at_pinch, rel=1e-15)
        assert sp.plasma_inductance == pytest.approx(L_at_pinch, rel=1e-15)
        assert not sp.is_active


# =====================================================================
# 4. Radial phase physics formulas
# =====================================================================

class TestRadialPhysicsFormulas:
    """Verify radial phase formulas against Lee model references.

    All tests use _make_radial_snowplow to place the shock at a finite
    radius inside the cathode, ensuring nonzero swept mass and stable
    multi-step integration.
    """

    def test_radial_force_formula(self) -> None:
        """F_rad = (mu_0/4pi) * (f_c * I)^2 * z_f / r_s.

        The step returns the force computed at the Verlet half-step position,
        which differs slightly from the post-step position. We compare the
        returned force against the formula evaluated at the pre-step r_s.
        """
        sp = _make_radial_snowplow(r_shock=0.04, vr=-1e4)

        I_test = 500e3
        r_s_pre = sp.shock_radius
        z_f = sp.L_anode

        # Expected force at the pre-step position
        expected_F = (mu_0 / (4.0 * pi)) * (sp.f_c * I_test) ** 2 * z_f / r_s_pre

        dt = 1e-10
        result = sp.step(dt, I_test)

        assert result["F_magnetic"] > 0, "Radial force must be positive"
        # Velocity-Verlet recomputes force at new position, so allow 5% tolerance
        assert result["F_magnetic"] == pytest.approx(expected_F, rel=0.05)

    def test_radial_L_plasma_formula(self) -> None:
        """L_plasma = L_axial + (mu_0/2pi) * z_f * ln(b/r_s)."""
        sp = _make_radial_snowplow(r_shock=0.04, vr=-1e4)

        # Step a few times to evolve the state
        dt = 1e-10
        for _ in range(100):
            sp.step(dt, 500e3)

        r_s = max(sp.shock_radius, sp.r_pinch_min)
        z_f = sp.L_anode
        L_axial_expected = sp._L_axial_frozen
        L_radial_expected = (mu_0 / (2.0 * pi)) * z_f * np.log(sp.b / r_s)
        L_total_expected = L_axial_expected + L_radial_expected

        assert sp.plasma_inductance == pytest.approx(L_total_expected, rel=1e-10)

    def test_dL_dt_formula(self) -> None:
        """dL/dt = -(mu_0/2pi) * z_f * vr / r_s."""
        sp = _make_radial_snowplow(r_shock=0.04, vr=-1e4)

        dt = 1e-10
        result = sp.step(dt, 500e3)

        r_s = max(sp.shock_radius, 1e-10)
        z_f = sp.L_anode
        vr = sp.vr  # should be negative (inward)
        expected_dL_dt = -(mu_0 / (2.0 * pi)) * z_f * vr / r_s

        assert result["dL_dt"] == pytest.approx(expected_dL_dt, rel=1e-10)

    def test_radial_swept_mass_formula(self) -> None:
        """M_slug = f_mr * rho0 * pi * (b^2 - r_s^2) * z_f."""
        sp = _make_radial_snowplow(
            r_shock=0.04, vr=-1e4, radial_mass_fraction=0.2,
        )

        dt = 1e-10
        for _ in range(100):
            sp.step(dt, 500e3)

        expected = sp.f_mr * sp.rho0 * pi * (sp.b**2 - sp.r_shock**2) * sp.L_anode
        assert sp.radial_swept_mass == pytest.approx(expected, rel=1e-10)


# =====================================================================
# 5. PF-1000 I_peak consistency
# =====================================================================

class TestPF1000Consistency:
    """Ensure PF-1000 parameters are consistent between suite.py and experimental.py."""

    def test_peak_current_match(self) -> None:
        """suite.py PF-1000.peak_current_A == experimental.py PF1000_DATA.peak_current."""
        suite_Ipeak = DEVICE_REGISTRY["PF-1000"].peak_current_A
        exp_Ipeak = PF1000_DATA.peak_current
        assert suite_Ipeak == pytest.approx(exp_Ipeak, rel=1e-10), (
            f"Peak current mismatch: suite={suite_Ipeak}, experimental={exp_Ipeak}"
        )

    def test_peak_current_value_1_87MA(self) -> None:
        """Both sources should report 1.87 MA peak current."""
        assert DEVICE_REGISTRY["PF-1000"].peak_current_A == pytest.approx(1.87e6, rel=1e-10)
        assert PF1000_DATA.peak_current == pytest.approx(1.87e6, rel=1e-10)

    def test_anode_radius_consistency(self) -> None:
        """Anode radius should be 57.5 mm = 0.0575 m in both sources."""
        assert DEVICE_REGISTRY["PF-1000"].anode_radius == pytest.approx(0.0575, rel=1e-10)
        assert PF1000_DATA.anode_radius == pytest.approx(0.0575, rel=1e-10)

    def test_cathode_radius_consistency(self) -> None:
        """Cathode radius should be 80 mm = 0.08 m in both sources."""
        assert DEVICE_REGISTRY["PF-1000"].cathode_radius == pytest.approx(0.08, rel=1e-10)
        assert PF1000_DATA.cathode_radius == pytest.approx(0.08, rel=1e-10)

    def test_capacitance_consistency(self) -> None:
        """Bank capacitance should be consistent between suite and experimental."""
        suite_C = DEVICE_REGISTRY["PF-1000"].C
        exp_C = PF1000_DATA.capacitance
        assert suite_C == pytest.approx(exp_C, rel=1e-10)


# =====================================================================
# 6. Lee model docstring correctness
# =====================================================================

class TestDocstringCorrectness:
    """Verify Lee model force formula docstrings use mu_0/(4pi)."""

    def test_snowplow_docstring_force_formula(self) -> None:
        """snowplow.py module docstring must contain mu_0/(4pi) for force."""
        import dpf.fluid.snowplow as snowplow_mod

        docstring = snowplow_mod.__doc__
        assert docstring is not None, "snowplow module should have a docstring"
        # The docstring should reference mu_0 / 4pi (not mu_0/2)
        assert "mu_0 / 4pi" in docstring or "mu_0/4pi" in docstring, (
            "Snowplow module docstring should reference mu_0/(4pi) for force"
        )

    def test_snowplow_no_mu0_over_2_in_force(self) -> None:
        """snowplow.py F_coeff comment must not use mu_0/2 for force."""
        import inspect

        import dpf.fluid.snowplow as snowplow_mod

        source = inspect.getsource(snowplow_mod.SnowplowModel.__init__)
        # The F_coeff line should use mu_0/(4*pi), not mu_0/2
        assert "mu_0 / (4.0 * pi)" in source, (
            "F_coeff should be computed as mu_0 / (4.0 * pi) * ln_ba"
        )

    def test_lee_model_docstring_uses_4pi(self) -> None:
        """lee_model_comparison.py docstring must use mu_0/(4*pi) for force."""
        import dpf.validation.lee_model_comparison as lee_mod

        docstring = lee_mod.__doc__
        assert docstring is not None
        assert "mu_0 / (4*pi)" in docstring or "mu_0/(4*pi)" in docstring, (
            "Lee model docstring should use mu_0/(4*pi) for force formula"
        )


# =====================================================================
# 7. Coupled snowplow+circuit basic integration
# =====================================================================

class TestSnowplowCircuitCoupling:
    """Test snowplow integration with the SimulationEngine."""

    def test_config_snowplow_enabled(self) -> None:
        """SnowplowConfig should have enabled=True by default."""
        cfg = SnowplowConfig()
        assert cfg.enabled is True
        assert cfg.mass_fraction == pytest.approx(0.3)
        assert cfg.current_fraction == pytest.approx(0.7)
        assert cfg.fill_pressure_Pa == pytest.approx(400.0)
        assert cfg.anode_length == pytest.approx(0.16)

    def test_config_snowplow_disabled(self) -> None:
        """When snowplow.enabled=False, engine should not create SnowplowModel."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": False},
        )
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(config)
        assert engine.snowplow is None

    def test_config_snowplow_enabled_creates_model(self) -> None:
        """When snowplow.enabled=True, engine creates a SnowplowModel."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": True},
        )
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(config)
        assert engine.snowplow is not None
        assert isinstance(engine.snowplow, SnowplowModel)

    def test_snowplow_feeds_L_plasma_to_circuit(self) -> None:
        """Snowplow L_plasma should be fed to the circuit solver during stepping."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-8,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": True, "anode_length": 0.01},
        )
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(config)

        # Take a single step
        engine.step()

        # After stepping, the coupling should have non-zero L_plasma
        # from the snowplow model
        coupling = engine._coupling
        assert coupling.Lp > 0, "Snowplow should provide positive L_plasma"

    def test_current_dip_signature(self) -> None:
        """When dL/dt is large, circuit current should decrease (current dip).

        This is the characteristic DPF current dip caused by the rapidly
        increasing plasma inductance during radial compression.

        We use a standalone snowplow + circuit solver to demonstrate the effect,
        avoiding the full MHD compute.
        """
        from dpf.circuit.rlc_solver import RLCSolver
        from dpf.core.bases import CouplingState

        # Small, fast circuit for testing
        solver = RLCSolver(C=28e-6, V0=15000, L0=50e-9, R0=0.005)

        # Simulate: first without dL/dt, then with large dL/dt
        coupling_no_dL = CouplingState(Lp=50e-9, dL_dt=0.0, R_plasma=0.01)
        coupling_large_dL = CouplingState(Lp=50e-9, dL_dt=0.1, R_plasma=0.01)

        dt = 1e-8

        # Step circuit without dL/dt
        result_no_dL = solver.step(coupling_no_dL, back_emf=0.0, dt=dt)
        # Reset solver state for fair comparison
        solver2 = RLCSolver(C=28e-6, V0=15000, L0=50e-9, R0=0.005)
        result_with_dL = solver2.step(coupling_large_dL, back_emf=0.0, dt=dt)

        # Large dL/dt should reduce the current (the "dip")
        assert result_with_dL.current < result_no_dL.current, (
            f"Large dL/dt should reduce current: {result_with_dL.current} "
            f"< {result_no_dL.current}"
        )


# =====================================================================
# Additional snowplow construction and property tests
# =====================================================================

class TestSnowplowConstruction:
    """Tests for SnowplowModel construction and initial state."""

    def test_initial_phase_is_rundown(self) -> None:
        """Model should start in 'rundown' phase."""
        sp = _pf1000_snowplow()
        assert sp.phase == "rundown"
        assert not sp.rundown_complete
        assert not sp.pinch_complete
        assert sp.is_active

    def test_initial_shock_radius_is_cathode(self) -> None:
        """Initial shock radius should be the cathode radius."""
        sp = _pf1000_snowplow()
        assert sp.shock_radius == pytest.approx(sp.b, rel=1e-15)

    def test_initial_radial_velocity_zero(self) -> None:
        """Initial radial velocity should be zero."""
        sp = _pf1000_snowplow()
        assert sp.vr == pytest.approx(0.0, abs=1e-20)

    def test_pinch_min_radius(self) -> None:
        """Minimum pinch radius should be 10% of anode radius."""
        sp = _pf1000_snowplow()
        assert sp.r_pinch_min == pytest.approx(0.1 * sp.a, rel=1e-15)

    def test_radial_mass_fraction_default(self) -> None:
        """When radial_mass_fraction=None, it should default to mass_fraction."""
        sp = _pf1000_snowplow(mass_fraction=0.3, radial_mass_fraction=None)
        assert sp.f_mr == pytest.approx(0.3, rel=1e-15)

    def test_radial_mass_fraction_custom(self) -> None:
        """Custom radial_mass_fraction should override mass_fraction."""
        sp = _pf1000_snowplow(mass_fraction=0.3, radial_mass_fraction=0.15)
        assert sp.f_mr == pytest.approx(0.15, rel=1e-15)
        assert sp.f_m == pytest.approx(0.3, rel=1e-15)

    def test_axial_swept_mass_formula(self) -> None:
        """Axial swept mass: m = rho0 * pi * (b^2 - a^2) * z * f_m."""
        sp = _pf1000_snowplow()
        expected = sp.rho0 * pi * (sp.b**2 - sp.a**2) * sp.z * sp.f_m
        assert sp.swept_mass == pytest.approx(expected, rel=1e-12)

    def test_step_returns_correct_keys(self) -> None:
        """step() return dict must have all required keys."""
        sp = _pf1000_snowplow()
        result = sp.step(1e-9, 500e3)
        required_keys = {
            "z_sheath", "v_sheath", "r_shock", "vr_shock",
            "L_plasma", "dL_dt", "swept_mass",
            "F_magnetic", "F_pressure", "phase",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )


# =====================================================================
# SnowplowConfig Pydantic validation
# =====================================================================

class TestSnowplowConfigValidation:
    """Test SnowplowConfig Pydantic validators."""

    def test_snowplow_config_defaults(self) -> None:
        """Default SnowplowConfig values match Lee & Saw 2014."""
        cfg = SnowplowConfig()
        assert cfg.enabled is True
        assert cfg.mass_fraction == pytest.approx(0.3)
        assert cfg.fill_pressure_Pa == pytest.approx(400.0)
        assert cfg.anode_length == pytest.approx(0.16)
        assert cfg.current_fraction == pytest.approx(0.7)
        assert cfg.radial_mass_fraction is None

    def test_snowplow_config_custom(self) -> None:
        """Custom SnowplowConfig values are properly stored."""
        cfg = SnowplowConfig(
            enabled=True,
            mass_fraction=0.5,
            fill_pressure_Pa=300.0,
            anode_length=0.10,
            current_fraction=0.8,
            radial_mass_fraction=0.2,
        )
        assert cfg.mass_fraction == pytest.approx(0.5)
        assert cfg.fill_pressure_Pa == pytest.approx(300.0)
        assert cfg.anode_length == pytest.approx(0.10)
        assert cfg.current_fraction == pytest.approx(0.8)
        assert cfg.radial_mass_fraction == pytest.approx(0.2)

    def test_snowplow_config_invalid_mass_fraction(self) -> None:
        """mass_fraction must be in (0, 1]."""
        with pytest.raises(ValueError):
            SnowplowConfig(mass_fraction=0.0)
        with pytest.raises(ValueError):
            SnowplowConfig(mass_fraction=1.5)

    def test_snowplow_config_invalid_current_fraction(self) -> None:
        """current_fraction must be in (0, 1]."""
        with pytest.raises(ValueError):
            SnowplowConfig(current_fraction=0.0)
        with pytest.raises(ValueError):
            SnowplowConfig(current_fraction=1.5)
