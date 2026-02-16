"""Phase T tests: Radial phase physics for DPF snowplow model.

Tests for the Lee model radial inward shock (Phase 3) and pinch (Phase 4):
- Radial J x B force formula
- Radial inductance and dL/dt formulas
- Radial mass accumulation (slug model)
- Velocity-Verlet convergence order
- Radial shock direction, inductance monotonicity
- Pinch timing, phase transitions, parameter sensitivity
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import mu_0, pi
from dpf.fluid.snowplow import SnowplowModel

# ---------------------------------------------------------------------------
# PF-1000 default parameters (Lee & Saw 2014)
# ---------------------------------------------------------------------------
PF1000_PARAMS = {
    "anode_radius": 0.0575,        # 57.5 mm
    "cathode_radius": 0.08,        # 80 mm
    "fill_density": 4e-4,          # kg/m^3
    "anode_length": 0.16,          # 160 mm
    "mass_fraction": 0.3,
    "fill_pressure_Pa": 400.0,
    "current_fraction": 0.7,
}


def make_radial_snowplow(**kwargs) -> SnowplowModel:
    """Create a SnowplowModel already in radial phase.

    Fast-forwards past axial rundown by manually setting internal state
    to the start of the radial phase.  The shock is placed at 95% of the
    cathode radius so that M_slug > 0 (at r=b the swept mass is exactly
    zero, which causes infinite acceleration from the mass floor clamp).
    """
    params = {**PF1000_PARAMS, **kwargs}
    sp = SnowplowModel(
        anode_radius=params["anode_radius"],
        cathode_radius=params["cathode_radius"],
        fill_density=params["fill_density"],
        anode_length=params["anode_length"],
        mass_fraction=params["mass_fraction"],
        fill_pressure_Pa=params["fill_pressure_Pa"],
        current_fraction=params["current_fraction"],
        radial_mass_fraction=params.get("radial_mass_fraction"),
    )
    # Fast-forward to radial phase
    sp.z = sp.L_anode
    sp.v = 1e4  # typical axial end velocity [m/s]
    sp._rundown_complete = True
    sp._L_axial_frozen = sp.L_coeff * sp.L_anode
    sp.phase = "radial"
    # Start at 95% of cathode radius so M_slug > 0
    sp.r_shock = 0.95 * sp.b
    sp.vr = 0.0
    return sp


# ===================================================================
# Test 1: Radial force formula
# ===================================================================
class TestRadialForceFormula:
    """F_rad = (mu_0 / 4pi) * (f_c * I)^2 * z_f / r_s."""

    def test_radial_force_at_initial_radius(self) -> None:
        """Check F_magnetic matches analytic formula at the initial shock radius."""
        sp = make_radial_snowplow()
        r_init = sp.r_shock  # 0.95 * b
        I_current = 1.0e6  # 1 MA
        dt = 1e-10  # tiny step to keep shock ~stationary

        result = sp.step(dt, current=I_current)

        z_f = sp.L_anode
        expected_F = (mu_0 / (4.0 * pi)) * (sp.f_c * I_current) ** 2 * z_f / r_init

        # Force at the first step uses r_s at the start of the step
        assert result["F_magnetic"] == pytest.approx(expected_F, rel=1e-6)
        assert result["phase"] == "radial"

    def test_radial_force_at_intermediate_radius(self) -> None:
        """Check force formula at an intermediate shock radius."""
        sp = make_radial_snowplow()
        # Set shock to half-way between anode and cathode
        r_mid = 0.5 * (sp.a + sp.b)
        sp.r_shock = r_mid
        sp.vr = -5e3  # already moving inward

        I_current = 5e5
        dt = 1e-10  # very tiny to minimize integration effect

        result = sp.step(dt, current=I_current)

        z_f = sp.L_anode
        expected_F = (mu_0 / (4.0 * pi)) * (sp.f_c * I_current) ** 2 * z_f / r_mid
        assert result["F_magnetic"] == pytest.approx(expected_F, rel=1e-4)

    def test_radial_force_increases_as_shock_converges(self) -> None:
        """Force should increase as 1/r_s when shock moves inward."""
        I_current = 1e6
        dt = 1e-10

        forces = []
        # All radii must be between r_pinch_min and b
        radii = [0.07, 0.05, 0.03]
        for r in radii:
            sp_i = make_radial_snowplow()
            sp_i.r_shock = r
            sp_i.vr = -1e3
            result = sp_i.step(dt, current=I_current)
            forces.append(result["F_magnetic"])

        # Force should monotonically increase as radius decreases
        assert forces[1] > forces[0]
        assert forces[2] > forces[1]

        # Check 1/r scaling between first and second radii
        ratio_r = radii[0] / radii[1]
        ratio_F = forces[1] / forces[0]
        assert ratio_F == pytest.approx(ratio_r, rel=0.01)


# ===================================================================
# Test 2: Radial inductance formula
# ===================================================================
class TestRadialInductanceFormula:
    """L_plasma = L_axial + (mu_0 / 2pi) * z_f * ln(b / r_s)."""

    def test_inductance_at_cathode_radius(self) -> None:
        """At r_shock = b: L_radial = 0 so L_plasma = L_axial."""
        sp = make_radial_snowplow()
        # Manually set r_shock = b for this test
        sp.r_shock = sp.b
        # At r_shock = b, ln(b/b) = 0 so radial contribution is zero
        expected_L = sp._L_axial_frozen
        assert sp.plasma_inductance == pytest.approx(expected_L, rel=1e-10)

    def test_inductance_at_intermediate_radius(self) -> None:
        """Check L_plasma formula at an intermediate r_shock."""
        sp = make_radial_snowplow()
        r_test = 0.04  # 40 mm, between anode (57.5mm) and axis
        sp.r_shock = r_test

        L_axial = sp._L_axial_frozen
        L_radial_expected = (mu_0 / (2.0 * pi)) * sp.L_anode * np.log(sp.b / r_test)
        expected_total = L_axial + L_radial_expected

        assert sp.plasma_inductance == pytest.approx(expected_total, rel=1e-10)

    def test_inductance_increases_as_shock_converges(self) -> None:
        """L_plasma should increase as r_shock decreases (ln(b/r) grows)."""
        sp = make_radial_snowplow()
        radii = [sp.b, 0.06, 0.04, 0.02, 0.01]
        inductances = []
        for r in radii:
            sp.r_shock = r
            inductances.append(sp.plasma_inductance)

        for i in range(1, len(inductances)):
            assert inductances[i] > inductances[i - 1]

    def test_inductance_clamped_at_pinch_min(self) -> None:
        """L_plasma should use r_pinch_min as lower bound (avoid log divergence)."""
        sp = make_radial_snowplow()
        r_tiny = 1e-6  # Much smaller than r_pinch_min
        sp.r_shock = r_tiny
        r_eff = sp.r_pinch_min  # Model clamps to this
        L_expected = sp._L_axial_frozen + (mu_0 / (2.0 * pi)) * sp.L_anode * np.log(
            sp.b / r_eff
        )
        assert sp.plasma_inductance == pytest.approx(L_expected, rel=1e-10)


# ===================================================================
# Test 3: Radial dL/dt formula
# ===================================================================
class TestRadialDLDT:
    """dL/dt = -(mu_0 / 2pi) * z_f * vr / r_s."""

    def test_dL_dt_at_start(self) -> None:
        """At start of radial phase (vr=0), dL/dt should be ~0."""
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-10  # tiny step

        result = sp.step(dt, current=I_current)
        # After one tiny step, vr is small but negative
        # dL_dt should be small and non-negative (vr<=0 makes dL/dt>=0)
        assert result["dL_dt"] >= 0.0

    def test_dL_dt_formula_after_step(self) -> None:
        """After some radial compression, dL/dt should match formula."""
        sp = make_radial_snowplow()
        # Give it some inward velocity
        sp.r_shock = 0.06
        sp.vr = -1e4  # moving inward

        I_current = 1e6
        dt = 1e-10  # tiny step so state barely changes

        result = sp.step(dt, current=I_current)

        # dL/dt formula uses current vr and r_shock AFTER the step
        vr_after = sp.vr
        r_after = max(sp.r_shock, sp.r_pinch_min)
        z_f = sp.L_anode
        expected_dLdt = -(mu_0 / (2.0 * pi)) * z_f * vr_after / r_after

        assert result["dL_dt"] == pytest.approx(expected_dLdt, rel=1e-4)

    def test_dL_dt_positive_for_inward_motion(self) -> None:
        """dL/dt > 0 when vr < 0 (shock moving inward increases inductance)."""
        sp = make_radial_snowplow()
        sp.r_shock = 0.05
        sp.vr = -5e3

        result = sp.step(1e-10, current=1e6)
        assert result["dL_dt"] > 0.0


# ===================================================================
# Test 4: Radial mass accumulation
# ===================================================================
class TestRadialMassAccumulation:
    """M_slug = f_mr * rho_0 * pi * (b^2 - r_s^2) * z_f."""

    def test_zero_mass_at_cathode(self) -> None:
        """At r_shock = b, no mass has been swept radially."""
        sp = make_radial_snowplow()
        sp.r_shock = sp.b  # manually set to cathode for this test
        assert sp.radial_swept_mass == pytest.approx(0.0, abs=1e-20)

    def test_mass_formula_at_intermediate_radius(self) -> None:
        """Check radial mass formula at a known r_shock."""
        sp = make_radial_snowplow()
        r_test = 0.04
        sp.r_shock = r_test

        expected_M = sp.f_mr * sp.rho0 * pi * (sp.b**2 - r_test**2) * sp.L_anode
        assert sp.radial_swept_mass == pytest.approx(expected_M, rel=1e-10)

    def test_mass_increases_as_shock_converges(self) -> None:
        """Swept mass should increase as shock compresses inward."""
        sp = make_radial_snowplow()
        masses = []
        radii = [sp.b, 0.06, 0.04, 0.02]
        for r in radii:
            sp.r_shock = r
            masses.append(sp.radial_swept_mass)

        for i in range(1, len(masses)):
            assert masses[i] > masses[i - 1]

    def test_maximum_mass_at_axis(self) -> None:
        """At r_shock ~ 0, all annular mass within [0, b] is swept."""
        sp = make_radial_snowplow()
        sp.r_shock = 0.0  # hypothetical axis
        M_max = sp.f_mr * sp.rho0 * pi * sp.b**2 * sp.L_anode
        assert sp.radial_swept_mass == pytest.approx(M_max, rel=1e-10)


# ===================================================================
# Test 5: Velocity-Verlet convergence in radial phase
# ===================================================================
class TestRadialConvergence:
    """Velocity-Verlet integrator should show ~2nd order convergence."""

    def test_convergence_order(self) -> None:
        """Run with dt and dt/2, measure convergence toward dt/4 reference."""
        I_current = 8e5
        n_steps_coarse = 100
        dt_coarse = 1e-9

        # Coarse run
        sp1 = make_radial_snowplow()
        for _ in range(n_steps_coarse):
            sp1.step(dt_coarse, current=I_current)
        r_coarse = sp1.r_shock

        # Fine run (dt/2, 2x steps)
        sp2 = make_radial_snowplow()
        for _ in range(2 * n_steps_coarse):
            sp2.step(dt_coarse / 2.0, current=I_current)
        r_fine = sp2.r_shock

        # Reference run (dt/4, 4x steps)
        sp_ref = make_radial_snowplow()
        for _ in range(4 * n_steps_coarse):
            sp_ref.step(dt_coarse / 4.0, current=I_current)
        r_ref = sp_ref.r_shock

        err_coarse = abs(r_coarse - r_ref)
        err_fine = abs(r_fine - r_ref)

        # Velocity-Verlet is 2nd order: error ratio ~ 4 when dt halved
        # Be generous: accept order > 1.5
        if err_fine > 1e-15:
            order = np.log2(err_coarse / err_fine)
            assert order > 1.5, f"Expected ~2nd order, got {order:.2f}"


# ===================================================================
# Test 6: Radial shock always moves inward
# ===================================================================
class TestRadialShockDirection:
    """Radial shock velocity must be <= 0 (inward) throughout Phase 3."""

    def test_vr_always_non_positive(self) -> None:
        """Run through radial phase, check vr <= 0 at every step."""
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9

        for _ in range(5000):
            result = sp.step(dt, current=I_current)
            assert result["vr_shock"] <= 0.0, (
                f"Radial velocity became positive: vr={result['vr_shock']:.2e} "
                f"at r_shock={result['r_shock']:.4e}"
            )
            if sp.phase == "pinch":
                break

    def test_r_shock_monotonically_decreases(self) -> None:
        """Shock radius should be non-increasing throughout radial phase."""
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        prev_r = sp.r_shock

        for _ in range(5000):
            result = sp.step(dt, current=I_current)
            assert result["r_shock"] <= prev_r + 1e-15, (
                f"Shock radius increased: {result['r_shock']:.6e} > {prev_r:.6e}"
            )
            prev_r = result["r_shock"]
            if sp.phase == "pinch":
                break


# ===================================================================
# Test 7: L_plasma monotonically increases through radial phase
# ===================================================================
class TestRadialInductanceMonotonicity:
    """Plasma inductance should be non-decreasing through the radial phase."""

    def test_inductance_non_decreasing(self) -> None:
        """Track L_plasma through radial phase, verify non-decreasing."""
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        prev_L = sp.plasma_inductance

        for _ in range(5000):
            result = sp.step(dt, current=I_current)
            L_now = result["L_plasma"]
            assert L_now >= prev_L - 1e-15, (
                f"Inductance decreased: {L_now:.6e} < {prev_L:.6e}"
            )
            prev_L = L_now
            if sp.phase == "pinch":
                break

    def test_inductance_significantly_larger_after_pinch(self) -> None:
        """Total L at pinch should be much larger than L at radial start."""
        sp = make_radial_snowplow()
        L_start = sp.plasma_inductance
        I_current = 1e6
        dt = 1e-9

        for _ in range(50000):
            sp.step(dt, current=I_current)
            if sp.phase == "pinch":
                break

        L_end = sp.plasma_inductance
        # ln(b / r_pinch_min) >> ln(b / b)=0, so L should grow substantially
        assert L_end > L_start * 1.5, (
            f"L_end={L_end:.4e} not significantly larger than L_start={L_start:.4e}"
        )


# ===================================================================
# Test 8: Pinch timing order-of-magnitude
# ===================================================================
class TestPinchTiming:
    """Pinch should occur within ~1 microsecond of radial start for PF-1000."""

    def test_pinch_time_order_of_magnitude(self) -> None:
        """Radial compression for PF-1000 at ~1 MA should take ~0.1-10 us."""
        sp = make_radial_snowplow()
        I_current = 1e6  # 1 MA (typical PF-1000 peak current)
        dt = 1e-10  # 0.1 ns steps for resolution
        max_steps = 1_000_000  # 100 us upper limit

        n_taken = 0
        for _steps in range(max_steps):
            sp.step(dt, current=I_current)
            n_taken = _steps + 1
            if sp.phase == "pinch":
                break

        pinch_time = n_taken * dt
        assert sp.phase == "pinch", "Did not reach pinch within 100 us"
        # Pinch time should be on the order of 100 ns to 10 us
        assert pinch_time < 10e-6, f"Pinch too slow: {pinch_time:.2e} s"
        assert pinch_time > 10e-9, f"Pinch unrealistically fast: {pinch_time:.2e} s"

    def test_pinch_radius_at_minimum(self) -> None:
        """After pinch, r_shock should equal r_pinch_min."""
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9

        for _ in range(100_000):
            sp.step(dt, current=I_current)
            if sp.phase == "pinch":
                break

        assert sp.phase == "pinch"
        assert sp.r_shock == pytest.approx(sp.r_pinch_min, abs=1e-12)


# ===================================================================
# Test 9: Phase transitions
# ===================================================================
class TestPhaseTransitions:
    """Verify rundown -> radial -> pinch transition sequence."""

    def test_rundown_to_radial_transition(self) -> None:
        """Driving a fresh snowplow should first go through rundown then radial."""
        sp = SnowplowModel(**PF1000_PARAMS)
        I_current = 1e6
        dt = 1e-8  # 10 ns

        saw_rundown = False
        saw_radial = False
        prev_phase = "rundown"

        for _ in range(500_000):
            result = sp.step(dt, current=I_current)
            phase = result["phase"]

            if phase == "rundown":
                saw_rundown = True
            elif phase == "radial":
                if not saw_radial:
                    # Transition should come from rundown
                    assert prev_phase == "rundown" or prev_phase == "radial"
                saw_radial = True
            elif phase == "pinch":
                # Transition should come from radial
                assert saw_radial, "Jumped to pinch without radial phase"
                break

            prev_phase = phase

        assert saw_rundown, "Never saw rundown phase"
        assert saw_radial, "Never saw radial phase"

    @pytest.mark.slow
    def test_full_phase_sequence(self) -> None:
        """Drive through all three phases and verify ordering."""
        sp = SnowplowModel(**PF1000_PARAMS)
        I_current = 1e6
        dt = 5e-9

        phase_order = []
        for _ in range(1_000_000):
            result = sp.step(dt, current=I_current)
            phase = result["phase"]
            if not phase_order or phase_order[-1] != phase:
                phase_order.append(phase)
            if phase == "pinch":
                break

        assert phase_order == ["rundown", "radial", "pinch"]

    def test_rundown_complete_flag(self) -> None:
        """After transition to radial, rundown_complete should be True."""
        sp = make_radial_snowplow()
        assert sp.rundown_complete is True
        assert sp.phase == "radial"

    def test_pinch_complete_flag(self) -> None:
        """After reaching pinch, pinch_complete and is_active flags are correct."""
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9

        for _ in range(100_000):
            sp.step(dt, current=I_current)
            if sp.phase == "pinch":
                break

        assert sp.pinch_complete is True
        assert sp.is_active is False

    def test_frozen_state_after_pinch(self) -> None:
        """After pinch, further steps return frozen state with zero forces."""
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9

        for _ in range(100_000):
            sp.step(dt, current=I_current)
            if sp.phase == "pinch":
                break

        # Now take more steps -- should return frozen state
        result1 = sp.step(dt, current=I_current)
        result2 = sp.step(dt, current=I_current)

        assert result1["phase"] == "pinch"
        assert result1["F_magnetic"] == 0.0
        assert result1["F_pressure"] == 0.0
        assert result1["dL_dt"] == 0.0
        # State should not change
        assert result1["r_shock"] == result2["r_shock"]
        assert result1["L_plasma"] == result2["L_plasma"]


# ===================================================================
# Test 10: Radial mass fraction parameter
# ===================================================================
class TestRadialMassFraction:
    """f_mr can be set independently of f_m."""

    def test_default_f_mr_equals_f_m(self) -> None:
        """When radial_mass_fraction not specified, f_mr defaults to f_m."""
        sp = SnowplowModel(**PF1000_PARAMS)
        assert sp.f_mr == sp.f_m

    def test_custom_f_mr(self) -> None:
        """Setting radial_mass_fraction overrides the default."""
        sp = make_radial_snowplow(radial_mass_fraction=0.1)
        assert sp.f_mr == 0.1
        assert sp.f_m == 0.3  # axial still at default

    def test_f_mr_affects_radial_mass(self) -> None:
        """Lower f_mr should give less radial swept mass."""
        sp_low = make_radial_snowplow(radial_mass_fraction=0.1)
        sp_high = make_radial_snowplow(radial_mass_fraction=0.5)

        # Set both to same r_shock
        r_test = 0.04
        sp_low.r_shock = r_test
        sp_high.r_shock = r_test

        assert sp_low.radial_swept_mass < sp_high.radial_swept_mass
        ratio = sp_high.radial_swept_mass / sp_low.radial_swept_mass
        assert ratio == pytest.approx(0.5 / 0.1, rel=1e-10)


# ===================================================================
# Test 11: Current fraction effect on radial dynamics
# ===================================================================
class TestCurrentFractionEffect:
    """Higher f_c -> faster radial implosion (lower pinch time)."""

    def test_higher_fc_faster_pinch(self) -> None:
        """Higher current fraction should lead to faster pinch."""
        I_current = 1e6
        dt = 1e-10  # 0.1 ns for time resolution

        times = {}
        for f_c in [0.5, 0.8]:
            sp = make_radial_snowplow(current_fraction=f_c)
            n_taken = 0
            for _step_n in range(1_000_000):
                sp.step(dt, current=I_current)
                n_taken = _step_n + 1
                if sp.phase == "pinch":
                    break
            times[f_c] = n_taken * dt
            assert sp.phase == "pinch", f"f_c={f_c} did not reach pinch"

        # Higher f_c should pinch faster
        assert times[0.8] < times[0.5], (
            f"f_c=0.8 ({times[0.8]:.2e} s) should pinch faster than "
            f"f_c=0.5 ({times[0.5]:.2e} s)"
        )

    def test_fc_affects_radial_force(self) -> None:
        """F_rad scales as f_c^2."""
        I_current = 1e6
        dt = 1e-10

        sp_low = make_radial_snowplow(current_fraction=0.5)
        sp_high = make_radial_snowplow(current_fraction=1.0)

        r_low = sp_low.step(dt, current=I_current)
        r_high = sp_high.step(dt, current=I_current)

        ratio = r_high["F_magnetic"] / r_low["F_magnetic"]
        expected_ratio = (1.0 / 0.5) ** 2
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)


# ===================================================================
# Test 12: Zero current during radial phase
# ===================================================================
class TestZeroCurrentRadial:
    """With zero current, no radial force: shock should stall."""

    def test_shock_stalls_with_zero_current(self) -> None:
        """Zero current produces zero force, shock position barely changes."""
        sp = make_radial_snowplow()
        sp.r_shock = 0.06  # slightly inside cathode
        sp.vr = 0.0  # stationary
        r_initial = sp.r_shock

        dt = 1e-9
        for _ in range(1000):
            sp.step(dt, current=0.0)

        # With zero current and zero initial velocity, shock should not move
        assert sp.r_shock == pytest.approx(r_initial, abs=1e-12)

    def test_zero_force_with_zero_current(self) -> None:
        """F_magnetic = 0 when I = 0."""
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=0.0)
        assert result["F_magnetic"] == pytest.approx(0.0, abs=1e-20)

    def test_moving_shock_decelerates_with_zero_current(self) -> None:
        """A shock already moving inward should decelerate (mass pickup drag)."""
        sp = make_radial_snowplow()
        sp.r_shock = 0.05
        sp.vr = -1e4  # already moving inward fast

        dt = 1e-9
        initial_vr = sp.vr

        for _ in range(100):
            sp.step(dt, current=0.0)

        # With zero driving force, the mass-pickup drag slows it down
        # (momentum conservation: m*v = const, but m increases => v decreases)
        # vr is negative, so |vr| should decrease => vr should become less negative
        assert sp.vr > initial_vr, (
            f"Shock should decelerate: vr={sp.vr:.2e} not > initial {initial_vr:.2e}"
        )


# ===================================================================
# Additional tests for robustness
# ===================================================================
class TestRadialEdgeCases:
    """Edge cases and robustness checks for the radial phase."""

    def test_very_small_current(self) -> None:
        """Very small current should still produce valid (small) force."""
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=1.0)  # 1 A
        assert result["F_magnetic"] > 0.0
        assert np.isfinite(result["F_magnetic"])
        assert result["F_magnetic"] < 1e-6  # should be tiny

    def test_very_large_current(self) -> None:
        """Very large current should not produce NaN or Inf."""
        sp = make_radial_snowplow()
        result = sp.step(1e-12, current=1e8)  # 100 MA, tiny dt
        assert np.isfinite(result["F_magnetic"])
        assert np.isfinite(result["r_shock"])
        assert np.isfinite(result["vr_shock"])

    def test_pressure_not_used_in_radial(self) -> None:
        """Radial phase should return F_pressure=0 (no back-pressure)."""
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=1e6)
        assert result["F_pressure"] == 0.0

    def test_axial_position_frozen_in_radial(self) -> None:
        """z_sheath should remain at L_anode throughout radial phase."""
        sp = make_radial_snowplow()
        dt = 1e-9
        for _ in range(100):
            result = sp.step(dt, current=1e6)
            assert result["z_sheath"] == pytest.approx(sp.L_anode, rel=1e-10)

    def test_step_returns_all_keys(self) -> None:
        """Verify step() returns all expected keys."""
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=1e6)
        expected_keys = {
            "z_sheath", "v_sheath", "r_shock", "vr_shock",
            "L_plasma", "dL_dt", "swept_mass", "F_magnetic",
            "F_pressure", "phase",
        }
        assert set(result.keys()) == expected_keys


class TestRadialProperties:
    """Test property accessors during radial phase."""

    def test_shock_radius_property(self) -> None:
        """shock_radius property matches internal state."""
        sp = make_radial_snowplow()
        sp.r_shock = 0.03
        assert sp.shock_radius == 0.03

    def test_sheath_position_property(self) -> None:
        """sheath_position fixed at L_anode in radial phase."""
        sp = make_radial_snowplow()
        assert sp.sheath_position == sp.L_anode

    def test_is_active_in_radial(self) -> None:
        """Snowplow is active during radial phase."""
        sp = make_radial_snowplow()
        assert sp.is_active is True
        assert sp.pinch_complete is False
