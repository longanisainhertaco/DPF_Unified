"""Phase S circuit tests: crowbar model and 2nd-order dL/dt.

Tests cover:
- Crowbar trigger modes (voltage-zero crossing, fixed time)
- Energy conservation with crowbar active
- Post-crowbar L-R decay behavior
- Crowbar prevents voltage reversal
- 2nd-order dL/dt accuracy vs 1st-order on known L(t) profile
- PF-1000 quarter-period matches analytical
- Crowbar state reported in circuit state
"""

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState

# ---------------------------------------------------------------------------
# PF-1000 circuit parameters (from presets.py)
# ---------------------------------------------------------------------------
PF1000_C = 1.332e-3   # F
PF1000_V0 = 27e3      # V
PF1000_L0 = 33.5e-9   # H
PF1000_R0 = 2.3e-3    # Ohm


def _make_solver(**overrides) -> RLCSolver:
    """Create a solver with PF-1000 defaults, overridable."""
    kw = dict(
        C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0,
        anode_radius=0.0575, cathode_radius=0.08,
    )
    kw.update(overrides)
    return RLCSolver(**kw)


def _coupling(Lp: float = 0.0, dL_dt: float = 0.0, R_plasma: float = 0.0) -> CouplingState:
    """Convenience constructor for CouplingState."""
    return CouplingState(Lp=Lp, dL_dt=dL_dt, R_plasma=R_plasma)


# ===================================================================
# Crowbar trigger tests
# ===================================================================

class TestCrowbarTrigger:
    """Tests for crowbar switch triggering conditions."""

    def test_crowbar_fires_at_voltage_zero(self):
        """Crowbar fires when V_cap crosses zero (positive to non-positive).

        For a damped RLC: V(t) = V0*exp(-alpha*t)*[cos(wd*t) + (alpha/wd)*sin(wd*t)]
        V=0 when wd*t = pi - arctan(wd/alpha).
        PF-1000: alpha=R/(2L)=34328, wd=sqrt(1/(LC)-alpha^2)=145756 rad/s
        => t_zero = 12.37 us.
        """
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9  # 1 ns steps

        max_steps = int(20e-6 / dt)  # 20 us max
        fired = False
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                fired = True
                break

        assert fired, "Crowbar should have fired when V_cap crossed zero"
        assert solver.state.crowbar_fire_time > 0.0

        # Analytical zero-crossing for damped RLC
        alpha = PF1000_R0 / (2 * PF1000_L0)
        omega0_sq = 1.0 / (PF1000_C * PF1000_L0)
        omega_d = np.sqrt(omega0_sq - alpha**2)
        t_zero = (np.pi - np.arctan(omega_d / alpha)) / omega_d
        assert solver.state.crowbar_fire_time == pytest.approx(t_zero, rel=0.01)

    def test_crowbar_fires_at_fixed_time(self):
        """Crowbar fires at configured fixed time."""
        fire_time = 4e-6  # 4 us (PF-1000 typical crowbar time)
        solver = _make_solver(
            crowbar_enabled=True, crowbar_mode="fixed_time", crowbar_time=fire_time,
        )
        coupling = _coupling()
        dt = 1e-9

        max_steps = int(10e-6 / dt)
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                break

        assert solver.state.crowbar_fired
        # Should fire at or just after the configured time
        assert solver.state.crowbar_fire_time == pytest.approx(fire_time, abs=2 * dt)

    def test_crowbar_disabled_no_fire(self):
        """When crowbar_enabled=False, crowbar never fires."""
        solver = _make_solver(crowbar_enabled=False)
        coupling = _coupling()
        dt = 1e-9

        # Run well past where voltage would cross zero
        T_quarter = (np.pi / 2) * np.sqrt(PF1000_C * PF1000_L0)
        steps = int(2 * T_quarter / dt)
        for _ in range(steps):
            coupling = solver.step(coupling, 0.0, dt)

        assert not solver.state.crowbar_fired
        assert solver.state.crowbar_fire_time < 0

    def test_crowbar_fires_only_once(self):
        """Crowbar should fire at most once (latch behavior)."""
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9

        fire_times = []
        max_steps = int(30e-6 / dt)
        for _ in range(max_steps):
            was_fired = solver.state.crowbar_fired
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired and not was_fired:
                fire_times.append(solver.state.crowbar_fire_time)

        assert len(fire_times) == 1, "Crowbar should fire exactly once"


# ===================================================================
# Post-crowbar behavior tests
# ===================================================================

class TestPostCrowbar:
    """Tests for circuit behavior after crowbar fires."""

    def test_crowbar_prevents_voltage_reversal(self):
        """After crowbar fires, V_cap should remain at 0 (no reversal)."""
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9

        max_steps = int(30e-6 / dt)
        found_crowbar = False
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                found_crowbar = True
            if found_crowbar:
                # Voltage should be frozen at 0 after crowbar
                assert solver.state.voltage == pytest.approx(0.0, abs=1e-10), (
                    f"V_cap should be 0 after crowbar, got {solver.state.voltage}"
                )

        assert found_crowbar, "Crowbar should have fired"

    def test_post_crowbar_lr_decay(self):
        """After crowbar, current decays approximately as exp(-R*t/L).

        Uses fixed_time crowbar at 5 us to guarantee firing, then checks
        exponential decay over several L/R time constants.
        """
        R0 = PF1000_R0
        crowbar_R = 0.005
        solver = _make_solver(
            crowbar_enabled=True, crowbar_mode="fixed_time",
            crowbar_time=5e-6, crowbar_resistance=crowbar_R,
        )
        R_total = R0 + crowbar_R
        coupling = _coupling()
        dt = 1e-9

        # Run until crowbar fires
        max_steps = int(10e-6 / dt)
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if solver.state.crowbar_fired:
                break

        assert solver.state.crowbar_fired
        I_crowbar = solver.state.current
        t_crowbar = solver.state.time

        # Now advance for 5 L/R time constants
        # tau = L0 / R_total = 33.5e-9 / 7.3e-3 ~ 4.59 ns
        tau = PF1000_L0 / R_total
        post_steps = int(5 * tau / dt)
        # Need at least a few steps for meaningful test
        post_steps = max(post_steps, 50)
        for _ in range(post_steps):
            coupling = solver.step(coupling, 0.0, dt)

        elapsed = solver.state.time - t_crowbar
        expected_ratio = np.exp(-R_total * elapsed / PF1000_L0)
        actual_ratio = solver.state.current / I_crowbar

        # Allow 15% relative tolerance due to discrete-time integration
        # and short time constant relative to dt
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.15), (
            f"Post-crowbar decay: expected ratio {expected_ratio:.6f}, got {actual_ratio:.6f}"
        )

    def test_crowbar_energy_conservation(self):
        """Total energy (cap + ind + resistive) is conserved with crowbar."""
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="voltage_zero")
        coupling = _coupling()
        dt = 1e-9

        E_initial = solver.total_energy()
        assert E_initial == pytest.approx(0.5 * PF1000_C * PF1000_V0**2, rel=1e-10)

        # Run well past crowbar
        max_steps = int(25e-6 / dt)
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)

        assert solver.state.crowbar_fired
        E_final = solver.total_energy()
        # Energy should be conserved to within 1% (implicit midpoint is 2nd order)
        assert E_final == pytest.approx(E_initial, rel=0.01), (
            f"Energy not conserved: E_initial={E_initial:.2f}, E_final={E_final:.2f}"
        )


# ===================================================================
# Crowbar state tracking
# ===================================================================

class TestCrowbarState:
    """Tests for crowbar state reporting in CircuitState."""

    def test_initial_crowbar_state(self):
        """Initially, crowbar_fired is False and fire_time is -1."""
        solver = _make_solver(crowbar_enabled=True)
        assert solver.state.crowbar_fired is False
        assert solver.state.crowbar_fire_time == -1.0

    def test_crowbar_state_after_fire(self):
        """After crowbar fires, state reflects fired=True and fire_time > 0."""
        solver = _make_solver(crowbar_enabled=True, crowbar_mode="fixed_time", crowbar_time=1e-6)
        coupling = _coupling()
        dt = 1e-9

        steps = int(2e-6 / dt)
        for _ in range(steps):
            coupling = solver.step(coupling, 0.0, dt)

        assert solver.state.crowbar_fired is True
        assert solver.state.crowbar_fire_time > 0
        assert solver.state.crowbar_fire_time == pytest.approx(1e-6, abs=2 * dt)


# ===================================================================
# 2nd-order dL/dt tests
# ===================================================================

class TestDLdtSecondOrder:
    """Tests for the 2nd-order central difference dL/dt computation."""

    def test_dLdt_zero_history(self):
        """With no history, dL/dt should be 0."""
        solver = _make_solver()
        assert solver.compute_dLp_dt(1e-9) == 0.0

    def test_dLdt_first_order_fallback(self):
        """With 1 history point, uses 1st-order backward difference."""
        solver = _make_solver()
        # Manually add one history point
        solver._Lp_history.append((0.0, 1e-9))
        solver.state.time = 1e-9
        # dL/dt = (2e-9 - 1e-9) / 1e-9 = 1.0 H/s
        assert solver.compute_dLp_dt(2e-9) == pytest.approx(1.0, rel=1e-10)

    def test_dLdt_second_order_bdf2(self):
        """With 2+ history points, uses BDF2: (3*L_n - 4*L_{n-1} + L_{n-2}) / (2*dt)."""
        solver = _make_solver()
        dt = 1e-9
        # History: t=0: L=0, t=dt: L=1e-9
        solver._Lp_history.append((0.0, 0.0))
        solver._Lp_history.append((dt, 1e-9))
        solver.state.time = 2 * dt
        L_now = 4e-9
        # BDF2: (3*4e-9 - 4*1e-9 + 0) / (2*1e-9) = (12e-9 - 4e-9) / 2e-9 = 4.0 H/s
        result = solver.compute_dLp_dt(L_now)
        assert result == pytest.approx(4.0, rel=1e-10)

    def test_dLdt_accuracy_quadratic(self):
        """BDF2 is exact for quadratic L(t) = L0 + alpha*t^2.

        For L(t) = L0 + alpha*t^2, dL/dt = 2*alpha*t (exact).
        BDF2: (3*L_n - 4*L_{n-1} + L_{n-2}) / (2*dt)
          At t_n = 2*dt:
          = (3*(L0+4*alpha*dt^2) - 4*(L0+alpha*dt^2) + (L0)) / (2*dt)
          = (3*L0 + 12*alpha*dt^2 - 4*L0 - 4*alpha*dt^2 + L0) / (2*dt)
          = 8*alpha*dt^2 / (2*dt) = 4*alpha*dt = 2*alpha*(2*dt) (exact!)
        BDF2 is exact for polynomials up to degree 2.
        """
        alpha = 1e6  # H/s^2
        L0 = 1e-9    # H
        dt = 1e-9     # s

        solver = _make_solver()

        # Simulate 3 time steps: t=0, dt, 2*dt
        t0, t1, t2 = 0.0, dt, 2 * dt
        L_at_t0 = L0 + alpha * t0**2
        L_at_t1 = L0 + alpha * t1**2
        L_at_t2 = L0 + alpha * t2**2

        solver._Lp_history.append((t0, L_at_t0))
        solver._Lp_history.append((t1, L_at_t1))
        solver.state.time = t2

        computed = solver.compute_dLp_dt(L_at_t2)
        exact = 2 * alpha * t2  # dL/dt = 2*alpha*t at t2

        assert computed == pytest.approx(exact, rel=1e-10), (
            f"BDF2 should be exact for quadratic L(t): computed={computed}, exact={exact}"
        )

    def test_dLdt_vs_first_order_on_cubic(self):
        """BDF2 is more accurate than 1st-order on cubic L(t).

        For L(t) = beta*t^3, dL/dt = 3*beta*t^2.
        1st-order: (L(t) - L(t-dt))/dt has O(dt) error.
        BDF2: (3*L_n - 4*L_{n-1} + L_{n-2})/(2*dt) has O(dt^2) error.
        """
        beta = 1e12   # H/s^3
        dt = 1e-9

        t0, t1, t2 = 0.0, dt, 2 * dt
        L = lambda t: beta * t**3  # noqa: E731
        exact_dLdt = 3 * beta * t2**2

        # 1st-order: (L(t2) - L(t1)) / dt
        first_order = (L(t2) - L(t1)) / dt
        err_1st = abs(first_order - exact_dLdt) / abs(exact_dLdt)

        # BDF2
        solver = _make_solver()
        solver._Lp_history.append((t0, L(t0)))
        solver._Lp_history.append((t1, L(t1)))
        solver.state.time = t2
        second_order = solver.compute_dLp_dt(L(t2))
        err_2nd = abs(second_order - exact_dLdt) / abs(exact_dLdt)

        # BDF2 should be more accurate
        assert err_2nd < err_1st, (
            f"BDF2 error ({err_2nd:.4e}) should be less than "
            f"1st-order error ({err_1st:.4e})"
        )

    def test_dLdt_history_populated_during_stepping(self):
        """Verify that step() populates the Lp history for dL/dt."""
        solver = _make_solver()
        dt = 1e-9

        # Step 3 times with known Lp
        for i in range(3):
            coupling = _coupling(Lp=float(i + 1) * 1e-9)
            solver.step(coupling, 0.0, dt)

        # History should have 3 entries
        assert len(solver._Lp_history) == 3
        # Last entry should have Lp = 3e-9
        assert solver._Lp_history[-1][1] == pytest.approx(3e-9, rel=1e-10)


# ===================================================================
# PF-1000 analytical benchmarks
# ===================================================================

class TestPF1000Analytical:
    """Analytical benchmarks using PF-1000 parameters."""

    def test_pf1000_quarter_period(self):
        """T/4 for ideal LC matches analytical: (pi/2) * sqrt(LC).

        PF-1000: C = 1.332 mF, L0 = 33.5 nH
        T/4 = (pi/2) * sqrt(1.332e-3 * 33.5e-9) = 10.49 us
        """
        solver = _make_solver(R0=0.0)  # Zero resistance for ideal LC
        coupling = _coupling()
        dt = 1e-9

        T_quarter_analytical = (np.pi / 2) * np.sqrt(PF1000_C * PF1000_L0)
        # T_quarter_analytical ~ 10.49e-6 s

        # Run until current reaches peak (= quarter period for ideal LC)
        max_steps = int(20e-6 / dt)
        I_max = 0.0
        t_max = 0.0
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if abs(solver.state.current) > I_max:
                I_max = abs(solver.state.current)
                t_max = solver.state.time

        # Peak current time should match T/4 within 1%
        assert t_max == pytest.approx(T_quarter_analytical, rel=0.01), (
            f"Peak current at t={t_max*1e6:.2f} us, expected {T_quarter_analytical*1e6:.2f} us"
        )

    def test_pf1000_peak_current(self):
        """Peak current for ideal LC: I_peak = V0 * sqrt(C/L0).

        PF-1000: I_peak = 27000 * sqrt(1.332e-3 / 33.5e-9) = 5.38 MA
        """
        solver = _make_solver(R0=0.0)
        coupling = _coupling()
        dt = 1e-9

        I_peak_analytical = PF1000_V0 * np.sqrt(PF1000_C / PF1000_L0)

        max_steps = int(15e-6 / dt)
        I_max = 0.0
        for _ in range(max_steps):
            coupling = solver.step(coupling, 0.0, dt)
            if abs(solver.state.current) > I_max:
                I_max = abs(solver.state.current)

        assert I_max == pytest.approx(I_peak_analytical, rel=0.01), (
            f"Peak current: {I_max:.0f} A, expected {I_peak_analytical:.0f} A"
        )

    def test_pf1000_energy_conservation_no_crowbar(self):
        """Energy conservation for full LC oscillation (no crowbar, low R)."""
        solver = _make_solver()  # Low R0 = 2.3 mOhm
        coupling = _coupling()
        dt = 1e-9

        E_initial = solver.total_energy()
        steps = int(15e-6 / dt)
        for _ in range(steps):
            coupling = solver.step(coupling, 0.0, dt)

        E_final = solver.total_energy()
        assert E_final == pytest.approx(E_initial, rel=0.005), (
            f"Energy: initial={E_initial:.2f}, final={E_final:.2f}"
        )


# ===================================================================
# Config integration tests
# ===================================================================

class TestCircuitConfig:
    """Test CircuitConfig crowbar fields."""

    def test_config_crowbar_defaults(self):
        """Default CircuitConfig has crowbar disabled."""
        from dpf.config import CircuitConfig

        cfg = CircuitConfig(
            C=1e-6, V0=1e3, L0=1e-7, anode_radius=0.005, cathode_radius=0.01,
        )
        assert cfg.crowbar_enabled is False
        assert cfg.crowbar_mode == "voltage_zero"
        assert cfg.crowbar_time == 0.0
        assert cfg.crowbar_resistance == 0.0

    def test_config_crowbar_enabled(self):
        """CircuitConfig accepts crowbar parameters."""
        from dpf.config import CircuitConfig

        cfg = CircuitConfig(
            C=1e-6, V0=1e3, L0=1e-7, anode_radius=0.005, cathode_radius=0.01,
            crowbar_enabled=True, crowbar_mode="fixed_time",
            crowbar_time=4e-6, crowbar_resistance=0.001,
        )
        assert cfg.crowbar_enabled is True
        assert cfg.crowbar_mode == "fixed_time"
        assert cfg.crowbar_time == 4e-6
        assert cfg.crowbar_resistance == 0.001

    def test_config_invalid_crowbar_mode(self):
        """Invalid crowbar_mode raises ValueError."""
        from dpf.config import CircuitConfig

        with pytest.raises(ValueError, match="crowbar_mode"):
            CircuitConfig(
                C=1e-6, V0=1e3, L0=1e-7, anode_radius=0.005, cathode_radius=0.01,
                crowbar_mode="bogus",
            )
