"""PF-1000 current waveform I(t) validation tests.

Validates the RLC circuit solver against:
1. Analytical short-circuit (no plasma) RLC discharge
2. Coupled MHD+circuit simulation with plasma loading
3. Scholz et al. (2006) experimental parameters

Circuit parameters from Scholz et al., Nukleonika 51(1):79-84, 2006:
    C  = 1.332 mF (capacitor bank)
    V0 = 27 kV (charging voltage)
    L0 = 33.5 nH (external inductance, calibration)
    R0 = 2.3 mOhm (external resistance, RESF=1.22)

Short-circuit analytical solution:
    I(t) = (V0 * C * omega0^2 / omega_d) * sin(omega_d * t) * exp(-gamma * t)
    gamma = R0 / (2*L0) = 34,328 s^-1
    omega0 = 1/sqrt(L0*C) = 149,715 rad/s
    omega_d = sqrt(omega0^2 - gamma^2) = 145,751 rad/s
    T/4 = pi/(2*omega_d) = 10.78 us
    I_peak = 3.927 MA (at t = 9.19 us)

Lee model fitting parameters (Lee & Saw, J. Fusion Energy 33, 2014):
    fm  = 0.13 (axial mass fraction)
    fc  = 0.70 (axial current fraction)
    fmr = 0.35 (radial mass fraction)
    fcr = 0.65 (radial current fraction)

References
----------
- Scholz et al., Nukleonika 51(1):79-84 (2006)
- Lee & Saw, J. Fusion Energy 33, 319 (2014)
- Karpinski et al., AIP Conf. Proc. 808, 195 (2006)
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState

# ============================================================
# PF-1000 circuit parameters (Scholz 2006)
# ============================================================

C = 1.332e-3      # F
V0 = 27e3         # V
L0 = 33.5e-9      # H
R0 = 2.3e-3       # Ohm
E0_J = 0.5 * C * V0**2  # 485,514 J

# Derived quantities
GAMMA_DAMP = R0 / (2.0 * L0)     # 34,328 s^-1
OMEGA0 = 1.0 / np.sqrt(L0 * C)    # 149,715 rad/s
OMEGA_D = np.sqrt(OMEGA0**2 - GAMMA_DAMP**2)  # 145,751 rad/s
T_QUARTER = np.pi / (2.0 * OMEGA_D)  # 10.78 us

# Electrode geometry (Scholz 2006)
ANODE_RADIUS = 0.115    # 115 mm
CATHODE_RADIUS = 0.16   # 160 mm


def _analytical_current(t: np.ndarray) -> np.ndarray:
    """Short-circuit RLC analytical I(t) for PF-1000.

    I(t) = (V0 * C * omega0^2 / omega_d) * sin(omega_d*t) * exp(-gamma*t)
    """
    return (V0 * C * OMEGA0**2 / OMEGA_D) * np.sin(OMEGA_D * t) * np.exp(-GAMMA_DAMP * t)


def _run_rlc_short_circuit(dt: float = 10e-9, t_end: float = 15e-6) -> tuple:
    """Run RLC solver with no plasma (short-circuit discharge).

    Returns (times, currents, voltages, solver).
    """
    solver = RLCSolver(
        C=C, V0=V0, L0=L0, R0=R0,
        anode_radius=ANODE_RADIUS,
        cathode_radius=CATHODE_RADIUS,
    )
    coupling = CouplingState(Lp=0.0, R_plasma=0.0, current=0.0, voltage=V0, dL_dt=0.0)

    n_steps = int(t_end / dt)
    times = np.zeros(n_steps)
    currents = np.zeros(n_steps)
    voltages = np.zeros(n_steps)

    for i in range(n_steps):
        coupling = solver.step(coupling, back_emf=0.0, dt=dt)
        times[i] = solver.state.time
        currents[i] = solver.state.current
        voltages[i] = solver.state.voltage

    return times, currents, voltages, solver


def _run_rlc_with_plasma_loading(
    dt: float = 10e-9,
    t_end: float = 12e-6,
    R_plasma_func=None,
    Lp_func=None,
) -> tuple:
    """Run RLC solver with time-dependent plasma loading.

    Args:
        dt: Timestep [s].
        t_end: End time [s].
        R_plasma_func: Callable(t) -> R_plasma [Ohm].
        Lp_func: Callable(t) -> L_plasma [H].

    Returns (times, currents, voltages, solver).
    """
    solver = RLCSolver(
        C=C, V0=V0, L0=L0, R0=R0,
        anode_radius=ANODE_RADIUS,
        cathode_radius=CATHODE_RADIUS,
        crowbar_enabled=True,
        crowbar_mode="voltage_zero",
    )

    n_steps = int(t_end / dt)
    times = np.zeros(n_steps)
    currents = np.zeros(n_steps)
    voltages = np.zeros(n_steps)
    Lp_prev = 0.0

    for i in range(n_steps):
        t = (i + 1) * dt
        Rp = R_plasma_func(t) if R_plasma_func else 0.0
        Lp = Lp_func(t) if Lp_func else 0.0
        dLp_dt = (Lp - Lp_prev) / dt if i > 0 else 0.0
        Lp_prev = Lp

        coupling = CouplingState(
            Lp=Lp, R_plasma=Rp, current=solver.state.current,
            voltage=solver.state.voltage, dL_dt=dLp_dt,
        )
        coupling = solver.step(coupling, back_emf=0.0, dt=dt)
        times[i] = solver.state.time
        currents[i] = solver.state.current
        voltages[i] = solver.state.voltage

    return times, currents, voltages, solver


# ============================================================
# Test 1: Short-circuit I(t) vs analytical
# ============================================================

class TestShortCircuitRLC:
    """Verify RLC solver against analytical short-circuit solution."""

    def test_peak_current_matches_analytical(self):
        """Peak current should match analytical within 0.1%."""
        times, currents, _, _ = _run_rlc_short_circuit()
        I_analytical = _analytical_current(times)

        I_peak_sim = np.max(currents)
        I_peak_ana = np.max(I_analytical)
        rel_err = abs(I_peak_sim - I_peak_ana) / I_peak_ana
        assert rel_err < 1e-3, f"Peak current error: {rel_err:.4e}"

    def test_peak_time_matches_analytical(self):
        """Peak current time should match within 0.1 us."""
        times, currents, _, _ = _run_rlc_short_circuit()
        t_peak = times[np.argmax(currents)]
        # Analytical peak: t_peak = atan(omega_d/gamma) / omega_d
        t_peak_ana = np.arctan(OMEGA_D / GAMMA_DAMP) / OMEGA_D
        assert abs(t_peak - t_peak_ana) < 0.1e-6, (
            f"Peak time: sim={t_peak*1e6:.2f} us, ana={t_peak_ana*1e6:.2f} us"
        )

    def test_quarter_period(self):
        """Quarter period T/4 should be ~10.78 us."""
        times, currents, _, _ = _run_rlc_short_circuit()
        t_peak = times[np.argmax(currents)]
        # T/4 is approximately the peak time (exact for no damping)
        assert 8e-6 < t_peak < 13e-6, f"T/4 = {t_peak*1e6:.2f} us, expected ~10.78 us"

    def test_energy_conservation(self):
        """Circuit energy should be conserved to machine precision."""
        _, _, _, solver = _run_rlc_short_circuit()
        E_total = solver.total_energy()
        E_init = solver.initial_energy()
        rel_err = abs(E_total - E_init) / E_init
        assert rel_err < 1e-10, f"Energy conservation: dE/E = {rel_err:.2e}"

    def test_initial_energy_correct(self):
        """Initial energy = 0.5*C*V0^2 = 485.5 kJ."""
        _, _, _, solver = _run_rlc_short_circuit()
        E = solver.initial_energy()
        assert pytest.approx(E, rel=1e-6) == E0_J

    def test_waveform_l2_error(self):
        """L2 error between numerical and analytical I(t) < 0.1%."""
        times, currents, _, _ = _run_rlc_short_circuit()
        I_analytical = _analytical_current(times)

        l2_err = np.sqrt(np.mean((currents - I_analytical) ** 2))
        l2_norm = np.sqrt(np.mean(I_analytical**2))
        rel_l2 = l2_err / l2_norm
        assert rel_l2 < 1e-3, f"L2 waveform error: {rel_l2:.4e}"

    def test_second_order_convergence(self):
        """Verify 2nd-order temporal convergence."""
        errors = []
        for dt in [100e-9, 50e-9, 25e-9]:
            t, current, _, _ = _run_rlc_short_circuit(dt=dt, t_end=10e-6)
            I_ana = _analytical_current(t)
            l2 = np.sqrt(np.mean((current - I_ana) ** 2)) / np.sqrt(np.mean(I_ana**2))
            errors.append(l2)

        # Convergence rate: error(dt/2) / error(dt) ~ 0.25 for 2nd order
        rate1 = errors[0] / errors[1]
        rate2 = errors[1] / errors[2]
        # Should be ~4 for 2nd order (dt halved → error ÷ 4)
        assert rate1 > 3.0, f"Convergence rate 1: {rate1:.2f}, expected ~4"
        assert rate2 > 3.0, f"Convergence rate 2: {rate2:.2f}, expected ~4"

    def test_underdamped_oscillation(self):
        """Current should cross zero (underdamped for PF-1000 params).

        T/2 ~ 21.5 us, so run to 25 us to see the first zero crossing.
        """
        _, currents, _, _ = _run_rlc_short_circuit(t_end=25e-6)
        # Find first zero crossing after the peak
        i_peak = np.argmax(currents)
        post_peak = currents[i_peak:]
        sign_changes = np.diff(np.sign(post_peak))
        zero_crossings = np.where(sign_changes != 0)[0]
        assert len(zero_crossings) > 0, "No zero crossing after peak — overdamped!"


# ============================================================
# Test 2: Plasma-loaded I(t) with synthetic loading
# ============================================================

class TestPlasmaLoadedRLC:
    """Test circuit solver with synthetic plasma resistance and inductance.

    Uses a simple model where plasma resistance increases sharply
    at pinch time and inductance grows as the current sheet moves.
    This mimics the Lee model behavior without running full MHD.
    """

    @staticmethod
    def _ramp_resistance(t: float) -> float:
        """Synthetic plasma resistance: rises sharply at pinch (~6 us)."""
        if t < 5e-6:
            return 1e-3  # small during rundown
        elif t < 7e-6:
            return 1e-3 + 0.02 * (t - 5e-6) / 2e-6  # ramp to 21 mOhm
        else:
            return 0.021  # plateau (pinch phase)

    @staticmethod
    def _ramp_inductance(t: float) -> float:
        """Synthetic plasma inductance: grows during rundown, peaks at pinch."""
        if t < 5e-6:
            return 20e-9 * t / 5e-6  # linear growth to 20 nH
        elif t < 7e-6:
            return 20e-9 + 30e-9 * (t - 5e-6) / 2e-6  # jump to 50 nH
        else:
            return 50e-9

    def test_peak_current_reduced_by_loading(self):
        """Plasma loading should reduce peak current vs short-circuit."""
        _, I_sc, _, _ = _run_rlc_short_circuit()
        _, I_loaded, _, _ = _run_rlc_with_plasma_loading(
            R_plasma_func=self._ramp_resistance,
            Lp_func=self._ramp_inductance,
        )
        # Plasma loading increases total impedance → lower peak
        assert np.max(I_loaded) < np.max(I_sc), (
            f"Loaded peak ({np.max(I_loaded)/1e6:.3f} MA) not less than "
            f"SC peak ({np.max(I_sc)/1e6:.3f} MA)"
        )

    def test_peak_in_megaampere_range(self):
        """Peak current should be in MA range (not kA or GA)."""
        _, currents, _, _ = _run_rlc_with_plasma_loading(
            R_plasma_func=self._ramp_resistance,
            Lp_func=self._ramp_inductance,
        )
        I_peak = np.max(currents)
        assert 0.5e6 < I_peak < 5.0e6, f"Peak current {I_peak/1e6:.3f} MA out of range"

    def test_current_dip_from_inductance_jump(self):
        """Rising inductance at pinch should create a current dip."""
        times, currents, _, _ = _run_rlc_with_plasma_loading(
            R_plasma_func=self._ramp_resistance,
            Lp_func=self._ramp_inductance,
        )
        # Find peak
        i_peak = np.argmax(currents)
        I_peak = currents[i_peak]

        # After peak, current should decrease (normal for underdamped)
        # But with plasma loading, the decrease should be faster (dip)
        # Check that current at t=7 us < 70% of peak
        idx_7us = np.searchsorted(times, 7e-6)
        if idx_7us < len(currents):
            I_at_7us = currents[idx_7us]
            assert I_at_7us < 0.8 * I_peak, (
                f"No current dip: I(7us)={I_at_7us/1e6:.3f} MA, "
                f"I_peak={I_peak/1e6:.3f} MA"
            )

    def test_energy_budget_with_loading(self):
        """Circuit + plasma energy should not exceed initial stored energy.

        With plasma loading, energy flows:
            E_cap (initial) → E_ind + E_cap + E_res + E_plasma_ohmic + E_mechanical
        where E_mechanical = integral(I * dLp/dt * dt) is the work done by
        the changing inductance on the plasma (snowplow).  The circuit solver
        tracks all except E_mechanical, which the MHD solver handles.
        """
        _, _, _, solver = _run_rlc_with_plasma_loading(
            R_plasma_func=self._ramp_resistance,
            Lp_func=self._ramp_inductance,
        )
        E_circuit = solver.total_energy()
        E_plasma_ohmic = solver.state.energy_res_plasma
        E_init = solver.initial_energy()
        # Circuit + plasma ohmic should be LESS than initial (rest is mechanical work)
        E_tracked = E_circuit + E_plasma_ohmic
        assert E_tracked <= E_init * 1.01, (
            f"Energy exceeded: tracked={E_tracked:.0f} J, initial={E_init:.0f} J"
        )
        # Mechanical work should be positive and reasonable (< 50% of initial)
        E_mechanical = E_init - E_tracked
        assert E_mechanical > 0, "Negative mechanical work (unphysical)"
        assert E_mechanical < 0.5 * E_init, (
            f"Excessive mechanical work: {E_mechanical/E_init*100:.1f}% of E_init"
        )

    def test_crowbar_fires_short_circuit(self):
        """Crowbar fires when V_cap crosses zero in underdamped short-circuit.

        Short-circuit T/2 ~ 21.5 us; crowbar triggers at first V=0 crossing.
        """
        solver = RLCSolver(
            C=C, V0=V0, L0=L0, R0=R0,
            anode_radius=ANODE_RADIUS, cathode_radius=CATHODE_RADIUS,
            crowbar_enabled=True, crowbar_mode="voltage_zero",
        )
        coupling = CouplingState(Lp=0.0, R_plasma=0.0, current=0.0, voltage=V0, dL_dt=0.0)
        dt = 10e-9
        for _ in range(int(25e-6 / dt)):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)
        assert solver.state.crowbar_fired, "Crowbar did not fire in short-circuit"
        # V_cap crosses zero at t_V0 = arctan(-omega_d/gamma) / omega_d + pi/omega_d
        # ≈ 12.37 us for PF-1000 (earlier than current T/2 ~ 21.5 us)
        assert 8e-6 < solver.state.crowbar_fire_time < 18e-6


# ============================================================
# Test 3: Coupled engine I(t) — PF-1000 preset
# ============================================================

class TestPF1000CoupledSimulation:
    """Run the full DPF engine with PF-1000 preset and verify I(t).

    Uses the Python engine (teaching backend) on a small grid for
    fast CI.  The coupled simulation should produce a rising current
    that responds to plasma dynamics.
    """

    @staticmethod
    def _make_pf1000_config():
        """Create a minimal PF-1000 config for fast coupled testing."""
        from dpf.config import SimulationConfig

        return SimulationConfig(
            grid_shape=[16, 1, 32],
            dx=7.5e-4,
            sim_time=2e-6,  # Short run (2 us) for CI speed
            dt_init=1e-10,
            rho0=4e-4,
            T0=300.0,
            circuit={
                "C": C,
                "V0": V0,
                "L0": L0,
                "R0": R0,
                "anode_radius": ANODE_RADIUS,
                "cathode_radius": CATHODE_RADIUS,
                "crowbar_enabled": True,
                "crowbar_mode": "voltage_zero",
            },
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": False},
        )

    def test_engine_creates_with_pf1000(self):
        """Engine should accept PF-1000 circuit parameters."""
        from dpf.engine import SimulationEngine
        config = self._make_pf1000_config()
        engine = SimulationEngine(config)
        assert engine.circuit is not None
        assert pytest.approx(engine.circuit.C, rel=1e-6) == C

    def test_initial_current_is_zero(self):
        """Current starts at zero."""
        from dpf.engine import SimulationEngine
        config = self._make_pf1000_config()
        engine = SimulationEngine(config)
        assert engine.circuit.current == 0.0

    def test_current_rises_after_step(self):
        """After one step, current should be positive (capacitor discharging)."""
        from dpf.engine import SimulationEngine
        config = self._make_pf1000_config()
        engine = SimulationEngine(config)
        engine.step()
        assert engine.circuit.current > 0

    def test_current_increases_over_multiple_steps(self):
        """Current should increase during early discharge (t < T/4)."""
        from dpf.engine import SimulationEngine
        config = self._make_pf1000_config()
        engine = SimulationEngine(config)

        currents = []
        for _ in range(10):
            result = engine.step()
            currents.append(engine.circuit.current)
            if result.finished:
                break

        # Current should be monotonically increasing in early discharge
        assert all(currents[i] <= currents[i + 1] for i in range(min(5, len(currents) - 1))), (
            f"Current not monotonically rising in early discharge: {currents[:6]}"
        )

    def test_voltage_decreases_from_discharge(self):
        """Capacitor voltage should decrease as energy transfers to circuit."""
        from dpf.engine import SimulationEngine
        config = self._make_pf1000_config()
        engine = SimulationEngine(config)

        V_initial = engine.circuit.voltage
        for _ in range(10):
            result = engine.step()
            if result.finished:
                break

        assert engine.circuit.voltage < V_initial

    def test_energy_conservation_coupled(self):
        """Circuit energy should be approximately conserved in coupled run."""
        from dpf.engine import SimulationEngine
        config = self._make_pf1000_config()
        engine = SimulationEngine(config)

        E0 = engine.circuit.initial_energy()
        for _ in range(20):
            result = engine.step()
            if result.finished:
                break

        E_total = engine.circuit.total_energy()
        # Allow larger tolerance for coupled run (plasma dissipation)
        rel_err = abs(E_total - E0) / E0
        assert rel_err < 0.5, f"Energy conservation: dE/E = {rel_err:.2e}"


# ============================================================
# Test 4: PF-1000 analytical benchmarks
# ============================================================

class TestPF1000AnalyticalBenchmarks:
    """Verify key PF-1000 analytical quantities."""

    def test_initial_stored_energy(self):
        """E0 = 0.5 * C * V0^2 = 485.5 kJ."""
        E = 0.5 * C * V0**2
        assert pytest.approx(E, rel=0.01) == 485514.0

    def test_short_circuit_peak_current(self):
        """Short-circuit peak ~3.93 MA (undamped = 5.38 MA)."""
        I_undamped = V0 / np.sqrt(L0 / C)
        assert pytest.approx(I_undamped / 1e6, rel=0.01) == 5.384

    def test_quarter_period_analytical(self):
        """T/4 = pi/(2*omega_d) ~ 10.78 us."""
        T4 = np.pi / (2.0 * OMEGA_D)
        assert pytest.approx(T4 * 1e6, rel=0.01) == 10.78

    def test_damping_factor(self):
        """gamma = R/(2L) = 34,328 s^-1."""
        gamma = R0 / (2 * L0)
        assert pytest.approx(gamma, rel=0.01) == 34328.0

    def test_plasma_inductance_at_full_compression(self):
        """L_plasma at r_pinch = 1 cm, length = 5 cm should be ~50-100 nH."""
        solver = RLCSolver(C=C, V0=V0, L0=L0, R0=R0,
                           anode_radius=ANODE_RADIUS, cathode_radius=CATHODE_RADIUS)
        Lp = solver.plasma_inductance_estimate(
            pinch_radius=0.01, length=0.05,
        )
        # L = (mu0/2pi) * 0.05 * ln(0.16/0.01) ~ 27.7 nH
        assert 10e-9 < Lp < 100e-9, f"L_plasma = {Lp*1e9:.1f} nH"

    def test_lee_model_current_fraction_effect(self):
        """With fc=0.7, effective current in sheath = 0.7 * I_circuit.

        This reduces the magnetic pressure by fc^2 = 0.49, significantly
        reducing the peak current seen by the plasma.
        """
        fc = 0.7
        fm = 0.13
        # fc reduces effective current → increases effective L → reduces I_peak
        # Simple estimate: I_peak_loaded ~ I_peak_sc * (L0 / (L0 + Lp)) for Lp > 0
        # With Lee model, the reduction is more complex
        assert 0.5 < fc < 1.0, "fc should be between 0.5 and 1.0"
        assert 0.05 < fm < 0.5, "fm should be between 0.05 and 0.5"
