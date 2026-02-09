"""Analytical RLC circuit verification for underdamped, critically-damped, and overdamped regimes.

Tests the RLCSolver implementation against exact analytical solutions for:
- Underdamped oscillations (R < 2*sqrt(L/C))
- Critically damped response (R = 2*sqrt(L/C))
- Overdamped decay (R > 2*sqrt(L/C))
- Energy conservation (with and without resistance)

Each test compares numerical integration to closed-form solutions with
specified tolerances appropriate for the implicit midpoint method.
"""

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState


def test_underdamped_peak_current():
    """Verify peak current matches analytical solution for underdamped RLC.

    PF-1000-like parameters: V0=27 kV, C=1.332 mF, L0=15 nH, R0=2 mOhm.
    This produces highly underdamped oscillations. Peak current occurs near
    t = pi/(2*omega_d). Analytical peak: I_peak ≈ V0*sqrt(C/L)*exp(-gamma*t_peak).
    """
    # PF-1000-like parameters (underdamped)
    V0 = 27000.0  # V
    C = 1.332e-3  # F
    L0 = 15e-9  # H
    R0 = 2e-3  # Ohm (low resistance)

    # Analytical parameters
    omega_0 = 1.0 / np.sqrt(L0 * C)
    gamma = R0 / (2.0 * L0)
    omega_d = np.sqrt(omega_0**2 - gamma**2)

    # Peak occurs where dI/dt=0: t_peak = atan(omega_d/gamma)/omega_d

    # Exact analytical solution: I(t) = V0/(omega_d*L) * exp(-gamma*t) * sin(omega_d*t)
    # Peak occurs where dI/dt=0: t_peak = atan(omega_d/gamma)/omega_d
    t_peak = np.arctan(omega_d / gamma) / omega_d
    I_peak_analytical = (V0 / (omega_d * L0)) * np.exp(-gamma * t_peak) * np.sin(omega_d * t_peak)

    # Setup solver
    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    # Run to peak with fine timestep
    n_steps = 10000
    dt = t_peak / n_steps

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)
    I_max = 0.0

    for _ in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)
        I_max = max(I_max, abs(solver.state.current))

    # Check numerical peak matches analytical within 1%
    relative_error = abs(I_max - I_peak_analytical) / I_peak_analytical
    assert relative_error < 0.01, (
        f"Peak current error too large: {relative_error*100:.2f}%\n"
        f"Numerical: {I_max:.3e} A, Analytical: {I_peak_analytical:.3e} A"
    )


def test_underdamped_period():
    """Verify oscillation period matches analytical prediction.

    Runs for 2 full periods, finds zero crossings, and checks that the
    measured period matches T = 2*pi/omega_d within 1%.
    """
    # Underdamped parameters
    V0 = 10000.0
    C = 100e-6  # 100 µF
    L0 = 50e-9  # 50 nH
    R0 = 1e-3  # 1 mOhm

    # Analytical period
    omega_0 = 1.0 / np.sqrt(L0 * C)
    gamma = R0 / (2.0 * L0)
    omega_d = np.sqrt(omega_0**2 - gamma**2)
    T_analytical = 2.0 * np.pi / omega_d

    # Setup solver
    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    # Run for 2 periods
    t_end = 2.0 * T_analytical
    dt = t_end / 20000  # Fine timestep

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)

    times = []
    currents = []
    t = 0.0

    while t < t_end:
        solver.step(coupling, back_emf=0.0, dt=dt)
        t += dt
        times.append(t)
        currents.append(solver.state.current)

    # Find zero crossings (positive to negative)
    times = np.array(times)
    currents = np.array(currents)

    zero_crossings = []
    for i in range(1, len(currents)):
        if currents[i-1] > 0 and currents[i] <= 0:
            # Linear interpolation to find exact crossing
            t_cross = times[i-1] + (times[i] - times[i-1]) * (
                -currents[i-1] / (currents[i] - currents[i-1])
            )
            zero_crossings.append(t_cross)

    # Should have at least 2 crossings in 2 periods
    assert len(zero_crossings) >= 2, f"Expected at least 2 zero crossings, got {len(zero_crossings)}"

    # Measure period from consecutive crossings
    periods = np.diff(zero_crossings)
    T_numerical = np.mean(periods)

    relative_error = abs(T_numerical - T_analytical) / T_analytical
    assert relative_error < 0.01, (
        f"Period error too large: {relative_error*100:.2f}%\n"
        f"Numerical: {T_numerical:.6e} s, Analytical: {T_analytical:.6e} s"
    )


@pytest.mark.slow
def test_underdamped_waveform():
    """Compare full I(t) waveform to analytical solution over one quarter period.

    Runs 5000 steps and compares current at each timestep to the analytical
    underdamped solution. Checks L2 relative error < 2%.
    """
    # Underdamped parameters
    V0 = 15000.0
    C = 200e-6  # 200 µF
    L0 = 30e-9  # 30 nH
    R0 = 1.5e-3  # 1.5 mOhm

    # Analytical parameters
    omega_0 = 1.0 / np.sqrt(L0 * C)
    gamma = R0 / (2.0 * L0)
    omega_d = np.sqrt(omega_0**2 - gamma**2)

    # Quarter period
    t_quarter = np.pi / (2.0 * omega_d)
    n_steps = 5000
    dt = t_quarter / n_steps

    # Setup solver
    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)

    times = []
    I_numerical = []
    I_analytical = []
    t = 0.0

    for _ in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)
        t += dt

        times.append(t)
        I_numerical.append(solver.state.current)

        # Analytical solution: I(t) = (V0/(omega_d*L)) * exp(-gamma*t) * sin(omega_d*t)
        I_analytical.append(
            (V0 / (omega_d * L0)) * np.exp(-gamma * t) * np.sin(omega_d * t)
        )

    I_numerical = np.array(I_numerical)
    I_analytical = np.array(I_analytical)

    # L2 relative error
    l2_error = np.sqrt(np.sum((I_numerical - I_analytical)**2)) / np.sqrt(np.sum(I_analytical**2))

    assert l2_error < 0.02, (
        f"L2 relative error too large: {l2_error*100:.2f}%\n"
        f"Expected < 2%"
    )


@pytest.mark.slow
def test_critically_damped():
    """Verify critically damped response (R = 2*sqrt(L/C)).

    Current should have single peak and decay monotonically. Peak location
    should roughly match t_peak ≈ L/R, allowing 20% tolerance for numerical
    discretization effects.
    """
    # Parameters
    V0 = 10000.0
    C = 100e-6  # 100 µF
    L0 = 100e-9  # 100 nH

    # Critical resistance
    R0 = 2.0 * np.sqrt(L0 / C)  # ≈ 6.32e-3 Ohm

    # Critically damped: I(t) = (V0/L)*t*exp(-R*t/(2L))
    # dI/dt = 0 -> t_peak = 2L/R
    t_peak_expected = 2.0 * L0 / R0

    # Setup solver
    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    # Run for several time constants
    t_end = 5.0 * t_peak_expected
    n_steps = 5000
    dt = t_end / n_steps

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)

    times = []
    currents = []
    t = 0.0

    for _ in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)
        t += dt
        times.append(t)
        currents.append(solver.state.current)

    currents = np.array(currents)
    times = np.array(times)

    # Find peak
    i_peak = np.argmax(currents)
    t_peak_numerical = times[i_peak]

    # Check peak location within 20% (implicit midpoint has some discretization error)
    relative_error = abs(t_peak_numerical - t_peak_expected) / t_peak_expected
    assert relative_error < 0.20, (
        f"Peak time error: {relative_error*100:.2f}%\n"
        f"Numerical: {t_peak_numerical:.6e} s, Expected: {t_peak_expected:.6e} s"
    )

    # Verify no oscillations after peak (should decay monotonically)
    after_peak = currents[i_peak:]
    is_monotonic = np.all(np.diff(after_peak) <= 1e-10)  # Allow small numerical noise
    assert is_monotonic, "Current should decay monotonically after peak (critically damped)"


@pytest.mark.slow
def test_overdamped():
    """Verify overdamped response (R >> 2*sqrt(L/C)).

    Current should rise slowly and decay without oscillation. Should never
    go negative (no oscillations).
    """
    # Parameters
    V0 = 5000.0
    C = 50e-6  # 50 µF
    L0 = 20e-9  # 20 nH
    R0 = 0.1  # Much larger than 2*sqrt(L/C) ≈ 4e-3

    # Verify overdamped condition
    R_critical = 2.0 * np.sqrt(L0 / C)
    assert R_critical < R0, "Should be overdamped"

    # Setup solver
    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    # Run for many time constants
    tau = L0 / R0
    t_end = 10.0 * tau
    n_steps = 2000
    dt = t_end / n_steps

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)

    currents = []

    for _ in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)
        currents.append(solver.state.current)

    currents = np.array(currents)

    # Current should never be negative (no oscillations)
    assert np.all(currents >= 0), "Current should never be negative (overdamped)"

    # Find peak
    i_peak = np.argmax(currents)

    # After peak, should decay monotonically
    after_peak = currents[i_peak:]
    # Allow small numerical noise
    is_monotonic_decay = np.all(np.diff(after_peak) <= 1e-10)
    assert is_monotonic_decay, "Current should decay monotonically after peak (overdamped)"


def test_energy_conservation_no_resistance():
    """Verify energy conservation with R=0.

    With no resistance (R0=0, ESR=0, R_plasma=0), total energy
    E = E_cap + E_ind should be conserved. Checks within 1%.
    """
    # Parameters
    V0 = 8000.0
    C = 80e-6  # 80 µF
    L0 = 40e-9  # 40 nH
    R0 = 0.0  # No resistance

    # Initial energy (all capacitive)
    E_initial = 0.5 * C * V0**2

    # Setup solver
    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    # Run for one quarter period
    omega_0 = 1.0 / np.sqrt(L0 * C)
    t_quarter = np.pi / (2.0 * omega_0)
    n_steps = 1000
    dt = t_quarter / n_steps

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)

    for _ in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)

    # Calculate total energy
    E_cap = 0.5 * C * solver.state.voltage**2
    E_ind = 0.5 * L0 * solver.state.current**2
    E_total = E_cap + E_ind

    # Check conservation
    relative_error = abs(E_total - E_initial) / E_initial
    assert relative_error < 0.01, (
        f"Energy conservation violated: {relative_error*100:.2f}% error\n"
        f"Initial: {E_initial:.3e} J, Final: {E_total:.3e} J"
    )


def test_energy_conservation_with_resistance():
    """Verify energy accounting with resistance.

    With R>0, E_cap + E_ind + E_res should equal E_initial within 1%.
    The solver tracks cumulative resistive losses in energy_res.
    """
    # Parameters
    V0 = 6000.0
    C = 60e-6  # 60 µF
    L0 = 30e-9  # 30 nH
    R0 = 5e-3  # 5 mOhm

    # Initial energy
    E_initial = 0.5 * C * V0**2

    # Setup solver
    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    # Run for one quarter period
    omega_0 = 1.0 / np.sqrt(L0 * C)
    gamma = R0 / (2.0 * L0)
    omega_d = np.sqrt(omega_0**2 - gamma**2)
    t_quarter = np.pi / (2.0 * omega_d)
    n_steps = 1000
    dt = t_quarter / n_steps

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)

    for _ in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)

    # Total energy accounting
    E_cap = 0.5 * C * solver.state.voltage**2
    E_ind = 0.5 * L0 * solver.state.current**2
    E_res = solver.state.energy_res
    E_total = E_cap + E_ind + E_res

    # Check energy balance
    relative_error = abs(E_total - E_initial) / E_initial
    assert relative_error < 0.01, (
        f"Energy balance violated: {relative_error*100:.2f}% error\n"
        f"Initial: {E_initial:.3e} J, Final: {E_total:.3e} J\n"
        f"(E_cap={E_cap:.3e}, E_ind={E_ind:.3e}, E_res={E_res:.3e})"
    )


def test_initial_conditions():
    """Verify initial conditions at t=0.

    Should have I=0 and V=V0 at initialization.
    """
    V0 = 12000.0
    C = 100e-6
    L0 = 50e-9
    R0 = 3e-3

    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=0.01,
        cathode_radius=0.05,
    )

    # Check initial state
    assert solver.state.current == pytest.approx(0.0, abs=1e-12), "Initial current should be zero"
    assert solver.state.voltage == pytest.approx(V0, rel=1e-10), "Initial voltage should be V0"
    assert solver.state.energy_res == pytest.approx(0.0, abs=1e-12), "Initial resistive energy should be zero"


@pytest.mark.slow
def test_small_timestep_convergence():
    """Verify 2nd order convergence of implicit midpoint method.

    Runs with dt and dt/2. The error should scale as dt^2, so halving
    the timestep should reduce error by factor of ~4.
    """
    # Parameters
    V0 = 9000.0
    C = 90e-6  # 90 µF
    L0 = 45e-9  # 45 nH
    R0 = 2e-3  # 2 mOhm

    # Analytical parameters
    omega_0 = 1.0 / np.sqrt(L0 * C)
    gamma = R0 / (2.0 * L0)
    omega_d = np.sqrt(omega_0**2 - gamma**2)

    # Run time
    t_end = np.pi / (4.0 * omega_d)  # Quarter period

    # Coarse run (dt)
    n_steps_coarse = 500
    dt_coarse = t_end / n_steps_coarse

    solver_coarse = RLCSolver(
        C=C, V0=V0, L0=L0, R0=R0, ESR=0.0, ESL=0.0,
        anode_radius=0.01, cathode_radius=0.05
    )

    coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)
    t = 0.0
    for _ in range(n_steps_coarse):
        solver_coarse.step(coupling, back_emf=0.0, dt=dt_coarse)
        t += dt_coarse

    I_coarse = solver_coarse.state.current

    # Fine run (dt/2)
    n_steps_fine = 1000
    dt_fine = t_end / n_steps_fine

    solver_fine = RLCSolver(
        C=C, V0=V0, L0=L0, R0=R0, ESR=0.0, ESL=0.0,
        anode_radius=0.01, cathode_radius=0.05
    )

    t = 0.0
    for _ in range(n_steps_fine):
        solver_fine.step(coupling, back_emf=0.0, dt=dt_fine)
        t += dt_fine

    I_fine = solver_fine.state.current

    # Analytical solution at t_end
    I_analytical = (V0 / (omega_d * L0)) * np.exp(-gamma * t_end) * np.sin(omega_d * t_end)

    # Errors
    error_coarse = abs(I_coarse - I_analytical)
    error_fine = abs(I_fine - I_analytical)

    # Error ratio should be approximately 4 for 2nd order method
    error_ratio = error_coarse / error_fine

    # Allow range 3-5 (theoretical is 4, but discretization and nonlinearity introduce variance)
    assert 3.0 < error_ratio < 5.0, (
        f"Error ratio {error_ratio:.2f} not consistent with 2nd order convergence\n"
        f"Expected ~4 (range 3-5)\n"
        f"Error (dt): {error_coarse:.6e}, Error (dt/2): {error_fine:.6e}"
    )
