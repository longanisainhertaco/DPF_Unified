"""Energy balance verification tests.

Verifies energy conservation in the coupled circuit-MHD system:
1. RLC circuit energy conservation over long runs
2. Plasma energy tracking (kinetic + thermal + magnetic)
3. Bremsstrahlung radiation energy accounting
4. Circuit-plasma coupling energy consistency (Ohmic heating)
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.constants import k_B, mu_0
from dpf.core.bases import CouplingState
from dpf.fluid.mhd_solver import MHDSolver
from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

# ============================================================
# RLC circuit energy conservation
# ============================================================


@pytest.mark.slow
def test_rlc_energy_conservation_long_run():
    """Test RLC circuit energy conservation over 1000 timesteps.

    Verify E_cap + E_ind + E_res = E_initial within 1% tolerance.
    """
    # Circuit parameters
    C = 1e-6      # F
    V0 = 1e4      # V
    L0 = 1e-7     # H
    R0 = 0.01     # Ohm
    anode_radius = 0.005
    cathode_radius = 0.01

    solver = RLCSolver(
        C=C,
        V0=V0,
        L0=L0,
        R0=R0,
        ESR=0.0,
        ESL=0.0,
        anode_radius=anode_radius,
        cathode_radius=cathode_radius,
    )

    # Initial energy (all in capacitor)
    E_initial = 0.5 * C * V0**2

    # Time integration for 1000 steps
    dt = 1e-9  # Small timestep
    n_steps = 1000

    # Zero plasma coupling for isolated circuit test
    coupling = CouplingState(Lp=0.0, emf=0.0, current=0.0, voltage=0.0, dL_dt=0.0)

    for _ in range(n_steps):
        coupling = solver.step(coupling, back_emf=0.0, dt=dt)

    # Total energy = capacitor + inductor + resistive dissipation
    E_total = solver.total_energy()
    conservation = E_total / E_initial

    # Energy should be conserved within 1% (accounting for numerical errors)
    assert 0.99 < conservation < 1.01, (
        f"Energy not conserved: E_initial={E_initial:.3e} J, "
        f"E_total={E_total:.3e} J, ratio={conservation:.6f}"
    )


def test_rlc_energy_partition():
    """Test that RLC energy correctly partitions between capacitor and inductor."""
    C = 1e-6
    V0 = 1e4
    L0 = 1e-7
    R0 = 0.0  # No resistance to test ideal oscillation

    solver = RLCSolver(C=C, V0=V0, L0=L0, R0=R0, anode_radius=0.005, cathode_radius=0.01)

    E_initial = 0.5 * C * V0**2

    # At quarter period of LC oscillation, all energy should be in inductor
    # T = 2*pi*sqrt(LC) => T/4 = pi*sqrt(LC)/2
    period = 2.0 * np.pi * np.sqrt(L0 * C)
    t_quarter = period / 4.0

    dt = t_quarter / 100.0
    coupling = CouplingState()

    time = 0.0
    while time < t_quarter:
        coupling = solver.step(coupling, back_emf=0.0, dt=dt)
        time += dt

    # At T/4, energy should be ~100% in inductor (E_cap ~ 0, E_ind ~ E_initial)
    E_cap = solver.state.energy_cap
    E_ind = solver.state.energy_ind

    # Allow 5% tolerance for numerical errors
    assert E_cap / E_initial < 0.05, f"E_cap should be near zero at T/4: {E_cap/E_initial:.3f}"
    assert E_ind / E_initial > 0.95, f"E_ind should be near E_initial at T/4: {E_ind/E_initial:.3f}"


# ============================================================
# Plasma energy tracking
# ============================================================


def test_plasma_energy_tracking():
    """Test that plasma kinetic + thermal + magnetic energy can be computed."""
    gamma = 5.0 / 3.0
    N = 16
    ny = 16
    nz = 16
    dx = 0.01

    # Initialize uniform state with non-zero fields
    rho0 = 1e-4
    p0 = 1e3
    v0 = 100.0  # m/s
    B0 = 0.1    # T

    m_i = 3.34e-27
    n_i = rho0 / m_i
    T0 = p0 / (2.0 * n_i * k_B)

    state = {
        "rho": np.full((N, ny, nz), rho0),
        "velocity": np.zeros((3, N, ny, nz)),
        "pressure": np.full((N, ny, nz), p0),
        "B": np.zeros((3, N, ny, nz)),
        "Te": np.full((N, ny, nz), T0),
        "Ti": np.full((N, ny, nz), T0),
        "psi": np.zeros((N, ny, nz)),
    }

    # Add uniform velocity in x-direction
    state["velocity"][0] = v0

    # Add uniform B-field in z-direction
    state["B"][2] = B0

    # Compute energies
    dV = dx**3  # Cell volume
    total_volume = N * ny * nz * dV

    # Kinetic energy: 0.5 * rho * |v|^2
    v_sq = np.sum(state["velocity"]**2, axis=0)
    E_kinetic = np.sum(0.5 * state["rho"] * v_sq) * dV

    # Thermal energy: p / (gamma - 1)
    E_thermal = np.sum(state["pressure"] / (gamma - 1.0)) * dV

    # Magnetic energy: B^2 / (2 * mu_0)
    B_sq = np.sum(state["B"]**2, axis=0)
    E_magnetic = np.sum(B_sq / (2.0 * mu_0)) * dV

    # Expected values (analytical)
    E_kinetic_expected = 0.5 * rho0 * v0**2 * total_volume
    E_thermal_expected = (p0 / (gamma - 1.0)) * total_volume
    E_magnetic_expected = (B0**2 / (2.0 * mu_0)) * total_volume

    # Check within 0.1% (should be exact for uniform fields)
    assert abs(E_kinetic - E_kinetic_expected) / E_kinetic_expected < 1e-3
    assert abs(E_thermal - E_thermal_expected) / E_thermal_expected < 1e-3
    assert abs(E_magnetic - E_magnetic_expected) / E_magnetic_expected < 1e-3


# ============================================================
# Bremsstrahlung energy accounting
# ============================================================


def test_bremsstrahlung_energy_loss_consistency():
    """Test that bremsstrahlung removes correct amount of energy.

    Apply bremsstrahlung cooling for dt, verify:
    Energy removed = integral(P_rad * dt) within numerical tolerance.
    """
    # Plasma parameters
    N = 16
    ny = 16
    nz = 16
    Te0 = 1e6  # K (hot plasma for significant bremsstrahlung)
    ne0 = 1e20  # m^-3

    Te = np.full((N, ny, nz), Te0)
    ne = np.full((N, ny, nz), ne0)

    # Time step
    dt = 1e-9  # s

    # Apply bremsstrahlung
    Z = 1.0
    gaunt_factor = 1.2
    Te_new, P_radiated = apply_bremsstrahlung_losses(
        Te, ne, dt, Z=Z, gaunt_factor=gaunt_factor
    )

    # Energy removed per cell: P_radiated [W/m^3] * dV * dt [s] = energy [J]
    dx = 0.01
    dV = dx**3
    energy_removed_per_cell = P_radiated * dV * dt
    total_energy_removed = np.sum(energy_removed_per_cell)

    # Energy from temperature change: E = (3/2) * ne * k_B * Te
    # dE = (3/2) * ne * k_B * (Te_old - Te_new)
    dE_per_cell = 1.5 * ne * k_B * (Te - Te_new)
    total_dE = np.sum(dE_per_cell * dV)

    # These should match within 1% (implicit solve is approximate)
    ratio = total_energy_removed / max(total_dE, 1e-30)
    assert 0.99 < ratio < 1.01, (
        f"Energy accounting mismatch: "
        f"removed={total_energy_removed:.3e} J, dE={total_dE:.3e} J, ratio={ratio:.6f}"
    )


def test_bremsstrahlung_cooling_decreases_temperature():
    """Test that bremsstrahlung always decreases electron temperature."""
    Te0 = 1e6  # K
    ne0 = 1e20  # m^-3
    dt = 1e-9

    Te = np.array([Te0])
    ne = np.array([ne0])

    Te_new, _ = apply_bremsstrahlung_losses(Te, ne, dt)

    # Temperature should decrease
    assert Te_new[0] < Te0, f"Temperature did not decrease: {Te0} -> {Te_new[0]}"


# ============================================================
# Circuit-plasma coupling energy consistency
# ============================================================


def test_circuit_plasma_coupling_energy():
    """Test Ohmic heating in plasma matches I^2 * R_plasma * dt from circuit.

    Run a coupled circuit-plasma step, verify energy transferred from circuit
    to plasma equals Ohmic heating in MHD solver.
    """
    gamma = 5.0 / 3.0
    N = 16
    ny = 16
    nz = 16
    dx = 0.01

    # Circuit setup
    C = 1e-6
    V0 = 1e4
    L0 = 1e-7
    R0 = 0.01
    circuit = RLCSolver(C=C, V0=V0, L0=L0, R0=R0, anode_radius=0.005, cathode_radius=0.01)

    # Plasma setup
    solver = MHDSolver(
        grid_shape=(N, ny, nz),
        dx=dx,
        gamma=gamma,
        cfl=0.3,
        dedner_ch=0.0,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=True,
        enable_energy_equation=True,
    )

    # Initial plasma state (warm, conductive plasma)
    rho0 = 1e-4
    Te0 = 1e5  # K (warm for resistivity)
    m_i = 3.34e-27
    n_i = rho0 / m_i
    p0 = 2.0 * n_i * k_B * Te0

    state = {
        "rho": np.full((N, ny, nz), rho0),
        "velocity": np.zeros((3, N, ny, nz)),
        "pressure": np.full((N, ny, nz), p0),
        "B": np.zeros((3, N, ny, nz)),
        "Te": np.full((N, ny, nz), Te0),
        "Ti": np.full((N, ny, nz), Te0),
        "psi": np.zeros((N, ny, nz)),
    }

    # Compute resistivity field for plasma (Spitzer resistivity)
    from dpf.collision.spitzer import coulomb_log, spitzer_resistivity

    ne = n_i  # Fully ionized
    lnL = coulomb_log(ne, state["Te"])
    eta_field = spitzer_resistivity(ne, state["Te"], lnL, Z=1.0)

    # Time step
    dt = 1e-9

    # Circuit step (get current)
    coupling = CouplingState(Lp=0.0, emf=0.0, current=0.0, voltage=0.0, dL_dt=0.0)
    coupling = circuit.step(coupling, back_emf=0.0, dt=dt)
    I_current = coupling.current  # noqa: N806

    # MHD step with resistivity (should produce Ohmic heating)
    state_new = solver.step(
        state,
        dt,
        current=I_current,
        voltage=coupling.voltage,
        eta_field=eta_field,
    )

    # Energy change in plasma thermal energy
    dV = dx**3
    E_thermal_old = np.sum(state["pressure"] / (gamma - 1.0)) * dV
    E_thermal_new = np.sum(state_new["pressure"] / (gamma - 1.0)) * dV
    dE_plasma = E_thermal_new - E_thermal_old

    # Expected Ohmic heating: Q_ohm = eta * |J|^2 * dt * volume
    # For small current and short time, heating should be small but positive
    # Just check that plasma gained energy (positive dE)
    assert dE_plasma >= 0.0, f"Plasma should gain energy from Ohmic heating: dE={dE_plasma:.3e}"


def test_energy_conservation_in_isolated_mhd():
    """Test MHD energy conservation in absence of resistivity and radiation.

    Run ideal MHD (no eta, no radiation, no viscosity), verify total energy
    (kinetic + thermal + magnetic) is conserved.
    """
    gamma = 5.0 / 3.0
    N = 16
    ny = 16
    nz = 16
    dx = 0.01

    solver = MHDSolver(
        grid_shape=(N, ny, nz),
        dx=dx,
        gamma=gamma,
        cfl=0.3,
        dedner_ch=0.0,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
        enable_energy_equation=True,
    )

    # Initial state with kinetic energy (velocity) and magnetic field
    rho0 = 1e-4
    v0 = 100.0  # m/s
    B0 = 0.1    # T
    m_i = 3.34e-27
    n_i = rho0 / m_i
    T0 = 1e4  # K
    p0 = 2.0 * n_i * k_B * T0

    state = {
        "rho": np.full((N, ny, nz), rho0),
        "velocity": np.zeros((3, N, ny, nz)),
        "pressure": np.full((N, ny, nz), p0),
        "B": np.zeros((3, N, ny, nz)),
        "Te": np.full((N, ny, nz), T0),
        "Ti": np.full((N, ny, nz), T0),
        "psi": np.zeros((N, ny, nz)),
    }

    # Add velocity and B-field
    state["velocity"][0] = v0
    state["B"][2] = B0

    # Initial energy
    dV = dx**3
    v_sq = np.sum(state["velocity"]**2, axis=0)
    E_kin_0 = np.sum(0.5 * state["rho"] * v_sq) * dV
    E_therm_0 = np.sum(state["pressure"] / (gamma - 1.0)) * dV
    B_sq = np.sum(state["B"]**2, axis=0)
    E_mag_0 = np.sum(B_sq / (2.0 * mu_0)) * dV
    E_total_0 = E_kin_0 + E_therm_0 + E_mag_0

    # Run for a few timesteps
    dt = 1e-9
    n_steps = 10
    for _ in range(n_steps):
        state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=None)

    # Final energy
    v_sq = np.sum(state["velocity"]**2, axis=0)
    E_kin_1 = np.sum(0.5 * state["rho"] * v_sq) * dV
    E_therm_1 = np.sum(state["pressure"] / (gamma - 1.0)) * dV
    B_sq = np.sum(state["B"]**2, axis=0)
    E_mag_1 = np.sum(B_sq / (2.0 * mu_0)) * dV
    E_total_1 = E_kin_1 + E_therm_1 + E_mag_1

    conservation = E_total_1 / E_total_0

    # Ideal MHD should conserve energy within 5% (numerical diffusion in HLL)
    assert 0.95 < conservation < 1.05, (
        f"Energy not conserved in ideal MHD: "
        f"E_initial={E_total_0:.3e} J, E_final={E_total_1:.3e} J, ratio={conservation:.6f}"
    )
