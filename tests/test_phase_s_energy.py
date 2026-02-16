"""Phase S: Energy conservation audit across the DPF simulation pipeline.

Comprehensive energy tests for:
1. Circuit: E_cap + E_ind + E_res = E_initial = 0.5*C*V0^2
2. MHD: Uniform state should have zero energy change (no fluxes)
3. Circuit-MHD coupling: Energy split between external and plasma dissipation

References
----------
- Energy conservation in DPF circuits: Mather, Phys. Fluids 8, 366 (1965).
- Scholz et al., Nukleonika 51(1), 2006: PF-1000 reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.circuit.rlc_solver import CircuitState, RLCSolver
from dpf.constants import mu_0
from dpf.core.bases import CouplingState

# ── PF-1000 circuit parameters ──
PF1000_C = 1.332e-3
PF1000_V0 = 27e3
PF1000_L0 = 33.5e-9
PF1000_R0 = 2.3e-3
E_INITIAL_PF1000 = 0.5 * PF1000_C * PF1000_V0**2  # ~485.6 kJ

T_QUARTER = (np.pi / 2) * np.sqrt(PF1000_L0 * PF1000_C)


def _step_circuit(
    solver: RLCSolver,
    n_steps: int,
    dt: float,
    coupling: CouplingState | None = None,
) -> None:
    """Step the circuit solver n_steps times."""
    if coupling is None:
        coupling = CouplingState()
    for _ in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)


# ═══════════════════════════════════════════════════════
# Circuit energy accounting: field presence
# ═══════════════════════════════════════════════════════


class TestEnergyAccountingFields:
    """Verify CircuitState has all required energy tracking fields."""

    def test_circuit_state_has_energy_fields(self):
        """CircuitState must have energy_cap, energy_ind, energy_res."""
        state = CircuitState()
        assert hasattr(state, "energy_cap")
        assert hasattr(state, "energy_ind")
        assert hasattr(state, "energy_res")

    def test_circuit_state_has_plasma_dissipation(self):
        """CircuitState must track plasma Ohmic dissipation separately."""
        state = CircuitState()
        assert hasattr(state, "energy_res_plasma"), (
            "CircuitState missing energy_res_plasma — "
            "needed for circuit-MHD coupling energy audit"
        )

    def test_total_energy_excludes_plasma(self):
        """RLCSolver.total_energy() should NOT include energy_res_plasma.

        The MHD solver handles plasma Ohmic heating spatially.
        Including it in total_energy() would double-count.
        """
        solver = RLCSolver(C=1e-6, V0=1e3, L0=1e-7, R0=0.01)

        # Manually set plasma dissipation
        solver.state.energy_res_plasma = 100.0

        E_total = solver.total_energy()
        # total_energy = energy_cap + energy_ind + energy_res (external only)
        E_expected = (
            solver.state.energy_cap
            + solver.state.energy_ind
            + solver.state.energy_res
        )
        assert E_total == pytest.approx(E_expected, rel=1e-12)
        # And does NOT include plasma dissipation
        assert E_total != pytest.approx(E_expected + 100.0, rel=0.01)


# ═══════════════════════════════════════════════════════
# Circuit initial energy
# ═══════════════════════════════════════════════════════


class TestCircuitInitialEnergy:
    """Verify initial energy is correctly set."""

    def test_pf1000_initial_energy(self):
        """E_initial = 0.5*C*V0^2 ~ 485.6 kJ for PF-1000 at 27 kV."""
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)

        assert solver.state.energy_cap == pytest.approx(E_INITIAL_PF1000, rel=1e-10)
        assert solver.state.energy_ind == pytest.approx(0.0, abs=1e-30)
        assert solver.state.energy_res == pytest.approx(0.0, abs=1e-30)
        assert solver.state.energy_res_plasma == pytest.approx(0.0, abs=1e-30)

    def test_total_energy_equals_initial_cap(self):
        """At t=0, total energy should equal initial capacitor energy."""
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)

        assert solver.total_energy() == pytest.approx(E_INITIAL_PF1000, rel=1e-10)

    @pytest.mark.parametrize("V0", [1e3, 5e3, 14e3, 22e3, 27e3])
    def test_initial_energy_scales_as_v_squared(self, V0: float):
        """E_initial should scale as V0^2 for fixed C."""
        C = 1e-6
        solver = RLCSolver(C=C, V0=V0, L0=1e-7, R0=0.01)
        E_expected = 0.5 * C * V0**2
        assert solver.state.energy_cap == pytest.approx(E_expected, rel=1e-10)


# ═══════════════════════════════════════════════════════
# Circuit energy conservation: undamped (R=0)
# ═══════════════════════════════════════════════════════


class TestCircuitEnergyUndamped:
    """Energy conservation for ideal LC circuit (R=0)."""

    def test_energy_conservation_one_period(self):
        """E_cap + E_ind should remain constant over one full period."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0)
        E_init = solver.total_energy()

        # Run for one full period (4 quarter-periods)
        _step_circuit(solver, n_steps=20000, dt=dt)

        E_final = solver.total_energy()
        conservation = E_final / E_init

        assert conservation == pytest.approx(1.0, abs=1e-3), (
            f"Undamped energy conservation = {conservation:.8f}"
        )

    def test_energy_oscillates_between_cap_and_ind(self):
        """At T/4, energy should be in inductor; at T/2, back in capacitor."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0)

        # At T/4: current at peak, voltage near zero
        _step_circuit(solver, n_steps=5000, dt=dt)
        E_cap_quarter = solver.state.energy_cap
        E_ind_quarter = solver.state.energy_ind

        # Most energy in inductor at T/4
        assert E_ind_quarter > 0.9 * E_INITIAL_PF1000, (
            f"At T/4: E_ind = {E_ind_quarter:.0f} J, "
            f"expected > 90% of {E_INITIAL_PF1000:.0f} J"
        )
        assert E_cap_quarter < 0.1 * E_INITIAL_PF1000, (
            f"At T/4: E_cap = {E_cap_quarter:.0f} J, "
            f"expected < 10% of {E_INITIAL_PF1000:.0f} J"
        )

        # Continue to T/2: energy back in capacitor
        _step_circuit(solver, n_steps=5000, dt=dt)
        E_cap_half = solver.state.energy_cap
        E_ind_half = solver.state.energy_ind

        # Most energy back in capacitor at T/2
        assert E_cap_half > 0.9 * E_INITIAL_PF1000, (
            f"At T/2: E_cap = {E_cap_half:.0f} J, "
            f"expected > 90% of {E_INITIAL_PF1000:.0f} J"
        )
        assert E_ind_half < 0.1 * E_INITIAL_PF1000, (
            f"At T/2: E_ind = {E_ind_half:.0f} J, "
            f"expected < 10% of {E_INITIAL_PF1000:.0f} J"
        )

    def test_no_resistive_dissipation_when_r_zero(self):
        """With R=0, energy_res should remain exactly zero."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=0.0)

        _step_circuit(solver, n_steps=20000, dt=dt)

        assert solver.state.energy_res == pytest.approx(0.0, abs=1e-30), (
            f"energy_res = {solver.state.energy_res} J, expected 0 for R=0"
        )


# ═══════════════════════════════════════════════════════
# Circuit energy conservation: damped (R>0)
# ═══════════════════════════════════════════════════════


class TestCircuitEnergyDamped:
    """Energy conservation for damped RLC circuit (R>0)."""

    def test_damped_total_energy_conserved(self):
        """E_cap + E_ind + E_res_ext should equal E_initial."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)
        E_init = solver.total_energy()

        # Run for 2 full periods
        _step_circuit(solver, n_steps=40000, dt=dt)

        E_final = solver.total_energy()
        conservation = E_final / E_init

        # 2nd-order implicit midpoint: should hold within 1%
        assert conservation == pytest.approx(1.0, abs=0.01), (
            f"Damped energy conservation = {conservation:.6f}, "
            f"E_cap={solver.state.energy_cap:.0f}, "
            f"E_ind={solver.state.energy_ind:.0f}, "
            f"E_res={solver.state.energy_res:.0f}"
        )

    def test_resistive_dissipation_positive(self):
        """Resistive energy dissipation should be positive and growing."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)

        _step_circuit(solver, n_steps=20000, dt=dt)

        assert solver.state.energy_res > 0, (
            f"energy_res = {solver.state.energy_res} J, expected > 0"
        )

    def test_cap_plus_ind_decreases_with_damping(self):
        """E_cap + E_ind should decrease over time (energy going to resistance)."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)

        # After one quarter period
        _step_circuit(solver, n_steps=5000, dt=dt)
        E_reactive_1 = solver.state.energy_cap + solver.state.energy_ind

        # After one more quarter period
        _step_circuit(solver, n_steps=5000, dt=dt)
        E_reactive_2 = solver.state.energy_cap + solver.state.energy_ind

        # Reactive energy decreases as resistor dissipates
        assert E_reactive_2 < E_reactive_1 * 0.999, (
            f"Reactive energy not decreasing: "
            f"E1={E_reactive_1:.0f}, E2={E_reactive_2:.0f}"
        )


# ═══════════════════════════════════════════════════════
# Energy split: external vs plasma dissipation (Phase R.1)
# ═══════════════════════════════════════════════════════


class TestEnergySplitSeparate:
    """Verify external and plasma Ohmic dissipation are tracked separately."""

    def test_external_only_dissipation(self):
        """With R_plasma=0, only external resistance dissipates energy."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)
        coupling = CouplingState(R_plasma=0.0)

        for _ in range(10000):
            solver.step(coupling, back_emf=0.0, dt=dt)

        assert solver.state.energy_res > 0, "External dissipation should be > 0"
        assert solver.state.energy_res_plasma == pytest.approx(0.0, abs=1e-20), (
            f"Plasma dissipation = {solver.state.energy_res_plasma}, "
            "expected 0 with R_plasma=0"
        )

    def test_plasma_dissipation_tracked(self):
        """With R_plasma>0, plasma Ohmic dissipation should accumulate."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)
        coupling = CouplingState(R_plasma=0.01)  # 10 mOhm plasma

        for _ in range(10000):
            solver.step(coupling, back_emf=0.0, dt=dt)

        assert solver.state.energy_res > 0, "External dissipation should be > 0"
        assert solver.state.energy_res_plasma > 0, (
            "Plasma dissipation should be > 0 with R_plasma=10 mOhm"
        )

    def test_total_dissipation_with_plasma(self):
        """E_cap + E_ind + E_res_ext + E_res_plasma should ~ E_initial."""
        dt = T_QUARTER / 5000
        solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)
        coupling = CouplingState(R_plasma=0.005)  # 5 mOhm plasma resistance
        E_init = solver.total_energy()

        for _ in range(20000):
            solver.step(coupling, back_emf=0.0, dt=dt)

        # total_energy() = E_cap + E_ind + E_res_ext (excludes plasma)
        E_total_ext = solver.total_energy()
        E_all = E_total_ext + solver.state.energy_res_plasma

        conservation = E_all / E_init
        assert conservation == pytest.approx(1.0, abs=0.02), (
            f"Full energy (ext + plasma) conservation = {conservation:.4f}, "
            f"E_ext={E_total_ext:.0f}, E_plasma={solver.state.energy_res_plasma:.0f}"
        )


# ═══════════════════════════════════════════════════════
# MHD energy conservation: uniform state
# ═══════════════════════════════════════════════════════


class TestMHDEnergyUniform:
    """MHD energy conservation for a uniform state (no gradients)."""

    def _compute_mhd_energy(
        self,
        state: dict[str, np.ndarray],
        gamma: float = 5.0 / 3.0,
    ) -> float:
        """Compute total MHD energy: kinetic + magnetic + thermal.

        E_total = sum(0.5*rho*v^2 + B^2/(2*mu_0) + p/(gamma-1)) * dV
        """
        rho = state["rho"]
        vel = state["velocity"]
        B = state["B"]
        p = state["pressure"]

        # Kinetic energy density
        v_sq = np.sum(vel**2, axis=0)  # (nx, ny, nz)
        E_kin = 0.5 * rho * v_sq

        # Magnetic energy density
        B_sq = np.sum(B**2, axis=0)
        E_mag = B_sq / (2.0 * mu_0)

        # Thermal energy density
        E_therm = p / (gamma - 1.0)

        return float(np.sum(E_kin + E_mag + E_therm))

    def test_uniform_state_zero_energy_change(self):
        """Uniform MHD state should have zero energy change (no fluxes)."""
        from dpf.fluid.mhd_solver import MHDSolver

        nx, ny, nz = 8, 8, 8
        dx = 1e-2
        gamma = 5.0 / 3.0

        solver = MHDSolver(
            grid_shape=(nx, ny, nz),
            dx=dx,
            gamma=gamma,
        )

        # Uniform state
        state = {
            "rho": np.full((nx, ny, nz), 1e-4),
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.full((nx, ny, nz), 1.0),
            "B": np.zeros((3, nx, ny, nz)),
            "Te": np.full((nx, ny, nz), 300.0),
            "Ti": np.full((nx, ny, nz), 300.0),
            "psi": np.zeros((nx, ny, nz)),
        }

        E_before = self._compute_mhd_energy(state, gamma)

        # One MHD step
        dt = 1e-10
        state_after = solver.step(state, dt, current=0.0, voltage=0.0)

        E_after = self._compute_mhd_energy(state_after, gamma)

        # Energy should not change for uniform state
        if E_before > 0:
            rel_change = abs(E_after - E_before) / E_before
            assert rel_change < 1e-10, (
                f"MHD energy changed by {rel_change:.2e} for uniform state, "
                f"expected < 1e-10"
            )
        else:
            assert E_after == pytest.approx(E_before, abs=1e-20)

    def test_uniform_state_with_magnetic_field(self):
        """Uniform B-field should not change MHD energy (no curl -> no J)."""
        from dpf.fluid.mhd_solver import MHDSolver

        nx, ny, nz = 8, 8, 8
        dx = 1e-2
        gamma = 5.0 / 3.0

        solver = MHDSolver(
            grid_shape=(nx, ny, nz),
            dx=dx,
            gamma=gamma,
        )

        # Uniform state with uniform B_z = 1 T
        B = np.zeros((3, nx, ny, nz))
        B[2, :, :, :] = 1.0  # Bz = 1 T

        state = {
            "rho": np.full((nx, ny, nz), 1e-4),
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.full((nx, ny, nz), 1.0),
            "B": B,
            "Te": np.full((nx, ny, nz), 300.0),
            "Ti": np.full((nx, ny, nz), 300.0),
            "psi": np.zeros((nx, ny, nz)),
        }

        E_before = self._compute_mhd_energy(state, gamma)
        state_after = solver.step(state, 1e-10, current=0.0, voltage=0.0)
        E_after = self._compute_mhd_energy(state_after, gamma)

        if E_before > 0:
            rel_change = abs(E_after - E_before) / E_before
            assert rel_change < 1e-6, (
                f"Uniform B-field energy changed by {rel_change:.2e}"
            )


# ═══════════════════════════════════════════════════════
# Parametric energy conservation
# ═══════════════════════════════════════════════════════


class TestParametricEnergyConservation:
    """Energy conservation across different circuit parameters."""

    @pytest.mark.parametrize(
        "C, V0, L0, R0",
        [
            (1e-6, 1e3, 1e-7, 0.0),       # Tutorial (undamped)
            (1e-6, 1e3, 1e-7, 0.01),       # Tutorial (damped)
            (28e-6, 14e3, 20e-9, 5e-3),    # NX2-like
            (PF1000_C, PF1000_V0, PF1000_L0, PF1000_R0),  # PF-1000
        ],
        ids=["tutorial_undamped", "tutorial_damped", "nx2_like", "pf1000"],
    )
    def test_energy_conservation_parametric(
        self, C: float, V0: float, L0: float, R0: float
    ):
        """Energy conservation holds across various circuit configs."""
        solver = RLCSolver(C=C, V0=V0, L0=L0, R0=R0)
        E_init = solver.total_energy()

        T_q = (np.pi / 2) * np.sqrt(L0 * C)
        dt = T_q / 2000
        n_steps = 8000  # ~4 quarter periods

        _step_circuit(solver, n_steps=n_steps, dt=dt)

        E_final = solver.total_energy()
        conservation = E_final / E_init

        assert conservation == pytest.approx(1.0, abs=0.02), (
            f"Energy conservation = {conservation:.6f} for "
            f"C={C:.1e}, V0={V0:.0f}, L0={L0:.1e}, R0={R0:.1e}"
        )
