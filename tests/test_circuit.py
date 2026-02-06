"""Tests for the RLC circuit solver."""

from __future__ import annotations

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState


class TestRLCSolver:
    """Basic circuit solver tests."""

    def test_init(self, default_circuit_params):
        """Solver initializes with correct state."""
        solver = RLCSolver(**default_circuit_params)
        assert solver.current == 0.0
        assert solver.voltage == default_circuit_params["V0"]
        assert solver.total_energy() > 0

    def test_energy_conservation(self, default_circuit_params):
        """Total energy (cap + ind + resistive) is conserved."""
        solver = RLCSolver(**default_circuit_params)
        E0 = solver.total_energy()

        coupling = CouplingState()
        dt = 1e-10
        for _ in range(1000):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        E_final = solver.total_energy()
        # Energy should be conserved to within 1%
        assert abs(E_final - E0) / E0 < 0.01, f"Energy not conserved: {E_final/E0:.6f}"

    def test_current_increases(self, default_circuit_params):
        """Current should initially increase (capacitor discharging)."""
        solver = RLCSolver(**default_circuit_params)
        coupling = CouplingState()
        dt = 1e-10
        for _ in range(10):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        assert solver.current > 0, "Current should increase during discharge"

    def test_voltage_decreases(self, default_circuit_params):
        """Voltage should decrease as capacitor discharges."""
        solver = RLCSolver(**default_circuit_params)
        V0 = solver.voltage
        coupling = CouplingState()
        dt = 1e-10
        for _ in range(100):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        assert solver.voltage < V0, "Voltage should decrease during discharge"

    def test_coupling_output(self, default_circuit_params):
        """Step returns a valid CouplingState."""
        solver = RLCSolver(**default_circuit_params)
        coupling = CouplingState()
        result = solver.step(coupling, back_emf=0.0, dt=1e-10)

        assert isinstance(result, CouplingState)
        assert result.current == solver.current
        assert result.voltage == solver.voltage

    def test_plasma_inductance(self, default_circuit_params):
        """Plasma inductance estimate is positive and physical."""
        solver = RLCSolver(**default_circuit_params)
        Lp = solver.plasma_inductance_estimate(pinch_radius=0.003, length=0.05)
        assert Lp > 0
        assert Lp < 1e-3  # Should be nanohenries to microhenries

    def test_oscillation_frequency(self, default_circuit_params):
        """Check that oscillation period matches sqrt(LC)."""
        solver = RLCSolver(**default_circuit_params)
        coupling = CouplingState()
        dt = 1e-11

        # Run until current reverses (half period)
        max_current = 0.0
        steps_to_peak = 0
        for step in range(100000):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)
            if solver.current > max_current:
                max_current = solver.current
                steps_to_peak = step

        # Expected quarter-period: T/4 = pi/2 * sqrt(LC)
        L = default_circuit_params["L0"]
        C = default_circuit_params["C"]
        expected_quarter = 0.5 * np.pi * np.sqrt(L * C)
        actual_quarter = steps_to_peak * dt

        # Should match within 20% (finite resistance and dt effects)
        assert abs(actual_quarter - expected_quarter) / expected_quarter < 0.2
