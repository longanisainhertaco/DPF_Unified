"""Tests for energy balance tracking."""

from __future__ import annotations

import numpy as np

from dpf.diagnostics.energy_balance import EnergyReport, EnergyTracker


class TestEnergyTracker:
    def _make_state(self, rho=1e-3, p=1e5, v_mag=0.0, B_mag=0.0):
        nr, ny, nz = 8, 1, 16
        state = {
            "rho": np.ones((nr, ny, nz)) * rho,
            "pressure": np.ones((nr, ny, nz)) * p,
            "velocity": np.ones((3, nr, ny, nz)) * v_mag / np.sqrt(3),
            "B": np.ones((3, nr, ny, nz)) * B_mag / np.sqrt(3),
        }
        return state

    def test_thermal_energy(self):
        tracker = EnergyTracker(gamma=5.0 / 3.0)
        state = self._make_state(p=1e5)
        snap = tracker.compute_energies(state, cell_volume=1e-6)
        # E_th = p/(gamma-1) * V * N_cells = 1e5/0.667 * 1e-6 * 128
        assert snap.E_thermal > 0

    def test_kinetic_energy(self):
        tracker = EnergyTracker()
        state = self._make_state(rho=1e-3, v_mag=1e4)
        snap = tracker.compute_energies(state, cell_volume=1e-6)
        assert snap.E_kinetic > 0

    def test_magnetic_energy(self):
        tracker = EnergyTracker()
        state = self._make_state(B_mag=1.0)
        snap = tracker.compute_energies(state, cell_volume=1e-6)
        assert snap.E_magnetic > 0

    def test_conservation_uniform(self):
        """Uniform state with no sources should conserve energy."""
        tracker = EnergyTracker()
        state = self._make_state(p=1e5)
        for i in range(5):
            tracker.record(state, t=i * 1e-9, dt=1e-9, cell_volume=1e-6)
        report = tracker.get_report()
        assert report.max_conservation_error < 1e-10
        assert report.is_conserved

    def test_radiation_increases_error(self):
        """Radiation without compensating pressure drop breaks conservation."""
        tracker = EnergyTracker()
        state = self._make_state(p=1e5)
        tracker.record(state, t=0, dt=1e-9, cell_volume=1e-6)
        # Add radiation without changing state → conservation error
        tracker.record(state, t=1e-9, dt=1e-9, cell_volume=1e-6,
                       radiated_power=1e10)
        report = tracker.get_report()
        assert report.conservation_error[-1] > 0

    def test_circuit_energy(self):
        tracker = EnergyTracker()
        state = self._make_state(p=1e5)
        snap = tracker.compute_energies(
            state, cell_volume=1e-6,
            C=1e-3, V_cap=27e3, L_total=100e-9, I_current=1e6,
        )
        assert snap.E_circuit > 0
        # 0.5*C*V^2 = 0.5*1e-3*27e3^2 = 364.5 kJ
        assert snap.E_circuit > 300e3

    def test_summary(self):
        tracker = EnergyTracker()
        state = self._make_state(p=1e5)
        tracker.record(state, t=0, dt=1e-9, cell_volume=1e-6)
        s = tracker.summary()
        assert "E_total" in s
        assert "PASS" in s


class TestEnergyReport:
    def test_empty_report(self):
        r = EnergyReport()
        assert r.max_conservation_error == 0.0
        assert r.is_conserved

    def test_conserved_threshold(self):
        r = EnergyReport()
        r.conservation_error = [0.01, 0.03, 0.02]
        assert r.is_conserved  # max 3% < 5%

    def test_not_conserved(self):
        r = EnergyReport()
        r.conservation_error = [0.01, 0.06, 0.10]
        assert not r.is_conserved  # max 10% > 5%
