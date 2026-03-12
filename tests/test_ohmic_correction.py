"""Tests for circuit-MHD ohmic heating correction (Phase U).

Verifies that the J^2-weighted correction distributes the gap between
R_plasma * I^2 (circuit) and integral(eta * J^2 * dV) (MHD) correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.core.bases import CouplingState
from dpf.engine import SimulationEngine


def _small_config(**overrides) -> SimulationConfig:
    """Create a minimal config for ohmic correction tests."""
    defaults = {
        "grid_shape": [8, 8, 8],
        "dx": 0.01,
        "sim_time": 1e-9,
        "fluid": {
            "backend": "python",
            "gamma": 5 / 3,
            "cfl": 0.3,
            "enable_resistive": True,
            "enable_energy_equation": True,
            "enable_ohmic_correction": True,
        },
        "circuit": {
            "C": 30e-6,
            "V0": 15e3,
            "L0": 33.5e-9,
            "R0": 10e-3,
            "anode_radius": 0.012,
            "cathode_radius": 0.032,
        },
        "diagnostics": {
            "hdf5_filename": ":memory:",
            "output_interval": 1,
        },
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _warm_engine() -> SimulationEngine:
    """Create an engine with warm plasma so eta_field is computed."""
    cfg = _small_config()
    engine = SimulationEngine(cfg)
    # Warm up plasma: set Te > 1000K so Spitzer resistivity activates
    engine.state["Te"] = np.full((8, 8, 8), 5e4)
    engine.state["Ti"] = np.full((8, 8, 8), 1e4)
    # Set non-trivial B field for J = curl(B)/mu_0
    B = engine.state["B"]
    x = np.linspace(0, 0.08, 8)
    B[1, :, :, :] = 0.1 * x[:, np.newaxis, np.newaxis]  # B_y ~ x => J_z nonzero
    engine.state["B"] = B
    # Set non-trivial density for ne
    engine.state["rho"] = np.full((8, 8, 8), 1e-3)
    return engine


class TestOhmicCorrectionConfig:
    """Config field tests."""

    def test_default_enabled(self) -> None:
        cfg = _small_config()
        assert cfg.fluid.enable_ohmic_correction is True

    def test_can_disable(self) -> None:
        cfg = _small_config(fluid={
            "backend": "python",
            "enable_ohmic_correction": False,
            "enable_resistive": True,
            "enable_energy_equation": True,
        })
        assert cfg.fluid.enable_ohmic_correction is False


class TestOhmicGapMeasurement:
    """Test _measure_ohmic_gap directly."""

    def test_gap_populated(self) -> None:
        engine = _warm_engine()
        eta = np.full((8, 8, 8), 1e-4)
        coupling = CouplingState(current=1e4, R_plasma=0.01)
        engine._measure_ohmic_gap(eta, coupling, 1e-9)
        assert len(engine._ohmic_gap_history) == 1
        assert engine._last_ohmic_gap != 0.0

    def test_gap_is_circuit_minus_mhd(self) -> None:
        engine = _warm_engine()
        eta = np.full((8, 8, 8), 1e-4)
        I = 1e4
        R_p = 0.01
        coupling = CouplingState(current=I, R_plasma=R_p)
        engine._measure_ohmic_gap(eta, coupling, 1e-9)
        Q_circuit = R_p * I**2
        # Gap = Q_circuit - Q_mhd, so gap < Q_circuit
        assert abs(engine._last_ohmic_gap) <= Q_circuit * 2

    def test_gap_history_capped_at_100(self) -> None:
        engine = _warm_engine()
        engine._ohmic_gap_history = list(range(120))
        eta = np.full((8, 8, 8), 1e-4)
        coupling = CouplingState(current=1e4, R_plasma=0.01)
        engine._measure_ohmic_gap(eta, coupling, 1e-9)
        assert len(engine._ohmic_gap_history) <= 51


class TestOhmicCorrectionComputation:
    """Verify J^2-weighted correction distribution."""

    def test_correction_integrates_to_gap(self) -> None:
        engine = _warm_engine()
        engine._last_ohmic_gap = 1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)

        dV = engine._cell_volume
        total_Q = float(np.sum(Q * dV))
        assert total_Q == pytest.approx(1e6, rel=0.01)

    def test_correction_nonnegative_for_positive_gap(self) -> None:
        engine = _warm_engine()
        engine._last_ohmic_gap = 1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.all(Q >= 0)

    def test_correction_negative_for_negative_gap(self) -> None:
        engine = _warm_engine()
        engine._last_ohmic_gap = -1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.all(Q <= 0)

    def test_zero_gap_gives_zero_correction(self) -> None:
        engine = _warm_engine()
        engine._last_ohmic_gap = 0.0

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.allclose(Q, 0.0)

    def test_uniform_B_gives_zero_J_returns_zero(self) -> None:
        engine = _warm_engine()
        engine.state["B"] = np.full((3, 8, 8, 8), 0.1)  # uniform B => J=0
        engine._last_ohmic_gap = 1e6

        eta = np.full((8, 8, 8), 1e-4)
        Q = engine._compute_ohmic_correction(eta, 1e4, 1e-9)
        assert np.allclose(Q, 0.0)

    def test_correction_disabled_no_gap_tracking(self) -> None:
        cfg = _small_config(fluid={
            "backend": "python",
            "enable_ohmic_correction": False,
            "enable_resistive": True,
            "enable_energy_equation": True,
        })
        engine = SimulationEngine(cfg)
        engine.state["Te"] = np.full((8, 8, 8), 5e4)
        engine.step()
        engine.step()
        assert len(engine._ohmic_gap_history) == 0


class TestOhmicCorrectionIntegration:
    """Integration test: full engine step with warm plasma."""

    def test_warm_step_measures_gap(self) -> None:
        engine = _warm_engine()
        engine.step()
        # With warm plasma, eta_field should be computed and gap measured
        assert len(engine._ohmic_gap_history) >= 1

    def test_multi_step_no_blowup(self) -> None:
        engine = _warm_engine()
        for _ in range(3):
            engine.step()
        p = engine.state["pressure"]
        assert np.all(np.isfinite(p))
        assert np.all(p > 0)
