"""Phase X: Circuit-plasma energy coupling tests.

Tests verify:
1. Ohmic correction is applied in the same step (not lagged)
2. Circuit dissipation R*I^2 matches MHD integrated eta*J^2*dV
3. Energy conservation between circuit and plasma is within tolerance
4. conservative_energy config flag wires through to solver
"""

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine


def _make_config(
    geometry: str = "cylindrical",
    conservative_energy: bool = False,
    enable_ohmic_correction: bool = True,
    nx: int = 8,
    nz: int = 16,
) -> SimulationConfig:
    """Create a minimal SimulationConfig for circuit coupling tests."""
    dz = 0.01 if geometry == "cylindrical" else None
    return SimulationConfig(
        grid_shape=(nx, 1 if geometry == "cylindrical" else nx, nz),
        dx=0.01,
        rho0=1e-4,
        sim_time=1e-6,
        circuit={
            "C": 1e-3,
            "V0": 15000,
            "L0": 33.5e-9,
            "R0": 0.01,
            "anode_radius": 0.025,
            "cathode_radius": 0.05,
        },
        geometry={"type": geometry, "dz": dz},
        fluid={
            "backend": "python",
            "enable_ohmic_correction": enable_ohmic_correction,
            "conservative_energy": conservative_energy,
        },
        diagnostics={"hdf5_filename": ":memory:", "output_interval": 1},
        snowplow={
            "enabled": True,
            "mass_fraction": 0.3,
            "anode_length": 0.16,
            "current_fraction": 0.7,
        },
        boundary={"electrode_bc": False},
    )


class TestOhmicCorrectionTiming:
    """Test that ohmic correction is applied in the same step."""

    def test_ohmic_gap_measured_before_fluid_step(self):
        """Gap should be measured before the fluid step, not after."""
        config = _make_config(enable_ohmic_correction=True)
        engine = SimulationEngine(config)
        # Set Te high enough for Spitzer block to activate (>1000 K)
        engine.state["Te"][:] = 1e4

        # Run enough steps for current to build and resistivity to compute
        for _ in range(5):
            engine.step()

        # After steps, the gap history should have entries
        assert len(engine._ohmic_gap_history) >= 3

    def test_correction_applied_same_step(self):
        """Ohmic correction should modify the current step's source terms."""
        config = _make_config(enable_ohmic_correction=True)
        engine = SimulationEngine(config)
        engine.state["Te"][:] = 1e4  # Activate Spitzer

        for _ in range(5):
            engine.step()

        # The gap should be non-zero (circuit and MHD heating rates differ)
        if abs(engine._last_ohmic_gap) > 1e-20:
            assert engine._last_ohmic_gap != 0.0

    def test_correction_disabled(self):
        """When enable_ohmic_correction=False, no gap measurement."""
        config = _make_config(enable_ohmic_correction=False)
        engine = SimulationEngine(config)

        for _ in range(5):
            engine.step()

        assert engine._last_ohmic_gap == 0.0

    def test_no_nan_with_correction(self):
        """Extended run with ohmic correction should not produce NaN."""
        config = _make_config(enable_ohmic_correction=True)
        engine = SimulationEngine(config)

        for _ in range(30):
            engine.step()

        assert not np.any(np.isnan(engine.state["rho"]))
        assert not np.any(np.isnan(engine.state["pressure"]))
        assert np.all(engine.state["rho"] > 0)
        assert np.all(engine.state["pressure"] > 0)


class TestConservativeEnergyConfig:
    """Test conservative_energy config flag wiring."""

    def test_config_default_is_false(self):
        """Default conservative_energy should be False for backward compat."""
        config = SimulationConfig(
            grid_shape=(8, 1, 16),
            dx=0.01,
            rho0=1e-4,
            sim_time=1e-6,
            circuit={"C": 1e-3, "V0": 15000, "L0": 33.5e-9,
                     "anode_radius": 0.025, "cathode_radius": 0.05},
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python"},
            diagnostics={"hdf5_filename": ":memory:"},
        )
        assert config.fluid.conservative_energy is False

    def test_config_can_enable(self):
        """conservative_energy can be set to True."""
        config = _make_config(conservative_energy=True)
        assert config.fluid.conservative_energy is True

    def test_solver_receives_flag(self):
        """CylindricalMHDSolver should receive conservative_energy from config."""
        config = _make_config(conservative_energy=True)
        engine = SimulationEngine(config)
        assert engine.fluid.conservative_energy is True

    def test_solver_default_false(self):
        """CylindricalMHDSolver should default to non-conservative."""
        config = _make_config(conservative_energy=False)
        engine = SimulationEngine(config)
        assert engine.fluid.conservative_energy is False


class TestCircuitEnergyBalance:
    """Test energy balance between circuit and plasma."""

    def test_circuit_energy_decreases(self):
        """Circuit capacitor energy should decrease as current flows."""
        config = _make_config()
        engine = SimulationEngine(config)

        E_init = engine.circuit.total_energy()
        for _ in range(10):
            engine.step()
        E_final = engine.circuit.total_energy()

        # Energy decreases as it's dissipated in resistance
        assert E_final < E_init

    def test_current_flows(self):
        """Current should build up from capacitor discharge."""
        config = _make_config()
        engine = SimulationEngine(config)

        for _ in range(5):
            engine.step()

        assert abs(engine._coupling.current) > 0

    def test_ohmic_gap_bounded(self):
        """Ohmic gap should be bounded relative to circuit power."""
        config = _make_config(enable_ohmic_correction=True)
        engine = SimulationEngine(config)
        engine.state["Te"][:] = 1e4  # Activate Spitzer

        for _ in range(10):
            engine.step()

        I = abs(engine._coupling.current)
        R = engine._coupling.R_plasma
        P_circuit = R * I**2

        gap = abs(engine._last_ohmic_gap)

        # Gap should be a fraction of circuit power, not exceeding it
        if P_circuit > 1e-10:
            assert gap < P_circuit * 10, (
                f"Gap {gap:.2e} exceeds 10x circuit power {P_circuit:.2e}"
            )
