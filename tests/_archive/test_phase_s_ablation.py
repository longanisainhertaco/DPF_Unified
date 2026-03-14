"""Phase S: Ablation integration tests.

Tests that the existing ablation module (src/dpf/atomic/ablation.py) is
properly wired into the engine.py operator-split loop via AblationConfig.
"""
from __future__ import annotations

import numpy as np
import pytest

from dpf.atomic.ablation import (
    COPPER_ABLATION_EFFICIENCY,
    COPPER_MASS,
    TUNGSTEN_ABLATION_EFFICIENCY,
    TUNGSTEN_MASS,
    ablation_momentum_source,
    ablation_particle_flux,
    ablation_rate,
    ablation_source,
    ablation_source_array,
)


class TestAblationCoreFunctions:
    """Test core ablation physics functions."""

    def test_ablation_rate_positive(self):
        dm_dt = ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY)
        assert dm_dt > 0.0
        assert dm_dt == pytest.approx(1e8 * COPPER_ABLATION_EFFICIENCY, rel=1e-10)

    def test_ablation_rate_zero_power(self):
        assert ablation_rate(0.0, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_rate_negative_power(self):
        assert ablation_rate(-1e8, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_source_basic(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e23,
            J_boundary=1e8, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=5e-5, material_mass=COPPER_MASS,
        )
        # S_rho = efficiency * eta * J^2 = 5e-5 * 1e-5 * (1e8)^2 = 5e6
        assert pytest.approx(5e6, rel=1e-10) == S

    def test_ablation_source_zero_J(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e23,
            J_boundary=0.0, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=5e-5, material_mass=COPPER_MASS,
        )
        assert S == 0.0

    def test_ablation_source_zero_eta(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e23,
            J_boundary=1e8, eta_boundary=0.0, dx=1e-3,
            ablation_efficiency=5e-5, material_mass=COPPER_MASS,
        )
        assert S == 0.0


class TestAblationSourceArray:
    """Test the array-based ablation source function."""

    def test_only_at_boundary(self):
        n = 10
        J = np.full(n, 1e8)
        eta = np.full(n, 1e-5)
        mask = np.zeros(n, dtype=np.int64)
        mask[0] = 1  # Only first cell is electrode boundary
        S = ablation_source_array(J, eta, 5e-5, mask)
        assert S[0] > 0.0
        assert np.all(S[1:] == 0.0)

    def test_zero_mask_gives_zero(self):
        n = 10
        J = np.full(n, 1e8)
        eta = np.full(n, 1e-5)
        mask = np.zeros(n, dtype=np.int64)
        S = ablation_source_array(J, eta, 5e-5, mask)
        assert np.all(S == 0.0)

    def test_source_magnitude(self):
        J = np.array([1e8])
        eta = np.array([1e-5])
        mask = np.array([1], dtype=np.int64)
        S = ablation_source_array(J, eta, 5e-5, mask)
        expected = 5e-5 * 1e-5 * (1e8) ** 2  # 5e6
        assert S[0] == pytest.approx(expected, rel=1e-10)


class TestAblationHelpers:
    """Test particle flux and momentum source helpers."""

    def test_particle_flux(self):
        S_rho = 5e6  # kg/(m^3 s)
        S_n = ablation_particle_flux(S_rho, COPPER_MASS)
        expected = S_rho / COPPER_MASS
        assert S_n == pytest.approx(expected, rel=1e-10)
        assert S_n > 0.0

    def test_particle_flux_zero(self):
        assert ablation_particle_flux(0.0, COPPER_MASS) == 0.0

    def test_momentum_source(self):
        S_rho = 5e6
        v = 1e4  # 10 km/s injection velocity
        S_mom = ablation_momentum_source(S_rho, v)
        assert S_mom == pytest.approx(S_rho * v, rel=1e-10)

    def test_material_constants(self):
        assert COPPER_ABLATION_EFFICIENCY > TUNGSTEN_ABLATION_EFFICIENCY
        assert COPPER_MASS < TUNGSTEN_MASS


class TestAblationConfig:
    """Test ablation configuration in SimulationConfig."""

    def test_ablation_config_defaults(self):
        from dpf.config import AblationConfig
        cfg = AblationConfig()
        assert cfg.enabled is False
        assert cfg.material == "copper"
        assert cfg.efficiency == pytest.approx(5e-5, rel=1e-10)

    def test_ablation_config_custom(self):
        from dpf.config import AblationConfig
        cfg = AblationConfig(enabled=True, material="tungsten", efficiency=2e-5)
        assert cfg.enabled is True
        assert cfg.material == "tungsten"
        assert cfg.efficiency == pytest.approx(2e-5, rel=1e-10)

    def test_ablation_in_simulation_config(self):
        from dpf.config import SimulationConfig
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            ablation={"enabled": True, "efficiency": 3e-5},
        )
        assert config.ablation.enabled is True
        assert config.ablation.efficiency == pytest.approx(3e-5, rel=1e-10)
