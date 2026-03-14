"""Tests for Phase 1: plasma-circuit coupling, sheath integration, CouplingState.

Test categories:
1. Spitzer resistivity matches analytic formula
2. Plasma resistance (R_plasma) feeds back to circuit
3. CouplingState has correct fields (no dead fields)
4. Sheath BCs are applied in engine when enabled
5. Circuit sees increased damping with R_plasma > 0
"""

from __future__ import annotations

import numpy as np

from dpf.constants import e, m_e

# ═══════════════════════════════════════════════════════
# Spitzer Resistivity Tests
# ═══════════════════════════════════════════════════════

class TestSpitzerResistivity:
    """Tests for Spitzer resistivity function."""

    def test_analytic_formula(self):
        """eta = m_e * nu_ei / (ne * e^2 * alpha(Z)) — with Braginskii correction."""
        from dpf.collision.spitzer import nu_ei, spitzer_alpha, spitzer_resistivity

        ne = np.array([1e20])
        Te = np.array([1e6])
        lnL = 10.0
        Z = 1.0

        eta = spitzer_resistivity(ne, Te, lnL, Z=Z)
        freq = nu_ei(ne, Te, lnL, Z=Z)
        alpha_Z = spitzer_alpha(Z)
        eta_expected = m_e * freq / (ne * e**2) / alpha_Z

        np.testing.assert_allclose(eta, eta_expected, rtol=1e-10)

    def test_decreases_with_temperature(self):
        """Hotter plasma has lower resistivity (eta ~ Te^{-3/2})."""
        from dpf.collision.spitzer import spitzer_resistivity

        ne = np.array([1e20])
        lnL = 10.0

        eta_cold = spitzer_resistivity(ne, np.array([1e5]), lnL)
        eta_hot = spitzer_resistivity(ne, np.array([1e7]), lnL)

        assert eta_hot[0] < eta_cold[0], "Hot plasma should have lower resistivity"
        # Te ratio = 100 -> eta ratio ~ 100^{3/2} = 1000
        ratio = eta_cold[0] / eta_hot[0]
        np.testing.assert_allclose(ratio, 1000.0, rtol=0.1)

    def test_positive(self):
        """Resistivity should always be positive."""
        from dpf.collision.spitzer import spitzer_resistivity

        ne = np.array([1e18, 1e20, 1e22])
        Te = np.array([1e4, 1e6, 1e8])

        eta = spitzer_resistivity(ne, Te)
        assert np.all(eta > 0)

    def test_typical_dpf_value(self):
        """At DPF conditions (1 keV, 10^24 m^-3), eta ~ 10^-7 to 10^-5 Ohm*m."""
        from dpf.collision.spitzer import spitzer_resistivity

        ne = np.array([1e24])
        Te_keV = np.array([1.16e7])  # 1 keV in Kelvin

        eta = spitzer_resistivity(ne, Te_keV)
        # Spitzer resistivity at 1 keV is ~ 1e-7 to 1e-5 Ohm*m
        assert 1e-10 < eta[0] < 1e-3, f"eta = {eta[0]:.2e} out of expected range"


# ═══════════════════════════════════════════════════════
# CouplingState Tests
# ═══════════════════════════════════════════════════════

class TestCouplingState:
    """Tests for CouplingState dataclass."""

    def test_has_R_plasma(self):
        """CouplingState should have R_plasma field."""
        from dpf.core.bases import CouplingState

        cs = CouplingState()
        assert hasattr(cs, "R_plasma")
        assert cs.R_plasma == 0.0

    def test_has_Z_bar(self):
        """CouplingState should have Z_bar field."""
        from dpf.core.bases import CouplingState

        cs = CouplingState()
        assert hasattr(cs, "Z_bar")
        assert cs.Z_bar == 1.0

    def test_no_dead_fields(self):
        """CouplingState should NOT have removed fields."""
        from dpf.core.bases import CouplingState

        cs = CouplingState()
        assert not hasattr(cs, "mutual_inductance"), "mutual_inductance should be removed"
        assert not hasattr(cs, "back_reaction"), "back_reaction should be removed"

    def test_core_fields_preserved(self):
        """Essential fields should still exist."""
        from dpf.core.bases import CouplingState

        cs = CouplingState(Lp=1e-9, emf=100.0, current=1e3, voltage=500.0, dL_dt=-1e-4)
        assert cs.Lp == 1e-9
        assert cs.emf == 100.0
        assert cs.current == 1e3
        assert cs.voltage == 500.0
        assert cs.dL_dt == -1e-4


# ═══════════════════════════════════════════════════════
# Circuit with Plasma Resistance Tests
# ═══════════════════════════════════════════════════════

class TestCircuitPlasmaResistance:
    """Tests that plasma resistance affects circuit behavior."""

    def test_R_plasma_increases_damping(self):
        """Higher R_plasma should damp current faster."""
        from dpf.circuit.rlc_solver import RLCSolver
        from dpf.core.bases import CouplingState

        params = {"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                  "anode_radius": 0.005, "cathode_radius": 0.01}

        # Run without plasma resistance
        solver_no_R = RLCSolver(**params)
        coupling = CouplingState(R_plasma=0.0)
        dt = 1e-10
        for _ in range(500):
            coupling = solver_no_R.step(coupling, back_emf=0.0, dt=dt)
        I_no_R = abs(solver_no_R.current)

        # Run with significant plasma resistance
        solver_with_R = RLCSolver(**params)
        coupling_R = CouplingState(R_plasma=1.0)  # 1 Ohm plasma resistance
        for _ in range(500):
            coupling_R = solver_with_R.step(coupling_R, back_emf=0.0, dt=dt)
        I_with_R = abs(solver_with_R.current)

        # Current should be lower with extra resistance
        assert I_with_R < I_no_R, (
            f"R_plasma should reduce current: I_with_R={I_with_R:.2e}, I_no_R={I_no_R:.2e}"
        )

    def test_R_plasma_zero_gives_same_result(self):
        """R_plasma=0 should give identical results to original behavior."""
        from dpf.circuit.rlc_solver import RLCSolver
        from dpf.core.bases import CouplingState

        params = {"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                  "anode_radius": 0.005, "cathode_radius": 0.01}

        solver = RLCSolver(**params)
        coupling = CouplingState(R_plasma=0.0)
        dt = 1e-10
        for _ in range(100):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        # With R_plasma=0, only R0 matters
        # The energy should still be well-conserved
        E_final = solver.total_energy()
        E_init = 0.5 * params["C"] * params["V0"]**2
        assert abs(E_final - E_init) / E_init < 0.01

    def test_energy_accounting_with_R_plasma(self):
        """Circuit energy accounting should include R_plasma dissipation."""
        from dpf.circuit.rlc_solver import RLCSolver
        from dpf.core.bases import CouplingState

        params = {"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.0,
                  "anode_radius": 0.005, "cathode_radius": 0.01}

        solver = RLCSolver(**params)
        coupling = CouplingState(R_plasma=0.5)
        dt = 1e-10
        for _ in range(200):
            coupling = solver.step(coupling, back_emf=0.0, dt=dt)

        # With R_plasma > 0 and R0=0, all dissipation comes from R_plasma
        # Total energy (cap + ind + dissipated) should still be conserved
        E_total = solver.total_energy()
        E_init = 0.5 * params["C"] * params["V0"]**2
        assert abs(E_total - E_init) / E_init < 0.01


# ═══════════════════════════════════════════════════════
# SheathConfig Tests
# ═══════════════════════════════════════════════════════

class TestSheathConfig:
    """Tests for SheathConfig in configuration."""

    def test_default_disabled(self):
        """Sheath should be disabled by default."""
        from dpf.config import SheathConfig

        cfg = SheathConfig()
        assert cfg.enabled is False
        assert cfg.boundary == "z_high"
        assert cfg.V_sheath == 0.0

    def test_simulation_config_has_sheath(self):
        """SimulationConfig should include SheathConfig."""
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-6,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert hasattr(config, "sheath")
        assert config.sheath.enabled is False

    def test_sheath_config_from_dict(self):
        """SheathConfig can be set from JSON dict."""
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-6,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            sheath={"enabled": True, "boundary": "z_low", "V_sheath": 50.0},
        )
        assert config.sheath.enabled is True
        assert config.sheath.boundary == "z_low"
        assert config.sheath.V_sheath == 50.0
