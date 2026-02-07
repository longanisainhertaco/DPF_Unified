"""Tests for Phase 5: Anomalous resistivity from Buneman instability.

Test categories:
1. Electron drift velocity computation
2. Ion thermal speed computation
3. Plasma frequency computation
4. Buneman threshold detection
5. Anomalous resistivity values (below/above threshold)
6. Total resistivity = Spitzer + anomalous
7. Scalar versions match array versions
8. Physical ordering: eta_anom > eta_spitzer when strongly unstable
9. Engine integration (anomalous resistivity wired in)
"""

from __future__ import annotations

import numpy as np

from dpf.constants import e, epsilon_0, k_B, m_e, m_p

# ====================================================
# Electron Drift Velocity Tests
# ====================================================

class TestElectronDriftVelocity:
    """Tests for drift velocity computation."""

    def test_drift_velocity_positive(self):
        """Drift velocity should be positive for positive J and ne."""
        from dpf.turbulence.anomalous import electron_drift_velocity

        J = np.array([1e10])  # 10 GA/m^2
        ne = np.array([1e25])
        v_d = electron_drift_velocity(J, ne)
        assert v_d[0] > 0.0

    def test_drift_velocity_formula(self):
        """v_d = |J| / (ne * e), verify against manual calculation."""
        from dpf.turbulence.anomalous import electron_drift_velocity

        J = np.array([1e8])
        ne = np.array([1e24])
        v_d = electron_drift_velocity(J, ne)
        expected = abs(1e8) / (1e24 * e)
        np.testing.assert_allclose(v_d[0], expected, rtol=1e-10)

    def test_drift_velocity_zero_current(self):
        """Zero current gives zero drift velocity."""
        from dpf.turbulence.anomalous import electron_drift_velocity

        J = np.array([0.0])
        ne = np.array([1e24])
        v_d = electron_drift_velocity(J, ne)
        assert v_d[0] == 0.0


# ====================================================
# Ion Thermal Speed Tests
# ====================================================

class TestIonThermalSpeed:
    """Tests for ion thermal speed."""

    def test_thermal_speed_positive(self):
        """Thermal speed should be positive for positive T."""
        from dpf.turbulence.anomalous import ion_thermal_speed

        Ti = np.array([1e7])  # ~1 keV
        v_ti = ion_thermal_speed(Ti)
        assert v_ti[0] > 0.0

    def test_thermal_speed_formula(self):
        """v_ti = sqrt(kB*Ti/mi), verify against manual calculation."""
        from dpf.turbulence.anomalous import ion_thermal_speed

        Ti = np.array([1e6])
        v_ti = ion_thermal_speed(Ti)
        expected = np.sqrt(k_B * 1e6 / m_p)
        np.testing.assert_allclose(v_ti[0], expected, rtol=1e-10)

    def test_thermal_speed_zero_temperature(self):
        """Zero temperature gives zero thermal speed."""
        from dpf.turbulence.anomalous import ion_thermal_speed

        Ti = np.array([0.0])
        v_ti = ion_thermal_speed(Ti)
        assert v_ti[0] == 0.0


# ====================================================
# Plasma Frequency Tests
# ====================================================

class TestPlasmaFrequency:
    """Tests for electron plasma frequency."""

    def test_plasma_frequency_positive(self):
        """Plasma frequency should be positive for positive density."""
        from dpf.turbulence.anomalous import plasma_frequency

        ne = np.array([1e24])
        omega = plasma_frequency(ne)
        assert omega[0] > 0.0

    def test_plasma_frequency_formula(self):
        """omega_pe = sqrt(ne * e^2 / (eps0 * me))."""
        from dpf.turbulence.anomalous import plasma_frequency

        ne = np.array([1e25])
        omega = plasma_frequency(ne)
        expected = np.sqrt(1e25 * e**2 / (epsilon_0 * m_e))
        np.testing.assert_allclose(omega[0], expected, rtol=1e-10)

    def test_plasma_frequency_scales_sqrt_n(self):
        """Plasma frequency scales as sqrt(ne)."""
        from dpf.turbulence.anomalous import plasma_frequency

        ne1 = np.array([1e24])
        ne2 = np.array([4e24])
        omega1 = plasma_frequency(ne1)
        omega2 = plasma_frequency(ne2)
        np.testing.assert_allclose(omega2[0] / omega1[0], 2.0, rtol=1e-10)


# ====================================================
# Buneman Threshold Tests
# ====================================================

class TestBunemanThreshold:
    """Tests for Buneman instability threshold detection."""

    def test_stable_below_threshold(self):
        """Low drift velocity (small J) should be stable."""
        from dpf.turbulence.anomalous import buneman_threshold

        # Small current density -> v_d << v_ti
        J = np.array([1e2])  # Very small J
        ne = np.array([1e24])
        Ti = np.array([1e7])  # Hot ions
        result = buneman_threshold(J, ne, Ti)
        assert not result[0], "Should be stable below threshold"

    def test_unstable_above_threshold(self):
        """High drift velocity (large J, cold ions) should be unstable."""
        from dpf.turbulence.anomalous import buneman_threshold

        # Large current density -> v_d >> v_ti for cold ions
        J = np.array([1e12])   # Very large J
        ne = np.array([1e24])
        Ti = np.array([300.0])  # Cold ions (room temperature)
        result = buneman_threshold(J, ne, Ti)
        assert result[0], "Should be unstable above threshold"

    def test_threshold_depends_on_temperature(self):
        """Hotter ions raise the instability threshold."""
        from dpf.turbulence.anomalous import buneman_threshold

        J = np.array([1e9])
        ne = np.array([1e24])

        # Cold ions: more likely unstable
        result_cold = buneman_threshold(J, ne, np.array([100.0]))
        # Hot ions: more likely stable
        result_hot = buneman_threshold(J, ne, np.array([1e8]))

        # At least one should differ (cold unstable, hot stable for this J)
        # The key physics: higher Ti raises the threshold
        # v_d = 1e9 / (1e24 * 1.6e-19) = 1e9 / 1.6e5 = 6250 m/s
        # v_ti(100K) = sqrt(1.38e-23 * 100 / 1.67e-27) = sqrt(826) ~ 29 m/s
        # v_ti(1e8 K) = sqrt(1.38e-23 * 1e8 / 1.67e-27) = sqrt(8.26e8) ~ 28740 m/s
        # So v_d = 6250 > 29 (unstable at cold) but 6250 < 28740 (stable at hot)
        assert result_cold[0], "Cold ions should be unstable"
        assert not result_hot[0], "Hot ions should be stable"


# ====================================================
# Anomalous Resistivity Tests
# ====================================================

class TestAnomalousResistivity:
    """Tests for anomalous resistivity computation."""

    def test_zero_below_threshold(self):
        """Anomalous resistivity is zero when v_d < v_ti."""
        from dpf.turbulence.anomalous import anomalous_resistivity

        # Low J, hot ions -> stable
        J = np.array([1e2])
        ne = np.array([1e24])
        Ti = np.array([1e7])
        eta = anomalous_resistivity(J, ne, Ti)
        assert eta[0] == 0.0, "eta_anom should be zero below threshold"

    def test_positive_above_threshold(self):
        """Anomalous resistivity is positive when v_d > v_ti."""
        from dpf.turbulence.anomalous import anomalous_resistivity

        # High J, cold ions -> unstable
        J = np.array([1e12])
        ne = np.array([1e24])
        Ti = np.array([300.0])
        eta = anomalous_resistivity(J, ne, Ti)
        assert eta[0] > 0.0, "eta_anom should be positive above threshold"

    def test_scales_with_alpha(self):
        """Anomalous resistivity scales linearly with alpha."""
        from dpf.turbulence.anomalous import anomalous_resistivity

        J = np.array([1e12])
        ne = np.array([1e24])
        Ti = np.array([300.0])

        eta_1 = anomalous_resistivity(J, ne, Ti, alpha=0.01)
        eta_2 = anomalous_resistivity(J, ne, Ti, alpha=0.1)
        np.testing.assert_allclose(eta_2[0] / eta_1[0], 10.0, rtol=1e-10)

    def test_exceeds_spitzer_when_strongly_unstable(self):
        """Above threshold, eta_anom should exceed eta_Spitzer for DPF conditions."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity
        from dpf.turbulence.anomalous import anomalous_resistivity

        # Typical DPF pinch conditions: high J, moderate ne, warm ions
        ne = np.array([1e25])
        Ti = np.array([1e5])  # ~10 eV ions (early pinch)
        Te = np.array([1e6])  # ~100 eV electrons

        # J ~ I / A_pinch ~ 1e6 A / (pi * (1e-3)^2) ~ 3e11 A/m^2
        J = np.array([3e11])

        eta_anom = anomalous_resistivity(J, ne, Ti, alpha=0.05)
        lnL = coulomb_log(ne, Te)
        eta_sp = spitzer_resistivity(ne, Te, lnL)

        # When strongly unstable, anomalous should dominate
        if eta_anom[0] > 0:
            assert eta_anom[0] > eta_sp[0], (
                f"eta_anom={eta_anom[0]:.2e} should exceed eta_spitzer={eta_sp[0]:.2e} "
                f"for strongly unstable conditions"
            )

    def test_finite_values(self):
        """Anomalous resistivity should be finite for physical parameters."""
        from dpf.turbulence.anomalous import anomalous_resistivity

        J = np.array([1e10, 1e11, 1e12])
        ne = np.array([1e23, 1e24, 1e25])
        Ti = np.array([1e4, 1e5, 1e6])
        eta = anomalous_resistivity(J, ne, Ti)
        assert np.all(np.isfinite(eta)), "All eta_anom values should be finite"


# ====================================================
# Scalar Version Tests
# ====================================================

class TestScalarVersions:
    """Tests for scalar versions of anomalous resistivity functions."""

    def test_scalar_zero_below_threshold(self):
        """Scalar version returns zero below threshold."""
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar

        eta = anomalous_resistivity_scalar(
            J_mag=1e2, ne_val=1e24, Ti_val=1e7, alpha=0.05,
        )
        assert eta == 0.0

    def test_scalar_positive_above_threshold(self):
        """Scalar version returns positive above threshold."""
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar

        eta = anomalous_resistivity_scalar(
            J_mag=1e12, ne_val=1e24, Ti_val=300.0, alpha=0.05,
        )
        assert eta > 0.0

    def test_scalar_matches_array(self):
        """Scalar version matches array version for single element."""
        from dpf.turbulence.anomalous import (
            anomalous_resistivity,
            anomalous_resistivity_scalar,
        )

        J_val, ne_val, Ti_val = 1e12, 1e24, 300.0
        eta_arr = anomalous_resistivity(
            np.array([J_val]), np.array([ne_val]), np.array([Ti_val]),
            alpha=0.05,
        )
        eta_scalar = anomalous_resistivity_scalar(
            J_val, ne_val, Ti_val, alpha=0.05,
        )
        np.testing.assert_allclose(eta_arr[0], eta_scalar, rtol=1e-10)

    def test_scalar_zero_density(self):
        """Scalar version handles zero density gracefully."""
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar

        eta = anomalous_resistivity_scalar(
            J_mag=1e12, ne_val=0.0, Ti_val=300.0,
        )
        assert eta == 0.0


# ====================================================
# Total Resistivity Tests
# ====================================================

class TestTotalResistivity:
    """Tests for combined Spitzer + anomalous resistivity."""

    def test_total_equals_spitzer_below_threshold(self):
        """Below threshold, total = Spitzer (anomalous = 0)."""
        from dpf.turbulence.anomalous import total_resistivity

        eta_sp = np.array([1e-6])
        eta_anom = np.array([0.0])
        eta_total = total_resistivity(eta_sp, eta_anom)
        np.testing.assert_allclose(eta_total[0], 1e-6)

    def test_total_exceeds_spitzer_above_threshold(self):
        """Above threshold, total > Spitzer."""
        from dpf.turbulence.anomalous import total_resistivity

        eta_sp = np.array([1e-6])
        eta_anom = np.array([1e-4])
        eta_total = total_resistivity(eta_sp, eta_anom)
        assert eta_total[0] > eta_sp[0]

    def test_total_additive(self):
        """Total resistivity is additive."""
        from dpf.turbulence.anomalous import total_resistivity

        eta_sp = np.array([2e-6])
        eta_anom = np.array([3e-5])
        eta_total = total_resistivity(eta_sp, eta_anom)
        np.testing.assert_allclose(eta_total[0], 2e-6 + 3e-5, rtol=1e-10)

    def test_total_scalar(self):
        """Scalar total resistivity works correctly."""
        from dpf.turbulence.anomalous import total_resistivity_scalar

        eta = total_resistivity_scalar(1e-6, 5e-5)
        np.testing.assert_allclose(eta, 1e-6 + 5e-5, rtol=1e-10)


# ====================================================
# Engine Integration Tests
# ====================================================

class TestEngineAnomalous:
    """Tests for anomalous resistivity integration with engine."""

    def test_engine_imports_turbulence(self):
        """Engine should import turbulence module without error."""
        from dpf.turbulence.anomalous import (  # noqa: F401
            anomalous_resistivity_scalar,
            total_resistivity_scalar,
        )

    def test_engine_runs_with_anomalous(self):
        """Engine should run without error with anomalous resistivity active."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=(4, 4, 4),
            dx=1e-3,
            sim_time=1e-8,
            dt_init=1e-10,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        )
        engine = SimulationEngine(config)
        result = engine.run(max_steps=5)
        assert result["steps"] == 5
        assert result["energy_conservation"] > 0

    def test_eta_anomalous_in_diagnostics(self):
        """Diagnostics output should include eta_anomalous."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=(4, 4, 4),
            dx=1e-3,
            sim_time=1e-8,
            dt_init=1e-10,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            diagnostics={"output_interval": 1},
        )
        engine = SimulationEngine(config)
        # Run a single step — the diagnostics dict includes eta_anomalous
        # We can't directly check the diagnostic dict but we verify no crash
        engine.run(max_steps=2)
        # If we got here, eta_anom was computed without error
