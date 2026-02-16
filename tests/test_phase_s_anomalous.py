"""Phase S: Anomalous resistivity tests — threshold models and LHDI.

Tests for the corrected anomalous resistivity module with proper threshold
model naming (ion-acoustic, LHDI, Buneman classic) and backward compatibility.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dpf.constants import e, k_B, m_e, m_p

# Deuterium mass for DPF tests
M_D = 3.34358377e-27


class TestThresholdOrdering:
    """Verify LHDI < ion_acoustic < buneman_classic thresholds."""

    def test_lhdi_lowest_threshold(self):
        """LHDI triggers at lower v_d than ion-acoustic."""
        from dpf.turbulence.anomalous import (
            ion_acoustic_threshold,
            lhdi_threshold,
        )
        ne = np.array([1e23])
        Ti = np.array([1e6])  # 1 MK
        mi = M_D

        # Compute threshold velocities
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        lhdi_factor = (m_e / mi) ** 0.25

        # J that triggers LHDI but NOT ion-acoustic
        # v_d = J / (ne * e)
        # Need: lhdi_factor * v_ti < v_d < v_ti
        v_d_mid = 0.5 * (lhdi_factor + 1.0) * v_ti
        J_mid = np.array([v_d_mid * ne[0] * e])

        lhdi_active = lhdi_threshold(J_mid, ne, Ti, mi)
        ia_active = ion_acoustic_threshold(J_mid, ne, Ti, mi)

        assert lhdi_active[0] is True or lhdi_active[0]
        assert ia_active[0] is False or not ia_active[0]

    def test_buneman_classic_highest_threshold(self):
        """Buneman classic requires v_d > v_te, much higher than v_ti."""
        from dpf.turbulence.anomalous import (
            buneman_classic_threshold,
            ion_acoustic_threshold,
        )
        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        mi = M_D

        # J that triggers ion-acoustic but NOT buneman_classic
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        v_te = np.sqrt(k_B * Te[0] / m_e)
        v_d_mid = 0.5 * (v_ti + v_te)
        J_mid = np.array([v_d_mid * ne[0] * e])

        ia_active = ion_acoustic_threshold(J_mid, ne, Ti, mi)
        bc_active = buneman_classic_threshold(J_mid, ne, Te)

        assert ia_active[0]
        assert not bc_active[0]

    def test_threshold_factor_ordering(self):
        """LHDI factor < 1 < sqrt(m_i/m_e) for all ions."""
        from dpf.turbulence.anomalous import lhdi_factor
        factor = lhdi_factor(M_D)
        assert 0.0 < factor < 1.0
        # For deuterium: (m_e / (2*m_p))^(1/4) ~ 0.129
        assert factor == pytest.approx(0.129, abs=0.005)


class TestLHDIThreshold:
    """Test the LHDI threshold function."""

    def test_lhdi_factor_deuterium(self):
        """LHDI factor for deuterium ~ 0.129."""
        from dpf.turbulence.anomalous import lhdi_factor
        factor = lhdi_factor(M_D)
        expected = (m_e / M_D) ** 0.25
        assert factor == pytest.approx(expected, rel=1e-10)

    def test_lhdi_threshold_above(self):
        """Above LHDI threshold, instability is active."""
        from dpf.turbulence.anomalous import lhdi_threshold
        ne = np.array([1e23])
        Ti = np.array([1e5])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        factor = (m_e / mi) ** 0.25
        # v_d well above threshold
        v_d = 2.0 * factor * v_ti
        J = np.array([v_d * ne[0] * e])
        assert lhdi_threshold(J, ne, Ti, mi)[0]

    def test_lhdi_threshold_below(self):
        """Below LHDI threshold, instability is inactive."""
        from dpf.turbulence.anomalous import lhdi_threshold
        ne = np.array([1e23])
        Ti = np.array([1e5])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        factor = (m_e / mi) ** 0.25
        # v_d well below threshold
        v_d = 0.5 * factor * v_ti
        J = np.array([v_d * ne[0] * e])
        assert not lhdi_threshold(J, ne, Ti, mi)[0]


class TestIonAcousticThreshold:
    """Test the renamed ion-acoustic threshold."""

    def test_above_threshold(self):
        from dpf.turbulence.anomalous import ion_acoustic_threshold
        ne = np.array([1e23])
        Ti = np.array([1e5])
        v_ti = np.sqrt(k_B * Ti[0] / m_p)
        J = np.array([2.0 * v_ti * ne[0] * e])
        assert ion_acoustic_threshold(J, ne, Ti)[0]

    def test_below_threshold(self):
        from dpf.turbulence.anomalous import ion_acoustic_threshold
        ne = np.array([1e23])
        Ti = np.array([1e5])
        v_ti = np.sqrt(k_B * Ti[0] / m_p)
        J = np.array([0.5 * v_ti * ne[0] * e])
        assert not ion_acoustic_threshold(J, ne, Ti)[0]


class TestBunemanClassicThreshold:
    """Test the true Buneman threshold (v_d > v_te)."""

    def test_above_threshold(self):
        from dpf.turbulence.anomalous import buneman_classic_threshold
        ne = np.array([1e23])
        Te = np.array([1e5])
        v_te = np.sqrt(k_B * Te[0] / m_e)
        J = np.array([2.0 * v_te * ne[0] * e])
        assert buneman_classic_threshold(J, ne, Te)[0]

    def test_below_threshold(self):
        from dpf.turbulence.anomalous import buneman_classic_threshold
        ne = np.array([1e23])
        Te = np.array([1e5])
        v_te = np.sqrt(k_B * Te[0] / m_e)
        J = np.array([0.5 * v_te * ne[0] * e])
        assert not buneman_classic_threshold(J, ne, Te)[0]


class TestAnomalousResistivityField:
    """Test the dispatched anomalous_resistivity_field function."""

    def test_default_is_ion_acoustic(self):
        """Default threshold_model is ion_acoustic."""
        from dpf.turbulence.anomalous import anomalous_resistivity_field
        ne = np.array([1e23])
        Ti = np.array([1e6])
        v_ti = np.sqrt(k_B * Ti[0] / m_p)
        J = np.array([2.0 * v_ti * ne[0] * e])
        eta = anomalous_resistivity_field(J, ne, Ti)
        assert eta[0] > 0.0

    def test_ion_acoustic_model(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field
        ne = np.array([1e23])
        Ti = np.array([1e6])
        v_ti = np.sqrt(k_B * Ti[0] / m_p)
        J = np.array([2.0 * v_ti * ne[0] * e])
        eta = anomalous_resistivity_field(
            J, ne, Ti, threshold_model="ion_acoustic",
        )
        assert eta[0] > 0.0

    def test_lhdi_model(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field
        ne = np.array([1e23])
        Ti = np.array([1e6])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        factor = (m_e / mi) ** 0.25
        # J above LHDI but below ion-acoustic
        v_d = 0.5 * (factor + 1.0) * v_ti
        J = np.array([v_d * ne[0] * e])

        eta_lhdi = anomalous_resistivity_field(
            J, ne, Ti, mi=mi, threshold_model="lhdi",
        )
        eta_ia = anomalous_resistivity_field(
            J, ne, Ti, mi=mi, threshold_model="ion_acoustic",
        )
        # LHDI should give nonzero, ion-acoustic should give zero
        assert eta_lhdi[0] > 0.0
        assert eta_ia[0] == 0.0

    def test_buneman_classic_requires_Te(self):
        """buneman_classic should raise ValueError without Te."""
        from dpf.turbulence.anomalous import anomalous_resistivity_field
        ne = np.array([1e23])
        Ti = np.array([1e6])
        J = np.array([1e10])
        with pytest.raises(ValueError, match="Te.*required"):
            anomalous_resistivity_field(
                J, ne, Ti, threshold_model="buneman_classic",
            )

    def test_buneman_classic_model(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field
        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        v_te = np.sqrt(k_B * Te[0] / m_e)
        J = np.array([2.0 * v_te * ne[0] * e])
        eta = anomalous_resistivity_field(
            J, ne, Ti, threshold_model="buneman_classic", Te=Te,
        )
        assert eta[0] > 0.0

    def test_unknown_model_raises(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field
        ne = np.array([1e23])
        Ti = np.array([1e6])
        J = np.array([1e10])
        with pytest.raises(ValueError, match="Unknown threshold_model"):
            anomalous_resistivity_field(
                J, ne, Ti, threshold_model="invalid",
            )

    def test_below_threshold_gives_zero(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field
        ne = np.array([1e23])
        Ti = np.array([1e6])
        # Very small J — below all thresholds
        J = np.array([1.0])
        for model in ("ion_acoustic", "lhdi"):
            eta = anomalous_resistivity_field(J, ne, Ti, threshold_model=model)
            assert eta[0] == 0.0


class TestBackwardCompatibility:
    """Test that the old API still works."""

    def test_buneman_threshold_deprecated(self):
        """buneman_threshold should emit DeprecationWarning."""
        from dpf.turbulence.anomalous import buneman_threshold
        ne = np.array([1e23])
        Ti = np.array([1e5])
        J = np.array([1e10])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            buneman_threshold(J, ne, Ti)
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ion_acoustic" in str(w[0].message).lower()

    def test_anomalous_resistivity_scalar_unchanged(self):
        """Scalar function should still work identically."""
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar
        eta = anomalous_resistivity_scalar(1e10, 1e23, 1e6, alpha=0.05)
        assert isinstance(eta, float)

    def test_total_resistivity_scalar_unchanged(self):
        from dpf.turbulence.anomalous import total_resistivity_scalar
        assert total_resistivity_scalar(1e-6, 1e-5) == pytest.approx(1.1e-5, rel=1e-10)


class TestBraginskiiKappaZDependent:
    """Test the Z-dependent Braginskii kappa coefficient (Task #7)."""

    def test_Z1_coefficient(self):
        from dpf.collision.spitzer import braginskii_kappa_coefficient
        assert braginskii_kappa_coefficient(1.0) == pytest.approx(3.16, rel=1e-10)

    def test_Z2_coefficient(self):
        from dpf.collision.spitzer import braginskii_kappa_coefficient
        assert braginskii_kappa_coefficient(2.0) == pytest.approx(3.14, rel=1e-10)

    def test_Z3_coefficient(self):
        from dpf.collision.spitzer import braginskii_kappa_coefficient
        assert braginskii_kappa_coefficient(3.0) == pytest.approx(3.12, rel=1e-10)

    def test_Z4_coefficient(self):
        from dpf.collision.spitzer import braginskii_kappa_coefficient
        assert braginskii_kappa_coefficient(4.0) == pytest.approx(3.11, rel=1e-10)

    def test_Z_infinity_coefficient(self):
        from dpf.collision.spitzer import braginskii_kappa_coefficient
        assert braginskii_kappa_coefficient(100.0) == pytest.approx(3.21, rel=1e-10)
        assert braginskii_kappa_coefficient(1000.0) == pytest.approx(3.21, rel=1e-10)

    def test_Z_below_1(self):
        """Z < 1 should return Z=1 value."""
        from dpf.collision.spitzer import braginskii_kappa_coefficient
        assert braginskii_kappa_coefficient(0.5) == pytest.approx(3.16, rel=1e-10)

    def test_interpolation_monotonic_descent_to_Z4(self):
        """Coefficient decreases monotonically from Z=1 to Z=4."""
        from dpf.collision.spitzer import braginskii_kappa_coefficient
        z_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        coeffs = [braginskii_kappa_coefficient(z) for z in z_vals]
        for i in range(len(coeffs) - 1):
            assert coeffs[i] >= coeffs[i + 1]

    def test_braginskii_kappa_backward_compat(self):
        """braginskii_kappa with default Z=1.0 should match old behavior."""
        from dpf.collision.spitzer import braginskii_kappa
        ne = np.array([1e23])
        Te = np.array([1e6])
        Bmag = np.array([1.0])
        kpar, kperp = braginskii_kappa(ne, Te, Bmag)
        kpar_z1, kperp_z1 = braginskii_kappa(ne, Te, Bmag, Z=1.0)
        assert kpar[0] == pytest.approx(kpar_z1[0], rel=1e-10)
        assert kperp[0] == pytest.approx(kperp_z1[0], rel=1e-10)

    def test_braginskii_kappa_Z2_differs(self):
        """Z=2 should give slightly different kappa than Z=1."""
        from dpf.collision.spitzer import braginskii_kappa
        ne = np.array([1e23])
        Te = np.array([1e6])
        Bmag = np.array([1.0])
        kpar_z1, _ = braginskii_kappa(ne, Te, Bmag, Z=1.0)
        kpar_z2, _ = braginskii_kappa(ne, Te, Bmag, Z=2.0)
        # 3.14/3.16 ~ 0.994 — about 0.6% difference
        ratio = kpar_z2[0] / kpar_z1[0]
        assert 0.99 < ratio < 1.0
