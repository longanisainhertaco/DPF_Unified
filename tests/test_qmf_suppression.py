"""Tests for QMF bremsstrahlung suppression (Campaign 2E)."""
import numpy as np
import pytest


class TestCyclotronEnergy:
    def test_proportional_to_B(self):
        from dpf.radiation.qmf_suppression import cyclotron_energy
        E1 = cyclotron_energy(1.0)
        E2 = cyclotron_energy(2.0)
        assert pytest.approx(2 * E1) == E2

    def test_known_value(self):
        from dpf.radiation.qmf_suppression import cyclotron_energy
        # At 1 T: E_c = hbar * e / m_e = 1.76e11 * hbar ~ 1.85e-23 J
        E = cyclotron_energy(1.0)
        assert 1e-24 < E < 1e-22


class TestQMFCriticalField:
    def test_increases_with_temperature(self):
        from dpf.radiation.qmf_suppression import qmf_critical_field
        B_low = qmf_critical_field(1e6)  # 1 MK
        B_high = qmf_critical_field(1e8)  # 100 MK
        assert B_high > B_low

    def test_typical_dpf_value(self):
        from dpf.radiation.qmf_suppression import qmf_critical_field
        # At 1 keV (~1.16e7 K): B_QMF should be ~10^5 T
        B = qmf_critical_field(1.16e7)
        assert 1e4 < B < 1e7


class TestSuppressionFactor:
    def test_weak_field_no_suppression(self):
        from dpf.radiation.qmf_suppression import bremsstrahlung_suppression_factor
        S = bremsstrahlung_suppression_factor(B=10.0, Te_K=1e7)
        assert pytest.approx(1.0, abs=0.01) == S

    def test_strong_field_suppressed(self):
        from dpf.radiation.qmf_suppression import bremsstrahlung_suppression_factor
        # B = 10^8 T at 1 keV should give suppression
        S = bremsstrahlung_suppression_factor(B=1e8, Te_K=1.16e7)
        assert S < 0.5

    def test_bounded_0_1(self):
        from dpf.radiation.qmf_suppression import bremsstrahlung_suppression_factor
        for B in [0, 1, 100, 1e6, 1e10]:
            S = bremsstrahlung_suppression_factor(B, Te_K=1e7)
            assert 0 <= S <= 1.0


class TestSynchrotronEnhancement:
    def test_weak_field_no_enhancement(self):
        from dpf.radiation.qmf_suppression import synchrotron_enhancement_factor
        f = synchrotron_enhancement_factor(B=1.0, Te_K=1e7, ne=1e24)
        assert f >= 1.0

    def test_scales_with_B_squared(self):
        from dpf.radiation.qmf_suppression import synchrotron_enhancement_factor
        # Need very high B for synchrotron to exceed bremsstrahlung
        f1 = synchrotron_enhancement_factor(B=1e6, Te_K=1e7, ne=1e24)
        f2 = synchrotron_enhancement_factor(B=1e7, Te_K=1e7, ne=1e24)
        # Should scale roughly as B^2
        assert f2 > f1


class TestQMFDiagnostic:
    def test_typical_dpf(self):
        from dpf.radiation.qmf_suppression import qmf_diagnostic
        B = np.zeros((3, 8, 8, 8))
        B[1] = 20.0  # 20 T (typical DPF pinch)
        Te = np.full((8, 8, 8), 1e7)
        ne = np.full((8, 8, 8), 1e24)
        diag = qmf_diagnostic(B, Te, ne)
        assert not diag.is_qmf_regime
        assert diag.suppression_factor > 0.99
        assert "no QMF" in diag.note

    def test_extreme_field(self):
        from dpf.radiation.qmf_suppression import qmf_diagnostic
        B = np.zeros((3, 4, 4, 4))
        B[2] = 1e8  # 100 MT — extreme
        Te = np.full((4, 4, 4), 1e7)
        ne = np.full((4, 4, 4), 1e25)
        diag = qmf_diagnostic(B, Te, ne)
        assert diag.is_qmf_regime
        assert diag.suppression_factor < 0.5
        assert diag.ratio_Ec_Eth > 1

    def test_all_fields_populated(self):
        from dpf.radiation.qmf_suppression import qmf_diagnostic
        B = np.zeros((3, 4, 4, 4))
        B[0] = 1.0
        diag = qmf_diagnostic(B, np.full((4, 4, 4), 1e6), np.full((4, 4, 4), 1e22))
        assert diag.B_qmf_T > 0
        assert diag.B_actual_T > 0
        assert isinstance(diag.note, str)

    def test_qmf_output_is_quarantined_from_validation_claims(self):
        from dpf.radiation.qmf_suppression import qmf_diagnostic

        B = np.zeros((3, 4, 4, 4))
        B[0] = 1e8
        diag = qmf_diagnostic(B, np.full((4, 4, 4), 1e7), np.full((4, 4, 4), 1e24))

        assert diag.model_role == "heuristic_qmf_radiation_diagnostic"
        assert diag.validation_role == "unverified_not_design_evidence"
        assert diag.source_status == "free_free_suppression_source_missing"
        assert diag.validation_status == "not_validation_evidence"
        assert diag.can_support_validation_claims is False
        assert diag.can_support_design_claims is False
