"""Tests for DPF scaling law diagnostics."""

from __future__ import annotations

from dpf.diagnostics.scaling_laws import compute_scaling, scaling_narrative


class TestScaling:
    def test_pf1000(self):
        """PF-1000 at ~1.7 MA should give ~1e8 Yn from I^4."""
        r = compute_scaling(I_pinch_kA=1700, E_bank_kJ=486, a_mm=115, b_mm=160)
        assert 1e7 < r.Yn_lee_I4 < 1e9
        assert r.device_class == "large"
        assert r.T_bennett_keV > 0.1

    def test_unu_ictp(self):
        """UNU-ICTP at ~170 kA should give much less Yn."""
        r = compute_scaling(I_pinch_kA=170, E_bank_kJ=3, a_mm=9.5, b_mm=32)
        assert r.Yn_lee_I4 < r.Yn_energy_E2 * 100  # Both should be order 1e5-1e6
        assert r.device_class == "small"

    def test_pf400j(self):
        """PF-400J at ~100 kA, 0.3 kJ."""
        r = compute_scaling(I_pinch_kA=100, E_bank_kJ=0.3, a_mm=6, b_mm=15.5)
        assert r.device_class == "small"
        assert r.r_pinch_mm > 0
        assert r.T_bennett_keV > 0

    def test_saturation_flag(self):
        """Large current should trigger saturation."""
        r = compute_scaling(I_pinch_kA=3000, E_bank_kJ=734, a_mm=114, b_mm=200)
        assert r.saturation_flag is True

    def test_no_saturation_small(self):
        r = compute_scaling(I_pinch_kA=200, E_bank_kJ=3, a_mm=10, b_mm=30)
        assert r.saturation_flag is False
        assert r.model_role == "diagnostic_estimate"
        assert r.validation_role == "not_solver_validation"

    def test_higher_I_more_yield(self):
        r1 = compute_scaling(I_pinch_kA=100, E_bank_kJ=1, a_mm=10, b_mm=30)
        r2 = compute_scaling(I_pinch_kA=1000, E_bank_kJ=100, a_mm=100, b_mm=150)
        assert r2.Yn_lee_I4 > r1.Yn_lee_I4

    def test_summary_dict_marks_estimates_not_validation(self):
        r = compute_scaling(I_pinch_kA=1700, E_bank_kJ=486, a_mm=115, b_mm=160)
        summary = r.to_summary_dict()
        assert summary["Yn_I4"] == r.Yn_lee_I4
        assert summary["model_role"] == "diagnostic_estimate"
        assert summary["validation_role"] == "not_solver_validation"
        assert "pinch current" in summary["validity_notes"]["current_input"]
        assert "not as MHD solver validation" in summary["source_basis"]["energy_E2"]

    def test_narrative(self):
        r = compute_scaling(I_pinch_kA=1700, E_bank_kJ=486, a_mm=115, b_mm=160)
        text = scaling_narrative(r)
        assert "Scaling Law" in text
        assert "Bennett" in text
        assert "not solver validation" in text
        assert len(text) > 100
