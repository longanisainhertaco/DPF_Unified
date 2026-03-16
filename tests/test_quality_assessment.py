"""Tests for simulation quality assessment."""

from __future__ import annotations

from dpf.validation.quality_assessment import assess_quality


class TestQualityAssessment:
    def _good_result(self) -> dict:
        return {
            "I_peak": 1.733,
            "t_peak": 5.8,
            "dip_pct": 15.0,
            "n_steps": 500,
            "has_snowplow": True,
            "has_mhd": False,
            "bennett": {"T_bennett_keV": 0.48},
            "neutron_yield": {"Y_neutron": 1e8, "bt_fraction": 0.6},
            "breakdown": {"mechanism": "Paschen", "civ_ratio": 11020.0},
        }

    def test_good_result_high_grade(self):
        qa = assess_quality(self._good_result())
        assert qa.grade in ("A", "B")
        assert qa.score > 0.5
        assert qa.n_passed >= 4

    def test_empty_result_f_grade(self):
        qa = assess_quality({"I_peak": 0, "n_steps": 0})
        assert qa.grade in ("D", "F")
        assert qa.n_critical_failures > 0

    def test_no_dip_warning(self):
        r = self._good_result()
        r["dip_pct"] = 0
        qa = assess_quality(r)
        dip_check = [c for c in qa.checks if c.name == "Current dip"]
        assert len(dip_check) == 1
        assert not dip_check[0].passed

    def test_summary_not_empty(self):
        qa = assess_quality(self._good_result())
        assert len(qa.summary) > 50
        assert "Grade" in qa.summary

    def test_mhd_low_compression(self):
        import numpy as np
        r = {
            "I_peak": 1.0, "n_steps": 200,
            "has_mhd": True, "has_snowplow": False,
            "rho_max": np.array([1.5e-4]), "rho0": 1e-4,
        }
        qa = assess_quality(r)
        comp_check = [c for c in qa.checks if c.name == "Density compression"]
        assert len(comp_check) == 1
        assert not comp_check[0].passed  # 1.5x < 2.0 threshold

    def test_with_regime(self):
        r = self._good_result()
        r["plasma_regime"] = {
            "knudsen": 4.5, "mhd_valid": False,
            "summary": "Kinetic regime", "kinetic_needed": True,
        }
        qa = assess_quality(r)
        regime_check = [c for c in qa.checks if c.name == "Regime validity"]
        assert len(regime_check) == 1
