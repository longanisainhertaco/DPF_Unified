"""Tests for plasma regime classifier."""

from __future__ import annotations

from dpf.diagnostics.regime_classifier import classify_regime


class TestRegimeClassifier:
    def test_dpf_pinch_conditions(self):
        """DPF pinch: n_e=1e24, 1 keV, 50 T, L=5 mm.

        At these conditions, ion MFP ~ 23 mm > L=5 mm → Kn > 1.
        This is physically correct: DPF pinch IS kinetic, which is why
        beam-target neutrons dominate over thermonuclear.
        At higher density (1e25), MHD becomes valid.
        """
        r = classify_regime(
            n_e=1e24, T_e_eV=1000.0, B_T=50.0, L_m=0.005,
        )
        assert r.lundquist_S > 1
        assert r.knudsen > 1.0  # Kinetic regime — correct!
        assert r.kinetic_needed

    def test_low_density_kinetic(self):
        """Very low density should flag kinetic regime."""
        r = classify_regime(
            n_e=1e16, T_e_eV=100.0, B_T=0.1, L_m=0.01,
        )
        # At low density, MFP is long → Kn >> 1
        assert r.knudsen > 0.01
        assert r.kinetic_needed

    def test_high_density_large_scale_mhd(self):
        """Large scale, very high density should be valid MHD."""
        r = classify_regime(
            n_e=1e26, T_e_eV=100.0, B_T=100.0, L_m=0.05,
        )
        assert r.mhd_valid
        assert not r.kinetic_needed

    def test_lundquist_increases_with_size(self):
        """Larger devices have higher Lundquist number."""
        r_small = classify_regime(n_e=1e23, T_e_eV=500, B_T=10, L_m=0.001)
        r_large = classify_regime(n_e=1e23, T_e_eV=500, B_T=10, L_m=0.01)
        assert r_large.lundquist_S > r_small.lundquist_S

    def test_hall_parameter(self):
        """Hall parameter should depend on B and collision rate."""
        r = classify_regime(n_e=1e23, T_e_eV=1000, B_T=50, L_m=0.005)
        assert r.hall_parameter > 0

    def test_summary_not_empty(self):
        r = classify_regime(n_e=1e24, T_e_eV=1000, B_T=50, L_m=0.005)
        assert len(r.regime_summary) > 10

    def test_zero_B_handles(self):
        """Zero B-field should not crash."""
        r = classify_regime(n_e=1e23, T_e_eV=100, B_T=0.0, L_m=0.01)
        assert r.beta > 0
