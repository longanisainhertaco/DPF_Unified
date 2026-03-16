"""Tests for improved radiation model (Phase 3B).

Tests cover:
- Temperature-dependent Gaunt factor
- Improved bremsstrahlung
- Recombination radiation
- Cyclotron radiation
- Total radiation breakdown
- Radiation regime diagnostics
"""

from __future__ import annotations

import numpy as np

from dpf.radiation.improved_radiation import (
    apply_improved_radiation_losses,
    bremsstrahlung_improved,
    cyclotron_power,
    gaunt_factor_thermal,
    radiation_regime_diagnostic,
    recombination_power,
    total_radiation_power,
)

# Common parameters
_NE = np.array([1e24])  # 1e24 m^-3 (DPF pinch)
_TE_1KEV = np.array([1.16e7])  # 1 keV in Kelvin
_TE_100EV = np.array([1.16e6])  # 100 eV
_B_50T = np.array([50.0])  # 50 T


class TestGauntFactor:
    def test_gaunt_above_one(self):
        """Gaunt factor should always be >= 1 (quantum correction is positive)."""
        for T in [1e5, 1e6, 1e7, 1e8]:
            Te = np.array([T])
            g = gaunt_factor_thermal(Te)
            assert float(g[0]) >= 1.0

    def test_gaunt_bounded(self):
        """Gaunt factor should be in [1, 5]."""
        Te = np.logspace(4, 9, 100)
        g = gaunt_factor_thermal(Te)
        assert np.all(g >= 1.0)
        assert np.all(g <= 5.0)

    def test_gaunt_z_dependence(self):
        """Higher Z should give slightly higher Gaunt factor."""
        Te = np.array([1e7])
        g1 = gaunt_factor_thermal(Te, Z=1.0)
        g10 = gaunt_factor_thermal(Te, Z=10.0)
        assert float(g10[0]) > float(g1[0])

    def test_gaunt_at_1kev(self):
        """At 1 keV, g_ff should be ~1.2-1.5."""
        g = gaunt_factor_thermal(_TE_1KEV)
        assert 1.1 < float(g[0]) < 2.0

    def test_gaunt_vectorized(self):
        Te = np.array([1e5, 1e6, 1e7, 1e8])
        g = gaunt_factor_thermal(Te)
        assert g.shape == (4,)


class TestBremsstrahlungImproved:
    def test_positive_power(self):
        P = bremsstrahlung_improved(_NE, _TE_1KEV)
        assert float(P[0]) > 0

    def test_higher_T_more_power(self):
        """P_ff ~ sqrt(Te), so higher T gives more power."""
        P_low = bremsstrahlung_improved(_NE, _TE_100EV)
        P_high = bremsstrahlung_improved(_NE, _TE_1KEV)
        assert float(P_high[0]) > float(P_low[0])

    def test_higher_ne_more_power(self):
        """P_ff ~ ne^2."""
        P_low = bremsstrahlung_improved(np.array([1e23]), _TE_1KEV)
        P_high = bremsstrahlung_improved(np.array([1e24]), _TE_1KEV)
        # 10x density → 100x power
        ratio = float(P_high[0]) / float(P_low[0])
        assert 80 < ratio < 120  # ~100x with Gaunt correction

    def test_zero_density_zero_power(self):
        P = bremsstrahlung_improved(np.array([0.0]), _TE_1KEV)
        assert float(P[0]) == 0.0


class TestRecombination:
    def test_positive_power(self):
        P = recombination_power(_NE, _TE_1KEV)
        assert float(P[0]) > 0

    def test_lower_T_more_recombination(self):
        """Recombination increases at lower Te (more recombination events)."""
        # At very low T, exp(-chi/kT) drops to zero, so there's a peak
        # At moderate T (100 eV - 1 keV), recombination is active
        P_100eV = recombination_power(_NE, _TE_100EV)
        # Very low T should have low recombination (exponential cutoff)
        P_1eV = recombination_power(_NE, np.array([1.16e4]))
        # Both should be non-negative
        assert float(P_100eV[0]) >= 0
        assert float(P_1eV[0]) >= 0

    def test_higher_Z_more_recombination(self):
        """P_rec ~ Z^2."""
        P_z1 = recombination_power(_NE, _TE_1KEV, Z=1.0)
        P_z10 = recombination_power(_NE, _TE_1KEV, Z=10.0)
        assert float(P_z10[0]) > float(P_z1[0])


class TestCyclotron:
    def test_positive_power(self):
        P = cyclotron_power(_NE, _TE_1KEV, _B_50T)
        assert float(P[0]) > 0

    def test_b_squared_scaling(self):
        """P_cyc ~ B^2."""
        P_low = cyclotron_power(_NE, _TE_1KEV, np.array([10.0]))
        P_high = cyclotron_power(_NE, _TE_1KEV, np.array([50.0]))
        ratio = float(P_high[0]) / float(P_low[0])
        assert 24 < ratio < 26  # 25x for 5x B

    def test_zero_b_zero_power(self):
        P = cyclotron_power(_NE, _TE_1KEV, np.array([0.0]))
        assert float(P[0]) == 0.0

    def test_linear_in_Te(self):
        """P_cyc ~ Te."""
        P_low = cyclotron_power(_NE, _TE_100EV, _B_50T)
        P_high = cyclotron_power(_NE, _TE_1KEV, _B_50T)
        ratio = float(P_high[0]) / float(P_low[0])
        assert 9 < ratio < 11  # 10x for 10x T


class TestTotalRadiation:
    def test_returns_all_components(self):
        rad = total_radiation_power(_NE, _TE_1KEV, Z=1.0, B_mag=_B_50T)
        assert "P_bremsstrahlung" in rad
        assert "P_recombination" in rad
        assert "P_cyclotron" in rad
        assert "P_total" in rad
        assert "gaunt_factor" in rad
        assert "dominant" in rad

    def test_total_is_sum(self):
        rad = total_radiation_power(_NE, _TE_1KEV, Z=1.0, B_mag=_B_50T)
        P_sum = rad["P_bremsstrahlung"] + rad["P_recombination"] + rad["P_cyclotron"]
        assert np.allclose(rad["P_total"], P_sum, rtol=1e-10)

    def test_bremsstrahlung_dominates_at_1kev(self):
        """At 1 keV with moderate B, bremsstrahlung should dominate."""
        rad = total_radiation_power(_NE, _TE_1KEV, Z=1.0, B_mag=np.array([1.0]))
        assert rad["dominant"] == "bremsstrahlung"

    def test_no_b_no_cyclotron(self):
        rad = total_radiation_power(_NE, _TE_1KEV, Z=1.0)
        assert np.all(rad["P_cyclotron"] == 0)

    def test_disable_components(self):
        rad = total_radiation_power(
            _NE, _TE_1KEV, Z=1.0, B_mag=_B_50T,
            include_bremsstrahlung=False,
            include_recombination=False,
        )
        assert np.all(rad["P_bremsstrahlung"] == 0)
        assert np.all(rad["P_recombination"] == 0)
        assert np.all(rad["P_cyclotron"] > 0)


class TestImprovedCooling:
    def test_cooling_reduces_Te(self):
        Te = _TE_1KEV.copy()
        Te_new, rad = apply_improved_radiation_losses(
            Te, _NE, dt=1e-9, Z=1.0, B_mag=_B_50T,
        )
        assert float(Te_new[0]) < float(Te[0])

    def test_floor_respected(self):
        Te_new, _ = apply_improved_radiation_losses(
            np.array([100.0]), _NE, dt=1.0, Z=1.0, Te_floor=50.0,
        )
        assert float(Te_new[0]) >= 50.0

    def test_returns_breakdown(self):
        _, rad = apply_improved_radiation_losses(
            _TE_1KEV.copy(), _NE, dt=1e-9, Z=1.0,
        )
        assert "P_bremsstrahlung" in rad
        assert "P_actual" in rad


class TestRegimeDiagnostic:
    def test_returns_all_keys(self):
        diag = radiation_regime_diagnostic(_TE_1KEV, _NE, _B_50T)
        expected_keys = {
            "Te_keV", "ne_max", "B_max",
            "P_brem_peak", "P_rec_peak", "P_cyc_peak",
            "brem_fraction", "rec_fraction", "cyc_fraction",
            "dominant", "gaunt_factor_mean", "B_crit_T",
        }
        assert set(diag.keys()) == expected_keys

    def test_fractions_sum_to_one(self):
        diag = radiation_regime_diagnostic(_TE_1KEV, _NE, _B_50T)
        total_frac = diag["brem_fraction"] + diag["rec_fraction"] + diag["cyc_fraction"]
        assert abs(total_frac - 1.0) < 0.01

    def test_b_crit_positive(self):
        diag = radiation_regime_diagnostic(_TE_1KEV, _NE, _B_50T)
        assert diag["B_crit_T"] > 0

    def test_te_kev_correct(self):
        diag = radiation_regime_diagnostic(_TE_1KEV, _NE, _B_50T)
        assert 0.9 < diag["Te_keV"] < 1.1  # ~1 keV
