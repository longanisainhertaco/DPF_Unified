"""Physics verification tests for Spitzer resistivity and Coulomb logarithm.

Tests validate classical Spitzer resistivity formulas against NRL Plasma Formulary
reference values (p. 34). Key physics:

- Spitzer resistivity: eta ~ Z * ln(Lambda) * Te^(-3/2), independent of ne
- Coulomb logarithm: ln(Lambda) = ln(lambda_D / b_min)
- Temperature scaling: eta(10eV) / eta(100eV) ~ 31.6 (from Te^(-3/2))
- NRL formula: eta_perp = 1.03e-2 * Z * ln(Lambda) / Te_eV^(3/2) [Ohm*cm]
                        = 1.03e-4 * Z * ln(Lambda) / Te_eV^(3/2) [Ohm*m]

Note: The NRL coefficient is 1.03e-2 in Ohm*cm (CGS practical).
      Converting to SI: 1 Ohm*cm = 1e-2 Ohm*m, so coefficient = 1.03e-4 Ohm*m.

References:
    NRL Plasma Formulary (2019), p. 34
    Spitzer, L. (1962). Physics of Fully Ionized Gases.
"""

import numpy as np
import pytest

from dpf.collision.spitzer import coulomb_log, nu_ei, spitzer_resistivity
from dpf.constants import e, k_B


class TestCoulombLog:
    """Tests for Coulomb logarithm calculation."""

    def test_coulomb_log_typical_values(self):
        """Coulomb log should be 5-25 for typical DPF conditions."""
        ne = np.array([1e24])
        Te = np.array([1e7])
        # coulomb_log(ne, Te) — only 2 args
        lnL = coulomb_log(ne, Te)
        assert lnL[0] > 5.0, "Coulomb log too low for typical DPF conditions"
        assert lnL[0] < 25.0, "Coulomb log too high for typical DPF conditions"

    def test_coulomb_log_scales_with_density(self):
        """Higher density should decrease Coulomb log (Debye length shrinks)."""
        Te = np.array([1e6])
        lnL_low = coulomb_log(np.array([1e20]), Te)
        lnL_high = coulomb_log(np.array([1e24]), Te)
        assert lnL_low[0] > lnL_high[0], "Coulomb log should decrease with density"
        assert lnL_low[0] / lnL_high[0] < 5.0, "Density scaling too strong"


class TestSpitzerResistivity:
    """Tests for Spitzer resistivity against NRL Plasma Formulary."""

    def test_spitzer_resistivity_vs_nrl_10eV(self):
        """Verify against NRL at Te=10 eV."""
        Te_eV = 10.0
        Te_K = Te_eV * e / k_B
        ne = np.array([1e20])
        Te = np.array([Te_K])

        lnL_val = float(coulomb_log(ne, Te)[0])
        eta_NRL = 1.03e-4 * 1.0 * lnL_val / (Te_eV ** 1.5)
        eta = spitzer_resistivity(ne, Te, lnL_val, 1.0)[0]

        assert eta == pytest.approx(eta_NRL, rel=0.30), (
            f"Spitzer mismatch at 10 eV: computed={eta:.3e}, NRL={eta_NRL:.3e}"
        )

    def test_spitzer_resistivity_vs_nrl_100eV(self):
        """Verify against NRL at Te=100 eV."""
        Te_eV = 100.0
        Te_K = Te_eV * e / k_B
        ne = np.array([1e20])
        Te = np.array([Te_K])

        lnL_val = float(coulomb_log(ne, Te)[0])
        eta_NRL = 1.03e-4 * 1.0 * lnL_val / (Te_eV ** 1.5)
        eta = spitzer_resistivity(ne, Te, lnL_val, 1.0)[0]

        assert eta == pytest.approx(eta_NRL, rel=0.30), (
            f"Spitzer mismatch at 100 eV: computed={eta:.3e}, NRL={eta_NRL:.3e}"
        )

    def test_spitzer_resistivity_vs_nrl_1keV(self):
        """Verify against NRL at Te=1 keV."""
        Te_eV = 1000.0
        Te_K = Te_eV * e / k_B
        ne = np.array([1e20])
        Te = np.array([Te_K])

        lnL_val = float(coulomb_log(ne, Te)[0])
        eta_NRL = 1.03e-4 * 1.0 * lnL_val / (Te_eV ** 1.5)
        eta = spitzer_resistivity(ne, Te, lnL_val, 1.0)[0]

        assert 1e-9 < eta < 5e-6, f"Resistivity out of range at 1 keV: {eta:.3e}"
        assert eta == pytest.approx(eta_NRL, rel=0.30), (
            f"Spitzer mismatch at 1 keV: computed={eta:.3e}, NRL={eta_NRL:.3e}"
        )

    def test_spitzer_temperature_scaling(self):
        """Resistivity should scale as Te^(-3/2) with fixed ln(Lambda)."""
        ne = np.array([1e20])
        eta_10 = spitzer_resistivity(ne, np.array([10.0 * e / k_B]), 10.0, 1.0)[0]
        eta_100 = spitzer_resistivity(ne, np.array([100.0 * e / k_B]), 10.0, 1.0)[0]
        ratio = eta_10 / eta_100
        assert ratio == pytest.approx(10.0 ** 1.5, rel=0.05)

    def test_spitzer_Z_scaling(self):
        """Resistivity should scale linearly with Z at fixed ln(Lambda)."""
        ne = np.array([1e20])
        Te = np.array([1e6])
        eta_Z1 = spitzer_resistivity(ne, Te, 10.0, 1.0)[0]
        eta_Z2 = spitzer_resistivity(ne, Te, 10.0, 2.0)[0]
        assert eta_Z2 / eta_Z1 == pytest.approx(2.0, rel=0.05)

    def test_resistivity_independent_of_density(self):
        """Spitzer resistivity is independent of ne (cancels) with fixed ln(Lambda)."""
        Te = np.array([1e6])
        eta_low = spitzer_resistivity(np.array([1e20]), Te, 10.0, 1.0)[0]
        eta_high = spitzer_resistivity(np.array([1e24]), Te, 10.0, 1.0)[0]
        assert eta_low == pytest.approx(eta_high, rel=1e-10)


class TestCollisionFrequency:
    """Tests for electron-ion collision frequency."""

    def test_nu_ei_positive(self):
        """Collision frequency should always be positive."""
        nu = nu_ei(np.array([1e24]), np.array([1e7]), 10.0, 1.0)
        assert nu[0] > 0
        assert np.isfinite(nu[0])

    def test_nu_ei_scales_with_density(self):
        """nu_ei should scale linearly with ne."""
        Te = np.array([1e6])
        nu_low = nu_ei(np.array([1e20]), Te, 10.0, 1.0)[0]
        nu_high = nu_ei(np.array([1e24]), Te, 10.0, 1.0)[0]
        assert nu_high / nu_low == pytest.approx(1e4, rel=0.01)

    def test_nu_ei_decreases_with_temperature(self):
        """nu_ei should scale as Te^(-3/2)."""
        ne = np.array([1e24])
        nu_low_T = nu_ei(ne, np.array([1e6]), 10.0, 1.0)[0]
        nu_high_T = nu_ei(ne, np.array([1e7]), 10.0, 1.0)[0]
        ratio = nu_low_T / nu_high_T
        assert ratio == pytest.approx(10.0 ** 1.5, rel=0.01)
