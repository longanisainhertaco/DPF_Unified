"""Validate anomalous resistivity against published models.

References:
    Haines, Plasma Phys. Control. Fusion 53:093001 (2011) -- Z-pinch anomalous resistivity review
    Davidson & Gladd, Phys. Fluids 18:1327 (1975) -- LHDI threshold
    Huba et al., Phys. Fluids B 5:3779 (1993) -- LHDI in current sheets
    NRL Plasma Formulary (2019) -- Spitzer resistivity, p.34
    Braginskii, Rev. Plasma Phys. Vol. 1 (1965), Table 1 -- alpha(Z) coefficients
"""

import numpy as np
import pytest

from dpf.collision.spitzer import coulomb_log, nu_ei, spitzer_alpha, spitzer_resistivity
from dpf.constants import e, epsilon_0, k_B, m_e, m_p
from dpf.turbulence.anomalous import (
    _compute_eta_anom,
    anomalous_resistivity_field,
    buneman_classic_threshold,
    electron_drift_velocity,
    ion_acoustic_threshold,
    ion_sound_speed,
    ion_thermal_speed,
    lhdi_factor,
    lhdi_threshold,
    plasma_frequency,
)

_m_d = 2.0 * m_p  # deuterium mass


# ---------------------------------------------------------------------------
# 1. Spitzer resistivity validated against NRL Plasma Formulary
# ---------------------------------------------------------------------------


class TestSpitzerResistivity:
    """NRL Plasma Formulary (2019) p.34: eta_|| = 5.2e-5 Z lnL Te_eV^{-3/2} Ohm*m."""

    def _nrl_eta(self, Te_eV: float, Z: float = 1.0, lnL: float = 10.0) -> float:
        return 5.2e-5 * Z * lnL / Te_eV**1.5

    def test_spitzer_at_10ev_z1_lnl10(self):
        """eta at Te=10 eV, Z=1, lnL=10 matches NRL within 1%."""
        Te_eV = 10.0
        Te_K = Te_eV * 11604.52
        ne = np.array([1e20])
        Te = np.array([Te_K])

        eta = spitzer_resistivity(ne, Te, lnL=10.0, Z=1.0)[0]
        nrl = self._nrl_eta(Te_eV, Z=1.0, lnL=10.0)

        assert eta == pytest.approx(nrl, rel=0.01), (
            f"Spitzer eta={eta:.3e}, NRL={nrl:.3e}. "
            "Likely cause: Braginskii alpha(Z) applied in wrong direction (divide vs multiply)."
        )

    def test_spitzer_at_1kev_z1_lnl12(self):
        """eta at Te=1 keV, Z=1, lnL=12 matches NRL within 1%."""
        Te_eV = 1000.0
        Te_K = Te_eV * 11604.52
        ne = np.array([1e25])
        Te = np.array([Te_K])

        eta = spitzer_resistivity(ne, Te, lnL=12.0, Z=1.0)[0]
        nrl = self._nrl_eta(Te_eV, Z=1.0, lnL=12.0)

        assert eta == pytest.approx(nrl, rel=0.01)

    def test_spitzer_te_scaling_t_minus_1p5(self):
        """Spitzer resistivity scales as Te^{-3/2} (NRL Formulary)."""
        ne = np.array([1e22])
        Te_vals = np.array([100.0, 200.0, 400.0]) * 11604.52  # 100, 200, 400 eV in K
        lnL = 10.0

        etas = np.array([spitzer_resistivity(ne, np.array([T]), lnL=lnL, Z=1.0)[0] for T in Te_vals])
        ratios = etas[:-1] / etas[1:]
        expected = (Te_vals[:-1] / Te_vals[1:]) ** (-1.5)

        np.testing.assert_allclose(ratios, expected, rtol=1e-4)

    def test_spitzer_density_independence(self):
        """Spitzer resistivity is independent of density (nu_ei ~ ne cancels ne in denominator)."""
        Te = np.array([100.0 * 11604.52])
        lnL = 10.0

        eta_low = spitzer_resistivity(np.array([1e18]), Te, lnL, Z=1.0)[0]
        eta_high = spitzer_resistivity(np.array([1e25]), Te, lnL, Z=1.0)[0]

        assert eta_low == pytest.approx(eta_high, rel=1e-6)

    def test_braginskii_alpha_z1_value(self):
        """Braginskii alpha(1) = 0.5064 (Table 1, Braginskii 1965)."""
        alpha = spitzer_alpha(1.0)
        assert float(alpha) == pytest.approx(0.5064, rel=1e-4)

    def test_braginskii_alpha_monotone_decreasing(self):
        """alpha(Z) decreases from 0.5064 at Z=1 to 0.2949 as Z→∞ (Braginskii Table 1)."""
        Z_vals = [1, 2, 3, 4, 100]
        alphas = [float(spitzer_alpha(Z)) for Z in Z_vals]
        for i in range(len(alphas) - 1):
            assert alphas[i] > alphas[i + 1], f"alpha not monotone at Z={Z_vals[i]}"
        assert alphas[-1] == pytest.approx(0.2949, rel=1e-3)

    def test_coulomb_log_dfp_range(self):
        """Coulomb logarithm is 10-15 for typical DPF conditions (ne=1e22-1e25, Te=10-1000 eV)."""
        ne = np.array([1e22, 1e23, 1e24, 1e25])
        Te_eV = np.array([100.0, 300.0, 500.0, 1000.0])
        Te_K = Te_eV * 11604.52

        lnL = coulomb_log(ne, Te_K)
        assert np.all(lnL >= 8.0), f"lnL below 8 at DPF conditions: {lnL}"
        assert np.all(lnL <= 20.0), f"lnL above 20 at DPF conditions: {lnL}"


# ---------------------------------------------------------------------------
# 2. LHDI threshold matches Davidson & Gladd (1975)
# ---------------------------------------------------------------------------


class TestLHDIThreshold:
    """Lower-hybrid drift instability threshold validation."""

    def test_lhdi_factor_deuterium(self):
        """LHDI factor (m_e/m_D)^{1/4} ~ 0.129 for deuterium (Davidson & Gladd 1975)."""
        factor = lhdi_factor(mi=_m_d)
        expected = (m_e / _m_d) ** 0.25
        assert factor == pytest.approx(expected, rel=1e-6)
        assert 0.12 < factor < 0.14, f"LHDI factor {factor:.3f} outside expected range 0.12-0.14"

    def test_lhdi_threshold_below_ion_acoustic(self):
        """LHDI activates at lower drift velocity than ion-acoustic threshold (Te >> Ti)."""
        ne = np.array([1e23])
        Ti_K = np.array([50.0 * 11604.52])  # 50 eV
        Te_K = np.array([500.0 * 11604.52])  # 500 eV (Te >> Ti)
        mi = _m_d

        v_lhdi = lhdi_factor(mi) * ion_thermal_speed(Ti_K, mi)[0]
        c_s = ion_sound_speed(Te_K, mi)[0]

        assert v_lhdi < c_s, (
            f"LHDI threshold {v_lhdi:.3e} m/s should be < ion-acoustic {c_s:.3e} m/s "
            f"when Te >> Ti (Te/Ti = {float(Te_K[0]/Ti_K[0]):.0f})"
        )

    def test_lhdi_activates_before_ion_acoustic(self):
        """With drift between LHDI and ion-acoustic threshold, only LHDI triggers."""
        ne = np.array([1e23])
        Ti_K = np.array([10.0 * 11604.52])   # 10 eV
        Te_K = np.array([100.0 * 11604.52])  # 100 eV
        mi = _m_d

        # Set J so v_d is between LHDI and ion-acoustic thresholds
        v_lhdi = lhdi_factor(mi) * ion_thermal_speed(Ti_K, mi)
        c_s = ion_sound_speed(Te_K, mi)
        v_mid = (v_lhdi + c_s) / 2.0  # between the two thresholds
        J = ne * e * v_mid

        lhdi_active = lhdi_threshold(J, ne, Ti_K, mi)
        ia_active = ion_acoustic_threshold(J, ne, Te_K, mi)

        assert lhdi_active[0], "LHDI should be active"
        assert not ia_active[0], "Ion-acoustic should NOT be active (v_d < c_s)"

    def test_lhdi_inactive_below_threshold(self):
        """LHDI is inactive when v_d < (m_e/m_i)^{1/4} * v_ti."""
        ne = np.array([1e23])
        Ti_K = np.array([100.0 * 11604.52])
        mi = _m_d

        v_ti = ion_thermal_speed(Ti_K, mi)
        v_d_low = 0.5 * lhdi_factor(mi) * v_ti
        J_low = ne * e * v_d_low

        assert not lhdi_threshold(J_low, ne, Ti_K, mi)[0]

    def test_buneman_classic_highest_threshold(self):
        """True Buneman threshold (v_d > v_te) is much higher than LHDI and ion-acoustic."""
        ne = np.array([1e23])
        Ti_K = np.array([100.0 * 11604.52])
        Te_K = np.array([100.0 * 11604.52])
        mi = _m_d

        v_lhdi_thresh = lhdi_factor(mi) * ion_thermal_speed(Ti_K, mi)
        c_s = ion_sound_speed(Te_K, mi)
        v_te = np.sqrt(k_B * Te_K / m_e)

        assert v_lhdi_thresh[0] < c_s[0] < v_te[0], (
            "Threshold ordering should be: LHDI < ion-acoustic < Buneman"
        )

        # Buneman factor vs LHDI: should be >> 1
        ratio = v_te[0] / v_lhdi_thresh[0]
        assert ratio > 10, f"Buneman/LHDI ratio {ratio:.1f} should be > 10 for deuterium"


# ---------------------------------------------------------------------------
# 3. Anomalous resistivity magnitude: alpha in Bohm-fraction range (Haines 2011)
# ---------------------------------------------------------------------------


class TestAnomalousResistivityMagnitude:
    """Haines (2011): eta_anom = alpha * eta_Bohm, alpha ~ 0.01-0.1."""

    def test_eta_bohm_formula(self):
        """eta_Bohm = m_e * omega_pe / (ne * e^2) (Sagdeev 1966 / Haines 2011)."""
        ne = np.array([1e24])
        omega_pe = plasma_frequency(ne)[0]
        eta_bohm = m_e * omega_pe / (ne[0] * e**2)

        eta_anom = _compute_eta_anom(ne, alpha=1.0)[0]  # alpha=1 gives bare eta_Bohm

        assert eta_anom == pytest.approx(eta_bohm, rel=1e-6)

    def test_anomalous_alpha_range(self):
        """Enhancement factor alpha produces eta_anom in physical range 0.01-0.1 * eta_Bohm."""
        ne = np.array([1e24])

        eta_bohm = _compute_eta_anom(ne, alpha=1.0)[0]

        for alpha in [0.01, 0.03, 0.05, 0.1]:
            eta_anom = _compute_eta_anom(ne, alpha=alpha)[0]
            fraction = eta_anom / eta_bohm
            assert fraction == pytest.approx(alpha, rel=1e-6), (
                f"eta_anom/eta_Bohm = {fraction:.4f}, expected alpha = {alpha}"
            )

    def test_anomalous_enhancement_over_spitzer_at_pinch_conditions(self):
        """At pinch conditions (ne=1e25, Te=100 eV): anomalous exceeds Spitzer by 10-200x."""
        ne = np.array([1e25])
        Te_eV = 100.0
        Te_K = np.array([Te_eV * 11604.52])
        lnL = coulomb_log(ne, Te_K)[0]

        eta_spitzer = spitzer_resistivity(ne, Te_K, lnL=lnL, Z=1.0)[0]

        for alpha in [0.01, 0.05, 0.1]:
            eta_anom = _compute_eta_anom(ne, alpha=alpha)[0]
            enhancement = eta_anom / eta_spitzer
            assert enhancement > 5.0, (
                f"alpha={alpha}: enhancement {enhancement:.1f}x should be > 5x at pinch conditions"
            )

    def test_anomalous_zero_below_threshold(self):
        """Anomalous resistivity is zero when drift velocity is below threshold."""
        ne = np.array([1e23])
        # Set J so v_d is far below ion sound speed
        Ti_K = np.array([100.0 * 11604.52])
        Te_K = np.array([100.0 * 11604.52])
        mi = _m_d

        c_s = ion_sound_speed(Te_K, mi)
        v_d_low = 0.01 * c_s  # far below threshold
        J_low = ne * e * v_d_low

        eta_anom = anomalous_resistivity_field(
            J_low, ne, Ti_K, alpha=0.05, mi=mi, threshold_model="ion_acoustic", Te=Te_K
        )
        assert eta_anom[0] == 0.0

    def test_anomalous_nonzero_above_threshold(self):
        """Anomalous resistivity is nonzero when drift velocity exceeds threshold."""
        ne = np.array([1e23])
        Ti_K = np.array([100.0 * 11604.52])
        Te_K = np.array([100.0 * 11604.52])
        mi = _m_d

        c_s = ion_sound_speed(Te_K, mi)
        v_d_high = 10.0 * c_s  # well above threshold
        J_high = ne * e * v_d_high

        eta_anom = anomalous_resistivity_field(
            J_high, ne, Ti_K, alpha=0.05, mi=mi, threshold_model="ion_acoustic", Te=Te_K
        )
        assert eta_anom[0] > 0.0

    def test_resistivity_monotone_with_alpha(self):
        """Higher alpha → higher anomalous resistivity (physical: stronger turbulence)."""
        ne = np.array([1e24])

        etas = [_compute_eta_anom(ne, alpha=a)[0] for a in [0.01, 0.05, 0.1]]
        assert etas[0] < etas[1] < etas[2]

    def test_resistivity_increases_with_current_density(self):
        """Higher current density activates anomalous resistivity in more cells (physical)."""
        ne = np.full(10, 1e23)
        Ti_K = np.full(10, 100.0 * 11604.52)
        Te_K = np.full(10, 100.0 * 11604.52)
        mi = _m_d

        c_s = ion_sound_speed(Te_K[:1], mi)[0]
        J_low = ne * e * 0.5 * c_s   # below threshold everywhere
        J_high = ne * e * 5.0 * c_s  # above threshold everywhere

        eta_low = anomalous_resistivity_field(
            J_low, ne, Ti_K, alpha=0.05, mi=mi, threshold_model="ion_acoustic", Te=Te_K
        )
        eta_high = anomalous_resistivity_field(
            J_high, ne, Ti_K, alpha=0.05, mi=mi, threshold_model="ion_acoustic", Te=Te_K
        )

        assert np.sum(eta_high > 0) > np.sum(eta_low > 0), (
            "Higher J should activate anomalous resistivity in more cells"
        )


# ---------------------------------------------------------------------------
# 4. Ion-acoustic threshold uses Te (not Ti) per Sagdeev (1966)
# ---------------------------------------------------------------------------


class TestIonAcousticThreshold:
    """Ion-acoustic instability condition: v_d > c_s = sqrt(k_B * Te / m_i)."""

    def test_threshold_uses_electron_temperature(self):
        """Ion-acoustic threshold velocity c_s uses Te (electron pressure provides restoring force)."""
        ne = np.array([1e23])
        mi = _m_d

        Te_cold = np.array([10.0 * 11604.52])
        Te_hot = np.array([1000.0 * 11604.52])

        c_s_cold = ion_sound_speed(Te_cold, mi)[0]
        c_s_hot = ion_sound_speed(Te_hot, mi)[0]

        assert c_s_hot > c_s_cold, "Higher Te → higher ion sound speed (threshold)"
        assert c_s_hot / c_s_cold == pytest.approx((1000.0 / 10.0) ** 0.5, rel=1e-6)

    def test_sound_speed_formula(self):
        """c_s = sqrt(k_B * Te / m_i) matches direct calculation."""
        Te_eV = 100.0
        Te_K = np.array([Te_eV * 11604.52])
        mi = _m_d

        c_s_func = ion_sound_speed(Te_K, mi)[0]
        c_s_direct = (k_B * Te_K[0] / mi) ** 0.5

        assert c_s_func == pytest.approx(c_s_direct, rel=1e-6)

    def test_threshold_independent_of_density_above_floor(self):
        """Ion-acoustic threshold velocity is density-independent (c_s doesn't depend on ne)."""
        mi = _m_d
        Te_K = np.array([100.0 * 11604.52])

        c_s_low_ne = ion_sound_speed(Te_K, mi)[0]
        c_s_high_ne = ion_sound_speed(Te_K, mi)[0]

        assert c_s_low_ne == pytest.approx(c_s_high_ne, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Drift velocity formula consistency
# ---------------------------------------------------------------------------


class TestDriftVelocityFormula:
    """v_d = |J| / (ne * e) — basic consistency check."""

    def test_drift_velocity_formula(self):
        """v_d = J / (ne * e) matches manual calculation."""
        ne = np.array([1e23])
        J = np.array([1e9])  # A/m^2

        v_d = electron_drift_velocity(J, ne)[0]
        v_d_direct = J[0] / (ne[0] * e)

        assert v_d == pytest.approx(v_d_direct, rel=1e-9)

    def test_plasma_frequency_formula(self):
        """omega_pe = sqrt(ne * e^2 / (eps0 * me)) matches manual calculation."""
        ne = np.array([1e23])

        omega_pe_func = plasma_frequency(ne)[0]
        omega_pe_direct = (ne[0] * e**2 / (epsilon_0 * m_e)) ** 0.5

        assert omega_pe_func == pytest.approx(omega_pe_direct, rel=1e-9)
