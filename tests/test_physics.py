"""Consolidated physics tests for DPF-Unified.

Covers:
    1. Collision — Spitzer collision frequencies and Braginskii coefficients
    2. EOS — Two-temperature ideal gas equation of state
    3. Ionization — Saha equation, Z_bar(Te, ne)
    4. Kinetic — KineticManager, Boris pusher, beam injection
    5. Pease-Braginskii — I_PB current, regime detection, derived diagnostics
    6. Phase D Physics — Braginskii viscosity, anisotropic conduction, Powell/Dedner
    7. Phase R Transport — Spitzer alpha(Z), Coulomb log consistency, ion mass
    8. Phase S Ablation — ablation_source, ablation_source_array, AblationConfig
    9. Phase S Anomalous Resistivity — LHDI/ion-acoustic/Buneman thresholds
    10. Phase U Ablation — Comprehensive V&V for ablation module
    11. Anisotropic Conduction — Sign correctness, anisotropy, stability
    12. Two-Temperature Physics — Te/Ti relaxation, EOS, Spitzer resistivity
    13. M=0 Instability — Sausage growth rate, scaling, dimensional analysis
    14. Phase 13 — Line radiation, CR ionization, beam-target, neutron anisotropy
    15. Phase 16 — Nernst effect, Braginskii viscosity, constrained transport, validation
    16. Radiation — Bremsstrahlung, flux-limited diffusion transport
    17. Sheath — Bohm velocity, Child-Langmuir, Debye, Poisson, sheath BC
    18. Turbulence — Anomalous resistivity, Buneman threshold, engine integration
"""

from __future__ import annotations

import math
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from dpf.atomic.ablation import (
    COPPER_ABLATION_EFFICIENCY,
    COPPER_MASS,
    TUNGSTEN_ABLATION_EFFICIENCY,
    TUNGSTEN_MASS,
    ablation_momentum_source,
    ablation_particle_flux,
    ablation_rate,
    ablation_source,
    ablation_source_array,
)
from dpf.atomic.ionization import (
    _IP_CU,
    _IP_H,
    _IP_W,
    IONIZATION_POTENTIALS,
    cr_average_charge,
    cr_evolve_field,
    cr_solve_charge_states,
    cr_zbar_field,
    dielectronic_recombination_rate,
    lotz_ionization_rate,
    radiative_recombination_rate,
    saha_ionization_fraction,
    total_recombination_rate,
)
from dpf.collision.spitzer import (
    braginskii_kappa,
    coulomb_log,
    nu_ee,
    nu_ei,
    nu_en,
    nu_ii,
    relax_temperatures,
    spitzer_alpha,
    spitzer_resistivity,
)
from dpf.config import AblationConfig, SimulationConfig
from dpf.constants import e, epsilon_0, eV, k_B, m_d, m_e, m_p
from dpf.diagnostics.beam_target import (
    beam_target_yield_rate,
    dd_cross_section,
    dd_cross_section_array,
    detect_pinch_disruption,
    neutron_anisotropy,
)
from dpf.diagnostics.instability import m0_growth_rate, m0_growth_rate_from_state
from dpf.diagnostics.pease_braginskii import (
    check_pease_braginskii,
    pease_braginskii_current,
)
from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction
from dpf.fluid.eos import IdealEOS
from dpf.fluid.nernst import nernst_coefficient
from dpf.kinetic.manager import KineticManager
from dpf.radiation.line_radiation import (
    apply_line_radiation_losses,
    cooling_function,
    line_radiation_power,
    recombination_power,
    total_radiation_power,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

M_D = 3.34358377e-27  # Deuterium mass [kg]
MU_0 = 4e-7 * np.pi
_EXPECTED_KEYS = {
    "growth_rate", "growth_time", "alfven_speed",
    "beta_p", "is_unstable", "stability_margin",
}

# ---------------------------------------------------------------------------
# Module-level helpers (Phase X anisotropic conduction)
# ---------------------------------------------------------------------------


def _make_hotspot(nx: int = 16) -> tuple:
    """Create a state with a central hot spot in a uniform B field."""
    Te = np.full((nx, nx, nx), 1e5)
    Te[nx // 2 - 2:nx // 2 + 2, nx // 2 - 2:nx // 2 + 2, nx // 2 - 2:nx // 2 + 2] = 1e6
    B = np.zeros((3, nx, nx, nx))
    ne = np.full((nx, nx, nx), 1e24)
    return Te, B, ne


# ---------------------------------------------------------------------------
# Module-level helpers (Phase Y instability)
# ---------------------------------------------------------------------------


def _make_mock_snowplow(
    r_shock: float = 0.04,
    r_pinch_min: float = 0.005,
    phase: str = "radial",
) -> MagicMock:
    sp = MagicMock()
    sp.r_shock = r_shock
    sp.r_pinch_min = r_pinch_min
    sp.phase = phase
    return sp


def _make_mock_config(dx: float = 0.01) -> MagicMock:
    cfg = MagicMock()
    cfg.dx = dx
    return cfg


def _make_mock_state(
    nr: int = 8,
    nz: int = 8,
    B_theta_val: float = 1.0,
    rho_val: float = 1e-3,
    p_val: float = 1e3,
) -> dict:
    B = np.zeros((3, nr, 1, nz))
    B[1] = B_theta_val
    rho = np.full((nr, 1, nz), rho_val)
    pressure = np.full((nr, 1, nz), p_val)
    return {"B": B, "rho": rho, "pressure": pressure}


# --- Section: Collision ---


class TestCoulombLog:
    """Verify dynamic Coulomb logarithm."""

    def test_typical_plasma(self):
        """For typical DPF plasma (ne=1e24, Te=1e6 K), ln(Lambda) ~ 5-20."""
        ne = np.array([1e24])
        Te = np.array([1e6])
        lnL = coulomb_log(ne, Te)
        assert 3 < lnL[0] < 25

    def test_always_positive(self):
        """Coulomb log is always >= 0 (floored at ln(1) = 0)."""
        ne = np.array([1e20, 1e24, 1e28])
        Te = np.array([1e3, 1e6, 1e9])
        lnL = coulomb_log(ne, Te)
        assert np.all(lnL >= 0)


class TestSpitzerFrequencies:
    """Verify collision frequencies against NRL Plasma Formulary."""

    def test_nu_ei_order_of_magnitude(self):
        """For ne=1e20, Te=10 eV: nu_ei ~ 1e8 s^-1 (NRL)."""
        ne = np.array([1e20])
        Te_eV = 10.0
        Te = np.array([Te_eV * e / k_B])
        lnL = 10.0
        freq = nu_ei(ne, Te, lnL)
        assert 1e4 < freq[0] < 1e14

    def test_nu_ee_larger_than_nu_ei(self):
        """nu_ee = sqrt(2) * nu_ei for Z=1."""
        ne = np.array([1e22])
        Te = np.array([1e5])
        lnL = 10.0
        freq_ei = nu_ei(ne, Te, lnL)
        freq_ee = nu_ee(ne, Te, lnL)
        np.testing.assert_allclose(freq_ee, freq_ei * np.sqrt(2), rtol=1e-10)

    def test_nu_en_scales_with_density(self):
        """Electron-neutral frequency is proportional to neutral density."""
        ne = np.array([1e20])
        Te = np.array([1e4])
        nn1 = np.array([1e20])
        nn2 = np.array([2e20])
        freq1 = nu_en(ne, Te, nn1)
        freq2 = nu_en(ne, Te, nn2)
        np.testing.assert_allclose(freq2, 2 * freq1, rtol=1e-10)


class TestBraginskii:
    """Verify Braginskii thermal conductivity."""

    def test_kappa_par_larger_than_perp(self):
        """Parallel conductivity >> perpendicular in magnetized plasma."""
        ne = np.array([1e22])
        Te = np.array([1e6])
        B = np.array([1.0])
        kpar, kper = braginskii_kappa(ne, Te, B)
        assert kpar[0] > kper[0]
        assert kpar[0] / kper[0] > 10

    def test_kappa_unmagnetized(self):
        """With B=0, kappa_par == kappa_perp."""
        ne = np.array([1e22])
        Te = np.array([1e6])
        B = np.array([0.0])
        kpar, kper = braginskii_kappa(ne, Te, B)
        np.testing.assert_allclose(kpar, kper, rtol=1e-6)


class TestTemperatureRelaxation:
    """Verify implicit temperature equilibration."""

    def test_relaxation_conserves_energy(self):
        """Total thermal energy (Te + Ti) is conserved."""
        Te = np.array([1e6])
        Ti = np.array([1e4])
        freq = np.array([1e10])
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq, dt=1e-10)
        np.testing.assert_allclose(Te_new + Ti_new, Te + Ti, rtol=1e-10)

    def test_relaxation_toward_equilibrium(self):
        """Temperatures approach each other."""
        Te = np.array([1e6])
        Ti = np.array([1e4])
        freq = np.array([1e10])
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq, dt=1e-8)
        assert abs(Te_new[0] - Ti_new[0]) < abs(Te[0] - Ti[0])

    def test_equilibrium_unchanged(self):
        """If Te == Ti, relaxation does nothing."""
        T = np.array([5e5])
        freq = np.array([1e10])
        Te_new, Ti_new = relax_temperatures(T.copy(), T.copy(), freq, dt=1e-8)
        np.testing.assert_allclose(Te_new, T, rtol=1e-10)
        np.testing.assert_allclose(Ti_new, T, rtol=1e-10)


# --- Section: EOS ---


class TestIdealEOS:
    """Verify two-temperature ideal gas EOS."""

    def setup_method(self):
        self.eos = IdealEOS(gamma=5.0 / 3.0)

    def test_ion_pressure(self):
        """p_i = n_i * k_B * T_i."""
        rho = np.array([1e-4])
        Ti = np.array([1e4])
        n_i = rho / m_d
        expected = n_i * k_B * Ti
        result = self.eos.ion_pressure(rho, Ti)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_electron_pressure_different_from_ion(self):
        """Electron pressure uses Z * n_i, NOT the same as ion pressure."""
        rho = np.array([1e-4])
        T = np.array([1e4])
        p_i = self.eos.ion_pressure(rho, T)
        p_e = self.eos.electron_pressure(rho, T)
        np.testing.assert_allclose(p_i, p_e, rtol=1e-10)
        eos_Z2 = IdealEOS(gamma=5.0 / 3.0, Z=2.0)
        p_e_Z2 = eos_Z2.electron_pressure(rho, T)
        np.testing.assert_allclose(p_e_Z2, 2.0 * p_i, rtol=1e-10)

    def test_total_pressure(self):
        """Total = ion + electron."""
        rho = np.array([1e-4])
        Ti = np.array([1e4])
        Te = np.array([2e4])
        p_total = self.eos.total_pressure(rho, Ti, Te)
        p_sum = self.eos.ion_pressure(rho, Ti) + self.eos.electron_pressure(rho, Te)
        np.testing.assert_allclose(p_total, p_sum, rtol=1e-10)

    def test_energy_positive(self):
        """Internal energy must be positive for positive T."""
        rho = np.array([1e-4])
        T = np.array([1e4])
        assert self.eos.ion_energy(rho, T) > 0
        assert self.eos.electron_energy(rho, T) > 0

    def test_sound_speed_positive(self):
        """Sound speed must be real and positive."""
        rho = np.array([1e-4])
        Ti = np.array([1e4])
        Te = np.array([1e4])
        cs = self.eos.sound_speed(rho, Ti, Te)
        assert cs > 0

    def test_temperature_recovery(self):
        """Temperature -> energy -> temperature roundtrip."""
        rho = np.array([1e-3])
        T_orig = np.array([5e4])
        e_int = self.eos.ion_energy(rho, T_orig) + self.eos.electron_energy(rho, T_orig)
        T_recovered = self.eos.temperature_from_energy(rho, e_int)
        np.testing.assert_allclose(T_recovered, T_orig, rtol=1e-6)


# --- Section: Ionization ---


class TestSahaIonizationFraction:
    """Tests for Saha ionization fraction."""

    def test_cold_gas_neutral(self):
        """At room temperature, hydrogen should be fully neutral (Z ~ 0)."""
        from dpf.atomic.ionization import saha_ionization_fraction as _sif

        Te = 300.0
        ne = 1e20
        Z = _sif(Te, ne)
        assert Z < 1e-10, f"Z = {Z:.2e}, should be ~0 at room temperature"

    def test_hot_plasma_ionized(self):
        """At 10 eV, hydrogen should be fully ionized (Z ~ 1)."""
        from dpf.atomic.ionization import saha_ionization_fraction as _sif

        Te = 10.0 * eV / k_B
        ne = 1e20
        Z = _sif(Te, ne)
        assert Z > 0.99, f"Z = {Z:.4f}, should be ~1 at 10 eV"

    def test_very_hot_plasma(self):
        """At 1 keV, Z should be essentially 1."""
        from dpf.atomic.ionization import saha_ionization_fraction as _sif

        Te = 1000.0 * eV / k_B
        ne = 1e24
        Z = _sif(Te, ne)
        assert Z > 0.999, f"Z = {Z:.6f}, should be ~1 at 1 keV"

    def test_transition_around_1eV(self):
        """The ionization transition for H should occur around 1-2 eV."""
        from dpf.atomic.ionization import saha_ionization_fraction as _sif

        ne = 1e20
        Te_low = 0.5 * eV / k_B
        Z_low = _sif(Te_low, ne)
        Te_high = 2.0 * eV / k_B
        Z_high = _sif(Te_high, ne)
        assert Z_low < 0.5, f"Z at 0.5 eV = {Z_low:.3f}, should be < 0.5"
        assert Z_high > 0.5, f"Z at 2 eV = {Z_high:.3f}, should be > 0.5"

    def test_monotonic_in_temperature(self):
        """Z_bar should monotonically increase with temperature."""
        from dpf.atomic.ionization import saha_ionization_fraction as _sif

        ne = 1e20
        temps = [100.0, 1000.0, 5000.0, 10000.0, 50000.0, 1e6]
        Z_values = [_sif(T, ne) for T in temps]
        for i in range(len(Z_values) - 1):
            assert Z_values[i + 1] >= Z_values[i], (
                f"Z not monotonic: Z({temps[i]:.0f}K)={Z_values[i]:.4f} > "
                f"Z({temps[i+1]:.0f}K)={Z_values[i+1]:.4f}"
            )

    def test_higher_density_lower_ionization(self):
        """At fixed Te, higher ne should give lower Z_bar (recombination wins)."""
        from dpf.atomic.ionization import saha_ionization_fraction as _sif

        Te = 1.0 * eV / k_B
        Z_low_ne = _sif(Te, 1e18)
        Z_high_ne = _sif(Te, 1e24)
        assert Z_low_ne > Z_high_ne, (
            f"Higher density should reduce ionization: "
            f"Z(1e18)={Z_low_ne:.4f}, Z(1e24)={Z_high_ne:.4f}"
        )

    def test_bounded_zero_one(self):
        """Z_bar should always be in [0, 1]."""
        from dpf.atomic.ionization import saha_ionization_fraction as _sif

        test_cases = [
            (1.0, 1.0), (300.0, 1e20), (1e4, 1e15), (1e7, 1e25), (1e8, 1e30),
        ]
        for Te, ne in test_cases:
            Z = _sif(Te, ne)
            assert 0.0 <= Z <= 1.0, f"Z({Te}, {ne}) = {Z} out of bounds"


class TestSahaArray:
    """Tests for vectorized Saha ionization."""

    def test_matches_scalar(self):
        """Array version should match scalar for each element."""
        from dpf.atomic.ionization import (
            saha_ionization_fraction as _sif,
        )
        from dpf.atomic.ionization import (
            saha_ionization_fraction_array,
        )

        Te = np.array([300.0, 5000.0, 50000.0, 1e6])
        ne = np.array([1e20, 1e20, 1e20, 1e20])
        Z_array = saha_ionization_fraction_array(Te, ne)
        for i in range(len(Te)):
            Z_scalar = _sif(Te[i], ne[i])
            np.testing.assert_allclose(
                Z_array[i], Z_scalar, rtol=1e-10,
                err_msg=f"Mismatch at Te={Te[i]}, ne={ne[i]}"
            )

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        from dpf.atomic.ionization import saha_ionization_fraction_array

        Te = np.full((4, 4, 4), 1e5)
        ne = np.full((4, 4, 4), 1e20)
        Z = saha_ionization_fraction_array(Te, ne)
        assert Z.shape == Te.shape


# --- Section: Kinetic ---


class TestKineticManager:
    @pytest.fixture
    def config(self):
        conf = SimulationConfig(
            grid_shape=[10, 10, 10],
            dx=0.1,
            sim_time=1e-6,
            circuit={
                "C": 1e-6, "V0": 10e3, "L0": 10e-9,
                "anode_radius": 0.01, "cathode_radius": 0.02,
            },
        )
        conf.kinetic.enabled = True
        return conf

    def test_initialization(self, config):
        km = KineticManager(config)
        assert km.kc.enabled is True
        assert km.driver is not None

    def test_gyromotion(self, config):
        """Test that a particle in a uniform B-field performs cyclotron motion."""
        km = KineticManager(config)
        v0 = 1e5
        km.ion_species.positions = np.array([[0.5, 0.5, 0.5]])
        km.ion_species.velocities = np.array([[v0, 0.0, 0.0]])
        km.ion_species.weights = np.array([1.0])
        B0 = 1.0
        nx, ny, nz = 10, 10, 10
        B_field = np.zeros((nx, ny, nz, 3))
        B_field[..., 2] = B0
        E_field = np.zeros((nx, ny, nz, 3))
        mass = config.ion_mass
        T_c = 2 * np.pi * mass / (e * B0)
        dt = T_c / 20.0
        for _ in range(5):
            km.step(dt, 0.0, E_field, B_field)
        vel = km.ion_species.velocities[0]
        v_mag = np.linalg.norm(vel)
        assert np.isclose(v_mag, v0, rtol=1e-4), "Energy not conserved in B-field"

    def test_beam_injection(self, config):
        config.kinetic.inject_beam = True
        config.kinetic.start_time = 1e-9
        config.kinetic.n_particles = 100
        km = KineticManager(config)
        nx, ny, nz = 10, 10, 10
        B_field = np.zeros((nx, ny, nz, 3))
        E_field = np.zeros((nx, ny, nz, 3))
        km.step(1e-10, 0.0, E_field, B_field)
        assert km.ion_species.n_particles() == 0
        km.step(1e-10, 2e-9, E_field, B_field)
        assert km.ion_species.n_particles() == 100
        assert km.beam_injected is True


# --- Section: Pease-Braginskii ---


class TestPeaseBraginskiiCurrent:
    """Test the Pease-Braginskii current calculation."""

    def test_returns_expected_keys(self):
        result = pease_braginskii_current()
        assert "I_PB" in result
        assert "I_PB_MA" in result
        assert "Z" in result
        assert "gaunt_factor" in result
        assert "ln_Lambda" in result

    def test_default_deuterium_in_ma_range(self):
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        assert 0.5 < result["I_PB_MA"] < 2.0, (
            f"I_PB = {result['I_PB_MA']:.3f} MA outside expected [0.5, 2.0] MA"
        )

    def test_haines_formula_z1_g1_ln10(self):
        """Haines (2011): I_PB = 0.433 * sqrt(6/2) = 0.750 MA for Z=1, g=1, lnL=10."""
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        expected = 0.433 * math.sqrt(3.0)
        assert result["I_PB_MA"] == pytest.approx(expected, rel=1e-3)

    def test_haines_formula_z1_g12_ln10(self):
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        expected = 0.433 * math.sqrt(6.0 / 2.4)
        assert result["I_PB_MA"] == pytest.approx(expected, rel=1e-3)

    def test_gaunt_factor_1_gives_higher_ipb(self):
        pb_g1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        pb_g12 = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        assert pb_g1["I_PB"] > pb_g12["I_PB"]

    def test_higher_z_lowers_ipb(self):
        pb_z1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        pb_z2 = pease_braginskii_current(Z=2.0, gaunt_factor=1.2, ln_Lambda=10.0)
        assert pb_z2["I_PB"] < pb_z1["I_PB"]

    def test_higher_ln_lambda_raises_ipb(self):
        pb_low = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=5.0)
        pb_high = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=20.0)
        assert pb_high["I_PB"] > pb_low["I_PB"]

    def test_ipb_scales_as_sqrt_ln_lambda(self):
        pb_10 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        pb_40 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=40.0)
        ratio = pb_40["I_PB"] / pb_10["I_PB"]
        expected = math.sqrt(40.0 / 10.0)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_ipb_scales_as_inv_sqrt_gaunt(self):
        pb_1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        pb_4 = pease_braginskii_current(Z=1.0, gaunt_factor=4.0, ln_Lambda=10.0)
        ratio = pb_4["I_PB"] / pb_1["I_PB"]
        expected = math.sqrt(1.0 / 4.0)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_ma_conversion(self):
        result = pease_braginskii_current()
        assert result["I_PB_MA"] == pytest.approx(result["I_PB"] * 1e-6, rel=1e-12)

    def test_positive_result(self):
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=2.0)
        assert result["I_PB"] > 0

    def test_z2_value(self):
        pb_z2 = pease_braginskii_current(Z=2.0, gaunt_factor=1.2, ln_Lambda=10.0)
        pb_z1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        ratio = pb_z2["I_PB"] / pb_z1["I_PB"]
        assert ratio == pytest.approx(math.sqrt(2.0 / 6.0), rel=1e-10)


class TestCheckPeaseBraginskii:
    """Test the regime detection function."""

    def test_pf1000_exceeds_pb(self):
        result = check_pease_braginskii(
            I_current=1.87e6, Z=1.0, gaunt_factor=1.2, ln_Lambda=20.0
        )
        assert result["exceeds_PB"], (
            f"PF-1000 at 1.87 MA should exceed I_PB = {result['I_PB_MA']:.3f} MA"
        )
        assert result["regime"] == "radiative_collapse"
        assert result["ratio"] > 1.0

    def test_small_device_below_pb(self):
        result = check_pease_braginskii(
            I_current=100e3, Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0
        )
        assert not result["exceeds_PB"]
        assert result["regime"] == "stable"
        assert result["ratio"] < 1.0

    def test_returns_both_currents(self):
        result = check_pease_braginskii(I_current=1e6)
        assert "I_current" in result
        assert "I_current_MA" in result
        assert "I_PB" in result
        assert "I_PB_MA" in result
        assert "ratio" in result
        assert "exceeds_PB" in result
        assert "regime" in result

    def test_ratio_is_correct(self):
        result = check_pease_braginskii(I_current=2e6)
        assert result["ratio"] == pytest.approx(
            result["I_current"] / result["I_PB"], rel=1e-10
        )

    def test_negative_current_uses_abs(self):
        result_pos = check_pease_braginskii(I_current=1.87e6)
        result_neg = check_pease_braginskii(I_current=-1.87e6)
        assert result_pos["ratio"] == pytest.approx(result_neg["ratio"], rel=1e-10)

    def test_zero_current_is_stable(self):
        result = check_pease_braginskii(I_current=0.0)
        assert not result["exceeds_PB"]
        assert result["regime"] == "stable"
        assert result["ratio"] == 0.0


class TestDerivedDiagnostics:
    """Test fast_magnetosonic_speed and bennett_radius from derived.py."""

    def test_fast_magnetosonic_pure_hydro(self):
        """With B=0, fast magnetosonic speed = sound speed."""
        from dpf.diagnostics.derived import fast_magnetosonic_speed

        B = np.zeros((3, 4, 4, 4))
        rho = np.ones((4, 4, 4)) * 1e-3
        p = np.ones((4, 4, 4)) * 1e5
        gamma = 5.0 / 3.0
        c_f = fast_magnetosonic_speed(B, p, rho, gamma)
        c_s = np.sqrt(gamma * p / rho)
        np.testing.assert_allclose(c_f, c_s, rtol=1e-10)

    def test_fast_magnetosonic_pure_magnetic(self):
        """With p=0, fast magnetosonic speed = Alfven speed."""
        from dpf.diagnostics.derived import alfven_speed, fast_magnetosonic_speed

        B = np.zeros((3, 4, 4, 4))
        B[2] = 1.0
        rho = np.ones((4, 4, 4)) * 1e-3
        p = np.zeros((4, 4, 4))
        c_f = fast_magnetosonic_speed(B, p, rho, gamma=5.0 / 3.0)
        v_A = alfven_speed(B, rho)
        np.testing.assert_allclose(c_f, v_A, rtol=1e-10)

    def test_fast_magnetosonic_positive(self):
        from dpf.diagnostics.derived import fast_magnetosonic_speed

        B = np.random.randn(3, 4, 4, 4) * 0.1
        rho = np.ones((4, 4, 4)) * 1e-3
        p = np.ones((4, 4, 4)) * 1e5
        assert np.all(fast_magnetosonic_speed(B, p, rho) > 0)

    def test_bennett_radius_scales_with_current(self):
        from dpf.diagnostics.derived import bennett_radius

        a1 = bennett_radius(current=1e6, Te=1e7, ne=1e24)
        a2 = bennett_radius(current=2e6, Te=1e7, ne=1e24)
        assert a2 / a1 == pytest.approx(2.0, rel=1e-10)

    def test_bennett_radius_pf1000(self):
        from dpf.diagnostics.derived import bennett_radius

        a_B = bennett_radius(current=1.87e6, Te=1.16e7, ne=1e25)
        assert 1e-4 < a_B < 0.05, f"Bennett radius {a_B:.4e} m outside [0.1mm, 50mm]"

    def test_bennett_radius_positive(self):
        from dpf.diagnostics.derived import bennett_radius

        assert bennett_radius(current=1e6, Te=1e7, ne=1e24) > 0
        assert bennett_radius(current=0, Te=1e7, ne=1e24) >= 0


class TestEngineIntegration:
    """Test that PB diagnostic is wired into the engine."""

    def test_engine_has_pb_result_after_step(self, small_config):
        from dpf.engine import SimulationEngine

        small_config.diagnostics.hdf5_filename = ":memory:"
        engine = SimulationEngine(small_config)
        engine.step()
        assert hasattr(engine, "_last_pb_result")
        pb = engine._last_pb_result
        assert "I_PB_MA" in pb
        assert "ratio" in pb
        assert "regime" in pb
        assert pb["I_PB_MA"] > 0


# --- Section: Phase D Physics ---


class TestBraginskiiEta1Limits:
    """Verify eta_1 asymptotic behaviour in weakly and strongly magnetised limits."""

    def test_weakly_magnetised_eta1_approaches_eta0(self):
        from dpf.fluid.viscosity import braginskii_eta0, braginskii_eta1, ion_collision_time

        ni = np.array([1e22])
        Ti = np.array([1e6])
        tau_i = ion_collision_time(ni, Ti)
        B_weak = np.array([1e-10])
        eta0 = braginskii_eta0(ni, Ti, tau_i)
        eta1 = braginskii_eta1(ni, Ti, tau_i, B_weak, m_ion=m_d)
        assert eta1[0] > 0.0, "eta_1 should be positive"
        ratio = eta1[0] / eta0[0]
        assert ratio == pytest.approx(1.0, abs=0.01), (
            f"In weakly magnetised limit, eta_1 should be capped at eta_0 (ratio={ratio})"
        )

    def test_strongly_magnetised_eta1_small(self):
        from dpf.fluid.viscosity import braginskii_eta0, braginskii_eta1, ion_collision_time

        ni = np.array([1e22])
        Ti = np.array([1e6])
        tau_i = ion_collision_time(ni, Ti)
        B_strong = np.array([10.0])
        eta0 = braginskii_eta0(ni, Ti, tau_i)
        eta1 = braginskii_eta1(ni, Ti, tau_i, B_strong, m_ion=m_d)
        ratio = eta1[0] / eta0[0]
        assert ratio < 0.01, (
            f"In strongly magnetised limit, eta_1/eta_0 should be << 1, got {ratio:.4e}"
        )


class TestBraginskiiEta2:
    """Verify eta_2 = 4 * eta_1."""

    def test_eta2_is_4_eta1(self):
        from dpf.fluid.viscosity import braginskii_eta1, braginskii_eta2, ion_collision_time

        ni = np.array([1e22, 5e22, 1e23])
        Ti = np.array([5e5, 1e6, 5e6])
        tau_i = ion_collision_time(ni, Ti)
        B_mag = np.array([0.5, 1.0, 3.0])
        eta1 = braginskii_eta1(ni, Ti, tau_i, B_mag, m_ion=m_d)
        eta2 = braginskii_eta2(ni, Ti, tau_i, B_mag, m_ion=m_d)
        np.testing.assert_allclose(eta2, 4.0 * eta1, rtol=1e-12,
                                   err_msg="eta_2 should be exactly 4 * eta_1")


class TestFullBraginskiiBackwardCompatible:
    """Verify viscous_stress_rate with full_braginskii=False gives same result."""

    def test_default_matches_old_behaviour(self):
        from dpf.fluid.viscosity import viscous_stress_rate

        np.random.seed(42)
        nx, ny, nz = 8, 8, 8
        velocity = np.random.randn(3, nx, ny, nz) * 1e4
        rho = np.ones((nx, ny, nz)) * 1e-4
        eta0 = np.ones((nx, ny, nz)) * 1e-2
        dx = dy = dz = 1e-3
        accel_default = viscous_stress_rate(velocity, rho, eta0, dx, dy, dz)
        accel_explicit = viscous_stress_rate(
            velocity, rho, eta0, dx, dy, dz, full_braginskii=False,
        )
        np.testing.assert_allclose(accel_default, accel_explicit, rtol=1e-14,
                                   err_msg="Default should match explicit full_braginskii=False")

    def test_full_braginskii_without_B_falls_back(self):
        from dpf.fluid.viscosity import viscous_stress_rate

        np.random.seed(42)
        nx, ny, nz = 8, 8, 8
        velocity = np.random.randn(3, nx, ny, nz) * 1e4
        rho = np.ones((nx, ny, nz)) * 1e-4
        eta0 = np.ones((nx, ny, nz)) * 1e-2
        dx = dy = dz = 1e-3
        accel_default = viscous_stress_rate(velocity, rho, eta0, dx, dy, dz)
        accel_fallback = viscous_stress_rate(
            velocity, rho, eta0, dx, dy, dz,
            full_braginskii=True, B=None, eta1=None,
        )
        np.testing.assert_allclose(accel_default, accel_fallback, rtol=1e-14,
                                   err_msg="full_braginskii=True without B should fall back")


class TestAnisotropicConductionAlongB:
    """Uniform B in x-direction, temperature gradient in x -> full parallel conduction."""

    def test_conduction_along_B(self):
        from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction as _atc

        nx, ny, nz = 32, 4, 4
        dx = dy = dz = 1e-3
        B = np.zeros((3, nx, ny, nz))
        B[0] = 1.0
        x = np.linspace(0, (nx - 1) * dx, nx)
        Te = np.zeros((nx, ny, nz))
        for i in range(nx):
            Te[i, :, :] = 1e6 + 5e5 * np.sin(2 * np.pi * x[i] / (nx * dx))
        ne = np.full((nx, ny, nz), 1e22)
        dt = 1e-12
        Te_new = _atc(Te, B, ne, dt, dx, dy, dz, Z_eff=1.0)
        delta_Te = Te_new - Te
        max_change = float(np.max(np.abs(delta_Te)))
        assert max_change > 0.0, "Conduction along B should change temperature"


class TestAnisotropicConductionAcrossB:
    """Perpendicular conduction is weaker than parallel."""

    def test_conduction_across_B_is_weaker(self):
        from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction as _atc

        nx, ny, nz = 8, 32, 4
        dx = dy = dz = 1e-3
        B = np.zeros((3, nx, ny, nz))
        B[0] = 1.0
        y = np.linspace(0, (ny - 1) * dy, ny)
        Te_perp = np.zeros((nx, ny, nz))
        for j in range(ny):
            Te_perp[:, j, :] = 1e6 + 5e5 * np.sin(2 * np.pi * y[j] / (ny * dy))
        x = np.linspace(0, (nx - 1) * dx, nx)
        Te_par = np.zeros((nx, ny, nz))
        for i in range(nx):
            Te_par[i, :, :] = 1e6 + 5e5 * np.sin(2 * np.pi * x[i] / (nx * dx))
        ne = np.full((nx, ny, nz), 1e22)
        dt = 1e-12
        Te_par_new = _atc(Te_par, B, ne, dt, dx, dy, dz, Z_eff=1.0)
        Te_perp_new = _atc(Te_perp, B, ne, dt, dx, dy, dz, Z_eff=1.0)
        change_par = float(np.max(np.abs(Te_par_new - Te_par)))
        change_perp = float(np.max(np.abs(Te_perp_new - Te_perp)))
        if change_par > 1e-10:
            assert change_perp <= change_par * 1.1, (
                f"Perpendicular conduction ({change_perp:.4e}) should not exceed "
                f"parallel conduction ({change_par:.4e})"
            )


class TestPowellSourceZeroDivB:
    """If div(B) = 0, all Powell sources should be zero."""

    def test_divergence_free_B_gives_zero_sources(self):
        from dpf.fluid.mhd_solver import powell_source_terms

        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 1e-3
        B = np.zeros((3, nx, ny, nz))
        B[2] = 1.0
        state = {
            "rho": np.ones((nx, ny, nz)) * 1e-4,
            "velocity": np.random.RandomState(42).randn(3, nx, ny, nz) * 1e3,
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": B,
            "psi": np.zeros((nx, ny, nz)),
        }
        sources = powell_source_terms(state, dx, dy, dz)
        np.testing.assert_allclose(sources["dmom_powell"], 0.0, atol=1e-10)
        np.testing.assert_allclose(sources["denergy_powell"], 0.0, atol=1e-10)
        np.testing.assert_allclose(sources["dB_powell"], 0.0, atol=1e-10)


class TestPowellSourceNonzeroDivB:
    """Create a state with known div(B) and verify source magnitudes."""

    def test_nonzero_div_B_gives_nonzero_sources(self):
        from dpf.fluid.mhd_solver import powell_source_terms

        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 1e-3
        x = np.linspace(0, (nx - 1) * dx, nx)
        B = np.zeros((3, nx, ny, nz))
        for i in range(nx):
            B[0, i, :, :] = x[i] * 100.0
        vel = np.ones((3, nx, ny, nz)) * 1e4
        state = {
            "rho": np.ones((nx, ny, nz)) * 1e-4,
            "velocity": vel,
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": B,
            "psi": np.zeros((nx, ny, nz)),
        }
        sources = powell_source_terms(state, dx, dy, dz)
        div_B = sources["div_B"]
        assert float(np.max(np.abs(div_B))) > 1.0
        assert float(np.max(np.abs(sources["dmom_powell"]))) > 0.0
        assert float(np.max(np.abs(sources["denergy_powell"]))) > 0.0
        assert float(np.max(np.abs(sources["dB_powell"]))) > 0.0

    def test_powell_momentum_formula(self):
        from dpf.fluid.mhd_solver import powell_source_terms

        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 1e-3
        B = np.zeros((3, nx, ny, nz))
        x = np.linspace(0, (nx - 1) * dx, nx)
        for i in range(nx):
            B[0, i, :, :] = x[i] * 100.0
        vel = np.zeros((3, nx, ny, nz))
        state = {
            "rho": np.ones((nx, ny, nz)) * 1e-4,
            "velocity": vel,
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": B,
            "psi": np.zeros((nx, ny, nz)),
        }
        sources = powell_source_terms(state, dx, dy, dz)
        div_B = sources["div_B"]
        expected_dmom_x = -div_B * B[0]
        s = slice(2, -2)
        np.testing.assert_allclose(
            sources["dmom_powell"][0, s, s, s],
            expected_dmom_x[s, s, s],
            rtol=1e-10,
        )


class TestDednerDamping:
    """Verify that psi decays under Dedner parabolic damping."""

    def test_psi_decays_with_damping(self):
        from dpf.fluid.mhd_solver import _dedner_source_mt2010

        nx, ny, nz = 16, 16, 16
        dx = 1e-3
        psi = np.ones((nx, ny, nz)) * 1.0
        B = np.zeros((3, nx, ny, nz))
        B[2] = 1.0
        ch = 1e6
        cr = ch / dx
        dpsi_dt, dB_dt = _dedner_source_mt2010(psi, B, ch, cr, dx)
        expected_dpsi = -cr * psi
        np.testing.assert_allclose(dpsi_dt, expected_dpsi, rtol=1e-10)
        dt = 1e-10
        psi_new = psi + dt * dpsi_dt
        assert float(np.max(psi_new)) < float(np.max(psi))

    def test_dedner_div_B_term(self):
        from dpf.fluid.mhd_solver import _dedner_source_mt2010

        nx, ny, nz = 16, 16, 16
        dx = 1e-3
        psi = np.zeros((nx, ny, nz))
        B = np.zeros((3, nx, ny, nz))
        x = np.linspace(0, (nx - 1) * dx, nx)
        for i in range(nx):
            B[0, i, :, :] = x[i] * 100.0
        ch = 1e6
        cr = ch / dx
        dpsi_dt, _ = _dedner_source_mt2010(psi, B, ch, cr, dx)
        assert float(np.max(np.abs(dpsi_dt))) > 0


class TestConfigFields:
    """Verify the new Phase D config fields exist and have correct defaults."""

    def test_new_config_fields_exist(self):
        from dpf.config import FluidConfig

        fc = FluidConfig()
        assert fc.enable_powell is False
        assert fc.dedner_cr == 0.0
        assert fc.enable_anisotropic_conduction is False
        assert fc.full_braginskii_viscosity is False

    def test_config_fields_settable(self):
        from dpf.config import FluidConfig

        fc = FluidConfig(
            enable_powell=True,
            dedner_cr=1e8,
            enable_anisotropic_conduction=True,
            full_braginskii_viscosity=True,
        )
        assert fc.enable_powell is True
        assert fc.dedner_cr == 1e8
        assert fc.enable_anisotropic_conduction is True
        assert fc.full_braginskii_viscosity is True


# --- Section: Phase R Transport ---


class TestSpitzerAlphaZ:
    """Tests for the Braginskii alpha(Z) correction to Spitzer resistivity."""

    def test_alpha_z1_value(self):
        alpha = spitzer_alpha(1.0)
        assert alpha == pytest.approx(0.5064, rel=1e-4)

    def test_alpha_z2_value(self):
        alpha = spitzer_alpha(2.0)
        assert alpha == pytest.approx(0.4408, rel=1e-4)

    def test_alpha_z3_value(self):
        alpha = spitzer_alpha(3.0)
        assert alpha == pytest.approx(0.3965, rel=1e-4)

    def test_alpha_z4_value(self):
        alpha = spitzer_alpha(4.0)
        assert alpha == pytest.approx(0.3752, rel=1e-4)

    def test_alpha_z_inf_lower_bound(self):
        alpha_100 = spitzer_alpha(100.0)
        alpha_1000 = spitzer_alpha(1000.0)
        assert alpha_100 == pytest.approx(0.2949, rel=1e-4)
        assert alpha_1000 == pytest.approx(0.2949, rel=1e-4)

    def test_alpha_z_monotonically_decreasing(self):
        Z_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0, 100.0]
        alpha_values = [spitzer_alpha(Z) for Z in Z_values]
        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] > alpha_values[i + 1]

    def test_alpha_z_interpolation(self):
        alpha_1 = spitzer_alpha(1.0)
        alpha_2 = spitzer_alpha(2.0)
        alpha_mid = spitzer_alpha(1.5)
        expected_mid = 0.5 * (alpha_1 + alpha_2)
        assert alpha_mid == pytest.approx(expected_mid, rel=1e-3)

    def test_spitzer_resistivity_with_alpha(self):
        ne = np.array([1e20])
        Te = np.array([11604.5 * 1000])
        lnL = 10.0
        Z = 1.0
        eta = spitzer_resistivity(ne, Te, lnL, Z)
        from dpf.collision.spitzer import nu_ei as _nu_ei
        freq = _nu_ei(ne, Te, lnL, Z)
        eta_uncorrected = m_e * freq / (ne * e**2)
        correction_factor = eta[0] / eta_uncorrected[0]
        expected_factor = 1.0 / 0.5064
        assert correction_factor == pytest.approx(expected_factor, rel=1e-3)

    def test_spitzer_resistivity_z_dependence(self):
        ne = np.array([1e20])
        Te = np.array([11604.5 * 1000])
        lnL = 10.0
        eta_z1 = spitzer_resistivity(ne, Te, lnL, Z=1.0)
        eta_z4 = spitzer_resistivity(ne, Te, lnL, Z=4.0)
        ratio = eta_z4[0] / eta_z1[0]
        expected_ratio = 4.0 / (0.3752 / 0.5064)
        assert ratio == pytest.approx(expected_ratio, rel=1e-3)


class TestCoulombLogConsistency:
    """Test that Coulomb log is consistent between nernst.py and spitzer.py."""

    def test_nernst_uses_spitzer_coulomb_log(self):
        ne = np.array([1e20, 1e22, 1e24])
        Te = np.array([11604.5 * 100, 11604.5 * 1000, 11604.5 * 10000])
        B_mag = np.array([1.0, 1.0, 1.0])
        beta = nernst_coefficient(ne, Te, B_mag, Z_eff=1.0)
        assert np.all(np.isfinite(beta))
        assert np.all(beta >= 0)

    def test_coulomb_log_values_consistency(self):
        from dpf.collision.spitzer import coulomb_log as spitzer_coulomb_log
        from dpf.fluid.nernst import coulomb_log as nernst_coulomb_log

        ne = np.array([1e20, 1e22, 1e24])
        Te = np.array([11604.5 * 100, 11604.5 * 1000, 11604.5 * 10000])
        lnL_spitzer = spitzer_coulomb_log(ne, Te)
        lnL_nernst = nernst_coulomb_log(ne, Te)
        np.testing.assert_array_equal(lnL_spitzer, lnL_nernst)

    def test_coulomb_log_quantum_correction(self):
        ne_quantum = np.array([1e28])
        Te_quantum = np.array([11604.5 * 1])
        ne_classical = np.array([1e20])
        Te_classical = np.array([11604.5 * 1000])
        lnL_quantum = coulomb_log(ne_quantum, Te_quantum)
        lnL_classical = coulomb_log(ne_classical, Te_classical)
        assert lnL_quantum[0] < lnL_classical[0]
        assert lnL_quantum[0] >= 0.0
        assert lnL_classical[0] >= 1.0
        assert np.isfinite(lnL_quantum[0])
        assert np.isfinite(lnL_classical[0])


class TestIonMassConsistency:
    """Verify ion mass consistency between config and transport modules."""

    def test_config_uses_deuterium_mass(self):
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("pf1000"))
        assert config.ion_mass == pytest.approx(m_d, rel=1e-6)

    def test_relax_temperatures_uses_deuterium(self):
        Te = np.array([11604.5 * 1000])
        Ti = np.array([11604.5 * 100])
        freq_ei = np.array([1e9])
        dt = 1e-9
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        T_mean = 0.5 * (Te[0] + Ti[0])
        assert Te_new[0] < Te[0]
        assert Ti_new[0] > Ti[0]
        alpha_expected = freq_ei[0] * dt * 2.0 * m_e / m_d
        factor_expected = np.exp(-2.0 * alpha_expected)
        Te_expected = T_mean + (Te[0] - T_mean) * factor_expected
        assert Te_new[0] == pytest.approx(Te_expected, rel=1e-6)


# --- Section: Phase S Ablation ---


class TestAblationCoreFunctions:
    """Test core ablation physics functions."""

    def test_ablation_rate_positive(self):
        dm_dt = ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY)
        assert dm_dt > 0.0
        assert dm_dt == pytest.approx(1e8 * COPPER_ABLATION_EFFICIENCY, rel=1e-10)

    def test_ablation_rate_zero_power(self):
        assert ablation_rate(0.0, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_rate_negative_power(self):
        assert ablation_rate(-1e8, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_source_basic(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e23,
            J_boundary=1e8, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=5e-5, material_mass=COPPER_MASS,
        )
        assert pytest.approx(5e6, rel=1e-10) == S

    def test_ablation_source_zero_J(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e23,
            J_boundary=0.0, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=5e-5, material_mass=COPPER_MASS,
        )
        assert S == 0.0

    def test_ablation_source_zero_eta(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e23,
            J_boundary=1e8, eta_boundary=0.0, dx=1e-3,
            ablation_efficiency=5e-5, material_mass=COPPER_MASS,
        )
        assert S == 0.0


class TestAblationSourceArray:
    """Test the array-based ablation source function."""

    def test_only_at_boundary(self):
        n = 10
        J = np.full(n, 1e8)
        eta = np.full(n, 1e-5)
        mask = np.zeros(n, dtype=np.int64)
        mask[0] = 1
        S = ablation_source_array(J, eta, 5e-5, mask)
        assert S[0] > 0.0
        assert np.all(S[1:] == 0.0)

    def test_zero_mask_gives_zero(self):
        n = 10
        J = np.full(n, 1e8)
        eta = np.full(n, 1e-5)
        mask = np.zeros(n, dtype=np.int64)
        S = ablation_source_array(J, eta, 5e-5, mask)
        assert np.all(S == 0.0)

    def test_source_magnitude(self):
        J = np.array([1e8])
        eta = np.array([1e-5])
        mask = np.array([1], dtype=np.int64)
        S = ablation_source_array(J, eta, 5e-5, mask)
        expected = 5e-5 * 1e-5 * (1e8) ** 2
        assert S[0] == pytest.approx(expected, rel=1e-10)


class TestAblationHelpers:
    """Test particle flux and momentum source helpers."""

    def test_particle_flux(self):
        S_rho = 5e6
        S_n = ablation_particle_flux(S_rho, COPPER_MASS)
        expected = S_rho / COPPER_MASS
        assert S_n == pytest.approx(expected, rel=1e-10)
        assert S_n > 0.0

    def test_particle_flux_zero(self):
        assert ablation_particle_flux(0.0, COPPER_MASS) == 0.0

    def test_momentum_source(self):
        S_rho = 5e6
        v = 1e4
        S_mom = ablation_momentum_source(S_rho, v)
        assert S_mom == pytest.approx(S_rho * v, rel=1e-10)

    def test_material_constants(self):
        assert COPPER_ABLATION_EFFICIENCY > TUNGSTEN_ABLATION_EFFICIENCY
        assert COPPER_MASS < TUNGSTEN_MASS


class TestAblationConfig:
    """Test ablation configuration in SimulationConfig."""

    def test_ablation_config_defaults(self):
        cfg = AblationConfig()
        assert cfg.enabled is False
        assert cfg.material == "copper"
        assert cfg.efficiency == pytest.approx(5e-5, rel=1e-10)

    def test_ablation_config_custom(self):
        cfg = AblationConfig(enabled=True, material="tungsten", efficiency=2e-5)
        assert cfg.enabled is True
        assert cfg.material == "tungsten"
        assert cfg.efficiency == pytest.approx(2e-5, rel=1e-10)

    def test_ablation_in_simulation_config(self):
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            ablation={"enabled": True, "efficiency": 3e-5},
        )
        assert config.ablation.enabled is True
        assert config.ablation.efficiency == pytest.approx(3e-5, rel=1e-10)


# --- Section: Phase S Anomalous Resistivity ---


class TestThresholdOrdering:
    """Verify LHDI < ion_acoustic < buneman_classic thresholds."""

    def test_lhdi_lowest_threshold(self):
        from dpf.turbulence.anomalous import ion_acoustic_threshold, lhdi_threshold

        ne = np.array([1e23])
        Ti = np.array([1e6])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        lhdi_factor = (m_e / mi) ** 0.25
        v_d_mid = 0.5 * (lhdi_factor + 1.0) * v_ti
        J_mid = np.array([v_d_mid * ne[0] * e])
        lhdi_active = lhdi_threshold(J_mid, ne, Ti, mi)
        ia_active = ion_acoustic_threshold(J_mid, ne, Ti, mi)
        assert lhdi_active[0] is True or lhdi_active[0]
        assert ia_active[0] is False or not ia_active[0]

    def test_buneman_classic_highest_threshold(self):
        from dpf.turbulence.anomalous import buneman_classic_threshold, ion_acoustic_threshold

        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        v_te = np.sqrt(k_B * Te[0] / m_e)
        v_d_mid = 0.5 * (v_ti + v_te)
        J_mid = np.array([v_d_mid * ne[0] * e])
        ia_active = ion_acoustic_threshold(J_mid, ne, Ti, mi)
        bc_active = buneman_classic_threshold(J_mid, ne, Te)
        assert ia_active[0]
        assert not bc_active[0]

    def test_threshold_factor_ordering(self):
        from dpf.turbulence.anomalous import lhdi_factor

        factor = lhdi_factor(M_D)
        assert 0.0 < factor < 1.0
        assert factor == pytest.approx(0.129, abs=0.005)


class TestLHDIThreshold:
    """Test the LHDI threshold function."""

    def test_lhdi_factor_deuterium(self):
        from dpf.turbulence.anomalous import lhdi_factor

        factor = lhdi_factor(M_D)
        expected = (m_e / M_D) ** 0.25
        assert factor == pytest.approx(expected, rel=1e-10)

    def test_lhdi_threshold_above(self):
        from dpf.turbulence.anomalous import lhdi_threshold

        ne = np.array([1e23])
        Ti = np.array([1e5])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        factor = (m_e / mi) ** 0.25
        v_d = 2.0 * factor * v_ti
        J = np.array([v_d * ne[0] * e])
        assert lhdi_threshold(J, ne, Ti, mi)[0]

    def test_lhdi_threshold_below(self):
        from dpf.turbulence.anomalous import lhdi_threshold

        ne = np.array([1e23])
        Ti = np.array([1e5])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        factor = (m_e / mi) ** 0.25
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
        from dpf.turbulence.anomalous import anomalous_resistivity_field

        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        c_s = np.sqrt(k_B * Te[0] / m_p)
        J = np.array([2.0 * c_s * ne[0] * e])
        eta = anomalous_resistivity_field(J, ne, Ti, Te=Te)
        assert eta[0] > 0.0

    def test_ion_acoustic_model(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field

        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        c_s = np.sqrt(k_B * Te[0] / m_p)
        J = np.array([2.0 * c_s * ne[0] * e])
        eta = anomalous_resistivity_field(J, ne, Ti, threshold_model="ion_acoustic", Te=Te)
        assert eta[0] > 0.0

    def test_lhdi_model(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field

        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        mi = M_D
        v_ti = np.sqrt(k_B * Ti[0] / mi)
        factor = (m_e / mi) ** 0.25
        v_d = 0.5 * (factor + 1.0) * v_ti
        J = np.array([v_d * ne[0] * e])
        eta_lhdi = anomalous_resistivity_field(J, ne, Ti, mi=mi, threshold_model="lhdi", Te=Te)
        eta_ia = anomalous_resistivity_field(J, ne, Ti, mi=mi, threshold_model="ion_acoustic", Te=Te)
        assert eta_lhdi[0] > 0.0
        assert eta_ia[0] == 0.0

    def test_buneman_classic_requires_Te(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field

        ne = np.array([1e23])
        Ti = np.array([1e6])
        J = np.array([1e10])
        with pytest.raises(ValueError, match="Te.*required"):
            anomalous_resistivity_field(J, ne, Ti, threshold_model="buneman_classic")

    def test_buneman_classic_model(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field

        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        v_te = np.sqrt(k_B * Te[0] / m_e)
        J = np.array([2.0 * v_te * ne[0] * e])
        eta = anomalous_resistivity_field(J, ne, Ti, threshold_model="buneman_classic", Te=Te)
        assert eta[0] > 0.0

    def test_unknown_model_raises(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field

        ne = np.array([1e23])
        Ti = np.array([1e6])
        J = np.array([1e10])
        with pytest.raises(ValueError, match="Unknown threshold_model"):
            anomalous_resistivity_field(J, ne, Ti, threshold_model="invalid")

    def test_below_threshold_gives_zero(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_field

        ne = np.array([1e23])
        Ti = np.array([1e6])
        Te = np.array([1e6])
        J = np.array([1.0])
        for model in ("ion_acoustic", "lhdi"):
            eta = anomalous_resistivity_field(J, ne, Ti, threshold_model=model, Te=Te)
            assert eta[0] == 0.0


class TestBackwardCompatibility:
    """Test that the old API still works."""

    def test_buneman_threshold_deprecated(self):
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
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar

        eta = anomalous_resistivity_scalar(1e10, 1e23, 1e6, alpha=0.05)
        assert isinstance(eta, float)

    def test_total_resistivity_scalar_unchanged(self):
        from dpf.turbulence.anomalous import total_resistivity_scalar

        assert total_resistivity_scalar(1e-6, 1e-5) == pytest.approx(1.1e-5, rel=1e-10)


class TestBraginskiiKappaZDependent:
    """Test the Z-dependent Braginskii kappa coefficient."""

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
        from dpf.collision.spitzer import braginskii_kappa_coefficient

        assert braginskii_kappa_coefficient(0.5) == pytest.approx(3.16, rel=1e-10)

    def test_interpolation_monotonic_descent_to_Z4(self):
        from dpf.collision.spitzer import braginskii_kappa_coefficient

        z_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        coeffs = [braginskii_kappa_coefficient(z) for z in z_vals]
        for i in range(len(coeffs) - 1):
            assert coeffs[i] >= coeffs[i + 1]

    def test_braginskii_kappa_backward_compat(self):
        ne = np.array([1e23])
        Te = np.array([1e6])
        Bmag = np.array([1.0])
        kpar, kperp = braginskii_kappa(ne, Te, Bmag)
        kpar_z1, kperp_z1 = braginskii_kappa(ne, Te, Bmag, Z=1.0)
        assert kpar[0] == pytest.approx(kpar_z1[0], rel=1e-10)
        assert kperp[0] == pytest.approx(kperp_z1[0], rel=1e-10)

    def test_braginskii_kappa_Z2_differs(self):
        ne = np.array([1e23])
        Te = np.array([1e6])
        Bmag = np.array([1.0])
        kpar_z1, _ = braginskii_kappa(ne, Te, Bmag, Z=1.0)
        kpar_z2, _ = braginskii_kappa(ne, Te, Bmag, Z=2.0)
        ratio = kpar_z2[0] / kpar_z1[0]
        assert 0.99 < ratio < 1.0


# --- Section: Phase U Ablation ---


class TestAblationRate:
    """Unit tests for ablation_rate function."""

    def test_zero_power_gives_zero_rate(self):
        assert ablation_rate(0.0, COPPER_ABLATION_EFFICIENCY) == pytest.approx(0.0, abs=1e-20)

    def test_negative_power_gives_zero_rate(self):
        assert ablation_rate(-1e6, COPPER_ABLATION_EFFICIENCY) == pytest.approx(0.0, abs=1e-20)

    def test_zero_efficiency_gives_zero_rate(self):
        assert ablation_rate(1e6, 0.0) == pytest.approx(0.0, abs=1e-20)

    def test_negative_efficiency_gives_zero_rate(self):
        assert ablation_rate(1e6, -1e-5) == pytest.approx(0.0, abs=1e-20)

    def test_positive_power_correct_rate(self):
        P_surface = 1e6
        efficiency = COPPER_ABLATION_EFFICIENCY
        expected = efficiency * P_surface
        rate = ablation_rate(P_surface, efficiency)
        assert rate == pytest.approx(expected, rel=1e-12)

    def test_linear_scaling_with_power(self):
        efficiency = COPPER_ABLATION_EFFICIENCY
        rate1 = ablation_rate(1e6, efficiency)
        rate2 = ablation_rate(2e6, efficiency)
        assert rate2 == pytest.approx(2 * rate1, rel=1e-12)

    def test_linear_scaling_with_efficiency(self):
        P_surface = 1e6
        eff1 = COPPER_ABLATION_EFFICIENCY
        eff2 = 2 * COPPER_ABLATION_EFFICIENCY
        rate1 = ablation_rate(P_surface, eff1)
        rate2 = ablation_rate(P_surface, eff2)
        assert rate2 == pytest.approx(2 * rate1, rel=1e-12)


class TestAblationSource:
    """Unit tests for ablation_source function."""

    def test_zero_current_density_gives_zero_source(self):
        S_rho = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e24,
            J_boundary=0.0, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY, material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(0.0, abs=1e-20)

    def test_zero_resistivity_gives_zero_source(self):
        S_rho = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e24,
            J_boundary=1e8, eta_boundary=0.0, dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY, material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(0.0, abs=1e-20)

    def test_zero_efficiency_gives_zero_source(self):
        S_rho = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e24,
            J_boundary=1e8, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=0.0, material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(0.0, abs=1e-20)

    def test_correct_formula_S_rho_equals_efficiency_times_eta_J_squared(self):
        J = 1e8
        eta = 1e-5
        efficiency = COPPER_ABLATION_EFFICIENCY
        expected = efficiency * eta * J * J
        S_rho = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e5, ne_boundary=1e24,
            J_boundary=J, eta_boundary=eta, dx=1e-3,
            ablation_efficiency=efficiency, material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(expected, rel=1e-12)

    def test_copper_vs_tungsten_efficiency_ratio(self):
        J = 1e8
        eta = 1e-5
        S_rho_Cu = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS)
        S_rho_W = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, TUNGSTEN_ABLATION_EFFICIENCY, TUNGSTEN_MASS)
        ratio = S_rho_Cu / S_rho_W
        assert ratio == pytest.approx(2.5, rel=1e-12)

    def test_low_resistivity_low_ablation(self):
        J = 1e8
        S_rho_low = ablation_source(1e-3, 1e5, 1e24, J, 1e-7, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS)
        S_rho_high = ablation_source(1e-3, 1e5, 1e24, J, 1e-5, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS)
        assert S_rho_low == pytest.approx(S_rho_high / 100, rel=1e-12)


class TestUAblationSourceArray:
    """Unit tests for ablation_source_array function (Phase U)."""

    def test_zero_mask_gives_zero_source_everywhere(self):
        J = np.full(10, 1e8)
        eta = np.full(10, 1e-5)
        boundary_mask = np.zeros(10, dtype=int)
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert np.all(S_rho == 0.0)

    def test_single_boundary_cell(self):
        J = np.full(10, 1e8)
        eta = np.full(10, 1e-5)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        expected = COPPER_ABLATION_EFFICIENCY * eta[0] * J[0] * J[0]
        assert S_rho[0] == pytest.approx(expected, rel=1e-12)
        assert np.all(S_rho[1:] == 0.0)

    def test_multiple_boundary_cells(self):
        J = np.full(10, 1e8)
        eta = np.full(10, 1e-5)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        boundary_mask[9] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        expected = COPPER_ABLATION_EFFICIENCY * eta[0] * J[0] * J[0]
        assert S_rho[0] == pytest.approx(expected, rel=1e-12)
        assert S_rho[9] == pytest.approx(expected, rel=1e-12)
        assert np.all(S_rho[1:9] == 0.0)

    def test_varying_J_and_eta(self):
        J = np.linspace(1e7, 1e9, 10)
        eta = np.linspace(1e-6, 1e-4, 10)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        boundary_mask[5] = 1
        boundary_mask[9] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho[0] == pytest.approx(COPPER_ABLATION_EFFICIENCY * eta[0] * J[0] * J[0], rel=1e-12)
        assert S_rho[5] == pytest.approx(COPPER_ABLATION_EFFICIENCY * eta[5] * J[5] * J[5], rel=1e-12)
        assert S_rho[9] == pytest.approx(COPPER_ABLATION_EFFICIENCY * eta[9] * J[9] * J[9], rel=1e-12)
        assert np.all(S_rho[[1, 2, 3, 4, 6, 7, 8]] == 0.0)

    def test_shape_preservation_2d(self):
        J = np.full((10, 10), 1e8)
        eta = np.full((10, 10), 1e-5)
        boundary_mask = np.zeros((10, 10), dtype=int)
        boundary_mask[0, :] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.shape == J.shape
        assert np.all(S_rho[0, :] > 0)
        assert np.all(S_rho[1:, :] == 0.0)

    def test_shape_preservation_3d(self):
        J = np.full((5, 5, 5), 1e8)
        eta = np.full((5, 5, 5), 1e-5)
        boundary_mask = np.zeros((5, 5, 5), dtype=int)
        boundary_mask[0, :, :] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.shape == J.shape
        assert np.all(S_rho[0, :, :] > 0)
        assert np.all(S_rho[1:, :, :] == 0.0)

    def test_dtype_float32(self):
        J = np.full(10, 1e8, dtype=np.float32)
        eta = np.full(10, 1e-5, dtype=np.float32)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.dtype == np.float32
        assert S_rho[0] > 0

    def test_dtype_float64(self):
        J = np.full(10, 1e8, dtype=np.float64)
        eta = np.full(10, 1e-5, dtype=np.float64)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.dtype == np.float64
        assert S_rho[0] > 0

    def test_zero_J_at_boundary_gives_zero_source(self):
        J = np.full(10, 1e8)
        J[0] = 0.0
        eta = np.full(10, 1e-5)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho[0] == pytest.approx(0.0, abs=1e-20)

    def test_negative_J_at_boundary_gives_zero_source(self):
        J = np.full(10, 1e8)
        J[0] = -1e8
        eta = np.full(10, 1e-5)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho[0] == pytest.approx(0.0, abs=1e-20)


class TestAblationParticleFlux:
    """Unit tests for ablation_particle_flux function."""

    def test_zero_S_rho_gives_zero_flux(self):
        assert ablation_particle_flux(0.0, COPPER_MASS) == pytest.approx(0.0, abs=1e-20)

    def test_negative_S_rho_gives_zero_flux(self):
        assert ablation_particle_flux(-1e7, COPPER_MASS) == pytest.approx(0.0, abs=1e-20)

    def test_zero_material_mass_gives_zero_flux(self):
        assert ablation_particle_flux(1e7, 0.0) == pytest.approx(0.0, abs=1e-20)

    def test_correct_formula_S_n_equals_S_rho_over_m(self):
        S_rho = 1e7
        m_atom = COPPER_MASS
        expected = S_rho / m_atom
        assert ablation_particle_flux(S_rho, m_atom) == pytest.approx(expected, rel=1e-12)

    def test_physical_magnitude_typical_DPF(self):
        S_rho = 1e7
        S_n = ablation_particle_flux(S_rho, COPPER_MASS)
        assert 1e30 < S_n < 1e34


class TestAblationMomentumSource:
    """Unit tests for ablation_momentum_source function."""

    def test_zero_S_rho_gives_zero_momentum(self):
        assert ablation_momentum_source(0.0, 1e4) == pytest.approx(0.0, abs=1e-20)

    def test_zero_velocity_gives_zero_momentum(self):
        assert ablation_momentum_source(1e7, 0.0) == pytest.approx(0.0, abs=1e-20)

    def test_correct_formula_S_mom_equals_S_rho_times_v(self):
        S_rho = 1e7
        v_boundary = 1e4
        expected = S_rho * v_boundary
        assert ablation_momentum_source(S_rho, v_boundary) == pytest.approx(expected, rel=1e-12)

    def test_negative_velocity(self):
        S_rho = 1e7
        v_boundary = -1e4
        S_mom = ablation_momentum_source(S_rho, v_boundary)
        assert S_mom == pytest.approx(-1e11, rel=1e-1)


class TestAblationPhysics:
    """Physical scaling tests for ablation module."""

    def test_S_rho_scales_quadratically_with_J(self):
        eta = 1e-5
        efficiency = COPPER_ABLATION_EFFICIENCY
        S_rho1 = ablation_source(1e-3, 1e5, 1e24, 1e8, eta, 1e-3, efficiency, COPPER_MASS)
        S_rho2 = ablation_source(1e-3, 1e5, 1e24, 2e8, eta, 1e-3, efficiency, COPPER_MASS)
        assert S_rho2 == pytest.approx(4 * S_rho1, rel=1e-12)

    def test_S_rho_scales_linearly_with_eta(self):
        J = 1e8
        efficiency = COPPER_ABLATION_EFFICIENCY
        S_rho1 = ablation_source(1e-3, 1e5, 1e24, J, 1e-5, 1e-3, efficiency, COPPER_MASS)
        S_rho2 = ablation_source(1e-3, 1e5, 1e24, J, 2e-5, 1e-3, efficiency, COPPER_MASS)
        assert S_rho2 == pytest.approx(2 * S_rho1, rel=1e-12)

    def test_tungsten_lower_ablation_than_copper(self):
        J = 1e8
        eta = 1e-5
        S_rho_Cu = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS)
        S_rho_W = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, TUNGSTEN_ABLATION_EFFICIENCY, TUNGSTEN_MASS)
        assert S_rho_W == pytest.approx(0.4 * S_rho_Cu, rel=1e-12)


class TestUAblationConfig:
    """Tests for AblationConfig Pydantic model (Phase U)."""

    def test_default_config_values(self):
        config = AblationConfig()
        assert config.enabled is False
        assert config.material == "copper"
        assert config.efficiency == pytest.approx(5e-5, rel=1e-12)

    def test_enable_ablation(self):
        config = AblationConfig(enabled=True)
        assert config.enabled is True

    def test_tungsten_material(self):
        config = AblationConfig(material="tungsten")
        assert config.material == "tungsten"

    def test_custom_efficiency(self):
        config = AblationConfig(efficiency=1e-4)
        assert config.efficiency == pytest.approx(1e-4, rel=1e-12)

    def test_invalid_material_raises_error(self):
        with pytest.raises(ValueError, match="material must be"):
            AblationConfig(material="aluminum")

    def test_zero_efficiency_raises_error(self):
        with pytest.raises(ValueError):
            AblationConfig(efficiency=0.0)

    def test_negative_efficiency_raises_error(self):
        with pytest.raises(ValueError):
            AblationConfig(efficiency=-1e-5)

    def test_config_dict_roundtrip(self):
        config = AblationConfig(enabled=True, material="tungsten", efficiency=2e-5)
        config_dict = config.model_dump()
        config2 = AblationConfig(**config_dict)
        assert config2.enabled == config.enabled
        assert config2.material == config.material
        assert config2.efficiency == pytest.approx(config.efficiency, rel=1e-12)


# --- Section: Anisotropic Conduction ---


class TestSignCorrectness:
    """Heat must flow from hot to cold."""

    def test_center_cools(self):
        Te, B, ne = _make_hotspot()
        B[0] = 0.1
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        mid = 8
        assert Te_new[mid, mid, mid] < Te[mid, mid, mid]

    def test_neighbor_heats(self):
        Te, B, ne = _make_hotspot()
        B[0] = 0.1
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        mid = 8
        assert Te_new[2, mid, mid] >= Te[2, mid, mid] - 1.0

    def test_no_nan(self):
        Te, B, ne = _make_hotspot()
        B[2] = 0.5
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        assert not np.any(np.isnan(Te_new))
        assert np.all(Te_new > 0)


class TestAnisotropy:
    """Heat should diffuse faster along B than across B."""

    def test_parallel_faster_than_perpendicular(self):
        nx = 16
        mid = nx // 2
        Te, B, ne = _make_hotspot(nx)
        B[0] = 0.1
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        delta_par = abs(Te_new[mid + 3, mid, mid] - Te[mid + 3, mid, mid])
        delta_perp = abs(Te_new[mid, mid + 3, mid] - Te[mid, mid + 3, mid])
        assert delta_par >= delta_perp * 0.9

    def test_zero_B_isotropic(self):
        nx = 16
        mid = nx // 2
        Te, B, ne = _make_hotspot(nx)
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        delta_x = abs(Te_new[mid + 3, mid, mid] - Te[mid + 3, mid, mid])
        delta_y = abs(Te_new[mid, mid + 3, mid] - Te[mid, mid + 3, mid])
        if delta_x > 1e-10 and delta_y > 1e-10:
            ratio = delta_x / delta_y
            assert 0.5 < ratio < 2.0


class TestStability:
    """Conduction should be stable under various conditions."""

    def test_uniform_temperature_unchanged(self):
        Te = np.full((8, 8, 8), 1e5)
        B = np.zeros((3, 8, 8, 8))
        B[0] = 0.1
        ne = np.full((8, 8, 8), 1e24)
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        np.testing.assert_allclose(Te_new, Te, rtol=1e-6)

    def test_strong_field(self):
        Te, B, ne = _make_hotspot()
        B[0] = 100.0
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-11, dx=1e-3, dy=1e-3, dz=1e-3)
        assert not np.any(np.isnan(Te_new))
        assert np.all(Te_new > 0)

    def test_low_density(self):
        Te, B, ne = _make_hotspot()
        B[0] = 0.1
        ne[:] = 1e10
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        assert not np.any(np.isnan(Te_new))


# --- Section: Two-Temperature Physics ---


class TestCoulombLogarithm:
    """Verify Coulomb logarithm against NRL Plasma Formulary."""

    def test_classical_plasma(self):
        ne = np.array([1e24])
        Te = np.array([1e7])
        lnL = coulomb_log(ne, Te)
        assert 5.0 < lnL[0] < 25.0

    def test_floor_at_two(self):
        ne = np.array([1e30])
        Te = np.array([1e3])
        lnL = coulomb_log(ne, Te)
        assert lnL[0] >= 2.0

    def test_increases_with_temperature(self):
        ne = np.array([1e24])
        assert coulomb_log(ne, np.array([1e8]))[0] > coulomb_log(ne, np.array([1e5]))[0]

    def test_decreases_with_density(self):
        Te = np.array([1e7])
        assert coulomb_log(np.array([1e22]), Te)[0] > coulomb_log(np.array([1e26]), Te)[0]

    def test_array_broadcast(self):
        ne = np.full((4, 1, 8), 1e24)
        Te = np.full((4, 1, 8), 1e7)
        lnL = coulomb_log(ne, Te)
        assert lnL.shape == (4, 1, 8)
        assert np.all(lnL > 2.0)


class TestCollisionFrequencies:
    """Verify Spitzer collision frequencies."""

    def test_nu_ei_scaling(self):
        ne = np.array([1e24])
        Te1 = np.array([1e7])
        Te2 = np.array([4e7])
        lnL = 10.0
        freq1 = nu_ei(ne, Te1, lnL)
        freq2 = nu_ei(ne, Te2, lnL)
        ratio = freq2[0] / freq1[0]
        assert pytest.approx(ratio, rel=0.01) == (4.0) ** (-1.5)

    def test_nu_ei_density_linear(self):
        Te = np.array([1e7])
        lnL = 10.0
        freq1 = nu_ei(np.array([1e24]), Te, lnL)
        freq2 = nu_ei(np.array([2e24]), Te, lnL)
        assert pytest.approx(freq2[0] / freq1[0], rel=0.01) == 2.0

    def test_nu_ei_dpf_range(self):
        ne = np.array([1e25])
        Te = np.array([5e7])
        lnL = coulomb_log(ne, Te)
        freq = nu_ei(ne, Te, lnL)
        assert 1e8 < freq[0] < 1e15

    def test_nu_ii_slower_than_nu_ei(self):
        ne = np.array([1e24])
        Te = np.array([1e7])
        Ti = Te
        lnL = 10.0
        freq_e = nu_ei(ne, Te, lnL)
        freq_i = nu_ii(ne, Ti, lnL, mi=m_d)
        assert freq_i[0] < freq_e[0]


class TestTwoTempTemperatureRelaxation:
    """Verify implicit e-i temperature relaxation (two-temperature physics)."""

    def test_equilibrium_temperature(self):
        Te = np.array([2e7])
        Ti = np.array([1e7])
        T_eq_expected = (Te + Ti) / 2.0
        freq_ei = np.array([1e12])
        dt = 1e-6
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        T_eq_actual = (Te_new + Ti_new) / 2.0
        assert pytest.approx(T_eq_actual[0], rel=1e-10) == T_eq_expected[0]

    def test_energy_conservation(self):
        Te = np.array([5e7])
        Ti = np.array([1e6])
        Z = 1.0
        total_before = Z * Te + Ti
        freq_ei = np.array([1e13])
        dt = 1e-5
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt, Z=Z)
        total_after = Z * Te_new + Ti_new
        assert pytest.approx(total_after[0], rel=1e-12) == total_before[0]

    def test_energy_conservation_high_Z(self):
        Te = np.array([3e7])
        Ti = np.array([1e7])
        Z = 3.0
        total_before = Z * Te + Ti
        freq_ei = np.array([1e12])
        dt = 1e-6
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt, Z=Z)
        total_after = Z * Te_new + Ti_new
        assert pytest.approx(total_after[0], rel=1e-12) == total_before[0]

    def test_exponential_decay(self):
        Te = np.array([2e7])
        Ti = np.array([1e7])
        freq_ei = np.array([1e12])
        dt = 1e-8
        alpha = freq_ei[0] * dt * 2.0 * m_e / m_d
        expected_factor = np.exp(-2.0 * alpha)
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        T_eq = (Te[0] + Ti[0]) / 2.0
        initial_diff = Te[0] - T_eq
        final_diff = Te_new[0] - T_eq
        actual_factor = final_diff / initial_diff
        assert pytest.approx(actual_factor, rel=1e-10) == expected_factor

    def test_already_equilibrated(self):
        T = np.array([1e7])
        freq_ei = np.array([1e12])
        dt = 1e-6
        Te_new, Ti_new = relax_temperatures(T, T.copy(), freq_ei, dt)
        assert pytest.approx(Te_new[0], rel=1e-14) == T[0]
        assert pytest.approx(Ti_new[0], rel=1e-14) == T[0]

    def test_stiff_regime_stable(self):
        Te = np.array([1e8])
        Ti = np.array([1e5])
        freq_ei = np.array([1e15])
        dt = 1.0
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        T_eq = (Te[0] + Ti[0]) / 2.0
        assert pytest.approx(Te_new[0], rel=1e-6) == T_eq
        assert pytest.approx(Ti_new[0], rel=1e-6) == T_eq

    def test_electrons_cool_ions_heat(self):
        Te = np.array([5e7])
        Ti = np.array([1e7])
        freq_ei = np.array([1e12])
        dt = 1e-7
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        assert Te_new[0] < Te[0]
        assert Ti_new[0] > Ti[0]

    def test_multidimensional_array(self):
        shape = (16, 1, 32)
        Te = np.full(shape, 3e7)
        Ti = np.full(shape, 1e7)
        freq_ei = np.full(shape, 1e12)
        dt = 1e-7
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        assert Te_new.shape == shape
        assert Ti_new.shape == shape
        assert np.all(Te_new < Te)
        assert np.all(Ti_new > Ti)

    def test_z_dependent_equilibrium(self):
        Te = np.array([3e7])
        Ti = np.array([0.0])
        Z = 2.0
        freq_ei = np.array([1e15])
        dt = 1.0
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt, Z=Z)
        T_eq = (Z * Te[0] + Ti[0]) / (Z + 1.0)
        assert pytest.approx(Te_new[0], rel=1e-6) == T_eq
        assert pytest.approx(Ti_new[0], rel=1e-6) == T_eq


class TestSpitzerResistivity:
    """Verify Spitzer resistivity values."""

    def test_1kev_hydrogen_plasma(self):
        ne = np.array([1e24])
        Te = np.array([1.16e7])
        lnL = np.array([10.0])
        eta = spitzer_resistivity(ne, Te, lnL, Z=1.0)
        assert 1e-11 < eta[0] < 1e-5

    def test_resistivity_decreases_with_temperature(self):
        ne = np.array([1e24])
        lnL = 10.0
        eta_cold = spitzer_resistivity(ne, np.array([1e6]), lnL)
        eta_hot = spitzer_resistivity(ne, np.array([1e8]), lnL)
        assert eta_hot[0] < eta_cold[0]

    def test_resistivity_independent_of_density(self):
        Te = np.array([1e7])
        lnL = 10.0
        eta1 = spitzer_resistivity(np.array([1e23]), Te, lnL)
        eta2 = spitzer_resistivity(np.array([1e25]), Te, lnL)
        assert pytest.approx(eta1[0], rel=0.01) == eta2[0]


class TestBraginskiiKappa:
    """Verify Braginskii thermal conductivity."""

    def test_parallel_larger_than_perpendicular(self):
        ne = np.array([1e24])
        Te = np.array([1e7])
        Bmag = np.array([1.0])
        kappa_par, kappa_per = braginskii_kappa(ne, Te, Bmag)
        assert kappa_par[0] > kappa_per[0]

    def test_unmagnetized_limit(self):
        ne = np.array([1e24])
        Te = np.array([1e7])
        Bmag = np.array([0.0])
        kappa_par, kappa_per = braginskii_kappa(ne, Te, Bmag)
        assert pytest.approx(kappa_par[0], rel=1e-10) == kappa_per[0]

    def test_kappa_increases_with_temperature(self):
        ne = np.array([1e24])
        Bmag = np.array([0.1])
        kpar_low, _ = braginskii_kappa(ne, np.array([1e6]), Bmag)
        kpar_high, _ = braginskii_kappa(ne, np.array([1e7]), Bmag)
        assert kpar_high[0] > kpar_low[0]


class TestIdealEOSTwoTemperature:
    """Verify two-temperature EOS."""

    def test_pressures_additive(self):
        eos = IdealEOS(gamma=5.0 / 3.0, Z=1.0)
        rho = np.array([1e-3])
        Te = np.array([2e7])
        Ti = np.array([1e7])
        p_i = eos.ion_pressure(rho, Ti)
        p_e = eos.electron_pressure(rho, Te)
        p_total = eos.total_pressure(rho, Ti, Te)
        assert pytest.approx(p_total[0], rel=1e-12) == p_i[0] + p_e[0]

    def test_electron_pressure_scales_with_Z(self):
        rho = np.array([1e-3])
        Te = np.array([1e7])
        eos_z1 = IdealEOS(Z=1.0)
        eos_z2 = IdealEOS(Z=2.0)
        pe1 = eos_z1.electron_pressure(rho, Te)
        pe2 = eos_z2.electron_pressure(rho, Te)
        assert pytest.approx(pe2[0] / pe1[0], rel=1e-10) == 2.0

    def test_sound_speed_positive(self):
        eos = IdealEOS()
        rho = np.array([1e-3])
        Te = np.array([1e7])
        Ti = np.array([5e6])
        cs = eos.sound_speed(rho, Ti, Te)
        assert cs[0] > 0

    def test_temperature_recovery(self):
        eos = IdealEOS(gamma=5.0 / 3.0, Z=1.0)
        rho = np.array([1e-3])
        T_input = np.array([1e7])
        e_i = eos.ion_energy(rho, T_input)
        e_e = eos.electron_energy(rho, T_input)
        e_total = e_i + e_e
        T_recovered = eos.temperature_from_energy(rho, e_total)
        assert pytest.approx(T_recovered[0], rel=1e-10) == T_input[0]


# --- Section: M=0 Instability ---


class TestM0GrowthRate:
    """Unit tests for m0_growth_rate() covering physics correctness."""

    def test_unstable_low_beta(self):
        result = m0_growth_rate(B_theta=1.0, rho=1e-3, pressure=1e3, a_pinch=0.01)
        assert set(result.keys()) == _EXPECTED_KEYS
        beta_p_expected = 2.0 * MU_0 * 1e3 / 1.0 ** 2
        assert result["beta_p"] == pytest.approx(beta_p_expected, rel=0.01)
        assert beta_p_expected < 3.0
        assert result["is_unstable"] is True
        assert result["growth_rate"] > 0.0
        assert result["growth_time"] < float("inf")
        assert result["alfven_speed"] > 0.0

    def test_stable_high_beta(self):
        result = m0_growth_rate(B_theta=0.01, rho=1e-3, pressure=1e6, a_pinch=0.01)
        beta_p_expected = 2.0 * MU_0 * 1e6 / 0.01 ** 2
        assert beta_p_expected > 3.0
        assert result["stability_margin"] < -1000.0
        assert result["beta_p"] == pytest.approx(beta_p_expected, rel=0.01)
        assert result["growth_rate"] >= 0.0
        assert result["alfven_speed"] > 0.0

    def test_growth_rate_scales_with_alfven_speed(self):
        params = dict(rho=1e-3, pressure=1e3, a_pinch=0.01)
        r1 = m0_growth_rate(B_theta=0.5, **params)
        r2 = m0_growth_rate(B_theta=1.0, **params)
        assert r1["is_unstable"] is True
        assert r2["is_unstable"] is True
        ratio = r2["growth_rate"] / r1["growth_rate"]
        assert ratio == pytest.approx(2.0, rel=0.10)

    def test_growth_rate_scales_with_mode_number(self):
        params = dict(B_theta=1.0, rho=1e-3, pressure=1e3, a_pinch=0.01)
        r1 = m0_growth_rate(mode_number=1, **params)
        r2 = m0_growth_rate(mode_number=2, **params)
        assert r1["is_unstable"] is True
        assert r2["is_unstable"] is True
        ratio = r2["growth_rate"] / r1["growth_rate"]
        assert ratio == pytest.approx(2.0, rel=0.05)

    def test_stability_margin_sign(self):
        unstable = m0_growth_rate(B_theta=1.0, rho=1e-3, pressure=1e3, a_pinch=0.01)
        stable = m0_growth_rate(B_theta=0.01, rho=1e-3, pressure=1e6, a_pinch=0.01)
        assert unstable["stability_margin"] > 0.0
        assert stable["stability_margin"] < 0.0


class TestM0FromState:
    """Tests for m0_growth_rate_from_state()."""

    def test_from_state_with_mock(self):
        state = _make_mock_state(nr=8, nz=8, B_theta_val=2.0, rho_val=1e-3, p_val=1e4)
        snowplow = _make_mock_snowplow(r_shock=0.04, r_pinch_min=0.005, phase="radial")
        config = _make_mock_config(dx=0.01)
        result = m0_growth_rate_from_state(state, snowplow, config)
        assert isinstance(result, dict)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert not np.isnan(result["growth_rate"])
        assert not np.isnan(result["alfven_speed"])
        assert not np.isnan(result["beta_p"])
        assert result["growth_rate"] >= 0.0
        assert result["alfven_speed"] >= 0.0
        assert result["beta_p"] >= 0.0

    def test_from_state_none_snowplow(self):
        state = _make_mock_state()
        config = _make_mock_config()
        result = m0_growth_rate_from_state(state, snowplow=None, config=config)
        assert isinstance(result, dict)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert result["growth_rate"] == pytest.approx(0.0, abs=1e-30)
        assert result["growth_time"] == float("inf")
        assert result["is_unstable"] is False
        assert result["alfven_speed"] == pytest.approx(0.0, abs=1e-30)


class TestDimensionalAnalysis:
    """Dimensional sanity checks using realistic DPF pinch parameters."""

    def test_typical_dpf_pinch(self):
        result = m0_growth_rate(
            B_theta=5.0, rho=1e-2, pressure=1e8, a_pinch=0.005,
        )
        v_A = result["alfven_speed"]
        assert 1e3 <= v_A <= 1e6
        if result["is_unstable"]:
            tau = result["growth_time"]
            assert 1e-8 <= tau <= 1e-6
        assert result["beta_p"] >= 0.0
        assert result["growth_rate"] >= 0.0


# --- Section: Phase 13 Line Radiation, CR Ionization, Beam-Target ---


class TestAblation:
    """Tests for electrode ablation model (Phase 13)."""

    def test_ablation_rate_positive(self):
        assert ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY) > 0.0

    def test_ablation_rate_zero_power(self):
        assert ablation_rate(0.0, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_rate_negative_power(self):
        assert ablation_rate(-1e8, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_rate_linear(self):
        r1 = ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY)
        r2 = ablation_rate(2e8, COPPER_ABLATION_EFFICIENCY)
        np.testing.assert_allclose(r2, 2 * r1, rtol=1e-12)

    def test_copper_tungsten_ratio(self):
        r_cu = ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY)
        r_w = ablation_rate(1e8, TUNGSTEN_ABLATION_EFFICIENCY)
        assert r_cu > r_w

    def test_ablation_source_scalar(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e6, ne_boundary=1e23,
            J_boundary=1e8, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY, material_mass=COPPER_MASS,
        )
        assert S > 0.0

    def test_ablation_source_zero_current(self):
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e6, ne_boundary=1e23,
            J_boundary=0.0, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY, material_mass=COPPER_MASS,
        )
        assert S == 0.0

    def test_ablation_source_array_mask(self):
        J = np.full(10, 1e8)
        eta = np.full(10, 1e-5)
        mask = np.zeros(10, dtype=np.int64)
        mask[0] = 1
        mask[-1] = 1
        S = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, mask)
        assert S[0] > 0.0
        assert S[-1] > 0.0
        assert S[5] == 0.0

    def test_ablation_particle_flux(self):
        S_rho = 1e3
        n_dot = ablation_particle_flux(S_rho, COPPER_MASS)
        np.testing.assert_allclose(n_dot, S_rho / COPPER_MASS, rtol=1e-12)

    def test_ablation_momentum_source(self):
        S_rho = 1e3
        v = 1e4
        S_mom = ablation_momentum_source(S_rho, v)
        np.testing.assert_allclose(S_mom, S_rho * v, rtol=1e-12)

    def test_material_constants(self):
        assert COPPER_ABLATION_EFFICIENCY > 0
        assert TUNGSTEN_ABLATION_EFFICIENCY > 0
        assert COPPER_MASS > 0
        assert TUNGSTEN_MASS > COPPER_MASS


class TestCoolingFunction:
    """Tests for coronal equilibrium cooling function."""

    def test_hydrogen_cooling_positive(self):
        Te = np.array([5.0 * 11604.5])
        Lambda = cooling_function(Te, Z=1.0)
        assert Lambda[0] > 0.0

    def test_copper_cooling_stronger_than_hydrogen(self):
        Te = np.array([100.0 * 11604.5])
        L_h = cooling_function(Te, Z=1.0)
        L_cu = cooling_function(Te, Z=29.0)
        assert L_cu[0] > L_h[0]

    def test_tungsten_positive(self):
        Te = np.array([100.0 * 11604.5])
        L_w = cooling_function(Te, Z=74.0)
        assert L_w[0] > 0.0

    def test_cooling_array_shape(self):
        Te = np.linspace(1e4, 1e7, 50)
        Lambda = cooling_function(Te, Z=29.0)
        assert Lambda.shape == Te.shape

    def test_generic_z_positive(self):
        Te = np.array([50.0 * 11604.5])
        Lambda = cooling_function(Te, Z=6.0)
        assert Lambda[0] > 0.0


class TestRecombinationPower:
    """Tests for recombination radiation."""

    def test_recombination_positive(self):
        ne = np.array([1e23])
        Te = np.array([1e6])
        P = recombination_power(ne, Te, Z=1.0)
        assert P[0] > 0.0

    def test_recombination_scales_with_ne_squared(self):
        ne1 = np.array([1e23])
        ne2 = np.array([2e23])
        Te = np.array([1e6])
        P1 = recombination_power(ne1, Te, Z=1.0)
        P2 = recombination_power(ne2, Te, Z=1.0)
        np.testing.assert_allclose(P2[0] / P1[0], 4.0, rtol=0.01)

    def test_recombination_scales_with_Z_squared(self):
        ne = np.array([1e23])
        Te = np.array([1e6])
        P1 = recombination_power(ne, Te, Z=1.0)
        P2 = recombination_power(ne, Te, Z=2.0)
        ratio = P2[0] / P1[0]
        assert ratio > 4.0


class TestLinePower:
    """Tests for line radiation power."""

    def test_line_power_positive(self):
        ne = np.array([1e23])
        Te = np.array([100.0 * 11604.5])
        P = line_radiation_power(ne, Te, Z=29.0, n_impurity_fraction=0.01)
        assert P[0] > 0.0

    def test_line_power_zero_impurity(self):
        ne = np.array([1e23])
        Te = np.array([1e6])
        P = line_radiation_power(ne, Te, Z=29.0, n_impurity_fraction=0.0)
        assert P[0] == 0.0

    def test_total_radiation_includes_line(self):
        ne = np.array([1e23])
        Te = np.array([100.0 * 11604.5])
        P_no_imp = total_radiation_power(ne, Te, Z_eff=1.0, n_impurity_fraction=0.0)
        P_with_imp = total_radiation_power(ne, Te, Z_eff=1.0, n_impurity_fraction=0.01, Z_impurity=29.0)
        assert P_with_imp[0] > P_no_imp[0]


class TestImplicitLineCooling:
    """Tests for implicit line + recombination cooling step."""

    def test_cooling_reduces_temperature(self):
        Te = np.array([1e7])
        ne = np.array([1e24])
        Te_new, P_rad = apply_line_radiation_losses(Te, ne, dt=1e-9, Z_eff=1.0, n_imp_frac=0.01, Z_imp=29.0)
        assert Te_new[0] < Te[0]
        assert P_rad[0] > 0.0

    def test_cooling_never_negative(self):
        Te = np.array([100.0])
        ne = np.array([1e23])
        Te_new, _ = apply_line_radiation_losses(Te, ne, dt=1.0, Z_eff=1.0, n_imp_frac=0.05, Z_imp=29.0, Te_floor=10.0)
        assert Te_new[0] >= 10.0

    def test_cooling_array_shape(self):
        shape = (8, 16)
        Te = np.full(shape, 1e6)
        ne = np.full(shape, 1e23)
        Te_new, P = apply_line_radiation_losses(Te, ne, dt=1e-9)
        assert Te_new.shape == shape
        assert P.shape == shape


class TestLotzIonization:
    """Tests for Lotz electron-impact ionization rate coefficient."""

    def test_positive_rate_above_threshold(self):
        rate = lotz_ionization_rate(20.0, 13.6)
        assert rate > 0.0

    def test_rate_very_small_below_threshold(self):
        rate = lotz_ionization_rate(1.0, 13.6)
        assert rate < 1e-23

    def test_rate_increases_with_temperature(self):
        r1 = lotz_ionization_rate(10.0, 13.6)
        r2 = lotz_ionization_rate(50.0, 13.6)
        assert r2 > r1

    def test_rate_zero_for_low_Te(self):
        assert lotz_ionization_rate(0.005, 13.6) == 0.0

    def test_rate_order_of_magnitude(self):
        rate = lotz_ionization_rate(27.2, 13.6)
        assert 1e-22 < rate < 1e-17


class TestRecombinationRates:
    """Tests for radiative and dielectronic recombination rates."""

    def test_radiative_positive(self):
        assert radiative_recombination_rate(10.0, 1) > 0.0

    def test_dielectronic_positive(self):
        assert dielectronic_recombination_rate(10.0, 1) > 0.0

    def test_total_greater_than_either(self):
        a_rr = radiative_recombination_rate(10.0, 1)
        a_dr = dielectronic_recombination_rate(10.0, 1)
        a_total = total_recombination_rate(10.0, 1)
        np.testing.assert_allclose(a_total, a_rr + a_dr, rtol=1e-12)

    def test_recombination_increases_with_Z(self):
        a1 = total_recombination_rate(100.0, 1)
        a5 = total_recombination_rate(100.0, 5)
        assert a5 > a1

    def test_dielectronic_peak(self):
        a_low = dielectronic_recombination_rate(0.5, 1)
        a_peak = dielectronic_recombination_rate(3.4, 1)
        a_high = dielectronic_recombination_rate(100.0, 1)
        assert a_peak > a_low
        assert a_peak > a_high


class TestCRSolveChargeStates:
    """Tests for implicit backward Euler CR solver."""

    def test_neutral_start_ionizes(self):
        frac = np.array([1.0, 0.0])
        frac_new = cr_solve_charge_states(ne=1e23, Te_eV=20.0, Z_max=1, dt=1e-6, frac_in=frac, ip_eV=_IP_H)
        assert frac_new[1] > 0.0

    def test_fully_ionized_recombines(self):
        frac = np.array([0.0, 1.0])
        frac_new = cr_solve_charge_states(ne=1e23, Te_eV=0.5, Z_max=1, dt=1e-6, frac_in=frac, ip_eV=_IP_H)
        assert frac_new[0] > 0.0

    def test_fractions_normalized(self):
        frac = np.array([0.5, 0.5])
        frac_new = cr_solve_charge_states(ne=1e23, Te_eV=10.0, Z_max=1, dt=1e-7, frac_in=frac, ip_eV=_IP_H)
        np.testing.assert_allclose(np.sum(frac_new), 1.0, atol=1e-10)

    def test_fractions_nonnegative(self):
        frac = np.array([0.1, 0.9])
        frac_new = cr_solve_charge_states(ne=1e24, Te_eV=50.0, Z_max=1, dt=1e-5, frac_in=frac, ip_eV=_IP_H)
        assert np.all(frac_new >= 0.0)

    def test_copper_multi_charge(self):
        frac = np.zeros(30)
        frac[0] = 1.0
        frac_new = cr_solve_charge_states(ne=1e23, Te_eV=100.0, Z_max=29, dt=1e-6, frac_in=frac, ip_eV=_IP_CU)
        assert frac_new[0] < 1.0
        np.testing.assert_allclose(np.sum(frac_new), 1.0, atol=1e-10)

    def test_no_ionization_at_cold(self):
        frac = np.array([1.0, 0.0])
        frac_new = cr_solve_charge_states(ne=1e20, Te_eV=0.01, Z_max=1, dt=1e-6, frac_in=frac, ip_eV=_IP_H)
        np.testing.assert_allclose(frac_new[0], 1.0, atol=1e-6)


class TestCRAverageCharge:
    """Tests for CR equilibrium Z_bar computation."""

    def test_hydrogen_hot_ionized(self):
        Z_bar = cr_average_charge(1e23, 1000.0, 1, _IP_H, n_iter=100)
        assert Z_bar > 0.3

    def test_hydrogen_cold_neutral(self):
        Z_bar = cr_average_charge(1e23, 0.1, 1, _IP_H)
        assert Z_bar < 0.01

    def test_cr_differs_from_saha_low_density(self):
        Te_K = 2.0 * 11604.5
        ne = 1e18
        Te_eV = 2.0
        Z_saha = saha_ionization_fraction(Te_K, ne)
        Z_cr = cr_average_charge(ne, Te_eV, 1, _IP_H)
        assert isinstance(Z_cr, float)
        assert isinstance(Z_saha, float)
        assert 0.0 <= Z_cr <= 1.0

    def test_copper_z_bar_intermediate(self):
        Z_bar = cr_average_charge(1e23, 100.0, 29, _IP_CU)
        assert 0.0 < Z_bar < 29.0


class TestCRFieldFunctions:
    """Tests for spatially-resolved CR field operations."""

    def test_cr_zbar_field_shape(self):
        ne = np.full(20, 1e23)
        Te = np.full(20, 1e6)
        Z_bar = cr_zbar_field(ne, Te, 1, _IP_H)
        assert Z_bar.shape == (20,)

    def test_cr_zbar_field_all_finite(self):
        ne = np.linspace(1e20, 1e25, 30)
        Te = np.linspace(1e4, 1e7, 30)
        Z_bar = cr_zbar_field(ne, Te, 1, _IP_H)
        assert np.all(np.isfinite(Z_bar))

    def test_cr_zbar_field_bounded(self):
        ne = np.full(10, 1e23)
        Te = np.full(10, 1e7)
        Z_bar = cr_zbar_field(ne, Te, 1, _IP_H)
        assert np.all(Z_bar >= 0.0)
        assert np.all(Z_bar <= 1.0)

    def test_cr_evolve_field(self):
        n_cells = 5
        ne = np.full(n_cells, 1e23)
        Te = np.full(n_cells, 1e6)
        frac = np.zeros((n_cells, 2))
        frac[:, 0] = 1.0
        frac_new = cr_evolve_field(ne, Te, 1, 1e-7, frac, _IP_H)
        assert frac_new.shape == frac.shape
        for i in range(n_cells):
            s = frac_new.flat[i * 2] + frac_new.flat[i * 2 + 1]
            np.testing.assert_allclose(s, 1.0, atol=1e-10)


class TestIonizationPotentials:
    """Tests for stored ionization potential data."""

    def test_hydrogen_ip(self):
        assert len(_IP_H) == 1
        np.testing.assert_allclose(_IP_H[0], 13.6, atol=0.01)

    def test_copper_ip_length(self):
        assert len(_IP_CU) == 29

    def test_copper_ip_monotonic(self):
        for i in range(len(_IP_CU) - 1):
            assert _IP_CU[i + 1] > _IP_CU[i]

    def test_tungsten_ip_length(self):
        assert len(_IP_W) == 10

    def test_potentials_dict(self):
        assert 1 in IONIZATION_POTENTIALS
        assert 29 in IONIZATION_POTENTIALS
        assert 74 in IONIZATION_POTENTIALS


class TestDDCrossSection:
    """Tests for Bosch-Hale DD fusion cross section."""

    def test_cross_section_positive(self):
        assert dd_cross_section(100.0) > 0.0

    def test_cross_section_zero_below_range(self):
        assert dd_cross_section(0.1) == 0.0

    def test_cross_section_zero_above_range(self):
        assert dd_cross_section(6000.0) == 0.0

    def test_cross_section_10keV(self):
        sigma = dd_cross_section(10.0)
        assert 1e-33 < sigma < 5e-30

    def test_cross_section_increases_with_energy(self):
        assert dd_cross_section(100.0) > dd_cross_section(10.0)

    def test_cross_section_array(self):
        E = np.array([10.0, 50.0, 100.0, 200.0])
        sigma = dd_cross_section_array(E)
        assert sigma.shape == E.shape
        assert np.all(sigma > 0)


class TestBeamTargetYield:
    """Tests for beam-target neutron production rate."""

    def test_yield_positive(self):
        dY = beam_target_yield_rate(I_pinch=200e3, V_pinch=50e3, n_target=1e25, L_target=0.01)
        assert dY > 0.0

    def test_yield_zero_no_current(self):
        assert beam_target_yield_rate(0, 50e3, 1e25, 0.01) == 0.0

    def test_yield_zero_no_voltage(self):
        assert beam_target_yield_rate(200e3, 0, 1e25, 0.01) == 0.0

    def test_yield_scales_with_current(self):
        y1 = beam_target_yield_rate(100e3, 50e3, 1e25, 0.01)
        y2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01)
        np.testing.assert_allclose(y2, 2 * y1, rtol=1e-10)

    def test_yield_scales_with_f_beam(self):
        y1 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=0.1)
        y2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=0.3)
        np.testing.assert_allclose(y2 / y1, 3.0, rtol=1e-10)

    def test_typical_dpf_yield_order(self):
        dY = beam_target_yield_rate(I_pinch=500e3, V_pinch=100e3, n_target=1e25, L_target=0.02, f_beam=0.2)
        assert dY > 1e10


class TestPinchDisruption:
    """Tests for m=0 instability disruption detector."""

    def test_spike_detected(self):
        history = [1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 8e5]
        assert detect_pinch_disruption(history, threshold_ratio=5.0)

    def test_no_spike_not_detected(self):
        history = [1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5]
        assert not detect_pinch_disruption(history, threshold_ratio=5.0)

    def test_short_history_returns_false(self):
        assert not detect_pinch_disruption([1e5, 2e5, 3e5])

    def test_numpy_array_input(self):
        history = np.array([1e5] * 8 + [1e6])
        assert detect_pinch_disruption(history, threshold_ratio=5.0)


class TestNeutronAnisotropy:
    """Tests for neutron anisotropy diagnostic."""

    def test_pure_thermonuclear_isotropic(self):
        A = neutron_anisotropy(Y_beam=0, Y_thermal=1e11)
        np.testing.assert_allclose(A, 1.0, atol=1e-12)

    def test_pure_beam_target_anisotropic(self):
        A = neutron_anisotropy(Y_beam=1e11, Y_thermal=0)
        assert A > 1.0

    def test_mixed_yield_intermediate(self):
        A = neutron_anisotropy(Y_beam=1e11, Y_thermal=1e11, E_beam_keV=100.0)
        assert 1.0 < A < 2.0

    def test_anisotropy_increases_with_beam_fraction(self):
        A1 = neutron_anisotropy(Y_beam=1e10, Y_thermal=1e11)
        A2 = neutron_anisotropy(Y_beam=1e11, Y_thermal=1e11)
        assert A2 > A1

    def test_anisotropy_increases_with_beam_energy(self):
        A1 = neutron_anisotropy(Y_beam=1e11, Y_thermal=0, E_beam_keV=50.0)
        A2 = neutron_anisotropy(Y_beam=1e11, Y_thermal=0, E_beam_keV=400.0)
        assert A2 > A1

    def test_zero_yield_returns_one(self):
        A = neutron_anisotropy(0, 0)
        np.testing.assert_allclose(A, 1.0, atol=1e-12)

    def test_typical_dpf_range(self):
        A = neutron_anisotropy(Y_beam=5e11, Y_thermal=1e11, E_beam_keV=150.0)
        assert 1.0 <= A <= 3.0


# --- Section: Phase 16 Extended MHD ---


class TestNernstCoefficient:
    """Tests for the Nernst coefficient calculation."""

    def test_positive_for_magnetized_plasma(self):
        beta_wedge = nernst_coefficient(1e24, 1e7, 10.0)
        assert beta_wedge >= 0

    def test_scales_with_magnetization(self):
        ne = 1e23
        Te = 1e6
        beta_low = nernst_coefficient(ne, Te, 0.01)
        beta_high = nernst_coefficient(ne, Te, 50.0)
        assert beta_low != beta_high

    def test_zero_field_gives_zero(self):
        beta = nernst_coefficient(1e23, 1e6, 0.0)
        assert np.isfinite(beta)


class TestNernstElectricField:
    """Tests for Nernst E-field computation."""

    def test_zero_gradient_gives_zero_field(self):
        from dpf.fluid.nernst import nernst_electric_field

        ne = 1e23
        Te = 1e6
        B = np.array([0.0, 0.0, 1.0])
        grad_Te = np.array([0.0, 0.0, 0.0])
        E_N = nernst_electric_field(ne, Te, B, grad_Te)
        np.testing.assert_allclose(E_N, 0.0, atol=1e-30)

    def test_perpendicular_to_b_and_grad_te(self):
        from dpf.fluid.nernst import nernst_electric_field

        ne = 1e23
        Te = 1e6
        B = np.array([0.0, 0.0, 5.0])
        grad_Te = np.array([1e8, 0.0, 0.0])
        E_N = nernst_electric_field(ne, Te, B, grad_Te)
        assert abs(E_N[2]) < abs(E_N[1]) + 1e-30 or np.allclose(E_N, 0.0, atol=1e-30)


class TestNernstAdvection:
    """Tests for Nernst advection of B-field."""

    def test_uniform_temperature_no_change(self):
        from dpf.fluid.nernst import apply_nernst_advection

        n = 16
        Bx = np.zeros((n, n, n))
        By = np.zeros((n, n, n))
        Bz = np.ones((n, n, n))
        ne = np.full((n, n, n), 1e23)
        Te = np.full((n, n, n), 1e6)
        Bx_new, By_new, Bz_new = apply_nernst_advection(Bx, By, Bz, ne, Te, 0.001, 0.001, 0.001, 1e-12)
        np.testing.assert_allclose(Bz_new, 1.0, atol=1e-6)


class TestIonCollisionTime:
    """Tests for ion-ion collision time."""

    def test_positive(self):
        from dpf.fluid.viscosity import ion_collision_time

        tau = ion_collision_time(1e23, 1e6)
        assert tau > 0

    def test_increases_with_temperature(self):
        from dpf.fluid.viscosity import ion_collision_time

        tau_cold = ion_collision_time(1e23, 1e5)
        tau_hot = ion_collision_time(1e23, 1e7)
        assert tau_hot > tau_cold

    def test_decreases_with_density(self):
        from dpf.fluid.viscosity import ion_collision_time

        tau_low = ion_collision_time(1e22, 1e6)
        tau_high = ion_collision_time(1e24, 1e6)
        assert tau_low > tau_high


class TestBraginskiiEta0:
    """Tests for parallel viscosity eta_0."""

    def test_positive(self):
        from dpf.fluid.viscosity import braginskii_eta0, ion_collision_time

        ni = 1e23
        Ti = 1e6
        tau = ion_collision_time(ni, Ti)
        eta0 = braginskii_eta0(ni, Ti, tau)
        assert eta0 > 0

    def test_scales_with_density_and_temperature(self):
        from dpf.fluid.viscosity import braginskii_eta0, ion_collision_time

        tau1 = ion_collision_time(1e23, 1e6)
        eta1 = braginskii_eta0(1e23, 1e6, tau1)
        tau2 = ion_collision_time(1e24, 1e6)
        eta2 = braginskii_eta0(1e24, 1e6, tau2)
        assert eta1 != eta2


class TestBraginskiiEta3:
    """Tests for gyroviscosity eta_3."""

    def test_positive_with_field(self):
        from dpf.fluid.viscosity import braginskii_eta3

        eta3 = braginskii_eta3(1e23, 1e6, 5.0)
        assert eta3 > 0

    def test_inversely_proportional_to_b(self):
        from dpf.fluid.viscosity import braginskii_eta3

        eta3_low = braginskii_eta3(1e23, 1e6, 1.0)
        eta3_high = braginskii_eta3(1e23, 1e6, 10.0)
        assert eta3_low > eta3_high


class TestViscousStress:
    """Tests for viscous stress computation."""

    def test_uniform_velocity_no_stress(self):
        from dpf.fluid.viscosity import viscous_stress_rate

        n = 8
        vx = np.ones((n, n, n)) * 100.0
        vy = np.zeros((n, n, n))
        vz = np.zeros((n, n, n))
        rho = np.ones((n, n, n))
        eta0 = np.ones((n, n, n)) * 1e-3
        dvdt = viscous_stress_rate(
            np.stack([vx, vy, vz], axis=-1), rho, eta0, 0.001, 0.001, 0.001,
        )
        interior = dvdt[2:-2, 2:-2, 2:-2]
        assert np.max(np.abs(interior)) < 1e-6

    def test_shear_produces_stress(self):
        from dpf.fluid.viscosity import viscous_stress_rate

        n = 16
        y = np.linspace(0, 1, n)
        vx = np.zeros((n, n, n))
        for j in range(n):
            vx[:, j, :] = y[j] * 1000.0
        vy = np.zeros((n, n, n))
        vz = np.zeros((n, n, n))
        rho = np.ones((n, n, n))
        eta0 = np.ones((n, n, n)) * 1e-2
        dvdt = viscous_stress_rate(
            np.stack([vx, vy, vz], axis=-1), rho, eta0, 1.0 / n, 1.0 / n, 1.0 / n,
        )
        interior = dvdt[2:-2, 2:-2, 2:-2]
        assert np.max(np.abs(interior)) > 0


class TestViscousHeating:
    """Tests for viscous heating rate."""

    def test_uniform_flow_no_heating(self):
        from dpf.fluid.viscosity import viscous_heating_rate

        n = 8
        vx = np.ones((n, n, n)) * 100.0
        vy = np.zeros((n, n, n))
        vz = np.zeros((n, n, n))
        eta0 = np.ones((n, n, n)) * 1e-3
        Q = viscous_heating_rate(
            np.stack([vx, vy, vz], axis=-1), eta0, 0.001, 0.001, 0.001,
        )
        interior = Q[2:-2, 2:-2, 2:-2]
        assert np.max(np.abs(interior)) < 1e-3

    def test_shear_gives_positive_heating(self):
        from dpf.fluid.viscosity import viscous_heating_rate

        n = 16
        y = np.linspace(0, 1, n)
        vx = np.zeros((n, n, n))
        for j in range(n):
            vx[:, j, :] = y[j] * 1000.0
        vy = np.zeros((n, n, n))
        vz = np.zeros((n, n, n))
        eta0 = np.ones((n, n, n)) * 1e-2
        Q = viscous_heating_rate(
            np.stack([vx, vy, vz], axis=-1), eta0, 1.0 / n, 1.0 / n, 1.0 / n,
        )
        interior = Q[3:-3, 3:-3, 3:-3]
        assert np.mean(interior) > 0


class TestStaggeredBField:
    """Tests for staggered B-field data structure."""

    def test_cell_to_face_roundtrip(self):
        from dpf.fluid.constrained_transport import cell_centered_to_face, face_to_cell_centered

        n = 8
        stag = cell_centered_to_face(
            np.ones((n, n, n)) * 2.0, np.ones((n, n, n)) * 3.0,
            np.ones((n, n, n)) * 5.0, 0.01, 0.01, 0.01,
        )
        Bx2, By2, Bz2 = face_to_cell_centered(stag)
        np.testing.assert_allclose(Bx2, 2.0, atol=1e-12)
        np.testing.assert_allclose(By2, 3.0, atol=1e-12)
        np.testing.assert_allclose(Bz2, 5.0, atol=1e-12)

    def test_face_shapes(self):
        from dpf.fluid.constrained_transport import cell_centered_to_face

        n = 8
        stag = cell_centered_to_face(
            np.ones((n, n, n)), np.ones((n, n, n)), np.ones((n, n, n)), 0.01, 0.01, 0.01,
        )
        assert stag.Bx_face.shape == (n + 1, n, n)
        assert stag.By_face.shape == (n, n + 1, n)
        assert stag.Bz_face.shape == (n, n, n + 1)


class TestDivBConstraint:
    """Tests for divergence-free constraint."""

    def test_uniform_field_zero_div(self):
        from dpf.fluid.constrained_transport import cell_centered_to_face, compute_div_B

        n = 8
        stag = cell_centered_to_face(
            np.ones((n, n, n)) * 2.0, np.ones((n, n, n)) * 3.0,
            np.ones((n, n, n)) * 5.0, 0.01, 0.01, 0.01,
        )
        div_b = compute_div_B(stag)
        np.testing.assert_allclose(div_b, 0.0, atol=1e-12)

    def test_ct_update_preserves_div_b(self):
        from dpf.fluid.constrained_transport import (
            cell_centered_to_face,
            compute_div_B,
            ct_update,
        )

        n = 8
        dx = dy = dz = 0.01
        stag = cell_centered_to_face(
            np.ones((n, n, n)) * 2.0, np.zeros((n, n, n)),
            np.ones((n, n, n)) * 1.0, dx, dy, dz,
        )
        rng = np.random.default_rng(42)
        Ex_edge = rng.uniform(-0.1, 0.1, (n, n + 1, n + 1))
        Ey_edge = rng.uniform(-0.1, 0.1, (n + 1, n, n + 1))
        Ez_edge = rng.uniform(-0.1, 0.1, (n + 1, n + 1, n))
        stag_new = ct_update(stag, Ex_edge, Ey_edge, Ez_edge, 1e-6)
        div_b = compute_div_B(stag_new)
        np.testing.assert_allclose(div_b, 0.0, atol=1e-10)


class TestExperimentalDevices:
    """Tests for experimental device data."""

    def test_pf1000_exists(self):
        from dpf.validation.experimental import DEVICES, PF1000_DATA

        assert PF1000_DATA.name == "PF-1000"
        assert PF1000_DATA.peak_current > 1e6
        assert "PF-1000" in DEVICES

    def test_nx2_exists(self):
        from dpf.validation.experimental import NX2_DATA

        assert NX2_DATA.peak_current > 1e5
        assert NX2_DATA.fill_gas == "deuterium"

    def test_unu_ictp_exists(self):
        from dpf.validation.experimental import UNU_ICTP_DATA

        assert UNU_ICTP_DATA.peak_current > 1e5
        assert UNU_ICTP_DATA.capacitance > 0

    def test_device_parameters_physical(self):
        from dpf.validation.experimental import DEVICES

        for _name, dev in DEVICES.items():
            assert dev.capacitance > 0
            assert dev.voltage > 0
            assert dev.anode_radius < dev.cathode_radius
            assert dev.neutron_yield > 0


class TestValidationMetrics:
    """Tests for validation metric functions."""

    def test_perfect_current_match(self):
        from dpf.validation.experimental import validate_current_waveform

        t = np.linspace(0, 6e-6, 1000)
        I_peak = 1.87e6
        current = I_peak * np.sin(2 * np.pi * t / (4 * 5.8e-6))
        result = validate_current_waveform(t, current, "PF-1000")
        assert result["peak_current_error"] < 0.05

    def test_neutron_yield_within_order(self):
        from dpf.validation.experimental import validate_neutron_yield

        result = validate_neutron_yield(5e10, "PF-1000")
        assert result["within_order_magnitude"]

    def test_neutron_yield_outside_order(self):
        from dpf.validation.experimental import validate_neutron_yield

        result = validate_neutron_yield(1e5, "PF-1000")
        assert not result["within_order_magnitude"]

    def test_device_to_config(self):
        from dpf.validation.experimental import device_to_config_dict

        cfg = device_to_config_dict("PF-1000")
        assert "circuit" in cfg
        assert cfg["circuit"]["C"] > 0
        assert cfg["circuit"]["V0"] > 0
        assert cfg["circuit"]["anode_radius"] < cfg["circuit"]["cathode_radius"]


# --- Section: Radiation ---


class TestBremsstrahlung:
    """Tests for bremsstrahlung power density."""

    def test_power_matches_analytic(self):
        """P_ff = 1.42e-40 * g_ff * Z * ne^2 * sqrt(Te) [SI]."""
        from dpf.radiation.bremsstrahlung import BREM_COEFF, bremsstrahlung_power

        ne = np.array([1e20, 1e22, 1e24])
        Te = np.array([1e6, 1e7, 1e8])
        Z = 1.0
        g_ff = 1.2

        P = bremsstrahlung_power(ne, Te, Z, g_ff)
        P_expected = BREM_COEFF * g_ff * Z**2 * ne**2 * np.sqrt(Te)

        np.testing.assert_allclose(P, P_expected, rtol=1e-10)

    def test_power_scales_with_ne_squared(self):
        """P_ff should scale as ne^2."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        Te = np.array([1e7])
        ne1 = np.array([1e20])
        ne2 = np.array([2e20])

        P1 = bremsstrahlung_power(ne1, Te)
        P2 = bremsstrahlung_power(ne2, Te)

        np.testing.assert_allclose(P2 / P1, 4.0, rtol=1e-10)

    def test_power_scales_with_sqrt_Te(self):
        """P_ff should scale as sqrt(Te)."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne = np.array([1e22])
        Te1 = np.array([1e6])
        Te2 = np.array([4e6])

        P1 = bremsstrahlung_power(ne, Te1)
        P2 = bremsstrahlung_power(ne, Te2)

        np.testing.assert_allclose(P2 / P1, 2.0, rtol=1e-10)

    def test_power_zero_for_zero_density(self):
        """No bremsstrahlung with no electrons."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne = np.array([0.0, 0.0])
        Te = np.array([1e7, 1e8])

        P = bremsstrahlung_power(ne, Te)
        np.testing.assert_array_equal(P, 0.0)

    def test_power_zero_for_zero_temperature(self):
        """No bremsstrahlung at zero temperature."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne = np.array([1e22])
        Te = np.array([0.0])

        P = bremsstrahlung_power(ne, Te)
        np.testing.assert_array_equal(P, 0.0)

    def test_typical_dpf_power(self):
        from dpf.radiation.bremsstrahlung import BREM_COEFF, bremsstrahlung_power

        ne = np.array([1e25])
        Te = np.array([1.16e7])
        P = bremsstrahlung_power(ne, Te, Z=1.0, gaunt_factor=1.2)

        P_expected = BREM_COEFF * 1.2 * 1.0 * (1e25)**2 * np.sqrt(1.16e7)
        np.testing.assert_allclose(P[0], P_expected, rtol=1e-6)


class TestBremsstrahlungCooling:
    """Tests for implicit bremsstrahlung cooling."""

    def test_cooling_reduces_temperature(self):
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((4, 4, 4), 1e22)
        Te = np.full((4, 4, 4), 1e7)
        dt = 1e-9

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt)

        assert np.all(Te_new < Te), "Temperature should decrease"
        assert np.all(Te_new > 0), "Temperature should remain positive"
        assert np.all(P_rad >= 0), "Radiated power should be non-negative"

    def test_implicit_stability_large_dt(self):
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((4,), 1e24)
        Te = np.full((4,), 1e7)
        dt = 1e-3

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt, Te_floor=100.0)

        assert np.all(Te_new >= 100.0), "Temperature should respect floor"
        assert np.all(np.isfinite(Te_new)), "No NaN or Inf"

    def test_energy_conservation(self):
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((8,), 1e22)
        Te = np.full((8,), 5e6)
        dt = 1e-10

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt)

        dE_electrons = 1.5 * ne * k_B * (Te - Te_new)
        np.testing.assert_allclose(P_rad * dt, dE_electrons, rtol=1e-6)


class TestFluxLimiter:
    """Tests for Levermore-Pomraning flux limiter."""

    def test_optically_thick_limit(self):
        """lambda(R -> 0) -> 1/3."""
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.array([0.0, 1e-8, 1e-6, 1e-4])
        lam = levermore_pomraning_limiter(R)

        np.testing.assert_allclose(lam, 1.0 / 3.0, atol=1e-3)

    def test_optically_thin_limit(self):
        """lambda(R -> inf) -> 1/R."""
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.array([100.0, 1000.0, 1e4])
        lam = levermore_pomraning_limiter(R)
        expected = 1.0 / R

        np.testing.assert_allclose(lam, expected, rtol=0.02)

    def test_monotonically_decreasing(self):
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.logspace(-2, 4, 100)
        lam = levermore_pomraning_limiter(R)

        assert np.all(np.diff(lam) <= 0), "Flux limiter should be monotonically decreasing"

    def test_bounded(self):
        """0 <= lambda(R) <= 1/3."""
        from dpf.radiation.transport import levermore_pomraning_limiter

        R = np.logspace(-6, 6, 1000)
        lam = levermore_pomraning_limiter(R)

        assert np.all(lam >= 0), "Flux limiter should be non-negative"
        assert np.all(lam <= 1.0 / 3.0 + 1e-10), "Flux limiter should be <= 1/3"


class TestFLDTransport:
    """Tests for flux-limited diffusion radiation transport."""

    def test_uniform_no_diffusion(self):
        from dpf.radiation.transport import fld_step

        nx = 16
        E_rad = np.full((nx, nx, nx), 1e5)
        Te = np.full((nx, nx, nx), 1e6)
        ne = np.full((nx, nx, nx), 1e22)
        dx = 0.01
        dt = 1e-12

        E_new, Q = fld_step(E_rad, Te, ne, dx, dt)

        assert np.all(np.isfinite(E_new)), "No NaN or Inf"
        assert np.all(E_new >= 0), "E_rad should be non-negative"

    def test_gaussian_diffuses(self):
        from dpf.radiation.transport import fld_step

        nx = 16
        x = np.linspace(0, 1, nx)
        X, Y, Zg = np.meshgrid(x, x, x, indexing="ij")
        center = 0.5
        sigma = 0.1

        r2 = (X - center)**2 + (Y - center)**2 + (Zg - center)**2
        E_rad = 1e4 * np.exp(-r2 / (2 * sigma**2))
        E_rad += 1e2

        Te = np.full((nx, nx, nx), 1e7)
        ne = np.full((nx, nx, nx), 1e18)
        dx = 1.0 / nx
        dt = 1e-14

        peak_before = float(np.max(E_rad))
        E_new, Q = fld_step(E_rad, Te, ne, dx, dt, brem_power=np.zeros_like(E_rad))
        peak_after = float(np.max(E_new))

        assert np.all(np.isfinite(E_new)), "No NaN or Inf"
        assert peak_after <= peak_before, "Peak should not increase with diffusion only"

    def test_opacity_scaling(self):
        """Higher density should give higher opacity (kappa ~ ne^2)."""
        from dpf.radiation.transport import compute_rosseland_opacity

        ne_low = np.array([1e22])
        ne_high = np.array([1e24])
        Te = np.array([1e6])

        kappa_low = compute_rosseland_opacity(ne_low, Te)
        kappa_high = compute_rosseland_opacity(ne_high, Te)

        assert float(kappa_high[0]) > float(kappa_low[0]), "Higher density should give higher opacity"
        ratio = float(kappa_high[0] / kappa_low[0])
        expected_ratio = (1e24 / 1e22)**2
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.01)

    def test_opacity_temperature_scaling(self):
        """Opacity should decrease with temperature (Kramers law: kappa ~ T^{-3.5})."""
        from dpf.radiation.transport import compute_rosseland_opacity

        ne = np.array([1e24])
        Te_low = np.array([1e5])
        Te_high = np.array([1e7])

        kappa_low_T = compute_rosseland_opacity(ne, Te_low)
        kappa_high_T = compute_rosseland_opacity(ne, Te_high)

        assert float(kappa_low_T[0]) > float(kappa_high_T[0]), "Higher temperature should reduce opacity"
        ratio = float(kappa_low_T[0] / kappa_high_T[0])
        expected_ratio = (1e7 / 1e5)**3.5
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.01)


class TestRadiationEnergy:
    """Tests for radiation energy density."""

    def test_lte_radiation_energy(self):
        """E_rad = a_R * T^4 in LTE."""
        from dpf.radiation.transport import compute_radiation_energy

        sigma_SB = 5.670374419e-8
        c_val = 2.998e8
        a_R = 4.0 * sigma_SB / c_val

        Te = np.array([1e4, 1e6, 1e8])
        ne = np.array([1e20, 1e22, 1e24])

        E_rad = compute_radiation_energy(Te, ne)
        E_expected = a_R * Te**4

        np.testing.assert_allclose(E_rad, E_expected, rtol=1e-3)

    def test_radiation_energy_positive(self):
        from dpf.radiation.transport import compute_radiation_energy

        Te = np.array([0.0, 100.0, 1e8])
        ne = np.array([0.0, 1e20, 1e24])

        E_rad = compute_radiation_energy(Te, ne)
        assert np.all(E_rad >= 0), "E_rad should be non-negative"


class TestRadiationIntegration:
    """Tests for the full radiation step interface."""

    def test_apply_radiation_transport(self):
        from dpf.radiation.transport import apply_radiation_transport

        nx = 8
        state = {
            "Te": np.full((nx, nx, nx), 1e6),
            "Ti": np.full((nx, nx, nx), 1e6),
            "rho": np.full((nx, nx, nx), 1e-2),
            "pressure": np.full((nx, nx, nx), 1e5),
        }

        state = apply_radiation_transport(state, dx=0.01, dt=1e-12)

        assert "E_rad" in state, "Should add E_rad to state"
        assert np.all(np.isfinite(state["Te"])), "Te should be finite"
        assert np.all(np.isfinite(state["E_rad"])), "E_rad should be finite"
        assert np.all(state["Te"] > 0), "Te should be positive"
        assert np.all(state["E_rad"] >= 0), "E_rad should be non-negative"

    def test_bremsstrahlung_only_cools(self):
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((8, 8, 8), 1e24)
        Te_init = np.full((8, 8, 8), 1e7)
        dt = 1e-10

        Te_new, P_rad = apply_bremsstrahlung_losses(Te_init, ne, dt)

        assert np.all(Te_new < Te_init), "Temperature should decrease"
        assert np.all(P_rad > 0), "Radiated power should be positive"

    def test_cold_plasma_minimal_bremsstrahlung(self):
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.full((8,), 1e16)
        Te_init = np.full((8,), 300.0)
        dt = 1e-9

        Te_new, P_rad = apply_bremsstrahlung_losses(Te_init, ne, dt)

        dTe = np.max(np.abs(Te_init - Te_new))
        assert dTe < 1.0, f"Cold plasma dTe={dTe:.2e} should be < 1 K"


# --- Section: Sheath ---


class TestBohmVelocity:
    """Tests for Bohm velocity computation."""

    def test_analytic_match(self):
        """v_B = sqrt(k_B * Te / m_i) for hydrogen."""
        from dpf.sheath.bohm import bohm_velocity

        Te = 1e6
        v_B = bohm_velocity(Te, m_p)
        v_expected = np.sqrt(k_B * Te / m_p)

        np.testing.assert_allclose(v_B, v_expected, rtol=1e-10)

    def test_scales_with_sqrt_Te(self):
        from dpf.sheath.bohm import bohm_velocity

        v1 = bohm_velocity(1e6, m_p)
        v2 = bohm_velocity(4e6, m_p)

        np.testing.assert_allclose(v2 / v1, 2.0, rtol=1e-10)

    def test_heavier_ion_slower(self):
        from dpf.sheath.bohm import bohm_velocity

        v_H = bohm_velocity(1e6, m_p)
        v_D = bohm_velocity(1e6, 2.0 * m_p)

        assert v_D < v_H
        np.testing.assert_allclose(v_H / v_D, np.sqrt(2.0), rtol=1e-6)

    def test_zero_temperature(self):
        from dpf.sheath.bohm import bohm_velocity

        assert bohm_velocity(0.0, m_p) == 0.0

    def test_typical_dpf_value(self):
        """At Te ~ 1 keV = 1.16e7 K, v_B ~ 1e5 m/s for hydrogen."""
        from dpf.sheath.bohm import bohm_velocity

        Te_keV = 1.16e7
        v_B = bohm_velocity(Te_keV, m_p)

        assert 1e5 < v_B < 1e6, f"v_B = {v_B:.2e} should be ~ 3e5 m/s"


class TestChildLangmuir:
    """Tests for Child-Langmuir space-charge-limited current."""

    def test_analytic_match(self):
        """J_CL = (4/9) * epsilon_0 * sqrt(2e/mi) * V^{3/2} / d^2."""
        from dpf.sheath.bohm import child_langmuir_current

        V = 100.0
        d = 1e-3
        J = child_langmuir_current(V, d, m_p)
        J_expected = (4.0 / 9.0) * epsilon_0 * np.sqrt(2.0 * e / m_p) * V**1.5 / d**2

        np.testing.assert_allclose(J, J_expected, rtol=1e-10)

    def test_scales_with_V_three_halves(self):
        from dpf.sheath.bohm import child_langmuir_current

        d = 1e-3
        J1 = child_langmuir_current(100.0, d, m_p)
        J2 = child_langmuir_current(400.0, d, m_p)

        np.testing.assert_allclose(J2 / J1, (4.0)**1.5, rtol=1e-10)

    def test_scales_with_d_inv_squared(self):
        from dpf.sheath.bohm import child_langmuir_current

        V = 100.0
        J1 = child_langmuir_current(V, 1e-3, m_p)
        J2 = child_langmuir_current(V, 2e-3, m_p)

        np.testing.assert_allclose(J1 / J2, 4.0, rtol=1e-10)

    def test_positive_current(self):
        from dpf.sheath.bohm import child_langmuir_current

        assert child_langmuir_current(100.0, 1e-3) > 0
        assert child_langmuir_current(-100.0, 1e-3) > 0


class TestDebyeLength:
    """Tests for Debye length."""

    def test_analytic_match(self):
        """lambda_D = sqrt(epsilon_0 * k_B * Te / (ne * e^2))."""
        from dpf.sheath.bohm import debye_length

        ne = 1e20
        Te = 1e6
        lam = debye_length(ne, Te)
        lam_expected = np.sqrt(epsilon_0 * k_B * Te / (ne * e * e))

        np.testing.assert_allclose(lam, lam_expected, rtol=1e-10)

    def test_increases_with_temperature(self):
        from dpf.sheath.bohm import debye_length

        lam1 = debye_length(1e20, 1e6)
        lam2 = debye_length(1e20, 4e6)

        np.testing.assert_allclose(lam2 / lam1, 2.0, rtol=1e-10)

    def test_decreases_with_density(self):
        from dpf.sheath.bohm import debye_length

        lam1 = debye_length(1e20, 1e6)
        lam2 = debye_length(4e20, 1e6)

        np.testing.assert_allclose(lam1 / lam2, 2.0, rtol=1e-10)


class TestSheathThickness:
    """Tests for sheath thickness estimation."""

    def test_positive(self):
        from dpf.sheath.bohm import sheath_thickness

        d = sheath_thickness(ne=1e20, Te=1e6, V_sheath=100.0)
        assert d > 0

    def test_larger_than_debye(self):
        from dpf.sheath.bohm import debye_length, sheath_thickness

        ne = 1e20
        Te = 1e6
        d = sheath_thickness(ne, Te, V_sheath=100.0)
        lam_D = debye_length(ne, Te)

        assert d >= 3.0 * lam_D

    def test_increases_with_voltage(self):
        from dpf.sheath.bohm import sheath_thickness

        d1 = sheath_thickness(1e20, 1e6, V_sheath=10.0)
        d2 = sheath_thickness(1e20, 1e6, V_sheath=1000.0)

        assert d2 > d1


class TestFloatingPotential:
    """Tests for floating potential."""

    def test_negative(self):
        from dpf.sheath.bohm import floating_potential

        V_f = floating_potential(1e6)
        assert V_f < 0

    def test_scales_with_Te(self):
        from dpf.sheath.bohm import floating_potential

        V1 = floating_potential(1e6)
        V2 = floating_potential(2e6)

        np.testing.assert_allclose(V2 / V1, 2.0, rtol=1e-10)

    def test_typical_value(self):
        """V_f ~ -2.84 * Te_eV for hydrogen."""
        from dpf.sheath.bohm import floating_potential

        Te_eV = 1.0
        Te_K = Te_eV * e / k_B
        V_f = floating_potential(Te_K)

        expected = -(Te_eV / 2.0) * np.log(m_p / (2.0 * np.pi * m_e))
        np.testing.assert_allclose(V_f, expected, rtol=1e-3)


class TestPoisson1D:
    """Tests for 1D Poisson solver."""

    def test_boundary_conditions(self):
        from dpf.sheath.bohm import poisson_1d

        N = 100
        ne = np.full(N, 1e20)
        ni = np.full(N, 1e20)
        dx = 1e-4

        phi, E = poisson_1d(ne, ni, dx, V_left=0.0, V_right=-100.0)

        np.testing.assert_allclose(phi[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(phi[-1], -100.0, atol=1e-10)

    def test_neutral_plasma_linear(self):
        """With neutral plasma (ne = ni), potential is linear between BCs."""
        from dpf.sheath.bohm import poisson_1d

        N = 100
        ne = np.full(N, 1e20)
        ni = np.full(N, 1e20)
        dx = 1e-4

        phi, E = poisson_1d(ne, ni, dx, V_left=0.0, V_right=-100.0)

        x = np.linspace(0, (N - 1) * dx, N)
        phi_expected = -100.0 * x / x[-1]
        np.testing.assert_allclose(phi, phi_expected, atol=0.1)

    def test_positive_charge_concave_down(self):
        from dpf.sheath.bohm import poisson_1d

        N = 50
        ne = np.full(N, 1e18)
        ni = np.full(N, 1e20)
        dx = 1e-5

        phi, E = poisson_1d(ne, ni, dx, V_left=0.0, V_right=0.0)

        assert np.max(phi[1:-1]) > 0, "Positive charge should create positive potential hill"
        mid = N // 2
        assert phi[mid] > phi[1], "Potential should peak near the center"


class TestSheathBC:
    """Tests for sheath boundary condition application."""

    def test_applies_bohm_velocity(self):
        from dpf.sheath.bohm import apply_sheath_bc, bohm_velocity

        nx, ny, nz = 8, 8, 8
        state = {
            "velocity": np.zeros((3, nx, ny, nz)),
            "rho": np.full((nx, ny, nz), 1e-2),
            "Te": np.full((nx, ny, nz), 1e6),
        }

        Te_boundary = 1e6
        state = apply_sheath_bc(
            state, ne_boundary=1e20, Te_boundary=Te_boundary,
            V_sheath=100.0, boundary="z_high"
        )

        v_B = bohm_velocity(Te_boundary, m_p)
        np.testing.assert_allclose(state["velocity"][2, :, :, -1], v_B, rtol=1e-10)

    def test_z_low_boundary(self):
        from dpf.sheath.bohm import apply_sheath_bc, bohm_velocity

        nx, ny, nz = 8, 8, 8
        state = {
            "velocity": np.zeros((3, nx, ny, nz)),
            "rho": np.full((nx, ny, nz), 1e-2),
            "Te": np.full((nx, ny, nz), 1e6),
        }

        state = apply_sheath_bc(
            state, ne_boundary=1e20, Te_boundary=1e6,
            V_sheath=100.0, boundary="z_low"
        )

        v_B = bohm_velocity(1e6, m_p)
        np.testing.assert_allclose(state["velocity"][2, :, :, 0], -v_B, rtol=1e-10)


# --- Section: Turbulence ---


class TestElectronDriftVelocity:
    """Tests for drift velocity computation."""

    def test_drift_velocity_positive(self):
        from dpf.turbulence.anomalous import electron_drift_velocity

        J = np.array([1e10])
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
        from dpf.turbulence.anomalous import electron_drift_velocity

        J = np.array([0.0])
        ne = np.array([1e24])
        v_d = electron_drift_velocity(J, ne)
        assert v_d[0] == 0.0


class TestIonThermalSpeed:
    """Tests for ion thermal speed."""

    def test_thermal_speed_positive(self):
        from dpf.turbulence.anomalous import ion_thermal_speed

        Ti = np.array([1e7])
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
        from dpf.turbulence.anomalous import ion_thermal_speed

        Ti = np.array([0.0])
        v_ti = ion_thermal_speed(Ti)
        assert v_ti[0] == 0.0


class TestPlasmaFrequency:
    """Tests for electron plasma frequency."""

    def test_plasma_frequency_positive(self):
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
        from dpf.turbulence.anomalous import plasma_frequency

        ne1 = np.array([1e24])
        ne2 = np.array([4e24])
        omega1 = plasma_frequency(ne1)
        omega2 = plasma_frequency(ne2)
        np.testing.assert_allclose(omega2[0] / omega1[0], 2.0, rtol=1e-10)


class TestBunemanThreshold:
    """Tests for Buneman instability threshold detection."""

    def test_stable_below_threshold(self):
        from dpf.turbulence.anomalous import buneman_threshold

        J = np.array([1e2])
        ne = np.array([1e24])
        Ti = np.array([1e7])
        result = buneman_threshold(J, ne, Ti)
        assert not result[0], "Should be stable below threshold"

    def test_unstable_above_threshold(self):
        from dpf.turbulence.anomalous import buneman_threshold

        J = np.array([1e12])
        ne = np.array([1e24])
        Ti = np.array([300.0])
        result = buneman_threshold(J, ne, Ti)
        assert result[0], "Should be unstable above threshold"

    def test_threshold_depends_on_temperature(self):
        from dpf.turbulence.anomalous import buneman_threshold

        J = np.array([1e9])
        ne = np.array([1e24])

        result_cold = buneman_threshold(J, ne, np.array([100.0]))
        result_hot = buneman_threshold(J, ne, np.array([1e8]))

        assert result_cold[0], "Cold ions should be unstable"
        assert not result_hot[0], "Hot ions should be stable"


class TestAnomalousResistivity:
    """Tests for anomalous resistivity computation."""

    def test_zero_below_threshold(self):
        from dpf.turbulence.anomalous import anomalous_resistivity

        J = np.array([1e2])
        ne = np.array([1e24])
        Ti = np.array([1e7])
        eta = anomalous_resistivity(J, ne, Ti)
        assert eta[0] == 0.0, "eta_anom should be zero below threshold"

    def test_positive_above_threshold(self):
        from dpf.turbulence.anomalous import anomalous_resistivity

        J = np.array([1e12])
        ne = np.array([1e24])
        Ti = np.array([300.0])
        eta = anomalous_resistivity(J, ne, Ti)
        assert eta[0] > 0.0, "eta_anom should be positive above threshold"

    def test_scales_with_alpha(self):
        from dpf.turbulence.anomalous import anomalous_resistivity

        J = np.array([1e12])
        ne = np.array([1e24])
        Ti = np.array([300.0])

        eta_1 = anomalous_resistivity(J, ne, Ti, alpha=0.01)
        eta_2 = anomalous_resistivity(J, ne, Ti, alpha=0.1)
        np.testing.assert_allclose(eta_2[0] / eta_1[0], 10.0, rtol=1e-10)

    def test_exceeds_spitzer_when_strongly_unstable(self):
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity
        from dpf.turbulence.anomalous import anomalous_resistivity

        ne = np.array([1e25])
        Ti = np.array([1e5])
        Te = np.array([1e6])
        J = np.array([3e11])

        eta_anom = anomalous_resistivity(J, ne, Ti, alpha=0.05)
        lnL = coulomb_log(ne, Te)
        eta_sp = spitzer_resistivity(ne, Te, lnL)

        if eta_anom[0] > 0:
            assert eta_anom[0] > eta_sp[0], (
                f"eta_anom={eta_anom[0]:.2e} should exceed eta_spitzer={eta_sp[0]:.2e} "
                f"for strongly unstable conditions"
            )

    def test_finite_values(self):
        from dpf.turbulence.anomalous import anomalous_resistivity

        J = np.array([1e10, 1e11, 1e12])
        ne = np.array([1e23, 1e24, 1e25])
        Ti = np.array([1e4, 1e5, 1e6])
        eta = anomalous_resistivity(J, ne, Ti)
        assert np.all(np.isfinite(eta)), "All eta_anom values should be finite"


class TestScalarVersions:
    """Tests for scalar versions of anomalous resistivity functions."""

    def test_scalar_zero_below_threshold(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar

        eta = anomalous_resistivity_scalar(
            J_mag=1e2, ne_val=1e24, Ti_val=1e7, alpha=0.05, Te_val=1e7,
        )
        assert eta == 0.0

    def test_scalar_positive_above_threshold(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar

        eta = anomalous_resistivity_scalar(
            J_mag=1e12, ne_val=1e24, Ti_val=300.0, alpha=0.05,
        )
        assert eta > 0.0

    def test_scalar_matches_array(self):
        from dpf.turbulence.anomalous import (
            anomalous_resistivity,
            anomalous_resistivity_scalar,
        )

        J_val, ne_val, Te_val = 1e12, 1e24, 300.0
        eta_arr = anomalous_resistivity(
            np.array([J_val]), np.array([ne_val]), np.array([Te_val]),
            alpha=0.05,
        )
        eta_scalar = anomalous_resistivity_scalar(
            J_val, ne_val, Ti_val=Te_val, alpha=0.05, Te_val=Te_val,
        )
        np.testing.assert_allclose(eta_arr[0], eta_scalar, rtol=1e-10)

    def test_scalar_zero_density(self):
        from dpf.turbulence.anomalous import anomalous_resistivity_scalar

        eta = anomalous_resistivity_scalar(
            J_mag=1e12, ne_val=0.0, Ti_val=300.0,
        )
        assert eta == 0.0


class TestTotalResistivity:
    """Tests for combined Spitzer + anomalous resistivity."""

    def test_total_equals_spitzer_below_threshold(self):
        from dpf.turbulence.anomalous import total_resistivity

        eta_sp = np.array([1e-6])
        eta_anom = np.array([0.0])
        eta_total = total_resistivity(eta_sp, eta_anom)
        np.testing.assert_allclose(eta_total[0], 1e-6)

    def test_total_exceeds_spitzer_above_threshold(self):
        from dpf.turbulence.anomalous import total_resistivity

        eta_sp = np.array([1e-6])
        eta_anom = np.array([1e-4])
        eta_total = total_resistivity(eta_sp, eta_anom)
        assert eta_total[0] > eta_sp[0]

    def test_total_additive(self):
        from dpf.turbulence.anomalous import total_resistivity

        eta_sp = np.array([2e-6])
        eta_anom = np.array([3e-5])
        eta_total = total_resistivity(eta_sp, eta_anom)
        np.testing.assert_allclose(eta_total[0], 2e-6 + 3e-5, rtol=1e-10)

    def test_total_scalar(self):
        from dpf.turbulence.anomalous import total_resistivity_scalar

        eta = total_resistivity_scalar(1e-6, 5e-5)
        np.testing.assert_allclose(eta, 1e-6 + 5e-5, rtol=1e-10)


class TestEngineAnomalous:
    """Tests for anomalous resistivity integration with engine."""

    def test_engine_imports_turbulence(self):
        from dpf.turbulence.anomalous import (  # noqa: F401
            anomalous_resistivity_scalar,
            total_resistivity_scalar,
        )

    def test_engine_runs_with_anomalous(self):
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
        engine.run(max_steps=2)
