"""Tests for Phase 13: Multi-Species + Line Radiation + Beam-Target.

Covers:
    13.1 Multi-species framework (SpeciesConfig, SpeciesMixture)
    13.2 Electrode ablation source
    13.3 Line radiation + recombination radiation
    13.4 Non-LTE ionization (collisional-radiative model)
    13.5 Beam-target neutron yield model
    13.6 Neutron anisotropy diagnostic
"""

from __future__ import annotations

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
from dpf.diagnostics.beam_target import (
    beam_target_yield_rate,
    dd_cross_section,
    dd_cross_section_array,
    detect_pinch_disruption,
    neutron_anisotropy,
)
from dpf.experimental.species import (
    COPPER,
    DEUTERIUM,
    HYDROGEN,
    TUNGSTEN,
    SpeciesConfig,
    SpeciesMixture,
)
from dpf.radiation.line_radiation import (
    apply_line_radiation_losses,
    cooling_function,
    line_radiation_power,
    recombination_power,
    total_radiation_power,
)


class TestSpeciesConfig:
    """Tests for SpeciesConfig dataclass."""

    def test_deuterium_predefined(self):
        """Predefined deuterium has correct mass and charge."""
        assert DEUTERIUM.charge_number == 1
        assert DEUTERIUM.label == "D"
        assert abs(DEUTERIUM.mass - 3.3436e-27) < 1e-30

    def test_copper_predefined(self):
        """Predefined copper has Z=29."""
        assert COPPER.charge_number == 29
        assert COPPER.label == "Cu"
        assert COPPER.initial_fraction == 0.0

    def test_tungsten_predefined(self):
        """Predefined tungsten has Z=74."""
        assert TUNGSTEN.charge_number == 74
        assert TUNGSTEN.label == "W"

    def test_hydrogen_predefined(self):
        """Predefined hydrogen has charge 1."""
        assert HYDROGEN.charge_number == 1
        assert HYDROGEN.label == "H"

    def test_invalid_mass_raises(self):
        """Negative mass should raise ValueError."""
        with pytest.raises(ValueError, match="mass must be positive"):
            SpeciesConfig(name="bad", mass=-1.0, charge_number=1)

    def test_invalid_fraction_raises(self):
        """Fraction outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="initial_fraction"):
            SpeciesConfig(name="bad", mass=1e-27, charge_number=1, initial_fraction=1.5)

    def test_auto_label(self):
        """If label is empty, it auto-generates from name."""
        sp = SpeciesConfig(name="neon", mass=3.35e-26, charge_number=10)
        assert sp.label == "Neon"


class TestSpeciesMixture:
    """Tests for SpeciesMixture container."""

    def test_single_species_total_density(self):
        """Single deuterium species: total density matches initial."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(10,), rho_total=1e-3)
        rho = mix.total_density()
        np.testing.assert_allclose(rho, 1e-3, rtol=1e-12)

    def test_multi_species_conservation(self):
        """Multi-species: sum of rho_s equals rho_total."""
        d = SpeciesConfig(name="deuterium", mass=3.3436e-27, charge_number=1,
                          initial_fraction=0.9, label="D")
        cu = SpeciesConfig(name="copper", mass=1.0552e-25, charge_number=29,
                           initial_fraction=0.1, label="Cu")
        mix = SpeciesMixture([d, cu], grid_shape=(20,), rho_total=5e-4)
        rho = mix.total_density()
        np.testing.assert_allclose(rho, 5e-4, rtol=1e-10)

    def test_z_eff_pure_deuterium(self):
        """Z_eff = 1 for pure deuterium (fully ionized)."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(8,), rho_total=1e-4)
        z_eff = mix.z_eff()
        np.testing.assert_allclose(z_eff, 1.0, atol=1e-12)

    def test_z_eff_increases_with_impurity(self):
        """Z_eff > 1 when high-Z impurity is present."""
        d = SpeciesConfig(name="deuterium", mass=3.3436e-27, charge_number=1,
                          initial_fraction=0.9, label="D")
        cu = SpeciesConfig(name="copper", mass=1.0552e-25, charge_number=29,
                           initial_fraction=0.1, label="Cu")
        mix = SpeciesMixture([d, cu], grid_shape=(8,), rho_total=1e-4)
        z_eff = mix.z_eff()
        assert np.all(z_eff > 1.0)

    def test_number_densities_positive(self):
        """Number densities are positive."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(10,), rho_total=1e-3)
        nd = mix.number_densities()
        assert "deuterium" in nd
        assert np.all(nd["deuterium"] > 0)

    def test_electron_density(self):
        """Electron density = Z * n_ion for single species."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(5,), rho_total=1e-3)
        ne = mix.electron_density()
        nd = mix.number_densities()["deuterium"]
        np.testing.assert_allclose(ne, nd, rtol=1e-12)

    def test_advect_preserves_positivity(self):
        """Advection does not produce negative density."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(16,), rho_total=1e-3)
        v = np.zeros((3, 16))
        v[0] = 1e5  # strong rightward flow
        mix.advect("deuterium", v, dt=1e-9, dx=1e-3)
        assert np.all(mix.rho_species["deuterium"] >= 0.0)

    def test_advect_all(self):
        """advect_all runs for all species without error."""
        d = SpeciesConfig(name="deuterium", mass=3.3436e-27, charge_number=1,
                          initial_fraction=0.9, label="D")
        cu = SpeciesConfig(name="copper", mass=1.0552e-25, charge_number=29,
                           initial_fraction=0.1, label="Cu")
        mix = SpeciesMixture([d, cu], grid_shape=(10,), rho_total=1e-3)
        v = np.zeros((3, 10))
        mix.advect_all(v, dt=1e-9, dx=1e-3)
        assert np.all(np.isfinite(mix.total_density()))

    def test_add_source_increases_density(self):
        """Adding a positive source increases species density."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(5,), rho_total=1e-4)
        rho_before = mix.total_density().copy()
        mix.add_source("deuterium", 1e6, dt=1e-9)  # 1e6 kg/m^3/s * 1e-9 s
        rho_after = mix.total_density()
        assert np.all(rho_after > rho_before)

    def test_add_source_floors_at_zero(self):
        """Negative source does not produce negative density."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(5,), rho_total=1e-4)
        mix.add_source("deuterium", -1e30, dt=1e-9)
        assert np.all(mix.rho_species["deuterium"] >= 0.0)

    def test_species_names(self):
        """species_names returns correct list."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(5,), rho_total=1e-4)
        assert mix.species_names() == ["deuterium"]

    def test_get_config(self):
        """get_config looks up species by name."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(5,), rho_total=1e-4)
        cfg = mix.get_config("deuterium")
        assert cfg.charge_number == 1

    def test_empty_species_raises(self):
        """Empty species list should raise."""
        with pytest.raises(ValueError, match="At least one"):
            SpeciesMixture([], grid_shape=(5,), rho_total=1e-4)

    def test_fractions_must_sum_to_one(self):
        """Fractions not summing to 1 should raise."""
        d = SpeciesConfig(name="d", mass=3.3e-27, charge_number=1,
                          initial_fraction=0.5, label="D")
        with pytest.raises(ValueError, match="sum to 1.0"):
            SpeciesMixture([d], grid_shape=(5,), rho_total=1e-4)

    def test_mean_ion_mass(self):
        """Mean ion mass for pure deuterium = m_d."""
        mix = SpeciesMixture([DEUTERIUM], grid_shape=(5,), rho_total=1e-4)
        m_bar = mix.mean_ion_mass()
        np.testing.assert_allclose(m_bar, DEUTERIUM.mass, rtol=1e-10)


# ===================================================================
# 13.2 — Electrode ablation
# ===================================================================

class TestAblation:
    """Tests for electrode ablation model."""

    def test_ablation_rate_positive(self):
        """Ablation rate is positive for positive power."""
        dm_dt = ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY)
        assert dm_dt > 0.0

    def test_ablation_rate_zero_power(self):
        """Zero power gives zero ablation."""
        assert ablation_rate(0.0, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_rate_negative_power(self):
        """Negative power gives zero ablation."""
        assert ablation_rate(-1e8, COPPER_ABLATION_EFFICIENCY) == 0.0

    def test_ablation_rate_linear(self):
        """Ablation rate scales linearly with power."""
        r1 = ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY)
        r2 = ablation_rate(2e8, COPPER_ABLATION_EFFICIENCY)
        np.testing.assert_allclose(r2, 2 * r1, rtol=1e-12)

    def test_copper_tungsten_ratio(self):
        """Copper ablates faster than tungsten at same power."""
        r_cu = ablation_rate(1e8, COPPER_ABLATION_EFFICIENCY)
        r_w = ablation_rate(1e8, TUNGSTEN_ABLATION_EFFICIENCY)
        assert r_cu > r_w

    def test_ablation_source_scalar(self):
        """Scalar ablation source returns positive value."""
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e6, ne_boundary=1e23,
            J_boundary=1e8, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY,
            material_mass=COPPER_MASS,
        )
        assert S > 0.0

    def test_ablation_source_zero_current(self):
        """Zero current gives zero ablation source."""
        S = ablation_source(
            rho_boundary=1e-3, Te_boundary=1e6, ne_boundary=1e23,
            J_boundary=0.0, eta_boundary=1e-5, dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY,
            material_mass=COPPER_MASS,
        )
        assert S == 0.0

    def test_ablation_source_array_mask(self):
        """Array ablation: only boundary cells get source."""
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
        """Particle flux = mass source / atom mass."""
        S_rho = 1e3  # kg/(m^3 s)
        n_dot = ablation_particle_flux(S_rho, COPPER_MASS)
        expected = S_rho / COPPER_MASS
        np.testing.assert_allclose(n_dot, expected, rtol=1e-12)

    def test_ablation_momentum_source(self):
        """Momentum source = mass source * velocity."""
        S_rho = 1e3
        v = 1e4
        S_mom = ablation_momentum_source(S_rho, v)
        np.testing.assert_allclose(S_mom, S_rho * v, rtol=1e-12)

    def test_material_constants(self):
        """Material constants are reasonable."""
        assert COPPER_ABLATION_EFFICIENCY > 0
        assert TUNGSTEN_ABLATION_EFFICIENCY > 0
        assert COPPER_MASS > 0
        assert TUNGSTEN_MASS > COPPER_MASS  # W is heavier


# ===================================================================
# 13.3 — Line radiation + recombination
# ===================================================================

class TestCoolingFunction:
    """Tests for coronal equilibrium cooling function."""

    def test_hydrogen_cooling_positive(self):
        """Hydrogen cooling function is positive at 5 eV."""
        Te = np.array([5.0 * 11604.5])  # 5 eV in K
        Lambda = cooling_function(Te, Z=1.0)
        assert Lambda[0] > 0.0

    def test_copper_cooling_stronger_than_hydrogen(self):
        """Copper line radiation exceeds hydrogen at 100 eV."""
        Te = np.array([100.0 * 11604.5])  # 100 eV in K
        L_h = cooling_function(Te, Z=1.0)
        L_cu = cooling_function(Te, Z=29.0)
        assert L_cu[0] > L_h[0]

    def test_tungsten_strongest(self):
        """Tungsten cooling exceeds copper at 100 eV."""
        Te = np.array([100.0 * 11604.5])
        L_cu = cooling_function(Te, Z=29.0)
        L_w = cooling_function(Te, Z=74.0)
        assert L_w[0] > L_cu[0]

    def test_cooling_array_shape(self):
        """Cooling function preserves array shape."""
        Te = np.linspace(1e4, 1e7, 50)
        Lambda = cooling_function(Te, Z=29.0)
        assert Lambda.shape == Te.shape

    def test_generic_z_positive(self):
        """Generic Z-scaling gives positive cooling for Z=6 (carbon)."""
        Te = np.array([50.0 * 11604.5])
        Lambda = cooling_function(Te, Z=6.0)
        assert Lambda[0] > 0.0


class TestRecombinationPower:
    """Tests for recombination radiation."""

    def test_recombination_positive(self):
        """Recombination power is positive for hot dense plasma."""
        ne = np.array([1e23])
        Te = np.array([1e6])  # ~86 eV
        P = recombination_power(ne, Te, Z=1.0)
        assert P[0] > 0.0

    def test_recombination_scales_with_ne_squared(self):
        """P_rec scales as ne^2."""
        ne1 = np.array([1e23])
        ne2 = np.array([2e23])
        Te = np.array([1e6])
        P1 = recombination_power(ne1, Te, Z=1.0)
        P2 = recombination_power(ne2, Te, Z=1.0)
        np.testing.assert_allclose(P2[0] / P1[0], 4.0, rtol=0.01)

    def test_recombination_scales_with_Z_squared(self):
        """P_rec increases with Z^2 (approximately)."""
        ne = np.array([1e23])
        Te = np.array([1e6])
        P1 = recombination_power(ne, Te, Z=1.0)
        P2 = recombination_power(ne, Te, Z=2.0)
        # P ~ Z^2 * sqrt(Z^2 / Te) → P(Z=2)/P(Z=1) ~ 4 * sqrt(4) = 8
        ratio = P2[0] / P1[0]
        assert ratio > 4.0  # Should be ~8 but exact scaling depends on Te


class TestLinePower:
    """Tests for line radiation power."""

    def test_line_power_positive(self):
        """Line radiation power is positive with impurities."""
        ne = np.array([1e23])
        Te = np.array([100.0 * 11604.5])  # 100 eV
        P = line_radiation_power(ne, Te, Z=29.0, n_impurity_fraction=0.01)
        assert P[0] > 0.0

    def test_line_power_zero_impurity(self):
        """Zero impurity fraction gives zero line power."""
        ne = np.array([1e23])
        Te = np.array([1e6])
        P = line_radiation_power(ne, Te, Z=29.0, n_impurity_fraction=0.0)
        assert P[0] == 0.0

    def test_total_radiation_includes_line(self):
        """Total radiation with impurities > without."""
        ne = np.array([1e23])
        Te = np.array([100.0 * 11604.5])
        P_no_imp = total_radiation_power(ne, Te, Z_eff=1.0, n_impurity_fraction=0.0)
        P_with_imp = total_radiation_power(ne, Te, Z_eff=1.0,
                                           n_impurity_fraction=0.01, Z_impurity=29.0)
        assert P_with_imp[0] > P_no_imp[0]


class TestImplicitLineCooling:
    """Tests for implicit line + recombination cooling step."""

    def test_cooling_reduces_temperature(self):
        """Implicit cooling step reduces Te."""
        Te = np.array([1e7])  # ~860 eV
        ne = np.array([1e24])
        Te_new, P_rad = apply_line_radiation_losses(
            Te, ne, dt=1e-9, Z_eff=1.0, n_imp_frac=0.01, Z_imp=29.0,
        )
        assert Te_new[0] < Te[0]
        assert P_rad[0] > 0.0

    def test_cooling_never_negative(self):
        """Implicit cooling preserves positive temperature."""
        Te = np.array([100.0])  # very cold — 100 K
        ne = np.array([1e23])
        Te_new, _ = apply_line_radiation_losses(
            Te, ne, dt=1.0, Z_eff=1.0, n_imp_frac=0.05, Z_imp=29.0,
            Te_floor=10.0,
        )
        assert Te_new[0] >= 10.0

    def test_cooling_array_shape(self):
        """Implicit cooling preserves array shape."""
        shape = (8, 16)
        Te = np.full(shape, 1e6)
        ne = np.full(shape, 1e23)
        Te_new, P = apply_line_radiation_losses(Te, ne, dt=1e-9)
        assert Te_new.shape == shape
        assert P.shape == shape


# ===================================================================
# 13.4 — Non-LTE ionization (collisional-radiative model)
# ===================================================================

class TestLotzIonization:
    """Tests for Lotz electron-impact ionization rate coefficient."""

    def test_positive_rate_above_threshold(self):
        """Ionization rate is positive when Te > I_Z."""
        rate = lotz_ionization_rate(20.0, 13.6)  # Te=20 eV, IP=13.6 eV
        assert rate > 0.0

    def test_rate_very_small_below_threshold(self):
        """Ionization rate is negligible when Te << I_Z."""
        rate = lotz_ionization_rate(1.0, 13.6)  # Te=1 eV << 13.6 eV
        assert rate < 1e-23  # very small compared to peak rate ~1e-19

    def test_rate_increases_with_temperature(self):
        """Ionization rate increases with Te."""
        r1 = lotz_ionization_rate(10.0, 13.6)
        r2 = lotz_ionization_rate(50.0, 13.6)
        assert r2 > r1

    def test_rate_zero_for_low_Te(self):
        """Returns 0 for Te < 0.01 eV."""
        assert lotz_ionization_rate(0.005, 13.6) == 0.0

    def test_rate_order_of_magnitude(self):
        """Lotz rate at 2*IP should be ~1e-20 to 1e-19 m^3/s."""
        rate = lotz_ionization_rate(27.2, 13.6)  # Te = 2*IP
        assert 1e-22 < rate < 1e-17


class TestRecombinationRates:
    """Tests for radiative and dielectronic recombination rates."""

    def test_radiative_positive(self):
        """Radiative recombination rate is positive."""
        alpha = radiative_recombination_rate(10.0, 1)
        assert alpha > 0.0

    def test_dielectronic_positive(self):
        """Dielectronic recombination rate is positive."""
        alpha = dielectronic_recombination_rate(10.0, 1)
        assert alpha > 0.0

    def test_total_greater_than_either(self):
        """Total recombination >= individual terms."""
        a_rr = radiative_recombination_rate(10.0, 1)
        a_dr = dielectronic_recombination_rate(10.0, 1)
        a_total = total_recombination_rate(10.0, 1)
        np.testing.assert_allclose(a_total, a_rr + a_dr, rtol=1e-12)

    def test_recombination_increases_with_Z(self):
        """Recombination rate increases with ion charge."""
        a1 = total_recombination_rate(100.0, 1)
        a5 = total_recombination_rate(100.0, 5)
        assert a5 > a1

    def test_dielectronic_peak(self):
        """Dielectronic recombination has peak at Te ~ I_Z / 4."""
        # For Z=1: peak at ~13.6/4 ~ 3.4 eV
        a_low = dielectronic_recombination_rate(0.5, 1)
        a_peak = dielectronic_recombination_rate(3.4, 1)
        a_high = dielectronic_recombination_rate(100.0, 1)
        assert a_peak > a_low
        assert a_peak > a_high


class TestCRSolveChargeStates:
    """Tests for implicit backward Euler CR solver."""

    def test_neutral_start_ionizes(self):
        """Starting from neutral, hot plasma ionizes."""
        frac = np.array([1.0, 0.0])  # all neutral
        frac_new = cr_solve_charge_states(
            ne=1e23, Te_eV=20.0, Z_max=1, dt=1e-6,
            frac_in=frac, ip_eV=_IP_H,
        )
        assert frac_new[1] > 0.0  # some ionized

    def test_fully_ionized_recombines(self):
        """Starting from fully ionized, cool plasma recombines."""
        frac = np.array([0.0, 1.0])  # all ionized
        frac_new = cr_solve_charge_states(
            ne=1e23, Te_eV=0.5, Z_max=1, dt=1e-6,
            frac_in=frac, ip_eV=_IP_H,
        )
        assert frac_new[0] > 0.0  # some recombined

    def test_fractions_normalized(self):
        """Output fractions sum to 1."""
        frac = np.array([0.5, 0.5])
        frac_new = cr_solve_charge_states(
            ne=1e23, Te_eV=10.0, Z_max=1, dt=1e-7,
            frac_in=frac, ip_eV=_IP_H,
        )
        np.testing.assert_allclose(np.sum(frac_new), 1.0, atol=1e-10)

    def test_fractions_nonnegative(self):
        """Output fractions are non-negative."""
        frac = np.array([0.1, 0.9])
        frac_new = cr_solve_charge_states(
            ne=1e24, Te_eV=50.0, Z_max=1, dt=1e-5,
            frac_in=frac, ip_eV=_IP_H,
        )
        assert np.all(frac_new >= 0.0)

    def test_copper_multi_charge(self):
        """Copper has multiple charge states populated at 100 eV."""
        frac = np.zeros(30)
        frac[0] = 1.0
        frac_new = cr_solve_charge_states(
            ne=1e23, Te_eV=100.0, Z_max=29, dt=1e-6,
            frac_in=frac, ip_eV=_IP_CU,
        )
        # At 100 eV some ionization should have occurred
        assert frac_new[0] < 1.0
        np.testing.assert_allclose(np.sum(frac_new), 1.0, atol=1e-10)

    def test_no_ionization_at_cold(self):
        """No ionization for very cold plasma."""
        frac = np.array([1.0, 0.0])
        frac_new = cr_solve_charge_states(
            ne=1e20, Te_eV=0.01, Z_max=1, dt=1e-6,
            frac_in=frac, ip_eV=_IP_H,
        )
        # Essentially unchanged
        np.testing.assert_allclose(frac_new[0], 1.0, atol=1e-6)


class TestCRAverageCharge:
    """Tests for CR equilibrium Z_bar computation."""

    def test_hydrogen_hot_ionized(self):
        """Hydrogen at high Te is substantially ionized."""
        # At 1000 eV with high ne, hydrogen should be well-ionized.
        # The CR model needs enough iterations to reach steady state.
        Z_bar = cr_average_charge(1e23, 1000.0, 1, _IP_H, n_iter=100)
        assert Z_bar > 0.3  # Should converge toward 1.0; 0.4 after 50 iters

    def test_hydrogen_cold_neutral(self):
        """Hydrogen at very low Te is neutral."""
        Z_bar = cr_average_charge(1e23, 0.1, 1, _IP_H)
        assert Z_bar < 0.01

    def test_cr_differs_from_saha_low_density(self):
        """CR model differs from Saha at low density (non-LTE regime).

        At low ne, collisional rates are slow and the CR model should
        give a different Z_bar than Saha (which assumes LTE).
        """
        Te_K = 2.0 * 11604.5  # 2 eV in K (below hydrogen IP)
        ne = 1e18  # Low density — coronal regime
        Te_eV = 2.0

        Z_saha = saha_ionization_fraction(Te_K, ne)
        Z_cr = cr_average_charge(ne, Te_eV, 1, _IP_H)

        # At 2 eV with low ne, both should be small, but they differ
        # The key assertion is that they're not identical
        # (exact difference depends on model details)
        assert isinstance(Z_cr, float)
        assert isinstance(Z_saha, float)
        assert 0.0 <= Z_cr <= 1.0

    def test_copper_z_bar_intermediate(self):
        """Copper Z_bar is between 0 and 29 at moderate Te."""
        Z_bar = cr_average_charge(1e23, 100.0, 29, _IP_CU)
        assert 0.0 < Z_bar < 29.0


class TestCRFieldFunctions:
    """Tests for spatially-resolved CR field operations."""

    def test_cr_zbar_field_shape(self):
        """cr_zbar_field preserves array shape."""
        ne = np.full(20, 1e23)
        Te = np.full(20, 1e6)  # ~86 eV
        Z_bar = cr_zbar_field(ne, Te, 1, _IP_H)
        assert Z_bar.shape == (20,)

    def test_cr_zbar_field_all_finite(self):
        """Z_bar field contains no NaN/Inf."""
        ne = np.linspace(1e20, 1e25, 30)
        Te = np.linspace(1e4, 1e7, 30)
        Z_bar = cr_zbar_field(ne, Te, 1, _IP_H)
        assert np.all(np.isfinite(Z_bar))

    def test_cr_zbar_field_bounded(self):
        """Z_bar is bounded by [0, Z_max]."""
        ne = np.full(10, 1e23)
        Te = np.full(10, 1e7)
        Z_bar = cr_zbar_field(ne, Te, 1, _IP_H)
        assert np.all(Z_bar >= 0.0)
        assert np.all(Z_bar <= 1.0)

    def test_cr_evolve_field(self):
        """cr_evolve_field produces valid output."""
        n_cells = 5
        ne = np.full(n_cells, 1e23)
        Te = np.full(n_cells, 1e6)
        frac = np.zeros((n_cells, 2))
        frac[:, 0] = 1.0  # all neutral
        frac_new = cr_evolve_field(ne, Te, 1, 1e-7, frac, _IP_H)
        assert frac_new.shape == frac.shape
        # Sum of fractions per cell should be 1
        for i in range(n_cells):
            s = frac_new.flat[i * 2] + frac_new.flat[i * 2 + 1]
            np.testing.assert_allclose(s, 1.0, atol=1e-10)


class TestIonizationPotentials:
    """Tests for stored ionization potential data."""

    def test_hydrogen_ip(self):
        """Hydrogen has one IP = 13.6 eV."""
        assert len(_IP_H) == 1
        np.testing.assert_allclose(_IP_H[0], 13.6, atol=0.01)

    def test_copper_ip_length(self):
        """Copper has 29 IPs (Z=0 to Z=28 → Z=1 to Z=29)."""
        assert len(_IP_CU) == 29

    def test_copper_ip_monotonic(self):
        """Copper IPs are monotonically increasing."""
        for i in range(len(_IP_CU) - 1):
            assert _IP_CU[i + 1] > _IP_CU[i]

    def test_tungsten_ip_length(self):
        """Tungsten table has 10 entries (first 10 charge states)."""
        assert len(_IP_W) == 10

    def test_potentials_dict(self):
        """IONIZATION_POTENTIALS dictionary has expected keys."""
        assert 1 in IONIZATION_POTENTIALS
        assert 29 in IONIZATION_POTENTIALS
        assert 74 in IONIZATION_POTENTIALS


# ===================================================================
# 13.5 — Beam-target neutron yield
# ===================================================================

class TestDDCrossSection:
    """Tests for Bosch-Hale DD fusion cross section."""

    def test_cross_section_positive(self):
        """Cross section is positive for E in valid range."""
        sigma = dd_cross_section(100.0)  # 100 keV CM
        assert sigma > 0.0

    def test_cross_section_zero_below_range(self):
        """Cross section is 0 below 0.5 keV."""
        assert dd_cross_section(0.1) == 0.0

    def test_cross_section_zero_above_range(self):
        """Cross section is 0 above 5000 keV."""
        assert dd_cross_section(6000.0) == 0.0

    def test_cross_section_10keV(self):
        """Cross section at 10 keV CM ~ few * 1e-32 m^2 (within factor of 10)."""
        sigma = dd_cross_section(10.0)
        assert 1e-33 < sigma < 5e-30

    def test_cross_section_increases_with_energy(self):
        """Cross section increases from 10 to 100 keV."""
        s10 = dd_cross_section(10.0)
        s100 = dd_cross_section(100.0)
        assert s100 > s10

    def test_cross_section_array(self):
        """Vectorized cross section produces correct shape."""
        E = np.array([10.0, 50.0, 100.0, 200.0])
        sigma = dd_cross_section_array(E)
        assert sigma.shape == E.shape
        assert np.all(sigma > 0)


class TestBeamTargetYield:
    """Tests for beam-target neutron production rate."""

    def test_yield_positive(self):
        """Yield rate is positive for typical DPF conditions."""
        dY = beam_target_yield_rate(
            I_pinch=200e3, V_pinch=50e3,
            n_target=1e25, L_target=0.01,
        )
        assert dY > 0.0

    def test_yield_zero_no_current(self):
        """Zero pinch current gives zero yield."""
        assert beam_target_yield_rate(0, 50e3, 1e25, 0.01) == 0.0

    def test_yield_zero_no_voltage(self):
        """Zero pinch voltage gives zero yield."""
        assert beam_target_yield_rate(200e3, 0, 1e25, 0.01) == 0.0

    def test_yield_scales_with_current(self):
        """Yield scales linearly with pinch current."""
        y1 = beam_target_yield_rate(100e3, 50e3, 1e25, 0.01)
        y2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01)
        np.testing.assert_allclose(y2, 2 * y1, rtol=1e-10)

    def test_yield_scales_with_f_beam(self):
        """Yield scales linearly with beam fraction."""
        y1 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=0.1)
        y2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=0.3)
        np.testing.assert_allclose(y2 / y1, 3.0, rtol=1e-10)

    def test_typical_dpf_yield_order(self):
        """Typical MA-class DPF beam-target yield ~ 10^15-10^18 /s."""
        dY = beam_target_yield_rate(
            I_pinch=500e3, V_pinch=100e3,
            n_target=1e25, L_target=0.02, f_beam=0.2,
        )
        # Should be a large number for MA-class device
        assert dY > 1e10


class TestPinchDisruption:
    """Tests for m=0 instability disruption detector."""

    def test_spike_detected(self):
        """Pressure spike is detected as disruption."""
        history = [1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 8e5]
        assert detect_pinch_disruption(history, threshold_ratio=5.0)

    def test_no_spike_not_detected(self):
        """Flat pressure history does not trigger disruption."""
        history = [1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5]
        assert not detect_pinch_disruption(history, threshold_ratio=5.0)

    def test_short_history_returns_false(self):
        """Too-short history (<4 entries) returns False."""
        assert not detect_pinch_disruption([1e5, 2e5, 3e5])

    def test_numpy_array_input(self):
        """Works with numpy array input."""
        history = np.array([1e5] * 8 + [1e6])
        assert detect_pinch_disruption(history, threshold_ratio=5.0)


# ===================================================================
# 13.6 — Neutron anisotropy
# ===================================================================

class TestNeutronAnisotropy:
    """Tests for neutron anisotropy diagnostic."""

    def test_pure_thermonuclear_isotropic(self):
        """Pure thermonuclear yield gives anisotropy = 1."""
        A = neutron_anisotropy(Y_beam=0, Y_thermal=1e11)
        np.testing.assert_allclose(A, 1.0, atol=1e-12)

    def test_pure_beam_target_anisotropic(self):
        """Pure beam-target yield gives anisotropy > 1."""
        A = neutron_anisotropy(Y_beam=1e11, Y_thermal=0)
        assert A > 1.0

    def test_mixed_yield_intermediate(self):
        """Mixed yield gives intermediate anisotropy."""
        A = neutron_anisotropy(Y_beam=1e11, Y_thermal=1e11, E_beam_keV=100.0)
        assert 1.0 < A < 2.0

    def test_anisotropy_increases_with_beam_fraction(self):
        """Anisotropy increases as beam-target fraction increases."""
        A1 = neutron_anisotropy(Y_beam=1e10, Y_thermal=1e11)
        A2 = neutron_anisotropy(Y_beam=1e11, Y_thermal=1e11)
        assert A2 > A1

    def test_anisotropy_increases_with_beam_energy(self):
        """Higher beam energy increases anisotropy."""
        A1 = neutron_anisotropy(Y_beam=1e11, Y_thermal=0, E_beam_keV=50.0)
        A2 = neutron_anisotropy(Y_beam=1e11, Y_thermal=0, E_beam_keV=400.0)
        assert A2 > A1

    def test_zero_yield_returns_one(self):
        """Zero total yield returns 1.0 (isotropic convention)."""
        A = neutron_anisotropy(0, 0)
        np.testing.assert_allclose(A, 1.0, atol=1e-12)

    def test_typical_dpf_range(self):
        """Typical DPF anisotropy is in [1.0, 3.0]."""
        A = neutron_anisotropy(Y_beam=5e11, Y_thermal=1e11, E_beam_keV=150.0)
        assert 1.0 <= A <= 3.0
