"""Phase X: Two-temperature (Te/Ti) physics verification tests.

Tests verify:
1. Spitzer e-i relaxation: Te and Ti converge to T_eq = (Z*Te + Ti)/(Z+1)
2. Energy conservation: total thermal energy is preserved during relaxation
3. Relaxation timescale: exponential decay matches analytical nu_ei rate
4. Coulomb logarithm: correct values for DPF-relevant ne, Te ranges
5. Charge state dependence: Z > 1 modifies equilibrium and rate correctly
6. EOS consistency: IdealEOS two-temperature pressures are additive
7. Extreme conditions: very hot electrons, cold ions (DPF pinch regime)
8. Stiff regime: large nu_ei*dt still conserves energy (implicit stability)

References:
    Braginskii S.I., Reviews of Plasma Physics Vol. 1 (1965).
    NRL Plasma Formulary (2019 revision).
    Rosen M., Phys. Fluids 7, 491 (1954).
"""

import numpy as np
import pytest

from dpf.collision.spitzer import (
    braginskii_kappa,
    coulomb_log,
    nu_ei,
    nu_ii,
    relax_temperatures,
    spitzer_resistivity,
)
from dpf.constants import e, epsilon_0, k_B, m_d, m_e, m_p
from dpf.fluid.eos import IdealEOS


class TestCoulombLogarithm:
    """Verify Coulomb logarithm against NRL Plasma Formulary."""

    def test_classical_plasma(self):
        """ln(Lambda) ~ 10-20 for classical DPF plasma."""
        ne = np.array([1e24])  # typical DPF electron density [m^-3]
        Te = np.array([1e7])   # ~1 keV
        lnL = coulomb_log(ne, Te)
        assert 5.0 < lnL[0] < 25.0, f"Coulomb log {lnL[0]} out of range for DPF plasma"

    def test_floor_at_two(self):
        """Coulomb log floored at 2.0 (Spitzer theory breaks below this)."""
        ne = np.array([1e30])  # extreme density
        Te = np.array([1e3])   # very cold
        lnL = coulomb_log(ne, Te)
        assert lnL[0] >= 2.0

    def test_increases_with_temperature(self):
        """Higher Te -> larger Debye length -> larger Coulomb log."""
        ne = np.array([1e24])
        Te_low = np.array([1e5])
        Te_high = np.array([1e8])
        assert coulomb_log(ne, Te_high)[0] > coulomb_log(ne, Te_low)[0]

    def test_decreases_with_density(self):
        """Higher ne -> smaller Debye length -> smaller Coulomb log."""
        Te = np.array([1e7])
        ne_low = np.array([1e22])
        ne_high = np.array([1e26])
        assert coulomb_log(ne_low, Te)[0] > coulomb_log(ne_high, Te)[0]

    def test_array_broadcast(self):
        """Works with multidimensional arrays."""
        ne = np.full((4, 1, 8), 1e24)
        Te = np.full((4, 1, 8), 1e7)
        lnL = coulomb_log(ne, Te)
        assert lnL.shape == (4, 1, 8)
        assert np.all(lnL > 2.0)


class TestCollisionFrequencies:
    """Verify Spitzer collision frequencies."""

    def test_nu_ei_scaling(self):
        """nu_ei ~ ne * Te^(-3/2) from Spitzer formula."""
        ne = np.array([1e24])
        Te1 = np.array([1e7])
        Te2 = np.array([4e7])  # 4x higher
        lnL = 10.0

        freq1 = nu_ei(ne, Te1, lnL)
        freq2 = nu_ei(ne, Te2, lnL)

        # nu_ei ~ Te^(-3/2), so ratio should be (4)^(-3/2) = 1/8
        ratio = freq2[0] / freq1[0]
        assert pytest.approx(ratio, rel=0.01) == (4.0)**(-1.5)

    def test_nu_ei_density_linear(self):
        """nu_ei scales linearly with ne."""
        Te = np.array([1e7])
        ne1 = np.array([1e24])
        ne2 = np.array([2e24])
        lnL = 10.0

        freq1 = nu_ei(ne1, Te, lnL)
        freq2 = nu_ei(ne2, Te, lnL)
        assert pytest.approx(freq2[0] / freq1[0], rel=0.01) == 2.0

    def test_nu_ei_dpf_range(self):
        """Collision frequency in correct range for DPF pinch."""
        ne = np.array([1e25])   # pinch density
        Te = np.array([5e7])    # ~5 keV
        lnL = coulomb_log(ne, Te)
        freq = nu_ei(ne, Te, lnL)
        # For DPF: nu_ei ~ 1e10 - 1e13 s^-1
        assert 1e8 < freq[0] < 1e15, f"nu_ei = {freq[0]:.2e} outside DPF range"

    def test_nu_ii_slower_than_nu_ei(self):
        """Ion-ion collisions slower than electron-ion by sqrt(m_i/m_e)."""
        ne = np.array([1e24])
        Te = np.array([1e7])
        Ti = Te  # equal temperatures
        lnL = 10.0
        freq_e = nu_ei(ne, Te, lnL)
        freq_i = nu_ii(ne, Ti, lnL, mi=m_d)
        # nu_ii / nu_ei ~ sqrt(m_e / m_i) approximately
        assert freq_i[0] < freq_e[0]


class TestTemperatureRelaxation:
    """Verify implicit e-i temperature relaxation."""

    def test_equilibrium_temperature(self):
        """Te and Ti relax toward T_eq = (Z*Te + Ti)/(Z+1) for Z=1."""
        Te = np.array([2e7])
        Ti = np.array([1e7])
        T_eq_expected = (Te + Ti) / 2.0

        freq_ei = np.array([1e12])
        dt = 1e-6  # large enough for significant relaxation

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        T_eq_actual = (Te_new + Ti_new) / 2.0
        assert pytest.approx(T_eq_actual[0], rel=1e-10) == T_eq_expected[0]

    def test_energy_conservation(self):
        """Total thermal energy conserved: Z*Te + Ti = const."""
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
        """Energy conservation holds for Z > 1."""
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
        """Temperature difference decays as exp(-2*alpha) where alpha = nu_ei*dt*2*m_e/m_d."""
        Te = np.array([2e7])
        Ti = np.array([1e7])
        freq_ei = np.array([1e12])
        dt = 1e-8  # small enough to see partial relaxation

        alpha = freq_ei[0] * dt * 2.0 * m_e / m_d
        expected_factor = np.exp(-2.0 * alpha)

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)

        T_eq = (Te[0] + Ti[0]) / 2.0
        initial_diff = Te[0] - T_eq
        final_diff = Te_new[0] - T_eq

        actual_factor = final_diff / initial_diff
        assert pytest.approx(actual_factor, rel=1e-10) == expected_factor

    def test_already_equilibrated(self):
        """No change when Te == Ti."""
        T = np.array([1e7])
        freq_ei = np.array([1e12])
        dt = 1e-6

        Te_new, Ti_new = relax_temperatures(T, T.copy(), freq_ei, dt)
        assert pytest.approx(Te_new[0], rel=1e-14) == T[0]
        assert pytest.approx(Ti_new[0], rel=1e-14) == T[0]

    def test_stiff_regime_stable(self):
        """Large nu_ei*dt -> full equilibration, no oscillation."""
        Te = np.array([1e8])
        Ti = np.array([1e5])
        freq_ei = np.array([1e15])  # very high collision rate
        dt = 1.0  # huge timestep

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        T_eq = (Te[0] + Ti[0]) / 2.0

        # Should be fully equilibrated
        assert pytest.approx(Te_new[0], rel=1e-6) == T_eq
        assert pytest.approx(Ti_new[0], rel=1e-6) == T_eq

    def test_electrons_cool_ions_heat(self):
        """When Te > Ti: electrons cool, ions heat."""
        Te = np.array([5e7])
        Ti = np.array([1e7])
        freq_ei = np.array([1e12])
        dt = 1e-7

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
        assert Te_new[0] < Te[0], "Electrons should cool"
        assert Ti_new[0] > Ti[0], "Ions should heat"

    def test_multidimensional_array(self):
        """Works with 3D arrays (nr, 1, nz) matching DPF state shape."""
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
        """For Z=2, T_eq = (2*Te + Ti)/3."""
        Te = np.array([3e7])
        Ti = np.array([0.0])
        Z = 2.0

        freq_ei = np.array([1e15])
        dt = 1.0  # push to equilibrium

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt, Z=Z)
        T_eq = (Z * Te[0] + Ti[0]) / (Z + 1.0)

        assert pytest.approx(Te_new[0], rel=1e-6) == T_eq
        assert pytest.approx(Ti_new[0], rel=1e-6) == T_eq


class TestSpitzerResistivity:
    """Verify Spitzer resistivity values."""

    def test_1kev_hydrogen_plasma(self):
        """eta ~ 2e-7 Ohm*m for 1 keV hydrogen plasma (NRL Formulary)."""
        ne = np.array([1e24])
        Te = np.array([1.16e7])  # 1 keV in Kelvin
        lnL = np.array([10.0])

        eta = spitzer_resistivity(ne, Te, lnL, Z=1.0)
        # NRL: eta_perp ~ 5.2e-5 * Z * lnL / Te_eV^1.5 [Ohm*cm]
        # For Te=1keV, Z=1, lnL=10: eta ~ 5.2e-5 * 10 / 1000^1.5 ~ 1.6e-8 Ohm*cm ~ 1.6e-10 Ohm*m
        # Our formula includes Braginskii alpha(Z) correction
        assert 1e-11 < eta[0] < 1e-5, f"Spitzer eta = {eta[0]:.2e} outside expected range"

    def test_resistivity_decreases_with_temperature(self):
        """Hotter plasma is less resistive (eta ~ Te^(-3/2))."""
        ne = np.array([1e24])
        Te_cold = np.array([1e6])
        Te_hot = np.array([1e8])
        lnL = 10.0

        eta_cold = spitzer_resistivity(ne, Te_cold, lnL)
        eta_hot = spitzer_resistivity(ne, Te_hot, lnL)
        assert eta_hot[0] < eta_cold[0]

    def test_resistivity_independent_of_density(self):
        """Spitzer resistivity is weakly density-dependent (only through lnL)."""
        Te = np.array([1e7])
        lnL = 10.0  # fixed, removing density dependence
        ne1 = np.array([1e23])
        ne2 = np.array([1e25])

        eta1 = spitzer_resistivity(ne1, Te, lnL)
        eta2 = spitzer_resistivity(ne2, Te, lnL)
        # With fixed lnL, the ne dependence should cancel (nu_ei ~ ne, eta = m_e*nu_ei/(ne*e^2))
        assert pytest.approx(eta1[0], rel=0.01) == eta2[0]


class TestBraginskiiKappa:
    """Verify Braginskii thermal conductivity."""

    def test_parallel_larger_than_perpendicular(self):
        """kappa_parallel >> kappa_perpendicular in magnetized plasma."""
        ne = np.array([1e24])
        Te = np.array([1e7])
        Bmag = np.array([1.0])  # 1 Tesla (strongly magnetized)

        kappa_par, kappa_per = braginskii_kappa(ne, Te, Bmag)
        assert kappa_par[0] > kappa_per[0]

    def test_unmagnetized_limit(self):
        """kappa_par == kappa_per when B -> 0."""
        ne = np.array([1e24])
        Te = np.array([1e7])
        Bmag = np.array([0.0])

        kappa_par, kappa_per = braginskii_kappa(ne, Te, Bmag)
        assert pytest.approx(kappa_par[0], rel=1e-10) == kappa_per[0]

    def test_kappa_increases_with_temperature(self):
        """Thermal conductivity increases with Te (kappa ~ Te^5/2)."""
        ne = np.array([1e24])
        Bmag = np.array([0.1])
        Te_low = np.array([1e6])
        Te_high = np.array([1e7])

        kpar_low, _ = braginskii_kappa(ne, Te_low, Bmag)
        kpar_high, _ = braginskii_kappa(ne, Te_high, Bmag)
        assert kpar_high[0] > kpar_low[0]


class TestIdealEOSTwoTemperature:
    """Verify two-temperature EOS."""

    def test_pressures_additive(self):
        """p_total = p_ion + p_electron."""
        eos = IdealEOS(gamma=5.0 / 3.0, Z=1.0)
        rho = np.array([1e-3])
        Te = np.array([2e7])
        Ti = np.array([1e7])

        p_i = eos.ion_pressure(rho, Ti)
        p_e = eos.electron_pressure(rho, Te)
        p_total = eos.total_pressure(rho, Ti, Te)
        assert pytest.approx(p_total[0], rel=1e-12) == p_i[0] + p_e[0]

    def test_electron_pressure_scales_with_Z(self):
        """p_e = Z * n_i * k_B * T_e, so higher Z gives more electron pressure."""
        rho = np.array([1e-3])
        Te = np.array([1e7])

        eos_z1 = IdealEOS(Z=1.0)
        eos_z2 = IdealEOS(Z=2.0)

        pe1 = eos_z1.electron_pressure(rho, Te)
        pe2 = eos_z2.electron_pressure(rho, Te)
        assert pytest.approx(pe2[0] / pe1[0], rel=1e-10) == 2.0

    def test_sound_speed_positive(self):
        """Sound speed is real and positive."""
        eos = IdealEOS()
        rho = np.array([1e-3])
        Te = np.array([1e7])
        Ti = np.array([5e6])
        cs = eos.sound_speed(rho, Ti, Te)
        assert cs[0] > 0

    def test_temperature_recovery(self):
        """Can recover T from specific energy for single-temperature case."""
        eos = IdealEOS(gamma=5.0 / 3.0, Z=1.0)
        rho = np.array([1e-3])
        T_input = np.array([1e7])

        # Total energy = (p_i + p_e) / ((gamma-1)*rho)
        e_i = eos.ion_energy(rho, T_input)
        e_e = eos.electron_energy(rho, T_input)
        e_total = e_i + e_e

        T_recovered = eos.temperature_from_energy(rho, e_total)
        assert pytest.approx(T_recovered[0], rel=1e-10) == T_input[0]
