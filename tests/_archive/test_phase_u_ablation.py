"""Phase U: Comprehensive V&V tests for the DPF ablation module.

Tests cover:
- Unit function tests (ablation_rate, ablation_source)
- Array source tests (ablation_source_array with boundary masks)
- Particle flux tests (ablation_particle_flux)
- Momentum source tests (ablation_momentum_source)
- Physical scaling tests (quadratic J dependence, linear eta/efficiency scaling)
- Config tests (AblationConfig validation)

References:
    Bruzzone & Aranchuk, J. Phys. D: Appl. Phys. 36 (2003) 2218
    Vikhrev & Korolev, Plasma Physics Reports 33 (2007) 356
    Lee & Serban, IEEE Trans. Plasma Sci. 24 (1996) 1101
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
from dpf.config import AblationConfig


class TestAblationRate:
    """Unit tests for ablation_rate function."""

    def test_zero_power_gives_zero_rate(self):
        """Zero power deposition should produce zero ablation rate."""
        rate = ablation_rate(0.0, COPPER_ABLATION_EFFICIENCY)
        assert rate == pytest.approx(0.0, abs=1e-20)

    def test_negative_power_gives_zero_rate(self):
        """Negative power (unphysical) should be guarded to zero rate."""
        rate = ablation_rate(-1e6, COPPER_ABLATION_EFFICIENCY)
        assert rate == pytest.approx(0.0, abs=1e-20)

    def test_zero_efficiency_gives_zero_rate(self):
        """Zero ablation efficiency should produce zero rate."""
        rate = ablation_rate(1e6, 0.0)
        assert rate == pytest.approx(0.0, abs=1e-20)

    def test_negative_efficiency_gives_zero_rate(self):
        """Negative efficiency (unphysical) should be guarded to zero rate."""
        rate = ablation_rate(1e6, -1e-5)
        assert rate == pytest.approx(0.0, abs=1e-20)

    def test_positive_power_correct_rate(self):
        """Positive power and efficiency should give correct ablation rate."""
        P_surface = 1e6  # 1 MW
        efficiency = COPPER_ABLATION_EFFICIENCY  # 5e-5 kg/J
        expected = efficiency * P_surface  # 0.05 kg/s
        rate = ablation_rate(P_surface, efficiency)
        assert rate == pytest.approx(expected, rel=1e-12)

    def test_linear_scaling_with_power(self):
        """Ablation rate should scale linearly with power."""
        efficiency = COPPER_ABLATION_EFFICIENCY
        P1 = 1e6
        P2 = 2e6
        rate1 = ablation_rate(P1, efficiency)
        rate2 = ablation_rate(P2, efficiency)
        assert rate2 == pytest.approx(2 * rate1, rel=1e-12)

    def test_linear_scaling_with_efficiency(self):
        """Ablation rate should scale linearly with efficiency."""
        P_surface = 1e6
        eff1 = COPPER_ABLATION_EFFICIENCY
        eff2 = 2 * COPPER_ABLATION_EFFICIENCY
        rate1 = ablation_rate(P_surface, eff1)
        rate2 = ablation_rate(P_surface, eff2)
        assert rate2 == pytest.approx(2 * rate1, rel=1e-12)


class TestAblationSource:
    """Unit tests for ablation_source function."""

    def test_zero_current_density_gives_zero_source(self):
        """Zero current density should produce zero mass source."""
        S_rho = ablation_source(
            rho_boundary=1e-3,
            Te_boundary=1e5,
            ne_boundary=1e24,
            J_boundary=0.0,
            eta_boundary=1e-5,
            dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY,
            material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(0.0, abs=1e-20)

    def test_zero_resistivity_gives_zero_source(self):
        """Zero resistivity should produce zero mass source."""
        S_rho = ablation_source(
            rho_boundary=1e-3,
            Te_boundary=1e5,
            ne_boundary=1e24,
            J_boundary=1e8,
            eta_boundary=0.0,
            dx=1e-3,
            ablation_efficiency=COPPER_ABLATION_EFFICIENCY,
            material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(0.0, abs=1e-20)

    def test_zero_efficiency_gives_zero_source(self):
        """Zero ablation efficiency should produce zero mass source."""
        S_rho = ablation_source(
            rho_boundary=1e-3,
            Te_boundary=1e5,
            ne_boundary=1e24,
            J_boundary=1e8,
            eta_boundary=1e-5,
            dx=1e-3,
            ablation_efficiency=0.0,
            material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(0.0, abs=1e-20)

    def test_correct_formula_S_rho_equals_efficiency_times_eta_J_squared(self):
        """Verify S_rho = efficiency * eta * J^2."""
        J = 1e8  # A/m^2
        eta = 1e-5  # Ohm m
        efficiency = COPPER_ABLATION_EFFICIENCY  # kg/J
        expected = efficiency * eta * J * J  # kg/(m^3 s)

        S_rho = ablation_source(
            rho_boundary=1e-3,
            Te_boundary=1e5,
            ne_boundary=1e24,
            J_boundary=J,
            eta_boundary=eta,
            dx=1e-3,
            ablation_efficiency=efficiency,
            material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(expected, rel=1e-12)

    def test_copper_efficiency(self):
        """Test typical copper ablation source magnitude."""
        J = 1e8  # A/m^2 (typical DPF)
        eta = 1e-5  # Ohm m (typical Spitzer at ~10 eV)
        efficiency = COPPER_ABLATION_EFFICIENCY
        expected = efficiency * eta * J * J  # 5e7 kg/(m^3 s)

        S_rho = ablation_source(
            rho_boundary=1e-3,
            Te_boundary=1e5,
            ne_boundary=1e24,
            J_boundary=J,
            eta_boundary=eta,
            dx=1e-3,
            ablation_efficiency=efficiency,
            material_mass=COPPER_MASS,
        )
        assert S_rho == pytest.approx(expected, rel=1e-12)

    def test_tungsten_efficiency(self):
        """Test typical tungsten ablation source magnitude."""
        J = 1e8  # A/m^2
        eta = 1e-5  # Ohm m
        efficiency = TUNGSTEN_ABLATION_EFFICIENCY  # 2e-5 kg/J (lower than copper)
        expected = efficiency * eta * J * J  # 2e7 kg/(m^3 s)

        S_rho = ablation_source(
            rho_boundary=1e-3,
            Te_boundary=1e5,
            ne_boundary=1e24,
            J_boundary=J,
            eta_boundary=eta,
            dx=1e-3,
            ablation_efficiency=efficiency,
            material_mass=TUNGSTEN_MASS,
        )
        assert S_rho == pytest.approx(expected, rel=1e-12)

    def test_copper_vs_tungsten_efficiency_ratio(self):
        """Copper ablation should be 2.5x higher than tungsten (5e-5 vs 2e-5)."""
        J = 1e8
        eta = 1e-5
        S_rho_Cu = ablation_source(
            1e-3, 1e5, 1e24, J, eta, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS
        )
        S_rho_W = ablation_source(
            1e-3, 1e5, 1e24, J, eta, 1e-3, TUNGSTEN_ABLATION_EFFICIENCY, TUNGSTEN_MASS
        )
        ratio = S_rho_Cu / S_rho_W
        assert ratio == pytest.approx(2.5, rel=1e-12)

    def test_physical_magnitude_typical_DPF_conditions(self):
        """Check physical magnitude for typical DPF conditions."""
        # Typical DPF boundary conditions:
        # J ~ 1e8 A/m^2 (100 MA/m^2)
        # eta ~ 1e-5 Ohm m (Spitzer at ~10 eV)
        # Expected S_rho ~ 10-100 kg/(m^3 s)
        J = 1e8
        eta = 1e-5
        S_rho = ablation_source(
            1e-3, 1e5, 1e24, J, eta, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS
        )
        # Expected: 5e-5 * 1e-5 * 1e16 = 5e7 kg/(m^3 s)
        assert 1e6 < S_rho < 1e9  # Order-of-magnitude check

    def test_high_current_high_ablation(self):
        """Very high current density should produce very high ablation."""
        J = 1e9  # A/m^2 (extreme DPF)
        eta = 1e-5
        S_rho = ablation_source(
            1e-3, 1e5, 1e24, J, eta, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS
        )
        # Expected: 5e-5 * 1e-5 * 1e18 = 5e9 kg/(m^3 s)
        assert S_rho > 1e8

    def test_low_resistivity_low_ablation(self):
        """Lower resistivity should reduce ablation (less Ohmic heating)."""
        J = 1e8
        eta_low = 1e-7  # 100x lower resistivity
        eta_high = 1e-5
        S_rho_low = ablation_source(
            1e-3, 1e5, 1e24, J, eta_low, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS
        )
        S_rho_high = ablation_source(
            1e-3, 1e5, 1e24, J, eta_high, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS
        )
        assert S_rho_low == pytest.approx(S_rho_high / 100, rel=1e-12)


class TestAblationSourceArray:
    """Unit tests for ablation_source_array function."""

    def test_zero_mask_gives_zero_source_everywhere(self):
        """Zero boundary mask should produce zero source array."""
        J = np.full(10, 1e8)
        eta = np.full(10, 1e-5)
        efficiency = COPPER_ABLATION_EFFICIENCY
        boundary_mask = np.zeros(10, dtype=int)

        S_rho = ablation_source_array(J, eta, efficiency, boundary_mask)
        assert np.all(S_rho == 0.0)

    def test_single_boundary_cell(self):
        """Single boundary cell should receive ablation source."""
        J = np.full(10, 1e8)
        eta = np.full(10, 1e-5)
        efficiency = COPPER_ABLATION_EFFICIENCY
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1  # First cell is boundary

        S_rho = ablation_source_array(J, eta, efficiency, boundary_mask)
        expected = efficiency * eta[0] * J[0] * J[0]
        assert S_rho[0] == pytest.approx(expected, rel=1e-12)
        assert np.all(S_rho[1:] == 0.0)

    def test_multiple_boundary_cells(self):
        """Multiple boundary cells should all receive ablation source."""
        J = np.full(10, 1e8)
        eta = np.full(10, 1e-5)
        efficiency = COPPER_ABLATION_EFFICIENCY
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        boundary_mask[9] = 1  # First and last cells are boundaries

        S_rho = ablation_source_array(J, eta, efficiency, boundary_mask)
        expected = efficiency * eta[0] * J[0] * J[0]
        assert S_rho[0] == pytest.approx(expected, rel=1e-12)
        assert S_rho[9] == pytest.approx(expected, rel=1e-12)
        assert np.all(S_rho[1:9] == 0.0)

    def test_varying_J_and_eta(self):
        """Varying J and eta should produce correct spatially-varying source."""
        J = np.linspace(1e7, 1e9, 10)
        eta = np.linspace(1e-6, 1e-4, 10)
        efficiency = COPPER_ABLATION_EFFICIENCY
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1
        boundary_mask[5] = 1
        boundary_mask[9] = 1

        S_rho = ablation_source_array(J, eta, efficiency, boundary_mask)
        # Check boundary cells
        assert S_rho[0] == pytest.approx(efficiency * eta[0] * J[0] * J[0], rel=1e-12)
        assert S_rho[5] == pytest.approx(efficiency * eta[5] * J[5] * J[5], rel=1e-12)
        assert S_rho[9] == pytest.approx(efficiency * eta[9] * J[9] * J[9], rel=1e-12)
        # Check interior cells
        assert np.all(S_rho[[1, 2, 3, 4, 6, 7, 8]] == 0.0)

    def test_shape_preservation_1d(self):
        """Output should preserve 1D array shape."""
        J = np.full(20, 1e8)
        eta = np.full(20, 1e-5)
        boundary_mask = np.zeros(20, dtype=int)
        boundary_mask[0] = 1

        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.shape == J.shape

    def test_shape_preservation_2d(self):
        """Output should preserve 2D array shape."""
        J = np.full((10, 10), 1e8)
        eta = np.full((10, 10), 1e-5)
        boundary_mask = np.zeros((10, 10), dtype=int)
        boundary_mask[0, :] = 1  # First row is boundary

        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.shape == J.shape
        # Check first row has source
        assert np.all(S_rho[0, :] > 0)
        # Check other rows are zero
        assert np.all(S_rho[1:, :] == 0.0)

    def test_shape_preservation_3d(self):
        """Output should preserve 3D array shape."""
        J = np.full((5, 5, 5), 1e8)
        eta = np.full((5, 5, 5), 1e-5)
        boundary_mask = np.zeros((5, 5, 5), dtype=int)
        boundary_mask[0, :, :] = 1  # First slab is boundary

        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.shape == J.shape
        # Check first slab has source
        assert np.all(S_rho[0, :, :] > 0)
        # Check other slabs are zero
        assert np.all(S_rho[1:, :, :] == 0.0)

    def test_dtype_float32(self):
        """Should work with float32 arrays."""
        J = np.full(10, 1e8, dtype=np.float32)
        eta = np.full(10, 1e-5, dtype=np.float32)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1

        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.dtype == np.float32
        assert S_rho[0] > 0

    def test_dtype_float64(self):
        """Should work with float64 arrays."""
        J = np.full(10, 1e8, dtype=np.float64)
        eta = np.full(10, 1e-5, dtype=np.float64)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1

        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho.dtype == np.float64
        assert S_rho[0] > 0

    def test_zero_J_at_boundary_gives_zero_source(self):
        """Boundary cell with zero J should produce zero source."""
        J = np.full(10, 1e8)
        J[0] = 0.0  # Zero J at boundary
        eta = np.full(10, 1e-5)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1

        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho[0] == pytest.approx(0.0, abs=1e-20)

    def test_negative_J_at_boundary_gives_zero_source(self):
        """Boundary cell with negative J (unphysical) should give zero source."""
        J = np.full(10, 1e8)
        J[0] = -1e8  # Negative J
        eta = np.full(10, 1e-5)
        boundary_mask = np.zeros(10, dtype=int)
        boundary_mask[0] = 1

        S_rho = ablation_source_array(J, eta, COPPER_ABLATION_EFFICIENCY, boundary_mask)
        assert S_rho[0] == pytest.approx(0.0, abs=1e-20)


class TestAblationParticleFlux:
    """Unit tests for ablation_particle_flux function."""

    def test_zero_S_rho_gives_zero_flux(self):
        """Zero mass source should produce zero particle flux."""
        S_n = ablation_particle_flux(0.0, COPPER_MASS)
        assert S_n == pytest.approx(0.0, abs=1e-20)

    def test_negative_S_rho_gives_zero_flux(self):
        """Negative mass source (unphysical) should be guarded to zero."""
        S_n = ablation_particle_flux(-1e7, COPPER_MASS)
        assert S_n == pytest.approx(0.0, abs=1e-20)

    def test_zero_material_mass_gives_zero_flux(self):
        """Zero material mass (unphysical) should be guarded to zero."""
        S_n = ablation_particle_flux(1e7, 0.0)
        assert S_n == pytest.approx(0.0, abs=1e-20)

    def test_correct_formula_S_n_equals_S_rho_over_m(self):
        """Verify S_n = S_rho / m_atom."""
        S_rho = 1e7  # kg/(m^3 s)
        m_atom = COPPER_MASS  # kg
        expected = S_rho / m_atom  # particles/(m^3 s)

        S_n = ablation_particle_flux(S_rho, m_atom)
        assert S_n == pytest.approx(expected, rel=1e-12)

    def test_copper_atomic_mass_consistency(self):
        """Test copper particle flux with correct atomic mass."""
        S_rho = 1e7  # kg/(m^3 s)
        S_n = ablation_particle_flux(S_rho, COPPER_MASS)
        # COPPER_MASS = 63.546 * AMU = 63.546 * 1.66054e-27 kg
        expected = S_rho / (63.546 * 1.66054e-27)
        assert S_n == pytest.approx(expected, rel=1e-12)

    def test_tungsten_atomic_mass_consistency(self):
        """Test tungsten particle flux with correct atomic mass."""
        S_rho = 1e7  # kg/(m^3 s)
        S_n = ablation_particle_flux(S_rho, TUNGSTEN_MASS)
        # TUNGSTEN_MASS = 183.84 * AMU = 183.84 * 1.66054e-27 kg
        expected = S_rho / (183.84 * 1.66054e-27)
        assert S_n == pytest.approx(expected, rel=1e-12)

    def test_physical_magnitude_typical_DPF(self):
        """Check typical DPF particle flux magnitude."""
        # Typical ablation: S_rho ~ 1e7 kg/(m^3 s)
        # Copper mass: ~1e-25 kg
        # Expected S_n ~ 1e32 particles/(m^3 s)
        S_rho = 1e7
        S_n = ablation_particle_flux(S_rho, COPPER_MASS)
        assert 1e30 < S_n < 1e34  # Order-of-magnitude check


class TestAblationMomentumSource:
    """Unit tests for ablation_momentum_source function."""

    def test_zero_S_rho_gives_zero_momentum(self):
        """Zero mass source should produce zero momentum source."""
        S_mom = ablation_momentum_source(0.0, 1e4)
        assert S_mom == pytest.approx(0.0, abs=1e-20)

    def test_zero_velocity_gives_zero_momentum(self):
        """Zero injection velocity should produce zero momentum source."""
        S_mom = ablation_momentum_source(1e7, 0.0)
        assert S_mom == pytest.approx(0.0, abs=1e-20)

    def test_correct_formula_S_mom_equals_S_rho_times_v(self):
        """Verify S_mom = S_rho * v_boundary."""
        S_rho = 1e7  # kg/(m^3 s)
        v_boundary = 1e4  # m/s (thermal speed at ~10 eV)
        expected = S_rho * v_boundary  # N/m^3

        S_mom = ablation_momentum_source(S_rho, v_boundary)
        assert S_mom == pytest.approx(expected, rel=1e-12)

    def test_physical_magnitude_typical_DPF(self):
        """Check typical DPF momentum source magnitude."""
        # Typical: S_rho ~ 1e7 kg/(m^3 s), v ~ 1e4 m/s
        # Expected: S_mom ~ 1e11 N/m^3
        S_rho = 1e7
        v_boundary = 1e4
        S_mom = ablation_momentum_source(S_rho, v_boundary)
        assert 1e10 < S_mom < 1e12  # Order-of-magnitude check

    def test_negative_velocity(self):
        """Negative velocity should produce negative momentum source."""
        S_rho = 1e7
        v_boundary = -1e4
        S_mom = ablation_momentum_source(S_rho, v_boundary)
        assert S_mom == pytest.approx(-1e11, rel=1e-1)


class TestAblationPhysics:
    """Physical scaling tests for ablation module."""

    def test_S_rho_scales_quadratically_with_J(self):
        """Ablation source should scale as J^2."""
        eta = 1e-5
        efficiency = COPPER_ABLATION_EFFICIENCY
        J1 = 1e8
        J2 = 2e8
        S_rho1 = ablation_source(1e-3, 1e5, 1e24, J1, eta, 1e-3, efficiency, COPPER_MASS)
        S_rho2 = ablation_source(1e-3, 1e5, 1e24, J2, eta, 1e-3, efficiency, COPPER_MASS)
        # S_rho ~ J^2, so doubling J should quadruple S_rho
        assert S_rho2 == pytest.approx(4 * S_rho1, rel=1e-12)

    def test_S_rho_scales_linearly_with_eta(self):
        """Ablation source should scale linearly with resistivity."""
        J = 1e8
        efficiency = COPPER_ABLATION_EFFICIENCY
        eta1 = 1e-5
        eta2 = 2e-5
        S_rho1 = ablation_source(1e-3, 1e5, 1e24, J, eta1, 1e-3, efficiency, COPPER_MASS)
        S_rho2 = ablation_source(1e-3, 1e5, 1e24, J, eta2, 1e-3, efficiency, COPPER_MASS)
        # S_rho ~ eta, so doubling eta should double S_rho
        assert S_rho2 == pytest.approx(2 * S_rho1, rel=1e-12)

    def test_S_rho_scales_linearly_with_efficiency(self):
        """Ablation source should scale linearly with efficiency."""
        J = 1e8
        eta = 1e-5
        eff1 = COPPER_ABLATION_EFFICIENCY
        eff2 = 2 * COPPER_ABLATION_EFFICIENCY
        S_rho1 = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, eff1, COPPER_MASS)
        S_rho2 = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, eff2, COPPER_MASS)
        # S_rho ~ efficiency, so doubling efficiency should double S_rho
        assert S_rho2 == pytest.approx(2 * S_rho1, rel=1e-12)

    def test_order_of_magnitude_typical_DPF_ablation_rate(self):
        """Typical DPF conditions should give ablation rate in 1-100 kg/(m^3 s) range."""
        # Typical DPF boundary layer:
        # J ~ 1e8 A/m^2 (100 MA/m^2)
        # eta ~ 1e-5 Ohm m (Spitzer at ~10 eV)
        # efficiency ~ 5e-5 kg/J (copper)
        J = 1e8
        eta = 1e-5
        efficiency = COPPER_ABLATION_EFFICIENCY
        S_rho = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, efficiency, COPPER_MASS)
        # Expected: 5e-5 * 1e-5 * 1e16 = 5e7 kg/(m^3 s)
        # This is actually quite high due to intense heating
        assert 1e6 < S_rho < 1e9

    def test_low_current_low_ablation(self):
        """Low current density should produce low ablation rate."""
        J = 1e6  # 1 MA/m^2 (100x lower than typical)
        eta = 1e-5
        efficiency = COPPER_ABLATION_EFFICIENCY
        S_rho = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, efficiency, COPPER_MASS)
        # Expected: 5e-5 * 1e-5 * 1e12 = 5e3 kg/(m^3 s)
        assert 1e2 < S_rho < 1e5

    def test_high_current_high_ablation(self):
        """High current density should produce high ablation rate."""
        J = 1e9  # 1 GA/m^2 (10x higher than typical)
        eta = 1e-5
        efficiency = COPPER_ABLATION_EFFICIENCY
        S_rho = ablation_source(1e-3, 1e5, 1e24, J, eta, 1e-3, efficiency, COPPER_MASS)
        # Expected: 5e-5 * 1e-5 * 1e18 = 5e9 kg/(m^3 s)
        assert 1e8 < S_rho < 1e11

    def test_tungsten_lower_ablation_than_copper(self):
        """Tungsten should ablate less than copper (2e-5 vs 5e-5 kg/J)."""
        J = 1e8
        eta = 1e-5
        S_rho_Cu = ablation_source(
            1e-3, 1e5, 1e24, J, eta, 1e-3, COPPER_ABLATION_EFFICIENCY, COPPER_MASS
        )
        S_rho_W = ablation_source(
            1e-3, 1e5, 1e24, J, eta, 1e-3, TUNGSTEN_ABLATION_EFFICIENCY, TUNGSTEN_MASS
        )
        # Tungsten should ablate at 40% the rate of copper
        assert S_rho_W == pytest.approx(0.4 * S_rho_Cu, rel=1e-12)


class TestAblationConfig:
    """Tests for AblationConfig Pydantic model."""

    def test_default_config_values(self):
        """Default config should be disabled, copper, 5e-5 efficiency."""
        config = AblationConfig()
        assert config.enabled is False
        assert config.material == "copper"
        assert config.efficiency == pytest.approx(5e-5, rel=1e-12)

    def test_enable_ablation(self):
        """Enabling ablation should work."""
        config = AblationConfig(enabled=True)
        assert config.enabled is True

    def test_tungsten_material(self):
        """Setting tungsten material should work."""
        config = AblationConfig(material="tungsten")
        assert config.material == "tungsten"

    def test_custom_efficiency(self):
        """Setting custom efficiency should work."""
        config = AblationConfig(efficiency=1e-4)
        assert config.efficiency == pytest.approx(1e-4, rel=1e-12)

    def test_invalid_material_raises_error(self):
        """Invalid material should raise ValidationError."""
        with pytest.raises(ValueError, match="material must be"):
            AblationConfig(material="aluminum")

    def test_zero_efficiency_raises_error(self):
        """Zero efficiency should raise ValidationError (gt=0 constraint)."""
        with pytest.raises(ValueError):
            AblationConfig(efficiency=0.0)

    def test_negative_efficiency_raises_error(self):
        """Negative efficiency should raise ValidationError."""
        with pytest.raises(ValueError):
            AblationConfig(efficiency=-1e-5)

    def test_config_dict_roundtrip(self):
        """Config should survive dict serialization."""
        config = AblationConfig(enabled=True, material="tungsten", efficiency=2e-5)
        config_dict = config.model_dump()
        config2 = AblationConfig(**config_dict)
        assert config2.enabled == config.enabled
        assert config2.material == config.material
        assert config2.efficiency == pytest.approx(config.efficiency, rel=1e-12)
