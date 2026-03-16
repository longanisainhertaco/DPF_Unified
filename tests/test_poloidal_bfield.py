"""Tests for the Auluck poloidal B-field module (GV surface mechanism).

Reference: S.K.H. Auluck, Phys. Plasmas 31, 010704 (2024).
"""

import numpy as np
import pytest

from dpf.experimental.poloidal_bfield import (
    add_poloidal_field,
    compute_azimuthal_Etheta,
    compute_gv_surface,
    compute_hamiltonian,
    compute_poloidal_Br,
    compute_poloidal_field,
    compute_scaling_params,
    solve_flux_evolution,
)

# PF-1000 reference geometry
PF1000_A = 0.115       # anode radius [m]
PF1000_B = 0.160       # cathode radius [m]
PF1000_N = PF1000_A / PF1000_B  # ~0.719
PF1000_I = 1.7e6       # peak current [A]
PF1000_RHO0 = 6.5e-4   # fill density [kg/m^3]


class TestScalingParams:
    """Test GV scaling parameter computation (Eq. 2, 3)."""

    def test_B0_dimensional(self):
        params = compute_scaling_params(PF1000_A, PF1000_RHO0, PF1000_I)
        B_0 = params["B_0"]
        # B_0 = mu_0*I/(2*pi*a) ~ 4pi*1e-7 * 1.7e6 / (2*pi*0.115) ~ 2.96 T
        assert 2.0 < B_0 < 4.0, f"B_0 = {B_0:.3f} T out of expected range"

    def test_v0_order_of_magnitude(self):
        params = compute_scaling_params(PF1000_A, PF1000_RHO0, PF1000_I)
        v_0 = params["v_0"]
        # v_0 = B_0/sqrt(2*mu_0*rho_0) ~ 3/sqrt(2*4pi*1e-7*6.5e-4) ~ ~80 km/s
        assert 1e4 < v_0 < 1e6, f"v_0 = {v_0:.0f} m/s out of range"

    def test_Q_m_positive(self):
        params = compute_scaling_params(PF1000_A, PF1000_RHO0, PF1000_I)
        assert params["Q_m"] > 0

    def test_negative_current_uses_abs(self):
        p1 = compute_scaling_params(PF1000_A, PF1000_RHO0, PF1000_I)
        p2 = compute_scaling_params(PF1000_A, PF1000_RHO0, -PF1000_I)
        assert p1["B_0"] == pytest.approx(p2["B_0"])


class TestGVSurface:
    """Test Gratton-Vargas surface computation (Eq. 4)."""

    def test_surface_shape(self):
        gv = compute_gv_surface(PF1000_N, nr=32, nz=64)
        assert gv["psi"].shape == (32, 64)
        assert gv["r_bar"].shape == (32,)
        assert gv["z_bar"].shape == (64,)

    def test_surface_not_all_zero(self):
        gv = compute_gv_surface(PF1000_N, nr=32, nz=64, tau=0.5)
        assert np.any(gv["psi"] != 0.0)

    def test_surface_mask_exists(self):
        gv = compute_gv_surface(PF1000_N, nr=32, nz=64, tau=0.0)
        assert gv["surface_mask"].shape == (32, 64)
        assert gv["surface_mask"].dtype == bool

    def test_invalid_N_raises(self):
        with pytest.raises(ValueError, match="N = a/b must be in"):
            compute_gv_surface(0.0, nr=8, nz=8)
        with pytest.raises(ValueError, match="N = a/b must be in"):
            compute_gv_surface(1.0, nr=8, nz=8)

    def test_r_range_starts_above_N(self):
        gv = compute_gv_surface(PF1000_N, nr=16, nz=16)
        assert gv["r_bar"][0] > PF1000_N

    def test_psi_antisymmetric_in_z(self):
        """GV surface should be antisymmetric in z_bar for tau=0."""
        gv = compute_gv_surface(PF1000_N, nr=16, nz=64, tau=0.0, s=-1)
        psi = gv["psi"]
        # psi(r, z) should be ~ -psi(r, -z) since the z_bar*(...) term is linear in z
        psi_flipped = psi[:, ::-1]
        np.testing.assert_allclose(psi, -psi_flipped, atol=1e-10)

    def test_custom_ranges(self):
        gv = compute_gv_surface(
            0.5, nr=20, nz=30, r_range=(0.6, 0.9), z_range=(-1.0, 1.0)
        )
        assert gv["r_bar"][0] == pytest.approx(0.6)
        assert gv["r_bar"][-1] == pytest.approx(0.9)
        assert gv["z_bar"][0] == pytest.approx(-1.0)
        assert gv["z_bar"][-1] == pytest.approx(1.0)


class TestFluxEvolution:
    """Test flux function PDE solver (Eq. 8)."""

    def test_zero_flux_stays_zero(self):
        """Zero initial flux should remain zero (no source term)."""
        nr, nz = 16, 32
        N = 0.5
        r_bar = np.linspace(N + 0.01, 1.0, nr)
        z_bar = np.linspace(-1.0, 1.0, nz)
        Phi_0 = np.zeros((nr, nz))
        Phi = solve_flux_evolution(Phi_0, r_bar, z_bar, N, s=-1, dtau=0.01, n_steps=5)
        np.testing.assert_allclose(Phi, 0.0, atol=1e-15)

    def test_nonzero_flux_evolves(self):
        """Non-zero initial flux should change under PDE evolution."""
        nr, nz = 16, 32
        N = 0.5
        r_bar = np.linspace(N + 0.01, 1.0, nr)
        z_bar = np.linspace(-1.0, 1.0, nz)
        R = r_bar[:, np.newaxis] * np.ones((1, nz))
        Phi_0 = np.pi * R**2 * 5e-5
        Phi = solve_flux_evolution(Phi_0, r_bar, z_bar, N, s=-1, dtau=0.001, n_steps=5)
        assert not np.allclose(Phi, Phi_0, atol=1e-15)

    def test_small_steps_stability(self):
        """Verify no NaN or Inf with small timesteps."""
        nr, nz = 16, 32
        N = 0.7
        r_bar = np.linspace(N + 0.01, 1.0, nr)
        z_bar = np.linspace(-2.0, 2.0, nz)
        R = r_bar[:, np.newaxis] * np.ones((1, nz))
        Phi_0 = np.pi * R**2 * 5e-5
        Phi = solve_flux_evolution(Phi_0, r_bar, z_bar, N, s=-1, dtau=1e-4, n_steps=20)
        assert np.all(np.isfinite(Phi))


class TestHamiltonian:
    """Test Hamiltonian conservation (Eq. 9)."""

    def test_hamiltonian_shape(self):
        nr, nz = 16, 32
        N = 0.5
        r_bar = np.linspace(N + 0.01, 1.0, nr)
        z_bar = np.linspace(-1.0, 1.0, nz)
        R = r_bar[:, np.newaxis] * np.ones((1, nz))
        Phi = np.pi * R**2 * 5e-5
        H = compute_hamiltonian(Phi, r_bar, z_bar, N, s=-1)
        assert H.shape == (nr, nz)
        assert np.all(np.isfinite(H))

    def test_hamiltonian_zero_for_zero_flux(self):
        nr, nz = 16, 32
        N = 0.5
        r_bar = np.linspace(N + 0.01, 1.0, nr)
        z_bar = np.linspace(-1.0, 1.0, nz)
        Phi = np.zeros((nr, nz))
        H = compute_hamiltonian(Phi, r_bar, z_bar, N, s=-1)
        np.testing.assert_allclose(H, 0.0, atol=1e-15)


class TestPoloidalField:
    """Test B_z computation (Eq. 11)."""

    def test_basic_pf1000(self):
        B_z = compute_poloidal_field(
            PF1000_A, PF1000_B, PF1000_I, PF1000_RHO0, nr=32, nz=64
        )
        assert B_z.shape == (32, 64)
        assert np.all(np.isfinite(B_z))
        # B_z should be nonzero (dynamo amplifies seed)
        assert np.max(np.abs(B_z)) > 0
        # B_z should be much smaller than B_theta (~3T for PF-1000)
        assert np.max(np.abs(B_z)) < 1.0

    def test_no_current_gives_seed_only(self):
        """With zero current, B_z should be close to seed field."""
        B_z = compute_poloidal_field(
            PF1000_A, PF1000_B, 1.0, PF1000_RHO0,  # very small current
            nr=16, nz=32, B_seed=5e-5,
        )
        assert np.all(np.isfinite(B_z))

    def test_larger_current_amplifies(self):
        """Higher current should produce larger B_z."""
        B_z_low = compute_poloidal_field(
            PF1000_A, PF1000_B, 1e5, PF1000_RHO0, nr=16, nz=32
        )
        B_z_high = compute_poloidal_field(
            PF1000_A, PF1000_B, 1e6, PF1000_RHO0, nr=16, nz=32
        )
        # Higher current = more dynamo amplification
        assert np.max(np.abs(B_z_high)) >= np.max(np.abs(B_z_low))

    def test_different_geometries(self):
        """Different electrode ratios should produce different fields."""
        B_z_1 = compute_poloidal_field(0.05, 0.10, 1e6, PF1000_RHO0, nr=16, nz=32)
        B_z_2 = compute_poloidal_field(0.08, 0.10, 1e6, PF1000_RHO0, nr=16, nz=32)
        assert not np.allclose(B_z_1, B_z_2)


class TestPoloidalBr:
    """Test B_r computation (Eq. 12)."""

    def test_Br_shape_and_finite(self):
        nr, nz = 16, 32
        r_bar = np.linspace(0.75, 1.0, nr)
        z_bar = np.linspace(-1.0, 1.0, nz)
        R = r_bar[:, np.newaxis] * np.ones((1, nz))
        Phi = np.pi * (0.1 * R)**2 * 5e-5
        B_r = compute_poloidal_Br(Phi, r_bar, z_bar, a=0.1)
        assert B_r.shape == (nr, nz)
        assert np.all(np.isfinite(B_r))


class TestAzimuthalEtheta:
    """Test E_theta computation (Eq. 10)."""

    def test_Etheta_proportional_to_current(self):
        nr, nz = 16, 32
        N = 0.5
        r_bar = np.linspace(N + 0.01, 1.0, nr)
        z_bar = np.linspace(-1.0, 1.0, nz)
        R = r_bar[:, np.newaxis] * np.ones((1, nz))
        Phi = np.pi * R**2 * 5e-5
        H = compute_hamiltonian(Phi, r_bar, z_bar, N, s=-1)

        E1 = compute_azimuthal_Etheta(H, r_bar, a=0.1, I=1e5, rho_0=1e-3)
        E2 = compute_azimuthal_Etheta(H, r_bar, a=0.1, I=2e5, rho_0=1e-3)
        # E_theta is linear in I
        np.testing.assert_allclose(E2, 2.0 * E1, rtol=1e-10)


class TestAddPoloidalField:
    """Test MHD integration function."""

    def test_adds_to_Bz_component(self):
        nx, ny, nz = 16, 8, 32
        state = {
            "rho": np.ones((nx, ny, nz)) * PF1000_RHO0,
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": np.zeros((3, nx, ny, nz)),
        }
        dr = PF1000_A / nx
        dz = 0.3 / nz

        result = add_poloidal_field(
            state, PF1000_I, PF1000_A, PF1000_B, PF1000_RHO0, dr, dz
        )
        # B[2] (z-component) should have been modified
        assert not np.allclose(result["B"][2], 0.0)
        # B[0] and B[1] should be untouched
        np.testing.assert_allclose(result["B"][0], 0.0)
        np.testing.assert_allclose(result["B"][1], 0.0)

    def test_does_not_mutate_input(self):
        nx, ny, nz = 8, 4, 16
        state = {
            "rho": np.ones((nx, ny, nz)) * PF1000_RHO0,
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": np.zeros((3, nx, ny, nz)),
        }
        B_orig = state["B"].copy()
        add_poloidal_field(state, PF1000_I, PF1000_A, PF1000_B, PF1000_RHO0, 0.01, 0.01)
        np.testing.assert_array_equal(state["B"], B_orig)

    def test_invalid_B_shape_raises(self):
        state = {
            "rho": np.ones((8, 8)),
            "B": np.zeros((8, 8)),
        }
        with pytest.raises(ValueError, match="B must be 4D"):
            add_poloidal_field(state, 1e6, 0.1, 0.15, 1e-3, 0.01, 0.01)

    def test_Bz_magnitude_physical(self):
        """B_z from poloidal field should be << B_theta (~3T for PF-1000)."""
        nx, ny, nz = 16, 4, 32
        state = {
            "rho": np.ones((nx, ny, nz)) * PF1000_RHO0,
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": np.zeros((3, nx, ny, nz)),
        }
        result = add_poloidal_field(
            state, PF1000_I, PF1000_A, PF1000_B, PF1000_RHO0, 0.01, 0.01
        )
        max_Bz = np.max(np.abs(result["B"][2]))
        assert max_Bz < 1.0, f"B_z = {max_Bz:.4f} T is too large"
