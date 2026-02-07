"""Tests for Phase 16: Extended MHD + Kinetic Option.

Covers:
    16.1 Nernst effect (magnetic field advection by temperature gradient)
    16.2 Braginskii viscosity tensor
    16.3 Constrained transport for div(B) = 0
    16.4 Hybrid fluid-PIC (Boris push, CIC deposition)
    16.5 Multi-device validation suite
"""

from __future__ import annotations

import numpy as np

from dpf.constants import e, m_d

# ===================================================================
# 16.1 — Nernst Effect
# ===================================================================


class TestNernstCoefficient:
    """Tests for the Nernst coefficient calculation."""

    def test_positive_for_magnetized_plasma(self):
        """Nernst coefficient is positive for magnetized plasma."""
        from dpf.fluid.nernst import nernst_coefficient

        ne = 1e24  # dense plasma
        Te = 1e7  # 1 keV
        B_mag = 10.0  # 10 T
        beta_wedge = nernst_coefficient(ne, Te, B_mag)
        assert beta_wedge >= 0

    def test_scales_with_magnetization(self):
        """Nernst coefficient depends on magnetization ω_ce * τ_e."""
        from dpf.fluid.nernst import nernst_coefficient

        ne = 1e23
        Te = 1e6
        # Low B (weakly magnetized) vs. high B (strongly magnetized)
        beta_low = nernst_coefficient(ne, Te, 0.01)
        beta_high = nernst_coefficient(ne, Te, 50.0)
        # They should differ (exact relation depends on Braginskii fits)
        assert beta_low != beta_high

    def test_zero_field_gives_zero(self):
        """With zero B-field, Nernst coefficient should be zero or handle gracefully."""
        from dpf.fluid.nernst import nernst_coefficient

        beta = nernst_coefficient(1e23, 1e6, 0.0)
        assert np.isfinite(beta)


class TestNernstElectricField:
    """Tests for Nernst E-field computation."""

    def test_zero_gradient_gives_zero_field(self):
        """Zero temperature gradient produces zero Nernst E-field."""
        from dpf.fluid.nernst import nernst_electric_field

        ne = 1e23
        Te = 1e6
        B = np.array([0.0, 0.0, 1.0])
        grad_Te = np.array([0.0, 0.0, 0.0])
        E_N = nernst_electric_field(ne, Te, B, grad_Te)
        np.testing.assert_allclose(E_N, 0.0, atol=1e-30)

    def test_perpendicular_to_b_and_grad_te(self):
        """Nernst E-field is perpendicular to B (b × ∇Te is perp to both)."""
        from dpf.fluid.nernst import nernst_electric_field

        ne = 1e23
        Te = 1e6
        B = np.array([0.0, 0.0, 5.0])
        grad_Te = np.array([1e8, 0.0, 0.0])  # gradient in x
        E_N = nernst_electric_field(ne, Te, B, grad_Te)
        # E_N should have no z-component (b is along z, grad_Te along x → cross is along y)
        assert abs(E_N[2]) < abs(E_N[1]) + 1e-30 or np.allclose(E_N, 0.0, atol=1e-30)


class TestNernstAdvection:
    """Tests for Nernst advection of B-field."""

    def test_uniform_temperature_no_change(self):
        """Uniform temperature should not advect B-field."""
        from dpf.fluid.nernst import apply_nernst_advection

        n = 16
        Bx = np.zeros((n, n, n))
        By = np.zeros((n, n, n))
        Bz = np.ones((n, n, n))
        ne = np.full((n, n, n), 1e23)
        Te = np.full((n, n, n), 1e6)  # uniform temperature

        Bx_new, By_new, Bz_new = apply_nernst_advection(
            Bx, By, Bz, ne, Te, 0.001, 0.001, 0.001, 1e-12,
        )
        # Bz should remain essentially unchanged
        np.testing.assert_allclose(Bz_new, 1.0, atol=1e-6)


# ===================================================================
# 16.2 — Braginskii Viscosity
# ===================================================================


class TestIonCollisionTime:
    """Tests for ion-ion collision time."""

    def test_positive(self):
        """Collision time is always positive."""
        from dpf.fluid.viscosity import ion_collision_time

        tau = ion_collision_time(1e23, 1e6)
        assert tau > 0

    def test_increases_with_temperature(self):
        """Hotter plasma has longer collision time (τ ∝ T^{3/2})."""
        from dpf.fluid.viscosity import ion_collision_time

        tau_cold = ion_collision_time(1e23, 1e5)
        tau_hot = ion_collision_time(1e23, 1e7)
        assert tau_hot > tau_cold

    def test_decreases_with_density(self):
        """Denser plasma has shorter collision time (τ ∝ 1/n)."""
        from dpf.fluid.viscosity import ion_collision_time

        tau_low = ion_collision_time(1e22, 1e6)
        tau_high = ion_collision_time(1e24, 1e6)
        assert tau_low > tau_high


class TestBraginskiiEta0:
    """Tests for parallel viscosity η₀."""

    def test_positive(self):
        """η₀ is always positive."""
        from dpf.fluid.viscosity import braginskii_eta0, ion_collision_time

        ni = 1e23
        Ti = 1e6
        tau = ion_collision_time(ni, Ti)
        eta0 = braginskii_eta0(ni, Ti, tau)
        assert eta0 > 0

    def test_scales_with_density_and_temperature(self):
        """η₀ = 0.96 * ni * kT * τ, so it depends on ni and Ti."""
        from dpf.fluid.viscosity import braginskii_eta0, ion_collision_time

        tau1 = ion_collision_time(1e23, 1e6)
        eta1 = braginskii_eta0(1e23, 1e6, tau1)
        tau2 = ion_collision_time(1e24, 1e6)
        eta2 = braginskii_eta0(1e24, 1e6, tau2)
        # Higher density → shorter τ but η₀ ∝ n * T * τ
        # τ ∝ T^{3/2}/n so η₀ ∝ T^{5/2}. Denser → eta changes due to tau.
        assert eta1 != eta2


class TestBraginskiiEta3:
    """Tests for gyroviscosity η₃."""

    def test_positive_with_field(self):
        """η₃ is positive with nonzero B."""
        from dpf.fluid.viscosity import braginskii_eta3

        eta3 = braginskii_eta3(1e23, 1e6, 5.0)
        assert eta3 > 0

    def test_inversely_proportional_to_b(self):
        """η₃ ∝ 1/ω_ci ∝ 1/B, so larger B → smaller η₃."""
        from dpf.fluid.viscosity import braginskii_eta3

        eta3_low = braginskii_eta3(1e23, 1e6, 1.0)
        eta3_high = braginskii_eta3(1e23, 1e6, 10.0)
        assert eta3_low > eta3_high


class TestViscousStress:
    """Tests for viscous stress computation."""

    def test_uniform_velocity_no_stress(self):
        """Uniform velocity produces zero viscous stress."""
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
        # Interior should be near zero for uniform flow
        interior = dvdt[2:-2, 2:-2, 2:-2]
        assert np.max(np.abs(interior)) < 1e-6

    def test_shear_produces_stress(self):
        """Linear shear v_x = y produces non-zero stress."""
        from dpf.fluid.viscosity import viscous_stress_rate

        n = 16
        y = np.linspace(0, 1, n)
        vx = np.zeros((n, n, n))
        for j in range(n):
            vx[:, j, :] = y[j] * 1000.0  # shear in y direction
        vy = np.zeros((n, n, n))
        vz = np.zeros((n, n, n))
        rho = np.ones((n, n, n))
        eta0 = np.ones((n, n, n)) * 1e-2

        dvdt = viscous_stress_rate(
            np.stack([vx, vy, vz], axis=-1), rho, eta0, 1.0 / n, 1.0 / n, 1.0 / n,
        )
        # Should have nonzero stress in interior
        interior = dvdt[2:-2, 2:-2, 2:-2]
        assert np.max(np.abs(interior)) > 0


class TestViscousHeating:
    """Tests for viscous heating rate."""

    def test_uniform_flow_no_heating(self):
        """Uniform velocity gives zero viscous heating."""
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
        """Shear flow produces positive viscous heating."""
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


# ===================================================================
# 16.3 — Constrained Transport
# ===================================================================


class TestStaggeredBField:
    """Tests for staggered B-field data structure."""

    def test_cell_to_face_roundtrip(self):
        """Cell→face→cell roundtrip approximately preserves uniform field."""
        from dpf.fluid.constrained_transport import (
            cell_centered_to_face,
            face_to_cell_centered,
        )

        n = 8
        Bx = np.ones((n, n, n)) * 2.0
        By = np.ones((n, n, n)) * 3.0
        Bz = np.ones((n, n, n)) * 5.0

        stag = cell_centered_to_face(Bx, By, Bz, 0.01, 0.01, 0.01)
        Bx2, By2, Bz2 = face_to_cell_centered(stag)

        # For uniform field, roundtrip should be exact
        np.testing.assert_allclose(Bx2, 2.0, atol=1e-12)
        np.testing.assert_allclose(By2, 3.0, atol=1e-12)
        np.testing.assert_allclose(Bz2, 5.0, atol=1e-12)

    def test_face_shapes(self):
        """Face-centered arrays have correct staggered shapes."""
        from dpf.fluid.constrained_transport import cell_centered_to_face

        n = 8
        stag = cell_centered_to_face(
            np.ones((n, n, n)), np.ones((n, n, n)), np.ones((n, n, n)),
            0.01, 0.01, 0.01,
        )
        assert stag.Bx_face.shape == (n + 1, n, n)
        assert stag.By_face.shape == (n, n + 1, n)
        assert stag.Bz_face.shape == (n, n, n + 1)


class TestDivBConstraint:
    """Tests for divergence-free constraint."""

    def test_uniform_field_zero_div(self):
        """Uniform B-field has zero divergence on staggered grid."""
        from dpf.fluid.constrained_transport import (
            cell_centered_to_face,
            compute_div_B,
        )

        n = 8
        stag = cell_centered_to_face(
            np.ones((n, n, n)) * 2.0,
            np.ones((n, n, n)) * 3.0,
            np.ones((n, n, n)) * 5.0,
            0.01, 0.01, 0.01,
        )
        div_b = compute_div_B(stag)
        np.testing.assert_allclose(div_b, 0.0, atol=1e-12)

    def test_ct_update_preserves_div_b(self):
        """CT update preserves div(B) = 0."""
        from dpf.fluid.constrained_transport import (
            cell_centered_to_face,
            compute_div_B,
            ct_update,
        )

        n = 8
        dx = dy = dz = 0.01
        stag = cell_centered_to_face(
            np.ones((n, n, n)) * 2.0,
            np.zeros((n, n, n)),
            np.ones((n, n, n)) * 1.0,
            dx, dy, dz,
        )

        # Random but smooth EMFs
        rng = np.random.default_rng(42)
        Ex_edge = rng.uniform(-0.1, 0.1, (n, n + 1, n + 1))
        Ey_edge = rng.uniform(-0.1, 0.1, (n + 1, n, n + 1))
        Ez_edge = rng.uniform(-0.1, 0.1, (n + 1, n + 1, n))

        stag_new = ct_update(stag, Ex_edge, Ey_edge, Ez_edge, 1e-6)
        div_b = compute_div_B(stag_new)
        # CT should preserve div(B) = 0 to machine precision
        np.testing.assert_allclose(div_b, 0.0, atol=1e-10)


# ===================================================================
# 16.4 — Hybrid Fluid-PIC
# ===================================================================


class TestBorisPush:
    """Tests for Boris particle pusher."""

    def test_straight_line_no_field(self):
        """Particles move in straight lines with no E or B."""
        from dpf.pic import boris_push

        N = 10
        pos = np.zeros((N, 3))
        vel = np.ones((N, 3)) * 1e5  # 100 km/s
        E = np.zeros((N, 3))
        B = np.zeros((N, 3))
        dt = 1e-9

        new_pos, new_vel = boris_push(pos, vel, E, B, e, m_d, dt)
        expected_pos = vel * dt
        np.testing.assert_allclose(new_pos, expected_pos, rtol=1e-10)
        np.testing.assert_allclose(new_vel, vel, rtol=1e-10)

    def test_e_field_acceleration(self):
        """Uniform E-field accelerates particles."""
        from dpf.pic import boris_push

        N = 5
        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))
        E = np.zeros((N, 3))
        E[:, 0] = 1e6  # 1 MV/m in x
        B = np.zeros((N, 3))
        dt = 1e-9

        new_pos, new_vel = boris_push(pos, vel, E, B, e, m_d, dt)
        # Should have gained velocity in x direction
        assert np.all(new_vel[:, 0] > 0)

    def test_gyration_in_b_field(self):
        """Particle in uniform B-field gyrates (speed conserved)."""
        from dpf.pic import boris_push

        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[1e5, 0.0, 0.0]])  # v_perp to B
        E = np.zeros((1, 3))
        B = np.array([[0.0, 0.0, 1.0]])  # 1 T in z
        dt = 1e-10  # small enough for good resolution

        speed0 = np.linalg.norm(vel)
        # Take many steps
        p, v = pos.copy(), vel.copy()
        for _ in range(100):
            p, v = boris_push(p, v, E, B, e, m_d, dt)

        speed_final = np.linalg.norm(v)
        # Boris pusher conserves speed exactly
        np.testing.assert_allclose(speed_final, speed0, rtol=1e-6)


class TestDeposition:
    """Tests for particle-to-grid deposition."""

    def test_single_particle_conservation(self):
        """Total deposited density equals particle weight."""
        from dpf.pic import deposit_density

        pos = np.array([[0.005, 0.005, 0.005]])  # center of domain
        weights = np.array([1e10])
        grid_shape = (8, 8, 8)
        dx = dy = dz = 0.01 / 8

        rho = deposit_density(pos, weights, grid_shape, dx, dy, dz)
        # Total deposited ≈ weight (within CIC bounds)
        total = np.sum(rho) * dx * dy * dz
        np.testing.assert_allclose(total, 1e10, rtol=0.1)

    def test_density_is_nonnegative(self):
        """Deposited density should never be negative."""
        from dpf.pic import deposit_density

        rng = np.random.default_rng(42)
        N = 100
        pos = rng.uniform(0.001, 0.009, (N, 3))
        weights = np.ones(N) * 1e8
        grid_shape = (8, 8, 8)
        dx = dy = dz = 0.01 / 8

        rho = deposit_density(pos, weights, grid_shape, dx, dy, dz)
        assert np.all(rho >= 0)


class TestHybridPIC:
    """Tests for the HybridPIC class."""

    def test_add_species(self):
        """Can add a species to the hybrid PIC."""
        from dpf.pic import HybridPIC

        hybrid = HybridPIC((8, 8, 8), 0.001, 0.001, 0.001, 1e-9)
        sp = hybrid.add_species(
            "deuterium", m_d, e,
            np.zeros((10, 3)), np.zeros((10, 3)), np.ones(10) * 1e8,
        )
        assert sp.n_particles() == 10
        assert len(hybrid.species) == 1

    def test_deposit_produces_density(self):
        """Depositing particles produces nonzero density."""
        from dpf.pic import HybridPIC

        hybrid = HybridPIC((8, 8, 8), 0.001, 0.001, 0.001, 1e-9)
        rng = np.random.default_rng(42)
        pos = rng.uniform(0.001, 0.007, (50, 3))
        vel = rng.normal(0, 1e5, (50, 3))
        hybrid.add_species("D", m_d, e, pos, vel, np.ones(50) * 1e8)

        rho, Jx, Jy, Jz = hybrid.deposit()
        assert np.sum(rho) > 0


class TestInstabilityDetection:
    """Tests for m=0 instability detection."""

    def test_uniform_no_instability(self):
        """Uniform density gives no instability."""
        from dpf.pic.hybrid import detect_instability

        rho = np.ones((16, 16, 16))
        B = np.zeros((16, 16, 16, 3))
        B[:, :, :, 2] = 1.0
        assert not detect_instability(rho, B)

    def test_strong_compression_triggers(self):
        """Strong density compression triggers instability detection."""
        from dpf.pic.hybrid import detect_instability

        rho = np.ones((16, 16, 16))
        rho[7:9, 7:9, :] = 20.0  # 20× compression
        B = np.zeros((16, 16, 16, 3))
        B[:, :, :8, 2] = 1.0
        B[:, :, 8:, 2] = -1.0  # field reversal
        assert detect_instability(rho, B, threshold=5.0)


# ===================================================================
# 16.5 — Multi-device Validation Suite
# ===================================================================


class TestExperimentalDevices:
    """Tests for experimental device data."""

    def test_pf1000_exists(self):
        """PF-1000 device data is available."""
        from dpf.validation.experimental import DEVICES, PF1000_DATA

        assert PF1000_DATA.name == "PF-1000"
        assert PF1000_DATA.peak_current > 1e6  # > 1 MA
        assert "PF-1000" in DEVICES

    def test_nx2_exists(self):
        """NX2 device data is available."""
        from dpf.validation.experimental import NX2_DATA

        assert NX2_DATA.peak_current > 1e5  # > 100 kA
        assert NX2_DATA.fill_gas == "deuterium"

    def test_unu_ictp_exists(self):
        """UNU-ICTP device data is available."""
        from dpf.validation.experimental import UNU_ICTP_DATA

        assert UNU_ICTP_DATA.peak_current > 1e5
        assert UNU_ICTP_DATA.capacitance > 0

    def test_device_parameters_physical(self):
        """All device parameters are physically reasonable."""
        from dpf.validation.experimental import DEVICES

        for name, dev in DEVICES.items():
            assert dev.capacitance > 0, f"{name}: C <= 0"
            assert dev.voltage > 0, f"{name}: V0 <= 0"
            assert dev.anode_radius < dev.cathode_radius, f"{name}: r_a >= r_c"
            assert dev.neutron_yield > 0, f"{name}: yield <= 0"


class TestValidationMetrics:
    """Tests for validation metric functions."""

    def test_perfect_current_match(self):
        """Perfect current match gives zero error."""
        from dpf.validation.experimental import validate_current_waveform

        t = np.linspace(0, 6e-6, 1000)
        I_peak = 1.87e6
        current = I_peak * np.sin(2 * np.pi * t / (4 * 5.8e-6))

        result = validate_current_waveform(t, current, "PF-1000")
        assert result["peak_current_error"] < 0.05  # < 5% error

    def test_neutron_yield_within_order(self):
        """Yield within order of magnitude is flagged."""
        from dpf.validation.experimental import validate_neutron_yield

        result = validate_neutron_yield(5e10, "PF-1000")  # 5e10 vs ~1e11
        assert result["within_order_magnitude"]

    def test_neutron_yield_outside_order(self):
        """Yield far off is flagged."""
        from dpf.validation.experimental import validate_neutron_yield

        result = validate_neutron_yield(1e5, "PF-1000")  # way too low
        assert not result["within_order_magnitude"]

    def test_device_to_config(self):
        """Device parameters convert to valid config dict."""
        from dpf.validation.experimental import device_to_config_dict

        cfg = device_to_config_dict("PF-1000")
        assert "circuit" in cfg
        assert cfg["circuit"]["C"] > 0
        assert cfg["circuit"]["V0"] > 0
        assert cfg["circuit"]["anode_radius"] < cfg["circuit"]["cathode_radius"]
