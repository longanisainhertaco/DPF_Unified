"""Tests for mlx_sources: geometric sources, ohmic heating, bremsstrahlung, J computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from dpf.metal.mlx_kernels import IDN, IEE, IEN, IBR, IBT, IBZ, IMR, IMT, IMZ, ISR, NVAR  # noqa: E402, I001
from dpf.metal.mlx_sources import (  # noqa: E402
    apply_bremsstrahlung,
    apply_geometric_sources,
    apply_ohmic_heating,
    compute_current_density,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAMMA = 5.0 / 3.0
_MU0 = 4.0 * math.pi * 1e-7
_KBOLTZ = 1.380649e-23
_ION_MASS = 3.34358377e-27


def _make_uniform_state(nr: int, nz: int, gamma: float = _GAMMA) -> mx.array:
    """Uniform state: rho=1, vr=vz=vt=0, p=1, B=0, Srho=0."""
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    U[IDN] = 1.0
    U[IEN] = 1.0 / (gamma - 1.0)
    return mx.array(U)


def _make_rotating_state(
    nr: int, nz: int, rho: float, vt: float, p: float, gamma: float = _GAMMA
) -> mx.array:
    """Solid-body rotation: uniform rho, vtheta=vt, vr=vz=0, B=0."""
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    U[IDN] = rho
    U[IMT] = rho * vt
    e_kin = 0.5 * rho * vt**2
    U[IEN] = p / (gamma - 1.0) + e_kin
    return mx.array(U)


def _make_grid(nr: int, nz: int, dr: float = 0.01, r_inner: float = 0.0):
    r_cell = mx.array(
        [r_inner + (i + 0.5) * dr for i in range(nr)], dtype=mx.float32
    )
    inv_r_list = []
    for i in range(nr):
        rc = r_inner + (i + 0.5) * dr
        inv_r_list.append(2.0 / dr if rc == 0.0 else 1.0 / rc)
    inv_r = mx.array(inv_r_list, dtype=mx.float32)
    return r_cell, inv_r


# ---------------------------------------------------------------------------
# Geometric source tests
# ---------------------------------------------------------------------------


class TestGeometricSources:
    def test_zero_vtheta_zero_B_no_azimuthal_source(self):
        """Zero vtheta, zero B → no Coriolis source on azimuthal momentum (S_mt=0)."""
        nr, nz = 8, 8
        r_cell, inv_r = _make_grid(nr, nz)
        U = _make_uniform_state(nr, nz)  # vt=0, B=0
        dt = 1e-7
        U_out = apply_geometric_sources(U, r_cell, inv_r, dt, use_metal_kernel=False)
        mx.eval(U_out)
        # S_mt = 0 when vr=vtheta=0 and B=0
        dmt = np.asarray(U_out[IMT]) - np.asarray(U[IMT])
        np.testing.assert_allclose(dmt, 0.0, atol=1e-10)

    def test_zero_vtheta_zero_B_no_centrifugal(self):
        """Zero vtheta, zero B → no centrifugal contribution to S_mr (only pressure hoop)."""
        nr, nz = 8, 8
        dr = 0.01
        r_cell, inv_r = _make_grid(nr, nz, dr=dr)
        # Uniform pressure: p/r term is present but centrifugal (rho*vt^2/r) is zero
        U = _make_uniform_state(nr, nz)
        dt = 1e-7
        U_out = apply_geometric_sources(U, r_cell, inv_r, dt, use_metal_kernel=False)
        mx.eval(U_out)
        dmr = np.asarray(U_out[IMR]) - np.asarray(U[IMR])
        # Interior cells: S_mr = p/r (purely geometric), no vtheta contribution.
        # p_total stored = E * (gamma-1), since vr=vz=vt=0 and B=0:
        # U[IEN] = 1/(gamma-1), so p = (gamma-1)*U[IEN] = 1.0
        p_val = 1.0
        for ir in range(1, nr):  # skip axis (L'Hopital)
            rc = (ir + 0.5) * dr
            expected_mr = p_val / rc * dt
            np.testing.assert_allclose(
                dmr[ir, 0], expected_mr, rtol=0.02,
                err_msg=f"pressure hoop mismatch at ir={ir}"
            )

    def test_uniform_rotation_centrifugal(self):
        """Uniform rotation produces outward centrifugal force on radial momentum."""
        nr, nz = 16, 4
        dr = 0.01
        r_cell, inv_r = _make_grid(nr, nz, dr=dr)
        rho, vt, p = 1.0, 1e3, 1e4
        U = _make_rotating_state(nr, nz, rho, vt, p)
        dt = 1e-9

        U_out = apply_geometric_sources(U, r_cell, inv_r, dt, use_metal_kernel=False)
        mx.eval(U_out)

        dmr = np.asarray(U_out[IMR]) - np.asarray(U[IMR])
        # Interior cells (not axis): centrifugal term rho*vt^2/r > 0 → dmr > 0
        for ir in range(2, nr):
            rc = (ir + 0.5) * dr
            expected = rho * vt**2 / rc * dt
            np.testing.assert_allclose(
                dmr[ir].mean(), expected, rtol=0.05,
                err_msg=f"centrifugal mismatch at ir={ir}, r={rc:.3f}"
            )

    def test_axis_source_finite_not_nan(self):
        """L'Hopital at axis: sources must be finite (not NaN/Inf)."""
        nr, nz = 8, 8
        r_cell, inv_r = _make_grid(nr, nz, r_inner=0.0)
        U = _make_rotating_state(nr, nz, rho=1.0, vt=100.0, p=1e3)
        U_out = apply_geometric_sources(U, r_cell, inv_r, dt=1e-9, use_metal_kernel=False)
        mx.eval(U_out)
        arr = np.asarray(U_out)
        assert np.all(np.isfinite(arr)), "NaN/Inf after geometric source at axis"

    def test_density_conserved_by_geometric_source(self):
        """Geometric sources do not modify density."""
        nr, nz = 8, 8
        r_cell, inv_r = _make_grid(nr, nz)
        U = _make_rotating_state(nr, nz, rho=1.2, vt=500.0, p=5e3)
        U_out = apply_geometric_sources(U, r_cell, inv_r, dt=1e-9, use_metal_kernel=False)
        mx.eval(U_out)
        np.testing.assert_array_equal(np.asarray(U_out[IDN]), np.asarray(U[IDN]))

    def test_coriolis_term_zero_when_vtheta_zero(self):
        """S_mt = 0 when vtheta = 0 (no Coriolis without azimuthal flow)."""
        nr, nz = 8, 8
        r_cell, inv_r = _make_grid(nr, nz)
        U = _make_uniform_state(nr, nz)  # vt=0
        U_out = apply_geometric_sources(U, r_cell, inv_r, dt=1e-8, use_metal_kernel=False)
        mx.eval(U_out)
        dmt = np.asarray(U_out[IMT]) - np.asarray(U[IMT])
        np.testing.assert_allclose(dmt, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Current density tests
# ---------------------------------------------------------------------------


class TestCurrentDensity:
    def test_zero_B_gives_zero_J_sq(self):
        """Zero magnetic field → J = 0."""
        nr, nz = 8, 8
        U = _make_uniform_state(nr, nz)  # B=0
        r_cell, _ = _make_grid(nr, nz)
        J_sq = compute_current_density(U, dr=0.01, dz=0.01, r_cell=r_cell)
        mx.eval(J_sq)
        np.testing.assert_allclose(np.asarray(J_sq), 0.0, atol=1e-10)

    def test_uniform_Bz_gives_zero_J(self):
        """Uniform Bz only → J = curl(Bz hat) = 0 analytically."""
        nr, nz = 8, 8
        U_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        U_np[IDN] = 1.0
        U_np[IBZ] = 1.0
        U_np[IEN] = 1.0
        U = mx.array(U_np)
        r_cell, _ = _make_grid(nr, nz)
        J_sq = compute_current_density(U, dr=0.01, dz=0.01, r_cell=r_cell)
        mx.eval(J_sq)
        np.testing.assert_allclose(np.asarray(J_sq), 0.0, atol=1e-8)

    def test_azimuthal_current_from_Btheta_gradient(self):
        """Bt = B0 * r / r_max → Jz = (1/r) d(r*Bt)/dr = 2*B0/r_max analytically."""
        nr, nz = 16, 4
        dr = 0.01
        r_max = nr * dr
        r_cell, _ = _make_grid(nr, nz, dr=dr)

        U_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        U_np[IDN] = 1.0
        U_np[IEN] = 1.0
        B0 = 1.0
        for ir in range(nr):
            rc = (ir + 0.5) * dr
            U_np[IBT, ir, :] = B0 * rc / r_max
        U = mx.array(U_np)

        J_sq = compute_current_density(U, dr=dr, dz=0.01, r_cell=r_cell)
        mx.eval(J_sq)

        # Jz_analytic = 2*B0/r_max everywhere (uniform current density)
        Jz_expected = 2.0 * B0 / r_max
        J_sq_expected = Jz_expected**2
        # Interior cells only (boundary stencils are one-sided)
        J_sq_np = np.asarray(J_sq)
        np.testing.assert_allclose(
            J_sq_np[2:-2, 1:-1], J_sq_expected, rtol=0.05
        )

    def test_J_sq_nonnegative(self):
        """J^2 must always be non-negative."""
        nr, nz = 8, 8
        rng = np.random.default_rng(42)
        U_np = rng.standard_normal((NVAR, nr, nz)).astype(np.float32)
        U_np[IDN] = np.abs(U_np[IDN]) + 0.1
        U = mx.array(U_np)
        r_cell, _ = _make_grid(nr, nz)
        J_sq = compute_current_density(U, dr=0.01, dz=0.01, r_cell=r_cell)
        mx.eval(J_sq)
        assert np.all(np.asarray(J_sq) >= 0.0), "J^2 has negative values"


# ---------------------------------------------------------------------------
# Ohmic heating tests
# ---------------------------------------------------------------------------


class TestOhmicHeating:
    def test_energy_increases_by_eta_Jsq_dt_volume(self):
        """Total energy increase = eta * J^2 * dt integrated over all cells."""
        nr, nz = 8, 8
        U = _make_uniform_state(nr, nz)
        eta = 0.5
        J_sq_val = 4.0
        J_sq = mx.full((nr, nz), J_sq_val, dtype=mx.float32)
        dt = 1e-6

        U_out = apply_ohmic_heating(U, eta, J_sq, dt)
        mx.eval(U_out)

        dE = np.asarray(U_out[IEN]) - np.asarray(U[IEN])
        expected = eta * J_sq_val * dt
        # float32 accumulation; 2% tolerance is appropriate
        np.testing.assert_allclose(dE, expected, rtol=0.02)

    def test_entropy_tracer_increases(self):
        """Entropy tracer S*rho must increase due to ohmic heating."""
        nr, nz = 8, 8
        U = _make_uniform_state(nr, nz)
        eta = 1.0
        J_sq = mx.full((nr, nz), 2.0, dtype=mx.float32)
        U_out = apply_ohmic_heating(U, eta, J_sq, dt=1e-6)
        mx.eval(U_out)
        dSrho = np.asarray(U_out[ISR]) - np.asarray(U[ISR])
        assert np.all(dSrho >= 0.0), "entropy tracer decreased during ohmic heating"

    def test_ohmic_does_not_change_momentum_or_B(self):
        """Ohmic heating only touches IEN and ISR."""
        nr, nz = 8, 8
        U = _make_uniform_state(nr, nz)
        J_sq = mx.full((nr, nz), 1.0, dtype=mx.float32)
        U_out = apply_ohmic_heating(U, 0.1, J_sq, dt=1e-6)
        mx.eval(U_out)
        for idx in [IDN, IMR, IMZ, IMT, IBR, IBZ, IBT, IEE]:
            np.testing.assert_array_equal(
                np.asarray(U_out[idx]), np.asarray(U[idx]),
                err_msg=f"Variable index {idx} changed unexpectedly"
            )

    def test_ohmic_zero_eta_no_change(self):
        """eta=0 → no change to any variable."""
        nr, nz = 8, 8
        U = _make_uniform_state(nr, nz)
        J_sq = mx.full((nr, nz), 100.0, dtype=mx.float32)
        U_out = apply_ohmic_heating(U, 0.0, J_sq, dt=1e-6)
        mx.eval(U_out)
        np.testing.assert_allclose(np.asarray(U_out), np.asarray(U), atol=1e-10)

    def test_ohmic_array_eta(self):
        """Spatially varying eta (mx.array) accepted without error."""
        nr, nz = 4, 4
        U = _make_uniform_state(nr, nz)
        eta = mx.full((nr, nz), 0.2, dtype=mx.float32)
        J_sq = mx.ones((nr, nz), dtype=mx.float32)
        U_out = apply_ohmic_heating(U, eta, J_sq, dt=1e-7)
        mx.eval(U_out)
        assert np.all(np.isfinite(np.asarray(U_out)))


# ---------------------------------------------------------------------------
# Bremsstrahlung tests
# ---------------------------------------------------------------------------


class TestBremsstrahlung:
    def test_energy_decreases(self):
        """Bremsstrahlung removes energy from total energy."""
        nr, nz = 8, 8
        # Hot dense plasma: rho=1 kg/m^3, Te~1e7 K
        rho = 1.0
        Te_target = 1e7
        ne = rho / _ION_MASS
        p = ne * _KBOLTZ * Te_target
        U_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        U_np[IDN] = rho
        U_np[IEN] = p / (_GAMMA - 1.0)
        U = mx.array(U_np)

        dt = 1e-9
        U_out = apply_bremsstrahlung(U, dt)
        mx.eval(U_out)

        dE = np.asarray(U_out[IEN]) - np.asarray(U[IEN])
        assert np.all(dE <= 0.0), "bremsstrahlung added energy (should remove)"

    def test_bremsstrahlung_magnitude_matches_analytic(self):
        """Q_rad magnitude matches 1.42e-40 * g_ff * Z * ne^2 * sqrt(Te)."""
        nr, nz = 4, 4
        rho = 1.0
        Te_target = 1e7
        ne = rho / _ION_MASS
        # p = 2 * rho * kB * T / m_i for fully ionized Z=1 (n_e + n_i)
        p = 2.0 * ne * _KBOLTZ * Te_target
        U_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        U_np[IDN] = rho
        U_np[IEN] = float(p / (_GAMMA - 1.0))
        U = mx.array(U_np)

        Z_eff = 1.0
        g_ff = 1.2
        dt = 1e-12
        U_out = apply_bremsstrahlung(U, dt, Z_eff=Z_eff, gaunt_factor=g_ff)
        mx.eval(U_out)

        dE = np.asarray(U[IEN]) - np.asarray(U_out[IEN])
        Q_expected = 1.42e-40 * g_ff * Z_eff * ne**2 * math.sqrt(Te_target)
        expected_dE = Q_expected * dt
        np.testing.assert_allclose(dE, expected_dE, rtol=0.05)

    def test_bremsstrahlung_no_negative_energy(self):
        """Bremsstrahlung clamp prevents energy from going negative."""
        nr, nz = 4, 4
        # State with significant kinetic energy so the floor is non-trivial in float32
        rho = 1e3
        vr_val = 1e3   # kinetic energy = 0.5 * 1e3 * 1e6 = 5e8 J/m^3
        U_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        U_np[IDN] = rho
        U_np[IMR] = rho * vr_val
        e_kin = 0.5 * rho * vr_val**2
        U_np[IEN] = float(e_kin + 1.0)  # only 1 J/m^3 thermal above kinetic
        U = mx.array(U_np)
        U_out = apply_bremsstrahlung(U, dt=1e3)  # huge dt drives clamping
        mx.eval(U_out)
        E_out = np.asarray(U_out[IEN])
        # Energy must not go below kinetic floor (within float32 resolution)
        e_kin_np = 0.5 * rho * vr_val**2
        assert np.all(E_out >= e_kin_np - 1.0), "energy dropped below kinetic floor"

    def test_bremsstrahlung_does_not_change_momentum_or_B(self):
        """Bremsstrahlung only touches IEN."""
        nr, nz = 4, 4
        rho = 0.1
        p = 1e5
        U_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        U_np[IDN] = rho
        U_np[IEN] = float(p / (_GAMMA - 1.0))
        U_np[IMR] = 10.0
        U_np[IBZ] = 0.5
        U = mx.array(U_np)
        U_out = apply_bremsstrahlung(U, dt=1e-9)
        mx.eval(U_out)
        for idx in [IDN, IMR, IMZ, IMT, ISR, IBR, IBZ, IBT, IEE]:
            np.testing.assert_array_equal(
                np.asarray(U_out[idx]), np.asarray(U[idx]),
                err_msg=f"Variable index {idx} changed unexpectedly"
            )


# ---------------------------------------------------------------------------
# Entropy consistency test
# ---------------------------------------------------------------------------


class TestEntropyConsistency:
    def test_ohmic_entropy_consistent_with_energy(self):
        """Entropy increase from ohmic heating is consistent with Q_ohm*dt*(gamma-1)*rho/p."""
        nr, nz = 4, 4
        gamma = _GAMMA
        rho = 1.0
        p = 1.0  # low pressure so dE is a significant fraction of E
        U_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        U_np[IDN] = rho
        U_np[IEN] = float(p / (gamma - 1.0))
        U = mx.array(U_np)

        eta = 1.0
        J_sq_val = 1.0
        dt = 1e-3  # large enough that dE ~ 1e-3, resolvable vs E~1.5
        J_sq = mx.full((nr, nz), J_sq_val, dtype=mx.float32)
        U_out = apply_ohmic_heating(U, eta, J_sq, dt, gamma=gamma)
        mx.eval(U_out)

        dSrho = np.asarray(U_out[ISR]) - np.asarray(U[ISR])
        Q_ohm = eta * J_sq_val
        dE_formula = Q_ohm * dt

        # dSrho = Q_ohm * dt * (gamma-1) * rho / p
        expected_dSrho = dE_formula * (gamma - 1.0) * rho / p
        # 5% tolerance to account for float32 arithmetic on both sides
        np.testing.assert_allclose(dSrho, expected_dSrho, rtol=0.05)
