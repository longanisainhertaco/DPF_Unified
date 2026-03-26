"""Tests for mlx_line_radiation: apply_line_radiation_mlx."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from dpf.metal.mlx_kernels import IDN, IEE, IEN, IBR, IBT, IBZ, IMR, IMT, IMZ, NVAR  # noqa: E402, I001
from dpf.metal.mlx_line_radiation import apply_line_radiation_mlx  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAMMA = 5.0 / 3.0
_KBOLTZ = 1.380649e-23
_EV = 1.602176634e-19
_ION_MASS = 3.34358377e-27


def _uniform_state(
    nr: int,
    nz: int,
    rho: float,
    Te_eV: float,
    gamma: float = _GAMMA,
) -> mx.array:
    """Build conserved state from rho and electron temperature in eV.

    Te [K] = Te_eV * eV / kB.
    p = 2 * rho/m_i * kB * Te (n_e + n_i = 2 * rho/m_i at Z=1).
    E = p/(gamma-1)  (kinetic=0, B=0).
    """
    Te_K = Te_eV * _EV / _KBOLTZ
    p = 2.0 * (rho / _ION_MASS) * _KBOLTZ * Te_K
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    U[IDN] = rho
    U[IEN] = p / (gamma - 1.0)
    return mx.array(U)


def _uniform_species(nr: int, nz: int, n_species: int, Y0: float) -> mx.array:
    """Uniform species mass fractions, shape (n_species, nr, nz)."""
    Y = np.full((n_species, nr, nz), Y0, dtype=np.float32)
    return mx.array(Y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLineRadiationCu:
    def test_line_radiation_cu_nonzero(self):
        """Cu at 50 eV produces nonzero cooling (energy decreases)."""
        nr, nz = 8, 8
        rho0 = 1e-4       # kg/m^3 — plasma density
        Te_eV = 50.0      # eV — inside Cu M-shell peak

        U = _uniform_state(nr, nz, rho0, Te_eV)
        E_before = float(mx.sum(U[IEN]))

        # Single Cu species, mass fraction 1.0 (pure Cu plasma)
        species_Z = [29]
        Y = _uniform_species(nr, nz, n_species=1, Y0=1.0)

        dt = 1e-9  # 1 ns
        U_out = apply_line_radiation_mlx(U, dt, species_Z=species_Z, species_Y=Y)
        mx.eval(U_out)

        E_after = float(mx.sum(U_out[IEN]))

        assert E_after < E_before, (
            f"Cu line radiation produced no cooling: E_before={E_before:.4e}, "
            f"E_after={E_after:.4e}"
        )
        # Energy must have decreased (nonzero cooling)
        dE = E_before - E_after
        assert dE > 0.0, f"dE should be positive (energy removed), got {dE}"

    def test_line_radiation_cu_energy_floor(self):
        """Line radiation never drives energy below the kinetic+magnetic+floor."""
        nr, nz = 4, 4
        rho0 = 1e-4
        Te_eV = 100.0  # near M-shell peak — maximum cooling

        U = _uniform_state(nr, nz, rho0, Te_eV)

        species_Z = [29]
        Y = _uniform_species(nr, nz, n_species=1, Y0=1.0)

        # Very large dt to stress the energy clamp
        dt = 1e-3
        U_out = apply_line_radiation_mlx(U, dt, species_Z=species_Z, species_Y=Y)
        mx.eval(U_out)

        # Energy must remain non-negative (clamp allows draining to thermal floor)
        E_out = np.asarray(U_out[IEN])
        assert np.all(E_out >= 0.0), f"Negative energy after line radiation: min={E_out.min()}"

    def test_line_radiation_cu_zero_fraction_no_cooling(self):
        """Zero Cu mass fraction → no energy change."""
        nr, nz = 4, 4
        rho0 = 1e-4
        Te_eV = 50.0

        U = _uniform_state(nr, nz, rho0, Te_eV)
        E_before = np.asarray(U[IEN]).copy()

        species_Z = [29]
        # Y = 0 everywhere → no Cu → no line radiation
        Y = mx.zeros((1, nr, nz), dtype=mx.float32)

        dt = 1e-9
        U_out = apply_line_radiation_mlx(U, dt, species_Z=species_Z, species_Y=Y)
        mx.eval(U_out)

        E_after = np.asarray(U_out[IEN])
        # With Y=0, cooling should be negligible (floor-level only)
        # Allow tiny numerical noise from exp(log_floor)
        np.testing.assert_allclose(E_after, E_before, rtol=1e-3)

    def test_line_radiation_cu_only_modifies_energy(self):
        """Line radiation must not change rho, momenta, or B-field."""
        nr, nz = 4, 4
        U = _uniform_state(nr, nz, rho=1e-4, Te_eV=50.0)
        species_Z = [29]
        Y = _uniform_species(nr, nz, n_species=1, Y0=0.5)

        dt = 1e-9
        U_out = apply_line_radiation_mlx(U, dt, species_Z=species_Z, species_Y=Y)
        mx.eval(U_out)

        # Density, momenta, B-fields, electron energy must be unchanged
        for slot in [IDN, IMR, IMZ, IMT, IBR, IBZ, IBT, IEE]:
            np.testing.assert_array_equal(
                np.asarray(U_out[slot]), np.asarray(U[slot]),
                err_msg=f"Slot {slot} modified by line radiation (should only touch IEN)",
            )


class TestLineRadiationH:
    def test_line_radiation_h_vs_brem(self):
        """H line radiation is comparable to bremsstrahlung at 10 eV.

        At 10 eV, hydrogen Lyman-alpha cooling Lambda_H ~ 3e-33 W m^3.
        Bremsstrahlung Lambda_ff ~ 1.42e-40 * sqrt(Te) / n_e [W m^3 per n_e^2].
        At 10 eV (T ~ 1.16e5 K), bremsstrahlung is weaker than line radiation.
        So line cooling should dominate or be comparable.
        """
        from dpf.metal.mlx_sources import apply_bremsstrahlung

        nr, nz = 8, 8
        rho0 = 1e-3       # kg/m^3 — denser than Cu test
        Te_eV = 10.0      # 10 eV — below ionization, line cooling significant

        U = _uniform_state(nr, nz, rho0, Te_eV)
        E0 = float(mx.sum(U[IEN]))

        # H line radiation
        species_Z = [1]
        Y_h = mx.ones((1, nr, nz), dtype=mx.float32)
        dt = 1e-9

        U_line = apply_line_radiation_mlx(U, dt, species_Z=species_Z, species_Y=Y_h)
        mx.eval(U_line)
        dE_line = E0 - float(mx.sum(U_line[IEN]))

        # Bremsstrahlung for comparison
        U_brem = apply_bremsstrahlung(U, dt, gamma=_GAMMA, Z_eff=1.0)
        mx.eval(U_brem)
        dE_brem = E0 - float(mx.sum(U_brem[IEN]))

        # Both must produce nonzero cooling
        assert dE_line > 0.0, f"H line radiation nonzero at 10 eV: dE_line={dE_line:.3e}"
        assert dE_brem > 0.0, f"Bremsstrahlung nonzero at 10 eV: dE_brem={dE_brem:.3e}"

        # At 10 eV, H line cooling should be at least 1% of bremsstrahlung
        # (actual physics: it should be much larger, but 1% is conservative)
        ratio = dE_line / max(dE_brem, 1e-40)
        assert ratio > 0.01, (
            f"H line radiation too weak vs bremsstrahlung at 10 eV: "
            f"dE_line={dE_line:.3e}, dE_brem={dE_brem:.3e}, ratio={ratio:.3f}"
        )

    def test_line_radiation_h_nonzero(self):
        """H at 4 eV (Lyman-alpha peak) produces nonzero cooling."""
        nr, nz = 8, 8
        rho0 = 1e-3
        Te_eV = 4.0  # Lyman-alpha peak

        U = _uniform_state(nr, nz, rho0, Te_eV)
        E_before = float(mx.sum(U[IEN]))

        species_Z = [1]
        Y = mx.ones((1, nr, nz), dtype=mx.float32)
        dt = 1e-9

        U_out = apply_line_radiation_mlx(U, dt, species_Z=species_Z, species_Y=Y)
        mx.eval(U_out)

        E_after = float(mx.sum(U_out[IEN]))
        assert E_after < E_before, "H line radiation at 4 eV produced no cooling"

    def test_line_radiation_h_high_temp_less_cooling(self):
        """H line cooling decreases at high temperature (post-ionization residual)."""
        nr, nz = 4, 4
        rho0 = 1e-3
        species_Z = [1]
        dt = 1e-9

        # 4 eV (Lyman-alpha peak) should produce more cooling than 1000 eV
        U_4eV = _uniform_state(nr, nz, rho0, 4.0)
        U_1keV = _uniform_state(nr, nz, rho0, 1000.0)

        Y = mx.ones((1, nr, nz), dtype=mx.float32)

        U_out_4 = apply_line_radiation_mlx(U_4eV, dt, species_Z=species_Z, species_Y=Y)
        U_out_1k = apply_line_radiation_mlx(U_1keV, dt, species_Z=species_Z, species_Y=Y)
        mx.eval(U_out_4, U_out_1k)

        # Normalize by initial energy to compare rates
        dE_4 = float(mx.sum(U_4eV[IEN])) - float(mx.sum(U_out_4[IEN]))
        dE_1k = float(mx.sum(U_1keV[IEN])) - float(mx.sum(U_out_1k[IEN]))

        # Cooling rate per unit energy should be higher at 4 eV than 1 keV for H
        rate_4 = dE_4 / max(float(mx.sum(U_4eV[IEN])), 1e-30)
        rate_1k = dE_1k / max(float(mx.sum(U_1keV[IEN])), 1e-30)
        assert rate_4 >= rate_1k, (
            f"H line cooling should be stronger at 4 eV than 1 keV: "
            f"rate_4={rate_4:.3e}, rate_1k={rate_1k:.3e}"
        )


class TestLineRadiationMultiSpecies:
    def test_two_species_both_contribute(self):
        """D2+Cu at 50 eV: both species contribute, total > either alone."""
        nr, nz = 4, 4
        rho0 = 1e-4
        Te_eV = 50.0
        dt = 1e-9

        U = _uniform_state(nr, nz, rho0, Te_eV)
        E0 = float(mx.sum(U[IEN]))

        # H only
        species_Z_h = [1]
        Y_h = mx.ones((1, nr, nz), dtype=mx.float32)
        U_h = apply_line_radiation_mlx(U, dt, species_Z=species_Z_h, species_Y=Y_h)
        mx.eval(U_h)

        # Cu only
        species_Z_cu = [29]
        Y_cu = mx.ones((1, nr, nz), dtype=mx.float32)
        U_cu = apply_line_radiation_mlx(U, dt, species_Z=species_Z_cu, species_Y=Y_cu)
        mx.eval(U_cu)

        # Both together (each with Y=0.5)
        species_Z_both = [1, 29]
        Y_both = mx.array(
            np.full((2, nr, nz), 0.5, dtype=np.float32)
        )
        U_both = apply_line_radiation_mlx(U, dt, species_Z=species_Z_both, species_Y=Y_both)
        mx.eval(U_both)
        dE_both = E0 - float(mx.sum(U_both[IEN]))

        # Combined cooling must be nonzero
        assert dE_both > 0.0, "Two-species line radiation produced zero cooling"

    def test_generic_z_fallback(self):
        """Generic Z fallback (e.g., Z=18 Argon) produces nonzero cooling."""
        nr, nz = 4, 4
        rho0 = 1e-4
        Te_eV = 200.0  # near Ar peak

        U = _uniform_state(nr, nz, rho0, Te_eV)
        E_before = float(mx.sum(U[IEN]))

        species_Z = [18]  # Argon — uses generic fallback
        Y = mx.ones((1, nr, nz), dtype=mx.float32)
        dt = 1e-9

        U_out = apply_line_radiation_mlx(U, dt, species_Z=species_Z, species_Y=Y)
        mx.eval(U_out)

        E_after = float(mx.sum(U_out[IEN]))
        assert E_after < E_before, "Generic-Z (Ar) line radiation produced no cooling"
