"""Regression test for the cylindrical conservative-form pressure recovery.

Guards against dropping the 1/mu_0 factor in the dE/dt -> dp/dt conversion
inside CylindricalMHDSolver._compute_rhs.

Conservative MHD total energy in SI units (per KR:
a-constrained-transport-embedded-boundary-method-for-compressible-resistive-magnetohydrodynamics.md
section 2.2 page 3 equation (9), translated from MHD code units to SI):

    E = p/(gamma-1) + 0.5 * rho * v^2 + B^2 / (2 * mu_0)

Time-differentiating:

    dE/dt = (dp/dt)/(gamma-1) + v . dmom/dt - 0.5 * v^2 * drho/dt
            + (B . dB/dt) / mu_0

Solving for dp/dt:

    dp/dt = (gamma-1) * (dE/dt - v . dmom/dt + 0.5 * v^2 * drho/dt
            - (B . dB/dt) / mu_0)

The historical bug at src/dpf/fluid/cylindrical_mhd.py:1189 dropped the
/ mu_0 on the (B . dB/dt) term, which inflated the magnetic-energy term
in the recovered pressure by mu_0^-1 ~ 7.96e5.  The Cartesian companion
at src/dpf/fluid/mhd_solver.py:1865 has always carried the / mu_0 factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import mu_0
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver


def _build_zero_velocity_state(
    nr: int,
    nz: int,
    dr: float,
    dz: float,
    rho0: float,
    p0: float,
    B_theta_amp: float,
):
    """Uniform rho, uniform p, zero velocity, non-uniform B_theta(r) and B_z(z).

    With v = 0 everywhere and rho, p uniform:
      drho/dt = -div(rho v) = 0  (v = 0)
      v . dmom/dt = 0             (v = 0)
      0.5 * v^2 * drho/dt = 0     (v = 0)

    And the energy flux F_E = (E + p_total) v - B (v . B) vanishes with v = 0,
    so -div(F_E) = 0.  The only contribution to dE/dt is ohmic heating
    (we enable resistivity to drive a non-zero curl(E) -> dB/dt != 0):

      dE/dt = ohmic_heating

    Therefore the conservative-form identity for pressure recovery reduces to:

      dp/dt = (gamma-1) * (ohmic_heating - (B . dB/dt) / mu_0)

    The B field is given a (theta, z) profile with both r-variation and
    z-variation so that J = curl(B)/mu_0 has multiple components and
    curl(E_resistive) = curl(eta J) is non-zero.
    """
    geom_r = (np.arange(nr) + 0.5) * dr
    geom_z = (np.arange(nz) + 0.5) * dz
    rho = np.full((nr, 1, nz), rho0, dtype=np.float64)
    p = np.full((nr, 1, nz), p0, dtype=np.float64)
    velocity = np.zeros((3, nr, 1, nz), dtype=np.float64)
    B = np.zeros((3, nr, 1, nz), dtype=np.float64)
    # B_theta(r, z) varies in both r and z so curl(eta * J) has axial component
    # via d/dz(eta J_r) and d/dr(eta J_theta).
    B_theta = B_theta_amp * geom_r[:, None] * (1.0 + 0.5 * geom_z[None, :] / (dz * nz))
    B[1, :, 0, :] = B_theta
    # B_z varies with r so J_theta = -dB_z/dr is non-zero.
    B_z = 0.3 * B_theta_amp * (dr * nr) * (1.0 + 0.2 * np.cos(np.pi * geom_r / (dr * nr)))[:, None]
    B[2, :, 0, :] = B_z * np.ones(nz)[None, :]
    return {
        "rho": rho,
        "velocity": velocity,
        "pressure": p,
        "B": B,
        "psi": np.zeros((nr, 1, nz), dtype=np.float64),
        "Te": np.full((nr, 1, nz), 1e4, dtype=np.float64),
        "Ti": np.full((nr, 1, nz), 1e4, dtype=np.float64),
    }


def test_pressure_recovery_includes_mu0_factor():
    """dp/dt must contain the /mu_0 factor on the (B . dB/dt) term.

    With v = 0 everywhere, uniform rho, uniform p, no source terms, and
    resistivity enabled to drive a non-zero curl(E) -> dB/dt != 0:

      drho/dt = 0
      v . dmom/dt = 0
      energy flux F_E = (E + p_total) v - B(v . B) = 0
      dE/dt = -div(F_E) + ohmic_heating = ohmic_heating

    Therefore the conservative-form pressure recovery becomes:

      dp/dt = (gamma - 1) * (ohmic_heating - (B . dB/dt) / mu_0)

    Both ohmic_heating (returned by _compute_rhs) and dB/dt are provided
    by the solver, so we can verify this identity to machine precision.
    """
    nr, nz = 16, 8
    dr = 1e-3
    dz = 1e-3
    rho0 = 1e-4       # kg/m^3 (typical DPF prefill scale)
    p0 = 1e3          # Pa (avoid being so small that pressure floor activates)
    B_theta_amp = 5e-7  # gives peak B ~ 1e-8 T -> small JxB so v stays ~ 0

    solver = CylindricalMHDSolver(
        nr=nr,
        nz=nz,
        dr=dr,
        dz=dz,
        gamma=5.0 / 3.0,
        enable_hall=False,         # Hall E = (J x B)/(n_e e) -- if J || B, no Hall E
        enable_resistive=True,     # E_resistive = eta * J -- always non-zero if J != 0
        enable_energy_equation=True,
        conservative_energy=True,
        riemann_solver="hll",
        time_integrator="ssp_rk2",
        use_godunov_flux=False,
    )

    state = _build_zero_velocity_state(nr, nz, dr, dz, rho0, p0, B_theta_amp)

    # Squeeze to 2D as _compute_rhs expects
    rho_2d = solver._squeeze(state["rho"])
    vel_2d = solver._squeeze(state["velocity"])
    p_2d = solver._squeeze(state["pressure"])
    B_2d = solver._squeeze(state["B"])
    psi_2d = solver._squeeze(state["psi"])

    # Uniform resistivity field eta = 1.0 Ohm.m (large value to make
    # ohmic terms numerically prominent vs roundoff).
    eta_field = np.ones((nr, nz), dtype=np.float64)

    rhs = solver._compute_rhs(
        rho_2d, vel_2d, p_2d, B_2d, psi_2d,
        eta_field=eta_field,
        source_terms=None,
        e_electron=None,
    )

    assert "dp_dt" in rhs, (
        "Conservative-energy path must return dp_dt (post-conversion)."
    )
    assert "dB_dt" in rhs
    assert "drho_dt" in rhs
    assert "dmom_dt" in rhs
    assert "ohmic_heating" in rhs

    dp_dt = rhs["dp_dt"]
    dB_dt = rhs["dB_dt"]
    drho_dt = rhs["drho_dt"]
    ohmic_heating = rhs["ohmic_heating"]

    # Sanity: with v = 0, drho/dt must be zero (no advection of mass).
    assert np.max(np.abs(drho_dt)) < 1e-20, (
        f"drho/dt should vanish with v=0, got max abs = {np.max(np.abs(drho_dt))}"
    )

    # Sanity: dB/dt must be non-zero somewhere or the test is vacuous.
    B_dot_dB = np.sum(B_2d * dB_dt, axis=0)  # shape (nr, nz)
    assert np.max(np.abs(B_dot_dB)) > 0.0, (
        "Test is vacuous: B . dB/dt is identically zero. "
        "Increase B_theta_amp or check resistivity is enabled."
    )

    # Sanity: ohmic_heating > 0 (eta * |J|^2 must be non-negative; non-zero if J != 0).
    assert np.max(np.abs(ohmic_heating)) > 0.0, (
        "Test is vacuous: ohmic_heating is identically zero."
    )

    gamma_minus_1 = solver.gamma - 1.0
    expected_dp_dt = gamma_minus_1 * (ohmic_heating - B_dot_dB / mu_0)

    abs_scale = np.max(np.abs(expected_dp_dt))
    err = np.max(np.abs(dp_dt - expected_dp_dt))
    rel_err = err / max(abs_scale, 1e-300)

    # If the bug were present (missing /mu_0), expected_dp_dt would be
    # off by ~mu_0^-1 = 7.96e5 on the magnetic-energy term.  Machine
    # precision for float64 is ~2.2e-16; allow some headroom for PLM
    # roundoff: rtol = 1e-10.
    assert rel_err < 1e-10, (
        f"Cylindrical pressure recovery diverges from the SI conservative-form "
        f"identity dp/dt = (gamma-1)(ohmic_heating - (B.dB/dt)/mu_0):\n"
        f"  max |dp_dt - expected| = {err:.3e}\n"
        f"  max |expected|         = {abs_scale:.3e}\n"
        f"  relative error         = {rel_err:.3e}\n"
        f"This usually means the / mu_0 factor was dropped at "
        f"cylindrical_mhd.py:1189 (or moved)."
    )


def test_pressure_recovery_factor_of_mu0_sensitivity():
    """If we deliberately drop the /mu_0 factor, the test must fail loudly.

    This guards against a future refactor that silently 're-introduces'
    the bug under a different variable name.  We simulate the broken
    formula and confirm the relative error explodes (~mu_0^-1).
    """
    nr, nz = 16, 8
    dr = 1e-3
    dz = 1e-3
    rho0 = 1e-4
    p0 = 1e3
    B_theta_amp = 5e-7

    solver = CylindricalMHDSolver(
        nr=nr, nz=nz, dr=dr, dz=dz,
        gamma=5.0 / 3.0,
        enable_hall=False,
        enable_resistive=True,
        enable_energy_equation=True,
        conservative_energy=True,
        riemann_solver="hll",
        time_integrator="ssp_rk2",
        use_godunov_flux=False,
    )

    state = _build_zero_velocity_state(nr, nz, dr, dz, rho0, p0, B_theta_amp)
    rho_2d = solver._squeeze(state["rho"])
    vel_2d = solver._squeeze(state["velocity"])
    p_2d = solver._squeeze(state["pressure"])
    B_2d = solver._squeeze(state["B"])
    psi_2d = solver._squeeze(state["psi"])
    eta_field = np.ones((nr, nz), dtype=np.float64)

    rhs = solver._compute_rhs(
        rho_2d, vel_2d, p_2d, B_2d, psi_2d,
        eta_field=eta_field, source_terms=None, e_electron=None,
    )

    dp_dt = rhs["dp_dt"]
    dB_dt = rhs["dB_dt"]
    ohmic_heating = rhs["ohmic_heating"]
    B_dot_dB = np.sum(B_2d * dB_dt, axis=0)

    gamma_minus_1 = solver.gamma - 1.0
    correct_dp_dt = gamma_minus_1 * (ohmic_heating - B_dot_dB / mu_0)
    wrong_dp_dt = gamma_minus_1 * (ohmic_heating - B_dot_dB)

    # The correct and wrong formulas must differ on the magnetic term by
    # a factor close to mu_0^-1 ~ 7.96e5.  Their difference is
    # gamma_minus_1 * B_dot_dB * (1 - 1/mu_0), so
    # |correct - wrong| should equal gamma_minus_1 * (1/mu_0 - 1) * |B_dot_dB|.
    diff_correct_wrong = np.max(np.abs(correct_dp_dt - wrong_dp_dt))
    expected_ratio = gamma_minus_1 * (1.0 / mu_0 - 1.0) * np.max(np.abs(B_dot_dB))
    assert abs(diff_correct_wrong / max(expected_ratio, 1e-300) - 1.0) < 1e-6, (
        f"Sanity: correct minus wrong should be (gamma-1)(1/mu_0 - 1) * |B.dB/dt|. "
        f"Got diff = {diff_correct_wrong:.3e}, expected = {expected_ratio:.3e}."
    )

    # And the actual solver's dp_dt must align with the CORRECT formula,
    # not the wrong one:
    err_vs_correct = np.max(np.abs(dp_dt - correct_dp_dt))
    err_vs_wrong = np.max(np.abs(dp_dt - wrong_dp_dt))
    assert err_vs_correct < 1e-10 * np.max(np.abs(correct_dp_dt)), (
        f"dp_dt does not match correct formula: err_vs_correct = {err_vs_correct:.3e}"
    )
    assert err_vs_wrong > 100.0 * err_vs_correct, (
        f"dp_dt is suspiciously close to the wrong (no /mu_0) formula: "
        f"err_vs_correct = {err_vs_correct:.3e}, err_vs_wrong = {err_vs_wrong:.3e}. "
        f"This suggests the /mu_0 factor was dropped."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
