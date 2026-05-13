"""Formula audit checks for local-KR MHD/circuit formulas."""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import mu_0
from dpf.geometry.cylindrical import CylindricalGeometry


def test_cylindrical_geometry_hoop_stress_is_inward_for_toroidal_field() -> None:
    """For pure B_theta, the FV radial source is p/r - B_theta^2/(2*mu0*r)."""
    geom = CylindricalGeometry(nr=4, nz=2, dr=0.01, dz=0.02)
    rho = np.ones((4, 2))
    velocity = np.zeros((3, 4, 2))
    pressure = np.full((4, 2), 3.0)
    B = np.zeros((3, 4, 2))
    B[1] = 0.5

    source = geom.geometric_source_momentum(rho, velocity, pressure, B)
    ir = 2
    r = (ir + 0.5) * 0.01
    expected = (3.0 - 0.5 * 0.5**2 / mu_0) / r

    np.testing.assert_allclose(source[0, ir, 0], expected, rtol=1.0e-12)


def test_mlx_geometric_source_momentum_is_not_density_multiplied_twice() -> None:
    mlx = pytest.importorskip("mlx.core")

    from dpf.metal.mlx_kernels import IDN, IEN, IMR
    from dpf.metal.mlx_sources import apply_geometric_sources

    gamma = 5.0 / 3.0
    nr, nz = 4, 2
    rho = 2.0
    vt = 3.0
    pressure = 5.0
    dt = 1.0e-9
    r_cell = mlx.array(np.array([0.005, 0.015, 0.025, 0.035], dtype=np.float32))
    inv_r = 1.0 / r_cell
    U = mlx.zeros((10, nr, nz), dtype=mlx.float32)
    U = mlx.concatenate(
        [
            mlx.full((1, nr, nz), rho),
            mlx.zeros((1, nr, nz)),
            mlx.zeros((1, nr, nz)),
            mlx.full((1, nr, nz), rho * vt),
            mlx.full((1, nr, nz), pressure / (gamma - 1.0) + 0.5 * rho * vt * vt),
            mlx.zeros((5, nr, nz)),
        ],
        axis=0,
    )

    U_out = apply_geometric_sources(U, r_cell, inv_r, dt, gamma=gamma, use_metal_kernel=False)
    ir = 2
    expected_dmr = (pressure + rho * vt * vt) / float(r_cell[ir]) * dt
    actual_dmr = float(U_out[IMR, ir, 0] - U[IMR, ir, 0])

    np.testing.assert_allclose(actual_dmr, expected_dmr, rtol=1.0e-5)
    np.testing.assert_allclose(np.asarray(U_out[IDN]), np.asarray(U[IDN]))
    assert np.all(np.isfinite(np.asarray(U_out[IEN])))


def test_circuit_coupler_dlpdt_does_not_duplicate_inductive_back_emf() -> None:
    from dpf.circuit.coupler import BACK_EMF_CLAMP_V, _clamp_dlpdt_to_back_emf

    current = 2.0e6
    dlpdt, back_emf = _clamp_dlpdt_to_back_emf(current, 1.0)

    assert abs(dlpdt * current) <= BACK_EMF_CLAMP_V
    assert back_emf == 0.0
