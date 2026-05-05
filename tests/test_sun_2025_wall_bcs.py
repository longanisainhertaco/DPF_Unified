"""Sun 2025 §2.4 Eq.18 wall BC unit tests.

Verbatim Eq.18 (electrode surfaces AF, EF, BC):
    ∂B/∂n = 0,  ∂Ti/∂n = 0,  ∂Te/∂n = 0,  u_n = 0

This test verifies the opt-in `enable_sun2025_wall_bcs=True` enforcement on
the CylindricalMHDSolver. Default behavior (flag False) must be unchanged.

KR: 2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md §2.4 lines 516-537
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver


def _build_state(nr: int, nz: int) -> dict[str, np.ndarray]:
    rho = np.full((nr, 1, nz), 1e-3)
    vel = np.zeros((3, nr, 1, nz))
    p = np.full((nr, 1, nz), 1e2)
    B = np.zeros((3, nr, 1, nz))
    Te = np.full((nr, 1, nz), 1.0e4)
    Ti = np.full((nr, 1, nz), 1.0e4)
    psi = np.zeros((nr, 1, nz))
    return {
        "rho": rho,
        "velocity": vel,
        "pressure": p,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": psi,
    }


def _make_solver(enable_bc: bool, inlet_active: bool = False) -> CylindricalMHDSolver:
    return CylindricalMHDSolver(
        nr=32,
        nz=16,
        dr=1e-3,
        dz=1e-3,
        gamma=5.0 / 3.0,
        cfl=0.3,
        enable_hall=False,
        enable_resistive=False,
        enable_energy_equation=True,
        conservative_energy=True,
        use_godunov_flux=True,
        time_integrator="ssp_rk3",
        riemann_solver="hll",
        enable_sun2025_wall_bcs=enable_bc,
        inlet_low_v_low_rho=inlet_active,
    )


class TestSun2025WallBCs:
    """Eq.18 enforcement: u_n=0, ∂Te/∂n=∂Ti/∂n=0 on AF/EF/BC."""

    def test_normal_velocity_zeroed_on_anode_column(self) -> None:
        """u_n (= v_r at anode) must be zero on anode column after a step."""
        solver = _make_solver(enable_bc=True)
        state = _build_state(nr=solver.nr, nz=solver.nz)

        anode_radius = 5e-3
        cathode_radius = 25e-3
        idx_anode = int(np.argmin(np.abs(solver.geom.r - anode_radius)))

        # Inject non-zero v_r at the anode column to verify the BC clamps it.
        state["velocity"][0, idx_anode, :, :] = 1.0e4

        out = solver.step(
            state,
            dt=1e-10,
            current=0.0,
            voltage=0.0,
            anode_radius=anode_radius,
            cathode_radius=cathode_radius,
            apply_electrode_bc=False,
        )

        v_r_anode = out["velocity"][0, idx_anode, :, :]
        max_abs = float(np.max(np.abs(v_r_anode)))
        assert max_abs == 0.0, (
            f"Sun 2025 Eq.18 u_n=0 violated on anode: max|v_r| = {max_abs:.3e}"
        )

    def test_normal_velocity_zeroed_on_cathode_column(self) -> None:
        """u_n (= v_r at cathode) must be zero on cathode column after a step."""
        solver = _make_solver(enable_bc=True)
        state = _build_state(nr=solver.nr, nz=solver.nz)

        anode_radius = 5e-3
        cathode_radius = 25e-3
        idx_cath = int(np.argmin(np.abs(solver.geom.r - cathode_radius)))

        state["velocity"][0, idx_cath, :, :] = -1.0e4

        out = solver.step(
            state,
            dt=1e-10,
            current=0.0,
            voltage=0.0,
            anode_radius=anode_radius,
            cathode_radius=cathode_radius,
            apply_electrode_bc=False,
        )

        v_r_cath = out["velocity"][0, idx_cath, :, :]
        max_abs = float(np.max(np.abs(v_r_cath)))
        assert max_abs == 0.0, (
            f"Sun 2025 Eq.18 u_n=0 violated on cathode: max|v_r| = {max_abs:.3e}"
        )

    def test_te_gradient_zero_on_anode_column(self) -> None:
        """∂Te/∂n = 0 on anode: Te[idx_anode] must equal Te[idx_anode+1]."""
        solver = _make_solver(enable_bc=True)
        state = _build_state(nr=solver.nr, nz=solver.nz)

        anode_radius = 5e-3
        cathode_radius = 25e-3
        idx_anode = int(np.argmin(np.abs(solver.geom.r - anode_radius)))

        # Set a Te ramp across the anode column so ∂Te/∂n is initially non-zero.
        ramp = np.linspace(2.0e4, 5.0e4, solver.nr)[:, np.newaxis, np.newaxis]
        state["Te"] = np.broadcast_to(ramp, (solver.nr, 1, solver.nz)).copy()
        state["Ti"] = state["Te"].copy()

        out = solver.step(
            state,
            dt=1e-10,
            current=0.0,
            voltage=0.0,
            anode_radius=anode_radius,
            cathode_radius=cathode_radius,
            apply_electrode_bc=False,
        )

        Te = out["Te"][:, 0, :]
        # Zero-gradient: Te on the anode column copied from the radial-interior neighbour.
        diff_anode = float(np.max(np.abs(Te[idx_anode, :] - Te[idx_anode + 1, :])))
        assert diff_anode == 0.0, (
            f"Sun 2025 Eq.18 ∂Te/∂n=0 violated on anode column: "
            f"max|ΔTe| = {diff_anode:.3e}"
        )

    def test_ti_gradient_zero_on_cathode_column(self) -> None:
        """∂Ti/∂n = 0 on cathode: Ti[idx_cath] must equal Ti[idx_cath-1]."""
        solver = _make_solver(enable_bc=True)
        state = _build_state(nr=solver.nr, nz=solver.nz)

        anode_radius = 5e-3
        cathode_radius = 25e-3
        idx_cath = int(np.argmin(np.abs(solver.geom.r - cathode_radius)))

        ramp = np.linspace(2.0e4, 5.0e4, solver.nr)[:, np.newaxis, np.newaxis]
        state["Te"] = np.broadcast_to(ramp, (solver.nr, 1, solver.nz)).copy()
        state["Ti"] = state["Te"].copy()

        out = solver.step(
            state,
            dt=1e-10,
            current=0.0,
            voltage=0.0,
            anode_radius=anode_radius,
            cathode_radius=cathode_radius,
            apply_electrode_bc=False,
        )

        Ti = out["Ti"][:, 0, :]
        diff_cath = float(np.max(np.abs(Ti[idx_cath, :] - Ti[idx_cath - 1, :])))
        assert diff_cath == 0.0, (
            f"Sun 2025 Eq.18 ∂Ti/∂n=0 violated on cathode column: "
            f"max|ΔTi| = {diff_cath:.3e}"
        )

    def test_default_behavior_unchanged_when_flag_off(self) -> None:
        """With enable_sun2025_wall_bcs=False, anode v_r is NOT zeroed."""
        solver = _make_solver(enable_bc=False)
        state = _build_state(nr=solver.nr, nz=solver.nz)

        anode_radius = 5e-3
        cathode_radius = 25e-3
        idx_anode = int(np.argmin(np.abs(solver.geom.r - anode_radius)))

        state["velocity"][0, idx_anode, :, :] = 1.0e4

        out = solver.step(
            state,
            dt=1e-10,
            current=0.0,
            voltage=0.0,
            anode_radius=anode_radius,
            cathode_radius=cathode_radius,
            apply_electrode_bc=False,
        )

        # Without the flag, the BC is not applied; v_r should still be non-zero
        # at the anode column (advection alone over 1e-10 s won't drain it).
        v_r_anode = out["velocity"][0, idx_anode, :, :]
        max_abs = float(np.max(np.abs(v_r_anode)))
        assert max_abs > 0.0, (
            f"Default-off path unexpectedly zeroed v_r on anode: max|v_r| = {max_abs:.3e}"
        )

    def test_closed_face_zeroed_when_inlet_inactive(self) -> None:
        """When inlet_low_v_low_rho=False, closed face z=0 wall: v_z=0 on annulus."""
        solver = _make_solver(enable_bc=True, inlet_active=False)
        state = _build_state(nr=solver.nr, nz=solver.nz)

        anode_radius = 5e-3
        cathode_radius = 25e-3
        idx_anode = int(np.argmin(np.abs(solver.geom.r - anode_radius)))
        idx_cath = int(np.argmin(np.abs(solver.geom.r - cathode_radius)))

        state["velocity"][2, :, :, 0] = 1.0e4  # non-zero v_z on z=0 face

        out = solver.step(
            state,
            dt=1e-10,
            current=0.0,
            voltage=0.0,
            anode_radius=anode_radius,
            cathode_radius=cathode_radius,
            apply_electrode_bc=False,
        )

        v_z_face = out["velocity"][2, idx_anode:idx_cath + 1, 0, 0]
        max_abs = float(np.max(np.abs(v_z_face)))
        assert max_abs == 0.0, (
            f"Sun 2025 Eq.18 u_n=0 violated on z=0 closed face: max|v_z| = {max_abs:.3e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
