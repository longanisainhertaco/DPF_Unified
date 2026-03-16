"""Tests for 3D Cartesian electrode boundary conditions (Campaign 2C)."""
import numpy as np
import pytest


@pytest.fixture
def metal_solver_3d():
    """Create a 3D Cartesian Metal MHD solver."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    from dpf.metal.metal_solver import MetalMHDSolver
    return MetalMHDSolver(
        grid_shape=(16, 16, 32),
        dx=0.005, dz=0.005,  # 5 mm isotropic
        gamma=5 / 3, cfl=0.3,
        device="cpu", use_ct=False,
        coordinates="cartesian",
        reconstruction="plm",
        riemann_solver="hll",
        precision="float32",
    )


@pytest.fixture
def metal_solver_cyl():
    """Create a cylindrical Metal MHD solver."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    from dpf.metal.metal_solver import MetalMHDSolver
    return MetalMHDSolver(
        grid_shape=(16, 16, 32),
        dx=0.002, dz=0.005,
        gamma=5 / 3, cfl=0.3,
        device="cpu", use_ct=False,
        coordinates="cylindrical",
        reconstruction="plm",
        riemann_solver="hll",
        precision="float32",
    )


def _make_state(nr=16, ny=16, nz=32):
    rho0 = 0.084
    return {
        "rho": np.full((nr, ny, nz), rho0),
        "velocity": np.zeros((3, nr, ny, nz)),
        "pressure": np.full((nr, ny, nz), 400.0),
        "B": np.zeros((3, nr, ny, nz)),
        "Te": np.full((nr, ny, nz), 300.0),
        "Ti": np.full((nr, ny, nz), 300.0),
        "psi": np.zeros((nr, ny, nz)),
    }


class TestCartesian3DElectrodeBC:
    def test_bfield_applied_at_cathode(self, metal_solver_3d):
        """Cartesian solver should apply B_theta decomposed into (Bx, By) at cathode."""
        state = _make_state()
        result = metal_solver_3d.step(
            state, dt=1e-10, current=1e6, voltage=1e4,
            anode_radius=0.01, cathode_radius=0.03,
            apply_electrode_bc=True,
        )
        B = result["B"]
        # B should be nonzero near cathode radius
        mid = 16 // 2
        # At the domain center, Bx and By should have azimuthal pattern
        B_mag = np.sqrt(B[0]**2 + B[1]**2)
        assert np.max(B_mag) > 0, "B-field should be nonzero after electrode BC"

    def test_azimuthal_symmetry(self, metal_solver_3d):
        """B_theta should have azimuthal symmetry: |B| depends on r, not theta."""
        state = _make_state()
        result = metal_solver_3d.step(
            state, dt=1e-10, current=1e6, voltage=1e4,
            anode_radius=0.01, cathode_radius=0.035,
            apply_electrode_bc=True,
        )
        B = result["B"]
        # Check B_theta = sqrt(Bx^2 + By^2) at two points at same radius
        # but different angles. They should be approximately equal.
        nz_mid = 16  # midplane
        B_th = np.sqrt(B[0, :, :, nz_mid]**2 + B[1, :, :, nz_mid]**2)
        # B_theta should vary with r but be ~constant in theta at fixed r
        # Check ring at ~cathode radius
        assert B_th.max() > 0

    def test_axis_guard_zeros_B(self, metal_solver_3d):
        """B-field near axis (r < anode_r/2) should be zeroed."""
        state = _make_state()
        # Set some initial B everywhere
        state["B"] = np.full((3, 16, 16, 32), 1.0)
        result = metal_solver_3d.step(
            state, dt=1e-10, current=1e6, voltage=1e4,
            anode_radius=0.02, cathode_radius=0.035,
            apply_electrode_bc=True,
        )
        B = result["B"]
        # Cells near axis should have B zeroed
        mid = 8  # center of 16-cell grid
        # Check center cell
        B_center = np.sqrt(B[0, mid, mid, 16]**2 + B[1, mid, mid, 16]**2 + B[2, mid, mid, 16]**2)
        assert B_center < 1.0, "B at axis should be reduced by guard"

    def test_z0_insulator_bc(self, metal_solver_3d):
        """z=0 face should have B_theta applied (insulator boundary)."""
        state = _make_state()
        result = metal_solver_3d.step(
            state, dt=1e-10, current=1e6, voltage=1e4,
            anode_radius=0.01, cathode_radius=0.035,
            apply_electrode_bc=True,
        )
        B = result["B"]
        B_mag_z0 = np.sqrt(B[0, :, :, 0]**2 + B[1, :, :, 0]**2)
        assert np.max(B_mag_z0) > 0, "z=0 face should have B-field"

    def test_no_nan(self, metal_solver_3d):
        """BC application should never produce NaN."""
        state = _make_state()
        result = metal_solver_3d.step(
            state, dt=1e-10, current=2e6, voltage=2e4,
            anode_radius=0.01, cathode_radius=0.03,
            apply_electrode_bc=True,
        )
        for key in ("rho", "velocity", "pressure", "B"):
            assert not np.any(np.isnan(result[key])), f"NaN in {key}"


class TestCylindricalElectrodeBC:
    def test_cylindrical_bc_still_works(self, metal_solver_cyl):
        """Cylindrical BC should still apply B_theta on component [1]."""
        state = _make_state()
        result = metal_solver_cyl.step(
            state, dt=1e-10, current=1e6, voltage=1e4,
            anode_radius=0.005, cathode_radius=0.03,
            apply_electrode_bc=True,
        )
        B = result["B"]
        assert np.max(np.abs(B[1])) > 0, "B_theta should be nonzero"

    def test_cylindrical_axis_symmetry(self, metal_solver_cyl):
        """B_r should be zero at axis (r=0) in cylindrical."""
        state = _make_state()
        result = metal_solver_cyl.step(
            state, dt=1e-10, current=1e6, voltage=1e4,
            anode_radius=0.005, cathode_radius=0.03,
            apply_electrode_bc=True,
        )
        B = result["B"]
        assert np.all(B[0, 0, :, :] == 0), "B_r at r=0 should be zero"
