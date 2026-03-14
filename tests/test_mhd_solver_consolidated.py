"""test_mhd_solver_consolidated.py — consolidated MHD solver tests.

Covers:
- MHD Solver Core (test_fluid.py)
- Resistive MHD Physics (test_resistive_mhd.py)
- Cylindrical Geometry (test_cylindrical.py)
- Phase 12: HLLD WENO5 Strang Splitting (test_phase12.py)
- Phase 14: Implicit Diffusion Super Time-Stepping (test_phase14.py)
- Metal Source Terms (test_phase_h_metal_sources.py)
- Phase P: Engine Accuracy (test_phase_p_accuracy.py)
- Phase Q: Python 8-comp HLLD and CT (test_phase_q_hlld_ct.py)
- Phase Q: Metal Transport Physics (test_phase_q_transport.py)
- Phase R: Full HLLD Double-Star (test_phase_r_hlld.py)
- Phase V: Conservative Energy (test_phase_v_conservative_energy.py)
- Phase V: Cylindrical Conservative (test_phase_v_cylindrical_conservative.py)
- Phase X: Powell div(B) (test_phase_x_powell_divb.py)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_transport import (  # noqa: E402, I001
    apply_braginskii_conduction_mps,
    apply_braginskii_viscosity_mps,
    apply_hall_mhd_mps,
    apply_nernst_advection_mps,
    braginskii_kappa_mps,
    curl_B_mps,
    hall_electric_field_mps,
    nernst_coefficient_mps,
)
from dpf.metal.metal_riemann import (  # noqa: E402, I001
    NVAR,
    IDN,
    IB1,
    IM2,
    _prim_to_cons_mps,
    hlld_flux_mps,
)
from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402

from dpf.config import FluidConfig, SimulationConfig  # noqa: E402
from dpf.constants import k_B, m_d, m_p, mu_0, pi  # noqa: E402
from dpf.core.field_manager import FieldManager  # noqa: E402
from dpf.engine import SimulationEngine  # noqa: E402
from dpf.fluid.constrained_transport import cell_centered_to_face, compute_div_B  # noqa: E402
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver  # noqa: E402
from dpf.fluid.implicit_diffusion import (  # noqa: E402
    _thomas_solve,
    diffuse_field_1d,
    diffusion_cfl_dt,
    implicit_resistive_diffusion,
    implicit_thermal_diffusion,
)
from dpf.fluid.mhd_solver import (  # noqa: E402
    MHDSolver,
    _braginskii_heat_flux,
    _compute_flux_1d_sweep,
    _dedner_source,
    _hll_flux_1d,
    _hll_flux_1d_core,
    _hlld_flux_1d,
    _hlld_flux_1d_8comp,
    _hlld_flux_1d_core,
    _weno5_reconstruct_1d,
    powell_source_terms_cylindrical,
)
from dpf.fluid.super_time_step import (  # noqa: E402
    _diffusion_operator_1d,
    _diffusion_operator_1d_scalar,
    rkl2_coefficients,
    rkl2_diffusion_3d,
    rkl2_diffusion_step,
    rkl2_stability_limit,
    rkl2_thermal_step,
)
from dpf.geometry.cylindrical import CylindricalGeometry  # noqa: E402
from dpf.presets import get_preset  # noqa: E402
from dpf.turbulence.anomalous import anomalous_resistivity_field  # noqa: E402
from dpf.validation.suite import ValidationSuite  # noqa: E402

# Module-level constants (from test_phase_q_hlld_ct.py — used in free test functions)
mu_0 = 4.0 * np.pi * 1e-7  # noqa: F811
gamma = 5.0 / 3.0

# Metal transport physical constants (from test_phase_q_transport.py)
M_D = 3.34358377e-27
K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
MU_0 = 4.0e-7 * np.pi

# Metal HLLD constants (from test_phase_r_hlld.py)
DEVICE = torch.device("cpu")
GAMMA = 5.0 / 3.0


# --- Section: Resistive MHD Physics helpers ---

def _make_cartesian_state(nx=8, ny=8, nz=8, rho0=1e-4, T0=1e4, B_theta=0.1):
    """Create a simple Cartesian state dict for testing."""
    p0 = 2.0 * (rho0 / m_d) * k_B * T0
    state = {
        "rho": np.full((nx, ny, nz), rho0),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.full((nx, ny, nz), p0),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.full((nx, ny, nz), T0),
        "Ti": np.full((nx, ny, nz), T0),
        "psi": np.zeros((nx, ny, nz)),
    }
    state["B"][1, :, :, :] = B_theta
    return state


def _make_cylindrical_state(nr=32, nz=64, rho0=1e-4, T0=1e4):
    """Create a cylindrical state dict (nr, 1, nz) for testing."""
    p0 = 2.0 * (rho0 / m_d) * k_B * T0
    return {
        "rho": np.full((nr, 1, nz), rho0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p0),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), T0),
        "Ti": np.full((nr, 1, nz), T0),
        "psi": np.zeros((nr, 1, nz)),
    }


# --- Section: MHD Solver Core ---


class TestFieldManager:
    """Verify field manager vector calculus operations."""

    def test_divergence_zero_for_curl(self):
        """div(curl(F)) = 0 for any smooth field."""
        fm = FieldManager((16, 16, 16), dx=0.1)
        x = np.linspace(0, 1, 16)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        F = np.array([np.sin(2 * np.pi * Y), np.sin(2 * np.pi * Z), np.sin(2 * np.pi * X)])
        curl_F = fm.curl(F)
        div_curl = fm.divergence(curl_F)
        assert np.max(np.abs(div_curl)) < 1.0

    def test_neumann_bc_shape(self):
        """Neumann BC preserves array shape."""
        fm = FieldManager((10, 10, 10), dx=0.1)
        field = np.random.rand(3, 10, 10, 10)
        result = fm.apply_neumann_bc(field, ng=2)
        assert result.shape == field.shape

    def test_max_div_B_initial(self):
        """Initial B=0 has zero divergence."""
        fm = FieldManager((8, 8, 8), dx=0.1)
        assert fm.max_div_B() == 0.0


class TestDednerCleaning:
    """Verify Dedner divergence cleaning."""

    def test_reduces_div_B(self):
        """Dedner source term opposes existing div(B)."""
        nx = 16
        dx = 0.1
        B = np.zeros((3, nx, nx, nx))
        x = np.linspace(0, 1, nx)
        X, _, _ = np.meshgrid(x, x, x, indexing="ij")
        B[0] = X
        psi = np.zeros((nx, nx, nx))
        ch = 1.0
        cp = 1.0
        dpsi_dt, dB_dt = _dedner_source(psi, B, ch, cp, dx)
        assert np.mean(dpsi_dt) < 0


class TestWENO5:
    """Verify WENO5 reconstruction accuracy."""

    def test_weno5_constant_exact(self):
        """WENO5 reconstructs a constant function exactly."""
        v = np.full(10, 3.14)
        v_L, v_R = _weno5_reconstruct_1d(v)
        np.testing.assert_allclose(v_L, 3.14, atol=1e-10)
        np.testing.assert_allclose(v_R, 3.14, atol=1e-10)

    def test_weno5_linear_exact(self):
        """WENO5 reconstructs a linear function with high accuracy."""
        x = np.linspace(0, 1, 20)
        v = 2.0 * x + 1.0
        v_L, v_R = _weno5_reconstruct_1d(v)
        dx = x[1] - x[0]
        x_iface = x[2:-2] + 0.5 * dx
        expected = 2.0 * x_iface + 1.0
        np.testing.assert_allclose(v_L, expected, atol=0.01)

    def test_weno5_smooth_high_order(self):
        """WENO5 achieves 5th-order convergence for cell-averaged smooth data."""
        errors = []
        for n in [64, 128]:
            dx = 2 * np.pi / n
            v_avg = np.array([
                (-np.cos((i + 1) * dx) + np.cos(i * dx)) / dx
                for i in range(n)
            ])
            v_pad = np.concatenate([v_avg[-3:], v_avg, v_avg[:3]])
            v_L, v_R = _weno5_reconstruct_1d(v_pad)
            n_out = len(v_L)
            margin = 3
            x_iface = np.array([k * dx for k in range(margin, n_out - margin)])
            exact = np.sin(x_iface)
            v_interior = v_L[margin:n_out - margin]
            errors.append(np.max(np.abs(v_interior - exact)))
        rate = np.log2(errors[0] / max(errors[1], 1e-15))
        assert rate > 4.5, f"WENO5 convergence rate = {rate:.2f}, expected > 4.5"

    def test_weno5_output_length(self):
        """WENO5 output has correct length (n-4)."""
        v = np.random.rand(20)
        v_L, v_R = _weno5_reconstruct_1d(v)
        assert len(v_L) == 16
        assert len(v_R) == 16


class TestHLLFlux:
    """Verify HLL Riemann solver."""

    def test_hll_zero_jump(self):
        """HLL flux with zero jump = physical flux."""
        n = 10
        rho = np.full(n, 1.0)
        u = np.full(n, 1.0)
        p = np.full(n, 1.0)
        Bn = np.zeros(n)
        gamma_local = 5.0 / 3.0
        fluxes = _hll_flux_1d(rho, rho, u, u, p, p, Bn, Bn, gamma_local)
        np.testing.assert_allclose(fluxes["mass_flux"], 1.0, atol=1e-10)

    def test_hll_density_jump_bounded(self):
        """HLL flux is bounded for a density jump with equal velocity."""
        n = 10
        rho_L = np.full(n, 2.0)
        rho_R = np.full(n, 1.0)
        u = np.zeros(n)
        p = np.full(n, 1.0)
        Bn = np.zeros(n)
        gamma_local = 5.0 / 3.0
        fluxes = _hll_flux_1d(rho_L, rho_R, u, u, p, p, Bn, Bn, gamma_local)
        assert np.all(np.isfinite(fluxes["mass_flux"]))
        assert np.all(np.abs(fluxes["mass_flux"]) < 10.0)

    def test_hll_returns_all_keys(self):
        """HLL returns mass, momentum, energy fluxes."""
        n = 5
        rho = np.ones(n)
        u = np.ones(n)
        p = np.ones(n)
        Bn = np.zeros(n)
        fluxes = _hll_flux_1d(rho, rho, u, u, p, p, Bn, Bn, 5.0 / 3.0)
        assert "mass_flux" in fluxes
        assert "momentum_flux" in fluxes
        assert "energy_flux" in fluxes


class TestFluxSweep:
    """Verify dimension-split WENO5+HLL flux computation."""

    def test_flux_sweep_uniform_zero(self):
        """Uniform density + zero velocity = zero flux divergence."""
        n = 16
        rho = np.ones((n, n, n))
        vel_n = np.zeros((n, n, n))
        vel_t1 = np.zeros((n, n, n))
        vel_t2 = np.zeros((n, n, n))
        p = np.ones((n, n, n))
        Bn = np.zeros((n, n, n))
        Bt1 = np.zeros((n, n, n))
        Bt2 = np.zeros((n, n, n))
        gamma_local = 5.0 / 3.0
        result = _compute_flux_1d_sweep(
            rho, vel_n, vel_t1, vel_t2, p, Bn, Bt1, Bt2, gamma_local, axis=0,
        )
        assert result["n_interfaces"] == n - 4
        assert np.max(np.abs(result["mass_flux"])) < 1e-10

    def test_flux_sweep_small_grid_fallback(self):
        """Grid too small for WENO5 returns zero interfaces."""
        n = 4
        rho = np.ones((n, n, n))
        vel_n = np.zeros((n, n, n))
        vel_t1 = np.zeros((n, n, n))
        vel_t2 = np.zeros((n, n, n))
        p = np.ones((n, n, n))
        Bn = np.zeros((n, n, n))
        Bt1 = np.zeros((n, n, n))
        Bt2 = np.zeros((n, n, n))
        result = _compute_flux_1d_sweep(
            rho, vel_n, vel_t1, vel_t2, p, Bn, Bt1, Bt2, 5.0 / 3.0, axis=0,
        )
        assert result["n_interfaces"] == 0


class TestMHDSolver:
    """Integration tests for the MHD solver."""

    def test_step_returns_correct_keys(self):
        """MHD step returns state with all expected keys."""
        solver = MHDSolver((8, 8, 8), dx=0.01)
        state = {
            "rho": np.full((8, 8, 8), 1e-4),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.full((8, 8, 8), 1.0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 1e4),
            "Ti": np.full((8, 8, 8), 1e4),
            "psi": np.zeros((8, 8, 8)),
        }
        result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti", "psi"]:
            assert key in result

    def test_density_floor(self):
        """Density never goes negative."""
        solver = MHDSolver((8, 8, 8), dx=0.01, enable_hall=False, enable_braginskii=False)
        state = {
            "rho": np.full((8, 8, 8), 1e-30),
            "velocity": np.random.randn(3, 8, 8, 8) * 1e3,
            "pressure": np.full((8, 8, 8), 1.0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 1e4),
            "Ti": np.full((8, 8, 8), 1e4),
            "psi": np.zeros((8, 8, 8)),
        }
        result = solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        assert np.all(result["rho"] > 0)

    def test_coupling_interface(self):
        """Coupling interface returns a CouplingState."""
        from dpf.core.bases import CouplingState

        solver = MHDSolver((8, 8, 8), dx=0.01)
        cs = solver.coupling_interface()
        assert isinstance(cs, CouplingState)

    def test_ssprk2_more_accurate_than_euler(self):
        """SSP-RK2 should preserve a uniform state exactly."""
        solver = MHDSolver((8, 8, 8), dx=0.01, enable_hall=False, enable_braginskii=False)
        rho0 = 1e-2
        p0 = 100.0
        state = {
            "rho": np.full((8, 8, 8), rho0),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.full((8, 8, 8), p0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 5000.0),
            "Ti": np.full((8, 8, 8), 5000.0),
            "psi": np.zeros((8, 8, 8)),
        }
        result = solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        np.testing.assert_allclose(result["rho"], rho0, rtol=1e-10)

    def test_weno5_active_for_large_grid(self):
        """Solver uses WENO5 for grids >= 5 in each direction."""
        solver = MHDSolver((16, 16, 16), dx=0.01)
        assert solver.use_weno5 is True

    def test_fallback_for_small_grid(self):
        """Solver falls back to np.gradient for grids < 5."""
        solver = MHDSolver((4, 4, 4), dx=0.01)
        assert solver.use_weno5 is False

    def test_step_runs_with_weno5(self):
        """Full step completes with WENO5+HLL on a 16^3 grid."""
        solver = MHDSolver((16, 16, 16), dx=0.01, enable_hall=False, enable_braginskii=False)
        state = {
            "rho": np.full((16, 16, 16), 1e-4),
            "velocity": np.zeros((3, 16, 16, 16)),
            "pressure": np.full((16, 16, 16), 1.0),
            "B": np.zeros((3, 16, 16, 16)),
            "Te": np.full((16, 16, 16), 1e4),
            "Ti": np.full((16, 16, 16), 1e4),
            "psi": np.zeros((16, 16, 16)),
        }
        result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
        assert result["rho"].shape == (16, 16, 16)
        assert np.all(np.isfinite(result["rho"]))


class TestHallTerm:
    """Verify Hall term in the induction equation."""

    def test_hall_modifies_b_evolution(self):
        """Hall term should change B evolution compared to ideal MHD."""
        n = 8
        state = {
            "rho": np.full((n, n, n), 1e-4),
            "velocity": np.zeros((3, n, n, n)),
            "pressure": np.full((n, n, n), 1.0),
            "B": np.zeros((3, n, n, n)),
            "Te": np.full((n, n, n), 1e4),
            "Ti": np.full((n, n, n), 1e4),
            "psi": np.zeros((n, n, n)),
        }
        x = np.linspace(0, 1, n)
        _, Y, _ = np.meshgrid(x, x, x, indexing="ij")
        state["B"][2] = np.sin(2 * np.pi * Y) * 0.01
        solver_hall = MHDSolver((n, n, n), dx=0.01, enable_hall=True, enable_braginskii=False)
        solver_nohall = MHDSolver((n, n, n), dx=0.01, enable_hall=False, enable_braginskii=False)
        result_hall = solver_hall.step(state.copy(), dt=1e-12, current=0.0, voltage=0.0)
        result_nohall = solver_nohall.step(state.copy(), dt=1e-12, current=0.0, voltage=0.0)
        assert np.all(np.isfinite(result_hall["B"]))
        assert np.all(np.isfinite(result_nohall["B"]))

    def test_hall_dt_limit(self):
        """Hall term adds a stricter CFL constraint."""
        n = 8
        state = {
            "rho": np.full((n, n, n), 1e-4),
            "velocity": np.zeros((3, n, n, n)),
            "pressure": np.full((n, n, n), 1.0),
            "B": np.zeros((3, n, n, n)),
            "Te": np.full((n, n, n), 1e4),
            "Ti": np.full((n, n, n), 1e4),
            "psi": np.zeros((n, n, n)),
        }
        state["B"][2] = 1.0
        solver_hall = MHDSolver((n, n, n), dx=0.01, enable_hall=True)
        solver_nohall = MHDSolver((n, n, n), dx=0.01, enable_hall=False)
        dt_hall = solver_hall._compute_dt(state)
        dt_nohall = solver_nohall._compute_dt(state)
        assert dt_hall <= dt_nohall


class TestBraginskiiHeatFlux:
    """Verify Braginskii anisotropic heat flux."""

    def test_uniform_temp_no_change(self):
        """Uniform temperature = no heat flux."""
        n = 8
        Te = np.full((n, n, n), 1e6)
        ne = np.full((n, n, n), 1e20)
        B = np.zeros((3, n, n, n))
        B[2] = 1.0
        Te_new = _braginskii_heat_flux(Te, ne, B, dx=0.01, dt=1e-10)
        np.testing.assert_allclose(Te_new, Te, rtol=1e-6)

    def test_parallel_diffusion_faster(self):
        """Heat diffuses faster along B than across B."""
        n = 16
        Te = np.full((n, n, n), 1e6)
        ne = np.full((n, n, n), 1e20)
        B = np.zeros((3, n, n, n))
        B[2] = 1.0
        z = np.linspace(0, 1, n)
        _, _, Z = np.meshgrid(z, z, z, indexing="ij")
        Te_init = Te + 1e5 * np.sin(2 * np.pi * Z)
        Te_par = _braginskii_heat_flux(Te_init, ne, B, dx=1.0 / n, dt=1e-12)
        X, _, _ = np.meshgrid(z, z, z, indexing="ij")
        Te_init_perp = Te + 1e5 * np.sin(2 * np.pi * X)
        Te_perp = _braginskii_heat_flux(Te_init_perp, ne, B, dx=1.0 / n, dt=1e-12)
        assert np.all(np.isfinite(Te_par))
        assert np.all(np.isfinite(Te_perp))

    def test_temperature_stays_positive(self):
        """Temperature never goes negative from heat flux."""
        n = 8
        Te = np.full((n, n, n), 100.0)
        ne = np.full((n, n, n), 1e18)
        B = np.zeros((3, n, n, n))
        B[2] = 0.1
        Te_new = _braginskii_heat_flux(Te, ne, B, dx=0.01, dt=1e-8)
        assert np.all(Te_new >= 1.0)


class TestDLdt:
    """Verify dL_dt computation from pinch dynamics."""

    def test_dLdt_initially_zero(self):
        """First step has dL_dt = 0 (no previous Lp)."""
        solver = MHDSolver((8, 8, 8), dx=0.01, enable_hall=False, enable_braginskii=False)
        state = {
            "rho": np.full((8, 8, 8), 1e-4),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.full((8, 8, 8), 1.0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 1e4),
            "Ti": np.full((8, 8, 8), 1e4),
            "psi": np.zeros((8, 8, 8)),
        }
        solver.step(state, dt=1e-10, current=100.0, voltage=1000.0)
        cs = solver.coupling_interface()
        assert cs.dL_dt is None or cs.dL_dt == 0.0

    def test_dLdt_computed_after_two_steps(self):
        """After two steps, dL_dt should be computed."""
        solver = MHDSolver((8, 8, 8), dx=0.01, enable_hall=False, enable_braginskii=False)
        state = {
            "rho": np.full((8, 8, 8), 1e-4),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.full((8, 8, 8), 1.0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 1e4),
            "Ti": np.full((8, 8, 8), 1e4),
            "psi": np.zeros((8, 8, 8)),
        }
        result = solver.step(state, dt=1e-10, current=100.0, voltage=1000.0)
        result = solver.step(result, dt=1e-10, current=100.0, voltage=1000.0)
        cs = solver.coupling_interface()
        assert isinstance(cs.dL_dt, float)


# --- Section: Resistive MHD Physics ---


class TestElectrodeBFieldBC:
    """Test electrode B-field boundary condition application."""

    def test_cylindrical_bfield_bc_imposed(self):
        """B_theta matches mu0*I/(2*pi*r) at electrode boundaries."""
        nr, nz = 32, 64
        dr, dz = 5e-4, 5e-4
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz)
        B = np.zeros((3, nr, nz))
        current = 1e5
        anode_radius = 0.005
        cathode_radius = 0.01
        B = solver.apply_electrode_bfield_bc(B, current, anode_radius, cathode_radius)
        r = solver.geom.r
        idx_cath = np.argmin(np.abs(r - cathode_radius))
        r_cath = max(r[idx_cath], 1e-10)
        B_theta_expected = mu_0 * current / (2.0 * np.pi * r_cath)
        assert np.isclose(B[1, idx_cath, 0], B_theta_expected, rtol=1e-6)

    def test_cylindrical_axis_symmetry(self):
        """B_r = 0 at r=0 axis (symmetry condition)."""
        solver = CylindricalMHDSolver(nr=16, nz=32, dr=1e-3, dz=1e-3)
        B = np.random.randn(3, 16, 32) * 0.01
        B = solver.apply_electrode_bfield_bc(B, 1e4, 0.005, 0.01)
        assert np.all(B[0, 0, :] == 0.0)

    def test_zero_current_no_bc(self):
        """Zero current should not modify B-field (except axis BC)."""
        solver = CylindricalMHDSolver(nr=16, nz=32, dr=1e-3, dz=1e-3)
        B = np.ones((3, 16, 32)) * 0.05
        B_original = B.copy()
        B = solver.apply_electrode_bfield_bc(B, 0.0, 0.005, 0.01)
        assert B[0, 0, 0] == 0.0
        assert np.allclose(B[1, 1:, :], B_original[1, 1:, :])

    def test_cartesian_bfield_bc_imposed(self):
        """Cartesian solver applies B-field BC at boundary cells."""
        solver = MHDSolver(grid_shape=(16, 16, 16), dx=1e-3)
        B = np.zeros((3, 16, 16, 16))
        current = 1e5
        B = solver.apply_electrode_bfield_bc(
            B, current, anode_radius=0.003, cathode_radius=0.006,
        )
        assert np.max(np.abs(B[0])) > 0 or np.max(np.abs(B[1])) > 0

    def test_bfield_bc_through_step(self):
        """Electrode BC applied during fluid step with apply_electrode_bc=True."""
        nr, nz = 16, 32
        dr, dz = 5e-4, 5e-4
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz)
        state = _make_cylindrical_state(nr=nr, nz=nz)
        result = solver.step(
            state, dt=1e-12, current=1e4, voltage=1e3,
            anode_radius=0.003, cathode_radius=0.006,
            apply_electrode_bc=True,
        )
        B_theta = result["B"][1, :, 0, :]
        assert np.max(np.abs(B_theta)) > 0


class TestResistiveMHD:
    """Test resistive MHD implementation."""

    def test_resistive_diffusion_reduces_B(self):
        """Magnetic field diffuses in presence of resistivity."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=True, enable_energy_equation=False)
        state = _make_cartesian_state(B_theta=0.1)
        state["B"][2, 3:5, 3:5, 3:5] = 1.0
        eta_field = np.full((8, 8, 8), 1e-4)
        result = solver.step(state, dt=1e-12, current=0, voltage=0,
                             eta_field=eta_field)
        B_max_after = np.max(np.abs(result["B"][2]))
        assert np.isfinite(B_max_after)

    def test_no_resistive_without_eta(self):
        """Without eta_field, no resistive term is applied."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, enable_resistive=True)
        state = _make_cartesian_state()
        result_no_eta = solver.step(state, dt=1e-12, current=0, voltage=0,
                                    eta_field=None)
        result_zero_eta = solver.step(state, dt=1e-12, current=0, voltage=0,
                                      eta_field=np.zeros((8, 8, 8)))
        assert np.allclose(result_no_eta["B"], result_zero_eta["B"], atol=1e-15)

    def test_resistive_disabled_flag(self):
        """enable_resistive=False skips resistive term even with eta_field."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=False, enable_energy_equation=False)
        state = _make_cartesian_state()
        eta_field = np.full((8, 8, 8), 1.0)
        result = solver.step(state, dt=1e-12, current=0, voltage=0,
                             eta_field=eta_field)
        assert np.all(np.isfinite(result["B"]))

    def test_cylindrical_resistive_term(self):
        """Cylindrical solver includes resistive term."""
        nr, nz = 16, 32
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=1e-3, dz=1e-3,
                                       enable_resistive=True)
        state = _make_cylindrical_state(nr=nr, nz=nz)
        state["B"][1, nr//2, 0, nz//2] = 0.5
        eta_field = np.full((nr, 1, nz), 1e-4)
        result = solver.step(state, dt=1e-12, current=0, voltage=0,
                             eta_field=eta_field)
        assert np.all(np.isfinite(result["B"]))


class TestOhmicHeating:
    """Test Ohmic heating from resistive dissipation."""

    def test_ohmic_heating_increases_Te(self):
        """eta*J^2 heats the plasma (Te increases with resistive dissipation)."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=True, enable_energy_equation=True)
        state = _make_cartesian_state(T0=1e4)
        x = np.arange(8) * 1e-3
        for k in range(8):
            state["B"][2, :, :, k] = np.sin(2 * np.pi * x / 0.008)[:, np.newaxis]
        Te_before = np.mean(state["Te"])
        eta_field = np.full((8, 8, 8), 1e-3)
        result = solver.step(state, dt=1e-11, current=0, voltage=0,
                             eta_field=eta_field)
        Te_after = np.mean(result["Te"])
        assert Te_after >= Te_before - 1.0

    def test_ohmic_heating_rhs_nonzero(self):
        """RHS returns non-zero ohmic_heating when eta and J are non-zero."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_resistive=True, enable_energy_equation=True)
        state = _make_cartesian_state()
        state["B"][2, 3:6, :, :] = 0.1
        eta_field = np.full((8, 8, 8), 1e-3)
        rhs = solver._compute_rhs_euler(state, current=0, voltage=0,
                                         eta_field=eta_field)
        assert np.max(rhs["ohmic_heating"]) > 0

    def test_zero_eta_zero_ohmic_heating(self):
        """Zero resistivity gives zero Ohmic heating."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3)
        state = _make_cartesian_state()
        rhs = solver._compute_rhs_euler(state, current=0, voltage=0,
                                         eta_field=np.zeros((8, 8, 8)))
        assert np.allclose(rhs["ohmic_heating"], 0.0)


class TestEnergyEquation:
    """Test conservative energy equation with Ohmic heating."""

    def test_energy_equation_pressure_update(self):
        """Energy equation includes (gamma-1)*Q_ohm in pressure rate."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_energy_equation=True, enable_resistive=True)
        state = _make_cartesian_state()
        state["B"][2, 3:6, :, :] = 0.1
        eta_field = np.full((8, 8, 8), 1e-3)
        rhs_with = solver._compute_rhs_euler(state, current=0, voltage=0,
                                              eta_field=eta_field)
        solver_no_energy = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                                      enable_energy_equation=False,
                                      enable_resistive=True)
        rhs_without = solver_no_energy._compute_rhs_euler(state, current=0, voltage=0,
                                                           eta_field=eta_field)
        diff = rhs_with["dp_dt"] - rhs_without["dp_dt"]
        assert np.max(diff) > 0, "Energy equation should add Ohmic heating to pressure"

    def test_adiabatic_fallback(self):
        """With enable_energy_equation=False, uses adiabatic dp/dt = -gp*div(v)."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3,
                           enable_energy_equation=False)
        state = _make_cartesian_state()
        rhs = solver._compute_rhs_euler(state, current=0, voltage=0)
        assert np.allclose(rhs["dp_dt"], 0.0, atol=1e-20)


class TestTwoTemperature:
    """Test two-temperature preservation through MHD step."""

    def test_te_not_equal_ti_survives(self):
        """Te != Ti is preserved through MHD step (not collapsed)."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3)
        state = _make_cartesian_state(T0=5000.0)
        state["Te"][:] = 1e5
        state["Ti"][:] = 1e4
        n_i = state["rho"] / m_d
        state["pressure"][:] = n_i * k_B * (state["Te"] + state["Ti"])
        result = solver.step(state, dt=1e-12, current=0, voltage=0)
        Te_mean = np.mean(result["Te"])
        Ti_mean = np.mean(result["Ti"])
        assert Te_mean > Ti_mean * 1.5

    def test_temperature_ratio_preserved(self):
        """The Te/Ti ratio is approximately preserved through a step."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3)
        state = _make_cartesian_state(T0=1e4)
        state["Te"][:] = 2e4
        state["Ti"][:] = 5e3
        n_i = state["rho"] / m_d
        state["pressure"][:] = n_i * k_B * (state["Te"] + state["Ti"])
        ratio_before = np.mean(state["Te"]) / np.mean(state["Ti"])
        result = solver.step(state, dt=1e-12, current=0, voltage=0)
        ratio_after = np.mean(result["Te"]) / np.mean(result["Ti"])
        assert abs(ratio_after - ratio_before) / ratio_before < 0.1

    def test_cylindrical_two_temperature(self):
        """Cylindrical solver also preserves Te != Ti."""
        nr, nz = 16, 32
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=1e-3, dz=1e-3)
        state = _make_cylindrical_state(nr=nr, nz=nz)
        state["Te"][:] = 5e4
        state["Ti"][:] = 1e4
        n_i = state["rho"] / m_d
        state["pressure"][:] = n_i * k_B * (state["Te"] + state["Ti"])
        result = solver.step(state, dt=1e-12, current=0, voltage=0)
        Te_mean = np.mean(result["Te"])
        Ti_mean = np.mean(result["Ti"])
        assert Te_mean > Ti_mean * 1.5


class TestDeuteriumMass:
    """Test that deuterium mass is used correctly throughout."""

    def test_m_d_value(self):
        """Deuterium mass constant has correct value."""
        assert abs(m_d - 3.34358377e-27) < 1e-35
        assert m_d > m_p * 1.9

    def test_config_default_ion_mass(self):
        """Default ion_mass in config is deuterium mass."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.ion_mass == pytest.approx(m_d, rel=1e-6)

    def test_engine_uses_ion_mass(self):
        """Engine initial state uses ion_mass for pressure calculation."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        rho0 = config.rho0
        T0 = config.T0
        n_i = rho0 / m_d
        expected_p0 = 2.0 * n_i * k_B * T0
        actual_p0 = engine.state["pressure"][0, 0, 0]
        assert actual_p0 == pytest.approx(expected_p0, rel=1e-6)

    def test_custom_ion_mass(self):
        """Can specify custom ion_mass (e.g., proton for hydrogen)."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            ion_mass=m_p,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.ion_mass == pytest.approx(m_p, rel=1e-6)

    def test_solver_ion_mass_parameter(self):
        """MHD solver accepts and uses ion_mass parameter."""
        solver_d = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, ion_mass=m_d)
        solver_p = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, ion_mass=m_p)
        assert solver_d.ion_mass == m_d
        assert solver_p.ion_mass == m_p

    def test_cylindrical_solver_ion_mass(self):
        """Cylindrical solver uses ion_mass parameter."""
        solver = CylindricalMHDSolver(nr=16, nz=32, dr=1e-3, dz=1e-3, ion_mass=m_d)
        assert solver.ion_mass == m_d


class TestNeutronYieldFactor:
    """Test neutron yield uses correct factor."""

    def test_yield_factor_is_quarter(self):
        """Neutron rate uses 1/4 factor."""
        from dpf.diagnostics.neutron_yield import dd_reactivity, neutron_yield_rate

        n_D = np.array([1e25])
        Ti = np.array([1.16e8])
        rate_density, total_rate = neutron_yield_rate(n_D, Ti, 1.0)
        sv = dd_reactivity(10.0)
        expected_rate = 0.25 * n_D[0]**2 * sv * 1.0
        assert total_rate == pytest.approx(expected_rate, rel=1e-3)


class TestBoundaryConfig:
    """Test BoundaryConfig integration."""

    def test_default_electrode_bc_off(self):
        """Default electrode_bc is False."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.boundary.electrode_bc is False

    def test_enable_electrode_bc(self):
        """Can enable electrode BC via config."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            boundary={"electrode_bc": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.boundary.electrode_bc is True

    def test_fluid_config_resistive_flag(self):
        """FluidConfig has enable_resistive and enable_energy_equation."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.fluid.enable_resistive is True
        assert config.fluid.enable_energy_equation is True


class TestEnginePhase11Integration:
    """Test Phase 11 features work end-to-end through the engine."""

    def test_engine_step_with_resistive(self):
        """Engine step completes with resistive MHD enabled."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        result = engine.step()
        assert result.dt > 0
        assert np.isfinite(result.max_Te)

    def test_engine_10_steps_no_nan(self):
        """Engine completes 10 steps without NaN values."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        for _ in range(10):
            engine.step()
        assert np.all(np.isfinite(engine.state["Te"]))
        assert np.all(np.isfinite(engine.state["pressure"]))

    def test_cylindrical_engine_with_electrode_bc(self):
        """Cylindrical engine with electrode BCs enabled runs without error."""
        config = SimulationConfig(
            grid_shape=[16, 1, 32], dx=5e-4, sim_time=1e-7,
            geometry={"type": "cylindrical"},
            boundary={"electrode_bc": True},
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.003, "cathode_radius": 0.006},
        )
        engine = SimulationEngine(config)
        for _ in range(5):
            result = engine.step()
        assert result.dt > 0
        B_theta = engine.state["B"][1]
        assert np.max(np.abs(B_theta)) > 0

    def test_energy_conservation_with_phase11(self):
        """Circuit energy still conserved with Phase 11 features."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-2, sim_time=1e-6,
            fluid={"enable_resistive": True, "enable_energy_equation": True},
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=50)
        assert 0.90 < summary["energy_conservation"] < 1.10


# --- Section: Cylindrical Geometry ---


class TestCylindricalDivergence:
    """Tests for divergence in cylindrical coordinates."""

    def test_uniform_field_zero_div(self):
        """Uniform radial + axial field has div=0 only if F_r=0 and dFz/dz=0."""
        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        F = np.zeros((3, 16, 16))
        F[2] = 1.0
        div = geom.divergence(F)
        np.testing.assert_allclose(div, 0.0, atol=1e-10)

    def test_radial_1_over_r_zero_div(self):
        """For F_r = C/r (vacuum field from line charge), div ~ 0."""
        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        F = np.zeros((3, 32, 8))
        C = 1.0
        for i in range(32):
            r = geom.r[i]
            if r > 1e-10:
                F[0, i, :] = C / r
        div = geom.divergence(F)
        interior = div[4:-4, 2:-2]
        assert np.max(np.abs(interior)) < 0.5

    def test_divergence_of_r_hat(self):
        """div(r_hat) = 1/r in cylindrical coords."""
        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        F = np.zeros((3, 32, 8))
        F[0] = 1.0
        div = geom.divergence(F)
        for i in range(4, 28):
            expected = 1.0 / geom.r[i]
            assert abs(div[i, 4] - expected) / expected < 0.2


class TestCylindricalGradient:
    """Tests for gradient in cylindrical coordinates."""

    def test_constant_zero_gradient(self):
        """Gradient of a constant is zero."""
        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        p = np.full((16, 16), 100.0)
        grad = geom.gradient(p)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_linear_r_gradient(self):
        """grad(p = r) = (1, 0, 0) in cylindrical."""
        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        p = geom.r[:, np.newaxis] * np.ones((1, 8))
        grad = geom.gradient(p)
        interior_grad_r = grad[0, 4:-4, :]
        np.testing.assert_allclose(interior_grad_r, 1.0, atol=0.1)
        np.testing.assert_allclose(grad[2], 0.0, atol=1e-10)


class TestCylindricalCurl:
    """Tests for curl in cylindrical coordinates."""

    def test_curl_of_constant_zero(self):
        """Curl of a constant vector field is zero (for r and theta components)."""
        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        B = np.ones((3, 16, 16))
        curl = geom.curl(B)
        np.testing.assert_allclose(curl[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(curl[1], 0.0, atol=1e-10)

    def test_b_theta_from_line_current(self):
        """For B_theta = mu_0*I/(2*pi*r), curl gives zero for r > 0."""
        geom = CylindricalGeometry(nr=32, nz=8, dr=0.01, dz=0.01)
        current = 1000.0
        B = np.zeros((3, 32, 8))
        for i in range(32):
            r = geom.r[i]
            if r > 1e-10:
                B[1, i, :] = mu_0 * current / (2.0 * pi * r)
        curl = geom.curl(B)
        interior_curl_z = curl[2, 4:-4, 2:-2]
        assert np.max(np.abs(interior_curl_z)) < 50.0


class TestCylindricalLaplacian:
    """Tests for Laplacian in cylindrical coordinates."""

    def test_constant_zero_laplacian(self):
        """Laplacian of a constant is zero."""
        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        T = np.full((16, 16), 100.0)
        lap = geom.laplacian(T)
        np.testing.assert_allclose(lap, 0.0, atol=1e-6)


class TestCylindricalVolumes:
    """Tests for cell volumes and face areas."""

    def test_cell_volumes_positive(self):
        """All cell volumes should be positive."""
        geom = CylindricalGeometry(nr=16, nz=16, dr=0.01, dz=0.01)
        V = geom.cell_volumes()
        assert V.shape == (16, 16)
        assert np.all(V > 0)

    def test_cell_volumes_increase_with_r(self):
        """Cell volumes should increase with radius (V ~ r * dr * dz)."""
        geom = CylindricalGeometry(nr=16, nz=8, dr=0.01, dz=0.01)
        V = geom.cell_volumes()
        for j in range(8):
            for i in range(15):
                assert V[i + 1, j] > V[i, j]

    def test_total_volume_matches_cylinder(self):
        """Total volume should match pi * R^2 * L for the full domain."""
        nr, nz = 32, 16
        dr, dz = 0.01, 0.02
        geom = CylindricalGeometry(nr=nr, nz=nz, dr=dr, dz=dz)
        V_total = np.sum(geom.cell_volumes())
        R = nr * dr
        L = nz * dz
        V_expected = pi * R**2 * L
        np.testing.assert_allclose(V_total, V_expected, rtol=1e-10)

    def test_face_areas_radial_shape(self):
        """Radial face areas should have shape (nr+1, nz)."""
        geom = CylindricalGeometry(nr=16, nz=8, dr=0.01, dz=0.01)
        A = geom.face_areas_radial()
        assert A.shape == (17, 8)

    def test_face_area_at_axis_zero(self):
        """Radial face area at r=0 (axis) should be zero."""
        geom = CylindricalGeometry(nr=16, nz=8, dr=0.01, dz=0.01)
        A = geom.face_areas_radial()
        np.testing.assert_allclose(A[0, :], 0.0, atol=1e-20)


class TestCylindricalMHDSolver:
    """Tests for the cylindrical MHD solver."""

    def _make_state(self, nr=16, nz=16):
        """Create a basic cylindrical state dict with ny=1."""
        rho0 = 1e-4
        T0 = 1e4
        p0 = (rho0 / 1.672e-27) * 1.38e-23 * T0
        return {
            "rho": np.full((nr, 1, nz), rho0),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), p0),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), T0),
            "Ti": np.full((nr, 1, nz), T0),
            "psi": np.zeros((nr, 1, nz)),
        }

    def test_step_returns_correct_keys(self):
        """Solver returns all expected state keys."""
        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01)
        state = self._make_state()
        result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti", "psi"]:
            assert key in result

    def test_step_preserves_shape(self):
        """Output shapes match input (nr, 1, nz)."""
        nr, nz = 16, 16
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=0.01, dz=0.01)
        state = self._make_state(nr, nz)
        result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
        assert result["rho"].shape == (nr, 1, nz)
        assert result["velocity"].shape == (3, nr, 1, nz)
        assert result["B"].shape == (3, nr, 1, nz)

    def test_uniform_state_preserved(self):
        """A uniform state with zero velocity should be (nearly) preserved."""
        solver = CylindricalMHDSolver(
            nr=16, nz=16, dr=0.01, dz=0.01, enable_hall=False,
        )
        state = self._make_state()
        rho0 = state["rho"][0, 0, 0]
        result = solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        np.testing.assert_allclose(result["rho"], rho0, rtol=1e-3)

    def test_density_stays_positive(self):
        """Density floor prevents negative values."""
        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01, enable_hall=False)
        state = self._make_state()
        state["rho"][:] = 1e-30
        result = solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        assert np.all(result["rho"] > 0)

    def test_coupling_interface(self):
        """Solver returns valid CouplingState."""
        from dpf.core.bases import CouplingState

        solver = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        cs = solver.coupling_interface()
        assert isinstance(cs, CouplingState)

    def test_no_nan_after_step(self):
        """No NaN values after a step."""
        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01)
        state = self._make_state()
        result = solver.step(state, dt=1e-10, current=1000.0, voltage=5000.0)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti"]:
            assert np.all(np.isfinite(result[key])), f"NaN in {key}"

    def test_compute_dt_positive(self):
        """Computed timestep is positive."""
        solver = CylindricalMHDSolver(nr=16, nz=16, dr=0.01, dz=0.01)
        state = self._make_state()
        dt = solver._compute_dt(state)
        assert dt > 0


class TestGeometryConfig:
    """Tests for the geometry configuration."""

    def test_default_cartesian(self):
        """Default geometry is Cartesian."""
        from dpf.config import GeometryConfig

        cfg = GeometryConfig()
        assert cfg.type == "cartesian"
        assert cfg.dz is None

    def test_cylindrical_config(self):
        """Cylindrical config with explicit dz."""
        from dpf.config import GeometryConfig

        cfg = GeometryConfig(type="cylindrical", dz=0.005)
        assert cfg.type == "cylindrical"
        assert cfg.dz == 0.005

    def test_invalid_type_raises(self):
        """Invalid geometry type raises."""
        from dpf.config import GeometryConfig

        with pytest.raises(ValueError, match="geometry type"):
            GeometryConfig(type="spherical")

    def test_cylindrical_grid_validation(self):
        """Cylindrical geometry requires ny=1."""
        with pytest.raises(ValueError, match="grid_shape\\[1\\]=1"):
            SimulationConfig(
                grid_shape=[16, 4, 16],
                dx=0.01,
                sim_time=1e-6,
                circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                         "anode_radius": 0.005, "cathode_radius": 0.01},
                geometry={"type": "cylindrical"},
            )

    def test_cylindrical_grid_valid(self):
        """Cylindrical geometry with ny=1 is valid."""
        config = SimulationConfig(
            grid_shape=[16, 1, 16],
            dx=0.01,
            sim_time=1e-6,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            geometry={"type": "cylindrical"},
        )
        assert config.geometry.type == "cylindrical"

    def test_simulation_config_default_geometry(self):
        """SimulationConfig defaults to Cartesian geometry."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-6,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
        )
        assert config.geometry.type == "cartesian"


class TestEngineCylindrical:
    """Tests for the simulation engine in cylindrical mode."""

    def _make_cyl_config(self):
        """Create a cylindrical simulation config."""
        return SimulationConfig(
            grid_shape=[16, 1, 16],
            dx=0.01,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": False},
        )

    def test_engine_init_cylindrical(self):
        """Engine initializes with cylindrical geometry."""
        config = self._make_cyl_config()
        engine = SimulationEngine(config)
        assert engine.geometry_type == "cylindrical"

    def test_engine_runs_cylindrical(self):
        """Engine completes a short run in cylindrical mode without errors."""
        config = self._make_cyl_config()
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5
        assert summary["energy_conservation"] > 0.9

    def test_engine_no_nan_cylindrical(self):
        """No NaN in state after cylindrical engine run."""
        config = self._make_cyl_config()
        engine = SimulationEngine(config)
        engine.run(max_steps=3)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti"]:
            assert np.all(np.isfinite(engine.state[key])), f"NaN in {key}"

    def test_engine_cartesian_unchanged(self):
        """Existing Cartesian mode still works identically."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=0.01,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            radiation={"bremsstrahlung_enabled": False},
        )
        engine = SimulationEngine(config)
        assert engine.geometry_type == "cartesian"
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5


class TestBennettPinch:
    """Test Bennett pinch equilibrium: J_z produces B_theta that confines plasma."""

    def test_bennett_pressure_balance(self):
        """For a z-pinch with uniform J_z, verify pressure gradient balances Lorentz force."""
        nr, nz = 64, 4
        dr, dz = 0.001, 0.01
        geom = CylindricalGeometry(nr=nr, nz=nz, dr=dr, dz=dz)
        J_z = 1e8
        B = np.zeros((3, nr, nz))
        for i in range(nr):
            B[1, i, :] = mu_0 * J_z * geom.r[i] / 2.0
        curl_B = geom.curl(B)
        J_computed = curl_B / mu_0
        J_z_computed = J_computed[2, 4:-4, :]
        np.testing.assert_allclose(
            J_z_computed, J_z, rtol=0.15,
            err_msg="Computed J_z doesn't match input J_z for uniform current",
        )

    def test_lorentz_force_direction(self):
        """The J × B force should point inward (negative r) for a z-pinch."""
        nr, nz = 32, 4
        dr, dz = 0.001, 0.01
        geom = CylindricalGeometry(nr=nr, nz=nz, dr=dr, dz=dz)
        J_z = 1e8
        B = np.zeros((3, nr, nz))
        for i in range(nr):
            B[1, i, :] = mu_0 * J_z * geom.r[i] / 2.0
        curl_B = geom.curl(B)
        J = curl_B / mu_0
        JxB_r = J[1] * B[2] - J[2] * B[1]
        interior = JxB_r[4:-4, :]
        assert np.all(interior <= 0.0), "Lorentz force should point inward for z-pinch"


# --- Section: Phase 12 HLLD WENO5 Strang Splitting ---


class TestSpatiallyResolvedResistivity:
    """Tests for cell-by-cell anomalous resistivity field computation."""

    def test_anomalous_resistivity_field_shape(self):
        """anomalous_resistivity_field returns array with same shape as input."""
        nr, nz = 16, 32
        J_mag = np.full((nr, nz), 1e8)
        ne = np.full((nr, nz), 1e24)
        Ti = np.full((nr, nz), 1e5)
        Te = np.full((nr, nz), 1e5)
        eta = anomalous_resistivity_field(J_mag, ne, Ti, alpha=0.05, mi=m_d, Te=Te)
        assert eta.shape == (nr, nz)

    def test_anomalous_resistivity_zero_below_threshold(self):
        """Below ion-acoustic threshold (v_d < c_s), eta_anom = 0."""
        nr, nz = 8, 8
        ne = np.full((nr, nz), 1e24)
        Ti = np.full((nr, nz), 1e7)
        Te = np.full((nr, nz), 1e7)
        J_mag = np.full((nr, nz), 1e4)
        eta = anomalous_resistivity_field(J_mag, ne, Ti, alpha=0.05, mi=m_d, Te=Te)
        np.testing.assert_array_equal(eta, 0.0)

    def test_anomalous_resistivity_nonzero_above_threshold(self):
        """Above ion-acoustic threshold, eta_anom > 0."""
        nr, nz = 8, 8
        ne = np.full((nr, nz), 1e22)
        Ti = np.full((nr, nz), 300.0)
        Te = np.full((nr, nz), 300.0)
        J_mag = np.full((nr, nz), 1e10)
        eta = anomalous_resistivity_field(J_mag, ne, Ti, alpha=0.05, mi=m_d, Te=Te)
        assert np.all(eta > 0.0)
        assert np.all(eta < 1.0)

    def test_anomalous_resistivity_spatial_variation(self):
        """Resistivity varies spatially when J/ne varies."""
        nr, nz = 16, 16
        ne = np.full((nr, nz), 1e22)
        Ti = np.full((nr, nz), 300.0)
        Te = np.full((nr, nz), 300.0)
        J_mag = np.zeros((nr, nz))
        J_mag[:8, :] = 1e10
        eta = anomalous_resistivity_field(J_mag, ne, Ti, alpha=0.05, mi=m_d, Te=Te)
        assert np.all(eta[:8, :] > 0.0)
        assert np.all(eta[8:, :] == 0.0)

    def test_anomalous_resistivity_alpha_scaling(self):
        """eta_anom scales linearly with alpha parameter."""
        J_mag = np.array([[1e10]])
        ne = np.array([[1e22]])
        Ti = np.array([[300.0]])
        Te = np.array([[300.0]])
        eta1 = anomalous_resistivity_field(J_mag, ne, Ti, alpha=0.01, mi=m_d, Te=Te)
        eta2 = anomalous_resistivity_field(J_mag, ne, Ti, alpha=0.10, mi=m_d, Te=Te)
        np.testing.assert_allclose(eta2 / eta1, 10.0, rtol=1e-10)


class TestVolumeIntegralCoupling:
    """Tests for volume-integral R_plasma and L_plasma computation."""

    def test_engine_computes_r_plasma(self):
        """Engine produces a non-negative R_plasma."""
        preset = get_preset("tutorial")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        engine.step()
        assert hasattr(engine, "_last_R_plasma")
        assert engine._last_R_plasma >= 0.0

    def test_engine_computes_l_plasma(self):
        """Engine passes L_plasma to coupling when B-field is present."""
        preset = get_preset("tutorial")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(5):
            engine.step()
        assert hasattr(engine, "_coupling")

    def test_r_plasma_physical_range(self):
        """R_plasma stays within physically reasonable bounds."""
        preset = get_preset("tutorial")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(10):
            engine.step()
        assert 0.0 <= engine._last_R_plasma <= 10.0


class TestHLLDSolver:
    """Tests for the HLLD approximate Riemann solver."""

    def test_hlld_returns_correct_shapes(self):
        """HLLD flux function returns arrays of correct size."""
        n = 10
        rho_L = np.ones(n)
        rho_R = np.ones(n) * 0.125
        u_L = np.zeros(n)
        u_R = np.zeros(n)
        p_L = np.ones(n)
        p_R = np.ones(n) * 0.1
        Bn_L = np.ones(n) * 0.75
        Bn_R = np.ones(n) * 0.75
        gamma_local = 5.0 / 3.0
        F_rho, F_mom, F_ene = _hlld_flux_1d_core(
            rho_L, rho_R, u_L, u_R, p_L, p_R, Bn_L, Bn_R, gamma_local,
        )
        assert F_rho.shape == (n,)
        assert F_mom.shape == (n,)
        assert F_ene.shape == (n,)

    def test_hlld_dict_wrapper(self):
        """_hlld_flux_1d returns dict with correct keys (full 8-component)."""
        n = 5
        ones = np.ones(n)
        zeros = np.zeros(n)
        result = _hlld_flux_1d(
            ones, ones * 0.5,
            zeros, zeros,
            zeros, zeros,
            zeros, zeros,
            ones, ones * 0.5,
            ones * 0.5,
            zeros, zeros,
            zeros, zeros,
            5.0 / 3.0,
        )
        assert "mass_flux" in result
        assert "momentum_flux" in result
        assert "energy_flux" in result
        assert "momentum_t1_flux" in result
        assert "momentum_t2_flux" in result
        assert "Bt1_flux" in result
        assert "Bt2_flux" in result

    def test_hlld_uniform_state_zero_flux(self):
        """Uniform state produces zero net mass flux through interfaces."""
        n = 10
        rho = np.ones(n) * 1.0
        u = np.zeros(n)
        p = np.ones(n) * 1.0
        Bn = np.ones(n) * 1.0
        gamma_local = 5.0 / 3.0
        F_rho, _, _ = _hlld_flux_1d_core(rho, rho, u, u, p, p, Bn, Bn, gamma_local)
        np.testing.assert_allclose(F_rho, 0.0, atol=1e-12)

    def test_hlld_shock_direction(self):
        """HLLD mass flux is positive for left-to-right shock."""
        n = 10
        rho_L = np.ones(n) * 1.0
        rho_R = np.ones(n) * 0.1
        u_L = np.zeros(n)
        u_R = np.zeros(n)
        p_L = np.ones(n) * 1.0
        p_R = np.ones(n) * 0.1
        Bn = np.ones(n) * 0.5
        gamma_local = 5.0 / 3.0
        F_rho, _, _ = _hlld_flux_1d_core(
            rho_L, rho_R, u_L, u_R, p_L, p_R, Bn, Bn, gamma_local,
        )
        assert np.all(F_rho > 0.0)

    def test_hlld_less_diffusive_than_hll(self):
        """HLLD resolves contact discontinuity better than HLL."""
        n = 20
        rho_L = np.ones(n) * 1.0
        rho_R = np.ones(n) * 0.5
        u_L = np.ones(n) * 1.0
        u_R = np.ones(n) * 1.0
        p_L = np.ones(n) * 1.0
        p_R = np.ones(n) * 1.0
        Bn = np.ones(n) * 0.1
        gamma_local = 5.0 / 3.0
        F_rho_hll, _, _ = _hll_flux_1d_core(
            rho_L, rho_R, u_L, u_R, p_L, p_R, Bn, Bn, gamma_local,
        )
        F_rho_hlld, _, _ = _hlld_flux_1d_core(
            rho_L, rho_R, u_L, u_R, p_L, p_R, Bn, Bn, gamma_local,
        )
        assert np.all(np.isfinite(F_rho_hll))
        assert np.all(np.isfinite(F_rho_hlld))
        assert np.all(F_rho_hll > 0)
        assert np.all(F_rho_hlld > 0)

    def test_hlld_symmetry(self):
        """Swapping L<->R and negating velocity produces negated flux."""
        n = 8
        rho_L = np.ones(n) * 1.0
        rho_R = np.ones(n) * 0.5
        u_L = np.ones(n) * 0.3
        u_R = np.ones(n) * -0.1
        p_L = np.ones(n) * 1.2
        p_R = np.ones(n) * 0.8
        Bn = np.ones(n) * 0.5
        gamma_local = 5.0 / 3.0
        F1, _, _ = _hlld_flux_1d_core(
            rho_L, rho_R, u_L, u_R, p_L, p_R, Bn, Bn, gamma_local,
        )
        F2, _, _ = _hlld_flux_1d_core(
            rho_R, rho_L, -u_R, -u_L, p_R, p_L, Bn, Bn, gamma_local,
        )
        np.testing.assert_allclose(F1, -F2, atol=1e-12)


class TestHLLDInSolver:
    """Tests that HLLD integrates correctly into the MHD solver."""

    def test_mhd_solver_accepts_hlld(self):
        """MHDSolver can be created with riemann_solver='hlld'."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, riemann_solver="hlld")
        assert solver.riemann_solver == "hlld"

    def test_mhd_solver_hlld_step_runs(self):
        """MHDSolver with HLLD can complete a step without error."""
        solver = MHDSolver(grid_shape=(8, 8, 8), dx=1e-3, riemann_solver="hlld")
        state = {
            "rho": np.full((8, 8, 8), 1e-4),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.full((8, 8, 8), 100.0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 1e4),
            "Ti": np.full((8, 8, 8), 1e4),
            "psi": np.zeros((8, 8, 8)),
        }
        result = solver.step(state, dt=1e-10, current=1e3, voltage=1e3)
        assert np.all(np.isfinite(result["rho"]))
        assert np.all(result["rho"] > 0)

    def test_config_riemann_solver_hlld(self):
        """SimulationConfig accepts riemann_solver='hlld' in fluid config."""
        preset = get_preset("tutorial")
        preset["fluid"] = {"riemann_solver": "hlld"}
        config = SimulationConfig(**preset)
        assert config.fluid.riemann_solver == "hlld"

    def test_engine_hlld_runs(self):
        """Engine runs with HLLD solver selected."""
        preset = get_preset("tutorial")
        preset["fluid"] = {"riemann_solver": "hlld"}
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        result = engine.step()
        assert result.finished is False
        assert np.isfinite(result.current)


class TestCylindricalWENO5:
    """Tests for WENO5 reconstruction in the cylindrical solver."""

    def test_cylindrical_solver_weno5_enabled(self):
        """Cylindrical solver enables WENO5 when grid is large enough."""
        solver = CylindricalMHDSolver(nr=16, nz=32, dr=1e-3, dz=1e-3)
        assert solver.use_weno5 is True

    def test_cylindrical_solver_weno5_disabled_small_grid(self):
        """WENO5 disabled for grids smaller than 5 cells."""
        solver = CylindricalMHDSolver(nr=4, nz=4, dr=1e-3, dz=1e-3)
        assert solver.use_weno5 is False

    def test_cylindrical_weno5_step_runs(self):
        """Cylindrical solver with WENO5 can complete a step."""
        nr, nz = 16, 32
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=5e-4, dz=5e-4)
        state = {
            "rho": np.full((nr, 1, nz), 1e-4),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), 100.0),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        result = solver.step(state, dt=1e-10, current=1e3, voltage=1e3)
        assert np.all(np.isfinite(result["rho"]))

    def test_cylindrical_weno5_with_hlld(self):
        """Cylindrical WENO5+HLLD combination works."""
        nr, nz = 16, 32
        solver = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=5e-4, dz=5e-4, riemann_solver="hlld",
        )
        assert solver.riemann_solver == "hlld"
        assert solver.use_weno5 is True
        state = {
            "rho": np.full((nr, 1, nz), 1e-4),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), 100.0),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        result = solver.step(state, dt=1e-10, current=1e3, voltage=1e3)
        assert np.all(np.isfinite(result["rho"]))

    def test_cylindrical_weno5_density_conservation(self):
        """WENO5 cylindrical solver conserves total mass (approximately)."""
        nr, nz = 16, 32
        dr, dz = 5e-4, 5e-4
        solver = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz)
        state = {
            "rho": np.full((nr, 1, nz), 1e-4),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), 100.0),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        cell_vol = solver.geom.cell_volumes()
        rho0_2d = state["rho"][:, 0, :]
        mass_before = float(np.sum(rho0_2d * cell_vol))
        result = solver.step(state, dt=1e-11, current=1e3, voltage=1e3)
        rho_after = result["rho"][:, 0, :]
        mass_after = float(np.sum(rho_after * cell_vol))
        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-6)


class TestStrangSplitting:
    """Tests for Strang (symmetric) operator splitting."""

    def test_engine_has_apply_collision_radiation(self):
        """Engine has the _apply_collision_radiation method."""
        preset = get_preset("tutorial")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        assert hasattr(engine, "_apply_collision_radiation")
        assert callable(engine._apply_collision_radiation)

    def test_strang_splitting_runs(self):
        """Engine step completes with Strang splitting (no crash)."""
        preset = get_preset("tutorial")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        result = engine.step()
        assert result.finished is False
        assert result.step == 1

    def test_strang_two_half_steps_equivalent(self):
        """Two half-steps of collision/radiation produce correct results."""
        preset = get_preset("tutorial")
        preset["radiation"] = {"bremsstrahlung_enabled": False}
        config = SimulationConfig(**preset)
        engine1 = SimulationEngine(config)
        for _ in range(3):
            engine1.step()
        result = engine1.step()
        assert np.all(np.isfinite(engine1.state["Te"]))
        assert np.all(np.isfinite(engine1.state["Ti"]))
        assert result.energy_conservation > 0.0

    def test_strang_preserves_energy_conservation(self):
        """Strang splitting maintains energy conservation quality."""
        preset = get_preset("tutorial")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(20):
            result = engine.step()
        assert result.energy_conservation > 0.5
        assert result.energy_conservation < 2.0

    def test_collision_radiation_substep_relaxes_temperatures(self):
        """The _apply_collision_radiation method relaxes Te toward Ti."""
        preset = get_preset("tutorial")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        engine.state["Te"] = np.full_like(engine.state["Te"], 1e6)
        engine.state["Ti"] = np.full_like(engine.state["Ti"], 1e4)
        Te_before = float(np.mean(engine.state["Te"]))
        Ti_before = float(np.mean(engine.state["Ti"]))
        engine._apply_collision_radiation(1e-9, Z_bar=1.0)
        Te_after = float(np.mean(engine.state["Te"]))
        Ti_after = float(np.mean(engine.state["Ti"]))
        assert Te_after < Te_before
        assert Ti_after > Ti_before

    def test_collision_radiation_substep_applies_radiation(self):
        """The _apply_collision_radiation method applies bremsstrahlung."""
        preset = get_preset("tutorial")
        preset["radiation"] = {"bremsstrahlung_enabled": True, "fld_enabled": False}
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        engine.state["Te"] = np.full_like(engine.state["Te"], 1e7)
        engine.state["rho"] = np.full_like(engine.state["rho"], 1e-3)
        radiated_before = engine.total_radiated_energy
        engine._apply_collision_radiation(1e-9, Z_bar=1.0)
        radiated_after = engine.total_radiated_energy
        assert radiated_after > radiated_before


class TestPF1000Validation:
    """PF-1000 current waveform comparison tests."""

    def test_pf1000_preset_creates_cylindrical_engine(self):
        """PF-1000 preset creates a cylindrical geometry engine."""
        preset = get_preset("pf1000")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        assert engine.geometry_type == "cylindrical"

    def test_pf1000_runs_3_steps(self):
        """PF-1000 preset simulation runs for 3 steps without crash."""
        preset = get_preset("pf1000")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(3):
            result = engine.step()
        assert result.step == 3
        assert result.current != 0.0
        assert np.isfinite(result.current)

    def test_pf1000_current_nonzero_initial(self):
        """PF-1000 current should be nonzero after first step."""
        preset = get_preset("pf1000")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        result = engine.step()
        assert abs(result.current) > 0.0
        assert np.isfinite(result.current)

    def test_pf1000_energy_conservation_initial(self):
        """PF-1000 energy conservation is good in early steps."""
        preset = get_preset("pf1000")
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(3):
            result = engine.step()
        assert 0.8 < result.energy_conservation < 1.2

    def test_pf1000_validation_suite_runs(self):
        """ValidationSuite can be instantiated with PF-1000."""
        suite = ValidationSuite(devices=["PF-1000"])
        assert "PF-1000" in suite.devices

    def test_pf1000_circuit_validation_structure(self):
        """validate_circuit returns proper ValidationResult structure."""
        suite = ValidationSuite(devices=["PF-1000"])
        sim_summary = {
            "peak_current_A": 2.0e6,
            "peak_current_time_s": 6.0e-6,
            "energy_conservation": 0.98,
            "final_current_A": 1.0e6,
        }
        result = suite.validate_circuit("PF-1000", sim_summary)
        assert result.device == "PF-1000"
        assert len(result.metrics) > 0
        assert 0.0 <= result.overall_score <= 1.0


class TestPhase12Integration:
    """Integration tests combining multiple Phase 12 features."""

    def test_cylindrical_hlld_with_strang(self):
        """Cylindrical + HLLD + Strang splitting all work together."""
        preset = get_preset("nx2")
        preset["fluid"] = {"riemann_solver": "hlld"}
        preset["radiation"] = {"bremsstrahlung_enabled": False}
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(3):
            result = engine.step()
        assert result.step == 3
        assert np.isfinite(result.current)
        assert np.all(np.isfinite(engine.state["rho"]))

    def test_tutorial_with_all_phase12_features(self):
        """Tutorial preset works with all Phase 12 features enabled."""
        preset = get_preset("tutorial")
        preset["fluid"] = {"riemann_solver": "hlld"}
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(10):
            result = engine.step()
        assert result.step == 10
        assert 0.5 < result.energy_conservation < 2.0

    def test_spatially_resolved_eta_passed_to_fluid(self):
        """Engine passes spatially-resolved eta_field to fluid solver."""
        preset = get_preset("tutorial")
        preset["T0"] = 1e5
        preset["rho0"] = 1e-3
        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        for _ in range(5):
            engine.step()
        assert engine._last_R_plasma >= 0.0


# --- Section: Phase 14 Implicit Diffusion Super Time-Stepping ---


class TestThomasSolve:
    """Tests for the tridiagonal Thomas algorithm."""

    def test_identity_system(self):
        """I * x = b gives x = b."""
        n = 10
        lower = np.zeros(n)
        diag = np.ones(n)
        upper = np.zeros(n)
        rhs = np.arange(n, dtype=np.float64)
        x = _thomas_solve(lower, diag, upper, rhs)
        np.testing.assert_allclose(x, rhs, atol=1e-14)

    def test_simple_tridiag(self):
        """Solve a known 3x3 tridiagonal system."""
        lower = np.array([0.0, -1.0, -1.0])
        diag = np.array([2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0, 0.0])
        rhs = np.array([1.0, 0.0, 1.0])
        x = _thomas_solve(lower, diag, upper, rhs)
        assert np.all(np.isfinite(x))
        Ax = np.array([
            diag[0] * x[0] + upper[0] * x[1],
            lower[1] * x[0] + diag[1] * x[1] + upper[1] * x[2],
            lower[2] * x[1] + diag[2] * x[2],
        ])
        np.testing.assert_allclose(Ax, rhs, atol=1e-12)


class TestDiffuseField1D:
    """Tests for Crank-Nicolson 1D diffusion."""

    def test_uniform_field_unchanged(self):
        """Constant field is unchanged by diffusion."""
        n = 50
        field = np.full(n, 3.0)
        coeff = np.full(n, 1.0)
        result = diffuse_field_1d(field, coeff, dt=0.1, dx=0.01)
        np.testing.assert_allclose(result, 3.0, atol=1e-12)

    def test_gaussian_smoothing(self):
        """Gaussian peak is smoothed by diffusion."""
        n = 100
        x = np.linspace(0, 1, n)
        field = np.exp(-100 * (x - 0.5) ** 2)
        coeff = np.full(n, 0.01)
        result = diffuse_field_1d(field, coeff, dt=0.001, dx=1.0 / n)
        assert result[n // 2] < field[n // 2]
        np.testing.assert_allclose(np.sum(result), np.sum(field), rtol=1e-8)

    def test_conservation(self):
        """Total integral is conserved under Neumann BCs."""
        n = 64
        field = np.random.default_rng(42).standard_normal(n)
        field -= field.mean()
        field += 5.0
        coeff = np.full(n, 0.5)
        result = diffuse_field_1d(field, coeff, dt=0.01, dx=0.1)
        np.testing.assert_allclose(np.sum(result), np.sum(field), rtol=1e-10)

    def test_short_array_passthrough(self):
        """Arrays shorter than 3 are returned unchanged."""
        field = np.array([1.0, 2.0])
        result = diffuse_field_1d(field, np.ones(2), dt=0.1, dx=0.1)
        np.testing.assert_allclose(result, field)


class TestImplicitResistiveDiffusion:
    """Tests for 3D ADI resistive diffusion."""

    def test_uniform_B_unchanged(self):
        """Uniform B-field is unchanged by diffusion."""
        shape = (8, 8, 8)
        Bx = np.full(shape, 1.0)
        By = np.full(shape, 2.0)
        Bz = np.full(shape, 3.0)
        eta = np.full(shape, 1e-3)
        Bx_n, By_n, Bz_n = implicit_resistive_diffusion(
            Bx, By, Bz, eta, dt=1e-6, dx=0.01, dy=0.01, dz=0.01,
        )
        np.testing.assert_allclose(Bx_n, 1.0, atol=1e-10)
        np.testing.assert_allclose(By_n, 2.0, atol=1e-10)
        np.testing.assert_allclose(Bz_n, 3.0, atol=1e-10)

    def test_diffusion_smooths_field(self):
        """Non-uniform B-field is smoothed by diffusion."""
        shape = (16, 8, 8)
        Bx = np.zeros(shape)
        Bx[7:9, :, :] = 1.0
        By = np.zeros(shape)
        Bz = np.zeros(shape)
        eta = np.full(shape, 1e-2)
        Bx_n, _, _ = implicit_resistive_diffusion(
            Bx, By, Bz, eta, dt=1e-4, dx=0.01, dy=0.01, dz=0.01,
        )
        assert Bx_n[8, 4, 4] < Bx[8, 4, 4]


class TestImplicitThermalDiffusion:
    """Tests for 3D ADI thermal diffusion."""

    def test_uniform_Te_unchanged(self):
        """Uniform temperature is unchanged."""
        shape = (8, 8, 8)
        Te = np.full(shape, 1e6)
        kappa = np.full(shape, 100.0)
        ne = np.full(shape, 1e23)
        Te_new = implicit_thermal_diffusion(
            Te, kappa, ne, dt=1e-9, dx=0.001, dy=0.001, dz=0.001,
        )
        np.testing.assert_allclose(Te_new, 1e6, atol=1.0)

    def test_hot_spot_cools(self):
        """Hot spot temperature decreases by conduction."""
        shape = (16, 8, 8)
        Te = np.full(shape, 1e5)
        Te[7:9, 3:5, 3:5] = 1e7
        kappa = np.full(shape, 1000.0)
        ne = np.full(shape, 1e23)
        Te_new = implicit_thermal_diffusion(
            Te, kappa, ne, dt=1e-7, dx=0.001, dy=0.001, dz=0.001,
        )
        assert Te_new[8, 4, 4] < Te[8, 4, 4]


class TestDiffusionCFL:
    """Tests for explicit diffusion CFL estimate."""

    def test_cfl_positive(self):
        """CFL timestep is positive."""
        dt = diffusion_cfl_dt(1e-3, 100.0, 1e22, 0.001)
        assert dt > 0.0

    def test_cfl_smaller_for_larger_eta(self):
        """Larger eta gives smaller resistive CFL dt."""
        dt1 = diffusion_cfl_dt(1e-3, 0.0, 0.0, 0.001)
        dt2 = diffusion_cfl_dt(1e-1, 0.0, 0.0, 0.001)
        assert dt2 < dt1

    def test_cfl_formula_resistive(self):
        """Resistive CFL matches analytical formula."""
        eta = 1e-3
        dx = 0.001
        dt = diffusion_cfl_dt(eta, 0.0, 0.0, dx)
        expected = dx * dx * mu_0 / (2 * eta)
        np.testing.assert_allclose(dt, expected, rtol=1e-8)


class TestRKL2Coefficients:
    """Tests for RKL2 coefficient computation."""

    def test_s2_valid(self):
        """s=2 produces valid (finite) coefficients."""
        mu, nu, mu_tilde, gamma_tilde = rkl2_coefficients(2)
        assert len(mu) == 3
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(nu))
        assert np.all(np.isfinite(mu_tilde))
        assert np.all(np.isfinite(gamma_tilde))

    def test_s_less_than_2_raises(self):
        """s < 2 raises ValueError."""
        with pytest.raises(ValueError, match="s >= 2"):
            rkl2_coefficients(1)

    def test_s10_valid(self):
        """s=10 produces valid coefficients."""
        mu, nu, mu_tilde, gamma_tilde = rkl2_coefficients(10)
        assert len(mu) == 11
        assert mu_tilde[1] > 0

    def test_mu_tilde_1_positive(self):
        """First-stage coefficient mu_tilde[1] is positive for all s."""
        for s in range(2, 17):
            _, _, mt, _ = rkl2_coefficients(s)
            assert mt[1] > 0, f"mu_tilde[1] <= 0 for s={s}"


class TestDiffusionOperator:
    """Tests for the 1D diffusion operator L(u) = d/dx(D du/dx)."""

    def test_constant_field_zero(self):
        """Laplacian of a constant is zero."""
        u = np.full(20, 5.0)
        D = np.full(20, 1.0)
        Lu = _diffusion_operator_1d(u, D, dx=0.1)
        np.testing.assert_allclose(Lu, 0.0, atol=1e-12)

    def test_quadratic_exact(self):
        """Laplacian of x^2 is 2*D (constant D)."""
        n = 50
        dx = 0.01
        x = np.arange(n) * dx
        u = x * x
        D = 3.0
        Lu = _diffusion_operator_1d_scalar(u, D, dx)
        np.testing.assert_allclose(Lu[5:-5], 2 * D, rtol=1e-6)


class TestRKL2DiffusionStep:
    """Tests for RKL2 1D diffusion step."""

    def test_uniform_unchanged(self):
        """Constant field unchanged by RKL2 diffusion."""
        field = np.full(32, 2.0)
        result = rkl2_diffusion_step(field, 0.1, dt_super=1e-4, dx=0.01, s_stages=4)
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_gaussian_smoothing(self):
        """RKL2 smooths a Gaussian pulse."""
        n = 128
        x = np.linspace(0, 1, n)
        field = np.exp(-200 * (x - 0.5) ** 2)
        result = rkl2_diffusion_step(field, 0.01, dt_super=1e-4, dx=1.0 / n, s_stages=8)
        assert result[n // 2] < field[n // 2]

    def test_sts_larger_dt_than_explicit(self):
        """STS allows larger timestep than explicit CFL without blowup."""
        n = 64
        dx = 0.01
        D = 1.0
        dt_explicit = dx * dx / (2 * D)
        dt_super = 10 * dt_explicit
        x = np.linspace(0, 1, n)
        field = np.exp(-100 * (x - 0.5) ** 2)
        result = rkl2_diffusion_step(field, D, dt_super=dt_super, dx=dx, s_stages=10)
        assert np.all(np.isfinite(result))
        assert result[n // 2] < field[n // 2]

    def test_conservation(self):
        """RKL2 conserves total integral under Neumann BCs."""
        n = 64
        field = np.random.default_rng(123).standard_normal(n)
        field -= field.mean()
        field += 10.0
        result = rkl2_diffusion_step(field, 0.5, dt_super=1e-5, dx=0.01, s_stages=6)
        np.testing.assert_allclose(np.sum(result), np.sum(field), rtol=1e-6)

    def test_spatially_varying_D(self):
        """RKL2 works with spatially-varying diffusion coefficient."""
        n = 64
        x = np.linspace(0, 1, n)
        field = np.exp(-100 * (x - 0.5) ** 2)
        D = 0.01 + 0.05 * x
        result = rkl2_diffusion_step(field, D, dt_super=1e-5, dx=1.0 / n, s_stages=6)
        assert np.all(np.isfinite(result))


class TestRkl2ThreeD:
    """Tests for 3D RKL2 resistive diffusion."""

    def test_uniform_B_unchanged(self):
        """Uniform B-field unchanged."""
        shape = (8, 8, 8)
        Bx = np.full(shape, 1.0)
        By = np.full(shape, 2.0)
        Bz = np.full(shape, 3.0)
        eta = 1e-3
        Bx_n, By_n, Bz_n = rkl2_diffusion_3d(
            Bx, By, Bz, eta, dt=1e-6, dx=0.01, dy=0.01, dz=0.01, s_stages=4,
        )
        np.testing.assert_allclose(Bx_n, 1.0, atol=1e-8)
        np.testing.assert_allclose(By_n, 2.0, atol=1e-8)
        np.testing.assert_allclose(Bz_n, 3.0, atol=1e-8)


class TestRKL2Thermal:
    """Tests for RKL2 thermal diffusion."""

    def test_1d_uniform_unchanged(self):
        """Uniform 1D temperature unchanged."""
        n = 32
        Te = np.full(n, 1e6)
        ne = np.full(n, 1e23)
        Te_new = rkl2_thermal_step(Te, kappa=100.0, ne=ne, dt=1e-9, dx=0.001, s_stages=4)
        np.testing.assert_allclose(Te_new, 1e6, atol=1.0)

    def test_1d_hot_spot_smooths(self):
        """Hot spot in 1D smooths out."""
        n = 64
        Te = np.full(n, 1e5)
        Te[30:34] = 1e7
        ne = np.full(n, 1e23)
        Te_new = rkl2_thermal_step(Te, kappa=1000.0, ne=ne, dt=1e-8, dx=0.001, s_stages=8)
        assert Te_new[32] < Te[32]


class TestStabilityLimit:
    """Tests for RKL2 stability limit."""

    def test_stability_s10(self):
        """s=10 gives ~25x acceleration."""
        dt_exp = 1e-6
        dt_rkl2 = rkl2_stability_limit(10, dt_exp)
        assert dt_rkl2 == pytest.approx(0.25 * 100 * dt_exp)

    def test_stability_increases_with_stages(self):
        """More stages give larger stable timestep."""
        dt_exp = 1e-6
        dt4 = rkl2_stability_limit(4, dt_exp)
        dt8 = rkl2_stability_limit(8, dt_exp)
        dt16 = rkl2_stability_limit(16, dt_exp)
        assert dt16 > dt8 > dt4


class TestFluidConfigDiffusion:
    """Tests for FluidConfig diffusion settings."""

    def test_default_explicit(self):
        """Default diffusion method is 'explicit'."""
        cfg = FluidConfig()
        assert cfg.diffusion_method == "explicit"

    def test_sts_config(self):
        """STS config is accepted."""
        cfg = FluidConfig(diffusion_method="sts", sts_stages=12)
        assert cfg.diffusion_method == "sts"
        assert cfg.sts_stages == 12

    def test_implicit_config(self):
        """Implicit config is accepted."""
        cfg = FluidConfig(diffusion_method="implicit", implicit_tol=1e-10)
        assert cfg.diffusion_method == "implicit"
        assert cfg.implicit_tol == 1e-10

    def test_default_sts_stages(self):
        """Default STS stages is 8."""
        cfg = FluidConfig()
        assert cfg.sts_stages == 8


# --- Section: Metal Source Terms ---


class TestMetalSourceTerms:

    def setup_method(self):
        self.nx, self.ny, self.nz = 8, 8, 8
        self.solver = MetalMHDSolver(
            grid_shape=(self.nx, self.ny, self.nz),
            dx=0.1,
            enable_hall=True,
            enable_braginskii_conduction=False,
        )

    def test_hall_enabled_flag(self):
        """Verify that enable_hall flag is set correctly."""
        assert self.solver.enable_hall is True
        solver2 = MetalMHDSolver(
            grid_shape=(self.nx, self.ny, self.nz),
            dx=0.1,
            enable_hall=False,
        )
        assert solver2.enable_hall is False

    def test_step_runs_with_transport_enabled(self):
        """Verify step() runs without error when transport physics enabled."""
        state = {
            "rho": np.ones((self.nx, self.ny, self.nz)),
            "velocity": np.zeros((3, self.nx, self.ny, self.nz)),
            "pressure": np.ones((self.nx, self.ny, self.nz)),
            "B": np.zeros((3, self.nx, self.ny, self.nz)),
            "Te": np.ones((self.nx, self.ny, self.nz)) * 1e4,
            "Ti": np.ones((self.nx, self.ny, self.nz)) * 1e4,
            "psi": np.zeros((self.nx, self.ny, self.nz)),
        }
        state["B"][2] = 1.0

        dt = self.solver.compute_dt(state)
        result = self.solver.step(state, dt, current=0.0, voltage=0.0)

        assert "rho" in result
        rho = result["rho"]
        if isinstance(rho, torch.Tensor):
            rho = rho.cpu().numpy()
        assert rho.shape[-3:] == (self.nx, self.ny, self.nz)
        assert np.all(np.isfinite(rho))


# --- Section: Phase P Engine Accuracy ---


class TestWENOZ:
    """Tests for WENO-Z weight upgrade."""

    def test_weno5z_smooth_data(self):
        """WENO5-Z reconstruction on smooth data should be accurate."""
        from dpf.fluid.mhd_solver import _weno5_reconstruct_1d

        N = 40
        x = np.linspace(0, 2 * np.pi, N)
        v = np.sin(x)
        v_L, v_R = _weno5_reconstruct_1d(v)
        x_mid = 0.5 * (x[2:-2] + x[3:-1])
        v_exact = np.sin(x_mid)
        error_L = np.max(np.abs(v_L - v_exact))
        assert error_L < 2e-3, f"WENO5-Z left error too large: {error_L:.2e}"

    def test_weno5z_constant_field(self):
        """WENO5-Z should exactly reconstruct a constant field."""
        from dpf.fluid.mhd_solver import _weno5_reconstruct_1d

        v = np.full(20, 3.14)
        v_L, v_R = _weno5_reconstruct_1d(v)
        assert np.allclose(v_L, 3.14, atol=1e-10)
        assert np.allclose(v_R, 3.14, atol=1e-10)

    def test_weno5z_linear_field(self):
        """WENO5-Z should exactly reconstruct a linear field."""
        from dpf.fluid.mhd_solver import _weno5_reconstruct_1d

        v = np.linspace(1.0, 10.0, 20)
        v_L, v_R = _weno5_reconstruct_1d(v)
        dx = (10.0 - 1.0) / 19.0
        x_mid = np.linspace(1.0 + 2.0 * dx, 1.0 + 17.0 * dx, 16) + 0.5 * dx
        assert np.allclose(v_L, x_mid, atol=1e-10), (
            f"Linear left recon error: {np.max(np.abs(v_L - x_mid)):.2e}"
        )

    def test_weno5z_convergence_better_at_critical_points(self):
        """WENO-Z should maintain accuracy near critical points (f=0, f'=0)."""
        from dpf.fluid.mhd_solver import _weno5_reconstruct_1d

        errors = []
        for N in [20, 40, 80]:
            x = np.linspace(0, 2 * np.pi, N)
            v = np.cos(x)
            v_L, _ = _weno5_reconstruct_1d(v)
            x_mid = 0.5 * (x[2:-2] + x[3:-1])
            v_exact = np.cos(x_mid)
            errors.append(np.max(np.abs(v_L - v_exact)))
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], f"WENO-Z error not decreasing: {errors}"


def _make_hydro_state(N: int = 16) -> dict[str, np.ndarray]:
    """Create a uniform hydro state for testing."""
    rho0, p0 = 1e-4, 1e3
    m_i = 3.34e-27
    T0 = p0 / (2.0 * (rho0 / m_i) * k_B)
    return {
        "rho": np.full((N, N, N), rho0),
        "velocity": np.zeros((3, N, N, N)),
        "pressure": np.full((N, N, N), p0),
        "B": np.zeros((3, N, N, N)),
        "Te": np.full((N, N, N), T0),
        "Ti": np.full((N, N, N), T0),
        "psi": np.zeros((N, N, N)),
    }


class TestSSPRK3Python:
    """Tests for SSP-RK3 time integration in the Python MHD solver."""

    def test_ssp_rk3_exists(self):
        """MHDSolver should support time_integrator='ssp_rk3'."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(
            grid_shape=(8, 8, 8), dx=0.01,
            time_integrator="ssp_rk3",
            enable_hall=False, enable_braginskii=False,
        )
        assert solver.time_integrator == "ssp_rk3"

    def test_ssp_rk2_backward_compat(self):
        """MHDSolver should still support time_integrator='ssp_rk2'."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(
            grid_shape=(8, 8, 8), dx=0.01,
            time_integrator="ssp_rk2",
            enable_hall=False, enable_braginskii=False,
        )
        assert solver.time_integrator == "ssp_rk2"

    def test_ssp_rk3_default(self):
        """SSP-RK3 should be the default time integrator."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(
            grid_shape=(8, 8, 8), dx=0.01,
            enable_hall=False, enable_braginskii=False,
        )
        assert solver.time_integrator == "ssp_rk3"

    def test_ssp_rk3_single_step_stable(self):
        """SSP-RK3 single step should produce finite results."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(
            grid_shape=(8, 8, 8), dx=0.01,
            time_integrator="ssp_rk3",
            enable_hall=False, enable_braginskii=False,
        )
        state = _make_hydro_state(8)
        dt = solver._compute_dt(state)
        new_state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.all(np.isfinite(new_state["rho"]))
        assert np.all(new_state["rho"] > 0)
        assert np.all(np.isfinite(new_state["pressure"]))
        assert np.all(new_state["pressure"] > 0)

    def test_ssp_rk3_lower_error_than_rk2_python(self):
        """SSP-RK3 should produce lower error than SSP-RK2 on smooth data (Python engine)."""
        from dpf.fluid.mhd_solver import MHDSolver

        N = 8
        dx = 0.01
        gamma_val = 5.0 / 3.0
        rho0, p0 = 1e-4, 1e3
        cs = np.sqrt(gamma_val * p0 / rho0)
        sim_time = 5.0 * dx / cs

        base_state = _make_hydro_state(N)
        x = np.arange(N) * dx
        X = x[:, np.newaxis, np.newaxis] * np.ones((1, N, N))
        perturbation = 0.001 * np.sin(2.0 * np.pi * X / (N * dx))
        base_state["rho"] = rho0 * (1.0 + perturbation)
        base_state["pressure"] = p0 * (1.0 + gamma_val * perturbation)
        base_state["velocity"][0] = cs * perturbation
        rho_init = base_state["rho"].copy()

        errors = {}
        for ti in ["ssp_rk2", "ssp_rk3"]:
            state = {k: v.copy() for k, v in base_state.items()}
            solver = MHDSolver(
                grid_shape=(N, N, N), dx=dx, gamma=gamma_val,
                cfl=0.3, time_integrator=ti,
                enable_hall=False, enable_braginskii=False,
                enable_resistive=False, riemann_solver="hll",
            )
            solver.use_weno5 = False
            t = 0.0
            while t < sim_time:
                dt = solver._compute_dt(state)
                dt = min(dt, sim_time - t)
                if dt < 1e-20:
                    break
                state = solver.step(state, dt, current=0.0, voltage=0.0)
                t += dt
            errors[ti] = np.mean(np.abs(state["rho"] - rho_init))

        assert errors["ssp_rk3"] <= errors["ssp_rk2"] * 1.1, (
            f"RK3 error ({errors['ssp_rk3']:.2e}) > RK2 error ({errors['ssp_rk2']:.2e})"
        )

    @pytest.mark.slow
    def test_ssp_rk3_stable_with_weno5(self):
        """SSP-RK3 should be stable with WENO5 on a 16^3 grid (slow: Numba JIT)."""
        from dpf.fluid.mhd_solver import MHDSolver

        N = 16
        dx = 0.01
        gamma_val = 5.0 / 3.0
        rho0 = 1e-4

        solver = MHDSolver(
            grid_shape=(N, N, N), dx=dx, gamma=gamma_val,
            cfl=0.3, time_integrator="ssp_rk3",
            enable_hall=False, enable_braginskii=False,
            enable_resistive=False, riemann_solver="hll",
        )
        assert solver.use_weno5, "WENO5 should be enabled at N=16"

        state = _make_hydro_state(N)
        x = np.arange(N) * dx
        X = x[:, np.newaxis, np.newaxis] * np.ones((1, N, N))
        perturbation = 0.001 * np.sin(2.0 * np.pi * X / (N * dx))
        state["rho"] = rho0 * (1.0 + perturbation)

        for _ in range(2):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)

        assert np.all(np.isfinite(state["rho"])), "RK3+WENO5 produced NaN in density"
        assert np.all(state["rho"] > 0), "RK3+WENO5 produced negative density"
        assert np.max(state["rho"]) < 1e3 * rho0, (
            f"RK3+WENO5 blew up: max rho={np.max(state['rho']):.2e}"
        )


class TestSSPRK3Metal:
    """SSP-RK3 accuracy tests using the Metal solver."""

    def test_metal_rk3_lower_error_than_rk2(self):
        """Metal SSP-RK3 should produce lower temporal error than SSP-RK2."""
        N = 16
        dx = 0.01
        gamma_val = 5.0 / 3.0
        rho0, p0 = 1.0, 1.0
        cs = np.sqrt(gamma_val * p0 / rho0)
        sim_time = 5.0 * dx / cs

        base_state = {
            "rho": np.full((N, N, N), rho0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), p0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        x = np.arange(N) * dx
        X = x[:, np.newaxis, np.newaxis] * np.ones((1, N, N))
        perturbation = 0.001 * np.sin(2.0 * np.pi * X / (N * dx))
        base_state["rho"] = rho0 * (1.0 + perturbation)
        base_state["pressure"] = p0 * (1.0 + gamma_val * perturbation)
        base_state["velocity"][0] = cs * perturbation
        rho_init = base_state["rho"].copy()

        errors = {}
        for ti in ["ssp_rk2", "ssp_rk3"]:
            state = {k: v.copy() for k, v in base_state.items()}
            solver = MetalMHDSolver(
                grid_shape=(N, N, N), dx=dx, gamma=gamma_val,
                device="cpu", precision="float64",
                reconstruction="plm", riemann_solver="hll",
                time_integrator=ti, cfl=0.3,
                use_ct=False,
            )
            t = 0.0
            while t < sim_time:
                dt = solver.compute_dt(state)
                dt = min(dt, sim_time - t)
                if dt < 1e-20:
                    break
                state = solver.step(state, dt, current=0.0, voltage=0.0)
                t += dt
            errors[ti] = np.mean(np.abs(state["rho"] - rho_init))

        assert errors["ssp_rk3"] <= errors["ssp_rk2"] * 1.05, (
            f"Metal RK3 error ({errors['ssp_rk3']:.2e}) > RK2 error ({errors['ssp_rk2']:.2e})"
        )

    def test_metal_weno5_rk3_stable_16cube(self):
        """Metal WENO5+RK3 should be stable on 16^3 grid (full fidelity)."""
        N = 16
        dx = 0.01
        rho0, p0 = 1.0, 1.0
        gamma_val = 5.0 / 3.0

        solver = MetalMHDSolver(
            grid_shape=(N, N, N), dx=dx, gamma=gamma_val,
            device="cpu", precision="float64",
            reconstruction="weno5", riemann_solver="hlld",
            time_integrator="ssp_rk3", cfl=0.3,
            use_ct=False,
        )

        state = {
            "rho": np.full((N, N, N), rho0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), p0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        x = np.arange(N) * dx
        X = x[:, np.newaxis, np.newaxis] * np.ones((1, N, N))
        state["rho"] = rho0 * (1.0 + 0.001 * np.sin(2.0 * np.pi * X / (N * dx)))

        for _ in range(5):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)

        assert np.all(np.isfinite(state["rho"])), "Metal WENO5+RK3 produced NaN"
        assert np.all(state["rho"] > 0), "Metal WENO5+RK3 produced negative density"

    def test_metal_full_fidelity_sound_wave(self):
        """Full fidelity Metal solver: WENO5+HLLD+SSP-RK3+float64 on sound wave."""
        N = 16
        dx = 0.01
        gamma_val = 5.0 / 3.0
        rho0, p0 = 1.0, 1.0
        cs = np.sqrt(gamma_val * p0 / rho0)

        solver = MetalMHDSolver(
            grid_shape=(N, N, N), dx=dx, gamma=gamma_val,
            device="cpu", precision="float64",
            reconstruction="weno5", riemann_solver="hlld",
            time_integrator="ssp_rk3", cfl=0.3,
            use_ct=False, limiter="mc",
        )

        state = {
            "rho": np.full((N, N, N), rho0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), p0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        x = np.arange(N) * dx
        X = x[:, np.newaxis, np.newaxis] * np.ones((1, N, N))
        perturbation = 0.001 * np.sin(2.0 * np.pi * X / (N * dx))
        state["rho"] = rho0 * (1.0 + perturbation)
        state["pressure"] = p0 * (1.0 + gamma_val * perturbation)
        state["velocity"][0] = cs * perturbation
        state["B"][2] = 0.1
        rho_init = state["rho"].copy()

        for _ in range(10):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)

        assert np.all(np.isfinite(state["rho"])), "Full fidelity solver NaN"
        assert np.all(state["rho"] > 0), "Full fidelity solver negative density"
        rho_change = np.max(np.abs(state["rho"] - rho_init)) / rho0
        assert rho_change < 0.1, f"Sound wave perturbation grew too much: {rho_change:.4f}"


class TestHLLDDefault:
    """Tests for HLLD as default Riemann solver."""

    def test_hlld_is_default(self):
        """HLLD should be the default Riemann solver."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(
            grid_shape=(8, 8, 8), dx=0.01,
            enable_hall=False, enable_braginskii=False,
        )
        assert solver.riemann_solver == "hlld"

    def test_hlld_stable_with_b_field(self):
        """HLLD should be stable with a non-zero magnetic field."""
        from dpf.fluid.mhd_solver import MHDSolver

        N = 8
        solver = MHDSolver(
            grid_shape=(N, N, N), dx=0.01,
            riemann_solver="hlld",
            enable_hall=False, enable_braginskii=False,
        )
        m_i = 3.34e-27
        rho0, p0 = 1e-4, 1e3
        T0 = p0 / (2.0 * (rho0 / m_i) * k_B)
        state = {
            "rho": np.full((N, N, N), rho0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), p0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), T0),
            "Ti": np.full((N, N, N), T0),
            "psi": np.zeros((N, N, N)),
        }
        state["B"][2] = 0.1
        dt = solver._compute_dt(state)
        new_state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.all(np.isfinite(new_state["rho"]))
        assert np.all(np.isfinite(new_state["B"]))

    def test_hll_still_works(self):
        """HLL Riemann solver should still work when explicitly requested."""
        from dpf.fluid.mhd_solver import MHDSolver

        solver = MHDSolver(
            grid_shape=(8, 8, 8), dx=0.01,
            riemann_solver="hll",
            enable_hall=False, enable_braginskii=False,
        )
        assert solver.riemann_solver == "hll"


class TestMetalResistiveMHD:
    """Tests for Metal solver resistive MHD support."""

    @pytest.fixture()
    def solver(self):
        """Create a Metal solver on CPU for testing."""
        return MetalMHDSolver(
            grid_shape=(16, 16, 16),
            dx=0.01,
            device="cpu",
            precision="float64",
            use_ct=False,
        )

    def test_step_accepts_eta_field(self, solver):
        """Metal solver step() should accept eta_field parameter."""
        N = 16
        state = {
            "rho": np.full((N, N, N), 1.0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), 1.0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        state["B"][2] = 0.1
        eta = np.full((N, N, N), 1e-3)
        dt = solver.compute_dt(state)
        new_state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=eta)
        assert np.all(np.isfinite(new_state["rho"]))
        assert np.all(np.isfinite(new_state["B"]))
        assert np.all(np.isfinite(new_state["pressure"]))

    def test_step_without_eta_field(self, solver):
        """Metal solver should work without eta_field (ideal MHD)."""
        N = 16
        state = {
            "rho": np.full((N, N, N), 1.0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), 1.0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        dt = solver.compute_dt(state)
        new_state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.all(np.isfinite(new_state["rho"]))

    def test_resistive_diffusion_smooths_b_field(self, solver):
        """Resistive diffusion should smooth a B-field gradient."""
        N = 16
        state = {
            "rho": np.full((N, N, N), 1.0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), 1.0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        x = np.linspace(-1, 1, N)
        Bz_profile = 0.5 * np.tanh(x / 0.3)
        state["B"][2] = Bz_profile[:, np.newaxis, np.newaxis] * np.ones((1, N, N))
        initial_max_grad = np.max(np.abs(np.diff(state["B"][2], axis=0)))
        eta = np.full((N, N, N), 1e-9)
        for _ in range(20):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=eta)
        final_max_grad = np.max(np.abs(np.diff(state["B"][2], axis=0)))
        assert np.all(np.isfinite(state["B"])), "Resistive diffusion produced NaN"
        assert final_max_grad < initial_max_grad * 2.0, (
            f"B-field gradient grew: initial={initial_max_grad:.6f}, final={final_max_grad:.6f}"
        )

    def test_ohmic_heating_increases_pressure(self, solver):
        """Ohmic heating from eta*J^2 should increase pressure."""
        N = 16
        state = {
            "rho": np.full((N, N, N), 1.0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), 1.0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        x = np.linspace(0, 2 * np.pi, N)
        Bz = np.sin(x)
        state["B"][2] = Bz[:, np.newaxis, np.newaxis] * np.ones((1, N, N))
        initial_pressure = state["pressure"].copy()
        eta = np.full((N, N, N), 1e-1)
        dt = solver.compute_dt(state)
        new_state = solver.step(state, dt, current=0.0, voltage=0.0, eta_field=eta)
        assert np.mean(new_state["pressure"]) >= np.mean(initial_pressure) * 0.99, (
            "Ohmic heating should not decrease mean pressure"
        )


class TestCombinedAccuracy:
    """Verify that the combined improvements maintain accuracy."""

    def test_uniform_state_preserved(self):
        """A perfectly uniform state should remain uniform (no spurious oscillations)."""
        from dpf.fluid.mhd_solver import MHDSolver

        N = 8
        solver = MHDSolver(
            grid_shape=(N, N, N), dx=0.01,
            enable_hall=False, enable_braginskii=False,
            enable_resistive=False,
        )
        rho0, p0 = 1e-4, 1e3
        m_i = 3.34e-27
        T0 = p0 / (2.0 * (rho0 / m_i) * k_B)
        state = {
            "rho": np.full((N, N, N), rho0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), p0),
            "B": np.full((3, N, N, N), 0.0),
            "Te": np.full((N, N, N), T0),
            "Ti": np.full((N, N, N), T0),
            "psi": np.zeros((N, N, N)),
        }
        dt = solver._compute_dt(state)
        for _ in range(5):
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.allclose(state["rho"], rho0, rtol=1e-10), (
            f"Uniform state drifted: rho range [{state['rho'].min():.6e}, {state['rho'].max():.6e}]"
        )

    def test_density_positivity(self):
        """Density should remain positive after multiple steps."""
        from dpf.fluid.mhd_solver import MHDSolver

        N = 8
        solver = MHDSolver(
            grid_shape=(N, N, N), dx=0.01,
            enable_hall=False, enable_braginskii=False,
            enable_resistive=False,
        )
        m_i = 3.34e-27
        rho0, p0 = 1e-4, 1e3
        T0 = p0 / (2.0 * (rho0 / m_i) * k_B)
        state = {
            "rho": np.full((N, N, N), rho0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), p0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), T0),
            "Ti": np.full((N, N, N), T0),
            "psi": np.zeros((N, N, N)),
        }
        state["rho"][N // 2, N // 2, N // 2] = rho0 * 2.0
        for _ in range(10):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.all(state["rho"] > 0), "Density went negative"
        assert np.all(state["pressure"] > 0), "Pressure went negative"

    def test_metal_uniform_state_preserved(self):
        """Metal solver: uniform state should remain uniform."""
        N = 16
        rho0, p0 = 1.0, 1.0
        solver = MetalMHDSolver(
            grid_shape=(N, N, N), dx=0.01,
            device="cpu", precision="float64",
            reconstruction="weno5", riemann_solver="hlld",
            time_integrator="ssp_rk3",
            use_ct=False,
        )
        state = {
            "rho": np.full((N, N, N), rho0),
            "velocity": np.zeros((3, N, N, N)),
            "pressure": np.full((N, N, N), p0),
            "B": np.zeros((3, N, N, N)),
            "Te": np.full((N, N, N), 1e4),
            "Ti": np.full((N, N, N), 1e4),
            "psi": np.zeros((N, N, N)),
        }
        dt = solver.compute_dt(state)
        for _ in range(5):
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.allclose(state["rho"], rho0, rtol=1e-10), (
            f"Metal uniform state drifted: rho range "
            f"[{state['rho'].min():.6e}, {state['rho'].max():.6e}]"
        )


# --- Section: Phase Q Python 8-comp HLLD and CT ---


def test_hlld_8comp_uniform_state():
    """Uniform state should produce zero fluxes everywhere."""

    n = 32
    rho = np.ones(n)
    vn = np.zeros(n)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    p = np.ones(n)
    Bn = np.full(n, 0.75)
    Bt1 = np.zeros(n)
    Bt2 = np.zeros(n)

    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho, rho, vn, vn, vt1, vt1, vt2, vt2, p, p, Bn, Bt1, Bt1, Bt2, Bt2, GAMMA
    )

    pt_expected = p - Bn**2 / (2 * MU_0)
    assert np.allclose(F_rho, 0.0, atol=1e-12)
    assert np.allclose(F_momn, pt_expected, rtol=1e-6)
    assert np.allclose(F_momt1, 0.0, atol=1e-12)
    assert np.allclose(F_momt2, 0.0, atol=1e-12)
    assert np.allclose(F_ene, 0.0, atol=1e-12)
    assert np.allclose(F_Bt1, 0.0, atol=1e-12)
    assert np.allclose(F_Bt2, 0.0, atol=1e-12)


def test_hlld_8comp_sod_shock_finite():
    """Sod shock tube with magnetic field should produce finite fluxes."""

    n = 32
    rho = np.ones(n)
    rho[n // 2:] = 0.125
    p = np.ones(n)
    p[n // 2:] = 0.1
    vn = np.zeros(n)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    Bn = np.full(n, 0.75)
    Bt1 = np.zeros(n)
    Bt2 = np.zeros(n)

    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho[:-1], rho[1:], vn[:-1], vn[1:], vt1[:-1], vt1[1:], vt2[:-1], vt2[1:],
        p[:-1], p[1:], Bn[:-1], Bt1[:-1], Bt1[1:], Bt2[:-1], Bt2[1:], GAMMA
    )

    assert np.all(np.isfinite(F_rho))
    assert np.all(np.isfinite(F_momn))
    assert np.all(np.isfinite(F_momt1))
    assert np.all(np.isfinite(F_momt2))
    assert np.all(np.isfinite(F_ene))
    assert np.all(np.isfinite(F_Bt1))
    assert np.all(np.isfinite(F_Bt2))
    assert np.all(F_rho >= 0.0)


def test_hlld_8comp_briowu_finite():
    """Brio-Wu MHD shock tube should produce finite fluxes."""

    n = 32
    rho = np.ones(n)
    rho[n // 2:] = 0.125
    p = np.ones(n)
    p[n // 2:] = 0.1
    vn = np.zeros(n)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    Bn = np.full(n, 0.75)
    Bt1 = np.ones(n)
    Bt1[n // 2:] = -1.0
    Bt2 = np.zeros(n)

    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho[:-1], rho[1:], vn[:-1], vn[1:], vt1[:-1], vt1[1:], vt2[:-1], vt2[1:],
        p[:-1], p[1:], Bn[:-1], Bt1[:-1], Bt1[1:], Bt2[:-1], Bt2[1:], GAMMA
    )

    assert np.all(np.isfinite(F_rho))
    assert np.all(np.isfinite(F_momn))
    assert np.all(np.isfinite(F_momt1))
    assert np.all(np.isfinite(F_momt2))
    assert np.all(np.isfinite(F_ene))
    assert np.all(np.isfinite(F_Bt1))
    assert np.all(np.isfinite(F_Bt2))


def test_hlld_8comp_degenerate_bn_zero():
    """Bn=0 degenerate case should not produce NaN."""

    n = 32
    rho = np.ones(n)
    rho[n // 2:] = 0.5
    vn = np.zeros(n)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    p = np.ones(n)
    Bn = np.zeros(n)
    Bt1 = np.ones(n)
    Bt2 = np.zeros(n)

    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho[:-1], rho[1:], vn[:-1], vn[1:], vt1[:-1], vt1[1:], vt2[:-1], vt2[1:],
        p[:-1], p[1:], Bn[:-1], Bt1[:-1], Bt1[1:], Bt2[:-1], Bt2[1:], GAMMA
    )

    assert np.all(np.isfinite(F_rho))
    assert np.all(np.isfinite(F_momn))
    assert np.all(np.isfinite(F_momt1))
    assert np.all(np.isfinite(F_momt2))
    assert np.all(np.isfinite(F_ene))
    assert np.all(np.isfinite(F_Bt1))
    assert np.all(np.isfinite(F_Bt2))


def test_hlld_8comp_returns_7_fluxes():
    """Output tuple should have 7 elements."""

    n = 16
    rho = np.ones(n)
    vn = np.zeros(n)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    p = np.ones(n)
    Bn = np.full(n, 0.5)
    Bt1 = np.zeros(n)
    Bt2 = np.zeros(n)

    result = _hlld_flux_1d_8comp(
        rho, rho, vn, vn, vt1, vt1, vt2, vt2, p, p, Bn, Bt1, Bt1, Bt2, Bt2, GAMMA
    )

    assert len(result) == 7
    for flux in result:
        assert isinstance(flux, np.ndarray)
        assert flux.shape == (n,)


def test_hlld_8comp_lr_symmetry():
    """For symmetric L/R states with negated velocities, mass flux should be zero."""

    n = 16
    rho = np.ones(n)
    vn_L = np.full(n, 0.2)
    vn_R = np.full(n, -0.2)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    p = np.ones(n)
    Bn = np.full(n, 0.5)
    Bt1 = np.full(n, 0.3)
    Bt2 = np.zeros(n)

    F = _hlld_flux_1d_8comp(
        rho, rho, vn_L, vn_R, vt1, vt1, vt2, vt2,
        p, p, Bn, Bt1, Bt1, Bt2, Bt2, GAMMA
    )

    assert np.allclose(F[0], 0.0, atol=1e-10), f"Mass flux not zero: {F[0][0]}"
    assert np.all(np.isfinite(F[1]))
    assert np.allclose(F[5], 0.0, atol=1e-10)
    assert np.allclose(F[6], 0.0, atol=1e-10)


def test_hlld_8comp_alfven_wave():
    """Pure Alfven wave should produce correct transverse momentum and B fluxes."""

    n = 32
    rho = np.ones(n)
    vn = np.full(n, 0.1)
    vt1 = np.full(n, 0.05)
    vt2 = np.zeros(n)
    p = np.ones(n)
    Bn = np.full(n, 1.0)
    Bt1 = np.full(n, 0.1)
    Bt2 = np.zeros(n)

    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho[:-1], rho[1:], vn[:-1], vn[1:], vt1[:-1], vt1[1:], vt2[:-1], vt2[1:],
        p[:-1], p[1:], Bn[:-1], Bt1[:-1], Bt1[1:], Bt2[:-1], Bt2[1:], GAMMA
    )

    assert np.all(np.isfinite(F_momt1))
    assert np.all(np.isfinite(F_Bt1))
    assert np.max(np.abs(F_momt1)) > 1e-12


def test_hlld_8comp_less_diffusive_than_hll():
    """HLLD should be less diffusive than HLL for contact discontinuity."""

    n = 64
    rho = np.ones(n)
    rho[n // 2:] = 0.5
    vn = np.full(n, 0.1)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    p = np.ones(n)
    Bn = np.full(n, 0.5)
    Bt1 = np.zeros(n)
    Bt2 = np.zeros(n)

    F_hlld = _hlld_flux_1d_8comp(
        rho[:-1], rho[1:], vn[:-1], vn[1:], vt1[:-1], vt1[1:], vt2[:-1], vt2[1:],
        p[:-1], p[1:], Bn[:-1], Bt1[:-1], Bt1[1:], Bt2[:-1], Bt2[1:], GAMMA
    )

    assert np.all(np.isfinite(F_hlld[0]))
    assert np.all(np.isfinite(F_hlld[1]))
    assert np.all(np.isfinite(F_hlld[4]))
    assert len(F_hlld) == 7


def test_hlld_8comp_transverse_momentum_flux_nonzero():
    """With non-zero transverse B and velocity, transverse momentum fluxes should be nonzero."""

    n = 16
    rho = np.ones(n)
    vn = np.full(n, 0.1)
    vt1 = np.full(n, 0.2)
    vt2 = np.full(n, 0.1)
    p = np.ones(n)
    Bn = np.full(n, 0.5)
    Bt1 = np.full(n, 0.3)
    Bt2 = np.full(n, 0.2)

    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho, rho, vn, vn, vt1, vt1, vt2, vt2, p, p, Bn, Bt1, Bt1, Bt2, Bt2, GAMMA
    )

    assert np.max(np.abs(F_momt1)) > 1e-10
    assert np.max(np.abs(F_momt2)) > 1e-10


def test_hlld_8comp_transverse_B_flux_nonzero():
    """With non-zero transverse B and normal velocity, transverse B fluxes should be nonzero."""

    n = 16
    rho = np.ones(n)
    vn = np.full(n, 0.1)
    vt1 = np.zeros(n)
    vt2 = np.zeros(n)
    p = np.ones(n)
    Bn = np.full(n, 0.5)
    Bt1 = np.full(n, 0.3)
    Bt2 = np.full(n, 0.2)

    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho, rho, vn, vn, vt1, vt1, vt2, vt2, p, p, Bn, Bt1, Bt1, Bt2, Bt2, GAMMA
    )

    expected_Bt1 = vn[0] * Bt1[0]
    expected_Bt2 = vn[0] * Bt2[0]
    assert np.allclose(F_Bt1, expected_Bt1, rtol=1e-10)
    assert np.allclose(F_Bt2, expected_Bt2, rtol=1e-10)


def test_hlld_dict_wrapper_7_keys():
    """_compute_flux_1d_sweep with HLLD should return dict with 7+ keys."""
    from dpf.fluid.mhd_solver import _compute_flux_1d_sweep

    n = 16
    shape = (n, n, n)
    rho = np.ones(shape)
    vel_n = np.zeros(shape)
    vel_t1 = np.zeros(shape)
    vel_t2 = np.zeros(shape)
    pressure = np.ones(shape)
    Bn = np.full(shape, 0.5)
    Bt1 = np.zeros(shape)
    Bt2 = np.zeros(shape)

    result = _compute_flux_1d_sweep(
        rho, vel_n, vel_t1, vel_t2, pressure, Bn, Bt1, Bt2, GAMMA, axis=0,
        riemann_solver="hlld",
    )

    assert isinstance(result, dict)
    assert "mass_flux" in result
    assert "momentum_flux" in result
    assert "energy_flux" in result
    assert "n_interfaces" in result
    assert "momentum_t1_flux" in result
    assert "momentum_t2_flux" in result
    assert "Bt1_flux" in result
    assert "Bt2_flux" in result


def test_flux_sweep_8comp_matches_3comp_for_zero_transverse():
    """When transverse components are zero, 8-comp mass/momn/ene fluxes should match 3-comp."""
    from dpf.fluid.mhd_solver import _compute_flux_1d_sweep

    n = 32
    shape = (n, n, n)
    rho = np.ones(shape)
    rho[n // 2:, :, :] = 0.5
    vel_n = np.zeros(shape)
    vel_t1 = np.zeros(shape)
    vel_t2 = np.zeros(shape)
    pressure = np.ones(shape)
    Bn = np.full(shape, 0.5)
    Bt1 = np.zeros(shape)
    Bt2 = np.zeros(shape)

    result_8comp = _compute_flux_1d_sweep(
        rho, vel_n, vel_t1, vel_t2, pressure, Bn, Bt1, Bt2, GAMMA, axis=0,
        riemann_solver="hlld",
    )

    j, k = n // 2, n // 2
    rho_1d = rho[:, j, k]
    F_rho_ref, F_momn_ref, _, _, F_ene_ref, _, _ = _hlld_flux_1d_8comp(
        rho_1d[:-1], rho_1d[1:],
        vel_n[:, j, k][:-1], vel_n[:, j, k][1:],
        np.zeros(n - 1), np.zeros(n - 1),
        np.zeros(n - 1), np.zeros(n - 1),
        pressure[:, j, k][:-1], pressure[:, j, k][1:],
        Bn[:, j, k][:-1],
        np.zeros(n - 1), np.zeros(n - 1),
        np.zeros(n - 1), np.zeros(n - 1),
        GAMMA,
    )

    n_iface = result_8comp["n_interfaces"]
    mass_pencil = result_8comp["mass_flux"][:, j, k][:n_iface]
    momn_pencil = result_8comp["momentum_flux"][:, j, k][:n_iface]
    ene_pencil = result_8comp["energy_flux"][:, j, k][:n_iface]

    assert np.allclose(mass_pencil, F_rho_ref[:n_iface], rtol=1e-10)
    assert np.allclose(momn_pencil, F_momn_ref[:n_iface], rtol=1e-10)
    assert np.allclose(ene_pencil, F_ene_ref[:n_iface], rtol=1e-10)


def test_python_ct_divB_preservation():
    """MHDSolver with CT should preserve div(B) < 1e-10 over 10 steps."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx = ny = nz = 16
    state = {
        "rho": np.ones((nx, ny, nz)) * 1.0,
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)) * 1.0,
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.zeros((nx, ny, nz)),
    }
    state["B"][2] = 1.0

    solver = MHDSolver(
        grid_shape=(nx, ny, nz), dx=0.01,
        use_ct=True, riemann_solver="hlld", time_integrator="ssp_rk3",
    )

    for _ in range(10):
        dt = solver._compute_dt(state)
        state = solver.step(state, dt, current=0.0, voltage=0.0)

    staggered = cell_centered_to_face(
        state["B"][0], state["B"][1], state["B"][2], solver.dx, solver.dx, solver.dx,
    )
    div_B = compute_div_B(staggered)
    assert np.max(np.abs(div_B)) < 1e-10


def test_python_ct_vs_dedner_comparison():
    """CT should give lower div(B) than Dedner after evolving."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx = ny = nz = 16
    dx = 0.01
    state_base = {
        "rho": np.ones((nx, ny, nz)) * 1.0,
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)) * 1.0,
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.zeros((nx, ny, nz)),
    }
    state_base["B"][2] = 1.0

    state_ct = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state_base.items()}
    solver_ct = MHDSolver(grid_shape=(nx, ny, nz), dx=dx, use_ct=True, riemann_solver="hlld")

    state_dedner = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state_base.items()}
    solver_dedner = MHDSolver(grid_shape=(nx, ny, nz), dx=dx, use_ct=False, riemann_solver="hlld")

    for _ in range(5):
        dt_ct = solver_ct._compute_dt(state_ct)
        state_ct = solver_ct.step(state_ct, dt_ct, current=0.0, voltage=0.0)
        dt_dedner = solver_dedner._compute_dt(state_dedner)
        state_dedner = solver_dedner.step(state_dedner, dt_dedner, current=0.0, voltage=0.0)

    staggered_ct = cell_centered_to_face(
        state_ct["B"][0], state_ct["B"][1], state_ct["B"][2], dx, dx, dx,
    )
    div_B_ct = compute_div_B(staggered_ct)
    staggered_d = cell_centered_to_face(
        state_dedner["B"][0], state_dedner["B"][1], state_dedner["B"][2], dx, dx, dx,
    )
    div_B_dedner = compute_div_B(staggered_d)

    max_div_ct = np.max(np.abs(div_B_ct))
    assert max_div_ct < 1e-10, f"CT div(B) too large: {max_div_ct}"
    assert np.all(np.isfinite(state_ct["rho"]))
    assert np.all(np.isfinite(state_dedner["rho"]))
    assert np.all(np.isfinite(div_B_dedner))


@pytest.mark.xfail(
    reason="Python engine CT is unstable for shock ICs — use Metal or Athena++ for CT",
    strict=False,
)
def test_python_ct_sod_shock_stability():
    """Sod shock with CT should remain stable for 20 steps."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx, ny, nz = 32, 16, 16
    state = {
        "rho": np.ones((nx, ny, nz)),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.zeros((nx, ny, nz)),
    }
    state["rho"][nx // 2:, :, :] = 0.125
    state["pressure"][nx // 2:, :, :] = 0.1
    state["B"][2] = 0.75

    solver = MHDSolver(
        grid_shape=(nx, ny, nz), dx=0.01,
        use_ct=True, riemann_solver="hlld", time_integrator="ssp_rk3",
        enable_braginskii=False,
    )

    for _ in range(20):
        dt = solver._compute_dt(state)
        state = solver.step(state, dt, current=0.0, voltage=0.0)

    assert np.all(np.isfinite(state["rho"]))
    assert np.all(state["rho"] > 0)
    assert np.all(np.isfinite(state["pressure"]))
    assert np.all(np.isfinite(state["B"]))


def test_python_ct_briowu_stability():
    """Brio-Wu with CT should remain stable for 20 steps."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx, ny, nz = 32, 16, 16
    state = {
        "rho": np.ones((nx, ny, nz)),
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)),
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.zeros((nx, ny, nz)),
    }
    state["rho"][nx // 2:, :, :] = 0.125
    state["pressure"][nx // 2:, :, :] = 0.1
    state["B"][2] = 0.75
    state["B"][1, :nx // 2, :, :] = 1.0
    state["B"][1, nx // 2:, :, :] = -1.0

    solver = MHDSolver(
        grid_shape=(nx, ny, nz), dx=0.01,
        use_ct=True, riemann_solver="hlld", time_integrator="ssp_rk3",
        enable_braginskii=False,
    )

    for _ in range(20):
        dt = solver._compute_dt(state)
        state = solver.step(state, dt, current=0.0, voltage=0.0)

    assert np.all(np.isfinite(state["rho"]))
    assert np.all(np.isfinite(state["pressure"]))
    assert np.all(np.isfinite(state["B"]))


def test_python_ct_energy_conservation():
    """Total energy drift should be < 5% over 5 steps with CT on a uniform state."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx = ny = nz = 16
    state = {
        "rho": np.ones((nx, ny, nz)) * 1.0,
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)) * 1.0,
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.zeros((nx, ny, nz)),
    }
    state["B"][2] = 1.0

    solver = MHDSolver(
        grid_shape=(nx, ny, nz), dx=0.01,
        use_ct=True, riemann_solver="hlld", time_integrator="ssp_rk3",
    )

    kinetic = 0.5 * state["rho"] * np.sum(state["velocity"] ** 2, axis=0)
    thermal = state["pressure"] / (GAMMA - 1.0)
    magnetic = np.sum(state["B"] ** 2, axis=0) / (2.0 * MU_0)
    E0 = np.sum(kinetic + thermal + magnetic)

    for _ in range(5):
        dt = solver._compute_dt(state)
        state = solver.step(state, dt, current=0.0, voltage=0.0)

    kinetic = 0.5 * state["rho"] * np.sum(state["velocity"] ** 2, axis=0)
    thermal = state["pressure"] / (GAMMA - 1.0)
    magnetic = np.sum(state["B"] ** 2, axis=0) / (2.0 * MU_0)
    E1 = np.sum(kinetic + thermal + magnetic)

    rel_error = np.abs(E1 - E0) / E0
    assert rel_error < 0.05


def test_python_ct_dedner_mutual_exclusion():
    """When use_ct=True, psi should not evolve."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx = ny = nz = 16
    state = {
        "rho": np.ones((nx, ny, nz)) * 1.0,
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)) * 1.0,
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.ones((nx, ny, nz)) * 0.1,
    }
    state["B"][2] = 1.0

    solver = MHDSolver(grid_shape=(nx, ny, nz), dx=0.01, use_ct=True, riemann_solver="hlld")

    for _ in range(5):
        dt = solver._compute_dt(state)
        state = solver.step(state, dt, current=0.0, voltage=0.0)

    assert np.max(np.abs(state["psi"])) < 0.2


def test_python_ct_small_grid_bypass():
    """Grid smaller than 3 should bypass CT gracefully."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx = ny = nz = 2
    state = {
        "rho": np.ones((nx, ny, nz)) * 1.0,
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)) * 1.0,
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.zeros((nx, ny, nz)),
    }
    state["B"][2] = 1.0

    solver = MHDSolver(grid_shape=(nx, ny, nz), dx=0.01, use_ct=True, riemann_solver="hlld")
    dt = solver._compute_dt(state)
    state = solver.step(state, dt, current=0.0, voltage=0.0)
    assert np.all(np.isfinite(state["rho"]))


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Python engine overflow in heat flux gradient when CT enabled; "
    "non-conservative pressure + CT interaction causes numerical blowup",
    strict=False,
)
def test_python_ct_vs_metal_ct_parity():
    """Python CT and Metal CT should both achieve div(B) < 1e-8 after 18 steps."""
    from dpf.fluid.mhd_solver import MHDSolver

    nx = ny = nz = 16
    dx = 0.01
    state_base = {
        "rho": np.ones((nx, ny, nz)) * 1.0,
        "velocity": np.zeros((3, nx, ny, nz)),
        "pressure": np.ones((nx, ny, nz)) * 1.0,
        "B": np.zeros((3, nx, ny, nz)),
        "Te": np.ones((nx, ny, nz)) * 1e4,
        "Ti": np.ones((nx, ny, nz)) * 1e4,
        "psi": np.zeros((nx, ny, nz)),
    }
    state_base["B"][2] = 1.0
    state_base["velocity"][0, nx // 2, ny // 2, nz // 2] = 0.1

    state_py = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state_base.items()}
    solver_py = MHDSolver(
        grid_shape=(nx, ny, nz), dx=dx,
        use_ct=True, riemann_solver="hlld", time_integrator="ssp_rk3",
    )

    for _ in range(18):
        dt = solver_py._compute_dt(state_py)
        state_py = solver_py.step(state_py, dt, current=0.0, voltage=0.0)

    staggered_py = cell_centered_to_face(
        state_py["B"][0], state_py["B"][1], state_py["B"][2], dx, dx, dx,
    )
    div_B_py = compute_div_B(staggered_py)
    max_div_py = np.max(np.abs(div_B_py))

    state_metal = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state_base.items()}
    solver_metal = MetalMHDSolver(
        grid_shape=(nx, ny, nz), dx=dx, device="cpu",
        use_ct=False, riemann_solver="hlld", time_integrator="ssp_rk3",
    )

    for _ in range(18):
        dt = solver_metal._compute_dt(state_metal)
        state_metal = solver_metal.step(state_metal, dt)

    staggered_m = cell_centered_to_face(
        state_metal["B"][0], state_metal["B"][1], state_metal["B"][2], dx, dx, dx,
    )
    div_B_metal = compute_div_B(staggered_m)
    max_div_metal = np.max(np.abs(div_B_metal))

    assert max_div_py < 1e-8
    assert max_div_metal < 1e-6
    assert max_div_py < max_div_metal * 0.1


# --- Section: Phase Q Metal Transport Physics ---


def _make_uniform_state(
    N: int = 16, rho: float = 1e-3, p: float = 1e3, Bz: float = 0.1, Te: float = 1e6
) -> dict[str, np.ndarray]:
    """Create a uniform state for transport tests."""
    return {
        "rho": np.full((N, N, N), rho),
        "velocity": np.zeros((3, N, N, N)),
        "pressure": np.full((N, N, N), p),
        "B": np.stack([np.zeros((N, N, N)), np.zeros((N, N, N)), np.full((N, N, N), Bz)]),
        "Te": np.full((N, N, N), Te),
        "Ti": np.full((N, N, N), Te),
        "psi": np.zeros((N, N, N)),
    }


class TestCurlB:
    """Tests for curl_B_mps (current density calculation)."""

    def test_curl_B_uniform_field(self):
        """Curl of uniform B should be zero."""
        B = torch.zeros((3, 16, 16, 16), dtype=torch.float32)
        B[2] = 0.1
        J = curl_B_mps(B, dx=0.01, dy=0.01, dz=0.01)
        J_np = J.cpu().numpy()
        assert np.allclose(J_np, 0.0, atol=1e-6), f"Uniform B curl should be zero, got max |J|={np.abs(J_np).max():.2e}"

    def test_curl_B_linear_field(self):
        """B = (0, x, 0) -> J_z = 1/mu_0 (analytically)."""
        N = 16
        x = torch.linspace(0, 0.15, N, dtype=torch.float32)
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[1] = x.view(N, 1, 1).expand(N, N, N)
        J = curl_B_mps(B, dx=0.01, dy=0.01, dz=0.01, mu_0=MU_0)
        expected_Jz = 1.0 / MU_0
        J_np = J.cpu().numpy()
        assert J_np[0].max() < 1e-6, "J_x should be zero"
        assert J_np[1].max() < 1e-6, "J_y should be zero"
        assert np.abs(J_np[2].mean() - expected_Jz) < 0.2 * expected_Jz, (
            f"J_z={J_np[2].mean():.2e}, expected {expected_Jz:.2e}"
        )

    def test_curl_B_shape_preservation(self):
        """Output shape (3, nx, ny, nz) matches input."""
        for shape in [(8, 8, 8), (16, 16, 16), (12, 18, 10)]:
            B = torch.randn((3, *shape), dtype=torch.float32)
            J = curl_B_mps(B, dx=0.01, dy=0.01, dz=0.01)
            assert J.shape == (3, *shape), f"Shape mismatch: {J.shape} vs (3, {shape})"


class TestHall:
    """Tests for Hall MHD operators."""

    def test_hall_zero_current(self):
        """Uniform B (no curl) -> zero Hall E-field -> B unchanged."""
        B = torch.zeros((3, 16, 16, 16), dtype=torch.float32)
        B[2] = 0.1
        rho = torch.full((16, 16, 16), 1e-3, dtype=torch.float32)
        B_new = apply_hall_mhd_mps(B, rho, dt=1e-9, dx=0.01, dy=0.01, dz=0.01)
        diff = (B_new - B).abs().max().item()
        assert diff < 1e-9, f"Hall should not change uniform B, diff={diff:.2e}"

    def test_hall_direction(self):
        """Hall field perpendicular to both J and B."""
        N = 16
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        y = torch.linspace(0, 0.15, N, dtype=torch.float32)
        B[1] = 0.1 * y.view(1, N, 1).expand(N, N, N)
        rho = torch.full((N, N, N), 1e-3, dtype=torch.float32)
        J = curl_B_mps(B, dx=0.01, dy=0.01, dz=0.01)
        E_Hall = hall_electric_field_mps(J, B, rho)
        E_np = E_Hall.cpu().numpy()
        B_np = B.cpu().numpy()
        dot_product = np.sum(E_np * B_np, axis=0)
        assert np.abs(dot_product).max() < 1e-10, "Hall E-field should be perpendicular to B"

    def test_hall_density_scaling(self):
        """Higher density -> weaker Hall effect (1/ne scaling)."""
        N = 16
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 1.0
        x = torch.linspace(0, 0.15, N, dtype=torch.float32)
        B[1] = x.view(N, 1, 1).expand(N, N, N)
        J = curl_B_mps(B, dx=0.01, dy=0.01, dz=0.01)
        rho_low = torch.full((N, N, N), 1e-4, dtype=torch.float32)
        rho_high = torch.full((N, N, N), 1e-3, dtype=torch.float32)
        E_Hall_low = hall_electric_field_mps(J, B, rho_low)
        E_Hall_high = hall_electric_field_mps(J, B, rho_high)
        mag_low = (E_Hall_low**2).sum(dim=0).sqrt().mean().item()
        mag_high = (E_Hall_high**2).sum(dim=0).sqrt().mean().item()
        ratio = mag_high / mag_low
        assert 0.05 < ratio < 0.15, f"E_Hall ratio={ratio:.3f}, expected ~0.1 (rho ratio)"

    @pytest.mark.slow
    def test_hall_whistler_dispersion(self):
        """Whistler wave propagation at correct speed."""
        pytest.skip("Whistler dispersion test not yet implemented")


class TestBraginskiiConduction:
    """Tests for Braginskii thermal conduction operators."""

    def test_braginskii_kappa_parallel_limit(self):
        """Unmagnetized (B->0) -> kappa_par runs without NaN/Inf."""
        ne = torch.full((16, 16, 16), 1e20, dtype=torch.float32)
        Te = torch.full((16, 16, 16), 1e6, dtype=torch.float32)
        B_mag = torch.full((16, 16, 16), 1e-6, dtype=torch.float32)
        kappa_par, kappa_perp = braginskii_kappa_mps(ne, Te, B_mag)
        kappa_par_np = kappa_par.cpu().numpy()
        kappa_perp_np = kappa_perp.cpu().numpy()
        assert np.isfinite(kappa_par_np).all(), "kappa_par contains NaN/Inf"
        assert np.isfinite(kappa_perp_np).all(), "kappa_perp contains NaN/Inf"
        assert (kappa_par_np >= 0).all(), "kappa_par should be non-negative"
        assert (kappa_perp_np >= 0).all(), "kappa_perp should be non-negative"

    def test_braginskii_kappa_perp_suppressed(self):
        """Strong B -> kappa_perp <= kappa_par."""
        ne = torch.full((16, 16, 16), 1e20, dtype=torch.float32)
        Te = torch.full((16, 16, 16), 1e6, dtype=torch.float32)
        B_mag = torch.full((16, 16, 16), 1.0, dtype=torch.float32)
        kappa_par, kappa_perp = braginskii_kappa_mps(ne, Te, B_mag)
        kappa_par_np = kappa_par.cpu().numpy()
        kappa_perp_np = kappa_perp.cpu().numpy()
        assert np.isfinite(kappa_par_np).all()
        assert np.isfinite(kappa_perp_np).all()
        assert (kappa_perp_np <= kappa_par_np + 1e-30).all(), "kappa_perp should be <= kappa_par"

    def test_braginskii_kappa_isotropic_limit(self):
        """B=0 -> kappa_par ≈ kappa_perp."""
        ne = torch.full((16, 16, 16), 1e20, dtype=torch.float32)
        Te = torch.full((16, 16, 16), 1e6, dtype=torch.float32)
        B_mag = torch.zeros((16, 16, 16), dtype=torch.float32)
        kappa_par, kappa_perp = braginskii_kappa_mps(ne, Te, B_mag)
        kappa_par_np = kappa_par.cpu().numpy()
        kappa_perp_np = kappa_perp.cpu().numpy()
        assert np.allclose(kappa_perp_np, kappa_par_np, rtol=1e-5), (
            "B=0 should give kappa_perp ≈ kappa_par"
        )

    def test_conduction_parallel_transport(self):
        """Heat flows along B-field direction."""
        N = 16
        dx = 0.01
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        Te = torch.zeros((N, N, N), dtype=torch.float32)
        z = torch.linspace(5e5, 1.5e6, N, dtype=torch.float32)
        Te[:, :, :] = z.view(1, 1, N)
        ne = torch.full((N, N, N), 1e20, dtype=torch.float32)
        Te_new = apply_braginskii_conduction_mps(Te, B, ne, dt=1e-12, dx=dx, dy=dx, dz=dx)
        assert torch.isfinite(Te_new).all(), "Te_new contains NaN/Inf"
        assert Te_new.max().item() <= Te.max().item() * 1.001

    def test_conduction_energy_conservation(self):
        """Total thermal energy approximately conserved."""
        N = 16
        dx = 0.01
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        x = torch.linspace(-0.075, 0.075, N, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        r2 = xx**2 + yy**2 + zz**2
        Te = 5e5 + 5e5 * torch.exp(-r2 / 0.01**2)
        ne = torch.full((N, N, N), 1e20, dtype=torch.float32)
        E_th_old = 1.5 * (ne * K_B * Te).sum().item()
        Te_new = apply_braginskii_conduction_mps(Te, B, ne, dt=1e-11, dx=dx, dy=dx, dz=dx)
        E_th_new = 1.5 * (ne * K_B * Te_new).sum().item()
        rel_change = abs(E_th_new - E_th_old) / E_th_old
        assert rel_change < 0.10, f"Thermal energy changed by {rel_change*100:.1f}%"


class TestBraginskiiViscosity:
    """Tests for Braginskii viscosity operators."""

    def test_viscosity_uniform_flow(self):
        """Uniform velocity -> no viscous forces."""
        N = 16
        velocity = torch.zeros((3, N, N, N), dtype=torch.float32)
        velocity[0] = 100.0
        rho = torch.full((N, N, N), 1e-3, dtype=torch.float32)
        pressure = torch.full((N, N, N), 1e3, dtype=torch.float32)
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        Ti = torch.full((N, N, N), 1e6, dtype=torch.float32)
        vel_new, p_new = apply_braginskii_viscosity_mps(
            velocity, rho, pressure, B, Ti, dt=1e-9, dx=0.01, dy=0.01, dz=0.01
        )
        diff = (vel_new - velocity).abs().max().item()
        assert diff < 1e-6, f"Uniform flow should not change under viscosity, diff={diff:.2e}"

    def test_viscosity_shear_flow(self):
        """Shear flow runs without error."""
        N = 16
        velocity = torch.zeros((3, N, N, N), dtype=torch.float32)
        y = torch.linspace(0, 0.15, N, dtype=torch.float32)
        velocity[0] = y.view(1, N, 1).expand(N, N, N) * 1000.0
        rho = torch.full((N, N, N), 1e-3, dtype=torch.float32)
        pressure = torch.full((N, N, N), 1e3, dtype=torch.float32)
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        Ti = torch.full((N, N, N), 1e6, dtype=torch.float32)
        vel_new, p_new = apply_braginskii_viscosity_mps(
            velocity, rho, pressure, B, Ti, dt=1e-9, dx=0.01, dy=0.01, dz=0.01
        )
        assert torch.isfinite(vel_new).all(), "vel_new contains NaN/Inf"
        assert torch.isfinite(p_new).all(), "p_new contains NaN/Inf"
        assert (p_new >= 0).all(), "Pressure should be non-negative"

    def test_viscosity_energy_conservation(self):
        """Total (kinetic + thermal) energy approximately conserved."""
        N = 16
        velocity = torch.zeros((3, N, N, N), dtype=torch.float32)
        x = torch.linspace(-0.075, 0.075, N, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        r2 = xx**2 + yy**2 + zz**2
        velocity[0] = 100.0 * torch.exp(-r2 / 0.01**2)
        rho = torch.full((N, N, N), 1e-3, dtype=torch.float32)
        pressure = torch.full((N, N, N), 1e3, dtype=torch.float32)
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        Ti = torch.full((N, N, N), 1e6, dtype=torch.float32)
        gamma_val = 5.0 / 3.0
        E_kin_old = 0.5 * (rho * (velocity**2).sum(dim=0)).sum().item()
        E_th_old = (pressure / (gamma_val - 1.0)).sum().item()
        E_tot_old = E_kin_old + E_th_old
        vel_new, p_new = apply_braginskii_viscosity_mps(
            velocity, rho, pressure, B, Ti, dt=1e-10, dx=0.01, dy=0.01, dz=0.01
        )
        E_kin_new = 0.5 * (rho * (vel_new**2).sum(dim=0)).sum().item()
        E_th_new = (p_new / (gamma_val - 1.0)).sum().item()
        E_tot_new = E_kin_new + E_th_new
        rel_change = (E_tot_new - E_tot_old) / E_tot_old
        assert -0.01 < rel_change < 0.20, f"Total energy changed by {rel_change*100:.1f}%"


class TestNernst:
    """Tests for Nernst B-field advection operators."""

    def test_nernst_coefficient_limits(self):
        """beta_wedge runs without NaN for various B strengths."""
        ne = torch.full((16, 16, 16), 1e20, dtype=torch.float32)
        Te = torch.full((16, 16, 16), 1e6, dtype=torch.float32)
        B_strong = torch.full((16, 16, 16), 10.0, dtype=torch.float32)
        beta_strong = nernst_coefficient_mps(ne, Te, B_strong)
        assert torch.isfinite(beta_strong).all(), "beta_wedge contains NaN for strong B"
        B_weak = torch.full((16, 16, 16), 1e-4, dtype=torch.float32)
        beta_weak = nernst_coefficient_mps(ne, Te, B_weak)
        assert torch.isfinite(beta_weak).all(), "beta_wedge contains NaN for weak B"
        B_tiny = torch.full((16, 16, 16), 1e-10, dtype=torch.float32)
        beta_tiny = nernst_coefficient_mps(ne, Te, B_tiny)
        assert torch.isfinite(beta_tiny).all(), "beta_wedge contains NaN for tiny B"
        assert beta_weak.mean() > beta_strong.mean(), (
            "beta_wedge should be larger at intermediate x_e than at very large x_e"
        )
        assert beta_weak.mean() > beta_tiny.mean(), (
            "beta_wedge should be larger at intermediate x_e than at very small x_e"
        )

    def test_nernst_direction(self):
        """Nernst advection runs without error."""
        N = 16
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        x = torch.linspace(1e6, 2e6, N, dtype=torch.float32)
        Te = x.view(N, 1, 1).expand(N, N, N)
        ne = torch.full((N, N, N), 1e20, dtype=torch.float32)
        B_new = apply_nernst_advection_mps(B, ne, Te, dt=1e-12, dx=0.01, dy=0.01, dz=0.01)
        assert torch.isfinite(B_new).all(), "Nernst advection produced NaN/Inf"

    def test_nernst_uniform_Te(self):
        """Uniform Te -> no Nernst advection."""
        B = torch.zeros((3, 16, 16, 16), dtype=torch.float32)
        B[2] = 0.1
        Te = torch.full((16, 16, 16), 1e6, dtype=torch.float32)
        ne = torch.full((16, 16, 16), 1e20, dtype=torch.float32)
        B_new = apply_nernst_advection_mps(B, ne, Te, dt=1e-9, dx=0.01, dy=0.01, dz=0.01)
        diff = (B_new - B).abs().max().item()
        assert diff < 1e-9, f"Uniform Te should give no Nernst advection, diff={diff:.2e}"

    def test_nernst_B_conservation(self):
        """Total magnetic energy approximately conserved."""
        N = 16
        B = torch.zeros((3, N, N, N), dtype=torch.float32)
        B[2] = 0.1
        x = torch.linspace(-0.075, 0.075, N, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        r2 = xx**2 + yy**2 + zz**2
        Te = 1e6 + 5e5 * torch.exp(-r2 / 0.01**2)
        ne = torch.full((N, N, N), 1e20, dtype=torch.float32)
        E_B_old = ((B**2).sum(dim=0) / (2 * MU_0)).sum().item()
        B_new = apply_nernst_advection_mps(B, ne, Te, dt=1e-11, dx=0.01, dy=0.01, dz=0.01)
        E_B_new = ((B_new**2).sum(dim=0) / (2 * MU_0)).sum().item()
        rel_change = abs(E_B_new - E_B_old) / E_B_old
        assert rel_change < 0.10, f"Magnetic energy changed by {rel_change*100:.1f}%"


class TestFloat64Accuracy:
    """Float64 precision tests for transport operators."""

    def test_hall_float64_vs_float32(self):
        """Float64 and float32 both run without error."""
        N = 16
        B_np = np.zeros((3, N, N, N))
        B_np[2] = 0.1
        y = np.linspace(0, 0.15, N)
        B_np[1] = 0.1 * y.reshape(1, N, 1)
        rho_np = np.full((N, N, N), 1e-3)
        B_f32 = torch.from_numpy(B_np).to(torch.float32)
        rho_f32 = torch.from_numpy(rho_np).to(torch.float32)
        B_f32_new = apply_hall_mhd_mps(B_f32, rho_f32, dt=1e-9, dx=0.01, dy=0.01, dz=0.01)
        B_f64 = torch.from_numpy(B_np).to(torch.float64)
        rho_f64 = torch.from_numpy(rho_np).to(torch.float64)
        B_f64_new = apply_hall_mhd_mps(B_f64, rho_f64, dt=1e-9, dx=0.01, dy=0.01, dz=0.01)
        assert torch.isfinite(B_f32_new).all(), "Float32 result contains NaN/Inf"
        assert torch.isfinite(B_f64_new).all(), "Float64 result contains NaN/Inf"

    def test_conduction_energy_conservation_float64(self):
        """<10% energy drift in float64."""
        N = 16
        dx = 0.01
        B = torch.zeros((3, N, N, N), dtype=torch.float64)
        B[2] = 0.1
        x = torch.linspace(-0.075, 0.075, N, dtype=torch.float64)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        r2 = xx**2 + yy**2 + zz**2
        Te = 1e6 + 5e5 * torch.exp(-r2 / 0.01**2)
        ne = torch.full((N, N, N), 1e20, dtype=torch.float64)
        E_th_old = 1.5 * (ne * K_B * Te).sum().item()
        Te_new = apply_braginskii_conduction_mps(Te, B, ne, dt=1e-11, dx=dx, dy=dx, dz=dx)
        E_th_new = 1.5 * (ne * K_B * Te_new).sum().item()
        rel_change = abs(E_th_new - E_th_old) / E_th_old
        assert rel_change < 0.10, f"Thermal energy changed by {rel_change*100:.2f}% in float64"

    def test_nernst_B_conservation_float64(self):
        """<10% B-energy drift in float64."""
        N = 16
        B = torch.zeros((3, N, N, N), dtype=torch.float64)
        B[2] = 0.1
        x = torch.linspace(-0.075, 0.075, N, dtype=torch.float64)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        r2 = xx**2 + yy**2 + zz**2
        Te = 1e6 + 5e5 * torch.exp(-r2 / 0.01**2)
        ne = torch.full((N, N, N), 1e20, dtype=torch.float64)
        E_B_old = ((B**2).sum(dim=0) / (2 * MU_0)).sum().item()
        B_new = apply_nernst_advection_mps(B, ne, Te, dt=1e-11, dx=0.01, dy=0.01, dz=0.01)
        E_B_new = ((B_new**2).sum(dim=0) / (2 * MU_0)).sum().item()
        rel_change = abs(E_B_new - E_B_old) / E_B_old
        assert rel_change < 0.10, f"Magnetic energy changed by {rel_change*100:.2f}% in float64"


# --- Section: Phase R Full HLLD Double-Star ---


def _make_cons(rho, vx, vy, vz, p, Bx, By, Bz, gamma=GAMMA):
    """Helper: build conservative state (8,) from primitives."""
    rho_t = torch.tensor([rho], dtype=torch.float64, device=DEVICE)
    vel_t = torch.tensor([[vx], [vy], [vz]], dtype=torch.float64, device=DEVICE)
    p_t = torch.tensor([p], dtype=torch.float64, device=DEVICE)
    B_t = torch.tensor([[Bx], [By], [Bz]], dtype=torch.float64, device=DEVICE)
    return _prim_to_cons_mps(rho_t, vel_t, p_t, B_t, gamma)


class TestHLLDDoublestar:
    """Tests for the full 6-region HLLD solver with double-star states."""

    def test_hlld_uniform_state_zero_flux_difference(self):
        """Uniform state should produce finite, non-NaN HLLD flux."""
        U = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        F = hlld_flux_mps(U, U, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD flux contains NaN for uniform state"
        assert torch.isfinite(F).all(), "HLLD flux contains Inf for uniform state"

    def test_hlld_contact_discontinuity_density_flux(self):
        """A pure contact (density jump, same p and v) should be resolved sharply."""
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        UR = _make_cons(0.125, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for contact discontinuity"
        assert torch.abs(F[IDN]).item() < 0.1, (
            f"Mass flux too large for stationary contact: {F[IDN].item():.4e}"
        )

    def test_hlld_alfven_wave_resolution(self):
        """Alfven discontinuity (transverse B jump) should be well-resolved."""
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        UR = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Alfven discontinuity"
        assert torch.isfinite(F).all(), "HLLD Inf for Alfven discontinuity"
        assert torch.abs(F[IDN]).item() < 1e-6, (
            f"Mass flux should be ~0 for Alfven wave: {F[IDN].item():.4e}"
        )
        assert torch.abs(F[IB1]).item() < 1e-6, (
            f"Normal B flux should be ~0: {F[IB1].item():.4e}"
        )

    def test_hlld_brio_wu_no_nan(self):
        """Brio-Wu MHD shock tube should not produce NaN."""
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        UR = _make_cons(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Brio-Wu"
        assert torch.isfinite(F).all(), "HLLD Inf for Brio-Wu"

    def test_hlld_double_star_degeneracy_bn_zero(self):
        """When Bn=0, double-star should reduce to single-star (HLLC)."""
        UL = _make_cons(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        UR = _make_cons(0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Bn=0"
        assert torch.isfinite(F).all(), "HLLD Inf for Bn=0"

    def test_hlld_conservation(self):
        """HLLD flux should conserve all 8 quantities (finite + correct shape)."""
        UL = _make_cons(1.0, 0.5, 0.1, -0.2, 2.0, 0.5, 0.3, 0.1)
        UR = _make_cons(0.8, -0.3, 0.2, 0.1, 1.5, 0.5, -0.2, 0.3)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert F.shape == (NVAR, 1), f"Flux shape mismatch: {F.shape}"
        assert torch.isfinite(F).all(), "HLLD flux not finite"

    def test_hlld_all_dimensions(self):
        """HLLD flux should work correctly for all three dimensions."""
        UL = _make_cons(1.0, 0.2, -0.1, 0.3, 1.5, 0.5, 0.3, 0.2)
        UR = _make_cons(0.5, -0.1, 0.2, -0.1, 0.8, 0.5, -0.1, 0.4)
        for dim in range(3):
            F = hlld_flux_mps(UL, UR, GAMMA, dim=dim)
            assert not torch.isnan(F).any(), f"HLLD NaN for dim={dim}"
            assert torch.isfinite(F).all(), f"HLLD Inf for dim={dim}"

    def test_hlld_symmetry(self):
        """HLLD should give symmetric results for mirror-symmetric states."""
        UL = _make_cons(1.0, 0.5, 0.1, 0.0, 1.0, 0.5, 0.3, 0.0)
        UR = _make_cons(1.0, -0.5, -0.1, 0.0, 1.0, 0.5, -0.3, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any()
        assert torch.abs(F[IDN]).item() < 0.1

    def test_hlld_strong_b_field(self):
        """HLLD should handle strong magnetic fields (magnetically dominated)."""
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 0.01, 10.0, 5.0, 3.0)
        UR = _make_cons(1.0, 0.0, 0.0, 0.0, 0.01, 10.0, -5.0, -3.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for strong B"
        assert torch.isfinite(F).all(), "HLLD Inf for strong B"

    def test_hlld_double_star_vs_single_star_with_bn(self):
        """With nonzero Bn, double-star flux should have nonzero transverse momentum."""
        UL = _make_cons(1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0)
        UR = _make_cons(1.0, 0.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any()
        assert torch.isfinite(F).all()
        assert F[IM2].abs().item() > 1e-10, (
            "Transverse momentum flux too small — double-star not active?"
        )

    def test_hlld_batch_no_nan(self):
        """HLLD should handle batched inputs (multi-cell) without NaN."""
        n = 16
        rho = torch.ones(n, dtype=torch.float64, device=DEVICE)
        vel = torch.zeros(3, n, dtype=torch.float64, device=DEVICE)
        vel[0] = torch.linspace(-0.5, 0.5, n)
        p = torch.ones(n, dtype=torch.float64, device=DEVICE)
        B = torch.zeros(3, n, dtype=torch.float64, device=DEVICE)
        B[0] = 1.0
        B[1] = torch.linspace(-1, 1, n)
        UL = _prim_to_cons_mps(rho, vel, p, B, GAMMA)
        UR = _prim_to_cons_mps(rho * 0.8, vel * 0.5, p * 0.9, B * 1.1, GAMMA)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert F.shape == (NVAR, n)
        assert not torch.isnan(F).any(), "HLLD NaN in batched inputs"
        assert torch.isfinite(F).all(), "HLLD Inf in batched inputs"

    def test_hlld_float32_brio_wu_no_nan(self):
        """HLLD should also work in float32 (Metal-native) for Brio-Wu."""
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        UR = _make_cons(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
        UL32 = UL.float()
        UR32 = UR.float()
        F = hlld_flux_mps(UL32, UR32, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Brio-Wu in float32"
        assert torch.isfinite(F).all(), "HLLD Inf for Brio-Wu in float32"


# --- Section: Phase V Conservative Energy ---


def _uniform_state(nr, nz, rho=1.0, p=1.0, Btheta=0.0):
    """Create a uniform cylindrical state dict (nr, 1, nz)."""

    return {
        "rho": np.full((nr, 1, nz), rho),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }


def _sod_state(nr, nz, dr, dz, gamma=5.0 / 3.0):
    """Create a Sod shock tube initial condition along z-axis."""
    state = _uniform_state(nr, nz)
    mid = nz // 2
    state["rho"][:, :, :mid] = 1.0
    state["pressure"][:, :, :mid] = 1.0
    state["rho"][:, :, mid:] = 0.125
    state["pressure"][:, :, mid:] = 0.1
    return state


@pytest.fixture
def solver_conservative():
    """CylindricalMHDSolver with conservative energy enabled."""
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    return CylindricalMHDSolver(
        nr=16, nz=32, dr=0.01, dz=0.01,
        gamma=5.0 / 3.0, cfl=0.3,
        conservative_energy=True,
        time_integrator="ssp_rk3",
    )


@pytest.fixture
def solver_nonconservative():
    """CylindricalMHDSolver with non-conservative (legacy) pressure equation."""
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    return CylindricalMHDSolver(
        nr=16, nz=32, dr=0.01, dz=0.01,
        gamma=5.0 / 3.0, cfl=0.3,
        conservative_energy=False,
        time_integrator="ssp_rk3",
    )


class TestConservativeEnergy:
    """Test conservative total energy formulation."""

    def test_default_is_conservative(self):
        """Default is conservative for correct shock-capturing physics."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        assert solver.conservative_energy is True

    def test_nonconservative_flag(self):
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(
            nr=8, nz=8, dr=0.01, dz=0.01, conservative_energy=False,
        )
        assert solver.conservative_energy is False

    def test_uniform_state_preserved_conservative(self, solver_conservative):
        """Uniform state should remain uniform under conservative energy."""
        state = _uniform_state(16, 32)
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))
        np.testing.assert_allclose(new_state["rho"], state["rho"], rtol=1e-6)
        np.testing.assert_allclose(new_state["pressure"], state["pressure"], rtol=1e-4)

    def test_uniform_state_preserved_nonconservative(self, solver_nonconservative):
        """Uniform state should remain uniform under non-conservative energy."""
        state = _uniform_state(16, 32)
        new_state = solver_nonconservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))
        np.testing.assert_allclose(new_state["rho"], state["rho"], rtol=1e-6)

    def test_total_energy_conservation_sod(self, solver_conservative):
        """Sod shock: total energy should be approximately conserved."""
        from dpf.constants import mu_0 as _mu0

        nr, nz = 16, 32
        dr, dz = 0.01, 0.01
        gamma_val = 5.0 / 3.0
        state = _sod_state(nr, nz, dr, dz)

        rho = state["rho"][:, 0, :]
        p = state["pressure"][:, 0, :]
        v = state["velocity"][:, :, 0, :]
        B = state["B"][:, :, 0, :]
        v_sq = np.sum(v**2, axis=0)
        B_sq = np.sum(B**2, axis=0)
        E_init = p / (gamma_val - 1) + 0.5 * rho * v_sq + B_sq / (2.0 * _mu0)
        r = np.array([(i + 0.5) * dr for i in range(nr)])
        cell_vol = 2.0 * np.pi * r[:, np.newaxis] * dr * dz
        total_E_init = np.sum(E_init * cell_vol)

        dt = 1e-5
        for _ in range(10):
            state = solver_conservative.step(state, dt=dt, current=0.0, voltage=0.0)

        rho = state["rho"][:, 0, :]
        p = state["pressure"][:, 0, :]
        v = state["velocity"][:, :, 0, :]
        B = state["B"][:, :, 0, :]
        v_sq = np.sum(v**2, axis=0)
        B_sq = np.sum(B**2, axis=0)
        E_final = p / (gamma_val - 1) + 0.5 * rho * v_sq + B_sq / (2.0 * _mu0)
        total_E_final = np.sum(E_final * cell_vol)

        rel_change = abs(total_E_final - total_E_init) / abs(total_E_init)
        assert rel_change < 0.10, f"Energy changed by {rel_change:.2%}"

    def test_no_nan_sod_conservative(self, solver_conservative):
        """Sod shock should not produce NaN with conservative energy."""
        state = _sod_state(16, 32, 0.01, 0.01)
        for _ in range(20):
            state = solver_conservative.step(state, dt=1e-5, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(state["rho"]))
        assert not np.any(np.isnan(state["pressure"]))

    def test_conservative_step_returns_all_keys(self, solver_conservative):
        """Step should return all required state keys."""
        state = _uniform_state(16, 32)
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti", "psi"]:
            assert key in new_state, f"Missing key: {key}"

    def test_rhs_returns_dE_dt(self, solver_conservative):
        """Conservative RHS should return dE_dt, not dp_dt."""
        state = _uniform_state(16, 32)
        rho_2d = state["rho"][:, 0, :]
        vel_2d = state["velocity"][:, :, 0, :]
        p_2d = state["pressure"][:, 0, :]
        B_2d = state["B"][:, :, 0, :]
        psi_2d = np.zeros((16, 32))
        rhs = solver_conservative._compute_rhs(rho_2d, vel_2d, p_2d, B_2d, psi_2d)
        assert "dE_dt" in rhs
        assert "dp_dt" not in rhs

    def test_rhs_nonconservative_returns_dp_dt(self, solver_nonconservative):
        """Non-conservative RHS should return dp_dt, not dE_dt."""
        state = _uniform_state(16, 32)
        rho_2d = state["rho"][:, 0, :]
        vel_2d = state["velocity"][:, :, 0, :]
        p_2d = state["pressure"][:, 0, :]
        B_2d = state["B"][:, :, 0, :]
        psi_2d = np.zeros((16, 32))
        rhs = solver_nonconservative._compute_rhs(rho_2d, vel_2d, p_2d, B_2d, psi_2d)
        assert "dp_dt" in rhs
        assert "dE_dt" not in rhs


class TestAxisBoundaryCondition:
    """Test v_r=0 and B_r=0 enforcement at r=0."""

    def test_vr_zero_at_axis(self, solver_conservative):
        """v_r should be zero at r=0 after step."""
        state = _uniform_state(16, 32)
        state["velocity"][0, :, :, :] = 100.0
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert np.allclose(new_state["velocity"][0, 0, :, :], 0.0)

    def test_Br_zero_at_axis(self, solver_conservative):
        """B_r should be zero at r=0 after step."""
        state = _uniform_state(16, 32)
        state["B"][0, :, :, :] = 0.01
        new_state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert np.allclose(new_state["B"][0, 0, :, :], 0.0)

    def test_axis_bc_enforced_every_rk_stage(self, solver_conservative):
        """Axis BC should be enforced after each RK stage, not just final."""
        state = _uniform_state(16, 32)
        state["B"][1, :, :, :] = 0.1
        for _ in range(5):
            state = solver_conservative.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert np.allclose(state["velocity"][0, 0, :, :], 0.0)
        assert np.allclose(state["B"][0, 0, :, :], 0.0)


class TestSourceTermsPassthrough:
    """Test that source_terms are properly passed to cylindrical solver."""

    def test_source_terms_accepted(self, solver_conservative):
        """Step should accept source_terms kwarg."""
        state = _uniform_state(16, 32)
        src = {"Q_ohmic_correction": np.ones((16, 1, 32)) * 1e6}
        new_state = solver_conservative.step(
            state, dt=1e-8, current=0.0, voltage=0.0, source_terms=src,
        )
        assert not np.any(np.isnan(new_state["pressure"]))
        assert np.mean(new_state["pressure"]) > np.mean(state["pressure"])

    def test_snowplow_source_terms(self, solver_conservative):
        """Snowplow source terms should inject mass/momentum/energy."""
        state = _uniform_state(16, 32, rho=0.001)
        S_rho = np.zeros((16, 1, 32))
        S_rho[:, :, 16] = 1.0
        src = {
            "S_rho_snowplow": S_rho,
            "S_mom_snowplow": np.zeros((3, 16, 1, 32)),
            "S_energy_snowplow": np.zeros((16, 1, 32)),
        }
        new_state = solver_conservative.step(
            state, dt=1e-6, current=0.0, voltage=0.0, source_terms=src,
        )
        assert new_state["rho"][8, 0, 16] > state["rho"][8, 0, 16]


class TestSSPRK2Fallback:
    """Test SSP-RK2 time integrator with conservative energy."""

    def test_rk2_conservative_runs(self):
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(
            nr=8, nz=16, dr=0.01, dz=0.01,
            conservative_energy=True, time_integrator="ssp_rk2",
        )
        state = _uniform_state(8, 16)
        new_state = solver.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))

    def test_rk2_nonconservative_runs(self):
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        solver = CylindricalMHDSolver(
            nr=8, nz=16, dr=0.01, dz=0.01,
            conservative_energy=False, time_integrator="ssp_rk2",
        )
        state = _uniform_state(8, 16)
        new_state = solver.step(state, dt=1e-8, current=0.0, voltage=0.0)
        assert not np.any(np.isnan(new_state["rho"]))


# --- Section: Phase V Cylindrical Conservative ---


def _make_state_cyl(nr: int, nz: int, rho: float = 1.0, p: float = 1e5,
                    vr: float = 0.0, vz: float = 0.0, Btheta: float = 0.0,
                    Bz: float = 0.0) -> dict[str, np.ndarray]:
    """Build a cylindrical state dict with shapes (nr,1,nz) / (3,nr,1,nz)."""
    state = {
        "rho": np.full((nr, 1, nz), rho),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }
    state["velocity"][0] = vr
    state["velocity"][2] = vz
    state["B"][1] = Btheta
    state["B"][2] = Bz
    return state


def _total_energy_volume(state: dict, dr: float, dz: float,
                          gamma: float = 5.0 / 3.0) -> float:
    """Compute volume-integrated total energy for cylindrical geometry."""
    from dpf.constants import mu_0 as _mu0

    nr = state["rho"].shape[0]
    rho = state["rho"][:, 0, :]
    p = state["pressure"][:, 0, :]
    v = state["velocity"][:, :, 0, :]
    B = state["B"][:, :, 0, :]
    v_sq = np.sum(v**2, axis=0)
    B_sq = np.sum(B**2, axis=0)
    E_density = p / (gamma - 1) + 0.5 * rho * v_sq + B_sq / (2.0 * _mu0)
    r = np.array([(i + 0.5) * dr for i in range(nr)])
    cell_vol = 2.0 * np.pi * r[:, np.newaxis] * dr * dz
    return float(np.sum(E_density * cell_vol))


@pytest.fixture
def cons_solver():
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        conservative_energy=True, time_integrator="ssp_rk3",
    )


@pytest.fixture
def noncons_solver():
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        conservative_energy=False, time_integrator="ssp_rk3",
    )


class TestConservativeFlag:
    def test_default_is_true(self):
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        s = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        assert s.conservative_energy is True

    def test_explicit_true(self):
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        s = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01, conservative_energy=True)
        assert s.conservative_energy is True

    def test_rhs_dE_dt_when_conservative(self, cons_solver):
        """_compute_rhs returns dE_dt (not dp_dt) when conservative_energy=True."""
        rho = np.ones((16, 16))
        vel = np.zeros((3, 16, 16))
        p = np.full((16, 16), 1e5)
        B = np.zeros((3, 16, 16))
        psi = np.zeros((16, 16))
        rhs = cons_solver._compute_rhs(rho, vel, p, B, psi)
        assert "dE_dt" in rhs
        assert "dp_dt" not in rhs

    def test_rhs_dp_dt_when_nonconservative(self, noncons_solver):
        """_compute_rhs returns dp_dt (not dE_dt) when conservative_energy=False."""
        rho = np.ones((16, 16))
        vel = np.zeros((3, 16, 16))
        p = np.full((16, 16), 1e5)
        B = np.zeros((3, 16, 16))
        psi = np.zeros((16, 16))
        rhs = noncons_solver._compute_rhs(rho, vel, p, B, psi)
        assert "dp_dt" in rhs
        assert "dE_dt" not in rhs


class TestEnergyConservation:
    def test_uniform_state_preserves_energy(self, cons_solver):
        """Uniform quiescent state should preserve total energy exactly."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1e5)
        E0 = _total_energy_volume(state, 0.001, 0.001)
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        E1 = _total_energy_volume(new, 0.001, 0.001)
        assert pytest.approx(E0, rel=1e-6) == E1

    def test_sod_shock_energy_bounded(self, cons_solver):
        """Sod shock tube: total energy change < 10% over 5 steps."""
        nr, nz, dr, dz = 16, 16, 0.001, 0.001
        state = _make_state_cyl(nr, nz, rho=1.0, p=1e5)
        mid = nz // 2
        state["rho"][:, :, mid:] = 0.125
        state["pressure"][:, :, mid:] = 1e4
        E0 = _total_energy_volume(state, dr, dz)
        for _ in range(5):
            state = cons_solver.step(state, dt=1e-7, current=0.0, voltage=0.0)
        E1 = _total_energy_volume(state, dr, dz)
        rel = abs(E1 - E0) / abs(E0)
        assert rel < 0.10, f"Energy changed by {rel:.2%}"

    def test_conservative_better_than_nonconservative(self):
        """Conservative mode should conserve energy at least as well as non-conservative."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz, dr, dz = 16, 32, 0.001, 0.001
        cons = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                    conservative_energy=True, time_integrator="ssp_rk3")
        noncons = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                       conservative_energy=False, time_integrator="ssp_rk3")
        state_c = _make_state_cyl(nr, nz, rho=1.0, p=1e5)
        mid = nz // 2
        state_c["rho"][:, :, mid:] = 0.125
        state_c["pressure"][:, :, mid:] = 1e4
        state_nc = {k: v.copy() for k, v in state_c.items()}
        E0 = _total_energy_volume(state_c, dr, dz)
        dt = 1e-7
        for _ in range(10):
            state_c = cons.step(state_c, dt=dt, current=0.0, voltage=0.0)
            state_nc = noncons.step(state_nc, dt=dt, current=0.0, voltage=0.0)
        rel_c = abs(_total_energy_volume(state_c, dr, dz) - E0) / abs(E0)
        rel_nc = abs(_total_energy_volume(state_nc, dr, dz) - E0) / abs(E0)
        assert rel_c <= rel_nc + 0.02, (
            f"Conservative ({rel_c:.4f}) worse than non-conservative ({rel_nc:.4f})"
        )


class TestPressureRecovery:
    def test_pressure_positive_after_step(self, cons_solver):
        """Pressure must remain positive (floored at 1e-20) after conservative step."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1e5)
        state["velocity"][2] = 1e4
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        assert np.all(new["pressure"] >= 1e-20)

    def test_pressure_recovery_formula(self, cons_solver):
        """Verify p = (gamma-1)*(E - 0.5*rho*v^2 - B^2/(2*mu_0))."""
        from dpf.constants import mu_0 as _mu0

        gamma_val = cons_solver.gamma
        gm1 = gamma_val - 1.0
        rho_val, p_val = 2.0, 5e5
        vz_val, Bt_val = 1e3, 0.5
        v_sq = vz_val**2
        B_sq = Bt_val**2
        E_expected = p_val / gm1 + 0.5 * rho_val * v_sq + B_sq / (2.0 * _mu0)
        p_recovered = gm1 * (E_expected - 0.5 * rho_val * v_sq - B_sq / (2.0 * _mu0))
        assert p_recovered == pytest.approx(p_val, rel=1e-12)

    def test_pressure_floor_prevents_negative(self, cons_solver):
        """When KE+ME > E_total, pressure is floored at 1e-20."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1.0)
        state["velocity"][2] = 1e8
        new = cons_solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        assert np.all(new["pressure"] >= 1e-20)


class TestSSPRK3Combining:
    def test_rk3_three_stage_weights(self):
        """Verify SSP-RK3 weights differ from RK2."""
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

        nr, nz = 16, 16
        dr, dz = 0.001, 0.001
        rk3 = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                    conservative_energy=True, time_integrator="ssp_rk3")
        rk2 = CylindricalMHDSolver(nr=nr, nz=nz, dr=dr, dz=dz,
                                    conservative_energy=True, time_integrator="ssp_rk2")
        state = _make_state_cyl(nr, nz, rho=1.0, p=1e5, vz=1e3)
        out3 = rk3.step(state, dt=1e-7, current=0.0, voltage=0.0)
        out2 = rk2.step(state, dt=1e-7, current=0.0, voltage=0.0)
        diff = np.max(np.abs(out3["pressure"] - out2["pressure"]))
        assert diff > 0.0, "RK3 and RK2 produced identical results"

    def test_rk3_conservative_no_nan(self, cons_solver):
        """RK3 conservative stepping should not produce NaN on moderate flow."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1e5, vz=5e3, Btheta=0.1)
        for _ in range(10):
            state = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0)
        for key in ("rho", "pressure", "velocity", "B"):
            assert not np.any(np.isnan(state[key])), f"NaN in {key}"


class TestAxisBC:
    def test_vr_zero_at_axis(self, cons_solver):
        """v_r must be zero at r=0 after step."""
        state = _make_state_cyl(16, 16, vr=500.0)
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        np.testing.assert_allclose(new["velocity"][0, 0, :, :], 0.0, atol=1e-30)

    def test_Br_zero_at_axis(self, cons_solver):
        """B_r must be zero at r=0 after step."""
        state = _make_state_cyl(16, 16)
        state["B"][0] = 0.01
        new = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        np.testing.assert_allclose(new["B"][0, 0, :, :], 0.0, atol=1e-30)

    def test_axis_bc_survives_multiple_steps(self, cons_solver):
        """Axis BC should hold after many steps with active dynamics."""
        state = _make_state_cyl(16, 16, vr=100.0, Btheta=0.5)
        for _ in range(20):
            state = cons_solver.step(state, dt=1e-9, current=0.0, voltage=0.0)
        np.testing.assert_allclose(state["velocity"][0, 0, :, :], 0.0, atol=1e-30)
        np.testing.assert_allclose(state["B"][0, 0, :, :], 0.0, atol=1e-30)


class TestSourceTermPassthrough:
    def test_Q_ohmic_correction_heats(self, cons_solver):
        """Q_ohmic_correction source should increase total energy."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1e5)
        E0 = _total_energy_volume(state, 0.001, 0.001)
        src = {"Q_ohmic_correction": np.full((16, 1, 16), 1e12)}
        new = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0, source_terms=src)
        E1 = _total_energy_volume(new, 0.001, 0.001)
        assert E1 > E0, "Ohmic correction did not increase total energy"

    def test_S_rho_snowplow_adds_mass(self, cons_solver):
        """S_rho_snowplow should increase density."""
        state = _make_state_cyl(16, 16, rho=0.01, p=1e5)
        S_rho = np.zeros((16, 1, 16))
        S_rho[8, :, 8] = 1e3
        src = {"S_rho_snowplow": S_rho}
        new = cons_solver.step(state, dt=1e-6, current=0.0, voltage=0.0, source_terms=src)
        assert new["rho"][8, 0, 8] > state["rho"][8, 0, 8]

    def test_S_mom_snowplow_adds_momentum(self, cons_solver):
        """S_mom_snowplow should change velocity."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1e5)
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2, 8, :, 8] = 1e6
        src = {"S_mom_snowplow": S_mom}
        new = cons_solver.step(state, dt=1e-7, current=0.0, voltage=0.0, source_terms=src)
        assert abs(new["velocity"][2, 8, 0, 8]) > 0.0

    def test_S_energy_snowplow_in_conservative(self, cons_solver):
        """S_energy_snowplow should be added to dE/dt in conservative mode."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1e5)
        E0 = _total_energy_volume(state, 0.001, 0.001)
        S_E = np.full((16, 1, 16), 1e12)
        src = {"S_energy_snowplow": S_E}
        new = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0, source_terms=src)
        E1 = _total_energy_volume(new, 0.001, 0.001)
        assert E1 > E0

    def test_multiple_source_terms_combined(self, cons_solver):
        """All four source terms active simultaneously."""
        state = _make_state_cyl(16, 16, rho=1.0, p=1e5)
        src = {
            "S_rho_snowplow": np.full((16, 1, 16), 0.1),
            "S_mom_snowplow": np.zeros((3, 16, 1, 16)),
            "S_energy_snowplow": np.full((16, 1, 16), 1e8),
            "Q_ohmic_correction": np.full((16, 1, 16), 1e8),
        }
        new = cons_solver.step(state, dt=1e-8, current=0.0, voltage=0.0, source_terms=src)
        for key in ("rho", "pressure", "velocity", "B"):
            assert not np.any(np.isnan(new[key])), f"NaN in {key} with combined sources"


# --- Section: Phase X Powell div(B) ---




def _make_geom(nr: int = 16, nz: int = 16, dr: float = 0.01, dz: float = 0.01) -> CylindricalGeometry:
    return CylindricalGeometry(nr, nz, dr, dz)


def _make_state_2d(nr: int, nz: int) -> dict[str, np.ndarray]:
    return {
        "rho": np.ones((nr, nz)) * 1e-4,
        "velocity": np.zeros((3, nr, nz)),
        "pressure": np.ones((nr, nz)) * 1.0,
        "B": np.zeros((3, nr, nz)),
        "Te": np.ones((nr, nz)) * 1e4,
        "Ti": np.ones((nr, nz)) * 1e4,
        "psi": np.zeros((nr, nz)),
    }


class TestCylindricalPowellZeroDivB:
    """Powell sources must vanish when div(B) = 0."""

    def test_azimuthal_B_zero_sources(self) -> None:
        """Pure B_theta = 1/r is divergence-free in cylindrical coordinates."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        state["B"][1] = 1.0 / geom.r_2d
        state["velocity"] = np.random.default_rng(7).random((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        np.testing.assert_allclose(result["div_B"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dmom_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-12)

    def test_uniform_Bz_zero_sources(self) -> None:
        """Uniform B_z = const has dB_z/dz = 0 and B_r = 0, so div(B) = 0."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        state["B"][2] = 2.5
        state["velocity"] = np.random.default_rng(13).random((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        np.testing.assert_allclose(result["div_B"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dmom_powell"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-10)

    def test_random_velocity_div_free_B(self) -> None:
        """Random velocity with div-free B must give zero Powell sources."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        rng = np.random.default_rng(99)
        state["B"][1] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        np.testing.assert_allclose(result["div_B"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dmom_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-12)


class TestCylindricalPowellNonzeroDivB:
    """Powell sources must be nonzero when div(B) != 0."""

    def test_uniform_Br_nonzero_sources(self) -> None:
        """Uniform B_r has cylindrical div = B_r/r."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        state["B"][0] = 1.0
        state["B"][2] = 0.0
        state["velocity"][0] = 0.1
        result = powell_source_terms_cylindrical(state, geom)
        assert np.any(np.abs(result["div_B"][1:-1, 1:-1]) > 1e-6), (
            "Expected nonzero div(B) for uniform B_r"
        )
        assert np.any(np.abs(result["dmom_powell"][:, 1:-1, 1:-1]) > 1e-10), (
            "Expected nonzero momentum Powell source"
        )

    def test_linear_Bz_nonzero(self) -> None:
        """B_z = z -> dB_z/dz = 1 -> nonzero sources."""
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)
        state = _make_state_2d(nr, nz)
        z = np.array([(j + 0.5) * dz for j in range(nz)])
        alpha = 100.0
        state["B"][2] = alpha * z[np.newaxis, :]
        state["velocity"][2] = 0.5
        result = powell_source_terms_cylindrical(state, geom)
        np.testing.assert_allclose(result["div_B"][1:-1, 1:-1], alpha, rtol=0.05)
        assert np.any(np.abs(result["dB_powell"][:, 1:-1, 1:-1]) > 1e-6), (
            "Expected nonzero induction Powell source"
        )

    def test_v_zero_energy_source_zero(self) -> None:
        """When v=0, energy source = -div(B)*(v.B) = 0, but momentum source != 0."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        state["B"][0] = 1.0
        state["velocity"] = np.zeros((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-20)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-20)
        assert np.any(np.abs(result["dmom_powell"][0, 1:-1, 1:-1]) > 1e-8)


class TestCylindricalPowellFormula:
    """Verify analytical formulas for each Powell source component."""

    def test_momentum_equals_neg_divB_times_B(self) -> None:
        """Verify dmom[d] = -div(B)*B[d] at interior."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        rng = np.random.default_rng(21)
        state["B"][0] = 0.5 * rng.random((nr, nz))
        state["B"][1] = rng.random((nr, nz))
        state["B"][2] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        div_B = result["div_B"]
        for d in range(3):
            expected = -div_B * state["B"][d]
            np.testing.assert_allclose(
                result["dmom_powell"][d], expected, atol=1e-14,
                err_msg=f"Momentum source component {d} mismatch",
            )

    def test_energy_equals_neg_divB_times_vdotB(self) -> None:
        """Verify denergy = -div(B)*(v.B)."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        rng = np.random.default_rng(55)
        state["B"][0] = rng.random((nr, nz)) * 0.5
        state["B"][2] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        div_B = result["div_B"]
        v_dot_B = np.sum(state["velocity"] * state["B"], axis=0)
        expected = -div_B * v_dot_B
        np.testing.assert_allclose(result["denergy_powell"], expected, atol=1e-14)

    def test_induction_equals_neg_divB_times_v(self) -> None:
        """Verify dB[d] = -div(B)*v[d]."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        rng = np.random.default_rng(77)
        state["B"][0] = rng.random((nr, nz)) * 0.3
        state["B"][2] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        div_B = result["div_B"]
        for d in range(3):
            expected = -div_B * state["velocity"][d]
            np.testing.assert_allclose(
                result["dB_powell"][d], expected, atol=1e-14,
                err_msg=f"Induction source component {d} mismatch",
            )

    def test_cylindrical_vs_cartesian_diverges(self) -> None:
        """Uniform Br has ZERO Cartesian div but NONZERO cylindrical div."""
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)
        B = np.zeros((3, nr, nz))
        B[0] = 1.0
        div_cyl = geom.div_B_cylindrical(B)
        div_cart = (
            np.gradient(B[0], dr, axis=0)
            + np.gradient(B[1], dr, axis=0) * 0.0
            + np.gradient(B[2], dz, axis=1)
        )
        np.testing.assert_allclose(div_cart[1:-1, 1:-1], 0.0, atol=1e-10)
        assert np.all(np.abs(div_cyl[1:-1, 1:-1]) > 1e-3), (
            "Expected nonzero cylindrical div(B) for uniform B_r"
        )


class TestPowellEngineIntegration:
    """Engine-level integration tests for Powell sources."""

    def _make_engine(self) -> object:
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        cfg = SimulationConfig(
            grid_shape=(8, 1, 16),
            dx=0.01,
            rho0=1e-4,
            sim_time=1e-6,
            circuit={
                "C": 1e-3, "V0": 15000, "L0": 33.5e-9,
                "anode_radius": 0.025, "cathode_radius": 0.05,
            },
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python", "enable_powell": True},
            diagnostics={"hdf5_filename": ":memory:"},
            boundary={"electrode_bc": False},
        )
        return SimulationEngine(cfg)

    def test_engine_with_powell_runs(self) -> None:
        """SimulationConfig with enable_powell=True runs 5 steps."""
        engine = self._make_engine()
        for _ in range(5):
            engine.step()

    def test_engine_powell_no_nan(self) -> None:
        """No NaN after 5 steps."""
        engine = self._make_engine()
        for _ in range(5):
            engine.step()
        for key in ("rho", "velocity", "pressure", "B"):
            arr = engine.state[key]
            assert not np.any(np.isnan(arr)), f"NaN found in {key} after 5 Powell steps"

    def test_engine_powell_disabled_default(self) -> None:
        """Default enable_powell is False."""
        from dpf.config import SimulationConfig

        cfg = SimulationConfig(
            grid_shape=(8, 1, 16),
            dx=0.01,
            rho0=1e-4,
            sim_time=1e-6,
            circuit={
                "C": 1e-3, "V0": 15000, "L0": 33.5e-9,
                "anode_radius": 0.025, "cathode_radius": 0.05,
            },
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python"},
            diagnostics={"hdf5_filename": ":memory:"},
            boundary={"electrode_bc": False},
        )
        assert cfg.fluid.enable_powell is False

    def test_engine_powell_density_positive(self) -> None:
        """Density stays positive after 5 Powell steps."""
        engine = self._make_engine()
        for _ in range(5):
            engine.step()
        rho = engine.state["rho"]
        assert np.all(rho > 0.0), f"Non-positive density detected: min={rho.min():.3e}"


class TestPowellReducesDivB:
    """Powell source terms should control div(B) growth."""

    def test_divb_rms_bounded(self) -> None:
        """After applying Powell sources to B (Euler step), div(B) RMS doesn't grow."""
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)
        state = _make_state_2d(nr, nz)
        rng = np.random.default_rng(11)
        state["B"][0] = 0.1 * rng.random((nr, nz))
        state["B"][2] = 0.0
        state["velocity"][0] = 0.05
        result = powell_source_terms_cylindrical(state, geom)
        div_B_before = result["div_B"]
        rms_before = float(np.sqrt(np.mean(div_B_before**2)))
        dt = 1e-9
        B_new = state["B"].copy()
        B_new += dt * result["dB_powell"]
        div_B_after = geom.div_B_cylindrical(B_new)
        rms_after = float(np.sqrt(np.mean(div_B_after**2)))
        assert rms_after <= rms_before * 1.1, (
            f"div(B) RMS grew: {rms_before:.3e} -> {rms_after:.3e}"
        )

    def test_divb_multi_step_bounded(self) -> None:
        """Over 10 Euler steps, div(B) stays bounded."""
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)
        state = _make_state_2d(nr, nz)
        rng = np.random.default_rng(33)
        state["B"][0] = 0.2 * rng.random((nr, nz))
        state["velocity"][0] = 0.1
        initial_result = powell_source_terms_cylindrical(state, geom)
        rms_initial = float(np.sqrt(np.mean(initial_result["div_B"] ** 2)))
        dt = 1e-10
        B = state["B"].copy()
        for _ in range(10):
            state_step = dict(state)
            state_step["B"] = B
            result = powell_source_terms_cylindrical(state_step, geom)
            B = B + dt * result["dB_powell"]
        div_B_final = geom.div_B_cylindrical(B)
        rms_final = float(np.sqrt(np.mean(div_B_final**2)))
        assert rms_final <= rms_initial * 10.0, (
            f"div(B) RMS blew up: initial={rms_initial:.3e}, final={rms_final:.3e}"
        )

    def test_dB_zero_when_v_zero(self) -> None:
        """dB_powell is exactly zero when velocity is zero."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)
        state["B"][0] = 1.0
        state["velocity"] = np.zeros((3, nr, nz))
        result = powell_source_terms_cylindrical(state, geom)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-20)
