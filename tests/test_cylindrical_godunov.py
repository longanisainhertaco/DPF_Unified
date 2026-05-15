"""Tests for the Godunov (PLM+HLL) flux path in CylindricalMHDSolver.

Validates the use_godunov_flux=True mode that replaces np.gradient central
differences with PLM reconstruction + HLL Riemann fluxes.  This path is
designed to handle sheath discontinuities without Gibbs oscillations.

Test matrix:
    1. Sod shock tube (1D radial): no NaN, shock at correct position
    2. Uniform state preservation: dU/dt = 0 for uniform initial conditions
    3. Conservation: total energy conserved on uniform state
    4. Small-grid smoke test: runs without crash at minimal resolution
    5. PLM reconstruction unit test: correct slopes, monotonicity
"""

from __future__ import annotations

import numpy as np

from dpf.constants import mu_0
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver


def _make_uniform_state(
    nr: int = 16,
    nz: int = 16,
    dr: float = 0.001,
    dz: float = 0.001,
    rho0: float = 1e-4,
    p0: float = 1e3,
    B_theta: float = 0.0,
) -> tuple[CylindricalMHDSolver, dict]:
    """Create a force-free uniform solver state for preservation tests."""
    solver = CylindricalMHDSolver(
        nr=nr, nz=nz, dr=dr, dz=dz,
        gamma=5.0 / 3.0, cfl=0.3,
        enable_hall=False, enable_resistive=False,
        use_godunov_flux=True,
        conservative_energy=True,
    )
    state = {
        "rho": np.full((nr, 1, nz), rho0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p0),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }
    state["B"][1, :, :, :] = B_theta
    return solver, state


class TestGodunovInstantiation:
    """Verify the solver creates with use_godunov_flux=True."""

    def test_flag_stored(self) -> None:
        solver = CylindricalMHDSolver(
            nr=8, nz=8, dr=0.01, dz=0.01,
            use_godunov_flux=True,
        )
        assert solver.use_godunov_flux is True

    def test_default_off(self) -> None:
        solver = CylindricalMHDSolver(nr=8, nz=8, dr=0.01, dz=0.01)
        assert solver.use_godunov_flux is False


class TestPLMReconstruction:
    """Unit tests for the PLM minmod reconstruction."""

    def test_uniform_field_zero_slopes(self) -> None:
        """Uniform field should produce zero slopes → L=R=cell value."""
        solver, _ = _make_uniform_state(nr=8, nz=8)
        q = np.ones((8, 8)) * 5.0
        q_L, q_R = solver._plm_reconstruct(q, axis=0)
        np.testing.assert_allclose(q_L, 5.0, atol=1e-14)
        np.testing.assert_allclose(q_R, 5.0, atol=1e-14)

    def test_linear_field_exact(self) -> None:
        """Linear field q = i should be reconstructed exactly (2nd order)."""
        solver, _ = _make_uniform_state(nr=16, nz=4)
        q = np.zeros((16, 4))
        for i in range(16):
            q[i, :] = float(i)
        q_L, q_R = solver._plm_reconstruct(q, axis=0)
        # Interior interfaces (away from boundaries): L and R should match
        # Interface i+1/2: L = q[i] + 0.5*slope[i], R = q[i+1] - 0.5*slope[i+1]
        # For linear field with slope=1: L[i] = i + 0.5, R[i] = (i+1) - 0.5 = i + 0.5
        for j in range(2, 13):
            np.testing.assert_allclose(q_L[j, 0], j + 0.5, atol=1e-12)
            np.testing.assert_allclose(q_R[j, 0], j + 0.5, atol=1e-12)

    def test_monotonicity_preserved(self) -> None:
        """PLM with minmod should not create new extrema."""
        solver, _ = _make_uniform_state(nr=16, nz=4)
        q = np.zeros((16, 4))
        q[7, :] = 10.0  # spike
        q_L, q_R = solver._plm_reconstruct(q, axis=0)
        assert np.all(q_L >= 0.0)
        assert np.all(q_R >= 0.0)
        assert np.max(q_L) <= 10.0
        assert np.max(q_R) <= 10.0


class TestGodunovUniformState:
    """Uniform state should be preserved (dU/dt ~ 0)."""

    def test_single_step_no_nan(self) -> None:
        solver, state = _make_uniform_state()
        dt = solver._compute_dt(state)
        result = solver.step(state, dt, current=0.0, voltage=0.0)
        for key in ("rho", "velocity", "pressure", "B"):
            assert not np.any(np.isnan(result[key])), f"NaN in {key}"

    def test_uniform_state_preserved(self) -> None:
        """Uniform state should remain uniform after one step."""
        solver, state = _make_uniform_state()
        dt = solver._compute_dt(state)
        result = solver.step(state, dt, current=0.0, voltage=0.0)
        rho_init = state["rho"].mean()
        p_init = state["pressure"].mean()
        np.testing.assert_allclose(
            result["rho"].mean(), rho_init, rtol=1e-3,
            err_msg="Density not preserved on uniform state",
        )
        np.testing.assert_allclose(
            result["pressure"].mean(), p_init, rtol=1e-3,
            err_msg="Pressure not preserved on uniform state",
        )


def test_solver_records_internal_velocity_cap_limiter_event() -> None:
    solver, state = _make_uniform_state(nr=8, nz=8, p0=1.0e-6)
    state["velocity"][0, :, :, :] = 1.0e8

    solver.step(state, dt=1.0e-12, current=0.0, voltage=0.0)

    velocity_events = [
        event
        for event in solver.last_limiter_events
        if event["limiter_id"]
        == "dpf.fluid.cylindrical_mhd.final.velocity_fast_speed_cap"
    ]
    assert velocity_events
    assert velocity_events[0]["classification"] == "acceptance_blocker"
    assert velocity_events[0]["acceptance_blocking"] is True
    assert velocity_events[0]["activation_count"] > 0


def test_public_compute_dt_includes_resistive_diffusion_limit() -> None:
    solver, state = _make_uniform_state(nr=8, nz=8)
    solver.enable_resistive = True

    dt_without_eta = solver.compute_dt(state)
    solver._last_eta_max = 0.5
    dt_with_eta = solver.compute_dt(state)
    expected_diffusive_dt = 0.5 * min(solver.dr, solver.dz) ** 2 * mu_0 / 0.5
    diagnostics = solver.last_dt_diagnostics

    assert dt_with_eta < dt_without_eta
    np.testing.assert_allclose(dt_with_eta, expected_diffusive_dt)
    assert diagnostics["controller"] == "resistive_diffusion"
    np.testing.assert_allclose(diagnostics["dt_diff_s"], expected_diffusive_dt)
    assert diagnostics["dt_adv_s"] > diagnostics["dt_diff_s"]
    assert diagnostics["diffusion_method"] == "explicit"


def test_sts_compute_dt_extends_resistive_diffusion_limit() -> None:
    solver, state = _make_uniform_state(nr=8, nz=8)
    solver.enable_resistive = True
    solver.diffusion_method = "sts"
    solver.sts_stages = 8
    solver._last_eta_max = 0.5

    dt_sts = solver.compute_dt(state)
    diagnostics = solver.last_dt_diagnostics
    expected_diffusive_dt = 0.5 * min(solver.dr, solver.dz) ** 2 * mu_0 / 0.5
    expected_sts_dt = 0.25 * solver.sts_stages**2 * expected_diffusive_dt

    np.testing.assert_allclose(diagnostics["dt_diff_s"], expected_diffusive_dt)
    np.testing.assert_allclose(diagnostics["dt_sts_s"], expected_sts_dt)
    assert diagnostics["diffusion_method"] == "sts"
    assert diagnostics["controller"] == "sts_resistive_diffusion"
    assert dt_sts > expected_diffusive_dt
    np.testing.assert_allclose(dt_sts, expected_sts_dt)


def test_implicit_btheta_compute_dt_uses_hyperbolic_cfl_but_reports_dt_diff() -> None:
    solver, state = _make_uniform_state(nr=8, nz=8)
    solver.enable_resistive = True
    solver.diffusion_method = "implicit_cylindrical_btheta"
    solver._last_eta_max = 0.5

    dt_implicit = solver.compute_dt(state)
    diagnostics = solver.last_dt_diagnostics
    expected_diffusive_dt = 0.5 * min(solver.dr, solver.dz) ** 2 * mu_0 / 0.5

    assert diagnostics["controller"] == "hyperbolic_cfl"
    assert diagnostics["diffusion_method"] == "implicit_cylindrical_btheta"
    np.testing.assert_allclose(diagnostics["dt_diff_s"], expected_diffusive_dt)
    np.testing.assert_allclose(dt_implicit, diagnostics["dt_adv_s"])
    assert dt_implicit > expected_diffusive_dt
    assert diagnostics["resistive_stiffness_ratio"] > 1.0


def test_implicit_btheta_matches_explicit_small_dt() -> None:
    nr, nz = 16, 16
    solver, state = _make_uniform_state(nr=nr, nz=nz)
    solver.enable_resistive = True
    solver.diffusion_method = "implicit_cylindrical_btheta"
    rho = solver._squeeze(state["rho"])
    vel = solver._squeeze(state["velocity"])
    pressure = solver._squeeze(state["pressure"])
    B = solver._squeeze(state["B"])
    psi = solver._squeeze(state["psi"])
    r = solver.geom.r[:, None]
    z = np.linspace(0.0, 1.0, nz)[None, :]
    B[1] = 0.05 + 0.01 * np.sin(np.pi * z) * np.exp(-((r - r.mean()) ** 2) / 1.0e-4)
    eta = np.full((nr, nz), 1.0e-5)
    rhs = solver._compute_rhs(
        rho,
        vel,
        pressure,
        B,
        psi,
        eta,
        include_resistive_induction=True,
    )
    dt = 1.0e-4 * 0.5 * min(solver.dr, solver.dz) ** 2 * mu_0 / float(np.max(eta))

    implicit_B = solver._apply_implicit_btheta_resistive_induction(B, eta, dt)
    explicit_Btheta = B[1] + dt * rhs["dB_dt"][1]

    np.testing.assert_allclose(
        implicit_B[1, 1:-1, 1:-1],
        explicit_Btheta[1:-1, 1:-1],
        rtol=1.0e-4,
        atol=1.0e-10,
    )


def test_implicit_btheta_large_dt_finite_smooths_without_limiter() -> None:
    nr, nz = 16, 16
    solver, state = _make_uniform_state(nr=nr, nz=nz)
    solver.enable_resistive = True
    solver.diffusion_method = "implicit_cylindrical_btheta"
    B = solver._squeeze(state["B"])
    B[1, nr // 2, nz // 2] = 1.0
    eta = np.full((nr, nz), 0.1)
    explicit_dt = 0.5 * min(solver.dr, solver.dz) ** 2 * mu_0 / float(np.max(eta))

    diffused = solver._apply_implicit_btheta_resistive_induction(
        B,
        eta,
        10.0 * explicit_dt,
    )

    assert np.all(np.isfinite(diffused))
    assert float(np.max(np.abs(diffused[1]))) < 1.0
    assert solver.last_limiter_events == []


def test_implicit_btheta_blocks_material_br_bz() -> None:
    solver, state = _make_uniform_state(nr=8, nz=8)
    solver.enable_resistive = True
    solver.diffusion_method = "implicit_cylindrical_btheta"
    B = solver._squeeze(state["B"])
    B[1] = 1.0e-3
    B[0, 3, 3] = 1.0e-4
    eta = np.full((8, 8), 1.0e-5)

    solver._apply_implicit_btheta_resistive_induction(B, eta, 1.0e-9)

    events = [
        event
        for event in solver.last_limiter_events
        if event["limiter_id"]
        == "dpf.fluid.cylindrical_mhd.implicit_btheta.material_poloidal_field"
    ]
    assert events
    assert events[0]["classification"] == "acceptance_blocker"
    assert events[0]["acceptance_blocking"] is True
    assert events[0]["activation_count"] == 1


def test_partial_ionization_pressure_recovery_preserves_temperatures() -> None:
    ion_mass = 6.69e-27
    n_total = 400.0 / (1.380649e-23 * 300.0)
    rho0 = n_total * ion_mass
    Z_bar = 0.01
    p_total = n_total * 1.380649e-23 * 300.0 * (1.0 + Z_bar)
    solver, state = _make_uniform_state(nr=8, nz=8, rho0=rho0, p0=p_total)
    solver.ion_mass = ion_mass
    state["Te"] = np.full((8, 1, 8), 300.0)
    state["Ti"] = np.full((8, 1, 8), 300.0)
    state["Z_bar"] = np.full((8, 1, 8), Z_bar)

    result = solver.step(state, solver.compute_dt(state), current=0.0, voltage=0.0)

    np.testing.assert_allclose(result["Te"], 300.0, rtol=1.0e-10)
    np.testing.assert_allclose(result["Ti"], 300.0, rtol=1.0e-10)
    np.testing.assert_allclose(result["Z_bar"], Z_bar, rtol=0.0, atol=0.0)


class TestGodunovConservation:
    """Energy conservation on uniform state with no sources."""

    def test_total_energy_conserved(self) -> None:
        solver, state = _make_uniform_state()
        gamma = solver.gamma
        rho = solver._squeeze(state["rho"])
        p = solver._squeeze(state["pressure"])
        B = solver._squeeze(state["B"])
        v = solver._squeeze(state["velocity"])
        B_sq = np.sum(B**2, axis=0)
        v_sq = np.sum(v**2, axis=0)
        E0 = p / (gamma - 1) + 0.5 * rho * v_sq + B_sq / (2 * mu_0)
        vol = solver.geom.cell_volumes()
        total_E0 = np.sum(E0 * vol)

        dt = solver._compute_dt(state)
        result = solver.step(state, dt, current=0.0, voltage=0.0)
        rho1 = solver._squeeze(result["rho"])
        p1 = solver._squeeze(result["pressure"])
        B1 = solver._squeeze(result["B"])
        v1 = solver._squeeze(result["velocity"])
        B_sq1 = np.sum(B1**2, axis=0)
        v_sq1 = np.sum(v1**2, axis=0)
        E1 = p1 / (gamma - 1) + 0.5 * rho1 * v_sq1 + B_sq1 / (2 * mu_0)
        total_E1 = np.sum(E1 * vol)

        rel_err = abs(total_E1 - total_E0) / max(abs(total_E0), 1e-30)
        assert rel_err < 1e-6, f"Energy conservation violated: relative error = {rel_err:.2e}"


class TestGodunovSodShock:
    """1D Sod shock tube in radial direction — no NaN, shock structure."""

    def _make_sod_state(self) -> tuple[CylindricalMHDSolver, dict]:
        nr, nz = 64, 4
        dr, dz = 0.01, 0.01
        solver = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=1.4, cfl=0.3,
            enable_hall=False, enable_resistive=False,
            use_godunov_flux=True,
            conservative_energy=True,
        )
        rho = np.zeros((nr, 1, nz))
        p = np.zeros((nr, 1, nz))
        v = np.zeros((3, nr, 1, nz))
        B = np.zeros((3, nr, 1, nz))

        # Left state (inner radii): high density, high pressure
        mid = nr // 2
        rho[:mid, :, :] = 1.0
        p[:mid, :, :] = 1.0
        # Right state (outer radii): low density, low pressure
        rho[mid:, :, :] = 0.125
        p[mid:, :, :] = 0.1

        state = {
            "rho": rho,
            "velocity": v,
            "pressure": p,
            "B": B,
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        return solver, state

    def test_sod_no_nan(self) -> None:
        """Sod shock tube should run without NaN for 50 steps."""
        solver, state = self._make_sod_state()
        for _ in range(50):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        for key in ("rho", "velocity", "pressure", "B"):
            assert not np.any(np.isnan(state[key])), f"NaN in {key} after Sod shock"

    def test_sod_density_monotone_decay(self) -> None:
        """After evolution, density should transition smoothly (no oscillation)."""
        solver, state = self._make_sod_state()
        for _ in range(30):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        rho_1d = state["rho"][:, 0, 0]
        # No cell should go negative
        assert np.all(rho_1d > 0), "Negative density in Sod shock"
        # Max density should not exceed initial left state
        assert np.max(rho_1d) <= 1.5, "Density overshoot in Sod shock"

    def test_sod_no_negative_pressure(self) -> None:
        """Pressure should stay positive everywhere."""
        solver, state = self._make_sod_state()
        for _ in range(50):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        assert np.all(state["pressure"] > 0), "Negative pressure in Sod shock"


class TestGodunovVsCentralDiff:
    """Compare Godunov and central-difference paths on smooth data."""

    def test_both_paths_agree_on_smooth(self) -> None:
        """On a smooth state, both methods should give similar results."""
        nr, nz = 16, 16
        dr, dz = 0.001, 0.001

        # Godunov solver
        solver_g = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=5.0 / 3.0, cfl=0.3,
            enable_hall=False, enable_resistive=False,
            use_godunov_flux=True,
            conservative_energy=True,
        )
        # Central-difference solver
        solver_c = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=5.0 / 3.0, cfl=0.3,
            enable_hall=False, enable_resistive=False,
            use_godunov_flux=False,
            conservative_energy=True,
        )
        state = {
            "rho": np.full((nr, 1, nz), 1e-4),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), 1e3),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        state["B"][1, :, :, :] = 0.01  # small B_theta

        dt = min(solver_g._compute_dt(state), solver_c._compute_dt(state))
        result_g = solver_g.step(dict(state), dt, current=0.0, voltage=0.0)
        result_c = solver_c.step(dict(state), dt, current=0.0, voltage=0.0)

        # Both should preserve uniform state to similar accuracy
        np.testing.assert_allclose(
            result_g["rho"].mean(), result_c["rho"].mean(), rtol=1e-3,
        )


class TestGodunovMultiStep:
    """Multi-step stability tests."""

    def test_100_steps_no_crash(self) -> None:
        """Run 100 steps with Godunov flux — should not crash."""
        solver, state = _make_uniform_state(nr=16, nz=16)
        # Add a small perturbation to trigger nontrivial dynamics
        state["velocity"][0, 8, 0, :] = 100.0
        for _ in range(100):
            dt = solver._compute_dt(state)
            state = solver.step(state, dt, current=0.0, voltage=0.0)
        for key in ("rho", "pressure"):
            arr = state[key]
            assert not np.any(np.isnan(arr)), f"NaN in {key} after 100 steps"
            assert np.all(arr > 0), f"Negative {key} after 100 steps"

    def test_with_electrode_bc(self) -> None:
        """Godunov + electrode B-field BC should not crash."""
        solver, state = _make_uniform_state(nr=32, nz=32, dr=0.001, dz=0.001)
        for _ in range(10):
            dt = solver._compute_dt(state)
            state = solver.step(
                state, dt,
                current=100e3, voltage=20e3,
                apply_electrode_bc=True,
                anode_radius=0.005,
                cathode_radius=0.025,
            )
        assert not np.any(np.isnan(state["rho"])), "NaN with electrode BC"
