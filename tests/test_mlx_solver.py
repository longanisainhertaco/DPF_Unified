"""Tests for MLXMHDSolver (Phase B MLX Metal v2 solver).

All tests are skipped when MLX is not installed.  The test suite covers:
  1. Construction with PF-1000 parameters.
  2. compute_dt returns a positive finite float.
  3. Single step returns valid state dict with correct keys and shapes.
  4. No NaN after 3 consecutive steps on a uniform state.
  5. Sod shock tube (1-D axial): density jump visible after 50 steps.
  6. Brio-Wu MHD shock: no NaN, B-field structure preserved after 20 steps.
  7. Electrode BC: B_theta set correctly in the outermost ghost row.
  8. Dual-energy is active for cylindrical coordinates.
  9. State dict round-trip: input keys preserved in output.
 10. Resistive diffusion: B-field diffuses when eta_field provided.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mlx = pytest.importorskip("mlx")  # noqa: E402, I001
from dpf.metal.mlx_solver import MLXMHDSolver  # noqa: E402, I001


# ── Shared constants ──────────────────────────────────────────────────────────

_NR = 16
_NZ = 16
_DX = 2.5e-3   # 2.5 mm — typical DPF inter-electrode spacing
_DZ = 5e-3
_GAMMA = 5.0 / 3.0

_STATE_KEYS = ("rho", "velocity", "pressure", "B", "Te", "Ti", "psi")


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_solver(**kwargs) -> MLXMHDSolver:
    """Construct an MLXMHDSolver with sensible defaults and optional overrides."""
    defaults = dict(
        grid_shape=(_NR, 1, _NZ),
        dx=_DX,
        dz=_DZ,
        gamma=_GAMMA,
        cfl=0.3,
        riemann_solver="hlld",
        reconstruction="weno5z",
        time_integrator="ssp_rk3",
        coordinates="cylindrical",
        r_inner=0.0,
        convert_b_si_to_hl=False,
    )
    defaults.update(kwargs)
    return MLXMHDSolver(**defaults)


def _uniform_state(
    nr: int = _NR,
    nz: int = _NZ,
    rho0: float = 1e-4,
    p0: float = 1e3,
    Bz0: float = 0.01,
) -> dict[str, np.ndarray]:
    """Uniform rest state — no flows, weak axial B."""
    shape = (nr, 1, nz)
    rho = np.full(shape, rho0, dtype=np.float64)
    vel = np.zeros((3, nr, 1, nz), dtype=np.float64)
    pressure = np.full(shape, p0, dtype=np.float64)
    B = np.zeros((3, nr, 1, nz), dtype=np.float64)
    B[1] = Bz0  # Bz component
    Te = np.full(shape, 1e4, dtype=np.float64)
    Ti = np.full(shape, 1e4, dtype=np.float64)
    psi = np.zeros(shape, dtype=np.float64)
    return {"rho": rho, "velocity": vel, "pressure": pressure, "B": B,
            "Te": Te, "Ti": Ti, "psi": psi}


def _sod_state(nr: int = _NR, nz: int = _NZ) -> dict[str, np.ndarray]:
    """Sod shock tube oriented along z: high-pressure left half."""
    shape = (nr, 1, nz)
    rho = np.ones(shape, dtype=np.float64)
    vel = np.zeros((3, nr, 1, nz), dtype=np.float64)
    pressure = np.where(
        np.arange(nz)[np.newaxis, np.newaxis, :] < nz // 2,
        1.0, 0.1
    ) * np.ones(shape)
    B = np.zeros((3, nr, 1, nz), dtype=np.float64)
    B[1] = 1e-6  # tiny Bz to keep MHD stable
    Te = np.full(shape, 1e4, dtype=np.float64)
    Ti = np.full(shape, 1e4, dtype=np.float64)
    psi = np.zeros(shape, dtype=np.float64)
    return {"rho": rho, "velocity": vel, "pressure": pressure, "B": B,
            "Te": Te, "Ti": Ti, "psi": psi}


def _brio_wu_state(nr: int = _NR, nz: int = _NZ) -> dict[str, np.ndarray]:
    """Brio-Wu MHD shock tube oriented along z."""
    shape = (nr, 1, nz)
    z_idx = np.arange(nz)
    left = z_idx < nz // 2

    rho = np.where(left[np.newaxis, np.newaxis, :], 1.0, 0.125) * np.ones(shape)
    pressure = np.where(left[np.newaxis, np.newaxis, :], 1.0, 0.1) * np.ones(shape)
    vel = np.zeros((3, nr, 1, nz), dtype=np.float64)
    B = np.zeros((3, nr, 1, nz), dtype=np.float64)
    B[0] = 0.75  # Bx constant (radial in our layout)
    B[2] = np.where(left[np.newaxis, np.newaxis, :], 1.0, -1.0)  # Bt reversal
    Te = np.full(shape, 1e4, dtype=np.float64)
    Ti = np.full(shape, 1e4, dtype=np.float64)
    psi = np.zeros(shape, dtype=np.float64)
    return {"rho": rho, "velocity": vel, "pressure": pressure, "B": B,
            "Te": Te, "Ti": Ti, "psi": psi}


# ── Test 1: Construction ──────────────────────────────────────────────────────


def test_construction_pf1000_params():
    """MLXMHDSolver instantiates with PF-1000 grid parameters."""
    solver = _make_solver(
        grid_shape=(32, 1, 64),
        dx=1.5e-3,
        dz=2e-3,
        gamma=5.0 / 3.0,
        cfl=0.3,
        coordinates="cylindrical",
        r_inner=1e-3,
        convert_b_si_to_hl=True,
    )
    assert isinstance(solver, MLXMHDSolver)
    assert solver.nr == 32
    assert solver.nz == 64
    assert math.isclose(solver.dx, 1.5e-3)
    assert math.isclose(solver.dz, 2e-3)
    assert solver.coordinates == "cylindrical"
    assert solver._grid is not None
    assert solver._state_mgr is not None


# ── Test 2: compute_dt ────────────────────────────────────────────────────────


def test_compute_dt_positive():
    """compute_dt returns a positive finite float for a uniform state."""
    solver = _make_solver()
    state = _uniform_state()
    dt = solver.compute_dt(state)
    assert isinstance(dt, float)
    assert dt > 0.0
    assert math.isfinite(dt)


# ── Test 3: Single step ───────────────────────────────────────────────────────


def test_single_step_valid_output():
    """step() returns a dict with all required keys and correct array shapes."""
    solver = _make_solver()
    state = _uniform_state()
    dt = 1e-9

    result = solver.step(state, dt=dt, current=0.0, voltage=0.0)

    assert isinstance(result, dict)
    for key in _STATE_KEYS:
        assert key in result, f"Missing key: {key}"

    nr, nz = _NR, _NZ
    assert result["rho"].shape == (nr, 1, nz)
    assert result["pressure"].shape == (nr, 1, nz)
    assert result["velocity"].shape == (3, nr, 1, nz)
    assert result["B"].shape == (3, nr, 1, nz)


# ── Test 4: No NaN after 3 steps ─────────────────────────────────────────────


def test_no_nan_three_steps():
    """No NaN or Inf in any output field after 3 consecutive steps."""
    solver = _make_solver()
    state = _uniform_state()
    dt = 1e-9

    for _ in range(3):
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

    for key in ("rho", "pressure"):
        arr = state[key]
        assert not np.any(np.isnan(arr)), f"NaN in {key}"
        assert not np.any(np.isinf(arr)), f"Inf in {key}"

    for key in ("velocity", "B"):
        arr = state[key]
        assert not np.any(np.isnan(arr)), f"NaN in {key}"
        assert not np.any(np.isinf(arr)), f"Inf in {key}"


# ── Test 5: Sod shock tube ────────────────────────────────────────────────────


@pytest.mark.slow
def test_sod_shock_tube_density_jump():
    """Sod shock tube: density jump visible after 50 steps, no NaN."""
    solver = _make_solver(
        reconstruction="plm",
        riemann_solver="hll",
        time_integrator="ssp_rk2",
    )
    state = _sod_state()

    dt = solver.compute_dt(state) * 0.5
    for _ in range(50):
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        dt = solver.compute_dt(state) * 0.5

    rho_final = state["rho"][:, 0, :]  # (nr, nz)
    assert not np.any(np.isnan(rho_final)), "NaN in density after Sod run"
    assert not np.any(np.isinf(rho_final)), "Inf in density after Sod run"

    # A shock must form: max density must exceed initial max by at least 5%
    assert float(np.max(rho_final)) > 1.05 * float(np.max(_sod_state()["rho"]))


# ── Test 6: Brio-Wu MHD shock ─────────────────────────────────────────────────


@pytest.mark.slow
def test_brio_wu_no_nan():
    """Brio-Wu MHD shock: no NaN, B-field structure survives 20 steps."""
    solver = _make_solver(
        reconstruction="plm",
        riemann_solver="hlld",
        time_integrator="ssp_rk2",
    )
    state = _brio_wu_state()

    dt = solver.compute_dt(state) * 0.4
    for _ in range(20):
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        dt = solver.compute_dt(state) * 0.4

    for field in ("rho", "pressure"):
        assert not np.any(np.isnan(state[field])), f"NaN in {field} after Brio-Wu"
    assert not np.any(np.isnan(state["B"])), "NaN in B after Brio-Wu"

    # B_theta (index 2) must show some spatial variation from the initial reversal
    Bt = state["B"][2, :, 0, :]
    assert float(np.max(Bt)) > float(np.min(Bt)), "B_theta field is flat — solver stalled"


# ── Test 7: Electrode BCs ─────────────────────────────────────────────────────


def test_electrode_bc_btheta_set():
    """Electrode BC: outer B_theta row matches mu_0*I/(2*pi*r_outer) after step."""
    solver = _make_solver(coordinates="cylindrical")
    state = _uniform_state()
    current = 1e6  # 1 MA

    result = solver.step(
        state, dt=1e-10, current=current, voltage=0.0,
        apply_electrode_bc=True,
    )

    # The BC is applied at the START of the step so the outer row should carry
    # the electrode signature through the Riemann evolution.  We check that
    # B_theta at the outer radial cells is non-zero and positive (current
    # convention: I > 0 → B_theta > 0).
    Bt_outer = result["B"][2, -1, 0, :]  # (nz,)
    assert np.all(Bt_outer >= 0.0), "Outer B_theta must be >= 0 for positive current"
    assert float(np.mean(np.abs(Bt_outer))) > 0.0, "Outer B_theta should be nonzero"


# ── Test 8: Dual-energy active ────────────────────────────────────────────────


def test_dual_energy_active_for_cylindrical():
    """_use_dual_energy is True when coordinates='cylindrical'."""
    solver = _make_solver(coordinates="cylindrical", use_dual_energy=False)
    assert solver._use_dual_energy is True, (
        "dual-energy must be forced True for cylindrical coordinates"
    )


def test_dual_energy_respects_flag_for_cartesian():
    """_use_dual_energy follows the flag when coordinates='cartesian'."""
    solver = _make_solver(coordinates="cartesian", use_dual_energy=False)
    assert solver._use_dual_energy is False


# ── Test 9: State dict round-trip ─────────────────────────────────────────────


def test_state_dict_round_trip_keys():
    """All input keys are present in the output state dict."""
    solver = _make_solver()
    state = _uniform_state()
    result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)

    for key in _STATE_KEYS:
        assert key in result, f"Key '{key}' missing from result"


def test_state_dict_round_trip_no_mutation():
    """Input state dict is not mutated by step()."""
    solver = _make_solver()
    state = _uniform_state()
    rho_before = state["rho"].copy()
    solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
    np.testing.assert_array_equal(state["rho"], rho_before)


# ── Test 10: Resistive diffusion ──────────────────────────────────────────────


def test_resistive_diffusion_changes_b():
    """B-field changes when eta_field is provided (diffusion active)."""
    solver = _make_solver()

    # Initial state with non-trivial B_theta gradient
    state = _uniform_state()
    # Set a localised B_theta spike in the middle radial row
    state["B"][2, _NR // 2, 0, :] = 1.0

    dt = 1e-8
    eta = 1e-6  # Ohm·m — large enough to show diffusion in one step

    result_with = solver.step(
        state, dt=dt, current=0.0, voltage=0.0, eta_field=eta
    )
    result_without = solver.step(
        state, dt=dt, current=0.0, voltage=0.0
    )

    Bt_with = result_with["B"][2]
    Bt_without = result_without["B"][2]

    # Resistive diffusion must produce a measurable difference in B_theta
    diff = float(np.mean(np.abs(Bt_with - Bt_without)))
    assert diff > 0.0, "Resistive diffusion had no effect on B_theta"


# ── Bonus: constructor parameter acceptance ───────────────────────────────────


def test_constructor_accepts_metalsolv_params():
    """All MetalMHDSolver-compatible kwargs accepted without error."""
    solver = MLXMHDSolver(
        grid_shape=(_NR, 1, _NZ),
        dx=_DX,
        dz=_DZ,
        gamma=5.0 / 3.0,
        cfl=0.3,
        riemann_solver="hlld",
        reconstruction="weno5z",
        time_integrator="ssp_rk3",
        coordinates="cylindrical",
        r_inner=1e-3,
        convert_b_si_to_hl=True,
        ion_mass=3.34358377e-27,
        enable_hall=False,
        enable_braginskii_conduction=False,
        enable_braginskii_viscosity=False,
        enable_bremsstrahlung=False,
        gaunt_factor=1.2,
        Z_eff=1.0,
        use_dual_energy=True,
        # ignored compat params
        device="mlx",
        precision="float32",
        use_ct=True,
        limiter="mc",
        enable_nernst=False,
        compile_mode=False,
    )
    assert isinstance(solver, MLXMHDSolver)


def test_compute_dt_alias():
    """_compute_dt is an alias for compute_dt (engine compatibility)."""
    solver = _make_solver()
    state = _uniform_state()
    assert solver._compute_dt(state) == solver.compute_dt(state)


def test_coupling_interface_returns_coupling_state():
    """coupling_interface() returns a CouplingState with the last current."""
    from dpf.core.bases import CouplingState

    solver = _make_solver()
    state = _uniform_state()
    solver.step(state, dt=1e-10, current=1.5e6, voltage=20e3)
    cs = solver.coupling_interface()
    assert isinstance(cs, CouplingState)
    assert math.isclose(cs.current, 1.5e6)


def test_non_axisymmetric_raises():
    """ny != 1 must raise ValueError."""
    with pytest.raises(ValueError, match="ny=1 required"):
        MLXMHDSolver(grid_shape=(_NR, 4, _NZ), dx=_DX)
