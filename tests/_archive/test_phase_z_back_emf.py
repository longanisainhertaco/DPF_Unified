"""Phase Z: Back-EMF coupling tests for SimulationEngine._compute_back_emf().

Tests the motional back-EMF method that computes the -(v x B) electric field
contribution to the circuit voltage.  The back-EMF arises from plasma bulk flow
advecting the magnetic field, coupling the MHD and RLC circuit subsystems.

Physics:
    Cylindrical:  EMF = mean(-(v_r * B_theta)) * z_length   [V]
    Cartesian:    EMF = mean(-(v_x * B_y - v_y * B_x)) * z_length  [V]
    z_length = nz * dz  (electrode gap)

Guard conditions (return 0.0):
    - velocity is None
    - B is None
    - velocity.shape[0] < 2  (insufficient components)
    - B.shape[0] < 2          (insufficient components)

References:
    Freidberg J.P., Ideal MHD, Cambridge 2014, §4.3 (Ohm's law).
    Lee S. & Saw S.H., J. Fusion Energy 27, 292 (2008).
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.presets import get_preset

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_cylindrical_engine(nr: int = 8, nz: int = 8) -> SimulationEngine:
    """Create a small cylindrical engine for back-EMF tests.

    Args:
        nr: Number of radial cells.
        nz: Number of axial cells.

    Returns:
        Initialized SimulationEngine with cylindrical geometry.
    """
    preset = get_preset("pf1000")
    preset["grid_shape"] = [nr, 1, nz]
    preset["sim_time"] = 1e-8
    preset["dx"] = 1e-3
    preset["geometry"] = {"type": "cylindrical"}
    # Disable heavy physics to keep init fast
    preset["snowplow"] = {"enabled": False}
    preset["radiation"] = {"bremsstrahlung_enabled": False}
    preset["sheath"] = {"enabled": False}
    preset["fluid"] = {"backend": "python"}
    preset["diagnostics"] = {"hdf5_filename": ":memory:"}
    config = SimulationConfig(**preset)
    return SimulationEngine(config)


def _make_cartesian_engine(nx: int = 8, nz: int = 8) -> SimulationEngine:
    """Create a small Cartesian engine for back-EMF tests.

    Args:
        nx: Number of x cells (also used for y).
        nz: Number of z cells.

    Returns:
        Initialized SimulationEngine with Cartesian geometry.
    """
    preset = get_preset("tutorial")
    preset["grid_shape"] = [nx, nx, nz]
    preset["sim_time"] = 1e-8
    preset["diagnostics"] = {"hdf5_filename": ":memory:"}
    config = SimulationConfig(**preset)
    return SimulationEngine(config)


def _z_length_cylindrical(engine: SimulationEngine) -> float:
    """Return the axial electrode gap z_length = nz * dz for a cylindrical engine."""
    nz = engine.config.grid_shape[2]
    dx = engine.config.dx
    dz = engine.config.geometry.dz if engine.config.geometry.dz else dx
    return nz * dz


def _z_length_cartesian(engine: SimulationEngine) -> float:
    """Return z_length = nz * dz for a Cartesian engine."""
    nz = engine.config.grid_shape[2]
    dx = engine.config.dx
    dz = engine.config.geometry.dz if engine.config.geometry.dz else dx
    return nz * dz


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Zero velocity → zero EMF (cylindrical)
# ═══════════════════════════════════════════════════════════════════════════════


def test_zero_velocity_gives_zero_emf() -> None:
    """Zero velocity field should produce exactly 0 V back-EMF.

    With velocity = 0 everywhere, the motional EMF -(v x B) vanishes
    regardless of the B-field configuration.
    """
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    # Set B non-zero so we verify it really is velocity that matters
    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["B"] = np.ones((3, nr, 1, nz)) * 0.5

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(0.0, abs=1e-30), (
        f"Expected 0 V with zero velocity, got {result} V"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Zero B-field → zero EMF
# ═══════════════════════════════════════════════════════════════════════════════


def test_zero_bfield_gives_zero_emf() -> None:
    """Zero magnetic field should produce exactly 0 V back-EMF.

    With B = 0 everywhere, the cross product v x B is zero, giving no EMF.
    """
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    engine.state["velocity"] = np.ones((3, nr, 1, nz)) * 1e5
    engine.state["B"] = np.zeros((3, nr, 1, nz))

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(0.0, abs=1e-30), (
        f"Expected 0 V with zero B-field, got {result} V"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Uniform v_r and B_theta (cylindrical) — analytic check
# ═══════════════════════════════════════════════════════════════════════════════


def test_uniform_vr_btheta_cylindrical() -> None:
    """Uniform v_r=1e5 m/s and B_theta=0.5 T should give back_emf = -5e4 * z_length.

    Formula: EMF = mean(-(v_r * B_theta)) * z_length
                 = -(1e5 * 0.5) * z_length
                 = -5e4 * z_length  [V]
    """
    engine = _make_cylindrical_engine(nr=8, nz=8)
    nr, _, nz = engine.config.grid_shape
    v_r = 1e5   # m/s — radial velocity component (index 0)
    B_th = 0.5  # T  — theta component (index 1)

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r  # v_r uniform
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th  # B_theta uniform

    z_len = _z_length_cylindrical(engine)
    expected_emf = -(v_r * B_th) * z_len  # -5e4 * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Expected {expected_emf:.6e} V, got {result:.6e} V "
        f"(z_length={z_len:.4e} m)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Sign convention — imploding plasma gives positive back-EMF
# ═══════════════════════════════════════════════════════════════════════════════


def test_sign_convention_imploding_plasma() -> None:
    """Imploding plasma (v_r < 0) with B_theta > 0 should give positive back-EMF.

    During DPF implosion the sheath moves inward (v_r < 0).  The motional
    EMF -(v_r * B_theta) = -(-|v| * B_theta) = +|v|*B_theta > 0, opposing
    the driving voltage (Lenz's law).
    """
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    v_r = -2e5  # m/s — inward (imploding)
    B_th = 1.0  # T  — azimuthal field

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th

    result = engine._compute_back_emf(dt=1e-9)

    assert result > 0.0, (
        f"Imploding plasma (v_r<0, B_theta>0) should give positive back-EMF, "
        f"got {result:.4e} V"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Cartesian geometry — cross product v_x*B_y - v_y*B_x
# ═══════════════════════════════════════════════════════════════════════════════


def test_cartesian_geometry_cross_product() -> None:
    """Cartesian EMF uses -(v_x*B_y - v_y*B_x).

    Set v_x=1e5, B_y=0.3 T, v_y=0, B_x=0.
    Expected: EMF = -(1e5 * 0.3 - 0) * z_length = -3e4 * z_length [V].
    """
    engine = _make_cartesian_engine(nx=8, nz=8)
    nx, ny, nz = engine.config.grid_shape
    v_x = 1e5
    B_y = 0.3

    engine.state["velocity"] = np.zeros((3, nx, ny, nz))
    engine.state["velocity"][0] = v_x  # v_x
    engine.state["B"] = np.zeros((3, nx, ny, nz))
    engine.state["B"][1] = B_y  # B_y

    z_len = _z_length_cartesian(engine)
    expected_emf = -(v_x * B_y) * z_len  # -3e4 * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Cartesian back-EMF: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


def test_cartesian_geometry_both_components() -> None:
    """Cartesian EMF with both (v_x, B_y) and (v_y, B_x) active.

    Set v_x=1e5, B_y=0.3, v_y=5e4, B_x=0.1.
    Cross term: v_x*B_y - v_y*B_x = 1e5*0.3 - 5e4*0.1 = 3e4 - 5e3 = 2.5e4
    Expected EMF = -2.5e4 * z_length [V].
    """
    engine = _make_cartesian_engine(nx=8, nz=8)
    nx, ny, nz = engine.config.grid_shape
    v_x = 1e5
    B_y = 0.3
    v_y = 5e4
    B_x = 0.1

    engine.state["velocity"] = np.zeros((3, nx, ny, nz))
    engine.state["velocity"][0] = v_x
    engine.state["velocity"][1] = v_y
    engine.state["B"] = np.zeros((3, nx, ny, nz))
    engine.state["B"][0] = B_x
    engine.state["B"][1] = B_y

    z_len = _z_length_cartesian(engine)
    cross = v_x * B_y - v_y * B_x  # 2.5e4
    expected_emf = -cross * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Cartesian both-component test: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: None velocity returns 0.0
# ═══════════════════════════════════════════════════════════════════════════════


def test_none_velocity_returns_zero() -> None:
    """When velocity is None, _compute_back_emf must return 0.0 (guard condition)."""
    engine = _make_cylindrical_engine()
    engine.state["velocity"] = None
    engine.state["B"] = np.ones((3, 8, 1, 8)) * 0.5

    result = engine._compute_back_emf(dt=1e-9)

    assert result == 0.0, f"Expected 0.0 when velocity is None, got {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: None B-field returns 0.0
# ═══════════════════════════════════════════════════════════════════════════════


def test_none_bfield_returns_zero() -> None:
    """When B is None, _compute_back_emf must return 0.0 (guard condition)."""
    engine = _make_cylindrical_engine()
    engine.state["velocity"] = np.ones((3, 8, 1, 8)) * 1e5
    engine.state["B"] = None

    result = engine._compute_back_emf(dt=1e-9)

    assert result == 0.0, f"Expected 0.0 when B is None, got {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Undersized velocity array returns 0.0
# ═══════════════════════════════════════════════════════════════════════════════


def test_undersized_velocity_returns_zero() -> None:
    """When velocity.shape[0] < 2, method must return 0.0.

    This guards against 1-component arrays that lack the second velocity
    component required to form the cross product.
    """
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    # Only 1 component — insufficient for any cross product
    engine.state["velocity"] = np.ones((1, nr, 1, nz)) * 1e5
    engine.state["B"] = np.ones((3, nr, 1, nz)) * 0.5

    result = engine._compute_back_emf(dt=1e-9)

    assert result == 0.0, (
        f"Expected 0.0 when velocity has only 1 component, got {result}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: Scales with z_length (nz doubles → EMF doubles)
# ═══════════════════════════════════════════════════════════════════════════════


def test_scales_with_z_length() -> None:
    """Doubling nz (electrode gap) should double the back-EMF.

    z_length = nz * dz, so EMF ∝ nz when dz is fixed.
    """
    v_r = 1e5
    B_th = 0.5

    engine_8 = _make_cylindrical_engine(nr=8, nz=8)
    nr_8, _, nz_8 = engine_8.config.grid_shape
    engine_8.state["velocity"] = np.zeros((3, nr_8, 1, nz_8))
    engine_8.state["velocity"][0] = v_r
    engine_8.state["B"] = np.zeros((3, nr_8, 1, nz_8))
    engine_8.state["B"][1] = B_th
    emf_8 = engine_8._compute_back_emf(dt=1e-9)

    engine_16 = _make_cylindrical_engine(nr=8, nz=16)
    nr_16, _, nz_16 = engine_16.config.grid_shape
    engine_16.state["velocity"] = np.zeros((3, nr_16, 1, nz_16))
    engine_16.state["velocity"][0] = v_r
    engine_16.state["B"] = np.zeros((3, nr_16, 1, nz_16))
    engine_16.state["B"][1] = B_th
    emf_16 = engine_16._compute_back_emf(dt=1e-9)

    # nz doubled → EMF should double (dz is the same for both)
    assert emf_16 == pytest.approx(2.0 * emf_8, rel=1e-10), (
        f"EMF should double when nz doubles: emf_8={emf_8:.4e}, emf_16={emf_16:.4e}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: Scales linearly with velocity
# ═══════════════════════════════════════════════════════════════════════════════


def test_scales_with_velocity() -> None:
    """Doubling v_r should double the back-EMF magnitude.

    EMF = mean(-(v_r * B_theta)) * z_length  ∝  v_r.
    """
    B_th = 0.5
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape

    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = 1e5
    emf_1x = engine._compute_back_emf(dt=1e-9)

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = 2e5
    emf_2x = engine._compute_back_emf(dt=1e-9)

    assert emf_2x == pytest.approx(2.0 * emf_1x, rel=1e-10), (
        f"EMF should double when v doubles: emf_1x={emf_1x:.4e}, emf_2x={emf_2x:.4e}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 11: Scales linearly with B-field
# ═══════════════════════════════════════════════════════════════════════════════


def test_scales_with_bfield() -> None:
    """Doubling B_theta should double the back-EMF magnitude.

    EMF = mean(-(v_r * B_theta)) * z_length  ∝  B_theta.
    """
    v_r = 1e5
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r

    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = 0.5
    emf_1x = engine._compute_back_emf(dt=1e-9)

    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = 1.0
    emf_2x = engine._compute_back_emf(dt=1e-9)

    assert emf_2x == pytest.approx(2.0 * emf_1x, rel=1e-10), (
        f"EMF should double when B doubles: emf_1x={emf_1x:.4e}, emf_2x={emf_2x:.4e}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12: Non-uniform fields — mean is computed correctly
# ═══════════════════════════════════════════════════════════════════════════════


def test_nonuniform_fields_mean_computed_correctly() -> None:
    """Non-uniform velocity and B fields: back-EMF equals mean(-(v_r * B_theta)) * z_len.

    This verifies the spatial mean is taken over the full grid (not just a
    single cell), and that non-uniform distributions are handled correctly.
    """
    engine = _make_cylindrical_engine(nr=4, nz=4)
    nr, _, nz = engine.config.grid_shape

    # Linearly varying v_r: 1e4, 2e4, 3e4, 4e4 along r-axis
    v_r_values = np.array([1e4, 2e4, 3e4, 4e4])
    # Constant B_theta
    B_th_value = 0.8

    velocity = np.zeros((3, nr, 1, nz))
    for ir in range(nr):
        velocity[0, ir, 0, :] = v_r_values[ir]
    engine.state["velocity"] = velocity

    B = np.zeros((3, nr, 1, nz))
    B[1] = B_th_value
    engine.state["B"] = B

    # mean(-(v_r * B_theta)) over all cells
    emf_density = -(velocity[0] * B[1])
    expected_mean = float(np.mean(emf_density))
    z_len = _z_length_cylindrical(engine)
    expected_emf = expected_mean * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Non-uniform fields: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 13: Non-uniform Cartesian fields — mean over all grid cells
# ═══════════════════════════════════════════════════════════════════════════════


def test_nonuniform_fields_cartesian() -> None:
    """Non-uniform Cartesian fields: EMF = mean(-(v_x*B_y - v_y*B_x)) * z_len."""
    engine = _make_cartesian_engine(nx=4, nz=4)
    nx, ny, nz = engine.config.grid_shape

    rng = np.random.default_rng(42)
    v_x = rng.uniform(0, 1e5, (nx, ny, nz))
    v_y = rng.uniform(0, 5e4, (nx, ny, nz))
    B_x = rng.uniform(0, 0.5, (nx, ny, nz))
    B_y = rng.uniform(0, 0.5, (nx, ny, nz))

    velocity = np.zeros((3, nx, ny, nz))
    velocity[0] = v_x
    velocity[1] = v_y
    engine.state["velocity"] = velocity

    B = np.zeros((3, nx, ny, nz))
    B[0] = B_x
    B[1] = B_y
    engine.state["B"] = B

    emf_density = -(v_x * B_y - v_y * B_x)
    z_len = _z_length_cartesian(engine)
    expected_emf = float(np.mean(emf_density)) * z_len

    result = engine._compute_back_emf(dt=1e-9)

    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Cartesian non-uniform fields: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14: Returns float (not ndarray)
# ═══════════════════════════════════════════════════════════════════════════════


def test_returns_float_type() -> None:
    """_compute_back_emf must return a Python float (not ndarray or np.floating).

    The circuit solver expects a scalar float for the back-EMF argument.
    """
    engine = _make_cylindrical_engine()
    nr, _, nz = engine.config.grid_shape
    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = 1e5
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = 0.3

    result = engine._compute_back_emf(dt=1e-9)

    assert isinstance(result, float), (
        f"_compute_back_emf must return float, got {type(result)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 15: Back-EMF integrates in circuit — nonzero after fields are set
# ═══════════════════════════════════════════════════════════════════════════════


def test_back_emf_integrates_in_circuit() -> None:
    """After setting velocity and B, back-EMF should be nonzero and affect the circuit.

    This is an integration test verifying that the computed EMF is passed
    into the RLC solver on each step.  We manually set state fields and call
    the method, then confirm the result is consistent with the analytic formula.
    """
    engine = _make_cylindrical_engine(nr=8, nz=8)
    nr, _, nz = engine.config.grid_shape

    # Representative DPF-like conditions: fast inward flow + azimuthal B
    v_r = -5e4   # m/s inward
    B_th = 0.2   # T

    engine.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine.state["velocity"][0] = v_r
    engine.state["B"] = np.zeros((3, nr, 1, nz))
    engine.state["B"][1] = B_th

    z_len = _z_length_cylindrical(engine)
    expected_emf = -(v_r * B_th) * z_len  # positive (opposing voltage)

    result = engine._compute_back_emf(dt=1e-9)

    # Verify nonzero and correct sign (positive for imploding plasma)
    assert result > 0.0, f"Back-EMF should be positive for imploding plasma, got {result:.4e} V"
    assert result == pytest.approx(expected_emf, rel=1e-10), (
        f"Integration test: expected {expected_emf:.4e} V, got {result:.4e} V"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 16: dz parameter affects z_length correctly
# ═══════════════════════════════════════════════════════════════════════════════


def test_dz_parameter_affects_z_length() -> None:
    """When geometry.dz is explicitly set, it should be used instead of dx.

    z_length = nz * dz.  Setting dz = 2*dx should double EMF vs dz=dx.
    """
    v_r = 1e5
    B_th = 0.5
    nr, nz = 8, 8

    # Engine 1: dz defaults to dx (1e-3 m)
    preset_a = get_preset("pf1000")
    preset_a["grid_shape"] = [nr, 1, nz]
    preset_a["sim_time"] = 1e-8
    preset_a["dx"] = 1e-3
    preset_a["geometry"] = {"type": "cylindrical", "dz": None}
    preset_a["snowplow"] = {"enabled": False}
    preset_a["radiation"] = {"bremsstrahlung_enabled": False}
    preset_a["sheath"] = {"enabled": False}
    preset_a["fluid"] = {"backend": "python"}
    preset_a["diagnostics"] = {"hdf5_filename": ":memory:"}
    engine_a = SimulationEngine(SimulationConfig(**preset_a))
    engine_a.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine_a.state["velocity"][0] = v_r
    engine_a.state["B"] = np.zeros((3, nr, 1, nz))
    engine_a.state["B"][1] = B_th
    emf_a = engine_a._compute_back_emf(dt=1e-9)

    # Engine 2: explicit dz = 2e-3 m (double)
    preset_b = get_preset("pf1000")
    preset_b["grid_shape"] = [nr, 1, nz]
    preset_b["sim_time"] = 1e-8
    preset_b["dx"] = 1e-3
    preset_b["geometry"] = {"type": "cylindrical", "dz": 2e-3}
    preset_b["snowplow"] = {"enabled": False}
    preset_b["radiation"] = {"bremsstrahlung_enabled": False}
    preset_b["sheath"] = {"enabled": False}
    preset_b["fluid"] = {"backend": "python"}
    preset_b["diagnostics"] = {"hdf5_filename": ":memory:"}
    engine_b = SimulationEngine(SimulationConfig(**preset_b))
    engine_b.state["velocity"] = np.zeros((3, nr, 1, nz))
    engine_b.state["velocity"][0] = v_r
    engine_b.state["B"] = np.zeros((3, nr, 1, nz))
    engine_b.state["B"][1] = B_th
    emf_b = engine_b._compute_back_emf(dt=1e-9)

    # dz doubled → EMF should double
    assert emf_b == pytest.approx(2.0 * emf_a, rel=1e-10), (
        f"dz=2e-3 should give 2x EMF vs dz=1e-3: emf_a={emf_a:.4e}, emf_b={emf_b:.4e}"
    )
