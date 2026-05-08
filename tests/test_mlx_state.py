"""Tests for MLXState: DPF state dict <-> packed MLX array conversion.

Covers:
  - Round-trip fidelity (from_state_dict -> to_state_dict)
  - Conservation: total energy computed correctly from p, KE, ME
  - Entropy: S*rho matches p * rho^(1-gamma)
  - Shape: output restores (nr, 1, nz) DPF convention
  - Edge cases: floor values, zero velocity, zero B
  - Zero-copy: MLX <-> NumPy transfer correctness
  - B-field unit conversion (SI -> HL -> SI round-trip)
  - Electron energy from Te
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="MLX not installed")

from dpf.metal.mlx_state import (  # noqa: E402, I001
    IDN, IEN, ISR, IBR, IBZ, IBT, IEE, IMR, IMZ, IMT,
    MU0, NVAR, P_FLOOR, RHO_FLOOR, MLXState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GAMMA = 5.0 / 3.0
_K_B = 1.380649e-23
_M_D  = 3.34358377e-27

NR = 8
NZ = 12


def _make_state(
    rho_val: float = 1.0,
    vr_val: float = 2.0e4,
    vz_val: float = -1.0e4,
    vt_val: float = 5.0e3,
    p_val: float = 1.0e3,
    Br_val: float = 0.1,
    Bz_val: float = 0.2,
    Bt_val: float = 0.05,
    Te_val: float | None = None,
) -> dict[str, np.ndarray]:
    """Build a uniform DPF state dict with shape (nr, 1, nz)."""
    rho = np.full((NR, 1, NZ), rho_val, dtype=np.float64)
    vel = np.zeros((3, NR, 1, NZ), dtype=np.float64)
    vel[0] = vr_val
    vel[1] = vz_val
    vel[2] = vt_val
    p   = np.full((NR, 1, NZ), p_val,  dtype=np.float64)
    B   = np.zeros((3, NR, 1, NZ), dtype=np.float64)
    B[0] = Br_val
    B[1] = Bz_val
    B[2] = Bt_val
    Ti  = np.full((NR, 1, NZ), p_val * _M_D / (rho_val * _K_B), dtype=np.float64)
    state: dict[str, np.ndarray] = {
        "rho": rho, "velocity": vel, "pressure": p, "B": B, "Ti": Ti, "psi": np.zeros_like(rho)
    }
    if Te_val is not None:
        state["Te"] = np.full((NR, 1, NZ), Te_val, dtype=np.float64)
    return state


def _mlx_state(nr: int = NR, nz: int = NZ) -> MLXState:
    return MLXState(nr=nr, nz=nz, gamma=GAMMA)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMLXStateInit:
    def test_default_shape(self) -> None:
        ms = _mlx_state()
        assert ms.nr == NR
        assert ms.nz == NZ
        assert ms.gamma == pytest.approx(GAMMA)

    def test_initial_U_shape(self) -> None:
        ms = _mlx_state()
        arr = np.array(ms.U)
        assert arr.shape == (NVAR, NR, NZ)
        assert np.all(arr == 0.0)


# ---------------------------------------------------------------------------
# from_state_dict: shape and packing
# ---------------------------------------------------------------------------

class TestFromStateDict:
    def test_output_shape(self) -> None:
        ms = _mlx_state()
        state = _make_state()
        U = ms.from_state_dict(state)
        arr = np.array(U)
        assert arr.shape == (NVAR, NR, NZ)

    def test_density_packed(self) -> None:
        ms = _mlx_state()
        state = _make_state(rho_val=2.5)
        U = ms.from_state_dict(state)
        rho = np.array(U[IDN])
        assert rho == pytest.approx(2.5, rel=1e-6)

    def test_momentum_packed(self) -> None:
        rho, vr, vz, vt = 1.5, 3.0e4, -2.0e4, 1.0e3
        ms = _mlx_state()
        state = _make_state(rho_val=rho, vr_val=vr, vz_val=vz, vt_val=vt)
        U = ms.from_state_dict(state)
        arr = np.array(U)
        assert arr[IMR].flat[0] == pytest.approx(rho * vr, rel=1e-5)
        assert arr[IMZ].flat[0] == pytest.approx(rho * vz, rel=1e-5)
        assert arr[IMT].flat[0] == pytest.approx(rho * vt, rel=1e-5)

    def test_B_field_packed(self) -> None:
        ms = _mlx_state()
        state = _make_state(Br_val=0.3, Bz_val=0.7, Bt_val=0.15)
        U = ms.from_state_dict(state)
        arr = np.array(U)
        assert arr[IBR].flat[0] == pytest.approx(0.3, rel=1e-5)
        assert arr[IBZ].flat[0] == pytest.approx(0.7, rel=1e-5)
        assert arr[IBT].flat[0] == pytest.approx(0.15, rel=1e-5)

    def test_total_energy_correctness(self) -> None:
        rho, vr, vz, vt = 1.0, 1.0e4, 0.0, 0.0
        p, Br, Bz, Bt = 1.0e3, 0.1, 0.2, 0.0
        ms = _mlx_state()
        state = _make_state(rho_val=rho, vr_val=vr, vz_val=vz, vt_val=vt,
                            p_val=p, Br_val=Br, Bz_val=Bz, Bt_val=Bt)
        U = ms.from_state_dict(state)
        E_computed = float(np.array(U[IEN]).flat[0])
        KE = 0.5 * rho * (vr**2 + vz**2 + vt**2)
        ME = 0.5 * (Br**2 + Bz**2 + Bt**2)
        E_expected = p / (GAMMA - 1.0) + KE + ME
        assert E_computed == pytest.approx(E_expected, rel=1e-5)

    def test_entropy_tracer_correctness(self) -> None:
        rho, p = 2.0, 5.0e2
        ms = _mlx_state()
        state = _make_state(rho_val=rho, p_val=p)
        U = ms.from_state_dict(state)
        Srho_computed = float(np.array(U[ISR]).flat[0])
        Srho_expected = p * rho ** (1.0 - GAMMA)
        assert Srho_computed == pytest.approx(Srho_expected, rel=1e-5)

    def test_electron_energy_from_Te(self) -> None:
        rho, Te = 1.0, 2.0e6
        ms = _mlx_state()
        state = _make_state(rho_val=rho, Te_val=Te)
        U = ms.from_state_dict(state)
        e_elec = float(np.array(U[IEE]).flat[0])
        e_expected = 0.5 * rho * (_K_B / _M_D) * Te
        assert e_elec == pytest.approx(e_expected, rel=1e-5)

    def test_electron_energy_zero_without_Te(self) -> None:
        ms = _mlx_state()
        state = _make_state()
        U = ms.from_state_dict(state)
        e_elec = np.array(U[IEE])
        assert np.all(e_elec == 0.0)


# ---------------------------------------------------------------------------
# to_state_dict: shape and output keys
# ---------------------------------------------------------------------------

class TestToStateDict:
    def _packed(self) -> tuple[MLXState, object]:
        ms = _mlx_state()
        state = _make_state()
        U = ms.from_state_dict(state)
        return ms, U

    def test_output_keys(self) -> None:
        ms, U = self._packed()
        out = ms.to_state_dict(U)
        assert set(out.keys()) >= {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}

    def test_rho_shape(self) -> None:
        ms, U = self._packed()
        out = ms.to_state_dict(U)
        assert out["rho"].shape == (NR, 1, NZ)

    def test_velocity_shape(self) -> None:
        ms, U = self._packed()
        out = ms.to_state_dict(U)
        assert out["velocity"].shape == (3, NR, 1, NZ)

    def test_B_shape(self) -> None:
        ms, U = self._packed()
        out = ms.to_state_dict(U)
        assert out["B"].shape == (3, NR, 1, NZ)

    def test_output_dtype_float64(self) -> None:
        ms, U = self._packed()
        out = ms.to_state_dict(U)
        assert out["rho"].dtype == np.float64
        assert out["pressure"].dtype == np.float64


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _roundtrip(self, **kwargs: float) -> tuple[dict, dict]:
        ms = _mlx_state()
        orig = _make_state(**kwargs)
        U = ms.from_state_dict(orig)
        recovered = ms.to_state_dict(U)
        return orig, recovered

    def test_density_roundtrip(self) -> None:
        orig, rec = self._roundtrip(rho_val=3.14)
        assert rec["rho"] == pytest.approx(orig["rho"], rel=1e-5)

    def test_velocity_roundtrip(self) -> None:
        orig, rec = self._roundtrip(vr_val=5.0e4, vz_val=-3.0e4, vt_val=1.5e3)
        assert rec["velocity"] == pytest.approx(orig["velocity"], rel=1e-5)

    def test_pressure_roundtrip(self) -> None:
        orig, rec = self._roundtrip(p_val=2.0e4)
        assert rec["pressure"] == pytest.approx(orig["pressure"], rel=1e-5)

    def test_B_roundtrip(self) -> None:
        orig, rec = self._roundtrip(Br_val=0.5, Bz_val=1.0, Bt_val=0.25)
        assert rec["B"] == pytest.approx(orig["B"], rel=1e-5)

    def test_full_state_roundtrip(self) -> None:
        kwargs = dict(
            rho_val=1.2, vr_val=2.5e4, vz_val=-1.1e4, vt_val=8.0e2,
            p_val=3.0e3, Br_val=0.15, Bz_val=0.4, Bt_val=0.08,
        )
        orig, rec = self._roundtrip(**kwargs)
        assert rec["rho"] == pytest.approx(orig["rho"], rel=1e-5)
        assert rec["velocity"] == pytest.approx(orig["velocity"], rel=1e-5)
        assert rec["pressure"] == pytest.approx(orig["pressure"], rel=1e-5)
        assert rec["B"] == pytest.approx(orig["B"], rel=1e-5)

    def test_zero_velocity_roundtrip(self) -> None:
        orig, rec = self._roundtrip(vr_val=0.0, vz_val=0.0, vt_val=0.0)
        assert rec["velocity"] == pytest.approx(orig["velocity"], abs=1e-10)

    def test_zero_B_roundtrip(self) -> None:
        orig, rec = self._roundtrip(Br_val=0.0, Bz_val=0.0, Bt_val=0.0)
        assert rec["B"] == pytest.approx(orig["B"], abs=1e-10)


# ---------------------------------------------------------------------------
# Edge cases: floors
# ---------------------------------------------------------------------------

class TestFloors:
    def test_rho_floor_applied_on_pack(self) -> None:
        ms = _mlx_state()
        state = _make_state(rho_val=RHO_FLOOR * 1e-2)
        U = ms.from_state_dict(state)
        rho_out = float(np.array(U[IDN]).flat[0])
        # float32 can represent 1e-12 with ~1 ULP error; allow a small margin
        assert rho_out >= RHO_FLOOR * (1.0 - 1e-6)

    def test_p_floor_applied_on_pack(self) -> None:
        ms = _mlx_state()
        state = _make_state(p_val=0.0)
        U = ms.from_state_dict(state)
        arr = np.array(U)
        # Check entropy tracer positive (would be zero/negative without floor)
        Srho = arr[ISR].flat[0]
        assert Srho > 0.0

    def test_rho_floor_applied_on_unpack(self) -> None:
        ms = _mlx_state()
        # Build a state that packs near-zero rho
        state = _make_state(rho_val=RHO_FLOOR * 0.1, p_val=1.0e-20)
        U = ms.from_state_dict(state)
        out = ms.to_state_dict(U)
        assert np.all(out["rho"] >= 0.0)

    def test_pressure_floor_on_unpack(self) -> None:
        ms = _mlx_state()
        state = _make_state(p_val=P_FLOOR * 0.1)
        U = ms.from_state_dict(state)
        out = ms.to_state_dict(U)
        assert np.all(out["pressure"] >= 0.0)

    def test_pressure_unpack_ignores_rejected_nonfinite_candidate(self) -> None:
        ms = _mlx_state()
        U = ms.from_state_dict(_make_state(p_val=10.0, Br_val=0.0, Bz_val=0.0, Bt_val=0.0))
        U_np = np.asarray(U).astype(np.float32)
        U_np[ISR, 0, 0] = np.inf

        out = ms.to_state_dict(mlx.array(U_np))

        assert np.all(np.isfinite(out["pressure"]))
        assert out["pressure"][0, 0, 0] >= P_FLOOR * 0.999


# ---------------------------------------------------------------------------
# entropy_from_primitives
# ---------------------------------------------------------------------------

class TestEntropyFromPrimitives:
    def test_scalar_match(self) -> None:
        ms = _mlx_state()
        rho_val, p_val = 2.0, 1.0e3
        rho_np = np.full((NR, NZ), rho_val, dtype=np.float32)
        p_np   = np.full((NR, NZ), p_val,   dtype=np.float32)
        rho_mx = MLXState.zero_copy_to_mlx(rho_np)
        p_mx   = MLXState.zero_copy_to_mlx(p_np)
        Srho = ms.entropy_from_primitives(rho_mx, p_mx)
        result = float(np.array(Srho).flat[0])
        expected = p_val * rho_val ** (1.0 - GAMMA)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_recovery_round_trip(self) -> None:
        ms = _mlx_state()
        rho_val, p_val = 3.0, 2.5e3
        rho_np = np.full((NR, NZ), rho_val, dtype=np.float32)
        p_np   = np.full((NR, NZ), p_val,   dtype=np.float32)
        rho_mx = MLXState.zero_copy_to_mlx(rho_np)
        p_mx   = MLXState.zero_copy_to_mlx(p_np)
        Srho = ms.entropy_from_primitives(rho_mx, p_mx)
        # Recovery: p = Srho * rho^(gamma-1)
        import mlx.core as mx
        p_recovered = np.array(Srho * mx.power(rho_mx, GAMMA - 1.0))
        assert p_recovered == pytest.approx(p_val, rel=1e-5)


# ---------------------------------------------------------------------------
# Zero-copy transfer
# ---------------------------------------------------------------------------

class TestZeroCopy:
    def test_zero_copy_to_mlx_correct_values(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        mx_arr = MLXState.zero_copy_to_mlx(arr)
        result = np.array(mx_arr, copy=False)
        np.testing.assert_array_equal(result, arr)

    def test_zero_copy_to_mlx_float64_input_converted(self) -> None:
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        mx_arr = MLXState.zero_copy_to_mlx(arr)
        result = np.array(mx_arr, copy=False)
        np.testing.assert_allclose(result, arr.astype(np.float32), rtol=1e-6)

    def test_zero_copy_to_mlx_non_contiguous_handled(self) -> None:
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        non_contig = base[::2, ::2]  # not C-contiguous
        assert not non_contig.flags["C_CONTIGUOUS"]
        mx_arr = MLXState.zero_copy_to_mlx(non_contig)
        result = np.array(mx_arr, copy=False)
        np.testing.assert_array_equal(result, non_contig)

    def test_zero_copy_to_numpy_correct_values(self) -> None:
        import mlx.core as mx
        mx_arr = mx.array(np.array([[5.0, 6.0]], dtype=np.float32))
        result = MLXState.zero_copy_to_numpy(mx_arr)
        assert result[0, 0] == pytest.approx(5.0)
        assert result[0, 1] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# B-field unit conversion
# ---------------------------------------------------------------------------

class TestBFieldConversion:
    def test_si_to_hl_to_si_roundtrip(self) -> None:
        ms = _mlx_state()
        orig = _make_state(Br_val=1.0, Bz_val=2.0, Bt_val=0.5)
        # Pack with SI->HL conversion
        U = ms.from_state_dict(orig, convert_b_si_to_hl=True)
        # Unpack with HL->SI conversion
        out = ms.to_state_dict(U, convert_b_hl_to_si=True)
        assert out["B"] == pytest.approx(orig["B"], rel=1e-5)

    def test_hl_b_magnitude_scaled(self) -> None:
        ms = _mlx_state()
        B_val = 1.0
        orig = _make_state(Br_val=B_val, Bz_val=0.0, Bt_val=0.0)
        U_hl = ms.from_state_dict(orig, convert_b_si_to_hl=True)
        U_si = ms.from_state_dict(orig, convert_b_si_to_hl=False)
        Br_hl = float(np.array(U_hl[IBR]).flat[0])
        Br_si = float(np.array(U_si[IBR]).flat[0])
        expected_ratio = 1.0 / math.sqrt(MU0)
        assert Br_hl / Br_si == pytest.approx(expected_ratio, rel=1e-5)


# ---------------------------------------------------------------------------
# Conservation: energy components sum correctly
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    def test_E_decomposition(self) -> None:
        rho, vr, vz, vt = 1.0, 3.0e4, 0.0, 0.0
        p, Br, Bz, Bt = 5.0e3, 0.2, 0.0, 0.0
        ms = _mlx_state()
        state = _make_state(rho_val=rho, vr_val=vr, vz_val=vz, vt_val=vt,
                            p_val=p, Br_val=Br, Bz_val=Bz, Bt_val=Bt)
        U = ms.from_state_dict(state)
        arr = np.array(U)
        E = arr[IEN, 0, 0]
        KE = 0.5 * rho * (vr**2 + vz**2 + vt**2)
        ME = 0.5 * (Br**2 + Bz**2 + Bt**2)
        p_E = (GAMMA - 1.0) * (E - KE - ME)
        # float32 cancellation in E - KE - ME at high Mach; 0.3% tolerance is appropriate
        assert p_E == pytest.approx(p, rel=3e-3)

    def test_magnetic_energy_contribution(self) -> None:
        rho, p = 1.0, 1.0e2
        Br, Bz, Bt = 0.5, 0.3, 0.2
        ms = _mlx_state()
        s1 = _make_state(rho_val=rho, p_val=p, vr_val=0.0, vz_val=0.0,
                         vt_val=0.0, Br_val=0.0, Bz_val=0.0, Bt_val=0.0)
        s2 = _make_state(rho_val=rho, p_val=p, vr_val=0.0, vz_val=0.0,
                         vt_val=0.0, Br_val=Br, Bz_val=Bz, Bt_val=Bt)
        U1 = ms.from_state_dict(s1)
        U2 = ms.from_state_dict(s2)
        E1 = float(np.array(U1[IEN]).flat[0])
        E2 = float(np.array(U2[IEN]).flat[0])
        delta_E = E2 - E1
        ME_expected = 0.5 * (Br**2 + Bz**2 + Bt**2)
        assert delta_E == pytest.approx(ME_expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Stores last-packed state in self.U
# ---------------------------------------------------------------------------

class TestUAttribute:
    def test_U_updated_after_pack(self) -> None:
        ms = _mlx_state()
        state = _make_state(rho_val=7.0)
        U = ms.from_state_dict(state)
        stored = np.array(ms.U)
        returned = np.array(U)
        np.testing.assert_array_equal(stored, returned)
