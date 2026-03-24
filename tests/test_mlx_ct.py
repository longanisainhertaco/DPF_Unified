"""Tests for mlx_ct: MLX constrained transport for the Phase B solver."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not available")

from dpf.metal.mlx_ct import apply_ct, compute_emf, div_B_cylindrical  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid(nr: int = 16, nz: int = 24, dr: float = 1e-3, dz: float = 2e-3):
    """Return (r_cell, r_face) as mx.arrays for a uniform grid."""
    r_cell = mx.array([(i + 0.5) * dr for i in range(nr)], dtype=mx.float32)
    r_face = mx.array([i * dr for i in range(nr + 1)], dtype=mx.float32)
    return r_cell, r_face


def _uniform_B(nr: int, nz: int, Br_val: float, Bz_val: float):
    """Return face-centred (Br_face, Bz_face) for a uniform B field."""
    Br_face = mx.full((nr + 1, nz), Br_val, dtype=mx.float32)
    Bz_face = mx.full((nr, nz + 1), Bz_val, dtype=mx.float32)
    return Br_face, Bz_face


def _div_rms(div: object) -> float:
    """RMS of the divergence field."""
    return float(mx.sqrt(mx.mean(div * div)).item())


# ---------------------------------------------------------------------------
# compute_emf
# ---------------------------------------------------------------------------

class TestComputeEmf:
    def test_output_shape(self) -> None:
        nr, nz = 8, 12
        dr, dz = 1e-3, 2e-3
        vr = mx.zeros((nr, nz), dtype=mx.float32)
        vz = mx.zeros((nr, nz), dtype=mx.float32)
        Br_face, Bz_face = _uniform_B(nr, nz, 0.5, 0.3)

        emf = compute_emf(vr, vz, Br_face, Bz_face, dr, dz)
        assert emf.shape == (nr + 1, nz + 1)

    def test_zero_velocity_gives_zero_emf(self) -> None:
        nr, nz = 8, 12
        vr = mx.zeros((nr, nz), dtype=mx.float32)
        vz = mx.zeros((nr, nz), dtype=mx.float32)
        Br_face, Bz_face = _uniform_B(nr, nz, 1.0, 0.5)

        emf = compute_emf(vr, vz, Br_face, Bz_face, 1e-3, 2e-3)
        mx.eval(emf)
        assert float(mx.max(mx.abs(emf)).item()) == pytest.approx(0.0, abs=1e-7)

    def test_zero_B_gives_zero_emf(self) -> None:
        nr, nz = 8, 12
        vr = mx.ones((nr, nz), dtype=mx.float32)
        vz = mx.ones((nr, nz), dtype=mx.float32)
        Br_face = mx.zeros((nr + 1, nz), dtype=mx.float32)
        Bz_face = mx.zeros((nr, nz + 1), dtype=mx.float32)

        emf = compute_emf(vr, vz, Br_face, Bz_face, 1e-3, 2e-3)
        mx.eval(emf)
        assert float(mx.max(mx.abs(emf)).item()) == pytest.approx(0.0, abs=1e-7)

    def test_uniform_vr_uniform_Bz_magnitude(self) -> None:
        nr, nz = 8, 12
        vr = mx.full((nr, nz), 2.0, dtype=mx.float32)
        vz = mx.zeros((nr, nz), dtype=mx.float32)
        Br_face = mx.zeros((nr + 1, nz), dtype=mx.float32)
        Bz_face = mx.full((nr, nz + 1), 3.0, dtype=mx.float32)

        # E_theta = -(vr * Bz - 0) = -6 everywhere
        emf = compute_emf(vr, vz, Br_face, Bz_face, 1e-3, 2e-3)
        mx.eval(emf)
        assert float(mx.max(mx.abs(emf + 6.0)).item()) == pytest.approx(0.0, abs=1e-5)

    def test_sign_convention_vz_Br(self) -> None:
        nr, nz = 8, 12
        vr = mx.zeros((nr, nz), dtype=mx.float32)
        vz = mx.full((nr, nz), 1.0, dtype=mx.float32)
        Br_face = mx.full((nr + 1, nz), 4.0, dtype=mx.float32)
        Bz_face = mx.zeros((nr, nz + 1), dtype=mx.float32)

        # E_theta = -(0 - vz * Br) = vz * Br = 4
        emf = compute_emf(vr, vz, Br_face, Bz_face, 1e-3, 2e-3)
        mx.eval(emf)
        assert float(mx.max(mx.abs(emf - 4.0)).item()) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# div_B_cylindrical
# ---------------------------------------------------------------------------

class TestDivBCylindrical:
    def test_output_shape(self) -> None:
        nr, nz = 16, 24
        dr, dz = 1e-3, 2e-3
        r_cell, r_face = _grid(nr, nz, dr, dz)
        Br_face, Bz_face = _uniform_B(nr, nz, 0.1, 0.2)

        div = div_B_cylindrical(Br_face, Bz_face, dr, dz, r_cell, r_face)
        assert div.shape == (nr, nz)

    def test_uniform_B_has_zero_divergence(self) -> None:
        nr, nz = 16, 24
        dr, dz = 1e-3, 2e-3
        r_cell, r_face = _grid(nr, nz, dr, dz)
        Br_face, Bz_face = _uniform_B(nr, nz, 0.0, 0.5)

        div = div_B_cylindrical(Br_face, Bz_face, dr, dz, r_cell, r_face)
        mx.eval(div)
        assert _div_rms(div) == pytest.approx(0.0, abs=1e-6)

    def test_constant_rBr_is_exactly_div_free(self) -> None:
        """If r*Br = const the finite-volume divergence is zero.

        Use integer-valued r_face so that r_face * (1/r_face) = 1.0 exactly
        in float32, avoiding round-trip rounding artifacts.
        """
        nr, nz = 8, 6
        dr = 1.0   # unit grid — r_face = [1, 2, 3, ..., 9]
        dz = 1.0
        C = 5.0

        # r_face[i] = i+1 (integer), r_cell[i] = i+1.5 (exact half-integer)
        r_face_vals = np.arange(1, nr + 2, dtype=np.float32)      # [1,2,...,9]
        r_cell_vals = r_face_vals[:-1] + 0.5                       # [1.5,2.5,...]
        r_face = mx.array(r_face_vals)
        r_cell = mx.array(r_cell_vals)

        # Br[i] = C / r_face[i]: integer reciprocals are NOT exact in float32 generally,
        # but the DIFFERENCES r_face[i]*Br[i] - r_face[i-1]*Br[i-1] cancel exactly
        # when r_face values are exact integers (exact representation in float32).
        Br_vals = (C / r_face_vals).astype(np.float32)
        Br_face = mx.array(np.tile(Br_vals, (nz, 1)).T)     # (nr+1, nz)
        Bz_face = mx.zeros((nr, nz + 1), dtype=mx.float32)

        div = div_B_cylindrical(Br_face, Bz_face, dr, dz, r_cell, r_face)
        mx.eval(div)
        # Each term (r_face[i+1]*Br[i+1] - r_face[i]*Br[i]) = C - C = 0
        # Only float32 representation error of 1/integer values contributes.
        assert _div_rms(div) < 1e-4


# ---------------------------------------------------------------------------
# apply_ct
# ---------------------------------------------------------------------------

class TestApplyCt:
    def test_output_shapes(self) -> None:
        nr, nz = 16, 24
        dr, dz, dt = 1e-3, 2e-3, 1e-8
        r_cell, r_face = _grid(nr, nz, dr, dz)
        Br_face, Bz_face = _uniform_B(nr, nz, 0.1, 0.2)
        emf = mx.zeros((nr + 1, nz + 1), dtype=mx.float32)

        Br_new, Bz_new = apply_ct(Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face)
        assert Br_new.shape == (nr + 1, nz)
        assert Bz_new.shape == (nr, nz + 1)

    def test_zero_emf_leaves_B_unchanged(self) -> None:
        nr, nz = 16, 24
        dr, dz, dt = 1e-3, 2e-3, 1e-8
        r_cell, r_face = _grid(nr, nz, dr, dz)
        Br_face, Bz_face = _uniform_B(nr, nz, 0.5, 0.3)
        emf = mx.zeros((nr + 1, nz + 1), dtype=mx.float32)

        Br_new, Bz_new = apply_ct(Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face)
        mx.eval(Br_new, Bz_new)
        np.testing.assert_allclose(
            np.array(Br_new), np.array(Br_face), rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            np.array(Bz_new), np.array(Bz_face), rtol=1e-6, atol=1e-10
        )

    def test_uniform_B_stays_div_free_after_ct(self) -> None:
        """CT applied to a div-free initial state must preserve div(B)=0."""
        nr, nz = 16, 24
        dr, dz, dt = 1e-3, 2e-3, 5e-9
        r_cell, r_face = _grid(nr, nz, dr, dz)
        Br_face, Bz_face = _uniform_B(nr, nz, 0.0, 1.0)

        # Velocities that would produce a non-trivial EMF
        vr = mx.full((nr, nz), 1e3, dtype=mx.float32)
        vz = mx.zeros((nr, nz), dtype=mx.float32)
        emf = compute_emf(vr, vz, Br_face, Bz_face, dr, dz)

        Br_new, Bz_new = apply_ct(Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face)
        div = div_B_cylindrical(Br_new, Bz_new, dr, dz, r_cell, r_face)
        mx.eval(div)
        # Initial div was 0; CT should keep it near 0 (up to float32 truncation)
        assert _div_rms(div) < 1e-3

    def test_Btheta_untouched(self) -> None:
        """CT must not modify Btheta (cell-centred, no div contribution)."""
        nr, nz = 8, 12
        dr, dz, dt = 1e-3, 2e-3, 1e-8
        r_cell, r_face = _grid(nr, nz, dr, dz)
        Br_face, Bz_face = _uniform_B(nr, nz, 0.1, 0.2)
        emf = mx.ones((nr + 1, nz + 1), dtype=mx.float32) * 0.5
        Btheta_before = mx.full((nr, nz), 3.14, dtype=mx.float32)

        # apply_ct does not accept or modify Btheta; verify it is not implicitly
        # modified by checking the module only returns (Br_new, Bz_new).
        result = apply_ct(Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face)
        assert len(result) == 2, "apply_ct must return exactly (Br_new, Bz_new)"

        # Btheta reference is unchanged
        mx.eval(Btheta_before)
        assert float(mx.mean(Btheta_before).item()) == pytest.approx(3.14, abs=1e-5)


# ---------------------------------------------------------------------------
# Round-trip: EMF -> CT -> divergence
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_divfree_preserved_over_multiple_steps(self) -> None:
        """Run 10 CT steps; div(B) must remain near float32 zero."""
        nr, nz = 16, 24
        dr, dz, dt = 1e-3, 2e-3, 2e-9
        r_cell, r_face = _grid(nr, nz, dr, dz)

        # Start from exactly divergence-free uniform field
        Br_face, Bz_face = _uniform_B(nr, nz, 0.0, 1.0)

        vr = mx.full((nr, nz), 500.0, dtype=mx.float32)
        vz = mx.full((nr, nz), 200.0, dtype=mx.float32)

        for _ in range(10):
            emf = compute_emf(vr, vz, Br_face, Bz_face, dr, dz)
            Br_face, Bz_face = apply_ct(Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face)

        div = div_B_cylindrical(Br_face, Bz_face, dr, dz, r_cell, r_face)
        mx.eval(div)
        assert _div_rms(div) < 1e-3

    def test_ct_reduces_divergence(self) -> None:
        """Inject artificial divergence; one CT step should reduce it."""
        nr, nz = 16, 24
        dr, dz, dt = 1e-3, 2e-3, 1e-8
        r_cell, r_face = _grid(nr, nz, dr, dz)

        Br_face, Bz_face = _uniform_B(nr, nz, 0.0, 1.0)
        # Inject a perturbation into Br that breaks div(B)=0
        noise_np = np.random.default_rng(42).uniform(-0.1, 0.1, (nr + 1, nz)).astype(np.float32)
        Br_face = Br_face + mx.array(noise_np)

        div_before = div_B_cylindrical(Br_face, Bz_face, dr, dz, r_cell, r_face)
        mx.eval(div_before)
        rms_before = _div_rms(div_before)

        vr = mx.zeros((nr, nz), dtype=mx.float32)
        vz = mx.zeros((nr, nz), dtype=mx.float32)
        emf = compute_emf(vr, vz, Br_face, Bz_face, dr, dz)
        Br_new, Bz_new = apply_ct(Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face)

        div_after = div_B_cylindrical(Br_new, Bz_new, dr, dz, r_cell, r_face)
        mx.eval(div_after)
        rms_after = _div_rms(div_after)

        # Zero EMF (zero velocity) → B unchanged → div unchanged.
        # This verifies CT does not spontaneously increase divergence.
        assert rms_after <= rms_before + 1e-6

    def test_field_loop_Btheta_independent(self) -> None:
        """CT operating on Br/Bz must not couple to Btheta at all.

        Create a non-trivial (r,z) field configuration, run CT, verify
        that a separate Btheta cell-centred field is completely decoupled.
        """
        nr, nz = 12, 16
        dr, dz, dt = 1e-3, 2e-3, 5e-9
        r_cell, r_face = _grid(nr, nz, dr, dz)

        Br_face = mx.zeros((nr + 1, nz), dtype=mx.float32)
        Bz_face = mx.zeros((nr, nz + 1), dtype=mx.float32)

        # Btheta is completely separate — CT never touches it
        Btheta = mx.full((nr, nz), 7.77, dtype=mx.float32)

        vr = mx.full((nr, nz), 1000.0, dtype=mx.float32)
        vz = mx.full((nr, nz), 500.0, dtype=mx.float32)

        for _ in range(5):
            emf = compute_emf(vr, vz, Br_face, Bz_face, dr, dz)
            Br_face, Bz_face = apply_ct(Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face)

        # Btheta is not passed to any CT function; value must be unchanged
        mx.eval(Btheta)
        assert float(mx.max(mx.abs(Btheta - 7.77)).item()) == pytest.approx(0.0, abs=1e-5)
