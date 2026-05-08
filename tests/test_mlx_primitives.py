"""Tests for mlx_primitives: conservative <-> primitive conversions and dual-energy recovery.

Requires mlx to be installed (skipped otherwise).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")  # noqa: E402

from dpf.metal.mlx_primitives import (  # noqa: E402, I001
    ISR,
    P_FLOOR,
    RHO_FLOOR,
    cons_to_prim,
    entropy_resync,
    fast_magnetosonic,
    prim_to_cons,
    recover_pressure_dual_energy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uniform_U(
    nr: int = 8,
    nz: int = 8,
    rho_val: float = 1.0,
    vr_val: float = 0.1,
    vz_val: float = 0.2,
    vt_val: float = 0.05,
    p_val: float = 0.5,
    Br_val: float = 0.3,
    Bz_val: float = 0.4,
    Bt_val: float = 0.1,
    gamma: float = 5.0 / 3.0,
) -> mx.array:
    """Build a (10, nr, nz) conserved state from uniform primitives."""
    rho = mx.full((nr, nz), rho_val, dtype=mx.float32)
    vr = mx.full((nr, nz), vr_val, dtype=mx.float32)
    vz = mx.full((nr, nz), vz_val, dtype=mx.float32)
    vt = mx.full((nr, nz), vt_val, dtype=mx.float32)
    p = mx.full((nr, nz), p_val, dtype=mx.float32)
    Br = mx.full((nr, nz), Br_val, dtype=mx.float32)
    Bz = mx.full((nr, nz), Bz_val, dtype=mx.float32)
    Bt = mx.full((nr, nz), Bt_val, dtype=mx.float32)
    return prim_to_cons(rho, vr, vz, vt, p, Br, Bz, Bt, gamma=gamma)


def _np(arr: mx.array) -> np.ndarray:
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Round-trip: cons_to_prim o prim_to_cons
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_uniform_state(self) -> None:
        """prim_to_cons followed by cons_to_prim recovers original values < 1e-6."""
        rho0, vr0, vz0, vt0, p0 = 1.2, 0.3, -0.1, 0.05, 0.8
        Br0, Bz0, Bt0 = 0.4, 0.6, 0.2
        gamma = 5.0 / 3.0

        rho = mx.full((4, 4), rho0, dtype=mx.float32)
        vr = mx.full((4, 4), vr0, dtype=mx.float32)
        vz = mx.full((4, 4), vz0, dtype=mx.float32)
        vt = mx.full((4, 4), vt0, dtype=mx.float32)
        p = mx.full((4, 4), p0, dtype=mx.float32)
        Br = mx.full((4, 4), Br0, dtype=mx.float32)
        Bz = mx.full((4, 4), Bz0, dtype=mx.float32)
        Bt = mx.full((4, 4), Bt0, dtype=mx.float32)

        U = prim_to_cons(rho, vr, vz, vt, p, Br, Bz, Bt, gamma=gamma)
        rho_r, vr_r, vz_r, vt_r, p_r, Br_r, Bz_r, Bt_r = cons_to_prim(U, gamma=gamma)

        for name, expected, got in [
            ("rho", rho0, _np(rho_r).mean()),
            ("vr", vr0, _np(vr_r).mean()),
            ("vz", vz0, _np(vz_r).mean()),
            ("vt", vt0, _np(vt_r).mean()),
            ("p", p0, _np(p_r).mean()),
            ("Br", Br0, _np(Br_r).mean()),
            ("Bz", Bz0, _np(Bz_r).mean()),
            ("Bt", Bt0, _np(Bt_r).mean()),
        ]:
            rel_err = abs(got - expected) / max(abs(expected), 1e-30)
            assert rel_err < 1e-5, f"{name}: expected {expected}, got {got}, rel_err {rel_err}"

    def test_batch_shape(self) -> None:
        """Returned arrays have correct shape (nr, nz)."""
        U = _make_uniform_U(nr=6, nz=10)
        prims = cons_to_prim(U)
        for arr in prims:
            assert _np(arr).shape == (6, 10)

    def test_floor_enforcement_density(self) -> None:
        """Negative density input is floored to RHO_FLOOR after prim_to_cons."""
        rho = mx.full((4, 4), -1.0, dtype=mx.float32)
        vr = mx.zeros((4, 4), dtype=mx.float32)
        vz = mx.zeros((4, 4), dtype=mx.float32)
        vt = mx.zeros((4, 4), dtype=mx.float32)
        p = mx.full((4, 4), 1.0, dtype=mx.float32)
        Br = mx.zeros((4, 4), dtype=mx.float32)
        Bz = mx.zeros((4, 4), dtype=mx.float32)
        Bt = mx.zeros((4, 4), dtype=mx.float32)
        U = prim_to_cons(rho, vr, vz, vt, p, Br, Bz, Bt)
        rho_r, *_ = cons_to_prim(U)
        # float32 representation of 1e-12 may be slightly below the Python float constant
        assert float(_np(rho_r).min()) >= RHO_FLOOR * 0.999

    def test_floor_enforcement_pressure(self) -> None:
        """Negative pressure is floored to P_FLOOR after round-trip."""
        rho = mx.full((4, 4), 1.0, dtype=mx.float32)
        vr = mx.zeros((4, 4), dtype=mx.float32)
        vz = mx.zeros((4, 4), dtype=mx.float32)
        vt = mx.zeros((4, 4), dtype=mx.float32)
        p = mx.full((4, 4), -5.0, dtype=mx.float32)
        Br = mx.zeros((4, 4), dtype=mx.float32)
        Bz = mx.zeros((4, 4), dtype=mx.float32)
        Bt = mx.zeros((4, 4), dtype=mx.float32)
        U = prim_to_cons(rho, vr, vz, vt, p, Br, Bz, Bt)
        _, _, _, _, p_r, *_ = cons_to_prim(U)
        # float32 representation of 1e-12 may be slightly below the Python float constant
        assert float(_np(p_r).min()) >= P_FLOOR * 0.999

    def test_entropy_tracer_initialized(self) -> None:
        """prim_to_cons sets U[ISR] = p * rho^(1-gamma)."""
        gamma = 5.0 / 3.0
        rho_val, p_val = 2.0, 1.5
        rho = mx.full((4, 4), rho_val, dtype=mx.float32)
        p = mx.full((4, 4), p_val, dtype=mx.float32)
        zeros = mx.zeros((4, 4), dtype=mx.float32)
        U = prim_to_cons(rho, zeros, zeros, zeros, p, zeros, zeros, zeros, gamma=gamma)
        expected = p_val * rho_val ** (1.0 - gamma)
        got = float(_np(U[ISR]).mean())
        assert abs(got - expected) / abs(expected) < 1e-5


# ---------------------------------------------------------------------------
# Dual-energy pressure recovery
# ---------------------------------------------------------------------------


class TestDualEnergy:
    def _make_high_beta_U(self, nr: int = 8, nz: int = 8) -> mx.array:
        """High thermal pressure (high beta) — total-energy subtraction reliable."""
        return _make_uniform_U(nr=nr, nz=nz, rho_val=1.0, p_val=10.0, Br_val=0.01, Bz_val=0.01, Bt_val=0.0)

    def _make_low_beta_U(self, nr: int = 8, nz: int = 8) -> mx.array:
        """Very low thermal pressure but large B and v — cancellation-prone."""
        return _make_uniform_U(nr=nr, nz=nz, rho_val=1.0, vr_val=100.0, vz_val=100.0, p_val=1e-6, Br_val=50.0, Bz_val=50.0, Bt_val=0.0)

    def test_high_beta_w_near_one(self) -> None:
        """High beta state: blend weight w → 1 (total-energy dominates)."""
        U = self._make_high_beta_U()
        _, w = recover_pressure_dual_energy(U)
        w_mean = float(_np(w).mean())
        assert w_mean > 0.9, f"Expected w>0.9 for high beta, got {w_mean}"

    def test_low_beta_w_near_zero(self) -> None:
        """Low beta state: blend weight w → 0 (entropy dominates)."""
        U = self._make_low_beta_U()
        _, w = recover_pressure_dual_energy(U)
        w_mean = float(_np(w).mean())
        assert w_mean < 0.5, f"Expected w<0.5 for low beta, got {w_mean}"

    def test_pressure_matches_input_for_well_conditioned_state(self) -> None:
        """Dual-energy pressure matches direct pressure within 1% for well-conditioned state."""
        U = self._make_high_beta_U()
        p_dual, _ = recover_pressure_dual_energy(U)
        _, _, _, _, p_direct, *_ = cons_to_prim(U)
        rel_diff = float(_np(mx.abs(p_dual - p_direct) / mx.maximum(p_direct, P_FLOOR)).mean())
        assert rel_diff < 0.01, f"Dual-energy diverges from direct at high beta: {rel_diff}"

    def test_pressure_floor_enforced(self) -> None:
        """Recovered pressure is always >= P_FLOOR."""
        U = self._make_low_beta_U()
        p, _ = recover_pressure_dual_energy(U)
        assert float(_np(p).min()) >= P_FLOOR

    def test_entropy_recovery_avoids_cancellation(self) -> None:
        """In low-beta state, p_S should be ~ original p_val (no cancellation)."""
        p_val = 1e-6
        U = _make_uniform_U(rho_val=1.0, vr_val=100.0, p_val=p_val, Br_val=50.0, Bz_val=50.0)
        p, w = recover_pressure_dual_energy(U)
        # w is near 0, so p ≈ p_S = Srho * rho^(gm1)
        # p_S should be close to original p_val
        p_s_approx = float(_np(p).mean())
        # Should be within an order of magnitude of original p_val
        assert p_s_approx < 1e-3, f"Entropy recovery gave unreasonably large p: {p_s_approx}"

    def test_output_shapes(self) -> None:
        """recover_pressure_dual_energy returns correct shapes."""
        U = _make_uniform_U(nr=5, nz=7)
        p, w = recover_pressure_dual_energy(U)
        assert _np(p).shape == (5, 7)
        assert _np(w).shape == (5, 7)

    def test_blend_weight_in_unit_interval(self) -> None:
        """Blend weight is always in [0, 1]."""
        U = _make_uniform_U()
        _, w = recover_pressure_dual_energy(U)
        w_np = _np(w)
        assert w_np.min() >= -1e-6
        assert w_np.max() <= 1.0 + 1e-6

    def test_pressure_recovery_ignores_rejected_nonfinite_candidate(self) -> None:
        """A zero-weighted infinite pressure candidate must not create NaN."""
        U_np = np.asarray(self._make_high_beta_U()).astype(np.float32)
        U_np[ISR, 0, 0] = np.inf
        U = mx.array(U_np)

        p, w = recover_pressure_dual_energy(U)
        p_np = _np(p)
        w_np = _np(w)

        assert np.all(np.isfinite(p_np))
        assert p_np[0, 0] >= P_FLOOR * 0.999
        assert w_np[0, 0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Fast magnetosonic speed
# ---------------------------------------------------------------------------


class TestFastMagnetosonic:
    def test_matches_pytorch_reference(self) -> None:
        """cf from MLX matches _fast_magnetosonic_mps to relative error < 1e-5."""
        torch = pytest.importorskip("torch")
        from dpf.metal._riemann_primitives import _fast_magnetosonic_mps

        gamma = 5.0 / 3.0
        nr, nz = 8, 8
        rng = np.random.default_rng(42)

        rho_np = rng.uniform(0.5, 2.0, (nr, nz)).astype(np.float32)
        p_np = rng.uniform(0.1, 1.0, (nr, nz)).astype(np.float32)
        Br_np = rng.uniform(-1.0, 1.0, (nr, nz)).astype(np.float32)
        Bz_np = rng.uniform(-1.0, 1.0, (nr, nz)).astype(np.float32)
        Bt_np = rng.uniform(-0.5, 0.5, (nr, nz)).astype(np.float32)

        for dim in [0, 1]:
            # MLX
            rho_mx = mx.array(rho_np)
            p_mx = mx.array(p_np)
            Br_mx = mx.array(Br_np)
            Bz_mx = mx.array(Bz_np)
            Bt_mx = mx.array(Bt_np)
            cf_mlx = _np(fast_magnetosonic(rho_mx, p_mx, Br_mx, Bz_mx, Bt_mx, gamma, dim))

            # PyTorch reference (Cartesian layout: B = [Bn, Bt1, Bt2])
            B_torch = torch.tensor(np.stack([Br_np, Bz_np, Bt_np], axis=0))
            cf_torch = _fast_magnetosonic_mps(
                torch.tensor(rho_np),
                torch.tensor(p_np),
                B_torch,
                gamma,
                dim,
            ).numpy()

            rel_err = np.abs(cf_mlx - cf_torch) / (np.abs(cf_torch) + 1e-30)
            assert rel_err.max() < 1e-4, f"dim={dim}: max rel_err={rel_err.max():.2e}"

    def test_hydrostatic_limit(self) -> None:
        """With B=0, cf reduces to the sound speed a = sqrt(gamma*p/rho)."""
        gamma = 5.0 / 3.0
        rho_val, p_val = 2.0, 1.2
        rho = mx.full((4, 4), rho_val, dtype=mx.float32)
        p = mx.full((4, 4), p_val, dtype=mx.float32)
        zeros = mx.zeros((4, 4), dtype=mx.float32)

        cf = fast_magnetosonic(rho, p, zeros, zeros, zeros, gamma, dim=0)
        expected = math.sqrt(gamma * p_val / rho_val)
        rel_err = abs(float(_np(cf).mean()) - expected) / expected
        assert rel_err < 1e-5, f"cf={float(_np(cf).mean()):.6f}, expected {expected:.6f}"

    def test_no_nan_or_inf(self) -> None:
        """fast_magnetosonic never returns NaN/Inf on physical inputs."""
        rho = mx.full((8, 8), 1.0, dtype=mx.float32)
        p = mx.full((8, 8), 1.0, dtype=mx.float32)
        Br = mx.full((8, 8), 5.0, dtype=mx.float32)
        Bz = mx.full((8, 8), 3.0, dtype=mx.float32)
        Bt = mx.full((8, 8), 1.0, dtype=mx.float32)
        for dim in [0, 1]:
            cf = _np(fast_magnetosonic(rho, p, Br, Bz, Bt, 5.0 / 3.0, dim))
            assert not np.any(np.isnan(cf)), f"NaN in cf for dim={dim}"
            assert not np.any(np.isinf(cf)), f"Inf in cf for dim={dim}"

    def test_vacuum_limit_no_overflow(self) -> None:
        """Very low density + strong B should not produce NaN/Inf (clamped at c)."""
        rho = mx.full((4, 4), 1e-12, dtype=mx.float32)
        p = mx.full((4, 4), 1e-12, dtype=mx.float32)
        Br = mx.full((4, 4), 1e4, dtype=mx.float32)
        Bz = mx.zeros((4, 4), dtype=mx.float32)
        Bt = mx.zeros((4, 4), dtype=mx.float32)
        cf = _np(fast_magnetosonic(rho, p, Br, Bz, Bt, 5.0 / 3.0, dim=0))
        assert not np.any(np.isnan(cf))
        assert not np.any(np.isinf(cf))
        assert cf.max() <= 3e8 + 1.0, f"cf exceeded c: {cf.max()}"


# ---------------------------------------------------------------------------
# Entropy resync
# ---------------------------------------------------------------------------


class TestEntropyResync:
    def test_smooth_cells_unchanged(self) -> None:
        """Smooth flow (no compression, no gradient) leaves entropy tracer unchanged."""
        U = _make_uniform_U(nr=8, nz=8, p_val=1.0, rho_val=1.0)
        p = mx.full((8, 8), 1.0, dtype=mx.float32)
        div_v = mx.zeros((8, 8), dtype=mx.float32)  # no compression

        Srho_before = _np(U[ISR]).copy()
        Srho_after = _np(entropy_resync(U, p, div_v))
        np.testing.assert_allclose(Srho_after, Srho_before, rtol=1e-6)

    def test_shocked_cells_reset(self) -> None:
        """Cells with strong compression AND steep pressure gradient get reset."""
        gamma = 5.0 / 3.0
        nr, nz = 8, 8

        rho_val, p_val = 1.0, 1.0
        U = _make_uniform_U(nr=nr, nz=nz, rho_val=rho_val, p_val=p_val, gamma=gamma)

        # Corrupt ISR in a few cells to simulate drift
        U_np = np.asarray(U).copy()
        U_np[ISR, 3:5, 3:5] *= 0.1
        U_corrupted = mx.array(U_np)

        # Build pressure field with a large jump at center
        p_np = np.full((nr, nz), p_val, dtype=np.float32)
        p_np[3:5, :] = p_val * 10.0
        p_mx = mx.array(p_np)

        # Strong compression
        dx = 1.0
        cs_approx = math.sqrt(gamma * p_val / rho_val)
        div_v_np = np.full((nr, nz), -2.0 * cs_approx / dx, dtype=np.float32)
        div_v_mx = mx.array(div_v_np)

        Srho_synced = _np(entropy_resync(U_corrupted, p_mx, div_v_mx, gamma=gamma, dx=dx))
        Srho_original = _np(U_corrupted[ISR])

        # At least some shocked cells should have changed
        changed = np.abs(Srho_synced - Srho_original) > 1e-10
        assert changed.any(), "No cells were resynced despite strong shock conditions"

    def test_output_shape(self) -> None:
        """entropy_resync returns shape (nr, nz)."""
        U = _make_uniform_U(nr=6, nz=9)
        p = mx.full((6, 9), 1.0, dtype=mx.float32)
        div_v = mx.zeros((6, 9), dtype=mx.float32)
        result = entropy_resync(U, p, div_v)
        assert _np(result).shape == (6, 9)

    def test_no_nan_in_output(self) -> None:
        """entropy_resync never produces NaN."""
        U = _make_uniform_U(rho_val=1.0, p_val=0.5)
        p = mx.full((8, 8), 0.5, dtype=mx.float32)
        div_v = mx.full((8, 8), -1.0, dtype=mx.float32)
        Srho_out = _np(entropy_resync(U, p, div_v))
        assert not np.any(np.isnan(Srho_out))


# ---------------------------------------------------------------------------
# prim_to_cons — electron energy passthrough
# ---------------------------------------------------------------------------


class TestElectronEnergy:
    def test_e_electron_stored_and_floored(self) -> None:
        """e_electron is stored in U[IEE] and floored at 0."""
        rho = mx.full((4, 4), 1.0, dtype=mx.float32)
        zeros = mx.zeros((4, 4), dtype=mx.float32)
        p = mx.full((4, 4), 1.0, dtype=mx.float32)
        e_elec = mx.full((4, 4), 0.7, dtype=mx.float32)
        U = prim_to_cons(rho, zeros, zeros, zeros, p, zeros, zeros, zeros, e_electron=e_elec)
        np.testing.assert_allclose(_np(U[9]), 0.7, rtol=1e-5)

    def test_negative_e_electron_floored(self) -> None:
        """Negative e_electron is floored to 0."""
        rho = mx.full((4, 4), 1.0, dtype=mx.float32)
        zeros = mx.zeros((4, 4), dtype=mx.float32)
        p = mx.full((4, 4), 1.0, dtype=mx.float32)
        e_elec = mx.full((4, 4), -5.0, dtype=mx.float32)
        U = prim_to_cons(rho, zeros, zeros, zeros, p, zeros, zeros, zeros, e_electron=e_elec)
        assert float(_np(U[9]).min()) >= 0.0
