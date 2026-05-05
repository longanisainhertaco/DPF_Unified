"""Pinch convergence sweep: toh_psi_ni vs minmod.

Compares L1 errors in rho, P, B for CylindricalMHDSolver with
use_toh_limiter=False (minmod) vs use_toh_limiter=True (Toh 2025 psi(n_i))
over N = [32, 64, 128, 256] on a 1D axial smooth-wave problem.

Problem:
    Smooth sinusoidal density/pressure perturbation propagating axially in
    a magnetized slab. Initial conditions admit an analytical reference at
    t=0 (frozen); we measure the L1 reconstruction error directly from
    a single `_plm_reconstruct` call rather than time-evolving, so we test
    *limiter* accuracy rather than time integrator accuracy.

    For the near-vacuum (low-density) regime we also run a step-function
    profile (rho_low << rho_high) to confirm Toh damps oscillations that
    minmod would otherwise produce from interface diffusion artifacts.

References:
    Toh 2025, KnowledgeReference/asymptotic-preserving-semi-implicit-
    finite-volume-scheme-for-extended-magnetohydrodynamics-yi-han.md
    §3.1 Eq.31 (ψ(n_i) = (1 + exp((λ₀/Δx)³ - n_i/n_ref))^(-1))

    LeVeque 2002 §6.5 — PLM slope limiter convergence theory.
    Minmod is 1st-order at smooth peaks, 2nd-order elsewhere.
    MC is 2nd-order everywhere (but not in this solver; this solver uses
    minmod). Toh should preserve minmod's order in smooth+dense regions
    while damping oscillations in the low-density limit.
"""
from __future__ import annotations

import os
import sys

# Ensure the Toh worktree src is on the path so CylindricalMHDSolver has the
# use_toh_limiter kwarg.  The worktree lives at a fixed path set by the agent.
_WORKTREE = (
    "/Users/anthonyzamora/dpf-unified/.claude/worktrees/agent-a92314a9ee378475a"
)
if os.path.isdir(_WORKTREE):
    _wt_src = os.path.join(_WORKTREE, "src")
    if _wt_src not in sys.path:
        sys.path.insert(0, _wt_src)

import math  # noqa: E402

import numpy as np  # noqa: E402

from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver  # noqa: E402

# ---------------------------------------------------------------------------
# Helper — build a minimal solver with the right grid
# ---------------------------------------------------------------------------

def _make_solver(nz: int, use_toh: bool) -> CylindricalMHDSolver:
    """1D-axial solver: minimal nr=4, varying nz."""
    dz = 1.0 / nz
    return CylindricalMHDSolver(
        nr=4,
        nz=nz,
        dr=1e-3,
        dz=dz,
        gamma=5.0 / 3.0,
        cfl=0.4,
        enable_hall=False,
        enable_resistive=False,
        enable_energy_equation=True,
        use_godunov_flux=True,
        use_toh_limiter=use_toh,
        toh_lambda_dx_cubed=0.005,
        toh_n_ref=None,  # auto from field max
    )


# ---------------------------------------------------------------------------
# Problem 1 — smooth sinusoidal wave (should be ~1st-order with minmod,
#             Toh should match or exceed)
# ---------------------------------------------------------------------------

def _smooth_wave(nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth 1-mode sinusoidal density on [0, 1].

    rho(z) = 1 + 0.3 * sin(2*pi*z)
    p(z)   = 0.5 + 0.1 * sin(2*pi*z)
    B_z(z) = 0.2 + 0.05 * sin(2*pi*z)

    All fields are positive and well above the vacuum floor.
    Returns (rho, p, Bz) each of shape (nz,).
    """
    z = (np.arange(nz) + 0.5) / nz  # cell centres
    rho = 1.0 + 0.3 * np.sin(2.0 * math.pi * z)
    p   = 0.5 + 0.1 * np.sin(2.0 * math.pi * z)
    bz  = 0.2 + 0.05 * np.sin(2.0 * math.pi * z)
    return rho, p, bz


def _l1_reconstruction_error(
    solver: CylindricalMHDSolver,
    q: np.ndarray,
    rho: np.ndarray,
) -> float:
    """L1 error of PLM reconstruction vs cell-centre exact value.

    For a smooth periodic function, the reconstruction QL[i+1/2] should
    converge to q[i] at rate ~O(h). We measure the error as:

        E_L1 = (1/N) * sum_i |QL[i] - q[i-1]|

    by comparing the reconstructed left-face value QL at interface i+1/2
    to the cell-centre value q[i] (the left cell).  This measures the
    reconstruction quality directly without time-evolution contamination.

    The *analytical* reference is the cell-centre value itself, not a
    derived expected value — this is a self-consistency convergence
    metric, not an external truth comparison.
    """
    # q and rho shaped (4, nz) for axis=1 reconstruction
    # Expand to (4, nz) — solver is nr=4
    nr = solver.nr
    q2d  = np.tile(q[np.newaxis, :], (nr, 1))   # (nr, nz)
    rho2d = np.tile(rho[np.newaxis, :], (nr, 1)) # (nr, nz)

    ql, qr = solver._plm_reconstruct(q2d, axis=1, density=rho2d)
    # ql[i, j] = left-face value at interface j+1/2 = reconstruction of cell j
    # Error: ql[i, j] vs q2d[i, j] (original cell centre)
    # Both have shape (nr, nz-1); compare to q2d[:, :-1]
    err = np.mean(np.abs(ql - q2d[:, :-1]))
    return float(err)


def _run_smooth_sweep(ns: list[int], use_toh: bool) -> dict[str, list[float]]:
    """Return L1 errors for rho, p, Bz at each resolution."""
    errs: dict[str, list[float]] = {"rho": [], "p": [], "Bz": []}
    for n in ns:
        solver = _make_solver(n, use_toh)
        rho, p, bz = _smooth_wave(n)
        errs["rho"].append(_l1_reconstruction_error(solver, rho, rho))
        errs["p"].append(_l1_reconstruction_error(solver, p, rho))
        errs["Bz"].append(_l1_reconstruction_error(solver, bz, rho))
    return errs


# ---------------------------------------------------------------------------
# Problem 2 — low-density step (Toh should reduce oscillations vs minmod)
# ---------------------------------------------------------------------------

def _low_density_step(nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step function representing a low-density vacuum-edge region.

    Left half: rho = 1e-4 (near-vacuum)
    Right half: rho = 1.0  (bulk)

    Toh psi(n_i) with n_ref = max(rho) = 1.0:
      - Left cells: psi ≈ 0.499  (reduces slope by ~50%)
      - Right cells: psi ≈ 1.0   (full minmod slope)

    Minmod at a sharp step returns zero slope (already TVD), so the
    *slope* test is degenerate for a perfect step. Instead we test on a
    smoothed step (tanh profile) so minmod CAN produce non-zero slopes
    in the transition zone, and Toh additionally damps them in low-rho.
    """
    z = (np.arange(nz) + 0.5) / nz
    width = 4.0 / nz   # ~4 cells wide
    rho = 1e-4 + (1.0 - 1e-4) * 0.5 * (1.0 + np.tanh((z - 0.5) / width))
    p   = 1e-5 + (0.5 - 1e-5) * 0.5 * (1.0 + np.tanh((z - 0.5) / width))
    bz  = 0.01 + (0.2 - 0.01) * 0.5 * (1.0 + np.tanh((z - 0.5) / width))
    return rho, p, bz


def _total_slope_norm(
    solver: CylindricalMHDSolver,
    q: np.ndarray,
    rho: np.ndarray,
) -> float:
    """Sum of |slope[i]| across all cells (internal).

    The Toh limiter multiplies slope[i] by ψ(n_i) <= 1.  To measure this
    directly we compute slope = 2*(QL[i] - q[i]) for interior cells, since
    QL[i+1/2] = q[i] + 0.5*slope[i].  Smaller total = more limiter damping.
    """
    nr = solver.nr
    q2d   = np.tile(q[np.newaxis, :], (nr, 1))
    rho2d = np.tile(rho[np.newaxis, :], (nr, 1))
    ql, _ = solver._plm_reconstruct(q2d, axis=1, density=rho2d)
    # ql has shape (nr, nz-1); ql[:, i] = q2d[:, i] + 0.5 * slope[:, i]
    # => slope[:, i] = 2 * (ql[:, i] - q2d[:, i]) for i in 0..nz-2
    slope_est = 2.0 * np.abs(ql - q2d[:, :-1])
    return float(np.sum(slope_est))


# ---------------------------------------------------------------------------
# Convergence rate computation
# ---------------------------------------------------------------------------

def _convergence_rate(err1: float, err2: float, n1: int, n2: int) -> float:
    """Order p from err ~ h^p: p = log(e1/e2) / log(n2/n1)."""
    if err1 <= 0 or err2 <= 0:
        return float("nan")
    return math.log(err1 / err2) / math.log(n2 / n1)


# ===========================================================================
# Tests
# ===========================================================================

NS = [32, 64, 128, 256]


class TestSmoothWaveConvergence:
    """Smooth periodic wave — both methods should converge, rates compared."""

    def _errors_and_rates(self, use_toh: bool):
        errs = _run_smooth_sweep(NS, use_toh)
        rates: dict[str, list[float]] = {k: [] for k in errs}
        for k, e_list in errs.items():
            for i in range(1, len(NS)):
                rates[k].append(_convergence_rate(e_list[i - 1], e_list[i], NS[i - 1], NS[i]))
        return errs, rates

    def test_minmod_converges(self):
        """Minmod must achieve >= 1st-order (rate >= 0.8) on smooth data."""
        _, rates = self._errors_and_rates(use_toh=False)
        for field, r_list in rates.items():
            for r in r_list:
                assert r >= 0.8, (
                    f"minmod field={field} converge rate {r:.2f} < 0.8 — "
                    "PLM baseline is broken"
                )

    def test_toh_converges(self):
        """Toh limiter must achieve >= 1st-order on smooth high-density data."""
        _, rates = self._errors_and_rates(use_toh=True)
        for field, r_list in rates.items():
            for r in r_list:
                assert r >= 0.8, (
                    f"toh_psi_ni field={field} converge rate {r:.2f} < 0.8"
                )

    def test_toh_not_worse_than_minmod(self):
        """Toh L1 errors on smooth data should be <= 2x minmod errors.

        On smooth high-density data psi → 1 everywhere, so Toh reverts to
        plain minmod. Errors should be near-identical (within float noise).
        Tolerance 2x guards against accidental degradation.
        """
        errs_minmod = _run_smooth_sweep(NS, use_toh=False)
        errs_toh    = _run_smooth_sweep(NS, use_toh=True)
        for field in ("rho", "p", "Bz"):
            for n_idx, n in enumerate(NS):
                em = errs_minmod[field][n_idx]
                et = errs_toh[field][n_idx]
                assert et <= 2.0 * em + 1e-15, (
                    f"toh worse than minmod: field={field} N={n} "
                    f"toh={et:.3e} minmod={em:.3e}"
                )


class TestLowDensityStep:
    """Near-vacuum step — Toh should produce smaller total variation."""

    def test_toh_reduces_tv_in_low_density(self):
        """Toh ψ < 1 in near-vacuum; total reconstruction TV must be smaller.

        With the tanh step, minmod produces non-zero slopes in the transition
        zone. Toh multiplies slopes by ψ(n_i) < 1 for low-rho cells,
        so the total variation of QL-QR across interfaces must be strictly
        smaller with Toh on.
        """
        for n in NS:
            rho, p, bz = _low_density_step(n)
            solver_off = _make_solver(n, use_toh=False)
            solver_on  = _make_solver(n, use_toh=True)
            tv_off = _total_slope_norm(solver_off, rho, rho)
            tv_on  = _total_slope_norm(solver_on,  rho, rho)
            assert tv_on < tv_off, (
                f"N={n}: Toh TV={tv_on:.4e} >= minmod TV={tv_off:.4e} — "
                "limiter has no effect in low-density region"
            )

    def test_toh_damps_less_in_high_density_than_low_density(self):
        """Toh ψ is higher in the bulk (high-rho) than in the vacuum (low-rho).

        With n_ref = max(rho), n_i/n_ref is larger in the bulk, so the
        sigmoid argument (alpha - n_i/n_ref) is more negative, giving
        larger ψ.  We verify: mean ψ in bulk > mean ψ in vacuum region.
        """
        n = 64
        rho, _, _ = _low_density_step(n)
        solver_on = _make_solver(n, use_toh=True)
        rho2d = np.tile(rho[np.newaxis, :], (solver_on.nr, 1))

        psi = solver_on._toh_psi(rho2d)  # shape (nr, nz)
        vacuum_half = psi[:, : n // 2]
        bulk_half   = psi[:, n // 2 :]
        psi_vacuum_mean = float(np.mean(vacuum_half))
        psi_bulk_mean   = float(np.mean(bulk_half))
        assert psi_bulk_mean > psi_vacuum_mean, (
            f"Toh ψ should be larger in bulk ({psi_bulk_mean:.4f}) than vacuum "
            f"({psi_vacuum_mean:.4f}) — limiter is not density-sensitive"
        )


# ===========================================================================
# Standalone convergence table (run directly for the report)
# ===========================================================================

def _print_convergence_table() -> None:
    print("\n" + "=" * 72)
    print("PINCH CONVERGENCE SWEEP: toh_psi_ni vs minmod")
    print("Problem: smooth axial sinusoidal wave, 1D (nr=4 slab)")
    print("=" * 72)

    header = f"{'N':>6} | {'method':>10} | {'L1(rho)':>11} | {'L1(P)':>11} | {'L1(Bz)':>11}"
    print(header)
    print("-" * 72)

    all_errs: dict[str, dict[str, list[float]]] = {}
    for label, use_toh in [("minmod", False), ("toh_psi_ni", True)]:
        errs = _run_smooth_sweep(NS, use_toh)
        all_errs[label] = errs
        for i, n in enumerate(NS):
            row = (
                f"{n:>6} | {label:>10} | "
                f"{errs['rho'][i]:>11.3e} | "
                f"{errs['p'][i]:>11.3e} | "
                f"{errs['Bz'][i]:>11.3e}"
            )
            print(row)
        print()

    print("Convergence rates (per doubling):")
    rate_header = f"{'pair':>12} | {'method':>10} | {'rate(rho)':>9} | {'rate(P)':>9} | {'rate(Bz)':>9}"
    print(rate_header)
    print("-" * 60)
    for label in ("minmod", "toh_psi_ni"):
        errs = all_errs[label]
        for i in range(1, len(NS)):
            pair = f"{NS[i-1]}->{NS[i]}"
            rr = _convergence_rate(errs["rho"][i-1], errs["rho"][i], NS[i-1], NS[i])
            rp = _convergence_rate(errs["p"][i-1],   errs["p"][i],   NS[i-1], NS[i])
            rb = _convergence_rate(errs["Bz"][i-1],  errs["Bz"][i],  NS[i-1], NS[i])
            print(f"{pair:>12} | {label:>10} | {rr:>9.2f} | {rp:>9.2f} | {rb:>9.2f}")

    print()
    print("Low-density step: total variation (TV) of reconstruction")
    tv_header = f"{'N':>6} | {'method':>10} | {'TV(rho)':>11}"
    print(tv_header)
    print("-" * 40)
    for n in NS:
        rho, p, bz = _low_density_step(n)
        for label, use_toh in [("minmod", False), ("toh_psi_ni", True)]:
            solver = _make_solver(n, use_toh)
            tv = _total_slope_norm(solver, rho, rho)
            print(f"{n:>6} | {label:>10} | {tv:>11.4e}")

    print("=" * 72)


if __name__ == "__main__":
    _print_convergence_table()
