"""PIC V4 Diagnostic — step-by-step particle/velocity/J stats for full discharge.

Runs the same PF-1000 8x1x16 + PIC setup as test_pic_v4_short_discharge,
but prints diagnostics at every step and stops at the first NaN, reporting
the failure chain.

Usage:
    python3 scripts/pic_v4_diagnostic.py [--steps N] [--particles N]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# Ensure src/ is on path when running from repo root
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _build_config(n_steps_budget: int, n_particles: int):  # type: ignore[return]
    """Build a minimal PF-1000 PIC config."""
    from dpf.config import SimulationConfig
    from dpf.presets import get_preset

    base = get_preset("pf1000")

    base["grid_shape"] = [8, 1, 16]
    base["dx"] = 7.5e-4
    # sim_time ~ n_steps * 1 ns headroom
    base["sim_time"] = n_steps_budget * 2e-9

    base["kinetic"] = {
        "enabled": True,
        "start_time": 1e-15,
        "inject_beam": True,
        "n_particles": n_particles,
        "beam_energy": 100e3,
        "beam_position_ratio": [0.5, 0.5, 0.1],
        "beam_direction": [0.0, 0.0, 1.0],
        "beam_weight_total": 1e16,
    }

    if "fluid" not in base:
        base["fluid"] = {}
    base["fluid"]["backend"] = "python"  # type: ignore[index]

    return SimulationConfig(**base)


def _particle_stats(engine) -> dict:  # type: ignore[return]
    """Extract particle count, max |v|, and max |J_kin| from engine state."""
    stats: dict = {
        "n_particles": 0,
        "max_v": float("nan"),
        "max_J": float("nan"),
        "nan_v": False,
        "nan_J": False,
    }

    km = getattr(engine, "kinetic", None)
    if km is None or not km.kc.enabled:
        stats["n_particles"] = -1  # PIC not active
        return stats

    sp = km.ion_species
    stats["n_particles"] = sp.n_particles()

    if sp.n_particles() > 0:
        speeds = np.sqrt(np.sum(sp.velocities ** 2, axis=1))
        stats["max_v"] = float(np.max(speeds))
        stats["nan_v"] = bool(not np.all(np.isfinite(sp.velocities)))
    else:
        stats["max_v"] = 0.0
        stats["nan_v"] = False

    try:
        _, Jx, Jy, Jz = km.driver.deposit()
        J_total = np.abs(Jx) + np.abs(Jy) + np.abs(Jz)
        stats["max_J"] = float(np.max(J_total))
        stats["nan_J"] = bool(not np.all(np.isfinite(J_total)))
    except Exception as exc:
        stats["max_J"] = float("nan")
        stats["nan_J"] = True
        stats["deposit_exc"] = str(exc)

    return stats


def _nan_fields(state: dict) -> list[str]:
    return [
        k for k, v in state.items()
        if isinstance(v, np.ndarray) and not np.all(np.isfinite(v))
    ]


def run_diagnostic(n_steps: int = 100, n_particles: int = 50) -> None:
    print(f"PIC V4 Diagnostic: {n_steps} steps, {n_particles} particles")
    print("Setup: PF-1000 8x1x16, Python backend, PIC start_time=1e-15 s")
    print("-" * 72)

    try:
        cfg = _build_config(n_steps, n_particles)
    except Exception as exc:
        print(f"FAIL: Config construction error: {exc}")
        sys.exit(1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from dpf.engine.core import SimulationEngine
            engine = SimulationEngine(cfg)
    except Exception as exc:
        print(f"FAIL: Engine __init__ error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"{'Step':>5}  {'n_part':>7}  {'max|v| m/s':>14}  {'max|J| A/m2':>14}  {'NaN?':>6}  {'status'}")
    print("-" * 72)

    c_light = 2.998e8

    for i in range(n_steps):
        # Pre-step particle stats
        pstats = _particle_stats(engine)

        try:
            result = engine.step()
        except Exception as exc:
            print(
                f"{i:>5}  {pstats['n_particles']:>7}  "
                f"{'--':>14}  {'--':>14}  {'CRASH':>6}  "
                f"EXCEPTION: {type(exc).__name__}: {exc}"
            )
            print()
            print("=== FAILURE CHAIN ===")
            print(f"Step {i}: engine.step() raised {type(exc).__name__}")
            print(f"  {exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        nan_in_state = _nan_fields(engine.state)

        # Format NaN flag
        nan_flag = ""
        if pstats["nan_v"]:
            nan_flag += "VEL "
        if pstats["nan_J"]:
            nan_flag += "J "
        if nan_in_state:
            nan_flag += "MHD"
        nan_flag = nan_flag.strip() or "OK"

        v_str = f"{pstats['max_v']:.3e}" if np.isfinite(pstats["max_v"]) else "NaN"
        j_str = f"{pstats['max_J']:.3e}" if np.isfinite(pstats["max_J"]) else "NaN"

        status = ""
        if pstats["max_v"] > c_light:
            status = f"SUPERLUMINAL ({pstats['max_v']/c_light:.2f}c)"
        elif nan_in_state:
            status = f"NaN MHD fields: {nan_in_state}"

        print(
            f"{i:>5}  {pstats['n_particles']:>7}  "
            f"{v_str:>14}  {j_str:>14}  {nan_flag:>6}  {status}"
        )

        if nan_in_state or pstats["nan_v"] or pstats["nan_J"]:
            print()
            print("=== FAILURE CHAIN ===")
            print(f"Step {i}: NaN detected")
            if nan_in_state:
                print(f"  MHD state NaN in: {nan_in_state}")
            if pstats["nan_v"]:
                print("  Particle velocities contain NaN")
                km = engine.kinetic
                if km and km.ion_species.n_particles() > 0:
                    sp = km.ion_species
                    bad_mask = ~np.isfinite(sp.velocities).all(axis=1)
                    print(f"  Bad particles: {int(np.sum(bad_mask))} / {sp.n_particles()}")
                    if int(np.sum(bad_mask)) > 0:
                        print(f"  First bad velocity: {sp.velocities[bad_mask][0]}")
                        print(f"  First bad position: {sp.positions[bad_mask][0]}")
            if pstats["nan_J"]:
                print("  J_kin contains NaN")
            sys.exit(1)

        if pstats["max_v"] > c_light:
            print()
            print("=== FAILURE CHAIN ===")
            print(f"Step {i}: Superluminal velocity detected")
            print(f"  max|v| = {pstats['max_v']:.4e} m/s = {pstats['max_v']/c_light:.2f}c")
            print(
                "  Root cause: Non-relativistic Boris push + DPF E-field. "
                "See pic_compound_bugs.md §5.2."
            )
            sys.exit(1)

        if result.finished:
            print()
            print(f"Simulation finished at step {i} (sim_time reached).")
            break

    else:
        print()
        print(f"Completed all {n_steps} steps without NaN or crash.")

    print()
    print("=== FINAL SUMMARY ===")
    pstats = _particle_stats(engine)
    print(f"  Steps completed : {engine.step_count}")
    print(f"  Sim time        : {engine.time:.3e} s")
    print(f"  Particles       : {pstats['n_particles']}")
    print(f"  max|v|          : {pstats['max_v']:.3e} m/s ({pstats['max_v']/c_light:.3f}c)")
    print(f"  max|J_kin|      : {pstats['max_J']:.3e} A/m²")
    print(f"  NaN velocity    : {pstats['nan_v']}")
    print(f"  NaN J           : {pstats['nan_J']}")
    nan_final = _nan_fields(engine.state)
    print(f"  NaN MHD fields  : {nan_final if nan_final else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of engine steps")
    parser.add_argument("--particles", type=int, default=50, help="PIC macro-particle count")
    args = parser.parse_args()
    run_diagnostic(n_steps=args.steps, n_particles=args.particles)


if __name__ == "__main__":
    main()
