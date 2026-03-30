#!/usr/bin/env python3
# ruff: noqa: E701, E702, E402
"""DPF-Unified V&V Campaign: 6-campaign verification and validation suite.

Runs all campaigns sequentially against the MLX MHD solver, produces JSON
data files and a final markdown report. Total budget: 5 hours (18,000s).

Usage:
    python3 scripts/run_vv_campaign.py
    python3 scripts/run_vv_campaign.py --campaign 1
    python3 scripts/run_vv_campaign.py --dry-run
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dpf.metal.mlx_engine import run_mlx_discharge

OUT_DIR = _ROOT / "training" / "vv_campaign"

# PF-1000 analytic reference
PF1000_C = 1.332e-3
PF1000_V0 = 27e3
PF1000_L0 = 33.5e-9
PF1000_R0 = 2.3e-3
PF1000_I_SC_MA = PF1000_V0 / math.sqrt(PF1000_L0 / PF1000_C) / 1e6
PF1000_T_QUARTER_US = 0.5 * math.pi * math.sqrt(PF1000_L0 * PF1000_C) * 1e6
PF1000_E_STORED_KJ = 0.5 * PF1000_C * PF1000_V0**2 / 1e3

DEVICE_REFERENCES = {
    "pf1000":        {"I_peak_kA": 1870.0, "source": "Scholz 2006"},
    "unu_ictp":      {"I_peak_kA": 170.0,  "source": "Lee 1988"},
    "faeton":        {"I_peak_kA": 900.0,  "source": "Damideh 2025"},
    "poseidon_60kv": {"I_peak_kA": 3190.0, "source": "IPFS digitized"},
    "mjolnir":       {"I_peak_kA": 3000.0, "source": "Goyon 2025"},
    "poseidon":      {"I_peak_kA": 2600.0, "source": "Herold 1989"},
}

GRID_LOW = (16, 1, 32)
GRID_MED = (32, 1, 64)
GRID_HIGH = (64, 1, 128)


def _shot(preset: str = "pf1000", fc: float | None = None, fm: float | None = None,
          V0_kV: float | None = None, P: float | None = None,
          grid: tuple[int, int, int] = GRID_MED, max_steps: int = 50000) -> dict:
    t0 = time.perf_counter()
    try:
        r = run_mlx_discharge(preset_name=preset, mode="mhd", max_steps=max_steps,
                              fc=fc, fm=fm, V0_kV=V0_kV, pressure_torr=P, grid_shape=grid)
        r["wall_s"] = round(time.perf_counter() - t0, 3)
        r["error"] = None
        return r
    except Exception as e:
        return {"error": str(e)[:200], "wall_s": round(time.perf_counter() - t0, 3),
                "I_peak_MA": float("nan"), "t_peak_us": float("nan"), "n_steps": 0}


def _log(c: int, i: int, n: int, r: dict) -> None:
    if i % 10 == 0 or r.get("error") or i == n - 1:
        I = f"{r['I_peak_MA']:.3f}" if not math.isnan(r.get("I_peak_MA", float("nan"))) else "NaN"
        s = "ERR" if r.get("error") else "OK"
        total = f"/{n}" if n > 0 else ""
        print(f"  C{c} [{i+1:4d}{total}] I={I}MA {r['wall_s']:.1f}s {s}")


# ── Campaign 1: Grid Convergence ─────────────────────────────────────
def campaign_1() -> dict:
    grids = [GRID_LOW, GRID_MED, GRID_HIGH]
    results = []
    t0 = time.perf_counter()
    for g in grids:
        print(f"  Grid {g}...")
        r = _shot(fc=0.7, fm=0.08, V0_kV=27.0, P=3.5, grid=g)
        r["grid_shape"] = list(g)
        results.append(r)

    I = [r["I_peak_MA"] for r in results]
    e12 = abs(I[0] - I[1])
    e23 = abs(I[1] - I[2])
    order = math.log(e12 / e23) / math.log(2.0) if e23 > 1e-10 and e12 > 1e-10 else float("inf")
    rel_err = abs(I[1] - I[2]) / max(abs(I[2]), 1e-30) * 100

    return {"campaign": 1, "name": "grid_convergence",
            "status": "PASS" if (order >= 1.0 or order == float("inf")) and rel_err < 5 else "FAIL",
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "grids": results, "I_peaks_MA": I,
            "convergence_order": round(order, 2) if order != float("inf") else "inf",
            "relative_error_pct": round(rel_err, 3),
            "pass_criteria": {"order_ge_1": order >= 1.0, "relative_error_lt_5pct": rel_err < 5}}


# ── Campaign 2: Reproducibility ──────────────────────────────────────
def campaign_2(n: int = 50) -> dict:
    shots = []
    t0 = time.perf_counter()
    for i in range(n):
        r = _shot(fc=0.7, fm=0.08, V0_kV=27.0, P=3.5)
        shots.append(r)
        _log(2, i, n, r)

    I = np.array([s["I_peak_MA"] for s in shots])
    std = float(np.std(I))
    return {"campaign": 2, "name": "reproducibility",
            "status": "PASS" if std < 1e-6 else "FAIL",
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "n_shots": n, "I_peak_mean": round(float(np.mean(I)), 6),
            "I_peak_std": std, "I_peak_range": round(float(np.ptp(I)), 10),
            "wall_time_mean": round(float(np.mean([s["wall_s"] for s in shots])), 2),
            "pass_criteria": {"std_lt_1e6": std < 1e-6}}


# ── Campaign 3: Analytic Limits ──────────────────────────────────────
def _c3a_voltage():
    V0s = np.linspace(15, 40, 10).tolist()
    shots = [_shot(fc=0.7, fm=0.08, V0_kV=v, P=3.5) for v in V0s]
    for i, s in enumerate(shots):
        s["V0_kV"] = V0s[i]; _log(3, i, 10, s)
    I = np.array([s["I_peak_MA"] for s in shots])
    r = float(np.corrcoef(V0s, I)[0, 1])
    return {"sub": "3a_voltage", "V0": V0s, "I": I.tolist(), "r": round(r, 4),
            "pass": r > 0.99, "shots": shots}

def _c3b_pressure():
    Ps = np.linspace(0.5, 20, 10).tolist()
    shots = [_shot(fc=0.7, fm=0.08, V0_kV=27.0, P=p) for p in Ps]
    for i, s in enumerate(shots):
        s["P"] = Ps[i]; _log(3, i, 10, s)
    I = np.array([s["I_peak_MA"] for s in shots])
    return {"sub": "3b_pressure", "P": Ps, "I": I.tolist(), "shots": shots}

def _c3c_fm():
    fms = np.linspace(0.01, 0.50, 10).tolist()
    shots = [_shot(fc=0.7, fm=f, V0_kV=27.0, P=3.5) for f in fms]
    for i, s in enumerate(shots):
        s["fm"] = fms[i]
        s["Lp_max_nH"] = max(s.get("Lp_nH", [0])); _log(3, i, 10, s)
    Lp = [s["Lp_max_nH"] for s in shots]
    mono = all(Lp[i] <= Lp[i+1] for i in range(len(Lp)-1) if not (math.isnan(Lp[i]) or math.isnan(Lp[i+1])))
    return {"sub": "3c_fm", "fm": fms, "Lp_max": Lp, "I": [s["I_peak_MA"] for s in shots],
            "monotonic": mono, "pass": mono, "shots": shots}

def _c3d_fc():
    fcs = np.linspace(0.3, 0.9, 10).tolist()
    shots = [_shot(fc=f, fm=0.08, V0_kV=27.0, P=3.5) for f in fcs]
    for i, s in enumerate(shots):
        s["fc"] = fcs[i]; _log(3, i, 10, s)
    I = np.array([s["I_peak_MA"] for s in shots])
    r = float(np.corrcoef(fcs, I)[0, 1])
    return {"sub": "3d_fc", "fc": fcs, "I": I.tolist(), "r": round(r, 4), "shots": shots}

def _c3e_low_pressure():
    V0s = [20.0, 25.0, 27.0, 30.0, 35.0]
    shots = [_shot(fc=0.7, fm=0.08, V0_kV=v, P=0.1) for v in V0s]
    for i, s in enumerate(shots):
        I_sc = (V0s[i] * 1e3) / math.sqrt(PF1000_L0 / PF1000_C) / 1e6
        s["V0_kV"] = V0s[i]; s["I_sc_MA"] = round(I_sc, 3)
        s["loading"] = round(s["I_peak_MA"] / I_sc, 4) if I_sc > 0 else 0; _log(3, i, 5, s)
    return {"sub": "3e_low_P", "shots": shots, "loadings": [s["loading"] for s in shots]}

def _c3f_high_pressure():
    Ps = [30.0, 40.0, 50.0, 60.0, 80.0]
    shots = [_shot(fc=0.7, fm=0.08, V0_kV=27.0, P=p) for p in Ps]
    for i, s in enumerate(shots):
        s["P"] = Ps[i]; _log(3, i, 5, s)
    I = [s["I_peak_MA"] for s in shots]
    decr = all(I[i] >= I[i+1] for i in range(len(I)-1))
    return {"sub": "3f_high_P", "P": Ps, "I": I, "decreasing": decr, "pass": decr, "shots": shots}

def campaign_3() -> dict:
    t0 = time.perf_counter()
    subs = {}
    for name, fn in [("3a", _c3a_voltage), ("3b", _c3b_pressure), ("3c", _c3c_fm),
                     ("3d", _c3d_fc), ("3e", _c3e_low_pressure), ("3f", _c3f_high_pressure)]:
        print(f"  {name}...")
        subs[name] = fn()
    p3a = subs["3a"].get("pass", False)
    p3c = subs["3c"].get("pass", False)
    p3f = subs["3f"].get("pass", False)
    return {"campaign": 3, "name": "analytic_limits",
            "status": "PASS" if p3a and p3c and p3f else "FAIL",
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "sub_campaigns": subs, "pass_summary": {"3a": p3a, "3c": p3c, "3f": p3f}}


# ── Campaign 4: Cross-Device ─────────────────────────────────────────
def campaign_4() -> dict:
    devices = ["pf1000", "unu_ictp", "faeton", "poseidon_60kv", "mjolnir", "poseidon"]
    t0 = time.perf_counter()
    results = []
    for i, dev in enumerate(devices):
        print(f"  {dev}...")
        r = _shot(preset=dev)
        ref = DEVICE_REFERENCES.get(dev, {})
        I_exp = ref.get("I_peak_kA", 0)
        I_sim = r["I_peak_MA"] * 1e3
        err = abs(I_sim - I_exp) / I_exp * 100 if I_exp > 0 else float("nan")
        results.append({"preset": dev, "I_sim_kA": round(I_sim, 1), "I_exp_kA": I_exp,
                        "error_pct": round(err, 1), "within_25": err < 25, "wall_s": r["wall_s"],
                        "source": ref.get("source", "")})
        _log(4, i, len(devices), r)
    all_ok = all(d["within_25"] for d in results)
    errs = [d["error_pct"] for d in results if not math.isnan(d["error_pct"])]
    return {"campaign": 4, "name": "cross_device",
            "status": "PASS" if all_ok else "FAIL",
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "devices": results, "mean_error_pct": round(float(np.mean(errs)), 1)}


# ── Campaign 5: Statistical Power ────────────────────────────────────
def campaign_5(n: int = 500) -> dict:
    rng = np.random.default_rng(2026)
    V0 = rng.uniform(20, 35, n); P = rng.uniform(2, 8, n)
    fc = rng.uniform(0.55, 0.85, n); fm = rng.uniform(0.04, 0.20, n)
    t0 = time.perf_counter()
    I_arr = np.zeros(n); t_arr = np.zeros(n); w_arr = np.zeros(n)
    n_fail = 0
    for i in range(n):
        r = _shot(fc=float(fc[i]), fm=float(fm[i]), V0_kV=float(V0[i]),
                  P=float(P[i]), grid=GRID_LOW)
        I_arr[i] = r["I_peak_MA"]; t_arr[i] = r["t_peak_us"]; w_arr[i] = r["wall_s"]
        if r.get("error"): n_fail += 1
        _log(5, i, n, r)

    ok = ~np.isnan(I_arr)
    pearson = {}
    for name, x in [("V0", V0), ("P", P), ("fc", fc), ("fm", fm)]:
        pearson[name] = round(float(np.corrcoef(x[ok], I_arr[ok])[0, 1]), 4) if ok.sum() > 2 else 0

    R2 = 0; coeffs = {}; resid_std = 0
    if ok.sum() > 10:
        X = np.column_stack([V0[ok], P[ok], fc[ok], fm[ok], np.ones(ok.sum())])
        beta, _, _, _ = np.linalg.lstsq(X, I_arr[ok], rcond=None)
        pred = X @ beta
        ss_res = float(np.sum((I_arr[ok] - pred)**2))
        ss_tot = float(np.sum((I_arr[ok] - np.mean(I_arr[ok]))**2))
        R2 = round(1 - ss_res / max(ss_tot, 1e-30), 4)
        resid_std = round(float(np.std(I_arr[ok] - pred)), 6)
        coeffs = {k: round(float(v), 6) for k, v in zip(["V0", "P", "fc", "fm", "intercept"], beta, strict=False)}

    sobol = {}
    for name, x in [("V0", V0), ("P", P), ("fc", fc), ("fm", fm)]:
        if ok.sum() > 20:
            bins = np.linspace(x[ok].min(), x[ok].max(), 11)
            idx = np.clip(np.digitize(x[ok], bins) - 1, 0, 9)
            gm = np.mean(I_arr[ok])
            ss_b = sum(np.sum(idx == b) * (np.mean(I_arr[ok][idx == b]) - gm)**2
                       for b in range(10) if np.sum(idx == b) > 0)
            ss_t = np.sum((I_arr[ok] - gm)**2)
            sobol[name] = round(float(ss_b / max(ss_t, 1e-30)), 4)

    return {"campaign": 5, "name": "statistical_power", "status": "COMPLETE",
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "n_shots": n, "n_ok": int(ok.sum()), "n_fail": n_fail,
            "I_mean": round(float(np.nanmean(I_arr)), 4),
            "I_std": round(float(np.nanstd(I_arr)), 4),
            "pearson": pearson, "R2": R2, "coefficients": coeffs,
            "residual_std": resid_std, "sobol_proxy": sobol}


# ── Campaign 6: Endurance ────────────────────────────────────────────
def campaign_6(budget_s: float = 10000.0) -> dict:
    t0 = time.perf_counter()
    I_list = []; w_list = []; n_fail = 0; shot = 0
    while (time.perf_counter() - t0) < budget_s:
        r = _shot(fc=0.7, fm=0.08, V0_kV=27.0, P=3.5, grid=GRID_LOW)
        I_list.append(r["I_peak_MA"]); w_list.append(r["wall_s"])
        if r.get("error"): n_fail += 1
        _log(6, shot, -1, r)
        shot += 1

    n = len(w_list); I = np.array(I_list); w = np.array(w_list)
    nb = min(100, n // 4)
    drift = (float(np.mean(w[-nb:])) - float(np.mean(w[:nb]))) / max(float(np.mean(w[:nb])), 1e-30) * 100 if nb > 0 else 0

    bins = []
    bs = max(1, n // 10)
    for b in range(10):
        s, e = b * bs, min((b + 1) * bs, n)
        if s >= n: break
        bins.append({"bin": b, "shots": f"{s}-{e-1}",
                     "wall_mean": round(float(np.mean(w[s:e])), 3),
                     "wall_std": round(float(np.std(w[s:e])), 3)})

    return {"campaign": 6, "name": "endurance",
            "status": "PASS" if n_fail == 0 and abs(drift) < 20 else "FAIL",
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "n_shots": n, "n_fail": n_fail,
            "wall_mean": round(float(np.mean(w)), 3), "wall_std": round(float(np.std(w)), 3),
            "drift_pct": round(drift, 2),
            "I_mean": round(float(np.nanmean(I)), 6), "I_std": round(float(np.nanstd(I)), 8),
            "thermal_bins": bins,
            "pass_criteria": {"zero_failures": n_fail == 0, "drift_lt_20": abs(drift) < 20}}


# ── Report Generator ─────────────────────────────────────────────────
def report(cs: dict[int, dict]) -> str:
    L = ["# DPF-Unified MLX MHD V&V Report\n",
         f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
         "**Platform**: M3 Pro, MLX Metal GPU",
         "**Solver**: HLLS + PLM + SSP-RK2, cylindrical\n",
         "## Summary\n",
         "| # | Campaign | Status | Time |",
         "|---|----------|--------|------|"]
    total_t = 0
    for i in sorted(cs):
        c = cs[i]
        total_t += c["elapsed_s"]
        L.append(f"| {i} | {c['name']} | {c['status']} | {c['elapsed_s']:.0f}s |")
    L.append(f"| | **Total** | | **{total_t:.0f}s ({total_t/3600:.1f}h)** |\n")

    if 1 in cs:
        c = cs[1]
        L += ["\n## C1: Grid Convergence\n",
              "| Grid | I_peak (MA) | Wall (s) |", "|------|-------------|----------|"]
        for g in c["grids"]:
            gs = "x".join(str(x) for x in g["grid_shape"])
            L.append(f"| {gs} | {g['I_peak_MA']:.4f} | {g['wall_s']:.1f} |")
        L.append(f"\nOrder: {c['convergence_order']}, Error: {c['relative_error_pct']:.2f}% — **{c['status']}**\n")

    if 2 in cs:
        c = cs[2]
        L += ["\n## C2: Reproducibility\n",
              f"- {c['n_shots']} shots, I_peak mean={c['I_peak_mean']:.6f} MA",
              f"- I_peak std={c['I_peak_std']:.2e}, range={c['I_peak_range']:.2e}",
              f"- Wall time mean={c['wall_time_mean']:.2f}s — **{c['status']}**\n"]

    if 3 in cs:
        c = cs[3]
        L.append(f"\n## C3: Analytic Limits — **{c['status']}**\n")
        for _k, s in c["sub_campaigns"].items():
            p = s.get("pass", "N/A")
            extra = ""
            if "r" in s: extra = f", r={s['r']}"
            if "monotonic" in s: extra = f", monotonic={s['monotonic']}"
            if "decreasing" in s: extra = f", decreasing={s['decreasing']}"
            L.append(f"- **{s['sub']}**: pass={p}{extra}")

    if 4 in cs:
        c = cs[4]
        L += [f"\n## C4: Cross-Device — **{c['status']}**\n",
              "| Device | I_sim (kA) | I_exp (kA) | Error | Source |",
              "|--------|-----------|-----------|-------|--------|"]
        for d in c["devices"]:
            L.append(f"| {d['preset']} | {d['I_sim_kA']:.0f} | {d['I_exp_kA']:.0f} | {d['error_pct']:.1f}% | {d['source']} |")
        L.append(f"\nMean error: {c['mean_error_pct']:.1f}%\n")

    if 5 in cs:
        c = cs[5]
        L += [f"\n## C5: Statistical Power ({c['n_ok']}/{c['n_shots']} shots)\n",
              f"- I_peak: {c['I_mean']:.3f} +/- {c['I_std']:.3f} MA",
              f"- R^2 = {c['R2']} (linear model)\n",
              "| Param | Pearson r | Sobol eta^2 |",
              "|-------|-----------|-------------|"]
        for p in ["V0", "P", "fc", "fm"]:
            L.append(f"| {p} | {c['pearson'].get(p, 0):.3f} | {c['sobol_proxy'].get(p, 0):.3f} |")

    if 6 in cs:
        c = cs[6]
        L += [f"\n## C6: Endurance — **{c['status']}**\n",
              f"- {c['n_shots']} shots, {c['n_fail']} failures",
              f"- Wall: {c['wall_mean']:.3f} +/- {c['wall_std']:.3f}s",
              f"- Thermal drift: {c['drift_pct']:.1f}%",
              f"- I_peak: {c['I_mean']:.6f} +/- {c['I_std']:.2e} MA\n",
              "| Bin | Shots | Wall mean | Wall std |",
              "|-----|-------|-----------|----------|"]
        for b in c.get("thermal_bins", []):
            L.append(f"| {b['bin']} | {b['shots']} | {b['wall_mean']:.3f} | {b['wall_std']:.3f} |")

    L += ["\n## Reference\n",
          f"- I_sc = {PF1000_I_SC_MA:.2f} MA, T/4 = {PF1000_T_QUARTER_US:.2f} us, E = {PF1000_E_STORED_KJ:.0f} kJ"]
    return "\n".join(L) + "\n"


# ── Main ─────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--endurance-budget", type=float, default=10000.0)
    args = ap.parse_args()

    if args.dry_run:
        print("C1 Convergence:    ~70s\nC2 Reproducibility: ~350s\nC3 Analytic:       ~350s")
        print("C4 Cross-device:   ~60s\nC5 Statistical:    ~1000s\nC6 Endurance:      ~budget")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runners = {1: ("c1_convergence.json", campaign_1),
               2: ("c2_reproducibility.json", campaign_2),
               3: ("c3_analytic_limits.json", campaign_3),
               4: ("c4_cross_device.json", campaign_4),
               5: ("c5_statistical.json", lambda: campaign_5()),
               6: ("c6_endurance.json", lambda: campaign_6(args.endurance_budget))}

    to_run = {args.campaign: runners[args.campaign]} if args.campaign else runners
    results: dict[int, dict] = {}
    t0 = time.perf_counter()

    for num, (fname, fn) in sorted(to_run.items()):
        print(f"\n{'='*60}\nCAMPAIGN {num}\n{'='*60}")
        c = fn()
        results[num] = c
        # Strip time series for JSON
        def _clean(d):
            skip = {"t_us", "I_MA", "V_kV", "Lp_nH", "phases", "shots"}
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items() if k not in skip}
            if isinstance(d, list) and len(d) > 100:
                return f"[{len(d)} items]"
            return d
        with open(OUT_DIR / fname, "w") as f:
            json.dump(_clean(c), f, indent=2, default=str)
        print(f"  -> {c['status']} ({c['elapsed_s']:.0f}s)")

    total = time.perf_counter() - t0
    print(f"\n{'='*60}\nALL COMPLETE: {total:.0f}s ({total/3600:.1f}h)\n{'='*60}")

    rpt = report(results)
    (OUT_DIR / "vv_report.md").write_text(rpt)
    print(rpt)


if __name__ == "__main__":
    main()
