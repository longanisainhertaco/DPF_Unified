"""Symbolic regression to discover DPF scaling laws.

Pulls device parameters + neutron yields from dpf_research.db and
cross-references with lee_fits to build a feature matrix, then:
1. Attempts PySR symbolic regression (requires Julia).
2. Falls back to correlation analysis + manual I^4 law validation.

Outputs:
    docs/research-reference/scaling_laws.json
"""

import sys
import json
import warnings
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "docs/research-reference/dpf_research.db"
OUTPUT_JSON = ROOT / "docs/research-reference/scaling_laws.json"


def _parse_lee_fields(conditions: str | None, notes: str | None) -> dict:
    """Extract Ipeak_kA, Ipinch_kA, Yn from lee_fits conditions + notes strings."""
    import re
    result = {}
    combined = f"{conditions or ''} {notes or ''}"

    m = re.search(r"Ipeak=(\d+(?:\.\d+)?)\s*kA", combined)
    if m:
        result["I_peak_MA"] = float(m.group(1)) / 1e3

    m = re.search(r"Ipinch=(\d+(?:\.\d+)?)\s*kA", combined)
    if m:
        result["I_pinch_MA"] = float(m.group(1)) / 1e3

    # Prefer Yn_meas over Yn_code; plain "Yn=" last
    m = re.search(r"Yn_meas=(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", combined)
    if m:
        result["Yn"] = float(m.group(1))
    else:
        m = re.search(r"(?<![_\w])Yn=(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", combined)
        if m:
            result["Yn"] = float(m.group(1))

    return result


def load_device_data() -> list[dict]:
    """Build per-device feature dicts from experimental_data + lee_fits notes."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    param_interest = {
        "I_peak", "peak_current", "peak_current_experimental",
        "anode_radius", "cathode_radius",
        "capacitance", "stored_energy",
        "Yn_avg_DD", "neutron_yield", "Yn_DD", "Yn_max_DD",
        "V_charge", "V0", "rise_time",
    }

    c.execute(
        "SELECT device, parameter, value, unit FROM experimental_data WHERE parameter IN ({})".format(
            ",".join(f"'{p}'" for p in param_interest)
        )
    )
    rows = c.fetchall()

    raw: dict[str, dict] = {}
    for device, param, value, unit in rows:
        if device not in raw:
            raw[device] = {}
        raw[device][param] = (value, unit)

    # Normalise to common keys (MA, cm, kJ, n/shot)
    devices = []
    for device, fields in raw.items():
        d: dict = {"device": device}

        # I_peak in MA
        for k in ("I_peak", "peak_current", "peak_current_experimental"):
            if k in fields:
                val, unit = fields[k]
                if unit in ("kA", "kA peak"):
                    d["I_peak_MA"] = val / 1e3
                elif unit in ("MA",):
                    d["I_peak_MA"] = val
                elif unit in ("A",):
                    d["I_peak_MA"] = val / 1e6
                break

        # anode radius in cm
        if "anode_radius" in fields:
            val, unit = fields["anode_radius"]
            if unit == "cm":
                d["anode_r_cm"] = val
            elif unit == "mm":
                d["anode_r_cm"] = val / 10

        # stored energy in kJ
        for k in ("stored_energy",):
            if k in fields:
                val, unit = fields[k]
                if unit == "MJ":
                    d["E_kJ"] = val * 1e3
                elif unit in ("kJ",):
                    d["E_kJ"] = val
                elif unit in ("J",):
                    d["E_kJ"] = val / 1e3

        # neutron yield
        for k in ("Yn_avg_DD", "Yn_max_DD", "Yn_DD", "neutron_yield"):
            if k in fields:
                val, _ = fields[k]
                d["Yn"] = val
                break

        if "I_peak_MA" in d and "Yn" in d:
            devices.append(d)

    # Supplement with shot-level data from lee_fits (Akel 2021 PF-1000 24-shot campaign)
    c.execute("SELECT device, conditions, notes FROM lee_fits")
    for device, conditions, notes in c.fetchall():
        parsed = _parse_lee_fields(conditions, notes)
        if "I_peak_MA" in parsed and "Yn" in parsed:
            shot_label = f"{device}:{conditions[:20] if conditions else 'unknown'}"
            devices.append({
                "device": shot_label,
                "I_peak_MA": parsed["I_peak_MA"],
                "I_pinch_MA": parsed.get("I_pinch_MA"),
                "Yn": parsed["Yn"],
            })

    conn.close()
    return devices


def i4_law_fit(I_MA: np.ndarray, Yn: np.ndarray) -> dict:
    """Fit Yn = A * I^alpha via log-linear regression."""
    mask = (I_MA > 0) & (Yn > 0)
    log_I = np.log(I_MA[mask])
    log_Y = np.log(Yn[mask])
    # linear fit: log_Y = log_A + alpha * log_I
    coeffs = np.polyfit(log_I, log_Y, 1)
    alpha = float(coeffs[0])
    log_A = float(coeffs[1])
    A = float(np.exp(log_A))
    residuals = log_Y - np.polyval(coeffs, log_I)
    r2 = float(1 - np.var(residuals) / np.var(log_Y))
    return {"alpha": alpha, "A": A, "log_A": log_A, "r2": r2, "n": int(mask.sum())}


def correlation_matrix(
    data: list[dict], features: list[str]
) -> dict[tuple[str, str], float]:
    """Spearman rank correlations between all pairs in log space."""
    from scipy.stats import spearmanr  # type: ignore[import-untyped]

    valid_rows = [
        d for d in data
        if all(d.get(f) is not None and d[f] > 0 for f in features)
    ]
    if len(valid_rows) < 3:
        return {}

    mat = np.array([[np.log(d[f]) for f in features] for d in valid_rows])
    corr: dict = {}
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            if i < j:
                rho, pval = spearmanr(mat[:, i], mat[:, j])
                corr[(fi, fj)] = {"rho": float(rho), "pval": float(pval)}
    return corr


def attempt_pysr(devices: list[dict]) -> dict | None:
    """Try PySR symbolic regression; return result or None on import failure."""
    try:
        from pysr import PySRRegressor  # type: ignore[import-untyped]
    except ImportError:
        return None

    features = ["I_peak_MA", "anode_r_cm", "E_kJ"]
    valid = [
        d for d in devices
        if all(d.get(f) is not None and d[f] > 0 for f in features + ["Yn"])
    ]
    if len(valid) < 5:
        return {"status": "insufficient_data", "n": len(valid)}

    X = np.array([[d[f] for f in features] for d in valid])
    y = np.log10(np.array([d["Yn"] for d in valid]))  # predict log10(Yn)

    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp"],
        complexity_of_operators={"^": 2, "sqrt": 1, "log": 1},
        maxsize=20,
        populations=15,
        verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, variable_names=features)

    best = model.get_best()
    return {
        "status": "success",
        "best_equation": str(best["sympy_format"]),
        "complexity": int(best["complexity"]),
        "loss": float(best["loss"]),
        "feature_names": features,
        "n_samples": len(valid),
    }


def main() -> None:
    print(f"Loading device data from {DB_PATH}")
    devices = load_device_data()
    print(f"Found {len(devices)} devices with I_peak + Yn")

    for d in devices:
        print(f"  {d['device']}: I={d.get('I_peak_MA', '?'):.3f} MA, Yn={d.get('Yn', '?'):.2e}")

    I_MA = np.array([d["I_peak_MA"] for d in devices])
    Yn = np.array([d["Yn"] for d in devices])

    print("\n=== I^alpha power law fit (I_peak) ===")
    fit = i4_law_fit(I_MA, Yn)
    print(f"  Yn = {fit['A']:.3e} * I_peak^{fit['alpha']:.2f}   (R²={fit['r2']:.3f}, n={fit['n']})")
    print(f"  Published Lee/Saw law: alpha ≈ 3.3-4.5")
    print(f"  Note: PF-1000 campaign spans only 1.13–1.33 MA; shot scatter driven by radial Lee params")

    # Also fit on I_pinch where available (the true physical quantity for Yn scaling)
    devices_with_pinch = [d for d in devices if d.get("I_pinch_MA") and d.get("Yn")]
    if len(devices_with_pinch) >= 3:
        I_pinch_arr = np.array([d["I_pinch_MA"] for d in devices_with_pinch])
        Yn_pinch_arr = np.array([d["Yn"] for d in devices_with_pinch])
        fit_pinch = i4_law_fit(I_pinch_arr, Yn_pinch_arr)
        print(f"\n=== I^alpha power law fit (I_pinch, n={fit_pinch['n']}) ===")
        print(f"  Yn = {fit_pinch['A']:.3e} * I_pinch^{fit_pinch['alpha']:.2f}   (R²={fit_pinch['r2']:.3f})")
    else:
        fit_pinch = None

    corr = correlation_matrix(devices, ["I_peak_MA", "anode_r_cm", "E_kJ", "Yn"])
    print("\n=== Spearman correlations (log-space) ===")
    for (fi, fj), v in corr.items():
        print(f"  {fi} vs {fj}: rho={v['rho']:.3f} (p={v['pval']:.3f})")

    print("\n=== Attempting PySR symbolic regression ===")
    pysr_result = attempt_pysr(devices)
    if pysr_result is None:
        print("  PySR not installed (Julia dependency). Skipping.")
        pysr_result = {"status": "not_installed"}
    else:
        print(f"  Status: {pysr_result['status']}")
        if pysr_result.get("best_equation"):
            print(f"  Best eq: {pysr_result['best_equation']}")

    results = {
        "n_devices": len(devices),
        "devices": [
            {
                "name": d["device"],
                "I_peak_MA": d.get("I_peak_MA"),
                "I_pinch_MA": d.get("I_pinch_MA"),
                "Yn": d.get("Yn"),
                "anode_r_cm": d.get("anode_r_cm"),
                "E_kJ": d.get("E_kJ"),
            }
            for d in devices
        ],
        "power_law_I_peak": fit,
        "power_law_I_pinch": fit_pinch,
        "correlations": {f"{fi}_vs_{fj}": v for (fi, fj), v in corr.items()},
        "pysr": pysr_result,
        "notes": (
            "power_law_I_peak: Yn = A * I_peak^alpha. "
            "power_law_I_pinch: Yn = A * I_pinch^alpha (physically motivated — pinch current drives thermonuclear yield). "
            "Published Lee/Saw: alpha in [3.3, 4.5]. "
            "PF-1000 24-shot campaign spans narrow I_peak range; shot scatter driven by radial Lee params (fmr, fcr). "
            "Correlations in log-space via Spearman rank."
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
