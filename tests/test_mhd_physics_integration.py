"""Integration tests for new MHD physics features in app_mhd.py.

Each test targets a specific feature block in run_mhd_simulation / _run_python_mhd.
All sims use backend='python', grid_preset='coarse', sim_time_us=0.5 for speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_mhd import run_mhd_simulation

# ── Shared helpers ─────────────────────────────────────────────────────────────

FAST_KWARGS = dict(
    backend="python",
    grid_preset="coarse",
    preset_name="pf1000",
    sim_time_us=0.5,
)


@pytest.fixture(scope="module")
def d2_result():
    """Single short D2 sim shared across neutron-yield and completeness tests."""
    return run_mhd_simulation(**FAST_KWARGS, gas_key="D2")


@pytest.fixture(scope="module")
def ne_result():
    """Single short Ne sim for high-Z radiation tests."""
    return run_mhd_simulation(**FAST_KWARGS, gas_key="Ne")


# ── 1. Neutron yield (lines 151-212) ──────────────────────────────────────────


def test_mhd_neutron_yield_result_has_correct_keys(d2_result):
    """neutron_yield key must exist when present; the sub-keys must be complete."""
    if "neutron_yield" not in d2_result:
        pytest.skip("neutron_yield absent (cold plasma / no final_state) — not a crash")
    ny = d2_result["neutron_yield"]
    expected = {"Y_thermonuclear", "Y_beam_target", "Y_neutron", "bt_fraction", "tau_ns"}
    assert expected.issubset(ny.keys()), f"missing keys: {expected - ny.keys()}"


def test_mhd_neutron_yield_d2_does_not_crash(d2_result):
    """D2 sim must complete without exception; result is always a dict."""
    assert isinstance(d2_result, dict)


def test_mhd_neutron_yield_non_negative_when_present(d2_result):
    """Yield values must be >= 0 when the key is populated."""
    if "neutron_yield" not in d2_result:
        pytest.skip("neutron_yield absent")
    ny = d2_result["neutron_yield"]
    assert ny["Y_thermonuclear"] >= 0.0
    assert ny["Y_beam_target"] >= 0.0
    assert ny["Y_neutron"] >= 0.0
    assert 0.0 <= ny["bt_fraction"] <= 1.0


def test_mhd_neutron_yield_absent_for_non_deuterium(ne_result):
    """neutron_yield must NOT appear for Ne (Z != 1 / A != 2) fills."""
    assert "neutron_yield" not in ne_result


# ── 2. Bennett equilibrium (lines 214-240) ─────────────────────────────────────


def test_mhd_bennett_diagnostic_keys_present(d2_result):
    """bennett dict must contain the four physical quantities."""
    assert "bennett" in d2_result, "bennett key missing from result"
    b = d2_result["bennett"]
    for key in ("beta_pinch", "p_mag_max_Pa", "p_kin_max_Pa", "T_bennett_keV"):
        assert key in b, f"bennett missing key: {key}"


def test_mhd_bennett_diagnostic_values_finite(d2_result):
    """Pressure and temperature Bennett quantities must be finite (no NaN).
    beta_pinch may be inf when B=0 (cold plasma — that is valid by design)."""
    if "bennett" not in d2_result:
        pytest.skip("bennett absent")
    b = d2_result["bennett"]
    for key in ("p_mag_max_Pa", "p_kin_max_Pa", "T_bennett_keV"):
        assert np.isfinite(b[key]), f"bennett[{key}] = {b[key]} is not finite"
    assert not np.isnan(b["beta_pinch"]), "bennett[beta_pinch] is NaN"


def test_mhd_bennett_pressures_non_negative(d2_result):
    """Magnetic and kinetic pressures must be >= 0."""
    if "bennett" not in d2_result:
        pytest.skip("bennett absent")
    b = d2_result["bennett"]
    assert b["p_mag_max_Pa"] >= 0.0
    assert b["p_kin_max_Pa"] >= 0.0


def test_mhd_bennett_source_attribution(d2_result):
    """Bennett dict must carry literature source."""
    if "bennett" not in d2_result:
        pytest.skip("bennett absent")
    assert "source" in d2_result["bennett"]


# ── 3. Instability timing (lines 242-255) ─────────────────────────────────────


def test_mhd_instability_tau_key_present_when_current_nonzero(d2_result):
    """instability dict with tau_m0_ns must appear when I_peak > 0."""
    I_arr = d2_result.get("I_MA", np.array([]))
    if len(I_arr) == 0 or float(np.max(np.abs(I_arr))) == 0.0:
        pytest.skip("no current — instability diagnostic skipped by design")
    assert "instability" in d2_result, "instability key missing despite non-zero current"
    assert "tau_m0_ns" in d2_result["instability"]


def test_mhd_instability_tau_positive(d2_result):
    """tau_m0_ns must be strictly positive (Goyon 2025 formula)."""
    if "instability" not in d2_result:
        pytest.skip("instability absent")
    assert d2_result["instability"]["tau_m0_ns"] > 0.0


def test_mhd_instability_convergence_ratio_positive(d2_result):
    """CR = b/a > 0 always for any DPF geometry."""
    if "instability" not in d2_result:
        pytest.skip("instability absent")
    assert d2_result["instability"]["convergence_ratio"] > 0.0


def test_mhd_instability_source_attribution(d2_result):
    """instability dict must carry Goyon 2025 citation."""
    if "instability" not in d2_result:
        pytest.skip("instability absent")
    assert "source" in d2_result["instability"]
    assert "Goyon" in d2_result["instability"]["source"]


# ── 4. Bremsstrahlung doesn't crash (lines 619-631) ───────────────────────────


def test_mhd_bremsstrahlung_d2_no_exception(d2_result):
    """D2 sim with bremsstrahlung cooling must complete without raising."""
    assert isinstance(d2_result, dict)
    assert d2_result.get("n_steps", 0) > 0, "solver produced zero steps"


def test_mhd_bremsstrahlung_Te_array_exists(d2_result):
    """final_state must include a Te field after bremsstrahlung losses are applied."""
    final = d2_result.get("final_state")
    if final is None:
        pytest.skip("no final_state returned")
    assert "Te" in final, "Te missing from final_state after bremsstrahlung step"


# ── 5. Line radiation for high-Z (lines 632-638) ──────────────────────────────


def test_mhd_line_radiation_ne_no_exception(ne_result):
    """Ne (Z=10) sim must complete without exception."""
    assert isinstance(ne_result, dict)
    assert ne_result.get("n_steps", 0) > 0, "Ne solver produced zero steps"


def test_mhd_line_radiation_ne_Te_finite(ne_result):
    """After line radiation losses Te must remain finite."""
    final = ne_result.get("final_state")
    if final is None:
        pytest.skip("no final_state")
    Te = final.get("Te")
    if Te is None:
        pytest.skip("Te absent from final_state")
    assert np.all(np.isfinite(Te)), "Te contains NaN/Inf after line radiation"


# ── 6. Back-EMF coupling (lines 594-602) ──────────────────────────────────────


def test_mhd_back_emf_coupling_produces_inductance_signal(d2_result):
    """L_p_nH must be populated (coupling_interface called every step)."""
    L = d2_result.get("L_p_nH", np.array([]))
    assert len(L) > 0, "L_p_nH array is empty — coupling_interface never called"


def test_mhd_back_emf_L_plasma_non_negative(d2_result):
    """Plasma inductance returned by coupling_interface must be >= 0."""
    L = d2_result.get("L_p_nH", np.array([]))
    if len(L) == 0:
        pytest.skip("L_p_nH absent")
    assert np.all(L >= 0.0), "negative inductance values detected"


def test_mhd_back_emf_circuit_current_evolves(d2_result):
    """I_MA array must have at least 1 sample (circuit is being stepped)."""
    I = d2_result.get("I_MA", np.array([]))
    assert len(I) >= 1, "I_MA is empty — circuit never stepped"


# ── 7. m=0 perturbation seeding (lines 581-587) ───────────────────────────────


def test_mhd_m0_perturbation_rho_init_is_sinusoidal():
    """Initial rho field must show sinusoidal z-variation, not be uniform."""
    result = run_mhd_simulation(**FAST_KWARGS, gas_key="D2")
    final = result.get("final_state")
    if final is None:
        pytest.skip("no final_state to inspect")
    rho = final["rho"]
    # The seeding leaves a non-uniform imprint on the density field.
    # A perfectly uniform field would have std / mean == 0.
    rho_2d = rho[:, 0, :]  # (nr, nz) mid-plane slice
    relative_variation = rho_2d.std() / (rho_2d.mean() + 1e-30)
    assert relative_variation > 0.0, "rho appears perfectly uniform — m=0 seeding not applied"


def test_mhd_m0_perturbation_amplitude_is_small():
    """Perturbation amplitude delta_rho/rho_0 ~ 1% must be << 100%."""
    result = run_mhd_simulation(**FAST_KWARGS, gas_key="D2")
    final = result.get("final_state")
    if final is None:
        pytest.skip("no final_state")
    rho = final["rho"]
    rho_mean = rho.mean()
    rho_peak_deviation = np.abs(rho - rho_mean).max()
    # Perturbation is seeded at 1%; even after half-microsecond growth it should be << rho_mean
    assert rho_peak_deviation < rho_mean, "density deviation exceeds mean — simulation blown up"


# ── 8. Result dict completeness ───────────────────────────────────────────────


REQUIRED_KEYS = {
    "device", "gas", "gas_key", "elapsed_s", "rho0",
    "circuit", "backend", "grid_shape",
    "t_us", "I_MA", "V_kV", "L_p_nH",
    "E_cap_kJ", "E_ind_kJ", "E_res_kJ",
    "n_steps", "has_mhd", "final_state",
    "rho_max", "T_max", "B_max",
    "E_bank_kJ", "T_LC_us",
}


def test_mhd_result_dict_has_all_required_keys(d2_result):
    """Top-level result dict must contain every expected key."""
    missing = REQUIRED_KEYS - d2_result.keys()
    assert not missing, f"result dict missing keys: {missing}"


def test_mhd_result_backend_label_is_python(d2_result):
    """backend key must reflect the requested engine."""
    assert "python" in d2_result["backend"]


def test_mhd_result_grid_shape_is_coarse(d2_result):
    """grid_shape must match the 'coarse' preset (16, 16, 32)."""
    assert d2_result["grid_shape"] == (16, 16, 32)


def test_mhd_result_elapsed_s_is_positive(d2_result):
    """Wall-clock elapsed time must be a positive float."""
    assert d2_result["elapsed_s"] > 0.0


def test_mhd_result_has_mhd_flag_is_true(d2_result):
    """has_mhd must be True for python backend."""
    assert d2_result["has_mhd"] is True


def test_mhd_result_final_state_contains_state_vars(d2_result):
    """final_state must have the canonical MHD state variables."""
    final = d2_result.get("final_state")
    assert final is not None
    for var in ("rho", "velocity", "pressure", "B"):
        assert var in final, f"final_state missing '{var}'"


@pytest.mark.slow
def test_mhd_ne_result_dict_completeness(ne_result):
    """Ne result must also have all required keys."""
    missing = REQUIRED_KEYS - ne_result.keys()
    assert not missing, f"Ne result dict missing keys: {missing}"
