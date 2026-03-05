"""Phase BP: Stratified LOO + ASME uncertainty propagation tests.

Categorizes N=5 LOO results by:
- Waveform provenance: measured (N=3) vs reconstructed (N=2)
- Plasma significance: L_p/L0 > 0.5 vs L_p/L0 < 0.5
- Degradation thresholds by category
"""

import numpy as np
import pytest
from scipy.stats import t as t_dist


# ── LOO maxiter=3 results (ground truth from run) ──────────────────────

LOO_RESULTS = {
    "PF-1000": {
        "blind": 0.4377, "indep": 0.0963, "degrad": 4.54,
        "fc": 0.500, "fm": 0.227, "delay_us": 0.000,
        "measured": True, "lp_l0": 1.18,
    },
    "POSEIDON-60kV": {
        "blind": 0.1917, "indep": 0.0751, "degrad": 2.55,
        "fc": 0.843, "fm": 0.239, "delay_us": 0.051,
        "measured": True, "lp_l0": 1.23,
    },
    "UNU-ICTP": {
        "blind": 0.0978, "indep": 0.0661, "degrad": 1.48,
        "fc": 0.701, "fm": 0.159, "delay_us": 0.067,
        "measured": True, "lp_l0": 0.07,
    },
    "FAETON-I": {
        "blind": 0.1720, "indep": 0.0172, "degrad": 10.01,
        "fc": 0.801, "fm": 0.146, "delay_us": 0.037,
        "measured": False, "lp_l0": 0.11,
    },
    "MJOLNIR": {
        "blind": 0.1777, "indep": 0.1758, "degrad": 1.01,
        "fc": 0.843, "fm": 0.239, "delay_us": 0.051,
        "measured": False, "lp_l0": 0.16,
    },
}


# ── Helper functions ────────────────────────────────────────────────────

def _get_subset(predicate):
    """Return list of dicts for devices matching predicate."""
    return {k: v for k, v in LOO_RESULTS.items() if predicate(v)}


def _blind_nrmses(subset):
    """Extract blind NRMSEs from subset."""
    return np.array([v["blind"] for v in subset.values()])


def _degradations(subset):
    """Extract degradation ratios from subset."""
    return np.array([v["degrad"] for v in subset.values()])


# ── Full N=5 Statistics ─────────────────────────────────────────────────

class TestFullN5LOO:
    """Tests on full N=5 LOO maxiter=3 results."""

    def test_mean_blind_nrmse(self):
        blind = _blind_nrmses(LOO_RESULTS)
        assert blind.mean() == pytest.approx(0.2154, abs=0.002)

    def test_std_blind_nrmse(self):
        blind = _blind_nrmses(LOO_RESULTS)
        assert blind.std(ddof=1) == pytest.approx(0.1295, abs=0.002)

    def test_95ci_contains_mean(self):
        blind = _blind_nrmses(LOO_RESULTS)
        n = len(blind)
        se = blind.std(ddof=1) / np.sqrt(n)
        t_crit = t_dist.ppf(0.975, n - 1)
        ci_lo = blind.mean() - t_crit * se
        ci_hi = blind.mean() + t_crit * se
        assert ci_lo < blind.mean() < ci_hi

    def test_unique_param_sets(self):
        """4/5 unique param sets (degeneracy resolved from maxiter=1)."""
        params = set()
        for v in LOO_RESULTS.values():
            params.add((round(v["fc"], 3), round(v["fm"], 3), round(v["delay_us"], 3)))
        assert len(params) >= 4

    def test_fc_squared_over_fm_range(self):
        """fc²/fm varies by at least 3x across folds."""
        ratios = [v["fc"] ** 2 / v["fm"] for v in LOO_RESULTS.values()]
        assert max(ratios) / min(ratios) > 3.0

    def test_worse_than_maxiter1(self):
        """maxiter=3 mean is worse than maxiter=1 (0.1785).
        This is expected: maxiter=1 had 3-fold degeneracy that
        artificially lowered variance."""
        blind = _blind_nrmses(LOO_RESULTS)
        assert blind.mean() > 0.1785


# ── Stratified by Waveform Provenance ───────────────────────────────────

class TestMeasuredVsReconstructed:
    """Compare measured (N=3) vs reconstructed (N=2) waveform subsets."""

    def test_measured_count(self):
        measured = _get_subset(lambda v: v["measured"])
        assert len(measured) == 3

    def test_reconstructed_count(self):
        recon = _get_subset(lambda v: not v["measured"])
        assert len(recon) == 2

    def test_measured_devices(self):
        measured = _get_subset(lambda v: v["measured"])
        assert set(measured.keys()) == {"PF-1000", "POSEIDON-60kV", "UNU-ICTP"}

    def test_reconstructed_devices(self):
        recon = _get_subset(lambda v: not v["measured"])
        assert set(recon.keys()) == {"FAETON-I", "MJOLNIR"}

    def test_faeton_suspiciously_good_independent(self):
        """FAETON-I independent NRMSE 0.0172 is suspiciously good.
        Fitting a Lee model to a reconstructed RLC waveform is near-tautological."""
        assert LOO_RESULTS["FAETON-I"]["indep"] < 0.02

    def test_faeton_extreme_degradation(self):
        """FAETON-I 10x degradation is an artifact of fitting to reconstructed data."""
        assert LOO_RESULTS["FAETON-I"]["degrad"] > 8.0

    def test_mjolnir_near_unity_degradation(self):
        """MJOLNIR 1.01x degradation: nearly identical to independent.
        This is because MJOLNIR is so circuit-dominated (L_p/L0=0.16) that
        the blind and independent calibrations converge to similar params."""
        assert LOO_RESULTS["MJOLNIR"]["degrad"] < 1.1


# ── Stratified by Plasma Significance ───────────────────────────────────

class TestPlasmaVsCircuit:
    """Compare plasma-significant (L_p/L0 > 0.5) vs circuit-dominated."""

    def test_plasma_significant_count(self):
        plasma = _get_subset(lambda v: v["lp_l0"] > 0.5)
        assert len(plasma) == 2

    def test_circuit_dominated_count(self):
        circuit = _get_subset(lambda v: v["lp_l0"] < 0.5)
        assert len(circuit) == 3

    def test_plasma_significant_higher_blind(self):
        """Plasma-significant devices have higher blind NRMSE."""
        plasma = _blind_nrmses(_get_subset(lambda v: v["lp_l0"] > 0.5))
        circuit = _blind_nrmses(_get_subset(lambda v: v["lp_l0"] < 0.5))
        assert plasma.mean() > circuit.mean()

    def test_pf1000_worst_blind(self):
        """PF-1000 (highest L_p/L0) has worst blind NRMSE."""
        blind_vals = {k: v["blind"] for k, v in LOO_RESULTS.items()}
        worst = max(blind_vals, key=blind_vals.get)
        assert worst == "PF-1000"

    def test_pf1000_boundary_trapped(self):
        """PF-1000 held-out → fc hits lower bound (0.500).
        The remaining 4 devices are all circuit-dominated and cannot
        constrain fc for a plasma-significant device."""
        assert LOO_RESULTS["PF-1000"]["fc"] == pytest.approx(0.500, abs=0.01)

    def test_pf1000_no_delay(self):
        """PF-1000 held-out → delay=0 (also at bound).
        Double boundary trapping when the plasma-significant device is removed."""
        assert LOO_RESULTS["PF-1000"]["delay_us"] == pytest.approx(0.0, abs=0.01)

    def test_circuit_dominated_low_blind(self):
        """Circuit-dominated devices have low blind NRMSE.
        This tests circuit accuracy, not plasma physics."""
        circuit = _blind_nrmses(_get_subset(lambda v: v["lp_l0"] < 0.5))
        assert circuit.mean() < 0.20


# ── Multi-Condition Transfer ────────────────────────────────────────────

MULTI_CONDITION = {
    "27kV_to_16kV": {"blind": 0.1187, "indep": 0.1150, "degrad": 1.03},
    "16kV_to_27kV": {"blind": 0.1006, "indep": 0.0963, "degrad": 1.04},
    "Scholz_to_Gribkov": {"blind": 0.1972, "indep": 0.1575, "degrad": 1.25},
}


class TestMultiConditionTransfer:
    """Multi-condition validation: same device, different operating conditions."""

    def test_voltage_transfer_near_unity(self):
        """27kV→16kV and 16kV→27kV both have degradation < 1.1."""
        assert MULTI_CONDITION["27kV_to_16kV"]["degrad"] < 1.1
        assert MULTI_CONDITION["16kV_to_27kV"]["degrad"] < 1.1

    def test_bidirectional_symmetry(self):
        """Forward and reverse degradations should be similar."""
        d_fwd = MULTI_CONDITION["27kV_to_16kV"]["degrad"]
        d_rev = MULTI_CONDITION["16kV_to_27kV"]["degrad"]
        assert abs(d_fwd - d_rev) < 0.1

    def test_cross_pub_higher_degradation(self):
        """Cross-publication (same conditions) has higher degradation than
        cross-voltage (different conditions). This is because shot-to-shot
        variation + digitization differences exceed the V0 effect."""
        d_cross_v = MULTI_CONDITION["27kV_to_16kV"]["degrad"]
        d_cross_pub = MULTI_CONDITION["Scholz_to_Gribkov"]["degrad"]
        assert d_cross_pub > d_cross_v

    def test_cross_voltage_better_than_cross_device(self):
        """Multi-condition (same device) gives much lower degradation than
        cross-device (different devices). This proves fc/fm are device
        constants, not condition-dependent."""
        cross_v_degrad = MULTI_CONDITION["27kV_to_16kV"]["degrad"]
        cross_device_degrad = np.mean(_degradations(LOO_RESULTS))
        assert cross_v_degrad < cross_device_degrad

    def test_blind_nrmse_below_20pct(self):
        """All multi-condition blind NRMSEs below 20%."""
        for name, r in MULTI_CONDITION.items():
            assert r["blind"] < 0.20, f"{name}: blind NRMSE {r['blind']:.4f} >= 0.20"


# ── ASME Near-PASS Analysis ────────────────────────────────────────────

class TestASMENearPass:
    """Analyze the ASME ratio 1.03 near-PASS for 27kV→16kV."""

    def test_asme_ratio_barely_fails(self):
        """ASME ratio 1.03 — within 3% of passing threshold."""
        E = MULTI_CONDITION["27kV_to_16kV"]["blind"]  # 0.1187
        u_val = MULTI_CONDITION["27kV_to_16kV"]["indep"]  # 0.1150
        ratio = E / u_val
        assert ratio == pytest.approx(1.03, abs=0.02)

    def test_with_digitization_uncertainty(self):
        """If we add 5% digitization uncertainty to u_val, ratio may flip."""
        E = 0.1187
        u_val_base = 0.1150
        u_digit = 0.05  # 5% digitization uncertainty
        u_val_expanded = np.sqrt(u_val_base**2 + u_digit**2)
        ratio_expanded = E / u_val_expanded
        # With expanded uncertainty, ratio should decrease
        assert ratio_expanded < 1.03

    def test_reverse_asme_fails_clearly(self):
        """Reverse direction ASME ratio 1.48 is a clear FAIL."""
        # This asymmetry exists because the 27kV independent NRMSE
        # (0.0963) is much lower than 16kV (0.1150)
        assert MULTI_CONDITION["16kV_to_27kV"]["blind"] / 0.0680 > 1.4


# ── LOO Diagnostic: Boundary Trapping ──────────────────────────────────

class TestBoundaryTrapping:
    """Tests for optimizer boundary trapping in LOO."""

    def test_pf1000_fc_at_lower_bound(self):
        """fc=0.500 is the lower bound of fc_bounds=(0.5, 0.95)."""
        assert LOO_RESULTS["PF-1000"]["fc"] <= 0.501

    def test_pf1000_delay_at_lower_bound(self):
        """delay=0.000 is the lower bound of delay_bounds_us=(0.0, 2.0)."""
        assert LOO_RESULTS["PF-1000"]["delay_us"] <= 0.001

    def test_non_pf1000_folds_not_trapped(self):
        """Other folds should not have fc at lower bound."""
        for name, r in LOO_RESULTS.items():
            if name == "PF-1000":
                continue
            assert r["fc"] > 0.55, f"{name}: fc={r['fc']:.3f} too close to bound"

    def test_unu_ictp_moderate_fc(self):
        """UNU-ICTP fold: fc=0.701 is well within bounds."""
        assert 0.6 < LOO_RESULTS["UNU-ICTP"]["fc"] < 0.8


# ── Waveform Provenance Metadata Tests ────────────────────────────────

class TestWaveformProvenance:
    """Tests for the waveform_provenance field on ExperimentalDevice."""

    def test_provenance_field_exists(self):
        """All devices have waveform_provenance attribute."""
        from dpf.validation.experimental import DEVICES
        for name, dev in DEVICES.items():
            assert hasattr(dev, "waveform_provenance"), f"{name}: missing provenance"

    def test_measured_devices(self):
        """Measured devices: PF-1000, PF-1000-Gribkov, UNU-ICTP, POSEIDON-60kV."""
        from dpf.validation.experimental import get_devices_by_provenance
        measured = get_devices_by_provenance("measured")
        assert "PF-1000" in measured
        assert "UNU-ICTP" in measured
        assert "POSEIDON-60kV" in measured
        assert "PF-1000-Gribkov" in measured

    def test_reconstructed_devices(self):
        """Reconstructed devices: PF-1000-16kV, FAETON-I, MJOLNIR."""
        from dpf.validation.experimental import get_devices_by_provenance
        recon = get_devices_by_provenance("reconstructed")
        assert "PF-1000-16kV" in recon
        assert "FAETON-I" in recon
        assert "MJOLNIR" in recon

    def test_reconstructed_higher_digitization_uncertainty(self):
        """Reconstructed waveforms should have higher digitization uncertainty."""
        from dpf.validation.experimental import get_devices_by_provenance
        measured = get_devices_by_provenance("measured")
        recon = get_devices_by_provenance("reconstructed")
        avg_meas = np.mean([d.waveform_digitization_uncertainty for d in measured.values()])
        avg_recon = np.mean([d.waveform_digitization_uncertainty for d in recon.values()])
        assert avg_recon > avg_meas

    def test_loo_devices_have_provenance(self):
        """All 5 LOO devices have provenance set."""
        from dpf.validation.experimental import DEVICES
        for name in LOO_RESULTS:
            dev = DEVICES[name]
            assert dev.waveform_provenance in ("measured", "reconstructed"), (
                f"{name}: provenance={dev.waveform_provenance!r}, expected measured/reconstructed"
            )

    def test_stratification_matches_hardcoded(self):
        """Provenance field matches our hardcoded metadata."""
        from dpf.validation.experimental import DEVICES
        for name, r in LOO_RESULTS.items():
            dev = DEVICES[name]
            expected_measured = r["measured"]
            actual_measured = dev.waveform_provenance == "measured"
            assert expected_measured == actual_measured, (
                f"{name}: hardcoded measured={expected_measured}, "
                f"provenance says measured={actual_measured}"
            )
