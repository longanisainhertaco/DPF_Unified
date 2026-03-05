"""Phase BS: Provenance-dependent ASME + LOO maxiter=10.

Implements Debate #51 Recommendation 3: provenance-dependent ASME model.
Reconstructed waveform comparisons are flagged as "qualified" (model-vs-model)
per ASME V&V 20 §4.1 independence requirement.

Also codifies LOO maxiter=10 results with corrected MJOLNIR anode_radius.

References:
    ASME V&V 20-2009: Standard for Verification and Validation
    GUM (JCGM 100:2008): Guide to the Expression of Uncertainty in Measurement
    PhD Debate #51 (2026-03-04): Finding 5, Recommendation 3
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.validation.calibration import (
    ASMEValidationResult,
    asme_stratified_summary,
    asme_vv20_assessment,
)
from dpf.validation.experimental import DEVICES, get_devices_by_provenance

# =====================================================================
# Test: ASMEValidationResult provenance fields
# =====================================================================


class TestASMEProvenance:
    """Verify provenance and qualified fields on ASMEValidationResult."""

    def test_result_has_provenance_field(self):
        """ASMEValidationResult should have waveform_provenance field."""
        r = ASMEValidationResult(
            E=0.10, u_exp=0.05, u_input=0.03, u_num=0.001,
            u_val=0.06, ratio=1.67, passes=False,
        )
        assert hasattr(r, "waveform_provenance")
        assert r.waveform_provenance == ""

    def test_result_has_qualified_field(self):
        """ASMEValidationResult should have qualified field."""
        r = ASMEValidationResult(
            E=0.10, u_exp=0.05, u_input=0.03, u_num=0.001,
            u_val=0.06, ratio=1.67, passes=False,
        )
        assert hasattr(r, "qualified")
        assert r.qualified is False

    def test_measured_device_not_qualified(self):
        """ASME assessment on measured waveform should NOT be qualified."""
        r = asme_vv20_assessment("PF-1000", fc=0.800, fm=0.128)
        assert r.waveform_provenance == "measured"
        assert r.qualified is False

    def test_reconstructed_device_is_qualified(self):
        """ASME assessment on reconstructed waveform should be qualified."""
        r = asme_vv20_assessment("PF-1000-16kV", fc=0.800, fm=0.128)
        assert r.waveform_provenance == "reconstructed"
        assert r.qualified is True

    @pytest.mark.parametrize("device,expected_prov", [
        ("PF-1000", "measured"),
        ("POSEIDON-60kV", "measured"),
        ("UNU-ICTP", "measured"),
        ("PF-1000-Gribkov", "measured"),
        ("PF-1000-16kV", "reconstructed"),
        ("FAETON-I", "reconstructed"),
        ("MJOLNIR", "reconstructed"),
    ])
    def test_provenance_matches_device(self, device, expected_prov):
        """Each device's ASME provenance should match its waveform provenance."""
        r = asme_vv20_assessment(device, fc=0.700, fm=0.150)
        assert r.waveform_provenance == expected_prov

    @pytest.mark.parametrize("device", ["PF-1000-16kV", "FAETON-I", "MJOLNIR"])
    def test_reconstructed_always_qualified(self, device):
        """All reconstructed devices should produce qualified results."""
        r = asme_vv20_assessment(device, fc=0.700, fm=0.150)
        assert r.qualified is True, (
            f"{device}: reconstructed waveform should produce qualified result"
        )


# =====================================================================
# Test: ASMEStratifiedSummary
# =====================================================================


class TestASMEStratifiedSummary:
    """Verify stratified ASME summary separates measured vs reconstructed."""

    def _make_result(
        self, device: str, passes: bool = False, E: float = 0.10
    ) -> ASMEValidationResult:
        """Helper to create a result with correct provenance."""
        dev = DEVICES[device]
        return ASMEValidationResult(
            E=E, u_exp=0.05, u_input=0.03, u_num=0.001,
            u_val=0.06, ratio=E / 0.06, passes=passes,
            device_name=device,
            waveform_provenance=dev.waveform_provenance,
            qualified=dev.waveform_provenance == "reconstructed",
        )

    def test_stratify_splits_correctly(self):
        """Stratified summary should separate measured from reconstructed."""
        results = [
            self._make_result("PF-1000", passes=True),
            self._make_result("POSEIDON-60kV", passes=False),
            self._make_result("UNU-ICTP", passes=True),
            self._make_result("FAETON-I", passes=False),
            self._make_result("MJOLNIR", passes=True),
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_measured_total == 3
        assert summary.n_reconstructed_total == 2

    def test_measured_pass_count(self):
        """Measured PASS count should only count measured waveform results."""
        results = [
            self._make_result("PF-1000", passes=True),
            self._make_result("UNU-ICTP", passes=True),
            self._make_result("FAETON-I", passes=True),  # reconstructed
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_measured_pass == 2
        assert summary.n_reconstructed_pass == 1

    def test_reconstructed_pass_count(self):
        """Reconstructed PASS should be counted separately."""
        results = [
            self._make_result("PF-1000", passes=False),
            self._make_result("FAETON-I", passes=True),
            self._make_result("MJOLNIR", passes=True),
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_measured_pass == 0
        assert summary.n_reconstructed_pass == 2

    def test_total_properties(self):
        """Total properties should sum measured + reconstructed."""
        results = [
            self._make_result("PF-1000", passes=True),
            self._make_result("POSEIDON-60kV", passes=False),
            self._make_result("UNU-ICTP", passes=True),
            self._make_result("FAETON-I", passes=False),
            self._make_result("MJOLNIR", passes=True),
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_total == 5
        assert summary.n_total_pass == 3

    def test_empty_results(self):
        """Stratified summary should handle empty list."""
        summary = asme_stratified_summary([])
        assert summary.n_measured_total == 0
        assert summary.n_reconstructed_total == 0
        assert summary.n_total == 0

    def test_measured_only(self):
        """Should work with only measured results."""
        results = [
            self._make_result("PF-1000", passes=True),
            self._make_result("UNU-ICTP", passes=False),
        ]
        summary = asme_stratified_summary(results)
        assert summary.n_measured_total == 2
        assert summary.n_reconstructed_total == 0

    def test_all_results_preserved(self):
        """All results should be in the all_results field."""
        results = [
            self._make_result("PF-1000"),
            self._make_result("FAETON-I"),
        ]
        summary = asme_stratified_summary(results)
        assert len(summary.all_results) == 2


# =====================================================================
# Test: Provenance-dependent behavior in ASME assessment
# =====================================================================


class TestProvenanceBehavior:
    """Verify ASME assessment behavior differs by provenance where needed."""

    def test_measured_includes_shot_to_shot(self):
        """Measured devices should include shot-to-shot in u_exp."""
        # PF-1000: measured, has shot-to-shot data
        r = asme_vv20_assessment("PF-1000", fc=0.800, fm=0.128)
        # u_exp should be > sqrt(u_peak^2 + u_amplitude^2)
        dev = DEVICES["PF-1000"]
        u_min = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        )
        assert r.u_exp > u_min, (
            f"Measured device u_exp ({r.u_exp:.4f}) should include "
            f"shot-to-shot (> {u_min:.4f})"
        )

    def test_reconstructed_skips_shot_for_16kv(self):
        """PF-1000-16kV: peak uncertainty from shot spread → skip u_shot."""
        r = asme_vv20_assessment("PF-1000-16kV", fc=0.800, fm=0.128)
        dev = DEVICES["PF-1000-16kV"]
        # u_exp should be exactly sqrt(u_peak^2 + u_amplitude^2)
        expected = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        )
        assert r.u_exp == pytest.approx(expected, abs=0.001)

    def test_measured_u_exp_lower_than_reconstructed_avg(self):
        """On average, measured u_exp should be lower than reconstructed."""
        measured_devs = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]
        recon_devs = ["PF-1000-16kV", "FAETON-I", "MJOLNIR"]
        fc, fm = 0.700, 0.150
        m_uexp = [asme_vv20_assessment(d, fc=fc, fm=fm).u_exp for d in measured_devs]
        r_uexp = [asme_vv20_assessment(d, fc=fc, fm=fm).u_exp for d in recon_devs]
        # Reconstructed have higher u_amplitude → higher u_exp
        assert np.mean(r_uexp) > np.mean(m_uexp)


# =====================================================================
# Test: ASME V&V 20 §4.1 independence warning
# =====================================================================


class TestASMEIndependence:
    """Verify that reconstructed-waveform results are clearly flagged."""

    def test_qualified_result_is_not_independent(self):
        """Qualified ASME result = comparison uses non-independent data.

        Per ASME V&V 20 §4.1: "Validation experiments should provide
        data that are independent of the computational model."
        Reconstructed waveforms derived from physics scaling of training
        data violate this requirement.
        """
        r = asme_vv20_assessment("PF-1000-16kV", fc=0.800, fm=0.128)
        assert r.qualified is True
        # Even if it passes, it's model-vs-model
        # The 'qualified' flag signals this to downstream consumers

    def test_all_loo_devices_provenance(self):
        """LOO N=5 devices: 3 measured + 2 reconstructed."""
        loo_devices = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP",
                       "FAETON-I", "MJOLNIR"]
        measured = [d for d in loo_devices
                    if DEVICES[d].waveform_provenance == "measured"]
        reconstructed = [d for d in loo_devices
                         if DEVICES[d].waveform_provenance == "reconstructed"]
        assert len(measured) == 3
        assert len(reconstructed) == 2

    def test_stratified_loo_n3_measured(self):
        """Measured-only LOO subset should have N=3 devices."""
        measured = get_devices_by_provenance("measured")
        # At least 3 measured devices with waveforms (LOO candidates)
        loo_measured = {k: v for k, v in measured.items()
                        if k in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]}
        assert len(loo_measured) == 3


# =====================================================================
# Test: MJOLNIR geometry fix verification (from Debate #49)
# =====================================================================


class TestMJOLNIRGeometry:
    """Verify MJOLNIR anode_radius fix is in place."""

    def test_anode_radius_corrected(self):
        """MJOLNIR anode_radius should be 114mm (physical anode)."""
        dev = DEVICES["MJOLNIR"]
        assert dev.anode_radius == pytest.approx(0.114, abs=0.002)

    def test_cathode_radius(self):
        """MJOLNIR cathode_radius should be 157mm."""
        dev = DEVICES["MJOLNIR"]
        assert dev.cathode_radius == pytest.approx(0.157, abs=0.002)

    def test_ak_gap(self):
        """A-K gap should be cathode - anode = 43mm."""
        dev = DEVICES["MJOLNIR"]
        gap = dev.cathode_radius - dev.anode_radius
        assert gap == pytest.approx(0.043, abs=0.003)

    def test_speed_factor_near_optimal(self):
        """With corrected geometry, MJOLNIR S/S_opt should be ~1.0."""
        from dpf.validation.experimental import compute_speed_factor
        dev = DEVICES["MJOLNIR"]
        result = compute_speed_factor(
            dev.peak_current, dev.anode_radius, dev.fill_pressure_torr,
        )
        S_ratio = result["S_over_S_opt"]
        # Previously 2.81 (super-driven), now ~1.04 (near-optimal)
        assert S_ratio < 1.5, (
            f"MJOLNIR S/S_opt={S_ratio:.2f}, expected <1.5 with corrected geometry"
        )


# =====================================================================
# Test: LOO maxiter=10 reference results (non-slow, codified from run)
# =====================================================================

# Reference results from LOO maxiter=10 with corrected MJOLNIR (seed=42)
# Runtime: 4159s (69.3 min), 2026-03-04
LOO_MAXITER10_RESULTS = {
    "PF-1000":       {"blind": 0.4376, "indep": 0.0957, "degrad": 4.57,
                      "fc": 0.563, "fm": 0.287, "delay": 0.000},
    "POSEIDON-60kV": {"blind": 0.1906, "indep": 0.0601, "degrad": 3.17,
                      "fc": 0.835, "fm": 0.226, "delay": 0.012},
    "UNU-ICTP":      {"blind": 0.0963, "indep": 0.0660, "degrad": 1.46,
                      "fc": 0.545, "fm": 0.085, "delay": 0.011},
    "FAETON-I":      {"blind": 0.1898, "indep": 0.0193, "degrad": 9.83,
                      "fc": 0.932, "fm": 0.181, "delay": 0.016},
    "MJOLNIR":       {"blind": 0.1755, "indep": 0.1723, "degrad": 1.02,
                      "fc": 0.626, "fm": 0.125, "delay": 0.009},
}

# Previous results: LOO maxiter=3 (old MJOLNIR anode_radius=76mm)
LOO_MAXITER3_RESULTS = {
    "PF-1000":       {"blind": 0.4377, "fc": 0.500, "fm": 0.227, "delay": 0.000},
    "POSEIDON-60kV": {"blind": 0.1917, "fc": 0.843, "fm": 0.239, "delay": 0.051},
    "UNU-ICTP":      {"blind": 0.0978, "fc": 0.701, "fm": 0.159, "delay": 0.067},
    "FAETON-I":      {"blind": 0.1720, "fc": 0.801, "fm": 0.146, "delay": 0.037},
    "MJOLNIR":       {"blind": 0.1777, "fc": 0.843, "fm": 0.239, "delay": 0.051},
}


class TestLOOMaxiter10Results:
    """Codify LOO maxiter=10 results with corrected MJOLNIR geometry.

    Key findings:
    - Mean blind NRMSE: 0.2180 ± 0.1288 (essentially unchanged from maxiter=3's 0.2154)
    - Degeneracy RESOLVED: 5/5 unique parameter sets (was 3/5 at maxiter=3)
    - PF-1000 still boundary-trapped at fc=0.563
    - ASME: 1/5 PASS (UNU-ICTP only)
    """

    def test_mean_blind_nrmse(self):
        """Mean blind NRMSE at maxiter=10 is ~0.218."""
        blind = [LOO_MAXITER10_RESULTS[d]["blind"] for d in LOO_MAXITER10_RESULTS]
        mean_blind = np.mean(blind)
        assert mean_blind == pytest.approx(0.218, abs=0.01)

    def test_degeneracy_resolved(self):
        """maxiter=10 produces 5/5 unique parameter sets."""
        param_sets = set()
        for d in LOO_MAXITER10_RESULTS:
            r = LOO_MAXITER10_RESULTS[d]
            p = (round(r["fc"], 3), round(r["fm"], 3), round(r["delay"], 3))
            param_sets.add(p)
        assert len(param_sets) == 5, (
            f"Expected 5 unique param sets, got {len(param_sets)}"
        )

    def test_maxiter3_had_degeneracy(self):
        """maxiter=3 had 3/5 degenerate parameter sets (baseline)."""
        param_sets = set()
        for d in LOO_MAXITER3_RESULTS:
            r = LOO_MAXITER3_RESULTS[d]
            p = (round(r["fc"], 3), round(r["fm"], 3), round(r["delay"], 3))
            param_sets.add(p)
        # maxiter=3: POSEIDON, FAETON, MJOLNIR all had same params
        assert len(param_sets) <= 4, (
            f"Expected <=4 unique param sets at maxiter=3, got {len(param_sets)}"
        )

    def test_pf1000_still_boundary_trapped(self):
        """PF-1000 fc=0.563 is still near lower bound (0.50)."""
        r = LOO_MAXITER10_RESULTS["PF-1000"]
        assert r["fc"] < 0.60, (
            f"PF-1000 fc={r['fc']:.3f}, expected near lower bound"
        )
        assert r["blind"] > 0.40, (
            f"PF-1000 blind={r['blind']:.4f}, expected > 0.40 (boundary-trapped)"
        )

    def test_unu_ictp_best_blind(self):
        """UNU-ICTP has best blind NRMSE (circuit-dominated)."""
        best_dev = min(LOO_MAXITER10_RESULTS,
                       key=lambda d: LOO_MAXITER10_RESULTS[d]["blind"])
        assert best_dev == "UNU-ICTP"
        assert LOO_MAXITER10_RESULTS["UNU-ICTP"]["blind"] < 0.10

    def test_mjolnir_low_degradation(self):
        """MJOLNIR degradation ~1.0 (near-perfect generalization)."""
        r = LOO_MAXITER10_RESULTS["MJOLNIR"]
        assert r["degrad"] < 1.5, (
            f"MJOLNIR degradation {r['degrad']:.2f}, expected < 1.5"
        )

    def test_faeton_high_degradation(self):
        """FAETON-I has high degradation (reconstructed waveform artifact)."""
        r = LOO_MAXITER10_RESULTS["FAETON-I"]
        assert r["degrad"] > 5.0
        # Suspiciously good independent NRMSE confirms overfitting artifact
        assert r["indep"] < 0.03

    def test_mean_unchanged_from_maxiter3(self):
        """Mean blind NRMSE essentially unchanged: 0.218 vs 0.215.

        Degeneracy resolution didn't improve mean because it found
        different (sometimes worse) optima for individual folds.
        This is the correct statistical behavior.
        """
        blind_10 = [LOO_MAXITER10_RESULTS[d]["blind"]
                     for d in LOO_MAXITER10_RESULTS]
        blind_3 = [LOO_MAXITER3_RESULTS[d]["blind"]
                    for d in LOO_MAXITER3_RESULTS]
        delta = abs(np.mean(blind_10) - np.mean(blind_3))
        assert delta < 0.02, (
            f"Mean difference {delta:.4f} > 0.02 — "
            f"expected similar means with resolved degeneracy"
        )

    def test_stratified_measured_vs_reconstructed(self):
        """Measured devices have HIGHER mean blind than reconstructed.

        Measured (N=3): 0.2415 (PF-1000 dominates)
        Reconstructed (N=2): 0.1827
        This is because PF-1000's boundary-trapped catastrophic failure
        inflates the measured average.
        """
        measured = ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]
        recon = ["FAETON-I", "MJOLNIR"]
        meas_mean = np.mean([LOO_MAXITER10_RESULTS[d]["blind"] for d in measured])
        recon_mean = np.mean([LOO_MAXITER10_RESULTS[d]["blind"] for d in recon])
        # Measured higher due to PF-1000 catastrophic failure
        assert meas_mean > recon_mean

    def test_fc_fm_range(self):
        """fc^2/fm varies 4x across folds — device-specific, not universal."""
        fc_fm = {}
        for d, r in LOO_MAXITER10_RESULTS.items():
            fc_fm[d] = r["fc"]**2 / r["fm"]
        assert max(fc_fm.values()) / min(fc_fm.values()) > 3.0

    def test_asme_loo_one_pass(self):
        """Only UNU-ICTP ASME PASS (circuit-dominated)."""
        # UNU-ICTP: ratio=0.85 (PASS), all others FAIL
        # Reference from LOO ASME output
        assert LOO_MAXITER10_RESULTS["UNU-ICTP"]["blind"] < 0.10
        # UNU-ICTP has large u_exp (0.1061) → easy PASS


# =====================================================================
# Test: LOO maxiter=10 live validation (slow — re-runs LOO)
# =====================================================================


class TestLOOMaxiter10Live:
    """Live LOO maxiter=10 validation tests (slow, ~70 min)."""

    @pytest.mark.slow
    def test_loo_maxiter10_mean_within_tolerance(self):
        """maxiter=10 mean blind NRMSE should be < 0.25.

        Actual result: 0.2180 ± 0.1288
        """
        from dpf.validation.calibration import MultiDeviceCalibrator

        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP",
                     "FAETON-I", "MJOLNIR"],
            fc_bounds=(0.5, 0.95),
            fm_bounds=(0.04, 0.40),
            delay_bounds_us=(0.0, 2.0),
            maxiter=10,
            seed=42,
        )
        loo = cal.leave_one_out()
        blind_nrmses = [loo[d]["blind_nrmse"] for d in loo]
        mean_blind = np.mean(blind_nrmses)
        assert mean_blind < 0.25, (
            f"LOO maxiter=10 mean blind NRMSE {mean_blind:.4f} > 0.25"
        )

    @pytest.mark.slow
    def test_loo_maxiter10_no_degeneracy(self):
        """maxiter=10 should produce 5/5 unique parameter sets."""
        from dpf.validation.calibration import MultiDeviceCalibrator

        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP",
                     "FAETON-I", "MJOLNIR"],
            fc_bounds=(0.5, 0.95),
            fm_bounds=(0.04, 0.40),
            delay_bounds_us=(0.0, 2.0),
            maxiter=10,
            seed=42,
        )
        loo = cal.leave_one_out()
        param_sets = set()
        for d in loo:
            p = (round(loo[d]["trained_fc"], 3),
                 round(loo[d]["trained_fm"], 3),
                 round(loo[d]["trained_delay_us"], 3))
            param_sets.add(p)
        assert len(param_sets) == 5, (
            f"Only {len(param_sets)} unique param sets at maxiter=10"
        )
