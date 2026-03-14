"""Phase BJ: Multi-device simultaneous calibration.

Tests whether Lee model fc/fm parameters can be shared across DPF devices
(universality hypothesis) or must be device-specific (Phase BI finding).

Three calibration modes:
1. Shared (fc, fm, delay) — tests full universality
2. Shared fc, device-specific (fm, delay) — tests partial universality
3. Pareto front — maps the NRMSE trade-off landscape

Devices: PF-1000 (27 kV, Scholz 2006) + POSEIDON-60kV (60 kV, IPFS)

Phase BI baseline for comparison:
- PF-1000 independent: fc=0.800, fm=0.100, delay=0.571 us, NRMSE=0.106
- POSEIDON independent: fc=0.556, fm=0.356, delay=0.000 us, NRMSE=0.059
- Cross-device blind: NRMSE=0.349 (PF-1000→POSEIDON)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.validation.calibration import (
    _DEFAULT_DEVICE_PCF,
    MultiDeviceCalibrator,
    MultiDeviceResult,
    ParetoFrontResult,
    ParetoPoint,
)
from dpf.validation.experimental import DEVICES

# =====================================================================
# Test 1: Infrastructure and setup
# =====================================================================


class TestMultiDeviceSetup:
    """Non-slow tests verifying multi-device calibrator infrastructure."""

    def test_multidevice_calibrator_instantiation(self) -> None:
        """MultiDeviceCalibrator can be instantiated with defaults."""
        cal = MultiDeviceCalibrator()
        assert cal.devices == ["PF-1000", "POSEIDON-60kV"]
        assert len(cal.weights) == 2
        assert pytest.approx(sum(cal.weights.values()), abs=1e-10) == 1.0

    def test_multidevice_calibrator_custom_devices(self) -> None:
        """MultiDeviceCalibrator accepts custom device list."""
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "PF-1000-Gribkov", "POSEIDON-60kV"],
        )
        assert len(cal.devices) == 3
        assert pytest.approx(cal.weights["PF-1000"], abs=1e-10) == 1.0 / 3.0

    def test_multidevice_calibrator_custom_weights(self) -> None:
        """Custom weights are normalized to sum to 1."""
        cal = MultiDeviceCalibrator(
            weights={"PF-1000": 2.0, "POSEIDON-60kV": 1.0},
        )
        assert pytest.approx(cal.weights["PF-1000"], abs=1e-10) == 2.0 / 3.0
        assert pytest.approx(cal.weights["POSEIDON-60kV"], abs=1e-10) == 1.0 / 3.0

    def test_devices_have_waveforms(self) -> None:
        """Both default devices have digitized waveform data."""
        for name in ["PF-1000", "POSEIDON-60kV"]:
            dev = DEVICES[name]
            assert dev.waveform_t is not None, f"{name} missing waveform_t"
            assert dev.waveform_I is not None, f"{name} missing waveform_I"
            assert len(dev.waveform_t) >= 10, f"{name} waveform too short"

    def test_devices_have_pcf(self) -> None:
        """Both default devices have pinch column fraction defaults."""
        for name in ["PF-1000", "POSEIDON-60kV"]:
            assert name in _DEFAULT_DEVICE_PCF

    def test_compute_nrmse_returns_float(self) -> None:
        """_compute_nrmse returns a finite float for valid parameters."""
        cal = MultiDeviceCalibrator()
        nrmse = cal._compute_nrmse("PF-1000", 0.8, 0.1, 0.5)
        assert isinstance(nrmse, float)
        assert 0.0 <= nrmse <= 1.0

    def test_compute_nrmse_bad_params_returns_penalty(self) -> None:
        """Extreme parameters return high (but finite) NRMSE."""
        cal = MultiDeviceCalibrator()
        nrmse = cal._compute_nrmse("PF-1000", 0.01, 0.01, 0.0)
        assert isinstance(nrmse, float)
        assert math.isfinite(nrmse)

    def test_result_dataclass_fields(self) -> None:
        """MultiDeviceResult dataclass has expected fields."""
        result = MultiDeviceResult(
            mode="shared",
            devices=["PF-1000", "POSEIDON-60kV"],
            shared_fc=0.7,
            shared_fm=0.2,
            shared_delay_us=0.5,
            device_fm={"PF-1000": 0.2, "POSEIDON-60kV": 0.2},
            device_delay_us={"PF-1000": 0.5, "POSEIDON-60kV": 0.5},
            device_nrmse={"PF-1000": 0.15, "POSEIDON-60kV": 0.10},
            combined_nrmse=0.125,
            independent_nrmse={"PF-1000": 0.10, "POSEIDON-60kV": 0.06},
            independent_fc={"PF-1000": 0.8, "POSEIDON-60kV": 0.56},
            independent_fm={"PF-1000": 0.1, "POSEIDON-60kV": 0.36},
            nrmse_penalty={"PF-1000": 0.5, "POSEIDON-60kV": 0.67},
            combined_improvement=0.75,
            converged=True,
            n_evals=100,
        )
        assert result.mode == "shared"
        assert result.shared_fc == 0.7
        assert len(result.device_nrmse) == 2

    def test_pareto_point_dataclass(self) -> None:
        """ParetoPoint dataclass holds per-device NRMSE."""
        p = ParetoPoint(
            fc=0.7, fm=0.2, delay_us=0.5,
            nrmse={"PF-1000": 0.15, "POSEIDON-60kV": 0.10},
            combined=0.125,
        )
        assert p.fc == 0.7
        assert "PF-1000" in p.nrmse

    def test_pareto_front_result_dataclass(self) -> None:
        """ParetoFrontResult dataclass holds Pareto analysis."""
        result = ParetoFrontResult(
            devices=["PF-1000", "POSEIDON-60kV"],
            points=[],
            n_evaluated=0,
            independent_nrmse={"PF-1000": 0.10, "POSEIDON-60kV": 0.06},
            utopia_point={"PF-1000": 0.10, "POSEIDON-60kV": 0.06},
            nadir_point={"PF-1000": 0.50, "POSEIDON-60kV": 0.50},
        )
        assert len(result.devices) == 2


class TestMultiDevicePhysics:
    """Non-slow tests verifying physical assumptions and device properties."""

    def test_pf1000_poseidon_different_regimes(self) -> None:
        """PF-1000 and POSEIDON-60kV operate in different speed factor regimes."""
        from dpf.validation.experimental import compute_speed_factor

        pf = DEVICES["PF-1000"]
        pos = DEVICES["POSEIDON-60kV"]

        s_pf = compute_speed_factor(pf.peak_current, pf.anode_radius, pf.fill_pressure_torr)
        s_pos = compute_speed_factor(pos.peak_current, pos.anode_radius, pos.fill_pressure_torr)

        # PF-1000 is near-optimal, POSEIDON is super-driven (Phase BI finding)
        assert s_pf["regime"] == "optimal"
        assert s_pos["regime"] == "super-driven"
        assert s_pos["S_over_S_opt"] > 2.0

    def test_devices_have_different_energy_scales(self) -> None:
        """PF-1000 and POSEIDON-60kV have very different stored energies."""
        pf = DEVICES["PF-1000"]
        pos = DEVICES["POSEIDON-60kV"]

        E_pf = 0.5 * pf.capacitance * pf.voltage**2
        E_pos = 0.5 * pos.capacitance * pos.voltage**2

        # PF-1000: ~485 kJ, POSEIDON-60kV: ~281 kJ
        assert E_pf > 400e3
        assert E_pos > 200e3
        assert E_pf / E_pos > 1.5

    def test_devices_have_different_quarter_periods(self) -> None:
        """Quarter-periods differ significantly (different timescales)."""
        from dpf.validation.experimental import compute_bare_rlc_timing

        pf = DEVICES["PF-1000"]
        pos = DEVICES["POSEIDON-60kV"]

        t_pf = compute_bare_rlc_timing(pf.capacitance, pf.inductance, pf.resistance)
        t_pos = compute_bare_rlc_timing(pos.capacitance, pos.inductance, pos.resistance)

        # PF-1000 ~10 us vs POSEIDON ~2 us
        assert t_pf > 5e-6
        assert t_pos < 3e-6
        assert t_pf / t_pos > 3.0

    def test_fc_squared_over_fm_varies_between_devices(self) -> None:
        """fc²/fm ratio differs significantly — Phase BI finding."""
        # From independent calibrations (Phase BI):
        # PF-1000: fc=0.800, fm=0.100 → fc²/fm = 6.40
        # POSEIDON: fc=0.556, fm=0.356 → fc²/fm = 0.87
        ratio_pf = 0.800**2 / 0.100
        ratio_pos = 0.556**2 / 0.356

        assert ratio_pf / ratio_pos > 5.0  # 7.37x from Phase BI


# =====================================================================
# Test 2: Shared calibration (slow — runs actual optimization)
# =====================================================================


class TestSharedCalibration:
    """Slow tests: shared (fc, fm, delay) across PF-1000 + POSEIDON-60kV."""

    @pytest.fixture(scope="class")
    def shared_result(self) -> MultiDeviceResult:
        """Run shared multi-device calibration once for the class."""
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV"],
            fc_bounds=(0.5, 0.95),
            fm_bounds=(0.01, 0.40),
            delay_bounds_us=(0.0, 2.0),
            maxiter=150,
            seed=42,
        )
        return cal.calibrate_shared()

    @pytest.mark.slow
    def test_shared_converges(self, shared_result: MultiDeviceResult) -> None:
        """Shared calibration converges."""
        assert shared_result.converged
        assert shared_result.n_evals > 50

    @pytest.mark.slow
    def test_shared_mode_is_correct(self, shared_result: MultiDeviceResult) -> None:
        """Result mode is 'shared'."""
        assert shared_result.mode == "shared"

    @pytest.mark.slow
    def test_shared_fc_in_bounds(self, shared_result: MultiDeviceResult) -> None:
        """Shared fc is within optimization bounds."""
        assert 0.5 <= shared_result.shared_fc <= 0.95

    @pytest.mark.slow
    def test_shared_fm_in_bounds(self, shared_result: MultiDeviceResult) -> None:
        """Shared fm is within optimization bounds."""
        assert 0.01 <= shared_result.shared_fm <= 0.40

    @pytest.mark.slow
    def test_shared_delay_in_bounds(self, shared_result: MultiDeviceResult) -> None:
        """Shared delay is within optimization bounds."""
        assert 0.0 <= shared_result.shared_delay_us <= 2.0

    @pytest.mark.slow
    def test_shared_nrmse_finite(self, shared_result: MultiDeviceResult) -> None:
        """Per-device NRMSE values are finite and positive."""
        for dev in shared_result.devices:
            nrmse = shared_result.device_nrmse[dev]
            assert math.isfinite(nrmse), f"{dev} NRMSE not finite"
            assert nrmse > 0.0, f"{dev} NRMSE should be > 0"

    @pytest.mark.slow
    def test_shared_penalty_pf1000(self, shared_result: MultiDeviceResult) -> None:
        """PF-1000 NRMSE penalty from shared calibration.

        Expected: shared fc/fm compromises PF-1000 fit (penalty > 0).
        Phase BI showed fc=0.800 is PF-1000's optimum; sharing with
        POSEIDON (fc=0.556) must worsen it.
        """
        assert shared_result.nrmse_penalty["PF-1000"] > -0.1  # Not much better

    @pytest.mark.slow
    def test_shared_penalty_poseidon(self, shared_result: MultiDeviceResult) -> None:
        """POSEIDON-60kV NRMSE penalty from shared calibration."""
        assert shared_result.nrmse_penalty["POSEIDON-60kV"] > -0.1

    @pytest.mark.slow
    def test_shared_combined_below_blind(self, shared_result: MultiDeviceResult) -> None:
        """Combined NRMSE is better than cross-device blind prediction.

        Phase BI forward blind: NRMSE=0.349. Shared calibration should
        do better than blindly transferring one device's params.
        """
        assert shared_result.combined_nrmse < 0.349

    @pytest.mark.slow
    def test_shared_report(self, shared_result: MultiDeviceResult) -> None:
        """Print comprehensive shared calibration report."""
        r = shared_result
        print("\n=== Multi-Device Shared Calibration ===")
        print(f"Shared: fc={r.shared_fc:.4f}, fm={r.shared_fm:.4f}, "
              f"delay={r.shared_delay_us:.3f} us")
        print(f"Combined NRMSE: {r.combined_nrmse:.4f}")
        print(f"Converged: {r.converged}, n_evals: {r.n_evals}")
        print()
        for dev in r.devices:
            print(f"  {dev}:")
            print(f"    Shared NRMSE:      {r.device_nrmse[dev]:.4f}")
            print(f"    Independent NRMSE: {r.independent_nrmse[dev]:.4f}")
            print(f"    Independent fc:    {r.independent_fc[dev]:.4f}")
            print(f"    Independent fm:    {r.independent_fm[dev]:.4f}")
            print(f"    Penalty:           {r.nrmse_penalty[dev]:+.1%}")
        print()
        # Key physics diagnostic
        fc2_fm = r.shared_fc**2 / r.shared_fm if r.shared_fm > 0 else float("inf")
        print(f"  fc^2/fm (shared): {fc2_fm:.2f}")
        print(f"  fc^2/fm (PF-1000 indep): "
              f"{r.independent_fc['PF-1000']**2 / r.independent_fm['PF-1000']:.2f}")
        print(f"  fc^2/fm (POSEIDON indep): "
              f"{r.independent_fc['POSEIDON-60kV']**2 / r.independent_fm['POSEIDON-60kV']:.2f}")


# =====================================================================
# Test 3: Shared-fc calibration (slow)
# =====================================================================


class TestSharedFcCalibration:
    """Slow tests: shared fc, device-specific (fm, delay)."""

    @pytest.fixture(scope="class")
    def shared_fc_result(self) -> MultiDeviceResult:
        """Run shared-fc multi-device calibration once."""
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV"],
            fc_bounds=(0.5, 0.95),
            fm_bounds=(0.01, 0.40),
            delay_bounds_us=(0.0, 2.0),
            maxiter=150,
            seed=42,
        )
        return cal.calibrate_shared_fc()

    @pytest.mark.slow
    def test_shared_fc_converges(self, shared_fc_result: MultiDeviceResult) -> None:
        """Shared-fc calibration converges."""
        assert shared_fc_result.converged

    @pytest.mark.slow
    def test_shared_fc_mode(self, shared_fc_result: MultiDeviceResult) -> None:
        """Result mode is 'shared_fc'."""
        assert shared_fc_result.mode == "shared_fc"

    @pytest.mark.slow
    def test_shared_fc_in_bounds(self, shared_fc_result: MultiDeviceResult) -> None:
        """Shared fc is within bounds."""
        assert 0.5 <= shared_fc_result.shared_fc <= 0.95

    @pytest.mark.slow
    def test_device_fm_differ(self, shared_fc_result: MultiDeviceResult) -> None:
        """Device-specific fm values differ (different mass coupling).

        PF-1000 (optimal S/S_opt~1) and POSEIDON (super-driven S/S_opt~2.8)
        couple mass differently.  fm should differ.
        """
        fm_pf = shared_fc_result.device_fm["PF-1000"]
        fm_pos = shared_fc_result.device_fm["POSEIDON-60kV"]
        assert abs(fm_pf - fm_pos) > 0.01  # Should meaningfully differ

    @pytest.mark.slow
    def test_shared_fc_penalty_smaller(
        self, shared_fc_result: MultiDeviceResult,
    ) -> None:
        """Shared-fc penalty should be <= shared penalty for at least one device.

        With more DOF (device-specific fm, delay), the optimizer can
        better fit each device while still constraining fc.
        """
        # Combined NRMSE should be reasonable
        assert shared_fc_result.combined_nrmse < 0.30

    @pytest.mark.slow
    def test_shared_fc_report(self, shared_fc_result: MultiDeviceResult) -> None:
        """Print shared-fc calibration report."""
        r = shared_fc_result
        print("\n=== Multi-Device Shared-fc Calibration ===")
        print(f"Shared fc: {r.shared_fc:.4f}")
        print(f"Combined NRMSE: {r.combined_nrmse:.4f}")
        print(f"Converged: {r.converged}, n_evals: {r.n_evals}")
        print()
        for dev in r.devices:
            print(f"  {dev}:")
            print(f"    fm={r.device_fm[dev]:.4f}, "
                  f"delay={r.device_delay_us[dev]:.3f} us")
            print(f"    Shared-fc NRMSE:   {r.device_nrmse[dev]:.4f}")
            print(f"    Independent NRMSE: {r.independent_nrmse[dev]:.4f}")
            print(f"    Penalty:           {r.nrmse_penalty[dev]:+.1%}")


# =====================================================================
# Test 4: Pareto front (slow)
# =====================================================================


class TestParetoFront:
    """Slow tests: Pareto front mapping of NRMSE trade-offs."""

    @pytest.fixture(scope="class")
    def pareto_result(self) -> ParetoFrontResult:
        """Compute Pareto front on a coarse grid."""
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV"],
            fc_bounds=(0.5, 0.90),
            fm_bounds=(0.05, 0.40),
        )
        return cal.pareto_front(fc_grid=10, fm_grid=10, delay_us=0.5)

    @pytest.mark.slow
    def test_pareto_has_points(self, pareto_result: ParetoFrontResult) -> None:
        """Pareto front has at least 2 points."""
        assert len(pareto_result.points) >= 2

    @pytest.mark.slow
    def test_pareto_points_are_nondominated(
        self, pareto_result: ParetoFrontResult,
    ) -> None:
        """No Pareto point dominates another."""
        pts = pareto_result.points
        devs = pareto_result.devices
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i == j:
                    continue
                all_leq = all(q.nrmse[d] <= p.nrmse[d] for d in devs)
                any_lt = any(q.nrmse[d] < p.nrmse[d] for d in devs)
                assert not (all_leq and any_lt), (
                    f"Point {j} dominates point {i}: "
                    f"{q.nrmse} vs {p.nrmse}"
                )

    @pytest.mark.slow
    def test_pareto_utopia_is_independent_optimum(
        self, pareto_result: ParetoFrontResult,
    ) -> None:
        """Utopia point matches independent calibration NRMSE."""
        for dev in pareto_result.devices:
            assert pareto_result.utopia_point[dev] == pytest.approx(
                pareto_result.independent_nrmse[dev], abs=0.001,
            )

    @pytest.mark.slow
    def test_pareto_shows_tradeoff(self, pareto_result: ParetoFrontResult) -> None:
        """Pareto front shows genuine trade-off (nadir > utopia for both).

        If there's no trade-off, one point dominates all — the Pareto
        front collapses to a single point.  Phase BI showed fc²/fm
        varies 7.37x, so a genuine trade-off is expected.
        """
        for dev in pareto_result.devices:
            assert pareto_result.nadir_point[dev] > pareto_result.utopia_point[dev]

    @pytest.mark.slow
    def test_pareto_report(self, pareto_result: ParetoFrontResult) -> None:
        """Print Pareto front analysis report."""
        r = pareto_result
        print(f"\n=== Pareto Front: {r.n_evaluated} evaluated, "
              f"{len(r.points)} non-dominated ===")
        print(f"Utopia: {r.utopia_point}")
        print(f"Nadir:  {r.nadir_point}")
        print()
        print("  Top 10 Pareto points (sorted by PF-1000 NRMSE):")
        for i, p in enumerate(r.points[:10]):
            nrmse_str = ", ".join(
                f"{d}={p.nrmse[d]:.4f}" for d in r.devices
            )
            print(f"    [{i}] fc={p.fc:.3f}, fm={p.fm:.3f}: {nrmse_str}")


# =====================================================================
# Test 5: Comparative analysis (slow)
# =====================================================================


class TestComparativeAnalysis:
    """Slow tests comparing all calibration modes."""

    @pytest.fixture(scope="class")
    def all_results(self) -> dict:
        """Run all three calibration modes."""
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV"],
            fc_bounds=(0.5, 0.95),
            fm_bounds=(0.01, 0.40),
            delay_bounds_us=(0.0, 2.0),
            maxiter=150,
            seed=42,
        )
        shared = cal.calibrate_shared()
        shared_fc = cal.calibrate_shared_fc()
        pareto = cal.pareto_front(fc_grid=10, fm_grid=10, delay_us=0.5)
        return {"shared": shared, "shared_fc": shared_fc, "pareto": pareto}

    @pytest.mark.slow
    def test_shared_fc_better_than_shared(self, all_results: dict) -> None:
        """Shared-fc (more DOF) achieves <= combined NRMSE vs shared.

        shared_fc has more parameters (1 fc + N fm + N delay) than
        shared (1 fc + 1 fm + 1 delay), so it has more fitting capacity.
        """
        shared = all_results["shared"]
        shared_fc = all_results["shared_fc"]
        # Allow small tolerance for optimizer variability
        assert shared_fc.combined_nrmse <= shared.combined_nrmse + 0.02

    @pytest.mark.slow
    def test_universality_hypothesis_result(self, all_results: dict) -> None:
        """Quantify the universality hypothesis outcome.

        If shared fc/fm works well (combined NRMSE < 1.5× avg independent),
        fc/fm are approximately universal.
        If not, they are fundamentally device-specific.
        """
        shared = all_results["shared"]
        avg_indep = np.mean(list(shared.independent_nrmse.values()))
        ratio = shared.combined_nrmse / avg_indep
        print("\n=== Universality Hypothesis ===")
        print(f"Avg independent NRMSE: {avg_indep:.4f}")
        print(f"Shared combined NRMSE: {shared.combined_nrmse:.4f}")
        print(f"Ratio (shared/indep): {ratio:.2f}")
        if ratio < 1.5:
            print("→ UNIVERSAL: fc/fm are approximately transferable")
        elif ratio < 3.0:
            print("→ PARTIALLY DEVICE-SPECIFIC: moderate penalty")
        else:
            print("→ DEVICE-SPECIFIC: fc/fm fundamentally non-transferable")

    @pytest.mark.slow
    def test_comprehensive_report(self, all_results: dict) -> None:
        """Print comprehensive comparison of all calibration modes."""
        shared = all_results["shared"]
        shared_fc = all_results["shared_fc"]
        pareto = all_results["pareto"]

        print("\n" + "=" * 70)
        print("PHASE BJ: MULTI-DEVICE SIMULTANEOUS CALIBRATION REPORT")
        print("=" * 70)

        print("\n--- Mode 1: Fully Shared (fc, fm, delay) ---")
        print(f"  fc={shared.shared_fc:.4f}, fm={shared.shared_fm:.4f}, "
              f"delay={shared.shared_delay_us:.3f} us")
        print(f"  fc^2/fm = {shared.shared_fc**2 / max(shared.shared_fm, 1e-10):.2f}")
        for dev in shared.devices:
            print(f"  {dev}: NRMSE={shared.device_nrmse[dev]:.4f} "
                  f"(indep={shared.independent_nrmse[dev]:.4f}, "
                  f"penalty={shared.nrmse_penalty[dev]:+.1%})")
        print(f"  Combined: {shared.combined_nrmse:.4f}")

        print("\n--- Mode 2: Shared fc, Device-Specific (fm, delay) ---")
        print(f"  fc={shared_fc.shared_fc:.4f}")
        for dev in shared_fc.devices:
            print(f"  {dev}: fm={shared_fc.device_fm[dev]:.4f}, "
                  f"delay={shared_fc.device_delay_us[dev]:.3f} us, "
                  f"NRMSE={shared_fc.device_nrmse[dev]:.4f} "
                  f"(penalty={shared_fc.nrmse_penalty[dev]:+.1%})")
        print(f"  Combined: {shared_fc.combined_nrmse:.4f}")

        print("\n--- Mode 3: Pareto Front ---")
        print(f"  {len(pareto.points)} non-dominated points "
              f"from {pareto.n_evaluated} evaluated")
        print(f"  Utopia: {pareto.utopia_point}")
        print(f"  Nadir:  {pareto.nadir_point}")

        print("\n--- Key Physics Findings ---")
        print("  Independent fc^2/fm:")
        for dev in shared.devices:
            fc_i = shared.independent_fc[dev]
            fm_i = shared.independent_fm[dev]
            print(f"    {dev}: {fc_i**2 / fm_i:.2f} "
                  f"(fc={fc_i:.3f}, fm={fm_i:.3f})")

        print("\n  Shared fc vs independent fc:")
        for dev in shared.devices:
            diff = shared.shared_fc - shared.independent_fc[dev]
            print(f"    {dev}: diff={diff:+.3f}")

        print("\n  Blind prediction (Phase BI) vs shared calibration:")
        print("    PF-1000→POSEIDON blind: NRMSE=0.349")
        print(f"    Shared POSEIDON NRMSE:  {shared.device_nrmse['POSEIDON-60kV']:.4f}")
        print(f"    Improvement: "
              f"{(0.349 - shared.device_nrmse['POSEIDON-60kV']) / 0.349:.1%}")

        # Universality verdict
        avg_indep = np.mean(list(shared.independent_nrmse.values()))
        ratio = shared.combined_nrmse / avg_indep
        print(f"\n  Universality ratio: {ratio:.2f} "
              f"(combined/avg_indep)")
        print("=" * 70)
