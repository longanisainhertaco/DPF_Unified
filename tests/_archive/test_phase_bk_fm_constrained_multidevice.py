"""Phase BK: fm-constrained multi-device calibration.

Combines Phase BF (fm >= 0.10 physical constraint) with Phase BJ
(multi-device simultaneous calibration) to test whether physically
reasonable parameters can fit both PF-1000 and POSEIDON-60kV
simultaneously.

Phase BJ baseline (unconstrained, fm_bounds=(0.01, 0.40)):
- Mode 1 shared: fc=0.880, fm=0.146, delay=0.065 us, combined=0.189
- Mode 2 shared_fc: fc=0.547, combined=0.080 (PF-1000 fm=0.037 NON-PHYSICAL)
- Independent: PF-1000 fc=0.914, fm=0.104; POSEIDON fc=0.556, fm=0.355

Key question: Does constraining fm >= 0.10 (published Lee & Saw range)
produce acceptable NRMSE while eliminating non-physical parameters?

Also includes leave-one-out cross-validation (2-device) infrastructure.
"""

from __future__ import annotations

import math

import pytest

from dpf.validation.calibration import (
    MultiDeviceCalibrator,
    MultiDeviceResult,
)

# =====================================================================
# Constants from Phase BJ unconstrained results (for comparison)
# =====================================================================
BJ_MODE1_COMBINED = 0.189  # Unconstrained Mode 1
BJ_MODE2_COMBINED = 0.080  # Unconstrained Mode 2
BJ_INDEP_PF1000_FM = 0.104  # Already above 0.10 threshold
BJ_INDEP_POSEIDON_FM = 0.355
BJ_SHARED_FC = 0.547  # Phase BJ Mode 2 shared fc

FM_MIN = 0.10  # Physical lower bound (Lee & Saw 2014)


# =====================================================================
# Module-scoped fixtures: run all heavy calibrations ONCE
# =====================================================================

@pytest.fixture(scope="module")
def constrained_mode1() -> MultiDeviceResult:
    """Mode 1 (shared fc, fm, delay) with fm >= 0.10."""
    cal = MultiDeviceCalibrator(
        fm_bounds=(FM_MIN, 0.40),
        maxiter=150,
        seed=42,
    )
    return cal.calibrate_shared()


@pytest.fixture(scope="module")
def constrained_mode2() -> MultiDeviceResult:
    """Mode 2 (shared fc, device-specific fm/delay) with fm >= 0.10."""
    cal = MultiDeviceCalibrator(
        fm_bounds=(FM_MIN, 0.40),
        maxiter=150,
        seed=42,
    )
    return cal.calibrate_shared_fc()


@pytest.fixture(scope="module")
def unconstrained_mode2() -> MultiDeviceResult:
    """Mode 2 (shared fc, device-specific fm/delay) UNCONSTRAINED."""
    cal = MultiDeviceCalibrator(
        fm_bounds=(0.01, 0.40),
        maxiter=150,
        seed=42,
    )
    return cal.calibrate_shared_fc()


@pytest.fixture(scope="module")
def loo_constrained() -> dict[str, dict[str, float]]:
    """Leave-one-out with fm >= 0.10 constraint."""
    cal = MultiDeviceCalibrator(
        fm_bounds=(FM_MIN, 0.40),
        maxiter=100,  # Reduced for LOO (sub-calibrations are cheaper)
        seed=42,
    )
    return cal.leave_one_out()


# =====================================================================
# Test 1: Infrastructure (non-slow)
# =====================================================================


class TestFmConstrainedSetup:
    """Non-slow tests verifying fm-constrained calibrator setup."""

    def test_constrained_instantiation(self) -> None:
        """MultiDeviceCalibrator accepts fm_bounds=(0.10, 0.40)."""
        cal = MultiDeviceCalibrator(fm_bounds=(FM_MIN, 0.40))
        assert cal.fm_bounds == (FM_MIN, 0.40)
        assert cal.devices == ["PF-1000", "POSEIDON-60kV"]

    def test_constrained_fm_lower_bound_enforced(self) -> None:
        """fm lower bound is FM_MIN, not the unconstrained 0.01."""
        cal = MultiDeviceCalibrator(fm_bounds=(FM_MIN, 0.40))
        assert cal.fm_bounds[0] == FM_MIN

    def test_leave_one_out_requires_two_devices(self) -> None:
        """leave_one_out() raises with <2 devices."""
        cal = MultiDeviceCalibrator(devices=["PF-1000"])
        with pytest.raises(ValueError, match="Need >= 2"):
            cal.leave_one_out()

    def test_leave_one_out_method_exists(self) -> None:
        """MultiDeviceCalibrator has leave_one_out() method."""
        cal = MultiDeviceCalibrator()
        assert hasattr(cal, "leave_one_out")
        assert callable(cal.leave_one_out)


# =====================================================================
# Test 2: Mode 1 shared with fm >= 0.10
# =====================================================================


class TestFmConstrainedShared:
    """Slow tests for Mode 1 (fully shared) with fm >= 0.10."""

    @pytest.mark.slow
    def test_converged(self, constrained_mode1: MultiDeviceResult) -> None:
        """Optimizer converges (3D problem, maxiter=150 sufficient)."""
        assert constrained_mode1.converged

    @pytest.mark.slow
    def test_fm_above_threshold(
        self, constrained_mode1: MultiDeviceResult,
    ) -> None:
        """Shared fm >= FM_MIN."""
        assert constrained_mode1.shared_fm >= FM_MIN - 1e-6

    @pytest.mark.slow
    def test_combined_nrmse_acceptable(
        self, constrained_mode1: MultiDeviceResult,
    ) -> None:
        """Combined NRMSE < 0.30."""
        assert constrained_mode1.combined_nrmse < 0.30

    @pytest.mark.slow
    def test_nrmse_close_to_unconstrained(
        self, constrained_mode1: MultiDeviceResult,
    ) -> None:
        """Constrained NRMSE within 20% of unconstrained.

        Mode 1 unconstrained fm=0.146 is already > 0.10, so the
        constraint shouldn't bind much.
        """
        assert constrained_mode1.combined_nrmse < 1.20 * BJ_MODE1_COMBINED

    @pytest.mark.slow
    def test_fc_in_bounds(
        self, constrained_mode1: MultiDeviceResult,
    ) -> None:
        """Shared fc in optimizer bounds."""
        assert 0.5 <= constrained_mode1.shared_fc <= 0.95

    @pytest.mark.slow
    def test_fc_squared_over_fm(
        self, constrained_mode1: MultiDeviceResult,
    ) -> None:
        """fc^2/fm ratio is finite and positive."""
        ratio = constrained_mode1.shared_fc ** 2 / constrained_mode1.shared_fm
        assert math.isfinite(ratio) and ratio > 0

    @pytest.mark.slow
    def test_report(self, constrained_mode1: MultiDeviceResult) -> None:
        """Print Mode 1 constrained results."""
        r = constrained_mode1
        print(f"\n=== Mode 1 Constrained (fm >= {FM_MIN}) ===")
        print(f"Shared: fc={r.shared_fc:.4f}, fm={r.shared_fm:.4f}, "
              f"delay={r.shared_delay_us:.3f} us")
        print(f"Combined NRMSE: {r.combined_nrmse:.4f} "
              f"(unconstrained: {BJ_MODE1_COMBINED:.4f})")
        print(f"fc^2/fm = {r.shared_fc**2/r.shared_fm:.2f}")
        for d in r.devices:
            print(f"  {d}: NRMSE={r.device_nrmse[d]:.4f} "
                  f"(indep={r.independent_nrmse[d]:.4f}, "
                  f"penalty={r.nrmse_penalty[d]:+.1%})")


# =====================================================================
# Test 3: Mode 2 shared-fc with fm >= 0.10
# =====================================================================


class TestFmConstrainedSharedFc:
    """Slow tests for Mode 2 (shared fc) with fm >= 0.10."""

    @pytest.mark.slow
    def test_result_valid(
        self, constrained_mode2: MultiDeviceResult,
    ) -> None:
        """Result is usable (5D may not converge in 150 iter)."""
        assert constrained_mode2.combined_nrmse < 0.50

    @pytest.mark.slow
    def test_all_device_fm_above_threshold(
        self, constrained_mode2: MultiDeviceResult,
    ) -> None:
        """All device-specific fm values >= FM_MIN."""
        for d in constrained_mode2.devices:
            fm_d = constrained_mode2.device_fm[d]
            assert fm_d >= FM_MIN - 1e-6, (
                f"{d}: fm={fm_d:.4f} below {FM_MIN}"
            )

    @pytest.mark.slow
    def test_pf1000_fm_physical(
        self, constrained_mode2: MultiDeviceResult,
    ) -> None:
        """PF-1000 fm >= 0.10, NOT the unconstrained 0.037."""
        fm_pf = constrained_mode2.device_fm["PF-1000"]
        assert fm_pf >= FM_MIN - 1e-6

    @pytest.mark.slow
    def test_combined_nrmse_bounded(
        self, constrained_mode2: MultiDeviceResult,
    ) -> None:
        """Combined NRMSE with fm constraint < 0.20."""
        assert constrained_mode2.combined_nrmse < 0.20

    @pytest.mark.slow
    def test_nrmse_penalty_vs_unconstrained(
        self, constrained_mode2: MultiDeviceResult,
    ) -> None:
        """Constrained < 3x unconstrained (generous for physical constraint)."""
        assert constrained_mode2.combined_nrmse < 3.0 * BJ_MODE2_COMBINED

    @pytest.mark.slow
    def test_shared_fc_in_bounds(
        self, constrained_mode2: MultiDeviceResult,
    ) -> None:
        """Shared fc within optimizer bounds."""
        assert 0.5 <= constrained_mode2.shared_fc <= 0.95

    @pytest.mark.slow
    def test_report(
        self, constrained_mode2: MultiDeviceResult,
    ) -> None:
        """Print Mode 2 constrained results."""
        r = constrained_mode2
        print(f"\n=== Mode 2 Constrained (fm >= {FM_MIN}) ===")
        print(f"Shared fc: {r.shared_fc:.4f} "
              f"(unconstrained: {BJ_SHARED_FC:.4f})")
        print(f"Combined NRMSE: {r.combined_nrmse:.4f} "
              f"(unconstrained: {BJ_MODE2_COMBINED:.4f})")
        for d in r.devices:
            fm_d = r.device_fm[d]
            delay_d = r.device_delay_us[d]
            print(f"  {d}: fm={fm_d:.4f}, delay={delay_d:.3f} us, "
                  f"NRMSE={r.device_nrmse[d]:.4f} "
                  f"(penalty={r.nrmse_penalty[d]:+.1%})")


# =====================================================================
# Test 4: Leave-one-out cross-validation
# =====================================================================


class TestLeaveOneOut:
    """Slow tests for leave-one-out cross-validation."""

    @pytest.mark.slow
    def test_loo_returns_both_devices(
        self, loo_constrained: dict,
    ) -> None:
        """LOO returns results for both held-out devices."""
        assert "PF-1000" in loo_constrained
        assert "POSEIDON-60kV" in loo_constrained

    @pytest.mark.slow
    def test_loo_has_required_keys(
        self, loo_constrained: dict,
    ) -> None:
        """LOO results have all required keys."""
        required = {
            "train_nrmse", "blind_nrmse", "independent_nrmse",
            "degradation", "trained_fc", "trained_fm", "trained_delay_us",
        }
        for _dev, result in loo_constrained.items():
            assert required.issubset(result.keys())

    @pytest.mark.slow
    def test_loo_blind_finite(
        self, loo_constrained: dict,
    ) -> None:
        """All LOO blind NRMSE values are finite."""
        for _dev, result in loo_constrained.items():
            assert math.isfinite(result["blind_nrmse"])

    @pytest.mark.slow
    def test_loo_fm_physical(
        self, loo_constrained: dict,
    ) -> None:
        """LOO with fm constraint produces physical fm values."""
        for dev, result in loo_constrained.items():
            assert result["trained_fm"] >= FM_MIN - 1e-6, (
                f"Held={dev}: trained fm={result['trained_fm']:.4f}"
            )

    @pytest.mark.slow
    def test_loo_blind_reasonable(
        self, loo_constrained: dict,
    ) -> None:
        """LOO blind NRMSE < 0.60 (generous; tests transferability)."""
        for dev, result in loo_constrained.items():
            assert result["blind_nrmse"] < 0.60, (
                f"{dev}: blind={result['blind_nrmse']:.4f}"
            )

    @pytest.mark.slow
    def test_loo_report(
        self, loo_constrained: dict,
    ) -> None:
        """Print LOO results."""
        print(f"\n=== Leave-One-Out (fm >= {FM_MIN}) ===")
        for dev, r in loo_constrained.items():
            print(f"  Held={dev}: blind={r['blind_nrmse']:.4f}, "
                  f"indep={r['independent_nrmse']:.4f}, "
                  f"degrad={r['degradation']:.2f}x, "
                  f"fc={r['trained_fc']:.4f} "
                  f"fm={r['trained_fm']:.4f} "
                  f"delay={r['trained_delay_us']:.3f}")


# =====================================================================
# Test 5: Comparative analysis (constrained vs unconstrained)
# =====================================================================


class TestComparativeAnalysis:
    """Slow tests comparing constrained vs unconstrained."""

    @pytest.mark.slow
    def test_constrained_eliminates_nonphysical_fm(
        self,
        unconstrained_mode2: MultiDeviceResult,
        constrained_mode2: MultiDeviceResult,
    ) -> None:
        """fm constraint eliminates PF-1000 fm=0.037 non-physical value."""
        pf_fm_u = unconstrained_mode2.device_fm["PF-1000"]
        pf_fm_c = constrained_mode2.device_fm["PF-1000"]
        # Unconstrained should have low fm
        assert pf_fm_u < FM_MIN, (
            f"Unconstrained PF-1000 fm={pf_fm_u:.4f} already physical"
        )
        # Constrained must be physical
        assert pf_fm_c >= FM_MIN - 1e-6

    @pytest.mark.slow
    def test_nrmse_cost_of_constraint(
        self,
        unconstrained_mode2: MultiDeviceResult,
        constrained_mode2: MultiDeviceResult,
    ) -> None:
        """Report NRMSE cost and verify it's bounded."""
        u = unconstrained_mode2
        c = constrained_mode2
        cost = c.combined_nrmse - u.combined_nrmse
        cost_pct = cost / u.combined_nrmse * 100 if u.combined_nrmse > 0 else 0
        print(f"\n=== NRMSE Cost of fm >= {FM_MIN} Constraint ===")
        print(f"Unconstrained: {u.combined_nrmse:.4f}")
        print(f"Constrained:   {c.combined_nrmse:.4f}")
        print(f"Cost: {cost:+.4f} ({cost_pct:+.1f}%)")
        for d in u.devices:
            d_cost = c.device_nrmse[d] - u.device_nrmse[d]
            print(f"  {d}: {u.device_nrmse[d]:.4f} → {c.device_nrmse[d]:.4f} "
                  f"({d_cost:+.4f})")
        # Constrained should not be catastrophically worse
        assert c.combined_nrmse < 3.0 * u.combined_nrmse

    @pytest.mark.slow
    def test_shared_fc_shift(
        self,
        unconstrained_mode2: MultiDeviceResult,
        constrained_mode2: MultiDeviceResult,
    ) -> None:
        """Report and validate shared fc shift."""
        shift = constrained_mode2.shared_fc - unconstrained_mode2.shared_fc
        print("\n=== Shared fc Shift ===")
        print(f"Unconstrained: fc={unconstrained_mode2.shared_fc:.4f}")
        print(f"Constrained:   fc={constrained_mode2.shared_fc:.4f}")
        print(f"Shift: {shift:+.4f}")
        assert 0.5 <= constrained_mode2.shared_fc <= 0.95


# =====================================================================
# Test 6: Full report
# =====================================================================


class TestPhaseReport:
    """Single slow test that generates the Phase BK summary report."""

    @pytest.mark.slow
    def test_full_report(
        self,
        constrained_mode1: MultiDeviceResult,
        constrained_mode2: MultiDeviceResult,
        unconstrained_mode2: MultiDeviceResult,
        loo_constrained: dict[str, dict[str, float]],
    ) -> None:
        """Generate comprehensive Phase BK report."""
        print("\n" + "=" * 70)
        print("PHASE BK: FM-CONSTRAINED MULTI-DEVICE CALIBRATION REPORT")
        print("=" * 70)

        m1 = constrained_mode1
        print(f"\n--- Mode 1: Shared (fc, fm, delay), fm >= {FM_MIN} ---")
        print(f"  fc={m1.shared_fc:.4f}, fm={m1.shared_fm:.4f}, "
              f"delay={m1.shared_delay_us:.3f} us")
        print(f"  fc^2/fm = {m1.shared_fc**2/m1.shared_fm:.2f}")
        for d in m1.devices:
            print(f"  {d}: NRMSE={m1.device_nrmse[d]:.4f} "
                  f"(indep={m1.independent_nrmse[d]:.4f}, "
                  f"penalty={m1.nrmse_penalty[d]:+.1%})")
        print(f"  Combined: {m1.combined_nrmse:.4f} "
              f"(Phase BJ: {BJ_MODE1_COMBINED:.4f})")

        m2 = constrained_mode2
        print(f"\n--- Mode 2: Shared fc, device fm/delay, fm >= {FM_MIN} ---")
        print(f"  fc={m2.shared_fc:.4f} "
              f"(Phase BJ: {BJ_SHARED_FC:.4f})")
        for d in m2.devices:
            print(f"  {d}: fm={m2.device_fm[d]:.4f}, "
                  f"delay={m2.device_delay_us[d]:.3f} us, "
                  f"NRMSE={m2.device_nrmse[d]:.4f} "
                  f"(penalty={m2.nrmse_penalty[d]:+.1%})")
        print(f"  Combined: {m2.combined_nrmse:.4f} "
              f"(Phase BJ: {BJ_MODE2_COMBINED:.4f})")

        print(f"\n--- Leave-One-Out (fm >= {FM_MIN}) ---")
        for dev, r in loo_constrained.items():
            print(f"  Held={dev}: blind={r['blind_nrmse']:.4f}, "
                  f"degrad={r['degradation']:.2f}x, "
                  f"fc={r['trained_fc']:.4f}, "
                  f"fm={r['trained_fm']:.4f}")

        u = unconstrained_mode2
        nrmse_cost = m2.combined_nrmse - u.combined_nrmse
        pf_fm_old = u.device_fm["PF-1000"]
        pf_fm_new = m2.device_fm["PF-1000"]

        print("\n--- Key Physics Findings ---")
        print(f"  PF-1000 fm: {pf_fm_old:.4f} (unconstrained) "
              f"→ {pf_fm_new:.4f} (constrained)")
        print(f"  NRMSE cost of fm >= {FM_MIN}: {nrmse_cost:+.4f}")
        print("  Independent constrained fc^2/fm:")
        for d in m2.devices:
            ratio = m2.independent_fc[d] ** 2 / m2.independent_fm[d]
            print(f"    {d}: {ratio:.2f} "
                  f"(fc={m2.independent_fc[d]:.4f}, "
                  f"fm={m2.independent_fm[d]:.4f})")
        print("=" * 70)
