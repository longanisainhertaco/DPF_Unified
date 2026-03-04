"""Phase BL: Third device (UNU-ICTP) integration + N=3 LOO cross-validation.

Integrates the UNU-ICTP PFF as a third calibration device with a measured
I(t) waveform (from IPFS 'UNU ICTPPFF D2 05.15.xls', 13.5 kV, 3 Torr D2).
Enables N=3 leave-one-out cross-validation, giving the first LOO result with
a well-defined t-distribution (df=2) confidence interval.

Key questions:
    1. Does UNU-ICTP calibrate independently with physical fc/fm?
    2. Does N=3 LOO produce a valid confidence interval (df=2)?
    3. Does the LOO mean error decrease or increase vs N=2?
    4. Is UNU-ICTP more like PF-1000 or POSEIDON in parameter space?
"""

from __future__ import annotations

import math

import pytest

from dpf.validation.experimental import (
    DEVICES,
    PF1000_DATA,
    POSEIDON_60KV_DATA,
    UNU_ICTP_DATA,
    compute_lp_l0_ratio,
)

# =====================================================================
# Non-slow tests: data integrity + basic checks
# =====================================================================


class TestUNUICTPWaveform:
    """Verify UNU-ICTP digitized waveform data integrity."""

    def test_waveform_exists(self):
        """UNU-ICTP must have a digitized waveform."""
        assert UNU_ICTP_DATA.waveform_t is not None
        assert UNU_ICTP_DATA.waveform_I is not None

    def test_waveform_length(self):
        """Waveform should have 40-50 points."""
        assert 40 <= len(UNU_ICTP_DATA.waveform_t) <= 50
        assert len(UNU_ICTP_DATA.waveform_t) == len(UNU_ICTP_DATA.waveform_I)

    def test_waveform_monotonic_time(self):
        """Time array must be strictly increasing."""
        import numpy as np
        dt = np.diff(UNU_ICTP_DATA.waveform_t)
        assert (dt > 0).all(), "Time not monotonically increasing"

    def test_waveform_units_si(self):
        """Waveform must be in SI units (seconds, amperes)."""
        # Time should be in seconds (0 to ~5e-6)
        assert UNU_ICTP_DATA.waveform_t[0] >= 0.0
        assert UNU_ICTP_DATA.waveform_t[-1] < 10e-6
        # Current should be in amperes (0 to ~170e3)
        assert max(UNU_ICTP_DATA.waveform_I) > 100e3
        assert max(UNU_ICTP_DATA.waveform_I) < 300e3

    def test_peak_current_matches(self):
        """Peak in waveform should match ExperimentalDevice.peak_current."""
        waveform_peak = max(UNU_ICTP_DATA.waveform_I)
        assert abs(waveform_peak - UNU_ICTP_DATA.peak_current) / UNU_ICTP_DATA.peak_current < 0.05

    def test_voltage_is_13_5kv(self):
        """Voltage should be 13.5 kV (from IPFS measured data, not 14 kV)."""
        assert UNU_ICTP_DATA.voltage == pytest.approx(13.5e3, rel=0.01)

    def test_circuit_parameters(self):
        """Verify circuit parameters match published values."""
        assert UNU_ICTP_DATA.capacitance == pytest.approx(30e-6, rel=0.01)
        assert UNU_ICTP_DATA.inductance == pytest.approx(110e-9, rel=0.01)
        assert UNU_ICTP_DATA.resistance == pytest.approx(12e-3, rel=0.01)

    def test_geometry(self):
        """Verify geometry matches published values."""
        assert UNU_ICTP_DATA.anode_radius == pytest.approx(0.0095, rel=0.01)
        assert UNU_ICTP_DATA.cathode_radius == pytest.approx(0.032, rel=0.01)
        assert UNU_ICTP_DATA.anode_length == pytest.approx(0.16, rel=0.01)

    def test_in_devices_registry(self):
        """UNU-ICTP must be in the DEVICES registry."""
        assert "UNU-ICTP" in DEVICES
        assert DEVICES["UNU-ICTP"] is UNU_ICTP_DATA

    def test_digitization_uncertainty(self):
        """Digitization uncertainty should reflect 9.3 kA quantization (GUM)."""
        # GUM rectangular: 9.3 kA / (2*sqrt(3)*169 kA) = 1.6%
        assert UNU_ICTP_DATA.waveform_digitization_uncertainty == pytest.approx(0.016, abs=0.005)

    def test_current_dip_exists(self):
        """Waveform should show a current dip after the peak."""
        import numpy as np
        current = UNU_ICTP_DATA.waveform_I
        t = UNU_ICTP_DATA.waveform_t
        peak_idx = np.argmax(current)
        # Look for minimum in the 2.6-3.0 us window
        mask = (t > 2.6e-6) & (t < 3.0e-6)
        if mask.any():
            dip_I = np.min(current[mask])
            peak_I = current[peak_idx]
            dip_frac = 1 - dip_I / peak_I
            # Dip should be 5-30% below peak
            assert 0.05 < dip_frac < 0.30, f"Dip fraction {dip_frac:.2%} outside expected range"


class TestUNUICTPLpL0:
    """Verify L_p/L0 diagnostic for UNU-ICTP."""

    def test_circuit_dominated(self):
        """UNU-ICTP should be circuit-dominated (L_p/L0 < 0.5)."""
        result = compute_lp_l0_ratio(
            L0=UNU_ICTP_DATA.inductance,
            anode_radius=UNU_ICTP_DATA.anode_radius,
            cathode_radius=UNU_ICTP_DATA.cathode_radius,
            anode_length=UNU_ICTP_DATA.anode_length,
        )
        assert result["L_p_over_L0"] < 0.5

    def test_three_device_lp_l0_spread(self):
        """PF-1000 and POSEIDON are plasma-significant; UNU-ICTP is not."""
        devs = {"PF-1000": PF1000_DATA, "POSEIDON-60kV": POSEIDON_60KV_DATA,
                "UNU-ICTP": UNU_ICTP_DATA}
        ratios = {}
        for name, d in devs.items():
            r = compute_lp_l0_ratio(
                L0=d.inductance,
                anode_radius=d.anode_radius,
                cathode_radius=d.cathode_radius,
                anode_length=d.anode_length,
            )
            ratios[name] = r["L_p_over_L0"]
        # PF-1000 and POSEIDON should both be plasma-significant (>1.0)
        assert ratios["PF-1000"] > 1.0
        assert ratios["POSEIDON-60kV"] > 1.0
        # UNU-ICTP should be circuit-dominated (<0.5)
        assert ratios["UNU-ICTP"] < 0.5


class TestThreeDeviceSetup:
    """Verify three-device MultiDeviceCalibrator instantiation."""

    def test_instantiate_three_devices(self):
        """MultiDeviceCalibrator should accept 3 devices."""
        from dpf.validation.calibration import MultiDeviceCalibrator
        cal = MultiDeviceCalibrator(
            devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP"],
            maxiter=1,  # just test instantiation
        )
        assert len(cal.devices) == 3
        assert "UNU-ICTP" in cal.devices

    def test_all_devices_have_waveforms(self):
        """All three devices must have digitized waveforms."""
        for name in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]:
            dev = DEVICES[name]
            assert dev.waveform_t is not None, f"{name} missing waveform_t"
            assert dev.waveform_I is not None, f"{name} missing waveform_I"

    def test_energy_ordering(self):
        """Devices span 3 orders of magnitude in energy."""
        E_unu = 0.5 * UNU_ICTP_DATA.capacitance * UNU_ICTP_DATA.voltage**2
        E_pos = 0.5 * POSEIDON_60KV_DATA.capacitance * POSEIDON_60KV_DATA.voltage**2
        E_pf = 0.5 * PF1000_DATA.capacitance * PF1000_DATA.voltage**2
        # UNU-ICTP < POSEIDON < PF-1000
        assert E_unu < E_pos < E_pf
        # Ratio should be > 100x
        assert E_pf / E_unu > 100

    def test_speed_factor_variety(self):
        """Devices should span a range of speed factors."""
        import numpy as np
        mu0 = 4 * np.pi * 1e-7
        devices = [UNU_ICTP_DATA, POSEIDON_60KV_DATA, PF1000_DATA]
        speed_factors = []
        for d in devices:
            c = d.cathode_radius / d.anode_radius
            S_factor = (d.voltage / d.anode_length) * (
                d.capacitance * (math.log(c))**2 / (
                    mu0 * d.anode_radius * d.fill_pressure_torr * 133.322
                )
            )**0.5
            speed_factors.append(S_factor)
        # At least 2x range
        assert max(speed_factors) / min(speed_factors) > 2


# =====================================================================
# Slow tests: calibration + LOO
# =====================================================================


@pytest.fixture(scope="module")
def unu_independent():
    """Independent calibration of UNU-ICTP (3-param with liftoff delay)."""
    from dpf.validation.calibration import calibrate_with_liftoff
    return calibrate_with_liftoff(
        device_name="UNU-ICTP",
        fc_bounds=(0.5, 0.95),
        fm_bounds=(0.01, 0.40),
        delay_bounds_us=(0.0, 2.0),
        pinch_column_fraction=0.06,  # ~1 cm pinch of 16 cm anode (Lee & Saw 2009)
        maxiter=20,
    )


@pytest.fixture(scope="module")
def three_device_shared():
    """Three-device shared calibration.

    Uses maxiter=3 for DE (minimum for reasonable exploration).
    Full-quality runs should use maxiter=200.
    """
    from dpf.validation.calibration import MultiDeviceCalibrator
    cal = MultiDeviceCalibrator(
        devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP"],
        fc_bounds=(0.5, 0.95),
        fm_bounds=(0.10, 0.40),  # fm >= 0.10 physical constraint
        delay_bounds_us=(0.0, 2.0),
        maxiter=3,
        seed=42,
    )
    return cal.calibrate_shared()


@pytest.fixture(scope="module")
def three_device_loo():
    """Three-device LOO cross-validation.

    Uses maxiter=1 for DE (minimum for mechanism verification).
    Even at maxiter=1, the 3 independent + 3 shared DE calibrations
    take ~12 min on M3 Pro due to Lee model ODE integration cost.
    Full-quality runs should use maxiter=200.
    """
    from dpf.validation.calibration import MultiDeviceCalibrator
    cal = MultiDeviceCalibrator(
        devices=["PF-1000", "POSEIDON-60kV", "UNU-ICTP"],
        fc_bounds=(0.5, 0.95),
        fm_bounds=(0.10, 0.40),
        delay_bounds_us=(0.0, 2.0),
        maxiter=1,
        seed=42,
    )
    return cal.leave_one_out()


@pytest.mark.slow
class TestUNUICTPCalibration:
    """Independent calibration of UNU-ICTP."""

    def test_optimization_ran(self, unu_independent):
        """Calibration should complete with a reasonable NRMSE.

        Note: DE with low maxiter (20) may not formally converge, but
        produces good results (NRMSE < 0.15 for UNU-ICTP).
        """
        assert unu_independent.n_evals > 100
        assert unu_independent.nrmse < 0.15

    def test_nrmse_reasonable(self, unu_independent):
        """NRMSE should be < 0.30 (not great due to quantization)."""
        assert unu_independent.nrmse < 0.30

    def test_fc_in_range(self, unu_independent):
        """fc should be in published range (0.5-0.95)."""
        assert 0.5 <= unu_independent.best_fc <= 0.95

    def test_fm_physical(self, unu_independent):
        """fm should be >= 0.01 (Lee & Saw published range)."""
        assert unu_independent.best_fm >= 0.01

    def test_fc_squared_over_fm(self, unu_independent):
        """fc^2/fm should be finite and positive."""
        ratio = unu_independent.best_fc**2 / unu_independent.best_fm
        assert 0.1 < ratio < 100


@pytest.mark.slow
class TestThreeDeviceShared:
    """Three-device shared parameter calibration."""

    def test_result_has_three_devices(self, three_device_shared):
        """Result should cover all three devices."""
        assert len(three_device_shared.devices) == 3

    def test_combined_nrmse_finite(self, three_device_shared):
        """Combined NRMSE should be finite."""
        assert 0 < three_device_shared.combined_nrmse < 1.0

    def test_all_device_nrmse_finite(self, three_device_shared):
        """All per-device NRMSEs should be finite."""
        for dev, nrmse in three_device_shared.device_nrmse.items():
            assert 0 < nrmse < 2.0, f"{dev} NRMSE={nrmse}"

    def test_shared_fc_in_range(self, three_device_shared):
        """Shared fc should be within bounds."""
        assert 0.5 <= three_device_shared.shared_fc <= 0.95

    def test_shared_fm_physical(self, three_device_shared):
        """Shared fm should satisfy fm >= 0.10 constraint."""
        assert three_device_shared.shared_fm >= 0.10


@pytest.mark.slow
class TestThreeDeviceLOO:
    """N=3 leave-one-out cross-validation."""

    def test_three_held_out(self, three_device_loo):
        """Should have results for all 3 held-out devices."""
        assert len(three_device_loo) == 3
        assert "PF-1000" in three_device_loo
        assert "POSEIDON-60kV" in three_device_loo
        assert "UNU-ICTP" in three_device_loo

    def test_degradation_factors_finite(self, three_device_loo):
        """All degradation factors should be positive and finite."""
        for _dev, metrics in three_device_loo.items():
            assert metrics["degradation"] > 0
            assert math.isfinite(metrics["degradation"])

    def test_blind_nrmse_finite(self, three_device_loo):
        """All blind NRMSEs should be finite."""
        for dev, metrics in three_device_loo.items():
            assert 0 < metrics["blind_nrmse"] < 2.0, f"{dev} blind={metrics['blind_nrmse']}"

    def test_mean_loo_error(self, three_device_loo):
        """Mean LOO error should be computable (not Cauchy like N=2)."""
        import numpy as np
        blind_nrmses = [m["blind_nrmse"] for m in three_device_loo.values()]
        mean_loo = np.mean(blind_nrmses)
        assert 0 < mean_loo < 2.0

    def test_loo_std_finite(self, three_device_loo):
        """LOO standard deviation should be finite (df=2, not Cauchy)."""
        import numpy as np
        blind_nrmses = [m["blind_nrmse"] for m in three_device_loo.values()]
        std_loo = np.std(blind_nrmses, ddof=1)
        assert math.isfinite(std_loo)
        assert std_loo > 0  # should have variance

    def test_t_distribution_ci(self, three_device_loo):
        """N=3 gives df=2 t-distribution CI (finite, unlike N=2 Cauchy)."""
        import numpy as np
        from scipy import stats
        blind_nrmses = np.array([m["blind_nrmse"] for m in three_device_loo.values()])
        mean = np.mean(blind_nrmses)
        se = np.std(blind_nrmses, ddof=1) / np.sqrt(len(blind_nrmses))
        # t-distribution with df=2
        t_crit = stats.t.ppf(0.975, df=2)  # ~4.303 for 95% CI
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
        # CI should be finite (unlike N=2 where it's undefined)
        assert math.isfinite(ci_low)
        assert math.isfinite(ci_high)
        assert ci_low < ci_high


@pytest.mark.slow
class TestPhaseReport:
    """Generate Phase BL summary report."""

    def test_report(self, unu_independent, three_device_shared, three_device_loo, capsys):
        """Print comprehensive Phase BL report."""
        import numpy as np
        from scipy import stats

        print("\n" + "=" * 70)
        print("Phase BL: Third Device (UNU-ICTP) + N=3 LOO Cross-Validation")
        print("=" * 70)

        # UNU-ICTP independent
        cal = unu_independent
        print("\n=== UNU-ICTP Independent Calibration ===")
        print(f"fc={cal.best_fc:.4f}, fm={cal.best_fm:.4f}, "
              f"delay={cal.best_delay_us:.3f} us")
        print(f"NRMSE={cal.nrmse:.4f}")
        print(f"fc^2/fm = {cal.best_fc**2 / cal.best_fm:.2f}")
        print(f"Converged: {cal.converged}")

        # Three-device shared
        s = three_device_shared
        print("\n=== Three-Device Shared Calibration ===")
        print(f"Shared: fc={s.shared_fc:.4f}, fm={s.shared_fm:.4f}, "
              f"delay={s.shared_delay_us:.3f} us")
        print(f"Combined NRMSE: {s.combined_nrmse:.4f}")
        for dev, nrmse in s.device_nrmse.items():
            indep = s.independent_nrmse.get(dev, float("nan"))
            penalty = (nrmse - indep) / indep * 100 if indep > 0 else float("nan")
            print(f"  {dev}: NRMSE={nrmse:.4f} (indep={indep:.4f}, penalty={penalty:+.1f}%)")

        # LOO results
        print("\n=== N=3 Leave-One-Out Cross-Validation ===")
        blind_nrmses = []
        for dev, m in three_device_loo.items():
            print(f"  Hold {dev}: blind={m['blind_nrmse']:.4f}, "
                  f"indep={m['independent_nrmse']:.4f}, "
                  f"degradation={m['degradation']:.2f}x")
            blind_nrmses.append(m["blind_nrmse"])

        blind_arr = np.array(blind_nrmses)
        mean_loo = np.mean(blind_arr)
        std_loo = np.std(blind_arr, ddof=1)
        se_loo = std_loo / np.sqrt(len(blind_arr))
        t_crit = stats.t.ppf(0.975, df=len(blind_arr) - 1)
        ci_low = mean_loo - t_crit * se_loo
        ci_high = mean_loo + t_crit * se_loo

        print(f"\nMean LOO blind NRMSE: {mean_loo:.4f} +/- {std_loo:.4f}")
        print(f"95% CI (t-dist, df={len(blind_arr)-1}): [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"t_crit(0.975, df=2) = {t_crit:.3f}")

        # Compare to N=2 LOO (from Phase BK)
        print("\n=== Comparison: N=2 vs N=3 LOO ===")
        print("N=2 (Phase BK): E_LOO = 0.430 (Cauchy, no valid CI)")
        print(f"N=3 (Phase BL): E_LOO = {mean_loo:.4f} (t-dist, CI valid)")

        # L_p/L0 for all three devices
        print("\n=== L_p/L0 Diagnostic ===")
        for name in ["PF-1000", "POSEIDON-60kV", "UNU-ICTP"]:
            d = DEVICES[name]
            r = compute_lp_l0_ratio(
                L0=d.inductance,
                anode_radius=d.anode_radius,
                cathode_radius=d.cathode_radius,
                anode_length=d.anode_length,
            )
            regime = "plasma-significant" if r["L_p_over_L0"] > 1.0 else "circuit-dominated"
            print(f"  {name}: L_p/L0 = {r['L_p_over_L0']:.2f} ({regime})")

        print("\n" + "=" * 70)
        # Always pass - this is just a report
        assert True
