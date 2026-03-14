"""Tests for validation suite: scoring, RMSE, device data, reports."""

from __future__ import annotations

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════
# Scoring Function Tests
# ═══════════════════════════════════════════════════════

class TestNormalizedRMSE:
    """Tests for normalized RMSE computation."""

    def test_perfect_match(self):
        """NRMSE = 0 for identical arrays."""
        from dpf.validation.suite import normalized_rmse

        x = np.array([1.0, 2.0, 3.0])
        assert normalized_rmse(x, x) == 0.0

    def test_known_value(self):
        """Test NRMSE for a known case."""
        from dpf.validation.suite import normalized_rmse

        sim = np.array([1.0, 2.0, 3.0])
        ref = np.array([1.1, 2.1, 3.1])

        # MSE = mean(0.01, 0.01, 0.01) = 0.01
        # RMSE = 0.1
        # Range = 3.1 - 1.1 = 2.0
        # NRMSE = 0.1 / 2.0 = 0.05
        np.testing.assert_allclose(normalized_rmse(sim, ref), 0.05, rtol=1e-10)

    def test_empty_returns_inf(self):
        """Empty arrays should return infinity."""
        from dpf.validation.suite import normalized_rmse

        assert normalized_rmse(np.array([]), np.array([])) == float("inf")


class TestRelativeError:
    """Tests for relative error."""

    def test_zero_error(self):
        """rel_error(x, x) = 0."""
        from dpf.validation.suite import relative_error

        assert relative_error(1.0, 1.0) == 0.0

    def test_known_value(self):
        """10% overestimate should give 0.1."""
        from dpf.validation.suite import relative_error

        np.testing.assert_allclose(relative_error(1.1, 1.0), 0.1, rtol=1e-10)

    def test_symmetric(self):
        """Error should be same for over/underestimate."""
        from dpf.validation.suite import relative_error

        np.testing.assert_allclose(
            relative_error(1.1, 1.0), relative_error(0.9, 1.0), rtol=1e-10
        )


class TestConfigHash:
    """Tests for config hashing."""

    def test_deterministic(self):
        """Same config should give same hash."""
        from dpf.validation.suite import config_hash

        cfg = {"a": 1, "b": "test"}
        assert config_hash(cfg) == config_hash(cfg)

    def test_different_configs(self):
        """Different configs should give different hashes."""
        from dpf.validation.suite import config_hash

        h1 = config_hash({"a": 1})
        h2 = config_hash({"a": 2})
        assert h1 != h2


# ═══════════════════════════════════════════════════════
# Device Registry Tests
# ═══════════════════════════════════════════════════════

class TestDeviceRegistry:
    """Tests for device reference data."""

    def test_registry_has_devices(self):
        """Registry should contain known devices."""
        from dpf.validation.suite import DEVICE_REGISTRY

        assert "PF-1000" in DEVICE_REGISTRY
        assert "NX2" in DEVICE_REGISTRY
        assert "LLNL-DPF" in DEVICE_REGISTRY

    def test_pf1000_data(self):
        """PF-1000 reference data should be physically reasonable."""
        from dpf.validation.suite import PF1000

        assert PF1000.peak_current_A > 1e6     # > 1 MA
        assert PF1000.peak_current_A < 5e6     # < 5 MA
        assert PF1000.C > 1e-4                  # > 100 uF
        assert PF1000.V0 > 10e3                  # > 10 kV
        assert PF1000.anode_radius < PF1000.cathode_radius

    def test_nx2_data(self):
        """NX2 is a small device — peak current < 1 MA."""
        from dpf.validation.suite import NX2

        assert NX2.peak_current_A < 1e6
        assert NX2.peak_current_A > 100e3
        assert NX2.anode_radius < NX2.cathode_radius

    def test_all_devices_have_tolerances(self):
        """All devices should define at least peak_current tolerance."""
        from dpf.validation.suite import DEVICE_REGISTRY

        for name, device in DEVICE_REGISTRY.items():
            assert "peak_current" in device.tolerances, f"{name} missing tolerance"


# ═══════════════════════════════════════════════════════
# Validation Suite Tests
# ═══════════════════════════════════════════════════════

class TestValidationSuite:
    """Tests for the validation suite."""

    def test_init_all_devices(self):
        """Default init should use all devices."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite()
        assert len(suite.devices) == 3

    def test_init_specific_devices(self):
        """Can initialize with specific devices."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["NX2"])
        assert suite.devices == ["NX2"]

    def test_unknown_device_raises(self):
        """Unknown device name should raise ValueError."""
        from dpf.validation.suite import ValidationSuite

        with pytest.raises(ValueError, match="Unknown device"):
            ValidationSuite(devices=["NONEXISTENT"])

    def test_validate_perfect_match(self):
        """Perfect match should give score ~1.0."""
        from dpf.validation.suite import NX2, ValidationSuite

        suite = ValidationSuite(devices=["NX2"])

        # Simulate a perfect match to NX2
        sim_summary = {
            "peak_current_A": NX2.peak_current_A,
            "peak_current_time_s": NX2.peak_current_time_s,
            "energy_conservation": 1.0,
            "final_current_A": 100e3,
        }

        result = suite.validate_circuit("NX2", sim_summary)
        assert result.passed
        assert result.overall_score > 0.95

    def test_validate_poor_match(self):
        """50% error should give low score."""
        from dpf.validation.suite import NX2, ValidationSuite

        suite = ValidationSuite(devices=["NX2"])

        sim_summary = {
            "peak_current_A": NX2.peak_current_A * 0.5,  # 50% error
            "peak_current_time_s": NX2.peak_current_time_s * 1.5,  # 50% error
            "energy_conservation": 0.7,  # 30% energy loss
            "final_current_A": 50e3,
        }

        result = suite.validate_circuit("NX2", sim_summary)
        assert not result.passed
        assert result.overall_score < 0.7

    def test_validate_with_config_hash(self):
        """Config hash should be included in result."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite(devices=["NX2"])
        config = {"C": 28e-6, "V0": 14e3}

        result = suite.validate_circuit("NX2", {"energy_conservation": 1.0}, config)
        assert len(result.config_hash) == 16  # SHA256[:16]

    def test_validate_all(self):
        """validate_all should return results for all configured devices."""
        from dpf.validation.suite import ValidationSuite

        suite = ValidationSuite()
        sim_summary = {
            "energy_conservation": 1.0,
            "final_current_A": 100e3,
        }

        results = suite.validate_all(sim_summary)
        assert len(results) == 3
        assert "PF-1000" in results
        assert "NX2" in results
        assert "LLNL-DPF" in results


class TestValidationReport:
    """Tests for report generation."""

    def test_report_string(self):
        """Report should contain device names and scores."""
        from dpf.validation.suite import NX2, ValidationSuite

        suite = ValidationSuite(devices=["NX2"])
        sim_summary = {
            "peak_current_A": NX2.peak_current_A,
            "energy_conservation": 1.0,
            "final_current_A": 300e3,
        }

        results = suite.validate_all(sim_summary)
        report = suite.report(results)

        assert "NX2" in report
        assert "PASS" in report or "FAIL" in report
        assert "Score" in report

    def test_report_contains_metrics(self):
        """Report should show individual metric results."""
        from dpf.validation.suite import NX2, ValidationSuite

        suite = ValidationSuite(devices=["NX2"])
        sim_summary = {
            "peak_current_A": NX2.peak_current_A * 1.1,  # 10% off
            "energy_conservation": 0.98,
            "final_current_A": 280e3,
        }

        results = suite.validate_all(sim_summary)
        report = suite.report(results)

        assert "peak_current" in report
        assert "energy_conservation" in report
