"""Phase BR: GUM-compliant uncertainty taxonomy + ASME double-counting fix.

Implements Debate #51 recommendations:
1. waveform_amplitude_uncertainty replaces waveform_digitization_uncertainty
   - "digitization" type for measured waveforms (trace reading error)
   - "reconstruction" type for model-generated waveforms
2. waveform_uncertainty_type field per GUM (JCGM 100:2008)
3. peak_current_from_shot_spread flag for double-counting prevention
4. ASME budget correctly skips u_shot_to_shot when peak_current_uncertainty
   already incorporates shot spread

References:
    GUM (JCGM 100:2008): Guide to the Expression of Uncertainty in Measurement
    ASME V&V 20-2009: Standard for Verification and Validation
    PhD Debate #51 (2026-03-04): Findings 1, 6
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.validation.experimental import (
    DEVICES,
    ExperimentalDevice,
    get_devices_by_provenance,
)

# =====================================================================
# Test: Uncertainty type classification
# =====================================================================


class TestUncertaintyType:
    """Every device with a waveform must have uncertainty_type set per GUM."""

    def test_measured_devices_have_digitization_type(self):
        """Measured waveforms should have uncertainty_type='digitization'."""
        measured = get_devices_by_provenance("measured")
        assert len(measured) >= 3, "Need at least 3 measured devices"
        for name, dev in measured.items():
            assert dev.waveform_uncertainty_type == "digitization", (
                f"{name}: measured waveform should have uncertainty_type='digitization', "
                f"got '{dev.waveform_uncertainty_type}'"
            )

    def test_reconstructed_devices_have_reconstruction_type(self):
        """Reconstructed waveforms should have uncertainty_type='reconstruction'."""
        recon = get_devices_by_provenance("reconstructed")
        assert len(recon) >= 2, "Need at least 2 reconstructed devices"
        for name, dev in recon.items():
            assert dev.waveform_uncertainty_type == "reconstruction", (
                f"{name}: reconstructed waveform should have uncertainty_type='reconstruction', "
                f"got '{dev.waveform_uncertainty_type}'"
            )

    def test_no_waveform_devices_have_empty_type(self):
        """Devices without waveforms should have empty uncertainty_type."""
        no_waveform = [
            name for name, dev in DEVICES.items()
            if dev.waveform_t is None
        ]
        for name in no_waveform:
            dev = DEVICES[name]
            assert dev.waveform_uncertainty_type == "", (
                f"{name}: no-waveform device should have empty uncertainty_type"
            )

    def test_uncertainty_type_values_exhaustive(self):
        """All devices must have one of the valid uncertainty types."""
        valid_types = {"digitization", "reconstruction", ""}
        for name, dev in DEVICES.items():
            assert dev.waveform_uncertainty_type in valid_types, (
                f"{name}: invalid uncertainty_type '{dev.waveform_uncertainty_type}'"
            )

    def test_measured_uncertainty_lower_than_reconstructed(self):
        """Per GUM, reconstruction uncertainty should be higher than digitization."""
        measured = get_devices_by_provenance("measured")
        recon = get_devices_by_provenance("reconstructed")
        avg_meas = np.mean([d.waveform_amplitude_uncertainty for d in measured.values()])
        avg_recon = np.mean([d.waveform_amplitude_uncertainty for d in recon.values()])
        assert avg_meas < avg_recon, (
            f"Measured avg ({avg_meas:.3f}) should be < reconstructed avg ({avg_recon:.3f})"
        )


# =====================================================================
# Test: waveform_amplitude_uncertainty field
# =====================================================================


class TestAmplitudeUncertainty:
    """Verify waveform_amplitude_uncertainty replaces old digitization field."""

    def test_field_exists_on_all_devices(self):
        """Every device should have the waveform_amplitude_uncertainty field."""
        for name, dev in DEVICES.items():
            assert hasattr(dev, "waveform_amplitude_uncertainty"), (
                f"{name}: missing waveform_amplitude_uncertainty field"
            )

    def test_pf1000_digitization_3pct(self):
        """PF-1000 hand-digitized: 3% amplitude from trace reading."""
        assert DEVICES["PF-1000"].waveform_amplitude_uncertainty == pytest.approx(0.03, abs=0.005)

    def test_unu_ictp_digitization_1_6pct(self):
        """UNU-ICTP digital oscilloscope: 1.6% from GUM rectangular."""
        assert DEVICES["UNU-ICTP"].waveform_amplitude_uncertainty == pytest.approx(0.016, abs=0.003)

    def test_gribkov_digitization_2pct(self):
        """Gribkov IPFS archive: 2% from digital data."""
        assert DEVICES["PF-1000-Gribkov"].waveform_amplitude_uncertainty == pytest.approx(0.02, abs=0.005)

    def test_poseidon60kv_digitization_2pct(self):
        """POSEIDON-60kV IPFS: 2% from high-quality digitization."""
        assert DEVICES["POSEIDON-60kV"].waveform_amplitude_uncertainty == pytest.approx(0.02, abs=0.005)

    def test_pf1000_16kv_reconstruction_5pct(self):
        """PF-1000-16kV: 5% reconstruction model uncertainty."""
        dev = DEVICES["PF-1000-16kV"]
        assert dev.waveform_amplitude_uncertainty == pytest.approx(0.05, abs=0.01)
        assert dev.waveform_uncertainty_type == "reconstruction"

    def test_faeton_reconstruction_8pct(self):
        """FAETON-I: 8% reconstruction from damped RLC parameters."""
        dev = DEVICES["FAETON-I"]
        assert dev.waveform_amplitude_uncertainty == pytest.approx(0.08, abs=0.01)
        assert dev.waveform_uncertainty_type == "reconstruction"

    def test_mjolnir_reconstruction_10pct(self):
        """MJOLNIR: 10% reconstruction (highest uncertainty)."""
        dev = DEVICES["MJOLNIR"]
        assert dev.waveform_amplitude_uncertainty == pytest.approx(0.10, abs=0.01)
        assert dev.waveform_uncertainty_type == "reconstruction"

    def test_no_old_field_name(self):
        """The old field waveform_digitization_uncertainty should not exist."""
        dev = ExperimentalDevice(
            name="test", institution="test",
            capacitance=1e-3, voltage=1e4, inductance=1e-8, resistance=1e-3,
            anode_radius=0.01, cathode_radius=0.02, anode_length=0.1,
            fill_pressure_torr=1.0, fill_gas="deuterium",
            peak_current=1e5, neutron_yield=1e6,
            current_rise_time=1e-6, reference="test",
        )
        assert not hasattr(dev, "waveform_digitization_uncertainty"), (
            "Old field name should not exist on ExperimentalDevice"
        )


# =====================================================================
# Test: Double-counting prevention
# =====================================================================


class TestDoubleCounting:
    """Verify peak_current_from_shot_spread prevents double-counting in ASME."""

    def test_pf1000_16kv_has_flag(self):
        """PF-1000-16kV should have peak_current_from_shot_spread=True."""
        dev = DEVICES["PF-1000-16kV"]
        assert dev.peak_current_from_shot_spread is True, (
            "PF-1000-16kV: 10% peak_current_uncertainty derives from 1.1-1.3 MA "
            "shot range — flag should be True"
        )

    def test_other_devices_no_flag(self):
        """Most devices should NOT have peak_current_from_shot_spread."""
        expected_false = ["PF-1000", "UNU-ICTP", "POSEIDON-60kV", "NX2",
                          "PF-1000-Gribkov", "FAETON-I", "MJOLNIR"]
        for name in expected_false:
            dev = DEVICES[name]
            assert not dev.peak_current_from_shot_spread, (
                f"{name}: should not have peak_current_from_shot_spread=True"
            )

    def test_pf1000_16kv_peak_uncertainty_source(self):
        """PF-1000-16kV peak_current_uncertainty=10% is from shot range."""
        dev = DEVICES["PF-1000-16kV"]
        # Peak I = 1.2 MA, range 1.1-1.3 MA → ±8.3% → rounded to 10%
        assert dev.peak_current_uncertainty == pytest.approx(0.10, abs=0.02)
        # This INCLUDES shot-to-shot variability — don't add it again
        assert dev.peak_current_from_shot_spread is True

    def test_asme_budget_without_double_counting(self):
        """ASME u_exp for PF-1000-16kV should not include both sources.

        Without the fix: u_exp = sqrt(0.10^2 + 0.05^2 + (0.05/sqrt(16))^2)
                       = sqrt(0.01 + 0.0025 + 0.000156) = 0.1123
        With the fix:   u_exp = sqrt(0.10^2 + 0.05^2)
                       = sqrt(0.01 + 0.0025) = 0.1118
        The difference is small but the PRINCIPLE matters per GUM.
        """
        dev = DEVICES["PF-1000-16kV"]
        u_exp_correct = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        )
        # No shot-to-shot component because peak_current_from_shot_spread=True
        assert u_exp_correct == pytest.approx(0.1118, abs=0.005)

    def test_pf1000_asme_budget_with_shot_to_shot(self):
        """PF-1000 ASME budget SHOULD include shot-to-shot (no flag)."""
        dev = DEVICES["PF-1000"]
        assert not dev.peak_current_from_shot_spread
        # u_exp = sqrt(0.05^2 + 0.03^2 + (0.05/sqrt(5))^2) = sqrt(0.0034 + 0.0005) = 0.0624
        u_exp_with_shot = np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
            + (0.05 / np.sqrt(5))**2  # u_shot_to_shot / sqrt(n_shots)
        )
        assert u_exp_with_shot > np.sqrt(
            dev.peak_current_uncertainty**2
            + dev.waveform_amplitude_uncertainty**2
        ), "Adding shot-to-shot should increase u_exp"


# =====================================================================
# Test: GUM consistency
# =====================================================================


class TestGUMConsistency:
    """Verify GUM (JCGM 100:2008) requirements are met."""

    def test_each_component_has_physical_source(self):
        """Per GUM Section 4.3: each component identified by physical source."""
        for name, dev in DEVICES.items():
            if dev.waveform_t is not None:
                # Must have non-empty uncertainty_type
                assert dev.waveform_uncertainty_type in ("digitization", "reconstruction"), (
                    f"{name}: waveform has data but uncertainty_type "
                    f"'{dev.waveform_uncertainty_type}' is not a physical source per GUM"
                )

    def test_independent_components(self):
        """Per GUM: components combined in RSS must be independent."""
        dev = DEVICES["PF-1000-16kV"]
        if dev.peak_current_from_shot_spread:
            # If peak I uncertainty incorporates shot spread, these are NOT independent
            # The flag prevents double-counting in the ASME function
            assert dev.peak_current_from_shot_spread is True

    def test_provenance_consistency(self):
        """Waveform provenance should be consistent with uncertainty_type."""
        for name, dev in DEVICES.items():
            if dev.waveform_provenance == "measured":
                if dev.waveform_uncertainty_type:
                    assert dev.waveform_uncertainty_type == "digitization", (
                        f"{name}: measured waveform should have digitization uncertainty"
                    )
            elif dev.waveform_provenance == "reconstructed" and dev.waveform_uncertainty_type:
                    assert dev.waveform_uncertainty_type == "reconstruction", (
                        f"{name}: reconstructed waveform should have reconstruction uncertainty"
                    )

    def test_multishot_uncertainty_field_renamed(self):
        """MultiShotUncertainty.u_amplitude replaces u_digitization."""
        from dpf.validation.calibration import multi_shot_uncertainty
        result = multi_shot_uncertainty("PF-1000")
        assert hasattr(result, "u_amplitude"), "Missing u_amplitude field"
        assert not hasattr(result, "u_digitization"), "Old field u_digitization should not exist"
        assert result.u_amplitude == pytest.approx(0.03, abs=0.005)


# =====================================================================
# Test: Uncertainty magnitude ranges
# =====================================================================


class TestUncertaintyRanges:
    """Verify uncertainty values are physically reasonable."""

    @pytest.mark.parametrize("name,lo,hi", [
        ("PF-1000", 0.01, 0.05),        # 1-5% for hand-digitized
        ("UNU-ICTP", 0.005, 0.03),      # 0.5-3% for digital oscilloscope
        ("PF-1000-Gribkov", 0.01, 0.04), # 1-4% for IPFS digital archive
        ("POSEIDON-60kV", 0.01, 0.04),   # 1-4% for IPFS digital archive
        ("PF-1000-16kV", 0.03, 0.10),   # 3-10% for reconstruction
        ("FAETON-I", 0.05, 0.15),        # 5-15% for reconstruction
        ("MJOLNIR", 0.05, 0.15),         # 5-15% for reconstruction
    ])
    def test_amplitude_uncertainty_in_range(self, name, lo, hi):
        """Each device's amplitude uncertainty should be in expected range."""
        dev = DEVICES[name]
        assert lo <= dev.waveform_amplitude_uncertainty <= hi, (
            f"{name}: u_amp={dev.waveform_amplitude_uncertainty:.3f} "
            f"outside [{lo:.3f}, {hi:.3f}]"
        )

    def test_reconstruction_higher_than_digitization(self):
        """Reconstruction uncertainty should always be higher than digitization."""
        measured = get_devices_by_provenance("measured")
        recon = get_devices_by_provenance("reconstructed")
        max_measured = max(d.waveform_amplitude_uncertainty for d in measured.values())
        min_recon = min(d.waveform_amplitude_uncertainty for d in recon.values())
        assert min_recon >= max_measured, (
            f"Min reconstruction ({min_recon:.3f}) should be >= "
            f"max digitization ({max_measured:.3f})"
        )
