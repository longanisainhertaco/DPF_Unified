from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from dpf.first_principles import (
    build_engineering_current_waveform_comparator,
    extract_all_gv_current_waveform_packets,
    extract_gv_current_waveform_packet,
    gv_waveform_packet_summary,
)

GV_ROOT = Path("/Users/anthonyzamora/Downloads/GV")

pytestmark = pytest.mark.skipif(
    not (GV_ROOT / "PF-24-KRAKOW-16092202.xlsx").exists(),
    reason="verified local GV workbook bundle is not available",
)


def test_extract_gv_pf24_current_waveform_packet_is_non_promoting() -> None:
    packet = extract_gv_current_waveform_packet("pf24_krakow_16092202")

    assert packet["task_id"] == "gv_pf24_krakow_16092202_current_waveform_candidate"
    assert packet["validation_scope"] == "gv_verified_pf24_krakow_16092202"
    assert packet["accepted_for_first_principles_validation"] is False
    assert packet["can_seed_engineering_comparator"] is True
    assert packet["workbook"]["time_column"] == "L"
    assert packet["workbook"]["current_column"] == "M"
    assert packet["workbook"]["series_kind"] == "workbook_experimental_waveform"
    assert packet["source_hashes"]["workbook_hash_matches_expected"] is True

    series = packet["digitized_series"][0]
    assert series["name"] == "measured_current"
    assert series["x_unit"] == "us"
    assert series["y_unit"] == "kA"
    assert series["point_count"] == 649
    assert len(series["x"]) == 649
    assert len(series["y"]) == 649
    assert packet["summary"]["time_min_us"] == pytest.approx(-0.5)
    assert packet["summary"]["time_max_us"] == pytest.approx(6.0)
    assert packet["summary"]["current_max_kA"] == pytest.approx(401.6)
    assert packet["summary"]["monotonic_time_non_decreasing"] is True
    assert packet["verification"]["review_status"] == "candidate_not_reviewed"
    assert packet["verification"]["accepted_for_validation"] is False
    assert "per_point_current_uncertainty" in packet[
        "missing_for_first_principles_acceptance"
    ]


def test_extract_gv_preferred_uses_raw_columns_when_available() -> None:
    packet = extract_gv_current_waveform_packet("pf360_20140122_7")

    assert packet["workbook"]["time_column"] == "AC"
    assert packet["workbook"]["current_column"] == "AD"
    assert packet["workbook"]["series_kind"] == "raw_workbook_experimental_waveform"
    assert packet["digitized_series"][0]["name"] == "raw_measured_current"
    assert packet["summary"]["point_count"] == 22980
    assert packet["summary"]["time_min_us"] == pytest.approx(-0.5564)
    assert packet["summary"]["current_max_kA"] == pytest.approx(2015.325)

    smoothed = extract_gv_current_waveform_packet(
        "pf360_20140122_7",
        series="smoothed",
    )
    assert smoothed["workbook"]["time_column"] == "L"
    assert smoothed["workbook"]["current_column"] == "M"
    assert smoothed["digitized_series"][0]["name"] == "smoothed_measured_current"
    assert smoothed["summary"]["point_count"] == 22581


def test_gv_waveform_packet_summary_omits_full_arrays() -> None:
    packet = extract_gv_current_waveform_packet("pf24_krakow_16092202")
    summary = gv_waveform_packet_summary(packet)

    assert summary["task_id"] == packet["task_id"]
    assert summary["point_count"] == 649
    assert summary["series"] == "measured_current"
    assert "digitized_series" not in summary
    assert summary["accepted_for_first_principles_validation"] is False
    assert len(summary["packet_sha256"]) == 64


def test_gv_waveform_can_seed_nonpromoting_engineering_comparator() -> None:
    comparator = build_engineering_current_waveform_comparator(
        declared_scope="gv_pf24_krakow_16092202_engineering_candidate",
        device_name="PF-24-KRAKOW",
        validation_targets=[
            {
                "name": "PF-24 workbook current waveform",
                "observable": "current_waveform",
                "status": "user_verified_waveform_candidate_not_comparator_bound",
                "source_reference": {
                    "path": str(GV_ROOT / "PF-24-KRAKOW-16092202.xlsx"),
                    "sha256": (
                        "43ef75fd63caf1aaa4fc7be72b6e92c63851c6e9220daa6291be974a92a02e73"
                    ),
                    "record_id": "gv:pf24_krakow_16092202:workbook",
                },
            }
        ],
        simulation_telemetry={
            "circuit": {
                "current_history": [
                    {"time_us": 0.0, "current_A": 0.0},
                    {"time_us": 0.1, "current_A": 1.0e3},
                    {"time_us": 0.2, "current_A": 2.0e3},
                ]
            }
        },
    )

    assert comparator["status"] == "engineering_current_waveform_comparison_not_validation"
    assert comparator["target_packet"]["shot_id"] == "pf24_krakow_16092202"
    assert comparator["series_counts"]["simulation_points"] == 3
    assert comparator["series_counts"]["overlap_points"] == 3
    assert comparator["metrics"]["rmse_kA"] > 0.0
    assert comparator["first_principles_policy"]["experimental_waveform_used_as_drive"] is False
    assert comparator["first_principles_policy"]["reduced_model_used"] is False
    assert comparator["can_support_first_principles_acceptance"] is False
    assert "whole_shot_temporal_coverage" in comparator["missing_for_acceptance"]


def test_extract_all_gv_waveform_packets_covers_eight_verified_shots() -> None:
    packets = extract_all_gv_current_waveform_packets()

    assert len(packets) == 8
    assert {packet["shot_id"] for packet in packets} >= {
        "lpp_ff1_05_24_16_6",
        "pf24_krakow_16092202",
        "pf360_20140122_7",
        "gemini_rog_i005_20130716",
        "onesys_rog01004_20051208",
    }
    assert all(packet["accepted_for_first_principles_validation"] is False for packet in packets)


def test_extract_gv_raw_series_fails_when_raw_columns_are_absent() -> None:
    with pytest.raises(ValueError, match="raw"):
        extract_gv_current_waveform_packet("pf24_krakow_16092202", series="raw")


def test_first_principles_gv_waveform_cli_summary() -> None:
    from dpf.cli.main import cli

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-gv-waveform",
            "--shot-id",
            "pf24_krakow_16092202",
            "--summary",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["task_id"] == "gv_pf24_krakow_16092202_current_waveform_candidate"
    assert payload["point_count"] == 649
    assert payload["accepted_for_first_principles_validation"] is False
    assert "digitized_series" not in payload
