from __future__ import annotations

import json
from pathlib import Path

from dpf.first_principles.figure_source_manifest import (
    DEFAULT_PHASE5_FIGURE_SOURCE_MANIFEST_PATH,
    build_phase5_figure_source_packets,
    load_phase5_figure_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_phase5_builder_loads_manifest() -> None:
    manifest = load_phase5_figure_source_manifest()

    assert DEFAULT_PHASE5_FIGURE_SOURCE_MANIFEST_PATH.exists()
    assert manifest["manifest_id"] == "ss12_p1_phase5_figure_source_manifest"
    assert len(manifest["figure_sources"]) >= 4


def test_phase5_builder_stages_all_manifest_rows_without_acceptance() -> None:
    packet = build_phase5_figure_source_packets()

    assert packet["status"] == "staged_phase5_figure_sources_not_accepted"
    assert packet["accepted_figure_claim"] is False
    assert packet["accepted_observable_claim"] is False
    assert packet["promotes_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["summary"]["total_rows"] >= 4
    assert packet["summary"]["accepted_rows"] == 0
    assert packet["summary"]["staged_packets"] == packet["summary"]["total_rows"]

    for staged in packet["staged_figure_packets"]:
        assert staged["accepted_observable_claim"] is False
        assert staged["can_support_first_principles_acceptance"] is False
        assert "review_certificate_missing" in staged["blocking_reasons"]


def test_phase5_builder_keeps_transfer_density_blocked() -> None:
    packet = build_phase5_figure_source_packets()
    density = next(
        staged
        for staged in packet["staged_figure_packets"]
        if staged["channel"] == "density_history"
    )

    assert density["scope_classification"] == "transfer_figure_candidate"
    assert "scope_not_same_source_accepted" in density["blocking_reasons"]
    assert density["accepted_observable_claim"] is False


def test_phase5_builder_missing_manifest_fails_closed(tmp_path) -> None:
    packet = build_phase5_figure_source_packets(manifest_path=tmp_path / "missing.json")

    assert packet["status"] == "blocked_phase5_manifest_missing"
    assert packet["accepted_figure_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert "phase5_figure_source_manifest_missing" in packet["blocking_reasons"]


def test_phase5_builder_rejects_invalid_manifest_before_staging(tmp_path) -> None:
    manifest = load_phase5_figure_source_manifest()
    manifest["figure_sources"][0]["status"] = "accepted"
    mutated = tmp_path / "accepted_manifest.json"
    mutated.write_text(json.dumps(manifest))

    packet = build_phase5_figure_source_packets(manifest_path=mutated)

    assert packet["status"] == "blocked_phase5_manifest_invalid"
    assert packet["accepted_figure_claim"] is False
    assert "figure_source_accepted_status_forbidden" in packet["blocking_reasons"]


def test_phase5_builder_rejects_source_path_escape_before_staging(tmp_path) -> None:
    manifest = load_phase5_figure_source_manifest()
    manifest["figure_sources"][0]["source_path"] = "../outside.md"
    mutated = tmp_path / "escape_manifest.json"
    mutated.write_text(json.dumps(manifest))

    packet = build_phase5_figure_source_packets(manifest_path=mutated)

    assert packet["status"] == "blocked_phase5_manifest_invalid"
    assert "figure_source_outside_repo" in packet["blocking_reasons"]
