"""Tests for digitization provenance and verification gates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from dpf.validation import (
    a14_axis_calibration_draft_packets,
    a14_cikhardtova_fig6_extraction_blocker,
    a14_crop_boundary_review_status,
    a14_independent_review_handoff,
    a14_klir_fig2_timing_response_draft_packet,
    a14_remaining_extraction_backlog,
    a14_springham_fig5_gaussian_curve_draft_packet,
    a14_springham_fig5_monoenergetic_draft_packet,
    a14_table_extraction_draft_packets,
    akel_fig1_draft_digitization_packet,
    digitization_verification_evidence,
    scientific_closure_digitization_queue,
    scientific_closure_digitization_status,
    scientific_closure_source_acquisition_queue,
    sha256_file,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _copy_akel_source(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source_name = "radiation-physics-and-chemistry-188-2021-109633.md"
    source_bytes = (repo_root / "KnowledgeReference" / source_name).read_bytes()
    return _write(tmp_path / "KnowledgeReference" / source_name, source_bytes)


def _accepted_akel_fig1_packet(tmp_path: Path) -> dict[str, object]:
    source = _copy_akel_source(tmp_path)
    figure = _write(
        tmp_path / "KnowledgeReference" / "figures" / "akel-2021-fig1.png",
        b"fake figure image bytes",
    )
    return {
        "task_id": "akel_2021_fig1_current_waveform_shot_12581",
        "validation_scope": "pf1000_16kv_2021_akel",
        "packet_sha256": "synthetic-akel-accepted-packet-sha256",
        "source_path": (
            "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
        ),
        "source_sha256": sha256_file(source),
        "source_pdf_sha256": (
            "9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b"
        ),
        "source_lines": "294-295",
        "figure_image_path": "KnowledgeReference/figures/akel-2021-fig1.png",
        "figure_image_sha256": sha256_file(figure),
        "figure_id": "Fig. 1",
        "page": 3,
        "extraction_type": "figure",
        "axis_calibration": {
            "x": {
                "pixel_points": [50.0, 250.0],
                "data_values": [0.0, 10.0],
                "unit": "us",
                "rms_residual_px": 0.2,
            },
            "y": {
                "pixel_points": [300.0, 100.0],
                "data_values": [0.0, 2.0],
                "unit": "MA",
                "rms_residual_px": 0.2,
            },
        },
        "digitized_series": [
            {
                "name": "measured_current",
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.8, 1.5],
                "x_unit": "us",
                "y_unit": "MA",
            },
            {
                "name": "computed_current",
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.7, 1.4],
                "x_unit": "us",
                "y_unit": "MA",
            },
        ],
        "verification": {
            "overlay_rms_residual_px": 0.8,
            "independent_review_count": 1,
            "review_status": "accepted",
            "review_metadata": {
                "reviewed_packet_sha256": "synthetic-akel-accepted-packet-sha256",
                "reviewed_source_sha256": sha256_file(source),
                "reviewed_figure_image_sha256": sha256_file(figure),
                "task_id": "akel_2021_fig1_current_waveform_shot_12581",
                "validation_scope": "pf1000_16kv_2021_akel",
                "reviewer": "independent-reviewer",
                "review_date": "2026-05-08",
                "review_notes": "Independent review accepted the source-bound packet.",
                "decision": "accepted",
            },
        },
    }


def test_digitization_packet_passes_with_source_hashes_and_review(tmp_path):
    source = _write(
        tmp_path / "KnowledgeReference" / "verified-paper.md",
        b"verified source text",
    )
    figure = _write(
        tmp_path / "KnowledgeReference" / "figures" / "verified-paper-fig3.png",
        b"fake image bytes",
    )
    packet = {
        "packet_sha256": "synthetic-generic-packet-sha256",
        "source_path": "KnowledgeReference/verified-paper.md",
        "source_sha256": sha256_file(source),
        "figure_image_path": "KnowledgeReference/figures/verified-paper-fig3.png",
        "figure_image_sha256": sha256_file(figure),
        "figure_id": "Fig. 3",
        "page": 7,
        "extraction_type": "figure",
        "axis_calibration": {
            "x": {
                "pixel_points": [10.0, 210.0],
                "data_values": [0.0, 10.0],
                "unit": "us",
                "rms_residual_px": 0.2,
            },
            "y": {
                "pixel_points": [300.0, 100.0],
                "data_values": [0.0, 2.0],
                "unit": "MA",
                "rms_residual_px": 0.3,
            },
        },
        "digitized_series": [
            {
                "name": "current",
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.8, 1.4, 1.7],
                "x_unit": "us",
                "y_unit": "MA",
            },
        ],
        "verification": {
            "overlay_rms_residual_px": 0.8,
            "independent_review_count": 1,
            "review_status": "accepted",
            "review_metadata": {
                "reviewed_packet_sha256": "synthetic-generic-packet-sha256",
                "reviewed_source_sha256": sha256_file(source),
                "reviewed_figure_image_sha256": sha256_file(figure),
                "reviewer": "independent-reviewer",
                "review_date": "2026-05-08",
                "review_notes": "Independent review accepted the source-bound packet.",
                "decision": "accepted",
            },
        },
    }

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is True
    assert evidence["model_role"] == "digitized_source_verification_audit"
    assert evidence["missing_or_failed_checks"] == []


def test_digitization_packet_rejects_hash_mismatch_and_missing_review(tmp_path):
    source = _write(
        tmp_path / "KnowledgeReference" / "verified-paper.md",
        b"verified source text",
    )
    figure = _write(
        tmp_path / "KnowledgeReference" / "figures" / "verified-paper-fig3.png",
        b"fake image bytes",
    )
    packet = {
        "source_path": "KnowledgeReference/verified-paper.md",
        "source_sha256": sha256_file(source),
        "figure_image_path": "KnowledgeReference/figures/verified-paper-fig3.png",
        "figure_image_sha256": "not-the-real-hash",
        "figure_id": "Fig. 3",
        "page": 7,
        "axis_calibration": {
            "x": {
                "pixel_points": [10.0, 210.0],
                "data_values": [0.0, 10.0],
                "unit": "us",
                "rms_residual_px": 0.2,
            },
            "y": {
                "pixel_points": [300.0, 100.0],
                "data_values": [0.0, 2.0],
                "unit": "MA",
                "rms_residual_px": 0.3,
            },
        },
        "digitized_series": [
            {
                "name": "current",
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.8, 1.4],
                "x_unit": "us",
                "y_unit": "MA",
            },
        ],
        "verification": {
            "overlay_rms_residual_px": 0.8,
            "independent_review_count": 0,
            "review_status": "draft",
        },
    }
    assert figure.exists()

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is False
    assert "figure_image_sha256_mismatch" in evidence["missing_or_failed_checks"]
    assert "independent_review_missing" in evidence["missing_or_failed_checks"]
    assert "review_status_not_accepted" in evidence["missing_or_failed_checks"]


def test_digitization_packet_rejects_path_traversal_and_bad_review_count(tmp_path):
    source = _write(
        tmp_path / "KnowledgeReference" / "verified-paper.md",
        b"verified source text",
    )
    packet = {
        "source_path": "KnowledgeReference/../verified-paper.md",
        "source_sha256": sha256_file(source),
        "table_id": "Table 1",
        "page": 2,
        "extraction_type": "table",
        "digitized_series": [
            {
                "name": "yield",
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x_unit": "shot",
                "y_unit": "neutrons",
            },
        ],
        "verification": {
            "independent_review_count": "not-an-integer",
            "review_status": "accepted",
        },
    }

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is False
    assert "source_path_not_knowledge_reference" in (
        evidence["missing_or_failed_checks"]
    )
    assert "independent_review_missing" in evidence["missing_or_failed_checks"]


def test_scientific_closure_source_acquisition_queue_tracks_live_blockers():
    queue = scientific_closure_source_acquisition_queue()
    items = {item["group"]: item for item in queue["items"]}

    assert queue["model_role"] == "scientific_closure_source_acquisition_queue"
    assert queue["source_of_truth_rule"].startswith("Candidate links are not")
    assert queue["validation_scope"] == "pf1000_full_energy_2007_gribkov_scholz"
    assert "circuit_waveform" in items
    assert items["circuit_waveform"]["priority"] == 1
    assert "digitized_current_trace_points" in (
        items["circuit_waveform"]["required_data_to_complete"]
    )
    assert items["neutron_detector_response"]["physics_need"] == (
        "neutron_detector_or_activation_response"
    )
    assert items["neutron_yield"]["priority"] == 1
    assert items["neutron_yield"]["physics_need"] == (
        "absolute_or_shot_resolved_neutron_yield"
    )
    neutron_yield_local = {
        source["doi"]: source
        for source in items["neutron_yield"]["local_sources_available"]
    }
    assert neutron_yield_local["10.1016/j.radphyschem.2021.109633"][
        "local_status"
    ] == "parity_verified_knowledge_reference"
    assert neutron_yield_local["10.1016/j.radphyschem.2021.109633"][
        "local_kr_source"
    ] == "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    assert neutron_yield_local["10.1063/1.3559548"]["local_status"] == (
        "source_fidelity_reviewed_target_extraction_needed"
    )
    assert neutron_yield_local["10.1063/1.3559548"]["local_kr_source"] == (
        "KnowledgeReference/fusion-neutron-detector-for-time-of-flight-"
        "measurements-in-z-pinch-and-plasma-focus-214fbdae.md"
    )
    assert "yield_calibration_uncertainty" in (
        items["neutron_yield"]["required_data_to_complete"]
    )


def test_source_acquisition_queue_provides_candidate_links_for_blockers():
    queue = scientific_closure_source_acquisition_queue()

    assert queue["items"]
    for item in queue["items"]:
        candidates = item["candidate_sources"]
        assert candidates, item["group"]
        for candidate in candidates:
            assert candidate["doi"].startswith("10.")
            assert candidate["url"].startswith("https://doi.org/")
            assert candidate["why"]


def test_scientific_closure_digitization_queue_tracks_akel_figures():
    queue = scientific_closure_digitization_queue()
    tasks = {task["task_id"]: task for task in queue["items"]}

    assert queue["model_role"] == "scientific_closure_digitization_queue"
    assert queue["validation_scope"] == "pf1000_16kv_2021_akel"
    assert queue["source_path"] == (
        "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    )
    assert queue["source_pdf_sha256"] == (
        "9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b"
    )
    assert queue["summary"]["task_count"] == 6
    assert queue["summary"]["priority_1_count"] == 4
    assert queue["summary"]["extracted_figure_count"] == 1
    assert queue["summary"]["not_extracted_figure_count"] == 5
    assert queue["summary"]["draft_digitization_packet_count"] == 1

    for task in queue["items"]:
        assert task["digitization_gate"] == "digitization_verification_evidence"
        assert task["source_markdown_pdf_text_parity_passed"] is True
        assert "pdftoppm" in task["rendering_tool_candidates"]
        assert "figure_image_sha256" in task["required_digitization_packet_fields"]
        assert task["requires_independent_review"] is True

    fig1 = tasks["akel_2021_fig1_current_waveform_shot_12581"]
    assert fig1["shot"] == 12581
    assert fig1["page"] == 3
    assert fig1["source_lines"] == "294-295"
    assert fig1["figure_image_status"] == "extracted_not_digitized"
    assert fig1["figure_image_path"] == (
        "KnowledgeReference/figures/"
        "akel-2021-fig1-current-waveform-shot-12581.png"
    )
    assert fig1["figure_image_sha256"] == (
        "4c574525f1de413e54cd02bd06aa35d549db700270281310a3809edc54ab255e"
    )
    assert fig1["axis_calibration_candidate"]["x"]["data_values"] == [0.0, 10.0]
    assert fig1["axis_calibration_candidate"]["y"]["data_values"] == [0.0, 1400.0]
    extraction = fig1["series_extraction_candidate"]
    assert extraction["measured_current_candidate"]["point_count"] == 294
    assert extraction["computed_current_candidate"]["point_count"] == 34
    assert "legend glyphs" in extraction["legend_text_exclusion"]
    assert "not an accepted digitization packet" in extraction[
        "acceptance_boundary"
    ]
    assert fig1["draft_digitization_packet_status"] == "draft_unreviewed"
    assert fig1["draft_digitization_packet_path"] == (
        "KnowledgeReference/digitization/"
        "akel-2021-fig1-current-waveform-shot-12581-draft-packet.json"
    )
    assert fig1["draft_digitization_packet_sha256"] == (
        "abe4a283ee154f84f6061da8ea508d3871faf3b14dddb2d1cfc8a7a0a5f8e0e7"
    )
    assert "not accepted digitized waveform evidence" in fig1["extraction_note"]
    for task_id, task in tasks.items():
        if task_id != "akel_2021_fig1_current_waveform_shot_12581":
            assert task["figure_image_status"] == "not_extracted"

    assert tasks["akel_2021_fig4_current_waveform_shot_12604"]["pressure_torr"] == (
        1.05
    )
    assert tasks["akel_2021_fig5_neutron_yield_1p20_torr"]["group"] == (
        "neutron_yield"
    )
    assert tasks["akel_2021_fig5_neutron_yield_1p20_torr"][
        "table_backed_scalar_target"
    ] == "pf1000_16kv_shot_table_2021_akel"
    assert tasks["akel_2021_fig6_neutron_yield_1p05_torr"]["source_lines"] == "917"
    assert tasks["akel_2021_fig6_neutron_yield_1p05_torr"]["page"] == 5


def test_akel_digitization_queue_separates_waveform_and_yield_priorities():
    queue = scientific_closure_digitization_queue()
    current_tasks = [
        task for task in queue["items"] if task["group"] == "circuit_waveform"
    ]
    yield_tasks = [task for task in queue["items"] if task["group"] == "neutron_yield"]

    assert len(current_tasks) == 4
    assert len(yield_tasks) == 2
    assert {task["priority"] for task in current_tasks} == {1}
    assert {task["priority"] for task in yield_tasks} == {2}
    for task in current_tasks:
        assert task["required_series"] == [
            "measured_current",
            "computed_current",
        ]
        assert task["target_after_digitization"] == (
            "pf1000_16kv_current_waveform_targets"
        )
    for task in yield_tasks:
        assert task["required_series"] == [
            "measured_neutron_yield",
            "computed_neutron_yield",
        ]


def test_akel_fig1_extracted_image_matches_queue_hash():
    repo_root = Path(__file__).resolve().parents[1]
    queue = scientific_closure_digitization_queue()
    fig1 = {
        task["task_id"]: task for task in queue["items"]
    }["akel_2021_fig1_current_waveform_shot_12581"]

    figure_path = repo_root / fig1["figure_image_path"]

    assert figure_path.exists()
    assert sha256_file(figure_path) == fig1["figure_image_sha256"]


def test_akel_fig1_draft_digitization_packet_loads_candidate_arrays():
    packet = akel_fig1_draft_digitization_packet()
    series = {item["name"]: item for item in packet["digitized_series"]}

    assert packet["task_id"] == "akel_2021_fig1_current_waveform_shot_12581"
    assert packet["extraction_status"] == "draft_unreviewed"
    assert packet["source_sha256"] == (
        "31a68fe51d3ccc5b8181392ae18f66245d0b0926784371fb53eaf2306674cf7a"
    )
    assert packet["figure_image_sha256"] == (
        "4c574525f1de413e54cd02bd06aa35d549db700270281310a3809edc54ab255e"
    )
    assert packet["draft_packet_hash_verified"] is True
    assert packet["draft_packet_sha256"] == (
        "abe4a283ee154f84f6061da8ea508d3871faf3b14dddb2d1cfc8a7a0a5f8e0e7"
    )
    assert packet["verification"]["overlay_rms_residual_px"] == 0.213455189
    assert packet["verification"]["overlay_residual_source_svg_sha256"] == (
        "b045c3b7033e50bd355e025ecf7c40d96edc1ffc7fcb6ef26832fe065fe99d3f"
    )
    assert len(series["measured_current"]["x"]) == 294
    assert len(series["measured_current"]["y"]) == 294
    assert len(series["computed_current"]["x"]) == 34
    assert len(series["computed_current"]["y"]) == 34
    assert packet["draft_extraction_metadata"]["legend_text_exclusion"][
        "excluded_point_count"
    ] == 67


def test_akel_fig1_draft_digitization_packet_fails_only_review_gates():
    packet = akel_fig1_draft_digitization_packet()

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert set(evidence["missing_or_failed_checks"]) == {
        "independent_review_missing",
        "review_status_not_accepted",
    }


def test_a14_table_extraction_draft_packets_are_review_blocked():
    bundle = a14_table_extraction_draft_packets()

    assert bundle["model_role"] == "a14_table_extraction_draft_packets"
    assert bundle["packet_count"] == 6
    assert bundle["accepted_for_validation_count"] == 0
    assert bundle["draft_packet_path"] == (
        "KnowledgeReference/digitization/a14-2026-05-11-table-draft-packets.json"
    )
    assert bundle["draft_packet_sha256"]

    packets = {packet["task_id"]: packet for packet in bundle["packets"]}
    assert set(packets) == {
        "a14_springham_2021_table1_shot_counts",
        "a14_springham_2021_table2_four_mbar_activation_summary",
        "a14_catenacci_2020_table_i_shadow_bar_scale_factors",
        "a14_catenacci_2020_table_ii_detector_positions",
        "a14_catenacci_2020_table_iii_peak_energy_yield",
        "a14_catenacci_2020_table_iv_max_energy_comparison",
    }

    springham_table1 = packets["a14_springham_2021_table1_shot_counts"]
    assert springham_table1["table_id"] == "Table 1"
    assert springham_table1["extraction_type"] == "table"
    assert springham_table1["extraction_status"] == "draft_unreviewed"
    assert len(springham_table1["table_rows"]) == 10
    assert springham_table1["table_rows"][0]["d2_pressure_mbar"] == 1.5
    assert springham_table1["table_rows"][0]["op_shots"] is None

    table2 = packets["a14_springham_2021_table2_four_mbar_activation_summary"]
    table2_rows = {row["condition"]: row for row in table2["table_rows"]}
    assert table2_rows["w_o_op_0deg"]["mean_corrected_be_count"] == 75404.0
    assert table2_rows["op_90deg"]["effective_neutron_energy_mev"] == 2.55
    assert table2["draft_packet_item_sha256"]
    assert table2["draft_packet_bundle_sha256"] == bundle["draft_packet_sha256"]

    catenacci_table4 = packets[
        "a14_catenacci_2020_table_iv_max_energy_comparison"
    ]
    assert catenacci_table4["table_rows"][2]["predicted_max_energy_mev"] == 3.01
    assert catenacci_table4["table_rows"][2][
        "reconstructed_max_energy_mev"
    ] == 3.04

    for packet in bundle["packets"]:
        evidence = digitization_verification_evidence(packet)
        assert evidence["passed"] is False
        assert evidence["extraction_type"] == "table"
        assert set(evidence["missing_or_failed_checks"]) == {
            "independent_review_missing",
            "review_status_not_accepted",
        }


def _accepted_a14_table_packet() -> dict[str, object]:
    packet = deepcopy(a14_table_extraction_draft_packets()["packets"][0])
    verification = packet["verification"]
    verification["independent_review_count"] = 1
    verification["review_status"] = "accepted"
    verification["review_metadata"] = {
        "task_id": packet["task_id"],
        "validation_scope": packet["validation_scope"],
        "reviewed_packet_sha256": packet["draft_packet_item_sha256"],
        "reviewed_source_sha256": packet["source_sha256"],
        "reviewed_source_pdf_sha256": packet["source_pdf_sha256"],
        "reviewed_crop_image_sha256": packet["crop_image_sha256"],
        "reviewer": "independent-review-fixture",
        "review_date": "2026-05-11",
        "review_notes": "Synthetic accepted-review fixture for gate testing.",
        "decision": "accepted",
    }
    return packet


def test_a14_table_review_can_bind_to_item_and_crop_hashes():
    packet = _accepted_a14_table_packet()

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is True
    assert evidence["missing_or_failed_checks"] == []


def test_a14_table_review_rejects_crop_hash_mismatch():
    packet = _accepted_a14_table_packet()
    packet["verification"]["review_metadata"][
        "reviewed_crop_image_sha256"
    ] = "stale-crop-hash"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert "review_crop_image_hash_mismatch" in evidence["missing_or_failed_checks"]


def test_a14_table_review_rejects_source_pdf_hash_mismatch():
    packet = _accepted_a14_table_packet()
    packet["verification"]["review_metadata"][
        "reviewed_source_pdf_sha256"
    ] = "stale-source-pdf-hash"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert "review_source_pdf_hash_mismatch" in evidence["missing_or_failed_checks"]


def test_a14_table_packet_requires_current_crop_hash():
    packet = deepcopy(a14_table_extraction_draft_packets()["packets"][0])
    packet["crop_image_sha256"] = "stale-crop-hash"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert "crop_image_sha256_mismatch" in evidence["missing_or_failed_checks"]


def test_a14_table_packet_requires_current_source_pdf_hash():
    packet = deepcopy(a14_table_extraction_draft_packets()["packets"][0])
    packet["source_pdf_sha256"] = "stale-source-pdf-hash"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert "source_pdf_sha256_mismatch" in evidence["missing_or_failed_checks"]


def test_a14_crop_boundary_review_status_tracks_all_crops_and_blocks_validation():
    report = a14_crop_boundary_review_status()

    assert report["crop_boundary_review_path"] == (
        "docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.json"
    )
    assert report["crop_boundary_review_sha256"]
    assert report["source_report_path"] == (
        "docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.json"
    )
    assert report["total_crop_count"] == 36
    assert report["figure_crop_count"] == 30
    assert report["table_crop_count"] == 6
    assert report["accepted_for_validation_count"] == 0
    assert report["boundary_review_status_counts"] == {
        "boundary_ready_for_draft_extraction": 21,
        "draft_extracted_review_blocked": 6,
        "manual_review_required": 9,
    }
    assert report["recommended_next_axis_calibration_crops"] == [
        {
            "figure_id": "Fig. 6",
            "reason": "Clean 2D numeric plot with multiple labeled traces.",
            "source_slug": "cikhardtova-2015-linear-density",
        },
        {
            "figure_id": "Fig. 2",
            "reason": "Clean response-versus-voltage calibration plot with error bars.",
            "source_slug": "klir-2011-tof-detector",
        },
        {
            "figure_id": "Fig. 5",
            "reason": "Clean calibration curve with axes and legend fully visible.",
            "source_slug": "springham-2021-zrbe-activation",
        },
    ]

    entries = report["review_entries"]
    assert len(entries) == 36
    assert len({entry["crop_path"] for entry in entries}) == 36

    repo_root = Path(__file__).resolve().parents[1]
    table_entries = [entry for entry in entries if entry["extraction_kind"] == "table"]
    figure_entries = [
        entry for entry in entries if entry["extraction_kind"] == "figure"
    ]
    assert len(table_entries) == 6
    assert len(figure_entries) == 30
    assert {
        (entry["source_slug"], entry["figure_id"]) for entry in table_entries
    } == {
        ("springham-2021-zrbe-activation", "Table 1"),
        ("springham-2021-zrbe-activation", "Table 2"),
        ("catenacci-2020-neutron-tomography", "Table I"),
        ("catenacci-2020-neutron-tomography", "Table II"),
        ("catenacci-2020-neutron-tomography", "Table III"),
        ("catenacci-2020-neutron-tomography", "Table IV"),
    }

    for entry in table_entries:
        assert entry["boundary_review_status"] == "draft_extracted_review_blocked"
        assert entry["checklist"]["requires_numeric_extraction"] is False
        assert entry["accepted_for_validation"] is False
    figure_statuses = {
        (entry["source_slug"], entry["figure_id"]): entry["boundary_review_status"]
        for entry in figure_entries
    }
    assert figure_statuses[("cikhardtova-2015-linear-density", "Fig. 6")] == (
        "boundary_ready_for_draft_extraction"
    )
    assert figure_statuses[("klir-2011-tof-detector", "Fig. 2")] == (
        "boundary_ready_for_draft_extraction"
    )
    assert figure_statuses[("springham-2021-zrbe-activation", "Fig. 2")] == (
        "manual_review_required"
    )
    assert figure_statuses[("catenacci-2020-neutron-tomography", "Fig. 2")] == (
        "boundary_ready_for_draft_extraction"
    )
    assert figure_statuses[("catenacci-2020-neutron-tomography", "Fig. 1")] == (
        "manual_review_required"
    )
    for entry in figure_entries:
        assert entry["checklist"]["requires_numeric_extraction"] is True
        assert entry["accepted_for_validation"] is False
        if entry["boundary_review_status"] == "boundary_ready_for_draft_extraction":
            assert entry["checklist"]["axes_or_table_visible"] == "visually_ready"
        else:
            assert entry["checklist"]["axes_or_table_visible"] == (
                "manual_or_adjustment_review_needed"
            )

    for entry in entries:
        crop_path = repo_root / entry["crop_path"]
        assert crop_path.exists()
        assert sha256_file(crop_path) == entry["crop_sha256"]
        assert entry["crop_sha256"] == entry["source_report_crop_sha256"]
        assert entry["checklist"]["crop_file_exists"] is True
        assert entry["checklist"]["crop_hash_matches_report"] is True
        assert entry["validation_gate"] == "digitization_verification_evidence"


def test_a14_axis_calibration_draft_packets_are_non_promoting_scaffolds():
    bundle = a14_axis_calibration_draft_packets()

    assert bundle["model_role"] == "a14_axis_calibration_draft_packets"
    assert bundle["packet_count"] == 3
    assert bundle["accepted_for_validation_count"] == 0
    assert bundle["draft_packet_path"] == (
        "KnowledgeReference/digitization/"
        "a14-2026-05-11-axis-calibration-draft-packets.json"
    )
    assert bundle["draft_packet_sha256"]

    packets = {packet["task_id"]: packet for packet in bundle["packets"]}
    assert set(packets) == {
        "a14_cikhardtova_2015_fig6_axis_calibration_draft",
        "a14_klir_2011_fig2_axis_calibration_draft",
        "a14_springham_2021_fig5_axis_calibration_draft",
    }

    cikhardtova = packets["a14_cikhardtova_2015_fig6_axis_calibration_draft"]
    assert cikhardtova["figure_id"] == "Fig. 6"
    assert cikhardtova["source_lines"] == "200-222"
    assert cikhardtova["axis_calibration_candidate"]["x"]["data_range"] == [
        10.0,
        50.0,
    ]
    assert cikhardtova["axis_calibration_candidate"]["y"]["data_range"] == [
        5.0e17,
        5.0e18,
    ]
    assert len(cikhardtova["visible_series"]) == 5

    klir = packets["a14_klir_2011_fig2_axis_calibration_draft"]
    assert klir["axis_calibration_candidate"]["x"]["unit"] == "kV"
    assert klir["axis_calibration_candidate"]["y"]["unit"] == "ns"
    assert {series["name"] for series in klir["visible_series"]} == {
        "FWHM",
        "Rise time",
    }

    springham = packets["a14_springham_2021_fig5_axis_calibration_draft"]
    assert springham["axis_calibration_candidate"]["x"]["data_range"] == [
        0.0,
        0.14,
    ]
    assert springham["axis_calibration_candidate"]["y"]["unit"] == "MeV"
    assert "mono-energetic neutrons" in {
        series["name"] for series in springham["visible_series"]
    }

    repo_root = Path(__file__).resolve().parents[1]
    for packet in bundle["packets"]:
        assert packet["extraction_type"] == "figure"
        assert packet["extraction_status"] == "axis_calibration_draft_no_series"
        assert packet["digitized_series"] == []
        assert packet["accepted_for_validation"] is False
        assert packet["verification"]["independent_review_count"] == 0
        assert packet["verification"]["review_status"] == "draft_unreviewed"
        assert sha256_file(repo_root / packet["source_path"]) == packet["source_sha256"]
        assert sha256_file(repo_root / packet["source_pdf_path"]) == (
            packet["source_pdf_sha256"]
        )
        assert sha256_file(repo_root / packet["figure_image_path"]) == (
            packet["figure_image_sha256"]
        )


def test_a14_springham_fig5_monoenergetic_draft_packet_needs_review():
    packet = a14_springham_fig5_monoenergetic_draft_packet()

    assert packet["task_id"] == (
        "a14_springham_2021_fig5_monoenergetic_response_draft"
    )
    assert packet["figure_id"] == "Fig. 5"
    assert packet["source_lines"] == "546-616"
    assert packet["draft_packet_path"] == (
        "KnowledgeReference/digitization/"
        "a14-2026-05-11-springham-fig5-monoenergetic-draft-packet.json"
    )
    assert packet["draft_packet_sha256"]
    assert packet["accepted_for_validation"] is False

    series = packet["digitized_series"][0]
    assert series["name"] == "mono_energetic_neutrons_candidate"
    assert len(series["x"]) == 14
    assert len(series["y"]) == 14
    assert len(series["draft_pixel_points"]) == 14
    assert series["x"][0] == 0.002018
    assert series["y"][0] == 2.300414
    assert series["x"][-1] == 0.136585
    assert series["y"][-1] == 3.60
    assert packet["axis_calibration"]["x"]["data_values"] == [0.0, 0.14]
    assert packet["axis_calibration"]["y"]["data_values"] == [2.2, 3.6]

    assert packet["verification"]["overlay_rms_residual_px"] < 0.01
    assert packet["verification"]["overlay_max_residual_px"] < 0.01
    assert packet["verification"]["residual_status"] == "draft_round_trip_measured"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert evidence["extraction_type"] == "figure"
    assert set(evidence["missing_or_failed_checks"]) == {
        "independent_review_missing",
        "review_status_not_accepted",
    }


def test_a14_springham_fig5_gaussian_curve_draft_packet_needs_review():
    packet = a14_springham_fig5_gaussian_curve_draft_packet()

    assert packet["task_id"] == "a14_springham_2021_fig5_gaussian_curves_draft"
    assert packet["figure_id"] == "Fig. 5"
    assert packet["source_lines"] == "546-616"
    assert packet["draft_packet_path"] == (
        "KnowledgeReference/digitization/"
        "a14-2026-05-11-springham-fig5-gaussian-curves-draft-packet.json"
    )
    assert packet["draft_packet_sha256"]
    assert packet["accepted_for_validation"] is False

    series_by_name = {series["name"]: series for series in packet["digitized_series"]}
    assert set(series_by_name) == {
        "gaussian_peak_200kev_fwhm_candidate",
        "gaussian_peak_400kev_fwhm_candidate",
    }
    for series in series_by_name.values():
        assert len(series["x"]) == 13
        assert len(series["y"]) == 13
        assert len(series["draft_pixel_points"]) == 13
    assert series_by_name["gaussian_peak_200kev_fwhm_candidate"]["y"][0] == 2.339034
    assert series_by_name["gaussian_peak_400kev_fwhm_candidate"]["y"][-1] == 3.590345
    assert packet["verification"]["overlay_rms_residual_px"] < 0.01
    assert packet["verification"]["overlay_max_residual_px"] < 0.01

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert evidence["extraction_type"] == "figure"
    assert set(evidence["missing_or_failed_checks"]) == {
        "independent_review_missing",
        "review_status_not_accepted",
    }


def test_a14_klir_fig2_timing_response_draft_packet_needs_review():
    packet = a14_klir_fig2_timing_response_draft_packet()

    assert packet["task_id"] == "a14_klir_2011_fig2_timing_response_draft"
    assert packet["figure_id"] == "Fig. 2"
    assert packet["source_lines"] == "172-209"
    assert packet["draft_packet_path"] == (
        "KnowledgeReference/digitization/"
        "a14-2026-05-11-klir-fig2-timing-response-draft-packet.json"
    )
    assert packet["draft_packet_sha256"]
    assert packet["accepted_for_validation"] is False

    series_by_name = {series["name"]: series for series in packet["digitized_series"]}
    assert set(series_by_name) == {
        "fwhm_candidate",
        "rise_time_candidate",
    }
    assert series_by_name["fwhm_candidate"]["x"] == [
        1.0,
        1.1,
        1.2,
        1.4,
        1.6,
        1.9,
        2.2,
        2.4,
    ]
    assert series_by_name["fwhm_candidate"]["y"][0] == 3.935927
    assert series_by_name["rise_time_candidate"]["y"][-1] == 1.430206
    assert packet["draft_extraction_metadata"]["error_bar_status"] == (
        "Curve centerlines are sampled; numeric error-bar extents are not "
        "extracted in this packet."
    )
    assert packet["verification"]["overlay_rms_residual_px"] < 0.3
    assert packet["verification"]["overlay_max_residual_px"] < 0.6

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert evidence["extraction_type"] == "figure"
    assert set(evidence["missing_or_failed_checks"]) == {
        "independent_review_missing",
        "review_status_not_accepted",
    }


def test_a14_cikhardtova_fig6_extraction_blocker_records_manual_need():
    blocker = a14_cikhardtova_fig6_extraction_blocker()

    assert blocker["model_role"] == "a14_extraction_blocker_report"
    assert blocker["task_id"] == (
        "a14_cikhardtova_2015_fig6_linear_density_extraction_blocker"
    )
    assert blocker["blocker_path"] == (
        "docs/A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.json"
    )
    assert blocker["blocker_sha256"]
    assert blocker["figure_id"] == "Fig. 6"
    assert blocker["source_lines"] == "200-222"
    assert blocker["draft_extraction_status"] == (
        "blocked_manual_curve_separation_required"
    )
    assert blocker["accepted_for_validation"] is False
    assert len(blocker["visible_series"]) == 5
    assert "perform manual or vector-assisted curve separation for all five series" in (
        blocker["required_next_steps"]
    )

    repo_root = Path(__file__).resolve().parents[1]
    assert sha256_file(repo_root / blocker["source_path"]) == blocker["source_sha256"]
    assert sha256_file(repo_root / blocker["source_pdf_path"]) == (
        blocker["source_pdf_sha256"]
    )
    assert sha256_file(repo_root / blocker["figure_image_path"]) == (
        blocker["figure_image_sha256"]
    )


def test_a14_remaining_extraction_backlog_tracks_open_work():
    backlog = a14_remaining_extraction_backlog()

    assert backlog["model_role"] == "a14_remaining_extraction_backlog"
    assert backlog["backlog_path"] == (
        "docs/A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.json"
    )
    assert backlog["backlog_sha256"]
    assert backlog["total_crop_count"] == 36
    assert backlog["reviewable_draft_packet_count"] == 9
    assert backlog["distinct_reviewable_crop_count"] == 8
    assert backlog["accepted_for_validation_count"] == 0
    assert backlog["status_counts"] == {
        "extraction_blocked": 1,
        "manual_review_required": 9,
        "ready_not_started": 18,
        "reviewable_draft_packet_exists": 8,
    }

    items = {
        (item["source_slug"], item["figure_id"]): item
        for item in backlog["backlog_items"]
    }
    assert items[("cikhardtova-2015-linear-density", "Fig. 6")][
        "extraction_status"
    ] == "extraction_blocked"
    assert items[("klir-2011-tof-detector", "Fig. 2")][
        "extraction_status"
    ] == "reviewable_draft_packet_exists"
    assert len(
        items[("springham-2021-zrbe-activation", "Fig. 5")]["draft_packets"]
    ) == 2
    assert items[("catenacci-2020-neutron-tomography", "Fig. 2")][
        "extraction_status"
    ] == "ready_not_started"


def _accepted_a14_springham_fig5_packet() -> dict[str, object]:
    packet = deepcopy(a14_springham_fig5_monoenergetic_draft_packet())
    verification = packet["verification"]
    verification["independent_review_count"] = 1
    verification["review_status"] = "accepted"
    verification["review_metadata"] = {
        "task_id": packet["task_id"],
        "validation_scope": packet["validation_scope"],
        "reviewed_packet_sha256": packet["draft_packet_sha256"],
        "reviewed_source_sha256": packet["source_sha256"],
        "reviewed_source_pdf_sha256": packet["source_pdf_sha256"],
        "reviewed_figure_image_sha256": packet["figure_image_sha256"],
        "reviewer": "independent-review-fixture",
        "review_date": "2026-05-11",
        "review_notes": "Synthetic accepted-review fixture for gate testing.",
        "decision": "accepted",
    }
    return packet


def test_a14_springham_fig5_review_can_bind_current_hashes():
    packet = _accepted_a14_springham_fig5_packet()

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is True
    assert evidence["missing_or_failed_checks"] == []


def test_a14_springham_fig5_review_rejects_source_pdf_hash_mismatch():
    packet = _accepted_a14_springham_fig5_packet()
    packet["verification"]["review_metadata"][
        "reviewed_source_pdf_sha256"
    ] = "stale-source-pdf-hash"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert "review_source_pdf_hash_mismatch" in evidence["missing_or_failed_checks"]


def test_a14_springham_fig5_review_rejects_figure_hash_mismatch():
    packet = _accepted_a14_springham_fig5_packet()
    packet["verification"]["review_metadata"][
        "reviewed_figure_image_sha256"
    ] = "stale-figure-hash"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert "review_figure_image_hash_mismatch" in evidence["missing_or_failed_checks"]


def test_a14_independent_review_handoff_lists_only_review_blocked_drafts():
    handoff = a14_independent_review_handoff()

    assert handoff["model_role"] == "a14_independent_review_handoff"
    assert handoff["handoff_path"] == "docs/A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.json"
    assert handoff["handoff_sha256"]
    assert handoff["validation_gate"] == "digitization_verification_evidence"
    assert handoff["accepted_for_validation_count"] == 0
    assert handoff["review_item_count"] == 9
    assert handoff["axis_context_item_count"] == 3
    assert "reviewed_packet_sha256" in handoff["review_fields_required"]
    assert "reviewed_crop_image_sha256" in handoff["review_fields_required"]
    assert "review_status" in handoff["review_fields_required"]
    assert "series_or_table_values_checked_against_source" in (
        handoff["review_checklist_required"]
    )

    review_items = handoff["review_items"]
    assert len(review_items) == 9
    assert {
        item["task_id"] for item in review_items
    } == {
        "a14_springham_2021_table1_shot_counts",
        "a14_springham_2021_table2_four_mbar_activation_summary",
        "a14_catenacci_2020_table_i_shadow_bar_scale_factors",
        "a14_catenacci_2020_table_ii_detector_positions",
        "a14_catenacci_2020_table_iii_peak_energy_yield",
        "a14_catenacci_2020_table_iv_max_energy_comparison",
        "a14_springham_2021_fig5_monoenergetic_response_draft",
        "a14_springham_2021_fig5_gaussian_curves_draft",
        "a14_klir_2011_fig2_timing_response_draft",
    }
    assert sum(item["packet_kind"] == "table_draft_in_bundle" for item in review_items) == 6
    assert sum(item["packet_kind"] == "figure_digitization_draft" for item in review_items) == 3

    repo_root = Path(__file__).resolve().parents[1]
    for item in review_items:
        assert item["accepted_for_validation"] is False
        assert item["current_gate_passed"] is False
        assert set(item["current_gate_missing_or_failed_checks"]) == {
            "independent_review_missing",
            "review_status_not_accepted",
        }
        assert sha256_file(repo_root / item["packet_path"]) == (
            item["packet_bundle_sha256"]
        )
        assert sha256_file(repo_root / item["source_path"]) == item["source_sha256"]
        assert sha256_file(repo_root / item["source_pdf_path"]) == (
            item["source_pdf_sha256"]
        )
        assert sha256_file(repo_root / item["figure_or_crop_path"]) == (
            item["figure_or_crop_sha256"]
        )

    axis_items = handoff["axis_context_items"]
    assert len(axis_items) == 3
    assert {
        item["task_id"] for item in axis_items
    } == {
        "a14_cikhardtova_2015_fig6_axis_calibration_draft",
        "a14_klir_2011_fig2_axis_calibration_draft",
        "a14_springham_2021_fig5_axis_calibration_draft",
    }
    for item in axis_items:
        assert item["accepted_for_validation"] is False
        assert item["current_gate_status"] == (
            "context_only_axis_scaffold_no_digitized_series_or_residual"
        )
    assert any(
        artifact["artifact_role"] == "cikhardtova_fig6_extraction_blocker"
        for artifact in handoff["context_artifacts"]
    )
    assert any(
        artifact["artifact_role"] == "remaining_extraction_backlog"
        for artifact in handoff["context_artifacts"]
    )


def test_akel_fig1_status_flip_without_review_metadata_stays_blocked():
    packet = deepcopy(akel_fig1_draft_digitization_packet())
    packet["verification"]["independent_review_count"] = 1
    packet["verification"]["review_status"] = "accepted"

    evidence = digitization_verification_evidence(packet)

    assert evidence["passed"] is False
    assert evidence["missing_or_failed_checks"] == [
        "independent_review_metadata_missing",
    ]


def test_akel_fig1_review_rejects_stale_packet_hash(tmp_path):
    packet = _accepted_akel_fig1_packet(tmp_path)
    packet["verification"]["review_metadata"][
        "reviewed_packet_sha256"
    ] = "stale-packet-hash"

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is False
    assert "review_packet_hash_mismatch" in evidence["missing_or_failed_checks"]


def test_akel_fig1_review_requires_packet_hash(tmp_path):
    packet = _accepted_akel_fig1_packet(tmp_path)
    del packet["packet_sha256"]

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is False
    assert "packet_hash_missing" in evidence["missing_or_failed_checks"]


def test_akel_fig1_review_rejects_missing_reviewer_fields(tmp_path):
    packet = _accepted_akel_fig1_packet(tmp_path)
    review = packet["verification"]["review_metadata"]
    review["reviewer"] = ""
    review["review_date"] = ""
    review["review_notes"] = ""

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is False
    assert "reviewer_missing" in evidence["missing_or_failed_checks"]
    assert "review_date_missing" in evidence["missing_or_failed_checks"]
    assert "review_notes_missing" in evidence["missing_or_failed_checks"]


def test_akel_fig1_review_rejects_source_and_figure_hash_mismatch(tmp_path):
    packet = _accepted_akel_fig1_packet(tmp_path)
    review = packet["verification"]["review_metadata"]
    review["reviewed_source_sha256"] = "stale-source-hash"
    review["reviewed_figure_image_sha256"] = "stale-figure-hash"

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is False
    assert "review_source_hash_mismatch" in evidence["missing_or_failed_checks"]
    assert "review_figure_image_hash_mismatch" in evidence[
        "missing_or_failed_checks"
    ]


def test_akel_fig1_review_rejects_non_accepted_review_decision(tmp_path):
    packet = _accepted_akel_fig1_packet(tmp_path)
    packet["verification"]["review_status"] = "needs_revision"
    packet["verification"]["review_metadata"]["decision"] = "needs_revision"

    evidence = digitization_verification_evidence(packet, base_path=tmp_path)

    assert evidence["passed"] is False
    assert "review_status_not_accepted" in evidence["missing_or_failed_checks"]
    assert "review_decision_not_accepted" in evidence["missing_or_failed_checks"]


def test_scientific_closure_digitization_status_starts_open():
    status = scientific_closure_digitization_status()

    assert status["model_role"] == "scientific_closure_digitization_status"
    assert status["queue_complete"] is False
    assert status["accepted_task_count"] == 0
    assert status["failed_task_count"] == 0
    assert status["open_task_count"] == 6
    assert "akel_2021_fig1_current_waveform_shot_12581" in (
        status["missing_or_failed_tasks"]
    )
    fig1_status = {
        item["task_id"]: item for item in status["task_statuses"]
    }["akel_2021_fig1_current_waveform_shot_12581"]
    assert fig1_status["figure_image_status"] == "extracted_not_digitized"
    assert fig1_status["figure_image_path"] == (
        "KnowledgeReference/figures/"
        "akel-2021-fig1-current-waveform-shot-12581.png"
    )
    assert fig1_status["draft_digitization_packet_status"] == "draft_unreviewed"
    assert fig1_status["draft_digitization_packet_path"] == (
        "KnowledgeReference/digitization/"
        "akel-2021-fig1-current-waveform-shot-12581-draft-packet.json"
    )


def test_scientific_closure_digitization_status_reports_draft_fig1_as_failed():
    packet = akel_fig1_draft_digitization_packet()

    status = scientific_closure_digitization_status([packet])
    fig1_status = {
        item["task_id"]: item for item in status["task_statuses"]
    }["akel_2021_fig1_current_waveform_shot_12581"]

    assert status["queue_complete"] is False
    assert status["accepted_task_count"] == 0
    assert status["failed_task_count"] == 1
    assert status["open_task_count"] == 5
    assert fig1_status["status"] == "failed"
    assert fig1_status["draft_digitization_packet_status"] == "draft_unreviewed"
    assert set(fig1_status["missing_or_failed_checks"]) == {
        "independent_review_missing",
        "review_status_not_accepted",
    }
    assert "digitization_packet_missing" not in (
        fig1_status["missing_or_failed_checks"]
    )


def test_scientific_closure_digitization_status_accepts_verified_task(tmp_path):
    packet = _accepted_akel_fig1_packet(tmp_path)

    status = scientific_closure_digitization_status([packet], base_path=tmp_path)
    fig1_status = {
        item["task_id"]: item for item in status["task_statuses"]
    }["akel_2021_fig1_current_waveform_shot_12581"]

    assert status["queue_complete"] is False
    assert status["accepted_task_count"] == 1
    assert status["open_task_count"] == 5
    assert fig1_status["status"] == "accepted"
    assert fig1_status["missing_or_failed_checks"] == []


def test_scientific_closure_digitization_status_requires_task_series(tmp_path):
    packet = _accepted_akel_fig1_packet(tmp_path)
    packet["digitized_series"] = packet["digitized_series"][:1]

    status = scientific_closure_digitization_status([packet], base_path=tmp_path)
    fig1_status = {
        item["task_id"]: item for item in status["task_statuses"]
    }["akel_2021_fig1_current_waveform_shot_12581"]

    assert status["accepted_task_count"] == 0
    assert status["failed_task_count"] == 1
    assert fig1_status["status"] == "failed"
    assert "missing_required_series:computed_current" in (
        fig1_status["missing_or_failed_checks"]
    )
