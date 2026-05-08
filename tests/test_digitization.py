"""Tests for digitization provenance and verification gates."""

from __future__ import annotations

from pathlib import Path

from dpf.validation import (
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
    neutron_yield_acquisition = {
        source["doi"]: source
        for source in items["neutron_yield"]["candidate_sources_for_acquisition"]
    }
    assert neutron_yield_local["10.1016/j.radphyschem.2021.109633"][
        "local_status"
    ] == "parity_verified_knowledge_reference"
    assert neutron_yield_local["10.1016/j.radphyschem.2021.109633"][
        "local_kr_source"
    ] == "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    assert neutron_yield_acquisition["10.1063/1.3559548"]["local_status"] == (
        "not_found_as_exact_local_pdf"
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
