from __future__ import annotations

import json

from dpf.first_principles.figure_asset_inventory import (
    DEFAULT_PHASE5B_FIGURE_ASSET_INVENTORY_PATH,
    build_phase5b_extraction_packet_plan,
    load_phase5b_figure_asset_inventory,
)


def test_phase5b_loads_default_asset_inventory() -> None:
    inventory = load_phase5b_figure_asset_inventory()

    assert DEFAULT_PHASE5B_FIGURE_ASSET_INVENTORY_PATH.exists()
    assert inventory["inventory_id"] == "ss12_p1_phase5b_figure_asset_inventory"
    assert len(inventory["figure_assets"]) == 4


def test_phase5b_extraction_packet_plan_is_fail_closed() -> None:
    plan = build_phase5b_extraction_packet_plan()

    assert plan["status"] == "phase5b_extraction_assets_located_not_extracted"
    assert plan["accepted_asset_claim"] is False
    assert plan["accepted_digitization_claim"] is False
    assert plan["accepted_runtime_claim"] is False
    assert plan["can_support_first_principles_acceptance"] is False
    assert plan["promotes_acceptance"] is False
    assert plan["summary"]["total_assets"] == 4
    assert plan["summary"]["assets_located"] == 4
    assert plan["summary"]["extracted_packets"] == 0
    assert set(plan["next_required_actions"]) >= {
        "crop_figure_region",
        "compute_region_or_digitization_hash",
        "calibrate_axes",
        "assign_uncertainty_budget",
        "attach_review_certificate",
    }


def test_phase5b_extraction_tasks_keep_one_task_per_asset() -> None:
    plan = build_phase5b_extraction_packet_plan()
    task_ids = {task["figure_source_id"] for task in plan["extraction_tasks"]}

    assert task_ids == {
        "pf1000_recent_progress_fig6_current_waveform",
        "pf1000_scholz_fig4_density_distribution",
        "pf1000_krauz_fig8_magnetic_probe_current",
        "pf1000_scholz_fig9_neutron_timing",
    }
    for task in plan["extraction_tasks"]:
        assert task["task_status"] == "planned_not_executed"
        assert task["digitization_hash"] is None
        assert task["accepted_task_claim"] is False
        assert task["promotes_acceptance"] is False


def test_phase5b_plan_missing_inventory_fails_closed(tmp_path) -> None:
    plan = build_phase5b_extraction_packet_plan(inventory_path=tmp_path / "missing.json")

    assert plan["status"] == "blocked_phase5b_asset_inventory_missing"
    assert plan["accepted_asset_claim"] is False
    assert plan["can_support_first_principles_acceptance"] is False
    assert "phase5b_asset_inventory_missing" in plan["blocking_reasons"]


def test_phase5b_plan_rejects_invalid_inventory_before_task_generation(tmp_path) -> None:
    inventory = load_phase5b_figure_asset_inventory()
    inventory["figure_assets"][0]["accepted_asset_claim"] = True
    mutated = tmp_path / "invalid_inventory.json"
    mutated.write_text(json.dumps(inventory))

    plan = build_phase5b_extraction_packet_plan(inventory_path=mutated)

    assert plan["status"] == "blocked_phase5b_asset_inventory_invalid"
    assert plan["accepted_asset_claim"] is False
    assert "asset_acceptance_flag_not_false" in plan["blocking_reasons"]
