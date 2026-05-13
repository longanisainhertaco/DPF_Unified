from dpf.validation.source_acquisition import (
    scientific_closure_source_acquisition_queue,
)


def test_source_acquisition_queue_reports_summary_counts():
    queue = scientific_closure_source_acquisition_queue()

    assert queue["summary"]["blocker_count"] == len(queue["items"])
    assert queue["summary"]["priority_1_count"] == 5
    assert queue["summary"]["priority_2_count"] == 5
    assert queue["summary"]["partial_group_count"] == 10
    assert queue["summary"]["missing_group_count"] == 0
    assert queue["summary"]["complete_group_count"] >= 1
    assert queue["summary"]["local_digitization_or_target_extraction_count"] >= 1
    assert queue["summary"]["user_acquisition_required_count"] >= 1


def test_source_acquisition_queue_reports_all_same_scope_group_statuses():
    queue = scientific_closure_source_acquisition_queue()
    statuses = {
        item["group"]: item
        for item in queue["same_scope_group_statuses"]
    }

    assert statuses["spatial_density"]["status"] == "complete_in_current_scope"
    assert statuses["spatial_density"]["blocks_validation_tiers"] == [
        "tier_4_spatial_validation"
    ]
    assert statuses["spatial_magnetic_or_em"]["status"] == (
        "partial_in_current_scope"
    )
    assert statuses["neutron_detector_response"]["status"] == (
        "partial_in_current_scope"
    )
    assert statuses["uncertainty"]["blocks_validation_tiers"] == [
        "tier_1_circuit_waveform",
        "tier_2_phase_validation",
        "tier_4_spatial_validation",
        "tier_5_neutron_validation",
        "high_fidelity_readiness",
    ]


def test_source_acquisition_queue_names_source_action_and_blocked_tiers():
    queue = scientific_closure_source_acquisition_queue()
    items = {item["group"]: item for item in queue["items"]}

    assert items["circuit_waveform"]["source_action"] == (
        "local_digitization_or_target_extraction"
    )
    assert items["neutron_detector_response"]["source_action"] == (
        "local_digitization_or_target_extraction"
    )
    neutron_detector_acquisition = {
        source["doi"]
        for source in items["neutron_detector_response"][
            "candidate_sources_for_acquisition"
        ]
    }
    assert "10.1515/nuka-2017-0003" in neutron_detector_acquisition
    neutron_detector_local = {
        source["doi"]
        for source in items["neutron_detector_response"]["local_sources_available"]
    }
    assert "10.1063/1.3559548" in neutron_detector_local
    assert items["phase_timing"]["blocks_validation_tiers"] == [
        "tier_2_phase_validation"
    ]
    assert items["neutron_spectrum"]["blocks_validation_tiers"] == [
        "tier_5_neutron_validation"
    ]
