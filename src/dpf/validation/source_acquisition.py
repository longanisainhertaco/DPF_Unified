"""Source-acquisition queue for high-fidelity DPF scientific closure."""

from __future__ import annotations

from collections.abc import Mapping

from dpf.validation.kr_targets import kr_validation_same_scope_target_report

_PHYSICS_NEED_BY_GROUP = {
    "circuit_waveform": "circuit_current_and_voltage",
    "phase_timing": "rundown_radial_pinch_phase_timing",
    "spatial_density": "spatial_density",
    "spatial_magnetic_or_em": "spatial_magnetic_or_electromagnetic_field",
    "spatial_temperature": "spatial_temperature",
    "neutron_yield": "absolute_or_shot_resolved_neutron_yield",
    "neutron_timing": "neutron_time_history",
    "neutron_spectrum": "neutron_energy_spectrum",
    "neutron_anisotropy": "neutron_angular_anisotropy",
    "neutron_detector_response": "neutron_detector_or_activation_response",
    "uncertainty": "uncertainty_budget",
}


_BLOCKED_VALIDATION_TIERS_BY_GROUP = {
    "circuit_waveform": ["tier_1_circuit_waveform", "akel_s1_s2_waveform"],
    "phase_timing": ["tier_2_phase_validation"],
    "spatial_density": ["tier_4_spatial_validation"],
    "spatial_magnetic_or_em": ["tier_4_spatial_validation"],
    "spatial_temperature": ["tier_4_spatial_validation"],
    "neutron_yield": ["tier_5_neutron_validation"],
    "neutron_timing": ["tier_5_neutron_validation"],
    "neutron_spectrum": ["tier_5_neutron_validation"],
    "neutron_anisotropy": ["tier_5_neutron_validation"],
    "neutron_detector_response": ["tier_5_neutron_validation"],
    "uncertainty": [
        "tier_1_circuit_waveform",
        "tier_2_phase_validation",
        "tier_4_spatial_validation",
        "tier_5_neutron_validation",
        "high_fidelity_readiness",
    ],
}


_ACQUISITION_PROCESS = [
    "AI researches candidate source documents and provides links.",
    "User acquires the correct source document.",
    "Document is added locally under KnowledgeReference.",
    "Codex reviews the local document under the KR-only rule.",
    "If needed, digitization is performed with one-for-one verification.",
    "Typed KR targets are extracted and same-scope closure is rerun.",
]


_CANDIDATE_SOURCES_BY_GROUP = {
    "circuit_waveform": [
        {
            "title": (
                "Comparison of measured and computed neutron yield from "
                "PF1000 plasma focus device operated with deuterium gas"
            ),
            "authors_year": "Akel, Kubes, Paduch, Lee (2021)",
            "doi": "10.1016/j.radphyschem.2021.109633",
            "url": "https://doi.org/10.1016/j.radphyschem.2021.109633",
            "why": "Measured PF-1000 current wave shapes at 16 kV and neutron yield.",
        },
        {
            "title": (
                "Plasma dynamics in PF-1000 device under full-scale energy "
                "storage: I. Pinch dynamics, shock-wave diffraction, and "
                "inertial electrode"
            ),
            "authors_year": "Gribkov et al. (2007)",
            "doi": "10.1088/0022-3727/40/7/021",
            "url": "https://doi.org/10.1088/0022-3727/40/7/021",
            "why": "Full-energy PF-1000 electrical and phase diagnostics.",
        },
    ],
    "phase_timing": [
        {
            "title": (
                "Plasma dynamics in PF-1000 device under full-scale energy "
                "storage: I. Pinch dynamics, shock-wave diffraction, and "
                "inertial electrode"
            ),
            "authors_year": "Gribkov et al. (2007)",
            "doi": "10.1088/0022-3727/40/7/021",
            "url": "https://doi.org/10.1088/0022-3727/40/7/021",
            "why": "Breakdown, rundown, radial collapse, pinch, x-ray, and neutron timing context.",
        },
        {
            "title": (
                "Measuring characteristic differences between high- and "
                "low-performing discharges on the MJOLNIR DPF"
            ),
            "authors_year": "Schmidt et al. (2022)",
            "doi": "10.1063/5.0089121",
            "url": "https://doi.org/10.1063/5.0089121",
            "why": "Current traces, optical gates, framing-camera velocities, and restrike context.",
        },
    ],
    "spatial_density": [
        {
            "title": (
                "Comparison of density profiles measured via laser "
                "interferometry with MHD simulations during shock wave "
                "reflection on mega-ampere dense plasma focus"
            ),
            "authors_year": "Malir et al. (2024)",
            "doi": "10.1063/5.0193268",
            "url": "https://doi.org/10.1063/5.0193268",
            "why": "PF-1000 same-device interferometric density profiles and uncertainty context.",
        },
        {
            "title": "Temporal distribution of linear densities of the plasma column in a plasma focus discharge",
            "authors_year": "Cikhardtova et al. (2015)",
            "doi": "10.1515/nuka-2015-0065",
            "url": "https://doi.org/10.1515/nuka-2015-0065",
            "why": "PF-1000 multi-frame interferometry and plasma column linear density timing.",
        },
    ],
    "spatial_magnetic_or_em": [
        {
            "title": (
                "Plasma dynamics in PF-1000 device under full-scale energy "
                "storage: I. Pinch dynamics, shock-wave diffraction, and "
                "inertial electrode"
            ),
            "authors_year": "Gribkov et al. (2007)",
            "doi": "10.1088/0022-3727/40/7/021",
            "url": "https://doi.org/10.1088/0022-3727/40/7/021",
            "why": "Magnetic-probe/Faraday-current-pinch context for PF-1000.",
        },
        {
            "title": (
                "Measuring characteristic differences between high- and "
                "low-performing discharges on the MJOLNIR DPF"
            ),
            "authors_year": "Schmidt et al. (2022)",
            "doi": "10.1063/5.0089121",
            "url": "https://doi.org/10.1063/5.0089121",
            "why": "Restrike and parasitic-current evidence relevant to EM coupling closure.",
        },
    ],
    "spatial_temperature": [
        {
            "title": (
                "Plasma dynamics in PF-1000 device under full-scale energy "
                "storage: I. Pinch dynamics, shock-wave diffraction, and "
                "inertial electrode"
            ),
            "authors_year": "Gribkov et al. (2007)",
            "doi": "10.1088/0022-3727/40/7/021",
            "url": "https://doi.org/10.1088/0022-3727/40/7/021",
            "why": "PF-1000 pinch temperature estimates and diagnostic context.",
        },
        {
            "title": (
                "Neutron generation dynamics inside a MA-class dense plasma "
                "focus Z-pinch"
            ),
            "authors_year": "Goyon et al. (2025)",
            "doi": "10.1063/5.0253547",
            "url": "https://doi.org/10.1063/5.0253547",
            "why": "MJOLNIR stagnation-temperature/neutron mechanism comparison for physics closure.",
        },
    ],
    "neutron_timing": [
        {
            "title": (
                "Plasma dynamics in the PF-1000 device under full-scale energy "
                "storage: II. Fast electron and ion characteristics versus "
                "neutron emission parameters and gun optimization perspectives"
            ),
            "authors_year": "Gribkov et al. (2007)",
            "doi": "10.1088/0022-3727/40/12/008",
            "url": "https://doi.org/10.1088/0022-3727/40/12/008",
            "why": "PF-1000 fast ion/electron timing versus neutron emission.",
        },
        {
            "title": (
                "Neutron generation dynamics inside a MA-class dense plasma "
                "focus Z-pinch"
            ),
            "authors_year": "Goyon et al. (2025)",
            "doi": "10.1063/5.0253547",
            "url": "https://doi.org/10.1063/5.0253547",
            "why": "Neutron pulse-shape comparison and mechanism-separated timing.",
        },
    ],
    "neutron_spectrum": [
        {
            "title": "Measurements of fast ions and neutrons emitted from PF-1000 plasma focus device",
            "authors_year": "Sadowski/Scholz/PF-1000 team (2004)",
            "doi": "10.1016/j.vacuum.2004.07.040",
            "url": "https://doi.org/10.1016/j.vacuum.2004.07.040",
            "why": "PF-1000 neutron spectra, anisotropy, yield, and fast-ion measurements.",
        },
        {
            "title": "Tomographic Reconstruction of the Neutron Time-Energy Spectrum from a Dense Plasma Focus",
            "authors_year": "Catenacci et al. (2020)",
            "doi": "10.1109/TPS.2020.3012104",
            "url": "https://doi.org/10.1109/TPS.2020.3012104",
            "why": "DPF neutron time-energy spectrum reconstruction method with scatter subtraction.",
        },
    ],
    "neutron_anisotropy": [
        {
            "title": "Measurements of fast ions and neutrons emitted from PF-1000 plasma focus device",
            "authors_year": "Sadowski/Scholz/PF-1000 team (2004)",
            "doi": "10.1016/j.vacuum.2004.07.040",
            "url": "https://doi.org/10.1016/j.vacuum.2004.07.040",
            "why": "PF-1000 angular neutron anisotropy and fast-ion diagnostics.",
        },
        {
            "title": "Plasma focus neutron energy and anisotropy measurements using zirconium-beryllium pair activation detectors",
            "authors_year": "Springham et al. (2021)",
            "doi": "10.1016/j.nima.2020.164830",
            "url": "https://doi.org/10.1016/j.nima.2020.164830",
            "why": "Activation-detector method for neutron energy and anisotropy.",
        },
    ],
    "neutron_yield": [
        {
            "title": (
                "Comparison of measured and computed neutron yield from "
                "PF1000 plasma focus device operated with deuterium gas"
            ),
            "authors_year": "Akel, Kubes, Paduch, Lee (2021)",
            "doi": "10.1016/j.radphyschem.2021.109633",
            "url": "https://doi.org/10.1016/j.radphyschem.2021.109633",
            "why": "Shot-resolved PF-1000 scalar neutron yields and fitted current parameters.",
        },
        {
            "title": "Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments",
            "authors_year": "Klir et al. (2011)",
            "doi": "10.1063/1.3559548",
            "url": "https://doi.org/10.1063/1.3559548",
            "why": "Detector timing and sensitivity calibration needed for predictive yield closure.",
        },
    ],
    "neutron_detector_response": [
        {
            "title": "Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments",
            "authors_year": "Klir et al. (2011)",
            "doi": "10.1063/1.3559548",
            "url": "https://doi.org/10.1063/1.3559548",
            "why": "PF-1000-relevant TOF detector calibration and response.",
        },
        {
            "title": "A new concept of fusion neutron monitoring for PF-1000 device",
            "authors_year": "Jednorog et al. (2017)",
            "doi": "10.1515/nuka-2017-0003",
            "url": "https://doi.org/10.1515/nuka-2017-0003",
            "why": "PF-1000 activation monitoring concept and diagnostic response.",
        },
    ],
    "uncertainty": [
        {
            "title": (
                "Comparison of density profiles measured via laser "
                "interferometry with MHD simulations during shock wave "
                "reflection on mega-ampere dense plasma focus"
            ),
            "authors_year": "Malir et al. (2024)",
            "doi": "10.1063/5.0193268",
            "url": "https://doi.org/10.1063/5.0193268",
            "why": "Published density-profile uncertainty and setup limitations.",
        },
        {
            "title": "Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments",
            "authors_year": "Klir et al. (2011)",
            "doi": "10.1063/1.3559548",
            "url": "https://doi.org/10.1063/1.3559548",
            "why": "Detector timing and sensitivity calibration uncertainty.",
        },
    ],
}


_LOCAL_SOURCE_STATUS_BY_DOI = {
    "10.1016/j.radphyschem.2021.109633": {
        "local_status": "parity_verified_knowledge_reference",
        "local_kr_source": (
            "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
        ),
        "local_pdf_sha256": (
            "9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b"
        ),
        "remaining_local_action": (
            "Akel Tables 1 and 2 are typed; Figs. 1-6 require verified "
            "digitization before plot arrays can be used."
        ),
    },
    "10.1088/0022-3727/40/7/021": {
        "local_status": "parity_verified_knowledge_reference",
        "local_kr_source": "KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md",
        "local_pdf_sha256": (
            "7acfb46d1db6ee5894978f70e1372edda7efaa5171d8e7c3bdf0baf7025eff43"
        ),
        "remaining_local_action": (
            "Extract or digitize any same-scope waveform, phase, and spatial "
            "observables not already represented by coded KR targets."
        ),
    },
    "10.1088/0022-3727/40/12/008": {
        "local_status": "parity_verified_knowledge_reference",
        "local_kr_source": "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md",
        "local_pdf_sha256": (
            "c4d62f5015bc6040aa85070e43f3cb6e7e4a8329e5d2baf33fa4d38f828caa4f"
        ),
        "remaining_local_action": (
            "Extract or digitize any same-scope fast-ion, neutron, and "
            "uncertainty observables not already represented by coded targets."
        ),
    },
    "10.1063/5.0089121": {
        "local_status": "parity_verified_knowledge_reference",
        "local_kr_source": "KnowledgeReference/goyon-2022-mjolnir-high-low.md",
        "local_pdf_sha256": (
            "89877f5c880dcd9c4454925984398cf51984f95d2ff78ac4437f5f755e98fe6a"
        ),
        "remaining_local_action": (
            "Use only coded target records or verified digitized observables for "
            "MJOLNIR comparison."
        ),
    },
    "10.1063/5.0193268": {
        "local_status": "parity_verified_knowledge_reference",
        "local_kr_source": "KnowledgeReference/malir-2024-interferometry-dpf.md",
        "local_pdf_sha256": (
            "fafc32261c9172702b1c8dfdc92bcc33b1a32aeeb4cb9680d535478191db46c9"
        ),
        "remaining_local_action": (
            "Use coded density-profile targets and uncertainty records; digitize "
            "figures only if additional profile arrays are needed."
        ),
    },
    "10.1063/5.0253547": {
        "local_status": "parity_verified_knowledge_reference",
        "local_kr_source": (
            "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
            "dense-plasma-focus-z-pinch.md"
        ),
        "local_pdf_sha256": (
            "9c0bc58d72ced9c914914aabdab63937a2b9c7820950eb0fa2412be9fd9d0f8c"
        ),
        "remaining_local_action": (
            "Use coded MJOLNIR neutron timing, temperature, and detector-response "
            "targets; digitize only for additional arrays."
        ),
    },
    "10.1515/nuka-2015-0065": {
        "local_status": "source_fidelity_reviewed_target_extraction_needed",
        "local_kr_source": "KnowledgeReference/cikhardtova-plazma-indd-9dfed6c0.md",
        "local_pdf_sha256": (
            "9dfed6c03000668c5a4926b539b8dc50824fbf8785a069d8a88b319e915fc7f9"
        ),
        "remaining_local_action": (
            "Clean bibliographic title metadata, then extract linear-density "
            "timing, spatial-density observables, and uncertainty."
        ),
    },
    "10.1016/j.vacuum.2004.07.040": {
        "local_status": "source_fidelity_reviewed_target_extraction_needed",
        "local_kr_source": (
            "KnowledgeReference/doi-10-1016-j-vacuum-2004-07-040-6de67a98.md"
        ),
        "local_pdf_sha256": (
            "6de67a98c1c059193e8e3d8bc56288a2c85f7956d662c83d98e64d0c0a06fe7d"
        ),
        "remaining_local_action": (
            "Clean bibliographic title metadata, then extract neutron spectra, "
            "anisotropy, silver-activation geometry, CR-39 layout, and "
            "uncertainty."
        ),
    },
    "10.1109/TPS.2020.3012104": {
        "local_status": "source_fidelity_reviewed_target_extraction_needed",
        "local_kr_source": (
            "KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-"
            "spectrum-from-a-dense-plasma-focus-b78f1154.md"
        ),
        "local_pdf_sha256": (
            "b78f115458d7d25960d9f6596c7af0449c6809b69516372f54a61f1006c86a47"
        ),
        "remaining_local_action": (
            "Extract reconstruction constraints, detector-pair geometry, "
            "scatter subtraction, time/energy resolution, and uncertainty."
        ),
    },
    "10.1016/j.nima.2020.164830": {
        "local_status": "source_fidelity_reviewed_target_extraction_needed",
        "local_kr_source": (
            "KnowledgeReference/nuclear-inst-and-methods-in-physics-research-a-988-"
            "2021-164830-bc8edab3.md"
        ),
        "local_pdf_sha256": (
            "bc8edab30c159ab76609cce7e1505a0d615f99108cd959f0e06bb4ce29dcc33f"
        ),
        "remaining_local_action": (
            "Clean bibliographic title metadata, then extract Zr/Be activation "
            "geometry, MCNP ratio relationship, energy/fluence anisotropy, and "
            "uncertainty."
        ),
    },
    "10.1063/1.3559548": {
        "local_status": "source_fidelity_reviewed_target_extraction_needed",
        "local_kr_source": (
            "KnowledgeReference/fusion-neutron-detector-for-time-of-flight-"
            "measurements-in-z-pinch-and-plasma-focus-214fbdae.md"
        ),
        "local_pdf_sha256": (
            "214fbdae9607094628e9cfcf55157b9d59ad72ab9de3d04a3011cb63b1972747"
        ),
        "remaining_local_action": (
            "Extract detector timing, pulse-height method, neutron sensitivity, "
            "calibration terms, and uncertainty before yield/detector closure."
        ),
    },
}


_NOT_FOUND_AS_EXACT_LOCAL_PDF = {
    "10.1515/nuka-2017-0003",
}


def _is_local_kr_source(candidate: Mapping[str, object]) -> bool:
    return candidate.get("local_status") in {
        "parity_verified_knowledge_reference",
        "text_parity_extracted_review_needed",
        "source_fidelity_reviewed_target_extraction_needed",
    }


def _annotated_candidate_sources(group: str) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for candidate in _CANDIDATE_SOURCES_BY_GROUP.get(group, []):
        item = dict(candidate)
        doi = str(item.get("doi", ""))
        local_status = _LOCAL_SOURCE_STATUS_BY_DOI.get(doi)
        if local_status:
            item.update(local_status)
        elif doi in _NOT_FOUND_AS_EXACT_LOCAL_PDF:
            item["local_status"] = "not_found_as_exact_local_pdf"
            item["remaining_local_action"] = (
                "User acquisition is still required before this source can "
                "enter KnowledgeReference review."
            )
        else:
            item["local_status"] = "not_checked"
            item["remaining_local_action"] = (
                "Local PDF audit has not assigned a definitive status."
            )
        annotated.append(item)
    return annotated


def _same_scope_group_statuses(widest: Mapping[str, object]) -> list[dict[str, object]]:
    present_groups = {
        str(group)
        for group in widest.get("present_groups", [])
        if str(group)
    }
    missing_groups = {
        str(group)
        for group in widest.get("missing_groups", [])
        if str(group)
    }
    partial_groups = {
        str(group)
        for group in widest.get("partial_groups", [])
        if str(group)
    }
    all_groups = sorted(
        set(_PHYSICS_NEED_BY_GROUP)
        | present_groups
        | missing_groups
        | partial_groups
    )

    statuses: list[dict[str, object]] = []
    for group in all_groups:
        if group in partial_groups:
            status = "partial_in_current_scope"
        elif group in missing_groups:
            status = "missing_from_current_scope"
        elif group in present_groups:
            status = "complete_in_current_scope"
        else:
            status = "not_reported_for_current_scope"
        statuses.append({
            "group": group,
            "physics_need": _PHYSICS_NEED_BY_GROUP.get(group, group),
            "status": status,
            "blocks_validation_tiers": list(
                _BLOCKED_VALIDATION_TIERS_BY_GROUP.get(group, [])
            ),
        })
    return statuses


def _source_action(local_sources: list[dict[str, object]]) -> str:
    if local_sources:
        return "local_digitization_or_target_extraction"
    return "user_acquisition_then_knowledge_reference_ingestion"


def scientific_closure_source_acquisition_queue() -> dict[str, object]:
    """Return the current source-acquisition queue from KR closure blockers."""
    report = kr_validation_same_scope_target_report()
    widest = report.get("widest_available_scope", {})
    if not isinstance(widest, Mapping):
        widest = {}
    blockers = widest.get("closure_blockers", {})
    if not isinstance(blockers, Mapping):
        blockers = {}

    items: list[dict[str, object]] = []
    for group, blocker in sorted(blockers.items()):
        if not isinstance(blocker, Mapping):
            continue
        required_data = [
            str(item)
            for item in blocker.get("required_data_to_complete", [])
            if str(item)
        ]
        sources = [
            str(source)
            for source in blocker.get("sources", [])
            if str(source)
        ]
        candidate_sources = _annotated_candidate_sources(str(group))
        local_sources = [
            candidate
            for candidate in candidate_sources
            if _is_local_kr_source(candidate)
        ]
        acquisition_sources = [
            candidate
            for candidate in candidate_sources
            if not _is_local_kr_source(candidate)
        ]
        items.append({
            "group": str(group),
            "physics_need": _PHYSICS_NEED_BY_GROUP.get(str(group), str(group)),
            "priority": 1 if group in {
                "circuit_waveform",
                "phase_timing",
                "neutron_yield",
                "spatial_temperature",
                "uncertainty",
            } else 2,
            "data_availability": blocker.get("data_availability", "unknown"),
            "current_kr_sources": sources,
            "required_data_to_complete": required_data,
            "candidate_sources": candidate_sources,
            "local_sources_available": local_sources,
            "candidate_sources_for_acquisition": acquisition_sources,
            "source_action": _source_action(local_sources),
            "blocks_validation_tiers": list(
                _BLOCKED_VALIDATION_TIERS_BY_GROUP.get(str(group), [])
            ),
            "status": (
                "awaiting_local_digitization_or_target_extraction"
                if local_sources
                else "awaiting_source_research_or_acquisition"
            ),
            "done_condition": (
                "User-acquired KR source plus verified digitized/table data "
                "closes this group for the selected validation scope."
            ),
        })

    items.sort(key=lambda item: (int(item["priority"]), str(item["group"])))
    same_scope_group_statuses = _same_scope_group_statuses(widest)
    summary = {
        "blocker_count": len(items),
        "priority_1_count": sum(1 for item in items if item["priority"] == 1),
        "priority_2_count": sum(1 for item in items if item["priority"] == 2),
        "local_digitization_or_target_extraction_count": sum(
            1
            for item in items
            if item["source_action"] == "local_digitization_or_target_extraction"
        ),
        "user_acquisition_required_count": sum(
            1
            for item in items
            if item["candidate_sources_for_acquisition"]
        ),
        "complete_group_count": sum(
            1
            for item in same_scope_group_statuses
            if item["status"] == "complete_in_current_scope"
        ),
        "partial_group_count": sum(
            1
            for item in same_scope_group_statuses
            if item["status"] == "partial_in_current_scope"
        ),
        "missing_group_count": sum(
            1
            for item in same_scope_group_statuses
            if item["status"] == "missing_from_current_scope"
        ),
    }
    return {
        "passed": False,
        "model_role": "scientific_closure_source_acquisition_queue",
        "validation_scope": widest.get("validation_scope", ""),
        "device": widest.get("device", ""),
        "source_of_truth_rule": (
            "Candidate links are not source-of-truth. They become usable only "
            "after user acquisition and local KnowledgeReference ingestion."
        ),
        "acquisition_process": list(_ACQUISITION_PROCESS),
        "summary": summary,
        "same_scope_group_statuses": same_scope_group_statuses,
        "items": items,
    }
