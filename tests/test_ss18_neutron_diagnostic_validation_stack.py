from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / "docs/SS18_NEUTRON_DIAGNOSTIC_VALIDATION_STACK_2026_05_23.json"
_SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_validator = importlib.import_module("validate_ss18_neutron_diagnostic_packet")
validate_packet = _validator.validate_packet

REQUIRED_MECHANISMS = [
    "yield",
    "timing",
    "spectrum",
    "anisotropy",
    "detector_activation_response",
    "diagnostic_mapping",
    "uncertainty_blockers",
]

REQUIRED_OBSERVABLES = {
    "yield": "max_neutron_yield_new_large_electrodes",
    "timing": "neutron_pulse_timing_vs_xray",
    "spectrum": "tof_spectrum_future_method_blocker",
    "anisotropy": "anisotropy_pressure_trend",
    "detector_activation_response": "activation_counter_calibration_source",
    "diagnostic_mapping": "tof_scintillator_distance_and_angle_mapping",
    "uncertainty_blockers": "missing_same_scope_uncertainty_budget",
}


def test_ss18_packet_has_mechanism_separated_fail_closed_stack() -> None:
    packet = json.loads(PACKET_PATH.read_text())

    assert packet["packet_id"] == "ss18_neutron_diagnostic_validation_stack"
    assert packet["validation_scope"] == "pf1000_neutron_diagnostics_full_energy_27_to_40_kv"
    assert packet["authority_rule"] == "local_line_cited_sources_only_retrieval_is_not_authority"
    assert packet["acceptance_boundary"] == {
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "promotes_acceptance": False,
    }

    mechanisms = packet["mechanisms"]
    assert [row["mechanism"] for row in mechanisms] == REQUIRED_MECHANISMS
    assert not any(row["status"] == "accepted" for row in mechanisms)
    assert all(row["blocked_reason"] for row in mechanisms)
    assert all(row["diagnostic_channel"] for row in mechanisms)
    assert all(row.get("promotes_acceptance") is False for row in mechanisms)

    mechanisms_by_name = {row["mechanism"]: row for row in mechanisms}
    for mechanism, required_observable in REQUIRED_OBSERVABLES.items():
        names = {observable["name"] for observable in mechanisms_by_name[mechanism]["observables"]}
        assert required_observable in names


def test_ss18_source_refs_resolve_to_knowledge_reference_and_quotes_match() -> None:
    packet = json.loads(PACKET_PATH.read_text())

    checked_refs = 0
    for mechanism in packet["mechanisms"]:
        for observable in mechanism.get("observables", []):
            for ref in observable.get("source_refs", []):
                source_path = ROOT / ref["source_path"]
                assert source_path.exists(), ref
                assert source_path.is_relative_to(ROOT / "KnowledgeReference"), ref
                lines = source_path.read_text(errors="ignore").splitlines()
                start = ref["line_start"]
                end = ref["line_end"]
                assert 1 <= start <= end <= len(lines), ref
                assert end - start + 1 <= 24, ref
                extracted = " ".join(line.strip() for line in lines[start - 1 : end])
                assert ref["quote"] == extracted, (ref, extracted)
                checked_refs += 1

    assert checked_refs >= 10


def test_ss18_diagnostic_mapping_and_uncertainty_are_non_promoting() -> None:
    packet = json.loads(PACKET_PATH.read_text())
    mapping = {row["mechanism"]: row for row in packet["mechanisms"]}["diagnostic_mapping"]
    uncertainty = {row["mechanism"]: row for row in packet["mechanisms"]}["uncertainty_blockers"]

    assert mapping["status"] == "candidate"
    assert mapping["promotes_acceptance"] is False
    assert any("15 m" in str(observable.get("value")) for observable in mapping["observables"])
    assert uncertainty["status"] == "blocked"
    assert uncertainty["observables"][0]["value"] is None
    assert uncertainty["observables"][0]["uncertainty"] is None
    assert "uncertainty" in uncertainty["blocked_reason"].lower()


def _write_mutated_packet(tmp_path: Path, mutator: object) -> Path:
    packet = json.loads(PACKET_PATH.read_text())
    mutator(packet)  # type: ignore[operator]
    target = tmp_path / "packet.json"
    target.write_text(json.dumps(packet, indent=2))
    return target


def _rules(path: Path) -> set[str]:
    return {issue["rule"] for issue in validate_packet(path, ROOT)}


def test_validator_accepts_live_ss18_packet() -> None:
    assert validate_packet(PACKET_PATH, ROOT) == []


def test_validator_flags_acceptance_promotion_and_accepted_rows(tmp_path: Path) -> None:
    def mutate(packet: dict[str, Any]) -> None:
        packet["acceptance_boundary"]["promotes_acceptance"] = True
        packet["mechanisms"][0]["status"] = "accepted"
        packet["mechanisms"][0]["promotes_acceptance"] = True

    rules = _rules(_write_mutated_packet(tmp_path, mutate))
    assert "acceptance_flag_not_false" in rules
    assert "accepted_mechanism_forbidden_in_ss18" in rules
    assert "mechanism_promotes_acceptance" in rules


def test_validator_flags_fabricated_quote_and_non_kr_source(tmp_path: Path) -> None:
    def mutate(packet: dict[str, Any]) -> None:
        ref = packet["mechanisms"][0]["observables"][0]["source_refs"][0]
        ref["source_path"] = "docs/DPF_UNIFIED_FULL_PROJECT_KANBAN_PLAN_2026_05_22.md"
        ref["line_start"] = 149
        ref["line_end"] = 152
        ref["quote"] = "fabricated"

    rules = _rules(_write_mutated_packet(tmp_path, mutate))
    assert "source_ref_not_knowledge_reference" in rules


def test_validator_requires_diagnostic_completeness(tmp_path: Path) -> None:
    def mutate(packet: dict[str, Any]) -> None:
        packet["mechanisms"] = packet["mechanisms"][:-1]

    assert "missing_required_mechanism" in _rules(_write_mutated_packet(tmp_path, mutate))


def test_validator_flags_diagnostic_completeness_acceptance_shortcuts(tmp_path: Path) -> None:
    def mutate(packet: dict[str, Any]) -> None:
        packet["diagnostic_completeness_check"]["complete_for_acceptance"] = True
        packet["diagnostic_completeness_check"]["blocking_reasons"] = []
        packet["diagnostic_completeness_check"]["required_mechanisms"] = ["yield"]

    rules = _rules(_write_mutated_packet(tmp_path, mutate))
    assert "diagnostic_completeness_promotes_acceptance" in rules
    assert "diagnostic_completeness_blockers_missing" in rules
    assert "diagnostic_completeness_required_mechanisms_mismatch" in rules
