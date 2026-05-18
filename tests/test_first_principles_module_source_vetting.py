from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_first_principles_module_source_vetting.py"


def _load_audit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("module_source_vetting", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_by_path(payload: dict[str, object], path: str) -> dict[str, object]:
    modules = payload["modules"]
    assert isinstance(modules, list)
    for item in modules:
        assert isinstance(item, dict)
        if item["path"] == path:
            return item
    raise AssertionError(f"module not found in vetting payload: {path}")


def test_module_vetting_keeps_active_first_principles_source_closed() -> None:
    audit = _load_audit_module()
    payload = audit.build_module_vetting_report(date_slug="test")
    summary = payload["summary"]

    assert summary["total_modules"] >= 250
    assert summary["active_physics_unvetted_count"] == 0
    assert summary["missing_source_reference_count"] == 0
    assert summary["inactive_physics_unvetted_count"] == 0
    assert payload["strict_passed"] is True


def test_module_vetting_classifies_active_and_legacy_surfaces() -> None:
    audit = _load_audit_module()
    payload = audit.build_module_vetting_report(date_slug="test")

    ionization = _module_by_path(payload, "src/dpf/atomic/ionization.py")
    assert ionization["active_first_principles_closure"] is True
    assert ionization["classification"] == "active_physics_source_referenced"
    assert (
        "KnowledgeReference/2019nrlplasma-formulary-037290d4.md"
        in ionization["knowledge_reference_paths"]
    )

    comparator = _module_by_path(
        payload,
        "src/dpf/first_principles/current_waveform_comparator.py",
    )
    assert comparator["classification"] == "active_physics_source_referenced"
    assert (
        "docs/FIRST_PRINCIPLES_GV_SHOT_INFO_TRIAGE_2026_05_16.md"
        in comparator["docs_reference_paths"]
    )

    export_surface = _module_by_path(payload, "src/dpf/fields/__init__.py")
    assert export_surface["classification"] == "package_export_surface"
    assert export_surface["status"] == "module_vetted_nonpromoting"

    legacy_circuit = _module_by_path(payload, "src/dpf/circuit/rlc_solver.py")
    assert (
        legacy_circuit["classification"]
        == "inactive_legacy_physics_not_first_principles_authority"
    )
    assert legacy_circuit["status"] == "module_vetted_nonpromoting"

    beam_target = _module_by_path(payload, "src/dpf/diagnostics/beam_target.py")
    assert (
        "KnowledgeReference/bosch-hale-1992-fusion-reactivity.md"
        in beam_target["knowledge_reference_paths"]
    )
    assert beam_target["can_support_first_principles_acceptance"] is False
