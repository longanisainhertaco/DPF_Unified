"""Create target-extraction triage for the 2026-05-12 promoted PDF batch.

This is a planning/backlog report. It does not extract or accept validation
targets; it only ranks promoted local sources for future source-line review,
figure/table digitization, and method-reference mapping.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
PROMOTION_JSON = DOCS_DIR / "USER_PDF_KR_PROMOTION_2026_05_12.json"
FIDELITY_JSON = DOCS_DIR / "USER_PDF_KR_SOURCE_FIDELITY_AUDIT_2026_05_12.json"
REPORT_JSON = DOCS_DIR / "USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.json"
REPORT_MD = DOCS_DIR / "USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.md"


P1_TARGET_FILES = {
    "10.1088@1742-6596@370@1@012059.pdf": "dense plasma focus expansion phase and Langmuir probe observables",
    "kasperczuk2002.pdf": "PF-1000 final plasma-column evolution",
    "kubes2020.pdf": "PF closed currents and external magnetic fields",
    "trunk1975.pdf": "already-local DPF/PF source requiring title cleanup and source-line review",
}

P2_TARGET_FILES = {
    "lindemuth1982.pdf": "shock dynamics and neutron-production context outside PF-1000 scope",
}

REVIEW_CONTEXT_FILES = {
    "auluck2014.pdf": "DPF review/source-map context; use primary sources for validation targets",
    "sadowski2008.pdf": "invited DPF/DMP review and source-map context; use primary sources for validation targets",
}

METHOD_FILES = {
    "A-Hybrid-Quantum-Classical-Particle-in-Cell-Method-for-Plasma-Simulations.pdf": "PIC/quantum-classical method review",
    "alexiou2002.pdf": "dense-plasma hydrogen line-broadening and spectroscopy method reference",
    "baxevanis2018.pdf": "axisymmetric plasma simulation technique review",
    "bilbao2006.pdf": "finite-volume ALE plasma simulation method review",
    "chen2019.pdf": "electromagnetic particle simulation method review",
    "Hybrid-Modeling-and-Simulation.pdf": "hybrid modeling textbook/method reference",
    "Linear-Algebra-for-Physics.pdf": "linear algebra method reference",
    "Mathematical-Modeling-and-Simulation-of-Systems.pdf": "modeling/simulation method reference",
    "Mathematics-for-Engineers-and-Scientists.pdf": "math method reference",
    "matsumoto2007.pdf": "relativistic electron-fluid simulation method review",
    "Monte-Carlo-Simulation-of-Neutral-Particle-Transport.pdf": "neutral transport method reference",
    "Numerical-simulation-of-equilibrium-air-plasma-flow-in-the-induction-chamber-of-a-high-power.pdf": "plasma-flow numerical method reference",
    "Numerical-Simulation-of-Pulsed-Plasma-Thruster.pdf": "pulsed plasma thruster simulation method reference",
    "oh2014.pdf": "high-density plasma fluid algorithm comparison",
    "Precalculus.pdf": "math support reference",
    "Simulation-and-Modeling.pdf": "general simulation and modeling method reference",
    "The-role-of-Pauli-principle-in-simulations-of-classical-plasma.pdf": "classical plasma simulation method review",
    "timofeev2011.pdf": "electron-beam plasma-heating simulation method/context",
    "urano2018.pdf": "full-particle electromagnetic-wave simulation method review",
    "verboncoeur2005.pdf": "particle plasma simulation review",
}

MATERIALS_CONTEXT_FILES = {
    "Numerical-simulation-of-deuterium-retention-in-tungsten-under-ELM-like-conditions.pdf": "materials/plasma-wall interaction context, not DPF same-scope validation",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fidelity_by_source(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["source"]): record for record in payload.get("records", [])}


def priority_for(path: str) -> tuple[str, str, str]:
    if path in P1_TARGET_FILES:
        return "P1", "target_extraction_candidate", P1_TARGET_FILES[path]
    if path in P2_TARGET_FILES:
        return "P2", "target_extraction_candidate", P2_TARGET_FILES[path]
    if path in REVIEW_CONTEXT_FILES:
        return "P3", "review_context_only", REVIEW_CONTEXT_FILES[path]
    if path in MATERIALS_CONTEXT_FILES:
        return "P3", "materials_context", MATERIALS_CONTEXT_FILES[path]
    if path in METHOD_FILES:
        return "method", "method_reference_mapping", METHOD_FILES[path]
    return "review", "manual_classification_needed", "review source text before assigning validation role"


def cleanup_needed(title: str) -> bool:
    weak_fragments = (
        "pdf",
        "author",
        "this content has been downloaded",
        "full terms",
        "plasma physics and controlled fusion",
        "plasma and fusion research",
        "research article",
        "continuum mech",
        "yuanzheng zhao",
        "nikolaos",
        "vinh phu",
        "mehdi",
    )
    normalized = title.lower()
    return any(fragment in normalized for fragment in weak_fragments)


def make_entry(item: dict[str, Any], fidelity: dict[str, Any]) -> dict[str, Any]:
    source = str(item["path"])
    priority, role, rationale = priority_for(source)
    counts = dict(fidelity.get("counts", {}))
    next_actions: list[str] = []
    if cleanup_needed(str(item.get("title", ""))):
        next_actions.append("bibliographic_title_cleanup")
    if role == "target_extraction_candidate":
        next_actions.extend(["source_line_review", "typed_target_extraction"])
        if counts.get("figure_captions", 0) or counts.get("image_blocks", 0):
            next_actions.append("figure_digitization_triage")
        if counts.get("table_captions", 0) or counts.get("extracted_tables", 0):
            next_actions.append("table_extraction_triage")
    elif role == "method_reference_mapping":
        next_actions.append("map_to_method_or_numerical_verification_claims")
    elif role == "review_context_only":
        next_actions.append("bibliography_map_only_use_primary_sources_for_targets")
    elif role == "materials_context":
        next_actions.append("scope_for_wall_material_or_plasma_wall_context_only")
    else:
        next_actions.append("manual_source_review")
    return {
        "source": source,
        "title": item.get("title", ""),
        "priority": priority,
        "role": role,
        "rationale": rationale,
        "pages": item.get("pages", 0),
        "doi": item.get("doi", ""),
        "sha256": item.get("sha256", ""),
        "knowledge_reference_markdown": item.get("markdown", ""),
        "knowledge_reference_json": item.get("json", ""),
        "source_fidelity_status": fidelity.get("status", "not_checked"),
        "source_fidelity_counts": counts,
        "next_actions": next_actions,
        "accepted_for_validation": False,
        "guardrail": "triage_only_not_validation_evidence",
    }


def build_report() -> dict[str, Any]:
    promotion = load_json(PROMOTION_JSON)
    fidelity = fidelity_by_source(load_json(FIDELITY_JSON))
    entries = [
        make_entry(item, fidelity.get(str(item["path"]), {}))
        for item in promotion.get("promoted", [])
    ]
    for item in promotion.get("skipped_existing", []):
        source = str(item["path"])
        entry = make_entry(
            {
                **item,
                "pages": None,
                "markdown": fidelity.get(source, {}).get("markdown", ""),
                "json": fidelity.get(source, {}).get("json", ""),
            },
            fidelity.get(source, {}),
        )
        entry["source_status"] = "selected_but_existing_kr_record_reused"
        entries.append(entry)
    priority_counts = Counter(entry["priority"] for entry in entries)
    role_counts = Counter(entry["role"] for entry in entries)
    target_entries = [entry for entry in entries if entry["role"] == "target_extraction_candidate"]
    return {
        "date": date.today().isoformat(),
        "source_reports": {
            "promotion": PROMOTION_JSON.relative_to(ROOT).as_posix(),
            "source_fidelity": FIDELITY_JSON.relative_to(ROOT).as_posix(),
        },
        "guardrail": (
            "This triage ranks newly local source records for future review. It "
            "does not accept any target, table, plotted curve, formula, or "
            "validation threshold."
        ),
        "entry_count": len(entries),
        "priority_counts": dict(sorted(priority_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "target_extraction_candidate_count": len(target_entries),
        "entries": sorted(entries, key=lambda e: (str(e["priority"]), str(e["source"]).lower())),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# May 12 User PDF Target Triage",
        "",
        f"Generated: {report['date']}",
        "",
        report["guardrail"],
        "",
        "## Summary",
        "",
        f"- Entries triaged: {report['entry_count']}",
        f"- Target-extraction candidates: {report['target_extraction_candidate_count']}",
        "",
        "### Priority Counts",
        "",
        "| priority | count |",
        "| --- | ---: |",
    ]
    for key, value in report["priority_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "### Role Counts",
            "",
            "| role | count |",
            "| --- | ---: |",
        ]
    )
    for key, value in report["role_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Triage Entries",
            "",
            "| priority | role | source | figures | tables | targets | uncertainties | next actions | KR json |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for entry in report["entries"]:
        counts = entry["source_fidelity_counts"]
        table_total = int(counts.get("table_captions", 0)) + int(counts.get("extracted_tables", 0))
        lines.append(
            "| {priority} | `{role}` | `{source}` | {figures} | {tables} | {targets} | {uncertainties} | {actions} | `{json}` |".format(
                priority=entry["priority"],
                role=entry["role"],
                source=entry["source"],
                figures=counts.get("figure_captions", 0),
                tables=table_total,
                targets=counts.get("numeric_target_contexts", 0),
                uncertainties=counts.get("uncertainty_contexts", 0),
                actions=", ".join(f"`{action}`" for action in entry["next_actions"]),
                json=entry["knowledge_reference_json"],
            )
        )
    lines.extend(
        [
            "",
            "## Target-Extraction Guardrail",
            "",
            "Every target candidate above still needs source-line review, typed target extraction, unit normalization, and uncertainty handling before any code threshold can cite it. Figure/table data still needs a digitization packet and independent accepted review before validation use.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    report = build_report()
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    write_markdown(report)
    print(
        "entries={entry_count} target_candidates={target_extraction_candidate_count}".format(
            **report
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
