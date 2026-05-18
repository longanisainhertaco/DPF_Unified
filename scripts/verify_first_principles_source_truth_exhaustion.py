#!/usr/bin/env python3
"""Audit whether the first-principles source truth index is exhausted.

This is a read-only gate for the first-principles development track. It does
not invoke the validation workflow and does not promote any source, target, or
simulation artifact. Its job is to prove whether every local source-truth file
and user-verified ledger entry is represented in the current index/triage
surface, and to fail closed when the index has drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INDEX_PATH = ROOT / "docs" / "FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json"
FINISH_LINE_PLAN = ROOT / "docs" / "FIRST_PRINCIPLES_FINISH_LINE_PLAN.md"
SOURCE_QUEUE = ROOT / "docs" / "SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md"
LEDGER_PATHS = (
    ROOT / "docs" / "USER_VALIDATED_PDF_KR_PROMOTION_2026_05_15.json",
    ROOT / "docs" / "USER_VALIDATED_THESES_KR_PROMOTION_2026_05_16.json",
)
TRIAGE_PATHS = (
    ROOT / "docs" / "FIRST_PRINCIPLES_USER_PDF_BATCH_TRIAGE_2026_05_15.json",
    ROOT / "docs" / "FIRST_PRINCIPLES_MAY16_VALIDATED_THESES_TRIAGE_2026_05_16.json",
    ROOT / "docs" / "FIRST_PRINCIPLES_GV_SHOT_INFO_TRIAGE_2026_05_16.md",
)
EXPECTED_SOURCE_SEARCH_DOCS = (
    "FIRST_PRINCIPLES_BLOCKER_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_CERTIFICATE_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_CLOSURE_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_COMPARATOR_UQ_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_DIMENSIONALITY_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_GENERALIZATION_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_LIMITER_READINESS_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_NEUTRON_AUTHORITY_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_PF1000_PACKAGE_DECK_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_POWER_PORT_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_SAME_SCOPE_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_SPATIAL_FIELD_TEMPERATURE_SOURCE_SEARCH_2026_05_15.md",
    "FIRST_PRINCIPLES_WAVEFORM_PHASE_SOURCE_SEARCH_2026_05_15.md",
)
PROMOTING_ACCEPTANCE_KEYS = {
    "accepted_for_whole_shot_first_principles",
    "accepted_for_first_principles_validation",
    "accepted_for_validation",
    "can_support_first_principles_acceptance",
}
CAPABILITY_LABELS = {
    "dpf_device": "DPF/device scope",
    "maxwell_fields": "Maxwell/vacuum EM fields",
    "mhd": "MHD/fluid dynamics",
    "hybrid_pic_fluid": "Hybrid PIC-fluid / kinetic ions",
    "generalized_ohm": "Generalized Ohm/current closure",
    "circuit_coupling": "Circuit coupling",
    "poynting_power": "Poynting/J.E power port",
    "startup_breakdown": "Startup/breakdown/liftoff",
    "electrode_boundary": "Electrode/conductor/boundaries",
    "ionization_eos": "Ionization/EOS/transport coefficients",
    "two_temperature": "Separate electron/ion temperature",
    "radiation": "Radiation/spectroscopy",
    "collisions": "Collisions/stopping",
    "neutron_thermonuclear": "Thermonuclear neutron production",
    "neutron_beam_target": "Beam-target neutron production",
    "detector_response": "Detector/diagnostic response",
    "uncertainty_uq": "Uncertainty/UQ",
    "three_d_geometry": "True 3D geometry/instabilities",
    "validation_target": "Experimental validation target",
    "numerical_method": "Numerical method/verification",
}
CAPABILITY_KEYWORDS = {
    "dpf_device": ("dense plasma focus", "plasma focus", "dpf", "mather"),
    "maxwell_fields": ("maxwell", "electromagnetic", "electric field", "magnetic field"),
    "mhd": ("magnetohydrodynamic", "mhd", "fluid", "hydrodynamic"),
    "hybrid_pic_fluid": ("hybrid pic", "pic-fluid", "particle-in-cell", "kinetic ion"),
    "generalized_ohm": ("generalized ohm", "ohm's law", "hall term", "conductivity"),
    "circuit_coupling": ("circuit", "capacitor", "inductance", "current waveform"),
    "poynting_power": ("poynting", "j.e", "j dot e", "power port", "terminal voltage"),
    "startup_breakdown": ("breakdown", "flashover", "preionization", "liftoff", "sheath"),
    "electrode_boundary": ("electrode", "anode", "cathode", "insulator", "boundary"),
    "ionization_eos": ("ionization", "equation of state", "eos", "resistivity"),
    "two_temperature": ("electron temperature", "ion temperature", "two-temperature"),
    "radiation": ("radiation", "x-ray", "x ray", "spectroscopy", "bremsstrahlung"),
    "collisions": ("collision", "stopping", "spitzer", "braginskii"),
    "neutron_thermonuclear": ("thermonuclear", "fusion reactivity", "dd fusion"),
    "neutron_beam_target": ("beam-target", "beam target", "ion beam", "fast ion"),
    "detector_response": ("detector", "activation", "tof", "scintillator", "diagnostic"),
    "uncertainty_uq": ("uncertainty", "error bar", "uq", "bayesian", "confidence"),
    "three_d_geometry": ("3d", "three-dimensional", "instability", "asymmetry"),
    "validation_target": ("experiment", "measured", "validation", "shot", "waveform"),
    "numerical_method": ("simulation", "numerical", "convergence", "mesh", "algorithm"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=datetime.now(UTC).strftime("%Y_%m_%d"),
        help="Date slug for generated docs, default UTC today.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when the exhaustion verdict is false.",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Regenerate the source-truth index before running the exhaustion audit.",
    )
    args = parser.parse_args()

    if args.refresh_index:
        refreshed = refresh_source_truth_index()
        INDEX_PATH.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n")
        (ROOT / "docs" / "FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md").write_text(
            _source_truth_index_markdown(refreshed)
        )

    payload = build_exhaustion_report(date_slug=args.date)
    output_json = (
        ROOT / "docs" / f"FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_{args.date}.json"
    )
    output_md = (
        ROOT / "docs" / f"FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_{args.date}.md"
    )
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown_report(payload))
    print(json.dumps({
        "exhausted": payload["exhausted"],
        "open_issue_count": payload["open_issue_count"],
        "json": str(output_json.relative_to(ROOT)),
        "markdown": str(output_md.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    if args.strict and not payload["exhausted"]:
        return 1
    return 0


def build_exhaustion_report(*, date_slug: str) -> dict[str, Any]:
    index = _read_json(INDEX_PATH)
    inventory = _inventory_status(index)
    ledgers = _ledger_status(index)
    source_targets = _source_target_status()
    gate_docs = _gate_doc_status()
    triage_docs = _triage_doc_status()

    issue_groups = {
        "unindexed_source_files": inventory["unindexed_files"],
        "missing_indexed_files": inventory["missing_indexed_files"],
        "changed_indexed_files": inventory["changed_indexed_files"],
        "ledger_without_kr_record": ledgers["ledger_without_kr_record"],
        "ledger_missing_files": ledgers["ledger_missing_files"],
        "ledger_parity_failures": ledgers["ledger_parity_failures"],
        "promoting_source_target_packets": source_targets[
            "promoting_source_target_packets"
        ],
        "missing_gate_source_search_docs": gate_docs["missing_docs"],
        "missing_triage_docs": triage_docs["missing_docs"],
    }
    open_issue_count = sum(len(items) for items in issue_groups.values())
    exhausted = open_issue_count == 0
    return {
        "date": date_slug,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "first_principles_source_truth_exhaustion",
        "authority_policy": (
            "KnowledgeReference plus explicitly user-verified staged sources only; "
            "this audit is non-promoting and does not invoke validation workflow"
        ),
        "exhausted": exhausted,
        "open_issue_count": open_issue_count,
        "issue_groups": issue_groups,
        "inventory": inventory,
        "ledgers": ledgers,
        "source_targets": source_targets,
        "gate_docs": gate_docs,
        "triage_docs": triage_docs,
        "next_required_actions": _next_required_actions(issue_groups),
    }


def _inventory_status(index: Mapping[str, Any]) -> dict[str, Any]:
    indexed_inventory = {
        str(item["path"]): str(item.get("sha256", ""))
        for item in index.get("file_inventory", ())
        if isinstance(item, Mapping) and item.get("path")
    }
    actual_inventory = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in (ROOT / "KnowledgeReference").rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    }

    indexed_paths = set(indexed_inventory)
    actual_paths = set(actual_inventory)
    changed = []
    for path in sorted(indexed_paths & actual_paths):
        if indexed_inventory[path] and indexed_inventory[path] != actual_inventory[path]:
            changed.append({
                "path": path,
                "indexed_sha256": indexed_inventory[path],
                "actual_sha256": actual_inventory[path],
            })

    return {
        "index_path": str(INDEX_PATH.relative_to(ROOT)),
        "indexed_file_count": len(indexed_paths),
        "actual_file_count": len(actual_paths),
        "unindexed_files": sorted(actual_paths - indexed_paths),
        "missing_indexed_files": sorted(indexed_paths - actual_paths),
        "changed_indexed_files": changed,
        "index_stats": index.get("stats", {}),
    }


def refresh_source_truth_index() -> dict[str, Any]:
    """Build a current source-truth index while preserving reviewed records."""

    existing = _read_json(INDEX_PATH) if INDEX_PATH.exists() else {}
    existing_records = _existing_records_by_path(existing)
    file_inventory = _build_file_inventory()
    records = _build_records(existing_records)
    external_pdf_records = existing.get("external_pdf_records", [])
    stats = _index_stats(records, file_inventory, external_pdf_records)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo": str(ROOT),
        "scope": {
            "scientific_source_scope": (
                "KnowledgeReference/ plus explicitly user-verified staged sources"
            ),
            "first_principles_verdicts_are_nonpromoting": True,
            "validation_workflow_invoked": False,
        },
        "stats": stats,
        "capability_index": _capability_index(records),
        "external_pdf_records": external_pdf_records,
        "file_inventory": file_inventory,
        "records": records,
    }


def _existing_records_by_path(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for record in index.get("records", ()):
        if not isinstance(record, Mapping):
            continue
        copied = dict(record)
        paths = copied.get("paths", {})
        if isinstance(paths, Mapping):
            for key in ("markdown", "json"):
                path = paths.get(key)
                if path:
                    by_path[str(path)] = copied
    return by_path


def _build_file_inventory() -> list[dict[str, Any]]:
    inventory = []
    for path in sorted((ROOT / "KnowledgeReference").rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rel = str(path.relative_to(ROOT))
        inventory.append({
            "path": rel,
            "extension": path.suffix,
            "kind": _file_kind(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    return inventory


def _build_records(
    existing_records: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    kr_root = ROOT / "KnowledgeReference"
    stems = sorted(
        {
            path.stem
            for path in kr_root.iterdir()
            if path.is_file() and path.suffix in {".md", ".json"}
        }
    )
    records: list[dict[str, Any]] = []
    for stem in stems:
        md_path = kr_root / f"{stem}.md"
        json_path = kr_root / f"{stem}.json"
        rel_md = str(md_path.relative_to(ROOT)) if md_path.exists() else None
        rel_json = str(json_path.relative_to(ROOT)) if json_path.exists() else None
        existing = (
            existing_records.get(rel_md or "")
            or existing_records.get(rel_json or "")
            or {}
        )
        metadata = _safe_json(json_path) if json_path.exists() else {}
        text = _read_prefix(md_path, limit=60_000) if md_path.exists() else ""
        record = dict(existing) if existing else _new_record(stem, metadata, text)
        record["record_kind"] = "knowledge_reference_document"
        record["paths"] = {
            "markdown": rel_md,
            "json": rel_json,
        }
        hashes: dict[str, str] = {}
        if md_path.exists():
            hashes["markdown_sha256"] = _sha256(md_path)
        if json_path.exists():
            hashes["json_sha256"] = _sha256(json_path)
        record["hashes"] = hashes
        record["page_count"] = _page_count(metadata, record)
        record["table_count"] = _count_field(metadata, "table_count", record)
        record["figure_count"] = _count_field(metadata, "figure_count", record)
        record.setdefault("ingestion_status", "text_parity_extracted_review_needed")
        record.setdefault("source_fidelity_status", "review_needed")
        record.setdefault("validation_status", "source_available_not_target_extracted")
        records.append(record)
    return records


def _new_record(stem: str, metadata: Mapping[str, Any], text: str) -> dict[str, Any]:
    title = _metadata_title(metadata) or _markdown_title(text) or stem
    tags = _tags_for_text(" ".join([stem, title, text[:60_000]]))
    verdict = _verdict_for_tags(tags)
    return {
        "id": f"kr:{stem}",
        "title": title,
        "authors": _metadata_authors(metadata),
        "year": _metadata_year(metadata),
        "doi": str(metadata.get("doi") or ""),
        "accession": str(metadata.get("accession") or ""),
        "source_pdf": str(metadata.get("source_pdf") or ""),
        "source_pdf_relative_path": str(metadata.get("source_pdf_relative_path") or ""),
        "source_pdf_sha256": str(metadata.get("source_pdf_sha256") or ""),
        "source_type": _source_type_for_tags(tags),
        "tags": tags,
        "first_principles_verdict": verdict,
        "first_principles_use": _first_principles_use(tags, verdict),
    }


def _index_stats(
    records: list[Mapping[str, Any]],
    file_inventory: list[Mapping[str, Any]],
    external_pdf_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    file_kind_counts: dict[str, int] = {}
    for item in file_inventory:
        kind = str(item.get("kind", "unknown"))
        file_kind_counts[kind] = file_kind_counts.get(kind, 0) + 1
    verdict_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for record in records:
        verdict = str(record.get("first_principles_verdict", "unknown"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        for tag in record.get("tags", ()):
            tag = str(tag)
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "knowledge_reference_file_count": len(file_inventory),
        "knowledge_reference_document_count": len(records),
        "markdown_source_count": file_kind_counts.get("source_markdown", 0),
        "json_metadata_count": file_kind_counts.get("source_json_metadata", 0),
        "external_pdf_count": len(external_pdf_records),
        "file_kind_counts": dict(sorted(file_kind_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
    }


def _capability_index(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for tag, label in CAPABILITY_LABELS.items():
        tagged = [record for record in records if tag in record.get("tags", ())]
        direct = [
            record
            for record in tagged
            if record.get("first_principles_verdict") == "directly_needed"
        ]
        representatives = direct[:12] or tagged[:12]
        index[tag] = {
            "label": label,
            "record_count": len(tagged),
            "direct_record_count": len(direct),
            "representative_records": [
                {
                    "id": record.get("id"),
                    "title": record.get("title"),
                    "path": (record.get("paths") or {}).get("markdown"),
                    "verdict": record.get("first_principles_verdict"),
                }
                for record in representatives
            ],
        }
    return index


def _ledger_status(index: Mapping[str, Any]) -> dict[str, Any]:
    indexed_paths = {
        str(item["path"])
        for item in index.get("file_inventory", ())
        if isinstance(item, Mapping) and item.get("path")
    }
    ledger_without_kr_record: list[dict[str, str]] = []
    ledger_missing_files: list[dict[str, str]] = []
    ledger_parity_failures: list[dict[str, Any]] = []
    ledger_summaries: list[dict[str, Any]] = []

    for path in LEDGER_PATHS:
        ledger = _read_json(path)
        ledger_summaries.append({
            "path": str(path.relative_to(ROOT)),
            "files_scanned": ledger.get("files_scanned"),
            "promoted_count": ledger.get("promoted_count"),
            "skipped_existing_count": ledger.get("skipped_existing_count"),
            "failed_count": ledger.get("failed_count"),
        })
        for entry in ledger.get("promoted", ()):
            if not isinstance(entry, Mapping):
                continue
            for key in ("markdown", "json"):
                kr_path = str(entry.get(key, ""))
                if not kr_path:
                    ledger_missing_files.append({
                        "ledger": str(path.relative_to(ROOT)),
                        "entry": str(entry.get("path", "unknown")),
                        "missing_field": key,
                    })
                    continue
                absolute = ROOT / kr_path
                if not absolute.exists():
                    ledger_missing_files.append({
                        "ledger": str(path.relative_to(ROOT)),
                        "entry": str(entry.get("path", "unknown")),
                        "missing_path": kr_path,
                    })
                if kr_path not in indexed_paths:
                    ledger_without_kr_record.append({
                        "ledger": str(path.relative_to(ROOT)),
                        "entry": str(entry.get("path", "unknown")),
                        "kr_path": kr_path,
                    })
            parity = entry.get("parity", {})
            if not isinstance(parity, Mapping) or parity.get("passed") is not True:
                ledger_parity_failures.append({
                    "ledger": str(path.relative_to(ROOT)),
                    "entry": str(entry.get("path", "unknown")),
                    "parity": parity,
                })
        for entry in ledger.get("skipped_existing", ()):
            if not isinstance(entry, Mapping):
                continue
            reason = str(entry.get("reason", ""))
            referenced = _extract_knowledge_reference_path(reason)
            if referenced and referenced not in indexed_paths:
                ledger_without_kr_record.append({
                    "ledger": str(path.relative_to(ROOT)),
                    "entry": str(entry.get("path", "unknown")),
                    "kr_path": referenced,
                })

    return {
        "ledgers": ledger_summaries,
        "ledger_without_kr_record": ledger_without_kr_record,
        "ledger_missing_files": ledger_missing_files,
        "ledger_parity_failures": ledger_parity_failures,
    }


def _source_target_status() -> dict[str, Any]:
    from dpf.first_principles.source_targets import (  # noqa: PLC0415
        gv_verified_shot_targets,
        may15_user_validated_source_targets,
        may16_validated_thesis_source_targets,
    )

    packets = {
        "may15_user_validated_source_targets": may15_user_validated_source_targets(),
        "may16_validated_thesis_source_targets": may16_validated_thesis_source_targets(),
        "gv_verified_shot_targets": gv_verified_shot_targets(),
    }
    promoting = []
    summaries = {}
    for name, packet in packets.items():
        promoting.extend(
            {
                "packet": name,
                "path": path,
                "key": key,
                "value": value,
            }
            for path, key, value in _iter_acceptance_flags(packet)
            if bool(value) is True
        )
        summaries[name] = {
            "accepted_for_whole_shot_first_principles": packet.get(
                "accepted_for_whole_shot_first_principles"
            ),
            "source_status": packet.get("source_status"),
        }

    return {
        "packets": summaries,
        "promoting_source_target_packets": promoting,
    }


def _gate_doc_status() -> dict[str, Any]:
    docs = [ROOT / "docs" / name for name in EXPECTED_SOURCE_SEARCH_DOCS]
    gate_rows = _extract_fp_gate_rows(FINISH_LINE_PLAN.read_text())
    return {
        "finish_line_plan": str(FINISH_LINE_PLAN.relative_to(ROOT)),
        "source_queue": str(SOURCE_QUEUE.relative_to(ROOT)),
        "expected_doc_count": len(docs),
        "present_doc_count": sum(1 for path in docs if path.exists()),
        "missing_docs": [
            str(path.relative_to(ROOT))
            for path in docs
            if not path.exists()
        ],
        "fp_gate_rows": gate_rows,
        "fp_gate_count": len(gate_rows),
    }


def _triage_doc_status() -> dict[str, Any]:
    return {
        "expected_doc_count": len(TRIAGE_PATHS),
        "present_doc_count": sum(1 for path in TRIAGE_PATHS if path.exists()),
        "missing_docs": [
            str(path.relative_to(ROOT))
            for path in TRIAGE_PATHS
            if not path.exists()
        ],
    }


def _iter_acceptance_flags(
    value: Any,
    *,
    path: str = "$",
) -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_str = str(key)
            child_path = f"{path}.{key_str}"
            if key_str in PROMOTING_ACCEPTANCE_KEYS:
                yield child_path, key_str, item
            yield from _iter_acceptance_flags(item, path=child_path)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _iter_acceptance_flags(item, path=f"{path}[{index}]")


def _extract_fp_gate_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("| FP-"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 4:
            continue
        rows.append({
            "gate": columns[0],
            "name": columns[1],
            "status": columns[2],
        })
    return rows


def _next_required_actions(issue_groups: Mapping[str, list[Any]]) -> list[str]:
    actions: list[str] = []
    if issue_groups["unindexed_source_files"] or issue_groups["changed_indexed_files"]:
        actions.append(
            "regenerate docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json and .md "
            "after the latest KnowledgeReference promotions"
        )
    if issue_groups["ledger_without_kr_record"]:
        actions.append(
            "reconcile promoted ledger entries against the regenerated source-truth index"
        )
    if issue_groups["promoting_source_target_packets"]:
        actions.append(
            "repair source_targets helpers so every packet remains non-promoting"
        )
    if issue_groups["missing_gate_source_search_docs"]:
        actions.append("restore or regenerate missing first-principles source-search docs")
    if issue_groups["missing_triage_docs"]:
        actions.append("restore or regenerate missing user-verified triage docs")
    if not actions:
        actions.append(
            "source-truth exhaustion gate is clean; proceed to the next physics-code blocker"
        )
    return actions


def _markdown_report(payload: Mapping[str, Any]) -> str:
    issue_groups = payload["issue_groups"]
    lines = [
        "# First-Principles Source Truth Exhaustion",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"Exhausted: `{str(payload['exhausted']).lower()}`",
        f"Open issue count: `{payload['open_issue_count']}`",
        "",
        "## Policy",
        "",
        str(payload["authority_policy"]),
        "",
        "## Issue Groups",
        "",
    ]
    for name, items in issue_groups.items():
        lines.append(f"- `{name}`: {len(items)}")
    lines.extend(["", "## Inventory", ""])
    inventory = payload["inventory"]
    lines.append(f"- Indexed files: {inventory['indexed_file_count']}")
    lines.append(f"- Actual files: {inventory['actual_file_count']}")
    lines.append(f"- Unindexed files: {len(inventory['unindexed_files'])}")
    lines.append(f"- Changed indexed files: {len(inventory['changed_indexed_files'])}")
    lines.extend(["", "## Next Required Actions", ""])
    for action in payload["next_required_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def _source_truth_index_markdown(index: Mapping[str, Any]) -> str:
    stats = index["stats"]
    capability_index = index["capability_index"]
    records = index["records"]
    direct = [
        record
        for record in records
        if record.get("first_principles_verdict") == "directly_needed"
    ][:15]
    lines = [
        "# First-Principles Source Truth Index",
        "",
        f"Generated: {index['generated_at_utc']}",
        "",
        "## Boundary",
        "",
        "- Scientific source scope: `KnowledgeReference/` plus explicitly user-verified staged sources.",
        "- Internal planning, findings, tests, and result files are program-state evidence only; they do not become physics authority in this index.",
        "- Index verdicts are retrieval and engineering triage labels. They do not promote any source claim into accepted experimental validation data.",
        "",
        "## Corpus Counts",
        "",
        f"- KnowledgeReference files: {stats['knowledge_reference_file_count']}",
        f"- Top-level source documents: {stats['knowledge_reference_document_count']}",
        f"- Top-level markdown source records: {stats['markdown_source_count']}",
        f"- Top-level JSON metadata records: {stats['json_metadata_count']}",
        f"- External PDF records: {stats['external_pdf_count']}",
        "",
        "| File kind | Count |",
        "| --- | ---: |",
    ]
    for kind, count in stats["file_kind_counts"].items():
        lines.append(f"| `{kind}` | {count} |")
    lines.extend([
        "",
        "## First-Principles Verdict Counts",
        "",
        "| Verdict | Count |",
        "| --- | ---: |",
    ])
    for verdict, count in stats["verdict_counts"].items():
        lines.append(f"| `{verdict}` | {count} |")
    lines.extend([
        "",
        "## Capability Coverage",
        "",
        "| Capability | Records | Direct | Representative source paths |",
        "| --- | ---: | ---: | --- |",
    ])
    for _tag, packet in capability_index.items():
        reps = "<br>".join(
            f"`{record['path']}`"
            for record in packet["representative_records"][:3]
            if record.get("path")
        )
        lines.append(
            f"| {packet['label']} | {packet['record_count']} | "
            f"{packet['direct_record_count']} | {reps} |"
        )
    lines.extend([
        "",
        "## Strongest First-Principles Source Material Found",
        "",
    ])
    for record in direct:
        path = (record.get("paths") or {}).get("markdown")
        lines.append(
            f"- `{path}`: **{record.get('title', 'untitled')}**. "
            f"Verdict `{record.get('first_principles_verdict')}`. "
            f"{record.get('first_principles_use', '')}"
        )
    lines.extend([
        "",
        "## Machine-Readable Index",
        "",
        "- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json` contains full hashes, records, tags, and file inventory.",
        "",
    ])
    return "\n".join(lines)


def _file_kind(path: Path) -> str:
    if path.name == ".DS_Store":
        return "system_metadata"
    rel_parts = path.relative_to(ROOT).parts
    if "chunks" in rel_parts:
        return "derived_chunk"
    if "figures" in rel_parts or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
        return "figure_or_image"
    if path.parent == ROOT / "KnowledgeReference" and path.suffix == ".md":
        return "source_markdown"
    if path.parent == ROOT / "KnowledgeReference" and path.suffix == ".json":
        return "source_json_metadata"
    return "system_metadata"


def _safe_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _read_prefix(path: Path, *, limit: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _metadata_title(metadata: Mapping[str, Any]) -> str | None:
    candidates = [
        metadata.get("title"),
        (metadata.get("citation") or {}).get("title")
        if isinstance(metadata.get("citation"), Mapping)
        else None,
        (metadata.get("metadata") or {}).get("/Title")
        if isinstance(metadata.get("metadata"), Mapping)
        else None,
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def _metadata_authors(metadata: Mapping[str, Any]) -> list[str]:
    authors = metadata.get("authors")
    if isinstance(authors, list):
        return [str(author) for author in authors]
    citation = metadata.get("citation")
    if isinstance(citation, Mapping) and isinstance(citation.get("authors"), list):
        return [str(author) for author in citation["authors"]]
    return []


def _metadata_year(metadata: Mapping[str, Any]) -> int | None:
    for key in ("year", "publication_year"):
        try:
            value = metadata.get(key)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:160]
    return None


def _tags_for_text(text: str) -> list[str]:
    lowered = text.lower()
    tags = [
        tag
        for tag, needles in CAPABILITY_KEYWORDS.items()
        if any(needle in lowered for needle in needles)
    ]
    return tags


def _verdict_for_tags(tags: list[str]) -> str:
    if "dpf_device" in tags and (
        "maxwell_fields" in tags
        or "circuit_coupling" in tags
        or "validation_target" in tags
    ):
        return "directly_needed"
    if tags:
        return "supporting_method"
    return "not_relevant"


def _source_type_for_tags(tags: list[str]) -> str:
    return "dpf_or_first_principles" if "dpf_device" in tags else "background_or_adjacent"


def _first_principles_use(tags: list[str], verdict: str) -> str:
    if verdict == "not_relevant":
        return (
            "No direct first-principles use found by this keyword index; keep "
            "searchable as background unless manual review says otherwise."
        )
    labels = [CAPABILITY_LABELS[tag] for tag in tags[:8] if tag in CAPABILITY_LABELS]
    prefix = "Direct source candidate for" if verdict == "directly_needed" else "Supporting source for"
    return f"{prefix} {'; '.join(labels)}."


def _page_count(metadata: Mapping[str, Any], record: Mapping[str, Any]) -> int:
    for candidate in (metadata.get("page_count"), record.get("page_count")):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            pass
    pages = metadata.get("pages")
    return len(pages) if isinstance(pages, list) else 0


def _count_field(
    metadata: Mapping[str, Any],
    key: str,
    record: Mapping[str, Any],
) -> int:
    for candidate in (metadata.get(key), record.get(key)):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            pass
    if key == "figure_count" and isinstance(metadata.get("figures"), list):
        return len(metadata["figures"])
    return 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_knowledge_reference_path(text: str) -> str | None:
    marker = "KnowledgeReference/"
    if marker not in text:
        return None
    suffix = text.split(marker, 1)[1].strip().rstrip(".)]")
    return marker + suffix


if __name__ == "__main__":
    raise SystemExit(main())
