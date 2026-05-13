"""Validate May 12 user PDF intake at source-authority level.

This script validates local source identity, hash consistency, KR promotion
mapping, text parity status, source-fidelity mapping, and target-candidate
classification. It does not validate any scientific target values or accept
figure/table digitization.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import promote_research_papers_to_kr as promote


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
INTAKE_JSON = DOCS_DIR / "USER_PDF_INTAKE_2026_05_12.json"
PROMOTION_JSON = DOCS_DIR / "USER_PDF_KR_PROMOTION_2026_05_12.json"
FIDELITY_JSON = DOCS_DIR / "USER_PDF_KR_SOURCE_FIDELITY_AUDIT_2026_05_12.json"
TRIAGE_JSON = DOCS_DIR / "USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.json"
REPORT_JSON = DOCS_DIR / "USER_PDF_MAY12_SOURCE_VALIDATION_2026_05_12.json"
REPORT_MD = DOCS_DIR / "USER_PDF_MAY12_SOURCE_VALIDATION_2026_05_12.md"

KORTANEK_JSON = (
    ROOT
    / "KnowledgeReference"
    / "this-content-has-been-downloaded-from-iopscience-please-scroll-down-to-see-the-full-text-7dbd9199.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def kr_record(path: str) -> dict[str, Any]:
    return load_json(ROOT / path)


def validate_kr_mapping(item: dict[str, Any], fidelity_by_source: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = str(item["path"])
    json_path = str(item["json"])
    payload = kr_record(json_path)
    source_rel = str(payload.get("source_pdf_relative_path", ""))
    source_path = ROOT / source_rel
    actual_sha = promote.sha256_file(source_path) if source_path.exists() else ""
    failures: list[str] = []
    if not source_path.exists():
        failures.append("source_pdf_missing")
    if actual_sha and actual_sha != item["sha256"]:
        failures.append("source_pdf_sha_mismatch")
    if str(payload.get("source_pdf_sha256", "")) != item["sha256"]:
        failures.append("kr_source_sha_mismatch")
    parity = item.get("parity", {})
    if parity.get("passed") is not True:
        failures.append("promotion_text_parity_not_passed")
    fidelity = fidelity_by_source.get(source)
    if not fidelity:
        failures.append("source_fidelity_record_missing")
    else:
        if fidelity.get("sha256") != item["sha256"]:
            failures.append("source_fidelity_sha_mismatch")
        if fidelity.get("json") != json_path:
            failures.append("source_fidelity_json_mismatch")
    return {
        "source": source,
        "title": item.get("title", ""),
        "doi": item.get("doi", ""),
        "sha256": item["sha256"],
        "knowledge_reference_json": json_path,
        "knowledge_reference_markdown": item["markdown"],
        "source_pdf_relative_path": source_rel,
        "source_pdf_exists": source_path.exists(),
        "parity_passed": parity.get("passed") is True,
        "source_fidelity_status": fidelity.get("status", "missing") if fidelity else "missing",
        "status": "source_validated" if not failures else "source_validation_failed",
        "failures": failures,
    }


def validate_stage_only(item: dict[str, Any], intake_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = str(item["path"])
    intake = intake_by_name.get(source, {})
    destination = str(intake.get("destination", ""))
    source_path = ROOT / destination if destination else None
    failures: list[str] = []
    if not source_path or not source_path.exists():
        failures.append("stage_only_pdf_missing")
    if source_path and source_path.exists() and promote.sha256_file(source_path) != intake.get("sha256"):
        failures.append("stage_only_sha_mismatch")
    return {
        "source": source,
        "title": item.get("title", ""),
        "reason": item.get("reason", ""),
        "sha256": intake.get("sha256", ""),
        "staged_path": destination,
        "status": "stage_only_validated_not_physics_authority" if not failures else "stage_only_validation_failed",
        "failures": failures,
    }


def validate_kortanek_repair() -> dict[str, Any]:
    payload = load_json(KORTANEK_JSON)
    review = payload.get("source_fidelity_review", {})
    source_rel = str(payload.get("source_pdf_relative_path", ""))
    source_path = ROOT / source_rel
    sha = str(payload.get("source_pdf_sha256", ""))
    failures: list[str] = []
    if not source_path.exists():
        failures.append("kortanek_source_missing")
    elif promote.sha256_file(source_path) != sha:
        failures.append("kortanek_source_sha_mismatch")
    if review.get("source_pdf") != "2026-05-11-user-ingest/kortanek2014.pdf":
        failures.append("kortanek_review_source_not_restored")
    if review.get("source_pdf_sha256") != sha:
        failures.append("kortanek_review_sha_not_restored")
    return {
        "knowledge_reference_json": KORTANEK_JSON.relative_to(ROOT).as_posix(),
        "source_pdf_relative_path": source_rel,
        "source_pdf_sha256": sha,
        "source_fidelity_review_source": review.get("source_pdf", ""),
        "source_fidelity_review_sha256": review.get("source_pdf_sha256", ""),
        "status": "repaired_and_validated" if not failures else "repair_validation_failed",
        "failures": failures,
    }


def build_report() -> dict[str, Any]:
    intake = load_json(INTAKE_JSON)
    promotion = load_json(PROMOTION_JSON)
    fidelity = load_json(FIDELITY_JSON)
    triage = load_json(TRIAGE_JSON)

    intake_by_name = {
        Path(str(record.get("destination") or record.get("path", ""))).name: record
        for record in intake.get("records", [])
        if record.get("destination") or record.get("path")
    }
    fidelity_by_source = {str(record["source"]): record for record in fidelity.get("records", [])}
    triage_by_source = {str(entry["source"]): entry for entry in triage.get("entries", [])}

    source_records = [
        validate_kr_mapping(item, fidelity_by_source)
        for item in promotion.get("promoted", [])
    ]
    stage_only_records = [
        validate_stage_only(item, intake_by_name)
        for item in promotion.get("staged_not_promoted", [])
    ]
    for record in source_records:
        triage_entry = triage_by_source.get(record["source"], {})
        record["validation_role"] = triage_entry.get("role", "not_triaged")
        record["priority"] = triage_entry.get("priority", "not_triaged")
        record["next_actions"] = triage_entry.get("next_actions", [])
        if record["status"] == "source_validated":
            if record["validation_role"] == "target_extraction_candidate":
                record["source_validation_status"] = "source_validated_target_extraction_needed"
            elif record["validation_role"] == "method_reference_mapping":
                record["source_validation_status"] = "source_validated_method_reference"
            elif record["validation_role"] == "review_context_only":
                record["source_validation_status"] = "source_validated_review_context_only"
            elif record["validation_role"] == "materials_context":
                record["source_validation_status"] = "source_validated_materials_context"
            else:
                record["source_validation_status"] = "source_validated_manual_triage_needed"
        else:
            record["source_validation_status"] = "source_validation_failed"

    failures = [
        {"source": record["source"], "failures": record["failures"]}
        for record in source_records + stage_only_records
        if record["failures"]
    ]
    kortanek_repair = validate_kortanek_repair()
    if kortanek_repair["failures"]:
        failures.append({"source": "kortanek2014_repair", "failures": kortanek_repair["failures"]})

    role_counts = Counter(record["source_validation_status"] for record in source_records)
    stage_counts = Counter(record["status"] for record in stage_only_records)
    return {
        "date": date.today().isoformat(),
        "guardrail": (
            "This validates local source identity and source-authority readiness. "
            "It does not accept scientific target values, plotted curves, tables, "
            "formula thresholds, uncertainty values, or validation pass/fail criteria."
        ),
        "source_reports": {
            "intake": INTAKE_JSON.relative_to(ROOT).as_posix(),
            "promotion": PROMOTION_JSON.relative_to(ROOT).as_posix(),
            "source_fidelity": FIDELITY_JSON.relative_to(ROOT).as_posix(),
            "triage": TRIAGE_JSON.relative_to(ROOT).as_posix(),
        },
        "promoted_source_records_checked": len(source_records),
        "stage_only_records_checked": len(stage_only_records),
        "target_extraction_candidates_source_validated": role_counts["source_validated_target_extraction_needed"],
        "method_or_context_sources_validated": sum(
            role_counts[key]
            for key in (
                "source_validated_method_reference",
                "source_validated_review_context_only",
                "source_validated_materials_context",
                "source_validated_manual_triage_needed",
            )
        ),
        "stage_only_validated_count": stage_counts["stage_only_validated_not_physics_authority"],
        "failure_count": len(failures),
        "role_counts": dict(sorted(role_counts.items())),
        "stage_only_counts": dict(sorted(stage_counts.items())),
        "false_existing_promotions": promotion.get("false_existing_promotions", []),
        "kortanek_repair_validation": kortanek_repair,
        "source_records": sorted(source_records, key=lambda r: str(r["source"]).lower()),
        "stage_only_records": sorted(stage_only_records, key=lambda r: str(r["source"]).lower()),
        "failures": failures,
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# May 12 User PDF Source Validation",
        "",
        f"Generated: {report['date']}",
        "",
        report["guardrail"],
        "",
        "## Summary",
        "",
        f"- Promoted source records checked: {report['promoted_source_records_checked']}",
        f"- Stage-only records checked: {report['stage_only_records_checked']}",
        f"- Source-validated target-extraction candidates: {report['target_extraction_candidates_source_validated']}",
        f"- Source-validated method/context records: {report['method_or_context_sources_validated']}",
        f"- Stage-only records validated as non-authority: {report['stage_only_validated_count']}",
        f"- Validation failures: {report['failure_count']}",
        "",
        "## Target Candidates",
        "",
        "| priority | source | status | KR json | next actions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in report["source_records"]:
        if record["source_validation_status"] != "source_validated_target_extraction_needed":
            continue
        lines.append(
            "| {priority} | `{source}` | `{status}` | `{json}` | {actions} |".format(
                priority=record["priority"],
                source=record["source"],
                status=record["source_validation_status"],
                json=record["knowledge_reference_json"],
                actions=", ".join(f"`{action}`" for action in record["next_actions"]),
            )
        )
    lines.extend(
        [
            "",
            "## Method And Context Sources",
            "",
            "| role | source | status | KR json |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in report["source_records"]:
        if record["source_validation_status"] == "source_validated_target_extraction_needed":
            continue
        lines.append(
            "| `{role}` | `{source}` | `{status}` | `{json}` |".format(
                role=record["validation_role"],
                source=record["source"],
                status=record["source_validation_status"],
                json=record["knowledge_reference_json"],
            )
        )
    lines.extend(
        [
            "",
            "## Stage-Only Records",
            "",
            "| source | status | reason |",
            "| --- | --- | --- |",
        ]
    )
    for record in report["stage_only_records"]:
        lines.append(
            "| `{source}` | `{status}` | {reason} |".format(
                source=record["source"],
                status=record["status"],
                reason=str(record["reason"]).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Repaired False Match",
            "",
            "| item | status | details |",
            "| --- | --- | --- |",
        ]
    )
    for item in report["false_existing_promotions"]:
        lines.append(
            "| `{path}` | `promoted_as_distinct_source` | {details} |".format(
                path=item["path"],
                details=str(item.get("manual_reconciliation", "")).replace("|", "\\|"),
            )
        )
    repair = report["kortanek_repair_validation"]
    lines.append(
        "| `kortanek2014_source_fidelity` | `{status}` | `{source}` |".format(
            status=repair["status"],
            source=repair["source_fidelity_review_source"],
        )
    )
    lines.extend(
        [
            "",
            "## Validation Boundary",
            "",
            "The five target candidates are validated only as local source-authority candidates. They still need source-line review, typed target extraction, unit normalization, uncertainty handling, and independent accepted review for any figure/table digitization before they can support validation thresholds.",
        ]
    )
    if report["failures"]:
        lines.extend(["", "## Failures", "", "| source | failures |", "| --- | --- |"])
        for failure in report["failures"]:
            lines.append(f"| `{failure['source']}` | {', '.join(failure['failures'])} |")
    REPORT_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    report = build_report()
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    write_markdown(report)
    print(
        "promoted={promoted_source_records_checked} stage_only={stage_only_records_checked} "
        "target_candidates={target_extraction_candidates_source_validated} failures={failure_count}".format(
            **report
        )
    )
    return 0 if report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
