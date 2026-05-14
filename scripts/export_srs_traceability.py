#!/usr/bin/env python3
"""Export the DPF SRS candidate baseline into a traceability matrix.

The project plans to use Doorstop for repository-native requirements
management, but Doorstop may not be installed in every development shell. This
script keeps the candidate baseline machine-readable and import-ready without
requiring external dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KNOWN_PRIORITIES = {"P0", "P1", "P2"}
KNOWN_STATUSES = {
    "implemented",
    "partial",
    "planned",
    "blocked",
    "deferred",
    "rejected",
}
KNOWN_VERIFICATION_METHODS = {
    "test",
    "inspection",
    "analysis",
    "demonstration",
    "review",
}


@dataclass(frozen=True)
class RequirementRow:
    req_id: str
    priority: str
    requirement: str
    owner: str
    status: str
    verification_methods: tuple[str, ...]
    evidence: str
    source_section: str

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(re.findall(r"`([^`]+)`", self.evidence)))

    @property
    def doorstop_item(self) -> dict[str, object]:
        return {
            "uid": self.req_id,
            "level": "shall",
            "normative": self.status != "rejected",
            "status": self.status,
            "priority": self.priority,
            "owner": self.owner,
            "verification": list(self.verification_methods),
            "evidence_refs": list(self.evidence_refs),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.req_id,
            "priority": self.priority,
            "requirement": self.requirement,
            "owner": self.owner,
            "status": self.status,
            "verification_methods": list(self.verification_methods),
            "acceptance_evidence_or_blocker": self.evidence,
            "evidence_refs": list(self.evidence_refs),
            "source_section": self.source_section,
            "doorstop_import": self.doorstop_item,
            "import_guardrail": (
                "Import as satisfied only when status is implemented and "
                "evidence_refs or explicit inspection evidence are present."
            ),
        }


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise ValueError(f"not a markdown table row: {line!r}")
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_separator(cells: Iterable[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _normalise_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(method.strip().lower() for method in raw.split(",") if method.strip())
    unknown = sorted(set(methods) - KNOWN_VERIFICATION_METHODS)
    if unknown:
        raise ValueError(f"unknown verification method(s): {', '.join(unknown)}")
    return methods


def parse_requirements_baseline(path: Path) -> list[RequirementRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[RequirementRow] = []
    current_section = ""
    headers: list[str] | None = None

    for line in lines:
        if line.startswith("## "):
            current_section = line.removeprefix("## ").strip()
            headers = None
            continue

        if not line.strip().startswith("|"):
            continue

        cells = _split_markdown_row(line)
        if _is_separator(cells):
            continue

        if "ID" in cells and "Requirement" in cells:
            headers = cells
            continue

        if not headers or "Candidate Requirements" not in current_section and "Deferred" not in current_section:
            continue

        if len(cells) != len(headers):
            raise ValueError(f"table row has {len(cells)} cells but expected {len(headers)}: {line}")

        record = dict(zip(headers, cells))
        evidence = record.get("Acceptance evidence or blocker", record.get("Decision needed", "")).strip()
        row = RequirementRow(
            req_id=record["ID"],
            priority=record["Priority"],
            requirement=record["Requirement"],
            owner=record["Owner"],
            status=record["Status"].lower(),
            verification_methods=_normalise_methods(record.get("Verification method", "")),
            evidence=evidence,
            source_section=current_section,
        )
        rows.append(row)

    validate_requirements(rows)
    return rows


def validate_requirements(rows: list[RequirementRow]) -> None:
    if not rows:
        raise ValueError("no requirements found")

    ids = [row.req_id for row in rows]
    duplicates = sorted({req_id for req_id in ids if ids.count(req_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate requirement id(s): {', '.join(duplicates)}")

    for row in rows:
        if row.priority not in KNOWN_PRIORITIES:
            raise ValueError(f"{row.req_id}: unknown priority {row.priority!r}")
        if row.status not in KNOWN_STATUSES:
            raise ValueError(f"{row.req_id}: unknown status {row.status!r}")
        if not row.requirement:
            raise ValueError(f"{row.req_id}: empty requirement")
        if row.priority in {"P0", "P1"} and not row.verification_methods:
            raise ValueError(f"{row.req_id}: P0/P1 requirement lacks verification method")
        if row.status == "implemented" and not row.evidence:
            raise ValueError(f"{row.req_id}: implemented requirement lacks evidence")


def build_traceability_matrix(rows: list[RequirementRow], source_path: Path) -> dict[str, object]:
    status_counts = {status: 0 for status in sorted(KNOWN_STATUSES)}
    priority_counts = {priority: 0 for priority in sorted(KNOWN_PRIORITIES)}
    for row in rows:
        status_counts[row.status] += 1
        priority_counts[row.priority] += 1

    return {
        "schema": "dpf.srs.traceability.v1",
        "source_document": source_path.as_posix(),
        "doorstop_status": "staged_not_imported",
        "doorstop_document_prefix": "DPF",
        "summary": {
            "requirement_count": len(rows),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "p0_or_p1_without_verification": [],
        },
        "requirements": [row.to_record() for row in rows],
    }


def write_json(matrix: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(rows: list[RequirementRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "id",
                "priority",
                "requirement",
                "owner",
                "status",
                "verification_methods",
                "acceptance_evidence_or_blocker",
                "evidence_refs",
                "source_section",
                "doorstop_uid",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.req_id,
                    "priority": row.priority,
                    "requirement": row.requirement,
                    "owner": row.owner,
                    "status": row.status,
                    "verification_methods": ";".join(row.verification_methods),
                    "acceptance_evidence_or_blocker": row.evidence,
                    "evidence_refs": ";".join(row.evidence_refs),
                    "source_section": row.source_section,
                    "doorstop_uid": row.req_id,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/DPF_REQUIREMENTS_BASELINE.md"),
        help="Path to the markdown candidate requirements baseline.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("docs/SRS_TRACEABILITY_MATRIX.json"),
        help="Output JSON traceability matrix.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/SRS_TRACEABILITY_MATRIX.csv"),
        help="Output CSV traceability matrix.",
    )
    args = parser.parse_args()

    rows = parse_requirements_baseline(args.baseline)
    matrix = build_traceability_matrix(rows, args.baseline)
    write_json(matrix, args.json)
    write_csv(rows, args.csv)
    print(
        f"exported {len(rows)} requirements to {args.json.as_posix()} "
        f"and {args.csv.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
