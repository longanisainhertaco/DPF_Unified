"""Promote the Sprint 6 2026-05-20 acquired PDFs into KR.

This is a scoped source-ingestion utility for the three free Nukleonika
PDFs downloaded under the Sprint 6 goal (WS1 + WS2):

- Bruzzone & Bernal (2001) Nukleonika 46:59-61 anomalous resistivity LHI
- Bruzzone (2001) Nukleonika 46 suppl.1:S3-S7 PF anomalous resistivity
- Szydlowski/Miklaszewski et al. (2001) Nukleonika 46 suppl.1:S61-S64
  PF-1000 large electrodes neutron/ion emission

Promotion means local PDF text becomes searchable KnowledgeReference
material; it does NOT accept figures, tables, plotted curves, numeric
targets, runtime closures, or validation claims. Each KR record is marked
``text_parity_extracted_review_needed`` /
``source_available_not_target_extracted``. Runtime acceptance still
requires KR target extraction, runtime consumption, numerical acceptance,
same-scope comparator, and certificate gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from promote_research_papers_to_kr import (
    ROOT,
    IntakeFile,
    already_represented,
    extract_pdf,
    load_existing_kr_index,
    parity_check,
    sha256_file,
    slugify,
    write_kr_pair,
)

INTAKE_DIR = (
    ROOT
    / "downloaded_books_papers"
    / "Research Papers"
    / "2026-05-20-sprint6-acquisitions"
)
PROMOTION_JSON = ROOT / "docs" / "SPRINT6_KR_PROMOTION_2026_05_20.json"
PROMOTION_MD = ROOT / "docs" / "SPRINT6_KR_PROMOTION_2026_05_20.md"

SOURCES: tuple[dict[str, str], ...] = (
    {
        "filename": "bruzzone_bernal_2001_nukleonika_v46n2p059.pdf",
        "title": (
            "The need of using anomalous resistivity due to Lower Hybrid "
            "Instabilities in plasma-magnetic field interfaces"
        ),
        "authors": "Bruzzone, H.; Bernal, L.",
        "journal": "Nukleonika 46(2):59-61 (2001)",
        "status": "text_parity_extracted_review_needed",
        "priority": "P1",
        "scope": "dpf_lhi_anomalous_resistivity_quantitative_candidate",
        "resolves_blocker_candidates": "CLOSURE-BLK-ANOM-001 (after target extraction + review)",
        "source_origin": "ICHTJ Nukleonika open-access archive",
        "url": "http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46n2p059f.pdf",
    },
    {
        "filename": "bruzzone_2001_nukleonika_v46s1p003.pdf",
        "title": "The role of anomalous resistivities in Plasma Focus discharges",
        "authors": "Bruzzone, H.",
        "journal": "Nukleonika 46 suppl.1:S3-S7 (2001)",
        "status": "text_parity_extracted_review_needed",
        "priority": "P1",
        "scope": "dpf_anomalous_resistivity_dpf_scope_candidate",
        "resolves_blocker_candidates": "CLOSURE-BLK-ANOM-001 (after target extraction + review)",
        "source_origin": "ICHTJ Nukleonika open-access archive",
        "url": "http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p003f.pdf",
    },
    {
        "filename": "szydlowski_miklaszewski_2001_nukleonika_v46s1p061.pdf",
        "title": (
            "Neutron and fast ion emission from PF-1000 facility equipped "
            "with new large electrodes"
        ),
        "authors": (
            "Szydlowski, A.; Scholz, M.; Karpinski, L.; Sadowski, M.; "
            "Tomaszewski, K.; Paduch, M.; Miklaszewski, R."
        ),
        "journal": "Nukleonika 46 suppl.1:S61-S64 (2001)",
        "status": "text_parity_extracted_review_needed",
        "priority": "P2",
        "scope": "pf1000_large_electrodes_geometry_neutron_emission_candidate",
        "resolves_blocker_candidates": (
            "PF1000-BLK-009 hardware-scope hollow-bore "
            "(after target extraction + review)"
        ),
        "source_origin": "ICHTJ Nukleonika open-access archive",
        "url": "http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p061f.pdf",
    },
)


def _intake_file(path: Path) -> IntakeFile:
    return IntakeFile(
        path=path,
        relpath=path.relative_to(INTAKE_DIR).as_posix(),
        sha256=sha256_file(path),
        size=path.stat().st_size,
        accession="",
        title_hint="",
        relevance="sprint6_free_acquisition_p1_p2_candidate",
    )


def _promote_source(
    source: dict[str, str],
    kr_records: list[Any],
    *,
    apply: bool,
) -> dict[str, Any]:
    path = INTAKE_DIR / source["filename"]
    item = _intake_file(path)
    extracted = extract_pdf(path)
    title = source["title"]

    represented, reason = already_represented(
        item,
        title,
        str(extracted.get("doi", "")),
        kr_records,
    )
    if represented:
        return {
            "path": item.relpath,
            "sha256": item.sha256,
            "title": title,
            "doi": str(extracted.get("doi", "")),
            "status": "skipped_existing",
            "reason": reason,
        }

    slug = f"{slugify(title, path.stem)}-{item.sha256[:8]}"
    md_path, json_path, chunk_paths = write_kr_pair(
        item,
        extracted,
        title,
        slug,
        apply=apply,
    )
    parity = {"passed": None, "failures": [], "markdown_missing_pages": []}
    if apply:
        payload = json.loads(json_path.read_text())
        ingestion = dict(payload.get("kr_ingestion", {}))
        ingestion["source"] = INTAKE_DIR.relative_to(ROOT).as_posix()
        ingestion["status"] = source["status"]
        ingestion["validation_status"] = "source_available_not_target_extracted"
        ingestion["priority"] = source["priority"]
        ingestion["scope"] = source["scope"]
        ingestion["promotion_report"] = PROMOTION_MD.relative_to(ROOT).as_posix()
        ingestion["sprint"] = "sprint6_2026_05_20"
        ingestion["resolves_blocker_candidates"] = (
            source["resolves_blocker_candidates"]
        )
        ingestion["external_url_at_acquisition"] = source["url"]
        ingestion["source_origin"] = source["source_origin"]
        ingestion["authors_at_acquisition"] = source["authors"]
        ingestion["journal_at_acquisition"] = source["journal"]
        ingestion["accepted_runtime_claim"] = False
        ingestion["can_support_first_principles_acceptance"] = False
        ingestion["notes"] = (
            "Promoted from the 2026-05-20 Sprint 6 acquisition of three free "
            "Nukleonika open-access PDFs. Text-parity ingestion only. "
            "Figures, tables, plotted curves, numeric validation targets, "
            "runtime closures, and first-principles claims are not accepted "
            "until separately reviewed and target-extracted. accepted_runtime_"
            "claim and can_support_first_principles_acceptance both remain "
            "False."
        )
        payload["kr_ingestion"] = ingestion
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
            + "\n"
        )
        parity = parity_check(md_path, payload, extracted)

    return {
        "path": item.relpath,
        "sha256": item.sha256,
        "title": title,
        "authors": source["authors"],
        "journal": source["journal"],
        "url": source["url"],
        "doi": str(extracted.get("doi", "")),
        "pages": int(extracted["page_count"]),
        "nonempty_pages": int(extracted["nonempty_pages"]),
        "markdown": md_path.relative_to(ROOT).as_posix(),
        "markdown_chunks": [
            chunk_path.relative_to(ROOT).as_posix() for chunk_path in chunk_paths
        ],
        "json": json_path.relative_to(ROOT).as_posix(),
        "priority": source["priority"],
        "scope": source["scope"],
        "resolves_blocker_candidates": source["resolves_blocker_candidates"],
        "status": source["status"],
        "parity": parity,
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
    }


def promote(apply: bool) -> dict[str, Any]:
    kr_records = load_existing_kr_index()
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for source in SOURCES:
        try:
            record = _promote_source(source, kr_records, apply=apply)
        except Exception as exc:  # pragma: no cover - source-file dependent.
            failed.append({"path": source["filename"], "reason": repr(exc)})
            continue
        if record.get("status") == "skipped_existing":
            skipped.append(record)
        else:
            promoted.append(record)

    return {
        "date": "2026-05-20",
        "sprint": "sprint6",
        "applied": apply,
        "intake_dir": INTAKE_DIR.relative_to(ROOT).as_posix(),
        "files_scanned": len(SOURCES),
        "promoted_count": len(promoted),
        "skipped_existing_count": len(skipped),
        "failed_count": len(failed),
        "promoted": promoted,
        "skipped_existing": skipped,
        "failed": failed,
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "guardrail": (
            "Promotion is local KnowledgeReference text-parity ingestion only. "
            "Raw PDFs and text-parity KR records remain "
            "source_available_not_target_extracted until typed target "
            "extraction and review are complete. accepted_runtime_claim and "
            "can_support_first_principles_acceptance both remain False on "
            "every promoted record."
        ),
    }


def write_reports(result: dict[str, Any], apply: bool) -> None:
    if not apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    PROMOTION_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    )
    lines = [
        "# Sprint 6 KR Promotion (2026-05-20)",
        "",
        "Generated: 2026-05-20",
        "",
        "Sprint 6 (`/goal`) WS1 + WS2: three free Nukleonika open-access PDFs",
        "downloaded with SHA-256 verification and ingested into",
        "`KnowledgeReference/` as text-parity records.",
        "",
        "Source guardrail: text-parity ingestion only. Figures, tables, plotted",
        "curves, numeric validation targets, runtime closures, and first-",
        "principles claims are NOT accepted until separately reviewed and",
        "target-extracted. `accepted_runtime_claim` and",
        "`can_support_first_principles_acceptance` both remain `False` on every",
        "promoted record.",
        "",
        "## Summary",
        "",
        f"- Files scanned: {result['files_scanned']}",
        f"- Promoted into `KnowledgeReference/`: {result['promoted_count']}",
        (
            f"- Skipped because already represented: "
            f"{result['skipped_existing_count']}"
        ),
        f"- Failed or not promoted: {result['failed_count']}",
        f"- accepted_runtime_claim: `{result['accepted_runtime_claim']}`",
        (
            f"- can_support_first_principles_acceptance: "
            f"`{result['can_support_first_principles_acceptance']}`"
        ),
        "",
        "## Promoted Sources",
        "",
        (
            "| source | title | authors | journal | URL | pages | sha12 | KR md "
            "| KR json | priority | scope | resolves | parity |"
        ),
        (
            "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- "
            "| --- | --- |"
        ),
    ]
    for item in result["promoted"]:
        parity = item.get("parity", {})
        lines.append(
            "| {src} | {ti} | {au} | {jr} | {url} | {pg} | {sha} | {md} "
            "| {js} | {pr} | {sc} | {rs} | {pa} |".format(
                src=item["path"],
                ti=str(item["title"]).replace("|", "\\|"),
                au=str(item["authors"]).replace("|", "\\|"),
                jr=str(item["journal"]).replace("|", "\\|"),
                url=item["url"],
                pg=item["pages"],
                sha=str(item["sha256"])[:12],
                md=item["markdown"],
                js=item["json"],
                pr=item["priority"],
                sc=item["scope"],
                rs=str(item["resolves_blocker_candidates"]).replace("|", "\\|"),
                pa=parity.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Existing KR Coverage",
            "",
            "| source | title | sha12 | reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in result["skipped_existing"]:
        lines.append(
            "| {src} | {ti} | {sha} | {rs} |".format(
                src=item["path"],
                ti=str(item["title"]).replace("|", "\\|"),
                sha=str(item["sha256"])[:12],
                rs=str(item["reason"]).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Failures / Not Promoted",
            "",
            "| source | reason |",
            "| --- | --- |",
        ]
    )
    for item in result["failed"]:
        lines.append(f"| {item['path']} | {item['reason']} |")
    PROMOTION_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="write KR records and reports"
    )
    args = parser.parse_args()
    result = promote(apply=args.apply)
    write_reports(result, apply=args.apply)
    print(
        "files={files_scanned} promoted={promoted_count} "
        "skipped_existing={skipped_existing_count} failed={failed_count}".format(
            **result
        )
    )
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
