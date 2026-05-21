"""Promote the 2026-05-20 user-supplied DPF PDFs into KR.

The user supplied nine local files:

- ``scholz_Recent progress.pdf`` -- Scholz et al. 2001 PF-1000 1 MJ
  hardware/diagnostics paper.
- ``The_need_of_using_anomalous_resisti.pdf`` -- Bruzzone and Bernal 2001
  LHI anomalous-resistivity paper.
- ``scholz_PF-1000 device.pdf`` -- Scholz et al. 2000 PF-1000 hardware,
  pulsed-power, chamber, and diagnostics paper.
- ``herold1989.pdf`` -- Herold et al. 1989 POSEIDON/PF-360 cross-machine
  comparative plasma-focus study.
- ``scholz1999.pdf`` -- Scholz et al. 1999 PF-1000 foam-liner experiment.
- ``loarer2007.pdf`` -- Loarer et al. 2007 tokamak gas-balance and fuel
  retention review.
- ``chouhan,+Artical-8.pdf`` -- Shakya et al. 2015 Lee-model PF1000/PF400
  comparison paper.
- ``gribkov2007.pdf`` -- Gribkov et al. 2007 PF-1000 Part II beam and
  neutron-emission paper.
- ``Dense_magnetized_plasma_and_its_app.pdf`` -- Gribkov and Malaquias 2006
  IAEA CRP dense-magnetized-plasma applications review.

This utility promotes only new source material.  Exact SHA duplicates of
existing KR records are reported but not duplicated.  Promotion remains
text-parity source availability only; it does not accept figures, numerical
targets, runtime closures, or first-principles validation.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    / "2026-05-20-user-supplied-papers"
)
PROMOTION_JSON = ROOT / "docs" / "USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.json"
PROMOTION_MD = ROOT / "docs" / "USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.md"

SCHOLZ_SOURCE = Path("/Users/anthonyzamora/Downloads/scholz_Recent progress.pdf")
BRUZZONE_BERNAL_SOURCE = Path(
    "/Users/anthonyzamora/Downloads/The_need_of_using_anomalous_resisti.pdf"
)
SCHOLZ_PF1000_DEVICE_SOURCE = Path(
    "/Users/anthonyzamora/Downloads/scholz_PF-1000 device.pdf"
)
HEROLD_1989_SOURCE = Path("/Users/anthonyzamora/Downloads/herold1989.pdf")
SCHOLZ_1999_SOURCE = Path("/Users/anthonyzamora/Downloads/scholz1999.pdf")
LOARER_2007_SOURCE = Path("/Users/anthonyzamora/Downloads/loarer2007.pdf")
SHAKYA_2015_SOURCE = Path("/Users/anthonyzamora/Downloads/chouhan,+Artical-8.pdf")
GRIBKOV_2007_SOURCE = Path("/Users/anthonyzamora/Downloads/gribkov2007.pdf")
DENSE_MAGNETIZED_PLASMA_SOURCE = Path(
    "/Users/anthonyzamora/Downloads/Dense_magnetized_plasma_and_its_app.pdf"
)

SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_path": SCHOLZ_SOURCE,
        "canonical_filename": "scholz_2001_recent_progress_1mj_pf_research.pdf",
        "title": "Recent progress in 1 MJ Plasma-Focus research",
        "authors": (
            "Scholz, M.; Karpinski, L.; Paduch, M.; Tomaszewski, K.; "
            "Miklaszewski, R.; Szydlowski, A."
        ),
        "journal": "Nukleonika 46(1):35-39 (2001)",
        "priority": "P1",
        "scope": "pf1000_2001_hardware_geometry_diagnostics_candidate",
        "resolves_blocker_candidates": (
            "PF1000-BLK-004 cathode rod length; PF1000-BLK-015 insulator "
            "outer radius; PF1000-BLK-009 anode end-face hole context only "
            "(after target extraction + review)"
        ),
        "source_origin": "user_supplied_local_pdf",
    },
    {
        "source_path": BRUZZONE_BERNAL_SOURCE,
        "canonical_filename": "bruzzone_bernal_2001_nukleonika_v46n2p059.pdf",
        "title": (
            "The need of using anomalous resistivity due to Lower Hybrid "
            "Instabilities in plasma-magnetic field interfaces"
        ),
        "authors": "Bruzzone, H.; Bernal, L.",
        "journal": "Nukleonika 46(2):59-61 (2001)",
        "priority": "P1",
        "scope": "dpf_lhi_anomalous_resistivity_quantitative_candidate",
        "resolves_blocker_candidates": (
            "CLOSURE-BLK-ANOM-001 (already represented in local KR; "
            "target extraction/review still required)"
        ),
        "source_origin": "user_supplied_local_pdf_duplicate_of_sprint6_source",
    },
    {
        "source_path": SCHOLZ_PF1000_DEVICE_SOURCE,
        "canonical_filename": "scholz_2000_pf1000_device_nukleonika_v45p155.pdf",
        "title": "PF-1000 device",
        "authors": "Scholz, M.; Miklaszewski, R.; Gribkov, V. A.; Mezzetti, F.",
        "journal": "Nukleonika 45(3):155-158 (2000)",
        "priority": "P1",
        "scope": "pf1000_facility_hardware_bank_diagnostics_candidate",
        "resolves_blocker_candidates": (
            "PF1000-BLK-004 cathode rod length; PF1000 hardware cage/rod "
            "context; PF1000 circuit/bank source context; chamber geometry "
            "context (after target extraction + review)"
        ),
        "source_origin": "user_supplied_local_pdf",
    },
    {
        "source_path": HEROLD_1989_SOURCE,
        "canonical_filename": (
            "herold_1989_large_plasma_focus_comparative_analysis.pdf"
        ),
        "title": (
            "Comparative analysis of large plasma focus experiments performed "
            "at IPF, Stuttgart, and IPJ, Swierk"
        ),
        "authors": "Herold, H.; Jerzykiewicz, A.; Sadowski, M.; Schmidt, H.",
        "journal": "Nuclear Fusion 29(8):1255-1269 (1989)",
        "priority": "P2",
        "scope": "poseidon_pf360_cross_machine_scaling_startup_context",
        "resolves_blocker_candidates": (
            "Cross-machine startup, insulator, electrode, yield-scaling, and "
            "saturation context only; not PF-1000 same-scope acceptance"
        ),
        "source_origin": "user_supplied_local_pdf",
    },
    {
        "source_path": SCHOLZ_1999_SOURCE,
        "canonical_filename": (
            "scholz_1999_foam_liner_driven_by_plasma_focus_current_sheath.pdf"
        ),
        "title": "Foam liner driven by a plasma focus current sheath",
        "authors": (
            "Scholz, M.; Karpinski, L.; Stepniewski, W.; Branitski, A. V.; "
            "Fedulov, M. V.; Medovschikov, S. F.; Nedoseev, S. L.; "
            "Smirnov, V. P.; Zurin, M. V.; Szydlowski, A."
        ),
        "journal": "Physics Letters A 262:453-456 (1999)",
        "priority": "P2",
        "scope": "pf1000_modified_foam_liner_current_sheath_context",
        "resolves_blocker_candidates": (
            "PF-1000 current-sheath interaction/radiation context only; "
            "modified foam-liner load is not standard PF-1000 same-scope "
            "geometry or shot acceptance"
        ),
        "source_origin": "user_supplied_local_pdf",
    },
    {
        "source_path": LOARER_2007_SOURCE,
        "canonical_filename": (
            "loarer_2007_gas_balance_and_fuel_retention_in_fusion_devices.pdf"
        ),
        "title": "Gas balance and fuel retention in fusion devices",
        "authors": (
            "Loarer, T.; Brosset, C.; Bucalossi, J.; Coad, P.; Esser, G.; "
            "Hogan, J.; Likonen, J.; Mayer, M.; Morgan, Ph.; Philipps, V.; "
            "Rohde, V.; Roth, J.; Rubel, M.; Tsitrone, E.; Widdowson, A."
        ),
        "journal": "Nuclear Fusion 47:1112-1120 (2007)",
        "doi": "10.1088/0029-5515/47/9/007",
        "priority": "P3",
        "scope": "tokamak_pwi_gas_balance_fuel_retention_context",
        "resolves_blocker_candidates": (
            "Plasma-wall fuel-retention methodology context only; not a DPF "
            "source and not accepted for PF-1000 first-principles closure"
        ),
        "source_origin": "user_supplied_local_pdf",
    },
    {
        "source_path": SHAKYA_2015_SOURCE,
        "canonical_filename": (
            "shakya_2015_comparison_pf1000_pf400_lee_model.pdf"
        ),
        "title": "Comparison of Plasma Dynamics in Plasma Focus Devices PF1000 and PF400",
        "authors": "Shakya, A.; Gautam, P.; Khanal, R.",
        "journal": "Journal of Nepal Physical Society 3(1):55-62 (2015)",
        "priority": "P2",
        "scope": "pf1000_pf400_reduced_lee_model_comparison_context",
        "resolves_blocker_candidates": (
            "Reduced Lee-model PF1000/PF400 comparison context only; may "
            "support baseline/comparator bookkeeping but not first-principles "
            "runtime acceptance"
        ),
        "source_origin": "user_supplied_local_pdf",
    },
    {
        "source_path": GRIBKOV_2007_SOURCE,
        "canonical_filename": (
            "gribkov_2007_pf1000_part2_fast_electron_ion_neutron_jphysd.pdf"
        ),
        "title": (
            "Plasma dynamics in the PF-1000 device under full-scale energy "
            "storage: II. Fast electron and ion characteristics versus "
            "neutron emission parameters and gun optimization perspectives"
        ),
        "authors": (
            "Gribkov, V. A.; Banaszak, A.; Bienkowska, B.; Dubrovsky, A. V.; "
            "Ivanova-Stanik, I.; Jakubowski, L.; Karpinski, L.; "
            "Miklaszewski, R. A.; Paduch, M.; Sadowski, M. J.; Scholz, M.; "
            "Szydlowski, A.; Tomaszewski, K."
        ),
        "journal": "Journal of Physics D: Applied Physics 40:3592-3607 (2007)",
        "doi": "10.1088/0022-3727/40/12/008",
        "priority": "P1",
        "scope": "pf1000_full_energy_fast_electron_ion_neutron_authority",
        "resolves_blocker_candidates": (
            "Already represented in KR under scholz-2007-pf1000-part2-jphysd; "
            "supports neutron/beam target extraction and gun-geometry "
            "optimization context after target extraction/review"
        ),
        "source_origin": "user_supplied_local_pdf_existing_kr_equivalent",
    },
    {
        "source_path": DENSE_MAGNETIZED_PLASMA_SOURCE,
        "canonical_filename": (
            "gribkov_malaquias_2006_dense_magnetized_plasma_applications.pdf"
        ),
        "title": (
            "Dense magnetized plasma and its applications: review of the "
            "3-year activity of the IAEA Co-ordinated Research Programme"
        ),
        "authors": "Gribkov, V. A.; Malaquias, A.",
        "journal": "Nukleonika 51(1):5-13 (2006)",
        "priority": "P2",
        "scope": "dense_magnetized_plasma_applications_context",
        "resolves_blocker_candidates": (
            "DMP/DPF applications and device-technology context only unless "
            "a later target extraction identifies source-specific PF-1000 "
            "hardware or diagnostic values"
        ),
        "source_origin": "user_supplied_local_pdf",
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
        relevance="user_supplied_2026_05_20_dpf_source",
    )


def _stage(source: dict[str, Any], *, apply: bool) -> Path:
    source_path = Path(source["source_path"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    staged = INTAKE_DIR / str(source["canonical_filename"])
    if apply:
        INTAKE_DIR.mkdir(parents=True, exist_ok=True)
        if not staged.exists() or sha256_file(staged) != sha256_file(source_path):
            shutil.copy2(source_path, staged)
    return staged


def _promote_one(
    source: dict[str, Any],
    kr_records: list[Any],
    *,
    apply: bool,
) -> dict[str, Any]:
    staged = _stage(source, apply=apply)
    if apply:
        item = _intake_file(staged)
    else:
        # Dry-runs use the original location so the file does not need to be
        # copied before duplicate detection.
        item = IntakeFile(
            path=Path(source["source_path"]),
            relpath=str(source["source_path"]),
            sha256=sha256_file(Path(source["source_path"])),
            size=Path(source["source_path"]).stat().st_size,
            accession="",
            title_hint="",
            relevance="user_supplied_2026_05_20_dpf_source",
        )
    title = str(source["title"])
    source_doi = str(source.get("doi", "")).strip()
    extracted = dict(extract_pdf(item.path))
    if not source_doi:
        extracted["doi"] = ""
    else:
        extracted["doi"] = source_doi
    represented, reason = already_represented(
        item,
        title,
        source_doi,
        kr_records,
    )
    base_record: dict[str, Any] = {
        "source_path": str(source["source_path"]),
        "staged_path": (
            staged.relative_to(ROOT).as_posix() if staged.is_absolute() else str(staged)
        ),
        "sha256": item.sha256,
        "title": title,
        "authors": str(source["authors"]),
        "journal": str(source["journal"]),
        "priority": str(source["priority"]),
        "scope": str(source["scope"]),
        "resolves_blocker_candidates": str(source["resolves_blocker_candidates"]),
        "source_origin": str(source["source_origin"]),
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
    }
    if represented:
        return {
            **base_record,
            "status": "skipped_existing_kr_source",
            "reason": reason,
        }

    slug = f"{slugify(title, item.path.stem)}-{item.sha256[:8]}"
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
        ingestion.update(
            {
                "source": INTAKE_DIR.relative_to(ROOT).as_posix(),
                "status": "text_parity_extracted_review_needed",
                "validation_status": "source_available_not_target_extracted",
                "priority": source["priority"],
                "scope": source["scope"],
                "promotion_report": PROMOTION_MD.relative_to(ROOT).as_posix(),
                "sprint": "sprint6_user_supplied_2026_05_20",
                "resolves_blocker_candidates": source[
                    "resolves_blocker_candidates"
                ],
                "source_origin": source["source_origin"],
                "authors_at_acquisition": source["authors"],
                "journal_at_acquisition": source["journal"],
                "accepted_runtime_claim": False,
                "can_support_first_principles_acceptance": False,
                "notes": (
                    "User-supplied local PDF promoted to KnowledgeReference as "
                    "text-parity source material only. Figures, tables, plotted "
                    "curves, numeric validation targets, runtime closures, and "
                    "first-principles claims remain unaccepted until separately "
                    "target-extracted, reviewed, consumed by code, and passed "
                    "through numerical and same-scope gates."
                ),
            }
        )
        payload["kr_ingestion"] = ingestion
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
            + "\n"
        )
        parity = parity_check(md_path, payload, extracted)

    return {
        **base_record,
        "status": "text_parity_extracted_review_needed",
        "pages": int(extracted["page_count"]),
        "nonempty_pages": int(extracted["nonempty_pages"]),
        "markdown": md_path.relative_to(ROOT).as_posix(),
        "markdown_chunks": [
            chunk_path.relative_to(ROOT).as_posix() for chunk_path in chunk_paths
        ],
        "json": json_path.relative_to(ROOT).as_posix(),
        "parity": parity,
    }


def promote(apply: bool) -> dict[str, Any]:
    kr_records = load_existing_kr_index()
    promoted: list[dict[str, Any]] = []
    skipped_existing: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for source in SOURCES:
        try:
            record = _promote_one(source, kr_records, apply=apply)
        except Exception as exc:  # pragma: no cover - file-system dependent.
            failed.append(
                {
                    "source_path": str(source["source_path"]),
                    "title": str(source["title"]),
                    "reason": repr(exc),
                }
            )
            continue
        if record["status"] == "skipped_existing_kr_source":
            skipped_existing.append(record)
        else:
            promoted.append(record)

    return {
        "date": "2026-05-20",
        "sprint": "sprint6_user_supplied_sources",
        "applied": apply,
        "intake_dir": INTAKE_DIR.relative_to(ROOT).as_posix(),
        "files_scanned": len(SOURCES),
        "promoted_count": len(promoted),
        "skipped_existing_count": len(skipped_existing),
        "failed_count": len(failed),
        "promoted": promoted,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "guardrail": (
            "Promotion is source availability only. Existing duplicates are "
            "not re-promoted. New KR records stay "
            "source_available_not_target_extracted and cannot support "
            "runtime or first-principles acceptance until target extraction, "
            "review, code consumption, numerical acceptance, and same-scope "
            "certificate gates all pass."
        ),
    }


def write_reports(result: dict[str, Any], *, apply: bool) -> None:
    if not apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    PROMOTION_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    )
    lines = [
        "# User-Supplied Paper Intake (2026-05-20)",
        "",
        "Generated: 2026-05-20",
        "",
        "The user supplied nine local PDFs. This intake promotes only new",
        "source material into `KnowledgeReference/` and records exact SHA",
        "duplicates without creating duplicate KR records.",
        "",
        "Guardrail: source availability only. `accepted_runtime_claim` and",
        "`can_support_first_principles_acceptance` remain `False`.",
        "",
        "## Summary",
        "",
        f"- Files scanned: {result['files_scanned']}",
        f"- Promoted into `KnowledgeReference/`: {result['promoted_count']}",
        f"- Skipped existing KR source: {result['skipped_existing_count']}",
        f"- Failed: {result['failed_count']}",
        f"- accepted_runtime_claim: `{result['accepted_runtime_claim']}`",
        (
            f"- can_support_first_principles_acceptance: "
            f"`{result['can_support_first_principles_acceptance']}`"
        ),
        "",
        "## Promoted Sources",
        "",
        (
            "| source | title | journal | pages | sha12 | KR md | KR json | "
            "priority | scope | candidate support | parity |"
        ),
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not result["promoted"]:
        lines.extend(
            [
                "",
                "No new KR records were created by this idempotent pass. The",
                "sources below were already represented in `KnowledgeReference/`",
                "and are target-extracted separately by",
                "`src/dpf/first_principles/sprint6_user_target_extractions.py`.",
            ]
        )
    for item in result["promoted"]:
        parity = item.get("parity", {})
        lines.append(
            "| {src} | {title} | {journal} | {pages} | {sha} | {md} | {js} | "
            "{priority} | {scope} | {resolves} | {parity} |".format(
                src=item["staged_path"],
                title=str(item["title"]).replace("|", "\\|"),
                journal=str(item["journal"]).replace("|", "\\|"),
                pages=item.get("pages", ""),
                sha=str(item["sha256"])[:12],
                md=item.get("markdown", ""),
                js=item.get("json", ""),
                priority=item["priority"],
                scope=item["scope"],
                resolves=str(item["resolves_blocker_candidates"]).replace(
                    "|", "\\|"
                ),
                parity=parity.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Existing KR Sources",
            "",
            "| source | title | sha12 | reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in result["skipped_existing"]:
        lines.append(
            "| {src} | {title} | {sha} | {reason} |".format(
                src=item["source_path"],
                title=str(item["title"]).replace("|", "\\|"),
                sha=str(item["sha256"])[:12],
                reason=str(item["reason"]).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "| source | title | reason |",
            "| --- | --- | --- |",
        ]
    )
    for item in result["failed"]:
        lines.append(
            f"| {item['source_path']} | {item['title']} | {item['reason']} |"
        )
    PROMOTION_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write KR/report files")
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
