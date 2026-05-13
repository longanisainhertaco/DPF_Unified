"""Reconcile May 12 intake reports after manual out-of-scope demotions."""

from __future__ import annotations

import json
from pathlib import Path

import promote_research_papers_to_kr as promote
import promote_user_pdf_batch_2026_05_12 as may12


ROOT = Path(__file__).resolve().parents[1]
PROMOTION_JSON = ROOT / "docs" / "USER_PDF_KR_PROMOTION_2026_05_12.json"

DEMOTIONS = {
    "symons1994.pdf": "out_of_scope_jstor_social_science_review_not_dpf_or_simulation_source",
}

FALSE_EXISTING_PROMOTIONS = {
    "trunk1975.pdf": {
        "title": "Numerical Parameter Studies for the Dense Plasma Focus",
        "reason": (
            "generic IOP cover-page title matched an unrelated Kortanek 2014 "
            "KR record; SHA differs, so Trunk 1975 requires its own KR pair"
        ),
    }
}


def promote_false_existing(path_name: str, title: str) -> dict[str, object]:
    path = may12.BATCH_DIR / path_name
    sha = promote.sha256_file(path)
    item = promote.IntakeFile(
        path=path,
        relpath=path_name,
        sha256=sha,
        size=path.stat().st_size,
        accession=promote.extract_accession(path),
        title_hint=title,
        relevance="promote_to_kr_source_review",
    )
    extracted = promote.extract_pdf(path)
    slug = f"{promote.slugify(title, path.stem)}-{sha[:8]}"
    md_path, json_path, chunk_paths = promote.write_kr_pair(item, extracted, title, slug, apply=True)
    json_payload = json.loads(json_path.read_text())
    parity = promote.parity_check(md_path, json_payload, extracted)
    return {
        "path": path_name,
        "sha256": sha,
        "title": title,
        "doi": str(extracted.get("doi", "")),
        "accession": item.accession,
        "relevance": item.relevance,
        "pages": int(extracted["page_count"]),
        "nonempty_pages": int(extracted["nonempty_pages"]),
        "markdown": md_path.relative_to(ROOT).as_posix(),
        "markdown_chunks": [chunk.relative_to(ROOT).as_posix() for chunk in chunk_paths],
        "json": json_path.relative_to(ROOT).as_posix(),
        "parity": parity,
        "status": "text_parity_extracted_review_needed",
        "manual_reconciliation": FALSE_EXISTING_PROMOTIONS[path_name]["reason"],
    }


def main() -> int:
    may12.configure_promoter()
    result = json.loads(PROMOTION_JSON.read_text())
    demoted: list[dict[str, object]] = []
    retained_promoted = []
    for item in result.get("promoted", []):
        path = str(item.get("path", ""))
        if path in DEMOTIONS:
            demoted.append(
                {
                    "path": path,
                    "title": item.get("title", ""),
                    "subject_class": "manual_demoted",
                    "relevance": "stage_for_review_not_physics_evidence",
                    "reason": DEMOTIONS[path],
                    "previous_markdown": item.get("markdown", ""),
                    "previous_json": item.get("json", ""),
                }
            )
        else:
            retained_promoted.append(item)
    result["promoted"] = retained_promoted
    result["promoted_count"] = len(retained_promoted)
    result["selected_for_promotion_count"] = int(result.get("selected_for_promotion_count", 0)) - len(demoted)
    result["files_scanned"] = int(result.get("files_scanned", 0)) - len(demoted)
    result["unique_sha256_payloads"] = int(result.get("unique_sha256_payloads", 0)) - len(demoted)
    result["retained"] = [
        item for item in result.get("retained", []) if item.get("canonical_path") not in DEMOTIONS
    ]
    result["retained_count"] = len(result["retained"])
    stage_only = list(result.get("staged_not_promoted", []))
    existing_stage_paths = {str(item.get("path", "")) for item in stage_only}
    for item in demoted:
        if str(item["path"]) not in existing_stage_paths:
            stage_only.append(item)
    result["staged_not_promoted"] = stage_only
    result["staged_not_promoted_count"] = len(stage_only)
    result["manual_demotions"] = demoted

    false_existing_promoted: list[dict[str, object]] = []
    skipped_existing = []
    existing_promoted_paths = {str(item.get("path", "")) for item in result.get("promoted", [])}
    for item in result.get("skipped_existing", []):
        path = str(item.get("path", ""))
        if path in FALSE_EXISTING_PROMOTIONS and path not in existing_promoted_paths:
            promoted = promote_false_existing(path, FALSE_EXISTING_PROMOTIONS[path]["title"])
            false_existing_promoted.append(promoted)
            result["promoted"].append(promoted)
            existing_promoted_paths.add(path)
        else:
            skipped_existing.append(item)
    result["skipped_existing"] = skipped_existing
    result["skipped_existing_count"] = len(skipped_existing)
    result["promoted_count"] = len(result["promoted"])
    result["false_existing_promotions"] = false_existing_promoted

    result["selection_guardrail"] = (
        result.get("selection_guardrail", "")
        + " Manual demotions are stage-only and must not be cited as DPF physics authority. "
        + "False existing-source matches are promoted only when SHA-256 and source identity prove they are distinct."
    ).strip()
    promote.write_reports(result, apply=True)
    may12.append_exclusion_report(stage_only)
    print(
        "demoted={demoted} false_existing_promoted={false_existing} promoted={promoted_count} "
        "skipped_existing={skipped_existing_count} stage_only={staged_not_promoted_count}".format(
            demoted=len(demoted),
            false_existing=len(false_existing_promoted),
            **result,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
