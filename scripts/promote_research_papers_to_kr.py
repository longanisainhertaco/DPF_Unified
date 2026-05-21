"""Promote local research-paper intake PDFs into KnowledgeReference records.

This is a local source-ingestion utility, not a scientific acceptance tool.
It creates markdown/JSON text-parity records for unique intake PDFs that are
not already represented in KnowledgeReference, then optionally removes exact
byte-for-byte duplicate intake files while preserving one canonical copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:
    import fitz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: PyMuPDF is required for KR promotion") from exc


ROOT = Path(__file__).resolve().parents[1]
INTAKE_DIR = ROOT / "downloaded_books_papers" / "Research Papers"
KR_DIR = ROOT / "KnowledgeReference"
KR_CHUNKS_DIR = KR_DIR / "chunks"
DOCS_DIR = ROOT / "docs"
AUDIT_CSV = DOCS_DIR / "RESEARCH_PAPERS_INTAKE_AUDIT_2026_05_11.csv"
PROMOTION_JSON = DOCS_DIR / "RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.json"
PROMOTION_MD = DOCS_DIR / "RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.md"
CHUNKING_JSON = DOCS_DIR / "KR_TEXTBOOK_CHUNKING_2026_05_11.json"
CHUNKING_MD = DOCS_DIR / "KR_TEXTBOOK_CHUNKING_2026_05_11.md"


PDF_LIKE_SUFFIXES = (".pdf", ".pdf.crdownload")
BOOK_PAGE_THRESHOLD = 120
BOOK_CHUNK_PAGES = 25
KNOWN_EXISTING_ACCESSIONS = {
    "AD1194691": "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md",
    "AD1100306": "KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md",
}
FORCE_PROMOTE_ACCESSIONS = {
    # This source was previously only cited by other KR files. The exact PDF
    # itself was not promoted before this intake pass.
    "1169854",
}
TITLE_OVERRIDES = {
    "1169854": "Fully Kinetic Simulations of MegaJoule-Scale Dense Plasma Focus",
}


@dataclass(frozen=True)
class IntakeFile:
    path: Path
    relpath: str
    sha256: str
    size: int
    accession: str
    title_hint: str
    relevance: str


@dataclass(frozen=True)
class ExistingKRRecord:
    path: str
    title: str
    title_norm: str
    header_text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_spaces(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_match(text: str) -> str:
    text = normalize_spaces(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def slugify(text: str, fallback: str) -> str:
    source = text or fallback
    source = unicodedata.normalize("NFKD", source).encode("ascii", "ignore").decode()
    source = source.lower()
    source = re.sub(r"[^a-z0-9]+", "-", source)
    source = re.sub(r"-+", "-", source).strip("-")
    return (source[:90].strip("-") or fallback.lower())[:100]


def extract_accession(path: Path) -> str:
    name = path.name
    patterns = [
        r"(AD[A-Z]?\d{6,7})",
        r"(DSIAC-\d+)",
        r"(HDIAC-\d+)",
        r"(\d{6,7})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def load_audit_titles() -> dict[str, dict[str, str]]:
    if not AUDIT_CSV.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with AUDIT_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            relpath = row.get("path", "") or row.get("relative_path", "")
            if not relpath:
                continue
            rows[relpath] = {
                "title": normalize_spaces(row.get("title", "")),
                "relevance": normalize_spaces(row.get("relevance", "")),
            }
    return rows


def scan_intake() -> list[IntakeFile]:
    audit = load_audit_titles()
    files: list[IntakeFile] = []
    for path in sorted(INTAKE_DIR.rglob("*")):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if not lower_name.endswith(PDF_LIKE_SUFFIXES):
            continue
        relpath = path.relative_to(INTAKE_DIR).as_posix()
        audit_row = audit.get(relpath, {})
        files.append(
            IntakeFile(
                path=path,
                relpath=relpath,
                sha256=sha256_file(path),
                size=path.stat().st_size,
                accession=extract_accession(path),
                title_hint=audit_row.get("title", ""),
                relevance=audit_row.get("relevance", ""),
            )
        )
    return files


def canonical_sort_key(item: IntakeFile) -> tuple[int, int, int, str]:
    is_crdownload = int(item.relpath.lower().endswith(".crdownload"))
    wave_depth = item.relpath.count("/")
    has_title = 0 if item.title_hint else 1
    return (is_crdownload, has_title, wave_depth, item.relpath.lower())


def group_by_sha(files: list[IntakeFile]) -> dict[str, list[IntakeFile]]:
    groups: dict[str, list[IntakeFile]] = {}
    for item in files:
        groups.setdefault(item.sha256, []).append(item)
    return groups


def markdown_h1(text: str) -> str:
    for line in text.splitlines()[:40]:
        if line.startswith("# "):
            return normalize_spaces(line[2:])
    return ""


def load_existing_kr_index() -> list[ExistingKRRecord]:
    records: list[ExistingKRRecord] = []
    for path in sorted(KR_DIR.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".json"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except UnicodeDecodeError:
            continue
        title = ""
        header_text = path.name + "\n"
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {}
            citation = payload.get("citation", {}) if isinstance(payload, dict) else {}
            if isinstance(citation, dict):
                title = normalize_spaces(str(citation.get("title", "")))
                header_text += json.dumps(citation, ensure_ascii=False)
            for key in ("source_pdf", "source_pdf_sha256"):
                if isinstance(payload, dict) and payload.get(key):
                    header_text += "\n" + str(payload[key])
        else:
            title = markdown_h1(text)
            header_text += text[:5000]
        records.append(
            ExistingKRRecord(
                path=path.relative_to(ROOT).as_posix(),
                title=title,
                title_norm=normalize_match(title),
                header_text=header_text.lower(),
            )
        )
    return records


def extract_doi(text: str) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    if not match:
        return ""
    return match.group(0).rstrip(".,);]")


def is_bad_title(title: str) -> bool:
    normalized = normalize_match(title)
    if len(normalized) < 8:
        return True
    bad_fragments = [
        "untitled",
        "microsoft word",
        "report documentation page",
        "approved for public release",
        "defense technical information center",
        "llnl jrnl",
        "this article has been accepted",
        "this content has been downloaded",
        "please scroll down to see the full text",
        "download details",
        "prescribed by ansi",
        "public reporting burden",
        "maintaining the data needed",
        "needed and completing and reviewing",
        "standard form 298",
        "naval research laboratory",
        "afrl afosr va tr",
        "afrl ry wp tp",
    ]
    return any(fragment in normalized for fragment in bad_fragments)


def candidate_title(metadata_title: str, title_hint: str, pages: list[dict[str, object]], fallback: str) -> str:
    for title in (title_hint, metadata_title):
        title = normalize_spaces(title)
        if title and not is_bad_title(title):
            return title

    first_text = str(pages[0].get("text", "")) if pages else ""
    lines = [normalize_spaces(line) for line in first_text.splitlines()]
    skip = (
        "ad",
        "ada",
        "llnl",
        "approved",
        "distribution",
        "defense technical",
        "form approved",
        "report documentation",
        "public reporting",
        "standard form",
        "technical report",
        "abstract",
        "this content has been downloaded",
        "download details",
        "ip address",
        "please note",
        "home",
        "search",
        "collections",
        "journals",
        "about",
        "contact us",
        "my iopscience",
    )
    for line in lines[:80]:
        if not line or len(line) < 12:
            continue
        if line.lower().startswith(skip):
            continue
        if re.search(r"[A-Za-z]{4}", line):
            return line[:220]
    return fallback


def extract_pdf(path: Path) -> dict[str, object]:
    doc = fitz.open(path)
    try:
        metadata = dict(doc.metadata or {})
        pages = []
        for index, page in enumerate(doc, start=1):
            pages.append({"page": index, "text": page.get_text("text").strip(), "tables": []})
        full_text = "\n".join(str(page["text"]) for page in pages)
        return {
            "page_count": doc.page_count,
            "metadata": metadata,
            "pages": pages,
            "doi": extract_doi(full_text),
            "nonempty_pages": sum(1 for page in pages if str(page["text"]).strip()),
        }
    finally:
        doc.close()


def already_represented(
    item: IntakeFile,
    title: str,
    doi: str,
    kr_records: list[ExistingKRRecord],
) -> tuple[bool, str]:
    for record in kr_records:
        if item.sha256.lower() in record.header_text:
            return True, f"source SHA already appears in source-level KR metadata: {record.path}"
    if item.accession in FORCE_PROMOTE_ACCESSIONS:
        return False, ""
    if item.accession in KNOWN_EXISTING_ACCESSIONS:
        return True, f"known existing KR source: {KNOWN_EXISTING_ACCESSIONS[item.accession]}"
    normalized_title = normalize_match(title)
    if "nrl plasma formulary" in normalized_title and (KR_DIR / "plasma-formulary.md").exists():
        return True, "formulary coverage already exists; edition review still needed"
    for record in kr_records:
        if doi and doi.lower() in record.header_text:
            return True, f"DOI already appears in source-level KR metadata: {doi} ({record.path})"
        if (
            normalized_title
            and not is_bad_title(title)
            and len(normalized_title) >= 35
            and record.title_norm
            and (
                normalized_title == record.title_norm
                or (
                    len(normalized_title) >= 55
                    and len(record.title_norm) >= 35
                    and (normalized_title in record.title_norm or record.title_norm in normalized_title)
                )
            )
        ):
            return True, f"title already appears as a source-level KR title: {record.path}"
    return False, ""


def page_markdown(page: dict[str, object]) -> list[str]:
    page_number = int(page["page"])
    return [
        f"## Page {page_number}",
        "",
        f"_pp. {page_number}-{page_number}_",
        "",
        str(page["text"]).strip(),
        "",
    ]


def should_chunk_markdown(page_count: int) -> bool:
    return page_count >= BOOK_PAGE_THRESHOLD


def write_markdown_chunks(
    *,
    title: str,
    source_rel: str,
    source_sha256: str,
    accession: str,
    doi: str,
    pages: list[dict[str, object]],
    slug: str,
    apply: bool,
) -> list[Path]:
    chunk_dir = KR_CHUNKS_DIR / slug
    chunk_paths: list[Path] = []
    for start in range(0, len(pages), BOOK_CHUNK_PAGES):
        chunk = pages[start : start + BOOK_CHUNK_PAGES]
        first_page = int(chunk[0]["page"])
        last_page = int(chunk[-1]["page"])
        chunk_path = chunk_dir / f"pages-{first_page:04d}-{last_page:04d}.md"
        lines = [
            f"# {title} - Pages {first_page}-{last_page}",
            "",
            f"**Source PDF:** `{source_rel}`  ",
            f"**Source PDF SHA-256:** `{source_sha256}`  ",
            f"**Page range:** {first_page}-{last_page}  ",
            f"**Accession:** `{accession or 'unknown'}`  ",
            f"**DOI:** `{doi or 'not detected'}`  ",
            "**KR ingestion status:** `text_parity_extracted_review_needed`  ",
            "**Validation status:** `source_available_not_target_extracted`  ",
            "",
            "This chunk is extracted text for local source review only. Figures, tables, "
            "plotted curves, and numeric validation targets are not accepted until "
            "separately reviewed and target-extracted.",
            "",
            "---",
            "",
        ]
        for page in chunk:
            lines.extend(page_markdown(page))
        if apply:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path.write_text("\n".join(lines).rstrip() + "\n")
        chunk_paths.append(chunk_path)
    return chunk_paths


def header_lines(
    *,
    title: str,
    source_rel: str,
    source_sha256: str,
    page_count: int,
    nonempty_pages: int,
    accession: str,
    doi: str,
) -> list[str]:
    return [
        f"# {title}",
        "",
        f"**Source PDF:** `{source_rel}`  ",
        f"**Source PDF SHA-256:** `{source_sha256}`  ",
        f"**Pages:** {page_count}  ",
        f"**Nonempty extracted pages:** {nonempty_pages}  ",
        f"**Accession:** `{accession or 'unknown'}`  ",
        f"**DOI:** `{doi or 'not detected'}`  ",
        "**KR ingestion status:** `text_parity_extracted_review_needed`  ",
        "**Validation status:** `source_available_not_target_extracted`  ",
        "",
        "Text extraction is available for local source review. Figures, tables, plotted curves, "
        "and numeric validation targets are not accepted until separately reviewed and target-extracted.",
        "",
        "---",
        "",
    ]


def write_kr_pair(
    item: IntakeFile,
    extracted: dict[str, object],
    title: str,
    slug: str,
    apply: bool,
) -> tuple[Path, Path, list[Path]]:
    md_path = KR_DIR / f"{slug}.md"
    json_path = KR_DIR / f"{slug}.json"
    counter = 2
    while md_path.exists() or json_path.exists():
        md_path = KR_DIR / f"{slug}-{counter}.md"
        json_path = KR_DIR / f"{slug}-{counter}.json"
        counter += 1

    source_rel = item.path.relative_to(ROOT).as_posix()
    pages = extracted["pages"]
    metadata = extracted["metadata"]
    doi = str(extracted.get("doi", ""))
    page_count = int(extracted["page_count"])
    nonempty_pages = int(extracted["nonempty_pages"])
    chunk_paths: list[Path] = []
    if should_chunk_markdown(page_count):
        chunk_paths = write_markdown_chunks(
            title=title,
            source_rel=source_rel,
            source_sha256=item.sha256,
            accession=item.accession,
            doi=doi,
            pages=pages,
            slug=md_path.stem,
            apply=apply,
        )

    json_payload = {
        "source_pdf": item.path.name,
        "source_pdf_relative_path": source_rel,
        "source_pdf_sha256": item.sha256,
        "source_file_size_bytes": item.size,
        "page_count": page_count,
        "citation": {
            "title": title,
            "doi": doi,
            "accession": item.accession,
            "metadata": metadata,
        },
        "kr_ingestion": {
            "date": date.today().isoformat(),
            "source": "downloaded_books_papers/Research Papers",
            "method": "PyMuPDF page.get_text('text')",
            "status": "text_parity_extracted_review_needed",
            "validation_status": "source_available_not_target_extracted",
            "notes": (
                "Local PDF text was extracted for KnowledgeReference search and source review. "
                "Figures, tables, plotted curves, and numeric validation targets are not accepted "
                "until separately reviewed and target-extracted."
            ),
            "nonempty_pages": nonempty_pages,
            "markdown_layout": "chunked" if chunk_paths else "inline",
            "markdown_chunk_pages": BOOK_CHUNK_PAGES if chunk_paths else None,
            "markdown_chunks": [
                path.relative_to(ROOT).as_posix() for path in chunk_paths
            ],
        },
        "table_count": 0,
        "figures": [],
        "pages": pages,
    }

    header = header_lines(
        title=title,
        source_rel=source_rel,
        source_sha256=item.sha256,
        page_count=page_count,
        nonempty_pages=nonempty_pages,
        accession=item.accession,
        doi=doi,
    )
    body: list[str] = []
    if chunk_paths:
        body.extend(
            [
                "## Chunked Text Index",
                "",
                "This book-length source is split into page-range Markdown chunks so the text is readable in review tools. The JSON file preserves the full page list.",
                "",
                "| chunk | page range |",
                "| --- | --- |",
            ]
        )
        for chunk_path in chunk_paths:
            match = re.search(r"pages-(\d+)-(\d+)\.md$", chunk_path.name)
            page_range = "unknown"
            if match:
                page_range = f"{int(match.group(1))}-{int(match.group(2))}"
            body.append(
                f"| `{chunk_path.relative_to(ROOT).as_posix()}` | {page_range} |"
            )
        body.append("")
    else:
        for page in pages:
            body.extend(page_markdown(page))
    md_text = "\n".join(header + body).rstrip() + "\n"

    if apply:
        md_path.write_text(md_text)
        json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    return md_path, json_path, chunk_paths


def parity_check(md_path: Path, json_payload: dict[str, object], extracted: dict[str, object]) -> dict[str, object]:
    if not md_path.exists():
        return {"passed": False, "failures": ["markdown_not_written"]}
    markdown_texts = [md_path.read_text(errors="ignore")]
    chunk_failures: list[str] = []
    ingestion = json_payload.get("kr_ingestion", {})
    chunk_refs = []
    if isinstance(ingestion, dict):
        chunk_refs = list(ingestion.get("markdown_chunks", []) or [])
    for chunk_ref in chunk_refs:
        chunk_path = ROOT / str(chunk_ref)
        if not chunk_path.exists():
            chunk_failures.append(f"markdown_chunk_missing:{chunk_ref}")
            continue
        markdown_texts.append(chunk_path.read_text(errors="ignore"))
    md_norm = normalize_match("\n".join(markdown_texts))
    json_pages = json_payload.get("pages", [])
    failures: list[str] = []
    failures.extend(chunk_failures)
    if int(json_payload.get("page_count", -1)) != int(extracted["page_count"]):
        failures.append("page_count_mismatch")
    if len(json_pages) != int(extracted["page_count"]):
        failures.append("json_page_count_mismatch")
    missing_md_pages: list[int] = []
    for page in extracted["pages"]:
        text = normalize_match(str(page["text"]))
        if text and text not in md_norm:
            missing_md_pages.append(int(page["page"]))
    if missing_md_pages:
        failures.append("markdown_missing_extracted_pages")
    return {"passed": not failures, "failures": failures, "markdown_missing_pages": missing_md_pages}


def promotion_run(apply: bool) -> dict[str, object]:
    files = scan_intake()
    groups = group_by_sha(files)
    kr_records = load_existing_kr_index()

    promoted: list[dict[str, object]] = []
    skipped_existing: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    deleted_duplicates: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []

    for sha, group in sorted(groups.items(), key=lambda entry: sorted(i.relpath for i in entry[1])[0].lower()):
        canonical = sorted(group, key=canonical_sort_key)[0]
        retained.append({"sha256": sha, "canonical_path": canonical.relpath, "duplicate_count": len(group) - 1})
        for duplicate in sorted(group, key=lambda i: i.relpath):
            if duplicate.path == canonical.path:
                continue
            record = {
                "path": duplicate.relpath,
                "sha256": duplicate.sha256,
                "retained_canonical": canonical.relpath,
            }
            if apply:
                duplicate.path.unlink()
                record["deleted"] = True
            else:
                record["deleted"] = False
            deleted_duplicates.append(record)

        if canonical.relpath.lower().endswith(".crdownload"):
            failed.append(
                {
                    "path": canonical.relpath,
                    "sha256": canonical.sha256,
                    "reason": "canonical file is incomplete-download suffix",
                }
            )
            continue

        try:
            extracted = extract_pdf(canonical.path)
        except Exception as exc:  # pragma: no cover - depends on PDFs
            failed.append({"path": canonical.relpath, "sha256": canonical.sha256, "reason": repr(exc)})
            continue

        metadata = extracted["metadata"]
        metadata_title = str(metadata.get("title", ""))
        title = TITLE_OVERRIDES.get(
            canonical.accession,
            candidate_title(metadata_title, canonical.title_hint, extracted["pages"], canonical.path.stem),
        )
        doi = str(extracted.get("doi", ""))
        represented, reason = already_represented(canonical, title, doi, kr_records)
        if represented:
            skipped_existing.append(
                {
                    "path": canonical.relpath,
                    "sha256": canonical.sha256,
                    "title": title,
                    "doi": doi,
                    "accession": canonical.accession,
                    "reason": reason,
                }
            )
            continue

        slug_base = slugify(title, canonical.accession or canonical.path.stem)
        slug = f"{slug_base}-{canonical.sha256[:8]}"
        md_path, json_path, chunk_paths = write_kr_pair(canonical, extracted, title, slug, apply=apply)
        parity = {"passed": None, "failures": [], "markdown_missing_pages": []}
        if apply:
            json_payload = json.loads(json_path.read_text())
            parity = parity_check(md_path, json_payload, extracted)
        promoted.append(
            {
                "path": canonical.relpath,
                "sha256": canonical.sha256,
                "title": title,
                "doi": doi,
                "accession": canonical.accession,
                "relevance": canonical.relevance,
                "pages": int(extracted["page_count"]),
                "nonempty_pages": int(extracted["nonempty_pages"]),
                "markdown": md_path.relative_to(ROOT).as_posix(),
                "markdown_chunks": [
                    path.relative_to(ROOT).as_posix() for path in chunk_paths
                ],
                "json": json_path.relative_to(ROOT).as_posix(),
                "parity": parity,
                "status": "text_parity_extracted_review_needed",
            }
        )

    result = {
        "date": date.today().isoformat(),
        "applied": apply,
        "intake_dir": INTAKE_DIR.relative_to(ROOT).as_posix(),
        "files_scanned": len(files),
        "unique_sha256_payloads": len(groups),
        "promoted_count": len(promoted),
        "skipped_existing_count": len(skipped_existing),
        "failed_count": len(failed),
        "deleted_duplicate_count": len(deleted_duplicates),
        "retained_count": len(retained),
        "promoted": promoted,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "deleted_duplicates": deleted_duplicates,
        "retained": retained,
        "notes": (
            "Promotion means local PDF text was extracted into KnowledgeReference markdown/JSON. "
            "It does not accept figures, tables, plotted curves, numeric targets, or validation claims."
        ),
    }
    return result


def rewrite_existing_large_markdown(json_path: Path, payload: dict[str, object], apply: bool) -> dict[str, object]:
    md_path = json_path.with_suffix(".md")
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        pages = []
    citation = payload.get("citation", {})
    if not isinstance(citation, dict):
        citation = {}
    ingestion = payload.get("kr_ingestion", {})
    if not isinstance(ingestion, dict):
        ingestion = {}

    title = normalize_spaces(str(citation.get("title", "") or md_path.stem))
    doi = str(citation.get("doi", ""))
    accession = str(citation.get("accession", ""))
    source_rel = str(payload.get("source_pdf_relative_path", payload.get("source_pdf", "")))
    source_sha256 = str(payload.get("source_pdf_sha256", ""))
    page_count = int(payload.get("page_count", len(pages)))
    nonempty_pages = int(ingestion.get("nonempty_pages", sum(1 for page in pages if str(page.get("text", "")).strip())))

    chunk_paths = write_markdown_chunks(
        title=title,
        source_rel=source_rel,
        source_sha256=source_sha256,
        accession=accession,
        doi=doi,
        pages=pages,
        slug=md_path.stem,
        apply=apply,
    )
    updated_payload = dict(payload)
    updated_ingestion = dict(ingestion)
    updated_ingestion.update(
        {
            "markdown_layout": "chunked",
            "markdown_chunk_pages": BOOK_CHUNK_PAGES,
            "markdown_chunks": [
                path.relative_to(ROOT).as_posix() for path in chunk_paths
            ],
        }
    )
    updated_payload["kr_ingestion"] = updated_ingestion

    body = [
        "## Chunked Text Index",
        "",
        "This book-length source is split into page-range Markdown chunks so the text is readable in review tools. The JSON file preserves the full page list.",
        "",
        "| chunk | page range |",
        "| --- | --- |",
    ]
    for chunk_path in chunk_paths:
        match = re.search(r"pages-(\d+)-(\d+)\.md$", chunk_path.name)
        page_range = "unknown"
        if match:
            page_range = f"{int(match.group(1))}-{int(match.group(2))}"
        body.append(f"| `{chunk_path.relative_to(ROOT).as_posix()}` | {page_range} |")
    body.append("")
    md_text = "\n".join(
        header_lines(
            title=title,
            source_rel=source_rel,
            source_sha256=source_sha256,
            page_count=page_count,
            nonempty_pages=nonempty_pages,
            accession=accession,
            doi=doi,
        )
        + body
    ).rstrip() + "\n"

    if apply:
        md_path.write_text(md_text)
        json_path.write_text(json.dumps(updated_payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
        parity = parity_check(
            md_path,
            updated_payload,
            {"page_count": page_count, "pages": pages},
        )
    else:
        parity = {"passed": None, "failures": [], "markdown_missing_pages": []}

    return {
        "markdown": md_path.relative_to(ROOT).as_posix(),
        "json": json_path.relative_to(ROOT).as_posix(),
        "title": title,
        "source_pdf_relative_path": source_rel,
        "source_pdf_sha256": source_sha256,
        "page_count": page_count,
        "chunk_count": len(chunk_paths),
        "markdown_chunks": [
            path.relative_to(ROOT).as_posix() for path in chunk_paths
        ],
        "parity": parity,
    }


def chunk_existing_large_records(apply: bool, only_stems: set[str] | None = None) -> dict[str, object]:
    chunked: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for json_path in sorted(KR_DIR.glob("*.json")):
        if only_stems and json_path.stem not in only_stems:
            continue
        try:
            payload = json.loads(json_path.read_text())
        except Exception as exc:
            failed.append({"json": json_path.relative_to(ROOT).as_posix(), "reason": repr(exc)})
            continue
        pages = payload.get("pages", [])
        page_count = int(payload.get("page_count", len(pages) if isinstance(pages, list) else 0))
        if page_count < BOOK_PAGE_THRESHOLD:
            skipped.append(
                {
                    "json": json_path.relative_to(ROOT).as_posix(),
                    "page_count": page_count,
                    "reason": "below_chunk_page_threshold",
                }
            )
            continue
        if not isinstance(pages, list) or not pages:
            failed.append(
                {
                    "json": json_path.relative_to(ROOT).as_posix(),
                    "reason": "missing_pages_array",
                }
            )
            continue
        chunked.append(rewrite_existing_large_markdown(json_path, payload, apply=apply))
    return {
        "date": date.today().isoformat(),
        "applied": apply,
        "page_threshold": BOOK_PAGE_THRESHOLD,
        "chunk_pages": BOOK_CHUNK_PAGES,
        "only_stems": sorted(only_stems or []),
        "chunked_count": len(chunked),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "chunked": chunked,
        "skipped": skipped,
        "failed": failed,
        "notes": (
            "Chunking rewrites book-length top-level markdown as an index and "
            "stores full extracted text in page-range markdown chunks. It does "
            "not accept figures, tables, plotted curves, numeric targets, or "
            "validation claims."
        ),
    }


def write_chunking_reports(result: dict[str, object], apply: bool) -> None:
    if not apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    CHUNKING_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    lines = [
        "# KR Textbook Chunking",
        "",
        f"Generated: {result['date']}",
        "",
        "Source guardrail: chunking changes Markdown layout only. It does not accept figures, tables, plotted curves, numeric targets, or validation claims.",
        "",
        "## Summary",
        "",
        f"- Page threshold: {result['page_threshold']}",
        f"- Pages per chunk: {result['chunk_pages']}",
        f"- Chunked records: {result['chunked_count']}",
        f"- Skipped below threshold: {result['skipped_count']}",
        f"- Failed: {result['failed_count']}",
        "",
        "## Chunked Records",
        "",
        "| title | pages | chunks | markdown index | json | parity |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in result["chunked"]:
        parity = item.get("parity", {})
        lines.append(
            "| {title} | {pages} | {chunks} | {md} | {json} | {parity} |".format(
                title=str(item["title"]).replace("|", "\\|"),
                pages=item["page_count"],
                chunks=item["chunk_count"],
                md=item["markdown"],
                json=item["json"],
                parity=parity.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "| json | reason |",
            "| --- | --- |",
        ]
    )
    for item in result["failed"]:
        lines.append(f"| {item['json']} | {item['reason']} |")
    CHUNKING_MD.write_text("\n".join(lines).rstrip() + "\n")


def write_reports(result: dict[str, object], apply: bool) -> None:
    if not apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    PROMOTION_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n")

    promoted = result["promoted"]
    skipped = result["skipped_existing"]
    failed = result["failed"]
    deleted = result["deleted_duplicates"]

    lines = [
        "# Research Papers KR Promotion",
        "",
        f"Generated: {result['date']}",
        "",
        "Source guardrail: this report records local source ingestion only. Text parity does not accept figures, tables, plotted curves, numeric targets, or validation claims.",
        "",
        "## Summary",
        "",
        f"- Files scanned: {result['files_scanned']}",
        f"- Unique SHA-256 payloads: {result['unique_sha256_payloads']}",
        f"- Promoted into `KnowledgeReference/`: {result['promoted_count']}",
        f"- Skipped because already represented: {result['skipped_existing_count']}",
        f"- Failed or not promoted: {result['failed_count']}",
        f"- Exact duplicate intake files deleted: {result['deleted_duplicate_count']}",
        "",
        "## Promoted Sources",
        "",
        "| source | title | pages | sha12 | KR markdown | chunks | KR json | status |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for item in promoted:
        lines.append(
            "| {source} | {title} | {pages} | {sha12} | {md} | {chunks} | {json} | {status} |".format(
                source=item["path"],
                title=str(item["title"]).replace("|", "\\|"),
                pages=item["pages"],
                sha12=str(item["sha256"])[:12],
                md=item["markdown"],
                chunks=len(item.get("markdown_chunks", [])),
                json=item["json"],
                status=item["status"],
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
    for item in skipped:
        lines.append(
            "| {source} | {title} | {sha12} | {reason} |".format(
                source=item["path"],
                title=str(item["title"]).replace("|", "\\|"),
                sha12=str(item["sha256"])[:12],
                reason=str(item["reason"]).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Deleted Exact Duplicates",
            "",
            "| deleted path | retained canonical | sha12 |",
            "| --- | --- | --- |",
        ]
    )
    for item in deleted:
        lines.append(
            f"| {item['path']} | {item['retained_canonical']} | {str(item['sha256'])[:12]} |"
        )
    lines.extend(
        [
            "",
            "## Failures / Not Promoted",
            "",
            "| source | sha12 | reason |",
            "| --- | --- | --- |",
        ]
    )
    for item in failed:
        lines.append(f"| {item['path']} | {str(item['sha256'])[:12]} | {item['reason']} |")
    PROMOTION_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write KR records/reports")
    parser.add_argument(
        "--chunk-existing-large",
        action="store_true",
        help="rewrite existing large KR markdown records as chunked indexes",
    )
    parser.add_argument(
        "--chunk-stem",
        action="append",
        default=[],
        help="when chunking existing large records, limit to this KR json/md stem",
    )
    args = parser.parse_args()

    if args.chunk_existing_large:
        only_stems = set(args.chunk_stem) if args.chunk_stem else None
        result = chunk_existing_large_records(apply=args.apply, only_stems=only_stems)
        write_chunking_reports(result, apply=args.apply)
        print(
            "chunked={chunked_count} skipped={skipped_count} failed={failed_count}".format(
                **result
            )
        )
        return 0 if not result["failed"] else 1

    result = promotion_run(apply=args.apply)
    write_reports(result, apply=args.apply)
    print(
        "files={files_scanned} unique={unique_sha256_payloads} promoted={promoted_count} "
        "skipped_existing={skipped_existing_count} failed={failed_count} "
        "deleted_duplicates={deleted_duplicate_count}".format(**result)
    )
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
