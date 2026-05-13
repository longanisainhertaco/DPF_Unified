"""Stage the 2026-05-12 user-supplied PDF batch for KR intake.

This is a local file intake utility, not a scientific acceptance tool. It
copies one canonical file for each unique SHA-256 payload into the active
research-paper intake tree and writes an audit report for the batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:
    import fitz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: PyMuPDF is required for PDF intake") from exc


ROOT = Path(__file__).resolve().parents[1]
INTAKE_DIR = ROOT / "downloaded_books_papers" / "Research Papers"
BATCH_DIR = INTAKE_DIR / "2026-05-12-user-ingest"
DOCS_DIR = ROOT / "docs"
REPORT_JSON = DOCS_DIR / "USER_PDF_INTAKE_2026_05_12.json"
REPORT_MD = DOCS_DIR / "USER_PDF_INTAKE_2026_05_12.md"
AUDIT_CSV = DOCS_DIR / "USER_PDF_INTAKE_2026_05_12.csv"

INPUT_PATHS = [
    "/Users/anthonyzamora/Downloads/apostolou2020.pdf",
    "/Users/anthonyzamora/Downloads/symons1994.pdf",
    "/Users/anthonyzamora/Downloads/kasperczuk2002.pdf",
    "/Users/anthonyzamora/Downloads/10.1088@1742-6596@370@1@012059.pdf",
    "/Users/anthonyzamora/Downloads/auluck2014.pdf",
    "/Users/anthonyzamora/Downloads/trunk1975.pdf",
    "/Users/anthonyzamora/Downloads/sadowski2008.pdf",
    "/Users/anthonyzamora/Downloads/alexiou2002.pdf",
    "/Users/anthonyzamora/Downloads/lindemuth1982.pdf",
    "/Users/anthonyzamora/Downloads/kubes2020.pdf",
    "/Users/anthonyzamora/Downloads/Simulation-and-Modeling.pdf",
    "/Users/anthonyzamora/Downloads/Numerical-Simulation-of-Pulsed-Plasma-Thruster.pdf",
    "/Users/anthonyzamora/Downloads/Mathematical-Modeling-and-Simulation-of-Systems.pdf",
    "/Users/anthonyzamora/Downloads/Mathematical-Modeling-and-Simulation-of-Systems-2.pdf",
    "/Users/anthonyzamora/Downloads/Monte-Carlo-Simulation-of-Neutral-Particle-Transport.pdf",
    "/Users/anthonyzamora/Downloads/Hybrid-Modeling-and-Simulation.pdf",
    "/Users/anthonyzamora/Downloads/Numerical-Simulation-of-Pulsed-Plasma-Thruster-2.pdf",
    "/Users/anthonyzamora/Downloads/Numerical-simulation-of-equilibrium-air-plasma-flow-in-the-induction-chamber-of-a-high-power.pdf",
    "/Users/anthonyzamora/Downloads/The-role-of-Pauli-principle-in-simulations-of-classical-plasma.pdf",
    "/Users/anthonyzamora/Downloads/A-Hybrid-Quantum-Classical-Particle-in-Cell-Method-for-Plasma-Simulations.pdf",
    "/Users/anthonyzamora/Downloads/timofeev2011.pdf",
    "/Users/anthonyzamora/Downloads/Monte-Carlo-Simulation-of-Neutral-Particle-Transport-2.pdf",
    "/Users/anthonyzamora/Downloads/Hybrid-Modeling-and-Simulation-2.pdf",
    "/Users/anthonyzamora/Downloads/baxevanis2018.pdf",
    "/Users/anthonyzamora/Downloads/chen2019.pdf",
    "/Users/anthonyzamora/Downloads/matsumoto2007.pdf",
    "/Users/anthonyzamora/Downloads/verboncoeur2005.pdf",
    "/Users/anthonyzamora/Downloads/oh2014.pdf",
    "/Users/anthonyzamora/Downloads/Numerical-simulation-of-deuterium-retention-in-tungsten-under-ELM-like-conditions.pdf",
    "/Users/anthonyzamora/Downloads/urano2018.pdf",
    "/Users/anthonyzamora/Downloads/bilbao2006.pdf",
    "/Users/anthonyzamora/Downloads/Large-Language-Models-A-Deep-Dive.pdf",
    "/Users/anthonyzamora/Downloads/A-survey-of-large-language-models-for-cyber-threat-detection.pdf",
    "/Users/anthonyzamora/Downloads/Gradient-Based-Physics-Informed-Neural-Network.pdf",
    "/Users/anthonyzamora/Downloads/Poisson-Boltzmann-based-machine-learning-model-for-electrostatic-analysis.pdf",
    "/Users/anthonyzamora/Downloads/Machine-Learning-of-Forced-Convection-Heat-Transfer.pdf",
    "/Users/anthonyzamora/Downloads/Linear-Algebra-for-Physics.pdf",
    "/Users/anthonyzamora/Downloads/Precalculus.pdf",
    "/Users/anthonyzamora/Library/Application Support/Claude/local-agent-mode-sessions/54aa705f-32d4-4552-8759-47722d5fe095/96556f3c-1f13-4a64-8f2d-95f9e6e1aaf4/local_41cfa8ca-9d72-4194-b964-54a062bb06c4/uploads/Mathematics-for-Engineers-and-Scientists.pdf",
]


@dataclass(frozen=True)
class InputRecord:
    input_index: int
    path: Path
    exists: bool
    sha256: str
    size: int
    page_count: int
    nonempty_pages: int
    title: str
    doi: str
    subject_class: str
    relevance: str
    destination: str
    canonical_for_sha: bool
    duplicate_of: str
    already_in_kr: str
    error: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def normalize_match(text: str) -> str:
    text = normalize_spaces(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def safe_filename(path: Path, sha256: str, used: set[str]) -> str:
    name = unicodedata.normalize("NFKD", path.name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9._@+-]+", "-", name).strip("-") or f"{sha256[:12]}.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    candidate = name
    if candidate in used:
        stem = Path(name).stem
        candidate = f"{stem}-{sha256[:12]}.pdf"
    used.add(candidate)
    return candidate


def extract_doi(text: str) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    return match.group(0).rstrip(".,);]") if match else ""


def bad_title(title: str) -> bool:
    norm = normalize_match(title)
    return (
        len(norm) < 8
        or "untitled" in norm
        or "microsoft word" in norm
        or "report documentation page" in norm
        or "form approved" in norm
    )


def first_text_title(text: str, fallback: str) -> str:
    skip_prefixes = (
        "abstract",
        "keywords",
        "doi",
        "arxiv",
        "accepted",
        "copyright",
        "available online",
        "journal of",
        "proceedings",
    )
    for line in text.splitlines()[:80]:
        clean = normalize_spaces(line)
        if len(clean) < 12:
            continue
        if clean.lower().startswith(skip_prefixes):
            continue
        if re.search(r"[A-Za-z]{4}", clean):
            return clean[:220]
    return fallback


def pdf_metadata(path: Path) -> tuple[int, int, str, str]:
    doc = fitz.open(path)
    try:
        metadata_title = normalize_spaces(str((doc.metadata or {}).get("title", "")))
        first_text = ""
        full_head = []
        nonempty = 0
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                nonempty += 1
            if index < 3:
                full_head.append(text)
            if index == 0:
                first_text = text
        title = metadata_title if metadata_title and not bad_title(metadata_title) else ""
        if not title:
            title = first_text_title(first_text, path.stem)
        doi = extract_doi("\n".join(full_head))
        return doc.page_count, nonempty, title, doi
    finally:
        doc.close()


def classify(path: Path, title: str) -> tuple[str, str]:
    haystack = normalize_match(f"{path.name} {title}")
    dpf_terms = (
        "dense plasma focus",
        "plasma focus",
        "pf 1000",
        "pf1000",
        "z pinch",
        "zpinch",
        "neutron",
        "deuterium",
        "kubes",
        "sadowski",
        "auluck",
        "trunk",
        "kasperczuk",
        "symons",
        "lindemuth",
        "alexiou",
    )
    simulation_terms = (
        "simulation",
        "modeling",
        "modelling",
        "particle in cell",
        "pic",
        "monte carlo",
        "pulsed plasma thruster",
        "neutral particle transport",
        "plasma flow",
        "pauli",
    )
    materials_terms = ("tungsten", "retention", "elm like")
    math_terms = ("linear algebra", "precalculus", "mathematics for engineers")
    ai_terms = (
        "large language model",
        "cyber threat",
        "machine learning",
        "physics informed",
        "poisson boltzmann",
        "forced convection",
    )
    if any(term in haystack for term in dpf_terms):
        return "dpf_plasma_physics", "promote_to_kr_source_review"
    if any(term in haystack for term in materials_terms):
        return "plasma_materials_reference", "promote_to_kr_source_review"
    if any(term in haystack for term in simulation_terms):
        return "simulation_or_plasma_methods", "promote_to_kr_method_review"
    if any(term in haystack for term in math_terms):
        return "math_textbook_support", "promote_to_kr_method_review"
    if any(term in haystack for term in ai_terms):
        return "ai_ml_supporting_reference", "stage_for_review_not_physics_evidence"
    return "unclassified_review_needed", "stage_for_review_not_physics_evidence"


def kr_sha_index() -> dict[str, list[str]]:
    by_sha: dict[str, list[str]] = defaultdict(list)
    for json_path in (ROOT / "KnowledgeReference").glob("*.json"):
        try:
            payload = json.loads(json_path.read_text(errors="ignore"))
        except Exception:
            continue
        sha = str(payload.get("source_pdf_sha256", "")).lower()
        if sha:
            by_sha[sha].append(json_path.relative_to(ROOT).as_posix())
    return by_sha


def collect_records(apply: bool) -> dict[str, Any]:
    kr_by_sha = kr_sha_index()
    used_names: set[str] = set()
    records: list[InputRecord] = []
    seen_by_sha: dict[str, str] = {}
    if apply:
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for index, raw_path in enumerate(INPUT_PATHS, start=1):
        path = Path(raw_path)
        if not path.exists():
            records.append(
                InputRecord(index, path, False, "", 0, 0, 0, "", "", "missing", "not_staged", "", False, "", "", "file_missing")
            )
            continue
        try:
            sha = sha256_file(path)
            page_count, nonempty_pages, title, doi = pdf_metadata(path)
            subject_class, relevance = classify(path, title)
            already = ", ".join(kr_by_sha.get(sha.lower(), []))
            duplicate_of = seen_by_sha.get(sha, "")
            canonical = not duplicate_of
            destination = ""
            if canonical:
                filename = safe_filename(path, sha, used_names)
                destination_path = BATCH_DIR / filename
                destination = destination_path.relative_to(ROOT).as_posix()
                seen_by_sha[sha] = destination
                if apply:
                    shutil.copy2(path, destination_path)
            records.append(
                InputRecord(
                    input_index=index,
                    path=path,
                    exists=True,
                    sha256=sha,
                    size=path.stat().st_size,
                    page_count=page_count,
                    nonempty_pages=nonempty_pages,
                    title=title,
                    doi=doi,
                    subject_class=subject_class,
                    relevance=relevance,
                    destination=destination,
                    canonical_for_sha=canonical,
                    duplicate_of=duplicate_of,
                    already_in_kr=already,
                    error="",
                )
            )
        except Exception as exc:
            records.append(
                InputRecord(index, path, True, "", path.stat().st_size, 0, 0, "", "", "read_error", "not_staged", "", False, "", "", repr(exc))
            )

    existing = [record for record in records if record.exists and not record.error]
    missing = [record for record in records if not record.exists]
    failed = [record for record in records if record.error and record.exists]
    unique_sha = {record.sha256 for record in existing}
    copied = [
        record
        for record in existing
        if record.canonical_for_sha and record.destination and not record.already_in_kr
    ]
    staged_existing = [
        record
        for record in existing
        if record.canonical_for_sha and record.destination and record.already_in_kr
    ]
    duplicate_records = [record for record in existing if not record.canonical_for_sha]
    class_counts = defaultdict(int)
    relevance_counts = defaultdict(int)
    for record in existing:
        class_counts[record.subject_class] += 1
        relevance_counts[record.relevance] += 1

    result = {
        "date": date.today().isoformat(),
        "applied": apply,
        "batch_dir": BATCH_DIR.relative_to(ROOT).as_posix(),
        "input_count": len(INPUT_PATHS),
        "existing_input_count": len(existing),
        "missing_input_count": len(missing),
        "failed_input_count": len(failed),
        "unique_sha256_count": len(unique_sha),
        "staged_canonical_count": len([r for r in existing if r.canonical_for_sha]),
        "duplicate_input_count": len(duplicate_records),
        "already_in_kr_count": len([r for r in existing if r.already_in_kr]),
        "copied_new_or_review_needed_count": len(copied),
        "staged_existing_sha_count": len(staged_existing),
        "subject_class_counts": dict(sorted(class_counts.items())),
        "relevance_counts": dict(sorted(relevance_counts.items())),
        "records": [record.__dict__ | {"path": str(record.path)} for record in records],
        "guardrail": (
            "Staging copies local PDFs for source review only. Staged PDFs are not "
            "scientific evidence until promoted, reviewed, and mapped to specific "
            "KnowledgeReference target/digitization records."
        ),
    }
    return result


def write_reports(result: dict[str, Any], apply: bool) -> None:
    if not apply:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    REPORT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with AUDIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "title",
                "relevance",
                "subject_class",
                "sha256",
                "page_count",
                "doi",
                "destination",
                "duplicate_of",
                "already_in_kr",
                "error",
            ],
        )
        writer.writeheader()
        intake_prefix = INTAKE_DIR.relative_to(ROOT).as_posix() + "/"
        for record in result["records"]:
            if record["destination"]:
                rel = str(record["destination"]).removeprefix(intake_prefix)
            else:
                rel = ""
            writer.writerow(
                {
                    "path": rel,
                    "title": record["title"],
                    "relevance": record["relevance"],
                    "subject_class": record["subject_class"],
                    "sha256": record["sha256"],
                    "page_count": record["page_count"],
                    "doi": record["doi"],
                    "destination": record["destination"],
                    "duplicate_of": record["duplicate_of"],
                    "already_in_kr": record["already_in_kr"],
                    "error": record["error"],
                }
            )

    lines = [
        "# User PDF Intake - 2026-05-12",
        "",
        f"Generated: {result['date']}",
        "",
        result["guardrail"],
        "",
        "## Summary",
        "",
        f"- Input paths: {result['input_count']}",
        f"- Existing readable inputs: {result['existing_input_count']}",
        f"- Missing inputs: {result['missing_input_count']}",
        f"- Read failures: {result['failed_input_count']}",
        f"- Unique SHA-256 payloads: {result['unique_sha256_count']}",
        f"- Staged canonical PDFs: {result['staged_canonical_count']}",
        f"- Duplicate input paths: {result['duplicate_input_count']}",
        f"- Already represented in KR by SHA: {result['already_in_kr_count']}",
        "",
        "## Subject Classes",
        "",
        "| class | count |",
        "| --- | ---: |",
    ]
    for key, value in result["subject_class_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Records",
            "",
            "| input | title | class | pages | sha12 | destination | duplicate/already |",
            "| --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for record in result["records"]:
        duplicate_or_existing = record["duplicate_of"] or record["already_in_kr"] or record["error"]
        lines.append(
            "| {input} | {title} | `{cls}` | {pages} | {sha12} | `{dest}` | {dup} |".format(
                input=Path(record["path"]).name.replace("|", "\\|"),
                title=str(record["title"] or "").replace("|", "\\|"),
                cls=record["subject_class"],
                pages=record["page_count"],
                sha12=str(record["sha256"])[:12],
                dest=record["destination"],
                dup=str(duplicate_or_existing).replace("|", "\\|"),
            )
        )
    REPORT_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="copy unique PDFs and write reports")
    args = parser.parse_args()
    result = collect_records(apply=args.apply)
    write_reports(result, apply=args.apply)
    print(
        "inputs={input_count} existing={existing_input_count} unique={unique_sha256_count} "
        "staged={staged_canonical_count} duplicates={duplicate_input_count} "
        "missing={missing_input_count} failed={failed_input_count}".format(**result)
    )
    return 0 if result["missing_input_count"] == 0 and result["failed_input_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
