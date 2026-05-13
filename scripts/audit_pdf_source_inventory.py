"""Audit PDF-like source inventory across local DPF source pools.

This is an inventory utility, not a scientific acceptance tool. It reports
where PDF-like files exist and how many unique SHA-256 payloads are present.
The output is intended to explain intake scope before KR promotion.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DEFAULT_JSON = DOCS_DIR / "PDF_SOURCE_INVENTORY_2026_05_11.json"
DEFAULT_MD = DOCS_DIR / "PDF_SOURCE_INVENTORY_2026_05_11.md"
PDF_LIKE_SUFFIXES = (".pdf", ".pdf.crdownload")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    scope: str
    relpath: str
    sha256: str
    size: int


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pdf_like(path: Path) -> bool:
    return path.name.lower().endswith(PDF_LIKE_SUFFIXES)


def classify(path: Path) -> tuple[str, str]:
    try:
        rel = path.relative_to(ROOT)
        parts = rel.parts
        if parts and parts[0] == "KnowledgeReference":
            return "knowledge_reference_excluded", rel.as_posix()
        if parts[:2] == ("downloaded_books_papers", "Research Papers"):
            return "active_research_papers_intake", rel.as_posix()
        if parts and parts[0] == "downloaded_books_papers":
            return "downloaded_books_papers_other", rel.as_posix()
        if parts[:3] == ("archive_reference_OLD", "references", "papers"):
            return "archive_reference_old_papers", rel.as_posix()
        if parts and parts[0] == "archive_reference_OLD":
            return "archive_reference_old_other", rel.as_posix()
        if parts and parts[0] == "external":
            return "external_vendor_or_backend_docs", rel.as_posix()
        return "project_other", rel.as_posix()
    except ValueError:
        try:
            rel = path.relative_to(Path.home())
            return "downloads", rel.as_posix()
        except ValueError:
            return "outside_project", str(path)


def iter_project_pdfs() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        if "KnowledgeReference" in path.parts:
            continue
        if path.is_file() and is_pdf_like(path):
            paths.append(path)
    return paths


def iter_downloads_pdfs(max_depth: int) -> list[Path]:
    downloads = Path.home() / "Downloads"
    if not downloads.exists():
        return []
    paths: list[Path] = []
    for path in downloads.rglob("*"):
        try:
            depth = len(path.relative_to(downloads).parts)
        except ValueError:
            continue
        if depth > max_depth:
            continue
        if path.is_file() and is_pdf_like(path):
            paths.append(path)
    return paths


def collect(max_download_depth: int) -> list[SourceFile]:
    seen_paths = {path.resolve() for path in iter_project_pdfs()}
    all_paths = list(seen_paths)
    for path in iter_downloads_pdfs(max_download_depth):
        resolved = path.resolve()
        if resolved not in seen_paths:
            all_paths.append(resolved)
            seen_paths.add(resolved)

    records: list[SourceFile] = []
    for path in sorted(all_paths, key=lambda p: str(p).lower()):
        scope, relpath = classify(path)
        records.append(
            SourceFile(
                path=path,
                scope=scope,
                relpath=relpath,
                sha256=sha256_file(path),
                size=path.stat().st_size,
            )
        )
    return records


def summarize(records: list[SourceFile]) -> dict[str, object]:
    by_scope: dict[str, list[SourceFile]] = defaultdict(list)
    by_dir: dict[str, list[SourceFile]] = defaultdict(list)
    by_sha: dict[str, list[SourceFile]] = defaultdict(list)
    for record in records:
        by_scope[record.scope].append(record)
        by_dir[str(record.path.parent)].append(record)
        by_sha[record.sha256].append(record)

    scopes = []
    for scope, items in sorted(by_scope.items()):
        scopes.append(
            {
                "scope": scope,
                "file_count": len(items),
                "unique_sha256_count": len({item.sha256 for item in items}),
                "bytes": sum(item.size for item in items),
            }
        )

    top_directories = []
    for directory, items in sorted(
        by_dir.items(), key=lambda entry: (-len(entry[1]), entry[0])
    )[:80]:
        top_directories.append(
            {
                "directory": directory,
                "file_count": len(items),
                "unique_sha256_count": len({item.sha256 for item in items}),
            }
        )

    duplicates = [
        {
            "sha256": sha,
            "count": len(items),
            "paths": [item.relpath for item in sorted(items, key=lambda i: i.relpath)],
        }
        for sha, items in by_sha.items()
        if len(items) > 1
    ]
    duplicates.sort(key=lambda item: (-int(item["count"]), str(item["sha256"])))

    active = by_scope.get("active_research_papers_intake", [])
    project = [record for record in records if record.scope != "downloads"]
    downloads = by_scope.get("downloads", [])

    return {
        "date": date.today().isoformat(),
        "summary": {
            "total_pdf_like_files": len(records),
            "total_unique_sha256_payloads": len(by_sha),
            "project_pdf_like_files_excluding_knowledge_reference": len(project),
            "project_unique_sha256_payloads_excluding_knowledge_reference": len(
                {record.sha256 for record in project}
            ),
            "downloads_pdf_like_files_scanned": len(downloads),
            "downloads_unique_sha256_payloads": len({record.sha256 for record in downloads}),
            "active_research_papers_intake_files": len(active),
            "active_research_papers_intake_unique_sha256_payloads": len(
                {record.sha256 for record in active}
            ),
            "duplicate_sha256_groups": len(duplicates),
        },
        "scopes": scopes,
        "top_directories": top_directories,
        "duplicates": duplicates[:200],
        "records": [
            {
                "scope": record.scope,
                "path": record.relpath,
                "sha256": record.sha256,
                "size": record.size,
            }
            for record in records
        ],
    }


def write_reports(result: dict[str, object], json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = result["summary"]
    lines = [
        "# PDF Source Inventory",
        "",
        f"Generated: {result['date']}",
        "",
        "Source guardrail: this is a file inventory only. A PDF-like file is not "
        "scientific evidence until it is reviewed into `KnowledgeReference/`, "
        "hashed, and mapped to source/target records.",
        "",
        "## Summary",
        "",
        f"- Total scanned PDF-like files: {summary['total_pdf_like_files']}",
        f"- Total unique SHA-256 payloads: {summary['total_unique_sha256_payloads']}",
        "- Project PDF-like files excluding `KnowledgeReference/`: "
        f"{summary['project_pdf_like_files_excluding_knowledge_reference']}",
        "- Project unique SHA-256 payloads excluding `KnowledgeReference/`: "
        f"{summary['project_unique_sha256_payloads_excluding_knowledge_reference']}",
        f"- Downloads PDF-like files scanned: {summary['downloads_pdf_like_files_scanned']}",
        f"- Downloads unique SHA-256 payloads: {summary['downloads_unique_sha256_payloads']}",
        "- Active research-paper intake files: "
        f"{summary['active_research_papers_intake_files']}",
        "- Active research-paper intake unique SHA-256 payloads: "
        f"{summary['active_research_papers_intake_unique_sha256_payloads']}",
        f"- Duplicate SHA-256 groups across scanned scopes: {summary['duplicate_sha256_groups']}",
        "",
        "The previous 91-unique count was only the active "
        "`downloaded_books_papers/Research Papers` intake scope. The broader "
        "local source inventory is much larger and should be triaged before "
        "bulk KR promotion.",
        "",
        "## Scope Counts",
        "",
        "| scope | files | unique SHA-256 | bytes |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scope in result["scopes"]:
        lines.append(
            f"| {scope['scope']} | {scope['file_count']} | "
            f"{scope['unique_sha256_count']} | {scope['bytes']} |"
        )

    lines.extend(
        [
            "",
            "## Top Directories",
            "",
            "| directory | files | unique SHA-256 |",
            "| --- | ---: | ---: |",
        ]
    )
    for item in result["top_directories"][:40]:
        directory = str(item["directory"]).replace(str(ROOT), ".")
        lines.append(
            f"| `{directory}` | {item['file_count']} | {item['unique_sha256_count']} |"
        )

    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "- Keep the active research-paper intake as the reviewed promotion surface.",
            "- Triage `archive_reference_OLD/references/papers` before copying into active intake.",
            "- Do not bulk-promote vendor docs, generated plots, logos, or stale simulator artifacts.",
            "- Promote textbooks with chunked Markdown indexes so full-book context remains readable.",
        ]
    )
    md_path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--downloads-depth", type=int, default=2)
    args = parser.parse_args()

    result = summarize(collect(args.downloads_depth))
    write_reports(result, args.json, args.md)
    summary = result["summary"]
    print(
        "files={total_pdf_like_files} unique={total_unique_sha256_payloads} "
        "project_files={project_pdf_like_files_excluding_knowledge_reference} "
        "project_unique={project_unique_sha256_payloads_excluding_knowledge_reference} "
        "downloads_files={downloads_pdf_like_files_scanned} downloads_unique={downloads_unique_sha256_payloads} "
        "active_intake_unique={active_research_papers_intake_unique_sha256_payloads}".format(
            **summary
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
