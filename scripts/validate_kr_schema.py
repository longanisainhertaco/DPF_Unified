#!/usr/bin/env python3
"""KnowledgeReference JSON schema validator.
Usage: python3 scripts/validate_kr_schema.py <file.json> [...]
"""
import json, sys
from pathlib import Path

REQUIRED_TOP = {"source_pdf", "page_count", "table_count", "toc", "sections", "pages"}
KNOWN_OPTIONAL = {
    "file_size_bytes","pdf_version","citation","abstract","key_equations",
    "figures","references","dpf_relevance","provenance","authors","doi",
    "journal","year","volume","affiliations","keywords","phases",
    "source_pdf_path","model_parameters","tables",
}
TOC_KEYS = {"level","title","start_page"}
SECTION_KEYS = {"level","title","start_page","end_page","text"}
PAGE_KEYS = {"page","text","tables"}


def validate(path: Path) -> tuple[bool, list[str], list[str]]:
    errors, warnings = [], []
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return False, [f"parse error: {e}"], []

    top = set(d)
    missing = REQUIRED_TOP - top
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")

    novel = top - REQUIRED_TOP - KNOWN_OPTIONAL
    if novel:
        warnings.append(f"SCHEMA DRIFT — unknown keys: {sorted(novel)}")

    for key, expected in [("source_pdf", str), ("page_count", int), ("table_count", int)]:
        if key in top and not isinstance(d[key], expected):
            errors.append(f"{key}: expected {expected.__name__}, got {type(d[key]).__name__}")

    for i, e in enumerate(d.get("toc", [])):
        m = TOC_KEYS - set(e)
        if m: errors.append(f"toc[{i}] missing: {sorted(m)}")

    for i, s in enumerate(d.get("sections", [])):
        m = SECTION_KEYS - set(s)
        if m: errors.append(f"sections[{i}] missing: {sorted(m)}")

    for i, p in enumerate(d.get("pages", [])):
        m = PAGE_KEYS - set(p)
        if m: errors.append(f"pages[{i}] missing: {sorted(m)}")

    return not errors, errors, warnings


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: validate_kr_schema.py <file.json> [...]"); sys.exit(1)
    any_fail = False
    for arg in sys.argv[1:]:
        path = Path(arg)
        ok, errors, warnings = validate(path)
        print(f"{'PASS' if ok else 'FAIL'}  {path.name}")
        for e in errors: print(f"  ERROR: {e}")
        for w in warnings: print(f"  WARN:  {w}")
        if not ok: any_fail = True
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
