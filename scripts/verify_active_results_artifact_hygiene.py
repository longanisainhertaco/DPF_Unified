#!/usr/bin/env python3
"""Audit active result artifacts for stale same-scope / LLNL-like scope patterns.

This is a read-only gate for the results artifact policy.  It does not modify
any artifact and does not promote any source, target, or simulation claim.

Its job is to prove that no active (non-archive) result artifact under results/
still contains stale selected-scope or same-scope LLNL-like emission patterns
that contradict the current runtime contract.

Archive policy:
  Any path component that matches the glob ``archive_*`` is considered an
  explicitly archived stale artifact and is excluded from the active scan.
  Only results/**/*.json files whose full path contains no ``archive_*``
  component are subject to the hygiene gate.

Exit codes:
  0 — all active artifacts are clean (no forbidden patterns found)
  1 — one or more active artifacts contain forbidden patterns, or --strict and
      the clean flag is false
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "same_scope_3d_validation_packet",
    "llnl_like_180ka_axisymmetric_hybrid_pic",
)


def _is_archive_path(path: Path) -> bool:
    """Return True when any component of *path* starts with 'archive_'."""
    return any(part.startswith("archive_") for part in path.parts)


def scan_active_results(repo_root: Path) -> list[dict[str, Any]]:
    """Scan non-archive results/*.json and results/**/*.json for forbidden patterns.

    Returns a list of issue dicts, each with keys:
      - ``file``: str, path relative to repo_root
      - ``pattern``: str, the forbidden pattern found
      - ``lines``: list[int], 1-based line numbers where the pattern appears
    """
    results_dir = repo_root / "results"
    if not results_dir.is_dir():
        return []

    issues: list[dict[str, Any]] = []
    for json_path in sorted(results_dir.rglob("*.json")):
        if _is_archive_path(json_path.relative_to(repo_root)):
            continue
        try:
            lines = json_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for pattern in FORBIDDEN_PATTERNS:
            hit_lines = [
                line_no + 1
                for line_no, line in enumerate(lines)
                if pattern in line
            ]
            if hit_lines:
                issues.append({
                    "file": str(json_path.relative_to(repo_root)),
                    "pattern": pattern,
                    "lines": hit_lines,
                })
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit active results/ JSON artifacts for stale same-scope / "
            "LLNL-like scope patterns.  Excludes archive_* directories."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any active artifact contains a forbidden pattern.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Read-only verification mode: renders the report and fails if any "
            "forbidden pattern is found in a non-archive artifact.  Use in CI."
        ),
    )
    args = parser.parse_args()

    issues = scan_active_results(ROOT)
    clean = len(issues) == 0

    payload: dict[str, Any] = {
        "scope": "active_results_artifact_hygiene",
        "authority_policy": (
            "results/ JSON artifacts outside archive_* directories must not "
            "contain same_scope_3d_validation_packet or "
            "llnl_like_180ka_axisymmetric_hybrid_pic; "
            "stale artifacts are relocated (not rewritten) to archive_* dirs"
        ),
        "clean": clean,
        "active_hit_count": len(issues),
        "forbidden_patterns": list(FORBIDDEN_PATTERNS),
        "issues": issues,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))

    if (args.strict or args.check) and not clean:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
