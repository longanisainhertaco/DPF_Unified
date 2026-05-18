#!/usr/bin/env python3
"""Artifact linter for first-principles result JSON files.

Implements work package WP-N0 (evidence hygiene) from
``docs/FIRST_PRINCIPLES_EXTERNAL_TEAM_AUDIT_AND_NEXT_INSTRUCTIONS_2026_05_18.md``.

The linter scans first-principles runtime artifacts and rejects any artifact
that carries a stale (pre-fix) schema or lacks required provenance. It exits
nonzero if ANY scanned first-principles artifact fails one of six checks:

  C1  top-level ``conservation_telemetry.passed`` key present (deprecated;
      superseded by ``finite_state`` /
      ``energy_conservation_assessed: not_assessed_no_accepted_tolerance``)
  C2  no top-level ``artifact_generation_commit``
  C3  no top-level ``command_argv``
  C4  no ``telemetry_packets.power_port.stage0_packet_scaffolds``
  C5  no manifest ``candidate_evidence.deck_diff_packet`` for PF-1000/Akel runs
  C6  ``can_support_first_principles_acceptance: true`` found anywhere in the
      artifact (a first-principles acceptance claim that the audit forbids)

Non first-principles JSON files (calibration sweeps, inverse-parameter
screens, checkpoint ``.npz`` siblings, etc.) are reported as ``skipped`` and
do not affect the exit code.

Usage:
    python scripts/audit_first_principles_artifacts.py results/*.json
    python scripts/audit_first_principles_artifacts.py 'results/**/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- check identifiers -------------------------------------------------------

CHECK_DESCRIPTIONS: dict[str, str] = {
    "C1": "top-level conservation_telemetry.passed key present",
    "C2": "missing top-level artifact_generation_commit",
    "C3": "missing top-level command_argv",
    "C4": "missing telemetry_packets.power_port.stage0_packet_scaffolds",
    "C5": "missing manifest candidate_evidence.deck_diff_packet (PF-1000/Akel)",
    "C6": "can_support_first_principles_acceptance: true found anywhere",
}

# Substrings that identify a PF-1000 / Akel-scope run from the deck name,
# deck preset, or cited source path. Check C5 only applies to these runs.
_PF1000_AKEL_MARKERS: tuple[str, ...] = ("pf1000", "pf-1000", "akel", "auluck")


# --- result containers -------------------------------------------------------


@dataclass
class ArtifactResult:
    """Lint result for a single artifact path."""

    path: Path
    is_first_principles: bool = False
    is_pf1000_akel: bool = False
    parse_error: str | None = None
    failed_checks: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.parse_error is not None:
            return "ERROR"
        if not self.is_first_principles:
            return "SKIP"
        return "FAIL" if self.failed_checks else "PASS"

    @property
    def counts_against_exit(self) -> bool:
        """A parse error or a failing first-principles artifact fails the run."""
        return self.status in ("FAIL", "ERROR")


# --- artifact classification -------------------------------------------------


def _is_first_principles_artifact(doc: dict) -> bool:
    """Return True if the JSON document is a first-principles runtime artifact.

    First-principles artifacts are produced by the ``first-principles*``,
    ``experimental-limiter-proof``, ``experimental-whole-shot``, and
    ``experimental-state-checkpoint`` CLI tools. They are recognized by their
    ``tool`` string or by carrying conservation/telemetry runtime structures.
    """
    tool = doc.get("tool")
    if isinstance(tool, str):
        lowered = tool.lower()
        if any(
            marker in lowered
            for marker in (
                "first-principles",
                "experimental-limiter-proof",
                "experimental-whole-shot",
                "experimental-state-checkpoint",
            )
        ):
            return True
    return "conservation_telemetry" in doc or "telemetry_packets" in doc


def _is_pf1000_akel_run(doc: dict) -> bool:
    """Return True if the artifact is a PF-1000 / Akel-scope run."""
    deck = doc.get("deck")
    fragments: list[str] = []
    if isinstance(deck, dict):
        for key in ("name", "preset"):
            value = deck.get(key)
            if isinstance(value, str):
                fragments.append(value)
    source = doc.get("source")
    if isinstance(source, str):
        fragments.append(source)
    blob = " ".join(fragments).lower()
    return any(marker in blob for marker in _PF1000_AKEL_MARKERS)


def _contains_acceptance_true(node: object) -> bool:
    """Recursively search for ``can_support_first_principles_acceptance: true``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "can_support_first_principles_acceptance" and value is True:
                return True
            if _contains_acceptance_true(value):
                return True
    elif isinstance(node, list):
        for item in node:
            if _contains_acceptance_true(item):
                return True
    return False


# --- the six checks ----------------------------------------------------------


def _check_artifact(path: Path, doc: dict) -> ArtifactResult:
    """Apply the six WP-N0 checks to a parsed first-principles artifact."""
    result = ArtifactResult(path=path, is_first_principles=True)
    result.is_pf1000_akel = _is_pf1000_akel_run(doc)

    # C1: deprecated top-level conservation_telemetry.passed key.
    conservation = doc.get("conservation_telemetry")
    if isinstance(conservation, dict) and "passed" in conservation:
        result.failed_checks.append("C1")

    # C2: missing artifact_generation_commit provenance.
    if "artifact_generation_commit" not in doc:
        result.failed_checks.append("C2")

    # C3: missing command_argv provenance.
    if "command_argv" not in doc:
        result.failed_checks.append("C3")

    # C4: missing Stage-0 power-port packet scaffolds.
    telemetry_packets = doc.get("telemetry_packets")
    power_port = (
        telemetry_packets.get("power_port")
        if isinstance(telemetry_packets, dict)
        else None
    )
    if not (
        isinstance(power_port, dict) and "stage0_packet_scaffolds" in power_port
    ):
        result.failed_checks.append("C4")

    # C5: PF-1000/Akel runs must carry a manifest deck-diff packet.
    if result.is_pf1000_akel:
        manifest = doc.get("manifest")
        candidate_evidence = (
            manifest.get("candidate_evidence")
            if isinstance(manifest, dict)
            else None
        )
        if not (
            isinstance(candidate_evidence, dict)
            and "deck_diff_packet" in candidate_evidence
        ):
            result.failed_checks.append("C5")

    # C6: forbidden first-principles acceptance claim anywhere in the artifact.
    if _contains_acceptance_true(doc):
        result.failed_checks.append("C6")

    return result


def lint_artifact(path: Path) -> ArtifactResult:
    """Load and lint a single artifact path."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return ArtifactResult(path=path, parse_error=str(exc))

    if not isinstance(doc, dict) or not _is_first_principles_artifact(doc):
        return ArtifactResult(path=path, is_first_principles=False)

    return _check_artifact(path, doc)


# --- CLI ---------------------------------------------------------------------


def _expand_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns into a sorted, de-duplicated list of file paths."""
    seen: dict[str, Path] = {}
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for match in matches:
            candidate = Path(match)
            if candidate.is_file():
                seen[str(candidate.resolve())] = candidate
    return [seen[key] for key in sorted(seen)]


def _print_table(results: list[ArtifactResult]) -> None:
    """Print a per-artifact pass/fail table."""
    name_width = max((len(r.path.name) for r in results), default=8)
    name_width = max(name_width, len("ARTIFACT"))
    header = f"{'ARTIFACT':<{name_width}}  {'STATUS':<6}  DETAIL"
    print(header)
    print("-" * len(header))
    for result in results:
        if result.status == "ERROR":
            detail = f"parse error: {result.parse_error}"
        elif result.status == "SKIP":
            detail = "not a first-principles artifact"
        elif result.status == "FAIL":
            detail = ", ".join(
                f"{check} ({CHECK_DESCRIPTIONS[check]})"
                for check in result.failed_checks
            )
        else:
            detail = "all 6 checks passed"
        print(f"{result.path.name:<{name_width}}  {result.status:<6}  {detail}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code (0 ok, 1 lint failure, 2 misuse)."""
    parser = argparse.ArgumentParser(
        prog="audit_first_principles_artifacts.py",
        description=(
            "Lint first-principles result JSON artifacts for stale schema and "
            "missing provenance (WP-N0 evidence hygiene)."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="GLOB",
        help="Artifact path glob(s), e.g. 'results/*.json'.",
    )
    args = parser.parse_args(argv)

    paths = _expand_paths(args.paths)
    if not paths:
        print("audit: no files matched the given path glob(s).", file=sys.stderr)
        return 2

    results = [lint_artifact(path) for path in paths]
    _print_table(results)

    first_principles = [r for r in results if r.is_first_principles]
    failures = [r for r in results if r.counts_against_exit]
    skipped = sum(1 for r in results if r.status == "SKIP")
    passed = sum(1 for r in results if r.status == "PASS")

    print()
    print(
        f"audit: {len(results)} file(s) scanned -- "
        f"{len(first_principles)} first-principles, {skipped} skipped, "
        f"{passed} passed, {len(failures)} failed."
    )
    if failures:
        print("audit: FAIL -- stale or non-provenant first-principles artifacts found.")
        return 1
    print("audit: PASS -- all first-principles artifacts are current-schema.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
