#!/usr/bin/env python3
"""Artifact linter for first-principles result JSON files.

Implements work package WP-N0 (evidence hygiene) from
``docs/FIRST_PRINCIPLES_EXTERNAL_TEAM_AUDIT_AND_NEXT_INSTRUCTIONS_2026_05_18.md``
and Codex audit findings A-1/A-2 from
``docs/FIRST_PRINCIPLES_CODEX_AGENT_AUDIT_AND_NEXT_INSTRUCTIONS_2026_05_18.md``.

The linter scans first-principles runtime artifacts and rejects any artifact
that carries a stale (pre-fix) schema or lacks required provenance. It exits
nonzero if ANY scanned first-principles artifact fails one of eight checks:

  C1  top-level ``conservation_telemetry.passed`` key present (deprecated;
      superseded by ``finite_state`` /
      ``energy_conservation_assessed: not_assessed_no_accepted_tolerance``)
  C2  no top-level ``artifact_generation_commit``
  C3  no top-level ``command_argv``
  C4  no ``telemetry_packets.power_port.stage0_packet_scaffolds``
  C5  no manifest ``candidate_evidence.deck_diff_packet`` for PF-1000/Akel runs
  C6  ``can_support_first_principles_acceptance: true`` found anywhere in the
      artifact (a first-principles acceptance claim that the audit forbids)
  C7  ``manifest.provenance_complete`` missing or false (Codex A-1: an
      artifact whose run manifest lacks complete source/command provenance
      cannot back a first-principles claim)
  C8  active artifact was not generated from current HEAD (Codex RC-5):
      top-level ``artifact_generation_commit``, nested
      ``manifest.git_commit``, and nested
      ``manifest.artifact_generation_commit`` must all equal the live HEAD
      SHA, and ``dirty_worktree`` must be exactly ``False``; skipped
      gracefully when git is unavailable

Scope policy (Codex A-2). Three dispositions, never a silent skip of a
PF-1000 engineering evidence surface:

  * archived artifacts (anything under ``results/archive_stale_pre_ssr*/``)
    and non-authority evidence surfaces (checkpoint/restart, reproducibility,
    split-continuation, numerical-family probes) are reported ``EXEMPT`` with
    an explicit status reason proving they cannot support first-principles
    acceptance. EXEMPT artifacts do not affect the exit code, but the reason
    is printed so the exemption is auditable rather than invisible.
  * genuinely unrelated JSON (calibration sweeps, inverse-parameter screens,
    cross-backend MHD evidence) is reported ``SKIP``.
  * everything that is an authority-scope first-principles artifact is
    fully checked against C1-C7.

Usage:
    python scripts/audit_first_principles_artifacts.py results/*.json
    python scripts/audit_first_principles_artifacts.py 'results/**/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
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
    "C7": "manifest.provenance_complete missing or false",
    "C8": "active artifact commit != HEAD or dirty_worktree is not False",
}

# C7 required manifest provenance fields (mirrors
# ``dpf.first_principles.manifest.REQUIRED_PROVENANCE_FIELDS``; kept in sync
# by the RC-7 drift test in tests/test_first_principles_artifact_linter.py).
C7_REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "command_argv",
    "git_commit",
    "source_truth_index_sha256",
    "source_packet_hashes",
    "input_deck_sha256",
    "artifact_schema_version",
    "artifact_generation_commit",
)

# Substrings that identify a PF-1000 / Akel-scope run from the deck name,
# deck preset, or cited source path. Check C5 only applies to these runs.
_PF1000_AKEL_MARKERS: tuple[str, ...] = ("pf1000", "pf-1000", "akel", "auluck")

# Codex A-2 scope policy ------------------------------------------------------
#
# Path fragment marking an artifact as quarantined. Anything under a
# ``results/archive_stale_pre_ssr*/`` directory is stale evidence that was
# deliberately removed from the active root and cannot support acceptance.
_ARCHIVE_PATH_FRAGMENT: str = "archive_stale_pre_ssr"

# Non-authority first-principles evidence surfaces. These tools emit real
# runtime probes but, by construction, only exercise numerical machinery
# (restart equivalence, run-to-run reproducibility, segment continuation,
# mesh/timestep families). None of them carry a candidate physics ledger or
# source-backed manifest, so none can ever back a first-principles claim.
# Mapping value is the audit-facing reason the surface is non-authority.
_NON_AUTHORITY_TOOLS: dict[str, str] = {
    "experimental-checkpoint-restart": (
        "checkpoint/restart probe: exercises restart equivalence only, "
        "carries no candidate physics ledger or source-backed manifest"
    ),
    "experimental-state-checkpoint": (
        "state-checkpoint probe: serializes runtime state only, "
        "carries no candidate physics ledger or source-backed manifest"
    ),
    "experimental-reproducibility": (
        "reproducibility probe: measures run-to-run determinism only, "
        "carries no candidate physics ledger or source-backed manifest"
    ),
    "experimental-split-continuation": (
        "split-continuation probe: exercises segment hand-off only, "
        "carries no candidate physics ledger or source-backed manifest"
    ),
    "experimental-numerical-family": (
        "numerical-family probe: sweeps mesh/timestep variants only, "
        "carries no candidate physics ledger or source-backed manifest"
    ),
}


# --- result containers -------------------------------------------------------


@dataclass
class ArtifactResult:
    """Lint result for a single artifact path."""

    path: Path
    is_first_principles: bool = False
    is_pf1000_akel: bool = False
    parse_error: str | None = None
    failed_checks: list[str] = field(default_factory=list)
    exempt_reason: str | None = None

    @property
    def status(self) -> str:
        if self.parse_error is not None:
            return "ERROR"
        if self.exempt_reason is not None:
            return "EXEMPT"
        if not self.is_first_principles:
            return "SKIP"
        return "FAIL" if self.failed_checks else "PASS"

    @property
    def counts_against_exit(self) -> bool:
        """A parse error or a failing first-principles artifact fails the run.

        An ``EXEMPT`` artifact never fails the run: it is, by an explicit
        and printed reason, structurally unable to back a first-principles
        claim, so there is nothing for the linter to enforce against it.
        """
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


def _exemption_reason(path: Path, doc: dict) -> str | None:
    """Return an audit reason if the artifact is exempt from C1-C7, else None.

    Codex A-2: an exempt artifact is one that *cannot* support a
    first-principles acceptance claim by construction. Two cases:

      * the artifact lives under a stale archive directory; or
      * it was produced by a non-authority evidence tool (checkpoint/restart,
        reproducibility, split-continuation, numerical-family).

    Returning a non-None reason routes the artifact to ``EXEMPT`` with that
    reason printed, so the exemption is auditable -- never a silent skip.
    """
    if _ARCHIVE_PATH_FRAGMENT in path.as_posix():
        return (
            "archived under results/archive_stale_pre_ssr*: quarantined "
            "stale evidence, removed from the active root, cannot support "
            "first-principles acceptance"
        )
    tool = doc.get("tool")
    if isinstance(tool, str):
        lowered = tool.lower()
        for marker, reason in _NON_AUTHORITY_TOOLS.items():
            if marker in lowered:
                return reason
    return None


# --- git HEAD resolution (run-scope) -----------------------------------------


def _resolve_head_commit() -> str | None:
    """Return the current git HEAD SHA (full 40-char), or None if unavailable.

    Runs once per linter invocation; callers cache the result and skip C8
    rather than crashing when git is absent or the working tree is not inside
    a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        print(
            "audit: WARNING -- git unavailable; C8 (commit-match gate) skipped.",
            file=sys.stderr,
        )
        return None


# --- the eight checks --------------------------------------------------------


def _check_artifact(path: Path, doc: dict, head_commit: str | None = None) -> ArtifactResult:
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
    manifest = doc.get("manifest")
    if result.is_pf1000_akel:
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

    # C7 (Codex A-1): independently verify run-manifest provenance without
    # trusting the self-reported ``provenance_complete`` boolean.  A stale or
    # lying manifest can carry ``provenance_complete: true`` while omitting
    # ``source_packet_hashes`` or other required fields -- this check
    # re-derives completeness from the raw manifest fields.  Required field
    # list is the module-level ``C7_REQUIRED_PROVENANCE_FIELDS`` constant,
    # kept in sync with ``dpf.first_principles.manifest.REQUIRED_PROVENANCE_FIELDS``
    # by the RC-7 drift test in tests/test_first_principles_artifact_linter.py.
    _c7_fail = False
    if not isinstance(manifest, dict) or manifest.get("provenance_complete") is not True:
        _c7_fail = True
    else:
        for _field in C7_REQUIRED_PROVENANCE_FIELDS:
            _val = manifest.get(_field)
            if _val is None or (isinstance(_val, (str, list, dict, tuple)) and len(_val) == 0):
                _c7_fail = True
                break
    if _c7_fail:
        result.failed_checks.append("C7")

    # C8 (Codex RC-5): active artifact must have been generated from the current
    # HEAD commit with a clean worktree.  Skipped (not failed) when git is
    # unavailable so the linter degrades safely in offline/CI environments.
    if head_commit is not None:
        _c8_fail = False
        # top-level commit field
        if doc.get("artifact_generation_commit") != head_commit:
            _c8_fail = True
        # nested manifest fields
        if isinstance(manifest, dict):
            if manifest.get("git_commit") != head_commit:
                _c8_fail = True
            if manifest.get("artifact_generation_commit") != head_commit:
                _c8_fail = True
        else:
            _c8_fail = True
        # dirty worktree check: must be exactly False (True or missing fails)
        if doc.get("dirty_worktree") is not False:
            _c8_fail = True
        if _c8_fail:
            result.failed_checks.append("C8")

    return result


def lint_artifact(path: Path, head_commit: str | None = None) -> ArtifactResult:
    """Load and lint a single artifact path.

    Disposition order (Codex A-2): a parse error is ERROR; an archived or
    non-authority artifact is EXEMPT with a printed reason; an unrelated JSON
    is SKIP; an authority-scope first-principles artifact is checked C1-C8.
    The EXEMPT branch is evaluated before the first-principles classification
    so a quarantined first-principles artifact is reported as exempt-with-
    reason rather than silently failing or being skipped.

    ``head_commit`` is the current git HEAD SHA resolved once per linter run
    by the caller.  When ``None`` (git unavailable), C8 is skipped.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return ArtifactResult(path=path, parse_error=str(exc))

    if not isinstance(doc, dict):
        return ArtifactResult(path=path, is_first_principles=False)

    reason = _exemption_reason(path, doc)
    if reason is not None:
        return ArtifactResult(
            path=path,
            is_first_principles=_is_first_principles_artifact(doc),
            exempt_reason=reason,
        )

    if not _is_first_principles_artifact(doc):
        return ArtifactResult(path=path, is_first_principles=False)

    return _check_artifact(path, doc, head_commit=head_commit)


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
        elif result.status == "EXEMPT":
            detail = f"exempt: {result.exempt_reason}"
        elif result.status == "SKIP":
            detail = "not a first-principles artifact"
        elif result.status == "FAIL":
            detail = ", ".join(
                f"{check} ({CHECK_DESCRIPTIONS[check]})"
                for check in result.failed_checks
            )
        else:
            detail = "all checks passed"
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

    head_commit = _resolve_head_commit()
    results = [lint_artifact(path, head_commit=head_commit) for path in paths]
    _print_table(results)

    first_principles = [r for r in results if r.is_first_principles]
    failures = [r for r in results if r.counts_against_exit]
    skipped = sum(1 for r in results if r.status == "SKIP")
    exempt = sum(1 for r in results if r.status == "EXEMPT")
    passed = sum(1 for r in results if r.status == "PASS")

    print()
    print(
        f"audit: {len(results)} file(s) scanned -- "
        f"{len(first_principles)} first-principles, {skipped} skipped, "
        f"{exempt} exempt, {passed} passed, {len(failures)} failed."
    )
    if failures:
        print("audit: FAIL -- stale or non-provenant first-principles artifacts found.")
        return 1
    print("audit: PASS -- all first-principles artifacts pass C1-C8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
