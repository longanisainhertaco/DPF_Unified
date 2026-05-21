#!/usr/bin/env python3
"""Run repeatable Codex audit gates for dpf-unified.

Default behavior is one audit cycle. Pass ``--loop`` to repeat at a fixed
interval, or ``--cycles N`` for a bounded loop. Logs are written outside the git
worktree by default so the audit does not dirty the repository it is checking.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = Path("/private/tmp/dpf-unified-audit-logs")

# ---------------------------------------------------------------------------
# Sprint 9 WS9-0 — narrow PDF-symlink typechange exception
# See docs/SPRINT9_WS9_0_PDF_SYMLINK_DECISION_2026_05_20.md for rationale.
#
# EXCEPTION SCOPE (narrow, intentional):
#   Only ` T ` (typechange) lines whose path starts with one of these
#   well-known PDF reference directories are excused from git_status_clean.
#   All other status codes ( M, D, ??, A, R, C, U, X) and any ` T ` line
#   whose path falls outside these directories still fail the gate.
# ---------------------------------------------------------------------------
_PDF_SYMLINK_DIRS: tuple[str, ...] = (
    "downloaded_books_papers/",
    "tmp/pdfs/",
)
_PDF_SYMLINK_DIRS_REPR = ", ".join(f"`{d}`" for d in _PDF_SYMLINK_DIRS)

# The two-char git porcelain XY field for typechange is " T" (unstaged) or
# "T " (staged). We only excuse the *unstaged* form because the corpus files
# are never staged intentionally.
_TYPECHANGE_PREFIX = " T "


def _is_excused_pdf_typechange(line: str) -> bool:
    """Return True iff *line* is a ` T ` typechange inside a known PDF dir.

    The narrow exception: the XY code must be exactly ` T ` (space-T-space,
    unstaged typechange) and the path must begin with one of the known PDF
    reference directories. Everything else returns False.
    """
    if not line.startswith(_TYPECHANGE_PREFIX):
        return False
    # Strip the three-char XY+space prefix; unquote if git quoted the path.
    raw_path = line[len(_TYPECHANGE_PREFIX):]
    if raw_path.startswith('"') and raw_path.endswith('"'):
        raw_path = raw_path[1:-1]
    return any(raw_path.startswith(d) for d in _PDF_SYMLINK_DIRS)


def _classify_git_status_lines(
    dirty_lines: list[str],
) -> tuple[list[str], list[str]]:
    """Split *dirty_lines* into (excused, real_dirty) based on the PDF exception.

    Returns:
        excused    — lines matched by the narrow PDF-symlink exception.
        real_dirty — all other dirty lines; these still fail the gate.
    """
    excused: list[str] = []
    real_dirty: list[str] = []
    for line in dirty_lines:
        if _is_excused_pdf_typechange(line):
            excused.append(line)
        else:
            real_dirty.append(line)
    return excused, real_dirty


@dataclass(frozen=True)
class Gate:
    """One command-backed audit gate."""

    name: str
    command: tuple[str, ...]
    require_clean_status: bool = False


def _venv_python() -> str:
    candidate = ROOT / ".venv312" / "bin" / "python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def _ruff_command() -> tuple[str, ...]:
    candidate = ROOT / ".venv312" / "bin" / "ruff"
    if candidate.exists():
        return (str(candidate), "check", "src/", "tests/")
    return (_venv_python(), "-m", "ruff", "check", "src/", "tests/")


def _latest_source_truth_date() -> str:
    pattern = re.compile(r"FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_(\d{4}_\d{2}_\d{2})\.json$")
    candidates: list[str] = []
    for path in (ROOT / "docs").glob("FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_*.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        date_slug = match.group(1)
        module_path = ROOT / "docs" / f"FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_{date_slug}.json"
        if module_path.exists():
            candidates.append(date_slug)
    if not candidates:
        raise FileNotFoundError("no matching first-principles source-truth baseline date found")
    return sorted(candidates)[-1]


def _expand_tests(patterns: Sequence[str]) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = sorted((ROOT).glob(pattern)) if any(ch in pattern for ch in "*?[") else [ROOT / pattern]
        for match in matches:
            if match.exists() and match.is_file():
                rel = match.relative_to(ROOT).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    paths.append(rel)
    return tuple(paths)


def _build_gates(baseline_date: str, include_broad: bool) -> list[Gate]:
    py = _venv_python()
    focused_tests = (
        "tests/test_external_team_submission_package.py",
        "tests/test_first_principles_artifact_linter.py",
        "tests/test_first_principles_manifest.py",
        "tests/test_first_principles_segmented_whole_shot.py",
        "tests/test_srs_traceability_export.py",
        "tests/test_first_principles_verification_check_mode.py",
    )
    broad_tests = _expand_tests(
        (
            "tests/test_first_principles_*.py",
            "tests/test_hybrid_3d_*.py",
            "tests/test_cli_first_principles_3d.py",
        )
    )
    gates = [
        Gate("git_status_clean", ("git", "status", "--short", "--branch"), require_clean_status=True),
        Gate("git_head", ("git", "rev-parse", "HEAD")),
        Gate("git_diff_check", ("git", "diff", "--check")),
        Gate(
            "source_truth_exhaustion",
            (
                py,
                "scripts/verify_first_principles_source_truth_exhaustion.py",
                "--strict",
                "--check",
                "--date",
                baseline_date,
            ),
        ),
        Gate(
            "module_source_vetting",
            (
                py,
                "scripts/verify_first_principles_module_source_vetting.py",
                "--strict",
                "--check",
                "--date",
                baseline_date,
            ),
        ),
        Gate("artifact_linter_active", (py, "scripts/audit_first_principles_artifacts.py", "results/*.json")),
        Gate(
            "artifact_linter_recursive",
            (py, "scripts/audit_first_principles_artifacts.py", "results/**/*.json"),
        ),
        Gate("ruff_src_tests", _ruff_command()),
        Gate("focused_pytest", (py, "-m", "pytest", *focused_tests, "-q", "-rx")),
    ]
    if include_broad:
        gates.append(Gate("broad_first_principles_pytest", (py, "-m", "pytest", *broad_tests, "-q", "-rx")))
    return gates


def _run_gate(gate: Gate, cycle_dir: Path, timeout_s: int) -> dict[str, object]:
    stdout_path = cycle_dir / f"{gate.name}.stdout.txt"
    stderr_path = cycle_dir / f"{gate.name}.stderr.txt"
    started = datetime.now(UTC)
    try:
        proc = subprocess.run(
            list(gate.command),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = f"{stderr}\nTIMEOUT after {timeout_s} seconds\n"
        returncode = 124
        timed_out = True

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    finished = datetime.now(UTC)
    ok = returncode == 0
    note = ""
    if ok and gate.require_clean_status:
        dirty_lines = [line for line in stdout.splitlines() if line and not line.startswith("##")]
        if dirty_lines:
            excused, real_dirty = _classify_git_status_lines(dirty_lines)
            if real_dirty:
                ok = False
                note = "git status reported worktree changes"
            elif excused:
                # Gate passes; report excused churn explicitly so the audit is
                # not silently calling the tree clean. See:
                # docs/SPRINT9_WS9_0_PDF_SYMLINK_DECISION_2026_05_20.md
                note = (
                    f"APPROVED EXCEPTION: {len(excused)} PDF-symlink typechange(s) in "
                    f"known external-storage dirs excused (Sprint 9 WS9-0 decision). "
                    f"Dirs: {_PDF_SYMLINK_DIRS_REPR}"
                )
    return {
        "name": gate.name,
        "command": list(gate.command),
        "returncode": returncode,
        "ok": ok,
        "timed_out": timed_out,
        "note": note,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _write_markdown(cycle_dir: Path, summary: dict[str, object]) -> None:
    gates = summary["gates"]
    assert isinstance(gates, list)
    lines = [
        "# Codex Periodic Audit Cycle",
        "",
        f"- Started UTC: `{summary['started_utc']}`",
        f"- Finished UTC: `{summary['finished_utc']}`",
        f"- HEAD: `{summary.get('head', 'unknown')}`",
        f"- Baseline date: `{summary['baseline_date']}`",
        f"- Passed: `{summary['ok']}`",
        "",
        "| Gate | Result | Seconds | Note |",
        "| --- | --- | ---: | --- |",
    ]
    for gate in gates:
        assert isinstance(gate, dict)
        result = "PASS" if gate["ok"] else "FAIL"
        note = str(gate.get("note") or "")
        lines.append(f"| `{gate['name']}` | {result} | {gate['duration_s']:.2f} | {note} |")
    lines.append("")
    (cycle_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_cycle(log_root: Path, baseline_date: str, include_broad: bool, timeout_s: int) -> dict[str, object]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cycle_dir = log_root / stamp
    cycle_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC)
    gates: list[dict[str, object]] = []
    for gate in _build_gates(baseline_date, include_broad=include_broad):
        result = _run_gate(gate, cycle_dir, timeout_s=timeout_s)
        gates.append(result)
    finished = datetime.now(UTC)
    head = "unknown"
    for gate in gates:
        if gate["name"] == "git_head":
            stdout = Path(str(gate["stdout_path"])).read_text(encoding="utf-8").strip()
            if stdout:
                head = stdout
    summary: dict[str, object] = {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "head": head,
        "baseline_date": baseline_date,
        "include_broad": include_broad,
        "ok": all(bool(gate["ok"]) for gate in gates),
        "cycle_dir": str(cycle_dir),
        "gates": gates,
    }
    (cycle_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(cycle_dir, summary)
    latest = log_root / "latest.json"
    latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="repeat audit cycles until interrupted")
    parser.add_argument("--cycles", type=int, default=1, help="number of cycles; 0 means unlimited")
    parser.add_argument("--interval-minutes", type=float, default=30.0, help="delay between cycles")
    parser.add_argument("--baseline-date", default=None, help="source-truth baseline date slug, e.g. 2026_05_18")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="directory for audit logs")
    parser.add_argument("--skip-broad", action="store_true", help="skip the broad first-principles pytest slice")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="timeout for each gate command")
    args = parser.parse_args(argv)

    baseline_date = args.baseline_date or _latest_source_truth_date()
    interval_s = max(args.interval_minutes, 0.0) * 60.0
    total_cycles = 0 if args.loop and args.cycles == 1 else args.cycles
    cycle_index = 0
    while True:
        cycle_index += 1
        summary = run_cycle(
            args.log_root,
            baseline_date=baseline_date,
            include_broad=not args.skip_broad,
            timeout_s=args.timeout_seconds,
        )
        print(
            f"cycle {cycle_index}: {'PASS' if summary['ok'] else 'FAIL'} "
            f"head={summary['head']} log={summary['cycle_dir']}",
            flush=True,
        )
        if not args.loop and cycle_index >= args.cycles:
            return 0 if summary["ok"] else 1
        if total_cycles and cycle_index >= total_cycles:
            return 0 if summary["ok"] else 1
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
