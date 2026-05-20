#!/usr/bin/env python3
"""Generate a Codex evidence packet and optionally ask Claude Code to audit it.

This runner is intentionally conservative:

* it gathers local command evidence from this checkout;
* it writes immutable timestamped packets under ``docs/dual_agent_audits``;
* Claude is invoked only when ``--run-claude`` is set;
* Claude receives no edit tools by default, only a compact evidence packet;
* ``--continue-claude`` resumes the current/most-recent Claude Code session for
  this working directory.

The output is designed for the DPF first-principles workflow: Codex remains the
source-grounded auditor, Claude provides a second independent review, and no
physics acceptance state can change without the normal KR/source/test gates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "dual_agent_audits"
MAX_PROMPT_OUTPUT_CHARS = 2500
DEFAULT_TEST_COMMAND = (
    ".venv312/bin/python",
    "-m",
    "pytest",
    "tests/test_sprint5_target_extractions.py",
    "tests/test_first_principles_physics_acceptance_protocol.py",
    "tests/test_first_principles_v2_handoff_ledgers.py",
    "tests/test_external_team_submission_package.py",
    "-q",
)


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _run(cmd: tuple[str, ...], timeout_s: int = 120) -> dict[str, Any]:
    started = dt.datetime.now(dt.UTC)
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    finished = dt.datetime.now(dt.UTC)
    return {
        "cmd": list(cmd),
        "returncode": returncode,
        "ok": returncode == 0,
        "timed_out": timed_out,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "stdout": stdout,
        "stderr": stderr,
    }


def collect_evidence(*, skip_tests: bool) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "repo_root": str(REPO_ROOT),
        "commands": {},
        "notes": [
            "KnowledgeReference is the source of truth for physics claims.",
            "This packet is evidence for review, not runtime physics acceptance.",
            "A dirty worktree must be treated as a release/audit blocker unless "
            "the dirty paths are explicitly waived.",
        ],
    }

    commands: dict[str, tuple[str, ...]] = {
        "git_status": ("git", "status", "--short", "--branch"),
        "git_head": ("git", "rev-parse", "HEAD"),
        "git_ahead_count": (
            "git",
            "rev-list",
            "--count",
            "origin/codex/corpus..HEAD",
        ),
        "git_last_commits": ("git", "log", "--oneline", "-8"),
        "git_head_stat": ("git", "show", "--stat", "--oneline", "HEAD", "--"),
        "git_diff_stat": ("git", "diff", "--stat"),
        "latest_periodic_audit": (
            "sed",
            "-n",
            "1,140p",
            "/private/tmp/dpf-unified-audit-logs/latest.json",
        ),
    }

    for name, cmd in commands.items():
        evidence["commands"][name] = _run(cmd, timeout_s=30)

    if not skip_tests:
        evidence["commands"]["focused_tests"] = _run(DEFAULT_TEST_COMMAND, timeout_s=180)

    status = evidence["commands"]["git_status"]["stdout"]
    status_lines = [line for line in status.splitlines() if line.strip()]
    dirty_line_count = max(0, len(status_lines) - 1)
    evidence["derived"] = {
        "worktree_clean": dirty_line_count == 0,
        "dirty_line_count": dirty_line_count,
        "head": evidence["commands"]["git_head"]["stdout"].strip(),
        "ahead_count": evidence["commands"]["git_ahead_count"]["stdout"].strip(),
        "focused_tests_ok": (
            True
            if skip_tests
            else bool(evidence["commands"]["focused_tests"]["ok"])
        ),
    }
    return evidence


def _trim_for_prompt(value: str, limit: int = MAX_PROMPT_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    head_len = limit // 2
    tail_len = limit - head_len
    return (
        value[:head_len]
        + "\n...[truncated for Claude prompt; full output is in evidence JSON]...\n"
        + value[-tail_len:]
    )


def _compact_for_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "created_utc": evidence.get("created_utc"),
        "repo_root": evidence.get("repo_root"),
        "notes": evidence.get("notes", []),
        "derived": evidence.get("derived", {}),
        "commands": {},
    }
    for name, result in evidence.get("commands", {}).items():
        if not isinstance(result, dict):
            continue
        compact["commands"][name] = {
            "cmd": result.get("cmd"),
            "returncode": result.get("returncode"),
            "ok": result.get("ok"),
            "timed_out": result.get("timed_out"),
            "duration_s": result.get("duration_s"),
            "stdout": _trim_for_prompt(str(result.get("stdout", ""))),
            "stderr": _trim_for_prompt(str(result.get("stderr", ""))),
        }
    return compact


def build_claude_prompt(evidence: dict[str, Any]) -> str:
    compact = _compact_for_prompt(evidence)
    return (
        "You are the independent Claude Code reviewer for the dpf-unified "
        "first-principles DPF simulator effort.\n\n"
        "Rules:\n"
        "1. Treat KnowledgeReference as the only scientific source of truth.\n"
        "2. Do not promote any physics claim from this packet.\n"
        "3. Fail closed on dirty worktrees, missing source evidence, or tests "
        "that only assert copied constants.\n"
        "4. Focus on audit findings, release blockers, and the next concrete "
        "engineering tasks.\n"
        "5. Return Markdown with sections: Verdict, Confirmed, Findings, "
        "Next Steps, Questions For Codex.\n\n"
        "Evidence packet JSON follows:\n\n"
        "```json\n"
        f"{json.dumps(compact, indent=2, sort_keys=True)}\n"
        "```\n"
    )


def write_packet(
    output_dir: Path,
    stamp: str,
    evidence: dict[str, Any],
    prompt: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / f"{stamp}_codex_evidence.json"
    prompt_path = output_dir / f"{stamp}_claude_prompt.md"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    return {"evidence": evidence_path, "prompt": prompt_path}


def invoke_claude(
    prompt: str,
    *,
    continue_claude: bool,
    claude_tools: str,
    max_budget_usd: str,
    model: str | None,
    timeout_s: int,
) -> dict[str, Any]:
    claude = shutil.which("claude")
    if not claude:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": "claude executable not found on PATH",
        }

    cmd = [
        claude,
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
        "--tools",
        claude_tools,
        "--max-budget-usd",
        max_budget_usd,
    ]
    if continue_claude:
        cmd.append("--continue")
    if model:
        cmd.extend(["--model", model])

    result = _run_with_input(tuple(cmd), prompt, timeout_s=timeout_s)
    result["cmd"] = cmd
    try:
        parsed = json.loads(result.get("stdout", ""))
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        result["claude_json"] = parsed
        subtype = str(parsed.get("subtype", ""))
        is_error = bool(parsed.get("is_error", False))
        empty_result = subtype == "success" and not str(parsed.get("result", "")).strip()
        if subtype.startswith("error") or is_error:
            result["ok"] = False
            result["claude_error_subtype"] = subtype or "is_error_true"
        elif empty_result:
            result["ok"] = False
            result["claude_error_subtype"] = "empty_result"
    return result


def _run_with_input(cmd: tuple[str, ...], prompt: str, timeout_s: int) -> dict[str, Any]:
    started = dt.datetime.now(dt.UTC)
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    finished = dt.datetime.now(dt.UTC)
    return {
        "returncode": returncode,
        "ok": returncode == 0,
        "timed_out": timed_out,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "stdout": stdout,
        "stderr": stderr,
    }


def write_claude_response(
    output_dir: Path,
    stamp: str,
    response: dict[str, Any],
) -> dict[str, Path]:
    raw_path = output_dir / f"{stamp}_claude_response.json"
    md_path = output_dir / f"{stamp}_claude_response.md"
    raw_path.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
    content = response.get("stdout", "")
    try:
        parsed = json.loads(content)
        subtype = str(parsed.get("subtype", ""))
        if subtype.startswith("error") or parsed.get("is_error"):
            content = (
                "Claude CLI did not return a usable review.\n\n"
                f"- subtype: `{subtype or 'unknown'}`\n"
                f"- stop_reason: `{parsed.get('stop_reason', 'unknown')}`\n"
                f"- total_cost_usd: `{parsed.get('total_cost_usd', 'unknown')}`\n"
            )
        elif subtype == "success" and not str(parsed.get("result", "")).strip():
            content = (
                "Claude CLI returned success but no review content.\n\n"
                "- subtype: `success`\n"
                "- usable_review: `false`\n"
                f"- total_cost_usd: `{parsed.get('total_cost_usd', 'unknown')}`\n"
            )
        else:
            content = parsed.get("result") or parsed.get("text") or content
    except json.JSONDecodeError:
        pass
    md_path.write_text(str(content).strip() + "\n", encoding="utf-8")
    return {"claude_raw": raw_path, "claude_markdown": md_path}


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    stamp = _utc_stamp()
    output_dir = Path(args.output_dir)
    evidence = collect_evidence(skip_tests=args.skip_tests)
    prompt = build_claude_prompt(evidence)
    paths = write_packet(output_dir, stamp, evidence, prompt)
    result: dict[str, Any] = {
        "stamp": stamp,
        "paths": {key: str(value) for key, value in paths.items()},
        "head": evidence["derived"]["head"],
        "worktree_clean": evidence["derived"]["worktree_clean"],
        "focused_tests_ok": evidence["derived"]["focused_tests_ok"],
    }

    if args.run_claude:
        response = invoke_claude(
            prompt,
            continue_claude=args.continue_claude,
            claude_tools=args.claude_tools,
            max_budget_usd=args.max_budget_usd,
            model=args.model,
            timeout_s=args.claude_timeout_s,
        )
        response_paths = write_claude_response(output_dir, stamp, response)
        result["claude_ok"] = response.get("ok", False)
        result["paths"].update({key: str(value) for key, value in response_paths.items()})
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--run-claude", action="store_true")
    parser.add_argument(
        "--claude-tools",
        default="",
        help=(
            "Claude Code tools allowlist. Default disables tools. For a "
            "read-only current-Claude review, use: Read,Grep,Glob,mcp__qmd__get"
        ),
    )
    parser.add_argument(
        "--continue-claude",
        action="store_true",
        help="Resume the most recent Claude Code conversation in this directory.",
    )
    parser.add_argument("--max-budget-usd", default="1.00")
    parser.add_argument("--model", default=None)
    parser.add_argument("--claude-timeout-s", type=int, default=600)
    parser.add_argument(
        "--repeat-minutes",
        type=float,
        default=0.0,
        help="Run repeatedly every N minutes. Default runs once.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations when --repeat-minutes is set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    iterations = max(1, args.iterations)
    summaries: list[dict[str, Any]] = []
    for index in range(iterations):
        summary = run_once(args)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.repeat_minutes > 0 and index < iterations - 1:
            time.sleep(args.repeat_minutes * 60)
    return 0 if all(item.get("focused_tests_ok") for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
