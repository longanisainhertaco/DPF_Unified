#!/usr/bin/env python3
"""DPF Memory Manager — persistent memory system for Claude Code sessions.

Provides search, status, checkpoint/resume, indexing, and validation
for the project's memory files.

Usage:
    python tools/memory_manager.py search <query>
    python tools/memory_manager.py status
    python tools/memory_manager.py checkpoint "description"
    python tools/memory_manager.py resume
    python tools/memory_manager.py index
    python tools/memory_manager.py validate
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-anthonyzamora-dpf-unified"
    / "memory"
)
SESSION_FILE = MEMORY_DIR / "session.md"
INDEX_FILE = MEMORY_DIR / "MEMORY.md"

MEMORY_LINE_LIMIT = 150
STALE_DAYS = 7


def cmd_search(query: str) -> int:
    """Search all memory files for a query string (case-insensitive)."""
    if not MEMORY_DIR.is_dir():
        print(f"ERROR: Memory directory not found: {MEMORY_DIR}", file=sys.stderr)
        return 1

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    found = False
    for md_file in sorted(MEMORY_DIR.glob("*.md")):
        matches = []
        for i, line in enumerate(md_file.read_text().splitlines(), 1):
            if pattern.search(line):
                matches.append((i, line.rstrip()))
        if matches:
            found = True
            print(f"\n--- {md_file.name} ---")
            for lineno, text in matches:
                print(f"  {lineno:4d}: {text}")

    if not found:
        print(f"No matches for '{query}' in {MEMORY_DIR}")
        return 1
    return 0


def cmd_status() -> int:
    """Show memory file inventory with sizes and health warnings."""
    if not MEMORY_DIR.is_dir():
        print(f"ERROR: Memory directory not found: {MEMORY_DIR}", file=sys.stderr)
        return 1

    files = sorted(MEMORY_DIR.glob("*.md"))
    if not files:
        print("No memory files found.")
        return 1

    total_bytes = 0
    total_lines = 0

    print(f"{'File':<30s} {'Lines':>6s} {'Size':>8s} {'Modified':<20s} {'Status'}")
    print("-" * 90)

    for f in files:
        text = f.read_text()
        lines = len(text.splitlines())
        size = f.stat().st_size
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        total_bytes += size
        total_lines += lines

        warnings = []
        if f.name == "MEMORY.md" and lines > MEMORY_LINE_LIMIT:
            warnings.append(f"OVER LIMIT ({lines}/{MEMORY_LINE_LIMIT})")
        if lines > 200:
            warnings.append("LARGE")

        status = ", ".join(warnings) if warnings else "OK"
        size_str = f"{size / 1024:.1f}K"
        print(f"{f.name:<30s} {lines:>6d} {size_str:>8s} {mtime:<20s} {status}")

    print("-" * 90)
    print(
        f"{'TOTAL':<30s} {total_lines:>6d} {total_bytes / 1024:.1f}K".rstrip()
    )

    # Session freshness check
    if SESSION_FILE.exists():
        age_days = (
            datetime.now().timestamp() - SESSION_FILE.stat().st_mtime
        ) / 86400
        if age_days > STALE_DAYS:
            print(f"\nWARN: session.md is {age_days:.0f} days old (stale)")
        else:
            print(f"\nSession checkpoint: {age_days:.1f} days old")
    else:
        print("\nNo session checkpoint found (session.md missing)")

    return 0


def cmd_checkpoint(description: str) -> int:
    """Create/overwrite session.md with a checkpoint template."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # Get git branch
    branch = "unknown"
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip() or "detached"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    template = f"""# Session Checkpoint

## Metadata
- **Saved**: {now}
- **Branch**: {branch}
- **Description**: {description}

## Objective
<!-- What was the session working on? (fill in) -->

## Progress
<!-- Checklist of completed and remaining steps -->
- [ ] Step 1

## Key Decisions
<!-- Important choices made during this session -->

## Files Modified
<!-- List of files changed with brief descriptions -->

## Blockers / Open Questions
<!-- Anything blocking progress -->

## Context for Next Session
<!-- Critical state, variable values, partial implementations the next session needs -->
"""

    SESSION_FILE.write_text(template)
    print(f"Session checkpoint saved: {SESSION_FILE}")
    print(f"  Description: {description}")
    print(f"  Branch: {branch}")
    print(f"  Time: {now}")

    # Update MEMORY.md session pointer
    if INDEX_FILE.exists():
        content = INDEX_FILE.read_text()
        # Update last checkpoint line
        content = re.sub(
            r"- \*\*Last checkpoint\*\*:.*",
            f"- **Last checkpoint**: {now} — {description}",
            content,
        )
        content = re.sub(
            r"- \*\*Resume available\*\*:.*",
            "- **Resume available**: yes",
            content,
        )
        INDEX_FILE.write_text(content)
        print("Updated MEMORY.md session pointer.")

    return 0


def cmd_resume() -> int:
    """Display the last session checkpoint."""
    if not SESSION_FILE.exists():
        print("No session checkpoint found.")
        print(f"Expected at: {SESSION_FILE}")
        print("Use '/session-save' or 'python tools/memory_manager.py checkpoint \"desc\"' to create one.")
        return 1

    print(SESSION_FILE.read_text())
    return 0


def cmd_index() -> int:
    """Generate a markdown table index of all memory files."""
    if not MEMORY_DIR.is_dir():
        print(f"ERROR: Memory directory not found: {MEMORY_DIR}", file=sys.stderr)
        return 1

    files = sorted(MEMORY_DIR.glob("*.md"))
    print(f"| {'File':<30s} | {'Title':<50s} | {'Lines':>5s} | {'Modified':<16s} |")
    print(f"|{'-' * 32}|{'-' * 52}|{'-' * 7}|{'-' * 18}|")

    for f in files:
        text = f.read_text()
        lines_list = text.splitlines()
        line_count = len(lines_list)
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        # Extract first H1 or H2 heading as title
        title = f.stem
        for line in lines_list:
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        print(f"| {f.name:<30s} | {title:<50s} | {line_count:>5d} | {mtime:<16s} |")

    return 0


def cmd_validate() -> int:
    """Check memory system health and report issues."""
    if not MEMORY_DIR.is_dir():
        print(f"ERROR: Memory directory not found: {MEMORY_DIR}", file=sys.stderr)
        return 1

    issues = []
    warnings = []

    # Check MEMORY.md line count
    if INDEX_FILE.exists():
        line_count = len(INDEX_FILE.read_text().splitlines())
        if line_count > MEMORY_LINE_LIMIT:
            issues.append(
                f"MEMORY.md is {line_count} lines (limit: {MEMORY_LINE_LIMIT}). "
                "Content is being truncated! Move detailed content to topic files."
            )
        elif line_count > MEMORY_LINE_LIMIT - 20:
            warnings.append(
                f"MEMORY.md is {line_count} lines (limit: {MEMORY_LINE_LIMIT}). "
                "Getting close to truncation threshold."
            )
    else:
        issues.append("MEMORY.md not found!")

    # Check session.md staleness
    if SESSION_FILE.exists():
        age_days = (
            datetime.now().timestamp() - SESSION_FILE.stat().st_mtime
        ) / 86400
        if age_days > STALE_DAYS:
            warnings.append(
                f"session.md is {age_days:.0f} days old. "
                "Consider updating with /session-save."
            )
    else:
        warnings.append("No session.md found. Use /session-save to create checkpoints.")

    # Check for TODO/FIXME in memory files
    for md_file in sorted(MEMORY_DIR.glob("*.md")):
        for i, line in enumerate(md_file.read_text().splitlines(), 1):
            if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", line):
                warnings.append(f"{md_file.name}:{i}: contains {line.strip()[:60]}")

    # Check for very large topic files
    for md_file in sorted(MEMORY_DIR.glob("*.md")):
        if md_file.name == "MEMORY.md":
            continue
        line_count = len(md_file.read_text().splitlines())
        if line_count > 200:
            warnings.append(
                f"{md_file.name} is {line_count} lines. Consider splitting."
            )

    # Report
    if issues:
        print("ISSUES (must fix):")
        for issue in issues:
            print(f"  [!] {issue}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  [~] {warning}")

    if not issues and not warnings:
        print("Memory system healthy. No issues found.")

    return 1 if issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DPF Memory Manager — persistent memory for Claude Code sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python tools/memory_manager.py search "back_emf"\n'
            "  python tools/memory_manager.py status\n"
            '  python tools/memory_manager.py checkpoint "Phase AA: C1 back-EMF fix"\n'
            "  python tools/memory_manager.py resume\n"
            "  python tools/memory_manager.py index\n"
            "  python tools/memory_manager.py validate\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search memory files for a query")
    search_p.add_argument("query", help="Search string (case-insensitive)")

    sub.add_parser("status", help="Show memory file inventory and health")

    cp_p = sub.add_parser("checkpoint", help="Save session checkpoint")
    cp_p.add_argument("description", help="Brief description of current state")

    sub.add_parser("resume", help="Display last session checkpoint")
    sub.add_parser("index", help="Generate memory file index table")
    sub.add_parser("validate", help="Check memory system health")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "search": lambda: cmd_search(args.query),
        "status": cmd_status,
        "checkpoint": lambda: cmd_checkpoint(args.description),
        "resume": cmd_resume,
        "index": cmd_index,
        "validate": cmd_validate,
    }

    return dispatch[args.command]()


if __name__ == "__main__":
    sys.exit(main())
