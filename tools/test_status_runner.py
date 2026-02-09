#!/usr/bin/env python3
"""Pytest runner that writes live JSON status for the desktop dashboard.

Runs slow tests and writes a JSON status file after each test completes.
The HTML dashboard polls this file for live updates.

Usage:
    python tools/test_status_runner.py              # run all slow tests
    python tools/test_status_runner.py --all        # run ALL tests (slow + fast)
    python tools/test_status_runner.py --watch      # re-run on file changes
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Status file path — dashboard reads this
STATUS_FILE = Path(__file__).parent / "test_status.json"
PROJECT_ROOT = Path(__file__).parents[1]


class StatusReporter:
    """Tracks test results and writes JSON status after each test."""

    def __init__(self, total_collected: int = 0):
        self.results: list[dict] = []
        self.total = total_collected
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0
        self.running: str | None = None
        self.start_time = time.time()
        self._test_start: float = 0.0

    def to_dict(self) -> dict:
        elapsed = time.time() - self.start_time
        completed = self.passed + self.failed + self.skipped + self.errors
        return {
            "timestamp": time.time(),
            "elapsed_seconds": round(elapsed, 1),
            "total": self.total,
            "completed": completed,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "running": self.running,
            "progress_pct": round(100 * completed / max(self.total, 1), 1),
            "status": "running" if self.running else ("done" if completed >= self.total else "idle"),
            "results": self.results[-100:],  # keep last 100
        }

    def write(self):
        data = self.to_dict()
        # Atomic write: write to temp then rename
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(STATUS_FILE)


class LiveStatusPlugin:
    """Pytest plugin that updates the JSON status file in real-time."""

    def __init__(self):
        self.reporter = StatusReporter()

    def pytest_collection_modifyitems(self, items):
        self.reporter.total = len(items)
        self.reporter.write()

    def pytest_deselected(self, items):
        """Correct total when items are deselected by markers like -m slow."""
        self.reporter.total = max(self.reporter.total - len(items), 0)
        self.reporter.write()

    def pytest_runtest_logstart(self, nodeid, location):
        self.reporter.running = nodeid
        self.reporter._test_start = time.time()
        self.reporter.write()

    def pytest_runtest_logreport(self, report):
        if report.when != "call" and not (report.when == "setup" and report.skipped):
            return

        duration = time.time() - self.reporter._test_start
        entry = {
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "duration": round(duration, 2),
            "timestamp": time.time(),
        }

        if report.outcome == "passed":
            self.reporter.passed += 1
            entry["symbol"] = "\u2705"
        elif report.outcome == "failed":
            self.reporter.failed += 1
            entry["symbol"] = "\u274c"
            # Include short failure message
            if hasattr(report, "longreprtext"):
                lines = report.longreprtext.strip().split("\n")
                entry["message"] = lines[-1][:200] if lines else ""
        elif report.outcome == "skipped":
            self.reporter.skipped += 1
            entry["symbol"] = "\u23ed\ufe0f"

        self.reporter.results.append(entry)
        self.reporter.running = None
        self.reporter.write()

    def pytest_runtest_logfinish(self, nodeid, location):
        pass

    def pytest_sessionfinish(self, session, exitstatus):
        self.reporter.running = None
        self.reporter.write()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run tests with live status output")
    parser.add_argument("--all", action="store_true", help="Run ALL tests, not just slow")
    parser.add_argument("--watch", action="store_true", help="Re-run on completion (every 60s)")
    parser.add_argument("--fast", action="store_true", help="Run only non-slow tests")
    args = parser.parse_args()

    # Clear old status
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()

    # Build pytest args
    pytest_args = [
        str(PROJECT_ROOT / "tests"),
        "-v",
        "--tb=short",
        "-q",
    ]

    if args.fast:
        pytest_args.extend(["-m", "not slow"])
    elif not args.all:
        pytest_args.extend(["-m", "slow"])

    plugin = LiveStatusPlugin()

    while True:
        # Reset for new run
        plugin.reporter = StatusReporter()
        plugin.reporter.write()

        print(f"\n{'='*60}")
        print(f"Starting test run at {time.strftime('%H:%M:%S')}")
        print(f"Status file: {STATUS_FILE}")
        print(f"{'='*60}\n")

        exit_code = pytest.main(pytest_args, plugins=[plugin])

        summary = plugin.reporter.to_dict()
        print(f"\n{'='*60}")
        print(f"Done: {summary['passed']}P / {summary['failed']}F / "
              f"{summary['skipped']}S in {summary['elapsed_seconds']}s")
        print(f"{'='*60}")

        if not args.watch:
            break

        print(f"\nWaiting 60s before next run... (Ctrl+C to stop)")
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            break

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
