#!/usr/bin/env python3
"""Rich-based terminal dashboard for DPF slow test monitoring.

Displays live progress, pass/fail counts, and per-test timing in a beautiful
terminal UI powered by the Rich library.

Usage:
    # Run slow tests with live terminal dashboard:
    python tools/test_dashboard.py

    # Run ALL tests (slow + fast):
    python tools/test_dashboard.py --all

    # Run only non-slow tests:
    python tools/test_dashboard.py --fast

    # Watch mode — re-read test_status.json written by test_status_runner.py:
    python tools/test_dashboard.py --watch

    # Dry-run — show the dashboard UI with fake data (no tests executed):
    python tools/test_dashboard.py --dry-run

    # Specify custom test path:
    python tools/test_dashboard.py --path tests/test_phase_o_physics_accuracy.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_FILE = Path(__file__).resolve().parent / "test_status.json"

# ---------------------------------------------------------------------------
# Phase extraction from test node IDs
# ---------------------------------------------------------------------------


def extract_phase(nodeid: str) -> str:
    """Extract DPF phase from test node ID.

    Examples:
        tests/test_phase_o_physics_accuracy.py::... -> Phase O
        tests/test_phase_aa_zpinch.py::...          -> Phase AA
        tests/test_metal_production.py::...          -> Metal
        tests/test_verification_rlc.py::...          -> Verification
        tests/test_stress.py::...                    -> Stress
    """
    name = nodeid.split("::")[0].split("/")[-1].replace(".py", "")
    if name.startswith("test_phase_"):
        suffix = name[len("test_phase_"):]
        # Extract letter(s) before next underscore
        parts = suffix.split("_", 1)
        phase = parts[0].upper()
        return f"Phase {phase}"
    if "metal" in name:
        return "Metal"
    if "verification" in name or "phase17" in name:
        return "Verification"
    if "athena" in name:
        return "Athena"
    if "stress" in name:
        return "Stress"
    if "well" in name:
        return "WALRUS"
    return "Other"


# ---------------------------------------------------------------------------
# Rich dashboard
# ---------------------------------------------------------------------------


def run_rich_dashboard(pytest_args: list[str] | None = None, watch_mode: bool = False):
    """Run the Rich live terminal dashboard.

    Args:
        pytest_args: Args to pass to pytest. If None, enters watch mode.
        watch_mode: If True, just watch test_status.json instead of running pytest.
    """
    import threading

    import pytest
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # ── State ──
    results: list[dict] = []
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    running_test: str | None = None
    start_time = time.time()
    done = False

    def fmt_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.1f}s"
        m = int(secs // 60)
        s = secs % 60
        return f"{m}m {s:.0f}s"

    def fmt_clock(secs: float) -> str:
        m = int(secs // 60)
        s = int(secs % 60)
        return f"{m:02d}:{s:02d}"

    def make_layout() -> Layout:
        """Build the dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats", size=5),
            Layout(name="progress", size=3),
            Layout(name="running", size=3),
            Layout(name="results", ratio=1),
            Layout(name="footer", size=3),
        )
        return layout

    def render_header() -> Panel:
        elapsed = time.time() - start_time
        status_text = "[bold red]FAILURES[/]" if failed > 0 and done else (
            "[bold green]DONE[/]" if done else "[bold blue]RUNNING[/]"
        )
        grid = Table.grid(padding=1)
        grid.add_row(
            Text("DPF Test Dashboard", style="bold white"),
            Text(f"Elapsed: {fmt_clock(elapsed)}", style="dim"),
            status_text,
        )
        return Panel(grid, style="blue")

    def render_stats() -> Panel:
        tbl = Table(show_header=False, expand=True, box=None, padding=(0, 2))
        tbl.add_column("label", style="dim", width=10)
        tbl.add_column("value", justify="right", width=8)
        tbl.add_column("label2", style="dim", width=10)
        tbl.add_column("value2", justify="right", width=8)
        tbl.add_column("label3", style="dim", width=10)
        tbl.add_column("value3", justify="right", width=8)
        tbl.add_column("label4", style="dim", width=10)
        tbl.add_column("value4", justify="right", width=8)

        completed = passed + failed + skipped
        eta = "--:--"
        if completed > 0 and not done and total > 0:
            elapsed = time.time() - start_time
            avg = elapsed / completed
            remaining = (total - completed) * avg
            eta = fmt_clock(remaining)

        tbl.add_row(
            "PASSED", f"[bold green]{passed}[/]",
            "FAILED", f"[bold red]{failed}[/]",
            "SKIPPED", f"[bold yellow]{skipped}[/]",
            "TOTAL", f"[bold blue]{total}[/]",
        )
        tbl.add_row(
            "", "",
            "", "",
            "ETA", f"[dim]{eta}[/]",
            "DONE", f"[dim]{completed}/{total}[/]",
        )
        return Panel(tbl, title="Statistics", border_style="cyan")

    def render_progress() -> Panel:
        completed = passed + failed + skipped
        pct = (completed / max(total, 1)) * 100
        bar_width = 50
        fill_pass = int(bar_width * passed / max(total, 1))
        fill_fail = int(bar_width * failed / max(total, 1))
        fill_skip = int(bar_width * skipped / max(total, 1))
        fill_empty = bar_width - fill_pass - fill_fail - fill_skip

        bar = (
            f"[green]{'█' * fill_pass}[/]"
            f"[red]{'█' * fill_fail}[/]"
            f"[yellow]{'█' * fill_skip}[/]"
            f"[dim]{'░' * max(fill_empty, 0)}[/]"
        )
        return Panel(f"  {bar}  {pct:.1f}%", title="Progress", border_style="blue")

    def render_running() -> Panel:
        if running_test:
            short = running_test.split("::", 1)[-1] if "::" in running_test else running_test
            phase = extract_phase(running_test)
            return Panel(
                f"[bold blue]⟳[/] [cyan]{short}[/]  [dim]({phase})[/]",
                title="Currently Running",
                border_style="blue",
            )
        if done:
            return Panel("[dim]All tests completed[/]", title="Currently Running", border_style="green")
        return Panel("[dim]Waiting...[/]", title="Currently Running", border_style="dim")

    def render_results() -> Panel:
        tbl = Table(expand=True, show_lines=False, padding=(0, 1))
        tbl.add_column("", width=2)
        tbl.add_column("Test", ratio=3, no_wrap=True, overflow="ellipsis")
        tbl.add_column("Phase", width=14)
        tbl.add_column("Duration", width=10, justify="right")

        # Show most recent results (newest first), limit to ~20
        for r in reversed(results[-20:]):
            outcome = r.get("outcome", "?")
            nodeid = r.get("nodeid", "?")
            dur = r.get("duration", 0)
            phase = extract_phase(nodeid)
            short = nodeid.split("::", 1)[-1] if "::" in nodeid else nodeid

            if outcome == "passed":
                icon = "[green]✓[/]"
                name_style = "white"
            elif outcome == "failed":
                icon = "[red]✗[/]"
                name_style = "red"
            elif outcome == "skipped":
                icon = "[yellow]⊘[/]"
                name_style = "yellow"
            else:
                icon = "[dim]?[/]"
                name_style = "dim"

            tbl.add_row(icon, f"[{name_style}]{short}[/]", f"[dim]{phase}[/]", f"[dim]{fmt_duration(dur)}[/]")

        return Panel(tbl, title="Recent Results", border_style="cyan")

    def render_footer() -> Panel:
        completed = passed + failed + skipped
        avg = (time.time() - start_time) / max(completed, 1) if completed > 0 else 0
        mode = "watch" if watch_mode else ("slow" if pytest_args and "-m" in pytest_args else "all")
        return Panel(
            f"[dim]Mode: {mode} | Avg: {fmt_duration(avg)}/test | "
            f"Status file: {STATUS_FILE.name} | Press Ctrl+C to stop[/]",
            style="dim",
        )

    # ── Pytest plugin (inline) ──
    class RichDashboardPlugin:
        """Pytest plugin that feeds results to the Rich dashboard state."""

        def pytest_collection_modifyitems(self, items):  # noqa: N805
            nonlocal total
            total = len(items)

        def pytest_deselected(self, items):  # noqa: N805
            nonlocal total
            total = max(total - len(items), 0)

        def pytest_runtest_logstart(self, nodeid, location):  # noqa: N805
            nonlocal running_test
            running_test = nodeid

        def pytest_runtest_logreport(self, report):  # noqa: N805
            nonlocal passed, failed, skipped, running_test
            if report.when != "call" and not (report.when == "setup" and report.skipped):
                return

            entry = {
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "duration": round(report.duration, 2) if hasattr(report, "duration") else 0,
                "timestamp": time.time(),
            }

            if report.outcome == "passed":
                passed += 1
            elif report.outcome == "failed":
                failed += 1
                if hasattr(report, "longreprtext"):
                    lines = report.longreprtext.strip().split("\n")
                    entry["message"] = lines[-1][:200] if lines else ""
            elif report.outcome == "skipped":
                skipped += 1

            results.append(entry)
            running_test = None

            # Also write to status JSON for the HTML dashboard
            _write_status_json()

        def pytest_sessionfinish(self, session, exitstatus):  # noqa: N805
            nonlocal done, running_test
            running_test = None
            done = True
            _write_status_json()

    def _write_status_json():
        """Write current state to test_status.json for the HTML dashboard."""
        completed = passed + failed + skipped
        data = {
            "timestamp": time.time(),
            "elapsed_seconds": round(time.time() - start_time, 1),
            "total": total,
            "completed": completed,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": 0,
            "running": running_test,
            "progress_pct": round(100 * completed / max(total, 1), 1),
            "status": "done" if done else ("running" if running_test else "idle"),
            "results": results[-100:],
        }
        tmp = STATUS_FILE.with_suffix(".tmp")
        with contextlib.suppress(OSError):
            tmp.write_text(json.dumps(data, indent=2))
            tmp.rename(STATUS_FILE)

    # ── Main loop ──
    layout = make_layout()

    def refresh():
        layout["header"].update(render_header())
        layout["stats"].update(render_stats())
        layout["progress"].update(render_progress())
        layout["running"].update(render_running())
        layout["results"].update(render_results())
        layout["footer"].update(render_footer())

    # Run mode: execute pytest with our plugin
    console.print("[bold blue]DPF Test Dashboard[/] — starting test run...")
    console.print(f"[dim]pytest args: {' '.join(pytest_args or [])}[/]\n")

    plugin = RichDashboardPlugin()
    exit_code = [0]

    def run_pytest():
        exit_code[0] = pytest.main(pytest_args or [], plugins=[plugin])

    # Run pytest in a background thread so we can update the display
    thread = threading.Thread(target=run_pytest, daemon=True)

    with Live(layout, console=console, refresh_per_second=4, screen=True):
        thread.start()
        try:
            while thread.is_alive():
                refresh()
                time.sleep(0.25)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted![/]")

        # Final refresh
        refresh()
        time.sleep(0.5)

    # Print final summary
    console.print()
    elapsed = time.time() - start_time
    if failed > 0:
        console.print(f"[bold red]FAILED[/]: {passed} passed, {failed} failed, "
                       f"{skipped} skipped in {fmt_clock(elapsed)}")
    else:
        console.print(f"[bold green]PASSED[/]: {passed} passed, {skipped} skipped "
                       f"in {fmt_clock(elapsed)}")

    # Save summary JSON
    summary_file = STATUS_FILE.parent / "test_summary.json"
    summary = {
        "timestamp": time.time(),
        "elapsed_seconds": round(elapsed, 1),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }
    with contextlib.suppress(OSError):
        summary_file.write_text(json.dumps(summary, indent=2))
        console.print(f"[dim]Summary saved to {summary_file}[/]")

    return exit_code[0]


# ---------------------------------------------------------------------------
# Watch-mode (standalone, simpler implementation)
# ---------------------------------------------------------------------------


def run_watch_mode():
    """Simplified watch mode that reads test_status.json and displays Rich UI."""
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print("[bold blue]DPF Test Dashboard[/] — watch mode")
    console.print(f"[dim]Reading {STATUS_FILE}[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")

    def render(data: dict | None) -> Panel:
        if data is None:
            return Panel(
                "[dim]Waiting for test_status.json...[/]\n\n"
                "Start the runner:\n  python tools/test_status_runner.py",
                title="DPF Test Dashboard", border_style="blue",
            )

        status = data.get("status", "idle")
        elapsed = data.get("elapsed_seconds", 0)
        total = data.get("total", 0)
        completed = data.get("completed", 0)
        p = data.get("passed", 0)
        f = data.get("failed", 0)
        s = data.get("skipped", 0)
        running = data.get("running")
        result_list = data.get("results", [])

        # Status line
        if status == "done":
            if f > 0:
                status_str = "[bold red]DONE -- FAILURES[/]"
            else:
                status_str = "[bold green]DONE -- ALL PASSED[/]"
        elif status == "running":
            status_str = "[bold blue]RUNNING[/]"
        else:
            status_str = "[dim]IDLE[/]"

        m = int(elapsed // 60)
        sec = int(elapsed % 60)

        # Progress bar
        bar_width = 40
        fill = int(bar_width * completed / max(total, 1))
        pct = completed / max(total, 1) * 100
        bar = f"[green]{'█' * int(bar_width * p / max(total, 1))}[/]"
        bar += f"[red]{'█' * int(bar_width * f / max(total, 1))}[/]"
        bar += f"[yellow]{'█' * int(bar_width * s / max(total, 1))}[/]"
        bar += f"[dim]{'░' * max(bar_width - fill, 0)}[/]"

        # ETA
        if completed > 0 and status == "running":
            avg = elapsed / completed
            eta_secs = (total - completed) * avg
            eta_m = int(eta_secs // 60)
            eta_s = int(eta_secs % 60)
            eta = f"{eta_m:02d}:{eta_s:02d}"
        elif status == "done":
            eta = "00:00"
        else:
            eta = "--:--"

        lines = [
            f"  Status: {status_str}   Elapsed: [bold]{m:02d}:{sec:02d}[/]   ETA: [dim]{eta}[/]",
            "",
            f"  [green]Passed: {p}[/]   [red]Failed: {f}[/]   "
            f"[yellow]Skipped: {s}[/]   [blue]Total: {total}[/]",
            "",
            f"  {bar}  {pct:.1f}%",
            "",
        ]

        # Currently running
        if running:
            short = running.split("::", 1)[-1] if "::" in running else running
            phase = extract_phase(running)
            lines.append(f"  [bold blue]⟳[/] [cyan]{short}[/]  [dim]({phase})[/]")
        elif status == "done":
            lines.append("  [dim]All tests completed[/]")
        else:
            lines.append("  [dim]Waiting...[/]")

        lines.append("")

        # Recent results table
        tbl = Table(expand=True, show_lines=False, padding=(0, 1))
        tbl.add_column("", width=2)
        tbl.add_column("Test", ratio=3, no_wrap=True, overflow="ellipsis")
        tbl.add_column("Phase", width=14)
        tbl.add_column("Time", width=10, justify="right")

        for r in reversed(result_list[-15:]):
            outcome = r.get("outcome", "?")
            nodeid = r.get("nodeid", "?")
            dur = r.get("duration", 0)
            phase = extract_phase(nodeid)
            short = nodeid.split("::", 1)[-1] if "::" in nodeid else nodeid

            if outcome == "passed":
                icon, style = "[green]✓[/]", "white"
            elif outcome == "failed":
                icon, style = "[red]✗[/]", "red"
            elif outcome == "skipped":
                icon, style = "[yellow]⊘[/]", "yellow"
            else:
                icon, style = "[dim]?[/]", "dim"

            dur_str = f"{dur:.1f}s" if dur < 60 else f"{int(dur // 60)}m{int(dur % 60)}s"
            tbl.add_row(icon, f"[{style}]{short}[/]", f"[dim]{phase}[/]", f"[dim]{dur_str}[/]")

        content = "\n".join(lines)
        return Panel(f"{content}\n", title="DPF Test Dashboard", border_style="blue",
                     subtitle="[dim]Ctrl+C to stop[/]")

    with Live(render(None), console=console, refresh_per_second=2) as live:
        try:
            while True:
                data = None
                if STATUS_FILE.exists():
                    with contextlib.suppress(json.JSONDecodeError, OSError):
                        data = json.loads(STATUS_FILE.read_text())
                live.update(render(data))
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    console.print("\n[dim]Dashboard stopped.[/]")


# ---------------------------------------------------------------------------
# Dry-run mode (fake data for UI testing)
# ---------------------------------------------------------------------------


def run_dry_run():
    """Show the dashboard with fake data cycling through states."""
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print("[bold blue]DPF Test Dashboard[/] — dry run mode\n")

    fake_tests = [
        ("tests/test_phase_o_physics_accuracy.py::TestWENO5Z::test_weno5z_smooth_convergence", "passed", 12.3),
        ("tests/test_metal_production.py::TestMetalMHD::test_sod_shock_float32", "passed", 8.7),
        ("tests/test_phase_c_verification.py::TestBennett::test_bennett_equilibrium", "failed", 45.2),
        ("tests/test_verification_rlc.py::test_rlc_energy_conservation", "passed", 3.1),
        ("tests/test_phase_n_cross_backend.py::TestCrossBackend::test_sod_parity", "passed", 15.8),
        ("tests/test_verification_comprehensive.py::TestOT::test_orszag_tang_mass", "skipped", 0.01),
        ("tests/test_phase_p_accuracy.py::TestPythonWENOZ::test_weno_z_convergence", "passed", 22.4),
        ("tests/test_phase_y_crossval.py::TestCrossVal::test_metal_vs_python", "passed", 31.0),
        ("tests/test_phase_z_calibration_benchmark.py::TestCalib::test_fm_calibration", "passed", 55.7),
        ("tests/test_verification_walrus.py::TestWALRUS::test_surrogate_inference", "passed", 67.3),
    ]

    total = len(fake_tests) + 5  # pretend some are pending
    results: list[dict] = []
    passed = failed = skipped = 0
    start = time.time()

    def render_dry(running_name: str | None) -> Panel:
        elapsed = time.time() - start
        m, s = int(elapsed // 60), int(elapsed % 60)
        completed = passed + failed + skipped
        pct = completed / max(total, 1) * 100

        bar_w = 40
        bar = (f"[green]{'█' * int(bar_w * passed / max(total, 1))}[/]"
               f"[red]{'█' * int(bar_w * failed / max(total, 1))}[/]"
               f"[yellow]{'█' * int(bar_w * skipped / max(total, 1))}[/]"
               f"[dim]{'░' * max(bar_w - int(bar_w * completed / max(total, 1)), 0)}[/]")

        lines = [
            f"  Status: [bold blue]RUNNING[/]   Elapsed: [bold]{m:02d}:{s:02d}[/]",
            "",
            f"  [green]Passed: {passed}[/]   [red]Failed: {failed}[/]   "
            f"[yellow]Skipped: {skipped}[/]   [blue]Total: {total}[/]",
            "",
            f"  {bar}  {pct:.1f}%",
            "",
        ]

        if running_name:
            short = running_name.split("::", 1)[-1]
            phase = extract_phase(running_name)
            lines.append(f"  [bold blue]⟳[/] [cyan]{short}[/]  [dim]({phase})[/]")
        else:
            lines.append("  [dim]Waiting...[/]")

        lines.append("")

        tbl = Table(expand=True, show_lines=False, padding=(0, 1))
        tbl.add_column("", width=2)
        tbl.add_column("Test", ratio=3, no_wrap=True, overflow="ellipsis")
        tbl.add_column("Phase", width=14)
        tbl.add_column("Time", width=10, justify="right")

        for r in reversed(results[-12:]):
            outcome = r["outcome"]
            nodeid = r["nodeid"]
            dur = r["duration"]
            phase = extract_phase(nodeid)
            short = nodeid.split("::", 1)[-1]
            if outcome == "passed":
                icon, style = "[green]✓[/]", "white"
            elif outcome == "failed":
                icon, style = "[red]✗[/]", "red"
            else:
                icon, style = "[yellow]⊘[/]", "yellow"
            dur_str = f"{dur:.1f}s"
            tbl.add_row(icon, f"[{style}]{short}[/]", f"[dim]{phase}[/]", f"[dim]{dur_str}[/]")

        return Panel("\n".join(lines) + "\n", title="DPF Test Dashboard [DRY RUN]",
                     border_style="magenta", subtitle="[dim]Ctrl+C to stop[/]")

    with Live(render_dry(None), console=console, refresh_per_second=4) as live:
        try:
            for nodeid, outcome, duration in fake_tests:
                # Show as running
                live.update(render_dry(nodeid))
                time.sleep(1.0)

                # Record result
                results.append({"nodeid": nodeid, "outcome": outcome, "duration": duration})
                if outcome == "passed":
                    passed += 1
                elif outcome == "failed":
                    failed += 1
                elif outcome == "skipped":
                    skipped += 1

                live.update(render_dry(None))
                time.sleep(0.3)

            # Final state
            live.update(render_dry(None))
            time.sleep(2)
        except KeyboardInterrupt:
            pass

    console.print(f"\n[bold]Dry run complete[/]: {passed} passed, {failed} failed, {skipped} skipped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Rich terminal dashboard for DPF test monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/test_dashboard.py              # Run slow tests with live dashboard
  python tools/test_dashboard.py --all        # Run ALL tests
  python tools/test_dashboard.py --fast       # Run only non-slow tests
  python tools/test_dashboard.py --watch      # Watch test_status.json (run separately)
  python tools/test_dashboard.py --dry-run    # Demo the UI with fake data
        """,
    )
    parser.add_argument("--all", action="store_true", help="Run ALL tests, not just slow")
    parser.add_argument("--fast", action="store_true", help="Run only non-slow tests")
    parser.add_argument("--watch", action="store_true", help="Watch test_status.json (don't run pytest)")
    parser.add_argument("--dry-run", action="store_true", help="Demo UI with fake data")
    parser.add_argument("--path", type=str, default=None, help="Custom test path")
    parser.add_argument("-x", action="store_true", help="Stop on first failure")
    args = parser.parse_args()

    # Check rich is available
    if importlib.util.find_spec("rich") is None:
        print("ERROR: 'rich' is not installed. Install with:")
        print("  pip install rich")
        sys.exit(1)

    if args.dry_run:
        run_dry_run()
        return

    if args.watch:
        run_watch_mode()
        return

    # Build pytest args
    test_path = args.path or str(PROJECT_ROOT / "tests")
    pytest_args = [test_path, "-v", "--tb=short", "-q", "--no-header"]

    if args.fast:
        pytest_args.extend(["-m", "not slow"])
    elif not args.all:
        pytest_args.extend(["-m", "slow"])

    if args.x:
        pytest_args.append("-x")

    exit_code = run_rich_dashboard(pytest_args=pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
