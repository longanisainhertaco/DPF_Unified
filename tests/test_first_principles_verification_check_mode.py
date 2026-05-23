from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXHAUSTION_SCRIPT = ROOT / "scripts" / "verify_first_principles_source_truth_exhaustion.py"
VETTING_SCRIPT = ROOT / "scripts" / "verify_first_principles_module_source_vetting.py"
FIXED_DATE = "2026_05_23"
MISSING_DATE = "1999_01_01"


def _docs_snapshot() -> dict[str, str]:
    """sha256 of every file currently in docs/."""
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in DOCS.rglob("*")
        if path.is_file()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# verify_first_principles_source_truth_exhaustion.py --check
# ---------------------------------------------------------------------------


def test_exhaustion_check_exits_zero_when_in_sync() -> None:
    result = _run(EXHAUSTION_SCRIPT, "--check", "--date", FIXED_DATE)
    assert result.returncode == 0, (
        f"--check exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_exhaustion_check_strict_exits_zero_when_in_sync() -> None:
    result = _run(EXHAUSTION_SCRIPT, "--check", "--strict", "--date", FIXED_DATE)
    assert result.returncode == 0, (
        f"--check --strict exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_exhaustion_check_writes_nothing() -> None:
    before = _docs_snapshot()
    result = _run(EXHAUSTION_SCRIPT, "--check", "--date", FIXED_DATE)
    after = _docs_snapshot()
    assert before == after, (
        f"--check wrote or removed docs/ files\nchanged: {set(after) ^ set(before)}\n"
        f"returncode={result.returncode}"
    )


def test_exhaustion_check_missing_date_exits_nonzero() -> None:
    result = _run(EXHAUSTION_SCRIPT, "--check", "--date", MISSING_DATE)
    assert result.returncode != 0, (
        f"--check with nonexistent date should exit nonzero, got 0\nstderr: {result.stderr}"
    )


def test_exhaustion_check_reports_missing_files_to_stderr() -> None:
    result = _run(EXHAUSTION_SCRIPT, "--check", "--date", MISSING_DATE)
    assert "MISSING" in result.stderr, (
        f"Expected 'MISSING' in stderr\nstderr: {result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# verify_first_principles_module_source_vetting.py --check
# ---------------------------------------------------------------------------


def test_vetting_check_exits_zero_when_in_sync() -> None:
    result = _run(VETTING_SCRIPT, "--check", "--date", FIXED_DATE)
    assert result.returncode == 0, (
        f"--check exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_vetting_check_strict_exits_zero_when_in_sync() -> None:
    result = _run(VETTING_SCRIPT, "--check", "--strict", "--date", FIXED_DATE)
    assert result.returncode == 0, (
        f"--check --strict exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_vetting_check_writes_nothing() -> None:
    before = _docs_snapshot()
    result = _run(VETTING_SCRIPT, "--check", "--date", FIXED_DATE)
    after = _docs_snapshot()
    assert before == after, (
        f"--check wrote or removed docs/ files\nchanged: {set(after) ^ set(before)}\n"
        f"returncode={result.returncode}"
    )


def test_vetting_check_missing_date_exits_nonzero() -> None:
    result = _run(VETTING_SCRIPT, "--check", "--date", MISSING_DATE)
    assert result.returncode != 0, (
        f"--check with nonexistent date should exit nonzero, got 0\nstderr: {result.stderr}"
    )


def test_vetting_check_reports_missing_files_to_stderr() -> None:
    result = _run(VETTING_SCRIPT, "--check", "--date", MISSING_DATE)
    assert "MISSING" in result.stderr, (
        f"Expected 'MISSING' in stderr\nstderr: {result.stderr!r}"
    )
