"""Structured verification report for DPF V&V results.

Collects per-test results with literature references and quantitative
tolerances, then produces a summary table suitable for publication or
CI review.

Usage with pytest (session-scoped fixture)::

    @pytest.fixture(scope="session")
    def vv_report():
        report = VerificationReport()
        yield report
        report.print_summary()

    def test_saha(vv_report):
        Z = saha_ionization_fraction(1e5, 1e22)
        vv_report.record("Saha Ionization", "H at 100 kK", Z, 1.0, 0.01,
                         "Saha equation, Griem 1997")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Single verification measurement."""

    module: str
    test_name: str
    status: str  # "PASS" or "FAIL"
    computed: float
    expected: float
    tolerance: float
    reference: str  # Literature citation
    notes: str = ""


@dataclass
class VerificationReport:
    """Accumulates verification results and produces summary output."""

    results: list[VerificationResult] = field(default_factory=list)

    # ── Recording ───────────────────────────────────────────

    def record(
        self,
        module: str,
        test_name: str,
        computed: float,
        expected: float,
        tolerance: float,
        reference: str,
        *,
        notes: str = "",
    ) -> VerificationResult:
        """Record a verification measurement.

        Returns the result (also stored internally).
        """
        if tolerance > 0:
            rel_err = abs(computed - expected) / max(abs(expected), 1e-300)
            passed = rel_err <= tolerance
        else:
            passed = True

        status = "PASS" if passed else "FAIL"
        result = VerificationResult(
            module=module,
            test_name=test_name,
            status=status,
            computed=computed,
            expected=expected,
            tolerance=tolerance,
            reference=reference,
            notes=notes,
        )
        self.results.append(result)
        return result

    # ── Summary ─────────────────────────────────────────────

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    def print_summary(self) -> None:
        """Print a human-readable verification summary."""
        sep = "=" * 72
        print(f"\n{sep}")
        print("DPF Physics Verification Report")
        print(sep)

        current_module = ""
        for r in self.results:
            if r.module != current_module:
                current_module = r.module
                print(f"\n  Module: {current_module}")
            tag = "\033[92mPASS\033[0m" if r.status == "PASS" else "\033[91mFAIL\033[0m"
            print(
                f"    [{tag}] {r.test_name}: "
                f"computed={r.computed:.4e}, expected={r.expected:.4e}, "
                f"tol={r.tolerance:.2e}"
            )
            if r.notes:
                print(f"           {r.notes}")

        print(f"\n{sep}")
        print(f"  Total: {len(self.results)}  |  Pass: {self.n_pass}  |  Fail: {self.n_fail}")
        print(sep)

    # ── Export ──────────────────────────────────────────────

    def to_json(self, filepath: str | Path) -> None:
        """Export results as JSON."""
        data: dict[str, Any] = {
            "n_total": len(self.results),
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "results": [asdict(r) for r in self.results],
        }
        Path(filepath).write_text(json.dumps(data, indent=2))

    def to_markdown(self, filepath: str | Path) -> None:
        """Export results as a Markdown table."""
        lines = [
            "# DPF Physics Verification Report\n",
            "| Module | Test | Status | Computed | Expected | Tol | Reference |",
            "|--------|------|--------|----------|----------|-----|-----------|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.module} | {r.test_name} | {r.status} "
                f"| {r.computed:.4e} | {r.expected:.4e} "
                f"| {r.tolerance:.2e} | {r.reference} |"
            )
        lines.append(
            f"\n**Total**: {len(self.results)} | "
            f"**Pass**: {self.n_pass} | **Fail**: {self.n_fail}\n"
        )
        Path(filepath).write_text("\n".join(lines))
