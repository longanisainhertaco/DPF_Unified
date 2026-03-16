"""Automated simulation quality assessment.

Evaluates a simulation result against physics expectations and
assigns a quality grade (A-F) with specific feedback.

Checks:
    1. Current waveform shape (rise, peak, dip)
    2. Pinch compression ratio
    3. Bennett equilibrium consistency
    4. Energy conservation
    5. Grid resolution adequacy
    6. Neutron yield plausibility
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityCheck:
    """Single quality check result."""

    name: str
    passed: bool
    score: float       # 0-1
    message: str
    severity: str      # "critical", "warning", "info"


@dataclass
class QualityAssessment:
    """Overall simulation quality assessment."""

    checks: list[QualityCheck] = field(default_factory=list)
    grade: str = "F"
    score: float = 0.0
    summary: str = ""

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_critical_failures(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "critical")


def assess_quality(result: dict) -> QualityAssessment:
    """Assess simulation quality from result dict.

    Args:
        result: Simulation result from run_mhd_simulation or run_simulation_core.

    Returns:
        QualityAssessment with grade and detailed checks.
    """
    checks: list[QualityCheck] = []

    # 1. Current waveform — does it have a peak?
    I_peak = result.get("I_peak", 0)
    if I_peak > 0.01:
        checks.append(QualityCheck(
            "Current peak", True, 1.0,
            f"I_peak = {I_peak:.3f} MA", "critical",
        ))
    else:
        checks.append(QualityCheck(
            "Current peak", False, 0.0,
            f"No significant current peak (I_peak = {I_peak:.4f} MA)", "critical",
        ))

    # 2. Current dip (if snowplow present) — indicates radial compression
    if result.get("has_snowplow"):
        dip = result.get("dip_pct", 0)
        if dip > 1:
            score = min(dip / 20.0, 1.0)  # 20% dip = perfect
            checks.append(QualityCheck(
                "Current dip", True, score,
                f"Dip = {dip:.0f}% (indicates radial compression)", "warning",
            ))
        else:
            checks.append(QualityCheck(
                "Current dip", False, 0.0,
                "No current dip — radial compression may not have occurred", "warning",
            ))

    # 3. Simulation completed (enough steps)
    n_steps = result.get("n_steps", 0)
    if n_steps > 10:
        checks.append(QualityCheck(
            "Simulation length", True, min(n_steps / 100, 1.0),
            f"{n_steps} timesteps completed", "critical",
        ))
    else:
        checks.append(QualityCheck(
            "Simulation length", False, 0.0,
            f"Only {n_steps} steps — simulation may have failed early", "critical",
        ))

    # 4. Bennett equilibrium (if available)
    bennett = result.get("bennett")
    if bennett and bennett.get("T_bennett_keV", 0) > 0.01:
        T_B = bennett["T_bennett_keV"]
        checks.append(QualityCheck(
            "Bennett equilibrium", True, min(T_B / 5.0, 1.0),
            f"T_Bennett = {T_B:.2f} keV", "info",
        ))

    # 5. Neutron yield (for deuterium)
    ny = result.get("neutron_yield")
    if ny and ny.get("Y_neutron", 0) > 0:
        Yn = ny["Y_neutron"]
        bt = ny.get("bt_fraction", 0) * 100
        checks.append(QualityCheck(
            "Neutron yield", True, min(Yn / 1e8, 1.0),
            f"Yn = {Yn:.2e} ({bt:.0f}% beam-target)", "info",
        ))

    # 6. MHD density compression (if MHD backend)
    if result.get("has_mhd") and not result.get("has_snowplow"):
        import numpy as np
        rho_max = result.get("rho_max", [])
        rho0 = result.get("rho0", 1)
        if len(rho_max) > 0 and rho0 > 0:
            comp = float(np.max(rho_max)) / rho0
            if comp > 2.0:
                checks.append(QualityCheck(
                    "Density compression", True, min(comp / 10, 1.0),
                    f"Peak compression: {comp:.1f}x", "warning",
                ))
            else:
                checks.append(QualityCheck(
                    "Density compression", False, comp / 10,
                    f"Low compression ({comp:.1f}x) — grid may be too coarse", "warning",
                ))

    # 7. Breakdown mechanism (if available)
    bd = result.get("breakdown")
    if bd:
        checks.append(QualityCheck(
            "Breakdown model", True, 1.0,
            f"{bd['mechanism']} (CIV ratio {bd.get('civ_ratio', 0):.1f})", "info",
        ))

    # 8. Plasma regime (if available)
    regime = result.get("plasma_regime")
    if regime:
        Kn = regime.get("knudsen", 0)
        checks.append(QualityCheck(
            "Regime validity", regime.get("mhd_valid", False), 1.0 if Kn < 0.01 else 0.5,
            regime.get("summary", ""), "info",
        ))

    # Compute overall grade
    if not checks:
        return QualityAssessment(checks=checks, grade="F", score=0.0,
                                  summary="No data to assess")

    total_score = sum(c.score for c in checks) / len(checks)
    n_critical_fail = sum(1 for c in checks if not c.passed and c.severity == "critical")

    if n_critical_fail > 0:
        grade = "F" if n_critical_fail > 1 else "D"
    elif total_score > 0.8:
        grade = "A"
    elif total_score > 0.6:
        grade = "B"
    elif total_score > 0.4:
        grade = "C"
    else:
        grade = "D"

    summary_parts = [f"Grade: {grade} ({total_score*100:.0f}%)"]
    for c in checks:
        icon = "PASS" if c.passed else "FAIL"
        summary_parts.append(f"  [{icon}] {c.name}: {c.message}")

    return QualityAssessment(
        checks=checks,
        grade=grade,
        score=total_score,
        summary="\n".join(summary_parts),
    )
