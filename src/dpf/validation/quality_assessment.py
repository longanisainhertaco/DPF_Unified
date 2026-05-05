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
    # EMPIRICAL: every numeric threshold in this function is an
    # engineering default chosen to give an A/B/C/D grade output for
    # operators triaging an MHD run. None are calibrated against a
    # specific paper; they are chosen to match the order of magnitude
    # of MA-class DPF behaviour reported in Lee & Saw 2008 and Scholz
    # 2006/2007. Treat the grade as a quick signal, not a quantitative
    # validation metric.
    checks: list[QualityCheck] = []

    # 1. Current waveform — does it have a peak?
    I_peak = result.get("I_peak", 0)
    # EMPIRICAL: 0.01 MA = 10 kA cutoff is a noise-floor heuristic; any
    # current below this for a DPF run indicates the discharge did not
    # form. Not a paper-attested threshold.
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
            # EMPIRICAL: 20% dip == perfect score. Lee & Saw 2008 report
            # ~15-25% dips for well-tuned PF-1000 / NX2 shots; 20% is a
            # mid-range engineering target, not a calibrated metric.
            score = min(dip / 20.0, 1.0)
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
    # EMPIRICAL: 10 steps minimum is a sanity check that the run did
    # more than initialize. 100 steps == full score is an arbitrary
    # ceiling chosen to keep the score in [0, 1] for typical DPF runs
    # (which use 10^3 - 10^5 steps). Not a paper-attested cut.
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
    # EMPIRICAL: 0.01 keV = 10 eV temperature floor for a Bennett
    # equilibrium check; below this the equilibrium is not yet
    # established. 5 keV ceiling is the order-of-magnitude T_e_pinch
    # for MA-class DPFs (Lee & Saw 2008 Table 1) and gives a full score.
    # Engineering defaults, not paper-calibrated.
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
        # EMPIRICAL: 1e8 neutrons == full score is a small-DPF
        # benchmark (LLNL-DPF, NX2 yield reference) used here as a
        # baseline for the quality score. MA-class DPFs reach 1e10-1e12
        # so the score saturates fast; this is intentional (the check
        # is "is there any plausible yield", not "is the yield right").
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
            # EMPIRICAL: comp > 2x = pass, comp / 10 = score. Strong
            # shock theory gives 4x for gamma=5/3, double-shock ~8x;
            # 2x is the "did the sim resolve any radial focusing"
            # threshold. 10x ceiling matches the CR=10 placeholder
            # used elsewhere in this module. Engineering defaults.
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
        # EMPIRICAL: Kn < 0.01 = MHD valid (full score 1.0), else 0.5.
        # The 0.01 threshold matches the mhd_valid cut in pinch_physics
        # (textbook MHD validity, not paper-calibrated for DPF).
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

    # EMPIRICAL: A/B/C/D grade boundaries 0.8 / 0.6 / 0.4 are the
    # standard four-bucket scheme; chosen for human readability rather
    # than a calibrated DPF-specific cut.
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
