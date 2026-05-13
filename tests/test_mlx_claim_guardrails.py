"""Fail-closed claim guards for MLX coupling methods."""

from pathlib import Path

from dpf.metal.mlx_coupling import (
    coupling_method_authority,
    evaluate_mhd_coupling_gate,
)


def test_mlx_coupling_methods_are_not_validation_evidence() -> None:
    authority = coupling_method_authority()

    assert set(authority) == {
        "density_weighted_lp",
        "voltage_flux",
        "poynting_voltage",
    }
    for record in authority.values():
        assert record["validation_status"] == "not_validation_evidence"
        assert record["can_support_scientific_claims"] is False
        assert "accepted" not in record["classification"]
        assert record["guardrails"]


def test_mlx_coupling_text_does_not_reintroduce_authority_claims() -> None:
    files = [
        Path("src/dpf/metal/mlx_coupling.py"),
        Path("src/dpf/metal/mlx_engine.py"),
        Path("tests/test_mlx_circuit_coupling.py"),
    ]
    banned_phrases = [
        "CORRECT method",
        "CORRECT first-principles",
        "from first principles",
        "proves that",
        "is the MHD physics real",
        "trust-gated",
    ]

    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for phrase in banned_phrases:
            if phrase in text:
                offenders.append(f"{path}: {phrase}")

    assert offenders == []


def test_mhd_coupling_gate_rejects_finite_positive_only() -> None:
    gate = evaluate_mhd_coupling_gate(
        phase="startup",
        lp_mhd=1.0e-9,
        lp_snowplow=1.0e-9,
        dlp_dt_mhd=0.0,
        resistance_mhd=0.0,
        allowed_phases={"rundown", "radial", "pinch"},
    )

    assert gate["checks"]["lp_finite"] is True
    assert gate["checks"]["lp_positive"] is True
    assert gate["checks"]["phase_allowed"] is False
    assert gate["eligible_for_engineering_blend"] is False
    assert gate["can_support_scientific_claims"] is False
    assert gate["validation_status"] == "not_validation_evidence"


def test_mhd_coupling_gate_can_be_engineering_eligible_but_not_scientific() -> None:
    gate = evaluate_mhd_coupling_gate(
        phase="rundown",
        lp_mhd=1.0e-9,
        lp_snowplow=1.5e-9,
        dlp_dt_mhd=0.0,
        resistance_mhd=1.0e-3,
        allowed_phases={"rundown", "radial", "pinch"},
    )

    assert gate["eligible_for_engineering_blend"] is True
    assert gate["failed_checks"] == []
    assert gate["scientific_gate_status"] == "blocked_missing_same_scope_validation_packet"
    assert gate["can_support_scientific_claims"] is False
