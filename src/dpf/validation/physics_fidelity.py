"""Physics-fidelity audit for high-fidelity DPF claims.

The audit is conservative by design. It does not promote a simulation to
validation; it records which physics effects still block high-fidelity claims
under the local KnowledgeReference-only rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_KR_SOURCE_BASIS = {
    "ordinary_mhd_limits": (
        "KnowledgeReference/unlimited-release-printed-september-2009-"
        "alegra-hedp-simulations-of-the-dense-plasma-focus.md"
    ),
    "mhd_kinetic_transition": (
        "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
        "dense-plasma-focus-z-pinch-5.md"
    ),
    "review_kinetic_3d_limits": (
        "KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-"
        "for-diverse-applications.md"
    ),
    "kr_doped_mhd_requirements": "KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md",
}


_REQUIRED_EFFECTS = {
    "tabulated_eos_and_conductivity": {
        "source_key": "ordinary_mhd_limits",
        "source_lines": "332-362",
        "blocks_claims": (
            "circuit_waveform_prediction",
            "phase_dynamics_prediction",
            "spatial_mhd_prediction",
            "high_z_radiation_prediction",
            "p_b11_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "High-fidelity HEDP DPF modeling needs EOS and advanced transport/"
            "conductivity closures appropriate to the material state."
        ),
    },
    "ionization_and_charge_state_kinetics": {
        "source_key": "ordinary_mhd_limits",
        "source_lines": "332-362",
        "blocks_claims": (
            "circuit_waveform_prediction",
            "phase_dynamics_prediction",
            "spatial_mhd_prediction",
            "high_z_radiation_prediction",
            "p_b11_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "Ionization, dissociation, and charge-state evolution must be "
            "modeled or bounded for breakdown, sheath, radiation, and high-Z work."
        ),
    },
    "two_temperature_energy_partition": {
        "source_key": "ordinary_mhd_limits",
        "source_lines": "332-362",
        "blocks_claims": (
            "spatial_mhd_prediction",
            "neutron_prediction",
            "high_z_radiation_prediction",
            "p_b11_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "Electron/ion temperature separation and relaxation are required "
            "when one-temperature MHD is outside the claimed regime."
        ),
    },
    "radiation_transport_opacity": {
        "source_key": "kr_doped_mhd_requirements",
        "source_lines": "184-190",
        "blocks_claims": (
            "high_z_radiation_prediction",
            "p_b11_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "High-Z and radiating DPF claims need opacities and radiation "
            "transport, not only local reduced cooling fits."
        ),
    },
    "material_ablation_impurity_mixing": {
        "source_key": "kr_doped_mhd_requirements",
        "source_lines": "488-517",
        "blocks_claims": (
            "spatial_mhd_prediction",
            "high_z_radiation_prediction",
            "p_b11_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "Electrode material and dopant effects need material mixing and "
            "validated impurity/radiation coupling."
        ),
    },
    "hall_flr_kinetic_or_pic_effects": {
        "source_key": "mhd_kinetic_transition",
        "source_lines": "174-215",
        "blocks_claims": (
            "neutron_prediction",
            "p_b11_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "Finite-Larmor-radius, mean-free-path, Hall, and kinetic/PIC "
            "effects must be handled when late pinch or neutron mechanisms "
            "leave ordinary MHD validity."
        ),
    },
    "three_dimensional_instabilities": {
        "source_key": "review_kinetic_3d_limits",
        "source_lines": "184-190",
        "blocks_claims": (
            "phase_dynamics_prediction",
            "spatial_mhd_prediction",
            "neutron_prediction",
            "p_b11_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "Instability, beam formation, and non-axisymmetric pinch behavior "
            "must be modeled or bounded for final-pinch predictive claims."
        ),
    },
    "flashover_sheath_initiation": {
        "source_key": "review_kinetic_3d_limits",
        "source_lines": "184-190",
        "blocks_claims": (
            "circuit_waveform_prediction",
            "phase_dynamics_prediction",
        ),
        "requirement": (
            "Breakdown/flashover and sheath initiation need validated treatment "
            "for startup-sensitive predictions."
        ),
    },
    "restrike_and_anomalous_resistance": {
        "source_key": "ordinary_mhd_limits",
        "source_lines": "287-326",
        "blocks_claims": (
            "circuit_waveform_prediction",
            "phase_dynamics_prediction",
            "neutron_prediction",
            "late_pinch_prediction",
        ),
        "requirement": (
            "Post-pinch disruption, restrike, and anomalous-resistance behavior "
            "must be validated before full-discharge predictive claims."
        ),
    },
    "beam_generation_and_target_coupling": {
        "source_key": "mhd_kinetic_transition",
        "source_lines": "405-448",
        "blocks_claims": (
            "neutron_prediction",
            "p_b11_prediction",
        ),
        "requirement": (
            "Neutron prediction needs validated beam generation and beam-target "
            "coupling, not only scalar yield formulas."
        ),
    },
}


_CLAIM_LABELS = {
    "circuit_waveform_prediction": "circuit waveform prediction",
    "phase_dynamics_prediction": "snowplow/phase dynamics prediction",
    "spatial_mhd_prediction": "spatial MHD state prediction",
    "neutron_prediction": "neutron yield/mechanism prediction",
    "high_z_radiation_prediction": "high-Z radiation prediction",
    "p_b11_prediction": "p-B11 DPF prediction",
    "late_pinch_prediction": "late-pinch/final-pinch prediction",
}


def _active_modules(active_modules: Sequence[object] | None) -> set[str]:
    return {str(module).strip().lower() for module in (active_modules or [])}


def _has_module(active: set[str], *needles: str) -> bool:
    return any(any(needle in module for needle in needles) for module in active)


def _effect_record(
    effect: str,
    *,
    status: str,
    implemented: bool,
    verified: bool = False,
    validated: bool,
    notes: str,
    evidence_keys: Sequence[str] = (),
) -> dict[str, object]:
    meta = _REQUIRED_EFFECTS[effect]
    source = _KR_SOURCE_BASIS[str(meta["source_key"])]
    blocks_claims = list(meta.get("blocks_claims", ()))
    if status == "bounded_out":
        fidelity_status = "bounded_out"
    elif validated:
        fidelity_status = "validated"
    elif verified:
        fidelity_status = "verified"
    elif implemented and any(
        marker in status
        for marker in ("empirical", "diagnostic", "estimate")
    ):
        fidelity_status = "empirical"
    elif implemented:
        fidelity_status = "implemented"
    else:
        fidelity_status = "absent"
    return {
        "status": status,
        "fidelity_status": fidelity_status,
        "implemented": implemented,
        "verified": verified,
        "validated": validated,
        "source": source,
        "source_lines": meta["source_lines"],
        "requirement": meta["requirement"],
        "blocks_claims": blocks_claims,
        "evidence_keys": list(evidence_keys),
        "notes": notes,
    }


def physics_effect_validation_evidence(
    effect: str,
    *,
    validation_scope: str,
    source: str | None = None,
    source_lines: str | None = None,
    implemented: bool = True,
    verified: bool = True,
    bounded_out: bool = False,
    notes: str = "",
) -> dict[str, object]:
    """Build line-referenced evidence for one high-fidelity physics effect."""
    effect_key = str(effect).strip().lower()
    known_effect = effect_key in _REQUIRED_EFFECTS
    if known_effect:
        meta = _REQUIRED_EFFECTS[effect_key]
        default_source = _KR_SOURCE_BASIS[str(meta["source_key"])]
        default_lines = str(meta["source_lines"])
    else:
        default_source = ""
        default_lines = ""
    source_value = source or default_source
    line_value = source_lines or default_lines
    passed = (
        known_effect
        and bool(validation_scope)
        and str(source_value).startswith("KnowledgeReference/")
        and bool(line_value)
        and (implemented or bounded_out)
        and (verified or bounded_out)
    )
    return {
        "passed": passed,
        "validation_tier": "high_fidelity_physics",
        "model_role": "physics_effect_validation",
        "effect": effect_key,
        "implemented": bool(implemented),
        "verified": bool(verified),
        "bounded_out": bool(bounded_out),
        "validation_scope": validation_scope,
        "source": source_value,
        "source_lines": line_value,
        "details": {
            "known_effect": known_effect,
            "notes": notes,
        },
        "validity_notes": {
            "claim_scope": (
                "This evidence applies to one high-fidelity physics effect "
                "within the stated validation scope; it does not validate other "
                "effects or observables."
            ),
        },
    }


def _valid_effect_evidence(
    evidence: object,
    effect: str,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "physics_effect_validation":
        return None
    if evidence.get("validation_tier") != "high_fidelity_physics":
        return None
    if str(evidence.get("effect", "")).strip().lower() != effect:
        return None
    if not str(evidence.get("source", "")).startswith("KnowledgeReference/"):
        return None
    if not evidence.get("validation_scope"):
        return None
    if evidence.get("implemented") is not True and evidence.get("bounded_out") is not True:
        return None
    if evidence.get("verified", True) is not True and evidence.get("bounded_out") is not True:
        return None
    return evidence


def _validated_effects(
    result: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], str]]:
    found: dict[str, tuple[Mapping[str, object], str]] = {}

    for container_key in ("physics_effect_validation", "physics_effect_validations"):
        container = result.get(container_key)
        if isinstance(container, Mapping):
            for effect, candidate in container.items():
                effect_key = str(effect).strip().lower()
                evidence = _valid_effect_evidence(candidate, effect_key)
                if evidence is not None:
                    found[effect_key] = (evidence, container_key)
        elif isinstance(container, Sequence) and not isinstance(
            container, (str, bytes, bytearray)
        ):
            for candidate in container:
                if not isinstance(candidate, Mapping):
                    continue
                effect_key = str(candidate.get("effect", "")).strip().lower()
                evidence = _valid_effect_evidence(candidate, effect_key)
                if evidence is not None:
                    found[effect_key] = (evidence, container_key)

    for effect in _REQUIRED_EFFECTS:
        key = f"{effect}_validation"
        evidence = _valid_effect_evidence(result.get(key), effect)
        if evidence is not None:
            found[effect] = (evidence, key)

    return found


def _claim_blocker_matrix(
    effects: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    matrix: dict[str, dict[str, object]] = {
        claim: {
            "claim": claim,
            "label": label,
            "blocked": False,
            "blocking_effects": [],
            "blocking_statuses": {},
        }
        for claim, label in _CLAIM_LABELS.items()
    }
    for effect, record in effects.items():
        if record.get("validated") is True or record.get("fidelity_status") == "bounded_out":
            continue
        for claim in record.get("blocks_claims", []):
            claim_key = str(claim)
            if claim_key not in matrix:
                continue
            matrix[claim_key]["blocked"] = True
            blocking_effects = matrix[claim_key]["blocking_effects"]
            if isinstance(blocking_effects, list):
                blocking_effects.append(effect)
            blocking_statuses = matrix[claim_key]["blocking_statuses"]
            if isinstance(blocking_statuses, dict):
                blocking_statuses[effect] = record.get("fidelity_status", "absent")
    return matrix


def physics_fidelity_evidence_from_result(
    result: Mapping[str, object],
    *,
    active_modules: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build a conservative physics-fidelity evidence record for a run."""
    active = _active_modules(active_modules)
    if not active:
        reproducibility = result.get("reproducibility", {})
        if isinstance(reproducibility, Mapping):
            active = _active_modules(
                reproducibility.get("advanced_physics", [])  # type: ignore[arg-type]
            )

    effects: dict[str, dict[str, object]] = {}

    effects["tabulated_eos_and_conductivity"] = _effect_record(
        "tabulated_eos_and_conductivity",
        status="reduced_or_absent",
        implemented=False,
        validated=False,
        notes=(
            "The run does not provide KR-validated tabulated EOS/opacities/"
            "conductivity evidence. Ideal/Saha-style local models remain "
            "reduced closures for high-fidelity claims."
        ),
    )

    cr_enabled = _has_module(active, "cr ionization")
    effects["ionization_and_charge_state_kinetics"] = _effect_record(
        "ionization_and_charge_state_kinetics",
        status="implemented_not_validated" if cr_enabled else "reduced_or_absent",
        implemented=cr_enabled,
        validated=False,
        notes=(
            "CR ionization was active, but no KR validation evidence for charge-state "
            "evolution is attached."
            if cr_enabled
            else "No run-level KR-validated ionization/charge-state evidence is attached."
        ),
    )

    two_t = bool(
        result.get("two_temperature")
        or result.get("has_two_temperature")
        or result.get("electron_temperature")
        or result.get("ion_temperature")
    )
    effects["two_temperature_energy_partition"] = _effect_record(
        "two_temperature_energy_partition",
        status="implemented_not_validated" if two_t else "reduced_or_absent",
        implemented=two_t,
        validated=False,
        notes=(
            "Electron/ion temperature fields are present, but no KR validation "
            "evidence for two-temperature energy exchange is attached."
            if two_t
            else "No run-level two-temperature validation evidence is attached."
        ),
    )

    fld_enabled = _has_module(active, "fld radiation")
    radiation_diag = "radiation_regime" in result or "line_radiation_metadata" in result
    effects["radiation_transport_opacity"] = _effect_record(
        "radiation_transport_opacity",
        status=(
            "implemented_not_validated"
            if fld_enabled else
            "diagnostic_or_empirical" if radiation_diag else
            "reduced_or_absent"
        ),
        implemented=fld_enabled or radiation_diag,
        validated=False,
        notes=(
            "Radiation transport or radiation diagnostics are present, but no "
            "KR validation evidence for opacity/multigroup transport is attached."
            if fld_enabled or radiation_diag
            else "No run-level KR-validated radiation-transport/opacity evidence is attached."
        ),
    )

    ablation_enabled = _has_module(active, "ablation")
    effects["material_ablation_impurity_mixing"] = _effect_record(
        "material_ablation_impurity_mixing",
        status="empirical_not_validated" if ablation_enabled else "reduced_or_absent",
        implemented=ablation_enabled,
        validated=False,
        notes=(
            "Ablation was active, but no KR validation evidence for material "
            "mixing or impurity transport is attached."
            if ablation_enabled
            else "No validated material ablation/impurity mixing evidence is attached."
        ),
    )

    regime = result.get("plasma_regime", {})
    kinetic_needed = isinstance(regime, Mapping) and regime.get("kinetic_needed") is True
    hall_like = _has_module(active, "hall", "nernst") or bool(result.get("beam_tracker"))
    effects["hall_flr_kinetic_or_pic_effects"] = _effect_record(
        "hall_flr_kinetic_or_pic_effects",
        status=(
            "required_unvalidated"
            if kinetic_needed else
            "implemented_not_validated" if hall_like else
            "reduced_or_absent"
        ),
        implemented=hall_like,
        validated=False,
        notes=(
            "The regime classifier flags kinetic physics as needed; no validated "
            "kinetic/PIC/FLR closure is attached."
            if kinetic_needed
            else "Hall/Nernst or particle diagnostics are present but not validated "
            "as a late-pinch kinetic closure."
            if hall_like
            else "No validated Hall/FLR/kinetic/PIC evidence is attached."
        ),
    )

    instability_diag = any(
        key in result
        for key in (
            "filamentation",
            "plasmoids",
            "m0_instability",
            "m1_instability",
            "instability_diagnostics",
        )
    )
    effects["three_dimensional_instabilities"] = _effect_record(
        "three_dimensional_instabilities",
        status="diagnostic_only" if instability_diag else "reduced_or_absent",
        implemented=instability_diag,
        validated=False,
        notes=(
            "Instability diagnostics are present, but no KR validation evidence "
            "for 3D instability evolution is attached."
            if instability_diag
            else "No validated 3D instability evidence is attached."
        ),
    )

    sheath_or_breakdown = _has_module(active, "sheath") or "breakdown" in result
    effects["flashover_sheath_initiation"] = _effect_record(
        "flashover_sheath_initiation",
        status="implemented_not_validated" if sheath_or_breakdown else "reduced_or_absent",
        implemented=sheath_or_breakdown,
        validated=False,
        notes=(
            "Breakdown/sheath information is present, but no KR validation evidence "
            "for startup timing or flashover dynamics is attached."
            if sheath_or_breakdown
            else "No validated flashover/sheath-initiation evidence is attached."
        ),
    )

    post_pinch_empirical = bool(
        result.get("post_pinch_empirical_resistance")
        or result.get("snowplow_post_pinch")
        or result.get("R_anom")
    )
    effects["restrike_and_anomalous_resistance"] = _effect_record(
        "restrike_and_anomalous_resistance",
        status="empirical_not_validated" if post_pinch_empirical else "reduced_or_absent",
        implemented=post_pinch_empirical,
        validated=False,
        notes=(
            "Post-pinch/anomalous resistance terms are present as empirical "
            "closures, not KR-validated disruption/restrike physics."
            if post_pinch_empirical
            else "No validated post-pinch restrike/anomalous-resistance evidence is attached."
        ),
    )

    beam_evidence = bool(result.get("beam_tracker") or result.get("neutron_yield_details"))
    effects["beam_generation_and_target_coupling"] = _effect_record(
        "beam_generation_and_target_coupling",
        status="estimate_or_diagnostic" if beam_evidence else "reduced_or_absent",
        implemented=beam_evidence,
        validated=False,
        notes=(
            "Beam/yield diagnostics are present, but no KR validation evidence "
            "for self-consistent beam generation and beam-target coupling is attached."
            if beam_evidence
            else "No validated beam-generation/beam-target coupling evidence is attached."
        ),
    )

    validated_effect_scopes: dict[str, str] = {}
    for effect, (effect_evidence, evidence_key) in _validated_effects(result).items():
        if effect not in effects:
            continue
        validated_effect_scopes[effect] = str(
            effect_evidence.get("validation_scope", "")
        )
        bounded = effect_evidence.get("bounded_out") is True
        effects[effect] = _effect_record(
            effect,
            status="bounded_out" if bounded else "validated",
            implemented=bool(effect_evidence.get("implemented", not bounded)),
            verified=bool(effect_evidence.get("verified", True)),
            validated=True,
            evidence_keys=[evidence_key],
            notes=(
                "KR-backed evidence explicitly bounds this effect out of the "
                "stated claim scope."
                if bounded
                else "KR-backed validation evidence is attached for this effect."
            ),
        )

    missing = [
        name for name, effect in effects.items()
        if effect.get("validated") is not True
    ]
    scope_values = {
        scope for scope in validated_effect_scopes.values()
        if scope
    }
    same_scope_passed = (
        not missing
        and bool(scope_values)
        and len(scope_values) == 1
    )
    if not missing and not same_scope_passed:
        missing.append("same_scope_physics_packet")
    passed = not missing
    claim_blockers = _claim_blocker_matrix(effects)
    blocked_claims = [
        claim for claim, record in claim_blockers.items()
        if record.get("blocked") is True
    ]
    return {
        "passed": passed,
        "validation_tier": "high_fidelity_physics",
        "model_role": "physics_fidelity_audit",
        "source": _KR_SOURCE_BASIS["ordinary_mhd_limits"],
        "source_basis": _KR_SOURCE_BASIS,
        "required_effects": effects,
        "claim_blockers": claim_blockers,
        "blocked_claims": blocked_claims,
        "engineering_run_blocked": False,
        "effect_validation_scopes": validated_effect_scopes,
        "same_scope_passed": same_scope_passed,
        "missing_or_unvalidated_effects": missing,
        "validity_notes": {
            "claim_scope": (
                "A run is not high-fidelity predictive unless each required "
                "effect is implemented, verified, and validated or explicitly "
                "bounded out for one claimed validation scope."
            ),
            "claim_blocker_scope": (
                "Missing physics blocks only the listed predictive claims; it "
                "does not by itself invalidate non-predictive engineering runs."
            ),
            "audit_role": (
                "This audit reports physics-fidelity blockers; it is not a "
                "substitute for experimental validation evidence."
            ),
        },
    }
