"""Fail-closed circuit power-port packet for first-principles DPF gates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dpf.first_principles.numerical_fidelity import (
    DEFAULT_PHASE3_TRANSFER_MATRIX_PATH,
    load_phase3_transfer_candidate_linkage,
)
from dpf.validation.circuit_field_coupling import field_coupling_evidence_from_result

_REQUIRED_WAVEFORM_REVIEW_KEYS = (
    "source_path",
    "figure_id",
    "extraction_method",
    "digitization_hash",
    "uncertainty",
)
_REQUIRED_POWER_PORT_REVIEW_KEYS = (
    "sign_convention",
    "time_centering",
    "poynting_or_j_dot_e_residual",
)


def build_circuit_power_port_packet(
    *,
    validation_scope: str,
    bank_parameters: Mapping[str, Any] | None = None,
    coupling_result: Mapping[str, Any] | None = None,
    waveform_packet: Mapping[str, Any] | None = None,
    review_certificate: Mapping[str, Any] | None = None,
    phase3_transfer_matrix_path: str | Path | None = DEFAULT_PHASE3_TRANSFER_MATRIX_PATH,
) -> dict[str, Any]:
    """Return a non-promoting Phase 4-B circuit power-port gate packet.

    This packet is intentionally stricter than component-level coupling helpers:
    bank/circuit scalars, density-weighted MHD inductance, and unreviewed
    waveform histories are useful engineering signals, but none can support a
    first-principles acceptance claim without reviewed sign convention,
    time-centering, Poynting/J·E residual, and review certificate metadata.
    """

    coupling = dict(coupling_result or {})
    field_evidence = field_coupling_evidence_from_result(coupling)
    transfer_linkage = load_phase3_transfer_candidate_linkage(phase3_transfer_matrix_path)
    waveform = dict(waveform_packet or {})
    certificate = dict(review_certificate or {})

    blocking_reasons: list[str] = []
    bank_present = bool(bank_parameters)
    if not coupling and not waveform:
        blocking_reasons.append("waveform_or_power_history_missing")

    if field_evidence.get("passed") is not True:
        blocking_reasons.append("field_coupling_packet_not_passed")

    if _has_density_weighted_or_metadata_only_coupling(coupling, field_evidence):
        blocking_reasons.append("density_weighted_or_metadata_only_coupling")

    missing_waveform_keys = [
        key for key in _REQUIRED_WAVEFORM_REVIEW_KEYS if key not in waveform
    ]
    if waveform or coupling:
        for key in missing_waveform_keys:
            blocking_reasons.append(f"waveform_{key}_missing")

    for key in _REQUIRED_POWER_PORT_REVIEW_KEYS:
        if key not in waveform:
            if key == "poynting_or_j_dot_e_residual":
                blocking_reasons.append("poynting_or_j_dot_e_residual_review_missing")
            else:
                blocking_reasons.append(f"{key}_missing")
    if waveform and not _power_port_residual_reviewed(
        waveform.get("poynting_or_j_dot_e_residual")
    ):
        blocking_reasons.append("poynting_or_j_dot_e_residual_review_missing")

    if certificate.get("accepted") is not True:
        blocking_reasons.append("review_certificate_missing")

    if transfer_linkage.get("promotes_acceptance") is not False:
        blocking_reasons.append("transfer_linkage_promotion_not_forbidden")
    if transfer_linkage.get("status") != "loaded_transfer_candidates_non_promoting":
        blocking_reasons.append("transfer_candidate_linkage_not_loaded")
    if transfer_linkage.get("all_transfer_candidates_non_promoting") is not True:
        blocking_reasons.append("transfer_candidate_non_promotion_not_verified")
    transfer_blockers = transfer_linkage.get("blocking_reasons")
    if isinstance(transfer_blockers, list):
        blocking_reasons.extend(str(reason) for reason in transfer_blockers)

    blocking_reasons = sorted(set(blocking_reasons))
    return {
        "status": "blocked_circuit_power_port_not_accepted",
        "validation_scope": validation_scope,
        "accepted_power_port_claim": False,
        "can_support_numerical_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "bank_circuit_transfer_candidate": {
            "present": bank_present,
            "can_support_acceptance": False,
            "reason": (
                "bank/circuit scalar parameters are transfer candidates only; "
                "they do not close waveform, field-power, or review gates"
            ),
        },
        "field_coupling_evidence": field_evidence,
        "phase3_transfer_candidate_linkage": transfer_linkage,
        "waveform_review_status": {
            "present": bool(waveform),
            "required_keys": list(_REQUIRED_WAVEFORM_REVIEW_KEYS)
            + list(_REQUIRED_POWER_PORT_REVIEW_KEYS),
            "missing_keys": missing_waveform_keys
            + [key for key in _REQUIRED_POWER_PORT_REVIEW_KEYS if key not in waveform],
            "can_support_acceptance": False,
        },
        "review_certificate_status": {
            "present": bool(certificate),
            "accepted": certificate.get("accepted") is True,
            "can_support_acceptance": False,
        },
        "blocking_reasons": blocking_reasons,
        "acceptance_rule": (
            "Circuit power-port acceptance requires reviewed same-scope waveform "
            "or field-power history, explicit sign convention, time-centering, "
            "Poynting/J·E residual closure, transfer-boundary review, and an "
            "accepted independent review certificate."
        ),
    }


def _power_port_residual_reviewed(residual: Any) -> bool:
    if not isinstance(residual, Mapping):
        return False
    if residual.get("passed") is not True:
        return False
    return residual.get("reviewed") is True or residual.get("review_status") == "accepted"


def _has_density_weighted_or_metadata_only_coupling(
    result: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> bool:
    if "Lp_mhd_nH" in result and not any(
        key in result
        for key in (
            "field_inductance",
            "field_derived_inductance",
            "magnetic_energy_inductance",
        )
    ):
        return True
    required = evidence.get("required_evidence")
    if not isinstance(required, Mapping):
        return False
    field_inductance = required.get("field_derived_inductance")
    if not isinstance(field_inductance, Mapping):
        return False
    return "density-weighted" in str(field_inductance.get("notes", ""))
