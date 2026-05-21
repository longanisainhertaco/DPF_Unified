"""Control-plane scope packet for the Sprint 8 runtime demonstrator.

This module encodes the Option B scope decision (PF-1000 full-energy 27–40 kV,
Gribkov/Scholz 2007 era) as a typed governance record.  It is CONTROL-PLANE
GOVERNANCE, NOT scientific evidence or a KnowledgeReference-backed claim.

Authoritative decision: docs/FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md
(Option B recommendation) reaffirmed by Sprint 7 WS-B and Sprint 8 lead.

The selected scope label is:
    pf1000_full_energy_27_to_40_kv

All downstream artifacts (runtime deck preset, source ledgers, same-scope
packet, comparator decision) must use exactly this string.
"""

from __future__ import annotations

from typing import Any

# ── canonical scope label ────────────────────────────────────────────────────
SELECTED_SCOPE_LABEL: str = "pf1000_full_energy_27_to_40_kv"

# ── in-scope source packet identifiers ───────────────────────────────────────
# These are PF-1000 full-energy / Scholz 2000-2001 / Gribkov-Scholz 2007 era
# sources whose scope_tag is consistent with pf1000_full_energy_27_to_40_kv.
IN_SCOPE_SOURCES: tuple[str, ...] = (
    "scholz_gribkov_2007_partii",                       # KR: scholz-2007-pf1000-part2-jphysd.md
    "gribkov_2007_pf1000_part2_existing_kr_equivalent", # sprint6 extraction — same paper
    "scholz_2001_recent_progress_pf1000_hardware",      # KR: recent-progress-in-1-mj-plasma-focus...md
    "scholz_2000_pf1000_device",                        # KR: pf-1000-device-a2d6bc15.md
    "scholz_gribkov_2007_part1",                        # Scholz 2006 mega-joule KR (SXR/HXR/Yn)
    "malir_2024_interferometry_dpf",                    # KR: malir-2024-interferometry-dpf.md (ne)
    "klir_2011_tof_detector_pf1000",                    # KR: fusion-neutron-detector-for-tof...md
    "krasa_2008_anisotropy_pf1000",                     # KR: anisotropy-of-the-emission-of-dd-fusion...md
    "auluck_2021_plasma_focus_update",                  # sprint5 extraction — full-energy context
    "stepniewski_2004_pf1000_mhd_modelling",            # sprint5 extraction — pf1000 simulation context
)

# ── context-only source packet identifiers ───────────────────────────────────
# Sources that belong to the PF-1000 corpus but are cross-configuration,
# reduced-model comparators, or application reviews — usable for requirements
# and schema only, not same-scope acceptance.
CONTEXT_ONLY_SOURCES: tuple[str, ...] = (
    "shakya_2015_pf1000_pf400_lee_model",          # reduced Lee-model comparison
    "gribkov_malaquias_2006_dmp_applications",      # IAEA CRP applications review
    "scholz_1999_foam_liner_current_sheath",        # modified PF-1000 with foam liner target
    "herold_1989_poseidon_pf360_comparative",       # POSEIDON/PF-360 cross-machine context
    "loarer_2007_tokamak_gas_balance_fuel_retention",  # tokamak PWI methodology context
    "bruzzone_bernal_2001_lhi_duplicate_verification", # anomalous-resistivity context
)

# ── wrong-scope source packet identifiers ────────────────────────────────────
# Sources that are explicitly wrong-scope for pf1000_full_energy_27_to_40_kv:
# different device, different machine class, or historical Mather geometry.
# Cross-scope use requires a reviewed transfer rule (Sprint 8 guardrail 7).
WRONG_SCOPE_SOURCES: tuple[str, ...] = (
    "talebitaher_2012_nx2_detector_anisotropy",     # NX2 (NTU Singapore, 3 kJ) — wrong device
    "bernard_1977_dpf_high_intensity_neutron_source",  # historical Mather DPF review
    "ucsd_beg_current_sheath_initiation",           # UCSD 10 kJ Mather — wrong device class
    "bennett_2017_kinetic_dpf_breakdown",           # startup context; scope_tag pf1000_generic
                                                    # and not verified at full-energy operating point
    "akel_2021_pf1000_neutron_yield_16kv",          # PF-1000 Akel 16 kV — wrong voltage scope
)


def runtime_demonstrator_scope_packet() -> dict[str, Any]:
    """Return the control-plane scope packet for the Sprint 8 runtime demonstrator.

    This function encodes the Option B decision (PF-1000 full-energy 27–40 kV).
    The returned dict is GOVERNANCE, not scientific authority.

    Returns
    -------
    dict[str, Any]
        Keys include:
        - selected_scope_label: the canonical scope string
        - governance_class: always "control_plane"
        - is_scientific_authority: always False
        - accepted_runtime_claim: always False
        - in_scope_sources: tuple of source IDs
        - context_only_sources: tuple of source IDs
        - wrong_scope_sources: tuple of source IDs
        - scope_change_note: documents the change from Akel 16 kV default
    """
    return {
        "selected_scope_label": SELECTED_SCOPE_LABEL,
        "governance_class": "control_plane",
        "is_scientific_authority": False,
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "decision": "Option_B_PF1000_full_energy_27_to_40_kV",
        "decision_memo": (
            "docs/FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md"
        ),
        "governance_memo": (
            "docs/SPRINT8_WS2_RUNTIME_DEMONSTRATOR_SCOPE_LOCK_2026_05_20.md"
        ),
        "rationale_summary": (
            "Full-energy PF-1000 (27-40 kV, ~810 kJ, Gribkov/Scholz 2007) "
            "provides 7 of 9 required diagnostic channels in KR: I(t), V(t), "
            "ne (interferometry), X-ray (SXR+HXR), Yn, neutron TOF spectrum, "
            "and angular anisotropy. Te and Ti are structurally absent for all "
            "DPF devices in this corpus. Option A (Akel 16 kV) has only I(t) "
            "and Yn, and acquiring the missing channels requires a 3-6 month "
            "IPPLM campaign not yet proposed."
        ),
        "scope_change_note": (
            "This is a scope change from the prior Akel-16-kV runtime defaults. "
            "The full-energy operating point differs substantially: "
            "I_peak ~2 MA vs ~1 MA, energy ~810 kJ vs ~170 kJ. "
            "The V&V certificate must document this change explicitly and note "
            "that 16 kV results are extrapolated from the 27-40 kV regime."
        ),
        "te_ti_gap_acknowledgement": (
            "Directly measured pinch-phase Te and Ti are absent corpus-wide. "
            "They are flagged as structurally absent, not as a gap specific to "
            "this scope choice. Model-derived estimates (≤1 keV) are TEXT-ONLY."
        ),
        "in_scope_sources": IN_SCOPE_SOURCES,
        "context_only_sources": CONTEXT_ONLY_SOURCES,
        "wrong_scope_sources": WRONG_SCOPE_SOURCES,
        "cross_scope_policy": (
            "Wrong-scope sources may not close a selected-scope comparator "
            "channel without a reviewed transfer rule (Sprint 8 guardrail 7)."
        ),
        "scope_consistency_rule": (
            "A source packet is in-scope only if its scope_tag or declared_scope "
            "resolves to pf1000_full_energy_27_to_40_kv or an equivalent alias. "
            "Mixed-scope sets (in-scope + wrong-scope without transfer rule) fail "
            "the scope-consistency check."
        ),
        "sprint": "Sprint_8_WS2",
        "date": "2026-05-20",
    }


def check_scope_consistency(source_ids: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Check whether a set of source IDs mixes in-scope with wrong-scope sources.

    Parameters
    ----------
    source_ids:
        Iterable of source packet IDs to check.

    Returns
    -------
    dict[str, Any]
        ``consistent`` is True only when no wrong-scope sources are present
        alongside in-scope sources without a transfer rule.  The dict also
        lists which IDs fell into each category.
    """
    ids = list(source_ids)
    in_scope_found = [s for s in ids if s in IN_SCOPE_SOURCES]
    wrong_scope_found = [s for s in ids if s in WRONG_SCOPE_SOURCES]
    context_only_found = [s for s in ids if s in CONTEXT_ONLY_SOURCES]
    unknown = [
        s for s in ids
        if s not in IN_SCOPE_SOURCES
        and s not in WRONG_SCOPE_SOURCES
        and s not in CONTEXT_ONLY_SOURCES
    ]

    mixed = bool(in_scope_found) and bool(wrong_scope_found)
    consistent = not mixed

    return {
        "consistent": consistent,
        "selected_scope_label": SELECTED_SCOPE_LABEL,
        "in_scope_found": in_scope_found,
        "wrong_scope_found": wrong_scope_found,
        "context_only_found": context_only_found,
        "unknown_sources": unknown,
        "failure_reason": (
            "mixed_in_scope_and_wrong_scope_without_transfer_rule"
            if mixed else None
        ),
    }
