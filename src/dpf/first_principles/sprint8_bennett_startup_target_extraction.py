"""Sprint 8 WS4 — Bennett 2017 startup BVP target extraction and candidate channel packet.

Source: Bennett, N. et al. (2017). "Kinetic simulations of gas breakdown in the dense
plasma focus," Phys. Plasmas 24, 062705. DOI: 10.1063/1.4985313.

KR canonical record: KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md
(created Sprint 8 WS4, 2026-05-20).

Line/page verification: Sprint 7 WS-E
(docs/extractions/SPRINT7_WSE_NEXT_PHYSICS_SOURCE_PACKETS_2026_05_20.md §Packet 2).
All 14 verbatim targets confirmed against on-disk PDF pages.

On-disk PDF (mislabeled filename):
  archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf
  SHA-256: c5e6f5f1e2ca150a41c18c83f82a2fbaf35a2deb75d4a50b60cb7a45b0f0b92a
  Actual first author: N. Bennett (filename mislabel documented in Sprint 5 WS2).

SCOPE INTERACTION (non-negotiable):
  Phase A WS2 locked the runtime-demonstrator scope to pf1000_full_energy_27_to_40_kv
  and classified bennett_2017_kinetic_dpf_breakdown as WRONG-SCOPE for that
  demonstrator. Therefore:
  - CH03/CH04/CH07/CH08 are source-backed runtime CANDIDATE channels
    (engineering evidence only).
  - Their same-scope status for pf1000_full_energy_27_to_40_kv MUST stay
    blocked_wrong_scope absent a reviewed transfer rule.
  - Bennett values are startup-model context, NOT accepted same-scope startup evidence.

ACCEPTANCE FLAGS (immutable — do NOT change):
  accepted_runtime_claim = False
  can_support_first_principles_acceptance = False

CHANNELS RESOLVED (source-backed CANDIDATE only):
  CH03 — seed density 1e7 cm⁻³
  CH04 — breakdown delay ~20 ns
  CH07 — explosive emission thresholds 250 / 10 kV/cm; Te 3.5–4 eV (startup model context)
  CH08 — sheath current fraction 71% at 1 µs; ionization landmarks 1e13 cm⁻³ @ 100 ns;
          1e15 cm⁻³ @ 400 ns along insulator; ~1e15 cm⁻³ @ 500 ns channel

CHANNELS KEPT BLOCKED/WRONG-SCOPE:
  CH01 — blocked_wrong_scope: MA-scale 5.5 Torr fill vs Akel/PF-1000 scope
  CH02 — blocked_missing_source: no Townsend alpha(E/p) table in this source
  CH05 — blocked_missing_source: no Cu/pyrex material secondary emission gamma
  CH06 — blocked_missing_source: photons neglected; no photoemission closure supplied
  CH09 — blocked_missing_source: no start-of-discharge species/charge-state fields
  CH10 — blocked_missing_source: no D2 alpha(E/p) ionization coefficient table
  CH11 — blocked_missing_source: no DPF-valid Te/Ti closure (homogeneous-field only)
  CH12 — blocked_missing_source: no closed breakdown-BVP sheath initial state
  CH13 — blocked_missing_source: no numerical handoff-interval definition

WHOLE-SHOT STARTUP: BLOCKED. All 13 startup channels remain candidate or blocked.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Source identity (canonical KR record)
# ---------------------------------------------------------------------------

#: KR canonical record path (relative to repo root).
BENNETT_2017_KR_PATH: str = "KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md"

#: Citation (Bennett et al. 2017, Phys. Plasmas 24, 062705).
BENNETT_2017_CITATION: str = (
    "Bennett, N. et al. (2017). 'Kinetic simulations of gas breakdown in the dense "
    "plasma focus,' Phys. Plasmas 24, 062705. DOI: 10.1063/1.4985313"
)

#: On-disk PDF path (relative to repo root). Filename is mislabeled.
BENNETT_2017_PDF_PATH: str = (
    "archive_reference_OLD/references/papers/core-dpf/"
    "schmidt-2017-kinetic-dpf-breakdown.pdf"
)

#: PDF SHA-256 — verified Sprint 7 WS-E and Sprint 8 WS4.
BENNETT_2017_PDF_SHA256: str = (
    "c5e6f5f1e2ca150a41c18c83f82a2fbaf35a2deb75d4a50b60cb7a45b0f0b92a"
)

#: Scope tag for this source.
BENNETT_2017_SCOPE_TAG: str = "pf1000_generic"

#: Same-scope status for the locked demonstrator scope.
BENNETT_2017_DEMONSTRATOR_SAME_SCOPE_STATUS: str = "blocked_wrong_scope"

#: Demonstrator scope string (from Phase A WS2 lock).
BENNETT_2017_DEMONSTRATOR_SCOPE: str = "pf1000_full_energy_27_to_40_kv"

# ---------------------------------------------------------------------------
# Unit-conversion helpers (SI ↔ stated units — enforced via round-trip test)
# ---------------------------------------------------------------------------


def cm3_to_m3(value_cm3: float) -> float:
    """Convert a number density from cm⁻³ to m⁻³.

    1 cm = 1e-2 m → 1 cm³ = 1e-6 m³ → 1 cm⁻³ = 1e6 m⁻³.

    Args:
        value_cm3: number density in cm⁻³.

    Returns:
        Number density in m⁻³.
    """
    return value_cm3 * 1.0e6


def m3_to_cm3(value_m3: float) -> float:
    """Convert a number density from m⁻³ to cm⁻³.

    Args:
        value_m3: number density in m⁻³.

    Returns:
        Number density in cm⁻³.
    """
    return value_m3 * 1.0e-6


def ns_to_s(value_ns: float) -> float:
    """Convert nanoseconds to seconds.

    Args:
        value_ns: time in nanoseconds.

    Returns:
        Time in seconds.
    """
    return value_ns * 1.0e-9


def s_to_ns(value_s: float) -> float:
    """Convert seconds to nanoseconds.

    Args:
        value_s: time in seconds.

    Returns:
        Time in nanoseconds.
    """
    return value_s * 1.0e9


def kv_cm_to_v_m(value_kv_cm: float) -> float:
    """Convert electric field from kV/cm to V/m.

    1 kV = 1e3 V, 1 cm = 1e-2 m → 1 kV/cm = 1e3/1e-2 V/m = 1e5 V/m.

    Args:
        value_kv_cm: electric field in kV/cm.

    Returns:
        Electric field in V/m.
    """
    return value_kv_cm * 1.0e5


def v_m_to_kv_cm(value_v_m: float) -> float:
    """Convert electric field from V/m to kV/cm.

    Args:
        value_v_m: electric field in V/m.

    Returns:
        Electric field in kV/cm.
    """
    return value_v_m * 1.0e-5


def ev_to_K(value_ev: float) -> float:
    """Convert electron temperature from eV to Kelvin.

    k_B = 1.380649e-23 J/K, e = 1.602176634e-19 C → 1 eV / k_B = 11604.52 K.

    Args:
        value_ev: temperature in eV.

    Returns:
        Temperature in Kelvin.
    """
    return value_ev * 11604.5221


def K_to_ev(value_K: float) -> float:
    """Convert temperature from Kelvin to eV.

    Args:
        value_K: temperature in Kelvin.

    Returns:
        Temperature in eV.
    """
    return value_K / 11604.5221


def torr_to_Pa(value_torr: float) -> float:
    """Convert pressure from Torr to Pascal.

    1 Torr = 133.322368 Pa.

    Args:
        value_torr: pressure in Torr.

    Returns:
        Pressure in Pascal.
    """
    return value_torr * 133.322368


def Pa_to_torr(value_Pa: float) -> float:
    """Convert pressure from Pascal to Torr.

    Args:
        value_Pa: pressure in Pascal.

    Returns:
        Pressure in Torr.
    """
    return value_Pa / 133.322368


def us_to_s(value_us: float) -> float:
    """Convert microseconds to seconds.

    Args:
        value_us: time in microseconds.

    Returns:
        Time in seconds.
    """
    return value_us * 1.0e-6


def s_to_us(value_s: float) -> float:
    """Convert seconds to microseconds.

    Args:
        value_s: time in seconds.

    Returns:
        Time in microseconds.
    """
    return value_s * 1.0e6


# ---------------------------------------------------------------------------
# CH03 — Seed plasma density (source-backed CANDIDATE)
# ---------------------------------------------------------------------------

#: CH03: seed plasma density in stated units (cm⁻³).
CH03_N_SEED_CM3: float = 1.0e7

#: CH03: seed plasma density in SI units (m⁻³).
CH03_N_SEED_M3: float = cm3_to_m3(CH03_N_SEED_CM3)

#: CH03: source page (journal page number).
CH03_SOURCE_PAGE: str = "p.2 (062705-2)"

#: CH03: verbatim quote from PDF p.2 (confirmed Sprint 7 WS-E).
CH03_VERBATIM: str = (
    "The DPF volume is also initialized with a 10^7-cm^-3 density plasma of "
    "deuterium ions and electrons to provide seed electrons for the avalanche "
    "ionization process."
)

#: CH03: KR line reference (KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md).
CH03_KR_LINES: str = "§Section 2"

#: CH03: same-scope status for demonstrator.
CH03_DEMONSTRATOR_SAME_SCOPE_STATUS: str = "blocked_wrong_scope"

#: CH03: candidate channel record.
BENNETT_CH03_SEED_DENSITY: Mapping[str, Any] = {
    "channel_id": "STARTUP-BVP-CH03",
    "channel_name": "preionization_seed_density",
    "source_id": "bennett_2017_kinetic_dpf_breakdown",
    "kr_path": BENNETT_2017_KR_PATH,
    "kr_section": "§Section 2 — CH03 target: seed plasma density",
    "source_page": CH03_SOURCE_PAGE,
    "verbatim": CH03_VERBATIM,
    "values": {
        "n_seed": {
            "stated_value": CH03_N_SEED_CM3,
            "stated_units": "cm^-3",
            "si_value": CH03_N_SEED_M3,
            "si_units": "m^-3",
            "conversion_factor": 1.0e6,
            "conversion_note": "1 cm^-3 = 1e6 m^-3",
        },
    },
    "symbol_map": {"n_seed": "seed plasma (ion + electron) number density at t=0"},
    "source_backed_candidate": True,
    "same_scope_status_for_demonstrator": CH03_DEMONSTRATOR_SAME_SCOPE_STATUS,
    "demonstrator_scope": BENNETT_2017_DEMONSTRATOR_SCOPE,
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "scope_tag": BENNETT_2017_SCOPE_TAG,
    "scope_caveat": (
        "pf1000_generic kinetic-PIC MA-scale DPF; NOT same-scope for "
        "pf1000_full_energy_27_to_40_kv without a reviewed transfer rule"
    ),
    "whole_shot_startup_blocked": True,
}

# ---------------------------------------------------------------------------
# CH04 — Breakdown delay (source-backed CANDIDATE)
# ---------------------------------------------------------------------------

#: CH04: breakdown delay in stated units (ns).
CH04_T_BREAKDOWN_NS: float = 20.0

#: CH04: breakdown delay in SI units (s).
CH04_T_BREAKDOWN_S: float = ns_to_s(CH04_T_BREAKDOWN_NS)

#: CH04: qualifier for the breakdown delay value.
CH04_QUALIFIER: str = "approximate"

#: CH04: source page for breakdown timing.
CH04_SOURCE_PAGE_TIMING: str = "p.4 (062705-4)"

#: CH04: verbatim quote — breakdown delay.
CH04_VERBATIM_TIMING: str = (
    "The measured breakdown time (the time between the rise of voltage and the "
    "rapid rise in current) is approximately 20 ns."
)

#: CH04: source page for pressure regimes.
CH04_SOURCE_PAGE_REGIMES: str = "p.5 (062705-5)"

#: CH04: pressure regime boundaries (qualitative context).
CH04_PRESSURE_REGIMES: Mapping[str, Any] = {
    "low": {
        "criterion": "lambda_ioniz > 20 cm (exceeds electrode gap)",
        "mode": "volumetric uniform breakdown",
        "source_page": CH04_SOURCE_PAGE_REGIMES,
        "verbatim": (
            "At low pressures, the electron ionization path length exceeds 20 cm "
            "so electrons traveling axially are more likely to ionize the gas "
            "leading to bulk breakdown in the DPF volume."
        ),
    },
    "medium": {
        "criterion": "lambda_ioniz ~ L_insulator > coaxial gap",
        "mode": "surface ionization along insulator (optimal for sheath uniformity)",
        "source_page": CH04_SOURCE_PAGE_REGIMES,
        "verbatim": (
            "In an intermediate pressure range, the ionization path length may "
            "exceed the coaxial gap but approach the length of the insulator, "
            "which is longer than the gap in typical DPF designs."
        ),
    },
    "high": {
        "criterion": "lambda_ioniz within a few cm (pressure > ~15 Torr)",
        "mode": "radial filamentation across coaxial gap",
        "source_page": CH04_SOURCE_PAGE_REGIMES,
        "verbatim": (
            "At pressures above 15 Torr, electron impact ionization occurs within "
            "a few cms, so the gas may breakdown radially across the coaxial gap."
        ),
    },
}

#: CH04: photoionization negligibility (context for CH06 blocked status).
CH04_PHOTOIONIZATION_CONTEXT: Mapping[str, Any] = {
    "value_percent": 1.2,
    "by_time_ns": 125.0,
    "source_page": "p.3 (062705-3)",
    "verbatim": (
        "Preliminary simulations run with the addition of photoionization showed a "
        "1.2% increase in electron density by 125 ns. This results from a relatively "
        "low photon population (below 10^10 cm^-3 from excited deuterium) and a "
        "photoionization cross section that is an order of magnitude smaller than "
        "the electron impact ionization cross section. Photons are, therefore, "
        "neglected here."
    ),
    "significance": "justifies CH06 (photoemission) blocked status",
}

#: CH04: candidate channel record.
BENNETT_CH04_BREAKDOWN_DELAY: Mapping[str, Any] = {
    "channel_id": "STARTUP-BVP-CH04",
    "channel_name": "flashover_breakdown_delay",
    "source_id": "bennett_2017_kinetic_dpf_breakdown",
    "kr_path": BENNETT_2017_KR_PATH,
    "kr_section": "§Section 3 — CH04 target: breakdown delay",
    "source_page_timing": CH04_SOURCE_PAGE_TIMING,
    "source_page_regimes": CH04_SOURCE_PAGE_REGIMES,
    "verbatim_timing": CH04_VERBATIM_TIMING,
    "values": {
        "t_breakdown": {
            "stated_value": CH04_T_BREAKDOWN_NS,
            "stated_units": "ns",
            "si_value": CH04_T_BREAKDOWN_S,
            "si_units": "s",
            "conversion_factor": 1.0e-9,
            "conversion_note": "1 ns = 1e-9 s",
            "qualifier": CH04_QUALIFIER,
        },
    },
    "symbol_map": {
        "t_breakdown": (
            "time from voltage rise to rapid current rise (breakdown completion)"
        )
    },
    "pressure_regimes": CH04_PRESSURE_REGIMES,
    "photoionization_context": CH04_PHOTOIONIZATION_CONTEXT,
    "source_backed_candidate": True,
    "same_scope_status_for_demonstrator": "blocked_wrong_scope",
    "demonstrator_scope": BENNETT_2017_DEMONSTRATOR_SCOPE,
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "scope_tag": BENNETT_2017_SCOPE_TAG,
    "whole_shot_startup_blocked": True,
}

# ---------------------------------------------------------------------------
# CH07 — Explosive emission thresholds and electron temperature (source-backed CANDIDATE)
# ---------------------------------------------------------------------------

#: CH07: bulk cathode surface explosive emission threshold in stated units (kV/cm).
CH07_E_THRESHOLD_BULK_KV_CM: float = 250.0

#: CH07: bulk threshold in SI units (V/m).
CH07_E_THRESHOLD_BULK_V_M: float = kv_cm_to_v_m(CH07_E_THRESHOLD_BULK_KV_CM)

#: CH07: cathode knife-edge explosive emission threshold in stated units (kV/cm).
CH07_E_THRESHOLD_KNIFE_KV_CM: float = 10.0

#: CH07: knife-edge threshold in SI units (V/m).
CH07_E_THRESHOLD_KNIFE_V_M: float = kv_cm_to_v_m(CH07_E_THRESHOLD_KNIFE_KV_CM)

#: CH07: source page for thresholds.
CH07_SOURCE_PAGE_THRESHOLDS: str = "p.3 (062705-3)"

#: CH07: verbatim quote — thresholds (both in same sentence).
CH07_VERBATIM_THRESHOLDS: str = (
    "We use an electric field stress threshold of 250 kV/cm except for the cathode "
    "knife-edge, where the threshold is reduced to 10 kV/cm to approximate the field "
    "enhancement of its 3D structures in our 2D model."
)

#: CH07: electron temperature at breakdown, lower bound (eV).
CH07_TE_EV_LOW: float = 3.5

#: CH07: electron temperature at breakdown, upper bound (eV).
CH07_TE_EV_HIGH: float = 4.0

#: CH07: electron temperature lower bound in Kelvin.
CH07_TE_K_LOW: float = ev_to_K(CH07_TE_EV_LOW)

#: CH07: electron temperature upper bound in Kelvin.
CH07_TE_K_HIGH: float = ev_to_K(CH07_TE_EV_HIGH)

#: CH07: source page for electron temperature.
CH07_SOURCE_PAGE_TE: str = "p.5 (062705-5)"

#: CH07: verbatim quote — electron temperature.
CH07_VERBATIM_TE: str = (
    "the mean local temperatures (T_e) in the electron distributions from simulation "
    "remain near 4 eV, well into breakdown, as shown in Fig. 7."
)

#: CH07: candidate channel record.
BENNETT_CH07_EXPLOSIVE_EMISSION: Mapping[str, Any] = {
    "channel_id": "STARTUP-BVP-CH07",
    "channel_name": "surface_plasma_explosive_emission",
    "source_id": "bennett_2017_kinetic_dpf_breakdown",
    "kr_path": BENNETT_2017_KR_PATH,
    "kr_section": (
        "§Section 4 — CH07 targets: explosive emission thresholds and electron temperature"
    ),
    "source_page_thresholds": CH07_SOURCE_PAGE_THRESHOLDS,
    "source_page_te": CH07_SOURCE_PAGE_TE,
    "verbatim_thresholds": CH07_VERBATIM_THRESHOLDS,
    "verbatim_te": CH07_VERBATIM_TE,
    "values": {
        "E_threshold_bulk": {
            "stated_value": CH07_E_THRESHOLD_BULK_KV_CM,
            "stated_units": "kV/cm",
            "si_value": CH07_E_THRESHOLD_BULK_V_M,
            "si_units": "V/m",
            "conversion_factor": 1.0e5,
            "conversion_note": "1 kV/cm = 1e5 V/m",
        },
        "E_threshold_knife": {
            "stated_value": CH07_E_THRESHOLD_KNIFE_KV_CM,
            "stated_units": "kV/cm",
            "si_value": CH07_E_THRESHOLD_KNIFE_V_M,
            "si_units": "V/m",
            "conversion_factor": 1.0e5,
            "conversion_note": "1 kV/cm = 1e5 V/m",
        },
        "T_e": {
            "stated_value_eV": CH07_TE_EV_HIGH,
            "stated_range_eV": (CH07_TE_EV_LOW, CH07_TE_EV_HIGH),
            "stated_units": "eV",
            "si_value_K": CH07_TE_K_HIGH,
            "si_range_K": (CH07_TE_K_LOW, CH07_TE_K_HIGH),
            "si_units": "K",
            "conversion_factor": 11604.5221,
            "conversion_note": "1 eV = 11604.52 K",
            "scope_caveat": (
                "startup model context only — NOT an accepted runtime claim "
                "for T_e in the pf1000_full_energy demonstrator"
            ),
        },
    },
    "symbol_map": {
        "E_threshold_bulk": "explosive emission onset E-field, bulk cathode surface",
        "E_threshold_knife": "explosive emission onset E-field, cathode knife-edge",
        "T_e": "mean local electron temperature during breakdown",
    },
    "source_backed_candidate": True,
    "same_scope_status_for_demonstrator": "blocked_wrong_scope",
    "demonstrator_scope": BENNETT_2017_DEMONSTRATOR_SCOPE,
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "scope_tag": BENNETT_2017_SCOPE_TAG,
    "whole_shot_startup_blocked": True,
}

# ---------------------------------------------------------------------------
# CH08 — Sheath current fraction and ionization landmarks (source-backed CANDIDATE)
# ---------------------------------------------------------------------------

#: CH08: sheath current fraction at 1 µs (dimensionless, fraction).
CH08_F_SHEATH: float = 0.71

#: CH08: reference time for sheath current fraction (µs stated, s SI).
CH08_T_REF_US: float = 1.0

#: CH08: reference time in SI units.
CH08_T_REF_S: float = us_to_s(CH08_T_REF_US)

#: CH08: source page for sheath current fraction.
CH08_SOURCE_PAGE_SHEATH: str = "p.3 (062705-3)"

#: CH08: verbatim quote — sheath current fraction at 1 µs.
CH08_VERBATIM_SHEATH: str = "by 1 us [Fig. 4(c)], it is carrying 71% of the current."

#: CH08: Sprint 5 audit-row-7 correction confirmation.
CH08_AUDIT_ROW7_NOTE: str = (
    "Codex audit row-7 correction CONFIRMED: 71% is at 1 µs, not 500 ns. "
    "The 500 ns entry (Fig. 4(b)) gives only channel formation; no "
    "current-fraction percentage at 500 ns."
)

#: CH08: bulk ionization density at 100 ns in stated units (cm⁻³).
CH08_N_IONIZ_100NS_CM3: float = 1.0e13

#: CH08: bulk ionization density at 100 ns in SI units (m⁻³).
CH08_N_IONIZ_100NS_M3: float = cm3_to_m3(CH08_N_IONIZ_100NS_CM3)

#: CH08: plasma density along insulator at 400 ns in stated units (cm⁻³).
CH08_N_INSULATOR_400NS_CM3: float = 1.0e15

#: CH08: plasma density along insulator at 400 ns in SI units (m⁻³).
CH08_N_INSULATOR_400NS_M3: float = cm3_to_m3(CH08_N_INSULATOR_400NS_CM3)

#: CH08: plasma channel density at 500 ns in stated units (cm⁻³); ~1e15 cm⁻³ level.
CH08_N_CHANNEL_500NS_CM3: float = 1.0e15

#: CH08: plasma channel density at 500 ns in SI units (m⁻³).
CH08_N_CHANNEL_500NS_M3: float = cm3_to_m3(CH08_N_CHANNEL_500NS_CM3)

#: CH08: verbatim quote — bulk ionization at 100 ns.
CH08_VERBATIM_100NS: str = (
    "By 100 ns, as the plasma sheath is forming, a bulk ionization of order "
    "10^13 cm^-3 has already occurred in the volume."
)

#: CH08: verbatim quote — plasma channel at 500 ns.
CH08_VERBATIM_500NS: str = (
    "By 500 ns [Fig. 4(b)], a plasma channel has formed across the coaxial "
    "electrode gap."
)

#: CH08: verbatim quote — insulator plasma at 400 ns.
CH08_VERBATIM_400NS: str = (
    "By 400 ns into the pulse a plasma of 10^15 cm^-3 density has formed along "
    "the insulator with the aid of the cathode knife-edge."
)

#: CH08: candidate channel record.
BENNETT_CH08_SHEATH_IONIZATION: Mapping[str, Any] = {
    "channel_id": "STARTUP-BVP-CH08",
    "channel_name": "initial_e_b_j_and_ionization_landmarks",
    "source_id": "bennett_2017_kinetic_dpf_breakdown",
    "kr_path": BENNETT_2017_KR_PATH,
    "kr_section": (
        "§Section 5 — CH08 targets: sheath current fraction and ionization landmarks"
    ),
    "source_page": CH08_SOURCE_PAGE_SHEATH,
    "verbatim_sheath": CH08_VERBATIM_SHEATH,
    "verbatim_100ns": CH08_VERBATIM_100NS,
    "verbatim_400ns": CH08_VERBATIM_400NS,
    "verbatim_500ns": CH08_VERBATIM_500NS,
    "audit_row7_note": CH08_AUDIT_ROW7_NOTE,
    "values": {
        "f_sheath": {
            "stated_value_percent": 71.0,
            "si_value_fraction": CH08_F_SHEATH,
            "si_units": "dimensionless",
            "at_time_us": CH08_T_REF_US,
            "at_time_s": CH08_T_REF_S,
            "conversion_note": "71% / 100 = 0.71 fraction; 1 µs = 1e-6 s",
        },
        "n_ioniz_100ns": {
            "stated_value": CH08_N_IONIZ_100NS_CM3,
            "stated_units": "cm^-3",
            "si_value": CH08_N_IONIZ_100NS_M3,
            "si_units": "m^-3",
            "conversion_factor": 1.0e6,
            "conversion_note": "1 cm^-3 = 1e6 m^-3",
            "qualifier": "order of magnitude",
            "at_time_ns": 100.0,
            "at_time_s": ns_to_s(100.0),
        },
        "n_insulator_400ns": {
            "stated_value": CH08_N_INSULATOR_400NS_CM3,
            "stated_units": "cm^-3",
            "si_value": CH08_N_INSULATOR_400NS_M3,
            "si_units": "m^-3",
            "conversion_factor": 1.0e6,
            "conversion_note": "1 cm^-3 = 1e6 m^-3",
            "at_time_ns": 400.0,
            "at_time_s": ns_to_s(400.0),
            "location": "along insulator near cathode knife-edge",
        },
        "n_channel_500ns": {
            "stated_value": CH08_N_CHANNEL_500NS_CM3,
            "stated_units": "cm^-3",
            "si_value": CH08_N_CHANNEL_500NS_M3,
            "si_units": "m^-3",
            "conversion_factor": 1.0e6,
            "conversion_note": "1 cm^-3 = 1e6 m^-3",
            "at_time_ns": 500.0,
            "at_time_s": ns_to_s(500.0),
            "location": "plasma channel across coaxial electrode gap",
        },
    },
    "symbol_map": {
        "f_sheath": "sheath current fraction (dimensionless, 0–1)",
        "n_ioniz_100ns": "bulk ionization number density at t=100 ns",
        "n_insulator_400ns": "plasma density along insulator surface at t=400 ns",
        "n_channel_500ns": "plasma channel density across coaxial gap at t=500 ns",
    },
    "source_backed_candidate": True,
    "same_scope_status_for_demonstrator": "blocked_wrong_scope",
    "demonstrator_scope": BENNETT_2017_DEMONSTRATOR_SCOPE,
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "scope_tag": BENNETT_2017_SCOPE_TAG,
    "whole_shot_startup_blocked": True,
}

# ---------------------------------------------------------------------------
# Channels kept blocked or wrong-scope (not resolved by this source)
# ---------------------------------------------------------------------------

#: Status record for channels NOT resolved by Bennett 2017.
BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE: Mapping[str, Mapping[str, str]] = {
    "CH01": {
        "channel_id": "STARTUP-BVP-CH01",
        "status": "blocked_wrong_scope",
        "reason": (
            "Bennett 2017 MA-scale 5.5 Torr D2 fill pressure is corroborative "
            "context only; does NOT close CH01 fill-condition for Akel/PF-1000 scope"
        ),
    },
    "CH02": {
        "channel_id": "STARTUP-BVP-CH02",
        "status": "blocked_missing_source",
        "reason": (
            "Bennett 2017 does not supply Townsend alpha(E/p) table or Paschen A/B "
            "constants for D2; no breakdown-closure data in this source"
        ),
    },
    "CH05": {
        "channel_id": "STARTUP-BVP-CH05",
        "status": "blocked_missing_source",
        "reason": (
            "Bennett 2017 does not supply secondary electron emission yields for "
            "Cu anode, pyrex/alumina insulator (DPF materials)"
        ),
    },
    "CH06": {
        "channel_id": "STARTUP-BVP-CH06",
        "status": "blocked_missing_source",
        "reason": (
            "Bennett 2017 explicitly neglects photons (1.2% effect at 125 ns); "
            "no photoemission boundary model supplied"
        ),
    },
    "CH09": {
        "channel_id": "STARTUP-BVP-CH09",
        "status": "blocked_missing_source",
        "reason": (
            "Bennett 2017 does not supply start-of-discharge species/charge-state "
            "field distributions for the breakdown BVP"
        ),
    },
    "CH10": {
        "channel_id": "STARTUP-BVP-CH10",
        "status": "blocked_missing_source",
        "reason": (
            "Bennett 2017 uses a PIC model; does not supply D2 alpha(E/p) "
            "ionization coefficient table or beta_ep recombination data"
        ),
    },
    "CH11": {
        "channel_id": "STARTUP-BVP-CH11",
        "status": "blocked_missing_source",
        "reason": (
            "Te 3.5-4 eV in Bennett 2017 is a simulation result for startup model "
            "context only; not a DPF-valid Te/Ti closure (homogeneous-field "
            "assumption invalid for coaxial gap)"
        ),
    },
    "CH12": {
        "channel_id": "STARTUP-BVP-CH12",
        "status": "blocked_missing_source",
        "reason": (
            "Bennett 2017 describes sheath formation qualitatively; does not supply "
            "a closed breakdown-BVP sheath initial state (mask, thickness, density, "
            "conductivity, velocity)"
        ),
    },
    "CH13": {
        "channel_id": "STARTUP-BVP-CH13",
        "status": "blocked_missing_source",
        "reason": (
            "Bennett 2017 does not supply a numerical handoff-interval definition "
            "(t_start, t_handoff, tolerance) or a same-device reviewed PIC import payload"
        ),
    },
}

# ---------------------------------------------------------------------------
# Full WS4 packet
# ---------------------------------------------------------------------------

#: Sprint 8 WS4 complete target extraction packet.
SPRINT8_WS4_BENNETT_STARTUP_PACKET: Mapping[str, Any] = {
    "packet_id": "sprint8_ws4_bennett_2017_startup_target_extraction",
    "sprint": "Sprint 8 WS4",
    "controlling_doc": (
        "docs/SPRINT8_SUPER_SPRINT_SOURCE_TO_RUNTIME_INSTRUCTIONS_2026_05_20.md §WS4"
    ),
    "citation": BENNETT_2017_CITATION,
    "kr_path": BENNETT_2017_KR_PATH,
    "pdf_path": BENNETT_2017_PDF_PATH,
    "pdf_sha256": BENNETT_2017_PDF_SHA256,
    "scope_tag": BENNETT_2017_SCOPE_TAG,
    "demonstrator_scope": BENNETT_2017_DEMONSTRATOR_SCOPE,
    "demonstrator_same_scope_status": BENNETT_2017_DEMONSTRATOR_SAME_SCOPE_STATUS,
    "candidate_channels": {
        "CH03": BENNETT_CH03_SEED_DENSITY,
        "CH04": BENNETT_CH04_BREAKDOWN_DELAY,
        "CH07": BENNETT_CH07_EXPLOSIVE_EMISSION,
        "CH08": BENNETT_CH08_SHEATH_IONIZATION,
    },
    "blocked_or_wrong_scope_channels": dict(BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE),
    "whole_shot_startup_blocked": True,
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "already_in_kr": True,
    "kr_ingestion_sprint": "Sprint 8 WS4",
    "line_page_verification_sprint": "Sprint 7 WS-E",
}


# ---------------------------------------------------------------------------
# PDF SHA-256 verification helper
# ---------------------------------------------------------------------------


def verify_pdf_sha256(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Verify the Bennett 2017 on-disk PDF SHA-256 against the KR-recorded value.

    Args:
        repo_root: path to the repository root. If None, inferred from this file's
            location (src/dpf/first_principles/ → ../../..).

    Returns:
        dict with keys ``path``, ``expected``, ``computed``, ``match``.

    Raises:
        FileNotFoundError: if the PDF is not present at the expected path.
    """
    if repo_root is None:
        # This file lives at src/dpf/first_principles/sprint8_bennett_*.py
        # Repo root is three levels up.
        repo_root = Path(__file__).resolve().parents[3]
    root = Path(repo_root)
    pdf_path = root / BENNETT_2017_PDF_PATH
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Bennett 2017 PDF not found at {pdf_path}; "
            f"expected repo-relative path: {BENNETT_2017_PDF_PATH}"
        )
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return {
        "path": str(pdf_path),
        "expected": BENNETT_2017_PDF_SHA256,
        "computed": sha256,
        "match": sha256 == BENNETT_2017_PDF_SHA256,
    }


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------


def sprint8_ws4_bennett_startup_packet() -> Mapping[str, Any]:
    """Return the Sprint 8 WS4 Bennett 2017 startup BVP target extraction packet.

    The packet carries CH03/CH04/CH07/CH08 as source-backed runtime CANDIDATE
    channels (engineering evidence only). All acceptance flags are False. All
    13 startup channels remain candidate or blocked; whole-shot startup acceptance
    remains blocked.

    Returns:
        The complete WS4 packet mapping (immutable view).
    """
    return SPRINT8_WS4_BENNETT_STARTUP_PACKET


__all__ = (
    "BENNETT_2017_KR_PATH",
    "BENNETT_2017_CITATION",
    "BENNETT_2017_PDF_PATH",
    "BENNETT_2017_PDF_SHA256",
    "BENNETT_2017_SCOPE_TAG",
    "BENNETT_2017_DEMONSTRATOR_SCOPE",
    "BENNETT_2017_DEMONSTRATOR_SAME_SCOPE_STATUS",
    # Unit conversion helpers
    "cm3_to_m3",
    "m3_to_cm3",
    "ns_to_s",
    "s_to_ns",
    "kv_cm_to_v_m",
    "v_m_to_kv_cm",
    "ev_to_K",
    "K_to_ev",
    "torr_to_Pa",
    "Pa_to_torr",
    "us_to_s",
    "s_to_us",
    # CH03 constants
    "CH03_N_SEED_CM3",
    "CH03_N_SEED_M3",
    "BENNETT_CH03_SEED_DENSITY",
    # CH04 constants
    "CH04_T_BREAKDOWN_NS",
    "CH04_T_BREAKDOWN_S",
    "BENNETT_CH04_BREAKDOWN_DELAY",
    # CH07 constants
    "CH07_E_THRESHOLD_BULK_KV_CM",
    "CH07_E_THRESHOLD_BULK_V_M",
    "CH07_E_THRESHOLD_KNIFE_KV_CM",
    "CH07_E_THRESHOLD_KNIFE_V_M",
    "CH07_TE_EV_LOW",
    "CH07_TE_EV_HIGH",
    "CH07_TE_K_LOW",
    "CH07_TE_K_HIGH",
    "BENNETT_CH07_EXPLOSIVE_EMISSION",
    # CH08 constants
    "CH08_F_SHEATH",
    "CH08_T_REF_S",
    "CH08_N_IONIZ_100NS_CM3",
    "CH08_N_IONIZ_100NS_M3",
    "CH08_N_INSULATOR_400NS_CM3",
    "CH08_N_INSULATOR_400NS_M3",
    "BENNETT_CH08_SHEATH_IONIZATION",
    # Blocked channels
    "BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE",
    # Packet
    "SPRINT8_WS4_BENNETT_STARTUP_PACKET",
    "sprint8_ws4_bennett_startup_packet",
    "verify_pdf_sha256",
)
