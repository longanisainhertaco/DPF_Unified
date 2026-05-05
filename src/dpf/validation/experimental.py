"""Backward-compatible re-export facade for DPF experimental validation data.

This module re-exports everything that external code imports from the
``dpf.validation.experimental`` namespace.  The implementation is split across
five sub-modules; do not import from them directly — use this facade so that
internal refactors remain transparent to callers.

Sub-module layout
-----------------
experimental_device.py
    ``ExperimentalDevice`` dataclass — one instance per real device.
experimental_waveforms.py
    Digitized or reconstructed I(t) waveform arrays for devices that have them.
experimental_devices.py
    All device instances, the ``DEVICES`` active registry, the ``_REFERENCE_ONLY``
    exclusion registry, and the ``get_devices_by_provenance`` helper.
experimental_diagnostics.py
    L_p/L0 ratio, speed factor, and bare-RLC timing diagnostics.
experimental_comparison.py
    NRMSE metrics, ``validate_current_waveform``, ``validate_neutron_yield``,
    and ``device_to_config_dict``.

Public API
----------
Device registries
~~~~~~~~~~~~~~~~~
DEVICES : dict[str, ExperimentalDevice]
    Active validation set.  Every entry has been fitted by the Lee model and
    its parameters are traceable to a source in ``KnowledgeReference/`` or a
    cited publication.  Use these devices for pass/fail validation claims.

    Current members (9 devices):

    ============== ================================= =====================
    Key            Instance                          kr_status
    ============== ================================= =====================
    "PF-1000"      PF1000_DATA                       "unverified"
    "PF-1000-Gribkov" PF1000_GRIBKOV_DATA            "unverified"
    "PF-1000-16kV" PF1000_16KV_DATA                  "unverified"
    "PF-1000-20kV" PF1000_20KV_DATA                  "unverified"
    "NX2"          NX2_DATA                          "unverified"
    "UNU-ICTP"     UNU_ICTP_DATA                     "verified"
    "POSEIDON-60kV" POSEIDON_60KV_DATA               "verified"
    "FAETON-I"     FAETON_DATA                       "unverified"
    "MJOLNIR"      MJOLNIR_DATA                      "unverified"
    ============== ================================= =====================

_REFERENCE_ONLY : dict[str, ExperimentalDevice]
    Devices excluded from active validation.  Two reasons for exclusion:

    * **Missing source paper** — parameters cannot be traced to a file in
      ``KnowledgeReference/``.
    * **Type 2 unfittable** — Lee & Saw 2014 explicitly states the device
      cannot be reproduced by the Lee model.

    Do *not* use these for pass/fail validation claims or Lee-model fitting.

    Current members:

    ========== ================== =============================================
    Key        Instance           Reason
    ========== ================== =============================================
    "POSEIDON" POSEIDON_DATA      Herold 1989 not on disk; 40 kV variant unverified
    "AECS-PF2" AECS_PF2_DATA      Type 2 unfittable (Lee & Saw 2014 §lines 4177-4183)
    ========== ================== =============================================

Per-device dataclass instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PF1000_DATA          — PF-1000 at 27 kV / 3.5 Torr D2 (IPPLM Warsaw; Scholz 2006)
PF1000_GRIBKOV_DATA  — PF-1000 at 27 kV, Gribkov (2007) shot; different waveform
PF1000_16KV_DATA     — PF-1000 at 16 kV / 1.05 Torr D2 (Akel 2021)
PF1000_20KV_DATA     — PF-1000 at 20 kV / 2.0 Torr D2 (voltage-scaled, estimated)
NX2_DATA             — NX2 compact Mather-type DPF (NIE Singapore; Lee & Saw 2008)
UNU_ICTP_DATA        — UNU-ICTP PFF training device (Lee & Saw 2014)
POSEIDON_DATA        — POSEIDON at 40 kV (IPF Stuttgart; _REFERENCE_ONLY)
POSEIDON_60KV_DATA   — POSEIDON at 60 kV (IPF Stuttgart; Lee & Saw 2014)
FAETON_DATA          — FAETON-I at 100 kV (Fuse Energy; Damideh 2025)
MJOLNIR_DATA         — MJOLNIR at 100 kV / 1 MJ (LLNL; Schmidt 2021)
AECS_PF2_DATA        — AECS-PF2 Syrian device (_REFERENCE_ONLY; Type 2 unfittable)

ExperimentalDevice dataclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``ExperimentalDevice`` holds all published parameters for one real device:
circuit values (C, L, R), electrode geometry, fill conditions, scalar
observables (I_peak, neutron yield, rise time), Lee model fitting coefficients
(fc, fm, fmr, fcr), digitized waveform arrays, uncertainty budgets (GUM /
ASME V&V 20-2009), and two provenance flags:

reliability : str
    ``"measured"``     — parameters from direct Rogowski / probe measurement.
    ``"reference_only"`` — model output or unreliable source; exclude from
    validation claims.
    ``"estimated"``    — interpolated or scaled from another operating point.

kr_status : str
    ``"verified"``      — all parameters traced to a file under
    ``KnowledgeReference/`` in the current session.
    ``"unverified"``    — a KR source exists but the parameters have not been
    cross-checked line-by-line yet.
    ``"reference_only"`` — no valid KR source; device lives in
    ``_REFERENCE_ONLY``, not ``DEVICES``.

Helpers
~~~~~~~
get_devices_by_provenance(provenance)
    Filter ``DEVICES`` by ``waveform_provenance`` field.
    Values: ``"measured"`` | ``"reconstructed"`` | ``""`` (no waveform).
kr_status_filter(status)
    Filter both ``DEVICES`` and ``_REFERENCE_ONLY`` by ``kr_status`` field.
    Values: ``"verified"`` | ``"unverified"`` | ``"reference_only"``.

Usage
-----
::

    from dpf.validation.experimental import (
        DEVICES, _REFERENCE_ONLY,
        PF1000_DATA, MJOLNIR_DATA,
        validate_current_waveform,
        validate_neutron_yield,
        device_to_config_dict,
        kr_status_filter,
    )

    # Active validation devices only
    for name, dev in DEVICES.items():
        print(name, dev.kr_status)

    # Only KR-verified devices
    for name, dev in kr_status_filter("verified").items():
        print(name)

    metrics = validate_current_waveform(t_sim, I_sim, "PF-1000")
    print(f"Peak current error: {metrics['peak_current_error']:.1%}")

Units: SI throughout (metres, seconds, amperes, farads, henries, ohms).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export everything that external code imports from this module.
# Keep this list in sync with __init__.py.
# ---------------------------------------------------------------------------
# Validation metrics and comparison utilities.
# Origin: experimental_comparison.py
from dpf.validation.experimental_comparison import (  # noqa: F401
    _find_first_peak,  # internal peak-finder used by validate_current_waveform
    device_to_config_dict,  # build solver config dict from an ExperimentalDevice
    normalized_rmse,  # NRMSE over a full waveform window
    nrmse_peak,  # NRMSE restricted to the peak region
    validate_current_waveform,  # scalar pass/fail metrics vs measured I(t)
    validate_neutron_yield,  # scalar pass/fail metric vs measured Yn
)

# ExperimentalDevice dataclass definition.
# Origin: experimental_device.py
from dpf.validation.experimental_device import ExperimentalDevice  # noqa: F401

# Device instances, active registry, exclusion registry, and provenance filter.
# Origin: experimental_devices.py
from dpf.validation.experimental_devices import (  # noqa: F401
    _REFERENCE_ONLY,  # Exclusion registry — not for active validation
    AECS_PF2_DATA,  # Syrian device — _REFERENCE_ONLY (Type 2 unfittable)
    DEVICES,  # Active validation registry (9 devices)
    FAETON_DATA,  # FAETON-I 100 kV (Fuse Energy; Damideh 2025)
    MJOLNIR_DATA,  # MJOLNIR 100 kV / 1 MJ (LLNL; Schmidt 2021)
    NX2_DATA,  # NX2 compact Mather-type (NIE Singapore; Lee & Saw 2008)
    PF1000_16KV_DATA,  # PF-1000 at 16 kV / 1.05 Torr (Akel 2021)
    PF1000_20KV_DATA,  # PF-1000 at 20 kV / 2.0 Torr (voltage-scaled estimate)
    PF1000_DATA,  # PF-1000 at 27 kV / 3.5 Torr (IPPLM Warsaw; Scholz 2006)
    PF1000_GRIBKOV_DATA,  # PF-1000 at 27 kV, Gribkov (2007) shot / IPFS waveform
    POSEIDON_60KV_DATA,  # POSEIDON at 60 kV / 156 uF (Lee & Saw 2014)
    POSEIDON_DATA,  # POSEIDON at 40 kV — _REFERENCE_ONLY (Herold 1989 not on disk)
    UNU_ICTP_DATA,  # UNU-ICTP PFF training device (Lee & Saw 2014)
    get_devices_by_provenance,  # Filter DEVICES by waveform_provenance field
)

# Electrode and RLC diagnostics.
# Origin: experimental_diagnostics.py
from dpf.validation.experimental_diagnostics import (  # noqa: F401
    _S_OPTIMAL_KA_CM_TORR,  # deprecated alias for S_OPTIMAL — kept for external callers
    _S_TYPICAL_PF1000,  # PF-1000 reference speed factor [kA/(cm·Torr^0.5)]
    compute_bare_rlc_timing,  # quarter-period and damping from C, L, R
    compute_lp_l0_ratio,  # plasma inductance fraction L_p/L0
    compute_speed_factor,  # S = I_pinch / (a * sqrt(p0)) [kA/(cm·Torr^0.5)]
    lp_l0_for_device,  # L_p/L0 for a named device in DEVICES
)

# ---------------------------------------------------------------------------
# kr_status_filter — filter both registries by KnowledgeReference status
# ---------------------------------------------------------------------------

def kr_status_filter(status: str) -> dict[str, ExperimentalDevice]:
    """Return all devices (from both DEVICES and _REFERENCE_ONLY) matching *status*.

    Args:
        status: One of:
            ``"verified"``       — parameters cross-checked against a KnowledgeReference file.
            ``"unverified"``     — KR source exists but not yet line-by-line verified.
            ``"reference_only"`` — no valid KR source; device excluded from active validation.

    Returns:
        Combined dict of ``device_name -> ExperimentalDevice`` from whichever
        registry (or both) contains devices with ``kr_status == status``.

    Examples::

        # Devices safe to cite in a validation paper
        verified = kr_status_filter("verified")

        # Devices still awaiting KR cross-check
        todo = kr_status_filter("unverified")

        # Paper-only devices; never use for pass/fail claims
        refs = kr_status_filter("reference_only")
    """
    all_devices = {**DEVICES, **_REFERENCE_ONLY}
    return {name: dev for name, dev in all_devices.items() if dev.kr_status == status}
