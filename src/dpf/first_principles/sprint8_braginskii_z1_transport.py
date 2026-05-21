"""Sprint 8 WS5 -- Braginskii 1965 Z=1 transport candidate closure packet.

This module encodes the render-verified Braginskii (1965) Z=1 parallel
transport closure -- electrical resistivity and electron/ion thermal
conductivity -- as a typed *candidate* source packet.

Source authority
----------------
Braginskii, S. I. (1965). "Transport processes in a plasma", in
M. A. Leontovich (ed.), *Reviews of Plasma Physics, Vol. 1*, Consultants
Bureau, New York, pp. 205-311 (trans. H. Lashinsky).

- PDF on disk (acquisition / line-page verification only):
  ``archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf``
  SHA-256 ``9687440676b43b02758373ef83d5a471730e9ca4bc207cb98f248fcafcb9d404``.
- KR target-extracted Table 2 packet:
  ``KnowledgeReference/braginskii-1965-table-2-TARGET-EXTRACTED.md``.
- Sprint 6 WS3 typed Table-2 extraction:
  :data:`dpf.first_principles.sprint6_braginskii_table2_target_extraction.BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION`.

Sprint 8 WS5 equation-level render verification
-----------------------------------------------
Sprint 6 WS3 promoted Table 2 only and left Eqs. 4.30-4.45 BLOCKED. Sprint 8
WS5 re-rendered the equation region with PyMuPDF (``fitz``) at 420 dpi from
the on-disk PDF and render-verified the closure equations:

- Journal p.249 (right half of PDF p.25 2-up spread): Eqs. 4.30-4.38.
- Journal p.250 (left half of PDF p.26 2-up spread): Eqs. 4.39-4.42.
- Journal p.252-253 (PDF p.27 2-up spread): Eqs. 4.43-4.45.

The 2-up spread caveat from Sprint 6 holds: each PDF page renders two
consecutive journal pages; the equation citations below carry the journal
page, not the PDF page.

Render-verified equations consumed by this packet
-------------------------------------------------
- **Eq. 4.34** (journal p.249) -- electron friction coefficients::

      alpha_par  = (m_e n_e / tau_e) * alpha_0
      alpha_perp = (m_e n_e / tau_e) * (1 - (alpha_1' x^2 + alpha_0') / Delta)
      alpha_wedge= (m_e n_e / tau_e) * x (alpha_1'' x^2 + alpha_0'') / Delta

- **Eq. 4.37** (journal p.249) -- electron thermal conductivities::

      kappa_par^e  = (n_e T_e tau_e / m_e) * gamma_0
      kappa_perp^e = (n_e T_e tau_e / m_e) * (gamma_1' x^2 + gamma_0') / Delta
      kappa_wedge^e= (n_e T_e tau_e / m_e) * x (gamma_1'' x^2 + gamma_0'') / Delta

- **Eq. 4.38** (journal p.249) -- ``x = omega_e tau_e``,
  ``Delta = x^4 + delta_1 x^2 + delta_0``.

- **Eq. 4.40** (journal p.250) -- ion thermal conductivities::

      kappa_par^i  = 3.906 n_i T_i tau_i / m_i
      kappa_perp^i = (n_i T_i tau_i / m_i)(2 x^2 + 2.645) / Delta_i
      kappa_wedge^i= (n_i T_i tau_i / m_i) x (5/2 x^2 + 4.65) / Delta_i

  with ``x = omega_i tau_i``, ``Delta_i = x^4 + 2.70 x^2 + 0.677``.

In Braginskii's equations ``T`` is in *energy units* (i.e. ``k_B T``); the
``q`` closure is ``q = -kappa grad T``. The runtime helpers below convert to
SI ``W/(m K)`` by carrying the explicit ``k_B`` factor (heat flux vs a
temperature gradient stated in kelvin).

Resistivity from the friction coefficient
-----------------------------------------
Eq. 4.30 (journal p.249) gives the friction force
``R_u = -alpha_par u_par - alpha_perp u_perp + alpha_wedge [h x u]`` with
``u = V_e - V_i``. The current density is ``j = -e n_e u``, so for the
parallel branch ``u_par = -j_par / (e n_e)`` and the parallel friction
electric field is ``e n_e E_par = R_u,par = alpha_par j_par / (e n_e)``.
Hence the **parallel electrical resistivity** is::

      eta_par = alpha_par / (e^2 n_e^2)
              = m_e alpha_0 / (e^2 n_e tau_e)        [Ohm m]

using ``alpha_par = (m_e n_e / tau_e) alpha_0`` (Eq. 4.34). For Z=1,
``alpha_0 = 0.5129`` (Table 2, journal p.251, column Z=1).

Acceptance posture -- explicit non-promotion
--------------------------------------------
This is a *candidate* source-backed closure. It is engineering evidence
only. Per the Super-Sprint 8 guardrails:

- ``accepted_runtime_claim = False``
- ``can_support_first_principles_acceptance = False``

A candidate closure may run engineering cases. It can never set acceptance.
Runtime acceptance of the transport closure still requires, at one commit:
(i) numerical-fidelity tests against a same-scope reference, (ii) a
same-scope comparator run, and (iii) a certificate-gate pass for the broader
transport-closure chain. None of those are claimed here.

PlasmaPy is used ONLY as a pinned cross-check lane
(``scripts/plasmapy_braginskii_z1_crosscheck.py``); it is never source
authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dpf.constants import e, k_B, m_d, m_e

__all__ = [
    "BRAGINSKII_Z1_RENDER_EVIDENCE",
    "BRAGINSKII_Z1_TABLE2_COEFFICIENTS",
    "BRAGINSKII_Z1_REVIEW_REQUIRED_CELLS_UNAVAILABLE",
    "BRAGINSKII_Z1_TRANSPORT_PACKET",
    "braginskii_z1_parallel_resistivity",
    "braginskii_z1_electron_parallel_conductivity",
    "braginskii_z1_ion_parallel_conductivity",
    "braginskii_z1_transport_source_packet",
]

# ---------------------------------------------------------------------------
# Render evidence (Sprint 6 WS3 manifest + Sprint 8 WS5 equation re-render).
# ---------------------------------------------------------------------------
BRAGINSKII_Z1_RENDER_EVIDENCE: Mapping[str, Any] = {
    "pdf_path": (
        "archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf"
    ),
    "pdf_sha256": (
        "9687440676b43b02758373ef83d5a471730e9ca4bc207cb98f248fcafcb9d404"
    ),
    "kr_target_extraction_path": (
        "KnowledgeReference/braginskii-1965-table-2-TARGET-EXTRACTED.md"
    ),
    "render_manifest_path": (
        "docs/extractions/braginskii_1965_render_evidence/render_manifest.json"
    ),
    "table_2_render_path": (
        "docs/extractions/braginskii_1965_render_evidence/"
        "pdf_p026_journal_p250_p251.png"
    ),
    "table_2_journal_page": 251,
    # Sprint 8 WS5 equation-level render verification.
    "equations_4_30_to_4_45_render_verified": True,
    "equation_render_pages": {
        "eqs_4_30_to_4_38_journal_page": 249,
        "eqs_4_39_to_4_42_journal_page": 250,
        "eqs_4_43_to_4_45_journal_pages": (252, 253),
    },
    "two_up_spread_caveat": (
        "Each PDF page is a 2-up scanned spread of two consecutive journal "
        "pages; equation citations carry the journal page, not the PDF page."
    ),
    "renderer": "PyMuPDF (fitz) 1.27.2.3; Table 2 at 200 dpi, equations re-rendered at 420 dpi",
}

# ---------------------------------------------------------------------------
# Braginskii 1965 Table 2, column Z=1 (journal p.251, render-verified).
# Verbatim string values from the Sprint 6 WS3 target extraction.
# ---------------------------------------------------------------------------
BRAGINSKII_Z1_TABLE2_COEFFICIENTS: Mapping[str, str] = {
    # Spot transport coefficients (used by Eqs. 4.34/4.35/4.37).
    "alpha_0": "0.5129",  # electron parallel friction -> parallel resistivity
    "beta_0": "0.7110",  # electron parallel thermoelectric coefficient
    "gamma_0": "3.1616",  # electron parallel thermal conductivity
    "delta_0": "3.7703",  # Delta polynomial constant term (Eq. 4.38)
    "delta_1": "14.79",  # Delta polynomial x^2 coefficient (Eq. 4.38)
    # Perpendicular / wedge polynomial coefficients (Eqs. 4.34/4.37).
    "alpha_1_prime": "6.416",
    "alpha_0_prime": "1.837",
    "alpha_1_double_prime": "1.704",
    "alpha_0_double_prime": "0.7796",
    "beta_1_prime": "5.101",
    "beta_0_prime": "2.681",
    "beta_1_double_prime": "3/2",
    "beta_0_tilde_double_prime": "3.053",
    "gamma_1_prime": "4.664",
    "gamma_0_prime": "11.92",
    "gamma_1_double_prime": "5/2",
    "gamma_0_double_prime": "21.67",
}

# Z=1 column entry for the table 2 spot row; cited per (coefficient, page).
_TABLE2_Z1_CITATION = (
    "Braginskii 1965 Table 2, journal p.251, column Z=1 "
    "(render: docs/extractions/braginskii_1965_render_evidence/"
    "pdf_p026_journal_p250_p251.png, right half)"
)

# ---------------------------------------------------------------------------
# Five review-required Table-2 cells (Sprint 6 WS3).  None are Z=1 column
# cells; they are tracked here as UNAVAILABLE so a consumer cannot silently
# read one.  alpha_0_double_prime at Z=3 is the anomalous-Z-trend cell.
# ---------------------------------------------------------------------------
BRAGINSKII_Z1_REVIEW_REQUIRED_CELLS_UNAVAILABLE: tuple[Mapping[str, Any], ...] = (
    {"coefficient": "alpha_0_prime", "z_column": 3, "status": "unavailable_review_required"},
    {"coefficient": "alpha_0_prime", "z_column": 4, "status": "unavailable_review_required"},
    {"coefficient": "alpha_0_prime", "z_column": "inf", "status": "unavailable_review_required"},
    {"coefficient": "alpha_0_double_prime", "z_column": 3, "status": "unavailable_review_required"},
    {"coefficient": "gamma_1_prime", "z_column": "inf", "status": "unavailable_review_required"},
)


def _z1(name: str) -> float:
    """Return a Z=1 Table-2 coefficient as a float, parsing rational strings."""

    raw = BRAGINSKII_Z1_TABLE2_COEFFICIENTS[name]
    if "/" in raw:
        num, _, den = raw.partition("/")
        return float(num) / float(den)
    return float(raw)


# Float views of the Z=1 coefficients consumed by the runtime helpers.
ALPHA_0_Z1: float = _z1("alpha_0")
GAMMA_0_Z1: float = _z1("gamma_0")
# Eq. 4.40 ion parallel conductivity numeric coefficient (render-verified,
# journal p.250); not a Table-2 cell -- it is printed in the equation itself.
ION_PARALLEL_CONDUCTIVITY_COEFF: float = 3.906


def braginskii_z1_parallel_resistivity(
    n_e: NDArray[np.float64] | float,
    tau_e: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Return the Braginskii Z=1 parallel electrical resistivity [Ohm*m].

    Implements ``eta_par = m_e * alpha_0 / (e^2 * n_e * tau_e)`` derived from
    Eq. 4.30 (friction force) and Eq. 4.34 (``alpha_par = (m_e n_e / tau_e)
    alpha_0``), Braginskii 1965, journal p.249, with ``alpha_0 = 0.5129``
    from Table 2 column Z=1 (journal p.251).

    Parameters
    ----------
    n_e:
        Electron number density [m^-3].
    tau_e:
        Electron collision time [s] (Braginskii definition).

    This is a CANDIDATE closure output. It is engineering evidence and never
    sets acceptance.
    """

    n_e_arr = np.asarray(n_e, dtype=np.float64)
    tau_e_arr = np.asarray(tau_e, dtype=np.float64)
    return m_e * ALPHA_0_Z1 / (e**2 * n_e_arr * tau_e_arr)


def braginskii_z1_electron_parallel_conductivity(
    n_e: NDArray[np.float64] | float,
    T_e_kelvin: NDArray[np.float64] | float,
    tau_e: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Return the Braginskii Z=1 electron parallel thermal conductivity.

    Implements Eq. 4.37 (Braginskii 1965, journal p.249)::

        kappa_par^e = gamma_0 * n_e * (k_B T_e) * tau_e / m_e

    with ``gamma_0 = 3.1616`` from Table 2 column Z=1 (journal p.251). The
    ``k_B`` factor converts the Braginskii energy-unit ``T`` form to an SI
    conductivity in ``W/(m*K)`` against a temperature gradient in kelvin.

    Parameters
    ----------
    n_e:
        Electron number density [m^-3].
    T_e_kelvin:
        Electron temperature [K].
    tau_e:
        Electron collision time [s].

    Returns
    -------
    Electron parallel thermal conductivity [W/(m*K)].

    CANDIDATE closure output -- engineering evidence only.
    """

    n_e_arr = np.asarray(n_e, dtype=np.float64)
    T_e_arr = np.asarray(T_e_kelvin, dtype=np.float64)
    tau_e_arr = np.asarray(tau_e, dtype=np.float64)
    return GAMMA_0_Z1 * n_e_arr * (k_B * T_e_arr) * tau_e_arr * k_B / m_e


def braginskii_z1_ion_parallel_conductivity(
    n_i: NDArray[np.float64] | float,
    T_i_kelvin: NDArray[np.float64] | float,
    tau_i: NDArray[np.float64] | float,
    m_i: float = m_d,
) -> NDArray[np.float64]:
    """Return the Braginskii ion parallel thermal conductivity [W/(m*K)].

    Implements Eq. 4.40 (Braginskii 1965, journal p.250)::

        kappa_par^i = 3.906 * n_i * (k_B T_i) * tau_i / m_i

    The ``3.906`` coefficient is render-verified verbatim in Eq. 4.40 (it is
    printed in the equation, not a Table-2 cell). The ``k_B`` factor converts
    to SI ``W/(m*K)``. Defaults ``m_i`` to the deuteron mass for the DPF
    deuterium-plasma Z=1 case.

    Parameters
    ----------
    n_i:
        Ion number density [m^-3].
    T_i_kelvin:
        Ion temperature [K].
    tau_i:
        Ion collision time [s].
    m_i:
        Ion mass [kg]; defaults to the deuteron mass.

    CANDIDATE closure output -- engineering evidence only.
    """

    n_i_arr = np.asarray(n_i, dtype=np.float64)
    T_i_arr = np.asarray(T_i_kelvin, dtype=np.float64)
    tau_i_arr = np.asarray(tau_i, dtype=np.float64)
    return (
        ION_PARALLEL_CONDUCTIVITY_COEFF
        * n_i_arr
        * (k_B * T_i_arr)
        * tau_i_arr
        * k_B
        / m_i
    )


# ---------------------------------------------------------------------------
# The typed Sprint 8 WS5 source packet.
# ---------------------------------------------------------------------------
BRAGINSKII_Z1_TRANSPORT_PACKET: Mapping[str, Any] = {
    "packet_id": "sprint8_ws5_braginskii_z1_transport_candidate_2026_05_20",
    "workstream": "WS5",
    "blocker_id": "CLOSURE-BLK-BRAG-001",
    "scope_tag": "generic_formulary",
    "charge_state": 1,
    "classification": "active_source_backed_candidate",
    # Acceptance boundary markers -- not negotiable by data content.
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "candidate_only_engineering_evidence": True,
    "render_evidence": dict(BRAGINSKII_Z1_RENDER_EVIDENCE),
    "table_2_z1_coefficients": dict(BRAGINSKII_Z1_TABLE2_COEFFICIENTS),
    "table_2_z1_citation": _TABLE2_Z1_CITATION,
    "review_required_cells_unavailable": [
        dict(cell) for cell in BRAGINSKII_Z1_REVIEW_REQUIRED_CELLS_UNAVAILABLE
    ],
    # Render-verified equation-level closure forms.
    "closure_outputs": {
        "parallel_electrical_resistivity": {
            "symbol": "eta_par",
            "in_code_form": "eta_par = m_e * alpha_0 / (e^2 * n_e * tau_e)",
            "derived_from": "Eq. 4.30 friction force + Eq. 4.34 alpha_par",
            "equation_journal_page": 249,
            "coefficient": "alpha_0",
            "coefficient_value": BRAGINSKII_Z1_TABLE2_COEFFICIENTS["alpha_0"],
            "units": "Ohm*m",
            "implementation": (
                "dpf.first_principles.sprint8_braginskii_z1_transport."
                "braginskii_z1_parallel_resistivity"
            ),
        },
        "electron_parallel_thermal_conductivity": {
            "symbol": "kappa_par_e",
            "in_code_form": (
                "kappa_par_e = gamma_0 * n_e * (k_B T_e) * tau_e * k_B / m_e"
            ),
            "derived_from": "Eq. 4.37 electron heat flux",
            "equation_journal_page": 249,
            "coefficient": "gamma_0",
            "coefficient_value": BRAGINSKII_Z1_TABLE2_COEFFICIENTS["gamma_0"],
            "units": "W/(m*K)",
            "implementation": (
                "dpf.first_principles.sprint8_braginskii_z1_transport."
                "braginskii_z1_electron_parallel_conductivity"
            ),
        },
        "ion_parallel_thermal_conductivity": {
            "symbol": "kappa_par_i",
            "in_code_form": (
                "kappa_par_i = 3.906 * n_i * (k_B T_i) * tau_i * k_B / m_i"
            ),
            "derived_from": "Eq. 4.40 ion heat flux",
            "equation_journal_page": 250,
            "coefficient": "3.906",
            "coefficient_value": "3.906",
            "coefficient_note": (
                "printed verbatim in Eq. 4.40; not a Table-2 cell"
            ),
            "units": "W/(m*K)",
            "implementation": (
                "dpf.first_principles.sprint8_braginskii_z1_transport."
                "braginskii_z1_ion_parallel_conductivity"
            ),
        },
        "electron_parallel_thermoelectric_coefficient": {
            "symbol": "beta_0",
            "derived_from": "Eq. 4.31 thermal force + Eq. 4.35 beta_par",
            "equation_journal_page": 249,
            "coefficient": "beta_0",
            "coefficient_value": BRAGINSKII_Z1_TABLE2_COEFFICIENTS["beta_0"],
            "units": "dimensionless_coefficient",
            "implementation": (
                "extracted_not_wired_thermoelectric_runtime_path_out_of_ws5_scope"
            ),
        },
    },
    "symbol_map": {
        "n_e": {"meaning": "electron number density", "unit": "m^-3"},
        "n_i": {"meaning": "ion number density", "unit": "m^-3"},
        "T_e": {"meaning": "electron temperature (kelvin in code)", "unit": "K"},
        "T_i": {"meaning": "ion temperature (kelvin in code)", "unit": "K"},
        "tau_e": {"meaning": "Braginskii electron collision time", "unit": "s"},
        "tau_i": {"meaning": "Braginskii ion collision time", "unit": "s"},
        "alpha_0": {"meaning": "Z=1 parallel friction coefficient", "unit": "-"},
        "gamma_0": {"meaning": "Z=1 electron parallel conductivity coefficient", "unit": "-"},
        "x": {"meaning": "Hall parameter omega tau", "unit": "-"},
        "Delta": {"meaning": "x^4 + delta_1 x^2 + delta_0 polynomial (Eq. 4.38)", "unit": "-"},
    },
    "validity_regime": {
        "valid_when": (
            "fully_ionized_z1_plasma",
            "classical_collisional_transport",
            "coulomb_log_much_greater_than_1",
        ),
        "parallel_branch_only": (
            "this WS5 packet wires the unmagnetized / parallel branch; the "
            "perpendicular and wedge (Hall) branches of Eqs. 4.34/4.37/4.40 "
            "are render-verified but not wired in WS5"
        ),
    },
    "cross_check_lane_not_authority": {
        "name": "plasmapy_classical_transport_braginskii",
        "script": "scripts/plasmapy_braginskii_z1_crosscheck.py",
        "source_equivalence_granted": False,
        "role": "pinned_cross_check_lane_never_source_authority",
    },
    "acceptance_gate_note": (
        "candidate source-backed closure; numerical-fidelity, same-scope "
        "comparator, and certificate gates remain BLOCKED and unclaimed"
    ),
}


def braginskii_z1_transport_source_packet() -> Mapping[str, Any]:
    """Return the Sprint 8 WS5 Braginskii Z=1 transport candidate packet.

    The returned packet enforces, non-negotiably:

    - ``accepted_runtime_claim = False``
    - ``can_support_first_principles_acceptance = False``

    It is a source-backed *candidate* closure: render-verified equations plus
    render-verified Z=1 Table-2 coefficients. It may run engineering cases. It
    can never set acceptance -- that requires separate numerical-fidelity,
    comparator, and certificate-gate work at one commit.
    """

    return dict(BRAGINSKII_Z1_TRANSPORT_PACKET)
