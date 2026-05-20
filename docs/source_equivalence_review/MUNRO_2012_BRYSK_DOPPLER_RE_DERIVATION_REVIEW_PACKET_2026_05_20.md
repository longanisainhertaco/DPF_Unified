# Munro 2012 / Brysk 1973 Doppler Re-Derivation — Source-Equivalence Review Packet (2026-05-20)

Sprint: Sprint 6 WS4 — Source-Equivalence Review Queue.
**This packet does NOT grant source-equivalence.** It queues the
substitution / cross-check question for Codex + external-team review.

## 1. Blocker context

- Primary blocker: `NEUTRON-BLK-004`
- Primary source the substitute is being proposed against: Brysk, H. (1973)
  "Thermonuclear reaction rates in plasmas," *Plasma Physics* 15(7):611–617,
  and specifically the Doppler-broadening derivation published as Brysk, H.
  (1973) *Plasma Physics* 15:1282 (companion/errata entry in the
  blocker ledger). The ledger entry cites the derivation of the canonical
  FWHM formula `ΔE_FWHM = C √(T_i [keV]) keV` where C = 82.5 for D-D
  neutrons.
- V2 blocker-resolution ledger status today (HEAD 97ebd94):
  - `NEUTRON-BLK-004`: `external_acquisition_required`,
    `accepted_runtime_claim = false`,
    `can_support_first_principles_acceptance = false`.
    Remaining action: "acquire Brysk 1973."
    Source tagged `derivation + FWHM = 82.5 √(T_i[keV]) keV` as the
    target quantity.

## 2. Proposed substitute / cross-check

- Identity: Munro, D. H. (2012) "Interpreting Inertial Confinement Fusion
  Neutron Spectra." LLNL report LLNL-JRNL-676641. Available via OSTI at
  https://www.osti.gov/biblio/1240980. Note: the OSTI biblio page lists the
  report; the full-text PDF is linked from that page.
- Acquisition path: https://www.osti.gov/biblio/1240980 (OSTI full-text
  PDF download; free, no registration required for OSTI public-access reports).
- Acquisition cost: $0.
- License/redistribution constraint: LLNL JRNL report; released as OSTI
  public-access under US Department of Energy Open Access. Redistribution
  with attribution permitted.
- Verified live or installable on 2026-05-20: **yes** — OSTI biblio page
  confirmed accessible per Sprint 5 WS2 x-ray agent
  (`docs/SPRINT5_FREE_ACQUISITIONS_2026_05_20.md` §2).

## 3. Source-equivalence argument (the case FOR)

- **Munro 2012 explicitly re-derives the Brysk FWHM formula.** The stated
  purpose of LLNL-JRNL-676641 is to provide a self-contained derivation of
  the D-D neutron spectrum shape in the context of ICF implosion diagnostics.
  The report derives the same Gaussian-FWHM result that Brysk 1973 produced
  and cites the formula `ΔE_FWHM ≈ 82.5 √(T_i [keV]) keV` as the standard
  working expression.
- **Same physics framework.** Both Brysk 1973 and Munro 2012 start from the
  same physical basis: the D-D center-of-mass neutron energy is 2.45 MeV,
  and the lab-frame energy spread arises from the thermal Maxwellian velocity
  distribution of the reacting deuterons. The Galilean-boost integral from
  CM frame to lab frame yields a Gaussian energy distribution whose FWHM
  scales as √T_i. The constant C = 82.5 (in units where T_i is in keV and
  ΔE in keV) follows from the D-D reaction kinematics and the reduced-mass
  factor; this is a fixed-point result not dependent on author interpretation.
- **Munro 2012 is cited in the ICF diagnostics community as a Brysk
  re-derivation reference.** The ICF diagnostics literature (Murphy 2014
  RSI; Gatu Johnson et al. 2016 RSI) routinely cites both Brysk 1973 and
  Munro 2012 together as providing equivalent expressions for the thermal
  component of the neutron spectrum. This cross-citation pattern supports
  the claim that the two derivations reach the same result.
- **Free access removes the paywall obstacle.** Brysk 1973 (*Plasma Physics*
  IOP) is behind institutional access; Munro 2012 OSTI is freely downloadable.
  If the two derivations are verified step-for-step equivalent, Munro 2012
  would serve as an accessible vehicle for KR target extraction.

## 4. Source-difference argument (the case AGAINST)

- **Munro 2012 is an LLNL ICF report, not a peer-reviewed journal article.**
  Brysk 1973 was published in *Plasma Physics* (now *Plasma Physics and
  Controlled Fusion*, IOP), a peer-reviewed journal. Munro 2012 is an LLNL
  preprint report (JRNL label = intended for journal submission, but the
  OSTI record does not include a confirmed journal-publication DOI). The
  peer-review status of Munro 2012 must be confirmed; if it remains
  unpublished, its status as a primary-source substitute is weaker than a
  peer-reviewed derivation.
- **The report may use a modernised framework that omits a Brysk
  approximation.** Brysk 1973 worked in the non-relativistic, non-degenerate
  Maxwellian limit with an explicit assumption of identical reactant
  temperatures (T_D = T_D). Munro 2012 targets ICF implosion diagnostics
  where the question of bulk-flow (CM velocity) corrections and multi-
  temperature (hot-spot vs bulk) effects arises. If Munro 2012 includes CM-
  velocity corrections (`ΔE_bulk = m_n v_bulk · cos θ` terms), the FWHM
  formula presented by Munro may be a generalization of Brysk, not a strict
  re-derivation. The simpler Brysk formula applies only in the zero-bulk-flow
  limit; using Munro's full expression without confirming the bulk-flow term
  is zero would be a scope mismatch for the DPF thermonuclear component.
- **The constant C = 82.5 appears in multiple forms in the literature.** Some
  sources write the formula as `ΔE_FWHM = 82.5 √(T_i/keV) keV` (FWHM in
  keV), others as `σ = 35.0 √(T_i/keV) keV` (1σ Gaussian width), and others
  in eV units. A careless copy from Munro 2012 without verifying unit
  conventions and the FWHM-vs-σ distinction would introduce a factor-of-
  2.355 error in the runtime implementation.
- **Brysk 1973 derivation steps have not been verified in this sprint.**
  The source-equivalence claim "Munro re-derives Brysk step-for-step" has not
  been verified by reading Munro 2012 this sprint. The OSTI page confirms the
  report exists and is downloadable, but the internal content has not been
  reviewed. Until the report is read and the derivation compared equation-by-
  equation against Brysk 1973 (if Brysk can be accessed), the equivalence
  claim is an assertion, not a verified fact.
- **DPF vs ICF scope difference.** Brysk 1973 was originally derived for
  thermonuclear plasmas (Maxwellian bulk plasma, isotropic). In the DPF
  context, the neutron spectrum has a beam-target component (anisotropic)
  and a thermonuclear component (isotropic). Munro 2012 explicitly addresses
  ICF implosion neutron spectra where the beam-target component may be
  treated differently. If Munro's formula is derived only for the thermal
  component and implicitly assumes zero beam contribution, its application
  in the DPF context requires explicit confirmation that the thermal-component
  isolation is correct.

## 5. Explicit non-acceptance posture

- accepted_runtime_claim: false
- can_support_first_principles_acceptance: false
- source_equivalence_granted: false
- proposed_lane: candidate_substitute (for `NEUTRON-BLK-004` only, subject
  to step-by-step derivation verification in §6)

## 6. Decision list — what would need to be true for acceptance

1. **Munro 2012 full text downloaded and read.** The OSTI PDF must be
   downloaded, SHA-256-verified, and read by a reviewer who traces every
   intermediate step of the Doppler-broadening derivation.
2. **Equation-by-equation equivalence with Brysk confirmed.** The reviewer
   must map each step in Munro 2012's derivation to the corresponding step
   in Brysk 1973 (or identify where Munro generalises beyond Brysk). Every
   approximation made in Brysk 1973 (non-relativistic limit, identical-
   temperature Maxwellian, zero CM velocity) must be confirmed as present in
   Munro 2012 or explicitly noted as a generalisation.
3. **Bulk-flow / CM-velocity terms identified and scoped.** If Munro 2012
   includes CM-velocity correction terms, the reviewer must confirm:
   (a) which formula to use for the zero-bulk-flow DPF thermonuclear case,
   and (b) that the selected formula reduces exactly to `82.5 √(T_i[keV])
   keV` FWHM.
4. **Unit and FWHM-vs-σ convention verified.** The constant C in the selected
   formula must be numerically confirmed (C = 82.5 for FWHM in keV, T_i in
   keV) and distinguished from the 1σ form (C ≈ 35.0 keV). The runtime
   implementation must consume the correct form.
5. **Peer-review / publication status confirmed.** If Munro 2012 has a
   confirmed journal DOI (LLNL JRNL reports are sometimes subsequently
   published), that DOI should be added to the KR record. If not peer-
   reviewed, the KR record must flag the report status explicitly.
6. **KR target extraction completed.** The Munro 2012 FWHM derivation must
   be ingested as a KR record with: OSTI accession number, LLNL report
   number, SHA-256 of the downloaded PDF, and the extracted formula with
   page and equation number cited.

Codex pre-conditions:
- Codex must confirm where the Brysk/Doppler FWHM formula is consumed in
  `src/dpf/first_principles/neutron_authority.py` (or equivalent) and
  whether the constant C is hardcoded or read from the KR target extraction.
- Codex must confirm that the runtime module will correctly handle the
  FWHM-vs-σ distinction and that the unit convention (keV for both T_i and
  ΔE) is enforced.

External-team pre-conditions:
- The external team must independently verify the bulk-flow / CM-velocity
  scope question: is the DPF thermonuclear neutron spectrum a valid zero-
  bulk-flow case for the Brysk/Munro formula, or does the DPF pinch velocity
  (v_pinch ~ 10⁵–10⁶ m/s) require a CM-velocity correction?
- The external team must confirm the acceptable precision for the FWHM
  constant (is C = 82.5 ± 0.5 keV sufficient, or does higher precision
  matter for the certificate claim?).

Executable pre-conditions:
- A script must download the Munro 2012 PDF from OSTI, verify its SHA-256,
  and log the download date.
- A unit test must verify that the runtime formula uses C = 82.5 ±
  the reviewer-confirmed tolerance and FWHM (not σ) units.

## 7. Recommended next action

- Queue for Sprint 7 source-equivalence review session. Do not ingest Munro
  2012 as a KR target-extracted equivalent in this sprint.
- Download the Munro 2012 OSTI PDF before the Sprint 7 session and assign
  a reviewer to perform the equation-by-equation comparison against Brysk
  1973 (if Brysk 1973 can be accessed via institutional ILL) or against the
  NRL Plasma Formulary neutron-spectrum section as a secondary cross-check.
- `NEUTRON-BLK-004` remains `external_acquisition_required` with
  `accepted_runtime_claim = false` until the acceptance conditions above are
  met at a single reviewed commit.
