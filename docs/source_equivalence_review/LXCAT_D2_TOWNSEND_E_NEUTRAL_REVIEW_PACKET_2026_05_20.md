# LXCat D₂ Townsend / e-Neutral — Source-Equivalence Review Packet (2026-05-20)

Sprint: Sprint 6 WS4 — Source-Equivalence Review Queue.
**This packet does NOT grant source-equivalence.** It queues the
substitution / cross-check question for Codex + external-team review.

## 1. Blocker context

- Primary blockers: `STARTUP-BVP-CH02`, `CLOSURE-BLK-D2-EN-001`
- Primary sources the substitute is being proposed against:
  - `STARTUP-BVP-CH02`: Raizer, Yu. P. (1991) *Gas Discharge Physics*, 2nd
    ed., Springer-Verlag, §4 (Townsend ionization coefficient α, Paschen A/B
    constants for D₂).
  - `CLOSURE-BLK-D2-EN-001`: Itikawa, Y. & Mason, N. (2005) *J. Phys. Chem.
    Ref. Data* 34(1):1–22, DOI 10.1063/1.1799251 (D₂ electron–neutral momentum-
    transfer cross-section σ_m(ε) as a function of electron energy ε).
- V2 blocker-resolution ledger status today (HEAD 97ebd94):
  - `STARTUP-BVP-CH02`: `external_acquisition_required`,
    `accepted_runtime_claim = false`,
    `can_support_first_principles_acceptance = false`.
    Remaining action: "acquire Raizer or ingest LXCat dataset."
  - `CLOSURE-BLK-D2-EN-001`: `external_acquisition_required`,
    `accepted_runtime_claim = false`,
    `can_support_first_principles_acceptance = false`.
    Remaining action: "acquire Itikawa Mason or LXCat dataset."

## 2. Proposed substitute / cross-check

- Identity: LXCat — The Plasma Data Exchange Project (lxcat.net). Specific
  datasets relevant to these blockers:
  - **Phelps database** (A. V. Phelps / JILA): H₂ and D₂ cross sections
    compiled from Phelps 1968, 1990, and associated JILA reports.
  - **Morgan database** (W. L. Morgan): collects electron-impact cross
    sections from multiple literature sources for H₂/D₂.
  - **Biagi database** / BSR (S. Biagi): computed Boltzmann-solver-derived
    cross sections for H₂; D₂ derived via isotope substitution rules.
  - LXCat version: web platform; datasets are versioned at download time.
    Current platform: LXCat3 (launched 2022; replaces legacy LXCat2).
- Acquisition path: https://www.lxcat.net — free account registration
  required; datasets downloadable in Siglo/LXCat native format and plain
  ASCII. H₂ and D₂ datasets listed under "Electron Scattering" section.
- Acquisition cost: $0 (free account required; no institutional license).
- License/redistribution constraint: LXCat datasets carry per-contributor
  terms; most Phelps/Morgan datasets allow academic redistribution with
  citation; BSR/Biagi datasets carry a JILA/university attribution
  requirement. Exact terms must be confirmed at download time from the
  per-dataset license displayed on lxcat.net.
- Verified live or installable on 2026-05-20: **yes** — lxcat.net was
  confirmed accessible via HTTP 200 by the Sprint 5 WS2 x-ray agent
  (documented in `docs/SPRINT5_FREE_ACQUISITIONS_2026_05_20.md` §2).

## 3. Source-equivalence argument (the case FOR)

- **Electron cross-sections are nuclear-mass-independent at the energies of
  interest.** Electron–molecule cross-sections for momentum transfer,
  ionization, and vibrational excitation are determined by the molecular
  electronic structure, not nuclear mass. At breakdown-regime electron
  energies (1 – 100 eV), the Born–Oppenheimer approximation holds and the
  H₂–D₂ cross-section difference arises only through:
  (a) vibrational energy spacing (D₂ ≈ H₂/√2 due to heavier reduced mass),
  which affects vibrational-excitation thresholds and near-threshold behaviour
  by at most a few percent; and (b) rotational-state distribution at the fill
  temperature, which is negligible for the Townsend α coefficient at field
  strengths E/n > 10 Td relevant to PF-1000 breakdown.
- **Itikawa & Mason 2005** (the primary source for `CLOSURE-BLK-D2-EN-001`)
  is itself a compiled recommendation; its D₂ data draw on the same
  experimental measurements (Crompton, Elford, Schmidt, Buckman groups) that
  are archived in LXCat Phelps and Morgan databases. The Itikawa–Mason 2005
  recommended cross-section set for D₂ was benchmarked against the LXCat
  precursor database at the time of publication.
- **LXCat is used as the community standard** for electron-impact cross
  sections in discharge modelling. Pitchford et al. (2017) *Plasma Sources
  Sci. Technol.* 26:055010 (the LXCat platform paper) identifies the Phelps
  and Morgan datasets as the de facto standard for H₂/D₂ swarm parameter
  reproduction. Review article Karwasz et al. (2022) *Electron Scattering in
  Gases* (KR: `electron-scattering-in-gases-from-cross-sections-to-plasma-
  modeling.md`) cites LXCat as the canonical source for plasma-modelling cross
  sections.
- **LXCat + Raizer §4 linkage:** The Phelps H₂ dataset reproduces the
  Townsend α vs E/n curve that Raizer §4 tabulates. The same Paschen minimum
  in D₂ (~3.5 × 10⁻⁴ Torr·cm, shifted from H₂ by √2 mass ratio) is
  consistent with both Raizer §4 and the LXCat-derived swarm parameters
  computed from the Phelps cross-section set.
- **Industry practice:** NRL Plasma Formulary (2019), cited in KR as
  `2019nrlplasma-formulary-037290d4.md`, does not tabulate D₂ Townsend
  parameters directly but references the LXCat/Phelps dataset family as the
  standard for low-temperature discharge modelling.

## 4. Source-difference argument (the case AGAINST)

- **Dataset version and provenance are not locked at acquisition time.**
  LXCat datasets are curator-maintained and can be updated between visits.
  A cross-section set downloaded in 2026 may not match the dataset a reviewer
  used in 2024. Without pinning the exact version hash (no hash mechanism
  exists at present on lxcat.net), the ingested data cannot be reproduced
  deterministically from the acquisition URL alone.
- **Phelps vs Morgan vs Biagi: three different σ_m(ε) sets.** The three LXCat
  database families do not agree everywhere. Biagi (BSR computed) tends to
  show a momentum-transfer cross-section peak near 3 eV that is 10–15% lower
  than Phelps for H₂; in D₂ via isotope substitution the disagreement is of
  similar magnitude. Which dataset to use for the primary source target
  extraction is not self-evident, and the choice propagates directly into the
  computed Townsend α and transport closure.
- **Itikawa & Mason 2005 is a critical evaluation, not a database.** The
  primary source for `CLOSURE-BLK-D2-EN-001` includes recommended uncertainty
  bounds and identifies which measurements to trust and why. LXCat provides
  individual datasets without the same critical-evaluation structure. A
  LXCat ingest does not automatically reproduce the Itikawa–Mason uncertainty
  tables or the recommended-set selection logic.
- **H₂ → D₂ mass-scaling approximation.** LXCat Phelps and Morgan datasets
  for D₂ are not fully independent experimental measurements of D₂; some
  entries are taken directly from H₂ with mass-scaling applied (ε_threshold
  shifted by M_D₂/M_H₂ = 2). This approximation is well-supported at energies
  above 5 eV but its accuracy for low-energy vibrational-excitation thresholds
  (1–4 eV) has not been formally quantified in the LXCat metadata.
- **Raizer §4 is a textbook derivation, not a database.** Raizer derives
  Townsend α from first principles (ionization integral over electron energy
  distribution) and tabulates A/B constants for several gases. LXCat provides
  raw σ(ε); the user must separately run a Boltzmann solver (e.g., BOLSIG+)
  to derive α(E/n) from the LXCat cross sections. The conversion chain
  introduces solver choice and convergence as additional uncertainty sources
  not present in the Raizer tabulation.
- **Scope overlap between two blockers.** LXCat Phelps/Morgan covers both
  `STARTUP-BVP-CH02` (Townsend α, Paschen) and `CLOSURE-BLK-D2-EN-001`
  (momentum-transfer σ_m). Accepting a single LXCat ingest as closing both
  blockers requires the reviewer to confirm that the same dataset family
  is appropriate for both the discharge-initiation (CH02) and the transport-
  closure (D2-EN-001) contexts, which have different energy-weighting regimes.

## 5. Explicit non-acceptance posture

- accepted_runtime_claim: false
- can_support_first_principles_acceptance: false
- source_equivalence_granted: false
- proposed_lane: candidate_substitute (for both `STARTUP-BVP-CH02` and
  `CLOSURE-BLK-D2-EN-001`, subject to the decision list in §6)

## 6. Decision list — what would need to be true for acceptance

1. **Dataset version locked.** An LXCat download must be archived as a
   SHA-256-verified flat file in the repo (or attached to the KR record)
   so that the exact data ingested is reproducible. The acquisition date
   and the displayed dataset version string must be recorded.
2. **Dataset family chosen and justified.** The reviewer must explicitly
   select one of {Phelps, Morgan, Biagi-BSR} for D₂ and document why that
   family is preferred over the others for the PF-1000 breakdown regime
   (E/n range 10–200 Td, gas temperature ~300 K at fill).
3. **Boltzmann-solver path documented for CH02.** If LXCat cross-sections
   are used to derive Townsend α(E/n), the solver (e.g., BOLSIG+ or
   LoKI-B), version, and convergence criterion must be recorded and
   reproducible. The derived α(E/n) table must be compared against the
   Paschen minimum from Raizer §4 at the D₂ mass-scaling prediction.
4. **Itikawa–Mason critical-evaluation delta documented.** For
   `CLOSURE-BLK-D2-EN-001`, the reviewer must confirm that the LXCat
   σ_m(ε) for D₂ agrees with the Itikawa–Mason 2005 recommended set within
   their stated uncertainty bounds (typically ±15–20% for momentum transfer
   at low energies). Any exceedances must be flagged as scope mismatches.
5. **H₂ → D₂ scaling approximation quantified.** The reviewer must document
   the energy range over which the mass-scaling approximation is applied in
   the chosen LXCat dataset and confirm the approximation error is acceptable
   for the PF-1000 breakdown energy range.
6. **KR target extraction completed.** The downloaded dataset must be ingested
   as a KR record (with SHA-256, acquisition date, dataset version, and
   source attribution) before the blocker can move to
   `existing_kr_target_extraction_pending`.

Codex pre-conditions:
- Codex must confirm that the Boltzmann-solver chain from LXCat σ(ε) to
  Townsend α(E/n) is implemented (or externally computed) and that the
  result is consumed by the correct runtime module in
  `src/dpf/first_principles/startup_bvp.py`.
- Codex must confirm that the transport-closure module consuming σ_m for
  `CLOSURE-BLK-D2-EN-001` does not silently fall back to an NRL-Formulary
  approximation when the LXCat table is loaded.

External-team pre-conditions:
- The external team must independently verify the H₂ → D₂ mass-scaling
  approximation error is within the acceptance tolerance for the
  startup-BVP physics claim.
- The external team must confirm which LXCat dataset family (Phelps, Morgan,
  or Biagi-BSR) is appropriate for the PF-1000 operating regime.

Executable pre-conditions:
- A reproducible download script must exist in `scripts/` that fetches and
  SHA-256-verifies the chosen LXCat dataset at a pinned URL or API endpoint.
- If BOLSIG+ is used: `scripts/run_bolsig_d2.py` (or equivalent) must be
  committed and must reproduce the Townsend α table from the locked LXCat
  dataset.

## 7. Recommended next action

- Queue for Sprint 7 source-equivalence review session. Do not ingest LXCat
  as a KR target-extracted equivalent in this sprint.
- Assign one team member to register at lxcat.net, download the Phelps D₂
  dataset, and produce a SHA-256-verified archive prior to the Sprint 7
  review session.
- The Sprint 7 review session should explicitly vote on {Phelps, Morgan,
  Biagi-BSR} dataset selection before any KR ingestion begins.
- `STARTUP-BVP-CH02` and `CLOSURE-BLK-D2-EN-001` remain
  `external_acquisition_required` with `accepted_runtime_claim = false`
  until the acceptance conditions above are met at a single reviewed commit.
