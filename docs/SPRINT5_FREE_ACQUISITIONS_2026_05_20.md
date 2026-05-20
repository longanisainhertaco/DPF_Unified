# Sprint 5 Free / Low-Cost Acquisition Opportunities (2026-05-20)

The x-ray pass found **6 of the V2 P1+P2 external-acquisition blockers are
free-downloadable today or have free machine-equivalent substitutes** that
the source-acquisition ledger did not flag.

This memo lists the URLs / paths. No physics is accepted here — these are
acquisition pointers, not source-verified target extractions. After download
the documents still require KR ingestion, target extraction, and runtime
consumption before any blocker moves toward `accepted_runtime_claim = true`.

## 1. Immediately free downloads (HTTP 200 verified)

| Blocker | Source | URL | Pages | Notes |
| --- | --- | --- | --- | --- |
| `CLOSURE-BLK-ANOM-001` (DPF LHI) | Bruzzone, H.; Bernal, L. (2001) *Nukleonika* 46:59-61 "The need of using anomalous resistivity due to Lower Hybrid Instabilities in plasma-magnetic field interfaces" | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46n2p059f.pdf | 3 | ICHTJ open-access Nukleonika archive |
| `CLOSURE-BLK-ANOM-001` (DPF LHI) | Bruzzone, H. (2001) *Nukleonika* 46 suppl.1:S3-S7 "The role of anomalous resistivities in Plasma Focus discharges" | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p003f.pdf | 5 | Companion paper; DPF-scope |
| `PF1000-BLK-009` (hardware-scope hollow-bore) | Szydlowski, A.; Scholz, M.; Karpinski, L.; Sadowski, M.; Tomaszewski, K.; Paduch, M.; Miklaszewski, R.; et al. (2001) *Nukleonika* 46 suppl.1:S61-S64 "Neutron and fast ion emission from PF-1000 facility equipped with new large electrodes" | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p061f.pdf | 4 | Title suggests it directly covers the "new large electrodes" geometry — likely candidate for the bore-radius hardware source |

**Combined effort: ~30 minutes to download all three.** All confirmed live at
HTTP 200 by the x-ray agent's WebFetch probes.

**What "free download" means and does not mean.** Per the Codex Sprint 5 WS2
audit (`docs/CODEX_SPRINT5_WS2_AUDIT_2026_05_20.md` finding A3): an
immediately-downloadable PDF *may close source-availability after acquisition,
KR ingestion, target extraction, and review*. It does **not** by itself close
the corresponding blocker for runtime acceptance. The acquisition pipeline is:
download → SHA-256 verification → KR text-parity ingestion → target-extracted
KR file → runtime citation → review. Until all five steps complete, the
blocker remains source-not-yet-extracted, not "closed."

## 2. Free machine-equivalent substitutes for paywalled sources

| Blocker | Paywalled source | Free substitute | URL |
| --- | --- | --- | --- |
| `NEUTRON-BLK-002` (deuteron stopping) | ICRU Report 49 (1993) — $50-80 from ICRU | **SRIM-2013** (Stopping and Range of Ions in Matter) — runs deuteron-on-D₂ Bethe-Bloch tables locally | http://www.srim.org |
| `NEUTRON-BLK-002` (deuteron stopping) | ICRU Report 49 (1993) | NIST PSTAR (online stopping calculator) | https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html |
| `NEUTRON-BLK-002` (deuteron stopping) | ICRU Report 49 (1993) | IAEA Nuclear Data Services stopping-power calculator | https://nds.iaea.org/stopping-legacy/stopping_202002/stopping_prog.html |
| `NEUTRON-BLK-004` (Brysk Doppler) | Brysk, H. (1973) *Plasma Phys.* 15:1282 | **Munro 2012** OSTI report (LLNL-JRNL-676641) re-derives the canonical FWHM = 82.5√(T_i[keV]) keV formula with citation | https://www.osti.gov/biblio/1240980 |
| `CLOSURE-BLK-BRAG-001` (Braginskii Z-table) | Braginskii (1965) — PDF on disk, render-verified | **PlasmaPy** `formulary.braginskii.ClassicalTransport` hardcoded Z=1,2,3,4,∞ coefficient tables (cross-validation reference) | https://docs.plasmapy.org/en/stable/formulary/braginskii.html |
| `STARTUP-BVP-CH02` (D₂ Townsend/Paschen) | Raizer 1991 §4 — ~$40/chapter Springer | **LXCat** Phelps/Morgan/Biagi D₂ cross-section datasets | https://www.lxcat.net (free account required) |
| `CLOSURE-BLK-D2-EN-001` (D₂ e-neutral) | Itikawa & Mason 2005 *JPCRD* 34:1 | LXCat (same as above; one acquisition resolves both) | https://www.lxcat.net |

## 3. Combined acquisition matrix update (V2 ledger context)

Before this x-ray: V2 source-acquisition ledger listed 13 `external_acquisition_required` rows with no free-source notes.

After this x-ray: **6 of those 13 rows have free or no-cost paths today.**

Recommended acquisition sequence for the next sprint (ordered by leverage):

1. **Download Miklaszewski 2001** (HTTP 200 confirmed) — 30 min. If it
   publishes a hardware-scope bore radius value, `PF1000-BLK-009` moves from
   `existing_kr_target_extraction_pending` to `existing_kr_source_supported`
   on the geometry side.
2. **Download both Bruzzone 2001 PDFs** (HTTP 200 confirmed) — 30 min. These
   are the primary DPF-scope LHI anomalous-resistivity references named in
   Auluck 2022 Theory Part 1 (ref [34]) and the Plasma Focus Update 2021
   bibliography (refs [352]/[353]). After acquisition, KR ingestion, target
   extraction, and review they *may close source availability* for
   `CLOSURE-BLK-ANOM-001` on the quantitative-formula side.
3. **Register at LXCat and download D₂ datasets** — 1-2 h. After ingestion +
   target extraction + review, *may close source availability* for two
   blockers (`STARTUP-BVP-CH02` and `CLOSURE-BLK-D2-EN-001`) in a single
   acquisition.
4. **Download SRIM-2013 + run D-on-D₂ stopping tables** — 2-4 h. **Candidate
   substitute** for ICRU 49 (`NEUTRON-BLK-002`); source-equivalence to ICRU 49
   for DPF-regime deuteron stopping is a review decision, not an automatic
   substitution.
5. **Download Munro 2012 OSTI report** — 1-2 h. **Re-export / cross-check
   lane** for the Brysk 1973 FWHM formula derivation (`NEUTRON-BLK-004`);
   accepted as a Brysk substitute only after source-equivalence review.
6. **PlasmaPy Braginskii cross-check** — 1-2 h. Use as second-witness against
   the render-verified Table 2 values for `CLOSURE-BLK-BRAG-001`; PlasmaPy is
   a **cross-check lane**, not a source-equivalent replacement for the
   primary Braginskii target extraction.

**Total Sprint 5 acquisition effort:** ~6-12 hours focused work; ~$0 cost
across all six items. After this, **6 of 13 V2 P1+P2 external acquisitions
may have closed source availability** (pending ingestion + target extraction
+ review per the pipeline above); 7 remain (Hagstrum 1956, Voronov 1997,
Janev-Smith 1993, Itikawa-Mason 2005 primary, Schmidt 2002, Schmidt 2012
PRL, Davidson-Gladd 1975) — each behind institutional access or
interlibrary loan.

## 4. What this does NOT change

- `can_support_first_principles_acceptance = False` everywhere.
- `accepted_runtime_claim = false` on every blocker-ledger row.
- The 4 `absent_from_literature` PF-1000 internal-geometry blockers
  (insulator outer radius / wall thickness, backplate radial extent / axial
  thickness) — those still require IPPLM facility outreach.
- The structural Te/Ti gap — **no accepted same-scope PF-1000 bulk pinch
  Te/Ti history exists for the selected certificate scope** (per Codex
  Sprint 5 WS2 audit finding A2). Bernard 1977 contains direct historical
  filament-phase Ti evidence, but it is wrong-scope for PF-1000 pinch
  validation. Plasma Focus Update 2021 contains PF-1000 local hot-spot Te
  method context (Zaloga 2018 filter-ratio with D₂+Ne admixture), but it
  is text-only / method context and not accepted as bulk same-scope Te
  validation.
- The 3D-runner-vs-acceptance-gate code gap — still requires code work, not
  data acquisition.

The acquisitions in this memo, after acquisition + KR ingestion + target
extraction + review, *may close source-availability* gaps. Runtime
acceptance still requires the downstream chain: KR target extraction →
runtime module consumption → numerical convergence → same-scope comparator
→ certificate gate. The Sprint 5 acquisitions are necessary but not
sufficient.

## 5. Verification posture

All URLs in §1 were probed via WebFetch and returned HTTP 200 in the x-ray
audit run (2026-05-20). The substitutes in §2 are open-access tools/calculators
with no paywall as of the audit date.

These URLs may rot; the source-acquisition ledger should be re-probed at
each sprint start. The blocker IDs cited here are authoritative; the URLs
are convenience pointers.
