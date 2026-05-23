# SS12 P1 Phase 2 Source Packet Matrix Design

Date: 2026-05-22 UTC
Scope: PF-1000 full-energy P1 source closure
Validation scope: `pf1000_full_energy_27_to_40_kv`
Selected-machine source scope: `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`

## Purpose

Define the fail-closed, machine-readable source packet matrix used before any physics acceptance implementation. This matrix is the bridge between HeliosMatrix_KB retrieval, `/Users/anthonyzamora/PDFs`, repository `KnowledgeReference/`, and DPF Unified runtime validation packets.

HeliosMatrix_KB may suggest candidates. It is not source authority by itself. Authority requires local, reviewed source text or PDF-derived text with path and line/page provenance.

## Output files

Recommended Phase 2 artifacts:

- `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_2026_05_22.md`
- `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_2026_05_22.json`
- later code-facing artifact, after review: `data/first_principles/pf1000_full_energy_source_packet.json`

## Matrix schema

```json
{
  "packet_id": "pf1000_full_energy_same_scope_source_packet",
  "generated_at": "2026-05-22T00:00:00Z",
  "validation_scope": "pf1000_full_energy_27_to_40_kv",
  "selected_source_scope": "pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source",
  "authority_rule": "local_reviewed_sources_only",
  "acceptance_boundary": {
    "accepted_runtime_claim": false,
    "can_support_first_principles_acceptance": false,
    "promotes_acceptance": false
  },
  "channels": [
    {
      "channel": "geometry",
      "status": "candidate|accepted|blocked|cross_scope_context|not_found",
      "scope_match": "same_scope|partial_same_scope|cross_scope|unknown",
      "observables": [
        {
          "name": "anode_radius_m",
          "value": null,
          "unit": "m",
          "uncertainty": null,
          "source_refs": [
            {
              "source_path": "KnowledgeReference/example.md",
              "line_start": 1,
              "line_end": 2,
              "pdf_path": "/Users/anthonyzamora/PDFs/.../example.pdf",
              "page": null,
              "quote": "short exact quote",
              "review_status": "candidate|reviewed|accepted|rejected"
            }
          ]
        }
      ],
      "blocked_reason": null,
      "notes": ""
    }
  ],
  "global_blockers": [],
  "review_certificate": {
    "status": "blocked",
    "reason": "independent review not complete"
  }
}
```

## Required channels

Every channel must be present in the matrix. Missing evidence is not omission; it is an explicit `blocked` or `not_found` row.

1. `geometry`
2. `bank_circuit`
3. `gas_fill`
4. `current_waveform`
5. `startup`
6. `density_history`
7. `em_field_history`
8. `temperature_or_distribution_history`
9. `neutron_scalar_yield`
10. `neutron_timing`
11. `neutron_spectrum`
12. `neutron_anisotropy`
13. `detector_response`
14. `uncertainty_budget`
15. `review_certificate`

## Status rules

- `accepted`: same-scope, line-cited, reviewed, typed observable extracted, uncertainty handled or explicitly bounded.
- `candidate`: plausible but not yet reviewed or not yet typed.
- `partial_same_scope`: same facility/paper family but not enough to prove exact selected-machine source scope.
- `cross_scope_context`: useful context only; cannot close a gate.
- `blocked`: required evidence absent, ambiguous, cross-scope, raw/unreviewed, or lacking uncertainty/review.
- `not_found`: searched and no local evidence found.

## Fail-closed rules

- A channel is accepted only if all required observables in that channel are accepted.
- The packet cannot support first-principles acceptance unless every required channel is accepted and the certificate is accepted.
- Cross-scope sources may inform blockers or transfer-rule proposals only; they cannot silently fill same-scope evidence.
- Imported-PIC/Bennett/LLNL-like material remains context unless an explicit reviewed transfer rule exists.
- Raw PDF existence is not evidence; there must be extracted, line/page-cited content.
- Helios retrieval hits must be traced back to local KnowledgeReference text or PDF paths.

## Candidate source priority from initial reconnaissance

### Primary same-scope / near-same-scope candidates

1. `KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.md`
   - strongest candidate for full-energy large-electrode PF-1000 source scope
   - candidate channels: geometry, bank/circuit, gas/fill, current, neutron scalar yield, anisotropy, detector response, uncertainty/calibration

2. `KnowledgeReference/pf-1000-device-a2d6bc15.md`
   - same facility / Scholz 2000 baseline
   - candidate channels: baseline geometry, bank, circuit, diagnostics, startup, current waveform, X-ray/EM diagnostics

### Same-family but likely cross-scope candidates

3. `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
   - candidate channels: geometry, bank, current, density/EM, diagnostics, startup
   - likely same-family/cross-scope due later publication and mixed electrode variants

4. `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
   - candidate channels: current, neutron yield/timing, diagnostics, uncertainty/calibration, D2 pressure
   - likely same-family/cross-scope

5. `KnowledgeReference/scholz-2006-pf1000-mega-joule.md`
   - candidate channels: status/review context, diagnostics overview, density/current/neutron timing
   - secondary/cross-scope unless exact scope can be established

6. `KnowledgeReference/final-stages-of-the-plasma-column-evolution-in-the-plasma-focus-pf1000-device-plasma-scien-fa128cfd.md`
   - candidate channels: density, plasma column dynamics, continuum/bremsstrahlung, temperature proxy
   - cross-scope until exact scope is proven

7. `KnowledgeReference/sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md`
   - candidate channels: density history and timing diagnostics
   - cross-scope/later diagnostic source

8. `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`
   - candidate channels: current sheath, magnetic probes, current diagnostics, uncertainty
   - cross-scope because geometry differs

## Extraction workflow

1. Query HeliosMatrix_KB for each channel and candidate source.
2. Resolve hits to local source files under `KnowledgeReference/` or `/Users/anthonyzamora/PDFs`.
3. Read exact line ranges from markdown/text sources.
4. If only PDF exists, extract text into a reviewed markdown/source note before using it as evidence.
5. Fill matrix row with exact quote and line range.
6. Assign scope match.
7. Assign status.
8. Run matrix validation tests.

## Recommended validation checks

- JSON schema validation: every required channel appears exactly once.
- No `accepted` row without at least one source ref.
- No `accepted` row with `scope_match != same_scope`.
- No `accepted` row with `review_status != accepted`.
- No top-level acceptance flags true.
- All blocked rows have `blocked_reason`.
- Every source path exists locally.
- Markdown line ranges resolve and quote text matches source lines.

## Immediate next task

Generate the preliminary matrix JSON with all 15 required channels and candidate sources above. Keep all rows `candidate`, `partial_same_scope`, `cross_scope_context`, `blocked`, or `not_found` until exact line-cited review upgrades them.
