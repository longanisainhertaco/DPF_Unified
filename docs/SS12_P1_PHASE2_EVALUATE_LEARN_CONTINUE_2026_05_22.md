# SS12 P1 Phase 2 — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: P1 source-packet matrix extraction bootstrap

## Evaluate

Artifacts created:

- `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_DESIGN_2026_05_22.md`
- `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_2026_05_22.json`
- `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_EXTRACTED_2026_05_22.json`
- `tests/test_ss12_phase2_source_packet_matrix.py`

Verification from `/Users/anthonyzamora/dpf-unified`:

```text
.venv312/bin/python -m pytest tests/test_ss12_phase2_source_packet_matrix.py -q
```

Result:

```text
2 passed in 0.41s
```

Combined focused suite:

```text
.venv312/bin/python -m pytest \
  tests/test_results_artifact_hygiene.py \
  tests/test_ss12_phase2_source_packet_matrix.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q
ruff check scripts/verify_active_results_artifact_hygiene.py tests/test_results_artifact_hygiene.py tests/test_ss12_phase2_source_packet_matrix.py
```

Result:

```text
32 passed in 3.69s
All checks passed!
```

## Learn

The strongest same-scope candidate source is:

`KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.md`

It supports reviewed-candidate, line-cited extraction for:

- large-electrode geometry: outer electrode 400 mm, inner electrode 230 mm, electrode length about 600 mm
- stored energy/operating table: up to 1 MJ, table spanning 27.25-40.00 kV and 500.0-1078.0 kJ
- D2 fill pressure table/context: 1, 2, 4, 5 Torr; fast ions only at 1-2 Torr
- Imax table range: 768-2156 kA
- scalar neutron yield: about 2e11 neutrons/shot
- neutron anisotropy definition and table range
- detector response candidates: four silver activation counters calibrated with Am-Be source, scintillation probe around 15 m from electrode outlet
- neutron timing candidate: scintillation signal first peak hard X-rays, second peak 2.45 MeV neutrons

Hard blockers remain:

- No channel is accepted yet.
- No complete uncertainty budget.
- No review certificate.
- No same-scope neutron spectrum. The source says TOF spectrum measurement was future work.
- Density history, EM field history, temperature/distribution history, and startup BVP remain unclosed.

## Continue

Next executable phase:

1. Add a reusable source-packet validator script so the extracted matrix can become a stable artifact, not just a doc/test pair.
2. Use HeliosMatrix_KB hybrid/hybrid_rerank retrieval to locate same-scope or explicit transfer-rule candidates for the blocked channels.
3. Keep every channel fail-closed until uncertainty/review are present.
4. Do not modify runtime acceptance flags.

## Active external work

- Helios gold eval force-clean rerun is still running in `hybrid_rerank`; previous clean modes showed bm25 R@20=0.9655, dense R@20=0.8276, hybrid R@20=1.0.
- Codex validator agent launched.
- Claude read-only review agent launched.
