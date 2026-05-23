# SS18 Neutron Diagnostic Validation Stack — 2026-05-23

## Scope

Validation scope: `pf1000_neutron_diagnostics_full_energy_27_to_40_kv`.

Authority rule: local `KnowledgeReference/` line-cited source text only. Retrieval hits and earlier packets are candidate staging, not authority.

## Outputs

- Packet: `docs/SS18_NEUTRON_DIAGNOSTIC_VALIDATION_STACK_2026_05_23.json`
- Validator: `scripts/validate_ss18_neutron_diagnostic_packet.py`
- Tests: `tests/test_ss18_neutron_diagnostic_validation_stack.py`

## Mechanism-separated neutron evidence

The SS18 packet separates the neutron diagnostic evidence into seven mechanisms:

1. `yield` — candidate scalar-yield evidence from PF-1000 large-electrode shots; remains blocked by calibration, shot-selection, review, and uncertainty gaps.
2. `timing` — candidate scintillator/neutron-vs-X-ray timing evidence; remains blocked by raw-trace digitization, detector impulse response, propagation correction, and uncertainty gaps.
3. `spectrum` — blocked; 2.45 MeV identification and future ToF-spectrum language do not constitute a same-scope measured spectrum packet.
4. `anisotropy` — candidate angular activation/yield-ratio evidence; remains blocked by angular detector response, scatter correction, shot-resolved uncertainty, and review gaps.
5. `detector_activation_response` — candidate activation-counter calibration and diagnostic-suite context; remains blocked by response-matrix, efficiency, scatter-correction, and uncertainty gaps.
6. `diagnostic_mapping` — candidate mapping from source observables to comparator channels; remains blocked by detector geometry, offsets, efficiencies, and response-kernel review gaps.
7. `uncertainty_blockers` — blocked; no complete uncertainty budget or independent review certificate exists.

## Acceptance boundary

All acceptance flags remain false:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- `promotes_acceptance=false`

No SS18 row is accepted, and no row promotes runtime or first-principles acceptance.

## Verification

Commands run:

```bash
python3 -m pytest tests/test_ss18_neutron_diagnostic_validation_stack.py -q
python3 scripts/validate_ss18_neutron_diagnostic_packet.py docs/SS18_NEUTRON_DIAGNOSTIC_VALIDATION_STACK_2026_05_23.json
```

The validator checks:

- required diagnostic mechanisms are present exactly once;
- non-accepted mechanisms include blocked reasons and diagnostic-channel mapping;
- acceptance flags remain false;
- per-mechanism `promotes_acceptance` remains false;
- source refs resolve inside `KnowledgeReference/` only;
- line windows are narrow and exact-quote matched.

## Evaluate / Learn / Continue

- **Evaluate:** SS18 now has a mechanism-separated neutron packet, exact source-ref quote validation, diagnostic completeness checks, and acceptance-promotion guards. Focused tests pass, and the live validator reports no issues.
- **Learn:** Local PF-1000 sources contain useful candidate evidence for scalar yield, timing, anisotropy, detector calibration context, and diagnostic mapping. They still do not close neutron-spectrum measurement, response-matrix, full diagnostic uncertainty, or independent review requirements.
- **Continue:** SS19 should consume this packet as non-promoting evidence and keep certificate acceptance refused until neutron-spectrum, detector-response, uncertainty, comparator, and review-certificate blockers close in the same commit.

## Fix/Reverify addendum — 2026-05-23

- **Evaluate:** Independent review returned PASS with no blocking requested changes. Reverification still tightened the neutron validator around `diagnostic_completeness_check`: the validator now rejects completeness shortcuts (`complete_for_acceptance=true`), empty blocker lists, and required-mechanism drift. Focused RED/GREEN regression added in `tests/test_ss18_neutron_diagnostic_validation_stack.py`.
- **Learn:** The original packet already carried the diagnostic completeness object, but the validator did not directly enforce that object. That was a latent R1/R8 gap: a later edit could have flipped completeness or narrowed the required mechanism list while the packet-level acceptance flags stayed false.
- **Continue:** SS19 can rely on SS18 only as a fail-closed, non-promoting packet. It must still refuse certificate acceptance until neutron spectrum, detector/activation response matrix, same-scope uncertainty budget, comparator mapping, and independent review certificate close together.
