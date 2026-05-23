# SS12 P1 Phase 4 Numerical-Fidelity and Coupling Design

Date: 2026-05-22 UTC
Scope: Convert source-gated P1 findings into the next executable code-verification work, without promoting physics acceptance.

## Evaluate

Existing numerical-fidelity scaffolding is already present:

- `src/dpf/first_principles/numerical_fidelity.py`
- `src/dpf/validation/mhd_numerical_fidelity.py`
- `tests/test_mhd_numerical_fidelity.py`
- `scripts/build_mhd_tier3_numerical_packet.py`
- `results/mhd_tier3_numerical_packet.json`

Existing coupling surfaces are present:

- `src/dpf/engine/circuit_coupling.py`
- `src/dpf/fields/pic_coupling.py`
- `src/dpf/validation/circuit_field_coupling.py`
- `tests/test_circuit_field_coupling.py`
- `tests/test_mlx_circuit_coupling.py`

Phase 2/3 evidence says:

- same-source PF-1000 large-electrode packet is still only a target-extraction candidate;
- current waveform beyond Imax requires figure-backed transfer candidates;
- EM field history requires magnetic-probe transfer candidates;
- circuit coupling has PF-1000 transfer candidates but not same-source closure;
- startup BVP remains blocked by explicit source warning that accurate quantitative model is missing;
- neutron spectrum remains blocked by same-source future-work statement.

## Learn

The next code work should not attempt experimental DPF validation. It should harden Tier-3 numerical verification and coupling-accounting infrastructure so that, when source channels close, the simulator can produce auditable evidence.

The correct next executable work is a non-promoting numerical/coupling packet, not acceptance promotion.

## Phase 4 implementation target

Create/extend a Phase 4 packet that records:

1. Required numerical surfaces from `REQUIRED_NUMERICAL_TEST_SURFACES`.
2. Which existing tests are candidate component tests only.
3. Which surfaces have actual mesh/timestep/order/restart/backend evidence.
4. Which surfaces are blocked by missing source/observable mapping.
5. Circuit-power coupling status:
   - bank/circuit transfer source exists;
   - exact waveform/source-scope evidence missing;
   - Poynting/J·E power-port closure not accepted;
   - density-weighted or metadata-only coupling cannot promote.
6. Transfer-candidate linkage to `docs/SS12_P1_PHASE3_TRANSFER_CANDIDATE_MATRIX_2026_05_22.json`.
7. Required next artifacts before code acceptance:
   - figure digitization review packet for Rogowski/dI/dt/current traces;
   - magnetic-probe dB/dt review packet;
   - power-port sign/time-centering/residual test packet;
   - mesh/timestep convergence packet;
   - restart reproducibility packet;
   - backend/precision parity packet;
   - independent review certificate.

## TDD task split

### P4-A: Non-promoting transfer linkage

Test first:

- Given the Phase 3 transfer matrix, the numerical-fidelity packet must list transfer candidates separately from accepted source channels.
- Any transfer candidate must map to `promotes_acceptance=false`.
- Any missing Phase 3 transfer matrix path must make the packet blocked.

Implementation:

- Add a loader/helper in `src/dpf/first_principles/numerical_fidelity.py` or a new small module if cleaner.
- Do not change existing acceptance flags.

### P4-B: Circuit power-port fail-closed evidence

Test first:

- A circuit-power packet with only bank parameters and no waveform/Poynting/J·E residual remains blocked.
- A packet with density-weighted coupling metadata remains blocked.
- A packet with inconsistent sign convention or missing time-centering remains blocked.

Implementation:

- Extend `src/dpf/validation/circuit_field_coupling.py` only if the existing API is the right home.
- Otherwise create a first-principles packet builder under `src/dpf/first_principles/`.

### P4-C: Figure-backed waveform/density target staging

Test first:

- Figure-backed candidates are invalid unless they contain source path, figure identifier, extraction method, reviewer, digitization hash, and uncertainty.
- Candidate figure data cannot become an accepted observable without review certificate.

Implementation:

- Add staging schema/tests only. Do not digitize figures in this phase.

### P4-D: Acceptance shield

Test first:

- Even if every transfer candidate is present, first-principles acceptance remains false unless same-source source packet, uncertainty budget, numerical packet, and review certificate are all accepted.

Implementation:

- Add explicit cross-packet acceptance shield if not already present.

## Verification command set

Run after each subtask:

```text
.venv312/bin/python -m pytest tests/test_ss12_phase2_source_packet_matrix.py \
  tests/test_ss12_phase3_transfer_candidate_matrix.py -q
```

Run after Phase 4 code changes:

```text
.venv312/bin/python -m pytest tests/test_mhd_numerical_fidelity.py \
  tests/test_circuit_field_coupling.py \
  tests/test_ss12_phase2_source_packet_matrix.py \
  tests/test_ss12_phase3_transfer_candidate_matrix.py -q
```

Then:

```text
ruff check src/dpf/first_principles src/dpf/validation tests
```

## Continue decision

Proceed with P4-A first. It is low-risk, source-gate preserving, and creates the bridge from the new corpus-derived transfer matrix into existing numerical-fidelity infrastructure.

Acceptance flags remain false.
