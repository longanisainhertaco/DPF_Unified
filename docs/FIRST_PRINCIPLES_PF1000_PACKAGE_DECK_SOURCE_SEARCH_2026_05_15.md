# FP-1 PF-1000/Akel Package-Native Deck Source Search

Date: 2026-05-15  
Scope: local `KnowledgeReference/`, package-native deck/CLI surfaces, and first-principles runner tests.

## Verdict

FP-1 is improved but not accepted. The package-native `first-principles-3d`
surface now has a built-in PF-1000/Akel 16 kV shot-12581 engineering deck based
on the local source truth. This removes the immediate blocker where the 3-D
package-native runner only had a generic LLNL-like smoke default while the
demonstrator scope remained app-backed.

The deck remains `engineering_candidate_not_validation`. It does not provide
accepted startup, same-scope waveform, spatial/field/temperature, neutron
authority, comparator/UQ, numerical-fidelity, limiter-zero, certificate, or
generalization evidence.

## Source Findings

### PF-1000/Akel Machine And Operating Values

Source: `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.

- Lines `111-117`: PF-1000 has 480 mm coaxial electrodes, copper anode tube
  diameter 231 mm, capacitor bank `C0 = 1332 uF`, operated at `16 kV`, stored
  energy `170.5 kJ`, maximal discharge current `(1100-1300) kA`, and
  deuterium pressures `1.05` and `1.2 Torr`.
- Lines `120-139`: current diagnostics, scintillator detector directions,
  activation counters, neutron-yield uncertainty, timing reference, and timing
  uncertainty.
- Lines `262-268`: shot-12581 example parameters:
  `L0 = 25 nH`, `C0 = 1332 uF`, `r0 = 6.1 mOhm`, `b = 16 cm`,
  `a = 11.55 cm`, `z0 = 48 cm`, `V0 = 16 kV`, `p0 = 1.2 Torr`,
  deuterium gas.
- Lines `282-289`: shot-12581 Lee-output values and measured scalar neutron
  yield are useful reference context only, not first-principles validation
  authority.

## Implementation Ratchet

- Added `pf1000_akel_16kv_engineering_deck()` in
  `src/dpf/first_principles/deck.py`.
- Exported it from `src/dpf/first_principles/__init__.py`.
- Changed the built-in `dpf first-principles-3d` default to use this
  source-scoped PF-1000/Akel engineering deck.
- Preserved compact JSON deck compatibility by keeping a separate compact
  default for user-provided small decks.
- Added tests in `tests/test_first_principles_input_deck.py` and updated
  `tests/test_cli_first_principles_3d.py`.

## Required Follow-Up

This does not complete FP-1 because the package-native PF-1000/Akel path still
needs:

- full shared package-native routing from `dpf first-principles`,
  `dpf first-principles-3d`, `dpf simulate --run-mode=first_principles_mhd`,
  API, and app surfaces;
- package-native PF-1000/Akel geometry, startup, power-port, limiter, numerical,
  closure, same-scope, neutron, comparator/UQ, certificate, and generalization
  packets attached to one run manifest;
- rejection tests proving app-backed, reduced-model, and fallback paths cannot
  be promoted.

## Validated Commands

- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_cli_first_principles_3d.py tests/test_first_principles_runner.py` -> 18 passed.
- `python3 -m json.tool docs/FIRST_PRINCIPLES_PF1000_PACKAGE_DECK_SOURCE_SEARCH_2026_05_15.json` -> valid JSON.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py` -> 34 passed.
