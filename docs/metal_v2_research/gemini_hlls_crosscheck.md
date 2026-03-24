# Gemini HLLS Analysis — Cross-Reference Notes

**Source**: Gemini Ultra analysis of arXiv:2211.02438 (Popovas 2025)
**Date**: 2026-03-24

## Key Confirmations

1. **HLLS replaces E with entropy tracer ρK where K = p/ρ^γ** — matches our spec's Sρ variable
2. **Pressure via p = K·ρ^γ** — pure multiplication, no subtraction. Guarantees positivity.
3. **CT is fully decoupled** — EMFs use v and B fluxes only, thermodynamic variable doesn't participate
4. **Overhead ~5-10%** — one extra scalar per interface, minimal vs stability benefit
5. **arXiv:2211.02438** — concrete preprint reference for our agents to verify

## Discrepancy: Switching Criterion

| Source | Criterion | Uses corrupted subtraction? |
|--------|-----------|---------------------------|
| Gemini | η = E_int / (E_kin + E_mag) > 10⁻² | YES — E_int = E - E_kin - E_mag |
| Our spec | η = p_from_S / E_total | NO — numerator from entropy tracer |
| Enzo (Bryan 2014) | e_int / E_total > η₁ | YES — same circular problem |

**Our entropy-based criterion is strictly superior.** Both Gemini and Enzo compute E_int via the catastrophic subtraction IN the switching criterion itself. At β << 0.01, this ratio is garbage in float32. Using p_from_S (which never subtracts) as the numerator avoids this entirely.

## Gemini Error

Gemini says: "If η_tol > 10⁻² (High-Beta/Shocked): Use E to recover pressure."

This is the standard approach but **doesn't acknowledge the circular problem**. In float32 at β = 10⁻⁶, E_int has ~0 significant digits, making η_tol itself garbage. The switch will fire randomly in the transition zone. Our spec's smooth Hermite blend with entropy-based η handles this correctly.

## Cross-Check Items for Our Agents

- [ ] Agent 2 (DISPATCH): Verify arXiv:2211.02438 exists, confirm Popovas as author
- [ ] Agent 2 (DISPATCH): Does the paper discuss float32 at all?
- [ ] Agent 3 (Enzo): Confirm Enzo's switching uses e_int/E_total (Gemini says E_int/(E_kin+E_mag) which is different)
- [ ] Agent 5 (FLASH): Compare FLASH's eint_switch with both Enzo and Gemini formulations
- [ ] Agent 4 (MLX API): Verify mx.fast.metal_kernel() exists — Gemini didn't address this
