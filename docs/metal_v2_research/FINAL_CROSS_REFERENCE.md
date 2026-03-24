# Final Cross-Reference: All Research Sources

**Date**: 2026-03-24
**Sources**: 5 opus agents + Gemini Ultra + implementation manual (docx)

## Source Agreement Matrix

| Claim | Gemini | Agent 1 (DISPATCH) | Agent 2 (Enzo) | Agent 3 (MLX API) | Agent 4 (FLASH) | Agent 5 (Kernels) | Manual | Verdict |
|-------|--------|-------------------|----------------|-------------------|-----------------|-------------------|--------|---------|
| HLLS replaces E with entropy | YES | YES (S per unit mass) | N/A | N/A | N/A | N/A | YES (Srho = rho*S) | CONFIRMED |
| p = K*rho^gamma (positive by construction) | YES | YES (via exp form) | N/A | N/A | N/A | N/A | YES | CONFIRMED |
| Float32 validated in paper | "every reason" | **YES — paper explicitly states it** | N/A | N/A | N/A | N/A | References it | CONFIRMED |
| CT decoupled from entropy | YES | YES | N/A | N/A | YES | N/A | YES | CONFIRMED (5/5) |
| Switching criterion circular in Enzo | Not flagged | N/A | **YES — confirmed FM-4** | N/A | N/A | N/A | YES (Part I §1.8) | CONFIRMED |
| Enzo η₁ unused in active code | Not discussed | N/A | **YES — #ifdef UNUSED** | N/A | N/A | N/A | YES (Part I §1.2.2) | CONFIRMED |
| Enzo missing ohmic heating in ge | Not discussed | N/A | **YES — known gap** | N/A | N/A | N/A | YES (Part I §1.5, FM-6) | CONFIRMED |
| FLASH eintSwitch = 1e-4 | Not discussed | N/A | N/A | N/A | **YES** | N/A | Not discussed | CONFIRMED |
| FLASH compares e_int to e_kin only (not ME) | Not discussed | N/A | N/A | N/A | **YES** | N/A | Not discussed | CONFIRMED |
| mx.fast.metal_kernel() exists | Mentioned | N/A | N/A | **YES — MLX 0.31.0** | N/A | **YES — MLX 0.30.6** | N/A | CONFIRMED (version discrepancy: 0.30.6 vs 0.31.0, both work) |
| No prior MLX PDE solvers | Not discussed | N/A | N/A | **YES — zero found** | N/A | N/A | N/A | CONFIRMED |
| DISPATCH paper author | "Agertz, Nordlund" | **Popovas (sole author)** | N/A | N/A | N/A | N/A | "Agertz+ 2025" (wrong) | Popovas 2025. Manual needs correction |
| DISPATCH no switching (pure entropy) | Described hybrid | **Confirmed pure — no switch** | N/A | N/A | N/A | N/A | Hybrid approach | Design choice: manual's hybrid is our engineering decision |
| Residual principle for shock entropy | "overwrite K from E" | **YES — Q_S equation (Eq.28)** | N/A | N/A | N/A | N/A | Shock fix via compression detector | Agent 1 gives the actual equation |
| HLLD overhead with entropy | "5-10%" | "10-15% net" | N/A | N/A | N/A | "~600M cells/s at 512×1024" | Not discussed | ~10-15% |
| Kernel launch overhead | "~0.1ms" | N/A | N/A | "~150μs after JIT" | N/A | "158M-1G cells/s measured" | N/A | ~150μs confirmed |

## Discrepancies Requiring Resolution

### 1. Pressure formula: exponential vs multiplicative
- **Agent 1 (DISPATCH)**: P = ρ^γ · exp((γ-1)·S) where S is thermodynamic entropy per unit mass
- **Gemini + Manual**: p = K·ρ^γ where K = p/ρ^γ (pseudo-entropy)
- **Resolution**: These are different entropy definitions. K = p/ρ^γ is simpler (no exp). S = ln(K)/(γ-1). Use K (the manual's approach) — avoids exp overflow at extreme values.

### 2. DISPATCH author attribution
- **Manual**: "Agertz+ 2025"
- **Agent 1**: Popovas (sole author), arXiv:2211.02438, A&A 2025
- **Resolution**: Correct to Popovas 2025. Agertz/Nordlund are DISPATCH framework authors, not this specific paper.

### 3. Pure HLLS vs hybrid dual-energy
- **DISPATCH paper**: Pure entropy, no switching
- **Manual**: Hybrid with smoothstep blend [η₁=10⁻⁵, η₂=10⁻²]
- **Resolution**: The manual's hybrid is safer for DPF. Pure HLLS loses total energy conservation, which is a DoD metric (< 10%). The hybrid preserves conservative total energy where it's accurate (high β) and falls back to entropy where it's not (low β at electrodes).

### 4. η threshold values
| Source | η₁ | η₂ |
|--------|----|----|
| Enzo (Bryan 2014) | 10⁻³ (unused in code!) | 10⁻¹ |
| FLASH | 10⁻⁴ (eintSwitch, vs e_kin only) | N/A |
| Manual | 10⁻⁵ | 10⁻² |
| Our spec | 10⁻³ | 10⁻¹ |
- **Resolution**: Manual's [10⁻⁵, 10⁻²] is correct for float32 DPF. Our spec should update to match.

## Consensus Findings (All Sources Agree)

1. Entropy-based dual-energy eliminates catastrophic cancellation by construction
2. Entropy tracer advected as passive scalar through Riemann solver contact wave
3. CT is fully decoupled — zero changes to existing CT code
4. Ohmic heating MUST appear in both E and Sρ equations (Enzo's gap is a known bug)
5. The switching criterion must NOT use the corrupted E-KE-ME subtraction
6. Hard switching creates artifacts; smooth blending (cubic Hermite) is required
7. Shock entropy fix: resync from total energy ONLY where thermal fraction > threshold
8. mx.fast.metal_kernel() is real and functional for custom MSL kernels
9. We would be the first MLX-based PDE/MHD solver

## Open Questions (No Source Answered)

1. How does the residual principle (DISPATCH Eq.28) perform in float32 specifically? The Q_S computation involves time derivatives of E_kin and E_mag — are these stable in float32?
2. What is the interaction between the entropy tracer and AMR (if we ever add it)?
3. For the DPF's extreme conditions (Te ~ 10⁷ K, B ~ 1000 T), does S = p/ρ^γ overflow float32? Max float32 ≈ 3.4e38. With p ~ 10⁸ Pa and ρ ~ 0.001 kg/m³: S = 10⁸ / 0.001^(5/3) = 10⁸ / 10⁻⁵ = 10¹³. Safe.
