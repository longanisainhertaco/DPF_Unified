# Differentiable MHD Competitive Landscape (2024-2026)

**Date**: 2026-03-27
**Purpose**: Scope the novelty claim for the CPC paper. The "first-ever differentiable MHD solver" claim is FALSE — jf1uids/astronomix published first (Oct 2024).

## Published Differentiable MHD/Plasma Codes

| Code | Framework | Dims | Riemann | AD? | Published | Domain |
|------|-----------|------|---------|-----|-----------|--------|
| jf1uids/astronomix | JAX | 1D→3D | HLL, HLLC | Full JAX AD | Oct 2024, arXiv:2410.23093 | Stellar winds, astrophysics |
| TORAX | JAX (DeepMind) | 1D transport | N/A | Full JAX AD | Jun 2024, arXiv:2406.06718 | Tokamak transport |
| JAX-Fluids 2.0 | JAX | 3D | HLLC, Roe | Full JAX AD | Feb 2024, arXiv:2402.05193 | Compressible two-phase CFD |
| DPF-Unified | MLX | 2D cyl | HLL, HLLS, HLLD | mx.grad | Not yet published | Dense Plasma Focus |

## What astronomix has that we don't
- Multi-GPU via jax.pmap
- 3D MHD
- Published and cited
- Growing community (astronomix GitHub)

## What we have that astronomix doesn't
- Circuit-plasma coupling (RLC + snowplow + MHD handoff)
- Electrode boundary conditions (B_theta = mu0*I/2*pi*r)
- Boris vacuum correction (Gombosi 2002)
- Entropy-stable HLLS Riemann solver (Popovas 2025)
- Dual-energy formalism with switching criterion
- Lee-More + anomalous resistivity (drift-velocity model)
- Flux-limited Braginskii conduction
- Calibration against experimental DPF devices
- Apple MLX backend (consumer hardware)
- Educational frontend with Babylon.js 3D visualization

## Defensible Novelty Claims for CPC Paper

1. **First differentiable MHD solver for pulsed power / Dense Plasma Focus devices**
   - No other code does AD through circuit-coupled z-pinch MHD
   - The circuit-MHD coupling (dLp/dt feedback) is unique to DPF

2. **First entropy-stable differentiable Riemann solver (HLLS)**
   - Popovas 2025 entropy-based pressure recovery in a differentiable context
   - astronomix uses HLL/HLLC without entropy stabilization

3. **First MHD solver on Apple MLX framework**
   - Verified: no other MLX-based PDE solver in literature (2024-2026)
   - Demonstrates consumer-hardware HPC for computational physics

4. **First gradient-based DPF calibration**
   - d(I_peak)/d(fc) computed via automatic differentiation
   - Compared to Optuna TPE black-box optimization

## Paper Framing (revised)

**Title suggestion**: "Differentiable magnetohydrodynamics for Dense Plasma Focus simulation: entropy-stable Riemann solving and gradient-based calibration on consumer hardware"

**Key differentiators from astronomix**:
- Application domain (pulsed power vs astrophysics)
- Physics complexity (circuit coupling, electrode BCs, vacuum handling)
- Validation methodology (experimental device data, not analytical solutions)
- Platform (MLX/Apple Silicon vs JAX/NVIDIA)

## References
- Storcks & Buck, arXiv:2410.23093 (Oct 2024) — jf1uids
- astronomix GitHub: github.com/leo1200/astronomix
- Storcks & Buck, arXiv:2512.05999 (Dec 2025) — solver-in-the-loop MHD
- Citrin et al., arXiv:2406.06718 (Jun 2024) — TORAX
- Bezgin et al., arXiv:2402.05193 (Feb 2024) — JAX-Fluids 2.0
