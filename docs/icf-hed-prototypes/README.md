# ICF/HED Prototype Research Library

Standalone research documents with governing equations, literature surveys, and
runnable Python prototypes for physics modules relevant to ICF and HED programs
but **not needed** for DPF at our operating conditions (kJ-MJ, D2 fill gas, <100 kV).

These are shelf-ready references. None are integrated into DPF-Unified.

## Modules

| Module | Lines | Prototypes | Key Physics | When Needed |
|--------|-------|------------|-------------|-------------|
| [Multi-material ALE](multi_material_ale.md) | 1174 | 1D two-material Sod | VOF/MOF interface, Lagrangian+remap | Multiple materials at interfaces |
| [Tabular EOS](tabular_eos.md) | 1110 | QEOS for deuterium | SESAME/QEOS/FEOS, T(rho,e) inversion | High compression, phase changes |
| [Laser-plasma](laser_plasma.md) | 795 | 1D ray-trace + IB absorption | Geometric optics, SRS/SBS/TPD, CBET | Laser-driven HED (NIF) |
| [Multi-group radiation](multigroup_radiation.md) | 958 | 1D grey FLD Marshak wave | FLD, M1, IMC, opacity tables | Radiation-pressure-dominated regimes |
| [GRMHD](grmhd.md) | 1113 | 1D SRMHD Balsara test | Valencia formulation, primitive recovery | Relativistic jets, BH accretion |
| [Nuclear burn](nuclear_burn.md) | 1078 | D-D/D-T burn network | Bosch-Hale reactivities, beam-target vs TN | ICF ignition, self-heating |
| [Self-gravity](self_gravity.md) | 862 | 2D FFT Poisson solver | FFT/multigrid/FMM, Jeans criterion | Star formation, galaxy sims |
| [Wire array dynamics](wire_array_dynamics.md) | 1155 | 1D thin-shell implosion | Rocket ablation, MRT, MagLIF | Z-machine, wire array Z-pinch |

## Why These Are Not in DPF-Unified

| Module | DPF Relevance | Quantitative Argument |
|--------|--------------|----------------------|
| Multi-material ALE | Single D2 fill gas | No material interfaces to track |
| Tabular EOS | Ideal gas sufficient | theta = k_BT/E_F ~ 2900, Gamma ~ 0.002 |
| Laser-plasma | Electrically driven | No laser in DPF |
| Multi-group radiation | Radiation is loss term | P_rad/P_th ~ 10^-3 to 10^-4 |
| GRMHD | Non-relativistic | v/c ~ 10^-3, GR corrections ~ 10^-6 |
| Nuclear burn | Beam-target dominates | BT exceeds TN by ~4 orders at PF-1000 |
| Self-gravity | Laboratory scale | g_grav/g_magnetic ~ 10^-20 |
| Wire array | Gas fill, not wires | Different driver topology entirely |

## Production Code Survey

| Code | ALE | Tab EOS | Laser | Rad Transport | GR | Burn | Gravity | Wires |
|------|-----|---------|-------|---------------|-----|------|---------|-------|
| HYDRA (LLNL) | Y | LEOS | Y | IMC | - | Y | - | - |
| ALEGRA (Sandia) | Y | SESAME | - | FLD+Sn | - | - | - | - |
| GORGON (Imperial) | Y | QEOS | - | Multi-group FLD | - | - | - | Y |
| FLASH (Chicago) | - | Helmholtz | Y | Grey FLD | - | Y | Y | - |
| Athena++ (Princeton) | - | General | - | - | Y | - | Y | - |
| PERSEUS (Princeton) | - | Ideal | - | - | - | - | - | - |

## Usage

Each document contains fenced Python code blocks that can be extracted and run
independently. No DPF-Unified dependencies required — only numpy, scipy, matplotlib.

```bash
# Example: extract and run the nuclear burn prototype
python3 -c "
# Copy the prototype code block from nuclear_burn.md and paste here
"
```

## Review Status

All 8 documents underwent Six Sigma review (Cycle 2) by dpf-mhd-physicist agents
checking equations against literature, code correctness, and physics consistency.
