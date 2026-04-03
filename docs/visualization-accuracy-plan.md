# DPF Visualization Scientific Accuracy Plan

*All citations verified against PDFs read in session 2026-04-02. Each citation marked with page number and equation/figure number from the actual paper.*

---

## What the Narrative Leaves Out from MHD

### 1. Ohmic Heating
The primary plasma heating mechanism during rundown and early radial phases. Chen Eq. 5.75, p.170: E = eta*j (Ohm's law). Power density = j^2 * eta. Spitzer resistivity eta proportional to T_e^(-3/2) (Chen Eq. 5.71, p.169; exact form Eq. 5.76, p.171: eta_parallel = 5.2e-5 * Z * ln(Lambda) / T^(3/2) ohm-m). The plasma becomes a better conductor as it heats — nearly collisionless at keV.

### 2. Magnetic Field Diffusion
Chen Ch. 6.4 (title in TOC p.192 — section content not read in this session). Resistive diffusion allows B to penetrate the plasma on timescale tau ~ mu_0 * sigma * L^2. At early times the sheath is resistive and thick; as it heats, it becomes thin and sharp.

### 3. Dynamic Pressure Imbalance
The equilibrium dp/dr = -J_z*B_theta (Ch.13 Eq. 2.1, p.340) is reached only at stagnation. During implosion, magnetic pressure EXCEEDS kinetic pressure — that imbalance IS the implosion. p + B^2/(2*mu_0) = constant (Chen Eq. 6.7, p.191) only holds in equilibrium.

### 4. Plasma Beta Evolution
beta = sum(nkT) / (B^2/(2*mu_0)) (Chen Eq. 6.8, p.191). Low beta during rundown (B dominates), rising through radial phase, potentially >1 at stagnation in pinch core. Chen Fig. 6.5, p.191: "In a finite-beta plasma, the diamagnetic current significantly decreases the magnetic field, keeping the sum of magnetic and particle pressures a constant."

### 5. Slug Structure (Shock Front vs. Magnetic Piston)
Lee 2014 p.324 Fig.4: r_s (shock front) and r_p (piston) as DISTINCT trajectories with compressed slug between them. Lee 2014 p.325 Fig.5: four radial zones — P_m (vacuum inside shock), P/rho/T (compressed slug), current sheath (magnetic piston), rho_0/P_0/T_0 (undisturbed gas outside). Lee 2014 p.324: "In this model, the magnetic pressure drives a shock wave ahead of it, creating a space for the magnetic piston (also called current sheet CS) to move into."

### 6. Radiation Cooling and Radiative Enhancement
Lee 2014 p.321, phase 4: "for gases such as neon, argon, krypton and xenon, radiation emission may actually enhance the compression where we have included energy loss/gain terms from Joule heating and radiation losses into the piston equation of motion." Bremsstrahlung P_rad proportional to n_e^2 * sqrt(T_e).

### 7. Two-Temperature Physics
T_e and T_i decouple: electrons heat faster (Ohmic), equilibrate with ions on collision timescale. Ch.13 Eq. 2.13, p.342: p(r) = n(r) k (T_e + T_i) — total pressure involves both species. Chen Eq. 5.76, p.171: resistivity depends on T_e only.

### 8. Circuit Coupling and Back-EMF
Lee 2014 p.323, Eq. (10): V = f_c*I*dL/dt + f_c*L*dI/dt, where L = (mu/2pi)(ln c)z. The changing inductance L(t) drives back-EMF opposing current. The current "dip" at pinch time is the diagnostic signature. Damideh 2025 p.4 Fig.2: shows experimental current dip + voltage spike, with dynamics-induced voltage exceeding 150 kV on FAETON-I.

### 9. Re-strikes
Damideh 2025 p.1: "A major difficulty anticipated for PF operation at high-voltage and high-current is the likelihood of re-strikes which divert current away from the compressing plasma." Secondary breakdowns in the inter-electrode space that short-circuit the pinch current.

---

## Implementation Plan: Babylon.js 9.x

### Tier 1: Fix What Exists (1-2 days)

#### 1a. Breakdown/Flashover Phase
- **Physics**: Goyon 2025 p.3: "the high voltage applied to the electrodes generates field-emitted electrons from the cathode that ionize the gas en route to the anode. This forms a conducting channel that completes the circuit." This is stage (1) in Goyon Fig.2, labeled "Insulator flashover."
- **Implementation**: Add phase `"breakdown"` before `"rundown"`. Render a bright arc across the insulator surface using **GreasedLine** (`BabylonJS/packages/dev/core/src/Meshes/GreasedLine/greasedLineMesh.ts`) with noise-modulated width and emissive color. Arc connects anode surface to cathode at the insulator location. Fade in over ~3 frames, then transition to rundown sheath.

#### 1b. Slug Structure During Radial Phase
- **Physics**: Lee 2014 p.324-325 Figs. 4-5: r_s (shock) and r_p (piston) are distinct. Between them: compressed slug at pressure P, density rho, temperature T. Lee 2014 p.324: "the magnetic pressure drives a shock wave ahead of it."
- **Implementation**: From rho(r,z) MHD snapshots, extract TWO radii per z-slice: r_shock (outermost radius where rho > 2*rho_ambient) and r_piston (peak density gradient). Render as two concentric `CreateTube` meshes with `instance:` update. Outer tube = shock front (transparent blue-white). Inner tube = existing sheath. Gap between them = compressed slug (semi-transparent gradient).

#### 1c. B-field 1/r Radial Variation
- **Physics**: Ch.13 Eq. 2.4, p.341: B_theta(r) = mu_0 * I_z(r) / (2*pi*r). Field is strongest near the anode, falls as 1/r.
- **Implementation**: Replace 5 identical tori with 5 tori at geometrically spaced radii (r = a*1.2, a*1.6, a*2.2, a*3.0, a*4.0). Scale emissive intensity proportional to 1/r.

#### 1d. Beam Timing and Bidirectional Beams
- **Physics**: Goyon 2025 Fig.4: first neutron peak at t_stag (thermonuclear). Goyon 2025 p.4: "electrons propagate primarily toward the anode where they generate x-ray radiation via bremsstrahlung... ions are accelerated toward the plasma assembled downstream of the pinch region." Damideh 2025 p.1: "the first dynamics-induced pulse of fast ion beams is produced just before stagnation."
- **Implementation**: Show ion beam cone starting at `phase === "pinch"` with `pI > 0.5`, not just `post_pinch`. Add a SECOND cone pointing toward the anode (electron beam, -x direction). Smaller, blue-white color.

#### 1e. Fix Bennett Profile Axis
- **Physics**: Ch.13 Eq. 3.8, p.348: n(r) = n_0 / (1 + n_0*b*r^2)^2 — RADIAL profile, not axial. Ch.13 Fig.5, p.348: bell-curve density peaked on axis.
- **Implementation**: Apply Bennett-like taper to the RADIAL cross-section of the pinch tube at each z-slice, not the axial envelope. The tube radius variation along z should come from MHD data (Tier 2a) or be uniform; the density concentration at the axis should be shown via the r-z cross-section heatmap (already enabled).

---

### Tier 2: MHD-Driven Geometry (3-5 days)

#### 2a. Density Isosurface for Sheath/Pinch
- **Physics**: The actual sheath and pinch shapes emerge from MHD conservation laws (Chen Eq. 5.85, p.173: rho * dv/dt = j x B - grad(p)), not from 0D scalars.
- **Implementation**: From rho(r,z,t) snapshots, compute isosurface at rho = 0.5*rho_max as r(z,t). Render using `CreateTube` with `instance:` update per frame. During pinch, this naturally produces the Bennett radial profile. During post-pinch, m=0 instability structure emerges from the data, not imposed sinusoids.

#### 2b. Field Line Tracing from Br/Bz Data
- **Physics**: Auluck 2024 p.1: B has both toroidal (B_theta) and poloidal (B_r, B_z) components. Chen Ch. 6.2, p.189 Fig.6.2: j and B lie on constant-pressure surfaces.
- **Data**: Pipeline already exports Br and Bz per frame (app_visualization.py:146-147, 232-235).
- **Implementation**: 4th-order Runge-Kutta field line tracer in JS. Seed at r = [0.2, 0.4, 0.6, 0.8] * cathode_radius. Render as **GreasedLine** meshes with width proportional to |B|. Color: blue=B_theta-dominated, red=B_z-dominated. Update per MHD frame.

#### 2c. Velocity Streamlines from vr/vz Data
- **Physics**: Chen Eq. 5.85, p.173: rho * dv/dt = j x B - grad(p). Velocity field shows the implosion dynamics. Goyon 2025 p.4 Eq.1: v_imp ~ 950 km/s * I_imp / (R_imp * sqrt(P_fill)).
- **Data**: Pipeline exports vr, vz, vmag per frame (app_visualization.py:247-248).
- **Implementation**: Same RK4 tracer. Render as GreasedLines with arrow-head cones via **thin instances** (`BabylonJS/packages/dev/core/src/Meshes/thinInstanceMesh.ts`). Color by |v| using viridis.

#### 2d. Pressure and Beta Overlays
- **Physics**: Chen Eq. 6.7, p.191: p + B^2/(2*mu_0) = constant. Chen Eq. 6.8, p.191: beta = nkT / (B^2/(2*mu_0)). Ch.13 Eq. 2.25, p.344: p_avg = B_theta^2(R) / (2*mu_0) — average kinetic pressure equals magnetic pressure at boundary.
- **Data**: P_mid and B_mid already in MHD snapshots.
- **Implementation**: Compute beta(r,z) = 2*mu_0*P_mid / B_mid^2. Add as heatmap overlay option. Diverging colormap centered at beta=1: blue=magnetically dominated, red=pressure dominated.

#### 2e. Current Density J Overlay
- **Physics**: Ch.13 Eq. 2.17, p.343: (1/r)*d/dr[r*B_theta] = mu_0*J_z. Ch.13 Eq. 2.18, p.343: J_z = (1/mu_0)*dB_theta/dr + (1/mu_0)*B_theta/r. Ch.13 Figs. 3-4, p.344-345: radial profiles of J_z, B_theta, p for uniform-J and surface-current models.
- **Implementation**: Compute J from curl(B) using finite differences on B_theta(r) data. Display as heatmap overlay.

---

### Tier 3: Advanced Physics Visualization (1-2 weeks)

#### 3a. Energy Flow Sankey Diagram
- **Physics**: Lee 2014 p.322 Fig.3: circuit energy flows through L_0, r_0, into L(t) (plasma inductance). Energy chain: E_cap -> E_magnetic -> E_kinetic -> E_thermal -> E_radiation -> E_neutron.
- **Implementation**: HTML Canvas overlay (not Babylon). Time-evolving stacked bars with animated flow arrows. Lee model computes all terms.

#### 3b. Ohmic Heating Visualization
- **Physics**: Chen Eq. 5.75, p.170: E = eta*j. Power = j^2*eta. Chen Eq. 5.76, p.171: eta = 5.2e-5 * Z * ln(Lambda) / T_eV^(3/2).
- **Implementation**: Compute j^2*eta from MHD data (J from 2e, eta from Spitzer using T_e). Heatmap overlay: bright where Ohmic heating is strongest.

#### 3c. Radiation Cooling Visualization
- **Physics**: Lee 2014 p.321 phase 4: "energy loss/gain terms from Joule heating and radiation losses." Already partially implemented: P_rad = 1.69e-32 * n_e^2 * sqrt(T_e) (app_visualization.py:125).
- **Implementation**: Spawn dim particles at high-P_rad cells using **GPUParticleSystem** with **noise texture** (`baseParticleSystem.ts:227`). Particles drift outward (radiation escapes). During pinch, show self-absorption via reduced particle drift distance.

#### 3d. Re-strike Visualization
- **Physics**: Damideh 2025 p.1: "re-strikes which divert current away from the compressing plasma." Damideh 2025 p.4 Fig.2: current dip + voltage spike at re-strike time.
- **Implementation**: When simulation detects re-strike (current dip + voltage spike), flash a GreasedLine arc between anode and cathode at the re-strike z-location. HUD annotation: "Re-strike: X% current diverted."

#### 3e. Two-Temperature Display
- **Physics**: Ch.13 Eq. 2.13, p.342: p = n*k*(T_e + T_i). Chen Eq. 5.76, p.171: eta depends on T_e only.
- **Implementation**: If solver provides T_e and T_i separately, show T_e on left half of r-z cross-section, T_i on right. Toggle button. Annotate T_e/T_i ratio in HUD.

#### 3f. Circuit Waveform Overlay
- **Physics**: Lee 2014 p.323 Eq.10: V = f_c*I*dL/dt + f_c*L*dI/dt. Damideh 2025 p.4 Fig.2: I(t) and V(t) with characteristic current dip at pinch.
- **Implementation**: HTML Canvas inset plot showing I(t) and V(t). Vertical marker tracks playback position. Annotate current dip at pinch time (back-EMF signature from inductance change).

---

## Babylon.js 9.x Features Per Tier

| Feature | BJS Source File | Tier |
|---------|----------------|------|
| GreasedLine | `Meshes/GreasedLine/greasedLineMesh.ts` | 1a, 2b, 2c, 3d |
| Thin Instances | `Meshes/thinInstanceMesh.ts` | 2c (arrow-heads) |
| RawTexture | `Materials/Textures/rawTexture.ts` | 2d, 2e, 3b |
| CreateTube + instance | `Meshes/Builders/tubeBuilder.ts` | 1b, 2a |
| GPUParticleSystem + noise | `Particles/gpuParticleSystem.ts`, `baseParticleSystem.ts:227` | 3c |
| ShaderMaterial | `Materials/shaderMaterial.ts` | 3b |
| DynamicTexture | `Materials/Textures/dynamicTexture.ts` | 3f |

---

## Citation Verification Status

All citations below were read as PDFs in session 2026-04-02:

| Citation | Page | Eq/Fig | Status |
|----------|------|--------|--------|
| Lee 2014, J. Fusion Energy 33:319 | p.321 | 5-phase summary | VERIFIED |
| Lee 2014 | p.322 | Fig.3 (circuit), Eq.1 | VERIFIED |
| Lee 2014 | p.323 | Eq.10 (voltage) | VERIFIED |
| Lee 2014 | p.324 | Fig.4 (r_s, r_p trajectories) | VERIFIED |
| Lee 2014 | p.325 | Fig.5 (slug zones) | VERIFIED |
| Chen 3rd Ed. | p.169 | Eq.5.71 (Spitzer eta) | VERIFIED |
| Chen 3rd Ed. | p.170 | Eq.5.75 (Ohm's law) | VERIFIED |
| Chen 3rd Ed. | p.171 | Eq.5.76 (eta_parallel) | VERIFIED |
| Chen 3rd Ed. | p.173 | Eq.5.85 (MHD momentum) | VERIFIED |
| Chen 3rd Ed. | p.191 | Eq.6.7 (total pressure) | VERIFIED |
| Chen 3rd Ed. | p.191 | Eq.6.8 (beta) | VERIFIED |
| Chen 3rd Ed. | p.189 | Fig.6.2 (j x B equilibrium) | VERIFIED |
| Chen 3rd Ed. | p.191 | Fig.6.5 (finite-beta) | VERIFIED |
| Chen 3rd Ed. | p.192 | Ch.6.4 title (B diffusion) | TOC ONLY — section not read |
| Fundamentals Ch.13 | p.339 | Fig.1 (pinch J,B) | VERIFIED |
| Fundamentals Ch.13 | p.340 | Eq.2.1 (dp/dr) | VERIFIED |
| Fundamentals Ch.13 | p.341 | Eq.2.4 (B_theta) | VERIFIED |
| Fundamentals Ch.13 | p.342 | Eq.2.13 (p = nk(Te+Ti)) | VERIFIED |
| Fundamentals Ch.13 | p.342 | Eq.2.15 (Bennett relation) | VERIFIED |
| Fundamentals Ch.13 | p.343 | Eq.2.17-2.18 (J from curl B) | VERIFIED |
| Fundamentals Ch.13 | p.344 | Eq.2.25-2.27, Fig.3 | VERIFIED |
| Fundamentals Ch.13 | p.345 | Fig.4 (surface current) | VERIFIED |
| Fundamentals Ch.13 | p.348 | Eq.3.8 (Bennett n(r)), Fig.5 | VERIFIED |
| Goyon 2025, Phys. Plasmas 32:033105 | p.2 | 24 cathode rods | VERIFIED |
| Goyon 2025 | p.3 | 4-stage discharge, J x B description | VERIFIED |
| Goyon 2025 | p.4 | Eq.1 (v_imp), Eq.2 (T_st) | VERIFIED |
| Goyon 2025 | p.4 | Fig.4 (3 neutron peaks) | VERIFIED |
| Goyon 2025 | p.4 | Bidirectional beams | VERIFIED |
| Damideh 2025, Sci. Rep. 15:23048 | p.1 | Re-strikes, beam timing | VERIFIED |
| Damideh 2025 | p.4 | Fig.2 (I(t), V(t) waveforms) | VERIFIED |
| Auluck 2024, Phys. Plasmas 31:010704 | p.1 | Poloidal B-field | VERIFIED |
| Lee RADPF theory | p.2 | Fig.1a (axial phase), Eq.I | VERIFIED |

---

## Priority Order

1. **Tier 1** (1-2 days): 1d (beam timing) > 1c (B-field 1/r) > 1b (slug structure) > 1a (breakdown) > 1e (Bennett axis)
2. **Tier 2** (3-5 days): 2a (density isosurface) > 2b (field lines) > 2d (beta overlay) > 2c (velocity streamlines) > 2e (J overlay)
3. **Tier 3** (1-2 weeks): 3f (circuit waveform) > 3a (energy Sankey) > 3b (Ohmic heating) > 3c (radiation particles) > 3d (re-strikes) > 3e (two-T display)

The single highest-impact change is **2a**: replacing Lee-scalar geometry with MHD density isosurfaces. Every other visualization anomaly (slug structure, Bennett profile, m=0 shape) resolves itself when the 3D geometry comes from the actual MHD solution.
