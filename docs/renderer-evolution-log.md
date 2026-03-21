# DPF Renderer v8 — Evolution Log

## Session: 2026-03-20/21 (overnight evolution loop)

### Summary
- **Starting point**: v8_final.js (944 lines) — research-informed base renderer
- **Final state**: v8_final.js (1403 lines) — 29 evolution cycles, all features
- **All 5 device presets pass**: PF-1000, NX2, MJOLNIR, UNU-ICTP, POSEIDON
- **MHD backend verified**: metal_plm with density snapshots

### Evolution Cycles
| # | Feature | Category |
|---|---|---|
| 1 | Custom GLSL plasma noise shader (simplex FBM) | Shader |
| 2 | Heat distortion post-process | Post-process |
| 3 | Volumetric god rays from pinch | Post-process |
| 4 | Particle size/color gradients (spark trails) | Particles |
| 5 | Three-point lighting + camera vertical bob | Lighting/Camera |
| 6 | Device Fresnel holographic edge glow | Materials |
| 7 | Noise-based heat distortion (Perlin upgrade) | Post-process |
| 8 | Pinch column Fresnel edge glow (core+mantle) | Materials |
| 9 | B-field ring slow rotation + ambient dust | Animation/Atmosphere |
| 10 | Dynamic plasma point light + lens breathing | Lighting/Camera |
| 11 | Halation post-process (warm red film artifact) | Post-process |
| 12 | ACES tonemapping + film grain + vignette + CA | Pipeline |
| 13 | Conditional DOF during pinch + CA ramp | Pipeline |
| 14 | Glow+bloom stacking (edge+radiance) | Pipeline |
| 15 | Multi-preset verification (all 5 pass) | Testing |
| 16 | Performance: cached Color4 objects | Optimization |
| 17 | Anode thermal glow + insulator breakdown flash | Materials |
| 18 | Scientifically accurate D-alpha plasma colors | Physics |
| 19 | Motion afterimage ghost torus | Animation |
| 20 | HDR color convergence in glow layer | Post-process |
| 21 | Inferno colormap + field-specific routing | Colormaps |
| 22 | Heatmap Fresnel + emissive boost + flicker | Materials |
| 23 | Cinematic auto-zoom during pinch | Camera |
| 24 | Exposure pulse + flicker-driven intensity | Pipeline |
| 25 | Soft shadow generator on wireframe | Lighting |
| 26 | Exponential fog for atmospheric depth | Atmosphere |
| 27 | UX text fixes (D-alpha colors, inferno mention) | UX |
| 28 | Frame-rate independent animation playback | Performance |
| 29 | SSAO 32→16, shadow 1024→512 | Performance |

### Research Streams (informing the evolution)
1. Film VFX techniques (Interstellar, Oppenheimer, Marvel)
2. Museum exhibit design (EPFL tokamak, PPPL virtual tokamak)
3. Fusion startup pitch decks (CFS, Lockheed, TAE)
4. Game industry energy effects (Fresnel, bloom, noise)
5. Babylon.js forum techniques (GlowLayer, depth prepass, particles)
6. Data visualization best practices (Tufte, Moreland colormaps)
7. Plasma emission spectroscopy (D-alpha colors, blackbody)
8. VFX supervisor review (temporal coherence, light bleeding, surface response)
9. WebGL performance optimization

### Key Technical Decisions
- Wireframe ghost device eliminates clipping permanently
- StandardMaterial + custom GLSL (no PBR, no env textures)
- D-alpha red-orange emission colors (656nm, scientifically accurate)
- GlowLayer.addIncludedOnlyMesh (plasma-only glow)
- BLENDMODE_ONEONE for GPU particles
- needDepthPrePass on device materials
- Cylindrical heatmap wrap (360 degrees)
- Field-specific colormaps (viridis=density, inferno=temperature)
