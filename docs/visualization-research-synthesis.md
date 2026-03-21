# DPF Visualization Research Synthesis (2026-03-21)

## Three Research Streams Combined

### Stream 1: What Audiences Want

**The universal formula** (works for ALL audiences):
1. Start with the device — grounded in physical hardware
2. Trigger the discharge — visible sheath forming, accelerating
3. Show the invisible — B-field lines, pressure, current density
4. The pinch — compression with bloom, particle burst (the money shot)
5. Overlay the data — current traces, temperature readouts
6. Pull back — return to device exterior with understanding

**Per-audience needs:**
| Audience | Priority | Wow Trigger |
|---|---|---|
| Public | Visceral beauty | "I had no idea plasma looked like that" |
| Students | Conceptual clarity | "NOW I understand the magnetic field" |
| Researchers | Data fidelity | "Field lines match the MHD solution" |
| Investors | Credibility + vision | "This team built something real" |
| Engineers | Functional accuracy | "I can see the instability forming" |

**Key insights from film/VFX:**
- Interstellar: physics-accurate rendering looks MORE real than artistic approximation
- Oppenheimer: show what the physics FEELS like, not literal depictions
- Iron Man: wireframe + cyan glow + floating data = universal "advanced tech" language
- Imperfection signals reality — behavioral consistency matters more than photorealism

**From museums:**
- EPFL tokamak: game-quality rendering + adjustable physics parameters
- Two layers: visceral beauty for public, interactive data for experts

**From fusion startups:**
- CFS + Nvidia/Siemens: visualization IS the engineering tool (not decoration)
- Investor presentations use pre-rendered video, not live demos
- Visualization quality signals engineering quality

**From education:**
- MIT TEAL: 2x learning gains with 3D electromagnetic visualization
- Guided inquiry > free exploration
- Abstract-to-concrete transfer is the goal

**From games:**
- Fresnel rim glow = universal "energy" visual language
- Depth intersection highlights ground energy effects in physical reality
- Scrolling noise = organic turbulent flow
- Bloom is the single highest-impact post-process for wow factor

### Stream 2: Babylon.js Techniques (from forums + Reddit)

**Critical techniques to implement:**
1. `GlowLayer.addIncludedOnlyMesh()` — glow ONLY plasma meshes
2. `needDepthPrePass = true` on device materials — fixes transparency
3. `alphaIndex` for explicit render ordering
4. Wireframe clone + `zOffset = -1` prevents z-fighting
5. `BLENDMODE_ONEONE` for GPU particles (not MULTIPLYADD)
6. DynamicTexture for real-time heatmap updates
7. Custom GLSL plasma shader (noise + time = turbulent glow)
8. VolumetricLightScattering for pinch zone god rays

**Notable references:**
- SparkOne Labs: WebGL plasma around toroid magnet
- Virtual Beamline (VBL): WebXR for laser-plasma PIC viz
- Anu.js (JPMorgan): D3-style data viz for Babylon.js

### Stream 3: Export Pipeline Verdict

**YES — export is beneficial. Phased implementation:**

| Phase | Effort | Value | What |
|---|---|---|---|
| 1 | 2-3 hours | High | VTK Legacy ASCII export button (client-side JS) |
| 2 | 1 day | Very High | Bundled ParaView Python script (one command → publication images) |
| 3 | 1-2 days | High | HDF5 time-series via h5wasm (WASM in browser) |
| 4 | 2-3 days | Medium | Blender Python script for hero renders |

**Key findings:**
- ParaView is dominant in computational plasma physics (LLNL, Sandia, PPPL)
- VTK Legacy ASCII is trivially writable from JavaScript, readable by everything
- h5wasm (NIST) enables HDF5 writing from the browser (zero server dependency)
- Researchers want "give me a file I can open in ParaView" — 60-70% of serious users
- Investor presentations use pre-rendered video from offline tools
- The browser renderer serves 80% of interactions; export serves the critical 20%

## Design Principles for Final Renderer

### Tier 1: Non-Negotiable
1. Perceptually uniform colormaps (viridis/inferno) — never rainbow
2. Bloom/glow on plasma meshes only (GlowLayer.addIncludedOnlyMesh)
3. Data-driven motion — all plasma from simulation data
4. Fresnel rim glow on plasma surfaces
5. Proportionally accurate geometry

### Tier 2: High Impact
6. Intersection highlights where plasma contacts electrodes
7. B-field streamlines (blue, per NASA convention)
8. Time-series overlay synced to 3D animation
9. Scrolling noise on plasma for turbulent appearance
10. Cinematic camera presets

### Tier 3: Polish
11. Heat distortion near high-temperature regions
12. Scale indicators / human silhouette
13. Exploded/cutaway view toggle
14. VolumetricLightScattering at pinch
15. VTK export button

### Mode Switching
- **Presentation**: cinematic camera, bloom, dramatic, minimal UI
- **Analysis**: orthographic, colorbars, readouts, measurement
- **Education**: labeled field lines, parameter sliders, annotations
- **Publication**: clean background, colorbars, export-quality
