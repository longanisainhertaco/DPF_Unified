# Babylon.js Techniques for DPF Renderer

## Key Findings (from forum + Reddit research, 2026-03-21)

### GlowLayer — Per-Mesh Control
- `gl.addIncludedOnlyMesh(plasmaMesh)` — glow ONLY plasma
- `customEmissiveColorSelector` for per-mesh glow color
- GlowLayer uses emissive properties; BloomEffect uses luminance thresholds
- For our case: GlowLayer with selective inclusion is correct

### Fresnel Edge Glow
- `emissiveFresnelParameters`: bias=0.2, power=4 for energy field look
- leftColor = edge color, rightColor = center color
- Combine with low alpha (0.2) for transparent energy field
- Node Material Fresnel for advanced remap control

### Transparency Sorting
- `renderingGroupId`: 0=device, 1=plasma (critical)
- `needDepthPrePass = true` on device materials
- `alphaIndex` for fine ordering within groups
- `separateCullingPass = true` for concave transparent meshes
- Order Independent Transparency available but heavy

### Wireframe Overlay
- Clone mesh + wireframe material + `zOffset = -1`
- Prevents z-fighting with solid mesh underneath
- `disableLighting = true` for consistent wireframe color

### Materials Without Environment
- PBR: `mat._getReflectionTexture = () => null` or `mat.unlit = true`
- StandardMaterial with emissive for plasma (no env needed)
- `BABYLON.MaterialFlags.ReflectionTextureEnabled = false` globally

### Particle Systems
- `BLENDMODE_ONEONE` for additive (not MULTIPLYADD with GPU particles)
- Custom soft PNG texture (not default flare.png)
- DynamicTexture for procedural particle texture

### Heatmap on Curved Surfaces
- DynamicTexture + `emissiveTexture` for real-time data
- UV must be homogeneous for cylinder meshes
- World-position sampling avoids UV seam artifacts
- Node Material blue-channel lerp for overlay blending

### Custom Shaders
- Plasma noise: multi-frequency sin() with time offset
- SDF glow: inverse distance falloff
- Energy shield: Fresnel + depth intersection + noise scroll

### Volumetric Effects
- VolumetricLightScatteringPostProcess for god rays at pinch
- Disappears when source mesh off-screen (fixed position OK for us)

### WebGPU
- Target WebGL2 as primary, WebGPU optional
- Mixed performance results in 2025-2026
- Compute shader advantages not mature enough in Babylon pipeline

### Notable Libraries
- Anu.js (JPMorgan) — D3-style data viz for Babylon.js
- SparkOne Labs — WebGL plasma simulation around toroid
- Virtual Beamline (VBL) — WebGL+WebXR for laser-plasma PIC visualization
