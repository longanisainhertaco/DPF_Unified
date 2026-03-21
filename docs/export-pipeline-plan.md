# DPF Export Pipeline Plan (saved for future implementation)

## Phase 1: VTK Export Button (2-3 hours)
- Client-side JS generates VTK Legacy ASCII from simulation data
- "Download VTK" button in Gradio UI
- Readable by ParaView, VisIt, PyVista, meshio

## Phase 2: ParaView Python Script (4-8 hours)
- Bundled visualize_dpf.py using paraview.simple
- One command: pvpython visualize_dpf.py → publication images
- Volume rendering, B-field streamlines, scientific colorbars

## Phase 3: HDF5 Time-Series (1-2 days)
- h5wasm (NIST WASM library) for browser-side HDF5 writing
- XDMF+HDF5 for full animation data
- ParaView time slider support

## Phase 4: Blender Script (2-3 days, optional)
- Hero renders for investor/marketing materials
- Cycles path tracer for photorealistic plasma volumetrics

## Research backing: docs/visualization-research-synthesis.md
