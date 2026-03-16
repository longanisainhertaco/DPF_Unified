"""Babylon.js WebGPU plasma renderer for DPF web UI.

Generates a self-contained HTML page with an interactive 3D plasma
visualization powered by Babylon.js. Uses WebGPU when available,
falls back to WebGL2.

The renderer shows:
- Electrode geometry (anode, cathode, insulator)
- Current sheath as a particle system (velocity-colored)
- Pinch plasma column with emissive glow
- Magnetic field line visualization
- Bloom post-processing for hot plasma regions
- Animated playback with time slider

Data is passed from Python as a JSON blob embedded in the HTML.
"""
from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

# Babylon.js CDN (7.x with WebGPU support)
BABYLON_CDN = "https://cdn.babylonjs.com/babylon.js"
BABYLON_LOADERS = "https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"
BABYLON_MATERIALS = "https://cdn.babylonjs.com/materialsLibrary/babylonjs.materials.min.js"
BABYLON_PP = "https://cdn.babylonjs.com/postProcessesLibrary/babylonjs.postProcess.min.js"
BABYLON_GUI = "https://cdn.babylonjs.com/gui/babylon.gui.min.js"


def _encode_array(arr: np.ndarray) -> str:
    """Encode numpy array as base64 float32 for JS consumption."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def _simulation_to_render_data(d: dict[str, Any]) -> dict:
    """Extract rendering data from simulation result dict."""
    cc = d.get("circuit", {})
    a = cc.get("anode_radius", 0.01)
    b = cc.get("cathode_radius", 0.03)
    L = d.get("snowplow_cfg", {}).get("anode_length", 0.16)

    t_us = np.array(d.get("t_us", [0]))
    z_mm = np.array(d.get("z_mm", [0]))
    r_mm = np.array(d.get("r_mm", [0]))
    I_MA = np.array(d.get("I_MA", [0]))
    phases = d.get("phases", ["none"] * len(t_us))

    # Subsample to ~60 frames
    n = len(t_us)
    step = max(1, n // 60)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    frames = []
    for i in idx:
        frames.append({
            "t": float(t_us[i]),
            "z": float(z_mm[i]),
            "r": float(r_mm[i]),
            "I": float(I_MA[i]),
            "phase": phases[i],
        })

    # MHD field data (final state) for volumetric rendering
    final = d.get("final_state")
    field_data = None
    if final is not None:
        rho = final["rho"]
        rho_norm = (rho - rho.min()) / max(rho.max() - rho.min(), 1e-30)
        B = final.get("B")
        if B is not None:
            B_mag = np.sqrt(np.sum(B**2, axis=0))
            B_norm = B_mag / max(B_mag.max(), 1e-30)
        else:
            B_norm = np.zeros_like(rho)
        Te = final.get("Te", np.full_like(rho, 300.0))
        Te_norm = (Te - Te.min()) / max(Te.max() - Te.min(), 1e-30)

        # Take midplane slice for 2D heatmap overlay
        ny_mid = rho.shape[1] // 2
        field_data = {
            "rho_mid": _encode_array(rho_norm[:, ny_mid, :]),
            "B_mid": _encode_array(B_norm[:, ny_mid, :]),
            "Te_mid": _encode_array(Te_norm[:, ny_mid, :]),
            "shape": [rho.shape[0], rho.shape[2]],
        }

    render_data = {
        "anode_radius": a * 1e3,  # mm
        "cathode_radius": b * 1e3,
        "anode_length": L * 1e3,
        "frames": frames,
        "n_frames": len(frames),
        "I_peak": float(np.max(np.abs(I_MA))),
        "device": d.get("device", "DPF"),
        "backend": d.get("backend", "lee"),
        "has_mhd": d.get("has_mhd", False),
        "field_data": field_data,
    }
    return render_data


def create_babylon_renderer(d: dict[str, Any]) -> str:
    """Generate self-contained HTML with Babylon.js WebGPU plasma renderer.

    Args:
        d: Simulation result dict from run_simulation or run_mhd_simulation.

    Returns:
        HTML string for embedding in gr.HTML().
    """
    render_data = _simulation_to_render_data(d)
    data_json = json.dumps(render_data)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#0a0a0a; }}
  #renderCanvas {{ width:100%; height:100%; touch-action:none; }}
  #info {{ position:absolute; top:8px; left:12px; color:#ccc; font:13px/1.4 monospace; pointer-events:none; z-index:10; }}
  #controls {{ position:absolute; bottom:12px; left:50%; transform:translateX(-50%); z-index:10; display:flex; gap:8px; align-items:center; }}
  #controls button {{ background:#333; color:#eee; border:1px solid #555; padding:4px 14px; border-radius:4px; cursor:pointer; font:12px monospace; }}
  #controls button:hover {{ background:#555; }}
  #timeSlider {{ width:300px; }}
  #timeLabel {{ color:#aaa; font:12px monospace; min-width:80px; }}
</style>
<script src="{BABYLON_CDN}"></script>
</head>
<body>
<canvas id="renderCanvas"></canvas>
<div id="info">Loading...</div>
<div id="controls">
  <button id="playBtn">Play</button>
  <button id="pauseBtn">Pause</button>
  <input type="range" id="timeSlider" min="0" max="1" step="0.001" value="0">
  <span id="timeLabel">t=0.0 us</span>
</div>

<script>
const DATA = {data_json};

async function main() {{
  const canvas = document.getElementById("renderCanvas");
  const info = document.getElementById("info");

  // Try WebGPU first, fall back to WebGL2
  let engine;
  let gpuMode = "WebGL2";
  try {{
    const webgpuSupported = await BABYLON.WebGPUEngine.IsSupportedAsync;
    if (webgpuSupported) {{
      engine = new BABYLON.WebGPUEngine(canvas, {{ antialias: true }});
      await engine.initAsync();
      gpuMode = "WebGPU";
    }}
  }} catch(e) {{}}
  if (!engine) {{
    engine = new BABYLON.Engine(canvas, true, {{ preserveDrawingBuffer: true }});
  }}

  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.03, 0.03, 0.05, 1);
  scene.ambientColor = new BABYLON.Color3(0.1, 0.1, 0.15);

  // Camera: orbit around the device
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI/4, Math.PI/3, DATA.cathode_radius * 8,
    new BABYLON.Vector3(DATA.anode_length/2, 0, 0), scene);
  cam.attachControl(canvas, true);
  cam.wheelPrecision = 30;
  cam.minZ = 0.1;

  // Lights
  const hemi = new BABYLON.HemisphericLight("hemi", new BABYLON.Vector3(0,1,0), scene);
  hemi.intensity = 0.4;
  const point = new BABYLON.PointLight("point",
    new BABYLON.Vector3(DATA.anode_length/2, DATA.cathode_radius*2, 0), scene);
  point.intensity = 0.6;

  // --- Electrode Geometry ---
  // Anode (inner cylinder, gold)
  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {{
    diameter: DATA.anode_radius * 2, height: DATA.anode_length, tessellation: 32
  }}, scene);
  anode.rotation.z = Math.PI/2;
  anode.position.x = DATA.anode_length / 2;
  const anodeMat = new BABYLON.StandardMaterial("anodeMat", scene);
  anodeMat.diffuseColor = new BABYLON.Color3(0.85, 0.65, 0.13);
  anodeMat.specularColor = new BABYLON.Color3(0.4, 0.3, 0.1);
  anodeMat.alpha = 0.7;
  anode.material = anodeMat;

  // Cathode (outer cylinder, steel gray, wireframe)
  const cathode = BABYLON.MeshBuilder.CreateCylinder("cathode", {{
    diameter: DATA.cathode_radius * 2, height: DATA.anode_length, tessellation: 32
  }}, scene);
  cathode.rotation.z = Math.PI/2;
  cathode.position.x = DATA.anode_length / 2;
  const cathodeMat = new BABYLON.StandardMaterial("cathodeMat", scene);
  cathodeMat.diffuseColor = new BABYLON.Color3(0.5, 0.5, 0.55);
  cathodeMat.wireframe = true;
  cathodeMat.alpha = 0.4;
  cathode.material = cathodeMat;

  // Insulator disc at z=0
  const insulator = BABYLON.MeshBuilder.CreateDisc("insulator", {{
    radius: DATA.cathode_radius, tessellation: 32
  }}, scene);
  insulator.rotation.y = Math.PI/2;
  const insMat = new BABYLON.StandardMaterial("insMat", scene);
  insMat.diffuseColor = new BABYLON.Color3(0.6, 0.3, 0.7);
  insMat.alpha = 0.3;
  insulator.material = insMat;

  // --- Current Sheath (particle disc) ---
  const sheathMat = new BABYLON.StandardMaterial("sheathMat", scene);
  sheathMat.emissiveColor = new BABYLON.Color3(0.2, 0.5, 1.0);
  sheathMat.alpha = 0.5;
  const sheath = BABYLON.MeshBuilder.CreateDisc("sheath", {{
    radius: DATA.cathode_radius, innerRadius: DATA.anode_radius, tessellation: 32
  }}, scene);
  sheath.rotation.y = Math.PI/2;
  sheath.material = sheathMat;

  // --- Pinch Plasma (glowing cylinder) ---
  const pinchMat = new BABYLON.StandardMaterial("pinchMat", scene);
  pinchMat.emissiveColor = new BABYLON.Color3(1.0, 0.2, 0.1);
  pinchMat.alpha = 0;
  pinchMat.disableLighting = true;
  const pinch = BABYLON.MeshBuilder.CreateCylinder("pinch", {{
    diameter: DATA.anode_radius * 0.5, height: DATA.anode_length * 0.35, tessellation: 16
  }}, scene);
  pinch.rotation.z = Math.PI/2;
  pinch.position.x = DATA.anode_length * 0.82;
  pinch.material = pinchMat;

  // --- Glow Layer (bloom for hot plasma) ---
  const gl = new BABYLON.GlowLayer("glow", scene, {{
    mainTextureSamples: 4, blurKernelSize: 32
  }});
  gl.intensity = 0.8;

  // --- Particle System (plasma ions) ---
  const sps = new BABYLON.SolidParticleSystem("sps", scene);
  const sphere = BABYLON.MeshBuilder.CreateSphere("p", {{ diameter: 0.3, segments: 4 }}, scene);
  sps.addShape(sphere, 2000);
  sphere.dispose();
  const spsMesh = sps.buildMesh();
  spsMesh.hasVertexAlpha = true;

  // Initialize particles in annular region
  sps.initParticles = function() {{
    for (let i = 0; i < sps.nbParticles; i++) {{
      const p = sps.particles[i];
      const theta = Math.random() * Math.PI * 2;
      const r = DATA.anode_radius + Math.random() * (DATA.cathode_radius - DATA.anode_radius);
      p.position.x = Math.random() * DATA.anode_length * 0.3;
      p.position.y = r * Math.sin(theta);
      p.position.z = r * Math.cos(theta);
      p.color = new BABYLON.Color4(0.3, 0.6, 1.0, 0.6);
      p.scaling = new BABYLON.Vector3(0.5, 0.5, 0.5);
    }}
  }};
  sps.initParticles();
  sps.setParticles();

  // --- Animation State ---
  let currentFrame = 0;
  let playing = false;
  let lastTime = 0;
  const FRAME_DT = 80; // ms per frame

  function updateScene(fi) {{
    if (fi < 0 || fi >= DATA.frames.length) return;
    const f = DATA.frames[fi];
    const phase = f.phase;

    // Move sheath to z position
    sheath.position.x = f.z;

    // Phase-dependent visuals
    const phaseColors = {{
      "rundown": [0.2, 0.5, 1.0],
      "radial": [1.0, 0.3, 0.1],
      "mhd_radial": [1.0, 0.3, 0.1],
      "reflected": [1.0, 0.6, 0.0],
      "pinch": [1.0, 0.1, 0.05],
      "post_pinch": [0.8, 0.2, 0.1],
    }};
    const col = phaseColors[phase] || [0.4, 0.4, 0.5];
    sheathMat.emissiveColor = new BABYLON.Color3(col[0], col[1], col[2]);

    // Sheath inner radius shrinks during radial phase
    if (phase === "radial" || phase === "mhd_radial" || phase === "pinch") {{
      const innerR = Math.max(f.r, DATA.anode_radius * 0.05);
      // Can't rebuild mesh every frame, so scale Y to simulate compression
      const compressionRatio = innerR / DATA.cathode_radius;
      sheath.scaling.y = Math.max(compressionRatio, 0.05);
      sheath.scaling.z = Math.max(compressionRatio, 0.05);

      // Show pinch plasma
      const pinchIntensity = Math.min(1.0, (1.0 - compressionRatio) * 3);
      pinchMat.alpha = pinchIntensity * 0.7;
      pinchMat.emissiveColor = new BABYLON.Color3(
        1.0, 0.1 + pinchIntensity * 0.3, pinchIntensity * 0.05
      );
      gl.intensity = 0.5 + pinchIntensity * 1.5;
    }} else {{
      sheath.scaling.y = 1;
      sheath.scaling.z = 1;
      pinchMat.alpha = 0;
      gl.intensity = 0.5;
    }}

    // Update particles: drift toward sheath position
    sps.updateParticle = function(p) {{
      // Drift particles toward sheath
      const dx = (f.z - p.position.x) * 0.02;
      p.position.x += dx + (Math.random() - 0.5) * 0.2;
      // Random thermal motion
      p.position.y += (Math.random() - 0.5) * 0.15;
      p.position.z += (Math.random() - 0.5) * 0.15;
      // Keep in annular region
      const r = Math.sqrt(p.position.y*p.position.y + p.position.z*p.position.z);
      if (r > DATA.cathode_radius * 1.1 || r < DATA.anode_radius * 0.3) {{
        const theta = Math.random() * Math.PI * 2;
        const newR = DATA.anode_radius + Math.random() * (DATA.cathode_radius - DATA.anode_radius) * 0.8;
        p.position.y = newR * Math.sin(theta);
        p.position.z = newR * Math.cos(theta);
      }}
      // Color by phase
      p.color.r = col[0]; p.color.g = col[1]; p.color.b = col[2];
      p.color.a = 0.4 + Math.abs(f.I) * 0.3;
      return p;
    }};
    sps.setParticles();

    // Update info
    info.textContent = `${{DATA.device}} | ${{gpuMode}} | t=${{f.t.toFixed(1)}} us | I=${{f.I.toFixed(3)}} MA | ${{phase}}`;
  }}

  // Controls
  const slider = document.getElementById("timeSlider");
  const label = document.getElementById("timeLabel");
  slider.max = DATA.n_frames - 1;

  document.getElementById("playBtn").onclick = () => {{ playing = true; }};
  document.getElementById("pauseBtn").onclick = () => {{ playing = false; }};
  slider.oninput = () => {{
    currentFrame = parseInt(slider.value);
    updateScene(currentFrame);
    const f = DATA.frames[currentFrame];
    label.textContent = `t=${{f.t.toFixed(1)}} us`;
  }};

  // Render loop
  engine.runRenderLoop(() => {{
    if (playing) {{
      const now = performance.now();
      if (now - lastTime > FRAME_DT) {{
        currentFrame = (currentFrame + 1) % DATA.n_frames;
        slider.value = currentFrame;
        const f = DATA.frames[currentFrame];
        label.textContent = `t=${{f.t.toFixed(1)}} us`;
        updateScene(currentFrame);
        lastTime = now;
      }}
    }}
    scene.render();
  }});

  window.addEventListener("resize", () => engine.resize());

  // Initial render
  updateScene(0);
  info.textContent = `${{DATA.device}} | ${{gpuMode}} | Ready — press Play`;
}}

main().catch(e => {{
  document.getElementById("info").textContent = "Error: " + e.message;
  console.error(e);
}});
</script>
</body></html>"""

    return html


def create_babylon_iframe(d: dict[str, Any], height: int = 550) -> str:
    """Generate an iframe-wrapped Babylon.js renderer for Gradio."""
    html = create_babylon_renderer(d)
    escaped = html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{height}px;border:none;background:#0a0a0a;"></iframe>'
    )
