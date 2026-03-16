"""Babylon.js WebGPU plasma renderer for DPF web UI.

Generates a self-contained HTML page with an interactive 3D plasma
visualization powered by Babylon.js 7.x. Uses WebGPU when available,
falls back to WebGL2.

Features:
- Electrode geometry (copper anode, 8 cathode rods, ceramic insulator)
- GPUParticleSystem for 50K plasma ions (velocity-colored, additive blend)
- Two-layer pinch plasma (bright core + dim halo) with emissive glow
- DefaultRenderingPipeline (bloom, chromatic aberration, ACES tone mapping)
- GlowLayer for mesh-level emission
- ArcRotateCamera with orbit controls
- Animated playback with phase-dependent visuals
- Field overlay (density/|B|/Te heatmap on midplane)
"""
from __future__ import annotations

import base64
import html as html_mod
import json
from typing import Any

import numpy as np

BABYLON_CDN = "https://cdn.babylonjs.com/babylon.js"


def _encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-30)


def _simulation_to_render_data(d: dict[str, Any]) -> dict:
    cc = d.get("circuit", {})
    a = cc.get("anode_radius", 0.01)
    b = cc.get("cathode_radius", 0.03)
    L = d.get("snowplow_cfg", {}).get("anode_length", 0.16)

    t_us = np.array(d.get("t_us", [0]))
    z_mm = np.array(d.get("z_mm", [0]))
    r_mm = np.array(d.get("r_mm", [0]))
    I_MA = np.array(d.get("I_MA", [0]))
    phases = d.get("phases", ["none"] * len(t_us))

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

    final = d.get("final_state")
    field_data = None
    if final is not None:
        rho = final["rho"]
        Te = final.get("Te", np.full_like(rho, 300.0))
        B = final.get("B")
        ny_mid = rho.shape[1] // 2 if rho.ndim == 3 else 0

        def _mid(x):
            return x[:, ny_mid, :] if x.ndim == 3 else x

        field_data = {
            "rho_mid": _encode_array(_norm(_mid(rho))),
            "Te_mid": _encode_array(_norm(_mid(Te))),
            "shape": [_mid(rho).shape[0], _mid(rho).shape[1]],
        }
        if B is not None:
            B_mag = np.sqrt(np.sum(B**2, axis=0))
            field_data["B_mid"] = _encode_array(_norm(_mid(B_mag)))

    return {
        "anode_radius": a * 1e3,
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


def create_babylon_renderer(d: dict[str, Any]) -> str:
    render_data = _simulation_to_render_data(d)
    data_json = json.dumps(render_data)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#050508}}
  #renderCanvas{{width:100%;height:100%;touch-action:none;display:block}}
  #hud{{position:absolute;top:8px;left:10px;color:#9cf;font:12px/1.6 monospace;pointer-events:none;z-index:10;text-shadow:0 0 6px #00f8}}
  #controls{{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);z-index:10;display:flex;gap:8px;align-items:center;background:rgba(0,0,0,0.55);padding:6px 14px;border-radius:8px}}
  #controls button{{background:#1a1a2e;color:#9cf;border:1px solid #336;padding:4px 12px;border-radius:4px;cursor:pointer;font:12px monospace}}
  #controls button:hover{{background:#223}}
  #timeSlider{{width:260px;accent-color:#48f}}
  #timeLabel{{color:#8af;font:12px monospace;min-width:90px}}
  #fieldSelect{{background:#111;color:#9cf;border:1px solid #336;font:12px monospace;border-radius:3px;padding:2px}}
</style>
<script src="{BABYLON_CDN}"></script>
</head>
<body>
<canvas id="renderCanvas"></canvas>
<div id="hud">Initializing...</div>
<div id="controls">
  <button id="playBtn">Play</button>
  <button id="pauseBtn">Pause</button>
  <button id="resetBtn">Reset</button>
  <input type="range" id="timeSlider" min="0" max="1" step="1" value="0">
  <span id="timeLabel">t=0.0 us</span>
  <select id="fieldSelect">
    <option value="none">Overlay: None</option>
    <option value="rho">Density</option>
    <option value="B">|B| field</option>
    <option value="Te">Temperature</option>
  </select>
</div>

<script>
const DATA = {data_json};
const PHASE_COLORS = {{
  rundown:[0.2,0.5,1.0], radial:[1.0,0.3,0.1], mhd_radial:[1.0,0.3,0.1],
  reflected:[1.0,0.6,0.0], pinch:[1.0,0.1,0.05], post_pinch:[0.8,0.2,0.1]
}};
const PHASE_LABELS = {{
  rundown:"Axial rundown", radial:"Radial implosion", mhd_radial:"MHD radial",
  reflected:"Reflected shock", pinch:"Pinch", post_pinch:"Post-pinch", none:""
}};

function decodeF32(b64, shape) {{
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const u8 = new Uint8Array(buf);
  for(let i=0;i<raw.length;i++) u8[i]=raw.charCodeAt(i);
  return {{data:new Float32Array(buf), shape}};
}}

async function main() {{
  const canvas = document.getElementById("renderCanvas");
  const hud = document.getElementById("hud");

  let engine, gpuMode="WebGL2";
  try {{
    if(await BABYLON.WebGPUEngine.IsSupportedAsync) {{
      engine = new BABYLON.WebGPUEngine(canvas, {{antialias:true, adaptToDeviceRatio:true, powerPreference:"high-performance"}});
      await engine.initAsync();
      gpuMode = "WebGPU";
    }}
  }} catch(_) {{}}
  if(!engine) engine = new BABYLON.Engine(canvas, true, {{preserveDrawingBuffer:false, stencil:true, adaptToDeviceRatio:true}});

  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.02,0.02,0.04,1);
  scene.ambientColor = new BABYLON.Color3(0.08,0.08,0.12);

  // Camera
  const D = DATA;
  const cam = new BABYLON.ArcRotateCamera("cam", -Math.PI/3, Math.PI/3.5,
    D.cathode_radius*10, new BABYLON.Vector3(D.anode_length/2,0,0), scene);
  cam.attachControl(canvas,true);
  cam.lowerRadiusLimit = D.cathode_radius*2.5;
  cam.upperRadiusLimit = D.cathode_radius*30;
  cam.wheelPrecision = 30;
  cam.minZ = 0.05;
  cam.inertia = 0.75;

  // Lights
  new BABYLON.HemisphericLight("hemi", new BABYLON.Vector3(0,1,0.3), scene).intensity = 0.35;
  const pt = new BABYLON.PointLight("pt", new BABYLON.Vector3(D.anode_length/2, D.cathode_radius*2, D.cathode_radius), scene);
  pt.intensity = 0.5;

  // ---- Electrodes ----
  // Anode (copper)
  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {{
    diameter:D.anode_radius*2, height:D.anode_length, tessellation:48, cap:BABYLON.Mesh.CAP_ALL
  }}, scene);
  anode.rotation.z = Math.PI/2;
  anode.position.x = D.anode_length/2;
  const anodeMat = new BABYLON.StandardMaterial("anodeMat", scene);
  anodeMat.diffuseColor = new BABYLON.Color3(0.90,0.72,0.18);
  anodeMat.specularColor = new BABYLON.Color3(0.5,0.4,0.1);
  anodeMat.specularPower = 64;
  anode.material = anodeMat;

  // Cathode (8 steel rods)
  const N_RODS = 8;
  for(let i=0;i<N_RODS;i++) {{
    const angle = (i/N_RODS)*Math.PI*2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod"+i, {{
      diameter:D.cathode_radius*0.08, height:D.anode_length, tessellation:8
    }}, scene);
    rod.rotation.z = Math.PI/2;
    rod.position.x = D.anode_length/2;
    rod.position.y = D.cathode_radius*Math.sin(angle);
    rod.position.z = D.cathode_radius*Math.cos(angle);
    const m = new BABYLON.StandardMaterial("rodM"+i, scene);
    m.diffuseColor = new BABYLON.Color3(0.55,0.55,0.60);
    m.alpha = 0.8;
    rod.material = m;
  }}

  // Insulator disc
  const ins = BABYLON.MeshBuilder.CreateCylinder("ins", {{
    diameter:D.cathode_radius*2, height:D.anode_radius*0.3, tessellation:48
  }}, scene);
  ins.rotation.z = Math.PI/2;
  ins.position.x = -D.anode_radius*0.15;
  const insMat = new BABYLON.StandardMaterial("insM", scene);
  insMat.diffuseColor = new BABYLON.Color3(0.85,0.80,0.55);
  insMat.alpha = 0.5;
  ins.material = insMat;

  // ---- Current sheath (torus = donut sweeping gas) ----
  const sheathMat = new BABYLON.StandardMaterial("sheath", scene);
  sheathMat.emissiveColor = new BABYLON.Color3(0.2,0.5,1.0);
  sheathMat.alpha = 0.55;
  sheathMat.disableLighting = true;
  sheathMat.backFaceCulling = false;
  // Torus: major radius = midpoint between anode and cathode, tube = gap/3
  const sheathR = (D.anode_radius + D.cathode_radius) / 2;
  const sheathTube = (D.cathode_radius - D.anode_radius) / 2.5;
  const sheath = BABYLON.MeshBuilder.CreateTorus("sheath", {{
    diameter: sheathR * 2, thickness: sheathTube * 2, tessellation: 32
  }}, scene);
  sheath.rotation.z = Math.PI/2;  // align torus ring perpendicular to z-axis
  sheath.material = sheathMat;

  // Swept plasma trail (faint glow behind the sheath)
  const trailMat = new BABYLON.StandardMaterial("trailMat", scene);
  trailMat.emissiveColor = new BABYLON.Color3(0.1,0.2,0.5);
  trailMat.alpha = 0.15;
  trailMat.disableLighting = true;
  trailMat.backFaceCulling = false;
  const trail = BABYLON.MeshBuilder.CreateCylinder("trail", {{
    diameter:(D.anode_radius+D.cathode_radius), height:1, tessellation:24
  }}, scene);
  trail.rotation.z = Math.PI/2;
  trail.material = trailMat;

  // ---- Pinch plasma (core + halo) ----
  const coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(1.0,0.4,0.1);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  const core = BABYLON.MeshBuilder.CreateCylinder("pinchCore", {{
    diameter:D.anode_radius*0.3, height:D.anode_length*0.4, tessellation:20
  }}, scene);
  core.rotation.z = Math.PI/2;
  core.position.x = D.anode_length*0.82;
  core.material = coreMat;

  const haloMat = new BABYLON.StandardMaterial("haloMat", scene);
  haloMat.emissiveColor = new BABYLON.Color3(0.8,0.15,0.05);
  haloMat.disableLighting = true;
  haloMat.alpha = 0;
  haloMat.backFaceCulling = false;
  const halo = BABYLON.MeshBuilder.CreateCylinder("pinchHalo", {{
    diameter:D.anode_radius*0.8, height:D.anode_length*0.5, tessellation:20,
    sideOrientation:BABYLON.Mesh.BACKSIDE
  }}, scene);
  halo.rotation.z = Math.PI/2;
  halo.position.x = D.anode_length*0.82;
  halo.material = haloMat;

  // ---- GPU Particle System (plasma ions) ----
  const useGPU = BABYLON.GPUParticleSystem.IsSupported;
  const PSCtor = useGPU ? BABYLON.GPUParticleSystem : BABYLON.ParticleSystem;
  const capacity = useGPU ? 50000 : 4000;
  const ps = new PSCtor("ions", {{capacity}}, scene);
  ps.emitter = new BABYLON.Vector3(0,0,0);

  const emitter = new BABYLON.SphereParticleEmitter();
  emitter.radius = D.cathode_radius*0.85;
  emitter.radiusRange = 0.35;
  ps.particleEmitterType = emitter;

  ps.minLifeTime = 0.06;
  ps.maxLifeTime = 0.18;
  ps.emitRate = useGPU ? 10000 : 600;
  ps.minSize = 0.06;
  ps.maxSize = 0.22;
  ps.minEmitPower = 0.5;
  ps.maxEmitPower = 2.5;

  ps.addColorGradient(0.0, new BABYLON.Color4(0.1,0.3,1.0,0.0));
  ps.addColorGradient(0.2, new BABYLON.Color4(0.3,0.7,1.0,0.7));
  ps.addColorGradient(0.6, new BABYLON.Color4(1.0,0.9,0.5,0.8));
  ps.addColorGradient(1.0, new BABYLON.Color4(1.0,0.2,0.1,0.0));

  ps.addSizeGradient(0.0, 0.04);
  ps.addSizeGradient(0.3, 0.18);
  ps.addSizeGradient(0.8, 0.12);
  ps.addSizeGradient(1.0, 0.0);

  ps.isBillboardBased = true;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;

  // Inline 8x8 bright dot texture
  ps.particleTexture = new BABYLON.Texture(
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAP0lEQVQY02P4z8DwHwMDw38GBgYGJiCBDYMEMDAw/Gf4z/CfAQv/k4EFA3CAgQkHAAAAAElFTkSuQmCC",
    scene);
  ps.start();

  // ---- Post-Processing Pipeline ----
  const pipeline = new BABYLON.DefaultRenderingPipeline("dflt", true, scene, [cam]);
  pipeline.bloomEnabled = true;
  pipeline.bloomThreshold = 0.6;
  pipeline.bloomWeight = 0.5;
  pipeline.bloomKernel = 64;
  pipeline.bloomScale = 0.5;
  pipeline.chromaticAberrationEnabled = true;
  pipeline.chromaticAberration.aberrationAmount = 1.2;
  pipeline.imageProcessingEnabled = true;
  pipeline.imageProcessing.toneMappingEnabled = true;
  pipeline.imageProcessing.toneMappingType = BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pipeline.imageProcessing.exposure = 1.15;
  pipeline.imageProcessing.contrast = 1.05;

  // ---- Glow Layer ----
  const gl = new BABYLON.GlowLayer("glow", scene, {{
    mainTextureSamples:2, blurKernelSize:32, mainTextureFixedSize:512
  }});
  gl.intensity = 0.5;
  gl.customEmissiveColorSelector = (mesh, _s, _m, result) => {{
    const glowNames = ["pinchCore","pinchHalo","sheath"];
    if(glowNames.includes(mesh.name)) {{
      result.set(mesh.material.emissiveColor.r, mesh.material.emissiveColor.g,
                 mesh.material.emissiveColor.b, mesh.material.alpha);
    }} else {{ result.set(0,0,0,0); }}
  }};

  // ---- Animation ----
  let currentFrame=0, playing=false, lastAdv=0;
  const FRAME_MS = 80;
  const slider = document.getElementById("timeSlider");
  const label = document.getElementById("timeLabel");
  slider.max = D.n_frames - 1;

  function applyFrame(fi) {{
    if(fi<0||fi>=D.frames.length) return;
    const f = D.frames[fi];
    const col = PHASE_COLORS[f.phase]||[0.4,0.4,0.5];

    // Sheath position + color
    sheath.position.x = f.z;
    sheathMat.emissiveColor.set(col[0],col[1],col[2]);

    // Swept plasma trail: stretches from insulator to sheath position
    const trailLen = Math.max(f.z, 0.5);
    trail.scaling.x = trailLen;
    trail.position.x = f.z / 2;
    trailMat.emissiveColor.set(col[0]*0.4, col[1]*0.4, col[2]*0.4);
    trailMat.alpha = 0.12 + Math.abs(f.I) * 0.08;

    // Compression: during radial phase, torus shrinks (donut compressing inward)
    const isPinch = ["radial","mhd_radial","pinch","reflected","post_pinch"].includes(f.phase);
    const cr = Math.max(0.03, f.r / D.cathode_radius);
    if (isPinch) {{
      // Torus shrinks: scale Y and Z to compress the ring inward
      sheath.scaling.y = cr;
      sheath.scaling.z = cr;
      // Move to anode tip during radial phase
      sheath.position.x = D.anode_length;
      sheathMat.alpha = 0.65;
    }} else {{
      sheath.scaling.y = 1;
      sheath.scaling.z = 1;
      sheathMat.alpha = 0.55;
    }}

    // Pinch intensity
    const pInt = isPinch ? Math.min(1.0, Math.pow(1.0-cr,2)*2.5) : 0;
    coreMat.alpha = pInt * 0.85;
    haloMat.alpha = pInt * 0.35;
    coreMat.emissiveColor.set(1.0, pInt*0.6, pInt*0.4);
    haloMat.emissiveColor.set(1.0, pInt*0.25, 0.05);
    const rScale = Math.max(0.05, cr*0.6);
    core.scaling.set(1, rScale, rScale);
    halo.scaling.set(1, rScale*2.5, rScale*2.5);
    gl.intensity = 0.4 + pInt*2.2;

    // Particles follow sheath
    ps.emitter.x = f.z;
    if(f.phase==="rundown") {{
      ps.gravity = new BABYLON.Vector3(2,0,0);
      ps.minEmitPower=1; ps.maxEmitPower=3;
    }} else if(isPinch) {{
      ps.gravity = new BABYLON.Vector3(0, -f.r*0.5, 0);
      ps.minEmitPower=2; ps.maxEmitPower=6;
    }}

    const phaseLabel = PHASE_LABELS[f.phase]||f.phase;
    hud.textContent = D.device+" | "+gpuMode+" | t="+f.t.toFixed(1)+" us | I="+f.I.toFixed(3)+" MA | "+phaseLabel;
  }}

  document.getElementById("playBtn").onclick = ()=>{{playing=true}};
  document.getElementById("pauseBtn").onclick = ()=>{{playing=false}};
  document.getElementById("resetBtn").onclick = ()=>{{currentFrame=0;slider.value=0;applyFrame(0);playing=false;
    label.textContent="t="+D.frames[0].t.toFixed(1)+" us"}};
  slider.oninput = ()=>{{
    currentFrame=parseInt(slider.value);
    applyFrame(currentFrame);
    label.textContent="t="+D.frames[currentFrame].t.toFixed(1)+" us";
  }};

  engine.runRenderLoop(()=>{{
    if(playing) {{
      const now=performance.now();
      if(now-lastAdv>FRAME_MS) {{
        currentFrame=(currentFrame+1)%D.n_frames;
        slider.value=currentFrame;
        label.textContent="t="+D.frames[currentFrame].t.toFixed(1)+" us";
        applyFrame(currentFrame);
        lastAdv=now;
      }}
    }}
    scene.render();
  }});
  window.addEventListener("resize", ()=>engine.resize());
  applyFrame(0);
  hud.textContent = D.device+" | "+gpuMode+" | Ready";
}}

main().catch(e=>{{
  document.getElementById("hud").textContent="Error: "+e.message;
  console.error(e);
}});
</script>
</body></html>"""


def create_babylon_iframe(d: dict[str, Any], height: int = 580) -> str:
    html = create_babylon_renderer(d)
    escaped = html_mod.escape(html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{height}px;border:none;background:#050508;" '
        f'allow="accelerometer; camera; gyroscope; xr-spatial-tracking" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def create_cross_section_renderer(d: dict[str, Any]) -> str:
    """Babylon.js 2D cross-section: r-z plane with electrodes + glowing sheath + pinch."""
    render_data = _simulation_to_render_data(d)
    data_json = json.dumps(render_data)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#060610}}
  #renderCanvas{{width:100%;height:100%;touch-action:none;display:block}}
  #hud{{position:absolute;top:6px;left:10px;color:#9cf;font:12px/1.5 monospace;pointer-events:none;z-index:10;text-shadow:0 0 4px #00f6}}
  #ctrl{{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:10;display:flex;gap:6px;align-items:center;background:rgba(0,0,0,0.5);padding:5px 12px;border-radius:6px}}
  #ctrl button{{background:#1a1a2e;color:#9cf;border:1px solid #336;padding:3px 10px;border-radius:3px;cursor:pointer;font:11px monospace}}
  #ctrl button:hover{{background:#223}}
  #tSlider{{width:240px;accent-color:#48f}}
  #tLabel{{color:#8af;font:11px monospace;min-width:80px}}
</style>
<script src="{BABYLON_CDN}"></script>
</head>
<body>
<canvas id="renderCanvas"></canvas>
<div id="hud">Loading...</div>
<div id="ctrl">
  <button id="playBtn">Play</button>
  <button id="pauseBtn">Pause</button>
  <input type="range" id="tSlider" min="0" max="1" step="1" value="0">
  <span id="tLabel">t=0.0 us</span>
</div>
<script>
const D = {data_json};
const PC = {{
  rundown:[0.15,0.4,1.0], radial:[1.0,0.25,0.08], mhd_radial:[1.0,0.25,0.08],
  reflected:[1.0,0.55,0.0], pinch:[1.0,0.08,0.03], post_pinch:[0.7,0.15,0.08]
}};
const PL = {{
  rundown:"Axial rundown — sheath sweeps gas",
  radial:"Radial implosion — plasma compressing",
  mhd_radial:"MHD radial implosion",
  reflected:"Reflected shock",
  pinch:"Pinch — peak compression",
  post_pinch:"Post-pinch disruption",
  none:""
}};

async function main() {{
  const canvas = document.getElementById("renderCanvas");
  const hud = document.getElementById("hud");

  let engine, gpu="WebGL2";
  try {{
    if(await BABYLON.WebGPUEngine.IsSupportedAsync){{
      engine=new BABYLON.WebGPUEngine(canvas,{{antialias:true}});
      await engine.initAsync(); gpu="WebGPU";
    }}
  }}catch(_){{}}
  if(!engine) engine=new BABYLON.Engine(canvas,true);

  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.025,0.025,0.05,1);

  // Orthographic camera looking at r-z plane (side view)
  const span = Math.max(D.cathode_radius*2.5, D.anode_length*1.2);
  const cam = new BABYLON.FreeCamera("cam", new BABYLON.Vector3(D.anode_length/2, 0, -span*1.5), scene);
  cam.setTarget(new BABYLON.Vector3(D.anode_length/2, 0, 0));
  cam.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
  const aspect = canvas.width / canvas.height;
  cam.orthoTop = span * 0.6;
  cam.orthoBottom = -span * 0.6;
  cam.orthoLeft = -span * aspect * 0.6;
  cam.orthoRight = span * aspect * 0.6;

  const hemi = new BABYLON.HemisphericLight("h", new BABYLON.Vector3(0,0,-1), scene);
  hemi.intensity = 0.3;

  // ---- Electrodes (2D cross-section = rectangles in x-y plane) ----
  // Anode: two bars (top + bottom) along z-axis (displayed as x)
  const a = D.anode_radius, b = D.cathode_radius, L = D.anode_length;

  function makeBar(name, x, y, w, h, color, alpha) {{
    const bar = BABYLON.MeshBuilder.CreatePlane(name, {{width:w, height:h}}, scene);
    bar.position.set(x, y, 0);
    const m = new BABYLON.StandardMaterial(name+"M", scene);
    m.diffuseColor = new BABYLON.Color3(...color);
    m.emissiveColor = new BABYLON.Color3(color[0]*0.3, color[1]*0.3, color[2]*0.3);
    m.alpha = alpha;
    m.backFaceCulling = false;
    bar.material = m;
    return bar;
  }}

  // Anode bars (copper, top + bottom)
  makeBar("anodeTop", L/2, a, L, a*0.15, [0.9,0.72,0.18], 0.9);
  makeBar("anodeBot", L/2, -a, L, a*0.15, [0.9,0.72,0.18], 0.9);
  // Cathode bars (steel, top + bottom)
  makeBar("cathTop", L/2, b, L, b*0.08, [0.5,0.5,0.55], 0.7);
  makeBar("cathBot", L/2, -b, L, b*0.08, [0.5,0.5,0.55], 0.7);
  // Insulator at z=0
  makeBar("ins", 0, 0, b*0.06, b*2.2, [0.7,0.5,0.85], 0.5);

  // ---- Swept plasma (trail behind sheath) ----
  const trailMat = new BABYLON.StandardMaterial("trailM", scene);
  trailMat.emissiveColor = new BABYLON.Color3(0.08,0.15,0.4);
  trailMat.alpha = 0.2;
  trailMat.disableLighting = true;
  trailMat.backFaceCulling = false;
  const trail = BABYLON.MeshBuilder.CreatePlane("trail", {{width:1, height:(b-a)*1.8}}, scene);
  trail.material = trailMat;

  // ---- Current sheath (bright vertical bar) ----
  const sMat = new BABYLON.StandardMaterial("sM", scene);
  sMat.emissiveColor = new BABYLON.Color3(0.2,0.5,1.0);
  sMat.alpha = 0.7;
  sMat.disableLighting = true;
  sMat.backFaceCulling = false;
  const sheathH = (b - a) * 2;
  const sheathBar = BABYLON.MeshBuilder.CreatePlane("sheath", {{width:b*0.12, height:sheathH}}, scene);
  sheathBar.material = sMat;

  // ---- Pinch region (glowing rectangle at anode tip) ----
  const pMat = new BABYLON.StandardMaterial("pM", scene);
  pMat.emissiveColor = new BABYLON.Color3(1,0.3,0.1);
  pMat.alpha = 0;
  pMat.disableLighting = true;
  pMat.backFaceCulling = false;
  const pinchRect = BABYLON.MeshBuilder.CreatePlane("pinch", {{width:L*0.35, height:a*0.5}}, scene);
  pinchRect.position.x = L*0.82;
  pinchRect.material = pMat;

  // Pinch halo (wider, dimmer)
  const phMat = new BABYLON.StandardMaterial("phM", scene);
  phMat.emissiveColor = new BABYLON.Color3(0.8,0.12,0.04);
  phMat.alpha = 0;
  phMat.disableLighting = true;
  phMat.backFaceCulling = false;
  const pinchHalo = BABYLON.MeshBuilder.CreatePlane("pinchH", {{width:L*0.4, height:a*1.2}}, scene);
  pinchHalo.position.x = L*0.82;
  pinchHalo.material = phMat;

  // ---- Glow + Bloom ----
  const gl = new BABYLON.GlowLayer("gl", scene, {{blurKernelSize:24, mainTextureFixedSize:256}});
  gl.intensity = 0.6;
  gl.customEmissiveColorSelector = (mesh,_s,_m,res) => {{
    const glow=["sheath","pinch","pinchH","trail"];
    if(glow.includes(mesh.name))
      res.set(mesh.material.emissiveColor.r,mesh.material.emissiveColor.g,mesh.material.emissiveColor.b,mesh.material.alpha);
    else res.set(0,0,0,0);
  }};

  const pp = new BABYLON.DefaultRenderingPipeline("pp",true,scene,[cam]);
  pp.bloomEnabled = true;
  pp.bloomThreshold = 0.5;
  pp.bloomWeight = 0.6;
  pp.bloomKernel = 48;

  // ---- GPU Particles (small sparks along sheath) ----
  const useGPU = BABYLON.GPUParticleSystem.IsSupported;
  const PS = useGPU ? BABYLON.GPUParticleSystem : BABYLON.ParticleSystem;
  const ps = new PS("sp", {{capacity: useGPU?8000:500}}, scene);
  ps.emitter = new BABYLON.Vector3(0,0,0);
  const em = new BABYLON.SphereParticleEmitter();
  em.radius = (b-a)*0.8;
  em.radiusRange = 0.5;
  ps.particleEmitterType = em;
  ps.minLifeTime=0.04; ps.maxLifeTime=0.12;
  ps.emitRate = useGPU?3000:200;
  ps.minSize=0.02; ps.maxSize=0.08;
  ps.minEmitPower=0.3; ps.maxEmitPower=1.5;
  ps.addColorGradient(0, new BABYLON.Color4(0.2,0.5,1,0));
  ps.addColorGradient(0.3, new BABYLON.Color4(0.5,0.8,1,0.8));
  ps.addColorGradient(1, new BABYLON.Color4(1,0.3,0.1,0));
  ps.isBillboardBased = true;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.particleTexture = new BABYLON.Texture(
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAP0lEQVQY02P4z8DwHwMDw38GBgYGJiCBDYMEMDAw/Gf4z/CfAQv/k4EFA3CAgQkHAAAAAElFTkSuQmCC",scene);
  ps.start();

  // ---- Animation ----
  let fi=0, playing=false, lastT=0;
  const slider=document.getElementById("tSlider");
  const tl=document.getElementById("tLabel");
  slider.max=D.n_frames-1;

  function apply(i) {{
    if(i<0||i>=D.frames.length)return;
    const f=D.frames[i];
    const col=PC[f.phase]||[0.3,0.3,0.4];
    const isPinch=["radial","mhd_radial","pinch","reflected","post_pinch"].includes(f.phase);

    // Sheath bar position + height
    sheathBar.position.x = isPinch ? L : f.z;
    const sr = isPinch ? Math.max(f.r, a*0.05) : b;
    sheathBar.scaling.y = sr / b;  // shrink height during compression
    sMat.emissiveColor.set(col[0],col[1],col[2]);
    sMat.alpha = 0.65 + Math.abs(f.I)*0.15;

    // Trail from insulator to sheath
    const tLen = Math.max(isPinch ? L : f.z, 0.3);
    trail.scaling.x = tLen;
    trail.position.x = tLen/2;
    trailMat.emissiveColor.set(col[0]*0.35, col[1]*0.35, col[2]*0.5);
    trailMat.alpha = 0.15 + Math.abs(f.I)*0.06;

    // Pinch intensity
    const cr = Math.max(0.02, f.r/b);
    const pI = isPinch ? Math.min(1, Math.pow(1-cr,2)*3) : 0;
    pMat.alpha = pI*0.8;
    phMat.alpha = pI*0.3;
    pMat.emissiveColor.set(1, pI*0.5, pI*0.3);
    pinchRect.scaling.y = Math.max(0.05, cr*1.2);
    pinchHalo.scaling.y = Math.max(0.08, cr*2.5);
    gl.intensity = 0.5 + pI*2;

    // Particles at sheath
    ps.emitter.x = sheathBar.position.x;
    ps.emitter.z = 0;

    hud.textContent = D.device+" | "+gpu+" | t="+f.t.toFixed(1)+" us | I="+f.I.toFixed(3)+" MA | "+(PL[f.phase]||f.phase);
  }}

  document.getElementById("playBtn").onclick=()=>{{playing=true}};
  document.getElementById("pauseBtn").onclick=()=>{{playing=false}};
  slider.oninput=()=>{{fi=+slider.value;apply(fi);tl.textContent="t="+D.frames[fi].t.toFixed(1)+" us"}};

  engine.runRenderLoop(()=>{{
    if(playing){{
      const now=performance.now();
      if(now-lastT>90){{
        fi=(fi+1)%D.n_frames;
        slider.value=fi;
        tl.textContent="t="+D.frames[fi].t.toFixed(1)+" us";
        apply(fi);
        lastT=now;
      }}
    }}
    scene.render();
  }});
  window.addEventListener("resize",()=>engine.resize());
  apply(0);
  hud.textContent=D.device+" | "+gpu+" | Ready";
}}
main().catch(e=>document.getElementById("hud").textContent="Error: "+e.message);
</script>
</body></html>"""


def create_cross_section_iframe(d: dict[str, Any], height: int = 450) -> str:
    html = create_cross_section_renderer(d)
    escaped = html_mod.escape(html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{height}px;border:none;background:#060610;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )
