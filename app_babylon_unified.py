"""Unified Babylon.js WebGPU physics visualization — showcase quality.

Loads dpf_renderer.js as a standalone module. Python provides data,
JS handles all rendering at Babylon.js's maximum fidelity.
"""
from __future__ import annotations

import html as html_mod
import json
from pathlib import Path
from typing import Any

from app_visualization import extract_all_layers

BABYLON_CDN = "https://cdn.babylonjs.com/babylon.js"
BABYLON_GUI = "https://cdn.babylonjs.com/gui/babylon.gui.min.js"

# Read the renderer JS at import time (it's static)
_RENDERER_JS_PATH = Path(__file__).parent / "static" / "renderer" / "dpf_renderer.js"
_RENDERER_JS = _RENDERER_JS_PATH.read_text() if _RENDERER_JS_PATH.exists() else ""


def create_unified_renderer(d: dict[str, Any]) -> str:
    layers = extract_all_layers(d)
    data_json = json.dumps(layers)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#050508}}
  #c{{width:100%;height:100%;touch-action:none;display:block}}
  #hud{{position:absolute;top:6px;right:10px;color:#9cf;font:11px/1.6 monospace;
    pointer-events:none;z-index:10;text-shadow:0 0 5px #0008;text-align:right;white-space:pre;
    background:rgba(0,0,0,0.4);padding:6px 10px;border-radius:6px}}
  #bar{{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:10;
    display:flex;gap:6px;align-items:center;background:rgba(0,0,0,0.65);padding:6px 14px;border-radius:8px;flex-wrap:wrap;max-width:95%}}
  #bar button{{background:#111828;color:#9cf;border:1px solid #2a3a5a;padding:4px 12px;border-radius:4px;cursor:pointer;font:11px monospace;transition:background 0.2s}}
  #bar button:hover{{background:#1a2848}}
  #bar button.active{{background:#2a4a8a;border-color:#4a7aff}}
  .sl{{accent-color:#48f}}
  .lbl{{color:#8af;font:10px monospace;min-width:50px}}
  #sl{{width:200px}}
  #spd{{width:60px;accent-color:#fa8}}
</style>
<script src="{BABYLON_CDN}"></script>
<script src="{BABYLON_GUI}"></script>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">Initializing engine...</div>
<div id="bar">
  <button id="pb">Play</button>
  <button id="sb">Pause</button>
  <button id="stepB">Step</button>
  <button id="rb">Reset</button>
  <input type="range" id="sl" class="sl" min="0" max="1" step="1" value="0">
  <span id="tl" class="lbl">t=0</span>
  <span style="color:#444">|</span>
  <span class="lbl" style="color:#fa8">Speed</span>
  <input type="range" id="spd" class="sl" min="1" max="8" step="1" value="4">
  <span id="spdL" class="lbl" style="color:#fa8">1x</span>
</div>

<script>
// ---- Renderer module (inlined from dpf_renderer.js) ----
{_RENDERER_JS}

// ---- Data from simulation ----
const DATA = {data_json};

// ---- Host: wire GUI + animation ----
(async function(){{
  const canvas = document.getElementById("c");
  const hud = document.getElementById("hud");
  const sl = document.getElementById("sl");
  const tl = document.getElementById("tl");
  const spdSlider = document.getElementById("spd");
  const spdLabel = document.getElementById("spdL");

  let scene;
  try {{
    scene = await createDPFScene(canvas, DATA);
  }} catch(e) {{
    hud.textContent = "Engine error: " + e.message;
    console.error(e);
    return;
  }}

  sl.max = scene.S.n_frames - 1;

  // ---- Animation state ----
  let fi = 0, playing = false, lastAdv = 0;
  let speedIdx = 4;  // 1x default
  spdSlider.value = speedIdx;
  spdLabel.textContent = SPEEDS[speedIdx] + "x";
  spdSlider.oninput = () => {{
    speedIdx = +spdSlider.value;
    spdLabel.textContent = SPEEDS[speedIdx] + "x";
  }};

  function frameMS() {{ return 160 / Math.max(SPEEDS[speedIdx], 0.01); }}

  function renderFrame(i) {{
    const result = scene.applyFrame(i);
    if (!result) return;
    const {{ f, isP, cr, pI, rippleAmp }} = result;

    // HUD data readout
    let info = scene.L.device + " | " + scene.gpuBackend +
      (scene.useGPU ? " (GPU particles)" : "") + "\\n";
    info += "Phase: " + (PHASE_LABELS[f.phase] || f.phase) + "\\n";
    info += "t = " + f.t.toFixed(2) + " us\\n";
    info += "I = " + f.I.toFixed(3) + " MA  (" +
      (f.I / scene.S.I_peak * 100).toFixed(0) + "% of peak)\\n";
    if (isP) {{
      info += "r = " + f.r.toFixed(2) + " mm  (" +
        (scene.G.cathode_radius / Math.max(f.r, 0.01)).toFixed(0) + ":1 compression)\\n";
    }}
    if (scene.L.density) info += "rho_max = " + scene.L.density.max_val.toExponential(2) + " kg/m3\\n";
    if (scene.L.temperature) info += "Te_max = " + scene.L.temperature.max_eV.toFixed(0) + " eV\\n";
    if (scene.L.bfield) info += "|B|_max = " + scene.L.bfield.max_T.toFixed(1) + " T\\n";
    if (pI > 0.1 && rippleAmp > 0) info += "m=0 instability: " + (rippleAmp * 100).toFixed(0) + "%\\n";
    info += "\\n" + (PHASE_DESCRIPTIONS[f.phase] || "");
    hud.textContent = info;
  }}

  // ---- Controls ----
  document.getElementById("pb").onclick = () => {{ playing = true; }};
  document.getElementById("sb").onclick = () => {{ playing = false; }};
  document.getElementById("stepB").onclick = () => {{
    playing = false;
    fi = Math.min(fi + 1, scene.S.n_frames - 1);
    sl.value = fi;
    tl.textContent = "t=" + scene.S.frames[fi].t.toFixed(1) + " us";
    renderFrame(fi);
  }};
  document.getElementById("rb").onclick = () => {{
    fi = 0; sl.value = 0; playing = false;
    tl.textContent = "t=0";
    renderFrame(0);
  }};
  sl.oninput = () => {{
    fi = +sl.value;
    renderFrame(fi);
    tl.textContent = "t=" + scene.S.frames[fi].t.toFixed(1) + " us";
  }};

  // ---- GUI Panel (Babylon.GUI inside scene) ----
  const ui = BABYLON.GUI.AdvancedDynamicTexture.CreateFullscreenUI("UI");
  const panel = new BABYLON.GUI.StackPanel();
  panel.width = "195px"; panel.isVertical = true;
  panel.horizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.verticalAlignment = BABYLON.GUI.Control.VERTICAL_ALIGNMENT_TOP;
  panel.paddingTop = "10px"; panel.paddingLeft = "10px";
  ui.addControl(panel);

  function addHeader(text, color) {{
    const h = new BABYLON.GUI.TextBlock();
    h.text = text; h.color = color; h.fontSize = 12; h.height = "20px";
    h.textHorizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    panel.addControl(h);
  }}

  function addToggle(label, initial, fn) {{
    const cb = BABYLON.GUI.Checkbox.AddCheckBoxWithHeader(label, (v) => fn(v));
    cb.children[0].isChecked = initial; cb.children[0].color = "#7af";
    cb.children[1].color = "#cdf"; cb.children[1].fontSize = 11; cb.height = "24px";
    panel.addControl(cb); fn(initial);
  }}

  addHeader("Scene", "#8af");
  addToggle("Electrodes", true, v => {{
    scene.anode.isVisible = v;
    scene.cathodeRods.forEach(r => r.isVisible = v);
    scene.insulator.isVisible = v;
  }});
  addToggle("Current Sheath", true, v => {{
    scene.sheath.isVisible = v; scene.trail.isVisible = v;
  }});
  addToggle("Plasma Ions", true, v => {{ if (v) scene.ps.start(); else scene.ps.stop(); }});
  addToggle("Pinch Column", true, v => {{
    scene.pinch.isVisible = v; scene.halo.isVisible = v;
  }});
  addToggle("B-Field Lines", !!scene.L.bfield, v => {{
    scene.fieldLines.forEach(l => l.isVisible = v);
  }});

  addHeader("Field Overlay", "#fa8");
  const overlayOpts = [
    ["None", "none"], ["Density", "density"], ["Temperature", "temperature"],
    ["|B| Magnetic", "bfield"], ["Radiation", "radiation"], ["Yield Map", "yield_map"],
  ];
  overlayOpts.forEach(([label, key]) => {{
    addToggle(label, key === "none", v => {{
      if (v) {{ scene.setOverlay(key); showColorbar(key); }}
    }});
  }});

  addHeader("Accessibility", "#af8");
  addToggle("Cividis (colorblind)", false, v => {{
    scene.setCmap(v);
    if (scene.activeOverlay !== "none") showColorbar(scene.activeOverlay);
  }});
  addToggle("Bloom", true, v => {{ scene.pipeline.bloomEnabled = v; }});
  addToggle("Ambient Occlusion", !!scene.ssao, v => {{
    if (scene.ssao) scene.ssao.totalStrength = v ? 0.8 : 0;
  }});

  // ---- Colorbar (right side) ----
  const cbPanel = new BABYLON.GUI.StackPanel();
  cbPanel.width = "65px"; cbPanel.isVertical = true;
  cbPanel.horizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_RIGHT;
  cbPanel.verticalAlignment = BABYLON.GUI.Control.VERTICAL_ALIGNMENT_CENTER;
  cbPanel.paddingRight = "8px";
  ui.addControl(cbPanel); cbPanel.isVisible = false;

  const cbTitle = new BABYLON.GUI.TextBlock(); cbTitle.color = "#ddd"; cbTitle.fontSize = 11;
  cbTitle.height = "18px"; cbPanel.addControl(cbTitle);
  const cbMax = new BABYLON.GUI.TextBlock(); cbMax.color = "#eee"; cbMax.fontSize = 10;
  cbMax.height = "16px"; cbPanel.addControl(cbMax);

  const N_CB = 20; const cbRects = [];
  for (let i = N_CB - 1; i >= 0; i--) {{
    const rect = new BABYLON.GUI.Rectangle();
    rect.width = "35px"; rect.height = "10px"; rect.thickness = 0;
    cbPanel.addControl(rect); cbRects.push(rect);
  }}
  const cbMin = new BABYLON.GUI.TextBlock(); cbMin.color = "#eee"; cbMin.fontSize = 10;
  cbMin.height = "16px"; cbPanel.addControl(cbMin);
  const cbUnits = new BABYLON.GUI.TextBlock(); cbUnits.color = "#aaa"; cbUnits.fontSize = 9;
  cbUnits.height = "14px"; cbPanel.addControl(cbUnits);

  function showColorbar(key) {{
    if (key === "none") {{ cbPanel.isVisible = false; return; }}
    cbPanel.isVisible = true;
    for (let i = 0; i < N_CB; i++) {{
      const t = (N_CB - 1 - i) / (N_CB - 1);
      const [r, g, b] = window.cmap ? window.cmap(t) : [t, t, t];
      // Use the module's cmap through the scene
      const c = scene.L._cmapFn ? scene.L._cmapFn(t) : [t, t, t];
      cbRects[i].background = `rgb(${{Math.round(r*255)}},${{Math.round(g*255)}},${{Math.round(b*255)}})`;
    }}
    // Try to get cmap colors from the renderer
    try {{
      for (let i = 0; i < N_CB; i++) {{
        const t = (N_CB - 1 - i) / (N_CB - 1);
        // Access cmap through global scope (it's in dpf_renderer.js)
        const [r, g, b] = cmap(t);
        cbRects[i].background = `rgb(${{Math.round(r*255)}},${{Math.round(g*255)}},${{Math.round(b*255)}})`;
      }}
    }} catch(_) {{}}

    const info = {{
      density: {{ title: "Density", max: scene.L.density?.max_val?.toExponential(1), min: scene.L.density?.min_val?.toExponential(1), unit: "kg/m3" }},
      temperature: {{ title: "Te", max: scene.L.temperature?.max_eV?.toFixed(0), min: scene.L.temperature?.min_eV?.toFixed(1), unit: "eV" }},
      bfield: {{ title: "|B|", max: scene.L.bfield?.max_T?.toFixed(1), min: "0", unit: "Tesla" }},
      radiation: {{ title: "P_rad", max: scene.L.radiation?.max_W_m3?.toExponential(1), min: "0", unit: "W/m3" }},
      yield_map: {{ title: "Yield", max: scene.L.yield_map?.max_rate?.toExponential(1), min: "0", unit: "n/m3/s" }},
    }};
    const d = info[key] || {{ title: key, max: "1", min: "0", unit: "" }};
    cbTitle.text = d.title || ""; cbMax.text = d.max || "?"; cbMin.text = d.min || "0"; cbUnits.text = d.unit || "";
  }}

  // ---- Render loop ----
  scene.engine.runRenderLoop(() => {{
    if (playing) {{
      const now = performance.now();
      if (now - lastAdv > frameMS()) {{
        fi = (fi + 1) % scene.S.n_frames;
        sl.value = fi;
        tl.textContent = "t=" + scene.S.frames[fi].t.toFixed(1) + " us";
        renderFrame(fi);
        lastAdv = now;
      }}
    }}
    scene.scene.render();
  }});

  window.addEventListener("resize", () => scene.engine.resize());
  renderFrame(0);
  hud.textContent = scene.L.device + " | " + scene.gpuBackend + " | Ready — drag to orbit, scroll to zoom";
}})();
</script>
</body></html>"""


def create_unified_iframe(d: dict[str, Any], height: int = 620) -> str:
    html = create_unified_renderer(d)
    escaped = html_mod.escape(html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{height}px;border:none;background:#050508;" '
        f'allow="accelerometer; camera; gyroscope; xr-spatial-tracking" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )
