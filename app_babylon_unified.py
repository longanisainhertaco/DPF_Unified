"""Unified Babylon.js WebGPU physics visualization — showcase quality.

Loads dpf_renderer.js as a standalone module. Python provides data,
JS handles all rendering at Babylon.js's maximum fidelity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app_visualization import extract_all_layers

BABYLON_CDN = "https://cdn.babylonjs.com/babylon.js"
BABYLON_GUI = "https://cdn.babylonjs.com/gui/babylon.gui.min.js"

_RENDERER_JS_PATH = Path(__file__).parent / "static" / "renderer" / "dpf_renderer.js"
_RENDERER_JS = _RENDERER_JS_PATH.read_text() if _RENDERER_JS_PATH.exists() else ""

# HTML template split into STATIC parts (no f-string needed for JS)
_HTML_HEAD = (
    '<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">\n<style>\n'
    "  html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#050508}\n"
    "  #c{width:100%;height:100%;touch-action:none;display:block}\n"
    "  #hud{position:absolute;top:6px;right:10px;color:#9cf;font:11px/1.6 monospace;"
    "pointer-events:none;z-index:10;text-shadow:0 0 5px #0008;text-align:right;white-space:pre;"
    "background:rgba(0,0,0,0.4);padding:6px 10px;border-radius:6px}\n"
    "  #bar{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:10;"
    "display:flex;gap:6px;align-items:center;background:rgba(0,0,0,0.65);padding:6px 14px;"
    "border-radius:8px;flex-wrap:wrap;max-width:95%}\n"
    "  #bar button{background:#111828;color:#9cf;border:1px solid #2a3a5a;padding:4px 12px;"
    "border-radius:4px;cursor:pointer;font:11px monospace;transition:background 0.2s}\n"
    "  #bar button:hover{background:#1a2848}\n"
    "  .sl{accent-color:#48f}\n"
    "  .lbl{color:#8af;font:10px monospace;min-width:50px}\n"
    "  #sl{width:200px}\n"
    "  #spd{width:60px;accent-color:#fa8}\n"
    "</style>\n"
    f'<script src="{BABYLON_CDN}"></script>\n'
    f'<script src="{BABYLON_GUI}"></script>\n'
    "</head>\n<body>\n"
    '<canvas id="c" tabindex="0"></canvas>\n'
    '<div id="hud">Initializing engine...</div>\n'
    '<div id="bar">\n'
    '  <button id="pb">Play</button>\n'
    '  <button id="sb">Pause</button>\n'
    '  <button id="stepB">Step</button>\n'
    '  <button id="rb">Reset</button>\n'
    '  <input type="range" id="sl" class="sl" min="0" max="1" step="1" value="0">\n'
    '  <span id="tl" class="lbl">t=0</span>\n'
    '  <span style="color:#444">|</span>\n'
    '  <span class="lbl" style="color:#fa8">Speed</span>\n'
    '  <input type="range" id="spd" class="sl" min="1" max="8" step="1" value="4">\n'
    '  <span id="spdL" class="lbl" style="color:#fa8">1x</span>\n'
    "</div>\n"
)

# Host code template — uses {data_json} placeholder (NOT f-string braces in JS)
_HTML_HOST = r"""
// ---- Data from simulation ----
const DATA = %%DATA_JSON%%;

// ---- Host: wire GUI + animation ----
// Wait for all CDN scripts to load before initializing
window.addEventListener("load", async function(){
  const canvas = document.getElementById("c");
  const hud = document.getElementById("hud");
  const sl = document.getElementById("sl");
  const tl = document.getElementById("tl");
  const spdSlider = document.getElementById("spd");
  const spdLabel = document.getElementById("spdL");

  var scene;
  try {
    scene = await createDPFScene(canvas, DATA);
    window._dpf = scene;  // expose for debugging
  } catch(e) {
    hud.textContent = "Engine error: " + e.message + "\n\nCheck browser console (F12) for details.";
    console.error("DPF Renderer Error:", e);
    return;
  }

  sl.max = scene.S.n_frames - 1;

  let fi = 0, playing = false, lastAdv = 0;
  let speedIdx = 4;
  spdSlider.value = speedIdx;
  spdLabel.textContent = SPEEDS[speedIdx] + "x";
  spdSlider.oninput = function() {
    speedIdx = +spdSlider.value;
    spdLabel.textContent = SPEEDS[speedIdx] + "x";
  };

  function frameMS() { return 160 / Math.max(SPEEDS[speedIdx], 0.01); }

  function renderFrame(i) {
    var result = scene.applyFrame(i);
    if (!result) return;
    var f = result.f, isP = result.isP, cr = result.cr, pI = result.pI, rippleAmp = result.rippleAmp;

    var info = scene.L.device + " | " + scene.gpuBackend +
      (scene.useGPU ? " (GPU particles)" : "") + "\n";
    info += "Phase: " + (PHASE_LABELS[f.phase] || f.phase) + "\n";
    info += "t = " + f.t.toFixed(2) + " us\n";
    info += "I = " + f.I.toFixed(3) + " MA  (" +
      (f.I / scene.S.I_peak * 100).toFixed(0) + "% of peak)\n";
    if (isP) {
      info += "r = " + f.r.toFixed(2) + " mm  (" +
        (scene.G.cathode_radius / Math.max(f.r, 0.01)).toFixed(0) + ":1 compression)\n";
    }
    if (scene.L.density) info += "rho_max = " + scene.L.density.max_val.toExponential(2) + " kg/m3\n";
    if (scene.L.temperature) info += "Te_max = " + scene.L.temperature.max_eV.toFixed(0) + " eV\n";
    if (scene.L.bfield) info += "|B|_max = " + scene.L.bfield.max_T.toFixed(1) + " T\n";
    if (pI > 0.1 && rippleAmp > 0) info += "m=0 instability: " + (rippleAmp * 100).toFixed(0) + "%\n";
    info += "\n" + (PHASE_DESCRIPTIONS[f.phase] || "");
    hud.textContent = info;
  }

  document.getElementById("pb").onclick = function() { playing = true; };
  document.getElementById("sb").onclick = function() { playing = false; };
  document.getElementById("stepB").onclick = function() {
    playing = false;
    fi = Math.min(fi + 1, scene.S.n_frames - 1);
    sl.value = fi;
    tl.textContent = "t=" + scene.S.frames[fi].t.toFixed(1) + " us";
    renderFrame(fi);
  };
  document.getElementById("rb").onclick = function() {
    fi = 0; sl.value = 0; playing = false;
    tl.textContent = "t=0";
    renderFrame(0);
  };
  sl.oninput = function() {
    fi = +sl.value;
    renderFrame(fi);
    tl.textContent = "t=" + scene.S.frames[fi].t.toFixed(1) + " us";
  };

  // ---- GUI Panel ----
  var ui = BABYLON.GUI.AdvancedDynamicTexture.CreateFullscreenUI("UI");
  var panel = new BABYLON.GUI.StackPanel();
  panel.width = "195px"; panel.isVertical = true;
  panel.horizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.verticalAlignment = BABYLON.GUI.Control.VERTICAL_ALIGNMENT_TOP;
  panel.paddingTop = "10px"; panel.paddingLeft = "10px";
  ui.addControl(panel);

  function addHeader(text, color) {
    var h = new BABYLON.GUI.TextBlock();
    h.text = text; h.color = color; h.fontSize = 12; h.height = "20px";
    h.textHorizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    panel.addControl(h);
  }
  function addToggle(label, initial, fn) {
    var cb = BABYLON.GUI.Checkbox.AddCheckBoxWithHeader(label, function(v) { fn(v); });
    cb.children[0].isChecked = initial; cb.children[0].color = "#7af";
    cb.children[1].color = "#cdf"; cb.children[1].fontSize = 11; cb.height = "24px";
    panel.addControl(cb); fn(initial);
  }

  addHeader("Scene", "#8af");
  addToggle("Electrodes", true, function(v) {
    scene.anode.isVisible = v;
    scene.cathodeRods.forEach(function(r) { r.isVisible = v; });
    scene.insulator.isVisible = v;
  });
  addToggle("Current Sheath", true, function(v) {
    scene.sheath.isVisible = v; scene.trail.isVisible = v;
  });
  addToggle("Plasma Ions", true, function(v) {
    if (v) scene.ps.start(); else scene.ps.stop();
  });
  addToggle("Pinch Column", true, function(v) {
    scene.pinch.isVisible = v; scene.halo.isVisible = v;
  });
  addToggle("B-Field Lines", !!scene.L.bfield, function(v) {
    scene.fieldLines.forEach(function(l) { l.isVisible = v; });
  });

  addHeader("Field Overlay", "#fa8");
  var overlayOpts = [
    ["None", "none"], ["Density", "density"], ["Temperature", "temperature"],
    ["|B| Magnetic", "bfield"], ["Radiation", "radiation"], ["Yield Map", "yield_map"]
  ];
  overlayOpts.forEach(function(pair) {
    addToggle(pair[0], pair[1] === "none", function(v) {
      if (v) scene.setOverlay(pair[1]);
    });
  });

  addHeader("Rendering", "#af8");
  addToggle("Cividis (colorblind)", false, function(v) { scene.setCmap(v); });
  addToggle("Bloom", true, function(v) { scene.pipeline.bloomEnabled = v; });
  if (scene.ssao) {
    addToggle("Ambient Occlusion", true, function(v) {
      scene.ssao.totalStrength = v ? 0.8 : 0;
    });
  }

  // ---- Render loop ----
  scene.engine.runRenderLoop(function() {
    try {
      if (playing) {
        var now = performance.now();
        if (now - lastAdv > frameMS()) {
          fi = (fi + 1) % scene.S.n_frames;
          sl.value = fi;
          tl.textContent = "t=" + scene.S.frames[fi].t.toFixed(1) + " us";
          renderFrame(fi);
          lastAdv = now;
        }
      }
      scene.scene.render();
    } catch(err) {
      hud.textContent = "RENDER ERROR at frame " + fi + ":\n" + err.message + "\n" + err.stack.substring(0, 200);
      console.error("Render loop error:", err);
    }
  });

  window.addEventListener("resize", function() { scene.engine.resize(); });
  renderFrame(0);
  hud.textContent = scene.L.device + " | " + scene.gpuBackend + " | Ready — drag to orbit, scroll to zoom";
});
"""


def create_unified_renderer(d: dict[str, Any]) -> str:
    layers = extract_all_layers(d)
    data_json = json.dumps(layers)

    # Assemble HTML without f-string touching the JS code
    host_code = _HTML_HOST.replace("%%DATA_JSON%%", data_json)
    return (
        _HTML_HEAD
        + "<script>\n"
        + "// ---- Renderer module ----\n"
        + _RENDERER_JS + "\n\n"
        + host_code + "\n"
        + "</script>\n</body></html>"
    )


def create_unified_iframe(d: dict[str, Any], height: int = 620) -> str:
    html = create_unified_renderer(d)
    escaped = html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{height}px;border:none;background:#050508;" '
        f'allow="accelerometer; camera; gyroscope; xr-spatial-tracking"></iframe>'
    )
