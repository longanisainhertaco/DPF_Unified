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

_HTML_HEAD = (
    '<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">\n<style>\n'
    "  html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#050508}\n"
    "  #c{width:100%;height:100%;touch-action:none;display:block}\n"
    # Phase banner — large centered text at top
    "  #phase-banner{position:absolute;top:0;left:0;right:0;z-index:12;"
    "text-align:center;pointer-events:none;padding:14px 0 10px;"
    "background:linear-gradient(180deg,rgba(0,0,0,0.7) 0%,rgba(0,0,0,0) 100%)}\n"
    "  #phase-name{color:#fff;font:bold 22px/1 'Helvetica Neue',Arial,sans-serif;"
    "text-shadow:0 0 12px rgba(100,180,255,0.5);letter-spacing:1px}\n"
    "  #phase-desc{color:#8cf;font:14px/1.4 'Helvetica Neue',Arial,sans-serif;"
    "margin-top:4px;opacity:0.9}\n"
    # Data HUD — right side
    "  #hud{position:absolute;top:70px;right:12px;color:#bdf;font:13px/1.7 monospace;"
    "pointer-events:none;z-index:10;text-shadow:0 0 4px #000a;text-align:right;white-space:pre;"
    "background:rgba(0,0,0,0.45);padding:10px 14px;border-radius:8px;"
    "border:1px solid rgba(100,160,255,0.15)}\n"
    # Backend badge
    "  #badge{position:absolute;top:70px;left:12px;z-index:10;pointer-events:none;"
    "background:rgba(20,30,60,0.8);border:1px solid rgba(100,160,255,0.3);"
    "border-radius:6px;padding:6px 12px;font:12px/1.5 monospace;color:#8af}\n"
    # Transport bar
    "  #bar{position:absolute;bottom:0;left:0;right:0;z-index:10;"
    "display:flex;gap:10px;align-items:center;justify-content:center;"
    "background:linear-gradient(0deg,rgba(0,0,0,0.85) 0%,rgba(0,0,0,0.5) 80%,rgba(0,0,0,0) 100%);"
    "padding:16px 20px 14px;flex-wrap:nowrap;white-space:nowrap}\n"
    "  #bar button{background:rgba(20,30,60,0.9);color:#bdf;border:1px solid rgba(100,160,255,0.3);"
    "padding:10px 20px;border-radius:6px;cursor:pointer;font:bold 15px 'Helvetica Neue',Arial,sans-serif;"
    "transition:all 0.15s;text-transform:uppercase;letter-spacing:0.5px}\n"
    "  #bar button:hover{background:rgba(40,60,120,0.9);border-color:rgba(100,180,255,0.6);"
    "box-shadow:0 0 8px rgba(80,140,255,0.3)}\n"
    "  #bar button:active{transform:scale(0.96)}\n"
    "  .sl{accent-color:#48f;height:8px}\n"
    "  .lbl{color:#8af;font:14px 'Helvetica Neue',Arial,sans-serif}\n"
    "  #sl{width:220px}\n"
    "  #spd{width:70px;accent-color:#fa8}\n"
    # Timeline phases bar
    "  #timeline{position:absolute;bottom:62px;left:20px;right:20px;height:4px;"
    "z-index:11;border-radius:2px;overflow:hidden;pointer-events:none;"
    "background:rgba(255,255,255,0.08)}\n"
    "  #tl-progress{height:100%;border-radius:2px;transition:width 0.1s}\n"
    "</style>\n"
    f'<script src="{BABYLON_CDN}"></script>\n'
    f'<script src="{BABYLON_GUI}"></script>\n'
    "</head>\n<body>\n"
    '<canvas id="c" tabindex="0"></canvas>\n'
    '<div id="phase-banner"><div id="phase-name">Initializing...</div>'
    '<div id="phase-desc"></div></div>\n'
    '<div id="hud"></div>\n'
    '<div id="badge"></div>\n'
    '<div id="timeline"><div id="tl-progress"></div></div>\n'
    '<div id="bar">\n'
    '  <button id="pb">&#9654; Play</button>\n'
    '  <button id="sb">&#10074;&#10074; Pause</button>\n'
    '  <button id="stepB">&#9654;| Step</button>\n'
    '  <button id="rb">&#8634; Reset</button>\n'
    '  <input type="range" id="sl" class="sl" min="0" max="1" step="1" value="0">\n'
    '  <span id="tl" class="lbl">t = 0 us</span>\n'
    '  <span style="color:rgba(255,255,255,0.15);font-size:18px">|</span>\n'
    '  <span class="lbl" style="color:#fa8">Speed</span>\n'
    '  <input type="range" id="spd" class="sl" min="1" max="8" step="1" value="4">\n'
    '  <span id="spdL" class="lbl" style="color:#fa8">1x</span>\n'
    "</div>\n"
)

_HTML_HOST = r"""
const DATA = %%DATA_JSON%%;

// Phase color map for timeline bar
var PHASE_BAR_COLORS = {
  rundown: "#2070ff",
  radial: "#ff6020",
  mhd_radial: "#ff6020",
  reflected: "#ff9000",
  pinch: "#ff2008",
  post_pinch: "#b03020",
  none: "#334"
};

window.addEventListener("load", async function(){
  var canvas = document.getElementById("c");
  var hudEl = document.getElementById("hud");
  var phaseName = document.getElementById("phase-name");
  var phaseDesc = document.getElementById("phase-desc");
  var badgeEl = document.getElementById("badge");
  var tlProgress = document.getElementById("tl-progress");
  var sl = document.getElementById("sl");
  var tl = document.getElementById("tl");
  var spdSlider = document.getElementById("spd");
  var spdLabel = document.getElementById("spdL");

  var scene;
  try {
    scene = await createDPFScene(canvas, DATA);
    window._dpf = scene;
  } catch(e) {
    phaseName.textContent = "Engine Error";
    phaseDesc.textContent = e.message;
    console.error("DPF Renderer Error:", e);
    return;
  }

  sl.max = scene.S.n_frames - 1;

  // Badge: device + backend + engine info
  var layers = [];
  if (scene.L.density) layers.push("density");
  if (scene.L.temperature) layers.push("temperature");
  if (scene.L.bfield) layers.push("B-field");
  if (scene.L.instability) layers.push("instability");
  if (scene.L.radiation) layers.push("radiation");
  if (scene.L.yield_map) layers.push("yield");
  badgeEl.innerHTML = "<b>" + scene.L.device + "</b><br>" +
    "Backend: " + (scene.L.backend || "lee") + " | " + scene.gpuBackend +
    (scene.useGPU ? " (GPU)" : "") + "<br>" +
    "Physics: " + (layers.length > 0 ? layers.join(", ") : "circuit only");

  var fi = 0, playing = false, lastAdv = 0;
  var speedIdx = 4;
  spdSlider.value = speedIdx;
  spdLabel.textContent = SPEEDS[speedIdx] + "x";
  spdSlider.oninput = function() {
    speedIdx = +spdSlider.value;
    spdLabel.textContent = SPEEDS[speedIdx] + "x";
  };

  function frameMS() { return 160 / Math.max(SPEEDS[speedIdx], 0.01); }

  var lastPhase = "";

  function renderFrame(i) {
    var result = scene.applyFrame(i);
    if (!result) return;
    var f = result.f, isP = result.isP, cr = result.cr, pI = result.pI, rippleAmp = result.rippleAmp;

    // Phase banner (update only on change for smooth feel)
    if (f.phase !== lastPhase) {
      phaseName.textContent = (PHASE_LABELS[f.phase] || f.phase).toUpperCase();
      phaseDesc.textContent = PHASE_DESCRIPTIONS[f.phase] || "";
      phaseName.style.color = PHASE_BAR_COLORS[f.phase] ? "#fff" : "#888";
      lastPhase = f.phase;
    }

    // Timeline progress bar
    var pct = (i / Math.max(scene.S.n_frames - 1, 1)) * 100;
    tlProgress.style.width = pct + "%";
    tlProgress.style.background = PHASE_BAR_COLORS[f.phase] || "#48f";

    // Data HUD — structured physics readout
    var lines = [];
    lines.push("t = " + f.t.toFixed(2) + " us");
    lines.push("I = " + f.I.toFixed(3) + " MA  (" +
      (f.I / scene.S.I_peak * 100).toFixed(0) + "%)");
    if (isP) {
      var ratio = (scene.G.cathode_radius / Math.max(f.r, 0.01)).toFixed(0);
      lines.push("r = " + f.r.toFixed(1) + " mm  (" + ratio + ":1)");
    }
    if (scene.L.density) lines.push("rho = " + scene.L.density.max_val.toExponential(1) + " kg/m3");
    if (scene.L.temperature && scene.L.temperature.max_eV > 1)
      lines.push("Te = " + scene.L.temperature.max_eV.toFixed(0) + " eV");
    if (scene.L.bfield) lines.push("|B| = " + scene.L.bfield.max_T.toFixed(1) + " T");
    if (pI > 0.1 && rippleAmp > 0) lines.push("m=0: " + (rippleAmp * 100).toFixed(0) + "%");
    hudEl.textContent = lines.join("\n");
  }

  document.getElementById("pb").onclick = function() { playing = true; };
  document.getElementById("sb").onclick = function() { playing = false; };
  document.getElementById("stepB").onclick = function() {
    playing = false;
    fi = Math.min(fi + 1, scene.S.n_frames - 1);
    sl.value = fi;
    tl.textContent = "t = " + scene.S.frames[fi].t.toFixed(1) + " us";
    renderFrame(fi);
  };
  document.getElementById("rb").onclick = function() {
    fi = 0; sl.value = 0; playing = false;
    tl.textContent = "t = 0 us";
    renderFrame(0);
  };
  sl.oninput = function() {
    fi = +sl.value;
    renderFrame(fi);
    tl.textContent = "t = " + scene.S.frames[fi].t.toFixed(1) + " us";
  };

  // ---- GUI Panel (layer toggles with dark background) ----
  var ui = BABYLON.GUI.AdvancedDynamicTexture.CreateFullscreenUI("UI");

  var panelBg = new BABYLON.GUI.Rectangle("panelBg");
  panelBg.width = "240px"; panelBg.adaptHeightToChildren = true;
  panelBg.horizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panelBg.verticalAlignment = BABYLON.GUI.Control.VERTICAL_ALIGNMENT_TOP;
  panelBg.left = "8px"; panelBg.top = "68px";
  panelBg.background = "rgba(5,8,20,0.75)";
  panelBg.color = "rgba(80,120,200,0.2)";
  panelBg.thickness = 1;
  panelBg.cornerRadius = 8;
  panelBg.paddingBottom = "8px";
  ui.addControl(panelBg);

  var panel = new BABYLON.GUI.StackPanel();
  panel.width = "220px"; panel.isVertical = true;
  panel.paddingTop = "8px"; panel.paddingLeft = "8px";
  panelBg.addControl(panel);

  function addHeader(text, color) {
    var h = new BABYLON.GUI.TextBlock();
    h.text = text; h.color = color; h.fontSize = 14; h.fontWeight = "bold";
    h.height = "26px";
    h.textHorizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    panel.addControl(h);
  }
  function addToggle(label, initial, fn, enabled) {
    if (enabled === false) return;
    var cb = BABYLON.GUI.Checkbox.AddCheckBoxWithHeader(label, function(v) { fn(v); });
    cb.children[0].isChecked = initial; cb.children[0].color = "#7af";
    cb.children[0].width = "22px"; cb.children[0].height = "22px";
    cb.children[1].color = "#cdf"; cb.children[1].fontSize = 13; cb.height = "30px";
    panel.addControl(cb); fn(initial);
  }

  addHeader("Layers", "#8af");
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

  // Only show overlay options if MHD data exists
  if (scene.L.density || scene.L.temperature || scene.L.bfield) {
    addHeader("Field Overlay", "#fa8");
    addToggle("None", true, function(v) { if (v) scene.setOverlay("none"); });
    addToggle("Density", false, function(v) { if (v) scene.setOverlay("density"); }, !!scene.L.density);
    addToggle("Temperature", false, function(v) { if (v) scene.setOverlay("temperature"); }, !!scene.L.temperature);
    addToggle("|B| Magnetic", false, function(v) { if (v) scene.setOverlay("bfield"); }, !!scene.L.bfield);
    addToggle("Radiation", false, function(v) { if (v) scene.setOverlay("radiation"); }, !!scene.L.radiation);
    addToggle("Yield Map", false, function(v) { if (v) scene.setOverlay("yield_map"); }, !!scene.L.yield_map);
  }

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
          tl.textContent = "t = " + scene.S.frames[fi].t.toFixed(1) + " us";
          renderFrame(fi);
          lastAdv = now;
        }
      }
      scene.scene.render();
    } catch(err) {
      phaseName.textContent = "RENDER ERROR";
      phaseDesc.textContent = err.message;
      console.error("Render loop error:", err);
    }
  });

  window.addEventListener("resize", function() { scene.engine.resize(); });
  renderFrame(0);
  phaseName.textContent = "READY";
  phaseDesc.textContent = "Drag to orbit \u2022 Scroll to zoom \u2022 Press Play to animate";
});
"""


def create_unified_renderer(d: dict[str, Any]) -> str:
    layers = extract_all_layers(d)
    data_json = json.dumps(layers)

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
