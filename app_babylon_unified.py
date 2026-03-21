"""Unified Babylon.js WebGPU physics visualization -- showcase quality.

DATA PIPELINE (Python -> JSON -> Babylon.js):

1. Simulation runs in Python (Lee model + optional MHD backend).
2. app_engine.run_simulation_core() returns a result dict with:
   - Circuit scalars: t_us[], z_mm[], r_mm[], I_MA[], phases[]
   - MHD final_state: {rho, Te, B, velocity, pressure} as NumPy arrays
   - MHD snapshots: [{t_us, rho_mid, B_mid, P_mid}, ...] for animation
   - Derived: beam_tracker, instability, pinch metrics
3. app_visualization.extract_all_layers() transforms this into 10 layers:
   L1 geometry, L2 sheath timeline, L3 density, L4 temperature, L5 bfield,
   L6 pinch, L7 beam, L8 instability, L9 radiation, L10 yield_map.
   MHD 2D arrays are base64-encoded Float32 with [nr, nz] shape.
   MHD snapshots are globally normalized per-field for consistent coloring.
4. This module (app_babylon_unified.py) serializes layers to JSON, embeds
   them as `const DATA = {...}` inside an HTML page alongside dpf_renderer.js.
5. dpf_renderer.js (createDPFScene) builds the Babylon.js scene:
   - Decodes base64 arrays into Float32Arrays
   - Pre-builds RGBA snap caches for animated heatmap textures
   - Creates 3D meshes (electrodes, sheath torus, pinch tube, particles)
   - applyFrame(i) drives animation from Lee scalars + snap textures
6. The HTML page is served via Gradio gr.HTML as a sandboxed iframe.

Loads dpf_renderer.js as a standalone module. Python provides data,
JS handles all rendering at Babylon.js's maximum fidelity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app_visualization import extract_all_layers

BABYLON_CDN = "https://cdn.babylonjs.com/babylon.js"
BABYLON_MAT = "https://cdn.babylonjs.com/materialsLibrary/babylonjs.materials.min.js"
BABYLON_LOADERS = "https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"
# GUI library removed — overlays are HTML/CSS, not Babylon.GUI

_RENDERER_JS_PATH = Path(__file__).parent / "static" / "renderer" / "dpf_renderer.js"
_RENDERER_JS = _RENDERER_JS_PATH.read_text() if _RENDERER_JS_PATH.exists() else ""
_VOLUMETRIC_JS_PATH = Path(__file__).parent / "static" / "renderer" / "dpf_volumetric.js"
_VOLUMETRIC_JS = _VOLUMETRIC_JS_PATH.read_text() if _VOLUMETRIC_JS_PATH.exists() else ""

_HTML_HEAD = (
    '<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">\n<style>\n'
    "  html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#1e2025}\n"
    "  #c{width:100%;height:100%;touch-action:none;display:block}\n"
    # Phase banner — large centered text at top
    "  #phase-banner{position:absolute;top:0;left:0;right:0;z-index:12;"
    "text-align:center;pointer-events:none;padding:14px 0 10px;"
    "background:linear-gradient(180deg,rgba(0,0,0,0.6) 0%,rgba(0,0,0,0) 100%)}\n"
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
    # Visualization mode banner — persistent label for data source honesty
    "  #vis-mode{position:absolute;top:70px;left:50%;transform:translateX(-50%);"
    "z-index:11;pointer-events:none;text-align:center;"
    "padding:5px 16px;border-radius:6px;font:bold 11px/1.5 'Helvetica Neue',Arial,sans-serif;"
    "letter-spacing:0.5px;text-transform:uppercase}\n"
    "  #vis-mode.lee{background:rgba(180,120,30,0.85);color:#fff;"
    "border:1px solid rgba(255,180,50,0.5)}\n"
    "  #vis-mode.mhd{background:rgba(30,120,80,0.85);color:#fff;"
    "border:1px solid rgba(50,200,120,0.5)}\n"
    # Layer toggles panel (HTML, not Babylon.GUI — scales on Retina)
    "  #layers{position:absolute;bottom:70px;left:10px;z-index:12;"
    "background:rgba(5,8,20,0.85);border:1px solid rgba(80,120,200,0.2);"
    "border-radius:8px;padding:10px 14px;font:14px/1.8 'Helvetica Neue',Arial,sans-serif;"
    "color:#cdf;max-height:50vh;overflow-y:auto}\n"
    "  #layers label{display:flex;align-items:center;gap:8px;cursor:pointer;"
    "padding:2px 0;transition:color 0.15s}\n"
    "  #layers label:hover{color:#fff}\n"
    "  #layers input[type=checkbox]{width:18px;height:18px;accent-color:#48f;cursor:pointer}\n"
    "  #layers .hdr{color:#8af;font-weight:bold;font-size:13px;margin-top:6px;"
    "margin-bottom:2px;letter-spacing:0.5px}\n"
    # Physics info panel — dynamic explanation of what the user is seeing
    "  #info-panel{position:absolute;bottom:70px;right:10px;z-index:12;"
    "width:260px;background:rgba(5,8,20,0.88);border:1px solid rgba(80,120,200,0.2);"
    "border-radius:8px;padding:12px 14px;font:13px/1.6 'Helvetica Neue',Arial,sans-serif;"
    "color:#cdf;max-height:40vh;overflow-y:auto}\n"
    "  #info-panel .info-title{color:#8cf;font-weight:bold;font-size:14px;"
    "margin-bottom:6px;letter-spacing:0.5px}\n"
    "  #info-panel .info-body{color:#bcd;font-size:12px;line-height:1.5;margin-bottom:8px}\n"
    "  #info-panel .info-phase{color:#fa8;font-size:11px;line-height:1.4;"
    "border-top:1px solid rgba(255,180,80,0.2);padding-top:6px;margin-top:4px}\n"
    # Colorbar — vertical scale indicator for heatmap overlays
    "  #colorbar{position:absolute;right:12px;top:50%;transform:translateY(-50%);z-index:12;"
    "pointer-events:none;display:none;width:28px;height:200px;border-radius:4px;"
    "border:1px solid rgba(100,160,255,0.3);overflow:hidden}\n"
    "  #cb-gradient{width:100%;height:100%}\n"
    "  #cb-max{position:absolute;top:-18px;right:0;color:#cdf;font:bold 11px monospace;"
    "text-shadow:0 0 4px #000;white-space:nowrap}\n"
    "  #cb-min{position:absolute;bottom:-18px;right:0;color:#cdf;font:bold 11px monospace;"
    "text-shadow:0 0 4px #000;white-space:nowrap}\n"
    "  #cb-label{position:absolute;top:50%;right:34px;transform:translateY(-50%) rotate(-90deg);"
    "color:#8af;font:bold 11px monospace;text-shadow:0 0 4px #000;white-space:nowrap;"
    "transform-origin:center center}\n"
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
    "  #timeline{position:absolute;bottom:62px;left:20px;right:20px;height:6px;"
    "z-index:11;border-radius:2px;overflow:hidden;pointer-events:none;"
    "background:rgba(255,255,255,0.08)}\n"
    "  #tl-progress{height:100%;border-radius:2px;transition:width 0.1s}\n"
    "</style>\n"
    f'<script src="{BABYLON_CDN}"></script>\n'
    f'<script src="{BABYLON_MAT}"></script>\n'
    f'<script src="{BABYLON_LOADERS}"></script>\n'
    "</head>\n<body>\n"
    '<canvas id="c" tabindex="0"></canvas>\n'
    '<div id="phase-banner"><div id="phase-name">Initializing...</div>'
    '<div id="phase-desc"></div></div>\n'
    '<div id="hud"></div>\n'
    '<div id="badge"></div>\n'
    '<div id="vis-mode"></div>\n'
    '<div id="timeline"><div id="tl-progress"></div></div>\n'
    '<div id="layers"></div>\n'
    '<div id="info-panel"></div>\n'
    '<div id="colorbar">\n'
    '  <canvas id="cb-gradient" width="28" height="200"></canvas>\n'
    '  <span id="cb-max"></span>\n'
    '  <span id="cb-min"></span>\n'
    '  <span id="cb-label"></span>\n'
    '</div>\n'
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
  mhd: "#5050aa",
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
  if (scene.L.velocity) layers.push("velocity");
  if (scene.L.instability) layers.push("instability");
  if (scene.L.radiation) layers.push("radiation");
  if (scene.L.yield_map) layers.push("yield");
  badgeEl.innerHTML = "<b>" + scene.L.device + "</b><br>" +
    "Backend: " + (scene.L.backend || "lee") + " | " + scene.gpuBackend +
    (scene.useGPU ? " (GPU)" : "") + "<br>" +
    "Physics: " + (layers.length > 0 ? layers.join(", ") : "circuit only");

  // Visualization mode banner — honest labeling of data source.
  // The 3D scene geometry (sheath, pinch, particles, B-field rings) is ALWAYS
  // driven by Lee model 0D scalars (z_mm, r_mm, I_MA, phase). MHD field data
  // only appears in the optional midplane heatmaps and HUD peak values.
  // This banner makes that distinction visible to the user.
  var visModeEl = document.getElementById("vis-mode");
  var backend = (scene.L.backend || "lee").toLowerCase();
  var hasMHD = !!scene.L.has_mhd;
  var hasMHDFields = !!(scene.L.density || scene.L.temperature || scene.L.bfield || scene.L.velocity);
  if (hasMHD && hasMHDFields) {
    visModeEl.className = "mhd";
    visModeEl.innerHTML = "3D geometry: MHD-driven (isosurface, field lines, particles) &bull; Heatmaps: MHD field data (" + backend + ")";
  } else {
    visModeEl.className = "lee";
    visModeEl.innerHTML = "Visualization: Lee model schematic (not MHD field data)";
  }

  // ---- Phase timeline colored bands ----
  var tlEl = document.getElementById("timeline");
  var prevPhase = "", segStart = 0;
  for (var fi2 = 0; fi2 <= scene.S.n_frames; fi2++) {
    var curPhase = fi2 < scene.S.n_frames ? scene.S.frames[fi2].phase : "__end__";
    if (curPhase !== prevPhase && fi2 > 0) {
      var pct0 = (segStart / scene.S.n_frames) * 100;
      var pctW = ((fi2 - segStart) / scene.S.n_frames) * 100;
      var seg = document.createElement("div");
      seg.style.cssText = "position:absolute;top:0;height:100%;opacity:0.6;left:" +
        pct0 + "%;width:" + pctW + "%;background:" + (PHASE_BAR_COLORS[prevPhase] || "#334");
      seg.title = (PHASE_LABELS[prevPhase] || prevPhase);
      tlEl.appendChild(seg);
      segStart = fi2;
    }
    prevPhase = curPhase;
  }

  // ---- Physics explanation panel — updates dynamically with heatmap mode and phase ----
  var infoEl = document.getElementById("info-panel");
  var activeHeatmapMode = "none";

  var HEATMAP_INFO = {
    none: {
      title: "3D DPF Device Overview",
      body: "The <b>wireframe cylinder</b> is the anode (inner electrode, copper). " +
            "<b>Wireframe rods</b> are the cathode (outer electrode cage, steel). " +
            "During discharge, a <b>glowing torus</b> sweeps along the device (current sheath) " +
            "then compresses at the tip. The <b>bright column</b> at the anode tip is the pinch " +
            "-- fusion-relevant conditions.<br><br>" +
            "Color progression: <b>red-orange</b> (rundown, D-alpha emission) to <b>amber</b> (compression) to <b>white-hot</b> (pinch).<br>" +
            "<b>Enable a heatmap</b> to wrap MHD field data around the device."
    },
    density: {
      title: "Density Heatmap (MHD)",
      body: "Color shows <b>plasma mass density</b> rho(r,z) wrapped around the device. " +
            "<b>Blue/purple = low density</b> (background fill gas). " +
            "<b>Green/yellow = high density</b> (compressed plasma). " +
            "The viridis colormap wraps 360 degrees around the midplane cylinder. " +
            "Density structure reveals sheath thickness and compression zones."
    },
    temperature: {
      title: "Temperature Heatmap (MHD)",
      body: "Color shows <b>electron temperature</b> Te(r,z) wrapped around the device. " +
            "Uses <b>inferno colormap</b> (black to purple to orange to yellow). " +
            "<b>Dark = cold plasma</b> (~1 eV). <b>Bright yellow = hot plasma</b>. " +
            "Temperature peaks at the pinch axis from adiabatic compression and Ohmic heating."
    },
    bfield: {
      title: "Magnetic Field |B| (MHD)",
      body: "Color shows <b>magnetic field magnitude</b> |B|(r,z) on the midplane. " +
            "<b>Blue = weak field</b>. <b>Yellow/red = strong field</b> (up to several Tesla). " +
            "The toroidal B_theta is generated by the axial current: B ~ mu_0*I/(2*pi*r). " +
            "Strongest near the anode surface, it provides the J x B force driving the implosion."
    },
    radiation: {
      title: "Radiation Power (MHD)",
      body: "Color shows <b>radiation power density</b> P_rad(r,z) in W/m3. " +
            "<b>Blue = low emission</b>. <b>Yellow/red = intense radiation</b>. " +
            "Scales as ne^2 * sqrt(Te) (bremsstrahlung). Strongest in the dense hot pinch, " +
            "it acts as an energy loss mechanism limiting peak temperature."
    }
  };

  var PHASE_INFO = {
    rundown: "AXIAL RUNDOWN: Current sheath sweeps neutral gas from insulator to anode tip. " +
             "Duration ~1-3 us. Sheath accelerates under J x B force.",
    radial: "RADIAL IMPLOSION: Sheath compresses inward as a magnetic piston. " +
            "Compression ratio 10-50:1. Duration ~100-300 ns.",
    mhd_radial: "MHD RADIAL IMPLOSION: Full MHD simulation of radial compression.",
    reflected: "REFLECTED SHOCK: After pinch, shock bounces outward from axis, reheating the plasma.",
    pinch: "PEAK COMPRESSION: Maximum density and temperature at the axis. " +
           "Fusion reactions occur here. Ion temperature can exceed 1 keV.",
    post_pinch: "POST-PINCH: m=0 sausage instability breaks up the plasma column. " +
                "Pinch neck-off accelerates beam ions to MeV energies."
  };

  function updateInfoPanel(hmMode, phase) {
    if (!infoEl) return;
    var info = HEATMAP_INFO[hmMode] || HEATMAP_INFO.none;
    var phaseNote = PHASE_INFO[phase] || "";
    infoEl.innerHTML = "<div class='info-title'>" + info.title + "</div>" +
      "<div class='info-body'>" + info.body + "</div>" +
      (phaseNote ? "<div class='info-phase'>" + phaseNote + "</div>" : "");
  }
  updateInfoPanel("none", "rundown");

  // ---- Colorbar rendering ----
  var cbEl = document.getElementById("colorbar");
  var cbCanvas = document.getElementById("cb-gradient");
  var cbMax = document.getElementById("cb-max");
  var cbMin = document.getElementById("cb-min");
  var cbLabel = document.getElementById("cb-label");

  var COLORBAR_CONFIG = {
    density: { label: "Density [kg/m\u00b3]", min: scene.L.density ? scene.L.density.min_val : 0,
               max: scene.L.density ? scene.L.density.max_val : 1, cmap: "viridis" },
    temperature: { label: "Temperature [eV]", min: scene.L.temperature ? scene.L.temperature.min_eV : 0,
                   max: scene.L.temperature ? scene.L.temperature.max_eV : 1, cmap: "inferno" },
    bfield: { label: "|B| [Tesla]", min: 0, max: scene.L.bfield ? scene.L.bfield.max_T : 1, cmap: "viridis" },
    radiation: { label: "P_rad [W/m\u00b3]", min: 0, max: scene.L.radiation ? scene.L.radiation.max_W_m3 : 1, cmap: "inferno" }
  };

  var CMAP_STOPS = {
    viridis: [[0,"#440154"],[0.25,"#3b528b"],[0.5,"#21918c"],[0.75,"#5ec962"],[1.0,"#fde725"]],
    inferno: [[0,"#000004"],[0.25,"#420a68"],[0.5,"#932667"],[0.75,"#dd513a"],[1.0,"#fcffa4"]]
  };

  function updateColorbar(mode) {
    if (mode === "none" || !COLORBAR_CONFIG[mode]) {
      cbEl.style.display = "none";
      return;
    }
    cbEl.style.display = "block";
    var cfg = COLORBAR_CONFIG[mode];
    var cmapKey = cfg.cmap || "viridis";
    var stops = CMAP_STOPS[cmapKey] || CMAP_STOPS.viridis;

    var ctx = cbCanvas.getContext("2d");
    var grad = ctx.createLinearGradient(0, 0, 0, 200);
    for (var s = 0; s < stops.length; s++) {
      grad.addColorStop(1 - stops[s][0], stops[s][1]);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 28, 200);

    function fmtVal(v) {
      if (Math.abs(v) < 0.01 || Math.abs(v) > 1e5) return v.toExponential(1);
      if (Math.abs(v) < 10) return v.toFixed(2);
      return v.toFixed(0);
    }
    cbMax.textContent = fmtVal(cfg.max);
    cbMin.textContent = fmtVal(cfg.min);
    cbLabel.textContent = cfg.label;
  }

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
      updateInfoPanel(activeHeatmapMode, f.phase);
    }

    // Timeline progress bar
    var pct = (i / Math.max(scene.S.n_frames - 1, 1)) * 100;
    tlProgress.style.width = pct + "%";
    tlProgress.style.background = PHASE_BAR_COLORS[f.phase] || "#48f";

    // Data HUD — show ALL available physics data
    var lines = [];
    lines.push("t = " + f.t.toFixed(2) + " us");
    lines.push("I = " + f.I.toFixed(3) + " MA  (" +
      (scene.S.I_peak > 0 ? (f.I / scene.S.I_peak * 100).toFixed(0) : "0") + "%)");
    // Sheath position during rundown, radius during radial
    if (!isP) {
      lines.push("z = " + f.z.toFixed(1) + " mm");
    } else {
      var ratio = (scene.G.cathode_radius / Math.max(f.r, 0.01)).toFixed(0);
      lines.push("r = " + f.r.toFixed(1) + " mm  (" + ratio + ":1)");
    }
    // MHD field data (peak values from final state)
    if (scene.L.density) lines.push("rho_peak = " + scene.L.density.max_val.toExponential(1) + " kg/m3");
    if (scene.L.temperature && scene.L.temperature.max_eV > 1)
      lines.push("Te_peak = " + scene.L.temperature.max_eV.toFixed(0) + " eV");
    if (scene.L.bfield) lines.push("|B|_peak = " + scene.L.bfield.max_T.toFixed(1) + " T");
    // Instability
    if (pI > 0.1 && rippleAmp > 0) {
      lines.push("m=0: " + (rippleAmp * 100).toFixed(0) + "%");
      if (scene.L.instability) lines.push("tau_m0 = " + scene.L.instability.tau_m0_ns.toFixed(0) + " ns");
    }
    // Radiation (MHD only)
    if (scene.L.radiation && isP)
      lines.push("P_rad = " + scene.L.radiation.max_W_m3.toExponential(1) + " W/m3");
    // Neutron yield (deuterium + MHD only)
    if (scene.L.yield_map && pI > 0.1)
      lines.push("Yn_rate = " + scene.L.yield_map.max_rate.toExponential(1) + " /m3/s");
    // Beam ions
    if (scene.L.beam && pI > 0.3)
      lines.push("Beam: " + scene.L.beam.n_particles + " ions, " +
        scene.L.beam.mean_energy_keV.toFixed(0) + " keV mean");
    // Pinch metrics
    if (scene.L.pinch && isP)
      lines.push("Pinch r = " + scene.L.pinch.radius_mm.toFixed(2) + " mm");
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
    fi = 0; smoothFi = 0; sl.value = 0; playing = false;
    tl.textContent = "t = 0 us";
    renderFrame(0);
  };
  sl.oninput = function() {
    fi = +sl.value; smoothFi = fi;
    renderFrame(fi);
    tl.textContent = "t = " + scene.S.frames[fi].t.toFixed(1) + " us";
  };

  // ---- Layer toggles (HTML DOM — scales properly on Retina) ----
  var lp = document.getElementById("layers");

  function addHdr(text) {
    var d = document.createElement("div");
    d.className = "hdr"; d.textContent = text;
    lp.appendChild(d);
  }
  function addTog(label, checked, fn) {
    var lb = document.createElement("label");
    var cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = checked;
    cb.onchange = function() { fn(cb.checked); };
    lb.appendChild(cb);
    lb.appendChild(document.createTextNode(label));
    lp.appendChild(lb);
    fn(checked);
  }

  addHdr("LAYERS");
  addTog("Electrodes", true, function(v) {
    scene.anode.isVisible = v;
    scene.cathodeRods.forEach(function(r) { r.isVisible = v; });
    scene.insulator.isVisible = v;
  });
  addTog("Current Sheath", true, function(v) {
    scene.sheathDisk.isVisible = v;
    if (scene.gasGlow) scene.gasGlow.isVisible = v;
  });
  addTog("Plasma Ions", true, function(v) {
    if (v) scene.ps.start(); else scene.ps.stop();
  });
  addTog("Pinch Column", true, function(v) {
    scene.pinchCore.isVisible = v;
    scene.pinchMantle.isVisible = v;
  });
  addTog("B-Field Rings", true, function(v) {
    scene.bRings.forEach(function(r) { r.isVisible = v; });
  });
  addTog("Beam Indicator", true, function(v) {
    scene.beamCone.isVisible = v;
  });
  addTog("Current Flow", false, function(v) {
    if (scene.currentArrows) {
      scene.currentArrows.axialArrows.forEach(function(a) { a._userVisible = v; });
      scene.currentArrows.radialArrows.forEach(function(a) { a._userVisible = v; });
      scene.currentArrows.returnArrows.forEach(function(a) { a._userVisible = v; });
    }
  });

  // Heatmap toggles — mutually exclusive (radio-like). Only one heatmap
  // overlay can be active at a time since they share a single midplane texture.
  var heatmapCheckboxes = [];
  function setHeatmapMode(key, sourceCb) {
    // Uncheck all other heatmap checkboxes
    heatmapCheckboxes.forEach(function(entry) {
      if (entry.cb !== sourceCb) entry.cb.checked = false;
    });
    var mode = sourceCb.checked ? key : "none";
    scene.setOverlay(mode);
    scene.updateHeatmap(mode);
    activeHeatmapMode = mode;
    updateInfoPanel(mode, lastPhase);
    updateColorbar(mode);
  }
  function addHeatTog(label, key) {
    var lb = document.createElement("label");
    var cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = false;
    cb.onchange = function() { setHeatmapMode(key, cb); };
    lb.appendChild(cb);
    lb.appendChild(document.createTextNode(label));
    lp.appendChild(lb);
    heatmapCheckboxes.push({ cb: cb, key: key });
  }
  if (scene.L.density || scene.L.temperature || scene.L.bfield) {
    addHdr("MHD FIELD DATA");
    if (scene.L.density)    addHeatTog("Density Heatmap", "density");
    if (scene.L.temperature) addHeatTog("Temperature Heatmap", "temperature");
    if (scene.L.bfield)     addHeatTog("|B| Heatmap", "bfield");
    if (scene.L.radiation)  addHeatTog("Radiation Heatmap", "radiation");
    // Overlay rendering mode selector (surface cylinder vs volumetric vs cross-section)
    if (scene.volField) {
      var modeDiv = document.createElement("div");
      modeDiv.style.cssText = "margin:6px 0 2px;font-size:11px;color:#aaa;";
      modeDiv.textContent = "Render Mode:";
      lp.appendChild(modeDiv);
      var modes = [["Surface", "surface"], ["Volume", "volume"], ["Cross-section", "xsec"], ["Vol+Xsec", "both"]];
      modes.forEach(function(m) {
        var lb = document.createElement("label");
        lb.style.cssText = "display:inline-block;margin-right:8px;font-size:11px;";
        var rb = document.createElement("input");
        rb.type = "radio"; rb.name = "ovMode"; rb.value = m[1];
        if (m[1] === "surface") rb.checked = true;
        rb.onchange = function() { scene.setOverlayMode(m[1]); };
        lb.appendChild(rb);
        lb.appendChild(document.createTextNode(m[0]));
        lp.appendChild(lb);
      });
    }
  }

  addHdr("RENDERING");
  addTog("Bloom", true, function(v) { scene.pipeline.bloomEnabled = v; });
  if (scene.ssao) {
    addTog("Ambient Occlusion", true, function(v) {
      scene.ssao.totalStrength = v ? 0.8 : 0;
    });
  }

  // ---- Render loop (smooth interpolated playback) ----
  var smoothFi = 0;
  var lastTime = performance.now();
  scene.engine.runRenderLoop(function() {
    try {
      var now = performance.now();
      var dt = Math.min(0.05, (now - lastTime) * 0.001);
      lastTime = now;
      if (playing) {
        var speed = SPEEDS[speedIdx] || 1;
        // Frame-rate-independent smooth advancement
        smoothFi += speed * dt * 60 / Math.max(scene.S.n_frames, 1) * 2;
        if (smoothFi >= scene.S.n_frames) smoothFi = 0;
        var newFi = Math.floor(smoothFi);
        if (newFi !== fi) {
          fi = newFi;
          sl.value = fi;
          tl.textContent = "t = " + scene.S.frames[fi].t.toFixed(1) + " us";
          renderFrame(fi);
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
  lastPhase = "";  // Reset so next renderFrame triggers phase banner update
});
"""


def create_unified_renderer(d: dict[str, Any]) -> str:
    """Build a self-contained HTML page with the Babylon.js DPF renderer.

    Pipeline step 4: Python dict -> JSON -> embedded <script> constant.
    The JSON payload contains all 10 physics layers from extract_all_layers().
    MHD field arrays are base64-encoded Float32, decoded client-side into
    RGBA textures via colormap lookup. Snapshot frames are pre-decoded at
    scene creation time into snapCache for zero-allocation playback.
    """
    # Pipeline step 3->4: extract layers, serialize to JSON
    layers = extract_all_layers(d)
    data_json = json.dumps(layers)

    # Pipeline step 4->5: inject JSON into JS host code, combine with renderer
    host_code = _HTML_HOST.replace("%%DATA_JSON%%", data_json)
    return (
        _HTML_HEAD
        + "<script>\n"
        + "// ---- Volumetric field module (dpf_volumetric.js) ----\n"
        + "// Ray-march + cross-section for 2D axisymmetric field data.\n"
        + _VOLUMETRIC_JS + "\n\n"
        + "// ---- Renderer module (dpf_renderer.js) ----\n"
        + "// Creates Babylon.js scene, decodes base64 field data, builds\n"
        + "// snap caches, and exposes applyFrame(i) + updateHeatmap(mode).\n"
        + _RENDERER_JS + "\n\n"
        + "// ---- Host code (frame loop, layer toggles, HUD) ----\n"
        + host_code + "\n"
        + "</script>\n</body></html>"
    )


def create_unified_iframe(d: dict[str, Any], height: int = 620) -> str:
    """Pipeline step 6: write renderer HTML to temp file, load via data: URI.

    Uses a file-based approach: writes HTML to static/ directory, then
    constructs a data: URI from the file content. This avoids both srcdoc
    escaping overhead AND Gradio file routing issues.
    """
    import base64 as _b64mod
    html = create_unified_renderer(d)
    # Encode as data: URI — no escaping, no file routing, no size limit
    html_b64 = _b64mod.b64encode(html.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{html_b64}" '
        f'title="3D Dense Plasma Focus Visualization" '
        f'role="img" aria-label="Interactive 3D animation of the DPF discharge" '
        f'style="width:100%;height:{height}px;border:none;background:#e0e4e8;" '
        f'allow="accelerometer; camera; gyroscope; xr-spatial-tracking"></iframe>'
    )
