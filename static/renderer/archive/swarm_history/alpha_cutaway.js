/**
 * DPF-Unified Plasma Renderer — Engineering Cutaway
 *
 * Half-section view: cathode rods from 0..PI only, flat r-z cross-section
 * plane at angle=0, sheath/compression shown as colored bands on the
 * cutaway face. Pinch column extends beyond anode tip as 3D geometry.
 *
 * Designed for investor/student presentations — Lockheed-style technical
 * illustration with clean, professional look.
 */

// ============================================================
// COLORMAPS
// ============================================================

const VIRIDIS = [
  [0.267,0.004,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],[0.207,0.372,0.553],
  [0.164,0.471,0.558],[0.128,0.567,0.551],[0.134,0.658,0.517],[0.267,0.749,0.441],
  [0.478,0.821,0.318],[0.741,0.873,0.150],[0.993,0.906,0.144]
];
const CIVIDIS = [
  [0.0,0.135,0.305],[0.0,0.206,0.380],[0.133,0.273,0.385],[0.259,0.335,0.384],
  [0.365,0.397,0.395],[0.463,0.461,0.420],[0.563,0.529,0.444],[0.666,0.604,0.452],
  [0.775,0.685,0.432],[0.888,0.775,0.380],[1.0,0.871,0.298]
];

let activeCmap = VIRIDIS;

// ============================================================
// CONSTANTS
// ============================================================

const PHASE_COLORS = {
  rundown:    [0.2, 0.5, 1.0],
  radial:     [1.0, 0.3, 0.08],
  mhd_radial: [1.0, 0.3, 0.08],
  reflected:  [1.0, 0.55, 0.0],
  pinch:      [1.0, 0.08, 0.03],
  post_pinch: [0.7, 0.15, 0.08],
};

const PHASE_LABELS = {
  rundown:    "Axial rundown",
  radial:     "Radial implosion",
  mhd_radial: "Radial compression",
  mhd:        "MHD simulation",
  reflected:  "Reflected shock",
  pinch:      "Pinch -- peak compression",
  post_pinch: "Post-pinch disruption",
  none:       "",
};

const PHASE_DESCRIPTIONS = {
  rundown:    "Current sheath sweeps neutral gas from insulator to anode tip -- magnetic snowplow",
  radial:     "Magnetic pressure compresses plasma ring inward toward the axis",
  mhd_radial: "J x B force drives radial implosion -- compression heating the plasma",
  mhd:        "Full MHD simulation of plasma dynamics",
  reflected:  "Reflected shock expands outward after axis convergence",
  pinch:      "PEAK COMPRESSION -- fusion-relevant conditions at the axis",
  post_pinch: "m=0 sausage instability breaks up the plasma column",
};

const SPEEDS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];

// ============================================================
// UTILITY
// ============================================================

function cmapLookup(v, cmap) {
  var t = Math.max(0, Math.min(1, v));
  var n = cmap.length - 1;
  var idx = t * n;
  var lo = Math.floor(idx), hi = Math.min(lo + 1, n);
  var f = idx - lo;
  return [
    cmap[lo][0] + (cmap[hi][0] - cmap[lo][0]) * f,
    cmap[lo][1] + (cmap[hi][1] - cmap[lo][1]) * f,
    cmap[lo][2] + (cmap[hi][2] - cmap[lo][2]) * f,
  ];
}

function b64ToFloat32(b64) {
  var raw = atob(b64);
  var buf = new ArrayBuffer(raw.length);
  var bytes = new Uint8Array(buf);
  for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(buf);
}

function isRadialPhase(phase) {
  return ["radial", "mhd_radial", "pinch", "reflected", "post_pinch"].indexOf(phase) >= 0;
}

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp01(v) { return Math.max(0, Math.min(1, v)); }

// ============================================================
// ENGINE INIT
// ============================================================

async function initEngine(canvas) {
  var engine, gpuBackend = "WebGL2";
  var useWebGPU = (new URLSearchParams(window.location.search)).get("webgpu") === "1";
  if (useWebGPU) {
    try {
      if (await BABYLON.WebGPUEngine.IsSupportedAsync) {
        engine = new BABYLON.WebGPUEngine(canvas, {
          antialias: true, adaptToDeviceRatio: true, powerPreference: "high-performance",
        });
        await engine.initAsync();
        gpuBackend = "WebGPU";
      }
    } catch (_) {}
  }
  if (!engine) {
    engine = new BABYLON.Engine(canvas, true, {
      stencil: true, adaptToDeviceRatio: true, preserveDrawingBuffer: true,
    });
  }
  engine.setHardwareScalingLevel(1 / window.devicePixelRatio);
  var scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.12, 0.13, 0.15, 1);
  return { engine: engine, scene: scene, gpuBackend: gpuBackend };
}

// ============================================================
// CAMERA
// ============================================================

function createCamera(scene, canvas, G) {
  var cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI * 0.35, Math.PI * 0.32, G.cathode_radius * 9,
    new BABYLON.Vector3(G.anode_length * 0.45, 0, 0), scene);
  cam.attachControl(canvas, false);
  cam.inputs.removeByType("ArcRotateCameraMouseWheelInput");
  canvas.addEventListener("wheel", function(e) {
    e.preventDefault();
    cam.radius -= e.deltaY * 0.05;
    cam.radius = Math.max(cam.lowerRadiusLimit, Math.min(cam.upperRadiusLimit, cam.radius));
  }, { passive: false });
  cam.lowerRadiusLimit = G.anode_radius * 0.5;
  cam.upperRadiusLimit = G.cathode_radius * 60;
  cam.pinchPrecision = 15;
  cam.panningSensibility = 60;
  cam.minZ = 0.0005;
  cam.inertia = 0.88;

  var autoOrbit = true, userInteracting = false, interactionTimeout = null;
  canvas.addEventListener("pointerdown", function() {
    userInteracting = true; autoOrbit = false;
    if (interactionTimeout) clearTimeout(interactionTimeout);
  });
  canvas.addEventListener("pointerup", function() {
    userInteracting = false;
    interactionTimeout = setTimeout(function() { autoOrbit = true; }, 6000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !userInteracting) cam.alpha += 0.0008;
  });
  return cam;
}

// ============================================================
// LIGHTS
// ============================================================

function createLights(scene) {
  var key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  key.intensity = 1.3;
  key.diffuse = new BABYLON.Color3(1, 0.98, 0.95);
  var back = new BABYLON.DirectionalLight("back", new BABYLON.Vector3(1, -1, -1), scene);
  back.intensity = 0.5;
  back.diffuse = new BABYLON.Color3(0.9, 0.92, 0.95);
  var fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.45;
  fill.diffuse = new BABYLON.Color3(0.9, 0.92, 1.0);
  fill.groundColor = new BABYLON.Color3(0.35, 0.35, 0.4);
}

// ============================================================
// DEVICE (half-section cutaway)
// ============================================================

function createDevice(scene, G) {
  // Materials
  var copperMat = new BABYLON.StandardMaterial("copper", scene);
  copperMat.diffuseColor = new BABYLON.Color3(0.68, 0.45, 0.25);
  copperMat.specularColor = new BABYLON.Color3(0.8, 0.6, 0.3);
  copperMat.specularPower = 48;

  var steelMat = new BABYLON.StandardMaterial("steel", scene);
  steelMat.diffuseColor = new BABYLON.Color3(0.41, 0.41, 0.48);
  steelMat.specularColor = new BABYLON.Color3(0.5, 0.5, 0.55);
  steelMat.specularPower = 32;

  var ceramicMat = new BABYLON.StandardMaterial("ceramic", scene);
  ceramicMat.diffuseColor = new BABYLON.Color3(0.95, 0.92, 0.85);
  ceramicMat.specularColor = new BABYLON.Color3(0.15, 0.15, 0.15);
  ceramicMat.specularPower = 8;

  // Anode: full cylinder (solid, always visible)
  var anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 64, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.renderingGroupId = 0;

  // Cathode rods: only angles 0..PI (half-section)
  var N_RODS = G.n_cathode_rods || 12;
  var rodDiam = G.cathode_rod_diameter || G.cathode_radius * 0.06;
  var cathodeRods = [];
  for (var i = 0; i < N_RODS; i++) {
    var angle = (i / N_RODS) * Math.PI * 2;
    if (angle > Math.PI + 0.01 && angle < Math.PI * 2 - 0.01) continue;
    var rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: rodDiam, height: G.anode_length * 1.05, tessellation: 12,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(G.anode_length / 2,
      G.cathode_radius * Math.sin(angle),
      G.cathode_radius * Math.cos(angle));
    rod.material = steelMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  // Cathode ring connectors (half arcs)
  var ringThk = (G.cathode_radius - G.anode_radius) * 0.18;
  var baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk,
    tessellation: 64, arc: 0.5,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.rotation.x = -Math.PI / 2;
  baseRing.position.x = -G.anode_length * 0.025;
  baseRing.material = steelMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);
  var topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length * 1.025;
  cathodeRods.push(topRing);

  // Insulator
  var insThk = G.anode_radius * 0.15;
  var insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.3;
  var insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 64, arc: 0.5,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.rotation.x = -Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  return { anode: anode, cathodeRods: cathodeRods, insulator: insulator,
           copperMat: copperMat, steelMat: steelMat, ceramicMat: ceramicMat };
}

// CROSS-SECTION PLANE (r-z cutaway face at angle=0)
function createCrossSection(scene, G) {
  var NR = 20, NZ = 40, paths = [];
  for (var ir = 0; ir <= NR; ir++) {
    var r = G.cathode_radius * ir / NR, row = [];
    for (var iz = 0; iz <= NZ; iz++) row.push(new BABYLON.Vector3(G.anode_length * iz / NZ, 0, r));
    paths.push(row);
  }
  var ribbon = BABYLON.MeshBuilder.CreateRibbon("crossSection", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false,
  }, scene);
  var mat = new BABYLON.StandardMaterial("crossMat", scene);
  mat.diffuseColor = new BABYLON.Color3(0.08, 0.09, 0.11);
  mat.specularColor = BABYLON.Color3.Black();
  mat.emissiveColor = new BABYLON.Color3(0.06, 0.07, 0.09);
  mat.alpha = 0.65; mat.backFaceCulling = false;
  ribbon.material = mat; ribbon.renderingGroupId = 0;
  return ribbon;
}

// SHEATH BAND on cross-section (dynamic texture overlay)
function createSheathBand(scene, G) {
  var texW = 512, texH = 256;
  var dynTex = new BABYLON.DynamicTexture("sheathTex", { width: texW, height: texH }, scene, false);
  dynTex.hasAlpha = true;

  var plane = BABYLON.MeshBuilder.CreatePlane("sheathBand", {
    width: G.anode_length, height: G.cathode_radius,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE,
  }, scene);
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(G.anode_length / 2, 0.002, G.cathode_radius / 2);
  var mat = new BABYLON.StandardMaterial("sheathBandMat", scene);
  mat.diffuseTexture = dynTex;
  mat.emissiveTexture = dynTex;
  mat.disableLighting = true;
  mat.alpha = 0.9;
  mat.useAlphaFromDiffuseTexture = true;
  mat.backFaceCulling = false;
  plane.material = mat;
  plane.renderingGroupId = 1;

  return { plane: plane, dynTex: dynTex, texW: texW, texH: texH };
}

function drawSheathOnTexture(band, G, f) {
  var ctx = band.dynTex.getContext();
  var w = band.texW, h = band.texH;
  ctx.clearRect(0, 0, w, h);

  var col = PHASE_COLORS[f.phase] || [0.3, 0.5, 1.0];
  var Ifrac = clamp01(Math.abs(f.I / 2.1));
  if (Ifrac < 0.01) { band.dynTex.update(); return; }

  var isP = isRadialPhase(f.phase);
  var zNorm = clamp01(f.z / G.anode_length);
  var rNorm = clamp01(f.r / G.cathode_radius);
  var anodeRNorm = clamp01(G.anode_radius / G.cathode_radius);

  // Draw anode region (darkened copper hint)
  var anodeBottom = Math.round((1 - anodeRNorm) * h);
  ctx.fillStyle = "rgba(120, 80, 40, 0.12)";
  ctx.fillRect(0, anodeBottom, w, h - anodeBottom);

  // Sheath band
  var alpha = Math.round(clamp01(Ifrac * 0.85) * 255);
  var rc = Math.round(col[0] * 255), gc = Math.round(col[1] * 255), bc = Math.round(col[2] * 255);

  if (!isP) {
    // Rundown: vertical band at z-position, from anode_radius to cathode_radius
    var zPx = Math.round(zNorm * w);
    var bandW = Math.max(6, Math.round(w * 0.04));
    var rTop = Math.round((1 - 1.0) * h);
    var rBot = Math.round((1 - anodeRNorm) * h);
    var grad = ctx.createLinearGradient(zPx - bandW, 0, zPx + bandW, 0);
    grad.addColorStop(0, "rgba(" + rc + "," + gc + "," + bc + ",0)");
    grad.addColorStop(0.3, "rgba(" + rc + "," + gc + "," + bc + "," + (alpha/255).toFixed(2) + ")");
    grad.addColorStop(0.7, "rgba(" + rc + "," + gc + "," + bc + "," + (alpha/255).toFixed(2) + ")");
    grad.addColorStop(1, "rgba(" + rc + "," + gc + "," + bc + ",0)");
    ctx.fillStyle = grad;
    ctx.fillRect(zPx - bandW * 2, rTop, bandW * 4, rBot - rTop);

    // Swept gas glow behind sheath
    if (zNorm > 0.05) {
      ctx.fillStyle = "rgba(" + rc + "," + gc + "," + bc + ",0.06)";
      ctx.fillRect(0, rTop, zPx - bandW, rBot - rTop);
    }
  } else {
    // Radial/pinch: horizontal band at r-position near anode tip
    var zStart = Math.round(0.85 * w);
    var rPx = Math.round((1 - rNorm) * h);
    var bandH = Math.max(4, Math.round(h * 0.03));
    var rMinPx = Math.round((1 - anodeRNorm) * h);

    // Compression zone: filled region from cathode_r down to current r
    var gradR = ctx.createLinearGradient(0, rPx - bandH * 2, 0, rMinPx);
    gradR.addColorStop(0, "rgba(" + rc + "," + gc + "," + bc + "," + (alpha * 0.7 / 255).toFixed(2) + ")");
    gradR.addColorStop(1, "rgba(" + rc + "," + gc + "," + bc + ",0.05)");
    ctx.fillStyle = gradR;
    ctx.fillRect(zStart, rPx - bandH * 2, w - zStart, rMinPx - rPx + bandH * 4);

    // Bright sheath front
    ctx.fillStyle = "rgba(" + rc + "," + gc + "," + bc + "," + (alpha/255).toFixed(2) + ")";
    ctx.fillRect(zStart, rPx - bandH, w - zStart, bandH * 2);

    // Axis compression indicator
    if (rNorm < 0.3) {
      var intensity = clamp01((0.3 - rNorm) / 0.3);
      ctx.fillStyle = "rgba(255,240,200," + (intensity * 0.5).toFixed(2) + ")";
      var axisPx = Math.round((1 - 0) * h);
      ctx.fillRect(zStart, axisPx - 4, w - zStart, 8);
    }
  }

  band.dynTex.update();
}

// ============================================================
// PINCH COLUMN (3D, extends beyond anode tip)
// ============================================================

function createPinchColumn(scene, G) {
  var N = 20;
  var columnLen = G.anode_length * 0.3;
  var tipX = G.anode_length;
  var path = [];
  for (var k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(tipX + columnLen * k / N, 0, 0));
  }
  var radii = new Array(N + 1).fill(G.anode_radius * 0.12);

  var coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(1, 0.95, 0.85);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  var core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path, radiusFunction: function(i) { return radii[i] * 0.3; },
    tessellation: 16, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;

  var mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(1, 0.4, 0.1);
  mantleMat.disableLighting = true;
  mantleMat.alpha = 0;
  mantleMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mantleMat.backFaceCulling = false;

  var mantle = BABYLON.MeshBuilder.CreateTube("pinchMantle", {
    path: path, radiusFunction: function(i) { return radii[i]; },
    tessellation: 20, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  mantle.material = mantleMat;
  mantle.renderingGroupId = 1;

  return { core: core, mantle: mantle, coreMat: coreMat, mantleMat: mantleMat,
           radii: radii, path: path, N: N };
}

// ============================================================
// BEAM CONE (post-pinch)
// ============================================================

function createBeamCone(scene, G) {
  var cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.08,
    height: G.anode_length * 0.35, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length * 1.35 + G.anode_length * 0.175;
  var mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.6, 0.75, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone: cone, mat: mat };
}

// ============================================================
// GAS GLOW (subtle fill glow during rundown)
// ============================================================

function createGasGlow(scene, G) {
  var disk = BABYLON.MeshBuilder.CreateDisc("gasGlow", {
    radius: G.anode_radius * 1.05, tessellation: 32,
  }, scene);
  disk.rotation.y = Math.PI / 2;
  disk.position.x = G.anode_length;
  var mat = new BABYLON.StandardMaterial("gasGlowMat", scene);
  mat.emissiveColor = new BABYLON.Color3(1, 0.6, 0.2);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  disk.material = mat;
  disk.renderingGroupId = 1;
  return { disk: disk, mat: mat };
}

// GROUND GRID
function createGrid(scene, G) {
  var s = Math.max(G.anode_length * 3, G.cathode_radius * 6);
  var ground = BABYLON.MeshBuilder.CreateGround("grid", { width: s, height: s, subdivisions: 1 }, scene);
  ground.position.y = -G.cathode_radius * 1.2; ground.position.x = G.anode_length / 2;
  var tex = new BABYLON.DynamicTexture("gridTex", 512, scene, false);
  var ctx = tex.getContext();
  ctx.fillStyle = "rgba(210,215,220,1)"; ctx.fillRect(0, 0, 512, 512);
  ctx.strokeStyle = "rgba(160,165,175,0.5)"; ctx.lineWidth = 1;
  for (var i = 0; i <= 20; i++) { var p = i * 25.6;
    ctx.beginPath(); ctx.moveTo(p,0); ctx.lineTo(p,512); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,p); ctx.lineTo(512,p); ctx.stroke(); }
  tex.update();
  var mat = new BABYLON.StandardMaterial("gridMat", scene);
  mat.diffuseTexture = tex; mat.specularColor = BABYLON.Color3.Black();
  mat.emissiveColor = new BABYLON.Color3(0.10, 0.12, 0.14); mat.alpha = 0.75;
  ground.material = mat;
}

// HEATMAP OVERLAY (midplane ribbon for MHD field data)
function createHeatmap(scene, G) {
  var nr = 16, nz = 32, paths = [];
  for (var ir = 0; ir <= nr; ir++) {
    var r = G.cathode_radius * ir / nr, row = [];
    for (var iz = 0; iz <= nz; iz++) row.push(new BABYLON.Vector3(G.anode_length * iz / nz, 0.004, r));
    paths.push(row);
  }
  var plane = BABYLON.MeshBuilder.CreateRibbon("heatPlane", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false }, scene);
  plane.isVisible = false; plane.isPickable = false;
  var mat = new BABYLON.StandardMaterial("heatMat", scene);
  mat.disableLighting = true; mat.backFaceCulling = false;
  plane.material = mat; plane.renderingGroupId = 2;
  return { plane: plane, mat: mat };
}

function buildSnapCache(fieldKey, layer, cache) {
  if (!layer || !layer.frames || !layer.frames.length) return;
  var shape = layer.frames_shape || layer.shape;
  if (!shape) return;
  var nr = shape[0], nz = shape[1];
  var n = layer.frames.length;
  var times = new Float64Array(n);
  var rgbaFrames = new Array(n);
  for (var fi = 0; fi < n; fi++) {
    times[fi] = layer.frames[fi].t_us;
    var vals = b64ToFloat32(layer.frames[fi].data);
    var rgba = new Uint8Array(nz * nr * 4);
    for (var ir = 0; ir < nr; ir++) {
      for (var iz = 0; iz < nz; iz++) {
        var v = vals[ir * nz + iz];
        var c = cmapLookup(v, activeCmap);
        var pi = ((nr - 1 - ir) * nz + iz) * 4;
        rgba[pi]     = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 200;
      }
    }
    rgbaFrames[fi] = rgba;
  }
  cache[fieldKey] = { times: times, rgba: rgbaFrames, texW: nz, texH: nr };
}

function nearestSnapIdx(cache, key, t) {
  var e = cache[key];
  if (!e) return -1;
  var lo = 0, hi = e.times.length - 1;
  while (lo < hi) {
    var m = (lo + hi) >> 1;
    if (e.times[m] < t) lo = m + 1; else hi = m;
  }
  if (lo > 0 && Math.abs(e.times[lo - 1] - t) < Math.abs(e.times[lo] - t)) return lo - 1;
  return lo;
}

// PIPELINE
var GLOW_MESHES = new Set(["pinchCore", "pinchMantle", "beamCone", "sheathBand"]);

function createPipeline(scene, cam) {
  var pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true; pipe.bloomWeight = 0.2; pipe.bloomThreshold = 0.85;
  pipe.bloomKernel = 64; pipe.bloomScale = 0.5; pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true; pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.0; pipe.sharpenEnabled = true; pipe.sharpen.edgeAmount = 0.15;
  var ssao = null;
  try { ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
    { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.6; ssao.radius = 1.5; ssao.samples = 16; ssao.base = 0.2;
  } catch (_) {}
  var glow = new BABYLON.GlowLayer("glow", scene, { blurKernelSize: 32, mainTextureFixedSize: 512 });
  glow.intensity = 0.5;
  glow.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (GLOW_MESHES.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      var ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else { result.set(0, 0, 0, 0); }
  };
  return { pipeline: pipe, ssao: ssao, glowLayer: glow };
}

// MAIN SCENE
async function createDPFScene(canvas, data) {
  var L = data, G = L.geometry, S = L.sheath;
  var init = await initEngine(canvas);
  var engine = init.engine, scene = init.scene, gpuBackend = init.gpuBackend;
  var cam = createCamera(scene, canvas, G);

  createLights(scene);
  var device = createDevice(scene, G);
  var crossSection = createCrossSection(scene, G);
  var band = createSheathBand(scene, G);
  var pinch = createPinchColumn(scene, G);
  var beam = createBeamCone(scene, G);
  var gasGlow = createGasGlow(scene, G);
  createGrid(scene, G);
  var heat = createHeatmap(scene, G);
  var pipeState = createPipeline(scene, cam);
  var pipeline = pipeState.pipeline, ssao = pipeState.ssao, glowLayer = pipeState.glowLayer;

  // Snap cache for MHD field overlays
  var snapCache = {}, lastSnapIdx = { density: -1, temperature: -1, bfield: -1 }, heatTex = null;
  buildSnapCache("density", L.density, snapCache);
  buildSnapCache("temperature", L.temperature, snapCache);
  buildSnapCache("bfield", L.bfield, snapCache);

  function applySnapTex(key) {
    var c = snapCache[key];
    if (!c) return;
    var idx = lastSnapIdx[key];
    if (idx < 0 || idx >= c.rgba.length) return;
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(c.rgba[idx], c.texW, c.texH,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene, false, false,
      BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.85;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  var activeOverlay = "none";

  function updateHeatmap(key) {
    if (!L || key === "none") { heat.plane.isVisible = false; return; }
    var layer = L[key];
    if (!layer || (!layer.data && !layer.frames)) { heat.plane.isVisible = false; return; }
    if (snapCache[key] && lastSnapIdx[key] >= 0) { applySnapTex(key); return; }
    if (!layer.data || !layer.shape) { heat.plane.isVisible = false; return; }
    var vals = b64ToFloat32(layer.data);
    var nr = layer.shape[0], nz = layer.shape[1];
    var rgba = new Uint8Array(nz * nr * 4);
    for (var ir = 0; ir < nr; ir++) {
      for (var iz = 0; iz < nz; iz++) {
        var v = vals[ir * nz + iz], c = cmapLookup(v, activeCmap);
        var pi = ((nr - 1 - ir) * nz + iz) * 4;
        rgba[pi]     = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 200;
      }
    }
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(rgba, nz, nr, BABYLON.Engine.TEXTUREFORMAT_RGBA,
      scene, false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.85;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  // Dummy sheath disk reference (the band plane serves this role)
  var sheathDisk = band.plane;

  // --------------------------------------------------------
  // applyFrame
  // --------------------------------------------------------
  function applyFrame(i) {
    if (i < 0 || i >= S.frames.length) return;
    var f = S.frames[i];
    var isP = isRadialPhase(f.phase);
    var cr = Math.max(0.02, f.r / G.cathode_radius);
    var Ifrac = clamp01(Math.abs(f.I / Math.max(S.I_peak, 0.001)));
    var pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;

    // Heatmap snap sync
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      var ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // Cross-section sheath band
    drawSheathOnTexture(band, G, f);

    // Gas glow at anode tip during compression
    if (isP && Ifrac > 0.1) {
      gasGlow.disk.isVisible = true;
      gasGlow.mat.alpha = Math.min(0.35, pI * 0.35);
      var h = Math.min(1, pI);
      gasGlow.mat.emissiveColor.set(0.3 + h * 0.7, 0.2 + h * 0.4, 0.1 + h * 0.1);
    } else {
      gasGlow.disk.isVisible = false;
    }

    // Pinch column (beyond anode tip)
    var showPinch = (f.phase === "pinch" || f.phase === "post_pinch" || f.phase === "reflected") ||
                    (isP && cr < 0.3);
    if (showPinch && pI > 0.05) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      var pinchR = Math.max(G.anode_radius * 0.012, cr * G.cathode_radius * 0.12);
      var instAmp = (f.phase === "post_pinch") ? 0.3 : 0;
      var nModes = Math.min(4, Math.max(1,
        Math.round(0.25 * G.anode_length / (6.28 * Math.max(pinchR, 0.001)))));
      for (var k = 0; k <= pinch.N; k++) {
        var zf = k / pinch.N;
        var taper = Math.sin(Math.PI * zf);
        var lr = pinchR * (0.3 + 0.7 * taper);
        var ripple = instAmp * lr * Math.cos(6.28 * nModes * zf);
        pinch.radii[k] = Math.max(0.0005, lr + ripple);
      }
      BABYLON.MeshBuilder.CreateTube("pinchCore", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j] * 0.35; },
        tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, instance: pinch.core,
      });
      BABYLON.MeshBuilder.CreateTube("pinchMantle", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j]; },
        tessellation: 16, cap: BABYLON.Mesh.NO_CAP,
        sideOrientation: BABYLON.Mesh.DOUBLESIDE, instance: pinch.mantle,
      });
      pinch.coreMat.alpha = pI * 0.7;
      pinch.mantleMat.alpha = pI * 0.15;
      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(1 + (pI - 0.6) * 2, 0.95, 0.85);
      } else {
        pinch.coreMat.emissiveColor.set(0.6 + pI * 0.5, 0.5 + pI * 0.4, 0.3 + pI * 0.5);
      }
      pinch.mantleMat.emissiveColor.set(pI, pI * 0.35, pI * 0.08);
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // Beam cone (post-pinch)
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.1;
    beam.mat.alpha = beam.cone.isVisible ? pI * 0.3 : 0;

    // Anode thermal glow
    if (pI > 0.5) {
      device.copperMat.emissiveColor.set((pI - 0.5) * 0.25, (pI - 0.5) * 0.08, 0.01);
    } else {
      device.copperMat.emissiveColor.set(0, 0, 0);
    }

    // Pipeline tuning
    glowLayer.intensity = 0.3 + pI * 0.5;
    pipeline.bloomWeight = 0.15 + pI * 0.2;

    return { f: f, isP: isP, cr: cr, pI: pI, rippleAmp: 0 };
  }

  // --------------------------------------------------------
  // Return API object
  // --------------------------------------------------------
  return {
    engine: engine,
    scene: scene,
    camera: cam,
    gpuBackend: gpuBackend,
    useGPU: false,
    G: G,
    S: S,
    L: L,
    anode: device.anode,
    cathodeRods: device.cathodeRods,
    insulator: device.insulator,
    sheathDisk: sheathDisk,
    pinchCore: pinch.core,
    pinchMantle: pinch.mantle,
    beamCone: beam.cone,
    gasGlow: gasGlow.disk,
    bRings: [],
    arrows: [],
    fieldLines: [],
    ps: { start: function() {}, stop: function() {} },
    pipeline: pipeline,
    ssao: ssao,
    glowLayer: glowLayer,
    activeOverlay: activeOverlay,
    updateHeatmap: updateHeatmap,
    applyFrame: applyFrame,
    setOverlay: function(key) { activeOverlay = key; },
    setCmap: function(useCividis) {
      activeCmap = useCividis ? CIVIDIS : VIRIDIS;
      snapCache = {};
      lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
      buildSnapCache("density", L.density, snapCache);
      buildSnapCache("temperature", L.temperature, snapCache);
      buildSnapCache("bfield", L.bfield, snapCache);
      if (activeOverlay !== "none") updateHeatmap(activeOverlay);
    },
  };
}

window.createDPFScene = createDPFScene;
window.PHASE_LABELS = PHASE_LABELS;
window.PHASE_DESCRIPTIONS = PHASE_DESCRIPTIONS;
window.SPEEDS = SPEEDS;
