/**
 * DPF Bravo Schematic Renderer — Animated Physics Diagram
 *
 * Visual philosophy: high-quality animated textbook figure.
 * The device is the hero. Physics communicated through EXTERNAL indicators
 * (position ring, tip glow, pinch column, beam cone) — no internal meshes.
 * Think NOVA documentary animation or animated patent drawing.
 *
 * StandardMaterial only (no PBR, no CDN environment textures).
 * Under 800 lines.
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
// CONSTANTS & LABELS
// ============================================================

const PHASE_COLORS = {
  rundown:    [0.2, 0.5, 1.0],
  radial:     [1.0, 0.4, 0.08],
  mhd_radial: [1.0, 0.4, 0.08],
  reflected:  [1.0, 0.55, 0.0],
  pinch:      [1.0, 1.0, 0.9],
  post_pinch: [0.7, 0.15, 0.08],
};

const PHASE_LABELS = {
  rundown:     "Axial rundown",
  radial:      "Radial implosion",
  mhd_radial:  "Radial compression",
  mhd:         "MHD simulation",
  reflected:   "Reflected shock",
  pinch:       "Pinch — peak compression",
  post_pinch:  "Post-pinch disruption",
  none:        "",
};

const PHASE_DESCRIPTIONS = {
  rundown:     "Current sheath sweeps neutral gas from insulator to anode tip — magnetic snowplow",
  radial:      "Magnetic pressure compresses plasma ring inward toward the axis",
  mhd_radial:  "J x B force drives radial implosion — compression heating the plasma",
  mhd:         "Full MHD simulation of plasma dynamics",
  reflected:   "Reflected shock expands outward after axis convergence",
  pinch:       "PEAK COMPRESSION — fusion-relevant conditions at the axis",
  post_pinch:  "m=0 sausage instability breaks up the plasma column",
};

const SPEEDS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];

const GLOW_NAMES = new Set([
  "sheathRing", "pinchCore", "pinchMantle", "beamCone", "tipGlow"
]);

// ============================================================
// UTILITY
// ============================================================

function cmapLookup(v, cmap) {
  const t = Math.max(0, Math.min(1, v));
  const n = cmap.length - 1;
  const idx = t * n;
  const lo = Math.floor(idx), hi = Math.min(lo + 1, n);
  const f = idx - lo;
  return [
    cmap[lo][0] + (cmap[hi][0] - cmap[lo][0]) * f,
    cmap[lo][1] + (cmap[hi][1] - cmap[lo][1]) * f,
    cmap[lo][2] + (cmap[hi][2] - cmap[lo][2]) * f,
  ];
}

function b64ToFloat32(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(buf);
}

function isRadialPhase(phase) {
  return phase === "radial" || phase === "mhd_radial" ||
         phase === "pinch" || phase === "reflected" || phase === "post_pinch";
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

// ============================================================
// ENGINE INIT
// ============================================================

async function initEngine(canvas) {
  let engine, gpuBackend = "WebGL2";
  const params = new URLSearchParams(window.location.search);
  if (params.get("webgpu") === "1") {
    try {
      if (await BABYLON.WebGPUEngine.IsSupportedAsync) {
        engine = new BABYLON.WebGPUEngine(canvas, {
          antialias: true, adaptToDeviceRatio: true, powerPreference: "high-performance",
        });
        await engine.initAsync();
        gpuBackend = "WebGPU";
      }
    } catch (_) { /* fall through to WebGL */ }
  }
  if (!engine) {
    engine = new BABYLON.Engine(canvas, true, {
      stencil: true, adaptToDeviceRatio: true, preserveDrawingBuffer: true,
    });
  }
  engine.setHardwareScalingLevel(1 / window.devicePixelRatio);
  return { engine, gpuBackend };
}

// ============================================================
// CAMERA
// ============================================================

function createCamera(scene, canvas, G) {
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI / 4, Math.PI / 3, G.cathode_radius * 10,
    new BABYLON.Vector3(G.anode_length / 2, 0, 0), scene);
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

  let autoOrbit = true, userInteracting = false, timeout = null;
  canvas.addEventListener("pointerdown", function() {
    userInteracting = true; autoOrbit = false;
    if (timeout) clearTimeout(timeout);
  });
  canvas.addEventListener("pointerup", function() {
    userInteracting = false;
    timeout = setTimeout(function() { autoOrbit = true; }, 5000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !userInteracting) cam.alpha += 0.001;
  });
  return cam;
}

// ============================================================
// LIGHTS
// ============================================================

function createLights(scene) {
  const key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  key.intensity = 1.4;
  key.diffuse = new BABYLON.Color3(1, 0.98, 0.95);

  const back = new BABYLON.DirectionalLight("back", new BABYLON.Vector3(1, -1, -1), scene);
  back.intensity = 0.6;
  back.diffuse = new BABYLON.Color3(0.9, 0.92, 0.95);

  const fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.5;
  fill.diffuse = new BABYLON.Color3(0.9, 0.92, 1.0);
  fill.groundColor = new BABYLON.Color3(0.4, 0.4, 0.45);
}

// ============================================================
// ELECTRODES — solid device geometry (the hero)
// ============================================================

function buildDevice(scene, G) {
  // Anode — copper cylinder
  const copperMat = new BABYLON.StandardMaterial("copper", scene);
  copperMat.diffuseColor = new BABYLON.Color3(0.68, 0.45, 0.25);
  copperMat.specularColor = new BABYLON.Color3(0.8, 0.6, 0.3);
  copperMat.specularPower = 48;

  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 64, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.renderingGroupId = 0;

  // Cathode rods — stainless steel cage
  const steelMat = new BABYLON.StandardMaterial("steel", scene);
  steelMat.diffuseColor = new BABYLON.Color3(0.41, 0.41, 0.48);
  steelMat.specularColor = new BABYLON.Color3(0.5, 0.5, 0.55);
  steelMat.specularPower = 32;

  const N_RODS = G.n_cathode_rods || 8;
  const rodDiam = G.cathode_rod_diameter || G.cathode_radius * 0.06;
  const cathodeRods = [];
  for (let i = 0; i < N_RODS; i++) {
    const angle = (i / N_RODS) * Math.PI * 2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: rodDiam, height: G.anode_length * 1.05, tessellation: 12,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(
      G.anode_length / 2,
      G.cathode_radius * Math.sin(angle),
      G.cathode_radius * Math.cos(angle)
    );
    rod.material = steelMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  // Cathode end rings
  const ringThk = (G.cathode_radius - G.anode_radius) * 0.25;
  const baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 64,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.position.x = -G.anode_length * 0.025;
  baseRing.material = steelMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);

  const topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length * 1.03;
  cathodeRods.push(topRing);

  // Insulator
  const ceramicMat = new BABYLON.StandardMaterial("ceramic", scene);
  ceramicMat.diffuseColor = new BABYLON.Color3(0.95, 0.92, 0.85);
  ceramicMat.specularColor = new BABYLON.Color3(0.15, 0.15, 0.15);
  ceramicMat.specularPower = 8;
  const insThk = G.insulator_thickness || G.anode_radius * 0.15;
  const insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.3;
  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 64,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  return { anode, cathodeRods, insulator, copperMat, steelMat };
}

// ============================================================
// EXTERNAL INDICATORS — nothing inside the electrode cage
// ============================================================

function buildSheathRing(scene, G) {
  const ring = BABYLON.MeshBuilder.CreateTorus("sheathRing", {
    diameter: G.cathode_radius * 2.4,
    thickness: G.cathode_radius * 0.09,
    tessellation: 64,
  }, scene);
  ring.rotation.z = Math.PI / 2;
  ring.position.x = 0;
  const mat = new BABYLON.StandardMaterial("sheathRingMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.2, 0.5, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  ring.material = mat;
  ring.renderingGroupId = 1;
  return { ring, mat };
}

function buildPinchColumn(scene, G) {
  const N = 16;
  const columnLen = G.anode_length * 0.25;
  const tipX = G.anode_length;
  const path = [];
  for (let k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(tipX + columnLen * k / N, 0, 0));
  }
  const radii = new Array(N + 1).fill(G.anode_radius * 0.15);

  const coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(1, 0.95, 0.85);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  const core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path,
    radiusFunction: function(i) { return radii[i] * 0.3; },
    tessellation: 16, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;

  const mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(1, 0.4, 0.1);
  mantleMat.disableLighting = true;
  mantleMat.alpha = 0;
  mantleMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mantleMat.backFaceCulling = false;

  const mantle = BABYLON.MeshBuilder.CreateTube("pinchMantle", {
    path: path,
    radiusFunction: function(i) { return radii[i]; },
    tessellation: 20, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  mantle.material = mantleMat;
  mantle.renderingGroupId = 1;

  return { core, mantle, coreMat, mantleMat, radii, path, N };
}

function buildBeamCone(scene, G) {
  const cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.1,
    height: G.anode_length * 0.35, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length * 1.25 + G.anode_length * 0.175;
  const mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.6, 0.75, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone, mat };
}

function buildTipGlow(scene, G) {
  const disk = BABYLON.MeshBuilder.CreateDisc("tipGlow", {
    radius: G.anode_radius * 1.05, tessellation: 32,
  }, scene);
  disk.rotation.y = Math.PI / 2;
  disk.position.x = G.anode_length;
  const mat = new BABYLON.StandardMaterial("tipGlowMat", scene);
  mat.emissiveColor = new BABYLON.Color3(1, 0.6, 0.2);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  disk.material = mat;
  disk.renderingGroupId = 1;
  return { disk, mat };
}

// ============================================================
// GROUND GRID
// ============================================================

function buildGrid(scene, G) {
  const gridSize = Math.max(G.anode_length * 3, G.cathode_radius * 6);
  const ground = BABYLON.MeshBuilder.CreateGround("grid", {
    width: gridSize, height: gridSize, subdivisions: 1,
  }, scene);
  ground.position.y = -G.cathode_radius * 1.2;
  ground.position.x = G.anode_length / 2;
  const tex = new BABYLON.DynamicTexture("gridTex", 512, scene, false);
  const ctx = tex.getContext();
  ctx.fillStyle = "rgba(210, 215, 220, 1.0)";
  ctx.fillRect(0, 0, 512, 512);
  ctx.strokeStyle = "rgba(160, 165, 175, 0.6)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 20; i++) {
    const p = i * 512 / 20;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 512); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(512, p); ctx.stroke();
  }
  tex.update();
  const mat = new BABYLON.StandardMaterial("gridMat", scene);
  mat.diffuseTexture = tex;
  mat.specularColor = new BABYLON.Color3(0, 0, 0);
  mat.emissiveColor = new BABYLON.Color3(0.12, 0.14, 0.16);
  mat.alpha = 0.85;
  ground.material = mat;
}

// ============================================================
// HEATMAP OVERLAY (midplane ribbon for MHD field data)
// ============================================================

function buildHeatmapRibbon(scene, G) {
  const nr = 16, nz = 32;
  const paths = [];
  for (let ir = 0; ir <= nr; ir++) {
    const r = G.anode_radius + (G.cathode_radius - G.anode_radius) * ir / nr;
    const row = [];
    for (let iz = 0; iz <= nz; iz++) {
      const z = G.anode_length * iz / nz;
      const angle = Math.PI * 0.33;
      row.push(new BABYLON.Vector3(z, r * Math.sin(angle), r * Math.cos(angle)));
    }
    paths.push(row);
  }
  const plane = BABYLON.MeshBuilder.CreateRibbon("heatPlane", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false,
  }, scene);
  plane.isVisible = false;
  plane.isPickable = false;
  const mat = new BABYLON.StandardMaterial("heatMat", scene);
  mat.disableLighting = true;
  mat.backFaceCulling = false;
  plane.material = mat;
  return { plane, mat };
}

function buildSnapCache(fieldKey, layer, cache) {
  if (!layer || !layer.frames || !layer.frames.length) return;
  const shape = layer.frames_shape || layer.shape;
  if (!shape) return;
  const nr = shape[0], nz = shape[1];
  const n = layer.frames.length;
  const times = new Float64Array(n);
  const rgbaFrames = new Array(n);
  for (let fi = 0; fi < n; fi++) {
    times[fi] = layer.frames[fi].t_us;
    const vals = b64ToFloat32(layer.frames[fi].data);
    const rgba = new Uint8Array(nz * nr * 4);
    for (let ir = 0; ir < nr; ir++) {
      for (let iz = 0; iz < nz; iz++) {
        const v = vals[ir * nz + iz];
        const c = cmapLookup(v, activeCmap);
        const pi = ((nr - 1 - ir) * nz + iz) * 4;
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
  const e = cache[key];
  if (!e) return -1;
  let lo = 0, hi = e.times.length - 1;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (e.times[m] < t) lo = m + 1; else hi = m;
  }
  if (lo > 0 && Math.abs(e.times[lo - 1] - t) < Math.abs(e.times[lo] - t)) return lo - 1;
  return lo;
}

// ============================================================
// POST-PROCESSING PIPELINE
// ============================================================

function buildPipeline(scene, cam) {
  const pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.2;
  pipe.bloomThreshold = 0.85;
  pipe.bloomKernel = 64;
  pipe.bloomScale = 0.5;
  pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.0;
  pipe.sharpenEnabled = true;
  pipe.sharpen.edgeAmount = 0.15;

  let ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.6;
    ssao.radius = 1.5;
    ssao.samples = 16;
    ssao.base = 0.2;
  } catch (_) { /* SSAO not supported */ }

  const glow = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 32, mainTextureFixedSize: 512,
  });
  glow.intensity = 0.5;
  glow.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (GLOW_NAMES.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      const ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline: pipe, ssao: ssao, glowLayer: glow };
}

// ============================================================
// MAIN: createDPFScene(canvas, data)
// ============================================================

async function createDPFScene(canvas, data) {
  const L = data;
  const G = L.geometry;
  const S = L.sheath;

  const { engine, gpuBackend } = await initEngine(canvas);
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.12, 0.13, 0.15, 1);

  const camera = createCamera(scene, canvas, G);
  createLights(scene);

  const dev = buildDevice(scene, G);
  const sheath = buildSheathRing(scene, G);
  const pinch = buildPinchColumn(scene, G);
  const beam = buildBeamCone(scene, G);
  const tip = buildTipGlow(scene, G);
  buildGrid(scene, G);
  const heat = buildHeatmapRibbon(scene, G);
  const post = buildPipeline(scene, camera);

  // Snap cache for MHD field animation
  let snapCache = {};
  let lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
  let heatTex = null;

  buildSnapCache("density", L.density, snapCache);
  buildSnapCache("temperature", L.temperature, snapCache);
  buildSnapCache("bfield", L.bfield, snapCache);

  function applySnapTex(key) {
    const c = snapCache[key];
    if (!c) return;
    const idx = lastSnapIdx[key];
    if (idx < 0 || idx >= c.rgba.length) return;
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(
      c.rgba[idx], c.texW, c.texH,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene,
      false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE
    );
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.8;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  let activeOverlay = "none";

  function updateHeatmap(key) {
    if (!L || key === "none") { heat.plane.isVisible = false; return; }
    const layer = L[key];
    if (!layer || (!layer.data && !layer.frames)) { heat.plane.isVisible = false; return; }

    // Animated snap frames available
    if (snapCache[key] && lastSnapIdx[key] >= 0) { applySnapTex(key); return; }

    // Static single-frame fallback
    if (!layer.data || !layer.shape) { heat.plane.isVisible = false; return; }
    const vals = b64ToFloat32(layer.data);
    const nr = layer.shape[0], nz = layer.shape[1];
    const rgba = new Uint8Array(nz * nr * 4);
    for (let ir = 0; ir < nr; ir++) {
      for (let iz = 0; iz < nz; iz++) {
        const v = vals[ir * nz + iz];
        const c = cmapLookup(v, activeCmap);
        const pi = ((nr - 1 - ir) * nz + iz) * 4;
        rgba[pi]     = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 200;
      }
    }
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(
      rgba, nz, nr, BABYLON.Engine.TEXTUREFORMAT_RGBA,
      scene, false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE
    );
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.8;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  // ---- applyFrame: drives all indicators from sheath data ----

  function applyFrame(i) {
    if (i < 0 || i >= S.frames.length) return;
    const f = S.frames[i];
    const isP = isRadialPhase(f.phase);
    const cr = Math.max(0.02, f.r / G.cathode_radius);
    const Ifrac = Math.abs(f.I / Math.max(S.I_peak, 0.001));
    let pI = isP ? clamp01(Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;

    // Heatmap snap sync
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      const ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // --- SHEATH RING: slides along device exterior ---
    if (Ifrac > 0.02) {
      sheath.ring.isVisible = true;
      sheath.ring.position.x = isP ? G.anode_length : f.z;
      sheath.mat.alpha = clamp01(Ifrac * 0.5);
      const col = PHASE_COLORS[f.phase] || [0.3, 0.5, 1.0];
      sheath.mat.emissiveColor.set(col[0], col[1], col[2]);
      if (isP) {
        const scale = Math.max(0.15, cr);
        sheath.ring.scaling.y = scale;
        sheath.ring.scaling.z = scale;
      } else {
        sheath.ring.scaling.y = 1;
        sheath.ring.scaling.z = 1;
      }
    } else {
      sheath.ring.isVisible = false;
    }

    // --- TIP GLOW: anode face heats during compression ---
    if (isP && Ifrac > 0.1) {
      tip.disk.isVisible = true;
      tip.mat.alpha = clamp01(pI * 0.4);
      const h = clamp01(pI);
      tip.mat.emissiveColor.set(0.3 + h * 0.7, 0.2 + h * 0.4, 0.1 + h * 0.1);
    } else {
      tip.disk.isVisible = false;
    }

    // --- PINCH COLUMN: beyond anode tip, visible during compression ---
    const showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                       f.phase === "reflected") || (isP && cr < 0.3);
    if (showPinch && pI > 0.05) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      const pinchR = Math.max(G.anode_radius * 0.015, cr * G.cathode_radius * 0.15);
      const instAmp = (f.phase === "post_pinch") ? 0.3 : 0;
      const nModes = Math.min(4, Math.max(1,
        Math.round(0.25 * G.anode_length / (6.28 * Math.max(pinchR, 0.001)))));
      for (let k = 0; k <= pinch.N; k++) {
        const zf = k / pinch.N;
        const taper = Math.sin(Math.PI * zf);
        const lr = pinchR * (0.3 + 0.7 * taper);
        const ripple = instAmp * lr * Math.cos(6.28 * nModes * zf);
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

    // --- BEAM CONE: post-pinch only ---
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.1;
    beam.mat.alpha = beam.cone.isVisible ? pI * 0.3 : 0;

    // --- ANODE GLOW: subtle thermal emission at high compression ---
    if (pI > 0.5) {
      dev.copperMat.emissiveColor.set((pI - 0.5) * 0.3, (pI - 0.5) * 0.1, 0.01);
    } else {
      dev.copperMat.emissiveColor.set(0, 0, 0);
    }

    // --- POST-PROCESSING: bloom tracks plasma intensity ---
    post.glowLayer.intensity = 0.3 + pI * 0.5;
    post.pipeline.bloomWeight = 0.15 + pI * 0.2;

    return { f: f, isP: isP, cr: cr, pI: pI, rippleAmp: 0 };
  }

  // ---- Return the full interface contract ----

  return {
    engine: engine,
    scene: scene,
    camera: camera,
    gpuBackend: gpuBackend,
    useGPU: gpuBackend === "WebGPU",
    G: G,
    S: S,
    L: L,

    // Electrode meshes
    anode: dev.anode,
    cathodeRods: dev.cathodeRods,
    insulator: dev.insulator,

    // Plasma indicator meshes
    sheathDisk: sheath.ring,
    pinchCore: pinch.core,
    pinchMantle: pinch.mantle,
    beamCone: beam.cone,
    gasGlow: tip.disk,

    // Field viz (empty arrays — schematic style has no B-field lines)
    bRings: [],
    arrows: [],
    fieldLines: [],

    // Particle system stub
    ps: { start: function() {}, stop: function() {} },

    // Post-processing
    pipeline: post.pipeline,
    ssao: post.ssao,
    glowLayer: post.glowLayer,

    // API methods
    applyFrame: applyFrame,

    setOverlay: function(key) {
      activeOverlay = key;
    },

    setCmap: function(useCividis) {
      activeCmap = useCividis ? CIVIDIS : VIRIDIS;
      snapCache = {};
      lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
      buildSnapCache("density", L.density, snapCache);
      buildSnapCache("temperature", L.temperature, snapCache);
      buildSnapCache("bfield", L.bfield, snapCache);
      if (activeOverlay !== "none") updateHeatmap(activeOverlay);
    },

    updateHeatmap: updateHeatmap,
  };
}

// ============================================================
// EXPORTS
// ============================================================

window.createDPFScene = createDPFScene;
window.PHASE_LABELS = PHASE_LABELS;
window.PHASE_DESCRIPTIONS = PHASE_DESCRIPTIONS;
window.SPEEDS = SPEEDS;
