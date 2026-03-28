/**
 * DPF X-Ray Renderer -- "Medical Imaging / CT Scan"
 *
 * Device rendered as faint gray wireframe outlines (alpha ~0.15) -- just enough
 * to orient the viewer. Plasma effects are vivid and dominant: bright cyan sheath,
 * white-hot pinch core, cool blue B-field tori. Like an X-ray where the bones
 * are ghostly and the pathology glows.
 *
 * Cool palette: blues, cyans, whites for plasma. Gray for structure.
 */

// ============================================================
// COLORMAPS & CONSTANTS
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

const PHASE_COLORS = {
  rundown:    [0.1, 0.6, 1.0],
  radial:     [0.0, 0.8, 1.0],
  mhd_radial: [0.0, 0.8, 1.0],
  reflected:  [0.3, 0.7, 1.0],
  pinch:      [0.9, 0.95, 1.0],
  post_pinch: [0.4, 0.6, 0.9],
};

const PHASE_LABELS = {
  rundown:    "Axial rundown",
  radial:     "Radial implosion",
  mhd_radial: "Radial compression (MHD)",
  mhd:        "MHD simulation",
  reflected:  "Reflected shock",
  pinch:      "Pinch -- peak compression",
  post_pinch: "Post-pinch disruption",
  none:       "",
};

const PHASE_DESCRIPTIONS = {
  rundown:    "Current sheath sweeps neutral gas from insulator to anode tip -- magnetic snowplow",
  radial:     "Magnetic pressure compresses plasma inward toward the axis",
  mhd_radial: "J x B force drives radial implosion -- compression heating",
  mhd:        "Full MHD simulation of plasma dynamics",
  reflected:  "Reflected shock expands outward after axis convergence",
  pinch:      "PEAK COMPRESSION -- fusion-relevant conditions at the axis",
  post_pinch: "m=0 sausage instability breaks up the plasma column",
};

const SPEEDS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];

let activeCmap = VIRIDIS;

// ============================================================
// UTILITIES
// ============================================================

function cmapLookup(v, cmap) {
  const t = Math.max(0, Math.min(1, v));
  const n = cmap.length - 1, idx = t * n;
  const lo = Math.floor(idx), hi = Math.min(lo + 1, n), f = idx - lo;
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

function isRadial(phase) {
  return phase === "radial" || phase === "mhd_radial" || phase === "pinch" ||
         phase === "reflected" || phase === "post_pinch";
}

function lerp(a, b, t) { return a + (b - a) * t; }
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
    } catch (_) {}
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
// X-RAY DEVICE -- faint gray wireframe, barely visible
// ============================================================

function buildXrayDevice(scene, G) {
  const ghostAlpha = 0.15;

  const anodeMat = new BABYLON.StandardMaterial("anodeXray", scene);
  anodeMat.diffuseColor = new BABYLON.Color3(0.4, 0.4, 0.45);
  anodeMat.emissiveColor = new BABYLON.Color3(0.06, 0.06, 0.08);
  anodeMat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.12);
  anodeMat.alpha = ghostAlpha;
  anodeMat.wireframe = true;

  const steelMat = new BABYLON.StandardMaterial("steelXray", scene);
  steelMat.diffuseColor = new BABYLON.Color3(0.35, 0.35, 0.4);
  steelMat.emissiveColor = new BABYLON.Color3(0.04, 0.04, 0.06);
  steelMat.specularColor = new BABYLON.Color3(0.08, 0.08, 0.1);
  steelMat.alpha = 0.2;

  const ceramicMat = new BABYLON.StandardMaterial("ceramicXray", scene);
  ceramicMat.diffuseColor = new BABYLON.Color3(0.5, 0.5, 0.55);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.04, 0.04, 0.05);
  ceramicMat.specularColor = new BABYLON.Color3(0.05, 0.05, 0.05);
  ceramicMat.alpha = ghostAlpha;
  ceramicMat.wireframe = true;

  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = anodeMat;
  anode.renderingGroupId = 0;

  const N_RODS = G.n_cathode_rods || 8;
  const cathodeRods = [];
  for (let i = 0; i < N_RODS; i++) {
    const angle = (i / N_RODS) * Math.PI * 2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: G.cathode_radius * 0.04, height: G.anode_length * 1.05, tessellation: 6,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(G.anode_length / 2,
      G.cathode_radius * Math.sin(angle), G.cathode_radius * Math.cos(angle));
    rod.material = steelMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  const ringThk = G.cathode_radius * 0.03;
  const baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 48,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.material = steelMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);
  const topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length;
  cathodeRods.push(topRing);

  const insThk = G.anode_radius * 0.15;
  const insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.3;
  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 48,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  return { anode, cathodeRods, insulator, anodeMat, steelMat, ceramicMat };
}

// ============================================================
// GROUND GRID -- faint reference plane
// ============================================================

function buildGroundGrid(scene, G) {
  const grid = BABYLON.MeshBuilder.CreateGround("grid", {
    width: G.cathode_radius * 12, height: G.cathode_radius * 12,
    subdivisions: 24,
  }, scene);
  grid.position.set(G.anode_length / 2, -G.cathode_radius * 1.5, 0);
  const mat = new BABYLON.StandardMaterial("gridMat", scene);
  mat.diffuseColor = new BABYLON.Color3(0.08, 0.08, 0.1);
  mat.emissiveColor = new BABYLON.Color3(0.02, 0.02, 0.04);
  mat.specularColor = BABYLON.Color3.Black();
  mat.alpha = 0.2;
  mat.wireframe = true;
  mat.backFaceCulling = false;
  grid.material = mat;
  grid.renderingGroupId = 0;
  return grid;
}

// ============================================================
// SHEATH TORUS -- bright cyan ring sweeping along z
// ============================================================

function buildSheathDisk(scene, G) {
  const paths = [];
  const nRadial = 12;
  for (let ir = 0; ir <= nRadial; ir++) {
    const r = G.anode_radius + (G.cathode_radius - G.anode_radius) * ir / nRadial;
    const ring = [];
    const nSeg = 48;
    for (let j = 0; j <= nSeg; j++) {
      const angle = (j / nSeg) * Math.PI * 2;
      ring.push(new BABYLON.Vector3(0, r * Math.sin(angle), r * Math.cos(angle)));
    }
    paths.push(ring);
  }
  const disk = BABYLON.MeshBuilder.CreateRibbon("sheathDisk", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);

  const mat = new BABYLON.StandardMaterial("sheathMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.0, 0.7, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  disk.material = mat;
  disk.renderingGroupId = 1;

  return { disk, mat, paths, nRadial };
}

// ============================================================
// PLASMA TRAIL -- cyan glow behind sheath
// ============================================================

function buildGasGlow(scene, G) {
  const glow = BABYLON.MeshBuilder.CreateCylinder("gasGlow", {
    diameter: (G.anode_radius + G.cathode_radius), height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.NO_CAP,
  }, scene);
  glow.rotation.z = Math.PI / 2;
  glow.position.x = G.anode_length / 2;

  const mat = new BABYLON.StandardMaterial("gasGlowMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.0, 0.15, 0.3);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  glow.material = mat;
  glow.renderingGroupId = 1;

  return { glow, mat };
}

// ============================================================
// PINCH COLUMN -- vivid white core, cyan mantle
// ============================================================

function buildPinchColumn(scene, G) {
  const N = 20;
  const columnLen = G.anode_length * 0.3;
  const tipX = G.anode_length;
  const path = [];
  for (let k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(tipX - columnLen * 0.1 + columnLen * k / N, 0, 0));
  }
  const radii = new Array(N + 1).fill(G.anode_radius * 0.12);

  const coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(1.0, 1.0, 1.0);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  const core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path, radiusFunction: function(i) { return radii[i] * 0.3; },
    tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;

  const mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(0.0, 0.6, 0.9);
  mantleMat.disableLighting = true;
  mantleMat.alpha = 0;
  mantleMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mantleMat.backFaceCulling = false;

  const mantle = BABYLON.MeshBuilder.CreateTube("pinchMantle", {
    path: path, radiusFunction: function(i) { return radii[i]; },
    tessellation: 16, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  mantle.material = mantleMat;
  mantle.renderingGroupId = 1;

  return { core, mantle, coreMat, mantleMat, radii, path, N };
}

// ============================================================
// BEAM CONE -- post-pinch particle emission
// ============================================================

function buildBeamCone(scene, G) {
  const cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.08,
    height: G.anode_length * 0.35, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length + G.anode_length * 0.175 + G.anode_length * 0.05;
  const mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.3, 0.6, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone, mat };
}

// ============================================================
// B-FIELD TORI -- faint blue rings showing toroidal field
// ============================================================

function buildBFieldRings(scene, G) {
  const bRings = [];
  const nRings = 5;
  const ringMat = new BABYLON.StandardMaterial("bRingMat", scene);
  ringMat.emissiveColor = new BABYLON.Color3(0.08, 0.15, 0.4);
  ringMat.disableLighting = true;
  ringMat.alpha = 0;
  ringMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  ringMat.backFaceCulling = false;
  ringMat.wireframe = true;

  for (let i = 0; i < nRings; i++) {
    const frac = (i + 1) / (nRings + 1);
    const ring = BABYLON.MeshBuilder.CreateTorus("bRing" + i, {
      diameter: G.anode_radius * 1.2 * (0.4 + 0.6 * frac),
      thickness: G.anode_radius * 0.02,
      tessellation: 32,
    }, scene);
    ring.rotation.z = Math.PI / 2;
    ring.position.x = G.anode_length * frac;
    ring.material = ringMat;
    ring.renderingGroupId = 1;
    ring.isVisible = false;
    bRings.push(ring);
  }
  return { bRings, ringMat };
}

// ============================================================
// PARTICLES -- white/cyan sparks, 3000 capacity
// ============================================================

function buildParticles(scene, G) {
  const emitter = new BABYLON.AbstractMesh("psEmitter", scene);
  emitter.position.x = G.anode_length;
  const ps = new BABYLON.ParticleSystem("sparks", 3000, scene);
  ps.emitter = emitter;
  ps.createSphereEmitter(G.anode_radius * 0.3);
  ps.color1 = new BABYLON.Color4(1.0, 1.0, 1.0, 0.9);
  ps.color2 = new BABYLON.Color4(0.3, 0.8, 1.0, 0.7);
  ps.colorDead = new BABYLON.Color4(0.05, 0.1, 0.2, 0);
  ps.minSize = G.cathode_radius * 0.004;
  ps.maxSize = G.cathode_radius * 0.012;
  ps.minLifeTime = 0.12;
  ps.maxLifeTime = 0.45;
  ps.emitRate = 0;
  ps.gravity = new BABYLON.Vector3(0, 0, 0);
  ps.minEmitPower = G.cathode_radius * 0.5;
  ps.maxEmitPower = G.cathode_radius * 2.5;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.start();
  return { ps, emitter };
}

// ============================================================
// HEATMAP OVERLAY
// ============================================================

function buildHeatmapPlane(scene, G) {
  const paths = [];
  const nr = 16, nz = 32;
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
  plane.renderingGroupId = 2;
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
  cache[fieldKey] = { times, rgba: rgbaFrames, texW: nz, texH: nr };
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
// PIPELINE -- bloom, glow, SSAO, FXAA
// ============================================================

function buildPipeline(scene, cam) {
  const pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.25;
  pipe.bloomThreshold = 0.6;
  pipe.bloomKernel = 64;
  pipe.bloomScale = 0.5;
  pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.0;

  let ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.3;
    ssao.radius = 1.5;
    ssao.samples = 12;
    ssao.base = 0.1;
  } catch (_) {}

  const glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 64, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.4;
  const glowNames = new Set([
    "sheathDisk", "pinchCore", "pinchMantle", "beamCone", "gasGlow",
    "bRing0", "bRing1", "bRing2", "bRing3", "bRing4",
  ]);
  glowLayer.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (glowNames.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      const ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline: pipe, ssao, glowLayer };
}

// ============================================================
// FIELD LINES -- dim helical lines along anode
// ============================================================

function buildFieldLines(scene, G) {
  const fieldLines = [];
  const nLines = 4;
  const lineMat = new BABYLON.StandardMaterial("fLineMat", scene);
  lineMat.emissiveColor = new BABYLON.Color3(0.05, 0.1, 0.25);
  lineMat.disableLighting = true;
  lineMat.alpha = 0;
  lineMat.alphaMode = BABYLON.Engine.ALPHA_ADD;

  for (let li = 0; li < nLines; li++) {
    const pts = [];
    const baseAngle = (li / nLines) * Math.PI * 2;
    const r = G.anode_radius * 1.3;
    for (let k = 0; k <= 60; k++) {
      const frac = k / 60;
      const angle = baseAngle + frac * Math.PI * 4;
      pts.push(new BABYLON.Vector3(
        G.anode_length * frac,
        r * Math.sin(angle),
        r * Math.cos(angle)
      ));
    }
    const line = BABYLON.MeshBuilder.CreateLines("fLine" + li, {
      points: pts, updatable: false,
    }, scene);
    line.color = new BABYLON.Color3(0.1, 0.2, 0.5);
    line.alpha = 0;
    line.renderingGroupId = 1;
    line.isVisible = false;
    fieldLines.push(line);
  }
  return fieldLines;
}

// ============================================================
// MAIN SCENE
// ============================================================

async function createDPFScene(canvas, data) {
  const L = data, G = L.geometry, S = L.sheath;
  const { engine, gpuBackend } = await initEngine(canvas);
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.04, 0.04, 0.06, 1);
  scene.setRenderingAutoClearDepthStencil(1, true, true, false);
  scene.setRenderingAutoClearDepthStencil(2, false, false, false);

  // Camera
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI * 0.3, Math.PI * 0.35, G.cathode_radius * 10,
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
  cam.minZ = 0.0005;
  cam.inertia = 0.88;

  // Auto-orbit
  let autoOrbit = true, interacting = false, orbitTimeout = null;
  canvas.addEventListener("pointerdown", function() {
    interacting = true; autoOrbit = false;
    if (orbitTimeout) clearTimeout(orbitTimeout);
  });
  canvas.addEventListener("pointerup", function() {
    interacting = false;
    orbitTimeout = setTimeout(function() { autoOrbit = true; }, 5000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !interacting) cam.alpha += 0.0006;
  });

  // Lights -- very dim, let emissives dominate the X-ray look
  const key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  key.intensity = 0.4;
  key.diffuse = new BABYLON.Color3(0.8, 0.85, 0.95);
  const fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.2;
  fill.diffuse = new BABYLON.Color3(0.7, 0.75, 0.85);
  fill.groundColor = new BABYLON.Color3(0.1, 0.1, 0.15);

  // Build all elements
  const dev = buildXrayDevice(scene, G);
  buildGroundGrid(scene, G);
  const sheath = buildSheathDisk(scene, G);
  const gas = buildGasGlow(scene, G);
  const pinch = buildPinchColumn(scene, G);
  const beam = buildBeamCone(scene, G);
  const bfield = buildBFieldRings(scene, G);
  const fieldLines = buildFieldLines(scene, G);
  const heat = buildHeatmapPlane(scene, G);
  const parts = buildParticles(scene, G);
  const { pipeline, ssao, glowLayer } = buildPipeline(scene, cam);

  // Snap cache for MHD heatmaps
  let snapCache = {};
  let lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
  let heatTex = null;
  buildSnapCache("density", L.density, snapCache);
  buildSnapCache("temperature", L.temperature, snapCache);
  buildSnapCache("bfield", L.bfield, snapCache);

  function applySnapTex(ovKey) {
    const c = snapCache[ovKey];
    if (!c) return;
    const idx = lastSnapIdx[ovKey];
    if (idx < 0 || idx >= c.rgba.length) return;
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(c.rgba[idx], c.texW, c.texH,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene, false, false,
      BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.7;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  let activeOverlay = "none";

  function updateHeatmap(ovKey) {
    if (!L || ovKey === "none") { heat.plane.isVisible = false; return; }
    const layer = L[ovKey];
    if (!layer || (!layer.data && !layer.frames)) { heat.plane.isVisible = false; return; }
    if (snapCache[ovKey] && lastSnapIdx[ovKey] >= 0) { applySnapTex(ovKey); return; }
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
    heatTex = new BABYLON.RawTexture(rgba, nz, nr, BABYLON.Engine.TEXTUREFORMAT_RGBA,
      scene, false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.7;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  // ============================================================
  // applyFrame(i)
  // ============================================================
  function applyFrame(i) {
    if (i < 0 || i >= S.frames.length) return;
    const f = S.frames[i];
    const isP = isRadial(f.phase);
    const cr = Math.max(0.02, f.r / G.cathode_radius);
    const Ifrac = clamp01(Math.abs(f.I / Math.max(S.I_peak, 0.001)));
    let pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;
    const col = PHASE_COLORS[f.phase] || [0.1, 0.6, 1.0];
    let rippleAmp = 0;

    // Heatmap snap sync
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      const ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // === SHEATH TORUS -- bright cyan ring ===
    if (Ifrac > 0.01) {
      sheath.disk.isVisible = true;
      sheath.disk.position.x = isP ? G.anode_length : f.z;
      sheath.mat.alpha = clamp01(Ifrac * 0.6);
      sheath.mat.emissiveColor.set(col[0], col[1], col[2]);

      if (isP) {
        const scaleF = Math.max(0.05, cr);
        sheath.disk.scaling.y = scaleF;
        sheath.disk.scaling.z = scaleF;
      } else {
        sheath.disk.scaling.y = 1;
        sheath.disk.scaling.z = 1;
      }
    } else {
      sheath.disk.isVisible = false;
    }

    // === PLASMA TRAIL -- cyan glow behind sheath ===
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      gas.glow.isVisible = true;
      const extent = isP ? G.anode_length : f.z;
      gas.glow.scaling.x = extent / G.anode_length;
      gas.glow.position.x = extent / 2;
      gas.mat.alpha = clamp01(Ifrac * 0.06);
      gas.mat.emissiveColor.set(col[0] * 0.2, col[1] * 0.3, col[2] * 0.5);
    } else {
      gas.glow.isVisible = false;
    }

    // === PINCH COLUMN -- white core, cyan mantle ===
    const showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                       f.phase === "reflected") || (isP && cr < 0.35);
    if (showPinch && pI > 0.03) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      const pinchR = Math.max(G.anode_radius * 0.01, cr * G.cathode_radius * 0.12);
      const instAmp = f.phase === "post_pinch" ? 0.35 : 0;
      rippleAmp = instAmp;
      const waveNum = Math.min(5, Math.max(1, Math.round(
        0.25 * G.anode_length / (6.28 * Math.max(pinchR, 0.001)))));

      for (let k = 0; k <= pinch.N; k++) {
        const zf = k / pinch.N;
        const taper = Math.sin(Math.PI * zf);
        const lr = pinchR * (0.25 + 0.75 * taper);
        const ripple = instAmp * lr * Math.cos(6.28 * waveNum * zf + f.t * 3);
        pinch.radii[k] = Math.max(0.0003, lr + ripple);
      }

      BABYLON.MeshBuilder.CreateTube("pinchCore", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j] * 0.3; },
        tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, instance: pinch.core,
      });
      BABYLON.MeshBuilder.CreateTube("pinchMantle", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j]; },
        tessellation: 16, cap: BABYLON.Mesh.NO_CAP,
        sideOrientation: BABYLON.Mesh.DOUBLESIDE, instance: pinch.mantle,
      });

      pinch.coreMat.alpha = clamp01(pI * 0.9);
      pinch.mantleMat.alpha = clamp01(pI * 0.25);

      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(1.0, 1.0, 1.0);
      } else {
        pinch.coreMat.emissiveColor.set(0.6 + pI * 0.4, 0.7 + pI * 0.3, 0.8 + pI * 0.2);
      }
      pinch.mantleMat.emissiveColor.set(
        lerp(0.0, 0.1, pI), lerp(0.4, 0.7, pI), lerp(0.7, 1.0, pI));
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // === BEAM CONE ===
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.08;
    beam.mat.alpha = beam.cone.isVisible ? clamp01(pI * 0.4) : 0;

    // === B-FIELD TORI -- subtle blue rings ===
    const showB = Ifrac > 0.05;
    for (let bi = 0; bi < bfield.bRings.length; bi++) {
      bfield.bRings[bi].isVisible = showB;
    }
    if (showB) {
      bfield.ringMat.alpha = clamp01(Ifrac * 0.12);
      bfield.ringMat.emissiveColor.set(0.06, 0.12, 0.35);
    }

    // === FIELD LINES -- dim helices ===
    const showFL = Ifrac > 0.1;
    for (let fi = 0; fi < fieldLines.length; fi++) {
      fieldLines[fi].isVisible = showFL;
      fieldLines[fi].alpha = showFL ? clamp01(Ifrac * 0.08) : 0;
    }

    // === PARTICLES -- white/cyan sparks near pinch ===
    if (showPinch && pI > 0.2) {
      parts.ps.emitRate = Math.round(pI * 1200);
      parts.emitter.position.x = G.anode_length;
    } else if (Ifrac > 0.3) {
      parts.ps.emitRate = Math.round(Ifrac * 200);
      parts.emitter.position.x = isP ? G.anode_length : f.z;
    } else {
      parts.ps.emitRate = 0;
    }

    // === DEVICE GLOW -- faint Fresnel on anode under compression ===
    if (pI > 0.3) {
      dev.anodeMat.emissiveColor.set(
        0.06 + (pI - 0.3) * 0.15, 0.06 + (pI - 0.3) * 0.2, 0.08 + (pI - 0.3) * 0.3);
    } else {
      dev.anodeMat.emissiveColor.set(0.06, 0.06, 0.08);
    }

    // === PIPELINE adjustments ===
    glowLayer.intensity = 0.3 + pI * 0.4;
    pipeline.bloomWeight = 0.2 + pI * 0.2;

    return { f, isP, cr, pI, rippleAmp };
  }

  // ============================================================
  // RETURN API
  // ============================================================
  return {
    engine, scene, camera: cam, gpuBackend, useGPU: gpuBackend === "WebGPU",
    G, S, L,
    anode: dev.anode, cathodeRods: dev.cathodeRods, insulator: dev.insulator,
    sheathDisk: sheath.disk, pinchCore: pinch.core, pinchMantle: pinch.mantle,
    beamCone: beam.cone, gasGlow: gas.glow,
    bRings: bfield.bRings, fieldLines: fieldLines,
    ps: { start: function() { parts.ps.start(); }, stop: function() { parts.ps.stop(); } },
    pipeline, ssao, glowLayer,
    applyFrame: applyFrame,
    updateHeatmap: updateHeatmap,
    activeOverlay: activeOverlay,
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
