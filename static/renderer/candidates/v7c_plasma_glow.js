/**
 * DPF Renderer v7c -- "Plasma Glow"
 *
 * Plasma-dominant visualization. Big, bright, alive plasma effects with subtle
 * wireframe device (alpha=0.2). Sheath torus with Fresnel edge glow, HDR pinch,
 * ionization wake, vivid color progression (blue->orange->white). Babylon.js 8.x.
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
  rundown:    [0.15, 0.45, 1.0],
  radial:     [1.0, 0.55, 0.08],
  mhd_radial: [1.0, 0.50, 0.10],
  reflected:  [1.0, 0.60, 0.15],
  pinch:      [1.0, 0.98, 0.90],
  post_pinch: [0.9, 0.35, 0.08],
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
function smoothstep(lo, hi, x) {
  const t = clamp01((x - lo) / (hi - lo));
  return t * t * (3 - 2 * t);
}

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
// WIREFRAME DEVICE -- very subtle, alpha=0.2
// ============================================================

function buildDevice(scene, G) {
  const wireMat = new BABYLON.StandardMaterial("wireMat", scene);
  wireMat.emissiveColor = new BABYLON.Color3(0.15, 0.20, 0.30);
  wireMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  wireMat.specularColor = new BABYLON.Color3(0, 0, 0);
  wireMat.alpha = 0.20;
  wireMat.wireframe = true;
  wireMat.backFaceCulling = false;

  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = wireMat;
  anode.renderingGroupId = 0;

  const N_RODS = G.n_cathode_rods || 8;
  const rodDiam = G.cathode_rod_diameter || G.cathode_radius * 0.04;
  const rodMat = new BABYLON.StandardMaterial("rodMat", scene);
  rodMat.emissiveColor = new BABYLON.Color3(0.12, 0.16, 0.25);
  rodMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  rodMat.specularColor = new BABYLON.Color3(0, 0, 0);
  rodMat.alpha = 0.50;
  rodMat.wireframe = true;

  const cathodeRods = [];
  for (let i = 0; i < N_RODS; i++) {
    const angle = (i / N_RODS) * Math.PI * 2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: rodDiam, height: G.anode_length * 1.05, tessellation: 6,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(G.anode_length / 2,
      G.cathode_radius * Math.sin(angle),
      G.cathode_radius * Math.cos(angle));
    rod.material = rodMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  const ringThk = (G.cathode_radius - G.anode_radius) * 0.12;
  const ringMat = new BABYLON.StandardMaterial("ringMat", scene);
  ringMat.emissiveColor = new BABYLON.Color3(0.12, 0.16, 0.25);
  ringMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  ringMat.alpha = 0.30;
  ringMat.wireframe = true;

  const baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 32,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.position.x = 0;
  baseRing.material = ringMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);
  const topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length;
  cathodeRods.push(topRing);

  const insThk = G.anode_radius * 0.12;
  const insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.3;
  const insMat = new BABYLON.StandardMaterial("insMat", scene);
  insMat.emissiveColor = new BABYLON.Color3(0.25, 0.22, 0.15);
  insMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  insMat.alpha = 0.25;
  insMat.wireframe = true;

  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 32,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = insMat;
  insulator.renderingGroupId = 0;

  return { anode, cathodeRods, insulator, wireMat, rodMat };
}

// ============================================================
// SHEATH TORUS -- Fresnel edge glow, spans anode->cathode
// ============================================================

function buildSheathTorus(scene, G) {
  const midR = (G.anode_radius + G.cathode_radius) / 2;
  const tubeR = (G.cathode_radius - G.anode_radius) / 2;
  const torus = BABYLON.MeshBuilder.CreateTorus("sheathTorus", {
    diameter: midR * 2, thickness: tubeR * 2,
    tessellation: 48, sideOrientation: BABYLON.Mesh.DOUBLESIDE,
  }, scene);
  torus.rotation.z = Math.PI / 2;
  torus.position.x = 0;

  const mat = new BABYLON.StandardMaterial("sheathMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.15, 0.45, 1.0);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.specularColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  torus.material = mat;
  torus.renderingGroupId = 1;

  const fresnelParams = new BABYLON.FresnelParameters();
  fresnelParams.bias = 0.2;
  fresnelParams.power = 2;
  fresnelParams.leftColor = new BABYLON.Color3(1.0, 1.0, 1.0);
  fresnelParams.rightColor = new BABYLON.Color3(0.0, 0.0, 0.0);
  mat.emissiveFresnelParameters = fresnelParams;

  return { torus, mat, midR, tubeR, fresnelParams };
}

// ============================================================
// IONIZATION WAKE -- dim trail behind sheath
// ============================================================

function buildWake(scene, G) {
  const wake = BABYLON.MeshBuilder.CreateCylinder("ionWake", {
    diameter: (G.anode_radius + G.cathode_radius), height: G.anode_length,
    tessellation: 24, cap: BABYLON.Mesh.NO_CAP,
  }, scene);
  wake.rotation.z = Math.PI / 2;
  wake.position.x = G.anode_length / 2;
  const mat = new BABYLON.StandardMaterial("wakeMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.06, 0.12, 0.35);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  wake.material = mat;
  wake.renderingGroupId = 1;
  return { wake, mat };
}

// ============================================================
// PINCH COLUMN -- HDR emissive (>1.0), Bennett profile, m=0 ripple
// ============================================================

function buildPinchColumn(scene, G) {
  const N = 24;
  const columnLen = G.anode_length * 0.22;
  const tipX = G.anode_length;
  const path = [];
  for (let k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(tipX - columnLen * 0.1 + columnLen * 1.2 * k / N, 0, 0));
  }
  const radii = new Array(N + 1).fill(G.anode_radius * 0.10);

  const coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(2.0, 1.9, 1.7);
  coreMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  const core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path, radiusFunction: function(i) { return radii[i] * 0.25; },
    tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;

  const mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(1.4, 0.55, 0.08);
  mantleMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
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
// HALO -- hot glow sphere at pinch
// ============================================================

function buildHalo(scene, G) {
  const halo = BABYLON.MeshBuilder.CreateSphere("halo", {
    diameter: G.anode_radius * 0.9, segments: 16,
  }, scene);
  halo.position.x = G.anode_length;
  const mat = new BABYLON.StandardMaterial("haloMat", scene);
  mat.emissiveColor = new BABYLON.Color3(1.5, 0.7, 0.2);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  halo.material = mat;
  halo.renderingGroupId = 1;
  return { halo, mat };
}

// ============================================================
// BEAM CONE -- post-pinch particle beam
// ============================================================

function buildBeamCone(scene, G) {
  const cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.06,
    height: G.anode_length * 0.35, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length + G.anode_length * 0.225;
  const mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(1.2, 0.9, 0.5);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone, mat };
}

// ============================================================
// B-FIELD TORI -- 4-5 tori, brightness proportional to I/I_peak
// ============================================================

function buildBFieldTori(scene, G) {
  const bRings = [];
  const N_RINGS = 5;
  const mat = new BABYLON.StandardMaterial("bFieldMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.2, 0.5, 1.0);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;

  for (let i = 0; i < N_RINGS; i++) {
    const frac = (i + 1) / (N_RINGS + 1);
    const ring = BABYLON.MeshBuilder.CreateTorus("bRing" + i, {
      diameter: G.anode_radius * (0.5 + frac * 0.9),
      thickness: G.anode_radius * 0.008,
      tessellation: 32,
    }, scene);
    ring.rotation.z = Math.PI / 2;
    ring.position.x = G.anode_length * (0.65 + frac * 0.30);
    ring.material = mat;
    ring.renderingGroupId = 1;
    ring.isVisible = false;
    bRings.push(ring);
  }
  return { bRings, mat };
}

// ============================================================
// PARTICLES -- 3000, long lifetime, gentle drift
// ============================================================

function buildParticles(scene, G) {
  const emitter = new BABYLON.AbstractMesh("psEmitter", scene);
  emitter.position.x = G.anode_length;
  const ps = new BABYLON.ParticleSystem("sparks", 3000, scene);
  ps.emitter = emitter;
  ps.createSphereEmitter(G.anode_radius * 0.3);
  ps.color1 = new BABYLON.Color4(0.3, 0.65, 1.0, 0.85);
  ps.color2 = new BABYLON.Color4(0.6, 0.8, 1.0, 0.6);
  ps.colorDead = new BABYLON.Color4(0.15, 0.25, 0.6, 0);
  ps.minSize = G.cathode_radius * 0.003;
  ps.maxSize = G.cathode_radius * 0.012;
  ps.minLifeTime = 0.6;
  ps.maxLifeTime = 1.8;
  ps.emitRate = 0;
  ps.gravity = new BABYLON.Vector3(0, 0, 0);
  ps.minEmitPower = G.cathode_radius * 0.15;
  ps.maxEmitPower = G.cathode_radius * 0.8;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.start();
  return { ps, emitter };
}

// ============================================================
// HEATMAP -- full 360-degree cylindrical wrap
// ============================================================

function buildHeatmapCylinder(scene, G) {
  const midR = (G.anode_radius + G.cathode_radius) / 2;
  const nArc = 48, nZ = 32;
  const paths = [];
  for (let iz = 0; iz <= nZ; iz++) {
    const z = G.anode_length * iz / nZ;
    const ring = [];
    for (let ia = 0; ia <= nArc; ia++) {
      const angle = (ia / nArc) * Math.PI * 2;
      ring.push(new BABYLON.Vector3(z, midR * Math.sin(angle), midR * Math.cos(angle)));
    }
    paths.push(ring);
  }
  const cyl = BABYLON.MeshBuilder.CreateRibbon("heatCyl", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false,
  }, scene);
  cyl.isVisible = false;
  cyl.isPickable = false;
  const mat = new BABYLON.StandardMaterial("heatMat", scene);
  mat.disableLighting = true;
  mat.backFaceCulling = false;
  mat.alpha = 0.55;
  cyl.material = mat;
  cyl.renderingGroupId = 2;
  return { cyl, mat };
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
        rgba[pi + 3] = 153;
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
// PIPELINE -- generous bloom, glow
// ============================================================

function buildPipeline(scene, cam) {
  const pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.35;
  pipe.bloomThreshold = 0.55;
  pipe.bloomKernel = 80;
  pipe.bloomScale = 0.6;
  pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.1;

  const glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 64, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.6;
  const glowNames = new Set([
    "sheathTorus", "pinchCore", "pinchMantle", "beamCone", "ionWake", "halo",
  ]);
  glowLayer.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (glowNames.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      const ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else if (mesh.name && mesh.name.indexOf("bRing") === 0 && mesh.material) {
      const ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, (mesh.material.alpha || 0) * 0.6);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline: pipe, glowLayer };
}

// ============================================================
// MAIN SCENE
// ============================================================

async function createDPFScene(canvas, data) {
  const L = data, G = L.geometry, S = L.sheath;
  const { engine, gpuBackend } = await initEngine(canvas);
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.02, 0.02, 0.04, 1);
  scene.setRenderingAutoClearDepthStencil(1, true, true, false);
  scene.setRenderingAutoClearDepthStencil(2, false, false, false);

  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI * 0.25, Math.PI * 0.32, G.cathode_radius * 9,
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

  const fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.15;
  fill.diffuse = new BABYLON.Color3(0.4, 0.5, 0.7);
  fill.groundColor = new BABYLON.Color3(0.05, 0.05, 0.10);

  const dev = buildDevice(scene, G);
  const sheath = buildSheathTorus(scene, G);
  const wake = buildWake(scene, G);
  const pinch = buildPinchColumn(scene, G);
  const haloObj = buildHalo(scene, G);
  const beam = buildBeamCone(scene, G);
  const bField = buildBFieldTori(scene, G);
  const heat = buildHeatmapCylinder(scene, G);
  const parts = buildParticles(scene, G);
  const { pipeline, glowLayer } = buildPipeline(scene, cam);

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
    heat.mat.alpha = 0.55;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.cyl.isVisible = true;
  }

  let activeOverlay = "none";

  function updateHeatmap(ovKey) {
    if (!L || ovKey === "none") { heat.cyl.isVisible = false; return; }
    const layer = L[ovKey];
    if (!layer || (!layer.data && !layer.frames)) { heat.cyl.isVisible = false; return; }
    if (snapCache[ovKey] && lastSnapIdx[ovKey] >= 0) { applySnapTex(ovKey); return; }
    if (!layer.data || !layer.shape) { heat.cyl.isVisible = false; return; }
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
        rgba[pi + 3] = 153;
      }
    }
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(rgba, nz, nr, BABYLON.Engine.TEXTUREFORMAT_RGBA,
      scene, false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.55;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.cyl.isVisible = true;
  }

  // Time counter for pulsing
  let tAccum = 0;

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
    const col = PHASE_COLORS[f.phase] || [0.15, 0.45, 1.0];
    let rippleAmp = 0;
    tAccum = f.t || tAccum;

    // Heatmap snap sync
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      const ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // Pulse factor for "alive" plasma
    const pulse = 1.0 + 0.04 * Math.sin(tAccum * 18) * Ifrac;

    // === SHEATH TORUS with Fresnel ===
    if (Ifrac > 0.01) {
      sheath.torus.isVisible = true;
      sheath.torus.position.x = isP ? G.anode_length : f.z;

      // Alpha modulated by current
      const sheathAlpha = clamp01(Ifrac * 0.55 * pulse);
      sheath.mat.alpha = sheathAlpha;

      // Color progression: electric blue -> hot orange -> white
      if (isP) {
        const warmT = smoothstep(0.3, 0.8, pI);
        sheath.mat.emissiveColor.set(
          lerp(col[0], 1.0, warmT),
          lerp(col[1], 0.6, warmT),
          lerp(col[2], 0.15, warmT));
      } else {
        sheath.mat.emissiveColor.set(col[0], col[1], col[2]);
      }

      // Radial compression: scale torus
      if (isP) {
        const scaleF = Math.max(0.08, cr);
        sheath.torus.scaling.set(1, scaleF, scaleF);
      } else {
        sheath.torus.scaling.set(1, 1, 1);
      }

      // Update Fresnel for brightness
      sheath.fresnelParams.bias = lerp(0.2, 0.35, Ifrac);
      sheath.fresnelParams.power = lerp(2.0, 1.5, Ifrac);
    } else {
      sheath.torus.isVisible = false;
    }

    // === IONIZATION WAKE ===
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      wake.wake.isVisible = true;
      const extent = isP ? G.anode_length : f.z;
      wake.wake.scaling.x = extent / G.anode_length;
      wake.wake.position.x = extent / 2;
      wake.mat.alpha = clamp01(Ifrac * 0.04 * pulse);
      const wCol = isP
        ? [lerp(0.06, 0.3, pI), lerp(0.12, 0.15, pI), lerp(0.35, 0.06, pI)]
        : [0.06, 0.12, 0.35];
      wake.mat.emissiveColor.set(wCol[0], wCol[1], wCol[2]);
    } else {
      wake.wake.isVisible = false;
    }

    // === PINCH COLUMN -- Bennett profile with m=0 ripple ===
    const showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                       f.phase === "reflected") || (isP && cr < 0.35);
    if (showPinch && pI > 0.03) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      const pinchR = Math.max(G.anode_radius * 0.008, cr * G.cathode_radius * 0.10);
      rippleAmp = f.phase === "post_pinch" ? 0.40 : (f.phase === "pinch" ? 0.08 : 0);
      const waveNum = Math.min(5, Math.max(1, Math.round(
        0.25 * G.anode_length / (6.28 * Math.max(pinchR, 0.001)))));

      for (let k = 0; k <= pinch.N; k++) {
        const zf = k / pinch.N;
        // Bennett profile: peaked center, tapers at ends
        const bennett = 1.0 / (1.0 + Math.pow((zf - 0.5) / 0.35, 4));
        const lr = pinchR * (0.15 + 0.85 * bennett);
        const ripple = rippleAmp * lr * Math.cos(6.28 * waveNum * zf + tAccum * 4);
        pinch.radii[k] = Math.max(0.0002, lr + ripple);
      }

      BABYLON.MeshBuilder.CreateTube("pinchCore", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j] * 0.25; },
        tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, instance: pinch.core,
      });
      BABYLON.MeshBuilder.CreateTube("pinchMantle", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j]; },
        tessellation: 16, cap: BABYLON.Mesh.NO_CAP,
        sideOrientation: BABYLON.Mesh.DOUBLESIDE, instance: pinch.mantle,
      });

      // HDR emissive core (>1.0)
      pinch.coreMat.alpha = clamp01(pI * 0.90 * pulse);
      pinch.mantleMat.alpha = clamp01(pI * 0.30 * pulse);

      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(2.0, 1.9, 1.7);
      } else {
        pinch.coreMat.emissiveColor.set(
          lerp(0.4, 2.0, pI), lerp(0.6, 1.9, pI), lerp(1.0, 1.7, pI));
      }
      pinch.mantleMat.emissiveColor.set(
        lerp(0.5, 1.4, pI), lerp(0.3, 0.55, pI), lerp(0.8, 0.08, pI));
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // === HALO ===
    if (showPinch && pI > 0.1) {
      haloObj.halo.isVisible = true;
      const haloScale = 1.0 + pI * 2.0;
      haloObj.halo.scaling.set(haloScale, haloScale, haloScale);
      haloObj.mat.alpha = clamp01(pI * 0.15 * pulse);
      haloObj.mat.emissiveColor.set(
        lerp(0.8, 1.5, pI), lerp(0.4, 0.7, pI), lerp(0.1, 0.2, pI));
    } else {
      haloObj.halo.isVisible = false;
    }

    // === BEAM CONE ===
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.08;
    beam.mat.alpha = beam.cone.isVisible ? clamp01(pI * 0.35) : 0;

    // === B-FIELD TORI -- brightness proportional to I/I_peak ===
    const showBField = isP && Ifrac > 0.15;
    for (let bi = 0; bi < bField.bRings.length; bi++) {
      bField.bRings[bi].isVisible = showBField;
    }
    if (showBField) {
      bField.mat.alpha = clamp01(Ifrac * 0.12 * pulse);
      bField.mat.emissiveColor.set(
        lerp(0.1, 0.4, Ifrac), lerp(0.3, 0.6, Ifrac), lerp(0.8, 1.0, Ifrac));
      const drift = tAccum * 0.6;
      for (let bi = 0; bi < bField.bRings.length; bi++) {
        bField.bRings[bi].rotation.x = drift + bi * 0.5;
        // Compress during radial phase
        if (isP) {
          const bScale = Math.max(0.15, cr);
          bField.bRings[bi].scaling.set(1, bScale, bScale);
        } else {
          bField.bRings[bi].scaling.set(1, 1, 1);
        }
      }
    }

    // === PARTICLES -- follow sheath ===
    if (Ifrac > 0.05) {
      const rate = isP
        ? Math.round(lerp(200, 2000, pI))
        : Math.round(Ifrac * 400);
      parts.ps.emitRate = rate;
      if (isP) {
        parts.emitter.position.x = G.anode_length;
        parts.ps.createSphereEmitter(Math.max(0.001, cr * G.cathode_radius * 0.3));
      } else {
        parts.emitter.position.x = f.z;
        parts.ps.createSphereEmitter(G.anode_radius * 0.3);
      }
      // Color follows phase
      const pc = PHASE_COLORS[f.phase] || [0.15, 0.45, 1.0];
      parts.ps.color1 = new BABYLON.Color4(pc[0], pc[1], pc[2], 0.8);
      parts.ps.color2 = new BABYLON.Color4(
        pc[0] * 0.7 + 0.3, pc[1] * 0.7 + 0.3, pc[2] * 0.7 + 0.3, 0.5);
      parts.ps.colorDead = new BABYLON.Color4(pc[0] * 0.3, pc[1] * 0.3, pc[2] * 0.3, 0);
    } else {
      parts.ps.emitRate = 0;
    }

    // === PIPELINE -- generous bloom, ramps with pinch ===
    glowLayer.intensity = 0.45 + pI * 0.5;
    pipeline.bloomWeight = 0.25 + pI * 0.35;
    pipeline.bloomThreshold = lerp(0.55, 0.35, pI);

    return { f, isP, cr, pI, rippleAmp };
  }

  // ============================================================
  // RETURN API
  // ============================================================
  return {
    engine, scene, camera: cam, gpuBackend, useGPU: gpuBackend === "WebGPU",
    G, S, L,
    anode: dev.anode, cathodeRods: dev.cathodeRods, insulator: dev.insulator,
    sheathDisk: sheath.torus, pinchCore: pinch.core, pinchMantle: pinch.mantle,
    beamCone: beam.cone, gasGlow: wake.wake,
    bRings: bField.bRings, fieldLines: [],
    ps: { start: function() { parts.ps.start(); }, stop: function() { parts.ps.stop(); } },
    pipeline, glowLayer,
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
