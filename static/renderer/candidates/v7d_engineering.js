/**
 * DPF Renderer v7d -- "Engineering CAD"
 *
 * Professional CAD-style visualization with wireframe device geometry,
 * dimensionally accurate proportions (PF-1000 scale), and muted technical
 * color palette. Sheath is a proper torus spanning anode-cathode gap.
 * Heatmap wraps full 360-degree cylinder. Clean, precise, SolidWorks feel.
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
  rundown:    [0.35, 0.55, 0.85],
  radial:     [0.85, 0.50, 0.20],
  mhd_radial: [0.85, 0.50, 0.20],
  reflected:  [0.80, 0.55, 0.25],
  pinch:      [0.95, 0.92, 0.85],
  post_pinch: [0.65, 0.35, 0.18],
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

function isRadialPhase(phase) {
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
// WIREFRAME DEVICE -- precise CAD geometry
// ============================================================

function buildDevice(scene, G) {
  const copperMat = new BABYLON.StandardMaterial("copper", scene);
  copperMat.diffuseColor = new BABYLON.Color3(0.62, 0.42, 0.22);
  copperMat.specularColor = new BABYLON.Color3(0.70, 0.55, 0.30);
  copperMat.emissiveColor = new BABYLON.Color3(0.04, 0.02, 0.01);
  copperMat.specularPower = 40;
  copperMat.wireframe = true;
  copperMat.alpha = 0.35;

  const steelMat = new BABYLON.StandardMaterial("steel", scene);
  steelMat.diffuseColor = new BABYLON.Color3(0.45, 0.45, 0.50);
  steelMat.specularColor = new BABYLON.Color3(0.50, 0.50, 0.55);
  steelMat.emissiveColor = new BABYLON.Color3(0.03, 0.03, 0.04);
  steelMat.specularPower = 28;
  steelMat.alpha = 0.50;

  const ceramicMat = new BABYLON.StandardMaterial("ceramic", scene);
  ceramicMat.diffuseColor = new BABYLON.Color3(0.88, 0.84, 0.72);
  ceramicMat.specularColor = new BABYLON.Color3(0.12, 0.10, 0.08);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.03, 0.02, 0.01);
  ceramicMat.specularPower = 6;
  ceramicMat.alpha = 0.40;

  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 48, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.renderingGroupId = 0;

  const N_RODS = 8;
  const rodDiam = G.cathode_radius * 0.04;
  const cathodeRods = [];
  for (let i = 0; i < N_RODS; i++) {
    const angle = (i / N_RODS) * Math.PI * 2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: rodDiam, height: G.anode_length * 1.05, tessellation: 8,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(G.anode_length / 2,
      G.cathode_radius * Math.sin(angle),
      G.cathode_radius * Math.cos(angle));
    rod.material = steelMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  const ringThk = (G.cathode_radius - G.anode_radius) * 0.14;
  const baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 48,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.position.x = 0;
  baseRing.material = steelMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);
  const topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length;
  cathodeRods.push(topRing);

  const insThk = G.anode_radius * 0.12;
  const insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.95;
  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 48,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  return { anode, cathodeRods, insulator, copperMat, steelMat, ceramicMat };
}

// ============================================================
// GROUND GRID -- technical drawing floor
// ============================================================

function buildGroundGrid(scene, G) {
  const N = 18;
  const span = G.cathode_radius * 10;
  const yPos = -G.cathode_radius * 1.9;
  const gridColor = new BABYLON.Color3(0.16, 0.17, 0.20);
  for (let i = -N; i <= N; i++) {
    const pos = (i / N) * span / 2;
    const h = BABYLON.MeshBuilder.CreateLines("gl" + i, {
      points: [
        new BABYLON.Vector3(G.anode_length / 2 - span / 2, yPos, pos),
        new BABYLON.Vector3(G.anode_length / 2 + span / 2, yPos, pos),
      ],
    }, scene);
    h.color = gridColor;
    h.alpha = 0.18;
    const v = BABYLON.MeshBuilder.CreateLines("gv" + i, {
      points: [
        new BABYLON.Vector3(pos + G.anode_length / 2, yPos, -span / 2),
        new BABYLON.Vector3(pos + G.anode_length / 2, yPos, span / 2),
      ],
    }, scene);
    v.color = gridColor;
    v.alpha = 0.18;
  }
}

// ============================================================
// SHEATH TORUS -- exact gap-spanning ring
// ============================================================

function buildSheathTorus(scene, G) {
  const midDiam = G.anode_radius + G.cathode_radius;
  const gapWidth = G.cathode_radius - G.anode_radius;
  const torus = BABYLON.MeshBuilder.CreateTorus("sheathDisk", {
    diameter: midDiam, thickness: gapWidth, tessellation: 48,
  }, scene);
  torus.rotation.z = Math.PI / 2;
  torus.position.x = 0;

  const mat = new BABYLON.StandardMaterial("sheathMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.35, 0.55, 0.85);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;

  mat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  mat.emissiveFresnelParameters.bias = 0.3;
  mat.emissiveFresnelParameters.power = 2;
  mat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(0.45, 0.65, 0.95);
  mat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.15, 0.25, 0.45);

  torus.material = mat;
  torus.renderingGroupId = 1;
  return { torus, mat };
}

// ============================================================
// GAS GLOW -- faint plasma trail behind sheath
// ============================================================

function buildGasGlow(scene, G) {
  const glow = BABYLON.MeshBuilder.CreateCylinder("gasGlow", {
    diameter: (G.anode_radius + G.cathode_radius), height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.NO_CAP,
  }, scene);
  glow.rotation.z = Math.PI / 2;
  glow.position.x = G.anode_length / 2;
  const mat = new BABYLON.StandardMaterial("gasGlowMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.15, 0.20, 0.30);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  glow.material = mat;
  glow.renderingGroupId = 1;
  return { glow, mat };
}

// ============================================================
// PINCH COLUMN -- dual tube (bright core + dim mantle)
// ============================================================

function buildPinchColumn(scene, G) {
  const N = 24;
  const startX = G.anode_length * 0.85;
  const endX = G.anode_length * 1.2;
  const columnLen = endX - startX;
  const path = [];
  for (let k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(startX + columnLen * k / N, 0, 0));
  }
  const radii = new Array(N + 1).fill(G.anode_radius * 0.08);

  const coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(0.95, 0.92, 0.85);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  const core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path, radiusFunction: function(i) { return radii[i] * 0.35; },
    tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;

  const mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(0.75, 0.40, 0.15);
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
// BEAM CONE -- post-pinch particle beam
// ============================================================

function buildBeamCone(scene, G) {
  const cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.06,
    height: G.anode_length * 0.40, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length * 1.2 + G.anode_length * 0.25;
  const mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.80, 0.70, 0.50);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone, mat };
}

// ============================================================
// B-FIELD TORI -- thin indicator rings behind sheath
// ============================================================

function buildBFieldTori(scene, G) {
  const bRings = [];
  const fracs = [0.25, 0.45, 0.65, 0.82];
  const midDiam = G.anode_radius + G.cathode_radius;
  const ringThk = G.cathode_radius * 0.015;
  const mat = new BABYLON.StandardMaterial("bFieldMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.30, 0.45, 0.65);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;

  for (let i = 0; i < fracs.length; i++) {
    const ring = BABYLON.MeshBuilder.CreateTorus("bRing" + i, {
      diameter: midDiam, thickness: ringThk, tessellation: 32,
    }, scene);
    ring.rotation.z = Math.PI / 2;
    ring.position.x = fracs[i] * G.anode_length;
    ring.material = mat;
    ring.renderingGroupId = 1;
    ring.isVisible = false;
    bRings.push(ring);
  }
  return { bRings, mat, fracs };
}

// ============================================================
// PARTICLES -- 3000, sphere emitter, additive
// ============================================================

function buildParticles(scene, G) {
  const emitter = new BABYLON.AbstractMesh("psEmitter", scene);
  emitter.position.x = G.anode_length;
  const ps = new BABYLON.ParticleSystem("sparks", 3000, scene);
  ps.emitter = emitter;
  ps.createSphereEmitter(G.anode_radius * 0.20);
  ps.color1 = new BABYLON.Color4(0.80, 0.75, 0.55, 0.8);
  ps.color2 = new BABYLON.Color4(0.70, 0.50, 0.30, 0.6);
  ps.colorDead = new BABYLON.Color4(0.25, 0.18, 0.08, 0);
  ps.minSize = G.cathode_radius * 0.003;
  ps.maxSize = G.cathode_radius * 0.012;
  ps.minLifeTime = 0.10;
  ps.maxLifeTime = 0.40;
  ps.emitRate = 0;
  ps.gravity = new BABYLON.Vector3(0, 0, 0);
  ps.minEmitPower = G.cathode_radius * 0.3;
  ps.maxEmitPower = G.cathode_radius * 1.5;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.start();
  return { ps, emitter };
}

// ============================================================
// HEATMAP -- 360-degree cylindrical wrap
// ============================================================

function buildHeatmapCylinder(scene, G) {
  const nTheta = 48, nZ = 32;
  const midR = (G.anode_radius + G.cathode_radius) / 2;
  const paths = [];
  for (let it = 0; it <= nTheta; it++) {
    const angle = (it / nTheta) * Math.PI * 2;
    const row = [];
    for (let iz = 0; iz <= nZ; iz++) {
      const z = G.anode_length * iz / nZ;
      row.push(new BABYLON.Vector3(z, midR * Math.sin(angle), midR * Math.cos(angle)));
    }
    paths.push(row);
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
  return { cyl, mat, nTheta, nZ };
}

// ============================================================
// SNAP CACHE -- precompute heatmap textures
// ============================================================

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
        rgba[pi + 3] = 140;
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
// PIPELINE -- bloom, SSAO, glow (muted for CAD look)
// ============================================================

function buildPipeline(scene, cam) {
  const pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.10;
  pipe.bloomThreshold = 0.88;
  pipe.bloomKernel = 48;
  pipe.bloomScale = 0.5;
  pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.0;

  let ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.6;
    ssao.radius = 2.0;
    ssao.samples = 16;
    ssao.base = 0.10;
  } catch (_) {}

  const glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 40, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.25;
  const glowNames = new Set([
    "sheathDisk", "pinchCore", "pinchMantle", "beamCone", "gasGlow",
  ]);
  glowLayer.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (glowNames.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      const ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else if (mesh.name && mesh.name.indexOf("bRing") === 0 && mesh.material) {
      const ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, (mesh.material.alpha || 0) * 0.4);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline: pipe, ssao, glowLayer };
}

// ============================================================
// MAIN SCENE
// ============================================================

async function createDPFScene(canvas, data) {
  const L = data, G = L.geometry, S = L.sheath;
  const { engine, gpuBackend } = await initEngine(canvas);
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.08, 0.09, 0.11, 1);
  scene.setRenderingAutoClearDepthStencil(1, true, true, false);
  scene.setRenderingAutoClearDepthStencil(2, false, false, false);

  // Camera -- technical viewing angle
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI * 0.28, Math.PI * 0.30, G.cathode_radius * 9,
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
    orbitTimeout = setTimeout(function() { autoOrbit = true; }, 6000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !interacting) cam.alpha += 0.0006;
  });

  // Technical lighting -- neutral, even
  const key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  key.intensity = 1.0;
  key.diffuse = new BABYLON.Color3(0.96, 0.96, 0.98);
  const back = new BABYLON.DirectionalLight("back", new BABYLON.Vector3(1, -1, -1), scene);
  back.intensity = 0.45;
  back.diffuse = new BABYLON.Color3(0.92, 0.92, 0.95);
  const fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.40;
  fill.diffuse = new BABYLON.Color3(0.94, 0.94, 0.96);
  fill.groundColor = new BABYLON.Color3(0.22, 0.22, 0.26);

  // Build scene elements
  const dev = buildDevice(scene, G);
  buildGroundGrid(scene, G);
  const sheath = buildSheathTorus(scene, G);
  const gas = buildGasGlow(scene, G);
  const pinch = buildPinchColumn(scene, G);
  const beam = buildBeamCone(scene, G);
  const bField = buildBFieldTori(scene, G);
  const heat = buildHeatmapCylinder(scene, G);
  const parts = buildParticles(scene, G);
  const { pipeline, ssao, glowLayer } = buildPipeline(scene, cam);

  // Snap cache
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
        rgba[pi + 3] = 140;
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

  // ============================================================
  // applyFrame(i)
  // ============================================================

  function applyFrame(i) {
    if (i < 0 || i >= S.frames.length) return;
    const f = S.frames[i];
    const isP = isRadialPhase(f.phase);
    const cr = Math.max(0.02, f.r / G.cathode_radius);
    const Ifrac = clamp01(Math.abs(f.I / Math.max(S.I_peak, 0.001)));
    let pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;
    const col = PHASE_COLORS[f.phase] || [0.35, 0.55, 0.85];
    let rippleAmp = 0;

    // Heatmap snap sync
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      const ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // === SHEATH TORUS ===
    if (Ifrac > 0.01) {
      sheath.torus.isVisible = true;
      sheath.torus.position.x = isP ? G.anode_length : f.z;
      sheath.mat.alpha = clamp01(Ifrac * 0.30);
      sheath.mat.emissiveColor.set(col[0], col[1], col[2]);
      sheath.mat.emissiveFresnelParameters.leftColor.set(
        col[0] * 1.2, col[1] * 1.1, col[2] * 1.05);
      sheath.mat.emissiveFresnelParameters.rightColor.set(
        col[0] * 0.4, col[1] * 0.35, col[2] * 0.30);
      if (isP) {
        const scaleF = Math.max(0.05, cr);
        sheath.torus.scaling.y = scaleF;
        sheath.torus.scaling.z = scaleF;
      } else {
        sheath.torus.scaling.y = 1;
        sheath.torus.scaling.z = 1;
      }
    } else {
      sheath.torus.isVisible = false;
    }

    // === GAS GLOW ===
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      gas.glow.isVisible = true;
      const extent = isP ? G.anode_length : f.z;
      gas.glow.scaling.x = extent / G.anode_length;
      gas.glow.position.x = extent / 2;
      gas.mat.alpha = clamp01(Ifrac * 0.025);
      gas.mat.emissiveColor.set(col[0] * 0.18, col[1] * 0.15, col[2] * 0.12);
    } else {
      gas.glow.isVisible = false;
    }

    // === PINCH COLUMN ===
    const showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                       f.phase === "reflected") || (isP && cr < 0.35);
    if (showPinch && pI > 0.03) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      const pinchR = Math.max(G.anode_radius * 0.02, cr * G.cathode_radius * 0.15);
      const columnR = pinchR;
      rippleAmp = f.phase === "post_pinch" ? 0.30 : 0;

      const instData = f.instability || {};
      const sausageAmp = instData.sausage_amplitude || rippleAmp;
      const waveLen = 2 * Math.PI * columnR;
      const waveNum = Math.max(1, Math.round(
        (G.anode_length * 0.35) / Math.max(waveLen, 0.001)));

      for (let k = 0; k <= pinch.N; k++) {
        const zf = k / pinch.N;
        const taper = Math.sin(Math.PI * zf);
        const baseR = columnR * (0.2 + 0.8 * taper);
        const ripple = sausageAmp * baseR * Math.cos(
          2 * Math.PI * waveNum * zf + f.t * 2.5);
        pinch.radii[k] = Math.max(0.0003, baseR + ripple);
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

      pinch.coreMat.alpha = clamp01(pI * 0.80);
      pinch.mantleMat.alpha = clamp01(pI * 0.18);

      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(0.95, 0.92, 0.85);
      } else {
        pinch.coreMat.emissiveColor.set(
          lerp(0.50, 0.95, pI), lerp(0.40, 0.92, pI), lerp(0.20, 0.85, pI));
      }
      pinch.mantleMat.emissiveColor.set(
        lerp(0.60, 0.75, pI), lerp(0.25, 0.40, pI), lerp(0.08, 0.15, pI));
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // === BEAM CONE ===
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.08;
    beam.mat.alpha = beam.cone.isVisible ? clamp01(pI * 0.25) : 0;

    // === B-FIELD TORI ===
    const sheathZ = isP ? G.anode_length : f.z;
    for (let bi = 0; bi < bField.bRings.length; bi++) {
      const ringZ = bField.fracs[bi] * G.anode_length;
      const behindSheath = ringZ < sheathZ;
      bField.bRings[bi].isVisible = behindSheath && Ifrac > 0.05;
      if (bField.bRings[bi].isVisible) {
        if (isP) {
          const bScale = Math.max(0.08, cr);
          bField.bRings[bi].scaling.y = bScale;
          bField.bRings[bi].scaling.z = bScale;
        } else {
          bField.bRings[bi].scaling.y = 1;
          bField.bRings[bi].scaling.z = 1;
        }
      }
    }
    bField.mat.alpha = clamp01(0.3 * Ifrac);

    // === PARTICLES ===
    if (showPinch && pI > 0.2) {
      parts.ps.emitRate = Math.round(pI * 900);
      parts.emitter.position.x = G.anode_length;
    } else {
      parts.ps.emitRate = 0;
    }

    // === ANODE THERMAL TINT ===
    if (pI > 0.30) {
      dev.copperMat.emissiveColor.set(
        0.04 + (pI - 0.30) * 0.20, 0.02 + (pI - 0.30) * 0.08, 0.01);
    } else {
      dev.copperMat.emissiveColor.set(0.04, 0.02, 0.01);
    }

    // === PIPELINE ===
    glowLayer.intensity = 0.20 + pI * 0.25;
    pipeline.bloomWeight = 0.08 + pI * 0.14;

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
    beamCone: beam.cone, gasGlow: gas.glow,
    bRings: bField.bRings, fieldLines: [],
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
