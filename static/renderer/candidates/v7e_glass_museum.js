/**
 * DPF Glass Museum Renderer v7e
 *
 * Transparent crystal exhibit — device is a glass museum model with Fresnel
 * edge highlights. Plasma effects glow through: sapphire sheath, ruby radial,
 * diamond white pinch. Torus sheath spans exact 45mm gap.
 *
 * Babylon.js 8.x, StandardMaterial only.
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
  rundown:    [0.12, 0.35, 0.95],   // sapphire blue
  radial:     [0.85, 0.15, 0.20],   // ruby red
  mhd_radial: [0.85, 0.15, 0.20],
  reflected:  [0.70, 0.25, 0.60],
  pinch:      [0.95, 0.95, 1.00],   // diamond white
  post_pinch: [0.55, 0.30, 0.75],
};

const PHASE_LABELS = {
  rundown:    "Axial rundown",
  radial:     "Radial implosion",
  mhd_radial: "Radial compression (MHD)",
  mhd:        "MHD simulation",
  reflected:  "Reflected shock",
  pinch:      "Pinch — peak compression",
  post_pinch: "Post-pinch disruption",
  none:       "",
};

const PHASE_DESCRIPTIONS = {
  rundown:    "Current sheath sweeps neutral gas from insulator to anode tip — magnetic snowplow",
  radial:     "Magnetic pressure compresses plasma inward toward the axis",
  mhd_radial: "J x B force drives radial implosion — compression heating",
  mhd:        "Full MHD simulation of plasma dynamics",
  reflected:  "Reflected shock expands outward after axis convergence",
  pinch:      "PEAK COMPRESSION — fusion-relevant conditions at the axis",
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
// GLASS MATERIAL FACTORY — museum exhibit transparency
// ============================================================

function makeGlassMat(name, scene, tint, alpha) {
  const mat = new BABYLON.StandardMaterial(name, scene);
  mat.diffuseColor = new BABYLON.Color3(tint[0] * 0.2, tint[1] * 0.2, tint[2] * 0.2);
  mat.specularColor = new BABYLON.Color3(0.7, 0.7, 0.75);
  mat.specularPower = 128;
  mat.emissiveColor = new BABYLON.Color3(tint[0] * 0.06, tint[1] * 0.06, tint[2] * 0.06);
  mat.alpha = alpha;
  mat.backFaceCulling = false;
  // Fresnel: edges glow, faces nearly invisible
  mat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  mat.emissiveFresnelParameters.bias = 0.08;
  mat.emissiveFresnelParameters.power = 3;
  mat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(tint[0] * 0.4, tint[1] * 0.4, tint[2] * 0.4);
  mat.emissiveFresnelParameters.rightColor = BABYLON.Color3.Black();
  mat.opacityFresnelParameters = new BABYLON.FresnelParameters();
  mat.opacityFresnelParameters.bias = 0.9;
  mat.opacityFresnelParameters.power = 2;
  mat.opacityFresnelParameters.leftColor = BABYLON.Color3.White();
  mat.opacityFresnelParameters.rightColor = new BABYLON.Color3(0.08, 0.08, 0.08);
  return mat;
}

// ============================================================
// GLASS DEVICE — exact geometry (mm)
// ============================================================

function buildGlassDevice(scene, G) {
  const anodeMat = makeGlassMat("anodeGlass", scene, [0.65, 0.75, 0.90], 0.10);
  const rodMat = makeGlassMat("rodGlass", scene, [0.50, 0.55, 0.65], 0.10);
  const insMat = makeGlassMat("insGlass", scene, [0.85, 0.85, 0.95], 0.15);

  // Anode cylinder: r=115mm, length=600mm
  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 48, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = anodeMat;
  anode.renderingGroupId = 0;

  // 8 cathode rods at cathode_radius=160mm
  const N_RODS = G.n_cathode_rods || 8;
  const cathodeRods = [];
  for (let i = 0; i < N_RODS; i++) {
    const angle = (i / N_RODS) * Math.PI * 2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: G.cathode_radius * 0.04, height: G.anode_length * 1.05, tessellation: 8,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(
      G.anode_length / 2,
      G.cathode_radius * Math.sin(angle),
      G.cathode_radius * Math.cos(angle)
    );
    rod.material = rodMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  // End rings at z=0 and z=anode_length
  const ringThk = G.cathode_radius * 0.025;
  const baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 48,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.material = rodMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);
  const topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length;
  cathodeRods.push(topRing);

  // Insulator disk at z=0
  const insThk = G.insulator_thickness || G.anode_radius * 0.15;
  const insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.3;
  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 48,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = insMat;
  insulator.renderingGroupId = 0;

  return { anode, cathodeRods, insulator, anodeMat, rodMat, insMat };
}

// ============================================================
// SHEATH TORUS — exact 45mm gap, Fresnel edge glow
// ============================================================

function buildSheathTorus(scene, G) {
  const gap = G.cathode_radius - G.anode_radius; // 45mm
  const torusDiam = G.anode_radius + G.cathode_radius; // 275mm
  const torus = BABYLON.MeshBuilder.CreateTorus("sheathTorus", {
    diameter: torusDiam, thickness: gap, tessellation: 48,
  }, scene);
  torus.rotation.z = Math.PI / 2;

  const mat = new BABYLON.StandardMaterial("sheathMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.12, 0.35, 0.95); // sapphire
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  // Sheath Fresnel: vivid edges
  mat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  mat.emissiveFresnelParameters.bias = 0.2;
  mat.emissiveFresnelParameters.power = 2;
  mat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(0.4, 0.6, 1.0);
  mat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.05, 0.15, 0.4);
  torus.material = mat;
  torus.renderingGroupId = 1;

  return { torus, mat };
}

// ============================================================
// B-FIELD TORI — 4-5 behind sheath
// ============================================================

function buildBFieldTori(scene, G) {
  const bRings = [];
  const nRings = 5;
  const ringMat = new BABYLON.StandardMaterial("bRingMat", scene);
  ringMat.emissiveColor = new BABYLON.Color3(0.15, 0.25, 0.65);
  ringMat.disableLighting = true;
  ringMat.alpha = 0;
  ringMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  ringMat.backFaceCulling = false;
  ringMat.wireframe = true;

  for (let i = 0; i < nRings; i++) {
    const r = G.anode_radius * (0.35 + 0.55 * i / nRings);
    const ring = BABYLON.MeshBuilder.CreateTorus("bRing" + i, {
      diameter: r * 2, thickness: r * 0.06, tessellation: 32,
    }, scene);
    ring.rotation.z = Math.PI / 2;
    ring.material = ringMat;
    ring.renderingGroupId = 1;
    ring.isVisible = false;
    bRings.push(ring);
  }
  return { bRings, ringMat };
}

// ============================================================
// PLASMA TRAIL — ionized channel behind sheath
// ============================================================

function buildPlasmaTrail(scene, G) {
  const trail = BABYLON.MeshBuilder.CreateCylinder("plasmaTrail", {
    diameter: (G.anode_radius + G.cathode_radius), height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.NO_CAP,
  }, scene);
  trail.rotation.z = Math.PI / 2;
  trail.position.x = G.anode_length / 2;

  const mat = new BABYLON.StandardMaterial("trailMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.04, 0.08, 0.25);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  trail.material = mat;
  trail.renderingGroupId = 1;

  return { trail, mat };
}

// ============================================================
// PINCH COLUMN — tube from 0.85*L to 1.2*L, core + mantle + m=0
// ============================================================

function buildPinchColumn(scene, G) {
  const N = 24;
  const zStart = G.anode_length * 0.85;
  const zEnd = G.anode_length * 1.2;
  const path = [];
  for (let k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(zStart + (zEnd - zStart) * k / N, 0, 0));
  }
  const baseR = G.anode_radius * 0.12;
  const radii = new Array(N + 1).fill(baseR);

  // Core: 35% of mantle radius, diamond white
  const coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(1.0, 0.97, 0.93);
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

  // Mantle: ruby -> diamond transition
  const mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(0.55, 0.12, 0.18);
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
// BEAM CONE — post_pinch only, at anode tip
// ============================================================

function buildBeamCone(scene, G) {
  const cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.08,
    height: G.anode_length * 0.35, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length * 1.225;
  const mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.35, 0.50, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone, mat };
}

// ============================================================
// PINCH HALO — glow sphere at anode tip
// ============================================================

function buildHalo(scene, G) {
  const sphere = BABYLON.MeshBuilder.CreateSphere("halo", {
    diameter: G.anode_radius * 0.8, segments: 16,
  }, scene);
  sphere.position.x = G.anode_length;
  const mat = new BABYLON.StandardMaterial("haloMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.15, 0.10, 0.30);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  sphere.material = mat;
  sphere.renderingGroupId = 1;
  return { sphere, mat };
}

// ============================================================
// PARTICLES — 3000, additive, follow sheath
// ============================================================

function buildParticles(scene, G) {
  const emitter = new BABYLON.AbstractMesh("psEmitter", scene);
  emitter.position.x = G.anode_length;
  const ps = new BABYLON.ParticleSystem("sparks", 3000, scene);
  ps.emitter = emitter;
  ps.createSphereEmitter(G.anode_radius * 0.3);
  ps.color1 = new BABYLON.Color4(0.2, 0.4, 1.0, 0.7);
  ps.color2 = new BABYLON.Color4(0.95, 0.95, 1.0, 0.5);
  ps.colorDead = new BABYLON.Color4(0.05, 0.05, 0.15, 0);
  ps.minSize = G.cathode_radius * 0.003;
  ps.maxSize = G.cathode_radius * 0.010;
  ps.minLifeTime = 0.10;
  ps.maxLifeTime = 0.40;
  ps.emitRate = 0;
  ps.gravity = new BABYLON.Vector3(0, 0, 0);
  ps.minEmitPower = G.cathode_radius * 0.5;
  ps.maxEmitPower = G.cathode_radius * 2.0;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.start();
  return { ps, emitter };
}

// ============================================================
// GROUND GRID — subtle museum floor
// ============================================================

function buildGroundGrid(scene, G) {
  const ext = G.cathode_radius * 4;
  const lines = [];
  const nLines = 20;
  const step = ext * 2 / nLines;
  for (let i = 0; i <= nLines; i++) {
    const p = -ext + i * step;
    lines.push([new BABYLON.Vector3(-ext * 0.5, -G.cathode_radius * 1.5, p),
                new BABYLON.Vector3(G.anode_length + ext * 0.5, -G.cathode_radius * 1.5, p)]);
    lines.push([new BABYLON.Vector3(p + G.anode_length / 2, -G.cathode_radius * 1.5, -ext),
                new BABYLON.Vector3(p + G.anode_length / 2, -G.cathode_radius * 1.5, ext)]);
  }
  const grid = BABYLON.MeshBuilder.CreateLineSystem("grid", { lines: lines }, scene);
  grid.color = new BABYLON.Color3(0.10, 0.11, 0.14);
  grid.alpha = 0.25;
  grid.renderingGroupId = 0;
  return grid;
}

// ============================================================
// HEATMAP — full 360 degree cylindrical wrap at midgap radius
// ============================================================

function buildHeatmapCylinder(scene, G) {
  const midR = (G.anode_radius + G.cathode_radius) / 2;
  const nSeg = 48, nZ = 32;
  const paths = [];
  for (let iz = 0; iz <= nZ; iz++) {
    const z = G.anode_length * iz / nZ;
    const ring = [];
    for (let j = 0; j <= nSeg; j++) {
      const angle = (j / nSeg) * Math.PI * 2;
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

// ============================================================
// SNAP CACHE — heatmap frame decoding
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
        rgba[pi + 3] = 140; // alpha 0.55
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
// PIPELINE — bloom, glow, SSAO
// ============================================================

function buildPipeline(scene, cam) {
  const pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.25;
  pipe.bloomThreshold = 0.75;
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
    ssao.radius = 1.0;
    ssao.samples = 12;
    ssao.base = 0.1;
  } catch (_) {}

  const glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 48, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.35;
  const glowNames = new Set([
    "sheathTorus", "pinchCore", "pinchMantle", "beamCone", "plasmaTrail", "halo",
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
// MAIN SCENE
// ============================================================

async function createDPFScene(canvas, data) {
  const L = data, G = L.geometry, S = L.sheath;
  const { engine, gpuBackend } = await initEngine(canvas);
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.05, 0.06, 0.09, 1);
  scene.setRenderingAutoClearDepthStencil(1, true, true, false);
  scene.setRenderingAutoClearDepthStencil(2, false, false, false);

  // Camera
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI / 4, Math.PI / 3, G.cathode_radius * 9,
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
  cam.minZ = 0.0005;
  cam.inertia = 0.88;

  // Auto-orbit
  let autoOrbit = true, interacting = false, orbTimeout = null;
  canvas.addEventListener("pointerdown", function() {
    interacting = true; autoOrbit = false;
    if (orbTimeout) clearTimeout(orbTimeout);
  });
  canvas.addEventListener("pointerup", function() {
    interacting = false;
    orbTimeout = setTimeout(function() { autoOrbit = true; }, 5000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !interacting) cam.alpha += 0.0008;
  });

  // Lighting — museum spotlights
  const key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  key.intensity = 0.55;
  key.diffuse = new BABYLON.Color3(0.92, 0.93, 0.97);
  const fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.25;
  fill.diffuse = new BABYLON.Color3(0.70, 0.75, 0.85);
  fill.groundColor = new BABYLON.Color3(0.12, 0.12, 0.18);

  // Build all elements
  const dev = buildGlassDevice(scene, G);
  const sheath = buildSheathTorus(scene, G);
  const trail = buildPlasmaTrail(scene, G);
  const pinch = buildPinchColumn(scene, G);
  const beam = buildBeamCone(scene, G);
  const haloObj = buildHalo(scene, G);
  const bField = buildBFieldTori(scene, G);
  const heat = buildHeatmapCylinder(scene, G);
  const parts = buildParticles(scene, G);
  buildGroundGrid(scene, G);
  const { pipeline, ssao, glowLayer } = buildPipeline(scene, cam);

  // Heatmap cache
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
    const isP = isRadial(f.phase);
    const cr = Math.max(0.02, f.r / G.cathode_radius);
    const Ifrac = clamp01(Math.abs(f.I / Math.max(S.I_peak, 0.001)));
    let pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;
    const col = PHASE_COLORS[f.phase] || [0.12, 0.35, 0.95];

    // Snap heatmap to time
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
      // Rundown: slides along z-axis
      sheath.torus.position.x = isP ? G.anode_length : f.z;
      sheath.mat.alpha = clamp01(Ifrac * 0.40);
      sheath.mat.emissiveColor.set(col[0], col[1], col[2]);
      // Update Fresnel left color to match phase
      sheath.mat.emissiveFresnelParameters.leftColor.set(
        col[0] * 0.8 + 0.2, col[1] * 0.8 + 0.2, col[2] * 0.8 + 0.2);
      // Radial: contract torus
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

    // === PLASMA TRAIL ===
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      trail.trail.isVisible = true;
      const extent = isP ? G.anode_length : f.z;
      trail.trail.scaling.x = extent / G.anode_length;
      trail.trail.position.x = extent / 2;
      trail.mat.alpha = clamp01(Ifrac * 0.03);
      trail.mat.emissiveColor.set(col[0] * 0.2, col[1] * 0.2, col[2] * 0.35);
    } else {
      trail.trail.isVisible = false;
    }

    // === PINCH COLUMN ===
    const showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                       f.phase === "reflected") || (isP && cr < 0.35);
    let rippleAmp = 0;
    if (showPinch && pI > 0.03) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      // Radius: max(anode_r*0.02, compression * cathode_r * 0.15)
      const pinchR = Math.max(G.anode_radius * 0.02, cr * G.cathode_radius * 0.15);
      const instAmp = f.phase === "post_pinch" ? 0.4 : (f.phase === "pinch" ? 0.08 : 0);
      rippleAmp = instAmp;
      const waveNum = Math.min(6, Math.max(1, Math.round(
        0.25 * G.anode_length / (6.28 * Math.max(pinchR, 0.001)))));

      for (let k = 0; k <= pinch.N; k++) {
        const zf = k / pinch.N;
        const taper = Math.sin(Math.PI * zf);
        const lr = pinchR * (0.25 + 0.75 * taper);
        const ripple = instAmp * lr * Math.cos(6.28 * waveNum * zf + f.t * 4);
        pinch.radii[k] = Math.max(0.0003, lr + ripple);
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

      pinch.coreMat.alpha = clamp01(pI * 0.85);
      pinch.mantleMat.alpha = clamp01(pI * 0.25);

      // Diamond white at peak, ruby transition during compression
      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(1.0, 0.97, 0.93);
      } else {
        pinch.coreMat.emissiveColor.set(
          lerp(0.55, 1.0, pI), lerp(0.12, 0.97, pI), lerp(0.18, 0.93, pI));
      }
      pinch.mantleMat.emissiveColor.set(
        lerp(0.55, 0.85, pI), lerp(0.12, 0.20, pI), lerp(0.18, 0.30, pI));
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // === HALO ===
    if (showPinch && pI > 0.15) {
      haloObj.sphere.isVisible = true;
      const hScale = 1 + pI * 0.6;
      haloObj.sphere.scaling.set(hScale, hScale, hScale);
      haloObj.mat.alpha = clamp01(pI * 0.12);
      haloObj.mat.emissiveColor.set(
        lerp(0.10, 0.25, pI), lerp(0.08, 0.15, pI), lerp(0.20, 0.45, pI));
    } else {
      haloObj.sphere.isVisible = false;
    }

    // === BEAM CONE ===
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.08;
    beam.mat.alpha = beam.cone.isVisible ? clamp01(pI * 0.35) : 0;

    // === B-FIELD TORI — behind sheath, alpha proportional to I, compress during radial ===
    const sheathX = isP ? G.anode_length : f.z;
    for (let bi = 0; bi < bField.bRings.length; bi++) {
      const showB = Ifrac > 0.05;
      bField.bRings[bi].isVisible = showB;
      if (showB) {
        const offset = (bi + 1) * G.anode_length * 0.04;
        bField.bRings[bi].position.x = Math.max(0, sheathX - offset);
        if (isP) {
          const bScale = Math.max(0.05, cr) * (0.5 + 0.5 * bi / bField.bRings.length);
          bField.bRings[bi].scaling.set(bScale, bScale, bScale);
        } else {
          bField.bRings[bi].scaling.set(1, 1, 1);
        }
      }
    }
    bField.ringMat.alpha = Ifrac > 0.05 ? clamp01(Ifrac * 0.18) : 0;

    // === PARTICLES — follow sheath ===
    if (Ifrac > 0.1) {
      const rate = isP && pI > 0.2 ? Math.round(pI * 800) : Math.round(Ifrac * 200);
      parts.ps.emitRate = rate;
      parts.emitter.position.x = sheathX;
      if (isP) {
        parts.ps.createSphereEmitter(Math.max(G.anode_radius * 0.05, cr * G.cathode_radius * 0.3));
      } else {
        parts.ps.createSphereEmitter(G.anode_radius * 0.3);
      }
    } else {
      parts.ps.emitRate = 0;
    }

    // === GLASS GLOW under compression ===
    if (pI > 0.3) {
      const gv = (pI - 0.3) * 0.12;
      dev.anodeMat.emissiveColor.set(0.06 + gv * 0.6, 0.06 + gv * 0.4, 0.06 + gv);
    } else {
      dev.anodeMat.emissiveColor.set(0.04, 0.045, 0.054);
    }

    // === PIPELINE ===
    glowLayer.intensity = 0.25 + pI * 0.40;
    pipeline.bloomWeight = 0.18 + pI * 0.18;

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
    beamCone: beam.cone, gasGlow: trail.trail,
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
