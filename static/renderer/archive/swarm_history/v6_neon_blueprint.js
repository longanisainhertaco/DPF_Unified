/** DPF Neon Blueprint Renderer — Holographic Engineering Display
 * Dark blueprint bg, cyan/teal wireframe device, neon plasma effects. Tron aesthetic. */
// ============================================================
// COLORMAPS & CONSTANTS

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
  rundown:    [0.1, 0.55, 1.0],
  radial:     [1.0, 0.45, 0.05],
  mhd_radial: [1.0, 0.5, 0.1],
  reflected:  [1.0, 0.6, 0.0],
  pinch:      [1.0, 0.95, 0.85],
  post_pinch: [0.8, 0.2, 0.1],
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
  var t = Math.max(0, Math.min(1, v));
  var n = cmap.length - 1, idx = t * n;
  var lo = Math.floor(idx), hi = Math.min(lo + 1, n), f = idx - lo;
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
  var engine, gpuBackend = "WebGL2";
  var params = new URLSearchParams(window.location.search);
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
  return { engine: engine, gpuBackend: gpuBackend };
}

// ============================================================
// NEON BLUEPRINT DEVICE — wireframe with cyan/teal aesthetic
// ============================================================

function buildDevice(scene, G) {
  // Anode — cyan wireframe cylinder
  var anodeMat = new BABYLON.StandardMaterial("anodeMat", scene);
  anodeMat.diffuseColor = new BABYLON.Color3(0.0, 0.0, 0.0);
  anodeMat.emissiveColor = new BABYLON.Color3(0.0, 0.7, 0.75);
  anodeMat.specularColor = new BABYLON.Color3(0.0, 0.3, 0.3);
  anodeMat.alpha = 0.35;
  anodeMat.wireframe = true;
  anodeMat.backFaceCulling = false;

  var anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = anodeMat;
  anode.renderingGroupId = 0;

  // Cathode rods — thin steel-blue lines
  var rodMat = new BABYLON.StandardMaterial("rodMat", scene);
  rodMat.diffuseColor = new BABYLON.Color3(0.0, 0.0, 0.0);
  rodMat.emissiveColor = new BABYLON.Color3(0.3, 0.45, 0.55);
  rodMat.specularColor = new BABYLON.Color3(0.1, 0.2, 0.3);
  rodMat.alpha = 0.5;

  var N_RODS = G.n_cathode_rods || 8;
  var cathodeRods = [];
  for (var i = 0; i < N_RODS; i++) {
    var angle = (i / N_RODS) * Math.PI * 2;
    var rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: G.cathode_radius * 0.025, height: G.anode_length * 1.05, tessellation: 6,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(G.anode_length / 2,
      G.cathode_radius * Math.sin(angle), G.cathode_radius * Math.cos(angle));
    rod.material = rodMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  // End rings
  var ringThk = G.cathode_radius * 0.02;
  var baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 48,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.material = rodMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);
  var topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length;
  cathodeRods.push(topRing);

  // Insulator — dim teal ring at base
  var insMat = new BABYLON.StandardMaterial("insMat", scene);
  insMat.diffuseColor = new BABYLON.Color3(0.0, 0.0, 0.0);
  insMat.emissiveColor = new BABYLON.Color3(0.15, 0.35, 0.35);
  insMat.alpha = 0.25;

  var insThk = G.anode_radius * 0.15;
  var insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.3;
  var insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 48,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = insMat;
  insulator.renderingGroupId = 0;

  return { anode: anode, cathodeRods: cathodeRods, insulator: insulator, anodeMat: anodeMat, rodMat: rodMat };
}

// ============================================================
// GROUND GRID — blueprint-style floor grid
// ============================================================

function buildGroundGrid(scene, G) {
  var extent = G.cathode_radius * 6;
  var grid = BABYLON.MeshBuilder.CreateGround("grid", { width: extent, height: extent, subdivisions: 1 }, scene);
  grid.position.y = -G.cathode_radius * 1.5;
  grid.position.x = G.anode_length / 2;
  var gridMat = new BABYLON.StandardMaterial("gridMat", scene);
  gridMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  gridMat.emissiveColor = new BABYLON.Color3(0.04, 0.08, 0.1);
  gridMat.wireframe = true;
  gridMat.alpha = 0.15;
  grid.material = gridMat;
  grid.isPickable = false;

  // Finer overlay grid
  var fine = BABYLON.MeshBuilder.CreateGround("gridFine", { width: extent, height: extent, subdivisions: 20 }, scene);
  fine.position.copyFrom(grid.position);
  var fineMat = new BABYLON.StandardMaterial("gridFineMat", scene);
  fineMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  fineMat.emissiveColor = new BABYLON.Color3(0.02, 0.06, 0.08);
  fineMat.wireframe = true;
  fineMat.alpha = 0.08;
  fine.material = fineMat;
  fine.isPickable = false;
  return grid;
}

// ============================================================
// SHEATH TORUS — annular ring between anode and cathode
// ============================================================

function buildSheathTorus(scene, G) {
  var midR = (G.anode_radius + G.cathode_radius) / 2;
  var tubeR = (G.cathode_radius - G.anode_radius) / 2;
  var torus = BABYLON.MeshBuilder.CreateTorus("sheathDisk", {
    diameter: midR * 2, thickness: tubeR * 2, tessellation: 48,
  }, scene);
  torus.rotation.z = Math.PI / 2;
  torus.position.x = 0;

  var mat = new BABYLON.StandardMaterial("sheathMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.1, 0.55, 1.0);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  torus.material = mat;
  torus.renderingGroupId = 1;

  return { mesh: torus, mat: mat, midR: midR, tubeR: tubeR };
}

// ============================================================
// PLASMA TRAIL — tube behind sheath showing swept gas
// ============================================================

function buildPlasmaTrail(scene, G) {
  var N = 24;
  var path = [];
  for (var k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(G.anode_length * k / N, 0, 0));
  }
  var trailR = G.anode_radius * 0.35;

  var mat = new BABYLON.StandardMaterial("trailMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.05, 0.2, 0.5);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;

  var tube = BABYLON.MeshBuilder.CreateTube("plasmaTrail", {
    path: path, radius: trailR, tessellation: 16,
    cap: BABYLON.Mesh.NO_CAP, updatable: true,
  }, scene);
  tube.material = mat;
  tube.renderingGroupId = 1;
  tube.isVisible = false;

  return { tube: tube, mat: mat, path: path, N: N, trailR: trailR };
}

// ============================================================
// PINCH COLUMN — bright core + dim mantle with Bennett profile
// ============================================================

function buildPinchColumn(scene, G) {
  var N = 24;
  var columnLen = G.anode_length * 0.3;
  var tipX = G.anode_length;
  var path = [];
  for (var k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(tipX - columnLen * 0.1 + columnLen * k / N, 0, 0));
  }
  var radii = new Array(N + 1).fill(G.anode_radius * 0.12);

  // Core: hot white-blue
  var coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(0.9, 0.95, 1.0);
  coreMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  var core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path, radiusFunction: function(i) { return radii[i] * 0.3; },
    tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;

  // Mantle: orange-red glow
  var mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(1.0, 0.5, 0.1);
  mantleMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mantleMat.disableLighting = true;
  mantleMat.alpha = 0;
  mantleMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mantleMat.backFaceCulling = false;

  var mantle = BABYLON.MeshBuilder.CreateTube("pinchMantle", {
    path: path, radiusFunction: function(i) { return radii[i]; },
    tessellation: 16, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  mantle.material = mantleMat;
  mantle.renderingGroupId = 1;

  return { core: core, mantle: mantle, coreMat: coreMat, mantleMat: mantleMat,
           radii: radii, path: path, N: N };
}

// ============================================================
// HALO — backside glow around pinch
// ============================================================

function buildHalo(scene, G) {
  var halo = BABYLON.MeshBuilder.CreateSphere("halo", { diameter: G.anode_radius * 0.8, segments: 16 }, scene);
  halo.position.x = G.anode_length;
  var mat = new BABYLON.StandardMaterial("haloMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.3, 0.5, 1.0);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = true;
  halo.material = mat;
  halo.renderingGroupId = 1;
  return { mesh: halo, mat: mat };
}

// ============================================================
// PARTICLES — 3000 capacity, additive, sphere emitter
// ============================================================

function buildParticles(scene, G) {
  var emitter = new BABYLON.AbstractMesh("psEmitter", scene);
  emitter.position.x = G.anode_length;
  var ps = new BABYLON.ParticleSystem("sparks", 3000, scene);
  ps.emitter = emitter;
  ps.createSphereEmitter(G.anode_radius * 0.25);
  ps.color1 = new BABYLON.Color4(0.3, 0.7, 1.0, 0.9);
  ps.color2 = new BABYLON.Color4(1.0, 0.6, 0.2, 0.7);
  ps.colorDead = new BABYLON.Color4(0.05, 0.1, 0.2, 0);
  ps.minSize = G.cathode_radius * 0.003;
  ps.maxSize = G.cathode_radius * 0.012;
  ps.minLifeTime = 0.1;
  ps.maxLifeTime = 0.45;
  ps.emitRate = 0;
  ps.gravity = new BABYLON.Vector3(0, 0, 0);
  ps.minEmitPower = G.cathode_radius * 0.4;
  ps.maxEmitPower = G.cathode_radius * 1.8;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.start();
  return { ps: ps, emitter: emitter };
}

// ============================================================
// B-FIELD TORI — concentric rings showing magnetic topology
// ============================================================

function buildBFieldTori(scene, G) {
  var bRings = [];
  var nRings = 5;
  var ringMat = new BABYLON.StandardMaterial("bRingMat", scene);
  ringMat.emissiveColor = new BABYLON.Color3(0.0, 0.35, 0.55);
  ringMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  ringMat.disableLighting = true;
  ringMat.alpha = 0;
  ringMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  ringMat.backFaceCulling = false;
  ringMat.wireframe = true;

  for (var i = 0; i < nRings; i++) {
    var frac = (i + 1) / (nRings + 1);
    var zPos = G.anode_length * frac;
    var bRadius = G.anode_radius * (0.6 + 0.3 * Math.sin(frac * Math.PI));
    var ring = BABYLON.MeshBuilder.CreateTorus("bRing" + i, {
      diameter: bRadius * 2, thickness: G.anode_radius * 0.02,
      tessellation: 32,
    }, scene);
    ring.rotation.z = Math.PI / 2;
    ring.position.x = zPos;
    ring.material = ringMat;
    ring.renderingGroupId = 1;
    ring.isVisible = false;
    bRings.push(ring);
  }
  return { rings: bRings, mat: ringMat };
}

// ============================================================
// BEAM CONE — post-pinch particle beam
// ============================================================

function buildBeamCone(scene, G) {
  var cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.08,
    height: G.anode_length * 0.35, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length + G.anode_length * 0.175 + G.anode_length * 0.05;
  var mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.4, 0.7, 1.0);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone: cone, mat: mat };
}

// ============================================================
// GAS GLOW — subtle swept region behind sheath
// ============================================================

function buildGasGlow(scene, G) {
  var glow = BABYLON.MeshBuilder.CreateCylinder("gasGlow", {
    diameter: (G.anode_radius + G.cathode_radius), height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.NO_CAP,
  }, scene);
  glow.rotation.z = Math.PI / 2;
  glow.position.x = G.anode_length / 2;
  var mat = new BABYLON.StandardMaterial("gasGlowMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.05, 0.15, 0.35);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  glow.material = mat;
  glow.renderingGroupId = 1;
  return { glow: glow, mat: mat };
}

// ============================================================
// HEATMAP OVERLAY
// ============================================================

function buildHeatmapCylinder(scene, G) {
  // Cylindrical heatmap: wraps MHD field data around the device at midplane radius
  // The r-z data is mapped onto a cylinder so it's visible from every camera angle
  var midR = (G.anode_radius + G.cathode_radius) / 2;
  var nTheta = 48;  // circumferential segments
  var nZ = 32;      // axial segments (matches MHD grid)
  var paths = [];
  for (var it = 0; it <= nTheta; it++) {
    var angle = (it / nTheta) * Math.PI * 2;
    var ring = [];
    for (var iz = 0; iz <= nZ; iz++) {
      var z = G.anode_length * iz / nZ;
      ring.push(new BABYLON.Vector3(z, midR * Math.sin(angle), midR * Math.cos(angle)));
    }
    paths.push(ring);
  }
  var cylinder = BABYLON.MeshBuilder.CreateRibbon("heatPlane", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false,
  }, scene);
  cylinder.isVisible = false;
  cylinder.isPickable = false;
  var mat = new BABYLON.StandardMaterial("heatMat", scene);
  mat.disableLighting = true;
  mat.backFaceCulling = false;
  mat.alpha = 0.6;
  cylinder.material = mat;
  cylinder.renderingGroupId = 2;
  return { plane: cylinder, mat: mat };
}

function buildSnapCache(fieldKey, layer, cache) {
  if (!layer || !layer.frames || !layer.frames.length) return;
  var shape = layer.frames_shape || layer.shape;
  if (!shape) return;
  var nr = shape[0], nz = shape[1];
  var n = layer.frames.length;
  var nTheta = 49;  // match cylindrical mesh (48 segments + 1)
  var times = new Float64Array(n);
  var rgbaFrames = new Array(n);
  for (var fi = 0; fi < n; fi++) {
    times[fi] = layer.frames[fi].t_us;
    var vals = b64ToFloat32(layer.frames[fi].data);
    // Build cylindrical texture: [nTheta x nZ]
    // Since data is axisymmetric, each theta row shows the radially-averaged z-profile
    // Average across r for each z position
    var zProfile = new Float32Array(nz);
    for (var iz = 0; iz < nz; iz++) {
      var sum = 0;
      for (var ir = 0; ir < nr; ir++) sum += vals[ir * nz + iz];
      zProfile[iz] = sum / nr;
    }
    var rgba = new Uint8Array(nTheta * nz * 4);
    for (var it = 0; it < nTheta; it++) {
      for (var iz2 = 0; iz2 < nz; iz2++) {
        var v = zProfile[iz2];
        var c = cmapLookup(v, activeCmap);
        var pi = (it * nz + iz2) * 4;
        rgba[pi]     = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 160;
      }
    }
    rgbaFrames[fi] = rgba;
  }
  cache[fieldKey] = { times: times, rgba: rgbaFrames, texW: nz, texH: nTheta };
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

// ============================================================
// PIPELINE — bloom + SSAO + glow
// ============================================================

function buildPipeline(scene, cam) {
  var pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.2;
  pipe.bloomThreshold = 0.7;
  pipe.bloomKernel = 64;
  pipe.bloomScale = 0.5;
  pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.0;

  var ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.5;
    ssao.radius = 1.5;
    ssao.samples = 12;
    ssao.base = 0.1;
  } catch (_) {}

  var glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 64, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.4;
  var glowNames = new Set([
    "sheathDisk", "pinchCore", "pinchMantle", "beamCone", "gasGlow",
    "plasmaTrail", "halo",
  ]);
  glowLayer.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (glowNames.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      var ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else if (mesh.name && mesh.name.indexOf("bRing") === 0 && mesh.material) {
      var bc = mesh.material.emissiveColor;
      result.set(bc.r, bc.g, bc.b, mesh.material.alpha || 0);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline: pipe, ssao: ssao, glowLayer: glowLayer };
}

// ============================================================
// MAIN SCENE
// ============================================================

async function createDPFScene(canvas, data) {
  var L = data, G = L.geometry, S = L.sheath;
  var init = await initEngine(canvas);
  var engine = init.engine, gpuBackend = init.gpuBackend;
  var scene = new BABYLON.Scene(engine);

  // Dark blueprint background
  scene.clearColor = new BABYLON.Color4(0.08, 0.09, 0.12, 1);
  scene.setRenderingAutoClearDepthStencil(1, true, true, false);
  scene.setRenderingAutoClearDepthStencil(2, false, false, false);
  scene.ambientColor = new BABYLON.Color3(0.02, 0.04, 0.06);

  // Camera
  var cam = new BABYLON.ArcRotateCamera("cam",
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
  var autoOrbit = true, interacting = false, orbitTimeout = null;
  canvas.addEventListener("pointerdown", function() {
    interacting = true; autoOrbit = false;
    if (orbitTimeout) clearTimeout(orbitTimeout);
  });
  canvas.addEventListener("pointerup", function() {
    interacting = false;
    orbitTimeout = setTimeout(function() { autoOrbit = true; }, 5000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !interacting) cam.alpha += 0.0008;
  });

  // Lights — dim, let emissives and glow dominate
  var keyLight = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  keyLight.intensity = 0.4;
  keyLight.diffuse = new BABYLON.Color3(0.6, 0.7, 0.8);
  var fillLight = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fillLight.intensity = 0.2;
  fillLight.diffuse = new BABYLON.Color3(0.5, 0.6, 0.7);
  fillLight.groundColor = new BABYLON.Color3(0.05, 0.08, 0.12);

  // Build all elements
  var dev = buildDevice(scene, G);
  var grid = buildGroundGrid(scene, G);
  var sheath = buildSheathTorus(scene, G);
  var trail = buildPlasmaTrail(scene, G);
  var pinch = buildPinchColumn(scene, G);
  var halo = buildHalo(scene, G);
  var gas = buildGasGlow(scene, G);
  var beam = buildBeamCone(scene, G);
  var bField = buildBFieldTori(scene, G);
  var parts = buildParticles(scene, G);
  var heat = buildHeatmapCylinder(scene, G);
  var pipeResult = buildPipeline(scene, cam);
  var pipeline = pipeResult.pipeline;
  var ssao = pipeResult.ssao;
  var glowLayer = pipeResult.glowLayer;

  // Snap cache for heatmaps
  var snapCache = {};
  var lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
  var heatTex = null;
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
    heat.mat.alpha = 0.75;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  var activeOverlay = "none";

  function updateHeatmap(ovKey) {
    if (!L || ovKey === "none") { heat.plane.isVisible = false; return; }
    var layer = L[ovKey];
    if (!layer || (!layer.data && !layer.frames)) { heat.plane.isVisible = false; return; }
    if (snapCache[ovKey] && lastSnapIdx[ovKey] >= 0) { applySnapTex(ovKey); return; }
    if (!layer.data || !layer.shape) { heat.plane.isVisible = false; return; }
    // Build cylindrical texture from static field data
    var vals = b64ToFloat32(layer.data);
    var nr = layer.shape[0], nz = layer.shape[1];
    var nTheta = 49;
    // Radially-averaged z-profile, tiled around circumference
    var zProf = new Float32Array(nz);
    for (var iz = 0; iz < nz; iz++) {
      var sum = 0;
      for (var ir = 0; ir < nr; ir++) sum += vals[ir * nz + iz];
      zProf[iz] = sum / nr;
    }
    var rgba = new Uint8Array(nTheta * nz * 4);
    for (var it = 0; it < nTheta; it++) {
      for (var iz2 = 0; iz2 < nz; iz2++) {
        var v = zProf[iz2];
        var c = cmapLookup(v, activeCmap);
        var pi = (it * nz + iz2) * 4;
        rgba[pi]     = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 160;
      }
    }
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(rgba, nz, nTheta, BABYLON.Engine.TEXTUREFORMAT_RGBA,
      scene, false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.55;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.plane.isVisible = true;
  }

  // ============================================================
  // applyFrame(i)
  // ============================================================

  function applyFrame(i) {
    if (i < 0 || i >= S.frames.length) return;
    var f = S.frames[i];
    var isP = isRadial(f.phase);
    var cr = Math.max(0.02, f.r / G.cathode_radius);
    var Ifrac = clamp01(Math.abs(f.I / Math.max(S.I_peak, 0.001)));
    var pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;
    var col = PHASE_COLORS[f.phase] || [0.1, 0.55, 1.0];
    var rippleAmp = 0;

    // Heatmap snap sync
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      var ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // === SHEATH TORUS ===
    if (Ifrac > 0.01) {
      sheath.mesh.isVisible = true;
      sheath.mesh.position.x = isP ? G.anode_length : f.z;
      sheath.mat.alpha = clamp01(Ifrac * 0.35);
      sheath.mat.emissiveColor.set(col[0], col[1], col[2]);

      if (isP) {
        var outerR = f.r;
        var innerR = G.anode_radius * 0.05;
        var newMidR = (innerR + outerR) / 2;
        var newTubeR = Math.max(0.001, (outerR - innerR) / 2);
        var scaleXY = newMidR / sheath.midR;
        var scaleThick = newTubeR / sheath.tubeR;
        sheath.mesh.scaling.set(1, scaleXY, scaleXY);
        sheath.mat.alpha = clamp01(Ifrac * 0.45);
      } else {
        sheath.mesh.scaling.set(1, 1, 1);
      }
    } else {
      sheath.mesh.isVisible = false;
    }

    // === PLASMA TRAIL — behind sheath ===
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      trail.tube.isVisible = true;
      var extent = isP ? G.anode_length : f.z;
      trail.tube.scaling.x = extent / G.anode_length;
      trail.tube.position.x = 0;
      trail.mat.alpha = clamp01(Ifrac * 0.06);
      trail.mat.emissiveColor.set(col[0] * 0.3, col[1] * 0.3, col[2] * 0.5);
    } else {
      trail.tube.isVisible = false;
    }

    // === GAS GLOW ===
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      gas.glow.isVisible = true;
      var gasExtent = isP ? G.anode_length : f.z;
      gas.glow.scaling.x = gasExtent / G.anode_length;
      gas.glow.position.x = gasExtent / 2;
      gas.mat.alpha = clamp01(Ifrac * 0.035);
      gas.mat.emissiveColor.set(col[0] * 0.15, col[1] * 0.2, col[2] * 0.4);
    } else {
      gas.glow.isVisible = false;
    }

    // === PINCH COLUMN ===
    var showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                     f.phase === "reflected") || (isP && cr < 0.35);
    if (showPinch && pI > 0.03) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      var pinchR = Math.max(G.anode_radius * 0.01, cr * G.cathode_radius * 0.12);
      rippleAmp = f.phase === "post_pinch" ? 0.35 : 0;
      var waveNum = Math.min(5, Math.max(1, Math.round(
        0.25 * G.anode_length / (6.28 * Math.max(pinchR, 0.001)))));

      for (var k = 0; k <= pinch.N; k++) {
        var zf = k / pinch.N;
        var taper = Math.sin(Math.PI * zf);
        var lr = pinchR * (0.25 + 0.75 * taper);
        var ripple = rippleAmp * lr * Math.cos(6.28 * waveNum * zf + f.t * 3);
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

      pinch.coreMat.alpha = clamp01(pI * 0.85);
      pinch.mantleMat.alpha = clamp01(pI * 0.2);

      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(1.0, 0.98, 0.95);
      } else {
        pinch.coreMat.emissiveColor.set(0.4 + pI * 0.6, 0.5 + pI * 0.4, 0.8 + pI * 0.2);
      }
      pinch.mantleMat.emissiveColor.set(
        lerp(0.8, 1.0, pI), lerp(0.3, 0.5, pI), lerp(0.05, 0.1, pI));
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // === HALO ===
    if (showPinch && pI > 0.1) {
      halo.mesh.isVisible = true;
      var haloScale = 1.0 + pI * 2.0;
      halo.mesh.scaling.set(haloScale, haloScale, haloScale);
      halo.mat.alpha = clamp01(pI * 0.12);
      halo.mat.emissiveColor.set(
        lerp(0.2, col[0], pI), lerp(0.3, col[1], pI), lerp(0.6, col[2], pI));
    } else {
      halo.mesh.isVisible = false;
    }

    // === BEAM CONE ===
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.08;
    beam.mat.alpha = beam.cone.isVisible ? clamp01(pI * 0.4) : 0;

    // === B-FIELD TORI ===
    var showBField = Ifrac > 0.1;
    for (var bi = 0; bi < bField.rings.length; bi++) {
      var ring = bField.rings[bi];
      ring.isVisible = showBField;
      if (showBField) {
        var bScale = 0.8 + 0.4 * Math.sin(f.t * 2 + bi * 1.2);
        ring.scaling.set(1, bScale, bScale);
        bField.mat.alpha = clamp01(Ifrac * 0.12);
      }
    }

    // === PARTICLES ===
    if (showPinch && pI > 0.2) {
      parts.ps.emitRate = Math.round(pI * 800);
      parts.emitter.position.x = G.anode_length;
    } else if (Ifrac > 0.3) {
      parts.ps.emitRate = Math.round(Ifrac * 150);
      parts.emitter.position.x = isP ? G.anode_length : f.z;
    } else {
      parts.ps.emitRate = 0;
    }

    // === ANODE WIREFRAME GLOW — intensifies during pinch ===
    if (pI > 0.3) {
      dev.anodeMat.emissiveColor.set(
        lerp(0.0, 0.2, pI), lerp(0.7, 0.9, pI), lerp(0.75, 1.0, pI));
    } else {
      dev.anodeMat.emissiveColor.set(0.0, 0.7, 0.75);
    }

    // === PIPELINE dynamics ===
    glowLayer.intensity = 0.3 + pI * 0.4;
    pipeline.bloomWeight = 0.15 + pI * 0.2;

    return { f: f, isP: isP, cr: cr, pI: pI, rippleAmp: rippleAmp };
  }

  // ============================================================
  // RETURN API
  // ============================================================

  return {
    engine: engine, scene: scene, camera: cam,
    gpuBackend: gpuBackend, useGPU: gpuBackend === "WebGPU",
    G: G, S: S, L: L,
    anode: dev.anode, cathodeRods: dev.cathodeRods, insulator: dev.insulator,
    sheathDisk: sheath.mesh, pinchCore: pinch.core, pinchMantle: pinch.mantle,
    beamCone: beam.cone, gasGlow: gas.glow,
    bRings: bField.rings, fieldLines: [],
    ps: { start: function() { parts.ps.start(); }, stop: function() { parts.ps.stop(); } },
    pipeline: pipeline, ssao: ssao, glowLayer: glowLayer,
    applyFrame: applyFrame,
    updateHeatmap: updateHeatmap,
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
