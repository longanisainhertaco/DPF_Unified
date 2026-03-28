/**
 * DPF Renderer v7a -- "Cinematic Physics"
 *
 * Proportionally accurate physics animation with cinematic camera and lighting.
 * Sheath is a torus spanning anode-to-cathode, B-field tori behind sheath,
 * pinch column with Bennett profile + m=0 sausage instability.
 * Color palette shifts: cool blue (rundown) -> warm orange (radial) -> white-hot (pinch).
 * Cylindrical heatmap wrap at midplane radius with viridis colormap.
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

const PHASE_COLORS_COOL = {
  rundown:    [0.20, 0.50, 1.00],
  radial:     [1.00, 0.55, 0.10],
  mhd_radial: [1.00, 0.55, 0.10],
  reflected:  [1.00, 0.65, 0.15],
  pinch:      [1.00, 0.97, 0.90],
  post_pinch: [0.85, 0.35, 0.10],
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
function smoothstep(lo, hi, v) {
  var t = clamp01((v - lo) / (hi - lo));
  return t * t * (3 - 2 * t);
}

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
// WIREFRAME DEVICE -- proportionally accurate
// ============================================================

function buildDevice(scene, G) {
  var copperMat = new BABYLON.StandardMaterial("copper", scene);
  copperMat.diffuseColor = new BABYLON.Color3(0.78, 0.55, 0.28);
  copperMat.specularColor = new BABYLON.Color3(0.6, 0.45, 0.2);
  copperMat.emissiveColor = new BABYLON.Color3(0.04, 0.02, 0.01);
  copperMat.specularPower = 32;
  copperMat.alpha = 0.3;
  copperMat.wireframe = true;
  copperMat.backFaceCulling = false;

  var steelMat = new BABYLON.StandardMaterial("steel", scene);
  steelMat.diffuseColor = new BABYLON.Color3(0.50, 0.48, 0.52);
  steelMat.specularColor = new BABYLON.Color3(0.4, 0.4, 0.45);
  steelMat.emissiveColor = new BABYLON.Color3(0.03, 0.03, 0.04);
  steelMat.specularPower = 24;
  steelMat.alpha = 0.5;

  var ceramicMat = new BABYLON.StandardMaterial("ceramic", scene);
  ceramicMat.diffuseColor = new BABYLON.Color3(0.88, 0.84, 0.72);
  ceramicMat.specularColor = new BABYLON.Color3(0.12, 0.10, 0.08);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.03, 0.02, 0.01);
  ceramicMat.specularPower = 8;
  ceramicMat.alpha = 0.35;

  var anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 48, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.renderingGroupId = 0;

  var N_RODS = 8;
  var rodDiam = G.cathode_radius * 0.04;
  var cathodeRods = [];
  for (var i = 0; i < N_RODS; i++) {
    var angle = (i / N_RODS) * Math.PI * 2;
    var rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
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

  var ringThk = (G.cathode_radius - G.anode_radius) * 0.14;
  var baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 48,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.position.x = 0;
  baseRing.material = steelMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);
  var topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length;
  cathodeRods.push(topRing);

  var insThk = G.anode_radius * 0.15;
  var insOuterR = G.cathode_radius;
  var insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 48,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  return { anode: anode, cathodeRods: cathodeRods, insulator: insulator,
           copperMat: copperMat, steelMat: steelMat, ceramicMat: ceramicMat };
}

// ============================================================
// SHEATH TORUS -- spans anode surface to cathode surface
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
  mat.emissiveColor = new BABYLON.Color3(0.20, 0.50, 1.00);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  mat.useFresnelOnDiffuse = false;
  torus.material = mat;
  torus.renderingGroupId = 1;

  var fresnelParams = new BABYLON.FresnelParameters();
  fresnelParams.bias = 0.1;
  fresnelParams.power = 2.0;
  fresnelParams.leftColor = new BABYLON.Color3(1, 1, 1);
  fresnelParams.rightColor = new BABYLON.Color3(0.2, 0.5, 1.0);
  mat.emissiveFresnelParameters = fresnelParams;

  return { disk: torus, mat: mat, midR: midR, tubeR: tubeR, fresnelParams: fresnelParams };
}

// ============================================================
// GAS GLOW -- swept plasma behind the sheath
// ============================================================

function buildGasGlow(scene, G) {
  var glow = BABYLON.MeshBuilder.CreateCylinder("gasGlow", {
    diameter: (G.anode_radius + G.cathode_radius), height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.NO_CAP,
  }, scene);
  glow.rotation.z = Math.PI / 2;
  glow.position.x = G.anode_length / 2;
  var mat = new BABYLON.StandardMaterial("gasGlowMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.08, 0.15, 0.35);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  glow.material = mat;
  glow.renderingGroupId = 1;
  return { glow: glow, mat: mat };
}

// ============================================================
// PINCH COLUMN -- Bennett profile, extends beyond anode tip
// ============================================================

function buildPinchColumn(scene, G) {
  var N = 24;
  var columnLen = G.anode_length * 0.3;
  var tipX = G.anode_length;
  var path = [];
  for (var k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(tipX - columnLen * 0.1 + columnLen * k / N, 0, 0));
  }
  var radii = new Array(N + 1).fill(G.anode_radius * 0.10);

  var coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(1.0, 0.97, 0.90);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  var core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path, radiusFunction: function(i) { return radii[i] * 0.35; },
    tessellation: 14, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;

  var mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(1.0, 0.45, 0.10);
  mantleMat.disableLighting = true;
  mantleMat.alpha = 0;
  mantleMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mantleMat.backFaceCulling = false;

  var mantle = BABYLON.MeshBuilder.CreateTube("pinchMantle", {
    path: path, radiusFunction: function(i) { return radii[i]; },
    tessellation: 18, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  mantle.material = mantleMat;
  mantle.renderingGroupId = 1;

  return { core: core, mantle: mantle, coreMat: coreMat, mantleMat: mantleMat,
           radii: radii, path: path, N: N, columnLen: columnLen };
}

// ============================================================
// BEAM CONE -- post-pinch particle beam
// ============================================================

function buildBeamCone(scene, G) {
  var cone = BABYLON.MeshBuilder.CreateCylinder("beamCone", {
    diameterTop: 0, diameterBottom: G.anode_radius * 0.08,
    height: G.anode_length * 0.35, tessellation: 12,
  }, scene);
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = G.anode_length + G.anode_length * 0.225;
  var mat = new BABYLON.StandardMaterial("beamMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.6, 0.8, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  cone.material = mat;
  cone.renderingGroupId = 1;
  return { cone: cone, mat: mat };
}

// ============================================================
// B-FIELD TORI -- mu0*I/(2*pi*r) visualization
// ============================================================

function buildBFieldTori(scene, G) {
  var bRings = [];
  var N_RINGS = 5;
  var mat = new BABYLON.StandardMaterial("bFieldMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.3, 0.6, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;

  for (var i = 0; i < N_RINGS; i++) {
    var frac = (i + 1) / (N_RINGS + 1);
    var ringR = G.anode_radius + (G.cathode_radius - G.anode_radius) * frac;
    var ring = BABYLON.MeshBuilder.CreateTorus("bRing" + i, {
      diameter: ringR * 2,
      thickness: G.anode_radius * 0.010,
      tessellation: 32,
    }, scene);
    ring.rotation.z = Math.PI / 2;
    ring.position.x = G.anode_length * 0.5;
    ring.material = mat;
    ring.renderingGroupId = 1;
    ring.isVisible = false;
    bRings.push(ring);
  }
  return { bRings: bRings, mat: mat };
}

// ============================================================
// PARTICLES -- 3000 capacity, additive blend
// ============================================================

function buildParticles(scene, G) {
  var emitter = new BABYLON.AbstractMesh("psEmitter", scene);
  emitter.position.x = G.anode_length;
  var ps = new BABYLON.ParticleSystem("sparks", 3000, scene);
  ps.emitter = emitter;
  ps.createSphereEmitter(G.anode_radius * 0.2);
  ps.color1 = new BABYLON.Color4(0.3, 0.6, 1.0, 0.9);
  ps.color2 = new BABYLON.Color4(1.0, 0.8, 0.3, 0.7);
  ps.colorDead = new BABYLON.Color4(0.1, 0.05, 0.02, 0);
  ps.minSize = G.cathode_radius * 0.003;
  ps.maxSize = G.cathode_radius * 0.012;
  ps.minLifeTime = 0.10;
  ps.maxLifeTime = 0.45;
  ps.emitRate = 0;
  ps.gravity = new BABYLON.Vector3(G.cathode_radius * 2, 0, 0);
  ps.minEmitPower = G.cathode_radius * 0.3;
  ps.maxEmitPower = G.cathode_radius * 1.5;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.start();
  return { ps: ps, emitter: emitter };
}

// ============================================================
// CYLINDRICAL HEATMAP WRAP at midplane radius
// ============================================================

function buildHeatmapCylinder(scene, G) {
  var midR = (G.anode_radius + G.cathode_radius) / 2;
  var nCirc = 48, nZ = 32;
  var paths = [];
  for (var iz = 0; iz <= nZ; iz++) {
    var z = G.anode_length * iz / nZ;
    var ring = [];
    for (var ic = 0; ic <= nCirc; ic++) {
      var angle = (ic / nCirc) * Math.PI * 2;
      ring.push(new BABYLON.Vector3(z, midR * Math.sin(angle), midR * Math.cos(angle)));
    }
    paths.push(ring);
  }
  var tube = BABYLON.MeshBuilder.CreateRibbon("heatPlane", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false,
  }, scene);
  tube.isVisible = false;
  tube.isPickable = false;
  var mat = new BABYLON.StandardMaterial("heatMat", scene);
  mat.disableLighting = true;
  mat.backFaceCulling = false;
  mat.alpha = 0.55;
  tube.material = mat;
  tube.renderingGroupId = 2;
  return { plane: tube, mat: mat };
}

// ============================================================
// SNAP CACHE for MHD field heatmaps
// ============================================================

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
        rgba[pi + 3] = 160;
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

// ============================================================
// PIPELINE -- bloom, SSAO, glow with cinematic tuning
// ============================================================

function buildPipeline(scene, cam) {
  var pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.15;
  pipe.bloomThreshold = 0.80;
  pipe.bloomKernel = 80;
  pipe.bloomScale = 0.5;
  pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.0;

  var ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.45;
    ssao.radius = 1.6;
    ssao.samples = 14;
    ssao.base = 0.12;
  } catch (_) {}

  var glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 56, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.30;
  var glowNames = new Set([
    "sheathDisk", "pinchCore", "pinchMantle", "beamCone", "gasGlow",
  ]);
  glowLayer.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (glowNames.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      var ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else if (mesh.name && mesh.name.indexOf("bRing") === 0 && mesh.material) {
      var ec2 = mesh.material.emissiveColor;
      result.set(ec2.r, ec2.g, ec2.b, (mesh.material.alpha || 0) * 0.6);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline: pipe, ssao: ssao, glowLayer: glowLayer };
}

// ============================================================
// CINEMATIC CAMERA CONTROLLER
// ============================================================

function buildCinematicCamera(scene, canvas, G) {
  var cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI * 0.25, Math.PI * 0.30, G.cathode_radius * 9,
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
  cam.inertia = 0.92;

  var autoOrbit = true, interacting = false, orbitTimeout = null;
  var targetRadius = cam.radius;
  var targetBeta = cam.beta;
  var targetAlpha = cam.alpha;
  var cinematicActive = false;

  canvas.addEventListener("pointerdown", function() {
    interacting = true; autoOrbit = false; cinematicActive = false;
    if (orbitTimeout) clearTimeout(orbitTimeout);
  });
  canvas.addEventListener("pointerup", function() {
    interacting = false;
    orbitTimeout = setTimeout(function() { autoOrbit = true; }, 5000);
  });

  scene.registerBeforeRender(function() {
    if (cinematicActive && !interacting) {
      cam.radius = lerp(cam.radius, targetRadius, 0.015);
      cam.beta = lerp(cam.beta, targetBeta, 0.012);
      cam.target.x = lerp(cam.target.x, G.anode_length * 0.7, 0.010);
    } else if (autoOrbit && !interacting) {
      cam.alpha += 0.0006;
    }
  });

  return {
    cam: cam,
    setCinematic: function(phase, pI) {
      if (interacting) return;
      if (phase === "radial" || phase === "mhd_radial" || phase === "pinch") {
        cinematicActive = true;
        targetRadius = G.cathode_radius * lerp(8, 4.5, pI);
        targetBeta = lerp(Math.PI * 0.30, Math.PI * 0.38, pI);
      } else if (phase === "post_pinch") {
        cinematicActive = true;
        targetRadius = G.cathode_radius * 5;
        targetBeta = Math.PI * 0.35;
      } else {
        cinematicActive = false;
      }
    }
  };
}

// ============================================================
// MAIN SCENE
// ============================================================

async function createDPFScene(canvas, data) {
  var L = data, G = L.geometry, S = L.sheath;
  var result = await initEngine(canvas);
  var engine = result.engine, gpuBackend = result.gpuBackend;
  var scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.06, 0.06, 0.08, 1);
  scene.setRenderingAutoClearDepthStencil(1, true, true, false);
  scene.setRenderingAutoClearDepthStencil(2, false, false, false);

  var camCtrl = buildCinematicCamera(scene, canvas, G);
  var cam = camCtrl.cam;

  var key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  key.intensity = 0.7;
  key.diffuse = new BABYLON.Color3(0.90, 0.92, 0.98);
  var fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.30;
  fill.diffuse = new BABYLON.Color3(0.80, 0.85, 0.95);
  fill.groundColor = new BABYLON.Color3(0.12, 0.12, 0.16);

  var dev = buildDevice(scene, G);
  var sheath = buildSheathTorus(scene, G);
  var gas = buildGasGlow(scene, G);
  var pinch = buildPinchColumn(scene, G);
  var beam = buildBeamCone(scene, G);
  var bField = buildBFieldTori(scene, G);
  var heat = buildHeatmapCylinder(scene, G);
  var parts = buildParticles(scene, G);
  var pipeResult = buildPipeline(scene, cam);
  var pipeline = pipeResult.pipeline;
  var ssao = pipeResult.ssao;
  var glowLayer = pipeResult.glowLayer;

  var snapCache = {};
  var lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
  var heatTex = null;
  buildSnapCache("density", L.density, snapCache);
  buildSnapCache("temperature", L.temperature, snapCache);
  buildSnapCache("bfield", L.bfield, snapCache);

  function applySnapTex(ovKey) {
    var c = snapCache[ovKey];
    if (!c) return;
    var idx = lastSnapIdx[ovKey];
    if (idx < 0 || idx >= c.rgba.length) return;
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(c.rgba[idx], c.texW, c.texH,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene, false, false,
      BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.55;
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
    var vals = b64ToFloat32(layer.data);
    var nr = layer.shape[0], nz = layer.shape[1];
    var rgba = new Uint8Array(nz * nr * 4);
    for (var ir = 0; ir < nr; ir++) {
      for (var iz = 0; iz < nz; iz++) {
        var v = vals[ir * nz + iz];
        var c = cmapLookup(v, activeCmap);
        var pi = ((nr - 1 - ir) * nz + iz) * 4;
        rgba[pi]     = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 160;
      }
    }
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(rgba, nz, nr, BABYLON.Engine.TEXTUREFORMAT_RGBA,
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
    var col = PHASE_COLORS_COOL[f.phase] || [0.20, 0.50, 1.00];
    var rippleAmp = 0;

    camCtrl.setCinematic(f.phase, pI);

    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      var ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // === SHEATH TORUS ===
    if (Ifrac > 0.01) {
      sheath.disk.isVisible = true;
      sheath.disk.position.x = isP ? G.anode_length : f.z;

      var sheathAlpha = clamp01(Ifrac * 0.30 + pI * 0.15);
      sheath.mat.alpha = sheathAlpha;
      sheath.mat.emissiveColor.set(col[0], col[1], col[2]);

      sheath.fresnelParams.leftColor.set(col[0], col[1], col[2]);
      sheath.fresnelParams.bias = lerp(0.1, 0.3, Ifrac);
      sheath.fresnelParams.power = lerp(2.0, 1.2, pI);

      if (isP) {
        var scaleF = Math.max(0.08, cr);
        sheath.disk.scaling.y = scaleF;
        sheath.disk.scaling.z = scaleF;
      } else {
        sheath.disk.scaling.y = 1;
        sheath.disk.scaling.z = 1;
      }
    } else {
      sheath.disk.isVisible = false;
    }

    // === GAS GLOW ===
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      gas.glow.isVisible = true;
      var extent = isP ? G.anode_length : f.z;
      gas.glow.scaling.x = extent / G.anode_length;
      gas.glow.position.x = extent / 2;
      gas.mat.alpha = clamp01(Ifrac * 0.035);
      gas.mat.emissiveColor.set(col[0] * 0.20, col[1] * 0.20, col[2] * 0.35);
    } else {
      gas.glow.isVisible = false;
    }

    // === PINCH COLUMN with Bennett profile + m=0 instability ===
    var showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                     f.phase === "reflected") || (isP && cr < 0.35);
    if (showPinch && pI > 0.03) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      var columnR = Math.max(G.anode_radius * 0.008, cr * G.cathode_radius * 0.10);
      rippleAmp = f.phase === "post_pinch" ? 0.40 : 0;

      var waveLen = 6.2832 * columnR;
      var waveNum = Math.min(6, Math.max(1,
        Math.round(pinch.columnLen / Math.max(waveLen, 0.001))));

      for (var k = 0; k <= pinch.N; k++) {
        var zf = k / pinch.N;
        var taper = Math.sin(Math.PI * zf);
        var bennett = 1.0 / (1.0 + Math.pow((zf - 0.5) * 3.5, 2));
        var lr = columnR * (0.20 + 0.80 * taper * bennett);
        var ripple = rippleAmp * lr * Math.cos(6.2832 * waveNum * zf + f.t * 4);
        pinch.radii[k] = Math.max(0.0002, lr + ripple);
      }

      BABYLON.MeshBuilder.CreateTube("pinchCore", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j] * 0.35; },
        tessellation: 14, cap: BABYLON.Mesh.CAP_ALL, instance: pinch.core,
      });
      BABYLON.MeshBuilder.CreateTube("pinchMantle", {
        path: pinch.path,
        radiusFunction: function(j) { return pinch.radii[j]; },
        tessellation: 18, cap: BABYLON.Mesh.NO_CAP,
        sideOrientation: BABYLON.Mesh.DOUBLESIDE, instance: pinch.mantle,
      });

      pinch.coreMat.alpha = clamp01(pI * 0.90);
      pinch.mantleMat.alpha = clamp01(pI * 0.22);

      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(1.0, 0.98, 0.92);
      } else {
        pinch.coreMat.emissiveColor.set(
          lerp(0.5, 1.0, pI), lerp(0.35, 0.98, pI), lerp(0.15, 0.92, pI));
      }
      pinch.mantleMat.emissiveColor.set(
        lerp(col[0] * 0.8, 1.0, pI),
        lerp(col[1] * 0.5, 0.45, pI),
        lerp(col[2] * 0.2, 0.10, pI));
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // === BEAM CONE ===
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.08;
    beam.mat.alpha = beam.cone.isVisible ? clamp01(pI * 0.30) : 0;

    // === B-FIELD TORI (behind sheath only, brightness proportional to I) ===
    var sheathZ = isP ? G.anode_length : f.z;
    var showBField = Ifrac > 0.05;
    for (var bi = 0; bi < bField.bRings.length; bi++) {
      var ringFrac = (bi + 1) / (bField.bRings.length + 1);
      var ringZ = sheathZ * ringFrac * 0.85;
      bField.bRings[bi].isVisible = showBField && ringZ < sheathZ * 0.95;
      if (bField.bRings[bi].isVisible) {
        bField.bRings[bi].position.x = ringZ;
        if (isP) {
          var ringScale = Math.max(0.08, cr);
          bField.bRings[bi].scaling.y = ringScale;
          bField.bRings[bi].scaling.z = ringScale;
        } else {
          bField.bRings[bi].scaling.y = 1;
          bField.bRings[bi].scaling.z = 1;
        }
      }
    }
    if (showBField) {
      bField.mat.alpha = clamp01(Ifrac * 0.10);
      var drift = f.t * 0.3;
      for (var bi2 = 0; bi2 < bField.bRings.length; bi2++) {
        bField.bRings[bi2].rotation.x = drift + bi2 * 0.5;
      }
      var bIntensity = Ifrac;
      bField.mat.emissiveColor.set(
        lerp(0.15, col[0] * 0.6, bIntensity),
        lerp(0.30, col[1] * 0.7, bIntensity),
        lerp(0.70, col[2], bIntensity));
    }

    // === PARTICLES ===
    if (f.phase === "rundown") {
      parts.ps.emitRate = Math.round(Ifrac * 400);
      parts.emitter.position.x = f.z;
      parts.ps.gravity.set(G.cathode_radius * 3, 0, 0);
      parts.ps.createSphereEmitter(G.cathode_radius * 0.8);
    } else if (isP && pI > 0.1) {
      parts.ps.emitRate = Math.round(pI * 1200);
      parts.emitter.position.x = G.anode_length;
      parts.ps.gravity.set(0, 0, 0);
      parts.ps.createSphereEmitter(G.anode_radius * cr * 0.5);
    } else if (showPinch && pI > 0.2) {
      parts.ps.emitRate = Math.round(pI * 2000);
      parts.emitter.position.x = G.anode_length;
      parts.ps.gravity.set(G.cathode_radius * 5, 0, 0);
      parts.ps.createSphereEmitter(G.anode_radius * 0.05);
    } else {
      parts.ps.emitRate = 0;
    }

    // Particle color follows phase
    parts.ps.color1.set(col[0], col[1], col[2], 0.85);
    parts.ps.color2.set(
      Math.min(1, col[0] + 0.3),
      Math.min(1, col[1] + 0.2),
      Math.min(1, col[2] + 0.1), 0.6);

    // === ANODE THERMAL GLOW ===
    if (pI > 0.3) {
      dev.copperMat.emissiveColor.set(
        0.04 + (pI - 0.3) * 0.4, 0.02 + (pI - 0.3) * 0.15, 0.01);
    } else {
      dev.copperMat.emissiveColor.set(0.04, 0.02, 0.01);
    }

    // === CINEMATIC PIPELINE TUNING ===
    glowLayer.intensity = lerp(0.25, 0.65, pI);
    pipeline.bloomWeight = lerp(0.12, 0.35, pI);

    var bgBlue = lerp(0.08, 0.04, pI);
    var bgWarm = lerp(0.06, 0.05, pI);
    scene.clearColor.set(bgWarm, bgWarm, bgBlue, 1);

    return { f: f, isP: isP, cr: cr, pI: pI, rippleAmp: rippleAmp };
  }

  // ============================================================
  // RETURN API
  // ============================================================
  return {
    engine: engine, scene: scene, camera: cam, gpuBackend: gpuBackend,
    useGPU: gpuBackend === "WebGPU",
    G: G, S: S, L: L,
    anode: dev.anode, cathodeRods: dev.cathodeRods, insulator: dev.insulator,
    sheathDisk: sheath.disk, pinchCore: pinch.core, pinchMantle: pinch.mantle,
    beamCone: beam.cone, gasGlow: gas.glow,
    bRings: bField.bRings, fieldLines: [],
    ps: { start: function() { parts.ps.start(); }, stop: function() { parts.ps.stop(); } },
    pipeline: pipeline, ssao: ssao, glowLayer: glowLayer,
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
