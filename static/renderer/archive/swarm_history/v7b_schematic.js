/**
 * DPF v7b Schematic Renderer — Clean Schematic Physics Animation
 *
 * Proportionally accurate PF-1000 geometry with physics-driven animation.
 * Wireframe device, torus sheath spanning anode-cathode, B-field tori,
 * Bennett pinch column, 3000 additive particles, cylindrical heatmap wrap.
 *
 * Babylon.js 8.x, StandardMaterial only, dark clearColor, under 950 lines.
 */

// ============================================================
// COLORMAPS
// ============================================================

const VIRIDIS = [
  [0.267,0.004,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],[0.207,0.372,0.553],
  [0.164,0.471,0.558],[0.128,0.567,0.551],[0.134,0.658,0.517],[0.267,0.749,0.441],
  [0.478,0.821,0.318],[0.741,0.873,0.150],[0.993,0.906,0.144]
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
  pinch:       "Pinch \u2014 peak compression",
  post_pinch:  "Post-pinch disruption",
  none:        "",
};

const PHASE_DESCRIPTIONS = {
  rundown:     "Current sheath sweeps neutral gas from insulator to anode tip \u2014 magnetic snowplow",
  radial:      "Magnetic pressure compresses plasma ring inward toward the axis",
  mhd_radial:  "J x B force drives radial implosion \u2014 compression heating the plasma",
  mhd:         "Full MHD simulation of plasma dynamics",
  reflected:   "Reflected shock expands outward after axis convergence",
  pinch:       "PEAK COMPRESSION \u2014 fusion-relevant conditions at the axis",
  post_pinch:  "m=0 sausage instability breaks up the plasma column",
};

const SPEEDS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];

const GLOW_NAMES = new Set([
  "sheathTorus", "pinchCore", "pinchMantle", "bfieldTorus"
]);

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
  return phase === "radial" || phase === "mhd_radial" ||
         phase === "pinch" || phase === "reflected" || phase === "post_pinch";
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

function lerp(a, b, t) { return a + (b - a) * t; }

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
    } catch (_) { /* fall through */ }
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
// CAMERA — stable, study-friendly
// ============================================================

function createCamera(scene, canvas, G) {
  var cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI / 4, Math.PI / 3, G.cathode_radius * 12,
    new BABYLON.Vector3(G.anode_length * 0.5, 0, 0), scene);
  cam.attachControl(canvas, false);
  cam.inputs.removeByType("ArcRotateCameraMouseWheelInput");
  canvas.addEventListener("wheel", function(e) {
    e.preventDefault();
    cam.radius -= e.deltaY * 0.05;
    cam.radius = Math.max(cam.lowerRadiusLimit, Math.min(cam.upperRadiusLimit, cam.radius));
  }, { passive: false });
  cam.lowerRadiusLimit = G.anode_radius * 2;
  cam.upperRadiusLimit = G.cathode_radius * 60;
  cam.pinchPrecision = 15;
  cam.panningSensibility = 60;
  cam.minZ = 0.0005;
  cam.inertia = 0.92;

  var autoOrbit = true, userInteracting = false, timeout = null;
  canvas.addEventListener("pointerdown", function() {
    userInteracting = true; autoOrbit = false;
    if (timeout) clearTimeout(timeout);
  });
  canvas.addEventListener("pointerup", function() {
    userInteracting = false;
    timeout = setTimeout(function() { autoOrbit = true; }, 6000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !userInteracting) cam.alpha += 0.0006;
  });
  return cam;
}

// ============================================================
// LIGHTS
// ============================================================

function createLights(scene) {
  var key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  key.intensity = 1.2;
  key.diffuse = new BABYLON.Color3(1, 0.98, 0.95);

  var back = new BABYLON.DirectionalLight("back", new BABYLON.Vector3(1, -1, -1), scene);
  back.intensity = 0.5;
  back.diffuse = new BABYLON.Color3(0.9, 0.92, 0.95);

  var fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.4;
  fill.diffuse = new BABYLON.Color3(0.85, 0.88, 0.95);
  fill.groundColor = new BABYLON.Color3(0.3, 0.3, 0.35);
}

// ============================================================
// WIREFRAME DEVICE — anode, cathode rods, end rings, insulator
// ============================================================

function buildDevice(scene, G) {
  var wireMat = new BABYLON.StandardMaterial("wireMat", scene);
  wireMat.diffuseColor = new BABYLON.Color3(0.5, 0.5, 0.55);
  wireMat.emissiveColor = new BABYLON.Color3(0.15, 0.15, 0.18);
  wireMat.specularColor = new BABYLON.Color3(0.3, 0.3, 0.3);
  wireMat.wireframe = true;
  wireMat.alpha = 0.3;

  var anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 32, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = wireMat;
  anode.renderingGroupId = 0;

  var rodMat = new BABYLON.StandardMaterial("rodMat", scene);
  rodMat.diffuseColor = new BABYLON.Color3(0.45, 0.45, 0.5);
  rodMat.emissiveColor = new BABYLON.Color3(0.12, 0.12, 0.15);
  rodMat.specularColor = new BABYLON.Color3(0.4, 0.4, 0.4);
  rodMat.alpha = 0.5;

  var N_RODS = G.n_cathode_rods || 8;
  var rodDiam = G.cathode_rod_diameter || G.cathode_radius * 0.05;
  var cathodeRods = [];
  for (var i = 0; i < N_RODS; i++) {
    var angle = (i / N_RODS) * Math.PI * 2;
    var rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: rodDiam, height: G.anode_length * 1.05, tessellation: 8,
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

  var ringThk = (G.cathode_radius - G.anode_radius) * 0.15;
  var baseRing = BABYLON.MeshBuilder.CreateTorus("cathodeBase", {
    diameter: G.cathode_radius * 2, thickness: ringThk, tessellation: 48,
  }, scene);
  baseRing.rotation.z = Math.PI / 2;
  baseRing.position.x = -G.anode_length * 0.025;
  baseRing.material = rodMat;
  baseRing.renderingGroupId = 0;
  cathodeRods.push(baseRing);

  var topRing = baseRing.clone("cathodeTop");
  topRing.position.x = G.anode_length * 1.025;
  cathodeRods.push(topRing);

  var ceramicMat = new BABYLON.StandardMaterial("ceramic", scene);
  ceramicMat.diffuseColor = new BABYLON.Color3(0.85, 0.82, 0.75);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.08, 0.07, 0.06);
  ceramicMat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.1);
  ceramicMat.alpha = 0.6;
  var insThk = G.insulator_thickness || G.anode_radius * 0.15;
  var insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.35;
  var insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 48,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  return { anode: anode, cathodeRods: cathodeRods, insulator: insulator, wireMat: wireMat, rodMat: rodMat };
}

// ============================================================
// SHEATH TORUS — spans anode to cathode surface exactly
// ============================================================

function buildSheathTorus(scene, G) {
  var midR = (G.anode_radius + G.cathode_radius) / 2;
  var tubeR = (G.cathode_radius - G.anode_radius) / 2;
  var torus = BABYLON.MeshBuilder.CreateTorus("sheathTorus", {
    diameter: midR * 2, thickness: tubeR * 2, tessellation: 48,
  }, scene);
  torus.rotation.z = Math.PI / 2;
  torus.position.x = 0;
  var mat = new BABYLON.StandardMaterial("sheathMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.2, 0.5, 1.0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  torus.material = mat;
  torus.renderingGroupId = 1;
  return { torus: torus, mat: mat, baseMidR: midR, baseTubeR: tubeR };
}

// ============================================================
// B-FIELD TORI — 4-5 tori trailing the sheath
// ============================================================

function buildBFieldTori(scene, G) {
  var count = 5;
  var tori = [];
  var mats = [];
  var midR = (G.anode_radius + G.cathode_radius) / 2;
  var tubeR = (G.cathode_radius - G.anode_radius) * 0.12;
  for (var i = 0; i < count; i++) {
    var t = BABYLON.MeshBuilder.CreateTorus("bfieldTorus" + i, {
      diameter: midR * 2, thickness: tubeR * 2, tessellation: 32,
    }, scene);
    t.rotation.z = Math.PI / 2;
    t.isVisible = false;
    var m = new BABYLON.StandardMaterial("bfieldMat" + i, scene);
    m.emissiveColor = new BABYLON.Color3(0.15, 0.35, 0.8);
    m.disableLighting = true;
    m.alpha = 0;
    m.alphaMode = BABYLON.Engine.ALPHA_ADD;
    m.backFaceCulling = false;
    t.material = m;
    t.renderingGroupId = 1;
    tori.push(t);
    mats.push(m);
  }
  return { tori: tori, mats: mats, baseMidR: midR, baseTubeR: tubeR, count: count };
}

// ============================================================
// PINCH COLUMN — Bennett profile, m=0 ripple post-pinch
// ============================================================

function buildPinchColumn(scene, G) {
  var N = 20;
  var columnLen = G.anode_length * 0.2;
  var tipX = G.anode_length;
  var path = [];
  for (var k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(tipX + columnLen * k / N, 0, 0));
  }
  var radii = new Array(N + 1).fill(G.anode_radius * 0.1);

  var coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(1, 1, 0.9);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;

  var core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path: path,
    radiusFunction: function(i) { return radii[i] * 0.3; },
    tessellation: 16, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;
  core.isVisible = false;

  var mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(1, 0.4, 0.1);
  mantleMat.disableLighting = true;
  mantleMat.alpha = 0;
  mantleMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mantleMat.backFaceCulling = false;

  var mantle = BABYLON.MeshBuilder.CreateTube("pinchMantle", {
    path: path,
    radiusFunction: function(i) { return radii[i]; },
    tessellation: 20, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  mantle.material = mantleMat;
  mantle.renderingGroupId = 1;
  mantle.isVisible = false;

  return { core: core, mantle: mantle, coreMat: coreMat, mantleMat: mantleMat, radii: radii, path: path, N: N, columnLen: columnLen };
}

// ============================================================
// PARTICLES — 3000, additive blend, follow sheath
// ============================================================

function buildParticles(scene, G) {
  var count = 3000;
  var SPS = new BABYLON.SolidParticleSystem("sps", scene, { updatable: true });
  var model = BABYLON.MeshBuilder.CreateBox("pBox", { size: 0.001 }, scene);
  SPS.addShape(model, count);
  model.dispose();
  var mesh = SPS.buildMesh();
  mesh.renderingGroupId = 1;

  var pMat = new BABYLON.StandardMaterial("particleMat", scene);
  pMat.emissiveColor = new BABYLON.Color3(0.4, 0.6, 1.0);
  pMat.disableLighting = true;
  pMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  pMat.backFaceCulling = false;
  mesh.material = pMat;
  mesh.hasVertexAlpha = true;

  var pData = [];
  for (var i = 0; i < count; i++) {
    pData.push({
      angle: Math.random() * Math.PI * 2,
      zFrac: Math.random(),
      rFrac: Math.random(),
      speed: 0.5 + Math.random() * 1.5,
      phase: Math.random() * Math.PI * 2,
    });
  }

  SPS.initParticles = function() {
    for (var i = 0; i < SPS.nbParticles; i++) {
      var p = SPS.particles[i];
      p.position.set(0, 0, 0);
      p.scaling.set(1, 1, 1);
      p.color = new BABYLON.Color4(0.4, 0.6, 1.0, 0);
    }
  };
  SPS.initParticles();
  SPS.setParticles();

  return { SPS: SPS, mesh: mesh, mat: pMat, data: pData, count: count };
}

// ============================================================
// CYLINDRICAL HEATMAP WRAP — 360-degree at midplane radius
// ============================================================

function buildHeatmapCylinder(scene, G) {
  var nTheta = 64, nZ = 32;
  var midR = (G.anode_radius + G.cathode_radius) / 2;
  var paths = [];
  for (var it = 0; it <= nTheta; it++) {
    var theta = (it / nTheta) * Math.PI * 2;
    var row = [];
    for (var iz = 0; iz <= nZ; iz++) {
      var z = G.anode_length * iz / nZ;
      row.push(new BABYLON.Vector3(z, midR * Math.sin(theta), midR * Math.cos(theta)));
    }
    paths.push(row);
  }
  var cyl = BABYLON.MeshBuilder.CreateRibbon("heatCyl", {
    pathArray: paths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false,
  }, scene);
  cyl.isVisible = false;
  cyl.isPickable = false;
  var mat = new BABYLON.StandardMaterial("heatMat", scene);
  mat.disableLighting = true;
  mat.backFaceCulling = false;
  mat.alpha = 0.55;
  cyl.material = mat;
  cyl.renderingGroupId = 1;
  return { cyl: cyl, mat: mat };
}

// ============================================================
// SNAP CACHE — precompute RGBA frames for heatmap animation
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
        rgba[pi + 3] = 150;
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
// POST-PROCESSING
// ============================================================

function buildPipeline(scene, cam) {
  var pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.15;
  pipe.bloomThreshold = 0.8;
  pipe.bloomKernel = 48;
  pipe.bloomScale = 0.5;
  pipe.fxaaEnabled = true;
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = false;
  pipe.imageProcessing.exposure = 1.0;
  pipe.sharpenEnabled = true;
  pipe.sharpen.edgeAmount = 0.12;

  var glow = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 24, mainTextureFixedSize: 512,
  });
  glow.intensity = 0.45;
  glow.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (GLOW_NAMES.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      var ec = mesh.material.emissiveColor;
      result.set(ec.r, ec.g, ec.b, mesh.material.alpha || 0);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline: pipe, glowLayer: glow };
}

// ============================================================
// MAIN: createDPFScene(canvas, data)
// ============================================================

async function createDPFScene(canvas, data) {
  var L = data;
  var G = L.geometry;
  var S = L.sheath;

  var eng = await initEngine(canvas);
  var engine = eng.engine;
  var scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.06, 0.06, 0.08, 1);

  var camera = createCamera(scene, canvas, G);
  createLights(scene);

  var dev = buildDevice(scene, G);
  var sheath = buildSheathTorus(scene, G);
  var bfield = buildBFieldTori(scene, G);
  var pinch = buildPinchColumn(scene, G);
  var particles = buildParticles(scene, G);
  var heat = buildHeatmapCylinder(scene, G);
  var post = buildPipeline(scene, camera);

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
    heatTex = new BABYLON.RawTexture(
      c.rgba[idx], c.texW, c.texH,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene,
      false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE
    );
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.55;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.cyl.isVisible = true;
  }

  var activeOverlay = "none";

  function updateHeatmap(key) {
    if (!L || key === "none") { heat.cyl.isVisible = false; return; }
    var layer = L[key];
    if (!layer || (!layer.data && !layer.frames)) { heat.cyl.isVisible = false; return; }
    if (snapCache[key] && lastSnapIdx[key] >= 0) { applySnapTex(key); return; }
    if (!layer.data || !layer.shape) { heat.cyl.isVisible = false; return; }
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
        rgba[pi + 3] = 150;
      }
    }
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(
      rgba, nz, nr, BABYLON.Engine.TEXTUREFORMAT_RGBA,
      scene, false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE
    );
    heat.mat.diffuseTexture = heatTex;
    heat.mat.emissiveTexture = heatTex;
    heat.mat.alpha = 0.55;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.cyl.isVisible = true;
  }

  // ---- Bennett profile: r_pinch / (1 + (r/a)^2) ----
  function bennettRadius(zFrac, baseR, rippleAmp) {
    var taper = 1.0 / (1.0 + Math.pow(2.0 * (zFrac - 0.5), 4));
    return baseR * taper + rippleAmp * Math.sin(zFrac * Math.PI * 6);
  }

  // ============================================================
  // applyFrame — drives all visuals from sheath data
  // ============================================================

  var frameTime = 0;

  function applyFrame(i) {
    if (i < 0 || i >= S.frames.length) return {};
    var f = S.frames[i];
    var isP = isRadialPhase(f.phase);
    var cr = Math.max(0.02, f.r / G.cathode_radius);
    var Ifrac = Math.abs(f.I / Math.max(S.I_peak, 0.001));
    var pI = isP ? clamp01(Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;
    frameTime = f.t || 0;

    var rippleAmp = 0;

    // Heatmap snap sync
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      var ni = nearestSnapIdx(snapCache, activeOverlay, f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    // --- SHEATH TORUS ---
    var col = PHASE_COLORS[f.phase] || [0.3, 0.5, 1.0];
    if (Ifrac > 0.02) {
      sheath.torus.isVisible = true;
      if (isP) {
        sheath.torus.position.x = G.anode_length;
        var currentMidR = lerp(sheath.baseMidR, G.anode_radius * 0.5, clamp01(1 - cr));
        var currentTubeR = lerp(sheath.baseTubeR, sheath.baseTubeR * 0.3, clamp01(1 - cr));
        var scaleYZ = currentMidR / sheath.baseMidR;
        sheath.torus.scaling.set(1, scaleYZ, scaleYZ);
      } else {
        sheath.torus.position.x = f.z;
        sheath.torus.scaling.set(1, 1, 1);
      }
      sheath.mat.emissiveColor.set(col[0], col[1], col[2]);
      sheath.mat.alpha = clamp01(Ifrac * 0.45);
    } else {
      sheath.torus.isVisible = false;
    }

    // --- B-FIELD TORI (trailing sheath) ---
    var sheathZ = isP ? G.anode_length : f.z;
    for (var bi = 0; bi < bfield.count; bi++) {
      var lag = (bi + 1) * G.anode_length * 0.08;
      var bz = sheathZ - lag;
      if (bz < 0 || Ifrac < 0.05) {
        bfield.tori[bi].isVisible = false;
        continue;
      }
      bfield.tori[bi].isVisible = true;
      bfield.tori[bi].position.x = bz;
      var fade = clamp01(1 - (bi + 1) / (bfield.count + 1));
      var brightness = Ifrac * fade * 0.35;
      bfield.mats[bi].alpha = brightness;
      bfield.mats[bi].emissiveColor.set(
        0.15 * fade, 0.35 * fade, 0.8 * fade
      );
      if (isP) {
        var bScale = lerp(1, cr, 0.5);
        bfield.tori[bi].scaling.set(1, bScale, bScale);
      } else {
        bfield.tori[bi].scaling.set(1, 1, 1);
      }
    }

    // --- PINCH COLUMN: Bennett profile, m=0 ripple ---
    var pinchRadius = cr * G.cathode_radius * 0.1;
    var showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                     f.phase === "reflected") || (isP && cr < 0.25);
    if (showPinch && pI > 0.05) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      rippleAmp = (f.phase === "post_pinch") ? pinchRadius * 0.35 : 0;

      for (var k = 0; k <= pinch.N; k++) {
        var zf = k / pinch.N;
        pinch.radii[k] = Math.max(0.0005, bennettRadius(zf, pinchRadius, rippleAmp));
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

      if (f.phase === "pinch") {
        pinch.coreMat.emissiveColor.set(1.0, 1.0, 0.9);
        pinch.coreMat.alpha = clamp01(pI * 0.85);
        pinch.mantleMat.emissiveColor.set(1.0, 0.6, 0.2);
        pinch.mantleMat.alpha = clamp01(pI * 0.25);
      } else if (f.phase === "post_pinch") {
        pinch.coreMat.emissiveColor.set(0.8, 0.3, 0.1);
        pinch.coreMat.alpha = clamp01(pI * 0.5);
        pinch.mantleMat.emissiveColor.set(0.5, 0.15, 0.05);
        pinch.mantleMat.alpha = clamp01(pI * 0.12);
      } else {
        var h = clamp01(pI);
        pinch.coreMat.emissiveColor.set(0.5 + h * 0.5, 0.4 + h * 0.6, 0.3 + h * 0.6);
        pinch.coreMat.alpha = clamp01(pI * 0.6);
        pinch.mantleMat.emissiveColor.set(pI * 0.8, pI * 0.3, pI * 0.08);
        pinch.mantleMat.alpha = clamp01(pI * 0.15);
      }
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // --- PARTICLES: 3000 following sheath ---
    var pd = particles.data;
    var sps = particles.SPS;
    var sheathR = isP ? f.r : G.cathode_radius;
    var sheathX = isP ? G.anode_length : f.z;
    var particleSpread = isP ? G.anode_length * 0.05 : G.anode_length * 0.15;

    particles.mat.emissiveColor.set(col[0], col[1], col[2]);

    sps.updateParticle = function(p) {
      var d = pd[p.idx];
      if (Ifrac < 0.02) {
        p.color.a = 0;
        return p;
      }
      d.angle += d.speed * 0.02;
      var r = sheathR * (0.85 + d.rFrac * 0.3);
      var zOff = (d.zFrac - 0.5) * particleSpread;
      p.position.x = sheathX + zOff;
      p.position.y = r * Math.sin(d.angle);
      p.position.z = r * Math.cos(d.angle);
      var flicker = 0.5 + 0.5 * Math.sin(d.phase + frameTime * d.speed * 10);
      p.color.a = Ifrac * flicker * 0.3;
      var sz = 0.6 + flicker * 0.4;
      p.scaling.set(sz, sz, sz);
      return p;
    };
    sps.setParticles();

    return {
      f: f,
      isP: isP,
      cr: cr,
      pI: pI,
      rippleAmp: rippleAmp,
    };
  }

  // ---- Render loop ----
  engine.runRenderLoop(function() { scene.render(); });
  window.addEventListener("resize", function() { engine.resize(); });

  // ---- Return full API ----
  return {
    engine: engine,
    scene: scene,
    camera: camera,
    gpuBackend: eng.gpuBackend,

    // Mesh refs
    device: dev,
    sheath: sheath,
    bfield: bfield,
    pinch: pinch,
    particles: particles,
    heatmap: heat,
    postProcessing: post,

    // Frame driver
    applyFrame: applyFrame,
    frameCount: S.frames.length,

    // Heatmap overlay control
    setOverlay: function(key) {
      activeOverlay = key;
      updateHeatmap(key);
    },

    // Colormap control
    setColormap: function(name) {
      activeCmap = (name === "viridis") ? VIRIDIS : VIRIDIS;
    },

    // Dispose
    dispose: function() {
      if (heatTex) heatTex.dispose();
      particles.SPS.dispose();
      scene.dispose();
      engine.dispose();
    },
  };
}

// ============================================================
// EXPORTS
// ============================================================

window.createDPFScene = createDPFScene;
window.PHASE_LABELS = PHASE_LABELS;
window.PHASE_DESCRIPTIONS = PHASE_DESCRIPTIONS;
window.SPEEDS = SPEEDS;
