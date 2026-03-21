/**
 * DPF-Unified Plasma Renderer v2 -- Babylon.js 8.x
 *
 * Modular rewrite: initEngine, createLights, createDevice, createPlasma,
 * createGrid, plus snap cache, heatmap, field lines, particles, and animator.
 *
 * Data sources:
 *   LEE MODEL (0D): sheath z/r/I/phase, pinch radius, particle emitter
 *   MHD FIELDS (2D): midplane heatmap, poloidal field lines (when available)
 *
 * Rendering architecture:
 *   Group 0 (opaque):  electrodes, insulator, grid, scale bars
 *   Group 1 (additive): sheath, trail, pinch, halo, particles
 *   Group 2 (overlay):  heatmap plane
 *
 * ONE DefaultRenderingPipeline. ONE light set. No needDepthPrePass on plasma.
 */

// ============================================================
// CONSTANTS
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
  rundown: [0.2, 0.5, 1.0],
  radial: [1.0, 0.3, 0.08],
  mhd_radial: [1.0, 0.3, 0.08],
  reflected: [1.0, 0.55, 0.0],
  pinch: [1.0, 0.08, 0.03],
  post_pinch: [0.7, 0.15, 0.08],
};

const PHASE_LABELS = {
  rundown: "Axial rundown",
  radial: "Radial implosion",
  mhd_radial: "MHD radial implosion",
  mhd: "MHD simulation",
  reflected: "Reflected shock",
  pinch: "Pinch \u2014 peak compression",
  post_pinch: "Post-pinch disruption",
  none: "",
};

const PHASE_DESCRIPTIONS = {
  rundown: "Current sheath sweeping gas toward anode tip",
  radial: "Plasma ring compressing inward \u2014 magnetic piston",
  mhd_radial: "MHD radial implosion in progress",
  mhd: "Full MHD simulation \u2014 no Lee-phase snowplow",
  reflected: "Reflected shock expanding outward",
  pinch: "PEAK COMPRESSION \u2014 fusion zone active",
  post_pinch: "Pinch disrupting via m=0 instability",
};

const SPEEDS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];

const GLOW_MESHES = new Set(["sheath", "pinch", "halo", "trail"]);

const GAS_COLORS = {
  D2:  [0.3, 0.6, 1.0],
  H2:  [0.5, 0.3, 0.7],
  Ne:  [1.0, 0.5, 0.1],
  Ar:  [0.6, 0.2, 0.8],
  Kr:  [0.4, 0.8, 0.3],
  Xe:  [0.4, 0.5, 1.0],
  N2:  [0.6, 0.2, 0.6],
};

const MATERIAL_COLORS = {
  copper:          { albedo: [0.97, 0.75, 0.50], metallic: 0.95, roughness: 0.20 },
  tungsten:        { albedo: [0.65, 0.65, 0.68], metallic: 0.90, roughness: 0.35 },
  aluminum:        { albedo: [0.91, 0.92, 0.93], metallic: 0.85, roughness: 0.30 },
  stainless_steel: { albedo: [0.75, 0.75, 0.80], metallic: 0.85, roughness: 0.25 },
  brass:           { albedo: [0.88, 0.78, 0.50], metallic: 0.90, roughness: 0.25 },
};

const INSULATOR_COLORS = {
  alumina:      [0.95, 0.92, 0.85],
  pyrex:        [0.85, 0.90, 0.85],
  borosilicate: [0.88, 0.90, 0.88],
};

const HDR_ENV = "https://assets.babylonjs.com/environments/Studio_Softbox_2Umbrellas_cube_specular.env";

let activeCmap = VIRIDIS;

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function cmapLookup(v, cmapArr) {
  const t = Math.max(0, Math.min(1, v));
  const n = cmapArr.length - 1;
  const i = Math.min(n - 1, Math.max(0, Math.floor(t * n)));
  const f = t * n - i;
  const a = cmapArr[i], b = cmapArr[i + 1];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

function b64ToFloat32(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(buf);
}

function decodeBase64Float32(b64str, shape) {
  const raw = atob(b64str);
  const buf = new ArrayBuffer(raw.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) u8[i] = raw.charCodeAt(i);
  return { data: new Float32Array(buf), shape };
}

function bilinearSample(arr, nx, nz, fx, fz) {
  const ix = Math.min(nx - 2, Math.max(0, Math.floor(fx)));
  const iz = Math.min(nz - 2, Math.max(0, Math.floor(fz)));
  const dx = fx - ix, dz = fz - iz;
  return (1 - dx) * (1 - dz) * arr[ix * nz + iz] +
    dx * (1 - dz) * arr[(ix + 1) * nz + iz] +
    (1 - dx) * dz * arr[ix * nz + iz + 1] +
    dx * dz * arr[(ix + 1) * nz + iz + 1];
}

function isRadialPhase(phase) {
  return ["radial", "mhd_radial", "pinch", "reflected", "post_pinch"].indexOf(phase) >= 0;
}

function getGasColor(L) {
  const species = (L && L.gas_species) ? L.gas_species : "D2";
  return GAS_COLORS[species] || GAS_COLORS.D2;
}

// ============================================================
// initEngine(canvas) -- engine, scene, camera, environment
// ============================================================

async function initEngine(canvas) {
  let engine, gpuBackend = "WebGL2";
  const useWebGPU = (new URLSearchParams(window.location.search)).get("webgpu") === "1";

  if (useWebGPU) {
    try {
      if (await BABYLON.WebGPUEngine.IsSupportedAsync) {
        engine = new BABYLON.WebGPUEngine(canvas, {
          antialias: true, adaptToDeviceRatio: true, powerPreference: "high-performance",
        });
        await engine.initAsync();
        gpuBackend = "WebGPU";
      }
    } catch (_) { /* fall through to WebGL2 */ }
  }
  if (!engine) {
    engine = new BABYLON.Engine(canvas, true, {
      stencil: true, adaptToDeviceRatio: true, preserveDrawingBuffer: true,
    });
  }
  engine.setHardwareScalingLevel(1 / window.devicePixelRatio);

  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.88, 0.90, 0.92, 1);

  // Environment texture for PBR reflections.
  // In data: URI iframes, cross-origin CDN requests may be blocked (CORS).
  // Fallback: create a basic environment so PBR materials still render solid.
  let envLoaded = false;
  try {
    const envTex = BABYLON.CubeTexture.CreateFromPrefilteredData(HDR_ENV, scene);
    envTex.onError = function() {
      // CDN blocked — create basic environment
      scene.environmentTexture = BABYLON.CubeTexture.CreateFromPrefilteredData(
        "https://assets.babylonjs.com/environments/environmentSpecular.env", scene
      );
    };
    scene.createDefaultEnvironment({
      createGround: false,
      createSkybox: true,
      skyboxSize: 5000,
      skyboxColor: new BABYLON.Color3(0.85, 0.87, 0.90),
      environmentTexture: envTex,
    });
    envLoaded = true;
  } catch (_) {
    // Both CDN attempts failed — use Babylon's built-in environment helper
    try {
      scene.createDefaultEnvironment({
        createGround: false,
        createSkybox: true,
        skyboxSize: 5000,
        skyboxColor: new BABYLON.Color3(0.85, 0.87, 0.90),
      });
    } catch (_2) {
      // Absolute fallback — no environment, rely on direct lighting only
    }
  }

  return { engine, scene, gpuBackend };
}

// ============================================================
// createCamera(scene, canvas, G)
// ============================================================

function createCamera(scene, canvas, G) {
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI / 4, Math.PI / 3, G.cathode_radius * 7,
    new BABYLON.Vector3(G.anode_length / 2, 0, 0), scene);
  cam.attachControl(canvas, false);
  cam.inputs.removeByType("ArcRotateCameraMouseWheelInput");
  canvas.addEventListener("wheel", function(e) {
    e.preventDefault();
    cam.radius -= e.deltaY * 0.05;
    cam.radius = Math.max(cam.lowerRadiusLimit, Math.min(cam.upperRadiusLimit, cam.radius));
  }, { passive: false });
  cam.lowerRadiusLimit = G.anode_radius * 0.2;
  cam.upperRadiusLimit = G.cathode_radius * 60;
  cam.pinchPrecision = 15;
  cam.panningSensibility = 60;
  cam.minZ = 0.0005;
  cam.inertia = 0.88;

  let autoOrbit = true;
  let userInteracting = false;
  let interactionTimeout = null;
  canvas.addEventListener("pointerdown", function() {
    userInteracting = true; autoOrbit = false;
    if (interactionTimeout) clearTimeout(interactionTimeout);
  });
  canvas.addEventListener("pointerup", function() {
    userInteracting = false;
    interactionTimeout = setTimeout(function() { autoOrbit = true; }, 5000);
  });
  scene.registerBeforeRender(function() {
    if (autoOrbit && !userInteracting) {
      cam.alpha += 0.001;
    }
  });

  return { cam, getAutoOrbit: () => autoOrbit, getUserInteracting: () => userInteracting };
}

// ============================================================
// createLights(scene) -- ONE key + ONE fill (no duplicates)
// ============================================================

function createLights(scene) {
  const keyLight = new BABYLON.DirectionalLight("key",
    new BABYLON.Vector3(-1, -2, 1), scene);
  keyLight.intensity = 1.5;
  keyLight.diffuse = new BABYLON.Color3(1, 0.98, 0.95);

  const fillLight = new BABYLON.HemisphericLight("fill",
    new BABYLON.Vector3(0, 1, 0), scene);
  fillLight.intensity = 0.6;
  fillLight.diffuse = new BABYLON.Color3(0.9, 0.92, 1.0);
  fillLight.groundColor = new BABYLON.Color3(0.3, 0.3, 0.35);

  return { keyLight, fillLight };
}

// ============================================================
// createDevice(scene, G) -- anode, cathode, insulator from G params
// ============================================================

function createDevice(scene, G) {
  const anodeMat = G.anode_material || "copper";
  const cathodeMat = G.cathode_material || "stainless_steel";
  const insMat = G.insulator_material || "alumina";

  const anodeSpec = MATERIAL_COLORS[anodeMat] || MATERIAL_COLORS.copper;
  const cathodeSpec = MATERIAL_COLORS[cathodeMat] || MATERIAL_COLORS.stainless_steel;
  const insColor = INSULATOR_COLORS[insMat] || INSULATOR_COLORS.alumina;

  // Copper PBR for anode
  const copperMat = new BABYLON.PBRMaterial("copper", scene);
  copperMat.metallic = anodeSpec.metallic;
  copperMat.roughness = anodeSpec.roughness;
  copperMat.albedoColor = new BABYLON.Color3(anodeSpec.albedo[0], anodeSpec.albedo[1], anodeSpec.albedo[2]);
  copperMat.emissiveColor = new BABYLON.Color3(0.05, 0.03, 0.01);
  copperMat.environmentIntensity = 1.5;
  copperMat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;

  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 64, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.renderingGroupId = 0;

  // Steel PBR for cathode rods
  const steelMat = new BABYLON.PBRMaterial("steel", scene);
  steelMat.metallic = cathodeSpec.metallic;
  steelMat.roughness = cathodeSpec.roughness;
  steelMat.albedoColor = new BABYLON.Color3(cathodeSpec.albedo[0], cathodeSpec.albedo[1], cathodeSpec.albedo[2]);
  steelMat.emissiveColor = new BABYLON.Color3(0.02, 0.02, 0.03);
  steelMat.environmentIntensity = 1.2;
  steelMat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;

  // Rod count from G.n_cathode_rods (data-driven, default 8)
  const N_RODS = G.n_cathode_rods || 8;
  const rodDiam = G.cathode_rod_diameter
    ? G.cathode_rod_diameter
    : G.cathode_radius * 0.06;

  const cathodeRods = [];
  for (let i = 0; i < N_RODS; i++) {
    const angle = (i / N_RODS) * Math.PI * 2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: rodDiam, height: G.anode_length * 1.05,
      tessellation: 12,
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

  // Insulator -- annular disc at z=0 (NOT torus)
  const ceramicMat = new BABYLON.PBRMaterial("ceramic", scene);
  ceramicMat.metallic = 0;
  ceramicMat.roughness = 0.5;
  ceramicMat.albedoColor = new BABYLON.Color3(insColor[0], insColor[1], insColor[2]);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.04, 0.03, 0.02);
  ceramicMat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;

  const insThickness = G.insulator_thickness || G.anode_radius * 0.15;
  const insOuterR = G.anode_radius + (G.cathode_radius - G.anode_radius) * 0.3;
  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2,
    diameterBottom: insOuterR * 2,
    height: insThickness,
    tessellation: 64,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThickness / 2;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  return { anode, cathodeRods, insulator, copperMat, steelMat, ceramicMat };
}

// ============================================================
// createPlasma(scene, G, L) -- sheath, trail, pinch, halo
// ============================================================

function createPlasma(scene, G, L) {
  const gasCol = getGasColor(L);

  // Current sheath -- additive, Fresnel edge glow
  const sheathMat = new BABYLON.StandardMaterial("sheathMat", scene);
  sheathMat.emissiveColor = new BABYLON.Color3(gasCol[0], gasCol[1], gasCol[2]);
  sheathMat.disableLighting = true;
  sheathMat.backFaceCulling = false;
  sheathMat.alpha = 0.5;
  sheathMat.alphaMode = BABYLON.Engine.ALPHA_ADD;

  sheathMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  sheathMat.emissiveFresnelParameters.bias = 0.3;
  sheathMat.emissiveFresnelParameters.power = 2;
  sheathMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(
    Math.min(1, gasCol[0] + 0.4),
    Math.min(1, gasCol[1] + 0.25),
    Math.min(1, gasCol[2] + 0.0)
  );
  sheathMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(
    gasCol[0] * 0.5,
    gasCol[1] * 0.5,
    gasCol[2] * 0.8
  );

  const sheathMidR = (G.anode_radius + G.cathode_radius) / 2;
  const sheathTubeR = (G.cathode_radius - G.anode_radius) / 2;
  const sheath = BABYLON.MeshBuilder.CreateTorus("sheath", {
    diameter: sheathMidR * 2,
    thickness: sheathTubeR * 2,
    tessellation: 48,
  }, scene);
  sheath.rotation.z = Math.PI / 2;
  sheath.isVisible = false;  // hidden until animation starts
  sheath.material = sheathMat;
  sheath.renderingGroupId = 1;

  // Plasma trail -- additive, very faint
  const trailMat = new BABYLON.StandardMaterial("trailMat", scene);
  trailMat.emissiveColor = new BABYLON.Color3(0.08, 0.12, 0.3);
  trailMat.disableLighting = true;
  trailMat.backFaceCulling = false;
  trailMat.alpha = 0.1;
  trailMat.alphaMode = BABYLON.Engine.ALPHA_ADD;

  const trail = BABYLON.MeshBuilder.CreateTube("trail", {
    path: [new BABYLON.Vector3(0, 0, 0), new BABYLON.Vector3(1, 0, 0)],
    radius: (G.anode_radius + G.cathode_radius) / 2,
    tessellation: 24, cap: BABYLON.Mesh.NO_CAP, updatable: true,
  }, scene);
  trail.material = trailMat;
  trail.renderingGroupId = 1;
  trail.isVisible = false;  // hidden until animation starts

  // Pinch column -- additive, updatable tube
  const pinchMat = new BABYLON.StandardMaterial("pinchMat", scene);
  pinchMat.emissiveColor = new BABYLON.Color3(1, 0.4, 0.1);
  pinchMat.disableLighting = true;
  pinchMat.backFaceCulling = false;
  pinchMat.alpha = 0;
  pinchMat.alphaMode = BABYLON.Engine.ALPHA_ADD;

  const N_PINCH = 24;
  const pinchPath = [];
  const pinchStart = G.anode_length * 0.85;
  const pinchEnd = G.anode_length * 1.02;
  for (let k = 0; k <= N_PINCH; k++) {
    pinchPath.push(new BABYLON.Vector3(
      pinchStart + (pinchEnd - pinchStart) * k / N_PINCH, 0, 0
    ));
  }
  const pinchRadii = new Array(N_PINCH + 1).fill(G.anode_radius * 0.3);
  const pinch = BABYLON.MeshBuilder.CreateTube("pinch", {
    path: pinchPath, radiusFunction: function(idx) { return pinchRadii[idx]; },
    tessellation: 48, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  pinch.material = pinchMat;
  pinch.renderingGroupId = 1;
  pinch.isVisible = false;  // hidden until pinch phase

  // Halo glow around pinch -- additive, backside
  const haloMat = new BABYLON.StandardMaterial("haloMat", scene);
  haloMat.emissiveColor = new BABYLON.Color3(0.7, 0.1, 0.03);
  haloMat.disableLighting = true;
  haloMat.alpha = 0;
  haloMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  haloMat.backFaceCulling = false;

  const haloRadii = new Array(N_PINCH + 1).fill(G.anode_radius * 0.6);
  const halo = BABYLON.MeshBuilder.CreateTube("halo", {
    path: pinchPath, radiusFunction: function(idx) { return haloRadii[idx]; },
    tessellation: 48, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.BACKSIDE, updatable: true,
  }, scene);
  halo.material = haloMat;
  halo.renderingGroupId = 1;
  halo.isVisible = false;  // hidden until pinch phase

  return {
    sheath, sheathMat, trail, trailMat,
    pinch, pinchMat, halo, haloMat,
    pinchRadii, haloRadii, pinchPath, N_PINCH,
    pinchStart, pinchEnd,
  };
}

// ============================================================
// createGrid(scene, G) -- ground grid + scale bar
// ============================================================

function createGrid(scene, G) {
  const gridSize = Math.max(G.anode_length * 3, G.cathode_radius * 6);
  const gridGround = BABYLON.MeshBuilder.CreateGround("grid", {
    width: gridSize, height: gridSize, subdivisions: 1,
  }, scene);
  gridGround.position.y = -G.cathode_radius * 1.2;
  gridGround.position.x = G.anode_length / 2;

  const gridTex = new BABYLON.DynamicTexture("gridTex", 512, scene, false);
  const gridCtx = gridTex.getContext();
  gridCtx.fillStyle = "rgba(210, 215, 220, 1.0)";
  gridCtx.fillRect(0, 0, 512, 512);
  gridCtx.strokeStyle = "rgba(160, 165, 175, 0.6)";
  gridCtx.lineWidth = 1;
  for (let gi = 0; gi <= 20; gi++) {
    const gpos = gi * 512 / 20;
    gridCtx.beginPath(); gridCtx.moveTo(gpos, 0); gridCtx.lineTo(gpos, 512); gridCtx.stroke();
    gridCtx.beginPath(); gridCtx.moveTo(0, gpos); gridCtx.lineTo(512, gpos); gridCtx.stroke();
  }
  gridCtx.strokeStyle = "rgba(120, 125, 135, 0.8)";
  gridCtx.lineWidth = 2;
  for (let gi = 0; gi <= 4; gi++) {
    const gpos = gi * 512 / 4;
    gridCtx.beginPath(); gridCtx.moveTo(gpos, 0); gridCtx.lineTo(gpos, 512); gridCtx.stroke();
    gridCtx.beginPath(); gridCtx.moveTo(0, gpos); gridCtx.lineTo(512, gpos); gridCtx.stroke();
  }
  gridTex.update();

  const gridMat = new BABYLON.StandardMaterial("gridMat", scene);
  gridMat.diffuseTexture = gridTex;
  gridMat.specularColor = new BABYLON.Color3(0, 0, 0);
  gridMat.alpha = 0.7;
  gridGround.material = gridMat;

  // Scale bar along anode axis
  const scaleLen = G.anode_length;
  BABYLON.MeshBuilder.CreateLines("scaleBar", {
    points: [
      new BABYLON.Vector3(0, -G.cathode_radius * 1.15, 0),
      new BABYLON.Vector3(scaleLen, -G.cathode_radius * 1.15, 0),
    ],
    colors: [new BABYLON.Color4(0.4, 0.4, 0.45, 1), new BABYLON.Color4(0.4, 0.4, 0.45, 1)],
  }, scene);

  for (let ti = 0; ti <= 4; ti++) {
    const tx = scaleLen * ti / 4;
    BABYLON.MeshBuilder.CreateLines("tick" + ti, {
      points: [
        new BABYLON.Vector3(tx, -G.cathode_radius * 1.15 - 0.005, 0),
        new BABYLON.Vector3(tx, -G.cathode_radius * 1.15 + 0.005, 0),
      ],
      colors: [new BABYLON.Color4(0.4, 0.4, 0.45, 1), new BABYLON.Color4(0.4, 0.4, 0.45, 1)],
    }, scene);
  }
}

// ============================================================
// createHeatmap(scene, G) -- midplane heatmap mesh + material
// ============================================================

function createHeatmap(scene, G) {
  const heatPaths = [];
  const heatNr = 16;
  const heatNz = 32;
  for (let ir = 0; ir <= heatNr; ir++) {
    const r = G.anode_radius + (G.cathode_radius - G.anode_radius) * ir / heatNr;
    const path = [];
    for (let iz = 0; iz <= heatNz; iz++) {
      const z = G.anode_length * iz / heatNz;
      const angle = Math.PI * 0.33;
      path.push(new BABYLON.Vector3(z, r * Math.sin(angle), r * Math.cos(angle)));
    }
    heatPaths.push(path);
  }
  const heatPlane = BABYLON.MeshBuilder.CreateRibbon("heatPlane", {
    pathArray: heatPaths,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE,
    updatable: false,
  }, scene);
  heatPlane.isVisible = false;
  heatPlane.isPickable = false;

  const heatMat = new BABYLON.StandardMaterial("heatMat", scene);
  heatMat.disableLighting = true;
  heatMat.backFaceCulling = false;
  heatPlane.material = heatMat;

  return { heatPlane, heatMat };
}

// ============================================================
// Snap Cache -- pre-decode MHD base64 -> RGBA
// ============================================================

function buildSnapCache(fieldKey, layer, cache) {
  if (!layer || !layer.frames || layer.frames.length === 0) return;
  const shape = layer.frames_shape || layer.shape;
  if (!shape) return;
  const nr = shape[0], nz = shape[1];
  const texW = nz, texH = nr;
  const n = layer.frames.length;
  const times = new Float64Array(n);
  const rgbaFrames = new Array(n);
  for (let fi = 0; fi < n; fi++) {
    const frame = layer.frames[fi];
    times[fi] = frame.t_us;
    const vals = b64ToFloat32(frame.data);
    const rgba = new Uint8Array(texW * texH * 4);
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
  cache[fieldKey] = { times, rgba: rgbaFrames, texW, texH };
}

function nearestSnapIdx(cache, fieldKey, t_us) {
  const entry = cache[fieldKey];
  if (!entry) return -1;
  const times = entry.times;
  // Binary search for nearest timestamp
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t_us) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t_us) < Math.abs(times[lo] - t_us)) return lo - 1;
  return lo;
}

// ============================================================
// createFieldLines(scene, G, L) -- B-field tori + poloidal lines
// ============================================================

function createFieldLines(scene, G, L) {
  const fieldLines = [];
  const fieldLineData = [];
  const N_RADII = 5, N_ZPOS = 4;

  for (let zi = 0; zi < N_ZPOS; zi++) {
    const zPos = G.anode_length * (0.15 + 0.7 * zi / (N_ZPOS - 1));
    for (let ri = 0; ri < N_RADII; ri++) {
      const minR = G.anode_radius * 1.4;
      const maxR = G.cathode_radius * 0.95;
      const baseR = minR + (maxR - minR) * ri / (N_RADII - 1);
      const bStrength = 1 - ri / N_RADII;
      const tube = BABYLON.MeshBuilder.CreateTorus("fl" + zi + "_" + ri, {
        diameter: baseR * 2,
        thickness: G.cathode_radius * 0.015 * (0.5 + bStrength),
        tessellation: 64,
      }, scene);
      tube.rotation.z = Math.PI / 2;
      tube.position.x = zPos;
      const lineMat = new BABYLON.StandardMaterial("flm" + zi + "_" + ri, scene);
      lineMat.emissiveColor = new BABYLON.Color3(
        0.1 + bStrength * 0.3, 0.3 + bStrength * 0.5, 0.8 + bStrength * 0.2
      );
      lineMat.disableLighting = true;
      lineMat.alpha = 0.35 + bStrength * 0.35;
      lineMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
      lineMat.backFaceCulling = false;
      tube.material = lineMat;
      tube.isVisible = false;
      tube.renderingGroupId = 1;
      fieldLines.push(tube);
      fieldLineData.push({ baseR, zi, ri, zPos });
    }
  }

  // Poloidal field lines from MHD Br/Bz data
  if (L.bfield && L.bfield.Br && L.bfield.Bz) {
    try {
      const fdBr = decodeBase64Float32(L.bfield.Br, L.bfield.shape);
      const fdBz = decodeBase64Float32(L.bfield.Bz, L.bfield.shape);
      const bnx = fdBr.shape[0], bnz = fdBr.shape[1];
      for (let s = 0; s < 8; s++) {
        let x = G.anode_length * (0.1 + 0.8 * s / 8), z = 0;
        const pts = [];
        const ds = G.anode_length / 60 * 0.6;
        for (let step = 0; step < 60; step++) {
          pts.push(new BABYLON.Vector3(x, 0, z));
          const fx = (x / G.anode_length) * (bnx - 1);
          const fz = ((z + G.cathode_radius) / (G.cathode_radius * 2)) * (bnz - 1);
          const br = bilinearSample(fdBr.data, bnx, bnz, fx, fz);
          const bz = bilinearSample(fdBz.data, bnx, bnz, fx, fz);
          const mag = Math.sqrt(br * br + bz * bz) + 1e-10;
          x += ds * br / mag; z += ds * bz / mag;
          if (x < 0 || x > G.anode_length || Math.abs(z) > G.cathode_radius) break;
        }
        if (pts.length > 4) {
          const line = BABYLON.MeshBuilder.CreateLines("flp" + s, { points: pts }, scene);
          line.color = new BABYLON.Color3(0.3, 0.7, 1.0);
          line.alpha = 0.5;
          line.isVisible = false;
          fieldLines.push(line);
        }
      }
    } catch (_) { /* MHD data unavailable or malformed */ }
  }

  return { fieldLines, fieldLineData, N_RADII, N_ZPOS };
}

// ============================================================
// createParticles(scene, G) -- GPU particle system
// ============================================================

function createParticles(scene, G) {
  const useGPU = BABYLON.GPUParticleSystem.IsSupported;
  const PSClass = useGPU ? BABYLON.GPUParticleSystem : BABYLON.ParticleSystem;
  const psCap = useGPU ? 50000 : 4000;
  const ps = new PSClass("ions", { capacity: psCap }, scene);
  ps.emitter = new BABYLON.Vector3(0, 0, 0);

  const psEmitter = new BABYLON.SphereParticleEmitter();
  psEmitter.radius = G.cathode_radius * 0.85;
  psEmitter.radiusRange = 0.4;
  ps.particleEmitterType = psEmitter;

  ps.minLifeTime = 0.1; ps.maxLifeTime = 0.4;
  ps.emitRate = useGPU ? 5000 : 400;
  ps.minSize = 0.3; ps.maxSize = 1.5;
  ps.minEmitPower = 0.2; ps.maxEmitPower = 1.5;

  ps.addColorGradient(0.0, new BABYLON.Color4(0.1, 0.2, 0.8, 0.0));
  ps.addColorGradient(0.15, new BABYLON.Color4(0.2, 0.5, 1.0, 0.5));
  ps.addColorGradient(0.4, new BABYLON.Color4(0.4, 0.8, 1.0, 0.6));
  ps.addColorGradient(0.7, new BABYLON.Color4(0.9, 0.95, 1.0, 0.4));
  ps.addColorGradient(1.0, new BABYLON.Color4(1.0, 1.0, 1.0, 0.0));

  ps.addSizeGradient(0.0, 0.3);
  ps.addSizeGradient(0.15, 1.2);
  ps.addSizeGradient(0.6, 0.8);
  ps.addSizeGradient(1.0, 0.0);

  ps.isBillboardBased = true;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.renderingGroupId = 1;

  // Soft gaussian procedural texture
  const ptexSize = 64;
  const ptex = new BABYLON.DynamicTexture("ptex", ptexSize, scene, false);
  const ptxCtx = ptex.getContext();
  const grad = ptxCtx.createRadialGradient(
    ptexSize / 2, ptexSize / 2, 0, ptexSize / 2, ptexSize / 2, ptexSize / 2
  );
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.2, "rgba(255,245,220,0.9)");
  grad.addColorStop(0.5, "rgba(220,180,100,0.4)");
  grad.addColorStop(0.8, "rgba(150,80,30,0.1)");
  grad.addColorStop(1, "rgba(80,30,10,0)");
  ptxCtx.fillStyle = grad;
  ptxCtx.fillRect(0, 0, ptexSize, ptexSize);
  ptex.update();
  ps.particleTexture = ptex;
  ps.start();

  // Fire preset for pinch
  let fireSet = null;
  try {
    BABYLON.ParticleHelper.CreateAsync("fire", scene).then(function(set) {
      fireSet = set;
      set.systems.forEach(function(sys) {
        sys.emitter = new BABYLON.Vector3(G.anode_length * 0.8, 0, 0);
        sys.minSize *= 0.3;
        sys.maxSize *= 0.3;
        sys.minEmitPower *= 0.5;
        sys.maxEmitPower *= 0.5;
        sys.renderingGroupId = 1;
      });
    }).catch(function() {});
  } catch (_) { /* fire preset unavailable */ }

  return {
    ps, psEmitter, useGPU,
    getFireSet: () => fireSet,
    setFireSet: (v) => { fireSet = v; },
  };
}

// ============================================================
// createPipeline(scene, cam) -- ONE pipeline, glow, SSAO
// ============================================================

function createPipeline(scene, cam) {
  const pipeline = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipeline.bloomEnabled = true;
  pipeline.bloomWeight = 0.25;
  pipeline.bloomThreshold = 0.5;
  pipeline.bloomKernel = 64;
  pipeline.bloomScale = 0.5;
  pipeline.fxaaEnabled = true;
  pipeline.imageProcessingEnabled = true;
  pipeline.imageProcessing.toneMappingEnabled = true;
  pipeline.imageProcessing.toneMappingType = BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pipeline.imageProcessing.exposure = 1.1;
  pipeline.imageProcessing.contrast = 1.1;
  pipeline.chromaticAberrationEnabled = false;
  pipeline.chromaticAberration.aberrationAmount = 0;
  pipeline.sharpenEnabled = true;
  pipeline.sharpen.edgeAmount = 0.2;

  let ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.6;
    ssao.radius = 1.5;
    ssao.samples = 16;
    ssao.base = 0.2;
  } catch (_) { /* SSAO not available */ }

  const glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 32, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.5;
  glowLayer.customEmissiveColorSelector = function(mesh, _sub, _mat, result) {
    if (GLOW_MESHES.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      const ec = mesh.material.emissiveColor;
      const mag = Math.max(ec.r, ec.g, ec.b);
      const boost = mag > 1.0 ? Math.sqrt(mag) : 1.0;
      result.set(ec.r * boost, ec.g * boost, ec.b * boost, mesh.material.alpha || 0);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  return { pipeline, ssao, glowLayer };
}

// ============================================================
// applyFrame sub-functions
// ============================================================

function updateSheath(ctx, f, col, isP, cr) {
  const { sheath, sheathMat, G, S } = ctx;
  sheath.position.x = isP ? G.anode_length : f.z;
  sheathMat.emissiveColor.set(col[0], col[1], col[2]);
  if (isP) {
    const compScale = Math.max(0.03, cr);
    sheath.scaling.set(1, compScale, compScale);
  } else {
    const zFrac = Math.min(1, f.z / G.anode_length);
    const rundownScale = 1.0 - zFrac * (1.0 - Math.max(0.03, cr)) * 0.3;
    sheath.scaling.set(1, rundownScale, rundownScale);
  }
  sheathMat.alpha = Math.min(0.7, 0.1 + Math.abs(f.I / S.I_peak) * 0.6);
  if ((f.phase === "post_pinch" || f.phase === "reflected") && Math.abs(f.I / S.I_peak) < 0.1) {
    sheath.isVisible = false;
  } else {
    sheath.isVisible = true;
  }
}

function updateTrail(ctx, f, col, isP, cr) {
  const { trail, trailMat, G } = ctx;
  // Trail visible only when current is flowing
  trail.isVisible = Math.abs(f.I) > 0.01;
  const tLen = Math.max(isP ? G.anode_length : f.z, 0.2);
  trail.scaling.x = tLen;
  trail.position.x = tLen / 2;
  if (isP) {
    const trailScale = Math.max(0.05, cr);
    trail.scaling.y = trailScale;
    trail.scaling.z = trailScale;
  } else {
    trail.scaling.y = 1;
    trail.scaling.z = 1;
  }
  trailMat.emissiveColor.set(col[0] * 0.25, col[1] * 0.25, col[2] * 0.3);
  trailMat.alpha = Math.min(0.12, 0.05 + Math.abs(f.I) * 0.04);
}

function updatePinchColumn(ctx, f, isP, cr, pI) {
  const { pinch, pinchMat, halo, haloMat, pinchRadii, haloRadii, pinchPath, N_PINCH, G, L } = ctx;
  const pinchEnd = G.anode_length * 1.02;
  const pinchStart = G.anode_length * 0.85;

  const instAmp = L.instability ? L.instability.amplitude : 0;
  const rippleAmp = isP ? instAmp * Math.min(1, (1 - cr) * 2) : 0;
  const pinchR = Math.max(G.anode_radius * 0.05, cr * G.cathode_radius * 0.25);
  const pinchLen = pinchEnd - pinchStart;
  let nModes = Math.max(1, Math.round(pinchLen / (2 * Math.PI * Math.max(pinchR, 0.001))));
  nModes = Math.min(nModes, 6);

  for (let pk = 0; pk <= N_PINCH; pk++) {
    const zFrac = pk / N_PINCH;
    const taper = Math.sin(Math.PI * zFrac);
    const localR = pinchR * (0.5 + 0.5 * taper);
    const ripple = rippleAmp * localR * Math.cos(2 * Math.PI * nModes * zFrac);
    pinchRadii[pk] = Math.max(0.001, localR + ripple);
    haloRadii[pk] = Math.min(G.cathode_radius * 0.5, Math.max(0.002, (localR + ripple) * 2.5));
  }

  BABYLON.MeshBuilder.CreateTube("pinch", {
    path: pinchPath, radiusFunction: function(idx) { return pinchRadii[idx]; },
    tessellation: 20, cap: BABYLON.Mesh.CAP_ALL, instance: pinch,
  });
  BABYLON.MeshBuilder.CreateTube("halo", {
    path: pinchPath, radiusFunction: function(idx) { return haloRadii[idx]; },
    tessellation: 48, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.BACKSIDE, instance: halo,
  });

  const pinchPhase = f.phase === "pinch" || f.phase === "post_pinch" || f.phase === "reflected";
  const pinchVisible = pinchPhase || (isP && cr < 0.3);
  pinch.isVisible = pinchVisible;
  halo.isVisible = pinchVisible;
  pinchMat.alpha = pinchVisible ? pI * 0.85 : 0;
  haloMat.alpha = pinchVisible ? pI * 0.25 : 0;

  // HDR pinch emissive
  if (pI > 0.7) {
    const hdr = 1.0 + (pI - 0.7) * 5.0;
    pinchMat.emissiveColor.set(hdr, hdr * 0.9, hdr * 0.75);
    haloMat.emissiveColor.set(hdr * 0.4, hdr * 0.15, hdr * 0.05);
  } else if (pI > 0.3) {
    pinchMat.emissiveColor.set(0.8 + pI * 0.4, 0.3 + pI * 0.6, pI * 0.4);
    haloMat.emissiveColor.set(0.5, 0.12, 0.04);
  } else {
    pinchMat.emissiveColor.set(pI * 1.5, pI * 0.4, pI * 0.15);
    haloMat.emissiveColor.set(pI * 0.5, pI * 0.1, 0.02);
  }

  return rippleAmp;
}

function updatePipelineForPhase(ctx, pI) {
  const { glowLayer, pipeline } = ctx;
  glowLayer.intensity = 0.5 + pI * 0.8;
  pipeline.bloomWeight = 0.25 + pI * 0.35;
  pipeline.bloomThreshold = 0.5 - pI * 0.2;
  pipeline.imageProcessing.exposure = 1.1 - pI * 0.15;
}

function updateAnodeGlow(ctx, pI) {
  if (pI > 0.5) {
    ctx.copperMat.emissiveColor.set(0.15 + pI * 0.3, 0.06 + pI * 0.08, 0.02);
  } else {
    ctx.copperMat.emissiveColor.set(0.05, 0.03, 0.01);
  }
}

function updateFirePreset(ctx, pI) {
  const fireSet = ctx.particles.getFireSet();
  if (fireSet) {
    if (pI > 0.5 && !fireSet._started) {
      fireSet.start();
      fireSet._started = true;
    } else if (pI < 0.1 && fireSet._started) {
      fireSet.dispose();
      ctx.particles.setFireSet(null);
    }
  }
}

function updateParticles(ctx, f, isP, pI) {
  const { ps, psEmitter, useGPU, G } = ctx;
  ps.emitter.x = isP ? G.anode_length : f.z;
  if (f.phase === "rundown") {
    psEmitter.radius = G.cathode_radius * 0.85;
    ps.gravity = new BABYLON.Vector3(1.5, 0, 0);
    ps.minEmitPower = 0.5; ps.maxEmitPower = 2;
    ps.emitRate = useGPU ? 6000 : 400;
  } else if (isP) {
    const compR = Math.max(f.r, G.anode_radius * 0.05);
    psEmitter.radius = compR * 0.8;
    const boost = Math.min(8, Math.pow(G.cathode_radius / Math.max(compR, 0.1), 1.5));
    ps.emitRate = useGPU ? Math.min(50000, (6000 * boost) | 0) : Math.min(4000, (400 * boost) | 0);
    ps.minSize = 0.3 + pI * 0.5;
    ps.maxSize = 1.0 + pI * 1.5;
    if (f.phase === "post_pinch" || f.phase === "reflected") {
      ps.gravity = new BABYLON.Vector3(0.5, 0, 0);
      ps.minEmitPower = 0.5; ps.maxEmitPower = 2;
      ps.emitRate = useGPU ? 2000 : 150;
      ps.minLifeTime = 0.05; ps.maxLifeTime = 0.15;
      psEmitter.radius = compR * 2;
    } else if (pI > 0.5) {
      ps.gravity = new BABYLON.Vector3(4, 0, 0);
      ps.minEmitPower = 3; ps.maxEmitPower = 8;
      ps.minLifeTime = 0.05; ps.maxLifeTime = 0.12;
    } else {
      ps.gravity = new BABYLON.Vector3(0, 0, 0);
      ps.minEmitPower = 0.5; ps.maxEmitPower = 2;
    }
  }
}

function updateFieldLinesFrame(ctx, f, isP, cr) {
  const { fieldLines, fieldLineData, S } = ctx;
  const N_RADII = ctx.N_RADII || 5;
  const Ifrac = Math.abs(f.I) / Math.max(S.I_peak, 0.001);

  for (let fli = 0; fli < fieldLines.length; fli++) {
    if (!fieldLines[fli].isVisible) continue;
    const fld = fieldLineData[fli];
    if (!fld) continue;
    const bStr = 1 - fld.ri / N_RADII;

    if (fieldLines[fli].material) {
      fieldLines[fli].material.alpha = Math.min(0.85, (0.1 + bStr * 0.3) * Ifrac * 2);
      if (fieldLines[fli].material.emissiveColor) {
        const glow = Math.min(1, Ifrac * 1.5);
        fieldLines[fli].material.emissiveColor.set(
          0.1 + bStr * 0.3 * glow, 0.3 + bStr * 0.5 * glow, 0.7 + bStr * 0.3 * glow
        );
      }
    }

    if (isP) {
      const scaleFactor = cr + (1 - cr) * (fld.ri / N_RADII);
      fieldLines[fli].scaling.y = Math.max(0.05, scaleFactor);
      fieldLines[fli].scaling.z = Math.max(0.05, scaleFactor);
    } else {
      fieldLines[fli].scaling.y = 1;
      fieldLines[fli].scaling.z = 1;
    }

    if (f.phase === "rundown" && fld.zi >= 2) {
      fieldLines[fli].position.x = Math.min(fld.zPos, f.z);
    } else {
      fieldLines[fli].position.x = fld.zPos;
    }
  }
}

function updateCinematicEffects(ctx, f, isP, pI) {
  const { pipeline, cam, pinch, G, cameraState } = ctx;
  const autoOrbit = cameraState.getAutoOrbit();
  const userInteracting = cameraState.getUserInteracting();

  // DOF on pinch
  if (isP && pI > 0.3) {
    pipeline.depthOfFieldEnabled = true;
    pipeline.depthOfField.focalLength = 60;
    pipeline.depthOfField.fStop = 2;
    pipeline.depthOfField.focusDistance =
      BABYLON.Vector3.Distance(cam.position, pinch.position) * 1000;
  } else {
    pipeline.depthOfFieldEnabled = false;
  }

  // Chromatic aberration
  if (pI > 0.5) {
    pipeline.chromaticAberrationEnabled = true;
    pipeline.chromaticAberration.aberrationAmount = pI * 30;
  } else {
    pipeline.chromaticAberrationEnabled = false;
  }

  // Camera auto-zoom
  if (isP && pI > 0.2 && autoOrbit) {
    const targetRadius = G.cathode_radius * (3 + (1 - pI) * 4);
    cam.radius += (targetRadius - cam.radius) * 0.02;
  } else if (autoOrbit && !userInteracting) {
    const defaultRadius = G.cathode_radius * 7;
    cam.radius += (defaultRadius - cam.radius) * 0.01;
  }

  // Exposure flash
  if (pI > 0.8) {
    pipeline.imageProcessing.exposure = 1.4 + (pI - 0.8) * 3;
  } else {
    pipeline.imageProcessing.exposure += (1.4 - pipeline.imageProcessing.exposure) * 0.1;
  }
}

// ============================================================
// MAIN: createDPFScene(canvas, data) -- async entry point
// ============================================================

async function createDPFScene(canvas, data) {
  const L = data;
  const G = L.geometry;
  const S = L.sheath;

  // Engine, scene, environment
  const { engine, scene, gpuBackend } = await initEngine(canvas);

  // Camera
  const cameraState = createCamera(scene, canvas, G);
  const cam = cameraState.cam;

  // ONE set of lights
  createLights(scene);

  // Device geometry (data-driven)
  const device = createDevice(scene, G);

  // Plasma effects
  const plasma = createPlasma(scene, G, L);

  // Ground grid + scale bar
  createGrid(scene, G);

  // Heatmap
  const { heatPlane, heatMat } = createHeatmap(scene, G);

  // Snap cache
  let snapCache = {};
  let lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
  let heatTex = null;
  buildSnapCache("density", L.density, snapCache);
  buildSnapCache("temperature", L.temperature, snapCache);
  buildSnapCache("bfield", L.bfield, snapCache);

  function applySnapTexture(fieldKey) {
    const cache = snapCache[fieldKey];
    if (!cache) return false;
    const idx = lastSnapIdx[fieldKey];
    if (idx < 0 || idx >= cache.rgba.length) return false;
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(
      cache.rgba[idx], cache.texW, cache.texH,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene,
      false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE
    );
    heatMat.diffuseTexture = heatTex;
    heatMat.emissiveTexture = heatTex;
    heatMat.alpha = 0.8;
    heatMat.useAlphaFromDiffuseTexture = true;
    heatPlane.isVisible = true;
    return true;
  }

  // Field lines
  const fl = createFieldLines(scene, G, L);

  // Particles
  const particles = createParticles(scene, G);

  // ONE pipeline
  const { pipeline, ssao, glowLayer } = createPipeline(scene, cam);

  // Active overlay tracking
  let activeOverlay = "none";

  // Context object for sub-functions
  const ctx = {
    sheath: plasma.sheath, sheathMat: plasma.sheathMat,
    trail: plasma.trail, trailMat: plasma.trailMat,
    pinch: plasma.pinch, pinchMat: plasma.pinchMat,
    halo: plasma.halo, haloMat: plasma.haloMat,
    pinchRadii: plasma.pinchRadii, haloRadii: plasma.haloRadii,
    pinchPath: plasma.pinchPath, N_PINCH: plasma.N_PINCH,
    G, S, L,
    copperMat: device.copperMat,
    glowLayer, pipeline,
    ps: particles.ps, psEmitter: particles.psEmitter, useGPU: particles.useGPU,
    particles,
    fieldLines: fl.fieldLines, fieldLineData: fl.fieldLineData, N_RADII: fl.N_RADII,
    cam, cameraState,
  };

  // updateHeatmap function
  function updateHeatmap(key) {
    if (!L || key === "none") {
      if (heatPlane) heatPlane.isVisible = false;
      return;
    }
    let layer = null;
    if (key === "density" && L.density) layer = L.density;
    else if (key === "temperature" && L.temperature) layer = L.temperature;
    else if (key === "bfield" && L.bfield) layer = L.bfield;
    else if (key === "radiation" && L.radiation) layer = L.radiation;
    else if (key === "yield" && L.yield_map) layer = L.yield_map;

    if (!layer || !layer.data || !layer.shape) {
      if (heatPlane) heatPlane.isVisible = false;
      return;
    }

    if (snapCache[key] && lastSnapIdx[key] >= 0) {
      applySnapTexture(key);
      return;
    }

    const raw = atob(layer.data);
    const buf = new ArrayBuffer(raw.length);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const vals = new Float32Array(buf);
    const nr = layer.shape[0], nz = layer.shape[1];
    const texW = nz, texH = nr;
    const rgba = new Uint8Array(texW * texH * 4);
    for (let ir = 0; ir < nr; ir++) {
      for (let iz = 0; iz < nz; iz++) {
        const v = vals[ir * nz + iz];
        const c = cmapLookup(v, activeCmap);
        const pi = ((nr - 1 - ir) * nz + iz) * 4;
        rgba[pi] = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 200;
      }
    }
    if (heatTex) heatTex.dispose();
    heatTex = new BABYLON.RawTexture(rgba, texW, texH,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene,
      false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    heatMat.diffuseTexture = heatTex;
    heatMat.emissiveTexture = heatTex;
    heatMat.alpha = 0.8;
    heatMat.useAlphaFromDiffuseTexture = true;
    heatPlane.isVisible = true;
  }

  // Return object matching host code expectations
  return {
    engine,
    scene,
    camera: cam,
    gpuBackend,
    useGPU: particles.useGPU,

    sheath: plasma.sheath,
    sheathMat: plasma.sheathMat,
    trail: plasma.trail,
    trailMat: plasma.trailMat,
    pinch: plasma.pinch,
    pinchMat: plasma.pinchMat,
    halo: plasma.halo,
    haloMat: plasma.haloMat,
    pinchRadii: plasma.pinchRadii,
    haloRadii: plasma.haloRadii,
    pinchPath: plasma.pinchPath,
    N_PINCH: plasma.N_PINCH,

    anode: device.anode,
    cathodeRods: device.cathodeRods,
    insulator: device.insulator,

    ps: particles.ps,
    psEmitter: particles.psEmitter,
    fireSet: particles.getFireSet(),
    pipeline,
    ssao,
    glowLayer,
    updateHeatmap,
    fieldLines: fl.fieldLines,
    fieldLineData: fl.fieldLineData,
    activeOverlay,
    G, S, L,

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
      if (activeOverlay !== "none" && heatPlane && heatPlane.isVisible) {
        updateHeatmap(activeOverlay);
      }
    },

    applyFrame: function(i) {
      if (i < 0 || i >= S.frames.length) return;
      const f = S.frames[i];

      // Advance heatmap snap to nearest MHD timestamp
      if (activeOverlay !== "none" && snapCache[activeOverlay]) {
        const newIdx = nearestSnapIdx(snapCache, activeOverlay, f.t);
        if (newIdx !== lastSnapIdx[activeOverlay]) {
          lastSnapIdx[activeOverlay] = newIdx;
          applySnapTexture(activeOverlay);
        }
      }

      const col = PHASE_COLORS[f.phase] || [0.3, 0.3, 0.4];
      const isP = isRadialPhase(f.phase);
      const cr = Math.max(0.02, f.r / G.cathode_radius);
      let pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
      if (f.phase === "post_pinch") pI *= 0.3;
      if (f.phase === "reflected") pI *= 0.5;

      updateSheath(ctx, f, col, isP, cr);
      updateTrail(ctx, f, col, isP, cr);
      const rippleAmp = updatePinchColumn(ctx, f, isP, cr, pI);
      updatePipelineForPhase(ctx, pI);
      updateAnodeGlow(ctx, pI);
      updateFirePreset(ctx, pI);
      updateParticles(ctx, f, isP, pI);
      updateFieldLinesFrame(ctx, f, isP, cr);
      updateCinematicEffects(ctx, f, isP, pI);

      return { f, col, isP, cr, pI, rippleAmp };
    },
  };
}

// Global exports for host code
window.createDPFScene = createDPFScene;
window.PHASE_LABELS = PHASE_LABELS;
window.PHASE_DESCRIPTIONS = PHASE_DESCRIPTIONS;
window.SPEEDS = SPEEDS;
