/**
 * DPF-Unified Plasma Renderer — Babylon.js 8.x
 *
 * Conference-quality 3D physics visualization.
 *
 * DATA SOURCE ARCHITECTURE:
 * -------------------------
 * The 3D scene elements are driven by TWO data sources with graceful fallback:
 *
 * MHD FIELD DATA (2D arrays from Metal/Python solvers — preferred when available):
 *   - Density isosurface: revolved contour from rho(r,z) via CreateLathe
 *   - RK4 poloidal field lines: traced per-frame from Br/Bz, rendered as tubes colored by |B|
 *   - Velocity-driven particles: emitter direction from MHD v_r/v_z at sheath front
 *   - Dynamic B-field rings: brightness and radius from actual B_theta(r,z)
 *   - Temperature-colored isosurface: hot colormap (blue->cyan->yellow->white)
 *   - Current density J_theta tube: computed from curl(B), brightness ~ |J|
 *   - Midplane heatmaps: density, temperature, |B|, velocity (toggled via layer panel)
 *
 * LEE MODEL SCALARS (0D ODE fallback — always used when MHD data unavailable):
 *   - Current sheath position (z_mm) and radius (r_mm)
 *   - Pinch column radius and m=0 instability ripple amplitude
 *   - Particle system emitter position, radius, and rate
 *   - B-field ring brightness (proportional to I/I_peak)
 *   - Trail tube length and scaling
 *   - All phase-dependent visual effects (colors, glow, DOF, chromatic aberration)
 *
 * Production techniques:
 *   - createDefaultEnvironment() for HDR IBL + skybox
 *   - needDepthPrePass on transparent materials (no rendering group hacks)
 *   - ParticleHelper presets + GPU particles
 *   - forceSharedVertices() for smooth normals
 *   - customEmissiveColorSelector for glow without bleed
 *   - DefaultRenderingPipeline (bloom, ACES, FXAA, SSAO)
 */

const HDR_ENV = "https://assets.babylonjs.com/environments/Studio_Softbox_2Umbrellas_cube_specular.env";

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
  pinch: "Pinch — peak compression",
  post_pinch: "Post-pinch disruption",
  none: "",
};

const PHASE_DESCRIPTIONS = {
  rundown: "Current sheath sweeping gas toward anode tip",
  radial: "Plasma ring compressing inward — magnetic piston",
  mhd_radial: "MHD radial implosion in progress",
  mhd: "Full MHD simulation — no Lee-phase snowplow",
  reflected: "Reflected shock expanding outward",
  pinch: "PEAK COMPRESSION — fusion zone active",
  post_pinch: "Pinch disrupting via m=0 instability",
};

const SPEEDS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];

const GLOW_MESHES = new Set(["sheath", "pinch", "halo", "trail"]);

let activeCmap = VIRIDIS;

function cmap(t) {
  const n = activeCmap.length - 1;
  const i = Math.min(n - 1, Math.max(0, Math.floor(t * n)));
  const f = t * n - i;
  const a = activeCmap[i], b = activeCmap[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
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

// Hot colormap for temperature: blue -> cyan -> yellow -> white
const HOT_CMAP = [
  [0.0, 0.0, 0.5], [0.0, 0.3, 0.8], [0.0, 0.7, 0.9],
  [0.2, 0.9, 0.7], [0.6, 0.95, 0.3], [1.0, 0.9, 0.1],
  [1.0, 0.7, 0.0], [1.0, 0.95, 0.6], [1.0, 1.0, 1.0],
];

// B-magnitude colormap: blue (weak) -> white (strong)
const B_MAG_CMAP = [
  [0.1, 0.2, 0.6], [0.2, 0.4, 0.9], [0.4, 0.6, 1.0],
  [0.6, 0.8, 1.0], [0.8, 0.9, 1.0], [1.0, 1.0, 1.0],
];

function cmapLookup(t, table) {
  var v = Math.max(0, Math.min(1, t));
  var n = table.length - 1;
  var idx = v * n;
  var lo = Math.floor(idx), hi = Math.min(lo + 1, n);
  var f = idx - lo;
  return [
    table[lo][0] * (1 - f) + table[hi][0] * f,
    table[lo][1] * (1 - f) + table[hi][1] * f,
    table[lo][2] * (1 - f) + table[hi][2] * f,
  ];
}

function extractContourFromDensity(rhoFlat, nr, nz, dr, dz, threshold) {
  var contour = [];
  for (var iz = 0; iz < nz; iz++) {
    var rMax = 0;
    for (var ir = nr - 1; ir >= 0; ir--) {
      if (rhoFlat[ir * nz + iz] > threshold) { rMax = (ir + 0.5) * dr; break; }
    }
    contour.push(new BABYLON.Vector3(rMax, 0, iz * dz));
  }
  return contour;
}

function traceRK4(fieldR, fieldZ, nr, nz, dr, dz, r0, z0, ds, maxSteps) {
  var pts = [{r: r0, z: z0}];
  var r = r0, z = z0;
  for (var s = 0; s < maxSteps; s++) {
    var k1r = bilinearSample(fieldR, nr, nz, r / dr, z / dz);
    var k1z = bilinearSample(fieldZ, nr, nz, r / dr, z / dz);
    var mag1 = Math.sqrt(k1r * k1r + k1z * k1z) || 1e-20;
    k1r /= mag1; k1z /= mag1;

    var r2 = r + 0.5 * ds * k1r, z2 = z + 0.5 * ds * k1z;
    var k2r = bilinearSample(fieldR, nr, nz, r2 / dr, z2 / dz);
    var k2z = bilinearSample(fieldZ, nr, nz, r2 / dr, z2 / dz);
    var mag2 = Math.sqrt(k2r * k2r + k2z * k2z) || 1e-20;
    k2r /= mag2; k2z /= mag2;

    var r3 = r + 0.5 * ds * k2r, z3 = z + 0.5 * ds * k2z;
    var k3r = bilinearSample(fieldR, nr, nz, r3 / dr, z3 / dz);
    var k3z = bilinearSample(fieldZ, nr, nz, r3 / dr, z3 / dz);
    var mag3 = Math.sqrt(k3r * k3r + k3z * k3z) || 1e-20;
    k3r /= mag3; k3z /= mag3;

    var r4 = r + ds * k3r, z4 = z + ds * k3z;
    var k4r = bilinearSample(fieldR, nr, nz, r4 / dr, z4 / dz);
    var k4z = bilinearSample(fieldZ, nr, nz, r4 / dr, z4 / dz);
    var mag4 = Math.sqrt(k4r * k4r + k4z * k4z) || 1e-20;
    k4r /= mag4; k4z /= mag4;

    r += ds * (k1r + 2 * k2r + 2 * k3r + k4r) / 6;
    z += ds * (k1z + 2 * k2z + 2 * k3z + k4z) / 6;
    if (r < 0 || r > nr * dr || z < 0 || z > nz * dz) break;
    pts.push({r: r, z: z});
  }
  return pts;
}

async function createDPFScene(canvas, data) {
  const L = data;
  const G = L.geometry;
  const S = L.sheath;

  // ---- Engine ----
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
    } catch (_) {}
  }
  if (!engine) {
    engine = new BABYLON.Engine(canvas, true, {
      stencil: true, adaptToDeviceRatio: true, preserveDrawingBuffer: true,
    });
  }
  // Force native resolution on Retina/HiDPI — prevents blurriness
  engine.setHardwareScalingLevel(1 / window.devicePixelRatio);

  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.02, 0.02, 0.05, 1);

  // ---- Environment (HDR IBL + skybox — the single biggest quality booster) ----
  var env = null;
  try {
    env = scene.createDefaultEnvironment({
      createGround: false,
      createSkybox: true,
      skyboxSize: 5000,
      skyboxColor: new BABYLON.Color3(0.01, 0.01, 0.03),
      environmentTexture: BABYLON.CubeTexture.CreateFromPrefilteredData(HDR_ENV, scene),
    });
  } catch (_) {
    scene.environmentTexture = BABYLON.CubeTexture.CreateFromPrefilteredData(HDR_ENV, scene);
  }

  // ---- Camera ----
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

  // Auto-orbit: slow rotation when not interacting
  var autoOrbit = true;
  var userInteracting = false;
  var interactionTimeout = null;
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

  // ---- Lighting ----
  var hemiLight = new BABYLON.HemisphericLight("hemi",
    new BABYLON.Vector3(0, 1, 0), scene);
  hemiLight.intensity = 0.6;
  hemiLight.groundColor = new BABYLON.Color3(0.12, 0.12, 0.18);

  var keyLight = new BABYLON.PointLight("key",
    new BABYLON.Vector3(G.anode_length * 0.3, G.cathode_radius * 4, G.cathode_radius * 3), scene);
  keyLight.intensity = 1.0;
  keyLight.diffuse = new BABYLON.Color3(1, 0.96, 0.9);

  var fillLight = new BABYLON.PointLight("fill",
    new BABYLON.Vector3(G.anode_length * 0.8, -G.cathode_radius * 2, -G.cathode_radius * 2), scene);
  fillLight.intensity = 0.4;
  fillLight.diffuse = new BABYLON.Color3(0.75, 0.85, 1.0);

  // ============================================================
  // ELECTRODES — PBR with environment reflections
  // ============================================================
  var copperMat = new BABYLON.PBRMaterial("copper", scene);
  copperMat.metallic = 0.95;
  copperMat.roughness = 0.2;
  copperMat.albedoColor = new BABYLON.Color3(0.97, 0.75, 0.5);
  copperMat.emissiveColor = new BABYLON.Color3(0.04, 0.02, 0.01);
  copperMat.environmentIntensity = 1.5;

  var anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 128, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.forceSharedVertices();

  var steelMat = new BABYLON.PBRMaterial("steel", scene);
  steelMat.metallic = 0.9;
  steelMat.roughness = 0.3;
  steelMat.albedoColor = new BABYLON.Color3(0.78, 0.78, 0.82);
  steelMat.emissiveColor = new BABYLON.Color3(0.03, 0.03, 0.04);
  steelMat.environmentIntensity = 1.2;

  var cathodeRods = [];
  for (var i = 0; i < 8; i++) {
    var angle = (i / 8) * Math.PI * 2;
    var rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: G.cathode_radius * 0.1, height: G.anode_length, tessellation: 48,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(
      G.anode_length / 2,
      G.cathode_radius * Math.sin(angle),
      G.cathode_radius * Math.cos(angle)
    );
    rod.material = steelMat;
    rod.forceSharedVertices();
    cathodeRods.push(rod);
  }

  // Insulator — translucent ceramic
  var ceramicMat = new BABYLON.PBRMaterial("ceramic", scene);
  ceramicMat.metallic = 0;
  ceramicMat.roughness = 0.45;
  ceramicMat.albedoColor = new BABYLON.Color3(0.92, 0.88, 0.75);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.03, 0.025, 0.015);
  ceramicMat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;

  var insulator = BABYLON.MeshBuilder.CreateTorus("insulator", {
    diameter: G.cathode_radius * 2,
    thickness: G.anode_radius * 0.3,
    tessellation: 128,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -G.anode_radius * 0.15;
  insulator.material = ceramicMat;
  insulator.forceSharedVertices();

  // ============================================================
  // CURRENT SHEATH — Fresnel edge glow with depth pre-pass
  // ============================================================
  var sheathMat = new BABYLON.StandardMaterial("sheathMat", scene);
  sheathMat.emissiveColor = new BABYLON.Color3(0.3, 0.6, 1.0);
  sheathMat.alpha = 0.6;
  sheathMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
  sheathMat.needDepthPrePass = true;
  sheathMat.disableLighting = true;
  sheathMat.backFaceCulling = false;

  sheathMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  sheathMat.emissiveFresnelParameters.bias = 0.4;
  sheathMat.emissiveFresnelParameters.power = 2;
  sheathMat.emissiveFresnelParameters.leftColor = BABYLON.Color3.White();
  sheathMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.2, 0.4, 0.9);

  sheathMat.opacityFresnelParameters = new BABYLON.FresnelParameters();
  sheathMat.opacityFresnelParameters.bias = 0.5;
  sheathMat.opacityFresnelParameters.power = 1.5;

  var sheathMidR = (G.anode_radius + G.cathode_radius) / 2;
  var sheathTubeR = (G.cathode_radius - G.anode_radius) / 2;
  var sheath = BABYLON.MeshBuilder.CreateTorus("sheath", {
    diameter: sheathMidR * 2,
    thickness: sheathTubeR * 2,
    tessellation: 64,
  }, scene);
  sheath.rotation.z = Math.PI / 2;
  sheath.material = sheathMat;
  sheath.forceSharedVertices();

  // Plasma trail
  var trailMat = new BABYLON.StandardMaterial("trailMat", scene);
  trailMat.emissiveColor = new BABYLON.Color3(0.1, 0.15, 0.4);
  trailMat.alpha = 0.15;
  trailMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
  trailMat.needDepthPrePass = true;
  trailMat.disableLighting = true;
  trailMat.backFaceCulling = false;
  var trail = BABYLON.MeshBuilder.CreateTube("trail", {
    path: [new BABYLON.Vector3(0, 0, 0), new BABYLON.Vector3(1, 0, 0)],
    radius: (G.anode_radius + G.cathode_radius) / 2,
    tessellation: 32, cap: BABYLON.Mesh.NO_CAP, updatable: true,
  }, scene);
  trail.material = trailMat;

  // ============================================================
  // PINCH COLUMN — tube with m=0 instability ripple
  // ============================================================
  var pinchMat = new BABYLON.StandardMaterial("pinchMat", scene);
  pinchMat.emissiveColor = new BABYLON.Color3(1, 0.35, 0.08);
  pinchMat.disableLighting = true;
  pinchMat.backFaceCulling = false;
  pinchMat.alpha = 0;
  pinchMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
  pinchMat.needDepthPrePass = true;

  pinchMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  pinchMat.emissiveFresnelParameters.bias = 0.2;
  pinchMat.emissiveFresnelParameters.power = 3;
  pinchMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(1, 1, 0.9);
  pinchMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(1, 0.2, 0.05);

  // Pinch column at anode tip: spans last 15% of anode length
  // This is where the reflected shock creates the dense plasma core
  var N_PINCH = 24;
  var pinchPath = [];
  var pinchStart = G.anode_length * 0.85;
  var pinchEnd = G.anode_length * 1.02;
  for (var k = 0; k <= N_PINCH; k++) {
    pinchPath.push(new BABYLON.Vector3(
      pinchStart + (pinchEnd - pinchStart) * k / N_PINCH, 0, 0
    ));
  }
  var pinchRadii = new Array(N_PINCH + 1).fill(G.anode_radius * 0.3);
  var pinch = BABYLON.MeshBuilder.CreateTube("pinch", {
    path: pinchPath, radiusFunction: function(idx) { return pinchRadii[idx]; },
    tessellation: 48, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  pinch.material = pinchMat;

  // Halo glow around pinch
  var haloMat = new BABYLON.StandardMaterial("haloMat", scene);
  haloMat.emissiveColor = new BABYLON.Color3(0.7, 0.1, 0.03);
  haloMat.disableLighting = true;
  haloMat.alpha = 0;
  haloMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
  haloMat.needDepthPrePass = true;
  haloMat.backFaceCulling = false;
  var haloRadii = new Array(N_PINCH + 1).fill(G.anode_radius * 0.6);
  var halo = BABYLON.MeshBuilder.CreateTube("halo", {
    path: pinchPath, radiusFunction: function(idx) { return haloRadii[idx]; },
    tessellation: 48, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.BACKSIDE, updatable: true,
  }, scene);
  halo.material = haloMat;

  // ============================================================
  // FIELD DATA — midplane heatmap from REAL MHD field arrays
  // This is the ONE section that uses actual MHD solver output (when available).
  // The heatmap renders normalized 2D density/temperature/B-field data as a
  // textured plane in the electrode gap. Data arrives as base64-encoded Float32
  // arrays with shape [nr, nz], decoded into RGBA textures via colormap lookup.
  // Snapshot animation: multiple frames at different times are pre-decoded into
  // snapCache and swapped during playback based on nearest timestamp.
  // ============================================================
  var activeOverlay = "none";
  var heatPlane = null;
  var heatTex = null;

  // Create midplane plane: spans r=[anode, cathode], z=[0, anode_length]
  // Oriented in x-z plane (x=axial, z & y = radial cross-section)
  // Plane width = anode_length, height = cathode_radius - anode_radius
  var planeW = G.anode_length;
  var planeH = G.cathode_radius - G.anode_radius;
  heatPlane = BABYLON.MeshBuilder.CreatePlane("heatPlane", {
    width: planeW, height: planeH, sideOrientation: BABYLON.Mesh.DOUBLESIDE,
  }, scene);
  // Position: centered in the electrode gap, in the y=0 midplane
  heatPlane.position.x = planeW / 2;
  heatPlane.position.y = (G.anode_radius + G.cathode_radius) / 2;
  heatPlane.position.z = 0;
  heatPlane.rotation.z = -Math.PI / 2;
  heatPlane.rotation.y = -Math.PI / 2;
  heatPlane.isVisible = false;
  heatPlane.isPickable = false;

  var heatMat = new BABYLON.StandardMaterial("heatMat", scene);
  heatMat.disableLighting = true;
  heatMat.backFaceCulling = false;
  heatPlane.material = heatMat;

  function _cmapLookup(v, cmap) {
    var t = Math.max(0, Math.min(1, v));
    var idx = t * (cmap.length - 1);
    var lo = Math.floor(idx), hi = Math.min(lo + 1, cmap.length - 1);
    var f = idx - lo;
    return [
      cmap[lo][0] * (1 - f) + cmap[hi][0] * f,
      cmap[lo][1] * (1 - f) + cmap[hi][1] * f,
      cmap[lo][2] * (1 - f) + cmap[hi][2] * f,
    ];
  }

  // Pre-decode all snapshot frames for each field layer into cached RGBA Uint8Arrays.
  // Shape: snapCache[fieldKey] = { times: Float64Array, rgba: Uint8Array[], texW, texH }
  // Built once at scene creation so applyFrame never base64-decodes per tick.
  var snapCache = {};

  function _b64ToFloat32(b64) {
    var raw = atob(b64);
    var buf = new ArrayBuffer(raw.length);
    var bytes = new Uint8Array(buf);
    for (var ci = 0; ci < raw.length; ci++) bytes[ci] = raw.charCodeAt(ci);
    return new Float32Array(buf);
  }

  function _buildSnapCache(fieldKey, layer) {
    if (!layer || !layer.frames || layer.frames.length === 0) return;
    var shape = layer.frames_shape || layer.shape;
    if (!shape) return;
    var nr = shape[0], nz = shape[1];
    var texW = nz, texH = nr;
    var n = layer.frames.length;
    var times = new Float64Array(n);
    var rgbaFrames = new Array(n);
    for (var fi = 0; fi < n; fi++) {
      var frame = layer.frames[fi];
      times[fi] = frame.t_us;
      var vals = _b64ToFloat32(frame.data);
      var rgba = new Uint8Array(texW * texH * 4);
      for (var ir = 0; ir < nr; ir++) {
        for (var iz = 0; iz < nz; iz++) {
          var v = vals[ir * nz + iz];
          var c = _cmapLookup(v, activeCmap);
          var pi = ((nr - 1 - ir) * nz + iz) * 4;
          rgba[pi]     = Math.round(c[0] * 255);
          rgba[pi + 1] = Math.round(c[1] * 255);
          rgba[pi + 2] = Math.round(c[2] * 255);
          rgba[pi + 3] = 200;
        }
      }
      rgbaFrames[fi] = rgba;
    }
    snapCache[fieldKey] = { times: times, rgba: rgbaFrames, texW: texW, texH: texH };
  }

  // Build snap caches at scene creation for the three animated layers
  _buildSnapCache("density", L.density);
  _buildSnapCache("temperature", L.temperature);
  _buildSnapCache("bfield", L.bfield);

  // Raw field snap cache: pre-decoded Float32Arrays for geometry extraction
  // (density contours, field line tracing, velocity sampling, temperature coloring)
  var rawSnapCache = {};

  function _buildRawSnapCache(fieldKey, layer) {
    if (!layer || !layer.frames || layer.frames.length === 0) return;
    var shape = layer.frames_shape || layer.shape;
    if (!shape) return;
    var nr = shape[0], nz = shape[1];
    var n = layer.frames.length;
    var times = new Float64Array(n);
    var rawFrames = [];
    for (var fi = 0; fi < n; fi++) {
      var frame = layer.frames[fi];
      times[fi] = frame.t_us;
      var entry = { data: _b64ToFloat32(frame.data) };
      if (frame.Br) entry.Br = _b64ToFloat32(frame.Br);
      if (frame.Bz) entry.Bz = _b64ToFloat32(frame.Bz);
      if (frame.Bt) entry.Bt = _b64ToFloat32(frame.Bt);
      if (frame.vr) entry.vr = _b64ToFloat32(frame.vr);
      if (frame.vz) entry.vz = _b64ToFloat32(frame.vz);
      if (frame.vmag) entry.vmag = _b64ToFloat32(frame.vmag);
      rawFrames.push(entry);
    }
    rawSnapCache[fieldKey] = { times: times, frames: rawFrames, nr: nr, nz: nz };
  }

  _buildRawSnapCache("density", L.density);
  _buildRawSnapCache("bfield", L.bfield);
  _buildRawSnapCache("temperature", L.temperature);
  if (L.velocity) _buildRawSnapCache("velocity", L.velocity);

  function _nearestRawSnapIdx(fieldKey, t_us) {
    var cache = rawSnapCache[fieldKey];
    if (!cache) return -1;
    var times = cache.times;
    var best = 0, bestDist = Math.abs(times[0] - t_us);
    for (var si = 1; si < times.length; si++) {
      var d = Math.abs(times[si] - t_us);
      if (d < bestDist) { bestDist = d; best = si; }
    }
    return best;
  }

  function _nearestSnapIdx(fieldKey, t_us) {
    var cache = snapCache[fieldKey];
    if (!cache) return -1;
    var times = cache.times;
    var best = 0, bestDist = Math.abs(times[0] - t_us);
    for (var si = 1; si < times.length; si++) {
      var d = Math.abs(times[si] - t_us);
      if (d < bestDist) { bestDist = d; best = si; }
    }
    return best;
  }

  // lastSnapIdx tracks the last applied snap index per field to avoid redundant texture swaps
  var lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };

  function _applySnapTexture(fieldKey) {
    var cache = snapCache[fieldKey];
    if (!cache) return false;
    var idx = lastSnapIdx[fieldKey];
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

  function updateHeatmap(key) {
    if (!L || key === "none") {
      if (heatPlane) heatPlane.isVisible = false;
      return;
    }
    // Pick the right layer data
    var layer = null;
    if (key === "density" && L.density) layer = L.density;
    else if (key === "temperature" && L.temperature) layer = L.temperature;
    else if (key === "bfield" && L.bfield) layer = L.bfield;
    else if (key === "radiation" && L.radiation) layer = L.radiation;
    else if (key === "yield" && L.yield_map) layer = L.yield_map;

    if (!layer || !layer.data || !layer.shape) {
      if (heatPlane) heatPlane.isVisible = false;
      return;
    }

    // If this layer has snapshot frames, use the most-recently-cached snap index.
    // If no snap index set yet, fall back to the final-state static data.
    if (snapCache[key] && lastSnapIdx[key] >= 0) {
      _applySnapTexture(key);
      return;
    }

    // Decode base64 float32 normalized data (final state / static layers)
    var raw = atob(layer.data);
    var buf = new ArrayBuffer(raw.length);
    var bytes = new Uint8Array(buf);
    for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    var vals = new Float32Array(buf);

    var nr = layer.shape[0], nz = layer.shape[1];
    // Build RGBA texture: map normalized value → colormap
    var texW = nz, texH = nr;
    var rgba = new Uint8Array(texW * texH * 4);
    for (var ir = 0; ir < nr; ir++) {
      for (var iz = 0; iz < nz; iz++) {
        var v = vals[ir * nz + iz];
        var c = _cmapLookup(v, activeCmap);
        var pi = ((nr - 1 - ir) * nz + iz) * 4;  // flip r for correct orientation
        rgba[pi] = Math.round(c[0] * 255);
        rgba[pi + 1] = Math.round(c[1] * 255);
        rgba[pi + 2] = Math.round(c[2] * 255);
        rgba[pi + 3] = 200;  // semi-transparent
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

  // ============================================================
  // B-FIELD LINES — azimuthal tori, brightness ~ I(t)
  // ============================================================
  var fieldLines = [];
  var fieldLineData = [];
  var N_RADII = 5, N_ZPOS = 4;
  for (var zi = 0; zi < N_ZPOS; zi++) {
    var zPos = G.anode_length * (0.15 + 0.7 * zi / (N_ZPOS - 1));
    for (var ri = 0; ri < N_RADII; ri++) {
      var minR = G.anode_radius * 1.4;
      var maxR = G.cathode_radius * 0.95;
      var baseR = minR + (maxR - minR) * ri / (N_RADII - 1);
      var bStrength = 1 - ri / N_RADII;
      var tube = BABYLON.MeshBuilder.CreateTorus("fl" + zi + "_" + ri, {
        diameter: baseR * 2,
        thickness: G.cathode_radius * 0.015 * (0.5 + bStrength),
        tessellation: 64,
      }, scene);
      tube.rotation.z = Math.PI / 2;
      tube.position.x = zPos;
      var lineMat = new BABYLON.StandardMaterial("flm" + zi + "_" + ri, scene);
      lineMat.emissiveColor = new BABYLON.Color3(
        0.1 + bStrength * 0.3, 0.3 + bStrength * 0.5, 0.8 + bStrength * 0.2
      );
      lineMat.disableLighting = true;
      lineMat.alpha = 0.35 + bStrength * 0.35;
      lineMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
      lineMat.needDepthPrePass = true;
      tube.material = lineMat;
      tube.isVisible = false;
      tube.forceSharedVertices();
      fieldLines.push(tube);
      fieldLineData.push({ baseR: baseR, zi: zi, ri: ri, zPos: zPos });
    }
  }

  // Poloidal field lines from MHD data — RK4 traced, rendered as tubes colored by |B|
  var poloidalTubes = [];
  var poloidalTubeMats = [];
  function _buildPoloidalFieldLineTubes(brData, bzData, bnx, bnz) {
    // Dispose old tubes
    for (var oi = 0; oi < poloidalTubes.length; oi++) {
      poloidalTubes[oi].dispose();
      poloidalTubeMats[oi].dispose();
    }
    poloidalTubes = [];
    poloidalTubeMats = [];

    var dr = G.anode_length / Math.max(bnx - 1, 1);
    var dz = (G.cathode_radius * 2) / Math.max(bnz - 1, 1);
    var dsStep = G.anode_length / 60 * 0.6;

    for (var s = 0; s < 8; s++) {
      var r0 = G.anode_length * (0.1 + 0.8 * s / 8);
      var z0 = G.cathode_radius;
      var traced = traceRK4(brData, bzData, bnx, bnz, dr, dz, r0, z0, dsStep, 80);
      if (traced.length < 4) continue;

      var tubePts = [];
      var bMags = [];
      for (var ti = 0; ti < traced.length; ti++) {
        var tp = traced[ti];
        tubePts.push(new BABYLON.Vector3(tp.r, 0, tp.z - G.cathode_radius));
        var br = bilinearSample(brData, bnx, bnz, tp.r / dr, tp.z / dz);
        var bz = bilinearSample(bzData, bnx, bnz, tp.r / dr, tp.z / dz);
        bMags.push(Math.sqrt(br * br + bz * bz));
      }
      var bMax = Math.max.apply(null, bMags) || 1;

      // Mid-point color from |B| magnitude
      var midIdx = Math.floor(bMags.length / 2);
      var bNorm = Math.min(1, bMags[midIdx] / bMax);
      var col = cmapLookup(bNorm, B_MAG_CMAP);

      var tubeMat = new BABYLON.StandardMaterial("flt" + s, scene);
      tubeMat.emissiveColor = new BABYLON.Color3(col[0], col[1], col[2]);
      tubeMat.disableLighting = true;
      tubeMat.alpha = 0.6;
      tubeMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
      tubeMat.needDepthPrePass = true;
      tubeMat.backFaceCulling = false;

      var tubeR = G.cathode_radius * 0.012 * (0.5 + bNorm * 0.5);
      var tube = BABYLON.MeshBuilder.CreateTube("flt" + s, {
        path: tubePts, radius: tubeR, tessellation: 12,
        cap: BABYLON.Mesh.CAP_ALL, updatable: false,
      }, scene);
      tube.material = tubeMat;
      tube.isVisible = false;
      poloidalTubes.push(tube);
      poloidalTubeMats.push(tubeMat);
      fieldLines.push(tube);
    }
  }

  if (L.bfield) {
    try {
      var fdBr = decodeBase64Float32(L.bfield.Br, L.bfield.shape);
      var fdBz = decodeBase64Float32(L.bfield.Bz, L.bfield.shape);
      _buildPoloidalFieldLineTubes(fdBr.data, fdBz.data, fdBr.shape[0], fdBr.shape[1]);
    } catch(_) {}
  }

  // ============================================================
  // DENSITY ISOSURFACE — revolved contour from MHD density via CreateLathe
  // Falls back to Lee torus sheath when MHD density data unavailable
  // ============================================================
  var isoMesh = null;
  var isoMat = new BABYLON.StandardMaterial("isoMat", scene);
  isoMat.emissiveColor = new BABYLON.Color3(0.2, 0.5, 1.0);
  isoMat.alpha = 0.5;
  isoMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
  isoMat.needDepthPrePass = true;
  isoMat.disableLighting = true;
  isoMat.backFaceCulling = false;

  isoMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  isoMat.emissiveFresnelParameters.bias = 0.3;
  isoMat.emissiveFresnelParameters.power = 2;
  isoMat.emissiveFresnelParameters.leftColor = BABYLON.Color3.White();
  isoMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.15, 0.35, 0.9);

  var lastIsoSnapIdx = -1;

  function _updateIsosurface(rhoFlat, nr, nz) {
    var dr = (G.cathode_radius - G.anode_radius) / Math.max(nr - 1, 1);
    var dz = G.anode_length / Math.max(nz - 1, 1);
    // Threshold at 20% of normalized range (data is 0-1 normalized)
    var contour = extractContourFromDensity(rhoFlat, nr, nz, dr, dz, 0.2);
    if (contour.length < 3) return;

    // Transform contour: x-axis is axial, radial in y-z plane
    var lathePath = [];
    for (var ci = 0; ci < contour.length; ci++) {
      lathePath.push(new BABYLON.Vector3(
        contour[ci].z + G.anode_radius,
        contour[ci].x,
        0
      ));
    }

    if (isoMesh) isoMesh.dispose();
    try {
      isoMesh = BABYLON.MeshBuilder.CreateLathe("isoSurf", {
        shape: lathePath, tessellation: 48, closed: true, updatable: false,
        sideOrientation: BABYLON.Mesh.DOUBLESIDE,
      }, scene);
      isoMesh.rotation.z = Math.PI / 2;
      isoMesh.position.x = G.anode_length / 2;
      isoMesh.material = isoMat;
      GLOW_MESHES.add("isoSurf");
    } catch(_) { isoMesh = null; }
  }

  // ============================================================
  // CURRENT DENSITY J_theta TUBE — computed from dBz/dr - dBr/dz
  // ============================================================
  var jTube = null;
  var jTubeMat = new BABYLON.StandardMaterial("jMat", scene);
  jTubeMat.emissiveColor = new BABYLON.Color3(1.0, 0.5, 0.1);
  jTubeMat.disableLighting = true;
  jTubeMat.alpha = 0;
  jTubeMat.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
  jTubeMat.needDepthPrePass = true;
  jTubeMat.backFaceCulling = false;
  var lastJSnapIdx = -1;

  function _updateJTube(brData, bzData, nr, nz, sheathRfrac) {
    var dr = (G.cathode_radius - G.anode_radius) / Math.max(nr - 1, 1);
    var dz = G.anode_length / Math.max(nz - 1, 1);
    // Sample J_theta along z at sheath radius
    var irSheath = Math.min(nr - 2, Math.max(1, Math.floor(sheathRfrac * (nr - 1))));
    var jVals = [];
    var jPath = [];
    for (var iz = 1; iz < nz - 1; iz++) {
      // J_theta ~ dBz/dr - dBr/dz (central differences)
      var dBzdr = (bzData[(irSheath + 1) * nz + iz] - bzData[(irSheath - 1) * nz + iz]) / (2 * dr);
      var dBrdz = (brData[irSheath * nz + iz + 1] - brData[irSheath * nz + iz - 1]) / (2 * dz);
      var jTheta = Math.abs(dBzdr - dBrdz);
      jVals.push(jTheta);
      jPath.push(new BABYLON.Vector3(iz * dz, 0, 0));
    }
    if (jPath.length < 3) return;

    var jMax = Math.max.apply(null, jVals) || 1;
    var midJ = jVals[Math.floor(jVals.length / 2)] / jMax;
    var jCol = cmapLookup(midJ, HOT_CMAP);

    if (jTube) jTube.dispose();
    var jRadii = [];
    for (var ji = 0; ji < jVals.length; ji++) {
      jRadii.push(G.anode_radius * 0.05 * (0.2 + 0.8 * jVals[ji] / jMax));
    }
    try {
      jTube = BABYLON.MeshBuilder.CreateTube("jTube", {
        path: jPath,
        radiusFunction: function(idx) { return jRadii[Math.min(idx, jRadii.length - 1)]; },
        tessellation: 16, cap: BABYLON.Mesh.CAP_ALL, updatable: false,
      }, scene);
      jTube.position.y = (irSheath * dr + G.anode_radius);
      jTubeMat.emissiveColor.set(jCol[0], jCol[1], jCol[2]);
      jTubeMat.alpha = 0.5;
      jTube.material = jTubeMat;
    } catch(_) { jTube = null; }
  }

  // ============================================================
  // PARTICLE SYSTEM — built-in fire preset + custom ion particles
  // ============================================================
  var useGPU = BABYLON.GPUParticleSystem.IsSupported;
  var fireSet = null;

  // Ion particle system (always active, phase-adaptive)
  var PSClass = useGPU ? BABYLON.GPUParticleSystem : BABYLON.ParticleSystem;
  var psCap = useGPU ? 50000 : 4000;
  var ps = new PSClass("ions", { capacity: psCap }, scene);
  ps.emitter = new BABYLON.Vector3(0, 0, 0);

  var psEmitter = new BABYLON.SphereParticleEmitter();
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

  // Soft gaussian texture
  var ptexSize = 64;
  var ptex = new BABYLON.DynamicTexture("ptex", ptexSize, scene, false);
  var ptxCtx = ptex.getContext();
  var grad = ptxCtx.createRadialGradient(
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

  // Fire preset for pinch (loaded async, activated during pinch phase)
  try {
    BABYLON.ParticleHelper.CreateAsync("fire", scene).then(function(set) {
      fireSet = set;
      set.systems.forEach(function(sys) {
        sys.emitter = new BABYLON.Vector3(G.anode_length * 0.8, 0, 0);
        sys.minSize *= 0.3;
        sys.maxSize *= 0.3;
        sys.minEmitPower *= 0.5;
        sys.maxEmitPower *= 0.5;
      });
    }).catch(function() {});
  } catch(_) {}

  // ============================================================
  // POST-PROCESSING PIPELINE
  // ============================================================
  var pipeline = new BABYLON.DefaultRenderingPipeline("pipeline", true, scene, [cam]);
  pipeline.bloomEnabled = true;
  pipeline.bloomThreshold = 0.4;
  pipeline.bloomWeight = 0.5;
  pipeline.bloomKernel = 64;
  pipeline.bloomScale = 0.5;
  pipeline.fxaaEnabled = true;
  pipeline.samples = 4;
  pipeline.imageProcessingEnabled = true;
  pipeline.imageProcessing.toneMappingEnabled = true;
  pipeline.imageProcessing.toneMappingType = BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pipeline.imageProcessing.exposure = 1.4;
  pipeline.imageProcessing.contrast = 1.1;
  pipeline.chromaticAberrationEnabled = false;
  pipeline.chromaticAberration.aberrationAmount = 0;
  pipeline.sharpenEnabled = true;
  pipeline.sharpen.edgeAmount = 0.2;

  var ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.6;
    ssao.radius = 1.5;
    ssao.samples = 16;
    ssao.base = 0.2;
  } catch (_) {}

  // ---- Glow: include ALL meshes, suppress non-glowing via color selector ----
  var glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 32, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.5;
  glowLayer.customEmissiveColorSelector = function(mesh, _sub, _mat, result) {
    if (GLOW_MESHES.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      result.set(
        mesh.material.emissiveColor.r,
        mesh.material.emissiveColor.g,
        mesh.material.emissiveColor.b,
        mesh.material.alpha || 0
      );
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  // ============================================================
  // SCENE CONTROLLER
  // ============================================================
  return {
    engine: engine,
    scene: scene,
    camera: cam,
    gpuBackend: gpuBackend,
    useGPU: useGPU,

    sheath: sheath, sheathMat: sheathMat, trail: trail, trailMat: trailMat,
    pinch: pinch, pinchMat: pinchMat, halo: halo, haloMat: haloMat,
    pinchRadii: pinchRadii, haloRadii: haloRadii, pinchPath: pinchPath, N_PINCH: N_PINCH,
    anode: anode, cathodeRods: cathodeRods, insulator: insulator,
    ps: ps, psEmitter: psEmitter, fireSet: fireSet,
    pipeline: pipeline, ssao: ssao, glowLayer: glowLayer,
    updateHeatmap: updateHeatmap,
    fieldLines: fieldLines, fieldLineData: fieldLineData,
    poloidalTubes: poloidalTubes, isoMesh: isoMesh, jTube: jTube,
    activeOverlay: activeOverlay,
    G: G, S: S, L: L,

    setOverlay: function(key) {
      activeOverlay = key;
    },

    setCmap: function(useCividis) {
      activeCmap = useCividis ? CIVIDIS : VIRIDIS;
      // Rebuild snapshot RGBA caches with the new colormap
      snapCache = {};
      lastSnapIdx = { density: -1, temperature: -1, bfield: -1 };
      _buildSnapCache("density", L.density);
      _buildSnapCache("temperature", L.temperature);
      _buildSnapCache("bfield", L.bfield);
      if (activeOverlay !== "none" && heatPlane && heatPlane.isVisible) {
        updateHeatmap(activeOverlay);
      }
    },

    applyFrame: function(i) {
      if (i < 0 || i >= S.frames.length) return;
      var f = S.frames[i];

      // Update heatmap texture to the nearest MHD snapshot for this frame time.
      if (activeOverlay !== "none" && snapCache[activeOverlay]) {
        var newIdx = _nearestSnapIdx(activeOverlay, f.t);
        if (newIdx !== lastSnapIdx[activeOverlay]) {
          lastSnapIdx[activeOverlay] = newIdx;
          _applySnapTexture(activeOverlay);
        }
      }

      var col = PHASE_COLORS[f.phase] || [0.3, 0.3, 0.4];
      var isP = ["radial", "mhd_radial", "pinch", "reflected", "post_pinch"].indexOf(f.phase) >= 0;
      var cr = Math.max(0.02, f.r / G.cathode_radius);
      var pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
      if (f.phase === "post_pinch") pI *= 0.3;
      if (f.phase === "reflected") pI *= 0.5;

      // Resolve nearest raw density/bfield/velocity/temperature snap indices
      var densIdx = _nearestRawSnapIdx("density", f.t);
      var bfIdx = _nearestRawSnapIdx("bfield", f.t);
      var velIdx = _nearestRawSnapIdx("velocity", f.t);
      var tempIdx = _nearestRawSnapIdx("temperature", f.t);
      var hasMHDDensity = rawSnapCache["density"] && densIdx >= 0;
      var hasMHDBfield = rawSnapCache["bfield"] && bfIdx >= 0;
      var hasMHDVelocity = rawSnapCache["velocity"] && velIdx >= 0;
      var hasMHDTemperature = rawSnapCache["temperature"] && tempIdx >= 0;

      // =============================================================
      // UPGRADE 1: Density isosurface via CreateLathe (MHD-driven sheath)
      // Falls back to Lee torus when MHD density unavailable
      // =============================================================
      var useMHDSheath = false;
      if (hasMHDDensity && densIdx !== lastIsoSnapIdx) {
        lastIsoSnapIdx = densIdx;
        var rdc = rawSnapCache["density"];
        try {
          _updateIsosurface(rdc.frames[densIdx].data, rdc.nr, rdc.nz);
          useMHDSheath = true;
        } catch(_) {}
      }
      if (hasMHDDensity && isoMesh) {
        useMHDSheath = true;
        isoMesh.isVisible = true;
        isoMat.alpha = Math.min(0.6, 0.15 + Math.abs(f.I / S.I_peak) * 0.45);
        isoMat.emissiveColor.set(col[0], col[1], col[2]);
      }

      // Lee torus sheath (fallback when no MHD data, or always visible as ghost overlay)
      sheath.position.x = isP ? G.anode_length : f.z;
      sheathMat.emissiveColor.set(col[0], col[1], col[2]);
      if (isP) {
        var compScale = Math.max(0.03, cr);
        sheath.scaling.set(1, compScale, compScale);
      } else {
        var zFrac = Math.min(1, f.z / G.anode_length);
        var rundownScale = 1.0 - zFrac * (1.0 - Math.max(0.03, cr)) * 0.3;
        sheath.scaling.set(1, rundownScale, rundownScale);
      }
      var sheathAlpha = Math.min(0.7, 0.1 + Math.abs(f.I / S.I_peak) * 0.6);
      if (useMHDSheath) sheathAlpha *= 0.3;
      sheathMat.alpha = sheathAlpha;
      if ((f.phase === "post_pinch" || f.phase === "reflected") && Math.abs(f.I / S.I_peak) < 0.1) {
        sheath.isVisible = false;
        if (isoMesh) isoMesh.isVisible = false;
      } else {
        sheath.isVisible = true;
      }

      // Trail
      var tLen = Math.max(isP ? G.anode_length : f.z, 0.2);
      trail.scaling.x = tLen;
      trail.position.x = tLen / 2;
      if (isP) {
        var trailScale = Math.max(0.05, cr);
        trail.scaling.y = trailScale;
        trail.scaling.z = trailScale;
      } else {
        trail.scaling.y = 1;
        trail.scaling.z = 1;
      }
      trailMat.emissiveColor.set(col[0] * 0.25, col[1] * 0.25, col[2] * 0.3);
      trailMat.alpha = Math.min(0.12, 0.05 + Math.abs(f.I) * 0.04);

      // =============================================================
      // UPGRADE 8: Instability perturbation from MHD frame data
      // =============================================================
      var instAmp = L.instability ? L.instability.amplitude : 0;
      // If frame carries instability data, override with it
      if (f.tau_m0 !== undefined && f.n_efolds !== undefined) {
        instAmp = Math.min(1.0, Math.expm1(Math.min(f.n_efolds, 50)));
      }
      var rippleAmp = isP ? instAmp * Math.min(1, (1 - cr) * 2) : 0;

      var pinchR = Math.max(G.anode_radius * 0.05, cr * G.cathode_radius * 0.25);
      var pinchLen = pinchEnd - pinchStart;
      var nModes = Math.max(1, Math.round(pinchLen / (2 * Math.PI * Math.max(pinchR, 0.001))));
      nModes = Math.min(nModes, 6);

      for (var pk = 0; pk <= N_PINCH; pk++) {
        var zFrac2 = pk / N_PINCH;
        var taper = Math.sin(Math.PI * zFrac2);
        var localR = pinchR * (0.5 + 0.5 * taper);
        var ripple = rippleAmp * localR * Math.cos(2 * Math.PI * nModes * zFrac2);
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

      var pinchPhase = f.phase === "pinch" || f.phase === "post_pinch" || f.phase === "reflected";
      var pinchVisible = pinchPhase || (isP && cr < 0.3);
      pinch.isVisible = pinchVisible;
      halo.isVisible = pinchVisible;
      pinchMat.alpha = pinchVisible ? pI * 0.85 : 0;
      haloMat.alpha = pinchVisible ? pI * 0.25 : 0;
      if (pinchMat.emissiveColor) pinchMat.emissiveColor.set(1, 0.15 + pI * 0.5, pI * 0.3);
      haloMat.emissiveColor.set(0.8, 0.08 + pI * 0.15, 0.03);
      glowLayer.intensity = 0.35 + pI * 1.2;

      // =============================================================
      // UPGRADE 5: Temperature color on isosurface
      // =============================================================
      if (useMHDSheath && isoMesh && hasMHDTemperature) {
        var trc = rawSnapCache["temperature"];
        var tData = trc.frames[tempIdx].data;
        // Sample temperature at density peak location (approximate center)
        var tMid = tData[Math.floor(tData.length / 2)] || 0.5;
        var tCol = cmapLookup(Math.min(1, tMid), HOT_CMAP);
        isoMat.emissiveColor.set(tCol[0], tCol[1], tCol[2]);
      }

      // Fire preset during pinch
      if (fireSet) {
        if (pI > 0.5 && !fireSet._started) {
          fireSet.start();
          fireSet._started = true;
        } else if (pI < 0.1 && fireSet._started) {
          fireSet.dispose();
          fireSet = null;
        }
      }

      // =============================================================
      // UPGRADE 4: Velocity-driven particles (MHD velocity field)
      // Falls back to Lee 0D scalars when velocity data unavailable
      // =============================================================
      ps.emitter.x = isP ? G.anode_length : f.z;
      if (f.phase === "rundown") {
        psEmitter.radius = G.cathode_radius * 0.85;
        ps.gravity = new BABYLON.Vector3(1.5, 0, 0);
        ps.minEmitPower = 0.5; ps.maxEmitPower = 2;
        ps.emitRate = useGPU ? 6000 : 400;
      } else if (isP) {
        var compR = Math.max(f.r, G.anode_radius * 0.05);
        psEmitter.radius = compR * 0.8;
        var boost = Math.min(8, Math.pow(G.cathode_radius / Math.max(compR, 0.1), 1.5));
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

      // MHD velocity override: set particle direction from v_r, v_z at sheath front
      if (hasMHDVelocity && isP) {
        var vc = rawSnapCache["velocity"];
        var vf = vc.frames[velIdx];
        if (vf.vr && vf.vz) {
          var sheathR_frac = cr * (vc.nr - 1);
          var sheathZ_frac = (f.z / G.anode_length) * (vc.nz - 1);
          var vr = bilinearSample(vf.vr, vc.nr, vc.nz, sheathR_frac, sheathZ_frac);
          var vz = bilinearSample(vf.vz, vc.nr, vc.nz, sheathR_frac, sheathZ_frac);
          var vMag = Math.sqrt(vr * vr + vz * vz) || 1e-10;
          // Map (vz -> x-axis axial, vr -> radial y)
          ps.direction1 = new BABYLON.Vector3(vz / vMag, vr / vMag, 0);
          ps.direction2 = new BABYLON.Vector3(vz / vMag, -vr / vMag, 0);
          ps.emitRate = Math.min(useGPU ? 50000 : 4000, Math.abs(vr + vz) * 100 + 500);
        }
      }

      // =============================================================
      // UPGRADE 2 & 3: RK4 animated field lines as tubes colored by |B|
      // Update poloidal field line tubes from per-frame Br/Bz snap data
      // =============================================================
      if (hasMHDBfield && bfIdx !== lastJSnapIdx) {
        var bfc = rawSnapCache["bfield"];
        var bfFrame = bfc.frames[bfIdx];
        if (bfFrame.Br && bfFrame.Bz) {
          _buildPoloidalFieldLineTubes(bfFrame.Br, bfFrame.Bz, bfc.nr, bfc.nz);
          // UPGRADE 6: Update J_theta tube from curl(B)
          _updateJTube(bfFrame.Br, bfFrame.Bz, bfc.nr, bfc.nz, cr);
          lastJSnapIdx = bfIdx;
        }
      }

      // =============================================================
      // UPGRADE 7: Dynamic B-field ring scaling from MHD B_theta
      // Falls back to Lee I/I_peak brightness when B_theta unavailable
      // =============================================================
      var Ifrac = Math.abs(f.I) / Math.max(S.I_peak, 0.001);
      var useMHDBRings = hasMHDBfield && rawSnapCache["bfield"].frames[bfIdx] &&
                         rawSnapCache["bfield"].frames[bfIdx].Bt;
      for (var fli = 0; fli < fieldLines.length; fli++) {
        if (!fieldLines[fli].isVisible) continue;
        var fld = fieldLineData[fli];
        if (!fld) continue;
        var bStr = 1 - fld.ri / N_RADII;

        // Ring brightness and radius: prefer MHD B_theta, fallback to Lee I/I_peak
        var ringBrightness = Ifrac;
        var ringScale = 1.0;
        if (useMHDBRings) {
          var btCache = rawSnapCache["bfield"];
          var btData = btCache.frames[bfIdx].Bt;
          var rFrac = (fld.baseR / G.cathode_radius) * (btCache.nr - 1);
          var zFrac3 = (fld.zPos / G.anode_length) * (btCache.nz - 1);
          var btVal = Math.abs(bilinearSample(btData, btCache.nr, btCache.nz, rFrac, zFrac3));
          var btMax = 1.0;
          for (var bsi = 0; bsi < btData.length; bsi++) {
            if (Math.abs(btData[bsi]) > btMax) btMax = Math.abs(btData[bsi]);
          }
          ringBrightness = Math.min(1, btVal / btMax);
          ringScale = 0.5 + 0.5 * ringBrightness;
        }

        if (fieldLines[fli].material) {
          fieldLines[fli].material.alpha = Math.min(0.85, (0.1 + bStr * 0.3) * ringBrightness * 2);
          if (fieldLines[fli].material.emissiveColor) {
            var glow = Math.min(1, ringBrightness * 1.5);
            fieldLines[fli].material.emissiveColor.set(
              0.1 + bStr * 0.3 * glow, 0.3 + bStr * 0.5 * glow, 0.7 + bStr * 0.3 * glow
            );
          }
        }

        if (isP) {
          var scaleFactor = cr + (1 - cr) * (fld.ri / N_RADII);
          if (useMHDBRings) scaleFactor *= ringScale;
          fieldLines[fli].scaling.y = Math.max(0.05, scaleFactor);
          fieldLines[fli].scaling.z = Math.max(0.05, scaleFactor);
        } else {
          fieldLines[fli].scaling.y = useMHDBRings ? ringScale : 1;
          fieldLines[fli].scaling.z = useMHDBRings ? ringScale : 1;
        }

        if (f.phase === "rundown" && fld.zi >= 2) {
          fieldLines[fli].position.x = Math.min(fld.zPos, f.z);
        } else {
          fieldLines[fli].position.x = fld.zPos;
        }
      }

      // J tube visibility tied to B-field rings
      if (jTube) {
        jTube.isVisible = hasMHDBfield && fieldLines.length > 0 && fieldLines[0].isVisible;
      }

      // ---- Cinematic effects ----
      if (isP && pI > 0.3) {
        pipeline.depthOfFieldEnabled = true;
        pipeline.depthOfField.focalLength = 60;
        pipeline.depthOfField.fStop = 2;
        pipeline.depthOfField.focusDistance =
          BABYLON.Vector3.Distance(cam.position, pinch.position) * 1000;
      } else {
        pipeline.depthOfFieldEnabled = false;
      }

      if (pI > 0.5) {
        pipeline.chromaticAberrationEnabled = true;
        pipeline.chromaticAberration.aberrationAmount = pI * 30;
      } else {
        pipeline.chromaticAberrationEnabled = false;
      }

      if (isP && pI > 0.2 && autoOrbit) {
        var targetRadius = G.cathode_radius * (3 + (1 - pI) * 4);
        cam.radius += (targetRadius - cam.radius) * 0.02;
      } else if (autoOrbit && !userInteracting) {
        var defaultRadius = G.cathode_radius * 7;
        cam.radius += (defaultRadius - cam.radius) * 0.01;
      }

      if (pI > 0.8) {
        pipeline.imageProcessing.exposure = 1.4 + (pI - 0.8) * 3;
      } else {
        pipeline.imageProcessing.exposure += (1.4 - pipeline.imageProcessing.exposure) * 0.1;
      }

      return { f: f, col: col, isP: isP, cr: cr, pI: pI, rippleAmp: rippleAmp };
    },
  };
}

window.createDPFScene = createDPFScene;
window.PHASE_LABELS = PHASE_LABELS;
window.PHASE_DESCRIPTIONS = PHASE_DESCRIPTIONS;
window.SPEEDS = SPEEDS;
