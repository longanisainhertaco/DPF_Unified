/**
 * DPF-Unified Plasma Renderer — Babylon.js 8.x
 *
 * Conference-quality 3D physics visualization.
 *
 * DATA SOURCE HONESTY NOTE:
 * -------------------------
 * The 3D scene elements are driven by TWO different data sources:
 *
 * LEE MODEL SCALARS (0D ODE output — always used for these elements):
 *   - Current sheath position (z_mm) and radius (r_mm)
 *   - Sheath scaling, compression ratio, and alpha
 *   - Pinch column radius and m=0 instability ripple amplitude
 *   - Particle system emitter position, radius, and rate
 *   - B-field ring brightness (proportional to I/I_peak)
 *   - Trail tube length and scaling
 *   - All phase-dependent visual effects (colors, glow, DOF, chromatic aberration)
 *
 * MHD FIELD DATA (2D arrays from Metal/Python solvers — when available):
 *   - Midplane heatmaps: density, temperature, |B| (toggled via layer panel)
 *   - Poloidal field lines (static, traced from final-state Br/Bz arrays)
 *   - HUD peak values (rho_peak, Te_peak, |B|_peak)
 *
 * WHAT WOULD NEED TO CHANGE TO WIRE REAL MHD DATA INTO THE 3D SCENE:
 *   1. Sheath mesh: Replace Lee z_mm/r_mm with isosurface extraction from
 *      the MHD density field (e.g., rho > 5*rho_fill contour in r-z plane,
 *      revolved to 3D). Requires per-frame 2D density snapshots.
 *   2. Pinch column: Extract pinch geometry from density/pressure peak region
 *      in MHD data rather than scaling by Lee's cr ratio. m=0 instability
 *      ripple should come from actual density perturbation spectrum.
 *   3. Particle system: Use MHD velocity field to set particle emitter
 *      direction and power, density field for emitter radius/rate.
 *   4. B-field rings: Replace I/I_peak brightness with actual B_theta(r,z)
 *      from MHD snapshots; animate ring radii from field profile.
 *   5. updateHeatmap() already works for midplane slices. To go further:
 *      volumetric rendering or marching cubes isosurfaces from full 2D
 *      field arrays (density, temperature, |B|) at each snapshot time.
 *   6. Data pipeline: Metal solver -> mhd_snapshots[] (already encoded as
 *      base64 Float32 per frame) -> snapCache (pre-decoded RGBA) ->
 *      RawTexture per frame. The infrastructure exists; what's missing is
 *      extracting geometric primitives (sheath contour, pinch shape) from
 *      the field data instead of using Lee scalars.
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
  scene.clearColor = new BABYLON.Color4(0.88, 0.90, 0.92, 1);  // light neutral gray

  // ---- Environment: light studio HDR for realistic PBR reflections ----
  var env = null;
  try {
    env = scene.createDefaultEnvironment({
      createGround: true,
      groundSize: 20,
      groundColor: new BABYLON.Color3(0.85, 0.87, 0.90),
      groundOpacity: 0.4,
      createSkybox: true,
      skyboxSize: 5000,
      skyboxColor: new BABYLON.Color3(0.85, 0.87, 0.90),
      environmentTexture: BABYLON.CubeTexture.CreateFromPrefilteredData(HDR_ENV, scene),
    });
  } catch (_) {
    scene.environmentTexture = BABYLON.CubeTexture.CreateFromPrefilteredData(HDR_ENV, scene);
  }

  // ---- Ground grid for spatial reference (engineering viewport style) ----
  // Grid uses StandardMaterial with a line texture since GridMaterial may not
  // be available in all CDN bundles. Simple approach: large ground plane with
  // procedural grid lines drawn on a DynamicTexture.
  var gridSize = Math.max(G.anode_length * 3, G.cathode_radius * 6);
  var gridGround = BABYLON.MeshBuilder.CreateGround("grid", {
    width: gridSize, height: gridSize, subdivisions: 1,
  }, scene);
  gridGround.position.y = -G.cathode_radius * 1.2;
  gridGround.position.x = G.anode_length / 2;

  var gridTex = new BABYLON.DynamicTexture("gridTex", 512, scene, false);
  var gridCtx = gridTex.getContext();
  gridCtx.fillStyle = "rgba(210, 215, 220, 1.0)";
  gridCtx.fillRect(0, 0, 512, 512);
  // Draw grid lines
  gridCtx.strokeStyle = "rgba(160, 165, 175, 0.6)";
  gridCtx.lineWidth = 1;
  for (var gi = 0; gi <= 20; gi++) {
    var gpos = gi * 512 / 20;
    gridCtx.beginPath(); gridCtx.moveTo(gpos, 0); gridCtx.lineTo(gpos, 512); gridCtx.stroke();
    gridCtx.beginPath(); gridCtx.moveTo(0, gpos); gridCtx.lineTo(512, gpos); gridCtx.stroke();
  }
  // Major grid lines (thicker)
  gridCtx.strokeStyle = "rgba(120, 125, 135, 0.8)";
  gridCtx.lineWidth = 2;
  for (var gi = 0; gi <= 4; gi++) {
    var gpos = gi * 512 / 4;
    gridCtx.beginPath(); gridCtx.moveTo(gpos, 0); gridCtx.lineTo(gpos, 512); gridCtx.stroke();
    gridCtx.beginPath(); gridCtx.moveTo(0, gpos); gridCtx.lineTo(512, gpos); gridCtx.stroke();
  }
  gridTex.update();
  var gridMat = new BABYLON.StandardMaterial("gridMat", scene);
  gridMat.diffuseTexture = gridTex;
  gridMat.specularColor = new BABYLON.Color3(0, 0, 0);
  gridMat.alpha = 0.7;
  gridGround.material = gridMat;

  // ---- Key light: directional light for solid-looking electrodes ----
  var keyLight = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  keyLight.intensity = 1.5;
  keyLight.diffuse = new BABYLON.Color3(1, 0.98, 0.95);
  // Fill light from opposite side (softer)
  var fillLight = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fillLight.intensity = 0.6;
  fillLight.diffuse = new BABYLON.Color3(0.9, 0.92, 1.0);
  fillLight.groundColor = new BABYLON.Color3(0.3, 0.3, 0.35);

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
  // ELECTRODES — Fully opaque PBR, rendering group 0 (drawn first)
  // ============================================================
  var copperMat = new BABYLON.PBRMaterial("copper", scene);
  copperMat.metallic = 0.95;
  copperMat.roughness = 0.2;
  copperMat.albedoColor = new BABYLON.Color3(0.97, 0.75, 0.5);
  copperMat.emissiveColor = new BABYLON.Color3(0.05, 0.03, 0.01);
  copperMat.environmentIntensity = 1.5;
  copperMat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;

  var anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 64, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.renderingGroupId = 0;

  var steelMat = new BABYLON.PBRMaterial("steel", scene);
  steelMat.metallic = 0.85;
  steelMat.roughness = 0.25;
  steelMat.albedoColor = new BABYLON.Color3(0.75, 0.75, 0.80);
  steelMat.emissiveColor = new BABYLON.Color3(0.02, 0.02, 0.03);
  steelMat.environmentIntensity = 1.2;
  steelMat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;

  // 12 cathode rods (real devices have 8-24), thicker for visibility
  var cathodeRods = [];
  var N_RODS = 12;
  for (var i = 0; i < N_RODS; i++) {
    var angle = (i / N_RODS) * Math.PI * 2;
    var rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: G.cathode_radius * 0.08, height: G.anode_length * 1.05,
      tessellation: 16,
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

  // Insulator — opaque ceramic
  var ceramicMat = new BABYLON.PBRMaterial("ceramic", scene);
  ceramicMat.metallic = 0;
  ceramicMat.roughness = 0.5;
  ceramicMat.albedoColor = new BABYLON.Color3(0.95, 0.92, 0.85);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.04, 0.03, 0.02);
  ceramicMat.transparencyMode = BABYLON.Material.MATERIAL_OPAQUE;

  var insulator = BABYLON.MeshBuilder.CreateTorus("insulator", {
    diameter: G.cathode_radius * 2,
    thickness: G.anode_radius * 0.35,
    tessellation: 64,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -G.anode_radius * 0.15;
  insulator.material = ceramicMat;
  insulator.renderingGroupId = 0;

  // ============================================================
  // PLASMA EFFECTS — Additive blending, rendering group 1 (drawn AFTER opaques)
  // Additive blend (MATERIAL_ALPHABLEND + alphaMode=ADD) eliminates
  // z-fighting and transparency sorting issues entirely. Plasma light
  // adds on top of whatever is behind it — no depth conflicts.
  // ============================================================

  // Current sheath
  var sheathMat = new BABYLON.StandardMaterial("sheathMat", scene);
  sheathMat.emissiveColor = new BABYLON.Color3(0.3, 0.6, 1.0);
  sheathMat.disableLighting = true;
  sheathMat.backFaceCulling = false;
  sheathMat.alpha = 0.5;
  sheathMat.alphaMode = BABYLON.Engine.ALPHA_ADD;  // additive: no z-fighting

  sheathMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  sheathMat.emissiveFresnelParameters.bias = 0.3;
  sheathMat.emissiveFresnelParameters.power = 2;
  sheathMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(0.7, 0.85, 1.0);
  sheathMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.15, 0.3, 0.8);

  var sheathMidR = (G.anode_radius + G.cathode_radius) / 2;
  var sheathTubeR = (G.cathode_radius - G.anode_radius) / 2;
  var sheath = BABYLON.MeshBuilder.CreateTorus("sheath", {
    diameter: sheathMidR * 2,
    thickness: sheathTubeR * 2,
    tessellation: 48,
  }, scene);
  sheath.rotation.z = Math.PI / 2;
  sheath.material = sheathMat;
  sheath.renderingGroupId = 1;

  // Plasma trail — additive, very transparent
  var trailMat = new BABYLON.StandardMaterial("trailMat", scene);
  trailMat.emissiveColor = new BABYLON.Color3(0.08, 0.12, 0.3);
  trailMat.disableLighting = true;
  trailMat.backFaceCulling = false;
  trailMat.alpha = 0.1;
  trailMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  var trail = BABYLON.MeshBuilder.CreateTube("trail", {
    path: [new BABYLON.Vector3(0, 0, 0), new BABYLON.Vector3(1, 0, 0)],
    radius: (G.anode_radius + G.cathode_radius) / 2,
    tessellation: 24, cap: BABYLON.Mesh.NO_CAP, updatable: true,
  }, scene);
  trail.material = trailMat;
  trail.renderingGroupId = 1;

  // ============================================================
  // PINCH COLUMN — additive, rendering group 1
  // ============================================================
  var pinchMat = new BABYLON.StandardMaterial("pinchMat", scene);
  pinchMat.emissiveColor = new BABYLON.Color3(1, 0.4, 0.1);
  pinchMat.disableLighting = true;
  pinchMat.backFaceCulling = false;
  pinchMat.alpha = 0;
  pinchMat.alphaMode = BABYLON.Engine.ALPHA_ADD;

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
  pinch.renderingGroupId = 1;

  // Halo glow around pinch — additive
  var haloMat = new BABYLON.StandardMaterial("haloMat", scene);
  haloMat.emissiveColor = new BABYLON.Color3(0.7, 0.1, 0.03);
  haloMat.disableLighting = true;
  haloMat.alpha = 0;
  haloMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  haloMat.backFaceCulling = false;
  var haloRadii = new Array(N_PINCH + 1).fill(G.anode_radius * 0.6);
  var halo = BABYLON.MeshBuilder.CreateTube("halo", {
    path: pinchPath, radiusFunction: function(idx) { return haloRadii[idx]; },
    tessellation: 48, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.BACKSIDE, updatable: true,
  }, scene);
  halo.material = haloMat;
  halo.renderingGroupId = 1;

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

  // Create midplane heatmap as a cylindrical half-pipe (not a flat plane).
  // The plasma lives in the annular gap between anode and cathode radii.
  // A half-cylinder (pi radians arc) shows the r-z cross-section on a
  // curved surface that follows the electrode geometry.
  var planeW = G.anode_length;
  var planeH = G.cathode_radius - G.anode_radius;
  // Build a custom ribbon mesh: rows at different radii, columns along z-axis
  var _heatPaths = [];
  var _heatNr = 16;  // radial resolution for the curved surface
  var _heatNz = 32;  // axial resolution
  for (var _ir = 0; _ir <= _heatNr; _ir++) {
    var r = G.anode_radius + (G.cathode_radius - G.anode_radius) * _ir / _heatNr;
    var path = [];
    for (var _iz = 0; _iz <= _heatNz; _iz++) {
      var z = G.anode_length * _iz / _heatNz;
      // Map to 3D: x = z (axial), y = r * sin(angle), z_3d = r * cos(angle)
      // Use a ~240 degree arc so user can see inside from one side
      var angle = Math.PI * 0.33;  // offset to position the opening toward camera
      path.push(new BABYLON.Vector3(z, r * Math.sin(angle), r * Math.cos(angle)));
    }
    _heatPaths.push(path);
  }
  heatPlane = BABYLON.MeshBuilder.CreateRibbon("heatPlane", {
    pathArray: _heatPaths,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE,
    updatable: false,
  }, scene);
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

  // Poloidal field lines from MHD data
  if (L.bfield) {
    try {
      var fdBr = decodeBase64Float32(L.bfield.Br, L.bfield.shape);
      var fdBz = decodeBase64Float32(L.bfield.Bz, L.bfield.shape);
      var bnx = fdBr.shape[0], bnz = fdBr.shape[1];
      for (var s = 0; s < 8; s++) {
        var x = G.anode_length * (0.1 + 0.8 * s / 8), z = 0;
        var pts = [];
        var ds = G.anode_length / 60 * 0.6;
        for (var step = 0; step < 60; step++) {
          pts.push(new BABYLON.Vector3(x, 0, z));
          var fx = (x / G.anode_length) * (bnx - 1);
          var fz = ((z + G.cathode_radius) / (G.cathode_radius * 2)) * (bnz - 1);
          var br = bilinearSample(fdBr.data, bnx, bnz, fx, fz);
          var bz = bilinearSample(fdBz.data, bnx, bnz, fx, fz);
          var mag = Math.sqrt(br * br + bz * bz) + 1e-10;
          x += ds * br / mag; z += ds * bz / mag;
          if (x < 0 || x > G.anode_length || Math.abs(z) > G.cathode_radius) break;
        }
        if (pts.length > 4) {
          var line = BABYLON.MeshBuilder.CreateLines("flp" + s, { points: pts }, scene);
          line.color = new BABYLON.Color3(0.3, 0.7, 1.0);
          line.alpha = 0.5;
          line.isVisible = false;
          fieldLines.push(line);
        }
      }
    } catch(_) {}
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
      var ec = mesh.material.emissiveColor;
      var mag = Math.max(ec.r, ec.g, ec.b);
      var boost = mag > 1.0 ? Math.sqrt(mag) : 1.0;  // HDR bloom, sqrt-compressed
      result.set(ec.r * boost, ec.g * boost, ec.b * boost, mesh.material.alpha || 0);
    } else {
      result.set(0, 0, 0, 0);
    }
  };

  // ---- AAA rendering pipeline: HDR bloom + ACES tone mapping + FXAA ----
  // Scientifically accurate AND visually stunning.
  // The pinch IS blindingly bright — that's physically correct.
  // ACES tone mapping handles the dynamic range gracefully.
  var pipeline = new BABYLON.DefaultRenderingPipeline("dpfPipeline", true, scene, [cam]);
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

  var _pipeline = pipeline;

  // ---- Scale bar + axis labels for dimensional reference ----
  // Axial (z) scale bar along the anode
  var scaleLen = G.anode_length;
  var scaleMat = new BABYLON.StandardMaterial("scaleMat", scene);
  scaleMat.emissiveColor = new BABYLON.Color3(0.3, 0.3, 0.35);
  scaleMat.disableLighting = true;
  var scaleBar = BABYLON.MeshBuilder.CreateLines("scaleBar", {
    points: [
      new BABYLON.Vector3(0, -G.cathode_radius * 1.15, 0),
      new BABYLON.Vector3(scaleLen, -G.cathode_radius * 1.15, 0),
    ],
    colors: [new BABYLON.Color4(0.4, 0.4, 0.45, 1), new BABYLON.Color4(0.4, 0.4, 0.45, 1)],
  }, scene);
  // Tick marks at 0, 25%, 50%, 75%, 100% of anode length
  for (var ti = 0; ti <= 4; ti++) {
    var tx = scaleLen * ti / 4;
    var tick = BABYLON.MeshBuilder.CreateLines("tick" + ti, {
      points: [
        new BABYLON.Vector3(tx, -G.cathode_radius * 1.15 - 0.005, 0),
        new BABYLON.Vector3(tx, -G.cathode_radius * 1.15 + 0.005, 0),
      ],
      colors: [new BABYLON.Color4(0.4, 0.4, 0.45, 1), new BABYLON.Color4(0.4, 0.4, 0.45, 1)],
    }, scene);
  }

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
      // f.z, f.r, f.I, f.phase are Lee model 0D scalars from the circuit ODE.
      // ALL 3D geometry below (sheath, pinch, particles, B-field rings) is
      // derived from these scalars — NOT from MHD field arrays. Only the
      // optional midplane heatmap uses real MHD data when available.

      // Update heatmap texture to the nearest MHD snapshot for this frame time.
      // This IS real MHD field data (when available from metal_cylindrical or python backends).
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

      // LEE MODEL SCHEMATIC: Sheath position from Lee 0D scalar f.z (not MHD density field)
      sheath.position.x = isP ? G.anode_length : f.z;
      sheathMat.emissiveColor.set(col[0], col[1], col[2]);
      if (isP) {
        // Smooth compression: lerp from 1.0 toward cr over first few radial frames
        var compScale = Math.max(0.03, cr);
        sheath.scaling.set(1, compScale, compScale);
      } else {
        // During rundown, smoothly narrow as sheath approaches anode tip
        var zFrac = Math.min(1, f.z / G.anode_length);
        var rundownScale = 1.0 - zFrac * (1.0 - Math.max(0.03, cr)) * 0.3;
        sheath.scaling.set(1, rundownScale, rundownScale);
      }
      var sheathAlpha = Math.min(0.7, 0.1 + Math.abs(f.I / S.I_peak) * 0.6);
      sheathMat.alpha = sheathAlpha;
      if ((f.phase === "post_pinch" || f.phase === "reflected") && Math.abs(f.I / S.I_peak) < 0.1) {
        sheath.isVisible = false;
      } else {
        sheath.isVisible = true;
      }

      // Trail — compress radially during pinch, fade during post-pinch
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

      // Pinch with m=0 instability
      var instAmp = L.instability ? L.instability.amplitude : 0;
      var rippleAmp = isP ? instAmp * Math.min(1, (1 - cr) * 2) : 0;

      // LEE MODEL SCHEMATIC: Pinch radius from Lee 0D compression ratio (not MHD density peak)
      // Bennett equilibrium: a_B ~ r_sheath * sqrt(kT / (I^2 * mu0/(8*pi*N*k)))
      // Approximate: pinch core ~ 20-30% of minimum sheath radius
      var pinchR = Math.max(G.anode_radius * 0.05, cr * G.cathode_radius * 0.25);

      // m=0 sausage: most unstable wavelength ~ 2*pi*a (circumference)
      // Mode number adapts to aspect ratio
      var pinchLen = pinchEnd - pinchStart;
      var nModes = Math.max(1, Math.round(pinchLen / (2 * Math.PI * Math.max(pinchR, 0.001))));
      nModes = Math.min(nModes, 6);

      for (var pk = 0; pk <= N_PINCH; pk++) {
        var zFrac = pk / N_PINCH;
        // Tapered ends (pinch is thicker at center, tapers at edges)
        var taper = Math.sin(Math.PI * zFrac);
        var localR = pinchR * (0.5 + 0.5 * taper);
        var ripple = rippleAmp * localR * Math.cos(2 * Math.PI * nModes * zFrac);
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

      // Pinch column appears after significant compression (cr < 0.3)
      // or during pinch/post_pinch/reflected phases
      var pinchPhase = f.phase === "pinch" || f.phase === "post_pinch" || f.phase === "reflected";
      var pinchVisible = pinchPhase || (isP && cr < 0.3);
      pinch.isVisible = pinchVisible;
      halo.isVisible = pinchVisible;
      pinchMat.alpha = pinchVisible ? pI * 0.85 : 0;
      haloMat.alpha = pinchVisible ? pI * 0.25 : 0;

      // HDR pinch: scientifically accurate AND visually stunning.
      // Real DPF pinch IS blindingly bright (10^8 K). ACES tone mapping
      // handles the dynamic range — the pinch glows intensely white while
      // the device structure remains visible on the light background.
      if (pI > 0.7) {
        // Peak: HDR white-hot (>1.0 emissive = bloom via ACES tone mapping)
        var hdr = 1.0 + (pI - 0.7) * 5.0;  // 1.0 → 2.5
        pinchMat.emissiveColor.set(hdr, hdr * 0.9, hdr * 0.75);
        haloMat.emissiveColor.set(hdr * 0.4, hdr * 0.15, hdr * 0.05);
      } else if (pI > 0.3) {
        // Compression: warming from blue-white to orange-white
        pinchMat.emissiveColor.set(0.8 + pI * 0.4, 0.3 + pI * 0.6, pI * 0.4);
        haloMat.emissiveColor.set(0.5, 0.12, 0.04);
      } else {
        pinchMat.emissiveColor.set(pI * 1.5, pI * 0.4, pI * 0.15);
        haloMat.emissiveColor.set(pI * 0.5, pI * 0.1, 0.02);
      }

      // Phase-adaptive bloom: ramps during pinch, ACES handles clipping gracefully
      glowLayer.intensity = 0.5 + pI * 0.8;
      if (_pipeline) {
        _pipeline.bloomWeight = 0.25 + pI * 0.35;
        _pipeline.bloomThreshold = 0.5 - pI * 0.2;
        _pipeline.imageProcessing.exposure = 1.1 - pI * 0.15;
      }

      // Anode tip thermal glow — physically accurate (ohmic + plasma heating)
      if (pI > 0.5) {
        copperMat.emissiveColor.set(0.15 + pI * 0.3, 0.06 + pI * 0.08, 0.02);
      } else {
        copperMat.emissiveColor.set(0.05, 0.03, 0.01);
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

      // LEE MODEL SCHEMATIC: Particle system driven by Lee 0D scalars (not MHD velocity field)
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
          // Post-pinch: plasma dispersing but contained within device
          ps.gravity = new BABYLON.Vector3(0.5, 0, 0);
          ps.minEmitPower = 0.5; ps.maxEmitPower = 2;
          ps.emitRate = useGPU ? 2000 : 150;
          ps.minLifeTime = 0.05; ps.maxLifeTime = 0.15;
          psEmitter.radius = compR * 2;
        } else if (pI > 0.5) {
          // Peak pinch: intense axial jets (beam ions) — short-lived
          ps.gravity = new BABYLON.Vector3(4, 0, 0);
          ps.minEmitPower = 3; ps.maxEmitPower = 8;
          ps.minLifeTime = 0.05; ps.maxLifeTime = 0.12;
        } else {
          ps.gravity = new BABYLON.Vector3(0, 0, 0);
          ps.minEmitPower = 0.5; ps.maxEmitPower = 2;
        }
      }

      // LEE MODEL SCHEMATIC: B-field ring brightness from Lee 0D current I/I_peak (not MHD B_theta)
      var Ifrac = Math.abs(f.I) / Math.max(S.I_peak, 0.001);
      for (var fli = 0; fli < fieldLines.length; fli++) {
        if (!fieldLines[fli].isVisible) continue;
        var fld = fieldLineData[fli];
        if (!fld) continue;
        var bStr = 1 - fld.ri / N_RADII;

        if (fieldLines[fli].material) {
          fieldLines[fli].material.alpha = Math.min(0.85, (0.1 + bStr * 0.3) * Ifrac * 2);
          if (fieldLines[fli].material.emissiveColor) {
            var glow = Math.min(1, Ifrac * 1.5);
            fieldLines[fli].material.emissiveColor.set(
              0.1 + bStr * 0.3 * glow, 0.3 + bStr * 0.5 * glow, 0.7 + bStr * 0.3 * glow
            );
          }
        }

        if (isP) {
          var scaleFactor = cr + (1 - cr) * (fld.ri / N_RADII);
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

      // ---- Cinematic effects ----

      // DOF focus on pinch during compression
      if (isP && pI > 0.3) {
        pipeline.depthOfFieldEnabled = true;
        pipeline.depthOfField.focalLength = 60;
        pipeline.depthOfField.fStop = 2;
        pipeline.depthOfField.focusDistance =
          BABYLON.Vector3.Distance(cam.position, pinch.position) * 1000;
      } else {
        pipeline.depthOfFieldEnabled = false;
      }

      // Chromatic aberration during peak pinch (lens stress effect)
      if (pI > 0.5) {
        pipeline.chromaticAberrationEnabled = true;
        pipeline.chromaticAberration.aberrationAmount = pI * 30;
      } else {
        pipeline.chromaticAberrationEnabled = false;
      }

      // Camera auto-zoom: pull in during radial, snap back after
      if (isP && pI > 0.2 && autoOrbit) {
        var targetRadius = G.cathode_radius * (3 + (1 - pI) * 4);
        cam.radius += (targetRadius - cam.radius) * 0.02;
      } else if (autoOrbit && !userInteracting) {
        var defaultRadius = G.cathode_radius * 7;
        cam.radius += (defaultRadius - cam.radius) * 0.01;
      }

      // Exposure flash during pinch onset
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
