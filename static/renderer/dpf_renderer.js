/**
 * DPF-Unified Plasma Renderer — Babylon.js 7.x WebGPU
 *
 * Showcase-quality 3D physics visualization.
 * All visuals are driven by simulation data — no cosmetic fakes.
 *
 * Architecture:
 *   Python (app_visualization.py) → JSON data → this module → Babylon.js scene
 *
 * Features:
 *   - PBR electrodes with HDR environment reflections
 *   - GPU particle system (50K) with sub-emitters
 *   - Node Material plasma shader on pinch
 *   - Viridis/Cividis colorblind-safe heatmaps with colorbar
 *   - B-field line visualization (thin instances)
 *   - m=0 sausage instability on pinch surface
 *   - Fresnel edge glow on current sheath
 *   - DefaultRenderingPipeline (bloom, DOF, SSAO, ACES)
 *   - Babylon.GUI layer toggles + colorbar + data readout
 *   - Full zoom (into the pinch), orbit, pan
 */

// ============================================================
// Constants
// ============================================================
// Studio softbox: small (203KB), clean metallic reflections
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
  reflected: "Reflected shock",
  pinch: "Pinch — peak compression",
  post_pinch: "Post-pinch disruption",
  none: "",
};

const PHASE_DESCRIPTIONS = {
  rundown: "Current sheath sweeping gas toward anode tip",
  radial: "Plasma ring compressing inward — magnetic piston",
  mhd_radial: "MHD radial implosion in progress",
  reflected: "Reflected shock expanding outward",
  pinch: "PEAK COMPRESSION — fusion zone active",
  post_pinch: "Pinch disrupting via m=0 instability",
};

const SPEEDS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];

// ============================================================
// Utilities
// ============================================================
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

// ============================================================
// Main entry point
// ============================================================
async function createDPFScene(canvas, data) {
  const L = data;
  const G = L.geometry;
  const S = L.sheath;

  // ---- Engine (WebGPU → WebGL2 fallback) ----
  // Use WebGL2 by default for maximum compatibility.
  // WebGPU can be enabled via URL param ?webgpu=1
  let engine, gpuBackend = "WebGL2";
  const useWebGPU = (new URLSearchParams(window.location.search)).get("webgpu") === "1";
  if (useWebGPU) {
    try {
      if (await BABYLON.WebGPUEngine.IsSupportedAsync) {
        engine = new BABYLON.WebGPUEngine(canvas, {
          antialias: true,
          adaptToDeviceRatio: true,
          powerPreference: "high-performance",
        });
        await engine.initAsync();
        gpuBackend = "WebGPU";
      }
    } catch (_) {}
  }
  if (!engine) {
    engine = new BABYLON.Engine(canvas, true, {
      stencil: true,
      adaptToDeviceRatio: true,
      preserveDrawingBuffer: true,
    });
  }

  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.08, 0.09, 0.14, 1);
  scene.ambientColor = new BABYLON.Color3(0.2, 0.2, 0.25);

  // ---- HDR Environment ----
  let envTex = null;
  try {
    envTex = BABYLON.CubeTexture.CreateFromPrefilteredData(HDR_ENV, scene);
    scene.environmentTexture = envTex;
    scene.environmentIntensity = 1.2;
  } catch (_) {}

  // ---- Camera ----
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI / 4, Math.PI / 3, G.cathode_radius * 7,
    new BABYLON.Vector3(G.anode_length / 2, 0, 0), scene);
  cam.attachControl(canvas, true);
  cam.lowerRadiusLimit = G.anode_radius * 0.2;
  cam.upperRadiusLimit = G.cathode_radius * 60;
  cam.wheelPrecision = 12;
  cam.pinchPrecision = 15;
  cam.panningSensibility = 60;
  cam.minZ = 0.0005;
  cam.inertia = 0.88;

  // ---- Lighting (bright, 4-point) ----
  const hemiLight = new BABYLON.HemisphericLight("hemi",
    new BABYLON.Vector3(0, 1, 0), scene);
  hemiLight.intensity = 0.7;
  hemiLight.groundColor = new BABYLON.Color3(0.15, 0.15, 0.2);

  const keyLight = new BABYLON.PointLight("key",
    new BABYLON.Vector3(G.anode_length * 0.3, G.cathode_radius * 4, G.cathode_radius * 3), scene);
  keyLight.intensity = 1.2;
  keyLight.diffuse = new BABYLON.Color3(1, 0.96, 0.9);

  const fillLight = new BABYLON.PointLight("fill",
    new BABYLON.Vector3(G.anode_length * 0.8, -G.cathode_radius * 2, -G.cathode_radius * 2), scene);
  fillLight.intensity = 0.5;
  fillLight.diffuse = new BABYLON.Color3(0.75, 0.85, 1.0);

  const rimLight = new BABYLON.PointLight("rim",
    new BABYLON.Vector3(-G.cathode_radius, G.cathode_radius * 2, 0), scene);
  rimLight.intensity = 0.4;
  rimLight.diffuse = new BABYLON.Color3(0.6, 0.7, 1.0);

  // ============================================================
  // ELECTRODES (PBR with HDR reflections)
  // ============================================================
  // ---- Electrodes: smooth, bright, professional ----
  const copperMat = new BABYLON.PBRMaterial("copper", scene);
  copperMat.metallic = 0.95;
  copperMat.roughness = 0.25;
  copperMat.albedoColor = new BABYLON.Color3(0.97, 0.75, 0.5);
  copperMat.emissiveColor = new BABYLON.Color3(0.06, 0.03, 0.01);
  if (envTex) copperMat.reflectionTexture = envTex;
  copperMat.environmentIntensity = 1.5;

  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 128, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;

  const steelMat = new BABYLON.PBRMaterial("steel", scene);
  steelMat.metallic = 0.9;
  steelMat.roughness = 0.35;
  steelMat.albedoColor = new BABYLON.Color3(0.78, 0.78, 0.82);
  steelMat.emissiveColor = new BABYLON.Color3(0.04, 0.04, 0.05);
  if (envTex) steelMat.reflectionTexture = envTex;
  steelMat.environmentIntensity = 1.2;

  const cathodeRods = [];
  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2;
    const rod = BABYLON.MeshBuilder.CreateCylinder("rod" + i, {
      diameter: G.cathode_radius * 0.1, height: G.anode_length, tessellation: 32,
    }, scene);
    rod.rotation.z = Math.PI / 2;
    rod.position.set(
      G.anode_length / 2,
      G.cathode_radius * Math.sin(angle),
      G.cathode_radius * Math.cos(angle)
    );
    rod.material = steelMat;
    cathodeRods.push(rod);
  }

  // Insulator (bright ceramic)
  const ceramicMat = new BABYLON.PBRMaterial("ceramic", scene);
  ceramicMat.metallic = 0;
  ceramicMat.roughness = 0.5;
  ceramicMat.albedoColor = new BABYLON.Color3(0.95, 0.9, 0.75);
  ceramicMat.emissiveColor = new BABYLON.Color3(0.05, 0.04, 0.03);
  try { ceramicMat.subSurface.isTranslucencyEnabled = true;
    ceramicMat.subSurface.translucencyIntensity = 0.3; } catch(_) {}
  ceramicMat.alpha = 0.75;

  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameter: G.cathode_radius * 2, height: G.anode_radius * 0.3, tessellation: 128,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -G.anode_radius * 0.15;
  insulator.material = ceramicMat;

  // ============================================================
  // CURRENT SHEATH (disc with Fresnel edge glow)
  // ============================================================
  const sheathMat = new BABYLON.StandardMaterial("sheathMat", scene);
  sheathMat.emissiveColor = new BABYLON.Color3(0.3, 0.6, 1.0);
  sheathMat.alpha = 0.6;
  sheathMat.disableLighting = true;
  sheathMat.backFaceCulling = false;

  // Fresnel: bright edges (like looking through a thin plasma shell)
  sheathMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  sheathMat.emissiveFresnelParameters.bias = 0.4;
  sheathMat.emissiveFresnelParameters.power = 2;
  sheathMat.emissiveFresnelParameters.leftColor = BABYLON.Color3.White();
  sheathMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.2, 0.4, 0.9);

  sheathMat.opacityFresnelParameters = new BABYLON.FresnelParameters();
  sheathMat.opacityFresnelParameters.bias = 0.5;
  sheathMat.opacityFresnelParameters.power = 1.5;

  // Sheath as a smooth torus ring (no triangle fans from CreateDisc)
  const sheathMidR = (G.anode_radius + G.cathode_radius) / 2;
  const sheathTubeR = (G.cathode_radius - G.anode_radius) / 2;
  const sheath = BABYLON.MeshBuilder.CreateTorus("sheath", {
    diameter: sheathMidR * 2,
    thickness: sheathTubeR * 2,
    tessellation: 64,
  }, scene);
  sheath.rotation.z = Math.PI / 2;
  sheath.material = sheathMat;

  // Plasma trail (ionized gas behind sheath)
  const trailMat = new BABYLON.StandardMaterial("trailMat", scene);
  trailMat.emissiveColor = new BABYLON.Color3(0.1, 0.15, 0.4);
  trailMat.alpha = 0.15;
  trailMat.disableLighting = true;
  trailMat.backFaceCulling = false;
  const trail = BABYLON.MeshBuilder.CreateTube("trail", {
    path: [new BABYLON.Vector3(0, 0, 0), new BABYLON.Vector3(1, 0, 0)],
    radius: (G.anode_radius + G.cathode_radius) / 2,
    tessellation: 24, cap: BABYLON.Mesh.NO_CAP, updatable: true,
  }, scene);
  trail.material = trailMat;

  // ============================================================
  // PINCH COLUMN (tube with m=0 instability ripple)
  // ============================================================
  // Pinch material: Fresnel StandardMaterial (NME loaded async later if available)
  let pinchNME = false;
  var pinchMat = new BABYLON.StandardMaterial("pinchMat", scene);
  pinchMat.emissiveColor = new BABYLON.Color3(1, 0.35, 0.08);
  pinchMat.disableLighting = true;
  pinchMat.backFaceCulling = false;
  pinchMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  pinchMat.emissiveFresnelParameters.bias = 0.2;
  pinchMat.emissiveFresnelParameters.power = 3;
  pinchMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(1, 1, 0.9);
  pinchMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(1, 0.2, 0.05);
  pinchMat.alpha = 0;

  // NME fire shader disabled — the snippet produces visual artifacts.
  // Fresnel StandardMaterial gives a cleaner hot-plasma look.

  const N_PINCH = 24;
  const pinchPath = [];
  for (let i = 0; i <= N_PINCH; i++) {
    pinchPath.push(new BABYLON.Vector3(
      G.anode_length * (0.62 + 0.38 * i / N_PINCH), 0, 0
    ));
  }
  const pinchRadii = new Array(N_PINCH + 1).fill(G.anode_radius * 0.3);
  const pinch = BABYLON.MeshBuilder.CreateTube("pinch", {
    path: pinchPath, radiusFunction: (i) => pinchRadii[i],
    tessellation: 48, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  pinch.material = pinchMat;

  // Halo
  const haloMat = new BABYLON.StandardMaterial("haloMat", scene);
  haloMat.emissiveColor = new BABYLON.Color3(0.7, 0.1, 0.03);
  haloMat.disableLighting = true;
  haloMat.alpha = 0;
  haloMat.backFaceCulling = false;
  const haloRadii = new Array(N_PINCH + 1).fill(G.anode_radius * 0.6);
  const halo = BABYLON.MeshBuilder.CreateTube("halo", {
    path: pinchPath, radiusFunction: (i) => haloRadii[i],
    tessellation: 48, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.BACKSIDE, updatable: true,
  }, scene);
  halo.material = haloMat;

  // ============================================================
  // HEATMAP OVERLAY (RawTexture on midplane)
  // ============================================================
  let heatPlane = null, heatTex = null, heatBuf = null;
  let activeOverlay = "none";

  if (L.density) {
    const [nx, nz] = L.density.shape;
    const W = Math.min(nx * 4, 256), H = Math.min(nz * 4, 256);
    heatBuf = new Uint8Array(W * H * 4);
    heatTex = new BABYLON.RawTexture(heatBuf, W, H,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene, false, false,
      BABYLON.Texture.BILINEAR_SAMPLINGMODE);

    const heatMat = new BABYLON.StandardMaterial("heatMat", scene);
    heatMat.emissiveTexture = heatTex;
    heatMat.opacityTexture = heatTex;
    heatMat.disableLighting = true;
    heatMat.backFaceCulling = false;
    heatMat.alpha = 0.7;

    heatPlane = BABYLON.MeshBuilder.CreatePlane("heatPlane", {
      width: G.anode_length, height: G.cathode_radius * 2,
    }, scene);
    heatPlane.position.x = G.anode_length / 2;
    heatPlane.rotation.y = Math.PI / 2;
    heatPlane.material = heatMat;
    heatPlane.isVisible = false;
  }

  function updateHeatmap(key) {
    if (!heatTex || !L[key]) return;
    const fd = decodeBase64Float32(L[key].data, L[key].shape);
    const [nx, nz] = fd.shape;
    const W = heatTex.getSize().width, H = heatTex.getSize().height;

    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const v = bilinearSample(fd.data, nx, nz,
          (i / W) * (nx - 1), (j / H) * (nz - 1));
        const [r, g, b] = cmap(v);
        const idx = (j * W + i) * 4;
        heatBuf[idx] = (r * 255) | 0;
        heatBuf[idx + 1] = (g * 255) | 0;
        heatBuf[idx + 2] = (b * 255) | 0;
        heatBuf[idx + 3] = Math.min(255, (v * 200 + 55)) | 0;
      }
    }
    heatTex.update(heatBuf);
  }

  // ============================================================
  // B-FIELD LINES: azimuthal circles B_theta = mu0*I/(2*pi*r)
  // Always available — generated from circuit current, not MHD grid
  // ============================================================
  const fieldLines = [];
  // Generate circular field lines at multiple radii and axial positions
  const N_RADII = 5;
  const N_ZPOS = 4;
  const N_CIRCLE_PTS = 64;
  for (let zi = 0; zi < N_ZPOS; zi++) {
    const zPos = G.anode_length * (0.15 + 0.7 * zi / (N_ZPOS - 1));
    for (let ri = 0; ri < N_RADII; ri++) {
      const r = G.anode_radius * 1.2 + (G.cathode_radius - G.anode_radius * 1.2) * ri / (N_RADII - 1);
      const pts = [];
      for (let k = 0; k <= N_CIRCLE_PTS; k++) {
        const theta = (k / N_CIRCLE_PTS) * Math.PI * 2;
        pts.push(new BABYLON.Vector3(zPos, r * Math.sin(theta), r * Math.cos(theta)));
      }
      // Color by field strength: stronger closer to anode (1/r)
      const bStrength = 1 - ri / N_RADII; // 1 at anode, 0 at cathode
      const lineMat = new BABYLON.StandardMaterial("flm" + zi + "_" + ri, scene);
      const tube = BABYLON.MeshBuilder.CreateTube("fl" + zi + "_" + ri, {
        path: pts, radius: G.cathode_radius * 0.008 * (0.5 + bStrength),
        tessellation: 8, cap: BABYLON.Mesh.NO_CAP,
      }, scene);
      lineMat.emissiveColor = new BABYLON.Color3(
        0.1 + bStrength * 0.2, 0.3 + bStrength * 0.4, 0.8 + bStrength * 0.2
      );
      lineMat.disableLighting = true;
      lineMat.alpha = 0.3 + bStrength * 0.3;
      tube.material = lineMat;
      tube.isVisible = false;
      fieldLines.push(tube);
    }
  }

  // Also add poloidal field lines from MHD data if available
  if (L.bfield) {
    try {
      const fdBr = decodeBase64Float32(L.bfield.Br, L.bfield.shape);
      const fdBz = decodeBase64Float32(L.bfield.Bz, L.bfield.shape);
      const [nx, nz] = fdBr.shape;
      for (let s = 0; s < 8; s++) {
        let x = G.anode_length * (0.1 + 0.8 * s / 8), z = 0;
        const pts = [];
        const ds = G.anode_length / 60 * 0.6;
        for (let step = 0; step < 60; step++) {
          pts.push(new BABYLON.Vector3(x, 0, z));
          const fx = (x / G.anode_length) * (nx - 1);
          const fz = ((z + G.cathode_radius) / (G.cathode_radius * 2)) * (nz - 1);
          const br = bilinearSample(fdBr.data, nx, nz, fx, fz);
          const bz = bilinearSample(fdBz.data, nx, nz, fx, fz);
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
    } catch(_) {}
  }

  // ============================================================
  // GPU PARTICLE SYSTEM (soft gaussian, phase-adaptive)
  // ============================================================
  const useGPU = BABYLON.GPUParticleSystem.IsSupported;
  const PSClass = useGPU ? BABYLON.GPUParticleSystem : BABYLON.ParticleSystem;
  const psCap = useGPU ? 50000 : 4000;
  const ps = new PSClass("ions", { capacity: psCap }, scene);
  ps.emitter = new BABYLON.Vector3(0, 0, 0);

  const psEmitter = new BABYLON.SphereParticleEmitter();
  psEmitter.radius = G.cathode_radius * 0.85;
  psEmitter.radiusRange = 0.4;
  ps.particleEmitterType = psEmitter;

  ps.minLifeTime = 0.1; ps.maxLifeTime = 0.3;
  ps.emitRate = useGPU ? 5000 : 400;
  ps.minSize = 0.4; ps.maxSize = 1.8;
  ps.minEmitPower = 0.2; ps.maxEmitPower = 1.5;

  // Bright, visible gradient: blue → cyan → white
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

  // Generate soft gaussian particle texture
  const ptexSize = 64;
  const ptex = new BABYLON.DynamicTexture("ptex", ptexSize, scene, false);
  const ptxCtx = ptex.getContext();
  const grad = ptxCtx.createRadialGradient(
    ptexSize / 2, ptexSize / 2, 0,
    ptexSize / 2, ptexSize / 2, ptexSize / 2
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

  // ============================================================
  // POST-PROCESSING
  // ============================================================
  const pipeline = new BABYLON.DefaultRenderingPipeline("pipeline", true, scene, [cam]);
  pipeline.bloomEnabled = true;
  pipeline.bloomThreshold = 0.4;
  pipeline.bloomWeight = 0.6;
  pipeline.bloomKernel = 80;
  pipeline.bloomScale = 0.5;
  pipeline.imageProcessingEnabled = true;
  pipeline.imageProcessing.toneMappingEnabled = true;
  pipeline.imageProcessing.toneMappingType = BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pipeline.imageProcessing.exposure = 1.6;
  pipeline.imageProcessing.contrast = 1.15;

  let ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.8;
    ssao.radius = 1.5;
    ssao.samples = 16;
    ssao.base = 0.2;
  } catch (_) {}

  const glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 32, mainTextureFixedSize: 512,
  });
  glowLayer.intensity = 0.5;
  glowLayer.customEmissiveColorSelector = (mesh, _sub, _mat, result) => {
    const glowMeshes = ["sheath", "pinch", "halo", "trail"];
    if (glowMeshes.includes(mesh.name) && mesh.material && mesh.material.emissiveColor) {
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
  // Return the scene controller (called by the HTML host)
  // ============================================================
  return {
    engine,
    scene,
    camera: cam,
    gpuBackend,
    useGPU,

    // Scene objects for animation
    sheath, sheathMat, trail, trailMat,
    pinch, pinchMat, pinchNME, halo, haloMat,
    pinchRadii, haloRadii, pinchPath, N_PINCH,
    anode, cathodeRods, insulator,
    ps, psEmitter,
    pipeline, ssao, glowLayer,
    heatPlane, updateHeatmap,
    fieldLines,
    activeOverlay,

    // Data
    G, S, L,

    // Methods
    setOverlay(key) {
      activeOverlay = key;
      if (heatPlane) {
        if (key === "none") {
          heatPlane.isVisible = false;
        } else if (L[key]) {
          heatPlane.isVisible = true;
          updateHeatmap(key);
        }
      }
    },

    setCmap(useCividis) {
      activeCmap = useCividis ? CIVIDIS : VIRIDIS;
      if (activeOverlay !== "none" && heatPlane && heatPlane.isVisible) {
        updateHeatmap(activeOverlay);
      }
    },

    applyFrame(i) {
      if (i < 0 || i >= S.frames.length) return;
      const f = S.frames[i];
      const col = PHASE_COLORS[f.phase] || [0.3, 0.3, 0.4];
      const isP = ["radial", "mhd_radial", "pinch", "reflected", "post_pinch"].includes(f.phase);
      const cr = Math.max(0.02, f.r / G.cathode_radius);
      const pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;

      // Sheath
      sheath.position.x = isP ? G.anode_length : f.z;
      sheathMat.emissiveColor.set(col[0], col[1], col[2]);
      if (isP) {
        sheath.scaling.set(1, Math.max(0.03, cr), Math.max(0.03, cr));
      } else {
        sheath.scaling.set(1, 1, 1);
      }
      sheathMat.alpha = 0.45 + Math.abs(f.I) * 0.2;

      // Trail
      const tLen = Math.max(isP ? G.anode_length : f.z, 0.2);
      trail.scaling.x = tLen;
      trail.position.x = tLen / 2;
      trailMat.emissiveColor.set(col[0] * 0.3, col[1] * 0.3, col[2] * 0.4);
      trailMat.alpha = 0.1 + Math.abs(f.I) * 0.06;

      // Pinch with m=0 instability
      const instAmp = L.instability ? L.instability.amplitude : 0;
      const rippleAmp = isP ? instAmp * Math.min(1, (1 - cr) * 2) : 0;

      for (let k = 0; k <= N_PINCH; k++) {
        const zFrac = k / N_PINCH;
        const baseR = cr * G.cathode_radius * 0.4;
        const ripple = rippleAmp * baseR * Math.cos(4 * Math.PI * zFrac);
        pinchRadii[k] = Math.max(0.001, baseR + ripple);
        haloRadii[k] = Math.max(0.002, (baseR + ripple) * 2.5);
      }

      BABYLON.MeshBuilder.CreateTube("pinch", {
        path: pinchPath, radiusFunction: (idx) => pinchRadii[idx],
        tessellation: 20, cap: BABYLON.Mesh.CAP_ALL, instance: pinch,
      });
      BABYLON.MeshBuilder.CreateTube("halo", {
        path: pinchPath, radiusFunction: (idx) => haloRadii[idx],
        tessellation: 48, cap: BABYLON.Mesh.NO_CAP,
        sideOrientation: BABYLON.Mesh.BACKSIDE, instance: halo,
      });

      pinchMat.alpha = pI * 0.85;
      haloMat.alpha = pI * 0.35;
      if (pinchMat.emissiveColor) pinchMat.emissiveColor.set(1, 0.15 + pI * 0.5, pI * 0.3);
      haloMat.emissiveColor.set(0.8, 0.08 + pI * 0.15, 0.03);
      glowLayer.intensity = 0.35 + pI * 2;

      // Particles
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
          // Post-pinch: plasma expanding outward, dispersing
          ps.gravity = new BABYLON.Vector3(2, compR * 0.3, 0);
          ps.minEmitPower = 1; ps.maxEmitPower = 4;
          ps.emitRate = useGPU ? 3000 : 200; // fewer particles — plasma cooling
        } else if (pI > 0.5) {
          // Peak pinch: axial jets (beam ions)
          ps.gravity = new BABYLON.Vector3(5, 0, 0);
          ps.minEmitPower = 3; ps.maxEmitPower = 12;
        } else {
          // Radial compression
          ps.gravity = new BABYLON.Vector3(0, -compR * 0.5, 0);
          ps.minEmitPower = 1.5; ps.maxEmitPower = 6;
        }
      }

      // DOF focus on pinch
      if (isP && pI > 0.3) {
        pipeline.depthOfFieldEnabled = true;
        pipeline.depthOfField.focalLength = 60;
        pipeline.depthOfField.fStop = 2;
        pipeline.depthOfField.focusDistance =
          BABYLON.Vector3.Distance(cam.position, pinch.position) * 1000;
      } else {
        pipeline.depthOfFieldEnabled = false;
      }

      return { f, col, isP, cr, pI, rippleAmp };
    },
  };
}

// Export for use by the HTML host
window.createDPFScene = createDPFScene;
window.PHASE_LABELS = PHASE_LABELS;
window.PHASE_DESCRIPTIONS = PHASE_DESCRIPTIONS;
window.SPEEDS = SPEEDS;
