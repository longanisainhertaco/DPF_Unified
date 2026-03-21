/**
 * DPF Renderer v9 — Educational Dense Plasma Focus Visualization
 *
 * Design goal: a new student can identify every component and understand
 * every phase of the discharge without external help.
 *
 * Architecture: build*() create meshes, applyFrame() delegates to 4 updaters:
 *   updatePlasma()    — sheath, trail, pinch, halo, reflected shock, beam
 *   updateFields()    — B-field rings, current flow arrows
 *   updateParticles() — phase-matched particle emission
 *   updatePostFX()    — bloom, glow, camera, lighting, post-processing
 */

// ============================================================
// COLORMAPS (Viridis: perceptually uniform, Moreland 2009)
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
// Inferno colormap — perceptually uniform, designed for temperature data
const INFERNO = [
  [0.001,0.000,0.014],[0.122,0.032,0.267],[0.280,0.047,0.398],[0.434,0.063,0.406],
  [0.571,0.117,0.337],[0.692,0.194,0.261],[0.798,0.291,0.182],[0.882,0.414,0.100],
  [0.942,0.559,0.028],[0.976,0.722,0.052],[0.988,0.998,0.645]
];

let activeCmap = VIRIDIS;

// ============================================================
// PHASE DATA
// ============================================================

// Colors based on real deuterium plasma emission spectroscopy:
// D-alpha (656nm) = red, Fulcher-alpha band (600-620nm) = orange-red
// Hot plasma (1 eV / 11600K) = warm white-yellow (#FFEC92)
// Very hot (10+ eV) = blue-white (#8AADFF) — counterintuitive but physics-correct
const PHASE_COLORS = {
  rundown:    [1.0, 0.33, 0.0],    // D-alpha red-orange (#FF5500) — real emission
  radial:     [1.0, 0.55, 0.10],   // orange-red — compression heating
  mhd_radial: [1.0, 0.50, 0.12],  // orange-red
  reflected:  [1.0, 0.65, 0.20],   // warm orange — reflected shock
  pinch:      [1.0, 0.92, 0.57],   // warm white-yellow (#FFEC92) — 1 eV blackbody
  post_pinch: [0.85, 0.30, 0.05],  // cooling red-orange
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
// ENGINE INIT (WebGPU optional, WebGL2 default)
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
      stencil: true, adaptToDeviceRatio: true, preserveDrawingBuffer: false,  // false = 10% perf gain
    });
  }
  engine.setHardwareScalingLevel(1 / window.devicePixelRatio);
  return { engine, gpuBackend };
}

// ============================================================
// DEVICE -- wireframe, the fix that solved clipping
// ============================================================

function buildDevice(scene, G) {
  const copperMat = new BABYLON.StandardMaterial("copper", scene);
  copperMat.emissiveColor = new BABYLON.Color3(0.30, 0.20, 0.08);  // brighter device wireframe
  copperMat.diffuseColor = new BABYLON.Color3(0.1, 0.07, 0.03);
  copperMat.specularColor = new BABYLON.Color3(0.15, 0.1, 0.05);
  copperMat.alpha = 0.38;
  copperMat.wireframe = true;
  copperMat.backFaceCulling = false;
  copperMat.needDepthPrePass = true;
  // Holographic Fresnel edge glow on device
  copperMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  copperMat.emissiveFresnelParameters.bias = 0.05;
  copperMat.emissiveFresnelParameters.power = 3;
  copperMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(0.35, 0.25, 0.10);
  copperMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0, 0, 0);

  const steelMat = new BABYLON.StandardMaterial("steel", scene);
  steelMat.emissiveColor = new BABYLON.Color3(0.12, 0.13, 0.18);
  steelMat.diffuseColor = new BABYLON.Color3(0.06, 0.06, 0.08);
  steelMat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.12);
  steelMat.alpha = 0.55;
  steelMat.wireframe = true;
  steelMat.backFaceCulling = false;
  steelMat.needDepthPrePass = true;
  steelMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  steelMat.emissiveFresnelParameters.bias = 0.04;
  steelMat.emissiveFresnelParameters.power = 3;
  steelMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(0.2, 0.22, 0.3);
  steelMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0, 0, 0);

  const anode = BABYLON.MeshBuilder.CreateCylinder("anode", {
    diameter: G.anode_radius * 2, height: G.anode_length,
    tessellation: 48, cap: BABYLON.Mesh.CAP_ALL,  // 48 for smoother wireframe silhouette
  }, scene);
  anode.rotation.z = Math.PI / 2;
  anode.position.x = G.anode_length / 2;
  anode.material = copperMat;
  anode.renderingGroupId = 0;

  const N_RODS = G.n_cathode_rods || 8;
  const rodDiam = G.cathode_rod_diameter || G.cathode_radius * 0.04;
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
    rod.material = steelMat;
    rod.renderingGroupId = 0;
    cathodeRods.push(rod);
  }

  const ringThk = (G.cathode_radius - G.anode_radius) * 0.12;
  const ringMat = new BABYLON.StandardMaterial("ringMat", scene);
  ringMat.emissiveColor = new BABYLON.Color3(0.08, 0.09, 0.12);
  ringMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  ringMat.alpha = 0.35;
  ringMat.wireframe = true;
  ringMat.needDepthPrePass = true;

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
  insMat.emissiveColor = new BABYLON.Color3(0.35, 0.30, 0.20);
  insMat.diffuseColor = new BABYLON.Color3(0.15, 0.12, 0.08);
  insMat.alpha = 0.40;
  insMat.wireframe = true;

  const insulator = BABYLON.MeshBuilder.CreateCylinder("insulator", {
    diameterTop: insOuterR * 2, diameterBottom: insOuterR * 2,
    height: insThk, tessellation: 32,
  }, scene);
  insulator.rotation.z = Math.PI / 2;
  insulator.position.x = -insThk / 2;
  insulator.material = insMat;
  insulator.renderingGroupId = 0;

  // Vacuum chamber — transparent cylindrical housing (real DPFs operate inside one)
  var chamberMat = new BABYLON.StandardMaterial("chamberMat", scene);
  chamberMat.emissiveColor = new BABYLON.Color3(0.04, 0.05, 0.06);
  chamberMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  chamberMat.alpha = 0.14;  // slightly more visible for physical context
  chamberMat.wireframe = true;
  chamberMat.backFaceCulling = false;
  var chamber = BABYLON.MeshBuilder.CreateCylinder("chamber", {
    diameter: G.cathode_radius * 2.8,
    height: G.anode_length * 1.3,
    tessellation: 24, cap: BABYLON.Mesh.CAP_ALL,
  }, scene);
  chamber.rotation.z = Math.PI / 2;
  chamber.position.x = G.anode_length * 0.45;
  chamber.material = chamberMat;
  chamber.renderingGroupId = 0;

  return { anode, cathodeRods, insulator, chamber };
}

// ============================================================
// SHEATH TORUS -- Fresnel edge glow, phase color progression
// ============================================================

// Plasma noise shader — scrolling fbm for "alive" plasma look
var _plasmaShaderRegistered = false;
function _registerPlasmaShader() {
  if (_plasmaShaderRegistered) return;
  _plasmaShaderRegistered = true;
  BABYLON.Effect.ShadersStore["plasmaVertexShader"] = [
    "precision highp float;",
    "attribute vec3 position; attribute vec3 normal; attribute vec2 uv;",
    "uniform mat4 world; uniform mat4 worldViewProjection;",
    "varying vec2 vUV; varying vec3 vWP; varying vec3 vWN;",
    "void main(){gl_Position=worldViewProjection*vec4(position,1.);",
    "vUV=uv;vWP=(world*vec4(position,1.)).xyz;vWN=normalize((world*vec4(normal,0.)).xyz);}"
  ].join("\n");
  BABYLON.Effect.ShadersStore["plasmaFragmentShader"] = [
    "precision highp float;",
    "varying vec2 vUV; varying vec3 vWP; varying vec3 vWN;",
    "uniform float time; uniform float alpha; uniform vec3 camPos;",
    "uniform vec3 colA; uniform vec3 colB;",
    "vec3 P3(vec3 x){return mod(((x*34.)+1.)*x,289.);}",
    "float sn(vec2 v){",
    "  vec4 C=vec4(.211,.366,-.577,.024);vec2 i=floor(v+dot(v,C.yy));",
    "  vec2 x0=v-i+dot(i,C.xx);vec2 i1=(x0.x>x0.y)?vec2(1,0):vec2(0,1);",
    "  vec4 x12=x0.xyxy+C.xxzz;x12.xy-=i1;i=mod(i,289.);",
    "  vec3 p=P3(P3(i.y+vec3(0,i1.y,1))+i.x+vec3(0,i1.x,1));",
    "  vec3 m=max(.5-vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.);",
    "  m=m*m*m*m;vec3 x=2.*fract(p*C.www)-1.;vec3 h=abs(x)-.5;",
    "  vec3 a=x-floor(x+.5);m*=1.793-.854*(a*a+h*h);",
    "  vec3 g;g.x=a.x*x0.x+h.x*x0.y;g.yz=a.yz*x12.xz+h.yz*x12.yw;",
    "  return 130.*dot(m,g);}",
    "float fb(vec2 p){float v=0.,a=.5;mat2 r=mat2(.877,.479,-.479,.877);",
    "  for(int i=0;i<4;i++){v+=a*sn(p);p=r*p*2.+100.;a*=.5;}return v;}",
    "void main(){",
    "  vec2 uv=vUV*3.;",
    "  float n=fb(uv+vec2(.3,.1)*time)*.6+fb(uv+vec2(-.1,.4)*time*.7+3.2)*.4;",
    "  n=n*.5+.5;",
    "  vec3 viewDir=normalize(camPos-vWP);",
    "  float fres=pow(1.-abs(dot(vWN,viewDir)),2.);",
    "  vec3 col=mix(colA,colB,n);col=mix(col,vec3(1),pow(n,4.));",
    "  float g=pow(n,2.)*2.;col*=g;",
    "  float a=clamp((fres*.7+n*.3)*alpha,0.,1.);",
    "  gl_FragColor=vec4(col,a);}"
  ].join("\n");
}

function buildSheath(scene, G) {
  // Physically accurate: thin curved shell (Gratton-Vargas parabolic profile)
  // NOT a torus — the sheath is a ~2-5mm thick current sheet, center leads, edges trail
  const N_R = 12, N_THETA = 48;
  const rInner = G.anode_radius * 1.05;
  const rOuter = G.cathode_radius * 0.95;
  const sheathThickness = (G.cathode_radius - G.anode_radius) * 0.08; // ~4mm on PF-1000

  function buildSheathPaths(zCenter, compR) {
    var paths = [];
    var effOuter = Math.min(rOuter, compR || rOuter);
    // Gratton-Vargas parabolic: z_lag = 0.25 * z_tip
    var zLag = Math.max(1, zCenter * 0.25);
    for (var ir = 0; ir <= N_R; ir++) {
      var rFrac = ir / N_R;
      var r = rInner + (effOuter - rInner) * rFrac;
      var curvature = -zLag * rFrac * rFrac;
      var path = [];
      for (var it = 0; it <= N_THETA; it++) {
        var angle = (it / N_THETA) * Math.PI * 2;
        path.push(new BABYLON.Vector3(
          zCenter + curvature,
          r * Math.sin(angle),
          r * Math.cos(angle)
        ));
      }
      paths.push(path);
    }
    // Add rear surface offset by sheathThickness for physical depth
    // The sheath has 4 zones: undisturbed gas | shock front | current layer | magnetic piston
    for (var ir2 = N_R; ir2 >= 0; ir2--) {
      var rFrac2 = ir2 / N_R;
      var r2 = rInner + (effOuter - rInner) * rFrac2;
      var curv2 = -zLag * rFrac2 * rFrac2;
      var path2 = [];
      for (var it2 = 0; it2 <= N_THETA; it2++) {
        var angle2 = (it2 / N_THETA) * Math.PI * 2;
        path2.push(new BABYLON.Vector3(
          zCenter + curv2 - sheathThickness,  // offset behind the front surface
          r2 * Math.sin(angle2),
          r2 * Math.cos(angle2)
        ));
      }
      paths.push(path2);
    }
    return paths;
  }

  var initPaths = buildSheathPaths(0, rOuter);
  var torus = BABYLON.MeshBuilder.CreateRibbon("sheathDisk", {
    pathArray: initPaths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  torus.renderingGroupId = 1;
  torus.alphaIndex = 10;
  var midR = (G.anode_radius + G.cathode_radius) / 2;
  var tubeR = (G.cathode_radius - G.anode_radius) / 2;

  // Try custom plasma shader, fall back to StandardMaterial
  var mat, fresnel = null, isShaderMat = false;
  try {
    _registerPlasmaShader();
    mat = new BABYLON.ShaderMaterial("sheathMat", scene,
      { vertex: "plasma", fragment: "plasma" },
      { attributes: ["position", "normal", "uv"],
        uniforms: ["world", "worldViewProjection", "time", "alpha", "camPos", "colA", "colB"],
        needAlphaBlending: true });
    mat.setFloat("time", 0); mat.setFloat("alpha", 0);
    mat.setVector3("colA", new BABYLON.Vector3(0.02, 0.05, 0.30));
    mat.setVector3("colB", new BABYLON.Vector3(0.10, 0.60, 1.00));
    mat.backFaceCulling = false;
    mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
    isShaderMat = true;
  } catch (_) {
    // Fallback: StandardMaterial with Fresnel
    mat = new BABYLON.StandardMaterial("sheathMat", scene);
    mat.emissiveColor = new BABYLON.Color3(0.15, 0.45, 1.0);
    mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
    mat.specularColor = new BABYLON.Color3(0, 0, 0);
    mat.disableLighting = true;
    mat.alpha = 0;
    mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
    mat.backFaceCulling = false;
    fresnel = new BABYLON.FresnelParameters();
    fresnel.bias = 0.25; fresnel.power = 1.8;
    fresnel.leftColor = new BABYLON.Color3(1, 1, 1);
    fresnel.rightColor = new BABYLON.Color3(0, 0, 0);
    mat.emissiveFresnelParameters = fresnel;
  }
  torus.material = mat;

  // Motion afterimage: second torus that trails the primary for temporal coherence
  var ghostMat = new BABYLON.StandardMaterial("sheathGhost", scene);
  ghostMat.emissiveColor = new BABYLON.Color3(1.0, 0.33, 0.0);
  ghostMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  ghostMat.disableLighting = true;
  ghostMat.alpha = 0;
  ghostMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  ghostMat.backFaceCulling = false;
  var ghost = torus.clone("sheathGhost");
  ghost.material = ghostMat;
  ghost.renderingGroupId = 1;
  ghost.alphaIndex = 8;
  var prevZ = 0;

  return { torus, mat, midR, tubeR, fresnel, isShaderMat, ghost, ghostMat, buildSheathPaths, getPrevZ: function() { return prevZ; }, setPrevZ: function(z) { prevZ = z; } };
}

// ============================================================
// PLASMA TRAIL -- dim tube behind sheath (inside anode, no clipping)
// ============================================================

function buildTrail(scene, G) {
  const trail = BABYLON.MeshBuilder.CreateCylinder("plasmaTrail", {
    diameter: G.anode_radius * 1.2, height: G.anode_length,
    tessellation: 24, cap: BABYLON.Mesh.NO_CAP,
  }, scene);
  trail.rotation.z = Math.PI / 2;
  trail.position.x = G.anode_length / 2;
  const mat = new BABYLON.StandardMaterial("trailMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.06, 0.12, 0.35);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  // Opacity Fresnel: more visible at grazing angles, transparent head-on
  mat.opacityFresnelParameters = new BABYLON.FresnelParameters();
  mat.opacityFresnelParameters.bias = 0.0;
  mat.opacityFresnelParameters.power = 2;
  mat.opacityFresnelParameters.leftColor = new BABYLON.Color3(1, 1, 1);
  mat.opacityFresnelParameters.rightColor = new BABYLON.Color3(0.2, 0.2, 0.2);
  trail.material = mat;
  trail.renderingGroupId = 1;
  trail.alphaIndex = 5;
  return { trail, mat };
}

// ============================================================
// PINCH COLUMN -- dual tube: core (35% radius, white-hot) + mantle
// Bennett radial profile, m=0 sausage instability ripple
// ============================================================

function buildPinch(scene, G) {
  const N = 24;
  // Lee & Serban (1996): pinch column length z_f ~ 0.8 * anode_radius
  // PF-1000: z_f ~ 92mm, r_min ~ 14mm, aspect ratio ~3:1
  const columnLen = G.anode_radius * 0.8;
  const tipX = G.anode_length;
  const path = [];
  for (let k = 0; k <= N; k++) {
    path.push(new BABYLON.Vector3(
      tipX - columnLen * 0.1 + columnLen * 1.1 * k / N, 0, 0));
  }
  // Lee & Serban scaling: r_min ~ 0.12 * anode_radius (deuterium)
  const radii = new Array(N + 1).fill(G.anode_radius * 0.12);

  const coreMat = new BABYLON.StandardMaterial("coreMat", scene);
  coreMat.emissiveColor = new BABYLON.Color3(2.0, 1.9, 1.7);
  coreMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  coreMat.disableLighting = true;
  coreMat.alpha = 0;
  coreMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  coreMat.backFaceCulling = false;
  // Fresnel on pinch core: white-hot edges
  coreMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  coreMat.emissiveFresnelParameters.bias = 0.3;
  coreMat.emissiveFresnelParameters.power = 1.5;
  coreMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(2.5, 2.3, 2.0);
  coreMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(1.0, 0.8, 0.4);

  const core = BABYLON.MeshBuilder.CreateTube("pinchCore", {
    path, radiusFunction: function(i) { return radii[i] * 0.35; },
    tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, updatable: true,
  }, scene);
  core.material = coreMat;
  core.renderingGroupId = 1;
  core.alphaIndex = 30;

  const mantleMat = new BABYLON.StandardMaterial("mantleMat", scene);
  mantleMat.emissiveColor = new BABYLON.Color3(1.4, 0.55, 0.08);
  mantleMat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mantleMat.disableLighting = true;
  mantleMat.alpha = 0;
  mantleMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mantleMat.backFaceCulling = false;
  // Fresnel on pinch mantle: orange-red energy edges
  mantleMat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  mantleMat.emissiveFresnelParameters.bias = 0.15;
  mantleMat.emissiveFresnelParameters.power = 2.0;
  mantleMat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(1.5, 0.6, 0.1);
  mantleMat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.3, 0.1, 0.02);

  const mantle = BABYLON.MeshBuilder.CreateTube("pinchMantle", {
    path, radiusFunction: function(i) { return radii[i]; },
    tessellation: 16, cap: BABYLON.Mesh.NO_CAP,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true,
  }, scene);
  mantle.material = mantleMat;
  mantle.renderingGroupId = 1;
  mantle.alphaIndex = 25;

  return { core, mantle, coreMat, mantleMat, radii, path, N, columnLen };
}

// ============================================================
// HALO -- tube at 2.5x pinch radius, backside rendered
// ============================================================

function buildHalo(scene, G) {
  const haloR = G.anode_radius * 0.45;
  const path = [
    new BABYLON.Vector3(G.anode_length * 0.88, 0, 0),
    new BABYLON.Vector3(G.anode_length * 1.12, 0, 0),
  ];
  const halo = BABYLON.MeshBuilder.CreateTube("halo", {
    path, radius: haloR, tessellation: 24,
    sideOrientation: BABYLON.Mesh.BACKSIDE,
  }, scene);
  const mat = new BABYLON.StandardMaterial("haloMat", scene);
  mat.emissiveColor = new BABYLON.Color3(1.2, 0.5, 0.12);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;
  halo.material = mat;
  halo.renderingGroupId = 1;
  halo.alphaIndex = 20;
  return { halo, mat };
}

// ============================================================
// REFLECTED SHOCK + BEAM CONE
// ============================================================

function buildReflectedShock(scene, G) {
  var refRing = BABYLON.MeshBuilder.CreateTorus("reflShock", {
    diameter: G.anode_radius * 0.1, thickness: G.cathode_radius * 0.02,
    tessellation: 32,
  }, scene);
  refRing.rotation.z = Math.PI / 2;
  refRing.position.x = G.anode_length;
  var refMat = new BABYLON.StandardMaterial("reflMat", scene);
  refMat.emissiveColor = new BABYLON.Color3(1.0, 0.7, 0.3);
  refMat.disableLighting = true;
  refMat.alpha = 0;
  refMat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  refMat.backFaceCulling = false;
  refRing.material = refMat;
  refRing.renderingGroupId = 1;
  refRing.isVisible = false;
  return { refRing, refMat };
}

function buildBeam(scene, G) {
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
  cone.alphaIndex = 35;
  return { cone, mat };
}

// ============================================================
// B-FIELD RINGS -- 5 tori, NASA convention blue, behind sheath only
// ============================================================

function buildBField(scene, G) {
  const bRings = [];
  const zFracs = [0.25, 0.45, 0.65, 0.82, 0.95];
  const mat = new BABYLON.StandardMaterial("bFieldMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.2, 0.5, 1.0);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.alphaMode = BABYLON.Engine.ALPHA_ADD;
  mat.backFaceCulling = false;

  for (let i = 0; i < zFracs.length; i++) {
    const ring = BABYLON.MeshBuilder.CreateTorus("bRing" + i, {
      diameter: (G.anode_radius + G.cathode_radius),
      thickness: G.cathode_radius * 0.012,
      tessellation: 32,
    }, scene);
    ring.rotation.z = Math.PI / 2;
    ring.position.x = G.anode_length * zFracs[i];
    ring.material = mat;
    ring.renderingGroupId = 1;
    ring.isVisible = false;
    ring.alphaIndex = 8;
    bRings.push(ring);
  }
  return { bRings, mat, zFracs };
}

// ============================================================
// CURRENT FLOW ARROWS -- axial (anode), radial (sheath), return (cathode)
// ============================================================

function buildCurrentArrows(scene, G) {
  const mat = new BABYLON.StandardMaterial("currentArrowMat", scene);
  mat.emissiveColor = new BABYLON.Color3(0.4, 1.2, 2.0);  // HDR cyan — catches bloom
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.disableLighting = true;
  mat.alpha = 0;
  mat.backFaceCulling = false;
  // Fresnel glow on arrow edges
  mat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  mat.emissiveFresnelParameters.bias = 0.3;
  mat.emissiveFresnelParameters.power = 2;
  mat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(0.5, 1.5, 2.5);
  mat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0.1, 0.3, 0.5);

  const arrowH = G.anode_radius * 0.45;   // 3x taller
  const arrowD = G.anode_radius * 0.22;   // 3x wider
  const axialArrows = [];
  const radialArrows = [];
  const returnArrows = [];

  // Axial arrows: up the anode in +x direction (3 arrows at 25%, 50%, 75%)
  for (let i = 0; i < 3; i++) {
    const cone = BABYLON.MeshBuilder.CreateCylinder("axialArrow" + i, {
      diameterTop: 0, diameterBottom: arrowD, height: arrowH, tessellation: 8,
    }, scene);
    cone.rotation.z = -Math.PI / 2;
    cone.position.set(G.anode_length * (0.25 + i * 0.25), G.anode_radius * 1.3, 0);
    cone.material = mat;
    cone.renderingGroupId = 1;
    cone.isVisible = false;
    axialArrows.push(cone);
  }

  // Radial arrows: across the plasma at sheath, 4 at 90-degree intervals
  for (let i = 0; i < 4; i++) {
    const angle = i * Math.PI / 2;
    const cone = BABYLON.MeshBuilder.CreateCylinder("radialArrow" + i, {
      diameterTop: 0, diameterBottom: arrowD, height: arrowH, tessellation: 8,
    }, scene);
    const midR = (G.anode_radius + G.cathode_radius) / 2;
    cone.position.set(G.anode_length, midR * Math.sin(angle), midR * Math.cos(angle));
    cone.lookAt(new BABYLON.Vector3(G.anode_length,
      G.cathode_radius * 1.5 * Math.sin(angle),
      G.cathode_radius * 1.5 * Math.cos(angle)));
    cone.material = mat;
    cone.renderingGroupId = 1;
    cone.isVisible = false;
    radialArrows.push(cone);
  }

  // Return arrows: down the cathode rod in -x direction (3 arrows)
  for (let i = 0; i < 3; i++) {
    const cone = BABYLON.MeshBuilder.CreateCylinder("returnArrow" + i, {
      diameterTop: 0, diameterBottom: arrowD, height: arrowH, tessellation: 8,
    }, scene);
    cone.rotation.z = Math.PI / 2;
    cone.position.set(G.anode_length * (0.75 - i * 0.25), G.cathode_radius, 0);
    cone.material = mat;
    cone.renderingGroupId = 1;
    cone.isVisible = false;
    returnArrows.push(cone);
  }

  return { axialArrows, radialArrows, returnArrows, mat };
}

// ============================================================
// PARTICLES -- 3000 cap, BLENDMODE_ONEONE, soft radial gradient
// ============================================================

function buildParticles(scene, G) {
  const emitter = new BABYLON.AbstractMesh("psEmitter", scene);
  emitter.position.x = G.anode_length;

  // Soft radial gradient texture via DynamicTexture
  const texSize = 128;
  const dynTex = new BABYLON.DynamicTexture("pTex", texSize, scene, false);
  const ctx = dynTex.getContext();
  const half = texSize / 2;
  const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
  grad.addColorStop(0, "rgba(255,255,255,1.0)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.6)");
  grad.addColorStop(0.7, "rgba(255,255,255,0.15)");
  grad.addColorStop(1.0, "rgba(255,255,255,0.0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, texSize, texSize);
  dynTex.update();

  const ps = new BABYLON.ParticleSystem("sparks", 3000, scene);
  ps.particleTexture = dynTex;
  ps.emitter = emitter;
  ps.createSphereEmitter(G.anode_radius * 0.3);
  // Color gradient over lifetime: bright core → colored → fade
  ps.addColorGradient(0.0, new BABYLON.Color4(1.0, 1.0, 1.0, 0.9));
  ps.addColorGradient(0.15, new BABYLON.Color4(0.5, 0.8, 1.0, 0.7));
  ps.addColorGradient(0.5, new BABYLON.Color4(0.2, 0.5, 0.9, 0.4));
  ps.addColorGradient(0.8, new BABYLON.Color4(0.1, 0.25, 0.6, 0.15));
  ps.addColorGradient(1.0, new BABYLON.Color4(0.05, 0.1, 0.3, 0.0));
  // Size gradient: grow then shrink (spark trail effect)
  ps.addSizeGradient(0.0, G.cathode_radius * 0.01);
  ps.addSizeGradient(0.15, G.cathode_radius * 0.06);
  ps.addSizeGradient(0.5, G.cathode_radius * 0.04);
  ps.addSizeGradient(1.0, G.cathode_radius * 0.005);
  ps.minLifeTime = 0.3;
  ps.maxLifeTime = 1.2;
  ps.emitRate = 0;
  ps.gravity = new BABYLON.Vector3(0, 0, 0);
  ps.minEmitPower = G.cathode_radius * 0.2;
  ps.maxEmitPower = G.cathode_radius * 1.0;
  ps.blendMode = BABYLON.ParticleSystem.BLENDMODE_ONEONE;
  ps.start();
  return { ps, emitter };
}

// ============================================================
// AMBIENT DUST -- faint floating particles for museum depth
// ============================================================

function buildAmbientDust(scene, G) {
  var dust = new BABYLON.ParticleSystem("dust", 200, scene);
  dust.emitter = new BABYLON.Vector3(G.anode_length / 2, 0, 0);
  dust.createBoxEmitter(
    new BABYLON.Vector3(-0.02, -0.02, -0.02),
    new BABYLON.Vector3(0.02, 0.02, 0.02),
    new BABYLON.Vector3(-G.cathode_radius * 3, -G.cathode_radius * 2, -G.cathode_radius * 3),
    new BABYLON.Vector3(G.cathode_radius * 3, G.cathode_radius * 2, G.cathode_radius * 3)
  );
  dust.addColorGradient(0.0, new BABYLON.Color4(0.3, 0.35, 0.5, 0.0));
  dust.addColorGradient(0.3, new BABYLON.Color4(0.3, 0.35, 0.5, 0.06));
  dust.addColorGradient(0.7, new BABYLON.Color4(0.25, 0.3, 0.45, 0.04));
  dust.addColorGradient(1.0, new BABYLON.Color4(0.2, 0.25, 0.4, 0.0));
  dust.minSize = G.cathode_radius * 0.005;
  dust.maxSize = G.cathode_radius * 0.015;
  dust.minLifeTime = 4;
  dust.maxLifeTime = 10;
  dust.emitRate = 15;
  dust.gravity = new BABYLON.Vector3(0, G.cathode_radius * 0.002, 0);
  dust.minEmitPower = 0;
  dust.maxEmitPower = G.cathode_radius * 0.01;
  dust.blendMode = BABYLON.ParticleSystem.BLENDMODE_ADD;
  dust.start();
  return dust;
}

// ============================================================
// GROUND GRID -- spatial reference
// ============================================================

function buildGrid(scene, G) {
  const sz = Math.max(G.anode_length * 3, G.cathode_radius * 6);
  const ground = BABYLON.MeshBuilder.CreateGround("grid", { width: sz, height: sz }, scene);
  ground.position.y = -G.cathode_radius * 1.3;
  ground.position.x = G.anode_length / 2;
  const tex = new BABYLON.DynamicTexture("gridTex", 1024, scene, false);
  const ctx = tex.getContext();
  ctx.fillStyle = "rgba(22,24,30,1)";
  ctx.fillRect(0, 0, 1024, 1024);
  // Fine grid (20 divisions)
  ctx.strokeStyle = "rgba(45,50,65,0.5)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 20; i++) {
    const p = i * 1024 / 20;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 1024); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(1024, p); ctx.stroke();
  }
  // Major grid (4 divisions) — bolder lines
  ctx.strokeStyle = "rgba(55,60,80,0.8)";
  ctx.lineWidth = 2;
  for (let i = 0; i <= 4; i++) {
    const p = i * 1024 / 4;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 1024); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(1024, p); ctx.stroke();
  }
  tex.update();
  const mat = new BABYLON.StandardMaterial("gridMat", scene);
  mat.diffuseTexture = tex;
  mat.specularColor = new BABYLON.Color3(0, 0, 0);
  mat.emissiveColor = new BABYLON.Color3(0.06, 0.07, 0.09);
  mat.alpha = 0.75;
  ground.material = mat;
  ground.renderingGroupId = 0;
  // Center cross
  ctx.strokeStyle = "rgba(70,75,95,0.9)";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(512, 0); ctx.lineTo(512, 1024); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, 512); ctx.lineTo(1024, 512); ctx.stroke();
  tex.update();
}

// ============================================================
// HEATMAP -- DUAL: r-z cross-section (full radial data) + cylindrical wrap (overview)
// The cross-section shows the REAL 2D radial structure
// The cylinder shows the radially-averaged z-profile for context
// ============================================================

function buildHeatmap(scene, G) {
  // PRIMARY: r-z cross-section ribbon showing full 2D data (at two azimuthal angles for visibility)
  var nr = 16, nz = 32;
  var xsecPaths = [];
  for (var ir = 0; ir <= nr; ir++) {
    var r = G.anode_radius + (G.cathode_radius - G.anode_radius) * ir / nr;
    var row = [];
    for (var iz = 0; iz <= nz; iz++) {
      var z = G.anode_length * iz / nz;
      // Cross-section at angle = 60 degrees (visible from most camera angles)
      var angle = Math.PI * 0.33;
      row.push(new BABYLON.Vector3(z, r * Math.sin(angle), r * Math.cos(angle)));
    }
    xsecPaths.push(row);
  }
  var xsec = BABYLON.MeshBuilder.CreateRibbon("heatXsec", {
    pathArray: xsecPaths, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: false,
  }, scene);
  xsec.isVisible = false;
  xsec.isPickable = false;
  var xsecMat = new BABYLON.StandardMaterial("heatXsecMat", scene);
  xsecMat.disableLighting = true;
  xsecMat.backFaceCulling = false;
  xsecMat.alpha = 0.70;
  xsecMat.emissiveColor = new BABYLON.Color3(0.2, 0.2, 0.2);
  xsec.material = xsecMat;
  xsec.renderingGroupId = 2;

  // SECONDARY: cylindrical wrap (radially averaged, for context from any angle)
  const midR = (G.anode_radius + G.cathode_radius) / 2;
  const nArc = 48;
  const paths = [];
  for (let iz = 0; iz <= nz; iz++) {
    const z = G.anode_length * iz / nz;
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
  mat.alpha = 0.60;
  // Fresnel edge glow for cinematic curvature on heatmap cylinder
  mat.emissiveFresnelParameters = new BABYLON.FresnelParameters();
  mat.emissiveFresnelParameters.bias = 0.02;
  mat.emissiveFresnelParameters.power = 2.5;
  mat.emissiveFresnelParameters.leftColor = new BABYLON.Color3(0.15, 0.15, 0.15);
  mat.emissiveFresnelParameters.rightColor = new BABYLON.Color3(0, 0, 0);
  mat.emissiveColor = new BABYLON.Color3(0.15, 0.15, 0.15);
  cyl.material = mat;
  cyl.renderingGroupId = 2;
  return { cyl, mat, xsec, xsecMat };
}

// ============================================================
// SNAP CACHE — decode base64 Float32, field-specific colormaps
// ============================================================

var FIELD_CMAPS = { density: VIRIDIS, temperature: INFERNO, bfield: VIRIDIS };

function buildSnapCache(fieldKey, layer, cache) {
  var cmap = FIELD_CMAPS[fieldKey] || activeCmap;
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
        const c = cmapLookup(v, cmap || activeCmap);
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
// POST-PROCESSING — GlowLayer, bloom, SSAO, FXAA, heat distortion, halation
// ============================================================

// Heat distortion — mirage effect near pinch zone
function buildHeatDistortion(scene, cam) {
  try {
    BABYLON.Effect.ShadersStore["heatDistortionFragmentShader"] = [
      "precision highp float;",
      "varying vec2 vUV;",
      "uniform sampler2D textureSampler;",
      "uniform float time;",
      "uniform float intensity;",
      "uniform vec2 center;",
      "uniform float radius;",
      "vec2 h2(vec2 p){p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));return -1.+2.*fract(sin(p)*43758.5453);}",
      "float ns(vec2 p){vec2 i=floor(p),f=fract(p),u=f*f*(3.-2.*f);",
      "  return mix(mix(dot(h2(i),f),dot(h2(i+vec2(1,0)),f-vec2(1,0)),u.x),",
      "  mix(dot(h2(i+vec2(0,1)),f-vec2(0,1)),dot(h2(i+vec2(1,1)),f-vec2(1,1)),u.x),u.y);}",
      "void main(){",
      "  float dist=length(vUV-center);",
      "  float mask=pow(1.-smoothstep(0.,radius,dist),1.5);",
      "  vec2 nc=vUV*6.+vec2(time*1.2,time*0.84);",
      "  vec2 offset=vec2(ns(nc),ns(nc+vec2(5.2,1.3)))*0.015*intensity*mask;",
      "  gl_FragColor=texture2D(textureSampler,vUV+offset);",
      "}"
    ].join("\n");
    var pp = new BABYLON.PostProcess("heatDistort", "heatDistortion",
      ["time", "intensity", "center", "radius"], null, 1.0, cam);
    pp.onApply = function(effect) {
      effect.setFloat("time", pp._heatTime || 0);
      effect.setFloat("intensity", pp._heatIntensity || 0);
      effect.setFloat2("center", 0.55, 0.45);
      effect.setFloat("radius", 0.3);
    };
    pp._heatTime = 0;
    pp._heatIntensity = 0;
    return pp;
  } catch (_) { return null; }
}

// Halation post-process — warm red-orange halo on highlights (film artifact)
// "Transforms energy perception from game to film instantly" — research finding
function buildHalation(scene, cam) {
  try {
    BABYLON.Effect.ShadersStore["halationFragmentShader"] = [
      "precision highp float;",
      "varying vec2 vUV;",
      "uniform sampler2D textureSampler;",
      "uniform float intensity;",
      "void main(){",
      "  vec4 c=texture2D(textureSampler,vUV);",
      "  float lum=dot(c.rgb,vec3(0.299,0.587,0.114));",
      "  float mask=smoothstep(0.6,1.0,lum);",
      "  vec2 px=vec2(1./1024.);",  // approximate pixel size
      "  vec3 blur=vec3(0.);",
      "  for(float i=-3.;i<=3.;i++){for(float j=-3.;j<=3.;j++){",
      "    blur+=texture2D(textureSampler,vUV+vec2(i,j)*px*3.).rgb;",
      "  }}",
      "  blur/=49.;",
      "  vec3 halation=blur*vec3(1.0,0.4,0.15)*mask*intensity;",
      "  gl_FragColor=vec4(c.rgb+halation,c.a);",
      "}"
    ].join("\n");
    var pp = new BABYLON.PostProcess("halation", "halation",
      ["intensity"], null, 0.5, cam);
    pp.onApply = function(effect) {
      effect.setFloat("intensity", pp._halIntensity || 0);
    };
    pp._halIntensity = 0;
    return pp;
  } catch (_) { return null; }
}

function buildPipeline(scene, cam) {
  const pipe = new BABYLON.DefaultRenderingPipeline("dpf", true, scene, [cam]);
  pipe.bloomEnabled = true;
  pipe.bloomWeight = 0.20;
  pipe.bloomThreshold = 0.80;
  pipe.bloomKernel = 128;
  pipe.bloomScale = 0.5;
  pipe.fxaaEnabled = true;
  pipe.sharpenEnabled = true;
  pipe.sharpen.edgeAmount = 0.12;  // Subtle — crisper wireframe edges
  pipe.imageProcessingEnabled = true;
  pipe.imageProcessing.toneMappingEnabled = true;
  pipe.imageProcessing.toneMappingType = BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pipe.imageProcessing.exposure = 1.05;
  pipe.imageProcessing.contrast = 1.05;
  // Film grain — breaks banding on smooth plasma gradients
  pipe.grainEnabled = true;
  pipe.grain.intensity = 8;  // Subtle — enough to break banding, not distracting
  pipe.grain.animated = true;
  // Slight warm color grade for cinematic feel
  pipe.imageProcessing.colorCurvesEnabled = true;
  var cc = new BABYLON.ColorCurves();
  cc.globalHue = 15;         // Slight warm shift
  cc.globalSaturation = 5;   // Subtle saturation boost
  cc.highlightsHue = 30;     // Warm highlights
  cc.shadowsHue = 220;       // Cool shadows (complementary)
  cc.shadowsSaturation = 10;
  pipe.imageProcessing.colorCurves = cc;
  // Vignette — draws eye toward center/pinch
  pipe.imageProcessing.vignetteEnabled = true;
  pipe.imageProcessing.vignetteWeight = 1.5;
  pipe.imageProcessing.vignetteBlendMode = BABYLON.ImageProcessingConfiguration.VIGNETTEMODE_MULTIPLY;
  pipe.imageProcessing.vignetteColor = new BABYLON.Color4(0, 0, 0, 0);
  // Chromatic aberration — subtle, "hot lens" on plasma
  pipe.chromaticAberrationEnabled = true;
  pipe.chromaticAberration.aberrationAmount = 15;
  pipe.chromaticAberration.radialIntensity = 0.8;

  let ssao = null;
  try {
    ssao = new BABYLON.SSAO2RenderingPipeline("ssao", scene,
      { ssaoRatio: 0.5, blurRatio: 1 }, [cam], false);
    ssao.totalStrength = 0.50;
    ssao.radius = 2.0;
    ssao.samples = 16;  // 16 is sufficient for wireframe device, saves GPU
    ssao.base = 0.08;
  } catch (_) {}

  const glowLayer = new BABYLON.GlowLayer("glow", scene, {
    blurKernelSize: 128, mainTextureFixedSize: 1024,
    mainTextureSamples: 4,
  });
  glowLayer.intensity = 0.50;

  const plasmaNames = new Set([
    "sheathDisk", "pinchCore", "pinchMantle", "beamCone", "plasmaTrail", "halo",
  ]);
  glowLayer.customEmissiveColorSelector = function(mesh, _s, _m, result) {
    if (plasmaNames.has(mesh.name) && mesh.material && mesh.material.emissiveColor) {
      var ec = mesh.material.emissiveColor;
      var a = mesh.material.alpha || 0;
      // Light bleeding: HDR emissive converges toward white (color convergence)
      var mag = Math.max(ec.r, ec.g, ec.b);
      if (mag > 1.0) {
        var whiteBlend = Math.min(1, (mag - 1.0) * 0.5);
        result.set(
          ec.r * (1 - whiteBlend) + whiteBlend,
          ec.g * (1 - whiteBlend) + whiteBlend,
          ec.b * (1 - whiteBlend) + whiteBlend, a);
      } else {
        result.set(ec.r, ec.g, ec.b, a);
      }
    } else if (mesh.name && mesh.name.indexOf("bRing") === 0 && mesh.material) {
      var ec2 = mesh.material.emissiveColor;
      result.set(ec2.r, ec2.g, ec2.b, (mesh.material.alpha || 0) * 0.5);
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
  scene.clearColor = new BABYLON.Color4(0.07, 0.08, 0.10, 1);

  // Subtle exponential fog for atmospheric depth (blue-tinted, very faint)
  scene.fogMode = BABYLON.Scene.FOGMODE_EXP2;
  scene.fogDensity = 0.00015;
  scene.fogColor = new BABYLON.Color3(0.05, 0.06, 0.10);

  // Rendering group depth clear: group 0 (device) clears, 1 (plasma) does not, 2 (heatmap) does not
  scene.setRenderingAutoClearDepthStencil(1, true, true, false);
  scene.setRenderingAutoClearDepthStencil(2, false, false, false);

  // -- Camera --
  const cam = new BABYLON.ArcRotateCamera("cam",
    -Math.PI * 0.25, Math.PI * 0.32, G.cathode_radius * 10,
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
  cam.inertia = 0.90;

  // Auto-orbit with 5-second user interaction timeout
  let autoOrbit = true, interacting = false, orbitTimeout = null;
  canvas.addEventListener("pointerdown", function() {
    interacting = true; autoOrbit = false;
    if (orbitTimeout) clearTimeout(orbitTimeout);
  });
  canvas.addEventListener("pointerup", function() {
    interacting = false;
    orbitTimeout = setTimeout(function() { autoOrbit = true; }, 5000);
  });
  var orbitT = 0;
  scene.registerBeforeRender(function() {
    if (autoOrbit && !interacting) {
      orbitT += 0.016;
      cam.alpha += 0.0008;
      // Gentle vertical bob for cinematic feel
      cam.beta = Math.PI * 0.32 + Math.sin(orbitT * 0.3) * 0.05;
    }
  });

  // -- Lighting: key + back + fill for cinematic device definition --
  const keyLight = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-1, -2, 1), scene);
  keyLight.intensity = 0.8;
  keyLight.diffuse = new BABYLON.Color3(0.9, 0.92, 0.95);

  const backLight = new BABYLON.DirectionalLight("back", new BABYLON.Vector3(1, -0.5, -1), scene);
  backLight.intensity = 0.4;
  backLight.diffuse = new BABYLON.Color3(0.6, 0.7, 0.9);

  const fill = new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0, 1, 0), scene);
  fill.intensity = 0.15;
  fill.diffuse = new BABYLON.Color3(0.4, 0.5, 0.7);
  fill.groundColor = new BABYLON.Color3(0.04, 0.04, 0.08);

  // Dynamic plasma light — pulses with discharge, color-matched to phase
  const plasmaLight = new BABYLON.PointLight("plasmaLight",
    new BABYLON.Vector3(G.anode_length, 0, 0), scene);
  plasmaLight.intensity = 0;
  plasmaLight.diffuse = new BABYLON.Color3(0.3, 0.6, 1.0);
  plasmaLight.range = G.cathode_radius * 8;

  // Soft shadows from key light on electrode wireframe
  try {
    var shadowGen = new BABYLON.ShadowGenerator(512, keyLight);  // 512 sufficient for wireframe
    shadowGen.useBlurExponentialShadowMap = true;
    shadowGen.blurKernel = 32;
  } catch (_) { var shadowGen = null; }

  // -- Build all scene objects --
  const dev = buildDevice(scene, G);
  // Add electrode wireframe as shadow casters
  if (shadowGen) {
    try {
      shadowGen.addShadowCaster(dev.anode, true);
      dev.cathodeRods.forEach(function(r) { shadowGen.addShadowCaster(r, true); });
    } catch (_) {}
  }
  const sheath = buildSheath(scene, G);
  const trail = buildTrail(scene, G);
  const pinch = buildPinch(scene, G);
  const haloObj = buildHalo(scene, G);
  const beam = buildBeam(scene, G);
  const reflShock = buildReflectedShock(scene, G);
  const bField = buildBField(scene, G);
  const currentArrows = buildCurrentArrows(scene, G);
  buildGrid(scene, G);
  buildAmbientDust(scene, G);
  const heat = buildHeatmap(scene, G);
  const parts = buildParticles(scene, G);
  const { pipeline, ssao, glowLayer } = buildPipeline(scene, cam);
  const heatPP = buildHeatDistortion(scene, cam);
  const halPP = buildHalation(scene, cam);

  // God rays from pinch core mesh
  let godRays = null;
  try {
    godRays = new BABYLON.VolumetricLightScatteringPostProcess(
      "godRays", { postProcessRatio: 1.0, passRatio: 0.5 },
      cam, pinch.core, 50, BABYLON.Texture.BILINEAR_SAMPLINGMODE, engine, false);
    godRays.weight = 0;
    godRays.decay = 0.96;
    godRays.exposure = 0.25;
  } catch (_) {}

  // -- Snap cache for heatmap animation --
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
    heat.mat.emissiveColor.set(0.25, 0.25, 0.25);
    heat.mat.alpha = 0.60;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.cyl.isVisible = true;
  }

  let activeOverlay = "none";

  function updateHeatmap(ovKey) {
    if (!L || ovKey === "none") { heat.cyl.isVisible = false; return; }
    var cmap = FIELD_CMAPS[ovKey] || activeCmap;
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
        const c = cmapLookup(v, cmap || activeCmap);
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
    heat.mat.emissiveColor.set(0.25, 0.25, 0.25);
    heat.mat.alpha = 0.60;
    heat.mat.useAlphaFromDiffuseTexture = true;
    heat.cyl.isVisible = true;
  }

  let tAccum = 0;

  // Pre-allocate reusable objects (zero GC pressure per frame)
  const _partColor1 = new BABYLON.Color4(0, 0, 0, 0);
  const _partColor2 = new BABYLON.Color4(0, 0, 0, 0);
  const _partColorDead = new BABYLON.Color4(0, 0, 0, 0);
  const _gravVec = new BABYLON.Vector3(0, 0, 0);
  const _focusTarget = new BABYLON.Vector3(0, 0, 0);

  // ============================================================
  // Frame state — computed once per frame, shared by all updaters
  // ============================================================

  var _fs = { f: null, isP: false, cr: 0, Ifrac: 0, pI: 0, col: null, pulse: 0, flicker: 0, rippleAmp: 0 };

  function computeFrameState(i) {
    var f = S.frames[i];
    var isP = isRadial(f.phase);
    var cr = Math.max(0.02, f.r / G.cathode_radius);
    var Ifrac = clamp01(Math.abs(f.I / Math.max(S.I_peak, 0.001)));
    var pI = isP ? Math.min(1, Math.pow(1 - cr, 2) * 3) : 0;
    if (f.phase === "post_pinch") pI *= 0.4;
    if (f.phase === "reflected") pI *= 0.5;
    tAccum = f.t || tAccum;
    var pulse = 1.0 + 0.03 * Math.sin(tAccum * 12) * Ifrac;
    var flicker = 1.0 + 0.15 * Math.sin(tAccum * 37) * Math.cos(tAccum * 23) * Ifrac;
    _fs.f = f; _fs.isP = isP; _fs.cr = cr; _fs.Ifrac = Ifrac; _fs.pI = pI;
    _fs.col = PHASE_COLORS[f.phase] || [0.15, 0.45, 1.0];
    _fs.pulse = pulse; _fs.flicker = flicker; _fs.rippleAmp = 0;
    return _fs;
  }

  // ============================================================
  // updatePlasma — sheath, trail, pinch column, halo, reflected shock, beam
  // ============================================================

  function updatePlasma(s) {
    var f = s.f, isP = s.isP, cr = s.cr, Ifrac = s.Ifrac, pI = s.pI, col = s.col, pulse = s.pulse;

    // Sheath — thin curved shell (Gratton-Vargas parabolic profile)
    if (Ifrac > 0.01) {
      sheath.torus.isVisible = true;
      var curZ = isP ? G.anode_length : f.z;
      var compR = isP ? Math.max(G.anode_radius * 1.1, f.r) : G.cathode_radius * 0.95;
      BABYLON.MeshBuilder.CreateRibbon("sheathDisk", {
        pathArray: sheath.buildSheathPaths(curZ, compR), instance: sheath.torus });

      // Motion afterimage ghost
      sheath.ghost.isVisible = true;
      BABYLON.MeshBuilder.CreateRibbon("sheathGhost", {
        pathArray: sheath.buildSheathPaths(sheath.getPrevZ(), compR), instance: sheath.ghost });
      sheath.ghostMat.alpha = clamp01(Ifrac * 0.12);
      sheath.setPrevZ(lerp(sheath.getPrevZ(), curZ, 0.25));

      if (sheath.isShaderMat) {
        sheath.mat.setFloat("time", tAccum);
        sheath.mat.setFloat("alpha", clamp01(Ifrac * 0.70 * pulse));
        sheath.mat.setVector3("camPos", cam.position);
        var warmT = isP ? smoothstep(0.3, 0.8, pI) : 0;
        if (isP) {
          sheath.mat.setVector3("colA", new BABYLON.Vector3(
            lerp(0.02, 0.30, warmT), lerp(0.05, 0.08, warmT), lerp(0.30, 0.02, warmT)));
          sheath.mat.setVector3("colB", new BABYLON.Vector3(
            lerp(0.10, 1.00, warmT), lerp(0.60, 0.65, warmT), lerp(1.00, 0.15, warmT)));
        } else {
          sheath.mat.setVector3("colA", new BABYLON.Vector3(0.02, 0.05, 0.30));
          sheath.mat.setVector3("colB", new BABYLON.Vector3(col[0], col[1], col[2]));
        }
      } else {
        sheath.mat.alpha = clamp01(Ifrac * 0.65 * pulse);
        if (isP) {
          var warmT2 = smoothstep(0.3, 0.8, pI);
          sheath.mat.emissiveColor.set(
            lerp(col[0], 1.0, warmT2), lerp(col[1], 0.65, warmT2), lerp(col[2], 0.15, warmT2));
        } else {
          sheath.mat.emissiveColor.set(col[0], col[1], col[2]);
        }
        if (sheath.fresnel) {
          sheath.fresnel.bias = lerp(0.2, 0.35, Ifrac);
          sheath.fresnel.power = lerp(2.0, 1.5, Ifrac);
        }
      }
    } else {
      sheath.torus.isVisible = false;
      sheath.ghost.isVisible = false;
    }

    // Plasma trail — dim tube behind sheath
    if (Ifrac > 0.02 && f.z > G.anode_length * 0.05) {
      trail.trail.isVisible = true;
      var extent = isP ? G.anode_length : f.z;
      trail.trail.scaling.x = extent / G.anode_length;
      trail.trail.position.x = extent / 2;
      trail.mat.alpha = clamp01(Ifrac * 0.08 * pulse);
      var wR = isP ? lerp(0.06, 0.30, pI) : 0.06;
      var wG = isP ? lerp(0.12, 0.15, pI) : 0.12;
      var wB = isP ? lerp(0.35, 0.06, pI) : 0.35;
      trail.mat.emissiveColor.set(wR, wG, wB);
    } else {
      trail.trail.isVisible = false;
    }

    // Pinch column — Bennett profile + m=0 sausage instability
    var showPinch = (f.phase === "pinch" || f.phase === "post_pinch" ||
                     f.phase === "reflected") || (isP && cr < 0.35);
    if (showPinch && pI > 0.03) {
      pinch.core.isVisible = true;
      pinch.mantle.isVisible = true;
      var pinchR = Math.max(G.anode_radius * 0.01, cr * G.anode_radius * 0.12);
      s.rippleAmp = f.phase === "post_pinch" ? 0.40 : (f.phase === "pinch" ? 0.08 : 0);
      var nModes = Math.min(5, Math.max(1, Math.round(
        0.25 * G.anode_length / (6.28 * Math.max(pinchR, 0.001)))));

      for (var k = 0; k <= pinch.N; k++) {
        var zf = k / pinch.N;
        var bennett = 1.0 / (1.0 + Math.pow((zf - 0.5) / 0.30, 4));
        var lr = pinchR * (0.15 + 0.85 * bennett);
        var ripple = s.rippleAmp * lr * Math.cos(2 * Math.PI * nModes * zf + tAccum * 4);
        pinch.radii[k] = Math.max(0.0002, lr + ripple);
      }
      BABYLON.MeshBuilder.CreateTube("pinchCore", {
        path: pinch.path, radiusFunction: function(j) { return pinch.radii[j] * 0.35; },
        tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, instance: pinch.core });
      BABYLON.MeshBuilder.CreateTube("pinchMantle", {
        path: pinch.path, radiusFunction: function(j) { return pinch.radii[j]; },
        tessellation: 16, cap: BABYLON.Mesh.NO_CAP,
        sideOrientation: BABYLON.Mesh.DOUBLESIDE, instance: pinch.mantle });

      pinch.coreMat.alpha = clamp01(pI * 0.85 * pulse);
      pinch.mantleMat.alpha = clamp01(pI * 0.30 * pulse);
      if (pI > 0.6) {
        pinch.coreMat.emissiveColor.set(3.0, 2.8, 2.5);
      } else {
        pinch.coreMat.emissiveColor.set(lerp(0.4, 2.0, pI), lerp(0.6, 1.9, pI), lerp(1.0, 1.7, pI));
      }
      pinch.mantleMat.emissiveColor.set(lerp(0.5, 1.4, pI), lerp(0.3, 0.55, pI), lerp(0.8, 0.08, pI));
    } else {
      pinch.core.isVisible = false;
      pinch.mantle.isVisible = false;
    }

    // Halo — dim orange-red glow around pinch
    if (showPinch && pI > 0.1) {
      haloObj.halo.isVisible = true;
      haloObj.halo.scaling.set(1, 1.0 + pI * 1.5, 1.0 + pI * 1.5);
      haloObj.mat.alpha = clamp01(pI * 0.12 * pulse);
      haloObj.mat.emissiveColor.set(lerp(0.6, 1.2, pI), lerp(0.25, 0.5, pI), lerp(0.06, 0.12, pI));
    } else {
      haloObj.halo.isVisible = false;
    }

    // Reflected shock — outward-expanding ring
    if (f.phase === "reflected") {
      reflShock.refRing.isVisible = true;
      var refScale = lerp(0.05, cr, 0.5);
      reflShock.refRing.scaling.set(1, refScale * 8, refScale * 8);
      reflShock.refMat.alpha = clamp01(0.3 * (1 - cr));
    } else {
      reflShock.refRing.isVisible = false;
    }

    // Beam cone — post-pinch particle beam indicator
    beam.cone.isVisible = f.phase === "post_pinch" && pI > 0.08;
    beam.mat.alpha = beam.cone.isVisible ? clamp01(pI * 0.35) : 0;

    // Device thermal response
    if (pI > 0.3) {
      dev.anode.material.emissiveColor.set(lerp(0.22, 0.6, pI), lerp(0.15, 0.25, pI), lerp(0.06, 0.08, pI));
    } else {
      dev.anode.material.emissiveColor.set(0.22, 0.15, 0.06);
    }
    if (Ifrac > 0.01 && !isP && f.z < G.anode_length * 0.15) {
      dev.insulator.material.emissiveColor.set(0.5, 0.4, 0.6);
      dev.insulator.material.alpha = 0.5;
    } else {
      dev.insulator.material.emissiveColor.set(0.35, 0.30, 0.20);
      dev.insulator.material.alpha = 0.40;
    }
  }

  // ============================================================
  // updateFields — B-field rings, current flow arrows
  // ============================================================

  function updateFields(s) {
    var f = s.f, isP = s.isP, cr = s.cr, Ifrac = s.Ifrac, pulse = s.pulse;
    var sheathZ = isP ? G.anode_length : f.z;

    // B-field rings — only visible behind sheath
    var showB = Ifrac > 0.10;
    for (var bi = 0; bi < bField.bRings.length; bi++) {
      bField.bRings[bi].isVisible = showB && G.anode_length * bField.zFracs[bi] < sheathZ;
    }
    if (showB) {
      bField.mat.alpha = clamp01(Ifrac * 0.30 * pulse);
      bField.mat.emissiveColor.set(lerp(0.1, 0.3, Ifrac), lerp(0.3, 0.6, Ifrac), lerp(0.8, 1.0, Ifrac));
      for (var bi2 = 0; bi2 < bField.bRings.length; bi2++) {
        if (!bField.bRings[bi2].isVisible) continue;
        bField.bRings[bi2].rotation.x += 0.008 * Ifrac * (1 + bi2 * 0.15);
        if (isP) {
          var bScale = Math.max(0.15, cr);
          bField.bRings[bi2].scaling.set(1, bScale, bScale);
        } else {
          bField.bRings[bi2].scaling.set(1, 1, 1);
        }
      }
    }

    // Current flow arrows
    var showArrows = Ifrac > 0.05;
    currentArrows.axialArrows.forEach(function(a) {
      a.isVisible = showArrows && (a._userVisible !== false);
    });
    currentArrows.radialArrows.forEach(function(a) {
      a.position.x = sheathZ;
      a.isVisible = showArrows && (a._userVisible !== false);
    });
    currentArrows.returnArrows.forEach(function(a) {
      a.isVisible = showArrows && (a._userVisible !== false);
    });
    currentArrows.mat.alpha = showArrows ? clamp01(Ifrac * 0.8) : 0;
  }

  // ============================================================
  // updateParticles — phase-matched particle emission
  // ============================================================

  function updateParticles(s) {
    var f = s.f, isP = s.isP, cr = s.cr, Ifrac = s.Ifrac, pI = s.pI;
    if (Ifrac < 0.05) { parts.ps.emitRate = 0; return; }

    if (f.phase === "rundown") {
      parts.emitter.position.x = f.z;
      parts.ps.createSphereEmitter(G.cathode_radius * 0.8);
      parts.ps.emitRate = Math.round(Ifrac * 150);
      _gravVec.set(G.cathode_radius * 1.5, 0, 0); parts.ps.gravity = _gravVec;
      parts.ps.minEmitPower = G.cathode_radius * 0.1;
      parts.ps.maxEmitPower = G.cathode_radius * 0.4;
    } else if (f.phase === "radial" || f.phase === "mhd_radial") {
      parts.emitter.position.x = G.anode_length;
      parts.ps.createSphereEmitter(Math.max(0.001, f.r * 0.8));
      parts.ps.emitRate = Math.round(lerp(200, 800, pI));
      _gravVec.set(0, 0, 0); parts.ps.gravity = _gravVec;
      parts.ps.minEmitPower = -f.r * 0.3;
      parts.ps.maxEmitPower = -f.r * 0.1;
    } else if (f.phase === "pinch") {
      parts.emitter.position.x = G.anode_length;
      parts.ps.createSphereEmitter(Math.max(0.001, cr * G.anode_radius * 0.24));
      parts.ps.emitRate = Math.round(lerp(100, 1500, pI));
      _gravVec.set(G.cathode_radius * 0.5, 0, 0); parts.ps.gravity = _gravVec;
      parts.ps.minEmitPower = G.cathode_radius * 0.1;
      parts.ps.maxEmitPower = G.cathode_radius * 0.8;
    } else if (f.phase === "post_pinch") {
      parts.emitter.position.x = G.anode_length;
      parts.ps.createSphereEmitter(Math.max(0.001, cr * G.anode_radius * 0.36));
      parts.ps.emitRate = Math.round(lerp(800, 2500, pI));
      _gravVec.set(G.cathode_radius * 3, 0, 0); parts.ps.gravity = _gravVec;
      parts.ps.minEmitPower = G.cathode_radius * 0.3;
      parts.ps.maxEmitPower = G.cathode_radius * 2.0;
    } else {
      parts.ps.emitRate = Math.round(Ifrac * 100);
      parts.emitter.position.x = isP ? G.anode_length : f.z;
    }

    // Pinch/post-pinch override: burst from column tip
    if (f.phase === "pinch" || f.phase === "post_pinch") {
      parts.ps.emitRate = Math.round(lerp(1500, 3000, pI));
      parts.ps.minEmitPower = G.cathode_radius * 0.5;
      parts.ps.maxEmitPower = G.cathode_radius * 2.0;
    }

    // Phase-matched colors (reuse pre-allocated Color4 objects)
    var pc = s.col;
    _partColor1.set(pc[0], pc[1], pc[2], 0.85);
    _partColor2.set(pc[0] * 0.7 + 0.3, pc[1] * 0.7 + 0.3, pc[2] * 0.7 + 0.3, 0.5);
    _partColorDead.set(pc[0] * 0.3, pc[1] * 0.3, pc[2] * 0.3, 0);
    parts.ps.color1 = _partColor1;
    parts.ps.color2 = _partColor2;
    parts.ps.colorDead = _partColorDead;
  }

  // ============================================================
  // updatePostFX — bloom, glow, camera, lighting, post-processing
  // ============================================================

  function updatePostFX(s) {
    var isP = s.isP, Ifrac = s.Ifrac, pI = s.pI, pulse = s.pulse, flicker = s.flicker;

    // Bloom ramps with pinch intensity (gate prevents over-saturation)
    glowLayer.intensity = lerp(0.25, 0.35, Ifrac);
    var bloomGate = 1.0 - Math.pow(pI, 1.5) * 0.40;
    pipeline.bloomWeight = lerp(0.10, 0.35, pI) * bloomGate;
    pipeline.bloomThreshold = lerp(0.85, 0.55, pI);
    pipeline.bloomScale = lerp(0.5, 0.7, pI);
    pipeline.imageProcessing.exposure = 1.05 + pI * 0.15 * flicker;
    pipeline.chromaticAberration.aberrationAmount = 15 + pI * 25;

    // Heat distortion — mirage near pinch
    if (heatPP) {
      heatPP._heatTime = tAccum;
      heatPP._heatIntensity = pI > 0.2 ? (pI - 0.2) * 1.25 : 0;
    }

    // God rays — volumetric light from pinch core
    if (godRays) {
      godRays.weight = pI > 0.3 ? lerp(0, 0.6, (pI - 0.3) / 0.7) : 0;
    }

    // Dynamic plasma light — color-matched point light at sheath
    plasmaLight.position.x = isP ? G.anode_length : s.f.z;
    plasmaLight.intensity = Ifrac * 0.6 * pulse * flicker;
    if (isP) {
      var warmT = smoothstep(0.3, 0.8, pI);
      plasmaLight.diffuse.set(lerp(0.3, 1.0, warmT), lerp(0.6, 0.7, warmT), lerp(1.0, 0.3, warmT));
    } else {
      plasmaLight.diffuse.set(0.3, 0.6, 1.0);
    }

    // Halation — warm red halo on highlights
    if (halPP) {
      halPP._halIntensity = Ifrac > 0.3 ? lerp(0, 0.8, (Ifrac - 0.3) / 0.7) * pulse : 0;
    }

    // Cinematic camera — zoom toward tip during pinch
    if (autoOrbit && !interacting) {
      var targetR = isP && pI > 0.2 ? G.cathode_radius * lerp(10, 5, pI) : G.cathode_radius * 10;
      cam.radius += (targetR - cam.radius) * 0.015;
      var targetX = isP ? lerp(G.anode_length * 0.45, G.anode_length * 0.75, pI) : G.anode_length * 0.45;
      cam.target.x += (targetX - cam.target.x) * 0.01;
    }

    // Lens breathing — subtle FOV pulse
    if (cam.fov) cam.fov = 0.8 + Math.sin(tAccum * 2) * 0.005 * Ifrac;

    // Depth of field — peak pinch only
    if (pI > 0.5) {
      pipeline.depthOfFieldEnabled = true;
      pipeline.depthOfField.focalLength = 80;
      pipeline.depthOfField.fStop = 3.0 - pI * 1.5;
      _focusTarget.set(G.anode_length, 0, 0);
      pipeline.depthOfField.focusDistance = BABYLON.Vector3.Distance(cam.position, _focusTarget) * 1000;
    } else {
      pipeline.depthOfFieldEnabled = false;
    }
  }

  // ============================================================
  // applyFrame(i) — orchestrator: compute state, delegate to updaters
  // ============================================================

  function applyFrame(i) {
    if (i < 0 || i >= S.frames.length) return;
    var s = computeFrameState(i);

    // Sync heatmap snapshot to current time
    if (activeOverlay !== "none" && snapCache[activeOverlay]) {
      var ni = nearestSnapIdx(snapCache, activeOverlay, s.f.t);
      if (ni !== lastSnapIdx[activeOverlay]) {
        lastSnapIdx[activeOverlay] = ni;
        applySnapTex(activeOverlay);
      }
    }

    updatePlasma(s);
    updateFields(s);
    updateParticles(s);
    updatePostFX(s);

    return { f: s.f, isP: s.isP, cr: s.cr, pI: s.pI, rippleAmp: s.rippleAmp };
  }

  // ============================================================
  // COMPONENT IDENTIFICATION — tooltip on hover/click
  // ============================================================

  var COMPONENT_INFO = {
    anode:        "Anode — central copper electrode, carries discharge current to the plasma",
    insulator:    "Insulator — ceramic sleeve at the breech, initiates surface flashover breakdown",
    chamber:      "Vacuum chamber — sealed enclosure filled with low-pressure deuterium gas",
    cathodeBase:  "Cathode end ring — connects cathode rods at the breech",
    cathodeTop:   "Cathode end ring — connects cathode rods at the muzzle",
    sheathDisk:   "Current sheath — thin current sheet swept by J\u00d7B magnetic force",
    pinchCore:    "Pinch core — hottest region at peak compression, fusion conditions",
    pinchMantle:  "Pinch mantle — outer plasma layer surrounding the hot core",
  };

  function getComponentInfo(name) {
    if (!name) return null;
    if (COMPONENT_INFO[name]) return COMPONENT_INFO[name];
    if (name.indexOf("rod") === 0)         return "Cathode rod — outer electrode, return current path";
    if (name.indexOf("bRing") === 0)       return "B-field ring — toroidal magnetic field from axial current (B \u221d I/r)";
    if (name.indexOf("axialArrow") === 0)  return "Current flow — discharge current travels up the anode (+z direction)";
    if (name.indexOf("radialArrow") === 0) return "Current flow — current crosses through the plasma (radially outward)";
    if (name.indexOf("returnArrow") === 0) return "Current flow — return current travels down the cathode rods (\u2212z direction)";
    return null;
  }

  var tooltip = document.createElement("div");
  tooltip.style.cssText = "position:absolute;z-index:20;pointer-events:none;display:none;" +
    "background:rgba(5,10,25,0.92);border:1px solid rgba(100,160,255,0.4);" +
    "border-radius:6px;padding:8px 12px;color:#cdf;font:13px/1.4 'Helvetica Neue',Arial,sans-serif;" +
    "max-width:280px;text-shadow:0 0 4px #000;box-shadow:0 2px 12px rgba(0,0,0,0.5)";
  canvas.parentElement.appendChild(tooltip);

  scene.onPointerMove = function(evt) {
    var pick = scene.pick(evt.offsetX, evt.offsetY);
    if (pick.hit && pick.pickedMesh) {
      var info = getComponentInfo(pick.pickedMesh.name);
      if (info) {
        tooltip.textContent = info;
        tooltip.style.display = "block";
        tooltip.style.left = (evt.offsetX + 15) + "px";
        tooltip.style.top  = (evt.offsetY - 10) + "px";
        return;
      }
    }
    tooltip.style.display = "none";
  };

  // ============================================================
  // RETURN API
  // ============================================================

  // Scene fade-in: start dark, smoothly reveal over 1.5 seconds
  pipeline.imageProcessing.exposure = 0;
  var fadeStart = performance.now();
  scene.registerBeforeRender(function() {
    var elapsed = (performance.now() - fadeStart) * 0.001;
    if (elapsed < 1.5) {
      var t = elapsed / 1.5;
      pipeline.imageProcessing.exposure = t * t * 1.05;  // ease-in quadratic
    }
  });

  return {
    engine, scene, camera: cam, gpuBackend, useGPU: gpuBackend === "WebGPU",
    G, S, L,
    anode: dev.anode, cathodeRods: dev.cathodeRods, insulator: dev.insulator,
    sheathDisk: sheath.torus, pinchCore: pinch.core, pinchMantle: pinch.mantle,
    beamCone: beam.cone, gasGlow: trail.trail,
    bRings: bField.bRings, fieldLines: [],
    currentArrows,
    ps: { start: function() { parts.ps.start(); }, stop: function() { parts.ps.stop(); } },
    pipeline, ssao, glowLayer,
    applyFrame,
    updateHeatmap,
    activeOverlay,
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
