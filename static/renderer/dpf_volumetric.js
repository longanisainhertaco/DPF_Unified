/**
 * DPF Volumetric Field Renderer
 *
 * Renders 2D axisymmetric (r,z) field data as a 3D volume via ray-marching.
 * The key insight: axisymmetric data means field(x,y,z) = field(sqrt(x^2+y^2), z).
 * The fragment shader converts each ray sample from Cartesian to (r,z) and looks up
 * the 2D data texture directly. No 3D texture needed.
 *
 * Also provides a cross-section half-plane for exact value inspection.
 *
 * Data format: normalized Float32 array of shape (nr, nz), row-major, values in [0,1].
 * nr = radial cells (e.g. 16), nz = axial cells (e.g. 32).
 */

// ============================================================
// VOLUMETRIC RAY-MARCH SHADERS
// ============================================================

const _VOL_VERT = `
precision highp float;
attribute vec3 position;
uniform mat4 world;
uniform mat4 worldViewProjection;
varying vec3 vWorldPos;
varying vec3 vLocalPos;
void main() {
  gl_Position = worldViewProjection * vec4(position, 1.0);
  vWorldPos = (world * vec4(position, 1.0)).xyz;
  vLocalPos = position;
}
`;

// The ray-marcher: steps through a bounding box, converts to (r,z), samples 2D texture.
// Colormaps are baked in as GLSL functions (viridis + inferno).
const _VOL_FRAG = `
precision highp float;

varying vec3 vWorldPos;
varying vec3 vLocalPos;

uniform vec3 camPos;
uniform mat4 worldInverse;
uniform float opacity;       // global opacity multiplier [0,1]
uniform float densityScale;  // absorption coefficient
uniform float rMax;          // outer radius in local coords
uniform float zLen;          // axial length in local coords
uniform int numSteps;        // ray-march steps (32-64 typical)
uniform float stepSize;      // precomputed: diagonal / numSteps
uniform sampler2D fieldTex;  // 2D (nr x nz) field data, R channel = normalized value
uniform int cmapIndex;       // 0=viridis, 1=inferno, 2=cividis

// Viridis (5-stop approximation)
vec3 viridis(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.267, 0.004, 0.329);
  vec3 c1 = vec3(0.128, 0.567, 0.551);
  vec3 c2 = vec3(0.267, 0.749, 0.441);
  vec3 c3 = vec3(0.741, 0.873, 0.150);
  vec3 c4 = vec3(0.993, 0.906, 0.144);
  float s = t * 4.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  return mix(c3, c4, s - 3.0);
}

// Inferno (5-stop)
vec3 inferno(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.001, 0.000, 0.014);
  vec3 c1 = vec3(0.434, 0.063, 0.406);
  vec3 c2 = vec3(0.692, 0.194, 0.261);
  vec3 c3 = vec3(0.882, 0.414, 0.100);
  vec3 c4 = vec3(0.988, 0.998, 0.645);
  float s = t * 4.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  return mix(c3, c4, s - 3.0);
}

// Cividis (5-stop)
vec3 cividis(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.0, 0.135, 0.305);
  vec3 c1 = vec3(0.259, 0.335, 0.384);
  vec3 c2 = vec3(0.463, 0.461, 0.420);
  vec3 c3 = vec3(0.775, 0.685, 0.432);
  vec3 c4 = vec3(1.0, 0.871, 0.298);
  float s = t * 4.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  return mix(c3, c4, s - 3.0);
}

vec3 applyColormap(float t) {
  if (cmapIndex == 1) return inferno(t);
  if (cmapIndex == 2) return cividis(t);
  return viridis(t);
}

// Ray-box intersection for axis-aligned box centered at origin
// box half-extents: (rMax, rMax, zLen/2)
vec2 intersectBox(vec3 ro, vec3 rd, vec3 halfExt) {
  vec3 invRd = 1.0 / rd;
  vec3 t0 = (-halfExt - ro) * invRd;
  vec3 t1 = ( halfExt - ro) * invRd;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tNear = max(max(tmin.x, tmin.y), tmin.z);
  float tFar  = min(min(tmax.x, tmax.y), tmax.z);
  return vec2(tNear, tFar);
}

void main() {
  // Ray in local (object) space
  vec3 camLocal = (worldInverse * vec4(camPos, 1.0)).xyz;
  vec3 rd = normalize(vLocalPos - camLocal);
  vec3 ro = camLocal;

  // Bounding box: x,y in [-rMax, rMax], z in [-zLen/2, zLen/2]
  vec3 halfExt = vec3(rMax, rMax, zLen * 0.5);
  vec2 tHit = intersectBox(ro, rd, halfExt);
  if (tHit.x > tHit.y) discard;

  float tStart = max(tHit.x, 0.0);
  float tEnd = tHit.y;

  // Accumulate color via front-to-back compositing
  vec4 accum = vec4(0.0);

  for (int i = 0; i < 128; i++) {
    if (i >= numSteps) break;
    float t = tStart + (float(i) + 0.5) * stepSize;
    if (t > tEnd) break;

    vec3 p = ro + rd * t;

    // Convert to cylindrical: r = sqrt(x^2 + y^2), z = z
    float r = length(p.xy);
    float z = p.z + zLen * 0.5;  // shift z from [-zLen/2, zLen/2] to [0, zLen]

    // Skip if outside cylinder
    if (r > rMax) continue;

    // Texture UV: u = z/zLen (axial), v = r/rMax (radial)
    // Texture layout: width=nz, height=nr, so u maps to x (axial), v maps to y (radial)
    float u = clamp(z / zLen, 0.0, 1.0);
    float v = clamp(r / rMax, 0.0, 1.0);

    float val = texture2D(fieldTex, vec2(u, v)).r;

    // Transfer function: color from colormap, opacity from value
    vec3 col = applyColormap(val);
    float sampleAlpha = val * val * densityScale * stepSize * opacity;
    sampleAlpha = clamp(sampleAlpha, 0.0, 1.0);

    // Front-to-back compositing
    accum.rgb += (1.0 - accum.a) * sampleAlpha * col;
    accum.a   += (1.0 - accum.a) * sampleAlpha;

    if (accum.a > 0.98) break;  // early exit
  }

  gl_FragColor = vec4(accum.rgb, accum.a);
}
`;

// ============================================================
// CROSS-SECTION HALF-PLANE SHADERS
// ============================================================

const _XSEC_VERT = `
precision highp float;
attribute vec3 position;
attribute vec2 uv;
uniform mat4 worldViewProjection;
varying vec2 vUV;
void main() {
  gl_Position = worldViewProjection * vec4(position, 1.0);
  vUV = uv;
}
`;

const _XSEC_FRAG = `
precision highp float;
varying vec2 vUV;
uniform sampler2D fieldTex;
uniform float alpha;
void main() {
  float val = texture2D(fieldTex, vUV).r;
  // Discard near-zero for transparency at axis
  if (val < 0.01) discard;
  gl_FragColor = vec4(texture2D(fieldTex, vUV).rgb, alpha);
}
`;

// ============================================================
// MODULE: buildVolumetricField
// ============================================================

/**
 * Build the volumetric field visualization.
 *
 * @param {BABYLON.Scene} scene
 * @param {Object} G - geometry: {anode_radius, cathode_radius, anode_length}
 * @param {Object} opts - optional overrides: {numSteps, opacity, densityScale}
 * @returns {Object} API: {update(fieldArray, nr, nz, cmapIdx), setOpacity(v), setDensity(v), dispose(), mesh, crossSection}
 */
function buildVolumetricField(scene, G, opts) {
  opts = opts || {};
  var rMax = G.cathode_radius || 0.04;
  var zLen = G.anode_length || 0.16;
  var nSteps = opts.numSteps || 48;
  var diagonal = Math.sqrt(4 * rMax * rMax + zLen * zLen);
  var sSize = diagonal / nSteps;

  // Register shaders
  BABYLON.Effect.ShadersStore["volFieldVertexShader"] = _VOL_VERT;
  BABYLON.Effect.ShadersStore["volFieldFragmentShader"] = _VOL_FRAG;
  BABYLON.Effect.ShadersStore["xsecFieldVertexShader"] = _XSEC_VERT;
  BABYLON.Effect.ShadersStore["xsecFieldFragmentShader"] = _XSEC_FRAG;

  // Bounding box mesh — a unit box scaled to the cylindrical domain
  var box = BABYLON.MeshBuilder.CreateBox("volFieldBox", {
    width: rMax * 2, height: rMax * 2, depth: zLen,
  }, scene);
  // Position: centered on anode axis. x=axial in the renderer, so z-local = x-world
  box.rotation.z = Math.PI / 2;
  box.position.x = zLen / 2;
  box.isPickable = false;
  box.renderingGroupId = 2;

  var mat = new BABYLON.ShaderMaterial("volFieldMat", scene,
    { vertex: "volField", fragment: "volField" },
    {
      attributes: ["position"],
      uniforms: [
        "world", "worldViewProjection", "worldInverse",
        "camPos", "opacity", "densityScale",
        "rMax", "zLen", "numSteps", "stepSize",
        "cmapIndex",
      ],
      samplers: ["fieldTex"],
      needAlphaBlending: true,
    }
  );
  mat.backFaceCulling = false;
  mat.setFloat("opacity", opts.opacity || 0.8);
  mat.setFloat("densityScale", opts.densityScale || 8.0);
  mat.setFloat("rMax", rMax);
  mat.setFloat("zLen", zLen);
  mat.setInt("numSteps", nSteps);
  mat.setFloat("stepSize", sSize);
  mat.setInt("cmapIndex", 0);
  box.material = mat;
  box.isVisible = false;

  // Dummy 2x2 texture until real data arrives
  var texData = new Float32Array([0, 0, 0, 0]);
  var fieldTex = new BABYLON.RawTexture(texData, 2, 2,
    BABYLON.Engine.TEXTUREFORMAT_R, scene, false, false,
    BABYLON.Texture.BILINEAR_SAMPLINGMODE, BABYLON.Engine.TEXTURETYPE_FLOAT);
  mat.setTexture("fieldTex", fieldTex);

  // Camera position uniform — update every frame
  scene.registerBeforeRender(function() {
    var cam = scene.activeCamera;
    if (cam) {
      mat.setVector3("camPos", cam.position);
      // World inverse for local-space ray computation
      var worldMat = box.getWorldMatrix();
      var inv = BABYLON.Matrix.Identity();
      worldMat.invertToRef(inv);
      mat.setMatrix("worldInverse", inv);
    }
  });

  // ---- Cross-section half-plane ----
  // A flat plane from axis (r=0) to cathode (r=rMax) along z,
  // showing the full r-z data as a colormapped texture.
  var xsecPlane = BABYLON.MeshBuilder.CreatePlane("xsecPlane", {
    width: zLen, height: rMax,
    sideOrientation: BABYLON.Mesh.DOUBLESIDE,
  }, scene);
  // Position: sits at y=rMax/2 (half-height), rotated to be a vertical slice
  xsecPlane.rotation.z = Math.PI / 2;
  xsecPlane.rotation.y = 0;
  xsecPlane.position.x = zLen / 2;
  xsecPlane.position.y = rMax / 2;
  xsecPlane.isPickable = false;
  xsecPlane.renderingGroupId = 2;
  xsecPlane.isVisible = false;

  var xsecMat = new BABYLON.StandardMaterial("xsecMat", scene);
  xsecMat.disableLighting = true;
  xsecMat.backFaceCulling = false;
  xsecMat.alpha = 0.85;
  xsecPlane.material = xsecMat;
  var xsecTex = null;

  // ---- API ----

  /**
   * Update the volumetric field with new data.
   * @param {Float32Array} fieldArray - normalized [0,1] values, shape (nr, nz), row-major
   * @param {number} nr - radial cells
   * @param {number} nz - axial cells
   * @param {number} cmapIdx - 0=viridis, 1=inferno, 2=cividis
   */
  function update(fieldArray, nr, nz, cmapIdx) {
    // Volumetric texture: just the raw float data, sampled as (u=z/nz, v=r/nr)
    if (fieldTex) fieldTex.dispose();
    fieldTex = new BABYLON.RawTexture(fieldArray, nz, nr,
      BABYLON.Engine.TEXTUREFORMAT_R, scene, false, false,
      BABYLON.Texture.BILINEAR_SAMPLINGMODE, BABYLON.Engine.TEXTURETYPE_FLOAT);
    mat.setTexture("fieldTex", fieldTex);
    mat.setInt("cmapIndex", cmapIdx || 0);
    box.isVisible = true;

    // Cross-section: create RGBA colormapped texture for the plane
    _updateCrossSection(fieldArray, nr, nz, cmapIdx || 0);
  }

  function _updateCrossSection(fieldArray, nr, nz, cmapIdx) {
    var rgba = new Uint8Array(nz * nr * 4);
    for (var ir = 0; ir < nr; ir++) {
      for (var iz = 0; iz < nz; iz++) {
        var val = fieldArray[ir * nz + iz];
        var col = _glslColormap(val, cmapIdx);
        // Flip radial axis so r=0 is at bottom of texture
        var pi = ((nr - 1 - ir) * nz + iz) * 4;
        rgba[pi]     = Math.round(col[0] * 255);
        rgba[pi + 1] = Math.round(col[1] * 255);
        rgba[pi + 2] = Math.round(col[2] * 255);
        rgba[pi + 3] = Math.round(val * 220 + 35);  // alpha: min 35, scales with value
      }
    }
    if (xsecTex) xsecTex.dispose();
    xsecTex = new BABYLON.RawTexture(rgba, nz, nr,
      BABYLON.Engine.TEXTUREFORMAT_RGBA, scene, false, false,
      BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    xsecMat.diffuseTexture = xsecTex;
    xsecMat.emissiveTexture = xsecTex;
    xsecMat.useAlphaFromDiffuseTexture = true;
  }

  // JS-side colormap matching the GLSL (for cross-section texture)
  function _glslColormap(t, idx) {
    t = Math.max(0, Math.min(1, t));
    var stops;
    if (idx === 1) {
      stops = [[0.001,0,0.014],[0.434,0.063,0.406],[0.692,0.194,0.261],[0.882,0.414,0.1],[0.988,0.998,0.645]];
    } else if (idx === 2) {
      stops = [[0,0.135,0.305],[0.259,0.335,0.384],[0.463,0.461,0.42],[0.775,0.685,0.432],[1,0.871,0.298]];
    } else {
      stops = [[0.267,0.004,0.329],[0.128,0.567,0.551],[0.267,0.749,0.441],[0.741,0.873,0.15],[0.993,0.906,0.144]];
    }
    var s = t * 4;
    var lo = Math.min(Math.floor(s), 3);
    var f = s - lo;
    return [
      stops[lo][0] + (stops[lo+1][0] - stops[lo][0]) * f,
      stops[lo][1] + (stops[lo+1][1] - stops[lo][1]) * f,
      stops[lo][2] + (stops[lo+1][2] - stops[lo][2]) * f,
    ];
  }

  function setOpacity(v) { mat.setFloat("opacity", v); }
  function setDensity(v) { mat.setFloat("densityScale", v); }

  function setMode(mode) {
    // "volume" = volumetric only, "xsec" = cross-section only, "both" = both
    box.isVisible = (mode === "volume" || mode === "both");
    xsecPlane.isVisible = (mode === "xsec" || mode === "both");
  }

  function dispose() {
    if (fieldTex) fieldTex.dispose();
    if (xsecTex) xsecTex.dispose();
    box.dispose();
    xsecPlane.dispose();
  }

  return {
    update: update,
    setOpacity: setOpacity,
    setDensity: setDensity,
    setMode: setMode,
    dispose: dispose,
    mesh: box,
    crossSection: xsecPlane,
  };
}

window.buildVolumetricField = buildVolumetricField;
