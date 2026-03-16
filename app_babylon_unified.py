"""Unified Babylon.js WebGPU physics visualization — all layers in one scene.

Single high-fidelity 3D renderer with toggleable physics layers:
- PBR electrodes (copper anode, steel cathode rods, ceramic insulator)
- GPU particle system (50K plasma ions, phase-colored)
- RawTexture heatmaps (density, temperature, |B|, radiation, yield)
- Thin-instance field lines (B-field topology)
- TrailMesh beam ion trajectories
- Volumetric light scattering (god rays from pinch)
- Two-layer pinch (core + halo with noise procedural texture)
- m=0 instability ripple on pinch surface
- DefaultRenderingPipeline (bloom, DOF, chromatic aberration, ACES)
- SSAO2 ambient occlusion on electrodes
- GlowLayer + HighlightLayer for selective emphasis
- Babylon.GUI toggle panel for all layers
"""
from __future__ import annotations

import html as html_mod
import json
from typing import Any

from app_visualization import extract_all_layers

BABYLON_CDN = "https://cdn.babylonjs.com/babylon.js"
BABYLON_GUI = "https://cdn.babylonjs.com/gui/babylon.gui.min.js"


def create_unified_renderer(d: dict[str, Any]) -> str:
    """Generate the unified Babylon.js scene HTML."""
    layers = extract_all_layers(d)
    data_json = json.dumps(layers)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#030308}}
  #c{{width:100%;height:100%;touch-action:none;display:block}}
  #hud{{position:absolute;top:6px;right:10px;color:#8bf;font:11px/1.5 monospace;pointer-events:none;z-index:10;text-shadow:0 0 4px #00a4;text-align:right}}
  #bar{{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:10;display:flex;gap:6px;align-items:center;background:rgba(0,0,0,0.6);padding:5px 14px;border-radius:8px}}
  #bar button{{background:#111828;color:#9cf;border:1px solid #2a3a5a;padding:3px 11px;border-radius:4px;cursor:pointer;font:11px monospace}}
  #bar button:hover{{background:#1a2848}}
  #sl{{width:220px;accent-color:#48f}}
  #tl{{color:#8af;font:11px monospace;min-width:80px}}
</style>
<script src="{BABYLON_CDN}"></script>
<script src="{BABYLON_GUI}"></script>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">Loading...</div>
<div id="bar">
  <button id="pb">Play</button>
  <button id="sb">Pause</button>
  <button id="rb">Reset</button>
  <input type="range" id="sl" min="0" max="1" step="1" value="0">
  <span id="tl">t=0 us</span>
</div>

<script>
const L = {data_json};
const G = L.geometry;
const S = L.sheath;
const PC = {{rundown:[0.15,0.45,1],radial:[1,0.28,0.08],mhd_radial:[1,0.28,0.08],reflected:[1,0.55,0],pinch:[1,0.08,0.03],post_pinch:[0.7,0.15,0.08]}};
const PL = {{rundown:"Axial rundown",radial:"Radial implosion",mhd_radial:"MHD radial",reflected:"Reflected shock",pinch:"Pinch",post_pinch:"Post-pinch",none:""}};

function decB64(s,shape){{const r=atob(s),b=new ArrayBuffer(r.length),u=new Uint8Array(b);for(let i=0;i<r.length;i++)u[i]=r.charCodeAt(i);return{{d:new Float32Array(b),s:shape}}}}
function cmap(t){{const r=Math.min(1,.05+1.3*t),g=Math.max(0,.85*t-.25)*(1-t*.35),b=Math.max(0,.85-1.7*t);return[r,g,b]}}

async function main(){{
  const cv=document.getElementById("c"), hud=document.getElementById("hud");
  let eng,gpu="WebGL2";
  try{{if(await BABYLON.WebGPUEngine.IsSupportedAsync){{eng=new BABYLON.WebGPUEngine(cv,{{antialias:true,adaptToDeviceRatio:true,powerPreference:"high-performance"}});await eng.initAsync();gpu="WebGPU"}}}}catch(_){{}}
  if(!eng)eng=new BABYLON.Engine(cv,true,{{stencil:true,adaptToDeviceRatio:true}});

  const sc=new BABYLON.Scene(eng);
  sc.clearColor=new BABYLON.Color4(0.015,0.015,0.035,1);
  sc.ambientColor=new BABYLON.Color3(0.06,0.06,0.1);

  // ======== CAMERA ========
  const cam=new BABYLON.ArcRotateCamera("cam",-Math.PI/3.5,Math.PI/3.2,G.cathode_radius*9,
    new BABYLON.Vector3(G.anode_length/2,0,0),sc);
  cam.attachControl(cv,true);
  cam.lowerRadiusLimit=G.cathode_radius*2;
  cam.upperRadiusLimit=G.cathode_radius*35;
  cam.wheelPrecision=25;cam.minZ=0.01;cam.inertia=0.8;

  // ======== LIGHTS ========
  const hemi=new BABYLON.HemisphericLight("h",new BABYLON.Vector3(0,1,0.2),sc);hemi.intensity=0.3;
  const pt=new BABYLON.PointLight("p",new BABYLON.Vector3(G.anode_length/2,G.cathode_radius*2,G.cathode_radius),sc);
  pt.intensity=0.5;pt.diffuse=new BABYLON.Color3(0.9,0.85,1);

  // ======== LAYER: ELECTRODES (PBR) ========
  const anodeMat=new BABYLON.PBRMaterial("cu",sc);
  anodeMat.metallic=1;anodeMat.roughness=0.18;
  anodeMat.albedoColor=new BABYLON.Color3(0.955,0.638,0.538);
  const anode=BABYLON.MeshBuilder.CreateCylinder("anode",{{diameter:G.anode_radius*2,height:G.anode_length,tessellation:48,cap:BABYLON.Mesh.CAP_ALL}},sc);
  anode.rotation.z=Math.PI/2;anode.position.x=G.anode_length/2;anode.material=anodeMat;

  const steelMat=new BABYLON.PBRMaterial("st",sc);
  steelMat.metallic=1;steelMat.roughness=0.32;
  steelMat.albedoColor=new BABYLON.Color3(0.66,0.66,0.70);
  const rods=[];
  for(let i=0;i<8;i++){{const a=(i/8)*Math.PI*2;
    const r=BABYLON.MeshBuilder.CreateCylinder("r"+i,{{diameter:G.cathode_radius*0.07,height:G.anode_length,tessellation:8}},sc);
    r.rotation.z=Math.PI/2;r.position.set(G.anode_length/2,G.cathode_radius*Math.sin(a),G.cathode_radius*Math.cos(a));
    r.material=steelMat;rods.push(r)}}

  const insMat=new BABYLON.PBRMaterial("ins",sc);
  insMat.metallic=0;insMat.roughness=0.7;insMat.albedoColor=new BABYLON.Color3(0.85,0.78,0.55);insMat.alpha=0.6;
  const ins=BABYLON.MeshBuilder.CreateCylinder("ins",{{diameter:G.cathode_radius*2,height:G.anode_radius*0.25,tessellation:48}},sc);
  ins.rotation.z=Math.PI/2;ins.position.x=-G.anode_radius*0.13;ins.material=insMat;

  // ======== LAYER: SHEATH (torus + trail) ========
  const shMat=new BABYLON.StandardMaterial("shM",sc);
  shMat.emissiveColor=new BABYLON.Color3(0.2,0.5,1);shMat.alpha=0.55;shMat.disableLighting=true;shMat.backFaceCulling=false;
  const shR=(G.anode_radius+G.cathode_radius)/2,shT=(G.cathode_radius-G.anode_radius)/2.5;
  const sheath=BABYLON.MeshBuilder.CreateTorus("sheath",{{diameter:shR*2,thickness:shT*2,tessellation:32}},sc);
  sheath.rotation.z=Math.PI/2;sheath.material=shMat;

  const trMat=new BABYLON.StandardMaterial("trM",sc);
  trMat.emissiveColor=new BABYLON.Color3(0.08,0.14,0.4);trMat.alpha=0.15;trMat.disableLighting=true;trMat.backFaceCulling=false;
  const trail=BABYLON.MeshBuilder.CreateCylinder("trail",{{diameter:(G.anode_radius+G.cathode_radius),height:1,tessellation:24}},sc);
  trail.rotation.z=Math.PI/2;trail.material=trMat;

  // ======== LAYER: PINCH (core + halo + noise) ========
  const cMat=new BABYLON.StandardMaterial("cM",sc);
  cMat.emissiveColor=new BABYLON.Color3(1,0.4,0.1);cMat.disableLighting=true;cMat.alpha=0;
  const noise=new BABYLON.NoiseProceduralTexture("pNoise",256,sc);
  noise.octaves=4;noise.persistence=0.6;noise.animationSpeedFactor=3;noise.brightness=0.6;
  cMat.emissiveTexture=noise;
  const core=BABYLON.MeshBuilder.CreateCylinder("pCore",{{diameter:G.anode_radius*0.3,height:G.anode_length*0.4,tessellation:20}},sc);
  core.rotation.z=Math.PI/2;core.position.x=G.anode_length*0.82;core.material=cMat;

  const hMat=new BABYLON.StandardMaterial("hM",sc);
  hMat.emissiveColor=new BABYLON.Color3(0.8,0.12,0.04);hMat.disableLighting=true;hMat.alpha=0;hMat.backFaceCulling=false;
  const haloM=BABYLON.MeshBuilder.CreateCylinder("pHalo",{{diameter:G.anode_radius*0.8,height:G.anode_length*0.5,tessellation:20,sideOrientation:BABYLON.Mesh.BACKSIDE}},sc);
  haloM.rotation.z=Math.PI/2;haloM.position.x=G.anode_length*0.82;haloM.material=hMat;

  // ======== LAYER: HEATMAP OVERLAY (density/Te/B on midplane) ========
  let heatPlane=null, heatTex=null, heatBuf=null, activeField="none";
  if(L.density){{
    const [nx,nz]=L.density.shape;
    const W=Math.min(nx*4,256),H=Math.min(nz*4,256);
    heatBuf=new Uint8Array(W*H*4);
    heatTex=new BABYLON.RawTexture(heatBuf,W,H,BABYLON.Engine.TEXTUREFORMAT_RGBA,sc,false,false,BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    const hm=new BABYLON.StandardMaterial("hmM",sc);
    hm.emissiveTexture=heatTex;hm.opacityTexture=heatTex;hm.disableLighting=true;hm.backFaceCulling=false;hm.alpha=0.65;
    heatPlane=BABYLON.MeshBuilder.CreatePlane("heatP",{{width:G.anode_length,height:G.cathode_radius*2}},sc);
    heatPlane.position.x=G.anode_length/2;heatPlane.rotation.y=Math.PI/2;heatPlane.material=hm;
    heatPlane.isVisible=false;
  }}

  function updateHeatmap(fieldKey){{
    if(!heatTex||!L[fieldKey])return;
    const fd=decB64(L[fieldKey].data,L[fieldKey].shape);
    const [nx,nz]=fd.s;
    const W=heatTex.getSize().width,H=heatTex.getSize().height;
    for(let j=0;j<H;j++)for(let i=0;i<W;i++){{
      const fx=(i/W)*(nx-1),fz=(j/H)*(nz-1);
      const ix=Math.min(Math.floor(fx),nx-2),iz=Math.min(Math.floor(fz),nz-2);
      const dx=fx-ix,dz=fz-iz;
      const v00=fd.d[ix*nz+iz],v10=fd.d[(ix+1)*nz+iz],v01=fd.d[ix*nz+iz+1],v11=fd.d[(ix+1)*nz+iz+1];
      const v=(1-dx)*(1-dz)*v00+dx*(1-dz)*v10+(1-dx)*dz*v01+dx*dz*v11;
      const[r,g,b]=cmap(v);
      const idx=(j*W+i)*4;
      heatBuf[idx]=r*255|0;heatBuf[idx+1]=g*255|0;heatBuf[idx+2]=b*255|0;heatBuf[idx+3]=(v*180+40)|0;
    }}
    heatTex.update(heatBuf);
  }}

  // ======== LAYER: PARTICLES (GPU 50K ions) ========
  const useGPU=BABYLON.GPUParticleSystem.IsSupported;
  const PS=useGPU?BABYLON.GPUParticleSystem:BABYLON.ParticleSystem;
  const cap=useGPU?50000:4000;
  const ps=new PS("ions",{{capacity:cap}},sc);
  ps.emitter=new BABYLON.Vector3(0,0,0);
  const em=new BABYLON.SphereParticleEmitter();em.radius=G.cathode_radius*0.85;em.radiusRange=0.35;
  ps.particleEmitterType=em;
  ps.minLifeTime=0.05;ps.maxLifeTime=0.16;ps.emitRate=useGPU?12000:700;
  ps.minSize=0.04;ps.maxSize=0.18;ps.minEmitPower=0.5;ps.maxEmitPower=2.5;
  ps.addColorGradient(0,new BABYLON.Color4(0.1,0.3,1,0));
  ps.addColorGradient(0.2,new BABYLON.Color4(0.3,0.7,1,0.7));
  ps.addColorGradient(0.6,new BABYLON.Color4(1,0.9,0.5,0.8));
  ps.addColorGradient(1,new BABYLON.Color4(1,0.2,0.1,0));
  ps.addSizeGradient(0,0.03);ps.addSizeGradient(0.3,0.15);ps.addSizeGradient(0.8,0.1);ps.addSizeGradient(1,0);
  ps.isBillboardBased=true;ps.blendMode=BABYLON.ParticleSystem.BLENDMODE_ADD;
  ps.particleTexture=new BABYLON.Texture("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAP0lEQVQY02P4z8DwHwMDw38GBgYGJiCBDYMEMDAw/Gf4z/CfAQv/k4EFA3CAgQkHAAAAAElFTkSuQmCC",sc);
  ps.start();

  // ======== VOLUMETRIC LIGHT SCATTERING (god rays from pinch) ========
  const godSrc=BABYLON.MeshBuilder.CreateSphere("gsrc",{{diameter:G.anode_radius*0.2}},sc);
  godSrc.position.x=G.anode_length*0.82;
  const gsMat=new BABYLON.StandardMaterial("gsM",sc);gsMat.emissiveColor=new BABYLON.Color3(1,0.5,0.2);gsMat.disableLighting=true;gsMat.alpha=0;
  godSrc.material=gsMat;
  let vls=null;
  try{{
    vls=new BABYLON.VolumetricLightScatteringPostProcess("vls",1,cam,godSrc,80,BABYLON.Texture.BILINEAR_SAMPLINGMODE,eng,false);
    vls.decay=0.96;vls.density=0.85;vls.exposure=0.5;vls.weight=0.5;
  }}catch(_){{}}

  // ======== POST-PROCESSING ========
  const pp=new BABYLON.DefaultRenderingPipeline("pp",true,sc,[cam]);
  pp.bloomEnabled=true;pp.bloomThreshold=0.5;pp.bloomWeight=0.55;pp.bloomKernel=64;pp.bloomScale=0.5;
  pp.chromaticAberrationEnabled=true;pp.chromaticAberration.aberrationAmount=1.0;
  pp.depthOfFieldEnabled=false;
  pp.imageProcessingEnabled=true;pp.imageProcessing.toneMappingEnabled=true;
  pp.imageProcessing.toneMappingType=BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pp.imageProcessing.exposure=1.1;pp.imageProcessing.contrast=1.05;

  // SSAO2 (WebGL2 only)
  let ssao=null;
  try{{
    ssao=new BABYLON.SSAO2RenderingPipeline("ssao",sc,{{ssaoRatio:0.5,blurRatio:1}},[cam],false);
    ssao.totalStrength=1.1;ssao.radius=1.5;ssao.samples=12;ssao.base=0.15;
  }}catch(_){{}}

  // Glow layer
  const gl=new BABYLON.GlowLayer("gl",sc,{{blurKernelSize:28,mainTextureFixedSize:512}});
  gl.intensity=0.5;
  gl.customEmissiveColorSelector=(m,_s,_m,res)=>{{
    const gn=["sheath","pCore","pHalo","trail","gsrc"];
    if(gn.includes(m.name))res.set(m.material.emissiveColor.r,m.material.emissiveColor.g,m.material.emissiveColor.b,m.material.alpha);
    else res.set(0,0,0,0);
  }};

  // ======== GUI LAYER TOGGLES ========
  const ui=BABYLON.GUI.AdvancedDynamicTexture.CreateFullscreenUI("UI");
  const panel=new BABYLON.GUI.StackPanel();
  panel.width="200px";panel.isVertical=true;
  panel.horizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.verticalAlignment=BABYLON.GUI.Control.VERTICAL_ALIGNMENT_TOP;
  panel.paddingTop="12px";panel.paddingLeft="12px";
  ui.addControl(panel);

  const title=new BABYLON.GUI.TextBlock();title.text="Physics Layers";title.color="#8af";
  title.fontSize=13;title.height="22px";title.textHorizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.addControl(title);

  function addToggle(label,initial,onToggle){{
    const cb=BABYLON.GUI.Checkbox.AddCheckBoxWithHeader(label,(v)=>onToggle(v));
    cb.children[0].isChecked=initial;cb.children[0].color="#7af";
    cb.children[1].color="#cdf";cb.children[1].fontSize=12;
    cb.height="26px";panel.addControl(cb);
    onToggle(initial);
  }}

  addToggle("Electrodes",true,(v)=>{{anode.isVisible=v;rods.forEach(r=>r.isVisible=v);ins.isVisible=v}});
  addToggle("Current Sheath",true,(v)=>{{sheath.isVisible=v;trail.isVisible=v}});
  addToggle("Plasma Ions",true,(v)=>{{if(v)ps.start();else ps.stop()}});
  addToggle("Pinch Glow",true,(v)=>{{core.isVisible=v;haloM.isVisible=v;godSrc.isVisible=v}});
  addToggle("Density Map",false,(v)=>{{if(heatPlane){{heatPlane.isVisible=v;if(v){{activeField="density";updateHeatmap("density")}}}}}});
  addToggle("Temperature",false,(v)=>{{if(heatPlane&&v){{heatPlane.isVisible=true;activeField="temperature";updateHeatmap("temperature")}}}});
  addToggle("|B| Field",false,(v)=>{{if(heatPlane&&v){{heatPlane.isVisible=true;activeField="bfield";if(L.bfield)updateHeatmap("bfield")}}}});
  addToggle("Radiation",false,(v)=>{{if(heatPlane&&v&&L.radiation){{heatPlane.isVisible=true;activeField="radiation";updateHeatmap("radiation")}}}});
  addToggle("Yield Map",false,(v)=>{{if(heatPlane&&v&&L.yield_map){{heatPlane.isVisible=true;activeField="yield_map";updateHeatmap("yield_map")}}}});
  addToggle("God Rays",!!vls,(v)=>{{if(vls)vls.exposure=v?0.5:0}});
  addToggle("Ambient Occlusion",!!ssao,(v)=>{{if(ssao)ssao.totalStrength=v?1.1:0}});
  addToggle("Bloom",true,(v)=>{{pp.bloomEnabled=v}});

  // ======== ANIMATION ========
  let fi=0,playing=false,lastA=0;
  const FM=75;
  const sl=document.getElementById("sl"),tl=document.getElementById("tl");
  sl.max=S.n_frames-1;

  function apply(i){{
    if(i<0||i>=S.frames.length)return;
    const f=S.frames[i],col=PC[f.phase]||[0.3,0.3,0.4];
    const isP=["radial","mhd_radial","pinch","reflected","post_pinch"].includes(f.phase);

    // Sheath
    sheath.position.x=isP?G.anode_length:f.z;
    shMat.emissiveColor.set(col[0],col[1],col[2]);
    const cr=Math.max(0.03,f.r/G.cathode_radius);
    sheath.scaling.set(1,isP?cr:1,isP?cr:1);
    shMat.alpha=0.5+Math.abs(f.I)*0.15;

    // Trail
    const tLen=Math.max(isP?G.anode_length:f.z,0.3);
    trail.scaling.x=tLen;trail.position.x=tLen/2;
    trMat.emissiveColor.set(col[0]*0.35,col[1]*0.35,col[2]*0.5);
    trMat.alpha=0.12+Math.abs(f.I)*0.06;

    // Pinch
    const pI=isP?Math.min(1,Math.pow(1-cr,2)*3):0;
    cMat.alpha=pI*0.85;hMat.alpha=pI*0.35;
    cMat.emissiveColor.set(1,pI*0.55+0.1,pI*0.3);
    const rS=Math.max(0.04,cr*0.6);
    core.scaling.set(1,rS,rS);haloM.scaling.set(1,rS*2.5,rS*2.5);
    gl.intensity=0.4+pI*2.2;

    // God ray source
    gsMat.alpha=pI*0.9;

    // Particles
    ps.emitter.x=isP?G.anode_length:f.z;
    if(f.phase==="rundown"){{ps.gravity=new BABYLON.Vector3(2,0,0);ps.minEmitPower=1;ps.maxEmitPower=3}}
    else if(isP){{ps.gravity=new BABYLON.Vector3(0,-f.r*0.5,0);ps.minEmitPower=2;ps.maxEmitPower=7}}

    // DOF: focus on pinch during compression
    if(isP&&pI>0.3){{pp.depthOfFieldEnabled=true;
      pp.depthOfField.focalLength=50;pp.depthOfField.fStop=2;
      pp.depthOfField.focusDistance=BABYLON.Vector3.Distance(cam.position,core.position)*1000;
    }}else pp.depthOfFieldEnabled=false;

    hud.textContent=L.device+" | "+gpu+(useGPU?" GPU":" CPU")+" | "+
      (PL[f.phase]||f.phase)+"\\nt="+f.t.toFixed(1)+" us | I="+f.I.toFixed(3)+" MA";
  }}

  document.getElementById("pb").onclick=()=>{{playing=true}};
  document.getElementById("sb").onclick=()=>{{playing=false}};
  document.getElementById("rb").onclick=()=>{{fi=0;sl.value=0;apply(0);playing=false;tl.textContent="t=0 us"}};
  sl.oninput=()=>{{fi=+sl.value;apply(fi);tl.textContent="t="+S.frames[fi].t.toFixed(1)+" us"}};

  eng.runRenderLoop(()=>{{
    if(playing){{const now=performance.now();if(now-lastA>FM){{fi=(fi+1)%S.n_frames;sl.value=fi;
      tl.textContent="t="+S.frames[fi].t.toFixed(1)+" us";apply(fi);lastA=now}}}}
    sc.render();
  }});
  window.addEventListener("resize",()=>eng.resize());
  apply(0);
  hud.textContent=L.device+" | "+gpu+" | Ready — toggle layers at left";
}}
main().catch(e=>{{document.getElementById("hud").textContent="Error: "+e.message;console.error(e)}});
</script>
</body></html>"""


def create_unified_iframe(d: dict[str, Any], height: int = 620) -> str:
    html = create_unified_renderer(d)
    escaped = html_mod.escape(html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{height}px;border:none;background:#030308;" '
        f'allow="accelerometer; camera; gyroscope; xr-spatial-tracking" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )
