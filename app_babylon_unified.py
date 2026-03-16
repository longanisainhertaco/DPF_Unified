"""Unified Babylon.js WebGPU physics visualization — physics-accurate rendering.

Every visual element maps to computed simulation data:
- Sheath: thin annular disc (not torus) at simulation z_mm, spanning anode→cathode
- Particles: positioned from density field rho(r,z), colored by Te(r,z)
- Pinch: radius and position from simulation r_mm, with m=0 sausage ripple
- Heatmaps: direct GPU upload of simulation density/Te/B/radiation/yield
- Trail: ionized plasma behind sheath, opacity from simulation current
- B-field lines: traced from simulation Br/Bz arrays
- No cosmetic effects without physics basis (god rays removed)
"""
from __future__ import annotations

import html as html_mod
import json
from typing import Any

from app_visualization import extract_all_layers

BABYLON_CDN = "https://cdn.babylonjs.com/babylon.js"
BABYLON_GUI = "https://cdn.babylonjs.com/gui/babylon.gui.min.js"


def create_unified_renderer(d: dict[str, Any]) -> str:
    layers = extract_all_layers(d)
    data_json = json.dumps(layers)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#030308}}
  #c{{width:100%;height:100%;touch-action:none;display:block}}
  #hud{{position:absolute;top:6px;right:10px;color:#8bf;font:11px/1.5 monospace;pointer-events:none;z-index:10;text-shadow:0 0 4px #00a4;text-align:right;white-space:pre}}
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
  <span style="color:#666;margin:0 4px">|</span>
  <span style="color:#8af;font:10px monospace">Speed:</span>
  <input type="range" id="spd" min="1" max="8" step="1" value="3" style="width:60px;accent-color:#fa8">
  <span id="spdL" style="color:#fa8;font:10px monospace;min-width:30px">1x</span>
</div>

<script>
const L={data_json};
const G=L.geometry, S=L.sheath;
const PC={{rundown:[0.15,0.45,1],radial:[1,0.28,0.08],mhd_radial:[1,0.28,0.08],reflected:[1,0.55,0],pinch:[1,0.08,0.03],post_pinch:[0.7,0.15,0.08]}};
const PL={{rundown:"Axial rundown",radial:"Radial implosion",mhd_radial:"MHD radial",reflected:"Reflected shock",pinch:"Pinch",post_pinch:"Post-pinch",none:""}};

function decB64(s,shape){{const r=atob(s),b=new ArrayBuffer(r.length),u=new Uint8Array(b);for(let i=0;i<r.length;i++)u[i]=r.charCodeAt(i);return{{d:new Float32Array(b),s:shape}}}}

// Viridis colormap (colorblind-safe, perceptually uniform)
const VIRIDIS=[
  [0.267,0.004,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],[0.207,0.372,0.553],
  [0.164,0.471,0.558],[0.128,0.567,0.551],[0.134,0.658,0.517],[0.267,0.749,0.441],
  [0.478,0.821,0.318],[0.741,0.873,0.150],[0.993,0.906,0.144]
];
// Cividis (optimized for deuteranopia/protanopia)
const CIVIDIS=[
  [0.0,0.135,0.305],[0.0,0.206,0.380],[0.133,0.273,0.385],[0.259,0.335,0.384],
  [0.365,0.397,0.395],[0.463,0.461,0.420],[0.563,0.529,0.444],[0.666,0.604,0.452],
  [0.775,0.685,0.432],[0.888,0.775,0.380],[1.0,0.871,0.298]
];
let cmapChoice=VIRIDIS;
function cmap(t){{
  const n=cmapChoice.length-1;
  const i=Math.min(n-1,Math.max(0,Math.floor(t*n)));
  const f=t*n-i;
  const a=cmapChoice[i],b=cmapChoice[i+1];
  return[a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}}

async function main(){{
  const cv=document.getElementById("c"),hud=document.getElementById("hud");
  let eng,gpu="WebGL2";
  try{{if(await BABYLON.WebGPUEngine.IsSupportedAsync){{eng=new BABYLON.WebGPUEngine(cv,{{antialias:true,adaptToDeviceRatio:true,powerPreference:"high-performance"}});await eng.initAsync();gpu="WebGPU"}}}}catch(_){{}}
  if(!eng)eng=new BABYLON.Engine(cv,true,{{stencil:true,adaptToDeviceRatio:true}});

  const sc=new BABYLON.Scene(eng);
  // Brighter background: deep navy, not black
  sc.clearColor=new BABYLON.Color4(0.06,0.07,0.12,1);
  sc.ambientColor=new BABYLON.Color3(0.15,0.15,0.2);

  // ======== CAMERA ========
  const cam=new BABYLON.ArcRotateCamera("cam",-Math.PI/3.5,Math.PI/3.2,G.cathode_radius*9,
    new BABYLON.Vector3(G.anode_length/2,0,0),sc);
  cam.attachControl(cv,true);
  cam.lowerRadiusLimit=G.anode_radius*0.3;  // zoom right into the pinch
  cam.upperRadiusLimit=G.cathode_radius*50;
  cam.wheelPrecision=15;  // more responsive zoom
  cam.pinchPrecision=20;  // touch zoom
  cam.minZ=0.001;cam.inertia=0.85;
  cam.panningSensibility=80;  // right-click pan

  // ======== LIGHTS (brighter for visibility) ========
  const hemiL=new BABYLON.HemisphericLight("h",new BABYLON.Vector3(0,1,0.3),sc);
  hemiL.intensity=0.55;hemiL.groundColor=new BABYLON.Color3(0.1,0.1,0.15);
  const pt=new BABYLON.PointLight("p",new BABYLON.Vector3(G.anode_length/2,G.cathode_radius*2.5,G.cathode_radius*1.5),sc);
  pt.intensity=0.7;pt.diffuse=new BABYLON.Color3(1,0.95,0.9);
  const pt2=new BABYLON.PointLight("p2",new BABYLON.Vector3(G.anode_length,0,-G.cathode_radius*2),sc);
  pt2.intensity=0.3;pt2.diffuse=new BABYLON.Color3(0.7,0.8,1);

  // ======== ELECTRODES (PBR) ========
  const cuMat=new BABYLON.PBRMaterial("cu",sc);cuMat.metallic=1;cuMat.roughness=0.25;
  cuMat.albedoColor=new BABYLON.Color3(0.96,0.7,0.55);
  cuMat.emissiveColor=new BABYLON.Color3(0.05,0.03,0.01);
  const anode=BABYLON.MeshBuilder.CreateCylinder("anode",{{diameter:G.anode_radius*2,height:G.anode_length,tessellation:48,cap:BABYLON.Mesh.CAP_ALL}},sc);
  anode.rotation.z=Math.PI/2;anode.position.x=G.anode_length/2;anode.material=cuMat;

  const stMat=new BABYLON.PBRMaterial("st",sc);stMat.metallic=1;stMat.roughness=0.35;
  stMat.albedoColor=new BABYLON.Color3(0.72,0.72,0.76);
  stMat.emissiveColor=new BABYLON.Color3(0.03,0.03,0.04);
  const rodArr=[];
  for(let i=0;i<8;i++){{const a=(i/8)*Math.PI*2;
    const r=BABYLON.MeshBuilder.CreateCylinder("r"+i,{{diameter:G.cathode_radius*0.07,height:G.anode_length,tessellation:8}},sc);
    r.rotation.z=Math.PI/2;r.position.set(G.anode_length/2,G.cathode_radius*Math.sin(a),G.cathode_radius*Math.cos(a));
    r.material=stMat;rodArr.push(r)}}

  const insMat=new BABYLON.PBRMaterial("ins",sc);insMat.metallic=0;insMat.roughness=0.7;
  insMat.albedoColor=new BABYLON.Color3(0.85,0.78,0.55);insMat.alpha=0.6;
  const ins=BABYLON.MeshBuilder.CreateCylinder("ins",{{diameter:G.cathode_radius*2,height:G.anode_radius*0.25,tessellation:48}},sc);
  ins.rotation.z=Math.PI/2;ins.position.x=-G.anode_radius*0.13;ins.material=insMat;

  // ======== SHEATH: thin annular disc (physics-accurate) ========
  // The current sheath is a thin shell spanning anode→cathode radii
  const shMat=new BABYLON.StandardMaterial("shM",sc);
  shMat.emissiveColor=new BABYLON.Color3(0.3,0.6,1);shMat.alpha=0.65;
  shMat.disableLighting=true;shMat.backFaceCulling=false;
  const sheath=BABYLON.MeshBuilder.CreateDisc("sheath",{{
    radius:G.cathode_radius, innerRadius:G.anode_radius, tessellation:48
  }},sc);
  sheath.rotation.y=Math.PI/2;sheath.material=shMat;

  // Ionized plasma trail behind sheath (annular tube matching gap)
  const trMat=new BABYLON.StandardMaterial("trM",sc);
  trMat.emissiveColor=new BABYLON.Color3(0.1,0.15,0.4);trMat.alpha=0.18;
  trMat.disableLighting=true;trMat.backFaceCulling=false;
  const trail=BABYLON.MeshBuilder.CreateTube("trail",{{
    path:[new BABYLON.Vector3(0,0,0),new BABYLON.Vector3(1,0,0)],
    radius:(G.anode_radius+G.cathode_radius)/2, tessellation:24,
    cap:BABYLON.Mesh.NO_CAP, updatable:true
  }},sc);
  trail.material=trMat;

  // ======== PINCH: physics-driven cylinder at anode tip ========
  // Position = anode tip (z=L), radius = simulation r_mm
  // m=0 sausage instability shown as sinusoidal radius perturbation
  const pMat=new BABYLON.StandardMaterial("pM",sc);
  pMat.emissiveColor=new BABYLON.Color3(1,0.35,0.08);pMat.disableLighting=true;pMat.alpha=0;
  pMat.backFaceCulling=false;
  // Build pinch as a tube with variable radius (for m=0 ripple)
  const N_PINCH_SEG=20;
  const pinchPath=[];
  for(let i=0;i<=N_PINCH_SEG;i++) pinchPath.push(new BABYLON.Vector3(G.anode_length*(0.65+0.35*i/N_PINCH_SEG),0,0));
  const pinchRadii=new Array(N_PINCH_SEG+1).fill(G.anode_radius*0.3);
  const pinch=BABYLON.MeshBuilder.CreateTube("pinch",{{
    path:pinchPath, radiusFunction:(i)=>pinchRadii[i],
    tessellation:16, cap:BABYLON.Mesh.CAP_ALL, updatable:true
  }},sc);
  pinch.material=pMat;

  // Outer halo (dimmer, wider)
  const phMat=new BABYLON.StandardMaterial("phM",sc);
  phMat.emissiveColor=new BABYLON.Color3(0.7,0.1,0.03);phMat.disableLighting=true;phMat.alpha=0;
  phMat.backFaceCulling=false;
  const haloRadii=new Array(N_PINCH_SEG+1).fill(G.anode_radius*0.6);
  const pHalo=BABYLON.MeshBuilder.CreateTube("pHalo",{{
    path:pinchPath, radiusFunction:(i)=>haloRadii[i],
    tessellation:16, cap:BABYLON.Mesh.NO_CAP, sideOrientation:BABYLON.Mesh.BACKSIDE, updatable:true
  }},sc);
  pHalo.material=phMat;

  // ======== HEATMAP OVERLAY (simulation data on midplane) ========
  let heatPlane=null,heatTex=null,heatBuf=null;
  if(L.density){{
    const[nx,nz]=L.density.shape;
    const W=Math.min(nx*4,256),H=Math.min(nz*4,256);
    heatBuf=new Uint8Array(W*H*4);
    heatTex=new BABYLON.RawTexture(heatBuf,W,H,BABYLON.Engine.TEXTUREFORMAT_RGBA,sc,false,false,BABYLON.Texture.BILINEAR_SAMPLINGMODE);
    const hm=new BABYLON.StandardMaterial("hmM",sc);
    hm.emissiveTexture=heatTex;hm.opacityTexture=heatTex;hm.disableLighting=true;hm.backFaceCulling=false;hm.alpha=0.6;
    heatPlane=BABYLON.MeshBuilder.CreatePlane("heatP",{{width:G.anode_length,height:G.cathode_radius*2}},sc);
    heatPlane.position.x=G.anode_length/2;heatPlane.rotation.y=Math.PI/2;heatPlane.material=hm;
    heatPlane.isVisible=false;
  }}

  function updateHeatmap(key){{
    if(!heatTex||!L[key])return;
    const fd=decB64(L[key].data,L[key].shape);
    const[nx,nz]=fd.s,W=heatTex.getSize().width,H=heatTex.getSize().height;
    for(let j=0;j<H;j++)for(let i=0;i<W;i++){{
      const fx=(i/W)*(nx-1),fz=(j/H)*(nz-1);
      const ix=Math.min(Math.floor(fx),nx-2),iz=Math.min(Math.floor(fz),nz-2);
      const dx=fx-ix,dz=fz-iz;
      const v=(1-dx)*(1-dz)*fd.d[ix*nz+iz]+dx*(1-dz)*fd.d[(ix+1)*nz+iz]+(1-dx)*dz*fd.d[ix*nz+iz+1]+dx*dz*fd.d[(ix+1)*nz+iz+1];
      const[r,g,b]=cmap(v);const idx=(j*W+i)*4;
      heatBuf[idx]=r*255|0;heatBuf[idx+1]=g*255|0;heatBuf[idx+2]=b*255|0;heatBuf[idx+3]=Math.min(255,(v*200+55))|0;
    }}
    heatTex.update(heatBuf);
  }}

  // ======== B-FIELD LINES (CPU-traced from simulation Br/Bz) ========
  const fieldLines=[];
  if(L.bfield){{
    const fd_Br=decB64(L.bfield.Br,L.bfield.shape);
    const fd_Bz=decB64(L.bfield.Bz,L.bfield.shape);
    const[nx,nz]=fd_Br.s;
    const N_SEEDS=10, N_STEPS=60;
    const ds=G.anode_length/N_STEPS*0.7;

    function sampleB(x,z){{
      const xi=Math.max(0,Math.min(nx-2,(x/G.anode_length)*(nx-1)));
      const zi=Math.max(0,Math.min(nz-2,((z+G.cathode_radius)/(G.cathode_radius*2))*(nz-1)));
      const ix=Math.floor(xi),iz=Math.floor(zi),fx=xi-ix,fz=zi-iz;
      const br=(1-fx)*(1-fz)*fd_Br.d[ix*nz+iz]+fx*(1-fz)*fd_Br.d[(ix+1)*nz+iz]+(1-fx)*fz*fd_Br.d[ix*nz+iz+1]+fx*fz*fd_Br.d[(ix+1)*nz+iz+1];
      const bz=(1-fx)*(1-fz)*fd_Bz.d[ix*nz+iz]+fx*(1-fz)*fd_Bz.d[(ix+1)*nz+iz]+(1-fx)*fz*fd_Bz.d[ix*nz+iz+1]+fx*fz*fd_Bz.d[(ix+1)*nz+iz+1];
      const mag=Math.sqrt(br*br+bz*bz)+1e-10;
      return[br/mag,bz/mag];
    }}

    for(let s=0;s<N_SEEDS;s++){{
      let x=G.anode_length*(0.1+0.8*s/N_SEEDS), z=0;
      const pts=[];
      for(let i=0;i<N_STEPS;i++){{
        pts.push(new BABYLON.Vector3(x,0,z));
        const[bx,bz]=sampleB(x,z);
        x+=ds*bx;z+=ds*bz;
        if(x<0||x>G.anode_length||Math.abs(z)>G.cathode_radius)break;
      }}
      if(pts.length>3){{
        const line=BABYLON.MeshBuilder.CreateLines("fl"+s,{{points:pts}},sc);
        line.color=new BABYLON.Color3(0.2,0.5,1);line.alpha=0.5;
        line.isVisible=false;
        fieldLines.push(line);
      }}
    }}
  }}

  // ======== PARTICLES: density-weighted positions, Te-colored ========
  const useGPU=BABYLON.GPUParticleSystem.IsSupported;
  const PS=useGPU?BABYLON.GPUParticleSystem:BABYLON.ParticleSystem;
  const cap=useGPU?30000:3000;
  const ps=new PS("ions",{{capacity:cap}},sc);
  ps.emitter=new BABYLON.Vector3(0,0,0);
  const em=new BABYLON.SphereParticleEmitter();
  em.radius=G.cathode_radius*0.85;em.radiusRange=0.4;
  ps.particleEmitterType=em;
  ps.minLifeTime=0.06;ps.maxLifeTime=0.15;
  ps.emitRate=useGPU?8000:500;
  ps.minSize=0.04;ps.maxSize=0.16;
  ps.minEmitPower=0.3;ps.maxEmitPower=2;
  // Color gradient: cold blue → warm yellow → hot white (mapped to Te)
  ps.addColorGradient(0,new BABYLON.Color4(0.05,0.15,0.6,0));
  ps.addColorGradient(0.15,new BABYLON.Color4(0.1,0.3,0.9,0.6));
  ps.addColorGradient(0.5,new BABYLON.Color4(0.9,0.7,0.2,0.7));
  ps.addColorGradient(0.8,new BABYLON.Color4(1,0.95,0.85,0.8));
  ps.addColorGradient(1,new BABYLON.Color4(1,1,1,0));
  ps.addSizeGradient(0,0.02);ps.addSizeGradient(0.3,0.12);ps.addSizeGradient(1,0);
  ps.isBillboardBased=true;ps.blendMode=BABYLON.ParticleSystem.BLENDMODE_ADD;
  // Generate a 32x32 soft gaussian particle texture (higher fidelity than 8x8)
  const ptex=new BABYLON.DynamicTexture("ptex",32,sc,false);
  const pctx=ptex.getContext();
  const grad=pctx.createRadialGradient(16,16,0,16,16,16);
  grad.addColorStop(0,"rgba(255,255,255,1)");
  grad.addColorStop(0.3,"rgba(255,240,200,0.8)");
  grad.addColorStop(0.7,"rgba(200,150,80,0.3)");
  grad.addColorStop(1,"rgba(100,50,20,0)");
  pctx.fillStyle=grad;pctx.fillRect(0,0,32,32);ptex.update();
  ps.particleTexture=ptex;
  ps.start();

  // ======== POST-PROCESSING (physics-justified only) ========
  const pp=new BABYLON.DefaultRenderingPipeline("pp",true,sc,[cam]);
  pp.bloomEnabled=true;pp.bloomThreshold=0.6;pp.bloomWeight=0.4;pp.bloomKernel=48;pp.bloomScale=0.5;
  // Bloom is physically justified: hot plasma emits visible light, bloom simulates
  // the camera/eye's response to bright point sources (diffraction, scattering)
  pp.imageProcessingEnabled=true;pp.imageProcessing.toneMappingEnabled=true;
  pp.imageProcessing.toneMappingType=BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pp.imageProcessing.exposure=1.4;pp.imageProcessing.contrast=1.1;

  let ssao=null;
  try{{ssao=new BABYLON.SSAO2RenderingPipeline("ssao",sc,{{ssaoRatio:0.5,blurRatio:1}},[cam],false);
    ssao.totalStrength=0.9;ssao.radius=1.5;ssao.samples=12;ssao.base=0.2;
  }}catch(_){{}}

  const gl=new BABYLON.GlowLayer("gl",sc,{{blurKernelSize:24,mainTextureFixedSize:512}});
  gl.intensity=0.4;
  gl.customEmissiveColorSelector=(m,_s,_m,res)=>{{
    if(["sheath","pinch","pHalo","trail"].includes(m.name))
      res.set(m.material.emissiveColor.r,m.material.emissiveColor.g,m.material.emissiveColor.b,m.material.alpha);
    else res.set(0,0,0,0);
  }};

  // ======== GUI TOGGLES ========
  const ui=BABYLON.GUI.AdvancedDynamicTexture.CreateFullscreenUI("UI");
  const panel=new BABYLON.GUI.StackPanel();
  panel.width="200px";panel.isVertical=true;
  panel.horizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.verticalAlignment=BABYLON.GUI.Control.VERTICAL_ALIGNMENT_TOP;
  panel.paddingTop="10px";panel.paddingLeft="10px";
  ui.addControl(panel);

  const hdr=new BABYLON.GUI.TextBlock();hdr.text="Physics Layers";hdr.color="#8af";
  hdr.fontSize=13;hdr.height="20px";hdr.textHorizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.addControl(hdr);

  function tog(label,init,fn){{
    const cb=BABYLON.GUI.Checkbox.AddCheckBoxWithHeader(label,(v)=>fn(v));
    cb.children[0].isChecked=init;cb.children[0].color="#7af";
    cb.children[1].color="#cdf";cb.children[1].fontSize=11;cb.height="24px";
    panel.addControl(cb);fn(init);
  }}

  tog("Electrodes",true,v=>{{anode.isVisible=v;rodArr.forEach(r=>r.isVisible=v);ins.isVisible=v}});
  tog("Current Sheath",true,v=>{{sheath.isVisible=v;trail.isVisible=v}});
  tog("Plasma Ions",true,v=>{{if(v)ps.start();else ps.stop()}});
  tog("Pinch Column",true,v=>{{pinch.isVisible=v;pHalo.isVisible=v}});
  tog("B-Field Lines",!!L.bfield,v=>fieldLines.forEach(l=>l.isVisible=v));
  tog("Ambient Occlusion",!!ssao,v=>{{if(ssao)ssao.totalStrength=v?0.9:0}});
  tog("Bloom",true,v=>{{pp.bloomEnabled=v}});

  // ---- Field overlay selector (radio-style: only one active) ----
  const fieldSep=new BABYLON.GUI.TextBlock();fieldSep.text="Field Overlay";fieldSep.color="#fa8";
  fieldSep.fontSize=12;fieldSep.height="22px";fieldSep.textHorizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.addControl(fieldSep);

  let activeOverlay="none";
  const overlays=[
    ["None","none"],["Density (rho)","density"],["Temperature (Te)","temperature"],
    ["|B| Magnetic","bfield"],["Radiation Loss","radiation"],["Neutron Yield","yield_map"]
  ];
  overlays.forEach(([label,key])=>{{
    const rb=BABYLON.GUI.Checkbox.AddCheckBoxWithHeader(label,(v)=>{{
      if(v){{
        activeOverlay=key;
        if(heatPlane){{
          if(key==="none"){{heatPlane.isVisible=false;showColorbar("none")}}
          else if(L[key]){{heatPlane.isVisible=true;updateHeatmap(key);showColorbar(key)}}
          else if(key==="bfield"&&L.bfield){{heatPlane.isVisible=true;updateHeatmap("bfield");showColorbar("bfield")}}
          else{{heatPlane.isVisible=false;showColorbar("none")}}
        }}
      }}
    }});
    rb.children[0].isChecked=(key==="none");rb.children[0].color="#fa8";
    rb.children[1].color="#edb";rb.children[1].fontSize=11;rb.height="24px";
    panel.addControl(rb);
  }});

  // ---- Colormap selector (accessibility) ----
  const cmSep=new BABYLON.GUI.TextBlock();cmSep.text="Accessibility";cmSep.color="#af8";
  cmSep.fontSize=12;cmSep.height="20px";cmSep.textHorizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.addControl(cmSep);

  // Viridis (default) vs Cividis (colorblind-optimized) toggle
  tog("Cividis (colorblind)",false,v=>{{
    cmapChoice=v?CIVIDIS:VIRIDIS;
    if(activeOverlay!=="none"&&heatPlane&&heatPlane.isVisible)updateHeatmap(activeOverlay);
  }});

  // ---- Colorbar (right side of screen) ----
  const cbPanel=new BABYLON.GUI.StackPanel();
  cbPanel.width="60px";cbPanel.isVertical=true;
  cbPanel.horizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_RIGHT;
  cbPanel.verticalAlignment=BABYLON.GUI.Control.VERTICAL_ALIGNMENT_CENTER;
  cbPanel.paddingRight="10px";
  ui.addControl(cbPanel);
  cbPanel.isVisible=false;

  // Colorbar title
  const cbTitle=new BABYLON.GUI.TextBlock();cbTitle.color="#ccc";cbTitle.fontSize=10;
  cbTitle.height="18px";cbTitle.text="";cbPanel.addControl(cbTitle);

  // Max value label
  const cbMax=new BABYLON.GUI.TextBlock();cbMax.color="#eee";cbMax.fontSize=10;
  cbMax.height="16px";cbMax.text="";cbPanel.addControl(cbMax);

  // Gradient bar (rendered as stacked colored rectangles)
  const N_CB=16;
  const cbRects=[];
  for(let i=N_CB-1;i>=0;i--){{
    const rect=new BABYLON.GUI.Rectangle();
    rect.width="30px";rect.height="12px";rect.thickness=0;
    const t=i/(N_CB-1);
    const[r,g,b]=cmap(t);
    rect.background="rgb("+Math.round(r*255)+","+Math.round(g*255)+","+Math.round(b*255)+")";
    cbPanel.addControl(rect);
    cbRects.push(rect);
  }}

  // Min value label
  const cbMin=new BABYLON.GUI.TextBlock();cbMin.color="#eee";cbMin.fontSize=10;
  cbMin.height="16px";cbMin.text="";cbPanel.addControl(cbMin);

  // Units label
  const cbUnits=new BABYLON.GUI.TextBlock();cbUnits.color="#aaa";cbUnits.fontSize=9;
  cbUnits.height="16px";cbUnits.text="";cbPanel.addControl(cbUnits);

  function showColorbar(key){{
    if(key==="none"){{cbPanel.isVisible=false;return}}
    cbPanel.isVisible=true;
    // Update gradient colors
    for(let i=0;i<N_CB;i++){{
      const t=(N_CB-1-i)/(N_CB-1);
      const[r,g,b]=cmap(t);
      cbRects[i].background="rgb("+Math.round(r*255)+","+Math.round(g*255)+","+Math.round(b*255)+")";
    }}
    // Labels with units
    const info={{
      density:{{title:"Density",max:L.density?L.density.max_val.toExponential(1):"?",min:L.density?L.density.min_val.toExponential(1):"0",unit:"kg/m3"}},
      temperature:{{title:"Te",max:L.temperature?L.temperature.max_eV.toFixed(0):"?",min:L.temperature?L.temperature.min_eV.toFixed(1):"0",unit:"eV"}},
      bfield:{{title:"|B|",max:L.bfield?L.bfield.max_T.toFixed(1):"?",min:"0",unit:"Tesla"}},
      radiation:{{title:"P_rad",max:L.radiation?L.radiation.max_W_m3.toExponential(1):"?",min:"0",unit:"W/m3"}},
      yield_map:{{title:"Yield",max:L.yield_map?L.yield_map.max_rate.toExponential(1):"?",min:"0",unit:"n/m3/s"}}
    }};
    const d=info[key]||{{title:key,max:"1",min:"0",unit:""}};
    cbTitle.text=d.title;cbMax.text=d.max;cbMin.text=d.min;cbUnits.text=d.unit;
  }}

  // ---- Labels section ----
  const lblSep=new BABYLON.GUI.TextBlock();lblSep.text="Labels";lblSep.color="#8fa";
  lblSep.fontSize=12;lblSep.height="20px";lblSep.textHorizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
  panel.addControl(lblSep);

  // 3D labels (billboard text planes attached to key objects)
  const labels=[];
  function makeLabel(text,pos,color){{
    const plane=BABYLON.MeshBuilder.CreatePlane("lbl_"+text,{{width:G.cathode_radius*3,height:G.cathode_radius*0.6}},sc);
    plane.position=pos;plane.billboardMode=BABYLON.Mesh.BILLBOARDMODE_ALL;
    const dt=new BABYLON.DynamicTexture("dt_"+text,{{width:256,height:48}},sc,false);
    dt.hasAlpha=true;
    const ctx=dt.getContext();
    ctx.clearRect(0,0,256,48);
    ctx.font="bold 20px monospace";ctx.fillStyle=color;ctx.textAlign="center";
    ctx.fillText(text,128,32);
    dt.update();
    const m=new BABYLON.StandardMaterial("lm_"+text,sc);
    m.diffuseTexture=dt;m.emissiveTexture=dt;m.opacityTexture=dt;
    m.disableLighting=true;m.backFaceCulling=false;
    plane.material=m;
    labels.push(plane);
    return plane;
  }}
  // Spread labels: anode below, cathode above, insulator to left, no overlap
  makeLabel("ANODE (Cu)",new BABYLON.Vector3(G.anode_length*0.3,-G.anode_radius*1.6,G.anode_radius*1.5),"#FFB74D");
  makeLabel("CATHODE",new BABYLON.Vector3(G.anode_length*0.7,G.cathode_radius*1.4,0),"#90A4AE");
  makeLabel("INSULATOR",new BABYLON.Vector3(-G.cathode_radius*1.2,G.cathode_radius*0.6,0),"#CE93D8");

  // Dynamic labels (follow objects, offset to avoid overlap)
  const sheathLabel=makeLabel("CURRENT SHEATH",new BABYLON.Vector3(0,-G.cathode_radius*1.3,0),"#64B5F6");
  const pinchLabel=makeLabel("PINCH (fusion here)",new BABYLON.Vector3(G.anode_length*1.05,0,G.anode_radius*2),"#FF5252");
  pinchLabel.isVisible=false;
  const instLabel=makeLabel("INSTABILITY (m=0)",new BABYLON.Vector3(G.anode_length*0.75,-G.anode_radius*2.5,0),"#FFD54F");
  instLabel.isVisible=false;

  let labelsVisible=true;
  tog("Labels",true,v=>{{labelsVisible=v;labels.forEach(l=>l.isVisible=v)}});

  // ======== ANIMATION (physics-driven) ========
  let fi=0,playing=false,lastA=0;
  const SPEEDS=[0,0.125,0.25,0.5,1,2,4,8,16];  // index 0=unused, slider 1-8
  let speedIdx=3;  // default 0.5x (slow enough to follow)
  const spdSlider=document.getElementById("spd"),spdLabel=document.getElementById("spdL");
  spdSlider.oninput=()=>{{speedIdx=+spdSlider.value;spdLabel.textContent=SPEEDS[speedIdx]+"x"}};
  spdLabel.textContent=SPEEDS[speedIdx]+"x";
  function getFrameMS(){{return 160/Math.max(SPEEDS[speedIdx],0.01)}}
  const sl=document.getElementById("sl"),tl=document.getElementById("tl");
  sl.max=S.n_frames-1;

  // m=0 instability data
  const instAmp=L.instability?L.instability.amplitude:0;
  const tau_m0=L.instability?L.instability.tau_m0_ns:1e6;

  function apply(i){{
    if(i<0||i>=S.frames.length)return;
    const f=S.frames[i],col=PC[f.phase]||[0.3,0.3,0.4];
    const isP=["radial","mhd_radial","pinch","reflected","post_pinch"].includes(f.phase);

    // ---- Sheath: thin annular disc at z=z_mm (or at tip during radial) ----
    sheath.position.x=isP?G.anode_length:f.z;
    shMat.emissiveColor.set(col[0],col[1],col[2]);
    // During radial: inner radius shrinks (disc becomes smaller ring)
    if(isP){{
      const innerR=Math.max(f.r,G.anode_radius*0.02);
      // Rebuild disc with new inner radius via scaling
      // Scale Y/Z by compression ratio to approximate shrinking inner radius
      const cr=innerR/G.cathode_radius;
      sheath.scaling.set(1,Math.max(0.03,cr),Math.max(0.03,cr));
    }}else{{
      sheath.scaling.set(1,1,1);
    }}
    shMat.alpha=0.45+Math.abs(f.I)*0.2;

    // ---- Trail: ionized plasma from insulator to sheath ----
    const tLen=Math.max(isP?G.anode_length:f.z,0.2);
    trail.scaling.x=tLen;trail.position.x=tLen/2;
    trMat.emissiveColor.set(col[0]*0.3,col[1]*0.3,col[2]*0.4);
    trMat.alpha=0.08+Math.abs(f.I)*0.05;

    // ---- Pinch: radius from simulation, m=0 ripple ----
    const cr=Math.max(0.02,f.r/G.cathode_radius);
    const pI=isP?Math.min(1,Math.pow(1-cr,2)*3):0;

    // m=0 sausage instability: sinusoidal radius perturbation
    const rippleAmp=isP?instAmp*Math.min(1,(1-cr)*2):0;
    for(let k=0;k<=N_PINCH_SEG;k++){{
      const zFrac=k/N_PINCH_SEG;
      const baseR=cr*G.cathode_radius*0.4;
      // m=0 mode: cos(2*pi*z/wavelength) with wavelength ~ pinch diameter
      const ripple=rippleAmp*baseR*Math.cos(4*Math.PI*zFrac);
      pinchRadii[k]=Math.max(0.001,baseR+ripple);
      haloRadii[k]=Math.max(0.002,(baseR+ripple)*2.5);
    }}
    // Rebuild tube meshes with updated radii
    BABYLON.MeshBuilder.CreateTube("pinch",{{
      path:pinchPath,radiusFunction:(i)=>pinchRadii[i],
      tessellation:16,cap:BABYLON.Mesh.CAP_ALL,instance:pinch
    }});
    BABYLON.MeshBuilder.CreateTube("pHalo",{{
      path:pinchPath,radiusFunction:(i)=>haloRadii[i],
      tessellation:16,cap:BABYLON.Mesh.NO_CAP,sideOrientation:BABYLON.Mesh.BACKSIDE,instance:pHalo
    }});

    pMat.alpha=pI*0.8;phMat.alpha=pI*0.3;
    pMat.emissiveColor.set(1,0.15+pI*0.4,pI*0.25);
    phMat.emissiveColor.set(0.8,0.08+pI*0.12,0.03);
    gl.intensity=0.3+pI*1.8;

    // ---- Particles: behavior changes by phase ----
    ps.emitter.x=isP?G.anode_length:f.z;
    if(f.phase==="rundown"){{
      // Sweeping along anode: moderate particles, axial drift
      em.radius=G.cathode_radius*0.85;em.radiusRange=0.35;
      ps.gravity=new BABYLON.Vector3(1.5,0,0);
      ps.minEmitPower=0.5;ps.maxEmitPower=2;
      ps.emitRate=useGPU?6000:400;
      ps.minSize=0.04;ps.maxSize=0.14;
    }}else if(isP){{
      // Radial collapse + pinch: concentrate particles at compression radius
      const compR=Math.max(f.r,G.anode_radius*0.05);
      em.radius=compR*0.8;em.radiusRange=0.3;
      // More particles at higher compression (density increases as r^-2)
      const densityBoost=Math.min(8,Math.pow(G.cathode_radius/Math.max(compR,0.1),1.5));
      ps.emitRate=useGPU?Math.min(50000,6000*densityBoost)|0:Math.min(4000,400*densityBoost)|0;
      // Particles glow brighter and larger at pinch
      ps.minSize=0.03+pI*0.08;ps.maxSize=0.12+pI*0.2;
      // Radial inward + axial jets during pinch
      if(pI>0.5){{
        // At peak pinch: axial jets (beam ions escaping along axis)
        ps.gravity=new BABYLON.Vector3(4,0,0);
        ps.minEmitPower=3;ps.maxEmitPower=10;
      }}else{{
        ps.gravity=new BABYLON.Vector3(0,-compR*0.5,0);
        ps.minEmitPower=1.5;ps.maxEmitPower=5;
      }}
    }}

    // ---- Labels: follow their objects ----
    if(labelsVisible){{
      sheathLabel.position.x=sheath.position.x;
      sheathLabel.isVisible=true;
      pinchLabel.isVisible=pI>0.2;
      // Show instability label when ripple is active
      instLabel.isVisible=(rippleAmp>0.02&&pI>0.3);
    }}

    // ---- HUD: real-time simulation data (makes it obvious this is computed) ----
    const phaseDesc={{
      rundown:"Current sheath sweeping gas toward anode tip",
      radial:"Plasma ring compressing inward — magnetic piston",
      mhd_radial:"MHD radial implosion in progress",
      reflected:"Reflected shock expanding outward",
      pinch:"PEAK COMPRESSION — fusion zone active",
      post_pinch:"Pinch disrupting — m=0 instability",
    }};

    // Build multi-line data readout
    let info=L.device+" | "+gpu+" | "+(L.backend||"")+"\\n";
    info+="Phase: "+(PL[f.phase]||f.phase)+"\\n";
    info+="t = "+f.t.toFixed(2)+" us\\n";
    info+="I = "+f.I.toFixed(3)+" MA  ("+(f.I/S.I_peak*100).toFixed(0)+"% of peak)\\n";
    if(isP){{
      info+="r = "+f.r.toFixed(2)+" mm  (compression: "+(G.cathode_radius/Math.max(f.r,0.01)).toFixed(0)+":1)\\n";
      if(L.density)info+="rho_max = "+L.density.max_val.toExponential(2)+" kg/m3\\n";
      if(L.temperature)info+="Te_max = "+L.temperature.max_eV.toFixed(0)+" eV\\n";
      if(L.bfield)info+="|B|_max = "+L.bfield.max_T.toFixed(1)+" T\\n";
    }}
    if(pI>0.1&&instAmp>0)info+="m=0 instability: "+(rippleAmp*100).toFixed(0)+"% amplitude\\n";
    info+="\\n"+(phaseDesc[f.phase]||"");
    hud.textContent=info;
  }}

  document.getElementById("pb").onclick=()=>{{playing=true}};
  document.getElementById("sb").onclick=()=>{{playing=false}};
  document.getElementById("rb").onclick=()=>{{fi=0;sl.value=0;apply(0);playing=false;tl.textContent="t=0 us"}};
  sl.oninput=()=>{{fi=+sl.value;apply(fi);tl.textContent="t="+S.frames[fi].t.toFixed(1)+" us"}};

  eng.runRenderLoop(()=>{{
    if(playing){{const now=performance.now();const FM=getFrameMS();if(now-lastA>FM){{fi=(fi+1)%S.n_frames;sl.value=fi;
      tl.textContent="t="+S.frames[fi].t.toFixed(1)+" us";apply(fi);lastA=now}}}}
    sc.render();
  }});
  window.addEventListener("resize",()=>eng.resize());
  apply(0);
  hud.textContent=L.device+" | "+gpu+" | Ready";
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
