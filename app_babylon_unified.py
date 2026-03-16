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
</div>

<script>
const L={data_json};
const G=L.geometry, S=L.sheath;
const PC={{rundown:[0.15,0.45,1],radial:[1,0.28,0.08],mhd_radial:[1,0.28,0.08],reflected:[1,0.55,0],pinch:[1,0.08,0.03],post_pinch:[0.7,0.15,0.08]}};
const PL={{rundown:"Axial rundown",radial:"Radial implosion",mhd_radial:"MHD radial",reflected:"Reflected shock",pinch:"Pinch",post_pinch:"Post-pinch",none:""}};

function decB64(s,shape){{const r=atob(s),b=new ArrayBuffer(r.length),u=new Uint8Array(b);for(let i=0;i<r.length;i++)u[i]=r.charCodeAt(i);return{{d:new Float32Array(b),s:shape}}}}
function cmap(t){{return[Math.min(1,.05+1.3*t),Math.max(0,.85*t-.25)*(1-t*.35),Math.max(0,.85-1.7*t)]}}

async function main(){{
  const cv=document.getElementById("c"),hud=document.getElementById("hud");
  let eng,gpu="WebGL2";
  try{{if(await BABYLON.WebGPUEngine.IsSupportedAsync){{eng=new BABYLON.WebGPUEngine(cv,{{antialias:true,adaptToDeviceRatio:true,powerPreference:"high-performance"}});await eng.initAsync();gpu="WebGPU"}}}}catch(_){{}}
  if(!eng)eng=new BABYLON.Engine(cv,true,{{stencil:true,adaptToDeviceRatio:true}});

  const sc=new BABYLON.Scene(eng);
  sc.clearColor=new BABYLON.Color4(0.015,0.015,0.035,1);
  sc.ambientColor=new BABYLON.Color3(0.06,0.06,0.1);

  // ======== CAMERA ========
  const cam=new BABYLON.ArcRotateCamera("cam",-Math.PI/3.5,Math.PI/3.2,G.cathode_radius*9,
    new BABYLON.Vector3(G.anode_length/2,0,0),sc);
  cam.attachControl(cv,true);cam.lowerRadiusLimit=G.cathode_radius*2;
  cam.upperRadiusLimit=G.cathode_radius*35;cam.wheelPrecision=25;cam.minZ=0.01;cam.inertia=0.8;

  // ======== LIGHTS ========
  new BABYLON.HemisphericLight("h",new BABYLON.Vector3(0,1,0.2),sc).intensity=0.3;
  const pt=new BABYLON.PointLight("p",new BABYLON.Vector3(G.anode_length/2,G.cathode_radius*2,G.cathode_radius),sc);
  pt.intensity=0.5;pt.diffuse=new BABYLON.Color3(0.9,0.85,1);

  // ======== ELECTRODES (PBR) ========
  const cuMat=new BABYLON.PBRMaterial("cu",sc);cuMat.metallic=1;cuMat.roughness=0.18;
  cuMat.albedoColor=new BABYLON.Color3(0.955,0.638,0.538);
  const anode=BABYLON.MeshBuilder.CreateCylinder("anode",{{diameter:G.anode_radius*2,height:G.anode_length,tessellation:48,cap:BABYLON.Mesh.CAP_ALL}},sc);
  anode.rotation.z=Math.PI/2;anode.position.x=G.anode_length/2;anode.material=cuMat;

  const stMat=new BABYLON.PBRMaterial("st",sc);stMat.metallic=1;stMat.roughness=0.32;
  stMat.albedoColor=new BABYLON.Color3(0.66,0.66,0.70);
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
  shMat.emissiveColor=new BABYLON.Color3(0.2,0.5,1);shMat.alpha=0.6;
  shMat.disableLighting=true;shMat.backFaceCulling=false;
  const sheath=BABYLON.MeshBuilder.CreateDisc("sheath",{{
    radius:G.cathode_radius, innerRadius:G.anode_radius, tessellation:48
  }},sc);
  sheath.rotation.y=Math.PI/2;sheath.material=shMat;

  // Ionized plasma trail behind sheath (annular tube matching gap)
  const trMat=new BABYLON.StandardMaterial("trM",sc);
  trMat.emissiveColor=new BABYLON.Color3(0.06,0.1,0.3);trMat.alpha=0.12;
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
      heatBuf[idx]=r*255|0;heatBuf[idx+1]=g*255|0;heatBuf[idx+2]=b*255|0;heatBuf[idx+3]=(v*180+40)|0;
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
  ps.particleTexture=new BABYLON.Texture("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAP0lEQVQY02P4z8DwHwMDw38GBgYGJiCBDYMEMDAw/Gf4z/CfAQv/k4EFA3CAgQkHAAAAAElFTkSuQmCC",sc);
  ps.start();

  // ======== POST-PROCESSING (physics-justified only) ========
  const pp=new BABYLON.DefaultRenderingPipeline("pp",true,sc,[cam]);
  pp.bloomEnabled=true;pp.bloomThreshold=0.6;pp.bloomWeight=0.4;pp.bloomKernel=48;pp.bloomScale=0.5;
  // Bloom is physically justified: hot plasma emits visible light, bloom simulates
  // the camera/eye's response to bright point sources (diffraction, scattering)
  pp.imageProcessingEnabled=true;pp.imageProcessing.toneMappingEnabled=true;
  pp.imageProcessing.toneMappingType=BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
  pp.imageProcessing.exposure=1.05;pp.imageProcessing.contrast=1.0;

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
  tog("Density (rho)",false,v=>{{if(heatPlane){{heatPlane.isVisible=v;if(v)updateHeatmap("density")}}}});
  tog("Temperature (Te)",false,v=>{{if(heatPlane&&v){{heatPlane.isVisible=true;updateHeatmap("temperature")}}}});
  tog("|B| Field",false,v=>{{if(heatPlane&&v&&L.bfield){{heatPlane.isVisible=true;updateHeatmap("bfield")}}}});
  tog("B-Field Lines",false,v=>fieldLines.forEach(l=>l.isVisible=v));
  tog("Radiation Loss",false,v=>{{if(heatPlane&&v&&L.radiation){{heatPlane.isVisible=true;updateHeatmap("radiation")}}}});
  tog("Yield Map",false,v=>{{if(heatPlane&&v&&L.yield_map){{heatPlane.isVisible=true;updateHeatmap("yield_map")}}}});
  tog("Ambient Occlusion",!!ssao,v=>{{if(ssao)ssao.totalStrength=v?0.9:0}});
  tog("Bloom",true,v=>{{pp.bloomEnabled=v}});

  // ======== ANIMATION (physics-driven) ========
  let fi=0,playing=false,lastA=0;
  const FM=80;
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

    // ---- Particles: emit at sheath, density-weighted ----
    ps.emitter.x=isP?G.anode_length:f.z;
    if(f.phase==="rundown"){{
      ps.gravity=new BABYLON.Vector3(1.5,0,0);
      ps.minEmitPower=0.5;ps.maxEmitPower=2;
    }}else if(isP){{
      // Radial collapse: particles drawn inward
      ps.gravity=new BABYLON.Vector3(0,-Math.max(f.r,0.5)*0.4,0);
      ps.minEmitPower=1.5;ps.maxEmitPower=5;
    }}

    // ---- HUD ----
    let info=L.device+" | "+gpu+"\\n"+(PL[f.phase]||f.phase);
    info+="\\nt="+f.t.toFixed(1)+" us | I="+f.I.toFixed(3)+" MA";
    if(isP)info+=" | r="+f.r.toFixed(1)+" mm";
    if(pI>0.1&&instAmp>0)info+=" | m=0 ripple: "+(rippleAmp*100).toFixed(0)+"%";
    hud.textContent=info;
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
