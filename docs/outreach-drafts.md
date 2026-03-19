# DPF-Unified Outreach Drafts

## Email to AAAPT Network

**To:** AAAPT Network Coordinators
**From:** Anthony Zamora
**Subject:** Free browser-based DPF simulator built on the Lee Model -- feedback welcome

---

Dear Professors,

I am writing to share a simulation tool that may be useful to AAAPT's plasma focus training programs. DPF-Unified is a free, open-source Dense Plasma Focus simulator that runs entirely in the browser -- no installation, no MATLAB license, no spreadsheet macros.

The project was built on the foundation of the Lee Model. Your RADPF spreadsheet has been the standard training tool across 44 institutes for good reason, and DPF-Unified is designed to complement it, not replace it. Students can start with RADPF to learn the coupled equations by hand, then move to DPF-Unified for parameter sweeps, MHD extensions, and 3D visualization.

**What it does:**
- Full Lee model implementation (5-phase: axial, radial inward, reflected shock, slow compression, expanded column), validated against 6 published devices including PF-1000 and UNU-ICTP
- Extended MHD solver (Braginskii transport, anomalous resistivity, constrained transport)
- 14 device presets, including UNU-ICTP with published Lee model parameters
- Interactive 3D visualization (Babylon.js), real-time current/voltage traces
- 3 onboarding notebooks written specifically for students new to plasma focus physics

**For your students:**
The tutorial preset is based on UNU-ICTP parameters. A student can load it, run a simulation, and see the current dip from radial collapse within seconds -- then start varying voltage, fill pressure, and mass fraction to build physical intuition.

**Links:**
- Live app: https://huggingface.co/spaces/tjlonganisa/dpf-unified
- Source: https://github.com/longanisainhertaco/DPF_Unified

I would welcome any feedback from your group, and would be glad to collaborate on adapting the tool for AAAPT training workshops. If any of your students or colleagues would be willing to test it and share their experience, that would be invaluable.

With respect for your decades of work in plasma focus education,

Anthony Zamora

---

## Twitter/X Post (280 chars)

```
DPF-Unified: free browser-based Dense Plasma Focus simulator. Lee model + MHD, 14 device presets, 3D viz. No install needed. Built for students and researchers.

https://huggingface.co/spaces/tjlonganisa/dpf-unified
```
