# Google Stitch Prompt — DPF-Unified Simulation Dashboard

## Copy this entire prompt into Stitch:

---

Design a 5-screen scientific simulation dashboard web application called "DPF-Unified" for Dense Plasma Focus physics simulations. A Dense Plasma Focus (DPF) is a pulsed-power device that pinches plasma to extreme temperatures using magnetic fields — used for fusion research, neutron generation, and X-ray sources.

**Dual-audience design:** The app serves PhD plasma physicists AND first-year undergraduates. The mechanism: show data first (numbers, charts, 3D), with plain-English explanations on hover/scroll. Include a "Student Mode / Expert Mode" toggle in the header that controls: (1) tooltip verbosity — student mode shows explanations on every metric, expert mode hides them; (2) narrative detail — student mode shows all explanations, expert mode collapses them to headers only; (3) equation display — student mode shows plain-English THEN equation, expert mode shows equation only.

**Design language:** Clean, modern, dark theme (dark navy #0a0e1a background), with accent colors for physics phases. Use Inter or IBM Plex Sans font. Cards with subtle borders (1px #1f2937), not heavy shadows. Scientific precision in the aesthetic — this is a research tool, not a toy.

**Typography hierarchy:**
- H1: 28px bold (screen titles)
- H2: 22px semibold (section headers)
- Body: 16px regular (narrative text)
- Metric values: 32px tabular-nums (the big numbers)
- Labels: 12px uppercase, letter-spacing 0.05em
- Code/equations: IBM Plex Mono 14px

**Color palette (Okabe-Ito colorblind-safe, verified):**
- Background: #0a0e1a (dark navy)
- Cards: #111827 (dark gray)
- Accent primary: #3b82f6 (blue — rundown phase)
- Accent warm: #f59e0b (amber — radial compression phase)
- Accent hot: #ef4444 (red — pinch phase)
- Accent post: #a855f7 (purple — post-pinch expansion)
- Success: #10b981 (green — within experimental tolerance)
- Warning: #f59e0b (amber — close to tolerance)
- Error: #ef4444 (red — outside tolerance or simulation failure)
- Text primary: #f9fafb (near-white)
- Text secondary: #9ca3af (gray)
- Note: Always pair phase colors with text labels ("Rundown", "Radial", etc.) for colorblind accessibility. Phase bands on charts use 15% opacity fill + text label at top.

---

### Screen 1: CONFIGURE (Landing Page)

```
+------------------------------------------------------------------+
|  [DPF-Unified Logo]           [Documentation] [GitHub] [About]   |
+------------------------------------------------------------------+
|                                                                    |
|  WELCOME BANNER (dismissable after first visit)                   |
|  "Simulate a Dense Plasma Focus in 3 steps:                      |
|   1. Pick a device  2. Click Run  3. Watch the physics"          |
|                                                                    |
+-------------------+----------------------------------------------+
| CONFIGURATION     |  DEVICE PREVIEW                              |
| (left, 30%)       |  (right, 70%)                                |
|                    |                                              |
| [Device Preset v]  |  +------------------------------------------+
|  PF-1000 (1 MJ)   |  | DEVICE CARD                              |
|  UNU-ICTP (3 kJ)  |  |                                          |
|  POSEIDON (60 kV)  |  | [Schematic diagram of selected device]   |
|  Custom...         |  |                                          |
|                    |  | Key Parameters:                          |
| Backend Level:     |  | Capacitance: 1.332 mF                   |
| [===========] 4    |  | Voltage: 27 kV                           |
|  1=Fast  9=Best    |  | Stored Energy: 486 kJ                    |
|                    |  | Anode Length: 600 mm                     |
| Grid Resolution:   |  | Fill Gas: Deuterium at 3.5 Torr          |
| [Coarse|Med|Fine]  |  |                                          |
|                    |  | "This device at IPPLM Warsaw is one of   |
| ---- Advanced ---- |  |  the largest plasma focus machines in     |
| (collapsible)      |  |  the world, storing enough energy to     |
|  fc: [====] 0.70   |  |  heat a house for 30 seconds."           |
|  fm: [====] 0.08   |  +------------------------------------------+
|  V0: [====] 27 kV  |                                              |
|  Fill P: [=] 3.5   |  [  >>> RUN SIMULATION >>>  ]               |
|  Sim Time: 10 us   |  (large, prominent, blue accent button)     |
|                    |                                              |
| Physics Toggles:   |  Estimated time: ~15 seconds                |
|  [x] Radiation     |  Backend: Metal GPU (Apple Silicon)         |
|  [x] Conduction    |                                              |
|  [ ] Anomalous eta |                                              |
|  [ ] Hall MHD      |                                              |
+-------------------+----------------------------------------------+
```

Key design notes:
- Device preset dropdown is the FIRST thing. One click to select.
- Backend level is a single slider (1-9) with human labels, not dropdown jargon
- Advanced parameters are hidden by default (accordion)
- The device preview card gives context — what IS this machine?
- Plain-English description alongside every technical parameter
- Run button is the largest element on the page

---

### Screen 2: SIMULATION RUNNING (Progress View)

```
+------------------------------------------------------------------+
|  [DPF-Unified]    PF-1000 @ 27 kV    [Cancel] [Settings]        |
+------------------------------------------------------------------+
|                                                                    |
|  PHASE PROGRESS BAR (full width, color-coded)                    |
|  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░]                     |
|   Rundown ──────> Radial ──> Pinch    Post-pinch                 |
|              ^                                                    |
|         YOU ARE HERE: Radial compression at t = 4.7 us           |
|         Sheath velocity: 141 km/s inward                         |
|         Current: 1.78 MA (approaching peak)                      |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  LIVE 3D PREVIEW (Babylon.js, 50vh)                              |
|  +--------------------------------------------------------------+|
|  |                                                                ||
|  |     [3D visualization of plasma sheath                        ||
|  |      compressing inward, with B-field lines                   ||
|  |      and current flow arrows visible]                         ||
|  |                                                                ||
|  |  Phase: RADIAL COMPRESSION                                   ||
|  |  "The magnetic field is squeezing the plasma                  ||
|  |   inward like a hydraulic press made of magnetism"            ||
|  |                                                                ||
|  +--------------------------------------------------------------+|
|                                                                    |
|  LIVE CURRENT WAVEFORM (mini chart, updating in real-time)       |
|  I(t) [===========*                          ] 1.78 MA           |
|                                                                    |
+------------------------------------------------------------------+
```

Key design notes:
- Phase progress bar is THE hero element — shows exactly where in the discharge
- Color transitions: blue (rundown) → orange (radial) → red (pinch) → purple (post-pinch)
- Plain-English description of what's happening RIGHT NOW
- 3D preview shows the current state, not a static image
- Mini waveform chart gives the researcher a data heartbeat
- No spinner. No percentage. The user knows WHERE they are.

---

### Screen 3: 3D + NARRATIVE HERO VIEW (Main Results)

```
+------------------------------------------------------------------+
|  [DPF-Unified]    PF-1000 Results    [Export v] [New Run]        |
+------------------------------------------------------------------+
|                                                                    |
|  METRICS CARDS (horizontal row, 4 primary metrics)               |
|  +----------------+ +----------------+ +----------------+ +----------------+
|  | Peak Current   | | Peak Time      | | Current Dip    | | Neutron Yield  |
|  | (I_peak)       | | (t_peak)       | | (I_dip/I_peak) | | (Y_n)          |
|  | 1.862 MA       | | 5.2 us         | | 48%            | | 1.3 x 10^11    |
|  | Experiment:    | | Experiment:    | | Experiment:    | | Experiment:    |
|  | 1.87 MA        | | 5.0 us         | | 60%            | | ~10^11         |
|  | Error: 0.4%    | | Error: 4.0%    | | Error: 20%     | | Error: 30%     |
|  | [green ██████] | | [green ██████] | | [amber ████░░] | | [amber ████░░] |
|  +----------------+ +----------------+ +----------------+ +----------------+
|  Each card: human name on top, symbol in parens, value large (32px),        |
|  experimental comparison below, colored accuracy bar (green <5%, amber      |
|  5-20%, red >20%). Hover shows tooltip with source reference.               |     |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  3D BABYLON.JS RENDERER (55vh, full width)                       |
|  +--------------------------------------------------------------+|
|  |                                                                ||
|  |  [Full interactive 3D scene]                                  ||
|  |  - Electrodes, insulator, plasma sheath                      ||
|  |  - B-field lines, current flow                               ||
|  |  - Phase animation with playback controls                    ||
|  |  - Layer toggles: Plasma | Fields | Particles | Labels       ||
|  |                                                                ||
|  |  [Play/Pause] [<<] [>>] [1x|2x|4x] [Screenshot] [Fullscreen]||
|  +--------------------------------------------------------------+|
|                                                                    |
|  NARRATIVE (scrollable, directly below 3D, same width)           |
|  +--------------------------------------------------------------+|
|  | ## What Just Happened                                         ||
|  |                                                                ||
|  | Your PF-1000 simulation completed 5 phases in 10 us:         ||
|  |                                                                ||
|  | ### 1. Rundown Phase (0 - 4.1 us)                            ||
|  | The 486 kJ capacitor bank discharged through deuterium gas,   ||
|  | creating a current sheath that swept down the anode at        ||
|  | 141 km/s — about 400x the speed of sound in air.             ||
|  |                                                                ||
|  | The governing equation:                                       ||
|  |   F = J x B = (mu_0 * I^2) / (4 * pi * r)                  ||
|  |                                                                ||
|  | With I = 1.78 MA and r = 80 mm, this force was               ||
|  | 6.3 x 10^6 N/m — enough to accelerate the gas to the        ||
|  | observed velocity in the available 600 mm anode length.       ||
|  |                                                                ||
|  | ### 2. Radial Compression (4.1 - 5.2 us) ...                ||
|  | ...                                                           ||
|  +--------------------------------------------------------------+|
|                                                                    |
+------------------------------------------------------------------+
|  TAB BAR (below narrative, 4 tabs for detailed analysis)         |
|  [ Waveforms | Energy & Physics | Fields & Phase | Compare ]     |
+------------------------------------------------------------------+
```

Key design notes:
- Metrics cards at top with experimental comparison (green/amber/red bars)
- 3D renderer is THE centerpiece — 55% of viewport height
- Narrative is DIRECTLY BELOW the 3D, not in a separate tab
- Narrative uses actual simulation values, not generic text
- Equations are shown WITH plain-English interpretation
- Detailed analysis tabs are BELOW the narrative, for deep dives
- Export dropdown in header: CSV, SVG charts, 3D screenshot, narrative PDF

---

### Screen 4: WAVEFORM ANALYSIS (Tab Detail)

```
+------------------------------------------------------------------+
|  BACK TO 3D VIEW                                                  |
+------------------------------------------------------------------+
|                                                                    |
|  CURRENT WAVEFORM (large, 50vh)                                  |
|  +--------------------------------------------------------------+|
|  |  2.0 MA ┤                                                    ||
|  |         |          ****                                       ||
|  |  1.5    |        **    ***    Simulation                     ||
|  |         |      **         *** ─────────                      ||
|  |  1.0    |    **              ***                              ||
|  |         |   *                   ****     Experimental        ||
|  |  0.5    |  *                        **** ─ ─ ─ ─ ─          ||
|  |         | *                                                   ||
|  |  0.0    +────────────────────────────────────────             ||
|  |         0    2    4    6    8   10   12  t (us)               ||
|  |                                                                ||
|  |  [Phase: Rundown (blue) | Radial (orange) | Pinch (red)]     ||
|  |  I_peak marker: star at 1.862 MA / 5.2 us                   ||
|  |  Current dip: 48% at 5.8 us                                  ||
|  +--------------------------------------------------------------+|
|                                                                    |
|  NARRATIVE PANEL (below chart)                                   |
|  "The current rises to 1.862 MA at 5.2 us — within 0.4% of     |
|   the experimental value (1.87 MA, Gribkov 2007). The dip at    |
|   5.8 us indicates plasma compression (pinch). Our simulation   |
|   shows 48% dip vs 60% observed — the difference is likely      |
|   from our simplified post-pinch expansion model."               |
|                                                                    |
|  SECONDARY CHARTS (2-column grid below)                          |
|  +----------------------------+  +----------------------------+  |
|  | dI/dt Derivative           |  | Voltage V(t)               |  |
|  | [chart with pinch spike]   |  | [chart with crowbar line]  |  |
|  +----------------------------+  +----------------------------+  |
|  +----------------------------+  +----------------------------+  |
|  | Energy Balance              |  | Residual (Sim - Exp)      |  |
|  | [stacked area: KE+ME+IE]  |  | [error chart with bounds]  |  |
|  +----------------------------+  +----------------------------+  |
|                                                                    |
|  [Export All Charts as SVG] [Export Data as CSV]                  |
+------------------------------------------------------------------+
```

Key design notes:
- Primary waveform is LARGE (50vh) with experimental overlay
- Phase coloring on the background (blue/orange/red bands)
- Narrative explains the chart IN CONTEXT — what does the dip mean?
- Secondary charts in 2x2 grid below
- Every chart has a narrative panel explaining what to look for
- Export buttons at the bottom for both vector graphics and data

---

### Screen 5: EXPORT & SHARE

```
+------------------------------------------------------------------+
|  [DPF-Unified]    Export & Share    [Back to Results]             |
+------------------------------------------------------------------+
|                                                                    |
|  EXPORT CENTER                                                    |
|  +--------------------------------------------------------------+|
|  |                                                                ||
|  |  What would you like to export?                               ||
|  |                                                                ||
|  |  +------------------+  +------------------+                   ||
|  |  | SIMULATION DATA  |  | CHARTS & FIGURES |                   ||
|  |  |                  |  |                  |                   ||
|  |  | [icon] CSV       |  | [icon] All SVG   |                   ||
|  |  | Full waveform    |  | Publication-ready |                   ||
|  |  | I, V, Te, Lp, Yn|  | vector graphics   |                   ||
|  |  |                  |  |                  |                   ||
|  |  | [Download CSV]   |  | [Download SVGs]  |                   ||
|  |  +------------------+  +------------------+                   ||
|  |                                                                ||
|  |  +------------------+  +------------------+                   ||
|  |  | 3D VISUALIZATION |  | PHYSICS REPORT   |                   ||
|  |  |                  |  |                  |                   ||
|  |  | [icon] PNG/MP4   |  | [icon] PDF/MD    |                   ||
|  |  | Screenshot or    |  | Full narrative    |                   ||
|  |  | animated GIF     |  | with equations    |                   ||
|  |  |                  |  |                  |                   ||
|  |  | [Capture Now]    |  | [Download PDF]   |                   ||
|  |  +------------------+  +------------------+                   ||
|  |                                                                ||
|  |  +------------------+  +------------------+                   ||
|  |  | CONFIGURATION    |  | SHARE LINK       |                   ||
|  |  |                  |  |                  |                   ||
|  |  | [icon] JSON      |  | [icon] URL       |                   ||
|  |  | Reproducible     |  | Anyone with this  |                   ||
|  |  | simulation config|  | link sees your    |                   ||
|  |  |                  |  | exact setup       |                   ||
|  |  | [Download JSON]  |  | [Copy Link]      |                   ||
|  |  +------------------+  +------------------+                   ||
|  |                                                                ||
|  +--------------------------------------------------------------+|
|                                                                    |
|  PREVIEW (shows what will be exported, live)                     |
|  +--------------------------------------------------------------+|
|  | [Preview of selected export — chart, narrative, or config]    ||
|  +--------------------------------------------------------------+|
|                                                                    |
+------------------------------------------------------------------+
```

Key design notes:
- 6 export options in a 2x3 card grid
- Each card explains what you get in plain English
- Live preview of the selected export before downloading
- Share link encodes the full configuration in the URL
- PDF report includes the narrative with equations rendered

---

**Global Design Principles:**
1. Every number has context (comparison to experiment or physical analogy)
2. Every chart has a narrative panel explaining what the user is seeing
3. The 3D visualization is never more than one click away
4. Dark theme with phase-coded accent colors throughout
5. Mobile responsive: on phones, 3D renderer goes full-width at 40vh, narrative collapses to accordion sections, tabs become a horizontal scroll strip
6. Accessibility: WCAG AA contrast, Okabe-Ito colorblind-safe, keyboard nav (Space=play/pause, arrows=step time, Escape=back, 1-5=switch tabs)
7. No jargon without explanation. "I_peak" always appears as "Peak Current (I_peak)"

**Dynamic Content Spec (critical for Stitch):**
- The narrative text is GENERATED AT RUNTIME from simulation data, not static. Design it as a scrollable content area that will be populated with Markdown including KaTeX math equations.
- Charts are INTERACTIVE: hover shows crosshair + (x,y) tooltip, click-drag to zoom, double-click to reset. Phase background bands are semi-transparent overlays with text labels.
- The phase progress bar segments are proportional to SIMULATION TIME (microseconds), not wall-clock computation time.
- Screen 2 also shows "Estimated time remaining: ~8 seconds" below the phase indicator.
- The narrative on Screen 3 has a "Jump to Phase" mini-nav (5 pill buttons: Rundown | Radial | Pinch | Post-pinch | Summary) for scrolling to the relevant section.

**Error & Loading States:**
- Simulation failure: red banner at top of Screen 2: "Simulation stopped at t=X.X us: [reason]. [Retry] [Adjust Parameters]"
- WebSocket disconnect: amber banner: "Connection lost. Reconnecting..." with spinner
- Backend unavailable: gray overlay on Run button: "GPU backend not available. Switch to CPU?" with link to Backend Guide
- No experimental data: metric card comparison row says "No published data available" instead of leaving blank

**Screen 5 Addition:** Add a 7th card at the top of the export grid: "DOWNLOAD ALL — Get everything as a ZIP file (CSV + SVG + PDF + JSON + Screenshot)" — the one-click option for researchers running parameter sweeps.

**Physics toggle tooltips (Screen 1):**
- Radiation: "Includes energy loss from X-ray emission — important for high-current devices"
- Conduction: "Heat flow along magnetic field lines — affects temperature distribution"
- Anomalous resistivity: "Turbulent plasma resistance at the pinch — improves post-pinch accuracy"
- Hall MHD: "Magnetic field decoupling from plasma flow — relevant at small scales near pinch"

