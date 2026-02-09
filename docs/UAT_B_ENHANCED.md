# UAT-B Enhanced: Student/Intern Persona — Hyper-Critical Usability Audit

## Table of Contents

1. [Persona Backstory](#1-persona-backstory)
2. [First Impressions Checklist](#2-first-impressions-checklist)
3. [Test Scenarios](#3-test-scenarios)
   - [S01: Learnability — Zero to First Simulation](#s01-learnability--zero-to-first-simulation)
   - [S02: Discoverability — Can I Find What I Need?](#s02-discoverability--can-i-find-what-i-need)
   - [S03: Tooltip Pedagogy — Learning Physics Through the UI](#s03-tooltip-pedagogy--learning-physics-through-the-ui)
   - [S04: Error Handling — What Happens When I Make Mistakes?](#s04-error-handling--what-happens-when-i-make-mistakes)
   - [S05: Input Validation — Boundary and Edge Cases](#s05-input-validation--boundary-and-edge-cases)
   - [S06: Chat Interface — Conversational Learning](#s06-chat-interface--conversational-learning)
   - [S07: Simulation Workflow — End-to-End Completeness](#s07-simulation-workflow--end-to-end-completeness)
   - [S08: Data Visualization — Can I Read the Plots?](#s08-data-visualization--can-i-read-the-plots)
   - [S09: Performance Under Stress — Long Runs and Heavy Loads](#s09-performance-under-stress--long-runs-and-heavy-loads)
   - [S10: AI Co-Pilot — Sweep and Inverse Design](#s10-ai-co-pilot--sweep-and-inverse-design)
   - [S11: Responsive Feedback — What Is the System Doing?](#s11-responsive-feedback--what-is-the-system-doing)
   - [S12: Data Export and CLI Consistency](#s12-data-export-and-cli-consistency)
   - [S13: Onboarding and Tutorial Absence](#s13-onboarding-and-tutorial-absence)
   - [S14: Multi-Run Comparison Workflow](#s14-multi-run-comparison-workflow)
4. ["Break It" Test Suite](#4-break-it-test-suite)
5. [Competitive Comparison](#5-competitive-comparison-with-other-physics-simulation-guis)
6. [Usability Scoring Rubric](#6-usability-scoring-rubric)
7. [Accessibility Audit Checklist](#7-accessibility-audit-checklist)
8. [Appendix: Alex's Lab Notebook](#appendix-alexs-lab-notebook)

---

## 1. Persona Backstory

### Who Is Alex Chen?

**Alex Chen** is a 2nd-year M.S. student in computational physics at a mid-tier research university. Alex just joined Professor Rivera's pulsed-power plasma lab two weeks ago. Their background:

- **Education**: B.S. in Physics with a minor in Computer Science. Took one semester of plasma physics (intro level, mostly Debye shielding and single-particle orbits). Has never heard the phrase "Dense Plasma Focus" before joining the lab.
- **Programming**: Proficient in Python (NumPy, matplotlib, Jupyter). Has built web apps in React for side projects. Comfortable with the command line on macOS. Has never written C++ or Fortran.
- **Domain knowledge gaps**: Does not know what MHD stands for without looking it up. Has never seen a Riemann solver, a CFL condition, or an athinput file. Thinks "WENO5" might be a radio station. Has no intuition for what 27 kV, 1.332 mF, or 15 nH physically mean in terms of a real device.
- **Learning style**: Explores by clicking first, reading second. Opens every collapsible section. Hovers over everything. Types random queries into chat interfaces to see what happens. Gets frustrated by jargon-heavy error messages that assume you already know the domain.
- **Expectations**: Has used COMSOL briefly in undergrad (for a heat transfer homework) and remembers it having a guided workflow with wizards, geometry builders, and context-sensitive help. Expects *some* level of hand-holding for a new user.

### Why Is Alex Testing?

Professor Rivera asked Alex to evaluate the DPF Simulator GUI as part of their onboarding. The professor wants to know:

1. Can a new grad student get from "zero knowledge" to "running a meaningful simulation" within 30 minutes using only the GUI?
2. Are the tooltips and chat interface sufficient to replace a manual or tutorial?
3. What are the top 5 friction points that would cause a new user to give up or ask for help?
4. Is the software ready for use in a graduate-level lab course on pulsed-power physics?

Alex is motivated but skeptical. They have seen too many research codes with GUIs that are clearly afterthoughts.

### What Alex Hopes to Learn

By the end of this evaluation, Alex wants to be able to:
- Explain to a friend what a Dense Plasma Focus does, using concepts picked up from the GUI
- Run a PF-1000 simulation and understand why the output values are physically meaningful
- Know which parameters matter most for neutron yield
- Form an opinion on whether the WALRUS AI features are genuinely useful or just marketing

---

## 2. First Impressions Checklist

**Context**: Alex launches the app for the first time with no prior reading. Timer starts.

| ID | Observation (First 5 Minutes) | Expected | Critical? |
|----|-------------------------------|----------|-----------|
| FI-01 | Is the window title or header self-explanatory? | "DPF SIMULATOR v1.0.0" visible. Alex wonders: "What is DPF?" No subtitle or tagline explains it. | Yes |
| FI-02 | Does the layout make sense at a glance? | Left panel = parameters, center = empty scope, right = hidden sidebar. Layout follows a "configure-then-run" mental model, which is standard. | No |
| FI-03 | Is the purpose of the ARM/FIRE buttons obvious? | "ARM" is ambiguous. Does it arm a nuclear weapon? A trap? Alex will click it to find out. The metaphor (capacitor bank arming) is domain-specific. | Yes |
| FI-04 | Can Alex identify which fields to fill in? | Preset section is expanded by default. Selecting a preset fills everything. This is the critical lifeline for new users. | Yes |
| FI-05 | Is the color scheme readable? | Dark theme with cyan accents. Text is small (text-xs = 12px). Labels use uppercase + letter-spacing. Readable but dense. | No |
| FI-06 | Are units visible for every parameter? | Units appear in a small gray span to the right of each input. Some inputs (CFL, gamma, anomalous alpha) have no unit shown. Alex does not know if CFL is dimensionless. | Yes |
| FI-07 | Is there any onboarding, welcome screen, or tutorial? | No. The app opens directly to the parameter editor. Compare: COMSOL shows a "Model Wizard" on first launch. | Yes |
| FI-08 | Can Alex tell what the status indicators mean? | Green dot = Idle. The word "Idle" is shown. Clear enough. Backend badges ("Python") are visible but unexplained. | No |
| FI-09 | Is the "AI Co-Pilot" button inviting? | Button is in the top-right. Label is clear. Alex will click it. Good. | No |
| FI-10 | Is the parameter panel scrollable and navigable? | Yes, sections are collapsible. Some are collapsed by default (Grid, Solver, Radiation, Transport, Boundary, Diagnostics). Alex may not know to expand them. | No |
| FI-11 | Does the app work at different window sizes? | Does the layout break at 1280x720? Does the parameter panel get a scrollbar? Does the oscilloscope resize? | No |
| FI-12 | Are there any console errors on launch? | Open DevTools (Cmd+Opt+I). Check for React warnings, failed API calls, or WebSocket errors. | Yes |

**Time checkpoint**: After 5 minutes, Alex should have identified (a) there are presets, (b) there is an ARM/FIRE workflow, (c) there is an AI sidebar. If any of these are not obvious, the UI has a discoverability problem.

---

## 3. Test Scenarios

### S01: Learnability — Zero to First Simulation

**Goal**: Can Alex run a complete simulation using only the GUI, with no documentation, in under 10 minutes?

| Step | Action | Alex's Internal Monologue | Expected Result | Pass/Fail |
|------|--------|--------------------------|-----------------|-----------|
| 1.1 | Open the app | "Ok, DPF Simulator. What's DPF?" | App loads without errors | |
| 1.2 | Look for a "New Simulation" or "Quick Start" button | "There's no wizard... let me look at the left panel." | Alex finds the Preset section | |
| 1.3 | Click the Preset dropdown | "PF-1000, NX2... these are device names? Let me pick PF-1000, it sounds important." | Parameters auto-fill | |
| 1.4 | Scan the parameter values | "27000 V, 1.332e-3 F, 1.5e-8 H... these are numbers but I don't know if they're reasonable." | All fields populated with valid values | |
| 1.5 | Click ARM | "ARM... like arming a weapon? OK let me try it." | Advisories appear in the Co-Pilot sidebar (or inline). Button changes to FIRE. | |
| 1.6 | Read the advisories | "Configuration looks nominal. OK, I guess that means it's fine." | Messages are in plain English | |
| 1.7 | Click FIRE | "Here goes nothing." | Oscilloscope starts showing live traces | |
| 1.8 | Watch the oscilloscope | "Cyan line going up and down... that must be current? Yellow line... max_Te? What unit is Te in?" | Axis labels show units (A, eV, kg/m3) | |
| 1.9 | Wait for completion | "How long will this take? Is there a progress bar?" | Status changes to "Finished". No progress percentage shown. | |
| 1.10 | Read Post-Shot panel | "PINCH TIME: 5.23 us. PEAK CURRENT: 1.45 MA. These are results!" | Metrics displayed with engineering notation | |
| 1.11 | Click RESET | "OK, back to the start. That was... fast? But I don't really understand what I just saw." | Returns to Idle state | |

**Scoring**: Record wall-clock time from launch to reading Post-Shot results. Target: < 10 minutes.

**Alex's naive questions that the UI should answer**:
- "Why is temperature measured in eV instead of Kelvin?" (Tooltip should explain.)
- "What is a 'pinch'? Why does it have a 'time'?" (Tooltip on PINCH TIME should explain.)
- "What does PEAK CURRENT mean physically? Is 1.45 MA a lot?" (No context given.)
- "Why is neutron yield important? What are neutrons doing in a plasma simulation?" (No explanation visible.)

---

### S02: Discoverability — Can I Find What I Need?

**Goal**: Test whether key features are findable without prior knowledge of the interface.

| Step | Action | Expected | Pass/Fail |
|------|--------|----------|-----------|
| 2.1 | Find where to change the gas species | Gas section is expanded by default. Dropdown visible. | |
| 2.2 | Find where to change the computational backend | Solver section is collapsed by default. Alex must expand it. Label says "Backend" — clear. | |
| 2.3 | Find where to enable/disable radiation | Radiation section is collapsed. Toggle labels are jargon-heavy ("Bremsstrahlung", "Flux-Limited Diffusion"). | |
| 2.4 | Find the AI chat feature | Must click "AI Co-Pilot" button in TopBar, then scroll to bottom of sidebar to find ChatPanel. Is this too hidden? | |
| 2.5 | Find where to export simulation results | Is there an "Export" button anywhere? Download CSV? Save HDF5? **This feature may not exist in the GUI.** | |
| 2.6 | Find documentation or help link | Is there a "?" button, a help menu, or a link to docs? | |
| 2.7 | Find where to change the grid resolution | Grid section is collapsed. "Grid Shape [nx, ny, nz]" label is clear for someone who knows what a grid is, but "nx" is jargon. | |
| 2.8 | Find the CFL number | Buried inside collapsed Solver section. CFL has no unit displayed (it is dimensionless, but Alex does not know this). | |
| 2.9 | Find the simulation time control | Grid section, "Simulation Time" label with "s" unit. Clear. | |
| 2.10 | Find where to view energy conservation | Post-Shot panel after simulation completes. Not discoverable in advance. | |

**Friction score**: Count how many features require expanding a collapsed section. Currently: Backend (1 click), Radiation (1 click), Transport (1 click), Boundary (1 click), Diagnostics (1 click), Grid (1 click). Total: 6 features behind collapsed panels.

---

### S03: Tooltip Pedagogy — Learning Physics Through the UI

**Goal**: Evaluate whether the tooltips teach plasma physics concepts effectively to a newcomer.

| Tooltip Target | Current Tooltip Text | Alex's Assessment | Improvement Needed? |
|---------------|---------------------|-------------------|---------------------|
| Capacitance | "Energy storage capacitance of the capacitor bank. E = 1/2 CV^2." | Formulaic but understandable. Alex knows E = 1/2 CV^2 from undergrad. Good. | Minor: add typical range |
| Charge Voltage | "Initial charge voltage across the capacitor bank before discharge." | Clear. Alex asks: "What's a typical value? Is 27 kV safe?" | Add: "Typical: 10-50 kV" |
| Inductance | "External circuit inductance. Limits peak current via I_peak = V0*sqrt(C/L0)." | Formula is helpful. Subscript rendering (V0 not V_0) is ugly in plain text. | Render subscripts properly |
| CFL Number | "Courant stability factor -- lower is more stable, higher is faster. Must be < 1." | Good plain-English explanation. Alex understands the tradeoff. | None |
| Nernst Effect | "Thermoelectric B-field advection by electron heat flux. Important at high Te gradients." | **Too jargon-heavy**. Alex does not know what "thermoelectric B-field advection" means. | Rewrite for newcomers |
| Bremsstrahlung | "Free-free radiation from electron-ion collisions. Dominant loss at high Te." | "Free-free" is jargon. "Dominant loss" — loss of what? | Add: "braking radiation — energy lost as X-rays when electrons scatter off ions" |
| Full Braginskii Viscosity | "Full anisotropic viscous stress tensor with parallel and gyroviscous components." | **Completely opaque** to a newcomer. Every word except "full" is jargon. | Needs plain-English primer |
| Anomalous alpha | "Anomalous resistivity coefficient -- enhances Ohmic heating near the pinch column." | Somewhat clear. Alex asks: "What's Ohmic heating? What's the pinch column?" | Add cross-references |
| Powell 8-wave | "Powell divergence cleaning -- adds source terms to reduce div(B) errors." | Alex does not know what div(B) means or why it should be zero. | Add: "keeps the magnetic field physically consistent (no magnetic monopoles)" |
| Flux Limiter lambda | "Radiation flux limiter. 1/3 (Levermore-Pomraning) is standard." | "Levermore-Pomraning" means nothing to Alex. | Remove proper name, explain function |
| Coulomb Logarithm | "ln(Lambda) -- logarithmic measure of the ratio of max to min impact parameters in Coulomb scattering." | Technically correct. Alex knows logarithms. "Impact parameters" is graduate-level plasma physics. | Simplify |
| Anisotropic Conduction | "Heat conducts primarily along B-field lines (Braginskii kappa_parallel >> kappa_perp)." | Uses Greek letters in plain text. Concept is clear if Alex knows B-field lines. | Minor improvement |

**Verdict**: Approximately 5/12 tooltips are accessible to a newcomer. The remaining 7 assume graduate-level plasma physics knowledge. For a tool intended to be used in lab courses, this is a significant gap.

**Test**: After reading all visible tooltips (without using chat), can Alex explain the following?
- What a Dense Plasma Focus does (partially -- "it discharges a capacitor through plasma")
- Why temperature matters (partially -- "higher Te means more radiation loss")
- What a pinch is (no -- the word "pinch" appears in tooltips but is never defined)
- Why neutrons are produced (no -- fusion is never mentioned in tooltips)

---

### S04: Error Handling — What Happens When I Make Mistakes?

**Goal**: Verify that all error states produce helpful, actionable messages.

| Step | Action | Expected Error | Actually Happens | Pass/Fail |
|------|--------|---------------|-----------------|-----------|
| 4.1 | Clear the voltage field entirely | "Voltage is required" or field highlights red | Does `parseFloat("")` return NaN? Does the store handle it? | |
| 4.2 | Enter "abc" in the capacitance field | Input rejected or error shown | `parseFloat("abc")` is NaN; handleChange silently does nothing. **No error message shown.** | |
| 4.3 | Enter -1000 for voltage | "Voltage must be positive" | ScientificInput has no min/max for voltage. Value is accepted. ARM may or may not warn. | |
| 4.4 | Set CFL to 1.5 (above max) | Clamped to 0.99 silently | Input is clamped by `Math.min(max, parsed)`. No feedback to user that the value was adjusted. | |
| 4.5 | Set anode_radius > cathode_radius | Advisory warning about geometry mismatch | Does the advisory system check this? If not, simulation may produce garbage. | |
| 4.6 | Set grid shape to [0, 0, 0] | "Grid dimensions must be positive" | Input has `min="4"` but HTML min is advisory, not enforced on all browsers. `parseInt("0") || 8` falls back to 8. | |
| 4.7 | Set simulation time to 0 | "Simulation time must be positive" | No min is set on this field. Value 0 is accepted. Simulation may hang or produce no data. | |
| 4.8 | Click FIRE without ARM | Button should not be clickable | CommitButton logic should disable FIRE if not armed. Verify. | |
| 4.9 | Double-click FIRE rapidly | Should not start two simulations | Verify debouncing or lock mechanism. | |
| 4.10 | Enter 1e999 (overflow) for voltage | "Value out of range" or NaN handling | `parseFloat("1e999")` is Infinity. What does the store do with Infinity? | |
| 4.11 | Enter "1.23e" (malformed scientific) | Should reject or ignore | `parseFloat("1.23e")` returns 1.23 in most browsers. Potentially misleading. | |
| 4.12 | Backend connection lost mid-simulation | "Connection lost — simulation may be incomplete" | WebSocket disconnection handling. Is there a reconnection attempt? Timeout? | |

**Key finding from code review**: The `ScientificInput` component silently ignores NaN values (`if (!isNaN(parsed))` — line 39). This means typing garbage into a field produces **no feedback at all**. The field just does not update. Alex will type "hello" into the voltage field, see nothing happen, and be confused.

---

### S05: Input Validation — Boundary and Edge Cases

**Goal**: Systematically test input boundaries to find crashes, silent failures, or nonsensical behavior.

| ID | Input | Field | Expected Behavior | Notes |
|----|-------|-------|-------------------|-------|
| V-01 | 0 | Capacitance | Warning: "Zero capacitance means no stored energy" | Division by zero risk in sqrt(C/L) |
| V-02 | -1e-6 | Capacitance | Rejected: "Capacitance must be positive" | No min constraint in the component |
| V-03 | 1e-20 | Capacitance | Warning: "Extremely small capacitance" | May cause numerical overflow in V0*sqrt(C/L) |
| V-04 | 1e6 | Capacitance (1 MF) | Warning: "Unrealistically large capacitance" | No max constraint |
| V-05 | 0 | Voltage | Warning: "No energy in the bank" | E = 0.5*C*V^2 = 0 |
| V-06 | 1e9 (1 GV) | Voltage | Warning: "Extreme voltage" | May cause overflow in energy calculation |
| V-07 | -100 | Resistance | Rejected: "Resistance must be non-negative" | Negative resistance is unphysical (for passive circuits) |
| V-08 | 0 | Resistance | Acceptable (ideal circuit) | Should work but may affect damping calculations |
| V-09 | 0 | All three grid dims | Rejected | `parseInt("0") || 8` silently falls back to 8 |
| V-10 | 1 | Any grid dim | Warning: "Single-cell grid" | Too coarse for any physics |
| V-11 | 10000 | Any grid dim | Warning: "Very fine grid — may be slow" | Memory/time implications |
| V-12 | 0.001 | Anode radius (1 mm) | Acceptable | Small but physically possible |
| V-13 | 100 | Anode radius (100 m) | Warning: "Unrealistically large electrode" | No max constraint |
| V-14 | 0 | Fill pressure (Torr) | Warning: "Vacuum — no gas to ionize" | Will produce division issues in density calc |
| V-15 | 1e-30 | Initial density | May cause floating point underflow | Near machine epsilon |
| V-16 | NaN | Any field (paste from clipboard) | Silent rejection, no feedback | parseFloat("NaN") is NaN, silently ignored |
| V-17 | Infinity | Any field | Accepted? Stored as Infinity? | parseFloat("Infinity") returns Infinity |
| V-18 | 1e308 | Any field | Close to Number.MAX_VALUE | May cause overflow in derived calculations |
| V-19 | 5e-324 | Any field | Number.MIN_VALUE (denormalized) | May cause precision issues |
| V-20 | Unicode input (e.g., "2.5 x 10^4") | Voltage | parseFloat("2.5 x 10^4") returns 2.5 | Misleading — user thinks they entered 25000 |

---

### S06: Chat Interface — Conversational Learning

**Goal**: Test the WALRUS chat with progressively complex and adversarial queries.

#### Basic Discovery

| Step | Query | Expected Response Type | Assessment |
|------|-------|----------------------|------------|
| 6.1 | "help" | Lists 7 question types | Clear, actionable |
| 6.2 | "what is a z-pinch?" | Glossary definition | Good, covers the basics |
| 6.3 | "what is bremsstrahlung?" | Glossary definition | Good |
| 6.4 | "explain CFL" | Glossary definition | Good, plain English |

#### Naive Student Questions (Should Work but May Not)

| Step | Query | Expected | Actually Happens |
|------|-------|----------|-----------------|
| 6.5 | "Why is temperature in eV not Kelvin?" | Explanation of eV convention in plasma physics | Not in glossary. Returns "term not found." |
| 6.6 | "What's a pinch?" | Same as "pinch" glossary entry | Depends on regex matching "what's" vs "what is" |
| 6.7 | "What does the yellow line on the graph mean?" | "max_Te — the peak electron temperature" | Not a glossary term. Returns unknown intent. |
| 6.8 | "Is 27 kV a lot?" | Contextual: "27 kV is typical for DPF devices..." | Not handled by any intent pattern. |
| 6.9 | "What should I set the CFL to?" | "0.3-0.5 is typical for explicit MHD..." | Not in glossary. No "recommendation" intent. |
| 6.10 | "Why did my simulation fail?" | Diagnostic guidance | No "debug" or "troubleshoot" intent. |
| 6.11 | "What is MHD?" | Glossary: "Magnetohydrodynamics..." | Works (exact match on "mhd"). |
| 6.12 | "explain magnetohydrodynamics" | Same as "mhd" entry | May fail — substring match of "magnetohydrodynamics" against "mhd" key unlikely to match. |

#### Adversarial and Edge-Case Queries

| Step | Query | Expected | Actually Happens |
|------|-------|----------|-----------------|
| 6.13 | "" (empty) | "Please type a question..." | Handled correctly by empty-string check |
| 6.14 | "banana" | Friendly fallback | "I didn't understand that question." Correct. |
| 6.15 | "aaaa" * 1000 (very long string) | Graceful handling | No input length limit visible. May cause layout overflow. |
| 6.16 | HTML injection: "<script>alert(1)</script>" | Sanitized output | React auto-escapes JSX, so this should be safe. |
| 6.17 | "explain explain explain" | Should match "explain" intent | Regex captures "explain explain" as the term. Glossary lookup fails. OK. |
| 6.18 | "HELP" (all caps) | Same as "help" | Regex is case-insensitive. Works. |
| 6.19 | "what maximizes banana?" | Inverse intent but no valid field | Regex matches "maximizes banana". detectTargetField returns "Te" (default). Misleading. |
| 6.20 | "sweep voltage from -100 to 100" | Should warn about negative voltage | Regex captures -100 and 100. No validation on values. |
| 6.21 | Rapid-fire: send 50 messages in 10 seconds | No crash, messages queue correctly | chatStatus === 'sending' disables input. Only one at a time. |
| 6.22 | "predict next step" (no model loaded) | "No surrogate model is currently loaded..." | Correct offline fallback. |
| 6.23 | "explain walrus" | Glossary definition of WALRUS | Works if "walrus" key matches. |
| 6.24 | "what is the meaning of life?" | Friendly fallback | Returns unknown intent. Good. |

**Chat UX issues identified from code review**:
- Max chat height is 256px (`max-h-64`). After ~8 messages, older messages scroll off. No way to search or expand the chat area.
- Suggestion chips populate the input field but do not auto-send. Alex must click Send after clicking a chip. This is a two-step interaction where one step is expected.
- No message timestamp shown. Alex cannot tell if a response is from 5 seconds ago or 5 minutes ago.
- No copy-to-clipboard button on responses.
- No markdown rendering in responses (plain text with whitespace-pre-wrap).

---

### S07: Simulation Workflow — End-to-End Completeness

**Goal**: Trace the full lifecycle of a simulation and identify any missing steps.

| Phase | Step | Detail | Gap? |
|-------|------|--------|------|
| Configure | Select preset | Works | No |
| Configure | Modify parameters | Works per-field | No |
| Configure | Save custom preset | **Not possible.** No "Save as Preset" button. | Yes |
| Validate | Click ARM | Triggers validation + advisory | No |
| Validate | Review advisories | Shows in Co-Pilot sidebar. Must have sidebar open. | Partial |
| Execute | Click FIRE | Starts simulation | No |
| Monitor | Watch oscilloscope | Live traces update | No |
| Monitor | See progress/ETA | **No progress bar. No ETA. No percentage.** Only step count and simulation time, which Alex cannot interpret. | Yes |
| Monitor | Cancel mid-run | Click STOP (FIRE button becomes STOP when running) | No |
| Analyze | Read Post-Shot | Pinch time, peak current, neutron yield, energy partition | No |
| Analyze | Zoom into traces | **Not possible.** Oscilloscope is a static chart that updates live but has no zoom, pan, or cursor. | Yes |
| Analyze | Export data | **Not possible in GUI.** No CSV export, no HDF5 download, no clipboard copy. | Yes |
| Compare | Compare with previous run | **Not possible.** No run history. No overlay. RESET clears all data. | Yes |
| Iterate | Adjust and re-run | Works (RESET then modify) | No |
| Document | Save/load session | **Not possible.** No project file, no save state. | Yes |

**Critical gaps for a student workflow**: No data export, no run comparison, no session persistence, no zoom on charts.

---

### S08: Data Visualization — Can I Read the Plots?

**Goal**: Evaluate the oscilloscope and post-shot visualizations for clarity and usefulness.

| Aspect | Assessment | Severity |
|--------|-----------|----------|
| Axis labels | "Current (A)" and "max_Te (eV)" are shown. "max_rho (kg/m3)" on secondary axis. Clear. | OK |
| Axis scaling | Auto-scaled. Values like 1.5e6 A are shown in raw form, not "1.5 MA". Long numbers clutter the axis. | Medium |
| Time axis | Shows simulation time in seconds. For microsecond timescales, values like 0.000005 appear. Not in engineering notation. | High |
| Trace colors | Cyan (current), Amber (max_Te), Red (max_rho). Distinct. But no legend is visible in the TraceChart component. | Medium |
| Trace legend | Title says "ELECTRICAL" or "PLASMA". Individual trace names (I(t), max_Te, max_rho) may not be visible as a legend. | Medium |
| Chart interactivity | No zoom, no pan, no cursor tooltip on hover, no data point inspection. | High |
| Chart resize | Charts are flex-1 inside the scope panel. Should resize with window. | Low |
| Dark background | Charts on #0A0A0A. Good contrast for colored traces. | OK |
| Grid lines | Not visible in the component code. Charts may lack reference grid lines. | Medium |
| Engineering notation on axes | Not applied. Raw SI values shown. 1500000 A instead of 1.5 MA. | High |
| Multiple y-axes | max_Te (left axis) and max_rho (right axis) on same chart. Dual-axis charts are notoriously confusing. | Medium |
| PostShot numbers | Pinch time in microseconds (good). Peak current in MA (good). Neutron yield in raw count (unclear magnitude). | Low |
| Energy partition | EnergyPartition component exists. How is it visualized? Bar chart? Table? Not clear from code alone. | Low |

---

### S09: Performance Under Stress — Long Runs and Heavy Loads

**Goal**: Test behavior during resource-intensive operations.

| Step | Action | Expected | Watch For |
|------|--------|----------|-----------|
| 9.1 | Run with 256x1x512 grid (large) | Simulation runs but slowly | Memory usage (check Activity Monitor). UI responsiveness during run. |
| 9.2 | Run with sim_time = 1e-3 (1 ms — very long for DPF) | Many timesteps | Does the oscilloscope bog down with thousands of data points? |
| 9.3 | Open AI sidebar during running simulation | Sidebar opens without lag | Does the WebSocket data stream affect sidebar rendering? |
| 9.4 | Run parameter sweep with 100 points | Sweep completes without crash | Memory accumulation from storing 100 simulation results. |
| 9.5 | Type in chat while simulation is running | Chat still responsive | Are WebSocket and chat sharing a thread/connection? |
| 9.6 | Resize window during simulation | Layout adapts without artifacts | Oscilloscope chart reflows correctly. |
| 9.7 | Close and reopen AI sidebar during simulation | No data loss | Simulation continues. Advisories preserved. |
| 9.8 | Run two simulations back-to-back without RESET | Second simulation should either work or give clear error | Does the state machine handle this? |
| 9.9 | Leave simulation running and return after 10 minutes | Data still streaming. UI responsive. | Memory leak from accumulated scalarHistory? |
| 9.10 | Kill the Python backend during a simulation | GUI shows error state | "Connection lost" or "Backend unavailable" message. Does the GUI recover when backend restarts? |

---

### S10: AI Co-Pilot — Sweep and Inverse Design

**Goal**: Test the AI features from a newcomer's perspective.

| Step | Action | Alex's Thought Process | Expected | Pass/Fail |
|------|--------|----------------------|----------|-----------|
| 10.1 | Open Sweep panel | "Parameter Sweep — what does that mean?" | Panel has variable selector, min/max, N points. Labels are clear. | |
| 10.2 | Run default sweep (V0, 10kV-50kV, 10 points) | "Let me see what voltage does." | Sweep runs (if backend available) or shows message about needing WALRUS | |
| 10.3 | Change sweep metric to "Neutron Rate" | "I want to see how voltage affects neutrons." | Scaling curve updates | |
| 10.4 | Set min > max in sweep | "What if I put 50000 in min and 10000 in max?" | Validation: `min >= max` check returns silently (line 28). No error message. **Silent failure.** | |
| 10.5 | Set N points to 0 | "What's the minimum?" | `nPoints < 2` check returns silently. **Silent failure.** | |
| 10.6 | Open Inverse Design | "Specify target outcomes — WALRUS finds optimal parameters. Interesting." | Panel loads with target metric selector | |
| 10.7 | Set target: max_Te = 1.0 keV | "Is 1 keV a reasonable target?" | No guidance on realistic target values | |
| 10.8 | Click FIND OPTIMAL CONFIG (no backend) | "Let me try..." | Error: "Inverse design failed — check AI status". Message is vague. | |
| 10.9 | Enable constraints, set V0 max = 50000 | "I want to constrain the voltage." | Constraints panel appears | |
| 10.10 | After a successful inverse run, click APPLY CONFIG | "Oh, it fills in the optimal parameters! That's cool." | Parameters update in the left panel | |

**UX issues in AI panels**:
- Sweep panel MIN/MAX fields have no units shown. Alex enters "10000" — is that Volts? The variable label says "V0 (Voltage)" but the min/max fields do not repeat the unit.
- Inverse Design says "keV" for temperature but the main parameter panel shows temperature in "K" (Kelvin). Unit inconsistency.
- No tooltip on the "Bayesian" vs "Evolutionary" method selector explaining the difference.
- The TRIALS field (10-1000) gives no guidance on how many trials are reasonable. 10? 100? 1000?

---

### S11: Responsive Feedback — What Is the System Doing?

**Goal**: Evaluate whether the GUI communicates its state clearly during all operations.

| State | Feedback Given | Feedback Missing |
|-------|---------------|-----------------|
| Idle | Green dot + "Idle" text | OK |
| Validating (ARM clicked) | ??? | No spinner or "Validating..." indicator. The advisory panel simply populates after a delay. |
| Armed | FIRE button pulses cyan | Good feedback |
| Running | Green dot + "Running" + step/time counters | No progress percentage. No ETA. No "step 150 of ~1000" estimate. |
| Paused | Yellow dot + "Paused" | Is there a PAUSE button? Only STOP is documented. |
| Finished | Checkmark + "Finished" + PostShot panel | Good |
| Error | Red dot + "Error" | No error details in the TopBar. Must check advisories or console. |
| Backend disconnected | ??? | No reconnection indicator. No retry mechanism visible. |
| Sweep running | "RUNNING..." on button | No progress (e.g., "3 of 10 configs"). No cancel button. |
| Inverse running | "OPTIMIZING..." on button | No progress. No trial counter. No cancel. |
| Chat waiting | Bouncing dots | Good. But no timeout handling — what if the backend never responds? |

---

### S12: Data Export and CLI Consistency

**Goal**: Test whether GUI results can be exported and whether they match CLI results.

| Step | Action | Expected | Gap? |
|------|--------|----------|------|
| 12.1 | Look for an "Export" button after simulation | CSV, JSON, or HDF5 download | **No export button exists in the GUI.** |
| 12.2 | Try to copy Post-Shot values to clipboard | Select text, Cmd+C | Text selection on styled divs may be unreliable. |
| 12.3 | Run identical config from CLI: `dpf run --preset pf1000` | Get output values | CLI output format differs from GUI Post-Shot format? |
| 12.4 | Compare CLI peak current with GUI peak current | Values should match within tolerance | Backend is the same, so values should be identical for same random seed. |
| 12.5 | Try to reproduce sweep results from CLI | `dpf ai sweep --variable V0 --min 10000 --max 50000 --npoints 10` | Is the CLI sweep API identical to the GUI sweep? |
| 12.6 | Check if oscilloscope data can be saved | No "Save Trace" button | **Data is only in-memory in the Zustand store.** |

---

### S13: Onboarding and Tutorial Absence

**Goal**: Audit what learning materials exist and what is missing.

| Resource | Exists? | Location | Quality |
|----------|---------|----------|---------|
| In-app tutorial/wizard | No | — | N/A |
| Welcome screen | No | — | N/A |
| Interactive walkthrough | No | — | N/A |
| Tooltips on parameters | Yes | Hover on labels | Mixed (see S03) |
| Chat-based help | Yes | "help" command | Good — lists capabilities |
| Chat-based glossary | Yes | "explain X" command | Good — ~30 terms |
| README or manual | Unknown | Not linked from GUI | Not accessible in-app |
| Video tutorial | No | — | N/A |
| Sample configurations | Yes | Presets dropdown | Good — provides starting points |
| "What's this?" context help | Partial | Tooltips only, no "?" icons | No panel-level explanations |
| Physics primer | No | Not in GUI | Would be valuable for students |
| Keyboard shortcuts reference | No | — | N/A |

**Minimum viable onboarding for a lab course**:
1. A 3-step overlay on first launch: "1. Select a preset. 2. Click ARM. 3. Click FIRE."
2. A "What is DPF?" link or subtitle in the TopBar.
3. Tooltip on the PINCH TIME metric in Post-Shot: "The time at which the plasma column reaches maximum compression."
4. A "Getting Started" item in the preset dropdown.

---

### S14: Multi-Run Comparison Workflow

**Goal**: Test the workflow for comparing multiple simulation runs with different parameters.

| Step | Action | Expected | Gap? |
|------|--------|----------|------|
| 14.1 | Run PF-1000 at V0 = 27 kV | Get Post-Shot results | OK |
| 14.2 | Write down the results on paper | Alex manually records: I_peak = 1.45 MA, t_pinch = 5.2 us | **No clipboard/export means manual transcription** |
| 14.3 | Click RESET | All data cleared | No way to keep the old run's data |
| 14.4 | Change V0 to 40 kV, run again | Get new results | OK |
| 14.5 | Compare with the first run | Must compare from memory or paper | **No overlay, no table, no history** |
| 14.6 | Attempt to undo the RESET | Not possible | No undo/redo at any level |
| 14.7 | Attempt to name or tag a run | Not possible | No run naming/tagging |

**Impact**: For a student performing parameter studies, the inability to compare runs side-by-side is the single most impactful missing feature. Every other physics simulation tool (COMSOL, OpenFOAM, ParaView) supports this.

---

## 4. "Break It" Test Suite

Intentional misuse scenarios designed to find crashes, hangs, data corruption, or confusing states.

### 4.1 Numerical Abuse

| ID | Action | Expected Resilience |
|----|--------|-------------------|
| BK-01 | Set every numeric field to 0 | No crash. Advisory should flag multiple issues. |
| BK-02 | Set every numeric field to Infinity | No crash. Validation should reject or clamp. |
| BK-03 | Set every numeric field to -1 | No crash. Negative values rejected where unphysical. |
| BK-04 | Set voltage to Number.MAX_VALUE (1.79e308) | No overflow in E = 0.5*C*V^2. |
| BK-05 | Set capacitance to Number.MIN_VALUE (5e-324) | No underflow in sqrt(C/L). |
| BK-06 | Set anode_radius = cathode_radius (zero gap) | Warning or error. Not a valid geometry. |
| BK-07 | Set grid to [4, 4, 4] (minimum allowed) | Simulation runs but is very coarse. Advisory should note this. |
| BK-08 | Set grid to [1000, 1, 1000] (1M cells) | Memory warning? Slow simulation. UI should not freeze. |

### 4.2 Timing Abuse

| ID | Action | Expected Resilience |
|----|--------|-------------------|
| BK-09 | Click ARM 100 times rapidly | No duplicate advisory generation. Debounced. |
| BK-10 | Click FIRE, immediately STOP, immediately ARM, immediately FIRE | State machine handles rapid transitions. |
| BK-11 | Click RESET while the ARM advisory is still loading | No stale advisory data. |
| BK-12 | Open/close AI sidebar 50 times in 10 seconds | No memory leak, no DOM accumulation. |
| BK-13 | Send chat message then immediately click Clear | Message either sends or is cancelled. No ghost messages. |

### 4.3 State Corruption

| ID | Action | Expected Resilience |
|----|--------|-------------------|
| BK-14 | Modify parameters while simulation is running | Inputs should be locked (disabled). Verify all fields. |
| BK-15 | Change backend selector while simulation is running | Should be locked. |
| BK-16 | Change preset while simulation is running | Should be locked. |
| BK-17 | Navigate away from the page (Electron: close tab?) during simulation | Simulation should stop or continue in background. No data corruption. |
| BK-18 | Kill the Python backend, restart it, then click FIRE | Should reconnect or show clear error. |
| BK-19 | Start two instances of the GUI simultaneously | Second instance should either connect to same backend or show "port in use". |

### 4.4 Input Injection

| ID | Action | Expected Resilience |
|----|--------|-------------------|
| BK-20 | Paste `<img src=x onerror=alert(1)>` into chat | React escapes HTML. No XSS. |
| BK-21 | Paste a 10 MB string into any input field | Browser should handle gracefully. No crash. |
| BK-22 | Paste emoji into numeric fields | parseFloat ignores emoji. Silent failure (see S04 assessment). |
| BK-23 | Use browser DevTools to modify Zustand store directly | Simulation still works or gracefully errors. |

---

## 5. Competitive Comparison with Other Physics Simulation GUIs

Alex has brief experience with COMSOL and has seen screenshots of other tools. How does DPF Simulator compare?

| Feature | DPF Simulator | COMSOL Multiphysics | SimFlow (OpenFOAM GUI) | Simcenter STAR-CCM+ |
|---------|---------------|--------------------|-----------------------|--------------------|
| **First-run wizard** | None | "Model Wizard" guides physics/geometry/mesh | Template selector | "New Simulation" dialog |
| **Geometry builder** | None (radius inputs only) | Full 2D/3D CAD | Import or built-in | Full CAD |
| **Mesh visualization** | None | Interactive 3D mesh view | 3D mesh view | Full 3D |
| **Physics module selection** | Toggles (Bremsstrahlung, Nernst, etc.) | Categorized physics tree | Dropdown per region | Physics continua tree |
| **Parameter units** | Shown but not editable (always SI) | Unit-aware inputs (can type "27 kV") | SI only | Multiple unit systems |
| **Tooltips** | Yes, ~30 fields covered | Context-sensitive help panel | Basic tooltips | Integrated docs |
| **AI assistant** | WALRUS chat (glossary + guided queries) | None built-in | None | None |
| **Run progress** | Step count and sim time only | Progress bar with ETA | Progress bar | Progress with metrics |
| **Post-processing** | Post-Shot panel (5 metrics) | Full 2D/3D plotting, exports | ParaView integration | Built-in post-processing |
| **Data export** | Not available in GUI | CSV, Excel, MATLAB, image | VTK, CSV | CSV, CGNS, images |
| **Multi-run comparison** | Not available | Parametric sweep with overlay | Manual only | Design exploration |
| **Session save/load** | Not available | .mph project files | .cfg save | .sim files |
| **Accessibility** | Not audited | WCAG partial | Not audited | Not audited |
| **Learning curve** | ~10 min to first simulation | ~2 hours to first simulation | ~1 hour | ~4 hours |
| **Price** | Free (open source) | $$$$ (tens of thousands) | Free (open source) | $$$$ |

**Where DPF Simulator excels**: Time to first simulation (fastest of all tools), AI chat assistant (unique), domain-focused UX (no unnecessary features).

**Where DPF Simulator lags**: No data export, no multi-run comparison, no progress indicator, no geometry/mesh visualization, no session persistence.

---

## 6. Usability Scoring Rubric

Adapted from the System Usability Scale (SUS), customized with physics simulation-specific dimensions. Each dimension scored 1-5 (1 = Strongly Disagree, 5 = Strongly Agree).

| ID | Dimension | Question (Alex rates 1-5) | Weight |
|----|-----------|--------------------------|--------|
| U-01 | **Learnability** | "I could run a simulation without reading any documentation." | Critical |
| U-02 | **Efficiency** | "I can complete a parameter study (5+ runs) in under 30 minutes." | High |
| U-03 | **Error Tolerance** | "When I make a mistake, the system tells me what went wrong and how to fix it." | Critical |
| U-04 | **Discoverability** | "I was able to find all the features I needed without asking for help." | High |
| U-05 | **Transparency** | "I understand what the simulation is doing and what the results mean." | Critical |
| U-06 | **Aesthetic Appeal** | "The interface looks professional and is pleasant to use." | Low |
| U-07 | **Cognitive Load** | "I do not feel overwhelmed by the number of parameters and options." | High |
| U-08 | **Feedback Quality** | "The system keeps me informed about what is happening at all times." | High |
| U-09 | **Recovery** | "If something goes wrong, I can get back to a working state easily." | High |
| U-10 | **Satisfaction** | "I would recommend this tool to another student in my lab." | Medium |
| U-11 | **Physics Pedagogy** | "I learned something about plasma physics by using this tool." | High |
| U-12 | **Data Utility** | "I can extract the simulation results I need for my research or report." | Critical |
| U-13 | **Workflow Completeness** | "The tool supports my full workflow from setup to analysis to documentation." | High |
| U-14 | **Terminology Clarity** | "I understand the labels and terms used in the interface." | High |
| U-15 | **Progressive Disclosure** | "The interface shows me basic options first and lets me access advanced options when I need them." | Medium |

**Composite Score**: (Sum of all ratings) / 75 * 100 = SUS-equivalent percentage.

**Interpretation**:
- 80-100%: Excellent — ready for classroom use
- 60-79%: Good — usable with minor training
- 40-59%: Fair — needs a tutorial or onboarding
- Below 40%: Poor — significant redesign needed

**Predicted score for Alex**: ~55-65% (Fair to Good). Strong on learnability (preset-first design) and aesthetic appeal (polished dark theme). Weak on data utility (no export), error tolerance (silent failures), and workflow completeness (no comparison, no save).

---

## 7. Accessibility Audit Checklist

Based on WCAG 2.1 AA guidelines, adapted for an Electron desktop application.

### 7.1 Perceivable

| ID | Criterion | Test | Status |
|----|-----------|------|--------|
| A-01 | Color contrast (text on background) | Measure contrast ratio of #999999 (labels) on #1E1E1E (background). Ratio = 5.3:1. AA requires 4.5:1 for normal text. | PASS |
| A-02 | Color contrast (cyan on dark) | #00E5FF on #121212. Ratio = 10.7:1. | PASS |
| A-03 | Color contrast (gray-500 text) | #666666 on #1E1E1E. Ratio = 3.4:1. AA requires 4.5:1. | FAIL |
| A-04 | Color contrast (error red) | #FF5252 on #2A2A2A. Ratio = 4.6:1. | PASS (barely) |
| A-05 | Non-color indicators | Status uses both color dots AND text ("Idle", "Running"). | PASS |
| A-06 | Advisory severity | Uses color + icon (dot, triangle, circle-X). Color-blind users can distinguish. | PASS |
| A-07 | Text sizing | Most text is text-xs (12px). May be too small for visually impaired users. | WARN |
| A-08 | Zoom support | Cmd+/- zoom in Electron. Does layout reflow? Or do elements overlap? | TEST |
| A-09 | Focus indicators | Custom inputs (dpf-input). Do they show focus rings? | TEST |
| A-10 | Alt text on images/icons | SVG icons have no aria-label. Chevron, status icons, etc. | FAIL |

### 7.2 Operable

| ID | Criterion | Test | Status |
|----|-----------|------|--------|
| A-11 | Keyboard navigation (Tab order) | Can Alex Tab through all parameters, buttons, and controls in logical order? | TEST |
| A-12 | Keyboard activation | Can all buttons be activated with Enter/Space? | TEST |
| A-13 | Focus trap in sidebar | When sidebar is open, is Tab cycling trapped? It should not trap focus. | TEST |
| A-14 | Skip to main content | Is there a skip link for keyboard users? (Unlikely in an Electron app.) | FAIL |
| A-15 | Keyboard shortcuts | No keyboard shortcuts documented. ARM (Cmd+Enter?), FIRE (F5?), etc. | FAIL |
| A-16 | Toggle switches (keyboard) | ToggleSwitch uses a `<button>`. Should be keyboard-accessible. But the button has no aria-role or aria-checked attribute. | FAIL |
| A-17 | Collapsible sections | Section toggle buttons: do they announce expanded/collapsed state to screen readers? No aria-expanded attribute. | FAIL |
| A-18 | Dropdown selects | Native `<select>` elements. Keyboard accessible by default. | PASS |

### 7.3 Understandable

| ID | Criterion | Test | Status |
|----|-----------|------|--------|
| A-19 | Language attribute | `<html lang="en">` set? Check index.html. | TEST |
| A-20 | Label association | `<label>` elements use className but no `htmlFor` attribute. Inputs are not explicitly associated. | FAIL |
| A-21 | Error identification | Errors shown as red text below the ARM button. Are they associated with specific fields via aria-describedby? | FAIL |
| A-22 | Consistent navigation | Layout is consistent across states (Idle, Running, Finished). | PASS |
| A-23 | Input purpose | No `autocomplete` attributes on inputs (not applicable for physics parameters). | N/A |
| A-24 | Form validation timing | Validation only on ARM click, not on field blur. Alex fills in a bad value and sees no feedback until ARM. | WARN |

### 7.4 Robust

| ID | Criterion | Test | Status |
|----|-----------|------|--------|
| A-25 | Valid HTML | Run through an HTML validator. React usually produces valid HTML. | TEST |
| A-26 | ARIA roles | ToggleSwitch should have `role="switch"` and `aria-checked`. Currently missing. | FAIL |
| A-27 | Screen reader testing | Run VoiceOver (macOS) and navigate the full interface. | TEST |
| A-28 | Name/role/value | Custom components (ScientificInput, ToggleSwitch, CommitButton) need proper ARIA attributes. | FAIL |

### Accessibility Summary

| Category | Tests | Pass | Fail | Warn | Untested |
|----------|-------|------|------|------|----------|
| Perceivable | 10 | 5 | 2 | 1 | 2 |
| Operable | 8 | 2 | 4 | 0 | 2 |
| Understandable | 6 | 1 | 2 | 1 | 0 |
| Robust | 4 | 0 | 2 | 0 | 2 |
| **Total** | **28** | **8** | **10** | **2** | **6** |

**Accessibility grade**: Approximately 29% pass rate (8/28). This is well below WCAG 2.1 AA compliance. The primary deficiencies are ARIA attributes on custom components, label associations, and keyboard navigation support.

---

## Appendix: Alex's Lab Notebook

*Fictional diary entries recording Alex's experience. Written in first person to convey the emotional arc of using the tool.*

### Day 1 — First Contact

> Opened the DPF Simulator for the first time. The dark theme looks cool, like something from a sci-fi control room. I picked the PF-1000 preset because it sounded like the most important one. Clicked ARM (weird button name), then FIRE, and watched cyan lines dance on the screen. The whole thing took maybe 3 minutes. I have no idea what I just simulated, but the numbers look impressive — 1.45 MA peak current? That sounds like a lot of current.

> The tooltips helped for some things (I now know what CFL means) but others were way over my head. "Thermoelectric B-field advection" — I'll need to look that up. The chat was nice — I asked "what is a z-pinch?" and got a good answer. But when I asked "why did my neutron yield come out so low?" it just said it didn't understand.

### Day 2 — Trying to Break Things

> I tried entering negative voltage (-5000 V) and the app just... accepted it. No warning, no error. The simulation ran and produced bizarre results. Same with zero capacitance — it ran and gave me NaN everywhere. I expected the software to stop me, like MATLAB does when you try to take the square root of a negative number.

> I also tried entering "hello" into the voltage field. Nothing happened — the value just stayed at whatever it was before. No error message, no red border, nothing. I had to look at the code to understand why (parseFloat returns NaN, which is silently ignored). A new user would just think the input is broken.

### Day 3 — The Missing Features

> Professor Rivera asked me to run simulations at 5 different voltages and plot peak current vs. voltage. This should have been easy. But there's no way to save data from a run — no export button, no CSV, nothing. I had to write down the numbers on a post-it note after each simulation. Then I realized the RESET button deletes everything, so I can't even scroll back to check my numbers.

> The sweep panel in the AI sidebar could do this automatically, but it needs a WALRUS checkpoint, which we don't have. So I'm stuck doing it manually, writing numbers on post-its like it's 1995.

### Day 4 — Making Peace

> OK, I've found my workflow: preset, modify one parameter, ARM, FIRE, write down results, RESET, repeat. It's not elegant but it works. The chat is genuinely useful for learning physics — I've been asking it about every term I don't understand. I now know what bremsstrahlung is, what Bennett equilibrium means, and why CFL matters.

> If I could have three wishes: (1) an Export button, (2) a run history panel, (3) better error messages. The core simulation experience is solid. The physics seems to work. The UI just needs to grow up a bit around the edges.

---

*End of UAT-B Enhanced document.*

*Test plan authored for: DPF Unified v1.0.0*
*Persona: Alex Chen, 2nd-year M.S. student, computational physics*
*Evaluation scope: GUI usability, learnability, error handling, accessibility*
*Total test steps: 200+ across 14 scenarios + break-it suite*
*Estimated execution time: 4-6 hours*
