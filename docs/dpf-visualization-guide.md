# DPF 3D Renderer: Research-Backed Visualization Guide

## What Experiments Actually Show

### Axial Rundown Phase (0-5 us)

**Current sheath geometry**: NOT a simple flat disk. Magnetic probing reveals an **axisymmetric parabolic current sheath** that propagates down the coaxial tube (plasma-universe.com; Krishnan 2012 IEEE TPS). The sheath has a multi-region structure with four distinct zones:

1. **Magnetic piston** (rear): driven by J x B force from azimuthal B_theta
2. **Current-carrying sheath** (middle): the dense, luminous layer — a few mm thick
3. **Shock front** (leading edge): hydrodynamic shock ahead of the current layer
4. **Undisturbed gas** (ahead): cold fill gas not yet reached by the shock

The Gratton-Vargas snowplow model gives the analytic surface shape: a curved axisymmetric surface that the Lee model simplifies to a flat piston. In reality, the sheath is **parabolic** — it curves backward near the outer electrode (cathode) because the outer gas sees more inertia and the magnetic pressure is lower at larger radius.

**What cameras see**: Framing cameras (5 ns gate, ICCD) show a luminous annular front propagating axially. Filamentary structure is visible — current does not flow uniformly but forms **filaments** that run along the electrodes (Stanford Computer Optics; Auluck 2019 APS). ALEGRA 3D simulations confirm filamentation during lift-off and rundown, matching experimental observations.

**Renderer prescription**:
- Annular shell between anode (inner) and cathode (outer), advancing in z
- Parabolic curvature: center leads, edges trail
- Thin luminous shell (~2-5 mm thickness for kJ devices, ~1 cm for MJ)
- Optional: filamentary substructure (azimuthal perturbations on the shell)
- Behind the sheath: swept-up ionized gas (dim glow)
- Ahead of the sheath: dark (cold neutral gas)

### Radial Implosion Phase (5-7 us)

**Sheath geometry**: As the sheath slides off the anode face, the endpoints remain attached to the electrode edges. This causes the plasma sheet to **bow outward into an umbrella or mushroom-cap shape** (Wikipedia/DPF; Krishnan 2012). The Gratton-Vargas model reproduces this umbrella-like profile.

The transition from axial to radial produces a **curved, roughly conical surface**: the sheath connects the anode tip (at the axis) to the cathode rim (at the outer radius), forming a surface of revolution. The radial phase proper begins when this surface starts imploding inward.

During implosion, two distinct fronts exist:
1. **Shock front**: leading edge, converging on axis. Visible in schlieren and interferometry.
2. **Magnetic piston** (current sheath): trailing the shock by ~1-2 mm. Drives the implosion.

**What cameras see**: Interferometry captures the implosion as converging annular fringes. 16 interferograms from a single PF shot (200 ns span) show the plasma transforming through **spherical-like, helical, and toroidal structures** (IEEE 2010 Interferometry). Schlieren imaging (10 MHz CMOS, Z-type apparatus) shows density gradients in the converging sheath (IEEE 2024 MJOLNIR schlieren).

**Renderer prescription**:
- Umbrella/mushroom-cap shaped surface transitioning to inward-converging cylinder
- Two concentric surfaces: outer shock front (transparent, sharp), inner piston (luminous)
- The "focus tube" geometry: cylinder of plasma converging on the axis above the anode tip
- Elongation: the pinch column grows axially as the radial compression proceeds
- B_theta field lines: concentric circles around the z-axis, compressed between piston and axis

### Pinch Phase (7-8 us)

**Pinch column structure**: When the shock coalesces on axis, a reflected shock emanates outward until meeting the inward-moving piston. This forms the **axisymmetric boundary of the focused hot plasma column** (plasma-universe.com). The column has:

- **Radius**: ~1-2 mm for kJ devices, up to ~1 cm for MJ devices
- **Length**: ~1-2 cm (kJ) to ~10 cm (MJ), extending above the anode tip
- **Density**: core densities of 10^20 - 10^21 cm^-3 (Stanford Computer Optics)
- **Temperature**: ~1-2 keV (kJ devices), creating a Bennett equilibrium profile

The Bennett equilibrium radial profile gives:
- Density: peaked on axis, falling as n(r) = n0 / (1 + r^2/a^2)^2
- Temperature: relatively flat near axis (high thermal conductivity), steep gradient at the characteristic radius a (low thermal conductivity due to strong magnetization)
- B_theta: peaks at the column boundary (~a), falls as 1/r outside
- Pressure balance: kinetic pressure gradient = J x B magnetic pressure

**What X-ray pinhole images show**: Soft X-ray (1-10 keV) pinhole cameras with CCD capture the hot pinch column as a **bright, narrow, elongated source** centered on the anode tip axis. The emission is brightest at the column center (highest T and n) and fades at the ends. Hard X-rays (>20 keV) come from beam-target interactions and Bremsstrahlung.

**What interferometry shows**: Electron density maps of the pinch region, colored on a log scale. The column appears as a high-density spike on axis, with density falling off radially following the Bennett profile.

**Renderer prescription**:
- Narrow luminous column on axis, extending ~1-2 anode radii above the anode tip
- Bennett radial density profile: bright core fading to edges
- Temperature visualization: hottest on axis (~1-2 keV), cooler at boundary
- B_theta field: concentric circles, strongest at column boundary
- Spontaneous poloidal B-field (Auluck 2024 PoP): axial field component generated by sheath curvature — shown as field lines threading the column along z
- Column may show early sinusoidal perturbations (onset of m=0)

### Post-Pinch Phase (8+ us)

**m=0 sausage instability**: The dominant instability in DPF pinches. The column develops periodic **necking** (constrictions) and **bulging** along its length. The nonlinear evolution produces a "spindle" or cylindrical "spike and bubble" shape with sharp radial maxima (ADS 1976 Phys. Fluids).

- **Wavelength**: roughly equal to the column diameter (lambda ~ 2*pi*a)
- **Growth rate**: on the Alfven timescale (~10-50 ns for typical DPF)
- **Nonlinear stage**: necks pinch off, column breaks into discrete plasmoids
- **The filaments pinch together, forming dense plasmoids** with core densities ~10^20-10^21 cm^-3

**m=1 kink instability**: The column bends helically. Less prominent in DPF than m=0 but present, especially in larger devices.

**Ion beams**: At the neck points of m=0, locally enhanced electric fields accelerate ions to **100 keV - several MeV** (Researchgate; Krishnan 2012). Deuteron beams are emitted predominantly along the axis (both toward and away from the anode). The beam pulse is <100 ns duration.

**Radiation zones**:
- **Bremsstrahlung**: thermal emission from the hot column, continuous spectrum
- **Line radiation**: from impurity ions (if present) or working gas
- **Recombination radiation**: as plasma cools
- The Lee RADPF model separately tracks Joule heating, Bremsstrahlung, line emission, and recombination losses

**What cameras see**: Visible light shows the column breaking up into discrete bright spots (plasmoids) connected by dim bridges. Time-resolved imaging captures this breakup over ~50-200 ns.

**Renderer prescription**:
- Column with sinusoidal m=0 perturbations growing in amplitude
- At late times: discrete bright plasmoids separated by thin necks
- Ion beam: narrow cone along the axis, emanating from the neck/break points
- X-ray emission zones: hot spots at the compression maxima
- Eventually: column disperses, large diffuse plasma fills the electrode region

---

## Simulation Visualization Conventions

### What Published Codes Show

**Lee Model (RADPF)**: 0D model. Standard output is **time-series line plots**:
- Figures 1-2: I(t), V(t) vs time (microseconds)
- Figures 2-3: Axial position z(t), axial speed vs time
- Figures 3-5: Radial shock front r_s(t), piston r_p(t), column length z_f(t) vs time (nanoseconds, referenced to radial phase onset)
- Figures 6-8: T_e(t), radiation powers (Bremsstrahlung, line, recombination), Joule heating
- All plotted as standard 2D line graphs. No spatial visualization.

**ALEGRA (Sandia)**: 2D/3D Eulerian MHD. Produces:
- **Density contour plots** in r-z cross-section
- Density plots show "plasma tendrils" during rundown and compact structure during pinch
- 3D simulations show filamentation matching experiments
- Log-scale density colormaps

**LA-COMPASS (LANL)**: 2D/3D MHD. Similar output to ALEGRA:
- Mass density perturbation contours
- Used to study pinch instabilities and neutron production

**USim (Tech-X)**: 2D axisymmetric, two-temperature MHD.
- Visualization of mass density (fluids/q_0) on unstructured mesh
- Time slider for animation through discharge
- Standard pseudocolor on axisymmetric domain

**Gorgon (Imperial College)**: 3D Eulerian resistive MHD.
- Finite volume hydro with constrained transport for B-field
- Van Leer advection
- Used for wire array Z-pinches and DPF
- Volume rendering and density contour capabilities

**LSP (Voss Scientific)**: Electromagnetic PIC code.
- Particle density maps
- Used for initiation/breakdown phase
- PIC output imported to MHD codes for later phases

**Chicago (Voss Scientific)**: Hybrid fluid/PIC.
- Hundreds of kinetic plasma simulations run for MJOLNIR optimization

### Standard 2D r-z Cross-Section Layout

The overwhelming majority of published DPF simulations use **2D axisymmetric r-z cross-sections**:
- Horizontal axis: radial distance r (0 at axis, increasing outward)
- Vertical axis: axial distance z (0 at insulator, increasing toward open end)
- Anode shown as solid region at small r
- Cathode shown at outer boundary
- Pseudocolor fill for scalar fields
- Typical domain: r in [0, cathode_radius], z in [0, anode_length + overshoot]

### Colormaps in Plasma Physics Publications

There is **no single enforced standard**, but strong conventions exist:

**Density (rho or n_e)**:
- Almost always on **logarithmic scale** (log10)
- Sequential colormaps: `viridis`, `inferno`, or custom blue-green-yellow-red
- Published DPF papers frequently use blue (low) -> yellow/red (high)
- Range: typically 4-6 orders of magnitude (10^17 to 10^23 cm^-3)

**Electron temperature (T_e)**:
- Often on **logarithmic or linear scale** depending on range
- Hot = red/yellow/white, cold = blue/black — thermally intuitive
- `inferno` or `hot` colormaps common
- Range: 1 eV (cold boundary) to 1-10 keV (pinch core)

**Magnetic field (B_theta)**:
- **Diverging colormap** if showing signed components (blue-white-red)
- For magnitude |B|: sequential (similar to density)
- Streamlines or Line Integral Convolution (LIC) overlays for topology
- LIC is preferred over arrow/quiver plots for showing field structure — it reveals topology without needing to choose seed points

**Velocity (v)**:
- Arrow/quiver plots for direction and magnitude
- Streamlines for flow patterns
- Background pseudocolor for |v| magnitude
- Often overlaid on density or temperature plots

**Vector field rendering approaches** (ranked by information density):
1. **LIC (Line Integral Convolution)**: best for showing full topology. Used in VisIt, ParaView. Convolves noise texture along field lines to show structure everywhere.
2. **Streamlines**: good for showing field line geometry, but requires choosing seed points
3. **Arrow/quiver plots**: simple but low spatial resolution, can obscure underlying scalar data
4. **Hedgehog plots**: arrows at grid points, useful for 3D

### Perceptually Uniform Colormaps (Kenneth Moreland recommendations)

For scientific publication, **perceptually uniform** maps are strongly preferred:
- `viridis`: blue-green-yellow. Monotonic luminance. Colorblind-safe. Good default for density.
- `inferno`: black-purple-orange-yellow. High contrast. Good for temperature.
- `plasma`: blue-purple-yellow. Good for general sequential data.
- `magma`: black-purple-pink-yellow. Softer than inferno.
- `coolwarm` (diverging): blue-white-red. Good for signed quantities (velocity, B-field components).

Avoid: `jet`/`rainbow` — not perceptually uniform, creates false features, fails in grayscale.

---

## 3D vs 2D Visualization

### Published 3D DPF Work

True 3D renderings of DPF simulations are **rare but exist**:

- **ALEGRA 3D** (McBride et al., IEEE 2010): Full 3D DPF simulation showing filamentation during lift-off/rundown. 3D mass density rendered as isosurfaces or volume rendering. This is one of the few published 3D DPF visualizations and shows azimuthal asymmetry and filament structure that 2D cannot capture.

- **LA-COMPASS 3D** (LANL): 3D MHD simulations of DPF pinch formation, studying instabilities. Mass density perturbation visualized in 3D.

- **USim 3D**: Capable of 3D but published DPF examples are 2D axisymmetric.

- **Virtual Reality plasma visualization** (IEEE 2003 Lewandowski PPPL): Scientific visualization of plasma simulation results and device data in VR space. Particle-based rendering used for large-scale PIC simulations.

### What 3D Adds Over 2D

1. **Filamentation**: Azimuthal structure invisible in 2D axisymmetric. Filaments break the symmetry that 2D enforces.
2. **m=1 kink instability**: Requires 3D — the column bends helically, which is invisible in r-z.
3. **Ion beam trajectories**: Beams have 3D structure, especially if deflected by instabilities.
4. **Device context**: Showing the electrodes, insulator, and surrounding hardware gives physical intuition.
5. **Public communication**: 3D renderings are far more accessible to non-specialists.

### What 2D r-z Is Better For

1. **Quantitative comparison**: Side-by-side with published simulation data (almost all 2D)
2. **Radial profiles**: Clear radial structure of Bennett equilibrium
3. **Sheath tracking**: Shock front and piston positions directly visible
4. **Performance**: No need for 3D rendering pipeline

---

## Phase-by-Phase Renderer Specification

### Phase 1: Axial Rundown

| Element | Geometry | Data Source | Visual |
|---------|----------|-------------|--------|
| Current sheath | Annular shell, parabolic cross-section | Lee: z(t), r_inner=a, r_outer=b | Luminous blue-white surface |
| Sheath thickness | ~2-5 mm (kJ), ~1 cm (MJ) | Empirical | Gaussian density falloff normal to surface |
| Swept gas | Volume behind sheath | Lee: mass swept | Dim warm glow (ionized) |
| Undisturbed gas | Volume ahead of sheath | fill pressure | Dark/transparent |
| B_theta field | Concentric circles around z-axis | mu0*I/(2*pi*r) behind sheath | LIC texture or field lines |
| Electrodes | Anode (center cylinder), cathode (outer) | Device geometry | Metallic solid |
| Insulator | Base disk | Device geometry | Ceramic white |

### Phase 2: Radial Implosion

| Element | Geometry | Data Source | Visual |
|---------|----------|-------------|--------|
| Sheath/piston | Inward-converging cylinder + umbrella cap | Lee: r_p(t), z_f(t) | Luminous shell, brighter than Phase 1 |
| Shock front | Leads piston by ~1-2 mm | Lee: r_s(t) | Sharp, transparent density discontinuity |
| Pinch column | Growing axial extent | Lee: z_f(t) | Forming bright column |
| B_theta | Compressed between piston and axis | mu0*I/(2*pi*r), r < r_p | Intense concentric field lines |
| Reflected shock | After axis convergence | Lee: phase 3 onset | Outward-moving front |

### Phase 3: Pinch

| Element | Geometry | Data Source | Visual |
|---------|----------|-------------|--------|
| Pinch column | Cylinder, r ~ r_min, height ~ z_f | Lee: r_min, z_f; MHD: rho(r,z) | Intensely bright, white-hot core |
| Bennett density profile | n(r) = n0/(1+r^2/a^2)^2 | MHD: rho(r,z); or analytic | Density colormap (log scale) |
| Temperature profile | Flat core, steep edge gradient | MHD: Te(r,z); or analytic | Temperature colormap (inferno) |
| B_theta | Peaks at r=a, falls as 1/r outside | MHD: B(r,z) or analytic | LIC overlay |
| Poloidal B (axial) | Threading column along z | Auluck theory | Field lines along axis |
| Radiation | Bremsstrahlung from hot core | Lee: P_brem(t) | Volumetric glow, brightest at center |
| Early m=0 | Sinusoidal density perturbation | Analytic or MHD | Slight column waviness |

### Phase 4: Post-Pinch

| Element | Geometry | Data Source | Visual |
|---------|----------|-------------|--------|
| m=0 sausage | Growing amplitude sine perturbation | Analytic: lambda ~ 2*pi*a | Column necking and bulging |
| Plasmoids | Discrete bright spots after breakup | MHD or analytic | Isolated bright blobs |
| Ion beam | Narrow cone along z-axis from neck | Analytic trajectory | Particle stream or glow cone |
| X-ray hotspots | At compression maxima | Peak density/temp locations | Bright point sources |
| Expanding plasma | Column radius growing | Lee: expanded column phase | Diffuse, dimming glow |
| Beam electrons | Toward anode | Accelerated by same E-field | Downward glow/cone |

---

## Key References

### Experimental Imaging
- Krishnan M (2012) "The Dense Plasma Focus: A Versatile Dense Pinch for Diverse Applications" IEEE TPS 40(12):3189-3221 — comprehensive review with diagnostic descriptions
- IEEE 2024 MJOLNIR schlieren/interferometry: laser Schlieren + interferometry for electron density in DPF pinch
- Stanford Computer Optics: 200 ps ICCD imaging of DPF filaments and plasmoid formation at 30 um resolution
- Damideh et al (2025) FAETON-I: current/voltage correlated with radial trajectories, 194 kV pinch voltage

### Simulation Visualization
- McBride & Stamm (2010) "PIC/MHD modeling of DPF in 2D and 3D" IEEE — ALEGRA 3D filamentation
- ALEGRA-HEDP (Sandia): 2D density contour plots of rundown and pinch, validated against experiments
- LA-COMPASS (LANL): 2D/3D MHD, mass density perturbation visualization
- USim (Tech-X): 2D axisymmetric MHD, mass density visualization on unstructured mesh
- Gorgon (Imperial): 3D Eulerian resistive MHD with volume rendering

### Sheath Geometry
- Gratton & Vargas (1983): Analytic 2D snowplow model defining sheath surface shape
- Auluck (2013) PoP 20:112501 — re-appraisal of Gratton-Vargas, curvilinear coordinate system on sheath
- Auluck (2017) PoP 24:112502 — axial B-field from sheath curvature conservation laws
- Auluck (2024) PoP 31:010704 — poloidal magnetic field in DPF

### Instabilities and Ion Beams
- Book (1976) Phys. Fluids 19:1982 — nonlinear m=0 sausage evolution, spindle/spike-and-bubble shape
- Auluck (2019) APS DPP — filamentation instability in DPF
- Physics of Plasmas (2025) 32:033105 — neutron generation dynamics, beam-target vs thermonuclear mechanisms

### Visualization Methods
- Moreland K — "Color Map Advice for Scientific Visualization" (kennethmoreland.com)
- Cabral & Leedom (1993) — Line Integral Convolution for vector field visualization
- Kulhanek & Smetana — "Visualization techniques in plasma numerical simulations" Czech J Phys

### Lee Model
- Lee S — plasmafocus.net/IPFS/modelpackage/File1RADPF.htm — full RADPF model documentation
- Lee (2014) J Fusion Energy 33 — "Plasma Focus Radiative Model: Review of the Lee Model Code"
