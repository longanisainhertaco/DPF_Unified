import React, { useState, useMemo } from 'react';
import { ScientificInput } from './ScientificInput';
import { CommitButton } from './CommitButton';
import { PresetSelector } from './PresetSelector';
import { useConfigStore } from '../../stores/config';
import { useSimulationStore } from '../../stores/simulation';
import { useAIStore } from '../../stores/ai';

// ── Gas species data ────────────────────────────────────────
const GAS_SPECIES = [
  { name: 'Deuterium', symbol: 'D\u2082', A_amu: 2.014, ion_mass: 3.344e-27, Z: 1 },
  { name: 'Hydrogen', symbol: 'H\u2082', A_amu: 1.008, ion_mass: 1.673e-27, Z: 1 },
  { name: 'Helium', symbol: 'He', A_amu: 4.003, ion_mass: 6.647e-27, Z: 2 },
  { name: 'Nitrogen', symbol: 'N\u2082', A_amu: 14.01, ion_mass: 2.326e-26, Z: 7 },
  { name: 'Neon', symbol: 'Ne', A_amu: 20.18, ion_mass: 3.351e-26, Z: 10 },
  { name: 'Argon', symbol: 'Ar', A_amu: 39.95, ion_mass: 6.634e-26, Z: 18 },
  { name: 'Krypton', symbol: 'Kr', A_amu: 83.80, ion_mass: 1.392e-25, Z: 36 },
  { name: 'Xenon', symbol: 'Xe', A_amu: 131.3, ion_mass: 2.180e-25, Z: 54 },
];

const K_B = 1.380649e-23; // Boltzmann constant
const AMU = 1.66053906660e-27; // atomic mass unit in kg

function pressureToRho(pressureTorr: number, A_amu: number, T: number): number {
  // rho = (P_torr * 133.322 Pa/Torr) * A_kg / (k_B * T)
  return (pressureTorr * 133.322) * (A_amu * AMU) / (K_B * T);
}

function rhoToPressure(rho: number, A_amu: number, T: number): number {
  return rho * K_B * T / ((A_amu * AMU) * 133.322);
}

export const ParameterStack: React.FC = () => {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    preset: true,
    bank: true,
    geometry: true,
    gas: true,
    grid: false,
    solver: false,
    radiation: false,
    transport: false,
    boundary: false,
    diagnostics: false,
  });
  const [selectedPresetName, setSelectedPresetName] = useState<string | null>(null);

  // ── Config store ────────────────────────────────────────────
  const config = useConfigStore((s) => s.config);
  const presets = useConfigStore((s) => s.presets);
  const isValid = useConfigStore((s) => s.isValid);
  const isArmed = useConfigStore((s) => s.isArmed);
  const errors = useConfigStore((s) => s.errors);
  const loadPreset = useConfigStore((s) => s.loadPreset);
  const updateCircuit = useConfigStore((s) => s.updateCircuit);
  const updateGrid = useConfigStore((s) => s.updateGrid);
  const updateFluid = useConfigStore((s) => s.updateFluid);
  const updateRadiation = useConfigStore((s) => s.updateRadiation);
  const updateCollision = useConfigStore((s) => s.updateCollision);
  const updateSheath = useConfigStore((s) => s.updateSheath);
  const updateBoundary = useConfigStore((s) => s.updateBoundary);
  const updateDiagnostics = useConfigStore((s) => s.updateDiagnostics);
  const updateGeometry = useConfigStore((s) => s.updateGeometry);
  const validate = useConfigStore((s) => s.validate);
  const commit = useConfigStore((s) => s.commit);
  const resetConfig = useConfigStore((s) => s.reset);

  // ── Simulation store ────────────────────────────────────────
  const simStatus = useSimulationStore((s) => s.status);
  const startSim = useSimulationStore((s) => s.start);
  const stopSim = useSimulationStore((s) => s.stop);
  const resetSim = useSimulationStore((s) => s.reset);

  // ── AI store ────────────────────────────────────────────────
  const generateAdvisory = useAIStore((s) => s.generatePreShotAdvisory);

  // ── Derived state ───────────────────────────────────────────
  const isRunning = simStatus === 'running' || simStatus === 'paused';
  const isFinished = simStatus === 'finished';
  const locked = isRunning || isFinished;

  // Extract current config values for form display
  const circuit = config.circuit ?? { C: 1e-6, V0: 1e3, L0: 1e-7, R0: 0.01, anode_radius: 0.005, cathode_radius: 0.01, ESR: 0.0, ESL: 0.0 };
  const gridShape = config.grid_shape ?? [8, 8, 8];
  const geometryType = config.geometry?.type ?? 'cylindrical';
  const fluid = config.fluid ?? {};
  const radiation = config.radiation ?? {};
  const collision = config.collision ?? {};
  const sheath = config.sheath ?? {};
  const boundary = config.boundary ?? {};
  const diagnostics = config.diagnostics ?? {};

  // Gas species detection from ion_mass
  const ionMass = config.ion_mass ?? 3.344e-27;
  const detectedSpecies = useMemo(() => {
    return GAS_SPECIES.find(s => Math.abs(s.ion_mass - ionMass) / ionMass < 0.05) ?? GAS_SPECIES[0];
  }, [ionMass]);

  const fillPressureTorr = useMemo(() => {
    return rhoToPressure(config.rho0 ?? 1e-4, detectedSpecies.A_amu, config.T0 ?? 300);
  }, [config.rho0, config.T0, detectedSpecies]);

  // ── Handlers ────────────────────────────────────────────────

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const handleArm = async () => {
    const valid = await validate();
    if (valid) {
      generateAdvisory(config);
    }
  };

  const handleFire = async () => {
    const simId = await commit();
    if (simId) {
      await startSim(simId);
    }
  };

  const handleStop = async () => {
    try {
      await stopSim();
    } catch (e) {
      console.error('Stop failed:', e);
    }
  };

  const handleReset = () => {
    resetSim();
    resetConfig();
  };

  const handlePresetSelect = (name: string) => {
    setSelectedPresetName(name);
    loadPreset(name);
  };

  const handleGasSpeciesChange = (speciesName: string) => {
    const species = GAS_SPECIES.find(s => s.name === speciesName);
    if (!species) return;
    const T = config.T0 ?? 300;
    const newRho = pressureToRho(fillPressureTorr, species.A_amu, T);
    updateGrid({ rho0: newRho, ion_mass: species.ion_mass } as any);
  };

  const handlePressureChange = (pressureTorr: number) => {
    const T = config.T0 ?? 300;
    const newRho = pressureToRho(pressureTorr, detectedSpecies.A_amu, T);
    updateGrid({ rho0: newRho } as any);
  };

  return (
    <div className="h-full flex flex-col bg-[#1E1E1E]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#333333]">
        <h2 className="dpf-label text-sm font-semibold tracking-wide">PARAMETERS</h2>
      </div>

      {/* Scrollable Parameter Sections */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {/* Preset Selector */}
        <Section
          title="Preset"
          expanded={expandedSections.preset}
          onToggle={() => toggleSection('preset')}
        >
          <PresetSelector
            presets={presets}
            selectedPreset={selectedPresetName}
            onSelect={handlePresetSelect}
            loading={isRunning}
          />
        </Section>

        {/* ── BANK ─────────────────────────────────────────── */}
        <Section
          title="Bank"
          expanded={expandedSections.bank}
          onToggle={() => toggleSection('bank')}
        >
          <div className="space-y-2">
            <ScientificInput
              label="Capacitance"
              value={circuit.C}
              unit="F"
              onChange={(v) => updateCircuit({ C: v })}
              validation="valid"
              disabled={locked}
              tooltip="Energy storage capacitance of the capacitor bank. E = ½CV²."
            />
            <ScientificInput
              label="Charge Voltage"
              value={circuit.V0}
              unit="V"
              onChange={(v) => updateCircuit({ V0: v })}
              validation="valid"
              disabled={locked}
              tooltip="Initial charge voltage across the capacitor bank before discharge."
            />
            <ScientificInput
              label="Inductance"
              value={circuit.L0}
              unit="H"
              onChange={(v) => updateCircuit({ L0: v })}
              validation="valid"
              disabled={locked}
              tooltip="External circuit inductance. Limits peak current via I_peak ≈ V₀√(C/L₀)."
            />
            <ScientificInput
              label="Resistance"
              value={circuit.R0}
              unit="Ω"
              onChange={(v) => updateCircuit({ R0: v })}
              validation="valid"
              disabled={locked}
              tooltip="External circuit resistance. Damps current oscillations and dissipates energy."
            />
            <ScientificInput
              label="ESR (Parasitic R)"
              value={circuit.ESR ?? 0}
              unit="Ω"
              onChange={(v) => updateCircuit({ ESR: v })}
              disabled={locked}
              tooltip="Equivalent Series Resistance of the capacitor bank — parasitic losses in leads and contacts."
            />
            <ScientificInput
              label="ESL (Parasitic L)"
              value={circuit.ESL ?? 0}
              unit="H"
              onChange={(v) => updateCircuit({ ESL: v })}
              disabled={locked}
              tooltip="Equivalent Series Inductance — stray inductance from bus bars and connections."
            />
            <ScientificInput
              label="Anomalous α"
              value={config.anomalous_alpha ?? 0.05}
              unit=""
              onChange={(v) => updateGrid({ anomalous_alpha: v } as any)}
              min={0}
              max={1}
              disabled={locked}
              tooltip="Anomalous resistivity coefficient — enhances Ohmic heating near the pinch column."
            />
          </div>
        </Section>

        {/* ── GEOMETRY ──────────────────────────────────────── */}
        <Section
          title="Geometry"
          expanded={expandedSections.geometry}
          onToggle={() => toggleSection('geometry')}
        >
          <div className="space-y-2">
            <div className="mb-3">
              <label className="dpf-label text-xs mb-2 block">Type</label>
              <div className="flex gap-2">
                <button
                  onClick={() => updateGeometry({ type: 'cylindrical' })}
                  disabled={locked}
                  className={`flex-1 py-2 px-3 rounded text-xs font-medium transition-all ${
                    geometryType === 'cylindrical'
                      ? 'bg-[#00E5FF] text-[#121212]'
                      : 'bg-[#2A2A2A] text-[#999999] hover:bg-[#333333]'
                  } ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  Cylindrical
                </button>
                <button
                  onClick={() => updateGeometry({ type: 'cartesian' })}
                  disabled={locked}
                  className={`flex-1 py-2 px-3 rounded text-xs font-medium transition-all ${
                    geometryType === 'cartesian'
                      ? 'bg-[#00E5FF] text-[#121212]'
                      : 'bg-[#2A2A2A] text-[#999999] hover:bg-[#333333]'
                  } ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  Cartesian
                </button>
              </div>
            </div>
            <ScientificInput
              label="Anode Radius"
              value={circuit.anode_radius}
              unit="m"
              onChange={(v) => updateCircuit({ anode_radius: v })}
              validation="valid"
              disabled={locked}
              tooltip="Inner electrode radius — defines the plasma column geometry."
            />
            <ScientificInput
              label="Cathode Radius"
              value={circuit.cathode_radius}
              unit="m"
              onChange={(v) => updateCircuit({ cathode_radius: v })}
              validation="valid"
              disabled={locked}
              tooltip="Outer electrode radius — sets the annular gap for the current sheath."
            />
          </div>
        </Section>

        {/* ── GAS ───────────────────────────────────────────── */}
        <Section
          title="Gas"
          expanded={expandedSections.gas}
          onToggle={() => toggleSection('gas')}
        >
          <div className="space-y-2">
            <div>
              <label className="dpf-label text-xs mb-2 block">Gas Species</label>
              <select
                value={detectedSpecies.name}
                onChange={(e) => handleGasSpeciesChange(e.target.value)}
                disabled={locked}
                className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {GAS_SPECIES.map((s) => (
                  <option key={s.name} value={s.name}>{s.symbol} ({s.name}) — Z={s.Z}, A={s.A_amu}</option>
                ))}
              </select>
            </div>
            <ScientificInput
              label="Fill Pressure"
              value={fillPressureTorr}
              unit="Torr"
              onChange={handlePressureChange}
              disabled={locked}
              tooltip="Gas fill pressure — converted to mass density via ideal gas law: ρ = P·A/(k_B·T)."
            />
            <ScientificInput
              label="Initial Density"
              value={config.rho0 ?? 1e-4}
              unit="kg/m³"
              onChange={(v) => updateGrid({ rho0: v })}
              disabled={locked}
              tooltip="Initial mass density of the fill gas. Set automatically when fill pressure changes."
            />
            <ScientificInput
              label="Ion Mass"
              value={ionMass}
              unit="kg"
              onChange={(v) => updateGrid({ ion_mass: v } as any)}
              disabled={locked}
              tooltip="Mass of a single ion. Set automatically from gas species selection."
            />
            <ScientificInput
              label="Initial Temperature"
              value={config.T0 ?? 300}
              unit="K"
              onChange={(v) => updateGrid({ T0: v })}
              validation="valid"
              disabled={locked}
              tooltip="Initial gas temperature — affects breakdown and initial density calculation."
            />
          </div>
        </Section>

        {/* ── GRID ──────────────────────────────────────────── */}
        <Section
          title="Grid"
          expanded={expandedSections.grid}
          onToggle={() => toggleSection('grid')}
        >
          <div className="space-y-2">
            <div>
              <label className="dpf-label text-xs mb-2 block">Grid Shape [nx, ny, nz]</label>
              <div className="flex gap-2">
                {[0, 1, 2].map((i) => (
                  <input
                    key={i}
                    type="number"
                    value={gridShape[i]}
                    onChange={(e) => {
                      const newShape = [...gridShape];
                      newShape[i] = parseInt(e.target.value) || 8;
                      updateGrid({ grid_shape: newShape });
                    }}
                    disabled={locked}
                    className={`dpf-input flex-1 text-center font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
                    min="4"
                  />
                ))}
              </div>
            </div>
            <ScientificInput
              label="Grid Spacing"
              value={config.dx ?? 1e-3}
              unit="m"
              onChange={(v) => updateGrid({ dx: v })}
              validation="valid"
              disabled={locked}
              tooltip="Cell size in each dimension. Smaller values increase resolution but slow computation."
            />
            <ScientificInput
              label="Simulation Time"
              value={config.sim_time ?? 1e-6}
              unit="s"
              onChange={(v) => updateGrid({ sim_time: v })}
              validation="valid"
              disabled={locked}
              tooltip="Total simulated time. Typical DPF shots: 1–10 μs."
            />
          </div>
        </Section>

        {/* ── SOLVER ────────────────────────────────────────── */}
        <Section
          title="Solver"
          expanded={expandedSections.solver}
          onToggle={() => toggleSection('solver')}
        >
          <div className="space-y-2">
            <div>
              <label className="dpf-label text-xs mb-2 block">Backend</label>
              <select
                value={fluid.backend ?? 'python'}
                onChange={(e) => updateFluid({ backend: e.target.value as any })}
                disabled={locked}
                className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <option value="python">Python (NumPy/Numba)</option>
                <option value="athena">Athena++ (C++)</option>
                <option value="athenak">AthenaK (Kokkos/GPU)</option>
                <option value="metal">Metal GPU (Apple Silicon)</option>
                <option value="auto">Auto (best available)</option>
              </select>
            </div>
            <div>
              <label className="dpf-label text-xs mb-2 block">Reconstruction</label>
              <select
                value={fluid.reconstruction ?? 'weno5'}
                onChange={(e) => updateFluid({ reconstruction: e.target.value })}
                disabled={locked}
                className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <option value="weno5">WENO5-Z (5th order)</option>
                <option value="plm">PLM (2nd order)</option>
                <option value="ppm">PPM (3rd order)</option>
              </select>
            </div>
            <div>
              <label className="dpf-label text-xs mb-2 block">Riemann Solver</label>
              <select
                value={fluid.riemann_solver ?? 'hlld'}
                onChange={(e) => updateFluid({ riemann_solver: e.target.value })}
                disabled={locked}
                className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <option value="hll">HLL (robust, 2-wave)</option>
                <option value="hlld">HLLD (accurate, 4-wave MHD)</option>
              </select>
            </div>
            <div>
              <label className="dpf-label text-xs mb-2 block">Time Integrator</label>
              <select
                value={fluid.time_integrator ?? 'ssp_rk3'}
                onChange={(e) => updateFluid({ time_integrator: e.target.value })}
                disabled={locked}
                className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <option value="ssp_rk2">SSP-RK2 (2nd order)</option>
                <option value="ssp_rk3">SSP-RK3 (3rd order, recommended)</option>
              </select>
            </div>
            <div>
              <label className="dpf-label text-xs mb-2 block">Precision</label>
              <select
                value={fluid.precision ?? 'float32'}
                onChange={(e) => updateFluid({ precision: e.target.value })}
                disabled={locked}
                className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <option value="float32">Float32 (fast, GPU compatible)</option>
                <option value="float64">Float64 (accurate, CPU only)</option>
              </select>
            </div>
            <ScientificInput
              label="CFL Number"
              value={fluid.cfl ?? 0.4}
              unit=""
              onChange={(v) => updateFluid({ cfl: v })}
              min={0.01}
              max={0.99}
              disabled={locked}
              tooltip="Courant stability factor — lower is more stable, higher is faster. Must be < 1."
            />
            <ScientificInput
              label="γ (Adiabatic Index)"
              value={fluid.gamma ?? 1.6667}
              unit=""
              onChange={(v) => updateFluid({ gamma: v })}
              min={1.01}
              max={3.0}
              disabled={locked}
              tooltip="Ratio of specific heats. 5/3 for monatomic ideal gas, 7/5 for diatomic."
            />
            <div>
              <label className="dpf-label text-xs mb-2 block">Diffusion Method</label>
              <select
                value={fluid.diffusion_method ?? 'explicit'}
                onChange={(e) => updateFluid({ diffusion_method: e.target.value })}
                disabled={locked}
                className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <option value="explicit">Explicit (CFL-limited)</option>
                <option value="sts">STS (RKL2 super time-stepping)</option>
                <option value="implicit">Implicit (ADI)</option>
              </select>
            </div>
            <ToggleSwitch label="Resistive MHD" checked={fluid.enable_resistive ?? true} onChange={(v) => updateFluid({ enable_resistive: v })} disabled={locked} tooltip="Enable Ohmic resistivity (η·J²). Required for current diffusion and Ohmic heating." />
            <ToggleSwitch label="Viscosity" checked={fluid.enable_viscosity ?? false} onChange={(v) => updateFluid({ enable_viscosity: v })} disabled={locked} tooltip="Enable scalar viscous dissipation — smooths velocity gradients." />
            <ToggleSwitch label="Energy Equation" checked={fluid.enable_energy_equation ?? true} onChange={(v) => updateFluid({ enable_energy_equation: v })} disabled={locked} tooltip="Evolve internal energy. Disable for isothermal simulations." />
            <ToggleSwitch label="Nernst Effect" checked={fluid.enable_nernst ?? false} onChange={(v) => updateFluid({ enable_nernst: v })} disabled={locked} tooltip="Thermoelectric B-field advection by electron heat flux. Important at high Te gradients." />
            <ToggleSwitch label="Powell 8-wave" checked={fluid.enable_powell ?? false} onChange={(v) => updateFluid({ enable_powell: v })} disabled={locked} tooltip="Powell divergence cleaning — adds source terms to reduce ∇·B errors." />
            <ToggleSwitch label="Anisotropic Conduction" checked={fluid.enable_anisotropic_conduction ?? false} onChange={(v) => updateFluid({ enable_anisotropic_conduction: v })} disabled={locked} tooltip="Heat conducts primarily along B-field lines (Braginskii κ∥ >> κ⊥)." />
            <ToggleSwitch label="Full Braginskii Viscosity" checked={fluid.full_braginskii_viscosity ?? false} onChange={(v) => updateFluid({ full_braginskii_viscosity: v })} disabled={locked} tooltip="Full anisotropic viscous stress tensor with parallel and gyroviscous components." />
            <ToggleSwitch label="Constrained Transport" checked={fluid.use_ct ?? false} onChange={(v) => updateFluid({ use_ct: v })} disabled={locked} tooltip="Maintain div(B)=0 to machine precision via CT flux averaging. Metal GPU + MPS device required." />
          </div>
        </Section>

        {/* ── RADIATION ─────────────────────────────────────── */}
        <Section
          title="Radiation"
          expanded={expandedSections.radiation}
          onToggle={() => toggleSection('radiation')}
        >
          <div className="space-y-2">
            <ToggleSwitch label="Bremsstrahlung" checked={radiation.bremsstrahlung_enabled ?? true} onChange={(v) => updateRadiation({ bremsstrahlung_enabled: v })} disabled={locked} tooltip="Free-free radiation from electron-ion collisions. Dominant loss at high Te." />
            <ScientificInput label="Gaunt Factor" value={radiation.gaunt_factor ?? 1.2} unit="" onChange={(v) => updateRadiation({ gaunt_factor: v })} disabled={locked} tooltip="Quantum correction to bremsstrahlung power. Typically 1.0–1.5." />
            <ToggleSwitch label="Flux-Limited Diffusion" checked={radiation.fld_enabled ?? false} onChange={(v) => updateRadiation({ fld_enabled: v })} disabled={locked} tooltip="FLD radiation transport — approximates photon diffusion with a flux limiter." />
            <ScientificInput label="Flux Limiter λ" value={radiation.flux_limiter ?? 0.333} unit="" onChange={(v) => updateRadiation({ flux_limiter: v })} min={0} max={1} disabled={locked} tooltip="Radiation flux limiter. 1/3 (Levermore-Pomraning) is standard." />
            <ToggleSwitch label="Line Radiation" checked={radiation.line_radiation_enabled ?? false} onChange={(v) => updateRadiation({ line_radiation_enabled: v })} disabled={locked} tooltip="Bound-bound line emission from impurity ions. Requires impurity species." />
            <ScientificInput label="Impurity Z" value={radiation.impurity_Z ?? 29} unit="" onChange={(v) => updateRadiation({ impurity_Z: v })} min={1} max={74} disabled={locked} tooltip="Atomic number of trace impurity species (e.g., Cu=29, W=74)." />
            <ScientificInput label="Impurity Fraction" value={radiation.impurity_fraction ?? 0} unit="" onChange={(v) => updateRadiation({ impurity_fraction: v })} min={0} max={1} disabled={locked} tooltip="Number fraction of impurity ions relative to main gas species." />
          </div>
        </Section>

        {/* ── TRANSPORT ─────────────────────────────────────── */}
        <Section
          title="Transport"
          expanded={expandedSections.transport}
          onToggle={() => toggleSection('transport')}
        >
          <div className="space-y-2">
            <ScientificInput label="Coulomb Logarithm" value={collision.coulomb_log ?? 10} unit="" onChange={(v) => updateCollision({ coulomb_log: v })} disabled={locked} tooltip="ln(Λ) — logarithmic measure of the ratio of max to min impact parameters in Coulomb collisions." />
            <ToggleSwitch label="Dynamic Coulomb Log" checked={collision.dynamic_coulomb_log ?? true} onChange={(v) => updateCollision({ dynamic_coulomb_log: v })} disabled={locked} tooltip="Recompute ln(Λ) from local Te and ne each timestep instead of using a fixed value." />
            <ScientificInput label="σ_en (e-n Cross Section)" value={collision.sigma_en ?? 1e-19} unit="m²" onChange={(v) => updateCollision({ sigma_en: v })} disabled={locked} tooltip="Electron-neutral collision cross section. Important at low ionization fractions." />
          </div>
        </Section>

        {/* ── BOUNDARY ──────────────────────────────────────── */}
        <Section
          title="Boundary"
          expanded={expandedSections.boundary}
          onToggle={() => toggleSection('boundary')}
        >
          <div className="space-y-2">
            <ToggleSwitch label="Electrode BC" checked={boundary.electrode_bc ?? false} onChange={(v) => updateBoundary({ electrode_bc: v })} disabled={locked} tooltip="Apply conducting-wall boundary conditions at the electrode surfaces." />
            <ToggleSwitch label="Axis BC (r=0)" checked={boundary.axis_bc ?? true} onChange={(v) => updateBoundary({ axis_bc: v })} disabled={locked} tooltip="Symmetry boundary condition on the cylindrical axis (r = 0)." />
            <ToggleSwitch label="Sheath BC" checked={sheath.enabled ?? false} onChange={(v) => updateSheath({ enabled: v })} disabled={locked} tooltip="Enable plasma sheath boundary layer model at electrode surfaces." />
            {sheath.enabled && (
              <>
                <div>
                  <label className="dpf-label text-xs mb-2 block">Sheath Boundary</label>
                  <select
                    value={sheath.boundary ?? 'z_high'}
                    onChange={(e) => updateSheath({ boundary: e.target.value })}
                    disabled={locked}
                    className={`dpf-input w-full text-xs font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <option value="z_high">z_high (cathode)</option>
                    <option value="z_low">z_low (anode)</option>
                  </select>
                </div>
                <ScientificInput label="Sheath Voltage" value={sheath.V_sheath ?? 0} unit="V" onChange={(v) => updateSheath({ V_sheath: v })} disabled={locked} tooltip="Voltage drop across the plasma sheath layer at the boundary." />
              </>
            )}
          </div>
        </Section>

        {/* ── DIAGNOSTICS ───────────────────────────────────── */}
        <Section
          title="Diagnostics"
          expanded={expandedSections.diagnostics}
          onToggle={() => toggleSection('diagnostics')}
        >
          <div className="space-y-2">
            <div>
              <label className="dpf-label text-xs mb-2 block">Output Interval (steps)</label>
              <input
                type="number"
                value={diagnostics.output_interval ?? 10}
                onChange={(e) => updateDiagnostics({ output_interval: parseInt(e.target.value) || 10 })}
                disabled={locked}
                className={`dpf-input w-full text-center font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
                min="1"
              />
            </div>
            <div>
              <label className="dpf-label text-xs mb-2 block">Field Snapshot Interval (0=off)</label>
              <input
                type="number"
                value={diagnostics.field_output_interval ?? 0}
                onChange={(e) => updateDiagnostics({ field_output_interval: parseInt(e.target.value) || 0 })}
                disabled={locked}
                className={`dpf-input w-full text-center font-mono ${locked ? 'opacity-50 cursor-not-allowed' : ''}`}
                min="0"
              />
            </div>
          </div>
        </Section>
      </div>

      {/* Footer: Commit Button + Errors */}
      <div className="px-4 py-4 border-t border-[#333333]">
        <CommitButton
          isValid={isValid}
          isArmed={isArmed}
          isRunning={isRunning}
          finished={isFinished}
          onArm={handleArm}
          onFire={handleFire}
          onStop={handleStop}
          onReset={handleReset}
        />
        {errors.length > 0 && (
          <div className="mt-2 space-y-1">
            {errors.map((err, i) => (
              <p key={i} className="text-xs text-red-400 font-mono">{err}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Section Header Component
interface SectionProps {
  title: string;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({ title, expanded, onToggle, children }) => (
  <div className="bg-[#2A2A2A] rounded-lg border border-[#333333] overflow-hidden">
    <button
      onClick={onToggle}
      className="w-full px-3 py-2 flex items-center justify-between hover:bg-[#333333] transition-colors"
    >
      <span className="dpf-label text-xs font-semibold tracking-wide">{title}</span>
      <svg
        className={`w-4 h-4 text-[#666666] transition-transform ${
          expanded ? 'rotate-180' : ''
        }`}
        fill="currentColor"
        viewBox="0 0 20 20"
      >
        <path
          fillRule="evenodd"
          d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
          clipRule="evenodd"
        />
      </svg>
    </button>
    {expanded && <div className="px-3 pb-3">{children}</div>}
  </div>
);

// Toggle Switch Component
interface ToggleSwitchProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  tooltip?: string;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ label, checked, onChange, disabled = false, tooltip }) => (
  <div className="flex items-center justify-between">
    <span className="dpf-label text-xs">
      {tooltip ? (
        <span data-tooltip={tooltip} className="cursor-help border-b border-dotted border-gray-600">
          {label}
        </span>
      ) : (
        label
      )}
    </span>
    <button
      onClick={() => !disabled && onChange(!checked)}
      disabled={disabled}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
        checked ? 'bg-[#00E5FF]' : 'bg-[#333333]'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-5' : 'translate-x-1'
        }`}
      />
    </button>
  </div>
);
