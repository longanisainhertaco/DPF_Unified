import { create } from 'zustand';
import { fetchPresets, validateConfig, createSimulation } from '../api/client';
import type {
  SimulationConfig,
  PresetInfo,
  ConfigValidationResponse,
  CollisionConfig,
  RadiationConfig,
  SheathConfig,
  BoundaryConfig,
  DiagnosticsConfig,
  FluidConfig,
  GeometryConfig,
} from '../api/types';

interface ConfigState {
  // Current form values (partial config)
  config: Partial<SimulationConfig>;

  // Available presets
  presets: PresetInfo[];

  // Validation state
  errors: string[];
  isValid: boolean;
  isArmed: boolean;

  // Last committed simulation ID
  simId: string | null;

  // Actions
  setPresets: (presets: PresetInfo[]) => void;
  loadPreset: (name: string) => Promise<void>;
  updateCircuit: (partial: Partial<SimulationConfig['circuit']>) => void;
  updateFluid: (partial: Partial<NonNullable<SimulationConfig['fluid']>>) => void;
  updateGrid: (partial: { grid_shape?: number[]; dx?: number; sim_time?: number; rho0?: number; T0?: number }) => void;
  updateCollision: (partial: Partial<NonNullable<SimulationConfig['collision']>>) => void;
  updateRadiation: (partial: Partial<NonNullable<SimulationConfig['radiation']>>) => void;
  updateSheath: (partial: Partial<NonNullable<SimulationConfig['sheath']>>) => void;
  updateBoundary: (partial: Partial<NonNullable<SimulationConfig['boundary']>>) => void;
  updateDiagnostics: (partial: Partial<NonNullable<SimulationConfig['diagnostics']>>) => void;
  updateGeometry: (partial: Partial<NonNullable<SimulationConfig['geometry']>>) => void;
  validate: () => Promise<boolean>;
  commit: () => Promise<string | null>;
  reset: () => void;
}

const DEFAULT_CONFIG: Partial<SimulationConfig> = {
  grid_shape: [8, 8, 8],
  dx: 1e-3,
  sim_time: 1e-7,
  dt_init: 1e-10,
  rho0: 1e-4,
  T0: 300,
  anomalous_alpha: 0.05,
  ion_mass: 3.34358377e-27,
  circuit: {
    C: 1e-6,
    V0: 1e3,
    L0: 1e-7,
    R0: 0.01,
    anode_radius: 0.005,
    cathode_radius: 0.01,
    ESR: 0.0,
    ESL: 0.0,
  },
  collision: {
    coulomb_log: 10.0,
    dynamic_coulomb_log: true,
    sigma_en: 1e-19,
  },
  radiation: {
    bremsstrahlung_enabled: true,
    gaunt_factor: 1.2,
    fld_enabled: false,
    flux_limiter: 0.333,
    line_radiation_enabled: false,
    impurity_Z: 29,
    impurity_fraction: 0.0,
  },
  sheath: {
    enabled: false,
    boundary: 'z_high',
    V_sheath: 0.0,
  },
  geometry: {
    type: 'cartesian',
  },
  fluid: {
    backend: 'python',
    reconstruction: 'weno5',
    riemann_solver: 'hll',
    cfl: 0.4,
    gamma: 1.6667,
    dedner_ch: 0.0,
    dedner_cr: 0.0,
    enable_resistive: true,
    enable_viscosity: false,
    enable_energy_equation: true,
    enable_nernst: false,
    enable_powell: false,
    enable_anisotropic_conduction: false,
    full_braginskii_viscosity: false,
    diffusion_method: 'explicit',
    sts_stages: 8,
    implicit_tol: 1e-8,
  },
  boundary: {
    electrode_bc: false,
    axis_bc: true,
  },
  diagnostics: {
    output_interval: 10,
    field_output_interval: 0,
  },
};

export const useConfigStore = create<ConfigState>((set, get) => ({
  config: DEFAULT_CONFIG,
  presets: [],
  errors: [],
  isValid: false,
  isArmed: false,
  simId: null,

  setPresets: (presets: PresetInfo[]) => {
    set({ presets });
  },

  loadPreset: async (name: string) => {
    try {
      // Look up preset info for grid_shape/geometry data
      const { presets } = get();
      const preset = presets.find((p) => p.name === name);

      if (!preset) {
        throw new Error(`Preset "${name}" not found`);
      }

      // Preset configs are known — map preset name to config values
      // The actual config will be sent to the server via the "preset" field
      // when creating a simulation. For now, update the form with preset metadata.
      const presetConfigs: Record<string, Partial<SimulationConfig>> = {
        tutorial: {
          grid_shape: [8, 8, 8], dx: 1e-3, sim_time: 1e-7, dt_init: 1e-10,
          rho0: 1e-4, T0: 300,
          circuit: { C: 1e-6, V0: 1e3, L0: 1e-7, R0: 0.01, anode_radius: 0.005, cathode_radius: 0.01, ESR: 0.0, ESL: 0.0 },
          boundary: { electrode_bc: false, axis_bc: false },
        },
        pf1000: {
          grid_shape: [64, 1, 128], dx: 5e-4, sim_time: 5e-6, dt_init: 1e-10,
          rho0: 4e-4, T0: 300, anomalous_alpha: 0.05,
          circuit: { C: 1.332e-3, V0: 27e3, L0: 15e-9, R0: 3e-3, anode_radius: 0.0575, cathode_radius: 0.08, ESR: 0.0, ESL: 0.0 },
          geometry: { type: 'cylindrical' },
          radiation: { bremsstrahlung_enabled: true, fld_enabled: true },
          collision: { coulomb_log: 10.0, dynamic_coulomb_log: true, sigma_en: 1e-19 },
          fluid: { backend: 'python' },
          boundary: { electrode_bc: true, axis_bc: true },
        },
        nx2: {
          grid_shape: [32, 1, 64], dx: 2e-4, sim_time: 1e-6, dt_init: 1e-11,
          rho0: 8e-5, T0: 300, anomalous_alpha: 0.03,
          circuit: { C: 0.9e-6, V0: 12e3, L0: 20e-9, R0: 10e-3, anode_radius: 0.006, cathode_radius: 0.015, ESR: 0.0, ESL: 0.0 },
          geometry: { type: 'cylindrical' },
          radiation: { bremsstrahlung_enabled: true },
          collision: { coulomb_log: 10.0, dynamic_coulomb_log: true, sigma_en: 1e-19 },
          fluid: { backend: 'python' },
          boundary: { electrode_bc: true, axis_bc: true },
        },
        llnl_dpf: {
          grid_shape: [48, 1, 96], dx: 3e-4, sim_time: 3e-6, dt_init: 1e-10,
          rho0: 2e-4, T0: 300, anomalous_alpha: 0.05,
          circuit: { C: 3.6e-4, V0: 24e3, L0: 30e-9, R0: 5e-3, anode_radius: 0.025, cathode_radius: 0.05, ESR: 0.0, ESL: 0.0 },
          geometry: { type: 'cylindrical' },
          radiation: { bremsstrahlung_enabled: true, fld_enabled: true },
          collision: { coulomb_log: 10.0, dynamic_coulomb_log: true, sigma_en: 1e-19 },
          fluid: { backend: 'python' },
          boundary: { electrode_bc: true, axis_bc: true },
        },
        cartesian_demo: {
          grid_shape: [32, 32, 32], dx: 5e-4, sim_time: 5e-7, dt_init: 1e-10,
          rho0: 1e-4, T0: 300,
          circuit: { C: 5e-6, V0: 5e3, L0: 5e-8, R0: 0.01, anode_radius: 0.005, cathode_radius: 0.01, ESR: 0.0, ESL: 0.0 },
          radiation: { bremsstrahlung_enabled: true },
          boundary: { electrode_bc: false, axis_bc: false },
        },
      };

      const config = presetConfigs[name] ?? DEFAULT_CONFIG;

      set({
        config,
        errors: [],
        isValid: false,
        isArmed: false,
      });
    } catch (error) {
      console.error('Failed to load preset:', error);
      set({
        errors: [`Failed to load preset: ${error instanceof Error ? error.message : 'Unknown error'}`],
        isValid: false,
      });
    }
  },

  updateCircuit: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        circuit: {
          ...(config.circuit ?? DEFAULT_CONFIG.circuit!),
          ...partial,
        } as SimulationConfig['circuit'],
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateFluid: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        fluid: {
          ...config.fluid,
          ...partial,
        },
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateGrid: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        ...partial,
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateCollision: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        collision: {
          ...config.collision,
          ...partial,
        },
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateRadiation: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        radiation: {
          ...config.radiation,
          ...partial,
        },
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateSheath: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        sheath: {
          ...config.sheath,
          ...partial,
        },
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateBoundary: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        boundary: {
          ...config.boundary,
          ...partial,
        },
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateDiagnostics: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        diagnostics: {
          ...config.diagnostics,
          ...partial,
        },
      },
      isValid: false,
      isArmed: false,
    });
  },

  updateGeometry: (partial) => {
    const { config } = get();
    set({
      config: {
        ...config,
        geometry: {
          ...config.geometry,
          ...partial,
        },
      },
      isValid: false,
      isArmed: false,
    });
  },

  validate: async () => {
    const { config } = get();

    try {
      const response: ConfigValidationResponse = await validateConfig(config as SimulationConfig);

      set({
        isValid: response.valid,
        errors: response.errors || [],
        isArmed: response.valid,
      });

      return response.valid;
    } catch (error) {
      console.error('Validation failed:', error);
      set({
        isValid: false,
        errors: [`Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`],
        isArmed: false,
      });
      return false;
    }
  },

  commit: async () => {
    const { config, isValid } = get();

    if (!isValid) {
      const valid = await get().validate();
      if (!valid) {
        return null;
      }
    }

    try {
      const simInfo = await createSimulation({ config: config as Record<string, unknown> });

      set({
        simId: simInfo.sim_id,
        isArmed: false,
      });

      return simInfo.sim_id;
    } catch (error) {
      console.error('Failed to create simulation:', error);
      set({
        errors: [`Failed to create simulation: ${error instanceof Error ? error.message : 'Unknown error'}`],
      });
      return null;
    }
  },

  reset: () => {
    set({
      config: DEFAULT_CONFIG,
      errors: [],
      isValid: false,
      isArmed: false,
      simId: null,
    });
  },
}));
