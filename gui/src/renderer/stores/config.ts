import { create } from 'zustand';
import { fetchPresets, validateConfig, createSimulation } from '../api/client';
import type { SimulationConfig, PresetInfo, ConfigValidationResponse } from '../api/types';

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
  circuit: {
    C: 1e-6,
    V0: 1e3,
    L0: 1e-7,
    R0: 0.01,
    anode_radius: 0.005,
    cathode_radius: 0.01,
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
          circuit: { C: 1e-6, V0: 1e3, L0: 1e-7, R0: 0.01, anode_radius: 0.005, cathode_radius: 0.01 },
        },
        pf1000: {
          grid_shape: [64, 1, 128], dx: 5e-4, sim_time: 5e-6, dt_init: 1e-10,
          rho0: 4e-4, T0: 300, anomalous_alpha: 0.05,
          circuit: { C: 1.332e-3, V0: 27e3, L0: 15e-9, R0: 3e-3, anode_radius: 0.0575, cathode_radius: 0.08 },
          geometry: { type: 'cylindrical' },
          radiation: { bremsstrahlung_enabled: true, fld_enabled: true },
        },
        nx2: {
          grid_shape: [32, 1, 64], dx: 2e-4, sim_time: 1e-6, dt_init: 1e-11,
          rho0: 8e-5, T0: 300, anomalous_alpha: 0.03,
          circuit: { C: 0.9e-6, V0: 12e3, L0: 20e-9, R0: 10e-3, anode_radius: 0.006, cathode_radius: 0.015 },
          geometry: { type: 'cylindrical' },
          radiation: { bremsstrahlung_enabled: true },
        },
        llnl_dpf: {
          grid_shape: [48, 1, 96], dx: 3e-4, sim_time: 3e-6, dt_init: 1e-10,
          rho0: 2e-4, T0: 300, anomalous_alpha: 0.05,
          circuit: { C: 3.6e-4, V0: 24e3, L0: 30e-9, R0: 5e-3, anode_radius: 0.025, cathode_radius: 0.05 },
          geometry: { type: 'cylindrical' },
          radiation: { bremsstrahlung_enabled: true, fld_enabled: true },
        },
        cartesian_demo: {
          grid_shape: [32, 32, 32], dx: 5e-4, sim_time: 5e-7, dt_init: 1e-10,
          rho0: 1e-4, T0: 300,
          circuit: { C: 5e-6, V0: 5e3, L0: 5e-8, R0: 0.01, anode_radius: 0.005, cathode_radius: 0.01 },
          radiation: { bremsstrahlung_enabled: true },
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
