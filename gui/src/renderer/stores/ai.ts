import { create } from 'zustand';
import { fetchAIStatus, runAISweep } from '../api/client';
import type { SimulationConfig } from '../api/types';

interface Advisory {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: number;
}

interface SweepResult {
  config: Partial<SimulationConfig>;
  metrics: Record<string, number>;
}

interface AIState {
  // AI availability
  aiAvailable: boolean;

  // Advisory system
  advisories: Advisory[];

  // Parameter sweep state
  sweepStatus: 'idle' | 'running' | 'complete';
  sweepResults: SweepResult[];

  // Actions
  checkAIStatus: () => Promise<void>;
  generatePreShotAdvisory: (config: Partial<SimulationConfig>) => void;
  addAdvisory: (severity: Advisory['severity'], message: string) => void;
  clearAdvisories: () => void;
  runSweep: (configs: Record<string, unknown>[], nSteps: number) => Promise<void>;
}

export const useAIStore = create<AIState>((set, get) => ({
  aiAvailable: false,
  advisories: [],
  sweepStatus: 'idle',
  sweepResults: [],

  checkAIStatus: async () => {
    try {
      const status = await fetchAIStatus();
      set({ aiAvailable: status.model_loaded });
    } catch {
      set({ aiAvailable: false });
    }
  },

  generatePreShotAdvisory: (config) => {
    const { addAdvisory, clearAdvisories } = get();

    // Clear previous advisories before generating new ones
    clearAdvisories();

    // Extract relevant parameters
    const V0 = config.circuit?.V0 ?? 0;
    const anodeRadius = config.circuit?.anode_radius ?? 0;
    const cathodeRadius = config.circuit?.cathode_radius ?? 1;
    const L0 = config.circuit?.L0 ?? 0;
    const rho0 = config.rho0 ?? 0;

    // Heuristic rules-based checks
    if (V0 < 5000) {
      addAdvisory('warning', 'Voltage may be too low for breakdown');
    }

    if (rho0 > 1e-2) {
      addAdvisory('warning', 'Initial density unusually high — sheath formation may be impeded');
    }

    if (anodeRadius > cathodeRadius * 0.8) {
      addAdvisory('warning', 'Annular gap too narrow');
    }

    if (L0 > 100e-9) {
      addAdvisory('warning', 'High inductance limits peak current');
    }

    // If no warnings were added, report nominal
    const { advisories } = get();
    const hasWarnings = advisories.some(a => a.severity === 'warning' || a.severity === 'critical');

    if (!hasWarnings) {
      addAdvisory('info', 'Configuration looks nominal');
    }
  },

  addAdvisory: (severity, message) => {
    const { advisories } = get();

    const newAdvisory: Advisory = {
      id: crypto.randomUUID(),
      severity,
      message,
      timestamp: Date.now(),
    };

    set({
      advisories: [...advisories, newAdvisory],
    });
  },

  clearAdvisories: () => {
    set({ advisories: [] });
  },

  runSweep: async (configs, nSteps) => {
    set({ sweepStatus: 'running', sweepResults: [] });

    try {
      const response = await runAISweep(configs, nSteps);

      // Map response results to SweepResult format
      const sweepResults: SweepResult[] = response.results.map((r) => ({
        config: (r as Record<string, unknown>).config as Partial<SimulationConfig> ?? {},
        metrics: (r as Record<string, unknown>).metrics as Record<string, number> ?? {},
      }));

      set({
        sweepStatus: 'complete',
        sweepResults,
      });
    } catch (error) {
      console.error('Parameter sweep failed:', error);
      set({ sweepStatus: 'idle' });
      throw error;
    }
  },
}));
