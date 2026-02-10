import { create } from 'zustand';
import { fetchAIStatus, runAISweep, runAIInverse, runAIConfidence, chatWithWALRUS } from '../api/client';
import { localChatRouter } from '../api/chatRouter';
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

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  suggestions?: string[];
  timestamp: number;
}

interface AIState {
  // AI availability
  aiAvailable: boolean;

  // Advisory system
  advisories: Advisory[];

  // Parameter sweep state
  sweepStatus: 'idle' | 'running' | 'complete';
  sweepResults: SweepResult[];
  sweepVariable: string | null;

  // Inverse design state
  inverseStatus: 'idle' | 'running' | 'complete' | 'error';
  inverseResult: {
    best_config: Record<string, number>;
    loss: number;
    n_trials: number;
  } | null;

  // Sweep metric selection
  sweepMetric: string;

  // Chat state
  chatMessages: ChatMessage[];
  chatStatus: 'idle' | 'sending' | 'error';

  // Actions
  checkAIStatus: () => Promise<void>;
  generatePreShotAdvisory: (config: Partial<SimulationConfig>) => void;
  addAdvisory: (severity: Advisory['severity'], message: string) => void;
  clearAdvisories: () => void;
  runSweep: (configs: Record<string, unknown>[], nSteps: number, variable?: string) => Promise<void>;
  runInverse: (targets: Record<string, number>, constraints?: Record<string, number>, method?: string, nTrials?: number) => Promise<void>;
  setSweepMetric: (metric: string) => void;
  sendChat: (question: string) => Promise<void>;
  clearChat: () => void;
}

export const useAIStore = create<AIState>((set, get) => ({
  aiAvailable: false,
  advisories: [],
  sweepStatus: 'idle',
  sweepResults: [],
  sweepVariable: null,
  inverseStatus: 'idle',
  inverseResult: null,
  sweepMetric: 'max_Te',
  chatMessages: [],
  chatStatus: 'idle',

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

    // Radiation check: bremsstrahlung disabled at high voltage
    const bremsEnabled = config.radiation?.bremsstrahlung_enabled ?? true;
    if (!bremsEnabled && V0 > 10000) {
      addAdvisory('warning', 'Bremsstrahlung disabled at high voltage — radiation losses may be underestimated');
    }

    // Grid resolution check
    const gridShape = config.grid_shape ?? [];
    if (gridShape.length > 0 && gridShape.every((dim) => dim <= 8)) {
      addAdvisory('warning', 'Very coarse grid (8\u00b3) — results are qualitative only');
    }

    // CFL stability check
    const cfl = config.fluid?.cfl ?? 0.4;
    if (cfl > 0.8) {
      addAdvisory('warning', 'High CFL number may cause numerical instabilities');
    }

    // Phase P solver checks
    const riemannSolver = config.fluid?.riemann_solver ?? 'hlld';
    if (riemannSolver !== 'hlld') {
      addAdvisory('warning', 'HLLD Riemann solver recommended for MHD accuracy (Phase P default). HLL is more diffusive.');
    }

    const timeIntegrator = config.fluid?.time_integrator ?? 'ssp_rk3';
    if (timeIntegrator === 'ssp_rk2') {
      addAdvisory('info', 'SSP-RK3 provides 3rd-order temporal accuracy vs 2nd-order for RK2.');
    }

    const backend = config.fluid?.backend ?? 'python';
    const reconstruction = config.fluid?.reconstruction ?? 'weno5';
    if (backend === 'python' && reconstruction === 'weno5') {
      addAdvisory('warning', 'Python WENO5 has boundary instabilities on dynamic problems. Use Metal backend or PLM for stability.');
    }

    const precision = config.fluid?.precision ?? 'float32';
    if (precision === 'float32' && reconstruction === 'weno5') {
      addAdvisory('info', 'Float64 precision improves WENO5 accuracy. Set precision to float64 for V&V runs.');
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

  runSweep: async (configs, nSteps, variable) => {
    set({ sweepStatus: 'running', sweepResults: [], sweepVariable: variable ?? null });

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

  runInverse: async (targets, constraints, method = 'bayesian', nTrials = 100) => {
    set({ inverseStatus: 'running', inverseResult: null });
    try {
      const response = await runAIInverse(targets, constraints, method, nTrials);
      set({
        inverseStatus: 'complete',
        inverseResult: {
          best_config: response.best_config,
          loss: response.loss,
          n_trials: response.n_trials,
        },
      });
    } catch (error) {
      console.error('Inverse design failed:', error);
      set({ inverseStatus: 'error' });
      throw error;
    }
  },

  setSweepMetric: (metric) => {
    set({ sweepMetric: metric });
  },

  sendChat: async (question) => {
    const { chatMessages } = get();

    // Add user message
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
      timestamp: Date.now(),
    };

    set({
      chatMessages: [...chatMessages, userMsg],
      chatStatus: 'sending',
    });

    try {
      // Try the Python backend first
      const response = await chatWithWALRUS(question);

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.response,
        intent: response.intent,
        suggestions: response.suggestions,
        timestamp: Date.now(),
      };

      set((state) => ({
        chatMessages: [...state.chatMessages, assistantMsg],
        chatStatus: 'idle',
      }));
    } catch {
      // Backend unreachable — fall back to client-side pattern matcher
      console.info('Backend unreachable, using local chat router');
      const local = localChatRouter(question);

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: local.response,
        intent: local.intent,
        suggestions: local.suggestions,
        timestamp: Date.now(),
      };

      set((state) => ({
        chatMessages: [...state.chatMessages, assistantMsg],
        chatStatus: 'idle',
      }));
    }
  },

  clearChat: () => {
    set({ chatMessages: [], chatStatus: 'idle' });
  },
}));
