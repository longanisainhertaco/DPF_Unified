import { create } from 'zustand';
import { SimulationWebSocket } from '../api/websocket';
import { startSimulation, pauseSimulation, resumeSimulation, stopSimulation } from '../api/client';
import type { SimulationStatus, ScalarUpdate } from '../api/types';
import { MAX_SCALAR_HISTORY } from '@shared/constants';

interface SimulationState {
  // Simulation identity and status
  simId: string | null;
  status: SimulationStatus;

  // Progress tracking
  currentStep: number;
  currentTime: number;

  // Real-time scalar data (ring buffer)
  scalarHistory: ScalarUpdate[];

  // WebSocket connection state
  wsConnected: boolean;

  // Internal WebSocket instance (not serialized)
  _ws: SimulationWebSocket | null;

  // Actions
  start: (simId: string) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  pushScalar: (update: ScalarUpdate) => void;
  reset: () => void;
}

export const useSimulationStore = create<SimulationState>((set, get) => ({
  simId: null,
  status: 'idle',
  currentStep: 0,
  currentTime: 0,
  scalarHistory: [],
  wsConnected: false,
  _ws: null,

  start: async (simId: string) => {
    try {
      // Start the simulation on the backend
      await startSimulation(simId);

      // Create WebSocket connection with callbacks
      const ws = new SimulationWebSocket(
        simId,
        // onScalarUpdate
        (update: ScalarUpdate) => {
          get().pushScalar(update);
          set({
            currentStep: update.step,
            currentTime: update.time,
          });
          // If simulation reported finished via WebSocket
          if (update.finished) {
            set({ status: 'finished' });
          }
        },
        // onConnectionChange
        (connected: boolean) => {
          set({ wsConnected: connected });
        }
      );

      // Connect the WebSocket
      ws.connect();

      set({
        simId,
        status: 'running',
        _ws: ws,
      });
    } catch (error) {
      console.error('Failed to start simulation:', error);
      set({ status: 'error' });
      throw error;
    }
  },

  pause: async () => {
    const { simId } = get();

    if (!simId) {
      throw new Error('No active simulation');
    }

    try {
      await pauseSimulation(simId);
      set({ status: 'paused' });
    } catch (error) {
      console.error('Failed to pause simulation:', error);
      throw error;
    }
  },

  resume: async () => {
    const { simId } = get();

    if (!simId) {
      throw new Error('No active simulation');
    }

    try {
      await resumeSimulation(simId);
      set({ status: 'running' });
    } catch (error) {
      console.error('Failed to resume simulation:', error);
      throw error;
    }
  },

  stop: async () => {
    const { simId, _ws } = get();

    if (!simId) {
      throw new Error('No active simulation');
    }

    try {
      // Stop the simulation on the backend
      await stopSimulation(simId);

      // Disconnect WebSocket
      if (_ws) {
        _ws.disconnect();
      }

      set({
        status: 'finished',
        wsConnected: false,
        _ws: null,
      });
    } catch (error) {
      console.error('Failed to stop simulation:', error);
      throw error;
    }
  },

  pushScalar: (update: ScalarUpdate) => {
    const { scalarHistory } = get();

    // Add new update, trim to ring buffer size
    const newHistory = [...scalarHistory, update];
    if (newHistory.length > MAX_SCALAR_HISTORY) {
      newHistory.shift();
    }

    set({ scalarHistory: newHistory });
  },

  reset: () => {
    const { _ws } = get();

    // Disconnect WebSocket if still connected
    if (_ws) {
      _ws.disconnect();
    }

    set({
      simId: null,
      status: 'idle',
      currentStep: 0,
      currentTime: 0,
      scalarHistory: [],
      wsConnected: false,
      _ws: null,
    });
  },
}));
