/**
 * REST API client for the DPF Python backend.
 *
 * All functions make HTTP requests to localhost:<port>/api/...
 */

import { API_BASE_URL } from "@shared/constants";

import type {
  ConfigValidationResponse,
  CreateSimulationRequest,
  HealthResponse,
  PresetInfo,
  SimulationConfig,
  SimulationInfo,
} from "./types";

/** Base URL — can be overridden for testing */
let baseUrl = API_BASE_URL;

export function setBaseUrl(url: string): void {
  baseUrl = url;
}

// ── Generic fetch wrapper ──────────────────────────────────────

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${baseUrl}${path}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => "Unknown error");
    throw new Error(
      `API error ${response.status} on ${path}: ${errorBody}`
    );
  }

  return response.json() as Promise<T>;
}

// ── Health ──────────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health");
}

// ── Presets ─────────────────────────────────────────────────────

export async function fetchPresets(): Promise<PresetInfo[]> {
  return apiFetch<PresetInfo[]>("/api/presets");
}

// ── Config ──────────────────────────────────────────────────────

export async function fetchConfigSchema(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>("/api/config/schema");
}

export async function validateConfig(
  config: Partial<SimulationConfig>
): Promise<ConfigValidationResponse> {
  return apiFetch<ConfigValidationResponse>("/api/config/validate", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

// ── Simulations ─────────────────────────────────────────────────

export async function createSimulation(
  req: CreateSimulationRequest
): Promise<SimulationInfo> {
  return apiFetch<SimulationInfo>("/api/simulations", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getSimulation(
  simId: string
): Promise<SimulationInfo> {
  return apiFetch<SimulationInfo>(`/api/simulations/${simId}`);
}

export async function startSimulation(
  simId: string
): Promise<SimulationInfo> {
  return apiFetch<SimulationInfo>(`/api/simulations/${simId}/start`, {
    method: "POST",
  });
}

export async function pauseSimulation(
  simId: string
): Promise<SimulationInfo> {
  return apiFetch<SimulationInfo>(`/api/simulations/${simId}/pause`, {
    method: "POST",
  });
}

export async function resumeSimulation(
  simId: string
): Promise<SimulationInfo> {
  return apiFetch<SimulationInfo>(`/api/simulations/${simId}/resume`, {
    method: "POST",
  });
}

export async function stopSimulation(
  simId: string
): Promise<SimulationInfo> {
  return apiFetch<SimulationInfo>(`/api/simulations/${simId}/stop`, {
    method: "POST",
  });
}

// ── AI endpoints ────────────────────────────────────────────────

export async function fetchAIStatus(): Promise<{
  torch_available: boolean;
  model_loaded: boolean;
  device: string;
  ensemble_size: number;
}> {
  return apiFetch("/api/ai/status");
}

export async function runAISweep(
  configs: Record<string, unknown>[],
  nSteps: number = 100
): Promise<{
  results: Record<string, unknown>[];
  n_configs: number;
}> {
  return apiFetch(`/api/ai/sweep?n_steps=${nSteps}`, {
    method: "POST",
    body: JSON.stringify(configs),
  });
}
