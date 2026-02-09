/**
 * Wire-format types mirroring the DPF server Pydantic models.
 *
 * These types match `src/dpf/server/models.py` exactly.
 */

// ── Enums ───────────────────────────────────────────────────

export type SimulationStatus =
  | "idle"
  | "running"
  | "paused"
  | "finished"
  | "error";

// ── REST request / response types ────────────────────────────

export interface CreateSimulationRequest {
  config: Record<string, unknown>;
  max_steps?: number | null;
  preset?: string | null;
}

export interface SimulationInfo {
  sim_id: string;
  status: SimulationStatus;
  backend: string;
  step: number;
  time: number;
  current: number;
  voltage: number;
  energy_conservation: number;
  max_Te: number;
  max_rho: number;
  total_radiated_energy: number;
  error_message?: string | null;
}

export interface ConfigValidationResponse {
  valid: boolean;
  errors: string[];
}

export interface PresetInfo {
  name: string;
  description: string;
  device: string;
  geometry: string;
  grid_shape: number[];
}

export interface HealthResponse {
  status: string;
  backends: {
    python: boolean;
    athena: boolean;
    athenak: boolean;
  };
}

// ── WebSocket message types ──────────────────────────────────

export interface ScalarUpdate {
  type: "scalar";
  step: number;
  time: number;
  dt: number;
  current: number;
  voltage: number;
  energy_conservation: number;
  max_Te: number;
  max_rho: number;
  Z_bar: number;
  R_plasma: number;
  eta_anomalous: number;
  total_radiated_energy: number;
  neutron_rate: number;
  total_neutron_yield: number;
  finished: boolean;
}

export interface FieldRequest {
  type: "request_fields";
  fields: string[];
  downsample: number;
}

export interface FieldHeader {
  type: "field_header";
  fields: Record<
    string,
    {
      shape: number[];
      dtype: string;
      offset: number;
      nbytes: number;
    }
  >;
  total_bytes: number;
}

// ── AI types ─────────────────────────────────────────────────

export interface AIStatusResponse {
  torch_available: boolean;
  model_loaded: boolean;
  device: string;
  ensemble_size: number;
}

export interface SweepResult {
  config: Record<string, unknown>;
  trajectory: Record<string, unknown>[];
  metrics?: Record<string, number>;
}

// ── Config types (mirrors SimulationConfig sub-models) ────────

export interface CircuitConfig {
  C: number;
  V0: number;
  L0: number;
  R0: number;
  anode_radius: number;
  cathode_radius: number;
}

export interface FluidConfig {
  backend: "python" | "athena" | "athenak";
  reconstruction: string;
  riemann_solver: string;
  cfl: number;
  resistive: boolean;
  viscosity: boolean;
  thermal_conduction: boolean;
  braginskii: boolean;
  hall: boolean;
}

export interface SimulationConfig {
  grid_shape: number[];
  dx: number;
  sim_time: number;
  dt_init: number;
  rho0: number;
  T0: number;
  anomalous_alpha?: number;
  circuit: CircuitConfig;
  fluid?: Partial<FluidConfig>;
  geometry?: { type: string };
  radiation?: {
    bremsstrahlung_enabled?: boolean;
    fld_enabled?: boolean;
  };
  sheath?: {
    enabled?: boolean;
    boundary?: string;
  };
}
