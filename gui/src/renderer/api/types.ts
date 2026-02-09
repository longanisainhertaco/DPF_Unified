/**
 * Wire-format types mirroring the DPF server Pydantic models.
 *
 * These types match `src/dpf/server/models.py` and `src/dpf/config.py` exactly.
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

export interface InverseDesignResult {
  best_config: Record<string, number>;
  predicted_outcomes: Record<string, unknown>;
  loss: number;
  n_trials: number;
}

export interface PredictionResult {
  predicted_state: Record<string, unknown>;
  inference_time_ms: number;
}

export interface ConfidenceResult {
  predicted_state: Record<string, unknown>;
  confidence: Record<string, unknown>;
  ood_score: number;
  confidence_score: number;
  n_models: number;
  inference_time_ms: number;
}

export interface RolloutResult {
  trajectory: Record<string, unknown>[];
  n_steps: number;
  total_inference_time_ms: number;
}

// ── Config types (mirrors src/dpf/config.py Pydantic models) ──

export interface CircuitConfig {
  C: number;
  V0: number;
  L0: number;
  R0: number;
  anode_radius: number;
  cathode_radius: number;
  ESR: number;
  ESL: number;
}

export interface CollisionConfig {
  coulomb_log: number;
  dynamic_coulomb_log: boolean;
  sigma_en: number;
}

export interface RadiationConfig {
  bremsstrahlung_enabled: boolean;
  gaunt_factor: number;
  fld_enabled: boolean;
  flux_limiter: number;
  line_radiation_enabled: boolean;
  impurity_Z: number;
  impurity_fraction: number;
}

export interface SheathConfig {
  enabled: boolean;
  boundary: string;
  V_sheath: number;
}

export interface GeometryConfig {
  type: string;
  dz?: number;
}

export interface BoundaryConfig {
  electrode_bc: boolean;
  axis_bc: boolean;
}

export interface FluidConfig {
  backend: "python" | "athena" | "athenak" | "auto";
  reconstruction: string;
  riemann_solver: string;
  cfl: number;
  dedner_ch: number;
  gamma: number;
  enable_resistive: boolean;
  enable_energy_equation: boolean;
  enable_nernst: boolean;
  enable_viscosity: boolean;
  diffusion_method: string;
  sts_stages: number;
  implicit_tol: number;
  enable_powell: boolean;
  dedner_cr: number;
  enable_anisotropic_conduction: boolean;
  full_braginskii_viscosity: boolean;
}

export interface DiagnosticsConfig {
  hdf5_filename: string;
  output_interval: number;
  field_output_interval: number;
}

export interface SimulationConfig {
  grid_shape: number[];
  dx: number;
  sim_time: number;
  dt_init?: number;
  rho0: number;
  T0: number;
  anomalous_alpha: number;
  ion_mass: number;
  circuit: CircuitConfig;
  collision?: Partial<CollisionConfig>;
  radiation?: Partial<RadiationConfig>;
  sheath?: Partial<SheathConfig>;
  geometry?: Partial<GeometryConfig>;
  fluid?: Partial<FluidConfig>;
  boundary?: Partial<BoundaryConfig>;
  diagnostics?: Partial<DiagnosticsConfig>;
}
