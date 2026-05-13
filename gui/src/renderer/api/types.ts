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
  run_mode?: string | null;
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
  validation_status: string;
  result_classification: Record<string, unknown>;
  predictive_readiness: Record<string, unknown>;
  high_fidelity_readiness: Record<string, unknown>;
  first_principles_mhd_readiness: Record<string, unknown>;
  first_principles_energy_accounting: Record<string, unknown>;
  first_principles_startup_initialization: Record<string, unknown>;
  digitization_status: Record<string, unknown>;
  readiness_scope: Record<string, unknown>;
  source_blockers: string[];
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
  source_scope: string;
  source_scope_status: string;
  source_scope_note: string;
  validation_scope: string;
}

export interface CreateProjectRequest {
  root: string;
  name: string;
  config?: Record<string, unknown>;
  outputs?: string[];
  run_manifests?: string[];
  validation_status?: string;
  result_classification?: Record<string, unknown>;
  artifact_classification?: Record<string, unknown>;
  logs?: string[];
  provenance?: Record<string, unknown>;
}

export interface LoadProjectRequest {
  root: string;
}

export interface DuplicateProjectRequest {
  source_root: string;
  destination_root: string;
  name?: string | null;
}

export interface ArchiveProjectRequest {
  root: string;
  reason?: string;
}

export interface ProjectManifest {
  manifest_version: "1.0";
  project_id: string;
  name: string;
  status: "active" | "archived";
  created_utc: string;
  updated_utc: string;
  archived_utc?: string | null;
  archive_reason?: string | null;
  source_project_id?: string | null;
  config_path: string;
  config_hash: string;
  outputs: string[];
  run_manifests: string[];
  validation_status: string;
  result_classification: Record<string, unknown>;
  artifact_classification: Record<string, unknown>;
  logs: string[];
  provenance: Record<string, unknown>;
}

export interface ProjectInfo {
  root: string;
  manifest: ProjectManifest;
  config: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  backends: {
    python: boolean;
    athena: boolean;
    athenak: boolean;
    metal: boolean;
    mlx: boolean;
    hybrid: boolean;
  };
}

export interface UnitsMetadata {
  time_base: Record<string, unknown>;
  scalars: Record<string, { units: string; dimension: string }>;
  fields: Record<string, { units: string; dimension: string }>;
  authority: Record<string, { units: string; dimension: string }>;
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
  backend: "python" | "athena" | "athenak" | "metal" | "mlx" | "hybrid" | "auto";
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
  time_integrator?: string;
  precision?: string;
  use_ct?: boolean;
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
  run_mode?: string;
  validation_scope?: string;
  source_scope?: string;
  source_scope_status?: string;
  preset_name?: string;
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
