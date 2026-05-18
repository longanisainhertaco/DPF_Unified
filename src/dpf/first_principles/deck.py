"""Input deck contract for package-native first-principles DPF runs.

This module is intentionally independent of the older validation/readiness
workflow.  It describes the physical inputs needed by the package-native 3-D
hybrid EM/PIC-fluid runner and rejects reduced-model authority fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dpf import constants as dpf_constants
from dpf.experimental.civ_breakdown import compute_breakdown
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE

ELEMENTARY_CHARGE = dpf_constants.e
K_B = dpf_constants.k_B

REDUCED_MODEL_AUTHORITY_FIELDS = {
    "lee_model",
    "lee_fit_factors",
    "lee",
    "radpf",
    "snowplow",
    "snowplow_closure",
    "snowplow_model",
    "mass_sweep_fraction",
    "current_factor",
    "fcr",
    "fcr_eff",
    "beam_fraction",
    "empirical_yield",
}


def _ideal_gas_number_density_m3(pressure_Pa: float, temperature_K: float) -> float:
    if pressure_Pa <= 0.0:
        raise ValueError("pressure_Pa must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    return float(pressure_Pa) / (float(K_B) * float(temperature_K))

STARTUP_MODES = {
    "imported_pic_sheath_state",
    "source_backed_end_rundown_sheath",
    "surface_breakdown_bvp",
    "plasma_injection_startup",
    "seeded_layer",
    # Backward-compatible legacy names. These remain engineering-only until
    # replaced by one of the explicit source-truth startup modes above.
    "source_backed_candidate_uniform",
    "source_backed_profile",
}

ENGINEERING_ONLY_STARTUP_MODES = {
    "source_backed_end_rundown_sheath",
    "plasma_injection_startup",
    "seeded_layer",
    "source_backed_candidate_uniform",
    "source_backed_profile",
}


@dataclass(frozen=True)
class SourceReference:
    """A local source-truth reference used by a first-principles deck."""

    path: str
    sha256: str | None = None
    record_id: str | None = None
    capability_tags: tuple[str, ...] = ()
    role: str = "source"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SourceReference:
        return cls(
            path=str(value["path"]),
            sha256=_optional_str(value.get("sha256")),
            record_id=_optional_str(value.get("record_id") or value.get("source_id")),
            capability_tags=tuple(str(item) for item in value.get("capability_tags", ())),
            role=str(value.get("role", "source")),
        )


@dataclass(frozen=True)
class DeviceGeometryDeck:
    """Machine geometry required by the first-principles runner."""

    name: str
    anode_radius_m: float
    cathode_radius_m: float
    anode_length_m: float
    insulator_length_m: float = 0.0
    anode_inner_radius_m: float | None = None
    cathode_rod_count: int | None = None
    cathode_rod_diameter_m: float | None = None
    cathode_rod_length_m: float | None = None
    insulator_outer_radius_m: float | None = None
    insulator_material: str | None = None
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        _require_positive("anode_radius_m", self.anode_radius_m)
        _require_positive("cathode_radius_m", self.cathode_radius_m)
        _require_positive("anode_length_m", self.anode_length_m)
        if self.insulator_length_m < 0.0:
            raise ValueError("insulator_length_m must be non-negative")
        if self.anode_radius_m >= self.cathode_radius_m:
            raise ValueError("anode_radius_m must be less than cathode_radius_m")
        if self.anode_inner_radius_m is not None:
            if self.anode_inner_radius_m < 0.0:
                raise ValueError("anode_inner_radius_m must be non-negative")
            if self.anode_inner_radius_m >= self.anode_radius_m:
                raise ValueError("anode_inner_radius_m must be less than anode_radius_m")
        if self.cathode_rod_count is not None and self.cathode_rod_count <= 0:
            raise ValueError("cathode_rod_count must be positive when supplied")
        if self.cathode_rod_diameter_m is not None:
            _require_positive("cathode_rod_diameter_m", self.cathode_rod_diameter_m)
        if self.cathode_rod_length_m is not None:
            _require_positive("cathode_rod_length_m", self.cathode_rod_length_m)
        if self.insulator_outer_radius_m is not None:
            _require_positive("insulator_outer_radius_m", self.insulator_outer_radius_m)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DeviceGeometryDeck:
        return cls(
            name=str(value["name"]),
            anode_radius_m=float(value["anode_radius_m"]),
            cathode_radius_m=float(value["cathode_radius_m"]),
            anode_length_m=float(value["anode_length_m"]),
            insulator_length_m=float(value.get("insulator_length_m", 0.0)),
            anode_inner_radius_m=_optional_float(value.get("anode_inner_radius_m")),
            cathode_rod_count=(
                None
                if value.get("cathode_rod_count") is None
                else int(value["cathode_rod_count"])
            ),
            cathode_rod_diameter_m=_optional_float(
                value.get("cathode_rod_diameter_m")
            ),
            cathode_rod_length_m=_optional_float(value.get("cathode_rod_length_m")),
            insulator_outer_radius_m=_optional_float(
                value.get("insulator_outer_radius_m")
            ),
            insulator_material=_optional_str(value.get("insulator_material")),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class CircuitDeck:
    """External circuit inputs for the resolved field power-port candidate.

    ``initial_charge_C`` follows the source circuit variable ``Q = integral I dt``.
    It is not the initial capacitor stored charge; the bank voltage is carried
    separately as ``voltage_V``.
    """

    capacitance_F: float
    voltage_V: float
    inductance_H: float
    resistance_ohm: float
    initial_current_A: float = 1.773e4
    initial_charge_C: float | None = None
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        _require_positive("capacitance_F", self.capacitance_F)
        _require_positive("voltage_V", self.voltage_V)
        _require_positive("inductance_H", self.inductance_H)
        if self.resistance_ohm < 0.0:
            raise ValueError("resistance_ohm must be non-negative")
        if self.initial_charge_C is not None and self.initial_charge_C < 0.0:
            raise ValueError("initial_charge_C must be non-negative")

    @property
    def charge_C(self) -> float:
        if self.initial_charge_C is not None:
            return float(self.initial_charge_C)
        return 0.0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> CircuitDeck:
        return cls(
            capacitance_F=float(value["capacitance_F"]),
            voltage_V=float(value["voltage_V"]),
            inductance_H=float(value["inductance_H"]),
            resistance_ohm=float(value.get("resistance_ohm", 0.0)),
            initial_current_A=float(value.get("initial_current_A", 1.773e4)),
            initial_charge_C=(
                None
                if value.get("initial_charge_C") is None
                else float(value["initial_charge_C"])
            ),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class GasDeck:
    """Fill gas and species inputs."""

    species: str = "D"
    pressure_Pa: float = 266.0
    temperature_K: float = 300.0
    ion_mass_kg: float = 3.344e-27
    ion_charge_C: float = ELEMENTARY_CHARGE
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        _require_positive("pressure_Pa", self.pressure_Pa)
        _require_positive("temperature_K", self.temperature_K)
        _require_positive("ion_mass_kg", self.ion_mass_kg)
        _require_positive("ion_charge_C", self.ion_charge_C)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> GasDeck:
        return cls(
            species=str(value.get("species", "D")),
            pressure_Pa=float(value.get("pressure_Pa", 266.0)),
            temperature_K=float(value.get("temperature_K", 300.0)),
            ion_mass_kg=float(value.get("ion_mass_kg", 3.344e-27)),
            ion_charge_C=float(value.get("ion_charge_C", ELEMENTARY_CHARGE)),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class GridDeck:
    """3-D Cartesian grid for the first-principles candidate."""

    shape: tuple[int, int, int]
    spacing_m: tuple[float, float, float]

    @property
    def dimensionality(self) -> str:
        return "3d"

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or len(self.spacing_m) != 3:
            raise ValueError("shape and spacing_m must be 3-tuples")
        if any(int(n) != n or n < 3 for n in self.shape):
            raise ValueError("all grid dimensions must be integers >= 3")
        if any(float(dx) <= 0.0 for dx in self.spacing_m):
            raise ValueError("all grid spacings must be positive")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> GridDeck:
        return cls(
            shape=tuple(int(item) for item in value["shape"]),  # type: ignore[arg-type]
            spacing_m=tuple(float(item) for item in value["spacing_m"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class StartupPolicy:
    """Startup state policy for a first-principles candidate run."""

    mode: str = "source_backed_end_rundown_sheath"
    background_density_m3: float = 1.0e20
    initial_ionization_fraction: float = 0.01
    electron_temperature_K: float = 1.0e5
    ion_temperature_K: float = 1.0e5
    initial_electric_field_V_m: tuple[float, float, float] = (1.0e5, 0.0, 0.0)
    initial_magnetic_field_T: tuple[float, float, float] = (0.0, 0.0, 0.0)
    particle_weight: float = 1.0e8
    evidence_status: str = "engineering_candidate_not_whole_shot"
    source_scope: str = "end_of_rundown_or_engineering_startup"
    can_support_whole_shot_acceptance: bool = False
    required_channels: tuple[str, ...] = (
        "breakdown_model",
        "preionization_state",
        "electrode_insulator_boundary",
        "initial_plasma_distribution",
        "current_density",
        "electric_field",
        "magnetic_field",
        "electron_temperature",
        "ion_temperature",
        "sheath_liftoff",
    )
    missing_channels: tuple[str, ...] = (
        "breakdown_model",
        "preionization_state",
        "surface_flashover_closure",
        "sheath_liftoff",
    )
    accepted_channels: tuple[str, ...] = ()
    startup_payload: dict[str, Any] = field(default_factory=dict)
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in STARTUP_MODES:
            allowed = ", ".join(sorted(STARTUP_MODES))
            raise ValueError(f"unknown startup mode {self.mode!r}; expected one of {allowed}")
        _require_positive("background_density_m3", self.background_density_m3)
        if not 0.0 <= self.initial_ionization_fraction <= 1.0:
            raise ValueError("initial_ionization_fraction must be in [0, 1]")
        _require_positive("electron_temperature_K", self.electron_temperature_K)
        _require_positive("ion_temperature_K", self.ion_temperature_K)
        _require_positive("particle_weight", self.particle_weight)
        if self.mode in ENGINEERING_ONLY_STARTUP_MODES and self.can_support_whole_shot_acceptance:
            raise ValueError(
                f"startup mode {self.mode!r} cannot support accepted whole-shot "
                "first-principles startup"
            )
        if self.mode == "surface_breakdown_bvp" and self.missing_channels:
            object.__setattr__(self, "can_support_whole_shot_acceptance", False)
        if (
            self.mode == "imported_pic_sheath_state"
            and self.evidence_status
            not in {"reviewed", "accepted", "accepted_same_scope_source"}
            and self.can_support_whole_shot_acceptance
        ):
            raise ValueError(
                "imported_pic_sheath_state can support accepted whole-shot startup "
                "only after reviewed or accepted evidence_status"
            )
        if (
            self.mode == "imported_pic_sheath_state"
            and self.can_support_whole_shot_acceptance
            and not self.startup_payload
        ):
            raise ValueError(
                "imported_pic_sheath_state requires a reviewed startup_payload "
                "before whole-shot startup support can be declared"
            )

    @property
    def whole_shot_startup_blocked(self) -> bool:
        return not self.can_support_whole_shot_acceptance

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> StartupPolicy:
        return cls(
            mode=str(value.get("mode", "source_backed_end_rundown_sheath")),
            background_density_m3=float(value.get("background_density_m3", 1.0e20)),
            initial_ionization_fraction=float(
                value.get(
                    "initial_ionization_fraction",
                    value.get("preionization_fraction", 0.01),
                )
            ),
            electron_temperature_K=float(value.get("electron_temperature_K", 1.0e5)),
            ion_temperature_K=float(value.get("ion_temperature_K", 1.0e5)),
            initial_electric_field_V_m=_triple(
                value.get("initial_electric_field_V_m", (1.0e5, 0.0, 0.0)),
                "initial_electric_field_V_m",
            ),
            initial_magnetic_field_T=_triple(
                value.get("initial_magnetic_field_T", (0.0, 0.0, 0.0)),
                "initial_magnetic_field_T",
            ),
            particle_weight=float(value.get("particle_weight", 1.0e8)),
            evidence_status=str(
                value.get("evidence_status", "engineering_candidate_not_whole_shot")
            ),
            source_scope=str(
                value.get("source_scope", "end_of_rundown_or_engineering_startup")
            ),
            can_support_whole_shot_acceptance=bool(
                value.get(
                    "can_support_whole_shot_acceptance",
                    value.get("can_support_first_principles_startup", False),
                )
            ),
            required_channels=_string_tuple(
                value.get(
                    "required_channels",
                    (
                        "breakdown_model",
                        "preionization_state",
                        "electrode_insulator_boundary",
                        "initial_plasma_distribution",
                        "current_density",
                        "electric_field",
                        "magnetic_field",
                        "electron_temperature",
                        "ion_temperature",
                        "sheath_liftoff",
                    ),
                )
            ),
            missing_channels=_string_tuple(
                value.get(
                    "missing_channels",
                    (
                        "breakdown_model",
                        "preionization_state",
                        "surface_flashover_closure",
                        "sheath_liftoff",
                    ),
                )
            ),
            accepted_channels=_string_tuple(value.get("accepted_channels", ())),
            startup_payload=dict(value.get("startup_payload", value.get("payload", {}))),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class ClosurePolicy:
    """Active physics closures for the 3-D first-principles candidate."""

    sigma0_S_m: float = 1.0e2
    ohmic_cfl_safety: float = 1.0
    density_floor_m3: float = 1.0e20
    include_hall: bool = False
    use_predictor_corrector: bool = True
    use_source_ordered_velocity_update: bool = True
    marder_factor_scale: float = 0.0
    marder_nondominance_threshold: float = 0.5
    apply_circuit_boundary: bool = True
    circuit_udpf_V: float = 0.0
    circuit_udpf_mode: str = "lagged_volume_j_dot_e"
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        if self.sigma0_S_m < 0.0:
            raise ValueError("sigma0_S_m must be non-negative")
        _require_positive("ohmic_cfl_safety", self.ohmic_cfl_safety)
        _require_positive("density_floor_m3", self.density_floor_m3)
        if self.marder_factor_scale < 0.0:
            raise ValueError("marder_factor_scale must be non-negative")
        if self.circuit_udpf_mode not in {"input_sequence", "lagged_volume_j_dot_e"}:
            raise ValueError(
                "circuit_udpf_mode must be 'input_sequence' or "
                "'lagged_volume_j_dot_e'"
            )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ClosurePolicy:
        return cls(
            sigma0_S_m=float(value.get("sigma0_S_m", 1.0e2)),
            ohmic_cfl_safety=float(value.get("ohmic_cfl_safety", 1.0)),
            density_floor_m3=float(value.get("density_floor_m3", 1.0e20)),
            include_hall=bool(value.get("include_hall", False)),
            use_predictor_corrector=bool(value.get("use_predictor_corrector", True)),
            use_source_ordered_velocity_update=bool(
                value.get("use_source_ordered_velocity_update", True)
            ),
            marder_factor_scale=float(value.get("marder_factor_scale", 0.0)),
            marder_nondominance_threshold=float(
                value.get("marder_nondominance_threshold", 0.5)
            ),
            apply_circuit_boundary=bool(value.get("apply_circuit_boundary", True)),
            circuit_udpf_V=float(value.get("circuit_udpf_V", 0.0)),
            circuit_udpf_mode=str(
                value.get("circuit_udpf_mode", "lagged_volume_j_dot_e")
            ),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class BoundaryPolicy:
    """Candidate field and particle boundary policy for the 3-D runner."""

    pml_cells: int = 0
    pml_strength: float = 0.0
    particle_absorption_enabled: bool = False
    open_boundary: bool = True
    conductor_mask_status: str = "not_supplied"
    conductor_mask_mode: str = "none"
    source_references: tuple[SourceReference, ...] = ()

    @property
    def can_support_first_principles_acceptance(self) -> bool:
        return False

    def __post_init__(self) -> None:
        if int(self.pml_cells) != self.pml_cells or self.pml_cells < 0:
            raise ValueError("pml_cells must be a non-negative integer")
        if self.pml_strength < 0.0:
            raise ValueError("pml_strength must be non-negative")
        if self.conductor_mask_status not in {
            "not_supplied",
            "candidate_geometry_mask",
            "reviewed_same_scope_geometry_mask",
        }:
            raise ValueError("unknown conductor_mask_status")
        if self.conductor_mask_mode not in {
            "none",
            "axisymmetric_coaxial_projection",
            "pf1000_rod_hollow_projection",
        }:
            raise ValueError("unknown conductor_mask_mode")
        if (
            self.conductor_mask_mode
            in {"axisymmetric_coaxial_projection", "pf1000_rod_hollow_projection"}
            and self.conductor_mask_status == "not_supplied"
        ):
            raise ValueError(
                f"{self.conductor_mask_mode} requires conductor_mask_status"
            )
        if (
            self.conductor_mask_status == "reviewed_same_scope_geometry_mask"
            and not self.source_references
        ):
            raise ValueError(
                "reviewed_same_scope_geometry_mask requires source_references"
            )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> BoundaryPolicy:
        return cls(
            pml_cells=int(value.get("pml_cells", 0)),
            pml_strength=float(value.get("pml_strength", 0.0)),
            particle_absorption_enabled=bool(
                value.get("particle_absorption_enabled", False)
            ),
            open_boundary=bool(value.get("open_boundary", True)),
            conductor_mask_status=str(
                value.get("conductor_mask_status", "not_supplied")
            ),
            conductor_mask_mode=str(value.get("conductor_mask_mode", "none")),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class DiagnosticPolicy:
    """Run length and output diagnostics requested by engineers."""

    n_steps: int = 1
    dt_s: float = 1.0e-13
    output_stride: int = 1
    history_stride: int = 1
    max_step_results: int | None = 256
    target_time_s: float | None = None
    emit_particle_history: bool = False

    def __post_init__(self) -> None:
        if int(self.n_steps) != self.n_steps or self.n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        _require_positive("dt_s", self.dt_s)
        if int(self.output_stride) != self.output_stride or self.output_stride <= 0:
            raise ValueError("output_stride must be a positive integer")
        if int(self.history_stride) != self.history_stride or self.history_stride <= 0:
            raise ValueError("history_stride must be a positive integer")
        if self.max_step_results is not None and (
            int(self.max_step_results) != self.max_step_results
            or self.max_step_results < 0
        ):
            raise ValueError("max_step_results must be a non-negative integer or None")
        if self.target_time_s is not None:
            _require_positive("target_time_s", self.target_time_s)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DiagnosticPolicy:
        return cls(
            n_steps=int(value.get("n_steps", 1)),
            dt_s=float(value.get("dt_s", 1.0e-13)),
            output_stride=int(value.get("output_stride", 1)),
            history_stride=int(
                value.get("history_stride", value.get("output_stride", 1))
            ),
            max_step_results=(
                None
                if "max_step_results" in value and value.get("max_step_results") is None
                else int(value.get("max_step_results", 256))
            ),
            target_time_s=(
                None
                if value.get("target_time_s") is None
                else float(value["target_time_s"])
            ),
            emit_particle_history=bool(value.get("emit_particle_history", False)),
        )


@dataclass(frozen=True)
class ValidationTargetReference:
    """Reference to an engineering comparison target."""

    name: str
    observable: str
    source_reference: SourceReference
    status: str = "candidate_or_missing"

    @property
    def target_sha256(self) -> str | None:
        return self.source_reference.sha256

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ValidationTargetReference:
        return cls(
            name=str(value["name"]),
            observable=str(value["observable"]),
            source_reference=SourceReference.from_mapping(value["source_reference"]),
            status=str(value.get("status", "candidate_or_missing")),
        )


@dataclass(frozen=True)
class FirstPrinciplesInputDeck:
    """Top-level first-principles-only input deck."""

    deck_id: str
    device: DeviceGeometryDeck
    circuit: CircuitDeck
    gas: GasDeck
    grid: GridDeck
    startup: StartupPolicy = field(default_factory=StartupPolicy)
    closures: ClosurePolicy = field(default_factory=ClosurePolicy)
    boundaries: BoundaryPolicy = field(default_factory=BoundaryPolicy)
    diagnostics: DiagnosticPolicy = field(default_factory=DiagnosticPolicy)
    source_references: tuple[SourceReference, ...] = ()
    validation_targets: tuple[ValidationTargetReference, ...] = ()
    scientific_status: str = "engineering_candidate_not_validation"
    schema_version: str = "dpf.first_principles.input_deck.v1"

    @property
    def validation_target_references(self) -> tuple[ValidationTargetReference, ...]:
        return self.validation_targets

    @classmethod
    def from_json_file(cls, path: str | Path) -> FirstPrinciplesInputDeck:
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def from_json(cls, value: str) -> FirstPrinciplesInputDeck:
        return cls.from_mapping(json.loads(value))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FirstPrinciplesInputDeck:
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> FirstPrinciplesInputDeck:
        try:
            _reject_reduced_model_authority(value)
            value = _normalize_deck_mapping(value)
            return cls(
                deck_id=str(value.get("deck_id", "first_principles_3d_engineering_deck")),
                device=DeviceGeometryDeck.from_mapping(value["device"]),
                circuit=CircuitDeck.from_mapping(value["circuit"]),
                gas=GasDeck.from_mapping(value.get("gas", {})),
                grid=GridDeck.from_mapping(value["grid"]),
                startup=StartupPolicy.from_mapping(value.get("startup", {})),
                closures=ClosurePolicy.from_mapping(value.get("closures", {})),
                boundaries=BoundaryPolicy.from_mapping(value.get("boundaries", {})),
                diagnostics=DiagnosticPolicy.from_mapping(value.get("diagnostics", {})),
                source_references=_source_refs(value.get("source_references", ())),
                validation_targets=tuple(
                    ValidationTargetReference.from_mapping(item)
                    for item in value.get("validation_targets", ())
                ),
                scientific_status=str(
                    value.get("scientific_status", "engineering_candidate_not_validation")
                ),
                schema_version=str(
                    value.get(
                        "schema_version",
                        "dpf.first_principles.input_deck.v1",
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _validation_error(str(exc), value) from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def minimal_engineering_deck(
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
) -> FirstPrinciplesInputDeck:
    """Return the smallest package-native 3-D first-principles engineering deck."""
    source = SourceReference(
        path=HYBRID_PIC_3D_SOURCE,
        record_id="kr:fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9",
        capability_tags=(
            "maxwell_fields",
            "hybrid_pic_fluid",
            "generalized_ohm",
            "circuit_coupling",
            "neutron_thermonuclear",
        ),
        role="architecture_source",
    )
    physical_radius_m = 0.015
    physical_length_m = 0.10
    spacing_m = (
        2.0 * physical_radius_m / max(int(shape[0]), 1),
        2.0 * physical_radius_m / max(int(shape[1]), 1),
        physical_length_m / max(int(shape[2]), 1),
    )
    return FirstPrinciplesInputDeck(
        deck_id="minimal_3d_hybrid_em_pic_fluid_engineering_candidate",
        device=DeviceGeometryDeck(
            name="LLNL-like engineering smoke geometry",
            anode_radius_m=1.0e-2,
            cathode_radius_m=1.5e-2,
            anode_length_m=5.0e-2,
            insulator_length_m=5.0e-3,
            source_references=(source,),
        ),
        circuit=CircuitDeck(
            capacitance_F=2.0e-5,
            voltage_V=1.5e4,
            inductance_H=1.1e-7,
            resistance_ohm=1.2e-2,
            initial_current_A=1.773e4,
            initial_charge_C=0.218,
            source_references=(source,),
        ),
        gas=GasDeck(source_references=(source,)),
        grid=GridDeck(shape=shape, spacing_m=spacing_m),
        startup=StartupPolicy(
            background_density_m3=6.7e22,
            electron_temperature_K=300.0,
            ion_temperature_K=300.0,
            startup_payload={
                "profile_type": "annular_axial_sheath",
                "background_density_m3": 6.7e22,
                "background_temperature_K": 300.0,
                "background_ionization_fraction": 1.0,
                "background_radial_min_m": 0.0,
                "background_radial_max_m": 0.015,
                "background_z_min_m": 0.050,
                "background_z_max_m": 0.100,
                "sheath_density_m3": 3.3e23,
                "sheath_temperature_K": 7.2e5,
                "sheath_ionization_fraction": 1.0,
                "sheath_radial_min_m": 0.010,
                "sheath_radial_max_m": 0.015,
                "sheath_z_min_m": 0.0445,
                "sheath_z_max_m": 0.0455,
                "sheath_drift_velocity_m_s": (0.0, 0.0, 1.1e5),
                "vacuum_density_floor_m3": 1.0,
                "vacuum_ionization_fraction": 0.0,
                "vacuum_temperature_K": 300.0,
                "source_references": (
                    {
                        "path": HYBRID_PIC_3D_SOURCE,
                        "lines": "632-740",
                        "role": "end_rundown_background_and_sheath_initialization",
                    },
                ),
            },
            source_references=(source,),
        ),
        closures=ClosurePolicy(source_references=(source,)),
        boundaries=BoundaryPolicy(
            pml_cells=1,
            pml_strength=0.1,
            particle_absorption_enabled=True,
            conductor_mask_status="candidate_geometry_mask",
            conductor_mask_mode="axisymmetric_coaxial_projection",
            source_references=(source,),
        ),
        diagnostics=DiagnosticPolicy(n_steps=n_steps, dt_s=dt_s),
        source_references=(source,),
    )


def pf1000_akel_16kv_engineering_deck(
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
    pressure_torr: float = 1.2,
) -> FirstPrinciplesInputDeck:
    """Return a source-scoped PF-1000/Akel engineering deck.

    This deck uses PF-1000/Akel shot-12581 machine, bank, and operating
    parameters as source metadata for the package-native runner.  It remains an
    engineering candidate because the source does not provide accepted
    first-principles startup, same-scope spatial/field/temperature, neutron
    mechanism, comparator/UQ, numerical-fidelity, or certificate packets.
    """

    source = SourceReference(
        path="KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        record_id="kr:akel-2021-pf1000-neutron-yield",
        capability_tags=(
            "dpf_device",
            "circuit_coupling",
            "startup_breakdown",
            "detector_response",
            "validation_target",
        ),
        role="pf1000_akel_16kv_shot_12581_source",
    )
    geometry_source = SourceReference(
        path=(
            "KnowledgeReference/"
            "experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        ),
        record_id="kr:krauz-2012-pf1000-electrode-geometry",
        capability_tags=(
            "dpf_device",
            "electrode_geometry",
            "cathode_rods",
            "insulator_geometry",
            "hollow_anode_context",
        ),
        role="pf1000_electrode_geometry_source",
    )
    pressure_pa = float(pressure_torr) * 133.32236842105263
    gas_temperature_K = 300.0
    background_density_m3 = _ideal_gas_number_density_m3(
        pressure_pa,
        gas_temperature_K,
    )
    cathode_inner_radius_m = 0.16
    cathode_rod_diameter_m = 0.080
    cathode_outer_radius_m = cathode_inner_radius_m + 0.5 * cathode_rod_diameter_m
    xy_spacing_m = (2.2 * cathode_outer_radius_m) / max(int(shape[0]) - 1, 1)
    z_spacing_m = (0.48 + 0.085) / max(int(shape[2]) - 1, 1)
    startup_breakdown = compute_breakdown(
        V0=1.6e4,
        fill_pressure_Pa=pressure_pa,
        anode_radius=0.1155,
        cathode_radius=cathode_inner_radius_m,
        insulator_length=0.085,
        gas_name="D2",
        I_seed=None,
    )
    return FirstPrinciplesInputDeck(
        deck_id="pf1000_akel_16kv_1p2torr_shot_12581_engineering_candidate",
        device=DeviceGeometryDeck(
            name="PF-1000/Akel shot 12581 engineering candidate",
            anode_radius_m=0.1155,
            cathode_radius_m=cathode_inner_radius_m,
            anode_length_m=0.48,
            insulator_length_m=0.085,
            cathode_rod_count=12,
            cathode_rod_diameter_m=cathode_rod_diameter_m,
            cathode_rod_length_m=0.48,
            insulator_material="alumina",
            source_references=(source, geometry_source),
        ),
        circuit=CircuitDeck(
            capacitance_F=1.332e-3,
            voltage_V=1.6e4,
            inductance_H=25.0e-9,
            resistance_ohm=6.1e-3,
            initial_current_A=0.0,
            initial_charge_C=0.0,
            source_references=(source,),
        ),
        gas=GasDeck(
            species="D",
            pressure_Pa=pressure_pa,
            temperature_K=gas_temperature_K,
            source_references=(source,),
        ),
        startup=StartupPolicy(
            mode="seeded_layer",
            background_density_m3=background_density_m3,
            initial_ionization_fraction=startup_breakdown.ionization_fraction,
            electron_temperature_K=startup_breakdown.Te_initial,
            ion_temperature_K=gas_temperature_K,
            initial_electric_field_V_m=(0.0, 0.0, 0.0),
            evidence_status="engineering_candidate_not_whole_shot",
            source_scope=(
                "pf1000_akel_candidate_paschen_insulator_seed_layer_not_startup_bvp"
            ),
            can_support_whole_shot_acceptance=False,
            missing_channels=(
                "breakdown_model",
                "preionization_state",
                "surface_flashover_closure",
                "initial_current_density_distribution",
                "sheath_liftoff",
            ),
            startup_payload={
                "profile_type": "annular_axial_sheath",
                "vacuum_density_floor_m3": background_density_m3,
                "vacuum_ionization_fraction": 0.0,
                "vacuum_temperature_K": gas_temperature_K,
                "sheath_radial_min_m": 0.1155,
                "sheath_radial_max_m": cathode_inner_radius_m,
                "sheath_z_min_m": 0.0,
                "sheath_z_max_m": max(0.085, 1.5 * z_spacing_m),
                "sheath_projection_note": (
                    "Coarse-grid candidate projection expands the seed layer to "
                    "the first non-PML axial cell when the physical insulator "
                    "length would otherwise be swallowed by the absorbing layer."
                ),
                "sheath_density_m3": background_density_m3,
                "sheath_ionization_fraction": startup_breakdown.ionization_fraction,
                "sheath_temperature_K": startup_breakdown.Te_initial,
                "initial_electric_field_note": (
                    "PF-1000 bank voltage is carried by the source-circuit state; "
                    "no reviewed startup BVP supplies a resolved chamber electric "
                    "field, so the candidate seeded layer starts with zero volume "
                    "electric field and is driven by the circuit boundary."
                ),
                "source_references": (
                    "KnowledgeReference/auluck-2021-dpf-circuit-element.md:151-209",
                    (
                        "KnowledgeReference/"
                        "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-"
                        "acb71fa9.md:657-666,734-736"
                    ),
                    (
                        "KnowledgeReference/"
                        "unlimited-release-printed-september-2009-alegra-hedp-"
                        "simulations-of-the-dense-plasma-focus.md:245-392"
                    ),
                ),
                "candidate_breakdown_initials": {
                    "status": "candidate_civ_paschen_initials_engineering_only",
                    "source_status": (
                        "civ_paschen_gas_coefficients_source_packets_missing"
                    ),
                    "mechanism": startup_breakdown.mechanism,
                    "initial_electron_temperature_K": startup_breakdown.Te_initial,
                    "initial_electron_temperature_eV": (
                        startup_breakdown.Te_initial_eV
                    ),
                    "initial_ionization_fraction": (
                        startup_breakdown.ionization_fraction
                    ),
                    "breakdown_time_s": startup_breakdown.breakdown_time,
                    "can_support_first_principles_acceptance": False,
                }
            },
            source_references=(source,),
        ),
        closures=ClosurePolicy(
            density_floor_m3=background_density_m3,
            apply_circuit_boundary=True,
            source_references=(source,),
        ),
        boundaries=BoundaryPolicy(
            pml_cells=1,
            pml_strength=0.1,
            particle_absorption_enabled=True,
            conductor_mask_status="candidate_geometry_mask",
            conductor_mask_mode="pf1000_rod_hollow_projection",
            source_references=(source, geometry_source),
        ),
        grid=GridDeck(shape=shape, spacing_m=(xy_spacing_m, xy_spacing_m, z_spacing_m)),
        diagnostics=DiagnosticPolicy(n_steps=n_steps, dt_s=dt_s),
        source_references=(source, geometry_source),
        validation_targets=(
            ValidationTargetReference(
                name="Akel PF-1000 shot 12581 current waveform figure",
                observable="current_waveform",
                source_reference=source,
                status="blocked_by_review",
            ),
            ValidationTargetReference(
                name="Akel PF-1000 shot 12581 scalar neutron yield",
                observable="neutron_scalar_yield",
                source_reference=source,
                status="candidate_text_reference_not_validation",
            ),
        ),
    )


def ir_mpf_100_engineering_deck(
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
    voltage_V: float = 2.0e4,
    pressure_torr: float = 1.9,
) -> FirstPrinciplesInputDeck:
    """Return a user-validated IR-MPF-100 engineering candidate deck.

    The deck is runnable by the package-native 3-D first-principles candidate,
    but it is not accepted validation evidence. It uses the May 15
    user-validated Salehizadeh 2012 source for machine, circuit, gas, and
    diagnostic context.
    """

    source = SourceReference(
        path="KnowledgeReference/original-research-f7894f85.md",
        sha256="f7894f85fd4d1826a5d98933453bd09664e260d46a2c9fedc4ce79491d2be4ad",
        record_id="kr:may15-user-validated-ir-mpf-100-salehizadeh-2012",
        capability_tags=(
            "dpf_device",
            "circuit_coupling",
            "startup_breakdown",
            "second_scope_candidate",
            "neutron_activation_context",
        ),
        role="ir_mpf_100_user_validated_source",
    )
    pressure_pa = float(pressure_torr) * 133.32236842105263
    gas_temperature_K = 300.0
    background_density_m3 = _ideal_gas_number_density_m3(
        pressure_pa,
        gas_temperature_K,
    )
    return FirstPrinciplesInputDeck(
        deck_id="ir_mpf_100_20kv_1p9torr_engineering_candidate",
        device=DeviceGeometryDeck(
            name="IR-MPF-100 20 kV / 1.9 Torr engineering candidate",
            anode_radius_m=6.25e-2,
            cathode_radius_m=1.02e-1,
            anode_length_m=2.2e-1,
            insulator_length_m=5.0e-2,
            source_references=(source,),
        ),
        circuit=CircuitDeck(
            capacitance_F=144.0e-6,
            voltage_V=float(voltage_V),
            inductance_H=120.0e-9,
            resistance_ohm=5.0e-3,
            initial_current_A=0.0,
            initial_charge_C=0.0,
            source_references=(source,),
        ),
        gas=GasDeck(
            species="D",
            pressure_Pa=pressure_pa,
            temperature_K=gas_temperature_K,
            source_references=(source,),
        ),
        grid=GridDeck(shape=shape, spacing_m=(2.6e-2, 2.6e-2, 5.5e-2)),
        startup=StartupPolicy(
            mode="source_backed_end_rundown_sheath",
            background_density_m3=background_density_m3,
            electron_temperature_K=gas_temperature_K,
            ion_temperature_K=gas_temperature_K,
            evidence_status="engineering_candidate_not_whole_shot",
            source_scope="ir_mpf_100_text_supported_machine_state_not_startup_bvp",
            can_support_whole_shot_acceptance=False,
            missing_channels=(
                "breakdown_model",
                "preionization_state",
                "surface_flashover_closure",
                "initial_current_density_distribution",
                "sheath_liftoff",
                "measured_current_waveform_digitization",
            ),
            source_references=(source,),
        ),
        closures=ClosurePolicy(
            density_floor_m3=background_density_m3,
            apply_circuit_boundary=True,
            source_references=(source,),
        ),
        boundaries=BoundaryPolicy(
            pml_cells=1,
            pml_strength=0.1,
            particle_absorption_enabled=True,
            conductor_mask_status="candidate_geometry_mask",
            conductor_mask_mode="axisymmetric_coaxial_projection",
            source_references=(source,),
        ),
        diagnostics=DiagnosticPolicy(n_steps=n_steps, dt_s=dt_s),
        source_references=(source,),
        validation_targets=(
            ValidationTargetReference(
                name="IR-MPF-100 current/voltage/hard-X-ray source figures",
                observable="waveform_and_diagnostic_targets",
                source_reference=source,
                status="source_accepted_targets_not_digitized",
            ),
            ValidationTargetReference(
                name="IR-MPF-100 activation neutron scalar yield",
                observable="neutron_scalar_yield",
                source_reference=source,
                status="source_accepted_target_not_mechanism_separated",
            ),
        ),
    )


def compact_chinese_dpf_engineering_deck(
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
    voltage_V: float = 2.0e4,
    pressure_Pa: float = 580.0,
) -> FirstPrinciplesInputDeck:
    """Return a user-validated compact Mather DPF engineering deck.

    The source gives bank capacitance, voltage range, approximate delivered
    current, geometry, pressure, focus time, and neutron-pulse targets. The
    inductance is inferred from the source's bank/current/voltage values only
    to make the engineering deck executable, and cannot support acceptance.
    """

    source = SourceReference(
        path="KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md",
        sha256="d1758d55ea9a32f6edb17107a86b033d8078cad337f0531ca10f18190fb220b5",
        record_id="kr:may15-user-validated-compact-chinese-dpf-2018",
        capability_tags=(
            "dpf_device",
            "circuit_coupling",
            "startup_breakdown",
            "second_scope_candidate",
            "tof_neutron_context",
        ),
        role="compact_chinese_dpf_user_validated_source",
    )
    capacitance_F = 40.0e-6
    gas_temperature_K = 300.0
    pressure_pa = float(pressure_Pa)
    background_density_m3 = _ideal_gas_number_density_m3(
        pressure_pa,
        gas_temperature_K,
    )
    delivered_current_A = 400.0e3
    inferred_inductance_H = capacitance_F * (float(voltage_V) / delivered_current_A) ** 2
    return FirstPrinciplesInputDeck(
        deck_id="compact_chinese_dpf_20kv_580pa_engineering_candidate",
        device=DeviceGeometryDeck(
            name="Compact Chinese Mather DPF 20 kV / 580 Pa engineering candidate",
            anode_radius_m=17.0e-3,
            cathode_radius_m=40.0e-3,
            anode_length_m=15.0e-2,
            insulator_length_m=40.0e-3,
            source_references=(source,),
        ),
        circuit=CircuitDeck(
            capacitance_F=capacitance_F,
            voltage_V=float(voltage_V),
            inductance_H=inferred_inductance_H,
            resistance_ohm=0.0,
            initial_current_A=0.0,
            initial_charge_C=0.0,
            source_references=(source,),
        ),
        gas=GasDeck(
            species="D",
            pressure_Pa=pressure_pa,
            temperature_K=gas_temperature_K,
            source_references=(source,),
        ),
        grid=GridDeck(shape=shape, spacing_m=(1.0e-2, 1.0e-2, 4.0e-2)),
        startup=StartupPolicy(
            mode="source_backed_end_rundown_sheath",
            background_density_m3=background_density_m3,
            electron_temperature_K=gas_temperature_K,
            ion_temperature_K=gas_temperature_K,
            evidence_status="engineering_candidate_not_whole_shot",
            source_scope=(
                "compact_chinese_dpf_text_supported_machine_state_with_"
                "inferred_circuit_inductance_not_startup_bvp"
            ),
            can_support_whole_shot_acceptance=False,
            missing_channels=(
                "breakdown_model",
                "preionization_state",
                "surface_flashover_closure",
                "initial_current_density_distribution",
                "sheath_liftoff",
                "visual_table_review",
                "translation_review",
            ),
            source_references=(source,),
        ),
        closures=ClosurePolicy(
            density_floor_m3=background_density_m3,
            apply_circuit_boundary=True,
            source_references=(source,),
        ),
        boundaries=BoundaryPolicy(
            pml_cells=1,
            pml_strength=0.1,
            particle_absorption_enabled=True,
            conductor_mask_status="candidate_geometry_mask",
            conductor_mask_mode="axisymmetric_coaxial_projection",
            source_references=(source,),
        ),
        diagnostics=DiagnosticPolicy(n_steps=n_steps, dt_s=dt_s),
        source_references=(source,),
        validation_targets=(
            ValidationTargetReference(
                name="Compact DPF pressure-yield and current waveform figures",
                observable="pressure_yield_current_waveform",
                source_reference=source,
                status="source_accepted_targets_not_digitized",
            ),
            ValidationTargetReference(
                name="Compact DPF neutron TOF/FWHM text target",
                observable="neutron_tof_fwhm",
                source_reference=source,
                status="source_accepted_target_not_detector_uq",
            ),
        ),
    )


def willenborg_hendricks_engineering_deck(
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
    voltage_V: float = 1.9e4,
    pressure_torr: float = 1.0,
) -> FirstPrinciplesInputDeck:
    """Return a user-validated Willenborg/Hendricks startup-design deck."""

    source = SourceReference(
        path=(
            "KnowledgeReference/"
            "design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md"
        ),
        sha256="12205ba4bb0d1edc11b069dda4e0e084b89597a8f14ff61c3a65e0b712926a75",
        record_id="kr:may15-user-validated-willenborg-hendricks-ada037245",
        capability_tags=(
            "dpf_device",
            "startup_breakdown",
            "insulator_conditioning",
            "diagnostic_design",
            "second_scope_candidate",
        ),
        role="willenborg_hendricks_user_validated_source",
    )
    pressure_pa = float(pressure_torr) * 133.32236842105263
    gas_temperature_K = 300.0
    background_density_m3 = _ideal_gas_number_density_m3(
        pressure_pa,
        gas_temperature_K,
    )
    return FirstPrinciplesInputDeck(
        deck_id="willenborg_hendricks_19kv_1torr_engineering_candidate",
        device=DeviceGeometryDeck(
            name="Willenborg/Hendricks DPF 19 kV / 1 Torr engineering candidate",
            anode_radius_m=(1.78 * 0.0254) / 2.0,
            cathode_radius_m=(1.78 * 0.0254) / 2.0 + (1.13 * 0.0254),
            anode_length_m=9.0 * 0.0254,
            insulator_length_m=2.93 * 0.0254,
            source_references=(source,),
        ),
        circuit=CircuitDeck(
            capacitance_F=43.5e-6,
            voltage_V=float(voltage_V),
            inductance_H=100.0e-9,
            resistance_ohm=0.03,
            initial_current_A=0.0,
            initial_charge_C=0.0,
            source_references=(source,),
        ),
        gas=GasDeck(
            species="D",
            pressure_Pa=pressure_pa,
            temperature_K=gas_temperature_K,
            source_references=(source,),
        ),
        grid=GridDeck(shape=shape, spacing_m=(1.3e-2, 1.3e-2, 5.8e-2)),
        startup=StartupPolicy(
            mode="surface_breakdown_bvp",
            background_density_m3=background_density_m3,
            electron_temperature_K=gas_temperature_K,
            ion_temperature_K=gas_temperature_K,
            evidence_status="engineering_candidate_not_whole_shot",
            source_scope="historical_startup_design_constraints_not_modern_startup_bvp",
            can_support_whole_shot_acceptance=False,
            missing_channels=(
                "surface_flashover_equations",
                "secondary_emission_or_material_model",
                "avalanche_streamer_closure",
                "preionization_model",
                "initial_current_density_distribution",
                "sheath_liftoff",
                "modern_device_scope_review",
            ),
            source_references=(source,),
        ),
        closures=ClosurePolicy(
            density_floor_m3=background_density_m3,
            apply_circuit_boundary=True,
            source_references=(source,),
        ),
        boundaries=BoundaryPolicy(
            pml_cells=1,
            pml_strength=0.1,
            particle_absorption_enabled=True,
            conductor_mask_status="candidate_geometry_mask",
            conductor_mask_mode="axisymmetric_coaxial_projection",
            source_references=(source,),
        ),
        diagnostics=DiagnosticPolicy(n_steps=n_steps, dt_s=dt_s),
        source_references=(source,),
        validation_targets=(
            ValidationTargetReference(
                name="Willenborg/Hendricks voltage current X-ray timing",
                observable="startup_diagnostic_timing",
                source_reference=source,
                status="source_accepted_targets_not_digitized",
            ),
        ),
    )


def gv_verified_engineering_deck(
    shot_id: str = "pf24_krakow_16092202",
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
) -> FirstPrinciplesInputDeck:
    """Return a non-promoting engineering deck from the verified GV shot bundle.

    The GV bundle supplies machine geometry, lumped circuit values, fitted gas
    pressure, a GV reduced-model current baseline, and workbook current
    waveform columns. It does not supply a first-principles startup BVP or
    spatial plasma state, so every deck built here remains an engineering
    candidate only.
    """

    row = _gv_verified_shot_row(shot_id)
    geometry_mm = row["geometry_mm"]
    circuit = row["circuit"]
    gas = row["gas"]

    anode_radius_m = float(geometry_mm["anode_radius"]) * 1.0e-3
    cathode_radius_m = float(geometry_mm["cathode_radius"]) * 1.0e-3
    anode_length_m = float(geometry_mm["anode_length"]) * 1.0e-3
    insulator_length_m = float(geometry_mm["insulator_length"]) * 1.0e-3
    pressure_torr = float(gas["fitted_pressure_torr"])
    gas_temperature_K = 300.0
    pressure_pa = pressure_torr * 133.32236842105263
    background_density_m3 = _ideal_gas_number_density_m3(
        pressure_pa,
        gas_temperature_K,
    )

    input_source = SourceReference(
        path=f"/Users/anthonyzamora/Downloads/GV/{row['input_file']}",
        sha256=str(row["input_sha256"]),
        record_id=f"gv:{shot_id}:input",
        capability_tags=(
            "dpf_device",
            "circuit_coupling",
            "second_scope_candidate",
        ),
        role="gv_verified_input_deck_candidate",
    )
    waveform_source = SourceReference(
        path=f"/Users/anthonyzamora/Downloads/GV/{row['xlsx_file']}",
        sha256=str(row["xlsx_sha256"]),
        record_id=f"gv:{shot_id}:workbook",
        capability_tags=(
            "current_waveform",
            "validation_target",
            "second_scope_candidate",
        ),
        role="gv_verified_workbook_waveform_candidate",
    )
    baseline_source = SourceReference(
        path=f"/Users/anthonyzamora/Downloads/GV/{row['txt_file']}",
        sha256=str(row["txt_sha256"]),
        record_id=f"gv:{shot_id}:reduced_model_output",
        capability_tags=(
            "reduced_model_baseline",
            "current_waveform",
        ),
        role="gv_reduced_model_baseline_not_authority",
    )

    radial_extent_m = max(2.2 * cathode_radius_m, 1.0e-3)
    axial_extent_m = max(anode_length_m + insulator_length_m, 1.0e-3)
    spacing_m = (
        radial_extent_m / max(int(shape[0]) - 1, 1),
        radial_extent_m / max(int(shape[1]) - 1, 1),
        axial_extent_m / max(int(shape[2]) - 1, 1),
    )

    return FirstPrinciplesInputDeck(
        deck_id=f"gv_{shot_id}_engineering_candidate",
        device=DeviceGeometryDeck(
            name=f"{row['device']} {shot_id} GV verified-shot engineering candidate",
            anode_radius_m=anode_radius_m,
            cathode_radius_m=cathode_radius_m,
            anode_length_m=anode_length_m,
            insulator_length_m=insulator_length_m,
            source_references=(input_source,),
        ),
        circuit=CircuitDeck(
            capacitance_F=float(circuit["capacitance_uF"]) * 1.0e-6,
            voltage_V=float(circuit["voltage_kV"]) * 1.0e3,
            inductance_H=float(circuit["inductance_nH"]) * 1.0e-9,
            resistance_ohm=float(circuit["resistance_milliohm"]) * 1.0e-3,
            initial_current_A=0.0,
            initial_charge_C=0.0,
            source_references=(input_source,),
        ),
        gas=GasDeck(
            species=str(gas.get("species", "D")),
            pressure_Pa=pressure_pa,
            temperature_K=gas_temperature_K,
            source_references=(input_source,),
        ),
        grid=GridDeck(shape=shape, spacing_m=spacing_m),
        startup=StartupPolicy(
            mode="source_backed_end_rundown_sheath",
            background_density_m3=background_density_m3,
            electron_temperature_K=gas_temperature_K,
            ion_temperature_K=gas_temperature_K,
            evidence_status="engineering_candidate_not_whole_shot",
            source_scope=(
                "gv_verified_machine_current_waveform_candidate_not_startup_bvp"
            ),
            can_support_whole_shot_acceptance=False,
            missing_channels=(
                "breakdown_model",
                "preionization_state",
                "surface_flashover_closure",
                "initial_current_density_distribution",
                "sheath_liftoff",
                "spatial_density_field_temperature_history",
                "neutron_mechanism_separation",
                "detector_response_and_uq",
            ),
            source_references=(input_source, waveform_source),
        ),
        closures=ClosurePolicy(
            density_floor_m3=background_density_m3,
            apply_circuit_boundary=True,
            source_references=(input_source,),
        ),
        boundaries=BoundaryPolicy(
            pml_cells=1,
            pml_strength=0.1,
            particle_absorption_enabled=True,
            conductor_mask_status="candidate_geometry_mask",
            conductor_mask_mode="axisymmetric_coaxial_projection",
            source_references=(input_source,),
        ),
        diagnostics=DiagnosticPolicy(n_steps=n_steps, dt_s=dt_s),
        source_references=(input_source, waveform_source, baseline_source),
        validation_targets=(
            ValidationTargetReference(
                name=f"{row['device']} {shot_id} workbook current waveform",
                observable="current_waveform",
                source_reference=waveform_source,
                status="user_verified_waveform_candidate_not_comparator_bound",
            ),
            ValidationTargetReference(
                name=f"{row['device']} {shot_id} GV current baseline",
                observable="reduced_model_current_baseline",
                source_reference=baseline_source,
                status="reduced_model_baseline_not_first_principles_closure",
            ),
        ),
    )


def gv_verified_engineering_decks(
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
) -> tuple[FirstPrinciplesInputDeck, ...]:
    """Return runnable non-promoting decks for every unique verified GV shot."""

    from dpf.first_principles.source_targets import GV_VERIFIED_SHOTS

    return tuple(
        gv_verified_engineering_deck(
            str(row["shot_id"]),
            n_steps=n_steps,
            shape=shape,
            dt_s=dt_s,
        )
        for row in GV_VERIFIED_SHOTS
    )


def may15_second_scope_engineering_decks(
    *,
    n_steps: int = 1,
    shape: tuple[int, int, int] = (5, 5, 5),
    dt_s: float = 1.0e-13,
) -> tuple[FirstPrinciplesInputDeck, ...]:
    """Return runnable non-promoting decks from the May 15 validated sources."""

    return (
        ir_mpf_100_engineering_deck(n_steps=n_steps, shape=shape, dt_s=dt_s),
        compact_chinese_dpf_engineering_deck(n_steps=n_steps, shape=shape, dt_s=dt_s),
        willenborg_hendricks_engineering_deck(n_steps=n_steps, shape=shape, dt_s=dt_s),
    )


def _gv_verified_shot_row(shot_id: str) -> dict[str, Any]:
    from dpf.first_principles.source_targets import GV_VERIFIED_SHOTS

    normalized = str(shot_id).strip().lower()
    for row in GV_VERIFIED_SHOTS:
        if str(row["shot_id"]).lower() == normalized:
            return row
    allowed = ", ".join(str(row["shot_id"]) for row in GV_VERIFIED_SHOTS)
    raise ValueError(f"unknown GV verified shot {shot_id!r}; expected one of {allowed}")


def deck_hash(deck: FirstPrinciplesInputDeck) -> str:
    import hashlib

    payload = json.dumps(deck.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_first_principles_input_deck(
    value: str | Path | dict[str, Any] | FirstPrinciplesInputDeck,
) -> FirstPrinciplesInputDeck:
    """Load a first-principles deck from an object, mapping, or JSON file."""
    if isinstance(value, FirstPrinciplesInputDeck):
        return value
    if isinstance(value, dict):
        return FirstPrinciplesInputDeck.from_mapping(value)
    return FirstPrinciplesInputDeck.from_json_file(value)


def _reject_reduced_model_authority(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            low = str(key).strip().lower()
            if low in REDUCED_MODEL_AUTHORITY_FIELDS:
                raise ValueError(
                    "Reduced-model authority fields are not allowed in a "
                    f"first-principles input deck: {path}.{key}"
                )
            _reject_reduced_model_authority(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_reduced_model_authority(item, path=f"{path}[{index}]")


def _normalize_deck_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Accept both the compact runner schema and the richer review schema."""
    if "device" in value and "grid" in value and "circuit" in value:
        return dict(value)

    normalized = dict(value)
    if "device_geometry" in normalized:
        geom = dict(normalized["device_geometry"])
        normalized["device"] = {
            "name": geom.get("name", normalized.get("deck_id", "device")),
            "anode_radius_m": geom["anode_radius_m"],
            "cathode_radius_m": geom["cathode_radius_m"],
            "anode_length_m": geom.get(
                "anode_length_m",
                geom.get("cathode_length_m", 1.0e-2),
            ),
            "insulator_length_m": geom.get("insulator_length_m", 0.0),
            "anode_inner_radius_m": geom.get("anode_inner_radius_m"),
            "cathode_rod_count": geom.get("cathode_rod_count"),
            "cathode_rod_diameter_m": geom.get("cathode_rod_diameter_m"),
            "cathode_rod_length_m": geom.get("cathode_rod_length_m"),
            "insulator_outer_radius_m": geom.get("insulator_outer_radius_m"),
            "insulator_material": geom.get("insulator_material"),
            "source_references": _references_from_ids(
                geom.get("source_reference_ids", ()),
                normalized.get("source_references", ()),
            ),
        }
    if "circuit" in normalized:
        circuit = dict(normalized["circuit"])
        if "voltage_V" not in circuit and "initial_voltage_V" in circuit:
            circuit["voltage_V"] = circuit["initial_voltage_V"]
        if "inductance_H" not in circuit and "static_inductance_H" in circuit:
            circuit["inductance_H"] = circuit["static_inductance_H"]
        if "resistance_ohm" not in circuit and "static_resistance_ohm" in circuit:
            circuit["resistance_ohm"] = circuit["static_resistance_ohm"]
        circuit["source_references"] = _references_from_ids(
            circuit.get("source_reference_ids", ()),
            normalized.get("source_references", ()),
        )
        normalized["circuit"] = circuit
    if "gas" in normalized:
        gas = dict(normalized["gas"])
        species = gas.get("species", "D")
        if isinstance(species, list) and species:
            first = dict(species[0])
            species_name = str(first.get("name", "D"))
            source_ids = first.get("source_reference_ids", ())
        else:
            species_name = str(species)
            source_ids = gas.get("source_reference_ids", ())
        normalized["gas"] = {
            "species": species_name,
            "pressure_Pa": gas.get("pressure_Pa", gas.get("fill_pressure_Pa", 266.0)),
            "temperature_K": gas.get(
                "temperature_K",
                gas.get("fill_temperature_K", 300.0),
            ),
            "ion_mass_kg": gas.get("ion_mass_kg", 3.344e-27),
            "ion_charge_C": gas.get("ion_charge_C", ELEMENTARY_CHARGE),
            "source_references": _references_from_ids(
                source_ids,
                normalized.get("source_references", ()),
            ),
        }
    if "startup_policy" in normalized:
        startup = dict(normalized["startup_policy"])
        normalized["startup"] = {
            "mode": startup.get(
                "mode",
                startup.get("initialization", "source_backed_end_rundown_sheath"),
            ),
            "background_density_m3": startup.get("background_density_m3", 1.0e20),
            "electron_temperature_K": startup.get("electron_temperature_K", 1.0e5),
            "ion_temperature_K": startup.get("ion_temperature_K", 1.0e5),
            "initial_electric_field_V_m": startup.get(
                "initial_electric_field_V_m",
                (1.0e5, 0.0, 0.0),
            ),
            "initial_magnetic_field_T": startup.get(
                "initial_magnetic_field_T",
                (0.0, 0.0, 0.0),
            ),
            "particle_weight": startup.get("particle_weight", 1.0e8),
            "evidence_status": startup.get(
                "evidence_status",
                "engineering_candidate_not_whole_shot",
            ),
            "source_scope": startup.get(
                "source_scope",
                "end_of_rundown_or_engineering_startup",
            ),
            "can_support_whole_shot_acceptance": startup.get(
                "can_support_whole_shot_acceptance",
                startup.get("can_support_first_principles_startup", False),
            ),
            "required_channels": startup.get("required_channels", ()),
            "missing_channels": startup.get("missing_channels", ()),
            "accepted_channels": startup.get("accepted_channels", ()),
            "startup_payload": startup.get(
                "startup_payload",
                startup.get("payload", {}),
            ),
            "source_references": _references_from_ids(
                startup.get("source_reference_ids", ()),
                normalized.get("source_references", ()),
            ),
        }
    if "closure_policy" in normalized:
        closures = dict(normalized["closure_policy"])
        normalized["closures"] = {
            "circuit_udpf_mode": closures.get(
                "circuit_udpf_mode",
                "lagged_volume_j_dot_e",
            ),
            "source_references": _references_from_ids(
                closures.get("source_reference_ids", ()),
                normalized.get("source_references", ()),
            ),
        }
    if "boundary_policy" in normalized:
        boundaries = dict(normalized["boundary_policy"])
        normalized["boundaries"] = {
            "pml_cells": boundaries.get("pml_cells", 0),
            "pml_strength": boundaries.get("pml_strength", 0.0),
            "particle_absorption_enabled": boundaries.get(
                "particle_absorption_enabled",
                False,
            ),
            "open_boundary": boundaries.get("open_boundary", True),
            "conductor_mask_status": boundaries.get(
                "conductor_mask_status",
                "not_supplied",
            ),
            "conductor_mask_mode": boundaries.get("conductor_mask_mode", "none"),
            "source_references": _references_from_ids(
                boundaries.get("source_reference_ids", ()),
                normalized.get("source_references", ()),
            ),
        }
    if "diagnostic_policy" in normalized:
        diag = dict(normalized["diagnostic_policy"])
        normalized["diagnostics"] = {
            "n_steps": diag.get("n_steps", 1),
            "dt_s": diag.get("dt_s", 1.0e-13),
            "output_stride": diag.get("sample_interval_steps", 1),
            "history_stride": diag.get(
                "history_stride",
                diag.get("sample_interval_steps", 1),
            ),
            "max_step_results": diag.get("max_step_results", 256),
            "target_time_s": diag.get("target_time_s"),
        }
    if "validation_target_references" in normalized:
        targets = []
        refs = {
            str(item.get("source_id") or item.get("record_id") or item.get("path")): item
            for item in normalized.get("source_references", ())
            if isinstance(item, dict)
        }
        for target in normalized["validation_target_references"]:
            target = dict(target)
            ref_id = str(target.get("source_reference_id", ""))
            if ref_id and ref_id not in refs:
                raise ValueError(f"unknown source_reference_id {ref_id!r}")
            source_ref = refs.get(ref_id) or {
                "path": target.get("target_path", "unknown"),
                "sha256": target.get("target_sha256"),
            }
            targets.append({
                "name": target.get("target_id", target.get("name", "target")),
                "observable": target["observable"],
                "status": target.get("status", "candidate_or_missing"),
                "source_reference": {
                    "path": target.get(
                        "target_path",
                        source_ref.get("path", "unknown"),
                    ),
                    "sha256": target.get(
                        "target_sha256",
                        source_ref.get("sha256"),
                    ),
                    "record_id": source_ref.get("source_id") or source_ref.get("record_id"),
                    "role": "validation_target_source",
                },
            })
        normalized["validation_targets"] = targets
    return normalized


def _references_from_ids(
    ids: Any,
    source_references: Any,
) -> tuple[dict[str, Any], ...]:
    refs = {
        str(item.get("source_id") or item.get("record_id") or item.get("path")): item
        for item in source_references or ()
        if isinstance(item, dict)
    }
    out = []
    for source_id in ids or ():
        key = str(source_id)
        if key not in refs:
            raise ValueError(f"unknown source_reference_id {key!r}")
        ref = refs[key]
        out.append({
            "path": ref["path"],
            "sha256": ref.get("sha256"),
            "record_id": ref.get("source_id") or ref.get("record_id"),
            "role": "source",
        })
    return tuple(out)


def _source_refs(value: Any) -> tuple[SourceReference, ...]:
    return tuple(SourceReference.from_mapping(item) for item in value or ())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _require_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three values")
    return items  # type: ignore[return-value]


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _validation_error(message: str, input_value: Any) -> ValidationError:
    return ValidationError.from_exception_data(
        "FirstPrinciplesInputDeck",
        [
            {
                "type": "value_error",
                "loc": ("__root__",),
                "input": input_value,
                "ctx": {"error": ValueError(message)},
            }
        ],
    )
