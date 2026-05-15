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
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE

ELEMENTARY_CHARGE = dpf_constants.e

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
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        _require_positive("anode_radius_m", self.anode_radius_m)
        _require_positive("cathode_radius_m", self.cathode_radius_m)
        _require_positive("anode_length_m", self.anode_length_m)
        if self.insulator_length_m < 0.0:
            raise ValueError("insulator_length_m must be non-negative")
        if self.anode_radius_m >= self.cathode_radius_m:
            raise ValueError("anode_radius_m must be less than cathode_radius_m")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DeviceGeometryDeck:
        return cls(
            name=str(value["name"]),
            anode_radius_m=float(value["anode_radius_m"]),
            cathode_radius_m=float(value["cathode_radius_m"]),
            anode_length_m=float(value["anode_length_m"]),
            insulator_length_m=float(value.get("insulator_length_m", 0.0)),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class CircuitDeck:
    """External circuit inputs for the resolved field power-port candidate."""

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
        return self.capacitance_F * self.voltage_V

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
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in STARTUP_MODES:
            allowed = ", ".join(sorted(STARTUP_MODES))
            raise ValueError(f"unknown startup mode {self.mode!r}; expected one of {allowed}")
        _require_positive("background_density_m3", self.background_density_m3)
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
            and self.evidence_status != "reviewed"
            and self.can_support_whole_shot_acceptance
        ):
            raise ValueError(
                "imported_pic_sheath_state can support accepted whole-shot startup "
                "only after evidence_status='reviewed'"
            )

    @property
    def whole_shot_startup_blocked(self) -> bool:
        return not self.can_support_whole_shot_acceptance

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> StartupPolicy:
        return cls(
            mode=str(value.get("mode", "source_backed_end_rundown_sheath")),
            background_density_m3=float(value.get("background_density_m3", 1.0e20)),
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
    marder_factor_scale: float = 1.0e-6
    marder_nondominance_threshold: float = 0.5
    apply_circuit_boundary: bool = True
    circuit_udpf_V: float = 0.0
    source_references: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        if self.sigma0_S_m < 0.0:
            raise ValueError("sigma0_S_m must be non-negative")
        _require_positive("ohmic_cfl_safety", self.ohmic_cfl_safety)
        _require_positive("density_floor_m3", self.density_floor_m3)
        if self.marder_factor_scale < 0.0:
            raise ValueError("marder_factor_scale must be non-negative")

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
            marder_factor_scale=float(value.get("marder_factor_scale", 1.0e-6)),
            marder_nondominance_threshold=float(
                value.get("marder_nondominance_threshold", 0.5)
            ),
            apply_circuit_boundary=bool(value.get("apply_circuit_boundary", True)),
            circuit_udpf_V=float(value.get("circuit_udpf_V", 0.0)),
            source_references=_source_refs(value.get("source_references", ())),
        )


@dataclass(frozen=True)
class DiagnosticPolicy:
    """Run length and output diagnostics requested by engineers."""

    n_steps: int = 1
    dt_s: float = 1.0e-13
    output_stride: int = 1
    emit_particle_history: bool = False

    def __post_init__(self) -> None:
        if int(self.n_steps) != self.n_steps or self.n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        _require_positive("dt_s", self.dt_s)
        if int(self.output_stride) != self.output_stride or self.output_stride <= 0:
            raise ValueError("output_stride must be a positive integer")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DiagnosticPolicy:
        return cls(
            n_steps=int(value.get("n_steps", 1)),
            dt_s=float(value.get("dt_s", 1.0e-13)),
            output_stride=int(value.get("output_stride", 1)),
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
    return FirstPrinciplesInputDeck(
        deck_id="minimal_3d_hybrid_em_pic_fluid_engineering_candidate",
        device=DeviceGeometryDeck(
            name="LLNL-like engineering smoke geometry",
            anode_radius_m=3.0e-3,
            cathode_radius_m=1.0e-2,
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
        grid=GridDeck(shape=shape, spacing_m=(1.0e-3, 1.0e-3, 1.0e-3)),
        startup=StartupPolicy(source_references=(source,)),
        closures=ClosurePolicy(source_references=(source,)),
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
    pressure_pa = float(pressure_torr) * 133.32236842105263
    return FirstPrinciplesInputDeck(
        deck_id="pf1000_akel_16kv_1p2torr_shot_12581_engineering_candidate",
        device=DeviceGeometryDeck(
            name="PF-1000/Akel shot 12581 engineering candidate",
            anode_radius_m=0.1155,
            cathode_radius_m=0.16,
            anode_length_m=0.48,
            insulator_length_m=0.0,
            source_references=(source,),
        ),
        circuit=CircuitDeck(
            capacitance_F=1.332e-3,
            voltage_V=1.6e4,
            inductance_H=25.0e-9,
            resistance_ohm=6.1e-3,
            initial_current_A=0.0,
            initial_charge_C=1.332e-3 * 1.6e4,
            source_references=(source,),
        ),
        gas=GasDeck(
            species="D",
            pressure_Pa=pressure_pa,
            temperature_K=300.0,
            source_references=(source,),
        ),
        grid=GridDeck(shape=shape, spacing_m=(4.0e-2, 4.0e-2, 1.2e-1)),
        startup=StartupPolicy(
            mode="source_backed_end_rundown_sheath",
            evidence_status="engineering_candidate_not_whole_shot",
            source_scope="pf1000_akel_text_supported_machine_state_not_startup_bvp",
            can_support_whole_shot_acceptance=False,
            missing_channels=(
                "breakdown_model",
                "preionization_state",
                "surface_flashover_closure",
                "initial_current_density_distribution",
                "sheath_liftoff",
            ),
            source_references=(source,),
        ),
        closures=ClosurePolicy(
            apply_circuit_boundary=True,
            source_references=(source,),
        ),
        diagnostics=DiagnosticPolicy(n_steps=n_steps, dt_s=dt_s),
        source_references=(source,),
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
            "source_references": _references_from_ids(
                startup.get("source_reference_ids", ()),
                normalized.get("source_references", ()),
            ),
        }
    if "closure_policy" in normalized:
        closures = dict(normalized["closure_policy"])
        normalized["closures"] = {
            "source_references": _references_from_ids(
                closures.get("source_reference_ids", ()),
                normalized.get("source_references", ()),
            ),
        }
    if "diagnostic_policy" in normalized:
        diag = dict(normalized["diagnostic_policy"])
        normalized["diagnostics"] = {
            "n_steps": diag.get("n_steps", 1),
            "dt_s": diag.get("dt_s", 1.0e-13),
            "output_stride": diag.get("sample_interval_steps", 1),
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
