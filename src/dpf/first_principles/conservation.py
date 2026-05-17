"""Conservation ledger helpers for package-native first-principles runs.

These helpers record engineering-candidate conservation telemetry for the 3-D
hybrid EM/PIC-fluid path. They deliberately do not import validation workflow
modules or promote a run to scientific validation evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np

EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

RUN_STATUS_ENGINEERING_CANDIDATE = "engineering_candidate"
VALIDATION_STATUS_NOT_VALIDATION = "not_validation"
ARTIFACT_STATUS_ENGINEERING_CANDIDATE = "engineering_candidate_not_validation"


@dataclass(frozen=True)
class SourceIndexReference:
    """Reference into a local source-truth index without accepting validation."""

    source_id: str
    path: str
    scope: str = "first_principles_3d_hybrid_pic_fluid"
    status: str = "source_reference_not_validation"
    lines: str | None = None
    digest: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnergySnapshot:
    """Total energy at one point in a run."""

    total_J: float
    electric_J: float | None = None
    magnetic_J: float | None = None
    capacitive_J: float | None = None
    inductive_J: float | None = None

    def __post_init__(self) -> None:
        _require_finite("total_J", self.total_J)
        for name in ("electric_J", "magnetic_J", "capacitive_J", "inductive_J"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnergyDeltaLedger:
    """Initial/final energy accounting for one subsystem."""

    status: str
    initial_J: float | None = None
    final_J: float | None = None
    delta_J: float | None = None
    relative_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldEnergyLedger:
    """Field-energy conservation record."""

    status: str
    total: EnergyDeltaLedger
    electric: EnergyDeltaLedger
    magnetic: EnergyDeltaLedger

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CircuitEnergyLedger:
    """Circuit-energy conservation record."""

    status: str
    total: EnergyDeltaLedger
    capacitive: EnergyDeltaLedger
    inductive: EnergyDeltaLedger

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParticleCountLedger:
    """Macroparticle-count accounting."""

    status: str
    initial_count: int | None = None
    final_count: int | None = None
    delta_count: int | None = None
    relative_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualLedger:
    """Measured divergence and current residual accounting."""

    status: str
    gauss_law_linf: float | None = None
    div_B_linf: float | None = None
    current_continuity_linf_A_m3: float | None = None
    current_residual_linf_A_m2: float | None = None
    missing_channels: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FirstPrinciplesConservationLedger:
    """Fail-closed conservation ledger for a first-principles candidate run."""

    ledger_version: str = "1.0"
    run_status: str = RUN_STATUS_ENGINEERING_CANDIDATE
    validation_status: str = VALIDATION_STATUS_NOT_VALIDATION
    artifact_status: str = ARTIFACT_STATUS_ENGINEERING_CANDIDATE
    field_energy: FieldEnergyLedger | None = None
    circuit_energy: CircuitEnergyLedger | None = None
    particle_count: ParticleCountLedger | None = None
    electron_energy: EnergyDeltaLedger | None = None
    residuals: ResidualLedger = field(default_factory=ResidualLedger)
    source_index_references: tuple[SourceIndexReference, ...] = field(default_factory=tuple)
    can_support_first_principles_acceptance: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_index_references"] = [
            ref.to_dict() for ref in self.source_index_references
        ]
        return payload


def compute_field_energy(
    electric_field_V_m: np.ndarray,
    magnetic_field_T: np.ndarray,
    *,
    cell_volume_m3: float,
) -> EnergySnapshot:
    """Compute cell-centered EM field energy in SI units."""

    E = _as_vector("electric_field_V_m", electric_field_V_m)
    B = _as_vector("magnetic_field_T", magnetic_field_T)
    if E.shape != B.shape:
        raise ValueError(f"field shapes differ: {E.shape} != {B.shape}")
    volume = _positive_float("cell_volume_m3", cell_volume_m3)
    electric = float(0.5 * EPSILON_0 * np.sum(E * E) * volume)
    magnetic = float(0.5 * np.sum(B * B) * volume / MU_0)
    return EnergySnapshot(
        total_J=electric + magnetic,
        electric_J=electric,
        magnetic_J=magnetic,
    )


def field_energy_from_diagnostics(diagnostics: Mapping[str, Any]) -> EnergySnapshot:
    """Build a field-energy snapshot from Maxwell diagnostics telemetry."""

    electric = _optional_float(diagnostics.get("electric_energy_J"))
    magnetic = _optional_float(diagnostics.get("magnetic_energy_J"))
    total = _optional_float(diagnostics.get("total_energy_J"))
    if total is None:
        total = _sum_known(electric, magnetic)
    if total is None:
        raise ValueError("field diagnostics must contain total_energy_J or both components")
    return EnergySnapshot(total_J=total, electric_J=electric, magnetic_J=magnetic)


def compute_circuit_energy(
    *,
    current_A: float,
    capacitance_F: float,
    inductance_H: float,
    voltage_V: float | None = None,
    charge_C: float | None = None,
) -> EnergySnapshot:
    """Compute lumped circuit energy from capacitor and inductor state."""

    current = _finite_float("current_A", current_A)
    capacitance = _positive_float("capacitance_F", capacitance_F)
    inductance = _positive_float("inductance_H", inductance_H)
    if voltage_V is None:
        if charge_C is None:
            raise ValueError("voltage_V or charge_C is required for capacitor energy")
        voltage = _finite_float("charge_C", charge_C) / capacitance
    else:
        voltage = _finite_float("voltage_V", voltage_V)
    capacitive = 0.5 * capacitance * voltage**2
    inductive = 0.5 * inductance * current**2
    return EnergySnapshot(
        total_J=float(capacitive + inductive),
        capacitive_J=float(capacitive),
        inductive_J=float(inductive),
    )


def compute_electron_energy(
    electron_energy_J_m3: np.ndarray,
    *,
    cell_volume_m3: float,
) -> EnergySnapshot:
    """Integrate an electron-energy density array to total electron energy."""

    density = np.asarray(electron_energy_J_m3, dtype=float)
    if density.size == 0:
        raise ValueError("electron_energy_J_m3 must not be empty")
    if not np.all(np.isfinite(density)):
        raise ValueError("electron_energy_J_m3 must be finite")
    volume = _positive_float("cell_volume_m3", cell_volume_m3)
    return EnergySnapshot(total_J=float(np.sum(density) * volume))


def count_macroparticles(value: Any) -> int:
    """Count active macroparticles from an integer, PIC object, or species list."""

    if isinstance(value, (int, np.integer)):
        if int(value) < 0:
            raise ValueError("particle count must be non-negative")
        return int(value)
    if hasattr(value, "species"):
        return count_macroparticles(value.species)
    if hasattr(value, "n_particles") and callable(value.n_particles):
        count = int(value.n_particles())
        if count < 0:
            raise ValueError("particle count must be non-negative")
        return count
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return sum(count_macroparticles(item) for item in value)
    raise TypeError("cannot count macroparticles from value")


def build_field_energy_ledger(
    initial: EnergySnapshot | Mapping[str, Any] | None,
    final: EnergySnapshot | Mapping[str, Any] | None,
) -> FieldEnergyLedger:
    """Build a field-energy ledger from optional initial/final snapshots."""

    initial_snapshot = _coerce_energy_snapshot(initial)
    final_snapshot = _coerce_energy_snapshot(final)
    return FieldEnergyLedger(
        status=_ledger_status(initial_snapshot, final_snapshot),
        total=_energy_delta(initial_snapshot, final_snapshot, "total_J"),
        electric=_energy_delta(initial_snapshot, final_snapshot, "electric_J"),
        magnetic=_energy_delta(initial_snapshot, final_snapshot, "magnetic_J"),
    )


def build_circuit_energy_ledger(
    initial: EnergySnapshot | Mapping[str, Any] | None,
    final: EnergySnapshot | Mapping[str, Any] | None,
) -> CircuitEnergyLedger:
    """Build a circuit-energy ledger from optional initial/final snapshots."""

    initial_snapshot = _coerce_energy_snapshot(initial)
    final_snapshot = _coerce_energy_snapshot(final)
    return CircuitEnergyLedger(
        status=_ledger_status(initial_snapshot, final_snapshot),
        total=_energy_delta(initial_snapshot, final_snapshot, "total_J"),
        capacitive=_energy_delta(initial_snapshot, final_snapshot, "capacitive_J"),
        inductive=_energy_delta(initial_snapshot, final_snapshot, "inductive_J"),
    )


def build_particle_count_ledger(
    initial: Any | None,
    final: Any | None,
) -> ParticleCountLedger:
    """Build macroparticle count accounting from counts, PICs, or species."""

    initial_count = None if initial is None else count_macroparticles(initial)
    final_count = None if final is None else count_macroparticles(final)
    delta = (
        None
        if initial_count is None or final_count is None
        else final_count - initial_count
    )
    relative = None
    if delta is not None and initial_count not in (None, 0):
        relative = float(delta / initial_count)
    return ParticleCountLedger(
        status=_count_status(initial_count, final_count),
        initial_count=initial_count,
        final_count=final_count,
        delta_count=delta,
        relative_delta=relative,
    )


def build_residual_ledger(
    *,
    gauss_law_linf: float | None = None,
    div_B_linf: float | None = None,
    current_continuity_linf_A_m3: float | None = None,
    current_residual_linf_A_m2: float | None = None,
    missing_channels: Sequence[str] | None = None,
    notes: str = "",
) -> ResidualLedger:
    """Build measured residual ledger fields without synthetic substitutes."""

    values = {
        "gauss_law_linf": gauss_law_linf,
        "div_B_linf": div_B_linf,
        "current_continuity_linf_A_m3": current_continuity_linf_A_m3,
        "current_residual_linf_A_m2": current_residual_linf_A_m2,
    }
    clean = {name: _optional_float(value) for name, value in values.items()}
    missing = tuple(
        str(channel)
        for channel in (
            missing_channels
            if missing_channels is not None
            else tuple(name for name, value in clean.items() if value is None)
        )
    )
    status = (
        "measured_residuals_not_validation"
        if not missing
        else "partial_measured_residuals_not_validation"
        if any(value is not None for value in clean.values())
        else "measured_residual_channels_missing_not_validation"
    )
    return ResidualLedger(status=status, missing_channels=missing, notes=notes, **clean)


def residual_ledger_from_hybrid_telemetry(telemetry: Mapping[str, Any]) -> ResidualLedger:
    """Extract measured residual channels from 3-D hybrid loop or run telemetry."""

    last_step = _mapping_or_none(telemetry.get("last_step")) or telemetry
    field_step = _mapping_or_none(last_step.get("field_step")) or last_step
    diagnostics = _mapping_or_none(field_step.get("diagnostics_after")) or {}
    marder = _mapping_or_none(field_step.get("marder")) or {}
    current_port = _mapping_or_none(field_step.get("current_port")) or {}
    ohm = _mapping_or_none(field_step.get("ohm_solver")) or {}
    predictor = _mapping_or_none(field_step.get("predictor_corrector")) or {}

    return build_residual_ledger(
        gauss_law_linf=marder.get("residual_after_linf"),
        div_B_linf=diagnostics.get("max_abs_div_B_T_per_m"),
        current_continuity_linf_A_m3=current_port.get("continuity_linf_A_per_m3"),
        current_residual_linf_A_m2=(
            predictor.get("corrected_max_residual_A_m2")
            if predictor.get("corrected_max_residual_A_m2") is not None
            else ohm.get("max_algebraic_residual_A_m2")
        ),
    )


def build_conservation_ledger(
    *,
    field_energy_initial: EnergySnapshot | Mapping[str, Any] | None = None,
    field_energy_final: EnergySnapshot | Mapping[str, Any] | None = None,
    circuit_energy_initial: EnergySnapshot | Mapping[str, Any] | None = None,
    circuit_energy_final: EnergySnapshot | Mapping[str, Any] | None = None,
    particle_count_initial: Any | None = None,
    particle_count_final: Any | None = None,
    electron_energy_initial: EnergySnapshot | Mapping[str, Any] | None = None,
    electron_energy_final: EnergySnapshot | Mapping[str, Any] | None = None,
    residuals: ResidualLedger | Mapping[str, Any] | None = None,
    source_index_references: Sequence[SourceIndexReference | Mapping[str, Any] | str] | None = None,
    notes: str = "",
) -> FirstPrinciplesConservationLedger:
    """Build the fail-closed first-principles conservation ledger."""

    electron_initial = _coerce_energy_snapshot(electron_energy_initial)
    electron_final = _coerce_energy_snapshot(electron_energy_final)
    if residuals is None:
        residual_ledger = build_residual_ledger()
    elif isinstance(residuals, ResidualLedger):
        residual_ledger = residuals
    elif isinstance(residuals, Mapping):
        residual_ledger = _coerce_residual_ledger(residuals)
    else:
        raise TypeError("residuals must be a ResidualLedger, mapping, or None")

    return FirstPrinciplesConservationLedger(
        field_energy=build_field_energy_ledger(field_energy_initial, field_energy_final),
        circuit_energy=build_circuit_energy_ledger(circuit_energy_initial, circuit_energy_final),
        particle_count=build_particle_count_ledger(
            particle_count_initial,
            particle_count_final,
        ),
        electron_energy=_energy_delta(electron_initial, electron_final, "total_J"),
        residuals=residual_ledger,
        source_index_references=normalize_source_index_references(source_index_references),
        notes=notes,
    )


def build_conservation_ledger_from_hybrid_telemetry(
    telemetry: Mapping[str, Any],
    *,
    source_index_references: Sequence[SourceIndexReference | Mapping[str, Any] | str] | None = None,
    notes: str = "",
) -> FirstPrinciplesConservationLedger:
    """Build a conservation ledger from HybridPIC3DSimulationTelemetry-like data."""

    initial_field = _field_snapshot_from_total(telemetry.get("initial_field_energy_J"))
    final_field = _field_snapshot_from_total(telemetry.get("final_field_energy_J"))
    return build_conservation_ledger(
        field_energy_initial=initial_field,
        field_energy_final=final_field,
        particle_count_initial=telemetry.get("n_particles_initial"),
        particle_count_final=telemetry.get("n_particles_final"),
        residuals=residual_ledger_from_hybrid_telemetry(telemetry),
        source_index_references=source_index_references,
        notes=notes,
    )


def normalize_source_index_references(
    references: Sequence[SourceIndexReference | Mapping[str, Any] | str] | None,
) -> tuple[SourceIndexReference, ...]:
    """Normalize source-index references for deterministic manifest output."""

    if not references:
        return ()
    normalized: list[SourceIndexReference] = []
    for ref in references:
        if isinstance(ref, SourceIndexReference):
            normalized.append(ref)
        elif isinstance(ref, str):
            normalized.append(SourceIndexReference(source_id=ref, path=ref))
        elif isinstance(ref, Mapping):
            source_id = ref.get("source_id") or ref.get("id") or ref.get("source")
            path = ref.get("path") or ref.get("source_path") or source_id
            if source_id is None or path is None:
                raise ValueError("source references require source_id/id and path")
            normalized.append(
                SourceIndexReference(
                    source_id=str(source_id),
                    path=str(path),
                    scope=str(ref.get("scope", "first_principles_3d_hybrid_pic_fluid")),
                    status=str(ref.get("status", "source_reference_not_validation")),
                    lines=None if ref.get("lines") is None else str(ref["lines"]),
                    digest=None if ref.get("digest") is None else str(ref["digest"]),
                    notes=str(ref.get("notes", "")),
                )
            )
        else:
            raise TypeError("source references must be SourceIndexReference, mapping, or str")
    return tuple(normalized)


def _field_snapshot_from_total(value: Any) -> EnergySnapshot | None:
    total = _optional_float(value)
    if total is None:
        return None
    return EnergySnapshot(total_J=total)


def _coerce_energy_snapshot(
    value: EnergySnapshot | Mapping[str, Any] | None,
) -> EnergySnapshot | None:
    if value is None:
        return None
    if isinstance(value, EnergySnapshot):
        return value
    if isinstance(value, Mapping):
        if "total_J" in value:
            total = value.get("total_J")
        elif "total_energy_J" in value:
            total = value.get("total_energy_J")
        else:
            total = _sum_known(
                _optional_float(value.get("electric_J") or value.get("electric_energy_J")),
                _optional_float(value.get("magnetic_J") or value.get("magnetic_energy_J")),
            )
        if total is None:
            raise ValueError("energy snapshot mapping requires total_J or components")
        return EnergySnapshot(
            total_J=_finite_float("total_J", total),
            electric_J=_optional_float(value.get("electric_J") or value.get("electric_energy_J")),
            magnetic_J=_optional_float(value.get("magnetic_J") or value.get("magnetic_energy_J")),
            capacitive_J=_optional_float(value.get("capacitive_J")),
            inductive_J=_optional_float(value.get("inductive_J")),
        )
    raise TypeError("energy snapshot must be EnergySnapshot, mapping, or None")


def _coerce_residual_ledger(value: Mapping[str, Any]) -> ResidualLedger:
    """Normalize serialized residual channels without trusting stale status."""

    allowed = {
        "gauss_law_linf",
        "div_B_linf",
        "current_continuity_linf_A_m3",
        "current_residual_linf_A_m2",
        "missing_channels",
        "notes",
    }
    payload = {key: item for key, item in value.items() if key in allowed}
    return build_residual_ledger(**payload)


def _energy_delta(
    initial: EnergySnapshot | None,
    final: EnergySnapshot | None,
    attr: str,
) -> EnergyDeltaLedger:
    initial_value = None if initial is None else getattr(initial, attr)
    final_value = None if final is None else getattr(final, attr)
    if initial_value is not None:
        _require_finite(f"initial_{attr}", initial_value)
    if final_value is not None:
        _require_finite(f"final_{attr}", final_value)
    delta = (
        None
        if initial_value is None or final_value is None
        else float(final_value - initial_value)
    )
    relative = None
    if delta is not None and initial_value not in (None, 0.0):
        relative = float(delta / initial_value)
    return EnergyDeltaLedger(
        status=_ledger_status(initial, final),
        initial_J=initial_value,
        final_J=final_value,
        delta_J=delta,
        relative_delta=relative,
    )


def _ledger_status(
    initial: EnergySnapshot | None,
    final: EnergySnapshot | None,
) -> str:
    if initial is None and final is None:
        return "not_available_not_validation"
    if initial is None or final is None:
        return "partial_accounting_not_validation"
    return "measured_accounting_not_validation"


def _count_status(initial: int | None, final: int | None) -> str:
    if initial is None and final is None:
        return "not_available_not_validation"
    if initial is None or final is None:
        return "partial_accounting_not_validation"
    if initial == final:
        return "conserved_not_validation"
    return "changed_not_validation"


def _as_vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have trailing vector dimension 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _sum_known(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _finite_float(name: str, value: Any) -> float:
    parsed = float(value)
    _require_finite(name, parsed)
    return parsed


def _positive_float(name: str, value: Any) -> float:
    parsed = _finite_float(name, value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float("value", value)


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        mapped = value.to_dict()
        if isinstance(mapped, Mapping):
            return mapped
    return None
