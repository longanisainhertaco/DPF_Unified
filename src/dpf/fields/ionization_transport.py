"""Source-backed candidate ionization transport for the 3-D DPF runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.atomic.ionization import radiative_recombination_rate
from dpf.constants import eV, k_B
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.maxwell_3d import Maxwell3DGrid

STEP_GROUND_STATE_IONIZATION_SOURCE = (
    "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md:252-259"
)
NRL_CHARGE_STATE_SOURCE = (
    "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4572-4648"
)
DEUTERIUM_IONIZATION_ENERGY_EV = 13.6


@dataclass
class DeuteriumIonizationState:
    """Single-stage D/D+ chemistry state on the field grid."""

    neutral_density_m3: np.ndarray
    ion_density_m3: np.ndarray
    electron_density_m3: np.ndarray
    mean_charge_state: np.ndarray


@dataclass(frozen=True)
class IonizationTransportTelemetry:
    """Telemetry for one candidate ionization/recombination transport step."""

    status: str
    source: str
    rate_source: str
    model: str
    ionization_energy_eV: float
    min_electron_temperature_eV: float
    max_electron_temperature_eV: float
    min_ionization_rate_m3_s: float
    max_ionization_rate_m3_s: float
    min_radiative_recombination_rate_m3_s: float
    max_radiative_recombination_rate_m3_s: float
    min_three_body_recombination_rate_m6_s: float
    max_three_body_recombination_rate_m6_s: float
    max_net_source_m3_s: float
    max_limited_density_change_m3: float
    min_ionization_fraction: float
    max_ionization_fraction: float
    validity_notes: tuple[str, ...]
    limitations: tuple[str, ...]
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IonizationParticleSourceTelemetry:
    """Telemetry for candidate neutral-ion chemistry coupling to PIC ions."""

    status: str
    source: str
    macro_particles_created: int
    macro_particles_removed: int
    physical_ions_created: float
    physical_ions_removed: float
    unrepresented_recombination_ions: float
    coupling_stage: str
    limitations: tuple[str, ...]
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeuteriumIonizationTransport:
    """Candidate D/D+ ionization kinetics using local source-truth equations."""

    capability_id = "candidate_deuterium_charge_state_transport"

    def __init__(
        self,
        grid: Maxwell3DGrid,
        *,
        ionization_energy_eV: float = DEUTERIUM_IONIZATION_ENERGY_EV,
    ) -> None:
        self.grid = grid
        self.ionization_energy_eV = float(ionization_energy_eV)
        if self.ionization_energy_eV <= 0.0:
            raise ValueError("ionization_energy_eV must be positive")

    def initialize(
        self,
        *,
        total_deuterium_density_m3: np.ndarray | float,
        ionization_fraction: np.ndarray | float,
    ) -> DeuteriumIonizationState:
        """Initialize neutral, D+, electron, and mean charge-state fields."""
        total = _grid_array(total_deuterium_density_m3, self.grid.shape)
        fraction = _grid_array(ionization_fraction, self.grid.shape)
        if np.any((fraction < 0.0) | (fraction > 1.0)):
            raise ValueError("ionization_fraction must be in [0, 1]")
        if np.any(total < 0.0):
            raise ValueError("total_deuterium_density_m3 must be non-negative")
        ion = total * fraction
        neutral = np.maximum(total - ion, 0.0)
        electron = ion.copy()
        mean_z = _mean_charge_state(neutral, ion, electron)
        return DeuteriumIonizationState(
            neutral_density_m3=neutral,
            ion_density_m3=ion,
            electron_density_m3=electron,
            mean_charge_state=mean_z,
        )

    def step(
        self,
        state: DeuteriumIonizationState,
        *,
        electron_temperature_K: np.ndarray | float,
        dt_s: float,
    ) -> tuple[DeuteriumIonizationState, IonizationTransportTelemetry]:
        """Advance single-stage D/D+ ionization and recombination by one step."""
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        neutral = _grid_array(state.neutral_density_m3, self.grid.shape)
        ion = _grid_array(state.ion_density_m3, self.grid.shape)
        electron = _grid_array(state.electron_density_m3, self.grid.shape)
        if np.any(neutral < 0.0) or np.any(ion < 0.0) or np.any(electron < 0.0):
            raise ValueError("ionization state densities must be non-negative")

        Te_K = _grid_array(electron_temperature_K, self.grid.shape)
        Te_eV = np.maximum(Te_K * k_B / eV, 1.0e-6)
        ionization_rate = nrl_ground_state_ionization_rate(
            Te_eV,
            self.ionization_energy_eV,
        )
        radiative_rate = _radiative_rate_array(Te_eV)
        three_body_rate = nrl_three_body_recombination_rate(Te_eV)

        ionization_source = electron * neutral * ionization_rate
        radiative_recombination = electron * ion * radiative_rate
        three_body_recombination = electron * electron * ion * three_body_rate
        net_source = (
            ionization_source - radiative_recombination - three_body_recombination
        )
        requested_delta = float(dt_s) * net_source
        limited_delta = np.clip(requested_delta, -ion, neutral)

        ion_next = np.maximum(ion + limited_delta, 0.0)
        neutral_next = np.maximum(neutral - limited_delta, 0.0)
        electron_next = ion_next.copy()
        mean_z_next = _mean_charge_state(neutral_next, ion_next, electron_next)
        next_state = DeuteriumIonizationState(
            neutral_density_m3=neutral_next,
            ion_density_m3=ion_next,
            electron_density_m3=electron_next,
            mean_charge_state=mean_z_next,
        )
        telemetry = IonizationTransportTelemetry(
            status=self.capability_id,
            source=STEP_GROUND_STATE_IONIZATION_SOURCE,
            rate_source=NRL_CHARGE_STATE_SOURCE,
            model=(
                "single_stage_deuterium_ground_state_ionization_radiative_"
                "three_body_recombination"
            ),
            ionization_energy_eV=self.ionization_energy_eV,
            min_electron_temperature_eV=float(np.min(Te_eV)),
            max_electron_temperature_eV=float(np.max(Te_eV)),
            min_ionization_rate_m3_s=float(np.min(ionization_rate)),
            max_ionization_rate_m3_s=float(np.max(ionization_rate)),
            min_radiative_recombination_rate_m3_s=float(np.min(radiative_rate)),
            max_radiative_recombination_rate_m3_s=float(np.max(radiative_rate)),
            min_three_body_recombination_rate_m6_s=float(np.min(three_body_rate)),
            max_three_body_recombination_rate_m6_s=float(np.max(three_body_rate)),
            max_net_source_m3_s=float(np.max(np.abs(net_source))),
            max_limited_density_change_m3=float(np.max(np.abs(limited_delta))),
            min_ionization_fraction=float(np.min(mean_z_next)),
            max_ionization_fraction=float(np.max(mean_z_next)),
            validity_notes=(
                "NRL ionization formula is used for ground-state ionization only.",
                "Three-body recombination uses the NRL singly ionized plasma form.",
                "Hydrogenic radiative recombination is applied to D+ as Z=1.",
            ),
            limitations=(
                "Candidate runtime closure only; no engineer review packet yet.",
                "Single-stage deuterium chemistry only; no molecular D2, impurities, or excited states.",
                "Neutral depletion changes the chemistry state but does not yet spawn/remove PIC macroparticles.",
                "Conductivity and EOS are not yet rebuilt from the evolving charge-state field.",
            ),
        )
        return next_state, telemetry


def nrl_ground_state_ionization_rate(
    electron_temperature_eV: np.ndarray | float,
    ionization_energy_eV: float,
) -> np.ndarray:
    """NRL ground-state electron-impact ionization rate coefficient [m^3/s]."""
    if ionization_energy_eV <= 0.0:
        raise ValueError("ionization_energy_eV must be positive")
    Te = np.asarray(electron_temperature_eV, dtype=float)
    Te_safe = np.maximum(Te, 1.0e-12)
    E = float(ionization_energy_eV)
    ratio = Te_safe / E
    rate_cgs = (
        1.0e-5
        * np.sqrt(ratio)
        / ((E ** 1.5) * (6.0 + ratio))
        * np.exp(-E / Te_safe)
    )
    return np.maximum(rate_cgs * 1.0e-6, 0.0)


def nrl_three_body_recombination_rate(
    electron_temperature_eV: np.ndarray | float,
) -> np.ndarray:
    """NRL three-body recombination rate for singly ionized plasma [m^6/s]."""
    Te = np.asarray(electron_temperature_eV, dtype=float)
    Te_safe = np.maximum(Te, 1.0e-12)
    return 8.75e-39 * Te_safe ** -4.5


def ionization_transport_candidate_evidence(
    telemetry: IonizationTransportTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for candidate charge-state transport."""
    return {
        "passed": telemetry.status
        == DeuteriumIonizationTransport.capability_id,
        "status": "candidate",
        "capability": DeuteriumIonizationTransport.capability_id,
        "source": telemetry.source,
        "rate_source": telemetry.rate_source,
        "implementation": "src/dpf/fields/ionization_transport.py",
        "evidence_type": "engineering_charge_state_transport_step",
        "model": telemetry.model,
        "ionization_fraction_range": [
            telemetry.min_ionization_fraction,
            telemetry.max_ionization_fraction,
        ],
        "can_support_first_principles_acceptance": False,
        "limitations": list(telemetry.limitations),
    }


def apply_ionization_particle_source(
    pic: HybridPIC,
    grid: Maxwell3DGrid,
    *,
    previous_state: DeuteriumIonizationState,
    next_state: DeuteriumIonizationState,
    species_name: str = "d",
    ion_mass_kg: float | None = None,
    ion_charge_C: float | None = None,
    velocity_m_s: np.ndarray | None = None,
) -> IonizationParticleSourceTelemetry:
    """Couple D/D+ chemistry changes into candidate PIC macroparticle weights."""
    delta_density = next_state.ion_density_m3 - previous_state.ion_density_m3
    cell_volume = float(grid.dx * grid.dy * grid.dz)
    delta_physical = delta_density * cell_volume
    positive_delta = np.maximum(delta_physical, 0.0)
    negative_delta = np.maximum(-delta_physical, 0.0)

    species = _find_or_create_species(
        pic,
        species_name=species_name,
        ion_mass_kg=ion_mass_kg,
        ion_charge_C=ion_charge_C,
    )
    new_positions: list[list[float]] = []
    new_velocities: list[list[float]] = []
    new_weights: list[float] = []
    velocity_field = None if velocity_m_s is None else _grid_vector(velocity_m_s, grid.shape)
    for index, physical_count in np.ndenumerate(positive_delta):
        if physical_count <= 0.0:
            continue
        i, j, k = index
        new_positions.append([
            (i + 0.5) * grid.dx,
            (j + 0.5) * grid.dy,
            (k + 0.5) * grid.dz,
        ])
        if velocity_field is None:
            new_velocities.append([0.0, 0.0, 0.0])
        else:
            new_velocities.append([
                float(velocity_field[index + (0,)]),
                float(velocity_field[index + (1,)]),
                float(velocity_field[index + (2,)]),
            ])
        new_weights.append(float(physical_count))

    macro_removed, physical_removed, unrepresented = _remove_recombined_weight(
        species,
        grid,
        negative_delta,
    )
    if new_weights:
        positions = np.asarray(new_positions, dtype=np.float64)
        velocities = np.asarray(new_velocities, dtype=np.float64)
        weights = np.asarray(new_weights, dtype=np.float64)
        species.positions = np.concatenate([species.positions, positions], axis=0)
        species.velocities = np.concatenate([species.velocities, velocities], axis=0)
        species.weights = np.concatenate([species.weights, weights], axis=0)
        species.positions_old = species.positions.copy()
    return IonizationParticleSourceTelemetry(
        status="candidate_ionization_pic_particle_source",
        source=STEP_GROUND_STATE_IONIZATION_SOURCE,
        macro_particles_created=len(new_weights),
        macro_particles_removed=macro_removed,
        physical_ions_created=float(np.sum(positive_delta)),
        physical_ions_removed=float(physical_removed),
        unrepresented_recombination_ions=float(unrepresented),
        coupling_stage="post_field_step_available_to_next_deposit",
        limitations=(
            "Numerical macroparticle source/sink only; no accepted startup breakdown packet.",
            "New ions are born at cell centers with the supplied plasma velocity or zero velocity.",
            "Recombination removes available same-cell ion weight and reports any unrepresented deficit.",
            "Electron density and conductivity feedback remain candidate loop plumbing.",
        ),
    )


def _grid_array(value: np.ndarray | float, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=float)
    if array.shape != shape:
        raise ValueError(f"expected grid-shaped array {shape}, got {array.shape}")
    return np.array(array, copy=True, dtype=float)


def _grid_vector(value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape + (3,):
        raise ValueError(f"expected grid vector array {shape + (3,)}, got {array.shape}")
    return array


def _mean_charge_state(
    neutral_density_m3: np.ndarray,
    ion_density_m3: np.ndarray,
    electron_density_m3: np.ndarray,
) -> np.ndarray:
    total_heavy = neutral_density_m3 + ion_density_m3
    return np.divide(
        electron_density_m3,
        total_heavy,
        out=np.zeros_like(electron_density_m3),
        where=total_heavy > 0.0,
    )


def _radiative_rate_array(Te_eV: np.ndarray) -> np.ndarray:
    values = np.empty_like(Te_eV, dtype=float)
    for index, value in np.ndenumerate(Te_eV):
        values[index] = radiative_recombination_rate(float(value), 1)
    return values


def _find_or_create_species(
    pic: HybridPIC,
    *,
    species_name: str,
    ion_mass_kg: float | None,
    ion_charge_C: float | None,
):
    for species in pic.species:
        if species.name == species_name:
            return species
    if ion_mass_kg is None or ion_charge_C is None:
        raise ValueError(
            "ion_mass_kg and ion_charge_C are required when the PIC species is absent"
        )
    return pic.add_species(
        species_name,
        ion_mass_kg,
        ion_charge_C,
        positions=np.empty((0, 3), dtype=np.float64),
        velocities=np.empty((0, 3), dtype=np.float64),
        weights=np.empty((0,), dtype=np.float64),
    )


def _remove_recombined_weight(
    species,
    grid: Maxwell3DGrid,
    negative_delta_by_cell: np.ndarray,
) -> tuple[int, float, float]:
    if species.n_particles() == 0 or not np.any(negative_delta_by_cell > 0.0):
        return 0, 0.0, float(np.sum(negative_delta_by_cell))
    requested = np.array(negative_delta_by_cell, copy=True, dtype=float)
    removed_particles = 0
    removed_weight = 0.0
    nx, ny, nz = grid.shape
    for p_index, position in enumerate(species.positions):
        i = min(max(int(position[0] / grid.dx), 0), nx - 1)
        j = min(max(int(position[1] / grid.dy), 0), ny - 1)
        k = min(max(int(position[2] / grid.dz), 0), nz - 1)
        need = requested[i, j, k]
        if need <= 0.0:
            continue
        available = species.weights[p_index]
        removed = min(float(available), float(need))
        species.weights[p_index] = available - removed
        requested[i, j, k] -= removed
        removed_weight += removed
    keep = species.weights > 0.0
    if not np.all(keep):
        removed_particles = int(np.count_nonzero(~keep))
        species.positions = species.positions[keep]
        species.velocities = species.velocities[keep]
        species.weights = species.weights[keep]
        species.positions_old = species.positions_old[keep]
    return removed_particles, removed_weight, float(np.sum(requested))
