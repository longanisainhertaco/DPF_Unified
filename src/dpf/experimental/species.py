"""Multi-species framework for Dense Plasma Focus simulations.

Tracks multiple ion species (e.g. deuterium fill gas, electrode ablation
products like copper or tungsten) with independent density fields,
composition-dependent Z_eff, and species continuity advection.

Each species is defined by a ``SpeciesConfig`` dataclass and collected
into a ``SpeciesMixture`` that manages the density arrays, computes
mixture-averaged quantities, and provides advection/source operators.

Predefined species constants are provided for common DPF gases and
electrode materials.

Units: SI throughout (kg, m, s, C, K).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dpf.constants import m_d, m_p

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMU: float = 1.66054e-27  # Atomic mass unit [kg]


# ---------------------------------------------------------------------------
# Species configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeciesConfig:
    """Configuration for a single ion species.

    Attributes:
        name: Human-readable species identifier (e.g. ``"deuterium"``).
        mass: Ion mass [kg].
        charge_number: Atomic number *Z* (number of protons; equals
            maximum ionization state for a fully-stripped ion).
        initial_fraction: Initial number-density fraction of this
            species relative to the total ion density.  Must be in
            [0, 1]; the sum over all species in a mixture must equal 1.
        label: Short label for diagnostics and HDF5 datasets
            (e.g. ``"D"``, ``"Cu"``).
    """

    name: str
    mass: float  # [kg]
    charge_number: int
    initial_fraction: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        if self.charge_number < 0:
            raise ValueError(f"charge_number must be non-negative, got {self.charge_number}")
        if not (0.0 <= self.initial_fraction <= 1.0):
            raise ValueError(
                f"initial_fraction must be in [0, 1], got {self.initial_fraction}"
            )
        if not self.label:
            self.label = self.name[:4].capitalize()


# ---------------------------------------------------------------------------
# Predefined species
# ---------------------------------------------------------------------------

DEUTERIUM = SpeciesConfig(
    name="deuterium",
    mass=m_d,           # 3.34358377e-27 kg
    charge_number=1,
    initial_fraction=1.0,
    label="D",
)

HYDROGEN = SpeciesConfig(
    name="hydrogen",
    mass=m_p,           # 1.67262192e-27 kg
    charge_number=1,
    initial_fraction=1.0,
    label="H",
)

COPPER = SpeciesConfig(
    name="copper",
    mass=63.546 * AMU,  # ~1.0552e-25 kg
    charge_number=29,
    initial_fraction=0.0,
    label="Cu",
)

TUNGSTEN = SpeciesConfig(
    name="tungsten",
    mass=183.84 * AMU,  # ~3.0535e-25 kg
    charge_number=74,
    initial_fraction=0.0,
    label="W",
)


# ---------------------------------------------------------------------------
# Species mixture
# ---------------------------------------------------------------------------

class SpeciesMixture:
    """Manages a collection of ion species with independent density fields.

    Provides composition-dependent plasma quantities (Z_eff, mean ion
    mass) and operators for species continuity advection and
    ablation/ionization source terms.

    Args:
        species: List of ``SpeciesConfig`` defining the mixture.  Initial
            fractions must sum to 1 (within floating-point tolerance).
        grid_shape: Shape of the spatial grid, e.g. ``(nx, ny, nz)``.
        rho_total: Total initial mass density [kg/m^3].  Each species
            density is initialised as ``rho_total * f_s * (m_s / m_bar)``
            where ``f_s`` is the number fraction and ``m_bar`` the
            mean ion mass.

    Attributes:
        species: Ordered list of ``SpeciesConfig``.
        rho_species: Dictionary mapping species name to its mass-density
            array [kg/m^3], shape ``grid_shape``.
    """

    def __init__(
        self,
        species: Sequence[SpeciesConfig],
        grid_shape: tuple[int, ...],
        rho_total: float,
    ) -> None:
        if len(species) == 0:
            raise ValueError("At least one species must be provided")

        self.species: list[SpeciesConfig] = list(species)
        self._name_to_idx: dict[str, int] = {
            sp.name: i for i, sp in enumerate(self.species)
        }
        self.grid_shape = grid_shape

        # Validate fractions sum to 1
        frac_sum = sum(sp.initial_fraction for sp in self.species)
        if abs(frac_sum - 1.0) > 1e-12:
            raise ValueError(
                f"Species initial_fraction values must sum to 1.0, got {frac_sum:.15g}"
            )

        # Compute mean ion mass from number fractions:
        #   m_bar = sum(f_s * m_s)
        m_bar = sum(sp.initial_fraction * sp.mass for sp in self.species)

        # Initialise per-species mass density arrays.
        # Total number density: n_total = rho_total / m_bar
        # Species number density: n_s = f_s * n_total
        # Species mass density:   rho_s = n_s * m_s = rho_total * f_s * m_s / m_bar
        self.rho_species: dict[str, np.ndarray] = {}
        for sp in self.species:
            rho_s = rho_total * sp.initial_fraction * sp.mass / m_bar
            self.rho_species[sp.name] = np.full(grid_shape, rho_s, dtype=np.float64)

    # ------------------------------------------------------------------
    # Mixture-averaged quantities
    # ------------------------------------------------------------------

    def total_density(self) -> np.ndarray:
        """Total mass density: rho = sum_s(rho_s).

        Returns:
            Array of total mass density [kg/m^3], shape ``grid_shape``.
        """
        rho = np.zeros(self.grid_shape, dtype=np.float64)
        for rho_s in self.rho_species.values():
            rho += rho_s
        return rho

    def number_densities(self) -> dict[str, np.ndarray]:
        """Per-species number density: n_s = rho_s / m_s.

        Returns:
            Dictionary mapping species name to number-density array [m^-3].
        """
        return {
            sp.name: self.rho_species[sp.name] / sp.mass
            for sp in self.species
        }

    def z_eff(self) -> np.ndarray:
        """Effective ion charge state for the mixture.

        Z_eff = sum_s(n_s * Z_s^2) / sum_s(n_s * Z_s)

        This is the charge-state weighting relevant for Bremsstrahlung
        power, Spitzer resistivity, and other collision-dominated
        transport.  For a single fully-ionized species, Z_eff = Z.

        Returns:
            Z_eff array, shape ``grid_shape``.  Floored at 1.0 to
            prevent unphysical values.
        """
        n_dict = self.number_densities()
        numerator = np.zeros(self.grid_shape, dtype=np.float64)
        denominator = np.zeros(self.grid_shape, dtype=np.float64)
        for sp in self.species:
            n_s = n_dict[sp.name]
            Z = sp.charge_number
            numerator += n_s * Z * Z
            denominator += n_s * Z
        return numerator / np.maximum(denominator, 1e-300)

    def mean_ion_mass(self) -> np.ndarray:
        """Mean ion mass weighted by number density.

        m_bar = sum_s(n_s * m_s) / sum_s(n_s) = rho_total / n_total

        Returns:
            Mean ion mass array [kg], shape ``grid_shape``.
        """
        rho = self.total_density()
        n_total = np.zeros(self.grid_shape, dtype=np.float64)
        for sp in self.species:
            n_total += self.rho_species[sp.name] / sp.mass
        return rho / np.maximum(n_total, 1e-300)

    def electron_density(self) -> np.ndarray:
        """Total electron density assuming full ionization.

        n_e = sum_s(Z_s * n_s)

        Returns:
            Electron number density [m^-3], shape ``grid_shape``.
        """
        n_dict = self.number_densities()
        ne = np.zeros(self.grid_shape, dtype=np.float64)
        for sp in self.species:
            ne += sp.charge_number * n_dict[sp.name]
        return ne

    # ------------------------------------------------------------------
    # Species continuity (advection)
    # ------------------------------------------------------------------

    def advect(
        self,
        species_name: str,
        velocity: np.ndarray,
        dt: float,
        dx: float,
    ) -> None:
        """Advect a species density field with the bulk plasma velocity.

        Solves the species continuity equation::

            d(rho_s)/dt = -div(rho_s * v)

        using first-order upwind finite differences.  Higher-order
        reconstruction (WENO5) will be added in a future phase.

        Args:
            species_name: Name of the species to advect.
            velocity: Bulk plasma velocity field, shape ``(3, *grid_shape)``
                for Cartesian [m/s].
            dt: Timestep [s].
            dx: Grid spacing [m].

        Raises:
            KeyError: If ``species_name`` is not in the mixture.
        """
        if species_name not in self.rho_species:
            raise KeyError(
                f"Unknown species '{species_name}'. "
                f"Available: {list(self.rho_species.keys())}"
            )

        rho_s = self.rho_species[species_name]
        ndim = len(self.grid_shape)

        # Upwind advection: F_{i+1/2} = v * rho_s (upwind selection)
        # d(rho_s)/dt = -sum_d dF_d/dx_d
        flux_div = np.zeros_like(rho_s)

        for d in range(min(ndim, 3)):
            v_d = velocity[d]
            # Forward difference for downwind flux
            rho_fwd = np.roll(rho_s, -1, axis=d)

            # Upwind flux: use rho_s where v > 0, rho_fwd where v < 0
            flux_plus = v_d * rho_s           # flux at i+1/2 (upwind, v > 0)
            flux_minus = v_d * rho_fwd        # flux at i+1/2 (upwind, v < 0)
            flux_half = np.where(v_d >= 0, flux_plus, flux_minus)

            # Divergence: (F_{i+1/2} - F_{i-1/2}) / dx
            flux_half_shifted = np.roll(flux_half, 1, axis=d)
            flux_div += (flux_half - flux_half_shifted) / dx

        rho_s_new = rho_s - dt * flux_div

        # Floor at zero (density cannot go negative)
        np.maximum(rho_s_new, 0.0, out=rho_s_new)

        self.rho_species[species_name] = rho_s_new

    def advect_all(
        self,
        velocity: np.ndarray,
        dt: float,
        dx: float,
    ) -> None:
        """Advect all species density fields with the bulk velocity.

        Convenience method that calls ``advect`` for every species.

        Args:
            velocity: Bulk plasma velocity field [m/s].
            dt: Timestep [s].
            dx: Grid spacing [m].
        """
        for sp in self.species:
            self.advect(sp.name, velocity, dt, dx)

    # ------------------------------------------------------------------
    # Source terms (ablation, ionization)
    # ------------------------------------------------------------------

    def add_source(
        self,
        species_name: str,
        source_rate: np.ndarray | float,
        dt: float,
    ) -> None:
        """Apply a mass-density source term to a species.

        Models ablation (electrode material injected into plasma) or
        ionization/recombination processes that transfer density between
        species.

        The update is::

            rho_s += source_rate * dt

        where ``source_rate`` is in [kg/m^3/s].  Positive values add
        material; negative values remove it (with a floor at zero).

        Args:
            species_name: Target species.
            source_rate: Volumetric mass source rate [kg/m^3/s].
                Can be a scalar (uniform) or an array matching
                ``grid_shape``.
            dt: Timestep [s].

        Raises:
            KeyError: If ``species_name`` is not in the mixture.
        """
        if species_name not in self.rho_species:
            raise KeyError(
                f"Unknown species '{species_name}'. "
                f"Available: {list(self.rho_species.keys())}"
            )

        self.rho_species[species_name] = np.maximum(
            self.rho_species[species_name] + source_rate * dt,
            0.0,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def species_names(self) -> list[str]:
        """Return ordered list of species names."""
        return [sp.name for sp in self.species]

    def get_config(self, name: str) -> SpeciesConfig:
        """Look up a species config by name.

        Args:
            name: Species name.

        Returns:
            Corresponding ``SpeciesConfig``.

        Raises:
            KeyError: If name is not found.
        """
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(
                f"Unknown species '{name}'. "
                f"Available: {list(self._name_to_idx.keys())}"
            )
        return self.species[idx]

    def __repr__(self) -> str:
        parts = []
        for sp in self.species:
            parts.append(
                f"  {sp.label}({sp.name}): Z={sp.charge_number}, "
                f"m={sp.mass:.4e} kg, f={sp.initial_fraction:.3f}"
            )
        body = "\n".join(parts)
        return f"SpeciesMixture(\n{body}\n)"
