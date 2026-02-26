
from __future__ import annotations

import logging

import numpy as np

from dpf.config import SimulationConfig
from dpf.constants import e as e_charge
from dpf.kinetic.hybrid import HybridPIC

logger = logging.getLogger(__name__)

class KineticManager:
    """Manages the Kinetic (Hybrid-PIC) subsystem.
    
    Wraps the ``HybridPIC`` driver and handles:
    1. Initialization from config.
    2. Beam injection logic.
    3. Time integration (push).
    4. Coupling (current deposition).
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.kc = config.kinetic

        # Initialize HybridPIC driver
        nx, ny, nz = config.grid_shape
        self.driver = HybridPIC(
            grid_shape=(nx, ny, nz),
            dx=config.dx,
            dy=config.dx, # Cartesian assumption for now
            dz=config.dx, # Cartesian assumption
            dt=1e-9,  # placeholder, will be overridden in step
        )

        self.beam_injected = False

        # Define simulation species (e.g. Deuterium)
        # For now, we only create the "Beam" species if requested,
        # or a background thermal species.
        # Let's create a placeholder "ions" species
        # In a real run, this might be initialized from the fluid state,
        # but for Phase 3.1 we focus on the BEAM.
        self.ion_species = self.driver.add_species(
            name="deuterium_beam",
            mass=config.ion_mass,
            charge=e_charge,
            positions=np.zeros((0, 3)),
            velocities=np.zeros((0, 3)),
            weights=np.zeros((0,)),
        )

        logger.info(
            "KineticManager initialized: enabled=%s, beam=%s, E=%.1f keV",
            self.kc.enabled, self.kc.inject_beam, self.kc.beam_energy / 1e3
        )

    def step(self, dt: float, time: float, E_field: np.ndarray, B_field: np.ndarray) -> dict[str, Any]:
        """Advance kinetic particles by one step.

        Args:
            dt: Timestep [s].
            time: Current simulation time [s].
            E_field: Electric field (nx, ny, nz, 3) [V/m].
            B_field: Magnetic field (nx, ny, nz, 3) [T].
            
        Returns:
            Dictionary of kinetic methods/stats (e.g. max_energy).
        """
        if not self.kc.enabled:
            return {}

        if time < self.kc.start_time:
            return {"status": "waiting"}

        # Beam Injection Trigger
        if self.kc.inject_beam and not self.beam_injected:
            self._inject_beam()
            self.beam_injected = True

        # Push Particles
        # Note: HybridPIC.push_particles expects (nx,ny,nz,3) fields
        # If simulation is 2D (cylindrical), we might need to conform dimensions.
        # engine.py keeps 3D arrays even for cylindrical (ny=1), so it should matches.

        self.driver.push_particles(E_field, B_field, dt=dt)

        # Diagnostics
        n_part = self.ion_species.n_particles()
        return {
            "n_particles": n_part,
            "beam_injected": self.beam_injected
        }

    def get_current_density(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the kinetic current density J_kin on the grid."""
        _, Jx, Jy, Jz = self.driver.deposit()
        return Jx, Jy, Jz

    def _inject_beam(self) -> None:
        """Inject the high-energy ion beam."""
        logger.info("Injecting kinetic ion beam at t=%.2e", self.kc.start_time)

        # Center of anode (approx) from config ratio
        center = np.array([
            self.config.dx * self.config.grid_shape[0] * self.kc.beam_position_ratio[0],
            self.config.dx * self.config.grid_shape[1] * self.kc.beam_position_ratio[1],
            self.config.dx * self.config.grid_shape[2] * self.kc.beam_position_ratio[2]
        ])

        # Direction from config
        direction = np.array(self.kc.beam_direction)
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction /= norm

        self.driver.inject_beam(
            species_idx=0, # deuterium_beam
            n_beam=self.kc.n_particles,
            energy_eV=self.kc.beam_energy,
            direction=direction,
            position=center,
            spread=0.1, # 0.1 rad spread
        )
