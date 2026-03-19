
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dpf.config import SimulationConfig
from dpf.constants import e as e_charge
from dpf.constants import k_B
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
        dz = (getattr(config.geometry, "dz", None) if hasattr(config, "geometry") else None) or config.dx
        self.driver = HybridPIC(
            grid_shape=(nx, ny, nz),
            dx=config.dx,
            dy=config.dx,
            dz=dz,
            dt=1e-9,  # initial dt; overridden each step() call
        )

        self.beam_injected = False

        # MHD state cache for Coulomb collision background (updated each step)
        self._n_bg: float = 1e25       # background density [m^-3]
        self._T_bg_eV: float = 100.0   # background electron temperature [eV]

        # Beam species — initialized empty, populated on first inject
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

    def update_mhd_state(self, state: dict[str, np.ndarray]) -> None:
        """Update the background density and temperature from the current MHD state.

        Called by the engine each step so that Coulomb collisions use the local
        plasma conditions rather than hardcoded defaults.

        Args:
            state: Engine state dict containing at least ``rho`` and ``Te``.
        """
        rho = state.get("rho")
        Te = state.get("Te")
        if rho is None or Te is None:
            return

        # Peak density: beam ions scatter most strongly in the dense pinch region
        n_peak = float(np.max(rho)) / self.config.ion_mass
        # Peak electron temperature (convert K → eV if > 1 K, floor at 1 eV)
        Te_peak_K = float(np.max(Te))
        Te_peak_eV = max(Te_peak_K * k_B / e_charge, 1.0)

        self._n_bg = max(n_peak, 1e10)
        self._T_bg_eV = Te_peak_eV

        # Keep the driver in sync
        if self.driver._collision_enabled:
            self.driver.enable_collisions(self._n_bg, self._T_bg_eV)

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
            # Enable Coulomb collisions using current MHD background state
            self.driver.enable_collisions(self._n_bg, self._T_bg_eV)
            logger.info(
                "Coulomb collisions enabled: n_bg=%.2e m^-3, T_bg=%.1f eV",
                self._n_bg, self._T_bg_eV,
            )

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
            species_idx=0,  # deuterium_beam
            n_beam=self.kc.n_particles,
            energy_eV=self.kc.beam_energy,
            direction=direction,
            position=center,
            spread=0.1,  # 0.1 rad spread
            weight_total=self.kc.beam_weight_total,
        )
