"""Beam-ion tracker for DPF neutron yield computation.

Lightweight PIC wrapper that tracks beam ions in the MHD fields
without requiring SimulationConfig. Used by the web UI to compute
beam-target neutron yield from kinetic ion trajectories.

Physics:
    After pinch disruption (m=0 instability), fast ions are
    accelerated by the inductive electric field E = -dL/dt * I.
    These beam ions interact with the background deuterium target
    to produce D-D neutron reactions (beam-target mechanism).

    The beam energy is: E_beam ~ V_pinch * Z * e
    where V_pinch = (dL/dt) * I is the pinch voltage.

References:
    Lee & Saw, J. Fusion Energy 27:292 (2008) — beam-target model
    Haines et al., Phys. Rev. Lett. 106:045002 (2011) — DPF neutron mechanisms
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeamTrackerResult:
    """Result from beam ion tracking."""

    n_particles: int
    mean_energy_keV: float
    max_energy_keV: float
    Y_bt_kinetic: float  # Beam-target yield from kinetic trajectories
    energy_spectrum: np.ndarray  # Energy histogram [keV]
    energy_bins: np.ndarray  # Bin edges [keV]
    trajectory_count: int


class BeamTracker:
    """Track beam ions through MHD fields using Boris push.

    Lightweight alternative to KineticManager — no SimulationConfig needed.

    Args:
        n_particles: Number of test particles.
        ion_mass: Ion mass [kg].
        ion_charge: Ion charge [C].
        grid_shape: (nx, ny, nz).
        dx: Grid spacing [m].
    """

    def __init__(
        self,
        n_particles: int = 1000,
        ion_mass: float = 3.34e-27,
        ion_charge: float = 1.602e-19,
        grid_shape: tuple[int, int, int] = (16, 16, 32),
        dx: float = 0.001,
    ) -> None:
        self.n_particles = n_particles
        self.ion_mass = ion_mass
        self.ion_charge = ion_charge
        self.grid_shape = grid_shape
        self.dx = dx

        # Particle arrays
        self.positions = np.zeros((n_particles, 3))
        self.velocities = np.zeros((n_particles, 3))
        self.alive = np.ones(n_particles, dtype=bool)
        self._initialized = False

    def inject_beam(
        self,
        center: np.ndarray,
        direction: np.ndarray,
        energy_eV: float,
        spread_rad: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Inject beam particles at given position and direction.

        Args:
            center: Injection point [m], shape (3,).
            direction: Mean beam direction, shape (3,).
            energy_eV: Beam energy [eV].
            spread_rad: Angular spread [rad].
            rng: Random generator (for reproducibility).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Compute beam speed from energy
        v_beam = np.sqrt(2.0 * energy_eV * 1.602e-19 / self.ion_mass)

        # Normalize direction
        d = np.array(direction, dtype=float)
        d /= max(np.linalg.norm(d), 1e-30)

        # Generate positions near center with small spread
        self.positions = center + rng.normal(0, self.dx, (self.n_particles, 3))

        # Generate velocities: beam direction + angular spread
        for i in range(self.n_particles):
            # Random perturbation to direction
            perturb = rng.normal(0, spread_rad, 3)
            v_dir = d + perturb
            v_dir /= max(np.linalg.norm(v_dir), 1e-30)
            self.velocities[i] = v_dir * v_beam

        self._initialized = True

    def push(
        self,
        E_field: np.ndarray,
        B_field: np.ndarray,
        dt: float,
    ) -> None:
        """Advance particles using Boris push.

        Args:
            E_field: Electric field (3, nx, ny, nz) [V/m].
            B_field: Magnetic field (3, nx, ny, nz) [T].
            dt: Timestep [s].
        """
        if not self._initialized:
            return

        qm = self.ion_charge / self.ion_mass
        nx, ny, nz = self.grid_shape

        for i in range(self.n_particles):
            if not self.alive[i]:
                continue

            # Nearest grid point interpolation
            ix = int(self.positions[i, 0] / self.dx) % nx
            iy = int(self.positions[i, 1] / self.dx) % ny
            iz = int(self.positions[i, 2] / self.dx) % nz

            Ex = E_field[0, ix, iy, iz]
            Ey = E_field[1, ix, iy, iz]
            Ez = E_field[2, ix, iy, iz]
            Bx = B_field[0, ix, iy, iz]
            By = B_field[1, ix, iy, iz]
            Bz = B_field[2, ix, iy, iz]

            # Boris push
            # Half-step E acceleration
            v_minus = self.velocities[i] + 0.5 * dt * qm * np.array([Ex, Ey, Ez])

            # Rotation by B
            t_vec = 0.5 * dt * qm * np.array([Bx, By, Bz])
            s_vec = 2.0 * t_vec / (1.0 + np.dot(t_vec, t_vec))
            v_prime = v_minus + np.cross(v_minus, t_vec)
            v_plus = v_minus + np.cross(v_prime, s_vec)

            # Half-step E acceleration
            self.velocities[i] = v_plus + 0.5 * dt * qm * np.array([Ex, Ey, Ez])

            # Position update
            self.positions[i] += self.velocities[i] * dt

            # Boundary check: kill particles outside domain
            domain_size = np.array([nx, ny, nz]) * self.dx
            if np.any(self.positions[i] < 0) or np.any(self.positions[i] > domain_size):
                self.alive[i] = False

    def get_result(self, n_target: float = 0.0, L_pinch: float = 0.0) -> BeamTrackerResult:
        """Compute beam tracker diagnostics.

        Args:
            n_target: Target deuterium density [m^-3] for yield calculation.
            L_pinch: Pinch column length [m].

        Returns:
            BeamTrackerResult with energy spectrum and yield.
        """
        alive_mask = self.alive
        n_alive = int(np.sum(alive_mask))

        if n_alive == 0:
            return BeamTrackerResult(
                n_particles=0, mean_energy_keV=0, max_energy_keV=0,
                Y_bt_kinetic=0, energy_spectrum=np.array([]),
                energy_bins=np.array([]), trajectory_count=0,
            )

        # Kinetic energy of alive particles [eV]
        v_sq = np.sum(self.velocities[alive_mask]**2, axis=1)
        KE_eV = 0.5 * self.ion_mass * v_sq / 1.602e-19
        KE_keV = KE_eV / 1e3

        mean_E = float(np.mean(KE_keV))
        max_E = float(np.max(KE_keV))

        # Energy spectrum
        bins = np.linspace(0, max(max_E * 1.1, 1.0), 50)
        hist, edges = np.histogram(KE_keV, bins=bins)

        # Beam-target yield estimate (if target density provided)
        Y_bt = 0.0
        if n_target > 0 and L_pinch > 0:
            # Simple estimate: Y_bt ~ n_beam * n_target * <sigma*v> * V * tau
            # Using Bosch-Hale at mean beam energy
            try:
                from dpf.diagnostics.beam_target import beam_target_yield_rate
                V_pinch = mean_E * 1e3 * 1.602e-19  # Convert to Joules
                I_equiv = 1e6  # Placeholder
                bt_rate = beam_target_yield_rate(
                    I_equiv, V_pinch, n_target, L_pinch, f_beam=0.14,
                )
                Y_bt = bt_rate * 1e-7  # rough confinement time
            except (ImportError, Exception):
                pass

        return BeamTrackerResult(
            n_particles=n_alive,
            mean_energy_keV=mean_E,
            max_energy_keV=max_E,
            Y_bt_kinetic=Y_bt,
            energy_spectrum=hist.astype(float),
            energy_bins=edges,
            trajectory_count=self.n_particles,
        )
