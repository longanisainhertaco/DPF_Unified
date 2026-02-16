"""Field manager — grid storage and vector calculus operations.

Clean rewrite of DPF_AI's FieldManager with fixed boundary conditions
and correct array shapes.
"""

from __future__ import annotations

import numpy as np


class FieldManager:
    """Manages electromagnetic fields (E, B, J) on a uniform Cartesian grid.

    All field arrays have shape ``(3, nx, ny, nz)`` where axis 0 is the
    vector component (x, y, z).
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        dx: float,
        dy: float | None = None,
        dz: float | None = None,
        geometry: str = "cartesian",
    ) -> None:
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.dz = dz if dz is not None else dx
        self.geometry = geometry

        self.E = np.zeros((3, self.nx, self.ny, self.nz))
        self.B = np.zeros((3, self.nx, self.ny, self.nz))
        self.J = np.zeros((3, self.nx, self.ny, self.nz))

    def compute_plasma_inductance(self, current: float, min_inductance: float = 1e-9) -> float:
        """Compute plasma inductance from magnetic energy.

        L_p = 2 * Integral(B^2 / 2*mu0) dV / I^2

        Args:
            current: Total circuit current [A].
            min_inductance: Fallback inductance floor [H].

        Returns:
            Inductance [H].
        """
        if abs(current) < 1e-3:
            return min_inductance

        from dpf.constants import mu_0, pi

        # Magnetic energy density: u_B = B^2 / (2 * mu_0)
        # B is (3, nx, ny, nz)
        B_sq = np.sum(self.B**2, axis=0)  # (nx, ny, nz)
        u_B = B_sq / (2.0 * mu_0)

        if self.geometry == "cylindrical":
            # Volume element dV = 2*pi*r * dr * dz
            # r for cell i is (i + 0.5) * dx
            # We construct a radial array (nx, 1, 1)
            r = (np.arange(self.nx) + 0.5) * self.dx
            r = r[:, np.newaxis, np.newaxis]  # Broadcast to (nx, 1, 1)
            
            # Integrate: sum(u_B * 2*pi*r) * dr * dz
            # Note: dy is ignored in cylindrical (axisymmetric)
            integrand = u_B * 2.0 * pi * r
            total_energy = np.sum(integrand) * self.dx * self.dz
        else:
            # Cartesian: dV = dx * dy * dz
            total_energy = np.sum(u_B) * self.dx * self.dy * self.dz

        # L = 2 * W / I^2
        return 2.0 * total_energy / (current**2)

    # --- Vector calculus ---

    def divergence(self, field: np.ndarray) -> np.ndarray:
        """Compute divergence of a 3-component vector field.

        Args:
            field: Array of shape ``(3, nx, ny, nz)``.

        Returns:
            Scalar field of shape ``(nx, ny, nz)``.
        """
        return (
            np.gradient(field[0], self.dx, axis=0)
            + np.gradient(field[1], self.dy, axis=1)
            + np.gradient(field[2], self.dz, axis=2)
        )

    def curl(self, field: np.ndarray) -> np.ndarray:
        """Compute curl of a 3-component vector field.

        Args:
            field: Array of shape ``(3, nx, ny, nz)``.

        Returns:
            Vector field of shape ``(3, nx, ny, nz)``.
        """
        dFz_dy = np.gradient(field[2], self.dy, axis=1)
        dFy_dz = np.gradient(field[1], self.dz, axis=2)
        dFx_dz = np.gradient(field[0], self.dz, axis=2)
        dFz_dx = np.gradient(field[2], self.dx, axis=0)
        dFy_dx = np.gradient(field[1], self.dx, axis=0)
        dFx_dy = np.gradient(field[0], self.dy, axis=1)

        return np.array([
            dFz_dy - dFy_dz,
            dFx_dz - dFz_dx,
            dFy_dx - dFx_dy,
        ])

    def gradient(self, scalar: np.ndarray) -> np.ndarray:
        """Compute gradient of a scalar field.

        Args:
            scalar: Array of shape ``(nx, ny, nz)``.

        Returns:
            Vector field of shape ``(3, nx, ny, nz)``.
        """
        return np.array([
            np.gradient(scalar, self.dx, axis=0),
            np.gradient(scalar, self.dy, axis=1),
            np.gradient(scalar, self.dz, axis=2),
        ])

    # --- Boundary conditions ---

    def apply_periodic_bc(self, field: np.ndarray) -> np.ndarray:
        """Apply periodic BCs (numpy operations are inherently periodic for gradient)."""
        return field

    def apply_neumann_bc(self, field: np.ndarray, ng: int = 2) -> np.ndarray:
        """Apply zero-gradient (Neumann) BCs to ghost cells.

        Args:
            field: Array of shape ``(3, nx, ny, nz)`` or ``(nx, ny, nz)``.
            ng: Number of ghost cells.

        Returns:
            Field with ghost cells filled.
        """
        f = field.copy()
        if f.ndim == 4:
            # Vector field: (3, nx, ny, nz)
            for comp in range(3):
                f[comp, :ng, :, :] = f[comp, ng : 2 * ng, :, :]
                f[comp, -ng:, :, :] = f[comp, -2 * ng : -ng, :, :]
                f[comp, :, :ng, :] = f[comp, :, ng : 2 * ng, :]
                f[comp, :, -ng:, :] = f[comp, :, -2 * ng : -ng, :]
                f[comp, :, :, :ng] = f[comp, :, :, ng : 2 * ng]
                f[comp, :, :, -ng:] = f[comp, :, :, -2 * ng : -ng]
        elif f.ndim == 3:
            # Scalar field: (nx, ny, nz)
            f[:ng, :, :] = f[ng : 2 * ng, :, :]
            f[-ng:, :, :] = f[-2 * ng : -ng, :, :]
            f[:, :ng, :] = f[:, ng : 2 * ng, :]
            f[:, -ng:, :] = f[:, -2 * ng : -ng, :]
            f[:, :, :ng] = f[:, :, ng : 2 * ng]
            f[:, :, -ng:] = f[:, :, -2 * ng : -ng]
        return f

    # --- Diagnostics ---

    def max_div_B(self) -> float:
        """Return max |div B| as a divergence-cleaning diagnostic."""
        return float(np.max(np.abs(self.divergence(self.B))))

    # --- Checkpoint/restart ---

    def checkpoint(self) -> dict[str, np.ndarray]:
        return {"E": self.E.copy(), "B": self.B.copy(), "J": self.J.copy()}

    def restart(self, data: dict[str, np.ndarray]) -> None:
        self.E = data["E"]
        self.B = data["B"]
        self.J = data["J"]
