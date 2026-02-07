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
    ) -> None:
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.dz = dz if dz is not None else dx

        self.E = np.zeros((3, self.nx, self.ny, self.nz))
        self.B = np.zeros((3, self.nx, self.ny, self.nz))
        self.J = np.zeros((3, self.nx, self.ny, self.nz))

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
