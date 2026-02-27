"""Exact Riemann solver for the 1D Euler equations.

Computes the exact solution to the Riemann problem given left/right initial
states, following Toro (2009) Chapter 4.  Supports arbitrary left/right
density, velocity, pressure, and adiabatic index.

The solution consists of three waves:
1. Left wave (shock or rarefaction)
2. Contact discontinuity
3. Right wave (shock or rarefaction)

Includes preset initial conditions for standard benchmark problems
(Sod, Lax, 123 problem, etc.).

Usage::

    from dpf.validation.riemann_exact import ExactRiemannSolver, SOD_LEFT, SOD_RIGHT
    solver = ExactRiemannSolver(SOD_LEFT, SOD_RIGHT, gamma=1.4)
    x = np.linspace(0, 1, 500)
    rho, u, p = solver.sample(x, t=0.2, x0=0.5)

References
----------
- Toro E.F. (2009), "Riemann Solvers and Numerical Methods for Fluid
  Dynamics", 3rd ed., Springer. Chapter 4.
- Sod G.A. (1978), J. Comput. Phys. 27, 1-31.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiemannState:
    """Primitive state (density, velocity, pressure) for one side."""

    rho: float
    u: float
    p: float


# Standard benchmark problems
SOD_LEFT = RiemannState(rho=1.0, u=0.0, p=1.0)
SOD_RIGHT = RiemannState(rho=0.125, u=0.0, p=0.1)

LAX_LEFT = RiemannState(rho=0.445, u=0.698, p=3.528)
LAX_RIGHT = RiemannState(rho=0.5, u=0.0, p=0.571)

# "123 problem" — two symmetric rarefactions
DOUBLE_RAREFACTION_LEFT = RiemannState(rho=1.0, u=-2.0, p=0.4)
DOUBLE_RAREFACTION_RIGHT = RiemannState(rho=1.0, u=2.0, p=0.4)

# Strong blast wave (Woodward & Colella 1984)
BLAST_LEFT = RiemannState(rho=1.0, u=0.0, p=1000.0)
BLAST_RIGHT = RiemannState(rho=1.0, u=0.0, p=0.01)


class ExactRiemannSolver:
    """Exact Riemann solver for the 1D Euler equations.

    Parameters
    ----------
    left : RiemannState
        Left initial state.
    right : RiemannState
        Right initial state.
    gamma : float
        Ratio of specific heats (> 1).
    tol : float
        Newton iteration convergence tolerance.
    max_iter : int
        Maximum Newton iterations.
    """

    def __init__(
        self,
        left: RiemannState,
        right: RiemannState,
        gamma: float = 1.4,
        tol: float = 1e-12,
        max_iter: int = 50,
    ):
        if gamma <= 1.0:
            raise ValueError("gamma must be > 1")
        if left.rho <= 0 or right.rho <= 0:
            raise ValueError("density must be > 0")
        if left.p <= 0 or right.p <= 0:
            raise ValueError("pressure must be > 0")

        self.L = left
        self.R = right
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

        # Derived constants
        self.g1 = (gamma - 1.0) / (2.0 * gamma)  # (gamma-1)/(2*gamma)
        self.g2 = (gamma + 1.0) / (2.0 * gamma)  # (gamma+1)/(2*gamma)
        self.g3 = 2.0 * gamma / (gamma - 1.0)    # 2*gamma/(gamma-1)
        self.g4 = 2.0 / (gamma - 1.0)            # 2/(gamma-1)
        self.g5 = 2.0 / (gamma + 1.0)            # 2/(gamma+1)
        self.g6 = (gamma - 1.0) / (gamma + 1.0)  # (gamma-1)/(gamma+1)
        self.g7 = (gamma - 1.0) / 2.0            # (gamma-1)/2

        # Sound speeds
        self.aL = np.sqrt(gamma * left.p / left.rho)
        self.aR = np.sqrt(gamma * right.p / right.rho)

        # Solve for star-state pressure and velocity
        self.pstar = self._solve_pressure()
        self.ustar = self._compute_velocity()

        # Compute star-state densities and wave speeds
        self._compute_wave_structure()

    def _pressure_function(self, p: float, state: RiemannState, a: float) -> float:
        """Evaluate pressure function f(p) for one side (Toro eq. 4.6-4.7)."""
        if p > state.p:
            # Shock
            A = self.g5 / state.rho
            B = self.g6 * state.p
            return (p - state.p) * np.sqrt(A / (p + B))
        else:
            # Rarefaction
            return self.g4 * a * ((p / state.p) ** self.g1 - 1.0)

    def _pressure_derivative(self, p: float, state: RiemannState, a: float) -> float:
        """Derivative df/dp for Newton iteration (Toro eq. 4.37)."""
        if p > state.p:
            # Shock
            A = self.g5 / state.rho
            B = self.g6 * state.p
            qrt = np.sqrt(A / (p + B))
            return qrt * (1.0 - (p - state.p) / (2.0 * (p + B)))
        else:
            # Rarefaction
            return (1.0 / (state.rho * a)) * (p / state.p) ** (-(self.gamma + 1.0) / (2.0 * self.gamma))

    def _solve_pressure(self) -> float:
        """Find star-state pressure p* via Newton-Raphson iteration."""
        # Two-rarefaction initial guess (Toro eq. 4.46)
        ppv = 0.5 * (self.L.p + self.R.p) - 0.125 * (
            self.R.u - self.L.u
        ) * (self.L.rho + self.R.rho) * (self.aL + self.aR)
        p0 = max(ppv, 1e-10)

        # Newton iteration
        for _ in range(self.max_iter):
            fL = self._pressure_function(p0, self.L, self.aL)
            fR = self._pressure_function(p0, self.R, self.aR)
            f = fL + fR + (self.R.u - self.L.u)

            dfL = self._pressure_derivative(p0, self.L, self.aL)
            dfR = self._pressure_derivative(p0, self.R, self.aR)
            df = dfL + dfR

            if abs(df) < 1e-30:
                break

            dp = -f / df
            p_new = p0 + dp

            if p_new < 0:
                p_new = self.tol * p0

            if abs(dp) < self.tol * (0.5 * (p0 + p_new)):
                return p_new
            p0 = p_new

        return p0

    def _compute_velocity(self) -> float:
        """Compute star-state velocity u* (Toro eq. 4.9)."""
        fL = self._pressure_function(self.pstar, self.L, self.aL)
        fR = self._pressure_function(self.pstar, self.R, self.aR)
        return 0.5 * (self.L.u + self.R.u) + 0.5 * (fR - fL)

    def _compute_wave_structure(self) -> None:
        """Determine wave types, speeds, and star-state densities."""
        p, u = self.pstar, self.ustar
        g = self.gamma

        # Left wave
        if p > self.L.p:
            # Left shock
            self.left_type = "shock"
            qL = np.sqrt(p / self.L.p * self.g2 + self.g1)
            self.SL = self.L.u - self.aL * qL
            self.rhostarL = self.L.rho * (
                (p / self.L.p + self.g6) / (self.g6 * p / self.L.p + 1.0)
            )
        else:
            # Left rarefaction
            self.left_type = "rarefaction"
            self.rhostarL = self.L.rho * (p / self.L.p) ** (1.0 / g)
            astarL = self.aL * (p / self.L.p) ** self.g1
            self.SHL = self.L.u - self.aL   # head (left-most)
            self.STL = u - astarL             # tail (right-most)

        # Right wave
        if p > self.R.p:
            # Right shock
            self.right_type = "shock"
            qR = np.sqrt(p / self.R.p * self.g2 + self.g1)
            self.SR = self.R.u + self.aR * qR
            self.rhostarR = self.R.rho * (
                (p / self.R.p + self.g6) / (self.g6 * p / self.R.p + 1.0)
            )
        else:
            # Right rarefaction
            self.right_type = "rarefaction"
            self.rhostarR = self.R.rho * (p / self.R.p) ** (1.0 / g)
            astarR = self.aR * (p / self.R.p) ** self.g1
            self.SHR = self.R.u + self.aR   # head (right-most)
            self.STR = u + astarR             # tail (left-most)

    def _sample_point(self, xi: float) -> tuple[float, float, float]:
        """Sample the exact solution at similarity variable xi = (x-x0)/t.

        Returns (density, velocity, pressure).
        """
        p, u = self.pstar, self.ustar
        g = self.gamma

        if xi < u:
            # Left of contact discontinuity
            if self.left_type == "shock":
                if xi < self.SL:
                    return self.L.rho, self.L.u, self.L.p
                else:
                    return self.rhostarL, u, p
            else:
                # Left rarefaction
                if xi < self.SHL:
                    return self.L.rho, self.L.u, self.L.p
                elif xi < self.STL:
                    # Inside rarefaction fan
                    c = self.g5 * (
                        self.aL + self.g7 * (self.L.u - xi)
                    )
                    v = self.g5 * (
                        self.aL * self.g4 + self.L.u + xi
                    ) / (1.0 + self.g4)
                    # Cleaner form from Toro:
                    v = (2.0 / (g + 1.0)) * (
                        self.aL + 0.5 * (g - 1.0) * self.L.u + xi
                    )
                    c = (2.0 / (g + 1.0)) * (
                        self.aL + 0.5 * (g - 1.0) * (self.L.u - xi)
                    )
                    rho = self.L.rho * (c / self.aL) ** self.g4
                    pr = self.L.p * (c / self.aL) ** self.g3
                    return rho, v, pr
                else:
                    return self.rhostarL, u, p
        else:
            # Right of contact discontinuity
            if self.right_type == "shock":
                if xi > self.SR:
                    return self.R.rho, self.R.u, self.R.p
                else:
                    return self.rhostarR, u, p
            else:
                # Right rarefaction
                if xi > self.SHR:
                    return self.R.rho, self.R.u, self.R.p
                elif xi > self.STR:
                    # Inside rarefaction fan
                    v = (2.0 / (g + 1.0)) * (
                        -self.aR + 0.5 * (g - 1.0) * self.R.u + xi
                    )
                    c = (2.0 / (g + 1.0)) * (
                        self.aR - 0.5 * (g - 1.0) * (self.R.u - xi)
                    )
                    rho = self.R.rho * (c / self.aR) ** self.g4
                    pr = self.R.p * (c / self.aR) ** self.g3
                    return rho, v, pr
                else:
                    return self.rhostarR, u, p

    def sample(
        self,
        x: np.ndarray,
        t: float,
        x0: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the exact Riemann solution at positions x and time t.

        Parameters
        ----------
        x : ndarray
            Spatial positions.
        t : float
            Time (must be > 0).
        x0 : float
            Initial discontinuity position.

        Returns
        -------
        rho : ndarray
            Density profile.
        u : ndarray
            Velocity profile.
        p : ndarray
            Pressure profile.
        """
        if t <= 0:
            raise ValueError("t must be > 0")

        x = np.asarray(x, dtype=np.float64)
        rho = np.empty_like(x)
        vel = np.empty_like(x)
        pre = np.empty_like(x)

        for i in range(len(x)):
            xi = (x[i] - x0) / t
            rho[i], vel[i], pre[i] = self._sample_point(xi)

        return rho, vel, pre

    def get_star_state(self) -> dict:
        """Return the star-state quantities.

        Returns
        -------
        dict
            Keys: pstar, ustar, rhostarL, rhostarR, left_type, right_type.
        """
        return {
            "pstar": self.pstar,
            "ustar": self.ustar,
            "rhostarL": self.rhostarL,
            "rhostarR": self.rhostarR,
            "left_type": self.left_type,
            "right_type": self.right_type,
        }
