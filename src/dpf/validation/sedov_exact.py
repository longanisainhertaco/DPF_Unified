"""Exact Sedov-Taylor self-similar blast wave solution.

Computes the density, pressure, velocity, and internal energy profiles
for the Sedov-Taylor point explosion in planar (n=1), cylindrical (n=2),
or spherical (n=3) geometry, following Kamm & Timmes (2007), LA-UR-07-2849.

The approach:
1. Compute the dimensionless energy integral alpha via numerical quadrature.
2. Given E0, rho0, gamma, geometry, and time t, compute shock radius R_shock.
3. Map the similarity variable v to the spatial variable lambda (= r/R_shock).
4. Evaluate Sedov functions f(v), g(v), h(v) to get velocity, density, pressure.
5. Interpolate from the v-parametric solution onto the desired radial grid.

Usage::

    from dpf.validation.sedov_exact import SedovExact
    sol = SedovExact(geometry=3, gamma=5./3., eblast=1.0, rho0=1.0)
    r, rho, p, u, e_int = sol.evaluate(r_pts, t=0.01)

References
----------
- Kamm J.R. & Timmes F.X. (2007), "On Efficient Generation of Numerically
  Robust Sedov Solutions", LA-UR-07-2849.
- Sedov L.I. (1959), "Similarity and Dimensional Methods in Mechanics",
  Academic Press.
"""

import numpy as np
from scipy import integrate as sci_int
from scipy import optimize as sci_opt
from scipy.interpolate import interp1d


class SedovExact:
    """Exact Sedov-Taylor blast wave solution.

    Parameters
    ----------
    geometry : int
        1 = planar, 2 = cylindrical, 3 = spherical.
    gamma : float
        Ratio of specific heats (> 1).
    eblast : float
        Total deposited energy.
    rho0 : float
        Ambient density.
    omega : float
        Power-law exponent for density profile rho = rho0 * r^(-omega).
        Default 0 (uniform medium).
    """

    def __init__(
        self,
        geometry: int = 3,
        gamma: float = 5.0 / 3.0,
        eblast: float = 1.0,
        rho0: float = 1.0,
        omega: float = 0.0,
    ):
        if geometry not in (1, 2, 3):
            raise ValueError("geometry must be 1 (planar), 2 (cylindrical), or 3 (spherical)")
        if gamma <= 1.0:
            raise ValueError("gamma must be > 1")
        if rho0 <= 0.0:
            raise ValueError("rho0 must be > 0")
        if eblast <= 0.0:
            raise ValueError("eblast must be > 0")
        if omega < 0 or omega >= geometry:
            raise ValueError("omega must be in [0, geometry)")

        self.geometry = geometry
        self.gamma = gamma
        self.eblast = eblast
        self.rho0 = rho0
        self.omega = omega

        # Frequently used combinations
        self.gamm1 = gamma - 1.0
        self.gamp1 = gamma + 1.0
        self.gpogm = self.gamp1 / self.gamm1  # (gamma+1)/(gamma-1)
        self.xg2 = geometry + 2.0 - omega
        self.denom2 = 2.0 * self.gamm1 + geometry - gamma * omega
        self.denom3 = geometry * (2.0 - gamma) - omega

        # Post-shock similarity velocity
        self.v2 = 4.0 / (self.xg2 * self.gamp1)
        self.vstar = 2.0 / (self.gamm1 * geometry + 2.0)

        # Solution type
        osmall = 1.0e-4
        if abs(self.v2 - self.vstar) <= osmall:
            self.solution_type = "singular"
        elif self.v2 < self.vstar - osmall:
            self.solution_type = "standard"
        else:
            self.solution_type = "vacuum"

        # Handle special singularities in denom2, denom3
        self.special = "none"
        if abs(self.denom2) <= osmall:
            self.special = "omega2"
            self.denom2 = 1.0e-8
        elif abs(self.denom3) <= osmall:
            self.special = "omega3"
            self.denom3 = 1.0e-8

        # Exponents (Kamm eqs. 42-47)
        self.a0 = 2.0 / self.xg2
        self.a2 = -self.gamm1 / self.denom2
        self.a1 = (
            self.xg2
            * gamma
            / (2.0 + geometry * self.gamm1)
            * (
                (2.0 * (geometry * (2.0 - gamma) - omega)) / (gamma * self.xg2**2)
                - self.a2
            )
        )
        self.a3 = (geometry - omega) / self.denom2
        self.a4 = self.xg2 * (geometry - omega) * self.a1 / self.denom3
        self.a5 = (omega * self.gamp1 - 2.0 * geometry) / self.denom3

        # Frequent combinations (Kamm eqs. 33-37)
        self.a_val = 0.25 * self.xg2 * self.gamp1
        self.b_val = self.gpogm
        self.c_val = 0.5 * self.xg2 * gamma
        self.d_val = (self.xg2 * self.gamp1) / (
            self.xg2 * self.gamp1 - 2.0 * (2.0 + geometry * self.gamm1)
        )
        self.e_val = 0.5 * (2.0 + geometry * self.gamm1)

        # Compute the dimensionless energy integral alpha
        self._compute_alpha()

    def _compute_alpha(self) -> None:
        """Compute the dimensionless energy integral alpha."""
        if self.solution_type == "singular":
            self.eval2 = self.gamp1 / (
                self.geometry * (self.gamm1 * self.geometry + 2.0) ** 2
            )
            self.eval1 = 2.0 / self.gamm1 * self.eval2
            self.alpha = (
                self.gpogm
                * 2**self.geometry
                / (self.geometry * (self.gamm1 * self.geometry + 2.0) ** 2)
            )
            if self.geometry != 1:
                self.alpha *= np.pi
        else:
            self.v0 = 2.0 / (self.xg2 * self.gamma)
            self.vv = 2.0 / self.xg2

            if self.solution_type == "standard":
                vmin = self.v0
            else:
                vmin = self.vv

            self.eval1 = sci_int.quad(
                self._efun01, vmin, self.v2, epsabs=1e-12, limit=200
            )[0]
            self.eval2 = sci_int.quad(
                self._efun02, vmin, self.v2, epsabs=1e-12, limit=200
            )[0]

            if self.geometry == 1:
                self.alpha = 0.5 * self.eval1 + self.eval2 / self.gamm1
            else:
                self.alpha = (self.geometry - 1.0) * np.pi * (
                    self.eval1 + 2.0 * self.eval2 / self.gamm1
                )

    def _sedov_funcs(self, v: float) -> tuple:
        """Evaluate Sedov functions at similarity variable v.

        Returns (lambda, dlambda/dv, f, g, h) where:
        - lambda = r / R_shock (dimensionless position)
        - f = V (dimensionless velocity)
        - g = D (dimensionless density)
        - h = P (dimensionless pressure)
        """
        eps = 1.0e-30

        x1 = self.a_val * v
        dx1dv = self.a_val

        cbag = max(eps, self.c_val * v - 1.0)
        x2 = self.b_val * cbag
        dx2dv = self.b_val * self.c_val

        ebag = 1.0 - self.e_val * v
        x3 = self.d_val * ebag
        dx3dv = -self.d_val * self.e_val

        x4 = self.b_val * (1.0 - 0.5 * self.xg2 * v)
        x4 = max(x4, 1.0e-12)
        dx4dv = -self.b_val * 0.5 * self.xg2

        if self.special == "omega2":
            beta0 = 1.0 / (2.0 * self.e_val)
            pp1 = self.gamm1 * beta0
            c6 = 0.5 * self.gamp1
            c2 = c6 / self.gamma
            y = 1.0 / (x1 - c2)
            z = (1.0 - x1) * y
            pp2 = self.gamp1 * beta0 * z
            dpp2dv = -self.gamp1 * beta0 * dx1dv * y * (1.0 + z)
            pp3 = (4.0 - self.geometry - 2.0 * self.gamma) * beta0
            pp4 = -self.geometry * self.gamma * beta0

            l_fun = x1 ** (-self.a0) * x2**pp1 * np.exp(pp2)
            dlamdv = (-self.a0 * dx1dv / x1 + pp1 * dx2dv / x2 + dpp2dv) * l_fun
            f_fun = x1 * l_fun
            g_fun = (
                x1 ** (self.a0 * self.omega)
                * x2**pp3
                * x4**self.a5
                * np.exp(-2.0 * pp2)
            )
            h_fun = (
                x1 ** (self.a0 * self.geometry)
                * x2**pp4
                * x4 ** (1.0 + self.a5)
            )

        elif self.special == "omega3":
            beta0 = 1.0 / (2.0 * self.e_val)
            pp1 = self.a3 + self.omega * self.a2
            pp2 = 1.0 - 4.0 * beta0
            c6 = 0.5 * self.gamp1
            pp3 = (
                -self.geometry
                * self.gamma
                * self.gamp1
                * beta0
                * (1.0 - x1)
                / (c6 - x1)
            )
            pp4 = 2.0 * (self.geometry * self.gamm1 - self.gamma) * beta0

            l_fun = x1 ** (-self.a0) * x2 ** (-self.a2) * x4 ** (-self.a1)
            dlamdv = (
                -(self.a0 * dx1dv / x1 + self.a2 * dx2dv / x2 + self.a1 * dx4dv / x4)
                * l_fun
            )
            f_fun = x1 * l_fun
            g_fun = (
                x1 ** (self.a0 * self.omega)
                * x2**pp1
                * x4**pp2
                * np.exp(pp3)
            )
            h_fun = (
                x1 ** (self.a0 * self.geometry) * x4**pp4 * np.exp(pp3)
            )

        else:
            # Standard or vacuum case (Kamm eqs. 38-41)
            l_fun = x1 ** (-self.a0) * x2 ** (-self.a2) * x3 ** (-self.a1)
            dlamdv = (
                -(
                    self.a0 * dx1dv / x1
                    + self.a2 * dx2dv / x2
                    + self.a1 * dx3dv / x3
                )
                * l_fun
            )
            f_fun = x1 * l_fun
            g_fun = (
                x1 ** (self.a0 * self.omega)
                * x2 ** (self.a3 + self.a2 * self.omega)
                * x3 ** (self.a4 + self.a1 * self.omega)
                * x4**self.a5
            )
            h_fun = (
                x1 ** (self.a0 * self.geometry)
                * x3 ** (self.a4 + self.a1 * (self.omega - 2.0))
                * x4 ** (1.0 + self.a5)
            )

        return l_fun, dlamdv, f_fun, g_fun, h_fun

    def _efun01(self, v: float) -> float:
        """Integrand for the kinetic energy component of alpha."""
        l_fun, dlamdv, f_fun, g_fun, h_fun = self._sedov_funcs(v)
        return dlamdv * l_fun ** (self.geometry + 1.0) * self.gpogm * g_fun * v**2

    def _efun02(self, v: float) -> float:
        """Integrand for the internal energy component of alpha."""
        l_fun, dlamdv, f_fun, g_fun, h_fun = self._sedov_funcs(v)
        z = 8.0 / ((self.geometry + 2.0 - self.omega) ** 2 * self.gamp1)
        return dlamdv * l_fun ** (self.geometry - 1.0) * h_fun * z

    def shock_radius(self, t: float) -> float:
        """Compute shock radius at time t.

        R_shock = (E / (alpha * rho0))^(1/xg2) * t^(2/xg2)

        For uniform medium (omega=0):
        - Spherical:  R = (E/(alpha*rho0))^(1/5) * t^(2/5)
        - Cylindrical: R = (E/(alpha*rho0))^(1/4) * t^(1/2)
        - Planar:     R = (E/(alpha*rho0))^(1/3) * t^(2/3)
        """
        return (self.eblast / (self.alpha * self.rho0)) ** (1.0 / self.xg2) * t ** (
            2.0 / self.xg2
        )

    def shock_velocity(self, t: float) -> float:
        """Compute shock velocity at time t: dR/dt = (2/xg2) * R / t."""
        r2 = self.shock_radius(t)
        return (2.0 / self.xg2) * r2 / t

    def evaluate(
        self,
        r_pts: np.ndarray,
        t: float,
        npts: int = 5001,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the Sedov solution at given radial points and time.

        Parameters
        ----------
        r_pts : ndarray
            Radial positions where the solution is desired.
        t : float
            Time at which to evaluate the solution (must be > 0).
        npts : int
            Number of points for internal parametric solution.

        Returns
        -------
        r : ndarray
            Same as input r_pts.
        density : ndarray
            Density profile.
        pressure : ndarray
            Pressure profile.
        velocity : ndarray
            Velocity profile.
        specific_internal_energy : ndarray
            Specific internal energy profile.
        """
        if t <= 0:
            raise ValueError("t must be > 0 for the Sedov solution")

        r_pts = np.asarray(r_pts, dtype=np.float64)

        # Shock conditions
        r2 = self.shock_radius(t)
        us = self.shock_velocity(t)
        rho1 = self.rho0 * r2 ** (-self.omega)
        rho2 = self.gpogm * rho1  # post-shock density
        u2 = 2.0 * us / self.gamp1  # post-shock velocity
        p2 = 2.0 * rho1 * us**2 / self.gamp1  # post-shock pressure

        # Build parametric solution on fine grid
        r_eval = np.linspace(0.0, np.max(r_pts) * 1.05, npts)[::-1]

        density = np.zeros(npts)
        velocity = np.zeros(npts)
        pressure = np.zeros(npts)

        if self.solution_type == "standard":
            vmin = 2.0 / (self.xg2 * self.gamma)
        elif self.solution_type == "vacuum":
            vmin = 2.0 / self.xg2
        else:
            vmin = None

        # Vacuum boundary
        rvv = 0.0
        if self.solution_type == "vacuum":
            l_rvv = self._sedov_funcs(2.0 / self.xg2)[0]
            rvv = l_rvv * r2

        vtol = 1.0e-8
        vconverged = False
        last_v = None

        for i in range(npts):
            rwant = r_eval[i]

            if rwant > r2:
                # Outside shock: ambient
                density[i] = self.rho0 * max(rwant, 1e-30) ** (-self.omega)
                velocity[i] = 0.0
                pressure[i] = 0.0
                continue

            if self.solution_type == "singular":
                lam = rwant / r2
                f_fun = lam
                g_fun = lam ** (self.geometry - 2.0) if lam > 0 else 0.0
                h_fun = lam**self.geometry if lam > 0 else 0.0
            elif self.solution_type == "vacuum" and rwant < rvv:
                f_fun = 0.0
                g_fun = 0.0
                h_fun = 0.0
            else:
                lam_want = rwant / r2

                def objective(v, _target=lam_want):
                    l_fun = self._sedov_funcs(v)[0]
                    return (l_fun - _target) ** 2

                if self.solution_type == "standard":
                    v_lo, v_hi = vmin, self.v2
                else:
                    v_lo, v_hi = self.v2, 2.0 / self.xg2

                try:
                    vwant = sci_opt.fminbound(
                        objective, v_lo, v_hi, xtol=1e-30, maxfun=2000
                    )
                except Exception:
                    vwant = 0.5 * (v_lo + v_hi)

                if last_v is not None and abs(vwant - last_v) < vtol:
                    vconverged = True
                last_v = vwant

                _, _, f_fun, g_fun, h_fun = self._sedov_funcs(vwant)

            density[i] = rho2 * g_fun
            velocity[i] = u2 * f_fun
            pressure[i] = p2 * h_fun

            if vconverged:
                # Fill remaining interior with origin values
                density[i + 1:] = density[i]
                velocity[i + 1:] = 0.0
                pressure[i + 1:] = pressure[i]
                break

        # Ensure origin values
        velocity[-1] = 0.0

        # Clean up: density should not be exactly zero for interpolation
        density = np.maximum(density, 1e-30)

        # Interpolate onto requested r_pts
        r_eval_clean = r_eval[:i + 2] if vconverged else r_eval
        density_clean = density[:i + 2] if vconverged else density
        velocity_clean = velocity[:i + 2] if vconverged else velocity
        pressure_clean = pressure[:i + 2] if vconverged else pressure

        # Sort for interp1d (needs increasing x)
        sort_idx = np.argsort(r_eval_clean)
        r_sorted = r_eval_clean[sort_idx]
        den_sorted = density_clean[sort_idx]
        vel_sorted = velocity_clean[sort_idx]
        pre_sorted = pressure_clean[sort_idx]

        # Remove duplicate r values
        unique_mask = np.concatenate(([True], np.diff(r_sorted) > 0))
        r_sorted = r_sorted[unique_mask]
        den_sorted = den_sorted[unique_mask]
        vel_sorted = vel_sorted[unique_mask]
        pre_sorted = pre_sorted[unique_mask]

        f_den = interp1d(r_sorted, den_sorted, kind="linear", fill_value="extrapolate")
        f_vel = interp1d(r_sorted, vel_sorted, kind="linear", fill_value="extrapolate")
        f_pre = interp1d(r_sorted, pre_sorted, kind="linear", fill_value="extrapolate")

        rho_out = np.maximum(f_den(r_pts), 1e-30)
        vel_out = f_vel(r_pts)
        pre_out = np.maximum(f_pre(r_pts), 0.0)

        # Outside shock
        outside = r_pts > r2
        rho_out[outside] = self.rho0
        vel_out[outside] = 0.0
        pre_out[outside] = 0.0

        sie_out = np.where(rho_out > 1e-20, pre_out / (self.gamm1 * rho_out), 0.0)

        return r_pts, rho_out, pre_out, vel_out, sie_out

    def get_alpha(self) -> float:
        """Return the dimensionless energy integral alpha."""
        return self.alpha

    def get_shock_info(self, t: float) -> dict:
        """Return shock properties at time t.

        Returns
        -------
        dict
            Keys: R_shock, U_shock, rho_pre, rho_post, u_post, p_post, alpha.
        """
        r2 = self.shock_radius(t)
        us = self.shock_velocity(t)
        rho1 = self.rho0 * r2 ** (-self.omega)
        rho2 = self.gpogm * rho1
        u2 = 2.0 * us / self.gamp1
        p2 = 2.0 * rho1 * us**2 / self.gamp1
        return {
            "R_shock": r2,
            "U_shock": us,
            "rho_pre": rho1,
            "rho_post": rho2,
            "u_post": u2,
            "p_post": p2,
            "alpha": self.alpha,
        }
