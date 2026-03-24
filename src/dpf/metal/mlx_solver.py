"""MLXMHDSolver -- cylindrical MHD on Apple Silicon via MLX Metal kernels.

Implements ``PlasmaSolverBase`` using the MLX-native MHD pipeline:
  - SSP-RK3 time integration (Shu & Osher 1988)
  - WENO5-Z or PLM reconstruction
  - HLLD or HLL Riemann solver
  - Cylindrical finite-volume with geometric source terms
  - Dual-energy entropy tracer (Popovas 2025 DISPATCH switching criterion)
  - Electrode ghost-cell BCs (B_theta = mu0*I / 2*pi*r at cathode face)
  - Operator-split implicit resistive diffusion (Thomas solver)
  - Operator-split Braginskii parallel thermal conduction

Conservative variable ordering (internal, shape (10, nr, nz))::

    [rho, rho*vr, rho*vz, rho*vt, E, S*rho, Br, Bz, Btheta, e_electron]

The solver accepts and returns NumPy ``dict[str, np.ndarray]`` per the
``PlasmaSolverBase`` contract and is drop-in compatible with MetalMHDSolver.

Note on units: the MLX solver works in Heaviside-Lorentz units (mu_0=1),
matching the Athena++ convention.  Pass ``convert_b_si_to_hl=True`` when
the incoming state dict carries B in SI Tesla.

References:
    Shu C.-W. & Osher S., JCP 77:439 (1988) -- SSP-RK3.
    Borges et al., JCP 227:3101 (2008) -- WENO-Z nonlinear weights.
    Miyoshi T. & Kusano K., JCP 208:315 (2005) -- HLLD Riemann solver.
    Popovas et al., arXiv:2211.02438 (2025) -- dual-energy entropy switch.
    Braginskii S.I., Rev. Plasma Phys. 1 (1965) -- transport coefficients.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from dpf.core.bases import CouplingState, PlasmaSolverBase
from dpf.metal.mlx_device import HAS_MLX, require_mlx

logger = logging.getLogger(__name__)

_MU0: float = 4.0 * math.pi * 1e-7
_SQRT_MU0: float = math.sqrt(_MU0)
_K_B: float = 1.380649e-23
_M_DEUTERIUM: float = 3.34358377e-27


class MLXMHDSolver(PlasmaSolverBase):
    """MLX-native cylindrical MHD solver for Apple Silicon.

    Fully compatible with ``MetalMHDSolver``'s constructor signature so the
    engine can swap backends without changing call sites.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        ``(nr, ny, nz)`` — ``ny`` must equal 1 (axisymmetric).
    dx : float
        Radial cell spacing [m].
    gamma : float
        Adiabatic index (default 5/3).
    cfl : float
        Courant number (default 0.3).
    dz : float | None
        Axial cell spacing [m].  Defaults to ``dx`` when None.
    riemann_solver : str
        ``"hlld"`` (default) or ``"hll"``.
    reconstruction : str
        ``"weno5z"`` (default) or ``"plm"``.  ``"weno5"`` accepted as alias.
    time_integrator : str
        ``"ssp_rk3"`` (default) or ``"ssp_rk2"``.
    coordinates : str
        ``"cylindrical"`` (default) or ``"cartesian"``.
    r_inner : float
        Inner radial boundary [m] (default 0.0 = axis).
    convert_b_si_to_hl : bool
        If True, divide B by ``sqrt(mu_0)`` on input / multiply on output.
    ion_mass : float
        Ion mass [kg] for temperature conversion (default deuterium).
    enable_hall : bool
        Hall term flag (stored; not yet wired — future use).
    enable_braginskii_conduction : bool
        Enable operator-split Braginskii parallel thermal conduction.
    enable_braginskii_viscosity : bool
        Viscosity flag (stored; not yet wired — future use).
    enable_bremsstrahlung : bool
        Bremsstrahlung radiation flag (stored; engine drives the operator split).
    gaunt_factor : float
        Free-free Gaunt factor for bremsstrahlung (default 1.2).
    Z_eff : float
        Effective charge number (default 1.0).
    use_dual_energy : bool
        Enable dual-energy entropy tracer.  Always forced True for cylindrical.
    device, precision, use_ct, limiter, enable_nernst, compile_mode
        Ignored — accepted for API compatibility with ``MetalMHDSolver``.
    """

    def __init__(  # noqa: PLR0913
        self,
        grid_shape: tuple[int, int, int],
        dx: float,
        gamma: float = 5.0 / 3.0,
        cfl: float = 0.3,
        dz: float | None = None,
        riemann_solver: str = "hlld",
        reconstruction: str = "weno5z",
        time_integrator: str = "ssp_rk3",
        coordinates: str = "cylindrical",
        r_inner: float = 0.0,
        convert_b_si_to_hl: bool = False,
        ion_mass: float = _M_DEUTERIUM,
        enable_hall: bool = False,
        enable_braginskii_conduction: bool = False,
        enable_braginskii_viscosity: bool = False,
        enable_bremsstrahlung: bool = False,
        gaunt_factor: float = 1.2,
        Z_eff: float = 1.0,
        use_dual_energy: bool = True,
        # API-compat ignored params
        device: str = "mlx",
        precision: str = "float32",
        use_ct: bool = True,
        limiter: str = "mc",
        enable_nernst: bool = False,
        compile_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        nr, ny, nz = grid_shape
        if ny != 1:
            raise ValueError(
                f"MLXMHDSolver is axisymmetric (ny=1 required), got ny={ny}."
            )

        self.grid_shape: tuple[int, int, int] = (nr, ny, nz)
        self.nr: int = nr
        self.nz: int = nz
        self.dx: float = float(dx)
        self.dz: float = float(dz) if dz is not None else float(dx)
        self.gamma: float = float(gamma)
        self.cfl: float = float(cfl)
        self.coordinates: str = coordinates
        self._r_inner: float = float(r_inner if r_inner is not None else 0.0)
        self._convert_b_si_to_hl: bool = convert_b_si_to_hl
        self.ion_mass: float = float(ion_mass)

        self.enable_hall: bool = enable_hall
        self.enable_braginskii_conduction: bool = enable_braginskii_conduction
        self.enable_braginskii_viscosity: bool = enable_braginskii_viscosity
        self.enable_bremsstrahlung: bool = enable_bremsstrahlung
        self.gaunt_factor: float = float(gaunt_factor)
        self.Z_eff: float = float(Z_eff)

        # dual-energy is always active for cylindrical coordinates
        self._use_dual_energy: bool = use_dual_energy or (coordinates == "cylindrical")

        # Normalise reconstruction: "weno5" is an alias for "weno5z"
        if reconstruction in ("weno5", "weno5z"):
            self._method: str = "weno5z"
        else:
            self._method = "plm"

        self._riemann: str = riemann_solver if riemann_solver in ("hlld", "hll") else "hlld"
        self._integrator: str = time_integrator

        # Circuit coupling state — updated each step
        self._coupling: CouplingState = CouplingState()
        self.total_radiated_energy: float = 0.0

        # Internal conserved state (mx.array, set after first step)
        self._U: Any = None
        self._entropy_initialized: bool = False

        # Grid and state manager — built eagerly if MLX is present
        self._grid: Any = None
        self._state_mgr: Any = None
        if HAS_MLX:
            self._build_internals()

        logger.info(
            "MLXMHDSolver: %dx%d grid  dr=%.3g  dz=%.3g  coords=%s  "
            "%s+%s+%s  dual_energy=%s  braginskii_cond=%s  brem=%s",
            nr, nz, self.dx, self.dz, coordinates,
            self._method, self._riemann, self._integrator,
            self._use_dual_energy, enable_braginskii_conduction, enable_bremsstrahlung,
        )

    # ------------------------------------------------------------------
    # Internal construction helpers
    # ------------------------------------------------------------------

    def _build_internals(self) -> None:
        """Instantiate CylindricalGrid and MLXState (MLX must be available)."""
        from dpf.metal.mlx_grid import CylindricalGrid
        from dpf.metal.mlx_state import MLXState

        self._grid = CylindricalGrid(
            nr=self.nr,
            nz=self.nz,
            dr=self.dx,
            dz=self.dz,
            r_inner=self._r_inner,
        )
        self._state_mgr = MLXState(
            nr=self.nr,
            nz=self.nz,
            gamma=self.gamma,
            ion_mass=self.ion_mass,
        )

    def _ensure_internals(self) -> None:
        """Lazy-init grid + state manager if MLX became importable after __init__."""
        if self._grid is None:
            self._build_internals()

    # ------------------------------------------------------------------
    # Electrode ghost-cell boundary condition
    # ------------------------------------------------------------------

    def _apply_electrode_bc(self, U: Any, current: float) -> Any:
        """Encode B_theta = mu_0 * I / (2*pi*r) at the outer radial boundary.

        The outermost radial cell row in IBT is overwritten with the
        current-sheet value so the Riemann solver sees the correct jump
        condition at the cathode face.  Inner boundary (axis) retains
        B_theta = 0 by symmetry.

        Parameters
        ----------
        U : mx.array
            Conserved state, shape (NVAR, nr, nz).
        current : float
            Circuit current [A].

        Returns
        -------
        mx.array
            Updated U with electrode BC applied to the outer IBT row.
        """
        mx = require_mlx()
        from dpf.metal.mlx_kernels import IBT, NVAR

        r_outer = float(self._grid.r_cell[-1])
        if r_outer <= 0.0:
            return U

        Bt_electrode = (_MU0 * current) / (2.0 * math.pi * r_outer)
        if self._convert_b_si_to_hl:
            Bt_electrode /= _SQRT_MU0

        Bt_slice = U[IBT]  # (nr, nz)
        inner = Bt_slice[: self.nr - 1, :]  # (nr-1, nz)
        outer_bc = mx.full((1, self.nz), float(Bt_electrode), dtype=mx.float32)
        Bt_new = mx.concatenate([inner, outer_bc], axis=0)

        rows = [U[i] for i in range(NVAR)]
        rows[IBT] = Bt_new
        return mx.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Operator-split: resistive diffusion
    # ------------------------------------------------------------------

    def _do_resistive_diffusion(self, U: Any, dt: float, eta: Any) -> Any:
        """Implicit resistive diffusion of the B-field with Ohmic heating.

        Parameters
        ----------
        U : mx.array
            Conserved state (NVAR, nr, nz).
        dt : float
            Timestep [s].
        eta : float or mx.array
            Resistivity [Ohm·m].

        Returns
        -------
        mx.array
            Updated U with diffused B and Ohmic pressure increment.
        """
        mx = require_mlx()
        from dpf.metal.mlx_kernels import IBR, IBT, IBZ, IEN, ISR, NVAR
        from dpf.metal.mlx_primitives import P_FLOOR, cons_to_prim
        from dpf.metal.mlx_transport import apply_resistive_diffusion

        rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, self.gamma)

        Br_new, Bz_new, Bt_new, p_new = apply_resistive_diffusion(
            Br=Br, Bz=Bz, Bt=Bt,
            rho=rho, p=p,
            eta=eta, dt=dt,
            dr=self._grid.dr, dz=self._grid.dz,
            r_cell=self._grid.r_cell,
            gamma=self.gamma,
        )

        gm1 = self.gamma - 1.0
        KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
        ME_new = 0.5 * (Br_new * Br_new + Bz_new * Bz_new + Bt_new * Bt_new)
        E_new = mx.maximum(p_new, P_FLOOR) / gm1 + KE + ME_new
        Srho_new = mx.maximum(p_new, P_FLOOR) * mx.power(
            mx.maximum(rho, 1e-30), 1.0 - self.gamma
        )

        rows = [U[i] for i in range(NVAR)]
        rows[IBR] = Br_new
        rows[IBZ] = Bz_new
        rows[IBT] = Bt_new
        rows[IEN] = E_new
        rows[ISR] = Srho_new
        return mx.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Operator-split: Braginskii thermal conduction
    # ------------------------------------------------------------------

    def _do_thermal_conduction(self, U: Any, dt: float, kappa: float | Any) -> Any:
        """Implicit Braginskii parallel conduction along z.

        Parameters
        ----------
        U : mx.array
            Conserved state (NVAR, nr, nz).
        dt : float
            Timestep [s].
        kappa : float or mx.array
            Parallel conductivity [W/(m·K)].

        Returns
        -------
        mx.array
            Updated U with thermally diffused pressure and energy.
        """
        mx = require_mlx()
        from dpf.metal.mlx_kernels import IEN, ISR, NVAR
        from dpf.metal.mlx_primitives import P_FLOOR, cons_to_prim
        from dpf.metal.mlx_transport import apply_thermal_conduction

        rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, self.gamma)

        T = p * self.ion_mass / (mx.maximum(rho, 1e-30) * _K_B)

        Te_new, Ti_new = apply_thermal_conduction(
            Te=T, Ti=T, rho=rho, B=Bz,
            kappa_parallel=kappa,
            dt=dt, dz=self._grid.dz,
        )

        T_avg = 0.5 * (Te_new + Ti_new)
        p_new = mx.maximum(rho * _K_B * T_avg / self.ion_mass, P_FLOOR)

        gm1 = self.gamma - 1.0
        KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
        ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
        E_new = p_new / gm1 + KE + ME
        Srho_new = p_new * mx.power(mx.maximum(rho, 1e-30), 1.0 - self.gamma)

        rows = [U[i] for i in range(NVAR)]
        rows[IEN] = E_new
        rows[ISR] = Srho_new
        return mx.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # PlasmaSolverBase — compute_dt
    # ------------------------------------------------------------------

    def compute_dt(self, state: dict[str, np.ndarray]) -> float:
        """CFL-limited timestep from the current plasma state.

        Parameters
        ----------
        state : dict[str, np.ndarray]
            DPF state dict.

        Returns
        -------
        float
            Maximum stable timestep [s].
        """
        self._ensure_internals()
        mx = require_mlx()
        from dpf.metal.mlx_timestepper import compute_dt_cfl

        U = self._state_mgr.from_state_dict(
            state, convert_b_si_to_hl=self._convert_b_si_to_hl
        )
        mx.eval(U)
        return compute_dt_cfl(U, self._grid, gamma=self.gamma, cfl=self.cfl)

    # Engine calls _compute_dt in some code paths
    _compute_dt = compute_dt

    # ------------------------------------------------------------------
    # PlasmaSolverBase — step
    # ------------------------------------------------------------------

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        source_terms: dict[str, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Advance the plasma state by one timestep via MLX Metal kernels.

        Pipeline
        --------
        1. Pack state dict → conserved ``mx.array`` (optional SI→HL B).
        2. Initialise entropy tracer on the first call.
        3. Apply electrode ghost-cell BC when cylindrical and |current| > 0.
        4. SSP-RK3 (or RK2) hyperbolic advance.
        5. Operator-split implicit resistive diffusion (if ``eta_field`` kwarg).
        6. Operator-split Braginskii conduction (if enabled + ``kappa_parallel``).
        7. Unpack conserved array → state dict (optional HL→SI B revert).
        8. Update coupling state and return.

        Parameters
        ----------
        state : dict[str, np.ndarray]
            Input DPF state dict.
        dt : float
            Timestep [s].
        current : float
            Circuit current [A].
        voltage : float
            Capacitor voltage [V].
        source_terms : dict | None
            Ignored — reserved for future operator-split sources.
        **kwargs
            - ``apply_electrode_bc`` (bool) — enable ghost-cell electrode BCs.
            - ``eta_field`` (float or np.ndarray) — resistivity [Ohm·m].
            - ``kappa_parallel`` (float) — Braginskii conductivity [W/(m·K)].

        Returns
        -------
        dict[str, np.ndarray]
            Updated DPF state dict.
        """
        self._ensure_internals()
        mx = require_mlx()

        from dpf.metal.mlx_timestepper import ssp_rk2_step, ssp_rk3_step

        # ── 1. Pack ──────────────────────────────────────────────────────
        U = self._state_mgr.from_state_dict(
            state, convert_b_si_to_hl=self._convert_b_si_to_hl
        )
        mx.eval(U)

        # ── 2. Entropy initialised flag ───────────────────────────────────
        # from_state_dict already computes the entropy tracer from p and rho,
        # so no additional work is needed.  Track that we have run at least once.
        self._entropy_initialized = True

        # ── 3. Electrode BC ──────────────────────────────────────────────
        apply_bc = kwargs.get("apply_electrode_bc", False)
        if (
            apply_bc
            and self.coordinates == "cylindrical"
            and abs(current) > 1e-10
        ):
            U = self._apply_electrode_bc(U, current)
            mx.eval(U)

        # ── 4. Hyperbolic step ───────────────────────────────────────────
        step_fn = ssp_rk3_step if self._integrator != "ssp_rk2" else ssp_rk2_step
        U = step_fn(
            U, self._grid, dt,
            gamma=self.gamma,
            method=self._method,
            riemann=self._riemann,
            use_dual_energy=self._use_dual_energy,
        )
        mx.eval(U)

        # ── 5. Resistive diffusion ───────────────────────────────────────
        eta_raw = kwargs.get("eta_field")
        if eta_raw is not None:
            eta_arg: float | Any
            if isinstance(eta_raw, np.ndarray):
                eta_arg = mx.array(eta_raw.astype(np.float32))
            else:
                eta_arg = float(eta_raw)
            U = self._do_resistive_diffusion(U, dt, eta_arg)
            mx.eval(U)

        # ── 6. Braginskii conduction ─────────────────────────────────────
        if self.enable_braginskii_conduction:
            kappa = float(kwargs.get("kappa_parallel", 1e3))
            U = self._do_thermal_conduction(U, dt, kappa)
            mx.eval(U)

        # ── 7. Unpack ────────────────────────────────────────────────────
        self._U = U
        result = self._state_mgr.to_state_dict(
            U, convert_b_hl_to_si=self._convert_b_si_to_hl
        )

        # ── 8. Coupling ──────────────────────────────────────────────────
        self._coupling = CouplingState(current=current, voltage=voltage)
        return result

    # ------------------------------------------------------------------
    # PlasmaSolverBase — coupling_interface
    # ------------------------------------------------------------------

    def coupling_interface(self) -> CouplingState:
        """Return coupling quantities for the circuit solver.

        Returns
        -------
        CouplingState
            Most recently recorded current / voltage.
        """
        return self._coupling

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True when MLX is installed.

        Returns
        -------
        bool
        """
        if not HAS_MLX:
            return False
        try:
            import mlx.core as mx

            return mx.default_device().type == mx.gpu
        except Exception:  # noqa: BLE001
            return HAS_MLX
