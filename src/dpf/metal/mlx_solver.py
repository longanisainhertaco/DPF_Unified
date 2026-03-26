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
        if coordinates == "cylindrical" and ny != 1:
            raise ValueError(
                f"MLXMHDSolver cylindrical mode requires ny=1, got ny={ny}."
            )

        self.grid_shape: tuple[int, int, int] = (nr, ny, nz)
        self.nr: int = nr
        self.ny: int = ny
        self.nz: int = nz
        self.dx: float = float(dx)
        self.dy: float = float(kwargs.get("dy", dx))
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
        self._use_ct: bool = bool(use_ct)

        # Dedner GLM div(B) cleaning: auto-enable when CT is off, or always
        # for cylindrical (cell-centered CT is approximate, not truly div-free)
        self._enable_dedner: bool = bool(kwargs.get(
            "enable_dedner",
            coordinates == "cartesian" or coordinates == "cylindrical" or not use_ct,
        ))
        self._enable_powell: bool = bool(kwargs.get("enable_powell", False))

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
        self._prev_Lp: float = 0.0
        self._Lp_max: float = 0.0
        self._cathode_radius: float = float(kwargs.get("cathode_radius", 0.025))
        self.config_two_temperature: bool = kwargs.get("two_temperature", False)

        # Internal conserved state (mx.array, set after first step)
        self._U: Any = None
        self._entropy_initialized: bool = False
        self._psi: Any = None  # Dedner cleaning scalar (sidecar, not in U)

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
        """Instantiate grid and MLXState (MLX must be available)."""
        from dpf.metal.mlx_state import MLXState

        if self.coordinates == "cartesian":
            from dpf.metal.mlx_grid import CartesianGrid

            self._grid = CartesianGrid(
                nx=self.nr,
                ny=self.ny,
                nz=self.nz,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
            )
        else:
            from dpf.metal.mlx_grid import CylindricalGrid

            self._grid = CylindricalGrid(
                nr=self.nr,
                nz=self.nz,
                dr=self.dx,
                dz=self.dz,
                r_inner=self._r_inner,
            )
        self._state_mgr = MLXState(
            nr=self.nr,
            ny=self.ny,
            nz=self.nz,
            gamma=self.gamma,
            ion_mass=self.ion_mass,
            coordinates=self.coordinates,
        )

    def _ensure_internals(self) -> None:
        """Lazy-init grid + state manager if MLX became importable after __init__."""
        if self._grid is None:
            self._build_internals()

    # ------------------------------------------------------------------
    # Electrode ghost-cell boundary condition (pad / strip)
    # ------------------------------------------------------------------

    # WENO5-Z needs 3 ghost cells; PLM needs 2.
    _GHOST_NG: int = 3

    def _pad_electrode_ghost(self, U: Any, current: float) -> tuple[Any, Any]:
        """Pad state with ghost cells encoding electrode BCs.

        Extends the radial domain by ``_GHOST_NG`` ghost cells on each side.
        Inner ghosts (axis): reflecting with sign-flip on B_theta, B_r, mom_r, mom_t.
        Outer ghosts (cathode): zero-gradient base + B_theta = mu0*I/(2*pi*r).

        This is the same strategy used by MetalMHDSolver._pad_electrode_ghost --
        the Riemann solver sees the electrode discontinuity through the ghost cells
        without overwriting any interior cell.

        Parameters
        ----------
        U : mx.array
            Conserved state, shape (NVAR, nr, nz).
        current : float
            Circuit current [A].

        Returns
        -------
        tuple[mx.array, object]
            Padded state (NVAR, nr + 2*ng, nz) and padded grid object.
        """
        mx = require_mlx()
        from dpf.metal.mlx_kernels import ghost_pad_mlx

        ng = self._GHOST_NG
        nr_g = self.nr + 2 * ng
        dr = self._grid.dr
        dz = self._grid.dz

        # Ghost-zone radial coordinates: extend below r_inner for inner ghosts.
        # Negative r values are geometrically necessary for reflecting BCs at axis.
        # Matches PyTorch MetalMHDSolver._pad_electrode_ghost coordinate layout.
        r_inner_g = self._r_inner - ng * dr

        # Cell centres of the padded grid (may include negative r for inner ghosts)
        r_cell_list = [r_inner_g + (i + 0.5) * dr for i in range(nr_g)]
        r_cell_np = np.array(r_cell_list, dtype=np.float32)

        # ghost_pad_mlx expects r_face param as cell-centre radii for B_theta calc
        U_padded = ghost_pad_mlx(
            U, ng, "electrode",
            current=current,
            r_face=r_cell_np,
            mu0=_MU0,
        )

        # ghost_pad_mlx writes B_theta in SI in ghost cells. Three corrections:
        # 1. Convert to HL if needed.
        # 2. Set B_theta = mu0*I/(2*pi*r) in the outermost ng interior cells
        #    as well, so the transition from interior to ghost is smooth (1/r
        #    profile). Without this, WENO5-Z sees a 0→1000 HL jump and
        #    produces NaN in float32.
        # 3. Update total energy E to account for changed B_theta (energy
        #    consistency). Without this, p = (gamma-1)(E - KE - B^2/2) goes
        #    negative when B_theta >> B_theta_old, causing HLLD NaN.
        from dpf.metal.mlx_kernels import GAMMA, IBR, IBT, IBZ, IDN, IEN, P_FLOOR
        _sqrt = _SQRT_MU0 if self._convert_b_si_to_hl else 1.0
        U_np = np.asarray(U_padded)

        def _update_bt_with_energy(idx: int, Bt_new: float) -> None:
            """Set B_theta at index idx and fix total energy for consistency."""
            B2_old = (U_np[IBR, idx, :] ** 2
                      + U_np[IBZ, idx, :] ** 2
                      + U_np[IBT, idx, :] ** 2)
            U_np[IBT, idx, :] = Bt_new
            B2_new = (U_np[IBR, idx, :] ** 2
                      + U_np[IBZ, idx, :] ** 2
                      + U_np[IBT, idx, :] ** 2)
            # Add magnetic energy difference to total energy
            U_np[IEN, idx, :] += 0.5 * (B2_new - B2_old)
            # Enforce minimum plasma beta
            p_mag = 0.5 * B2_new
            beta_floor = 1e-4
            p_min = beta_floor * np.maximum(p_mag, P_FLOOR)
            E_floor = p_min / (GAMMA - 1.0) + 0.5 * B2_new
            U_np[IEN, idx, :] = np.maximum(U_np[IEN, idx, :], E_floor)

        # Outer ghost cells: fix SI→HL if needed (ghost_pad wrote SI)
        for ig in range(ng):
            out_idx = ng + self.nr + ig
            r_pos = max(r_cell_list[out_idx], 1e-10)
            Bt_val = _MU0 * current / (2.0 * math.pi * r_pos) / _sqrt
            _update_bt_with_energy(out_idx, Bt_val)
            # Density floor in ghost cells
            U_np[IDN, out_idx, :] = np.maximum(U_np[IDN, out_idx, :], 1e-4)

        # Outermost ng interior cells: set B_theta to electrode 1/r profile.
        # This ensures the WENO5-Z stencil sees a smooth B_theta transition
        # rather than a step discontinuity at the ghost boundary.
        for ig in range(ng):
            int_idx = ng + self.nr - 1 - ig
            r_pos = max(r_cell_list[int_idx], 1e-10)
            Bt_val = _MU0 * current / (2.0 * math.pi * r_pos) / _sqrt
            # Blend: use max(existing, electrode) so we don't reduce B_theta
            existing = U_np[IBT, int_idx, :]
            new_Bt = np.where(
                np.abs(existing) > np.abs(Bt_val), existing, Bt_val
            )
            _update_bt_with_energy(int_idx, new_Bt)

        U_padded = mx.array(U_np)

        # Build a lightweight padded grid. CylindricalGrid rejects negative
        # r_inner, so we build geometry arrays directly on the ghost-extended
        # coordinate system, matching what PyTorch MetalMHDSolver does.
        r_cell_mx = mx.array(r_cell_np)
        r_face_list = [r_inner_g + i * dr for i in range(nr_g + 1)]
        r_face_mx = mx.array(np.array(r_face_list, dtype=np.float32))

        # 1/r with L'Hopital at axis (|r| < dr/2 → use 2/dr)
        inv_r_list = [
            2.0 / dr if abs(rc) < 0.5 * dr else 1.0 / max(abs(rc), 1e-30)
            for rc in r_cell_list
        ]
        inv_r_mx = mx.array(np.array(inv_r_list, dtype=np.float32))

        # Cell volumes and face areas
        r_out = r_face_mx[1:]
        r_in = r_face_mx[:-1]
        pi_f32 = mx.array(math.pi, dtype=mx.float32)
        cell_volume = pi_f32 * mx.abs(r_out * r_out - r_in * r_in) * dz
        face_area_r = mx.array(2.0 * math.pi * dz, dtype=mx.float32) * mx.abs(r_face_mx)
        face_area_z = pi_f32 * mx.abs(r_out * r_out - r_in * r_in)

        # z geometry unchanged
        z_cell_mx = self._grid.z_cell

        # Assemble a simple namespace object matching CylindricalGrid API
        class _PaddedGrid:
            pass

        grid_g = _PaddedGrid()
        grid_g.nr = nr_g
        grid_g.nz = self.nz
        grid_g.dr = dr
        grid_g.dz = dz
        grid_g.r_inner = r_inner_g
        grid_g.r_cell = r_cell_mx
        grid_g.r_face = r_face_mx
        grid_g.z_cell = z_cell_mx
        grid_g.inv_r = inv_r_mx
        grid_g.cell_volume = cell_volume
        grid_g.face_area_r = face_area_r
        grid_g.face_area_z = face_area_z

        mx.eval(r_cell_mx, r_face_mx, inv_r_mx, cell_volume, face_area_r, face_area_z)

        return U_padded, grid_g

    @staticmethod
    def _strip_ghost(U: Any, ng: int) -> Any:
        """Strip ghost cells from padded state, returning interior only.

        Parameters
        ----------
        U : mx.array
            Padded state, shape (NVAR, nr + 2*ng, nz).
        ng : int
            Number of ghost cells on each side.

        Returns
        -------
        mx.array
            Interior state, shape (NVAR, nr, nz).
        """
        return U[:, ng:-ng, :]

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

    def _do_braginskii_viscosity(self, U: Any, dt: float) -> Any:
        """Operator-split Braginskii parallel viscosity."""
        from dpf.metal.mlx_viscosity import apply_braginskii_viscosity

        return apply_braginskii_viscosity(
            U, dt, self._grid, self.gamma, self.ion_mass,
            coordinates=self.coordinates,
        )

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
            dr=self._grid.dr,
            Br=Br, Bz=Bz, Bt=Bt,
            anisotropic=True,
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

        # ── 2. Entropy initialised flag ───────────────────────────────────
        # from_state_dict already computes the entropy tracer from p and rho,
        # so no additional work is needed.  Track that we have run at least once.
        self._entropy_initialized = True

        # ── 3. Prepare resistivity for Strang splitting ──────────────
        eta_raw = kwargs.get("eta_field")
        _eta_arg: float | Any | None = None
        if eta_raw is not None:
            if isinstance(eta_raw, np.ndarray):
                eta_squeezed = np.squeeze(eta_raw)
                if eta_squeezed.ndim == 1:
                    eta_squeezed = eta_squeezed.reshape(self._nr, self._nz)
                _eta_arg = mx.array(eta_squeezed.astype(np.float32))
            else:
                _eta_arg = float(eta_raw)

        # ── 3.1. Strang split: first half-step resistive diffusion ─────
        # Must run BEFORE ghost padding (eta field is sized for un-padded grid)
        if _eta_arg is not None:
            U = self._do_resistive_diffusion(U, dt * 0.5, _eta_arg)
            mx.eval(U)

        # ── 3.2. Electrode BC (ghost-cell padding) ────────────────────
        apply_bc = kwargs.get("apply_electrode_bc", False)
        _ghost_active = (
            apply_bc
            and self.coordinates == "cylindrical"
            and abs(current) > 1e-10
        )
        grid_for_rk = self._grid
        if _ghost_active:
            U, grid_for_rk = self._pad_electrode_ghost(U, current)
            mx.eval(U)

        # ── 4. Hyperbolic step ───────────────────────────────────────────
        step_fn = ssp_rk3_step if self._integrator != "ssp_rk2" else ssp_rk2_step
        U = step_fn(
            U, grid_for_rk, dt,
            gamma=self.gamma,
            method=self._method,
            riemann=self._riemann,
            use_dual_energy=self._use_dual_energy,
            ghost_ng=self._GHOST_NG if _ghost_active else 0,
        )
        mx.eval(U)

        # ── 4.1. Strip ghost cells ───────────────────────────────────────
        if _ghost_active:
            U = self._strip_ghost(U, self._GHOST_NG)

        # ── 4.5. div(B) control ───────────────────────────────────────────
        if self._use_ct and self.coordinates == "cylindrical":
            U = self._apply_ct_correction(U, dt)

        if self._enable_dedner or self._enable_powell:
            U = self._apply_divb_cleaning(U, dt)
            mx.eval(U)

        # ── 5. Strang split: second half-step resistive diffusion ──────
        if _eta_arg is not None:
            U = self._do_resistive_diffusion(U, dt * 0.5, _eta_arg)
            mx.eval(U)

        # ── 6. Braginskii conduction ─────────────────────────────────────
        if self.enable_braginskii_conduction:
            kappa = float(kwargs.get("kappa_parallel", 1e3))
            U = self._do_thermal_conduction(U, dt, kappa)
            mx.eval(U)

        # ── 6.5. Braginskii viscosity ──────────────────────────────────
        if self.enable_braginskii_viscosity:
            U = self._do_braginskii_viscosity(U, dt)
            mx.eval(U)

        # ── 6.6. Hall MHD ─────────────────────────────────────────────
        if self.enable_hall and self.coordinates == "cylindrical":
            from dpf.metal.mlx_sources import apply_hall_mhd

            U = apply_hall_mhd(
                U, dt,
                dr=self._grid.dr, dz=self._grid.dz,
                r_cell=self._grid.r_cell,
                ion_mass=self.ion_mass,
            )
            mx.eval(U)

        # ── 7. Unpack ────────────────────────────────────────────────────
        self._U = U
        result = self._state_mgr.to_state_dict(
            U, convert_b_hl_to_si=self._convert_b_si_to_hl
        )

        # ── 7.5. Two-temperature source terms (CPU, matching Metal) ────
        if result.get("e_electron") is not None:
            self._do_two_temperature_sources(result, dt, kwargs.get("eta_field"))

        # ── 8. Coupling — compute Lp from density-weighted Lee formula ──
        self._update_coupling(U, current, voltage, dt)
        return result

    # ------------------------------------------------------------------
    # Constrained transport div(B)=0 correction
    # ------------------------------------------------------------------

    def _apply_ct_correction(self, U: Any, dt: float) -> Any:
        """Apply constrained transport correction to maintain div(B) = 0.

        Approximates the staggered CT update for a cell-centred solver:
        1. Reconstruct face-centred Br/Bz by averaging adjacent cell values.
        2. Compute corner EMF from cell-centred velocities and face B.
        3. Apply CT update to face fields (Gardiner & Stone 2005 §2.3).
        4. Average updated face fields back to cell centres.

        Parameters
        ----------
        U : mx.array
            Conserved state (NVAR, nr, nz), float32.
        dt : float
            Timestep [s].

        Returns
        -------
        mx.array
            U with Br (IBR) and Bz (IBZ) updated to reduce div(B) errors.
        """
        mx = require_mlx()
        from dpf.metal.mlx_ct import apply_ct, compute_emf
        from dpf.metal.mlx_kernels import IBR, IBZ, IDN, IMR, IMZ
        from dpf.metal.mlx_primitives import RHO_FLOOR

        rho = mx.maximum(U[IDN], RHO_FLOOR)
        inv_rho = 1.0 / rho
        vr = U[IMR] * inv_rho   # (nr, nz)
        vz = U[IMZ] * inv_rho   # (nr, nz)
        Br_cc = U[IBR]          # (nr, nz)
        Bz_cc = U[IBZ]          # (nr, nz)

        dr = self._grid.dr
        dz = self._grid.dz
        r_cell = self._grid.r_cell      # (nr,)
        r_face = self._grid.r_face      # (nr+1,)

        # --- Cell-centred → face-centred via averaging ---
        # Br on r-faces: shape (nr+1, nz) — replicate boundary rows
        Br_pad = mx.concatenate([Br_cc[:1, :], Br_cc, Br_cc[-1:, :]], axis=0)  # (nr+2, nz)
        Br_face = 0.5 * (Br_pad[:-1, :] + Br_pad[1:, :])                       # (nr+1, nz)

        # Bz on z-faces: shape (nr, nz+1) — replicate boundary cols
        Bz_pad = mx.concatenate([Bz_cc[:, :1], Bz_cc, Bz_cc[:, -1:]], axis=1)  # (nr, nz+2)
        Bz_face = 0.5 * (Bz_pad[:, :-1] + Bz_pad[:, 1:])                       # (nr, nz+1)

        # --- CT update ---
        emf = compute_emf(vr, vz, Br_face, Bz_face, dr, dz)
        Br_face_new, Bz_face_new = apply_ct(
            Br_face, Bz_face, emf, dt, dr, dz, r_cell, r_face,
        )

        # --- Face-centred → cell-centred by averaging adjacent faces ---
        # Br_cc[i] = 0.5 * (Br_face[i] + Br_face[i+1])
        Br_cc_new = 0.5 * (Br_face_new[:-1, :] + Br_face_new[1:, :])  # (nr, nz)
        # Bz_cc[k] = 0.5 * (Bz_face[k] + Bz_face[k+1])
        Bz_cc_new = 0.5 * (Bz_face_new[:, :-1] + Bz_face_new[:, 1:])  # (nr, nz)

        rows = list(mx.split(U, U.shape[0], axis=0))
        rows[IBR] = Br_cc_new[None]
        rows[IBZ] = Bz_cc_new[None]
        return mx.stack([r[0] for r in rows], axis=0).astype(mx.float32)

    # ------------------------------------------------------------------
    # Dedner GLM + Powell div(B) cleaning
    # ------------------------------------------------------------------

    def _apply_divb_cleaning(self, U: Any, dt: float) -> Any:
        """Operator-split Dedner GLM and/or Powell div(B) correction.

        Dedner evolves a sidecar psi scalar that propagates and damps
        divergence errors. Powell adds source terms proportional to div(B).
        """
        mx = require_mlx()
        from dpf.metal.mlx_divb import dedner_source, powell_source
        from dpf.metal.mlx_primitives import fast_magnetosonic

        # Initialize psi on first call
        spatial_shape = U.shape[1:]
        if self._psi is None:
            self._psi = mx.zeros(spatial_shape, dtype=mx.float32)

        if self._enable_dedner:
            # Compute cleaning speed ch from max fast magnetosonic speed
            rho, vr, vz, vt, p, Br, Bz, Bt = (
                mx.maximum(U[0], 1e-12), U[1] / mx.maximum(U[0], 1e-12),
                U[2] / mx.maximum(U[0], 1e-12), U[3] / mx.maximum(U[0], 1e-12),
                mx.maximum((self.gamma - 1) * (U[4] - 0.5 * U[0] * (
                    (U[1] / mx.maximum(U[0], 1e-12))**2 +
                    (U[2] / mx.maximum(U[0], 1e-12))**2 +
                    (U[3] / mx.maximum(U[0], 1e-12))**2
                ) - 0.5 * (U[6]**2 + U[7]**2 + U[8]**2)), 1e-12),
                U[6], U[7], U[8],
            )
            cf = fast_magnetosonic(rho, p, Br, Bz, Bt, self.gamma, dim=0)
            v_mag = mx.sqrt(vr * vr + vz * vz + vt * vt)
            ch = float(mx.max(v_mag + cf))
            if ch < 1e-10:
                ch = 1.0
            dx_min = min(self.dx, self.dz)
            if self.coordinates == "cartesian":
                dx_min = min(dx_min, self.dy)
            cr = ch / dx_min

            dpsi_dt, dU_dedner = dedner_source(
                self._psi, U, ch, cr, self._grid, self.coordinates,
            )
            U = U + dt * dU_dedner
            self._psi = self._psi + dt * dpsi_dt

        if self._enable_powell:
            dU_powell = powell_source(U, self.gamma, self._grid, self.coordinates)
            U = U + dt * dU_powell

        return U

    # ------------------------------------------------------------------
    # Two-temperature source terms (operator-split, CPU)
    # ------------------------------------------------------------------

    def _do_two_temperature_sources(
        self,
        result: dict[str, np.ndarray],
        dt: float,
        eta_field: float | np.ndarray | None = None,
    ) -> None:
        """Apply electron-ion equilibration, Ohmic heating, bremsstrahlung.

        Modifies result dict in-place. Matches Metal solver pattern at
        metal_solver.py:1580-1604.
        """
        from dpf.fluid.two_temperature import step_electron_energy

        rho = result["rho"]
        n_i = rho / self.ion_mass
        n_i_safe = np.maximum(n_i, 1e-30)
        eta_np = (
            np.asarray(eta_field) if eta_field is not None
            else np.zeros_like(rho)
        )
        # Approximate J^2 from Ohmic heating if available
        J_sq = np.zeros_like(rho)
        e_e_new, Te_new, Ti_new = step_electron_energy(
            rho_e_e=result["e_electron"],
            rho=rho,
            velocity=result["velocity"],
            eta=eta_np,
            J_sq=J_sq,
            Te=result["Te"],
            Ti=result["Ti"],
            n_e=n_i_safe,
            n_i=n_i_safe,
            dx=self.dx,
            dt=dt,
            Z=self.Z_eff,
            gaunt_factor=self.gaunt_factor,
            gamma=self.gamma,
        )
        result["Te"] = np.maximum(Te_new, 1.0)
        result["Ti"] = np.maximum(Ti_new, 1.0)
        result["e_electron"] = e_e_new

    # ------------------------------------------------------------------
    # Plasma inductance from density-weighted Lee formula
    # ------------------------------------------------------------------

    def _update_coupling(
        self, U: Any, current: float, voltage: float, dt: float,
    ) -> None:
        """Compute plasma inductance and update circuit coupling state.

        Uses the Lee formula: Lp = (mu0/2pi) * z_sheath * ln(b/r_eff)
        where r_eff is the density-weighted effective radius.

        References: Lee & Saw, Phys. Plasmas 21, 072501 (2014).
        """
        from dpf.metal.mlx_kernels import IDN

        # Plasma inductance only meaningful for cylindrical DPF geometry
        if self.coordinates == "cartesian":
            self._coupling = CouplingState(current=current, voltage=voltage)
            return

        rho_np = np.asarray(U[IDN])  # (nr, nz)
        nr, nz = rho_np.shape
        dr = self._grid.dr
        dz = self._grid.dz
        r_arr = self._r_inner + (np.arange(nr) + 0.5) * dr

        # Sheath position from column density peak
        col_density = np.sum(rho_np * r_arr[:, np.newaxis], axis=0) * dr
        iz_sheath = int(np.argmax(col_density))
        z_sheath = (iz_sheath + 0.5) * dz

        # Density-weighted effective radius
        rho_region = rho_np[:, : iz_sheath + 1]
        r_col = r_arr[:, np.newaxis]
        dV = 2.0 * math.pi * r_col * dr * dz
        mass = rho_region * dV
        total_mass = float(np.sum(mass))
        if total_mass > 0:
            r_eff = float(np.sum(r_col * mass) / total_mass)
        else:
            r_eff = 0.5 * self._cathode_radius
        r_eff = max(r_eff, 1e-6)
        r_eff = min(r_eff, self._cathode_radius * 0.999)

        # Lee formula
        if r_eff > 0 and z_sheath > 0:
            Lp = (_MU0 / (2.0 * math.pi)) * z_sheath * math.log(
                self._cathode_radius / r_eff
            )
        else:
            Lp = 0.0

        # Monotonicity enforcement (Lp can't decrease during compression)
        if Lp > self._Lp_max:
            self._Lp_max = Lp
        else:
            Lp = self._Lp_max

        # dL/dt via backward difference
        dL_dt: float | None = None
        if self._prev_Lp > 0 and dt > 0:
            dL_dt = (Lp - self._prev_Lp) / dt
        self._prev_Lp = Lp

        self._coupling = CouplingState(
            Lp=Lp, current=current, voltage=voltage, dL_dt=dL_dt,
        )

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
