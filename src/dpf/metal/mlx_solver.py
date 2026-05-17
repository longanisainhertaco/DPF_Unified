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
from types import SimpleNamespace
from typing import Any

import numpy as np

from dpf.core.bases import CouplingState, PlasmaSolverBase
from dpf.metal.mlx_device import HAS_MLX, require_mlx

logger = logging.getLogger(__name__)

_MU0: float = 4.0 * math.pi * 1e-7
_SQRT_MU0: float = math.sqrt(_MU0)
from dpf.metal.constants import K_B as _K_B  # noqa: E402

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
        Enable the cylindrical Hall-MHD source update in the MLX operator split.
    enable_braginskii_conduction : bool
        Enable operator-split Braginskii parallel thermal conduction.
    enable_braginskii_viscosity : bool
        Enable the Braginskii viscosity operator split.
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
        riemann_solver: str = "hll",  # HLL-GPU: conservative energy flux (V&V RIE-01)
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
        species_config: dict | None = None,
        amr_config: Any = None,
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

        # Density floor: prevents vacuum Alfven speed → ∞.
        # Set to ~1e-4 * fill density from engine, or 1e-6 absolute minimum.
        self._rho_floor: float = max(float(kwargs.get("rho_floor", 1e-6)), 1e-6)
        # Fill gas density for vacuum/dense cell classification.
        self._rho_fill: float = float(kwargs.get("rho_fill", self._rho_floor * 1e4))

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

        # HLL-GPU is default: uses entropy wavespeeds (no float32 cancellation)
        # with conservative energy flux (U[IEN], not entropy-derived E_tot).
        # HLLS trades conservation for robustness — use explicitly when needed.
        # HLLD adds contact/Alfven resolution but needs Boris (Task 2.3).
        self._riemann: str = riemann_solver if riemann_solver in ("hlld", "hll", "hlls") else "hll"
        self._precision: str = precision if precision in ("float32", "float64") else "float32"
        self._integrator: str = time_integrator

        # Circuit coupling state — updated each step
        self._coupling: CouplingState = CouplingState()
        self.total_radiated_energy: float = 0.0
        self._prev_Lp: float = 0.0
        self._Lp_max: float = 0.0
        self._Lp_history: list[tuple[float, float]] = []
        self._sim_time: float = 0.0
        self._cathode_radius: float = float(kwargs.get("cathode_radius", 0.025))
        self.config_two_temperature: bool = kwargs.get("two_temperature", False)
        self._resistivity_model: str = str(kwargs.get("resistivity_model", "constant"))
        self._anomalous_resistivity_model: str | None = kwargs.get("anomalous_resistivity")
        self._use_rkl2_transport: bool = bool(kwargs.get("use_rkl2_transport", True))

        # Saha EOS: variable Z_bar(T) instead of hardcoded Z=1
        self._enable_saha_eos: bool = bool(kwargs.get("enable_saha_eos", False))
        self._saha_eos = None
        if self._enable_saha_eos:
            from dpf.metal.mlx_eos import SahaEOS
            n_e_ref = float(kwargs.get("saha_n_e_ref", 1e22))
            self._saha_eos = SahaEOS(n_e_ref=n_e_ref, ion_mass=self.ion_mass)
            logger.info("Saha EOS enabled (n_e_ref=%.2e)", n_e_ref)

        # AMR: block-structured adaptive mesh refinement
        self._enable_amr: bool = bool(kwargs.get("enable_amr", False))
        self._amr_hierarchy: Any = None
        self._amr_rhs_fn: Any = None
        self._amr_block_nr: int = int(kwargs.get("amr_block_nr", max(nr // 2, 8)))
        self._amr_block_nz: int = int(kwargs.get("amr_block_nz", max(nz // 2, 8)))
        self._amr_ratio: int = int(kwargs.get("amr_ratio", 2))
        self._amr_refined_blocks: list | None = kwargs.get("amr_refined_blocks")
        if self._enable_amr:
            from dpf.metal.mlx_amr import build_amr_hierarchy, make_mlx_block_rhs
            self._amr_hierarchy = build_amr_hierarchy(
                nr, nz, self.dx, self.dz, self._r_inner,
                self._amr_block_nr, self._amr_block_nz, self._amr_ratio,
                refined_blocks=self._amr_refined_blocks,
            )
            self._amr_rhs_fn = make_mlx_block_rhs(coordinates=self.coordinates)
            logger.info(
                "AMR enabled: %d blocks (L0), ratio=%d, block=%dx%d",
                len(self._amr_hierarchy.levels[0].blocks),
                self._amr_ratio, self._amr_block_nr, self._amr_block_nz,
            )

        # Internal conserved state (mx.array, set after first step)
        self._U: Any = None
        self._entropy_initialized: bool = False
        self._psi: Any = None  # Dedner cleaning scalar (sidecar, not in U)

        # Multi-species impurity tracking (sidecar, not in U)
        self._species_mgr: Any = None
        self._Y: Any = None  # evolved species fractions (N_evolved, nr, nz)
        if species_config is not None:
            from dpf.metal.mlx_species import SpeciesManager
            self._species_mgr = SpeciesManager(**species_config)
            self._Y = self._species_mgr.init_mass_fractions(nr, nz)

        # Legacy AMR config (Phase A) — does NOT overwrite new enable_amr path
        self._amr_config: Any = amr_config
        if not self._enable_amr:
            self._amr_hierarchy = None  # only reset if not already built above
        self._step_count: int = 0

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

    @staticmethod
    def _electrode_bt_fixup_mlx(
        U_padded: Any,
        r_cell: Any,
        current: float,
        ng: int,
        nr_phys: int,
        convert_si_to_hl: bool = True,
        z_sheath_frac: float = 1.0,
    ) -> Any:
        """Apply electrode B_theta BC with z-dependent sheath masking.

        Ou Haibin et al. (2024): B_theta only behind sheath (low resistivity region).
        Sun et al. (2025): B at inlet boundary, conducting wall on electrodes.
        [UNVERIFIED: Sun et al. 2025 (Acta Physica Sinica 74:115201) — paper
        not on disk as of 2026-04-24. Implementation behavior kept but citation
        cannot be verified.]
        """
        from dpf.metal.mlx_bc import electrode_bt_fixup_mlx
        return electrode_bt_fixup_mlx(
            U_padded, r_cell, current, ng, nr_phys, convert_si_to_hl,
            z_sheath_frac=z_sheath_frac,
        )

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

        # Ghost cell padding.
        # When inlet BC is active (Sun 2025), use "outflow" (zero-gradient) at
        # cathode instead of "electrode" (Dirichlet B_theta). The cathode is a
        # conducting wall: dB/dn = 0. B_theta injection is at z=0 inlet only.
        # UNVERIFIED: Sun et al. 2025 (Acta Physica Sinica 74:115201) — paper
        # not on disk as of 2026-04-24. Implementation behavior kept but
        # citation cannot be verified.
        # Sun et al. (2025) Eq. 18: dB/dn = 0 on electrode surfaces.
        _use_inlet_bc = getattr(self, "_inlet_bc_active", False)
        _bc_type = "outflow" if _use_inlet_bc else "electrode"
        U_padded = ghost_pad_mlx(
            U, ng, _bc_type,
            current=current if not _use_inlet_bc else 0.0,
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
        # Apply electrode B_theta fixup ONLY when using cathode Dirichlet BC
        # (legacy mode). When inlet BC is active, cathode is conducting wall
        # (zero-gradient from ghost_pad), no fixup needed.
        r_cell_mx_g = mx.array(r_cell_np)
        if _use_inlet_bc:
            pass  # conducting wall — ghost cells already set by ghost_pad_mlx("outflow")
        else:
            try:
                U_padded = self._electrode_bt_fixup_mlx(
                    U_padded,
                    r_cell_mx_g,
                    current,
                    ng,
                    self.nr,
                    convert_si_to_hl=self._convert_b_si_to_hl,
                    z_sheath_frac=getattr(self, "_z_sheath_frac", 1.0),
                )
            except (RuntimeError, IndexError, ValueError, TypeError) as exc:
                logger.warning("MLX electrode BC failed (%s), using NumPy fallback", exc)
                from dpf.metal.mlx_kernels import GAMMA, IBR, IBT, IBZ, IDN, IEN, P_FLOOR
                _sqrt = _SQRT_MU0 if self._convert_b_si_to_hl else 1.0
                U_np = np.asarray(U_padded)

                def _update_bt_with_energy(cell_idx: int, Bt_new: float) -> None:
                    B2_old = (U_np[IBR, cell_idx, :] ** 2
                              + U_np[IBZ, cell_idx, :] ** 2
                              + U_np[IBT, cell_idx, :] ** 2)
                    U_np[IBT, cell_idx, :] = Bt_new
                    B2_new = (U_np[IBR, cell_idx, :] ** 2
                              + U_np[IBZ, cell_idx, :] ** 2
                              + U_np[IBT, cell_idx, :] ** 2)
                    U_np[IEN, cell_idx, :] += 0.5 * (B2_new - B2_old)
                    p_mag = 0.5 * B2_new
                    beta_floor = 1e-4
                    p_min = beta_floor * np.maximum(p_mag, P_FLOOR)
                    E_floor = p_min / (GAMMA - 1.0) + 0.5 * B2_new
                    U_np[IEN, cell_idx, :] = np.maximum(U_np[IEN, cell_idx, :], E_floor)

                for ig in range(ng):
                    out_idx = ng + self.nr + ig
                    r_pos = max(r_cell_list[out_idx], 1e-10)
                    Bt_val = _MU0 * current / (2.0 * math.pi * r_pos) / _sqrt
                    _update_bt_with_energy(out_idx, Bt_val)
                    U_np[IDN, out_idx, :] = np.maximum(U_np[IDN, out_idx, :], 1e-4)

                for ig in range(ng):
                    int_idx = ng + self.nr - 1 - ig
                    r_pos = max(r_cell_list[int_idx], 1e-10)
                    Bt_val = _MU0 * current / (2.0 * math.pi * r_pos) / _sqrt
                    existing = U_np[IBT, int_idx, :]
                    new_Bt = np.where(np.abs(existing) > np.abs(Bt_val), existing, Bt_val)
                    _update_bt_with_energy(int_idx, new_Bt)

                U_padded = mx.array(U_np)

        # Beresnyak velocity prescription in outer ghost cells.
        # verif_r.cpp line 242: v_ghost = v_interior + (v_interior/r - curr_rate) * delta_r
        # This ensures E = -v×B is consistent with dB/dt, preventing NaN.
        curr_rate = getattr(self, "_beresnyak_curr_rate", 0.0)
        if abs(curr_rate) > 0 and not _use_inlet_bc:
            U_np = np.asarray(U_padded)
            from dpf.metal.constants import IDN, IMR
            for ig in range(ng):
                out_idx = ng + self.nr + ig
                int_idx = ng + self.nr - 1  # last interior cell
                r_int = max(r_cell_list[int_idx], 1e-10)
                r_out = max(r_cell_list[out_idx], 1e-10)
                delta_r = r_out - r_int
                # Interior velocity (radial momentum / density)
                rho_int = np.maximum(U_np[IDN, int_idx, :], 1e-30)
                vr_int = U_np[IMR, int_idx, :] / rho_int
                # Beresnyak extrapolation with dI/dt correction
                vr_ghost = vr_int + (vr_int / r_int - curr_rate) * delta_r
                # Set ghost cell momentum = rho_ghost * vr_ghost
                rho_ghost = U_np[IDN, out_idx, :]
                U_np[IMR, out_idx, :] = rho_ghost * vr_ghost
                # Energy: B-field contribution already applied by electrode fixup above.
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

        grid_g = SimpleNamespace(
            nr=nr_g,
            nz=self.nz,
            dr=dr,
            dz=dz,
            r_inner=r_inner_g,
            r_cell=r_cell_mx,
            r_face=r_face_mx,
            z_cell=z_cell_mx,
            inv_r=inv_r_mx,
            cell_volume=cell_volume,
            face_area_r=face_area_r,
            face_area_z=face_area_z,
        )

        mx.eval(r_cell_mx, r_face_mx, inv_r_mx, cell_volume, face_area_r, face_area_z)

        return U_padded, grid_g

    @staticmethod
    def _strip_ghost(U: Any, ng: int) -> Any:
        """Strip ghost cells from padded state. Delegated to mlx_bc.py."""
        from dpf.metal.mlx_bc import strip_ghost
        return strip_ghost(U, ng)

    # ------------------------------------------------------------------
    # Operator-split: resistive diffusion
    # ------------------------------------------------------------------

    def _do_resistive_diffusion(self, U: Any, dt: float, eta: Any) -> Any:
        """Implicit resistive diffusion of the B-field with Ohmic heating."""
        from dpf.metal.mlx_operator_split import do_resistive_diffusion
        return do_resistive_diffusion(U, dt, eta, self._grid, self.gamma)

    def _do_resistive_diffusion_rkl2(self, U: Any, dt: float, eta: Any) -> Any:
        """RKL2 super-timestepped resistive diffusion -- fully on GPU."""
        from dpf.metal.mlx_operator_split import do_resistive_diffusion_rkl2
        return do_resistive_diffusion_rkl2(
            U, dt, eta, self._grid, self.gamma, self.coordinates,
        )

    # ------------------------------------------------------------------
    # Operator-split: Braginskii thermal conduction
    # ------------------------------------------------------------------

    def _do_braginskii_viscosity(self, U: Any, dt: float) -> Any:
        """Operator-split Braginskii parallel viscosity."""
        from dpf.metal.mlx_operator_split import do_braginskii_viscosity
        return do_braginskii_viscosity(
            U, dt, self._grid, self.gamma, self.ion_mass, self.coordinates,
        )

    def _do_thermal_conduction(self, U: Any, dt: float, kappa: float | Any) -> Any:
        """Implicit Braginskii parallel conduction along z."""
        from dpf.metal.mlx_operator_split import do_thermal_conduction
        return do_thermal_conduction(
            U, dt, kappa, self._grid, self.gamma, self.ion_mass,
            self.coordinates,
        )

    def _do_thermal_conduction_rkl2(self, U: Any, dt: float, kappa: float | Any) -> Any:
        """RKL2 super-timestepped thermal conduction -- fully on GPU."""
        from dpf.metal.mlx_operator_split import do_thermal_conduction_rkl2
        return do_thermal_conduction_rkl2(
            U, dt, kappa, self._grid, self.gamma, self.ion_mass,
            self.coordinates,
        )

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
        from dpf.metal.constants import C_BORIS

        U = self._state_mgr.from_state_dict(
            state, convert_b_si_to_hl=self._convert_b_si_to_hl
        )
        mx.eval(U)
        return compute_dt_cfl(
            U,
            self._grid,
            gamma=self.gamma,
            cfl=self.cfl,
            use_boris=True,
            c_boris=C_BORIS,
            enable_hall=self.enable_hall,
        )

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
        self._step_count += 1

        # ── AMR dispatch ─────────────────────────────────────────────────
        if self._amr_config is not None and getattr(self._amr_config, "enabled", False):
            return self._step_amr(state, dt, current, voltage, **kwargs)

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
                    eta_squeezed = eta_squeezed.reshape(self.nr, self.nz)
                _eta_arg = mx.array(eta_squeezed.astype(np.float32))
            else:
                _eta_arg = float(eta_raw)
        elif self._resistivity_model != "constant":
            # Compute spatially-varying resistivity from plasma conditions
            from dpf.metal.mlx_primitives import cons_to_prim
            from dpf.metal.mlx_transport import compute_resistivity
            rho_tmp, _, _, _, p_tmp, _, _, _ = cons_to_prim(U, self.gamma)
            rho_np = np.asarray(rho_tmp, dtype=np.float64)
            p_np_tmp = np.asarray(p_tmp, dtype=np.float64)
            Te_eV = self._temperature_eV(rho_np, p_np_tmp)
            # Compute J^2 for anomalous resistivity if enabled
            J_sq_np = None
            p_np = None
            if self._anomalous_resistivity_model is not None:
                from dpf.metal.mlx_sources import compute_current_density_si
                # KR Sun 2025 §2.2 Eq. 10: Ohm's law E = ηJ; SI Ampere
                # j_SI = curl(B_SI)/mu_0 (KR plasma-formulary p.18).
                # Athena HL convention (Stone 2008 §2): B_HL = B_SI/sqrt(mu_0)
                #   -> |J_SI|^2 = |curl(B_HL)|^2 / mu_0
                # Single-source-of-truth conversion lives in
                # compute_current_density_si (fix-anom-mu0-retry owns the sign).
                # b_packed_as_hl follows _convert_b_si_to_hl.
                J_sq_mx = compute_current_density_si(
                    U, self._grid.dr, self._grid.dz, self._grid.r_cell,
                    b_packed_as_hl=self._convert_b_si_to_hl,
                )
                J_sq_np = np.asarray(J_sq_mx, dtype=np.float64)
                p_np = np.asarray(p_tmp, dtype=np.float64)

            eta_computed = compute_resistivity(
                Te_eV, rho_np, model=self._resistivity_model,
                Z_eff=self.Z_eff, ion_mass=self.ion_mass,
                J_sq=J_sq_np, p=p_np,
                anomalous_model=self._anomalous_resistivity_model,
            )
            _eta_arg = mx.array(eta_computed.astype(np.float32))

        # ── 3.1. Strang split: first half-step resistive diffusion ─────
        # Must run BEFORE ghost padding (eta field is sized for un-padded grid)
        if _eta_arg is not None:
            if self._use_rkl2_transport:
                U = self._do_resistive_diffusion_rkl2(U, dt * 0.5, _eta_arg)
            else:
                U = self._do_resistive_diffusion(U, dt * 0.5, _eta_arg)

        # ── 3.2. Electrode BC ────────────────────────────────────────
        apply_bc = kwargs.get("apply_electrode_bc", False)
        _ghost_active = (
            apply_bc
            and self.coordinates == "cylindrical"
            and abs(current) > 1e-10
        )
        grid_for_rk = self._grid
        if _ghost_active:
            # Beresnyak et al. (2022), Phys. Plasmas 29:052712, verif_r.cpp:
            # Ghost zone BC with BOTH B_theta AND velocity prescribed.
            # B_theta = mu0*I/(2*pi*r) in ghost cells (Ampere's law).
            # Velocity = extrapolated from interior + dI/dt gradient correction:
            #   v_ghost = v_interior + (v_interior/r - curr_rate) * delta_r
            # where curr_rate = (1/I)(dI/dt).
            # This ensures E = -v×B is consistent with dB/dt from circuit,
            # preventing the NaN from E-field mismatch at the boundary.
            # The velocity prescription is the KEY — without it, v=0 in ghost
            # cells next to B=1500 creates massive E-field discontinuity.
            self._inlet_bc_active = False  # Dirichlet B + velocity in ghosts
            # Store curr_rate for the ghost BC
            self._beresnyak_curr_rate = kwargs.get("curr_rate", 0.0)
            U, grid_for_rk = self._pad_electrode_ghost(U, current)

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

        # ── 4.2. Density and pressure floor ─────────────────────────────
        # Every production MHD code (Athena++, FLASH, Pluto) enforces
        # conservative floors after the hyperbolic update. Without this,
        # vacuum cells reach rho~1e-12, producing v_Alfven→∞ and CFL
        # collapse. Beresnyak (2022) uses rho_min as an explicit parameter.
        # Our floor: 1e-4 * fill density, or 1e-6 kg/m^3 absolute minimum.
        from dpf.metal.constants import (
            IBR,
            IBT,
            IBZ,
            IDN,
            IEN,
            IMR,
            IMT,
            IMZ,
            ISR,
            P_FLOOR,
        )
        _rho_floor = max(getattr(self, "_rho_floor", 1e-6), 1e-6)
        _f32_limit = np.finfo(np.float32).max / 16.0
        _velocity_overflow_limit = 5.0e7
        U_np = np.asarray(U).astype(np.float64, copy=True)
        rho_raw = U_np[IDN]
        rho_valid = np.isfinite(rho_raw) & (rho_raw > 0.0)
        rho_old = np.where(rho_valid, rho_raw, _rho_floor)
        rho_new = np.maximum(rho_old, _rho_floor)
        needs_floor = (~rho_valid) | (rho_raw < _rho_floor)

        # Recover velocity from the old state, clamp it, then rebuild momentum
        # from the floored density.  Multiplying momentum by rho_floor/rho can
        # overflow when a vacuum cell reaches tiny or invalid density.
        rho_for_velocity = np.maximum(rho_old, _rho_floor)
        for idx in (IMR, IMZ, IMT):
            momentum = np.nan_to_num(U_np[idx], nan=0.0, posinf=0.0, neginf=0.0)
            velocity = momentum / rho_for_velocity
            velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            velocity = np.clip(velocity, -_velocity_overflow_limit, _velocity_overflow_limit)
            U_np[idx] = rho_new * velocity

        U_np[IDN] = rho_new
        U_np[ISR] = np.maximum(
            np.nan_to_num(U_np[ISR], nan=0.0, posinf=_f32_limit, neginf=0.0),
            0.0,
        )
        for idx in (IBR, IBZ, IBT):
            U_np[idx] = np.nan_to_num(U_np[idx], nan=0.0, posinf=0.0, neginf=0.0)

        # Floor total energy: E >= p_floor/(gamma-1) + KE + B^2/2
        KE = 0.5 * (U_np[IMR] ** 2 + U_np[IMZ] ** 2 + U_np[IMT] ** 2) / U_np[IDN]
        B2 = U_np[IBR] ** 2 + U_np[IBZ] ** 2 + U_np[IBT] ** 2
        _p_floor = max(P_FLOOR, 1e-8)  # Pa — cold but not singular
        E_floor = _p_floor / (self.gamma - 1.0) + KE + 0.5 * B2
        E_current = np.nan_to_num(
            U_np[IEN],
            nan=0.0,
            posinf=_f32_limit,
            neginf=0.0,
        )
        bad_energy = (~np.isfinite(U_np[IEN])) | (E_current < E_floor)
        U_np[IEN] = np.where(needs_floor | bad_energy, np.maximum(E_current, E_floor), E_current)
        U_np = np.clip(U_np, -_f32_limit, _f32_limit)
        U = mx.array(U_np.astype(np.float32))

        # ── 4.5. div(B) control ───────────────────────────────────────────
        if self._use_ct and self.coordinates == "cylindrical":
            U = self._apply_ct_correction(U, dt)

        if self._enable_dedner or self._enable_powell:
            U = self._apply_divb_cleaning(U, dt)

        # ── 5. Strang split: second half-step resistive diffusion ──────
        if _eta_arg is not None:
            if self._use_rkl2_transport:
                U = self._do_resistive_diffusion_rkl2(U, dt * 0.5, _eta_arg)
            else:
                U = self._do_resistive_diffusion(U, dt * 0.5, _eta_arg)

        # ── 6. Braginskii conduction ─────────────────────────────────────
        if self.enable_braginskii_conduction:
            kappa = float(kwargs.get("kappa_parallel", 1e3))
            if self._use_rkl2_transport:
                U = self._do_thermal_conduction_rkl2(U, dt, kappa)
            else:
                U = self._do_thermal_conduction(U, dt, kappa)

        # ── 6.5. Braginskii viscosity ──────────────────────────────────
        if self.enable_braginskii_viscosity:
            U = self._do_braginskii_viscosity(U, dt)

        # ── 6.6. Hall MHD ─────────────────────────────────────────────
        if self.enable_hall and self.coordinates == "cylindrical":
            from dpf.metal.mlx_sources import apply_hall_mhd

            U = apply_hall_mhd(
                U, dt,
                dr=self._grid.dr, dz=self._grid.dz,
                r_cell=self._grid.r_cell,
                ion_mass=self.ion_mass,
            )

        # ── 6.65b. Line radiation (multi-species) ──────────────────────
        if (
            self._species_mgr is not None
            and self._Y is not None
            and self.enable_bremsstrahlung
        ):
            from dpf.metal.mlx_line_radiation import apply_line_radiation_mlx

            U = apply_line_radiation_mlx(
                U, dt,
                species_Z=self._species_mgr.Z,
                species_Y=self._Y,
                gamma=self.gamma,
                ion_mass=self.ion_mass,
            )

        # Single eval after the entire operator-split physics block
        mx.eval(U)

        # ── 6.65. PIC kinetic current feedback ─────────────────────────
        # PIC deposits J_kin on the grid. The resistive E-field from PIC current
        # modifies B via Faraday's law: dB/dt = -curl(eta * J_kin).
        # For the MLX solver, this adds to the Ohmic heating without modifying
        # the induction equation directly (operator-split simplification).
        source_terms = kwargs.get("source_terms")
        if source_terms is not None and "J_kin" in source_terms:
            from dpf.metal.mlx_kernels import IEN
            J_kin_np = source_terms["J_kin"]  # (3, nr, nz) or (3, nr, 1, nz)
            if J_kin_np.ndim == 4:
                J_kin_np = J_kin_np[:, :, 0, :]
            J_sq_pic = np.sum(J_kin_np**2, axis=0)
            eta_pic = float(kwargs.get("eta_field", 1e-6)) if not isinstance(
                kwargs.get("eta_field"), np.ndarray
            ) else 1e-6
            # Ohmic heating from PIC current: Q = eta * J_pic^2
            Q_pic = eta_pic * J_sq_pic * dt
            U = U.at[IEN].add(mx.array(Q_pic.astype(np.float32)))

        # ── 6.7. Species advection + ablation sources ─────────────────
        if self._species_mgr is not None and self._Y is not None:
            from dpf.metal.mlx_species import (
                apply_ablation_sources,
                species_advection_step,
            )

            self._Y = species_advection_step(
                self._Y, U,
                dr=self._grid.dr, dz=self._grid.dz, dt=dt,
                gamma=self.gamma,
                r_cell=getattr(self._grid, "r_cell", None),
                r_face=getattr(self._grid, "r_face", None),
            )

            ablation_rate = kwargs.get("ablation_rate")
            if ablation_rate is not None:
                self._Y = apply_ablation_sources(
                    self._Y, dt,
                    mx.array(np.asarray(ablation_rate, dtype=np.float32)),
                    cu_idx=0,
                )
            mx.eval(self._Y)

        # ── 6.9. Z-dependent vacuum B_theta prescription ──────────────
        # Prescribe B_theta = mu0*I/(2*pi*r) only in vacuum cells that are:
        #   (a) BEHIND the sheath (z < z_sheath) — ahead, B should be zero
        #   (b) FAR from dense cells (buffer zone) — near sheath, MHD evolves B
        # This preserves B-field gradients near the sheath boundary, giving
        # nonzero J = curl(B) for Auluck (2021) Poynting coupling.
        #
        # Beresnyak (2022) verif_r.cpp:135: B in vacuum behind shell.
        # Auluck (2021) Eq. (1): V = -(1/I)*integral(J·E d³r) requires J≠0.
        if (
            abs(current) > 1e-10
            and self.coordinates == "cylindrical"
            and _ghost_active
        ):
            import math
            _MU0_LOCAL = 4.0 * math.pi * 1e-7
            _SQRT_MU0_LOCAL = math.sqrt(_MU0_LOCAL)
            U_np = np.asarray(U).astype(np.float64, copy=True)
            U_np[IDN] = np.maximum(
                np.nan_to_num(U_np[IDN], nan=_rho_floor, posinf=_rho_floor, neginf=_rho_floor),
                _rho_floor,
            )
            for idx in (IMR, IMZ, IMT, IBR, IBZ, IBT):
                U_np[idx] = np.nan_to_num(U_np[idx], nan=0.0, posinf=0.0, neginf=0.0)
            U_np[IEN] = np.nan_to_num(U_np[IEN], nan=0.0, posinf=_f32_limit, neginf=0.0)
            rho = U_np[IDN]
            nr_local, nz_local = rho.shape
            _rho_fill = self._rho_fill
            r_cells = self._r_inner + (np.arange(nr_local) + 0.5) * self._grid.dr
            r_safe = np.maximum(r_cells, 1e-10)

            # --- Z-sheath detection from B_theta radial integral ---
            # Behind sheath: Phi_radial ~ Phi_vacuum. Ahead: Phi_radial ~ 0.
            _div = _SQRT_MU0_LOCAL if self._convert_b_si_to_hl else 1.0
            Bt_si = U_np[IBT] * (_SQRT_MU0_LOCAL if self._convert_b_si_to_hl else 1.0)
            Phi_radial = np.sum(Bt_si * self._grid.dr, axis=0)  # (nz,)
            Phi_vacuum = (_MU0_LOCAL * current / (2.0 * math.pi)) * math.log(
                max(r_safe[-1] / max(r_safe[0], 1e-10), 1.01)
            )
            above = np.abs(Phi_radial) > 0.3 * abs(Phi_vacuum)
            iz_sheath = int(np.max(np.where(above)[0])) + 1 if np.any(above) else nz_local

            # --- Buffer zone: exclude N cells around dense regions ---
            dense = rho > 3.0 * _rho_fill
            _N_BUFFER = 3
            if np.any(dense):
                try:
                    from scipy.ndimage import binary_dilation
                    near_dense = binary_dilation(dense, iterations=_N_BUFFER)
                except ImportError:
                    # Fallback: simple dilation via max-filter
                    near_dense = dense.copy()
                    for _ in range(_N_BUFFER):
                        padded = np.pad(near_dense, 1, mode='constant', constant_values=False)
                        near_dense = (
                            padded[:-2, 1:-1] | padded[2:, 1:-1] |
                            padded[1:-1, :-2] | padded[1:-1, 2:] | near_dense
                        )
            else:
                near_dense = np.zeros_like(dense)

            # --- Prescription mask: vacuum AND behind sheath AND far from dense ---
            vacuum = rho < 3.0 * _rho_fill
            prescribe = vacuum & (~near_dense)
            prescribe[:, iz_sheath:] = False  # don't prescribe ahead of sheath

            if np.any(prescribe):
                Bt_ampere = (_MU0_LOCAL * current / (2.0 * math.pi)) / _div
                Bt_ampere_2d = Bt_ampere / r_safe[:, None] * np.ones_like(rho)
                B2_old = U_np[IBR] ** 2 + U_np[IBZ] ** 2 + U_np[IBT] ** 2
                Bt_new = np.where(prescribe, Bt_ampere_2d, U_np[IBT])
                B2_new = U_np[IBR] ** 2 + U_np[IBZ] ** 2 + Bt_new ** 2
                delta_B2 = np.nan_to_num(B2_new - B2_old, nan=0.0, posinf=0.0, neginf=0.0)
                U_np[IEN] += np.where(prescribe, 0.5 * delta_B2, 0.0)
                U_np[IBT] = Bt_new
                KE = 0.5 * (
                    U_np[IMR] ** 2 + U_np[IMZ] ** 2 + U_np[IMT] ** 2
                ) / np.maximum(U_np[IDN], _rho_floor)
                e_int = U_np[IEN] - KE - 0.5 * B2_new
                bad_p = e_int < 1e-8
                if np.any(bad_p & prescribe):
                    p_min_vac = 1e-4 * 0.5 * B2_new
                    E_floor_vac = p_min_vac / (self.gamma - 1.0) + KE + 0.5 * B2_new
                    U_np[IEN] = np.where(bad_p & prescribe, np.maximum(U_np[IEN], E_floor_vac), U_np[IEN])
                U_np = np.clip(U_np, -_f32_limit, _f32_limit)
                U = mx.array(U_np.astype(np.float32))

        # ── 7. Unpack ────────────────────────────────────────────────────
        self._U = U
        result = self._state_mgr.to_state_dict(
            U, convert_b_hl_to_si=self._convert_b_si_to_hl
        )

        # ── 7.5. Two-temperature source terms (CPU, matching Metal) ────
        if result.get("e_electron") is not None:
            self._do_two_temperature_sources(result, dt, kwargs.get("eta_field"))

        # ── 7.6. Species state in result ──────────────────────────────
        if self._species_mgr is not None and self._Y is not None:
            Y_full = self._species_mgr.recover_background(self._Y)
            result["species"] = {
                name: np.asarray(Y_full[i])
                for i, name in enumerate(self._species_mgr.species)
            }

        # ── 8. Coupling — compute Lp from density-weighted Lee formula ──
        self._update_coupling(U, current, voltage, dt)
        return result

    # ------------------------------------------------------------------
    # AMR step
    # ------------------------------------------------------------------

    def _step_amr(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """AMR-mode timestep: decompose -> advance blocks -> reassemble."""
        from dpf.metal.mlx_amr import (
            amr_step,
            assemble_global_state,
            build_amr_hierarchy,
            populate_blocks_from_state,
        )

        self._ensure_internals()
        cfg = self._amr_config

        # Build hierarchy on first call
        if self._amr_hierarchy is None:
            U_init = self._state_mgr.from_state_dict(
                state, convert_b_si_to_hl=self._convert_b_si_to_hl
            )
            self._amr_hierarchy = build_amr_hierarchy(
                nr=self.nr,
                nz=self.nz,
                dr=self.dx,
                dz=self.dz,
                r_inner=self._r_inner,
                block_nr=cfg.block_nr,
                block_nz=cfg.block_nz,
                ratio=cfg.refinement_ratio,
                refined_blocks=cfg.refined_blocks,
            )
            populate_blocks_from_state(
                self._amr_hierarchy.levels[0], U_init, cfg.block_nr, cfg.block_nz
            )

        # Advance AMR hierarchy
        self._amr_hierarchy, dt_used = amr_step(
            hierarchy=self._amr_hierarchy,
            dt=dt,
            gamma=self.gamma,
            method=self._method,
            riemann=self._riemann,
            ng=self._GHOST_NG,
            current=current,
            r_inner=self._r_inner,
            step_number=self._step_count,
            rhs_fn=self._amr_rhs_wrapper,
            use_refluxing=getattr(cfg, "use_refluxing", True),
        )

        # Reassemble global state from level 0
        U_global = assemble_global_state(
            self._amr_hierarchy.levels[0], self.nr, self.nz, cfg.block_nr, cfg.block_nz
        )
        result = self._state_mgr.to_state_dict(
            U_global, convert_b_hl_to_si=self._convert_b_si_to_hl
        )
        self._update_coupling(U_global, current, voltage, dt)
        return result

    @property
    def _amr_rhs_wrapper(self):
        """RHS function wrapper for AMR blocks — includes full operator-split physics."""
        from dpf.metal.mlx_riemann import mhd_rhs

        def _rhs(U, grid, dt=None):
            return mhd_rhs(
                U, grid, gamma=self.gamma,
                method=self._method, riemann=self._riemann,
                precision=self._precision,
            )
        return _rhs

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
            # Mignone & Tzeferacos (2010), JCP 229:5896, p.8, Eq. (27):
            #   alpha = dh * ch / cp^2, dimensionless, alpha in [0,1].
            #   cr = ch^2/cp^2, so alpha = dx * cr / ch.
            # M&T reject Dedner's cp^2/ch=0.18 as dimensionally incomplete (p.8).
            # M&T find errors minimized ~alpha=0.5. Using alpha=1.0 (stable for DPF).
            cr = ch / dx_min  # alpha = 1.0

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
    # AMR stepping
    # ------------------------------------------------------------------

    def amr_step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float = 0.0,
        step_number: int = 0,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Advance one AMR timestep using production MLX RHS.

        Decomposes the global state into AMR blocks, advances each block
        with the production flux pipeline (WENO5-Z/PLM + HLL/HLLD),
        then reassembles the global state.

        Args:
            state: Global state dict with 'rho', 'velocity', 'pressure', 'B', etc.
            dt: Timestep [s].
            current: Circuit current [A] for electrode BCs.
            step_number: Step counter for regrid interval.

        Returns:
            Updated global state dict.
        """
        from dpf.metal.mlx_amr import amr_step as _amr_step

        if self._amr_hierarchy is None:
            raise RuntimeError("AMR not initialized. Set enable_amr=True.")

        import mlx.core as mx

        # Pack state dict -> conserved (NVAR, nr, nz) via prim_to_cons
        from dpf.metal.mlx_primitives import cons_to_prim, prim_to_cons
        rho = mx.array(state["rho"].squeeze().astype(np.float32))
        vel = state["velocity"]
        vr = mx.array(vel[0].squeeze().astype(np.float32))
        vz = mx.array(vel[1].squeeze().astype(np.float32))
        vt = mx.array(vel[2].squeeze().astype(np.float32)) if vel.shape[0] > 2 else mx.zeros_like(rho)
        p = mx.array(state["pressure"].squeeze().astype(np.float32))
        B = state["B"]
        Br = mx.array(B[0].squeeze().astype(np.float32))
        Bz = mx.array(B[1].squeeze().astype(np.float32))
        Bt = mx.array(B[2].squeeze().astype(np.float32)) if B.shape[0] > 2 else mx.zeros_like(rho)
        U_global = prim_to_cons(rho, vr, vz, vt, p, Br, Bz, Bt, self.gamma)
        U_np = np.asarray(U_global)
        h = self._amr_hierarchy
        bnr = h.block_nr
        bnz = h.block_nz
        for idx, block in h.levels[0].blocks.items():
            ir, iz = idx
            r_s, z_s = ir * bnr, iz * bnz
            r_e = min(r_s + bnr, U_np.shape[1])
            z_e = min(z_s + bnz, U_np.shape[2])
            block.U = U_np[:, r_s:r_e, z_s:z_e].copy()

        # AMR step with production RHS
        h, _ = _amr_step(
            h, dt=dt, gamma=self.gamma,
            method=self._method, riemann=self._riemann,
            ng=3, current=current, r_inner=self._r_inner,
            step_number=step_number, rhs_fn=self._amr_rhs_fn,
        )
        self._amr_hierarchy = h

        U_out = self._amr_global_conserved_state(h, U_np.shape)

        # Unpack conserved -> state dict
        U_mx = mx.array(U_out)
        rho_o, vr_o, vz_o, vt_o, p_o, Br_o, Bz_o, Bt_o = cons_to_prim(U_mx, self.gamma)
        result = dict(state)
        result["rho"] = np.asarray(rho_o)
        result["velocity"] = np.stack([np.asarray(vr_o), np.asarray(vz_o), np.asarray(vt_o)])
        result["pressure"] = np.asarray(p_o)
        result["B"] = np.stack([np.asarray(Br_o), np.asarray(Bz_o), np.asarray(Bt_o)])
        return result

    def _amr_global_conserved_state(
        self,
        hierarchy: Any,
        global_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Gather AMR levels into one coarse-resolution conserved state.

        Level-0 blocks supply the full domain. Any active level-1 children are
        volume-restricted over their parent before the parent region is written
        to the returned global array, so refined data is represented in solver
        outputs instead of being discarded after the AMR step.
        """

        from dpf.metal.mlx_amr import AMRBlock, restrict_to_coarse

        U_out = np.zeros(global_shape, dtype=np.float32)
        coarse_level = hierarchy.levels[0]
        fine_level = hierarchy.levels[1] if len(hierarchy.levels) > 1 else None
        ratio = int(hierarchy.ratio)
        bnr = int(hierarchy.block_nr)
        bnz = int(hierarchy.block_nz)

        for idx, block in coarse_level.blocks.items():
            ir, iz = idx
            r_s, z_s = ir * bnr, iz * bnz
            bU = np.asarray(block.U).astype(np.float32, copy=True)
            if fine_level is not None and fine_level.blocks:
                children = [
                    child
                    for child in fine_level.active_blocks()
                    if child.index[0] // ratio == ir and child.index[1] // ratio == iz
                ]
                if children:
                    overlay = AMRBlock(
                        level=block.level,
                        index=block.index,
                        U=bU.copy(),
                        r_min=block.r_min,
                        z_min=block.z_min,
                        active=block.active,
                    )
                    restrict_to_coarse(children, overlay, fine_level, ratio, bnr, bnz)
                    bU = np.asarray(overlay.U).astype(np.float32, copy=False)
            r_e = min(r_s + bU.shape[1], global_shape[1])
            z_e = min(z_s + bU.shape[2], global_shape[2])
            U_out[:, r_s:r_e, z_s:z_e] = bU[:, : r_e - r_s, : z_e - z_s]
        return U_out

    # ------------------------------------------------------------------
    # Saha EOS: temperature with variable Z_bar
    # ------------------------------------------------------------------

    def _temperature_eV(
        self, rho: np.ndarray, p: np.ndarray,
    ) -> np.ndarray:
        """Compute electron temperature [eV] from (rho, p) with optional Saha Z_bar.

        Without Saha: T = p * m_i / (2 * rho * kB) (assumes Z=1).
        With Saha: T = p * m_i / ((1 + Z_bar(T)) * rho * kB) (one iteration).

        Args:
            rho: Mass density [kg/m^3], NumPy float64.
            p: Pressure [Pa], NumPy float64.

        Returns:
            Electron temperature [eV], NumPy float64.
        """
        rho_safe = np.maximum(rho, 1e-30)
        # Initial: assume Z=1
        Te_K = p * self.ion_mass / (2.0 * rho_safe * _K_B)

        if self._saha_eos is not None:
            Z_bar = self._saha_eos.zbar_numpy(Te_K)
            Te_K = p * self.ion_mass / ((1.0 + Z_bar) * rho_safe * _K_B)

        return Te_K / 11604.5  # K -> eV

    # ------------------------------------------------------------------
    # Two-temperature source terms (operator-split, CPU)
    # ------------------------------------------------------------------

    def _do_two_temperature_sources(
        self,
        result: dict[str, np.ndarray],
        dt: float,
        eta_field: float | np.ndarray | None = None,
    ) -> None:
        """Apply electron-ion equilibration, Ohmic heating, bremsstrahlung."""
        from dpf.metal.mlx_coupling import do_two_temperature_sources
        do_two_temperature_sources(
            result, dt, eta_field, self.ion_mass, self.dx,
            self.Z_eff, self.gaunt_factor, self.gamma,
        )

    # ------------------------------------------------------------------
    # Plasma inductance from density-weighted Lee formula
    # ------------------------------------------------------------------

    def _update_coupling(
        self, U: Any, current: float, voltage: float, dt: float,
    ) -> None:
        """Compute plasma inductance and update circuit coupling state."""
        from dpf.metal.mlx_coupling import update_coupling
        self._sim_time += dt
        self._coupling, self._prev_Lp, self._Lp_max = update_coupling(
            U, current, voltage, dt, self._grid, self._cathode_radius,
            self._r_inner, self._prev_Lp, self._Lp_max, self.coordinates,
            Lp_history=self._Lp_history, sim_time=self._sim_time,
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
