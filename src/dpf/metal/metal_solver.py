"""MetalMHDSolver -- ideal MHD on Apple Metal GPU via PyTorch MPS.

Production solver implementing ``PlasmaSolverBase`` with all physics
computation running on Metal GPU in float32.  Uses SSP-RK2 time
integration with PLM reconstruction and the HLL Riemann solver from
:mod:`dpf.metal.metal_riemann`.  Optionally applies constrained
transport (CT) divergence cleaning from :mod:`dpf.metal.metal_stencil`.

Conservative variable ordering (internal)::

    U = [rho, rho*vx, rho*vy, rho*vz, E_total, Bx, By, Bz]

The solver accepts and returns NumPy ``dict[str, np.ndarray]`` per the
``PlasmaSolverBase`` contract.  Data is transferred to/from the MPS
device at the boundary of each ``step()`` call.

Note on units: the Metal solver works in Heaviside-Lorentz code units
(mu_0 absorbed into B), matching the Athena++ convention used by
``metal_riemann.py``.  The Python engine uses SI with explicit mu_0;
the engine layer is responsible for any unit conversion if mixing
backends.

References:
    Shu C.-W. & Osher S., J. Comput. Phys. 77, 439 (1988) -- SSP-RK2.
    Harten A., Lax P.D., van Leer B., SIAM Rev. 25, 35 (1983) -- HLL.
    Gardiner T.A. & Stone J.M., JCP 205, 509 (2005) -- simple CT.
    Stone J.M. et al., ApJS 249, 4 (2020) -- Athena++ methods paper.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from dpf.constants import mu_0 as mu_0_si  # noqa: N812
from dpf.core.bases import CouplingState, PlasmaSolverBase
from dpf.metal.metal_riemann import (
    P_FLOOR,
    RHO_FLOOR,
    _cons_to_prim_mps,
    _fast_magnetosonic_mps,
    _prim_to_cons_mps,
    advance_nan_step_count,
    mhd_rhs_cylindrical_mps,
    mhd_rhs_mps,
    set_nan_safety_level,
)
from dpf.metal.metal_stencil import (
    ct_update_cylindrical_mps,
    ct_update_mps,
    div_B_mps,
    emf_from_fluxes_mps,
)
from dpf.metal.metal_transport import _safe_gradient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
#  torch.compile-compatible euler stage (module-level, no dicts / .item())
# ──────────────────────────────────────────────────────────────────────────── #

def _euler_stage_compilable(
    rho: torch.Tensor,
    vel: torch.Tensor,
    p: torch.Tensor,
    B: torch.Tensor,
    dt: float,
    gamma: float,
    dx: float,
    dy: float,
    dz: float,
    limiter: str,
    riemann_solver: str,
    reconstruction: str,
    bc_x: str,
    bc_y: str,
    bc_z: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor Euler stage for torch.compile acceleration (Cartesian only).

    Computes MHD RHS via ``mhd_rhs_mps`` then applies a forward Euler update
    with floor enforcement.  Returns only tensors — no dicts, no ``.item()``
    calls — so ``torch.compile`` can trace the full computation graph.

    Parameters
    ----------
    rho, vel, p, B : torch.Tensor
        Primitive state tensors.
    dt : float
        Timestep [s].
    gamma : float
        Adiabatic index.
    dx, dy, dz : float
        Grid spacings [m].
    limiter, riemann_solver, reconstruction : str
        Solver configuration strings.
    bc_x, bc_y, bc_z : str
        Boundary condition strings per axis.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        (rho_new, vel_new, p_new, B_new) after one Euler step.
    """
    state_dict: dict[str, torch.Tensor] = {
        "rho": rho, "velocity": vel, "pressure": p, "B": B,
    }
    rhs = mhd_rhs_mps(
        state_dict, gamma, dx, dy, dz,
        limiter, riemann_solver, reconstruction,
        bc=(bc_x, bc_y, bc_z),
    )

    rho_new = torch.clamp(rho + dt * rhs["rho"], min=RHO_FLOOR)
    vel_new = vel + dt * rhs["velocity"]
    p_new = torch.clamp(p + dt * rhs["pressure"], min=P_FLOOR)
    B_new = B + dt * rhs["B"]

    # Velocity clamping to 10x local fast magnetosonic speed
    cf = _fast_magnetosonic_mps(rho_new, p_new, B_new, gamma, dim=0)
    for d in range(1, 3):
        cf = torch.maximum(cf, _fast_magnetosonic_mps(rho_new, p_new, B_new, gamma, dim=d))
    v_max = torch.clamp(10.0 * cf, min=1e3, max=1e7)
    vel_new = torch.clamp(vel_new, min=-v_max, max=v_max)

    return rho_new, vel_new, p_new, B_new


# Physical constants for temperature derivation (SI)
_K_B: float = 1.380649e-23   # Boltzmann constant [J/K]
_M_D: float = 3.34358377e-27  # Deuterium mass [kg]


class MetalMHDSolver(PlasmaSolverBase):
    """MHD solver on Apple Metal GPU via PyTorch MPS.

    Uses SSP-RK2 time integration with selectable Riemann solver
    (HLL or HLLD) and reconstruction (PLM or WENO5).  All physics
    computation runs on Metal GPU in float32 (or CPU in float64).
    Accepts/returns ``dict[str, np.ndarray]`` per ``PlasmaSolverBase``
    contract.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Number of cells (nx, ny, nz).
    dx : float
        Grid spacing in x [m].
    gamma : float
        Adiabatic index (default 5/3).
    cfl : float
        CFL safety factor (default 0.3).
    device : str
        PyTorch device string: ``"mps"`` for Metal GPU, ``"cpu"`` for
        testing (default ``"mps"``).
    dy : float | None
        Grid spacing in y [m].  Defaults to *dx* if ``None``.
    dz : float | None
        Grid spacing in z [m].  Defaults to *dx* if ``None``.
    limiter : str
        Slope limiter for PLM reconstruction: ``"minmod"`` or ``"mc"``
        (default ``"minmod"``).
    use_ct : bool
        If ``True``, apply constrained transport for div(B) cleaning
        after each RK stage (default ``True``).

    Attributes
    ----------
    grid_shape : tuple[int, int, int]
    gamma : float
    cfl : float
    device : torch.device
    limiter : str
    use_ct : bool
    dx, dy, dz : float

    Notes
    -----
    MPS does **not** support float64.  All internal computation is float32.
    Inputs are cast on transfer; outputs are returned as float64 NumPy arrays
    for compatibility with the rest of the DPF engine.

    The solver evolves only the core MHD variables (rho, velocity, pressure, B).
    Auxiliary fields (Te, Ti, psi) are **passed through** unchanged -- the
    engine layer is responsible for evolving those quantities.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        dx: float,
        gamma: float = 5.0 / 3.0,
        cfl: float = 0.3,
        device: str = "mps",
        dy: float | None = None,
        dz: float | None = None,
        limiter: str = "minmod",
        use_ct: bool = True,
        riemann_solver: str = "hll",
        precision: str = "float32",
        reconstruction: str = "plm",
        time_integrator: str = "ssp_rk2",
        enable_hall: bool = False,
        enable_braginskii_conduction: bool = False,
        enable_braginskii_viscosity: bool = False,
        enable_nernst: bool = False,
        enable_bremsstrahlung: bool = False,
        gaunt_factor: float = 1.2,
        ion_mass: float = 3.34358377e-27,
        Z_eff: float = 1.0,
        coordinates: str = "cartesian",
        bc: str | tuple[str, str, str] = "outflow",
        r_inner: float = 0.0,
        convert_b_si_to_hl: bool = False,
        enable_betatron: bool = False,
        implicit_resistivity: bool = False,
        compile_mode: bool = False,
        reconstruction_precision: str = "float32",
        safety_level: str = "normal",
    ) -> None:
        self.grid_shape: tuple[int, int, int] = grid_shape
        self.dx: float = float(dx)
        self.dy: float = float(dy) if dy is not None else self.dx
        self.dz: float = float(dz) if dz is not None else self.dx
        self.gamma: float = float(gamma)
        self.cfl: float = float(cfl)
        self.limiter: str = limiter
        self.use_ct: bool = use_ct
        self.riemann_solver: str = riemann_solver
        self.precision: str = precision
        self.reconstruction_precision: str = reconstruction_precision
        self.reconstruction: str = reconstruction
        self.time_integrator: str = time_integrator
        self.enable_hall: bool = enable_hall
        self.enable_braginskii_conduction: bool = enable_braginskii_conduction
        self.enable_braginskii_viscosity: bool = enable_braginskii_viscosity
        self.enable_nernst: bool = enable_nernst
        self.enable_bremsstrahlung: bool = enable_bremsstrahlung
        self.gaunt_factor: float = float(gaunt_factor)
        self.ion_mass: float = float(ion_mass)
        self.Z_eff: float = float(Z_eff)
        self.coordinates: str = coordinates
        self._convert_b_si_to_hl: bool = convert_b_si_to_hl
        self.enable_betatron: bool = enable_betatron
        self.implicit_resistivity: bool = implicit_resistivity
        # Boundary conditions: "outflow" (zero-gradient) or "periodic"
        # Can be a single string (applied to all dims) or a 3-tuple per dim.
        if isinstance(bc, str):
            self.bc: tuple[str, str, str] = (bc, bc, bc)
        else:
            self.bc = tuple(bc)  # type: ignore[arg-type]
        self._last_P_radiated: float = 0.0
        self.total_radiated_energy: float = 0.0

        # Determine dtype from precision setting ---------------------------
        if precision == "float64":
            self._dtype: torch.dtype = torch.float64
            # MPS does not support float64 → force CPU for max accuracy
            device = "cpu"
            logger.info(
                "Float64 precision requested — using CPU backend for "
                "maximum numerical accuracy"
            )
        else:
            self._dtype = torch.float32

        # Resolve device --------------------------------------------------
        if device == "mps" and not self.is_available():
            logger.warning(
                "MPS device requested but not available; falling back to CPU"
            )
            device = "cpu"
        self.device: torch.device = torch.device(device)

        # Coupling state (updated each step) -------------------------------
        self._coupling: CouplingState = CouplingState()
        self._prev_Lp: float | None = None

        # Pre-allocate a zero tensor for quick allocation in _to_device ----
        nx, ny, nz = self.grid_shape
        self._zero_scalar: torch.Tensor = torch.zeros(
            nx, ny, nz, dtype=self._dtype, device=self.device
        )
        self._zero_vector: torch.Tensor = torch.zeros(
            3, nx, ny, nz, dtype=self._dtype, device=self.device
        )

        # Cylindrical coordinate support -----------------------------------
        # r_inner offsets the radial grid so domain covers [r_inner, r_inner + nx*dx].
        # For DPF inter-electrode gap: r_inner = anode_radius.
        self.compile_mode: bool = compile_mode
        self.safety_level: str = safety_level
        set_nan_safety_level(safety_level)
        self._r_inner: float = float(r_inner if r_inner is not None else 0.0)
        if self.coordinates == "cylindrical":
            r_1d = self._r_inner + (torch.arange(nx, dtype=self._dtype, device=self.device)
                    + 0.5) * self.dx
            self._r = r_1d.reshape(nx, 1, 1)
            self._inv_r = 1.0 / torch.clamp(self._r, min=1e-30)
            # Face radii: r_{i+1/2} = r_inner + i * dx, shape (nx+1, 1, 1)
            r_face_1d = self._r_inner + torch.arange(nx + 1, dtype=self._dtype, device=self.device) * self.dx
            self._r_face: torch.Tensor | None = r_face_1d.reshape(nx + 1, 1, 1)
        else:
            self._r = None
            self._inv_r = None
            self._r_face = None

        logger.info(
            "MetalMHDSolver initialized: grid=%s  dx=%.3e  dy=%.3e  dz=%.3e  "
            "gamma=%.4f  cfl=%.2f  device=%s  limiter=%s  use_ct=%s  "
            "riemann=%s  recon=%s  recon_prec=%s  time=%s  precision=%s  coords=%s  "
            "hall=%s  braginskii_cond=%s  braginskii_visc=%s  nernst=%s  "
            "bremsstrahlung=%s  betatron=%s  compile_mode=%s  safety_level=%s",
            self.grid_shape, self.dx, self.dy, self.dz,
            self.gamma, self.cfl, self.device, self.limiter, self.use_ct,
            self.riemann_solver, self.reconstruction, self.reconstruction_precision,
            self.time_integrator,
            self.precision, self.coordinates,
            self.enable_hall, self.enable_braginskii_conduction,
            self.enable_braginskii_viscosity, self.enable_nernst,
            self.enable_bremsstrahlung, self.enable_betatron,
            self.compile_mode, self.safety_level,
        )

        # Compile the hot euler stage if requested.
        # Only enabled for Cartesian coordinates (cylindrical uses a separate
        # RHS path with extra tensor arguments that torch.compile handles less
        # efficiently on first-call tracing).
        self._compiled_euler_stage: object | None = None
        if self.compile_mode and self.coordinates == "cartesian":
            try:
                self._compiled_euler_stage = torch.compile(
                    _euler_stage_compilable,
                    fullgraph=False,
                )
                logger.info("torch.compile applied to _euler_stage_compilable")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "torch.compile unavailable (%s); falling back to eager mode", exc
                )
                self._compiled_euler_stage = None

    # ------------------------------------------------------------------ #
    #  Device availability check
    # ------------------------------------------------------------------ #

    @classmethod
    def is_available(cls) -> bool:
        """Check whether the MPS (Metal) backend is usable.

        Returns
        -------
        bool
            ``True`` if PyTorch MPS is built and a Metal GPU is present.
        """
        try:
            return (
                torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
        except AttributeError:
            return False

    # ------------------------------------------------------------------ #
    #  NumPy <-> GPU transfer helpers
    # ------------------------------------------------------------------ #

    def _to_device(
        self, state: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Transfer a NumPy state dict to GPU tensors (float32).

        Parameters
        ----------
        state : dict[str, np.ndarray]
            State dict with at least ``rho``, ``velocity``, ``pressure``,
            ``B``.  Optional keys ``Te``, ``Ti``, ``psi`` are also
            transferred.

        Returns
        -------
        dict[str, torch.Tensor]
            Same keys, values on ``self.device`` in float32.
        """
        gpu_state: dict[str, torch.Tensor] = {}
        for key in ("rho", "velocity", "pressure", "B", "Te", "Ti", "psi", "e_electron"):
            arr = state.get(key)
            if arr is not None:
                gpu_state[key] = torch.as_tensor(
                    arr, dtype=self._dtype
                ).to(self.device)
            else:
                # Provide sensible defaults for optional fields
                if key in ("Te", "Ti"):
                    gpu_state[key] = torch.full_like(
                        self._zero_scalar, 1e4
                    )
                elif key == "psi":
                    gpu_state[key] = self._zero_scalar.clone()

        # SI → Heaviside-Lorentz unit conversion for B (cylindrical only):
        # B_HL = B_SI / sqrt(mu_0). This ensures 0.5*B_HL^2 = B_SI^2/(2*mu_0)
        # in the energy equation, giving correct magnetic pressure.
        # Only applied for cylindrical coordinates where J×B forces must be
        # physically correct for the snowplow to emerge from MHD.
        # The Cartesian/hybrid paths rely on Lee model for dynamics, not B forces.
        if self._convert_b_si_to_hl and "B" in gpu_state:
            import math
            gpu_state["B"] = gpu_state["B"] / math.sqrt(mu_0_si)

        return gpu_state

    @staticmethod
    def _to_numpy(state_gpu: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Transfer GPU tensors back to NumPy float64 arrays.

        Converts B from Heaviside-Lorentz code units back to SI Tesla:
        B_SI = B_HL * sqrt(mu_0).

        Parameters
        ----------
        state_gpu : dict[str, torch.Tensor]
            State dict on GPU (B in HL units).

        Returns
        -------
        dict[str, np.ndarray]
            Same keys, float64 NumPy arrays on CPU (B in SI Tesla).
        """
        return {
            key: tensor.detach().cpu().to(torch.float64).numpy()
            for key, tensor in state_gpu.items()
        }

    # ------------------------------------------------------------------ #
    #  Floor enforcement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_floors(state_gpu: dict[str, torch.Tensor]) -> None:
        """Enforce positivity floors on density and pressure (in-place).

        Parameters
        ----------
        state_gpu : dict[str, torch.Tensor]
            Mutable state dict on GPU.  Modified in-place.
        """
        state_gpu["rho"] = torch.clamp(state_gpu["rho"], min=RHO_FLOOR)
        state_gpu["pressure"] = torch.clamp(state_gpu["pressure"], min=P_FLOOR)
        if "e_electron" in state_gpu:
            state_gpu["e_electron"] = torch.clamp(
                state_gpu["e_electron"], min=0.0,
            )

    # ------------------------------------------------------------------ #
    #  CFL timestep
    # ------------------------------------------------------------------ #

    def _compute_dt_cfl(self, state_gpu: dict[str, torch.Tensor]) -> float:
        """Compute the CFL-limited timestep from the fast magnetosonic speed.

        Uses the maximum signal speed across all cells and all three
        coordinate directions:

            dt = CFL * min(dx, dy, dz) / max(|v| + c_f)

        Parameters
        ----------
        state_gpu : dict[str, torch.Tensor]
            Current GPU state.

        Returns
        -------
        float
            CFL-limited timestep [s].
        """
        rho = state_gpu["rho"]
        vel = state_gpu["velocity"]
        p = state_gpu["pressure"]
        B = state_gpu["B"]

        max_signal: float = 0.0
        dh = [self.dx, self.dy, self.dz]

        for dim in range(3):
            cf = _fast_magnetosonic_mps(rho, p, B, self.gamma, dim)
            vn_abs = torch.abs(vel[dim])
            signal = vn_abs + cf
            local_max = float(torch.max(signal).item())
            # Scale by inverse grid spacing for this direction
            if dh[dim] > 0.0:
                max_signal = max(max_signal, local_max / dh[dim])

        if max_signal <= 0.0:
            # Degenerate case: return a safe small dt
            return self.cfl * min(self.dx, self.dy, self.dz) * 1e-6

        dt_cfl = self.cfl / max_signal

        # Whistler wave CFL constraint (Hall MHD):
        # dt_hall < dx^2 * n_e * e / (|B| * c)
        # The whistler dispersion omega = k^2 * |B| / (mu_0 * n_e * e)
        # gives a phase speed v_w = k * |B| / (mu_0 * n_e * e) that grows
        # with k, making the CFL MORE restrictive on fine grids.
        # EMPIRICAL: This constraint dominates on grids finer than ~64^3.
        if self.enable_hall:
            _E_CHARGE = 1.602176634e-19
            B_sq = torch.sum(B ** 2, dim=0)
            B_max = float(torch.sqrt(torch.max(B_sq)).item())
            ne = rho / self.ion_mass
            ne_max = float(torch.max(ne).item())
            if B_max > 0 and ne_max > 0:
                dx_min = min(self.dx, self.dy, self.dz)
                # v_whistler = |B| / (mu_0 * n_e * e * dx)
                v_hall = B_max / (mu_0_si * max(ne_max, 1e-20)
                                  * _E_CHARGE * dx_min)
                dt_hall = self.cfl * dx_min / max(v_hall, 1e-30)
                dt_cfl = min(dt_cfl, dt_hall)

        return float(dt_cfl)

    # ------------------------------------------------------------------ #
    #  Primitive -> Conservative helpers (on GPU)
    # ------------------------------------------------------------------ #

    def _state_to_conservative(
        self, state_gpu: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert a primitive-variable GPU state dict to conservative vector.

        Parameters
        ----------
        state_gpu : dict[str, torch.Tensor]
            Must contain ``rho``, ``velocity``, ``pressure``, ``B``.

        Returns
        -------
        torch.Tensor
            Conservative state ``U``, shape ``(8, nx, ny, nz)``.
        """
        return _prim_to_cons_mps(
            state_gpu["rho"],
            state_gpu["velocity"],
            state_gpu["pressure"],
            state_gpu["B"],
            self.gamma,
        )

    def _conservative_to_state(
        self, U: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert conservative vector to a primitive-variable state dict.

        Parameters
        ----------
        U : torch.Tensor
            Conservative state, shape ``(8, nx, ny, nz)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys ``rho``, ``velocity``, ``pressure``, ``B``.
        """
        rho, vel, p, B = _cons_to_prim_mps(U, self.gamma)
        return {
            "rho": rho,
            "velocity": vel,
            "pressure": p,
            "B": B,
        }

    # ------------------------------------------------------------------ #
    #  SSP Runge-Kutta time integration  (main step method)
    # ------------------------------------------------------------------ #

    def _compute_rhs(
        self,
        rho: torch.Tensor,
        vel: torch.Tensor,
        p: torch.Tensor,
        B: torch.Tensor,
        e_electron: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute MHD RHS for a given primitive state."""
        state_dict: dict[str, torch.Tensor] = {
            "rho": rho, "velocity": vel, "pressure": p, "B": B,
        }
        if e_electron is not None:
            state_dict["e_electron"] = e_electron
        if self.coordinates == "cylindrical":
            return mhd_rhs_cylindrical_mps(
                state_dict,
                self.gamma,
                self.dx, self.dy, self.dz,
                r_cell=self._r,
                r_face=self._r_face,
                limiter=self.limiter,
                riemann_solver=self.riemann_solver,
                reconstruction=self.reconstruction,
                bc=self.bc,
                reconstruction_precision=self.reconstruction_precision,
            )
        return mhd_rhs_mps(
            state_dict,
            self.gamma,
            self.dx, self.dy, self.dz,
            self.limiter,
            self.riemann_solver,
            self.reconstruction,
            bc=self.bc,
            reconstruction_precision=self.reconstruction_precision,
        )

    def _euler_update(
        self,
        rho: torch.Tensor,
        vel: torch.Tensor,
        p: torch.Tensor,
        B: torch.Tensor,
        rhs: dict[str, torch.Tensor],
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward Euler update with floor enforcement and velocity clamping.

        After the primitive variable update, clamp velocity to a
        physically motivated upper bound: 10x the local fast
        magnetosonic speed.  This prevents catastrophic runaway at
        cells where reconstruction overshoots produced extreme RHS
        values.
        """
        rho_new = torch.clamp(rho + dt * rhs["rho"], min=RHO_FLOOR)
        vel_new = vel + dt * rhs["velocity"]
        p_new = torch.clamp(p + dt * rhs["pressure"], min=P_FLOOR)
        B_new = B + dt * rhs["B"]

        # Velocity clamping: cap at 10x local fast magnetosonic speed
        # (prevents runaway from extreme RHS at strong discontinuities)
        cf = _fast_magnetosonic_mps(rho_new, p_new, B_new, self.gamma, dim=0)
        for d in range(1, 3):
            cf = torch.maximum(cf, _fast_magnetosonic_mps(
                rho_new, p_new, B_new, self.gamma, dim=d,
            ))
        v_max = torch.clamp(10.0 * cf, min=1e3, max=1e7)  # at least 1 km/s
        vel_new = torch.clamp(vel_new, min=-v_max, max=v_max)

        return rho_new, vel_new, p_new, B_new

    # ------------------------------------------------------------------ #
    #  Cylindrical geometric source terms
    # ------------------------------------------------------------------ #

    def _apply_cylindrical_sources(
        self,
        rho: torch.Tensor,
        vel: torch.Tensor,
        p: torch.Tensor,
        B: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Operator-split cylindrical geometric source terms.

        Corrects for the difference between Cartesian flux divergence
        (dF/dr) and cylindrical flux divergence ((1/r) d(rF)/dr), plus
        geometric momentum sources from curvilinear coordinates.

        In Heaviside-Lorentz units (mu_0 absorbed into B), the total
        corrections relative to the Cartesian solver are:

        Continuity:
            S_rho = -rho * v_r / r

        r-momentum (flux correction + geometric source):
            S_mr = (rho(v_theta^2 - v_r^2) + B_r^2 - B_theta^2) / r

        theta-momentum:
            S_mtheta = -2(rho*v_r*v_theta - B_r*B_theta) / r

        z-momentum (flux correction only):
            S_mz = -(rho*v_r*v_z - B_r*B_z) / r

        Pressure (adiabatic correction):
            S_p = -gamma * p * v_r / r

        B_theta induction:
            S_Btheta = -(v_r*B_theta - v_theta*B_r) / r

        B_z induction:
            S_Bz = -(v_r*B_z - v_z*B_r) / r

        References
        ----------
        Stone & Norman, ApJS 80:753 (1992) -- ZEUS-2D.
        Mignone et al., ApJS 170:228 (2007) -- PLUTO code.
        Stone et al., ApJS 249:4 (2020) -- Athena++ methods paper.

        Parameters
        ----------
        rho, vel, p, B : torch.Tensor
            Primitive state after Cartesian RK integration.
        dt : float
            Timestep [s].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Corrected (rho, vel, p, B).
        """
        inv_r = self._inv_r  # shape (nx, 1, 1)

        v_r = vel[0]
        v_theta = vel[1]
        v_z = vel[2]
        B_r = B[0]
        B_theta = B[1]
        B_z = B[2]

        # --- Conservative source terms ---

        # Continuity: S_rho = -rho * v_r / r
        S_rho = -rho * v_r * inv_r

        # r-momentum (combined flux correction + geometric source):
        # S_mr = (rho(v_theta^2 - v_r^2) + B_r^2 - B_theta^2) / r
        S_mr = (rho * (v_theta ** 2 - v_r ** 2)
                + B_r ** 2 - B_theta ** 2) * inv_r

        # theta-momentum:
        # S_mtheta = -2(rho*v_r*v_theta - B_r*B_theta) / r
        S_mtheta = -2.0 * (rho * v_r * v_theta
                           - B_r * B_theta) * inv_r

        # z-momentum (flux correction only):
        # S_mz = -(rho*v_r*v_z - B_r*B_z) / r
        S_mz = -(rho * v_r * v_z - B_r * B_z) * inv_r

        # Pressure (adiabatic correction for cylindrical div(v)):
        # div(v)_cyl = dv_r/dr + v_r/r + dv_z/dz
        # Cartesian solver computes dv_r/dr + dv_z/dz, correction: v_r/r
        # dp/dt += -gamma * p * v_r / r
        S_p = -self.gamma * p * v_r * inv_r

        # Induction corrections:
        # B_theta: -(v_r*B_theta - v_theta*B_r) / r
        S_Btheta = -(v_r * B_theta - v_theta * B_r) * inv_r
        # B_z: -(v_r*B_z - v_z*B_r) / r
        S_Bz = -(v_r * B_z - v_z * B_r) * inv_r

        # --- Apply forward Euler update ---
        rho_new = torch.clamp(rho + dt * S_rho, min=RHO_FLOOR)

        # Convert conservative momentum sources to primitive velocity updates
        # dv = (S_mom - v * S_rho) / rho
        rho_safe = torch.clamp(rho, min=RHO_FLOOR)
        inv_rho = 1.0 / rho_safe

        vel_new = vel.clone()
        vel_new[0] = v_r + dt * (S_mr - v_r * S_rho) * inv_rho
        vel_new[1] = v_theta + dt * (S_mtheta - v_theta * S_rho) * inv_rho
        vel_new[2] = v_z + dt * (S_mz - v_z * S_rho) * inv_rho

        p_new = torch.clamp(p + dt * S_p, min=P_FLOOR)

        B_new = B.clone()
        B_new[1] = B_theta + dt * S_Btheta
        B_new[2] = B_z + dt * S_Bz

        return rho_new, vel_new, p_new, B_new

    def _apply_resistive_diffusion(
        self,
        B: torch.Tensor,
        p: torch.Tensor,
        rho: torch.Tensor,
        eta: torch.Tensor,
        dt: float,
        gamma: float,
        J_kin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply resistive MHD operator-split step (Code Units).

        Using Heaviside-Lorentz units (mu_0 = 1).
        Resistivity eta must be scaled: eta_code = eta_SI / mu_0_SI.

        When ``self.implicit_resistivity`` is True, uses backward-Euler
        implicit solve via the batched Thomas algorithm — unconditionally
        stable regardless of eta magnitude.  Otherwise falls back to
        explicit sub-cycling (capped at 20 sub-steps).
        """
        eta_eff = eta / mu_0_si

        # ── Current density (needed for Ohmic heating in both paths) ──
        dBz_dy = _safe_gradient(B[2], dim=1, spacing=self.dy)
        dBy_dz = _safe_gradient(B[1], dim=2, spacing=self.dz)
        dBx_dz = _safe_gradient(B[0], dim=2, spacing=self.dz)
        dBz_dx = _safe_gradient(B[2], dim=0, spacing=self.dx)
        dBy_dx = _safe_gradient(B[1], dim=0, spacing=self.dx)
        dBx_dy = _safe_gradient(B[0], dim=1, spacing=self.dy)

        Jx = dBz_dy - dBy_dz
        Jy = dBx_dz - dBz_dx
        Jz = dBy_dx - dBx_dy

        if J_kin is not None:
            Jx = Jx - J_kin[0]
            Jy = Jy - J_kin[1]
            Jz = Jz - J_kin[2]

        # Ohmic heating: Q = eta_eff * |J|^2
        J_sq = Jx ** 2 + Jy ** 2 + Jz ** 2
        Q_ohm = eta_eff * J_sq

        # ── Implicit path: batched Thomas algorithm ──────────────────
        if self.implicit_resistivity:
            from dpf.metal.metal_transport import implicit_resistive_step

            B_new = B.clone()
            # Diffuse B_theta (component 1) — primary resistive channel in DPF
            B_new = implicit_resistive_step(
                B_new, eta_eff, dt, self.dx,
                r=self._r,
                coordinates=self.coordinates,
                mu_0=1.0,  # already in code units
                component=1,
            )
            # Diffuse B_z (component 2) if 3D
            if B.shape[2] > 1 or B.shape[3] > 1:
                B_new = implicit_resistive_step(
                    B_new, eta_eff, dt, self.dx,
                    r=self._r,
                    coordinates=self.coordinates,
                    mu_0=1.0,
                    component=2,
                )

            # Ohmic heating (explicit, unconditionally stable for pressure)
            p_new = p + dt * (gamma - 1.0) * Q_ohm
            p_new = torch.clamp(p_new, min=1e-12)

            return B_new, p_new

        # ── Explicit path: sub-cycled forward Euler (legacy) ─────────
        eta_Jx = eta_eff * Jx
        eta_Jy = eta_eff * Jy
        eta_Jz = eta_eff * Jz

        d_etaJz_dy = _safe_gradient(eta_Jz, dim=1, spacing=self.dy)
        d_etaJy_dz = _safe_gradient(eta_Jy, dim=2, spacing=self.dz)
        d_etaJx_dz = _safe_gradient(eta_Jx, dim=2, spacing=self.dz)
        d_etaJz_dx = _safe_gradient(eta_Jz, dim=0, spacing=self.dx)
        d_etaJy_dx = _safe_gradient(eta_Jy, dim=0, spacing=self.dx)
        d_etaJx_dy = _safe_gradient(eta_Jx, dim=1, spacing=self.dy)

        dB_dt = torch.zeros_like(B)
        dB_dt[0] = -(d_etaJz_dy - d_etaJy_dz)
        dB_dt[1] = -(d_etaJx_dz - d_etaJz_dx)
        dB_dt[2] = -(d_etaJy_dx - d_etaJx_dy)

        # Sub-cycle for CFL stability: dt_res < dx^2 / (2 * eta_eff_max)
        dx_min = min(self.dx, self.dz)
        eta_eff_max = float(eta_eff.max().item()) if eta_eff.numel() > 0 else 0.0
        if eta_eff_max > 0:
            dt_res_cfl = dx_min**2 / (2.0 * eta_eff_max)
            n_sub = max(1, int(np.ceil(dt / dt_res_cfl)))
        else:
            n_sub = 1
        n_sub = min(n_sub, 20)  # cap sub-cycles to avoid excessive cost
        dt_sub = dt / n_sub

        B_new = B.clone()
        p_new = p.clone()
        for _ in range(n_sub):
            B_new = B_new + dt_sub * dB_dt
            p_new = p_new + dt_sub * (gamma - 1.0) * Q_ohm
        p_new = torch.clamp(p_new, min=1e-12)

        return B_new, p_new

    def _apply_bremsstrahlung(
        self,
        rho: torch.Tensor,
        p: torch.Tensor,
        Te: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply bremsstrahlung radiation cooling (operator-split).

        Uses implicit backward Euler for numerical stability on the
        nonlinear cooling equation (P_ff ~ sqrt(Te)):

            Te_new + alpha * sqrt(Te_new) = Te_old

        where alpha = dt * C_brem * g_ff * Z * ne / (1.5 * k_B).

        The electron pressure loss is coupled to the MHD pressure:
            dp = ne * k_B * (Te_new - Te_old)

        Parameters
        ----------
        rho : torch.Tensor
            Mass density, shape ``(nx, ny, nz)``.
        p : torch.Tensor
            Total pressure, shape ``(nx, ny, nz)``.
        Te : torch.Tensor
            Electron temperature [K], shape ``(nx, ny, nz)``.
        dt : float
            Timestep [s].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (p_new, Te_new, P_radiated) where P_radiated is volumetric
            power density [W/m^3], shape ``(nx, ny, nz)``.

        References
        ----------
        Rybicki & Lightman (1979), Eq. 5.14a.
        NRL Plasma Formulary (2019), p. 58.
        """
        # BREM_COEFF = 1.42e-40 [W m^3 K^{-1/2}] (Rybicki & Lightman 1979)
        # Pre-combine small constants to avoid float32 underflow:
        # alpha_coeff = BREM_COEFF / (1.5 * k_B) = 1.42e-40 / 2.07e-23 = 6.86e-18
        # Then alpha = alpha_coeff * gaunt * Z * ne * dt (all float32-safe)
        _ALPHA_COEFF = 1.42e-40 / (1.5 * _K_B)  # ~6.86e-18  [m^3 K^{-3/2} s]
        Te_floor = 1.0

        ne = torch.clamp(rho, min=0.0) / self.ion_mass

        # Implicit backward Euler coefficient (float32-safe ordering)
        alpha = (_ALPHA_COEFF * self.gaunt_factor * self.Z_eff) * ne * dt

        # Newton iteration: f(T) = T + alpha*sqrt(T) - Te = 0
        # Use 8 iterations (not 4) to ensure convergence when alpha > 1e3
        # (strong radiation regime at pinch conditions, ne~1e27, Te~1e7 K).
        Te_new = Te.clone()
        for _ in range(8):
            sqrt_T = torch.sqrt(torch.clamp(Te_new, min=Te_floor))
            f = Te_new + alpha * sqrt_T - Te
            fp = 1.0 + alpha / (2.0 * sqrt_T)
            Te_new = Te_new - f / fp
            Te_new = torch.clamp(Te_new, min=Te_floor)

        # Electron pressure loss: dp_e = ne * k_B * (Te_new - Te_old)
        dp_e = ne * _K_B * (Te_new - Te)
        p_new = torch.clamp(p + dp_e, min=P_FLOOR)

        # Volumetric radiated power for diagnostics
        P_radiated = (1.5 * ne * _K_B
                      * torch.clamp(Te - Te_new, min=0.0)
                      / max(dt, 1e-30))

        return p_new, Te_new, P_radiated

    def _apply_betatron_heating(
        self,
        Ti: torch.Tensor,
        B: torch.Tensor,
        B_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Operator-split betatron heating: T_perp * B = const (adiabatic compression).

        During magnetic compression, perpendicular ion temperature rises as
        dT_perp = -T_perp * (dB / B) per cell per timestep (magnetic moment
        conservation for ions).  Only applied in cells where |B| is increasing.

        Physics reference: NRL Plasma Formulary, adiabatic invariant mu = T_perp / B.

        Parameters
        ----------
        Ti : torch.Tensor
            Ion temperature [K], shape (nx, ny, nz).
        B : torch.Tensor
            Updated B field [code units], shape (3, nx, ny, nz).
        B_prev : torch.Tensor
            Previous-step B field [code units], shape (3, nx, ny, nz).

        Returns
        -------
        torch.Tensor
            Updated Ti with betatron heating applied, same shape as input.
        """
        B_mag = torch.sqrt(torch.sum(B ** 2, dim=0).clamp(min=1e-30))
        B_mag_prev = torch.sqrt(torch.sum(B_prev ** 2, dim=0).clamp(min=1e-30))

        # dB/B: fractional change in |B| this step
        dB_over_B = (B_mag - B_mag_prev) / B_mag_prev

        # Magnetic moment conservation: T_perp_new = T_perp_old * (B_new / B_old)
        # Equivalent to dT_perp = -T_perp * (dB/B) where dB = B_new - B_old
        # Apply only in compression cells (dB/B > 0)
        compression_mask = dB_over_B > 0.0
        Ti_new = torch.where(
            compression_mask,
            Ti * (B_mag / B_mag_prev),
            Ti,
        )
        return torch.clamp(Ti_new, min=1.0)  # 1 K floor

    def step_gpu(
        self,
        state_gpu: dict[str, torch.Tensor],
        dt: float,
        current: float,
        voltage: float,
        eta_field_gpu: torch.Tensor | None = None,
        source_terms: dict[str, np.ndarray] | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Advance the MHD state by one timestep, staying on GPU throughout.

        Identical physics to ``step()`` but accepts and returns
        ``dict[str, torch.Tensor]`` directly, eliminating the NumPy↔GPU
        transfers at the call boundary.  Use this method when the caller
        can keep state resident on the GPU across multiple steps.

        Parameters
        ----------
        state_gpu : dict[str, torch.Tensor]
            State dictionary on ``self.device`` with keys:
              - ``"rho"``:      ``(nx, ny, nz)`` density
              - ``"velocity"``: ``(3, nx, ny, nz)`` velocity
              - ``"pressure"``: ``(nx, ny, nz)`` pressure
              - ``"B"``:        ``(3, nx, ny, nz)`` magnetic field
              - ``"Te"``:       ``(nx, ny, nz)`` electron temperature
              - ``"Ti"``:       ``(nx, ny, nz)`` ion temperature
              - ``"psi"``:      ``(nx, ny, nz)`` divergence cleaning scalar
        dt : float
            Timestep size [s].
        current : float
            Circuit current [A].
        voltage : float
            Capacitor voltage [V].
        eta_field_gpu : torch.Tensor | None
            Spatially-resolved resistivity [Ohm*m] already on GPU,
            shape (nx, ny, nz).  If None, resistive diffusion is skipped.
        source_terms : dict[str, np.ndarray] | None
            Optional source terms (Q_ohmic_correction, J_kin).

        Returns
        -------
        dict[str, torch.Tensor]
            Updated state dictionary on ``self.device``.
        """
        # Save auxiliary fields (passed through, not evolved on GPU)
        Te_pass = state_gpu["Te"].clone()
        Ti_pass = state_gpu["Ti"].clone()
        psi_pass = state_gpu["psi"].clone()

        # -------------------------------------------------------------- #
        #  Save U^n (primitive form on GPU)
        # -------------------------------------------------------------- #
        rho_n = state_gpu["rho"].clone()
        vel_n = state_gpu["velocity"].clone()
        p_n = state_gpu["pressure"].clone()
        B_n = state_gpu["B"].clone()
        ee_n = state_gpu.get("e_electron")
        if ee_n is not None:
            ee_n = ee_n.clone()

        if self.time_integrator == "ssp_rk3":
            rho_new, vel_new, p_new, B_new, ee_new = self._step_ssp_rk3(
                rho_n, vel_n, p_n, B_n, dt, e_electron=ee_n,
            )
        else:
            rho_new, vel_new, p_new, B_new, ee_new = self._step_ssp_rk2(
                rho_n, vel_n, p_n, B_n, dt, e_electron=ee_n,
            )

        # -------------------------------------------------------------- #
        #  Cylindrical geometric source terms
        # -------------------------------------------------------------- #
        # Sources are now embedded in mhd_rhs_cylindrical_mps via the
        # r-weighted flux differencing + geometric source terms.
        # The operator-split _apply_cylindrical_sources is no longer called.

        # -------------------------------------------------------------- #
        #  Resistive MHD operator-split step (if eta_field provided)
        # -------------------------------------------------------------- #
        if eta_field_gpu is not None:
            # Accept torch.Tensor directly; no conversion needed.
            eta_gpu = eta_field_gpu.to(dtype=self._dtype, device=self.device)

            # Kinetic source terms (J_kin): PIC-MHD coupling not yet integrated
            # into the Metal resistive diffusion step.  When implemented, convert
            # J_kin to code units and subtract from J_tot before computing eta*J.
            if source_terms is not None and "J_kin" in source_terms:
                logger.debug(
                    "J_kin source term present but not yet applied in Metal solver"
                )

            # Circuit-MHD ohmic correction: add J^2-weighted gap to pressure
            if source_terms is not None and "Q_ohmic_correction" in source_terms:
                Q_corr = source_terms["Q_ohmic_correction"]
                Q_gpu = torch.as_tensor(
                    Q_corr, dtype=p_new.dtype, device=p_new.device,
                )
                p_new = p_new + (self.gamma - 1.0) * Q_gpu * dt

            # Apply resistivity (B is in Code Units here)
            # We must use Code Unit logic in _apply_resistive_diffusion

            B_new, p_new = self._apply_resistive_diffusion(
                B_new, p_new, rho_new, eta_gpu, dt, self.gamma,
            )

        # -------------------------------------------------------------- #
        #  Operator-split transport physics (Phase Q)
        # -------------------------------------------------------------- #
        if (self.enable_hall or self.enable_braginskii_conduction
                or self.enable_braginskii_viscosity or self.enable_nernst):
            from dpf.metal.metal_transport import (
                apply_braginskii_conduction_mps,
                apply_braginskii_viscosity_mps,
                apply_hall_mhd_mps,
                apply_nernst_advection_mps,
            )

            if self.enable_hall:
                B_new = apply_hall_mhd_mps(
                    B_new, rho_new, dt,
                    self.dx, self.dy, self.dz, self.ion_mass,
                )

            if self.enable_braginskii_conduction:
                ne = rho_new / self.ion_mass
                Te_pass = apply_braginskii_conduction_mps(
                    Te_pass, B_new, ne, dt,
                    self.dx, self.dy, self.dz,
                    Z_eff=self.Z_eff,
                )

            if self.enable_braginskii_viscosity:
                vel_new, p_new = apply_braginskii_viscosity_mps(
                    vel_new, rho_new, p_new, B_new,
                    Ti_pass, dt,
                    self.dx, self.dy, self.dz,
                    ion_mass=self.ion_mass,
                )

            if self.enable_nernst:
                ne = rho_new / self.ion_mass
                B_new = apply_nernst_advection_mps(
                    B_new, ne, Te_pass, dt,
                    self.dx, self.dy, self.dz,
                    Z_eff=self.Z_eff,
                )

        # -------------------------------------------------------------- #
        #  Betatron heating (operator-split)
        # -------------------------------------------------------------- #
        if self.enable_betatron:
            Ti_pass = self._apply_betatron_heating(Ti_pass, B_new, B_n)

        # -------------------------------------------------------------- #
        #  Bremsstrahlung radiation cooling (operator-split)
        # -------------------------------------------------------------- #
        if self.enable_bremsstrahlung:
            p_new, Te_pass, P_rad = self._apply_bremsstrahlung(
                rho_new, p_new, Te_pass, dt,
            )
            self._last_P_radiated = float(P_rad.sum().item())
            cell_vol = self.dx * self.dy * self.dz
            self.total_radiated_energy += float(
                P_rad.sum().item() * cell_vol * dt
            )

        # -------------------------------------------------------------- #
        #  Electrode B-field BC (circuit → MHD magnetic piston)
        # -------------------------------------------------------------- #
        anode_r = kwargs.get("anode_radius", 0.0)
        cathode_r = kwargs.get("cathode_radius", 0.0)
        apply_bc = kwargs.get("apply_electrode_bc", False)
        if apply_bc and cathode_r > 0 and abs(current) > 1e-10:
            import math
            mu0 = 4.0 * 3.141592653589793e-7
            pi = 3.141592653589793
            # For cylindrical coords, solver operates in HL; convert B_theta to HL.
            _sqrt_mu0 = math.sqrt(mu0) if self._convert_b_si_to_hl else 1.0
            nx, ny, nz = B_new.shape[1], B_new.shape[2], B_new.shape[3]

            if self.coordinates == "cartesian":
                # 3D Cartesian: decompose B_theta into (Bx, By)
                # Grid centered at domain midpoint
                x = torch.linspace(
                    -(nx / 2.0 - 0.5) * self.dx, (nx / 2.0 - 0.5) * self.dx,
                    nx, device=B_new.device, dtype=B_new.dtype,
                )
                y = torch.linspace(
                    -(ny / 2.0 - 0.5) * self.dy, (ny / 2.0 - 0.5) * self.dy,
                    ny, device=B_new.device, dtype=B_new.dtype,
                )
                X, Y = torch.meshgrid(x, y, indexing="ij")
                R = torch.sqrt(X**2 + Y**2)
                R_safe = torch.clamp(R, min=1e-10)
                sin_theta = Y / R_safe
                cos_theta = X / R_safe
                # Cathode boundary: cells within 1.5*dx of cathode_radius
                mask_cath = torch.abs(R - cathode_r) < 1.5 * self.dx
                # B_theta in HL code units (solver operates in HL internally)
                B_th = mu0 * current / (2.0 * pi * torch.clamp(R, min=cathode_r * 0.5)) / _sqrt_mu0
                # B_x = -B_theta * sin(theta), B_y = B_theta * cos(theta)
                if mask_cath.any():
                    for k in range(nz):
                        B_new[0, :, :, k] = torch.where(mask_cath, -B_th * sin_theta, B_new[0, :, :, k])
                        B_new[1, :, :, k] = torch.where(mask_cath, B_th * cos_theta, B_new[1, :, :, k])
                # Anode boundary: cells within 1.5*dx of anode_radius
                mask_anode = torch.abs(R - anode_r) < 1.5 * self.dx
                if anode_r > 0 and mask_anode.any():
                    for k in range(nz):
                        B_new[0, :, :, k] = torch.where(mask_anode, -B_th * sin_theta, B_new[0, :, :, k])
                        B_new[1, :, :, k] = torch.where(mask_anode, B_th * cos_theta, B_new[1, :, :, k])
                # z=0 face: insulator boundary (apply B_theta at all radii between anode and cathode)
                mask_z0 = (anode_r <= R) & (cathode_r >= R)
                if mask_z0.any():
                    B_new[0, :, :, 0] = torch.where(mask_z0, -B_th * sin_theta, B_new[0, :, :, 0])
                    B_new[1, :, :, 0] = torch.where(mask_z0, B_th * cos_theta, B_new[1, :, :, 0])
                # Axis guard: zero B at r < anode_r/2 to prevent singularity
                mask_axis = anode_r * 0.5 > R  # shape (nx, ny)
                if mask_axis.any():
                    mask_3d = mask_axis.unsqueeze(-1).expand(nx, ny, nz)
                    for comp in range(3):
                        B_new[comp] = torch.where(mask_3d, torch.zeros_like(B_new[comp]), B_new[comp])
            else:
                # Cylindrical: axis 0 = r, axis 1 = theta/y, axis 2 = z
                r_arr = self._r.squeeze()  # shape (nx,)
                idx_cath = int(torch.argmin(torch.abs(r_arr - cathode_r)).item())
                idx_anode = int(torch.argmin(torch.abs(r_arr - anode_r)).item())
                r_safe = torch.clamp(r_arr, min=1e-10)
                B_theta_profile = mu0 * current / (2.0 * pi * r_safe) / _sqrt_mu0

                # Energy-correcting electrode BC with relaxation.
                # Instead of hard-pinning B_theta (which creates a stationary
                # shock at the boundary that accumulates energy), relax toward
                # the target over a few timesteps. This mimics ghost-cell BCs
                # where the Riemann solver handles the transition naturally.
                B_sq_before = B_new[1].clone() ** 2

                # Relaxation rate: blend 10% per step toward target
                alpha_bc = 0.1

                # Cathode boundary
                B_target_cath = B_theta_profile[idx_cath]
                B_new[1, idx_cath, :, :] = (
                    (1.0 - alpha_bc) * B_new[1, idx_cath, :, :] + alpha_bc * B_target_cath
                )
                if idx_cath < nx - 1:
                    B_new[1, -1, :, :] = (
                        (1.0 - alpha_bc) * B_new[1, -1, :, :] + alpha_bc * B_theta_profile[-1]
                    )
                # Anode boundary
                if idx_anode > 0:
                    B_target_anode = B_theta_profile[idx_anode]
                    B_new[1, idx_anode, :, :] = (
                        (1.0 - alpha_bc) * B_new[1, idx_anode, :, :] + alpha_bc * B_target_anode
                    )
                # Insulator face (z=0): relax at all radii in gap
                for ir in range(idx_anode, min(idx_cath + 1, nx)):
                    B_new[1, ir, :, 0] = (
                        (1.0 - alpha_bc) * B_new[1, ir, :, 0] + alpha_bc * B_theta_profile[ir]
                    )

                # Energy correction
                B_sq_after = B_new[1] ** 2
                _mu0_bc = mu0 / (_sqrt_mu0 ** 2)
                delta_ME = (B_sq_after - B_sq_before) / (2.0 * _mu0_bc)
                p_new = torch.clamp(p_new + delta_ME * (self.gamma - 1.0), min=P_FLOOR)

                # Conducting wall BCs
                B_new[0, :idx_anode + 1, :, :] = 0.0
                vel_new[0, :idx_anode + 1, :, :] = 0.0
                B_new[0, idx_cath:, :, :] = 0.0
                vel_new[0, idx_cath:, :, :] = 0.0
                B_new[0, 0, :, :] = 0.0

        # -------------------------------------------------------------- #
        #  Neighbor-averaging NaN/Inf repair (before returning to engine)
        # -------------------------------------------------------------- #
        rho_new, vel_new, p_new, B_new = self._repair_nonfinite(
            rho_new, vel_new, p_new, B_new,
        )

        # -------------------------------------------------------------- #
        #  Update coupling state for the circuit solver
        # -------------------------------------------------------------- #
        self._update_coupling(B_new, current, voltage, dt)

        result = {
            "rho": rho_new,
            "velocity": vel_new,
            "pressure": p_new,
            "B": B_new,
            "Te": Te_pass,
            "Ti": Ti_pass,
            "psi": psi_pass,
        }
        if ee_new is not None:
            result["e_electron"] = ee_new
        return result

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        eta_field: np.ndarray | None = None,
        source_terms: dict[str, np.ndarray] | None = None,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        """Advance the MHD state by one timestep.

        Backwards-compatible wrapper around ``step_gpu()``.  Transfers
        NumPy state to GPU, delegates all physics to ``step_gpu()``, and
        transfers the result back to NumPy float64.

        Supports two SSP Runge-Kutta schemes (Shu & Osher 1988):

        **SSP-RK2** (2 stages, 2nd-order)::

            U^(1)   = U^n + dt * L(U^n)
            U^(n+1) = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))

        **SSP-RK3** (3 stages, 3rd-order)::

            U^(1)   = U^n + dt * L(U^n)
            U^(2)   = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
            U^(n+1) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))

        Optionally applies resistive MHD (operator-split) when ``eta_field``
        is provided: dB/dt += -curl(eta * J), dp/dt += (gamma-1)*eta*|J|^2.

        Parameters
        ----------
        state : dict[str, np.ndarray]
            State dictionary with keys:
              - ``"rho"``:      ``(nx, ny, nz)`` density
              - ``"velocity"``: ``(3, nx, ny, nz)`` velocity
              - ``"pressure"``: ``(nx, ny, nz)`` pressure
              - ``"B"``:        ``(3, nx, ny, nz)`` magnetic field
              - ``"Te"``:       ``(nx, ny, nz)`` electron temperature (passed through)
              - ``"Ti"``:       ``(nx, ny, nz)`` ion temperature (passed through)
              - ``"psi"``:      ``(nx, ny, nz)`` divergence cleaning scalar (passed through)
        dt : float
            Timestep size [s].
        current : float
            Circuit current [A].
        voltage : float
            Capacitor voltage [V].
        eta_field : np.ndarray | None
            Spatially-resolved resistivity [Ohm*m], shape (nx, ny, nz).
            If None, resistive diffusion is skipped (ideal MHD).

        Returns
        -------
        dict[str, np.ndarray]
            Updated state dictionary with the same keys.
        """
        state_gpu = self._to_device(state)

        eta_field_gpu: torch.Tensor | None = None
        if eta_field is not None:
            eta_field_gpu = torch.tensor(
                eta_field, dtype=self._dtype, device=self.device,
            )

        out_gpu = self.step_gpu(
            state_gpu, dt, current, voltage,
            eta_field_gpu=eta_field_gpu,
            source_terms=source_terms,
            **kwargs,
        )

        result = self._to_numpy(out_gpu)

        advance_nan_step_count()

        # HL → SI conversion for B (cylindrical only)
        if self._convert_b_si_to_hl:
            import math
            result["B"] = result["B"] * math.sqrt(mu_0_si)

        # Two-temperature operator-split: apply source terms on CPU.
        # Use the advected e_electron from the GPU step (result) if available,
        # otherwise fall back to the input state.
        e_electron_in = result.get("e_electron", state.get("e_electron"))
        if e_electron_in is not None:
            from dpf.fluid.two_temperature import step_electron_energy
            rho = result["rho"]
            n_i = rho / self.ion_mass
            n_i_safe = np.maximum(n_i, 1e-30)
            eta_np = eta_field if eta_field is not None else np.zeros_like(rho)
            # Reconstruct J^2 from Ohmic heating: Q_ohm = eta * J^2
            ohmic = self._last_ohmic if hasattr(self, "_last_ohmic") else np.zeros_like(rho)
            J_sq = ohmic / np.maximum(eta_np, 1e-30) if np.any(eta_np > 0) else np.zeros_like(rho)
            e_e_new, Te_new, Ti_new = step_electron_energy(
                rho_e_e=e_electron_in, rho=rho,
                velocity=result["velocity"], eta=eta_np,
                J_sq=J_sq, Te=result["Te"], Ti=result["Ti"],
                n_e=n_i_safe, n_i=n_i_safe,
                dx=self.dx, dt=dt,
                Z=1.0, gaunt_factor=1.2,
                gamma=self.gamma,
            )
            result["Te"] = np.maximum(Te_new, 1.0)
            result["Ti"] = np.maximum(Ti_new, 1.0)
            result["e_electron"] = e_e_new

        return result

    @staticmethod
    def _neighbor_average_3d(field: torch.Tensor) -> torch.Tensor:
        """Compute 6-neighbor average of a 3D field for NaN repair.

        At boundaries, uses replicate padding (copies boundary value).
        Returns a tensor where each cell contains the average of its
        6 face-neighbors (or fewer at boundaries).

        Args:
            field: 3D tensor, shape (nx, ny, nz).

        Returns:
            Neighbor-averaged tensor, same shape.
        """
        # Use conv3d with a 3x3x3 kernel that averages face neighbors
        inp = field.unsqueeze(0).unsqueeze(0)  # (1, 1, nx, ny, nz)
        padded = torch.nn.functional.pad(inp, (1, 1, 1, 1, 1, 1), mode="replicate")

        # Sum of 6 face-neighbors (no center cell)
        kernel = torch.zeros(1, 1, 3, 3, 3, device=field.device, dtype=field.dtype)
        kernel[0, 0, 1, 1, 0] = 1.0  # z-1
        kernel[0, 0, 1, 1, 2] = 1.0  # z+1
        kernel[0, 0, 1, 0, 1] = 1.0  # y-1
        kernel[0, 0, 1, 2, 1] = 1.0  # y+1
        kernel[0, 0, 0, 1, 1] = 1.0  # x-1
        kernel[0, 0, 2, 1, 1] = 1.0  # x+1

        result = torch.nn.functional.conv3d(padded, kernel, padding=0)
        return (result / 6.0).squeeze(0).squeeze(0)

    def _repair_nonfinite(
        self,
        rho: torch.Tensor,
        vel: torch.Tensor,
        p: torch.Tensor,
        B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace NaN/Inf values with neighbor-averaged values.

        Unlike floor replacement (which creates artificial discontinuities
        that regenerate NaN on the next step), neighbor averaging produces
        smooth transitions that are stable under subsequent evolution.

        This is called at the end of each Metal solver step, before the
        state is returned to the engine.
        """
        # Check for any non-finite values across all fields
        bad_rho = ~torch.isfinite(rho)
        bad_vel = ~torch.isfinite(vel)
        bad_p = ~torch.isfinite(p)
        bad_B = ~torch.isfinite(B)

        has_bad = bad_rho.any() or bad_vel.any() or bad_p.any() or bad_B.any()
        if not has_bad:
            return rho, vel, p, B

        # Replace NaN/Inf in each field before computing neighbor averages
        # (so NaN doesn't poison the averaging kernel)
        rho_clean = torch.where(bad_rho, torch.full_like(rho, RHO_FLOOR), rho)
        rho_avg = self._neighbor_average_3d(rho_clean)
        rho = torch.where(bad_rho, torch.clamp(rho_avg, min=RHO_FLOOR), rho)

        p_clean = torch.where(bad_p, torch.full_like(p, P_FLOOR), p)
        p_avg = self._neighbor_average_3d(p_clean)
        p = torch.where(bad_p, torch.clamp(p_avg, min=P_FLOOR), p)

        # Velocity: zero bad values before averaging, then repair
        for comp in range(3):
            bad_comp = bad_vel[comp]
            if bad_comp.any():
                v_clean = torch.where(bad_comp, torch.zeros_like(vel[comp]), vel[comp])
                v_avg = self._neighbor_average_3d(v_clean)
                vel = vel.clone()
                vel[comp] = torch.where(bad_comp, v_avg, vel[comp])

        # B-field: same approach
        for comp in range(3):
            bad_comp = bad_B[comp]
            if bad_comp.any():
                b_clean = torch.where(bad_comp, torch.zeros_like(B[comp]), B[comp])
                b_avg = self._neighbor_average_3d(b_clean)
                B = B.clone()
                B[comp] = torch.where(bad_comp, b_avg, B[comp])

        return rho, vel, p, B

    def _clamp_velocity(
        self,
        vel: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp velocity to 10x local fast magnetosonic speed."""
        cf = _fast_magnetosonic_mps(rho, p, B, self.gamma, dim=0)
        for d in range(1, 3):
            cf = torch.maximum(cf, _fast_magnetosonic_mps(
                rho, p, B, self.gamma, dim=d,
            ))
        v_max = torch.clamp(10.0 * cf, min=1e3, max=1e7)
        return torch.clamp(vel, min=-v_max, max=v_max)

    def _step_ssp_rk2(
        self,
        rho_n: torch.Tensor,
        vel_n: torch.Tensor,
        p_n: torch.Tensor,
        B_n: torch.Tensor,
        dt: float,
        e_electron: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """SSP-RK2 (Shu-Osher): 2 stages, 2nd-order."""
        # ── Compiled fast path (Cartesian, no e_electron) ──────────────────
        if self._compiled_euler_stage is not None and e_electron is None:
            # Stage 1: U^(1) = U^n + dt * L(U^n)
            rho_1, vel_1, p_1, B_1 = self._compiled_euler_stage(
                rho_n, vel_n, p_n, B_n, dt,
                self.gamma, self.dx, self.dy, self.dz,
                self.limiter, self.riemann_solver, self.reconstruction,
                self.bc[0], self.bc[1], self.bc[2],
            )
            if self.use_ct:
                B_1 = self._apply_ct_correction(B_1, B_n, dt)
            # Stage 2: U^(n+1) = 0.5*U^n + 0.5*(U^(1) + dt*L(U^(1)))
            rho_1b, vel_1b, p_1b, B_1b = self._compiled_euler_stage(
                rho_1, vel_1, p_1, B_1, dt,
                self.gamma, self.dx, self.dy, self.dz,
                self.limiter, self.riemann_solver, self.reconstruction,
                self.bc[0], self.bc[1], self.bc[2],
            )
            rho_new = torch.clamp(0.5 * rho_n + 0.5 * rho_1b, min=RHO_FLOOR)
            p_new = torch.clamp(0.5 * p_n + 0.5 * p_1b, min=P_FLOOR)
            vel_new = self._clamp_velocity(
                0.5 * vel_n + 0.5 * vel_1b, rho_new, p_new,
                0.5 * B_n + 0.5 * B_1b,
            )
            B_new = 0.5 * B_n + 0.5 * B_1b
            if self.use_ct:
                B_new = self._apply_ct_correction(B_new, B_n, dt)
            return rho_new, vel_new, p_new, B_new, None

        # ── Eager path (original, unchanged) ───────────────────────────────
        # Stage 1: U^(1) = U^n + dt * L(U^n)
        rhs1 = self._compute_rhs(rho_n, vel_n, p_n, B_n, e_electron=e_electron)
        rho_1, vel_1, p_1, B_1 = self._euler_update(
            rho_n, vel_n, p_n, B_n, rhs1, dt,
        )
        ee_1 = None
        if e_electron is not None:
            ee_1 = torch.clamp(e_electron + dt * rhs1["e_electron"], min=0.0)
        if self.use_ct:
            B_1 = self._apply_ct_correction(B_1, B_n, dt)

        # Stage 2: U^(n+1) = 0.5*U^n + 0.5*(U^(1) + dt*L(U^(1)))
        rhs2 = self._compute_rhs(rho_1, vel_1, p_1, B_1, e_electron=ee_1)
        rho_new = 0.5 * rho_n + 0.5 * (rho_1 + dt * rhs2["rho"])
        vel_new = 0.5 * vel_n + 0.5 * (vel_1 + dt * rhs2["velocity"])
        p_new = 0.5 * p_n + 0.5 * (p_1 + dt * rhs2["pressure"])
        B_new = 0.5 * B_n + 0.5 * (B_1 + dt * rhs2["B"])
        ee_new = None
        if e_electron is not None:
            ee_new = torch.clamp(
                0.5 * e_electron + 0.5 * (ee_1 + dt * rhs2["e_electron"]),
                min=0.0,
            )

        rho_new = torch.clamp(rho_new, min=RHO_FLOOR)
        p_new = torch.clamp(p_new, min=P_FLOOR)
        vel_new = self._clamp_velocity(vel_new, rho_new, p_new, B_new)
        if self.use_ct:
            B_new = self._apply_ct_correction(B_new, B_n, dt)

        return rho_new, vel_new, p_new, B_new, ee_new

    def _step_ssp_rk3(
        self,
        rho_n: torch.Tensor,
        vel_n: torch.Tensor,
        p_n: torch.Tensor,
        B_n: torch.Tensor,
        dt: float,
        e_electron: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """SSP-RK3 (Shu-Osher): 3 stages, 3rd-order.

        References:
            Shu C.-W. & Osher S., J. Comput. Phys. 77, 439 (1988).
            Gottlieb S. et al., SIAM Rev. 43, 89-112 (2001).
        """
        # ── Compiled fast path (Cartesian, no e_electron) ──────────────────
        if self._compiled_euler_stage is not None and e_electron is None:
            _ce = self._compiled_euler_stage
            _bc = (self.bc[0], self.bc[1], self.bc[2])
            _args = (self.gamma, self.dx, self.dy, self.dz,
                     self.limiter, self.riemann_solver, self.reconstruction,
                     *_bc)

            # Stage 1: U^(1) = U^n + dt * L(U^n)
            rho_1, vel_1, p_1, B_1 = _ce(rho_n, vel_n, p_n, B_n, dt, *_args)
            if self.use_ct:
                B_1 = self._apply_ct_correction(B_1, B_n, dt)

            # Stage 2: U^(2) = 3/4*U^n + 1/4*(U^(1) + dt*L(U^(1)))
            rho_1b, vel_1b, p_1b, B_1b = _ce(rho_1, vel_1, p_1, B_1, dt, *_args)
            rho_2 = torch.clamp(0.75 * rho_n + 0.25 * rho_1b, min=RHO_FLOOR)
            p_2 = torch.clamp(0.75 * p_n + 0.25 * p_1b, min=P_FLOOR)
            B_2 = 0.75 * B_n + 0.25 * B_1b
            vel_2 = self._clamp_velocity(
                0.75 * vel_n + 0.25 * vel_1b, rho_2, p_2, B_2,
            )
            if self.use_ct:
                B_2 = self._apply_ct_correction(B_2, B_n, dt)

            # Stage 3: U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt*L(U^(2)))
            rho_2b, vel_2b, p_2b, B_2b = _ce(rho_2, vel_2, p_2, B_2, dt, *_args)
            rho_new = torch.clamp(
                (1.0 / 3.0) * rho_n + (2.0 / 3.0) * rho_2b, min=RHO_FLOOR,
            )
            p_new = torch.clamp(
                (1.0 / 3.0) * p_n + (2.0 / 3.0) * p_2b, min=P_FLOOR,
            )
            B_new = (1.0 / 3.0) * B_n + (2.0 / 3.0) * B_2b
            vel_new = self._clamp_velocity(
                (1.0 / 3.0) * vel_n + (2.0 / 3.0) * vel_2b, rho_new, p_new, B_new,
            )
            if self.use_ct:
                B_new = self._apply_ct_correction(B_new, B_n, dt)

            return rho_new, vel_new, p_new, B_new, None

        # ── Eager path (original, unchanged) ───────────────────────────────
        # Stage 1: U^(1) = U^n + dt * L(U^n)
        rhs1 = self._compute_rhs(rho_n, vel_n, p_n, B_n, e_electron=e_electron)
        rho_1, vel_1, p_1, B_1 = self._euler_update(
            rho_n, vel_n, p_n, B_n, rhs1, dt,
        )
        ee_1 = None
        if e_electron is not None:
            ee_1 = torch.clamp(e_electron + dt * rhs1["e_electron"], min=0.0)
        if self.use_ct:
            B_1 = self._apply_ct_correction(B_1, B_n, dt)

        # Stage 2: U^(2) = 3/4*U^n + 1/4*(U^(1) + dt*L(U^(1)))
        rhs2 = self._compute_rhs(rho_1, vel_1, p_1, B_1, e_electron=ee_1)
        rho_2 = 0.75 * rho_n + 0.25 * (rho_1 + dt * rhs2["rho"])
        vel_2 = 0.75 * vel_n + 0.25 * (vel_1 + dt * rhs2["velocity"])
        p_2 = 0.75 * p_n + 0.25 * (p_1 + dt * rhs2["pressure"])
        B_2 = 0.75 * B_n + 0.25 * (B_1 + dt * rhs2["B"])
        ee_2 = None
        if e_electron is not None:
            ee_2 = torch.clamp(
                0.75 * e_electron + 0.25 * (ee_1 + dt * rhs2["e_electron"]),
                min=0.0,
            )

        rho_2 = torch.clamp(rho_2, min=RHO_FLOOR)
        p_2 = torch.clamp(p_2, min=P_FLOOR)
        vel_2 = self._clamp_velocity(vel_2, rho_2, p_2, B_2)
        if self.use_ct:
            B_2 = self._apply_ct_correction(B_2, B_n, dt)

        # Stage 3: U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt*L(U^(2)))
        rhs3 = self._compute_rhs(rho_2, vel_2, p_2, B_2, e_electron=ee_2)
        rho_new = (1.0 / 3.0) * rho_n + (2.0 / 3.0) * (rho_2 + dt * rhs3["rho"])
        vel_new = (1.0 / 3.0) * vel_n + (2.0 / 3.0) * (vel_2 + dt * rhs3["velocity"])
        p_new = (1.0 / 3.0) * p_n + (2.0 / 3.0) * (p_2 + dt * rhs3["pressure"])
        B_new = (1.0 / 3.0) * B_n + (2.0 / 3.0) * (B_2 + dt * rhs3["B"])
        ee_new = None
        if e_electron is not None:
            ee_new = torch.clamp(
                (1.0 / 3.0) * e_electron + (2.0 / 3.0) * (ee_2 + dt * rhs3["e_electron"]),
                min=0.0,
            )

        rho_new = torch.clamp(rho_new, min=RHO_FLOOR)
        p_new = torch.clamp(p_new, min=P_FLOOR)
        vel_new = self._clamp_velocity(vel_new, rho_new, p_new, B_new)
        if self.use_ct:
            B_new = self._apply_ct_correction(B_new, B_n, dt)

        return rho_new, vel_new, p_new, B_new, ee_new

    # ------------------------------------------------------------------ #
    #  Constrained transport correction
    # ------------------------------------------------------------------ #

    def _apply_ct_correction(
        self,
        B_new: torch.Tensor,
        B_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply constrained transport divergence correction to B.

        Constructs face-centred B from cell-centred values via simple
        averaging, computes EMFs from the induction equation dB/dt,
        applies the CT update to get divergence-free face B, then
        averages back to cell centres.

        This is a simplified CT approach suitable for cell-centred MHD:
        it does not eliminate div(B) exactly, but reduces it significantly
        compared to the uncorrected flux-difference update.

        Parameters
        ----------
        B_new : torch.Tensor
            Updated cell-centred B, shape ``(3, nx, ny, nz)``.
        B_old : torch.Tensor
            Previous cell-centred B, shape ``(3, nx, ny, nz)``.
        dt : float
            Timestep [s].

        Returns
        -------
        torch.Tensor
            Corrected cell-centred B, shape ``(3, nx, ny, nz)``.
        """
        nx, ny, nz = self.grid_shape

        # Skip CT if grid is too small for meaningful stencils
        if nx < 3 or ny < 3 or nz < 3:
            return B_new

        # Compute the effective induction dB/dt from the RK update
        dB_dt = (B_new - B_old) / max(dt, 1e-30)

        # Construct face-centred B_old by averaging cell centres
        # Bx_face: (nx+1, ny, nz)
        Bx_face = torch.zeros(
            nx + 1, ny, nz, dtype=self._dtype, device=self.device
        )
        Bx_face[1:-1, :, :] = 0.5 * (B_old[0, :-1, :, :] + B_old[0, 1:, :, :])
        Bx_face[0, :, :] = B_old[0, 0, :, :]
        Bx_face[-1, :, :] = B_old[0, -1, :, :]

        # By_face: (nx, ny+1, nz)
        By_face = torch.zeros(
            nx, ny + 1, nz, dtype=self._dtype, device=self.device
        )
        By_face[:, 1:-1, :] = 0.5 * (B_old[1, :, :-1, :] + B_old[1, :, 1:, :])
        By_face[:, 0, :] = B_old[1, :, 0, :]
        By_face[:, -1, :] = B_old[1, :, -1, :]

        # Bz_face: (nx, ny, nz+1)
        Bz_face = torch.zeros(
            nx, ny, nz + 1, dtype=self._dtype, device=self.device
        )
        Bz_face[:, :, 1:-1] = 0.5 * (B_old[2, :, :, :-1] + B_old[2, :, :, 1:])
        Bz_face[:, :, 0] = B_old[2, :, :, 0]
        Bz_face[:, :, -1] = B_old[2, :, :, -1]

        # Construct face-centred EMF contributions from dB/dt
        # For the simple CT scheme we need face fluxes that, when
        # differenced, reproduce dB/dt.  We use the induction equation:
        #   E = -v x B (ideal MHD).
        # As a simplified approach, construct face EMF contributions
        # directly from the cell-centred dB/dt by averaging to faces.

        # dBx/dt face contributions at x-faces
        flux_x = torch.zeros(
            nx + 1, ny, nz, dtype=self._dtype, device=self.device
        )
        flux_x[1:-1, :, :] = 0.5 * (dB_dt[0, :-1, :, :] + dB_dt[0, 1:, :, :])
        flux_x[0, :, :] = dB_dt[0, 0, :, :]
        flux_x[-1, :, :] = dB_dt[0, -1, :, :]

        # dBy/dt face contributions at y-faces
        flux_y = torch.zeros(
            nx, ny + 1, nz, dtype=self._dtype, device=self.device
        )
        flux_y[:, 1:-1, :] = 0.5 * (dB_dt[1, :, :-1, :] + dB_dt[1, :, 1:, :])
        flux_y[:, 0, :] = dB_dt[1, :, 0, :]
        flux_y[:, -1, :] = dB_dt[1, :, -1, :]

        # dBz/dt face contributions at z-faces
        flux_z = torch.zeros(
            nx, ny, nz + 1, dtype=self._dtype, device=self.device
        )
        flux_z[:, :, 1:-1] = 0.5 * (dB_dt[2, :, :, :-1] + dB_dt[2, :, :, 1:])
        flux_z[:, :, 0] = dB_dt[2, :, :, 0]
        flux_z[:, :, -1] = dB_dt[2, :, :, -1]

        # Compute edge EMFs from the face fluxes
        Ex_edge, Ey_edge, Ez_edge = emf_from_fluxes_mps(flux_x, flux_y, flux_z)

        # Apply CT update to face-centred B (r-weighted in cylindrical mode)
        if self.coordinates == "cylindrical" and self._r_face is not None:
            Bx_new_face, By_new_face, Bz_new_face = ct_update_cylindrical_mps(
                Bx_face, By_face, Bz_face,
                Ex_edge, Ey_edge, Ez_edge,
                self.dx, self.dy, self.dz, dt,
                r_cell=self._r, r_face=self._r_face,
            )
        else:
            Bx_new_face, By_new_face, Bz_new_face = ct_update_mps(
                Bx_face, By_face, Bz_face,
                Ex_edge, Ey_edge, Ez_edge,
                self.dx, self.dy, self.dz, dt,
            )

        # Average face-centred values back to cell centres
        B_corrected = torch.empty_like(B_new)
        B_corrected[0] = 0.5 * (Bx_new_face[:-1, :, :] + Bx_new_face[1:, :, :])
        B_corrected[1] = 0.5 * (By_new_face[:, :-1, :] + By_new_face[:, 1:, :])
        B_corrected[2] = 0.5 * (Bz_new_face[:, :, :-1] + Bz_new_face[:, :, 1:])

        return B_corrected

    # ------------------------------------------------------------------ #
    #  Circuit coupling
    # ------------------------------------------------------------------ #

    def _update_coupling(
        self,
        B: torch.Tensor,
        current: float,
        voltage: float,
        dt: float,
    ) -> None:
        """Update the plasma-circuit coupling state.

        Estimates plasma inductance from the mean azimuthal B-field
        and computes dL/dt for the circuit solver.

        Parameters
        ----------
        B : torch.Tensor
            Cell-centred B field, shape ``(3, nx, ny, nz)``.
        current : float
            Circuit current [A].
        voltage : float
            Capacitor voltage [V].
        dt : float
            Timestep [s].
        """
        # Estimate B_theta ~ sqrt(Bx^2 + By^2) averaged over domain
        B_theta_sq = B[0] ** 2 + B[1] ** 2
        B_theta_avg = float(torch.mean(torch.sqrt(B_theta_sq)).item())

        nx = self.grid_shape[0]

        # Lp = Φ/I = B_theta_avg * A / I, where A = radial_extent * axial_length
        # B_theta already contains µ₀ (since B = µ₀I/(2πr)), so no extra µ₀ factor
        if abs(current) > 0.0:
            radial_extent = self.dx * nx
            axial_length = (
                self.dx * self.grid_shape[2]
                if len(self.grid_shape) > 2
                else radial_extent
            )
            Lp_est = B_theta_avg * radial_extent * axial_length / (abs(current) + 1e-30)
        else:
            Lp_est = 0.0

        # Compute dL/dt from previous Lp; None signals RLC solver to use BDF2 internally
        if self._prev_Lp is not None and dt > 0.0:
            dL_dt: float | None = (Lp_est - self._prev_Lp) / dt
        else:
            dL_dt = None
        self._prev_Lp = Lp_est

        self._coupling = CouplingState(
            Lp=Lp_est,
            current=current,
            voltage=voltage,
            dL_dt=dL_dt,
        )

    # ------------------------------------------------------------------ #
    #  PlasmaSolverBase interface
    # ------------------------------------------------------------------ #

    def coupling_interface(self) -> CouplingState:
        """Return coupling quantities for the circuit solver.

        Returns
        -------
        CouplingState
            Current plasma-circuit coupling data.
        """
        return self._coupling

    # ------------------------------------------------------------------ #
    #  CFL query (public)
    # ------------------------------------------------------------------ #

    def compute_dt_gpu(self, state_gpu: dict[str, torch.Tensor]) -> float:
        """Compute the CFL-limited timestep from a GPU-resident state.

        Calls ``_compute_dt_cfl()`` directly without any NumPy→GPU transfer.
        Use this when state is already on the device (e.g., inside a
        GPU-resident multi-step loop).

        Parameters
        ----------
        state_gpu : dict[str, torch.Tensor]
            Current simulation state on ``self.device``.

        Returns
        -------
        float
            CFL-limited timestep [s].
        """
        return self._compute_dt_cfl(state_gpu)

    def compute_dt(self, state: dict[str, np.ndarray]) -> float:
        """Compute the CFL-limited timestep for the given state.

        Backwards-compatible wrapper around ``compute_dt_gpu()``.  Transfers
        state to GPU, then delegates to ``_compute_dt_cfl()``.

        Parameters
        ----------
        state : dict[str, np.ndarray]
            Current simulation state.

        Returns
        -------
        float
            CFL-limited timestep [s].
        """
        return self.compute_dt_gpu(self._to_device(state))

    # Alias for engine.py compatibility (uses _compute_dt internally)
    _compute_dt = compute_dt

    # ------------------------------------------------------------------ #
    #  Diagnostics helpers
    # ------------------------------------------------------------------ #

    def divergence_B(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Compute cell-centred div(B) for diagnostics.

        Constructs face-centred B from cell-centred values and computes
        the divergence using the stencil module.

        Parameters
        ----------
        state : dict[str, np.ndarray]
            Must contain ``"B"`` with shape ``(3, nx, ny, nz)``.

        Returns
        -------
        np.ndarray
            Divergence of B, shape ``(nx, ny, nz)``.
        """
        B = torch.as_tensor(
            state["B"], dtype=self._dtype
        ).to(self.device)

        nx, ny, nz = self.grid_shape

        # Build face-centred B by averaging cell centres
        Bx_face = torch.zeros(
            nx + 1, ny, nz, dtype=self._dtype, device=self.device
        )
        Bx_face[1:-1, :, :] = 0.5 * (B[0, :-1, :, :] + B[0, 1:, :, :])
        Bx_face[0, :, :] = B[0, 0, :, :]
        Bx_face[-1, :, :] = B[0, -1, :, :]

        By_face = torch.zeros(
            nx, ny + 1, nz, dtype=self._dtype, device=self.device
        )
        By_face[:, 1:-1, :] = 0.5 * (B[1, :, :-1, :] + B[1, :, 1:, :])
        By_face[:, 0, :] = B[1, :, 0, :]
        By_face[:, -1, :] = B[1, :, -1, :]

        Bz_face = torch.zeros(
            nx, ny, nz + 1, dtype=self._dtype, device=self.device
        )
        Bz_face[:, :, 1:-1] = 0.5 * (B[2, :, :, :-1] + B[2, :, :, 1:])
        Bz_face[:, :, 0] = B[2, :, :, 0]
        Bz_face[:, :, -1] = B[2, :, :, -1]

        div = div_B_mps(Bx_face, By_face, Bz_face, self.dx, self.dy, self.dz)
        return div.detach().cpu().to(torch.float64).numpy()

    # ------------------------------------------------------------------ #
    #  String representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"MetalMHDSolver(grid={self.grid_shape}, dx={self.dx:.3e}, "
            f"gamma={self.gamma:.4f}, cfl={self.cfl:.2f}, "
            f"device={self.device}, limiter={self.limiter!r}, "
            f"use_ct={self.use_ct}, riemann={self.riemann_solver!r}, "
            f"recon={self.reconstruction!r}, "
            f"time={self.time_integrator!r}, "
            f"coords={self.coordinates!r})"
        )
