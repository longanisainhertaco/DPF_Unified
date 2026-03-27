"""State dict <-> packed mx.array conversion for cylindrical MHD.

The MLX solver internally uses a packed (10, nr, nz) mx.array:
  [rho, rho*vr, rho*vz, rho*vtheta, E, S*rho, Br, Bz, Btheta, e_electron]

The DPF engine uses dict[str, np.ndarray] with keys:
  {rho, velocity, pressure, B, Te, Ti, psi}

MLXState handles the conversion, including:
  - Primitive -> conserved variable transformation (pack)
  - Conserved -> primitive transformation (unpack)
  - Zero-copy NumPy <-> MLX transfer on Apple Silicon unified memory
  - Unit conversion (SI -> Heaviside-Lorentz for B fields)
"""

from __future__ import annotations

import math

import numpy as np

from dpf.metal.mlx_device import require_mlx

# -- Physical constants --
MU0: float = 4.0 * math.pi * 1e-7
from dpf.metal.constants import K_B as _K_B  # noqa: E402
_M_DEUTERIUM: float = 3.34358377e-27

# -- Numerical floors (from single source of truth) --
from dpf.metal.constants import P_FLOOR, RHO_FLOOR  # noqa: E402

# -- Dual-energy switching thresholds --
_ETA1: float = 1e-5
_ETA2: float = 1e-2

# -- Variable index constants (conserved state vector) --
NVAR: int = 10
IDN: int = 0   # density                rho
IMR: int = 1   # radial momentum        rho * vr
IMZ: int = 2   # axial momentum         rho * vz
IMT: int = 3   # azimuthal momentum     rho * vtheta
IEN: int = 4   # total energy           p/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2
ISR: int = 5   # entropy tracer         p * rho^(1-gamma)
IBR: int = 6   # radial B-field         Br
IBZ: int = 7   # axial B-field          Bz
IBT: int = 8   # azimuthal B-field      Btheta
IEE: int = 9   # electron energy density

# DPF state dict keys consumed/produced
DPF_KEYS: tuple[str, ...] = ("rho", "velocity", "pressure", "B", "Te", "Ti", "psi")


class MLXState:
    """Manages conversion between DPF state dicts and packed MLX arrays.

    Parameters
    ----------
    nr : int
        Number of x/radial cells.
    ny : int
        Number of y cells (1 for cylindrical).
    nz : int
        Number of z/axial cells.
    gamma : float
        Adiabatic index (default 5/3).
    ion_mass : float
        Ion mass [kg] for temperature derivation (default deuterium).
    coordinates : str
        ``"cylindrical"`` or ``"cartesian"``.

    Attributes
    ----------
    U : mx.array
        Packed conserved state, float32.
        Cylindrical: shape (10, nr, nz).
        Cartesian: shape (10, nx, ny, nz).
    """

    def __init__(
        self,
        nr: int,
        nz: int,
        gamma: float = 5.0 / 3.0,
        ion_mass: float = _M_DEUTERIUM,
        ny: int = 1,
        coordinates: str = "cylindrical",
    ) -> None:
        mx = require_mlx()
        self.nr = nr
        self.ny = ny
        self.nz = nz
        self.gamma = gamma
        self.ion_mass = ion_mass
        self.coordinates = coordinates
        if coordinates == "cartesian":
            self.U: object = mx.zeros((NVAR, nr, ny, nz), dtype=mx.float32)
        else:
            self.U = mx.zeros((NVAR, nr, nz), dtype=mx.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_state_dict(
        self,
        state: dict[str, np.ndarray],
        convert_b_si_to_hl: bool = False,
    ) -> object:
        """Pack a DPF state dict into conserved (10, nr, nz) mx.array.

        Performs primitive -> conserved conversion:
          rho*v = rho * velocity
          E = p/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2
          S*rho = p * rho^(1-gamma)  [entropy tracer]

        Parameters
        ----------
        state : dict[str, np.ndarray]
            DPF state dict with keys: rho, velocity, pressure, B.
            Optional keys: Te, Ti, psi (psi ignored; Te used for IEE).
        convert_b_si_to_hl : bool
            If True, divide B by sqrt(mu_0) to convert from SI to
            Heaviside-Lorentz units (mu_0=1).

        Returns
        -------
        mx.array
            Packed conserved state, shape (10, nr, nz), float32.
        """
        mx = require_mlx()

        if self.coordinates == "cartesian":
            return self._from_state_dict_cartesian(state, convert_b_si_to_hl)

        # -- Extract and squeeze ny=1 dimension --
        rho_np = np.ascontiguousarray(state["rho"][:, 0, :].astype(np.float32))
        vel_np = state["velocity"].astype(np.float32)
        vr_np  = np.ascontiguousarray(vel_np[0, :, 0, :])
        vz_np  = np.ascontiguousarray(vel_np[1, :, 0, :])
        vt_np  = np.ascontiguousarray(vel_np[2, :, 0, :])
        p_np   = np.ascontiguousarray(state["pressure"][:, 0, :].astype(np.float32))
        B_np   = state["B"].astype(np.float32)
        Br_np  = np.ascontiguousarray(B_np[0, :, 0, :])
        Bz_np  = np.ascontiguousarray(B_np[1, :, 0, :])
        Bt_np  = np.ascontiguousarray(B_np[2, :, 0, :])

        if convert_b_si_to_hl:
            sqrt_mu0 = math.sqrt(MU0)
            Br_np = Br_np / sqrt_mu0
            Bz_np = Bz_np / sqrt_mu0
            Bt_np = Bt_np / sqrt_mu0

        # -- Transfer to MLX --
        rho = self.zero_copy_to_mlx(rho_np)
        vr  = self.zero_copy_to_mlx(vr_np)
        vz  = self.zero_copy_to_mlx(vz_np)
        vt  = self.zero_copy_to_mlx(vt_np)
        p   = self.zero_copy_to_mlx(p_np)
        Br  = self.zero_copy_to_mlx(Br_np)
        Bz  = self.zero_copy_to_mlx(Bz_np)
        Bt  = self.zero_copy_to_mlx(Bt_np)

        # Apply floors
        rho = mx.maximum(rho, RHO_FLOOR)
        p   = mx.maximum(p,   P_FLOOR)

        # -- Compute conserved variables --
        KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
        ME = 0.5 * (Br * Br + Bz * Bz + Bt * Bt)
        gm1 = self.gamma - 1.0

        E    = p / gm1 + KE + ME
        Srho = self.entropy_from_primitives(rho, p)

        # -- Electron energy from Te (optional) --
        if "Te" in state and state["Te"] is not None:
            Te_np = np.ascontiguousarray(state["Te"][:, 0, :].astype(np.float32))
            Te = self.zero_copy_to_mlx(Te_np)
            e_elec = 0.5 * rho * (_K_B / self.ion_mass) * Te
        else:
            e_elec = mx.zeros((self.nr, self.nz), dtype=mx.float32)

        # -- Pack into (NVAR, nr, nz) --
        U = mx.stack([rho, rho * vr, rho * vz, rho * vt, E, Srho, Br, Bz, Bt, e_elec], axis=0)
        self.U = U
        return U

    def _from_state_dict_cartesian(
        self,
        state: dict[str, np.ndarray],
        convert_b_si_to_hl: bool = False,
    ) -> object:
        """Pack a DPF state dict into conserved (10, nx, ny, nz) mx.array."""
        mx = require_mlx()
        import math as _math

        rho_np = np.ascontiguousarray(state["rho"].astype(np.float32))
        vel_np = state["velocity"].astype(np.float32)
        vx_np = np.ascontiguousarray(vel_np[0])
        vy_np = np.ascontiguousarray(vel_np[1])
        vz_np = np.ascontiguousarray(vel_np[2])
        p_np = np.ascontiguousarray(state["pressure"].astype(np.float32))
        B_np = state["B"].astype(np.float32)
        Bx_np = np.ascontiguousarray(B_np[0])
        By_np = np.ascontiguousarray(B_np[1])
        Bz_np = np.ascontiguousarray(B_np[2])

        if convert_b_si_to_hl:
            sqrt_mu0 = _math.sqrt(MU0)
            Bx_np = Bx_np / sqrt_mu0
            By_np = By_np / sqrt_mu0
            Bz_np = Bz_np / sqrt_mu0

        rho = self.zero_copy_to_mlx(rho_np)
        vx = self.zero_copy_to_mlx(vx_np)
        vy = self.zero_copy_to_mlx(vy_np)
        vz = self.zero_copy_to_mlx(vz_np)
        p = self.zero_copy_to_mlx(p_np)
        Bx = self.zero_copy_to_mlx(Bx_np)
        By = self.zero_copy_to_mlx(By_np)
        Bz_ = self.zero_copy_to_mlx(Bz_np)

        rho = mx.maximum(rho, RHO_FLOOR)
        p = mx.maximum(p, P_FLOOR)

        KE = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
        ME = 0.5 * (Bx * Bx + By * By + Bz_ * Bz_)
        gm1 = self.gamma - 1.0
        E = p / gm1 + KE + ME
        Srho = self.entropy_from_primitives(rho, p)

        if "Te" in state and state["Te"] is not None:
            Te_np = np.ascontiguousarray(state["Te"].astype(np.float32))
            Te = self.zero_copy_to_mlx(Te_np)
            e_elec = 0.5 * rho * (_K_B / self.ion_mass) * Te
        else:
            e_elec = mx.zeros_like(rho)

        U = mx.stack([rho, rho * vx, rho * vy, rho * vz,
                       E, Srho, Bx, By, Bz_, e_elec], axis=0)
        self.U = U
        return U

    def _to_state_dict_cartesian(
        self,
        U: object,
        convert_b_hl_to_si: bool = False,
    ) -> dict[str, np.ndarray]:
        """Unpack conserved (10, nx, ny, nz) to DPF state dict."""
        mx = require_mlx()
        import math as _math

        gm1 = self.gamma - 1.0
        rho = mx.maximum(U[IDN], RHO_FLOOR)
        inv_rho = 1.0 / rho
        vx = U[IMR] * inv_rho
        vy = U[IMZ] * inv_rho
        vz = U[IMT] * inv_rho
        Bx = U[IBR]
        By = U[IBZ]
        Bz_ = U[IBT]

        KE = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
        ME = 0.5 * (Bx * Bx + By * By + Bz_ * Bz_)
        p_E = gm1 * (U[IEN] - KE - ME)

        Srho = U[ISR]
        p_S = Srho * mx.power(mx.maximum(rho, RHO_FLOOR), gm1)
        E_abs = mx.maximum(mx.abs(U[IEN]), 1e-30)
        eta = mx.abs(p_S) / E_abs
        t = mx.clip((eta - _ETA1) / max(_ETA2 - _ETA1, 1e-30), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        p = mx.maximum(w * p_E + (1.0 - w) * p_S, P_FLOOR)

        T_ion = p * self.ion_mass / (2.0 * rho * _K_B)
        Ti = mx.maximum(T_ion, 0.0)
        Te = Ti

        if convert_b_hl_to_si:
            sqrt_mu0 = _math.sqrt(MU0)
            Bx = Bx * sqrt_mu0
            By = By * sqrt_mu0
            Bz_ = Bz_ * sqrt_mu0

        def _np64(arr: object) -> np.ndarray:
            return np.array(arr, copy=False).astype(np.float64)

        rho_out = _np64(rho)
        velocity = np.stack([_np64(vx), _np64(vy), _np64(vz)], axis=0)
        B = np.stack([_np64(Bx), _np64(By), _np64(Bz_)], axis=0)

        return {
            "rho": rho_out,
            "velocity": velocity,
            "pressure": _np64(p),
            "B": B,
            "Te": _np64(Te),
            "Ti": _np64(Ti),
            "psi": np.zeros_like(rho_out),
        }

    def to_state_dict(
        self,
        U: object,
        convert_b_hl_to_si: bool = False,
    ) -> dict[str, np.ndarray]:
        """Unpack conserved (10, nr, nz) mx.array to DPF state dict.

        Uses dual-energy pressure recovery: blends entropy-derived pressure
        (p_S = S*rho * rho^(gamma-1)) with total-energy-derived pressure
        (p_E = (gamma-1)*(E - KE - ME)) based on eta = p_S / |E|.

        Parameters
        ----------
        U : mx.array
            Packed conserved state, shape (10, nr, nz), float32.
        convert_b_hl_to_si : bool
            If True, multiply B by sqrt(mu_0) to convert from
            Heaviside-Lorentz units back to SI.

        Returns
        -------
        dict[str, np.ndarray]
            DPF state dict with float64 NumPy arrays, ny=1 restored.
        """
        mx = require_mlx()

        if self.coordinates == "cartesian":
            return self._to_state_dict_cartesian(U, convert_b_hl_to_si)

        gm1 = self.gamma - 1.0

        # -- Unpack each component --
        rho  = mx.maximum(U[IDN], RHO_FLOOR)
        vr   = U[IMR] / rho
        vz   = U[IMZ] / rho
        vt   = U[IMT] / rho
        E    = U[IEN]
        Srho = U[ISR]
        Br  = U[IBR]
        Bz_ = U[IBZ]
        Bt  = U[IBT]
        # IEE (electron energy) is preserved in the packed array but not
        # back-projected to Te in the single-fluid path; Te = Ti is derived from p.

        # -- Pressure via dual-energy --
        KE = 0.5 * rho * (vr * vr + vz * vz + vt * vt)
        ME = 0.5 * (Br * Br + Bz_ * Bz_ + Bt * Bt)

        p_E = gm1 * (E - KE - ME)
        p_S = Srho * mx.power(mx.maximum(rho, RHO_FLOOR), gm1)

        E_abs = mx.maximum(mx.abs(E), 1e-30)
        eta   = mx.abs(p_S) / E_abs

        # Smoothstep blend weight (0 = entropy, 1 = total-energy)
        t = mx.clip((eta - _ETA1) / max(_ETA2 - _ETA1, 1e-30), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)

        p = mx.maximum(w * p_E + (1.0 - w) * p_S, P_FLOOR)

        # -- Temperatures --
        # Fully ionized: p = (n_e + n_i) * kB * T = 2 * n_i * kB * T (Z=1)
        # So T = p * m_i / (2 * rho * kB)
        T_ion = p * self.ion_mass / (2.0 * rho * _K_B)
        Ti = mx.maximum(T_ion, 0.0)
        Te = Ti  # single-fluid default; IEE overrides if non-zero

        # -- B-field unit conversion --
        if convert_b_hl_to_si:
            sqrt_mu0 = math.sqrt(MU0)
            Br   = Br   * sqrt_mu0
            Bz_  = Bz_  * sqrt_mu0
            Bt   = Bt   * sqrt_mu0

        # -- Collect NumPy arrays and restore ny=1 dimension --
        def _np64(arr: object) -> np.ndarray:
            """MLX array -> float64 NumPy, shape (nr, nz) -> (nr, 1, nz)."""
            out = np.array(arr, copy=False).astype(np.float64)
            return out[:, np.newaxis, :]

        rho_out  = _np64(rho)
        vr_out   = _np64(vr)
        vz_out   = _np64(vz)
        vt_out   = _np64(vt)
        p_out    = _np64(p)
        Br_out   = _np64(Br)
        Bz_out   = _np64(Bz_)
        Bt_out   = _np64(Bt)
        Ti_out   = _np64(Ti)
        Te_out   = _np64(Te)

        # velocity: (3, nr, 1, nz)
        velocity = np.stack([vr_out, vz_out, vt_out], axis=0)
        # B: (3, nr, 1, nz)
        B = np.stack([Br_out, Bz_out, Bt_out], axis=0)

        return {
            "rho":      rho_out,
            "velocity": velocity,
            "pressure": p_out,
            "B":        B,
            "Te":       Te_out,
            "Ti":       Ti_out,
            "psi":      np.zeros_like(rho_out),
        }

    def entropy_from_primitives(
        self,
        rho: object,
        p: object,
    ) -> object:
        """Compute entropy tracer S*rho = p * rho^(1-gamma).

        Parameters
        ----------
        rho : mx.array
            Density, shape (nr, nz).
        p : mx.array
            Pressure, shape (nr, nz).

        Returns
        -------
        mx.array
            S * rho, shape (nr, nz).
        """
        mx = require_mlx()
        rho_safe = mx.maximum(rho, RHO_FLOOR)
        return p * mx.power(rho_safe, 1.0 - self.gamma)

    @staticmethod
    def zero_copy_to_mlx(arr: np.ndarray) -> object:
        """Transfer NumPy array to MLX with zero-copy on Apple Silicon.

        For zero-copy to occur the array must be float32 and C-contiguous.
        Non-conforming arrays are converted before transfer (one copy).

        Parameters
        ----------
        arr : np.ndarray
            Input array.  Float32, C-contiguous path is zero-copy.

        Returns
        -------
        mx.array
        """
        mx = require_mlx()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return mx.array(arr)

    @staticmethod
    def zero_copy_to_numpy(arr: object) -> np.ndarray:
        """Transfer MLX array to NumPy via np.array().

        On Apple Silicon unified memory this is effectively zero-copy
        (no device transfer needed).

        Parameters
        ----------
        arr : mx.array
            Source MLX array.

        Returns
        -------
        np.ndarray
        """
        return np.array(arr, copy=False)
