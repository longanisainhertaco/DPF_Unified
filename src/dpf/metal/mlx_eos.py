"""MLX-native equation of state with Saha ionization.

Provides GPU-accelerated Z_bar(T) computation for deuterium plasma,
replacing the hardcoded Z=1 assumption in the MLX solver pipeline.

For hydrogen/deuterium (Z_max=1), the Saha equation gives Z_bar in [0,1]:
    Z_bar = S / (1 + S)
    S = (2*pi*m_e*kB*T/h^2)^{3/2} * (2/n_e) * exp(-E_ion/kB*T)

At T < 1 eV (~11,600 K): Z_bar ~ 0 (neutral)
At T > 3 eV (~35,000 K): Z_bar ~ 1 (fully ionized)

Uses a pre-computed lookup table for fast interpolation on GPU.

References:
    NRL Plasma Formulary (2019), p.28 — Saha equation
    Griem, "Principles of Plasma Spectroscopy" (1997) — LTE validity
"""

from __future__ import annotations

import math

import numpy as np

from dpf.metal.mlx_device import HAS_MLX

if HAS_MLX:
    import mlx.core as mx

# ── Physical constants ────────────────────────────────────────────
_K_B = 1.380649e-23       # Boltzmann [J/K]
_M_E = 9.10938e-31        # Electron mass [kg]
_H = 6.62607015e-34       # Planck [J s]
_E_ION_H = 13.6 * 1.602176634e-19  # Hydrogen ionization energy [J]
_M_D = 3.34358377e-27     # Deuterium mass [kg]


def _saha_zbar_numpy(T: np.ndarray, n_e: float) -> np.ndarray:
    """Compute Z_bar for deuterium via Saha equation (NumPy).

    Args:
        T: Temperature array [K].
        n_e: Reference electron density [m^-3] for Saha parameter.

    Returns:
        Z_bar array in [0, 1].
    """
    kT = _K_B * np.maximum(T, 1.0)
    thermal = (2.0 * math.pi * _M_E * kT / (_H * _H)) ** 1.5
    exponent = np.clip(-_E_ION_H / kT, -500.0, 0.0)
    S = thermal * 2.0 * np.exp(exponent) / max(n_e, 1.0)
    return S / (1.0 + S)


class SahaEOS:
    """Pre-computed Saha Z_bar lookup table for MLX.

    Generates a 1D Z_bar(log10_T) table at initialization and provides
    fast interpolation via mx.searchsorted + linear interp.

    Args:
        n_e_ref: Reference electron density [m^-3] for the Saha parameter.
            Typical DPF fill: ~1e22. The table is mildly sensitive to n_e
            because the Saha equation depends on density, but for hydrogen
            the transition is narrow enough that a single reference n_e
            gives Z_bar accurate to ~10% across 2 orders of magnitude.
        T_min: Minimum temperature [K] (default 100).
        T_max: Maximum temperature [K] (default 1e9).
        n_points: Number of table points (default 500).
        ion_mass: Ion mass [kg] (default deuterium).
    """

    def __init__(
        self,
        n_e_ref: float = 1e22,
        T_min: float = 100.0,
        T_max: float = 1e9,
        n_points: int = 500,
        ion_mass: float = _M_D,
    ) -> None:
        self.ion_mass = ion_mass
        self.n_e_ref = n_e_ref

        # Build table in log10(T) space
        log_T = np.linspace(np.log10(T_min), np.log10(T_max), n_points)
        T_arr = 10.0 ** log_T
        Z_arr = _saha_zbar_numpy(T_arr, n_e_ref)

        # Store as MLX arrays for GPU interpolation
        if HAS_MLX:
            self._log_T = mx.array(log_T.astype(np.float32))
            self._Z_table = mx.array(Z_arr.astype(np.float32))
        self._log_T_np = log_T
        self._Z_table_np = Z_arr
        self._log_T_min = float(log_T[0])
        self._log_T_max = float(log_T[-1])
        self._n_points = n_points

    def zbar_mlx(self, T: mx.array) -> mx.array:
        """Look up Z_bar from temperature using MLX arrays.

        Linear interpolation in log10(T) space.

        Args:
            T: Temperature array [K], any shape.

        Returns:
            Z_bar array, same shape, in [0, 1].
        """
        log_T = mx.log10(mx.maximum(T, 1.0))
        log_T = mx.clip(log_T, self._log_T_min, self._log_T_max)

        # Normalized position in table [0, n_points-1]
        t = (log_T - self._log_T_min) / (self._log_T_max - self._log_T_min)
        t = t * (self._n_points - 1)

        # Integer indices and fractional part
        idx = mx.clip(mx.floor(t).astype(mx.int32), 0, self._n_points - 2)
        frac = t - idx.astype(mx.float32)

        # Linear interpolation
        Z_lo = self._Z_table[idx]
        Z_hi = self._Z_table[idx + 1]
        return Z_lo + frac * (Z_hi - Z_lo)

    def zbar_numpy(self, T: np.ndarray) -> np.ndarray:
        """Look up Z_bar from temperature using NumPy arrays.

        Args:
            T: Temperature array [K].

        Returns:
            Z_bar array in [0, 1].
        """
        log_T = np.clip(np.log10(np.maximum(T, 1.0)), self._log_T_min, self._log_T_max)
        return np.interp(log_T, self._log_T_np, self._Z_table_np)

    def temperature_from_pressure(
        self, rho: mx.array, p: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Compute temperature and Z_bar from (rho, p) with Saha correction.

        Iterates once: T0 = p*m_i/(2*rho*kB) assuming Z=1,
        then Z = Saha(T0), T1 = p*m_i/((1+Z)*rho*kB).

        For deuterium, one iteration is sufficient because Z transitions
        sharply from 0 to 1 over a narrow temperature range.

        Args:
            rho: Mass density [kg/m^3].
            p: Pressure [Pa].

        Returns:
            Tuple (T, Z_bar), each same shape as input.
        """
        rho_safe = mx.maximum(rho, 1e-12)
        p_safe = mx.maximum(p, 1e-12)
        n_i = rho_safe / self.ion_mass

        # Initial guess: Z=1 (fully ionized)
        T0 = p_safe / (2.0 * n_i * _K_B)

        # Saha correction
        Z_bar = self.zbar_mlx(T0)
        T_corrected = p_safe / ((1.0 + Z_bar) * n_i * _K_B)

        return T_corrected, Z_bar

    def ionization_energy_sink(
        self, rho: mx.array, Z_bar_new: mx.array, Z_bar_old: mx.array, dt: float
    ) -> mx.array:
        """Compute energy sink from ionization [W/m^3].

        When neutral gas ionizes, energy E_ion per particle is absorbed.
        dE/dt = n_i * dZ/dt * E_ion

        Args:
            rho: Mass density [kg/m^3].
            Z_bar_new: Current ionization state.
            Z_bar_old: Previous ionization state.
            dt: Time step [s].

        Returns:
            Volumetric energy sink [W/m^3] (positive = energy absorbed).
        """
        n_i = mx.maximum(rho, 1e-12) / self.ion_mass
        dZ = Z_bar_new - Z_bar_old
        # Only sink when ionizing (dZ > 0); recombination releases energy
        # but we ignore that for now (conservative: overestimates cooling)
        return n_i * mx.maximum(dZ, 0.0) * _E_ION_H / max(dt, 1e-30)
