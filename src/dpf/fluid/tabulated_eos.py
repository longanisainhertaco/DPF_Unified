"""Tabulated equation of state with bilinear interpolation.

Supports loading pre-computed EOS tables mapping (rho, T) to (p, e, Z)
for multi-material DPF simulations. Falls back to ideal gas EOS when no
table is loaded.

Tables are stored in log10 space for density and temperature, which
provides much better interpolation accuracy across the wide dynamic range
of DPF plasmas (densities spanning ~10 orders of magnitude, temperatures
from ~100 K to ~1 GK).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from dpf.constants import k_B, m_d

# ============================================================
# Data structure for a single material's EOS table
# ============================================================


@dataclass
class EOSTable:
    """Pre-computed EOS table for a single material.

    All thermodynamic quantities are tabulated on a regular grid in
    (log10(rho), log10(T)) space.

    Attributes:
        material: Material identifier (e.g. 'deuterium', 'copper').
        log_rho: 1D array of log10(rho [kg/m^3]) grid points.
        log_T: 1D array of log10(T [K]) grid points.
        pressure: 2D pressure table [Pa], shape (n_rho, n_T).
        energy: 2D specific internal energy table [J/kg], shape (n_rho, n_T).
        ionization: 2D mean ionization Z table, shape (n_rho, n_T), or None.
    """

    material: str
    log_rho: np.ndarray
    log_T: np.ndarray
    pressure: np.ndarray
    energy: np.ndarray
    ionization: np.ndarray | None


# ============================================================
# Numba-accelerated bilinear interpolation in log space
# ============================================================


@njit(cache=True)
def _bilinear_interp(
    log_rho_grid: np.ndarray,
    log_T_grid: np.ndarray,
    table: np.ndarray,
    log_rho_val: np.ndarray,
    log_T_val: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation on a regular (log rho, log T) grid.

    Clamps input values to the table bounds so that extrapolation never
    occurs -- out-of-range queries return the nearest boundary value.

    Args:
        log_rho_grid: Sorted 1D grid of log10(rho) values, length n_rho.
        log_T_grid: Sorted 1D grid of log10(T) values, length n_T.
        table: 2D data array, shape (n_rho, n_T).
        log_rho_val: 1D array of query log10(rho) values.
        log_T_val: 1D array of query log10(T) values.

    Returns:
        1D array of interpolated values, same length as the query arrays.
    """
    n_rho = len(log_rho_grid)
    n_T = len(log_T_grid)
    n_pts = len(log_rho_val)
    result = np.empty(n_pts)

    rho_min = log_rho_grid[0]
    rho_max = log_rho_grid[n_rho - 1]
    T_min = log_T_grid[0]
    T_max = log_T_grid[n_T - 1]

    for k in range(n_pts):
        # Clamp to grid bounds
        lr = log_rho_val[k]
        lt = log_T_val[k]
        if lr < rho_min:
            lr = rho_min
        elif lr > rho_max:
            lr = rho_max
        if lt < T_min:
            lt = T_min
        elif lt > T_max:
            lt = T_max

        # Find bracketing indices for rho via binary search
        i_lo = 0
        i_hi = n_rho - 1
        while i_hi - i_lo > 1:
            i_mid = (i_lo + i_hi) // 2
            if log_rho_grid[i_mid] <= lr:
                i_lo = i_mid
            else:
                i_hi = i_mid
        # i_lo is the lower bracket index
        if i_lo >= n_rho - 1:
            i_lo = n_rho - 2

        # Find bracketing indices for T via binary search
        j_lo = 0
        j_hi = n_T - 1
        while j_hi - j_lo > 1:
            j_mid = (j_lo + j_hi) // 2
            if log_T_grid[j_mid] <= lt:
                j_lo = j_mid
            else:
                j_hi = j_mid
        if j_lo >= n_T - 1:
            j_lo = n_T - 2

        # Interpolation weights
        dr = log_rho_grid[i_lo + 1] - log_rho_grid[i_lo]
        dt = log_T_grid[j_lo + 1] - log_T_grid[j_lo]

        # Guard against zero-width bins (degenerate grid)
        if dr < 1e-300:
            t_rho = 0.0
        else:
            t_rho = (lr - log_rho_grid[i_lo]) / dr

        if dt < 1e-300:
            t_T = 0.0
        else:
            t_T = (lt - log_T_grid[j_lo]) / dt

        # Bilinear combination
        f00 = table[i_lo, j_lo]
        f10 = table[i_lo + 1, j_lo]
        f01 = table[i_lo, j_lo + 1]
        f11 = table[i_lo + 1, j_lo + 1]

        result[k] = (
            f00 * (1.0 - t_rho) * (1.0 - t_T)
            + f10 * t_rho * (1.0 - t_T)
            + f01 * (1.0 - t_rho) * t_T
            + f11 * t_rho * t_T
        )

    return result


# ============================================================
# Main tabulated EOS class
# ============================================================


class TabulatedEOS:
    """Tabulated equation of state with bilinear interpolation.

    Supports loading pre-computed EOS tables mapping (rho, T) to (p, e, Z).
    Falls back to ideal gas EOS if no table is loaded for the queried
    material.

    The ideal-gas fallback uses:
        p = (1 + Z) * (rho / m_i) * k_B * T
        e = (1 / (gamma - 1)) * (1 + Z) * k_B * T / m_i

    Examples:
        >>> eos = TabulatedEOS(gamma=5/3)
        >>> eos.generate_ideal_table('deuterium', ion_mass=3.34e-27, Z=1)
        >>> p = eos.pressure(np.array([1e-3]), np.array([1e6]))
    """

    def __init__(self, gamma: float = 5.0 / 3.0) -> None:
        self.gamma = gamma
        self._tables: dict[str, EOSTable] = {}
        self._has_table = False

    # ----------------------------------------------------------
    # Table loading
    # ----------------------------------------------------------

    def load_table(
        self,
        material: str,
        log_rho: np.ndarray,
        log_T: np.ndarray,
        pressure: np.ndarray,
        energy: np.ndarray,
        ionization: np.ndarray | None = None,
    ) -> None:
        """Load a pre-computed EOS table for a material.

        Tables are stored in (log10(rho), log10(T)) space for better
        interpolation accuracy across the wide dynamic range of DPF
        plasmas.

        Args:
            material: Material name (e.g. 'deuterium', 'copper').
            log_rho: log10(rho [kg/m^3]) grid, 1D array of length n_rho.
            log_T: log10(T [K]) grid, 1D array of length n_T.
            pressure: Pressure table [Pa], shape (n_rho, n_T).
            energy: Specific internal energy table [J/kg], shape (n_rho, n_T).
            ionization: Mean ionization state Z table, shape (n_rho, n_T).
                Optional; if ``None``, ``ionization_state()`` returns 1.0.

        Raises:
            ValueError: If array shapes are inconsistent.
        """
        log_rho = np.asarray(log_rho, dtype=np.float64)
        log_T = np.asarray(log_T, dtype=np.float64)
        pressure = np.asarray(pressure, dtype=np.float64)
        energy = np.asarray(energy, dtype=np.float64)

        expected_shape = (len(log_rho), len(log_T))
        if pressure.shape != expected_shape:
            msg = (
                f"Pressure shape {pressure.shape} does not match "
                f"expected {expected_shape}"
            )
            raise ValueError(msg)
        if energy.shape != expected_shape:
            msg = (
                f"Energy shape {energy.shape} does not match "
                f"expected {expected_shape}"
            )
            raise ValueError(msg)

        if ionization is not None:
            ionization = np.asarray(ionization, dtype=np.float64)
            if ionization.shape != expected_shape:
                msg = (
                    f"Ionization shape {ionization.shape} does not match "
                    f"expected {expected_shape}"
                )
                raise ValueError(msg)

        self._tables[material] = EOSTable(
            material=material,
            log_rho=log_rho,
            log_T=log_T,
            pressure=pressure,
            energy=energy,
            ionization=ionization,
        )
        self._has_table = True

    # ----------------------------------------------------------
    # Table queries
    # ----------------------------------------------------------

    def _lookup(
        self,
        rho: np.ndarray,
        T: np.ndarray,
        material: str,
        field: str,
    ) -> np.ndarray | None:
        """Look up a field from the table via bilinear interpolation.

        Returns ``None`` if no table exists for the requested material
        or the requested field is not available.
        """
        if material not in self._tables:
            return None

        table = self._tables[material]
        data = getattr(table, field, None)
        if data is None:
            return None

        rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))

        log_rho_val = np.log10(np.maximum(rho_arr, 1e-300))
        log_T_val = np.log10(np.maximum(T_arr, 1e-300))

        return _bilinear_interp(
            table.log_rho,
            table.log_T,
            data,
            log_rho_val,
            log_T_val,
        )

    def pressure(
        self,
        rho: np.ndarray,
        T: np.ndarray,
        material: str = "deuterium",
    ) -> np.ndarray:
        """Evaluate pressure from (rho, T).

        Uses the tabulated value if a table is loaded for *material*,
        otherwise falls back to the ideal gas law:
        p = (1 + Z) * (rho / m_i) * k_B * T  with Z = 1, m_i = m_p.

        Args:
            rho: Mass density [kg/m^3].
            T: Temperature [K].
            material: Material identifier.

        Returns:
            Pressure [Pa], same shape as input arrays.
        """
        result = self._lookup(rho, T, material, "pressure")
        if result is not None:
            return result

        # Ideal gas fallback: assume Z=1, deuterium (m_i = m_d ~ 3.34e-27 kg)
        rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
        # p = rho * k_B * T / m_i * (1 + Z)  with Z=1 gives factor of 2
        # but we use gamma-based ideal: p = (gamma - 1) * rho * e
        # Simplest ideal gas: p = n * k_B * T  where n = rho / m_i
        # For fully ionized Z=1: p = 2 * n_i * k_B * T
        m_i = m_d  # deuterium mass fallback (default material)
        return 2.0 * (rho_arr / m_i) * k_B * T_arr

    def internal_energy(
        self,
        rho: np.ndarray,
        T: np.ndarray,
        material: str = "deuterium",
    ) -> np.ndarray:
        """Evaluate specific internal energy from (rho, T).

        Uses the tabulated value if a table is loaded for *material*,
        otherwise falls back to the ideal gas law:
        e = (1 / (gamma - 1)) * (1 + Z) * k_B * T / m_i.

        Args:
            rho: Mass density [kg/m^3].
            T: Temperature [K].
            material: Material identifier.

        Returns:
            Specific internal energy [J/kg], same shape as input arrays.
        """
        result = self._lookup(rho, T, material, "energy")
        if result is not None:
            return result

        # Ideal gas fallback
        T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
        m_i = m_d  # deuterium mass fallback (default material)
        return (
            (1.0 / (self.gamma - 1.0))
            * 2.0
            * k_B
            * T_arr
            / m_i
        )

    def ionization_state(
        self,
        rho: np.ndarray,
        T: np.ndarray,
        material: str = "deuterium",
    ) -> np.ndarray:
        """Evaluate mean ionization state Z from (rho, T).

        Returns 1.0 everywhere if no ionization table is loaded.

        Args:
            rho: Mass density [kg/m^3].
            T: Temperature [K].
            material: Material identifier.

        Returns:
            Mean ionization state Z (dimensionless), same shape as inputs.
        """
        result = self._lookup(rho, T, material, "ionization")
        if result is not None:
            return result

        rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        return np.ones_like(rho_arr)

    # ----------------------------------------------------------
    # Ideal table generation (for testing / fallback)
    # ----------------------------------------------------------

    def generate_ideal_table(
        self,
        material: str = "deuterium",
        ion_mass: float = 3.34e-27,
        Z: int = 1,
        gamma: float = 5.0 / 3.0,
        n_rho: int = 100,
        n_T: int = 100,
        rho_range: tuple[float, float] = (1e-6, 1e3),
        T_range: tuple[float, float] = (1e2, 1e9),
    ) -> None:
        """Generate an ideal gas EOS table for testing or as a fallback.

        The ideal gas model for a fully ionized plasma with charge state Z:
            p = (1 + Z) * (rho / m_i) * k_B * T
            e = (1 / (gamma - 1)) * (1 + Z) * k_B * T / m_i

        The ionization table is filled with a constant value of Z.

        Args:
            material: Material name to register the table under.
            ion_mass: Ion mass [kg] (default: deuterium).
            Z: Constant ionization state (default: 1).
            gamma: Adiabatic index (default: 5/3).
            n_rho: Number of density grid points (default: 100).
            n_T: Number of temperature grid points (default: 100).
            rho_range: (rho_min, rho_max) in kg/m^3 (default: 1e-6 to 1e3).
            T_range: (T_min, T_max) in K (default: 1e2 to 1e9).
        """
        log_rho = np.linspace(np.log10(rho_range[0]), np.log10(rho_range[1]), n_rho)
        log_T = np.linspace(np.log10(T_range[0]), np.log10(T_range[1]), n_T)

        # Build 2D grids in physical space
        rho_2d = 10.0 ** log_rho[:, np.newaxis]  # (n_rho, 1)
        T_2d = 10.0 ** log_T[np.newaxis, :]  # (1, n_T)

        # Number density
        n_i = rho_2d / ion_mass  # (n_rho, n_T) via broadcasting

        # Ideal gas: p = (1 + Z) * n_i * k_B * T
        pressure = (1.0 + Z) * n_i * k_B * T_2d

        # Specific internal energy: e = p / ((gamma - 1) * rho)
        #   = (1 + Z) * k_B * T / ((gamma - 1) * m_i)
        energy = (1.0 + Z) * k_B * T_2d / ((gamma - 1.0) * ion_mass)
        # Broadcast to full shape
        energy = np.broadcast_to(energy, (n_rho, n_T)).copy()

        # Constant ionization
        ionization = np.full((n_rho, n_T), float(Z))

        self.load_table(
            material=material,
            log_rho=log_rho,
            log_T=log_T,
            pressure=pressure,
            energy=energy,
            ionization=ionization,
        )
