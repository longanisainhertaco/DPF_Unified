"""Equation of State for two-temperature plasma.

Provides both ideal-gas and tabulated EOS. The ideal-gas model is the
default for MVP; tabulated EOS can be loaded from HDF5 files.

NOTE: This is a clean rewrite — the original DPF_AI eos.py had
electron_pressure and electron_energy as exact copies of the ion versions,
which broke two-temperature physics.
"""

from __future__ import annotations

import numpy as np

from dpf.constants import k_B, m_p


class IdealEOS:
    """Ideal gas equation of state for a two-temperature plasma.

    p_i = n_i * k_B * T_i
    p_e = n_e * k_B * T_e
    e_i = p_i / (gamma - 1) / rho
    e_e = p_e / (gamma - 1) / rho
    """

    def __init__(self, gamma: float = 5.0 / 3.0, ion_mass: float = m_p, Z: float = 1.0) -> None:
        self.gamma = gamma
        self.mi = ion_mass
        self.Z = Z

    def ion_pressure(self, rho: np.ndarray, Ti: np.ndarray) -> np.ndarray:
        """Ion pressure: p_i = (rho / m_i) * k_B * T_i."""
        n_i = rho / self.mi
        return n_i * k_B * Ti

    def electron_pressure(self, rho: np.ndarray, Te: np.ndarray) -> np.ndarray:
        """Electron pressure: p_e = Z * (rho / m_i) * k_B * T_e."""
        n_e = self.Z * rho / self.mi
        return n_e * k_B * Te

    def total_pressure(self, rho: np.ndarray, Ti: np.ndarray, Te: np.ndarray) -> np.ndarray:
        """Total thermal pressure: p = p_i + p_e."""
        return self.ion_pressure(rho, Ti) + self.electron_pressure(rho, Te)

    def ion_energy(self, rho: np.ndarray, Ti: np.ndarray) -> np.ndarray:
        """Ion specific internal energy [J/kg]."""
        return self.ion_pressure(rho, Ti) / ((self.gamma - 1.0) * np.maximum(rho, 1e-30))

    def electron_energy(self, rho: np.ndarray, Te: np.ndarray) -> np.ndarray:
        """Electron specific internal energy [J/kg]."""
        return self.electron_pressure(rho, Te) / ((self.gamma - 1.0) * np.maximum(rho, 1e-30))

    def sound_speed(self, rho: np.ndarray, Ti: np.ndarray, Te: np.ndarray) -> np.ndarray:
        """Adiabatic sound speed."""
        p = self.total_pressure(rho, Ti, Te)
        return np.sqrt(self.gamma * p / np.maximum(rho, 1e-30))

    def temperature_from_energy(self, rho: np.ndarray, e_int: np.ndarray) -> np.ndarray:
        """Recover temperature from specific internal energy (single-temperature)."""
        n = rho / self.mi
        return (self.gamma - 1.0) * e_int * rho / (np.maximum(n, 1e-30) * k_B)
