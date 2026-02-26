"""Unit conversion utilities for DPF.

Handles conversion between SI units (used by the Engine and Python components)
and Heaviside-Lorentz / Code units (used by the Metal MHD solver).

In Heaviside-Lorentz (HL) units, we set magnetic permeability mu_0 = 1.
This implies a scaling of the magnetic field B and current density J.

SI:
  Force = J x B
  Energy Density = B^2 / (2 * mu_0)
  Div B = 0

HL (Code Units):
  Force = J_hl x B_hl
  Energy Density = B_hl^2 / 2
  Div B_hl = 0

Relations:
  B_hl = B_si / sqrt(mu_0)
  J_hl = J_si * sqrt(mu_0)

  rho_hl = rho_si
  p_hl   = p_si
  v_hl   = v_si
  t_hl   = t_si
  x_hl   = x_si
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dpf.constants import mu_0 as MU_0  # noqa: N812

if TYPE_CHECKING:
    import torch

SQRT_MU_0 = np.sqrt(MU_0)

def to_code_units(B_si: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert Magnetic Field from SI (Tesla) to Code Units (HL)."""
    return B_si / SQRT_MU_0

def to_si_units(B_code: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert Magnetic Field from Code Units (HL) to SI (Tesla)."""
    return B_code * SQRT_MU_0

def current_to_code_units(J_si: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert Current Density from SI (A/m^2) to Code Units."""
    # J_hl = Curl(B_hl) = Curl(B_si / sqrt(mu0)) = Curl(B_si) / sqrt(mu0)
    # J_si = Curl(B_si) / mu0
    # So J_hl = (J_si * mu0) / sqrt(mu0) = J_si * sqrt(mu0)
    return J_si * SQRT_MU_0

def current_to_si_units(J_code: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert Current Density from Code Units to SI (A/m^2)."""
    return J_code / SQRT_MU_0
