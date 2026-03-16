"""Velocity shear stabilization diagnostic for DPF pinch.

Computes the shear stabilization margin based on the Shumlak-Hartman criterion:
the plasma is shear-stabilized when |dv_z/dr| > k_z * v_A, where v_A is the
Alfven speed. Extended to azimuthal shear: |dv_theta/dr| > k * v_A.

References:
- Shumlak & Hartman, Phys. Rev. Lett. 75, 3285 (1995)
- MCX: "Velocity Shear Stabilization of Centrifugally Confined Plasma"
- Zap Energy: "Whole Device Modeling of the FuZE SFS Z-Pinch" (2024)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dpf.constants import mu_0


def compute_shear_margin(
    state: dict[str, Any],
    dr: float,
    dz: float,
    L_anode: float,
) -> dict[str, Any] | None:
    """Compute azimuthal velocity shear stabilization margin.

    Args:
        state: MHD state dict with keys rho, velocity, B, pressure.
        dr: Radial grid spacing [m].
        dz: Axial grid spacing [m].
        L_anode: Anode length [m], used as longest unstable wavelength.

    Returns:
        Dict with shear margin, peak shear rate, peak Alfven speed, and
        stability assessment string. Returns None if state is insufficient.
    """
    rho = state.get("rho")
    velocity = state.get("velocity")
    B = state.get("B")

    if rho is None or velocity is None or B is None:
        return None

    rho = np.asarray(rho, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    B = np.asarray(B, dtype=float)

    if rho.ndim < 2 or velocity.ndim < 2:
        return None

    # v_theta is velocity component index 1 (azimuthal in cylindrical coords)
    if velocity.shape[0] < 2:
        return None
    v_theta = velocity[1]  # shape: (nr, [ntheta,] nz)

    # Collapse to (nr, nz) by taking midplane slice if 3D
    if v_theta.ndim == 3:
        v_theta_2d = v_theta[:, v_theta.shape[1] // 2, :]
        rho_2d = rho[:, rho.shape[1] // 2, :]
        B_2d = B[:, :, B.shape[2] // 2, :]  # B has shape (3, nr, ntheta, nz)
    elif v_theta.ndim == 2:
        v_theta_2d = v_theta
        rho_2d = rho
        B_2d = B  # shape (3, nr, nz)
    else:
        return None

    # dv_theta/dr along radial axis (axis=0)
    dvtheta_dr = np.gradient(v_theta_2d, dr, axis=0)

    # Alfven speed: v_A = |B| / sqrt(mu_0 * rho)
    # B_2d has shape (3, nr, nz) or (3, nr) — compute magnitude across field components
    if B_2d.ndim == 3 and B_2d.shape[0] == 3:
        B_mag = np.sqrt(B_2d[0] ** 2 + B_2d[1] ** 2 + B_2d[2] ** 2)
    elif B_2d.ndim == 2 and B_2d.shape[0] == 3:
        # (3, nr) case — B doesn't have a z axis
        B_mag = np.sqrt(np.sum(B_2d ** 2, axis=0))
        B_mag = B_mag[:, np.newaxis] * np.ones_like(v_theta_2d)
    else:
        return None

    rho_safe = np.maximum(rho_2d, 1e-20)
    v_A = B_mag / np.sqrt(mu_0 * rho_safe)

    # Perturbation wavenumber: k = 2*pi / lambda, lambda = L_anode
    lambda_unstable = L_anode if L_anode > 0 else 0.16
    k = 2.0 * np.pi / lambda_unstable

    # Shear stabilization margin: |dv_theta/dr| / (k * v_A)
    # margin > 1 => shear rate exceeds Alfven criterion => stable
    denom = k * v_A
    denom_safe = np.maximum(denom, 1e-30)
    margin_field = np.abs(dvtheta_dr) / denom_safe

    peak_margin = float(np.max(margin_field))
    peak_shear_rate = float(np.max(np.abs(dvtheta_dr)))
    peak_v_A = float(np.max(v_A))
    mean_margin = float(np.mean(margin_field))

    if peak_margin > 1.0:
        assessment = "shear-stabilized"
    elif peak_margin > 0.5:
        assessment = "marginally stable"
    else:
        assessment = "unstable (shear insufficient)"

    return {
        "peak_margin": peak_margin,
        "mean_margin": mean_margin,
        "peak_shear_rate_1_s": peak_shear_rate,
        "peak_v_A_m_s": peak_v_A,
        "k_unstable_1_m": k,
        "lambda_unstable_m": lambda_unstable,
        "assessment": assessment,
        "source": "Shumlak & Hartman, PRL 75, 3285 (1995)",
    }
