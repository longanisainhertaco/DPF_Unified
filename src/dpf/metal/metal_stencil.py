"""PyTorch MPS implementations of core MHD stencil operations.

Production physics code for Apple Metal GPU acceleration of the DPF
constrained transport, viscosity, and implicit diffusion kernels.  Every
function is a drop-in replacement for its Numba counterpart and must agree
within float32 tolerance (~1e-6 relative).

All tensors are float32 -- MPS has no float64 support.  Inputs and outputs
live on the MPS device; the helper ``_ensure_mps`` validates this.

Staggered (Yee) grid layout conventions follow Evans & Hawley (1988) and
Gardiner & Stone (2005):

    Bx_face: (nx+1, ny,   nz  )   on x-faces
    By_face: (nx,   ny+1, nz  )   on y-faces
    Bz_face: (nx,   ny,   nz+1)   on z-faces

    Ex_edge: (nx,   ny+1, nz+1)   on x-edges (y-z edges)
    Ey_edge: (nx+1, ny,   nz+1)   on y-edges (x-z edges)
    Ez_edge: (nx+1, ny+1, nz  )   on z-edges (x-y edges)

References:
    Evans C.R. & Hawley J.F., ApJ 332, 659 (1988).
    Gardiner T.A. & Stone J.M., JCP 205, 509 (2005).
    Braginskii S.I., Reviews of Plasma Physics Vol. 1 (1965).

Functions:
    ct_update_mps:            Faraday's law CT update on staggered grid.
    div_B_mps:                Face-centred divergence of B.
    emf_from_fluxes_mps:      Edge EMFs from face fluxes (simple CT).
    gradient_3d_mps:          Centred finite-difference gradient.
    strain_rate_mps:          Symmetric strain rate tensor.
    laplacian_3d_mps:         7-point Laplacian via conv3d.
    implicit_diffusion_step_mps: ADI Crank-Nicolson diffusion step.
"""

from __future__ import annotations  # noqa: I001

import torch
import torch.nn.functional as F  # noqa: N812


# ============================================================
# Device validation helper
# ============================================================


def _ensure_mps(t: torch.Tensor, name: str = "tensor") -> None:
    """Validate that a tensor resides on the MPS device.

    Args:
        t: Tensor to check.
        name: Human-readable label for error messages.

    Raises:
        ValueError: If the tensor is not on an MPS device.
    """
    if t.device.type != "mps":
        raise ValueError(
            f"{name} must be on MPS device, got {t.device}"
        )


def _check_no_nan(t: torch.Tensor, label: str = "result") -> None:
    """Assert that a tensor contains no NaN values.

    Args:
        t: Tensor to validate.
        label: Context label for the assertion error message.

    Raises:
        AssertionError: If any element is NaN.
    """
    assert not torch.isnan(t).any(), f"NaN detected in {label}"


# ============================================================
# 1. Constrained Transport update  (Faraday's law)
# ============================================================


def ct_update_mps(
    Bx_face: torch.Tensor,
    By_face: torch.Tensor,
    Bz_face: torch.Tensor,
    Ex_edge: torch.Tensor,
    Ey_edge: torch.Tensor,
    Ez_edge: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance face-centred B by one timestep using constrained transport.

    Implements Faraday's law on the staggered (Yee) grid:

        dBx/dt = -(dEz/dy - dEy/dz)   at x-faces (i+1/2, j, k)
        dBy/dt = -(dEx/dz - dEz/dx)   at y-faces (i, j+1/2, k)
        dBz/dt = -(dEy/dx - dEx/dy)   at z-faces (i, j, k+1/2)

    The discrete curl from edges to faces preserves div(B) = 0 exactly.

    Args:
        Bx_face: x-component on x-faces, shape (nx+1, ny, nz), float32, MPS.
        By_face: y-component on y-faces, shape (nx, ny+1, nz), float32, MPS.
        Bz_face: z-component on z-faces, shape (nx, ny, nz+1), float32, MPS.
        Ex_edge: x-edge EMF, shape (nx, ny+1, nz+1), float32, MPS.
        Ey_edge: y-edge EMF, shape (nx+1, ny, nz+1), float32, MPS.
        Ez_edge: z-edge EMF, shape (nx+1, ny+1, nz), float32, MPS.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].
        dt: Timestep [s].

    Returns:
        Tuple (Bx_new, By_new, Bz_new) of updated face-centred B fields.
    """
    _ensure_mps(Bx_face, "Bx_face")
    _ensure_mps(By_face, "By_face")
    _ensure_mps(Bz_face, "Bz_face")
    _ensure_mps(Ex_edge, "Ex_edge")
    _ensure_mps(Ey_edge, "Ey_edge")
    _ensure_mps(Ez_edge, "Ez_edge")

    # --- Update Bx at x-faces (i+1/2, j, k) ---
    # dEz/dy: Ez is (nx+1, ny+1, nz).  Difference along dim=1 gives (nx+1, ny, nz).
    dEz_dy = (Ez_edge[:, 1:, :] - Ez_edge[:, :-1, :]) / dy
    # dEy/dz: Ey is (nx+1, ny, nz+1).  Difference along dim=2 gives (nx+1, ny, nz).
    dEy_dz = (Ey_edge[:, :, 1:] - Ey_edge[:, :, :-1]) / dz
    Bx_new = Bx_face - dt * (dEz_dy - dEy_dz)

    # --- Update By at y-faces (i, j+1/2, k) ---
    # dEx/dz: Ex is (nx, ny+1, nz+1).  Difference along dim=2 gives (nx, ny+1, nz).
    dEx_dz = (Ex_edge[:, :, 1:] - Ex_edge[:, :, :-1]) / dz
    # dEz/dx: Ez is (nx+1, ny+1, nz).  Difference along dim=0 gives (nx, ny+1, nz).
    dEz_dx = (Ez_edge[1:, :, :] - Ez_edge[:-1, :, :]) / dx
    By_new = By_face - dt * (dEx_dz - dEz_dx)

    # --- Update Bz at z-faces (i, j, k+1/2) ---
    # dEy/dx: Ey is (nx+1, ny, nz+1).  Difference along dim=0 gives (nx, ny, nz+1).
    dEy_dx = (Ey_edge[1:, :, :] - Ey_edge[:-1, :, :]) / dx
    # dEx/dy: Ex is (nx, ny+1, nz+1).  Difference along dim=1 gives (nx, ny, nz+1).
    dEx_dy = (Ex_edge[:, 1:, :] - Ex_edge[:, :-1, :]) / dy
    Bz_new = Bz_face - dt * (dEy_dx - dEx_dy)

    _check_no_nan(Bx_new, "Bx_new after CT update")
    _check_no_nan(By_new, "By_new after CT update")
    _check_no_nan(Bz_new, "Bz_new after CT update")

    return Bx_new, By_new, Bz_new


# ============================================================
# 2. Divergence of face-centred B
# ============================================================


def div_B_mps(
    Bx_face: torch.Tensor,
    By_face: torch.Tensor,
    Bz_face: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """Compute cell-centred divergence of face-centred B.

    div(B)[i,j,k] = (Bx[i+1,j,k] - Bx[i,j,k]) / dx
                   + (By[i,j+1,k] - By[i,j,k]) / dy
                   + (Bz[i,j,k+1] - Bz[i,j,k]) / dz

    After a CT update, this should be zero to float32 machine precision
    (~1e-7) if the initial B was divergence-free.

    Args:
        Bx_face: x-faces, shape (nx+1, ny, nz), float32, MPS.
        By_face: y-faces, shape (nx, ny+1, nz), float32, MPS.
        Bz_face: z-faces, shape (nx, ny, nz+1), float32, MPS.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].

    Returns:
        Divergence tensor, shape (nx, ny, nz), float32, MPS.
    """
    _ensure_mps(Bx_face, "Bx_face")
    _ensure_mps(By_face, "By_face")
    _ensure_mps(Bz_face, "Bz_face")

    # Bx_face is (nx+1, ny, nz): difference along dim=0 -> (nx, ny, nz)
    dBx_dx = (Bx_face[1:, :, :] - Bx_face[:-1, :, :]) / dx
    # By_face is (nx, ny+1, nz): difference along dim=1 -> (nx, ny, nz)
    dBy_dy = (By_face[:, 1:, :] - By_face[:, :-1, :]) / dy
    # Bz_face is (nx, ny, nz+1): difference along dim=2 -> (nx, ny, nz)
    dBz_dz = (Bz_face[:, :, 1:] - Bz_face[:, :, :-1]) / dz

    result = dBx_dx + dBy_dy + dBz_dz

    _check_no_nan(result, "div_B")
    return result


# ============================================================
# 3. EMF from face-centred fluxes  (Gardiner & Stone simple CT)
# ============================================================


def emf_from_fluxes_mps(
    flux_x: torch.Tensor,
    flux_y: torch.Tensor,
    flux_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute edge-centred EMFs from face-centred fluxes via arithmetic averaging.

    Uses the "simple CT" method of Gardiner & Stone (2005, JCP 205, 509).
    Each edge EMF is the arithmetic average of the four adjacent face flux
    contributions that share that edge.

    Interior averaging for Ez at edge (i+1/2, j+1/2, k):
        Ez = 0.25 * (Fx[i,j,k] + Fx[i,j+1,k] + Fy[i,j,k] + Fy[i+1,j,k])

    Boundary edges use clamped indices (zero-gradient extrapolation).

    Args:
        flux_x: EMF contribution at x-faces, shape (nx+1, ny, nz), float32, MPS.
        flux_y: EMF contribution at y-faces, shape (nx, ny+1, nz), float32, MPS.
        flux_z: EMF contribution at z-faces, shape (nx, ny, nz+1), float32, MPS.

    Returns:
        Tuple (Ex_edge, Ey_edge, Ez_edge) with shapes:
            Ex_edge: (nx, ny+1, nz+1)
            Ey_edge: (nx+1, ny, nz+1)
            Ez_edge: (nx+1, ny+1, nz)
    """
    _ensure_mps(flux_x, "flux_x")
    _ensure_mps(flux_y, "flux_y")
    _ensure_mps(flux_z, "flux_z")

    # Shape references for documentation; actual shapes are consumed
    # implicitly by the padding and slicing operations below.
    # flux_x: (nx+1, ny, nz), flux_y: (nx, ny+1, nz), flux_z: (nx, ny, nz+1)

    # --- Ex at x-edges: shape (nx, ny+1, nz+1) ---
    # Average Fz along y-direction and Fy along z-direction.
    # Pad flux_z (nx, ny, nz+1) along dim=1: need (nx, ny+2, nz+1) then avg
    # Pad flux_y (nx, ny+1, nz) along dim=2: need (nx, ny+1, nz+2) then avg
    # For boundary handling: replicate-pad.
    # Fz contribution: average flux_z[i, j-1, k] and flux_z[i, j, k]
    #   -> pad dim=1 on both sides by 1, then average adjacent
    Fz_pad_y = F.pad(
        flux_z.unsqueeze(0).unsqueeze(0),  # (1,1, nx, ny, nz+1)
        (0, 0, 1, 1, 0, 0),               # pad dim=1 (ny) by 1 on each side
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (nx, ny+2, nz+1)
    Fz_avg_y = 0.5 * (Fz_pad_y[:, :-1, :] + Fz_pad_y[:, 1:, :])  # (nx, ny+1, nz+1)

    # Fy contribution: average flux_y[i, j, k-1] and flux_y[i, j, k]
    Fy_pad_z = F.pad(
        flux_y.unsqueeze(0).unsqueeze(0),  # (1,1, nx, ny+1, nz)
        (1, 1, 0, 0, 0, 0),               # pad dim=2 (nz) by 1 on each side
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (nx, ny+1, nz+2)
    Fy_avg_z = 0.5 * (Fy_pad_z[:, :, :-1] + Fy_pad_z[:, :, 1:])  # (nx, ny+1, nz+1)

    Ex_edge = 0.5 * (Fz_avg_y + Fy_avg_z)

    # --- Ey at y-edges: shape (nx+1, ny, nz+1) ---
    # Average Fx along z-direction and Fz along x-direction.
    # Fx (nx+1, ny, nz): pad dim=2 by 1 each side
    Fx_pad_z = F.pad(
        flux_x.unsqueeze(0).unsqueeze(0),
        (1, 1, 0, 0, 0, 0),
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (nx+1, ny, nz+2)
    Fx_avg_z = 0.5 * (Fx_pad_z[:, :, :-1] + Fx_pad_z[:, :, 1:])  # (nx+1, ny, nz+1)

    # Fz (nx, ny, nz+1): pad dim=0 by 1 each side
    Fz_pad_x = F.pad(
        flux_z.unsqueeze(0).unsqueeze(0),
        (0, 0, 0, 0, 1, 1),
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (nx+2, ny, nz+1)
    Fz_avg_x = 0.5 * (Fz_pad_x[:-1, :, :] + Fz_pad_x[1:, :, :])  # (nx+1, ny, nz+1)

    Ey_edge = 0.5 * (Fx_avg_z + Fz_avg_x)

    # --- Ez at z-edges: shape (nx+1, ny+1, nz) ---
    # Average Fy along x-direction and Fx along y-direction.
    # Fy (nx, ny+1, nz): pad dim=0 by 1 each side
    Fy_pad_x = F.pad(
        flux_y.unsqueeze(0).unsqueeze(0),
        (0, 0, 0, 0, 1, 1),
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (nx+2, ny+1, nz)
    Fy_avg_x = 0.5 * (Fy_pad_x[:-1, :, :] + Fy_pad_x[1:, :, :])  # (nx+1, ny+1, nz)

    # Fx (nx+1, ny, nz): pad dim=1 by 1 each side
    Fx_pad_y = F.pad(
        flux_x.unsqueeze(0).unsqueeze(0),
        (0, 0, 1, 1, 0, 0),
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (nx+1, ny+2, nz)
    Fx_avg_y = 0.5 * (Fx_pad_y[:, :-1, :] + Fx_pad_y[:, 1:, :])  # (nx+1, ny+1, nz)

    Ez_edge = 0.5 * (Fy_avg_x + Fx_avg_y)

    _check_no_nan(Ex_edge, "Ex_edge from fluxes")
    _check_no_nan(Ey_edge, "Ey_edge from fluxes")
    _check_no_nan(Ez_edge, "Ez_edge from fluxes")

    return Ex_edge, Ey_edge, Ez_edge


# ============================================================
# 4. Centred finite-difference gradient
# ============================================================


def gradient_3d_mps(
    field: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the gradient of a 3D scalar field using centred finite differences.

    Interior points use second-order centred differences:
        df/dx[i] = (f[i+1] - f[i-1]) / (2*dx)

    Boundary points use one-sided first-order differences (Neumann-compatible):
        df/dx[0]    = (f[1]    - f[0])      / dx
        df/dx[n-1]  = (f[n-1]  - f[n-2])    / dx

    Args:
        field: Scalar field, shape (nx, ny, nz), float32, MPS.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].

    Returns:
        Tuple (df_dx, df_dy, df_dz) each shape (nx, ny, nz), float32, MPS.
    """
    _ensure_mps(field, "field")
    nx, ny, nz = field.shape

    # --- df/dx ---
    df_dx = torch.empty_like(field)
    if nx >= 3:
        df_dx[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2.0 * dx)
    df_dx[0, :, :] = (field[1, :, :] - field[0, :, :]) / dx if nx > 1 else 0.0
    if nx > 1:
        df_dx[-1, :, :] = (field[-1, :, :] - field[-2, :, :]) / dx

    # --- df/dy ---
    df_dy = torch.empty_like(field)
    if ny >= 3:
        df_dy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * dy)
    df_dy[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / dy if ny > 1 else 0.0
    if ny > 1:
        df_dy[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / dy

    # --- df/dz ---
    df_dz = torch.empty_like(field)
    if nz >= 3:
        df_dz[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * dz)
    df_dz[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dz if nz > 1 else 0.0
    if nz > 1:
        df_dz[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dz

    _check_no_nan(df_dx, "df_dx")
    _check_no_nan(df_dy, "df_dy")
    _check_no_nan(df_dz, "df_dz")

    return df_dx, df_dy, df_dz


# ============================================================
# 5. Symmetric strain rate tensor
# ============================================================


def _derivative_with_bc(
    field: torch.Tensor,
    axis: int,
    spacing: float,
) -> torch.Tensor:
    """Compute a derivative along *axis* with boundary handling.

    Interior: centred second-order differences (f[i+1]-f[i-1])/(2*h).
    Boundaries: one-sided first-order differences, matching the Numba kernel.

    Args:
        field: 3D tensor, shape (nx, ny, nz).
        axis: Dimension along which to differentiate (0, 1, or 2).
        spacing: Grid spacing along that axis.

    Returns:
        Derivative tensor, same shape as *field*.
    """
    n = field.shape[axis]
    result = torch.empty_like(field)

    # Build slice helpers
    def _sl(dim: int, s: slice) -> tuple[slice, ...]:
        slices: list[slice] = [slice(None)] * 3
        slices[dim] = s
        return tuple(slices)

    # Interior: centred differences
    if n >= 3:
        result[_sl(axis, slice(1, -1))] = (
            field[_sl(axis, slice(2, None))] - field[_sl(axis, slice(None, -2))]
        ) / (2.0 * spacing)

    # Left boundary: one-sided forward difference
    if n > 1:
        result[_sl(axis, slice(0, 1))] = (
            field[_sl(axis, slice(1, 2))] - field[_sl(axis, slice(0, 1))]
        ) / spacing
    else:
        result[_sl(axis, slice(0, 1))] = 0.0

    # Right boundary: one-sided backward difference
    if n > 1:
        result[_sl(axis, slice(-1, None))] = (
            field[_sl(axis, slice(-1, None))] - field[_sl(axis, slice(-2, -1))]
        ) / spacing

    return result


def strain_rate_mps(
    velocity: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """Compute the symmetric strain rate tensor S_ij.

    S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)

    Uses second-order centred differences on the interior and one-sided
    differences at boundaries, matching the Numba ``_compute_strain_rate``
    kernel exactly.

    Args:
        velocity: Velocity field, shape (3, nx, ny, nz), float32, MPS.
            velocity[0] = vx, velocity[1] = vy, velocity[2] = vz.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].

    Returns:
        Symmetric strain rate tensor, shape (6, nx, ny, nz), float32, MPS.
        Components ordered as [Sxx, Syy, Szz, Sxy, Sxz, Syz].
    """
    _ensure_mps(velocity, "velocity")
    assert velocity.shape[0] == 3, "velocity must have shape (3, nx, ny, nz)"

    vx = velocity[0]  # (nx, ny, nz)
    vy = velocity[1]
    vz = velocity[2]

    # Compute all 9 velocity gradient components dv_i / dx_j
    # dvx/dx, dvx/dy, dvx/dz
    dvx_dx = _derivative_with_bc(vx, 0, dx)
    dvx_dy = _derivative_with_bc(vx, 1, dy)
    dvx_dz = _derivative_with_bc(vx, 2, dz)

    # dvy/dx, dvy/dy, dvy/dz
    dvy_dx = _derivative_with_bc(vy, 0, dx)
    dvy_dy = _derivative_with_bc(vy, 1, dy)
    dvy_dz = _derivative_with_bc(vy, 2, dz)

    # dvz/dx, dvz/dy, dvz/dz
    dvz_dx = _derivative_with_bc(vz, 0, dx)
    dvz_dy = _derivative_with_bc(vz, 1, dy)
    dvz_dz = _derivative_with_bc(vz, 2, dz)

    # Diagonal components
    Sxx = dvx_dx
    Syy = dvy_dy
    Szz = dvz_dz

    # Off-diagonal (symmetric average)
    Sxy = 0.5 * (dvx_dy + dvy_dx)
    Sxz = 0.5 * (dvx_dz + dvz_dx)
    Syz = 0.5 * (dvy_dz + dvz_dy)

    result = torch.stack([Sxx, Syy, Szz, Sxy, Sxz, Syz], dim=0)

    _check_no_nan(result, "strain_rate")
    return result


# ============================================================
# 6. 7-point Laplacian via conv3d
# ============================================================


def laplacian_3d_mps(
    field: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """Compute the 3D Laplacian using the standard 7-point stencil.

    nabla^2 f = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2

    where each second derivative uses the centred difference:
        d^2f/dx^2 = (f[i+1] - 2*f[i] + f[i-1]) / dx^2

    Implemented via ``torch.nn.functional.conv3d`` with a hand-crafted
    (1, 1, 3, 3, 3) kernel encoding the anisotropic weights.  The boundary
    is handled with replicate padding (Neumann zero-gradient), which sets
    the ghost value equal to the boundary value and produces zero boundary
    flux.

    Args:
        field: Scalar field, shape (nx, ny, nz), float32, MPS.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].

    Returns:
        Laplacian tensor, shape (nx, ny, nz), float32, MPS.
    """
    _ensure_mps(field, "field")

    device = field.device

    # Build the 3x3x3 kernel.  Only the 6 face-neighbours and the centre
    # have nonzero weights for the standard 7-point stencil.
    #
    # kernel[0, 0, 1, 1, :] handles d^2f/dz^2   (weights 1/dz^2 at [0] and [2], -2/dz^2 at [1])
    # kernel[0, 0, 1, :, 1] handles d^2f/dy^2
    # kernel[0, 0, :, 1, 1] handles d^2f/dx^2
    w_x = 1.0 / (dx * dx)
    w_y = 1.0 / (dy * dy)
    w_z = 1.0 / (dz * dz)

    kernel = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=torch.float32)
    # d^2f/dx^2 contributions (axis 0 of the field = axis 2 of conv3d input)
    kernel[0, 0, 0, 1, 1] = w_x
    kernel[0, 0, 2, 1, 1] = w_x
    # d^2f/dy^2 contributions
    kernel[0, 0, 1, 0, 1] = w_y
    kernel[0, 0, 1, 2, 1] = w_y
    # d^2f/dz^2 contributions
    kernel[0, 0, 1, 1, 0] = w_z
    kernel[0, 0, 1, 1, 2] = w_z
    # Centre weight
    kernel[0, 0, 1, 1, 1] = -2.0 * (w_x + w_y + w_z)

    # conv3d expects (batch, channels, D, H, W).
    # Replicate-pad by 1 on each side for Neumann BCs.
    inp = field.unsqueeze(0).unsqueeze(0)  # (1, 1, nx, ny, nz)
    inp_padded = F.pad(inp, (1, 1, 1, 1, 1, 1), mode="replicate")
    result = F.conv3d(inp_padded, kernel, padding=0).squeeze(0).squeeze(0)

    _check_no_nan(result, "laplacian_3d")
    return result


# ============================================================
# 7. Implicit diffusion step (ADI with Thomas algorithm)
# ============================================================


def _thomas_solve_batched(
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Batched Thomas algorithm for tridiagonal systems on MPS.

    Solves M independent n-point tridiagonal systems simultaneously.

    The algorithm runs a sequential forward-elimination / back-substitution
    sweep (inherently serial along the pencil direction), but processes all
    M pencils in parallel via tensor operations on MPS.

    Args:
        lower: Sub-diagonal, shape (M, n). lower[:, 0] is unused.
        diag:  Main diagonal, shape (M, n).
        upper: Super-diagonal, shape (M, n). upper[:, n-1] is unused.
        rhs:   Right-hand side, shape (M, n).

    Returns:
        Solution tensor, shape (M, n).
    """
    # Work on copies to avoid mutating caller data
    c = upper.clone()
    d = rhs.clone()
    b = diag.clone()

    M, n = b.shape

    # Forward elimination
    for i in range(1, n):
        # Avoid division by zero: where b[:, i-1] == 0, set m to 0
        b_prev = b[:, i - 1].clone()
        safe_b = torch.where(b_prev == 0.0, torch.ones_like(b_prev), b_prev)
        m = lower[:, i] / safe_b
        m = torch.where(b_prev == 0.0, torch.zeros_like(m), m)
        b[:, i] = b[:, i] - m * c[:, i - 1]
        d[:, i] = d[:, i] - m * d[:, i - 1]

    # Back substitution
    x = torch.empty_like(d)
    b_last = b[:, n - 1]
    safe_b_last = torch.where(
        b_last == 0.0, torch.ones_like(b_last), b_last
    )
    x[:, n - 1] = torch.where(
        b_last == 0.0, torch.zeros_like(d[:, n - 1]), d[:, n - 1] / safe_b_last
    )
    for i in range(n - 2, -1, -1):
        b_i = b[:, i]
        safe_b_i = torch.where(b_i == 0.0, torch.ones_like(b_i), b_i)
        x[:, i] = torch.where(
            b_i == 0.0,
            torch.zeros_like(d[:, i]),
            (d[:, i] - c[:, i] * x[:, i + 1]) / safe_b_i,
        )

    return x


def _build_cn_system_batched(
    pencils: torch.Tensor,
    coeff_pencils: torch.Tensor,
    dt: float,
    dh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the Crank-Nicolson tridiagonal system for a batch of 1D pencils.

    Discretises  du/dt = d/dx[ D(x) * du/dx ]  using Crank-Nicolson
    (theta = 0.5) with Neumann (zero-gradient) boundary conditions.

    The LHS is:  (I - dt/2 * D_h) u^{n+1} = rhs
    The RHS is:  (I + dt/2 * D_h) u^n

    Args:
        pencils: Field values along pencils, shape (M, n).
        coeff_pencils: Diffusion coefficient along pencils, shape (M, n).
        dt: Timestep [s].
        dh: Grid spacing along this axis [m].

    Returns:
        Tuple (lower, diag, upper, rhs) each shape (M, n).
    """
    M, n = pencils.shape
    dh2 = dh * dh
    device = pencils.device

    # Interface-centred diffusion coefficient:
    #   sigma[i] = dt / (2 * dh^2) * 0.5 * (coeff[i] + coeff[i+1])
    sigma = (0.5 * dt / dh2) * 0.5 * (
        coeff_pencils[:, :-1] + coeff_pencils[:, 1:]
    )  # (M, n-1)

    # --- Build RHS: (I + dt/2 * D_h) u^n ---
    rhs = torch.empty_like(pencils)

    # Interior points: i = 1..n-2
    if n >= 3:
        rhs[:, 1:-1] = (
            pencils[:, 1:-1]
            + sigma[:, 1:] * (pencils[:, 2:] - pencils[:, 1:-1])
            - sigma[:, :-1] * (pencils[:, 1:-1] - pencils[:, :-2])
        )

    # Neumann BC at left (i=0): ghost = boundary => zero diffusion flux at wall
    rhs[:, 0] = pencils[:, 0] + sigma[:, 0] * (pencils[:, 1] - pencils[:, 0])
    # Neumann BC at right (i=n-1)
    rhs[:, -1] = pencils[:, -1] - sigma[:, -1] * (
        pencils[:, -1] - pencils[:, -2]
    )

    # --- Build LHS tridiagonal: (I - dt/2 * D_h) ---
    lower = torch.zeros(M, n, device=device, dtype=torch.float32)
    diag = torch.ones(M, n, device=device, dtype=torch.float32)
    upper = torch.zeros(M, n, device=device, dtype=torch.float32)

    if n >= 3:
        lower[:, 1:-1] = -sigma[:, :-1]
        upper[:, 1:-1] = -sigma[:, 1:]
        diag[:, 1:-1] = 1.0 + sigma[:, :-1] + sigma[:, 1:]

    # Neumann BC rows
    diag[:, 0] = 1.0 + sigma[:, 0]
    upper[:, 0] = -sigma[:, 0]
    lower[:, -1] = -sigma[:, -1]
    diag[:, -1] = 1.0 + sigma[:, -1]

    return lower, diag, upper, rhs


def _adi_sweep(
    field: torch.Tensor,
    diff_coeff: torch.Tensor,
    dt: float,
    dh: float,
    axis: int,
) -> torch.Tensor:
    """Perform one ADI sweep: implicit Crank-Nicolson diffusion along *axis*.

    Reshapes the 3D field into batches of 1D pencils along the target axis,
    solves the tridiagonal system for all pencils in parallel, and reshapes
    back.

    Args:
        field: 3D field to diffuse, shape (nx, ny, nz), float32, MPS.
        diff_coeff: Spatially varying diffusion coefficient, shape (nx, ny, nz).
        dt: Timestep [s].
        dh: Grid spacing along the sweep axis [m].
        axis: Axis to sweep along (0, 1, or 2).

    Returns:
        Diffused field, same shape as input.
    """
    n = field.shape[axis]

    if n < 3:
        return field.clone()

    # Move the sweep axis to the last dimension for pencil extraction.
    # axes order: put 'axis' last
    perm = list(range(3))
    perm.remove(axis)
    perm.append(axis)
    # field_t: (d0, d1, n) where d0, d1 are the two non-sweep dims
    field_t = field.permute(*perm).contiguous()
    coeff_t = diff_coeff.permute(*perm).contiguous()

    # Reshape to (M, n) where M = d0 * d1
    d0, d1, n_ = field_t.shape
    M = d0 * d1
    pencils = field_t.reshape(M, n_)
    coeff_pencils = coeff_t.reshape(M, n_)

    # Build and solve the tridiagonal system
    lower, diag, upper, rhs = _build_cn_system_batched(
        pencils, coeff_pencils, dt, dh
    )
    solution = _thomas_solve_batched(lower, diag, upper, rhs)

    # Reshape back
    result_t = solution.reshape(d0, d1, n_)

    # Inverse permutation to restore original axis order
    inv_perm = [0] * 3
    for i, p in enumerate(perm):
        inv_perm[p] = i
    result = result_t.permute(*inv_perm).contiguous()

    return result


def implicit_diffusion_step_mps(
    field: torch.Tensor,
    coeff: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """ADI (Alternating Direction Implicit) diffusion step on MPS.

    Solves the diffusion equation:
        du/dt = div[ D(x) * grad(u) ]

    using dimension-by-dimension Crank-Nicolson splitting.  Each axis is
    swept sequentially (x -> y -> z) with a batched Thomas algorithm that
    processes all pencils in parallel on the GPU.

    Neumann (zero-gradient) boundary conditions are applied at all faces.

    The Crank-Nicolson scheme is unconditionally stable and second-order
    accurate in time:
        (I - dt/2 * D_h) u^{n+1} = (I + dt/2 * D_h) u^n

    Args:
        field: 3D scalar field to diffuse, shape (nx, ny, nz), float32, MPS.
        coeff: Spatially varying diffusion coefficient [m^2/s],
               shape (nx, ny, nz), float32, MPS.
        dt: Timestep [s].
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dz: Grid spacing in z [m].

    Returns:
        Diffused field, shape (nx, ny, nz), float32, MPS.
    """
    _ensure_mps(field, "field")
    _ensure_mps(coeff, "coeff")

    # Sequential ADI sweeps: x -> y -> z
    result = _adi_sweep(field, coeff, dt, dx, axis=0)
    result = _adi_sweep(result, coeff, dt, dy, axis=1)
    result = _adi_sweep(result, coeff, dt, dz, axis=2)

    _check_no_nan(result, "implicit_diffusion_step")
    return result
