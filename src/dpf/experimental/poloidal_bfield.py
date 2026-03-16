"""Auluck poloidal magnetic field via Gratton-Vargas (GV) surface mechanism.

Implements the theory from:
    S.K.H. Auluck, "Poloidal magnetic field in the dense plasma focus,"
    Phys. Plasmas 31, 010704 (2024). doi:10.1063/5.0189593

No DPF simulation code worldwide implements this mechanism. The poloidal
(axial) B-field arises from a simple dynamo: the geomagnetic field seeds
azimuthal electric fields via the curved, moving current sheath (the
Gratton-Vargas surface). This drives azimuthal currents that generate
B_z and B_r components invisible to standard magnetic probes.

Key equations (numbered per paper):
    (2)  B_0 = mu_0*I/(2*pi*a*r_bar), v_0 = B_0/sqrt(2*mu_0*rho_0)
    (3)  tau = Q_m^{-1} * int(I dt), Q_m = pi*mu_0^{-1}*a^2*sqrt(2*mu_0*rho_0)
    (4)  psi(tau, r_bar, z_bar) = GV surface equation
    (8)  dPhi/dtau = -(N/(2r))*(dPhi/dz) - (s/(2r))*sqrt(r^2-N^2)*(dPhi/dr)
    (9)  H = (s/(2r))*sqrt(r^2-N^2)*p_r + (N/(2r))*p_z
    (10) E_theta = (mu_0*I)/(pi*a^2*r*sqrt(2*mu_0*rho_0)) * H
    (11) B_z = p_r / (2*pi*a^2*r)
    (12) B_r = -p_z / (2*pi*a^2*r)

References:
    Auluck 2024, Phys. Plasmas 31, 010704
    Gratton & Vargas 1983, in Energy Storage, Compression and Switching
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Physical constants
MU_0 = 4.0 * np.pi * 1e-7  # vacuum permeability [H/m]


def compute_scaling_params(
    a: float, rho_0: float, I: float  # noqa: E741
) -> dict[str, float]:
    """Compute GV scaling parameters (Eq. 2, 3).

    Args:
        a: Anode radius [m].
        rho_0: Fill gas mass density [kg/m^3].
        I: Instantaneous current [A].

    Returns:
        Dict with B_0, v_0, Q_m scaling parameters.
    """
    B_0 = MU_0 * abs(I) / (2.0 * np.pi * a)  # at r_bar=1
    v_0 = B_0 / np.sqrt(2.0 * MU_0 * rho_0)
    Q_m = np.pi * a**2 * np.sqrt(2.0 * MU_0 * rho_0) / MU_0
    return {"B_0": B_0, "v_0": v_0, "Q_m": Q_m}


def compute_gv_surface(
    N: float,
    nr: int = 64,
    nz: int = 128,
    tau: float = 0.0,
    s: int = -1,
    r_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute the Gratton-Vargas surface psi(r_bar, z_bar) (Eq. 4).

    The GV surface is the traveling surface of rotation psi=0 that defines
    the current sheath (plasma armature) position. The surface equation:

        psi = 2*N*z_bar*s*(r_bar*sqrt(r_bar^2 - N^2)
              - N^2*ArcCosh(r_bar/N)) - tau - psi_GV

    We set psi_0 = psi_GV = 0 for the principal characteristic.

    Args:
        N: Electrode radius ratio a/b (dimensionless, 0 < N < 1).
        nr: Number of radial grid points.
        nz: Number of axial grid points.
        tau: Scaled time (dimensionless).
        s: Sign of radial velocity (-1 for implosion, +1 for expansion).
        r_range: (r_min, r_max) in normalized units. Default (N+eps, 1.0).
        z_range: (z_min, z_max) in normalized units. Default (-2.0, 2.0).

    Returns:
        Dict with keys:
            r_bar: 1D array of normalized radial coordinates.
            z_bar: 1D array of normalized axial coordinates.
            psi: 2D array shape (nr, nz), GV surface function.
            surface_mask: 2D bool array, True where |psi| < threshold
                          (near the current sheath).
    """
    if not 0.0 < N < 1.0:
        raise ValueError(f"N = a/b must be in (0, 1), got {N}")

    eps = 1e-6 * N
    if r_range is None:
        r_range = (N + eps, 1.0)
    if z_range is None:
        z_range = (-2.0, 2.0)

    r_bar = np.linspace(r_range[0], r_range[1], nr)
    z_bar = np.linspace(z_range[0], z_range[1], nz)

    R, Z = np.meshgrid(r_bar, z_bar, indexing="ij")

    # Eq. 4: psi(tau, r_bar, z_bar)
    # Handle sqrt(r_bar^2 - N^2) with clamp for r_bar near N
    r2_minus_N2 = np.clip(R**2 - N**2, 0.0, None)
    sqrt_term = np.sqrt(r2_minus_N2)

    # ArcCosh(r_bar/N) — defined for r_bar >= N
    ratio = np.clip(R / N, 1.0, None)
    acosh_term = np.arccosh(ratio)

    psi = 2.0 * N * Z * s * (R * sqrt_term - N**2 * acosh_term) - tau

    # Surface is where psi ~ 0
    psi_scale = max(np.abs(psi).max(), 1e-10)
    threshold = 0.05 * psi_scale
    surface_mask = np.abs(psi) < threshold

    return {
        "r_bar": r_bar,
        "z_bar": z_bar,
        "psi": psi,
        "surface_mask": surface_mask,
    }


def _flux_pde_rhs(
    Phi: NDArray[np.float64],
    r_bar: NDArray[np.float64],
    z_bar: NDArray[np.float64],
    N: float,
    s: int,
) -> NDArray[np.float64]:
    """RHS of the flux function PDE (Eq. 8).

    dPhi/dtau = -(N/(2*r_bar)) * dPhi/dz_bar
                - (s/(2*r_bar)) * sqrt(r_bar^2 - N^2) * dPhi/dr_bar

    This is the Hamilton-Jacobi equation for the flux function with
    Hamiltonian H = (s/(2r))*sqrt(r^2-N^2)*p_r + (N/(2r))*p_z (Eq. 9).

    Args:
        Phi: Flux function array, shape (nr, nz).
        r_bar: 1D radial coordinates (nr,).
        z_bar: 1D axial coordinates (nz,).
        N: Electrode radius ratio.
        s: Sign of radial velocity.

    Returns:
        dPhi/dtau array, shape (nr, nz).
    """
    nr, nz = Phi.shape
    dr = r_bar[1] - r_bar[0] if nr > 1 else 1.0
    dz = z_bar[1] - z_bar[0] if nz > 1 else 1.0

    # Compute gradients with upwind-biased central differences
    # dPhi/dr_bar
    dPhi_dr = np.zeros_like(Phi)
    dPhi_dr[1:-1, :] = (Phi[2:, :] - Phi[:-2, :]) / (2.0 * dr)
    dPhi_dr[0, :] = (Phi[1, :] - Phi[0, :]) / dr
    dPhi_dr[-1, :] = (Phi[-1, :] - Phi[-2, :]) / dr

    # dPhi/dz_bar
    dPhi_dz = np.zeros_like(Phi)
    dPhi_dz[:, 1:-1] = (Phi[:, 2:] - Phi[:, :-2]) / (2.0 * dz)
    dPhi_dz[:, 0] = (Phi[:, 1] - Phi[:, 0]) / dz
    dPhi_dz[:, -1] = (Phi[:, -1] - Phi[:, -2]) / dz

    # Build 2D r_bar array
    R = r_bar[:, np.newaxis] * np.ones((1, nz))

    # sqrt(r_bar^2 - N^2), clamped to avoid NaN at r_bar = N
    r2_minus_N2 = np.clip(R**2 - N**2, 0.0, None)
    sqrt_term = np.sqrt(r2_minus_N2)

    # Prevent division by zero at r_bar = 0
    R_safe = np.where(R > 1e-15, R, 1e-15)

    # Eq. 8: dPhi/dtau = -(N/(2r))*dPhi/dz - (s/(2r))*sqrt(r^2-N^2)*dPhi/dr
    rhs = -(N / (2.0 * R_safe)) * dPhi_dz - (s / (2.0 * R_safe)) * sqrt_term * dPhi_dr

    return rhs


def compute_hamiltonian(
    Phi: NDArray[np.float64],
    r_bar: NDArray[np.float64],
    z_bar: NDArray[np.float64],
    N: float,
    s: int,
) -> NDArray[np.float64]:
    """Compute the Hamiltonian H for conservation check (Eq. 9).

    H = (s/(2*r_bar))*sqrt(r_bar^2 - N^2)*p_r + (N/(2*r_bar))*p_z

    where p_r = dPhi/dr_bar, p_z = dPhi/dz_bar. Since H has no explicit
    time dependence, it is conserved along each characteristic curve.

    Args:
        Phi: Flux function array, shape (nr, nz).
        r_bar: 1D radial coordinates.
        z_bar: 1D axial coordinates.
        N: Electrode radius ratio.
        s: Sign of radial velocity.

    Returns:
        H array, shape (nr, nz).
    """
    nr, nz = Phi.shape
    dr = r_bar[1] - r_bar[0] if nr > 1 else 1.0
    dz = z_bar[1] - z_bar[0] if nz > 1 else 1.0

    # p_r = dPhi/dr_bar
    p_r = np.zeros_like(Phi)
    p_r[1:-1, :] = (Phi[2:, :] - Phi[:-2, :]) / (2.0 * dr)
    p_r[0, :] = (Phi[1, :] - Phi[0, :]) / dr
    p_r[-1, :] = (Phi[-1, :] - Phi[-2, :]) / dr

    # p_z = dPhi/dz_bar
    p_z = np.zeros_like(Phi)
    p_z[:, 1:-1] = (Phi[:, 2:] - Phi[:, :-2]) / (2.0 * dz)
    p_z[:, 0] = (Phi[:, 1] - Phi[:, 0]) / dz
    p_z[:, -1] = (Phi[:, -1] - Phi[:, -2]) / dz

    R = r_bar[:, np.newaxis] * np.ones((1, nz))
    R_safe = np.where(R > 1e-15, R, 1e-15)
    r2_minus_N2 = np.clip(R**2 - N**2, 0.0, None)
    sqrt_term = np.sqrt(r2_minus_N2)

    H = (s / (2.0 * R_safe)) * sqrt_term * p_r + (N / (2.0 * R_safe)) * p_z
    return H


def solve_flux_evolution(
    Phi_0: NDArray[np.float64],
    r_bar: NDArray[np.float64],
    z_bar: NDArray[np.float64],
    N: float,
    s: int,
    dtau: float,
    n_steps: int = 1,
) -> NDArray[np.float64]:
    """Evolve the flux function Phi by integrating the PDE (Eq. 8).

    Uses SSP-RK2 (Shu-Osher) for time integration:
        Phi^(1) = Phi^n + dtau * L(Phi^n)
        Phi^(n+1) = 0.5*Phi^n + 0.5*(Phi^(1) + dtau*L(Phi^(1)))

    Args:
        Phi_0: Initial flux function, shape (nr, nz).
        r_bar: 1D radial coordinates (nr,).
        z_bar: 1D axial coordinates (nz,).
        N: Electrode radius ratio.
        s: Sign of radial velocity.
        dtau: Scaled timestep.
        n_steps: Number of integration steps.

    Returns:
        Evolved flux function Phi, shape (nr, nz).
    """
    Phi = Phi_0.copy()
    for _ in range(n_steps):
        # SSP-RK2
        L0 = _flux_pde_rhs(Phi, r_bar, z_bar, N, s)
        Phi_1 = Phi + dtau * L0
        L1 = _flux_pde_rhs(Phi_1, r_bar, z_bar, N, s)
        Phi = 0.5 * Phi + 0.5 * (Phi_1 + dtau * L1)
    return Phi


def compute_poloidal_field(
    a: float,
    b: float,
    I: float,  # noqa: E741
    rho_0: float,
    nr: int = 64,
    nz: int = 128,
    tau: float | None = None,
    n_evolve_steps: int = 10,
    B_seed: float = 5e-5,
    r_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Compute poloidal B_z field from the GV mechanism.

    Workflow:
    1. Set up normalized grid with r_bar in [N+eps, 1], z_bar in [-2, 2].
    2. Seed the flux function Phi from the geomagnetic field B_seed.
    3. Evolve Phi using the Hamilton-Jacobi PDE (Eq. 8).
    4. Extract B_z = p_r / (2*pi*a^2*r_bar) (Eq. 11).

    The seed field (Earth's ~50 uT) provides the initial Phi that gets
    amplified by the dynamo action of the moving sheath.

    Args:
        a: Anode radius [m].
        b: Cathode radius [m].
        I: Instantaneous current [A].
        rho_0: Fill gas density [kg/m^3].
        nr: Radial grid points.
        nz: Axial grid points.
        tau: Scaled time. If None, computed from I and device params.
        n_evolve_steps: Number of PDE evolution sub-steps.
        B_seed: Seed axial magnetic field [T] (geomagnetic, ~50 uT).
        r_range: Normalized radial domain.
        z_range: Normalized axial domain.

    Returns:
        B_z array, shape (nr, nz) [Tesla].
    """
    N = a / b
    s = -1  # implosion phase

    gv = compute_gv_surface(N, nr, nz, tau=0.0, s=s, r_range=r_range, z_range=z_range)
    r_bar = gv["r_bar"]
    z_bar = gv["z_bar"]

    # Seed flux function from uniform axial B_seed
    # Phi = integral_0^r B_z * 2*pi*r dr = pi * r^2 * B_seed (for uniform B_z)
    # In normalized coords: Phi = pi * (a*r_bar)^2 * B_seed
    R = r_bar[:, np.newaxis] * np.ones((1, nz))
    Phi = np.pi * (a * R) ** 2 * B_seed

    # Compute scaled time if not provided
    if tau is None:
        params = compute_scaling_params(a, rho_0, I)
        # tau ~ Q_m^{-1} * I * t_char where t_char ~ a/v_0
        v_0 = params["v_0"]
        Q_m = params["Q_m"]
        t_char = a / max(v_0, 1e-10)
        tau = abs(I) * t_char / max(Q_m, 1e-30)

    # CFL-limited timestep for the PDE
    dr = r_bar[1] - r_bar[0] if nr > 1 else 1.0
    dz = z_bar[1] - z_bar[0] if nz > 1 else 1.0

    # Max wave speed: max of |N/(2r)| and |s*sqrt(r^2-N^2)/(2r)|
    r_min = r_bar[0]
    c_max_z = N / (2.0 * r_min)
    c_max_r = np.sqrt(max(r_bar[-1] ** 2 - N**2, 0.0)) / (2.0 * r_min)
    c_max = max(c_max_z, c_max_r, 1e-15)
    dtau_cfl = 0.4 * min(dr, dz) / c_max
    dtau = min(dtau_cfl, tau / max(n_evolve_steps, 1))

    actual_steps = max(int(np.ceil(tau / max(dtau, 1e-30))), 1)
    dtau = tau / actual_steps

    # Evolve Phi
    Phi = solve_flux_evolution(Phi, r_bar, z_bar, N, s, dtau, actual_steps)

    # Extract B_z from Eq. 11: B_z = p_r / (2*pi*a^2*r_bar)
    # p_r = dPhi/dr_bar
    p_r = np.zeros_like(Phi)
    p_r[1:-1, :] = (Phi[2:, :] - Phi[:-2, :]) / (2.0 * dr)
    p_r[0, :] = (Phi[1, :] - Phi[0, :]) / dr
    p_r[-1, :] = (Phi[-1, :] - Phi[-2, :]) / dr

    R_safe = np.where(R > 1e-15, R, 1e-15)
    B_z = p_r / (2.0 * np.pi * a**2 * R_safe)

    return B_z


def compute_poloidal_Br(
    Phi: NDArray[np.float64],
    r_bar: NDArray[np.float64],
    z_bar: NDArray[np.float64],
    a: float,
) -> NDArray[np.float64]:
    """Compute radial component B_r from the flux function (Eq. 12).

    B_r = -p_z / (2*pi*a^2*r_bar) where p_z = dPhi/dz_bar.

    Args:
        Phi: Flux function array, shape (nr, nz).
        r_bar: 1D radial coordinates.
        z_bar: 1D axial coordinates.
        a: Anode radius [m].

    Returns:
        B_r array, shape (nr, nz) [Tesla].
    """
    nr, nz = Phi.shape
    dz = z_bar[1] - z_bar[0] if nz > 1 else 1.0

    p_z = np.zeros_like(Phi)
    p_z[:, 1:-1] = (Phi[:, 2:] - Phi[:, :-2]) / (2.0 * dz)
    p_z[:, 0] = (Phi[:, 1] - Phi[:, 0]) / dz
    p_z[:, -1] = (Phi[:, -1] - Phi[:, -2]) / dz

    R = r_bar[:, np.newaxis] * np.ones((1, nz))
    R_safe = np.where(R > 1e-15, R, 1e-15)

    B_r = -p_z / (2.0 * np.pi * a**2 * R_safe)
    return B_r


def compute_azimuthal_Etheta(
    H: NDArray[np.float64],
    r_bar: NDArray[np.float64],
    a: float,
    I: float,  # noqa: E741
    rho_0: float,
) -> NDArray[np.float64]:
    """Compute azimuthal electric field from the Hamiltonian (Eq. 10).

    E_theta = (mu_0*I) / (pi*a^2*r_bar*sqrt(2*mu_0*rho_0)) * H

    This field drives the azimuthal current J_theta that generates
    the poloidal B-field. It is proportional to I(t) and H.

    Args:
        H: Hamiltonian array, shape (nr, nz).
        r_bar: 1D radial coordinates.
        a: Anode radius [m].
        I: Instantaneous current [A].
        rho_0: Fill gas density [kg/m^3].

    Returns:
        E_theta array, shape (nr, nz) [V/m].
    """
    nz = H.shape[1]
    R = r_bar[:, np.newaxis] * np.ones((1, nz))
    R_safe = np.where(R > 1e-15, R, 1e-15)

    prefactor = MU_0 * I / (np.pi * a**2 * np.sqrt(2.0 * MU_0 * rho_0))
    E_theta = (prefactor / R_safe) * H
    return E_theta


def add_poloidal_field(
    state: dict[str, NDArray],
    current: float,
    a: float,
    b: float,
    rho_0: float,
    dr: float,
    dz: float,
    B_seed: float = 5e-5,
) -> dict[str, NDArray]:
    """Add poloidal B_z contribution to the MHD state.

    This is the integration point for MHD solvers. Call after the main
    MHD step to superpose the dynamo-generated poloidal field.

    The poloidal field is computed on a 2D (r,z) slice and broadcast
    to the 3D MHD grid. Since the GV theory is axisymmetric, B_z
    depends only on (r,z), not theta.

    Args:
        state: MHD state dict with keys rho, velocity, pressure, B.
        current: Circuit current [A].
        a: Anode radius [m].
        b: Cathode radius [m].
        rho_0: Fill gas density [kg/m^3].
        dr: Radial grid spacing [m].
        dz: Axial grid spacing [m].
        B_seed: Seed geomagnetic field [T].

    Returns:
        Updated state dict with poloidal B_z added to B[2] (axial component).
    """
    B = state["B"]

    # B is shape (3, nx, ny, nz) — extract grid dimensions
    if B.ndim == 4:
        nx, ny, nz_grid = B.shape[1], B.shape[2], B.shape[3]
    else:
        raise ValueError(f"B must be 4D (3, nx, ny, nz), got shape {B.shape}")

    # Compute on the (r, z) = (x-axis, z-axis) plane
    # In cylindrical DPF: x -> r, z -> z (axis 0 and 2 of the grid)
    nr = nx
    nz = nz_grid

    rho_avg = float(np.mean(state["rho"]))
    rho_eff = max(rho_avg, rho_0)

    B_z_2d = compute_poloidal_field(
        a, b, current, rho_eff, nr=nr, nz=nz,
        B_seed=B_seed,
    )

    # Broadcast to 3D: B_z_2d is (nr, nz), B[2] is (nx, ny, nz)
    # Add along the y-axis (azimuthal in cylindrical) uniformly
    B_z_3d = B_z_2d[:, np.newaxis, :] * np.ones((1, ny, 1))

    state = dict(state)  # shallow copy
    B_new = B.copy()
    B_new[2] += B_z_3d
    state["B"] = B_new

    return state
