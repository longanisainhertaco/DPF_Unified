"""Circuit-MHD coupling and two-temperature source terms for the MLX solver.

Extracted from MLXMHDSolver (God Class decomposition). Contains:
  - Plasma inductance computation (Lee formula, density-weighted r_eff)
  - Two-temperature electron-ion equilibration source terms
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dpf.core.bases import CouplingState
from dpf.metal.constants import MU_0


_COUPLING_METHOD_AUTHORITY: dict[str, dict[str, Any]] = {
    "density_weighted_lp": {
        "method": "density_weighted_lp",
        "implementation_path": "dpf.metal.mlx_coupling.update_coupling",
        "classification": "engineering_coupling_scaffold",
        "source_status": "missing_same_scope_validation_packet",
        "validation_status": "not_validation_evidence",
        "can_support_scientific_claims": False,
        "claim_limit": (
            "Computes a finite field-derived coupling signal for engineering "
            "experiments; it is not accepted device-validation evidence."
        ),
        "guardrails": [
            "requires accepted same-scope source packet before scientific use",
            "synthetic Bennett tests verify plumbing only",
            "engine handoff must remain explicitly labeled",
        ],
    },
    "voltage_flux": {
        "method": "voltage_flux",
        "implementation_path": "dpf.metal.mlx_coupling.compute_upf_voltage_flux",
        "classification": "experimental_coupling_scaffold",
        "source_status": "unreviewed_or_missing_local_source",
        "validation_status": "not_validation_evidence",
        "can_support_scientific_claims": False,
        "claim_limit": (
            "Flux-derived terminal voltage is an experimental implementation "
            "path and is not currently wired as accepted validation evidence."
        ),
        "guardrails": [
            "do not describe as correct or source-closed",
            "requires local source review plus same-scope comparator",
        ],
    },
    "poynting_voltage": {
        "method": "poynting_voltage",
        "implementation_path": "dpf.metal.mlx_coupling.compute_voltage_poynting",
        "classification": "experimental_coupling_scaffold",
        "source_status": "theory_context_without_accepted_validation_packet",
        "validation_status": "not_validation_evidence",
        "can_support_scientific_claims": False,
        "claim_limit": (
            "Poynting-theorem voltage is available for experiments only until "
            "same-scope source and uncertainty packets are accepted."
        ),
        "guardrails": [
            "do not describe as first-principles validation",
            "requires accepted local source packet and uncertainty budget",
        ],
    },
}


def coupling_method_authority() -> dict[str, dict[str, Any]]:
    """Return fail-closed authority labels for MLX circuit-coupling methods."""
    return {
        name: {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in record.items()
        }
        for name, record in _COUPLING_METHOD_AUTHORITY.items()
    }


def evaluate_mhd_coupling_gate(
    *,
    phase: str,
    lp_mhd: float,
    lp_snowplow: float,
    dlp_dt_mhd: float | None,
    resistance_mhd: float,
    allowed_phases: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Evaluate engineering eligibility for MHD-derived circuit coupling.

    This gate is intentionally stricter than "finite and positive" while still
    failing closed for scientific claims. Passing it means the value may be used
    for the engineering blend only; accepted source evidence is still required
    before MHD-derived Lp, dL/dt, or resistance can support validation claims.
    """
    checks = {
        "phase_allowed": phase in allowed_phases,
        "lp_finite": bool(np.isfinite(lp_mhd)),
        "lp_positive": lp_mhd > 0.0,
        "lp_comparable_to_snowplow": (
            lp_snowplow <= 0.0 or lp_mhd >= 0.5 * lp_snowplow
        ),
        "dlp_dt_finite_or_unset": (
            dlp_dt_mhd is None or bool(np.isfinite(dlp_dt_mhd))
        ),
        "resistance_finite": bool(np.isfinite(resistance_mhd)),
        "resistance_nonnegative": resistance_mhd >= 0.0,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    eligible = not failed_checks
    return {
        "classification": "engineering_blend_gate",
        "validation_status": "not_validation_evidence",
        "scientific_gate_status": "blocked_missing_same_scope_validation_packet",
        "can_support_scientific_claims": False,
        "eligible_for_engineering_blend": eligible,
        "phase": phase,
        "allowed_phases": sorted(allowed_phases),
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _bdf2_dLp_dt(history: list[tuple[float, float]], Lp: float, t: float) -> float:
    """Compute dLp/dt using BDF2 (3-point) when history is available.

    Falls back to backward difference with 2 points, returns 0.0 with < 2.

    BDF2: dLp/dt = (3*Lp_n - 4*Lp_{n-1} + Lp_{n-2}) / (2*dt)
    Non-uniform dt variant used when timestep varies.

    References
    ----------
    Hairer & Wanner, "Solving ODEs II", Springer 1996, Sec. III.1.
    Matches CircuitCoupler._compute_dLp_dt in circuit/coupler.py.
    """
    if len(history) >= 2:
        t1, Lp1 = history[-1]
        t0, Lp0 = history[-2]
        dt1 = t - t1
        dt0 = t1 - t0
        if dt1 > 0 and dt0 > 0:
            # Non-uniform BDF2: weighted finite difference
            r = dt1 / dt0
            denom = dt1 * (1.0 + r)
            return ((1.0 + 2.0 * r) * Lp - (1.0 + r) ** 2 * Lp1 + r**2 * Lp0) / denom
    if len(history) >= 1:
        t1, Lp1 = history[-1]
        dt1 = t - t1
        if dt1 > 0:
            return (Lp - Lp1) / dt1
    return 0.0


def compute_upf_voltage_flux(
    U: Any,
    grid: Any,
    r_inner: float,
    cathode_radius: float,
    phi_history: list[tuple[float, float]],
    sim_time: float,
    current: float,
    voltage: float,
) -> tuple[CouplingState, float]:
    """Estimate DPF terminal voltage U_PF from magnetic flux at inlet boundary.

    This is an experimental MHD-circuit coupling path. Instead of computing
    Lp and dLp/dt from sheath detection and density-weighted radius, it computes
    magnetic flux Phi at the inlet boundary and derives U_PF = dPhi/dt. The
    circuit equation becomes:

        L0 * dI/dt = V_cap - R*I - U_PF

    This implementation is not accepted validation evidence. Keep it labeled
    as an engineering scaffold until local source review and same-scope
    comparator evidence exist.

    References
    ----------
    # UNVERIFIED: Sun et al. 2025 (Acta Physica Sinica 74:115201) — paper not on
    # disk as of 2026-04-24. Implementation behavior kept but citation cannot be
    # verified.
    Sun et al. (2025), Acta Physica Sinica 74:115201, Eq. (15)-(17):
        U_PF = d(Phi)/dt, Phi = integral(B * dS)
        [UNVERIFIED — referenced PDF path
        references/papers/core-dpf/2025_Theoretical_and_numerical_studies_
        on_motion_process_of_dense_plasma_focus.pdf does not exist on disk.]

    Beresnyak et al. (2018), IEEE TPS 46:3881 (NRL HAWK DPF):
        V_DPF = integral(E . dl) across device terminals.
        Extracted in memory/dpf-papers/dpf-high-impedance-sims.md.

    Auluck (2021), Phys. Plasmas 28:030703, Eq. (10)-(13):
        Theoretical context for limitations of inductance-only coupling.
        PDF: references/papers/core-dpf/auluck-2021-dpf-circuit-element.pdf

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz).
    grid : object
        Grid with .dr attribute.
    r_inner : float
        Anode radius [m].
    cathode_radius : float
        Cathode radius [m].
    phi_history : list of (time, Phi) tuples
        History for BDF2 dPhi/dt. Mutated in-place.
    sim_time : float
        Current simulation time [s].
    current : float
        Circuit current [A].
    voltage : float
        Capacitor voltage [V].

    Returns
    -------
    tuple[CouplingState, float]
        (coupling_state, U_PF_voltage)
    """
    from dpf.metal.mlx_kernels import IBT

    Bt_np = np.asarray(U[IBT])  # (nr, nz)
    dr = grid.dr
    dz = grid.dz
    _ = r_inner, cathode_radius

    # Magnetic flux linkage for a coaxial DPF device.
    #
    # UNVERIFIED: Sun et al. 2025 (Acta Physica Sinica 74:115201) — paper not on
    # disk as of 2026-04-24. Implementation behavior kept but citation cannot be
    # verified.
    # Sun et al. (2025) Eq. (17): Phi = integral(B . dS)
    # For the axisymmetric geometry, the flux per unit axial length is:
    #     Phi_per_length = integral_a^b B_theta(r) dr
    # The total flux linkage is the integral over the axial extent where
    # B_theta exists (i.e., where current flows between the electrodes):
    #     Phi_total = integral_0^z_max [ integral_a^b B_theta(r,z) dr ] dz
    #
    # This captures the axial extent of the current sheath in the computed
    # field state: where B_theta is zero ahead of the sheath, that z-column
    # contributes zero flux. This remains an experimental implementation path,
    # not accepted validation evidence.
    #
    # In HL units (mu0=1 in solver): B_HL = B_SI / sqrt(mu0)
    # Phi_SI = sqrt(mu0) * integral(B_HL * dr * dz)
    #
    # Verification: for vacuum coaxial B_theta = mu0*I/(2*pi*r):
    #   Phi_per_length = integral_a^b (mu0*I)/(2*pi*r) dr = (mu0*I)/(2*pi) * ln(b/a)
    #   Phi_total = Phi_per_length * z_domain
    #   L = Phi/I = (mu0/(2*pi)) * ln(b/a) * z = Lee formula
    # In this idealized vacuum-field check, the flux integral reduces to the
    # Lee-form inductance expression. Plasma-compression behavior still needs
    # same-scope source closure before it can support scientific claims.
    sqrt_mu0 = math.sqrt(MU_0)

    # Integrate B_theta over full (r,z) domain
    # Phi_HL = sum over all (r,z) cells of B_theta * dr * dz
    Phi_HL = float(np.sum(Bt_np * dr * dz))
    Phi_SI = Phi_HL * sqrt_mu0

    # dPhi/dt via BDF2 (reuse existing BDF2 function)
    dPhi_dt = _bdf2_dLp_dt(phi_history, Phi_SI, sim_time)
    phi_history.append((sim_time, Phi_SI))
    while len(phi_history) > 3:
        phi_history.pop(0)

    # U_PF = dPhi/dt [Volts]
    U_PF = dPhi_dt

    # Back-compute Lp for diagnostic comparison with snowplow
    current * current
    Lp_diagnostic = Phi_SI / max(abs(current), 1.0)  # Phi = L * I => L = Phi / I

    coupling = CouplingState(
        Lp=Lp_diagnostic,
        current=current,
        voltage=voltage,
        dL_dt=U_PF / max(abs(current), 1.0) if abs(current) > 1.0 else 0.0,
    )
    coupling._diag_Phi_SI = Phi_SI
    coupling._diag_U_PF = U_PF

    return coupling, U_PF


def compute_voltage_poynting(
    U: Any,
    grid: Any,
    current: float,
    eta: Any | None = None,
    convert_hl_to_si: bool = True,
) -> float:
    """Estimate device voltage from Poynting theorem (Auluck 2021 Eq. 1).

    V_12(t) = -(1/I(t)) * integral_Omega(J . E d^3r)

    This is an experimental coupling path, not accepted validation evidence.
    It avoids an inductance formula by integrating J.E over the domain, but it
    still needs local source review, same-scope comparator evidence, and an
    uncertainty budget before scientific use.

    References
    ----------
    Auluck (2021), Phys. Plasmas 28:030703, Eq. (1), (6).
        PDF: references/papers/core-dpf/auluck-2021-dpf-circuit-element.pdf
        Theoretical context for limits of Lp-based coupling and dPhi/dt.
        The motional impedance (1/2 I^2 dL/dt) != magnetic energy flux through
        the plasma boundary (Eq. 13). The difference is the "anomalous impedance."

    Parameters
    ----------
    U : array
        Conserved state (NVAR, nr, nz) in HL units.
    grid : object
        Grid with .dr, .dz, .r_cell attributes.
    current : float
        Circuit current [A].
    eta : float or array or None
        Resistivity [Ohm*m]. None = ideal MHD (eta=0).
    convert_hl_to_si : bool
        If True, B in U is HL and will be converted to SI.

    Returns
    -------
    float
        Device voltage U_PF [V]. Positive = back-EMF opposing current.
    """
    from dpf.metal.constants import IBR, IBT, IBZ, IDN, IMR, IMZ

    U_np = np.asarray(U)
    _nr, _nz = U_np.shape[1], U_np.shape[2]
    dr = grid.dr
    dz = grid.dz
    r = np.asarray(grid.r_cell)  # (nr,)
    r_safe = np.maximum(r, 1e-10)

    # Extract fields
    rho = np.maximum(U_np[IDN], 1e-30)
    vr = U_np[IMR] / rho
    vz = U_np[IMZ] / rho
    Bt = U_np[IBT]  # B_theta
    Br = U_np[IBR]
    Bz = U_np[IBZ]

    # Convert B to SI if stored in HL
    sqrt_mu0 = math.sqrt(MU_0)
    if convert_hl_to_si:
        Bt_si = Bt * sqrt_mu0
        Br_si = Br * sqrt_mu0
        Bz_si = Bz * sqrt_mu0
    else:
        Bt_si, Br_si, Bz_si = Bt, Br, Bz

    # Current density: J = curl(B)/mu0 in cylindrical (r, theta, z)
    # J_r = -(1/mu0) * dBt/dz
    # J_z = (1/(mu0*r)) * d(r*Bt)/dr
    # J_t = (1/mu0) * (dBr/dz - dBz/dr)
    dBt_dz = np.gradient(Bt_si, dz, axis=1)
    rBt = r_safe[:, None] * Bt_si
    d_rBt_dr = np.gradient(rBt, dr, axis=0)
    dBr_dz = np.gradient(Br_si, dz, axis=1)
    dBz_dr = np.gradient(Bz_si, dr, axis=0)

    inv_mu0 = 1.0 / MU_0
    Jr = -dBt_dz * inv_mu0
    Jz = d_rBt_dr / (r_safe[:, None]) * inv_mu0
    Jt = (dBr_dz - dBz_dr) * inv_mu0

    # Electric field: E = -v x B + eta*J  (Generalized Ohm's law)
    # Auluck (2021) Eq. (4): E = -v x B + eta*J
    # In cylindrical with v=(vr, 0, vz), B=(Br, Bt, Bz):
    #   (v x B)_r = vt*Bz - vz*Bt = -vz*Bt  (vt=0)
    #   (v x B)_z = vr*Bt - vt*Br = vr*Bt    (vt=0)
    #   (v x B)_t = vz*Br - vr*Bz
    eta_val = eta if eta is not None else 0.0
    if isinstance(eta_val, (int, float)):
        eta_arr = eta_val
    else:
        eta_arr = np.asarray(eta_val)

    Er = vz * Bt_si + eta_arr * Jr
    Ez = -vr * Bt_si + eta_arr * Jz
    Et = vz * Br_si - vr * Bz_si + eta_arr * Jt

    # J . E (power density, W/m^3)
    JdotE = Jr * Er + Jz * Ez + Jt * Et

    # Volume integral: P = 2*pi * sum(J.E * r * dr * dz)
    # The 2*pi comes from integrating over azimuthal angle (axisymmetry)
    P_total = 2.0 * math.pi * float(np.sum(JdotE * r_safe[:, None] * dr * dz))

    # V = -P / I  (Auluck 2021 Eq. 1)
    if abs(current) < 1.0:
        return 0.0
    return -P_total / current


def update_coupling(
    U: Any,
    current: float,
    voltage: float,
    dt: float,
    grid: Any,
    cathode_radius: float,
    r_inner: float,
    prev_Lp: float,
    Lp_max: float,
    coordinates: str,
    Lp_history: list[tuple[float, float]] | None = None,
    sim_time: float = 0.0,
) -> tuple[CouplingState, float, float]:
    """Compute plasma inductance and return circuit coupling state.

    Uses the Lee formula: Lp = (mu0/2pi) * z_sheath * ln(b/r_eff)
    where r_eff is the density-weighted effective radius.

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz).
    current, voltage : float
        Circuit current [A] and voltage [V].
    dt : float
        Timestep [s].
    grid : object
        Grid with .dr, .dz attributes.
    cathode_radius : float
        Outer electrode radius [m].
    r_inner : float
        Inner radial boundary [m].
    prev_Lp : float
        Previous timestep inductance [H].
    Lp_max : float
        Peak inductance seen so far [H].
    coordinates : str
        "cylindrical" or "cartesian".
    Lp_history : list of (time, Lp) tuples, optional
        History for BDF2 dLp/dt. Mutated in-place (appended to).
        Kept to max 3 entries.
    sim_time : float
        Current simulation time [s] (for BDF2 time stamps).

    Returns
    -------
    tuple[CouplingState, float, float]
        (coupling_state, new_prev_Lp, new_Lp_max)

    References
    ----------
    Lee & Saw, J. Fusion Energy 33:319 (2014), Eq. L_p = (mu0/2pi)*z*ln(b/r_p).
    Malir et al., Phys. Plasmas 31:042513 (2024) — density profiles confirm
        sheath appears as radial density peak detectable on-axis.

    Engineering regression
    ----------------------
    Synthetic Bennett pinch (a=5mm, b=160mm, z=300mm): L_p error 4.5%
    vs analytic in plumbing tests. This is not device-validation evidence.
    """
    from dpf.metal.mlx_kernels import IDN

    if coordinates == "cartesian":
        return CouplingState(current=current, voltage=voltage), prev_Lp, Lp_max

    rho_np = np.asarray(U[IDN])  # (nr, nz)
    nr, nz = rho_np.shape
    dr = grid.dr
    dz = grid.dz
    r_arr = r_inner + (np.arange(nr) + 0.5) * dr

    # Sheath position from ANNULAR density peak.
    #
    # During axial rundown, the sheath is an annular sheet between the
    # electrodes — it has NOT reached the axis. On-axis density is just
    # undisturbed fill gas, so on-axis detection reports z_sheath = full
    # anode length, making L_p ~2x too high (Gemini 3.1 Pro, 2026-04-09).
    #
    # Fix: average density across the full radial extent for each z-column.
    # The annular sheath produces a density peak at its z-position in
    # this average, regardless of whether it has reached the axis.
    #
    # Lee & Saw, J. Fusion Energy 33:319 (2014): L_p = (mu0/2pi)*z*ln(b/r_p)
    # Malir et al., Phys. Plasmas 31:042513 (2024): sheath is annular
    #   during rundown, confirmed by interferometric imaging (Figs. 4-5).
    rho_annular_avg = np.mean(rho_np, axis=0)  # average over r for each z
    rho_fill = float(np.median(rho_annular_avg))  # background fill density

    # Sheath front: first z where density exceeds 1.5x fill (compression)
    # Search from z=nz-1 (anode end) backward toward z=0 (insulator end)
    threshold = 1.5 * max(rho_fill, 1e-30)
    compressed = rho_annular_avg > threshold
    if np.any(compressed):
        iz_sheath = int(np.max(np.where(compressed)[0]))
    else:
        # No compression detected — use full domain as fallback
        iz_sheath = nz - 1
    z_sheath = (iz_sheath + 0.5) * dz

    # Density-weighted effective radius in the compressed region
    rho_region = rho_np[:, : iz_sheath + 1]
    r_col = r_arr[:, np.newaxis]
    dV = 2.0 * math.pi * r_col * dr * dz
    mass = rho_region * dV
    total_mass = float(np.sum(mass))
    if total_mass > 0:
        r_eff = float(np.sum(r_col * mass) / total_mass)
    else:
        r_eff = 0.5 * cathode_radius
    r_eff = max(r_eff, 1e-6)
    r_eff = min(r_eff, cathode_radius * 0.999)

    # Lee formula: L_p = (mu0/2pi) * z_sheath * ln(b/r_eff)
    if r_eff > 0 and z_sheath > 0:
        Lp = (MU_0 / (2.0 * math.pi)) * z_sheath * math.log(
            cathode_radius / r_eff
        )
    else:
        Lp = 0.0

    # Phase-aware monotonicity
    if Lp > Lp_max:
        Lp_max = Lp
    elif Lp < Lp_max * 0.98:
        pass
    else:
        Lp = Lp_max

    # dL/dt via BDF2 (3-point) or backward difference (2-point)
    if Lp_history is not None:
        dL_dt = _bdf2_dLp_dt(Lp_history, Lp, sim_time)
        Lp_history.append((sim_time, Lp))
        # Keep only last 3 entries
        while len(Lp_history) > 3:
            Lp_history.pop(0)
    else:
        dL_dt = (Lp - prev_Lp) / dt if prev_Lp > 0 and dt > 0 else None
    prev_Lp = Lp

    coupling = CouplingState(
        Lp=Lp, current=current, voltage=voltage, dL_dt=dL_dt,
    )
    # Diagnostic attributes for PIRT analysis (not part of CouplingState contract)
    coupling._diag_z_sheath = z_sheath
    coupling._diag_r_eff = r_eff
    coupling._diag_iz_sheath = iz_sheath
    return coupling, prev_Lp, Lp_max


def do_two_temperature_sources(
    result: dict[str, np.ndarray],
    dt: float,
    eta_field: float | np.ndarray | None,
    ion_mass: float,
    dx: float,
    Z_eff: float,
    gaunt_factor: float,
    gamma: float,
) -> None:
    """Apply electron-ion equilibration, Ohmic heating, bremsstrahlung.

    Modifies result dict in-place.
    """
    from dpf.fluid.two_temperature import step_electron_energy

    rho = result["rho"]
    n_i = rho / ion_mass
    n_i_safe = np.maximum(n_i, 1e-30)
    eta_np = (
        np.asarray(eta_field) if eta_field is not None
        else np.zeros_like(rho)
    )
    J_sq = np.zeros_like(rho)
    e_e_new, Te_new, Ti_new = step_electron_energy(
        rho_e_e=result["e_electron"],
        rho=rho,
        velocity=result["velocity"],
        eta=eta_np,
        J_sq=J_sq,
        Te=result["Te"],
        Ti=result["Ti"],
        n_e=n_i_safe,
        n_i=n_i_safe,
        dx=dx,
        dt=dt,
        Z=Z_eff,
        gaunt_factor=gaunt_factor,
        gamma=gamma,
    )
    result["Te"] = np.maximum(Te_new, 1.0)
    result["Ti"] = np.maximum(Ti_new, 1.0)
    result["e_electron"] = e_e_new
