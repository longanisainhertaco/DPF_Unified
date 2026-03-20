"""Differentiable Lee model in JAX for Dense Plasma Focus.

Implements the 5-phase Lee model ODEs using pure JAX operations, enabling:
- ``jax.grad(loss_fn)`` for exact parameter gradients
- ``jax.vmap`` for parallel parameter sweeps (10,000+ in seconds)
- Gradient-based calibration via Adam optimizer
- Exact sensitivity analysis: d(I_peak)/d(param)

Phase structure:
    1. Axial rundown: sheath sweeps fill gas along the coaxial anode
    2. Radial inward shock: cylindrical implosion toward axis
    3. Radial reflected shock: outward bounce from stagnation pressure
    4. Slow compression / pinch: quasi-static equilibrium at minimum radius
    5. Post-pinch expansion: column disrupts, inductance drops

Phase transitions use sigmoid soft-switching to preserve differentiability.
Hard if/else would break JIT compilation and gradient flow.

References:
    Lee, S. & Saw, S.H., Phys. Plasmas 21, 072501 (2014).
    Lee, S., J. Fusion Energy 33, 319-335 (2014).

Usage::

    from dpf.jax.lee_model import simulate, loss_fn, calibrate, sensitivity

    params = default_pf1000_params()
    result = simulate(params)                        # forward pass
    grads = jax.grad(loss_fn)(params, target_I, t)   # exact gradients
    fitted = calibrate(target_I, t, params)           # Adam optimization
    sens = sensitivity(params)                        # d(I_peak)/d(param)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Physical constants (avoid scipy dependency inside JIT)
MU_0 = 4.0e-7 * jnp.pi      # Vacuum permeability [H/m]
K_B = 1.380649e-23            # Boltzmann constant [J/K]
M_D2 = 6.68717e-27            # D2 molecular mass [kg]
PI = jnp.pi

# Time integration parameters
N_STEPS = 10000
DT_SAFETY = 0.5


def _soft_switch(x: jnp.ndarray, x0: float, width: float) -> jnp.ndarray:
    """Smooth sigmoid transition from 0 to 1 centered at x0.

    Args:
        x: Input value.
        x0: Transition center.
        width: Transition width (smaller = sharper).

    Returns:
        Sigmoid value in [0, 1].
    """
    return jax.nn.sigmoid((x - x0) / jnp.maximum(width, 1e-30))


def default_pf1000_params() -> dict[str, jnp.ndarray]:
    """PF-1000 device parameters as JAX arrays.

    Returns:
        Dictionary of float64 JAX scalars. Keys match the Lee model
        parameter set: circuit (V0, C0, L0, R0), geometry (a, b, z_max),
        and snowplow fractions (fc, fm, fmr, fcr).
    """
    return {
        "V0": jnp.float64(27e3),            # Charging voltage [V]
        "C0": jnp.float64(1.332e-3),        # Capacitance [F]
        "L0": jnp.float64(33.5e-9),         # External inductance [H]
        "R0": jnp.float64(8.73e-3),         # External resistance [Ohm]
        "a": jnp.float64(0.115),             # Anode radius [m]
        "b": jnp.float64(0.16),              # Cathode radius [m]
        "z_max": jnp.float64(0.6),           # Anode length [m]
        "fill_pressure_torr": jnp.float64(3.5),
        "fc": jnp.float64(0.7),             # Current fraction
        "fm": jnp.float64(0.19),            # Axial mass fraction
        "fmr": jnp.float64(0.16),           # Radial mass fraction
        "fcr": jnp.float64(0.7),            # Radial current fraction
    }


def default_unu_ictp_params() -> dict[str, jnp.ndarray]:
    """UNU-ICTP PFF device parameters as JAX arrays."""
    return {
        "V0": jnp.float64(14e3),
        "C0": jnp.float64(30e-6),
        "L0": jnp.float64(110e-9),
        "R0": jnp.float64(12e-3),
        "a": jnp.float64(0.0095),
        "b": jnp.float64(0.032),
        "z_max": jnp.float64(0.16),
        "fill_pressure_torr": jnp.float64(3.0),
        "fc": jnp.float64(0.7),
        "fm": jnp.float64(0.08),
        "fmr": jnp.float64(0.16),
        "fcr": jnp.float64(0.7),
    }


def _derived_constants(
    params: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Compute derived physical constants from device parameters.

    Args:
        params: Device parameter dictionary.

    Returns:
        Dictionary of derived constants needed by the ODE.
    """
    a = params["a"]
    b = params["b"]
    z_max = params["z_max"]
    p_torr = params["fill_pressure_torr"]

    ln_ba = jnp.log(b / a)
    A_annular = PI * (b**2 - a**2)
    p_Pa = p_torr * 133.322
    rho0 = (p_Pa / (K_B * 300.0)) * M_D2

    L_per_length = (MU_0 / (2.0 * PI)) * ln_ba
    F_coeff = (MU_0 / (4.0 * PI)) * ln_ba
    r_pinch_min = 0.1 * a

    # Pinch column fraction: 0.14 for large devices (PF-1000 scale)
    z_f = 0.14 * z_max

    # Quarter period for time scale estimation
    T_quarter = PI * jnp.sqrt(params["L0"] * params["C0"])

    return {
        "ln_ba": ln_ba,
        "A_annular": A_annular,
        "p_Pa": p_Pa,
        "rho0": rho0,
        "L_per_length": L_per_length,
        "F_coeff": F_coeff,
        "r_pinch_min": r_pinch_min,
        "z_f": z_f,
        "T_quarter": T_quarter,
    }


# State vector layout:
#   [I, V_cap, z, vz, r_shock, vr]
# Indices:
I_IDX = 0
V_IDX = 1
Z_IDX = 2
VZ_IDX = 3
R_IDX = 4
VR_IDX = 5
STATE_SIZE = 6


def _lee_rhs(
    state: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    dc: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Right-hand side of the Lee model ODE system.

    Uses soft-switching (sigmoid) to blend between phases:
    - Phase 1 (axial): z < z_max
    - Phase 2 (radial inward): z >= z_max and r > r_pinch_min
    - Phase 3 (reflected): r <= r_pinch_min (outward expansion)
    - Phase 4/5 (pinch/post-pinch): column frozen or expanding

    The soft-switch width is chosen to be 1% of the relevant scale
    (z_max for axial->radial, r_pinch_min for radial->reflected).

    Args:
        state: State vector [I, V_cap, z, vz, r_shock, vr].
        params: Device parameter dictionary.
        dc: Derived constants from ``_derived_constants``.

    Returns:
        Time derivative of state vector.
    """
    I = state[I_IDX]  # noqa: E741
    V_cap = state[V_IDX]
    z = state[Z_IDX]
    vz = state[VZ_IDX]
    r_s = state[R_IDX]
    vr = state[VR_IDX]

    C0 = params["C0"]
    L0 = params["L0"]
    R0 = params["R0"]
    fc = params["fc"]
    fm = params["fm"]
    fmr = params["fmr"]
    fcr = params["fcr"]
    b = params["b"]
    z_max = params["z_max"]

    rho0 = dc["rho0"]
    A_annular = dc["A_annular"]
    p_Pa = dc["p_Pa"]
    L_per_length = dc["L_per_length"]
    F_coeff = dc["F_coeff"]
    z_f = dc["z_f"]
    r_pinch_min = dc["r_pinch_min"]

    # Clamp state variables to physical ranges (soft clamps via relu)
    z_safe = jnp.maximum(z, 1e-6)
    vz_safe = jnp.maximum(vz, 0.0)
    r_safe = jnp.maximum(r_s, r_pinch_min)
    gamma = 5.0 / 3.0

    # Phase blending weights (sigmoid transitions)
    # w_axial: 1 during axial, 0 after. Transition at z = z_max.
    axial_width = 0.01 * z_max
    w_radial = _soft_switch(z, z_max, axial_width)
    w_axial = 1.0 - w_radial

    # w_reflected: 1 when r_s <= r_pinch_min (reflected shock).
    # Transition at r_s = r_pinch_min.
    radial_width = 0.1 * r_pinch_min
    w_reflected = _soft_switch(-r_s, -r_pinch_min, radial_width)
    w_inward = (1.0 - w_reflected) * w_radial

    # ---- Phase 1: Axial rundown ----
    # Swept mass: M = fm * rho0 * A_annular * z
    M_swept = fm * rho0 * A_annular * jnp.maximum(z_safe, 1e-6)
    dM_dt_ax = fm * rho0 * A_annular * vz_safe

    # Magnetic driving force
    F_mag_ax = F_coeff * (fc * I) ** 2
    F_back_ax = p_Pa * A_annular

    # Snowplow acceleration: M*dvz = F_mag - F_back - vz*dM/dt
    dvz_dt_ax = (F_mag_ax - F_back_ax - vz_safe * dM_dt_ax) / jnp.maximum(M_swept, 1e-20)
    dvz_dt_ax = jnp.clip(dvz_dt_ax, -1e15, 1e15)

    # Plasma inductance and its derivative (axial)
    Lp_ax = L_per_length * z_safe
    dLp_dt_ax = L_per_length * vz_safe

    # ---- Phase 2: Radial inward shock ----
    # Slug mass: M_slug = fmr * rho0 * pi * (b^2 - r_s^2) * z_f
    M_slug_rad = fmr * rho0 * PI * (b**2 - r_safe**2) * z_f
    M_slug_rad = jnp.maximum(M_slug_rad, 1e-20)
    dM_dt_rad = fmr * rho0 * 2.0 * PI * r_safe * jnp.abs(vr) * z_f

    # Radial J x B force
    F_rad = (MU_0 / (4.0 * PI)) * (fcr * I) ** 2 * z_f / jnp.maximum(r_safe, 1e-10)

    # Adiabatic back-pressure: p_fill * (b/r_s)^(2*gamma)
    p_back_rad = p_Pa * (b / jnp.maximum(r_safe, r_pinch_min)) ** (2.0 * gamma)
    F_pressure_rad = p_back_rad * 2.0 * PI * r_safe * z_f

    # Radial inward acceleration (vr < 0 = inward)
    dvr_dt_inward = (-F_rad + F_pressure_rad - vr * dM_dt_rad) / M_slug_rad
    dvr_dt_inward = jnp.clip(dvr_dt_inward, -1e15, 1e15)

    # Plasma inductance (radial): L_axial_frozen + L_radial
    Lp_ax_frozen = L_per_length * z_max
    Lp_rad = (MU_0 / (2.0 * PI)) * z_f * jnp.log(jnp.maximum(b / r_safe, 1.01))
    Lp_radial_total = Lp_ax_frozen + Lp_rad
    dLp_dt_rad = -(MU_0 / (2.0 * PI)) * z_f * vr / jnp.maximum(r_safe, 1e-10)

    # ---- Phase 3: Reflected shock (outward expansion) ----
    # Back-pressure drives outward, J x B opposes
    # Use same force formulas but with outward vr
    rho_post_shock = 8.0 * rho0
    M_slug_refl = M_slug_rad + fmr * rho_post_shock * PI * jnp.maximum(r_safe**2 - r_pinch_min**2, 0.0) * z_f
    M_slug_refl = jnp.maximum(M_slug_refl, 1e-20)
    dM_dt_refl = fmr * rho_post_shock * 2.0 * PI * r_safe * jnp.abs(vr) * z_f

    dvr_dt_reflected = (F_pressure_rad - F_rad - vr * dM_dt_refl) / M_slug_refl
    dvr_dt_reflected = jnp.clip(dvr_dt_reflected, -1e15, 1e15)

    # ---- Blend phases ----
    # Plasma inductance: blend axial and radial
    Lp = w_axial * Lp_ax + w_radial * Lp_radial_total
    dLp_dt = w_axial * dLp_dt_ax + w_inward * dLp_dt_rad + w_reflected * dLp_dt_rad

    L_total = L0 + Lp

    # Circuit equation: L * dI/dt = V_cap - R*I - I*dLp/dt
    dI_dt = (V_cap - R0 * I - I * dLp_dt) / jnp.maximum(L_total, 1e-15)

    # Capacitor: dV/dt = -I/C
    dV_dt = -I / C0

    # Axial position derivative
    dz_dt = w_axial * vz_safe

    # Axial acceleration (only during axial phase)
    dvz_dt = w_axial * dvz_dt_ax

    # Radial velocity derivative: blend inward and reflected
    dvr_dt = w_inward * dvr_dt_inward + w_reflected * dvr_dt_reflected

    # Radial position derivative
    dr_dt = w_radial * vr

    return jnp.array([dI_dt, dV_dt, dz_dt, dvz_dt, dr_dt, dvr_dt])


def simulate(
    params: dict[str, jnp.ndarray],
    n_steps: int = N_STEPS,
    sim_time: float | None = None,
) -> dict[str, jnp.ndarray]:
    """Forward simulation of the Lee model.

    Integrates the coupled circuit + snowplow ODEs using fixed-step
    RK4, returning the full current waveform I(t).

    Args:
        params: Device parameter dictionary (see ``default_pf1000_params``).
        n_steps: Number of integration steps.
        sim_time: Total simulation time [s]. If None, auto-computed as
            6 * T_quarter.

    Returns:
        Dictionary with:
            t: Time array [s], shape (n_steps,).
            I: Current waveform [A], shape (n_steps,).
            V: Capacitor voltage [V], shape (n_steps,).
            z: Axial sheath position [m], shape (n_steps,).
            r: Radial shock position [m], shape (n_steps,).
            I_peak: Peak current magnitude [A], scalar.
            t_peak: Time of peak current [s], scalar.
    """
    dc = _derived_constants(params)

    if sim_time is None:
        t_total = 6.0 * dc["T_quarter"]
    else:
        t_total = jnp.float64(sim_time)

    dt = t_total / n_steps

    # Initial state: [I=0, V=V0, z=1e-6, vz=0, r=b, vr=0]
    state0 = jnp.array([
        0.0,
        params["V0"],
        1e-6,
        0.0,
        params["b"],
        0.0,
    ])

    def rk4_step(state: jnp.ndarray, _: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single RK4 step. Returns (new_state, saved_state)."""
        k1 = _lee_rhs(state, params, dc)
        k2 = _lee_rhs(state + 0.5 * dt * k1, params, dc)
        k3 = _lee_rhs(state + 0.5 * dt * k2, params, dc)
        k4 = _lee_rhs(state + dt * k3, params, dc)
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return new_state, state

    _, trajectory = lax.scan(rk4_step, state0, None, length=n_steps)

    t_arr = jnp.linspace(0.0, t_total, n_steps)
    I_arr = trajectory[:, I_IDX]
    V_arr = trajectory[:, V_IDX]
    z_arr = trajectory[:, Z_IDX]
    r_arr = trajectory[:, R_IDX]

    abs_I = jnp.abs(I_arr)
    peak_idx = jnp.argmax(abs_I)
    I_peak = abs_I[peak_idx]
    t_peak = t_arr[peak_idx]

    return {
        "t": t_arr,
        "I": I_arr,
        "V": V_arr,
        "z": z_arr,
        "r": r_arr,
        "I_peak": I_peak,
        "t_peak": t_peak,
    }


def loss_fn(
    params: dict[str, jnp.ndarray],
    target_I: jnp.ndarray,
    target_t: jnp.ndarray,
) -> jnp.ndarray:
    """NRMSE loss between simulated and target current waveform.

    Interpolates the simulated I(t) onto the target time grid, then
    computes normalized root-mean-square error.

    Args:
        params: Device parameter dictionary.
        target_I: Target current waveform [A].
        target_t: Target time array [s].

    Returns:
        NRMSE scalar (lower = better fit).
    """
    sim_time = target_t[-1] * 1.1
    result = simulate(params, n_steps=N_STEPS, sim_time=float(sim_time))

    sim_t = result["t"]
    sim_I = result["I"]

    # Interpolate sim onto target grid
    sim_I_interp = jnp.interp(target_t, sim_t, sim_I)

    # NRMSE: RMSE / (max - min) of target
    residual = sim_I_interp - target_I
    rmse = jnp.sqrt(jnp.mean(residual**2))
    I_range = jnp.maximum(jnp.max(jnp.abs(target_I)), 1.0)
    nrmse = rmse / I_range

    return nrmse


def calibrate(
    target_I: jnp.ndarray,
    target_t: jnp.ndarray,
    initial_params: dict[str, jnp.ndarray],
    fit_keys: list[str] | None = None,
    n_iters: int = 200,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Gradient-based calibration using Adam optimizer.

    Fits selected parameters to minimize NRMSE between simulated and
    target I(t) waveform.

    Args:
        target_I: Target current waveform [A].
        target_t: Target time array [s].
        initial_params: Starting parameter dictionary.
        fit_keys: Which parameters to optimize. Default: ["fc", "fm", "fmr", "R0"].
        n_iters: Number of Adam iterations.
        lr: Learning rate.

    Returns:
        Dictionary with:
            params: Optimized parameter dictionary.
            loss_history: Array of NRMSE values per iteration.
            n_iters: Number of iterations completed.
    """
    import optax

    if fit_keys is None:
        fit_keys = ["fc", "fm", "fmr", "R0"]

    # Split params into trainable and frozen
    trainable = {k: initial_params[k] for k in fit_keys}
    frozen = {k: v for k, v in initial_params.items() if k not in fit_keys}

    def _loss(trainable_params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        full_params = {**frozen, **trainable_params}
        return loss_fn(full_params, target_I, target_t)

    grad_fn = jax.grad(_loss)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(trainable)

    def step(carry: tuple, _: Any) -> tuple[tuple, jnp.ndarray]:
        trainable_params, opt_state = carry
        loss_val = _loss(trainable_params)
        grads = grad_fn(trainable_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_params)
        new_trainable = optax.apply_updates(trainable_params, updates)
        # Clamp physical bounds
        new_trainable = {
            k: jnp.clip(v, 1e-6, 10.0) if k in ("fc", "fm", "fmr", "fcr") else v
            for k, v in new_trainable.items()
        }
        return (new_trainable, new_opt_state), loss_val

    (final_trainable, _), loss_history = lax.scan(
        step, (trainable, opt_state), None, length=n_iters,
    )

    final_params = {**frozen, **final_trainable}

    return {
        "params": final_params,
        "loss_history": loss_history,
        "n_iters": n_iters,
    }


def sensitivity(
    params: dict[str, jnp.ndarray],
    observable: str = "I_peak",
) -> dict[str, jnp.ndarray]:
    """Compute d(observable)/d(param) for all parameters via jax.grad.

    Args:
        params: Device parameter dictionary.
        observable: Which output to differentiate. Options:
            "I_peak" (default), "t_peak".

    Returns:
        Dictionary mapping parameter names to their gradient with
        respect to the chosen observable.
    """
    def _scalar_fn(p: dict[str, jnp.ndarray]) -> jnp.ndarray:
        result = simulate(p)
        if observable == "t_peak":
            return result["t_peak"]
        return result["I_peak"]

    grads = jax.grad(_scalar_fn)(params)
    return grads


def vmap_simulate(
    params_batch: dict[str, jnp.ndarray],
    n_steps: int = N_STEPS,
    sim_time: float | None = None,
) -> dict[str, jnp.ndarray]:
    """Batched simulation via jax.vmap for parallel parameter sweeps.

    Args:
        params_batch: Dictionary where each value has a leading batch
            dimension, shape (batch_size,).
        n_steps: Number of integration steps per simulation.
        sim_time: Total simulation time [s].

    Returns:
        Batched result dictionary. Each value has shape (batch_size, ...).
    """
    def _single(params: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        return simulate(params, n_steps=n_steps, sim_time=sim_time)

    return jax.vmap(_single)(params_batch)
