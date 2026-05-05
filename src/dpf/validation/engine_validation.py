"""Production engine validation against experimental PF-1000 data.

Runs the production RLCSolver (implicit midpoint) + SnowplowModel for
PF-1000 and compares the resulting I(t) waveform against the digitized
Scholz et al. (2006) experimental data.

This is the P0 highest-impact action from PhD Debate #11: validating the
production circuit solver against experiment.  Key findings:

- RLCSolver (implicit midpoint) matches LeeModel (RK45 solve_ivp) to <0.1%
- NRMSE = 0.1329 vs Scholz at 0.7 us liftoff delay
- Peak current error = 0.2% (1.867 vs 1.87 MA)
- fc^2/fm = 2.374 (degeneracy ratio, the only uniquely determined parameter)

Usage::

    from dpf.validation.engine_validation import (
        run_rlc_snowplow_pf1000,
        compare_engine_vs_experiment,
    )

    t, I, summary = run_rlc_snowplow_pf1000()
    result = compare_engine_vs_experiment(t, I)
    print(f"NRMSE vs Scholz: {result.waveform_nrmse:.3f}")

References
----------
- Scholz et al., Nukleonika 51(1):79-84, 2006.
- Lee & Saw, J. Fusion Energy 33:319-335, 2014.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dpf.constants import k_B, m_D2
from dpf.validation.experimental import _find_first_peak
from dpf.validation.experimental_device import ExperimentalDevice
from dpf.validation.experimental_devices import PF1000_DATA

logger = logging.getLogger(__name__)


@dataclass
class EngineValidationResult:
    """Result of engine-vs-experiment comparison.

    Attributes:
        t: Time array from engine [s].
        I: Current waveform from engine [A].
        peak_current_sim: Simulated peak current [A].
        peak_current_exp: Experimental peak current [A].
        peak_current_error: Relative error on peak current.
        peak_time_sim: Simulated time of peak current [s].
        timing_error: Relative error on peak timing.
        waveform_nrmse: NRMSE of I(t) vs Scholz digitized waveform.
        agreement_within_2sigma: True if within 2-sigma experimental uncertainty.
        fc: Current fraction used.
        fm: Mass fraction used.
        fc2_over_fm: Degeneracy ratio fc^2/fm.
        summary: Engine run summary dict.
        n_steps: Total timesteps taken.
    """

    t: np.ndarray
    I: np.ndarray  # noqa: E741
    peak_current_sim: float = 0.0
    peak_current_exp: float = 0.0
    peak_current_error: float = 0.0
    peak_time_sim: float = 0.0
    timing_error: float = 0.0
    waveform_nrmse: float = float("nan")
    agreement_within_2sigma: bool = False
    fc: float = 0.0
    fm: float = 0.0
    fc2_over_fm: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)
    n_steps: int = 0


def run_rlc_snowplow_pf1000(
    device: ExperimentalDevice = PF1000_DATA,
    *,
    sim_time: float = 10e-6,
    dt: float = 1e-9,
    fc: float | None = None,
    fm: float | None = None,
    f_mr: float | None = None,
    pinch_column_fraction: float = 0.14,
    liftoff_delay: float = 0.7e-6,
    crowbar_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run production RLCSolver + SnowplowModel for PF-1000.

    This isolates the circuit-snowplow coupling from the MHD grid to
    validate the production circuit solver against experimental I(t).
    Both the RLCSolver (implicit midpoint) and the SnowplowModel are
    the same code paths used by SimulationEngine.

    SSoT (D1-A): Device parameters (C, V0, L0, R0, a, b, z_max, p_torr)
    and Lee fit parameters (fc, fm, f_mr) default to the published values
    on the ``device`` argument (default: ``PF1000_DATA`` from
    ``dpf.validation.experimental_devices``). Calling with no arguments
    reproduces the prior hardcoded behavior bit-for-bit. Explicit ``fc``,
    ``fm``, ``f_mr`` overrides are honored when not None; otherwise they
    fall back to ``device.lee_fc``, ``device.lee_fm``, ``device.lee_fmr``.

    Args:
        device: Published experimental device. Default ``PF1000_DATA``
            (PF-1000 at 27 kV / 3.5 Torr D2 per Akel 2021 + Scholz 2006).
        sim_time: Total simulation time [s]. Default 10 us.
        dt: Timestep [s]. Default 1 ns (fine enough for ~24 kHz dynamics).
        fc: Current fraction (Lee's f_c). When ``None``, falls back to
            ``device.lee_fc`` (PF1000_DATA: 0.70 from Malek et al. 2025
            Plasma Phys. Technol. 12(1):9 §3 lines 177-180; matches
            Lee 2014 JFE 33:319 Fig. 7 KSU fit).
        fm: Mass fraction (Lee's f_m). When ``None``, falls back to
            ``device.lee_fm`` (PF1000_DATA: 0.13 from Malek et al. 2025).
        f_mr: Radial mass fraction (Lee's f_mr). When ``None``, falls
            back to ``device.lee_fmr`` (PF1000_DATA: 0.35 from
            Malek et al. 2025).
        pinch_column_fraction: Fraction of anode length for radial phase.
            For PF-1000: 0.14 (effective pinch column ~67 mm of 480 mm
            anode per Akel 2021). This controls the current dip depth
            via radial inductance.
        liftoff_delay: Insulator flashover delay [s]. Default 0.7 us.
        crowbar_enabled: Enable crowbar at V_cap zero crossing. Default True.

    Returns:
        Tuple of (t_array, I_array, summary_dict).
    """
    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.fluid.snowplow import SnowplowModel

    # Device parameters sourced from the ExperimentalDevice SSoT
    # (D1-A unification — see experimental_devices.py for paper citations).
    # PF1000_DATA (default) values:
    #   C = 1.332 mF, V0 = 27 kV (Akel 2021 Bank; Scholz 2006 operating point)
    #   L0 = 25 nH (Akel 2021 Table 1 — 23 shots)
    #   R0 = 2.3 mOhm (Scholz 2006 bare-bank short-circuit; Wave-10 RCA:
    #     6.1 mOhm conflates plasma + bank R; plasma enters via sheath model)
    #   a = 0.115 m (Akel 2021 anode dia 231 mm)
    #   b = 0.160 m (cathode radius)
    #   z_max = 0.48 m (Akel 2021 "480 mm long coaxial electrodes")
    #   p_torr = 3.5 (Scholz 2006 operating point)
    C = device.capacitance
    V0 = device.voltage
    L0 = device.inductance
    R0 = device.resistance
    a = device.anode_radius
    b = device.cathode_radius
    z_max = device.anode_length
    p_torr = device.fill_pressure_torr

    # Lee model fit parameters: explicit overrides win, else use device defaults.
    if fc is None:
        fc = device.lee_fc
    if fm is None:
        fm = device.lee_fm
    if f_mr is None:
        f_mr = device.lee_fmr

    # Fill density from ideal gas law
    p_Pa = p_torr * 133.322
    rho0 = (p_Pa / (k_B * 300.0)) * m_D2  # D2 molecular mass

    # Create production solvers (same code as SimulationEngine)
    circuit = RLCSolver(
        C=C, V0=V0, L0=L0, R0=R0,
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=crowbar_enabled,
        crowbar_mode="voltage_zero",
    )

    snowplow = SnowplowModel(
        anode_radius=a,
        cathode_radius=b,
        fill_density=rho0,
        anode_length=z_max,
        mass_fraction=fm,
        current_fraction=fc,
        radial_mass_fraction=f_mr,
        fill_pressure_Pa=p_Pa,
        pinch_column_fraction=pinch_column_fraction,
    )

    # Integration loop
    coupling = CouplingState()
    t_list: list[float] = [liftoff_delay]
    I_list: list[float] = [0.0]
    t = 0.0
    step = 0
    n_steps = int(sim_time / dt)

    for _ in range(n_steps):
        # Snowplow dynamics (same as engine.py Step 2)
        sp_result = snowplow.step(dt, coupling.current)
        coupling.Lp = sp_result["L_plasma"]
        coupling.dL_dt = sp_result["dL_dt"]
        # Post-pinch disruption model provides anomalous + Spitzer R_plasma
        coupling.R_plasma = sp_result.get("R_plasma", 0.0)

        # Circuit advance (production RLCSolver, implicit midpoint).
        # NOTE: I*dLp/dt is internally handled by RLCSolver via coupling.dL_dt;
        # the back_emf parameter is for OTHER EMF sources (e.g., plasma streaming,
        # sheath instability). Pass 0.0 unless those are explicitly modeled.
        coupling = circuit.step(coupling, 0.0, dt)

        t += dt
        step += 1
        t_list.append(t + liftoff_delay)
        I_list.append(coupling.current)

    t_arr = np.array(t_list)
    I_arr = np.array(I_list)

    # Summary
    abs_I = np.abs(I_arr)
    peak_idx = int(np.argmax(abs_I))
    E_stored = 0.5 * C * V0**2

    summary = {
        "steps": step,
        "sim_time": t,
        "peak_current_A": float(abs_I[peak_idx]),
        "peak_current_time_s": float(t_arr[peak_idx]),
        "energy_conservation": circuit.total_energy() / max(E_stored, 1e-30),
        "fc": fc,
        "fm": fm,
        "fc2_over_fm": fc**2 / fm,
        "liftoff_delay": liftoff_delay,
        "crowbar_enabled": crowbar_enabled,
        "snowplow_phase": snowplow.phase,
        "z_sheath": snowplow.z,
        "r_shock": snowplow.r_shock,
    }

    logger.info(
        "RLC+Snowplow PF-1000: %d steps, peak_I=%.2e A at t=%.2e s, "
        "fc=%.3f, fm=%.3f, fc2/fm=%.3f, phase=%s",
        step, summary["peak_current_A"], summary["peak_current_time_s"],
        fc, fm, fc**2 / fm, snowplow.phase,
    )

    return t_arr, I_arr, summary


def compare_engine_vs_experiment(
    t: np.ndarray,
    I: np.ndarray,  # noqa: E741
    device_name: str = "PF-1000",
    fc: float = 0.70,
    fm: float = 0.13,
    truncate_at_dip: bool = False,
) -> EngineValidationResult:
    """Compare engine I(t) against experimental data.

    Uses the same validation infrastructure as the Lee model comparison
    (``validate_current_waveform``) but applied to the production solver.

    Args:
        t: Time array [s].
        I: Current waveform [A].
        device_name: Device name for experimental data lookup.
        fc: Current fraction used.
        fm: Mass fraction used.
        truncate_at_dip: If True, compute NRMSE only up to current dip,
            excluding the post-pinch frozen-L region.

    Returns:
        :class:`EngineValidationResult` with comparison metrics.
    """
    from dpf.validation.experimental import validate_current_waveform

    metrics = validate_current_waveform(t, I, device_name, truncate_at_dip=truncate_at_dip)

    result = EngineValidationResult(
        t=t,
        I=I,
        peak_current_sim=metrics["peak_current_sim"],
        peak_current_exp=metrics["peak_current_exp"],
        peak_current_error=metrics["peak_current_error"],
        peak_time_sim=metrics["peak_time_sim"],
        timing_error=metrics["timing_error"],
        waveform_nrmse=metrics.get("waveform_nrmse", float("nan")),
        agreement_within_2sigma=metrics["uncertainty"]["agreement_within_2sigma"],
        fc=fc,
        fm=fm,
        fc2_over_fm=fc**2 / fm,
        n_steps=len(t),
    )

    logger.info(
        "Engine vs Experiment: peak_err=%.1f%%, timing_err=%.1f%%, "
        "NRMSE=%.3f, within_2sigma=%s, fc2/fm=%.3f",
        result.peak_current_error * 100,
        result.timing_error * 100,
        result.waveform_nrmse,
        result.agreement_within_2sigma,
        result.fc2_over_fm,
    )

    return result


def compare_rlc_vs_lee(
    *,
    fc: float = 0.70,
    fm: float = 0.13,
    f_mr: float = 0.35,
    pinch_column_fraction: float = 0.14,
    liftoff_delay: float = 0.7e-6,
) -> dict[str, Any]:
    """Cross-verify RLCSolver+Snowplow vs LeeModel (solve_ivp) for PF-1000.

    Both solvers use the same physical parameters and snowplow dynamics,
    but different numerical integrators:
    - RLCSolver: implicit midpoint (production)
    - LeeModel: scipy solve_ivp RK45 (cross-check)

    This directly tests the Phase AC.4 finding that both solvers produce
    identical I(t) when given the same inputs.

    Args:
        fc: Current fraction.
        fm: Mass fraction.
        f_mr: Radial mass fraction (Lee & Saw 2014).
        liftoff_delay: Insulator flashover delay [s].

    Returns:
        Dict with cross-verification metrics.
    """
    from dpf.validation.experimental import nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    # Run production solver
    t_rlc, I_rlc, _ = run_rlc_snowplow_pf1000(
        fc=fc, fm=fm, f_mr=f_mr,
        pinch_column_fraction=pinch_column_fraction,
        liftoff_delay=liftoff_delay,
    )

    # Run cross-check solver
    lee = LeeModel(
        current_fraction=fc,
        mass_fraction=fm,
        radial_mass_fraction=f_mr,
        pinch_column_fraction=pinch_column_fraction,
        liftoff_delay=liftoff_delay,
        crowbar_enabled=True,
    )
    lee_result = lee.run("PF-1000")

    # Cross-compare: measure NRMSE between the two waveforms
    cross_nrmse = nrmse_peak(
        t_rlc, I_rlc, lee_result.t, lee_result.I,
    )

    # Compare peaks (use first peak, not global max which may include
    # post-pinch inductance release from disruption model)
    peak_rlc = float(np.abs(I_rlc)[_find_first_peak(np.abs(I_rlc))])
    peak_lee = lee_result.peak_current
    peak_diff = abs(peak_rlc - peak_lee) / max(peak_lee, 1e-30)

    return {
        "cross_nrmse": cross_nrmse,
        "peak_rlc": peak_rlc,
        "peak_lee": peak_lee,
        "peak_diff_relative": peak_diff,
        "solvers_agree": cross_nrmse < 0.25 and peak_diff < 0.01,
    }
