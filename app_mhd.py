"""MHD backend connector for DPF web UI.

Wraps MetalMHDSolver and Python MHD engine to produce the same output
format as the Lee model runner, enabling backend switching in the UI.
"""
from __future__ import annotations

import logging
import time as wall_time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from app_engine import GAS_SPECIES, kB

logger = logging.getLogger(__name__)

BACKENDS = {
    "lee": "Level 1 — Circuit Model (< 1 sec)",
    "first_principles_mhd": "First-principles MHD - PF-1000/Akel fail-closed readiness",
    "hybrid": "Level 2 — Circuit + MHD (3-30 sec) [RECOMMENDED]",
    "python": "Level 3 — Full MHD (10-30 sec)",
    "metal_plm": "Level 4 — Full MHD, GPU (5-15 sec)",
    "metal_weno5": "Level 5 — Research Grade (30-120 sec)",
    "engine_python": "Level 3E — Engine + Python MHD (full CircuitCoupler)",
    "engine_metal": "Level 4E — Engine + Metal MHD (full CircuitCoupler, GPU)",
    "engine_athena": "Level 9 — Engine + Athena++ C++ pybind11 (9.0/10, ~0.5 us/sec)",
    "engine_athenak": "Level 9 — Engine + AthenaK Kokkos (9.0/10, ~0.2 us/sec, GPU-portable)",
}

BACKEND_CONFIGS = {
    "metal_plm": {
        "reconstruction": "plm", "riemann_solver": "hll",
        "time_integrator": "ssp_rk2", "precision": "float32",
    },
    "metal_weno5": {
        "reconstruction": "weno5", "riemann_solver": "hlld",
        "time_integrator": "ssp_rk3", "precision": "float64",
        "enable_hall": True,
    },
}


def _mhd_numerical_method_metadata(
    backend: str,
    grid_shape: tuple[int, int, int],
    dr: float,
    dz: float,
) -> dict[str, object]:
    """Return run-level MHD method metadata for validation audits."""
    backend_key = str(backend).lower().split()[0]
    cfg = BACKEND_CONFIGS.get(backend_key, {})
    if backend_key in {"python", "hybrid"}:
        cfg = {
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
            "precision": "float64",
        }
    if backend_key.startswith("metal_cylindrical"):
        cfg = BACKEND_CONFIGS.get("metal_plm", {})
    coordinates = "cartesian" if "3d" in backend_key else "cylindrical"
    return {
        "backend": backend,
        "finite_volume": True,
        "coordinates": coordinates,
        "grid_shape": tuple(int(value) for value in grid_shape),
        "dr_m": float(dr),
        "dz_m": float(dz),
        "reconstruction": cfg.get("reconstruction", "unknown"),
        "riemann_solver": cfg.get("riemann_solver", "unknown"),
        "time_integrator": cfg.get("time_integrator", "unknown"),
        "precision": cfg.get("precision", "unknown"),
        "source": "app_mhd backend configuration",
        "validity_notes": {
            "metadata_scope": (
                "Run-level method metadata supports numerical audits but is not "
                "a substitute for analytic verification or convergence evidence."
            ),
        },
    }


def _first_principles_eta_field(
    state: dict[str, np.ndarray],
    gas: dict,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a source-traced Spitzer/Braginskii resistivity field.

    The PF-1000 local source path supports starting after breakdown from a
    partially ionized plasma and using Braginskii transport. This helper keeps
    that transport explicit and leaves missing ionization/anomalous-resistivity
    closure as metadata blockers instead of hiding it behind eta caps.
    """
    from dpf.constants import e as electron_charge
    from dpf.constants import m_e
    from dpf.validation.first_principles_limiters import limiter_event

    limiter_events: list[dict[str, object]] = []
    rho = np.asarray(state["rho"], dtype=float)
    valid_rho = np.isfinite(rho) & (rho > 0.0)
    if not bool(np.all(valid_rho)):
        rho_safe = np.where(valid_rho, rho, np.nan)
        limiter_events.append(
            limiter_event(
                limiter_id="app_mhd.resistivity.invalid_density_domain",
                code_path="app_mhd._first_principles_eta_field",
                affected_field="rho",
                classification="debug_repair",
                activation_count=int(rho.size - np.count_nonzero(valid_rho)),
                before=rho,
                acceptance_blocking=True,
                justification=(
                    "Resistivity requires positive finite density; invalid "
                    "density was converted to NaN so the run remains blocked."
                ),
            )
        )
    else:
        rho_safe = rho

    Te_raw = state.get("Te")
    if Te_raw is None:
        pressure = np.asarray(
            state.get("pressure", np.zeros_like(rho_safe)),
            dtype=float,
        )
        Te_before = pressure * float(gas["m_mol"]) / (2.0 * rho_safe * kB)
    else:
        Te_before = np.asarray(Te_raw, dtype=float)
    valid_Te = np.isfinite(Te_before) & (Te_before > 0.0)
    if not bool(np.all(valid_Te)):
        Te = np.where(valid_Te, Te_before, np.nan)
        limiter_events.append(
            limiter_event(
                limiter_id="app_mhd.resistivity.invalid_temperature_domain",
                code_path="app_mhd._first_principles_eta_field",
                affected_field="Te",
                classification="debug_repair",
                activation_count=int(Te_before.size - np.count_nonzero(valid_Te)),
                before=Te_before,
                acceptance_blocking=True,
                justification=(
                    "Resistivity requires positive finite electron temperature; "
                    "invalid temperature was converted to NaN so the run remains blocked."
                ),
            )
        )
    else:
        Te = Te_before

    n_total = rho_safe / float(gas["m_mol"])
    Z_eff = float(gas.get("Z", 1.0))
    Z_bar_raw = state.get("Z_bar")
    if Z_bar_raw is None:
        Z_bar = np.ones_like(rho_safe)
    else:
        Z_bar = np.asarray(Z_bar_raw, dtype=float)
    valid_Z = np.isfinite(Z_bar) & (Z_bar > 0.0) & (Z_bar <= max(Z_eff, 1.0))
    if not bool(np.all(valid_Z)):
        Z_bar = np.where(valid_Z, Z_bar, np.nan)
        limiter_events.append(
            limiter_event(
                limiter_id="app_mhd.resistivity.invalid_ionization_domain",
                code_path="app_mhd._first_principles_eta_field",
                affected_field="Z_bar",
                classification="debug_repair",
                activation_count=int(np.size(Z_bar) - np.count_nonzero(valid_Z)),
                before=Z_bar_raw,
                after=Z_bar,
                threshold={"valid_range": "(0, Z_eff]"},
                acceptance_blocking=True,
                justification=(
                    "Resistivity requires a positive finite ionization fraction; "
                    "invalid ionization state was converted to NaN."
                ),
            )
        )

    ne = n_total * Z_bar
    valid_ne = np.isfinite(ne) & (ne > 0.0)
    if not bool(np.all(valid_ne)):
        limiter_events.append(
            limiter_event(
                limiter_id="app_mhd.resistivity.invalid_electron_density_domain",
                code_path="app_mhd._first_principles_eta_field",
                affected_field="ne",
                classification="debug_repair",
                activation_count=int(ne.size - np.count_nonzero(valid_ne)),
                before=ne,
                acceptance_blocking=True,
                justification=(
                    "Collisional resistivity requires positive finite electron density."
                ),
            )
        )

    Z_eff = float(gas.get("Z", 1.0))
    model = "partial_ionization_spitzer_braginskii_uncapped"
    try:
        from dpf.collision.spitzer import coulomb_log, nu_ei

        lnL = coulomb_log(ne, Te)
        nu_ei_field = nu_ei(ne, Te, lnL, Z=Z_eff)
        eta = m_e * nu_ei_field / (ne * electron_charge**2)
        lnL_range = [
            float(np.nanmin(lnL)),
            float(np.nanmax(lnL)),
        ]
    except Exception as exc:
        Te_eV = Te * kB / 1.602176634e-19
        eta = 5.2e-5 * max(Z_eff, 1.0) * 10.0 / np.power(Te_eV, 1.5)
        lnL_range = [10.0, 10.0]
        model = f"nrl_formula_fallback_blocking:{type(exc).__name__}"
        limiter_events.append(
            limiter_event(
                limiter_id="app_mhd.resistivity.nrl_formula_fallback",
                code_path="app_mhd._first_principles_eta_field",
                affected_field="eta",
                classification="debug_repair",
                activation_count=1,
                acceptance_blocking=True,
                justification=(
                    "Fallback resistivity formula used because Spitzer path "
                    f"raised {type(exc).__name__}."
                ),
            )
        )
    eta = np.asarray(eta, dtype=float)
    valid_eta = np.isfinite(eta) & (eta > 0.0)
    if not bool(np.all(valid_eta)):
        limiter_events.append(
            limiter_event(
                limiter_id="app_mhd.resistivity.invalid_eta_domain",
                code_path="app_mhd._first_principles_eta_field",
                affected_field="eta",
                classification="debug_repair",
                activation_count=int(eta.size - np.count_nonzero(valid_eta)),
                before=eta,
                acceptance_blocking=True,
                justification=(
                    "Resistivity model produced non-positive or non-finite eta; "
                    "the run cannot support first-principles acceptance."
                ),
            )
        )
    return eta, {
        "model": model,
        "validation_status": "source_traced_candidate_not_validation",
        "eta_floor_ohm_m": None,
        "eta_cap_ohm_m": None,
        "lnL_range": lnL_range,
        "electron_density_min_m3": float(np.nanmin(ne)),
        "electron_density_max_m3": float(np.nanmax(ne)),
        "ionization_fraction_min": float(np.nanmin(Z_bar)),
        "ionization_fraction_max": float(np.nanmax(Z_bar)),
        "source_basis": {
            "pf1000_mhd_model": (
                "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json:57-62"
            ),
            "pf1000_breakdown_transport_note": (
                "KnowledgeReference/scholz-2006-pf1000-mega-joule.md:149-208"
            ),
            "plasma_parameter_domain": (
                "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2594-2706"
            ),
        },
        "limiter_events": limiter_events,
        "limitations": [
            "ionization kinetics are not evolved yet",
            "electron-neutral transport coefficients are not source-closed yet",
            "anomalous resistivity trigger/strength is not implemented yet",
        ],
    }


def _apply_first_principles_engineering_bounds(
    state: dict[str, np.ndarray],
    gas: dict,
    rho0: float,
    *,
    dr: float | None = None,
    dz: float | None = None,
    r_cell_m: np.ndarray | None = None,
    magnetic_energy_cap_J: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Apply explicit finite-state bounds for the first-principles candidate.

    These limits prevent late explicit finite-volume overflow while preserving
    the resolved-field coupling path. They are engineering stability guards and
    must not be interpreted as accepted physics closure.
    """
    from dpf.validation.first_principles_limiters import limiter_event

    rho_floor = max(1.0e-12, 1.0e-6 * rho0)
    rho_cap = max(1.0e3 * rho0, rho_floor * 10.0)
    pressure_floor = 1.0e-6
    T_cap_K = 1.16e9
    pressure_cap = max(
        rho_cap * kB * T_cap_K / max(float(gas["m_mol"]), 1.0e-30),
        1.0e6,
    )
    B_cap_T = 50.0
    v_cap_m_s = 2.0e6
    mu_0_local = 4.0 * np.pi * 1.0e-7

    limiter_counts: dict[str, int] = {}
    limiter_events: list[dict[str, object]] = []

    def _classification_for_key(key: str) -> str:
        if "nonfinite" in key:
            return "debug_repair"
        if key == "psi":
            return "engineering_guard"
        return "acceptance_blocker"

    def _repair_array(
        key: str,
        arr: np.ndarray,
        *,
        floor: float | None = None,
        cap: float | None = None,
        abs_cap: float | None = None,
    ) -> np.ndarray:
        values = np.asarray(arr, dtype=float)
        repaired = values.copy()
        before = repaired.copy()
        finite = np.isfinite(repaired)
        if not np.all(finite):
            replacement = floor if floor is not None else 0.0
            repaired = np.where(finite, repaired, replacement)
        if floor is not None:
            repaired = np.maximum(repaired, floor)
        if cap is not None:
            repaired = np.minimum(repaired, cap)
        if abs_cap is not None:
            repaired = np.clip(repaired, -abs_cap, abs_cap)
        changed = int(np.count_nonzero(repaired != before))
        if changed:
            limiter_counts[key] = limiter_counts.get(key, 0) + changed
            limiter_events.append(
                limiter_event(
                    limiter_id=f"app_mhd.state_bounds.{key}",
                    code_path="app_mhd._apply_first_principles_engineering_bounds",
                    affected_field=key,
                    classification=_classification_for_key(key),
                    activation_count=changed,
                    before=before,
                    after=repaired,
                    threshold={
                        "floor": floor,
                        "cap": cap,
                        "abs_cap": abs_cap,
                    },
                    acceptance_blocking=True,
                    justification=(
                        "App-level first-principles finite-state bound changed "
                        f"{changed} value(s)."
                    ),
                )
            )
        return repaired

    if "rho" in state:
        state["rho"] = _repair_array(
            "rho",
            state["rho"],
            floor=rho_floor,
            cap=rho_cap,
        )
    if "pressure" in state:
        state["pressure"] = _repair_array(
            "pressure",
            state["pressure"],
            floor=pressure_floor,
            cap=pressure_cap,
        )
    if "Te" in state:
        state["Te"] = _repair_array("Te", state["Te"], floor=1.0, cap=T_cap_K)
    if "Ti" in state:
        state["Ti"] = _repair_array("Ti", state["Ti"], floor=1.0, cap=T_cap_K)
    if "psi" in state:
        state["psi"] = _repair_array("psi", state["psi"], abs_cap=1.0e6)
    if "velocity" in state:
        velocity = _repair_array("velocity_nonfinite", state["velocity"], abs_cap=1.0e12)
        v_mag = np.sqrt(np.sum(velocity**2, axis=0))
        mask = v_mag > v_cap_m_s
        if np.any(mask):
            velocity_before = velocity.copy()
            scale = np.ones_like(v_mag)
            scale[mask] = v_cap_m_s / np.maximum(v_mag[mask], 1.0e-30)
            velocity = velocity * scale[np.newaxis, :, :, :]
            limiter_counts["velocity_magnitude"] = int(np.count_nonzero(mask))
            limiter_events.append(
                limiter_event(
                    limiter_id="app_mhd.state_bounds.velocity_magnitude",
                    code_path="app_mhd._apply_first_principles_engineering_bounds",
                    affected_field="velocity",
                    classification="acceptance_blocker",
                    activation_count=int(np.count_nonzero(mask)),
                    before=velocity_before,
                    after=velocity,
                    threshold={"cap_m_s": v_cap_m_s},
                    acceptance_blocking=True,
                    justification="Velocity magnitude cap changed resolved velocity cells.",
                )
            )
        state["velocity"] = velocity
    if "B" in state:
        B = _repair_array("B_nonfinite", state["B"], abs_cap=1.0e12)
        B_mag = np.sqrt(np.sum(B**2, axis=0))
        mask = B_mag > B_cap_T
        if np.any(mask):
            B_before = B.copy()
            scale = np.ones_like(B_mag)
            scale[mask] = B_cap_T / np.maximum(B_mag[mask], 1.0e-30)
            B = B * scale[np.newaxis, :, :, :]
            limiter_counts["B_magnitude"] = int(np.count_nonzero(mask))
            limiter_events.append(
                limiter_event(
                    limiter_id="app_mhd.state_bounds.B_magnitude",
                    code_path="app_mhd._apply_first_principles_engineering_bounds",
                    affected_field="B",
                    classification="acceptance_blocker",
                    activation_count=int(np.count_nonzero(mask)),
                    before=B_before,
                    after=B,
                    threshold={"cap_T": B_cap_T},
                    acceptance_blocking=True,
                    justification="Magnetic-field magnitude cap changed resolved B cells.",
                )
            )
        if (
            magnetic_energy_cap_J is not None
            and magnetic_energy_cap_J > 0.0
            and dr is not None
            and dz is not None
            and r_cell_m is not None
        ):
            r = np.asarray(r_cell_m, dtype=float)
            volume = 2.0 * np.pi * np.maximum(r, 1.0e-12)[:, None] * dr * dz
            B2 = np.sum(B[:, :, 0, :] ** 2, axis=0)
            magnetic_energy_J = float(np.sum(0.5 * B2 / mu_0_local * volume))
            if np.isfinite(magnetic_energy_J) and magnetic_energy_J > magnetic_energy_cap_J:
                scale = float(np.sqrt(magnetic_energy_cap_J / magnetic_energy_J))
                B_before = B.copy()
                B = B * scale
                limiter_counts["B_energy"] = int(B2.size)
                limiter_events.append(
                    limiter_event(
                        limiter_id="app_mhd.state_bounds.B_energy",
                        code_path="app_mhd._apply_first_principles_engineering_bounds",
                        affected_field="B",
                        classification="acceptance_blocker",
                        activation_count=int(B2.size),
                        before=B_before,
                        after=B,
                        threshold={"magnetic_energy_cap_J": magnetic_energy_cap_J},
                        acceptance_blocking=True,
                        justification="Magnetic-energy cap scaled the resolved B field.",
                    )
                )
        state["B"] = B

    return state, {
        "validation_status": "engineering_probe_not_validation",
        "rho_floor": rho_floor,
        "rho_cap": rho_cap,
        "pressure_floor": pressure_floor,
        "pressure_cap": pressure_cap,
        "B_cap_T": B_cap_T,
        "magnetic_energy_cap_J": magnetic_energy_cap_J,
        "velocity_cap_m_s": v_cap_m_s,
        "counts": limiter_counts,
        "limiter_events": limiter_events,
    }


def _neutron_mechanism_output_summary(
    result: dict[str, Any],
) -> dict[str, object] | None:
    """Return mechanism-separated neutron output metadata without promoting it."""
    yield_block = result.get("neutron_yield")
    if not isinstance(yield_block, dict):
        yield_block = result.get("neutron_yield_details")
    history = result.get("yield_time_resolved")
    if not isinstance(yield_block, dict) and not isinstance(history, dict):
        return None

    def _number(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    y_thermo = (
        _number(yield_block.get("Y_thermonuclear"))
        if isinstance(yield_block, dict) else None
    )
    y_beam = (
        _number(yield_block.get("Y_beam_target"))
        if isinstance(yield_block, dict) else None
    )
    y_total = (
        _number(yield_block.get("Y_neutron"))
        if isinstance(yield_block, dict) else None
    )
    if y_total is None and y_thermo is not None and y_beam is not None:
        y_total = y_thermo + y_beam

    def _fraction(value: float | None) -> float | None:
        if value is None or y_total is None or y_total <= 0.0:
            return None
        return float(value / y_total)

    time_key = None
    thermo_history_key = None
    beam_history_key = None
    if isinstance(history, dict):
        if history.get("t_s") is not None:
            time_key = "yield_time_resolved.t_s"
        elif history.get("times_us") is not None:
            time_key = "yield_time_resolved.times_us"
        if history.get("dY_th") is not None:
            thermo_history_key = "yield_time_resolved.dY_th"
        elif history.get("dY_thermo") is not None:
            thermo_history_key = "yield_time_resolved.dY_thermo"
        if history.get("dY_bt") is not None:
            beam_history_key = "yield_time_resolved.dY_bt"
    timing_available = bool(time_key and thermo_history_key and beam_history_key)
    validation_blockers = [
        "same_scope_scalar_yield_validation",
        "mechanism_timing_validation",
        "neutron_spectrum_validation",
        "neutron_anisotropy_validation",
        "detector_activation_response_validation",
        "same_scope_neutron_uncertainty",
        "kinetic_or_hybrid_beam_target_model",
    ]
    thermo_authority = None
    beam_authority = None
    if isinstance(yield_block, dict):
        thermo_authority = yield_block.get("thermonuclear_input_authority")
        beam_authority = yield_block.get("beam_target_input_authority")
    return {
        "passed": False,
        "model_role": "mechanism_separated_neutron_output_summary",
        "validation_tier": 5,
        "validation_status": "estimate_not_validation",
        "first_principles_total_yield_authority": "blocked",
        "mechanisms": {
            "thermonuclear": {
                "yield_n": y_thermo,
                "fraction": _fraction(y_thermo),
                "status": "estimate" if y_thermo is not None else "not_produced",
                "authority": (
                    str(thermo_authority)
                    if thermo_authority else (
                        "field_derived_candidate"
                        if y_thermo is not None else "not_produced"
                    )
                ),
                "history_key": thermo_history_key if timing_available else None,
            },
            "beam_target": {
                "yield_n": y_beam,
                "fraction": _fraction(y_beam),
                "status": "estimate" if y_beam is not None else "not_produced",
                "authority": (
                    str(beam_authority)
                    if beam_authority else (
                        "baseline_reduced_model"
                        if y_beam is not None else "not_produced"
                    )
                ),
                "history_key": beam_history_key if timing_available else None,
            },
        },
        "total_yield_n": y_total,
        "timing_history": {
            "status": "candidate_available" if timing_available else "not_produced",
            "time_key": time_key if timing_available else None,
        },
        "spectrum": {
            "status": (
                "candidate_available"
                if isinstance(result.get("neutron_spectrum_samples_MeV"), dict)
                else "not_produced"
            ),
        },
        "anisotropy": {
            "status": (
                "candidate_available"
                if isinstance(result.get("neutron_anisotropy"), dict)
                else "not_produced"
            ),
        },
        "detector_activation_response": {
            "status": (
                "candidate_available"
                if isinstance(
                    result.get("neutron_detector_response")
                    or result.get("detector_response"),
                    dict,
                )
                else "not_produced"
            ),
            "required_for_tier5": True,
        },
        "validation_blockers": validation_blockers,
        "validity_notes": {
            "claim_scope": (
                "Mechanism-separated neutron outputs are estimates until "
                "same-scope KR-backed scalar yield, mechanism timing, spectrum, "
                "anisotropy, detector/activation response, and uncertainty "
                "evidence pass together."
            ),
            "first_principles_boundary": (
                "Total neutron-yield authority also requires a field-history "
                "thermonuclear integral plus kinetic/hybrid beam-target "
                "production; Lee/Saw or empirical beam-target terms stay "
                "baseline-only."
            ),
        },
    }


MHD_GRID_PRESETS = {
    "coarse": (16, 16, 32),
    "medium": (32, 32, 64),
    "fine": (64, 64, 128),
}


def _apply_advanced_physics(
    state: dict, dt: float, gas: dict, dr: float, dz: float,
    a: float, b: float,
    enable_fld: bool = False,
    enable_sheath: bool = False,
    enable_ablation: bool = False,
    enable_nernst: bool = False,
    enable_cr: bool = False,
    cr_fractions: np.ndarray | None = None,
) -> tuple[dict, np.ndarray | None]:
    """Operator-split advanced physics modules onto MHD state.

    Returns (updated_state, updated_cr_fractions).
    Each module is guarded by its own try/except so failures are non-fatal.
    """
    mu_0 = 4.0 * np.pi * 1e-7
    ion_mass = gas["m_mol"]
    Z_eff = float(gas.get("Z", 1))

    # 1. FLD radiation transport
    if enable_fld and "Te" in state:
        try:
            from dpf.radiation.transport import apply_radiation_transport
            state = apply_radiation_transport(state, dr, dt, Z=Z_eff)
        except (ImportError, Exception) as exc:
            logger.debug("FLD transport skipped: %s", exc)

    # 2. Sheath boundary conditions (electrode surfaces)
    if enable_sheath and "Te" in state:
        try:
            from dpf.sheath.bohm import apply_sheath_bc
            rho_bnd = float(state["rho"][0, state["rho"].shape[1] // 2, -1])
            ne_bnd = rho_bnd / ion_mass
            Te_bnd = float(state["Te"][0, state["Te"].shape[1] // 2, -1])
            V_sh = float(state.get("pressure", np.zeros(1)).flat[-1]) / max(ne_bnd * kB, 1e-30)
            V_sh = min(V_sh, 1000.0)  # cap sheath voltage at 1 kV
            state = apply_sheath_bc(
                state, ne_boundary=ne_bnd, Te_boundary=Te_bnd,
                V_sheath=V_sh, mi=ion_mass, Z=Z_eff, boundary="z_high",
            )
        except (ImportError, Exception) as exc:
            logger.debug("Sheath BC skipped: %s", exc)

    # 3. Electrode ablation (Cu anode mass injection)
    if enable_ablation and "Te" in state and "B" in state:
        try:
            from dpf.atomic.ablation import COPPER_ABLATION_EFFICIENCY, ablation_source_array
            B_field = state["B"]
            # J ~ curl(B)/mu_0 — approximate as |dBz/dr| / mu_0
            J_mag = np.abs(np.gradient(B_field[2], dr, axis=0)) / mu_0
            # Spitzer resistivity at boundary (crude estimate)
            Te_eV = state["Te"] * kB / 1.602e-19
            eta_spitzer = 5.2e-5 * Z_eff / np.maximum(Te_eV, 0.1) ** 1.5
            # Boundary mask: first radial cell = anode surface
            boundary_mask = np.zeros_like(J_mag, dtype=np.int32)
            boundary_mask[0, :, :] = 1
            S_rho = ablation_source_array(
                J_mag.ravel(), eta_spitzer.ravel(),
                COPPER_ABLATION_EFFICIENCY, boundary_mask.ravel(),
            ).reshape(state["rho"].shape)
            state["rho"] = state["rho"] + S_rho * dt
        except (ImportError, Exception) as exc:
            logger.debug("Ablation skipped: %s", exc)

    # 4. Nernst B-field advection
    if enable_nernst and "Te" in state and "B" in state:
        try:
            from dpf.fluid.nernst import apply_nernst_advection
            ne = state["rho"] / ion_mass
            Bx, By, Bz = state["B"][0], state["B"][1], state["B"][2]
            dy = dr  # approximate
            Bx_new, By_new, Bz_new = apply_nernst_advection(
                Bx, By, Bz, ne, state["Te"], dr, dy, dz, dt, Z_eff=Z_eff,
            )
            state["B"] = np.stack([Bx_new, By_new, Bz_new], axis=0)
        except (ImportError, Exception) as exc:
            logger.debug("Nernst advection skipped: %s", exc)

    # 5. Collisional-radiative ionization (non-LTE Z_bar evolution)
    if enable_cr and "Te" in state:
        try:
            from dpf.atomic.ionization import _IP_H, cr_evolve_field
            ne = state["rho"] / ion_mass
            # Use H ionization for deuterium, Cu for impurity tracking
            ip_eV = _IP_H
            Z_max = len(ip_eV)
            shape = state["rho"].shape
            if cr_fractions is None:
                # Initialize: fully neutral
                cr_fractions = np.zeros((*shape, Z_max + 1))
                cr_fractions[..., 0] = 1.0
            cr_fractions = cr_evolve_field(
                ne.ravel(), state["Te"].ravel(), Z_max, dt,
                cr_fractions.reshape(-1, Z_max + 1), ip_eV,
            ).reshape(*shape, Z_max + 1)
            # Compute Z_bar from fractions
            z_indices = np.arange(Z_max + 1)
            Z_bar = np.sum(cr_fractions * z_indices, axis=-1)
            state["Z_bar"] = Z_bar
        except (ImportError, Exception) as exc:
            logger.debug("CR ionization skipped: %s", exc)

    return state, cr_fractions


def run_mhd_simulation(
    backend: str,
    grid_preset: str,
    preset_name: str,
    sim_time_us: float,
    gas_key: str = "D2",
    V0_kV: float | None = None,
    pressure_torr: float | None = None,
    C_uF: float | None = None,
    L0_nH: float | None = None,
    R0_mOhm: float | None = None,
    anode_r_mm: float | None = None,
    cathode_r_mm: float | None = None,
    anode_len_mm: float | None = None,
    progress_fn=None,
    enable_fld: bool = False,
    enable_sheath: bool = False,
    enable_ablation: bool = False,
    enable_nernst: bool = False,
    enable_cr: bool = False,
) -> dict[str, Any]:
    """Run MHD simulation and return data in the same format as Lee model."""
    from dpf.presets import _PRESETS, get_preset

    requested_backend = str(backend)
    requested_run_mode = (
        "first_principles_mhd"
        if requested_backend == "first_principles_mhd"
        else str(backend)
    )
    if requested_backend == "first_principles_mhd":
        backend = "python"

    preset = get_preset(preset_name)
    cc = preset["circuit"]
    sc = preset.get("snowplow", {})
    gas = GAS_SPECIES.get(gas_key, GAS_SPECIES["D2"])

    if V0_kV is not None and V0_kV > 0:
        cc["V0"] = V0_kV * 1e3
    if C_uF is not None and C_uF > 0:
        cc["C"] = C_uF * 1e-6
    if L0_nH is not None and L0_nH > 0:
        cc["L0"] = L0_nH * 1e-9
    if R0_mOhm is not None and R0_mOhm > 0:
        cc["R0"] = R0_mOhm * 1e-3
    if anode_r_mm is not None and anode_r_mm > 0:
        cc["anode_radius"] = anode_r_mm * 1e-3
    if cathode_r_mm is not None and cathode_r_mm > 0:
        cc["cathode_radius"] = cathode_r_mm * 1e-3

    a = cc["anode_radius"]
    b = cc["cathode_radius"]
    L_anode = sc.get("anode_length", 0.16)
    if anode_len_mm is not None and anode_len_mm > 0:
        L_anode = anode_len_mm * 1e-3

    p_pa = sc.get("fill_pressure_Pa", 400.0)
    if pressure_torr is not None and pressure_torr > 0:
        p_pa = pressure_torr * 133.322
    rho0 = p_pa * gas["m_mol"] / (kB * 300.0)

    grid_shape = MHD_GRID_PRESETS.get(grid_preset, (32, 32, 64))
    nr, ny, nz = grid_shape
    dr = (b - a) / nr
    dz = L_anode / nz

    t_end = sim_time_us * 1e-6
    meta = _PRESETS.get(preset_name, {}).get("_meta", {})
    E_bank = 0.5 * cc["C"] * cc["V0"] ** 2

    t0_wall = wall_time.perf_counter()

    adv_physics = {
        "enable_fld": enable_fld, "enable_sheath": enable_sheath,
        "enable_ablation": enable_ablation, "enable_nernst": enable_nernst,
        "enable_cr": enable_cr,
    }

    if backend == "hybrid":
        result = _run_hybrid_lee_mhd(
            grid_shape, dr, dz, gas, rho0, p_pa,
            cc, sc, t_end, a, b, L_anode, progress_fn,
            **adv_physics,
        )
    elif backend.startswith("metal"):
        result = _run_metal(
            backend, grid_shape, dr, dz, gas, rho0, p_pa,
            cc, sc, t_end, a, b, L_anode, progress_fn,
            **adv_physics,
        )
    elif backend == "athena":
        from pathlib import Path
        _athena_bin = Path(__file__).resolve().parent / "external" / "athena" / "bin" / "athena_cylindrical"
        if not _athena_bin.exists():
            try:
                import gradio as gr
                gr.Warning(
                    "Athena++ binary not found — falling back to Metal PLM engine. "
                    "Build Athena++ with `cd external/athena && make -j8` for native C++ fidelity."
                )
            except ImportError:
                pass
            logger.warning(
                "Athena++ binary not found at %s — falling back to Metal PLM", _athena_bin
            )
            backend = "metal_plm"
            result = _run_metal(
                backend, grid_shape, dr, dz, gas, rho0, p_pa,
                cc, sc, t_end, a, b, L_anode, progress_fn,
                **adv_physics,
            )
            result["backend"] = "metal_plm (fallback from athena)"
        else:
            result = _run_athena(
                grid_shape, dr, dz, gas, rho0, p_pa,
                cc, sc, t_end, a, b, L_anode, progress_fn,
            )
    else:
        # Python MHD now uses Godunov (PLM+HLL) flux with conservative energy
        # and inter-stage velocity clamping. Stable at all resolutions (e787a13).
        result = _run_python_mhd(
            grid_shape, dr, dz, gas, rho0, p_pa,
            cc, t_end, a, b, L_anode, progress_fn,
            field_coupled_candidate=(
                requested_run_mode == "first_principles_mhd"
            ),
            )

    elapsed = wall_time.perf_counter() - t0_wall

    # Preserve custom backend label from redirect/fallback logic
    effective_backend = result.get("backend", backend)
    # Track which advanced physics modules are active
    active_modules = []
    if enable_fld:
        active_modules.append("FLD radiation transport")
    if enable_sheath:
        active_modules.append("Sheath BC (Bohm)")
    if enable_ablation:
        active_modules.append("Electrode ablation (Cu)")
    if enable_nernst:
        active_modules.append("Nernst B-advection")
    if enable_cr:
        active_modules.append("CR ionization (non-LTE)")

    result.update({
        "E_bank_kJ": E_bank / 1e3,
        "T_LC_us": 2 * np.pi * np.sqrt(cc["L0"] * cc["C"]) * 1e6,
        "elapsed_s": elapsed,
        "device": meta.get("device", preset_name),
        "circuit": cc, "snowplow_cfg": sc,
        "gas": gas, "gas_key": gas_key,
        "rho0": rho0,
        "requested_backend": requested_backend,
        "requested_run_mode": requested_run_mode,
        "backend": effective_backend,
        "grid_shape": grid_shape,
        "mhd_numerical_method": _mhd_numerical_method_metadata(
            effective_backend,
            grid_shape,
            dr,
            dz,
        ),
        "advanced_physics": active_modules,
    })

    _apply_post_processing(result, cc, gas, gas_key, p_pa, a, b, L_anode, dr, dz,
                           active_modules, preset_name, effective_backend,
                           grid_shape, sim_time_us,
                           requested_run_mode=requested_run_mode)

    return result


def run_pf1000_akel_first_principles(
    *,
    grid_preset: str = "coarse",
    sim_time_us: float = 0.2,
    gas_key: str = "D2",
    progress_fn=None,
) -> dict[str, Any]:
    """Run the locked PF-1000/Akel first-principles engineering candidate."""
    return run_mhd_simulation(
        backend="first_principles_mhd",
        grid_preset=grid_preset,
        preset_name="pf1000_akel",
        sim_time_us=sim_time_us,
        gas_key=gas_key,
        progress_fn=progress_fn,
    )


def _phase_stagnation_time_s(result: dict) -> float | None:
    """Return the first phase-labeled pinch time, if available."""
    phases = result.get("phases")
    times_us = result.get("t_us")
    if phases is None or times_us is None:
        return None

    n = min(len(phases), len(times_us))
    for idx in range(n):
        phase = str(phases[idx]).strip().lower()
        if phase in {"pinch", "reflected", "post_pinch"}:
            return float(times_us[idx]) * 1.0e-6
    return None


def _record_validation_error(result: dict, stage: str, exc: Exception) -> None:
    """Record validation post-processing failures without aborting the run."""
    result.setdefault("validation_errors", []).append({
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc),
    })


def _apply_post_processing(
    result: dict, cc: dict, gas: dict, gas_key: str,
    p_pa: float, a: float, b: float, L_anode: float,
    dr: float, dz: float, active_modules: list,
    preset_name: str, effective_backend: str,
    grid_shape: tuple, sim_time_us: float,
    requested_run_mode: str | None = None,
) -> None:
    """Apply all post-simulation diagnostics to the result dict."""
    import subprocess as _sp
    from datetime import datetime as _dt
    try:
        _git_hash = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_sp.DEVNULL, text=True,
        ).strip()
    except Exception:
        _git_hash = "unknown"

    final_state = result.get("final_state")

    # V_max diagnostic: peak inductive voltage L_total * max(|dI/dt|)
    I_arr_pp = result.get("I_MA", np.array([]))
    t_arr_pp = result.get("t_us", np.array([]))
    if len(I_arr_pp) > 1 and len(t_arr_pp) > 1:
        dI_dt_arr = np.gradient(I_arr_pp * 1e6, t_arr_pp * 1e-6)
        L_p_nH_arr = result.get("L_p_nH", np.array([]))
        L_ext = cc.get("L0", 1e-9)
        L_p_max = float(np.max(L_p_nH_arr) * 1e-9) if len(L_p_nH_arr) > 0 else 0.0
        L_total_est = L_ext + L_p_max
        result["V_max_kV"] = float(L_total_est * np.max(np.abs(dI_dt_arr))) / 1e3

    # Phase 1 breakdown diagnostic: CIV or Paschen
    try:
        from dpf.experimental.civ_breakdown import (
            breakdown_narrative,
            compute_breakdown,
            compute_initial_sheath_state,
        )
        bd = compute_breakdown(
            V0=cc["V0"],
            fill_pressure_Pa=p_pa,
            anode_radius=a,
            cathode_radius=b,
            gas_name=gas_key,
        )
        bd_state = compute_initial_sheath_state(bd, a, b, p_pa)
        result["breakdown"] = {
            "mechanism": bd.mechanism,
            "gas": bd.gas.formula,
            "v_crit_km_s": bd.v_crit / 1e3,
            "v_ExB_km_s": bd.v_ExB / 1e3,
            "civ_ratio": bd.civ_ratio,
            "sheath_thickness_mm": bd.sheath_thickness * 1e3,
            "Te_eV": bd.Te_initial_eV,
            "ionization_fraction": bd.ionization_fraction,
            "breakdown_time_ns": bd.breakdown_time * 1e9,
            "liftoff_delay_ns": bd_state["liftoff_delay"] * 1e9,
            "paschen_voltage_V": bd.paschen_voltage,
            "electron_magnetized": bd.is_magnetized,
            "narrative": breakdown_narrative(bd),
            "summary": bd.summary,
        }
    except Exception as exc:
        logger.debug("CIV breakdown computation skipped: %s", exc)

    # Compute neutron yield from MHD state for deuterium fills
    if gas.get("A") == 2 and gas.get("Z") == 1:
        final_state = result.get("final_state")
        existing_neutron_yield = result.get("neutron_yield")
        existing_yield_history = result.get("yield_time_resolved")
        existing_field_history_yield = (
            (
                isinstance(existing_neutron_yield, dict)
                and existing_neutron_yield.get("thermonuclear_input_authority")
                == "resolved_field_history_candidate"
            )
            or (
                isinstance(existing_yield_history, dict)
                and existing_yield_history.get("source_authority")
                == "resolved_field_history_candidate"
            )
        )
        if final_state is not None and not existing_field_history_yield:
            try:
                from dpf.diagnostics.neutron_yield import neutron_yield_rate

                rho = final_state["rho"]
                ion_mass = gas["m_mol"]
                n_D = rho / ion_mass
                Ti = final_state.get("Ti", final_state["pressure"] * ion_mass / (2.0 * rho * kB))
                nr, ny_g, nz = rho.shape
                cell_vol = dr * (b - a) / nr * dz  # approximate cell volume
                _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)

                # Estimate confinement time from MHD evolution
                t_arr = result.get("t_us", np.array([]))
                if len(t_arr) > 1:
                    tau_pinch = (t_arr[-1] - t_arr[0]) * 1e-6  # total sim time in seconds
                else:
                    tau_pinch = t_end

                Y_thermo = total_rate * tau_pinch

                # Beam-target from circuit (if current available)
                I_arr = result.get("I_MA", np.array([]))
                Y_bt = 0.0
                V_pinch = 0.0
                if len(I_arr) > 0:
                    try:
                        from dpf.diagnostics.beam_target import beam_target_yield_rate
                        I_peak_A = float(np.max(np.abs(I_arr))) * 1e6
                        n_target = float(np.max(n_D))
                        L_pinch = L_anode * 0.3  # EMPIRICAL: ~30% of anode length
                        # V_pinch from dL/dt * I
                        L_arr = result.get("L_p_nH", np.array([]))
                        if len(L_arr) > 1 and len(t_arr) > 1:
                            dLdt = np.gradient(L_arr * 1e-9, t_arr * 1e-6)
                            V_pinch = float(np.max(np.abs(I_arr * 1e6 * dLdt)))
                        if V_pinch > 1e3:
                            bt_rate = beam_target_yield_rate(
                                I_peak_A, V_pinch, n_target, L_pinch, f_beam=0.14,
                            )
                            Y_bt = bt_rate * tau_pinch
                    except ImportError:
                        pass

                Y_total = Y_thermo + Y_bt
                if Y_total > 0:
                    result["neutron_yield"] = {
                        "Y_thermonuclear": float(Y_thermo),
                        "Y_beam_target": float(Y_bt),
                        "Y_neutron": float(Y_total),
                        "bt_fraction": float(Y_bt / Y_total) if Y_total > 0 else 0.0,
                        "V_pinch_kV": float(V_pinch / 1e3),
                        "tau_ns": float(tau_pinch * 1e9),
                        "model_role": "mechanism_separated_neutron_yield_estimate",
                        "validation_status": "estimate_not_validation",
                        "first_principles_total_yield_authority": "blocked",
                        "thermonuclear_input_authority": (
                            "final_state_duration_approximation"
                        ),
                        "beam_target_input_authority": (
                            "lee_saw_reduced_model_with_empirical_length"
                            if Y_bt > 0.0 else "not_produced"
                        ),
                        "beam_target_length_authority": (
                            "empirical_fraction_of_anode_length"
                            if Y_bt > 0.0 else "not_used"
                        ),
                        "can_support_first_principles_neutron_yield": False,
                        "validity_notes": {
                            "first_principles_boundary": (
                                "This app-level total is a reporting estimate. "
                                "The thermonuclear term uses final-state fields "
                                "over the simulated duration, and the beam-target "
                                "term uses the Lee/Saw reduced estimator with an "
                                "empirical pinch-length proxy."
                            ),
                        },
                    }
            except (ImportError, Exception) as exc:
                logger.debug("Neutron yield computation skipped: %s", exc)

    # Bennett equilibrium diagnostic — check if pinch achieves pressure balance
    final_state = result.get("final_state")
    I_arr = result.get("I_MA", np.array([]))
    if final_state is not None and len(I_arr) > 0:
        I_peak_A = float(np.max(np.abs(I_arr))) * 1e6
        rho_final = final_state["rho"]
        p_final = final_state["pressure"]
        B_final = final_state["B"]
        # Magnetic pressure at peak B location
        B2 = np.sum(B_final**2, axis=0)
        mu_0 = 4 * np.pi * 1e-7
        p_mag_max = float(np.max(B2)) / (2 * mu_0)
        p_kin_max = float(np.max(p_final))
        beta_pinch = p_kin_max / p_mag_max if p_mag_max > 0 else float("inf")
        # Bennett temperature from I^2 = (8*pi*N_L*kB*(Te+Ti))/mu_0
        # N_L = n * pi * r_p^2 — use peak density and anode radius as proxy
        n_peak = float(np.max(rho_final)) / gas["m_mol"]
        N_L = n_peak * np.pi * a**2
        if N_L > 0:
            T_bennett_K = mu_0 * I_peak_A**2 / (8 * np.pi * N_L * 2 * kB)
            T_bennett_keV = T_bennett_K * kB / (1000 * 1.602e-19)
        else:
            T_bennett_keV = 0.0
        result["bennett"] = {
            "beta_pinch": float(beta_pinch),
            "p_mag_max_Pa": float(p_mag_max),
            "p_kin_max_Pa": float(p_kin_max),
            "T_bennett_keV": float(T_bennett_keV),
            "source": "Bennett 1934, Russell 2025",
        }

    # Instability timing diagnostic (Goyon 2025, Eq. 4)
    # tau_m0 = 31.0 * R_imp^2 * sqrt(P_fill) / (CR * I_imp)
    # where R_imp = cathode radius [cm], CR = convergence ratio, I_imp = implosion current [MA]
    I_arr = result.get("I_MA", np.array([]))
    if len(I_arr) > 0:
        I_imp_MA = float(np.max(np.abs(I_arr)))
        R_imp_cm = b * 100  # cathode radius in cm
        P_fill_Torr = p_pa / 133.322
        CR = b / a if a > 0 else 10.0  # convergence ratio = cathode/anode radius
        if I_imp_MA > 0:
            tau_m0_ns = 31.0 * R_imp_cm**2 * np.sqrt(P_fill_Torr) / (CR * I_imp_MA)
            result["instability"] = {
                "tau_m0_ns": float(tau_m0_ns),
                "convergence_ratio": float(CR),
                "I_imp_MA": float(I_imp_MA),
                "source": "Goyon et al. 2025, Eq. 4",
            }

    # Synthetic interferometry diagnostic (Challenge 15)
    final_state = result.get("final_state")
    if final_state is not None:
        try:
            from dpf.diagnostics.interferometry import abel_transform, fringe_shift
            rho_final = final_state["rho"]
            ion_mass = gas["m_mol"]
            nz_mid = rho_final.shape[-1] // 2
            if rho_final.ndim == 3:
                ne_mid = rho_final[:, rho_final.shape[1] // 2, nz_mid] / ion_mass
            else:
                ne_mid = rho_final[:, nz_mid] / ion_mass
            r_arr = np.linspace(a + dr * 0.5, b - dr * 0.5, len(ne_mid))
            N_L = abel_transform(ne_mid, r_arr)
            fringes = fringe_shift(N_L)
            result["interferometry"] = {
                "r_mm": (r_arr * 1e3).tolist(),
                "ne_midplane_m3": ne_mid.tolist(),
                "line_integrated_m2": N_L.tolist(),
                "fringes_HeNe": fringes.tolist(),
                "peak_fringes": float(np.max(np.abs(fringes))),
            }
        except (ImportError, Exception):
            pass

    # Filamentation diagnostic (3D only, Challenge 8)
    if final_state is not None and len(final_state["rho"].shape) == 3:
        rho_f = final_state["rho"]
        if rho_f.shape[1] > 1:  # True 3D (ny > 1)
            try:
                from dpf.diagnostics.filamentation import detect_filaments
                fil = detect_filaments(rho_f, dx=dr)
                result["filamentation"] = {
                    "n_filaments": fil.n_filaments,
                    "dominant_m": fil.dominant_m,
                    "density_contrast": fil.density_contrast,
                    "filament_width_mm": fil.filament_width_mm,
                    "is_filamented": fil.is_filamented,
                }
            except (ImportError, Exception):
                pass

    # Plasmoid detection + force-free diagnostic (Challenge 14)
    if final_state is not None:
        try:
            from dpf.diagnostics.plasmoid import detect_plasmoids, force_free_diagnostic
            plasmoid_result = detect_plasmoids(
                final_state["B"], final_state["rho"], dr, dz,
            )
            if plasmoid_result["n_plasmoids"] > 0 or plasmoid_result["magnetic_energy_J"] > 0:
                result["plasmoids"] = {
                    k: v for k, v in plasmoid_result.items() if k != "psi_field"
                }
            ff = force_free_diagnostic(final_state["B"], dr, dz)
            result["force_free"] = {
                "alpha_ff": ff.alpha_ff,
                "j_parallel_frac": ff.j_parallel_frac,
                "force_free_error": ff.force_free_error,
                "is_relaxed": ff.is_relaxed,
            }
        except (ImportError, Exception):
            pass

    # Velocity shear stabilization diagnostic (Shumlak-Hartman criterion)
    if final_state is not None:
        try:
            from dpf.diagnostics.shear_stabilization import compute_shear_margin
            shear = compute_shear_margin(final_state, dr, dz, L_anode)
            if shear:
                result["shear_stabilization"] = shear
        except (ImportError, Exception):
            pass

    # Sweet-Parker reconnection diagnostic + energy spectrum
    if final_state is not None and final_state.get("B") is not None:
        try:
            from dpf.turbulence.subgrid import sweet_parker_diagnostic
            rho_f = final_state["rho"]
            Te_f = final_state.get("Te")
            if Te_f is None:
                Te_f = final_state["pressure"] * gas["m_mol"] / (2.0 * rho_f * kB)
            ne_f = rho_f / gas["m_mol"]
            sp = sweet_parker_diagnostic(
                final_state["B"], rho_f, Te_f, ne_f,
                dx=dr, L_system=L_anode,
                ion_mass=gas["m_mol"],
            )
            result["reconnection"] = {
                "S_lundquist": sp.S_lundquist,
                "rate": sp.reconnection_rate,
                "delta_sp_mm": sp.delta_sp * 1e3,
                "regime": sp.regime,
                "plasmoid_unstable": sp.plasmoid_unstable,
                "n_plasmoids_est": sp.n_plasmoids_est,
            }
        except (ImportError, Exception):
            pass

        try:
            from dpf.turbulence.subgrid import compute_energy_spectrum
            spec = compute_energy_spectrum(
                final_state.get("velocity", np.zeros((3, *final_state["rho"].shape))),
                final_state["B"], final_state["rho"], dx=dr,
            )
            result["turbulence_spectrum"] = {
                "spectral_index": spec.spectral_index,
                "inertial_range": spec.inertial_range,
                "has_spectrum": bool(np.sum(spec.E_k) > 0),
            }
        except (ImportError, Exception):
            pass

    # Radiation regime diagnostic
    if final_state is not None:
        try:
            from dpf.radiation.improved_radiation import radiation_regime_diagnostic
            rho_f = final_state["rho"]
            Te_f = final_state.get("Te", final_state["pressure"] * gas["m_mol"] / (2.0 * rho_f * kB))
            ne_f = rho_f / gas["m_mol"]
            B_f = final_state.get("B")
            B_mag_f = np.sqrt(np.sum(B_f**2, axis=0)) if B_f is not None else np.zeros_like(rho_f)
            Z_eff = float(gas.get("Z", 1))
            rad_diag = radiation_regime_diagnostic(Te_f, ne_f, B_mag_f, Z=Z_eff)
            result["radiation_regime"] = rad_diag
        except (ImportError, Exception):
            pass

    # QMF bremsstrahlung suppression diagnostic (p-B11 relevance)
    if final_state is not None and final_state.get("B") is not None:
        try:
            from dpf.radiation.qmf_suppression import qmf_diagnostic
            rho_f = final_state["rho"]
            Te_f = final_state.get("Te")
            if Te_f is None:
                Te_f = final_state["pressure"] * gas["m_mol"] / (2.0 * rho_f * kB)
            ne_f = rho_f / gas["m_mol"]
            qmf = qmf_diagnostic(final_state["B"], Te_f, ne_f)
            result["qmf"] = {
                "B_qmf_T": qmf.B_qmf_T,
                "ratio_Ec_Eth": qmf.ratio_Ec_Eth,
                "suppression_factor": qmf.suppression_factor,
                "is_qmf_regime": bool(qmf.is_qmf_regime),
                "note": qmf.note,
            }
        except (ImportError, Exception):
            pass

    # Beam-ion tracker (post-processing: inject beam at pinch, push through fields)
    if final_state is not None and gas.get("A") == 2 and gas.get("Z") == 1:
        try:
            from dpf.diagnostics.beam_tracker import BeamTracker
            nr_bt, ny_bt, nz_bt = final_state["rho"].shape
            # Beam energy from pinch voltage
            V_pinch_V = 0.0
            L_arr = result.get("L_p_nH", np.array([]))
            t_arr_bt = result.get("t_us", np.array([]))
            I_arr_bt = result.get("I_MA", np.array([]))
            if len(L_arr) > 1 and len(t_arr_bt) > 1:
                dLdt = np.gradient(L_arr * 1e-9, t_arr_bt * 1e-6)
                V_pinch_V = float(np.max(np.abs(I_arr_bt * 1e6 * dLdt)))
            beam_energy_eV = max(V_pinch_V, 50e3)  # minimum 50 keV
            if beam_energy_eV > 10e3:
                bt = BeamTracker(
                    n_particles=200, ion_mass=gas["m_mol"],
                    grid_shape=(nr_bt, ny_bt, nz_bt), dx=dr,
                )
                domain = np.array([nr_bt * dr, ny_bt * dr, nz_bt * dz])
                center = domain / 2.0
                bt.inject_beam(center, direction=np.array([0, 0, 1]),
                               energy_eV=beam_energy_eV, spread_rad=0.3)
                # Push dt: particle travels ~dx per step to stay on grid
                v_beam = np.sqrt(2 * beam_energy_eV * 1.602e-19 / gas["m_mol"])
                dt_push = min(dr / max(v_beam, 1.0), 1e-9)
                E_field = np.zeros((3, nr_bt, ny_bt, nz_bt))
                n_push = min(200, int(0.5 * min(domain) / max(v_beam * dt_push, 1e-30)))
                for _ in range(n_push):
                    bt.push(E_field, final_state["B"], dt_push)
                n_target = float(np.max(final_state["rho"])) / gas["m_mol"]
                bt_result = bt.get_result(n_target=n_target, L_pinch=L_anode * 0.3)
                if bt_result.n_particles > 0:
                    result["beam_tracker"] = {
                        "n_particles": bt_result.n_particles,
                        "mean_energy_keV": bt_result.mean_energy_keV,
                        "max_energy_keV": bt_result.max_energy_keV,
                        "beam_energy_input_keV": beam_energy_eV / 1e3,
                    }
        except (ImportError, Exception):
            pass

    # Plasma regime classification
    if final_state is not None:
        try:
            from dpf.diagnostics.regime_classifier import classify_regime
            rho_f = final_state["rho"]
            Te_f = final_state.get("Te")
            if Te_f is None:
                Te_f = final_state["pressure"] * gas["m_mol"] / (2.0 * rho_f * kB)
            B_f = final_state.get("B")
            B_mag_f = float(np.max(np.sqrt(np.sum(B_f**2, axis=0)))) if B_f is not None else 0.0
            Te_eV = float(np.max(Te_f)) * kB / 1.602e-19
            ne_peak = float(np.max(rho_f)) / gas["m_mol"]
            regime = classify_regime(
                n_e=ne_peak, T_e_eV=Te_eV, B_T=B_mag_f,
                L_m=min(b - a, 0.01), ion_mass_kg=gas["m_mol"],
            )
            result["plasma_regime"] = {
                "lundquist_S": regime.lundquist_S,
                "magnetic_reynolds": regime.magnetic_reynolds,
                "beta": regime.beta,
                "knudsen": regime.knudsen,
                "mhd_valid": regime.mhd_valid,
                "kinetic_needed": regime.kinetic_needed,
                "recommended_backend": regime.recommended_backend,
                "summary": regime.regime_summary,
            }
        except (ImportError, Exception):
            pass

    # Scaling law predictions
    try:
        from dpf.diagnostics.scaling_laws import compute_scaling
        cc_s = result.get("circuit", {})
        if cc_s and result.get("I_peak", 0) > 0:
            scaling = compute_scaling(
                I_pinch_kA=result["I_peak"] * 1e3,
                E_bank_kJ=result.get("E_bank_kJ", 0),
                a_mm=cc_s.get("anode_radius", 0.01) * 1e3,
                b_mm=cc_s.get("cathode_radius", 0.03) * 1e3,
            )
            result["scaling_laws"] = scaling.to_summary_dict()
    except (ImportError, Exception):
        pass

    # Circuit waveform validation against registered experimental traces.
    try:
        from dpf.validation.experimental import DEVICES
        from dpf.validation.quality_assessment import (
            circuit_validation_evidence_from_waveform,
        )

        device_name = str(result.get("device", ""))
        t_us_cv = result.get("t_us", np.array([]))
        I_MA_cv = result.get("I_MA", np.array([]))
        if (
            device_name in DEVICES
            and "circuit_validation" not in result
            and len(t_us_cv) > 1
            and len(I_MA_cv) > 1
        ):
            result["circuit_validation"] = circuit_validation_evidence_from_waveform(
                np.asarray(t_us_cv, dtype=float) * 1.0e-6,
                np.asarray(I_MA_cv, dtype=float) * 1.0e6,
                device_name,
            )
    except Exception as exc:
        _record_validation_error(result, "circuit_waveform_validation", exc)

    # Snowplow phase validation. Phase labels alone stay as candidate evidence;
    # only caller-supplied reference targets are promoted to tier-2 validation.
    try:
        from dpf.validation.quality_assessment import (
            snowplow_phase_observation_from_history,
            snowplow_validation_evidence_from_phase_history,
        )
        from dpf.validation.kr_targets import (
            pf1000_16kv_derived_output_candidate_evidence,
            pf1000_16kv_phase_candidate_evidence_from_history,
        )

        phases_sp = result.get("phases")
        t_us_sp = result.get("t_us", np.array([]))
        phase_targets_s = (
            result.get("snowplow_phase_targets_s")
            or result.get("phase_timing_targets_s")
        )
        phase_history_present = (
            phases_sp is not None and len(t_us_sp) > 1 and len(phases_sp) > 1
        )
        if phase_history_present:
            if phase_targets_s and "snowplow_validation" not in result:
                phase_target_metadata = result.get("snowplow_phase_target_metadata", {})
                result["snowplow_validation"] = (
                    snowplow_validation_evidence_from_phase_history(
                        np.asarray(t_us_sp, dtype=float) * 1.0e-6,
                        phases_sp,
                        phase_targets_s,
                        reference_source=str(
                            phase_target_metadata.get("source", "")
                            if isinstance(phase_target_metadata, dict) else ""
                        ),
                        reference_kr_status=str(
                            phase_target_metadata.get("kr_status", "")
                            if isinstance(phase_target_metadata, dict) else ""
                        ),
                    )
                )
            elif "snowplow_validation_candidate" not in result:
                times_s = np.asarray(t_us_sp, dtype=float) * 1.0e-6
                device_phase = str(
                    result.get("device", preset_name)
                ).lower().replace("_", "-")
                v0_phase = float(cc.get("V0", 0.0))
                is_pf1000_16kv_phase = (
                    ("pf-1000" in device_phase or "pf1000" in device_phase)
                    and 0.95 * 16.0e3 <= v0_phase <= 1.05 * 16.0e3
                )
                if is_pf1000_16kv_phase:
                    result["snowplow_validation_candidate"] = (
                        pf1000_16kv_phase_candidate_evidence_from_history(
                            times_s,
                            phases_sp,
                        )
                    )
                else:
                    result["snowplow_validation_candidate"] = (
                        snowplow_phase_observation_from_history(
                            times_s,
                            phases_sp,
                        )
                    )

                if (
                    is_pf1000_16kv_phase
                    and "snowplow_dynamics_validation_candidate" not in result
                ):
                    observables = dict(
                        result.get("pf1000_16kv_derived_outputs", {})
                        if isinstance(result.get("pf1000_16kv_derived_outputs"), dict)
                        else {}
                    )
                    if result.get("I_peak") is not None:
                        observables.setdefault(
                            "peak_current_kA",
                            float(result["I_peak"]) * 1.0e3,
                        )
                    I_MA_sp = result.get("I_MA")
                    if I_MA_sp is not None:
                        n_phase_current = min(len(phases_sp), len(I_MA_sp))
                        for idx in range(n_phase_current):
                            phase = str(phases_sp[idx]).strip().lower()
                            if phase in {"pinch", "reflected", "post_pinch"}:
                                observables.setdefault(
                                    "pinch_current_kA",
                                    abs(float(I_MA_sp[idx])) * 1.0e3,
                                )
                                break
                    if observables:
                        result["snowplow_dynamics_validation_candidate"] = (
                            pf1000_16kv_derived_output_candidate_evidence(
                                observables,
                            )
                        )
        phase_validation = result.get("snowplow_validation")
        phase_candidate = result.get("snowplow_validation_candidate")
        if isinstance(phase_validation, dict) and phase_validation.get("passed") is True:
            phase_status = "supported"
            missing_phase_inputs: list[str] = []
        elif not phase_history_present:
            phase_status = "missing_phase_history"
            missing_phase_inputs = ["phase_history"]
        elif phase_targets_s:
            phase_status = "target_comparison_failed_or_blocked"
            missing_phase_inputs = ["passing_same_device_phase_comparison"]
        elif isinstance(phase_candidate, dict):
            phase_status = "candidate_observed_no_verified_targets"
            missing_phase_inputs = [
                "same_device_kr_verified_phase_targets",
                "phase_timing_uncertainty",
            ]
        else:
            phase_status = "missing_verified_targets"
            missing_phase_inputs = [
                "same_device_kr_verified_phase_targets",
                "phase_timing_uncertainty",
            ]
        result["snowplow_phase_validation_status"] = {
            "passed": phase_status == "supported",
            "validation_tier": 2,
            "model_role": "snowplow_phase_validation_status",
            "status": phase_status,
            "phase_history_present": phase_history_present,
            "phase_targets_present": bool(phase_targets_s),
            "candidate_present": isinstance(phase_candidate, dict),
            "missing_required_inputs": missing_phase_inputs,
            "validity_notes": {
                "tier_scope": (
                    "Observed phase labels are candidates. Tier-2 support "
                    "requires same-device KR-verified axial, radial, and "
                    "pinch/stagnation timing targets with uncertainty."
                ),
            },
        }
    except Exception as exc:
        _record_validation_error(result, "snowplow_phase_validation", exc)

    # Spatial validation components: candidates are exposed, but only complete
    # same-scope evidence is promoted to `spatial_validation`.
    try:
        from dpf.diagnostics.xray_imaging import radiating_pinch_geometry_from_image
        from dpf.validation.kr_targets import (
            dpf_pinch_temperature_evidence,
            llnl_12kj_em_fluctuation_evidence_from_signal,
            pf1000_interferometry_density_evidence_from_profile,
            pf1000_spatial_pinch_evidence_from_geometry,
        )
        from dpf.validation.quality_assessment import (
            combine_spatial_validation_evidence,
            spatial_validation_scope_closure_report,
        )

        xray_image = result.get("xray_image")
        if xray_image is None:
            xray_image = result.get("synthetic_xray_image")
        y_cell_sp = result.get("xray_y_cell_m")
        z_cell_sp = result.get("xray_z_cell_m")
        device_sp = str(result.get("device", preset_name)).lower().replace("_", "-")
        is_pf1000 = "pf-1000" in device_sp or "pf1000" in device_sp
        if (
            is_pf1000
            and xray_image is not None
            and y_cell_sp is not None
            and z_cell_sp is not None
        ):
            geometry = radiating_pinch_geometry_from_image(
                np.asarray(xray_image, dtype=float),
                np.asarray(y_cell_sp, dtype=float),
                np.asarray(z_cell_sp, dtype=float),
            )
            result["pf1000_radiating_pinch_geometry"] = geometry
            result.setdefault("spatial_validation_components", []).append(
                pf1000_spatial_pinch_evidence_from_geometry(geometry)
            )

        density_radius_cm = result.get("pf1000_interferometry_radius_cm")
        if density_radius_cm is None:
            density_radius_cm = result.get("density_profile_radius_cm")
        if density_radius_cm is None and result.get("density_profile_radius_m") is not None:
            density_radius_cm = (
                np.asarray(result["density_profile_radius_m"], dtype=float) * 100.0
            )

        density_profile_cm3 = result.get("pf1000_interferometry_density_cm3")
        if density_profile_cm3 is None:
            density_profile_cm3 = result.get("electron_density_profile_cm3")
        if (
            density_profile_cm3 is None
            and result.get("electron_density_profile_m3") is not None
        ):
            density_profile_cm3 = (
                np.asarray(result["electron_density_profile_m3"], dtype=float) * 1.0e-6
            )

        if (
            is_pf1000
            and density_radius_cm is not None
            and density_profile_cm3 is not None
        ):
            result.setdefault("spatial_validation_components", []).append(
                pf1000_interferometry_density_evidence_from_profile(
                    np.asarray(density_radius_cm, dtype=float),
                    np.asarray(density_profile_cm3, dtype=float),
                    shot=str(result.get("shot", "13328")),
                )
            )

        em_times_s = result.get("em_probe_times_s")
        em_signal = result.get("em_probe_signal")
        device_is_llnl = "llnl" in device_sp and "1.2" in device_sp
        if device_is_llnl and em_times_s is not None and em_signal is not None:
            result.setdefault("spatial_validation_components", []).append(
                llnl_12kj_em_fluctuation_evidence_from_signal(
                    np.asarray(em_times_s, dtype=float),
                    np.asarray(em_signal, dtype=float),
                )
            )

        T_arr_sp = result.get("T_max", np.array([]))
        if len(T_arr_sp) > 0:
            T_peak_K = float(np.nanmax(T_arr_sp))
            T_peak_keV = T_peak_K * kB / (1000.0 * 1.602e-19)
            temp_evidence = dpf_pinch_temperature_evidence(
                electron_temperature_keV=T_peak_keV,
            )
            result.setdefault("spatial_validation_components", []).append(temp_evidence)

        components = result.get("spatial_validation_components", [])
        if components and "spatial_validation" not in result:
            result["spatial_validation_scope_closure"] = (
                spatial_validation_scope_closure_report(components)
            )
            combined_spatial = combine_spatial_validation_evidence(components)
            if combined_spatial["passed"]:
                result["spatial_validation"] = combined_spatial
            else:
                result["spatial_validation_candidate"] = combined_spatial
        elif "spatial_validation_scope_closure" not in result:
            result["spatial_validation_scope_closure"] = (
                spatial_validation_scope_closure_report([])
            )
    except Exception as exc:
        _record_validation_error(result, "spatial_validation", exc)

    # PF-1000 16 kV Akel scalar-yield validation. This requires callers to
    # supply a full 24-shot prediction table; a single run yield is not enough.
    try:
        from dpf.validation.kr_targets import (
            pf1000_16kv_akel_table_candidate_evidence,
        )

        prediction_rows = (
            result.get("pf1000_16kv_akel_table_predictions")
            or result.get("akel_2021_table_predictions")
            or result.get("neutron_yield_validation_rows")
        )
        device_yield = str(result.get("device", preset_name)).lower().replace("_", "-")
        is_pf1000_yield = "pf-1000" in device_yield or "pf1000" in device_yield
        v0_yield = float(cc.get("V0", 0.0))
        if (
            prediction_rows is not None
            and is_pf1000_yield
            and 0.95 * 16.0e3 <= v0_yield <= 1.05 * 16.0e3
        ):
            yield_evidence = pf1000_16kv_akel_table_candidate_evidence(
                prediction_rows,
            )
            if yield_evidence["passed"]:
                result["neutron_yield_validation"] = yield_evidence
            else:
                result["neutron_yield_validation_candidate"] = yield_evidence
    except Exception as exc:
        _record_validation_error(result, "pf1000_16kv_akel_yield_validation", exc)

    # Mechanism-separated neutron output summary. This is production reporting,
    # not validation; it prevents total-yield estimates from hiding missing
    # timing, spectrum, anisotropy, detector/activation, and UQ evidence.
    try:
        from dpf.validation.first_principles_mhd import (
            first_principles_neutron_yield_authority_status,
        )

        mechanism_summary = _neutron_mechanism_output_summary(result)
        if mechanism_summary is not None:
            result.setdefault("neutron_mechanism_outputs", mechanism_summary)
            neutron_yield = result.get("neutron_yield")
            if isinstance(neutron_yield, dict):
                result.setdefault(
                    "neutron_yield_details",
                    {
                        **neutron_yield,
                        "model_role": "mechanism_separated_neutron_yield_estimate",
                        "validation_status": "estimate_not_validation",
                        "validation_blockers": mechanism_summary[
                            "validation_blockers"
                        ],
                    },
                )
            result["first_principles_neutron_yield_authority"] = (
                first_principles_neutron_yield_authority_status(result)
            )
    except Exception as exc:
        _record_validation_error(result, "neutron_mechanism_outputs", exc)

    # KnowledgeReference-backed MJOLNIR neutron timing comparison.
    # Only phase-timed comparisons are fed to the predictive-readiness gate.
    if (
        gas.get("A") == 2
        and gas.get("Z") == 1
        and result.get("yield_time_resolved")
        and str(result.get("device", preset_name)).lower() == "mjolnir"
    ):
        try:
            from dpf.validation.kr_targets import (
                mjolnir_neutron_anisotropy_evidence,
                mjolnir_neutron_detector_response_evidence,
                mjolnir_neutron_spectrum_evidence,
                mjolnir_neutron_timing_evidence_from_history,
            )

            stagnation_time_s = _phase_stagnation_time_s(result)
            neutron_timing = mjolnir_neutron_timing_evidence_from_history(
                result["yield_time_resolved"],
                stagnation_time_s=stagnation_time_s,
                require_measurement_correlation=True,
            )
            inferred = neutron_timing.get("details", {}).get(
                "stagnation_time_inferred_from_thermonuclear_peak", False
            )
            if inferred:
                result["neutron_timing_validation_candidate"] = neutron_timing
            else:
                result["neutron_mechanism_timing_validation"] = neutron_timing

            spectrum_samples = result.get("neutron_spectrum_samples_MeV")
            if isinstance(spectrum_samples, dict):
                thermo_spectrum = spectrum_samples.get("thermonuclear")
                beam_spectrum = spectrum_samples.get("beam_target")
                if thermo_spectrum is not None and beam_spectrum is not None:
                    result["neutron_spectrum_validation"] = (
                        mjolnir_neutron_spectrum_evidence(
                            thermo_spectrum,
                            beam_spectrum,
                        )
                    )

            anisotropy = result.get("neutron_anisotropy")
            if isinstance(anisotropy, dict):
                on_axis = anisotropy.get("on_axis_yield")
                off_axis = anisotropy.get("off_axis_yield")
                if on_axis is not None and off_axis is not None:
                    result["neutron_anisotropy_validation"] = (
                        mjolnir_neutron_anisotropy_evidence(
                            on_axis,
                            off_axis,
                            yield_regime=str(anisotropy.get("yield_regime", "high_yield")),
                        )
                    )

            detector_response = result.get("neutron_detector_response")
            if detector_response is None:
                detector_response = result.get("detector_response")
            if isinstance(detector_response, dict):
                response_evidence = mjolnir_neutron_detector_response_evidence(
                    detector_response,
                )
                if response_evidence["passed"]:
                    result["neutron_detector_response_validation"] = response_evidence
                else:
                    result["neutron_detector_response_validation_candidate"] = (
                        response_evidence
                    )
        except Exception as exc:
            _record_validation_error(result, "mjolnir_neutron_validation", exc)

    # Tier-5 neutron closure: yield, timing, spectrum, and anisotropy must share scope.
    try:
        if (
            result.get("neutron_yield_validation")
            or result.get("neutron_mechanism_timing_validation")
            or result.get("neutron_spectrum_validation")
            or result.get("neutron_anisotropy_validation")
        ):
            from dpf.validation.quality_assessment import (
                neutron_validation_scope_closure_report,
            )

            result["neutron_validation_scope_closure"] = (
                neutron_validation_scope_closure_report(result)
            )
    except Exception as exc:
        _record_validation_error(result, "neutron_scope_closure", exc)

    # High-fidelity physics audit. This is intentionally conservative and
    # reports missing/unvalidated physics effects rather than promoting claims.
    try:
        from dpf.validation.physics_fidelity import (
            physics_fidelity_evidence_from_result,
        )

        result["physics_fidelity_evidence"] = (
            physics_fidelity_evidence_from_result(
                result,
                active_modules=active_modules,
            )
        )
    except Exception as exc:
        _record_validation_error(result, "physics_fidelity", exc)

    # Circuit/field coupling audit. This records coupling signals and keeps
    # MHD current-prediction claims blocked until KR-backed validation exists.
    try:
        from dpf.validation.circuit_field_coupling import (
            dynamic_inductance_power_balance_from_waveforms,
            field_coupling_evidence_from_result,
        )

        if "dynamic_inductance_power_balance" not in result:
            inductance_nH = None
            for key in ("Lp_mhd_nH", "L_p_nH", "Lp_nH"):
                if key in result:
                    inductance_nH = result.get(key)
                    break
            if (
                inductance_nH is not None
                and result.get("t_us") is not None
                and result.get("I_MA") is not None
            ):
                result["dynamic_inductance_power_balance"] = (
                    dynamic_inductance_power_balance_from_waveforms(
                        np.asarray(result["t_us"], dtype=float) * 1.0e-6,
                        np.asarray(result["I_MA"], dtype=float) * 1.0e6,
                        np.asarray(inductance_nH, dtype=float) * 1.0e-9,
                    )
                )
        result["field_coupling_validation"] = (
            field_coupling_evidence_from_result(result)
        )
    except Exception as exc:
        _record_validation_error(result, "field_coupling", exc)

    # Uncertainty-budget audit. This exposes missing uncertainty components
    # without promoting nominal tolerances to high-fidelity UQ.
    try:
        from dpf.validation.uncertainty_budget import (
            uncertainty_evidence_from_result,
            validation_uncertainty_coverage_from_result,
        )

        result["validation_uncertainty_coverage"] = (
            validation_uncertainty_coverage_from_result(result)
        )
        result["uncertainty_validation"] = (
            uncertainty_evidence_from_result(result)
        )
    except Exception as exc:
        _record_validation_error(result, "uncertainty_budget", exc)

    # MHD numerical-fidelity audit. This keeps generic backend verification
    # separate from DPF-specific cylindrical/circuit/convergence requirements.
    try:
        from dpf.validation.mhd_numerical_fidelity import (
            mhd_numerical_fidelity_evidence_from_result,
            mhd_numerical_verification_packet_status,
            mhd_scope_limit_evidence_from_phases,
        )

        if "mhd_scope_limit" not in result and (
            result.get("has_mhd")
            or result.get("mhd_numerical_method")
            or "mhd" in str(result.get("backend", effective_backend)).lower()
        ):
            result["mhd_scope_limit"] = mhd_scope_limit_evidence_from_phases(
                applicable_phases=["formation", "first_collapse"],
                invalid_phases=["after_first_collapse", "post_disruption"],
                limit_reasons=[
                    "Rayleigh-Taylor instability",
                    "non-ideal electric fields beyond ideal MHD",
                ],
            )
        result["mhd_numerical_fidelity"] = (
            mhd_numerical_fidelity_evidence_from_result(result)
        )
        result["mhd_numerical_verification_packet_status"] = (
            mhd_numerical_verification_packet_status(result)
        )
    except Exception as exc:
        _record_validation_error(result, "mhd_numerical_fidelity", exc)

    # First-principles run-mode metadata. This is a fail-closed contract layer:
    # it never promotes app results to accepted evidence by itself.
    try:
        from dpf.presets import list_presets
        from dpf.validation.first_principles_mhd import (
            FIRST_PRINCIPLES_MHD_MODE,
            annotate_first_principles_mhd_result,
        )

        run_mode = str(
            requested_run_mode
            or result.get("requested_run_mode")
            or result.get("run_mode")
            or ""
        )
        if run_mode == FIRST_PRINCIPLES_MHD_MODE:
            preset_authority = next(
                (item for item in list_presets() if item.get("name") == preset_name),
                {},
            )
            annotate_first_principles_mhd_result(
                result,
                preset_name=preset_name,
                validation_scope=str(preset_authority.get("validation_scope", "")),
                source_scope=str(preset_authority.get("source_scope", "")),
                source_scope_status=str(
                    preset_authority.get("source_scope_status", "")
                ),
                requested_mode=FIRST_PRINCIPLES_MHD_MODE,
                execution_mode=str(effective_backend),
            )
    except Exception as exc:
        _record_validation_error(result, "first_principles_mhd", exc)

    # Validation and predictive-readiness gate
    try:
        from dataclasses import asdict

        from dpf.validation.digitization import (
            akel_fig1_draft_digitization_packet,
            scientific_closure_digitization_queue,
            scientific_closure_digitization_status,
        )
        from dpf.validation.kr_corpus import kr_corpus_review_status
        from dpf.validation.kr_targets import (
            kr_validation_same_scope_target_report,
            kr_validation_target_coverage_report,
            kr_validation_target_semantic_audit,
            kr_validation_target_source_audit,
            pf1000_16kv_current_waveform_comparison_candidate_evidence,
        )
        from dpf.validation.quality_assessment import (
            high_fidelity_readiness_report,
            predictive_readiness_report,
            scientific_accuracy_gap_report,
            source_authority_evidence_from_result,
            validation_tier_report,
        )
        from dpf.validation.source_acquisition import (
            scientific_closure_source_acquisition_queue,
        )

        result["kr_validation_target_source_audit"] = (
            kr_validation_target_source_audit()
        )
        result["kr_validation_target_semantic_audit"] = (
            kr_validation_target_semantic_audit()
        )
        result["kr_validation_target_coverage"] = (
            kr_validation_target_coverage_report()
        )
        result["kr_validation_same_scope_targets"] = (
            kr_validation_same_scope_target_report()
        )
        result["kr_corpus_review_status"] = kr_corpus_review_status()
        result["scientific_closure_source_acquisition_queue"] = (
            scientific_closure_source_acquisition_queue()
        )
        result["scientific_closure_digitization_queue"] = (
            scientific_closure_digitization_queue()
        )
        digitization_packets = (
            result.get("scientific_closure_digitization_packets")
            or result.get("digitization_packets")
        )
        result["scientific_closure_digitization_status"] = (
            scientific_closure_digitization_status(digitization_packets)
        )
        waveform_packet = (
            result.get("pf1000_16kv_current_waveform_digitization_packet")
            or result.get("akel_fig1_current_waveform_digitization_packet")
            or akel_fig1_draft_digitization_packet()
        )
        result["pf1000_16kv_current_waveform_comparison_candidate"] = (
            pf1000_16kv_current_waveform_comparison_candidate_evidence(
                result.get("t_us", []),
                result.get("I_MA", []),
                waveform_packet,
                uncertainty=(
                    result.get("pf1000_16kv_current_waveform_uncertainty")
                    or result.get("current_waveform_uncertainty")
                    or {}
                ),
            )
        )
        if "source_authority_validation" not in result:
            result["source_authority_validation"] = (
                source_authority_evidence_from_result(result)
            )
        result["validation_tiers"] = [
            asdict(tier) for tier in validation_tier_report(result)
        ]
        result["predictive_readiness"] = asdict(predictive_readiness_report(result))
        result["scientific_accuracy_gaps"] = [
            asdict(gap) for gap in scientific_accuracy_gap_report(result)
        ]
        result["high_fidelity_readiness"] = asdict(
            high_fidelity_readiness_report(result)
        )
    except Exception as exc:
        _record_validation_error(result, "predictive_readiness", exc)

    result["reproducibility"] = {
        "version": "v1.4.0",
        "git_hash": _git_hash,
        "timestamp": _dt.now().isoformat(),
        "backend": effective_backend,
        "grid_shape": grid_shape,
        "sim_time_us": sim_time_us,
        "preset": preset_name,
        "advanced_physics": active_modules,
    }


def _run_hybrid_lee_mhd(
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, sc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
    enable_fld: bool = False,
    enable_sheath: bool = False,
    enable_ablation: bool = False,
    enable_nernst: bool = False,
    enable_cr: bool = False,
) -> dict[str, Any]:
    """Hybrid Lee+MHD: Lee model runs axial rundown, MHD handles radial implosion.

    Phase 1 (Lee): Snowplow model sweeps gas along anode. Fast reduced-order model.
        Provides: circuit state (I, V), swept mass, sheath velocity at transition.
    Phase 2 (MHD): Metal solver takes over at start of radial phase.
        IC: compressed gas column with B_theta from circuit current.
        Resolves: radial implosion, pinch compression, instabilities.
    """
    import torch

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.fluid.snowplow import SnowplowModel
    from dpf.metal.metal_solver import MetalMHDSolver

    mu_0 = 4.0 * np.pi * 1e-7

    # ---- Phase 1: Lee model axial rundown ----
    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
        crowbar_inductance=cc.get("crowbar_inductance", 0.0),
    )

    snowplow = SnowplowModel(
        anode_radius=a, cathode_radius=b,
        fill_density=rho0,
        anode_length=L_anode,
        mass_fraction=sc.get("mass_fraction", 0.15),
        fill_pressure_Pa=sc.get("fill_pressure_Pa", p_pa),
        current_fraction=sc.get("current_fraction", 0.7),
        radial_mass_fraction=sc.get("radial_mass_fraction"),
        pinch_column_fraction=sc.get("pinch_column_fraction", 1.0),
    )

    # Run Lee model until radial phase begins (or t_end)
    L_total = cc["L0"] + 1e-9
    T_LC = 2 * np.pi * np.sqrt(L_total * cc["C"])
    dt_lee = T_LC / 5000

    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    sheath_zs, shock_rs, phases_list = [], [], []

    t = 0.0
    coupling = CouplingState()
    lee_steps = 0
    handoff_time = None

    while t < t_end:
        sp = snowplow.step(dt_lee, circuit.current)
        coupling.Lp = sp["L_plasma"]
        coupling.dL_dt = sp["dL_dt"]
        coupling.R_plasma = sp.get("R_plasma", 0.0)
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt_lee)
        t += dt_lee
        lee_steps += 1

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        sheath_zs.append(sp["z_sheath"] * 1e3)
        shock_rs.append(sp["r_shock"] * 1e3)
        phases_list.append(sp["phase"])

        if progress_fn and lee_steps % 50 == 0:
            progress_fn(
                min(t / t_end, 0.3),
                desc=f"Phase 1/2 — Axial rundown: t={t*1e6:.1f} us | sheath at z={sp['z_sheath']*1e3:.0f} mm | I={circuit.current/1e6:.2f} MA",
            )

        # Handoff when Lee model enters radial phase
        if sp["phase"] == "radial":
            handoff_time = t
            break

    if handoff_time is None:
        # Never reached radial phase — return Lee-only results
        logger.warning("Hybrid: Lee model didn't reach radial phase in %.1f us", t_end * 1e6)
        t_arr = np.array(times)
        I_arr = np.array(currents)
        I_peak_idx = int(np.argmax(np.abs(I_arr)))
        return {
            "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
            "L_p_nH": np.array(L_plasmas),
            "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
            "E_res_kJ": np.array(E_res),
            "z_mm": np.array(sheath_zs), "r_mm": np.array(shock_rs),
            "phases": phases_list,
            "I_peak": float(np.abs(I_arr[I_peak_idx])),
            "t_peak": float(t_arr[I_peak_idx]),
            "n_steps": lee_steps,
            "has_snowplow": True, "has_mhd": False,
            "mhd_snapshots": [], "final_state": None,
            "dip_pct": 0.0, "I_pre_dip": float(np.abs(I_arr[I_peak_idx])),
            "I_dip": 0.0, "t_dip": 0.0,
            "scaling": None, "crowbar_t": None,
            "snowplow_obj": snowplow, "dt_ns": dt_lee * 1e9,
            "rho_max": np.array([rho0] * len(times)),
            "T_max": np.array([300.0] * len(times)),
            "B_max": np.array([0.0] * len(times)),
        }

    # ---- Phase 2: MHD radial implosion ----
    I_handoff = circuit.current  # [A] at start of radial phase
    fc = sc.get("current_fraction", 0.7)
    fm = sc.get("mass_fraction", 0.15)
    fmr = sc.get("radial_mass_fraction", fm)
    z_f = sc.get("pinch_column_fraction", 1.0) * L_anode

    nr, ny, nz = grid_shape
    cfg = BACKEND_CONFIGS["metal_plm"]
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"

    # MHD domain: radial extent = cathode - anode, axial = z_f (pinch column)
    dr_mhd = (b - a) / nr
    dz_mhd = z_f / max(nz, 1)

    solver = MetalMHDSolver(
        grid_shape=grid_shape, dx=dr_mhd, dz=dz_mhd,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3, device=device,
        use_ct=False,
        coordinates="cylindrical",
        ion_mass=gas["m_mol"],
        r_inner=a,
        convert_b_si_to_hl=True,
        **cfg,
    )

    # Build physically motivated IC for MHD radial phase:
    # - Swept mass concentrated near cathode (outer boundary)
    # - Unswept gas fills the interior
    # - B_theta = mu_0 * fc * I / (2*pi*r) throughout
    r_cells = np.linspace(a + dr_mhd * 0.5, b - dr_mhd * 0.5, nr)

    # Density: swept mass forms a shell near cathode, unswept gas elsewhere
    rho_bg = rho0 * (1.0 - fmr)  # background (unswept)
    # Swept mass distributed in outer 20% of radial cells (current sheath)
    n_sheath = max(int(0.2 * nr), 2)
    rho_mhd = np.full((nr, ny, nz), rho_bg)
    # Sheath density: all swept mass in the thin shell
    shell_vol = sum(
        2.0 * np.pi * r_cells[nr - n_sheath + i] * dr_mhd * dz_mhd
        for i in range(n_sheath)
    )
    swept_mass_per_z = fmr * rho0 * np.pi * (b**2 - a**2)  # [kg/m]
    rho_sheath = swept_mass_per_z * dz_mhd / max(shell_vol, 1e-30)
    rho_mhd[nr - n_sheath:, :, :] = max(rho_sheath, rho_bg * 2.0)

    # B_theta profile from circuit current (the magnetic piston)
    B_theta_1d = mu_0 * fc * I_handoff / (2.0 * np.pi * r_cells)
    B_mhd = np.zeros((3, nr, ny, nz))
    B_mhd[1] = B_theta_1d[:, np.newaxis, np.newaxis]  # B_theta

    # Pressure: gas pressure + kinetic pressure from sheath velocity
    # In the Lee model, the sheath starts radial phase with vr = 0
    # but has high magnetic pressure behind it
    p_mhd = np.full((nr, ny, nz), p_pa)

    state = {
        "rho": rho_mhd,
        "velocity": np.zeros((3, nr, ny, nz)),
        "pressure": p_mhd,
        "B": B_mhd,
        "Te": np.full((nr, ny, nz), 300.0),
        "Ti": np.full((nr, ny, nz), 300.0),
        "psi": np.zeros((nr, ny, nz)),
    }

    # Continue circuit from handoff state
    rho_max_arr = [float(np.max(rho_mhd))]
    T_max_arr = [300.0]
    B_max_arr = [float(np.max(np.abs(B_mhd)))]
    mhd_snapshots = []

    t_mhd_start = t
    mhd_step = 0
    mu_0_local = 4.0 * np.pi * 1e-7
    # Initialize prev_Lp from Lee model's final inductance to avoid discontinuity
    prev_Lp = coupling.Lp
    # Maximum physically reasonable back-EMF: ~50 kV for any DPF device
    MAX_BACK_EMF = 50e3  # [V]
    # Target ~30 snapshots during MHD phase
    _target_snaps = 30
    snap_interval = 3
    _cr_fracs = None  # CR ionization charge-state fractions
    _L_axial_frozen = coupling.Lp  # cache axial inductance at Lee→MHD handoff

    # Time-resolved yield tracker (hybrid mode)
    yield_tracker = None
    try:
        from dpf.diagnostics.yield_tracker import YieldTracker
        yield_tracker = YieldTracker(ion_mass=gas["m_mol"], rho0=rho0)
    except ImportError:
        pass

    # tqdm progress bar — Gradio's track_tqdm=True captures this for real-time UI updates
    _pbar = tqdm(desc="Phase 2/2 — MHD compression", unit="step", leave=False)

    while t < t_end:
        dt_mhd = solver.compute_dt(state)
        dt = min(dt_mhd, t_end - t)
        if dt <= 0:
            break

        state = solver.step(
            state, dt, current=circuit.current, voltage=circuit.voltage,
            anode_radius=a, cathode_radius=b, apply_electrode_bc=True,
        )

        # Radiation cooling: improved model (T-dependent Gaunt + cyclotron)
        if "Te" in state:
            try:
                from dpf.radiation.improved_radiation import apply_improved_radiation_losses
                rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
                ne = rho_safe / gas["m_mol"]
                Z_eff = gas.get("Z", 1)
                B_field = state.get("B")
                B_mag = np.sqrt(np.sum(B_field**2, axis=0)) if B_field is not None else None
                state["Te"], _ = apply_improved_radiation_losses(
                    state["Te"], ne, dt, Z=Z_eff, B_mag=B_mag,
                )
            except ImportError:
                try:
                    from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses
                    rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
                    ne = rho_safe / gas["m_mol"]
                    state["Te"], _ = apply_bremsstrahlung_losses(state["Te"], ne, dt, Z=gas.get("Z", 1))
                except ImportError:
                    pass

        # Advanced physics operator-split (FLD, sheath, ablation, Nernst, CR)
        any_adv = enable_fld or enable_sheath or enable_ablation or enable_nernst or enable_cr
        if any_adv:
            state, _cr_fracs = _apply_advanced_physics(
                state, dt, gas, dr_mhd, dz_mhd, a, b,
                enable_fld=enable_fld, enable_sheath=enable_sheath,
                enable_ablation=enable_ablation, enable_nernst=enable_nernst,
                enable_cr=enable_cr, cr_fractions=_cr_fracs,
            )

        # Time-resolved yield accumulation (hybrid MHD phase)
        if yield_tracker is not None:
            V_p = abs(coupling.dL_dt) * abs(circuit.current) if coupling.dL_dt else 0.0
            cell_vol = dr_mhd * (b - a) / nr * dz_mhd
            yield_tracker.accumulate(
                state, dt,
                I_current=circuit.current, V_pinch=V_p,
                cell_volume=cell_vol,
            )

        # Compute L_plasma from MHD density profile using Lee-model formula.
        # Extract effective compression radius from density, then use the
        # analytic coaxial inductance: L_p = L_axial + (mu_0/2pi)*z_f*ln(b/r_eff)
        # This avoids including electrode BC energy (externally imposed, not plasma load).
        rho_mid = state["rho"][:, ny // 2, nz // 2]  # midplane radial profile
        rho_bg = rho0 * (1.0 - fmr)
        # Effective compression radius: density-weighted mean radius
        weights = np.maximum(rho_mid - rho_bg, 0.0)
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            r_eff = float(np.sum(r_cells * weights)) / w_sum
        else:
            r_eff = b  # no compression yet
        r_eff = max(r_eff, a * 0.01)  # floor to prevent log(0)
        # Lee-model inductance: axial (frozen at handoff) + radial compression
        Lp_mhd = _L_axial_frozen + (mu_0_local / (2.0 * np.pi)) * z_f * np.log(b / r_eff)
        I_current = circuit.current

        # Back-EMF from changing plasma inductance: V_back = (dL/dt) * I
        # Clamp to prevent numerical instability from Lp jumps at handoff
        # or from numerical diffusion artifacts on coarse grids.
        # Max physically reasonable back-EMF for any DPF: ~50 kV
        _MAX_BEMF = 50e3  # [V]
        if prev_Lp is not None and prev_Lp > 0 and dt > 0:
            dLdt_mhd = (Lp_mhd - prev_Lp) / dt
            back_emf = float(np.clip(dLdt_mhd * I_current, -_MAX_BEMF, _MAX_BEMF))
        else:
            dLdt_mhd = 0.0
            back_emf = 0.0
        prev_Lp = Lp_mhd

        # Feed MHD-computed inductance back to circuit
        coupling.Lp = Lp_mhd
        coupling.dL_dt = dLdt_mhd
        coupling = circuit.step(coupling, back_emf=back_emf, dt=dt)
        t += dt
        mhd_step += 1

        # Recalculate snap_interval after first step (now we know dt)
        if mhd_step == 1 and dt > 0:
            est_total_steps = max(1, int((t_end - t_mhd_start) / dt))
            snap_interval = max(1, est_total_steps // _target_snaps)

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        sheath_zs.append(L_anode * 1e3)  # sheath at anode tip during MHD phase
        rho_mid = state["rho"][:, ny // 2, nz // 2]
        r_grid = np.linspace(a, b, nr)
        rho_sum = np.sum(rho_mid)
        r_eff_mm = float(np.sum(rho_mid * r_grid) / rho_sum * 1e3) if rho_sum > 0 else a * 1e3
        shock_rs.append(r_eff_mm)
        phases_list.append("mhd_radial")

        rho_max_arr.append(float(np.max(state["rho"])))
        T_max_arr.append(float(np.max(state.get("Te", state["pressure"] * 3.34e-27 / (2.0 * state["rho"] * 1.380649e-23)))))
        B_max_arr.append(float(np.max(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if mhd_step % snap_interval == 0:
            _snap = {
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, ny // 2, :].copy(),
                "B_mid": state["B"][:, :, ny // 2, :].copy(),
                "P_mid": state["pressure"][:, ny // 2, :].copy(),
            }
            if "velocity" in state:
                _snap["vel_mid"] = state["velocity"][:, :, ny // 2, :].copy()
            if "Te" in state:
                _snap["Te_mid"] = state["Te"][:, ny // 2, :].copy()
            mhd_snapshots.append(_snap)

        if progress_fn and mhd_step % 5 == 0:
            _mhd_frac = min(0.3 + 0.7 * (t - t_mhd_start) / max(t_end - t_mhd_start, 1e-30), 1.0)
            progress_fn(
                max(_mhd_frac, 0.001),
                desc=f"Phase 2/2 — MHD compression: step {mhd_step} | t={t*1e6:.2f} us | dt={dt*1e9:.1f} ns",
            )

    _pbar.close()

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr)))

    # Find Lee-phase peak for snowplow dip detection
    lee_mask = [p != "mhd_radial" for p in phases_list]
    lee_I = I_arr[lee_mask] if any(lee_mask) else I_arr
    I_pre_dip = float(np.max(np.abs(lee_I)))

    # Find MHD-phase minimum for dip detection
    mhd_mask = np.array([p == "mhd_radial" for p in phases_list])
    # Find pre-dip peak (Lee phase) and dip (MHD phase) with timestamps
    lee_I_indices = [i for i, p in enumerate(phases_list) if p != "mhd_radial"]
    if lee_I_indices:
        lee_peak_idx = lee_I_indices[int(np.argmax(np.abs(I_arr[lee_I_indices])))]
        I_pre_dip = float(np.abs(I_arr[lee_peak_idx]))
        t_pre_dip = float(t_arr[lee_peak_idx])
    else:
        I_pre_dip = float(np.abs(I_arr[I_peak_idx]))
        t_pre_dip = float(t_arr[I_peak_idx])

    mhd_I_indices = [i for i, p in enumerate(phases_list) if p == "mhd_radial"]
    if mhd_I_indices:
        mhd_min_idx = mhd_I_indices[int(np.argmin(np.abs(I_arr[mhd_I_indices])))]
        I_dip = float(np.abs(I_arr[mhd_min_idx]))
        t_dip = float(t_arr[mhd_min_idx])
        dip_pct = (1 - I_dip / I_pre_dip) * 100 if I_pre_dip > 0 else 0
    else:
        I_dip = I_pre_dip
        t_dip = t_pre_dip
        dip_pct = 0.0

    result = {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "z_mm": np.array(sheath_zs), "r_mm": np.array(shock_rs),
        "phases": phases_list,
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])),
        "t_peak": float(t_arr[I_peak_idx]),
        "I_pre_dip": I_pre_dip,
        "t_pre_dip": t_pre_dip,
        "I_dip": I_dip,
        "t_dip": t_dip,
        "dip_pct": dip_pct,
        "n_steps": lee_steps + mhd_step,
        "has_snowplow": True,
        "has_mhd": True,
        "snowplow_obj": snowplow,
        "scaling": None, "crowbar_t": None,
        "dt_ns": 0,
        "handoff_time_us": handoff_time * 1e6,
        "lee_steps": lee_steps,
        "mhd_steps": mhd_step,
    }

    # Attach time-resolved yield data
    if yield_tracker is not None:
        yr = yield_tracker.get_result()
        if yr.Y_total > 0:
            result["yield_time_resolved"] = {
                "times_us": [t_v * 1e6 for t_v in yr.times],
                "dY_thermo": yr.dY_thermo,
                "dY_bt": yr.dY_bt,
                "Y_thermo_cumulative": yr.Y_thermo_cumulative,
                "Y_bt_cumulative": yr.Y_bt_cumulative,
                "T_peak_keV": yr.T_peak_keV,
                "Y_total": yr.Y_total,
                "bt_fraction": yr.bt_fraction,
                "peak_yield_time_us": yr.peak_yield_time * 1e6,
            }

    return result


def _run_metal(
    backend: str,
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, sc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
    enable_fld: bool = False,
    enable_sheath: bool = False,
    enable_ablation: bool = False,
    enable_nernst: bool = False,
    enable_cr: bool = False,
) -> dict[str, Any]:
    """Run Metal GPU MHD solver with Lee model axial rundown initialization.

    Phase 1 (Lee): Snowplow model sweeps gas along anode (0D, fast).
        Provides: circuit state (I, V), swept mass, sheath position at transition.
    Phase 2 (MHD): Metal solver takes over at radial phase onset.
        IC: compressed gas annulus with B_theta from circuit current.

    For 3D Cartesian (metal_3d), the Lee phase is skipped since the 0D
    axisymmetric snowplow model doesn't map to 3D Cartesian geometry.
    """
    import torch

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.metal.metal_solver import MetalMHDSolver

    mu_0 = 4.0 * np.pi * 1e-7
    cfg = BACKEND_CONFIGS.get(backend, BACKEND_CONFIGS["metal_plm"])

    use_mps = cfg["precision"] != "float64" and torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"

    is_3d = backend == "metal_3d"
    is_full_discharge = backend == "metal_cylindrical"
    coord_type = "cartesian" if is_3d else "cylindrical"
    solver_dx = (dr + dz) / 2.0 if is_3d else dr
    solver_dz = solver_dx if is_3d else dz

    nr, ny, nz = grid_shape

    # ---- Full-discharge cylindrical path (no Lee phase) ----
    # Starts from uniform gas fill at t=0; circuit provides B_theta from the
    # first timestep.  Uses axis BC (reflecting) on the left radial boundary.
    if is_full_discharge:
        return _run_metal_cylindrical(
            grid_shape, dr, dz, gas, rho0, p_pa,
            cc, sc, t_end, a, b, L_anode, progress_fn,
            enable_fld=enable_fld, enable_sheath=enable_sheath,
            enable_ablation=enable_ablation, enable_nernst=enable_nernst,
            enable_cr=enable_cr,
        )

    # ---- Phase 1: Lee model axial rundown (skip for 3D Cartesian) ----
    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
        crowbar_inductance=cc.get("crowbar_inductance", 0.0),
    )

    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    rho_max_arr, T_max_arr, B_max_arr = [], [], []
    sheath_zs, shock_rs, phases_list = [], [], []

    t = 0.0
    coupling = CouplingState()
    lee_steps = 0
    handoff_time = None
    snowplow = None

    fc = sc.get("current_fraction", 0.7)
    fm = sc.get("mass_fraction", 0.15)
    fmr = sc.get("radial_mass_fraction", fm)
    z_f = sc.get("pinch_column_fraction", 1.0) * L_anode

    if not is_3d:
        from dpf.fluid.snowplow import SnowplowModel

        snowplow = SnowplowModel(
            anode_radius=a, cathode_radius=b,
            fill_density=rho0,
            anode_length=L_anode,
            mass_fraction=fm,
            fill_pressure_Pa=sc.get("fill_pressure_Pa", p_pa),
            current_fraction=fc,
            radial_mass_fraction=sc.get("radial_mass_fraction"),
            pinch_column_fraction=sc.get("pinch_column_fraction", 1.0),
        )

        L_total = cc["L0"] + 1e-9
        T_LC = 2 * np.pi * np.sqrt(L_total * cc["C"])
        dt_lee = T_LC / 5000

        while t < t_end:
            sp = snowplow.step(dt_lee, circuit.current)
            coupling.Lp = sp["L_plasma"]
            coupling.dL_dt = sp["dL_dt"]
            coupling.R_plasma = sp.get("R_plasma", 0.0)
            coupling = circuit.step(coupling, back_emf=0.0, dt=dt_lee)
            t += dt_lee
            lee_steps += 1

            times.append(t * 1e6)
            currents.append(circuit.current / 1e6)
            voltages.append(circuit.voltage / 1e3)
            L_plasmas.append(coupling.Lp * 1e9)
            E_cap.append(circuit.state.energy_cap / 1e3)
            E_ind.append(circuit.state.energy_ind / 1e3)
            E_res.append(circuit.state.energy_res / 1e3)
            sheath_zs.append(sp["z_sheath"] * 1e3)
            shock_rs.append(sp["r_shock"] * 1e3)
            phases_list.append(sp["phase"])
            rho_max_arr.append(rho0)
            T_max_arr.append(300.0)
            B_max_arr.append(0.0)

            if progress_fn and lee_steps % 50 == 0:
                progress_fn(
                    min(t / t_end, 0.3),
                    desc=f"Phase 1/2 — Axial rundown: t={t*1e6:.1f} us | z={sp['z_sheath']*1e3:.0f} mm",
                )

            if sp["phase"] == "radial":
                handoff_time = t
                break

        if handoff_time is None:
            # Never reached radial phase — return Lee-only results
            logger.warning("Metal+Lee: Lee model didn't reach radial phase in %.1f us", t_end * 1e6)
            t_arr = np.array(times)
            I_arr = np.array(currents)
            I_peak_idx = int(np.argmax(np.abs(I_arr)))
            return {
                "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
                "L_p_nH": np.array(L_plasmas),
                "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
                "E_res_kJ": np.array(E_res),
                "z_mm": np.array(sheath_zs), "r_mm": np.array(shock_rs),
                "phases": phases_list,
                "I_peak": float(np.abs(I_arr[I_peak_idx])),
                "t_peak": float(t_arr[I_peak_idx]),
                "n_steps": lee_steps,
                "has_snowplow": True, "has_mhd": False,
                "mhd_snapshots": [], "final_state": None,
                "dip_pct": 0.0, "I_pre_dip": float(np.abs(I_arr[I_peak_idx])),
                "I_dip": 0.0, "t_dip": 0.0,
                "scaling": None, "crowbar_t": None,
                "snowplow_obj": snowplow, "dt_ns": dt_lee * 1e9,
                "rho_max": np.array(rho_max_arr),
                "T_max": np.array(T_max_arr),
                "B_max": np.array(B_max_arr),
            }

    # ---- Phase 2: MHD radial implosion (Metal solver) ----
    I_handoff = circuit.current

    # MHD domain: radial extent = cathode - anode, axial = z_f
    dr_mhd = (b - a) / nr
    dz_mhd = z_f / max(nz, 1)
    solver_dx_mhd = (dr_mhd + dz_mhd) / 2.0 if is_3d else dr_mhd
    solver_dz_mhd = solver_dx_mhd if is_3d else dz_mhd

    _cyl_kwargs = {"r_inner": a, "convert_b_si_to_hl": True} if not is_3d else {}
    solver = MetalMHDSolver(
        grid_shape=grid_shape, dx=solver_dx_mhd, dz=solver_dz_mhd,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3, device=device,
        use_ct=False,
        coordinates=coord_type,
        ion_mass=gas["m_mol"],
        **_cyl_kwargs,
        **cfg,
    )

    if is_3d:
        # 3D: uniform IC with azimuthal perturbation (no Lee phase)
        rho_ic = np.full((nr, ny, nz), rho0)
        x = (np.arange(nr) - nr / 2.0 + 0.5) * solver_dx_mhd
        y = (np.arange(ny) - ny / 2.0 + 0.5) * solver_dx_mhd
        X, Y = np.meshgrid(x, y, indexing="ij")
        theta = np.arctan2(Y, X)
        for m in (1, 4):
            pert = 0.01 * rho0 * np.cos(m * theta)  # EMPIRICAL: 1% amplitude
            rho_ic += pert[:, :, np.newaxis]
        rho_ic = np.maximum(rho_ic, rho0 * 0.01)

        state = {
            "rho": rho_ic,
            "velocity": np.zeros((3, nr, ny, nz)),
            "pressure": np.full((nr, ny, nz), p_pa),
            "B": np.zeros((3, nr, ny, nz)),
            "Te": np.full((nr, ny, nz), 300.0),
            "Ti": np.full((nr, ny, nz), 300.0),
            "psi": np.zeros((nr, ny, nz)),
        }
    else:
        # Cylindrical: build physically motivated IC from Lee handoff state.
        # Swept mass concentrated near cathode (outer boundary), unswept gas inside,
        # B_theta = mu_0 * fc * I / (2*pi*r) from circuit current (magnetic piston).
        r_cells = np.linspace(a + dr_mhd * 0.5, b - dr_mhd * 0.5, nr)

        rho_bg = rho0 * (1.0 - fmr)
        n_sheath = max(int(0.2 * nr), 2)
        rho_mhd = np.full((nr, ny, nz), rho_bg)
        shell_vol = sum(
            2.0 * np.pi * r_cells[nr - n_sheath + i] * dr_mhd * dz_mhd
            for i in range(n_sheath)
        )
        swept_mass_per_z = fmr * rho0 * np.pi * (b**2 - a**2)
        rho_sheath = swept_mass_per_z * dz_mhd / max(shell_vol, 1e-30)
        rho_mhd[nr - n_sheath:, :, :] = max(rho_sheath, rho_bg * 2.0)

        B_theta_1d = mu_0 * fc * I_handoff / (2.0 * np.pi * r_cells)
        B_mhd = np.zeros((3, nr, ny, nz))
        B_mhd[1] = B_theta_1d[:, np.newaxis, np.newaxis]

        state = {
            "rho": rho_mhd,
            "velocity": np.zeros((3, nr, ny, nz)),
            "pressure": np.full((nr, ny, nz), p_pa),
            "B": B_mhd,
            "Te": np.full((nr, ny, nz), 300.0),
            "Ti": np.full((nr, ny, nz), 300.0),
            "psi": np.zeros((nr, ny, nz)),
        }

    # Continue from handoff state
    rho_max_arr.append(float(np.max(state["rho"])))
    T_max_arr.append(300.0)
    B_max_arr.append(float(np.max(np.abs(state.get("B", np.zeros(1))))))
    mhd_snapshots = []

    t_mhd_start = t
    mhd_step = 0
    prev_Lp = coupling.Lp if not is_3d else 0.0
    _MAX_BEMF = 50e3  # [V]
    _target_snaps = 30
    snap_interval = 3
    _cr_fracs = None

    # Time-resolved yield tracker
    yield_tracker = None
    try:
        from dpf.diagnostics.yield_tracker import YieldTracker
        yield_tracker = YieldTracker(ion_mass=gas["m_mol"], rho0=rho0)
    except ImportError:
        pass

    # r_cells for Lp computation (cylindrical only)
    if not is_3d:
        r_cells_lp = np.linspace(a + dr_mhd * 0.5, b - dr_mhd * 0.5, nr)

    while t < t_end:
        dt_mhd = solver.compute_dt(state)
        dt = min(dt_mhd, t_end - t)
        if dt <= 0:
            break

        state = solver.step(
            state, dt, current=circuit.current, voltage=circuit.voltage,
            anode_radius=a, cathode_radius=b, apply_electrode_bc=True,
        )

        # Radiation cooling: improved model (T-dependent Gaunt + cyclotron)
        if "Te" in state:
            try:
                from dpf.radiation.improved_radiation import apply_improved_radiation_losses
                rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
                ne = rho_safe / gas["m_mol"]
                Z_eff = gas.get("Z", 1)
                B_field = state.get("B")
                B_mag = np.sqrt(np.sum(B_field**2, axis=0)) if B_field is not None else None
                state["Te"], _ = apply_improved_radiation_losses(
                    state["Te"], ne, dt, Z=Z_eff, B_mag=B_mag,
                )
            except ImportError:
                try:
                    from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses
                    rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
                    ne = rho_safe / gas["m_mol"]
                    state["Te"], _ = apply_bremsstrahlung_losses(state["Te"], ne, dt, Z=gas.get("Z", 1))
                except ImportError:
                    pass

            # Line radiation cooling (impurity species)
            try:
                from dpf.radiation.line_radiation import apply_line_radiation_losses
                rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
                ne_lr = rho_safe / gas["m_mol"]
                state["Te"], _ = apply_line_radiation_losses(
                    state["Te"], ne_lr, dt, Z_impurity=29,
                    impurity_fraction=0.001,  # EMPIRICAL: 0.1% Cu from electrode
                )
            except (ImportError, Exception):
                pass

        # Advanced physics operator-split (FLD, sheath, ablation, Nernst, CR)
        any_adv = enable_fld or enable_sheath or enable_ablation or enable_nernst or enable_cr
        if any_adv:
            adv_dx = (dr_mhd + dz_mhd) / 2.0 if is_3d else dr_mhd
            adv_dz = adv_dx if is_3d else dz_mhd
            state, _cr_fracs = _apply_advanced_physics(
                state, dt, gas, adv_dx, adv_dz, a, b,
                enable_fld=enable_fld, enable_sheath=enable_sheath,
                enable_ablation=enable_ablation, enable_nernst=enable_nernst,
                enable_cr=enable_cr, cr_fractions=_cr_fracs,
            )

        # Time-resolved yield accumulation
        if yield_tracker is not None:
            V_p = abs(coupling.dL_dt) * abs(circuit.current) if coupling.dL_dt else 0.0
            cell_vol = dr_mhd * dr_mhd * dz_mhd if is_3d else dr_mhd * (b - a) / nr * dz_mhd
            yield_tracker.accumulate(
                state, dt,
                I_current=circuit.current, V_pinch=V_p,
                cell_volume=cell_vol,
            )

        # Compute L_plasma from density profile (Lee-model formula)
        if not is_3d:
            rho_mid = state["rho"][:, ny // 2, nz // 2]
            rho_bg_lp = rho0 * (1.0 - fmr)
            weights = np.maximum(rho_mid - rho_bg_lp, 0.0)
            w_sum = float(np.sum(weights))
            if w_sum > 0:
                r_eff = float(np.sum(r_cells_lp * weights)) / w_sum
            else:
                r_eff = b
            r_eff = max(r_eff, a * 0.01)
            L_axial_frozen = coupling.Lp if mhd_step == 0 else (mu_0 / (2.0 * np.pi)) * np.log(b / a) * sc.get("anode_length", L_anode)
            Lp_mhd = L_axial_frozen + (mu_0 / (2.0 * np.pi)) * z_f * np.log(b / r_eff)
        else:
            mid = nr // 2
            rho_line = state["rho"][:, mid, nz // 2]
            r_vals_3d = np.linspace(-b + solver_dx_mhd * 0.5, b - solver_dx_mhd * 0.5, nr)
            r_abs = np.abs(r_vals_3d)
            weights = np.maximum(rho_line - rho0, 0.0)
            w_sum = float(np.sum(weights))
            r_eff = float(np.sum(r_abs * weights)) / w_sum if w_sum > 0 else b
            r_eff = max(r_eff, a * 0.01)
            Lp_mhd = (mu_0 / (2.0 * np.pi)) * L_anode * np.log(b / r_eff)
        I_current = circuit.current

        # Back-EMF from changing plasma inductance
        if prev_Lp is not None and prev_Lp > 0 and dt > 0:
            dLdt_mhd = (Lp_mhd - prev_Lp) / dt
            back_emf = float(np.clip(dLdt_mhd * I_current, -_MAX_BEMF, _MAX_BEMF))
        else:
            dLdt_mhd = 0.0
            back_emf = 0.0
        prev_Lp = Lp_mhd

        coupling.Lp = Lp_mhd
        coupling.dL_dt = dLdt_mhd
        coupling = circuit.step(coupling, back_emf=back_emf, dt=dt)
        t += dt
        mhd_step += 1

        if mhd_step == 1 and dt > 0:
            est_total = max(1, int((t_end - t_mhd_start) / dt))
            snap_interval = max(1, est_total // _target_snaps)

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)

        if not is_3d:
            # Use Lee z_sheath for axial phase, frozen for MHD phase
            sheath_zs.append(sp["z_sheath"] * 1e3)
            # Compute effective compression radius from MHD density
            rho_mid_r = state["rho"][:, ny // 2, nz // 2]
            r_grid = np.linspace(a, b, nr)
            rho_sum = np.sum(rho_mid_r)
            r_eff_mm = float(np.sum(rho_mid_r * r_grid) / rho_sum * 1e3) if rho_sum > 0 else a * 1e3
            shock_rs.append(r_eff_mm)
        else:
            sheath_zs.append(L_anode * 1e3)
            shock_rs.append(0.0)
        phases_list.append("mhd_radial")

        rho_max_arr.append(float(np.max(state["rho"])))
        T_max_arr.append(float(np.max(state.get("Te", state["pressure"] * 3.34e-27 / (2.0 * state["rho"] * 1.380649e-23)))))
        B_max_arr.append(float(np.max(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if mhd_step % snap_interval == 0:
            _snap = {
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, ny // 2, :].copy(),
                "B_mid": state["B"][:, :, ny // 2, :].copy(),
                "P_mid": state["pressure"][:, ny // 2, :].copy(),
            }
            if "velocity" in state:
                _snap["vel_mid"] = state["velocity"][:, :, ny // 2, :].copy()
            if "Te" in state:
                _snap["Te_mid"] = state["Te"][:, ny // 2, :].copy()
            mhd_snapshots.append(_snap)

        if progress_fn and mhd_step % 5 == 0:
            _mhd_frac = min(0.3 + 0.7 * (t - t_mhd_start) / max(t_end - t_mhd_start, 1e-30), 1.0)
            progress_fn(
                max(_mhd_frac, 0.001),
                desc=f"Phase 2/2 — MHD compression: step {mhd_step} | t={t*1e6:.2f} us | dt={dt*1e9:.1f} ns",
            )

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr))) if len(I_arr) > 0 else 0

    # Dip detection: Lee-phase peak vs MHD-phase minimum
    lee_I_indices = [i for i, p in enumerate(phases_list) if p != "mhd_radial"]
    if lee_I_indices:
        lee_peak_idx = lee_I_indices[int(np.argmax(np.abs(I_arr[lee_I_indices])))]
        I_pre_dip = float(np.abs(I_arr[lee_peak_idx]))
        t_pre_dip = float(t_arr[lee_peak_idx])
    else:
        I_pre_dip = float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0.0
        t_pre_dip = float(t_arr[I_peak_idx]) if len(t_arr) > 0 else 0.0

    mhd_I_indices = [i for i, p in enumerate(phases_list) if p == "mhd_radial"]
    if mhd_I_indices:
        mhd_min_idx = mhd_I_indices[int(np.argmin(np.abs(I_arr[mhd_I_indices])))]
        I_dip = float(np.abs(I_arr[mhd_min_idx]))
        t_dip = float(t_arr[mhd_min_idx])
        dip_pct = (1 - I_dip / I_pre_dip) * 100 if I_pre_dip > 0 else 0
    else:
        I_dip = I_pre_dip
        t_dip = t_pre_dip
        dip_pct = 0.0

    result = {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0,
        "t_peak": float(t_arr[I_peak_idx]) if len(t_arr) > 0 else 0,
        "n_steps": lee_steps + mhd_step,
        "has_snowplow": not is_3d,
        "has_mhd": True,
        "phases": phases_list,
        "z_mm": np.array(sheath_zs) if sheath_zs else np.full(len(times), L_anode * 1e3),
        "r_mm": np.array(shock_rs) if shock_rs else np.zeros(len(times)),
        "I_pre_dip": I_pre_dip,
        "t_pre_dip": t_pre_dip if not is_3d else 0.0,
        "I_dip": I_dip,
        "t_dip": t_dip,
        "dip_pct": dip_pct,
        "scaling": None, "crowbar_t": None,
        "snowplow_obj": snowplow, "dt_ns": 0,
    }

    if not is_3d and handoff_time is not None:
        result["handoff_time_us"] = handoff_time * 1e6
        result["lee_steps"] = lee_steps
        result["mhd_steps"] = mhd_step

    # Attach time-resolved yield data
    if yield_tracker is not None:
        yr = yield_tracker.get_result()
        if yr.Y_total > 0:
            result["yield_time_resolved"] = {
                "times_us": [t_v * 1e6 for t_v in yr.times],
                "dY_thermo": yr.dY_thermo,
                "dY_bt": yr.dY_bt,
                "Y_thermo_cumulative": yr.Y_thermo_cumulative,
                "Y_bt_cumulative": yr.Y_bt_cumulative,
                "T_peak_keV": yr.T_peak_keV,
                "Y_total": yr.Y_total,
                "bt_fraction": yr.bt_fraction,
                "peak_yield_time_us": yr.peak_yield_time * 1e6,
            }

    return result


def _detect_plasma_phase(
    state: dict,
    a: float,
    b: float,
    rho0: float,
    nr: int,
    nz: int,
    dr: float,
    dz: float,
) -> str:
    """Classify current MHD phase from density distribution.

    Returns one of: "rundown", "radial", "pinch".

    Heuristic rules:
    - If peak density is within the outer 30% radially -> still rundown/uniform
    - If density-weighted mean radius is advancing axially (not yet compressed)
      but B_max < 10x initial -> rundown
    - If density-weighted mean radius < 0.5*(a+b)/2 -> radial compression
    - If density-weighted mean radius < 2*a -> pinch
    """
    rho = state["rho"]
    rho_mid = rho[:, 0, nz // 2]
    weights = np.maximum(rho_mid - rho0 * 0.5, 0.0)
    w_sum = float(np.sum(weights))
    r_grid = (np.arange(nr) + 0.5) * dr + a
    if w_sum > 0:
        r_eff = float(np.sum(r_grid * weights)) / w_sum
    else:
        r_eff = b
    r_mid = 0.5 * (a + b)
    if r_eff > 0.7 * b:
        return "rundown"
    if r_eff > r_mid * 0.4:
        return "radial"
    return "pinch"


def _run_metal_cylindrical(
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, sc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
    enable_fld: bool = False,
    enable_sheath: bool = False,
    enable_ablation: bool = False,
    enable_nernst: bool = False,
    enable_cr: bool = False,
) -> dict[str, Any]:
    """Full-discharge cylindrical MHD from t=0 — no Lee model phase.

    The simulation domain covers the inter-electrode gap [a, b] radially
    and [0, L_anode] axially.  Initial conditions are a uniform gas fill
    at the specified pressure; B=0 at t=0.  The circuit starts from V0 and
    discharges through the plasma, providing B_theta = mu0*I/(2*pi*r) at
    the electrode walls each timestep.

    The axis (r=0 face) of the domain is at r=a (inner conductor / anode).
    Because the grid starts at the anode, not at r=0, no axis BC is needed
    for this domain — the left boundary is the anode conducting wall.

    This backend is intended for studying the full DPF discharge cycle from
    initial breakdown through axial rundown and radial implosion in a single
    coupled simulation.
    """
    import torch

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.metal.metal_solver import MetalMHDSolver

    mu_0 = 4.0 * np.pi * 1e-7
    cfg = BACKEND_CONFIGS.get("metal_cylindrical", BACKEND_CONFIGS["metal_plm"])

    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"

    nr, ny, nz = grid_shape

    # Grid: radial extent [a, b], axial extent [0, L_anode]
    dr_mhd = (b - a) / nr
    dz_mhd = L_anode / max(nz, 1)

    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
        crowbar_inductance=cc.get("crowbar_inductance", 0.0),
    )

    # Boundary conditions: outflow on left (anode wall) and right (cathode wall)
    # for the MHD solver — electrode B/v BCs are applied via apply_electrode_bc.
    # Axis is at r=a (inner conductor), so left BC is a conducting wall, not a
    # true axis.  Use "outflow" here; electrode BC enforces v_r=B_r=0 at walls.
    solver = MetalMHDSolver(
        grid_shape=(nr, ny, nz),
        dx=dr_mhd, dz=dz_mhd,
        gamma=gas.get("gamma", 5.0 / 3.0),
        cfl=0.3, device=device,
        use_ct=False,
        coordinates="cylindrical",
        ion_mass=gas["m_mol"],
        bc=("outflow", "outflow", "outflow"),
        r_inner=a,
        convert_b_si_to_hl=True,
        **cfg,
    )

    # Physical r coordinate for each cell centre: r = a + (i+0.5)*dr
    r_phys = a + (np.arange(nr) + 0.5) * dr_mhd  # shape (nr,)
    z_phys = (np.arange(nz) + 0.5) * dz_mhd  # shape (nz,)

    # Initial conditions: uniform gas fill with current-sheet B_theta near z=0.
    # The circuit breaks down in ~100 ns — seed current: I_seed = V0 * 100ns / L0.
    dt_seed = 1e-7  # 100 ns breakdown time  # EMPIRICAL
    I_seed = cc["V0"] * dt_seed / cc["L0"]
    B_theta_vac = mu_0 * I_seed / (2.0 * np.pi * r_phys)  # shape (nr,)
    # Current sheet at z=0: B_theta decays exponentially from insulator face.
    # Sheet thickness ~ 3 cells, consistent with Paschen breakdown channel.
    z_decay = 3.0 * dz_mhd  # EMPIRICAL
    z_profile = np.exp(-z_phys / z_decay)  # shape (nz,)
    B_theta_2d = B_theta_vac[:, np.newaxis] * z_profile[np.newaxis, :]  # (nr, nz)
    B_init = np.zeros((3, nr, ny, nz))
    B_init[1] = B_theta_2d[:, np.newaxis, :]  # (nr, 1, nz)

    state: dict[str, np.ndarray] = {
        "rho": np.full((nr, ny, nz), rho0),
        "velocity": np.zeros((3, nr, ny, nz)),
        "pressure": np.full((nr, ny, nz), p_pa),
        "B": B_init,
        "Te": np.full((nr, ny, nz), 300.0),
        "Ti": np.full((nr, ny, nz), 300.0),
        "psi": np.zeros((nr, ny, nz)),
    }

    # Anomalous resistivity for initial breakdown — allows B_theta to diffuse
    # into the plasma from electrode BCs.  Capped to satisfy resistive CFL:
    #   eta_max = dx^2 * mu_0 / (4 * dt_mhd)   (factor of 4 for safety)
    # Decreases with temperature as plasma ionizes (Spitzer-like scaling).
    _ETA_ANOMALOUS = 1e-4  # Ohm*m — typical weakly-ionized DPF gas  # EMPIRICAL

    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    rho_max_arr, T_max_arr, B_max_arr = [], [], []
    sheath_zs: list[float] = []
    shock_rs: list[float] = []
    phases_list: list[str] = []
    mhd_snapshots = []

    t = 0.0
    coupling = CouplingState()
    mhd_step = 0
    prev_Lp: float | None = None
    _MAX_BEMF = 50e3
    _target_snaps = 30
    snap_interval = 3
    _cr_fracs = None

    r_cells_lp = r_phys  # physical radii: a + (i+0.5)*dr

    yield_tracker = None
    try:
        from dpf.diagnostics.yield_tracker import YieldTracker
        yield_tracker = YieldTracker(ion_mass=gas["m_mol"], rho0=rho0)
    except ImportError:
        pass

    t_start = t

    # Maximum timestep: resolve circuit dynamics (T/4 ≈ 5 us → dt_max = 50 ns)
    _DT_MAX = 5e-8  # 50 ns  # EMPIRICAL
    # Safety limits — full-discharge is expensive; prevent runaway
    _MAX_STEPS = 500_000  # hard cap: avoid running for hours
    _MAX_WALL_SECONDS = 600  # 10 minutes wall-clock timeout
    _wall_start = wall_time.time()

    # ---- GPU-resident hot loop ----
    # Keep state on GPU as PyTorch tensors for the entire simulation.
    # Only convert to NumPy for: snapshots (every N steps), radiation (if enabled),
    # advanced physics (if enabled), and yield tracking (every 50 steps).
    # This eliminates ~99% of CPU↔GPU transfers vs the old per-step bounce.
    state_gpu = solver._to_device(state)
    r_cells_gpu = torch.as_tensor(r_cells_lp, device=solver.device, dtype=solver._dtype)
    fc = sc.get("current_fraction", 0.7)
    _DIAG_INTERVAL = 50  # run expensive diagnostics every N steps
    _RAD_INTERVAL = 10   # radiation cooling every N steps (operator-split OK)

    while t < t_end:
        # Safety: hard step cap and wall-clock timeout
        if mhd_step >= _MAX_STEPS:
            logger.warning("Cylindrical MHD hit %d step limit at t=%.3e s (%.1f%% of sim time)",
                           _MAX_STEPS, t, t / t_end * 100)
            break
        if mhd_step % 1000 == 0 and (wall_time.time() - _wall_start) > _MAX_WALL_SECONDS:
            logger.warning("Cylindrical MHD wall-clock timeout (%ds) at step %d, t=%.3e s",
                           _MAX_WALL_SECONDS, mhd_step, t)
            break

        dt_mhd = solver.compute_dt_gpu(state_gpu)
        dt = min(dt_mhd, _DT_MAX, t_end - t)
        if dt <= 0:
            break

        # Compute eta on GPU (no NumPy roundtrip)
        Te_gpu = state_gpu.get("Te", torch.full((nr, ny, nz), 300.0, device=solver.device, dtype=solver._dtype))
        Te_eV_gpu = torch.clamp(Te_gpu, min=300.0) / 11604.5
        eta_gpu = torch.clamp(5.2e-5 * 10.0 / torch.pow(Te_eV_gpu, 1.5), max=_ETA_ANOMALOUS)

        # MHD step — stays on GPU
        state_gpu = solver.step_gpu(
            state_gpu, dt,
            current=fc * circuit.current, voltage=circuit.voltage,
            anode_radius=a, cathode_radius=b, apply_electrode_bc=True,
            eta_field_gpu=eta_gpu,
        )

        # Update Te from pressure on GPU
        rho_safe_gpu = torch.clamp(state_gpu["rho"], min=1e-10)
        state_gpu["Te"] = state_gpu["pressure"] * gas["m_mol"] / (2.0 * rho_safe_gpu * 1.380649e-23)

        # Radiation cooling (requires NumPy — run every _RAD_INTERVAL steps)
        if mhd_step % _RAD_INTERVAL == 0:
            try:
                from dpf.radiation.improved_radiation import apply_improved_radiation_losses
                Te_np = state_gpu["Te"].detach().cpu().to(torch.float64).numpy()
                rho_np = state_gpu["rho"].detach().cpu().to(torch.float64).numpy()
                rho_safe_np = np.where(rho_np > 0, rho_np, 1.0)
                ne_np = rho_safe_np / gas["m_mol"]
                B_np = state_gpu["B"].detach().cpu().to(torch.float64).numpy()
                B_mag_np = np.sqrt(np.sum(B_np**2, axis=0))
                Te_np, _ = apply_improved_radiation_losses(
                    Te_np, ne_np, dt * _RAD_INTERVAL, Z=gas.get("Z", 1), B_mag=B_mag_np,
                )
                state_gpu["Te"] = torch.as_tensor(Te_np, dtype=solver._dtype).to(solver.device)
            except ImportError:
                try:
                    from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses
                    Te_np = state_gpu["Te"].detach().cpu().to(torch.float64).numpy()
                    rho_np = state_gpu["rho"].detach().cpu().to(torch.float64).numpy()
                    ne_np = np.where(rho_np > 0, rho_np, 1.0) / gas["m_mol"]
                    Te_np, _ = apply_bremsstrahlung_losses(Te_np, ne_np, dt * _RAD_INTERVAL, Z=gas.get("Z", 1))
                    state_gpu["Te"] = torch.as_tensor(Te_np, dtype=solver._dtype).to(solver.device)
                except ImportError:
                    pass

        # Advanced physics (requires NumPy — only when flags enabled)
        any_adv = enable_fld or enable_sheath or enable_ablation or enable_nernst or enable_cr
        if any_adv:
            state_np = solver._to_numpy(state_gpu)
            state_np, _cr_fracs = _apply_advanced_physics(
                state_np, dt, gas, dr_mhd, dz_mhd, a, b,
                enable_fld=enable_fld, enable_sheath=enable_sheath,
                enable_ablation=enable_ablation, enable_nernst=enable_nernst,
                enable_cr=enable_cr, cr_fractions=_cr_fracs,
            )
            state_gpu = solver._to_device(state_np)

        # Yield tracking (NumPy, deferred every _DIAG_INTERVAL steps)
        if yield_tracker is not None and mhd_step % _DIAG_INTERVAL == 0:
            state_np_yield = solver._to_numpy(state_gpu)
            V_p = abs(coupling.dL_dt) * abs(circuit.current) if coupling.dL_dt else 0.0
            cell_vol = dr_mhd * (b - a) / nr * dz_mhd
            yield_tracker.accumulate(
                state_np_yield, dt * _DIAG_INTERVAL,
                I_current=circuit.current, V_pinch=V_p,
                cell_volume=cell_vol,
            )

        # z_sheath and r_eff on GPU (no transfer)
        rho_z_avg = state_gpu["rho"][:, ny // 2, :].mean(dim=0)
        rho_z_max = float(rho_z_avg.max().item())
        if rho_z_max > 1.1 * rho0:
            z_sheath_idx = int(rho_z_avg.argmax().item())
            z_sheath = float((z_sheath_idx + 1) * dz_mhd)
        else:
            z_sheath = dz_mhd
        z_sheath = min(z_sheath, L_anode)

        rho_mid_gpu = state_gpu["rho"][:, ny // 2, nz // 2]
        weights_gpu = torch.clamp(rho_mid_gpu - rho0 * 0.5, min=0.0)
        w_sum = float(weights_gpu.sum().item())
        if w_sum > 0:
            r_eff = float((r_cells_gpu * weights_gpu).sum().item()) / w_sum
        else:
            r_eff = b
        r_eff = max(r_eff, a * 0.01)

        import math as _math
        Lp_axial = (mu_0 / (2.0 * _math.pi)) * _math.log(b / a) * z_sheath
        Lp_radial = (mu_0 / (2.0 * _math.pi)) * z_sheath * _math.log(b / max(r_eff, a))
        Lp_mhd = fc * (Lp_axial + Lp_radial)
        if prev_Lp is not None and Lp_mhd < prev_Lp:
            Lp_mhd = prev_Lp

        if prev_Lp is not None and prev_Lp > 0 and dt > 0:
            dLdt_mhd = (Lp_mhd - prev_Lp) / dt
            back_emf = max(-_MAX_BEMF, min(_MAX_BEMF, dLdt_mhd * circuit.current))
        else:
            dLdt_mhd = 0.0
            back_emf = 0.0
        prev_Lp = Lp_mhd

        coupling.Lp = Lp_mhd
        coupling.dL_dt = dLdt_mhd
        coupling = circuit.step(coupling, back_emf=back_emf, dt=dt)
        t += dt
        mhd_step += 1

        if mhd_step == 1 and dt > 0:
            est_total = max(1, int((t_end - t_start) / dt))
            snap_interval = max(1, est_total // _target_snaps)

        # Scalar diagnostics (cheap GPU reads — single .item() calls)
        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)

        rho_mid_r_gpu = state_gpu["rho"][:, ny // 2, nz // 2]
        rho_sum_s = float(rho_mid_r_gpu.sum().item())
        r_eff_mm = float((rho_mid_r_gpu * r_cells_gpu).sum().item()) / rho_sum_s * 1e3 if rho_sum_s > 0 else a * 1e3
        sheath_zs.append(L_anode * 1e3)
        shock_rs.append(r_eff_mm)

        phase = "mhd_cylindrical"  # phase detection deferred to reduce overhead
        phases_list.append(phase)

        rho_max_arr.append(float(state_gpu["rho"].max().item()))
        Te_max_val = float(state_gpu["Te"].max().item()) if "Te" in state_gpu else 300.0
        T_max_arr.append(Te_max_val)
        B_sq = (state_gpu["B"] ** 2).sum(dim=0)
        B_max_arr.append(float(B_sq.sqrt().max().item()))

        if mhd_step % snap_interval == 0:
            _snap = {
                "t_us": t * 1e6,
                "rho_mid": state_gpu["rho"][:, ny // 2, :].detach().cpu().to(torch.float64).numpy().copy(),
                "B_mid": state_gpu["B"][:, :, ny // 2, :].detach().cpu().to(torch.float64).numpy().copy(),
                "P_mid": state_gpu["pressure"][:, ny // 2, :].detach().cpu().to(torch.float64).numpy().copy(),
            }
            if "velocity" in state_gpu:
                _snap["vel_mid"] = state_gpu["velocity"][:, :, ny // 2, :].detach().cpu().to(torch.float64).numpy().copy()
            if "Te" in state_gpu:
                _snap["Te_mid"] = state_gpu["Te"][:, ny // 2, :].detach().cpu().to(torch.float64).numpy().copy()
            mhd_snapshots.append(_snap)

        if progress_fn and mhd_step % 5 == 0:
            t_frac = min((t - t_start) / max(t_end - t_start, 1e-30), 1.0)
            progress_fn(
                max(t_frac, 0.001),
                desc=f"MHD: step {mhd_step} | t={t*1e6:.2f}/{t_end*1e6:.0f} us ({t_frac*100:.0f}%) | dt={dt*1e9:.1f} ns | {phase}",
            )

    _wall_elapsed = wall_time.time() - _wall_start
    _incomplete = (mhd_step >= _MAX_STEPS) or (_wall_elapsed > _MAX_WALL_SECONDS)

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr))) if len(I_arr) > 0 else 0

    result: dict[str, Any] = {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "mhd_snapshots": mhd_snapshots,
        "final_state": solver._to_numpy(state_gpu),
        "I_peak": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0.0,
        "t_peak": float(t_arr[I_peak_idx]) if len(t_arr) > 0 else 0.0,
        "I_pre_dip": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0.0,
        "t_pre_dip": 0.0,
        "I_dip": 0.0,
        "t_dip": 0.0,
        "dip_pct": 0.0,
        "n_steps": mhd_step,
        "wall_time_s": _wall_elapsed,
        "incomplete": _incomplete,
        "incomplete_reason": (
            f"Hit {_MAX_STEPS} step limit at t={t*1e6:.2f}us ({t/t_end*100:.1f}% of sim time). "
            f"CFL timestep too small (dt={dt:.2e}s). Try: coarser grid or shorter sim_time."
            if mhd_step >= _MAX_STEPS else
            f"Wall-clock timeout ({_MAX_WALL_SECONDS}s) at step {mhd_step}, t={t*1e6:.2f}us. "
            f"Try: coarser grid or Hybrid backend."
            if _incomplete else ""
        ),
        "has_snowplow": False,
        "has_mhd": True,
        "phases": phases_list,
        "z_mm": np.array(sheath_zs) if sheath_zs else np.full(len(times), L_anode * 1e3),
        "r_mm": np.array(shock_rs) if shock_rs else np.zeros(len(times)),
        "scaling": None, "crowbar_t": None,
        "snowplow_obj": None, "dt_ns": 0,
        "backend": "metal_cylindrical",
    }

    if yield_tracker is not None:
        yr = yield_tracker.get_result()
        if yr.Y_total > 0:
            result["yield_time_resolved"] = {
                "times_us": [t_v * 1e6 for t_v in yr.times],
                "dY_thermo": yr.dY_thermo,
                "dY_bt": yr.dY_bt,
                "Y_thermo_cumulative": yr.Y_thermo_cumulative,
                "Y_bt_cumulative": yr.Y_bt_cumulative,
                "T_peak_keV": yr.T_peak_keV,
                "Y_total": yr.Y_total,
                "bt_fraction": yr.bt_fraction,
                "peak_yield_time_us": yr.peak_yield_time * 1e6,
            }

    return result


_ATHENA_STEP_TIMEOUT_S = 30


def _run_athena(
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, sc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
) -> dict[str, Any]:
    """Run Athena++ C++ MHD solver via subprocess mode."""
    import concurrent.futures
    from pathlib import Path

    from dpf.athena_wrapper import AthenaPPSolver
    from dpf.config import (
        CircuitConfig,
        DiagnosticsConfig,
        FluidConfig,
        GeometryConfig,
        SimulationConfig,
        SnowplowConfig,
    )

    nr, ny, nz = grid_shape

    circuit_cfg = CircuitConfig(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
        crowbar_inductance=cc.get("crowbar_inductance", 0.0),
    )

    sim_cfg = SimulationConfig(
        grid_shape=[nr, 1, nz],
        dx=dr,
        sim_time=t_end,
        rho0=rho0,
        T0=300.0,
        ion_mass=gas["m_mol"],
        circuit=circuit_cfg,
        geometry=GeometryConfig(type="cylindrical", dz=dz),
        fluid=FluidConfig(
            backend="athena",
            reconstruction="plm",
            riemann_solver="hlld",
            gamma=gas.get("gamma", 5 / 3),
            cfl=0.3,
            time_integrator="ssp_rk2",
        ),
        snowplow=SnowplowConfig(
            enabled=True,
            fill_pressure_Pa=p_pa,
            anode_length=L_anode,
            current_fraction=sc.get("current_fraction", 0.7),
            mass_fraction=sc.get("mass_fraction", 0.15),
        ),
        diagnostics=DiagnosticsConfig(hdf5_filename=":memory:"),
    )

    athena_bin = str(Path(__file__).resolve().parent / "external" / "athena" / "bin" / "athena_cylindrical")
    solver = AthenaPPSolver(sim_cfg, athena_binary=athena_bin, use_subprocess=True)

    state = solver.initial_state()
    t = 0.0
    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    rho_max_arr, T_max_arr, B_max_arr = [], [], []
    mhd_snapshots = []

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState

    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
        crowbar_inductance=cc.get("crowbar_inductance", 0.0),
    )
    coupling = CouplingState()

    step = 0
    _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
      while t < t_end:
        dt = solver._compute_dt(state)
        dt = min(dt, t_end - t)
        if dt <= 0:
            break

        _fut = _pool.submit(
            solver.step, state, dt,
            current=circuit.current, voltage=circuit.voltage,
        )
        try:
            state = _fut.result(timeout=_ATHENA_STEP_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.error(
                "Athena++ step timed out after %ds at t=%.3e s — aborting",
                _ATHENA_STEP_TIMEOUT_S, t,
            )
            break
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt)
        t += dt
        step += 1

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        rho_max_arr.append(float(np.max(state["rho"])))
        T_max_arr.append(float(np.max(state.get("Te", state["pressure"] * 3.34e-27 / (2.0 * state["rho"] * 1.380649e-23)))))
        B_max_arr.append(float(np.max(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if step % 80 == 0:
            _snap = {
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, 0, :].copy(),
                "P_mid": state["pressure"][:, 0, :].copy(),
            }
            if "B" in state:
                _snap["B_mid"] = state["B"][:, :, 0, :].copy()
            if "velocity" in state:
                _snap["vel_mid"] = state["velocity"][:, :, 0, :].copy()
            if "Te" in state:
                _snap["Te_mid"] = state["Te"][:, 0, :].copy()
            mhd_snapshots.append(_snap)

        if progress_fn and step % 20 == 0:
            progress_fn(min(t / t_end, 1.0), desc=f"Athena++ t={t*1e6:.1f}us, step={step}")
    finally:
      _pool.shutdown(wait=False)

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr))) if len(I_arr) > 0 else 0

    return {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0,
        "t_peak": float(t_arr[I_peak_idx]) if len(t_arr) > 0 else 0,
        "n_steps": step,
        "has_snowplow": False,
        "has_mhd": True,
        "phases": ["mhd"] * len(times),
        "z_mm": np.full(len(times), L_anode * 1e3),
        "r_mm": np.zeros(len(times)),
        "dip_pct": 0.0,
        "I_pre_dip": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0.0,
        "I_dip": 0.0, "t_dip": 0.0,
        "scaling": None, "crowbar_t": None,
        "snowplow_obj": None, "dt_ns": 0,
    }


def _run_python_mhd(
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
    field_coupled_candidate: bool = False,
) -> dict[str, Any]:
    """Run Python NumPy MHD solver (CylindricalMHDSolver).

    Uses Godunov (PLM+HLL) flux with conservative total energy for shock
    stability. Cross-platform — no PyTorch/Metal required. Stable at all
    grid resolutions including MA-class DPF discharges.
    """
    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver
    from dpf.diagnostics.neutron_yield import neutron_yield_rate
    from dpf.validation.circuit_field_coupling import (
        field_power_diagnostics_from_cylindrical_state,
        implicit_midpoint_power_port_back_emf,
    )
    from dpf.validation.first_principles_limiters import (
        limiter_event,
        summarize_limiter_ledger,
    )

    nr, ny, nz = grid_shape

    solver = CylindricalMHDSolver(
        nr=nr, nz=nz, dr=dr, dz=dz,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3,
        enable_hall=False,
        enable_resistive=True,
        ion_mass=gas["m_mol"],
        use_godunov_flux=True,
        conservative_energy=True,
        r_min=a,
        diffusion_method=(
            "implicit_cylindrical_btheta"
            if field_coupled_candidate
            else "explicit"
        ),
        sts_stages=8,
    )

    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
        crowbar_inductance=cc.get("crowbar_inductance", 0.0),
    )

    initial_ionization_fraction = 0.01 if field_coupled_candidate else 1.0
    startup_initialization = None
    if field_coupled_candidate:
        startup_initialization = {
            "classification": "source_traced_startup_candidate_not_validation",
            "model": "pf1000_post_breakdown_partially_ionized_initial_state",
            "initial_Te_K": 300.0,
            "initial_Ti_K": 300.0,
            "initial_ionization_fraction": initial_ionization_fraction,
            "fill_pressure_Pa": float(p_pa),
            "can_support_first_principles_startup": False,
            "source_basis": {
                "post_breakdown_initialization": (
                    "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json:62"
                ),
                "breakdown_and_transport_sensitivity": (
                    "KnowledgeReference/scholz-2006-pf1000-mega-joule.md:149-208"
                ),
            },
            "blockers": [
                "same_scope_akel_startup_packet_missing",
                "ionization_kinetics_not_evolved",
                "insulator_surface_state_not_source_closed",
            ],
        }

    # Uniform IC — no stochastic perturbation for Python solver (numerically fragile)
    initial_pressure_pa = (
        p_pa * (1.0 + initial_ionization_fraction)
        if field_coupled_candidate
        else p_pa
    )
    if startup_initialization is not None:
        startup_initialization["initial_total_pressure_Pa"] = float(
            initial_pressure_pa
        )
        startup_initialization["pressure_model"] = (
            "neutral_heavy_particle_pressure_plus_electron_partial_pressure"
        )
    state = {
        "rho": np.full((nr, 1, nz), rho0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), initial_pressure_pa),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 300.0),
        "Ti": np.full((nr, 1, nz), 300.0),
        "Z_bar": np.full((nr, 1, nz), initial_ionization_fraction),
        "psi": np.zeros((nr, 1, nz)),
    }

    coupling = CouplingState()
    t = 0.0
    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    rho_max_arr, T_max_arr, B_max_arr = [], [], []
    mhd_snapshots = []
    magnetic_energy_arr: list[float] = []
    field_L_arr: list[float] = []
    dL_field_dt_arr: list[float] = []
    j_dot_e_power_arr: list[float] = []
    poynting_power_arr: list[float] = []
    joule_power_arr: list[float] = []
    joule_energy_arr: list[float] = []
    back_emf_arr: list[float] = []
    field_energy_residual_arr: list[float] = []
    coupling_source_arr: list[str] = []
    field_terminal_voltage_arr: list[float] = []
    poynting_source_voltage_arr: list[float] = []
    j_dot_e_voltage_arr: list[float] = []
    magnetic_power_arr: list[float] = []
    field_load_power_arr: list[float] = []
    field_power_back_emf_arr: list[float] = []
    field_power_port_current_arr: list[float] = []
    field_power_port_residual_arr: list[float] = []
    eta_min_arr: list[float] = []
    eta_mean_arr: list[float] = []
    eta_max_arr: list[float] = []
    dt_s_arr: list[float] = []
    dt_adv_s_arr: list[float] = []
    dt_diff_s_arr: list[float] = []
    dt_sts_s_arr: list[float] = []
    dt_circuit_s_arr: list[float] = []
    resistive_stiffness_ratio_arr: list[float] = []
    dt_controller_arr: list[str] = []
    resistivity_meta: dict[str, object] | None = None
    limiter_meta: dict[str, object] | None = None
    limiter_event_log: list[dict[str, object]] = []
    if field_coupled_candidate and solver.use_godunov_flux:
        numerical_verification = {
            "verification_tests": [
                "tests/test_cylindrical_godunov.py",
                "tests/test_mhd_solver_consolidated.py",
            ],
            "claim_scope": "code_verification_only_not_experimental_validation",
        }
        limiter_event_log.extend([
            limiter_event(
                limiter_id="dpf.fluid.cylindrical_mhd.plm_minmod_reconstruction",
                code_path="dpf.fluid.cylindrical_mhd.CylindricalMHDSolver._plm_reconstruct",
                affected_field="reconstructed_state",
                classification="verified_numerical_method",
                activation_count=0,
                threshold={
                    "reconstruction": "plm",
                    "slope_limiter": "minmod",
                    **numerical_verification,
                },
                acceptance_blocking=False,
                justification=(
                    "PLM minmod reconstruction is an explicit numerical method, "
                    "not a hidden state repair; it remains code-verification "
                    "evidence only."
                ),
            ),
            limiter_event(
                limiter_id="dpf.fluid.cylindrical_mhd.hll_riemann_flux",
                code_path="dpf.fluid.cylindrical_mhd.CylindricalMHDSolver._hll_flux_8",
                affected_field="fluxes",
                classification="verified_numerical_method",
                activation_count=0,
                threshold={
                    "riemann_solver": "hll",
                    **numerical_verification,
                },
                acceptance_blocking=False,
                justification=(
                    "HLL flux is an explicit finite-volume method component, "
                    "not a hidden first-principles closure."
                ),
            ),
            limiter_event(
                limiter_id="dpf.fluid.cylindrical_mhd.reconstructed_state_positivity_floor",
                code_path="dpf.fluid.cylindrical_mhd.CylindricalMHDSolver._compute_godunov_rhs",
                affected_field="flux_reconstruction_inputs",
                classification="verified_numerical_method",
                activation_count=0,
                threshold={
                    "rho_floor": 1.0e-20,
                    "pressure_floor": 1.0e-20,
                    **numerical_verification,
                },
                acceptance_blocking=False,
                justification=(
                    "Reconstructed-state positivity floors are flux-local "
                    "finite-volume method controls; returned-state floors "
                    "remain separately recorded as acceptance blockers."
                ),
            ),
            limiter_event(
                limiter_id="dpf.fluid.cylindrical_mhd.cfl_timestep_control",
                code_path="dpf.fluid.cylindrical_mhd.CylindricalMHDSolver._compute_dt",
                affected_field="dt",
                classification="verified_numerical_method",
                activation_count=0,
                threshold={
                    "cfl": solver.cfl,
                    **numerical_verification,
                },
                acceptance_blocking=False,
                justification=(
                    "CFL timestep control is an explicit numerical stability "
                    "method; debug timestep fallbacks remain separate blockers."
                ),
            ),
            limiter_event(
                limiter_id=(
                    "dpf.fluid.cylindrical_mhd."
                    "implicit_cylindrical_btheta_resistive_induction"
                ),
                code_path=(
                    "dpf.fluid.cylindrical_mhd.CylindricalMHDSolver."
                    "_apply_implicit_btheta_resistive_induction"
                ),
                affected_field="B",
                classification="verified_numerical_method",
                activation_count=0,
                threshold={
                    "method": "Crank-Nicolson ADI",
                    "operator": "-curl(eta * curl(B) / mu_0)_theta",
                    "scope": "axisymmetric B_theta",
                    "source_basis": [
                        "KnowledgeReference/scholz-2006-pf1000-mega-joule.md:190-208",
                        "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2259-2283",
                    ],
                    **numerical_verification,
                },
                acceptance_blocking=False,
                justification=(
                    "The implicit split advances the source-traced cylindrical "
                    "B_theta resistive induction operator without an eta cap "
                    "or hidden explicit diffusion timestep collapse."
                ),
            ),
        ])
    limiter_activation_arr: list[int] = []
    limiter_nonfinite_repair_arr: list[int] = []
    cumulative_joule_energy = 0.0
    compute_thermonuclear_history = gas.get("A") == 2 and gas.get("Z") == 1
    neutron_cell_volumes = (
        2.0 * np.pi * np.maximum(solver.geom.r, 1.0e-12)[:, None, None] * dr * dz
        if compute_thermonuclear_history else None
    )
    neutron_times_s: list[float] = []
    neutron_dY_thermo: list[float] = []
    neutron_dY_beam: list[float] = []
    neutron_rate_thermo: list[float] = []
    neutron_Y_thermo_cumulative: list[float] = []
    neutron_T_peak_keV: list[float] = []
    neutron_n_peak: list[float] = []
    cumulative_thermonuclear_yield = 0.0
    previous_field_L: float | None = None
    previous_magnetic_energy_J = 0.0
    circuit_phase_step_rad = 2.0 * np.pi / 32768.0
    circuit_lc_time_s = float(np.sqrt(max(circuit.L_ext * circuit.C, 1.0e-300)))
    if field_coupled_candidate:
        limiter_event_log.append(
            limiter_event(
                limiter_id="app_mhd.field_coupling.implicit_midpoint_power_port",
                code_path="app_mhd._run_python_mhd",
                affected_field="field_coupling",
                classification="verified_numerical_method",
                activation_count=0,
                threshold={
                    "power_port": "P_load = I_mid * V_load",
                    "circuit_update": "RLCSolver implicit midpoint",
                    "source_basis": [
                        "KnowledgeReference/auluck-2021-dpf-circuit-element.md:443-450",
                        (
                            "KnowledgeReference/a-course-on-plasma-focus-numerical-"
                            "experiments-s-lee-and-s-h-saw-part-1-basic-course.md:"
                            "12103-12128"
                        ),
                    ],
                },
                acceptance_blocking=False,
                justification=(
                    "Field load power is coupled through the circuit power port "
                    "without an arbitrary minimum-current floor."
                ),
            )
        )
        limiter_event_log.append(
            limiter_event(
                limiter_id="app_mhd.circuit_coupling.lc_phase_timestep_control",
                code_path="app_mhd._run_python_mhd",
                affected_field="dt",
                classification="verified_numerical_method",
                activation_count=0,
                threshold={
                    "method": "LC phase resolution",
                    "phase_step_rad": circuit_phase_step_rad,
                    "base_lc_time_s": circuit_lc_time_s,
                    "source_basis": [
                        "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json:62",
                        (
                            "KnowledgeReference/a-course-on-plasma-focus-numerical-"
                            "experiments-s-lee-and-s-h-saw-part-1-basic-course.md:"
                            "870-872"
                        ),
                    ],
                },
                acceptance_blocking=False,
                justification=(
                    "The field-coupled split advances MHD and the capacitor "
                    "bank on a reported LC phase increment so removing the "
                    "resistive diffusion CFL does not decouple the field "
                    "boundary from the circuit current."
                ),
            )
        )

    step = 0
    nan_detected = False
    nonfinite_counts: dict[str, int] = {}
    while t < t_end:
        remaining = t_end - t
        if field_coupled_candidate and remaining <= max(1.0e-15, 1.0e-9 * t_end):
            break
        eta_field = None
        if field_coupled_candidate:
            eta_field, resistivity_meta = _first_principles_eta_field(state, gas)
            limiter_event_log.extend(
                event
                for event in resistivity_meta.get("limiter_events", [])
                if isinstance(event, dict)
            )
            eta_min_arr.append(float(np.nanmin(eta_field)))
            eta_mean_arr.append(float(np.nanmean(eta_field)))
            eta_max_arr.append(float(np.nanmax(eta_field)))
            finite_eta = eta_field[np.isfinite(eta_field) & (eta_field > 0.0)]
            solver._last_eta_max = float(np.max(finite_eta)) if finite_eta.size else 0.0

        dt_mhd = solver.compute_dt(state)
        dt_diag = solver.last_dt_diagnostics
        dt_circuit = np.nan
        if field_coupled_candidate:
            L_for_dt = circuit.L_ext
            if circuit.state.crowbar_fired:
                t_since_fire = max(
                    circuit.state.time - circuit.state.crowbar_fire_time,
                    0.0,
                )
                closure_time = getattr(circuit, "crowbar_closure_time", 0.0)
                ramp = (
                    t_since_fire / closure_time
                    if closure_time > 0.0 and t_since_fire < closure_time
                    else 1.0
                )
                L_for_dt += circuit.crowbar_inductance * ramp
            dt_circuit = circuit_phase_step_rad * float(
                np.sqrt(max(L_for_dt * circuit.C, 1.0e-300))
            )
        dt_limit = dt_mhd
        controller = str(dt_diag.get("controller") or "unknown")
        if np.isfinite(dt_circuit) and dt_circuit < dt_limit:
            dt_limit = dt_circuit
            controller = "circuit_lc_phase_resolution"
        dt = min(dt_limit, remaining)
        if remaining < dt_limit:
            controller = "remaining_interval"
        dt_s_arr.append(float(dt))
        dt_adv_s_arr.append(float(dt_diag.get("dt_adv_s") or np.nan))
        dt_diff_value = dt_diag.get("dt_diff_s")
        dt_diff_s_arr.append(
            float(dt_diff_value) if dt_diff_value is not None else np.nan
        )
        dt_sts_value = dt_diag.get("dt_sts_s")
        dt_sts_s_arr.append(
            float(dt_sts_value) if dt_sts_value is not None else np.nan
        )
        dt_circuit_s_arr.append(
            float(dt_circuit) if np.isfinite(dt_circuit) else np.nan
        )
        stiffness_value = dt_diag.get("resistive_stiffness_ratio")
        resistive_stiffness_ratio_arr.append(
            float(stiffness_value) if stiffness_value is not None else np.nan
        )
        dt_controller_arr.append(controller)
        if dt <= 0:
            break

        state = solver.step(
            state, dt, current=circuit.current, voltage=circuit.voltage,
            anode_radius=a, cathode_radius=b, apply_electrode_bc=True,
            eta_field=eta_field,
        )
        if field_coupled_candidate:
            state["Z_bar"] = np.full_like(
                state["rho"],
                initial_ionization_fraction,
                dtype=float,
            )
            limiter_event_log.extend(
                event
                for event in getattr(solver, "last_limiter_events", [])
                if isinstance(event, dict)
            )
            state, limiter_meta = _apply_first_principles_engineering_bounds(
                state,
                gas,
                rho0,
                dr=dr,
                dz=dz,
                r_cell_m=solver.geom.r,
                magnetic_energy_cap_J=0.8 * circuit.initial_energy(),
            )
            counts = dict(limiter_meta.get("counts", {}))
            limiter_event_log.extend(
                event
                for event in limiter_meta.get("limiter_events", [])
                if isinstance(event, dict)
            )
            limiter_activation_arr.append(int(sum(counts.values())))
            limiter_nonfinite_repair_arr.append(
                int(
                    sum(
                        value for key, value in counts.items()
                        if "nonfinite" in str(key)
                    )
                )
            )

        # Nonfinite detection — break early and return valid data so far.
        nonfinite_counts = {
            key: int(np.size(value) - np.count_nonzero(np.isfinite(value)))
            for key, value in state.items()
            if isinstance(value, np.ndarray)
        }
        nonfinite_counts = {
            key: count for key, count in nonfinite_counts.items() if count > 0
        }
        if nonfinite_counts:
            nan_detected = True
            logger.warning(
                "Python MHD: nonfinite state at step %d, t=%.3e: %s — stopping early",
                step,
                t,
                nonfinite_counts,
            )
            break

        # Radiation cooling (Frontier D): bremsstrahlung + line radiation
        if "Te" in state:
            try:
                from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses
                rho_positive = np.where(state["rho"] > 0, state["rho"], 0.0)
                Z_bar_radiation = np.asarray(
                    state.get("Z_bar", np.ones_like(rho_positive)),
                    dtype=float,
                )
                ne = rho_positive / gas["m_mol"] * np.maximum(Z_bar_radiation, 0.0)
                Z_eff = gas.get("Z", 1)
                state["Te"], _ = apply_bremsstrahlung_losses(
                    state["Te"], ne, dt, Z=Z_eff,
                )
                if gas.get("Z", 1) > 1:
                    from dpf.radiation.line_radiation import apply_line_radiation_losses
                    state["Te"], _ = apply_line_radiation_losses(
                        state["Te"], ne, dt, Z_eff=0,
                        n_imp_frac=0.0, Z_imp=gas.get("Z", 10),
                    )
            except ImportError:
                pass

        if compute_thermonuclear_history and neutron_cell_volumes is not None:
            try:
                rho_for_yield = np.maximum(state["rho"], 0.0)
                Z_bar_yield = np.asarray(
                    state.get("Z_bar", np.ones_like(rho_for_yield)),
                    dtype=float,
                )
                n_D = (
                    rho_for_yield
                    / max(float(gas["m_mol"]), 1.0e-30)
                    * np.maximum(Z_bar_yield, 0.0)
                )
                rho_safe_yield = np.maximum(rho_for_yield, 1.0e-30)
                Ti = state.get(
                    "Ti",
                    state["pressure"] * float(gas["m_mol"])
                    / (2.0 * rho_safe_yield * kB),
                )
                _, thermo_rate = neutron_yield_rate(
                    n_D,
                    Ti,
                    neutron_cell_volumes,
                )
                dY_thermo = max(float(thermo_rate) * dt, 0.0)
                cumulative_thermonuclear_yield += dY_thermo
                neutron_times_s.append(t + dt)
                neutron_dY_thermo.append(dY_thermo)
                neutron_dY_beam.append(0.0)
                neutron_rate_thermo.append(max(float(thermo_rate), 0.0))
                neutron_Y_thermo_cumulative.append(cumulative_thermonuclear_yield)
                neutron_T_peak_keV.append(
                    float(np.nanmax(Ti)) * kB / (1000.0 * 1.602176634e-19)
                )
                neutron_n_peak.append(float(np.nanmax(n_D)))
            except Exception as exc:
                logger.debug("Field-history thermonuclear yield skipped: %s", exc)

        # Field-derived circuit coupling for first_principles_mhd. The legacy
        # Python mode keeps its previous reduced coupling behavior.
        mhd_coupling = solver.coupling_interface()
        if field_coupled_candidate:
            field_diag = field_power_diagnostics_from_cylindrical_state(
                state,
                dr=dr,
                dz=dz,
                current_A=circuit.current,
                r_cell_m=solver.geom.r,
                eta_ohm_m=eta_field,
                previous_inductance_H=previous_field_L,
                dt_s=dt,
                current_floor_A=0.0,
            )
            L_field = float(field_diag["field_derived_inductance_H"])
            dL_field_dt = float(field_diag["dL_field_dt_H_s"])
            previous_field_L = L_field
            magnetic_energy_J = float(field_diag["magnetic_energy_J"])
            magnetic_power_W = (magnetic_energy_J - previous_magnetic_energy_J) / dt
            previous_magnetic_energy_J = magnetic_energy_J
            joule_power_W = max(float(field_diag["joule_power_W"]), 0.0)
            # The circuit load is the resolved-field energy draw. J.E remains a
            # diagnostic because the current boundary condition directly changes
            # magnetic energy at the electrode-facing cells.
            field_load_power_W = magnetic_power_W + joule_power_W
            power_port_L_total = circuit.L_ext
            power_port_R_eff = circuit.R_total
            if circuit.state.crowbar_fired:
                t_since_fire = max(
                    circuit.state.time - circuit.state.crowbar_fire_time,
                    0.0,
                )
                closure_time = getattr(circuit, "crowbar_closure_time", 0.0)
                ramp = (
                    t_since_fire / closure_time
                    if closure_time > 0.0 and t_since_fire < closure_time
                    else 1.0
                )
                power_port_R_eff += circuit.crowbar_resistance * ramp
                power_port_L_total += circuit.crowbar_inductance * ramp
            power_port = implicit_midpoint_power_port_back_emf(
                current_A=circuit.current,
                capacitor_voltage_V=circuit.voltage,
                L_total_H=power_port_L_total,
                resistance_ohm=power_port_R_eff,
                capacitance_F=circuit.C,
                dL_dt_H_s=0.0,
                dt_s=dt,
                power_W=field_load_power_W,
                crowbar_fired=circuit.state.crowbar_fired,
            )
            if bool(power_port.get("passed")):
                back_emf = float(power_port["back_emf_V"])
            else:
                back_emf = 0.0
                limiter_event_log.append(
                    limiter_event(
                        limiter_id=(
                            "app_mhd.field_coupling."
                            "midpoint_power_port_no_real_root"
                        ),
                        code_path="app_mhd._run_python_mhd",
                        affected_field="field_coupling",
                        classification="acceptance_blocker",
                        activation_count=1,
                        before=field_load_power_W,
                        after=0.0,
                        threshold={
                            "reason": str(power_port.get("reason", "unknown")),
                            "method": "implicit_midpoint_power_port",
                            "current_A": circuit.current,
                            "voltage_V": circuit.voltage,
                            "dt_s": dt,
                        },
                        acceptance_blocking=True,
                        justification=(
                            "Resolved-field load power could not be represented "
                            "as a real implicit-midpoint circuit terminal voltage."
                        ),
                    )
                )
            if not np.isfinite(back_emf):
                limiter_event_log.append(
                    limiter_event(
                        limiter_id="app_mhd.field_coupling.back_emf_nonfinite_repair",
                        code_path="app_mhd._run_python_mhd",
                        affected_field="back_emf",
                        classification="debug_repair",
                        activation_count=1,
                        before=back_emf,
                        after=0.0,
                        acceptance_blocking=True,
                        justification="Non-finite back-EMF repaired to zero.",
                    )
                )
                back_emf = 0.0
            # L_field is exported as a diagnostic. The plasma load enters the
            # circuit through field-power back-EMF to avoid double counting
            # magnetic energy in both Lp and the resolved field ledger.
            coupling_input = CouplingState(
                Lp=0.0,
                dL_dt=0.0,
                current=circuit.current,
                voltage=circuit.voltage,
            )
            coupling = circuit.step(coupling_input, back_emf=back_emf, dt=dt)
            Lp_mhd = L_field if np.isfinite(L_field) and L_field > 0.0 else 0.0
            cumulative_joule_energy += joule_power_W * dt
            external_inductive_J = 0.5 * circuit.L_ext * circuit.current**2
            energy_residual_J = (
                circuit.initial_energy()
                - circuit.state.energy_cap
                - external_inductive_J
                - magnetic_energy_J
                - circuit.state.energy_res
                - cumulative_joule_energy
            )
            magnetic_energy_arr.append(magnetic_energy_J / 1e3)
            field_L_arr.append(L_field * 1e9)
            dL_field_dt_arr.append(dL_field_dt)
            j_dot_e_power_arr.append(float(field_diag["j_dot_e_power_W"]))
            poynting_power_arr.append(field_load_power_W)
            joule_power_arr.append(joule_power_W)
            joule_energy_arr.append(cumulative_joule_energy / 1e3)
            field_energy_residual_arr.append(energy_residual_J / 1e3)
            field_terminal_voltage_arr.append(back_emf)
            poynting_source_voltage_arr.append(
                float(field_diag["poynting_voltage_source_orientation_V"])
            )
            j_dot_e_voltage_arr.append(float(field_diag["field_terminal_voltage_V"]))
            magnetic_power_arr.append(magnetic_power_W)
            field_load_power_arr.append(field_load_power_W)
            field_power_back_emf_arr.append(back_emf)
            field_power_port_current_arr.append(
                float(power_port.get("current_mid_A", 0.0))
            )
            field_power_port_residual_arr.append(
                float(power_port.get("power_residual_W", 0.0))
            )
            coupling_source_arr.append("field_coupled_candidate")
        else:
            back_emf = 0.0
            if mhd_coupling.dL_dt is not None and abs(circuit.current) > 1.0:
                back_emf = mhd_coupling.dL_dt * circuit.current
            coupling = circuit.step(coupling, back_emf=back_emf, dt=dt)
            Lp_mhd = mhd_coupling.Lp if mhd_coupling.Lp > 0 else coupling.Lp
            magnetic_energy_arr.append(0.0)
            field_L_arr.append(Lp_mhd * 1e9)
            dL_field_dt_arr.append(
                float(mhd_coupling.dL_dt) if mhd_coupling.dL_dt is not None else 0.0
            )
            j_dot_e_power_arr.append(0.0)
            poynting_power_arr.append(0.0)
            joule_power_arr.append(0.0)
            joule_energy_arr.append(0.0)
            field_energy_residual_arr.append(0.0)
            field_terminal_voltage_arr.append(back_emf)
            poynting_source_voltage_arr.append(-back_emf)
            j_dot_e_voltage_arr.append(back_emf)
            magnetic_power_arr.append(0.0)
            field_load_power_arr.append(0.0)
            field_power_back_emf_arr.append(back_emf)
            field_power_port_current_arr.append(circuit.current)
            field_power_port_residual_arr.append(0.0)
            coupling_source_arr.append("mhd_inductance_candidate")
        back_emf_arr.append(back_emf)
        t += dt
        step += 1

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(Lp_mhd * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        rho_max_arr.append(float(np.nanmax(state["rho"])))
        rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
        T_max_arr.append(float(np.nanmax(state.get("Te", state["pressure"] * 3.34e-27 / (2.0 * rho_safe * 1.380649e-23)))))
        B_max_arr.append(float(np.nanmax(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if step % 100 == 0:
            _snap = {
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, 0, :].copy(),
                "P_mid": state["pressure"][:, 0, :].copy(),
            }
            if "B" in state:
                _snap["B_mid"] = state["B"][:, :, 0, :].copy()
            if "velocity" in state:
                _snap["vel_mid"] = state["velocity"][:, :, 0, :].copy()
            if "Te" in state:
                _snap["Te_mid"] = state["Te"][:, 0, :].copy()
            mhd_snapshots.append(_snap)

        if progress_fn and step % 10 == 0:
            progress_fn(min(t / t_end, 1.0), desc=f"Level 3 MHD: step {step} | t={t*1e6:.1f}/{t_end*1e6:.0f} us ({min(t/t_end,1)*100:.0f}%)")

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr))) if len(I_arr) > 0 else 0
    limiter_ledger = summarize_limiter_ledger(
        limiter_event_log,
        source="app_mhd._run_python_mhd",
    )

    result = {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "magnetic_energy_kJ": np.array(magnetic_energy_arr),
        "magnetic_energy_J": np.array(magnetic_energy_arr) * 1e3,
        "magnetic_energy_inductance": np.array(field_L_arr),
        "field_derived_inductance": np.array(field_L_arr),
        "field_derived_inductance_H": np.array(field_L_arr) * 1e-9,
        "Lp_field_nH": np.array(field_L_arr),
        "Lp_mhd_nH": np.array(field_L_arr),
        "dL_field_dt_H_s": np.array(dL_field_dt_arr),
        "j_dot_e_power_W": np.array(j_dot_e_power_arr),
        "poynting_power_W": np.array(poynting_power_arr),
        "field_interface_power_W": np.array(poynting_power_arr),
        "joule_power_W": np.array(joule_power_arr),
        "joule_energy_kJ": np.array(joule_energy_arr),
        "magnetic_power_W": np.array(magnetic_power_arr),
        "field_load_power_W": np.array(field_load_power_arr),
        "field_power_back_emf_V": np.array(field_power_back_emf_arr),
        "field_power_port_current_A": np.array(field_power_port_current_arr),
        "field_power_port_residual_W": np.array(field_power_port_residual_arr),
        "j_dot_e_voltage_V": np.array(j_dot_e_voltage_arr),
        "field_terminal_voltage_V": np.array(field_terminal_voltage_arr),
        "poynting_voltage_source_orientation_V": np.array(poynting_source_voltage_arr),
        "back_emf_V": np.array(back_emf_arr),
        "field_energy_residual_kJ": np.array(field_energy_residual_arr),
        "circuit_energy_residual_kJ": np.array(field_energy_residual_arr),
        "eta_min_ohm_m": np.array(eta_min_arr),
        "eta_mean_ohm_m": np.array(eta_mean_arr),
        "eta_max_ohm_m": np.array(eta_max_arr),
        "dt_s": np.array(dt_s_arr),
        "dt_adv_s": np.array(dt_adv_s_arr),
        "dt_diff_s": np.array(dt_diff_s_arr),
        "dt_sts_s": np.array(dt_sts_s_arr),
        "dt_circuit_s": np.array(dt_circuit_s_arr),
        "resistive_stiffness_ratio": np.array(resistive_stiffness_ratio_arr),
        "dt_controller": dt_controller_arr,
        "timestep_controller": dt_controller_arr,
        "first_principles_resistivity": resistivity_meta or {
            "validation_status": "not_applied",
        },
        "field_limiter_activation_count": np.array(limiter_activation_arr),
        "field_limiter_nonfinite_repair_count": np.array(limiter_nonfinite_repair_arr),
        "first_principles_engineering_limiter": limiter_meta or {
            "validation_status": "not_applied",
        },
        "first_principles_limiter_ledger": limiter_ledger,
        "coupling_source": coupling_source_arr,
        "coupling_interval_authority": coupling_source_arr,
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0,
        "t_peak": float(t_arr[I_peak_idx]) if len(t_arr) > 0 else 0,
        "n_steps": step,
        "has_snowplow": False,
        "has_mhd": True,
        "phases": (
            ["field_coupled_candidate"] * len(times)
            if field_coupled_candidate
            else ["mhd"] * len(times)
        ),
        "z_mm": np.full(len(times), L_anode * 1e3),
        "r_mm": np.zeros(len(times)),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "dip_pct": 0.0,
        "I_pre_dip": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0.0,
        "I_dip": 0.0, "t_dip": 0.0,
        "scaling": None, "crowbar_t": None,
        "snowplow_obj": None, "dt_ns": 0,
        "nan_detected": nan_detected,
        "nan_step": step if nan_detected else None,
        "nonfinite_state_counts": nonfinite_counts if nan_detected else {},
        "field_coupled_candidate": field_coupled_candidate,
        "field_power_port": {
            "classification": "source_traced_numerical_power_port_candidate",
            "validation_status": "not_validation_evidence",
            "method": "implicit_midpoint_power_port",
            "power_equation": "P_load = I_mid * V_load",
            "source_basis": [
                "KnowledgeReference/auluck-2021-dpf-circuit-element.md:443-450",
                (
                    "KnowledgeReference/a-course-on-plasma-focus-numerical-"
                    "experiments-s-lee-and-s-h-saw-part-1-basic-course.md:"
                    "12103-12128"
                ),
            ],
            "limitation": (
                "The power port removes arbitrary current-floor suppression, "
                "but the field-load partition still requires same-scope "
                "field-coupling validation before predictive claims."
            ),
        },
        "field_power_sign_convention": (
            "field_terminal_voltage_V is the implicit-midpoint load voltage "
            "whose product with field_power_port_current_A equals the resolved "
            "field_load_power_W; it is passed as RLCSolver back_emf during "
            "first_principles_mhd."
        ),
    }
    if startup_initialization is not None:
        result["startup_sheath_initialization"] = startup_initialization
        result["breakdown_model"] = {
            "classification": "post_breakdown_source_traced_candidate",
            "validation_status": "candidate_not_validated",
            "source_basis": startup_initialization["source_basis"],
            "can_support_first_principles_startup": False,
        }
        result["preionization"] = {
            "classification": "source_traced_initial_preionization_candidate",
            "ionization_fraction": initial_ionization_fraction,
            "validation_status": "candidate_not_validated",
            "source_basis": startup_initialization["source_basis"],
        }
        result["initial_plasma_distribution"] = {
            "classification": "uniform_post_breakdown_candidate",
            "rho0_kg_m3": float(rho0),
            "pressure_Pa": float(p_pa),
            "Te_K": 300.0,
            "Ti_K": 300.0,
            "ionization_fraction": initial_ionization_fraction,
            "validation_status": "candidate_not_validated",
        }
    if compute_thermonuclear_history and neutron_times_s:
        Y_thermo = float(cumulative_thermonuclear_yield)
        result["yield_time_resolved"] = {
            "t_s": np.array(neutron_times_s),
            "times_us": np.array(neutron_times_s) * 1.0e6,
            "dY_thermo": np.array(neutron_dY_thermo),
            "dY_th": np.array(neutron_dY_thermo),
            "dY_bt": np.array(neutron_dY_beam),
            "thermonuclear_rate": np.array(neutron_rate_thermo),
            "Y_thermo_cumulative": np.array(neutron_Y_thermo_cumulative),
            "Y_bt_cumulative": np.zeros(len(neutron_times_s)),
            "T_peak_keV": np.array(neutron_T_peak_keV),
            "n_peak": np.array(neutron_n_peak),
            "Y_total": Y_thermo,
            "bt_fraction": 0.0,
            "peak_yield_time_us": (
                float(neutron_times_s[int(np.argmax(neutron_dY_thermo))]) * 1.0e6
                if neutron_dY_thermo else 0.0
            ),
            "source_authority": "resolved_field_history_candidate",
            "validation_status": "estimate_not_validation",
        }
        if Y_thermo > 0.0:
            result["neutron_yield"] = {
                "Y_thermonuclear": Y_thermo,
                "Y_beam_target": 0.0,
                "Y_neutron": Y_thermo,
                "bt_fraction": 0.0,
                "tau_ns": float((neutron_times_s[-1] - neutron_times_s[0]) * 1.0e9)
                if len(neutron_times_s) > 1 else 0.0,
                "model_role": "mechanism_separated_neutron_yield_estimate",
                "validation_status": "estimate_not_validation",
                "first_principles_total_yield_authority": "blocked",
                "thermonuclear_input_authority": "resolved_field_history_candidate",
                "beam_target_input_authority": "kinetic_hybrid_missing",
                "can_support_first_principles_neutron_yield": False,
                "validity_notes": {
                    "thermonuclear": (
                        "Thermonuclear DD yield is integrated over the "
                        "resolved MHD field history and cylindrical cell "
                        "volumes using Bosch-Hale reactivity."
                    ),
                    "beam_target": (
                        "No first-principles kinetic/hybrid beam-target "
                        "neutron model is accepted for this run; total-yield "
                        "authority remains blocked."
                    ),
                },
            }
    return result


def create_mhd_fields_fig(d: dict[str, Any]) -> Any:
    """Create 2D field plots from MHD snapshots with physical coordinates.

    Shows density + pressure (always), plus B-field magnitude, velocity magnitude,
    current density J, and temperature when available in the snapshot data.
    Uses Plotly dropdown buttons to toggle between field views (max 3x2 grid).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mu_0 = 4.0 * np.pi * 1e-7

    snapshots = d.get("mhd_snapshots", [])
    if not snapshots:
        fig = go.Figure()
        fig.add_annotation(
            text="No MHD snapshots available", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#aaa"),
        )
        fig.update_layout(height=400, template="plotly_dark")
        return fig

    snap = snapshots[-1]
    rho = snap["rho_mid"]
    P = snap.get("P_mid")

    cc = d.get("circuit", {})
    sc = d.get("snowplow_cfg", {})
    a_m = cc.get("anode_radius", 0.01)
    b_m = cc.get("cathode_radius", 0.02)
    L_m = sc.get("anode_length", 0.16)
    nr, nz = rho.shape

    r_mm = np.linspace(a_m * 1e3, b_m * 1e3, nr)
    z_mm = np.linspace(0, L_m * 1e3, nz)
    dr = (b_m - a_m) / max(nr - 1, 1)
    dz = L_m / max(nz - 1, 1)

    # Build field catalog: (label, data_2d, unit, colorscale)
    fields: list[tuple[str, np.ndarray, str, str]] = [
        ("Density", rho, "kg/m^3", "Viridis"),
    ]
    if P is not None:
        fields.append(("Pressure", P, "Pa", "Inferno"))

    B_mid = snap.get("B_mid")
    if B_mid is not None:
        B_mag = np.sqrt(np.sum(B_mid**2, axis=0))
        fields.append(("|B| (magnetic field)", B_mag, "T", "Magma"))

        # Current density J = curl(B)/mu_0
        # In 2D midplane (r,z): J_theta ~ (dBr/dz - dBz/dr) / mu_0
        if B_mid.shape[0] >= 3:
            dBr_dz = np.gradient(B_mid[0], dz, axis=1)
            dBz_dr = np.gradient(B_mid[2], dr, axis=0)
            J_theta = (dBr_dz - dBz_dr) / mu_0
            fields.append(("J_theta (current density)", np.abs(J_theta), "A/m^2", "Hot"))

    vel_mid = snap.get("vel_mid")
    if vel_mid is not None:
        v_mag = np.sqrt(np.sum(vel_mid**2, axis=0))
        fields.append(("|v| (velocity)", v_mag / 1e3, "km/s", "Cividis"))

    Te_mid = snap.get("Te_mid")
    if Te_mid is not None:
        Te_eV = Te_mid * 1.380649e-23 / 1.602e-19
        fields.append(("Te (electron temperature)", Te_eV, "eV", "Plasma"))
    elif P is not None:
        # Estimate T from ideal gas: T = P * m_ion / (2 * rho * kB)
        gas = d.get("gas", {})
        m_ion = gas.get("m_mol", 3.34e-27)
        rho_safe = np.where(rho > 0, rho, 1.0)
        T_est = P * m_ion / (2.0 * rho_safe * 1.380649e-23)
        T_eV = T_est * 1.380649e-23 / 1.602e-19
        fields.append(("T (estimated)", T_eV, "eV [Estimated]", "Plasma"))

    n_fields = len(fields)
    n_cols = min(n_fields, 2)
    n_rows = (n_fields + n_cols - 1) // n_cols

    t_label = f"t={snap['t_us']:.1f} us"
    subplot_titles = [f"{f[0]} ({t_label})" for f in fields]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.12,
    )

    for idx, (label, data_2d, unit, cscale) in enumerate(fields):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        cb_x = 0.42 if col == 1 else 1.0
        fig.add_trace(go.Heatmap(
            z=data_2d, x=z_mm, y=r_mm, colorscale=cscale, name=label,
            colorbar=dict(title=unit, x=cb_x, len=0.9 / n_rows),
        ), row=row, col=col)

    # Data source label
    backend = d.get("backend", "unknown")
    _src = "MHD Solver (2D)"
    if "lee" in str(backend).lower():
        _src = "Lee Model (0D)"
    elif "hybrid" in str(backend).lower():
        _src = "Hybrid: Lee (0D) + MHD (2D)"
    fig.add_annotation(
        x=0.01, y=0.01, xref="paper", yref="paper",
        text=f"Data: {_src} | Backend: {backend}", showarrow=False,
        font=dict(size=9, color="#888"), xanchor="left", yanchor="bottom",
    )

    fig.update_layout(
        height=350 * n_rows, template="plotly_dark",
        margin=dict(l=60, r=20, t=40, b=40),
    )
    fig.update_xaxes(title_text="z [mm]")
    fig.update_yaxes(title_text="r [mm]")
    return fig
