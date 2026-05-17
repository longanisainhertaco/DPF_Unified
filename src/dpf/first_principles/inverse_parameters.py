"""Experimental inverse-parameter packets for source-grounded DPF decks.

This module treats missing machine parameters as algebraic unknowns only where
the source packet supplies enough independent observables.  It deliberately
marks non-unique fits as underdetermined instead of choosing hidden closure
values, because the output is intended to seed experimental whole-shot runs,
not to certify first-principles acceptance.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from dpf.first_principles.deck import pf1000_akel_16kv_engineering_deck
from dpf.first_principles.source_targets import (
    GV_ROOT,
    GV_VERIFIED_SHOTS,
    TORR_TO_PA,
    gv_verified_shot_targets,
    may15_user_validated_source_targets,
)

STATUS_KNOWN = "known_source_value"
STATUS_DIRECT = "direct_algebraic_inference"
STATUS_BRACKETED = "bracketed_source_range"
STATUS_WAVEFORM = "waveform_derived_candidate"
STATUS_UNDETERMINED = "underdetermined_requires_additional_observable"
STATUS_CONTRADICTION = "contradiction_or_scope_mismatch"
STATUS_UNAVAILABLE = "unavailable_source_artifact"


def bank_energy_J(capacitance_F: float, voltage_V: float) -> float:
    """Return ideal capacitor-bank energy, ``E = 1/2 C V^2``."""

    C = _positive_float(capacitance_F, "capacitance_F")
    V = _positive_float(voltage_V, "voltage_V")
    return 0.5 * C * V * V


def voltage_from_bank_energy_V(energy_J: float, capacitance_F: float) -> float:
    """Return ideal bank voltage from ``V = sqrt(2E/C)``."""

    E = _positive_float(energy_J, "energy_J")
    C = _positive_float(capacitance_F, "capacitance_F")
    return math.sqrt(2.0 * E / C)


def capacitance_from_bank_energy_F(energy_J: float, voltage_V: float) -> float:
    """Return ideal capacitance from ``C = 2E/V^2``."""

    E = _positive_float(energy_J, "energy_J")
    V = _positive_float(voltage_V, "voltage_V")
    return 2.0 * E / (V * V)


def ideal_lc_peak_current_A(
    *,
    capacitance_F: float,
    voltage_V: float,
    inductance_H: float,
) -> float:
    """Return the undamped LC current upper-bound, ``I = V sqrt(C/L)``."""

    C = _positive_float(capacitance_F, "capacitance_F")
    V = _positive_float(voltage_V, "voltage_V")
    L = _positive_float(inductance_H, "inductance_H")
    return V * math.sqrt(C / L)


def current_implied_inductance_H(
    *,
    capacitance_F: float,
    voltage_V: float,
    peak_current_A: float,
) -> float:
    """Return the ideal LC inductance implied by ``L = C (V/I)^2``."""

    C = _positive_float(capacitance_F, "capacitance_F")
    V = _positive_float(voltage_V, "voltage_V")
    I = _positive_float(peak_current_A, "peak_current_A")
    return C * (V / I) ** 2


def ideal_lc_quarter_cycle_s(*, capacitance_F: float, inductance_H: float) -> float:
    """Return the undamped LC quarter-cycle time, ``pi/2 sqrt(LC)``."""

    C = _positive_float(capacitance_F, "capacitance_F")
    L = _positive_float(inductance_H, "inductance_H")
    return 0.5 * math.pi * math.sqrt(L * C)


def quarter_cycle_implied_inductance_H(
    *,
    capacitance_F: float,
    quarter_cycle_s: float,
) -> float:
    """Return ideal inductance from a measured quarter cycle."""

    C = _positive_float(capacitance_F, "capacitance_F")
    t_q = _positive_float(quarter_cycle_s, "quarter_cycle_s")
    return ((2.0 * t_q / math.pi) ** 2) / C


def build_experimental_inverse_parameter_packet(
    *,
    scope: str = "all",
    include_gv_waveforms: bool = True,
    gv_series: str = "preferred",
    gv_root: str | Path = GV_ROOT,
    require_gv_hash_match: bool = True,
) -> dict[str, Any]:
    """Build a non-promoting packet of source-backed algebraic parameter fills."""

    normalized_scope = scope.strip().lower()
    if normalized_scope not in {"all", "pf1000", "may15", "gv"}:
        raise ValueError("scope must be one of all, pf1000, may15, gv")

    machines: dict[str, dict[str, Any]] = {}
    source_packets: dict[str, Any] = {}

    if normalized_scope in {"all", "pf1000"}:
        machines["pf1000_akel_16kv_shot_12581"] = _pf1000_akel_packet()

    if normalized_scope in {"all", "may15"}:
        may15 = may15_user_validated_source_targets()
        source_packets["may15_user_validated"] = {
            "batch_id": may15["batch_id"],
            "source_status": may15["source_status"],
            "accepted_for_whole_shot_first_principles": may15[
                "accepted_for_whole_shot_first_principles"
            ],
        }
        targets = may15["device_deck_targets"]
        machines["ir_mpf_100_salehizadeh_2012"] = _ir_mpf_100_packet(
            targets["ir_mpf_100_salehizadeh_2012"]
        )
        machines["compact_chinese_dpf_2018"] = _compact_chinese_packet(
            targets["compact_chinese_dpf_2018"]
        )
        machines["willenborg_hendricks_1977_startup_design"] = (
            _willenborg_packet(targets["willenborg_hendricks_1977_startup_design"])
        )

    if normalized_scope in {"all", "gv"}:
        gv = gv_verified_shot_targets()
        source_packets["gv_verified_local_shots"] = {
            "batch_id": gv["batch_id"],
            "source_status": gv["source_status"],
            "root": gv["root"],
            "accepted_for_whole_shot_first_principles": gv[
                "accepted_for_whole_shot_first_principles"
            ],
            "shot_count": gv["shot_count"],
        }
        for row in GV_VERIFIED_SHOTS:
            machine_id = f"gv_{row['shot_id']}"
            machines[machine_id] = _gv_shot_packet(
                row,
                include_waveform=include_gv_waveforms,
                series=gv_series,
                root=gv_root,
                require_hash_match=require_gv_hash_match,
            )

    status_counts = _status_counts(tuple(machines.values()))
    unresolved = _unresolved_parameters(machines)
    contradiction_or_scope_mismatch = _entries_by_status(
        machines,
        STATUS_CONTRADICTION,
    )
    return {
        "task_id": "experimental_inverse_parameter_completion",
        "status": "experimental_inverse_parameter_completion_not_validation",
        "scope": normalized_scope,
        "source_policy": {
            "source_truth": (
                "local KnowledgeReference/user-verified source packets only; "
                "GV workbooks are local user-verified candidates until promoted"
            ),
            "reduced_models_used": False,
            "gv_reduced_model_output_used": False,
            "measured_waveforms_used_as_drive": False,
            "algebraic_inference_only": True,
        },
        "source_packets": source_packets,
        "machine_count": len(machines),
        "machines": machines,
        "status_counts": status_counts,
        "unresolved_parameter_count": len(unresolved),
        "unresolved_parameters": unresolved,
        "contradiction_or_scope_mismatch_count": len(contradiction_or_scope_mismatch),
        "contradiction_or_scope_mismatch": contradiction_or_scope_mismatch,
        "deck_completion_policy": {
            "may_fill_experimental_decks": True,
            "may_promote_to_first_principles_acceptance": False,
            "reason": (
                "Algebra can fill circuit/geometry/gas candidates, but startup "
                "state, dynamic plasma loading, transport closures, detector UQ, "
                "and same-scope histories remain independent physics requirements."
            ),
        },
        "can_support_first_principles_acceptance": False,
    }


def _pf1000_akel_packet() -> dict[str, Any]:
    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    C = float(deck.circuit.capacitance_F)
    V = float(deck.circuit.voltage_V)
    L = float(deck.circuit.inductance_H)
    R = float(deck.circuit.resistance_ohm)
    pressure_Pa = float(deck.gas.pressure_Pa)
    geometry = deck.device
    source_references = tuple(
        {
            "path": source.path,
            "record_id": source.record_id,
            "capability_tags": list(source.capability_tags),
            "role": source.role,
        }
        for source in deck.source_references
    )
    cathode_inner_radius_m = float(geometry.cathode_radius_m)
    rod_diameter_m = float(geometry.cathode_rod_diameter_m or 0.0)
    cathode_outer_radius_m = cathode_inner_radius_m + 0.5 * rod_diameter_m
    axial_extent_m = float(geometry.anode_length_m) + float(geometry.insulator_length_m)
    return _machine_packet(
        machine_id="pf1000_akel_16kv_shot_12581",
        device=geometry.name,
        source_status="built_in_pf1000_akel_knowledge_reference_source",
        source_references=source_references,
        known_parameters={
            "capacitance_F": _known(C, "F", "PF-1000/Akel source deck"),
            "voltage_V": _known(V, "V", "PF-1000/Akel source deck"),
            "static_inductance_H": _known(L, "H", "PF-1000/Akel source deck"),
            "resistance_ohm": _known(R, "ohm", "PF-1000/Akel source deck"),
            "fill_pressure_Pa": _known(pressure_Pa, "Pa", "PF-1000/Akel source deck"),
            "anode_radius_m": _known(
                float(geometry.anode_radius_m),
                "m",
                "PF-1000 geometry source deck",
            ),
            "cathode_inner_radius_m": _known(
                cathode_inner_radius_m,
                "m",
                "PF-1000 geometry source deck",
            ),
            "cathode_rod_diameter_m": _known(
                rod_diameter_m,
                "m",
                "PF-1000 geometry source deck",
            ),
            "anode_length_m": _known(
                float(geometry.anode_length_m),
                "m",
                "PF-1000 geometry source deck",
            ),
            "insulator_length_m": _known(
                float(geometry.insulator_length_m),
                "m",
                "PF-1000 geometry source deck",
            ),
        },
        derived_parameters={
            "bank_energy_J": _direct(
                bank_energy_J(C, V),
                "J",
                formula="0.5 * capacitance_F * voltage_V**2",
                inputs={"capacitance_F": C, "voltage_V": V},
            ),
            "ideal_peak_current_from_static_L_A": _direct(
                ideal_lc_peak_current_A(
                    capacitance_F=C,
                    voltage_V=V,
                    inductance_H=L,
                ),
                "A",
                formula="voltage_V * sqrt(capacitance_F / static_inductance_H)",
                inputs={
                    "capacitance_F": C,
                    "voltage_V": V,
                    "static_inductance_H": L,
                },
                notes=(
                    "Ideal bank current upper-bound; does not include sheath "
                    "loading or dynamic inductance."
                ),
            ),
            "ideal_static_lc_quarter_cycle_s": _direct(
                ideal_lc_quarter_cycle_s(capacitance_F=C, inductance_H=L),
                "s",
                formula="0.5 * pi * sqrt(static_inductance_H * capacitance_F)",
                inputs={"capacitance_F": C, "static_inductance_H": L},
            ),
            "fill_pressure_torr": _direct(
                pressure_Pa / TORR_TO_PA,
                "torr",
                formula="fill_pressure_Pa / TORR_TO_PA",
                inputs={"fill_pressure_Pa": pressure_Pa},
            ),
            "cathode_outer_radius_m": _direct(
                cathode_outer_radius_m,
                "m",
                formula="cathode_inner_radius_m + 0.5 * cathode_rod_diameter_m",
                inputs={
                    "cathode_inner_radius_m": cathode_inner_radius_m,
                    "cathode_rod_diameter_m": rod_diameter_m,
                },
            ),
            "axial_extent_m": _direct(
                axial_extent_m,
                "m",
                formula="anode_length_m + insulator_length_m",
                inputs={
                    "anode_length_m": geometry.anode_length_m,
                    "insulator_length_m": geometry.insulator_length_m,
                },
            ),
        },
        unresolved_parameters={
            "hollow_anode_inner_radius_m": _underdetermined(
                "reviewed hollow-anode inner radius/source mask",
                (
                    "The geometry source tags hollow-anode context, but the "
                    "current executable deck does not carry an anode_inner_radius_m."
                ),
            ),
            "measured_waveform_peak_time": _underdetermined(
                "reviewed Akel current waveform digitization",
                "The built-in target references a blocked-by-review figure, not a typed series.",
            ),
            "startup_initial_state": _underdetermined(
                "breakdown/preionization/sheath-liftoff observables",
                "Circuit and static geometry do not define a whole-shot startup BVP.",
            ),
        },
        consistency_checks={},
        deck_fill_candidates={
            "capacitance_F": _known(C, "F", "PF-1000/Akel source deck"),
            "voltage_V": _known(V, "V", "PF-1000/Akel source deck"),
            "inductance_H": _known(L, "H", "PF-1000/Akel source deck"),
            "resistance_ohm": _known(R, "ohm", "PF-1000/Akel source deck"),
            "pressure_Pa": _known(pressure_Pa, "Pa", "PF-1000/Akel source deck"),
        },
    )


def _ir_mpf_100_packet(target: dict[str, Any]) -> dict[str, Any]:
    circuit = target["circuit"]
    C = float(circuit["capacitance_F"])
    V = float(circuit["maximum_voltage_V"])
    L_source = float(circuit["total_inductance_H"])
    I_source = float(circuit["theoretical_peak_current_A"])
    E_source = float(circuit["maximum_stored_energy_J"])
    E_calc = bank_energy_J(C, V)
    I_ideal = ideal_lc_peak_current_A(
        capacitance_F=C,
        voltage_V=V,
        inductance_H=L_source,
    )
    L_from_I = current_implied_inductance_H(
        capacitance_F=C,
        voltage_V=V,
        peak_current_A=I_source,
    )
    return _machine_packet(
        machine_id="ir_mpf_100_salehizadeh_2012",
        device=str(target["device"]),
        source_status="may15_user_verified_knowledge_reference_source",
        source_references=(_source_reference(target),),
        known_parameters={
            "capacitance_F": _known(C, "F", "source circuit table"),
            "maximum_voltage_V": _known(V, "V", "source circuit table"),
            "source_total_inductance_H": _known(L_source, "H", "source circuit table"),
            "design_resistance_ohm": _known(
                float(circuit["design_resistance_ohm"]),
                "ohm",
                "source circuit table",
            ),
            "source_theoretical_peak_current_A": _known(
                I_source,
                "A",
                "source circuit table",
            ),
            "source_maximum_stored_energy_J": _known(
                E_source,
                "J",
                "source circuit table",
            ),
        },
        derived_parameters={
            "bank_energy_from_CV_J": _direct(
                E_calc,
                "J",
                formula="0.5 * capacitance_F * maximum_voltage_V**2",
                inputs={"capacitance_F": C, "maximum_voltage_V": V},
            ),
            "ideal_peak_current_from_source_L_A": _direct(
                I_ideal,
                "A",
                formula="maximum_voltage_V * sqrt(capacitance_F / source_total_inductance_H)",
                inputs={
                    "capacitance_F": C,
                    "maximum_voltage_V": V,
                    "source_total_inductance_H": L_source,
                },
                notes=(
                    "Undamped LC upper-bound only; not a plasma-loaded current "
                    "waveform substitute."
                ),
            ),
            "source_peak_current_implied_inductance_H": _direct(
                L_from_I,
                "H",
                formula="capacitance_F * (maximum_voltage_V / source_theoretical_peak_current_A)**2",
                inputs={
                    "capacitance_F": C,
                    "maximum_voltage_V": V,
                    "source_theoretical_peak_current_A": I_source,
                },
            ),
            "ideal_lc_quarter_cycle_s": _direct(
                ideal_lc_quarter_cycle_s(capacitance_F=C, inductance_H=L_source),
                "s",
                formula="0.5 * pi * sqrt(source_total_inductance_H * capacitance_F)",
                inputs={"capacitance_F": C, "source_total_inductance_H": L_source},
            ),
        },
        unresolved_parameters={
            "measured_waveform_peak_time": _underdetermined(
                "measured_current_waveform_digitization",
                "The source target is not yet digitized into a typed time-current series.",
            ),
            "startup_initial_state": _underdetermined(
                "breakdown/preionization/sheath-liftoff observables",
                "Circuit and geometry do not determine the initial plasma state.",
            ),
        },
        consistency_checks={
            "stored_energy_CV_vs_source": _consistency_entry(
                calculated=E_calc,
                source=E_source,
                unit="J",
                tolerance_fraction=0.02,
            ),
            "source_L_vs_source_theoretical_peak_current": _consistency_entry(
                calculated=I_ideal,
                source=I_source,
                unit="A",
                tolerance_fraction=0.05,
                contradiction_note=(
                    "The source inductance and theoretical peak current do not "
                    "match the undamped LC formula at the listed maximum voltage; "
                    "keep source inductance and current as separate observables."
                ),
            ),
        },
        deck_fill_candidates={
            "capacitance_F": _known(C, "F", "source circuit table"),
            "voltage_V": _known(20.0e3, "V", "engineering deck operating voltage"),
            "inductance_H": _known(L_source, "H", "source total inductance"),
            "resistance_ohm": _known(
                float(circuit["design_resistance_ohm"]),
                "ohm",
                "source design resistance",
            ),
            "pressure_Pa": _known(
                float(target["gas"]["measured_shot_pressure_Pa"]),
                "Pa",
                "source measured-shot pressure",
            ),
        },
    )


def _compact_chinese_packet(target: dict[str, Any]) -> dict[str, Any]:
    circuit = target["circuit"]
    operating = target["operating_targets"]
    C = float(circuit["capacitance_total_F"])
    voltage_range = tuple(float(item) for item in circuit["charging_voltage_range_V"])
    I = float(circuit["delivered_current_A_approx"])
    L_range = [
        current_implied_inductance_H(
            capacitance_F=C,
            voltage_V=voltage,
            peak_current_A=I,
        )
        for voltage in voltage_range
    ]
    energy_range = [bank_energy_J(C, voltage) for voltage in voltage_range]
    deck_voltage = max(voltage_range)
    deck_L = current_implied_inductance_H(
        capacitance_F=C,
        voltage_V=deck_voltage,
        peak_current_A=I,
    )
    return _machine_packet(
        machine_id="compact_chinese_dpf_2018",
        device=str(target["device"]),
        source_status="may15_user_verified_knowledge_reference_source",
        source_references=(_source_reference(target),),
        known_parameters={
            "capacitance_F": _known(C, "F", "source circuit table"),
            "charging_voltage_range_V": _known(
                list(voltage_range),
                "V",
                "source operating range",
            ),
            "delivered_current_A_approx": _known(
                I,
                "A",
                "source approximate delivered current",
            ),
            "reported_pressure_Pa": _known(
                float(operating["reported_pressure_Pa"]),
                "Pa",
                "source operating target",
            ),
        },
        derived_parameters={
            "bank_energy_range_J": _bracketed(
                energy_range,
                "J",
                formula="0.5 * capacitance_F * charging_voltage_range_V**2",
                inputs={
                    "capacitance_F": C,
                    "charging_voltage_range_V": list(voltage_range),
                },
            ),
            "current_implied_inductance_range_H": _bracketed(
                L_range,
                "H",
                formula="capacitance_F * (charging_voltage_range_V / delivered_current_A_approx)**2",
                inputs={
                    "capacitance_F": C,
                    "charging_voltage_range_V": list(voltage_range),
                    "delivered_current_A_approx": I,
                },
                notes=(
                    "This is an ideal LC range from approximate delivered "
                    "current; dynamic plasma inductance remains unresolved."
                ),
            ),
            "pressure_range_torr": _direct(
                [
                    float(item) / TORR_TO_PA
                    for item in operating["optimum_pressure_Pa_range"]
                ],
                "torr",
                formula="pressure_Pa / TORR_TO_PA",
                inputs={
                    "optimum_pressure_Pa_range": operating[
                        "optimum_pressure_Pa_range"
                    ],
                },
            ),
        },
        unresolved_parameters={
            "resistance_ohm": _underdetermined(
                "ringdown current decay or source resistance",
                "Capacitance, voltage, and one current amplitude do not uniquely solve resistance.",
            ),
            "startup_initial_state": _underdetermined(
                "breakdown/preionization/sheath-liftoff observables",
                "The source gives focus timing and pressure/yield targets, not the initial BVP.",
            ),
        },
        consistency_checks={},
        deck_fill_candidates={
            "capacitance_F": _known(C, "F", "source circuit table"),
            "voltage_V": _known(deck_voltage, "V", "upper end of source voltage range"),
            "inductance_H": _direct(
                deck_L,
                "H",
                formula="capacitance_F * (20 kV / delivered_current_A_approx)**2",
                inputs={
                    "capacitance_F": C,
                    "voltage_V": deck_voltage,
                    "delivered_current_A_approx": I,
                },
                notes="Matches the existing executable engineering deck candidate.",
            ),
            "pressure_Pa": _known(
                float(operating["reported_pressure_Pa"]),
                "Pa",
                "source reported pressure",
            ),
        },
    )


def _willenborg_packet(target: dict[str, Any]) -> dict[str, Any]:
    circuit = target["circuit"]
    startup = target["startup_constraints"]
    C_source = float(circuit["capacitance_F"])
    E_20kV = float(circuit["stored_energy_J_at_20kV"])
    V_rated = float(circuit["rated_voltage_V"])
    C_from_E = capacitance_from_bank_energy_F(E_20kV, V_rated)
    system_tq = float(circuit["system_quarter_cycle_s_approx"])
    L_from_tq = quarter_cycle_implied_inductance_H(
        capacitance_F=C_source,
        quarter_cycle_s=system_tq,
    )
    focus_delay_sum = (
        float(startup["breakdown_delay_s_approx"])
        + float(startup["sheath_travel_time_s_approx"])
    )
    return _machine_packet(
        machine_id="willenborg_hendricks_1977_startup_design",
        device=str(target["device"]),
        source_status="may15_user_verified_knowledge_reference_source",
        source_references=(_source_reference(target),),
        known_parameters={
            "capacitance_F": _known(C_source, "F", "source inferred capacitance"),
            "rated_voltage_V": _known(V_rated, "V", "source bank rating"),
            "operated_voltage_V_range": _known(
                circuit["operated_voltage_V_range"],
                "V",
                "source operation range",
            ),
            "stored_energy_J_at_20kV": _known(E_20kV, "J", "source bank energy"),
            "total_system_inductance_H_approx": _known(
                float(circuit["total_system_inductance_H_approx"]),
                "H",
                "source timing estimate",
            ),
            "average_device_impedance_ohm_approx": _known(
                float(circuit["average_device_impedance_ohm_approx"]),
                "ohm",
                "source device impedance estimate",
            ),
        },
        derived_parameters={
            "capacitance_from_energy_and_voltage_F": _direct(
                C_from_E,
                "F",
                formula="2 * stored_energy_J_at_20kV / rated_voltage_V**2",
                inputs={
                    "stored_energy_J_at_20kV": E_20kV,
                    "rated_voltage_V": V_rated,
                },
            ),
            "system_inductance_from_quarter_cycle_H": _direct(
                L_from_tq,
                "H",
                formula="((2 * system_quarter_cycle_s_approx / pi)**2) / capacitance_F",
                inputs={
                    "system_quarter_cycle_s_approx": system_tq,
                    "capacitance_F": C_source,
                },
            ),
            "focus_delay_from_breakdown_plus_travel_s": _direct(
                focus_delay_sum,
                "s",
                formula="breakdown_delay_s_approx + sheath_travel_time_s_approx",
                inputs={
                    "breakdown_delay_s_approx": startup[
                        "breakdown_delay_s_approx"
                    ],
                    "sheath_travel_time_s_approx": startup[
                        "sheath_travel_time_s_approx"
                    ],
                },
            ),
        },
        unresolved_parameters={
            "surface_flashover_state": _underdetermined(
                "surface flashover equations/material secondary emission",
                "Historical timing constraints do not uniquely define the plasma launch state.",
            ),
            "preionization_density": _underdetermined(
                "preionization measurement or breakdown BVP",
                "No electron-density initial condition is supplied.",
            ),
        },
        consistency_checks={
            "capacitance_energy_voltage_vs_source": _consistency_entry(
                calculated=C_from_E,
                source=C_source,
                unit="F",
                tolerance_fraction=0.02,
            ),
            "system_quarter_cycle_L_vs_source": _consistency_entry(
                calculated=L_from_tq,
                source=float(circuit["total_system_inductance_H_approx"]),
                unit="H",
                tolerance_fraction=0.05,
            ),
            "focus_delay_sum_vs_source": _consistency_entry(
                calculated=focus_delay_sum,
                source=float(startup["focus_delay_s_approx"]),
                unit="s",
                tolerance_fraction=0.05,
            ),
        },
        deck_fill_candidates={
            "capacitance_F": _known(C_source, "F", "source inferred capacitance"),
            "voltage_V": _known(19.0e3, "V", "upper operated voltage"),
            "inductance_H": _direct(
                L_from_tq,
                "H",
                formula="quarter-cycle inferred system inductance",
                inputs={
                    "system_quarter_cycle_s_approx": system_tq,
                    "capacitance_F": C_source,
                },
            ),
            "resistance_ohm": _known(
                float(circuit["average_device_impedance_ohm_approx"]),
                "ohm",
                "source average device impedance",
            ),
            "pressure_Pa": _known(
                float(startup["conditioning_pressure_torr"]) * TORR_TO_PA,
                "Pa",
                "source conditioning pressure",
            ),
        },
    )


def _gv_shot_packet(
    row: dict[str, Any],
    *,
    include_waveform: bool,
    series: str,
    root: str | Path,
    require_hash_match: bool,
) -> dict[str, Any]:
    geometry = row["geometry_mm"]
    circuit = row["circuit"]
    gas = row["gas"]
    C = float(circuit["capacitance_uF"]) * 1.0e-6
    V = float(circuit["voltage_kV"]) * 1.0e3
    L = float(circuit["inductance_nH"]) * 1.0e-9
    R = float(circuit["resistance_milliohm"]) * 1.0e-3
    known = {
        "capacitance_F": _known(C, "F", "GV verified input deck"),
        "voltage_V": _known(V, "V", "GV verified input deck"),
        "static_inductance_H": _known(L, "H", "GV verified input deck"),
        "resistance_ohm": _known(R, "ohm", "GV verified input deck"),
        "fitted_pressure_Pa": _known(
            float(gas["fitted_pressure_torr"]) * TORR_TO_PA,
            "Pa",
            "GV verified input deck fitted pressure",
        ),
        "anode_radius_m": _known(
            float(geometry["anode_radius"]) * 1.0e-3,
            "m",
            "GV verified geometry",
        ),
        "cathode_radius_m": _known(
            float(geometry["cathode_radius"]) * 1.0e-3,
            "m",
            "GV verified geometry",
        ),
        "anode_length_m": _known(
            float(geometry["anode_length"]) * 1.0e-3,
            "m",
            "GV verified geometry",
        ),
        "insulator_length_m": _known(
            float(geometry["insulator_length"]) * 1.0e-3,
            "m",
            "GV verified geometry",
        ),
    }
    derived = {
        "bank_energy_J": _direct(
            bank_energy_J(C, V),
            "J",
            formula="0.5 * capacitance_F * voltage_V**2",
            inputs={"capacitance_F": C, "voltage_V": V},
        ),
        "ideal_peak_current_from_static_L_A": _direct(
            ideal_lc_peak_current_A(
                capacitance_F=C,
                voltage_V=V,
                inductance_H=L,
            ),
            "A",
            formula="voltage_V * sqrt(capacitance_F / static_inductance_H)",
            inputs={
                "capacitance_F": C,
                "voltage_V": V,
                "static_inductance_H": L,
            },
            notes=(
                "Ideal static-circuit upper-bound; measured current includes "
                "switching and plasma dynamics."
            ),
        ),
        "ideal_static_lc_quarter_cycle_s": _direct(
            ideal_lc_quarter_cycle_s(capacitance_F=C, inductance_H=L),
            "s",
            formula="0.5 * pi * sqrt(static_inductance_H * capacitance_F)",
            inputs={"capacitance_F": C, "static_inductance_H": L},
        ),
    }
    consistency: dict[str, Any] = {}
    unresolved = {
        "startup_initial_state": _underdetermined(
            "breakdown/preionization/sheath-liftoff observables",
            "GV input decks and workbook current traces do not define the initial plasma state.",
        ),
        "dynamic_plasma_impedance": _underdetermined(
            "field/plasma state history or accepted inverse circuit model",
            "A measured terminal current alone does not uniquely split static and plasma inductance/resistance.",
        ),
    }
    if include_waveform:
        _attach_gv_waveform_derivations(
            row=row,
            derived=derived,
            consistency_checks=consistency,
            capacitance_F=C,
            voltage_V=V,
            static_inductance_H=L,
            series=series,
            root=root,
            require_hash_match=require_hash_match,
        )
    else:
        derived["waveform_peak_current_A"] = _entry(
            None,
            "A",
            STATUS_UNAVAILABLE,
            notes="GV waveform extraction disabled for this packet.",
        )
    return _machine_packet(
        machine_id=f"gv_{row['shot_id']}",
        device=str(row["device"]),
        source_status="gv_user_verified_local_source_candidate",
        source_references=(_gv_source_reference(row, root),),
        known_parameters=known,
        derived_parameters=derived,
        unresolved_parameters=unresolved,
        consistency_checks=consistency,
        deck_fill_candidates={
            "capacitance_F": _known(C, "F", "GV verified input deck"),
            "voltage_V": _known(V, "V", "GV verified input deck"),
            "inductance_H": _known(L, "H", "GV verified input deck"),
            "resistance_ohm": _known(R, "ohm", "GV verified input deck"),
            "pressure_Pa": _known(
                float(gas["fitted_pressure_torr"]) * TORR_TO_PA,
                "Pa",
                "GV fitted pressure",
            ),
        },
    )


def _attach_gv_waveform_derivations(
    *,
    row: dict[str, Any],
    derived: dict[str, Any],
    consistency_checks: dict[str, Any],
    capacitance_F: float,
    voltage_V: float,
    static_inductance_H: float,
    series: str,
    root: str | Path,
    require_hash_match: bool,
) -> None:
    from dpf.first_principles.gv_waveforms import extract_gv_current_waveform_packet

    try:
        packet = extract_gv_current_waveform_packet(
            str(row["shot_id"]),
            series=series,
            root=root,
            require_hash_match=require_hash_match,
        )
    except Exception as exc:  # noqa: BLE001 - packet must fail closed with reason.
        derived["waveform_peak_current_A"] = _entry(
            None,
            "A",
            STATUS_UNAVAILABLE,
            notes=f"GV waveform extraction failed: {exc}",
        )
        return

    values = packet["digitized_series"][0]
    currents = [float(item) for item in values["y"]]
    times = [float(item) for item in values["x"]]
    max_index = max(range(len(currents)), key=lambda index: abs(currents[index]))
    peak_current_A = abs(currents[max_index]) * 1.0e3
    peak_time_s = times[max_index] * 1.0e-6
    L_from_I = current_implied_inductance_H(
        capacitance_F=capacitance_F,
        voltage_V=voltage_V,
        peak_current_A=peak_current_A,
    )
    derived["waveform_peak_current_A"] = _entry(
        peak_current_A,
        "A",
        STATUS_WAVEFORM,
        formula="abs(workbook_current_max_kA) * 1e3",
        inputs={
            "shot_id": row["shot_id"],
            "series": values["name"],
            "workbook_series_sha256": values["series_sha256"],
        },
        notes="Measured workbook current is a candidate observable, not a drive waveform.",
    )
    derived["waveform_peak_time_s"] = _entry(
        peak_time_s,
        "s",
        STATUS_WAVEFORM,
        formula="workbook_peak_time_us * 1e-6",
        inputs={"shot_id": row["shot_id"], "series": values["name"]},
        notes=(
            "Time zero is workbook/diagnostic dependent; use as a candidate "
            "quarter-cycle observable only after review."
        ),
    )
    derived["waveform_current_implied_inductance_H"] = _entry(
        L_from_I,
        "H",
        STATUS_WAVEFORM,
        formula="capacitance_F * (voltage_V / waveform_peak_current_A)**2",
        inputs={
            "capacitance_F": capacitance_F,
            "voltage_V": voltage_V,
            "waveform_peak_current_A": peak_current_A,
        },
        notes=(
            "Effective ideal-current inductance from measured peak; dynamic "
            "plasma loading prevents using it as a unique source inductance."
        ),
    )
    if peak_time_s > 0.0:
        derived["waveform_quarter_cycle_implied_inductance_H"] = _entry(
            quarter_cycle_implied_inductance_H(
                capacitance_F=capacitance_F,
                quarter_cycle_s=peak_time_s,
            ),
            "H",
            STATUS_WAVEFORM,
            formula="((2 * waveform_peak_time_s / pi)**2) / capacitance_F",
            inputs={
                "capacitance_F": capacitance_F,
                "waveform_peak_time_s": peak_time_s,
            },
            notes="Candidate only until timing origin and switch delay are reviewed.",
        )
    else:
        derived["waveform_quarter_cycle_implied_inductance_H"] = _entry(
            None,
            "H",
            STATUS_UNDETERMINED,
            required_observable="positive current-rise time from discharge start",
            notes="Workbook peak time is not positive relative to its stored time origin.",
        )
    consistency_checks["waveform_current_implied_L_vs_static_source_L"] = (
        _comparison_entry(
            calculated=L_from_I,
            source=static_inductance_H,
            unit="H",
            status=STATUS_WAVEFORM,
            notes=(
                "This mismatch is expected for dynamic plasma-loaded current; "
                "the packet records it for deck tuning but does not overwrite "
                "the source static inductance."
            ),
        )
    )


def _machine_packet(
    *,
    machine_id: str,
    device: str,
    source_status: str,
    source_references: tuple[dict[str, Any], ...],
    known_parameters: dict[str, Any],
    derived_parameters: dict[str, Any],
    unresolved_parameters: dict[str, Any],
    consistency_checks: dict[str, Any],
    deck_fill_candidates: dict[str, Any],
) -> dict[str, Any]:
    status_counts = _status_counts(
        (
            {
                "known_parameters": known_parameters,
                "derived_parameters": derived_parameters,
                "unresolved_parameters": unresolved_parameters,
                "consistency_checks": consistency_checks,
                "deck_fill_candidates": deck_fill_candidates,
            },
        )
    )
    return {
        "machine_id": machine_id,
        "device": device,
        "source_status": source_status,
        "source_references": list(source_references),
        "known_parameters": known_parameters,
        "derived_parameters": derived_parameters,
        "unresolved_parameters": unresolved_parameters,
        "consistency_checks": consistency_checks,
        "deck_fill_candidates": deck_fill_candidates,
        "status_counts": status_counts,
        "can_seed_experimental_deck": True,
        "can_support_first_principles_acceptance": False,
    }


def _known(value: Any, unit: str, source: str) -> dict[str, Any]:
    return _entry(value, unit, STATUS_KNOWN, source=source)


def _direct(
    value: Any,
    unit: str,
    *,
    formula: str,
    inputs: dict[str, Any],
    notes: str | None = None,
) -> dict[str, Any]:
    return _entry(
        value,
        unit,
        STATUS_DIRECT,
        formula=formula,
        inputs=inputs,
        notes=notes,
    )


def _bracketed(
    value: Any,
    unit: str,
    *,
    formula: str,
    inputs: dict[str, Any],
    notes: str | None = None,
) -> dict[str, Any]:
    return _entry(
        value,
        unit,
        STATUS_BRACKETED,
        formula=formula,
        inputs=inputs,
        notes=notes,
    )


def _underdetermined(required_observable: str, notes: str) -> dict[str, Any]:
    return _entry(
        None,
        "not_applicable",
        STATUS_UNDETERMINED,
        required_observable=required_observable,
        notes=notes,
    )


def _entry(
    value: Any,
    unit: str,
    status: str,
    *,
    source: str | None = None,
    formula: str | None = None,
    inputs: dict[str, Any] | None = None,
    required_observable: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    entry = {
        "value": value,
        "unit": unit,
        "status": status,
    }
    if source is not None:
        entry["source"] = source
    if formula is not None:
        entry["formula"] = formula
    if inputs is not None:
        entry["inputs"] = inputs
    if required_observable is not None:
        entry["required_observable"] = required_observable
    if notes is not None:
        entry["notes"] = notes
    return entry


def _consistency_entry(
    *,
    calculated: float,
    source: float,
    unit: str,
    tolerance_fraction: float,
    contradiction_note: str | None = None,
) -> dict[str, Any]:
    fraction = _fraction_difference(calculated, source)
    status = STATUS_DIRECT
    notes = "calculated value matches source within tolerance"
    if fraction > tolerance_fraction:
        status = STATUS_CONTRADICTION
        notes = contradiction_note or "calculated value differs from source tolerance"
    return _comparison_entry(
        calculated=calculated,
        source=source,
        unit=unit,
        status=status,
        tolerance_fraction=tolerance_fraction,
        notes=notes,
    )


def _comparison_entry(
    *,
    calculated: float,
    source: float,
    unit: str,
    status: str,
    tolerance_fraction: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "calculated_value": calculated,
        "source_value": source,
        "unit": unit,
        "fraction_difference": _fraction_difference(calculated, source),
        "status": status,
        "tolerance_fraction": tolerance_fraction,
        "notes": notes,
    }


def _source_reference(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": target["source"],
        "sha256": target["source_sha256"],
        "source_lines": target.get("source_lines", {}),
    }


def _gv_source_reference(row: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    return {
        "input_deck": {
            "path": str(root_path / str(row["input_file"])),
            "sha256": row["input_sha256"],
        },
        "workbook": {
            "path": str(root_path / str(row["xlsx_file"])),
            "sha256": row["xlsx_sha256"],
        },
        "gv_reduced_model_output_not_used_for_inference": {
            "path": str(root_path / str(row["txt_file"])),
            "sha256": row["txt_sha256"],
        },
    }


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _fraction_difference(a: float, b: float) -> float:
    denominator = max(abs(float(a)), abs(float(b)), 1.0e-300)
    return abs(float(a) - float(b)) / denominator


def _status_counts(packets: Any) -> dict[str, int]:
    counts: dict[str, int] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            status = value.get("status")
            if isinstance(status, str):
                counts[status] = counts.get(status, 0) + 1
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(packets)
    return dict(sorted(counts.items()))


def _unresolved_parameters(machines: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return _entries_by_status(machines, STATUS_UNDETERMINED)


def _entries_by_status(
    machines: dict[str, dict[str, Any]],
    status: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []

    def visit(machine_id: str, section: str, name: str, value: Any) -> None:
        if isinstance(value, dict):
            if value.get("status") == status:
                matches.append(
                    {
                        "machine_id": machine_id,
                        "section": section,
                        "parameter": name,
                        "entry": value,
                    }
                )
            for child_name, child in value.items():
                if isinstance(child, (dict, list, tuple)):
                    visit(machine_id, section, str(child_name), child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(machine_id, section, f"{name}[{index}]", child)

    for machine_id, machine in machines.items():
        for section in (
            "known_parameters",
            "derived_parameters",
            "unresolved_parameters",
            "consistency_checks",
            "deck_fill_candidates",
        ):
            section_values = machine.get(section, {})
            if isinstance(section_values, dict):
                for name, value in section_values.items():
                    visit(machine_id, section, str(name), value)
    return matches
