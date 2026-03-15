"""DPF-Unified Web Interface — Dense Plasma Focus Simulator.

Lee model + MHD backends, 3D animated playback, physics narrative with math.
Comparison mode, neutron yield, phase portraits, save/load configurations.
Run: python3 app.py
"""
from __future__ import annotations

import atexit
import csv
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app_anim import create_animated_3d, create_animated_mhd
from app_calibrate import auto_calibrate, format_calibration_markdown, get_published_params
from app_compare import (
    add_to_comparison,
    clear_comparison,
    comparison_summary,
    load_config,
    remove_from_comparison,
    save_config,
)
from app_engine import GAS_SPECIES, run_simulation_core
from app_mhd import BACKENDS, MHD_GRID_PRESETS, create_mhd_fields_fig, run_mhd_simulation
from app_narrative import generate_narrative
from app_plots import (
    create_3d_plasma_fig,
    create_comparison_fig,
    create_phase_portrait,
    create_physics_fig,
    create_schematic_fig,
    create_waveform_fig,
    parse_experimental_csv,
    validate_experimental_csv,
)
from app_sweep import (
    create_2d_sweep_fig,
    create_sweep_fig,
    format_sweep_markdown,
    run_2d_sweep,
    run_parameter_sweep,
)
from app_validation import format_validation_markdown, validate_against_published
from dpf.presets import _PRESETS, get_preset, list_presets

RUNTIME_PER_US = {
    ("lee", "coarse"): 0.02, ("lee", "medium"): 0.02, ("lee", "fine"): 0.02,
    ("metal_plm", "coarse"): 0.8, ("metal_plm", "medium"): 4.0, ("metal_plm", "fine"): 25.0,
    ("metal_weno5", "coarse"): 2.5, ("metal_weno5", "medium"): 15.0, ("metal_weno5", "fine"): 90.0,
    ("python", "coarse"): 1.5, ("python", "medium"): 8.0, ("python", "fine"): 50.0,
    ("athena", "coarse"): 0.5, ("athena", "medium"): 3.0, ("athena", "fine"): 18.0,
}

FIDELITY = {
    "lee": "0D", "metal_plm": "6.5/10", "metal_weno5": "8.7/10",
    "athena": "9.0/10", "python": "7.0/10",
}

BACKEND_HELP = {
    "lee": "Fastest option. 0D lumped-parameter model (Lee snowplow + RLC circuit). "
           "Computes I(t) waveforms, sheath dynamics, and pinch radius. No spatial resolution.",
    "metal_plm": "Apple Metal GPU solver. 2nd-order accurate, fast for parameter scans. "
                 "PLM reconstruction, HLL Riemann solver, SSP-RK2 time integration.",
    "metal_weno5": "Apple Metal GPU solver. 5th-order WENO-Z + HLLD 4-wave solver + "
                   "SSP-RK3. High fidelity, significantly longer runtime.",
    "athena": "Princeton Athena++ C++ engine. PPM reconstruction, HLLD solver. "
              "Reference-quality MHD. Runs via compiled binary.",
    "python": "Pure Python MHD solver (NumPy). WENO5 + HLLD + SSP-RK3. "
              "Full physics including resistive MHD. Moderate speed.",
}


def get_preset_choices() -> list[tuple[str, str]]:
    presets = list_presets()
    return [(f"{p['device']} -- {p['description']}", p["name"]) for p in presets
            if p.get("geometry") == "cylindrical"]


def get_gas_choices() -> list[tuple[str, str]]:
    return [(v["name"], k) for k, v in GAS_SPECIES.items()]


def get_backend_choices() -> list[tuple[str, str]]:
    return [(desc, key) for key, desc in BACKENDS.items()]


def get_grid_choices() -> list[tuple[str, str]]:
    return [(f"{k} ({v[0]}x{v[1]}x{v[2]})", k) for k, v in MHD_GRID_PRESETS.items()]


def get_device_info(preset_name: str) -> str:
    if not preset_name:
        return ""
    preset = get_preset(preset_name)
    meta = _PRESETS.get(preset_name, {}).get("_meta", {})
    cc = preset["circuit"]
    E_kJ = 0.5 * cc["C"] * cc["V0"] ** 2 / 1e3
    sc = preset.get("snowplow", {})
    lines = [
        f"### {meta.get('device', preset_name)}",
        f"*{meta.get('description', '')}*", "",
        "| Parameter | Value |", "|-----------|-------|",
        f"| Charging Voltage | {cc['V0']/1e3:.0f} kV |",
        f"| Capacitance | {cc['C']*1e6:.0f} uF |",
        f"| Bank Energy | {E_kJ:.0f} kJ |",
        f"| Inductance L0 | {cc['L0']*1e9:.1f} nH |",
        f"| Resistance R0 | {cc.get('R0',0)*1e3:.1f} mOhm |",
        f"| Anode Radius | {cc['anode_radius']*1e3:.1f} mm |",
        f"| Cathode Radius | {cc['cathode_radius']*1e3:.1f} mm |",
    ]
    if sc:
        lines.extend([
            f"| Anode Length | {sc.get('anode_length',0)*1e3:.0f} mm |",
            f"| Current Fraction | {sc.get('current_fraction',0.7):.2f} |",
            f"| Mass Fraction | {sc.get('mass_fraction',0.15):.3f} |",
        ])
    ref = meta.get("reference", "")
    if ref:
        lines.append(f"\n*Ref: {ref}*")
    return "\n".join(lines)


def load_preset_values(preset_name: str):
    if not preset_name:
        return [None] * 12
    preset = get_preset(preset_name)
    cc = preset["circuit"]
    sc = preset.get("snowplow", {})
    return [
        cc["V0"] / 1e3, cc["C"] * 1e6, cc["L0"] * 1e9,
        cc.get("R0", 0) * 1e3,
        cc["anode_radius"] * 1e3, cc["cathode_radius"] * 1e3,
        sc.get("anode_length", 0.16) * 1e3,
        sc.get("current_fraction", 0.7), sc.get("mass_fraction", 0.15),
        cc.get("crowbar_enabled", False),
        cc.get("crowbar_resistance", 0) * 1e3,
        sc.get("fill_pressure_Pa", 400) / 133.322,
    ]


def on_settings_change(backend: str, grid_preset: str, sim_time_us: float):
    is_lee = backend == "lee"
    rate = RUNTIME_PER_US.get(
        (backend, grid_preset), RUNTIME_PER_US.get((backend, "medium"), 1.0),
    )
    est_s = rate * sim_time_us
    if est_s < 2:
        est = "< 2 seconds"
    elif est_s < 60:
        est = f"~{est_s:.0f} seconds"
    elif est_s < 3600:
        est = f"~{est_s/60:.1f} minutes"
    else:
        est = f"~{est_s/3600:.1f} hours (consider reducing grid or sim time)"

    grid = MHD_GRID_PRESETS.get(grid_preset, (32, 32, 64))
    fid = FIDELITY.get(backend, "?")
    help_text = BACKEND_HELP.get(backend, "")

    if is_lee:
        info = f"**{est}** | Lee model (0D) | {help_text}"
    else:
        total_cells = grid[0] * grid[1] * grid[2]
        info = (
            f"**{est}** | Grid: {grid[0]}x{grid[1]}x{grid[2]} "
            f"= {total_cells:,} cells | Fidelity: {fid}\n\n*{help_text}*"
        )
    return [gr.update(visible=is_lee), gr.update(visible=not is_lee), info]


def _validate_inputs(
    anode_r: float, cathode_r: float, V0_kV: float, C_uF: float,
    L0_nH: float, sim_time_us: float,
) -> str | None:
    if anode_r >= cathode_r:
        return f"Anode radius ({anode_r} mm) must be < cathode radius ({cathode_r} mm)."
    if V0_kV <= 0 or C_uF <= 0 or L0_nH <= 0:
        return "Charging voltage, capacitance, and inductance must all be positive."
    if sim_time_us <= 0:
        return "Simulation time must be positive."
    return None


def _build_metrics(data: dict, backend: str, val: dict | None = None) -> str:
    fid = FIDELITY.get(backend, "")
    fid_str = f" | Fidelity: {fid}" if fid != "0D" else ""
    parts = [f"**I_peak = {data['I_peak']:.3f} MA** at {data['t_peak']:.1f} us"]

    if val:
        dI = val["I_peak_dev_pct"]
        label = "PASS" if dI <= 5 else "FAIR" if dI <= 15 else "POOR" if dI <= 30 else "FAIL"
        parts.append(f"vs. expt: **{dI:.0f}% ({label})**")

    if data.get("has_snowplow") and data["dip_pct"] > 1:
        parts.append(f"Current dip: **{data['dip_pct']:.0f}%**")
        sp = data.get("snowplow_obj")
        if sp:
            r_p = sp.shock_radius * 1e3
            b_mm = data["circuit"]["cathode_radius"] * 1e3
            parts.append(f"Pinch: **{r_p:.1f} mm** ({b_mm/r_p:.0f}:1)")

    if data.get("scaling"):
        parts.append(f"T_stag: **{data['scaling']['T_stag_keV']:.1f} keV**")

    ny = data.get("neutron_yield")
    if ny and ny["Y_neutron"] > 0:
        bt_pct = ny.get("bt_fraction", 0) * 100
        V_kV = ny.get("V_pinch_kV", 0)
        rad_cool = ny.get("rad_cooling_factor", 1.0)
        extra = ""
        if V_kV > 1:
            extra += f", V_pinch={V_kV:.0f} kV"
        if rad_cool < 0.95:
            extra += f", T_rad/T_B={rad_cool:.2f}"
        parts.append(f"D-D yield: **{ny['Y_neutron']:.2e} n** ({bt_pct:.0f}% BT{extra})")

    if data.get("has_mhd"):
        rho_max = data.get("rho_max", [])
        rho0 = data.get("rho0", 1)
        if len(rho_max) > 0 and rho0 > 0:
            parts.append(f"Density ratio: {float(np.max(rho_max))/rho0:.1f}x")

    parts.append(
        f"{data.get('device', '?')} | {data.get('gas', {}).get('name', '?')} | "
        f"{backend} | {data['n_steps']} steps in {data['elapsed_s']:.2f}s{fid_str}"
    )
    return " | ".join(parts)


_TEMP_DIR = tempfile.mkdtemp(prefix="dpf_ui_")
os.environ["DPF_TEMP_DIR"] = _TEMP_DIR
atexit.register(shutil.rmtree, _TEMP_DIR, True)


def _export_csv(data: dict) -> str | None:
    if data is None:
        return None
    buf = io.StringIO()
    writer = csv.writer(buf)
    t = data["t_us"]
    writer.writerow(["t_us", "I_MA", "V_kV", "L_p_nH", "z_mm", "r_mm",
                      "phase", "E_cap_kJ", "E_ind_kJ", "E_res_kJ"])
    for i in range(len(t)):
        writer.writerow([
            f"{t[i]:.4f}", f"{data['I_MA'][i]:.6f}", f"{data['V_kV'][i]:.4f}",
            f"{data['L_p_nH'][i]:.2f}", f"{data['z_mm'][i]:.3f}",
            f"{data['r_mm'][i]:.3f}", data["phases"][i],
            f"{data['E_cap_kJ'][i]:.4f}", f"{data['E_ind_kJ'][i]:.4f}",
            f"{data['E_res_kJ'][i]:.4f}",
        ])
    path = os.path.join(_TEMP_DIR, f"dpf_{os.getpid()}_{id(data)}.csv")
    with open(path, "w") as f:
        f.write(buf.getvalue())
    return path


def run_simulation(
    backend, grid_preset, preset_name, sim_time_us, gas_key,
    V0_kV, C_uF, L0_nH, R0_mOhm,
    anode_r, cathode_r, anode_len,
    fc, fm, crowbar_on, crowbar_R, pressure,
    comparison_runs,
    experimental_csv=None,
    progress=gr.Progress(),  # noqa: B008
):
    err = _validate_inputs(anode_r, cathode_r, V0_kV, C_uF, L0_nH, sim_time_us)
    if err:
        raise gr.Error(err)

    try:
        if backend == "lee":
            data = run_simulation_core(
                preset_name=preset_name, sim_time_us=sim_time_us, gas_key=gas_key,
                V0_kV=V0_kV, C_uF=C_uF, L0_nH=L0_nH, R0_mOhm=R0_mOhm,
                anode_r_mm=anode_r, cathode_r_mm=cathode_r, anode_len_mm=anode_len,
                fc=fc, fm=fm, crowbar_on=crowbar_on, crowbar_R_mOhm=crowbar_R,
                pressure_torr=pressure, progress_fn=progress,
            )
        else:
            data = run_mhd_simulation(
                backend=backend, grid_preset=grid_preset,
                preset_name=preset_name, sim_time_us=sim_time_us, gas_key=gas_key,
                V0_kV=V0_kV, pressure_torr=pressure,
                C_uF=C_uF, L0_nH=L0_nH, R0_mOhm=R0_mOhm,
                anode_r_mm=anode_r, cathode_r_mm=cathode_r, anode_len_mm=anode_len,
                progress_fn=progress,
            )
    except Exception as exc:
        raise gr.Error(f"Simulation failed ({backend}): {exc}") from exc

    exp_data: dict | None = None
    if experimental_csv is not None:
        csv_path = Path(experimental_csv) if isinstance(experimental_csv, str) else Path(experimental_csv.name)
        try:
            csv_text = csv_path.read_text()
            validate_experimental_csv(csv_text)
            exp_data = parse_experimental_csv(csv_text)
        except gr.Error:
            raise
        except Exception as exc:
            gr.Warning(f"Could not parse experimental CSV: {exc}")
            exp_data = None

    fig_wave = create_waveform_fig(data, experimental_data=exp_data)
    fig_phys = create_physics_fig(data)
    fig_portrait = create_phase_portrait(data)

    if data.get("has_mhd"):
        fig_schem = create_mhd_fields_fig(data)
        if data.get("has_snowplow"):
            fig_3d = create_3d_plasma_fig(data)
        else:
            fig_3d = create_mhd_fields_fig(data)
            gr.Info("MHD mode: 3D Plasma tab shows 2D field maps (no snowplow trajectory).")
        anim_fig = create_animated_mhd(data)
        full_html = anim_fig.to_html(
            full_html=True, include_plotlyjs="cdn", config={"responsive": True},
        )
        escaped = full_html.replace("&", "&amp;").replace('"', "&quot;")
        anim_html = (
            f'<iframe srcdoc="{escaped}" '
            f'style="width:100%;height:580px;border:none;background:#111;"></iframe>'
        )
    else:
        fig_schem = create_schematic_fig(data)
        fig_3d = create_3d_plasma_fig(data)
        anim_fig = create_animated_3d(data)
        full_html = anim_fig.to_html(
            full_html=True, include_plotlyjs="cdn", config={"responsive": True},
        )
        escaped = full_html.replace("&", "&amp;").replace('"', "&quot;")
        anim_html = (
            f'<iframe srcdoc="{escaped}" '
            f'style="width:100%;height:620px;border:none;background:#111;"></iframe>'
        )

    narrative = generate_narrative(data)
    val = validate_against_published(data, preset_name)
    val_md = format_validation_markdown(val)
    if val_md:
        narrative = narrative + "\n\n" + val_md
    metrics = _build_metrics(data, backend, val)
    csv_path = _export_csv(data)

    runs = add_to_comparison(comparison_runs or [], data, backend)
    fig_compare = create_comparison_fig(runs)
    compare_md = comparison_summary(runs)

    return (
        metrics, narrative,
        fig_wave, fig_phys, fig_portrait, fig_schem, fig_3d, anim_html,
        fig_compare, compare_md,
        csv_path, data, runs,
    )


# ---- Build UI ----
CSS = """
.metrics-banner {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    color: #fff; font-size: 14px;
}
.metrics-banner strong { color: #90caf9; }
"""

with gr.Blocks(title="DPF-Unified Simulator") as app:
    sim_state = gr.State(None)
    comparison_state = gr.State([])

    gr.Markdown("# DPF-Unified Simulator")

    with gr.Accordion("Quick Start & Help", open=False):
        gr.Markdown("""**Dense Plasma Focus (DPF)** simulator with multi-fidelity backends.

**Workflow**: Select a device preset -> adjust parameters -> click **Run Simulation**.

**Backends** (speed vs. fidelity):
- **Lee model** — 0D lumped-parameter. Fastest (<2s). Good for parameter exploration and sweeps.
- **Metal PLM/WENO5** — Apple GPU MHD solver. 2D/3D fields. PLM is faster, WENO5 is more accurate.
- **Athena++** — Reference C++ MHD code from Princeton. Highest fidelity.
- **Python MHD** — Pure NumPy solver. Teaching/prototyping only.

**Key parameters**:
- **fc** (current fraction): fraction of circuit current carried by the plasma sheath (typical 0.6-0.8)
- **fm** (mass fraction): fraction of fill gas swept up by the sheath (typical 0.05-0.2)
- Use **Auto-Calibrate** to optimize fc/fm, or **Use Published Params** for literature values.

**Tips**: Upload experimental CSV (columns: `t_us,I_MA`) in the Waveforms tab for overlay comparison.
Parameter sweeps always use the Lee model for speed. Compare Runs stores up to 8 overlays.
""")

    gr.Markdown(
        "Dense Plasma Focus simulation with multi-fidelity backends. "
        "Lee model for fast exploration, MHD solvers for high-fidelity physics. "
        "[GitHub](https://github.com/longanisainhertaco/DPF_Unified)"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=340):
            backend_dd = gr.Dropdown(
                choices=get_backend_choices(), value="lee",
                label="Simulation Backend",
                info="Select physics fidelity vs. speed tradeoff",
            )
            grid_dd = gr.Dropdown(
                choices=get_grid_choices(), value="medium",
                label="MHD Grid Resolution", visible=False,
                info="Higher resolution = more accurate but slower",
            )
            runtime_est = gr.Markdown(
                value=f"**< 2 seconds** | Lee model (0D)\n\n*{BACKEND_HELP['lee']}*",
            )
            preset_dd = gr.Dropdown(
                choices=get_preset_choices(), value="pf1000",
                label="Device Preset",
                info="Select a known DPF device to auto-fill parameters",
            )
            device_info = gr.Markdown(value=get_device_info("pf1000"))
            gas_dd = gr.Dropdown(
                choices=get_gas_choices(), value="D2", label="Fill Gas",
                info="Gas species affects density, gamma, and radiation",
            )

            with gr.Accordion("Circuit Parameters", open=False):
                inp_V0 = gr.Number(value=27, label="Charging Voltage [kV]", minimum=1,
                                   info="Capacitor bank charging voltage")
                inp_C = gr.Number(value=1332, label="Capacitance [uF]", minimum=0.1,
                                  info="Total bank capacitance (parallel capacitors)")
                inp_L0 = gr.Number(value=33.5, label="Ext. Inductance L0 [nH]", minimum=0.1,
                                   info="External circuit inductance (cables, headers, switch)")
                inp_R0 = gr.Number(value=2.3, label="Ext. Resistance R0 [mOhm]", minimum=0,
                                   info="External circuit resistance")
                inp_crowbar = gr.Checkbox(value=True, label="Crowbar Switch")
                inp_crowbar_R = gr.Number(value=1.5, label="Crowbar Resistance [mOhm]",
                                          minimum=0, info="Resistance at V=0 crossing")

            with gr.Accordion("Electrode Geometry", open=False):
                inp_anode_r = gr.Number(value=115, label="Anode Radius [mm]", minimum=1,
                                        info="Inner electrode radius. Must be < cathode radius.")
                inp_cathode_r = gr.Number(value=160, label="Cathode Radius [mm]", minimum=2,
                                           info="Outer electrode radius. Must be > anode radius.")
                inp_anode_len = gr.Number(value=600, label="Anode Length [mm]", minimum=10,
                                          info="Rundown distance from insulator to anode tip")

            lee_params = gr.Group(visible=True)
            with lee_params, gr.Accordion("Plasma Model (Lee)", open=False):
                inp_fc = gr.Slider(0.3, 1.0, value=0.80, step=0.01,
                                   label="Current Fraction (fc)",
                                   info="Fraction of circuit current in sheath. Typical: 0.6-0.8")
                inp_fm = gr.Slider(0.01, 0.5, value=0.094, step=0.001,
                                   label="Mass Fraction (fm)",
                                   info="Fraction of fill gas swept by sheath. Typical: 0.05-0.2")
                inp_pressure = gr.Number(value=None, label="Fill Pressure [Torr]",
                                          info="Leave empty for preset default", minimum=0.1)

            sim_time = gr.Slider(1, 100, value=40, step=0.5, label="Simulation Time [us]",
                                  info="Total simulation duration in microseconds")
            with gr.Row():
                run_btn = gr.Button("Run Simulation", variant="primary", size="lg")
                cal_btn = gr.Button("Auto-Calibrate", variant="secondary", size="lg")
                pub_btn = gr.Button("Use Published Params", variant="secondary", size="lg")
            cal_output = gr.Markdown(visible=False)

            with gr.Accordion("Save / Load Configuration", open=False):
                save_btn = gr.Button("Save Current Config", size="sm")
                save_file = gr.File(label="Download Config", visible=False)
                load_file = gr.File(label="Load Config (.json)", file_types=[".json"])
                load_btn = gr.Button("Apply Loaded Config", size="sm")

            export_file = gr.File(label="Export Data (CSV)", visible=False)

        with gr.Column(scale=3):
            metrics_md = gr.Markdown(
                value="<div class='metrics-banner'>Run a simulation to see results.</div>",
                elem_classes=["metrics-banner"],
            )
            with gr.Tab("Physics Narrative"):
                narrative_md = gr.Markdown(
                    value="*Run a simulation for step-by-step physics breakdown with equations.*",
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )
            with gr.Tab("Waveforms"):
                exp_csv_upload = gr.File(
                    label="Experimental Data (CSV)", file_types=[".csv"],
                )
                fig_waveform = gr.Plot(label="Current & Voltage")
            with gr.Tab("Physics Breakdown"):
                fig_physics = gr.Plot(label="Physics")
            with gr.Tab("Phase Portrait"):
                fig_portrait = gr.Plot(label="Radial Implosion Dynamics")
            with gr.Tab("2D Fields"):
                fig_geometry = gr.Plot(label="Cross-Section / Fields")
            with gr.Tab("3D Plasma"):
                fig_3d_plot = gr.Plot(label="3D Visualization")
            with gr.Tab("3D Playback"):
                fig_anim = gr.HTML(
                    value="<div style='color:#999;padding:40px;text-align:center;'>"
                    "Run a Lee model simulation, then press Play.</div>",
                )
            with gr.Tab("Compare Runs"):
                fig_compare = gr.Plot(label="Comparison Overlay")
                compare_md = gr.Markdown("*Run multiple simulations to compare waveforms.*")
                with gr.Row():
                    remove_idx = gr.Number(value=0, label="Run # to remove (0-indexed)", precision=0,
                                           minimum=0, maximum=7)
                    remove_btn = gr.Button("Remove Run", size="sm", variant="secondary")
                    clear_btn = gr.Button("Clear All", size="sm")
            with gr.Tab("Parameter Sweep"):
                gr.Markdown(
                    "Sweep a parameter to visualize its effect on I_peak, dip, and neutron yield. "
                    "*Sweeps use the Lee model (0D) regardless of backend selection for speed.*"
                )
                with gr.Row():
                    sweep_param = gr.Dropdown(
                        choices=[("Mass Fraction (fm)", "fm"), ("Current Fraction (fc)", "fc"),
                                 ("Voltage (kV)", "V0_kV"), ("Fill Pressure (Torr)", "pressure")],
                        value="fm", label="Sweep Parameter",
                    )
                    sweep_min = gr.Number(value=0.05, label="Min")
                    sweep_max = gr.Number(value=0.30, label="Max")
                    sweep_n = gr.Slider(5, 30, value=15, step=1, label="Points")
                with gr.Row():
                    sweep_btn = gr.Button("Run 1D Sweep", variant="primary", size="sm")
                    sweep_2d_btn = gr.Button("Run 2D (fm x fc) Sweep", variant="secondary",
                                              size="sm")
                with gr.Accordion("2D Sweep Settings", open=False), gr.Row():
                    sweep_2d_fm_min = gr.Number(value=0.05, label="fm min")
                    sweep_2d_fm_max = gr.Number(value=0.30, label="fm max")
                    sweep_2d_fc_min = gr.Number(value=0.50, label="fc min")
                    sweep_2d_fc_max = gr.Number(value=0.90, label="fc max")
                    sweep_2d_n = gr.Slider(5, 20, value=10, step=1, label="Grid size")
                sweep_plot = gr.Plot(label="Sweep Results")
                sweep_md = gr.Markdown("*Click 'Run Sweep' to explore parameter space.*")

    # ---- Event wiring ----
    settings_inputs = [backend_dd, grid_dd, sim_time]
    settings_outputs = [lee_params, grid_dd, runtime_est]
    for trigger in (backend_dd.change, grid_dd.change, sim_time.change):
        trigger(fn=on_settings_change, inputs=settings_inputs, outputs=settings_outputs)

    preset_dd.change(fn=get_device_info, inputs=[preset_dd], outputs=[device_info])
    preset_dd.change(
        fn=load_preset_values, inputs=[preset_dd],
        outputs=[inp_V0, inp_C, inp_L0, inp_R0, inp_anode_r, inp_cathode_r,
                 inp_anode_len, inp_fc, inp_fm, inp_crowbar, inp_crowbar_R, inp_pressure],
    )

    all_param_inputs = [
        backend_dd, grid_dd, preset_dd, sim_time, gas_dd,
        inp_V0, inp_C, inp_L0, inp_R0,
        inp_anode_r, inp_cathode_r, inp_anode_len,
        inp_fc, inp_fm, inp_crowbar, inp_crowbar_R, inp_pressure,
    ]

    run_btn.click(
        fn=run_simulation,
        inputs=all_param_inputs + [comparison_state, exp_csv_upload],
        outputs=[
            metrics_md, narrative_md,
            fig_waveform, fig_physics, fig_portrait,
            fig_geometry, fig_3d_plot, fig_anim,
            fig_compare, compare_md,
            export_file, sim_state, comparison_state,
        ],
        concurrency_limit=2,
    )

    def refresh_waveform(sim_data: dict | None, exp_file):
        if sim_data is None:
            return None
        exp_data: dict | None = None
        if exp_file is not None:
            csv_path = Path(exp_file) if isinstance(exp_file, str) else Path(exp_file.name)
            try:
                csv_text = csv_path.read_text()
                validate_experimental_csv(csv_text)
                exp_data = parse_experimental_csv(csv_text)
            except gr.Error:
                raise
            except Exception:
                exp_data = None
        return create_waveform_fig(sim_data, experimental_data=exp_data)

    exp_csv_upload.change(
        fn=refresh_waveform,
        inputs=[sim_state, exp_csv_upload],
        outputs=[fig_waveform],
    )

    remove_btn.click(
        fn=lambda runs, idx: remove_from_comparison(runs, int(idx)),
        inputs=[comparison_state, remove_idx],
        outputs=[comparison_state, fig_compare, compare_md],
    )
    clear_btn.click(fn=clear_comparison,
                    outputs=[comparison_state, fig_compare, compare_md])

    sim_state.change(
        fn=lambda s: gr.update(visible=s is not None),
        inputs=[sim_state], outputs=[export_file],
    )

    save_btn.click(fn=save_config, inputs=all_param_inputs, outputs=[save_file]).then(
        fn=lambda: gr.update(visible=True), outputs=[save_file],
    )
    load_btn.click(fn=load_config, inputs=[load_file], outputs=all_param_inputs)

    def run_calibration(preset_name, sim_time_us, progress=gr.Progress()):  # noqa: B008
        progress(0.1, desc="Auto-calibrating...")
        cal = auto_calibrate(preset_name, sim_time_us=sim_time_us)
        progress(1.0, desc="Done")
        md = format_calibration_markdown(cal)
        if "error" not in cal:
            return (
                gr.update(visible=True, value=md),
                cal["best_fc"],
                cal["best_fm"],
            )
        return gr.update(visible=True, value=md), gr.update(), gr.update()

    cal_btn.click(
        fn=run_calibration,
        inputs=[preset_dd, sim_time],
        outputs=[cal_output, inp_fc, inp_fm],
        concurrency_limit=1,
    )

    def apply_published_params(preset_name):
        fc, fm = get_published_params(preset_name)
        if fc is not None and fm is not None:
            return fc, fm
        gr.Warning(f"No published Lee model parameters available for '{preset_name}'.")
        return gr.update(), gr.update()

    pub_btn.click(
        fn=apply_published_params,
        inputs=[preset_dd],
        outputs=[inp_fc, inp_fm],
    )

    def run_sweep(preset_name, param, pmin, pmax, n, sim_time_us,
                  progress=gr.Progress()):  # noqa: B008
        if pmin >= pmax:
            raise gr.Error(f"Sweep min ({pmin}) must be less than max ({pmax}).")
        progress(0.0, desc="Starting sweep...")
        results = run_parameter_sweep(
            preset_name, param, (pmin, pmax), n_points=int(n),
            sim_time_us=sim_time_us, progress_fn=progress,
        )
        fig = create_sweep_fig(results)
        md = format_sweep_markdown(results)
        return fig, md

    sweep_btn.click(
        fn=run_sweep,
        inputs=[preset_dd, sweep_param, sweep_min, sweep_max, sweep_n, sim_time],
        outputs=[sweep_plot, sweep_md],
        concurrency_limit=1,
    )

    def run_2d_sweep_handler(preset_name, sim_time_us,
                              fm_lo, fm_hi, fc_lo, fc_hi, grid_n,
                              progress=gr.Progress()):  # noqa: B008
        if fm_lo >= fm_hi:
            raise gr.Error(f"fm min ({fm_lo}) must be less than fm max ({fm_hi}).")
        if fc_lo >= fc_hi:
            raise gr.Error(f"fc min ({fc_lo}) must be less than fc max ({fc_hi}).")
        n = int(grid_n)
        progress(0.0, desc="Starting 2D sweep...")
        results = run_2d_sweep(
            preset_name, sim_time_us=sim_time_us,
            fm_range=(fm_lo, fm_hi), fc_range=(fc_lo, fc_hi),
            n_fm=n, n_fc=n, progress_fn=progress,
        )
        fig = create_2d_sweep_fig(results)
        return fig, f"**2D sweep**: {results['preset']}, {n}x{n} grid (fm x fc)"

    sweep_2d_btn.click(
        fn=run_2d_sweep_handler,
        inputs=[preset_dd, sim_time,
                sweep_2d_fm_min, sweep_2d_fm_max,
                sweep_2d_fc_min, sweep_2d_fc_max, sweep_2d_n],
        outputs=[sweep_plot, sweep_md],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    server_port = int(os.environ.get("DPF_UI_PORT", "7860"))
    app.queue(max_size=5)
    auth_user = os.environ.get("DPF_AUTH_USER")
    auth_pass = os.environ.get("DPF_AUTH_PASS")
    auth = (auth_user, auth_pass) if auth_user and auth_pass else None
    app.launch(
        server_name="0.0.0.0", server_port=server_port, share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        auth=auth,
    )
