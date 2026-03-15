"""Plotly figure generators for DPF web UI — 2D waveforms, physics, schematic, 3D plasma."""
from __future__ import annotations

import csv
import io
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PHASE_COLORS = {
    "rundown": "#2196F3", "radial": "#FF5722", "reflected": "#FF9800",
    "pinch": "#9C27B0", "none": "#607D8B",
}


def validate_experimental_csv(csv_content: str) -> None:
    """Validate CSV schema and monotonicity for experimental overlay upload.

    Raises gr.Error with a descriptive message if validation fails.
    Checks:
    - Required columns (time and current) are present.
    - Time column is strictly monotonically increasing.
    """
    if not csv_content or not csv_content.strip():
        raise gr.Error("CSV file is empty.")

    try:
        reader = csv.DictReader(io.StringIO(csv_content.strip()))
    except Exception as exc:
        raise gr.Error(f"CSV parse error: {exc}") from exc

    if reader.fieldnames is None:
        raise gr.Error("CSV has no header row.")

    time_col: str | None = None
    current_col: str | None = None
    time_scale = 1.0

    for h in reader.fieldnames:
        hl = h.strip().lower()
        if hl in ("time_us", "t_us"):
            time_col = h
            time_scale = 1.0
        elif hl in ("time_s", "t_s"):
            time_col = h
            time_scale = 1e6
        elif hl in ("t", "time") and time_col is None:
            time_col = h
            time_scale = 1.0

    for h in reader.fieldnames:
        hl = h.strip().lower()
        if hl in ("current_ma", "i_ma", "current_a", "i_a") or (
            hl in ("i", "current") and current_col is None
        ):
            current_col = h

    if time_col is None:
        raise gr.Error(
            "CSV missing a time column. Expected one of: "
            "time_us, t_us, time_s, t_s, t, time."
        )
    if current_col is None:
        raise gr.Error(
            "CSV missing a current column. Expected one of: "
            "current_MA, I_MA, current_A, I_A, I, current."
        )

    t_vals: list[float] = []
    for i, row in enumerate(reader, start=2):
        t_raw = row.get(time_col, "").strip()
        if not t_raw:
            continue
        try:
            t_vals.append(float(t_raw) * time_scale)
        except ValueError as exc:
            raise gr.Error(
                f"Non-numeric value in time column at row {i}: {t_raw!r}"
            ) from exc

    if len(t_vals) == 0:
        raise gr.Error("CSV contains a header but no data rows.")

    for i in range(1, len(t_vals)):
        if t_vals[i] <= t_vals[i - 1]:
            raise gr.Error(
                f"Time column is not monotonically increasing at row {i + 1}: "
                f"{t_vals[i - 1]} -> {t_vals[i]}. "
                "Ensure rows are sorted by time in ascending order."
            )


def parse_experimental_csv(csv_content: str) -> dict[str, list[float]] | None:
    """Parse CSV with time and current columns.

    Accepts: time_us/current_MA, time_s/current_A, t/I (assumes us/MA)
    Returns dict with t_us and I_MA keys, or None on failure.
    """
    try:
        reader = csv.DictReader(io.StringIO(csv_content.strip()))
        if reader.fieldnames is None:
            return None

        time_col: str | None = None
        current_col: str | None = None
        time_scale = 1.0
        current_scale = 1.0

        for h in reader.fieldnames:
            hl = h.strip().lower()
            if hl in ("time_us", "t_us"):
                time_col = h
                time_scale = 1.0
            elif hl in ("time_s", "t_s"):
                time_col = h
                time_scale = 1e6
            elif hl in ("t", "time") and time_col is None:
                time_col = h
                time_scale = 1.0

        for h in reader.fieldnames:
            hl = h.strip().lower()
            if hl in ("current_ma", "i_ma"):
                current_col = h
                current_scale = 1.0
            elif hl in ("current_a", "i_a"):
                current_col = h
                current_scale = 1e-6
            elif hl in ("i", "current") and current_col is None:
                current_col = h
                current_scale = 1.0

        if time_col is None or current_col is None:
            return None

        t_us: list[float] = []
        i_ma: list[float] = []
        for row in reader:
            t_val = row.get(time_col, "").strip()
            i_val = row.get(current_col, "").strip()
            if not t_val or not i_val:
                continue
            t_us.append(float(t_val) * time_scale)
            i_ma.append(float(i_val) * current_scale)

        if not t_us:
            return None

        return {"t_us": t_us, "I_MA": i_ma}

    except Exception:
        return None


def create_waveform_fig(
    d: dict[str, Any],
    experimental_data: dict[str, list[float]] | None = None,
) -> go.Figure:
    t = d["t_us"]
    I_arr = d["I_MA"]  # noqa: N806
    V = d["V_kV"]
    phases = d["phases"]

    has_exp = experimental_data is not None
    n_rows = 3 if has_exp else 2
    titles = ["Circuit Current I(t)", "Capacitor Voltage V(t)"]
    if has_exp:
        titles.append("Residual: Simulation - Experiment")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3] if has_exp else [0.5, 0.5],
    )

    if d.get("has_snowplow"):
        prev = phases[0]
        seg_start = 0
        shown_phases = set()
        for i in range(1, len(phases) + 1):
            if i == len(phases) or phases[i] != prev:
                end = i
                color = PHASE_COLORS.get(prev, "#607D8B")
                show = prev not in shown_phases
                if show:
                    shown_phases.add(prev)
                fig.add_trace(go.Scatter(
                    x=t[seg_start:end], y=I_arr[seg_start:end],
                    mode="lines", line=dict(color=color, width=2.5),
                    name=prev.capitalize() if show else None,
                    showlegend=show, legendgroup=prev,
                ), row=1, col=1)
                if i < len(phases):
                    seg_start = i
                    prev = phases[i]
    else:
        fig.add_trace(go.Scatter(
            x=t, y=I_arr, mode="lines",
            line=dict(color="#2196F3", width=2.5), name="I(t)",
        ), row=1, col=1)

    if experimental_data is not None:
        fig.add_trace(go.Scatter(
            x=experimental_data["t_us"], y=experimental_data["I_MA"],
            mode="lines", line=dict(color="red", width=2, dash="dash"),
            name="Experimental",
        ), row=1, col=1)

        t_exp = np.array(experimental_data["t_us"])
        I_exp = np.array(experimental_data["I_MA"])  # noqa: N806
        I_sim_interp = np.interp(t_exp, t, I_arr)  # noqa: N806
        residual = I_sim_interp - I_exp

        fig.add_trace(go.Scatter(
            x=t_exp, y=residual, mode="lines",
            line=dict(color="#FF9800", width=1.5),
            name="Residual (sim - exp)",
            fill="tozeroy", fillcolor="rgba(255,152,0,0.2)",
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#666", row=3, col=1)
        fig.update_yaxes(title_text="Residual [MA]", row=3, col=1)
        fig.update_xaxes(title_text="Time [us]", row=3, col=1)

        rmse = float(np.sqrt(np.mean(residual**2)))
        fig.add_annotation(
            x=0.98, y=0.95, xref="x3 domain", yref="y3 domain",
            text=f"RMSE = {rmse:.4f} MA",
            showarrow=False, font=dict(size=12, color="#FF9800"),
            bgcolor="rgba(0,0,0,0.5)",
        )

    fig.add_trace(go.Scatter(
        x=t, y=V, mode="lines",
        line=dict(color="#4CAF50", width=2), name="V_cap(t)",
    ), row=2, col=1)

    # Phase-colored background bands
    if d.get("has_snowplow"):
        phase_bg = {
            "rundown": "rgba(33,150,243,0.08)",
            "radial": "rgba(255,87,34,0.12)",
            "reflected": "rgba(255,152,0,0.08)",
            "pinch": "rgba(156,39,176,0.12)",
        }
        prev_phase = phases[0]
        seg_start_t = t[0]
        for i in range(1, len(phases)):
            if phases[i] != prev_phase or i == len(phases) - 1:
                end_t = t[i] if i < len(phases) - 1 else t[-1]
                bg = phase_bg.get(prev_phase)
                if bg:
                    fig.add_vrect(
                        x0=seg_start_t, x1=end_t,
                        fillcolor=bg, layer="below", line_width=0,
                        row=1, col=1,
                    )
                    fig.add_vrect(
                        x0=seg_start_t, x1=end_t,
                        fillcolor=bg, layer="below", line_width=0,
                        row=2, col=1,
                    )
                seg_start_t = t[i]
                prev_phase = phases[i]

        # Phase band legend entries (invisible traces just for legend)
        for phase_name, color_map in [("Rundown phase", "#2196F3"), ("Radial phase", "#FF5722"),
                                       ("Reflected phase", "#FF9800"), ("Pinch phase", "#9C27B0")]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color_map, symbol="square"),
                name=f"  {phase_name}",
                showlegend=True,
            ), row=1, col=1)

    # I_peak scatter marker + annotation
    fig.add_trace(go.Scatter(
        x=[d["t_peak"]], y=[d["I_peak"]], mode="markers",
        marker=dict(color="#FFEB3B", size=10, symbol="star"),
        showlegend=False,
    ), row=1, col=1)
    fig.add_annotation(
        x=d["t_peak"], y=d["I_peak"],
        text=f"I_peak = {d['I_peak']:.2f} MA",
        showarrow=True, arrowhead=2, ax=40, ay=-30, font=dict(size=11),
        row=1, col=1,
    )

    # Current dip marker (only when dip is significant)
    if d["has_snowplow"] and d["dip_pct"] > 5:
        fig.add_trace(go.Scatter(
            x=[d["t_dip"]], y=[d["I_dip"]], mode="markers",
            marker=dict(color="#FF5722", size=10, symbol="triangle-down"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_annotation(
            x=d["t_dip"], y=d["I_dip"],
            text=f"Dip: {d['dip_pct']:.0f}%",
            showarrow=True, arrowhead=2, arrowcolor="#FF5722",
            ax=-40, ay=-40,
            font=dict(size=12, color="#FF5722", family="Arial Black"),
            row=1, col=1,
        )
    elif d["has_snowplow"] and d["dip_pct"] > 1:
        fig.add_annotation(
            x=d["t_dip"], y=d["I_dip"],
            text=f"Dip = {d['I_dip']:.2f} MA ({d['dip_pct']:.0f}%)",
            showarrow=True, arrowhead=2, ax=-40, ay=-30,
            font=dict(size=11, color="#FF5722"), row=1, col=1,
        )

    # Crowbar marker
    if d["crowbar_t"]:
        fig.add_vline(x=d["crowbar_t"], line=dict(color="red", dash="dash", width=1),
                       row=1, col=1)
        fig.add_vline(x=d["crowbar_t"], line=dict(color="red", dash="dash", width=1),
                       annotation_text="Crowbar fires",
                       annotation_font=dict(size=10, color="red"),
                       row=2, col=1)

    # Phase boundary: rundown -> radial transition
    for i in range(1, len(phases)):
        if phases[i - 1] == "rundown" and phases[i] == "radial":
            fig.add_vline(
                x=t[i], line=dict(color="#FF9800", dash="dot", width=1.5),
                annotation_text="Radial phase begins",
                annotation_font=dict(size=10, color="#FF9800"),
                annotation_position="top right",
                row=1, col=1,
            )
            break

    fig.update_yaxes(title_text="Current [MA]", row=1, col=1)
    fig.update_yaxes(title_text="Voltage [kV]", row=2, col=1)
    fig.update_xaxes(title_text="Time [us]", row=2, col=1)
    fig.update_layout(
        height=650 if has_exp else 500, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_physics_fig(d: dict[str, Any]) -> go.Figure:
    t = d["t_us"]
    phases = d["phases"]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Where does the bank energy go?",
            "Where is the plasma?",
            "What drives the pinch?",
        ],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]],
        vertical_spacing=0.10,
        shared_xaxes=True,
    )

    # -- Panel 1: Energy Story (stacked area) --
    fig.add_trace(go.Scatter(
        x=t, y=d["E_cap_kJ"], mode="lines", fill="tozeroy",
        line=dict(color="#66BB6A", width=0), fillcolor="rgba(102,187,106,0.6)",
        name="Capacitor (stored)", stackgroup="energy",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=d["E_ind_kJ"], mode="lines", fill="tonexty",
        line=dict(color="#42A5F5", width=0), fillcolor="rgba(66,165,245,0.6)",
        name="Magnetic field (useful work)", stackgroup="energy",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=d["E_res_kJ"], mode="lines", fill="tonexty",
        line=dict(color="#EF5350", width=0), fillcolor="rgba(239,83,80,0.6)",
        name="Resistive loss (wasted heat)", stackgroup="energy",
    ), row=1, col=1)
    fig.update_yaxes(title_text="Energy [kJ]", row=1, col=1)

    # -- Panel 2: Position & Timing --
    fig.add_trace(go.Scatter(
        x=t, y=d["z_mm"], mode="lines",
        line=dict(color="#42A5F5", width=2.5), name="Sheath position z [mm]",
    ), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=d["r_mm"], mode="lines",
        line=dict(color="#EF5350", width=2.5), name="Shock radius r [mm]",
    ), row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Sheath z [mm]", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Shock r [mm]", row=2, col=1, secondary_y=True)

    # -- Panel 3: Pinch Diagnostics --
    L_nH = np.array(d["L_p_nH"])
    I_MA = np.array(d["I_MA"])
    t_arr = np.array(t)
    dLdt = np.gradient(L_nH * 1e-9, t_arr * 1e-6)
    V_pinch_kV = np.abs(I_MA * 1e6 * dLdt) / 1e3
    dLdt_nH_us = np.gradient(L_nH, t_arr)

    fig.add_trace(go.Scatter(
        x=t, y=V_pinch_kV, mode="lines",
        line=dict(color="#EC407A", width=2.5), name="Pinch voltage [kV]",
    ), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=dLdt_nH_us, mode="lines",
        line=dict(color="#26C6DA", width=2), name="dL/dt [nH/us]",
    ), row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="V_pinch [kV]", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="dL/dt [nH/us]", row=3, col=1, secondary_y=True)

    # Phase-colored background bands on panels 2 and 3
    if d.get("has_snowplow"):
        phase_bg = {
            "rundown": "rgba(33,150,243,0.08)",
            "radial": "rgba(255,87,34,0.12)",
            "reflected": "rgba(255,152,0,0.08)",
            "pinch": "rgba(156,39,176,0.12)",
        }
        prev_phase = phases[0]
        seg_start_t = t[0]
        for i in range(1, len(phases)):
            if phases[i] != prev_phase or i == len(phases) - 1:
                end_t = t[i] if i < len(phases) - 1 else t[-1]
                bg = phase_bg.get(prev_phase)
                if bg:
                    fig.add_vrect(
                        x0=seg_start_t, x1=end_t,
                        fillcolor=bg, layer="below", line_width=0,
                        row=2, col=1,
                    )
                    fig.add_vrect(
                        x0=seg_start_t, x1=end_t,
                        fillcolor=bg, layer="below", line_width=0,
                        row=3, col=1,
                    )
                seg_start_t = t[i]
                prev_phase = phases[i]

        # Phase band legend entries (invisible traces just for legend)
        for phase_name, color_map in [("Rundown phase", "#2196F3"), ("Radial phase", "#FF5722"),
                                       ("Reflected phase", "#FF9800"), ("Pinch phase", "#9C27B0")]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color_map, symbol="square"),
                name=f"  {phase_name}",
                showlegend=True,
            ), row=1, col=1)

    fig.update_xaxes(title_text="Time [us]", row=3, col=1)

    fig.update_layout(
        height=800, template="plotly_dark", showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            font=dict(size=10), bgcolor="rgba(0,0,0,0.5)",
        ),
        margin=dict(l=55, r=55, t=80, b=40),
        font=dict(size=11),
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color="#ccc")

    return fig


def create_schematic_fig(d: dict[str, Any]) -> go.Figure:
    cc = d["circuit"]
    sc = d["snowplow_cfg"]
    sp = d["snowplow_obj"]

    a = cc["anode_radius"] * 1e3
    b = cc["cathode_radius"] * 1e3
    L_anode = sc.get("anode_length", 0.16) * 1e3

    fig = go.Figure()

    fig.add_shape(type="rect", x0=0, y0=a, x1=L_anode, y1=b,
                  fillcolor="rgba(100,100,100,0.3)", line=dict(color="#888", width=2))
    fig.add_shape(type="rect", x0=0, y0=-b, x1=L_anode, y1=-a,
                  fillcolor="rgba(100,100,100,0.3)", line=dict(color="#888", width=2))
    fig.add_shape(type="rect", x0=-30, y0=-a, x1=0, y1=a,
                  fillcolor="rgba(200,150,50,0.5)", line=dict(color="#CCA030", width=2))

    fig.add_annotation(x=L_anode / 2, y=b + 10, text="Cathode", showarrow=False,
                       font=dict(color="#aaa", size=10))
    fig.add_annotation(x=-15, y=0, text="Anode", showarrow=False,
                       font=dict(color="#CCA030", size=10))

    if sp is not None and sp.pinch_complete:
        r_p = sp.shock_radius * 1e3
        z_f = sp.z_f * 1e3
        z_start = L_anode - z_f
        fig.add_shape(type="rect", x0=z_start, y0=-r_p, x1=L_anode, y1=r_p,
                      fillcolor="rgba(255,87,34,0.6)", line=dict(color="#FF5722", width=2))
        fig.add_annotation(x=z_start + z_f / 2, y=0,
                           text=f"Pinch\nr={r_p:.1f}mm", showarrow=False,
                           font=dict(color="white", size=11))

    if sp is not None and not sp.rundown_complete:
        z_s = sp.sheath_position * 1e3
        fig.add_shape(type="line", x0=z_s, y0=-b + 5, x1=z_s, y1=b - 5,
                      line=dict(color="#2196F3", width=3, dash="dash"))
        fig.add_annotation(x=z_s, y=b + 5, text="Sheath", showarrow=False,
                           font=dict(color="#2196F3", size=10))

    fig.add_annotation(x=L_anode + 10, y=a, text=f"a = {a:.0f} mm",
                       showarrow=True, arrowhead=0, ax=30, ay=0,
                       font=dict(size=9, color="#aaa"))
    fig.add_annotation(x=L_anode + 10, y=b, text=f"b = {b:.0f} mm",
                       showarrow=True, arrowhead=0, ax=30, ay=0,
                       font=dict(size=9, color="#aaa"))

    fig.update_layout(
        height=300, template="plotly_dark",
        xaxis=dict(title="z [mm]", range=[-50, L_anode + 60], scaleanchor="y"),
        yaxis=dict(title="r [mm]", range=[-(b + 30), b + 30]),
        title="Electrode Geometry & Pinch",
        margin=dict(l=60, r=20, t=40, b=40), showlegend=False,
    )
    return fig


def _cylinder_mesh(
    r: float, z0: float, z1: float, n_theta: int = 40, n_z: int = 10,
    color: str = "gray", opacity: float = 0.3, name: str = "",
) -> go.Mesh3d:
    """Generate a Mesh3d cylinder surface between z0 and z1 at radius r."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(z0, z1, n_z)
    th_grid, z_grid = np.meshgrid(theta, zs)
    x = (r * np.cos(th_grid)).ravel()
    y = (r * np.sin(th_grid)).ravel()
    z = z_grid.ravel()

    n_t = n_theta
    n_zz = n_z
    i_list, j_list, k_list = [], [], []
    for zi in range(n_zz - 1):
        for ti in range(n_t):
            t_next = (ti + 1) % n_t
            p0 = zi * n_t + ti
            p1 = zi * n_t + t_next
            p2 = (zi + 1) * n_t + ti
            p3 = (zi + 1) * n_t + t_next
            i_list.extend([p0, p0])
            j_list.extend([p1, p2])
            k_list.extend([p2, p3])

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_list, j=j_list, k=k_list,
        color=color, opacity=opacity, name=name,
        flatshading=True, showlegend=bool(name),
    )


def _disc_mesh(
    r_inner: float, r_outer: float, z_pos: float,
    n_theta: int = 40, n_r: int = 5,
    color: str = "#2196F3", opacity: float = 0.5, name: str = "",
) -> go.Mesh3d:
    """Generate a Mesh3d annular disc at z_pos."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    radii = np.linspace(r_inner, r_outer, n_r)
    th_grid, r_grid = np.meshgrid(theta, radii)
    x = (r_grid * np.cos(th_grid)).ravel()
    y = (r_grid * np.sin(th_grid)).ravel()
    z = np.full_like(x, z_pos)

    n_t = n_theta
    i_list, j_list, k_list = [], [], []
    for ri in range(n_r - 1):
        for ti in range(n_t):
            t_next = (ti + 1) % n_t
            p0 = ri * n_t + ti
            p1 = ri * n_t + t_next
            p2 = (ri + 1) * n_t + ti
            p3 = (ri + 1) * n_t + t_next
            i_list.extend([p0, p0])
            j_list.extend([p1, p2])
            k_list.extend([p2, p3])

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_list, j=j_list, k=k_list,
        color=color, opacity=opacity, name=name,
        flatshading=True, showlegend=bool(name),
    )


def create_phase_portrait(d: dict[str, Any]) -> go.Figure:
    """Compression view: r(t) and compression ratio during radial implosion."""
    t = d["t_us"]
    r = d["r_mm"]
    phases = d["phases"]

    rad_mask = np.array([(p in ("radial", "reflected", "pinch")) for p in phases])
    if not np.any(rad_mask):
        fig = go.Figure()
        fig.add_annotation(
            text="No radial phase detected -- run Lee model with snowplow",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#aaa"),
        )
        fig.update_layout(height=400, template="plotly_dark")
        return fig

    rad_idx = np.where(rad_mask)[0]
    r_rad = np.array([r[i] for i in rad_idx])
    t_rad = np.array([t[i] for i in rad_idx])
    ph_rad = [phases[i] for i in rad_idx]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        subplot_titles=[
            "Shock Radius (how far the plasma has compressed)",
            "Compression Ratio (how much denser the plasma is)",
            "Phase Portrait: r vs dr/dt (dynamical systems view)",
        ],
        vertical_spacing=0.10,
        row_heights=[0.35, 0.35, 0.30],
    )

    # Panel 1: r(t) with phase colors
    for phase_name, color in [("radial", "#EF5350"), ("reflected", "#FFA726"), ("pinch", "#AB47BC")]:
        mask = np.array([p == phase_name for p in ph_rad])
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=t_rad[mask], y=r_rad[mask], mode="lines+markers",
                marker=dict(color=color, size=3),
                line=dict(color=color, width=2.5),
                name=phase_name.capitalize(),
            ), row=1, col=1)

    fig.add_annotation(
        x=t_rad[0], y=r_rad[0],
        text=f"Starts at cathode ({r_rad[0]:.0f} mm)",
        showarrow=True, ax=50, ay=-20, font=dict(size=10),
        row=1, col=1,
    )
    min_idx = int(np.argmin(r_rad))
    fig.add_annotation(
        x=t_rad[min_idx], y=r_rad[min_idx],
        text=f"Minimum: {r_rad[min_idx]:.1f} mm",
        showarrow=True, ax=-40, ay=30,
        font=dict(size=11, color="#EF5350"),
        row=1, col=1,
    )

    # Peak velocity annotation
    if len(t_rad) > 1:
        dt_arr = np.diff(t_rad)
        dr_arr = np.diff(r_rad)
        v_r = np.where(dt_arr > 0, dr_arr / dt_arr, 0.0)
        v_peak_idx = int(np.argmin(v_r))  # most negative = fastest inward
        v_peak = abs(float(v_r[v_peak_idx]))
        t_vpeak = float(0.5 * (t_rad[v_peak_idx] + t_rad[v_peak_idx + 1]))
        r_vpeak = float(0.5 * (r_rad[v_peak_idx] + r_rad[v_peak_idx + 1]))
        fig.add_annotation(
            x=t_vpeak, y=r_vpeak,
            text=f"Peak velocity: {v_peak:.0f} mm/us ({v_peak*1e3:.0f} km/s)",
            showarrow=True, arrowhead=2, ax=60, ay=30,
            font=dict(size=10, color="#26C6DA"),
            row=1, col=1,
        )

    fig.update_yaxes(title_text="Radius [mm]", row=1, col=1)

    # Panel 2: Compression ratio (b/r)^2
    b_mm = d["circuit"]["cathode_radius"] * 1e3
    comp_ratio = (b_mm / np.maximum(r_rad, 0.1)) ** 2
    for phase_name, color in [("radial", "#EF5350"), ("reflected", "#FFA726"), ("pinch", "#AB47BC")]:
        mask = np.array([p == phase_name for p in ph_rad])
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=t_rad[mask], y=comp_ratio[mask], mode="lines+markers",
                marker=dict(color=color, size=3),
                line=dict(color=color, width=2.5),
                name=phase_name.capitalize(), showlegend=False,
            ), row=2, col=1)

    fig.add_hline(y=10, line_dash="dash", line_color="#666",
                  annotation_text="10x compression", row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="#999",
                  annotation_text="100x compression", row=2, col=1)

    fig.update_yaxes(title_text="Compression (b/r)^2", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Time [us]", row=2, col=1)

    # Panel 3: Classic phase portrait for researchers
    if len(t_rad) > 1:
        dt_arr = np.diff(t_rad)
        dr_arr = np.diff(r_rad)
        v_r = np.where(dt_arr > 0, dr_arr / dt_arr, 0.0)
        r_mid = 0.5 * (r_rad[:-1] + r_rad[1:])
        ph_mid = ph_rad[:-1]

        for phase_name, color in [("radial", "#EF5350"), ("reflected", "#FFA726"), ("pinch", "#AB47BC")]:
            mask = np.array([p == phase_name for p in ph_mid])
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=r_mid[mask], y=v_r[mask], mode="lines+markers",
                    marker=dict(color=color, size=3),
                    line=dict(color=color, width=1.5),
                    name=phase_name.capitalize(), showlegend=False,
                ), row=3, col=1)

        fig.update_xaxes(title_text="Shock Radius r [mm]", autorange="reversed", row=3, col=1)
        fig.update_yaxes(title_text="dr/dt [mm/us]", row=3, col=1)

    fig.update_layout(
        height=750, template="plotly_dark",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_comparison_fig(runs: list[dict[str, Any]]) -> go.Figure:
    """Overlay I(t) and V(t) from multiple simulation runs for comparison."""
    if not runs:
        fig = go.Figure()
        fig.add_annotation(
            text="Run multiple simulations to compare them here",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#aaa"),
        )
        fig.update_layout(height=500, template="plotly_dark")
        return fig

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Current Comparison I(t)", "Voltage Comparison V(t)"],
        vertical_spacing=0.08,
    )

    palette = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0",
               "#00BCD4", "#E91E63", "#8BC34A"]

    for i, run in enumerate(runs):
        color = palette[i % len(palette)]
        label = run.get("_label", f"Run {i+1}")
        fig.add_trace(go.Scatter(
            x=run["t_us"], y=run["I_MA"], mode="lines",
            line=dict(color=color, width=2),
            name=f"{label} I(t)", legendgroup=label,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=run["t_us"], y=run["V_kV"], mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            name=f"{label} V(t)", legendgroup=label,
            showlegend=False,
        ), row=2, col=1)

    fig.update_yaxes(title_text="Current [MA]", row=1, col=1)
    fig.update_yaxes(title_text="Voltage [kV]", row=2, col=1)
    fig.update_xaxes(title_text="Time [us]", row=2, col=1)
    fig.update_layout(
        height=500, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_3d_plasma_fig(d: dict[str, Any]) -> go.Figure:
    """3D visualization of electrode geometry, current sheath, and pinch column.

    Orientation: z=0 at the insulator (breech), z=L_anode at the open end (muzzle).
    Plasma forms at the anode tip (top of the visualization) — standard Mather-type DPF.
    Camera looks slightly downward so the pinch region is clearly visible at the top.
    """
    cc = d["circuit"]
    sc = d["snowplow_cfg"]
    sp = d["snowplow_obj"]

    a = cc["anode_radius"] * 1e3
    b = cc["cathode_radius"] * 1e3
    L_anode = sc.get("anode_length", 0.16) * 1e3

    fig = go.Figure()

    # Cathode (outer electrode) — extends full anode length
    fig.add_trace(_cylinder_mesh(
        b, 0, L_anode, color="#555555", opacity=0.15,
        name=f"Cathode (outer, r={b:.0f}mm)",
    ))
    # Anode (inner electrode) — extends from insulator into the gap
    fig.add_trace(_cylinder_mesh(
        a, 0, L_anode, color="#CCA030", opacity=0.4,
        name=f"Anode (inner, r={a:.0f}mm)",
    ))
    # Insulator disc at z=0 (breech)
    fig.add_trace(_disc_mesh(
        a * 0.3, b, 0, color="#4488AA", opacity=0.25,
        name="Insulator (breech, z=0)",
    ))

    # Label annotations for electrodes
    fig.add_trace(go.Scatter3d(
        x=[0], y=[b + 5], z=[L_anode / 2],
        mode="text", text=["Cathode"], textfont=dict(size=10, color="#999"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[a + 3], z=[L_anode * 0.8],
        mode="text", text=["Anode"], textfont=dict(size=10, color="#CCA030"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-8],
        mode="text", text=["Insulator (breech)"], textfont=dict(size=9, color="#4488AA"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[L_anode + 10],
        mode="text", text=["Open end (pinch forms here)"],
        textfont=dict(size=9, color="#FF5722"),
        showlegend=False,
    ))

    if sp is not None:
        z_sheath_mm = sp.sheath_position * 1e3

        if not sp.rundown_complete:
            # Current sheath sweeping gas upward along the anode
            fig.add_trace(_disc_mesh(
                a, b, z_sheath_mm, color="#2196F3", opacity=0.6,
                name=f"Current sheath (z={z_sheath_mm:.0f}mm, sweeping gas upward)",
            ))
        else:
            fig.add_trace(_disc_mesh(
                a, b, L_anode, color="#2196F3", opacity=0.3,
                name="Sheath (reached anode tip)",
            ))

        r_shock_mm = sp.shock_radius * 1e3
        if sp.rundown_complete and r_shock_mm < b * 0.95:
            z_f_mm = sp.z_f * 1e3
            z_start = L_anode - z_f_mm
            fig.add_trace(_cylinder_mesh(
                r_shock_mm, z_start, L_anode, n_theta=30, n_z=6,
                color="#FF5722", opacity=0.5,
                name=f"Radial shock (r={r_shock_mm:.1f}mm, compressing inward)",
            ))

        if sp.pinch_complete:
            r_p = sp.shock_radius * 1e3
            z_f_mm = sp.z_f * 1e3
            z_start = L_anode - z_f_mm
            fig.add_trace(_cylinder_mesh(
                r_p, z_start, L_anode, n_theta=30, n_z=8,
                color="#FF1744", opacity=0.7,
                name=f"Pinch column (r={r_p:.1f}mm, hot dense plasma)",
            ))

            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[z_start + z_f_mm / 2],
                mode="markers",
                marker=dict(size=8, color="#FF1744", symbol="diamond"),
                name="Pinch center (max compression)",
            ))

    margin = b * 0.3
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x [mm]", range=[-(b + margin), b + margin]),
            yaxis=dict(title="y [mm]", range=[-(b + margin), b + margin]),
            zaxis=dict(title="z [mm] (insulator=0, open end=top)", range=[-20, L_anode + 25]),
            aspectmode="data",
            camera=dict(
                # Look from the side and slightly above — insulator at bottom, pinch at top
                eye=dict(x=1.8, y=0.8, z=0.5),
                center=dict(x=0, y=0, z=0.1),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        height=550, template="plotly_dark",
        title="3D Plasma Visualization (Mather-type DPF — pinch forms at anode tip)",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
    )
    return fig
