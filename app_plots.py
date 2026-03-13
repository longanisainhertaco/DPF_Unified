"""Plotly figure generators for DPF web UI — 2D waveforms, physics, schematic, 3D plasma."""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PHASE_COLORS = {
    "rundown": "#2196F3", "radial": "#FF5722", "reflected": "#FF9800",
    "pinch": "#9C27B0", "none": "#607D8B",
}


def create_waveform_fig(d: dict[str, Any]) -> go.Figure:
    t = d["t_us"]
    I_arr = d["I_MA"]  # noqa: N806
    V = d["V_kV"]
    phases = d["phases"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Circuit Current I(t)", "Capacitor Voltage V(t)"],
        vertical_spacing=0.08,
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

    fig.add_trace(go.Scatter(
        x=t, y=V, mode="lines",
        line=dict(color="#4CAF50", width=2), name="V_cap(t)",
    ), row=2, col=1)

    fig.add_annotation(
        x=d["t_peak"], y=d["I_peak"],
        text=f"I_peak = {d['I_peak']:.2f} MA",
        showarrow=True, arrowhead=2, ax=40, ay=-30, font=dict(size=11),
        row=1, col=1,
    )

    if d["has_snowplow"] and d["dip_pct"] > 1:
        fig.add_annotation(
            x=d["t_dip"], y=d["I_dip"],
            text=f"Dip = {d['I_dip']:.2f} MA ({d['dip_pct']:.0f}%)",
            showarrow=True, arrowhead=2, ax=-40, ay=-30,
            font=dict(size=11, color="#FF5722"), row=1, col=1,
        )

    if d["crowbar_t"]:
        fig.add_vline(x=d["crowbar_t"], line=dict(color="red", dash="dash", width=1),
                       annotation_text="Crowbar", row=2, col=1)
        fig.add_vline(x=d["crowbar_t"], line=dict(color="red", dash="dash", width=1),
                       row=1, col=1)

    fig.update_yaxes(title_text="Current [MA]", row=1, col=1)
    fig.update_yaxes(title_text="Voltage [kV]", row=2, col=1)
    fig.update_xaxes(title_text="Time [us]", row=2, col=1)
    fig.update_layout(
        height=500, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_physics_fig(d: dict[str, Any]) -> go.Figure:
    t = d["t_us"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Plasma Inductance L_p(t)", "Sheath / Shock Position",
                         "Pinch Voltage V_pinch(t)",
                         "Energy Partition", "Phase Timeline", "dL/dt"],
        vertical_spacing=0.15, horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=t, y=d["L_p_nH"], mode="lines",
        line=dict(color="#FF9800", width=2), name="L_plasma",
    ), row=1, col=1)
    fig.update_yaxes(title_text="L_p [nH]", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=d["z_mm"], mode="lines",
        line=dict(color="#2196F3", width=2), name="z_sheath [mm]",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=t, y=d["r_mm"], mode="lines",
        line=dict(color="#FF5722", width=2, dash="dot"), name="r_shock [mm]",
    ), row=1, col=2)
    fig.update_yaxes(title_text="Position [mm]", row=1, col=2)

    # Pinch voltage: V_pinch = I * dL/dt (key diagnostic for beam-target neutrons)
    L_nH = np.array(d["L_p_nH"])
    I_MA = np.array(d["I_MA"])
    t_arr = np.array(t)
    dLdt = np.gradient(L_nH * 1e-9, t_arr * 1e-6)  # [H/s]
    V_pinch_kV = np.abs(I_MA * 1e6 * dLdt) / 1e3  # [kV]
    fig.add_trace(go.Scatter(
        x=t, y=V_pinch_kV, mode="lines",
        line=dict(color="#E91E63", width=2), name="V_pinch [kV]",
    ), row=1, col=3)
    fig.update_yaxes(title_text="V_pinch [kV]", row=1, col=3)

    fig.add_trace(go.Scatter(
        x=t, y=d["E_cap_kJ"], mode="lines", fill="tozeroy",
        line=dict(color="#4CAF50"), name="E_cap [kJ]",
        fillcolor="rgba(76,175,80,0.3)",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=d["E_ind_kJ"], mode="lines", fill="tozeroy",
        line=dict(color="#2196F3"), name="E_ind [kJ]",
        fillcolor="rgba(33,150,243,0.3)",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=d["E_res_kJ"], mode="lines", fill="tozeroy",
        line=dict(color="#F44336"), name="E_res [kJ]",
        fillcolor="rgba(244,67,54,0.3)",
    ), row=2, col=1)
    fig.update_yaxes(title_text="Energy [kJ]", row=2, col=1)

    phase_map = {"rundown": 1, "radial": 2, "reflected": 3, "pinch": 4, "none": 0, "mhd": 5}
    phase_nums = [phase_map.get(p, 0) for p in d["phases"]]
    fig.add_trace(go.Scatter(
        x=t, y=phase_nums, mode="lines",
        line=dict(color="#9C27B0", width=3), name="Phase",
    ), row=2, col=2)
    fig.update_yaxes(
        title_text="Phase", tickvals=[0, 1, 2, 3, 4, 5],
        ticktext=["None", "Rundown", "Radial", "Reflected", "Pinch", "MHD"],
        row=2, col=2,
    )

    # dL/dt plot: key for understanding current dip
    dLdt_nH_us = np.gradient(L_nH, t_arr)  # [nH/us]
    fig.add_trace(go.Scatter(
        x=t, y=dLdt_nH_us, mode="lines",
        line=dict(color="#00BCD4", width=2), name="dL/dt [nH/us]",
    ), row=2, col=3)
    fig.update_yaxes(title_text="dL/dt [nH/us]", row=2, col=3)

    for r in (1, 2):
        for c in (1, 2, 3):
            fig.update_xaxes(title_text="Time [us]", row=r, col=c)

    fig.update_layout(
        height=650, template="plotly_dark", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        margin=dict(l=50, r=20, t=60, b=40),
    )
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
    """Phase portrait: (r, dr/dt) during radial implosion. Shows shock dynamics."""
    t = d["t_us"]
    r = d["r_mm"]
    phases = d["phases"]

    rad_mask = np.array([(p in ("radial", "reflected", "pinch")) for p in phases])
    if not np.any(rad_mask):
        fig = go.Figure()
        fig.add_annotation(
            text="No radial phase detected — run Lee model with snowplow",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#aaa"),
        )
        fig.update_layout(height=400, template="plotly_dark")
        return fig

    rad_idx = np.where(rad_mask)[0]
    r_rad = np.array([r[i] for i in rad_idx])
    t_rad = np.array([t[i] for i in rad_idx])
    ph_rad = [phases[i] for i in rad_idx]

    dt = np.diff(t_rad)
    dr = np.diff(r_rad)
    v_r = np.where(dt > 0, dr / dt, 0.0)
    r_mid = 0.5 * (r_rad[:-1] + r_rad[1:])
    ph_mid = ph_rad[:-1]

    fig = go.Figure()

    for phase_name, color in [("radial", "#FF5722"), ("reflected", "#FF9800"), ("pinch", "#9C27B0")]:
        mask = np.array([p == phase_name for p in ph_mid])
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=r_mid[mask], y=v_r[mask], mode="markers+lines",
                marker=dict(color=color, size=4),
                line=dict(color=color, width=1.5),
                name=phase_name.capitalize(),
            ))

    fig.update_layout(
        height=400, template="plotly_dark",
        xaxis=dict(title="Shock Radius r [mm]", autorange="reversed"),
        yaxis=dict(title="dr/dt [mm/us]"),
        title="Phase Portrait: Radial Implosion Dynamics",
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
    """3D visualization of electrode geometry, current sheath, and pinch column."""
    cc = d["circuit"]
    sc = d["snowplow_cfg"]
    sp = d["snowplow_obj"]

    a = cc["anode_radius"] * 1e3
    b = cc["cathode_radius"] * 1e3
    L_anode = sc.get("anode_length", 0.16) * 1e3

    fig = go.Figure()

    fig.add_trace(_cylinder_mesh(
        b, 0, L_anode, color="#555555", opacity=0.15, name="Cathode",
    ))
    fig.add_trace(_cylinder_mesh(
        a, -30, 0, color="#CCA030", opacity=0.4, name="Anode",
    ))

    if sp is not None:
        z_sheath_mm = sp.sheath_position * 1e3

        if not sp.rundown_complete:
            fig.add_trace(_disc_mesh(
                a, b, z_sheath_mm, color="#2196F3", opacity=0.6,
                name=f"Sheath (z={z_sheath_mm:.0f}mm)",
            ))
        else:
            fig.add_trace(_disc_mesh(
                a, b, L_anode, color="#2196F3", opacity=0.3,
                name="Sheath (arrived)",
            ))

        r_shock_mm = sp.shock_radius * 1e3
        if sp.rundown_complete and r_shock_mm < b * 0.95:
            z_f_mm = sp.z_f * 1e3
            z_start = L_anode - z_f_mm
            fig.add_trace(_cylinder_mesh(
                r_shock_mm, z_start, L_anode, n_theta=30, n_z=6,
                color="#FF5722", opacity=0.5,
                name=f"Shock front (r={r_shock_mm:.1f}mm)",
            ))

        if sp.pinch_complete:
            r_p = sp.shock_radius * 1e3
            z_f_mm = sp.z_f * 1e3
            z_start = L_anode - z_f_mm
            fig.add_trace(_cylinder_mesh(
                r_p, z_start, L_anode, n_theta=30, n_z=8,
                color="#FF1744", opacity=0.7,
                name=f"Pinch (r={r_p:.1f}mm)",
            ))

            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[z_start + z_f_mm / 2],
                mode="markers",
                marker=dict(size=8, color="#FF1744", symbol="diamond"),
                name="Pinch center",
            ))

    margin = b * 0.3
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x [mm]", range=[-(b + margin), b + margin]),
            yaxis=dict(title="y [mm]", range=[-(b + margin), b + margin]),
            zaxis=dict(title="z [mm]", range=[-50, L_anode + 20]),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        height=550, template="plotly_dark",
        title="3D Plasma Visualization",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
    )
    return fig
