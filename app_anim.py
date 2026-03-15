"""3D animated plasma playback — shows plasma evolution from breakdown to pinch."""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

PHASE_COLORS = {
    "rundown": "#2196F3", "radial": "#FF5722", "reflected": "#FF9800",
    "pinch": "#FF1744", "none": "#607D8B",
}

STAGE_DESCRIPTIONS = {
    "rundown": "Stage: Current sheath sweeping gas toward anode tip",
    "radial": "Stage: Radial implosion -- plasma compressing inward",
    "reflected": "Stage: Shock reflected -- plasma expanding",
    "pinch": "Stage: Pinch -- peak compression and heating",
    "none": "",
}
N_THETA = 24


def _circle_coords(r: float, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, N_THETA)
    return r * np.cos(theta), r * np.sin(theta), np.full(N_THETA, z)


def _cylinder_wireframe(
    r: float, z0: float, z1: float, n_rings: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lightweight cylinder as stacked circle rings with connecting lines."""
    xs, ys, zs = [], [], []
    for z in np.linspace(z0, z1, n_rings):
        cx, cy, cz = _circle_coords(r, z)
        xs.extend(cx.tolist() + [None])
        ys.extend(cy.tolist() + [None])
        zs.extend(cz.tolist() + [None])
    theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    for th in theta[::4]:
        xs.extend([r * np.cos(th), r * np.cos(th), None])
        ys.extend([r * np.sin(th), r * np.sin(th), None])
        zs.extend([z0, z1, None])
    return np.array(xs), np.array(ys), np.array(zs)


def _filled_disc(
    r_inner: float, r_outer: float, z: float, n_r: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disc as concentric rings."""
    xs, ys, zs = [], [], []
    for r in np.linspace(max(r_inner, 0.1), r_outer, n_r):
        cx, cy, cz = _circle_coords(r, z)
        xs.extend(cx.tolist() + [None])
        ys.extend(cy.tolist() + [None])
        zs.extend(cz.tolist() + [None])
    return np.array(xs), np.array(ys), np.array(zs)


MAX_ANIMATION_FRAMES = 200


def create_animated_3d(d: dict[str, Any], n_frames: int = 80) -> go.Figure:
    """Build a Plotly animated 3D figure with play/pause slider."""
    cc = d["circuit"]
    sc = d["snowplow_cfg"]
    a = cc["anode_radius"] * 1e3
    b = cc["cathode_radius"] * 1e3
    L_anode = sc.get("anode_length", 0.16) * 1e3
    pcf = sc.get("pinch_column_fraction", 1.0)
    z_f = L_anode * pcf

    t_arr = d["t_us"]
    z_arr = d["z_mm"]
    r_arr = d["r_mm"]
    I_arr = d["I_MA"]
    phases = d["phases"]
    n_total = len(t_arr)
    n_frames = min(n_frames, MAX_ANIMATION_FRAMES)
    step_size = max(1, n_total // n_frames)
    frame_indices = list(range(0, n_total, step_size))[:MAX_ANIMATION_FRAMES]
    if frame_indices[-1] != n_total - 1:
        frame_indices[-1] = n_total - 1

    cathode_x, cathode_y, cathode_z = _cylinder_wireframe(b, 0, L_anode, n_rings=8)
    anode_x, anode_y, anode_z = _cylinder_wireframe(a, -30, 0, n_rings=4)

    idx0 = frame_indices[0]
    sheath_x, sheath_y, sheath_z = _build_sheath(
        a, b, L_anode, z_f, z_arr[idx0], r_arr[idx0], phases[idx0],
    )
    info_text = _frame_label(t_arr[idx0], I_arr[idx0], z_arr[idx0], r_arr[idx0], phases[idx0])
    phase_color = _phase_color(phases[idx0], r_arr[idx0], a)
    sheath_lw = _sheath_width(phases[idx0], r_arr[idx0], a)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=cathode_x, y=cathode_y, z=cathode_z,
                mode="lines", line=dict(color="#666", width=2),
                name="Cathode", showlegend=True,
            ),
            go.Scatter3d(
                x=anode_x, y=anode_y, z=anode_z,
                mode="lines", line=dict(color="#CCA030", width=3),
                name="Anode", showlegend=True,
            ),
            go.Scatter3d(
                x=sheath_x, y=sheath_y, z=sheath_z,
                mode="lines", line=dict(color=phase_color, width=sheath_lw),
                name="Plasma", showlegend=True,
            ),
            go.Scatter3d(
                x=[0], y=[0], z=[L_anode / 2],
                mode="text", text=[info_text],
                textfont=dict(size=12, color="white"),
                showlegend=False,
            ),
            go.Scatter3d(
                x=[0], y=[0], z=[L_anode + 15],
                mode="text", text=[""],
                textfont=dict(size=14, color="#FFD600"),
                showlegend=False,
            ),
        ],
    )

    frames = []
    slider_steps = []
    prev_phase = phases[frame_indices[0]] if frame_indices else "none"
    for fi, idx in enumerate(frame_indices):
        t_us = t_arr[idx]
        z_mm = z_arr[idx]
        r_mm = r_arr[idx]
        phase = phases[idx]
        I_MA = I_arr[idx]

        sx, sy, sz = _build_sheath(a, b, L_anode, z_f, z_mm, r_mm, phase)
        pc = _phase_color(phase, r_mm, a)
        lw = _sheath_width(phase, r_mm, a)
        label = _frame_label(t_us, I_MA, z_mm, r_mm, phase)

        transition_text = ""
        if phase != prev_phase:
            transition_text = f">> {prev_phase.upper()} -> {phase.upper()} <<"
        prev_phase = phase

        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=cathode_x, y=cathode_y, z=cathode_z,
                             mode="lines", line=dict(color="#666", width=2)),
                go.Scatter3d(x=anode_x, y=anode_y, z=anode_z,
                             mode="lines", line=dict(color="#CCA030", width=3)),
                go.Scatter3d(x=sx, y=sy, z=sz,
                             mode="lines", line=dict(color=pc, width=lw)),
                go.Scatter3d(x=[0], y=[0], z=[L_anode / 2],
                             mode="text", text=[label],
                             textfont=dict(size=12, color="white")),
                go.Scatter3d(x=[0], y=[0], z=[L_anode + 15],
                             mode="text", text=[transition_text],
                             textfont=dict(size=14, color="#FFD600")),
            ],
            name=f"f{fi}",
        ))
        slider_steps.append(dict(
            args=[[f"f{fi}"], dict(frame=dict(duration=50, redraw=True), mode="immediate")],
            label=f"{t_us:.1f}",
            method="animate",
        ))

    fig.frames = frames

    margin = b * 0.3
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x [mm]", range=[-(b + margin), b + margin]),
            yaxis=dict(title="y [mm]", range=[-(b + margin), b + margin]),
            zaxis=dict(title="z [mm]", range=[-50, L_anode + 30]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)),
        ),
        height=600, template="plotly_dark",
        title="Mather-Type DPF -- Plasma Evolution",
        margin=dict(l=0, r=0, t=60, b=0),
        annotations=[dict(
            text="Gas fill -> Discharge -> Sheath rundown -> Radial implosion -> Pinch -> Dissipation",
            showarrow=False, font=dict(size=11, color="#AAAAAA"),
            xref="paper", yref="paper", x=0.5, y=1.0, xanchor="center", yanchor="bottom",
        )],
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.05, y=0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(prefix="t = ", suffix=" us", font=dict(size=14)),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9, x=0.05, y=0,
            steps=slider_steps,
        )],
    )
    return fig


def _build_sheath(
    a: float, b: float, L_anode: float, z_f: float,
    z_mm: float, r_mm: float, phase: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the plasma geometry (sheath/shock/pinch) for one frame."""
    xs, ys, zs = [], [], []

    if phase == "rundown":
        dx, dy, dz = _filled_disc(a, b, z_mm, n_r=5)
        xs.extend(dx.tolist())
        ys.extend(dy.tolist())
        zs.extend(dz.tolist())
        for z_trail in np.linspace(max(z_mm - 20, 0), z_mm, 4)[:-1]:
            tx, ty, tz = _circle_coords(b * 0.7, z_trail)
            xs.extend(tx.tolist() + [None])
            ys.extend(ty.tolist() + [None])
            zs.extend(tz.tolist() + [None])

    elif phase in ("radial", "reflected"):
        dx, dy, dz = _filled_disc(a, b, L_anode, n_r=3)
        xs.extend(dx.tolist())
        ys.extend(dy.tolist())
        zs.extend(dz.tolist())
        if r_mm > 0.1:
            z_start = L_anode - z_f
            cx, cy, cz = _cylinder_wireframe(r_mm, z_start, L_anode, n_rings=5)
            xs.extend(cx.tolist())
            ys.extend(cy.tolist())
            zs.extend(cz.tolist())
            n_compression = max(2, int(3 + 4 * (1.0 - r_mm / a)))
            for frac in np.linspace(0.3, 0.9, n_compression):
                for z_ring in np.linspace(z_start, L_anode, 3):
                    rx, ry, rz = _circle_coords(r_mm * frac, z_ring)
                    xs.extend(rx.tolist() + [None])
                    ys.extend(ry.tolist() + [None])
                    zs.extend(rz.tolist() + [None])

    elif phase == "pinch":
        dx, dy, dz = _filled_disc(a, b, L_anode, n_r=3)
        xs.extend(dx.tolist())
        ys.extend(dy.tolist())
        zs.extend(dz.tolist())
        z_start = L_anode - z_f
        if r_mm > 0.01:
            cx, cy, cz = _cylinder_wireframe(r_mm, z_start, L_anode, n_rings=8)
            xs.extend(cx.tolist())
            ys.extend(cy.tolist())
            zs.extend(cz.tolist())
            cx2, cy2, cz2 = _cylinder_wireframe(r_mm * 0.3, z_start, L_anode, n_rings=4)
            xs.extend(cx2.tolist())
            ys.extend(cy2.tolist())
            zs.extend(cz2.tolist())

    return np.array(xs if xs else [0]), np.array(ys if ys else [0]), np.array(zs if zs else [0])


def _phase_color(phase: str, r_mm: float, a: float) -> str:
    """Dynamic color based on phase and compression ratio."""
    if phase == "rundown":
        return "#2196F3"
    if phase in ("radial", "reflected"):
        if a <= 0:
            return "#FF5722"
        frac = max(0.0, min(1.0, 1.0 - r_mm / a))
        r_int = int(0xFF * (0.6 + 0.4 * frac))
        g_int = int(0x57 * (1.0 - 0.7 * frac))
        b_int = int(0x22 * (1.0 - 0.9 * frac))
        return f"#{r_int:02X}{g_int:02X}{b_int:02X}"
    if phase == "pinch":
        return "#FF1744"
    return "#607D8B"


def _sheath_width(phase: str, r_mm: float, a: float) -> int:
    """Line width increases during compression."""
    if phase == "rundown":
        return 4
    if phase in ("radial", "reflected"):
        if a <= 0:
            return 5
        frac = max(0.0, min(1.0, 1.0 - r_mm / a))
        return int(4 + 6 * frac)
    if phase == "pinch":
        return 10
    return 4


def _frame_label(t_us: float, I_MA: float, z_mm: float, r_mm: float, phase: str) -> str:
    parts = [f"t={t_us:.1f}us  I={abs(I_MA):.2f}MA  {phase}"]
    if phase == "rundown":
        parts.append(f"z={z_mm:.0f}mm")
    elif phase in ("radial", "reflected", "pinch"):
        parts.append(f"r={r_mm:.1f}mm")
    stage = STAGE_DESCRIPTIONS.get(phase, "")
    if stage:
        parts.append(f"\n{stage}")
    return "  ".join(parts)


def create_animated_mhd_3d(d: dict[str, Any]) -> go.Figure:
    """3D animated MHD playback with density, pressure, and magnetic field.

    Shows electrode geometry with two cross-section planes:
    - x-z plane (y=0): density (Inferno colorscale)
    - y-z plane (x=0): magnetic pressure (Plasma colorscale)
    Annotations show current, compression, peak B-field, and peak pressure.
    """
    snapshots = d.get("mhd_snapshots", [])
    if not snapshots:
        if d.get("has_snowplow") and d.get("snowplow_obj"):
            return create_animated_3d(d)
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", height=550,
            annotations=[dict(
                text="No MHD snapshots — run a simulation first",
                showarrow=False, font=dict(size=16, color="#999"),
                xref="paper", yref="paper", x=0.5, y=0.5,
            )],
        )
        return fig

    cc = d["circuit"]
    sc = d.get("snowplow_cfg", {})
    a_mm = cc["anode_radius"] * 1e3
    b_mm = cc["cathode_radius"] * 1e3
    L_mm = sc.get("anode_length", 0.16) * 1e3
    rho0 = d.get("rho0", 1.0)
    I_arr = d.get("I_MA", np.array([0]))
    t_arr = d.get("t_us", np.array([0]))
    mu_0 = 4.0 * np.pi * 1e-7

    if len(snapshots) > MAX_ANIMATION_FRAMES:
        step = max(1, len(snapshots) // MAX_ANIMATION_FRAMES)
        indices = list(range(0, len(snapshots), step))[:MAX_ANIMATION_FRAMES]
        if indices[-1] != len(snapshots) - 1:
            indices[-1] = len(snapshots) - 1
        snapshots = [snapshots[i] for i in indices]

    cath_x, cath_y, cath_z = _cylinder_wireframe(b_mm, 0, L_mm, n_rings=8)
    an_x, an_y, an_z = _cylinder_wireframe(a_mm, 0, L_mm, n_rings=6)

    snap0 = snapshots[0]
    rho_2d = snap0["rho_mid"]
    nr, nz = rho_2d.shape
    r_vals = np.linspace(a_mm, b_mm, nr)
    z_vals = np.linspace(0, L_mm, nz)
    rr, zz = np.meshgrid(r_vals, z_vals, indexing="ij")

    # Density cross-section on x-z plane (y=0), mirrored
    x_rho = np.concatenate([rr.ravel(), -rr.ravel()])
    y_rho = np.zeros(2 * nr * nz)
    z_rho = np.concatenate([zz.ravel(), zz.ravel()])

    # Pressure/B cross-section on y-z plane (x=0), mirrored
    y_prs = np.concatenate([rr.ravel(), -rr.ravel()])
    x_prs = np.zeros(2 * nr * nz)
    z_prs = np.concatenate([zz.ravel(), zz.ravel()])

    # Color scale limits
    rho_gmax = max(float(np.max(s["rho_mid"])) for s in snapshots) / rho0
    rho_cmax = max(rho_gmax, 2.0)

    # Pressure: compute magnetic pressure from B_mid if available
    has_B = "B_mid" in snap0
    has_P = "P_mid" in snap0
    if has_P:
        p_gmax = max(float(np.max(s["P_mid"])) for s in snapshots)
        p_cmax = max(p_gmax, 1.0)
    else:
        p_cmax = 1.0

    def _I_at(t_snap):
        if len(t_arr) == 0:
            return 0.0
        return float(I_arr[int(np.argmin(np.abs(t_arr - t_snap)))])

    def _snap_data(snap):
        rho_n = (snap["rho_mid"] / rho0).ravel()
        rho_c = np.concatenate([rho_n, rho_n])
        comp = float(np.max(snap["rho_mid"])) / rho0

        if has_P:
            p_n = snap["P_mid"].ravel()
            p_c = np.concatenate([p_n, p_n])
            p_max = float(np.max(snap["P_mid"]))
        else:
            p_c = np.zeros(2 * nr * nz)
            p_max = 0.0

        if has_B:
            B_mid = snap["B_mid"]  # (3, nr, nz)
            B_mag = np.sqrt(np.sum(B_mid**2, axis=0))  # (nr, nz)
            B_peak = float(np.max(B_mag))
            p_mag = B_mag**2 / (2.0 * mu_0)
            p_mag_max = float(np.max(p_mag))
        else:
            B_peak = 0.0
            p_mag_max = 0.0

        return rho_c, p_c, comp, B_peak, p_max, p_mag_max

    rho_c0, p_c0, comp0, B0, pmax0, pmag0 = _snap_data(snap0)
    I0 = _I_at(snap0["t_us"])

    # 5 traces: cathode, anode, density plane, pressure plane, info text
    fig = go.Figure(data=[
        go.Scatter3d(x=cath_x, y=cath_y, z=cath_z, mode="lines",
                     line=dict(color="#666", width=2), name=f"Cathode (r={b_mm:.0f}mm)"),
        go.Scatter3d(x=an_x, y=an_y, z=an_z, mode="lines",
                     line=dict(color="#CCA030", width=3), name=f"Anode (r={a_mm:.0f}mm)"),
        go.Scatter3d(x=x_rho, y=y_rho, z=z_rho, mode="markers",
                     marker=dict(size=3, color=rho_c0, colorscale="Inferno",
                                 cmin=0, cmax=rho_cmax, opacity=0.85,
                                 colorbar=dict(title="Density<br>(rho/rho0)", x=1.0, len=0.4, y=0.8)),
                     name="Density (x-z plane)"),
        go.Scatter3d(x=x_prs, y=y_prs, z=z_prs, mode="markers",
                     marker=dict(size=3, color=p_c0, colorscale="Plasma",
                                 cmin=0, cmax=p_cmax, opacity=0.7,
                                 colorbar=dict(title="Pressure<br>(Pa)", x=1.0, len=0.4, y=0.3)),
                     name="Pressure (y-z plane)"),
        go.Scatter3d(x=[0], y=[b_mm + 10], z=[L_mm + 12], mode="text",
                     text=[f"t={snap0['t_us']:.1f}us | I={I0:.2f}MA | {comp0:.1f}x comp\n"
                           f"B={B0:.1f}T | P={pmax0:.0f}Pa"],
                     textfont=dict(size=10, color="white"), showlegend=False),
    ])

    frames, slider_steps = [], []
    for fi, snap in enumerate(snapshots):
        rho_c, p_c, comp, B_pk, p_pk, pmag_pk = _snap_data(snap)
        I_s = _I_at(snap["t_us"])
        frames.append(go.Frame(data=[
            go.Scatter3d(x=cath_x, y=cath_y, z=cath_z, mode="lines",
                         line=dict(color="#666", width=2)),
            go.Scatter3d(x=an_x, y=an_y, z=an_z, mode="lines",
                         line=dict(color="#CCA030", width=3)),
            go.Scatter3d(x=x_rho, y=y_rho, z=z_rho, mode="markers",
                         marker=dict(size=3, color=rho_c, colorscale="Inferno",
                                     cmin=0, cmax=rho_cmax, opacity=0.85,
                                     colorbar=dict(title="Density<br>(rho/rho0)", x=1.0, len=0.4, y=0.8))),
            go.Scatter3d(x=x_prs, y=y_prs, z=z_prs, mode="markers",
                         marker=dict(size=3, color=p_c, colorscale="Plasma",
                                     cmin=0, cmax=p_cmax, opacity=0.7,
                                     colorbar=dict(title="Pressure<br>(Pa)", x=1.0, len=0.4, y=0.3))),
            go.Scatter3d(x=[0], y=[b_mm + 10], z=[L_mm + 12], mode="text",
                         text=[f"t={snap['t_us']:.1f}us | I={I_s:.2f}MA | {comp:.1f}x comp\n"
                               f"B={B_pk:.1f}T | P={p_pk:.0f}Pa | P_mag={pmag_pk:.0f}Pa"],
                         textfont=dict(size=10, color="white")),
        ], name=f"f{fi}"))
        slider_steps.append(dict(
            args=[[f"f{fi}"], dict(frame=dict(duration=120, redraw=True), mode="immediate")],
            label=f"{snap['t_us']:.1f}", method="animate"))

    fig.frames = frames
    margin = b_mm * 0.3
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x [mm]", range=[-(b_mm + margin), b_mm + margin]),
            yaxis=dict(title="y [mm]", range=[-(b_mm + margin), b_mm + margin]),
            zaxis=dict(title="z [mm]", range=[-10, L_mm + 20]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.6), up=dict(x=0, y=0, z=1)),
        ),
        height=600, template="plotly_dark",
        title="3D MHD Playback — Density (x-z) + Pressure (y-z)",
        margin=dict(l=0, r=0, t=40, b=60),
        showlegend=True, legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.05, y=-0.05, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=150, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(prefix="t = ", suffix=" us", font=dict(size=13)),
            transition=dict(duration=0), pad=dict(b=10, t=50),
            len=0.9, x=0.05, y=-0.02, steps=slider_steps,
        )],
    )
    return fig


def create_animated_mhd(d: dict[str, Any]) -> go.Figure:
    """Animated 2D density heatmap from MHD snapshots with play/pause slider."""
    snapshots = d.get("mhd_snapshots", [])
    if not snapshots:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", height=500,
            annotations=[dict(
                text="No MHD snapshots recorded", showarrow=False,
                font=dict(size=16, color="#999"), xref="paper", yref="paper",
                x=0.5, y=0.5,
            )],
        )
        return fig

    cc = d["circuit"]
    sc = d.get("snowplow_cfg", {})
    a_mm = cc["anode_radius"] * 1e3
    b_mm = cc["cathode_radius"] * 1e3
    L_mm = sc.get("anode_length", 0.16) * 1e3

    rho0 = d.get("rho0", 1.0)
    snap0 = snapshots[0]
    rho_2d = snap0["rho_mid"]
    nr, nz = rho_2d.shape

    r_axis = np.linspace(a_mm, b_mm, nr)
    z_axis = np.linspace(0, L_mm, nz)

    global_max = max(float(np.max(s["rho_mid"])) for s in snapshots)
    zmax = max(global_max / rho0, 2.0)

    fig = go.Figure(
        data=[go.Heatmap(
            z=(rho_2d / rho0).T,
            x=r_axis, y=z_axis,
            colorscale="Inferno", zmin=0, zmax=zmax,
            colorbar=dict(title="rho/rho0"),
        )],
    )

    if len(snapshots) > MAX_ANIMATION_FRAMES:
        step = len(snapshots) // MAX_ANIMATION_FRAMES
        indices = list(range(0, len(snapshots), step))[:MAX_ANIMATION_FRAMES]
        if indices[-1] != len(snapshots) - 1:
            indices[-1] = len(snapshots) - 1
        snapshots = [snapshots[i] for i in indices]

    frames = []
    slider_steps = []
    for fi, snap in enumerate(snapshots):
        t_us = snap["t_us"]
        rho_norm = (snap["rho_mid"] / rho0).T
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=rho_norm, x=r_axis, y=z_axis,
                colorscale="Inferno", zmin=0, zmax=zmax,
                colorbar=dict(title="rho/rho0"),
            )],
            name=f"f{fi}",
        ))
        slider_steps.append(dict(
            args=[[f"f{fi}"], dict(frame=dict(duration=80, redraw=True), mode="immediate")],
            label=f"{t_us:.1f}",
            method="animate",
        ))

    fig.frames = frames

    I_arr = d.get("I_MA", np.array([0]))
    t_arr = d.get("t_us", np.array([0]))
    I_peak = float(np.max(np.abs(I_arr)))
    t_peak = float(t_arr[int(np.argmax(np.abs(I_arr)))])

    fig.update_layout(
        xaxis=dict(title="r [mm]"),
        yaxis=dict(title="z [mm]", scaleanchor="x"),
        height=550, template="plotly_dark",
        title=f"MHD Density Evolution | I_peak={I_peak:.2f} MA at {t_peak:.1f} us",
        margin=dict(l=60, r=20, t=50, b=60),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.05, y=-0.08, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(prefix="t = ", suffix=" us", font=dict(size=13)),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9, x=0.05, y=-0.02,
            steps=slider_steps,
        )],
    )
    return fig
