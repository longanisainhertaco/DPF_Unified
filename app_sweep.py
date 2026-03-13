"""Parameter sweep module for DPF web UI.

Runs multiple simulations across a parameter range and produces
heatmaps, contour plots, and optimization landscapes. A key
differentiator vs RADPF and other DPF tools.

Usage:
    results = run_parameter_sweep("pf1000", "fm", (0.05, 0.3), n_points=20)
    fig = create_sweep_fig(results)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_engine import run_simulation_core


def run_parameter_sweep(
    preset_name: str,
    param_name: str,
    param_range: tuple[float, float],
    n_points: int = 15,
    sim_time_us: float = 20.0,
    fixed_fm: float | None = None,
    fixed_fc: float | None = None,
    progress_fn=None,
) -> dict[str, Any]:
    """Sweep a single parameter and record key metrics.

    Args:
        preset_name: Device preset name.
        param_name: Parameter to sweep ("fm", "fc", "V0_kV", "pressure").
        param_range: (min, max) range for the parameter.
        n_points: Number of sweep points.
        sim_time_us: Simulation time per run.
        fixed_fm: Fixed mass fraction (overrides preset).
        fixed_fc: Fixed current fraction (overrides preset).
        progress_fn: Gradio progress callback.

    Returns:
        Dictionary with parameter values and metric arrays.
    """
    values = np.linspace(param_range[0], param_range[1], n_points)
    results: dict[str, list[float]] = {
        "param_values": [], "I_peak": [], "t_peak": [],
        "dip_pct": [], "Y_neutron": [], "V_pinch_kV": [],
        "T_bennett_keV": [],
    }

    for i, val in enumerate(values):
        if progress_fn:
            progress_fn((i + 1) / n_points, desc=f"{param_name}={val:.3f}")

        kwargs: dict[str, Any] = {"preset_name": preset_name, "sim_time_us": sim_time_us}
        if fixed_fm is not None:
            kwargs["fm"] = fixed_fm
        if fixed_fc is not None:
            kwargs["fc"] = fixed_fc

        if param_name == "fm":
            kwargs["fm"] = val
        elif param_name == "fc":
            kwargs["fc"] = val
        elif param_name == "V0_kV":
            kwargs["V0_kV"] = val
        elif param_name == "pressure":
            kwargs["pressure_torr"] = val
        else:
            continue

        try:
            data = run_simulation_core(**kwargs)
        except Exception:
            continue

        results["param_values"].append(float(val))
        results["I_peak"].append(data.get("I_pre_dip", data["I_peak"]))
        results["t_peak"].append(data.get("t_pre_dip", data["t_peak"]))
        results["dip_pct"].append(data.get("dip_pct", 0))

        ny = data.get("neutron_yield")
        if ny:
            results["Y_neutron"].append(ny["Y_neutron"])
            results["V_pinch_kV"].append(ny.get("V_pinch_kV", 0))
            results["T_bennett_keV"].append(ny.get("T_bennett_keV", 0))
        else:
            results["Y_neutron"].append(0)
            results["V_pinch_kV"].append(0)
            results["T_bennett_keV"].append(0)

    return {
        "param_name": param_name,
        "preset": preset_name,
        "n_points": len(results["param_values"]),
        **{k: np.array(v) for k, v in results.items()},
    }


def run_2d_sweep(
    preset_name: str,
    sim_time_us: float = 20.0,
    fm_range: tuple[float, float] = (0.05, 0.3),
    fc_range: tuple[float, float] = (0.5, 0.9),
    n_fm: int = 10,
    n_fc: int = 10,
    progress_fn=None,
) -> dict[str, Any]:
    """Sweep (fm, fc) 2D parameter space and record I_peak, Yn.

    Returns grid data suitable for heatmap/contour plots.
    """
    fm_vals = np.linspace(fm_range[0], fm_range[1], n_fm)
    fc_vals = np.linspace(fc_range[0], fc_range[1], n_fc)

    I_grid = np.zeros((n_fc, n_fm))
    Y_grid = np.zeros((n_fc, n_fm))
    dip_grid = np.zeros((n_fc, n_fm))

    total = n_fm * n_fc
    count = 0

    for j, fc in enumerate(fc_vals):
        for i, fm in enumerate(fm_vals):
            count += 1
            if progress_fn:
                progress_fn(count / total, desc=f"fm={fm:.3f}, fc={fc:.2f}")
            try:
                data = run_simulation_core(
                    preset_name, sim_time_us, fm=fm, fc=fc,
                )
                I_grid[j, i] = data.get("I_pre_dip", data["I_peak"])
                dip_grid[j, i] = data.get("dip_pct", 0)
                ny = data.get("neutron_yield")
                if ny:
                    Y_grid[j, i] = np.log10(max(ny["Y_neutron"], 1))
            except Exception:
                I_grid[j, i] = np.nan
                Y_grid[j, i] = np.nan
                dip_grid[j, i] = np.nan

    return {
        "preset": preset_name,
        "fm_vals": fm_vals,
        "fc_vals": fc_vals,
        "I_grid": I_grid,
        "Y_grid": Y_grid,
        "dip_grid": dip_grid,
    }


def create_sweep_fig(results: dict[str, Any]) -> go.Figure:
    """Create plots from a 1D parameter sweep."""
    param = results["param_name"]
    x = results["param_values"]

    has_yn = np.any(results["Y_neutron"] > 0)
    n_rows = 3 if has_yn else 2

    titles = [f"I_peak vs {param}", f"Current Dip vs {param}"]
    if has_yn:
        titles.append(f"Neutron Yield vs {param}")

    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                         vertical_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=x, y=results["I_peak"], mode="lines+markers",
        line=dict(color="#2196F3", width=2), marker=dict(size=5),
        name="I_peak [MA]",
    ), row=1, col=1)
    fig.update_yaxes(title_text="I_peak [MA]", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=results["dip_pct"], mode="lines+markers",
        line=dict(color="#FF5722", width=2), marker=dict(size=5),
        name="Current Dip [%]",
    ), row=2, col=1)
    fig.update_yaxes(title_text="Dip [%]", row=2, col=1)

    if has_yn:
        fig.add_trace(go.Scatter(
            x=x, y=results["Y_neutron"], mode="lines+markers",
            line=dict(color="#4CAF50", width=2), marker=dict(size=5),
            name="Yn [neutrons]",
        ), row=3, col=1)
        fig.update_yaxes(title_text="Yn", type="log", row=3, col=1)

    for r in range(1, n_rows + 1):
        fig.update_xaxes(title_text=param, row=r, col=1)

    fig.update_layout(
        height=200 * n_rows + 100, template="plotly_dark", showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40),
        title=f"Parameter Sweep: {results['preset']} ({results['n_points']} points)",
    )
    return fig


def create_2d_sweep_fig(results: dict[str, Any]) -> go.Figure:
    """Create heatmap from a 2D (fm, fc) parameter sweep."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["I_peak [MA]", "log10(Yn)"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Heatmap(
        x=results["fm_vals"], y=results["fc_vals"], z=results["I_grid"],
        colorscale="Viridis", name="I_peak",
        colorbar=dict(title="MA", x=0.45),
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=results["fm_vals"], y=results["fc_vals"], z=results["Y_grid"],
        colorscale="Hot", name="log10(Yn)",
        colorbar=dict(title="log10(n)", x=1.0),
    ), row=1, col=2)

    for c in (1, 2):
        fig.update_xaxes(title_text="Mass Fraction (fm)", row=1, col=c)
        fig.update_yaxes(title_text="Current Fraction (fc)", row=1, col=c)

    fig.update_layout(
        height=500, template="plotly_dark",
        title=f"Parameter Space: {results['preset']}",
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


def format_sweep_markdown(results: dict[str, Any]) -> str:
    """Format sweep results as markdown."""
    param = results["param_name"]
    x = results["param_values"]
    I_arr = results["I_peak"]
    Y_arr = results["Y_neutron"]

    best_I_idx = int(np.argmax(I_arr))
    lines = [
        f"**Parameter Sweep**: {results['preset']}, {param} = "
        f"[{x[0]:.3f}, {x[-1]:.3f}], {results['n_points']} points",
        "",
        f"Peak I_peak = **{I_arr[best_I_idx]:.3f} MA** at {param} = {x[best_I_idx]:.3f}",
    ]

    if np.any(Y_arr > 0):
        best_Y_idx = int(np.argmax(Y_arr))
        lines.append(
            f"Peak Yn = **{Y_arr[best_Y_idx]:.2e}** at {param} = {x[best_Y_idx]:.3f}"
        )

    return "\n".join(lines)
