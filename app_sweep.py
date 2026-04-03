"""Parameter sweep module for DPF web UI.

Runs multiple simulations across a parameter range and produces
heatmaps, contour plots, and optimization landscapes. A key
differentiator vs RADPF and other DPF tools.

Usage:
    results = run_parameter_sweep("pf1000", "fm", (0.05, 0.3), n_points=20)
    fig = create_sweep_fig(results)
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_engine import run_simulation_core

_TEMP_DIR = os.environ.get("DPF_TEMP_DIR", tempfile.gettempdir())

# Preset field -> JAX Lee model param key mapping
_PRESET_TO_JAX: dict[str, str] = {
    "fm": "fm",
    "fc": "fc",
    "V0_kV": "V0",   # converted: kV -> V in _preset_to_jax_params
    "pressure": "fill_pressure_torr",
}


def _preset_to_jax_params(
    preset_name: str,
    override_fm: float | None = None,
    override_fc: float | None = None,
    override_V0_kV: float | None = None,
    override_pressure_torr: float | None = None,
) -> dict:
    """Build JAX Lee model param dict from a preset with optional overrides.

    Args:
        preset_name: Device preset name.
        override_fm: Mass fraction override.
        override_fc: Current fraction override.
        override_V0_kV: Charging voltage override [kV].
        override_pressure_torr: Fill pressure override [Torr].

    Returns:
        JAX float64 parameter dictionary compatible with ``simulate``.
    """
    import jax.numpy as jnp  # noqa: E402, I001
    from dpf.jax.lee_model import default_pf1000_params
    from dpf.presets import _PRESETS

    preset = _PRESETS.get(preset_name, {})
    circuit = preset.get("circuit", {})
    snowplow = preset.get("snowplow", {})

    fill_pressure_pa = preset.get("rho0") or snowplow.get("fill_pressure_Pa", 466.0)
    fill_pressure_torr = fill_pressure_pa / 133.322

    base = default_pf1000_params()
    params = {
        "V0": jnp.float64(circuit.get("V0", float(base["V0"]))),
        "C0": jnp.float64(circuit.get("C", float(base["C0"]))),
        "L0": jnp.float64(circuit.get("L0", float(base["L0"]))),
        "R0": jnp.float64(circuit.get("R0", float(base["R0"]))),
        "a": jnp.float64(circuit.get("anode_radius", float(base["a"]))),
        "b": jnp.float64(circuit.get("cathode_radius", float(base["b"]))),
        "z_max": jnp.float64(snowplow.get("anode_length", float(base["z_max"]))),
        "fill_pressure_torr": jnp.float64(fill_pressure_torr),
        "fc": jnp.float64(snowplow.get("current_fraction", float(base["fc"]))),
        "fm": jnp.float64(snowplow.get("mass_fraction", float(base["fm"]))),
        "fmr": jnp.float64(snowplow.get("radial_mass_fraction", float(base["fmr"]))),
        "fcr": jnp.float64(snowplow.get("radial_current_fraction_2", float(base["fcr"]))),
    }

    if override_fm is not None:
        params["fm"] = jnp.float64(override_fm)
    if override_fc is not None:
        params["fc"] = jnp.float64(override_fc)
    if override_V0_kV is not None:
        params["V0"] = jnp.float64(override_V0_kV * 1e3)
    if override_pressure_torr is not None:
        params["fill_pressure_torr"] = jnp.float64(override_pressure_torr)

    return params


def run_jax_sweep(
    preset_name: str,
    param_name: str,
    param_values: list[float],
    sim_time_us: float,
    fixed_fm: float | None = None,
    fixed_fc: float | None = None,
) -> list[dict[str, Any]]:
    """Run N Lee model simulations in parallel via JAX vmap.

    All simulations share the same preset base params; only ``param_name``
    varies across the batch. Supports the same four parameter names as
    ``run_parameter_sweep``: "fm", "fc", "V0_kV", "pressure".

    Args:
        preset_name: Device preset name.
        param_name: Parameter to sweep ("fm", "fc", "V0_kV", "pressure").
        param_values: Values to sweep over.
        sim_time_us: Simulation time per run [µs].
        fixed_fm: Fixed mass fraction (overrides preset for all runs).
        fixed_fc: Fixed current fraction (overrides preset for all runs).

    Returns:
        List of result dicts, one per simulation, with keys:
        ``I_peak``, ``t_peak``.  neutron_yield / dip are not available
        from the JAX Lee model (scalar-only outputs).
    """
    import jax.numpy as jnp  # noqa: E402, I001
    from dpf.jax.lee_model import vmap_simulate

    if param_name not in _PRESET_TO_JAX:
        raise ValueError(
            f"JAX sweep does not support param '{param_name}'. "
            f"Supported: {list(_PRESET_TO_JAX)}"
        )

    sim_time_s = sim_time_us * 1e-6

    base = _preset_to_jax_params(
        preset_name,
        override_fm=fixed_fm,
        override_fc=fixed_fc,
    )

    vals_arr = jnp.array(param_values, dtype=jnp.float64)
    n = len(param_values)

    # Build batched param dict: replicate base params, override the sweep axis
    jax_key = _PRESET_TO_JAX[param_name]
    batched: dict[str, jnp.ndarray] = {}
    for k, v in base.items():
        if k == jax_key:
            if param_name == "V0_kV":
                batched[k] = vals_arr * 1e3
            else:
                batched[k] = vals_arr
        else:
            batched[k] = jnp.broadcast_to(v, (n,))

    result = vmap_simulate(batched, sim_time=sim_time_s)

    return [
        {
            "I_peak": float(result["I_peak"][i]) / 1e6,  # A -> MA
            "t_peak": float(result["t_peak"][i]) * 1e6,  # s -> µs
            "dip_pct": 0.0,
            "Y_neutron": 0.0,
            "V_pinch_kV": 0.0,
            "T_bennett_keV": 0.0,
        }
        for i in range(n)
    ]


def _get_published_params(preset_name: str) -> dict[str, float | None]:
    """Extract published fc/fm from preset snowplow config."""
    from dpf.presets import _PRESETS
    preset = _PRESETS.get(preset_name, {})
    snowplow = preset.get("snowplow", {})
    return {
        "fc": snowplow.get("current_fraction"),
        "fm": snowplow.get("mass_fraction"),
    }


def run_parameter_sweep(
    preset_name: str,
    param_name: str,
    param_range: tuple[float, float],
    n_points: int = 15,
    sim_time_us: float = 20.0,
    fixed_fm: float | None = None,
    fixed_fc: float | None = None,
    progress_fn=None,
    use_jax: bool = False,
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
        use_jax: When True, route through JAX vmap (all N sims in parallel).
            Faster for large sweeps; returns I_peak and t_peak only
            (neutron_yield, dip_pct are not available from the JAX Lee model).

    Returns:
        Dictionary with parameter values and metric arrays.
    """
    values = np.linspace(param_range[0], param_range[1], n_points)

    if use_jax and param_name in _PRESET_TO_JAX:
        jax_results = run_jax_sweep(
            preset_name=preset_name,
            param_name=param_name,
            param_values=list(values),
            sim_time_us=sim_time_us,
            fixed_fm=fixed_fm,
            fixed_fc=fixed_fc,
        )
        results: dict[str, list[float]] = {
            "param_values": list(values),
            "I_peak": [r["I_peak"] for r in jax_results],
            "t_peak": [r["t_peak"] for r in jax_results],
            "dip_pct": [r["dip_pct"] for r in jax_results],
            "Y_neutron": [r["Y_neutron"] for r in jax_results],
            "V_pinch_kV": [r["V_pinch_kV"] for r in jax_results],
            "T_bennett_keV": [r["T_bennett_keV"] for r in jax_results],
        }
        return {
            "param_name": param_name,
            "preset": preset_name,
            "n_points": len(results["param_values"]),
            **{k: np.array(v) for k, v in results.items()},
        }

    results_serial: dict[str, list[float]] = {
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

        results_serial["param_values"].append(float(val))
        results_serial["I_peak"].append(data.get("I_pre_dip", data["I_peak"]))
        results_serial["t_peak"].append(data.get("t_pre_dip", data["t_peak"]))
        results_serial["dip_pct"].append(data.get("dip_pct", 0))

        ny = data.get("neutron_yield")
        if ny:
            results_serial["Y_neutron"].append(ny["Y_neutron"])
            results_serial["V_pinch_kV"].append(ny.get("V_pinch_kV", 0))
            results_serial["T_bennett_keV"].append(ny.get("T_bennett_keV", 0))
        else:
            results_serial["Y_neutron"].append(0)
            results_serial["V_pinch_kV"].append(0)
            results_serial["T_bennett_keV"].append(0)

    return {
        "param_name": param_name,
        "preset": preset_name,
        "n_points": len(results_serial["param_values"]),
        **{k: np.array(v) for k, v in results_serial.items()},
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

    published = _get_published_params(results["preset"])
    pub_val = published.get(param)
    if pub_val is not None:
        for r in range(1, n_rows + 1):
            fig.add_vline(
                x=pub_val, line_dash="dash", line_color="yellow",
                annotation_text="Published", annotation_position="top",
                annotation_font_color="yellow",
                row=r, col=1,
            )

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

    published = _get_published_params(results["preset"])
    pub_fm = published.get("fm")
    pub_fc = published.get("fc")
    if pub_fm is not None and pub_fc is not None:
        for c in (1, 2):
            fig.add_trace(go.Scatter(
                x=[pub_fm], y=[pub_fc], mode="markers+text",
                text=["Published"], textposition="top center",
                textfont=dict(color="yellow", size=12),
                marker=dict(size=15, color="yellow", symbol="star",
                            line=dict(color="black", width=1)),
                showlegend=False,
            ), row=1, col=c)

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


def export_sweep_csv(results: dict[str, Any]) -> str:
    """Export 1D sweep results to a CSV file. Returns file path."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "parameter_value", "I_peak_MA", "t_peak_us",
        "dip_pct", "Y_neutron", "V_pinch_kV", "T_bennett_keV",
    ])
    for i in range(len(results["param_values"])):
        writer.writerow([
            f"{results['param_values'][i]:.6f}",
            f"{results['I_peak'][i]:.6f}",
            f"{results['t_peak'][i]:.4f}",
            f"{results['dip_pct'][i]:.2f}",
            f"{results['Y_neutron'][i]:.6e}",
            f"{results['V_pinch_kV'][i]:.2f}",
            f"{results['T_bennett_keV'][i]:.4f}",
        ])
    path = os.path.join(
        _TEMP_DIR,
        f"sweep_{results['preset']}_{results['param_name']}_{os.getpid()}.csv",
    )
    with open(path, "w") as f:
        f.write(buf.getvalue())
    return path
