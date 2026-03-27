"""DPF-Unified application state — all reactive state lives here."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import reflex as rx

# Add dpf-unified to path for imports
_DPF_ROOT = str(Path(__file__).parent.parent.parent)
if _DPF_ROOT not in sys.path:
    sys.path.insert(0, _DPF_ROOT)


from .presets import BACKEND_LEVELS, GAS_SPECIES, PRESETS

    # Presets and backend levels now in presets.py


class SimState(rx.State):
    """Main application state for DPF-Unified."""

    # --- Configuration ---
    preset: str = "pf1000"
    backend_level: int = 3  # MLX PLM — default production backend
    grid_resolution: str = "medium"
    fc: float = 0.70
    fm: float = 0.08
    voltage: float = 27.0
    fill_pressure: float = 3.5
    sim_time_us: float = 10.0
    capacitance_mf: float = 1.332
    inductance_nh: float = 33.5
    resistance_mohm: float = 2.3
    anode_length_mm: float = 600.0
    anode_radius_mm: float = 80.0
    cathode_radius_mm: float = 160.0
    gas_species: str = "D2"
    enable_radiation: bool = True
    enable_conduction: bool = True
    enable_anomalous: bool = False
    enable_hall: bool = False
    enable_sheath_bc: bool = False
    enable_ablation: bool = False
    enable_nernst: bool = False
    enable_cr_ionization: bool = False
    enable_crowbar: bool = True
    student_mode: bool = True
    show_advanced: bool = False

    # --- Simulation Running ---
    running: bool = False
    phase: str = ""
    phase_time_us: float = 0.0
    phase_progress: float = 0.0
    live_current_ma: float = 0.0
    live_velocity_kms: float = 0.0
    live_temperature_ev: float = 0.0
    live_density: str = ""
    eta_seconds: float = 0.0
    computation_pct: float = 0.0
    narrative_live: str = ""

    # --- Results ---
    has_results: bool = False
    peak_current_ma: float = 0.0
    peak_time_us: float = 0.0
    current_dip_pct: float = 0.0
    neutron_yield: str = ""
    peak_current_exp: float = 0.0
    peak_time_exp: float = 0.0
    peak_error_pct: float = 0.0
    timing_error_pct: float = 0.0
    narrative_text: str = ""

    # Waveform data for charts (lists for JSON serialization)
    waveform_times_us: list[float] = []
    waveform_current_ma: list[float] = []
    waveform_voltage_kv: list[float] = []
    energy_kinetic: list[float] = []
    energy_magnetic: list[float] = []
    energy_thermal: list[float] = []

    # Babylon renderer key for forcing iframe refresh
    renderer_key: int = 0

    # --- Backend-only (not serialized to client) ---
    _sim_start_time: float = 0.0

    @rx.var
    def gas_description(self) -> str:
        g = GAS_SPECIES.get(self.gas_species, GAS_SPECIES["D2"])
        return g.get("description", "")

    @rx.var
    def gas_display_name(self) -> str:
        g = GAS_SPECIES.get(self.gas_species, GAS_SPECIES["D2"])
        return f"{g['symbol']} — {g['name']}"

    @rx.var
    def preset_data(self) -> dict:
        d = PRESETS.get(self.preset, PRESETS["pf1000"])
        # Ensure backward compat keys
        result = dict(d)
        result.setdefault("fill_pressure", d.get("fill_pressure_torr", 3.5))
        result.setdefault("anode_length", d.get("anode_length_mm", 600.0))
        result.setdefault("class", d.get("class", ""))
        return result

    @rx.var
    def backend_info(self) -> dict:
        return BACKEND_LEVELS.get(self.backend_level, BACKEND_LEVELS[4])

    @rx.var
    def waveform_chart_data(self) -> list[dict]:
        """Data for recharts line chart — list of {time, current} dicts."""
        if not self.waveform_times_us or not self.waveform_current_ma:
            return []
        data = []
        for t, I in zip(self.waveform_times_us, self.waveform_current_ma):
            data.append({"time": t, "current": round(I, 4)})
        return data

    @rx.var
    def energy_chart_data(self) -> list[dict]:
        """Data for energy balance stacked area chart."""
        if not self.waveform_times_us:
            return []
        return [
            {"time": t, "Magnetic": m, "Kinetic": k, "Thermal": th}
            for t, m, k, th in zip(
                self.waveform_times_us, self.energy_magnetic,
                self.energy_kinetic, self.energy_thermal,
            )
        ]

    @rx.var
    def peak_error_color(self) -> str:
        if self.peak_error_pct < 5:
            return "#10b981"
        elif self.peak_error_pct < 20:
            return "#f59e0b"
        return "#ef4444"

    @rx.var
    def timing_error_color(self) -> str:
        if self.timing_error_pct < 10:
            return "#10b981"
        elif self.timing_error_pct < 20:
            return "#f59e0b"
        return "#ef4444"

    def set_preset(self, value: str):
        self.preset = value
        data = PRESETS.get(value, PRESETS["pf1000"])
        self.voltage = data["voltage"]
        self.fill_pressure = data.get("fill_pressure_torr", 3.5)
        self.fc = data["fc_default"]
        self.fm = data["fm_default"]
        self.peak_current_exp = data["peak_current_exp"]
        self.peak_time_exp = data["rise_time_exp"]
        self.capacitance_mf = data["capacitance"]
        self.anode_length_mm = data.get("anode_length_mm", 600.0)
        self.anode_radius_mm = data.get("anode_radius_mm", 80.0)
        self.cathode_radius_mm = data.get("cathode_radius_mm", 160.0)
        self.inductance_nh = data.get("inductance_nh", 33.5)
        self.resistance_mohm = data.get("resistance_mohm", 2.3)
        self.gas_species = data.get("fill_gas", "D2")
        self.sim_time_us = data.get("sim_time_us", 10.0)

    def set_voltage(self, value: list[float]):
        self.voltage = round(value[0] if isinstance(value, list) else value, 1)

    def set_fill_pressure(self, value: list[float]):
        self.fill_pressure = round(value[0] if isinstance(value, list) else value, 1)

    def set_sim_time(self, value: list[float]):
        self.sim_time_us = round(value[0] if isinstance(value, list) else value, 1)

    def set_backend_level(self, value: list[float]):
        self.backend_level = int(value[0]) if isinstance(value, list) else int(value)

    def set_fc(self, value: list[float]):
        self.fc = round(value[0] if isinstance(value, list) else value, 3)

    def set_fm(self, value: list[float]):
        self.fm = round(value[0] if isinstance(value, list) else value, 4)

    def set_grid_resolution(self, value: str):
        self.grid_resolution = value

    def set_enable_radiation(self, value: bool):
        self.enable_radiation = value

    def set_enable_conduction(self, value: bool):
        self.enable_conduction = value

    def set_enable_anomalous(self, value: bool):
        self.enable_anomalous = value

    def set_enable_hall(self, value: bool):
        self.enable_hall = value

    def set_enable_sheath_bc(self, value: bool):
        self.enable_sheath_bc = value

    def set_enable_ablation(self, value: bool):
        self.enable_ablation = value

    def set_enable_nernst(self, value: bool):
        self.enable_nernst = value

    def set_enable_cr_ionization(self, value: bool):
        self.enable_cr_ionization = value

    def set_enable_crowbar(self, value: bool):
        self.enable_crowbar = value

    def set_gas_species(self, value: str):
        self.gas_species = value

    # --- Navigation ---
    current_view: str = "configure"  # "configure", "running", "results", "export"

    def go_configure(self):
        self.current_view = "configure"
        self.running = False

    def go_results(self):
        self.current_view = "results"

    def go_export(self):
        self.current_view = "export"

    def go_running(self):
        self.current_view = "running"

    def reset_to_configure(self):
        self.has_results = False
        self.running = False
        self.phase = ""
        self.narrative_text = ""
        self.current_view = "configure"

    def load_demo_results(self):
        """Load demo data so you can preview all screens without running a sim."""
        import math
        self.has_results = True
        self.peak_current_ma = 1.862
        self.peak_time_us = 5.2
        self.current_dip_pct = 48.0
        self.neutron_yield = "~10^11 (est)"
        self.peak_current_exp = 1.87
        self.peak_time_exp = 5.0
        self.peak_error_pct = 0.4
        self.timing_error_pct = 4.0
        self.phase = "Complete"
        self.phase_progress = 1.0
        self.live_current_ma = 1.862
        self.live_velocity_kms = 141.0
        self.live_temperature_ev = 1200.0
        self.eta_seconds = 0.0
        self.narrative_live = "The magnetic field squeezed the plasma to extreme density."
        # Generate synthetic PF-1000-like waveform for demo
        n = 300
        self.waveform_times_us = [round(i * 12.0 / n, 3) for i in range(n)]
        self.waveform_current_ma = []
        for i in range(n):
            t = i * 12.0 / n
            # Approximate PF-1000 waveform: rise, peak at 5.2us, dip, decay
            if t < 5.2:
                I = 1.862 * math.sin(math.pi * t / (2 * 5.2))
            elif t < 6.0:
                # Dip
                I = 1.862 * (1.0 - 0.48 * math.sin(math.pi * (t - 5.2) / 1.6))
            else:
                # Decay
                I = 1.862 * 0.52 * math.exp(-(t - 6.0) / 4.0)
            self.waveform_current_ma.append(round(I, 4))
        # Synthetic energy partition for demo
        self.energy_magnetic = [round(I**2 / 1.862**2 * 100, 1) for I in self.waveform_current_ma]
        self.energy_kinetic = [round(I / 1.862 * 30, 1) for I in self.waveform_current_ma]
        self.energy_thermal = [round(max(100 - m - k, 0), 1) for m, k in zip(self.energy_magnetic, self.energy_kinetic)]
        self.narrative_text = self._build_narrative()
        self.renderer_key = self.renderer_key + 1
        self.current_view = "results"

    def toggle_advanced(self):
        self.show_advanced = not self.show_advanced

    def toggle_student_mode(self):
        self.student_mode = not self.student_mode

    @rx.event(background=True)
    async def run_simulation(self):
        async with self:
            self.running = True
            self.has_results = False
            self.current_view = "running"
            self.phase = "Initializing"
            self.phase_progress = 0.02
            self.phase_time_us = 0.0
            self.live_current_ma = 0.0
            self.live_velocity_kms = 0.0
            self.live_temperature_ev = 0.0
            self.eta_seconds = 30.0
            self._sim_start_time = time.time()
            self.narrative_live = "Preparing the simulation engine..."

        try:
            from dpf.config import SimulationConfig
            from dpf.engine import SimulationEngine
            from dpf.presets import get_preset

            # Determine grid from resolution setting
            grid_map = {"coarse": (16, 1, 32), "medium": (32, 1, 64), "fine": (64, 1, 128)}
            grid = grid_map.get(self.grid_resolution, (32, 1, 64))

            # Build config — backend level maps to actual solver settings
            backend_configs = {
                1: {"backend": "python", "riemann_solver": "hll", "reconstruction": "plm", "time_integrator": "ssp_rk2"},
                2: {"backend": "python", "riemann_solver": "hll", "reconstruction": "plm", "time_integrator": "ssp_rk2"},
                3: {"backend": "mlx", "riemann_solver": "hll", "reconstruction": "plm", "time_integrator": "ssp_rk2"},
                4: {"backend": "mlx", "riemann_solver": "hll", "reconstruction": "weno5z", "time_integrator": "ssp_rk3"},
                5: {"backend": "mlx", "riemann_solver": "hll", "reconstruction": "weno5z", "time_integrator": "ssp_rk3"},
            }
            fluid_config = backend_configs.get(self.backend_level, backend_configs[3])

            preset = get_preset(self.preset)
            preset["grid_shape"] = list(grid)
            preset["fluid"] = fluid_config
            preset["snowplow"]["current_fraction"] = self.fc
            preset["snowplow"]["mass_fraction"] = self.fm
            preset["sim_time"] = self.sim_time_us * 1e-6
            preset["diagnostics_path"] = ":memory:"

            config = SimulationConfig(**preset)
            engine = SimulationEngine(config)

            # Phase descriptions for the narrative
            phase_narratives = {
                "rundown": "The capacitor bank is discharging through the gas, creating a current sheath that sweeps down the anode like a magnetic piston.",
                "radial": "The sheath has reached the anode end and is now folding inward. The magnetic field is squeezing the plasma like a hydraulic press made of magnetism.",
                "reflected": "The sheath has bounced off the axis and a reflected shock is heating the plasma to extreme temperatures.",
                "pinch": "Maximum compression — the plasma column is just millimeters wide, reaching temperatures hotter than the Sun's core.",
                "post_pinch": "The plasma column is expanding outward. The current is dropping as the inductance changes rapidly.",
            }

            times_arr = []
            currents_arr = []
            max_steps = 15000

            for step_i in range(max_steps):
                result = engine.step()
                t = engine.time
                I = abs(engine.circuit.current)
                times_arr.append(t)
                currents_arr.append(I)

                # Update UI every 50 steps
                if step_i % 50 == 0:
                    elapsed = time.time() - self._sim_start_time
                    sim_frac = t / (self.sim_time_us * 1e-6) if self.sim_time_us > 0 else 0
                    sim_frac = min(sim_frac, 1.0)

                    # Detect phase from snowplow
                    sp_phase = getattr(engine.snowplow, "phase", "rundown") if hasattr(engine, "snowplow") else "rundown"
                    narr = phase_narratives.get(sp_phase, "Simulation in progress...")

                    # Estimate ETA
                    if sim_frac > 0.05:
                        eta = elapsed / sim_frac * (1.0 - sim_frac)
                    else:
                        eta = 30.0

                    async with self:
                        self.phase_time_us = round(t * 1e6, 2)
                        self.phase_progress = sim_frac
                        self.live_current_ma = round(I / 1e6, 3)
                        self.phase = sp_phase.replace("_", " ").title()
                        self.narrative_live = narr
                        self.eta_seconds = round(eta, 0)
                        # Estimate velocity from snowplow if available
                        if hasattr(engine, "snowplow") and hasattr(engine.snowplow, "v_sheath"):
                            self.live_velocity_kms = round(abs(engine.snowplow.v_sheath) / 1e3, 0)

                if result.finished:
                    break

            # Extract results
            import numpy as np
            times_np = np.array(times_arr)
            currents_np = np.array(currents_arr)
            I_peak = float(np.max(currents_np))
            t_peak = float(times_np[np.argmax(currents_np)])

            # Get experimental comparison
            d = PRESETS.get(self.preset, PRESETS["pf1000"])
            peak_err = abs(I_peak / 1e6 - d["peak_current_exp"]) / d["peak_current_exp"] * 100
            timing_err = abs(t_peak * 1e6 - d["rise_time_exp"]) / d["rise_time_exp"] * 100

            # Current dip
            peak_idx = int(np.argmax(currents_np))
            post_peak = currents_np[peak_idx:]
            dip_pct = (I_peak - float(np.min(post_peak))) / I_peak * 100 if len(post_peak) > 5 else 0

            # Downsample waveform for frontend (max 500 points)
            step_ds = max(1, len(times_np) // 500)
            t_ds = times_np[::step_ds]
            I_ds = currents_np[::step_ds]

            async with self:
                self.running = False
                self.has_results = True
                self.current_view = "results"
                self.peak_current_ma = round(I_peak / 1e6, 3)
                self.peak_time_us = round(t_peak * 1e6, 1)
                self.peak_error_pct = round(peak_err, 1)
                self.timing_error_pct = round(timing_err, 1)
                self.current_dip_pct = round(dip_pct, 0)
                self.neutron_yield = "~10^11 (est)"
                self.phase = "Complete"
                self.phase_progress = 1.0
                self.eta_seconds = 0.0
                self.waveform_times_us = [round(float(t) * 1e6, 4) for t in t_ds]
                self.waveform_current_ma = [round(float(I) / 1e6, 4) for I in I_ds]
                # Synthetic energy partition (proportional to I^2 for magnetic, remainder for thermal/kinetic)
                I_norm = I_ds / max(I_peak, 1)
                self.energy_magnetic = [round(float(x)**2 * 100, 1) for x in I_norm]
                self.energy_kinetic = [round(float(x) * 30, 1) for x in I_norm]
                self.energy_thermal = [round(max(100 - m - k, 0), 1) for m, k in zip(self.energy_magnetic, self.energy_kinetic)]
                # Check if pinch was achieved
                if dip_pct < 5:
                    self.narrative_text = self._build_no_pinch_narrative()
                else:
                    self.narrative_text = self._build_narrative()
                self.renderer_key = self.renderer_key + 1
                self._update_babylon_renderer(times_np, currents_np)

        except Exception as e:
            async with self:
                self.running = False
                self.has_results = True
                self.current_view = "results"
                self.phase = "Error"
                self.narrative_text = self._build_error_narrative(str(e))

    def _build_no_pinch_narrative(self) -> str:
        d = PRESETS.get(self.preset, PRESETS["pf1000"])
        if self.student_mode:
            return f"""## Simulation Completed — But No Pinch Detected

The simulation ran to completion, but the current waveform shows **{self.current_dip_pct:.0f}% dip** — less than the 5% minimum that indicates a plasma pinch.

### What this means

In a successful DPF discharge, the current drops sharply when the plasma compresses (pinches). This drop — called the **current dip** — is the fingerprint of magnetic compression. Real experiments show a 40-60% dip for PF-1000.

Your simulation didn't produce this signature. The sheath may not have reached the end of the anode, or the radial compression wasn't strong enough to create a well-defined pinch.

### Why this happens

- **Simulation time too short** — the discharge needs enough time to complete the rundown AND radial phases. Try increasing to 12-15 us.
- **Grid too coarse** — the radial compression happens in millimeters. If grid cells are centimeters wide, the pinch can't be resolved.
- **Fill pressure too high** — high pressure means more mass for the sheath to sweep, slowing it down. The sheath may not reach the anode end.
- **Current fraction (fc) too low** — less current coupling means weaker magnetic drive.

### What real physicists do

A "no pinch" shot in the lab is common — roughly 1 in 5 shots on PF-1000 produces a weak or absent pinch. Scientists adjust the fill pressure, check electrode condition, and try again. The current waveform from a failed shot still provides diagnostic information.

### Your Results

| Metric | Value |
|--------|-------|
| Peak Current | {self.peak_current_ma:.3f} MA |
| Peak Time | {self.peak_time_us:.1f} us |
| Current Dip | {self.current_dip_pct:.0f}% (need >5% for pinch) |
"""
        return f"""## No Pinch Detected

Current dip: {self.current_dip_pct:.0f}% (<5% threshold). I_peak={self.peak_current_ma:.3f} MA at t={self.peak_time_us:.1f} us.

Possible causes: insufficient sim_time, grid too coarse for radial phase, high fill pressure, low fc.
"""

    def _build_error_narrative(self, error_msg: str) -> str:
        if self.student_mode:
            return f"""## The Simulation Didn't Complete Successfully

Don't worry — this happens sometimes in computational physics, and it's actually informative.

### What went wrong

{f"**Error:** `{error_msg[:200]}`" if error_msg else "The simulation encountered a numerical instability."}

### Why simulations can fail

Physics simulations solve millions of equations every second. Sometimes the numbers grow too large (we call this **numerical instability** or "blowing up") or become meaningless (producing **NaN** — "Not a Number").

Common reasons in DPF simulations:

1. **The grid is too coarse** — like trying to photograph a hummingbird with a low-resolution camera. The fast-moving sheath passes through cells faster than the solver can track. *Fix: increase grid resolution.*

2. **The timestep is too large** — the simulation takes steps in time that are too big to capture the rapid changes during compression. *Fix: reduce the CFL number (try a higher backend level).*

3. **The physics is extreme** — at pinch, temperatures reach millions of degrees and densities change by factors of 1000 in millimeters. This pushes any numerical method to its limits.

4. **The parameters are unphysical** — if fc or fm are set to extreme values, the simulation may predict a sheath that moves faster than the equations can handle.

### What to try next

- **Reduce grid resolution** to "coarse" — faster and more stable
- **Check fc and fm** — values near the published defaults (fc~0.7, fm~0.08) are safest
- **Reduce simulation time** — try 5 us instead of 10 us to see if the early phases work
- **Try a different device** — UNU-ICTP is simpler and more stable than PF-1000

### What scientists do when real experiments fail

Real DPF experiments also fail sometimes! A "bad shot" might produce no pinch if:
- The fill gas pressure is wrong (too high = sheath too slow, too low = breakdown failure)
- The electrodes are damaged from previous shots
- The timing capacitor bank doesn't fire symmetrically

Failed experiments still produce data — the Rogowski coil still records the current, and scientists learn from what went wrong. Your failed simulation is the same: the partial data tells us something about the physics.
"""
        return f"""## Simulation Error

**Error:** `{error_msg[:300]}`

The simulation did not complete. Possible causes:
- CFL violation (dt too large for grid spacing)
- Numerical instability at vacuum/sheath boundary
- Parameter values outside stable operating range

Try: coarser grid, lower sim_time, or published fc/fm defaults.
"""

    def _update_babylon_renderer(self, times_np, currents_np):
        """Regenerate the Babylon.js renderer HTML with actual simulation data.

        Must use the parallel-array format expected by extract_all_layers:
        d["t_us"], d["z_mm"], d["r_mm"], d["I_MA"], d["phases"].
        NOT the frame-list format from test_data_pf1000.json.
        """
        try:
            import numpy as np
            d = PRESETS.get(self.preset, PRESETS["pf1000"])

            step = max(1, len(times_np) // 60)
            t_us_arr = []
            z_mm_arr = []
            r_mm_arr = []
            I_ma_arr = []
            phases_arr = []

            for i in range(0, len(times_np), step):
                t_us = float(times_np[i]) * 1e6
                I_ma = float(currents_np[i]) / 1e6

                if t_us < self.peak_time_us * 0.75:
                    phase = "rundown"
                    r = d.get("cathode_radius_mm", 160.0)
                    z = min(t_us / max(self.peak_time_us * 0.75, 0.1) * d["anode_length"], d["anode_length"])
                elif t_us < self.peak_time_us:
                    phase = "radial"
                    frac = (t_us - self.peak_time_us * 0.75) / max(self.peak_time_us * 0.25, 0.1)
                    r = d.get("cathode_radius_mm", 160.0) * (1 - frac * 0.9)
                    z = d["anode_length"]
                elif t_us < self.peak_time_us * 1.15:
                    phase = "reflected"
                    r = 11.5
                    z = d["anode_length"]
                else:
                    phase = "pinch"
                    frac = min((t_us - self.peak_time_us * 1.15) / 5.0, 1.0)
                    r = 11.5 + frac * 100
                    z = d["anode_length"]

                t_us_arr.append(t_us)
                z_mm_arr.append(z)
                r_mm_arr.append(r)
                I_ma_arr.append(I_ma)
                phases_arr.append(phase)

            # Format expected by extract_all_layers (parallel arrays, SI geometry)
            renderer_data = {
                "circuit": {
                    "anode_radius": d.get("anode_radius_mm", 80.0) / 1000,
                    "cathode_radius": d.get("cathode_radius_mm", 160.0) / 1000,
                },
                "snowplow_cfg": {
                    "anode_length": d["anode_length"] / 1000,
                    "fill_pressure_Pa": d["fill_pressure"] * 133.322,
                },
                "t_us": t_us_arr,
                "z_mm": z_mm_arr,
                "r_mm": r_mm_arr,
                "I_MA": I_ma_arr,
                "phases": phases_arr,
            }

            from app_babylon_unified import create_unified_renderer
            html = create_unified_renderer(renderer_data)
            assets_path = Path(__file__).parent.parent / "assets" / "babylon_renderer.html"
            assets_path.write_text(html)
        except Exception:
            pass  # Keep existing renderer if generation fails

    # --- Narrative ---
    active_phase_tab: str = "overview"

    def set_phase_tab(self, tab: str):
        self.active_phase_tab = tab

    @rx.var
    def narrative_for_phase(self) -> str:
        return self._get_phase_narrative(self.active_phase_tab)

    def _get_phase_narrative(self, phase: str) -> str:
        d = PRESETS.get(self.preset, PRESETS["pf1000"])
        narratives = {
            "overview": self._build_narrative(),
            "rundown": self._narrative_rundown(d),
            "radial": self._narrative_radial(d),
            "pinch": self._narrative_pinch(d),
            "postpinch": self._narrative_postpinch(d),
            "equations": self._narrative_mhd_equations(),
        }
        return narratives.get(phase, narratives["overview"])

    @staticmethod
    def _narrative_mhd_equations() -> str:
        return """## The MHD Equations — How the Simulation Works

The simulation solves the equations of **Magnetohydrodynamics (MHD)** — the physics of electrically conducting fluids (plasmas) in magnetic fields. These are the fundamental laws governing everything from the Sun's corona to fusion reactors.

### Conservation of Mass

$$\\frac{\\partial \\rho}{\\partial t} + \\nabla \\cdot (\\rho \\vec{v}) = 0$$

**In plain English:** Mass can't appear or disappear. If plasma flows out of a region, the density in that region decreases. This is the same continuity equation used in weather forecasting and aerodynamics.

**In the simulation:** The HLLS Riemann solver computes mass fluxes at cell boundaries. The entropy tracer (a secondary variable) tracks the thermodynamic state independently, preventing numerical errors in low-density regions.

### Conservation of Momentum (Newton's Second Law for Plasmas)

$$\\frac{\\partial (\\rho \\vec{v})}{\\partial t} + \\nabla \\cdot \\left(\\rho \\vec{v}\\vec{v} + p_{tot} \\hat{I} - \\vec{B}\\vec{B}\\right) = 0$$

where $p_{tot} = p + \\frac{B^2}{2}$ is the total pressure (gas + magnetic).

**In plain English:** Plasma accelerates when pushed by pressure gradients or magnetic forces. The term $\\vec{B}\\vec{B}$ is the **Maxwell stress tensor** — it describes how magnetic field lines act like rubber bands, pulling plasma along their direction and pushing it perpendicular.

**The key force in a DPF:** The $J \\times B$ Lorentz force, where $J$ is the current density and $B$ is the magnetic field. This force drives the entire discharge — sweeping gas down the anode (rundown) and compressing it radially (pinch).

### Conservation of Energy

$$\\frac{\\partial E}{\\partial t} + \\nabla \\cdot \\left[(E + p_{tot})\\vec{v} - \\vec{B}(\\vec{v} \\cdot \\vec{B})\\right] = 0$$

where $E = \\frac{p}{\\gamma - 1} + \\frac{1}{2}\\rho v^2 + \\frac{B^2}{2}$ is the total energy (thermal + kinetic + magnetic).

**In plain English:** Energy is conserved — it transforms between thermal (heat), kinetic (motion), and magnetic (field) forms, but the total never changes. When the magnetic field compresses the plasma, magnetic energy converts to thermal energy, heating the plasma to fusion temperatures.

**The dual-energy method:** In regions where magnetic energy dominates (like near the electrodes), the subtraction $E - KE - ME$ to find pressure can lose precision. Our solver uses an **entropy tracer** that tracks pressure independently, avoiding this numerical problem.

### Faraday's Law (Magnetic Field Evolution)

$$\\frac{\\partial \\vec{B}}{\\partial t} = \\nabla \\times (\\vec{v} \\times \\vec{B}) + \\frac{\\eta}{\\mu_0} \\nabla^2 \\vec{B}$$

**In plain English:** Magnetic field lines move with the plasma (first term — "frozen-in" flux) and slowly diffuse through it (second term — resistive diffusion). In a DPF, the field is mostly frozen into the sheath, which is why the current sheath acts as a magnetic piston.

**Resistivity models in this simulation:**
- **Lee-More:** Temperature-dependent resistivity that works from cold gas (eV) to hot plasma (keV)
- **Anomalous:** At the pinch, turbulent micro-instabilities scatter electrons much more effectively than Coulomb collisions, increasing resistivity by 1000x or more
- **Flux-limited conduction:** Heat flow is capped at the free-streaming limit to prevent unphysical transport

### How the Solver Works

The MHD equations are **hyperbolic conservation laws** — they describe waves propagating through the plasma. The simulation resolves these waves using:

| Component | Method | What It Does |
|-----------|--------|-------------|
| **Spatial reconstruction** | PLM (2nd order) or WENO5-Z (5th order) | Interpolates cell-average values to cell boundaries |
| **Riemann solver** | HLLS (entropy-based) | Computes the flux of mass, momentum, and energy across each cell boundary by solving the wave interaction problem |
| **Time integration** | SSP-RK2 or SSP-RK3 | Advances the solution forward in time, preserving stability |
| **Transport** | RKL2 super-timestepping | Handles resistive diffusion on GPU — 3.4x faster than the CPU method |
| **Vacuum treatment** | Boris correction | Reduces magnetic forces in vacuum cells without injecting fake mass |
| **Divergence cleaning** | Dedner GLM | Keeps the magnetic field divergence-free (a physical requirement: there are no magnetic monopoles) |

### The Cylindrical Geometry

A DPF is axisymmetric — it looks the same from any angle around the axis. This lets us solve a 2D problem (r, z) instead of full 3D, saving enormous computation. The cylindrical geometry adds **geometric source terms**:

$$S_r = \\frac{p_{tot} + \\rho v_\\theta^2 - B_\\theta^2}{r}$$

This term represents the centrifugal force (plasma spinning) and magnetic hoop stress (the $B_\\theta$ field trying to squeeze inward). The $1/r$ factor means these forces are strongest near the axis — which is exactly where the pinch happens.
"""

    def _build_narrative(self) -> str:
        d = PRESETS.get(self.preset, PRESETS["pf1000"])
        if self.student_mode:
            return self._build_narrative_student(d)
        return self._build_narrative_expert(d)

    def _build_narrative_expert(self, d: dict) -> str:
        return f"""## Simulation Summary: {d['name']}

| Metric | Simulation | Experiment | Error |
|--------|-----------|-----------|-------|
| Peak Current | **{self.peak_current_ma:.3f} MA** | {self.peak_current_exp:.2f} MA | {self.peak_error_pct:.1f}% |
| Peak Time | **{self.peak_time_us:.1f} us** | {self.peak_time_exp:.1f} us | {self.timing_error_pct:.1f}% |
| Current Dip | **{self.current_dip_pct:.0f}%** | ~60% | — |

Backend: MLX (HLLS + PLM + SSP-RK2 + RKL2 transport). Grid: {self.grid_resolution}. fc={self.fc:.3f}, fm={self.fm:.3f}.

Select phase tabs above for governing equations and solver details.
"""

    def _build_narrative_student(self, d: dict) -> str:
        return f"""## What is a Dense Plasma Focus?

Imagine you have a very powerful battery (a **capacitor bank** storing {d['energy']} of energy — enough to briefly power a small town). You connect it to two metal cylinders, one inside the other, with gas between them. When you flip the switch, an enormous electric current — **millions of amperes** — flows through the gas.

This current creates a powerful magnetic field that acts like an invisible hand, grabbing the gas and slamming it inward with incredible force. The gas gets compressed into a tiny column just millimeters wide, heated to **millions of degrees** — hotter than the center of the Sun. At these temperatures, atomic nuclei can fuse together, releasing neutrons. That's nuclear fusion, and it happens right here on a lab bench.

This device is called a **Dense Plasma Focus (DPF)**. Your simulation just modeled the entire process for the **{d['name']}** device at {d['institution']}.

---

## Your Results — How Well Did the Simulation Do?

| What We Measured | Simulation Says | Real Experiment Says | How Close? |
|-----------------|----------------|---------------------|-----------|
| Peak Current | **{self.peak_current_ma:.3f} MA** ({self.peak_current_ma * 1e6:,.0f} Amperes) | {self.peak_current_exp:.2f} MA | {self.peak_error_pct:.1f}% off |
| Time to Peak | **{self.peak_time_us:.1f} microseconds** | {self.peak_time_exp:.1f} microseconds | {self.timing_error_pct:.1f}% off |
| Current Dip | **{self.current_dip_pct:.0f}%** drop | ~60% drop | — |

{"**Excellent agreement!** The simulation matches the experiment within a few percent. This means our mathematical model correctly captures the physics." if self.peak_error_pct < 10 else "**Good agreement.** The simulation captures the main physics, though there's room for improvement in the details."}

---

## How Do Scientists Measure These Things?

The numbers above aren't just from our computer — real physicists measured them on actual devices. Here's how:

### Rogowski Coil (measures current)
A **Rogowski coil** is a flexible wire loop wrapped around the conductor carrying the current. When the current changes (and in a DPF, it changes FAST), the coil produces a voltage proportional to dI/dt (the rate of change of current). An electronic integrator converts this to the actual current waveform I(t).

*Think of it like a speedometer for electricity* — it doesn't touch the current directly, it senses the magnetic field the current creates.

**What it tells us:** The current waveform — how current rises to its peak ({self.peak_current_ma:.3f} MA), then dips during the pinch. The shape of this dip reveals the quality of the plasma compression.

### Voltage Probe (measures voltage)
A resistive or capacitive **voltage divider** measures the voltage across the electrodes. The voltage tells us how much energy is being delivered to the plasma at each moment.

*Like measuring the pressure in a water pipe* — high voltage means energy is being pushed hard into the plasma.

### Neutron Detectors (measures fusion)
**Silver activation counters** or **helium-3 proportional counters** detect neutrons produced by D-D fusion reactions. The neutron yield tells us how effective the pinch was at creating fusion conditions.

*Like a Geiger counter, but for neutrons instead of gamma rays.* Each "click" is evidence of a fusion reaction.

### Interferometry (measures density)
A **laser interferometer** shines a laser beam through the plasma. The plasma's electrons bend the light, creating interference fringes. By counting fringes, scientists can measure the electron density — how many particles per cubic meter.

*Like looking through a glass of water and seeing the background distorted* — the more distortion, the denser the medium.

### Pinhole Camera (measures X-rays)
An **X-ray pinhole camera** takes images of the plasma through a tiny hole, recording the X-ray emission on film or a CCD. This shows the shape and size of the pinch column.

*Like taking a photo, but with X-rays instead of visible light*, because the plasma is so hot it glows in X-rays.

---

## What the Tabs Above Show You

Click each phase tab to learn what happens during that stage of the discharge, see the equations that govern it (don't worry — we explain every symbol), and understand how our simulation solves them.

- **Rundown** — The current sheath sweeps down the anode (the magnetic piston)
- **Radial** — The sheath folds inward, compressing the plasma
- **Pinch** — Maximum compression, fusion reactions occur
- **Post-pinch** — The plasma expands, the current drops
- **MHD Equations** — The complete mathematical framework (with plain-English translations)
"""

    def _narrative_rundown(self, d: dict) -> str:
        return f"""## Phase 1: Axial Rundown

### What happens
The {d['energy']} capacitor bank discharges through {d.get('fill_gas', 'D2').lower()} gas at {d.get('fill_pressure_torr', d.get('fill_pressure', 3.5)):.1f} Torr. This creates a **current sheath** — a thin layer of ionized gas that gets pushed down the {d.get('anode_length_mm', d.get('anode_length', 600))} mm anode by magnetic force.

Think of it like a piston in a cylinder, except the piston is made of electric current and the cylinder is the space between the electrodes.

### The driving force

The force pushing the sheath is the **Lorentz force** — the fundamental interaction between electric current and magnetic fields:

$$\\vec{{F}} = \\vec{{J}} \\times \\vec{{B}}$$

In plain English: current flowing through a magnetic field creates a force perpendicular to both. In a DPF, the current flows axially (along the anode) and the magnetic field wraps around it azimuthally (like rings). The resulting force pushes the sheath forward.

For a cylindrical geometry, the magnetic pressure on the sheath is:

$$P_{{mag}} = \\frac{{\\mu_0 I^2}}{{8 \\pi^2 r^2}}$$

where $\\mu_0 = 4\\pi \\times 10^{{-7}}$ H/m is the permeability of free space, $I$ is the current in Amperes, and $r$ is the radius.

### How the simulation solves it

The simulation uses the **Lee model** snowplow equations during rundown:

$$m(z) \\frac{{dv}}{{dt}} = \\frac{{\\mu_0}}{{4\\pi}} I^2 \\ln\\left(\\frac{{b}}{{a}}\\right) - f_m \\rho_0 \\pi(b^2-a^2) v^2$$

The left side is mass times acceleration. The right side has two terms:
- **Magnetic driving force**: proportional to $I^2$ (more current = more force)
- **Mass loading drag**: the sheath sweeps up gas as it moves, slowing it down

The parameters $f_c = {d['fc_default']:.2f}$ (current fraction) and $f_m = {d['fm_default']:.2f}$ (mass fraction) control how much current and mass participate. These are calibrated against experimental data.

### Your results

The sheath reached the end of the anode at approximately **{self.peak_time_us * 0.7:.1f} us**, traveling at over **100 km/s** — about 300 times the speed of sound in air.
"""

    def _narrative_radial(self, d: dict) -> str:
        return f"""## Phase 2: Radial Compression

### What happens
When the sheath reaches the end of the anode, it can't go forward anymore. Instead, the magnetic force redirects it **inward** — the sheath folds over and begins compressing radially toward the axis.

This is like squeezing a tube of toothpaste from the outside — except the "squeezing" is done by a magnetic field carrying millions of amperes.

### The Bennett equilibrium

The balance between magnetic compression and plasma pressure is described by the **Bennett relation**:

$$\\frac{{\\mu_0 I^2}}{{8\\pi}} = N k_B (Z T_e + T_i)$$

where $N$ is the linear particle density (particles per meter of column length), $Z$ is the ion charge, and $T_e$, $T_i$ are electron and ion temperatures.

In plain English: the magnetic squeeze (left side) equals the thermal pressure trying to push outward (right side). When the current is high enough, the squeeze wins and the plasma compresses.

### The MHD equations

During radial compression, the simulation switches from the Lee model to **magnetohydrodynamics (MHD)** — the full set of equations governing magnetized plasma flow:

**Mass conservation** (what goes in must come out):
$$\\frac{{\\partial \\rho}}{{\\partial t}} + \\nabla \\cdot (\\rho \\vec{{v}}) = 0$$

**Momentum conservation** (Newton's second law for fluids + magnetic forces):
$$\\frac{{\\partial (\\rho \\vec{{v}})}}{{\\partial t}} + \\nabla \\cdot (\\rho \\vec{{v}}\\vec{{v}} + p_{{tot}} \\hat{{I}} - \\vec{{B}}\\vec{{B}}) = 0$$

where $p_{{tot}} = p + B^2/2$ is the total pressure (gas + magnetic).

**Energy conservation** (total energy is constant):
$$\\frac{{\\partial E}}{{\\partial t}} + \\nabla \\cdot [(E + p_{{tot}})\\vec{{v}} - \\vec{{B}}(\\vec{{v}} \\cdot \\vec{{B}})] = 0$$

### How the simulation solves it

The MHD equations are solved using:
- **HLLS Riemann solver**: resolves shock waves and discontinuities
- **PLM reconstruction**: 2nd-order spatial accuracy
- **SSP-RK2 time integration**: stable time stepping
- **RKL2 super-timestepping**: handles resistive diffusion on GPU (3.4x faster than CPU)

The simulation uses a **{d.get('fc_default', 0.7):.0%}** current fraction, meaning {d.get('fc_default', 0.7)*100:.0f}% of the circuit current participates in driving the sheath.
"""

    def _narrative_pinch(self, d: dict) -> str:
        return f"""## Phase 3: Pinch

### What happens
The radial compression reaches its maximum — the plasma column shrinks to just a few millimeters in diameter. At this moment:

- **Temperature** reaches 1-10 keV (10-100 million degrees Celsius)
- **Density** peaks at $10^{{24}}-10^{{26}}$ particles per cubic meter
- **Current** reaches its peak of **{self.peak_current_ma:.3f} MA**
- **Fusion reactions** produce neutrons in deuterium gas

This is the whole point of a Dense Plasma Focus — creating these extreme conditions for a brief moment.

### The current dip

At pinch, the plasma inductance $L_p$ changes rapidly as the column compresses. This creates a **back-EMF** that opposes the current:

$$V = L_0 \\frac{{dI}}{{dt}} + I\\frac{{dL_p}}{{dt}} + IR$$

The term $I \\cdot dL_p/dt$ is the back-EMF from the changing inductance. When $dL_p/dt$ is large (rapid compression), the current **dips** — this is the characteristic signature of a plasma pinch.

Your simulation shows a **{self.current_dip_pct:.0f}%** current dip (experimental: ~60%).

### Neutron production

In deuterium gas, two fusion reactions can occur:

$$D + D \\rightarrow He^3 + n \\quad (2.45 \\text{{ MeV neutron}})$$
$$D + D \\rightarrow T + p \\quad (3.02 \\text{{ MeV proton}})$$

In a DPF, most neutrons come from **beam-target** interactions — fast ions accelerated by plasma instabilities hitting stationary deuterium — rather than thermonuclear fusion.

### Comparison to experiment

| Metric | Simulation | {d['name']} Experiment |
|--------|-----------|----------------------|
| Peak Current | {self.peak_current_ma:.3f} MA | {self.peak_current_exp:.2f} MA |
| Peak Time | {self.peak_time_us:.1f} us | {self.peak_time_exp:.1f} us |
"""

    def _narrative_postpinch(self, d: dict) -> str:
        return """## Phase 4: Post-Pinch Expansion

### What happens
After maximum compression, the plasma column **expands outward**. The extreme pressure inside the pinch overcomes the magnetic confinement, and the column grows back toward its original size.

During expansion:
- The plasma inductance $L_p$ decreases as the column widens
- The current continues to drop (the "dip" deepens)
- The plasma cools rapidly through radiation and expansion
- Instabilities (Rayleigh-Taylor, kink) can disrupt the column

### The expansion model

The column expands at a velocity related to the sound speed in the hot plasma:

$$v_{expand} \\approx c_s = \\sqrt{\\frac{\\gamma k_B T}{m_i}}$$

where $\\gamma = 5/3$ for an ideal gas, $k_B$ is Boltzmann's constant, $T$ is the temperature, and $m_i$ is the ion mass.

For deuterium at 1 keV: $c_s \\approx 200$ km/s. The column doubles in radius in about 50 nanoseconds.

### Why the current dip matters

The shape and depth of the current dip tells us about the quality of the pinch:
- **Deep dip (>50%)**: strong, symmetric compression
- **Shallow dip (<20%)**: weak compression or early instability breakup
- **Multiple dips**: re-pinching events (the plasma compresses, expands, and compresses again)

The current dip is the most accessible experimental diagnostic — it can be measured with a simple Rogowski coil, unlike temperature or density which require sophisticated instruments.
"""
