"""DPF-Unified v2 — Reflex frontend for Dense Plasma Focus simulation."""

import reflex as rx

from .components import (
    babylon_renderer,
    export_card,
    live_metrics_bar,
    metric_card,
    navbar,
    narrative_section,
    phase_chip,
    progress_phase_bar,
)
from .presets import BACKEND_LEVELS, GAS_SPECIES, PRESETS
from .state import SimState
from .styles import (
    FONT_DISPLAY,
    FONT_MONO,
    GHOST_BORDER,
    PHASE_PINCH,
    PHASE_POST,
    PHASE_RADIAL,
    PHASE_RUNDOWN,
    PRIMARY,
    PRIMARY_CONTAINER,
    SECONDARY,
    SUCCESS,
    SURFACE,
    SURFACE_HIGH,
    SURFACE_HIGHEST,
    SURFACE_LOW,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    button_ghost,
    button_primary,
    card_style,
    label_style,
    unit_style,
    value_style,
)


# ─── Screen 1: Configure ────────────────────────────────────────────


def config_sidebar() -> rx.Component:
    return rx.vstack(
        rx.text("HARDWARE CONFIGURATION", style=label_style),
        rx.heading("Simulation Deck", font_family=FONT_DISPLAY, size="5",
                    font_weight="300", color=TEXT_PRIMARY),
        rx.separator(color=GHOST_BORDER, margin_y="0.75rem"),

        rx.text("DEVICE PRESET", style=label_style),
        rx.select(
            list(PRESETS.keys()),
            value=SimState.preset,
            on_change=SimState.set_preset,
            width="100%",
        ),
        rx.text(SimState.preset_data["class"], font_size="0.75rem", color=TEXT_MUTED),

        rx.text("FILL GAS", style=label_style),
        rx.select(
            list(GAS_SPECIES.keys()),
            value=SimState.gas_species,
            on_change=SimState.set_gas_species,
            width="100%",
        ),
        rx.cond(
            SimState.student_mode,
            rx.text(SimState.gas_description,
                     font_size="0.6875rem", color=TEXT_MUTED, line_height="1.4"),
        ),

        rx.text("BACKEND PRECISION", style=label_style),
        rx.slider(min=1, max=5, step=1, default_value=3,
                  on_change=SimState.set_backend_level, width="100%"),
        rx.text(SimState.backend_info["name"], font_size="0.875rem",
                color=PRIMARY, font_family=FONT_MONO, font_weight="500"),
        rx.text(SimState.backend_info["desc"], font_size="0.75rem",
                color=TEXT_SECONDARY, line_height="1.4"),
        rx.hstack(
            rx.text("Grid: ", font_size="0.6875rem", color=TEXT_MUTED),
            rx.text(SimState.backend_info["grid"], font_size="0.6875rem",
                     color=TEXT_SECONDARY, font_family=FONT_MONO),
            rx.text(" | Est: ", font_size="0.6875rem", color=TEXT_MUTED),
            rx.text(SimState.backend_info["time"], font_size="0.6875rem",
                     color=TEXT_SECONDARY, font_family=FONT_MONO),
        ),

        rx.text("GRID RESOLUTION", style=label_style),
        rx.select(["coarse", "medium", "fine"],
                  value=SimState.grid_resolution,
                  on_change=SimState.set_grid_resolution, width="100%"),

        rx.separator(color=GHOST_BORDER, margin_y="0.75rem"),

        # Advanced
        rx.hstack(
            rx.text("ADVANCED PARAMETERS", style=label_style),
            rx.cond(SimState.show_advanced,
                     rx.icon("chevron-up", size=14, color=TEXT_MUTED),
                     rx.icon("chevron-down", size=14, color=TEXT_MUTED)),
            cursor="pointer", on_click=SimState.toggle_advanced,
            width="100%", justify="between",
        ),
        rx.cond(
            SimState.show_advanced,
            rx.vstack(
                rx.hstack(
                    rx.text("fc", font_family=FONT_MONO, font_size="0.75rem",
                             color=TEXT_SECONDARY, width="2rem"),
                    rx.slider(min=0.3, max=0.95, step=0.01, default_value=0.7,
                              on_change=SimState.set_fc, flex="1"),
                    rx.text(SimState.fc.to(str), font_family=FONT_MONO,
                             font_size="0.75rem", color=PRIMARY, width="3rem"),
                    width="100%", align="center",
                ),
                rx.hstack(
                    rx.text("fm", font_family=FONT_MONO, font_size="0.75rem",
                             color=TEXT_SECONDARY, width="2rem"),
                    rx.slider(min=0.01, max=0.3, step=0.005, default_value=0.08,
                              on_change=SimState.set_fm, flex="1"),
                    rx.text(SimState.fm.to(str), font_family=FONT_MONO,
                             font_size="0.75rem", color=PRIMARY, width="3rem"),
                    width="100%", align="center",
                ),
                spacing="3", margin_top="0.5rem",
            ),
        ),

        rx.separator(color=GHOST_BORDER, margin_y="0.75rem"),

        # Circuit parameters (expert mode)
        rx.cond(
            ~SimState.student_mode,
            rx.vstack(
                rx.text("CIRCUIT PARAMETERS", style=label_style),
                rx.hstack(
                    rx.text("V₀", font_family=FONT_MONO, font_size="0.75rem",
                             color=TEXT_SECONDARY, width="2.5rem"),
                    rx.slider(min=5, max=80, step=1, default_value=27,
                              on_change=SimState.set_voltage, flex="1"),
                    rx.text(SimState.voltage.to(str), font_family=FONT_MONO,
                             font_size="0.75rem", color=PRIMARY, width="3rem"),
                    rx.text("kV", style={"font_size": "0.6875rem", "color": TEXT_MUTED}),
                    width="100%", align="center",
                ),
                rx.hstack(
                    rx.text("P₀", font_family=FONT_MONO, font_size="0.75rem",
                             color=TEXT_SECONDARY, width="2.5rem"),
                    rx.slider(min=0.5, max=20, step=0.5, default_value=3.5,
                              on_change=SimState.set_fill_pressure, flex="1"),
                    rx.text(SimState.fill_pressure.to(str), font_family=FONT_MONO,
                             font_size="0.75rem", color=PRIMARY, width="3rem"),
                    rx.text("Torr", style={"font_size": "0.6875rem", "color": TEXT_MUTED}),
                    width="100%", align="center",
                ),
                rx.hstack(
                    rx.text("t_sim", font_family=FONT_MONO, font_size="0.75rem",
                             color=TEXT_SECONDARY, width="2.5rem"),
                    rx.slider(min=5, max=30, step=1, default_value=10,
                              on_change=SimState.set_sim_time, flex="1"),
                    rx.text(SimState.sim_time_us.to(str), font_family=FONT_MONO,
                             font_size="0.75rem", color=PRIMARY, width="3rem"),
                    rx.text("us", style={"font_size": "0.6875rem", "color": TEXT_MUTED}),
                    width="100%", align="center",
                ),
                rx.hstack(
                    rx.text("C: ", font_size="0.6875rem", color=TEXT_MUTED),
                    rx.text(SimState.capacitance_mf.to(str), font_size="0.6875rem",
                             color=TEXT_SECONDARY, font_family=FONT_MONO),
                    rx.text(" mF | L₀: ", font_size="0.6875rem", color=TEXT_MUTED),
                    rx.text(SimState.inductance_nh.to(str), font_size="0.6875rem",
                             color=TEXT_SECONDARY, font_family=FONT_MONO),
                    rx.text(" nH | R₀: ", font_size="0.6875rem", color=TEXT_MUTED),
                    rx.text(SimState.resistance_mohm.to(str), font_size="0.6875rem",
                             color=TEXT_SECONDARY, font_family=FONT_MONO),
                    rx.text(" mΩ", font_size="0.6875rem", color=TEXT_MUTED),
                    flex_wrap="wrap",
                ),
                spacing="3",
            ),
        ),

        rx.text("PHYSICS TOGGLES", style=label_style),
        rx.vstack(
            rx.hstack(rx.switch(checked=SimState.enable_radiation,
                                 on_change=SimState.set_enable_radiation, size="1"),
                       rx.tooltip(rx.text("Radiation", font_size="0.8125rem"),
                                   content="X-ray energy loss — important for high-current devices"),
                       spacing="2"),
            rx.hstack(rx.switch(checked=SimState.enable_conduction,
                                 on_change=SimState.set_enable_conduction, size="1"),
                       rx.tooltip(rx.text("Conduction", font_size="0.8125rem"),
                                   content="Heat flow along magnetic field lines"),
                       spacing="2"),
            rx.hstack(rx.switch(checked=SimState.enable_anomalous,
                                 on_change=SimState.set_enable_anomalous, size="1"),
                       rx.tooltip(rx.text("Anomalous Resistivity", font_size="0.8125rem"),
                                   content="Turbulent plasma resistance at pinch — improves post-pinch accuracy"),
                       spacing="2"),
            rx.hstack(rx.switch(checked=SimState.enable_hall,
                                 on_change=SimState.set_enable_hall, size="1"),
                       rx.tooltip(rx.text("Hall MHD", font_size="0.8125rem"),
                                   content="Magnetic field decoupling from plasma — relevant near pinch"),
                       spacing="2"),
            # Expert-only advanced physics
            rx.cond(
                ~SimState.student_mode,
                rx.vstack(
                    rx.separator(color=GHOST_BORDER, margin_y="0.25rem"),
                    rx.text("ADVANCED PHYSICS", style=label_style),
                    rx.hstack(rx.switch(checked=SimState.enable_sheath_bc,
                                         on_change=SimState.set_enable_sheath_bc, size="1"),
                               rx.tooltip(rx.text("Sheath BC", font_size="0.8125rem"),
                                           content="Bohm criterion boundary condition at sheath edge"),
                               spacing="2"),
                    rx.hstack(rx.switch(checked=SimState.enable_ablation,
                                         on_change=SimState.set_enable_ablation, size="1"),
                               rx.tooltip(rx.text("Electrode Ablation", font_size="0.8125rem"),
                                           content="Copper erosion from electrodes — adds impurity species"),
                               spacing="2"),
                    rx.hstack(rx.switch(checked=SimState.enable_nernst,
                                         on_change=SimState.set_enable_nernst, size="1"),
                               rx.tooltip(rx.text("Nernst Effect", font_size="0.8125rem"),
                                           content="B-field advection by temperature gradients"),
                               spacing="2"),
                    rx.hstack(rx.switch(checked=SimState.enable_cr_ionization,
                                         on_change=SimState.set_enable_cr_ionization, size="1"),
                               rx.tooltip(rx.text("CR Ionization", font_size="0.8125rem"),
                                           content="Non-LTE collisional-radiative ionization model"),
                               spacing="2"),
                    rx.hstack(rx.switch(checked=SimState.enable_crowbar,
                                         on_change=SimState.set_enable_crowbar, size="1"),
                               rx.tooltip(rx.text("Crowbar Switch", font_size="0.8125rem"),
                                           content="Short-circuits capacitor at current peak to prevent ringing"),
                               spacing="2"),
                    spacing="2",
                ),
            ),
            spacing="2",
        ),

        spacing="3", width="100%", padding="1.25rem",
        style={"background": SURFACE_LOW, "border_right": f"1px solid {GHOST_BORDER}",
               "min_height": "100vh", "overflow_y": "auto"},
    )


def device_preview() -> rx.Component:
    return rx.vstack(
        rx.text("SELECTED INSTRUMENT", style=label_style),
        rx.heading(SimState.preset_data["name"],
                    font_family=FONT_DISPLAY, size="8", font_weight="300"),

        rx.box(
            rx.center(
                rx.vstack(
                    rx.icon("zap", size=64, color=TEXT_MUTED, stroke_width=0.5),
                    rx.text("Device Schematic", color=TEXT_MUTED, font_size="0.875rem"),
                    align="center", spacing="2",
                ),
            ),
            style={"background": SURFACE, "border_radius": "4px", "height": "220px",
                    "border": f"1px solid {GHOST_BORDER}"},
            width="100%",
        ),

        rx.grid(
            rx.box(rx.text("CAPACITANCE", style=label_style),
                    rx.hstack(rx.text(SimState.preset_data["capacitance"], style=value_style),
                              rx.text("mF", style=unit_style), align="end")),
            rx.box(rx.text("VOLTAGE", style=label_style),
                    rx.hstack(rx.text(SimState.voltage.to(str), style=value_style),
                              rx.text("kV", style=unit_style), align="end")),
            rx.box(rx.text("STORED ENERGY", style=label_style),
                    rx.text(SimState.preset_data["energy"], style=value_style)),
            rx.box(rx.text("ANODE LENGTH", style=label_style),
                    rx.hstack(rx.text(SimState.preset_data["anode_length"], style=value_style),
                              rx.text("mm", style=unit_style), align="end")),
            columns="4", spacing="4", width="100%",
        ),

        rx.separator(color=GHOST_BORDER, margin_y="1rem"),

        rx.box(
            rx.text("SCIENTIFIC CONTEXT", style=label_style),
            rx.text(SimState.preset_data["description"],
                     font_size="0.875rem", color=TEXT_SECONDARY, line_height="1.6",
                     margin_top="0.5rem"),
            style=card_style, width="100%",
        ),

        rx.button(
            rx.hstack(
                rx.text("RUN SIMULATION", font_family=FONT_DISPLAY,
                         letter_spacing="0.05em"),
                rx.icon("arrow-right", size=18),
                spacing="2", align="center",
            ),
            on_click=SimState.run_simulation,
            style=button_primary, width="100%", size="3",
            loading=SimState.running, margin_top="1.5rem",
        ),
        rx.hstack(
            rx.text("Estimated time: ", font_size="0.8125rem", color=TEXT_MUTED),
            rx.text(SimState.backend_info["time"], font_size="0.8125rem",
                     color=PRIMARY, font_family=FONT_MONO),
            rx.text(" | Backend: ", font_size="0.8125rem", color=TEXT_MUTED),
            rx.text(SimState.backend_info["name"], font_size="0.8125rem",
                     color=PRIMARY, font_family=FONT_MONO),
        ),

        spacing="4", padding="2rem", flex="1",
    )


def configure_page() -> rx.Component:
    return rx.box(
        rx.cond(
            SimState.student_mode,
            rx.box(
                rx.hstack(
                    rx.icon("lightbulb", size=16, color=SECONDARY),
                    rx.text(
                        "Simulate a Dense Plasma Focus in 3 steps: ",
                        rx.text.span("1. Pick a device  ", font_weight="600"),
                        rx.text.span("2. Click Run  ", font_weight="600"),
                        rx.text.span("3. Watch the physics", font_weight="600"),
                        font_size="0.8125rem", color=TEXT_SECONDARY,
                    ),
                    spacing="2", align="center",
                ),
                style={"background": SURFACE_LOW, "padding": "0.75rem 1.5rem",
                        "border_bottom": f"1px solid {GHOST_BORDER}"},
                width="100%",
            ),
        ),
        rx.hstack(
            rx.box(config_sidebar(), width="320px", flex_shrink="0"),
            rx.box(device_preview(), flex="1"),
            spacing="0", width="100%", align="start",
        ),
    )


# ─── Screen 2: Simulation Running ───────────────────────────────────


def running_page() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.badge(SimState.preset, color_scheme="blue"),
            rx.text("@", color=TEXT_MUTED),
            rx.text(SimState.voltage.to(str), font_family=FONT_MONO, color=PRIMARY),
            rx.text("kV", color=TEXT_MUTED),
            rx.spacer(),
            rx.badge("ACTIVE RUN", color_scheme="green", variant="solid"),
            rx.button("Cancel", variant="outline", color_scheme="red", size="1",
                       on_click=SimState.reset_to_configure),
            spacing="2", align="center", width="100%",
            padding="0.75rem 1.5rem", background=SURFACE_LOW,
            border_bottom=f"1px solid {GHOST_BORDER}",
        ),
        rx.heading(
            rx.text.span("Phase: ", color=TEXT_SECONDARY),
            rx.text.span(SimState.phase, color=PRIMARY),
            font_family=FONT_DISPLAY, size="6", font_weight="400",
            padding="1rem 1.5rem",
        ),
        progress_phase_bar(),
        live_metrics_bar(),
        # 3D preview during simulation — full width
        babylon_renderer(height="50vh"),
        # Narrative overlay below renderer
        rx.box(
            rx.center(
                rx.box(
                    rx.text(
                        SimState.narrative_live,
                        font_size="1rem", color=TEXT_PRIMARY, font_style="italic",
                        text_align="center", line_height="1.6",
                    ),
                    style={"background": "rgba(15, 19, 31, 0.85)",
                            "backdrop_filter": "blur(8px)",
                            "padding": "1rem 2rem", "border_radius": "4px",
                            "max_width": "700px", "border": f"1px solid {GHOST_BORDER}"},
                ),
            ),
            width="100%", padding="1rem 1.5rem",
        ),
        spacing="0", width="100%",
    )


# ─── Screen 3: Results Hero ─────────────────────────────────────────


def results_metrics() -> rx.Component:
    return rx.hstack(
        metric_card("Peak Current", "I_peak",
                     SimState.peak_current_ma.to(str), "MA",
                     f"Exp: {SimState.peak_current_exp} MA",
                     SimState.peak_error_pct.to(str) + "%",
                     SimState.peak_error_color),
        metric_card("Peak Time", "t_peak",
                     SimState.peak_time_us.to(str), "us",
                     f"Exp: {SimState.peak_time_exp} us",
                     SimState.timing_error_pct.to(str) + "%",
                     SimState.timing_error_color),
        metric_card("Current Dip", "I_dip/I_peak",
                     SimState.current_dip_pct.to(str), "%"),
        metric_card("Neutron Yield", "Y_n",
                     SimState.neutron_yield, ""),
        spacing="4", padding="1.5rem", width="100%",
        overflow_x="auto",
    )


def results_page() -> rx.Component:
    return rx.vstack(
        results_metrics(),
        babylon_renderer(height="55vh"),
        narrative_section(),
        # Analysis tabs below narrative
        rx.box(
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("Waveforms", value="waveforms"),
                    rx.tabs.trigger("Energy & Physics", value="energy"),
                    rx.tabs.trigger("2D Fields", value="fields"),
                    rx.tabs.trigger("Compare", value="compare"),
                ),
                rx.tabs.content(
                    waveform_tab(),
                    value="waveforms",
                ),
                rx.tabs.content(
                    energy_tab(),
                    value="energy",
                ),
                rx.tabs.content(
                    fields_tab(),
                    value="fields",
                ),
                rx.tabs.content(
                    rx.vstack(
                        rx.text("COMPARE RUNS", style=label_style),
                        rx.text("Run multiple simulations with different parameters to compare waveforms, "
                                 "energy balance, and pinch characteristics side by side.",
                                 font_size="0.875rem", color=TEXT_SECONDARY, line_height="1.6"),
                        rx.center(
                            rx.vstack(
                                rx.icon("git-compare", size=36, color=TEXT_MUTED),
                                rx.text("Run at least 2 simulations to enable comparison",
                                         color=TEXT_MUTED, font_size="0.875rem"),
                                align="center", spacing="2",
                            ),
                            padding="3rem",
                        ),
                        spacing="3", padding="1rem",
                    ),
                    value="compare",
                ),
                default_value="waveforms",
            ),
            padding="1.5rem", width="100%",
        ),
        spacing="0", width="100%",
    )


def energy_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            SimState.has_results,
            rx.vstack(
                rx.text("ENERGY PARTITION", style=label_style),
                rx.recharts.responsive_container(
                    rx.recharts.area_chart(
                        rx.recharts.area(
                            data_key="Magnetic", stroke="#adc6ff", fill="#adc6ff",
                            fill_opacity=0.3, stack_id="1", name="Magnetic Energy (%)",
                        ),
                        rx.recharts.area(
                            data_key="Kinetic", stroke="#ffb95f", fill="#ffb95f",
                            fill_opacity=0.3, stack_id="1", name="Kinetic Energy (%)",
                        ),
                        rx.recharts.area(
                            data_key="Thermal", stroke="#ff6b6b", fill="#ff6b6b",
                            fill_opacity=0.3, stack_id="1", name="Thermal Energy (%)",
                        ),
                        rx.recharts.x_axis(
                            data_key="time", stroke="#424754",
                            tick={"fill": "#9ca3af", "fontSize": 11},
                        ),
                        rx.recharts.y_axis(
                            stroke="#424754",
                            tick={"fill": "#9ca3af", "fontSize": 11},
                        ),
                        rx.recharts.cartesian_grid(stroke_dasharray="3 3", stroke="#262a37"),
                        rx.recharts.tooltip(
                            content_style={"backgroundColor": "#171b28",
                                            "border": "1px solid #424754",
                                            "borderRadius": "4px", "color": "#dfe2f3"},
                        ),
                        rx.recharts.legend(),
                        data=SimState.energy_chart_data,
                    ),
                    width="100%", height=350,
                ),
                rx.cond(
                    SimState.student_mode,
                    rx.box(
                        rx.text("WHAT THIS SHOWS", style=label_style),
                        rx.markdown(
                            """The energy in a DPF transforms between three forms:

- **Magnetic energy** (blue): Stored in the magnetic field created by the current. This is the *driver* — it does the work of compressing the plasma.
- **Kinetic energy** (amber): The energy of plasma motion. High during rundown (sheath moving fast) and radial compression.
- **Thermal energy** (red): Heat. Peaks at pinch when kinetic and magnetic energy convert to extreme temperatures.

The total energy comes from the capacitor bank. In an ideal system, magnetic + kinetic + thermal = 100% at all times (conservation of energy). In reality, some energy is lost to radiation (X-rays, bremsstrahlung) and resistive heating of the electrodes.""",
                            style={"font_size": "0.875rem", "color": TEXT_SECONDARY, "line_height": "1.6"},
                        ),
                        style=card_style, width="100%",
                    ),
                ),
                spacing="4", padding="1rem",
            ),
            rx.center(
                rx.text("Run a simulation to see energy partition.", color=TEXT_MUTED),
                padding="3rem",
            ),
        ),
    )


def fields_tab() -> rx.Component:
    return rx.vstack(
        rx.text("2D FIELD DIAGNOSTICS", style=label_style),
        rx.cond(
            SimState.has_results,
            rx.vstack(
                rx.grid(
                    rx.box(
                        rx.text("DENSITY (rho)", style=label_style),
                        rx.box(
                            rx.center(
                                rx.vstack(
                                    rx.icon("layers", size=32, color=TEXT_MUTED),
                                    rx.text("r-z density heatmap", color=TEXT_MUTED, font_size="0.8125rem"),
                                    rx.text("Shows where the plasma is concentrated", color=TEXT_MUTED, font_size="0.75rem"),
                                    align="center",
                                ),
                            ),
                            style={"background": SURFACE, "height": "200px", "border_radius": "4px",
                                    "border": f"1px solid {GHOST_BORDER}"},
                        ),
                        style=card_style,
                    ),
                    rx.box(
                        rx.text("TEMPERATURE (T)", style=label_style),
                        rx.box(
                            rx.center(
                                rx.vstack(
                                    rx.icon("thermometer", size=32, color=TEXT_MUTED),
                                    rx.text("r-z temperature heatmap", color=TEXT_MUTED, font_size="0.8125rem"),
                                    rx.text("Hottest at the pinch axis", color=TEXT_MUTED, font_size="0.75rem"),
                                    align="center",
                                ),
                            ),
                            style={"background": SURFACE, "height": "200px", "border_radius": "4px",
                                    "border": f"1px solid {GHOST_BORDER}"},
                        ),
                        style=card_style,
                    ),
                    rx.box(
                        rx.text("MAGNETIC FIELD (B_theta)", style=label_style),
                        rx.box(
                            rx.center(
                                rx.vstack(
                                    rx.icon("magnet", size=32, color=TEXT_MUTED),
                                    rx.text("r-z B-field heatmap", color=TEXT_MUTED, font_size="0.8125rem"),
                                    rx.text("Strongest at the sheath boundary", color=TEXT_MUTED, font_size="0.75rem"),
                                    align="center",
                                ),
                            ),
                            style={"background": SURFACE, "height": "200px", "border_radius": "4px",
                                    "border": f"1px solid {GHOST_BORDER}"},
                        ),
                        style=card_style,
                    ),
                    rx.box(
                        rx.text("CURRENT DENSITY (J)", style=label_style),
                        rx.box(
                            rx.center(
                                rx.vstack(
                                    rx.icon("zap", size=32, color=TEXT_MUTED),
                                    rx.text("r-z current density", color=TEXT_MUTED, font_size="0.8125rem"),
                                    rx.text("Concentrated in the sheath layer", color=TEXT_MUTED, font_size="0.75rem"),
                                    align="center",
                                ),
                            ),
                            style={"background": SURFACE, "height": "200px", "border_radius": "4px",
                                    "border": f"1px solid {GHOST_BORDER}"},
                        ),
                        style=card_style,
                    ),
                    columns="2", spacing="4", width="100%",
                ),
                rx.cond(
                    SimState.student_mode,
                    rx.box(
                        rx.text("UNDERSTANDING 2D FIELDS", style=label_style),
                        rx.markdown(
                            """These heatmaps show the plasma state on a 2D slice through the device. The horizontal axis is **z** (along the anode, left to right) and the vertical axis is **r** (radial distance from the center axis, bottom to top).

- **Density**: Bright = more plasma. The sheath appears as a bright band sweeping from left to right (rundown), then a bright spot at the tip (pinch).
- **Temperature**: Bright = hotter. The pinch region reaches millions of degrees — visible as an intense spot at the anode tip.
- **B-field**: The azimuthal (toroidal) magnetic field $B_\\theta$ wraps around the axis. It's strongest just outside the current sheath — this is the field doing the compression.
- **Current density**: Shows where the current flows. During rundown it's spread across the sheath; at pinch it concentrates into a tiny column.

These fields come directly from the MHD solver — they are the fundamental variables being evolved at each timestep.""",
                            style={"font_size": "0.875rem", "color": TEXT_SECONDARY, "line_height": "1.6"},
                        ),
                        style=card_style, width="100%",
                    ),
                ),
                spacing="4", padding="1rem",
            ),
            rx.center(
                rx.text("Run a simulation to see 2D field data.", color=TEXT_MUTED),
                padding="3rem",
            ),
        ),
    )


def waveform_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            SimState.has_results,
            rx.box(
                rx.text("CURRENT WAVEFORM I(t)", style=label_style, margin_bottom="0.5rem"),
                rx.recharts.responsive_container(
                    rx.recharts.line_chart(
                        rx.recharts.line(
                            data_key="current",
                            stroke="#adc6ff",
                            stroke_width=2,
                            dot=False,
                            name="Current (MA)",
                        ),
                        rx.recharts.x_axis(
                            data_key="time",
                            stroke="#424754",
                            tick={"fill": "#9ca3af", "fontSize": 11},
                            label={"value": "Time (us)", "fill": "#9ca3af",
                                    "position": "insideBottom", "offset": -5},
                        ),
                        rx.recharts.y_axis(
                            stroke="#424754",
                            tick={"fill": "#9ca3af", "fontSize": 11},
                            label={"value": "Current (MA)", "fill": "#9ca3af",
                                    "angle": -90, "position": "insideLeft"},
                        ),
                        rx.recharts.cartesian_grid(stroke_dasharray="3 3", stroke="#262a37"),
                        rx.recharts.tooltip(
                            content_style={"backgroundColor": "#171b28",
                                            "border": "1px solid #424754",
                                            "borderRadius": "4px",
                                            "color": "#dfe2f3"},
                        ),
                        rx.recharts.legend(),
                        # Phase reference lines
                        rx.recharts.reference_line(
                            x="4.1", stroke="#adc6ff", stroke_dasharray="5 5",
                            label="Radial",
                        ),
                        rx.recharts.reference_line(
                            x="5.2", stroke="#ffb95f", stroke_dasharray="5 5",
                            label="Pinch",
                        ),
                        data=SimState.waveform_chart_data,
                    ),
                    width="100%",
                    height=400,
                ),
                style={"background": SURFACE, "border_radius": "4px",
                        "padding": "1rem", "border": f"1px solid {GHOST_BORDER}"},
                width="100%",
            ),
            rx.box(
                rx.center(
                    rx.vstack(
                        rx.icon("line-chart", size=48, color=TEXT_MUTED, stroke_width=0.5),
                        rx.text("Current Waveform I(t)", color=TEXT_MUTED,
                                 font_family=FONT_DISPLAY),
                        rx.text("Run a simulation or click Demo Data to see charts",
                                 font_size="0.8125rem", color=TEXT_MUTED),
                        align="center", spacing="2",
                    ),
                ),
                style={"background": SURFACE, "border_radius": "4px",
                        "height": "400px", "border": f"1px solid {GHOST_BORDER}"},
                width="100%",
            ),
        ),
        rx.cond(
            SimState.student_mode,
            rx.box(
                rx.text("PHYSICS INSIGHT", style=label_style),
                rx.text(
                    "The current rises as the capacitor discharges, peaks when the sheath "
                    "reaches the end of the anode, then dips sharply during the pinch — "
                    "this is the magnetic compression squeezing the plasma to its smallest size.",
                    font_size="0.875rem", color=TEXT_SECONDARY, line_height="1.6",
                    margin_top="0.5rem",
                ),
                style=card_style, width="100%",
            ),
        ),
        rx.hstack(
            rx.button(rx.icon("download", size=14), "Export Charts as SVG",
                       style=button_ghost, size="1"),
            rx.button(rx.icon("download", size=14), "Export Data as CSV",
                       style=button_ghost, size="1"),
            spacing="2",
        ),
        spacing="4", padding="1rem",
    )


# ─── Screen 5: Export ────────────────────────────────────────────────


def export_page() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.link(rx.icon("arrow-left", size=14), "Back to Results",
                     href="/", color=TEXT_SECONDARY, font_size="0.8125rem"),
            spacing="2", padding="1rem 1.5rem",
        ),
        rx.heading("Export Center", font_family=FONT_DISPLAY, size="6",
                    font_weight="400", padding_x="1.5rem"),
        rx.text("Finalize simulation assets for publication, archival, or review.",
                 font_size="0.875rem", color=TEXT_SECONDARY, padding_x="1.5rem"),

        # Download All
        rx.box(
            rx.hstack(
                rx.icon("package", size=20, color=SUCCESS),
                rx.vstack(
                    rx.text("DOWNLOAD ALL", font_weight="600", color=SUCCESS),
                    rx.text("Get everything as a ZIP (CSV + SVG + PDF + JSON + Screenshot)",
                             font_size="0.8125rem", color=TEXT_SECONDARY),
                    spacing="1",
                ),
                rx.spacer(),
                rx.button("Generate Archive", style=button_primary, size="2"),
                spacing="3", align="center", width="100%",
            ),
            style=card_style, margin="1.5rem",
        ),

        # Export cards grid
        rx.grid(
            export_card("table", "Simulation Data",
                         "Full waveform I, V, Te, Lp, Yn in CSV tabular format.",
                         "Download CSV"),
            export_card("image", "Charts & Figures",
                         "Publication-ready vector graphics (SVG) of all plots.",
                         "Download SVGs"),
            export_card("box", "3D Visualization",
                         "Screenshot or animated GIF of the plasma pinch.",
                         "Capture Now"),
            export_card("file-text", "Physics Report",
                         "Full narrative with equations and results as PDF.",
                         "Download PDF"),
            export_card("settings", "Configuration",
                         "Reproducible simulation config as JSON for re-runs.",
                         "Download JSON"),
            export_card("share-2", "Share Link",
                         "Anyone with this link sees your exact setup and results.",
                         "Copy Link"),
            columns="3", spacing="4", padding="1.5rem", width="100%",
        ),
        spacing="2", width="100%",
    )


# ─── Router ──────────────────────────────────────────────────────────


def index() -> rx.Component:
    return rx.box(
        navbar(),
        rx.cond(
            SimState.current_view == "running",
            running_page(),
            rx.cond(
                SimState.current_view == "results",
                results_page(),
                rx.cond(
                    SimState.current_view == "export",
                    export_page(),
                    configure_page(),
                ),
            ),
        ),
        width="100%", min_height="100vh", background=SURFACE,
    )


app = rx.App(
    style={
        "font_family": "Inter, sans-serif",
        "background_color": SURFACE,
        "color": TEXT_PRIMARY,
    },
    theme=rx.theme(appearance="dark", accent_color="blue"),
    head_components=[
        rx.el.link(
            href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
            rel="stylesheet",
        ),
    ],
)
app.add_page(index, route="/", title="DPF-Unified | Plasma Focus Simulator")
