"""Reusable UI components for DPF-Unified v2."""

import reflex as rx

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
    glass_style,
    label_style,
    metric_card_style,
    nav_style,
    unit_style,
    value_style,
)


def navbar() -> rx.Component:
    return rx.hstack(
        rx.hstack(
            rx.text("DPF-Unified", font_family=FONT_DISPLAY, font_weight="600",
                     font_size="1.125rem", color=PRIMARY),
            rx.badge("v2", variant="outline", color_scheme="blue", size="1"),
            spacing="2", align="center",
        ),
        rx.spacer(),
        rx.hstack(
            rx.button("Configure", on_click=SimState.go_configure,
                       variant="ghost", size="1", color=TEXT_SECONDARY, cursor="pointer"),
            rx.button("Results", on_click=SimState.go_results,
                       variant="ghost", size="1", color=TEXT_SECONDARY, cursor="pointer"),
            rx.button("Export", on_click=SimState.go_export,
                       variant="ghost", size="1", color=TEXT_SECONDARY, cursor="pointer"),
            rx.text("|", color=GHOST_BORDER),
            rx.button("Demo Data", on_click=SimState.load_demo_results,
                       variant="ghost", size="1", color=SECONDARY, cursor="pointer"),
            spacing="3",
        ),
        rx.hstack(
            rx.button(
                rx.cond(SimState.student_mode, "Student Mode", "Expert Mode"),
                on_click=SimState.toggle_student_mode,
                size="1", variant="outline", color_scheme="blue",
            ),
            spacing="2",
        ),
        style=nav_style,
        width="100%",
        align="center",
    )


def phase_chip(label: str, color: str, is_active: bool = False) -> rx.Component:
    return rx.box(
        rx.text(label, font_size="0.6875rem", text_transform="uppercase",
                 letter_spacing="0.04em", font_weight="500",
                 color="#0a0e1a" if is_active else color),
        padding_x="0.75rem", padding_y="0.25rem",
        border_radius="2px",
        background=color if is_active else "transparent",
        border=f"1px solid {color}",
        cursor="pointer",
    )


def metric_card(
    label: str, symbol: str, value: rx.Var, unit: str,
    exp_label: str = "", error_label: str = "",
    error_color: rx.Var | str = TEXT_SECONDARY,
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(label, style=label_style),
            rx.text(f"({symbol})", font_size="0.6875rem", color=TEXT_MUTED,
                     font_family=FONT_MONO),
            rx.hstack(
                rx.text(value, style=value_style),
                rx.text(unit, style=unit_style),
                align="end", spacing="1",
            ),
            rx.cond(
                exp_label != "",
                rx.vstack(
                    rx.text(exp_label, font_size="0.75rem", color=TEXT_SECONDARY),
                    rx.text(error_label, font_size="0.75rem", color=error_color,
                             font_weight="600"),
                    spacing="1",
                ),
            ),
            spacing="2",
        ),
        style=metric_card_style,
    )


def progress_phase_bar() -> rx.Component:
    """Phase progress bar showing simulation phase with color-coded segments."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("SIMULATION PROGRESS", style=label_style),
                rx.spacer(),
                rx.text(
                    rx.text.span("ELAPSED TIME: ", color=TEXT_MUTED),
                    rx.text.span(
                        SimState.phase_time_us.to(str),
                        color=PRIMARY, font_family=FONT_MONO,
                    ),
                    rx.text.span(" us", color=TEXT_MUTED),
                    font_size="0.8125rem",
                ),
                width="100%",
            ),
            # Progress bar
            rx.box(
                rx.box(
                    width=rx.cond(
                        SimState.phase_progress > 0,
                        (SimState.phase_progress * 100).to(str) + "%",
                        "0%",
                    ),
                    height="100%",
                    background=f"linear-gradient(90deg, {PHASE_RUNDOWN}, {PHASE_RADIAL}, {PHASE_PINCH})",
                    border_radius="2px",
                    transition="width 0.3s ease",
                ),
                width="100%", height="6px",
                background=SURFACE, border_radius="2px",
                overflow="hidden",
            ),
            # Phase labels
            rx.hstack(
                rx.text("Rundown", font_size="0.6875rem", color=PHASE_RUNDOWN),
                rx.text("Radial", font_size="0.6875rem", color=PHASE_RADIAL),
                rx.text("Pinch", font_size="0.6875rem", color=PHASE_PINCH),
                rx.text("Post-pinch", font_size="0.6875rem", color=PHASE_POST),
                justify="between", width="100%",
            ),
            spacing="2", width="100%",
        ),
        padding="1rem 1.5rem",
        background=SURFACE_LOW,
        border_bottom=f"1px solid {GHOST_BORDER}",
        width="100%",
    )


def live_metrics_bar() -> rx.Component:
    """Real-time metrics during simulation."""
    return rx.hstack(
        rx.box(
            rx.text("SHEATH VELOCITY", style=label_style),
            rx.hstack(
                rx.text(SimState.live_velocity_kms.to(str), style=value_style, font_size="1.5rem"),
                rx.text("km/s", style=unit_style),
                align="end",
            ),
        ),
        rx.box(
            rx.text("ANODE CURRENT", style=label_style),
            rx.hstack(
                rx.text(SimState.live_current_ma.to(str), style=value_style, font_size="1.5rem"),
                rx.text("MA", style=unit_style),
                align="end",
            ),
        ),
        rx.box(
            rx.text("PLASMA TEMP", style=label_style),
            rx.hstack(
                rx.text(SimState.live_temperature_ev.to(str), style=value_style, font_size="1.5rem"),
                rx.text("eV", style=unit_style),
                align="end",
            ),
        ),
        rx.spacer(),
        rx.box(
            rx.text("ETA", style=label_style),
            rx.hstack(
                rx.text("~", color=TEXT_MUTED),
                rx.text(SimState.eta_seconds.to(str), style=value_style, font_size="1.5rem"),
                rx.text("sec", style=unit_style),
                align="end",
            ),
        ),
        spacing="6", padding="1rem 1.5rem", width="100%",
        background=SURFACE_LOW, border_bottom=f"1px solid {GHOST_BORDER}",
    )


def babylon_renderer(height: str = "55vh") -> rx.Component:
    """3D Babylon.js renderer — iframe to existing renderer or placeholder."""
    return rx.box(
        rx.box(
            rx.el.iframe(
                src="/babylon_renderer.html?v=" + SimState.renderer_key.to(str),
                width="100%",
                height="100%",
                style={"border": "none", "border_radius": "4px"},
            ),
            style={"background": SURFACE,
                    "border_radius": "4px",
                    "height": height, "border": f"1px solid {GHOST_BORDER}",
                    "overflow": "hidden"},
            width="100%",
        ),
        # Control bar
        rx.hstack(
            rx.button(rx.icon("play", size=14), "Play", style=button_ghost, size="1"),
            rx.button(rx.icon("skip-back", size=14), style=button_ghost, size="1"),
            rx.button(rx.icon("skip-forward", size=14), style=button_ghost, size="1"),
            rx.select(["1x", "2x", "4x", "8x"], default_value="1x", size="1", width="70px"),
            rx.spacer(),
            rx.button(rx.icon("camera", size=14), "Screenshot", style=button_ghost, size="1"),
            rx.button(rx.icon("maximize-2", size=14), "Fullscreen", style=button_ghost, size="1"),
            spacing="2", margin_top="0.5rem", width="100%",
        ),
        padding="1.5rem", width="100%",
    )


def narrative_section() -> rx.Component:
    """Scrollable narrative below the 3D renderer."""
    return rx.box(
        rx.hstack(
            rx.heading("What Just Happened", font_family=FONT_DISPLAY, size="5",
                        font_weight="400"),
            rx.spacer(),
            rx.hstack(
                rx.button(rx.icon("download", size=14), "PDF", style=button_ghost, size="1"),
                rx.button(rx.icon("copy", size=14), "Copy", style=button_ghost, size="1"),
                spacing="2",
            ),
            width="100%", align="center",
        ),
        rx.separator(color=GHOST_BORDER, margin_y="1rem"),
        # Interactive phase navigation pills
        rx.hstack(
            rx.box(
                rx.text("Overview", font_size="0.6875rem", text_transform="uppercase",
                         letter_spacing="0.04em", font_weight="500"),
                padding_x="0.75rem", padding_y="0.25rem", border_radius="2px",
                cursor="pointer", on_click=SimState.set_phase_tab("overview"),
                background=rx.cond(SimState.active_phase_tab == "overview", PRIMARY, "transparent"),
                color=rx.cond(SimState.active_phase_tab == "overview", "#0a0e1a", TEXT_MUTED),
                border=f"1px solid {PRIMARY}",
            ),
            rx.box(
                rx.text("Rundown", font_size="0.6875rem", text_transform="uppercase",
                         letter_spacing="0.04em", font_weight="500"),
                padding_x="0.75rem", padding_y="0.25rem", border_radius="2px",
                cursor="pointer", on_click=SimState.set_phase_tab("rundown"),
                background=rx.cond(SimState.active_phase_tab == "rundown", PHASE_RUNDOWN, "transparent"),
                color=rx.cond(SimState.active_phase_tab == "rundown", "#0a0e1a", PHASE_RUNDOWN),
                border=f"1px solid {PHASE_RUNDOWN}",
            ),
            rx.box(
                rx.text("Radial", font_size="0.6875rem", text_transform="uppercase",
                         letter_spacing="0.04em", font_weight="500"),
                padding_x="0.75rem", padding_y="0.25rem", border_radius="2px",
                cursor="pointer", on_click=SimState.set_phase_tab("radial"),
                background=rx.cond(SimState.active_phase_tab == "radial", PHASE_RADIAL, "transparent"),
                color=rx.cond(SimState.active_phase_tab == "radial", "#0a0e1a", PHASE_RADIAL),
                border=f"1px solid {PHASE_RADIAL}",
            ),
            rx.box(
                rx.text("Pinch", font_size="0.6875rem", text_transform="uppercase",
                         letter_spacing="0.04em", font_weight="500"),
                padding_x="0.75rem", padding_y="0.25rem", border_radius="2px",
                cursor="pointer", on_click=SimState.set_phase_tab("pinch"),
                background=rx.cond(SimState.active_phase_tab == "pinch", PHASE_PINCH, "transparent"),
                color=rx.cond(SimState.active_phase_tab == "pinch", "#0a0e1a", PHASE_PINCH),
                border=f"1px solid {PHASE_PINCH}",
            ),
            rx.box(
                rx.text("Post-pinch", font_size="0.6875rem", text_transform="uppercase",
                         letter_spacing="0.04em", font_weight="500"),
                padding_x="0.75rem", padding_y="0.25rem", border_radius="2px",
                cursor="pointer", on_click=SimState.set_phase_tab("postpinch"),
                background=rx.cond(SimState.active_phase_tab == "postpinch", PHASE_POST, "transparent"),
                color=rx.cond(SimState.active_phase_tab == "postpinch", "#0a0e1a", PHASE_POST),
                border=f"1px solid {PHASE_POST}",
            ),
            rx.box(
                rx.text("MHD Equations", font_size="0.6875rem", text_transform="uppercase",
                         letter_spacing="0.04em", font_weight="500"),
                padding_x="0.75rem", padding_y="0.25rem", border_radius="2px",
                cursor="pointer", on_click=SimState.set_phase_tab("equations"),
                background=rx.cond(SimState.active_phase_tab == "equations", "#10b981", "transparent"),
                color=rx.cond(SimState.active_phase_tab == "equations", "#0a0e1a", "#10b981"),
                border=f"1px solid #10b981",
            ),
            spacing="2", flex_wrap="wrap",
        ),
        # Narrative content — changes based on selected phase tab
        rx.cond(
            SimState.has_results,
            rx.markdown(
                SimState.narrative_for_phase,
                style={"margin_top": "1.5rem", "line_height": "1.8",
                        "font_size": "0.9375rem", "color": TEXT_SECONDARY,
                        "& h2": {"color": TEXT_PRIMARY, "font_family": FONT_DISPLAY,
                                  "font_weight": "400", "margin_top": "2rem"},
                        "& h3": {"color": PRIMARY, "font_family": FONT_DISPLAY,
                                  "font_weight": "400", "margin_top": "1.5rem"},
                        "& strong": {"color": TEXT_PRIMARY},
                        "& code": {"font_family": FONT_MONO, "color": SECONDARY,
                                    "background": SURFACE, "padding": "0.125rem 0.375rem",
                                    "border_radius": "2px"},
                        "& table": {"width": "100%", "border_collapse": "collapse",
                                     "margin": "1rem 0"},
                        "& th, & td": {"padding": "0.5rem 1rem", "text_align": "left",
                                        "border_bottom": f"1px solid {GHOST_BORDER}"},
                        "& th": {"color": PRIMARY, "font_weight": "600"},
                        },
            ),
            rx.center(
                rx.text("Run a simulation to see the physics narrative.",
                         color=TEXT_MUTED, font_style="italic"),
                padding="3rem",
            ),
        ),
        padding="1.5rem", width="100%",
        style={"max_height": "60vh", "overflow_y": "auto"},
    )


def export_card(icon_name: str, title: str, description: str,
                button_label: str) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.icon(icon_name, size=24, color=PRIMARY),
            rx.text(title, font_weight="600", font_size="0.9375rem", color=TEXT_PRIMARY),
            rx.text(description, font_size="0.8125rem", color=TEXT_SECONDARY, line_height="1.5"),
            rx.spacer(),
            rx.button(button_label, style=button_ghost, size="1", width="100%"),
            spacing="3", height="100%",
        ),
        style=card_style, height="200px",
    )
