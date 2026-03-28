"""Design system — Plasma Physics Precision (from Stitch DESIGN.md)."""

# Surface hierarchy (no borders, tonal shifts)
SURFACE = "#0f131f"
SURFACE_LOW = "#171b28"
SURFACE_HIGH = "#262a37"
SURFACE_HIGHEST = "#313442"
SURFACE_BRIGHT = "#353946"
GHOST_BORDER = "rgba(66, 71, 84, 0.2)"

# Accent colors (Okabe-Ito influenced, phase-coded)
PRIMARY = "#adc6ff"
PRIMARY_CONTAINER = "#4d8eff"
SECONDARY = "#ffb95f"
SECONDARY_CONTAINER = "#ee9800"
ERROR = "#ffb4ab"
TERTIARY = "#ddb7ff"
SUCCESS = "#10b981"
WARNING = "#f59e0b"

# Phase colors
PHASE_RUNDOWN = "#adc6ff"
PHASE_RADIAL = "#ffb95f"
PHASE_PINCH = "#ff6b6b"
PHASE_POST = "#ddb7ff"

# Text
TEXT_PRIMARY = "#dfe2f3"
TEXT_SECONDARY = "#9ca3af"
TEXT_MUTED = "#6b7280"

# Typography
FONT_DISPLAY = "Space Grotesk, sans-serif"
FONT_BODY = "Inter, sans-serif"
FONT_MONO = "IBM Plex Mono, monospace"

# Global styles
GLOBAL_STYLES = {
    "font_family": FONT_BODY,
    "background_color": SURFACE,
    "color": TEXT_PRIMARY,
    "::selection": {
        "background_color": "rgba(173, 198, 255, 0.3)",
    },
}

# Component style presets
card_style = {
    "background": SURFACE_LOW,
    "border_radius": "4px",
    "padding": "1.25rem",
    "border": f"1px solid {GHOST_BORDER}",
}

glass_style = {
    "background": "rgba(27, 31, 44, 0.6)",
    "backdrop_filter": "blur(12px)",
    "border_radius": "4px",
    "border": f"1px solid {GHOST_BORDER}",
}

metric_card_style = {
    "background": SURFACE_LOW,
    "border_radius": "4px",
    "padding": "1rem",
    "border": f"1px solid {GHOST_BORDER}",
    "min_width": "200px",
    "flex": "1",
}

nav_style = {
    "background": SURFACE,
    "border_bottom": f"1px solid {GHOST_BORDER}",
    "padding": "0.75rem 1.5rem",
    "position": "sticky",
    "top": "0",
    "z_index": "50",
}

button_primary = {
    "background": f"linear-gradient(135deg, {PRIMARY_CONTAINER}, {PRIMARY})",
    "color": "#0a0e1a",
    "border": "none",
    "border_radius": "2px",
    "font_weight": "600",
    "font_family": FONT_DISPLAY,
    "cursor": "pointer",
    "padding": "0.75rem 2rem",
    "font_size": "1rem",
    "_hover": {"opacity": "0.9"},
}

button_ghost = {
    "background": "transparent",
    "color": TEXT_PRIMARY,
    "border": f"1px solid {GHOST_BORDER}",
    "border_radius": "2px",
    "cursor": "pointer",
    "padding": "0.5rem 1rem",
    "_hover": {"background": SURFACE_HIGH},
}

label_style = {
    "font_size": "0.6875rem",
    "text_transform": "uppercase",
    "letter_spacing": "0.05em",
    "color": TEXT_SECONDARY,
    "font_weight": "500",
}

value_style = {
    "font_size": "2rem",
    "font_weight": "300",
    "font_family": FONT_DISPLAY,
    "font_variant_numeric": "tabular-nums",
    "color": TEXT_PRIMARY,
    "line_height": "1.1",
}

unit_style = {
    "font_size": "0.875rem",
    "color": TEXT_SECONDARY,
    "font_family": FONT_MONO,
    "margin_left": "0.25rem",
}
