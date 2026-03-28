# Design System Specification: Plasma Physics Precision

## 1. Overview & Creative North Star
**The Creative North Star: "The Observational Monolith"**

This design system moves beyond basic dashboarding into the realm of high-precision instrumentation. It is designed to feel like a high-end research tool—authoritative, dense, and intellectually rigorous. We reject the "bubbly" consumer web in favor of **Organic Brutalism**: a style characterized by mathematical precision, intentional asymmetry, and deep tonal layering.

To break the "template" look, we utilize **Data-Driven Asymmetry**. Large-scale metric values (`display-md`) are offset against dense, mono-spaced technical readouts. We use overlapping "Glass" layers to suggest a 3D simulation environment, ensuring the UI feels like a transparent lens over complex plasma physics data rather than a flat container.

---

## 2. Colors & Surface Logic

The palette is rooted in a "Dark Navy" void, providing the high-contrast foundation necessary for Okabe-Ito safe data visualization.

### Surface Hierarchy & The "No-Line" Rule
Traditional 1px borders are a crutch. In this system, **we prohibit 1px solid borders for sectioning.** Structural boundaries must be defined through tonal shifts.
- **Base Layer:** `surface` (#0f131f).
- **Secondary Sections:** `surface-container-low` (#171b28).
- **Interactive/Floating Elements:** `surface-container-high` (#262a37).

### The "Glass & Gradient" Rule
To elevate the "research tool" aesthetic, main CTAs and critical simulation status indicators should use a **Subtle Radial Gradient** (e.g., `primary` transitioning to `primary-container`). Floating panels must utilize **Glassmorphism**: 
- **Fill:** `surface-container` at 60% opacity.
- **Effect:** `backdrop-blur: 12px`.
- This creates "visual soul," preventing the dark theme from feeling "muddy" or flat.

### Signature Textures
Use the `outline-variant` (#424754) only as a "Ghost Border" at 10% opacity to provide a faint shimmer on the edges of glass panels, mimicking the light refraction on a physical lens.

---

## 3. Typography: The Editorial Scale

Typography is our primary tool for establishing hierarchy in a data-dense environment. We pair the geometric authority of **Space Grotesk** for headlines with the utilitarian clarity of **Inter** for UI and **IBM Plex Mono** for scientific precision.

- **Display (Space Grotesk):** For hero metrics (e.g., Plasma Density). Use `display-md` (2.75rem) with `tabular-nums` to prevent layout jitter during live simulations.
- **Headlines (Space Grotesk):** `headline-sm` (1.5rem) for major module titles.
- **The Technical Label:** `label-sm` (0.6875rem) must always be **UPPERCASE** with **0.05em letter spacing**. This signals "metadata" and differentiates instructions from data.
- **Monospaced Integration:** All equations, coordinates, and raw flux values must use `IBM Plex Mono` to ensure every character occupies the same horizontal space, vital for comparative analysis.

---

## 4. Elevation & Depth: Tonal Layering

We convey hierarchy through **Tonal Layering** rather than structural shadows.

- **The Layering Principle:** Depth is achieved by "stacking" the surface tiers. A `surface-container-lowest` (#0a0e1a) card should sit on a `surface-container-low` section to create a "recessed" effect, suggesting the data is etched into the dashboard.
- **Ambient Shadows:** Shadows are reserved for high-level modals or floating tooltips. Use a `16px` blur, `4%` opacity, tinted with the `primary` color (#adc6ff). Avoid neutral grey shadows.
- **Ghost Border Fallback:** If high-density data requires containment, use a `1px` border of `outline-variant` (#424754) at **20% opacity**. Never use 100% opaque lines.

---

## 5. Components

### Buttons & Interaction
- **Primary:** A soft gradient from `primary` to `primary-container`. No border. `sm` roundedness (0.125rem) for a sharp, technical feel.
- **Secondary (Ghost):** `outline-variant` at 20% opacity. On hover, increase background to `surface-bright`.
- **Tertiary:** Text-only, using `label-md` styling.

### Cards & Data Modules
**Strict Rule:** No divider lines. Separate internal card content using the **Spacing Scale** (e.g., `spacing-4` for logical blocks) or a subtle shift from `surface-container` to `surface-container-highest`.

### High-Precision Inputs
- **Numeric Fields:** Must use `IBM Plex Mono`. The "active" state is defined by a 1px `primary` bottom-border only—no full-box focus ring.
- **Status Chips:** Use Okabe-Ito safe tokens:
    - *Rundown:* `primary` (#adc6ff)
    - *Pinch:* `error` (#ffb4ab)
    - *Post-pinch:* `tertiary` (#ddb7ff)

### Scientific Tooltips
Tooltips should be `surface-container-highest` with a `primary` left-accent bar (2px). Use `body-sm` for descriptions and `IBM Plex Mono` for the exact coordinate or value being inspected.

---

## 6. Do's and Don'ts

### Do
- **Use Tabular Nums:** Always use `font-variant-numeric: tabular-nums` for live-updating data to prevent "jumping" text.
- **Embrace Negative Space:** Use `spacing-12` (2.75rem) between major dashboard modules to allow the eye to rest between dense data sets.
- **Layer for Importance:** Place the most critical simulation controls on the "Highest" surface tier to physically lift them toward the user.

### Don't
- **Don't use 100% white text:** Always use `on-surface` (#dfe2f3) for primary text to prevent eye strain in dark environments.
- **Don't use standard shadows:** Never use a `(0,0,0)` black shadow. It kills the depth of the #0a0e1a navy background.
- **Don't use large radii:** Avoid `rounded-xl` or `full`. Stick to `sm` (2px) or `md` (4px) to maintain the "scientific instrument" aesthetic.