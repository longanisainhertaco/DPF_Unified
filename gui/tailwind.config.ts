import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/renderer/**/*.{html,tsx,ts}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // DPF glass-cockpit palette
        dpf: {
          bg: "#121212",
          panel: "#1E1E1E",
          input: "#2A2A2A",
          border: "#333333",
          "border-hover": "#555555",
        },
        accent: {
          cyan: "#00E5FF",
          "cyan-dim": "#006677",
          amber: "#FFC107",
          "amber-dim": "#664D00",
          crimson: "#FF5252",
          "crimson-dim": "#661F1F",
          green: "#4CAF50",
          "green-dim": "#1B5E20",
        },
        scope: {
          bg: "#0A0A0A",
          grid: "#003300",
          "grid-dim": "#1A1A1A",
        },
      },
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "SF Mono",
          "Menlo",
          "monospace",
        ],
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "sans-serif",
        ],
      },
      fontSize: {
        "readout-lg": ["2rem", { lineHeight: "1", letterSpacing: "-0.02em" }],
        "readout-md": [
          "1.25rem",
          { lineHeight: "1.2", letterSpacing: "-0.01em" },
        ],
        "readout-sm": [
          "0.875rem",
          { lineHeight: "1.2", letterSpacing: "0.02em" },
        ],
        "label-xs": [
          "0.6875rem",
          { lineHeight: "1.2", letterSpacing: "0.05em" },
        ],
      },
      animation: {
        "pulse-glow": "pulseGlow 2s ease-in-out infinite",
        "fade-in": "fadeIn 0.3s ease-out",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
