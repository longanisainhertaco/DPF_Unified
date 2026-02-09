/**
 * Client-side WALRUS chat router (pattern-matching fallback).
 *
 * Mirrors the Python WALRUSChatRouter logic so that "explain", "help",
 * "status", "inverse", etc. work even when the Python backend is not
 * running.  The backend is always tried first; this module provides the
 * offline fallback.
 */

// ---------------------------------------------------------------------------
// Physics glossary (~30 terms)
// ---------------------------------------------------------------------------

const PHYSICS_GLOSSARY: Record<string, string> = {
  mhd:
    "Magnetohydrodynamics (MHD) treats plasma as a single electrically " +
    "conducting fluid coupled to Maxwell\u2019s equations.",
  pinch:
    "A pinch is a plasma confinement configuration where magnetic pressure " +
    "compresses the plasma column inward.",
  "z-pinch":
    "A z-pinch drives axial current through plasma so that the azimuthal " +
    "magnetic field pinches the column radially inward.",
  "bennett equilibrium":
    "Bennett equilibrium balances magnetic pinch pressure against kinetic " +
    "plasma pressure in a z-pinch: I\u00b2 = (8\u03c0 N k_B T) / \u03bc\u2080.",
  bremsstrahlung:
    "Bremsstrahlung (\u2018braking radiation\u2019) is electromagnetic radiation " +
    "emitted when charged particles are decelerated by Coulomb collisions.",
  "coulomb logarithm":
    "The Coulomb logarithm ln(\u039b) measures the ratio of maximum to " +
    "minimum impact parameters in Coulomb scattering, typically 10\u201320 in " +
    "laboratory plasmas.",
  cfl:
    "The Courant\u2013Friedrichs\u2013Lewy (CFL) condition limits the timestep so " +
    "that information cannot travel more than one cell per step: " +
    "dt \u2264 dx / v_max.",
  weno5:
    "WENO5 (Weighted Essentially Non-Oscillatory, 5th order) reconstruction " +
    "uses adaptive stencil weighting to achieve high-order accuracy near " +
    "shocks without spurious oscillations.",
  "riemann solver":
    "A Riemann solver computes inter-cell fluxes by solving the local " +
    "discontinuity problem at cell interfaces. Common choices: HLL, HLLD, Roe.",
  hlld:
    "HLLD (Harten\u2013Lax\u2013van Leer\u2013Discontinuities) is an approximate Riemann " +
    "solver for ideal MHD that resolves all four MHD wave families.",
  "spitzer resistivity":
    "Spitzer resistivity \u03b7 = 0.51 m_e \u03bd_ei / (n_e e\u00b2) describes " +
    "classical Coulomb-collision-driven resistivity, scaling as T_e^{\u22123/2}.",
  "anomalous resistivity":
    "Anomalous resistivity arises from micro-instabilities (e.g., " +
    "lower-hybrid drift) and can exceed Spitzer resistivity by orders of " +
    "magnitude near current sheets.",
  "nernst effect":
    "The Nernst effect is a cross-field thermal transport mechanism where " +
    "magnetic field lines are advected by the electron heat flux, important " +
    "in steep temperature gradients.",
  braginskii:
    "Braginskii transport theory provides anisotropic viscosity and thermal " +
    "conduction coefficients for magnetized plasmas, distinguishing parallel " +
    "and perpendicular directions relative to B.",
  "powell 8-wave":
    "The Powell 8-wave formulation adds a source term proportional to " +
    "div(B) to the ideal MHD equations, preventing monopole errors from " +
    "accumulating.",
  "constrained transport":
    "Constrained transport (CT) maintains div(B)=0 to machine precision by " +
    "evolving magnetic fluxes on cell faces rather than cell-centered B.",
  "flux limiter":
    "A flux limiter blends low-order (diffusive) and high-order (dispersive) " +
    "reconstructions to suppress oscillations near discontinuities while " +
    "maintaining accuracy in smooth regions.",
  revin:
    "Reversible Instance Normalization (RevIN) normalizes each sample by its " +
    "own RMS statistics and reverses the transform after prediction, " +
    "improving generalization for time-series models.",
  walrus:
    "WALRUS is a 1.3B-parameter Encoder\u2013Processor\u2013Decoder Transformer from " +
    "Polymathic AI for learning dynamics of continuum physical systems.",
  "surrogate model":
    "A surrogate model is a fast approximation (e.g., neural network) " +
    "trained on simulation data to replace expensive physics solvers for " +
    "parameter sweeps and optimization.",
  "delta prediction":
    "Delta prediction means the model outputs the state *change* " +
    "u(t+1) \u2212 u(t), not the absolute state. The final prediction is " +
    "u(t+1) = u(t) + model_output.",
  "dense plasma focus":
    "A Dense Plasma Focus (DPF) is a pulsed-power device that uses a " +
    "coaxial electrode geometry to accelerate, compress, and heat plasma to " +
    "fusion-relevant conditions via a z-pinch.",
  "rlc circuit":
    "The RLC circuit model describes the DPF driver as a capacitor bank " +
    "(C) discharging through inductance (L) and resistance (R) into the " +
    "plasma load.",
  "athena++":
    "Athena++ is an open-source, performance-portable MHD code from " +
    "Princeton using AMR, constrained transport, and multiple Riemann solvers.",
  athenak:
    "AthenaK is the Kokkos-based successor to Athena++, supporting GPU " +
    "execution (CUDA, HIP, SYCL) via Kokkos performance portability.",
  kokkos:
    "Kokkos is a C++ performance-portability library that maps parallel " +
    "patterns onto different hardware backends (Serial, OpenMP, CUDA, HIP, SYCL).",
  "well format":
    "The Well is an HDF5 dataset format by Polymathic AI for storing " +
    "physical simulation trajectories, used by WALRUS for training data.",
  amr:
    "Adaptive Mesh Refinement (AMR) dynamically increases grid resolution " +
    "where solution gradients are steep, improving accuracy without " +
    "uniformly refining the entire domain.",
  "ohmic heating":
    "Ohmic (Joule) heating deposits energy into the plasma at a rate " +
    "\u03b7 \u00b7 J\u00b2, where \u03b7 is resistivity and J is current density.",
  "magnetic reconnection":
    "Magnetic reconnection is a topological rearrangement of magnetic field " +
    "lines that converts magnetic energy into kinetic and thermal energy, " +
    "often explosively.",
};

// ---------------------------------------------------------------------------
// Intent patterns (match order = priority)
// ---------------------------------------------------------------------------

interface IntentMatch {
  intent: string;
  params: Record<string, string | number>;
}

const INTENT_PATTERNS: Array<{
  intent: string;
  pattern: RegExp;
}> = [
  {
    intent: "inverse",
    pattern:
      /(?:what|which|how\s+to)\s+(?:maximi[sz]es?|optimi[sz]es?|increases?|boosts?|improves?)\s+(?:the\s+)?(?:Te|Ti|temperature|density|rho|neutron|yield|pressure|compression|B|magnetic)/i,
  },
  {
    intent: "sweep",
    pattern:
      /(?:sweep|scan|vary)\s+(?:the\s+)?(?:voltage|V0|capacitance|C0?|pressure|fill[\s-]?pressure|inductance|L0?|resistance|R0?)\s+(?:from|between)\s+([\d.eE+-]+)\s*(?:kV|V|uF|mF|nH|uH|mOhm|Ohm|Torr|Pa|mbar)?\s+(?:to|and)\s+([\d.eE+-]+)\s*(?:kV|V|uF|mF|nH|uH|mOhm|Ohm|Torr|Pa|mbar)?/i,
  },
  {
    intent: "sweep_auto",
    pattern:
      /how\s+does\s+(?:the\s+)?(?:voltage|V0|capacitance|C0?|pressure|fill[\s-]?pressure|inductance|L0?|resistance|R0?)\s+(?:affect|influence|change|impact)/i,
  },
  {
    intent: "predict",
    pattern: /predict\s+(?:the\s+)?(?:next|future|forward)\s+(?:step|state|time)/i,
  },
  {
    intent: "status",
    pattern: /\b(?:status|loaded|available|ready|model\s+info)\b/i,
  },
  {
    intent: "explain",
    pattern: /(?:what\s+is|explain|define|describe|tell\s+me\s+about)\s+(.+)/i,
  },
  {
    intent: "help",
    pattern: /\b(?:help|what\s+can\s+you\s+do|capabilities|commands|usage)\b/i,
  },
];

// ---------------------------------------------------------------------------
// Response builder
// ---------------------------------------------------------------------------

export interface ChatResponse {
  response: string;
  intent: string;
  data: Record<string, unknown>;
  suggestions: string[];
}

function makeResponse(
  text: string,
  intent: string = "unknown",
  data: Record<string, unknown> = {},
  suggestions: string[] = []
): ChatResponse {
  return { response: text, intent, data, suggestions };
}

// ---------------------------------------------------------------------------
// Detect target field for inverse queries
// ---------------------------------------------------------------------------

function detectTargetField(question: string): string {
  const q = question.toLowerCase();
  if (q.includes("te") || q.includes("temperature")) return "Te";
  if (q.includes("rho") || q.includes("density")) return "rho";
  if (q.includes("neutron") || q.includes("yield")) return "neutron_yield";
  if (q.includes("pressure")) return "pressure";
  if (q.includes("b") || q.includes("magnetic")) return "B";
  return "Te";
}

// ---------------------------------------------------------------------------
// Glossary lookup
// ---------------------------------------------------------------------------

function lookupGlossary(term: string): ChatResponse | null {
  const raw = term.trim().toLowerCase().replace(/[?.,!]+$/, "");

  // Exact match
  if (PHYSICS_GLOSSARY[raw]) {
    const related = Object.keys(PHYSICS_GLOSSARY)
      .filter((k) => k !== raw && k.includes(raw.slice(0, 3)))
      .slice(0, 3);
    const suggestions =
      related.length > 0
        ? related.map((r) => `explain ${r}`)
        : ["explain MHD", "explain WALRUS"];
    return makeResponse(PHYSICS_GLOSSARY[raw], "explain", { term: raw }, suggestions);
  }

  // Substring match
  for (const [key, val] of Object.entries(PHYSICS_GLOSSARY)) {
    if (raw.includes(key) || key.includes(raw)) {
      const related = Object.keys(PHYSICS_GLOSSARY)
        .filter((k) => k !== key)
        .slice(0, 3);
      const suggestions = related.map((r) => `explain ${r}`);
      return makeResponse(val, "explain", { term: key }, suggestions);
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Main client-side router
// ---------------------------------------------------------------------------

export function localChatRouter(question: string): ChatResponse {
  const q = question.trim();
  if (!q) {
    return makeResponse(
      "Please type a question about DPF physics, parameter sweeps, or WALRUS.",
      "unknown",
      {},
      ["help", "explain dense plasma focus", "what maximizes Te?"]
    );
  }

  // Try each intent pattern in order
  for (const { intent, pattern } of INTENT_PATTERNS) {
    const match = pattern.exec(q);
    if (!match) continue;

    switch (intent) {
      case "inverse": {
        const target = detectTargetField(q);
        return makeResponse(
          `To find the configuration that maximizes ${target}, load a ` +
            `WALRUS surrogate checkpoint, then use the Inverse Design panel ` +
            `or run:\n  dpf ai inverse-design --target max_${target}\n` +
            `The optimizer will search over voltage, capacitance, and fill ` +
            `pressure to reach the target.`,
          "inverse",
          { target_field: target, surrogate_loaded: false },
          ["status", `explain ${target}`, "help"]
        );
      }

      case "sweep": {
        const lo = match[1] ?? "10000";
        const hi = match[2] ?? "50000";
        return makeResponse(
          `To sweep that parameter from ${lo} to ${hi}, start the Python ` +
            `backend with a WALRUS checkpoint:\n` +
            `  dpf serve --checkpoint path/to/walrus --device cpu\n` +
            `Then the surrogate can evaluate hundreds of configurations in seconds.`,
          "sweep",
          { lo: parseFloat(lo), hi: parseFloat(hi), surrogate_loaded: false },
          ["status", "help"]
        );
      }

      case "sweep_auto": {
        return makeResponse(
          `To see how that parameter affects plasma behavior, start the ` +
            `Python backend with a WALRUS checkpoint and use the Sweep panel. ` +
            `The surrogate evaluates parameter variations in seconds.`,
          "sweep_auto",
          { surrogate_loaded: false },
          ["status", "help", "explain dense plasma focus"]
        );
      }

      case "predict": {
        return makeResponse(
          "No surrogate model is currently loaded. Load a WALRUS checkpoint " +
            "to enable next-step predictions:\n" +
            "  dpf serve --checkpoint path/to/walrus --device cpu",
          "predict",
          { surrogate_loaded: false },
          ["status", "help"]
        );
      }

      case "status": {
        return makeResponse(
          "Surrogate loaded: no (offline mode)\n" +
            "Ensemble loaded:  no\n\n" +
            "The AI backend is not running. Start it with:\n" +
            "  dpf serve --checkpoint path/to/walrus --device cpu",
          "status",
          { surrogate_loaded: false, ensemble_loaded: false },
          ["help", "explain WALRUS", "what is a surrogate model?"]
        );
      }

      case "explain": {
        const term = match[1] ?? "";
        const glossaryResult = lookupGlossary(term);
        if (glossaryResult) return glossaryResult;

        const available = Object.keys(PHYSICS_GLOSSARY).sort().slice(0, 8).join(", ");
        return makeResponse(
          `Term '${term.trim()}' is not in the built-in glossary. ` +
            `Try one of: ${available} ...`,
          "explain",
          { term: term.trim(), found: false },
          ["explain MHD", "explain z-pinch", "help"]
        );
      }

      case "help": {
        return makeResponse(
          "Supported question types:\n" +
            "  \u2022 Inverse design: 'what maximizes Te?' / 'optimize neutron yield'\n" +
            "  \u2022 Parameter sweep: 'sweep voltage from 10kV to 50kV'\n" +
            "  \u2022 Auto sweep:     'how does capacitance affect temperature?'\n" +
            "  \u2022 Prediction:     'predict next step'\n" +
            "  \u2022 Model status:   'status' / 'is the model loaded?'\n" +
            "  \u2022 Physics terms:  'what is bremsstrahlung?' / 'explain CFL'\n" +
            "  \u2022 Help:           'help' / 'what can you do?'\n\n" +
            "Note: Sweep, inverse, and prediction require the Python " +
            "backend running with a WALRUS checkpoint.",
          "help",
          {},
          [
            "what maximizes neutron yield?",
            "sweep voltage from 15kV to 40kV",
            "explain z-pinch",
          ]
        );
      }
    }
  }

  // Unknown intent
  return makeResponse(
    "I didn\u2019t understand that question. Try asking about parameter " +
      "sweeps, inverse design, predictions, or physics terms.",
    "unknown",
    {},
    ["help", "what maximizes Te?", "explain dense plasma focus"]
  );
}
