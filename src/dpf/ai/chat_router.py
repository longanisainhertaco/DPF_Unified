"""Stateless pattern-matching question router for WALRUS chat interactions.

Routes natural-language questions about DPF plasma physics to the appropriate
AI module (surrogate sweeps, inverse design, predictions) or returns built-in
physics glossary answers.  All regex matching is case-insensitive.

Example usage::

    router = WALRUSChatRouter(surrogate=my_surrogate)
    result = await router.answer("what maximizes neutron yield?")
    print(result["response"])
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physics glossary (~30 terms)
# ---------------------------------------------------------------------------

PHYSICS_GLOSSARY: dict[str, str] = {
    "mhd": (
        "Magnetohydrodynamics (MHD) treats plasma as a single electrically "
        "conducting fluid coupled to Maxwell's equations."
    ),
    "pinch": (
        "A pinch is a plasma confinement configuration where magnetic pressure "
        "compresses the plasma column inward."
    ),
    "z-pinch": (
        "A z-pinch drives axial current through plasma so that the azimuthal "
        "magnetic field pinches the column radially inward."
    ),
    "bennett equilibrium": (
        "Bennett equilibrium balances magnetic pinch pressure against kinetic "
        "plasma pressure in a z-pinch: I^2 = (8 pi N k_B T) / mu_0."
    ),
    "bremsstrahlung": (
        "Bremsstrahlung ('braking radiation') is electromagnetic radiation "
        "emitted when charged particles are decelerated by Coulomb collisions."
    ),
    "coulomb logarithm": (
        "The Coulomb logarithm ln(Lambda) measures the ratio of maximum to "
        "minimum impact parameters in Coulomb scattering, typically 10-20 in "
        "laboratory plasmas."
    ),
    "cfl": (
        "The Courant-Friedrichs-Lewy (CFL) condition limits the timestep so "
        "that information cannot travel more than one cell per step: "
        "dt <= dx / v_max."
    ),
    "weno5": (
        "WENO5 (Weighted Essentially Non-Oscillatory, 5th order) reconstruction "
        "uses adaptive stencil weighting to achieve high-order accuracy near "
        "shocks without spurious oscillations."
    ),
    "riemann solver": (
        "A Riemann solver computes inter-cell fluxes by solving the local "
        "discontinuity problem at cell interfaces.  Common choices: HLL, HLLD, "
        "Roe."
    ),
    "hlld": (
        "HLLD (Harten-Lax-van Leer-Discontinuities) is an approximate Riemann "
        "solver for ideal MHD that resolves all four MHD wave families."
    ),
    "spitzer resistivity": (
        "Spitzer resistivity eta = 0.51 m_e nu_ei / (n_e e^2) describes "
        "classical Coulomb-collision-driven resistivity, scaling as T_e^{-3/2}."
    ),
    "anomalous resistivity": (
        "Anomalous resistivity arises from micro-instabilities (e.g., "
        "lower-hybrid drift) and can exceed Spitzer resistivity by orders of "
        "magnitude near current sheets."
    ),
    "nernst effect": (
        "The Nernst effect is a cross-field thermal transport mechanism where "
        "magnetic field lines are advected by the electron heat flux, important "
        "in steep temperature gradients."
    ),
    "braginskii": (
        "Braginskii transport theory provides anisotropic viscosity and thermal "
        "conduction coefficients for magnetized plasmas, distinguishing parallel "
        "and perpendicular directions relative to B."
    ),
    "powell 8-wave": (
        "The Powell 8-wave formulation adds a source term proportional to "
        "div(B) to the ideal MHD equations, preventing monopole errors from "
        "accumulating."
    ),
    "constrained transport": (
        "Constrained transport (CT) maintains div(B)=0 to machine precision by "
        "evolving magnetic fluxes on cell faces rather than cell-centered B."
    ),
    "flux limiter": (
        "A flux limiter blends low-order (diffusive) and high-order (dispersive) "
        "reconstructions to suppress oscillations near discontinuities while "
        "maintaining accuracy in smooth regions."
    ),
    "revin": (
        "Reversible Instance Normalization (RevIN) normalizes each sample by its "
        "own RMS statistics and reverses the transform after prediction, "
        "improving generalization for time-series models."
    ),
    "walrus": (
        "WALRUS is a 1.3B-parameter Encoder-Processor-Decoder Transformer from "
        "Polymathic AI for learning dynamics of continuum physical systems."
    ),
    "surrogate model": (
        "A surrogate model is a fast approximation (e.g., neural network) "
        "trained on simulation data to replace expensive physics solvers for "
        "parameter sweeps and optimization."
    ),
    "delta prediction": (
        "Delta prediction means the model outputs the state *change* "
        "u(t+1) - u(t), not the absolute state.  The final prediction is "
        "u(t+1) = u(t) + model_output."
    ),
    "dense plasma focus": (
        "A Dense Plasma Focus (DPF) is a pulsed-power device that uses a "
        "coaxial electrode geometry to accelerate, compress, and heat plasma to "
        "fusion-relevant conditions via a z-pinch."
    ),
    "rlc circuit": (
        "The RLC circuit model describes the DPF driver as a capacitor bank "
        "(C) discharging through inductance (L) and resistance (R) into the "
        "plasma load."
    ),
    "athena++": (
        "Athena++ is an open-source, performance-portable MHD code from "
        "Princeton using AMR, constrained transport, and multiple Riemann "
        "solvers."
    ),
    "athenak": (
        "AthenaK is the Kokkos-based successor to Athena++, supporting GPU "
        "execution (CUDA, HIP, SYCL) via Kokkos performance portability."
    ),
    "kokkos": (
        "Kokkos is a C++ performance-portability library that maps parallel "
        "patterns onto different hardware backends (Serial, OpenMP, CUDA, HIP, "
        "SYCL)."
    ),
    "well format": (
        "The Well is an HDF5 dataset format by Polymathic AI for storing "
        "physical simulation trajectories, used by WALRUS for training data."
    ),
    "amr": (
        "Adaptive Mesh Refinement (AMR) dynamically increases grid resolution "
        "where solution gradients are steep, improving accuracy without "
        "uniformly refining the entire domain."
    ),
    "ohmic heating": (
        "Ohmic (Joule) heating deposits energy into the plasma at a rate "
        "eta * J^2, where eta is resistivity and J is current density."
    ),
    "magnetic reconnection": (
        "Magnetic reconnection is a topological rearrangement of magnetic field "
        "lines that converts magnetic energy into kinetic and thermal energy, "
        "often explosively."
    ),
}


# ---------------------------------------------------------------------------
# Intent patterns
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "inverse",
        re.compile(
            r"(?:what|which|how\s+to)\s+(?:maximi[sz]es?|optimi[sz]es?|"
            r"increases?|boosts?|improves?)\s+"
            r"(?:the\s+)?(?:Te|Ti|temperature|density|rho|neutron|yield|"
            r"pressure|compression|B|magnetic)",
            re.IGNORECASE,
        ),
    ),
    (
        "sweep",
        re.compile(
            r"(?:sweep|scan|vary)\s+"
            r"(?:the\s+)?(?:voltage|V0|capacitance|C0?|pressure|"
            r"fill[\s-]?pressure|inductance|L0?|resistance|R0?)\s+"
            r"(?:from|between)\s+([\d.eE+-]+)\s*(?:kV|V|uF|mF|nH|uH|mOhm|Ohm|Torr|Pa|mbar)?\s+"
            r"(?:to|and)\s+([\d.eE+-]+)\s*(?:kV|V|uF|mF|nH|uH|mOhm|Ohm|Torr|Pa|mbar)?",
            re.IGNORECASE,
        ),
    ),
    (
        "sweep_auto",
        re.compile(
            r"how\s+does\s+(?:the\s+)?(?:voltage|V0|capacitance|C0?|pressure|"
            r"fill[\s-]?pressure|inductance|L0?|resistance|R0?)\s+"
            r"(?:affect|influence|change|impact)",
            re.IGNORECASE,
        ),
    ),
    (
        "predict",
        re.compile(
            r"predict\s+(?:the\s+)?(?:next|future|forward)\s+(?:step|state|time)",
            re.IGNORECASE,
        ),
    ),
    (
        "status",
        re.compile(
            r"\b(?:status|loaded|available|ready|model\s+info)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explain",
        re.compile(
            r"(?:what\s+is|explain|define|describe|tell\s+me\s+about)\s+(.+)",
            re.IGNORECASE,
        ),
    ),
    (
        "help",
        re.compile(
            r"\b(?:help|what\s+can\s+you\s+do|capabilities|commands|usage)\b",
            re.IGNORECASE,
        ),
    ),
]

# Map human-friendly parameter names to config keys
_PARAM_ALIASES: dict[str, str] = {
    "voltage": "V0",
    "v0": "V0",
    "capacitance": "C0",
    "c0": "C0",
    "c": "C0",
    "pressure": "pressure0",
    "fill pressure": "pressure0",
    "fill-pressure": "pressure0",
    "inductance": "L0",
    "l0": "L0",
    "resistance": "R0",
    "r0": "R0",
}

# Default sweep ranges when the user does not specify bounds
_DEFAULT_RANGES: dict[str, tuple[float, float, int]] = {
    "V0": (10e3, 50e3, 10),
    "C0": (10e-6, 200e-6, 10),
    "pressure0": (50.0, 1000.0, 10),
    "L0": (10e-9, 500e-9, 10),
    "R0": (1e-3, 100e-3, 10),
}


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------


class WALRUSChatRouter:
    """Stateless pattern-matching question router for WALRUS interactions.

    Routes natural-language questions to DPF AI modules (surrogate sweeps,
    inverse design, predictions) or returns built-in physics glossary answers.

    Args:
        surrogate: Optional loaded :class:`DPFSurrogate` instance.
        ensemble: Optional loaded :class:`EnsemblePredictor` instance.
    """

    def __init__(
        self,
        surrogate: Any | None = None,
        ensemble: Any | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.ensemble = ensemble

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, question: str) -> tuple[str, dict[str, Any]]:
        """Parse *question* into an intent tag and extracted parameters.

        Returns
        -------
        tuple[str, dict]
            ``(intent, params)`` where *intent* is one of
            ``"inverse"``, ``"sweep"``, ``"sweep_auto"``, ``"predict"``,
            ``"status"``, ``"explain"``, ``"help"``, or ``"unknown"``.
        """
        question = question.strip()
        for intent, pattern in _INTENT_PATTERNS:
            match = pattern.search(question)
            if match:
                params = self._extract_params(intent, match, question)
                return intent, params
        return "unknown", {}

    async def answer(
        self, question: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Route *question* to the appropriate handler and return a response.

        Parameters
        ----------
        question : str
            Natural-language question from the user.
        config : dict, optional
            Additional configuration overrides (e.g., simulation params).

        Returns
        -------
        dict
            Response with keys ``response``, ``intent``, ``data``, and
            ``suggestions``.
        """
        intent, params = self.parse(question)
        handler = getattr(self, f"_handle_{intent}", self._handle_unknown)
        try:
            return await handler(params, config or {})
        except Exception as exc:
            logger.error("Chat router error for intent=%s: %s", intent, exc)
            return _response(
                f"An error occurred while processing your request: {exc}",
                intent=intent,
                suggestions=["Try rephrasing your question", "Type 'help' for usage info"],
            )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    async def _handle_inverse(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle inverse-design questions (maximize/optimize targets)."""
        target_field = params.get("target_field", "Te")
        if self.surrogate is not None and self.surrogate.is_loaded:
            return _response(
                f"The surrogate is loaded.  To find the configuration that "
                f"maximizes {target_field}, use the inverse-design panel or run:\n"
                f"  dpf ai inverse-design --target max_{target_field}\n"
                f"This will launch a Bayesian search over device parameters.",
                intent="inverse",
                data={"target_field": target_field, "surrogate_loaded": True},
                suggestions=[
                    "sweep voltage from 10kV to 50kV",
                    f"what is {target_field}?",
                    "predict next step",
                ],
            )
        return _response(
            f"To find the configuration that maximizes {target_field}, load a "
            f"WALRUS surrogate checkpoint first, then run an inverse design:\n"
            f"  dpf ai inverse-design --target max_{target_field}\n"
            f"The optimizer will search over voltage, capacitance, and fill "
            f"pressure to reach the target.",
            intent="inverse",
            data={"target_field": target_field, "surrogate_loaded": False},
            suggestions=[
                "status",
                f"explain {target_field}",
                "help",
            ],
        )

    async def _handle_sweep(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle explicit parameter sweep (e.g., 'sweep voltage from 10kV to 50kV')."""
        param_key = params.get("param_key", "V0")
        lo = params.get("lo", 10e3)
        hi = params.get("hi", 50e3)
        n_points = params.get("n_points", 10)

        if self.surrogate is not None and self.surrogate.is_loaded:
            import numpy as np

            values = np.linspace(lo, hi, n_points).tolist()
            configs = [{param_key: v} for v in values]
            try:
                results = self.surrogate.parameter_sweep(configs, n_steps=50)
                return _response(
                    f"Swept {param_key} over {n_points} points from "
                    f"{lo:.4g} to {hi:.4g}.  Results attached in 'data'.",
                    intent="sweep",
                    data={"param_key": param_key, "values": values, "results": results},
                    suggestions=[
                        "what maximizes Te?",
                        f"sweep {param_key} from {lo} to {hi * 2}",
                        "predict next step",
                    ],
                )
            except Exception as exc:
                return _response(
                    f"Sweep failed: {exc}",
                    intent="sweep",
                    suggestions=["status", "help"],
                )

        return _response(
            f"To sweep {param_key} from {lo:.4g} to {hi:.4g}, load a WALRUS "
            f"surrogate checkpoint first.  Then the router can execute:\n"
            f"  surrogate.parameter_sweep([{{{param_key}: v}} for v in "
            f"linspace({lo}, {hi}, {n_points})])",
            intent="sweep",
            data={"param_key": param_key, "lo": lo, "hi": hi, "surrogate_loaded": False},
            suggestions=["status", "help"],
        )

    async def _handle_sweep_auto(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle auto-range sweep (e.g., 'how does voltage affect...')."""
        param_key = params.get("param_key", "V0")
        lo, hi, n_points = _DEFAULT_RANGES.get(param_key, (10e3, 50e3, 10))
        # Delegate to the explicit sweep handler with default bounds
        return await self._handle_sweep(
            {"param_key": param_key, "lo": lo, "hi": hi, "n_points": n_points},
            config,
        )

    async def _handle_predict(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle single-step prediction request."""
        if self.surrogate is not None and self.surrogate.is_loaded:
            return _response(
                "The surrogate is loaded and ready for predictions.  Provide a "
                "state history (list of DPF state dicts) to "
                "surrogate.predict_next_step(history).",
                intent="predict",
                data={"surrogate_loaded": True},
                suggestions=[
                    "status",
                    "what is delta prediction?",
                    "sweep voltage from 10kV to 50kV",
                ],
            )
        return _response(
            "No surrogate model is currently loaded.  Load a WALRUS checkpoint "
            "to enable next-step predictions:\n"
            "  surrogate = DPFSurrogate('path/to/checkpoint')\n"
            "  router = WALRUSChatRouter(surrogate=surrogate)",
            intent="predict",
            data={"surrogate_loaded": False},
            suggestions=["status", "help"],
        )

    async def _handle_status(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Report model readiness."""
        surrogate_ok = self.surrogate is not None and self.surrogate.is_loaded
        ensemble_ok = self.ensemble is not None
        ensemble_n = getattr(self.ensemble, "n_models", 0) if ensemble_ok else 0

        lines = [
            f"Surrogate loaded: {'yes' if surrogate_ok else 'no'}",
            f"Ensemble loaded:  {'yes' if ensemble_ok else 'no'}"
            + (f" ({ensemble_n} models)" if ensemble_ok else ""),
        ]
        if surrogate_ok:
            lines.append(
                f"Checkpoint: {self.surrogate.checkpoint_path}"  # type: ignore[union-attr]
            )
            lines.append(
                f"Device: {self.surrogate.device}"  # type: ignore[union-attr]
            )

        return _response(
            "\n".join(lines),
            intent="status",
            data={
                "surrogate_loaded": surrogate_ok,
                "ensemble_loaded": ensemble_ok,
                "ensemble_n_models": ensemble_n,
            },
            suggestions=["help", "predict next step", "what is WALRUS?"],
        )

    async def _handle_explain(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Look up a term in the physics glossary."""
        raw_term = params.get("term", "").strip().lower()
        # Try exact match first, then substring
        definition = PHYSICS_GLOSSARY.get(raw_term)
        if definition is None:
            for key, val in PHYSICS_GLOSSARY.items():
                if raw_term in key or key in raw_term:
                    definition = val
                    raw_term = key
                    break

        if definition is not None:
            related = [
                k for k in PHYSICS_GLOSSARY if k != raw_term and raw_term[:3] in k
            ][:3]
            suggestions = [f"explain {r}" for r in related] or [
                "explain MHD",
                "explain WALRUS",
            ]
            return _response(
                definition,
                intent="explain",
                data={"term": raw_term},
                suggestions=suggestions,
            )

        return _response(
            f"Term '{raw_term}' is not in the built-in glossary.  "
            f"Try one of: {', '.join(sorted(PHYSICS_GLOSSARY)[:8])} ...",
            intent="explain",
            data={"term": raw_term, "found": False},
            suggestions=["explain MHD", "explain z-pinch", "help"],
        )

    async def _handle_help(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """List supported question types."""
        text = (
            "Supported question types:\n"
            "  - Inverse design: 'what maximizes Te?' / 'optimize neutron yield'\n"
            "  - Parameter sweep: 'sweep voltage from 10kV to 50kV'\n"
            "  - Auto sweep:     'how does capacitance affect temperature?'\n"
            "  - Prediction:     'predict next step'\n"
            "  - Model status:   'status' / 'is the model loaded?'\n"
            "  - Physics terms:  'what is bremsstrahlung?' / 'explain CFL'\n"
            "  - Help:           'help' / 'what can you do?'"
        )
        return _response(
            text,
            intent="help",
            suggestions=[
                "what maximizes neutron yield?",
                "sweep voltage from 15kV to 40kV",
                "explain z-pinch",
            ],
        )

    async def _handle_unknown(
        self, params: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Fallback for unrecognized questions."""
        return _response(
            "I didn't understand that question.  Try asking about parameter "
            "sweeps, inverse design, predictions, or physics terms.",
            intent="unknown",
            suggestions=[
                "help",
                "what maximizes Te?",
                "explain dense plasma focus",
            ],
        )

    # ------------------------------------------------------------------
    # Param extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_params(
        intent: str, match: re.Match[str], question: str
    ) -> dict[str, Any]:
        """Extract structured parameters from a regex match."""
        params: dict[str, Any] = {}

        if intent == "inverse":
            # Detect which target field is mentioned
            q_lower = question.lower()
            if any(t in q_lower for t in ("te", "temperature")):
                params["target_field"] = "Te"
            elif any(t in q_lower for t in ("rho", "density")):
                params["target_field"] = "rho"
            elif any(t in q_lower for t in ("neutron", "yield")):
                params["target_field"] = "neutron_yield"
            elif "pressure" in q_lower:
                params["target_field"] = "pressure"
            elif any(t in q_lower for t in ("b", "magnetic")):
                params["target_field"] = "B"
            else:
                params["target_field"] = "Te"

        elif intent == "sweep":
            # Extract parameter name, lo, hi from match groups
            param_raw = _extract_param_name(question)
            params["param_key"] = _PARAM_ALIASES.get(param_raw, param_raw)
            try:
                params["lo"] = float(match.group(1))
                params["hi"] = float(match.group(2))
            except (IndexError, ValueError):
                params["lo"] = 10e3
                params["hi"] = 50e3
            params["n_points"] = 10

        elif intent == "sweep_auto":
            param_raw = _extract_param_name(question)
            params["param_key"] = _PARAM_ALIASES.get(param_raw, param_raw)

        elif intent == "explain":
            # Capture everything after 'what is' / 'explain' / 'define'
            term = match.group(1).strip().rstrip("?.,!")
            params["term"] = term

        return params


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _response(
    text: str,
    *,
    intent: str = "unknown",
    data: dict[str, Any] | None = None,
    suggestions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a standardized response dict."""
    return {
        "response": text,
        "intent": intent,
        "data": data or {},
        "suggestions": suggestions or [],
    }


def _extract_param_name(question: str) -> str:
    """Pull the first recognised parameter name from *question*."""
    q_lower = question.lower()
    for alias in sorted(_PARAM_ALIASES, key=len, reverse=True):
        if alias in q_lower:
            return alias
    return "voltage"
