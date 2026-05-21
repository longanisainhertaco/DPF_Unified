#!/usr/bin/env python3
"""Sprint 8 WS5 -- PlasmaPy pinned cross-check for the Braginskii Z=1 closure.

PlasmaPy is a CROSS-CHECK LANE ONLY. It is never a source authority and it can
NEVER promote or demote the Braginskii 1965 closure. The scientific authority
for the Z=1 transport closure is the render-verified Braginskii 1965 source
(Table 2 + Eqs. 4.30-4.45); see
``src/dpf/first_principles/sprint8_braginskii_z1_transport.py``.

This script compares, at a single representative DPF-relevant Z=1 deuterium
plasma point:

- ``braginskii_z1_parallel_resistivity``      vs PlasmaPy ``ClassicalTransport.resistivity``
- ``braginskii_z1_electron_parallel_conductivity`` vs PlasmaPy ``electron_thermal_conductivity``
- ``braginskii_z1_ion_parallel_conductivity``      vs PlasmaPy ``ion_thermal_conductivity``

It reports PASS / DISCREPANCY against a fixed relative tolerance. A DISCREPANCY
is review telemetry only -- it does not change any acceptance flag.

Pinned environment (cross-check reproducibility, NOT an authority claim):

- PlasmaPy ``2026.2.0``
- Astropy ``7.2.0``
- Python ``>=3.12`` (run with ``.venv312/bin/python``)

Run::

    cd /Users/anthonyzamora/dpf-unified
    .venv312/bin/python scripts/plasmapy_braginskii_z1_crosscheck.py

Exit code 0 always (cross-check is informational); the printed verdict is the
signal. Use ``--strict`` to make a DISCREPANCY exit non-zero for CI gating.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Pinned cross-check environment. PlasmaPy/Astropy are an optional audit
# extra; if absent, the cross-check is SKIPPED -- it can never block.
_PINNED = {
    "plasmapy": "2026.2.0",
    "astropy": "7.2.0",
}

# Fixed relative tolerance for the cross-check verdict. PlasmaPy and the
# Braginskii source use slightly different Coulomb-log conventions, so a
# few-percent spread is expected and is NOT a source defect.
_REL_TOLERANCE = 0.05

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _representative_point() -> dict[str, float]:
    """Return a single representative Z=1 deuterium-plasma cross-check point."""

    return {
        "n_e_m3": 1.0e23,
        "n_i_m3": 1.0e23,
        "T_e_K": 1.0e6,
        "T_i_K": 1.0e6,
    }


def _braginskii_collision_times(
    n_e: float, T_e_K: float, n_i: float, T_i_K: float, m_i: float
) -> tuple[float, float, float]:
    """Return ``(tau_e, tau_i, lnLambda)`` in the Braginskii convention.

    Braginskii electron collision time (RPP Vol. 1, eq. for tau_e)::

        tau_e = 3 sqrt(m_e) (k T_e)^1.5 (4 pi eps0)^2
                / (4 sqrt(2 pi) lnLambda e^4 n_i Z^2)

    and the ion collision time with ``4 sqrt(pi)`` and ``Z^4`` in the
    denominator. Z=1 here.
    """

    import numpy as np
    from astropy import units as u
    from astropy.constants import e, eps0, k_B, m_e
    from plasmapy.formulary import Coulomb_logarithm

    T_e = T_e_K * u.K
    T_i = T_i_K * u.K
    n_e_q = n_e * u.m**-3
    n_i_q = n_i * u.m**-3

    ln_ei = float(Coulomb_logarithm(T_e, n_e_q, ("e-", "D+")))
    ln_ii = float(Coulomb_logarithm(T_i, n_i_q, ("D+", "D+")))

    tau_e = (
        3 * np.sqrt(m_e) * (k_B * T_e) ** 1.5 * (4 * np.pi * eps0) ** 2
    ) / (4 * np.sqrt(2 * np.pi) * ln_ei * (e.si) ** 4 * n_i_q * 1**2)
    tau_e = float(tau_e.to(u.s).value)

    m_i_q = m_i * u.kg
    tau_i = (
        3 * np.sqrt(m_i_q) * (k_B * T_i) ** 1.5 * (4 * np.pi * eps0) ** 2
    ) / (4 * np.sqrt(np.pi) * ln_ii * (e.si) ** 4 * n_i_q * 1**4)
    tau_i = float(tau_i.to(u.s).value)

    return tau_e, tau_i, ln_ei


def run_cross_check(strict: bool = False) -> int:
    """Run the PlasmaPy cross-check; return a process exit code."""

    warnings.filterwarnings("ignore")

    try:
        import astropy
        import plasmapy
        from astropy import units as u
        from astropy.constants import m_p
        from plasmapy.formulary.braginskii import ClassicalTransport
    except ImportError as exc:  # cross-check optional -- never blocks
        print("[SKIP] PlasmaPy/Astropy not installed; cross-check skipped.")
        print(f"       reason: {exc}")
        print("       install: pip install 'plasmapy==2026.2.0' 'astropy==7.2.0'")
        return 0

    print("Braginskii 1965 Z=1 transport -- PlasmaPy pinned cross-check")
    print("PlasmaPy is a cross-check lane ONLY; it is never source authority.")
    print(f"  plasmapy {plasmapy.__version__} (pinned {_PINNED['plasmapy']})")
    print(f"  astropy  {astropy.__version__} (pinned {_PINNED['astropy']})")
    if plasmapy.__version__ != _PINNED["plasmapy"]:
        print("  [warn] plasmapy version drift from pin -- review the cross-check.")
    print()

    from dpf.first_principles.sprint8_braginskii_z1_transport import (
        braginskii_z1_electron_parallel_conductivity,
        braginskii_z1_ion_parallel_conductivity,
        braginskii_z1_parallel_resistivity,
    )

    point = _representative_point()
    m_i = float((2.013553 * m_p).to(u.kg).value)  # deuteron mass
    tau_e, tau_i, ln_ei = _braginskii_collision_times(
        point["n_e_m3"], point["T_e_K"], point["n_i_m3"], point["T_i_K"], m_i
    )

    print("Representative Z=1 D+ point:")
    for key, value in point.items():
        print(f"  {key:12s} = {value:.4e}")
    print(f"  tau_e        = {tau_e:.4e} s")
    print(f"  tau_i        = {tau_i:.4e} s")
    print(f"  lnLambda_ei  = {ln_ei:.4f}")
    print()

    # --- Braginskii source-backed candidate (this WS5 packet) ---
    brag_eta = float(
        braginskii_z1_parallel_resistivity(point["n_e_m3"], tau_e)
    )
    brag_kappa_e = float(
        braginskii_z1_electron_parallel_conductivity(
            point["n_e_m3"], point["T_e_K"], tau_e
        )
    )
    brag_kappa_i = float(
        braginskii_z1_ion_parallel_conductivity(
            point["n_i_m3"], point["T_i_K"], tau_i, m_i=m_i
        )
    )

    # --- PlasmaPy cross-check lane ---
    ct = ClassicalTransport(
        T_e=point["T_e_K"] * u.K,
        n_e=point["n_e_m3"] * u.m**-3,
        T_i=point["T_i_K"] * u.K,
        n_i=point["n_i_m3"] * u.m**-3,
        ion="D+",
        Z=1,
        B=0 * u.T,
        model="Braginskii",
        field_orientation="parallel",
    )
    pp_eta = float(ct.resistivity.to(u.ohm * u.m).value)
    pp_kappa_e = float(ct.electron_thermal_conductivity.to(u.W / (u.m * u.K)).value)
    pp_kappa_i = float(ct.ion_thermal_conductivity.to(u.W / (u.m * u.K)).value)

    rows = (
        ("parallel_resistivity [Ohm*m]", brag_eta, pp_eta),
        ("electron_thermal_conductivity [W/(m*K)]", brag_kappa_e, pp_kappa_e),
        ("ion_thermal_conductivity [W/(m*K)]", brag_kappa_i, pp_kappa_i),
    )

    print(f"{'quantity':42s} {'Braginskii(WS5)':>18s} {'PlasmaPy':>16s} {'rel.diff':>11s}  verdict")
    print("-" * 100)
    any_discrepancy = False
    for name, brag, pp in rows:
        rel = abs(brag - pp) / abs(pp) if pp != 0 else float("inf")
        verdict = "PASS" if rel <= _REL_TOLERANCE else "DISCREPANCY"
        if verdict == "DISCREPANCY":
            any_discrepancy = True
        print(f"{name:42s} {brag:18.6e} {pp:16.6e} {rel:10.4%}  {verdict}")
    print("-" * 100)
    print(f"tolerance: rel.diff <= {_REL_TOLERANCE:.0%}")
    print()

    if any_discrepancy:
        print("[DISCREPANCY] one or more quantities exceed the cross-check tolerance.")
        print("  This is review telemetry only. PlasmaPy CANNOT demote the")
        print("  Braginskii source closure; resolve via Braginskii-source review.")
        return 1 if strict else 0

    print("[PASS] all quantities agree with the PlasmaPy cross-check lane.")
    print("  Cross-check agreement does NOT promote acceptance: the closure")
    print("  stays a candidate pending numerical/comparator/certificate gates.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on a cross-check discrepancy (for CI gating)",
    )
    args = parser.parse_args()
    return run_cross_check(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
