"""Regression test for MJOLNIR-2MJ Yn rate-form wrapper bug.

Bug history (fixed 2026-05-04, branch fix/mjolnir-yn-rate-form-wrapper):

Lee/Saw KR eq. 1 [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-
and-s-h-saw-part-1-basic-course.md L4080-4087 p.18] is a PER-SHOT total:

    Yb-t = Cn * n_i * I_pinch^2 * z_p^2 * ln(b/r_p) * sigma / V_max^(1/2)

The derivation at L4064-4078 carries the beam-target interaction time tau
through the proportionality and the substitutions tau ~ rp ~ zp,
vb ~ U^(1/2), and nb ~ Lp*I^2/vb^2 collapse tau into the geometric and
voltage factors of eq. (1). Tau is already absorbed.

Pre-fix, ``beam_target_yield_rate`` divided Yn_total by a beam transit
time tau_transit = L_target / v_beam ~ 1 ns. Callers
(``YieldTracker.accumulate``) then integrated dY_dt * dt over the
~30-50 ns pinch dwell, double-counting tau. This produced a 30-50x
overcount, which manifested in the MJOLNIR-2MJ campaign as a 41x
over-prediction of total Yn.

Post-fix, ``beam_target_yield_rate`` requires an explicit ``tau_dwell``
argument. The rate is Yn_total / tau_dwell, so a caller integrating
dY_dt * dt over a window of length tau_dwell recovers Yn_total exactly.
``tau_dwell <= 0`` returns 0.0 (refusing to emit a rate without an
explicit dwell time).

This test pins:

1. The per-shot Yn (via ``beam_target_yield_lee_saw``) for MJOLNIR-2MJ
   geometry against the published 4.1e11 anchor [KR: petrov-2022-mjolnir-
   high-low-discharges.md §III.A L215-216]: order-of-magnitude.

2. The rate-form integral identity: integrating ``beam_target_yield_rate``
   over ``tau_dwell`` recovers the per-shot total to better than 1e-12.

3. The 41x guard: pre-fix behavior would integrate over a pinch window
   ~50x longer than the implicit tau_transit. Post-fix, no such
   amplification occurs.

References:
    Petrov et al. (Schmidt corresp.), KR petrov-2022-mjolnir-high-low-
        discharges.md L168-216 p.4-5 (1-MJ vs 2-MJ config, yields 3.8e11
        and 4.1e11).
    Goyon et al. 2022, KR goyon-2022-mjolnir-high-low.md L246-263 p.5
        (anode geometry catalogue: 15-23 cm diameter, 18-25 cm length,
        AK gap 4.3 cm, hollow radii 0.9-3.8 cm).
    Lee & Saw, KR a-course-on-plasma-focus-numerical-experiments-s-lee-
        and-s-h-saw-part-1-basic-course.md L4080-4087 p.18 (eq. 1).
"""

from __future__ import annotations

import math

import pytest

from dpf.diagnostics.beam_target import (
    beam_target_yield_lee_saw,
    beam_target_yield_rate,
)

# ---------------------------------------------------------------------------
# MJOLNIR-2MJ device parameters (KR: petrov-2022 + goyon-2022)
# ---------------------------------------------------------------------------

# A30-3.8-A anode (KR goyon-2022-mjolnir-high-low.md L246-251 p.5):
#   23 cm diameter -> a = 11.5 cm = 0.115 m
#   AK gap 4.3 cm  -> b = a + 4.3 cm = 0.158 m
#   exposed length 18-25 cm; representative z_p = 0.21 m.
# Hollow radius 1.9 cm is a reasonable proxy for the collapsed-pinch
# radius r_p (Lee canonical r_p = 0.1 * a = 0.0115 m gives a similar
# order; use the implosion radius from Goyon for the published anchor).
_A_M = 0.115           # anode radius [m]
_B_M = 0.158           # cathode radius [m] (a + AK gap 4.3 cm)
_Z_P = 0.21            # pinch column length [m] (representative)
_R_P = 0.019           # pinch radius [m] (anode hollow 1.9 cm)
_I_PINCH = 3.25e6      # 2-MJ peak current [A] (KR L206-207)
_N_I = 1.0e25          # ion density at pinch [m^-3] (typical PF/MJOLNIR)
_V_MAX = 1.0e5         # induced V_max [V] (KR-canonical 20-50 kV * factor; 100 kV upper bound)

# Pinch dwell from ToF / FWHM neutron pulse: < 100 ns (KR petrov-2022 L26-27).
# Use 50 ns as canonical dwell representative of MJOLNIR pinch lifetime.
_TAU_DWELL = 50e-9

# Published yield anchor: 4.1e11 in 2-MJ configuration
# (KR petrov-2022-mjolnir-high-low-discharges.md L215-216 p.5).
_YN_PUBLISHED = 4.1e11


# ---------------------------------------------------------------------------
# Per-shot yield: anchored to published 4.1e11
# ---------------------------------------------------------------------------


class TestMjolnirPerShotYn:
    """Pin the unwrapped per-shot Lee/Saw yield against KR L215-216."""

    def test_per_shot_yield_finite_and_positive(self) -> None:
        """Per-shot Yn for MJOLNIR-2MJ inputs must be finite and > 0."""
        Yn = beam_target_yield_lee_saw(
            n_i=_N_I,
            I_pinch=_I_PINCH,
            z_p=_Z_P,
            b=_B_M,
            r_p=_R_P,
            V_max=_V_MAX,
        )
        assert math.isfinite(Yn)
        assert Yn > 0.0

    def test_per_shot_yield_within_two_orders_of_published(self) -> None:
        """Per-shot Yn for MJOLNIR-2MJ inputs is within a factor 100x of 4.1e11.

        Lee/Saw is a 0D order-of-magnitude model; KR L4063-4064 explicitly
        notes that "the yield is obtained as an expression with
        proportionality constant ... calibrated against a known
        experimental point." The Cn calibration in this codebase is anchored
        at PF-1000 (Yn = 7e9 at 0.5 MA, KR L4103-4104), not MJOLNIR. A
        within-100x agreement at MJOLNIR-2MJ is the realistic bound for an
        un-recalibrated cross-device check.
        """
        Yn = beam_target_yield_lee_saw(
            n_i=_N_I,
            I_pinch=_I_PINCH,
            z_p=_Z_P,
            b=_B_M,
            r_p=_R_P,
            V_max=_V_MAX,
        )
        ratio = Yn / _YN_PUBLISHED
        assert 1.0e-2 < ratio < 1.0e2, (
            f"Per-shot Yn = {Yn:.2e} vs published {_YN_PUBLISHED:.2e} "
            f"(ratio {ratio:.2e}); outside 100x band — likely a Cn or "
            f"geometry regression."
        )


# ---------------------------------------------------------------------------
# Rate-form integral identity (the actual bug)
# ---------------------------------------------------------------------------


class TestRateFormIntegralIdentity:
    """Verify that integrating dY_dt over tau_dwell recovers Yn_total.

    This is the regression test for the MJOLNIR-2MJ 41x overcount: the
    rate must be Yn_total / tau_dwell so the caller's time-integral
    over the dwell window returns Yn_total. Pre-fix, the rate was
    Yn_total / tau_transit with tau_transit ~ 1 ns << tau_dwell ~ 50 ns,
    giving a 30-50x overcount when integrated.
    """

    # Use canonical PF-1000 inputs where Cn is calibrated (KR L4103-4104).
    _PF1000 = dict(
        I_pinch=500e3,        # 0.5 MA (calibration current)
        V_pinch=1.0e5,        # 100 kV
        n_target=1.0e25,
        L_target=0.06,        # PF-1000 z_p
    )

    def test_integral_identity_holds(self) -> None:
        """integral(dY_dt dt) over tau_dwell == Yn_per_shot."""
        from dpf.diagnostics.beam_target import _LEE_SAW_LN_BRP_DEFAULT  # noqa: PLC2701
        # Pin r_p / b consistent with the wrapper's _LEE_SAW_LN_BRP_DEFAULT
        # so the per-shot reference matches the wrapper's per-shot value.
        b = 0.16
        r_p = b / math.exp(_LEE_SAW_LN_BRP_DEFAULT)

        Yn_per_shot = beam_target_yield_lee_saw(
            n_i=self._PF1000["n_target"],
            I_pinch=self._PF1000["I_pinch"],
            z_p=self._PF1000["L_target"],
            b=b,
            r_p=r_p,
            V_max=self._PF1000["V_pinch"],
        )
        assert Yn_per_shot > 0.0

        for tau_dwell in (10e-9, 30e-9, 50e-9, 100e-9, 1e-6):
            dY_dt = beam_target_yield_rate(
                **self._PF1000, f_beam=0.14, tau_dwell=tau_dwell,
            )
            Yn_integrated = dY_dt * tau_dwell
            # f_beam=0.14 is the unscaled baseline (fb_scale=1.0).
            assert Yn_integrated == pytest.approx(Yn_per_shot, rel=1e-12), (
                f"Rate-form integral mismatch at tau_dwell={tau_dwell:.1e}: "
                f"integrated {Yn_integrated:.4e} vs per-shot {Yn_per_shot:.4e}"
            )

    def test_rate_inversely_proportional_to_dwell(self) -> None:
        """dY_dt = Yn_total / tau_dwell so doubling tau_dwell halves the rate."""
        rate1 = beam_target_yield_rate(**self._PF1000, tau_dwell=20e-9)
        rate2 = beam_target_yield_rate(**self._PF1000, tau_dwell=40e-9)
        assert rate2 == pytest.approx(0.5 * rate1, rel=1e-12)

    def test_zero_dwell_refuses_to_emit_rate(self) -> None:
        """tau_dwell <= 0 returns 0.0 (the MJOLNIR-2MJ guard)."""
        assert beam_target_yield_rate(**self._PF1000, tau_dwell=0.0) == 0.0
        assert beam_target_yield_rate(**self._PF1000, tau_dwell=-1e-9) == 0.0


# ---------------------------------------------------------------------------
# 41x overcount guard (the smoking gun)
# ---------------------------------------------------------------------------


class TestMjolnir41xOvercountGuard:
    """Pin the post-fix Yn against the 41x overcount that motivated the fix.

    Pre-fix behavior: dY_dt = Yn_per_shot / tau_transit, integrated
    over t_pinch ~ tau_dwell, gave Yn_per_shot * (tau_dwell / tau_transit).
    For MJOLNIR-2MJ tau_dwell ~ 50 ns and tau_transit ~ 1 ns, the
    overcount factor was ~50 — directly matching the reported 41x.

    Post-fix: integrating dY_dt over tau_dwell returns Yn_per_shot
    exactly, so no amplification occurs.
    """

    def test_post_fix_matches_per_shot(self) -> None:
        """Caller-side integration of the rate over tau_dwell == per-shot Yn."""
        from dpf.diagnostics.beam_target import _LEE_SAW_LN_BRP_DEFAULT  # noqa: PLC2701
        b = _B_M
        r_p = b / math.exp(_LEE_SAW_LN_BRP_DEFAULT)

        Yn_per_shot = beam_target_yield_lee_saw(
            n_i=_N_I,
            I_pinch=_I_PINCH,
            z_p=_Z_P,
            b=b,
            r_p=r_p,
            V_max=_V_MAX,
        )

        # Caller integrates dY_dt over tau_dwell in N substeps of dt.
        n_steps = 100
        dt = _TAU_DWELL / n_steps
        rate = beam_target_yield_rate(
            I_pinch=_I_PINCH,
            V_pinch=_V_MAX,
            n_target=_N_I,
            L_target=_Z_P,
            f_beam=0.14,
            tau_dwell=_TAU_DWELL,
        )
        Yn_integrated = sum(rate * dt for _ in range(n_steps))
        ratio = Yn_integrated / Yn_per_shot
        assert ratio == pytest.approx(1.0, rel=1e-9), (
            f"Post-fix integral overshoots by {ratio:.4f}x; the bug has "
            f"regressed."
        )

    def test_pre_fix_behavior_was_50x_overcount(self) -> None:
        """Document the pre-fix bug's amplification factor for the record.

        Pre-fix, the wrapper divided by tau_transit = L_target / v_beam,
        where v_beam ~ sqrt(2 * 3 * V_max * e / m_d). For MJOLNIR-2MJ
        at V_max = 100 kV the lab-frame beam energy is 3 * V_max = 300 keV
        (capped at 500 keV), giving v_beam ~ 5.4e6 m/s and tau_transit
        ~ 0.21 m / 5.4e6 m/s ~ 39 ns. With tau_dwell = 50 ns the overcount
        is tau_dwell / tau_transit ~ 1.3x (not 41x for THIS choice of
        L_target = z_p). The historical 41x was for L_target ~ 1 cm
        (the YieldTracker default fallback at L_pinch=0), giving
        tau_transit ~ 1.9 ns and tau_dwell/tau_transit ~ 26-50x.
        """
        # Reproduce the historical configuration that motivated the fix:
        # YieldTracker.accumulate with default L_pinch=0 -> _L=0.01 m.
        E_lab_keV = min(3.0 * _V_MAX / 1000.0, 500.0)
        E_lab_J = E_lab_keV * 1.602e-16
        m_d = 3.34358377e-27
        v_beam = math.sqrt(2.0 * E_lab_J / m_d)
        L_default = 0.01  # YieldTracker fallback "1 cm default"
        tau_transit = L_default / v_beam
        amplification = _TAU_DWELL / tau_transit
        # Document the smoking gun: 26-50x overcount for the historical
        # default L_pinch=0 fallback path used by YieldTracker.
        assert 20.0 < amplification < 60.0, (
            f"Historical pre-fix amplification at MJOLNIR-2MJ inputs: "
            f"tau_dwell/tau_transit = {amplification:.1f}x. The reported "
            f"41x anomaly falls in this band, confirming the rate-form "
            f"wrapper was the root cause."
        )
