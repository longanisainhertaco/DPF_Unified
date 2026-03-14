"""Phase Y — MHD Pressure Coupling Tests.

Tests for the dynamic MHD pressure feedback added to SimulationEngine:
1. _dynamic_sheath_pressure() method on SimulationEngine
   - Rundown phase: averages pressure for z > z_sheath cells
   - Radial/reflected phase: averages pressure for r < r_shock cells
   - Fallback to fill_pressure_Pa when snowplow inactive or no valid cells
2. SnowplowModel._step_radial accepts pressure kwarg via step()
   - pressure=None uses only adiabatic back-pressure
   - pressure > adiabatic increases F_pressure (max semantics)
3. Two-way coupling: L_plasma and dL_dt always present in step result
4. Phase dispatch correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.fluid.snowplow import SnowplowModel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PF1000 = {
    "anode_radius": 0.0575,
    "cathode_radius": 0.08,
    "fill_density": 1e-4,
    "anode_length": 0.16,
    "mass_fraction": 0.3,
    "fill_pressure_Pa": 400.0,
    "current_fraction": 0.7,
}


def _make_snowplow(**overrides) -> SnowplowModel:
    """Return a fresh SnowplowModel from PF-1000-like parameters."""
    params = {**_PF1000, **overrides}
    return SnowplowModel(
        anode_radius=params["anode_radius"],
        cathode_radius=params["cathode_radius"],
        fill_density=params["fill_density"],
        anode_length=params["anode_length"],
        mass_fraction=params["mass_fraction"],
        fill_pressure_Pa=params["fill_pressure_Pa"],
        current_fraction=params["current_fraction"],
    )


def _make_radial_snowplow(**overrides) -> SnowplowModel:
    """Return a SnowplowModel manually placed in radial phase at 95% of cathode."""
    sp = _make_snowplow(**overrides)
    sp.z = sp.L_anode
    sp.v = 1e4
    sp._rundown_complete = True
    sp._L_axial_frozen = sp.L_coeff * sp.L_anode
    sp.phase = "radial"
    sp.r_shock = 0.95 * sp.b
    sp.vr = 0.0
    return sp


def _drive_to_phase(sp: SnowplowModel, target: str, max_steps: int = 200_000) -> bool:
    """Drive a SnowplowModel until phase == target. Returns True on success."""
    for _ in range(max_steps):
        sp.step(1e-9, 1.5e6)
        if sp.phase == target:
            return True
    return False


# ---------------------------------------------------------------------------
# TestStepRadialAcceptsPressure
# ---------------------------------------------------------------------------


class TestStepRadialAcceptsPressure:
    """SnowplowModel.step() accepts pressure kwarg during all phases."""

    def test_step_accepts_pressure_during_rundown(self) -> None:
        """pressure kwarg accepted during rundown without raising."""
        sp = _make_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=500.0)
        assert "phase" in result
        assert result["phase"] == "rundown"

    def test_step_accepts_none_pressure_during_rundown(self) -> None:
        """pressure=None is accepted during rundown without raising."""
        sp = _make_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=None)
        assert "phase" in result

    def test_step_accepts_pressure_during_radial(self) -> None:
        """pressure kwarg accepted during radial phase without raising."""
        sp = _make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=1e5)
        assert "phase" in result
        assert result["phase"] in ("radial", "reflected", "pinch")

    def test_step_accepts_none_pressure_during_radial(self) -> None:
        """pressure=None is accepted during radial phase without raising."""
        sp = _make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=None)
        assert "phase" in result


# ---------------------------------------------------------------------------
# TestRadialPressureFeedback
# ---------------------------------------------------------------------------


class TestRadialPressureFeedback:
    """External pressure interacts with adiabatic back-pressure via max()."""

    def test_pressure_below_adiabatic_has_no_effect(self) -> None:
        """External pressure below adiabatic does not change F_pressure."""
        sp1 = _make_radial_snowplow()
        sp2 = _make_radial_snowplow()

        # Adiabatic at r=0.95*b: p_fill * (b/(0.95b))^(10/3)
        # Very small external pressure (1 Pa) should be dominated by adiabatic.
        r1 = sp1.step(1e-9, 1.5e6, pressure=None)
        r2 = sp2.step(1e-9, 1.5e6, pressure=1.0)

        # Both should give same F_pressure (adiabatic dominates 1 Pa)
        assert r1["F_pressure"] == pytest.approx(r2["F_pressure"], rel=1e-6)

    def test_pressure_above_adiabatic_increases_F_pressure(self) -> None:
        """External pressure > adiabatic back-pressure increases F_pressure."""
        sp1 = _make_radial_snowplow()
        sp2 = _make_radial_snowplow()

        # Get baseline F_pressure with no external pressure
        r_base = sp1.step(1e-9, 1.5e6, pressure=None)
        # Apply very large external pressure (1 TPa — definitely above adiabatic)
        r_high = sp2.step(1e-9, 1.5e6, pressure=1e12)

        assert r_high["F_pressure"] > r_base["F_pressure"]

    def test_high_pressure_slows_inward_velocity(self) -> None:
        """Very high external pressure reduces net inward acceleration."""
        sp1 = _make_radial_snowplow()
        sp2 = _make_radial_snowplow()

        # Many steps with and without extreme external pressure
        for _ in range(100):
            sp1.step(1e-9, 1.5e6, pressure=None)
            sp2.step(1e-9, 1.5e6, pressure=1e12)

        # High back-pressure case should have slower (less negative) vr
        assert abs(sp2.vr) <= abs(sp1.vr)

    def test_F_pressure_uses_max_semantics(self) -> None:
        """F_pressure = max(adiabatic, external) * 2*pi*r*z_f."""
        sp = _make_radial_snowplow()
        # Compute what adiabatic pressure should be
        r_s = sp.r_shock
        gamma = 5.0 / 3.0
        p_adiabatic = sp.p_fill * (sp.b / r_s) ** (2.0 * gamma)

        # External pressure well below adiabatic → F_pressure from adiabatic
        r_low = sp.step(1e-9, 1.5e6, pressure=1.0)
        expected_F = p_adiabatic * 2.0 * np.pi * r_s * sp.L_anode
        assert r_low["F_pressure"] == pytest.approx(expected_F, rel=0.01)

    def test_zero_external_pressure_same_as_none(self) -> None:
        """pressure=0.0 behaves same as pressure=None (adiabatic wins)."""
        sp1 = _make_radial_snowplow()
        sp2 = _make_radial_snowplow()

        r_none = sp1.step(1e-9, 1.5e6, pressure=None)
        r_zero = sp2.step(1e-9, 1.5e6, pressure=0.0)

        # max(adiabatic, 0.0) == adiabatic for positive fill_pressure_Pa
        assert r_none["F_pressure"] == pytest.approx(r_zero["F_pressure"], rel=1e-6)


# ---------------------------------------------------------------------------
# TestDynamicPressureFallback
# ---------------------------------------------------------------------------


class TestDynamicPressureFallback:
    """SimulationEngine._dynamic_sheath_pressure() fallback behaviour.

    Tests use object.__new__ to bypass SimulationEngine.__init__ and set
    only the attributes needed by _dynamic_sheath_pressure().
    """

    def _make_engine_stub(
        self,
        snowplow: SnowplowModel | None,
        state_pressure: np.ndarray | None,
        fill_pressure_Pa: float = 400.0,
        dx: float = 0.01,
        dz: float | None = None,
    ):
        """Build a minimal SimulationEngine stub for _dynamic_sheath_pressure."""
        from dpf.engine import SimulationEngine

        eng = object.__new__(SimulationEngine)

        # Minimal config stub
        class _GeomCfg:
            pass

        class _SnowplowCfg:
            pass

        class _Cfg:
            pass

        geom_cfg = _GeomCfg()
        geom_cfg.dz = dz  # type: ignore[attr-defined]

        snow_cfg = _SnowplowCfg()
        snow_cfg.fill_pressure_Pa = fill_pressure_Pa  # type: ignore[attr-defined]

        cfg = _Cfg()
        cfg.dx = dx  # type: ignore[attr-defined]
        cfg.geometry = geom_cfg  # type: ignore[attr-defined]
        cfg.snowplow = snow_cfg  # type: ignore[attr-defined]

        eng.config = cfg  # type: ignore[attr-defined]
        eng.snowplow = snowplow  # type: ignore[attr-defined]
        eng.state = {} if state_pressure is None else {"pressure": state_pressure}  # type: ignore[attr-defined]

        return eng

    def test_fallback_when_snowplow_is_none(self) -> None:
        """Returns fill_pressure_Pa when snowplow is None."""
        eng = self._make_engine_stub(snowplow=None, state_pressure=None, fill_pressure_Pa=500.0)
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(500.0)

    def test_fallback_when_snowplow_inactive(self) -> None:
        """Returns fill_pressure_Pa when snowplow.is_active is False (pinch)."""
        sp = _make_snowplow()
        # Force to pinch (inactive)
        sp._pinch_complete = True
        sp.phase = "pinch"
        assert not sp.is_active

        eng = self._make_engine_stub(snowplow=sp, state_pressure=None, fill_pressure_Pa=600.0)
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(600.0)

    def test_fallback_when_no_pressure_in_state(self) -> None:
        """Returns fill_pressure_Pa when state has no 'pressure' key."""
        sp = _make_snowplow()
        assert sp.is_active

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=None, fill_pressure_Pa=700.0,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(700.0)

    def test_rundown_phase_averages_ahead_of_sheath(self) -> None:
        """Rundown: uses mean pressure for z > z_sheath cells."""
        sp = _make_snowplow()
        assert sp.phase == "rundown"

        # Pressure array shape (1, 1, 10) — 10 z-cells
        nz = 10
        dx = 0.01
        # Sheath at z=0.02 m → iz = round(0.02/0.01) = 2
        sp.z = 0.02

        p = np.zeros((1, 1, nz), dtype=np.float64)
        # Cells iz+1 and beyond: set high pressure
        p[..., 3:] = 1000.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx, dz=dx,
        )
        result = eng._dynamic_sheath_pressure()
        # Mean of cells [3:10] = 1000 Pa > 400 Pa fallback
        assert result == pytest.approx(1000.0, rel=1e-6)

    def test_rundown_phase_returns_fallback_when_iz_at_end(self) -> None:
        """Rundown: fallback when iz >= nz-1 (sheath past all cells)."""
        sp = _make_snowplow()
        sp.z = 0.20  # beyond nz*dx = 0.10

        nz = 10
        dx = 0.01
        p = np.ones((1, 1, nz), dtype=np.float64) * 999.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx, dz=dx,
        )
        result = eng._dynamic_sheath_pressure()
        # iz = round(0.20/0.01) = 20 >= nz-1=9 → fallback
        assert result == pytest.approx(400.0)

    def test_radial_phase_averages_inside_shock(self) -> None:
        """Radial: uses mean pressure for r < r_shock cells."""
        sp = _make_radial_snowplow()
        assert sp.phase == "radial"

        nr = 16
        dx = 0.005  # 5 mm cells
        # r_shock at 0.95 * b = 0.95 * 0.08 = 0.076 m
        # ir = round(0.076 / 0.005) = 15
        # Ensure r_shock exactly maps
        sp.r_shock = 5 * dx  # 5 cells * 5mm = 0.025 m → ir=5

        p = np.zeros((nr, 1, 1), dtype=np.float64)
        p[:5, ...] = 5000.0   # first 5 cells: inside shock → high pressure
        p[5:, ...] = 100.0    # outside shock → low pressure

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx,
        )
        result = eng._dynamic_sheath_pressure()
        # Mean of p[:5] = 5000 > 400 fallback
        assert result == pytest.approx(5000.0, rel=1e-6)

    def test_radial_phase_returns_fallback_when_ir_is_zero(self) -> None:
        """Radial: fallback when r_shock / dx rounds to 0 (shock at origin)."""
        sp = _make_radial_snowplow()
        sp.r_shock = 0.0  # shock at centre → ir=0 → no valid cells

        nr = 16
        dx = 0.005
        p = np.ones((nr, 1, 1), dtype=np.float64) * 999.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(400.0)

    def test_result_never_below_fallback(self) -> None:
        """_dynamic_sheath_pressure never returns below fill_pressure_Pa."""
        sp = _make_snowplow()
        nz = 10
        dx = 0.01
        sp.z = dx  # iz = 1

        # Very low pressure ahead of sheath
        p = np.ones((1, 1, nz), dtype=np.float64) * 1.0  # 1 Pa

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx, dz=dx,
        )
        result = eng._dynamic_sheath_pressure()
        # max(mean=1.0, fallback=400.0) = 400.0
        assert result >= 400.0

    def test_reflected_phase_uses_radial_averaging(self) -> None:
        """Reflected phase uses the same r < r_shock averaging as radial."""
        sp = _make_radial_snowplow()
        # Force to reflected phase
        sp.phase = "reflected"
        sp.r_shock = 4 * 0.005  # ir = 4
        sp._M_slug_pinch = 1e-6
        sp._p_pinch = 400.0
        assert sp.is_active

        nr = 16
        dx = 0.005
        p = np.zeros((nr, 1, 1), dtype=np.float64)
        p[:4, ...] = 8000.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(8000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# TestTwoWayCoupling
# ---------------------------------------------------------------------------


class TestTwoWayCoupling:
    """Snowplow step result always carries L_plasma and dL_dt for circuit coupling."""

    def test_step_returns_L_plasma(self) -> None:
        """step() always returns 'L_plasma' key."""
        sp = _make_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert "L_plasma" in result
        assert result["L_plasma"] >= 0.0

    def test_step_returns_dL_dt(self) -> None:
        """step() always returns 'dL_dt' key."""
        sp = _make_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert "dL_dt" in result

    def test_L_plasma_increases_during_rundown(self) -> None:
        """L_plasma grows monotonically during axial rundown."""
        sp = _make_snowplow()
        L_prev = 0.0
        for _ in range(1000):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase != "rundown":
                break
            assert result["L_plasma"] >= L_prev
            L_prev = result["L_plasma"]

    def test_L_plasma_increases_during_radial(self) -> None:
        """L_plasma grows monotonically during radial compression."""
        sp = _make_radial_snowplow()
        L_prev = sp.plasma_inductance
        for _ in range(500):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase not in ("radial",):
                break
            assert result["L_plasma"] >= L_prev - 1e-20  # allow tiny float noise
            L_prev = result["L_plasma"]

    def test_dL_dt_positive_during_rundown(self) -> None:
        """dL/dt > 0 while sheath is advancing axially."""
        sp = _make_snowplow()
        for _ in range(500):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase != "rundown":
                break
            if sp.v > 0:
                assert result["dL_dt"] >= 0.0

    def test_dL_dt_positive_during_radial_compression(self) -> None:
        """dL/dt > 0 during inward radial compression (vr < 0 → dL/dt > 0)."""
        sp = _make_radial_snowplow()
        for _ in range(200):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase != "radial":
                break
            if sp.vr < 0:
                assert result["dL_dt"] >= 0.0

    def test_step_returns_all_required_keys(self) -> None:
        """step() result contains all required coupling/diagnostic keys."""
        required = {
            "z_sheath", "v_sheath", "r_shock", "vr_shock",
            "L_plasma", "dL_dt", "swept_mass", "F_magnetic", "F_pressure", "phase",
        }
        sp = _make_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert required.issubset(result.keys())

    def test_L_plasma_at_radial_entry_matches_L_axial(self) -> None:
        """At radial phase entry r_shock ~ b, L_plasma ~ L_axial_frozen."""
        sp = _make_radial_snowplow()
        # r_shock = 0.95*b → small radial contribution
        L_axial = sp._L_axial_frozen
        # Radial contribution at r=0.95b is small but positive
        assert sp.plasma_inductance >= L_axial

    def test_frozen_result_has_zero_forces(self) -> None:
        """After pinch completion, F_magnetic and F_pressure are zero."""
        sp = _make_radial_snowplow()
        # Fast-forward to pinch
        sp._pinch_complete = True
        sp.phase = "pinch"
        result = sp.step(1e-9, 1.5e6)
        assert result["F_magnetic"] == pytest.approx(0.0)
        assert result["F_pressure"] == pytest.approx(0.0)
        assert result["dL_dt"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestPhaseDispatch
# ---------------------------------------------------------------------------


class TestPhaseDispatch:
    """step() correctly dispatches to axial, radial, or reflected handlers."""

    def test_rundown_dispatch(self) -> None:
        """Fresh snowplow starts in rundown and step dispatches to _step_axial."""
        sp = _make_snowplow()
        assert sp.phase == "rundown"
        result = sp.step(1e-9, 1.5e6)
        assert result["phase"] == "rundown"

    def test_radial_dispatch(self) -> None:
        """Manually placed radial snowplow dispatches to _step_radial."""
        sp = _make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert result["phase"] in ("radial", "reflected")

    def test_reflected_dispatch(self) -> None:
        """Snowplow in reflected phase dispatches to _step_reflected."""
        sp = _make_radial_snowplow()
        # Force to reflected
        sp.phase = "reflected"
        sp.r_shock = sp.r_pinch_min
        sp.vr = 0.0
        sp._M_slug_pinch = max(sp.radial_swept_mass, 1e-20)
        sp._p_pinch = sp._adiabatic_back_pressure(sp.r_shock)
        result = sp.step(1e-9, 1.5e6)
        assert result["phase"] in ("reflected", "pinch")

    def test_pinch_complete_returns_frozen(self) -> None:
        """After _pinch_complete=True, step returns frozen result every time."""
        sp = _make_radial_snowplow()
        sp._pinch_complete = True
        sp.phase = "pinch"
        r1 = sp.step(1e-9, 1.5e6)
        r2 = sp.step(1e-9, 1.5e6)
        assert r1["phase"] == "pinch"
        assert r2["phase"] == "pinch"
        assert r1["L_plasma"] == pytest.approx(r2["L_plasma"])

    def test_rundown_transitions_to_radial(self) -> None:
        """Sheath reaching anode length triggers transition to radial phase."""
        sp = _make_snowplow()
        reached_radial = _drive_to_phase(sp, "radial", max_steps=500_000)
        assert reached_radial, "Snowplow should reach radial phase under 1.5 MA drive"

    def test_pressure_kwarg_does_not_affect_rundown_phase_label(self) -> None:
        """Passing pressure= during rundown does not change the phase label returned."""
        sp = _make_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=1e4)
        assert result["phase"] == "rundown"

    def test_pressure_kwarg_during_radial_returns_radial_or_later(self) -> None:
        """Passing pressure= during radial returns radial/reflected/pinch phase."""
        sp = _make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=1e5)
        assert result["phase"] in ("radial", "reflected", "pinch")


# ---------------------------------------------------------------------------
# TestRundownPressureEffect
# ---------------------------------------------------------------------------


class TestRundownPressureEffect:
    """External pressure affects axial back-pressure force during rundown."""

    def test_higher_pressure_reduces_axial_acceleration(self) -> None:
        """Higher fill pressure reduces net axial driving force."""
        sp_lo = _make_snowplow(fill_pressure_Pa=100.0)
        sp_hi = _make_snowplow(fill_pressure_Pa=100.0)

        # Step sp_hi with large external pressure
        for _ in range(100):
            sp_lo.step(1e-9, 1.5e6, pressure=None)
            sp_hi.step(1e-9, 1.5e6, pressure=1e8)

        # Low pressure case should be faster (higher v)
        assert sp_lo.v >= sp_hi.v

    def test_external_pressure_increases_F_pressure_axial(self) -> None:
        """External pressure > adiabatic raises F_pressure in axial step."""
        sp1 = _make_snowplow()
        sp2 = _make_snowplow()
        r1 = sp1.step(1e-9, 1.5e6, pressure=None)
        r2 = sp2.step(1e-9, 1.5e6, pressure=1e9)
        assert r2["F_pressure"] > r1["F_pressure"]

    def test_pressure_none_matches_fill_pressure_Pa_axial(self) -> None:
        """During rundown, pressure=None uses self.p_fill = fill_pressure_Pa."""
        sp1 = _make_snowplow(fill_pressure_Pa=800.0)
        sp2 = _make_snowplow(fill_pressure_Pa=800.0)
        r1 = sp1.step(1e-9, 1.5e6, pressure=None)
        r2 = sp2.step(1e-9, 1.5e6, pressure=800.0)
        assert r1["F_pressure"] == pytest.approx(r2["F_pressure"], rel=1e-9)
