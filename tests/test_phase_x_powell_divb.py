"""Tests for Phase X: Powell div(B) source terms — cylindrical implementation.

Covers the cylindrical-specific Powell terms not tested in test_phase_d_physics.py:
    X.1  Zero div(B) in cylindrical coords: curl-type B -> all sources zero
    X.2  Nonzero div(B) in cylindrical coords: monopole B -> nonzero sources
    X.3  Formula verification: momentum source == -div(B)*B at interior points
    X.4  Engine integration: enable_powell=True cylindrical run without NaN/crash
    X.5  Damping property: applying Powell sources reduces volume-integrated |div(B)|

Cylindrical div(B): (1/r) * d(r * B_r)/dr + dB_z/dz
Only B_r and B_z contribute; B_theta does not affect div(B) in axisymmetric geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geom(nr: int = 16, nz: int = 16, dr: float = 1e-3, dz: float = 1e-3):
    """Return a CylindricalGeometry instance."""
    from dpf.geometry.cylindrical import CylindricalGeometry
    return CylindricalGeometry(nr=nr, nz=nz, dr=dr, dz=dz)


def _make_state_2d(
    nr: int,
    nz: int,
    B: np.ndarray,
    velocity: np.ndarray | None = None,
    rho: float = 1e-4,
    pressure: float = 1e3,
) -> dict[str, np.ndarray]:
    """Build a 2D state dict with shape (nr, nz) scalars and (3, nr, nz) vectors."""
    if velocity is None:
        velocity = np.zeros((3, nr, nz))
    return {
        "rho": np.full((nr, nz), rho),
        "velocity": velocity,
        "pressure": np.full((nr, nz), pressure),
        "B": B,
        "psi": np.zeros((nr, nz)),
    }


def _azimuthal_only_B(nr: int, nz: int, strength: float = 1.0) -> np.ndarray:
    """Return a B field with only B_theta non-zero.

    B = (0, B_theta, 0).  In cylindrical coords:
        div(B) = (1/r)*d(r*0)/dr + d(0)/dz = 0  -> exactly divergence-free.
    """
    B = np.zeros((3, nr, nz))
    B[1] = strength
    return B


def _uniform_Bz_field(nr: int, nz: int, strength: float = 1.0) -> np.ndarray:
    """Return a uniform B_z field.

    B = (0, 0, B_z_const).
        div(B) = (1/r)*d(r*0)/dr + d(const)/dz = 0.
    """
    B = np.zeros((3, nr, nz))
    B[2] = strength
    return B


def _monopole_Br_field(geom, nz: int, amplitude: float = 100.0) -> np.ndarray:
    """Return B_r = amplitude (uniform) -> nonzero div(B).

    div(B) = (1/r)*d(r * amplitude)/dr + 0
           = (1/r)*(amplitude + r*0)
           = amplitude / r

    So div(B) > 0 everywhere except the axis.
    """
    nr = geom.nr
    B = np.zeros((3, nr, nz))
    B[0] = amplitude
    return B


def _linear_Bz_field(nz: int, nr: int, dz: float, amplitude: float = 100.0) -> np.ndarray:
    """Return B_z = amplitude * z -> dB_z/dz = amplitude -> nonzero div(B).

    Only B_z has a z-gradient; B_r = 0 so radial term vanishes.
    """
    B = np.zeros((3, nr, nz))
    z = np.array([(j + 0.5) * dz for j in range(nz)])
    B[2] = amplitude * z[np.newaxis, :]
    return B


# ============================================================
# X.1  TestCylindricalPowellZeroDivB
# ============================================================

class TestCylindricalPowellZeroDivB:
    """B fields that are divergence-free in cylindrical coords produce zero Powell sources."""

    def test_azimuthal_B_zero_sources(self):
        """Pure B_theta has div(B)=0 -> all Powell sources are zero."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        B = _azimuthal_only_B(nr, nz, strength=2.0)
        state_2d = _make_state_2d(nr, nz, B)

        sources = powell_source_terms_cylindrical(state_2d, geom)

        np.testing.assert_allclose(
            sources["div_B"], 0.0, atol=1e-10,
            err_msg="Pure B_theta should give div(B)=0 in cylindrical coords",
        )
        np.testing.assert_allclose(
            sources["dmom_powell"], 0.0, atol=1e-10,
            err_msg="Powell momentum source must vanish when div(B)=0",
        )
        np.testing.assert_allclose(
            sources["denergy_powell"], 0.0, atol=1e-10,
            err_msg="Powell energy source must vanish when div(B)=0",
        )
        np.testing.assert_allclose(
            sources["dB_powell"], 0.0, atol=1e-10,
            err_msg="Powell induction source must vanish when div(B)=0",
        )

    def test_uniform_Bz_zero_sources(self):
        """Uniform B_z has div(B)=0 -> all Powell sources are zero."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        B = _uniform_Bz_field(nr, nz, strength=1.5)
        rng = np.random.default_rng(0)
        vel = rng.standard_normal((3, nr, nz)) * 1e3
        state_2d = _make_state_2d(nr, nz, B, velocity=vel)

        sources = powell_source_terms_cylindrical(state_2d, geom)

        np.testing.assert_allclose(
            sources["div_B"], 0.0, atol=1e-10,
            err_msg="Uniform B_z should give div(B)=0",
        )
        np.testing.assert_allclose(
            sources["dmom_powell"], 0.0, atol=1e-10,
            err_msg="Zero div(B) must give zero momentum source regardless of velocity",
        )

    def test_zero_sources_with_nonzero_velocity(self):
        """Velocity does not affect div(B) or momentum source when div(B)=0."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        B = _azimuthal_only_B(nr, nz)
        rng = np.random.default_rng(99)
        vel = rng.standard_normal((3, nr, nz)) * 5e4
        state_2d = _make_state_2d(nr, nz, B, velocity=vel)

        sources = powell_source_terms_cylindrical(state_2d, geom)

        max_mom = float(np.max(np.abs(sources["dmom_powell"])))
        max_eng = float(np.max(np.abs(sources["denergy_powell"])))
        max_ind = float(np.max(np.abs(sources["dB_powell"])))
        assert max_mom < 1e-8, f"dmom_powell should be ~0, got {max_mom:.2e}"
        assert max_eng < 1e-8, f"denergy_powell should be ~0, got {max_eng:.2e}"
        assert max_ind < 1e-8, f"dB_powell should be ~0, got {max_ind:.2e}"


# ============================================================
# X.2  TestCylindricalPowellNonzeroDivB
# ============================================================

class TestCylindricalPowellNonzeroDivB:
    """B fields with nonzero cylindrical divergence produce nonzero Powell sources."""

    def test_uniform_Br_gives_nonzero_divB(self):
        """Uniform B_r = const has div(B) = B_r/r -> nonzero."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dr = 1e-3
        geom = _make_geom(nr, nz, dr=dr)
        B = _monopole_Br_field(geom, nz, amplitude=50.0)
        state_2d = _make_state_2d(nr, nz, B)

        sources = powell_source_terms_cylindrical(state_2d, geom)

        max_divB = float(np.max(np.abs(sources["div_B"])))
        assert max_divB > 1.0, (
            f"Uniform B_r should produce |div(B)| >> 0, got max={max_divB:.2e}"
        )
        max_mom = float(np.max(np.abs(sources["dmom_powell"])))
        assert max_mom > 0.0, "Nonzero div(B) must produce nonzero momentum source"

    def test_linear_Bz_gives_nonzero_sources(self):
        """B_z linearly increasing in z has dB_z/dz = const -> nonzero div(B)."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)
        B = _linear_Bz_field(nz, nr, dz, amplitude=200.0)
        vel = np.ones((3, nr, nz)) * 1e4
        state_2d = _make_state_2d(nr, nz, B, velocity=vel)

        sources = powell_source_terms_cylindrical(state_2d, geom)

        max_divB = float(np.max(np.abs(sources["div_B"])))
        assert max_divB > 10.0, (
            f"Linear B_z should give |div(B)| ~ 200 interior, got max={max_divB:.2e}"
        )

        max_mom = float(np.max(np.abs(sources["dmom_powell"])))
        assert max_mom > 0.0, "Nonzero div(B) must give nonzero momentum source"

        max_eng = float(np.max(np.abs(sources["denergy_powell"])))
        assert max_eng > 0.0, "Energy source nonzero when div(B)!=0 and v.B!=0"

        max_ind = float(np.max(np.abs(sources["dB_powell"])))
        assert max_ind > 0.0, "Induction source nonzero when div(B)!=0 and v!=0"

    def test_nonzero_divB_with_zero_velocity(self):
        """With v=0, energy and induction sources vanish but momentum source survives."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)
        B = _linear_Bz_field(nz, nr, dz, amplitude=100.0)
        state_2d = _make_state_2d(nr, nz, B, velocity=np.zeros((3, nr, nz)))

        sources = powell_source_terms_cylindrical(state_2d, geom)

        max_eng = float(np.max(np.abs(sources["denergy_powell"])))
        max_ind = float(np.max(np.abs(sources["dB_powell"])))
        max_mom = float(np.max(np.abs(sources["dmom_powell"])))

        # v=0 -> v.B=0 -> energy source=0
        assert max_eng < 1e-20, f"Energy source should be 0 when v=0, got {max_eng:.2e}"
        # v=0 -> induction source=-div(B)*v=0
        assert max_ind < 1e-20, f"Induction source should be 0 when v=0, got {max_ind:.2e}"
        # momentum source -div(B)*B is still nonzero
        assert max_mom > 0.0, "Momentum source -div(B)*B nonzero even when v=0"


# ============================================================
# X.3  TestCylindricalPowellFormula
# ============================================================

class TestCylindricalPowellFormula:
    """Verify Powell source term formulas are implemented correctly."""

    def test_momentum_source_equals_neg_divB_times_B(self):
        """Interior momentum source = -div(B) * B (component-wise)."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)
        B = _linear_Bz_field(nz, nr, dz, amplitude=150.0)
        state_2d = _make_state_2d(nr, nz, B)

        sources = powell_source_terms_cylindrical(state_2d, geom)
        div_B = sources["div_B"]

        # Interior slice to avoid one-sided boundary difference artifacts
        s = (slice(2, -2), slice(2, -2))
        for component in range(3):
            expected = -div_B[s] * B[component][s]
            np.testing.assert_allclose(
                sources["dmom_powell"][component][s],
                expected,
                rtol=1e-12,
                err_msg=f"Momentum source component {component} mismatch",
            )

    def test_energy_source_equals_neg_divB_vdotB(self):
        """Interior energy source = -div(B) * (v . B)."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)
        B = _linear_Bz_field(nz, nr, dz, amplitude=80.0)
        vel = np.ones((3, nr, nz)) * 2e4
        state_2d = _make_state_2d(nr, nz, B, velocity=vel)

        sources = powell_source_terms_cylindrical(state_2d, geom)
        div_B = sources["div_B"]
        v_dot_B = np.sum(vel * B, axis=0)

        s = (slice(2, -2), slice(2, -2))
        expected = -div_B[s] * v_dot_B[s]
        np.testing.assert_allclose(
            sources["denergy_powell"][s],
            expected,
            rtol=1e-12,
            err_msg="Energy source mismatch: should be -div(B)*(v.B)",
        )

    def test_induction_source_equals_neg_divB_times_v(self):
        """Interior induction source = -div(B) * v (component-wise)."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)
        B = _linear_Bz_field(nz, nr, dz, amplitude=60.0)
        rng = np.random.default_rng(42)
        vel = rng.standard_normal((3, nr, nz)) * 3e4
        state_2d = _make_state_2d(nr, nz, B, velocity=vel)

        sources = powell_source_terms_cylindrical(state_2d, geom)
        div_B = sources["div_B"]

        s = (slice(2, -2), slice(2, -2))
        for component in range(3):
            expected = -div_B[s] * vel[component][s]
            np.testing.assert_allclose(
                sources["dB_powell"][component][s],
                expected,
                rtol=1e-12,
                err_msg=f"Induction source component {component} mismatch",
            )

    def test_return_dict_keys_present(self):
        """Return dict must contain the four expected keys."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 8, 8
        geom = _make_geom(nr, nz)
        B = _azimuthal_only_B(nr, nz)
        state_2d = _make_state_2d(nr, nz, B)
        sources = powell_source_terms_cylindrical(state_2d, geom)
        for key in ("dmom_powell", "denergy_powell", "dB_powell", "div_B"):
            assert key in sources, f"Missing key: {key}"

    def test_output_shapes_match_input(self):
        """Output array shapes match (3, nr, nz) for vectors and (nr, nz) for scalars."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 12, 20
        geom = _make_geom(nr, nz)
        B = _azimuthal_only_B(nr, nz)
        state_2d = _make_state_2d(nr, nz, B)
        sources = powell_source_terms_cylindrical(state_2d, geom)

        assert sources["dmom_powell"].shape == (3, nr, nz)
        assert sources["dB_powell"].shape == (3, nr, nz)
        assert sources["denergy_powell"].shape == (nr, nz)
        assert sources["div_B"].shape == (nr, nz)

    def test_cylindrical_divB_uses_1_over_r_weight(self):
        """Cylindrical div(B) includes the 1/r weighting absent in Cartesian formula.

        For B_r = const, Cartesian div would give 0 (no radial gradient),
        but cylindrical div gives B_r/r (nonzero via the 1/r * d(r*B_r)/dr term).
        """
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dr = 1e-3
        geom = _make_geom(nr, nz, dr=dr)
        B = np.zeros((3, nr, nz))
        B[0] = 1.0  # uniform B_r — no radial gradient, but div(B) = 1/r != 0
        state_2d = _make_state_2d(nr, nz, B)

        sources = powell_source_terms_cylindrical(state_2d, geom)
        div_B = sources["div_B"]

        # Interior cells: div(B) should be approximately 1/r[i]
        for i in range(3, nr - 3):
            r_i = geom.r[i]
            expected_approx = 1.0 / r_i
            # Loose tolerance — finite differences on 1/r have discretisation error
            ratio = float(np.mean(np.abs(div_B[i, 3:-3]))) / expected_approx
            assert 0.8 < ratio < 1.2, (
                f"At i={i}, r={r_i:.3e}: mean|div_B|={float(np.mean(np.abs(div_B[i,3:-3]))):.3e}, "
                f"expected ~1/r={expected_approx:.3e} (ratio={ratio:.3f})"
            )


# ============================================================
# X.4  TestPowellEngineIntegration
# ============================================================

class TestPowellEngineIntegration:
    """Engine integration tests for enable_powell=True in cylindrical mode."""

    def _make_powell_cyl_config(self, nr: int = 8, nz: int = 16):
        """SimulationConfig for a cylindrical run with Powell enabled."""
        from dpf.config import SimulationConfig

        return SimulationConfig(
            grid_shape=[nr, 1, nz],
            dx=0.01,
            sim_time=1e-7,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            geometry={"type": "cylindrical"},
            fluid={"enable_powell": True, "backend": "python"},
            radiation={"bremsstrahlung_enabled": False},
        )

    def test_engine_init_with_powell_enabled(self):
        """Engine accepts enable_powell=True without raising during init."""
        from dpf.engine import SimulationEngine

        config = self._make_powell_cyl_config()
        engine = SimulationEngine(config)
        assert engine.config.fluid.enable_powell is True
        assert engine.geometry_type == "cylindrical"

    def test_engine_step_no_nan(self):
        """Single engine step with Powell enabled produces no NaN."""
        from dpf.engine import SimulationEngine

        config = self._make_powell_cyl_config()
        engine = SimulationEngine(config)
        engine.run(max_steps=1)
        for key in ("rho", "velocity", "pressure", "B"):
            arr = engine.state[key]
            assert np.all(np.isfinite(arr)), (
                f"NaN/Inf detected in '{key}' after one Powell step"
            )

    def test_engine_multi_step_no_nan(self):
        """Multiple engine steps with Powell enabled remain finite."""
        from dpf.engine import SimulationEngine

        config = self._make_powell_cyl_config()
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5
        for key in ("rho", "velocity", "pressure", "B"):
            arr = engine.state[key]
            assert np.all(np.isfinite(arr)), (
                f"NaN/Inf in '{key}' after 5 Powell steps"
            )

    def test_engine_pressure_stays_positive(self):
        """Pressure floor is respected after Powell source application."""
        from dpf.engine import SimulationEngine

        config = self._make_powell_cyl_config()
        engine = SimulationEngine(config)
        engine.run(max_steps=3)
        assert np.all(engine.state["pressure"] > 0), (
            "Pressure went non-positive during Powell integration"
        )

    def test_engine_density_stays_positive(self):
        """Density floor is respected; Powell sources do not modify density."""
        from dpf.engine import SimulationEngine

        config = self._make_powell_cyl_config()
        engine = SimulationEngine(config)
        engine.run(max_steps=3)
        assert np.all(engine.state["rho"] > 0), (
            "Density went non-positive during Powell integration"
        )

    def test_powell_disabled_by_default(self):
        """enable_powell defaults to False; engine runs identically without it."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[8, 1, 16],
            dx=0.01,
            sim_time=1e-7,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": False},
        )
        assert config.fluid.enable_powell is False
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=3)
        assert summary["steps"] == 3


# ============================================================
# X.5  TestPowellReducesDivB
# ============================================================

class TestPowellReducesDivB:
    """Applying Powell source terms should damp div(B) over time."""

    def test_single_application_reduces_divB_rms(self):
        """One explicit Powell update step decreases RMS div(B) in a monopole state.

        The Powell term adds -div(B)*v to the induction equation, which advects
        the divergence error away. For a non-trivial velocity field, a single
        step should reduce or maintain the volume-integrated |div(B)|.

        Note: Powell alone does not eliminate div(B); it advects/damps it in
        a characteristics-based way. This test verifies monotone behaviour
        under a single explicit-Euler update, not full elimination.
        """
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)

        # Non-trivial div(B): linear B_z growth
        B = _linear_Bz_field(nz, nr, dz, amplitude=50.0).copy()
        vel = np.zeros((3, nr, nz))
        # Uniform outflow velocity so -div(B)*v is nonzero
        vel[2] = 1e3
        state_2d = _make_state_2d(nr, nz, B, velocity=vel)

        # Initial div(B) RMS
        sources_before = powell_source_terms_cylindrical(state_2d, geom)
        divB_before = sources_before["div_B"]
        rms_before = float(np.sqrt(np.mean(divB_before**2)))

        # Apply one explicit-Euler Powell update to B
        dt = 1e-9  # small enough that Euler step is stable
        dB = sources_before["dB_powell"]
        B_updated = B + dt * dB

        state_updated = dict(state_2d)
        state_updated["B"] = B_updated

        sources_after = powell_source_terms_cylindrical(state_updated, geom)
        divB_after = sources_after["div_B"]
        rms_after = float(np.sqrt(np.mean(divB_after**2)))

        # Powell correction should not increase div(B) — allow marginal tolerance
        assert rms_after <= rms_before * 1.01, (
            f"Powell induction update must not increase div(B) RMS: "
            f"before={rms_before:.4e}, after={rms_after:.4e}"
        )

    def test_divB_bounded_after_multiple_steps(self):
        """Volume-integrated |div(B)| does not blow up over multiple Powell-applied steps."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)

        B = _linear_Bz_field(nz, nr, dz, amplitude=30.0).copy()
        vel = np.ones((3, nr, nz)) * 5e2
        state_2d = _make_state_2d(nr, nz, B, velocity=vel)

        sources_init = powell_source_terms_cylindrical(state_2d, geom)
        rms_init = float(np.sqrt(np.mean(sources_init["div_B"] ** 2)))

        dt = 1e-9
        current_B = B.copy()
        for _ in range(10):
            state_2d["B"] = current_B
            sources = powell_source_terms_cylindrical(state_2d, geom)
            current_B = current_B + dt * sources["dB_powell"]

        state_2d["B"] = current_B
        sources_final = powell_source_terms_cylindrical(state_2d, geom)
        rms_final = float(np.sqrt(np.mean(sources_final["div_B"] ** 2)))

        # Div(B) should not have grown by more than 2x after 10 small steps
        assert rms_final < rms_init * 2.0, (
            f"div(B) RMS grew unexpectedly: init={rms_init:.4e}, final={rms_final:.4e}"
        )

    def test_zero_velocity_preserves_divB(self):
        """With v=0, Powell induction source is zero so B field is unchanged."""
        from dpf.fluid.mhd_solver import powell_source_terms_cylindrical

        nr, nz = 16, 16
        dz = 1e-3
        geom = _make_geom(nr, nz, dz=dz)

        B = _linear_Bz_field(nz, nr, dz, amplitude=40.0).copy()
        state_2d = _make_state_2d(nr, nz, B, velocity=np.zeros((3, nr, nz)))

        sources = powell_source_terms_cylindrical(state_2d, geom)
        dB = sources["dB_powell"]

        # With zero velocity, induction source = -div(B)*v = 0
        np.testing.assert_allclose(
            dB, 0.0, atol=1e-20,
            err_msg="Induction source must be zero when velocity is zero",
        )
