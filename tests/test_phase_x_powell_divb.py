"""Phase X: Powell div(B) source terms — cylindrical MHD solver tests.

Tests Powell 8-wave formulation (Powell et al., J. Comp. Phys. 154, 284 (1999))
for cylindrical geometry, including zero/nonzero div(B) states, formula
verification, engine integration, and div(B) damping behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.fluid.mhd_solver import powell_source_terms, powell_source_terms_cylindrical
from dpf.geometry.cylindrical import CylindricalGeometry


# ============================================================
# Helper functions
# ============================================================


def _make_geom(nr: int = 16, nz: int = 16, dr: float = 0.01, dz: float = 0.01) -> CylindricalGeometry:
    return CylindricalGeometry(nr, nz, dr, dz)


def _make_state_2d(nr: int, nz: int) -> dict[str, np.ndarray]:
    return {
        "rho": np.ones((nr, nz)) * 1e-4,
        "velocity": np.zeros((3, nr, nz)),
        "pressure": np.ones((nr, nz)) * 1.0,
        "B": np.zeros((3, nr, nz)),
        "Te": np.ones((nr, nz)) * 1e4,
        "Ti": np.ones((nr, nz)) * 1e4,
        "psi": np.zeros((nr, nz)),
    }


# ============================================================
# 1. TestCylindricalPowellZeroDivB
# ============================================================


class TestCylindricalPowellZeroDivB:
    """Powell sources must vanish when div(B) = 0."""

    def test_azimuthal_B_zero_sources(self) -> None:
        """Pure B_theta = 1/r is divergence-free in cylindrical coordinates.

        div(B) = (1/r) * d(r*B_r)/dr + dB_z/dz; B_theta doesn't enter div.
        With B_r = 0 and B_z = 0, div(B) = 0 everywhere.
        """
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        # B_theta = 1/r (div-free: B_r = B_z = 0)
        state["B"][1] = 1.0 / geom.r_2d
        state["velocity"] = np.random.default_rng(7).random((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)

        np.testing.assert_allclose(result["div_B"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dmom_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-12)

    def test_uniform_Bz_zero_sources(self) -> None:
        """Uniform B_z = const has dB_z/dz = 0 and B_r = 0, so div(B) = 0."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        state["B"][2] = 2.5
        state["velocity"] = np.random.default_rng(13).random((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)

        np.testing.assert_allclose(result["div_B"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dmom_powell"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-10)

    def test_random_velocity_div_free_B(self) -> None:
        """Random velocity with div-free B must give zero Powell sources."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        rng = np.random.default_rng(99)
        # Only B_theta set (div-free)
        state["B"][1] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)

        # div(B) = 0 -> all sources zero regardless of velocity
        np.testing.assert_allclose(result["div_B"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dmom_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-12)


# ============================================================
# 2. TestCylindricalPowellNonzeroDivB
# ============================================================


class TestCylindricalPowellNonzeroDivB:
    """Powell sources must be nonzero when div(B) != 0."""

    def test_uniform_Br_nonzero_sources(self) -> None:
        """Uniform B_r has cylindrical div = B_r/r (from d(r*B_r)/dr/r = B_r/r).

        For B_r = const: d(r * B_r)/dr = B_r -> (1/r)*B_r != 0 for r > 0.
        """
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        state["B"][0] = 1.0
        state["B"][2] = 0.0
        state["velocity"][0] = 0.1

        result = powell_source_terms_cylindrical(state, geom)

        # div(B) = B_r/r which is nonzero for r > 0 (interior cells)
        assert np.any(np.abs(result["div_B"][1:-1, 1:-1]) > 1e-6), (
            "Expected nonzero div(B) for uniform B_r"
        )
        # Momentum source must be nonzero
        assert np.any(np.abs(result["dmom_powell"][:, 1:-1, 1:-1]) > 1e-10), (
            "Expected nonzero momentum Powell source"
        )

    def test_linear_Bz_nonzero(self) -> None:
        """B_z = z -> dB_z/dz = 1 -> nonzero sources."""
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)
        state = _make_state_2d(nr, nz)

        z = np.array([(j + 0.5) * dz for j in range(nz)])
        alpha = 100.0
        state["B"][2] = alpha * z[np.newaxis, :]
        state["velocity"][2] = 0.5

        result = powell_source_terms_cylindrical(state, geom)

        # dB_z/dz ~ alpha in interior
        np.testing.assert_allclose(
            result["div_B"][1:-1, 1:-1], alpha, rtol=0.05
        )
        # Induction source dB = -div(B) * v must be nonzero
        assert np.any(np.abs(result["dB_powell"][:, 1:-1, 1:-1]) > 1e-6), (
            "Expected nonzero induction Powell source"
        )

    def test_v_zero_energy_source_zero(self) -> None:
        """When v=0, energy source = -div(B)*(v.B) = 0, but momentum source != 0."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        # Nonzero div(B) via uniform B_r
        state["B"][0] = 1.0
        state["velocity"] = np.zeros((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)

        # Energy source must be zero (v.B = 0)
        np.testing.assert_allclose(result["denergy_powell"], 0.0, atol=1e-20)
        # Induction source must be zero (v = 0)
        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-20)
        # Momentum source nonzero since div(B) and B are nonzero
        assert np.any(np.abs(result["dmom_powell"][0, 1:-1, 1:-1]) > 1e-8)


# ============================================================
# 3. TestCylindricalPowellFormula
# ============================================================


class TestCylindricalPowellFormula:
    """Verify analytical formulas for each Powell source component."""

    def test_momentum_equals_neg_divB_times_B(self) -> None:
        """Verify dmom[d] = -div(B)*B[d] at interior."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        rng = np.random.default_rng(21)
        state["B"][0] = 0.5 * rng.random((nr, nz))
        state["B"][1] = rng.random((nr, nz))
        state["B"][2] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)
        div_B = result["div_B"]

        for d in range(3):
            expected = -div_B * state["B"][d]
            np.testing.assert_allclose(
                result["dmom_powell"][d],
                expected,
                atol=1e-14,
                err_msg=f"Momentum source component {d} mismatch",
            )

    def test_energy_equals_neg_divB_times_vdotB(self) -> None:
        """Verify denergy = -div(B)*(v.B)."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        rng = np.random.default_rng(55)
        state["B"][0] = rng.random((nr, nz)) * 0.5
        state["B"][2] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)
        div_B = result["div_B"]
        v_dot_B = np.sum(state["velocity"] * state["B"], axis=0)

        expected = -div_B * v_dot_B
        np.testing.assert_allclose(result["denergy_powell"], expected, atol=1e-14)

    def test_induction_equals_neg_divB_times_v(self) -> None:
        """Verify dB[d] = -div(B)*v[d]."""
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        rng = np.random.default_rng(77)
        state["B"][0] = rng.random((nr, nz)) * 0.3
        state["B"][2] = rng.random((nr, nz))
        state["velocity"] = rng.random((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)
        div_B = result["div_B"]

        for d in range(3):
            expected = -div_B * state["velocity"][d]
            np.testing.assert_allclose(
                result["dB_powell"][d],
                expected,
                atol=1e-14,
                err_msg=f"Induction source component {d} mismatch",
            )

    def test_cylindrical_vs_cartesian_diverges(self) -> None:
        """Uniform Br has ZERO Cartesian div but NONZERO cylindrical div.

        This verifies that the correct cylindrical formula is used and differs
        from the naive Cartesian formula for radial fields.
        """
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)

        B = np.zeros((3, nr, nz))
        B[0] = 1.0

        # Cylindrical divergence: (1/r) * d(r*B_r)/dr + dB_z/dz
        # For uniform B_r: d(r * 1)/dr = 1 -> (1/r)*1 = 1/r != 0
        div_cyl = geom.div_B_cylindrical(B)

        # Cartesian divergence of same field treats B_r as B_x (constant): dBx/dx = 0
        # Use the 2D arrays directly with Cartesian gradient along axes 0 (r) and 1 (z)
        div_cart = (
            np.gradient(B[0], dr, axis=0)
            + np.gradient(B[1], dr, axis=0) * 0.0  # B_theta const -> 0
            + np.gradient(B[2], dz, axis=1)
        )

        # Cartesian div should be ~0 (all components uniform)
        np.testing.assert_allclose(div_cart[1:-1, 1:-1], 0.0, atol=1e-10)

        # Cylindrical div should be nonzero (1/r contribution)
        assert np.all(np.abs(div_cyl[1:-1, 1:-1]) > 1e-3), (
            "Expected nonzero cylindrical div(B) for uniform B_r"
        )


# ============================================================
# 4. TestPowellEngineIntegration
# ============================================================


class TestPowellEngineIntegration:
    """Engine-level integration tests for Powell sources."""

    def _make_engine(self) -> object:
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        cfg = SimulationConfig(
            grid_shape=(8, 1, 16),
            dx=0.01,
            rho0=1e-4,
            sim_time=1e-6,
            circuit={
                "C": 1e-3,
                "V0": 15000,
                "L0": 33.5e-9,
                "anode_radius": 0.025,
                "cathode_radius": 0.05,
            },
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python", "enable_powell": True},
            diagnostics={"hdf5_filename": ":memory:"},
            boundary={"electrode_bc": False},
        )
        return SimulationEngine(cfg)

    def test_engine_with_powell_runs(self) -> None:
        """SimulationConfig with enable_powell=True, cylindrical, 8x1x16 grid runs 5 steps."""
        engine = self._make_engine()
        for _ in range(5):
            engine.step()

    def test_engine_powell_no_nan(self) -> None:
        """No NaN after 5 steps."""
        engine = self._make_engine()
        for _ in range(5):
            engine.step()

        for key in ("rho", "velocity", "pressure", "B"):
            arr = engine.state[key]
            assert not np.any(np.isnan(arr)), f"NaN found in {key} after 5 Powell steps"

    def test_engine_powell_disabled_default(self) -> None:
        """Default enable_powell is False."""
        from dpf.config import SimulationConfig

        cfg = SimulationConfig(
            grid_shape=(8, 1, 16),
            dx=0.01,
            rho0=1e-4,
            sim_time=1e-6,
            circuit={
                "C": 1e-3,
                "V0": 15000,
                "L0": 33.5e-9,
                "anode_radius": 0.025,
                "cathode_radius": 0.05,
            },
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python"},
            diagnostics={"hdf5_filename": ":memory:"},
            boundary={"electrode_bc": False},
        )
        assert cfg.fluid.enable_powell is False

    def test_engine_powell_density_positive(self) -> None:
        """Density stays positive after 5 Powell steps."""
        engine = self._make_engine()
        for _ in range(5):
            engine.step()

        rho = engine.state["rho"]
        assert np.all(rho > 0.0), f"Non-positive density detected: min={rho.min():.3e}"


# ============================================================
# 5. TestPowellReducesDivB
# ============================================================


class TestPowellReducesDivB:
    """Powell source terms should control div(B) growth."""

    def test_divb_rms_bounded(self) -> None:
        """After applying Powell sources to B (Euler step), div(B) RMS doesn't grow."""
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)
        state = _make_state_2d(nr, nz)

        # Seed a nonzero div(B) via nonzero B_r
        rng = np.random.default_rng(11)
        state["B"][0] = 0.1 * rng.random((nr, nz))
        state["B"][2] = 0.0
        state["velocity"][0] = 0.05

        result = powell_source_terms_cylindrical(state, geom)
        div_B_before = result["div_B"]
        rms_before = float(np.sqrt(np.mean(div_B_before**2)))

        dt = 1e-9
        B_new = state["B"].copy()
        B_new += dt * result["dB_powell"]

        div_B_after = geom.div_B_cylindrical(B_new)
        rms_after = float(np.sqrt(np.mean(div_B_after**2)))

        # Powell sources damp div(B) -- RMS should not increase
        assert rms_after <= rms_before * 1.1, (
            f"div(B) RMS grew: {rms_before:.3e} -> {rms_after:.3e}"
        )

    def test_divb_multi_step_bounded(self) -> None:
        """Over 10 Euler steps, div(B) stays bounded."""
        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        geom = _make_geom(nr, nz, dr, dz)
        state = _make_state_2d(nr, nz)

        rng = np.random.default_rng(33)
        state["B"][0] = 0.2 * rng.random((nr, nz))
        state["velocity"][0] = 0.1

        initial_result = powell_source_terms_cylindrical(state, geom)
        rms_initial = float(np.sqrt(np.mean(initial_result["div_B"] ** 2)))

        dt = 1e-10
        B = state["B"].copy()
        for _ in range(10):
            state_step = dict(state)
            state_step["B"] = B
            result = powell_source_terms_cylindrical(state_step, geom)
            B = B + dt * result["dB_powell"]

        div_B_final = geom.div_B_cylindrical(B)
        rms_final = float(np.sqrt(np.mean(div_B_final**2)))

        # div(B) must not explode over 10 steps
        assert rms_final <= rms_initial * 10.0, (
            f"div(B) RMS blew up: initial={rms_initial:.3e}, final={rms_final:.3e}"
        )

    def test_dB_zero_when_v_zero(self) -> None:
        """dB_powell is exactly zero when velocity is zero.

        dB[d] = -div(B) * v[d]. With v = 0, all induction corrections vanish.
        """
        nr, nz = 16, 16
        geom = _make_geom(nr, nz)
        state = _make_state_2d(nr, nz)

        # Nonzero div(B)
        state["B"][0] = 1.0
        state["velocity"] = np.zeros((3, nr, nz))

        result = powell_source_terms_cylindrical(state, geom)

        np.testing.assert_allclose(result["dB_powell"], 0.0, atol=1e-20)
