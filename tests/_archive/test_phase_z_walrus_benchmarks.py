"""Phase Z: WALRUS surrogate validation against analytical benchmarks.

Tests the ``benchmark_validation`` module that bridges Bennett equilibrium
and magnetized Noh exact solutions with the WALRUS surrogate validation
pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.ai.benchmark_validation import (
    create_bennett_trajectory,
    create_noh_trajectory,
    validate_surrogate_against_bennett,
    validate_surrogate_against_noh,
)

# ---------------------------------------------------------------------------
# Minimal placeholder surrogate (no torch dependency)
# ---------------------------------------------------------------------------

class _PlaceholderSurrogate:
    """Lightweight stand-in for DPFSurrogate that returns last input state."""

    def __init__(self, history_length: int = 4) -> None:
        self.history_length = history_length

    def predict_next_step(
        self, history: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in history[-1].items()}

    @property
    def is_loaded(self) -> bool:
        return True

    def validate_against_physics(
        self,
        trajectory: list[dict[str, np.ndarray]],
        fields: list[str] | None = None,
    ) -> dict[str, object]:
        """Replicate the core logic from DPFSurrogate.validate_against_physics."""
        hl = self.history_length
        if len(trajectory) < hl + 1:
            raise ValueError(
                f"Trajectory too short: need {hl + 1}, got {len(trajectory)}"
            )

        if fields is None:
            fields = ["rho", "Te", "Ti", "pressure", "psi", "B", "velocity"]

        per_field_l2: dict[str, list[float]] = {f: [] for f in fields}
        diverging_steps: list[int] = []
        all_l2: list[float] = []

        for i in range(hl, len(trajectory)):
            history = trajectory[i - hl:i]
            actual = trajectory[i]
            predicted = self.predict_next_step(history)

            step_l2_values: list[float] = []
            for field in fields:
                if field not in actual or field not in predicted:
                    continue
                pred_arr = predicted[field]
                actual_arr = actual[field]
                if pred_arr.shape != actual_arr.shape:
                    continue
                diff_norm = float(np.linalg.norm(pred_arr - actual_arr))
                actual_norm = max(float(np.linalg.norm(actual_arr)), 1e-10)
                l2 = diff_norm / actual_norm
                per_field_l2[field].append(l2)
                step_l2_values.append(l2)

            if step_l2_values:
                step_mean = float(np.mean(step_l2_values))
                all_l2.append(step_mean)
                if step_mean > 0.3:
                    diverging_steps.append(i)

        return {
            "n_steps": len(all_l2),
            "per_field_l2": per_field_l2,
            "mean_l2": float(np.mean(all_l2)) if all_l2 else 0.0,
            "max_l2": float(np.max(all_l2)) if all_l2 else 0.0,
            "diverging_steps": diverging_steps,
        }


# ---------------------------------------------------------------------------
# Bennett trajectory tests
# ---------------------------------------------------------------------------


class TestBennettTrajectory:
    """Tests for create_bennett_trajectory."""

    def test_returns_correct_length(self) -> None:
        traj = create_bennett_trajectory(n_steps=8, nr=16, nz=4)
        assert len(traj) == 8

    def test_all_states_have_required_keys(self) -> None:
        traj = create_bennett_trajectory(n_steps=3, nr=16, nz=4)
        required = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        for state in traj:
            assert required <= set(state.keys())

    def test_all_states_identical(self) -> None:
        """Bennett equilibrium is stationary — all timesteps must be equal."""
        traj = create_bennett_trajectory(n_steps=6, nr=16, nz=4)
        for i in range(1, len(traj)):
            for key in traj[0]:
                np.testing.assert_array_equal(
                    traj[0][key], traj[i][key],
                    err_msg=f"State {i} differs from state 0 for field '{key}'",
                )

    def test_states_are_deep_copies(self) -> None:
        """Mutating one state must not affect others."""
        traj = create_bennett_trajectory(n_steps=3, nr=16, nz=4)
        original_rho = traj[1]["rho"].copy()
        traj[0]["rho"][:] = 999.0
        np.testing.assert_array_equal(traj[1]["rho"], original_rho)

    def test_density_positive(self) -> None:
        traj = create_bennett_trajectory(n_steps=2, nr=16, nz=4)
        assert np.all(traj[0]["rho"] > 0)

    def test_pressure_positive(self) -> None:
        traj = create_bennett_trajectory(n_steps=2, nr=16, nz=4)
        assert np.all(traj[0]["pressure"] > 0)

    def test_velocity_zero(self) -> None:
        """Bennett equilibrium has no bulk flow."""
        traj = create_bennett_trajectory(n_steps=2, nr=16, nz=4)
        np.testing.assert_array_equal(traj[0]["velocity"], 0.0)

    def test_shapes_cylindrical(self) -> None:
        nr, nz = 32, 8
        traj = create_bennett_trajectory(n_steps=2, nr=nr, nz=nz)
        state = traj[0]
        assert state["rho"].shape == (nr, 1, nz)
        assert state["pressure"].shape == (nr, 1, nz)
        assert state["velocity"].shape == (3, nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)


# ---------------------------------------------------------------------------
# Noh trajectory tests
# ---------------------------------------------------------------------------


class TestNohTrajectory:
    """Tests for create_noh_trajectory."""

    def test_returns_correct_length(self) -> None:
        traj = create_noh_trajectory(n_steps=8, nr=16, nz=4)
        assert len(traj) == 8

    def test_all_states_have_required_keys(self) -> None:
        traj = create_noh_trajectory(n_steps=3, nr=16, nz=4)
        required = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        for state in traj:
            assert required <= set(state.keys())

    def test_states_evolve(self) -> None:
        """Noh solution is time-dependent — states must differ."""
        traj = create_noh_trajectory(
            n_steps=5, nr=32, nz=4, B_0=0.0, t_start=0.5, t_end=2.0,
        )
        # Density at different times should differ (shock moves)
        rho_first = traj[0]["rho"]
        rho_last = traj[-1]["rho"]
        assert not np.allclose(rho_first, rho_last), (
            "Noh trajectory states should evolve over time"
        )

    def test_density_positive(self) -> None:
        traj = create_noh_trajectory(n_steps=3, nr=16, nz=4)
        for state in traj:
            assert np.all(state["rho"] > 0)

    def test_inflow_velocity_negative(self) -> None:
        """Upstream radial velocity should be negative (inward flow)."""
        traj = create_noh_trajectory(
            n_steps=2, nr=32, nz=4, r_max=1.0, B_0=0.0,
            t_start=0.5, t_end=1.0,
        )
        # Outer cells are upstream and should have v_r = -V_0
        vr = traj[0]["velocity"][0, :, 0, 0]
        assert np.any(vr < 0), "Expected negative (inward) radial velocity upstream"

    def test_shapes_cylindrical(self) -> None:
        nr, nz = 32, 8
        traj = create_noh_trajectory(n_steps=2, nr=nr, nz=nz)
        state = traj[0]
        assert state["rho"].shape == (nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)

    def test_unmagnetized_noh(self) -> None:
        """B_0=0 should give B=0 everywhere."""
        traj = create_noh_trajectory(n_steps=2, nr=16, nz=4, B_0=0.0)
        np.testing.assert_array_equal(traj[0]["B"], 0.0)


# ---------------------------------------------------------------------------
# Validation wrapper tests
# ---------------------------------------------------------------------------


class TestValidateSurrogateAgainstBennett:
    """Tests for validate_surrogate_against_bennett with placeholder model."""

    def test_report_structure(self) -> None:
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert "n_steps" in report
        assert "per_field_l2" in report
        assert "mean_l2" in report
        assert "max_l2" in report
        assert "diverging_steps" in report

    def test_placeholder_bennett_l2_zero(self) -> None:
        """Placeholder returns last state = next state for stationary Bennett."""
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert report["mean_l2"] == pytest.approx(0.0, abs=1e-15)
        assert report["max_l2"] == pytest.approx(0.0, abs=1e-15)

    def test_no_diverging_steps_for_bennett(self) -> None:
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert report["diverging_steps"] == []

    def test_n_steps_correct(self) -> None:
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=10, nr=16, nz=4,
        )
        # With 10 states and history_length=4, we validate steps 4..9 = 6 steps
        assert report["n_steps"] == 6

    def test_specific_fields(self) -> None:
        surrogate = _PlaceholderSurrogate(history_length=2)
        report = validate_surrogate_against_bennett(
            surrogate, n_steps=6, nr=16, nz=4,
            fields=["rho", "pressure"],
        )
        assert "rho" in report["per_field_l2"]
        assert "pressure" in report["per_field_l2"]


class TestValidateSurrogateAgainstNoh:
    """Tests for validate_surrogate_against_noh with placeholder model."""

    def test_report_structure(self) -> None:
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_noh(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        assert "n_steps" in report
        assert "per_field_l2" in report
        assert "mean_l2" in report
        assert "max_l2" in report
        assert "diverging_steps" in report

    def test_placeholder_noh_l2_nonzero(self) -> None:
        """Placeholder returns last state, but Noh evolves — L2 should be > 0."""
        surrogate = _PlaceholderSurrogate(history_length=4)
        report = validate_surrogate_against_noh(
            surrogate, n_steps=8, nr=32, nz=4,
            B_0=0.0, t_start=0.5, t_end=2.0,
        )
        assert report["mean_l2"] > 0.0, (
            "Noh trajectory evolves — placeholder should have nonzero error"
        )

    def test_n_steps_correct(self) -> None:
        surrogate = _PlaceholderSurrogate(history_length=3)
        report = validate_surrogate_against_noh(
            surrogate, n_steps=8, nr=16, nz=4,
        )
        # With 8 states and history_length=3, we validate steps 3..7 = 5 steps
        assert report["n_steps"] == 5

    def test_ensures_minimum_trajectory_length(self) -> None:
        """Even if n_steps < history_length+2, wrapper should pad up."""
        surrogate = _PlaceholderSurrogate(history_length=4)
        # n_steps=3 is too short, wrapper should auto-increase
        report = validate_surrogate_against_noh(
            surrogate, n_steps=3, nr=16, nz=4,
        )
        assert report["n_steps"] >= 1
