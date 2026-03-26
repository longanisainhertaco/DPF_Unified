"""Tests for MLX calibration module (src/dpf/validation/mlx_calibration.py)
and auto_calibrate_mlx() in app_calibrate.py.

Mocking strategy:
- All tests except the @pytest.mark.slow integration test mock
  dpf.validation.mlx_calibration.run_mlx_forward_model so the real MLX
  solver (minutes per eval) never runs in CI.
- The slow integration test calls run_mlx_forward_model directly with a
  tiny grid (8x1x16) for a handful of steps.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial(
    fc: float = 0.70,
    fm: float = 0.08,
    *,
    success: bool = True,
    objective: float = 0.15,
    I_peak_A: float = 1.2e6,
    t_peak_s: float = 6.0e-6,
    nrmse: float = 0.12,
    peak_error: float = 0.05,
    timing_error: float = 0.07,
    wall_time_s: float = 0.5,
    steps: int = 120,
    grid_shape: tuple[int, int, int] = (32, 1, 64),
):
    from dpf.validation.mlx_calibration import MLXTrialResult

    return MLXTrialResult(
        fc=fc,
        fm=fm,
        I_peak_A=I_peak_A,
        t_peak_s=t_peak_s,
        nrmse=nrmse,
        peak_error=peak_error,
        timing_error=timing_error,
        objective=objective,
        wall_time_s=wall_time_s,
        steps=steps,
        success=success,
        grid_shape=grid_shape,
    )


def _make_cal_result(
    best_fc: float = 0.70,
    best_fm: float = 0.08,
    *,
    peak_current_error: float = 0.05,
    timing_error: float = 0.07,
    objective_value: float = 0.15,
    n_evals: int = 25,
    converged: bool = True,
    device_name: str = "pf1000",
):
    from dpf.validation._calibration_data import CalibrationResult

    return CalibrationResult(
        best_fc=best_fc,
        best_fm=best_fm,
        peak_current_error=peak_current_error,
        timing_error=timing_error,
        objective_value=objective_value,
        n_evals=n_evals,
        converged=converged,
        device_name=device_name,
    )


# ---------------------------------------------------------------------------
# 1. MLXTrialResult dataclass instantiation
# ---------------------------------------------------------------------------

class TestMLXTrialResult:
    def test_basic_instantiation(self):
        trial = _make_trial()
        assert trial.fc == pytest.approx(0.70, abs=1e-9)
        assert trial.fm == pytest.approx(0.08, abs=1e-9)
        assert trial.I_peak_A == pytest.approx(1.2e6, rel=1e-6)
        assert trial.t_peak_s == pytest.approx(6.0e-6, rel=1e-6)
        assert trial.nrmse == pytest.approx(0.12, abs=1e-9)
        assert trial.peak_error == pytest.approx(0.05, abs=1e-9)
        assert trial.timing_error == pytest.approx(0.07, abs=1e-9)
        assert trial.objective == pytest.approx(0.15, abs=1e-9)
        assert trial.wall_time_s == pytest.approx(0.5, abs=1e-9)
        assert trial.steps == 120
        assert trial.success is True
        assert trial.grid_shape == (32, 1, 64)

    def test_default_grid_shape(self):
        from dpf.validation.mlx_calibration import MLXTrialResult

        trial = MLXTrialResult(
            fc=0.65,
            fm=0.10,
            I_peak_A=1.0e6,
            t_peak_s=5.0e-6,
            nrmse=0.20,
            peak_error=0.10,
            timing_error=0.08,
            objective=0.30,
            wall_time_s=1.0,
            steps=50,
            success=True,
        )
        assert trial.grid_shape == (32, 1, 64)

    def test_failed_trial(self):
        trial = _make_trial(
            success=False,
            objective=10.0,
            nrmse=10.0,
            peak_error=1.0,
            timing_error=1.0,
            I_peak_A=0.0,
            t_peak_s=0.0,
        )
        assert trial.success is False
        assert trial.objective == pytest.approx(10.0, abs=1e-9)

    def test_grid_shape_tuple_stored(self):
        trial = _make_trial(grid_shape=(8, 1, 16))
        assert trial.grid_shape == (8, 1, 16)
        assert isinstance(trial.grid_shape, tuple)


# ---------------------------------------------------------------------------
# 2. run_mlx_forward_model with mocked engine
# ---------------------------------------------------------------------------

class TestRunMLXForwardModel:
    """Mock the heavy imports inside run_mlx_forward_model so we can test the
    waveform-extraction and objective-computation logic without a real solver."""

    def _build_engine_mock(self, n_steps: int = 50, I_peak: float = 1.2e6):
        """Return a mock that behaves like SimulationEngine."""

        times = [i * 1e-7 for i in range(1, n_steps + 1)]
        # Triangle current profile peaking at mid-point
        currents = []
        mid = n_steps // 2
        for i in range(n_steps):
            if i < mid:
                currents.append(I_peak * (i + 1) / mid)
            else:
                currents.append(I_peak * (n_steps - i) / mid)

        step_results = []
        for i in range(n_steps):
            r = MagicMock()
            r.finished = i == n_steps - 1
            step_results.append(r)

        circuit = MagicMock()
        circuit.current = I_peak / 2  # will be overridden per step

        engine = MagicMock()
        engine.circuit = circuit

        call_count = [0]

        def _step():
            idx = call_count[0]
            engine.time = times[idx]
            engine.circuit.current = currents[idx]
            call_count[0] += 1
            return step_results[idx]

        engine.step.side_effect = _step
        engine.time = 0.0

        return engine, times, currents

    def test_successful_run_returns_trial_result(self, monkeypatch):
        from dpf.validation.mlx_calibration import MLXTrialResult

        engine_mock, times, currents = self._build_engine_mock(n_steps=50)

        mock_config = MagicMock()
        mock_config.sim_time = times[-1] + 1e-10
        mock_config.grid_shape = [32, 1, 64]

        mock_device = MagicMock()
        mock_device.peak_current = 1.2e6
        mock_device.current_rise_time = times[25]
        mock_device.waveform_t = None
        mock_device.waveform_I = None

        monkeypatch.setitem(
            sys.modules,
            "dpf.config",
            MagicMock(SimulationConfig=MagicMock(return_value=mock_config)),
        )
        monkeypatch.setitem(
            sys.modules,
            "dpf.engine",
            MagicMock(SimulationEngine=MagicMock(return_value=engine_mock)),
        )
        monkeypatch.setitem(
            sys.modules,
            "dpf.presets",
            MagicMock(
                get_preset=MagicMock(
                    return_value={
                        "snowplow": {"current_fraction": 0.7, "mass_fraction": 0.08},
                        "fluid": {},
                        "grid_shape": [32, 1, 64],
                        "dx": 0.001,
                        "sim_time": times[-1],
                    }
                )
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "dpf.validation.experimental",
            MagicMock(DEVICES={"pf1000": mock_device}),
        )
        monkeypatch.setitem(
            sys.modules,
            "dpf.validation.experimental_comparison",
            MagicMock(nrmse_peak=MagicMock(return_value=0.14)),
        )
        monkeypatch.setitem(
            sys.modules,
            "app_validation",
            MagicMock(PRESET_TO_DEVICE={"pf1000": "pf1000"}),
        )

        from dpf.validation import mlx_calibration

        trial = mlx_calibration.run_mlx_forward_model(
            fc=0.70,
            fm=0.08,
            preset_name="pf1000",
        )

        assert isinstance(trial, MLXTrialResult)
        assert trial.success is True
        assert trial.fc == pytest.approx(0.70, abs=1e-9)
        assert trial.fm == pytest.approx(0.08, abs=1e-9)
        assert trial.I_peak_A > 0.0
        assert trial.steps > 0
        assert trial.wall_time_s >= 0.0
        assert trial.objective >= 0.0

    def test_engine_exception_returns_failure_trial(self, monkeypatch):

        engine_mock = MagicMock()
        engine_mock.time = 0.0
        engine_mock.step.side_effect = RuntimeError("MLX NaN")

        mock_config = MagicMock()
        mock_config.sim_time = 1e-5
        mock_config.grid_shape = [32, 1, 64]

        monkeypatch.setitem(
            sys.modules,
            "dpf.config",
            MagicMock(SimulationConfig=MagicMock(return_value=mock_config)),
        )
        monkeypatch.setitem(
            sys.modules,
            "dpf.engine",
            MagicMock(SimulationEngine=MagicMock(return_value=engine_mock)),
        )
        monkeypatch.setitem(
            sys.modules,
            "dpf.presets",
            MagicMock(
                get_preset=MagicMock(
                    return_value={
                        "snowplow": {"current_fraction": 0.7, "mass_fraction": 0.08},
                        "fluid": {},
                        "grid_shape": [32, 1, 64],
                        "dx": 0.001,
                        "sim_time": 1e-5,
                    }
                )
            ),
        )
        monkeypatch.setitem(sys.modules, "dpf.validation.experimental", MagicMock())
        monkeypatch.setitem(
            sys.modules,
            "dpf.validation.experimental_comparison",
            MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "app_validation", MagicMock())

        from dpf.validation import mlx_calibration

        trial = mlx_calibration.run_mlx_forward_model(fc=0.70, fm=0.08)

        assert trial.success is False
        assert trial.objective == pytest.approx(10.0, abs=1e-9)
        assert trial.peak_error == pytest.approx(1.0, abs=1e-9)
        assert trial.timing_error == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. coarse_grid_scan with mocked forward model
# ---------------------------------------------------------------------------

class TestCoarseGridScan:
    def _mock_forward(self, monkeypatch, results: list | None = None):
        """Patch run_mlx_forward_model in the mlx_calibration module."""
        call_log: list[dict] = []

        def _fake(fc, fm, preset_name="pf1000", grid_shape=None, **kwargs):
            call_log.append({"fc": fc, "fm": fm})
            if results:
                idx = len(call_log) - 1
                return results[idx % len(results)]
            return _make_trial(fc=fc, fm=fm, objective=abs(fc - 0.70) + abs(fm - 0.08))

        monkeypatch.setattr(
            "dpf.validation.mlx_calibration.run_mlx_forward_model",
            _fake,
        )
        return call_log

    def test_default_grid_size(self, monkeypatch):
        from dpf.validation.mlx_calibration import coarse_grid_scan

        log = self._mock_forward(monkeypatch)
        results = coarse_grid_scan()

        assert len(results) == 25  # 5×5
        assert len(log) == 25

    def test_custom_fc_fm_values(self, monkeypatch):
        from dpf.validation.mlx_calibration import coarse_grid_scan

        self._mock_forward(monkeypatch)
        fc_vals = [0.60, 0.70, 0.80]
        fm_vals = [0.05, 0.15]
        results = coarse_grid_scan(fc_values=fc_vals, fm_values=fm_vals)

        assert len(results) == 6  # 3×2
        fcs = {r.fc for r in results}
        fms = {r.fm for r in results}
        assert fcs == {0.60, 0.70, 0.80}
        assert fms == {0.05, 0.15}

    def test_returns_list_of_mlx_trial_results(self, monkeypatch):
        from dpf.validation.mlx_calibration import MLXTrialResult, coarse_grid_scan

        self._mock_forward(monkeypatch)
        results = coarse_grid_scan(fc_values=[0.70], fm_values=[0.08])

        assert len(results) == 1
        assert isinstance(results[0], MLXTrialResult)

    def test_all_combinations_covered(self, monkeypatch):
        from dpf.validation.mlx_calibration import coarse_grid_scan

        log = self._mock_forward(monkeypatch)
        fc_vals = [0.55, 0.65, 0.75]
        fm_vals = [0.04, 0.12, 0.20]
        coarse_grid_scan(fc_values=fc_vals, fm_values=fm_vals)

        for fc in fc_vals:
            for fm in fm_vals:
                assert any(
                    abs(e["fc"] - fc) < 1e-9 and abs(e["fm"] - fm) < 1e-9
                    for e in log
                ), f"Missing (fc={fc}, fm={fm})"

    def test_failed_trials_included(self, monkeypatch):
        from dpf.validation.mlx_calibration import coarse_grid_scan

        failed = _make_trial(success=False, objective=10.0)
        self._mock_forward(monkeypatch, results=[failed])

        results = coarse_grid_scan(fc_values=[0.70], fm_values=[0.08])
        assert results[0].success is False


# ---------------------------------------------------------------------------
# 4. optuna_optimize with mocked forward model
# ---------------------------------------------------------------------------

class TestOptunaOptimize:
    optuna = pytest.importorskip("optuna")

    def _patch_forward(self, monkeypatch, fixed_objective: float = 0.12):
        def _fake(fc, fm, preset_name="pf1000", grid_shape=None, **kwargs):
            return _make_trial(
                fc=fc,
                fm=fm,
                objective=fixed_objective + abs(fc - 0.70) * 0.01,
                success=True,
            )

        monkeypatch.setattr(
            "dpf.validation.mlx_calibration.run_mlx_forward_model",
            _fake,
        )

    def test_returns_calibration_result_and_trials(self, monkeypatch):
        from dpf.validation._calibration_data import CalibrationResult
        from dpf.validation.mlx_calibration import optuna_optimize

        self._patch_forward(monkeypatch)

        cal, trials = optuna_optimize(
            fc_bounds=(0.60, 0.80),
            fm_bounds=(0.05, 0.20),
            n_trials=5,
            preset_name="pf1000",
            seed=0,
        )

        assert isinstance(cal, CalibrationResult)
        assert isinstance(trials, list)
        assert len(trials) == 5

    def test_best_fc_within_bounds(self, monkeypatch):
        from dpf.validation.mlx_calibration import optuna_optimize

        self._patch_forward(monkeypatch)

        cal, _ = optuna_optimize(
            fc_bounds=(0.60, 0.80),
            fm_bounds=(0.05, 0.20),
            n_trials=5,
            preset_name="pf1000",
            seed=1,
        )

        assert 0.60 <= cal.best_fc <= 0.80
        assert 0.05 <= cal.best_fm <= 0.20

    def test_n_evals_matches_n_trials(self, monkeypatch):
        from dpf.validation.mlx_calibration import optuna_optimize

        self._patch_forward(monkeypatch)

        _, trials = optuna_optimize(
            fc_bounds=(0.60, 0.80),
            fm_bounds=(0.05, 0.20),
            n_trials=8,
            preset_name="pf1000",
            seed=2,
        )

        assert len(trials) == 8

    def test_converged_flag_when_low_objective(self, monkeypatch):
        from dpf.validation.mlx_calibration import optuna_optimize

        # objective < 0.5 → converged=True
        self._patch_forward(monkeypatch, fixed_objective=0.10)

        cal, _ = optuna_optimize(
            fc_bounds=(0.60, 0.80),
            fm_bounds=(0.05, 0.20),
            n_trials=5,
            preset_name="pf1000",
            seed=3,
        )

        assert cal.converged is True

    def test_all_failures_returns_unconverged(self, monkeypatch):
        from dpf.validation.mlx_calibration import optuna_optimize

        def _always_fail(fc, fm, **kwargs):
            return _make_trial(
                fc=fc, fm=fm,
                success=False, objective=10.0,
                peak_error=1.0, timing_error=1.0, nrmse=10.0,
            )

        monkeypatch.setattr(
            "dpf.validation.mlx_calibration.run_mlx_forward_model",
            _always_fail,
        )

        cal, trials = optuna_optimize(
            fc_bounds=(0.60, 0.80),
            fm_bounds=(0.05, 0.20),
            n_trials=4,
            preset_name="pf1000",
            seed=4,
        )

        assert cal.converged is False
        assert len(trials) == 4

    def test_seed_reproducibility(self, monkeypatch):
        from dpf.validation.mlx_calibration import optuna_optimize

        self._patch_forward(monkeypatch)

        cal_a, _ = optuna_optimize(
            fc_bounds=(0.60, 0.80),
            fm_bounds=(0.05, 0.20),
            n_trials=5,
            preset_name="pf1000",
            seed=99,
        )
        cal_b, _ = optuna_optimize(
            fc_bounds=(0.60, 0.80),
            fm_bounds=(0.05, 0.20),
            n_trials=5,
            preset_name="pf1000",
            seed=99,
        )

        assert cal_a.best_fc == pytest.approx(cal_b.best_fc, abs=1e-12)
        assert cal_a.best_fm == pytest.approx(cal_b.best_fm, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. run_calibration_pipeline with mocked forward model
# ---------------------------------------------------------------------------

class TestRunCalibrationPipeline:
    def _patch_forward(self, monkeypatch, objective: float = 0.12):
        def _fake(fc, fm, preset_name="pf1000", grid_shape=None, **kwargs):
            return _make_trial(
                fc=fc, fm=fm,
                objective=objective,
                success=True,
                grid_shape=grid_shape or (32, 1, 64),
            )

        monkeypatch.setattr(
            "dpf.validation.mlx_calibration.run_mlx_forward_model",
            _fake,
        )

    def test_phases_1_and_2_only(self, monkeypatch):
        pytest.importorskip("optuna")
        from dpf.validation.mlx_calibration import MLXCalibrationResult, run_calibration_pipeline

        self._patch_forward(monkeypatch)

        result = run_calibration_pipeline(
            preset_name="pf1000",
            n_optuna_trials=3,
            skip_phase3=True,
            skip_phase4=True,
        )

        assert isinstance(result, MLXCalibrationResult)
        assert result.phases_completed == 2
        assert len(result.trials) > 0

    def test_all_four_phases(self, monkeypatch):
        pytest.importorskip("optuna")
        from dpf.validation.mlx_calibration import run_calibration_pipeline

        self._patch_forward(monkeypatch)

        result = run_calibration_pipeline(
            preset_name="pf1000",
            n_optuna_trials=3,
            skip_phase3=False,
            skip_phase4=False,
        )

        assert result.phases_completed == 4

    def test_best_has_valid_fc_fm(self, monkeypatch):
        pytest.importorskip("optuna")
        from dpf.validation.mlx_calibration import run_calibration_pipeline

        self._patch_forward(monkeypatch)

        result = run_calibration_pipeline(
            preset_name="pf1000",
            n_optuna_trials=3,
            skip_phase3=True,
            skip_phase4=True,
        )

        assert 0.0 < result.best.best_fc < 1.0
        assert 0.0 < result.best.best_fm < 1.0

    def test_total_wall_time_positive(self, monkeypatch):
        pytest.importorskip("optuna")
        from dpf.validation.mlx_calibration import run_calibration_pipeline

        self._patch_forward(monkeypatch)

        result = run_calibration_pipeline(
            preset_name="pf1000",
            n_optuna_trials=3,
            skip_phase3=True,
            skip_phase4=True,
        )

        assert result.total_wall_time_s >= 0.0

    def test_no_successful_phase1_returns_early(self, monkeypatch):
        pytest.importorskip("optuna")
        from dpf.validation.mlx_calibration import run_calibration_pipeline

        def _always_fail(fc, fm, **kwargs):
            return _make_trial(
                fc=fc, fm=fm,
                success=False, objective=10.0,
                peak_error=1.0, timing_error=1.0, nrmse=10.0,
            )

        monkeypatch.setattr(
            "dpf.validation.mlx_calibration.run_mlx_forward_model",
            _always_fail,
        )

        result = run_calibration_pipeline(
            preset_name="pf1000",
            n_optuna_trials=3,
            skip_phase3=True,
            skip_phase4=True,
        )

        assert result.phases_completed == 1
        assert result.best.converged is False

    def test_trials_accumulate_across_phases(self, monkeypatch):
        pytest.importorskip("optuna")
        from dpf.validation.mlx_calibration import run_calibration_pipeline

        self._patch_forward(monkeypatch)

        result = run_calibration_pipeline(
            preset_name="pf1000",
            n_optuna_trials=3,
            skip_phase3=True,
            skip_phase4=True,
        )

        # Phase 1: 5x5=25, Phase 2: 3 Optuna trials → at least 28
        assert len(result.trials) >= 28

    def test_device_name_propagated(self, monkeypatch):
        pytest.importorskip("optuna")
        from dpf.validation.mlx_calibration import run_calibration_pipeline

        self._patch_forward(monkeypatch)

        result = run_calibration_pipeline(
            preset_name="pf1000",
            n_optuna_trials=3,
            skip_phase3=True,
            skip_phase4=True,
        )

        assert result.best.device_name == "pf1000"


# ---------------------------------------------------------------------------
# 6. auto_calibrate_mlx with mocked pipeline
# ---------------------------------------------------------------------------

class TestAutoCalibrateMlx:
    def _patch_pipeline(self, monkeypatch, phases_completed: int = 2):
        from dpf.validation.mlx_calibration import MLXCalibrationResult

        best = _make_cal_result(
            best_fc=0.72,
            best_fm=0.09,
            peak_current_error=0.04,
            timing_error=0.06,
            objective_value=0.13,
            n_evals=28,
            converged=True,
            device_name="pf1000",
        )
        trials = [
            _make_trial(fc=0.72, fm=0.09, objective=0.13, I_peak_A=1.21e6, t_peak_s=6.1e-6),
            _make_trial(fc=0.65, fm=0.10, objective=0.20, success=False),
        ]
        fake_result = MLXCalibrationResult(
            best=best,
            trials=trials,
            phases_completed=phases_completed,
            total_wall_time_s=12.3,
        )

        monkeypatch.setattr(
            "dpf.validation.mlx_calibration.run_calibration_pipeline",
            lambda *a, **kw: fake_result,
        )
        return fake_result

    def test_returns_dict(self, monkeypatch):
        import sys

        sys.path.insert(0, "/Users/anthonyzamora/dpf-unified")
        self._patch_pipeline(monkeypatch)

        from app_calibrate import auto_calibrate_mlx

        out = auto_calibrate_mlx(preset_name="pf1000", n_trials=3)
        assert isinstance(out, dict)

    def test_output_keys_present(self, monkeypatch):
        import sys

        sys.path.insert(0, "/Users/anthonyzamora/dpf-unified")
        self._patch_pipeline(monkeypatch)

        from app_calibrate import auto_calibrate_mlx

        out = auto_calibrate_mlx(preset_name="pf1000", n_trials=3)

        required_keys = {
            "backend", "best_fc", "best_fm", "I_peak_error",
            "t_peak_error", "objective", "n_evals", "converged",
            "device_name", "preset", "phases_completed", "wall_time_s",
            "n_trials_total",
        }
        assert required_keys.issubset(out.keys())

    def test_backend_is_mlx(self, monkeypatch):
        import sys

        sys.path.insert(0, "/Users/anthonyzamora/dpf-unified")
        self._patch_pipeline(monkeypatch)

        from app_calibrate import auto_calibrate_mlx

        out = auto_calibrate_mlx()
        assert out["backend"] == "mlx"

    def test_best_fc_fm_propagated(self, monkeypatch):
        import sys

        sys.path.insert(0, "/Users/anthonyzamora/dpf-unified")
        self._patch_pipeline(monkeypatch)

        from app_calibrate import auto_calibrate_mlx

        out = auto_calibrate_mlx(preset_name="pf1000", n_trials=3)
        assert out["best_fc"] == pytest.approx(0.72, abs=1e-9)
        assert out["best_fm"] == pytest.approx(0.09, abs=1e-9)

    def test_best_nrmse_present_when_successful_trial_exists(self, monkeypatch):
        import sys

        sys.path.insert(0, "/Users/anthonyzamora/dpf-unified")
        self._patch_pipeline(monkeypatch)

        from app_calibrate import auto_calibrate_mlx

        out = auto_calibrate_mlx(preset_name="pf1000")
        assert "best_nrmse" in out
        assert "best_I_peak_MA" in out
        assert "best_t_peak_us" in out
        assert out["best_I_peak_MA"] == pytest.approx(1.21, rel=1e-3)
        assert out["best_t_peak_us"] == pytest.approx(6.1, rel=1e-3)

    def test_phases_param_controls_skip_flags(self, monkeypatch):
        """Verify that phases=2 passes skip_phase3=True, skip_phase4=True."""
        import sys

        sys.path.insert(0, "/Users/anthonyzamora/dpf-unified")

        calls: list[dict] = []

        def _mock_pipeline(preset_name, n_optuna_trials, skip_phase3, skip_phase4):
            calls.append(
                {"skip_phase3": skip_phase3, "skip_phase4": skip_phase4}
            )
            from dpf.validation.mlx_calibration import MLXCalibrationResult

            return MLXCalibrationResult(
                best=_make_cal_result(),
                trials=[_make_trial()],
                phases_completed=2,
                total_wall_time_s=1.0,
            )

        monkeypatch.setattr(
            "dpf.validation.mlx_calibration.run_calibration_pipeline",
            _mock_pipeline,
        )

        from app_calibrate import auto_calibrate_mlx

        auto_calibrate_mlx(phases=2)
        assert calls[0]["skip_phase3"] is True
        assert calls[0]["skip_phase4"] is True

        auto_calibrate_mlx(phases=4)
        assert calls[1]["skip_phase3"] is False
        assert calls[1]["skip_phase4"] is False

    def test_phases_completed_in_output(self, monkeypatch):
        import sys

        sys.path.insert(0, "/Users/anthonyzamora/dpf-unified")
        self._patch_pipeline(monkeypatch, phases_completed=4)

        from app_calibrate import auto_calibrate_mlx

        out = auto_calibrate_mlx(phases=4)
        assert out["phases_completed"] == 4


# ---------------------------------------------------------------------------
# 7. @pytest.mark.slow integration test
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRunMLXForwardModelIntegration:
    """Real invocation of run_mlx_forward_model with a tiny grid.

    Skipped automatically unless the MLX backend and related imports are
    available. Runs with grid_shape=(8,1,16) for just a few nanoseconds of
    simulated time so it finishes in under 60 seconds on M3 Pro.
    """

    def test_tiny_grid_returns_valid_trial(self):
        mlx = pytest.importorskip("mlx.core")  # noqa: F841 — skip if MLX missing

        from dpf.validation.mlx_calibration import MLXTrialResult, run_mlx_forward_model

        trial = run_mlx_forward_model(
            fc=0.70,
            fm=0.08,
            preset_name="pf1000",
            grid_shape=(8, 1, 16),
            sim_time=1e-7,
        )

        assert isinstance(trial, MLXTrialResult)
        assert trial.grid_shape == (8, 1, 16)
        assert trial.wall_time_s >= 0.0
        assert trial.steps >= 0

        if trial.success:
            assert trial.I_peak_A >= 0.0
            assert trial.t_peak_s >= 0.0
            assert trial.objective >= 0.0
            assert trial.peak_error >= 0.0
            assert trial.timing_error >= 0.0
            assert trial.nrmse >= 0.0
        else:
            # Failure is acceptable on tiny grid — just verify sentinel values
            assert trial.objective == pytest.approx(10.0, abs=1e-9)


class TestParallelOptuna:
    """Tests for parallel_optuna_optimize structure and API."""

    def test_worker_eval_is_picklable(self):
        """_worker_eval must be module-level for ProcessPoolExecutor."""
        import pickle

        from dpf.validation.mlx_calibration import _worker_eval

        # Module-level functions are picklable
        pickled = pickle.dumps(_worker_eval)
        restored = pickle.loads(pickled)
        assert callable(restored)

    def test_parallel_optuna_function_exists(self):
        """parallel_optuna_optimize is importable and has correct signature."""
        import inspect

        from dpf.validation.mlx_calibration import parallel_optuna_optimize

        sig = inspect.signature(parallel_optuna_optimize)
        assert "n_workers" in sig.parameters
        assert "n_trials" in sig.parameters
        assert "fc_bounds" in sig.parameters
        assert "fm_bounds" in sig.parameters
        assert sig.parameters["n_workers"].default == 3

    def test_constant_liar_sampler(self):
        """Optuna TPESampler accepts constant_liar parameter."""
        import optuna

        sampler = optuna.samplers.TPESampler(seed=42, constant_liar=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        # ask/tell API works
        trial = study.ask()
        fc = trial.suggest_float("fc", 0.5, 0.85)
        fm = trial.suggest_float("fm", 0.03, 0.30)
        study.tell(trial, abs(fc - 0.7) + abs(fm - 0.08))
        assert len(study.trials) == 1
