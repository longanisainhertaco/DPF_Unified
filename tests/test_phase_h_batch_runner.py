from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from dpf.ai.batch_runner import BatchResult, BatchRunner, ParameterRange
from dpf.config import SimulationConfig


@pytest.fixture
def base_config():
    """Create base simulation config for testing."""
    return SimulationConfig(
        grid_shape=[8, 8, 8],
        dx=1e-2,
        sim_time=1e-6,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
    )


@pytest.fixture
def parameter_ranges():
    """Create sample parameter ranges for testing."""
    return [
        ParameterRange(name="circuit.V0", low=500.0, high=2000.0, log_scale=False),
        ParameterRange(name="circuit.C", low=1e-7, high=1e-5, log_scale=True),
        ParameterRange(name="dx", low=0.005, high=0.02, log_scale=False),
    ]


class TestParameterRange:
    """Test ParameterRange dataclass."""

    def test_parameter_range_creation(self):
        """Test ParameterRange dataclass fields."""
        param = ParameterRange(name="test_param", low=1.0, high=10.0, log_scale=False)
        assert param.name == "test_param"
        assert param.low == 1.0
        assert param.high == 10.0
        assert param.log_scale is False

    def test_parameter_range_default_log_scale(self):
        """Test ParameterRange default log_scale is False."""
        param = ParameterRange(name="test", low=1.0, high=10.0)
        assert param.log_scale is False

    def test_parameter_range_with_log_scale(self):
        """Test ParameterRange with log_scale enabled."""
        param = ParameterRange(name="test", low=1e-3, high=1e3, log_scale=True)
        assert param.log_scale is True


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_batch_result_defaults(self):
        """Test BatchResult default values."""
        result = BatchResult()
        assert result.n_total == 0
        assert result.n_success == 0
        assert result.n_failed == 0
        assert result.output_dir == ""
        assert result.failed_configs == []

    def test_batch_result_creation(self):
        """Test BatchResult with explicit values."""
        result = BatchResult(
            n_total=100,
            n_success=95,
            n_failed=5,
            output_dir="/tmp/test",
            failed_configs=[(2, "error1"), (5, "error2")],
        )
        assert result.n_total == 100
        assert result.n_success == 95
        assert result.n_failed == 5
        assert result.output_dir == "/tmp/test"
        assert len(result.failed_configs) == 2

    def test_batch_result_n_failed_computation(self):
        """Test n_failed is computed correctly from counts."""
        result = BatchResult(n_total=100, n_success=92, n_failed=8)
        assert result.n_failed == 8
        assert result.n_total == result.n_success + result.n_failed

    def test_batch_result_failed_configs_list(self):
        """Test failed_configs stores index and error message tuples."""
        failed = [(0, "timeout"), (3, "convergence failure"), (7, "NaN detected")]
        result = BatchResult(failed_configs=failed)
        assert len(result.failed_configs) == 3
        assert result.failed_configs[0] == (0, "timeout")
        assert result.failed_configs[1] == (3, "convergence failure")
        assert result.failed_configs[2] == (7, "NaN detected")


class TestBatchRunnerInit:
    """Test BatchRunner initialization."""

    def test_batch_runner_init_stores_parameters(self, base_config, parameter_ranges):
        """Test BatchRunner __init__ stores all parameters."""
        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            n_samples=50,
            output_dir="test_output",
            workers=2,
            field_interval=5,
        )

        assert runner.base_config is base_config
        assert runner.parameter_ranges is parameter_ranges
        assert runner.n_samples == 50
        assert runner.output_dir == Path("test_output")
        assert runner.workers == 2
        assert runner.field_interval == 5

    def test_batch_runner_init_default_values(self, base_config, parameter_ranges):
        """Test BatchRunner default values."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=parameter_ranges)

        assert runner.n_samples == 100
        assert runner.output_dir == Path("training_data")
        assert runner.workers == 4
        assert runner.field_interval == 10


class TestLatinHypercube:
    """Test Latin Hypercube sampling."""

    def test_latin_hypercube_returns_correct_shape(self):
        """Test _latin_hypercube returns array with correct shape."""
        samples = BatchRunner._latin_hypercube(n_samples=50, n_dims=3)
        assert samples.shape == (50, 3)

    def test_latin_hypercube_values_in_range(self):
        """Test _latin_hypercube values are in [0, 1]."""
        samples = BatchRunner._latin_hypercube(n_samples=100, n_dims=5)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_latin_hypercube_reproducible_with_seed(self):
        """Test _latin_hypercube with same seed produces identical results."""
        samples1 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=123)
        samples2 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=123)
        np.testing.assert_array_equal(samples1, samples2)

    def test_latin_hypercube_different_seeds_differ(self):
        """Test _latin_hypercube with different seeds produces different results."""
        samples1 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=1)
        samples2 = BatchRunner._latin_hypercube(n_samples=20, n_dims=3, seed=2)
        assert not np.allclose(samples1, samples2)

    def test_latin_hypercube_fallback_without_scipy(self, monkeypatch):
        """Test _latin_hypercube fallback when scipy not available."""

        def mock_import_error(*args, **kwargs):
            raise ImportError("No module named 'scipy'")

        # Monkeypatch the import to fail
        import builtins

        original_import = builtins.__import__

        def custom_import(name, *args, **kwargs):
            if "scipy" in name:
                raise ImportError("No module named 'scipy'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", custom_import)

        # Should still work with fallback
        samples = BatchRunner._latin_hypercube(n_samples=30, n_dims=2, seed=42)
        assert samples.shape == (30, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)


class TestGenerateSamples:
    """Test sample generation."""

    def test_generate_samples_returns_correct_count(self, base_config, parameter_ranges):
        """Test generate_samples returns correct number of samples."""
        runner = BatchRunner(
            base_config=base_config, parameter_ranges=parameter_ranges, n_samples=25
        )
        samples = runner.generate_samples()
        assert len(samples) == 25

    def test_generate_samples_linear_scaling_maps_to_range(self, base_config):
        """Test generate_samples linear scaling maps to correct range."""
        param_ranges = [ParameterRange(name="test_param", low=10.0, high=20.0, log_scale=False)]
        runner = BatchRunner(base_config=base_config, parameter_ranges=param_ranges, n_samples=100)
        samples = runner.generate_samples()

        values = [s["test_param"] for s in samples]
        assert all(10.0 <= v <= 20.0 for v in values)
        # Check we span the range reasonably
        assert min(values) < 11.0
        assert max(values) > 19.0

    def test_generate_samples_log_scale_mapping(self, base_config):
        """Test generate_samples with log_scale enabled."""
        param_ranges = [ParameterRange(name="test_log", low=1e-6, high=1e-2, log_scale=True)]
        runner = BatchRunner(base_config=base_config, parameter_ranges=param_ranges, n_samples=100)
        samples = runner.generate_samples()

        values = [s["test_log"] for s in samples]
        assert all(1e-6 <= v <= 1e-2 for v in values)

        # Check log-uniform distribution: log(values) should span range
        log_values = np.log10(values)
        assert min(log_values) < -5.5
        assert max(log_values) > -2.5

    def test_generate_samples_each_sample_has_correct_keys(self, base_config, parameter_ranges):
        """Test each sample dictionary has correct parameter keys."""
        runner = BatchRunner(
            base_config=base_config, parameter_ranges=parameter_ranges, n_samples=10
        )
        samples = runner.generate_samples()

        expected_keys = {"circuit.V0", "circuit.C", "dx"}
        for sample in samples:
            assert set(sample.keys()) == expected_keys


class TestBuildConfig:
    """Test configuration building."""

    def test_build_config_applies_top_level_parameter(self, base_config):
        """Test build_config applies top-level parameter overrides."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=[])
        params = {"dx": 0.015}

        config = runner.build_config(params)
        assert config.dx == pytest.approx(0.015)
        # Other fields unchanged
        assert config.grid_shape == [8, 8, 8]

    def test_build_config_applies_nested_dot_notation(self, base_config):
        """Test build_config applies nested dot-notation overrides."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=[])
        params = {"circuit.V0": 1500.0, "circuit.C": 5e-6}

        config = runner.build_config(params)
        assert pytest.approx(1500.0) == config.circuit.V0
        assert pytest.approx(5e-6) == config.circuit.C
        # Other circuit fields unchanged
        assert pytest.approx(1e-7) == config.circuit.L0

    def test_build_config_returns_valid_simulation_config(self, base_config):
        """Test build_config returns valid SimulationConfig."""
        runner = BatchRunner(base_config=base_config, parameter_ranges=[])
        params = {"circuit.V0": 1200.0, "dx": 0.01}

        config = runner.build_config(params)
        assert isinstance(config, SimulationConfig)
        assert config.grid_shape == [8, 8, 8]
        assert pytest.approx(0.01) == config.dx
        assert pytest.approx(1200.0) == config.circuit.V0


class TestRunSingle:
    """Test single simulation runs."""

    @staticmethod
    def _make_mock_engine():
        """Create a mock engine that supports step-by-step execution.

        The batch runner calls engine.step() in a loop (checking result.finished),
        captures field snapshots via engine.get_field_snapshot(), and reads
        engine.circuit.current, engine.circuit.voltage, engine.time.
        """
        mock_engine = MagicMock()
        # step() returns a StepResult-like object; finished=True on first call
        step_result = MagicMock()
        step_result.finished = True
        mock_engine.step.return_value = step_result
        # get_field_snapshot returns a minimal state dict
        mock_engine.get_field_snapshot.return_value = {
            "rho": np.zeros((8, 8, 8)),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.zeros((8, 8, 8)),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.zeros((8, 8, 8)),
            "Ti": np.zeros((8, 8, 8)),
        }
        mock_engine.circuit = MagicMock()
        mock_engine.circuit.current = 0.0
        mock_engine.circuit.voltage = 1000.0
        mock_engine.time = 1e-7
        mock_engine.diagnostics = MagicMock()
        return mock_engine

    def test_run_single_returns_success_tuple(self, base_config, parameter_ranges, monkeypatch, tmp_path):
        """Test run_single returns (idx, None) on success.

        The batch runner runs step-by-step, capturing field snapshots directly
        from the engine state, then exports to Well HDF5 format.
        """
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        # Mock WellExporter
        mock_exporter_class = MagicMock()
        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]  # Non-empty to check logging
        mock_exporter_class.return_value = mock_exporter
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            n_samples=5,
            output_dir=tmp_path / "test_single",
        )
        params = {"circuit.V0": 1000.0}

        idx, error = runner.run_single(0, params)
        assert idx == 0
        assert error is None
        # Verify step-by-step execution was used
        mock_engine.step.assert_called()
        # Verify field snapshots were captured
        mock_engine.get_field_snapshot.assert_called()
        # Verify exporter was finalized
        mock_exporter.finalize.assert_called_once()

    def test_run_single_returns_error_on_failure(self, base_config, parameter_ranges, monkeypatch):
        """Test run_single returns (idx, error_msg) on failure."""
        # Mock SimulationEngine to raise exception in step()
        mock_engine = self._make_mock_engine()
        mock_engine.step.side_effect = RuntimeError("Simulation diverged")
        mock_engine_class = MagicMock(return_value=mock_engine)

        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        runner = BatchRunner(
            base_config=base_config, parameter_ranges=parameter_ranges, n_samples=5
        )
        params = {"circuit.V0": 1000.0}

        idx, error = runner.run_single(0, params)
        assert idx == 0
        assert error is not None
        assert "RuntimeError" in error
        assert "Simulation diverged" in error


class TestRun:
    """Test batch run execution."""

    @staticmethod
    def _make_mock_engine():
        """Create a mock engine for step-by-step execution in batch runs."""
        mock_engine = MagicMock()
        step_result = MagicMock()
        step_result.finished = True
        mock_engine.step.return_value = step_result
        mock_engine.get_field_snapshot.return_value = {
            "rho": np.zeros((8, 8, 8)),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.zeros((8, 8, 8)),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.zeros((8, 8, 8)),
            "Ti": np.zeros((8, 8, 8)),
        }
        mock_engine.circuit = MagicMock()
        mock_engine.circuit.current = 0.0
        mock_engine.circuit.voltage = 1000.0
        mock_engine.time = 1e-7
        mock_engine.diagnostics = MagicMock()
        return mock_engine

    def test_run_with_workers_1_sequential(
        self, base_config, parameter_ranges, monkeypatch, tmp_path
    ):
        """Test run with workers=1 runs sequentially."""
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        # Mock WellExporter
        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            n_samples=3,
            workers=1,
            output_dir=tmp_path / "sequential",
        )

        result = runner.run()

        # Should have instantiated engine 3 times
        assert mock_engine_class.call_count == 3
        assert result.n_total == 3

    def test_run_returns_batch_result_with_correct_counts(
        self, base_config, parameter_ranges, monkeypatch, tmp_path
    ):
        """Test run returns BatchResult with correct counts."""
        # Mock engine where the 2nd instantiation fails during step()
        call_count = [0]

        def make_engine(*args, **kwargs):
            call_count[0] += 1
            engine = self._make_mock_engine()
            if call_count[0] == 2:
                engine.step.side_effect = ValueError("Simulation 2 failed")
            return engine

        monkeypatch.setattr("dpf.engine.SimulationEngine", make_engine)

        # Mock WellExporter
        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            n_samples=5,
            workers=1,
            output_dir=tmp_path / "batch_test",
        )

        result = runner.run()

        assert result.n_total == 5
        assert result.n_success == 4
        assert result.n_failed == 1
        assert len(result.failed_configs) == 1

    def test_run_progress_callback_is_called(
        self, base_config, parameter_ranges, monkeypatch, tmp_path
    ):
        """Test run calls progress_callback after each simulation."""
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        # Mock WellExporter
        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            n_samples=3,
            workers=1,
            output_dir=tmp_path / "progress_test",
        )

        # Track progress callback calls
        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        runner.run(progress_callback=progress_callback)

        # Should have been called 3 times
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[1] == (2, 3)
        assert progress_calls[2] == (3, 3)

    def test_run_creates_output_directory(self, base_config, parameter_ranges, monkeypatch, tmp_path):
        """Test run creates output directory."""
        mock_engine = self._make_mock_engine()
        mock_engine_class = MagicMock(return_value=mock_engine)
        monkeypatch.setattr("dpf.engine.SimulationEngine", mock_engine_class)

        # Mock WellExporter
        mock_exporter = MagicMock()
        mock_exporter._snapshots = [{}]
        mock_exporter_class = MagicMock(return_value=mock_exporter)
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", mock_exporter_class)

        output_dir = tmp_path / "test_output_dir"
        assert not output_dir.exists()

        runner = BatchRunner(
            base_config=base_config,
            parameter_ranges=parameter_ranges,
            n_samples=1,
            workers=1,
            output_dir=output_dir,
        )

        runner.run()

        assert output_dir.exists()
        assert output_dir.is_dir()
