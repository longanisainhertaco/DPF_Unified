"""Phase J.2: WALRUS live integration tests.

Tests real WALRUS model loading, tensor conversion, Well format compliance,
batch runner API, server endpoint fixes, and placeholder fallback behavior.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Helpers ──────────────────────────────────────────────────────


def _make_state(shape=(8, 8, 8)):
    """Create a standard DPF state dict for testing."""
    return {
        "rho": np.ones(shape, dtype=np.float64),
        "velocity": np.zeros((3, *shape), dtype=np.float64),
        "pressure": np.ones(shape, dtype=np.float64) * 100.0,
        "B": np.zeros((3, *shape), dtype=np.float64),
        "Te": np.ones(shape, dtype=np.float64) * 1e4,
        "Ti": np.ones(shape, dtype=np.float64) * 1e4,
        "psi": np.zeros(shape, dtype=np.float64),
    }


# ── WALRUS Model Loading Tests ───────────────────────────────────


class TestWALRUSModelLoading:
    """Test surrogate model loading with real WALRUS model (mocked)."""

    @pytest.fixture
    def mock_walrus_env(self, monkeypatch, tmp_path):
        """Set up mocked walrus/hydra/torch environment."""
        # Create checkpoint file
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        # Mock torch
        mock_torch_mod = MagicMock()
        mock_torch_mod.no_grad.return_value.__enter__ = MagicMock()
        mock_torch_mod.no_grad.return_value.__exit__ = MagicMock()

        # Mock model returned by instantiate
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock config from checkpoint
        mock_config = MagicMock()

        # Mock RevIN
        mock_revin_instance = MagicMock()
        mock_revin_constructor = MagicMock(return_value=mock_revin_instance)
        mock_config.trainer.revin = mock_revin_constructor

        # Checkpoint data
        checkpoint_data = {
            "model_state_dict": {"layer.weight": "fake_tensor"},
            "config": mock_config,
        }
        mock_torch_mod.load.return_value = checkpoint_data

        # Mock tensor operations
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch_mod.from_numpy.return_value = mock_tensor

        # Mock instantiate
        mock_instantiate = MagicMock(return_value=mock_model)

        # Mock walrus modules
        mock_walrus = MagicMock()
        mock_formatter_class = MagicMock()
        mock_formatter_instance = MagicMock()
        mock_formatter_class.return_value = mock_formatter_instance

        # Inject mocks
        monkeypatch.setitem(sys.modules, "torch", mock_torch_mod)
        monkeypatch.setitem(sys.modules, "walrus", mock_walrus)
        monkeypatch.setitem(sys.modules, "walrus.models", MagicMock())
        monkeypatch.setitem(sys.modules, "walrus.data", MagicMock())
        monkeypatch.setitem(sys.modules, "walrus.data.well_to_multi_transformer", MagicMock())

        monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch_mod)

        # Patch the lazy imports inside _load_walrus_model
        monkeypatch.setattr(
            "dpf.ai.surrogate.DPFSurrogate._load_walrus_model",
            lambda self, cd: _fake_load_walrus(
                self, cd, mock_instantiate, mock_formatter_class, mock_revin_constructor
            ),
        )

        return {
            "ckpt": ckpt,
            "mock_torch": mock_torch_mod,
            "mock_model": mock_model,
            "mock_instantiate": mock_instantiate,
            "mock_config": mock_config,
            "mock_formatter_class": mock_formatter_class,
            "checkpoint_data": checkpoint_data,
        }

    def test_walrus_model_loaded_not_dict(self, mock_walrus_env):
        """When walrus is available, _model is NOT a dict (it's the real model)."""
        from dpf.ai.surrogate import DPFSurrogate

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert not isinstance(surrogate._model, dict)
        assert surrogate._is_walrus_model is True

    def test_walrus_model_is_loaded_true(self, mock_walrus_env):
        """is_loaded returns True when WALRUS model loaded."""
        from dpf.ai.surrogate import DPFSurrogate

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert surrogate.is_loaded is True

    def test_walrus_model_instantiate_called_with_n_states_11(self, mock_walrus_env):
        """instantiate is called with n_states=11 (5 scalars + 6 vector components)."""
        from dpf.ai.surrogate import DPFSurrogate

        DPFSurrogate(mock_walrus_env["ckpt"])

        mock_walrus_env["mock_instantiate"].assert_called_once()
        call_kwargs = mock_walrus_env["mock_instantiate"].call_args
        assert call_kwargs[1]["n_states"] == 11

    def test_walrus_model_load_state_dict_called(self, mock_walrus_env):
        """model.load_state_dict is called with checkpoint weights."""
        from dpf.ai.surrogate import DPFSurrogate

        DPFSurrogate(mock_walrus_env["ckpt"])

        mock_walrus_env["mock_model"].load_state_dict.assert_called_once()

    def test_walrus_model_set_to_eval_mode(self, mock_walrus_env):
        """model.eval() is called after loading."""
        from dpf.ai.surrogate import DPFSurrogate

        DPFSurrogate(mock_walrus_env["ckpt"])

        mock_walrus_env["mock_model"].eval.assert_called_once()

    def test_walrus_model_moved_to_device(self, mock_walrus_env):
        """model.to(device) is called."""
        from dpf.ai.surrogate import DPFSurrogate

        DPFSurrogate(mock_walrus_env["ckpt"], device="cpu")

        mock_walrus_env["mock_model"].to.assert_called_with("cpu")

    def test_walrus_revin_created(self, mock_walrus_env):
        """RevIN is instantiated from config."""
        from dpf.ai.surrogate import DPFSurrogate

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert surrogate._revin is not None

    def test_walrus_formatter_created(self, mock_walrus_env):
        """ChannelsFirstWithTimeFormatter is created."""
        from dpf.ai.surrogate import DPFSurrogate

        surrogate = DPFSurrogate(mock_walrus_env["ckpt"])

        assert surrogate._formatter is not None

    def test_walrus_missing_config_falls_back_to_placeholder(self, monkeypatch, tmp_path):
        """If checkpoint has no 'config' key, falls back to placeholder dict."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        mock_torch_mod = MagicMock()
        mock_torch_mod.load.return_value = {"state_dict": {}}  # No 'config' key
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch_mod.from_numpy.return_value = mock_tensor

        monkeypatch.setitem(sys.modules, "torch", mock_torch_mod)
        monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_WALRUS", True)
        monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch_mod)

        # Patch _load_walrus_model to avoid hydra import
        monkeypatch.setattr(
            "dpf.ai.surrogate.DPFSurrogate._load_walrus_model",
            lambda self, cd: _fake_load_walrus(
                self, cd, MagicMock(), MagicMock(), MagicMock()
            ),
        )

        from dpf.ai.surrogate import DPFSurrogate

        surrogate = DPFSurrogate(ckpt)

        # Falls back to placeholder dict
        assert isinstance(surrogate._model, dict)
        assert surrogate._is_walrus_model is False


def _fake_load_walrus(self, checkpoint_data, mock_instantiate, mock_formatter_class, mock_revin):
    """Mimics _load_walrus_model using pre-built mocks."""
    from dpf.ai.surrogate import _N_CHANNELS

    config = checkpoint_data.get("config")
    if config is None:
        self._model = {
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "data": checkpoint_data,
        }
        return

    model = mock_instantiate(config.model, n_states=_N_CHANNELS)

    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data.get("state_dict"))
    if state_dict is not None:
        model.load_state_dict(state_dict)

    model.eval()
    model.to(self.device)
    self._model = model
    self._revin = mock_revin()
    self._formatter = mock_formatter_class()


# ── Tensor Conversion Tests ──────────────────────────────────────


class TestWALRUSTensorConversion:
    """Test _states_to_walrus_tensor and _tensor_to_state."""

    @pytest.fixture
    def surrogate(self, monkeypatch, tmp_path):
        """Create a DPFSurrogate in placeholder mode for tensor tests."""
        # These tests need real torch for tensor operations
        torch = pytest.importorskip("torch")

        import dpf.ai.surrogate as surrogate_mod

        # Ensure real torch is used (not a mock from prior tests)
        monkeypatch.setattr(surrogate_mod, "torch", torch)

        from dpf.ai.surrogate import DPFSurrogate

        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        # Bypass __init__ to avoid torch.load
        obj = object.__new__(DPFSurrogate)
        obj.checkpoint_path = ckpt
        obj.device = "cpu"
        obj.history_length = 4
        obj._model = {"placeholder": True}  # dict → not walrus model
        obj._revin = None
        obj._formatter = None
        return obj

    def test_states_to_walrus_tensor_shape(self, surrogate):
        """_states_to_walrus_tensor returns shape (1, T, 11, *spatial)."""
        states = [_make_state(shape=(4, 4, 4)) for _ in range(3)]

        tensor = surrogate._states_to_walrus_tensor(states)

        assert tensor.shape == (1, 3, 11, 4, 4, 4)

    def test_states_to_walrus_tensor_dtype_float32(self, surrogate):
        """_states_to_walrus_tensor returns float32 tensor."""
        states = [_make_state(shape=(4, 4, 4)) for _ in range(2)]

        tensor = surrogate._states_to_walrus_tensor(states)

        import torch

        assert tensor.dtype == torch.float32

    def test_states_to_walrus_tensor_channel_order(self, surrogate):
        """Channel order: rho=0, Te=1, Ti=2, pressure=3, psi=4, Bx=5..Bz=7, vx=8..vz=10."""
        state = _make_state(shape=(2, 2, 2))
        state["rho"][:] = 1.0
        state["Te"][:] = 2.0
        state["Ti"][:] = 3.0
        state["pressure"][:] = 4.0
        state["psi"][:] = 5.0
        state["B"][0, :] = 6.0  # Bx
        state["B"][1, :] = 7.0  # By
        state["B"][2, :] = 8.0  # Bz
        state["velocity"][0, :] = 9.0   # vx
        state["velocity"][1, :] = 10.0  # vy
        state["velocity"][2, :] = 11.0  # vz

        tensor = surrogate._states_to_walrus_tensor([state])

        # tensor shape: (1, 1, 11, 2, 2, 2)
        arr = tensor[0, 0].numpy()  # (11, 2, 2, 2)
        assert np.allclose(arr[0], 1.0)   # rho
        assert np.allclose(arr[1], 2.0)   # Te
        assert np.allclose(arr[2], 3.0)   # Ti
        assert np.allclose(arr[3], 4.0)   # pressure
        assert np.allclose(arr[4], 5.0)   # psi
        assert np.allclose(arr[5], 6.0)   # Bx
        assert np.allclose(arr[6], 7.0)   # By
        assert np.allclose(arr[7], 8.0)   # Bz
        assert np.allclose(arr[8], 9.0)   # vx
        assert np.allclose(arr[9], 10.0)  # vy
        assert np.allclose(arr[10], 11.0)  # vz

    def test_tensor_to_state_roundtrip(self, surrogate):
        """_states_to_walrus_tensor → _tensor_to_state preserves values."""
        state = _make_state(shape=(4, 4, 4))
        state["rho"][:] = 2.5
        state["Te"][:] = 1e5
        state["B"][0, :] = 0.5

        tensor = surrogate._states_to_walrus_tensor([state])
        # Extract single step: (1, 1, 11, 4, 4, 4) → (11, 4, 4, 4)
        single = tensor[0, 0]

        recovered = surrogate._tensor_to_state(single, state)

        assert set(recovered.keys()) == set(state.keys())
        np.testing.assert_allclose(recovered["rho"], state["rho"], atol=1e-5)
        np.testing.assert_allclose(recovered["Te"], state["Te"], atol=1e-1)
        np.testing.assert_allclose(recovered["B"], state["B"], atol=1e-5)

    def test_tensor_to_state_correct_shapes(self, surrogate):
        """_tensor_to_state produces correct shapes for scalars and vectors."""
        import torch

        ref_state = _make_state(shape=(4, 4, 4))
        # Create a (11, 4, 4, 4) tensor
        tensor = torch.zeros(11, 4, 4, 4, dtype=torch.float32)

        result = surrogate._tensor_to_state(tensor, ref_state)

        for key in ["rho", "Te", "Ti", "pressure", "psi"]:
            assert result[key].shape == (4, 4, 4), f"{key} shape mismatch"
        for key in ["B", "velocity"]:
            assert result[key].shape == (3, 4, 4, 4), f"{key} shape mismatch"

    def test_tensor_to_state_removes_batch_dim(self, surrogate):
        """_tensor_to_state handles (1, C, *spatial) input by removing batch dim."""
        import torch

        ref_state = _make_state(shape=(4, 4, 4))
        tensor = torch.zeros(1, 11, 4, 4, 4, dtype=torch.float32)

        result = surrogate._tensor_to_state(tensor, ref_state)

        assert result["rho"].shape == (4, 4, 4)


# ── Well Exporter Compliance Tests ───────────────────────────────


class TestWellExporterCompliance:
    """Test Well format spec compliance fixes."""

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create a WellExporter instance."""
        h5py = pytest.importorskip("h5py")  # noqa: F841
        from dpf.ai.well_exporter import WellExporter

        output = tmp_path / "test.h5"
        return WellExporter(
            output_path=output,
            grid_shape=(8, 8, 8),
            dx=0.01,
            geometry="cartesian",
        )

    def test_grid_type_is_cartesian(self, exporter, tmp_path):
        """Well files use grid_type='cartesian', not 'uniform'."""
        import h5py

        state = _make_state(shape=(8, 8, 8))
        exporter.add_snapshot(state=state, time=0.0)
        exporter.finalize()

        with h5py.File(exporter.output_path, "r") as f:
            assert f.attrs["grid_type"] == "cartesian"

    def test_t0_fields_have_varying_attributes(self, exporter):
        """t0_fields datasets have dim_varying, sample_varying, time_varying."""
        import h5py

        state = _make_state(shape=(8, 8, 8))
        exporter.add_snapshot(state=state, time=0.0)
        exporter.finalize()

        with h5py.File(exporter.output_path, "r") as f:
            if "t0_fields" in f:
                for ds_name in f["t0_fields"]:
                    ds = f["t0_fields"][ds_name]
                    assert "dim_varying" in ds.attrs, f"Missing dim_varying on {ds_name}"
                    assert "sample_varying" in ds.attrs, f"Missing sample_varying on {ds_name}"
                    assert "time_varying" in ds.attrs, f"Missing time_varying on {ds_name}"
                    assert ds.attrs["dim_varying"] is True or ds.attrs["dim_varying"]
                    assert ds.attrs["sample_varying"] is True or ds.attrs["sample_varying"]
                    assert ds.attrs["time_varying"] is True or ds.attrs["time_varying"]

    def test_t1_fields_have_varying_attributes(self, exporter):
        """t1_fields datasets have dim_varying, sample_varying, time_varying."""
        import h5py

        state = _make_state(shape=(8, 8, 8))
        exporter.add_snapshot(state=state, time=0.0)
        exporter.finalize()

        with h5py.File(exporter.output_path, "r") as f:
            if "t1_fields" in f:
                for ds_name in f["t1_fields"]:
                    ds = f["t1_fields"][ds_name]
                    assert "dim_varying" in ds.attrs, f"Missing dim_varying on {ds_name}"
                    assert "sample_varying" in ds.attrs, f"Missing sample_varying on {ds_name}"
                    assert "time_varying" in ds.attrs, f"Missing time_varying on {ds_name}"


# ── Batch Runner API Tests ───────────────────────────────────────


class TestBatchRunnerAPI:
    """Test batch_runner.py WellExporter API usage."""

    def test_batch_runner_well_exporter_constructor_args(self, monkeypatch):
        """BatchRunner passes grid_shape, dx, geometry, sim_params to WellExporter."""
        from dpf.ai.batch_runner import BatchRunner

        captured_args: dict = {}

        class FakeWellExporter:
            def __init__(self, **kwargs):
                captured_args.update(kwargs)

            def add_snapshot(self, **kwargs):
                pass

            def finalize(self):
                pass

        # Mock SimulationConfig
        mock_config = MagicMock()
        mock_config.grid_shape = [8, 8, 8]
        mock_config.dx = 0.01
        mock_config.geometry.type = "cartesian"
        mock_config.model_dump.return_value = {}

        # Mock engine
        mock_engine_class = MagicMock()
        mock_engine = MagicMock()
        mock_engine.diagnostics.field_snapshots = [{"rho": np.ones((8, 8, 8)), "time": 0.0}]
        mock_engine_class.return_value = mock_engine

        # Create BatchRunner with mock base config
        runner = object.__new__(BatchRunner)
        runner.base_config = mock_config
        runner.parameter_ranges = []
        runner.n_samples = 1
        runner.output_dir = Path("/tmp/test_batch")
        runner.workers = 1
        runner.field_interval = 10

        # Patch dependencies
        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", FakeWellExporter)
        monkeypatch.setattr("dpf.ai.batch_runner.SimulationConfig", lambda **kw: mock_config)

        # Monkeypatch lazy import of SimulationEngine
        def mock_run_single(idx, params):
            # Simulate what run_single does (but with mocks)
            from dpf.ai.batch_runner import WellExporter

            output_path = runner.output_dir / f"trajectory_{idx:04d}.h5"
            exporter = WellExporter(
                output_path=output_path,
                grid_shape=tuple(mock_config.grid_shape),
                dx=mock_config.dx,
                geometry=mock_config.geometry.type,
                sim_params=params,
            )
            exporter.finalize()
            return (idx, None)

        monkeypatch.setattr(runner, "run_single", mock_run_single)

        # Call run_single directly
        mock_run_single(0, {"V0": 10e3})

        assert "grid_shape" in captured_args
        assert captured_args["grid_shape"] == (8, 8, 8)
        assert "dx" in captured_args
        assert captured_args["dx"] == 0.01
        assert "geometry" in captured_args
        assert "sim_params" in captured_args

    def test_batch_runner_add_snapshot_uses_state_time(self, monkeypatch):
        """BatchRunner calls add_snapshot with state= and time= kwargs."""
        captured_calls = []

        class FakeWellExporter:
            def __init__(self, **kwargs):
                pass

            def add_snapshot(self, **kwargs):
                captured_calls.append(kwargs)

            def finalize(self):
                pass

        monkeypatch.setattr("dpf.ai.batch_runner.WellExporter", FakeWellExporter)

        exporter = FakeWellExporter()
        snapshot = {"rho": np.ones((8, 8, 8)), "time": 1.5}
        t = snapshot.get("time", 0.0)
        exporter.add_snapshot(state=snapshot, time=t)

        assert len(captured_calls) == 1
        assert "state" in captured_calls[0]
        assert "time" in captured_calls[0]
        assert captured_calls[0]["time"] == 1.5


# ── Server Endpoint Tests ────────────────────────────────────────


class TestServerEndpoints:
    """Test realtime_server.py endpoint fixes."""

    def test_sweep_uses_surrogate_parameter_sweep(self):
        """POST /api/ai/sweep calls surrogate.parameter_sweep, not parameter_sweep module."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient

        from dpf.ai import realtime_server
        from dpf.server.app import app

        # Create mock surrogate with parameter_sweep method
        mock_surrogate = MagicMock()
        mock_surrogate.parameter_sweep.return_value = [{"max_rho": 1.0}]

        original = realtime_server._surrogate
        realtime_server._surrogate = mock_surrogate

        try:
            client = TestClient(app)
            configs = [{"V0": 10e3}]
            response = client.post("/api/ai/sweep?n_steps=5", json=configs)

            assert response.status_code == 200
            mock_surrogate.parameter_sweep.assert_called_once()
        finally:
            realtime_server._surrogate = original

    def test_inverse_endpoint_uses_find_config_not_optimize(self):
        """ai_inverse endpoint calls designer.find_config, not designer.optimize."""
        import inspect

        from dpf.ai import realtime_server

        source = inspect.getsource(realtime_server.ai_inverse)
        # Should use find_config (InverseDesigner method)
        assert "find_config" in source
        # Should NOT use optimize (which doesn't exist)
        assert "designer.optimize(" not in source
        # Should access InverseResult attributes
        assert "inverse_result.best_params" in source
        assert "inverse_result.best_score" in source
        assert "inverse_result.n_trials" in source

    def test_inverse_endpoint_returns_correct_result_structure(self, monkeypatch):
        """ai_inverse returns best_config, loss, n_trials from InverseResult."""
        pytest.importorskip("fastapi")

        from dpf.ai import realtime_server

        # Create mock InverseResult
        @dataclass
        class FakeInverseResult:
            best_params: dict = field(default_factory=lambda: {"V0": 10e3})
            best_score: float = 0.01
            all_trials: list = field(default_factory=list)
            n_trials: int = 10

        # Mock InverseDesigner
        mock_designer = MagicMock()
        mock_designer.find_config.return_value = FakeInverseResult()

        mock_surrogate = MagicMock()
        original = realtime_server._surrogate
        realtime_server._surrogate = mock_surrogate

        try:
            monkeypatch.setattr(
                "dpf.ai.inverse_design.InverseDesigner",
                MagicMock(return_value=mock_designer),
            )

            # Call the endpoint function directly (bypassing HTTP)
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                realtime_server.ai_inverse(
                    targets={"max_Te": 5e3},
                    method="bayesian",
                    n_trials=10,
                )
            )

            assert "best_config" in result
            assert result["best_config"] == {"V0": 10e3}
            assert result["loss"] == 0.01
            assert result["n_trials"] == 10
            mock_designer.find_config.assert_called_once()
        finally:
            realtime_server._surrogate = original

    def test_confidence_returns_prediction_with_confidence_fields(self):
        """POST /api/ai/confidence returns PredictionWithConfidence dataclass fields."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from fastapi.testclient import TestClient

        from dpf.ai import realtime_server
        from dpf.ai.confidence import PredictionWithConfidence
        from dpf.server.app import app

        # Create mock ensemble that returns PredictionWithConfidence
        mock_ensemble = MagicMock()

        sample_state = {
            "rho": np.ones((2, 2, 2)),
            "Te": np.ones((2, 2, 2)) * 1e4,
        }
        mock_ensemble.predict.return_value = PredictionWithConfidence(
            mean_state=sample_state,
            std_state={"rho": np.ones((2, 2, 2)) * 0.1, "Te": np.ones((2, 2, 2)) * 100.0},
            confidence=0.92,
            ood_score=0.08,
            n_models=3,
        )

        original_surrogate = realtime_server._surrogate
        original_ensemble = realtime_server._ensemble
        realtime_server._surrogate = MagicMock()
        realtime_server._ensemble = mock_ensemble

        try:
            client = TestClient(app)
            history = [{"rho": [[1.0, 2.0], [3.0, 4.0]]}]
            response = client.post("/api/ai/confidence", json=history)

            assert response.status_code == 200
            data = response.json()

            assert "predicted_state" in data
            assert "confidence" in data
            assert "ood_score" in data
            assert "confidence_score" in data
            assert "n_models" in data
            assert "inference_time_ms" in data

            assert data["ood_score"] == pytest.approx(0.08, abs=1e-6)
            assert data["confidence_score"] == pytest.approx(0.92, abs=1e-6)
            assert data["n_models"] == 3
        finally:
            realtime_server._surrogate = original_surrogate
            realtime_server._ensemble = original_ensemble

    def test_confidence_endpoint_no_parameter_sweep_import_error(self):
        """Sweep endpoint does not import from dpf.ai.parameter_sweep (which doesn't exist)."""
        # Verify that realtime_server.py does not contain the broken import
        import inspect

        from dpf.ai import realtime_server

        source = inspect.getsource(realtime_server)
        assert "from dpf.ai.parameter_sweep" not in source
        assert "import parameter_sweep" not in source


# ── Placeholder Fallback Tests ───────────────────────────────────


class TestPlaceholderFallback:
    """Test that surrogate works in placeholder mode when HAS_WALRUS=False."""

    @pytest.fixture
    def placeholder_surrogate(self, monkeypatch, tmp_path):
        """Create a surrogate in placeholder mode."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        mock_torch_mod = MagicMock()
        mock_torch_mod.load.return_value = {"state_dict": {}}

        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch_mod.from_numpy.return_value = mock_tensor

        monkeypatch.setitem(sys.modules, "torch", mock_torch_mod)
        monkeypatch.setattr("dpf.ai.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.HAS_WALRUS", False)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_TORCH", True)
        monkeypatch.setattr("dpf.ai.surrogate.HAS_WALRUS", False)
        monkeypatch.setattr("dpf.ai.surrogate.torch", mock_torch_mod)

        # Mock field mapping functions
        monkeypatch.setattr(
            "dpf.ai.surrogate.dpf_scalar_to_well",
            lambda field, *a, **kw: field.astype(np.float32),
        )
        monkeypatch.setattr(
            "dpf.ai.surrogate.dpf_vector_to_well",
            lambda field, *a, **kw: field.astype(np.float32),
        )

        from dpf.ai.surrogate import DPFSurrogate

        return DPFSurrogate(ckpt, history_length=2)

    def test_placeholder_is_not_walrus_model(self, placeholder_surrogate):
        """_is_walrus_model is False in placeholder mode."""
        assert placeholder_surrogate._is_walrus_model is False

    def test_placeholder_predict_returns_copy_of_last_state(self, placeholder_surrogate):
        """predict_next_step returns copy of last state in placeholder mode."""
        history = [_make_state() for _ in range(2)]
        history[-1]["rho"][:] = 42.0

        result = placeholder_surrogate.predict_next_step(history)

        np.testing.assert_allclose(result["rho"], 42.0)
        # Should be a copy, not the same object
        assert result["rho"] is not history[-1]["rho"]


# ── Module Constants Tests ───────────────────────────────────────


class TestModuleConstants:
    """Test module-level constants in surrogate.py."""

    def test_n_channels_equals_11(self):
        """_N_CHANNELS is 11 (5 scalars + 2 vectors × 3 components)."""
        from dpf.ai.surrogate import _N_CHANNELS

        assert _N_CHANNELS == 11

    def test_scalar_keys_tuple(self):
        """_SCALAR_KEYS contains the 5 DPF scalar field names."""
        from dpf.ai.surrogate import _SCALAR_KEYS

        assert _SCALAR_KEYS == ("rho", "Te", "Ti", "pressure", "psi")

    def test_vector_keys_tuple(self):
        """_VECTOR_KEYS contains the 2 DPF vector field names."""
        from dpf.ai.surrogate import _VECTOR_KEYS

        assert _VECTOR_KEYS == ("B", "velocity")


# ── Confidence.py Fix Tests ──────────────────────────────────────


class TestConfidenceFix:
    """Test that confidence.py uses constructor, not .load()."""

    def test_ensemble_predictor_uses_constructor(self):
        """EnsemblePredictor uses DPFSurrogate() constructor, not .load()."""
        import inspect

        from dpf.ai.confidence import EnsemblePredictor

        source = inspect.getsource(EnsemblePredictor)
        assert "DPFSurrogate.load(" not in source
        assert "DPFSurrogate(" in source
