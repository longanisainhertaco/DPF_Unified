"""Tests for reproducibility package (export/import/verify)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from dpf.validation.reproducibility import (
    _get_git_info,
    _get_platform_info,
    _numpy_safe,
    create_reproducibility_package,
    format_package_summary,
    load_package,
    save_package,
)


class TestNumpySafe:
    def test_scalar(self):
        assert _numpy_safe(np.float64(1.5)) == 1.5
        assert _numpy_safe(np.int64(42)) == 42
        assert _numpy_safe(np.bool_(True)) is True

    def test_small_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _numpy_safe(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_large_array_summary(self):
        arr = np.random.rand(100, 100)  # 10,000 elements > 1000 threshold
        result = _numpy_safe(arr)
        assert result["_type"] == "ndarray_summary"
        assert result["shape"] == [100, 100]
        assert "checksum" in result

    def test_nested_dict(self):
        d = {"a": np.float64(1.0), "b": [np.int64(2)]}
        result = _numpy_safe(d)
        assert result == {"a": 1.0, "b": [2]}


class TestPlatformInfo:
    def test_has_required_keys(self):
        info = _get_platform_info()
        assert "system" in info
        assert "machine" in info
        assert "python" in info

    def test_git_info(self):
        info = _get_git_info()
        assert "commit" in info
        assert "branch" in info


class TestCreatePackage:
    def _mock_result(self) -> dict:
        return {
            "I_peak": 1.733,
            "t_peak": 5.8,
            "dip_pct": 12.0,
            "n_steps": 500,
            "elapsed_s": 2.5,
            "circuit": {"C": 1.332e-3, "V0": 27e3, "L0": 33.5e-9},
            "snowplow_cfg": {"anode_length": 0.6, "current_fraction": 0.7},
            "t_us": np.linspace(0, 10, 100),
            "I_MA": np.sin(np.linspace(0, np.pi, 100)) * 1.733,
            "neutron_yield": {"Y_neutron": 1e8, "bt_fraction": 0.6},
            "bennett": {"T_bennett_keV": 0.48},
            "breakdown": {
                "mechanism": "Paschen",
                "civ_ratio": 11020.8,
                "narrative": "long text...",
            },
        }

    def test_creates_valid_package(self):
        pkg = create_reproducibility_package(
            self._mock_result(), "pf1000", "hybrid",
        )
        assert pkg["dpf_unified_reproducibility"] == "1.0"
        assert "created" in pkg
        assert pkg["configuration"]["preset"] == "pf1000"
        assert pkg["outputs"]["I_peak_MA"] == 1.733

    def test_json_serializable(self):
        pkg = create_reproducibility_package(
            self._mock_result(), "pf1000", "hybrid",
        )
        # Should not raise
        json_str = json.dumps(pkg, default=str)
        assert len(json_str) > 100

    def test_excludes_narrative(self):
        pkg = create_reproducibility_package(
            self._mock_result(), "pf1000", "hybrid",
        )
        bd = pkg["outputs"].get("breakdown", {})
        assert "narrative" not in bd

    def test_waveform_checksum(self):
        pkg = create_reproducibility_package(
            self._mock_result(), "pf1000", "hybrid",
        )
        assert pkg["verification"]["waveform_checksum"] != "no_waveform"
        assert pkg["verification"]["waveform_points"] == 100


class TestSaveLoad:
    def test_round_trip(self):
        result = {
            "I_peak": 1.5, "t_peak": 5.0, "dip_pct": 10.0,
            "n_steps": 100, "elapsed_s": 1.0,
            "circuit": {"C": 1e-3}, "snowplow_cfg": {},
            "t_us": np.array([0, 1, 2]), "I_MA": np.array([0, 1, 0.5]),
        }
        pkg = create_reproducibility_package(result, "tutorial", "lee")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_package(pkg, path)
        loaded = load_package(path)

        assert loaded["configuration"]["preset"] == "tutorial"
        assert loaded["outputs"]["I_peak_MA"] == 1.5
        Path(path).unlink()


class TestFormatSummary:
    def test_summary_not_empty(self):
        pkg = create_reproducibility_package(
            {"I_peak": 1.5, "t_peak": 5.0, "dip_pct": 0, "n_steps": 100,
             "elapsed_s": 1.0, "circuit": {}, "snowplow_cfg": {},
             "t_us": np.array([]), "I_MA": np.array([])},
            "tutorial", "lee",
        )
        summary = format_package_summary(pkg)
        assert "DPF-Unified" in summary
        assert "1.500 MA" in summary
