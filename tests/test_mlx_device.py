"""Tests for src/dpf/metal/mlx_device.py and the has_mlx() helper in device.py."""
from __future__ import annotations

import pytest

# Skip the entire module when MLX is not installed.
# On the M3 Pro build machine this import succeeds; on CI without MLX it skips.
mx = pytest.importorskip("mlx.core", reason="MLX not installed — skipping Metal v2 device tests")

from dpf.metal.device import has_mlx  # noqa: E402
from dpf.metal.mlx_device import (  # noqa: E402
    HAS_MLX,
    mlx_default_stream,
    mlx_device_info,
    mlx_dtype,
    require_mlx,
)

# ---------------------------------------------------------------------------
# HAS_MLX module constant
# ---------------------------------------------------------------------------


def test_has_mlx_constant_is_true() -> None:
    """HAS_MLX must be True when the test module can be imported."""
    assert HAS_MLX is True


# ---------------------------------------------------------------------------
# has_mlx() in device.py
# ---------------------------------------------------------------------------


def test_has_mlx_function_returns_true() -> None:
    assert has_mlx() is True


def test_has_mlx_function_returns_bool() -> None:
    result = has_mlx()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# require_mlx()
# ---------------------------------------------------------------------------


def test_require_mlx_returns_module() -> None:
    module = require_mlx()
    assert module is not None


def test_require_mlx_has_array() -> None:
    module = require_mlx()
    assert hasattr(module, "array"), "mlx.core must expose 'array'"


def test_require_mlx_is_mlx_core() -> None:
    import mlx.core as expected

    module = require_mlx()
    assert module is expected


# ---------------------------------------------------------------------------
# mlx_dtype()
# ---------------------------------------------------------------------------


def test_mlx_dtype_float32() -> None:
    dt = mlx_dtype("float32")
    assert dt == mx.float32


def test_mlx_dtype_float16() -> None:
    dt = mlx_dtype("float16")
    assert dt == mx.float16


def test_mlx_dtype_bfloat16() -> None:
    dt = mlx_dtype("bfloat16")
    assert dt == mx.bfloat16


def test_mlx_dtype_default_is_float32() -> None:
    assert mlx_dtype() == mx.float32


def test_mlx_dtype_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown precision"):
        mlx_dtype("float64")


def test_mlx_dtype_invalid_string_raises() -> None:
    with pytest.raises(ValueError):
        mlx_dtype("int32")


# ---------------------------------------------------------------------------
# mlx_default_stream()
# ---------------------------------------------------------------------------


def test_mlx_default_stream_returns_stream_or_none() -> None:
    stream = mlx_default_stream()
    # On Metal hardware the stream must be a valid stream object, not None.
    assert stream is not None


def test_mlx_default_stream_type() -> None:
    stream = mlx_default_stream()
    assert isinstance(stream, mx.Stream)


# ---------------------------------------------------------------------------
# mlx_device_info()
# ---------------------------------------------------------------------------


def test_mlx_device_info_returns_dict() -> None:
    info = mlx_device_info()
    assert isinstance(info, dict)


def test_mlx_device_info_required_keys() -> None:
    info = mlx_device_info()
    required = {"has_mlx", "mlx_version", "metal_available", "device_name"}
    assert required.issubset(info.keys()), f"Missing keys: {required - info.keys()}"


def test_mlx_device_info_has_mlx_true() -> None:
    info = mlx_device_info()
    assert info["has_mlx"] is True


def test_mlx_device_info_version_is_string() -> None:
    info = mlx_device_info()
    assert isinstance(info["mlx_version"], str)
    assert info["mlx_version"] != "unavailable"


def test_mlx_device_info_metal_available_bool() -> None:
    info = mlx_device_info()
    assert isinstance(info["metal_available"], bool)


def test_mlx_device_info_metal_available_true_on_apple_silicon() -> None:
    info = mlx_device_info()
    assert info["metal_available"] is True


def test_mlx_device_info_device_name_is_string() -> None:
    info = mlx_device_info()
    assert isinstance(info["device_name"], str)
    assert len(info["device_name"]) > 0
