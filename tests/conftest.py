"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from dpf.config import SimulationConfig


@pytest.fixture
def grid_shape():
    """Small grid for fast unit tests."""
    return (8, 8, 8)


@pytest.fixture
def dx():
    return 1e-2


@pytest.fixture
def default_circuit_params():
    """Standard DPF circuit parameters."""
    return {
        "C": 1e-6,
        "V0": 1e3,
        "L0": 1e-7,
        "R0": 0.01,
        "ESR": 0.0,
        "ESL": 0.0,
        "anode_radius": 0.005,
        "cathode_radius": 0.01,
    }


@pytest.fixture
def sample_config_dict(grid_shape, dx, default_circuit_params):
    """Minimal valid SimulationConfig as a dictionary."""
    return {
        "grid_shape": list(grid_shape),
        "dx": dx,
        "sim_time": 1e-6,
        "circuit": default_circuit_params,
    }


@pytest.fixture
def small_config(sample_config_dict):
    """Small SimulationConfig for fast unit tests."""
    return SimulationConfig(**sample_config_dict)
