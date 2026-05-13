from __future__ import annotations

import json

import numpy as np
import pytest

from dpf.diagnostics.hdf5_writer import HDF5Writer
from dpf.io.export_scope import accepted_export_formats, export_scope_decisions


def test_export_scope_accepts_only_hdf5_and_well_for_v1() -> None:
    decisions = {decision.format_id: decision for decision in export_scope_decisions()}

    assert accepted_export_formats() == ["hdf5_diagnostics", "well_hdf5"]
    assert decisions["vtk_vtu"].status == "deferred"
    assert decisions["cgns_hdf5"].status == "deferred"
    assert decisions["openfoam"].status == "deferred"
    assert decisions["ansys_pymapdl"].status == "deferred"
    assert decisions["hdf5_diagnostics"].acceptance_evidence
    assert decisions["well_hdf5"].acceptance_evidence


def test_hdf5_diagnostics_export_carries_units_and_schema(tmp_path) -> None:
    import h5py

    path = tmp_path / "diag.h5"
    writer = HDF5Writer(path, field_output_interval=1)
    state = {
        "rho": np.ones((2, 2, 2)),
        "B": np.zeros((3, 2, 2, 2)),
        "Te": np.ones((2, 2, 2)) * 10.0,
        "Ti": np.ones((2, 2, 2)) * 20.0,
        "velocity": np.zeros((3, 2, 2, 2)),
        "pressure": np.ones((2, 2, 2)),
        "circuit": {"current": 1.0, "voltage": 2.0, "energy_total": 3.0},
    }

    writer.record(state, 1e-9)
    writer.finalize()

    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema_version"] == "dpf-hdf5-diagnostics-v1"
        assert handle.attrs["time_base_units"] == "s"
        assert handle["scalars/time"].attrs["units"] == "s"
        assert handle["scalars/current"].attrs["units"] == "A"
        assert handle["scalars/max_div_B"].attrs["units"] == "T/cell"
        assert (
            handle["scalars/max_div_B"].attrs["diagnostic_status"]
            == "rough_array_metric_not_physical_divergence"
        )
        assert (
            handle["scalars/max_div_B"].attrs["validation_status"]
            == "not_validation_evidence"
        )
        assert handle["fields/snapshot_0000/rho"].attrs["units"] == "kg/m^3"
        assert handle["fields/snapshot_0000/B"].attrs["units"] == "T"


def test_hdf5_max_div_b_is_array_metric_with_component_axes(tmp_path) -> None:
    import h5py

    nx, ny, nz = 3, 4, 5
    x = np.arange(nx, dtype=float)[:, None, None]
    y = np.arange(ny, dtype=float)[None, :, None]
    z = np.arange(nz, dtype=float)[None, None, :]

    B = np.zeros((3, nx, ny, nz), dtype=float)
    B[0] = x
    B[1] = 2.0 * y
    B[2] = 3.0 * z

    path = tmp_path / "diag_axes.h5"
    writer = HDF5Writer(path, field_output_interval=0)
    writer.record(
        {
            "rho": np.ones((nx, ny, nz)),
            "B": B,
            "Te": np.ones((nx, ny, nz)),
            "Ti": np.ones((nx, ny, nz)),
            "circuit": {},
        },
        0.0,
    )
    writer.finalize()

    with h5py.File(path, "r") as handle:
        assert handle["scalars/max_div_B"][0] == pytest.approx(6.0)
        method = handle["scalars/max_div_B"].attrs["diagnostic_method"]
        assert "geometry and grid spacing are not applied" in method


def test_engine_well_adapter_preserves_spacing_geometry_and_provenance(tmp_path, monkeypatch) -> None:
    from dpf.io import well_exporter as module
    from dpf.io.well_exporter import WellExporter

    created: list[dict] = []

    class FakeFullWellExporter:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.snapshots = []

        def add_snapshot(self, state, time, circuit_scalars=None):
            self.snapshots.append((state, time, circuit_scalars))

        def finalize(self):
            return created[-1]["output_path"]

    monkeypatch.setattr(module, "_FullWellExporter", FakeFullWellExporter)
    exporter = WellExporter(
        output_dir=tmp_path,
        filename_prefix="well",
        buffer_size=1,
        dx=0.002,
        dz=0.003,
        geometry="cylindrical",
        sim_params={"backend": "python"},
    )

    exporter.append_state({"rho": np.ones((2, 1, 3))}, time=1e-9)

    assert created[0]["grid_shape"] == (2, 1, 3)
    assert created[0]["dx"] == 0.002
    assert created[0]["dz"] == 0.003
    assert created[0]["geometry"] == "cylindrical"
    assert created[0]["sim_params"] == {"backend": "python"}
    assert created[0]["artifact_classification"] is None


def test_full_well_exporter_writes_artifact_classification_metadata(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    from dpf.ai.well_exporter import WellExporter as FullWellExporter

    path = tmp_path / "classified_well.h5"
    state = {
        "rho": np.ones((2, 2, 2)),
        "pressure": np.ones((2, 2, 2)),
        "B": np.ones((3, 2, 2, 2)),
        "velocity": np.zeros((3, 2, 2, 2)),
    }
    exporter = FullWellExporter(
        output_path=path,
        grid_shape=(2, 2, 2),
        dx=0.002,
        artifact_classification={
            "owner": "qa-team",
            "classification": "internal",
            "distribution": "project-only",
            "handling_notes": "training-data interchange only",
        },
    )
    exporter.add_snapshot(state, 0.0)
    exporter.finalize()

    with h5py.File(path, "r") as handle:
        assert handle.attrs["validation_status"] == "not_validation_evidence"
        assert handle.attrs["result_label"] == "Preview"
        assert not bool(handle.attrs["can_support_validation_claims"])
        assert handle.attrs["artifact_classification"] == "internal"
        assert handle.attrs["artifact_distribution"] == "project-only"
        assert handle.attrs["artifact_owner"] == "qa-team"
        assert "KnowledgeReference" in handle.attrs["dpf_source_authority"]
        payload = json.loads(handle.attrs["artifact_classification_json"])
        assert payload["classification"] == "internal"
        assert payload["distribution"] == "project-only"


def test_well_adapter_forwards_circuit_scalars(tmp_path, monkeypatch) -> None:
    from dpf.io import well_exporter as module
    from dpf.io.well_exporter import WellExporter

    instances = []

    class FakeFullWellExporter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.snapshots = []
            instances.append(self)

        def add_snapshot(self, state, time, circuit_scalars=None):
            self.snapshots.append((state, time, circuit_scalars))

        def finalize(self):
            return self.kwargs["output_path"]

    monkeypatch.setattr(module, "_FullWellExporter", FakeFullWellExporter)
    exporter = WellExporter(output_dir=tmp_path, buffer_size=1)
    circuit_scalars = {
        "current": 1.0e6,
        "voltage": 2.0e3,
        "energy_total": 3.0,
    }

    exporter.append_state(
        {"rho": np.ones((2, 2, 2)), "B": np.zeros((3, 2, 2, 2))},
        time=2e-9,
        circuit_scalars=circuit_scalars,
    )

    assert instances[0].snapshots[0][2] == circuit_scalars


def test_full_well_exporter_labels_cylindrical_grid_type(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    from dpf.ai.well_exporter import WellExporter as FullWellExporter

    path = tmp_path / "cylindrical_well.h5"
    state = {
        "rho": np.ones((2, 1, 3)),
        "pressure": np.ones((2, 1, 3)),
        "B": np.zeros((3, 2, 1, 3)),
        "velocity": np.zeros((3, 2, 1, 3)),
    }
    exporter = FullWellExporter(
        output_path=path,
        grid_shape=(2, 1, 3),
        dx=0.002,
        dz=0.003,
        geometry="cylindrical",
    )
    exporter.add_snapshot(state, 0.0)
    exporter.finalize()

    with h5py.File(path, "r") as handle:
        assert handle.attrs["grid_type"] == "cylindrical"
        assert handle.attrs["n_spatial_dims"] == 2
        assert "r" in handle["dimensions"]
        assert "theta" in handle["dimensions"]


def test_engine_run_flushes_well_exporter_without_manual_close(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={
            "hdf5_filename": str(tmp_path / "diag.h5"),
            "well_output_interval": 1,
            "well_filename_prefix": "well",
        },
    )

    engine = SimulationEngine(config)
    engine.run(max_steps=1)

    well_files = sorted(tmp_path.glob("well_*.h5"))
    assert well_files
    with h5py.File(well_files[0], "r") as handle:
        assert handle.attrs["validation_status"] == "not_validation_evidence"
        assert handle.attrs["result_label"] == "Preview"
        assert handle.attrs["artifact_classification"] == "owner_unspecified"
        assert "dpf_artifact_classification_json" in handle.attrs
        assert "scalars" in handle
        assert "voltage" in handle["scalars"]
