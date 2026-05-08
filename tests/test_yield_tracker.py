"""Tests for time-resolved neutron yield tracker."""

from __future__ import annotations

import numpy as np

from dpf.diagnostics.yield_tracker import YieldResult, YieldTracker


class TestYieldResult:
    def test_empty_result(self):
        r = YieldResult()
        assert r.Y_total == 0.0
        assert r.bt_fraction == 0.0
        assert r.peak_yield_time == 0.0

    def test_total_yield(self):
        r = YieldResult()
        r.Y_thermo_cumulative = [0, 1e6, 5e6]
        r.Y_bt_cumulative = [0, 2e6, 8e6]
        r.dY_thermo = [0, 1e6, 4e6]
        r.dY_bt = [0, 2e6, 6e6]
        r.times = [0, 1e-6, 2e-6]
        assert r.Y_total == 5e6 + 8e6
        assert r.bt_fraction > 0.5

    def test_peak_yield_time(self):
        r = YieldResult()
        r.times = [0, 1e-6, 2e-6, 3e-6]
        r.dY_thermo = [0, 100, 1000, 10]
        r.dY_bt = [0, 50, 500, 5]
        assert r.peak_yield_time == 2e-6

    def test_summary_dict_exposes_model_sources(self):
        r = YieldResult()
        summary = r.to_summary_dict()
        assert summary["Y_thermonuclear"] == 0.0
        assert summary["Y_beam_target"] == 0.0
        assert summary["Y_neutron"] == 0.0
        assert "Bosch-Hale" in summary["model_components"]["thermonuclear"]
        assert "Lee/Saw" in summary["model_components"]["beam_target"]
        assert "pure MHD" in summary["validity_notes"]["mhd_limit"]

    def test_summary_dict_preserves_accumulated_components(self):
        r = YieldResult()
        r.Y_thermo_cumulative = [1.0e5]
        r.Y_bt_cumulative = [3.0e5]
        r.dY_thermo = [1.0e5]
        r.dY_bt = [3.0e5]
        r.times = [2.0e-7]
        summary = r.to_summary_dict()
        assert summary["Y_thermonuclear"] == 1.0e5
        assert summary["Y_beam_target"] == 3.0e5
        assert summary["Y_neutron"] == 4.0e5
        assert summary["bt_fraction"] == 0.75
        assert summary["peak_yield_time_s"] == 2.0e-7


class TestYieldTracker:
    def _make_state(self, rho: float = 1e-3, T_K: float = 1e7) -> dict:
        nr, ny, nz = 8, 1, 16
        return {
            "rho": np.ones((nr, ny, nz)) * rho,
            "pressure": np.ones((nr, ny, nz)) * rho * 1.38e-23 * T_K / 3.34e-27,
        }

    def test_init(self):
        tracker = YieldTracker()
        assert tracker._Y_thermo == 0.0
        assert tracker._Y_bt == 0.0

    def test_accumulate_cold(self):
        """Cold plasma should produce zero yield."""
        tracker = YieldTracker()
        state = self._make_state(rho=1e-4, T_K=300.0)
        tracker.accumulate(state, dt=1e-9)
        result = tracker.get_result()
        assert len(result.times) == 1
        assert result.dY_thermo[0] == 0.0

    def test_accumulate_hot(self):
        """Hot dense plasma should produce nonzero thermonuclear yield."""
        tracker = YieldTracker()
        # 1 keV = 1.16e7 K, high density
        state = self._make_state(rho=1e-2, T_K=1.16e7)
        tracker.accumulate(state, dt=1e-9, cell_volume=1e-9)
        result = tracker.get_result()
        assert result.dY_thermo[0] > 0

    def test_multiple_steps(self):
        """Multiple accumulations should increase cumulative yield."""
        tracker = YieldTracker()
        state = self._make_state(rho=1e-2, T_K=1.16e7)
        for _ in range(5):
            tracker.accumulate(state, dt=1e-9, cell_volume=1e-9)
        result = tracker.get_result()
        assert len(result.times) == 5
        # Cumulative should be monotonically increasing
        for i in range(1, 5):
            assert result.Y_thermo_cumulative[i] >= result.Y_thermo_cumulative[i - 1]

    def test_summary(self):
        tracker = YieldTracker()
        state = self._make_state(rho=1e-2, T_K=1.16e7)
        tracker.accumulate(state, dt=1e-9, cell_volume=1e-9)
        s = tracker.summary()
        assert "Y_total" in s
        assert "thermo" in s

    def test_tracker_result_carries_source_metadata_after_accumulation(self):
        tracker = YieldTracker()
        state = self._make_state(rho=1e-2, T_K=1.16e7)
        tracker.accumulate(state, dt=1e-9, cell_volume=1e-9)
        summary = tracker.get_result().to_summary_dict()
        assert summary["Y_thermonuclear"] >= 0.0
        assert summary["Y_beam_target"] >= 0.0
        assert summary["model_components"]["total"] == (
            "sum of thermonuclear and beam-target estimates"
        )
        assert "phenomenological" in summary["validity_notes"]["beam_target"]

    def test_with_ti_field(self):
        """State with explicit Ti should work."""
        tracker = YieldTracker()
        nr, ny, nz = 8, 1, 16
        state = {
            "rho": np.ones((nr, ny, nz)) * 1e-2,
            "pressure": np.ones((nr, ny, nz)) * 1e5,
            "Ti": np.ones((nr, ny, nz)) * 1.16e7,  # 1 keV
        }
        tracker.accumulate(state, dt=1e-9, cell_volume=1e-9)
        result = tracker.get_result()
        assert result.T_peak_keV[0] > 0.5
