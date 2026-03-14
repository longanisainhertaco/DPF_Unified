"""Tests for experimental CSV overlay in waveform plots."""
from __future__ import annotations

import plotly.graph_objects as go
import pytest

from app_plots import create_waveform_fig, parse_experimental_csv

# ---------------------------------------------------------------------------
# Minimal simulation data dict for create_waveform_fig
# ---------------------------------------------------------------------------

def _minimal_sim_data() -> dict:
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    return {
        "t_us": t,
        "I_MA": [0.0, 0.5, 1.0, 0.8, 0.2],
        "V_kV": [27.0, 20.0, 10.0, 5.0, 0.0],
        "phases": ["rundown"] * 5,
        "has_snowplow": False,
        "dip_pct": 0.0,
        "t_peak": 2.0,
        "I_peak": 1.0,
        "I_dip": 0.8,
        "t_dip": 3.0,
        "crowbar_t": None,
        "circuit": {"anode_radius": 0.115, "cathode_radius": 0.160},
        "snowplow_cfg": {"anode_length": 0.60},
        "snowplow_obj": None,
    }


# ---------------------------------------------------------------------------
# parse_experimental_csv — column name formats
# ---------------------------------------------------------------------------

class TestParseExperimentalCsv:
    def test_time_us_current_ma(self) -> None:
        csv = "time_us,current_MA\n0.0,0.0\n1.0,0.5\n2.0,1.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([0.0, 1.0, 2.0])
        assert result["I_MA"] == pytest.approx([0.0, 0.5, 1.0])

    def test_t_us_i_ma_aliases(self) -> None:
        csv = "t_us,I_MA\n0.5,0.1\n1.5,0.9\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([0.5, 1.5])
        assert result["I_MA"] == pytest.approx([0.1, 0.9])

    def test_generic_t_i_columns(self) -> None:
        csv = "t,I\n0.0,0.0\n2.0,1.2\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([0.0, 2.0])
        assert result["I_MA"] == pytest.approx([0.0, 1.2])

    def test_time_current_generic_labels(self) -> None:
        csv = "time,current\n1.0,0.3\n3.0,0.9\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([1.0, 3.0])
        assert result["I_MA"] == pytest.approx([0.3, 0.9])


# ---------------------------------------------------------------------------
# parse_experimental_csv — unit conversion
# ---------------------------------------------------------------------------

class TestUnitConversion:
    def test_time_s_to_us(self) -> None:
        csv = "time_s,current_MA\n0.000001,0.5\n0.000002,1.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([1.0, 2.0], rel=1e-5)

    def test_current_a_to_ma(self) -> None:
        csv = "time_us,current_A\n0.0,500000.0\n1.0,1000000.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["I_MA"] == pytest.approx([0.5, 1.0], rel=1e-5)

    def test_time_s_current_a_both_converted(self) -> None:
        csv = "time_s,current_A\n0.000001,500000.0\n0.000002,1000000.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([1.0, 2.0], rel=1e-5)
        assert result["I_MA"] == pytest.approx([0.5, 1.0], rel=1e-5)

    def test_t_s_alias(self) -> None:
        csv = "t_s,I_MA\n0.000003,0.7\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([3.0], rel=1e-5)

    def test_i_a_alias(self) -> None:
        csv = "t_us,I_A\n1.0,750000.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["I_MA"] == pytest.approx([0.75], rel=1e-5)


# ---------------------------------------------------------------------------
# parse_experimental_csv — malformed input
# ---------------------------------------------------------------------------

class TestMalformedCsv:
    def test_empty_string(self) -> None:
        assert parse_experimental_csv("") is None

    def test_no_matching_columns(self) -> None:
        csv = "foo,bar\n1,2\n3,4\n"
        assert parse_experimental_csv(csv) is None

    def test_missing_current_column(self) -> None:
        csv = "time_us,voltage_kV\n0.0,27.0\n"
        assert parse_experimental_csv(csv) is None

    def test_missing_time_column(self) -> None:
        csv = "step,current_MA\n1,0.5\n"
        assert parse_experimental_csv(csv) is None

    def test_non_numeric_values_raise_handled(self) -> None:
        csv = "time_us,current_MA\nabc,def\n"
        assert parse_experimental_csv(csv) is None

    def test_empty_rows_skipped(self) -> None:
        csv = "time_us,current_MA\n0.0,0.5\n\n2.0,1.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert len(result["t_us"]) == 2

    def test_header_only_no_data(self) -> None:
        csv = "time_us,current_MA\n"
        assert parse_experimental_csv(csv) is None


# ---------------------------------------------------------------------------
# create_waveform_fig — experimental_data parameter
# ---------------------------------------------------------------------------

class TestCreateWaveformFig:
    def test_no_experimental_data(self) -> None:
        fig = create_waveform_fig(_minimal_sim_data())
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Experimental" not in names

    def test_with_experimental_data(self) -> None:
        exp = {"t_us": [0.0, 1.0, 2.0], "I_MA": [0.0, 0.4, 0.9]}
        fig = create_waveform_fig(_minimal_sim_data(), experimental_data=exp)
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Experimental" in names

    def test_experimental_trace_style(self) -> None:
        exp = {"t_us": [0.0, 1.0], "I_MA": [0.0, 1.0]}
        fig = create_waveform_fig(_minimal_sim_data(), experimental_data=exp)
        exp_trace = next(t for t in fig.data if t.name == "Experimental")
        assert exp_trace.line.color == "red"
        assert exp_trace.line.dash == "dash"

    def test_experimental_data_none_is_default(self) -> None:
        fig_no_arg = create_waveform_fig(_minimal_sim_data())
        fig_none = create_waveform_fig(_minimal_sim_data(), experimental_data=None)
        assert len(fig_no_arg.data) == len(fig_none.data)

    def test_experimental_data_on_correct_subplot(self) -> None:
        exp = {"t_us": [0.0, 1.0], "I_MA": [0.0, 1.0]}
        fig = create_waveform_fig(_minimal_sim_data(), experimental_data=exp)
        exp_trace = next(t for t in fig.data if t.name == "Experimental")
        assert exp_trace.xaxis == "x"
        assert exp_trace.yaxis == "y"
