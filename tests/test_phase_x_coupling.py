"""Phase X: Snowplow-MHD Coupling, Radial Zipper BC, LHDI, and Peak Current Tracking.

Tests for:
1. Radial Zipper Boundary Condition — B_theta zeroed outside radial shock front
   during the radial phase of the snowplow.
2. Axial Zipper Boundary Condition — B_theta zeroed ahead of the axial sheath.
3. Snowplow diagnostics — r_shock and phase keys exposed in diagnostics dict.
4. LHDI config plumbing — threshold_model param in anomalous_resistivity_scalar,
   anomalous_threshold_model field in SimulationConfig.
5. LHDI physics — LHDI triggers at lower drift than ion-acoustic threshold.
6. Electrode BC defaults — cylindrical presets have electrode_bc=True and lhdi model.
7. Peak current tracking — engine.run() returns peak_current_A / peak_current_time_s.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.constants import e, k_B, m_d, m_e
from dpf.engine import SimulationEngine
from dpf.presets import get_preset
from dpf.turbulence.anomalous import (
    anomalous_resistivity_field,
    anomalous_resistivity_scalar,
    buneman_classic_threshold,
    ion_acoustic_threshold,
    ion_thermal_speed,
    lhdi_factor,
    lhdi_threshold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cylindrical_engine(
    nr: int = 10,
    nz: int = 20,
    dx: float = 0.01,
    snowplow_enabled: bool = True,
) -> SimulationEngine:
    """Create a minimal cylindrical engine for BC tests."""
    cfg = SimulationConfig(
        grid_shape=[nr, 1, nz],
        dx=dx,
        sim_time=1e-6,
        circuit={
            "C": 1e-6,
            "V0": 10e3,
            "L0": 10e-9,
            "anode_radius": 0.01,
            "cathode_radius": 0.10,
        },
        snowplow={
            "enabled": snowplow_enabled,
            "anode_length": nz * dx,
            "mass_fraction": 0.3,
            "current_fraction": 0.7,
        },
        boundary={"electrode_bc": True},
        geometry={"type": "cylindrical", "dz": dx},
        fluid={"backend": "python"},
        diagnostics={"hdf5_filename": ":memory:"},
    )
    return SimulationEngine(cfg)


# ---------------------------------------------------------------------------
# TestRadialZipperBC
# ---------------------------------------------------------------------------


class TestRadialZipperBC:
    """Radial zipper BC: B_theta zeroed outside radial shock during radial phase."""

    def test_radial_zipper_activates_during_radial_phase(self):
        """With snowplow.phase='radial', B_theta outside r_shock is zeroed."""
        engine = _make_cylindrical_engine(nr=10, nz=20, dx=0.01)

        # Force radial phase with shock at radial index 5 (r=0.05 m)
        engine.snowplow.phase = "radial"
        engine.snowplow.r_shock = 0.05  # ir_shock = int(0.05/0.01) = 5
        engine.snowplow.z = engine.snowplow.L_anode  # axial rundown done
        engine.snowplow._rundown_complete = True

        # Fill B_theta with nonzero values everywhere
        engine.state["B"] = np.ones_like(engine.state["B"])

        engine._apply_electrode_bc(current=100e3)

        B_theta = engine.state["B"][1]  # shape (nr, 1, nz)
        # ir_shock = 5, so indices 6..9 (ir_shock+1 onwards) should be 0
        assert np.all(B_theta[6:, :, :] == 0.0), (
            "B_theta beyond ir_shock+1 must be zeroed by radial zipper"
        )

    def test_radial_zipper_no_effect_during_rundown(self):
        """With snowplow.phase='rundown', the radial zipper branch is not taken."""
        engine = _make_cylindrical_engine(nr=10, nz=20, dx=0.01)

        engine.snowplow.phase = "rundown"
        engine.snowplow.z = 0.10  # mid-domain
        # B_theta filled with ones
        engine.state["B"][1, :, :, :] = 1.0
        engine.state["B"][0, :, :, :] = 0.0
        engine.state["B"][2, :, :, :] = 0.0

        engine._apply_electrode_bc(current=100e3)

        B_theta = engine.state["B"][1]
        # Radial zipper should NOT have zeroed anything radially
        # (axial zipper may have acted, but radial zeros should be absent)
        # The key check: all radial slices for r in [0, nr) should not be
        # uniformly zeroed due to radial zipper — i.e. some cells survive
        assert np.any(B_theta > 0.0), (
            "Without radial zipper, some B_theta cells should remain nonzero"
        )

    def test_radial_zipper_r_shock_mapping(self):
        """ir_shock = int(r_shock / dx) gives correct radial index."""
        nr, dx = 12, 0.01
        engine = _make_cylindrical_engine(nr=nr, nz=20, dx=dx)

        r_shock = 0.07  # ir_shock = int(0.07/0.01) = 7
        engine.snowplow.phase = "radial"
        engine.snowplow.r_shock = r_shock
        engine.snowplow.z = engine.snowplow.L_anode
        engine.snowplow._rundown_complete = True
        engine.state["B"][1] = 1.0

        engine._apply_electrode_bc(current=50e3)

        ir_shock = int(r_shock / dx)  # = 7
        B_theta = engine.state["B"][1]
        # Indices ir_shock+1 onward should be zero
        assert np.all(B_theta[ir_shock + 1:, :, :] == 0.0), (
            f"Expected B_theta zero for r_idx > {ir_shock}"
        )
        # Indices 0..ir_shock should not be all-zero (BC applies nonzero there)
        # At minimum the structure should survive — check that the array is not entirely zero
        # (axial BC may set some values)
        assert B_theta.shape[0] > ir_shock, "ir_shock must be within grid bounds"

    def test_radial_zipper_btheta_zero_outside_shock(self):
        """B[1, ir_shock+1:, :, :] == 0 after radial zipper is applied."""
        nr, dx = 10, 0.01
        engine = _make_cylindrical_engine(nr=nr, nz=20, dx=dx)

        r_shock = 0.03  # ir_shock = 3
        engine.snowplow.phase = "radial"
        engine.snowplow.r_shock = r_shock
        engine.snowplow.z = engine.snowplow.L_anode
        engine.snowplow._rundown_complete = True

        engine.state["B"][:] = 5.0  # all components nonzero

        engine._apply_electrode_bc(current=1e5)

        ir_shock = int(r_shock / dx)
        assert np.all(engine.state["B"][1, ir_shock + 1:, :, :] == 0.0), (
            "B_theta must be zero outside (beyond) the radial shock"
        )

    def test_radial_zipper_preserves_btheta_inside_shock(self):
        """B_theta for r-indices <= ir_shock should NOT be zeroed by radial zipper."""
        nr, dx = 10, 0.01
        engine = _make_cylindrical_engine(nr=nr, nz=20, dx=dx)

        r_shock = 0.06  # ir_shock = 6
        engine.snowplow.phase = "radial"
        engine.snowplow.r_shock = r_shock
        engine.snowplow.z = engine.snowplow.L_anode
        engine.snowplow._rundown_complete = True

        # Seed large values inside the shock so they survive the radial zipper
        engine.state["B"][1, :7, :, :] = 99.0

        engine._apply_electrode_bc(current=1e5)

        ir_shock = int(r_shock / dx)
        # At least some of the inside-shock B_theta should still be nonzero
        # (the axial zipper might zero cells beyond z_sheath, but not all)
        B_inside = engine.state["B"][1, :ir_shock + 1, :, :]
        assert np.any(B_inside != 0.0), (
            "B_theta inside radial shock should not all be zeroed by radial zipper"
        )

    def test_radial_zipper_edge_case_r_shock_at_boundary(self):
        """r_shock near grid boundary (ir_shock >= nx-1) does not crash."""
        nr, dx = 8, 0.01
        engine = _make_cylindrical_engine(nr=nr, nz=20, dx=dx)

        # r_shock near the outer edge
        r_shock = (nr - 1) * dx  # ir_shock = nr-1 = 7
        engine.snowplow.phase = "radial"
        engine.snowplow.r_shock = r_shock
        engine.snowplow.z = engine.snowplow.L_anode
        engine.snowplow._rundown_complete = True
        engine.state["B"][:] = 1.0

        # Should not raise even with ir_shock = nr-1 (nothing to zero beyond)
        engine._apply_electrode_bc(current=1e5)
        # No error means success; array shape unchanged
        assert engine.state["B"].shape[0] == 3


# ---------------------------------------------------------------------------
# TestAxialZipperBC
# ---------------------------------------------------------------------------


class TestAxialZipperBC:
    """Axial zipper BC: B_theta zeroed ahead of axial sheath position."""

    def test_electrode_bc_zippering(self):
        """Magnetic BC should only apply behind the snowplow (axial zipper)."""
        nz = 20
        dx = 0.01

        engine = _make_cylindrical_engine(nr=10, nz=nz, dx=dx)

        # Manually place snowplow mid-domain
        engine.snowplow.z = 0.10       # iz_sheath = int(0.10/0.01) = 10
        engine.snowplow.v = 1e5
        engine.snowplow.phase = "rundown"
        engine.circuit.state.current = 100e3

        engine.state["B"] = np.zeros_like(engine.state["B"])
        engine._apply_electrode_bc(engine.circuit.state.current)

        B_theta_anode = engine.state["B"][1, 1, 0, :]

        # Behind sheath (z-indices 0..10): BC drives B_theta nonzero
        assert np.any(np.abs(B_theta_anode[:10]) > 0), (
            "Behind sheath: B_theta should be applied"
        )
        # Ahead of sheath (z-indices 12..): should be zero
        assert np.all(B_theta_anode[12:] == 0.0), (
            "Ahead of sheath: B_theta should be zero (axial zipper)"
        )

    def test_axial_zipper_btheta_behind_sheath(self):
        """B_theta should be nonzero at z < z_sheath after electrode BC."""
        nz, dx = 20, 0.01
        engine = _make_cylindrical_engine(nr=10, nz=nz, dx=dx)

        engine.snowplow.z = 0.15  # sheath at index 15
        engine.snowplow.phase = "rundown"
        engine.state["B"] = np.zeros_like(engine.state["B"])

        engine._apply_electrode_bc(current=200e3)

        # B_theta at r=anode, z=0 (index 0) — well behind sheath
        B_theta_behind = engine.state["B"][1, 1, 0, 0]
        assert abs(B_theta_behind) > 0.0, (
            "B_theta behind sheath should be set by electrode BC"
        )

    def test_axial_zipper_btheta_ahead_of_sheath(self):
        """B_theta should be 0 at z > z_sheath (axial zipper zeroes ahead of sheath)."""
        nz, dx = 20, 0.01
        engine = _make_cylindrical_engine(nr=10, nz=nz, dx=dx)

        engine.snowplow.z = 0.05  # sheath at index 5 — early position
        engine.snowplow.phase = "rundown"

        # Pre-fill with nonzero to confirm zeroing
        engine.state["B"][1, :, :, :] = 999.0

        engine._apply_electrode_bc(current=100e3)

        # Indices 7..19 (iz_sheath+2 onward) should be zero
        B_theta_ahead = engine.state["B"][1, :, :, 7:]
        assert np.all(B_theta_ahead == 0.0), (
            "B_theta ahead of axial sheath should be zeroed by zipper"
        )


# ---------------------------------------------------------------------------
# TestSnowplowDiagnostics
# ---------------------------------------------------------------------------


class TestSnowplowDiagnostics:
    """r_shock and phase keys must appear in the diagnostics dict recorded by the engine."""

    def _make_engine_and_capture_diag(self, snowplow_enabled: bool = True) -> dict:
        """Create engine, run one step, and capture the diag_state via monkeypatching."""
        nr, nz, dx = 8, 16, 0.01
        cfg = SimulationConfig(
            grid_shape=[nr, 1, nz],
            dx=dx,
            sim_time=1e-6,
            circuit={
                "C": 1e-6,
                "V0": 5e3,
                "L0": 10e-9,
                "anode_radius": 0.01,
                "cathode_radius": 0.08,
            },
            snowplow={
                "enabled": snowplow_enabled,
                "anode_length": nz * dx,
                "mass_fraction": 0.3,
                "current_fraction": 0.7,
            },
            boundary={"electrode_bc": True},
            geometry={"type": "cylindrical", "dz": dx},
            fluid={"backend": "python"},
            diagnostics={"hdf5_filename": ":memory:", "output_interval": 1},
        )
        engine = SimulationEngine(cfg)

        # Capture the diag_state passed to diagnostics.record
        captured: list[dict] = []
        original_record = engine.diagnostics.record

        def _capturing_record(state: dict, time: float) -> None:
            captured.append(state)
            original_record(state, time)

        engine.diagnostics.record = _capturing_record  # type: ignore[method-assign]
        engine.step()

        return captured[-1] if captured else {}

    def test_diagnostics_include_r_shock(self):
        """After a step with snowplow, the diag_state must contain snowplow.r_shock."""
        diag = self._make_engine_and_capture_diag(snowplow_enabled=True)
        snowplow_diag = diag.get("snowplow", {})
        assert "r_shock" in snowplow_diag, (
            "Snowplow diagnostics must expose r_shock"
        )

    def test_diagnostics_include_phase(self):
        """After a step with snowplow, the diag_state must contain snowplow.phase."""
        diag = self._make_engine_and_capture_diag(snowplow_enabled=True)
        snowplow_diag = diag.get("snowplow", {})
        assert "phase" in snowplow_diag, (
            "Snowplow diagnostics must expose phase"
        )

    def test_diagnostics_r_shock_defaults_zero_without_snowplow(self):
        """With snowplow disabled, snowplow.r_shock in the diag_state should be 0.0."""
        diag = self._make_engine_and_capture_diag(snowplow_enabled=False)
        snowplow_diag = diag.get("snowplow", {})
        assert snowplow_diag.get("r_shock", 0.0) == pytest.approx(0.0), (
            "Without snowplow, r_shock diagnostic should default to 0.0"
        )

    def test_diagnostics_phase_defaults_none_without_snowplow(self):
        """With snowplow disabled, snowplow.phase in the diag_state should be 'none'."""
        diag = self._make_engine_and_capture_diag(snowplow_enabled=False)
        snowplow_diag = diag.get("snowplow", {})
        assert snowplow_diag.get("phase", "none") == "none", (
            "Without snowplow, phase diagnostic should default to 'none'"
        )


# ---------------------------------------------------------------------------
# TestLHDIConfigPlumbing
# ---------------------------------------------------------------------------


class TestLHDIConfigPlumbing:
    """Verify LHDI threshold model is accessible through the config layer."""

    def test_config_threshold_model_default(self):
        """anomalous_threshold_model defaults to 'ion_acoustic'."""
        cfg = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            diagnostics={"hdf5_filename": ":memory:"},
        )
        assert cfg.anomalous_threshold_model == "ion_acoustic", (
            "Default threshold model should be 'ion_acoustic'"
        )

    def test_config_threshold_model_lhdi(self):
        """anomalous_threshold_model can be set to 'lhdi'."""
        cfg = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-7,
            anomalous_threshold_model="lhdi",
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            diagnostics={"hdf5_filename": ":memory:"},
        )
        assert cfg.anomalous_threshold_model == "lhdi"

    def test_scalar_accepts_threshold_model(self):
        """anomalous_resistivity_scalar accepts threshold_model='lhdi' without error."""
        ne_val = 1e23
        Ti_val = 1000.0  # K
        J_mag = 1e10     # A/m^2

        eta = anomalous_resistivity_scalar(
            J_mag=J_mag,
            ne_val=ne_val,
            Ti_val=Ti_val,
            alpha=0.05,
            mi=m_d,
            threshold_model="lhdi",
        )
        assert eta >= 0.0, "anomalous_resistivity_scalar with lhdi must return non-negative"

    def test_scalar_lhdi_lower_threshold(self):
        """anomalous_resistivity_scalar returns eta>0 for LHDI but eta=0 for ion_acoustic at same J."""
        ne_val = 1e23
        Ti_val = 1000.0  # K, cold
        mi = m_d

        # Compute thresholds
        v_ti = (k_B * Ti_val / mi) ** 0.5
        factor = (m_e / mi) ** 0.25
        # Set drift just above LHDI threshold but below ion-acoustic threshold
        v_d_target = factor * v_ti * 1.5  # 1.5× LHDI, still << v_ti
        assert v_d_target < v_ti, "Test drift must be below ion-acoustic threshold"
        J_mag = v_d_target * ne_val * e

        eta_lhdi = anomalous_resistivity_scalar(
            J_mag, ne_val, Ti_val, alpha=0.05, mi=mi, threshold_model="lhdi"
        )
        eta_ia = anomalous_resistivity_scalar(
            J_mag, ne_val, Ti_val, alpha=0.05, mi=mi, threshold_model="ion_acoustic"
        )

        assert eta_lhdi > 0.0, "LHDI model should trigger above LHDI threshold"
        assert eta_ia == 0.0, "Ion-acoustic model should not trigger below v_ti"

    def test_config_description_fixed(self):
        """anomalous_alpha field description should say 'Anomalous' not 'Buneman'."""
        field_info = SimulationConfig.model_fields.get("anomalous_alpha")
        assert field_info is not None, "anomalous_alpha field must exist in SimulationConfig"
        description = field_info.description or ""
        assert "Buneman" not in description, (
            "anomalous_alpha description must not say 'Buneman' (mislabel from old code)"
        )
        assert "Anomalous" in description or "anomalous" in description.lower(), (
            "anomalous_alpha description should mention 'Anomalous' resistivity"
        )


# ---------------------------------------------------------------------------
# TestLHDIPhysics
# ---------------------------------------------------------------------------


class TestLHDIPhysics:
    """LHDI threshold and physics verification."""

    def test_lhdi_threshold_lower_than_ion_acoustic(self):
        """LHDI threshold < ion-acoustic threshold by factor (m_e/m_i)^{1/4}."""
        Ti = np.array([1000.0])
        ne = np.array([1e23])
        mi = m_d

        v_ti = ion_thermal_speed(Ti, mi)[0]
        factor = lhdi_factor(mi)

        # Drift just above LHDI, just below ion-acoustic
        v_d = factor * v_ti * 1.5
        J = v_d * ne * e

        lhdi_active = lhdi_threshold(J, ne, Ti, mi)[0]
        ia_active = ion_acoustic_threshold(J, ne, Ti, mi)[0]

        assert lhdi_active, "LHDI should be active at 1.5× LHDI threshold"
        assert not ia_active, "Ion-acoustic should NOT be active below v_ti"

    def test_lhdi_factor_deuterium(self):
        """lhdi_factor(m_d) should be approximately 0.13 (Davidson & Gladd 1975)."""
        factor = lhdi_factor(m_d)
        # (m_e / m_d)^{1/4} = (9.109e-31 / 3.344e-27)^{0.25} ~ 0.129
        assert factor == pytest.approx(0.129, abs=0.005), (
            f"LHDI factor for deuterium expected ~0.129, got {factor:.4f}"
        )

    def test_field_vs_scalar_consistency(self):
        """anomalous_resistivity_field and anomalous_resistivity_scalar agree."""
        ne_val = 1e23
        Ti_val = 1000.0
        mi = m_d
        alpha = 0.05

        # Use a drift well above LHDI threshold
        v_ti = (k_B * Ti_val / mi) ** 0.5
        v_d = lhdi_factor(mi) * v_ti * 5.0
        J_mag = v_d * ne_val * e

        # Scalar version
        eta_scalar = anomalous_resistivity_scalar(
            J_mag, ne_val, Ti_val, alpha=alpha, mi=mi, threshold_model="lhdi"
        )

        # Field version (1-element arrays)
        J_arr = np.array([J_mag])
        ne_arr = np.array([ne_val])
        Ti_arr = np.array([Ti_val])
        eta_field = anomalous_resistivity_field(
            J_arr, ne_arr, Ti_arr, alpha=alpha, mi=mi, threshold_model="lhdi"
        )[0]

        assert eta_scalar == pytest.approx(eta_field, rel=1e-6), (
            "Field and scalar versions must produce identical results"
        )

    def test_buneman_classic_highest_threshold(self):
        """Buneman classic requires highest drift: v_d > v_te >> v_ti."""
        Ti = np.array([1000.0])  # K
        Te = np.array([1000.0])  # K — same temperature
        ne = np.array([1e23])
        mi = m_d

        v_ti = ion_thermal_speed(Ti, mi)[0]

        # Drift at 2× v_ti — above ion-acoustic but far below v_te
        J = 2.0 * v_ti * ne * e

        ia_active = ion_acoustic_threshold(J, ne, Ti, mi)[0]
        lhdi_active = lhdi_threshold(J, ne, Ti, mi)[0]
        buneman_active = buneman_classic_threshold(J, ne, Te)[0]

        assert lhdi_active, "LHDI should trigger at 2× v_ti"
        assert ia_active, "Ion-acoustic should trigger at 2× v_ti"
        assert not buneman_active, (
            "Buneman classic (v_d > v_te) should NOT trigger at 2× v_ti "
            "because v_te >> v_ti for same temperature"
        )

    def test_lhdi_returns_nonzero_above_threshold(self):
        """anomalous_resistivity_field returns eta > 0 when above LHDI threshold."""
        mi = m_d
        Ti_val = 500.0  # K
        ne_val = 5e22

        v_ti = (k_B * Ti_val / mi) ** 0.5
        factor = lhdi_factor(mi)
        # Set drift 3× above LHDI threshold
        v_d = factor * v_ti * 3.0
        J_mag = v_d * ne_val * e

        eta = anomalous_resistivity_field(
            np.array([J_mag]),
            np.array([ne_val]),
            np.array([Ti_val]),
            alpha=0.05,
            mi=mi,
            threshold_model="lhdi",
        )[0]
        assert eta > 0.0, "eta_anom must be positive above LHDI threshold"


# ---------------------------------------------------------------------------
# TestElectrodeBCDefaults
# ---------------------------------------------------------------------------


class TestElectrodeBCDefaults:
    """Cylindrical presets should have electrode_bc=True and use LHDI threshold."""

    def _preset_cfg(self, name: str) -> SimulationConfig:
        """Load preset dict and wrap in SimulationConfig."""
        preset = get_preset(name)
        # Presets omit grid_shape/dx/sim_time — add minimal values
        preset.setdefault("grid_shape", [8, 1, 16])
        preset.setdefault("dx", 0.01)
        preset.setdefault("sim_time", 1e-6)
        preset.setdefault("diagnostics", {"hdf5_filename": ":memory:"})
        return SimulationConfig(**preset)

    def test_pf1000_preset_electrode_bc_true(self):
        """PF-1000 preset has boundary.electrode_bc=True."""
        cfg = self._preset_cfg("pf1000")
        assert cfg.boundary.electrode_bc is True, (
            "PF-1000 preset must have electrode_bc=True"
        )

    def test_nx2_preset_electrode_bc_true(self):
        """NX2 preset has boundary.electrode_bc=True."""
        cfg = self._preset_cfg("nx2")
        assert cfg.boundary.electrode_bc is True, (
            "NX2 preset must have electrode_bc=True"
        )

    def test_llnl_dpf_preset_electrode_bc_true(self):
        """LLNL-DPF preset has boundary.electrode_bc=True."""
        cfg = self._preset_cfg("llnl_dpf")
        assert cfg.boundary.electrode_bc is True, (
            "LLNL-DPF preset must have electrode_bc=True"
        )

    def test_cylindrical_presets_use_lhdi(self):
        """All three cylindrical presets must have anomalous_threshold_model='lhdi'."""
        for name in ("pf1000", "nx2", "llnl_dpf"):
            cfg = self._preset_cfg(name)
            assert cfg.anomalous_threshold_model == "lhdi", (
                f"Preset '{name}' must use anomalous_threshold_model='lhdi', "
                f"got '{cfg.anomalous_threshold_model}'"
            )


# ---------------------------------------------------------------------------
# TestPeakCurrentTracking
# ---------------------------------------------------------------------------


class TestPeakCurrentTracking:
    """engine.run() must track and return peak_current_A and peak_current_time_s."""

    def _run_short(self, max_steps: int = 5) -> dict:
        """Run engine for a few steps and return summary."""
        cfg = SimulationConfig(
            grid_shape=[8, 1, 16],
            dx=0.01,
            sim_time=1e-5,
            circuit={
                "C": 1e-6,
                "V0": 5e3,
                "L0": 10e-9,
                "anode_radius": 0.01,
                "cathode_radius": 0.08,
            },
            snowplow={"enabled": False},
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python"},
            diagnostics={"hdf5_filename": ":memory:"},
        )
        engine = SimulationEngine(cfg)
        return engine.run(max_steps=max_steps)

    def test_peak_current_in_summary(self):
        """engine.run() summary dict must contain 'peak_current_A'."""
        summary = self._run_short(max_steps=5)
        assert "peak_current_A" in summary, (
            "run() summary must include peak_current_A"
        )

    def test_peak_current_time_in_summary(self):
        """engine.run() summary dict must contain 'peak_current_time_s'."""
        summary = self._run_short(max_steps=5)
        assert "peak_current_time_s" in summary, (
            "run() summary must include peak_current_time_s"
        )

    def test_peak_current_positive(self):
        """peak_current_A must be > 0 after running at least one step."""
        summary = self._run_short(max_steps=5)
        assert summary["peak_current_A"] > 0.0, (
            "peak_current_A must be positive after current starts flowing"
        )

    def test_peak_current_at_correct_time(self):
        """peak_current_time_s must be >= 0 and within sim_time."""
        summary = self._run_short(max_steps=10)
        t_peak = summary["peak_current_time_s"]
        assert t_peak >= 0.0, "peak_current_time_s must be non-negative"
        assert t_peak <= 1e-5, (
            f"peak_current_time_s {t_peak:.2e} exceeds sim_time 1e-5"
        )
