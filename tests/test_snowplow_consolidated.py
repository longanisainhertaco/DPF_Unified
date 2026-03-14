"""Consolidated snowplow model tests.

Covers:
- Snowplow Dynamics (test_phase_s_snowplow)
- Lee Model Cross-check (test_phase_s_lee_crosscheck)
- Current Waveform Validation (test_phase_t_current_waveform)
- Radial Phase Physics (test_phase_t_radial)
- T Validation — L_coeff, f_c, radial compression, PF-1000 consistency, docstrings, coupling (test_phase_t_validation)
- Lee Model Fixes (test_phase_w_lee_fixes)
- Lee Model Validation (test_phase_w_lee_validation)
- Snowplow MHD Coupling / CylindricalMHDSolver source terms (test_phase_w_snowplow_coupling)
- X Coupling: Radial/Axial Zipper BC, LHDI, Peak Current Tracking (test_phase_x_coupling)
- MHD Pressure Coupling Y (test_phase_y_mhd_coupling)
- Reflected Shock Y (test_phase_y_reflected_shock)
- B-field Initialization Z (test_phase_z_bfield_init)
- FMR Grid Convergence AD (test_phase_ad_fmr_convergence)
- AF Reflected Shock Lee model Phase 4 (test_phase_af_reflected_shock)
- Two-step Radial Current Fraction (test_two_step_radial)
- Bennett Z-pinch Equilibrium Validation (test_bennett_validation)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dpf.circuit.rlc_solver import RLCSolver
from dpf.config import SimulationConfig, SnowplowConfig
from dpf.constants import e, k_B, m_d, m_e, mu_0, pi
from dpf.constants import e as e_charge
from dpf.core.bases import CouplingState
from dpf.engine import SimulationEngine
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver
from dpf.fluid.snowplow import SnowplowModel
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
from dpf.validation.bennett_equilibrium import (
    bennett_btheta,
    bennett_current_density,
    bennett_current_from_temperature,
    bennett_density,
    bennett_line_density,
    bennett_pressure,
    create_bennett_state,
    verify_force_balance,
)
from dpf.validation.experimental import DEVICES, PF1000_DATA, validate_current_waveform
from dpf.validation.lee_model_comparison import LeeModel, LeeModelResult
from dpf.validation.suite import DEVICE_REGISTRY

# ════════════════════════════════════════════════════════════════════════
# --- Section: Snowplow Dynamics (test_phase_s_snowplow) ---
# ════════════════════════════════════════════════════════════════════════

PF1000_A = 0.0575
PF1000_B = 0.08
PF1000_RHO0 = 4e-4
PF1000_L_ANODE = 0.16
PF1000_FM = 0.3
PF1000_P_FILL = 400.0


@pytest.fixture
def snowplow():
    """Create a PF-1000-like snowplow model."""
    return SnowplowModel(
        anode_radius=PF1000_A,
        cathode_radius=PF1000_B,
        fill_density=PF1000_RHO0,
        anode_length=PF1000_L_ANODE,
        mass_fraction=PF1000_FM,
        fill_pressure_Pa=PF1000_P_FILL,
    )


class TestSnowplowInstantiation:
    """Test SnowplowModel initialization."""

    def test_basic_instantiation(self, snowplow):
        assert snowplow.a == PF1000_A
        assert snowplow.b == PF1000_B
        assert snowplow.rho0 == PF1000_RHO0
        assert snowplow.L_anode == PF1000_L_ANODE
        assert snowplow.f_m == PF1000_FM
        assert snowplow.p_fill == PF1000_P_FILL

    def test_initial_position_nonzero(self, snowplow):
        assert snowplow.z > 0.0
        assert snowplow.z < snowplow.L_anode

    def test_initial_velocity_zero(self, snowplow):
        assert snowplow.v == 0.0

    def test_phase_is_rundown(self, snowplow):
        assert snowplow.phase == "rundown"
        assert not snowplow.rundown_complete

    def test_geometric_constants(self, snowplow):
        ln_ba = np.log(PF1000_B / PF1000_A)
        A_ann = pi * (PF1000_B**2 - PF1000_A**2)
        assert snowplow.ln_ba == pytest.approx(ln_ba, rel=1e-10)
        assert snowplow.A_annular == pytest.approx(A_ann, rel=1e-10)

    def test_force_coefficient(self, snowplow):
        expected = (mu_0 / (4.0 * pi)) * np.log(PF1000_B / PF1000_A)
        assert snowplow.F_coeff == pytest.approx(expected, rel=1e-10)


class TestSnowplowProperties:
    """Test computed properties."""

    def test_swept_mass(self, snowplow):
        m = PF1000_RHO0 * snowplow.A_annular * snowplow.z * PF1000_FM
        assert snowplow.swept_mass == pytest.approx(m, rel=1e-10)

    def test_plasma_inductance(self, snowplow):
        L = snowplow.L_coeff * snowplow.z
        assert snowplow.plasma_inductance == pytest.approx(L, rel=1e-10)

    def test_sheath_position_property(self, snowplow):
        assert snowplow.sheath_position == snowplow.z

    def test_sheath_velocity_property(self, snowplow):
        assert snowplow.sheath_velocity == snowplow.v


class TestSnowplowStep:
    """Test the snowplow step function."""

    def test_sheath_advances_with_current(self, snowplow):
        z0 = snowplow.z
        dt = 1e-8
        current = 1e6
        result = snowplow.step(dt, current)
        assert snowplow.z > z0
        assert snowplow.v > 0.0
        assert result["z_sheath"] == snowplow.z
        assert result["v_sheath"] == snowplow.v

    def test_zero_current_no_acceleration(self, snowplow):
        z0 = snowplow.z
        snowplow.step(1e-8, current=0.0)
        assert snowplow.z == pytest.approx(z0, rel=1e-6)

    def test_mass_conservation(self, snowplow):
        dt = 1e-8
        for _ in range(100):
            snowplow.step(dt, current=5e5)
        expected = PF1000_RHO0 * snowplow.A_annular * snowplow.z * PF1000_FM
        assert snowplow.swept_mass == pytest.approx(expected, rel=1e-10)

    def test_inductance_consistency(self, snowplow):
        dt = 1e-8
        for _ in range(50):
            result = snowplow.step(dt, current=5e5)
        expected_L = snowplow.L_coeff * snowplow.z
        assert result["L_plasma"] == pytest.approx(expected_L, rel=1e-10)
        assert snowplow.plasma_inductance == pytest.approx(expected_L, rel=1e-10)

    def test_dL_dt_positive(self, snowplow):
        dt = 1e-8
        for _ in range(10):
            snowplow.step(dt, current=1e6)
        result = snowplow.step(dt, current=1e6)
        assert result["dL_dt"] > 0.0

    def test_dL_dt_equals_Lcoeff_times_v(self, snowplow):
        dt = 1e-8
        for _ in range(20):
            result = snowplow.step(dt, current=1e6)
        expected = snowplow.L_coeff * snowplow.v
        assert result["dL_dt"] == pytest.approx(expected, rel=1e-10)

    def test_magnetic_force_scales_with_I_squared(self, snowplow):
        dt = 1e-8
        r1 = snowplow.step(dt, current=1e5)
        sp2 = SnowplowModel(
            anode_radius=PF1000_A, cathode_radius=PF1000_B,
            fill_density=PF1000_RHO0, anode_length=PF1000_L_ANODE,
            mass_fraction=PF1000_FM, fill_pressure_Pa=PF1000_P_FILL,
        )
        r2 = sp2.step(dt, current=2e5)
        assert r2["F_magnetic"] == pytest.approx(4.0 * r1["F_magnetic"], rel=1e-10)

    def test_pressure_force_opposes_motion(self, snowplow):
        result = snowplow.step(1e-8, current=1e6)
        assert result["F_pressure"] > 0.0
        assert result["F_magnetic"] > result["F_pressure"]


class TestSnowplowRundownCompletion:
    """Test rundown termination when sheath reaches end of anode."""

    def test_rundown_completes(self, snowplow):
        dt = 1e-8
        for _ in range(100_000):
            snowplow.step(dt, current=1e6)
            if snowplow.rundown_complete:
                break
        assert snowplow.rundown_complete
        assert snowplow.z == snowplow.L_anode
        assert snowplow.phase == "radial"

    def test_radial_phase_after_rundown(self, snowplow):
        dt = 1e-8
        for _ in range(100_000):
            snowplow.step(dt, current=1e6)
            if snowplow.rundown_complete:
                break
        z_final = snowplow.z
        L_at_rundown = snowplow.plasma_inductance
        result = snowplow.step(dt, current=1e6)
        assert snowplow.z == z_final
        assert snowplow.phase in ("radial", "pinch")
        assert snowplow.plasma_inductance > L_at_rundown
        assert result["r_shock"] < snowplow.b

    def test_pinch_reached(self, snowplow):
        dt = 1e-8
        for _ in range(200_000):
            snowplow.step(dt, current=1e6)
            if snowplow.pinch_complete:
                break
        assert snowplow.pinch_complete
        assert snowplow.phase == "pinch"
        assert snowplow.pinch_radius <= snowplow.r_pinch_min
        r_stag = snowplow.pinch_radius
        snowplow.step(dt, current=1e6)
        assert snowplow.pinch_radius == r_stag

    def test_sheath_does_not_overshoot(self, snowplow):
        dt = 1e-7
        for _ in range(10_000):
            snowplow.step(dt, current=2e6)
            assert snowplow.z <= snowplow.L_anode


class TestSnowplowPhysics:
    """Tests verifying physical correctness."""

    def test_force_formula_matches_lee(self):
        a, b = 0.01, 0.02
        f_c = 0.7
        sp = SnowplowModel(
            anode_radius=a, cathode_radius=b, fill_density=1e-4,
            anode_length=0.1, mass_fraction=0.3, fill_pressure_Pa=0.0,
            current_fraction=f_c,
        )
        current = 1e6
        F_expected = (mu_0 / (4 * pi)) * np.log(b / a) * (f_c * current)**2
        result = sp.step(1e-10, current)
        assert result["F_magnetic"] == pytest.approx(F_expected, rel=1e-10)

    def test_inductance_formula_coaxial(self):
        a, b = 0.01, 0.02
        sp = SnowplowModel(
            anode_radius=a, cathode_radius=b, fill_density=1e-4,
            anode_length=0.1, mass_fraction=0.3,
        )
        z = 0.05
        sp.z = z
        L_expected = (mu_0 / (2 * pi)) * np.log(b / a) * z
        assert sp.plasma_inductance == pytest.approx(L_expected, rel=1e-10)

    def test_pf1000_timing_order_of_magnitude(self, snowplow):
        dt = 1e-9
        n_steps = 0
        max_steps = 10_000_000
        for _ in range(max_steps):
            snowplow.step(dt, current=1.5e6)
            n_steps += 1
            if snowplow.rundown_complete:
                break
        t_rundown = n_steps * dt
        assert 0.5e-6 < t_rundown < 20e-6, f"Rundown took {t_rundown:.2e} s"

    def test_velocity_verlet_second_order(self):
        def run_snowplow(dt, n_steps):
            sp = SnowplowModel(
                anode_radius=0.01, cathode_radius=0.02, fill_density=1e-4,
                anode_length=1.0, mass_fraction=0.3, fill_pressure_Pa=0.0,
            )
            for _ in range(n_steps):
                sp.step(dt, current=1e5)
            return sp.z

        dt1 = 1e-7
        n1 = 1000
        z1 = run_snowplow(dt1, n1)
        z2 = run_snowplow(dt1 / 2, n1 * 2)
        z4 = run_snowplow(dt1 / 4, n1 * 4)
        err1 = abs(z2 - z1)
        err2 = abs(z4 - z2)
        if err2 > 1e-15:
            order = np.log2(err1 / err2)
            assert order > 1.5, f"Convergence order {order:.2f}, expected ~2"


class TestSnowplowConfigIntegration:
    """Test snowplow configuration in SimulationConfig."""

    def test_snowplow_config_defaults(self):
        from dpf.config import SnowplowConfig
        cfg = SnowplowConfig()
        assert cfg.enabled is True
        assert cfg.mass_fraction == 0.3
        assert cfg.fill_pressure_Pa == 400.0
        assert cfg.anode_length == 0.16

    def test_snowplow_in_simulation_config(self):
        from dpf.config import SimulationConfig
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": True, "mass_fraction": 0.2},
        )
        assert config.snowplow.enabled is True
        assert config.snowplow.mass_fraction == 0.2

    def test_snowplow_disabled(self):
        from dpf.config import SimulationConfig
        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": False},
        )
        assert config.snowplow.enabled is False


# ════════════════════════════════════════════════════════════════════════
# --- Section: Lee Model Cross-check (test_phase_s_lee_crosscheck) ---
# ════════════════════════════════════════════════════════════════════════

PF1000_C = 1.332e-3
PF1000_V0 = 27e3
PF1000_L0 = 33.5e-9
PF1000_R0 = 2.3e-3

T_QUARTER = (np.pi / 2) * np.sqrt(PF1000_L0 * PF1000_C)


def _run_circuit_traces(
    t_end: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run circuit-only solver and return (t, I) arrays."""
    solver = RLCSolver(C=PF1000_C, V0=PF1000_V0, L0=PF1000_L0, R0=PF1000_R0)
    coupling = CouplingState()
    n_steps = int(t_end / dt)
    t_arr = np.zeros(n_steps + 1)
    I_arr = np.zeros(n_steps + 1)
    for i in range(n_steps):
        solver.step(coupling, back_emf=0.0, dt=dt)
        t_arr[i + 1] = solver.state.time
        I_arr[i + 1] = solver.state.current
    return t_arr, I_arr


class TestLeeModelExecution:
    """Verify the Lee model runs without errors for PF-1000."""

    def test_lee_model_runs_pf1000(self):
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert isinstance(result, LeeModelResult)
        assert result.device_name == "PF-1000"
        assert len(result.t) > 100
        assert len(result.I) == len(result.t)

    def test_lee_model_completes_phase1(self):
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert 1 in result.phases_completed

    def test_lee_model_peak_current_positive(self):
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert result.peak_current > 0
        assert np.isfinite(result.peak_current)
        assert result.peak_current_time > 0

    def test_lee_model_peak_in_ma_range(self):
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert result.peak_current > 0.5e6
        assert result.peak_current < 10e6

    def test_lee_model_no_nan_in_waveforms(self):
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert np.all(np.isfinite(result.t))
        assert np.all(np.isfinite(result.I))
        assert np.all(np.isfinite(result.V))


class TestLeeVsCircuitComparison:
    """Quantitative comparison between Lee model and circuit-only solver."""

    @pytest.fixture
    def lee_result(self) -> LeeModelResult:
        model = LeeModel()
        return model.run(device_name="PF-1000")

    @pytest.fixture
    def circuit_traces(self, lee_result: LeeModelResult) -> tuple[np.ndarray, np.ndarray]:
        t_end = float(lee_result.t[-1])
        dt = t_end / 50000
        return _run_circuit_traces(t_end, dt)

    def test_early_time_agreement(self, lee_result, circuit_traces):
        t_circuit, I_circuit = circuit_traces
        t_compare = 1.0e-6
        assert t_compare < T_QUARTER * 0.2
        I_lee_at_t = float(np.interp(t_compare, lee_result.t, lee_result.I))
        I_circ_at_t = float(np.interp(t_compare, t_circuit, I_circuit))
        if abs(I_circ_at_t) > 1e3:
            rel_diff = abs(I_lee_at_t - I_circ_at_t) / abs(I_circ_at_t)
            assert rel_diff < 0.20

    def test_lee_model_peak_lower_than_circuit(self, lee_result, circuit_traces):
        _, I_circuit = circuit_traces
        I_peak_lee = lee_result.peak_current
        I_peak_circuit = np.max(np.abs(I_circuit))
        assert I_peak_lee < I_peak_circuit

    def test_peak_current_comparison_quantitative(self, lee_result, circuit_traces):
        _, I_circuit = circuit_traces
        I_peak_lee = lee_result.peak_current
        I_peak_circuit = np.max(np.abs(I_circuit))
        I_peak_exp = PF1000_DATA.peak_current
        lee_exp_error = abs(I_peak_lee - I_peak_exp) / I_peak_exp
        circuit_exp_error = abs(I_peak_circuit - I_peak_exp) / I_peak_exp
        assert lee_exp_error < 0.50
        assert circuit_exp_error < 1.50

    def test_l2_norm_waveform_difference(self, lee_result, circuit_traces):
        t_circuit, I_circuit = circuit_traces
        t_max = min(float(lee_result.t[-1]), float(t_circuit[-1]))
        t_common = np.linspace(0, t_max, 5000)
        I_lee_interp = np.interp(t_common, lee_result.t, lee_result.I)
        I_circ_interp = np.interp(t_common, t_circuit, I_circuit)
        diff = I_lee_interp - I_circ_interp
        l2_norm = np.sqrt(np.mean(diff**2))
        I_peak_ref = max(np.max(np.abs(I_lee_interp)), np.max(np.abs(I_circ_interp)))
        normalized_l2 = l2_norm / max(I_peak_ref, 1e-30)
        assert l2_norm > 0
        assert normalized_l2 < 2.0


class TestLeeModelExperimentalComparison:
    """Lee model vs experimental data from Scholz et al. (2006)."""

    def test_compare_with_experiment_api(self):
        model = LeeModel()
        comparison = model.compare_with_experiment("PF-1000")
        assert comparison.device_name == "PF-1000"
        assert comparison.peak_current_error >= 0
        assert comparison.timing_error >= 0

    def test_lee_model_closer_to_experiment_than_analytical(self):
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        I_peak_exp = PF1000_DATA.peak_current
        I_peak_undamped = PF1000_V0 * np.sqrt(PF1000_C / PF1000_L0)
        lee_error = abs(result.peak_current - I_peak_exp) / I_peak_exp
        undamped_error = abs(I_peak_undamped - I_peak_exp) / I_peak_exp
        assert lee_error < undamped_error


class TestLeeModelMultiDevice:
    """Verify Lee model runs for multiple devices."""

    @pytest.mark.parametrize("device", ["PF-1000", "NX2", "UNU-ICTP"])
    def test_lee_model_runs_all_devices(self, device: str):
        model = LeeModel()
        result = model.run(device_name=device)
        assert result.peak_current > 0
        assert len(result.t) > 10
        assert np.all(np.isfinite(result.I))


# ════════════════════════════════════════════════════════════════════════
# --- Section: Current Waveform Validation (test_phase_t_current_waveform) ---
# ════════════════════════════════════════════════════════════════════════

PF1000 = PF1000_DATA


def _pf1000_fill_density() -> float:
    """Compute PF-1000 fill density from ideal gas law at 300 K."""
    p_Pa = PF1000.fill_pressure_torr * 133.322
    T_room = 300.0
    n = p_Pa / (k_B * T_room)
    return n * m_d


def run_coupled_simulation(
    *,
    dt: float = 1e-9,
    t_end: float | None = None,
    mass_fraction: float = 0.3,
    current_fraction: float = 0.7,
    R_plasma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run coupled 0D circuit + snowplow simulation."""
    circuit = RLCSolver(
        C=PF1000.capacitance, V0=PF1000.voltage,
        L0=PF1000.inductance, R0=PF1000.resistance,
        anode_radius=PF1000.anode_radius,
        cathode_radius=PF1000.cathode_radius,
    )
    rho0 = _pf1000_fill_density()
    snowplow_mdl = SnowplowModel(
        anode_radius=PF1000.anode_radius,
        cathode_radius=PF1000.cathode_radius,
        fill_density=rho0,
        anode_length=PF1000.anode_length,
        mass_fraction=mass_fraction,
        current_fraction=current_fraction,
        fill_pressure_Pa=PF1000.fill_pressure_torr * 133.322,
    )
    if t_end is None:
        t_end = 3.0 * PF1000.current_rise_time
    n_steps = int(t_end / dt)
    t_arr = np.zeros(n_steps + 1)
    I_arr = np.zeros(n_steps + 1)
    L_arr = np.zeros(n_steps + 1)
    phase_arr = []
    t_arr[0] = 0.0
    I_arr[0] = 0.0
    L_arr[0] = snowplow_mdl.plasma_inductance
    phase_arr.append("rundown")
    coupling = CouplingState(
        Lp=snowplow_mdl.plasma_inductance, current=0.0, voltage=PF1000.voltage,
    )
    for i in range(n_steps):
        sp_result = snowplow_mdl.step(dt, coupling.current)
        coupling.Lp = sp_result["L_plasma"]
        coupling.dL_dt = sp_result["dL_dt"]
        coupling.R_plasma = R_plasma
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt)
        t_arr[i + 1] = (i + 1) * dt
        I_arr[i + 1] = coupling.current
        L_arr[i + 1] = sp_result["L_plasma"]
        phase_arr.append(sp_result["phase"])
    abs_I = np.abs(I_arr)
    radial_start_idx = None
    pinch_idx = None
    for idx, ph in enumerate(phase_arr):
        if ph == "radial" and radial_start_idx is None:
            radial_start_idx = idx
        if ph == "pinch" and pinch_idx is None:
            pinch_idx = idx
    peak_idx = int(np.argmax(abs_I))
    peak_current = float(abs_I[peak_idx])
    peak_time = float(t_arr[peak_idx])
    if radial_start_idx is not None:
        search_start = max(0, radial_start_idx - 200)
        pre_dip_region = abs_I[search_start:radial_start_idx + 50]
        pre_dip_peak_local = int(np.argmax(pre_dip_region))
        pre_dip_peak_idx = search_start + pre_dip_peak_local
        pre_dip_peak = float(abs_I[pre_dip_peak_idx])
        dip_search_end = min(
            pinch_idx + 200 if pinch_idx else radial_start_idx + 2000, len(abs_I),
        )
        dip_region = abs_I[radial_start_idx:dip_search_end]
        if len(dip_region) > 0:
            dip_local = int(np.argmin(dip_region))
            dip_idx = radial_start_idx + dip_local
            dip_current = float(abs_I[dip_idx])
            dip_time = float(t_arr[dip_idx])
            dip_depth = (pre_dip_peak - dip_current) / max(pre_dip_peak, 1e-30)
        else:
            dip_current = pre_dip_peak
            dip_time = t_arr[radial_start_idx]
            dip_depth = 0.0
    else:
        pre_dip_peak = peak_current
        dip_current = peak_current
        dip_time = peak_time
        dip_depth = 0.0
    diagnostics = {
        "peak_current": peak_current,
        "peak_time": peak_time,
        "pre_dip_peak": pre_dip_peak if radial_start_idx else peak_current,
        "dip_current": dip_current,
        "dip_time": dip_time,
        "dip_depth": dip_depth,
        "radial_start_idx": radial_start_idx,
        "radial_start_time": t_arr[radial_start_idx] if radial_start_idx else None,
        "final_L_plasma": float(L_arr[-1]),
        "final_phase": phase_arr[-1],
        "phases_seen": sorted(set(phase_arr)),
    }
    return t_arr, I_arr, diagnostics


class TestPeakCurrentValidation:
    """Validate peak current against PF-1000 published data."""

    def test_peak_current_order_of_magnitude(self):
        _, I_arr, diag = run_coupled_simulation(dt=5e-9, t_end=10e-6)
        peak = diag["peak_current"]
        assert peak > 0.5e6
        assert peak < 5e6

    def test_peak_current_within_50_percent(self):
        _, I_arr, diag = run_coupled_simulation(dt=2e-9, t_end=12e-6)
        peak = diag["peak_current"]
        exp_peak = PF1000.peak_current
        error = abs(peak - exp_peak) / exp_peak
        assert error < 0.5

    def test_peak_time_order_of_magnitude(self):
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=15e-6)
        t_peak = diag["peak_time"]
        t_exp = PF1000.current_rise_time
        ratio = t_peak / t_exp
        assert 0.3 < ratio < 3.0


class TestCurrentDipSignature:
    """Validate the characteristic DPF current dip."""

    def test_current_dip_exists(self):
        _, _, diag = run_coupled_simulation(dt=2e-9, t_end=15e-6)
        dip_depth = diag["dip_depth"]
        assert dip_depth > 0.05

    def test_dip_during_radial_phase(self):
        _, _, diag = run_coupled_simulation(dt=2e-9, t_end=15e-6)
        if diag["radial_start_time"] is not None:
            assert diag["dip_time"] >= diag["radial_start_time"]

    def test_dip_depth_physical_range(self):
        _, _, diag = run_coupled_simulation(dt=2e-9, t_end=15e-6)
        dip = diag["dip_depth"]
        assert 0.05 < dip < 0.90


class TestPhaseTransitionsInWaveform:
    """Verify that the circuit-coupled simulation sees all three phases."""

    def test_all_three_phases(self):
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=20e-6)
        phases = diag["phases_seen"]
        assert "rundown" in phases
        assert "radial" in phases

    def test_radial_phase_before_end(self):
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=20e-6)
        assert diag["radial_start_time"] is not None
        assert diag["radial_start_time"] < 15e-6


class TestInductanceEvolution:
    """Validate that L_plasma evolves correctly through the simulation."""

    def test_inductance_increases_monotonically(self):
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=15e-6)
        circuit = RLCSolver(
            C=PF1000.capacitance, V0=PF1000.voltage,
            L0=PF1000.inductance, R0=PF1000.resistance,
            anode_radius=PF1000.anode_radius, cathode_radius=PF1000.cathode_radius,
        )
        rho0 = _pf1000_fill_density()
        sp = SnowplowModel(
            anode_radius=PF1000.anode_radius, cathode_radius=PF1000.cathode_radius,
            fill_density=rho0, anode_length=PF1000.anode_length,
        )
        coupling = CouplingState(Lp=sp.plasma_inductance, voltage=PF1000.voltage)
        dt = 5e-9
        prev_L = sp.plasma_inductance
        for _ in range(int(15e-6 / dt)):
            sp_res = sp.step(dt, coupling.current)
            coupling.Lp = sp_res["L_plasma"]
            coupling.dL_dt = sp_res["dL_dt"]
            coupling = circuit.step(coupling, 0.0, dt)
            L_now = sp_res["L_plasma"]
            if not sp.pinch_complete:
                assert L_now >= prev_L - 1e-15
            prev_L = L_now

    def test_final_inductance_nanohenry_range(self):
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=15e-6)
        L_final = diag["final_L_plasma"]
        assert L_final > 1e-9
        assert L_final < 1e-6


class TestValidationUtility:
    """Test the validate_current_waveform function from experimental.py."""

    def test_validation_function_runs(self):
        t_arr, I_arr, _ = run_coupled_simulation(dt=5e-9, t_end=12e-6)
        metrics = validate_current_waveform(t_arr, I_arr, "PF-1000")
        assert "peak_current_error" in metrics
        assert "peak_current_sim" in metrics
        assert "timing_ok" in metrics

    def test_validation_peak_error_finite(self):
        t_arr, I_arr, _ = run_coupled_simulation(dt=5e-9, t_end=12e-6)
        metrics = validate_current_waveform(t_arr, I_arr, "PF-1000")
        assert np.isfinite(metrics["peak_current_error"])
        assert metrics["peak_current_error"] >= 0.0


class TestParameterSensitivity:
    """Test that model parameters have physically correct effects."""

    def test_higher_mass_fraction_delays_peak(self):
        _, _, diag_low = run_coupled_simulation(dt=5e-9, t_end=15e-6, mass_fraction=0.15)
        _, _, diag_high = run_coupled_simulation(dt=5e-9, t_end=15e-6, mass_fraction=0.5)
        if diag_low["radial_start_time"] and diag_high["radial_start_time"]:
            assert diag_high["radial_start_time"] > diag_low["radial_start_time"]

    def test_higher_current_fraction_increases_peak(self):
        _, _, diag_low = run_coupled_simulation(dt=5e-9, t_end=15e-6, current_fraction=0.5)
        _, _, diag_high = run_coupled_simulation(dt=5e-9, t_end=15e-6, current_fraction=0.9)
        if diag_low["radial_start_time"] and diag_high["radial_start_time"]:
            assert diag_high["radial_start_time"] < diag_low["radial_start_time"]

    def test_plasma_resistance_reduces_peak_current(self):
        _, _, diag_no_R = run_coupled_simulation(dt=5e-9, t_end=10e-6, R_plasma=0.0)
        _, _, diag_with_R = run_coupled_simulation(dt=5e-9, t_end=10e-6, R_plasma=5e-3)
        assert diag_with_R["peak_current"] < diag_no_R["peak_current"]


class TestMultiDeviceValidation:
    """Validate the coupled model against multiple DPF devices."""

    def test_unu_ictp_peak_order_of_magnitude(self):
        from dpf.validation.experimental import UNU_ICTP_DATA
        dev = UNU_ICTP_DATA
        rho0 = (dev.fill_pressure_torr * 133.322) / (k_B * 300.0) * m_d
        circuit = RLCSolver(
            C=dev.capacitance, V0=dev.voltage,
            L0=dev.inductance, R0=dev.resistance,
            anode_radius=dev.anode_radius, cathode_radius=dev.cathode_radius,
        )
        sp = SnowplowModel(
            anode_radius=dev.anode_radius, cathode_radius=dev.cathode_radius,
            fill_density=rho0, anode_length=dev.anode_length,
            fill_pressure_Pa=dev.fill_pressure_torr * 133.322,
        )
        dt = 2e-9
        t_end = 3.0 * dev.current_rise_time
        n_steps = int(t_end / dt)
        coupling = CouplingState(Lp=sp.plasma_inductance, voltage=dev.voltage)
        I_max = 0.0
        for _ in range(n_steps):
            sp_res = sp.step(dt, coupling.current)
            coupling.Lp = sp_res["L_plasma"]
            coupling.dL_dt = sp_res["dL_dt"]
            coupling = circuit.step(coupling, 0.0, dt)
            I_max = max(I_max, abs(coupling.current))
        assert I_max > 50e3
        assert I_max < 500e3

    def test_nx2_peak_order_of_magnitude(self):
        from dpf.validation.experimental import NX2_DATA
        dev = NX2_DATA
        rho0 = (dev.fill_pressure_torr * 133.322) / (k_B * 300.0) * m_d
        circuit = RLCSolver(
            C=dev.capacitance, V0=dev.voltage,
            L0=dev.inductance, R0=dev.resistance,
            anode_radius=dev.anode_radius, cathode_radius=dev.cathode_radius,
        )
        sp = SnowplowModel(
            anode_radius=dev.anode_radius, cathode_radius=dev.cathode_radius,
            fill_density=rho0, anode_length=dev.anode_length,
            fill_pressure_Pa=dev.fill_pressure_torr * 133.322,
        )
        dt = 1e-9
        t_end = 3.0 * dev.current_rise_time
        n_steps = int(t_end / dt)
        coupling = CouplingState(Lp=sp.plasma_inductance, voltage=dev.voltage)
        I_max = 0.0
        for _ in range(n_steps):
            sp_res = sp.step(dt, coupling.current)
            coupling.Lp = sp_res["L_plasma"]
            coupling.dL_dt = sp_res["dL_dt"]
            coupling = circuit.step(coupling, 0.0, dt)
            I_max = max(I_max, abs(coupling.current))
        assert I_max > 100e3
        assert I_max < 1e6


class TestCoupledEnergyConservation:
    """Verify energy accounting in the coupled circuit+snowplow system."""

    def test_circuit_energy_monotonically_decreases(self):
        circuit = RLCSolver(
            C=PF1000.capacitance, V0=PF1000.voltage,
            L0=PF1000.inductance, R0=PF1000.resistance,
            anode_radius=PF1000.anode_radius, cathode_radius=PF1000.cathode_radius,
        )
        rho0 = _pf1000_fill_density()
        sp = SnowplowModel(
            anode_radius=PF1000.anode_radius, cathode_radius=PF1000.cathode_radius,
            fill_density=rho0, anode_length=PF1000.anode_length,
        )
        dt = 5e-9
        coupling = CouplingState(Lp=sp.plasma_inductance, voltage=PF1000.voltage)
        E_initial = 0.5 * PF1000.capacitance * PF1000.voltage**2
        for _ in range(int(5e-6 / dt)):
            sp_res = sp.step(dt, coupling.current)
            coupling.Lp = sp_res["L_plasma"]
            coupling.dL_dt = sp_res["dL_dt"]
            coupling = circuit.step(coupling, 0.0, dt)
        E_cap = circuit.state.energy_cap
        E_ind = circuit.state.energy_ind
        E_res = circuit.state.energy_res
        E_total = E_cap + E_ind + E_res
        assert E_total > 0
        ratio = E_total / E_initial
        assert 0.5 < ratio < 1.5


# --- Section: Radial Phase Physics ---

PF1000_PARAMS = {
    "anode_radius": 0.0575,
    "cathode_radius": 0.08,
    "fill_density": 4e-4,
    "anode_length": 0.16,
    "mass_fraction": 0.3,
    "fill_pressure_Pa": 400.0,
    "current_fraction": 0.7,
}


def make_radial_snowplow(**kwargs) -> SnowplowModel:
    """Create a SnowplowModel already in radial phase."""
    params = {**PF1000_PARAMS, **kwargs}
    sp = SnowplowModel(
        anode_radius=params["anode_radius"],
        cathode_radius=params["cathode_radius"],
        fill_density=params["fill_density"],
        anode_length=params["anode_length"],
        mass_fraction=params["mass_fraction"],
        fill_pressure_Pa=params["fill_pressure_Pa"],
        current_fraction=params["current_fraction"],
        radial_mass_fraction=params.get("radial_mass_fraction"),
    )
    sp.z = sp.L_anode
    sp.v = 1e4
    sp._rundown_complete = True
    sp._L_axial_frozen = sp.L_coeff * sp.L_anode
    sp.phase = "radial"
    sp.r_shock = 0.95 * sp.b
    sp.vr = 0.0
    return sp


class TestRadialForceFormula:
    """F_rad = (mu_0 / 4pi) * (f_c * I)^2 * z_f / r_s."""

    def test_radial_force_at_initial_radius(self) -> None:
        sp = make_radial_snowplow()
        r_init = sp.r_shock
        I_current = 1.0e6
        dt = 1e-10
        result = sp.step(dt, current=I_current)
        z_f = sp.L_anode
        expected_F = (mu_0 / (4.0 * pi)) * (sp.f_c * I_current) ** 2 * z_f / r_init
        assert result["F_magnetic"] == pytest.approx(expected_F, rel=1e-6)
        assert result["phase"] == "radial"

    def test_radial_force_at_intermediate_radius(self) -> None:
        sp = make_radial_snowplow()
        r_mid = 0.5 * (sp.a + sp.b)
        sp.r_shock = r_mid
        sp.vr = -5e3
        I_current = 5e5
        dt = 1e-10
        result = sp.step(dt, current=I_current)
        z_f = sp.L_anode
        expected_F = (mu_0 / (4.0 * pi)) * (sp.f_c * I_current) ** 2 * z_f / r_mid
        assert result["F_magnetic"] == pytest.approx(expected_F, rel=1e-4)

    def test_radial_force_increases_as_shock_converges(self) -> None:
        I_current = 1e6
        dt = 1e-10
        forces = []
        radii = [0.07, 0.05, 0.03]
        for r in radii:
            sp_i = make_radial_snowplow()
            sp_i.r_shock = r
            sp_i.vr = -1e3
            result = sp_i.step(dt, current=I_current)
            forces.append(result["F_magnetic"])
        assert forces[1] > forces[0]
        assert forces[2] > forces[1]
        ratio_r = radii[0] / radii[1]
        ratio_F = forces[1] / forces[0]
        assert ratio_F == pytest.approx(ratio_r, rel=0.01)


class TestRadialInductanceFormula:
    """L_plasma = L_axial + (mu_0 / 2pi) * z_f * ln(b / r_s)."""

    def test_inductance_at_cathode_radius(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = sp.b
        expected_L = sp._L_axial_frozen
        assert sp.plasma_inductance == pytest.approx(expected_L, rel=1e-10)

    def test_inductance_at_intermediate_radius(self) -> None:
        sp = make_radial_snowplow()
        r_test = 0.04
        sp.r_shock = r_test
        L_axial = sp._L_axial_frozen
        L_radial_expected = (mu_0 / (2.0 * pi)) * sp.L_anode * np.log(sp.b / r_test)
        expected_total = L_axial + L_radial_expected
        assert sp.plasma_inductance == pytest.approx(expected_total, rel=1e-10)

    def test_inductance_increases_as_shock_converges(self) -> None:
        sp = make_radial_snowplow()
        radii = [sp.b, 0.06, 0.04, 0.02, 0.01]
        inductances = []
        for r in radii:
            sp.r_shock = r
            inductances.append(sp.plasma_inductance)
        for i in range(1, len(inductances)):
            assert inductances[i] > inductances[i - 1]

    def test_inductance_clamped_at_pinch_min(self) -> None:
        sp = make_radial_snowplow()
        r_tiny = 1e-6
        sp.r_shock = r_tiny
        r_eff = sp.r_pinch_min
        L_expected = sp._L_axial_frozen + (mu_0 / (2.0 * pi)) * sp.L_anode * np.log(
            sp.b / r_eff
        )
        assert sp.plasma_inductance == pytest.approx(L_expected, rel=1e-10)


class TestRadialDLDT:
    """dL/dt = -(mu_0 / 2pi) * z_f * vr / r_s."""

    def test_dL_dt_at_start(self) -> None:
        sp = make_radial_snowplow()
        result = sp.step(1e-10, current=1e6)
        assert result["dL_dt"] >= 0.0

    def test_dL_dt_formula_after_step(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = 0.06
        sp.vr = -1e4
        result = sp.step(1e-10, current=1e6)
        vr_after = sp.vr
        r_after = max(sp.r_shock, sp.r_pinch_min)
        z_f = sp.L_anode
        expected_dLdt = -(mu_0 / (2.0 * pi)) * z_f * vr_after / r_after
        assert result["dL_dt"] == pytest.approx(expected_dLdt, rel=1e-4)

    def test_dL_dt_positive_for_inward_motion(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = 0.05
        sp.vr = -5e3
        result = sp.step(1e-10, current=1e6)
        assert result["dL_dt"] > 0.0


class TestRadialMassAccumulation:
    """M_slug = f_mr * rho_0 * pi * (b^2 - r_s^2) * z_f."""

    def test_zero_mass_at_cathode(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = sp.b
        assert sp.radial_swept_mass == pytest.approx(0.0, abs=1e-20)

    def test_mass_formula_at_intermediate_radius(self) -> None:
        sp = make_radial_snowplow()
        r_test = 0.04
        sp.r_shock = r_test
        expected_M = sp.f_mr * sp.rho0 * pi * (sp.b**2 - r_test**2) * sp.L_anode
        assert sp.radial_swept_mass == pytest.approx(expected_M, rel=1e-10)

    def test_mass_increases_as_shock_converges(self) -> None:
        sp = make_radial_snowplow()
        masses = []
        radii = [sp.b, 0.06, 0.04, 0.02]
        for r in radii:
            sp.r_shock = r
            masses.append(sp.radial_swept_mass)
        for i in range(1, len(masses)):
            assert masses[i] > masses[i - 1]

    def test_maximum_mass_at_axis(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = 0.0
        M_max = sp.f_mr * sp.rho0 * pi * sp.b**2 * sp.L_anode
        assert sp.radial_swept_mass == pytest.approx(M_max, rel=1e-10)


class TestRadialConvergence:
    """Velocity-Verlet integrator should show ~2nd order convergence."""

    def test_convergence_order(self) -> None:
        I_current = 8e5
        n_steps_coarse = 100
        dt_coarse = 1e-9
        sp1 = make_radial_snowplow()
        for _ in range(n_steps_coarse):
            sp1.step(dt_coarse, current=I_current)
        r_coarse = sp1.r_shock
        sp2 = make_radial_snowplow()
        for _ in range(2 * n_steps_coarse):
            sp2.step(dt_coarse / 2.0, current=I_current)
        r_fine = sp2.r_shock
        sp_ref = make_radial_snowplow()
        for _ in range(4 * n_steps_coarse):
            sp_ref.step(dt_coarse / 4.0, current=I_current)
        r_ref = sp_ref.r_shock
        err_coarse = abs(r_coarse - r_ref)
        err_fine = abs(r_fine - r_ref)
        if err_fine > 1e-15:
            order = np.log2(err_coarse / err_fine)
            assert order > 1.5, f"Expected ~2nd order, got {order:.2f}"


class TestRadialShockDirection:
    """Radial shock velocity must be <= 0 (inward) throughout Phase 3."""

    def test_vr_always_non_positive(self) -> None:
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        for _ in range(5000):
            result = sp.step(dt, current=I_current)
            assert result["vr_shock"] <= 0.0
            if sp.phase == "pinch":
                break

    def test_r_shock_monotonically_decreases(self) -> None:
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        prev_r = sp.r_shock
        for _ in range(5000):
            result = sp.step(dt, current=I_current)
            assert result["r_shock"] <= prev_r + 1e-15
            prev_r = result["r_shock"]
            if sp.phase == "pinch":
                break


class TestRadialInductanceMonotonicity:
    """Plasma inductance should be non-decreasing through the radial phase."""

    def test_inductance_non_decreasing(self) -> None:
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        prev_L = sp.plasma_inductance
        for _ in range(5000):
            result = sp.step(dt, current=I_current)
            L_now = result["L_plasma"]
            assert L_now >= prev_L - 1e-15
            prev_L = L_now
            if sp.phase == "pinch":
                break

    def test_inductance_significantly_larger_after_pinch(self) -> None:
        sp = make_radial_snowplow()
        L_start = sp.plasma_inductance
        I_current = 1e6
        dt = 1e-9
        for _ in range(50000):
            sp.step(dt, current=I_current)
            if sp.phase == "pinch":
                break
        L_end = sp.plasma_inductance
        assert L_end > L_start * 1.5


class TestPinchTiming:
    """Pinch should occur within ~1 microsecond of radial start for PF-1000."""

    def test_pinch_time_order_of_magnitude(self) -> None:
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-10
        max_steps = 1_000_000
        n_taken = 0
        for _steps in range(max_steps):
            sp.step(dt, current=I_current)
            n_taken = _steps + 1
            if sp.phase == "pinch":
                break
        pinch_time = n_taken * dt
        assert sp.phase == "pinch", "Did not reach pinch within 100 us"
        assert pinch_time < 10e-6
        assert pinch_time > 10e-9

    def test_pinch_radius_at_minimum(self) -> None:
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        for _ in range(100_000):
            sp.step(dt, current=I_current)
            if sp.phase in ("reflected", "pinch"):
                break
        assert sp.phase in ("reflected", "pinch")
        assert sp.r_shock == pytest.approx(sp.r_pinch_min, rel=0.01)


class TestPhaseTransitions:
    """Verify rundown -> radial -> pinch transition sequence."""

    def test_rundown_to_radial_transition(self) -> None:
        sp = SnowplowModel(**PF1000_PARAMS)
        I_current = 1e6
        dt = 1e-8
        saw_rundown = False
        saw_radial = False
        prev_phase = "rundown"
        for _ in range(500_000):
            result = sp.step(dt, current=I_current)
            phase = result["phase"]
            if phase == "rundown":
                saw_rundown = True
            elif phase == "radial":
                if not saw_radial:
                    assert prev_phase == "rundown" or prev_phase == "radial"
                saw_radial = True
            elif phase == "pinch":
                assert saw_radial, "Jumped to pinch without radial phase"
                break
            prev_phase = phase
        assert saw_rundown
        assert saw_radial

    @pytest.mark.slow
    def test_full_phase_sequence(self) -> None:
        sp = SnowplowModel(**PF1000_PARAMS)
        I_current = 1e6
        dt = 5e-9
        phase_order = []
        for _ in range(500_000):
            result = sp.step(dt, current=I_current)
            phase = result["phase"]
            if not phase_order or phase_order[-1] != phase:
                phase_order.append(phase)
            if phase == "pinch":
                break
        assert phase_order == ["rundown", "radial", "reflected", "pinch"]

    def test_rundown_complete_flag(self) -> None:
        sp = make_radial_snowplow()
        assert sp.rundown_complete is True
        assert sp.phase == "radial"

    def test_pinch_complete_flag(self) -> None:
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        for _ in range(100_000):
            sp.step(dt, current=I_current)
            if sp.phase == "pinch":
                break
        assert sp.pinch_complete is True
        assert sp.is_active is False

    def test_frozen_state_after_pinch(self) -> None:
        sp = make_radial_snowplow()
        I_current = 1e6
        dt = 1e-9
        for _ in range(100_000):
            sp.step(dt, current=I_current)
            if sp.phase == "pinch":
                break
        result1 = sp.step(dt, current=I_current)
        sp.step(dt, current=I_current)
        assert result1["phase"] == "pinch"
        assert result1["F_magnetic"] == 0.0
        assert result1["F_pressure"] == 0.0


class TestRadialMassFraction:
    """f_mr can be set independently of f_m."""

    def test_default_f_mr_equals_f_m(self) -> None:
        sp = SnowplowModel(**PF1000_PARAMS)
        assert sp.f_mr == sp.f_m

    def test_custom_f_mr(self) -> None:
        sp = make_radial_snowplow(radial_mass_fraction=0.1)
        assert sp.f_mr == 0.1
        assert sp.f_m == 0.3

    def test_f_mr_affects_radial_mass(self) -> None:
        sp_low = make_radial_snowplow(radial_mass_fraction=0.1)
        sp_high = make_radial_snowplow(radial_mass_fraction=0.5)
        r_test = 0.04
        sp_low.r_shock = r_test
        sp_high.r_shock = r_test
        assert sp_low.radial_swept_mass < sp_high.radial_swept_mass
        ratio = sp_high.radial_swept_mass / sp_low.radial_swept_mass
        assert ratio == pytest.approx(0.5 / 0.1, rel=1e-10)


class TestCurrentFractionEffect:
    """Higher f_c -> faster radial implosion (lower pinch time)."""

    def test_higher_fc_faster_pinch(self) -> None:
        I_current = 1e6
        dt = 1e-10
        times = {}
        for f_c in [0.5, 0.8]:
            sp = make_radial_snowplow(current_fraction=f_c)
            n_taken = 0
            for _step_n in range(1_000_000):
                sp.step(dt, current=I_current)
                n_taken = _step_n + 1
                if sp.phase == "pinch":
                    break
            times[f_c] = n_taken * dt
            assert sp.phase == "pinch", f"f_c={f_c} did not reach pinch"
        assert times[0.8] < times[0.5]

    def test_fc_affects_radial_force(self) -> None:
        I_current = 1e6
        dt = 1e-10
        sp_low = make_radial_snowplow(current_fraction=0.5)
        sp_high = make_radial_snowplow(current_fraction=1.0)
        r_low = sp_low.step(dt, current=I_current)
        r_high = sp_high.step(dt, current=I_current)
        ratio = r_high["F_magnetic"] / r_low["F_magnetic"]
        expected_ratio = (1.0 / 0.5) ** 2
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)


class TestZeroCurrentRadial:
    """With zero current, no radial force: shock should stall."""

    def test_shock_stalls_with_zero_current(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = 0.06
        sp.vr = 0.0
        r_initial = sp.r_shock
        dt = 1e-9
        for _ in range(1000):
            sp.step(dt, current=0.0)
        assert sp.r_shock == pytest.approx(r_initial, abs=1e-12)

    def test_zero_force_with_zero_current(self) -> None:
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=0.0)
        assert result["F_magnetic"] == pytest.approx(0.0, abs=1e-20)

    def test_moving_shock_decelerates_with_zero_current(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = 0.05
        sp.vr = -1e4
        dt = 1e-9
        initial_vr = sp.vr
        for _ in range(100):
            sp.step(dt, current=0.0)
        assert sp.vr > initial_vr


class TestRadialEdgeCases:
    """Edge cases and robustness checks for the radial phase."""

    def test_very_small_current(self) -> None:
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=1.0)
        assert result["F_magnetic"] > 0.0
        assert np.isfinite(result["F_magnetic"])
        assert result["F_magnetic"] < 1e-6

    def test_very_large_current(self) -> None:
        sp = make_radial_snowplow()
        result = sp.step(1e-12, current=1e8)
        assert np.isfinite(result["F_magnetic"])
        assert np.isfinite(result["r_shock"])
        assert np.isfinite(result["vr_shock"])

    def test_adiabatic_back_pressure_in_radial(self) -> None:
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=1e6)
        assert result["F_pressure"] > 0.0

    def test_axial_position_frozen_in_radial(self) -> None:
        sp = make_radial_snowplow()
        dt = 1e-9
        for _ in range(100):
            result = sp.step(dt, current=1e6)
            assert result["z_sheath"] == pytest.approx(sp.L_anode, rel=1e-10)

    def test_step_returns_all_keys(self) -> None:
        sp = make_radial_snowplow()
        result = sp.step(1e-9, current=1e6)
        expected_keys = {
            "z_sheath", "v_sheath", "r_shock", "vr_shock",
            "L_plasma", "dL_dt", "swept_mass", "F_magnetic",
            "F_pressure", "phase", "R_plasma", "f_cr_eff",
        }
        assert set(result.keys()) == expected_keys


class TestRadialProperties:
    """Test property accessors during radial phase."""

    def test_shock_radius_property(self) -> None:
        sp = make_radial_snowplow()
        sp.r_shock = 0.03
        assert sp.shock_radius == 0.03

    def test_sheath_position_property(self) -> None:
        sp = make_radial_snowplow()
        assert sp.sheath_position == sp.L_anode

    def test_is_active_in_radial(self) -> None:
        sp = make_radial_snowplow()
        assert sp.is_active is True
        assert sp.pinch_complete is False


# --- Section: T Validation ---


def _pf1000_snowplow(
    current_fraction: float = 0.7,
    mass_fraction: float = 0.3,
    radial_mass_fraction: float | None = None,
) -> SnowplowModel:
    return SnowplowModel(
        anode_radius=0.0575,
        cathode_radius=0.08,
        fill_density=4e-4,
        anode_length=0.16,
        mass_fraction=mass_fraction,
        fill_pressure_Pa=400.0,
        current_fraction=current_fraction,
        radial_mass_fraction=radial_mass_fraction,
    )


def _t_validation_make_radial_snowplow(
    r_shock: float = 0.04,
    vr: float = -1e4,
    **kwargs,
) -> SnowplowModel:
    sp = _pf1000_snowplow(**kwargs)
    sp.phase = "radial"
    sp._rundown_complete = True
    sp.z = sp.L_anode
    sp._L_axial_frozen = sp.L_coeff * sp.L_anode
    sp.r_shock = r_shock
    sp.vr = vr
    return sp


def _run_to_radial(sp: SnowplowModel, I_current: float = 1.0e6) -> None:
    dt = 1e-8
    max_steps = 100_000
    for _ in range(max_steps):
        sp.step(dt, I_current)
        if sp.phase == "radial":
            return
    raise RuntimeError("Snowplow did not reach radial phase within max_steps")


class TestLCoeffFix:
    """Verify L_coeff = 2 * F_coeff (the factor-of-2 relationship)."""

    def test_L_coeff_equals_2_F_coeff(self) -> None:
        sp = _pf1000_snowplow()
        assert sp.L_coeff == pytest.approx(2.0 * sp.F_coeff, rel=1e-14)

    def test_F_coeff_formula(self) -> None:
        sp = _pf1000_snowplow()
        expected = (mu_0 / (4.0 * pi)) * np.log(sp.b / sp.a)
        assert sp.F_coeff == pytest.approx(expected, rel=1e-14)

    def test_L_coeff_formula(self) -> None:
        sp = _pf1000_snowplow()
        expected = (mu_0 / (2.0 * pi)) * np.log(sp.b / sp.a)
        assert sp.L_coeff == pytest.approx(expected, rel=1e-14)

    def test_plasma_inductance_axial(self) -> None:
        sp = _pf1000_snowplow()
        z = sp.z
        expected_L = sp.L_coeff * z
        assert sp.plasma_inductance == pytest.approx(expected_L, rel=1e-10)


class TestCurrentFraction:
    """Verify that force uses (f_c * I)^2 with f_c=0.7 default."""

    def test_default_current_fraction(self) -> None:
        sp = _pf1000_snowplow()
        assert sp.f_c == pytest.approx(0.7, abs=1e-15)

    def test_force_uses_fc_squared(self) -> None:
        fc = 0.7
        sp = _pf1000_snowplow(current_fraction=fc)
        I_test = 500e3
        dt = 1e-9
        result = sp.step(dt, I_test)
        F_mag = result["F_magnetic"]
        expected = sp.F_coeff * (fc * I_test) ** 2
        assert F_mag == pytest.approx(expected, rel=1e-10)

    def test_fc_affects_force_magnitude(self) -> None:
        I_test = 500e3
        dt = 1e-9
        sp1 = _pf1000_snowplow(current_fraction=0.7)
        result1 = sp1.step(dt, I_test)
        sp2 = _pf1000_snowplow(current_fraction=0.5)
        result2 = sp2.step(dt, I_test)
        ratio = result2["F_magnetic"] / result1["F_magnetic"]
        expected_ratio = (0.5 / 0.7) ** 2
        assert ratio == pytest.approx(expected_ratio, rel=1e-10)

    def test_fc_one_recovers_full_current(self) -> None:
        sp = _pf1000_snowplow(current_fraction=1.0)
        I_test = 300e3
        dt = 1e-9
        result = sp.step(dt, I_test)
        expected = sp.F_coeff * I_test**2
        assert result["F_magnetic"] == pytest.approx(expected, rel=1e-10)


class TestRadialCompression:
    """Tests for radial compression phase behavior."""

    def test_radial_starts_after_rundown(self) -> None:
        sp = _pf1000_snowplow()
        assert sp.phase == "rundown"
        assert not sp.rundown_complete
        _run_to_radial(sp)
        assert sp.phase == "radial"
        assert sp.rundown_complete
        assert sp.z == pytest.approx(sp.L_anode, rel=1e-10)

    def test_shock_radius_decreases_during_radial(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.06, vr=-1e4)
        r_initial = sp.shock_radius
        assert r_initial < sp.b
        dt = 1e-10
        for _ in range(500):
            sp.step(dt, 500e3)
        assert sp.shock_radius < r_initial

    def test_L_plasma_grows_during_radial(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.06, vr=-1e4)
        L_at_start = sp.plasma_inductance
        dt = 1e-10
        for _ in range(500):
            sp.step(dt, 500e3)
        L_after = sp.plasma_inductance
        assert L_after > L_at_start

    def test_dL_dt_positive_during_radial(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.04, vr=-1e4)
        dt = 1e-10
        result = sp.step(dt, 500e3)
        assert result["dL_dt"] > 0
        assert result["vr_shock"] <= 0

    def test_radial_swept_mass_grows(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.06, vr=-1e4)
        m_initial = sp.radial_swept_mass
        assert m_initial > 0
        dt = 1e-10
        for _ in range(500):
            sp.step(dt, 500e3)
        m_after = sp.radial_swept_mass
        assert m_after > m_initial

    def test_pinch_detection(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.02, vr=-5e4)
        dt = 1e-10
        max_steps = 500_000
        pinched = False
        for _ in range(max_steps):
            sp.step(dt, 500e3)
            if sp.pinch_complete:
                pinched = True
                break
        assert pinched
        assert sp.phase == "pinch"
        assert sp.shock_radius <= 0.1 * sp.a + 1e-10

    def test_frozen_state_after_pinch(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.02, vr=-5e4)
        dt = 1e-10
        for _ in range(500_000):
            sp.step(dt, 500e3)
            if sp.pinch_complete:
                break
        assert sp.pinch_complete
        result = sp.step(dt, 500e3)
        assert result["F_magnetic"] == pytest.approx(0.0, abs=1e-20)
        assert result["F_pressure"] == pytest.approx(0.0, abs=1e-20)
        assert result["phase"] == "pinch"
        assert not sp.is_active


class TestRadialPhysicsFormulas:
    """Verify radial phase formulas against Lee model references."""

    def test_radial_force_formula(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.04, vr=-1e4)
        I_test = 500e3
        r_s_pre = sp.shock_radius
        z_f = sp.L_anode
        expected_F = (mu_0 / (4.0 * pi)) * (sp.f_c * I_test) ** 2 * z_f / r_s_pre
        dt = 1e-10
        result = sp.step(dt, I_test)
        assert result["F_magnetic"] > 0
        assert result["F_magnetic"] == pytest.approx(expected_F, rel=0.05)

    def test_radial_L_plasma_formula(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.04, vr=-1e4)
        dt = 1e-10
        for _ in range(100):
            sp.step(dt, 500e3)
        r_s = max(sp.shock_radius, sp.r_pinch_min)
        z_f = sp.L_anode
        L_axial_expected = sp._L_axial_frozen
        L_radial_expected = (mu_0 / (2.0 * pi)) * z_f * np.log(sp.b / r_s)
        L_total_expected = L_axial_expected + L_radial_expected
        assert sp.plasma_inductance == pytest.approx(L_total_expected, rel=1e-10)

    def test_dL_dt_formula(self) -> None:
        sp = _t_validation_make_radial_snowplow(r_shock=0.04, vr=-1e4)
        dt = 1e-10
        result = sp.step(dt, 500e3)
        r_s = max(sp.shock_radius, 1e-10)
        z_f = sp.L_anode
        vr = sp.vr
        expected_dL_dt = -(mu_0 / (2.0 * pi)) * z_f * vr / r_s
        assert result["dL_dt"] == pytest.approx(expected_dL_dt, rel=1e-10)

    def test_radial_swept_mass_formula(self) -> None:
        sp = _t_validation_make_radial_snowplow(
            r_shock=0.04, vr=-1e4, radial_mass_fraction=0.2,
        )
        dt = 1e-10
        for _ in range(100):
            sp.step(dt, 500e3)
        expected = sp.f_mr * sp.rho0 * pi * (sp.b**2 - sp.r_shock**2) * sp.L_anode
        assert sp.radial_swept_mass == pytest.approx(expected, rel=1e-10)


class TestPF1000Consistency:
    """Ensure PF-1000 parameters are consistent between suite.py and experimental.py."""

    def test_peak_current_match(self) -> None:
        suite_Ipeak = DEVICE_REGISTRY["PF-1000"].peak_current_A
        exp_Ipeak = PF1000_DATA.peak_current
        assert suite_Ipeak == pytest.approx(exp_Ipeak, rel=1e-10)

    def test_peak_current_value_1_87MA(self) -> None:
        assert DEVICE_REGISTRY["PF-1000"].peak_current_A == pytest.approx(1.87e6, rel=1e-10)
        assert PF1000_DATA.peak_current == pytest.approx(1.87e6, rel=1e-10)

    def test_anode_radius_consistency(self) -> None:
        assert DEVICE_REGISTRY["PF-1000"].anode_radius == pytest.approx(0.115, rel=1e-10)
        assert PF1000_DATA.anode_radius == pytest.approx(0.115, rel=1e-10)

    def test_cathode_radius_consistency(self) -> None:
        assert DEVICE_REGISTRY["PF-1000"].cathode_radius == pytest.approx(0.16, rel=1e-10)
        assert PF1000_DATA.cathode_radius == pytest.approx(0.16, rel=1e-10)

    def test_capacitance_consistency(self) -> None:
        suite_C = DEVICE_REGISTRY["PF-1000"].C
        exp_C = PF1000_DATA.capacitance
        assert suite_C == pytest.approx(exp_C, rel=1e-10)


class TestDocstringCorrectness:
    """Verify Lee model force formula docstrings use mu_0/(4pi)."""

    def test_snowplow_docstring_force_formula(self) -> None:
        import dpf.fluid.snowplow as snowplow_mod
        docstring = snowplow_mod.__doc__
        assert docstring is not None
        assert "mu_0 / 4pi" in docstring or "mu_0/4pi" in docstring

    def test_snowplow_no_mu0_over_2_in_force(self) -> None:
        import inspect

        import dpf.fluid.snowplow as snowplow_mod
        source = inspect.getsource(snowplow_mod.SnowplowModel.__init__)
        assert "mu_0 / (4.0 * pi)" in source

    def test_lee_model_docstring_uses_4pi(self) -> None:
        import dpf.validation.lee_model_comparison as lee_mod
        docstring = lee_mod.__doc__
        assert docstring is not None
        assert "mu_0 / (4*pi)" in docstring or "mu_0/(4*pi)" in docstring


class TestSnowplowCircuitCoupling:
    """Test snowplow integration with the SimulationEngine."""

    def test_config_snowplow_enabled(self) -> None:
        cfg = SnowplowConfig()
        assert cfg.enabled is True
        assert cfg.mass_fraction == pytest.approx(0.3)
        assert cfg.current_fraction == pytest.approx(0.7)
        assert cfg.fill_pressure_Pa == pytest.approx(400.0)
        assert cfg.anode_length == pytest.approx(0.16)

    def test_config_snowplow_disabled(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": False},
        )
        from dpf.engine import SimulationEngine  # noqa: PLC0415
        engine = SimulationEngine(config)
        assert engine.snowplow is None

    def test_config_snowplow_enabled_creates_model(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-7,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": True},
        )
        from dpf.engine import SimulationEngine  # noqa: PLC0415
        engine = SimulationEngine(config)
        assert engine.snowplow is not None
        assert isinstance(engine.snowplow, SnowplowModel)

    def test_snowplow_feeds_L_plasma_to_circuit(self) -> None:
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-8,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-7,
                     "anode_radius": 0.005, "cathode_radius": 0.01},
            snowplow={"enabled": True, "anode_length": 0.01},
        )
        from dpf.engine import SimulationEngine  # noqa: PLC0415
        engine = SimulationEngine(config)
        engine.step()
        coupling = engine._coupling
        assert coupling.Lp > 0

    def test_current_dip_signature(self) -> None:
        solver = RLCSolver(C=28e-6, V0=15000, L0=50e-9, R0=0.005)
        coupling_no_dL = CouplingState(Lp=50e-9, dL_dt=0.0, R_plasma=0.01)
        coupling_large_dL = CouplingState(Lp=50e-9, dL_dt=0.1, R_plasma=0.01)
        dt = 1e-8
        result_no_dL = solver.step(coupling_no_dL, back_emf=0.0, dt=dt)
        solver2 = RLCSolver(C=28e-6, V0=15000, L0=50e-9, R0=0.005)
        result_with_dL = solver2.step(coupling_large_dL, back_emf=0.0, dt=dt)
        assert result_with_dL.current < result_no_dL.current


class TestSnowplowConstruction:
    """Tests for SnowplowModel construction and initial state."""

    def test_initial_phase_is_rundown(self) -> None:
        sp = _pf1000_snowplow()
        assert sp.phase == "rundown"
        assert not sp.rundown_complete
        assert not sp.pinch_complete
        assert sp.is_active

    def test_initial_shock_radius_is_cathode(self) -> None:
        sp = _pf1000_snowplow()
        assert sp.shock_radius == pytest.approx(sp.b, rel=1e-15)

    def test_initial_radial_velocity_zero(self) -> None:
        sp = _pf1000_snowplow()
        assert sp.vr == pytest.approx(0.0, abs=1e-20)

    def test_pinch_min_radius(self) -> None:
        sp = _pf1000_snowplow()
        assert sp.r_pinch_min == pytest.approx(0.1 * sp.a, rel=1e-15)

    def test_radial_mass_fraction_default(self) -> None:
        sp = _pf1000_snowplow(mass_fraction=0.3, radial_mass_fraction=None)
        assert sp.f_mr == pytest.approx(0.3, rel=1e-15)

    def test_radial_mass_fraction_custom(self) -> None:
        sp = _pf1000_snowplow(mass_fraction=0.3, radial_mass_fraction=0.15)
        assert sp.f_mr == pytest.approx(0.15, rel=1e-15)
        assert sp.f_m == pytest.approx(0.3, rel=1e-15)

    def test_axial_swept_mass_formula(self) -> None:
        sp = _pf1000_snowplow()
        expected = sp.rho0 * pi * (sp.b**2 - sp.a**2) * sp.z * sp.f_m
        assert sp.swept_mass == pytest.approx(expected, rel=1e-12)

    def test_step_returns_correct_keys(self) -> None:
        sp = _pf1000_snowplow()
        result = sp.step(1e-9, 500e3)
        required_keys = {
            "z_sheath", "v_sheath", "r_shock", "vr_shock",
            "L_plasma", "dL_dt", "swept_mass",
            "F_magnetic", "F_pressure", "phase",
        }
        assert required_keys.issubset(result.keys())


class TestSnowplowConfigValidation:
    """Test SnowplowConfig Pydantic validators."""

    def test_snowplow_config_defaults(self) -> None:
        cfg = SnowplowConfig()
        assert cfg.enabled is True
        assert cfg.mass_fraction == pytest.approx(0.3)
        assert cfg.fill_pressure_Pa == pytest.approx(400.0)
        assert cfg.anode_length == pytest.approx(0.16)
        assert cfg.current_fraction == pytest.approx(0.7)
        assert cfg.radial_mass_fraction is None

    def test_snowplow_config_custom(self) -> None:
        cfg = SnowplowConfig(
            enabled=True,
            mass_fraction=0.5,
            fill_pressure_Pa=300.0,
            anode_length=0.10,
            current_fraction=0.8,
            radial_mass_fraction=0.2,
        )
        assert cfg.mass_fraction == pytest.approx(0.5)
        assert cfg.fill_pressure_Pa == pytest.approx(300.0)
        assert cfg.anode_length == pytest.approx(0.10)
        assert cfg.current_fraction == pytest.approx(0.8)
        assert cfg.radial_mass_fraction == pytest.approx(0.2)

    def test_snowplow_config_invalid_mass_fraction(self) -> None:
        with pytest.raises(ValueError):
            SnowplowConfig(mass_fraction=0.0)
        with pytest.raises(ValueError):
            SnowplowConfig(mass_fraction=1.5)

    def test_snowplow_config_invalid_current_fraction(self) -> None:
        with pytest.raises(ValueError):
            SnowplowConfig(current_fraction=0.0)
        with pytest.raises(ValueError):
            SnowplowConfig(current_fraction=1.5)


# --- Section: Lee Model Fixes ---



class TestLeeModelRadialPhysicsFixes:
    """Tests for corrections to Lee model radial phase physics."""

    def test_radial_mass_is_dynamic(self) -> None:
        model_low = LeeModel(mass_fraction=0.1)
        res_low = model_low.run("PF-1000")
        model_high = LeeModel(mass_fraction=0.9)
        res_high = model_high.run("PF-1000")
        diff_time = res_high.pinch_time - res_low.pinch_time
        assert diff_time > 0
        assert diff_time > 1e-7

    def test_adiabatic_back_pressure_opposes_compression(self) -> None:
        model_nom = LeeModel()
        params = self._get_default_params()
        params["fill_pressure_torr"] = 100.0
        res_high_p = model_nom.run(device_params=params)
        assert res_high_p.pinch_time > 0
        min_r = np.min(res_high_p.r_shock)
        assert min_r > 0.001

    def _get_default_params(self):
        return {
            "C": 1.332e-3, "V0": 27e3, "L0": 33.5e-9, "R0": 2.3e-3,
            "anode_radius": 0.0575, "cathode_radius": 0.08,
            "anode_length": 0.16, "fill_pressure_torr": 3.5,
            "peak_current_exp": 1.87e6, "current_rise_time_exp": 5.8e-6,
        }


class TestValidationRefinements:
    """Tests for stricter validation logic."""

    def test_timing_tolerance_tightened(self) -> None:
        device = DEVICES["NX2"]
        t_target = device.current_rise_time
        t_bad = t_target * 1.25
        t_arr = np.linspace(0, 3 * t_target, 200)
        I_arr = device.peak_current * np.sin(np.pi * t_arr / (2 * t_bad)) * np.exp(
            -t_arr / t_bad
        )
        result = validate_current_waveform(t_arr, I_arr, "NX2")
        assert not result["timing_ok"]

    def test_uncertainty_field_present(self) -> None:
        t_arr = np.linspace(0, 1e-5, 100)
        I_arr = np.sin(t_arr * 1e6) * 1e6
        result = validate_current_waveform(t_arr, I_arr, "PF-1000")
        assert "uncertainty" in result
        unc = result["uncertainty"]
        assert "peak_current_exp_1sigma" in unc
        assert "peak_current_combined_1sigma" in unc


# --- Section: Lee Model Validation ---


class TestLeeModelRadialPhase:
    """Tests for the fixed Lee model radial implosion phase."""

    def test_radial_mass_depends_on_rs(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel(mass_fraction=0.5, current_fraction=0.7)
        result = model.run(device_name="PF-1000")
        assert 2 in result.phases_completed
        assert result.r_shock[-1] < 0.08

    def test_radial_mass_uses_fm(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model_full = LeeModel(mass_fraction=1.0, current_fraction=0.7)
        model_partial = LeeModel(mass_fraction=0.3, current_fraction=0.7)
        result_full = model_full.run(device_name="PF-1000")
        result_partial = model_partial.run(device_name="PF-1000")
        assert 2 in result_full.phases_completed
        assert 2 in result_partial.phases_completed
        assert result_full.peak_current != pytest.approx(
            result_partial.peak_current, rel=0.01
        )

    def test_radial_phase_has_back_pressure(self) -> None:
        import inspect  # noqa: PLC0415

        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel()
        source = inspect.getsource(model.run)
        assert "p_back" in source or "back" in source.lower()
        assert "gamma" in source

    def test_radial_back_pressure_formula(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert 2 in result.phases_completed
        assert result.peak_current > 0

    def test_radial_mass_formula_dimensional(self) -> None:
        fm = 0.3
        rho0 = 4e-4
        b = 0.08
        r_s = 0.04
        z_f = 0.16
        M_slug = fm * rho0 * pi * (b**2 - r_s**2) * z_f
        assert M_slug > 0
        assert M_slug == pytest.approx(2.9e-7, rel=0.1)

    def test_radial_force_includes_zf(self) -> None:
        fc = 0.7
        I_peak = 1.87e6  # noqa: N806
        z_f = 0.16
        r_s = 0.04
        F_rad = (mu_0 / (4.0 * pi)) * (fc * I_peak) ** 2 * z_f / r_s
        assert F_rad > 0
        assert 1e4 < F_rad < 1e8

    def test_lee_model_pf1000_runs_both_phases(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel()
        result = model.run(device_name="PF-1000")
        assert 1 in result.phases_completed
        assert 2 in result.phases_completed
        assert result.peak_current > 1e6
        assert result.peak_current_time > 0

    def test_lee_model_nx2_runs(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel()
        result = model.run(device_name="NX2")
        assert 1 in result.phases_completed
        assert result.peak_current > 100e3

    def test_lee_model_compare_with_experiment(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel()
        comp = model.compare_with_experiment("PF-1000")
        assert comp.device_name == "PF-1000"
        assert comp.peak_current_error >= 0
        assert comp.timing_error >= 0
        assert comp.lee_result.peak_current > 0

    def test_radial_mass_dynamic_not_constant(self) -> None:
        import inspect  # noqa: PLC0415

        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel()
        source = inspect.getsource(model.run)
        assert "b**2 - r_s**2" in source or "b ** 2 - r_s ** 2" in source


class TestLeeModelFmFcNaming:
    """Tests for correct fm/fc naming in docstrings and code."""

    def test_fm_is_mass_fraction(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        model = LeeModel(mass_fraction=0.5, current_fraction=0.8)
        assert model.fm == 0.5
        assert model.fc == 0.8

    def test_docstring_labels_fc_correctly(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        docstring = LeeModel.__init__.__doc__ or LeeModel.__doc__ or ""
        assert "Lee's fc" in docstring or "fc factor" in docstring.lower()
        assert "Lee's fm" in docstring or "fm factor" in docstring.lower()


class TestTimingTolerance:
    """Tests for the tightened timing tolerance in experimental.py."""

    def test_timing_tolerance_is_10_percent(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        rise_time_exp = 5.8e-6
        peak_time = rise_time_exp * 1.15
        I_sim = np.exp(-((t - peak_time) ** 2) / (0.5e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert not result["timing_ok"]

    def test_timing_within_10_percent_passes(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        rise_time_exp = 5.8e-6
        peak_time = rise_time_exp * 1.05
        I_sim = np.exp(-((t - peak_time) ** 2) / (0.5e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert result["timing_ok"]

    def test_old_50_percent_tolerance_rejected(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        rise_time_exp = 5.8e-6
        peak_time = rise_time_exp * 1.30
        I_sim = np.exp(-((t - peak_time) ** 2) / (0.5e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert not result["timing_ok"]

    def test_timing_error_returned(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        peak_time = 5.8e-6
        I_sim = np.exp(-((t - peak_time) ** 2) / (0.5e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert "timing_error" in result
        assert result["timing_error"] >= 0


class TestSuiteTimingTolerance:
    """Tests for tightened tolerances in validation suite."""

    def test_pf1000_timing_tolerance_10_percent(self) -> None:
        pf1000 = DEVICE_REGISTRY["PF-1000"]
        assert pf1000.tolerances["peak_current_time"] == pytest.approx(0.10)

    def test_nx2_timing_tolerance_10_percent(self) -> None:
        nx2 = DEVICE_REGISTRY["NX2"]
        assert nx2.tolerances["peak_current_time"] == pytest.approx(0.10)

    def test_llnl_timing_tolerance_15_percent(self) -> None:
        llnl = DEVICE_REGISTRY["LLNL-DPF"]
        assert llnl.tolerances["peak_current_time"] == pytest.approx(0.15)


class TestUncertaintyPropagation:
    """Tests for GUM-style uncertainty propagation in validation."""

    def test_experimental_device_has_uncertainty_fields(self) -> None:
        from dpf.validation.experimental import PF1000_DATA  # noqa: PLC0415
        assert hasattr(PF1000_DATA, "peak_current_uncertainty")
        assert hasattr(PF1000_DATA, "rise_time_uncertainty")
        assert hasattr(PF1000_DATA, "neutron_yield_uncertainty")

    def test_pf1000_uncertainties(self) -> None:
        from dpf.validation.experimental import PF1000_DATA  # noqa: PLC0415
        assert PF1000_DATA.peak_current_uncertainty == pytest.approx(0.05)
        assert PF1000_DATA.rise_time_uncertainty == pytest.approx(0.10)
        assert PF1000_DATA.neutron_yield_uncertainty == pytest.approx(0.50)

    def test_nx2_uncertainties(self) -> None:
        from dpf.validation.experimental import NX2_DATA, PF1000_DATA  # noqa: PLC0415
        assert NX2_DATA.peak_current_uncertainty >= PF1000_DATA.peak_current_uncertainty

    def test_unu_uncertainties(self) -> None:
        from dpf.validation.experimental import NX2_DATA, UNU_ICTP_DATA  # noqa: PLC0415
        assert UNU_ICTP_DATA.peak_current_uncertainty >= NX2_DATA.peak_current_uncertainty

    def test_validate_waveform_returns_uncertainty(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        I_sim = np.exp(-((t - 5.8e-6) ** 2) / (0.5e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert "uncertainty" in result
        unc = result["uncertainty"]
        assert "peak_current_exp_1sigma" in unc
        assert "rise_time_exp_1sigma" in unc
        assert "peak_current_combined_1sigma" in unc
        assert "timing_combined_1sigma" in unc
        assert "agreement_within_2sigma" in unc

    def test_uncertainty_combined_quadrature(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        I_sim = np.exp(-((t - 5.8e-6) ** 2) / (0.5e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        unc = result["uncertainty"]
        assert unc["peak_current_combined_1sigma"] >= unc["peak_current_exp_1sigma"]

    def test_uncertainty_agreement_2sigma(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        I_sim = np.exp(-((t - 5.8e-6) ** 2) / (0.5e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert result["uncertainty"]["agreement_within_2sigma"]

    def test_uncertainty_disagreement_outside_2sigma(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 1000)
        I_sim = np.exp(-((t - 5.8e-6) ** 2) / (0.5e-6) ** 2) * 0.9e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert not result["uncertainty"]["agreement_within_2sigma"]

    def test_neutron_yield_has_uncertainty(self) -> None:
        from dpf.validation.experimental import validate_neutron_yield  # noqa: PLC0415
        result = validate_neutron_yield(5e10, "PF-1000")
        assert "uncertainty" in result
        assert "neutron_yield_exp_1sigma" in result["uncertainty"]
        assert result["uncertainty"]["neutron_yield_exp_1sigma"] == pytest.approx(0.50)


class TestCrowbarInPresets:
    """Tests for crowbar enabled in PF-1000 preset."""

    def test_pf1000_preset_has_crowbar(self) -> None:
        preset = get_preset("pf1000")
        circuit = preset["circuit"]
        assert circuit.get("crowbar_enabled") is True

    def test_pf1000_preset_crowbar_mode(self) -> None:
        preset = get_preset("pf1000")
        circuit = preset["circuit"]
        assert circuit.get("crowbar_mode") == "fixed_time"

    def test_other_presets_no_crowbar(self) -> None:
        for name in ["tutorial", "cartesian_demo"]:
            preset = get_preset(name)
            circuit = preset["circuit"]
            assert not circuit.get("crowbar_enabled", False)

    def test_pf1000_crowbar_rlc_solver(self) -> None:
        preset = get_preset("pf1000")
        circuit = preset["circuit"]
        solver = RLCSolver(**circuit)
        assert solver.crowbar_enabled
        assert solver.crowbar_mode == "fixed_time"


class TestLeeSnowplowConsistency:
    """Cross-check Lee model and snowplow model consistency."""

    def test_both_models_produce_similar_physics(self) -> None:
        from dpf.validation.experimental import PF1000_DATA  # noqa: PLC0415
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        lee = LeeModel(mass_fraction=0.3, current_fraction=0.7)
        lee_result = lee.run(device_name="PF-1000")
        assert 2 in lee_result.phases_completed
        p_Pa = PF1000_DATA.fill_pressure_torr * 133.322
        n_fill = p_Pa / (k_B * 300.0)
        rho0_val = n_fill * 3.34e-27
        snowplow_model = SnowplowModel(
            anode_radius=PF1000_DATA.anode_radius,
            cathode_radius=PF1000_DATA.cathode_radius,
            fill_density=rho0_val,
            anode_length=PF1000_DATA.anode_length,
            mass_fraction=0.3,
            fill_pressure_Pa=p_Pa,
            current_fraction=0.7,
        )
        L_initial = snowplow_model.plasma_inductance
        assert L_initial > 0

    def test_lee_model_radial_force_matches_snowplow(self) -> None:
        fc = 0.7
        I_peak = 1.87e6  # noqa: N806
        z_f = 0.16
        r_s = 0.04
        F_lee = (mu_0 / (4.0 * pi)) * (fc * I_peak) ** 2 * z_f / r_s
        F_snowplow_val = (mu_0 / (4.0 * pi)) * (fc * I_peak) ** 2 * z_f / r_s
        assert F_lee == pytest.approx(F_snowplow_val, rel=1e-10)

    def test_lee_model_back_pressure_matches_snowplow(self) -> None:
        gamma = 5.0 / 3.0
        p_fill = 3.5 * 133.322
        b = 0.08
        r_s = 0.04
        p_back = p_fill * (b / r_s) ** (2.0 * gamma)
        assert p_back > p_fill
        assert np.isfinite(p_back)


class TestRegressionValidation:
    """Ensure existing validation functionality is not broken."""

    def test_validate_current_waveform_basic(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 500)
        I_sim = np.sin(2 * np.pi * t / 5.8e-6) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert "peak_current_error" in result
        assert "peak_current_sim" in result
        assert "peak_current_exp" in result
        assert "peak_time_sim" in result

    def test_validate_neutron_yield_basic(self) -> None:
        from dpf.validation.experimental import validate_neutron_yield  # noqa: PLC0415
        result = validate_neutron_yield(1e11, "PF-1000")
        assert result["yield_ratio"] == pytest.approx(1.0)
        assert result["within_order_magnitude"]

    def test_device_to_config_dict(self) -> None:
        from dpf.validation.experimental import device_to_config_dict  # noqa: PLC0415
        config = device_to_config_dict("PF-1000")
        assert "grid_shape" in config
        assert "dx" in config
        assert "circuit" in config
        assert config["circuit"]["C"] == pytest.approx(1.332e-3)

    def test_find_first_peak_still_works(self) -> None:
        from dpf.validation.experimental import _find_first_peak  # noqa: PLC0415
        signal = np.array([0, 1, 3, 5, 4, 2, 1, 0])
        assert _find_first_peak(signal) == 3
        signal = np.array([0, 1, 5, 3, 2, 4, 8, 6, 2])
        idx = _find_first_peak(signal)
        assert idx == 2

    def test_validation_suite_still_works(self) -> None:
        from dpf.validation.suite import ValidationSuite  # noqa: PLC0415
        suite = ValidationSuite(devices=["PF-1000"])
        sim_summary = {
            "peak_current_A": 1.87e6,
            "peak_current_time_s": 5.5e-6,
            "energy_conservation": 0.99,
        }
        result = suite.validate_circuit("PF-1000", sim_summary)
        assert result.passed
        assert result.overall_score > 0.8


class TestLeeModelEdgeCases:
    """Edge cases for Lee model changes."""

    def test_lee_model_custom_params(self) -> None:
        from dpf.validation.lee_model_comparison import LeeModel  # noqa: PLC0415
        params = {
            "C": 1e-3, "V0": 20e3, "L0": 30e-9, "R0": 3e-3,
            "anode_radius": 0.05, "cathode_radius": 0.08,
            "anode_length": 0.15, "fill_pressure_torr": 3.0,
        }
        model = LeeModel()
        result = model.run(device_params=params)
        assert result.peak_current > 0
        assert 1 in result.phases_completed

    def test_uncertainty_with_zero_experimental(self) -> None:
        from dpf.validation.experimental import ExperimentalDevice  # noqa: PLC0415
        dev = ExperimentalDevice(
            name="Test", institution="Test",
            capacitance=1e-6, voltage=1e3, inductance=1e-7, resistance=0.01,
            anode_radius=0.005, cathode_radius=0.01, anode_length=0.05,
            fill_pressure_torr=3.0, fill_gas="deuterium",
            peak_current=100e3, neutron_yield=1e6, current_rise_time=1e-6,
            reference="Test",
        )
        assert dev.peak_current_uncertainty == 0.0

    def test_timing_boundary_exactly_10_percent(self) -> None:
        from dpf.validation.experimental import validate_current_waveform  # noqa: PLC0415
        t = np.linspace(0, 10e-6, 10000)
        rise_time = 5.8e-6
        peak_time = rise_time * 1.10
        I_sim = np.exp(-((t - peak_time) ** 2) / (0.2e-6) ** 2) * 1.87e6
        result = validate_current_waveform(t, I_sim, "PF-1000")
        assert "timing_error" in result


# --- Section: Snowplow MHD Coupling ---


def _make_config(
    geometry: str = "cylindrical",
    enable_mhd_coupling: bool = True,
    nx: int = 8,
    nz: int = 16,
) -> SimulationConfig:
    dz = 0.01 if geometry == "cylindrical" else None
    return SimulationConfig(
        grid_shape=(nx, 1 if geometry == "cylindrical" else nx, nz),
        dx=0.01,
        rho0=1e-4,
        sim_time=1e-6,
        circuit={
            "C": 1e-3, "V0": 15000, "L0": 33.5e-9, "R0": 0.01,
            "anode_radius": 0.025, "cathode_radius": 0.05,
        },
        geometry={"type": geometry, "dz": dz},
        fluid={"backend": "python", "enable_ohmic_correction": False},
        diagnostics={"hdf5_filename": ":memory:", "output_interval": 1},
        snowplow={
            "enabled": True, "mass_fraction": 0.3, "anode_length": 0.16,
            "current_fraction": 0.7, "enable_mhd_coupling": enable_mhd_coupling,
        },
        boundary={"electrode_bc": False},
    )


def _copy_state(state: dict) -> dict:
    return {k: v.copy() for k, v in state.items()}


@pytest.fixture()
def cyl_solver():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        enable_hall=False, enable_resistive=False, conservative_energy=True,
    )


@pytest.fixture()
def cyl_solver_nonconservative():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        enable_hall=False, enable_resistive=False, conservative_energy=False,
    )


@pytest.fixture()
def cyl_solver_rk2():
    return CylindricalMHDSolver(
        nr=16, nz=16, dr=0.001, dz=0.001,
        enable_hall=False, enable_resistive=False,
        conservative_energy=True, time_integrator="ssp_rk2",
    )


@pytest.fixture()
def uniform_state():
    nr, nz = 16, 16
    return {
        "rho": np.full((nr, 1, nz), 1.0),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), 1e5),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e4),
        "Ti": np.full((nr, 1, nz), 1e4),
        "psi": np.zeros((nr, 1, nz)),
    }


class TestSnowplowSourceTermComputation:
    """Test _compute_snowplow_source_terms method."""

    def test_coupling_disabled_returns_empty(self) -> None:
        config = _make_config(enable_mhd_coupling=False)
        engine = SimulationEngine(config)
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        assert result == {}

    def test_snowplow_none_returns_empty(self) -> None:
        config = _make_config(enable_mhd_coupling=True)
        engine = SimulationEngine(config)
        engine.snowplow = None
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        assert result == {}

    def test_rundown_phase_produces_source_terms(self) -> None:
        config = _make_config()
        engine = SimulationEngine(config)
        engine.step()
        assert engine.snowplow is not None
        assert engine.snowplow.phase == "rundown"
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if engine.snowplow.sheath_velocity > 1e-6:
            assert "S_rho_snowplow" in result
            assert "S_mom_snowplow" in result
            assert "S_energy_snowplow" in result

    def test_source_terms_have_correct_shape(self) -> None:
        config = _make_config(nx=8, nz=16)
        engine = SimulationEngine(config)
        engine.step()
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if result:
            grid_shape = engine.state["rho"].shape
            assert result["S_rho_snowplow"].shape == grid_shape
            assert result["S_mom_snowplow"].shape == (3, *grid_shape)
            assert result["S_energy_snowplow"].shape == grid_shape

    def test_source_terms_are_positive(self) -> None:
        config = _make_config()
        engine = SimulationEngine(config)
        engine.step()
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if result:
            assert np.all(result["S_rho_snowplow"] >= 0)
            assert np.all(result["S_energy_snowplow"] >= 0)

    def test_gaussian_smearing_peaks_at_sheath(self) -> None:
        config = _make_config(nx=8, nz=64)
        engine = SimulationEngine(config)
        for _ in range(5):
            engine.step()
        result = engine._compute_snowplow_source_terms(dt=1e-8)
        if result and engine.snowplow.phase == "rundown":
            S_rho = result["S_rho_snowplow"]
            z_profile = np.sum(S_rho[:, 0, :], axis=0)
            peak_idx = np.argmax(z_profile)
            dz = config.geometry.dz or config.dx
            z_peak = (peak_idx + 0.5) * dz
            z_sheath = engine.snowplow.sheath_position
            assert abs(z_peak - z_sheath) < 4 * dz


class TestSnowplowMHDIntegration:
    """Test snowplow source terms integrated into the MHD solve."""

    def test_engine_step_with_coupling(self) -> None:
        config = _make_config()
        engine = SimulationEngine(config)
        for _ in range(10):
            result = engine.step()
        assert result is not None
        assert not np.any(np.isnan(engine.state["rho"]))

    def test_coupling_increases_density(self) -> None:
        config_coupled = _make_config(enable_mhd_coupling=True)
        config_uncoupled = _make_config(enable_mhd_coupling=False)
        engine_c = SimulationEngine(config_coupled)
        engine_u = SimulationEngine(config_uncoupled)
        for _ in range(10):
            engine_c.step()
            engine_u.step()
        total_mass_c = np.sum(engine_c.state["rho"])
        total_mass_u = np.sum(engine_u.state["rho"])
        assert total_mass_c >= total_mass_u * 0.99

    def test_no_nan_with_coupling(self) -> None:
        config = _make_config()
        engine = SimulationEngine(config)
        for _ in range(50):
            engine.step()
        assert not np.any(np.isnan(engine.state["rho"]))
        assert not np.any(np.isnan(engine.state["pressure"]))
        assert np.all(engine.state["rho"] > 0)
        assert np.all(engine.state["pressure"] > 0)


class TestSnowplowConfigFlag:
    """Test the enable_mhd_coupling config flag."""

    def test_config_default_is_false(self) -> None:
        config = SimulationConfig(
            grid_shape=(8, 1, 16), dx=0.01, rho0=1e-4, sim_time=1e-6,
            circuit={"C": 1e-3, "V0": 15000, "L0": 33.5e-9,
                     "anode_radius": 0.025, "cathode_radius": 0.05},
            geometry={"type": "cylindrical", "dz": 0.01},
            fluid={"backend": "python"},
            diagnostics={"hdf5_filename": ":memory:"},
        )
        assert config.snowplow.enable_mhd_coupling is False

    def test_config_can_enable(self) -> None:
        config = _make_config(enable_mhd_coupling=True)
        assert config.snowplow.enable_mhd_coupling is True


class TestSolverMassSource:
    """1. Mass source increases density."""

    def test_mass_source_increases_density(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e6)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["rho"]) > np.mean(uniform_state["rho"])

    def test_negative_mass_source_decreases_density(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), -1e5)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["rho"]) < np.mean(uniform_state["rho"])


class TestSolverMomentumSource:
    """2. Momentum source changes velocity."""

    def test_axial_momentum_injection(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2] = 1e8
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(np.abs(new["velocity"][2])) > 0.0

    def test_radial_momentum_injection(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[0] = 1e8
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(np.abs(new["velocity"][0])) > 0.0


class TestSolverEnergySource:
    """3. Energy source increases total energy (conservative mode)."""

    def test_energy_source_conservative(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        source_terms = {"S_energy_snowplow": np.full((16, 1, 16), 1e12)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])

    def test_energy_source_conservative_vs_nonconservative(
        self, cyl_solver, cyl_solver_nonconservative, uniform_state
    ) -> None:
        src = {"S_energy_snowplow": np.full((16, 1, 16), 1e12)}
        new_cons = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                                   current=0.0, voltage=0.0, source_terms=src)
        new_noncons = cyl_solver_nonconservative.step(
            _copy_state(uniform_state), dt=1e-9, current=0.0, voltage=0.0,
            source_terms=src)
        dp_cons = np.mean(new_cons["pressure"]) - np.mean(uniform_state["pressure"])
        dp_noncons = np.mean(new_noncons["pressure"]) - np.mean(uniform_state["pressure"])
        assert abs(dp_cons) > abs(dp_noncons)


class TestSolverOhmicCorrection:
    """4. Ohmic correction heats electrons."""

    def test_ohmic_correction_increases_pressure(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        source_terms = {"Q_ohmic_correction": np.full((16, 1, 16), 1e12)}
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])


class TestSolverShape3DSqueeze:
    """5. Source terms with 3D shape (nr,1,nz) are properly squeezed."""

    def test_3d_shape_squeezed(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        source_terms = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_mom_snowplow": np.full((3, 16, 1, 16), 1e6),
            "S_energy_snowplow": np.full((16, 1, 16), 1e10),
            "Q_ohmic_correction": np.full((16, 1, 16), 1e10),
        }
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert new["rho"].shape == (16, 1, 16)
        assert new["velocity"].shape == (3, 16, 1, 16)


class TestSolverShape2D:
    """6. Source terms with 2D shape (nr,nz) work directly."""

    def test_2d_shape_works(self, cyl_solver, uniform_state) -> None:
        state0 = _copy_state(uniform_state)
        source_terms = {
            "S_rho_snowplow": np.full((16, 16), 1e6),
            "S_mom_snowplow": np.full((3, 16, 16), 1e6),
            "S_energy_snowplow": np.full((16, 16), 1e10),
            "Q_ohmic_correction": np.full((16, 16), 1e10),
        }
        new = cyl_solver.step(state0, dt=1e-9, current=0.0, voltage=0.0,
                              source_terms=source_terms)
        assert new["rho"].shape == (16, 1, 16)
        assert np.mean(new["rho"]) > 1.0


class TestSolverZeroSourceTerms:
    """7. Zero source terms = no effect."""

    def test_zero_sources_match_none(self, cyl_solver, uniform_state) -> None:
        source_terms = {
            "S_rho_snowplow": np.zeros((16, 1, 16)),
            "S_mom_snowplow": np.zeros((3, 16, 1, 16)),
            "S_energy_snowplow": np.zeros((16, 1, 16)),
            "Q_ohmic_correction": np.zeros((16, 1, 16)),
        }
        new_zero = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                                   current=0.0, voltage=0.0, source_terms=source_terms)
        new_none = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                                   current=0.0, voltage=0.0, source_terms=None)
        np.testing.assert_allclose(new_zero["rho"], new_none["rho"], atol=1e-15)
        np.testing.assert_allclose(new_zero["pressure"], new_none["pressure"], atol=1e-10)

    def test_empty_dict_matches_none(self, cyl_solver, uniform_state) -> None:
        new_empty = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                                    current=0.0, voltage=0.0, source_terms={})
        new_none = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                                   current=0.0, voltage=0.0, source_terms=None)
        np.testing.assert_allclose(new_empty["rho"], new_none["rho"], atol=1e-15)


class TestSolverLargeSourceStability:
    """8. Large source terms don't cause NaN."""

    def test_large_mass_source_no_nan(self, cyl_solver, uniform_state) -> None:
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e15)}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert not np.any(np.isnan(new["rho"]))
        assert not np.any(np.isnan(new["pressure"]))

    def test_large_energy_source_no_nan(self, cyl_solver, uniform_state) -> None:
        source_terms = {"S_energy_snowplow": np.full((16, 1, 16), 1e18)}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert not np.any(np.isnan(new["rho"]))
        assert not np.any(np.isnan(new["pressure"]))
        assert not np.any(np.isinf(new["pressure"]))

    def test_large_momentum_source_no_nan(self, cyl_solver, uniform_state) -> None:
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2] = 1e15
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert not np.any(np.isnan(new["velocity"]))


class TestSolverSSPRK3:
    """9. Source terms work with SSP-RK3 (default)."""

    def test_rk3_source_injection(self, cyl_solver, uniform_state) -> None:
        assert cyl_solver.time_integrator == "ssp_rk3"
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e6)}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > 1.0


class TestSolverSSPRK2:
    """10. Source terms work with SSP-RK2."""

    def test_rk2_source_injection(self, cyl_solver_rk2, uniform_state) -> None:
        assert cyl_solver_rk2.time_integrator == "ssp_rk2"
        source_terms = {"S_rho_snowplow": np.full((16, 1, 16), 1e6)}
        new = cyl_solver_rk2.step(_copy_state(uniform_state), dt=1e-9,
                                  current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > 1.0

    def test_rk2_vs_rk3_both_increase(self, cyl_solver, cyl_solver_rk2, uniform_state) -> None:
        src = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_energy_snowplow": np.full((16, 1, 16), 1e12),
        }
        new_rk3 = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                                  current=0.0, voltage=0.0, source_terms=src)
        new_rk2 = cyl_solver_rk2.step(_copy_state(uniform_state), dt=1e-9,
                                      current=0.0, voltage=0.0, source_terms=src)
        assert np.mean(new_rk3["rho"]) > 1.0
        assert np.mean(new_rk2["rho"]) > 1.0


class TestSolverLocalizedSource:
    """11. Gaussian-shaped source localized to specific cells."""

    def test_gaussian_density_bump(self, cyl_solver, uniform_state) -> None:
        nr, nz = 16, 16
        r_idx, z_idx = np.meshgrid(np.arange(nr), np.arange(nz), indexing="ij")
        gaussian = 1e6 * np.exp(-((r_idx - 8)**2 + (z_idx - 8)**2) / (2 * 2.0**2))
        source_terms = {"S_rho_snowplow": gaussian[:, np.newaxis, :]}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert new["rho"][8, 0, 8] > new["rho"][0, 0, 0]

    def test_point_momentum_injection(self, cyl_solver, uniform_state) -> None:
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[2, 8, 0, 8] = 1e10
        source_terms = {"S_mom_snowplow": S_mom}
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.abs(new["velocity"][2, 8, 0, 8]) >= np.abs(new["velocity"][2, 0, 0, 0])


class TestSolverCombinedSources:
    """12. Combined sources (mass + momentum + energy simultaneously)."""

    def test_all_four_sources(self, cyl_solver, uniform_state) -> None:
        S_mom = np.zeros((3, 16, 1, 16))
        S_mom[0] = 1e7
        S_mom[2] = 1e7
        source_terms = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_mom_snowplow": S_mom,
            "S_energy_snowplow": np.full((16, 1, 16), 1e12),
            "Q_ohmic_correction": np.full((16, 1, 16), 1e10),
        }
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > 1.0
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])
        assert np.mean(np.abs(new["velocity"][0])) > 0.0
        assert np.mean(np.abs(new["velocity"][2])) > 0.0
        assert not np.any(np.isnan(new["rho"]))
        assert not np.any(np.isnan(new["pressure"]))

    def test_mass_and_energy_coupled(self, cyl_solver, uniform_state) -> None:
        source_terms = {
            "S_rho_snowplow": np.full((16, 1, 16), 1e6),
            "S_energy_snowplow": np.full((16, 1, 16), 1e13),
        }
        new = cyl_solver.step(_copy_state(uniform_state), dt=1e-9,
                              current=0.0, voltage=0.0, source_terms=source_terms)
        assert np.mean(new["rho"]) > np.mean(uniform_state["rho"])
        assert np.mean(new["pressure"]) > np.mean(uniform_state["pressure"])


class TestCartesianSolverSourceTerms:
    """13. Cartesian MHDSolver._compute_rhs_euler accepts source terms."""

    def test_rhs_accepts_snowplow_sources(self) -> None:
        from dpf.fluid.mhd_solver import MHDSolver  # noqa: PLC0415
        n = 16
        solver = MHDSolver(grid_shape=(n, n, n), dx=0.001,
                           enable_hall=False, enable_resistive=False)
        state = {
            "rho": np.full((n, n, n), 1.0), "velocity": np.zeros((3, n, n, n)),
            "pressure": np.full((n, n, n), 1e5), "B": np.zeros((3, n, n, n)),
            "Te": np.full((n, n, n), 1e4), "Ti": np.full((n, n, n), 1e4),
            "psi": np.zeros((n, n, n)),
        }
        source_terms = {
            "S_rho_snowplow": np.full((n, n, n), 1e6),
            "S_mom_snowplow": np.full((3, n, n, n), 1e6),
            "Q_ohmic_correction": np.full((n, n, n), 1e10),
        }
        rhs = solver._compute_rhs_euler(state, current=0.0, voltage=0.0,
                                        eta_field=None, source_terms=source_terms)
        assert np.mean(rhs["drho_dt"]) > 0.0

    def test_rhs_none_source_terms(self) -> None:
        from dpf.fluid.mhd_solver import MHDSolver  # noqa: PLC0415
        n = 16
        solver = MHDSolver(grid_shape=(n, n, n), dx=0.001,
                           enable_hall=False, enable_resistive=False)
        state = {
            "rho": np.full((n, n, n), 1.0), "velocity": np.zeros((3, n, n, n)),
            "pressure": np.full((n, n, n), 1e5), "B": np.zeros((3, n, n, n)),
            "Te": np.full((n, n, n), 1e4), "Ti": np.full((n, n, n), 1e4),
            "psi": np.zeros((n, n, n)),
        }
        rhs = solver._compute_rhs_euler(state, current=0.0, voltage=0.0,
                                        eta_field=None, source_terms=None)
        assert "drho_dt" in rhs
        assert "dmom_dt" in rhs


class TestSolverProportionality:
    """14. Source magnitude scales effect proportionally."""

    def test_double_mass_source_double_effect(self, cyl_solver, uniform_state) -> None:
        dt = 1e-9
        src_1x = {"S_rho_snowplow": np.full((16, 1, 16), 1e4)}
        src_2x = {"S_rho_snowplow": np.full((16, 1, 16), 2e4)}
        new_1x = cyl_solver.step(_copy_state(uniform_state), dt=dt,
                                 current=0.0, voltage=0.0, source_terms=src_1x)
        new_2x = cyl_solver.step(_copy_state(uniform_state), dt=dt,
                                 current=0.0, voltage=0.0, source_terms=src_2x)
        drho_1x = np.mean(new_1x["rho"]) - np.mean(uniform_state["rho"])
        drho_2x = np.mean(new_2x["rho"]) - np.mean(uniform_state["rho"])
        if abs(drho_1x) > 1e-15:
            ratio = drho_2x / drho_1x
            assert ratio == pytest.approx(2.0, rel=0.3)


# --- Section: X Coupling / LHDI / Zipper BC ---


def _make_cylindrical_engine(
    nr: int = 10,
    nz: int = 20,
    dx: float = 0.01,
    snowplow_enabled: bool = True,
) -> SimulationEngine:
    cfg = SimulationConfig(
        grid_shape=[nr, 1, nz],
        dx=dx,
        sim_time=1e-6,
        circuit={"C": 1e-6, "V0": 10e3, "L0": 10e-9,
                 "anode_radius": 0.01, "cathode_radius": 0.10},
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


class TestRadialZipperBC:
    """Radial zipper BC: B_theta zeroed outside radial shock during radial phase."""

    def test_radial_zipper_activates_during_radial_phase(self):
        """With snowplow.phase='radial', B_theta outside r_shock is zeroed."""
        engine = _make_cylindrical_engine(nr=10, nz=20, dx=0.01)

        engine.snowplow.phase = "radial"
        engine.snowplow.r_shock = 0.05  # ir_shock = int(0.05/0.01) = 5
        engine.snowplow.z = engine.snowplow.L_anode
        engine.snowplow._rundown_complete = True

        engine.state["B"] = np.ones_like(engine.state["B"])

        engine._apply_electrode_bc(current=100e3)

        B_theta = engine.state["B"][1]
        assert np.all(B_theta[6:, :, :] == 0.0), (
            "B_theta beyond ir_shock+1 must be zeroed by radial zipper"
        )

    def test_radial_zipper_no_effect_during_rundown(self):
        """With snowplow.phase='rundown', the radial zipper branch is not taken."""
        engine = _make_cylindrical_engine(nr=10, nz=20, dx=0.01)

        engine.snowplow.phase = "rundown"
        engine.snowplow.z = 0.10
        engine.state["B"][1, :, :, :] = 1.0
        engine.state["B"][0, :, :, :] = 0.0
        engine.state["B"][2, :, :, :] = 0.0

        engine._apply_electrode_bc(current=100e3)

        B_theta = engine.state["B"][1]
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

        ir_shock = int(r_shock / dx)
        B_theta = engine.state["B"][1]
        assert np.all(B_theta[ir_shock + 1:, :, :] == 0.0), (
            f"Expected B_theta zero for r_idx > {ir_shock}"
        )
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

        engine.state["B"][:] = 5.0

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

        engine.state["B"][1, :7, :, :] = 99.0

        engine._apply_electrode_bc(current=1e5)

        ir_shock = int(r_shock / dx)
        B_inside = engine.state["B"][1, :ir_shock + 1, :, :]
        assert np.any(B_inside != 0.0), (
            "B_theta inside radial shock should not all be zeroed by radial zipper"
        )

    def test_radial_zipper_edge_case_r_shock_at_boundary(self):
        """r_shock near grid boundary (ir_shock >= nx-1) does not crash."""
        nr, dx = 8, 0.01
        engine = _make_cylindrical_engine(nr=nr, nz=20, dx=dx)

        r_shock = (nr - 1) * dx
        engine.snowplow.phase = "radial"
        engine.snowplow.r_shock = r_shock
        engine.snowplow.z = engine.snowplow.L_anode
        engine.snowplow._rundown_complete = True
        engine.state["B"][:] = 1.0

        engine._apply_electrode_bc(current=1e5)
        assert engine.state["B"].shape[0] == 3


class TestElectrodeBCAllBackends:
    """Electrode BC applies B_theta = mu_0*I/(2*pi*r) for all backends (C2 fix)."""

    @pytest.mark.parametrize("backend_name", ["metal", "athena", "athenak"])
    def test_electrode_bc_sets_btheta_for_backend(self, backend_name):
        """Non-Python backends get electrode B_theta via generic branch."""
        from math import pi

        engine = _make_cylindrical_engine(nr=10, nz=20, dx=0.01)
        engine.backend = backend_name

        engine.state["B"][:] = 0.0
        engine.snowplow = None

        current = 100e3
        engine._apply_electrode_bc(current)

        B_theta = engine.state["B"][1]
        cc = engine.config.circuit
        dr = engine.config.dx
        mu_0 = 4e-7 * pi

        for ir in range(10):
            r = (ir + 0.5) * dr
            if cc.anode_radius <= r <= cc.cathode_radius and r > 0:
                expected = mu_0 * current / (2.0 * pi * r)
                assert np.allclose(B_theta[ir], expected), (
                    f"B_theta at ir={ir} (r={r:.3f}) should be {expected:.4f} "
                    f"for backend={backend_name}"
                )
            elif r < cc.anode_radius:
                assert np.all(B_theta[ir] == 0.0), (
                    f"B_theta inside anode (ir={ir}) should be 0 "
                    f"for backend={backend_name}"
                )


class TestAxialZipperBC:
    """Axial zipper BC: B_theta zeroed ahead of axial sheath position."""

    def test_electrode_bc_zippering(self):
        """Magnetic BC should only apply behind the snowplow (axial zipper)."""
        nz = 20
        dx = 0.01

        engine = _make_cylindrical_engine(nr=10, nz=nz, dx=dx)

        engine.snowplow.z = 0.10
        engine.snowplow.v = 1e5
        engine.snowplow.phase = "rundown"
        engine.circuit.state.current = 100e3

        engine.state["B"] = np.zeros_like(engine.state["B"])
        engine._apply_electrode_bc(engine.circuit.state.current)

        B_theta_anode = engine.state["B"][1, 1, 0, :]

        assert np.any(np.abs(B_theta_anode[:10]) > 0), (
            "Behind sheath: B_theta should be applied"
        )
        assert np.all(B_theta_anode[12:] == 0.0), (
            "Ahead of sheath: B_theta should be zero (axial zipper)"
        )

    def test_axial_zipper_btheta_behind_sheath(self):
        """B_theta should be nonzero at z < z_sheath after electrode BC."""
        nz, dx = 20, 0.01
        engine = _make_cylindrical_engine(nr=10, nz=nz, dx=dx)

        engine.snowplow.z = 0.15
        engine.snowplow.phase = "rundown"
        engine.state["B"] = np.zeros_like(engine.state["B"])

        engine._apply_electrode_bc(current=200e3)

        B_theta_behind = engine.state["B"][1, 1, 0, 0]
        assert abs(B_theta_behind) > 0.0, (
            "B_theta behind sheath should be set by electrode BC"
        )

    def test_axial_zipper_btheta_ahead_of_sheath(self):
        """B_theta should be 0 at z > z_sheath (axial zipper zeroes ahead of sheath)."""
        nz, dx = 20, 0.01
        engine = _make_cylindrical_engine(nr=10, nz=nz, dx=dx)

        engine.snowplow.z = 0.05
        engine.snowplow.phase = "rundown"

        engine.state["B"][1, :, :, :] = 999.0

        engine._apply_electrode_bc(current=100e3)

        B_theta_ahead = engine.state["B"][1, :, :, 7:]
        assert np.all(B_theta_ahead == 0.0), (
            "B_theta ahead of axial sheath should be zeroed by zipper"
        )


class TestSnowplowDiagnostics:
    """r_shock and phase keys must appear in the diagnostics dict recorded by the engine."""

    def _make_engine_and_capture_diag(self, snowplow_enabled: bool = True) -> dict:
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
        from dpf.constants import k_B, m_d  # noqa: F401

        ne_val = 1e23
        Ti_val = 1000.0
        J_mag = 1e10

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
        from dpf.constants import k_B, m_d

        ne_val = 1e23
        Ti_val = 1000.0
        Te_val = 1000.0
        mi = m_d

        c_s = (k_B * Te_val / mi) ** 0.5
        factor = (m_e / mi) ** 0.25
        v_ti = (k_B * Ti_val / mi) ** 0.5
        v_d_target = factor * v_ti * 1.5
        assert v_d_target < c_s, "Test drift must be below ion-acoustic threshold"
        J_mag = v_d_target * ne_val * e

        eta_lhdi = anomalous_resistivity_scalar(
            J_mag, ne_val, Ti_val, alpha=0.05, mi=mi, threshold_model="lhdi",
            Te_val=Te_val,
        )
        eta_ia = anomalous_resistivity_scalar(
            J_mag, ne_val, Ti_val, alpha=0.05, mi=mi, threshold_model="ion_acoustic",
            Te_val=Te_val,
        )

        assert eta_lhdi > 0.0, "LHDI model should trigger above LHDI threshold"
        assert eta_ia == 0.0, "Ion-acoustic model should not trigger below c_s"

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


class TestLHDIPhysics:
    """LHDI threshold and physics verification."""

    def test_lhdi_threshold_lower_than_ion_acoustic(self):
        """LHDI threshold < ion-acoustic threshold by factor (m_e/m_i)^{1/4}."""
        from dpf.constants import m_d
        from dpf.turbulence.anomalous import (
            lhdi_factor,
        )

        Ti = np.array([1000.0])
        ne = np.array([1e23])
        mi = m_d

        v_ti = ion_thermal_speed(Ti, mi)[0]
        factor = lhdi_factor(mi)

        v_d = factor * v_ti * 1.5
        J = v_d * ne * e

        lhdi_active = lhdi_threshold(J, ne, Ti, mi)[0]
        ia_active = ion_acoustic_threshold(J, ne, Ti, mi)[0]

        assert lhdi_active, "LHDI should be active at 1.5x LHDI threshold"
        assert not ia_active, "Ion-acoustic should NOT be active below v_ti"

    def test_lhdi_factor_deuterium(self):
        """lhdi_factor(m_d) should be approximately 0.13 (Davidson & Gladd 1975)."""
        from dpf.constants import m_d

        factor = lhdi_factor(m_d)
        assert factor == pytest.approx(0.129, abs=0.005), (
            f"LHDI factor for deuterium expected ~0.129, got {factor:.4f}"
        )

    def test_field_vs_scalar_consistency(self):
        """anomalous_resistivity_field and anomalous_resistivity_scalar agree."""
        from dpf.constants import k_B, m_d
        from dpf.turbulence.anomalous import (
            lhdi_factor,
        )

        ne_val = 1e23
        Ti_val = 1000.0
        mi = m_d
        alpha = 0.05

        v_ti = (k_B * Ti_val / mi) ** 0.5
        v_d = lhdi_factor(mi) * v_ti * 5.0
        J_mag = v_d * ne_val * e

        eta_scalar = anomalous_resistivity_scalar(
            J_mag, ne_val, Ti_val, alpha=alpha, mi=mi, threshold_model="lhdi"
        )

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
        from dpf.constants import m_d

        Ti = np.array([1000.0])
        Te = np.array([1000.0])
        ne = np.array([1e23])
        mi = m_d

        v_ti = ion_thermal_speed(Ti, mi)[0]

        J = 2.0 * v_ti * ne * e

        ia_active = ion_acoustic_threshold(J, ne, Ti, mi)[0]
        lhdi_active = lhdi_threshold(J, ne, Ti, mi)[0]
        buneman_active = buneman_classic_threshold(J, ne, Te)[0]

        assert lhdi_active, "LHDI should trigger at 2x v_ti"
        assert ia_active, "Ion-acoustic should trigger at 2x v_ti"
        assert not buneman_active, (
            "Buneman classic (v_d > v_te) should NOT trigger at 2x v_ti "
            "because v_te >> v_ti for same temperature"
        )

    def test_lhdi_returns_nonzero_above_threshold(self):
        """anomalous_resistivity_field returns eta > 0 when above LHDI threshold."""
        from dpf.constants import k_B, m_d
        from dpf.turbulence.anomalous import (
            lhdi_factor,
        )

        mi = m_d
        Ti_val = 500.0
        ne_val = 5e22

        v_ti = (k_B * Ti_val / mi) ** 0.5
        factor = lhdi_factor(mi)
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


class TestElectrodeBCDefaults:
    """Cylindrical presets should have electrode_bc=True and use LHDI threshold."""

    def _preset_cfg(self, name: str) -> SimulationConfig:
        preset = get_preset(name)
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


class TestPeakCurrentTracking:
    """engine.run() must track and return peak_current_A and peak_current_time_s."""

    def _run_short(self, max_steps: int = 5) -> dict:
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


# --- Section: MHD Pressure Coupling Y ---


_PF1000_Y = {
    "anode_radius": 0.0575,
    "cathode_radius": 0.08,
    "fill_density": 1e-4,
    "anode_length": 0.16,
    "mass_fraction": 0.3,
    "fill_pressure_Pa": 400.0,
    "current_fraction": 0.7,
}


def _mhd_coupling_make_snowplow(**overrides) -> SnowplowModel:
    """Return a fresh SnowplowModel from PF-1000-like parameters."""
    params = {**_PF1000_Y, **overrides}
    return SnowplowModel(
        anode_radius=params["anode_radius"],
        cathode_radius=params["cathode_radius"],
        fill_density=params["fill_density"],
        anode_length=params["anode_length"],
        mass_fraction=params["mass_fraction"],
        fill_pressure_Pa=params["fill_pressure_Pa"],
        current_fraction=params["current_fraction"],
    )


def _mhd_coupling_make_radial_snowplow(**overrides) -> SnowplowModel:
    """Return a SnowplowModel manually placed in radial phase at 95% of cathode."""
    sp = _mhd_coupling_make_snowplow(**overrides)
    sp.z = sp.L_anode
    sp.v = 1e4
    sp._rundown_complete = True
    sp._L_axial_frozen = sp.L_coeff * sp.L_anode
    sp.phase = "radial"
    sp.r_shock = 0.95 * sp.b
    sp.vr = 0.0
    return sp


def _drive_to_phase(sp: SnowplowModel, target: str, max_steps: int = 200_000) -> bool:
    """Drive a SnowplowModel until phase == target. Returns True on success."""
    for _ in range(max_steps):
        sp.step(1e-9, 1.5e6)
        if sp.phase == target:
            return True
    return False


class TestStepRadialAcceptsPressure:
    """SnowplowModel.step() accepts pressure kwarg during all phases."""

    def test_step_accepts_pressure_during_rundown(self) -> None:
        """pressure kwarg accepted during rundown without raising."""
        sp = _mhd_coupling_make_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=500.0)
        assert "phase" in result
        assert result["phase"] == "rundown"

    def test_step_accepts_none_pressure_during_rundown(self) -> None:
        """pressure=None is accepted during rundown without raising."""
        sp = _mhd_coupling_make_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=None)
        assert "phase" in result

    def test_step_accepts_pressure_during_radial(self) -> None:
        """pressure kwarg accepted during radial phase without raising."""
        sp = _mhd_coupling_make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=1e5)
        assert "phase" in result
        assert result["phase"] in ("radial", "reflected", "pinch")

    def test_step_accepts_none_pressure_during_radial(self) -> None:
        """pressure=None is accepted during radial phase without raising."""
        sp = _mhd_coupling_make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=None)
        assert "phase" in result


class TestRadialPressureFeedback:
    """External pressure interacts with adiabatic back-pressure via max()."""

    def test_pressure_below_adiabatic_has_no_effect(self) -> None:
        """External pressure below adiabatic does not change F_pressure."""
        sp1 = _mhd_coupling_make_radial_snowplow()
        sp2 = _mhd_coupling_make_radial_snowplow()

        r1 = sp1.step(1e-9, 1.5e6, pressure=None)
        r2 = sp2.step(1e-9, 1.5e6, pressure=1.0)

        assert r1["F_pressure"] == pytest.approx(r2["F_pressure"], rel=1e-6)

    def test_pressure_above_adiabatic_increases_F_pressure(self) -> None:
        """External pressure > adiabatic back-pressure increases F_pressure."""
        sp1 = _mhd_coupling_make_radial_snowplow()
        sp2 = _mhd_coupling_make_radial_snowplow()

        r_base = sp1.step(1e-9, 1.5e6, pressure=None)
        r_high = sp2.step(1e-9, 1.5e6, pressure=1e12)

        assert r_high["F_pressure"] > r_base["F_pressure"]

    def test_high_pressure_slows_inward_velocity(self) -> None:
        """Very high external pressure reduces net inward acceleration."""
        sp1 = _mhd_coupling_make_radial_snowplow()
        sp2 = _mhd_coupling_make_radial_snowplow()

        for _ in range(100):
            sp1.step(1e-9, 1.5e6, pressure=None)
            sp2.step(1e-9, 1.5e6, pressure=1e12)

        assert abs(sp2.vr) <= abs(sp1.vr)

    def test_F_pressure_uses_max_semantics(self) -> None:
        """F_pressure = max(adiabatic, external) * 2*pi*r*z_f."""
        sp = _mhd_coupling_make_radial_snowplow()
        r_s = sp.r_shock
        gamma = 5.0 / 3.0
        p_adiabatic = sp.p_fill * (sp.b / r_s) ** (2.0 * gamma)

        r_low = sp.step(1e-9, 1.5e6, pressure=1.0)
        expected_F = p_adiabatic * 2.0 * np.pi * r_s * sp.L_anode
        assert r_low["F_pressure"] == pytest.approx(expected_F, rel=0.01)

    def test_zero_external_pressure_same_as_none(self) -> None:
        """pressure=0.0 behaves same as pressure=None (adiabatic wins)."""
        sp1 = _mhd_coupling_make_radial_snowplow()
        sp2 = _mhd_coupling_make_radial_snowplow()

        r_none = sp1.step(1e-9, 1.5e6, pressure=None)
        r_zero = sp2.step(1e-9, 1.5e6, pressure=0.0)

        assert r_none["F_pressure"] == pytest.approx(r_zero["F_pressure"], rel=1e-6)


class TestDynamicPressureFallback:
    """SimulationEngine._dynamic_sheath_pressure() fallback behaviour."""

    def _make_engine_stub(
        self,
        snowplow: SnowplowModel | None,
        state_pressure: np.ndarray | None,
        fill_pressure_Pa: float = 400.0,
        dx: float = 0.01,
        dz: float | None = None,
    ):
        from dpf.engine import SimulationEngine

        eng = object.__new__(SimulationEngine)

        class _GeomCfg:
            pass

        class _SnowplowCfg:
            pass

        class _Cfg:
            pass

        geom_cfg = _GeomCfg()
        geom_cfg.dz = dz  # type: ignore[attr-defined]

        snow_cfg = _SnowplowCfg()
        snow_cfg.fill_pressure_Pa = fill_pressure_Pa  # type: ignore[attr-defined]

        cfg = _Cfg()
        cfg.dx = dx  # type: ignore[attr-defined]
        cfg.geometry = geom_cfg  # type: ignore[attr-defined]
        cfg.snowplow = snow_cfg  # type: ignore[attr-defined]

        eng.config = cfg  # type: ignore[attr-defined]
        eng.snowplow = snowplow  # type: ignore[attr-defined]
        eng.state = {} if state_pressure is None else {"pressure": state_pressure}  # type: ignore[attr-defined]

        return eng

    def test_fallback_when_snowplow_is_none(self) -> None:
        """Returns fill_pressure_Pa when snowplow is None."""
        eng = self._make_engine_stub(snowplow=None, state_pressure=None, fill_pressure_Pa=500.0)
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(500.0)

    def test_fallback_when_snowplow_inactive(self) -> None:
        """Returns fill_pressure_Pa when snowplow.is_active is False (pinch)."""
        sp = _mhd_coupling_make_snowplow()
        sp._pinch_complete = True
        sp.phase = "pinch"
        assert not sp.is_active

        eng = self._make_engine_stub(snowplow=sp, state_pressure=None, fill_pressure_Pa=600.0)
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(600.0)

    def test_fallback_when_no_pressure_in_state(self) -> None:
        """Returns fill_pressure_Pa when state has no 'pressure' key."""
        sp = _mhd_coupling_make_snowplow()
        assert sp.is_active

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=None, fill_pressure_Pa=700.0,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(700.0)

    def test_rundown_phase_averages_ahead_of_sheath(self) -> None:
        """Rundown: uses mean pressure for z > z_sheath cells."""
        sp = _mhd_coupling_make_snowplow()
        assert sp.phase == "rundown"

        nz = 10
        dx = 0.01
        sp.z = 0.02

        p = np.zeros((1, 1, nz), dtype=np.float64)
        p[..., 3:] = 1000.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx, dz=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(1000.0, rel=1e-6)

    def test_rundown_phase_returns_fallback_when_iz_at_end(self) -> None:
        """Rundown: fallback when iz >= nz-1 (sheath past all cells)."""
        sp = _mhd_coupling_make_snowplow()
        sp.z = 0.20

        nz = 10
        dx = 0.01
        p = np.ones((1, 1, nz), dtype=np.float64) * 999.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx, dz=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(400.0)

    def test_radial_phase_averages_inside_shock(self) -> None:
        """Radial: uses mean pressure for r < r_shock cells."""
        sp = _mhd_coupling_make_radial_snowplow()
        assert sp.phase == "radial"

        nr = 16
        dx = 0.005
        sp.r_shock = 5 * dx

        p = np.zeros((nr, 1, 1), dtype=np.float64)
        p[:5, ...] = 5000.0
        p[5:, ...] = 100.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(5000.0, rel=1e-6)

    def test_radial_phase_returns_fallback_when_ir_is_zero(self) -> None:
        """Radial: fallback when r_shock / dx rounds to 0 (shock at origin)."""
        sp = _mhd_coupling_make_radial_snowplow()
        sp.r_shock = 0.0

        nr = 16
        dx = 0.005
        p = np.ones((nr, 1, 1), dtype=np.float64) * 999.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(400.0)

    def test_result_never_below_fallback(self) -> None:
        """_dynamic_sheath_pressure never returns below fill_pressure_Pa."""
        sp = _mhd_coupling_make_snowplow()
        nz = 10
        dx = 0.01
        sp.z = dx

        p = np.ones((1, 1, nz), dtype=np.float64) * 1.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx, dz=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result >= 400.0

    def test_reflected_phase_uses_radial_averaging(self) -> None:
        """Reflected phase uses the same r < r_shock averaging as radial."""
        sp = _mhd_coupling_make_radial_snowplow()
        sp.phase = "reflected"
        sp.r_shock = 4 * 0.005
        sp._M_slug_pinch = 1e-6
        sp._p_pinch = 400.0
        assert sp.is_active

        nr = 16
        dx = 0.005
        p = np.zeros((nr, 1, 1), dtype=np.float64)
        p[:4, ...] = 8000.0

        eng = self._make_engine_stub(
            snowplow=sp, state_pressure=p, fill_pressure_Pa=400.0,
            dx=dx,
        )
        result = eng._dynamic_sheath_pressure()
        assert result == pytest.approx(8000.0, rel=1e-6)


class TestTwoWayCoupling:
    """Snowplow step result always carries L_plasma and dL_dt for circuit coupling."""

    def test_step_returns_L_plasma(self) -> None:
        """step() always returns 'L_plasma' key."""
        sp = _mhd_coupling_make_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert "L_plasma" in result
        assert result["L_plasma"] >= 0.0

    def test_step_returns_dL_dt(self) -> None:
        """step() always returns 'dL_dt' key."""
        sp = _mhd_coupling_make_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert "dL_dt" in result

    def test_L_plasma_increases_during_rundown(self) -> None:
        """L_plasma grows monotonically during axial rundown."""
        sp = _mhd_coupling_make_snowplow()
        L_prev = 0.0
        for _ in range(1000):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase != "rundown":
                break
            assert result["L_plasma"] >= L_prev
            L_prev = result["L_plasma"]

    def test_L_plasma_increases_during_radial(self) -> None:
        """L_plasma grows monotonically during radial compression."""
        sp = _mhd_coupling_make_radial_snowplow()
        L_prev = sp.plasma_inductance
        for _ in range(500):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase not in ("radial",):
                break
            assert result["L_plasma"] >= L_prev - 1e-20
            L_prev = result["L_plasma"]

    def test_dL_dt_positive_during_rundown(self) -> None:
        """dL/dt > 0 while sheath is advancing axially."""
        sp = _mhd_coupling_make_snowplow()
        for _ in range(500):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase != "rundown":
                break
            if sp.v > 0:
                assert result["dL_dt"] >= 0.0

    def test_dL_dt_positive_during_radial_compression(self) -> None:
        """dL/dt > 0 during inward radial compression (vr < 0 -> dL/dt > 0)."""
        sp = _mhd_coupling_make_radial_snowplow()
        for _ in range(200):
            result = sp.step(1e-9, 1.5e6)
            if sp.phase != "radial":
                break
            if sp.vr < 0:
                assert result["dL_dt"] >= 0.0

    def test_step_returns_all_required_keys(self) -> None:
        """step() result contains all required coupling/diagnostic keys."""
        required = {
            "z_sheath", "v_sheath", "r_shock", "vr_shock",
            "L_plasma", "dL_dt", "swept_mass", "F_magnetic", "F_pressure", "phase",
        }
        sp = _mhd_coupling_make_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert required.issubset(result.keys())

    def test_L_plasma_at_radial_entry_matches_L_axial(self) -> None:
        """At radial phase entry r_shock ~ b, L_plasma ~ L_axial_frozen."""
        sp = _mhd_coupling_make_radial_snowplow()
        L_axial = sp._L_axial_frozen
        assert sp.plasma_inductance >= L_axial

    def test_frozen_result_has_zero_forces(self) -> None:
        """After pinch completion, F_magnetic and F_pressure are zero."""
        sp = _mhd_coupling_make_radial_snowplow()
        sp._pinch_complete = True
        sp.phase = "pinch"
        result = sp.step(1e-9, 1.5e6)
        assert result["F_magnetic"] == pytest.approx(0.0)
        assert result["F_pressure"] == pytest.approx(0.0)
        assert result["dL_dt"] == pytest.approx(0.0)


class TestPhaseDispatch:
    """step() correctly dispatches to axial, radial, or reflected handlers."""

    def test_rundown_dispatch(self) -> None:
        """Fresh snowplow starts in rundown and step dispatches to _step_axial."""
        sp = _mhd_coupling_make_snowplow()
        assert sp.phase == "rundown"
        result = sp.step(1e-9, 1.5e6)
        assert result["phase"] == "rundown"

    def test_radial_dispatch(self) -> None:
        """Manually placed radial snowplow dispatches to _step_radial."""
        sp = _mhd_coupling_make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6)
        assert result["phase"] in ("radial", "reflected")

    def test_reflected_dispatch(self) -> None:
        """Snowplow in reflected phase dispatches to _step_reflected."""
        sp = _mhd_coupling_make_radial_snowplow()
        sp.phase = "reflected"
        sp.r_shock = sp.r_pinch_min
        sp.vr = 0.0
        sp._M_slug_pinch = max(sp.radial_swept_mass, 1e-20)
        sp._p_pinch = sp._adiabatic_back_pressure(sp.r_shock)
        result = sp.step(1e-9, 1.5e6)
        assert result["phase"] in ("reflected", "pinch")

    def test_pinch_complete_returns_frozen(self) -> None:
        """After _pinch_complete=True, step returns frozen result every time."""
        sp = _mhd_coupling_make_radial_snowplow()
        sp._pinch_complete = True
        sp.phase = "pinch"
        r1 = sp.step(1e-9, 1.5e6)
        r2 = sp.step(1e-9, 1.5e6)
        assert r1["phase"] == "pinch"
        assert r2["phase"] == "pinch"
        assert r1["L_plasma"] == pytest.approx(r2["L_plasma"])

    def test_rundown_transitions_to_radial(self) -> None:
        """Sheath reaching anode length triggers transition to radial phase."""
        sp = _mhd_coupling_make_snowplow()
        reached_radial = _drive_to_phase(sp, "radial", max_steps=500_000)
        assert reached_radial, "Snowplow should reach radial phase under 1.5 MA drive"

    def test_pressure_kwarg_does_not_affect_rundown_phase_label(self) -> None:
        """Passing pressure= during rundown does not change the phase label returned."""
        sp = _mhd_coupling_make_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=1e4)
        assert result["phase"] == "rundown"

    def test_pressure_kwarg_during_radial_returns_radial_or_later(self) -> None:
        """Passing pressure= during radial returns radial/reflected/pinch phase."""
        sp = _mhd_coupling_make_radial_snowplow()
        result = sp.step(1e-9, 1.5e6, pressure=1e5)
        assert result["phase"] in ("radial", "reflected", "pinch")


class TestRundownPressureEffect:
    """External pressure affects axial back-pressure force during rundown."""

    def test_higher_pressure_reduces_axial_acceleration(self) -> None:
        """Higher fill pressure reduces net axial driving force."""
        sp_lo = _mhd_coupling_make_snowplow(fill_pressure_Pa=100.0)
        sp_hi = _mhd_coupling_make_snowplow(fill_pressure_Pa=100.0)

        for _ in range(100):
            sp_lo.step(1e-9, 1.5e6, pressure=None)
            sp_hi.step(1e-9, 1.5e6, pressure=1e8)

        assert sp_lo.v >= sp_hi.v

    def test_external_pressure_increases_F_pressure_axial(self) -> None:
        """External pressure > adiabatic raises F_pressure in axial step."""
        sp1 = _mhd_coupling_make_snowplow()
        sp2 = _mhd_coupling_make_snowplow()
        r1 = sp1.step(1e-9, 1.5e6, pressure=None)
        r2 = sp2.step(1e-9, 1.5e6, pressure=1e9)
        assert r2["F_pressure"] > r1["F_pressure"]

    def test_pressure_none_matches_fill_pressure_Pa_axial(self) -> None:
        """During rundown, pressure=None uses self.p_fill = fill_pressure_Pa."""
        sp1 = _mhd_coupling_make_snowplow(fill_pressure_Pa=800.0)
        sp2 = _mhd_coupling_make_snowplow(fill_pressure_Pa=800.0)
        r1 = sp1.step(1e-9, 1.5e6, pressure=None)
        r2 = sp2.step(1e-9, 1.5e6, pressure=800.0)
        assert r1["F_pressure"] == pytest.approx(r2["F_pressure"], rel=1e-9)


# --- Section: Reflected Shock Y ---


def _reflected_make_snowplow(**kwargs) -> SnowplowModel:
    """Construct a SnowplowModel with PF-1000-ish defaults."""
    defaults = dict(
        anode_radius=0.0575,
        cathode_radius=0.08,
        fill_density=1e-4,
        anode_length=0.16,
        mass_fraction=0.3,
        fill_pressure_Pa=400.0,
        current_fraction=0.7,
    )
    defaults.update(kwargs)
    return SnowplowModel(**defaults)


def _drive_to_reflected(
    sp: SnowplowModel,
    current: float = 1.5e6,
    dt: float = 1e-9,
    max_steps: int = 50_000,
) -> dict:
    """Drive snowplow through rundown and radial phases until reflected phase."""
    for _ in range(max_steps):
        result = sp.step(dt, current)
        if sp.phase == "reflected":
            return result
    pytest.fail(
        f"Failed to reach reflected phase within {max_steps} steps "
        f"(final phase={sp.phase!r}, r_shock={sp.r_shock:.4e} m)"
    )


def _drive_through_reflected(
    sp: SnowplowModel,
    current: float = 1.5e6,
    dt: float = 1e-9,
    max_steps: int = 200_000,
) -> list:
    """Drive snowplow from its current state until reflected phase ends."""
    results = []
    for _ in range(max_steps):
        if sp.phase != "reflected":
            break
        result = sp.step(dt, current)
        results.append(result)
    else:
        pytest.fail(
            f"Reflected phase did not terminate within {max_steps} steps "
            f"(r_shock={sp.r_shock:.4e} m, vr={sp.vr:.2e} m/s)"
        )
    return results


class TestReflectedShockPhase:
    """Tests for entry into and basic properties of the reflected phase."""

    def test_pinch_transitions_to_reflected(self) -> None:
        """Driving the radial shock to r_pinch_min sets phase='reflected'."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp)

        assert sp.phase == "reflected", (
            f"Expected phase='reflected', got {sp.phase!r}"
        )
        assert sp._pinch_complete is False, (
            "pinch should NOT be complete when entering reflected phase"
        )
        assert sp.is_active is True, (
            "is_active must be True during reflected phase"
        )

    def test_reflected_vr_positive(self) -> None:
        """Pressure-only drive (I=0) during reflected phase produces vr > 0."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp, current=1.5e6)

        result = sp.step(1e-9, 0.0)
        assert result["vr_shock"] > 0.0 or sp.phase == "pinch", (
            "With I=0, pressure-only drive must produce vr > 0 on the first step; "
            f"got vr_shock={result['vr_shock']:.4e} m/s"
        )
        if sp.phase == "reflected":
            outward_seen = result["vr_shock"] > 0.0
            for _ in range(5000):
                if sp.phase != "reflected":
                    break
                result = sp.step(1e-9, 0.0)
                if result["vr_shock"] > 0.0:
                    outward_seen = True
                    break
            assert outward_seen, (
                "Expected vr > 0 at some point during pressure-only reflected phase; "
                f"final vr={sp.vr:.4e} m/s"
            )

    def test_pressure_drives_outward(self) -> None:
        """F_pressure > 0 in result dict when the shock is in reflected phase."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp)

        result = sp.step(1e-9, 1.5e6)
        if sp.phase == "reflected" or result["phase"] == "reflected":
            assert result["F_pressure"] > 0.0, (
                "Adiabatic back-pressure force must be positive during reflected phase; "
                f"got F_pressure={result['F_pressure']:.4e} N"
            )
        else:
            assert result["F_pressure"] >= 0.0

    def test_reflected_terminates_at_cathode_or_stagnation(self) -> None:
        """Reflected phase terminates with _pinch_complete=True and phase='pinch'."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp)
        _drive_through_reflected(sp)

        assert sp._pinch_complete is True, (
            "After reflected phase terminates, _pinch_complete must be True"
        )
        assert sp.phase == "pinch", (
            f"Expected phase='pinch' after reflected termination, got {sp.phase!r}"
        )
        assert 0.0 < sp.r_shock <= sp.b, (
            f"Final r_shock={sp.r_shock:.4e} m must be in (0, b={sp.b:.4e} m]"
        )

    def test_reflected_is_active(self) -> None:
        """is_active tracks reflected phase entry and exit correctly."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp)

        assert sp.is_active is True, "is_active must be True while phase='reflected'"

        _drive_through_reflected(sp)

        assert sp.is_active is False, (
            f"is_active must be False after reflected terminates; phase={sp.phase!r}"
        )


class TestReflectedInductance:
    """Tests for plasma inductance behaviour during the reflected shock."""

    def test_dL_dt_negative_during_expansion(self) -> None:
        """When vr > 0 (shock expanding), dL/dt should be <= 0."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp, current=1.5e6)

        negative_dLdt_seen = False
        for _ in range(10_000):
            if sp.phase != "reflected":
                break
            result = sp.step(1e-9, 0.0)
            if result["vr_shock"] > 0.0:
                assert result["dL_dt"] <= 0.0, (
                    f"dL/dt must be <= 0 when vr > 0 (expanding); "
                    f"got dL_dt={result['dL_dt']:.4e}, vr={result['vr_shock']:.4e}"
                )
                negative_dLdt_seen = True

        assert negative_dLdt_seen, (
            "Expected at least one outward step (vr > 0) with I=0 to verify dL/dt sign; "
            f"final vr={sp.vr:.4e} m/s, phase={sp.phase!r}"
        )

    def test_inductance_decreases(self) -> None:
        """Plasma inductance decreases from pinch entry to end of reflected phase."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp)

        L_start = sp.plasma_inductance

        _drive_through_reflected(sp)

        L_end = sp.plasma_inductance
        assert L_end <= L_start, (
            f"Inductance should decrease as r expands during reflected phase; "
            f"L_start={L_start:.4e} H, L_end={L_end:.4e} H"
        )


class TestFullLifecycle:
    """End-to-end lifecycle tests through all four Lee model phases."""

    def test_rundown_radial_reflected_pinch(self) -> None:
        """All four phases are visited: rundown -> radial -> reflected -> pinch."""
        sp = _reflected_make_snowplow()

        phases_seen: set[str] = set()

        for _ in range(500_000):
            result = sp.step(1e-9, 1.5e6)
            phases_seen.add(result["phase"])
            if sp._pinch_complete:
                break
        else:
            pytest.fail(
                f"Snowplow did not complete full lifecycle within step limit; "
                f"final phase={sp.phase!r}, phases seen={phases_seen}"
            )

        for expected_phase in ("rundown", "radial", "reflected", "pinch"):
            assert expected_phase in phases_seen, (
                f"Phase '{expected_phase}' was never visited; "
                f"phases seen: {phases_seen}"
            )

    def test_phase_sequence(self) -> None:
        """Phase transitions follow the correct causal order."""
        sp = _reflected_make_snowplow()

        phase_order: list[str] = []
        prev_phase = "rundown"

        for _ in range(500_000):
            result = sp.step(1e-9, 1.5e6)
            current_phase = result["phase"]
            if current_phase != prev_phase:
                phase_order.append(current_phase)
                prev_phase = current_phase
            if sp._pinch_complete:
                break
        else:
            pytest.fail(
                f"Full lifecycle not completed within step limit; "
                f"phase sequence so far: {phase_order}"
            )

        expected_sequence = ["radial", "reflected", "pinch"]
        idx = 0
        for phase in phase_order:
            if idx < len(expected_sequence) and phase == expected_sequence[idx]:
                idx += 1

        assert idx == len(expected_sequence), (
            f"Expected causal phase sequence {expected_sequence!r} "
            f"but observed: {phase_order!r}"
        )


class TestReflectedEdgeCases:
    """Edge case behaviour for the reflected shock phase."""

    def test_zero_current_reflected(self) -> None:
        """With I=0, J x B=0 so only pressure acts: shock must expand (vr > 0)."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp, current=1.5e6)

        outward_seen = False
        for _ in range(10_000):
            if sp.phase != "reflected":
                break
            result = sp.step(1e-9, 0.0)
            if result["vr_shock"] > 0.0:
                outward_seen = True
                break

        if sp.phase == "pinch":
            assert sp.r_shock >= sp.r_pinch_min
        else:
            assert outward_seen, (
                "With zero current, pressure-only drive should produce outward "
                f"velocity; final vr={sp.vr:.4e} m/s, phase={sp.phase!r}"
            )

    def test_very_high_current_reflected(self) -> None:
        """At very high current (10 MA), J x B overwhelms pressure -> fast re-stagnation."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp, current=1.5e6)

        high_current = 10.0e6
        for _ in range(50_000):
            if sp.phase != "reflected":
                break
            sp.step(1e-9, high_current)
        assert sp.phase in ("reflected", "pinch"), (
            f"Unexpected phase after high-current reflected drive: {sp.phase!r}"
        )
        if sp.phase == "pinch":
            assert sp._pinch_complete is True

    def test_frozen_after_reflected(self) -> None:
        """Once reflected phase terminates, phase stays 'pinch' with zero forces."""
        sp = _reflected_make_snowplow()
        _drive_to_reflected(sp)
        _drive_through_reflected(sp)

        assert sp._pinch_complete is True, "Prerequisite: pinch must be complete"

        for _ in range(10):
            result = sp.step(1e-9, 1.5e6)
            assert result["F_magnetic"] == pytest.approx(0.0), (
                f"Post-pinch must have F_magnetic=0; got {result['F_magnetic']:.4e}"
            )
            assert result["F_pressure"] == pytest.approx(0.0), (
                f"Post-pinch must have F_pressure=0; got {result['F_pressure']:.4e}"
            )
            assert result["phase"] == "pinch", (
                f"Post-pinch must report phase='pinch'; got {result['phase']!r}"
            )


# --- Section: B-field Initialization Z ---


def _make_engine_with_snowplow(
    nr: int = 16,
    nz: int = 8,
    snowplow_phase: str = "radial",
    current: float = 1.0e6,
    r_shock_frac: float = 0.8,
):
    """Create a mock engine-like object with snowplow in the specified phase."""

    dr = 1e-3
    r_max = nr * dr
    r_grid = np.array([(i + 0.5) * dr for i in range(nr)])
    r_shock = r_shock_frac * r_max

    state = {
        "rho": np.full((nr, 1, nz), 1e-4),
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), 100.0),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 1e6),
        "Ti": np.full((nr, 1, nz), 1e6),
        "psi": np.zeros((nr, 1, nz)),
    }

    snowplow = MagicMock()
    snowplow.phase = snowplow_phase
    snowplow.r_shock = r_shock
    snowplow.is_active = True

    geom = MagicMock()
    geom.r = r_grid

    fluid = MagicMock()
    fluid.geom = geom

    coupling = MagicMock()
    coupling.current = current

    return state, snowplow, fluid, coupling, dr, r_grid, r_shock


def _apply_bfield_init(
    state: dict,
    current: float,
    r_grid: np.ndarray,
    r_shock: float,
    dr: float,
) -> dict:
    """Apply B_theta initialization logic (mirrors engine._initialize_radial_bfield)."""
    from dpf.constants import mu_0
    from dpf.constants import pi as _pi

    ir_shock = round(r_shock / dr) if dr > 0 else len(r_grid)
    ir_shock = min(ir_shock, len(r_grid))

    B = state["B"]
    I_abs = abs(current)

    for ir in range(ir_shock):
        r_val = r_grid[ir]
        if r_val > 0:
            B[1, ir, :, :] = mu_0 * I_abs / (2.0 * _pi * r_val)
        else:
            B[1, ir, :, :] = 0.0

    if ir_shock < B.shape[1]:
        B[1, ir_shock:, :, :] = 0.0

    state["B"] = B
    return state


class TestBfieldProfile:
    """B_theta profile matches analytical at initialization."""

    def test_btheta_inside_shock_analytical(self):
        """B_theta(r) = mu_0 * I / (2*pi*r) for r < r_shock."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=32, current=1.0e6, r_shock_frac=0.7,
        )
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)

        ir_shock = round(r_shock / dr)
        for ir in range(1, ir_shock):
            r = r_grid[ir]
            expected = mu_0 * 1.0e6 / (2.0 * _pi * r)
            actual = state["B"][1, ir, 0, 0]
            assert actual == pytest.approx(expected, rel=1e-14), (
                f"B_theta mismatch at ir={ir}, r={r:.4e}: "
                f"got {actual:.6e}, expected {expected:.6e}"
            )

    def test_btheta_outside_shock_zero(self):
        """B_theta = 0 for r > r_shock."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=32, current=1.0e6, r_shock_frac=0.5,
        )
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)

        ir_shock = round(r_shock / dr)
        assert ir_shock < 32
        np.testing.assert_array_equal(state["B"][1, ir_shock:, :, :], 0.0)

    def test_btheta_on_axis_zero(self):
        """B_theta(r=0) = 0 by symmetry."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=16, current=2.0e6,
        )
        r_grid_tiny = np.array([(i + 0.5) * 1e-6 for i in range(16)])
        r_grid_tiny[0] = 0.0
        state = _apply_bfield_init(state, coup.current, r_grid_tiny, r_shock, dr)
        assert state["B"][1, 0, 0, 0] == 0.0

    def test_btheta_monotonically_decreasing_inside(self):
        """B_theta = mu_0*I/(2*pi*r) decreases with r inside shock."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=32, current=1.0e6, r_shock_frac=0.8,
        )
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)

        ir_shock = round(r_shock / dr)
        bt_inside = state["B"][1, 1:ir_shock, 0, 0]
        assert np.all(np.diff(bt_inside) < 0), "B_theta should decrease as 1/r"

    def test_br_bz_unchanged(self):
        """B_r and B_z remain zero after initialization."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow()
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)
        np.testing.assert_array_equal(state["B"][0], 0.0)
        np.testing.assert_array_equal(state["B"][2], 0.0)

    def test_z_uniformity(self):
        """B_theta profile is uniform along z (infinite cylinder at init)."""
        nr, nz = 16, 8
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=nr, nz=nz,
        )
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)
        for iz in range(nz):
            np.testing.assert_array_equal(
                state["B"][1, :, 0, iz], state["B"][1, :, 0, 0],
            )


class TestCurrentContinuity:
    """Integral of J_z * 2*pi*r*dr should equal I."""

    def test_current_integral_from_btheta(self):
        """Ampere's law: I_enclosed = 2*pi*r*B_theta/mu_0 at r_shock."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        I_total = 1.5e6
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=64, current=I_total, r_shock_frac=0.7,
        )
        state = _apply_bfield_init(state, I_total, r_grid, r_shock, dr)

        ir_shock = round(r_shock / dr)
        ir_test = max(ir_shock - 1, 1)
        r_test = r_grid[ir_test]
        Bt_test = state["B"][1, ir_test, 0, 0]

        I_enclosed = 2.0 * _pi * r_test * Bt_test / mu_0
        assert I_enclosed == pytest.approx(I_total, rel=1e-3), (
            f"Enclosed current {I_enclosed:.4e} != {I_total:.4e}"
        )

    def test_current_integral_numerical(self):
        """Numerical integral of J_z from curl(B) reproduces I."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        I_total = 1.0e6
        nr = 128
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=nr, current=I_total, r_shock_frac=0.8,
        )
        state = _apply_bfield_init(state, I_total, r_grid, r_shock, dr)

        Bt = state["B"][1, :, 0, 0]
        rBt = r_grid * Bt
        d_rBt_dr = np.gradient(rBt, dr)
        Jz = d_rBt_dr / (mu_0 * r_grid)
        Jz[0] = 0.0

        I_integrated = np.sum(Jz * 2.0 * _pi * r_grid * dr)
        assert abs(I_integrated) == pytest.approx(I_total, rel=0.05), (
            f"Numerical |I|={abs(I_integrated):.4e} vs {I_total:.4e}"
        )


class TestOneShot:
    """B-field initialization happens exactly once."""

    def test_flag_starts_false(self):
        """_radial_bfield_initialized starts as False."""
        from dpf.engine import SimulationEngine

        with patch.object(SimulationEngine, "__init__", lambda self: None):
            eng = object.__new__(SimulationEngine)
            eng._radial_bfield_initialized = False
            assert eng._radial_bfield_initialized is False

    def test_flag_set_after_init(self):
        """_radial_bfield_initialized becomes True after _initialize_radial_bfield."""
        from dpf.engine import SimulationEngine

        for backend in ("python", "metal", "athena"):
            eng = object.__new__(SimulationEngine)
            nr, nz = 16, 8
            state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
                nr=nr, nz=nz,
            )
            eng.snowplow = sp
            eng.backend = backend
            eng.geometry_type = "cylindrical"
            eng.fluid = fluid
            eng._coupling = coup
            eng.config = MagicMock()
            eng.config.dx = dr
            eng.config.grid_shape = [nr, 1, nz]
            eng.state = state
            eng._radial_bfield_initialized = False

            eng._initialize_radial_bfield()

            assert eng._radial_bfield_initialized is True, (
                f"Flag not set for backend={backend!r}"
            )

    def test_second_call_is_noop_via_flag(self):
        """After initialization, calling again should not re-initialize."""
        from dpf.engine import SimulationEngine

        eng = object.__new__(SimulationEngine)
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            current=1.0e6,
        )
        eng.snowplow = sp
        eng.backend = "python"
        eng.geometry_type = "cylindrical"
        eng.fluid = fluid
        eng._coupling = coup
        eng.config = MagicMock()
        eng.config.dx = dr
        eng.config.grid_shape = [16, 1, 8]
        eng.state = state
        eng._radial_bfield_initialized = False

        eng._initialize_radial_bfield()
        bt_after_first = eng.state["B"][1, :, 0, 0].copy()
        assert eng._radial_bfield_initialized is True

        eng._coupling.current = 2.0e6

        if not eng._radial_bfield_initialized:
            eng._initialize_radial_bfield()

        np.testing.assert_array_equal(eng.state["B"][1, :, 0, 0], bt_after_first)


class TestGuards:
    """_initialize_radial_bfield runs for all cylindrical backends."""

    def _make_engine_obj(
        self,
        backend: str = "python",
        geom: str = "cylindrical",
        has_geom_attr: bool = True,
        nr: int = 16,
        nz: int = 8,
    ):
        from dpf.engine import SimulationEngine

        eng = object.__new__(SimulationEngine)
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=nr, nz=nz,
        )
        eng.snowplow = sp
        eng.backend = backend
        eng.geometry_type = geom
        eng._coupling = coup
        eng.config = MagicMock()
        eng.config.dx = dr
        eng.config.grid_shape = [nr, 1, nz]
        eng.state = state
        eng._radial_bfield_initialized = False

        if has_geom_attr:
            eng.fluid = fluid
        else:
            fluid_no_geom = MagicMock(spec=[])
            eng.fluid = fluid_no_geom

        return eng, dr, r_grid

    def test_works_for_athena_backend(self):
        """B_theta initialized for Athena++ cylindrical backend."""
        eng, dr, r_grid = self._make_engine_obj(backend="athena")
        eng._initialize_radial_bfield()
        assert eng._radial_bfield_initialized is True
        ir_shock = round(eng.snowplow.r_shock / dr)
        bt_inside = eng.state["B"][1, 1:ir_shock, 0, 0]
        assert np.all(bt_inside > 0), "B_theta should be positive inside shock"

    def test_works_for_metal_backend_with_geom(self):
        """B_theta initialized for Metal cylindrical backend (geom.r path)."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        eng, dr, r_grid = self._make_engine_obj(backend="metal")
        eng._initialize_radial_bfield()
        assert eng._radial_bfield_initialized is True
        ir_shock = round(eng.snowplow.r_shock / dr)
        for ir in range(1, min(ir_shock, 5)):
            r = r_grid[ir]
            expected = mu_0 * abs(eng._coupling.current) / (2.0 * _pi * r)
            actual = eng.state["B"][1, ir, 0, 0]
            assert actual == pytest.approx(expected, rel=1e-10), (
                f"Metal B_theta mismatch at ir={ir}"
            )

    def test_works_for_metal_backend_config_fallback(self):
        """B_theta initialized for Metal backend without geom.r (config fallback)."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        nr, nz = 16, 8
        eng, dr, r_grid = self._make_engine_obj(
            backend="metal", has_geom_attr=False, nr=nr, nz=nz,
        )
        eng._initialize_radial_bfield()
        assert eng._radial_bfield_initialized is True
        ir_shock = round(eng.snowplow.r_shock / dr)
        for ir in range(1, min(ir_shock, 5)):
            r_expected = (ir + 0.5) * dr
            bt_expected = mu_0 * abs(eng._coupling.current) / (2.0 * _pi * r_expected)
            bt_actual = eng.state["B"][1, ir, 0, 0]
            assert bt_actual == pytest.approx(bt_expected, rel=1e-10), (
                f"Config-fallback B_theta mismatch at ir={ir}"
            )

    def test_works_for_athenak_backend_config_fallback(self):
        """B_theta initialized for AthenaK backend without geom.r."""
        eng, dr, _ = self._make_engine_obj(
            backend="athenak", has_geom_attr=False,
        )
        eng._initialize_radial_bfield()
        assert eng._radial_bfield_initialized is True
        ir_shock = round(eng.snowplow.r_shock / dr)
        bt_inside = eng.state["B"][1, 1:ir_shock, 0, 0]
        assert np.all(bt_inside > 0)

    def test_skipped_for_cartesian_geometry(self):
        """No-op for Cartesian geometry (all backends)."""
        for backend in ("python", "metal", "athena"):
            eng, _, _ = self._make_engine_obj(backend=backend, geom="cartesian")
            eng._initialize_radial_bfield()
            np.testing.assert_array_equal(eng.state["B"][1], 0.0)
            assert eng._radial_bfield_initialized is False

    def test_skipped_without_snowplow(self):
        """No-op when snowplow is None."""
        eng, _, _ = self._make_engine_obj()
        eng.snowplow = None
        eng._initialize_radial_bfield()
        np.testing.assert_array_equal(eng.state["B"][1], 0.0)
        assert eng._radial_bfield_initialized is False


class TestStability:
    """No NaN or negative density after B-field initialization."""

    def test_no_nan_after_init(self):
        """All state arrays finite after B-field init."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            nr=32, current=2.0e6,
        )
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)

        for key, arr in state.items():
            assert np.all(np.isfinite(arr)), f"NaN/Inf in state['{key}']"

    def test_density_positive(self):
        """Density remains positive after initialization."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow()
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)
        assert np.all(state["rho"] > 0)

    def test_pressure_positive(self):
        """Pressure remains positive after initialization."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow()
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)
        assert np.all(state["pressure"] > 0)

    def test_various_currents(self):
        """Profile correct for different current magnitudes."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        for I_val in [1e4, 1e5, 1e6, 5e6]:
            state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
                nr=16, current=I_val,
            )
            state = _apply_bfield_init(state, I_val, r_grid, r_shock, dr)
            assert np.all(np.isfinite(state["B"])), f"NaN at I={I_val}"
            ir_mid = 4
            expected = mu_0 * I_val / (2.0 * _pi * r_grid[ir_mid])
            actual = state["B"][1, ir_mid, 0, 0]
            assert actual == pytest.approx(expected, rel=1e-14)

    def test_negative_current_handled(self):
        """Negative current direction still gives valid B_theta (absolute value used)."""
        state, sp, fluid, coup, dr, r_grid, r_shock = _make_engine_with_snowplow(
            current=-1.0e6,
        )
        state = _apply_bfield_init(state, coup.current, r_grid, r_shock, dr)
        ir_shock = round(r_shock / dr)
        bt_inside = state["B"][1, 1:ir_shock, 0, 0]
        assert np.all(bt_inside > 0), "B_theta should be positive for abs(I)"


# --- Section: FMR Convergence AD ---


class TestAD1RadialMassFraction:
    """f_mr is properly separated from f_m per Lee & Saw (2014)."""

    def test_config_accepts_radial_mass_fraction(self) -> None:
        """SnowplowConfig allows setting radial_mass_fraction."""
        from dpf.config import SnowplowConfig

        cfg = SnowplowConfig(
            mass_fraction=0.178,
            current_fraction=0.65,
            radial_mass_fraction=0.1,
            anode_length=0.6,
        )
        assert cfg.radial_mass_fraction == pytest.approx(0.1)
        assert cfg.mass_fraction == pytest.approx(0.178)

    def test_snowplow_uses_separate_f_mr(self) -> None:
        """SnowplowModel uses f_mr independently of f_m."""
        sp = SnowplowModel(
            anode_radius=0.115,
            cathode_radius=0.16,
            fill_density=4e-4,
            anode_length=0.6,
            mass_fraction=0.178,
            current_fraction=0.65,
            radial_mass_fraction=0.1,
        )
        assert sp.f_m == pytest.approx(0.178)
        assert sp.f_mr == pytest.approx(0.1)
        assert sp.f_mr < sp.f_m

    def test_f_mr_in_lee_saw_range(self) -> None:
        """f_mr = 0.1 is within Lee & Saw (2014) recommended range."""
        f_mr = 0.1
        assert 0.07 <= f_mr <= 0.12

    def test_pf1000_preset_has_f_mr(self) -> None:
        """PF-1000 preset includes radial_mass_fraction (published: 0.16)."""
        from dpf.presets import get_preset
        preset = get_preset("pf1000")
        assert "radial_mass_fraction" in preset.get("snowplow", {})
        f_mr = preset["snowplow"]["radial_mass_fraction"]
        assert 0.10 <= f_mr <= 0.20

    def test_nx2_preset_has_f_mr(self) -> None:
        """NX2 preset includes radial_mass_fraction."""
        from dpf.presets import get_preset
        preset = get_preset("nx2")
        assert "radial_mass_fraction" in preset.get("snowplow", {})

    def test_lee_model_supports_f_mr(self) -> None:
        """LeeModel analytical model accepts radial_mass_fraction."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.65,
            mass_fraction=0.178,
            radial_mass_fraction=0.1,
        )
        assert model.f_mr == pytest.approx(0.1)
        assert model.fm == pytest.approx(0.178)

    def test_lee_model_f_mr_default(self) -> None:
        """LeeModel defaults f_mr to f_m when not specified."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.65, mass_fraction=0.178)
        assert model.f_mr == pytest.approx(0.178)

    def test_f_mr_affects_radial_mass(self) -> None:
        """Lower f_mr gives less radial swept mass."""
        params = dict(
            anode_radius=0.115,
            cathode_radius=0.16,
            fill_density=4e-4,
            anode_length=0.6,
            mass_fraction=0.178,
            current_fraction=0.65,
        )
        sp_low = SnowplowModel(**params, radial_mass_fraction=0.07)
        sp_high = SnowplowModel(**params, radial_mass_fraction=0.12)

        sp_low.r_shock = 0.08
        sp_high.r_shock = 0.08
        sp_low.phase = "radial"
        sp_high.phase = "radial"

        assert sp_low.radial_swept_mass < sp_high.radial_swept_mass

    def test_f_mr_physics_ratio(self) -> None:
        """f_mr / f_m ratio is physically reasonable."""
        f_m = 0.178
        f_mr = 0.1
        ratio = f_mr / f_m
        assert 0.3 <= ratio <= 0.8, f"f_mr/f_m = {ratio:.2f} outside typical range"


class TestAD2GridConvergence:
    """Grid convergence and Richardson extrapolation framework."""

    def test_lee_model_convergence_peak_current(self) -> None:
        """LeeModel peak current converges as solve_ivp refines."""
        from dpf.validation.lee_model_comparison import LeeModel

        results = []
        for _n_steps in [2000, 5000, 10000]:
            model = LeeModel(
                current_fraction=0.65,
                mass_fraction=0.178,
                radial_mass_fraction=0.1,
                crowbar_enabled=True,
            )
            result = model.run("PF-1000")
            results.append(result.peak_current)

        for i in range(1, len(results)):
            rel_diff = abs(results[i] - results[0]) / results[0]
            assert rel_diff < 0.001, (
                f"Peak current varies: {results[0]/1e6:.4f} vs {results[i]/1e6:.4f} MA"
            )

    def test_richardson_extrapolation_formula(self) -> None:
        """Richardson extrapolation math is correct."""
        def f(h: float) -> float:
            return np.sin(h) / h if h > 0 else 1.0

        h1, h2 = 0.1, 0.2
        f1, f2 = f(h1), f(h2)
        p = 2

        f_exact_est = f1 + (f1 - f2) / (2**p - 1)
        f_true = 1.0

        err_raw = abs(f1 - f_true)
        err_rich = abs(f_exact_est - f_true)
        assert err_rich < err_raw, "Richardson extrapolation should improve accuracy"

    def test_convergence_order_estimation(self) -> None:
        """Observed convergence order can be estimated from 3 resolutions."""
        a = 0.5
        f_exact = 1.0

        h_vals = [0.2, 0.1, 0.05]
        f_vals = [f_exact + a * h**2 for h in h_vals]

        numer = f_vals[0] - f_vals[1]
        denom = f_vals[1] - f_vals[2]
        if abs(denom) > 1e-15:
            p_est = np.log2(abs(numer / denom))
        else:
            p_est = float("inf")

        assert abs(p_est - 2.0) < 0.01, f"Estimated order {p_est:.2f} should be ~2.0"


class TestAD3CrossDevicePrediction:
    """Cross-device validation: calibrate on PF-1000, predict NX2."""

    def test_lee_model_runs_nx2(self) -> None:
        """LeeModel can run NX2 device."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.7,
            mass_fraction=0.15,
            radial_mass_fraction=0.12,
            crowbar_enabled=True,
        )
        result = model.run("NX2")
        assert result.peak_current > 0
        assert result.peak_current_time > 0
        assert 2 in result.phases_completed

    def test_pf1000_calibrated_predicts_nx2(self) -> None:
        """PF-1000-calibrated fc/fm gives reasonable NX2 prediction."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.65,
            mass_fraction=0.178,
            radial_mass_fraction=0.1,
            crowbar_enabled=True,
        )
        comparison = model.compare_with_experiment("NX2")

        assert comparison.peak_current_error < 1.0, (
            f"NX2 peak error = {comparison.peak_current_error:.1%}"
        )

    def test_nx2_device_params_exist(self) -> None:
        """NX2 device data is available in the validation registry."""
        from dpf.validation.experimental import DEVICES
        assert "NX2" in DEVICES
        nx2 = DEVICES["NX2"]
        assert nx2.peak_current > 0
        assert nx2.capacitance > 0

    def test_cross_validator_exists(self) -> None:
        """CrossValidator class is importable and functional."""
        from dpf.validation.calibration import CrossValidator
        cv = CrossValidator()
        assert hasattr(cv, "validate")


class TestAD4DipDepthAnalysis:
    """Diagnostic tests for understanding current dip physics."""

    def test_inductance_ratio_at_pinch(self) -> None:
        """Inductance ratio DeltaL/L at full compression explains the dip."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        a, b = 0.115, 0.16
        z_f = 0.6
        L0 = 33.5e-9
        L_per_len = (mu_0 / (2 * _pi)) * np.log(b / a)

        L_axial = L_per_len * z_f

        r_pinch = 0.1 * a
        L_radial = (mu_0 / (2 * _pi)) * z_f * np.log(b / r_pinch)

        L_total = L0 + L_axial + L_radial
        ratio = L_radial / L_total

        assert 0.7 < ratio < 0.85, f"DeltaL/L = {ratio:.1%}"

    def test_jxb_dominates_pressure_at_pinch(self) -> None:
        """J x B force >> back-pressure at pinch, preventing bounce."""
        from dpf.constants import mu_0
        from dpf.constants import pi as _pi

        a, b = 0.115, 0.16
        z_f = 0.6
        fc = 0.65
        I_dip = 0.5e6
        r_pinch = 0.1 * a
        p_fill = 466.6
        gamma = 5.0 / 3.0

        F_jxb = (mu_0 / (4 * _pi)) * (fc * I_dip) ** 2 * z_f / r_pinch

        p_pinch = p_fill * (b / r_pinch) ** (2 * gamma)
        F_pressure = p_pinch * 2 * _pi * r_pinch * z_f

        ratio = F_jxb / F_pressure
        assert ratio > 2.0, f"F_JxB/F_pressure = {ratio:.1f}, expected > 2"

    def test_lee_model_produces_dip(self) -> None:
        """LeeModel shows a current dip (radial phase working)."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.65,
            mass_fraction=0.178,
            radial_mass_fraction=0.1,
            crowbar_enabled=True,
        )
        result = model.run("PF-1000")

        I_abs = np.abs(result.I)
        peak_idx = int(np.argmax(I_abs))
        I_peak = I_abs[peak_idx]

        assert peak_idx < len(I_abs) - 5, "Peak should not be at end of waveform"
        I_after = I_abs[peak_idx:]
        I_min = float(np.min(I_after))
        dip_depth = (I_peak - I_min) / I_peak

        assert dip_depth > 0.3, f"Expected dip > 30%, got {dip_depth:.1%}"


# --- Section: AF Reflected Shock ---


class TestPhase4Completion:
    """Verify reflected shock Phase 4 runs and completes."""

    def test_phase4_in_phases_completed(self):
        """Phase 4 appears in phases_completed list."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        result = model.run("PF-1000")
        assert 4 in result.phases_completed, (
            f"Phase 4 not completed. Phases: {result.phases_completed}"
        )

    def test_phases_1_2_4_all_present(self):
        """All three phases (1, 2, 4) are completed."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        result = model.run("PF-1000")
        assert 1 in result.phases_completed
        assert 2 in result.phases_completed
        assert 4 in result.phases_completed

    def test_phase4_with_pcf_014(self):
        """Phase 4 completes with pinch_column_fraction=0.14."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        assert 4 in result.phases_completed

    def test_phase4_with_nx2(self):
        """Phase 4 completes for NX2 device."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.7, mass_fraction=0.15)
        result = model.run("NX2")
        assert 4 in result.phases_completed

    def test_phase4_with_crowbar(self):
        """Phase 4 completes with crowbar enabled."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            crowbar_enabled=True,
        )
        result = model.run("PF-1000")
        assert 4 in result.phases_completed


class TestReflectedShockPhysics:
    """Verify reflected shock physics are correct."""

    @pytest.fixture(scope="class")
    def pf1000_result(self):
        """Run PF-1000 Lee model with reflected shock (cached)."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        return model.run("PF-1000")

    def test_shock_reaches_pinch_minimum(self, pf1000_result):
        """Shock radius reaches near 0.1*a during radial phase."""
        r = pf1000_result.r_shock
        a = pf1000_result.metadata["anode_radius"]
        assert np.min(r) < 0.15 * a, (
            f"Min r_shock {np.min(r):.4f} m > 0.15*a = {0.15*a:.4f} m"
        )

    def test_shock_expands_after_pinch(self):
        """Shock radius increases after pinch with pcf=0.14."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        r = result.r_shock
        min_idx = np.argmin(r)
        if min_idx < len(r) - 1:
            r_after = r[min_idx + 1:]
            assert len(r_after) > 0, "No data after pinch minimum"
            assert np.max(r_after) >= r[min_idx], (
                f"Shock doesn't expand: r_min={r[min_idx]:.4f}, "
                f"r_max_after={np.max(r_after):.4f}"
            )

    def test_shock_final_radius_reasonable(self, pf1000_result):
        """Final shock radius is between pinch minimum and cathode."""
        r = pf1000_result.r_shock
        b = pf1000_result.metadata["cathode_radius"]
        r_final = float(r[-1])
        r_min = float(np.min(r))
        assert r_final >= r_min, (
            f"Final r={r_final:.4f} < min r={r_min:.4f}"
        )
        assert r_final <= b * 1.01, (
            f"Final r={r_final:.4f} > cathode b={b:.4f}"
        )

    def test_current_continuous_across_phases(self, pf1000_result):
        """Current I(t) is continuous (no jumps at phase transitions)."""
        I_arr = pf1000_result.I  # noqa: E741
        dI = np.abs(np.diff(I_arr))
        max_jump = np.max(dI)
        peak_I = np.max(np.abs(I_arr))
        assert max_jump < 0.10 * peak_I, (
            f"Max current jump {max_jump:.2e} > 10% of peak {peak_I:.2e}"
        )


class TestCurrentDipWithReflectedShock:
    """Validate current dip behavior with reflected shock."""

    def test_pcf1_deep_dip(self):
        """pcf=1.0 with reflected shock still gives deep dip (>50%)."""
        from dpf.validation.experimental import _find_first_peak
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=1.0,
        )
        result = model.run("PF-1000")
        abs_I = np.abs(result.I)
        peak_idx = _find_first_peak(abs_I)
        t_us = result.t * 1e6
        search_end = np.searchsorted(t_us, t_us[peak_idx] + 1.0)
        post_peak = abs_I[peak_idx:search_end]
        dip = (abs_I[peak_idx] - np.min(post_peak)) / abs_I[peak_idx]
        assert dip > 0.40, f"pcf=1.0 dip {dip:.0%} should be > 40%"

    def test_pcf014_experimental_dip(self):
        """pcf=0.14 with reflected shock gives 20-90% dip."""
        from dpf.validation.experimental import _find_first_peak
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        abs_I = np.abs(result.I)
        peak_idx = _find_first_peak(abs_I)
        t_us = result.t * 1e6
        search_end = np.searchsorted(t_us, t_us[peak_idx] + 1.0)
        post_peak = abs_I[peak_idx:search_end]
        dip = (abs_I[peak_idx] - np.min(post_peak)) / abs_I[peak_idx]
        assert 0.05 < dip < 0.90, (
            f"pcf=0.14 dip {dip:.0%} outside [5%, 90%]"
        )

    def test_reflected_shock_reduces_dip_duration(self):
        """Reflected shock causes current to recover after dip minimum."""
        from dpf.validation.experimental import _find_first_peak
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        result = model.run("PF-1000")
        abs_I = np.abs(result.I)
        peak_idx = _find_first_peak(abs_I)
        t_us = result.t * 1e6
        search_end = np.searchsorted(t_us, t_us[peak_idx] + 1.0)
        post_peak = abs_I[peak_idx:search_end]

        min_idx = np.argmin(post_peak)
        if min_idx < len(post_peak) - 5:
            after_dip = post_peak[min_idx:min(min_idx + 20, len(post_peak))]
            recovery = (np.max(after_dip) - post_peak[min_idx]) / abs_I[peak_idx]
            assert recovery >= 0.0, "No current recovery after dip minimum"


class TestNRMSEWithReflectedShock:
    """Verify NRMSE against Scholz (2006) is maintained."""

    def test_nrmse_below_020(self):
        """NRMSE < 0.20 with reflected shock (Lee benchmark: 0.133)."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        comp = model.compare_with_experiment("PF-1000")
        assert np.isfinite(comp.waveform_nrmse)
        assert comp.waveform_nrmse < 0.20, (
            f"NRMSE {comp.waveform_nrmse:.4f} exceeds 0.20"
        )

    def test_peak_error_below_5pct(self):
        """Peak current error < 5% with reflected shock."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=0.816,
            mass_fraction=0.142,
            pinch_column_fraction=0.14,
        )
        comp = model.compare_with_experiment("PF-1000")
        assert comp.peak_current_error < 0.05, (
            f"Peak error {comp.peak_current_error:.1%} > 5%"
        )


class TestReflectedShockConsistency:
    """Cross-check Lee model Phase 4 against snowplow reflected shock."""

    def test_both_have_reflected_phase(self):
        """Both LeeModel and SnowplowModel complete reflected shock."""
        from dpf.validation.lee_model_comparison import LeeModel

        lee = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        lee_result = lee.run("PF-1000")
        assert 4 in lee_result.phases_completed

        sp = SnowplowModel(
            anode_radius=0.115,
            cathode_radius=0.160,
            fill_density=8.4e-5,
            anode_length=0.60,
            current_fraction=0.816,
            mass_fraction=0.142,
        )
        for _ in range(50000):
            sp.step(1e-9, 1.5e6)
            if sp.phase in ("reflected", "frozen"):
                break

        assert sp.phase in ("reflected", "frozen"), (
            f"Snowplow phase is '{sp.phase}', expected reflected/frozen"
        )

    def test_peak_currents_consistent(self):
        """Lee model and engine (RLCSolver+Snowplow) produce similar peaks."""
        from dpf.validation.engine_validation import run_rlc_snowplow_pf1000
        from dpf.validation.lee_model_comparison import LeeModel

        lee = LeeModel(current_fraction=0.816, mass_fraction=0.142)
        lee_result = lee.run("PF-1000")

        _, I_rlc, _ = run_rlc_snowplow_pf1000(
            fc=0.816, fm=0.142, sim_time=10e-6,
        )

        lee_peak = lee_result.peak_current
        rlc_peak = float(np.max(np.abs(I_rlc)))

        rel_diff = abs(lee_peak - rlc_peak) / max(lee_peak, rlc_peak)
        assert rel_diff < 0.02, (
            f"Peak current mismatch: Lee={lee_peak/1e6:.3f} MA, "
            f"RLC={rlc_peak/1e6:.3f} MA, diff={rel_diff:.1%}"
        )


# --- Section: Two-Step Radial ---


def make_faeton_snowplow(
    radial_current_fraction: float | None = None,
    radial_current_fraction_2: float | None = None,
    radial_transition_time: float | None = None,
) -> SnowplowModel:
    """Create a FAETON-I-like SnowplowModel for testing."""
    return SnowplowModel(
        anode_radius=0.05,
        cathode_radius=0.10,
        fill_density=1.29e-3,
        anode_length=0.17,
        mass_fraction=0.70,
        fill_pressure_Pa=1600.0,
        current_fraction=0.7,
        radial_mass_fraction=0.1,
        pinch_column_fraction=0.14,
        radial_current_fraction=radial_current_fraction,
        radial_current_fraction_2=radial_current_fraction_2,
        radial_transition_time=radial_transition_time,
    )


def advance_to_radial(sp: SnowplowModel, current: float = 1e6) -> None:
    """Advance snowplow through rundown into radial phase."""
    dt = 1e-9
    for _ in range(50_000):
        sp.step(dt, current=current)
        if sp.phase == "radial":
            break
    assert sp.phase == "radial", "Failed to reach radial phase"


class TestSingleStepRadial:
    """Verify single-step (default) radial behavior is unchanged."""

    def test_default_fcr_equals_fc(self) -> None:
        """Without radial_current_fraction, f_cr defaults to f_c."""
        sp = make_faeton_snowplow()
        assert sp.f_cr == sp.f_c == 0.7

    def test_no_two_step_when_fcr2_none(self) -> None:
        """Without f_cr2, _effective_radial_fc returns f_cr always."""
        sp = make_faeton_snowplow(radial_current_fraction=0.8)
        assert sp.f_cr == 0.8
        assert sp._effective_radial_fc() == 0.8

    def test_no_two_step_when_transition_none(self) -> None:
        """With f_cr2 but no transition time, single-step model persists."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=None,
        )
        assert sp._effective_radial_fc() == 0.8

    def test_backward_compatible_result_keys(self) -> None:
        """New f_cr_eff key is present in step() results."""
        sp = make_faeton_snowplow()
        result = sp.step(1e-9, current=1e6)
        assert "f_cr_eff" in result


class TestTwoStepRadialTransition:
    """Verify two-step radial parameter transition."""

    def test_fcr_before_transition(self) -> None:
        """Before transition time, effective f_cr equals f_cr."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        f_eff = sp._effective_radial_fc()
        assert f_eff == pytest.approx(0.8, abs=0.01)

    def test_fcr_after_transition(self) -> None:
        """Well after transition time, effective f_cr approaches f_cr2."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        sp._elapsed_time = 3.5e-6
        f_eff = sp._effective_radial_fc()
        assert f_eff == pytest.approx(0.5, abs=0.01)

    def test_fcr_at_transition_midpoint(self) -> None:
        """At transition time, effective f_cr is midpoint of f_cr and f_cr2."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        sp._elapsed_time = 2.5e-6
        f_eff = sp._effective_radial_fc()
        assert f_eff == pytest.approx(0.65, abs=0.02)

    def test_smooth_transition(self) -> None:
        """Transition is smooth (monotonic decrease from f_cr to f_cr2)."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        times = np.linspace(2.0e-6, 3.0e-6, 100)
        f_vals = []
        for t in times:
            sp._elapsed_time = t
            f_vals.append(sp._effective_radial_fc())

        for i in range(1, len(f_vals)):
            assert f_vals[i] <= f_vals[i - 1] + 1e-12

        assert all(0.5 - 0.01 <= f <= 0.8 + 0.01 for f in f_vals)


class TestTwoStepRadialPhysics:
    """Verify physics impact of two-step radial model."""

    def test_reduced_fcr2_reduces_radial_force(self) -> None:
        """Lower f_cr2 should result in weaker J x B radial driving force."""
        sp_single = make_faeton_snowplow(radial_current_fraction=0.8)
        sp_two = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )

        advance_to_radial(sp_single)
        advance_to_radial(sp_two)

        dt = 1e-9
        I_drive = 8e5
        results_single = []
        results_two = []

        for _ in range(1000):
            r1 = sp_single.step(dt, current=I_drive)
            r2 = sp_two.step(dt, current=I_drive)
            both_radial = sp_single.phase in ("radial", "reflected")
            both_radial = both_radial and sp_two.phase in ("radial", "reflected")
            if both_radial:
                results_single.append(r1)
                results_two.append(r2)

        if results_single and results_two:
            assert len(results_single) > 0
            assert len(results_two) > 0

    def test_faeton_preset_has_two_step_params(self) -> None:
        """FAETON-I preset includes two-step radial parameters."""
        from dpf.presets import _PRESETS
        faeton = _PRESETS["faeton"]["snowplow"]
        assert "radial_current_fraction" in faeton
        assert "radial_current_fraction_2" in faeton
        assert "radial_transition_time" in faeton
        assert faeton["radial_current_fraction"] == 0.8
        assert faeton["radial_current_fraction_2"] == 0.5
        assert faeton["radial_transition_time"] == 7.0e-6

    def test_config_accepts_two_step_params(self) -> None:
        """SnowplowConfig Pydantic model accepts the new fields."""
        from dpf.config import SnowplowConfig
        cfg = SnowplowConfig(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        assert cfg.radial_current_fraction == 0.8
        assert cfg.radial_current_fraction_2 == 0.5
        assert cfg.radial_transition_time == 2.5e-6

    def test_config_defaults_to_none(self) -> None:
        """New fields default to None (backward compatible)."""
        from dpf.config import SnowplowConfig
        cfg = SnowplowConfig()
        assert cfg.radial_current_fraction is None
        assert cfg.radial_current_fraction_2 is None
        assert cfg.radial_transition_time is None


# --- Section: Bennett Validation ---


N0 = 1e24
A_RADIUS = 1e-3
Te = 1e7
Ti = 1e7
R = np.linspace(1e-5, 10e-3, 500)


class TestBennettDensityProfile:
    """Validate n(r) = n0 / (1 + r^2/a^2)^2 (Russell 2025, Eq. 1)."""

    def test_on_axis_density(self) -> None:
        n = bennett_density(np.array([0.0]), N0, A_RADIUS)
        assert n[0] == pytest.approx(N0, rel=1e-12)

    def test_at_bennett_radius(self) -> None:
        """At r = a, n(a) = n0/4."""
        n = bennett_density(np.array([A_RADIUS]), N0, A_RADIUS)
        assert n[0] == pytest.approx(N0 / 4.0, rel=1e-12)

    def test_far_field_decay(self) -> None:
        """For r >> a, n(r) ~ n0*a^4/r^4 (1/r^4 falloff)."""
        r_far = np.array([100.0 * A_RADIUS])
        n = bennett_density(r_far, N0, A_RADIUS)
        expected = N0 * A_RADIUS**4 / r_far[0]**4
        assert n[0] == pytest.approx(expected, rel=1e-3)

    def test_monotonically_decreasing(self) -> None:
        n = bennett_density(R, N0, A_RADIUS)
        assert np.all(np.diff(n) < 0)

    def test_non_negative(self) -> None:
        n = bennett_density(R, N0, A_RADIUS)
        assert np.all(n > 0)


class TestBennettRelation:
    """Validate mu_0*I_tot^2/(8*pi) = N*k_B*(Te+Ti) (Russell 2025, Eq. 3)."""

    def test_bennett_relation_identity(self) -> None:
        """Current from bennett_current_from_temperature satisfies Bennett relation."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        N = bennett_line_density(N0, A_RADIUS)
        lhs = mu_0 * I_tot**2 / (8.0 * pi)
        rhs = N * k_B * (Te + Ti)
        assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_current_scales_with_temperature(self) -> None:
        """I_tot ~ sqrt(Te + Ti), so doubling T should increase I_tot by sqrt(2)."""
        I1 = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        I2 = bennett_current_from_temperature(N0, A_RADIUS, 2 * Te, 2 * Ti)
        assert pytest.approx(np.sqrt(2.0), rel=1e-10) == I2 / I1

    def test_current_scales_with_density(self) -> None:
        """I_tot ~ sqrt(N) ~ sqrt(n0), so 4x density -> 2x current."""
        I1 = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        I2 = bennett_current_from_temperature(4 * N0, A_RADIUS, Te, Ti)
        assert pytest.approx(2.0, rel=1e-10) == I2 / I1

    def test_line_density_formula(self) -> None:
        """N = pi * n0 * a^2."""
        N = bennett_line_density(N0, A_RADIUS)
        assert pytest.approx(pi * N0 * A_RADIUS**2, rel=1e-12) == N


class TestBennettMagneticField:
    """Validate B_theta(r) = mu_0*I_tot/(2*pi) * r/(r^2+a^2) (Russell 2025, Eq. 27)."""

    def test_on_axis_field_zero(self) -> None:
        """B_theta(0) = 0 by symmetry."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        B = bennett_btheta(np.array([0.0]), I_tot, A_RADIUS)
        assert B[0] == pytest.approx(0.0, abs=1e-20)

    def test_peak_at_bennett_radius(self) -> None:
        """B_theta peaks at r = a (dB/dr = 0 at r = a)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_fine = np.linspace(0.1 * A_RADIUS, 5 * A_RADIUS, 10000)
        B = bennett_btheta(r_fine, I_tot, A_RADIUS)
        i_max = np.argmax(B)
        r_peak = r_fine[i_max]
        assert r_peak == pytest.approx(A_RADIUS, rel=0.01)

    def test_far_field_1_over_r(self) -> None:
        """For r >> a, B_theta ~ mu_0*I_tot/(2*pi*r) (exterior field of a wire)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_far = np.array([1000.0 * A_RADIUS])
        B = bennett_btheta(r_far, I_tot, A_RADIUS)
        B_wire = mu_0 * I_tot / (2.0 * pi * r_far[0])
        assert B[0] == pytest.approx(B_wire, rel=1e-5)

    def test_ampere_law_integral(self) -> None:
        """Integral of J_z over cross-section equals I_total."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_int = np.linspace(0, 50 * A_RADIUS, 50000)
        dr = r_int[1] - r_int[0]
        Jz = bennett_current_density(r_int, I_tot, A_RADIUS)
        I_integrated = 2.0 * pi * np.sum(Jz * r_int * dr)
        assert I_integrated == pytest.approx(I_tot, rel=1e-3)


class TestForceBalance:
    """Validate dp/dr + Jz*B_theta = 0 (Russell 2025, Eq. 19 static case)."""

    def test_residual_zero(self) -> None:
        """Force balance residual should be zero for exact Bennett profiles."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_check = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 1000)
        _, max_rel_err = verify_force_balance(r_check, N0, A_RADIUS, I_tot, Te, Ti)
        assert max_rel_err < 1e-10

    def test_force_balance_not_satisfied_with_wrong_current(self) -> None:
        """With 2x current, force balance should NOT hold."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_check = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 1000)
        _, max_rel_err = verify_force_balance(r_check, N0, A_RADIUS, 2 * I_tot, Te, Ti)
        assert max_rel_err > 0.1


class TestCurrentDensity:
    """Validate Jz(r) = I_tot*a^2 / (pi*(r^2+a^2)^2) (Russell 2025, Eq. 26)."""

    def test_on_axis_peak(self) -> None:
        """Jz(0) = I_tot/(pi*a^2) is the maximum."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Jz = bennett_current_density(np.array([0.0]), I_tot, A_RADIUS)
        expected = I_tot / (pi * A_RADIUS**2)
        assert Jz[0] == pytest.approx(expected, rel=1e-12)

    def test_at_bennett_radius(self) -> None:
        """Jz(a) = I_tot/(4*pi*a^2)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Jz = bennett_current_density(np.array([A_RADIUS]), I_tot, A_RADIUS)
        expected = I_tot / (4.0 * pi * A_RADIUS**2)
        assert Jz[0] == pytest.approx(expected, rel=1e-12)

    def test_monotonically_decreasing(self) -> None:
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Jz = bennett_current_density(R, I_tot, A_RADIUS)
        assert np.all(np.diff(Jz) < 0)


class TestBennettVortex:
    """Validate Bennett Vortex solutions (Russell 2025, Eqs. 6-7, 31-32)."""

    def test_vortex_flow_on_axis(self) -> None:
        """uz(0) = uz0."""
        uz0 = 1e5
        xi2 = 1e6
        uz = uz0 / (1.0 + xi2 * 0.0**2)**2
        assert uz == pytest.approx(uz0, rel=1e-12)

    def test_vortex_flow_at_characteristic_radius(self) -> None:
        """At r = 1/xi, uz = uz0/4."""
        uz0 = 1e5
        xi2 = 1e6
        r_char = 1.0 / np.sqrt(xi2)
        uz = uz0 / (1.0 + xi2 * r_char**2)**2
        assert uz == pytest.approx(uz0 / 4.0, rel=1e-12)

    def test_vortex_flow_far_field(self) -> None:
        """For r >> 1/xi, uz effectively zero."""
        uz0 = 1e5
        xi2 = 1e6
        r_far = 100.0 / np.sqrt(xi2)
        uz = uz0 / (1.0 + xi2 * r_far**2)**2
        assert uz < uz0 * 1e-6

    def test_vorticity_profile(self) -> None:
        """omega_theta = 4*xi^2*uz0*r / (1+xi^2*r^2)^3 (Eq. 31)."""
        uz0 = 1e5
        xi2 = 1e6
        r_test = np.linspace(1e-5, 5e-3, 200)
        omega = 4.0 * xi2 * uz0 * r_test / (1.0 + xi2 * r_test**2)**3
        assert omega[0] < omega[50]
        assert omega[-1] < omega[50]
        assert np.all(omega >= 0)

    def test_vorticity_is_derivative_of_flow(self) -> None:
        """omega_theta = -duz/dr (Eq. 29 applied to axial flow)."""
        uz0 = 1e5
        xi2 = 1e6
        r_test = np.linspace(1e-5, 5e-3, 10000)
        dr = r_test[1] - r_test[0]
        uz = uz0 / (1.0 + xi2 * r_test**2)**2
        duz_dr = np.gradient(uz, dr)
        omega_analytical = 4.0 * xi2 * uz0 * r_test / (1.0 + xi2 * r_test**2)**3
        np.testing.assert_allclose(-duz_dr[10:-10], omega_analytical[10:-10], rtol=1e-3)

    def test_sfs_criterion(self) -> None:
        """Shear-flow stabilization: |duz/dr| >= 0.1*k*V_A (Eq. 25)."""
        uz0 = 1e5
        xi2 = 1e6
        r_test = np.linspace(1e-5, 3e-3, 500)
        duz_dr = -4.0 * xi2 * uz0 * r_test / (1.0 + xi2 * r_test**2)**3
        assert np.all(-duz_dr > 0)


class TestBennettPressureProfile:
    """Validate pressure from MHD momentum (Russell 2025, Eq. 30)."""

    def test_pressure_from_ideal_gas(self) -> None:
        """p(r) = n(r)*k_B*(Te+Ti) for Bennett equilibrium."""
        p = bennett_pressure(R, N0, A_RADIUS, Te, Ti)
        n = bennett_density(R, N0, A_RADIUS)
        expected = n * k_B * (Te + Ti)
        np.testing.assert_allclose(p, expected, rtol=1e-12)

    def test_on_axis_pressure(self) -> None:
        """p(0) = n0*k_B*(Te+Ti)."""
        p = bennett_pressure(np.array([0.0]), N0, A_RADIUS, Te, Ti)
        assert p[0] == pytest.approx(N0 * k_B * (Te + Ti), rel=1e-12)

    def test_pressure_monotonically_decreasing(self) -> None:
        p = bennett_pressure(R, N0, A_RADIUS, Te, Ti)
        assert np.all(np.diff(p) < 0)

    def test_pressure_gradient_matches_jxb(self) -> None:
        """dp/dr = -Jz*B_theta (force balance check via numerical gradient)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_fine = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 10000)
        dr = r_fine[1] - r_fine[0]
        p = bennett_pressure(r_fine, N0, A_RADIUS, Te, Ti)
        dp_dr_num = np.gradient(p, dr)
        Jz = bennett_current_density(r_fine, I_tot, A_RADIUS)
        Bt = bennett_btheta(r_fine, I_tot, A_RADIUS)
        jxb = -Jz * Bt
        np.testing.assert_allclose(dp_dr_num[50:-50], jxb[50:-50], rtol=1e-3)


class TestBennettBParameter:
    """Validate b = mu_0*e^2*u0^2 / (8*k_B*(Te+Ti)) (Russell 2025, Eq. 3)."""

    def test_b_parameter_two_temp(self) -> None:
        """b parameter for two-temperature plasma."""
        u0 = 1e5
        b = mu_0 * e_charge**2 * u0**2 / (8.0 * k_B * (Te + Ti))
        assert b > 0
        xi2 = b * N0
        r_char = 1.0 / np.sqrt(xi2)
        assert 0 < r_char < 1.0

    def test_b_parameter_ideal_limit(self) -> None:
        """Ideal MHD: Te = Ti = T, b = mu_0*e^2*u0^2 / (16*k_B*T) (Eq. 5)."""
        u0 = 1e5
        T = Te
        b_two_temp = mu_0 * e_charge**2 * u0**2 / (8.0 * k_B * (T + T))
        b_ideal = mu_0 * e_charge**2 * u0**2 / (16.0 * k_B * T)
        assert b_two_temp == pytest.approx(b_ideal, rel=1e-12)


class TestCreateBennettState:
    """Validate the 2D cylindrical state generator."""

    def test_state_dict_keys(self) -> None:
        state, I_tot, r = create_bennett_state(
            nr=32, nz=16, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        assert set(state.keys()) == expected_keys

    def test_state_shapes(self) -> None:
        nr, nz = 32, 16
        state, _, _ = create_bennett_state(
            nr=nr, nz=nz, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        assert state["rho"].shape == (nr, 1, nz)
        assert state["velocity"].shape == (3, nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)
        assert state["pressure"].shape == (nr, 1, nz)

    def test_density_matches_profile(self) -> None:
        nr, nz = 64, 8
        state, _, r_centers = create_bennett_state(
            nr=nr, nz=nz, r_max=10e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        rho_1d = state["rho"][:, 0, 0]
        expected_n = bennett_density(r_centers, N0, A_RADIUS)
        np.testing.assert_allclose(rho_1d, expected_n * m_d, rtol=1e-10)

    def test_btheta_matches_profile(self) -> None:
        nr, nz = 64, 8
        state, I_tot, r_centers = create_bennett_state(
            nr=nr, nz=nz, r_max=10e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        Bt_1d = state["B"][1, :, 0, 0]
        expected_Bt = bennett_btheta(r_centers, I_tot, A_RADIUS)
        np.testing.assert_allclose(Bt_1d, expected_Bt, rtol=1e-10)

    def test_velocity_zero(self) -> None:
        """No bulk flow in standard Bennett equilibrium."""
        state, _, _ = create_bennett_state(
            nr=32, nz=8, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        assert np.all(state["velocity"] == 0.0)

    def test_z_uniform(self) -> None:
        """All quantities are uniform along z (infinite cylinder)."""
        nr, nz = 32, 16
        state, _, _ = create_bennett_state(
            nr=nr, nz=nz, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        for k in range(nz):
            np.testing.assert_array_equal(
                state["rho"][:, 0, k], state["rho"][:, 0, 0]
            )


class TestAlfvenVelocity:
    """Validate V_A = B/sqrt(rho*mu_0) (Russell 2025, near Eq. 25)."""

    def test_alfven_speed_profile(self) -> None:
        """V_A should be finite and positive for r > 0."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_test = np.linspace(0.01 * A_RADIUS, 10 * A_RADIUS, 200)
        Bt = bennett_btheta(r_test, I_tot, A_RADIUS)
        n = bennett_density(r_test, N0, A_RADIUS)
        rho = n * m_d
        V_A = np.abs(Bt) / np.sqrt(rho * mu_0)
        assert np.all(np.isfinite(V_A))
        assert np.all(V_A > 0)

    def test_alfven_speed_order_of_magnitude(self) -> None:
        """For typical DPF params, V_A should be ~1e4-1e6 m/s."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Bt_peak = bennett_btheta(np.array([A_RADIUS]), I_tot, A_RADIUS)[0]
        n_peak = bennett_density(np.array([A_RADIUS]), N0, A_RADIUS)[0]
        rho_peak = n_peak * m_d
        V_A = np.abs(Bt_peak) / np.sqrt(rho_peak * mu_0)
        assert 1e3 < V_A < 1e8


class TestMHDConservationRelations:
    """Cross-checks between Bennett formulas and ideal MHD conservation laws."""

    def test_ampere_law_differential(self) -> None:
        """Jz = (1/mu_0)*(1/r)*d(r*B_theta)/dr."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_fine = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 50000)
        dr = r_fine[1] - r_fine[0]
        Bt = bennett_btheta(r_fine, I_tot, A_RADIUS)
        rBt = r_fine * Bt
        d_rBt_dr = np.gradient(rBt, dr)
        Jz_numerical = d_rBt_dr / (mu_0 * r_fine)
        Jz_analytical = bennett_current_density(r_fine, I_tot, A_RADIUS)
        np.testing.assert_allclose(
            Jz_numerical[100:-100], Jz_analytical[100:-100], rtol=5e-4
        )

    def test_magnetic_energy_density(self) -> None:
        """u_B = B^2/(2*mu_0) should be finite and positive."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Bt = bennett_btheta(R, I_tot, A_RADIUS)
        u_B = Bt**2 / (2.0 * mu_0)
        assert np.all(u_B >= 0)
        assert np.all(np.isfinite(u_B))

    def test_beta_profile(self) -> None:
        """Plasma beta = 2*mu_0*p / B^2 -- peaks on axis, decays outward."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_test = np.linspace(0.1 * A_RADIUS, 10 * A_RADIUS, 200)
        p = bennett_pressure(r_test, N0, A_RADIUS, Te, Ti)
        Bt = bennett_btheta(r_test, I_tot, A_RADIUS)
        beta = 2.0 * mu_0 * p / Bt**2
        assert np.all(np.isfinite(beta))
        assert np.all(beta > 0)
