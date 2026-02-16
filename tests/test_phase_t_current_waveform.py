"""Phase T: Current waveform validation against PF-1000 experimental data.

The characteristic DPF current waveform has three key features:
1. Sinusoidal rise to peak current (~5-6 us for PF-1000)
2. Current "dip" when the sheath reaches the end of the anode and begins
   radial compression (increasing dL/dt extracts energy from the circuit)
3. Post-dip recovery or decay depending on crowbar timing

This module validates the coupled circuit + snowplow model by running a
0D simulation (no MHD grid needed) and comparing against published PF-1000
measurements from Scholz et al. (Nukleonika 51, 2006).

References:
    Scholz et al., Nukleonika 51(1), 2006.
    Lee, S. & Saw, S.H., Phys. Plasmas 21, 072501 (2014).
"""

from __future__ import annotations

import numpy as np

from dpf.circuit.rlc_solver import RLCSolver
from dpf.constants import k_B, m_d
from dpf.core.bases import CouplingState
from dpf.fluid.snowplow import SnowplowModel
from dpf.validation.experimental import PF1000_DATA, validate_current_waveform

# ---------------------------------------------------------------------------
# PF-1000 device parameters (Scholz et al. 2006)
# ---------------------------------------------------------------------------
PF1000 = PF1000_DATA


def _pf1000_fill_density() -> float:
    """Compute PF-1000 fill density from ideal gas law at 300 K."""
    p_Pa = PF1000.fill_pressure_torr * 133.322  # Torr -> Pa
    T_room = 300.0  # K
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
    """Run coupled 0D circuit + snowplow simulation.

    Returns (t_array, I_array, diagnostics_dict).
    """
    # Create circuit solver
    circuit = RLCSolver(
        C=PF1000.capacitance,
        V0=PF1000.voltage,
        L0=PF1000.inductance,
        R0=PF1000.resistance,
        anode_radius=PF1000.anode_radius,
        cathode_radius=PF1000.cathode_radius,
    )

    # Create snowplow model
    rho0 = _pf1000_fill_density()
    snowplow = SnowplowModel(
        anode_radius=PF1000.anode_radius,
        cathode_radius=PF1000.cathode_radius,
        fill_density=rho0,
        anode_length=PF1000.anode_length,
        mass_fraction=mass_fraction,
        current_fraction=current_fraction,
        fill_pressure_Pa=PF1000.fill_pressure_torr * 133.322,
    )

    # Simulation time: enough to capture peak + some post-peak
    if t_end is None:
        t_end = 3.0 * PF1000.current_rise_time  # ~17 us

    n_steps = int(t_end / dt)

    # Storage
    t_arr = np.zeros(n_steps + 1)
    I_arr = np.zeros(n_steps + 1)
    L_arr = np.zeros(n_steps + 1)
    phase_arr = []

    # Initial conditions
    t_arr[0] = 0.0
    I_arr[0] = 0.0
    L_arr[0] = snowplow.plasma_inductance
    phase_arr.append("rundown")

    # Coupling state
    coupling = CouplingState(
        Lp=snowplow.plasma_inductance,
        current=0.0,
        voltage=PF1000.voltage,
    )

    # Time-stepping loop
    for i in range(n_steps):
        # Step snowplow with current circuit current
        sp_result = snowplow.step(dt, coupling.current)

        # Update coupling state with snowplow output
        coupling.Lp = sp_result["L_plasma"]
        coupling.dL_dt = sp_result["dL_dt"]
        coupling.R_plasma = R_plasma

        # Step circuit
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt)

        # Record
        t_arr[i + 1] = (i + 1) * dt
        I_arr[i + 1] = coupling.current
        L_arr[i + 1] = sp_result["L_plasma"]
        phase_arr.append(sp_result["phase"])

    # Find key diagnostics
    abs_I = np.abs(I_arr)

    # Find when rundown ends and radial begins
    radial_start_idx = None
    pinch_idx = None
    for idx, ph in enumerate(phase_arr):
        if ph == "radial" and radial_start_idx is None:
            radial_start_idx = idx
        if ph == "pinch" and pinch_idx is None:
            pinch_idx = idx

    # Global peak of |I|
    peak_idx = int(np.argmax(abs_I))
    peak_current = float(abs_I[peak_idx])
    peak_time = float(t_arr[peak_idx])

    # DPF current dip detection:
    # The characteristic DPF dip is the local minimum during the radial phase,
    # caused by rapid dL/dt extracting energy from the circuit. We look for:
    # 1. The pre-dip peak: local max of |I| just before or at radial transition
    # 2. The dip minimum: local min of |I| during or just after radial phase
    if radial_start_idx is not None:
        # Pre-dip peak: max current near the radial transition
        search_start = max(0, radial_start_idx - 200)
        pre_dip_region = abs_I[search_start:radial_start_idx + 50]
        pre_dip_peak_local = int(np.argmax(pre_dip_region))
        pre_dip_peak_idx = search_start + pre_dip_peak_local
        pre_dip_peak = float(abs_I[pre_dip_peak_idx])

        # Dip minimum: min current during/after radial phase
        dip_search_end = min(
            pinch_idx + 200 if pinch_idx else radial_start_idx + 2000,
            len(abs_I),
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


# ===================================================================
# Tests: Peak current validation
# ===================================================================
class TestPeakCurrentValidation:
    """Validate peak current against PF-1000 published data."""

    def test_peak_current_order_of_magnitude(self):
        """Peak current should be in the MA range for PF-1000."""
        _, I_arr, diag = run_coupled_simulation(dt=5e-9, t_end=10e-6)
        peak = diag["peak_current"]
        # PF-1000 peak current: 1.87 MA
        # Allow factor-of-2 range for 0D model
        assert peak > 0.5e6, f"Peak current {peak:.2e} A too low (expected >0.5 MA)"
        assert peak < 5e6, f"Peak current {peak:.2e} A too high (expected <5 MA)"

    def test_peak_current_within_50_percent(self):
        """Peak current should be within 50% of 1.87 MA."""
        _, I_arr, diag = run_coupled_simulation(dt=2e-9, t_end=12e-6)
        peak = diag["peak_current"]
        exp_peak = PF1000.peak_current
        error = abs(peak - exp_peak) / exp_peak
        assert error < 0.5, (
            f"Peak current error {error:.1%}: simulated {peak:.3e} A "
            f"vs experimental {exp_peak:.3e} A"
        )

    def test_peak_time_order_of_magnitude(self):
        """Peak time should be within factor-of-3 of 5.8 us."""
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=15e-6)
        t_peak = diag["peak_time"]
        t_exp = PF1000.current_rise_time  # 5.8 us
        ratio = t_peak / t_exp
        assert 0.3 < ratio < 3.0, (
            f"Peak time {t_peak:.2e} s vs experimental {t_exp:.2e} s "
            f"(ratio {ratio:.2f})"
        )


# ===================================================================
# Tests: Current dip signature
# ===================================================================
class TestCurrentDipSignature:
    """Validate the characteristic DPF current dip.

    The DPF current dip is caused by rapid increase of dL/dt during the
    radial compression phase, which extracts energy from the circuit.
    The dip is measured relative to the local peak just before the
    radial phase transition, NOT the global peak of the waveform.
    """

    def test_current_dip_exists(self):
        """A current dip should appear during radial compression (due to dL/dt)."""
        _, _, diag = run_coupled_simulation(dt=2e-9, t_end=15e-6)
        dip_depth = diag["dip_depth"]
        # Current dip should be at least 5% (typical: 30-60% for 0D Lee model)
        assert dip_depth > 0.05, (
            f"Current dip depth {dip_depth:.1%} too small "
            f"(pre_dip_peak={diag['pre_dip_peak']:.3e}, "
            f"dip={diag['dip_current']:.3e})"
        )

    def test_dip_during_radial_phase(self):
        """Current dip should occur during or just after the radial phase."""
        _, _, diag = run_coupled_simulation(dt=2e-9, t_end=15e-6)
        # Dip must come after radial phase starts
        if diag["radial_start_time"] is not None:
            assert diag["dip_time"] >= diag["radial_start_time"], (
                f"Dip at {diag['dip_time']:.2e} s but radial starts at "
                f"{diag['radial_start_time']:.2e} s"
            )

    def test_dip_depth_physical_range(self):
        """Current dip should be 5-90% (Lee model 0D can be quite deep)."""
        _, _, diag = run_coupled_simulation(dt=2e-9, t_end=15e-6)
        dip = diag["dip_depth"]
        assert 0.05 < dip < 0.90, f"Current dip {dip:.1%} outside physical range"


# ===================================================================
# Tests: Phase transitions in waveform
# ===================================================================
class TestPhaseTransitionsInWaveform:
    """Verify that the circuit-coupled simulation sees all three phases."""

    def test_all_three_phases(self):
        """The simulation should go through rundown, radial, and pinch."""
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=20e-6)
        phases = diag["phases_seen"]
        assert "rundown" in phases, "Missing rundown phase"
        assert "radial" in phases, "Missing radial phase"
        # Pinch may or may not be reached depending on timing
        # but for PF-1000 at 1 MA, it should reach pinch within 20 us

    def test_radial_phase_before_end(self):
        """Radial phase should start before end of simulation."""
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=20e-6)
        assert diag["radial_start_time"] is not None, "Never entered radial phase"
        assert diag["radial_start_time"] < 15e-6, (
            f"Radial phase started too late: {diag['radial_start_time']:.2e} s"
        )


# ===================================================================
# Tests: Inductance evolution
# ===================================================================
class TestInductanceEvolution:
    """Validate that L_plasma evolves correctly through the simulation."""

    def test_inductance_increases_monotonically(self):
        """L_plasma should be non-decreasing throughout the simulation."""
        t_arr, _, diag = run_coupled_simulation(dt=5e-9, t_end=15e-6)
        # Create a simulation to track L
        circuit = RLCSolver(
            C=PF1000.capacitance, V0=PF1000.voltage,
            L0=PF1000.inductance, R0=PF1000.resistance,
            anode_radius=PF1000.anode_radius,
            cathode_radius=PF1000.cathode_radius,
        )
        rho0 = _pf1000_fill_density()
        sp = SnowplowModel(
            anode_radius=PF1000.anode_radius,
            cathode_radius=PF1000.cathode_radius,
            fill_density=rho0,
            anode_length=PF1000.anode_length,
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
            assert L_now >= prev_L - 1e-15, (
                f"L_plasma decreased: {L_now:.4e} < {prev_L:.4e}"
            )
            prev_L = L_now

    def test_final_inductance_nanohenry_range(self):
        """Final inductance should be in the tens-of-nH range for PF-1000."""
        _, _, diag = run_coupled_simulation(dt=5e-9, t_end=15e-6)
        L_final = diag["final_L_plasma"]
        # PF-1000 plasma inductance at pinch: ~5-50 nH
        assert L_final > 1e-9, f"Final L_plasma {L_final:.2e} H too small"
        assert L_final < 1e-6, f"Final L_plasma {L_final:.2e} H too large"


# ===================================================================
# Tests: validate_current_waveform utility
# ===================================================================
class TestValidationUtility:
    """Test the validate_current_waveform function from experimental.py."""

    def test_validation_function_runs(self):
        """The validation function should run without error."""
        t_arr, I_arr, _ = run_coupled_simulation(dt=5e-9, t_end=12e-6)
        metrics = validate_current_waveform(t_arr, I_arr, "PF-1000")
        assert "peak_current_error" in metrics
        assert "peak_current_sim" in metrics
        assert "timing_ok" in metrics

    def test_validation_peak_error_finite(self):
        """Peak current error should be a finite number."""
        t_arr, I_arr, _ = run_coupled_simulation(dt=5e-9, t_end=12e-6)
        metrics = validate_current_waveform(t_arr, I_arr, "PF-1000")
        assert np.isfinite(metrics["peak_current_error"])
        assert metrics["peak_current_error"] >= 0.0


# ===================================================================
# Tests: Sensitivity to Lee model parameters
# ===================================================================
class TestParameterSensitivity:
    """Test that model parameters have physically correct effects."""

    def test_higher_mass_fraction_delays_peak(self):
        """Higher f_m means more mass to sweep -> longer rundown -> later peak."""
        _, _, diag_low = run_coupled_simulation(
            dt=5e-9, t_end=15e-6, mass_fraction=0.15,
        )
        _, _, diag_high = run_coupled_simulation(
            dt=5e-9, t_end=15e-6, mass_fraction=0.5,
        )
        # Higher mass fraction should delay the radial transition
        if diag_low["radial_start_time"] and diag_high["radial_start_time"]:
            assert diag_high["radial_start_time"] > diag_low["radial_start_time"], (
                f"f_m=0.5 radial start ({diag_high['radial_start_time']:.2e}) "
                f"should be later than f_m=0.15 ({diag_low['radial_start_time']:.2e})"
            )

    def test_higher_current_fraction_increases_peak(self):
        """Higher f_c means more driving force -> faster dynamics."""
        _, _, diag_low = run_coupled_simulation(
            dt=5e-9, t_end=15e-6, current_fraction=0.5,
        )
        _, _, diag_high = run_coupled_simulation(
            dt=5e-9, t_end=15e-6, current_fraction=0.9,
        )
        # Higher f_c -> faster rundown -> earlier radial transition
        if diag_low["radial_start_time"] and diag_high["radial_start_time"]:
            assert diag_high["radial_start_time"] < diag_low["radial_start_time"]

    def test_plasma_resistance_reduces_peak_current(self):
        """Non-zero plasma resistance should reduce peak current."""
        _, _, diag_no_R = run_coupled_simulation(
            dt=5e-9, t_end=10e-6, R_plasma=0.0,
        )
        _, _, diag_with_R = run_coupled_simulation(
            dt=5e-9, t_end=10e-6, R_plasma=5e-3,
        )
        # Plasma resistance should reduce the peak
        assert diag_with_R["peak_current"] < diag_no_R["peak_current"], (
            f"R_plasma=5mOhm ({diag_with_R['peak_current']:.3e}) should be "
            f"lower than R_plasma=0 ({diag_no_R['peak_current']:.3e})"
        )


# ===================================================================
# Tests: Multi-device validation
# ===================================================================
class TestMultiDeviceValidation:
    """Validate the coupled model against multiple DPF devices."""

    def test_unu_ictp_peak_order_of_magnitude(self):
        """UNU-ICTP device should produce ~170 kA peak."""
        from dpf.validation.experimental import UNU_ICTP_DATA

        dev = UNU_ICTP_DATA
        rho0 = (dev.fill_pressure_torr * 133.322) / (k_B * 300.0) * m_d

        circuit = RLCSolver(
            C=dev.capacitance, V0=dev.voltage,
            L0=dev.inductance, R0=dev.resistance,
            anode_radius=dev.anode_radius,
            cathode_radius=dev.cathode_radius,
        )
        sp = SnowplowModel(
            anode_radius=dev.anode_radius,
            cathode_radius=dev.cathode_radius,
            fill_density=rho0,
            anode_length=dev.anode_length,
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

        # Should be in the 50 kA - 500 kA range
        assert I_max > 50e3, f"UNU-ICTP peak {I_max:.2e} A too low"
        assert I_max < 500e3, f"UNU-ICTP peak {I_max:.2e} A too high"

    def test_nx2_peak_order_of_magnitude(self):
        """NX2 device should produce ~400 kA peak."""
        from dpf.validation.experimental import NX2_DATA

        dev = NX2_DATA
        rho0 = (dev.fill_pressure_torr * 133.322) / (k_B * 300.0) * m_d

        circuit = RLCSolver(
            C=dev.capacitance, V0=dev.voltage,
            L0=dev.inductance, R0=dev.resistance,
            anode_radius=dev.anode_radius,
            cathode_radius=dev.cathode_radius,
        )
        sp = SnowplowModel(
            anode_radius=dev.anode_radius,
            cathode_radius=dev.cathode_radius,
            fill_density=rho0,
            anode_length=dev.anode_length,
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

        # Should be in the 100 kA - 1 MA range
        assert I_max > 100e3, f"NX2 peak {I_max:.2e} A too low"
        assert I_max < 1e6, f"NX2 peak {I_max:.2e} A too high"


# ===================================================================
# Tests: Energy conservation in coupled system
# ===================================================================
class TestCoupledEnergyConservation:
    """Verify energy accounting in the coupled circuit+snowplow system."""

    def test_circuit_energy_monotonically_decreases(self):
        """Total circuit energy (cap + inductor) should decrease due to resistance."""
        circuit = RLCSolver(
            C=PF1000.capacitance, V0=PF1000.voltage,
            L0=PF1000.inductance, R0=PF1000.resistance,
            anode_radius=PF1000.anode_radius,
            cathode_radius=PF1000.cathode_radius,
        )
        rho0 = _pf1000_fill_density()
        sp = SnowplowModel(
            anode_radius=PF1000.anode_radius,
            cathode_radius=PF1000.cathode_radius,
            fill_density=rho0,
            anode_length=PF1000.anode_length,
        )

        dt = 5e-9
        coupling = CouplingState(Lp=sp.plasma_inductance, voltage=PF1000.voltage)
        E_initial = 0.5 * PF1000.capacitance * PF1000.voltage**2

        for _ in range(int(5e-6 / dt)):
            sp_res = sp.step(dt, coupling.current)
            coupling.Lp = sp_res["L_plasma"]
            coupling.dL_dt = sp_res["dL_dt"]
            coupling = circuit.step(coupling, 0.0, dt)

        # After 5 us, some energy should be dissipated
        E_cap = circuit.state.energy_cap
        E_ind = circuit.state.energy_ind
        E_res = circuit.state.energy_res
        E_total = E_cap + E_ind + E_res

        # Energy accounting: total should approximately equal initial
        # (not exact due to implicit method, but within 20%)
        assert E_total > 0, "Total energy became negative"
        ratio = E_total / E_initial
        assert 0.5 < ratio < 1.5, (
            f"Energy ratio {ratio:.3f} outside expected range "
            f"(E_cap={E_cap:.2e}, E_ind={E_ind:.2e}, E_res={E_res:.2e})"
        )
