"""Multi-device experimental validation data for DPF simulations.

Provides published experimental parameters and measured observables for
well-characterised Dense Plasma Focus devices.  These can be used to:

    1. Configure a simulation to match a real device geometry and
       electrical parameters.
    2. Validate simulated current waveforms and neutron yields against
       published measurements.

Devices included:
    - **PF-1000** (IPPLM Warsaw, Poland) -- the largest DPF in Europe.
    - **NX2** (NIE Singapore) -- compact Mather-type DPF.
    - **UNU-ICTP PFF** -- the widely-replicated training device.

Usage::

    from dpf.validation.experimental import (
        PF1000_DATA, NX2_DATA, UNU_ICTP_DATA,
        validate_current_waveform,
        validate_neutron_yield,
        device_to_config_dict,
    )

    metrics = validate_current_waveform(t_sim, I_sim, "PF-1000")
    print(f"Peak current error: {metrics['peak_current_error']:.1%}")

Units: SI throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dpf.constants import k_B, m_d

# =====================================================================
# Experimental device dataclass
# =====================================================================

@dataclass
class ExperimentalDevice:
    """Published experimental data for a Dense Plasma Focus device.

    Attributes
    ----------
    name : str
        Device name.
    institution : str
        Host institution / laboratory.
    capacitance : float
        Bank capacitance [F].
    voltage : float
        Charging voltage [V].
    inductance : float
        External (stray) inductance [H].
    resistance : float
        External (stray) resistance [Ohm].
    anode_radius : float
        Anode radius [m].
    cathode_radius : float
        Cathode radius [m].
    anode_length : float
        Anode length [m].
    fill_pressure_torr : float
        Fill gas pressure [Torr].
    fill_gas : str
        Fill gas species (e.g. ``"deuterium"``).
    peak_current : float
        Measured peak discharge current [A].
    neutron_yield : float
        Measured total DD neutron yield per shot.
    current_rise_time : float
        Measured current quarter-period (time to first peak) [s].
    reference : str
        Publication reference string.
    """

    name: str
    institution: str
    capacitance: float        # [F]
    voltage: float            # [V]
    inductance: float         # [H]
    resistance: float         # [Ohm]
    anode_radius: float       # [m]
    cathode_radius: float     # [m]
    anode_length: float       # [m]
    fill_pressure_torr: float
    fill_gas: str
    peak_current: float       # [A]
    neutron_yield: float
    current_rise_time: float  # [s]
    reference: str


# =====================================================================
# Device data
# =====================================================================

PF1000_DATA = ExperimentalDevice(
    name="PF-1000",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # 1.332 mF
    voltage=27e3,                  # 27 kV
    inductance=33e-9,              # 33 nH
    resistance=2.3e-3,             # 2.3 mOhm
    anode_radius=0.0575,           # 57.5 mm (Scholz et al. 2006)
    cathode_radius=0.08,           # 80 mm
    anode_length=0.16,             # 160 mm
    fill_pressure_torr=3.5,
    fill_gas="deuterium",
    peak_current=1.87e6,           # 1.87 MA
    neutron_yield=1e11,
    current_rise_time=5.8e-6,      # 5.8 us
    reference="Scholz et al., Nukleonika 51(1), 2006",
)

NX2_DATA = ExperimentalDevice(
    name="NX2",
    institution="NIE Singapore",
    capacitance=28e-6,             # 28 uF
    voltage=14e3,                  # 14 kV
    inductance=20e-9,              # 20 nH
    resistance=5e-3,               # 5 mOhm
    anode_radius=0.019,            # 19 mm
    cathode_radius=0.041,          # 41 mm
    anode_length=0.05,             # 50 mm
    fill_pressure_torr=4.0,
    fill_gas="deuterium",
    peak_current=400e3,            # 400 kA
    neutron_yield=1e8,
    current_rise_time=1.8e-6,      # 1.8 us
    reference="Lee & Saw, J. Fusion Energy 27, 2008",
)

UNU_ICTP_DATA = ExperimentalDevice(
    name="UNU-ICTP",
    institution="UNU-ICTP PFF",
    capacitance=30e-6,             # 30 uF
    voltage=14e3,                  # 14 kV
    inductance=110e-9,             # 110 nH
    resistance=12e-3,              # 12 mOhm
    anode_radius=0.0095,           # 9.5 mm
    cathode_radius=0.032,          # 32 mm
    anode_length=0.16,             # 160 mm
    fill_pressure_torr=3.0,
    fill_gas="deuterium",
    peak_current=170e3,            # 170 kA
    neutron_yield=1e8,
    current_rise_time=2.8e-6,      # 2.8 us
    reference="Lee et al., Am. J. Phys. 56, 1988",
)


# Registry mapping device name -> ExperimentalDevice
DEVICES: dict[str, ExperimentalDevice] = {
    "PF-1000": PF1000_DATA,
    "NX2": NX2_DATA,
    "UNU-ICTP": UNU_ICTP_DATA,
}


# =====================================================================
# Validation functions
# =====================================================================

def validate_current_waveform(
    t_sim: np.ndarray,
    I_sim: np.ndarray,
    device_name: str,
) -> dict[str, Any]:
    """Validate a simulated current waveform against experimental data.

    Compares the peak current magnitude and its timing against published
    measurements for the specified device.

    Parameters
    ----------
    t_sim : ndarray, shape (M,)
        Simulated time array [s].
    I_sim : ndarray, shape (M,)
        Simulated current waveform [A].
    device_name : str
        Key into ``DEVICES`` (e.g. ``"PF-1000"``, ``"NX2"``).

    Returns
    -------
    dict
        ``peak_current_error`` : float
            Relative error |I_peak_sim - I_peak_exp| / I_peak_exp.
        ``peak_current_sim`` : float
            Peak of simulated current [A].
        ``peak_current_exp`` : float
            Experimental peak current [A].
        ``timing_ok`` : bool
            True if simulated peak time is within 50 % of experimental
            rise time.

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    t_arr = np.asarray(t_sim, dtype=np.float64)
    I_arr = np.asarray(I_sim, dtype=np.float64)

    # Peak current (use absolute value to handle sign conventions)
    abs_I = np.abs(I_arr)
    peak_idx = int(np.argmax(abs_I))
    peak_current_sim = float(abs_I[peak_idx])
    peak_time_sim = float(t_arr[peak_idx])

    peak_current_exp = device.peak_current
    rise_time_exp = device.current_rise_time

    # Relative error on peak current
    peak_current_error = abs(peak_current_sim - peak_current_exp) / max(
        abs(peak_current_exp), 1e-300
    )

    # Timing check: peak time within 50% of experimental rise time
    timing_error = abs(peak_time_sim - rise_time_exp) / max(rise_time_exp, 1e-300)
    timing_ok = timing_error < 0.5

    return {
        "peak_current_error": peak_current_error,
        "peak_current_sim": peak_current_sim,
        "peak_current_exp": peak_current_exp,
        "timing_ok": timing_ok,
    }


def validate_neutron_yield(
    Y_sim: float,
    device_name: str,
) -> dict[str, Any]:
    """Validate simulated neutron yield against experimental data.

    Parameters
    ----------
    Y_sim : float
        Simulated total neutron yield.
    device_name : str
        Key into ``DEVICES``.

    Returns
    -------
    dict
        ``yield_ratio`` : float
            Y_sim / Y_exp.
        ``within_order_magnitude`` : bool
            True if 0.1 < ratio < 10.
        ``yield_sim`` : float
        ``yield_exp`` : float

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    yield_exp = device.neutron_yield
    yield_ratio = Y_sim / max(yield_exp, 1e-300)

    return {
        "yield_ratio": yield_ratio,
        "within_order_magnitude": 0.1 < yield_ratio < 10.0,
        "yield_sim": float(Y_sim),
        "yield_exp": yield_exp,
    }


def device_to_config_dict(device_name: str) -> dict[str, Any]:
    """Convert device parameters to a ``SimulationConfig``-compatible dict.

    Produces a configuration dictionary that can be passed directly to
    ``SimulationConfig(**device_to_config_dict("PF-1000"))`` or written
    to a JSON config file.

    The grid shape is chosen to give approximately 1 mm axial resolution
    (capped at 256 cells per dimension for tractability).

    The initial fill gas density is computed from the ideal gas law at
    room temperature (300 K)::

        n = p / (k_B * T)
        rho = n * m_d

    where *p* is the fill pressure converted from Torr to Pa.

    Parameters
    ----------
    device_name : str
        Key into ``DEVICES``.

    Returns
    -------
    dict
        Configuration dictionary with keys matching ``SimulationConfig``
        fields (``grid_shape``, ``dx``, ``sim_time``, ``circuit``, etc.).

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    # --- Grid resolution ------------------------------------------------
    # Target ~1 mm resolution; cap at 256 cells per dimension
    target_dx = 1e-3  # 1 mm

    # Domain size: slightly larger than cathode diameter x anode length
    domain_r = device.cathode_radius * 1.5
    domain_z = device.anode_length * 1.5

    nx = min(int(np.ceil(2.0 * domain_r / target_dx)), 256)
    ny = min(int(np.ceil(2.0 * domain_r / target_dx)), 256)
    nz = min(int(np.ceil(domain_z / target_dx)), 256)

    # Ensure at least 8 cells per dimension
    nx = max(nx, 8)
    ny = max(ny, 8)
    nz = max(nz, 8)

    # Actual grid spacing from chosen cell count
    dx = 2.0 * domain_r / nx

    # --- Fill gas density from ideal gas law ----------------------------
    # Convert Torr to Pa: 1 Torr = 133.322 Pa
    pressure_Pa = device.fill_pressure_torr * 133.322
    T_room = 300.0  # K

    n_fill = pressure_Pa / (k_B * T_room)  # number density [m^-3]
    rho_fill = n_fill * m_d                 # mass density [kg/m^3]

    # --- Simulation time ------------------------------------------------
    # A few quarter-periods is usually enough to capture peak current + pinch
    sim_time = 4.0 * device.current_rise_time

    return {
        "grid_shape": [nx, ny, nz],
        "dx": dx,
        "sim_time": sim_time,
        "rho0": rho_fill,
        "T0": T_room,
        "ion_mass": m_d,
        "circuit": {
            "C": device.capacitance,
            "V0": device.voltage,
            "L0": device.inductance,
            "R0": device.resistance,
            "anode_radius": device.anode_radius,
            "cathode_radius": device.cathode_radius,
        },
    }
