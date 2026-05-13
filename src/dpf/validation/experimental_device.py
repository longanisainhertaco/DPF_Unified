"""ExperimentalDevice dataclass — published experimental data for a DPF device."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    # Published Lee model fitting parameters (from RADPF calibration)
    lee_fc: float = 0.0       # Axial current fraction (published Lee model fit)
    lee_fm: float = 0.0       # Axial mass fraction (published Lee model fit)
    lee_fmr: float = 0.0      # Radial mass fraction (published Lee model fit)
    lee_fcr: float = 0.0      # Radial current fraction (published Lee model fit)
    lee_fcr2: float | None = None  # Second-step radial current fraction, when published
    lee_radial_transition_time: float | None = None  # Transition time for two-step radial fit [s]
    lee_reference: str = ""   # Reference for Lee model fit parameters
    # Experimental uncertainties (1-sigma, relative)
    # Following GUM (JCGM 100:2008) and ASME V&V 20-2009 uncertainty framework.
    peak_current_uncertainty: float = 0.0   # Relative uncertainty on peak current
    rise_time_uncertainty: float = 0.0      # Relative uncertainty on rise time
    neutron_yield_uncertainty: float = 0.0  # Relative uncertainty on neutron yield
    # Digitized waveform data (optional)
    waveform_t: np.ndarray | None = None    # Time array [s]
    waveform_I: np.ndarray | None = None    # Current array [A]
    # Waveform amplitude uncertainty (1-sigma, relative).
    # Per GUM (JCGM 100:2008), each component identified by physical source:
    #   - "digitization": genuine digitization error from reading a published figure
    #   - "reconstruction": model-based reconstruction error (physics scaling, RLC fit)
    #   - "": unknown/unset
    waveform_amplitude_uncertainty: float = 0.0     # Amplitude uncertainty (1-sigma, relative)
    waveform_time_uncertainty: float = 0.0          # Temporal uncertainty (1-sigma, relative)
    waveform_uncertainty_type: str = ""             # "digitization" or "reconstruction" per GUM
    # Whether peak_current_uncertainty already incorporates shot-to-shot spread.
    # True for devices where I_peak uncertainty was derived FROM shot spread range
    # (e.g. PF-1000-16kV: range 1.1-1.3 MA → 10%). When True, ASME budget should
    # NOT add separate u_shot_to_shot to avoid double-counting per GUM.
    peak_current_from_shot_spread: bool = False
    # Crowbar switch resistance [Ohm] — physical arc resistance of the
    # crowbar spark gap.  Default 0.0 for backward compatibility.
    # PF-1000: ~1-3 mOhm (spark gap arc, PhD Debate #30 Finding 4).
    crowbar_resistance: float = 0.0
    # Waveform provenance: "measured" (digitized from published oscillogram),
    # "reconstructed" (generated from RLC parameters or physics scaling),
    # or "" (unknown/unset)
    waveform_provenance: str = ""
    # KnowledgeReference verification status for the waveform trace itself.
    # This is separate from ``kr_status`` because a device parameter table can be
    # KR-verified while its digitized waveform comes from an external archive.
    waveform_kr_status: str = "unverified"
    # Measurement provenance note
    measurement_notes: str = ""
    # Data reliability: "measured" (direct Rogowski/probe measurement),
    # "reference_only" (model output or unreliable source — exclude from validation claims)
    reliability: str = "measured"
    reliability_note: str = ""
    # KnowledgeReference verification status.
    # "verified"      — all parameters sourced from a file under KnowledgeReference/
    # "reference_only"— parameters cannot be sourced from KR (wrong variant, missing paper)
    # "unverified"    — KR source exists but parameters not yet cross-checked line-by-line
    kr_status: str = "unverified"
