"""API metadata for units, dimensions, and schema hints."""

from __future__ import annotations

from typing import Any


def api_units_metadata() -> dict[str, Any]:
    """Return canonical API units and dimension hints for client display."""

    return {
        "time_base": {"field": "time", "units": "s", "display_units": ["ns", "us", "ms", "s"]},
        "scalars": {
            "step": {"units": "count", "dimension": "iteration"},
            "time": {"units": "s", "dimension": "time"},
            "current": {"units": "A", "dimension": "electric_current"},
            "voltage": {"units": "V", "dimension": "electric_potential"},
            "energy_conservation": {"units": "ratio", "dimension": "dimensionless"},
            "max_Te": {"units": "K", "dimension": "temperature"},
            "max_rho": {"units": "kg/m^3", "dimension": "mass_density"},
            "total_radiated_energy": {"units": "J", "dimension": "energy"},
        },
        "fields": {
            "rho": {"units": "kg/m^3", "dimension": "mass_density"},
            "B": {"units": "T", "dimension": "magnetic_flux_density"},
            "Te": {"units": "K", "dimension": "temperature"},
            "Ti": {"units": "K", "dimension": "temperature"},
            "velocity": {"units": "m/s", "dimension": "velocity"},
            "pressure": {"units": "Pa", "dimension": "pressure"},
        },
        "authority": {
            "validation_status": {"units": "enum", "dimension": "validation_state"},
            "result_classification": {"units": "enum", "dimension": "claim_authority"},
            "source_blockers": {"units": "list", "dimension": "readiness_blockers"},
        },
    }
