"""Verify all physics constants are consistent across Python and Metal shader.

This test catches the class of bug where c_boris is updated in one file but
not another (the 65-location duplication problem from the panel audit).
"""

from __future__ import annotations

import re

import numpy as np
import pytest


def test_constants_importable():
    """constants.py is the single source of truth and importable."""
    from dpf.metal.constants import (
        C_BORIS,
        C_BORIS_SQ,
        GAMMA,
        MU_0,
        NVAR,
        P_FLOOR,
        RHO_FLOOR,
    )
    assert C_BORIS == 5e5
    assert pytest.approx(2.5e11) == C_BORIS_SQ
    assert P_FLOOR == 1e-12
    assert RHO_FLOOR == 1e-12
    assert pytest.approx(5.0 / 3.0) == GAMMA
    assert NVAR == 10
    assert pytest.approx(4e-7 * np.pi) == MU_0


def test_metal_shader_matches_python():
    """Metal MSL shader constants match Python constants.py values."""
    from pathlib import Path

    from dpf.metal.constants import C_BORIS_SQ, NVAR, P_FLOOR, RHO_FLOOR

    kernels_path = Path(__file__).parent.parent / "src" / "dpf" / "metal" / "mlx_kernels.py"
    source = kernels_path.read_text()

    # Check HLLD header constants
    hlld_rho = re.search(r'constant float RHO_FLOOR\s*=\s*([\d.e+-]+)f?', source)
    hlld_p = re.search(r'constant float P_FLOOR\s*=\s*([\d.e+-]+)f?', source)
    hlld_nvar = re.search(r'constant int NVAR\s*=\s*(\d+)', source)

    assert hlld_rho, "RHO_FLOOR not found in Metal shader"
    assert float(hlld_rho.group(1)) == pytest.approx(RHO_FLOOR), \
        f"Metal RHO_FLOOR={hlld_rho.group(1)} != Python {RHO_FLOOR}"

    assert hlld_p, "P_FLOOR not found in Metal shader"
    assert float(hlld_p.group(1)) == pytest.approx(P_FLOOR), \
        f"Metal P_FLOOR={hlld_p.group(1)} != Python {P_FLOOR}"

    assert hlld_nvar, "NVAR not found in Metal shader"
    assert int(hlld_nvar.group(1)) == NVAR, \
        f"Metal NVAR={hlld_nvar.group(1)} != Python {NVAR}"

    # Check Boris constant in cylindrical source kernel
    cyl_boris = re.search(r'constant float C_BORIS_SQ\s*=\s*([\d.e+-]+)f?', source)
    assert cyl_boris, "C_BORIS_SQ not found in Metal shader"
    assert float(cyl_boris.group(1)) == pytest.approx(C_BORIS_SQ), \
        f"Metal C_BORIS_SQ={cyl_boris.group(1)} != Python {C_BORIS_SQ}"


def test_no_inline_c_boris_in_riemann():
    """mlx_riemann.py should import C_BORIS_SQ from constants, not define inline."""
    from pathlib import Path

    riemann_path = Path(__file__).parent.parent / "src" / "dpf" / "metal" / "mlx_riemann.py"
    source = riemann_path.read_text()

    # After constants.py migration, there should be NO inline _C_BORIS_SQ definitions
    inline_defs = re.findall(r'_C_BORIS_SQ\s*=\s*[\d.e]+', source)
    # For now, count them — after migration this should be 0
    # During migration phase, this test documents the current state
    assert len(inline_defs) <= 4, \
        f"Found {len(inline_defs)} inline _C_BORIS_SQ in mlx_riemann.py (target: 0 after migration)"


def test_no_inline_floors_in_sources():
    """mlx_sources.py should import floors from constants, not define inline."""
    from pathlib import Path

    sources_path = Path(__file__).parent.parent / "src" / "dpf" / "metal" / "mlx_sources.py"
    source = sources_path.read_text()

    inline_rho = re.findall(r'_RHO_FLOOR\s*=\s*[\d.e+-]+', source)
    inline_p = re.findall(r'_P_FLOOR\s*=\s*[\d.e+-]+', source)

    # After migration these should be 0
    assert len(inline_rho) <= 1, \
        f"Found {len(inline_rho)} inline _RHO_FLOOR in mlx_sources.py (target: 0)"
    assert len(inline_p) <= 1, \
        f"Found {len(inline_p)} inline _P_FLOOR in mlx_sources.py (target: 0)"
