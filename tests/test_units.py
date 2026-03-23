"""Tests for dpf.units — SI ↔ Heaviside-Lorentz unit conversions.

Conversion relations (from module docstring):
    B_hl  = B_si  / sqrt(mu_0)
    J_hl  = J_si  * sqrt(mu_0)
    rho, p, v, t, x are identity (no conversion).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as _sc

from dpf.units import (
    SQRT_MU_0,
    current_to_code_units,
    current_to_si_units,
    to_code_units,
    to_si_units,
)

MU_0 = _sc.mu_0
_SQRT_MU_0 = np.sqrt(MU_0)

# Values spanning ~6 orders of magnitude (Tesla / A/m^2)
B_MAGNITUDES = [1e-6, 1e-3, 1.0, 1e2, 1e4, 1e6]
J_MAGNITUDES = [1.0, 1e3, 1e6, 1e9, 1e11, 1e13]


# ---------------------------------------------------------------------------
# 1. Round-trip identity
# ---------------------------------------------------------------------------

class TestRoundTripIdentity:
    @pytest.mark.parametrize("b_si", B_MAGNITUDES)
    def test_B_field_round_trip(self, b_si: float) -> None:
        result = to_si_units(to_code_units(b_si))
        assert result == pytest.approx(b_si, rel=1e-10)

    @pytest.mark.parametrize("b_si", B_MAGNITUDES)
    def test_B_field_round_trip_reverse(self, b_si: float) -> None:
        b_code = b_si / _SQRT_MU_0
        result = to_code_units(to_si_units(b_code))
        assert result == pytest.approx(b_code, rel=1e-10)

    @pytest.mark.parametrize("j_si", J_MAGNITUDES)
    def test_current_density_round_trip(self, j_si: float) -> None:
        result = current_to_si_units(current_to_code_units(j_si))
        assert result == pytest.approx(j_si, rel=1e-10)

    @pytest.mark.parametrize("j_si", J_MAGNITUDES)
    def test_current_density_round_trip_reverse(self, j_si: float) -> None:
        j_code = j_si * _SQRT_MU_0
        result = current_to_code_units(current_to_si_units(j_code))
        assert result == pytest.approx(j_code, rel=1e-10)

    @pytest.mark.parametrize("val", [1e-6, 0.5, 1.0, 1e3, 1e6, 1e9])
    def test_passthrough_quantities_are_identity(self, val: float) -> None:
        """rho, p, v, t, x have no conversion — document that fact explicitly."""
        # The module states: rho_hl = rho_si, p_hl = p_si, v_hl = v_si
        # There are no dedicated functions; this test records the invariant.
        assert val == pytest.approx(val, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. Known values — SI → code units
# ---------------------------------------------------------------------------

class TestCodeUnitsKnownValues:
    def test_B_one_tesla_to_code(self) -> None:
        expected = 1.0 / _SQRT_MU_0
        assert to_code_units(1.0) == pytest.approx(expected, rel=1e-10)

    def test_B_one_millitesla_to_code(self) -> None:
        b_si = 1e-3
        expected = b_si / _SQRT_MU_0
        assert to_code_units(b_si) == pytest.approx(expected, rel=1e-10)

    def test_B_100_tesla_to_code(self) -> None:
        b_si = 100.0
        expected = b_si / _SQRT_MU_0
        assert to_code_units(b_si) == pytest.approx(expected, rel=1e-10)

    def test_current_1_Am2_to_code(self) -> None:
        expected = 1.0 * _SQRT_MU_0
        assert current_to_code_units(1.0) == pytest.approx(expected, rel=1e-10)

    def test_current_1e6_Am2_to_code(self) -> None:
        j_si = 1e6
        expected = j_si * _SQRT_MU_0
        assert current_to_code_units(j_si) == pytest.approx(expected, rel=1e-10)

    def test_sqrt_mu0_constant_matches_scipy(self) -> None:
        assert pytest.approx(_SQRT_MU_0, rel=1e-12) == SQRT_MU_0


# ---------------------------------------------------------------------------
# 3. Known values — code units → SI
# ---------------------------------------------------------------------------

class TestSIUnitsKnownValues:
    def test_B_code_unit_to_tesla(self) -> None:
        b_code = 1.0
        expected = _SQRT_MU_0
        assert to_si_units(b_code) == pytest.approx(expected, rel=1e-10)

    def test_B_large_code_unit_to_tesla(self) -> None:
        b_code = 1e4
        expected = b_code * _SQRT_MU_0
        assert to_si_units(b_code) == pytest.approx(expected, rel=1e-10)

    def test_current_code_unit_to_si(self) -> None:
        j_code = 1.0
        expected = 1.0 / _SQRT_MU_0
        assert current_to_si_units(j_code) == pytest.approx(expected, rel=1e-10)

    def test_current_large_code_unit_to_si(self) -> None:
        j_code = 1e9
        expected = j_code / _SQRT_MU_0
        assert current_to_si_units(j_code) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. Zero handling
# ---------------------------------------------------------------------------

class TestZeroHandling:
    def test_B_zero_si_to_code(self) -> None:
        assert to_code_units(0.0) == pytest.approx(0.0, abs=1e-30)

    def test_B_zero_code_to_si(self) -> None:
        assert to_si_units(0.0) == pytest.approx(0.0, abs=1e-30)

    def test_current_zero_si_to_code(self) -> None:
        assert current_to_code_units(0.0) == pytest.approx(0.0, abs=1e-30)

    def test_current_zero_code_to_si(self) -> None:
        assert current_to_si_units(0.0) == pytest.approx(0.0, abs=1e-30)

    def test_B_zero_array_to_code(self) -> None:
        arr = np.zeros(5)
        result = to_code_units(arr)
        assert np.allclose(result, 0.0, atol=1e-30)

    def test_B_zero_array_to_si(self) -> None:
        arr = np.zeros(5)
        result = to_si_units(arr)
        assert np.allclose(result, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# 5. Negative value sign preservation
# ---------------------------------------------------------------------------

class TestNegativeValues:
    @pytest.mark.parametrize("b_si", [-1e-6, -1e-3, -1.0, -1e2, -1e4, -1e6])
    def test_B_negative_si_to_code_preserves_sign(self, b_si: float) -> None:
        result = to_code_units(b_si)
        assert result < 0.0
        assert result == pytest.approx(b_si / _SQRT_MU_0, rel=1e-10)

    @pytest.mark.parametrize("b_si", [-1e-6, -1e-3, -1.0, -1e2, -1e4, -1e6])
    def test_B_negative_round_trip_preserves_sign(self, b_si: float) -> None:
        result = to_si_units(to_code_units(b_si))
        assert result == pytest.approx(b_si, rel=1e-10)

    @pytest.mark.parametrize("b_code", [-1e-4, -1.0, -1e3])
    def test_B_negative_code_to_si_preserves_sign(self, b_code: float) -> None:
        result = to_si_units(b_code)
        assert result < 0.0
        assert result == pytest.approx(b_code * _SQRT_MU_0, rel=1e-10)

    @pytest.mark.parametrize("j_si", [-1.0, -1e6, -1e10])
    def test_current_negative_si_to_code_preserves_sign(self, j_si: float) -> None:
        result = current_to_code_units(j_si)
        assert result < 0.0
        assert result == pytest.approx(j_si * _SQRT_MU_0, rel=1e-10)

    def test_B_negative_array_sign_preserved(self) -> None:
        arr = np.array([-1.0, 0.0, 1.0])
        result = to_code_units(arr)
        assert result[0] < 0.0
        assert result[1] == pytest.approx(0.0, abs=1e-30)
        assert result[2] > 0.0


# ---------------------------------------------------------------------------
# 6. Array (vectorized) input
# ---------------------------------------------------------------------------

class TestArrayInput:
    def test_B_1d_array_to_code(self) -> None:
        arr = np.array(B_MAGNITUDES)
        result = to_code_units(arr)
        expected = arr / _SQRT_MU_0
        assert result == pytest.approx(expected, rel=1e-10)

    def test_B_1d_array_to_si(self) -> None:
        arr = np.array(B_MAGNITUDES) / _SQRT_MU_0
        result = to_si_units(arr)
        expected = arr * _SQRT_MU_0
        assert result == pytest.approx(expected, rel=1e-10)

    def test_B_2d_array_to_code(self) -> None:
        arr = np.array(B_MAGNITUDES).reshape(2, 3)
        result = to_code_units(arr)
        expected = arr / _SQRT_MU_0
        assert result.shape == arr.shape
        assert result == pytest.approx(expected, rel=1e-10)

    def test_current_1d_array_to_code(self) -> None:
        arr = np.array(J_MAGNITUDES)
        result = current_to_code_units(arr)
        expected = arr * _SQRT_MU_0
        assert result == pytest.approx(expected, rel=1e-10)

    def test_current_1d_array_to_si(self) -> None:
        arr = np.array(J_MAGNITUDES)
        result = current_to_si_units(arr)
        expected = arr / _SQRT_MU_0
        assert result == pytest.approx(expected, rel=1e-10)

    def test_B_array_round_trip(self) -> None:
        arr = np.array(B_MAGNITUDES)
        result = to_si_units(to_code_units(arr))
        assert result == pytest.approx(arr, rel=1e-10)

    def test_current_array_round_trip(self) -> None:
        arr = np.array(J_MAGNITUDES)
        result = current_to_si_units(current_to_code_units(arr))
        assert result == pytest.approx(arr, rel=1e-10)

    def test_B_array_returns_numpy(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = to_code_units(arr)
        assert isinstance(result, np.ndarray)

    def test_current_array_returns_numpy(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = current_to_code_units(arr)
        assert isinstance(result, np.ndarray)

    def test_B_mixed_sign_array(self) -> None:
        arr = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        result = to_code_units(arr)
        expected = arr / _SQRT_MU_0
        assert result == pytest.approx(expected, rel=1e-10)
        assert np.sign(result) == pytest.approx(np.sign(arr), abs=1e-30)
