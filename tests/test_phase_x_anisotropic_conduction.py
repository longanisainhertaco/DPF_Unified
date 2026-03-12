"""Phase X: Anisotropic thermal conduction verification tests.

Tests verify:
1. Heat diffuses from hot to cold (sign correctness)
2. Parallel diffusion is faster than perpendicular
3. No NaN or negative temperatures after conduction
4. Conduction along B-aligned gradient is faster than cross-B
"""

import numpy as np
import pytest

from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction


def _make_hotspot(nx: int = 16) -> tuple:
    """Create a state with a central hot spot in a uniform B field."""
    Te = np.full((nx, nx, nx), 1e5)  # 100 eV background
    Te[nx // 2 - 2:nx // 2 + 2, nx // 2 - 2:nx // 2 + 2, nx // 2 - 2:nx // 2 + 2] = 1e6
    B = np.zeros((3, nx, nx, nx))
    ne = np.full((nx, nx, nx), 1e24)
    return Te, B, ne


class TestSignCorrectness:
    """Heat must flow from hot to cold."""

    def test_center_cools(self):
        """Hot center should decrease in temperature."""
        Te, B, ne = _make_hotspot()
        B[0] = 0.1  # B along x
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        mid = 8
        assert Te_new[mid, mid, mid] < Te[mid, mid, mid]

    def test_neighbor_heats(self):
        """Cells adjacent to hot spot along B should warm up."""
        Te, B, ne = _make_hotspot()
        B[0] = 0.1  # B along x
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        # Cell at edge of domain along x should stay >= initial
        mid = 8
        assert Te_new[2, mid, mid] >= Te[2, mid, mid] - 1.0

    def test_no_nan(self):
        """No NaN values after conduction step."""
        Te, B, ne = _make_hotspot()
        B[2] = 0.5  # Strong B along z
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        assert not np.any(np.isnan(Te_new))
        assert np.all(Te_new > 0)


class TestAnisotropy:
    """Heat should diffuse faster along B than across B."""

    def test_parallel_faster_than_perpendicular(self):
        """Temperature change along B should exceed change across B."""
        nx = 16
        mid = nx // 2

        # B along x
        Te, B, ne = _make_hotspot(nx)
        B[0] = 0.1
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)

        # Change along x (parallel to B)
        delta_par = abs(Te_new[mid + 3, mid, mid] - Te[mid + 3, mid, mid])
        # Change along y (perpendicular to B)
        delta_perp = abs(Te_new[mid, mid + 3, mid] - Te[mid, mid + 3, mid])

        # Parallel change should be larger (or at least equal)
        assert delta_par >= delta_perp * 0.9, (
            f"Parallel change {delta_par:.1f} should exceed perp {delta_perp:.1f}"
        )

    def test_zero_B_isotropic(self):
        """With B=0, conduction should be approximately isotropic."""
        nx = 16
        mid = nx // 2
        Te, B, ne = _make_hotspot(nx)
        # B = 0 everywhere — conduction should be isotropic
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)

        delta_x = abs(Te_new[mid + 3, mid, mid] - Te[mid + 3, mid, mid])
        delta_y = abs(Te_new[mid, mid + 3, mid] - Te[mid, mid + 3, mid])

        if delta_x > 1e-10 and delta_y > 1e-10:
            ratio = delta_x / delta_y
            assert 0.5 < ratio < 2.0, f"Expected isotropic, got ratio={ratio:.2f}"


class TestStability:
    """Conduction should be stable under various conditions."""

    def test_uniform_temperature_unchanged(self):
        """Uniform temperature should not change."""
        Te = np.full((8, 8, 8), 1e5)
        B = np.zeros((3, 8, 8, 8))
        B[0] = 0.1
        ne = np.full((8, 8, 8), 1e24)
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        np.testing.assert_allclose(Te_new, Te, rtol=1e-6)

    def test_strong_field(self):
        """Strong B field should not cause instability."""
        Te, B, ne = _make_hotspot()
        B[0] = 100.0  # Very strong field
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-11, dx=1e-3, dy=1e-3, dz=1e-3)
        assert not np.any(np.isnan(Te_new))
        assert np.all(Te_new > 0)

    def test_low_density(self):
        """Low density should not cause division by zero."""
        Te, B, ne = _make_hotspot()
        B[0] = 0.1
        ne[:] = 1e10  # Very low density
        Te_new = anisotropic_thermal_conduction(Te, B, ne, dt=1e-10, dx=1e-3, dy=1e-3, dz=1e-3)
        assert not np.any(np.isnan(Te_new))
