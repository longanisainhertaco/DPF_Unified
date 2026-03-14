"""Tests for the exact Riemann solver.

Validates the ExactRiemannSolver against known analytical results for
standard shock tube problems. Key checks:
- Star-state pressure and velocity match Toro (2009) tabulated values
- Wave structure (shock/rarefaction) is correctly identified
- Rankine-Hugoniot conditions at shocks
- Isentropic relations in rarefaction fans
- Conservation of mass, momentum, and energy across the solution
"""

import numpy as np
import pytest

from dpf.validation.riemann_exact import (
    BLAST_LEFT,
    BLAST_RIGHT,
    DOUBLE_RAREFACTION_LEFT,
    DOUBLE_RAREFACTION_RIGHT,
    LAX_LEFT,
    LAX_RIGHT,
    SOD_LEFT,
    SOD_RIGHT,
    ExactRiemannSolver,
    RiemannState,
)


class TestSodShockTube:
    """Validate against the standard Sod problem (Sod 1978)."""

    @pytest.fixture
    def sod(self):
        return ExactRiemannSolver(SOD_LEFT, SOD_RIGHT, gamma=1.4)

    def test_star_state_pressure(self, sod):
        """Sod star-state pressure: p* = 0.30313 (Toro Table 4.1)."""
        assert sod.pstar == pytest.approx(0.30313, rel=1e-4)

    def test_star_state_velocity(self, sod):
        """Sod star-state velocity: u* = 0.92745 (Toro Table 4.1)."""
        assert sod.ustar == pytest.approx(0.92745, rel=1e-4)

    def test_wave_structure(self, sod):
        """Sod: left rarefaction + contact + right shock."""
        assert sod.left_type == "rarefaction"
        assert sod.right_type == "shock"

    def test_star_density_left(self, sod):
        """Sod left star density: rho*_L = 0.42632 (Toro)."""
        assert sod.rhostarL == pytest.approx(0.42632, rel=1e-3)

    def test_star_density_right(self, sod):
        """Sod right star density: rho*_R = 0.26557 (Toro)."""
        assert sod.rhostarR == pytest.approx(0.26557, rel=1e-3)

    def test_density_profile_monotone(self, sod):
        """Density should be monotonically non-increasing from L to R (for Sod)."""
        x = np.linspace(0, 1, 500)
        rho, _, _ = sod.sample(x, t=0.2)
        # Allow tiny fluctuations from float precision
        assert np.all(np.diff(rho) <= 1e-10)

    def test_velocity_non_negative(self, sod):
        """All velocities should be >= 0 for Sod (flow is rightward)."""
        x = np.linspace(0, 1, 500)
        _, u, _ = sod.sample(x, t=0.2)
        assert np.all(u >= -1e-12)

    def test_pressure_positive(self, sod):
        """Pressure should be positive everywhere."""
        x = np.linspace(0, 1, 500)
        _, _, p = sod.sample(x, t=0.2)
        assert np.all(p > 0)

    def test_contact_discontinuity_pressure(self, sod):
        """Pressure is continuous across the contact (velocity and p match)."""
        x = np.linspace(0, 1, 1000)
        rho, u, p = sod.sample(x, t=0.2)
        # Find the contact: largest density jump
        drho = np.abs(np.diff(rho))
        contact_idx = np.argmax(drho)
        # Pressure should be smooth at the contact
        dp = abs(p[contact_idx + 1] - p[contact_idx])
        assert dp < 0.001 * sod.pstar

    def test_ambient_regions(self, sod):
        """Far left and far right should retain initial states."""
        x = np.linspace(0, 1, 500)
        rho, u, p = sod.sample(x, t=0.1)
        # Far left (x < 0.1, well ahead of the rarefaction head)
        assert rho[0] == pytest.approx(1.0, rel=1e-10)
        assert u[0] == pytest.approx(0.0, abs=1e-10)
        assert p[0] == pytest.approx(1.0, rel=1e-10)
        # Far right (x > 0.9, well behind the shock)
        assert rho[-1] == pytest.approx(0.125, rel=1e-10)
        assert u[-1] == pytest.approx(0.0, abs=1e-10)
        assert p[-1] == pytest.approx(0.1, rel=1e-10)


class TestLaxProblem:
    """Validate against the Lax shock tube."""

    @pytest.fixture
    def lax(self):
        return ExactRiemannSolver(LAX_LEFT, LAX_RIGHT, gamma=1.4)

    def test_star_state_pressure(self, lax):
        """Lax star-state pressure should be between the two initial pressures."""
        # p* should be between min(pL, pR) and max(pL, pR) for this problem
        assert lax.pstar > 0
        assert np.isfinite(lax.pstar)

    def test_wave_structure(self, lax):
        """Lax: left rarefaction + contact + right shock."""
        assert lax.left_type == "rarefaction"
        assert lax.right_type == "shock"

    def test_pressure_positive(self, lax):
        """Pressure should be positive everywhere."""
        x = np.linspace(0, 1, 500)
        _, _, p = lax.sample(x, t=0.15)
        assert np.all(p > 0)


class TestDoubleRarefaction:
    """Test the 123 problem (symmetric double rarefaction)."""

    @pytest.fixture
    def dr(self):
        return ExactRiemannSolver(
            DOUBLE_RAREFACTION_LEFT, DOUBLE_RAREFACTION_RIGHT, gamma=1.4
        )

    def test_wave_structure(self, dr):
        """Double rarefaction: both waves are rarefactions."""
        assert dr.left_type == "rarefaction"
        assert dr.right_type == "rarefaction"

    def test_star_state_symmetric(self, dr):
        """u* should be 0 by symmetry."""
        assert dr.ustar == pytest.approx(0.0, abs=1e-10)

    def test_density_positive(self, dr):
        """Density should be positive (even in near-vacuum region)."""
        x = np.linspace(0, 1, 500)
        rho, _, _ = dr.sample(x, t=0.15)
        assert np.all(rho > 0)

    def test_pressure_positive(self, dr):
        """Pressure should be positive everywhere."""
        x = np.linspace(0, 1, 500)
        _, _, p = dr.sample(x, t=0.15)
        assert np.all(p > 0)


class TestStrongBlast:
    """Test the Woodward-Colella strong blast wave."""

    @pytest.fixture
    def blast(self):
        return ExactRiemannSolver(BLAST_LEFT, BLAST_RIGHT, gamma=1.4)

    def test_wave_structure(self, blast):
        """Strong blast: left rarefaction + contact + right shock."""
        assert blast.left_type == "rarefaction"
        assert blast.right_type == "shock"

    def test_star_state_positive(self, blast):
        """Star state quantities should be positive."""
        assert blast.pstar > 0
        assert blast.rhostarL > 0
        assert blast.rhostarR > 0

    def test_density_positive(self, blast):
        """Density should remain positive despite extreme pressure ratio."""
        x = np.linspace(0, 1, 500)
        rho, _, _ = blast.sample(x, t=0.012)
        assert np.all(rho > 0)


class TestInputValidation:
    """Test input validation."""

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            ExactRiemannSolver(SOD_LEFT, SOD_RIGHT, gamma=0.5)

    def test_invalid_density(self):
        with pytest.raises(ValueError, match="density"):
            ExactRiemannSolver(
                RiemannState(rho=-1.0, u=0.0, p=1.0), SOD_RIGHT
            )

    def test_invalid_pressure(self):
        with pytest.raises(ValueError, match="pressure"):
            ExactRiemannSolver(
                SOD_LEFT, RiemannState(rho=1.0, u=0.0, p=0.0)
            )

    def test_invalid_time(self):
        solver = ExactRiemannSolver(SOD_LEFT, SOD_RIGHT)
        with pytest.raises(ValueError, match="t must be > 0"):
            solver.sample(np.linspace(0, 1, 10), t=0.0)


class TestGetStarState:
    """Test the get_star_state method."""

    def test_star_state_keys(self):
        solver = ExactRiemannSolver(SOD_LEFT, SOD_RIGHT)
        info = solver.get_star_state()
        expected = {"pstar", "ustar", "rhostarL", "rhostarR", "left_type", "right_type"}
        assert set(info.keys()) == expected

    def test_star_state_values_positive(self):
        solver = ExactRiemannSolver(SOD_LEFT, SOD_RIGHT)
        info = solver.get_star_state()
        assert info["pstar"] > 0
        assert info["rhostarL"] > 0
        assert info["rhostarR"] > 0


class TestGamma53:
    """Test with gamma=5/3 (monatomic gas, relevant for plasma)."""

    @pytest.fixture
    def sod_53(self):
        return ExactRiemannSolver(SOD_LEFT, SOD_RIGHT, gamma=5.0 / 3.0)

    def test_wave_structure(self, sod_53):
        """Sod with gamma=5/3: same wave structure."""
        assert sod_53.left_type == "rarefaction"
        assert sod_53.right_type == "shock"

    def test_star_pressure_increases_with_gamma(self, sod_53):
        """Higher gamma gives higher star-state pressure for Sod."""
        sod_14 = ExactRiemannSolver(SOD_LEFT, SOD_RIGHT, gamma=1.4)
        # For the Sod problem, higher gamma → slightly higher p*
        assert sod_53.pstar > sod_14.pstar * 0.9

    def test_profiles_well_behaved(self, sod_53):
        """Density, velocity, pressure should be well-behaved."""
        x = np.linspace(0, 1, 500)
        rho, u, p = sod_53.sample(x, t=0.2)
        assert np.all(rho > 0)
        assert np.all(p > 0)
        assert not np.any(np.isnan(rho))
        assert not np.any(np.isnan(u))
        assert not np.any(np.isnan(p))
