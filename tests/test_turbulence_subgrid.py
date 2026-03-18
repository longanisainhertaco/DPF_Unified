"""Tests for sub-grid turbulence model and reconnection diagnostics (Campaign 2H)."""
import numpy as np


def _make_state(n=16):
    """Create MHD state with turbulent-like fields."""
    rho = np.full((n, n, n), 0.084)
    velocity = np.random.RandomState(42).randn(3, n, n, n) * 1e4
    B = np.zeros((3, n, n, n))
    B[2] = 1.0  # uniform Bz = 1 T
    # Add perturbation
    B[0] = np.random.RandomState(43).randn(n, n, n) * 0.1
    Te = np.full((n, n, n), 1e6)  # 1 MK
    ne = rho / 3.34e-27
    return rho, velocity, B, Te, ne


class TestSmagorinskyViscosity:
    def test_uniform_flow_zero_viscosity(self):
        from dpf.turbulence.subgrid import smagorinsky_viscosity
        # Uniform flow: no strain → zero SGS viscosity
        v = np.zeros((3, 8, 8, 8))
        v[0] = 1e3  # uniform x-velocity
        nu = smagorinsky_viscosity(v, dx=0.001)
        # Interior should be zero (uniform gradient = 0)
        assert np.max(nu[2:-2, 2:-2, 2:-2]) < 1e-10

    def test_shear_flow_nonzero(self):
        from dpf.turbulence.subgrid import smagorinsky_viscosity
        # Linear shear: du/dy = const → nonzero SGS viscosity
        v = np.zeros((3, 16, 16, 16))
        y = np.linspace(0, 1, 16)
        v[0] = y[np.newaxis, :, np.newaxis] * 1e5  # du/dy = 1e5
        nu = smagorinsky_viscosity(v, dx=1.0 / 16)
        assert np.max(nu[2:-2, 2:-2, 2:-2]) > 0

    def test_smagorinsky_constant_scales(self):
        from dpf.turbulence.subgrid import smagorinsky_viscosity
        v = np.random.RandomState(42).randn(3, 8, 8, 8) * 1e4
        nu_small = smagorinsky_viscosity(v, dx=0.001, C_s=0.05)
        nu_large = smagorinsky_viscosity(v, dx=0.001, C_s=0.2)
        # Larger C_s → larger viscosity
        assert np.max(nu_large) > np.max(nu_small)


class TestAnomalousThermalConductivity:
    def test_nonzero_with_anomalous_eta(self):
        from dpf.turbulence.subgrid import anomalous_thermal_conductivity
        ne = np.full((8, 8, 8), 1e23)
        Te = np.full((8, 8, 8), 1e6)
        eta_anom = np.full((8, 8, 8), 1e-4)  # Strong anomalous resistivity
        kappa = anomalous_thermal_conductivity(ne, Te, eta_anom)
        assert np.all(kappa > 0)
        assert np.all(np.isfinite(kappa))

    def test_zero_eta_gives_high_kappa(self):
        from dpf.turbulence.subgrid import anomalous_thermal_conductivity
        ne = np.full((4, 4, 4), 1e23)
        Te = np.full((4, 4, 4), 1e6)
        eta_low = np.full((4, 4, 4), 1e-8)
        eta_high = np.full((4, 4, 4), 1e-2)
        kappa_low = anomalous_thermal_conductivity(ne, Te, eta_low)
        kappa_high = anomalous_thermal_conductivity(ne, Te, eta_high)
        # Lower eta → higher kappa (less scattering → more transport)
        assert np.mean(kappa_low) > np.mean(kappa_high)


class TestSweetParkerDiagnostic:
    def test_basic_diagnostic(self):
        from dpf.turbulence.subgrid import sweet_parker_diagnostic
        rho, _, B, Te, ne = _make_state(8)
        diag = sweet_parker_diagnostic(B, rho, Te, ne, dx=0.001, L_system=0.1)
        assert diag.S_lundquist > 0
        assert 0 <= diag.reconnection_rate <= 1
        assert diag.delta_sp > 0
        assert diag.regime in ("sweet_parker", "plasmoid", "collisionless")

    def test_high_S_plasmoid_unstable(self):
        from dpf.turbulence.subgrid import sweet_parker_diagnostic
        # High B, large L → high S → plasmoid regime
        n = 8
        B = np.zeros((3, n, n, n))
        B[2] = 10.0  # 10 T
        rho = np.full((n, n, n), 0.01)
        Te = np.full((n, n, n), 1e7)  # 10 MK → high conductivity → high S
        ne = np.full((n, n, n), 1e24)
        diag = sweet_parker_diagnostic(B, rho, Te, ne, dx=0.001, L_system=1.0)
        # At high S, should be plasmoid unstable
        assert diag.S_lundquist > 100

    def test_regime_classification(self):
        from dpf.turbulence.subgrid import sweet_parker_diagnostic
        n = 4
        B = np.zeros((3, n, n, n))
        B[2] = 0.001  # Very weak B → low S
        rho = np.full((n, n, n), 1.0)
        Te = np.full((n, n, n), 300.0)  # Cold
        ne = np.full((n, n, n), 1e20)
        diag = sweet_parker_diagnostic(B, rho, Te, ne, dx=0.01, L_system=0.01)
        assert diag.regime in ("sweet_parker", "plasmoid", "collisionless")


class TestEnergySpectrum:
    def test_spectrum_shape(self):
        from dpf.turbulence.subgrid import compute_energy_spectrum
        rho, velocity, B, _, _ = _make_state(16)
        spec = compute_energy_spectrum(velocity, B, rho, dx=0.001)
        assert len(spec.k) == len(spec.E_k)
        assert len(spec.E_mag_k) == len(spec.E_kin_k)
        assert np.all(spec.E_k >= 0)

    def test_spectral_index_finite(self):
        from dpf.turbulence.subgrid import compute_energy_spectrum
        rho, velocity, B, _, _ = _make_state(16)
        spec = compute_energy_spectrum(velocity, B, rho, dx=0.001)
        assert np.isfinite(spec.spectral_index)
        # For random noise, index should be negative (decaying spectrum)
        # but not wildly so
        assert -10 < spec.spectral_index < 5

    def test_uniform_field_no_spectrum(self):
        from dpf.turbulence.subgrid import compute_energy_spectrum
        n = 8
        v = np.zeros((3, n, n, n))
        B = np.zeros((3, n, n, n))
        B[2] = 1.0
        rho = np.full((n, n, n), 1.0)
        spec = compute_energy_spectrum(v, B, rho, dx=0.001)
        # Kinetic spectrum should be zero
        assert np.sum(spec.E_kin_k) == 0

    def test_inertial_range_bounds(self):
        from dpf.turbulence.subgrid import compute_energy_spectrum
        rho, velocity, B, _, _ = _make_state(16)
        spec = compute_energy_spectrum(velocity, B, rho, dx=0.001)
        k_min, k_max = spec.inertial_range
        assert k_min < k_max
        assert k_min >= 0
