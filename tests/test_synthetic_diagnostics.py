"""Tests for synthetic diagnostics modules (X-ray, nToF, runaway electrons)."""

from __future__ import annotations

import numpy as np


class TestXrayImaging:
    """Tests for xray_imaging.py."""

    def test_bremsstrahlung_emissivity_positive(self):
        from dpf.diagnostics.xray_imaging import bremsstrahlung_emissivity
        eps = bremsstrahlung_emissivity(1e24, 300.0)
        assert eps > 0

    def test_filtered_emissivity_less_than_total(self):
        from dpf.diagnostics.xray_imaging import bremsstrahlung_emissivity, filtered_emissivity
        total = bremsstrahlung_emissivity(1e24, 300.0)
        filtered = filtered_emissivity(1e24, 300.0, E_min_keV=1.0, E_max_keV=5.0)
        assert filtered < total

    def test_xray_image_shape(self):
        from dpf.diagnostics.xray_imaging import synthetic_xray_image
        nr, nz = 16, 32
        ne = np.full((nr, nz), 1e24)
        Te = np.full((nr, nz), 300.0)
        r = np.linspace(0.001, 0.01, nr)
        img = synthetic_xray_image(ne, Te, r, dr=0.001)
        assert img.shape == (nr, nz)
        assert np.all(img >= 0)

    def test_bdot_probe_returns_components(self):
        from dpf.diagnostics.xray_imaging import synthetic_bdot_probe
        B = np.zeros((3, 8, 16))
        B[1] = 1.0  # uniform Bz
        r = np.linspace(0.001, 0.01, 8)
        z = np.linspace(0, 0.1, 16)
        result = synthetic_bdot_probe(B, probe_r=0.005, probe_z=0.05, r_cell=r, z_cell=z)
        assert "Br" in result and "Bz" in result and "Bt" in result
        assert abs(result["Bz"] - 1.0) < 0.1

    def test_radiating_pinch_geometry_from_image(self):
        from dpf.diagnostics.xray_imaging import radiating_pinch_geometry_from_image
        y = np.linspace(0.0, 0.006, 13)
        z = np.linspace(0.0, 0.10, 101)
        image = np.zeros((len(y), len(z)))
        image[np.ix_(y <= 0.0025, (z >= 0.02) & (z <= 0.07))] = 10.0

        geometry = radiating_pinch_geometry_from_image(image, y, z)

        assert geometry["has_radiating_region"] is True
        assert abs(geometry["diameter_mm"] - 5.0) < 0.2
        assert abs(geometry["length_cm"] - 5.0) < 0.2
        assert geometry["diagnostic_role"] == (
            "density_proxy_bremsstrahlung_spatial_geometry"
        )

    def test_radiating_pinch_geometry_empty_image(self):
        from dpf.diagnostics.xray_imaging import radiating_pinch_geometry_from_image
        y = np.linspace(0.0, 0.006, 13)
        z = np.linspace(0.0, 0.10, 101)
        geometry = radiating_pinch_geometry_from_image(np.zeros((13, 101)), y, z)
        assert geometry["has_radiating_region"] is False


class TestNeutronToF:
    """Tests for neutron_tof.py."""

    def test_thermonuclear_centered_at_245(self):
        from dpf.diagnostics.neutron_tof import thermonuclear_spectrum
        E = thermonuclear_spectrum(n_neutrons=10000, Ti_eV=1000.0)
        assert abs(np.mean(E) / 1e6 - 2.45) < 0.1

    def test_beam_target_shifted(self):
        from dpf.diagnostics.neutron_tof import beam_target_spectrum, thermonuclear_spectrum
        E_thermo = np.mean(thermonuclear_spectrum(10000, 1000.0))
        E_bt = np.mean(beam_target_spectrum(10000, E_beam_eV=100000, theta_det=0.0))
        assert E_bt > E_thermo  # forward beam shifts energy up

    def test_combined_spectrum_shape(self):
        from dpf.diagnostics.neutron_tof import combined_tof_spectrum
        E, counts = combined_tof_spectrum(Y_thermo=1e10, Y_bt=1e11)
        assert len(E) == 100
        assert len(counts) == 100
        assert np.all(counts >= 0)


class TestRunawayElectrons:
    """Tests for runaway_electrons.py."""

    def test_dreicer_field_positive(self):
        from dpf.diagnostics.runaway_electrons import dreicer_field
        Ed = dreicer_field(1e24, 1e6)
        assert Ed > 0

    def test_dreicer_scales_with_density(self):
        from dpf.diagnostics.runaway_electrons import dreicer_field
        Ed_low = dreicer_field(1e23, 1e6)
        Ed_high = dreicer_field(1e24, 1e6)
        assert Ed_high > Ed_low

    def test_runaway_fraction_zero_below_threshold(self):
        from dpf.diagnostics.runaway_electrons import dreicer_field, runaway_fraction
        Ed = dreicer_field(1e24, 1e6)
        f = runaway_fraction(0.01 * Ed, Ed)  # well below threshold
        assert f == 0.0

    def test_runaway_fraction_nonzero_above_threshold(self):
        from dpf.diagnostics.runaway_electrons import dreicer_field, runaway_fraction
        Ed = dreicer_field(1e24, 1e6)
        f = runaway_fraction(2.0 * Ed, Ed)  # above threshold
        assert f > 0.0
