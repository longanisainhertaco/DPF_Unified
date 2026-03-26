"""Tests for POST /api/thomson endpoint.

Validates the Thomson scattering diagnostic REST endpoint:
- Returns correct response schema
- Spectral output has correct shape and wavelength range
- Physics sanity checks (peak near laser wavelength, Te-dependence)
- Error handling for malformed requests
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from dpf.server.app import app
    return TestClient(app)


def _make_chord(n: int = 4) -> dict:
    """Build a minimal Thomson request body."""
    return {
        "rho": [1e20] * n,          # electron density proxy [m^-3]
        "Te_eV": [500.0] * n,       # 500 eV electrons
        "Ti_eV": [200.0] * n,       # 200 eV ions
        "v_bulk": [0.0] * n,
        "laser_wavelength": 1064e-9,
        "scattering_angle": math.pi / 2,
        "chord_positions": [float(i) * 1e-3 for i in range(n)],
        "n_wavelength_points": 128,
    }


class TestThomsonEndpointSchema:
    """Response schema and HTTP status checks."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post("/api/thomson", json=_make_chord())
        assert resp.status_code == 200

    def test_response_keys_present(self, client: TestClient) -> None:
        resp = client.post("/api/thomson", json=_make_chord())
        data = resp.json()
        for key in ("wavelength_nm", "spectra", "chord_positions_m",
                    "laser_wavelength_nm", "scattering_angle_rad"):
            assert key in data, f"Missing key: {key}"

    def test_wavelength_count(self, client: TestClient) -> None:
        body = _make_chord()
        body["n_wavelength_points"] = 64
        resp = client.post("/api/thomson", json=body)
        data = resp.json()
        assert len(data["wavelength_nm"]) == 64

    def test_spectra_shape(self, client: TestClient) -> None:
        n = 5
        body = _make_chord(n=n)
        body["n_wavelength_points"] = 128
        resp = client.post("/api/thomson", json=body)
        data = resp.json()
        assert len(data["spectra"]) == n
        assert all(len(row) == 128 for row in data["spectra"])

    def test_chord_positions_echoed(self, client: TestClient) -> None:
        body = _make_chord(n=3)
        body["chord_positions"] = [0.01, 0.02, 0.03]
        resp = client.post("/api/thomson", json=body)
        data = resp.json()
        assert data["chord_positions_m"] == pytest.approx([0.01, 0.02, 0.03])

    def test_laser_wavelength_echoed_nm(self, client: TestClient) -> None:
        resp = client.post("/api/thomson", json=_make_chord())
        data = resp.json()
        assert data["laser_wavelength_nm"] == pytest.approx(1064.0, rel=1e-6)

    def test_scattering_angle_echoed(self, client: TestClient) -> None:
        resp = client.post("/api/thomson", json=_make_chord())
        data = resp.json()
        assert data["scattering_angle_rad"] == pytest.approx(math.pi / 2, rel=1e-6)


class TestThomsonEndpointPhysics:
    """Physics sanity checks on the spectral output."""

    def test_spectra_nonnegative(self, client: TestClient) -> None:
        resp = client.post("/api/thomson", json=_make_chord())
        data = resp.json()
        for row in data["spectra"]:
            assert all(v >= 0.0 for v in row), "Spectral power must be non-negative"

    def test_peak_near_laser_wavelength(self, client: TestClient) -> None:
        """Collective peak should sit within 20 nm of the laser wavelength."""
        resp = client.post("/api/thomson", json=_make_chord())
        data = resp.json()
        wl = np.array(data["wavelength_nm"])
        laser_nm = data["laser_wavelength_nm"]
        for i, row in enumerate(data["spectra"]):
            spectrum = np.array(row)
            peak_wl = float(wl[np.argmax(spectrum)])
            assert abs(peak_wl - laser_nm) < 20.0, (
                f"Chord {i}: peak at {peak_wl:.1f} nm, "
                f"expected within 20 nm of {laser_nm:.1f} nm"
            )

    def test_hotter_plasma_broader_spectrum(self, client: TestClient) -> None:
        """Higher Te_eV produces a broader (larger std-dev) spectrum."""
        def _peak_width(Te: float) -> float:
            body = {
                "rho": [1e20],
                "Te_eV": [Te],
                "laser_wavelength": 532e-9,  # green laser — larger Doppler window
                "scattering_angle": math.pi / 2,
                "n_wavelength_points": 256,
            }
            resp = client.post("/api/thomson", json=body)
            data = resp.json()
            wl = np.array(data["wavelength_nm"])
            spec = np.array(data["spectra"][0])
            if spec.max() < 1e-100:
                return 0.0
            mean_wl = float(np.sum(wl * spec) / np.sum(spec))
            return float(np.sqrt(np.sum((wl - mean_wl) ** 2 * spec) / np.sum(spec)))

        width_cold = _peak_width(100.0)
        width_hot = _peak_width(2000.0)
        assert width_hot > width_cold, (
            f"Hotter plasma should have broader spectrum: "
            f"cold={width_cold:.3f} nm, hot={width_hot:.3f} nm"
        )

    def test_default_chord_positions_when_omitted(self, client: TestClient) -> None:
        """chord_positions defaults to [0, 1, 2, ...] when not provided."""
        body = {
            "rho": [1e20, 1e20, 1e20],
            "Te_eV": [500.0, 500.0, 500.0],
            "n_wavelength_points": 32,
        }
        resp = client.post("/api/thomson", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["chord_positions_m"] == [0.0, 1.0, 2.0]


class TestThomsonEndpointErrors:
    """Error handling."""

    def test_mismatched_lengths_returns_422(self, client: TestClient) -> None:
        body = {
            "rho": [1e20, 1e20, 1e20],
            "Te_eV": [500.0, 500.0],  # wrong length
            "Ti_eV": [200.0, 200.0, 200.0],
            "v_bulk": [0.0, 0.0, 0.0],
        }
        resp = client.post("/api/thomson", json=body)
        assert resp.status_code == 422

    def test_empty_rho_returns_200_empty_spectra(self, client: TestClient) -> None:
        """Empty input is accepted; returns empty spectra list."""
        body = {"rho": [], "Te_eV": [], "n_wavelength_points": 32}
        resp = client.post("/api/thomson", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["spectra"] == []
