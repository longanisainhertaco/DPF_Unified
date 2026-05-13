from __future__ import annotations

from dpf.server.metadata import api_units_metadata


def test_api_units_metadata_exposes_scalars_fields_and_authority() -> None:
    metadata = api_units_metadata()

    assert metadata["time_base"]["units"] == "s"
    assert metadata["scalars"]["current"]["units"] == "A"
    assert metadata["fields"]["rho"]["dimension"] == "mass_density"
    assert metadata["authority"]["result_classification"]["dimension"] == "claim_authority"


def test_units_metadata_endpoint() -> None:
    from fastapi.testclient import TestClient

    from dpf.server.app import app

    response = TestClient(app).get("/api/metadata/units")

    assert response.status_code == 200
    payload = response.json()
    assert payload["scalars"]["voltage"]["units"] == "V"
    assert payload["fields"]["B"]["units"] == "T"
