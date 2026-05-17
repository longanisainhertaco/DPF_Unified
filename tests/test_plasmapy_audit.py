from __future__ import annotations

import numpy as np

from dpf.collision.spitzer import coulomb_log
from dpf.constants import e, epsilon_0, k_B, m_d, m_e, mu_0
from dpf.first_principles import plasmapy_audit


def test_plasmapy_audit_fails_closed_when_optional_dependency_missing() -> None:
    packet = plasmapy_audit.build_plasmapy_formulary_audit_packet()

    assert packet["dependency"] == "plasmapy"
    assert packet["install_extra"] == "dpf-unified[audit]"
    assert packet["python_requires"] == ">=3.12"
    assert packet["role"] == "optional_community_formula_cross_check_not_source_authority"
    assert packet["source_truth_policy"]["local_knowledge_reference_remains_authority"] is True
    assert packet["source_truth_policy"]["plasmapy_can_promote_claims"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_plasmapy_audit_records_cross_checks_when_dependency_available(monkeypatch) -> None:
    state = {
        "electron_density_m3": 1.0e22,
        "electron_temperature_K": 1.0e6,
        "magnetic_field_T": 10.0,
        "mass_density_kg_m3": 1.0e22 * m_d,
        "ion": "D+",
    }
    local_ln = float(coulomb_log(np.array([1.0e22]), np.array([1.0e6]))[0])
    local_debye = float(np.sqrt(epsilon_0 * k_B * 1.0e6 / (1.0e22 * e * e)))
    local_va = float(10.0 / np.sqrt(mu_0 * 1.0e22 * m_d))
    local_gyro = float(e * 10.0 / m_e)

    monkeypatch.setattr(
        plasmapy_audit,
        "_import_plasmapy_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        plasmapy_audit,
        "_plasmapy_coulomb_log",
        lambda *_args: {"status": "computed", "value": local_ln},
    )
    monkeypatch.setattr(
        plasmapy_audit,
        "_plasmapy_debye_length",
        lambda *_args: {"status": "computed", "value": local_debye},
    )
    monkeypatch.setattr(
        plasmapy_audit,
        "_plasmapy_alfven_speed",
        lambda *_args: {"status": "computed", "value": local_va},
    )
    monkeypatch.setattr(
        plasmapy_audit,
        "_plasmapy_gyrofrequency",
        lambda *_args: {"status": "computed", "value": local_gyro},
    )

    packet = plasmapy_audit.build_plasmapy_formulary_audit_packet(state)

    assert packet["status"] == "community_formula_audit_executed_not_authority"
    assert packet["dependency"] == "plasmapy"
    assert packet["python_requires"] == ">=3.12"
    assert packet["checked_quantity_count"] == 4
    assert packet["error_count"] == 0
    assert packet["quantities"]["debye_length_m"]["relative_difference"] == 0.0
    assert packet["can_support_first_principles_acceptance"] is False
