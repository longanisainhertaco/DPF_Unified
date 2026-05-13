"""Record backend-parity pytest results as Tier-3 evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_PARITY_NODES = [
    "tests/test_cross_backend_parity.py::test_cross_backend_current_nrmse_parity",
]


def _node_test_name(node_id: str) -> str:
    return node_id.rsplit("::", 1)[-1]


def _case_passed(case: ET.Element) -> bool:
    return not any(
        case.find(tag) is not None
        for tag in ("failure", "error", "skipped")
    )


def _status_for_nodes(root: ET.Element, node_ids: list[str]) -> dict[str, object]:
    wanted = {_node_test_name(node_id): node_id for node_id in node_ids}
    cases = {
        str(case.attrib.get("name", "")): case
        for case in root.iter("testcase")
    }
    found: dict[str, dict[str, object]] = {}
    for name, node_id in wanted.items():
        case = cases.get(name)
        if case is None:
            found[node_id] = {
                "found": False,
                "passed": False,
                "classname": "",
                "time_s": None,
            }
            continue
        found[node_id] = {
            "found": True,
            "passed": _case_passed(case),
            "classname": case.attrib.get("classname", ""),
            "time_s": case.attrib.get("time"),
        }
    return {
        "passed": all(item["passed"] for item in found.values()),
        "cases": found,
    }


def evidence_from_junit(
    junitxml: Path,
    *,
    verification_scope: str,
    parity_nodes: list[str] | None = None,
) -> dict[str, object]:
    root = ET.parse(junitxml).getroot()
    parity_status = _status_for_nodes(root, list(parity_nodes or DEFAULT_PARITY_NODES))
    passed = bool(parity_status["passed"])
    return {
        "passed": passed,
        "validation_tier": 3,
        "model_role": "code_verification_backend_parity",
        "evidence_class": "code_numerical_verification",
        "experimental_dpf_validation": False,
        "supports_predictive_scientific_claims": False,
        "supports_high_fidelity_scientific_claims": False,
        "supports_validation_tiers": [3],
        "cannot_substitute_for_validation_tiers": [4, 5],
        "cannot_substitute_for": [
            "same_scope_spatial_dpf_validation",
            "neutron_timing_spectrum_anisotropy_validation",
            "reference_scientific_authority",
        ],
        "authority_label": "BackendParityVerification",
        "verification_scope": verification_scope,
        "source": "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md",
        "source_lines": "1900-1903, 1939-1955",
        "source_basis": {
            "multi_code_verification_context": (
                "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md"
            ),
        },
        "source_line_basis": {
            "multi_code_verification_context": "1900-1903, 1939-1955",
        },
        "metrics": {
            "pytest_backend_parity_gate_passed": passed,
            "gate": "python-cylindrical vs mlx I(t) NRMSE: 1e-6 < NRMSE < 0.10",
        },
        "missing_or_failed_metrics": [] if passed else ["pytest_backend_parity_gate_passed"],
        "details": {
            "junitxml": str(junitxml),
            "backend_pair": ["python-cylindrical", "mlx"],
            "observables": ["I(t)"],
            "parity": parity_status,
            "suite": {
                "tests": root.attrib.get("tests"),
                "failures": root.attrib.get("failures"),
                "errors": root.attrib.get("errors"),
                "skipped": root.attrib.get("skipped"),
                "time_s": root.attrib.get("time"),
            },
        },
        "validity_notes": {
            "claim_scope": (
                "Supports backend parity only for the named pytest gate, "
                "backend pair, observable, and tolerance. It does not validate "
                "the observable against experiment."
            ),
            "authority_boundary": (
                "Backend parity is a numerical consistency label, not "
                "Reference scientific authority."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert backend-parity JUnit XML into Tier-3 evidence."
    )
    parser.add_argument("--junitxml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default="scheduled_tier3_cpu_mhd_numerical_2026_05_09",
        help="same-scope identifier to place on the evidence",
    )
    args = parser.parse_args(argv)

    evidence = evidence_from_junit(args.junitxml, verification_scope=args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": evidence["passed"],
        "metrics": evidence["metrics"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
