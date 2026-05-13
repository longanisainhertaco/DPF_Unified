"""Record finite-volume MHD pytest results as Tier-3 evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from dpf.validation.quality_assessment import mhd_verification_evidence_from_tests


DEFAULT_SOD_NODES = [
    "tests/test_mlx_acceptance.py::TestStandardShockTubes::test_s5_sod_cross_backend_parity",
    "tests/test_mlx_acceptance.py::TestStandardShockTubes::test_s7_sod_convergence",
]
DEFAULT_BRIO_NODES = [
    "tests/test_mlx_acceptance.py::TestStandardShockTubes::test_s6_briowu_compound_waves",
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
    sod_nodes: list[str] | None = None,
    brio_nodes: list[str] | None = None,
) -> dict[str, object]:
    root = ET.parse(junitxml).getroot()
    sod_status = _status_for_nodes(root, list(sod_nodes or DEFAULT_SOD_NODES))
    brio_status = _status_for_nodes(root, list(brio_nodes or DEFAULT_BRIO_NODES))
    evidence = mhd_verification_evidence_from_tests(
        {
            "sod": bool(sod_status["passed"]),
            "brio_wu": bool(brio_status["passed"]),
        },
        source="MLX preview finite-volume shock-tube pytest evidence",
    )
    evidence["verification_scope"] = verification_scope
    evidence["details"] = {
        "backend": "mlx",
        "backend_authority": "preview",
        "junitxml": str(junitxml),
        "sod": sod_status,
        "brio_wu": brio_status,
        "suite": {
            "tests": root.attrib.get("tests"),
            "failures": root.attrib.get("failures"),
            "errors": root.attrib.get("errors"),
            "skipped": root.attrib.get("skipped"),
            "time_s": root.attrib.get("time"),
        },
    }
    evidence["validity_notes"] = {
        "backend_boundary": (
            "This is preview-backend finite-volume code verification from "
            "MLX standalone shock-tube pytest results. Backend parity remains "
            "a separate required packet."
        ),
    }
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert MLX shock-tube JUnit XML into Tier-3 MHD evidence."
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
        "analytic_tests": evidence["analytic_tests"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
