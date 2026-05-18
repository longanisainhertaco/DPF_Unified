#!/usr/bin/env python3
"""Audit module-by-module source-truth vetting for the first-principles track.

This is a read-only, non-validating audit. It checks every ``src/dpf`` Python
module, resolves local source references against
``docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json``, and fails closed for active
physics modules that still lack source-truth grounding.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
DPF_ROOT = SRC_ROOT / "dpf"
INDEX_PATH = ROOT / "docs" / "FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json"

ACTIVE_ENTRY_MODULES = (
    "dpf.first_principles",
    "dpf.fields",
)

PHYSICS_PREFIXES = (
    "src/dpf/atomic/",
    "src/dpf/circuit/",
    "src/dpf/collision/",
    "src/dpf/diagnostics/",
    "src/dpf/engine/",
    "src/dpf/experimental/",
    "src/dpf/fields/",
    "src/dpf/first_principles/",
    "src/dpf/fluid/",
    "src/dpf/geometry/",
    "src/dpf/jax/",
    "src/dpf/kinetic/",
    "src/dpf/metal/",
    "src/dpf/radiation/",
    "src/dpf/sheath/",
    "src/dpf/turbulence/",
)
NONPHYSICS_PREFIXES = (
    "src/dpf/ai/",
    "src/dpf/athena_wrapper/",
    "src/dpf/athenak_wrapper/",
    "src/dpf/benchmarks/",
    "src/dpf/cli/",
    "src/dpf/core/",
    "src/dpf/io/",
    "src/dpf/project/",
    "src/dpf/release/",
    "src/dpf/security/",
    "src/dpf/server/",
    "src/dpf/testing/",
    "src/dpf/verification/",
)
VALIDATION_WORKFLOW_PREFIX = "src/dpf/validation/"
ROOT_INFRASTRUCTURE_FILES = {
    "src/dpf/__init__.py",
    "src/dpf/__main__.py",
    "src/dpf/config.py",
    "src/dpf/multi_shot.py",
    "src/dpf/units.py",
}
STANDARDS_SCOPED_FILES = {
    "src/dpf/constants.py",
}
NONPHYSICS_INFRASTRUCTURE_FILES = {
    "src/dpf/diagnostics/checkpoint.py",
    "src/dpf/diagnostics/evidence_manifest.py",
    "src/dpf/diagnostics/hdf5_writer.py",
    "src/dpf/diagnostics/test_lanes.py",
    "src/dpf/engine/athena_step.py",
    "src/dpf/engine/backend_capabilities.py",
    "src/dpf/engine/backend_dispatch.py",
    "src/dpf/engine/core.py",
    "src/dpf/engine/memory_preflight.py",
    "src/dpf/engine/runtime_telemetry.py",
    "src/dpf/engine/state_management.py",
}
INACTIVE_NUMERICAL_UTILITY_FILES = {
    "src/dpf/experimental/static_refinement.py",
    "src/dpf/fluid/implicit_diffusion.py",
    "src/dpf/fluid/super_time_step.py",
    "src/dpf/geometry/cylindrical.py",
    "src/dpf/kinetic/hybrid.py",
    "src/dpf/kinetic/initialize.py",
    "src/dpf/kinetic/manager.py",
    "src/dpf/metal/_dual_energy.py",
    "src/dpf/metal/_riemann_constants.py",
    "src/dpf/metal/_riemann_nan_safety.py",
    "src/dpf/metal/_riemann_primitives.py",
    "src/dpf/metal/_riemann_reconstruction.py",
    "src/dpf/metal/_riemann_solvers.py",
    "src/dpf/metal/_utils.py",
    "src/dpf/metal/constants.py",
    "src/dpf/metal/device.py",
    "src/dpf/metal/floor_telemetry.py",
    "src/dpf/metal/metal_riemann.py",
    "src/dpf/metal/metal_stencil.py",
    "src/dpf/metal/mlx_amr.py",
    "src/dpf/metal/mlx_ct.py",
    "src/dpf/metal/mlx_device.py",
    "src/dpf/metal/mlx_divb.py",
    "src/dpf/metal/mlx_fused_flux.py",
    "src/dpf/metal/mlx_gradient_patches.py",
    "src/dpf/metal/mlx_grid.py",
    "src/dpf/metal/mlx_kernels.py",
    "src/dpf/metal/mlx_operator_split.py",
    "src/dpf/metal/mlx_primitives.py",
    "src/dpf/metal/mlx_reconstruction.py",
    "src/dpf/metal/mlx_riemann.py",
    "src/dpf/metal/mlx_state.py",
    "src/dpf/metal/mlx_sts.py",
    "src/dpf/metal/mlx_sts_operators.py",
    "src/dpf/metal/mlx_surrogate.py",
    "src/dpf/metal/mlx_timestepper.py",
}
BASELINE_OR_FAIL_CLOSED_FILES = {
    "src/dpf/atomic/ablation.py",
    "src/dpf/circuit/coupler.py",
    "src/dpf/diagnostics/pb11_yield.py",
    "src/dpf/diagnostics/scaling_laws.py",
    "src/dpf/diagnostics/yield_tracker.py",
    "src/dpf/experimental/civ_breakdown.py",
    "src/dpf/fluid/nernst.py",
    "src/dpf/jax/__init__.py",
    "src/dpf/jax/lee_model.py",
    "src/dpf/metal/mlx_coupling.py",
    "src/dpf/metal/mlx_engine.py",
    "src/dpf/metal/mlx_snowplow.py",
    "src/dpf/radiation/line_radiation.py",
    "src/dpf/radiation/qmf_suppression.py",
    "src/dpf/radiation/transport.py",
    "src/dpf/sheath/bohm.py",
    "src/dpf/turbulence/anomalous.py",
}
INACTIVE_DIAGNOSTIC_SURFACE_FILES = {
    "src/dpf/diagnostics/beam_tracker.py",
    "src/dpf/diagnostics/derived.py",
    "src/dpf/diagnostics/energy_balance.py",
    "src/dpf/diagnostics/filamentation.py",
    "src/dpf/diagnostics/interferometry.py",
    "src/dpf/diagnostics/neutron_tof.py",
    "src/dpf/diagnostics/neutron_yield.py",
    "src/dpf/diagnostics/pease_braginskii.py",
    "src/dpf/diagnostics/plasma_regime.py",
    "src/dpf/diagnostics/plasmoid.py",
    "src/dpf/diagnostics/regime_classifier.py",
    "src/dpf/diagnostics/runaway_electrons.py",
    "src/dpf/diagnostics/shear_stabilization.py",
    "src/dpf/diagnostics/thomson_scattering.py",
    "src/dpf/diagnostics/xray_imaging.py",
}
INACTIVE_LEGACY_PHYSICS_FILES = {
    "src/dpf/circuit/rlc_solver.py",
    "src/dpf/engine/physics_operators.py",
    "src/dpf/experimental/poloidal_bfield.py",
    "src/dpf/fluid/anisotropic_conduction.py",
    "src/dpf/fluid/eos.py",
    "src/dpf/fluid/ionization.py",
    "src/dpf/fluid/viscosity.py",
    "src/dpf/metal/metal_transport.py",
    "src/dpf/metal/mlx_bc.py",
    "src/dpf/metal/mlx_circuit.py",
    "src/dpf/metal/mlx_eos.py",
    "src/dpf/metal/mlx_implicit_mhd.py",
    "src/dpf/metal/mlx_line_radiation.py",
    "src/dpf/metal/mlx_solver.py",
    "src/dpf/metal/mlx_sources.py",
    "src/dpf/metal/mlx_species.py",
    "src/dpf/metal/mlx_transport.py",
    "src/dpf/metal/mlx_viscosity.py",
    "src/dpf/radiation/improved_radiation.py",
    "src/dpf/turbulence/subgrid.py",
}
BASELINE_PATTERNS = (
    "lee_model",
    "snowplow",
    "scaling_laws",
    "surrogate",
    "radpf",
)
SOURCE_CONSTANT_PATHS = {
    "HYBRID_PIC_3D_SOURCE": (
        "KnowledgeReference/"
        "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
    ),
}
SOURCE_REFERENCE_MARKERS = (
    "SOURCE_REFS",
    "SOURCE_REFERENCES",
    "source_references",
    "source_index_references",
    "SourceIndexReference",
    "source_lines",
    "KnowledgeReference/",
)
CLAIM_TERMS = (
    "first_principles",
    "first-principles",
    "accepted",
    "validated",
    "validation",
)
FAIL_CLOSED_TERMS = (
    "not_validation",
    "not validation",
    "non-promoting",
    "non_promoting",
    "blocked",
    "candidate",
    "engineering",
    "cannot support",
    "can_support_first_principles_acceptance",
)

KR_PATH_RE = re.compile(r"KnowledgeReference/[^\s'\"),\]}]+?\.(?:md|json)")
DOC_PATH_RE = re.compile(
    r"docs/FIRST_PRINCIPLES_[A-Za-z0-9_./-]+\.(?:json|md)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=datetime.now(UTC).strftime("%Y_%m_%d"),
        help="Date slug for generated docs, default UTC today.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if active physics source-vetting blockers remain.",
    )
    args = parser.parse_args()

    payload = build_module_vetting_report(date_slug=args.date)
    output_json = (
        ROOT / "docs" / f"FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_{args.date}.json"
    )
    output_md = (
        ROOT / "docs" / f"FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_{args.date}.md"
    )
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown_report(payload))
    print(
        json.dumps(
            {
                "active_physics_unvetted_count": payload["summary"][
                    "active_physics_unvetted_count"
                ],
                "missing_source_reference_count": payload["summary"][
                    "missing_source_reference_count"
                ],
                "json": str(output_json.relative_to(ROOT)),
                "markdown": str(output_md.relative_to(ROOT)),
                "strict_passed": payload["strict_passed"],
                "total_modules": payload["summary"]["total_modules"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and not payload["strict_passed"]:
        return 1
    return 0


def build_module_vetting_report(*, date_slug: str) -> dict[str, Any]:
    index = _read_json(INDEX_PATH)
    indexed_paths, records_by_path = _indexed_source_paths(index)
    module_paths = sorted(DPF_ROOT.rglob("*.py"))
    module_map = {_module_name(path): path for path in module_paths}
    active_modules = _active_import_closure(module_map)

    modules = [
        _module_record(
            path=path,
            active_modules=active_modules,
            indexed_paths=indexed_paths,
            records_by_path=records_by_path,
        )
        for path in module_paths
    ]
    modules = sorted(modules, key=lambda item: item["path"])
    blockers = _blockers(modules)
    summary = _summary(modules, blockers)
    return {
        "date": date_slug,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "first_principles_module_source_vetting",
        "authority_policy": (
            "KnowledgeReference plus explicitly user-verified staged sources only; "
            "module source-vetting is non-validating and does not invoke the "
            "dpf-validation workflow"
        ),
        "source_index": {
            "path": str(INDEX_PATH.relative_to(ROOT)),
            "knowledge_reference_file_count": index.get("stats", {}).get(
                "knowledge_reference_file_count"
            ),
            "knowledge_reference_document_count": index.get("stats", {}).get(
                "knowledge_reference_document_count"
            ),
            "capability_keys": sorted(index.get("capability_index", {}).keys()),
        },
        "strict_passed": not blockers["active_physics_unvetted"]
        and not blockers["inactive_physics_unvetted"]
        and not blockers["missing_source_references"]
        and not blockers["unsafe_claim_surfaces"],
        "summary": summary,
        "blockers": blockers,
        "modules": modules,
        "next_required_actions": _next_required_actions(blockers),
    }


def _module_record(
    *,
    path: Path,
    active_modules: set[str],
    indexed_paths: set[str],
    records_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rel = str(path.relative_to(ROOT))
    module_name = _module_name(path)
    text = path.read_text(errors="replace")
    literals = _string_literals(path, text)
    imports = _module_imports(path, text)
    kr_paths = _source_paths_from_strings(literals, text)
    for marker, source_path in SOURCE_CONSTANT_PATHS.items():
        if marker in text:
            kr_paths.add(source_path)
    docs_paths = _doc_paths_from_strings(literals, text)
    missing_source_paths = sorted(path for path in kr_paths if path not in indexed_paths)
    source_records = [
        records_by_path[path]
        for path in sorted(kr_paths)
        if path in records_by_path
    ]
    tags = sorted({str(tag) for record in source_records for tag in record.get("tags", ())})
    verdicts = sorted(
        {
            str(record.get("first_principles_verdict", "unknown"))
            for record in source_records
        }
    )
    is_active = module_name in active_modules
    is_physics = rel in STANDARDS_SCOPED_FILES or rel.startswith(PHYSICS_PREFIXES)
    has_source_hook = bool(kr_paths or docs_paths or _has_source_marker(text))
    claim_without_source = _claim_without_source(text, has_source_hook, is_physics)
    classification = _classification(
        rel=rel,
        text=text,
        is_active=is_active,
        is_physics=is_physics,
        has_source_hook=has_source_hook,
    )
    status = _status_for_classification(
        classification=classification,
        is_active=is_active,
        missing_source_paths=missing_source_paths,
        claim_without_source=claim_without_source,
    )
    return {
        "path": rel,
        "module": module_name,
        "classification": classification,
        "status": status,
        "active_first_principles_closure": is_active,
        "physics_facing": is_physics,
        "source_reference_count": len(kr_paths) + len(docs_paths),
        "knowledge_reference_paths": sorted(kr_paths),
        "docs_reference_paths": sorted(docs_paths),
        "missing_source_paths": missing_source_paths,
        "capability_tags": tags,
        "first_principles_verdicts": verdicts,
        "imports": sorted(imports),
        "claim_without_source": claim_without_source,
        "can_support_first_principles_acceptance": False,
    }


def _classification(
    *,
    rel: str,
    text: str,
    is_active: bool,
    is_physics: bool,
    has_source_hook: bool,
) -> str:
    if rel in STANDARDS_SCOPED_FILES:
        return "standards_scoped_implementation_constants"
    if rel in NONPHYSICS_INFRASTRUCTURE_FILES:
        return "nonphysics_infrastructure"
    if rel in INACTIVE_NUMERICAL_UTILITY_FILES:
        return "inactive_backend_numerical_utility_not_first_principles_authority"
    if rel in BASELINE_OR_FAIL_CLOSED_FILES:
        return "reduced_or_baseline_only_not_first_principles_authority"
    if rel in INACTIVE_DIAGNOSTIC_SURFACE_FILES:
        return "inactive_diagnostic_surface_not_first_principles_authority"
    if rel in INACTIVE_LEGACY_PHYSICS_FILES:
        return "inactive_legacy_physics_not_first_principles_authority"
    if rel.endswith("/__init__.py") or rel == "src/dpf/__init__.py":
        return "package_export_surface"
    if rel.startswith(VALIDATION_WORKFLOW_PREFIX):
        return "validation_workflow_not_active_first_principles"
    if rel in ROOT_INFRASTRUCTURE_FILES or rel.startswith(NONPHYSICS_PREFIXES):
        return "nonphysics_infrastructure"
    if _is_baseline_module(rel, text):
        return "reduced_or_baseline_only_not_first_principles_authority"
    if has_source_hook and is_active and is_physics:
        return "active_physics_source_referenced"
    if has_source_hook and is_physics:
        return "physics_source_referenced"
    if has_source_hook:
        return "source_referenced_nonphysics_or_metadata"
    if is_active and is_physics:
        return "active_physics_needs_source_vetting"
    if is_physics:
        return "physics_needs_source_vetting"
    return "unclassified_needs_module_role_review"


def _status_for_classification(
    *,
    classification: str,
    is_active: bool,
    missing_source_paths: list[str],
    claim_without_source: bool,
) -> str:
    if missing_source_paths:
        return "blocked_missing_source_reference"
    if classification == "standards_scoped_implementation_constants":
        return "source_vetted_by_external_constants_api_not_kr_evidence"
    if classification == "package_export_surface":
        return "module_vetted_nonpromoting"
    if classification == "inactive_backend_numerical_utility_not_first_principles_authority":
        return "module_vetted_nonpromoting"
    if classification == "inactive_diagnostic_surface_not_first_principles_authority":
        return "module_vetted_nonpromoting"
    if classification == "inactive_legacy_physics_not_first_principles_authority":
        return "module_vetted_nonpromoting"
    if "baseline" in classification:
        return "baseline_only_nonpromoting"
    if claim_without_source:
        return "blocked_claim_surface_without_source_hook"
    if classification == "active_physics_needs_source_vetting":
        return "blocked_active_physics_source_vetting_required"
    if classification == "physics_needs_source_vetting":
        return "blocked_inactive_physics_source_vetting_required"
    if classification == "unclassified_needs_module_role_review":
        return "blocked_module_role_review_required"
    if classification == "validation_workflow_not_active_first_principles":
        return "out_of_active_first_principles_path"
    if is_active:
        return "active_module_vetted_nonpromoting"
    return "module_vetted_nonpromoting"


def _blockers(modules: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "active_physics_unvetted": [
            _slim_module(item)
            for item in modules
            if item["status"] == "blocked_active_physics_source_vetting_required"
        ],
        "inactive_physics_unvetted": [
            _slim_module(item)
            for item in modules
            if item["status"] == "blocked_inactive_physics_source_vetting_required"
        ],
        "missing_source_references": [
            {
                "path": item["path"],
                "module": item["module"],
                "missing_source_paths": item["missing_source_paths"],
            }
            for item in modules
            if item["missing_source_paths"]
        ],
        "unsafe_claim_surfaces": [
            _slim_module(item)
            for item in modules
            if item["status"] == "blocked_claim_surface_without_source_hook"
        ],
        "module_role_review_required": [
            _slim_module(item)
            for item in modules
            if item["status"] == "blocked_module_role_review_required"
        ],
    }


def _summary(
    modules: list[Mapping[str, Any]],
    blockers: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    classifications = Counter(str(item["classification"]) for item in modules)
    statuses = Counter(str(item["status"]) for item in modules)
    return {
        "total_modules": len(modules),
        "active_first_principles_closure_count": sum(
            1 for item in modules if item["active_first_principles_closure"]
        ),
        "physics_facing_count": sum(1 for item in modules if item["physics_facing"]),
        "source_referenced_count": sum(
            1 for item in modules if int(item["source_reference_count"]) > 0
        ),
        "active_physics_unvetted_count": len(blockers["active_physics_unvetted"]),
        "inactive_physics_unvetted_count": len(blockers["inactive_physics_unvetted"]),
        "missing_source_reference_count": len(blockers["missing_source_references"]),
        "unsafe_claim_surface_count": len(blockers["unsafe_claim_surfaces"]),
        "module_role_review_required_count": len(blockers["module_role_review_required"]),
        "classification_counts": dict(sorted(classifications.items())),
        "status_counts": dict(sorted(statuses.items())),
    }


def _next_required_actions(blockers: Mapping[str, list[Mapping[str, Any]]]) -> list[str]:
    actions: list[str] = []
    if blockers["missing_source_references"]:
        actions.append(
            "repair missing KnowledgeReference paths or regenerate the source-truth index"
        )
    if blockers["active_physics_unvetted"]:
        actions.append(
            "add explicit KnowledgeReference/source-index references or nonphysics/baseline "
            "classification to every active first-principles physics module"
        )
    if blockers["unsafe_claim_surfaces"]:
        actions.append(
            "remove or fail-close claim surfaces that mention validation/acceptance without "
            "an attached source hook"
        )
    if blockers["inactive_physics_unvetted"]:
        actions.append(
            "triage older physics modules as baseline-only, source-referenced, or outside "
            "the first-principles active path"
        )
    if blockers["module_role_review_required"]:
        actions.append(
            "classify remaining modules as physics, nonphysics infrastructure, or external "
            "standards-scoped utilities"
        )
    if not actions:
        actions.append(
            "module source-vetting gate is clean; proceed to closure implementation"
        )
    return actions


def _markdown_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    blockers = payload["blockers"]
    lines = [
        "# First-Principles Module Source-Vetting Audit",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"Strict passed: `{str(payload['strict_passed']).lower()}`",
        "",
        "## Policy",
        "",
        str(payload["authority_policy"]),
        "",
        "## Summary",
        "",
        f"- Total modules: `{summary['total_modules']}`",
        f"- Active first-principles import-closure modules: `{summary['active_first_principles_closure_count']}`",
        f"- Physics-facing modules: `{summary['physics_facing_count']}`",
        f"- Modules with source references: `{summary['source_referenced_count']}`",
        f"- Active physics modules needing source vetting: `{summary['active_physics_unvetted_count']}`",
        f"- Inactive physics modules needing source vetting: `{summary['inactive_physics_unvetted_count']}`",
        f"- Missing source-reference paths: `{summary['missing_source_reference_count']}`",
        f"- Unsafe claim surfaces: `{summary['unsafe_claim_surface_count']}`",
        f"- Module role reviews required: `{summary['module_role_review_required_count']}`",
        "",
        "## Classification Counts",
        "",
    ]
    for key, count in summary["classification_counts"].items():
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Blocking Groups", ""])
    _append_module_group(
        lines,
        "Active Physics Modules Needing Source Vetting",
        blockers["active_physics_unvetted"],
    )
    _append_module_group(
        lines,
        "Inactive Physics Modules Needing Source Vetting",
        blockers["inactive_physics_unvetted"],
        limit=80,
    )
    _append_missing_group(lines, blockers["missing_source_references"])
    _append_module_group(lines, "Unsafe Claim Surfaces", blockers["unsafe_claim_surfaces"])
    _append_module_group(
        lines,
        "Module Role Review Required",
        blockers["module_role_review_required"],
    )
    lines.extend(["", "## Next Required Actions", ""])
    for action in payload["next_required_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def _append_module_group(
    lines: list[str],
    title: str,
    items: list[Mapping[str, Any]],
    *,
    limit: int = 40,
) -> None:
    lines.extend([f"### {title}", ""])
    if not items:
        lines.extend(["None.", ""])
        return
    for item in items[:limit]:
        lines.append(
            f"- `{item['path']}`: `{item['classification']}` / `{item['status']}`"
        )
    if len(items) > limit:
        lines.append(f"- ... {len(items) - limit} more in JSON artifact")
    lines.append("")


def _append_missing_group(
    lines: list[str],
    items: list[Mapping[str, Any]],
    *,
    limit: int = 40,
) -> None:
    lines.extend(["### Missing Source References", ""])
    if not items:
        lines.extend(["None.", ""])
        return
    for item in items[:limit]:
        paths = ", ".join(f"`{path}`" for path in item["missing_source_paths"])
        lines.append(f"- `{item['path']}`: {paths}")
    if len(items) > limit:
        lines.append(f"- ... {len(items) - limit} more in JSON artifact")
    lines.append("")


def _active_import_closure(module_map: Mapping[str, Path]) -> set[str]:
    active: set[str] = set()
    queue: deque[str] = deque(ACTIVE_ENTRY_MODULES)
    while queue:
        module = queue.popleft()
        if module in active:
            continue
        active.add(module)
        path = module_map.get(module)
        if path is None:
            package_init = module_map.get(f"{module}.__init__")
            if package_init is None:
                continue
            path = package_init
        text = path.read_text(errors="replace")
        for imported in _module_imports(path, text):
            if not imported.startswith("dpf."):
                continue
            candidate = _best_module_candidate(imported, module_map)
            if candidate and candidate not in active:
                queue.append(candidate)
    return active


def _best_module_candidate(imported: str, module_map: Mapping[str, Path]) -> str | None:
    parts = imported.split(".")
    while len(parts) >= 2:
        candidate = ".".join(parts)
        if candidate in module_map:
            return candidate
        init_candidate = f"{candidate}.__init__"
        if init_candidate in module_map:
            return init_candidate
        parts.pop()
    return None


def _module_imports(path: Path, text: str) -> set[str]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return set()
    imports: set[str] = set()
    current_module = _module_name(path)
    current_package = current_module.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from_import(
                module=node.module,
                level=node.level,
                current_package=current_package,
            )
            if module:
                imports.add(module)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imports.add(f"{module}.{alias.name}")
    return imports


def _resolve_from_import(
    *,
    module: str | None,
    level: int,
    current_package: str,
) -> str | None:
    if level == 0:
        return module
    package_parts = current_package.split(".")
    if level > len(package_parts):
        return module
    base = ".".join(package_parts[: len(package_parts) - level + 1])
    if module:
        return f"{base}.{module}"
    return base


def _string_literals(path: Path, text: str) -> list[str]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    return literals


def _source_paths_from_strings(literals: Iterable[str], text: str) -> set[str]:
    paths: set[str] = set()
    for value in [*literals, text]:
        for match in KR_PATH_RE.findall(value):
            paths.add(_normalize_source_path(match))
    return {path for path in paths if path.startswith("KnowledgeReference/")}


def _doc_paths_from_strings(literals: Iterable[str], text: str) -> set[str]:
    paths: set[str] = set()
    for value in [*literals, text]:
        paths.update(DOC_PATH_RE.findall(value))
    return paths


def _normalize_source_path(path: str) -> str:
    path = path.rstrip(".,;:")
    for suffix in (".md", ".json", ".txt"):
        marker = f"{suffix}:"
        if marker in path:
            return path.split(marker, 1)[0] + suffix
    return path


def _has_source_marker(text: str) -> bool:
    return any(marker in text for marker in SOURCE_REFERENCE_MARKERS)


def _claim_without_source(text: str, has_source_hook: bool, is_physics: bool) -> bool:
    if has_source_hook or not is_physics:
        return False
    lower = text.lower()
    if not any(term in lower for term in CLAIM_TERMS):
        return False
    return not any(term in lower for term in FAIL_CLOSED_TERMS)


def _is_baseline_module(rel: str, text: str) -> bool:
    lower = f"{rel}\n{text}".lower()
    return any(pattern in lower for pattern in BASELINE_PATTERNS)


def _indexed_source_paths(
    index: Mapping[str, Any],
) -> tuple[set[str], dict[str, Mapping[str, Any]]]:
    indexed_paths = {
        str(item["path"])
        for item in index.get("file_inventory", ())
        if isinstance(item, Mapping) and item.get("path")
    }
    records_by_path: dict[str, Mapping[str, Any]] = {}
    for record in index.get("records", ()):
        if not isinstance(record, Mapping):
            continue
        paths = record.get("paths", {})
        if not isinstance(paths, Mapping):
            continue
        for key in ("markdown", "json"):
            source_path = paths.get(key)
            if source_path:
                records_by_path[str(source_path)] = record
                indexed_paths.add(str(source_path))
    return indexed_paths, records_by_path


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _slim_module(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": item["path"],
        "module": item["module"],
        "classification": item["classification"],
        "status": item["status"],
        "active_first_principles_closure": item["active_first_principles_closure"],
        "source_reference_count": item["source_reference_count"],
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required source-truth file missing: {path}")
    return json.loads(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
