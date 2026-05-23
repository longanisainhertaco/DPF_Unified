#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "docs/SS18_NEUTRON_DIAGNOSTIC_VALIDATION_STACK_2026_05_23.json"

REQUIRED_MECHANISMS: tuple[str, ...] = (
    "yield",
    "timing",
    "spectrum",
    "anisotropy",
    "detector_activation_response",
    "diagnostic_mapping",
    "uncertainty_blockers",
)

ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)

ALLOWED_STATUSES: frozenset[str] = frozenset({"blocked", "candidate", "cross_scope_candidate", "rejected"})


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    issue = {"rule": rule, "message": message}
    issue.update(detail)
    return issue


def _resolve_source_path(source_path: str, repo_root: Path) -> Path:
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _iter_source_refs(packet: dict[str, Any]):
    for row_index, row in enumerate(packet.get("mechanisms", []) or []):
        if not isinstance(row, dict):
            continue
        mechanism = row.get("mechanism", f"mechanisms[{row_index}]")
        for obs_index, observable in enumerate(row.get("observables", []) or []):
            if not isinstance(observable, dict):
                continue
            observable_name = observable.get("name", f"observables[{obs_index}]")
            for ref_index, ref in enumerate(observable.get("source_refs", []) or []):
                yield mechanism, observable_name, ref_index, ref


def _validate_required_mechanisms(packet: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    mechanisms = packet.get("mechanisms")
    if not isinstance(mechanisms, list):
        issues.append(_issue("mechanisms_not_list", "packet.mechanisms must be a list"))
        return

    seen: dict[str, int] = {}
    for index, row in enumerate(mechanisms):
        if not isinstance(row, dict):
            issues.append(_issue("mechanism_row_not_object", "mechanism row must be an object", index=index))
            continue
        mechanism = row.get("mechanism")
        if not isinstance(mechanism, str) or not mechanism:
            issues.append(_issue("mechanism_name_missing", "mechanism row has no mechanism name", index=index))
            continue
        seen[mechanism] = seen.get(mechanism, 0) + 1
        if mechanism not in REQUIRED_MECHANISMS:
            issues.append(_issue("unexpected_mechanism", f"mechanism {mechanism!r} is not in the SS18 required set", mechanism=mechanism))
        status = row.get("status")
        if status == "accepted":
            issues.append(_issue("accepted_mechanism_forbidden_in_ss18", "SS18 must not accept neutron mechanisms before diagnostic review/certificate closure", mechanism=mechanism))
        elif status not in ALLOWED_STATUSES:
            issues.append(_issue("mechanism_status_invalid", "mechanism status must be blocked/candidate/cross_scope_candidate/rejected", mechanism=mechanism, status=status))
        if row.get("promotes_acceptance") is not False:
            issues.append(_issue("mechanism_promotes_acceptance", "SS18 mechanisms must not promote acceptance", mechanism=mechanism, value=row.get("promotes_acceptance")))
        if not row.get("blocked_reason"):
            issues.append(_issue("blocked_reason_missing", "mechanisms require an explicit blocked_reason", mechanism=mechanism))
        if not row.get("diagnostic_channel"):
            issues.append(_issue("diagnostic_channel_missing", "mechanisms require diagnostic_channel mapping", mechanism=mechanism))
        observables = row.get("observables")
        if not isinstance(observables, list) or not observables:
            issues.append(_issue("observables_missing", "mechanisms require at least one observable/blocker row", mechanism=mechanism))

    for mechanism in REQUIRED_MECHANISMS:
        if mechanism not in seen:
            issues.append(_issue("missing_required_mechanism", f"required mechanism {mechanism!r} is absent", mechanism=mechanism))
    for mechanism, count in seen.items():
        if count > 1:
            issues.append(_issue("duplicate_required_mechanism", f"mechanism {mechanism!r} appears {count} times", mechanism=mechanism, count=count))


def _validate_acceptance_boundary(packet: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    boundary = packet.get("acceptance_boundary")
    if not isinstance(boundary, dict):
        issues.append(_issue("acceptance_boundary_missing", "acceptance_boundary must be an object"))
        return
    for flag in ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            issues.append(_issue("acceptance_flag_not_false", f"acceptance_boundary.{flag} must be false", flag=flag, value=boundary.get(flag)))


def _validate_diagnostic_completeness(packet: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    completeness = packet.get("diagnostic_completeness_check")
    if not isinstance(completeness, dict):
        issues.append(_issue("diagnostic_completeness_missing", "diagnostic_completeness_check must be an object"))
        return
    if completeness.get("complete_for_acceptance") is not False:
        issues.append(
            _issue(
                "diagnostic_completeness_promotes_acceptance",
                "diagnostic_completeness_check.complete_for_acceptance must stay false until the neutron certificate stack is complete",
                value=completeness.get("complete_for_acceptance"),
            )
        )
    if completeness.get("required_mechanisms") != list(REQUIRED_MECHANISMS):
        issues.append(
            _issue(
                "diagnostic_completeness_required_mechanisms_mismatch",
                "diagnostic_completeness_check.required_mechanisms must match the SS18 required mechanism set in order",
                value=completeness.get("required_mechanisms"),
            )
        )
    blocking_reasons = completeness.get("blocking_reasons")
    if not isinstance(blocking_reasons, list) or not blocking_reasons or not all(isinstance(reason, str) and reason for reason in blocking_reasons):
        issues.append(
            _issue(
                "diagnostic_completeness_blockers_missing",
                "diagnostic_completeness_check requires explicit non-empty blocking_reasons while complete_for_acceptance is false",
                value=blocking_reasons,
            )
        )


def _validate_source_refs(packet: dict[str, Any], repo_root: Path, issues: list[dict[str, Any]]) -> None:
    knowledge_root = (repo_root / "KnowledgeReference").resolve(strict=False)
    for mechanism, observable, ref_index, ref in _iter_source_refs(packet):
        if not isinstance(ref, dict):
            issues.append(_issue("source_ref_not_object", "source ref must be an object", mechanism=mechanism, observable=observable, ref_index=ref_index))
            continue
        source_path = ref.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            issues.append(_issue("source_ref_path_missing", "source ref must include source_path", mechanism=mechanism, observable=observable, ref_index=ref_index))
            continue
        resolved = _resolve_source_path(source_path, repo_root)
        if not resolved.is_relative_to(repo_root):
            issues.append(_issue("source_ref_outside_repo", "source ref must resolve inside the repository", mechanism=mechanism, observable=observable, source_path=source_path))
            continue
        if not resolved.is_relative_to(knowledge_root):
            issues.append(_issue("source_ref_not_knowledge_reference", "source refs must cite local KnowledgeReference extracted text", mechanism=mechanism, observable=observable, source_path=source_path))
            continue
        if resolved.suffix.lower() == ".pdf":
            issues.append(_issue("source_ref_pdf_not_line_validated", "PDF refs are not line-validated; cite extracted text", mechanism=mechanism, observable=observable, source_path=source_path))
            continue
        if not resolved.exists():
            issues.append(_issue("source_ref_missing", "source ref path does not exist", mechanism=mechanism, observable=observable, source_path=source_path))
            continue
        line_start = ref.get("line_start")
        line_end = ref.get("line_end")
        if not isinstance(line_start, int) or not isinstance(line_end, int) or line_start < 1 or line_end < line_start:
            issues.append(_issue("source_ref_line_range_invalid", "source ref line_start/line_end must be valid", mechanism=mechanism, observable=observable, source_path=source_path, line_start=line_start, line_end=line_end))
            continue
        if line_end - line_start + 1 > 24:
            issues.append(_issue("source_ref_line_window_too_wide", "source ref line windows must be <=24 lines", mechanism=mechanism, observable=observable, source_path=source_path, line_start=line_start, line_end=line_end))
        lines = resolved.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line_end > len(lines):
            issues.append(_issue("source_ref_line_range_invalid", "source ref line range exceeds source line count", mechanism=mechanism, observable=observable, source_path=source_path, line_start=line_start, line_end=line_end, line_count=len(lines)))
            continue
        extracted = " ".join(line.strip() for line in lines[line_start - 1 : line_end])
        if ref.get("quote") != extracted:
            issues.append(_issue("source_ref_quote_mismatch", "source ref quote must exactly match the cited line window", mechanism=mechanism, observable=observable, source_path=source_path, line_start=line_start, line_end=line_end))


def validate_packet(packet_path: Path, repo_root: Path | None = None) -> list[dict[str, Any]]:
    repo_root = (repo_root or ROOT).resolve()
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [_issue("packet_unreadable", "packet JSON could not be loaded", error=str(exc))]
    if not isinstance(packet, dict):
        return [_issue("packet_not_object", "packet root must be a JSON object")]

    issues: list[dict[str, Any]] = []
    _validate_required_mechanisms(packet, issues)
    _validate_acceptance_boundary(packet, issues)
    _validate_diagnostic_completeness(packet, issues)
    _validate_source_refs(packet, repo_root, issues)
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the SS18 neutron diagnostic packet.")
    parser.add_argument("packet", nargs="?", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="emit JSON issues")
    args = parser.parse_args(argv)
    issues = validate_packet(args.packet, args.repo_root)
    if args.json:
        print(json.dumps(issues, indent=2))
    elif issues:
        for issue in issues:
            print(f"{issue['rule']}: {issue['message']}")
    else:
        print("SS18 neutron diagnostic packet validation passed")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
