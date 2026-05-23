#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "docs/SS16_STARTUP_BVP_EVIDENCE_PACKET_MATRIX_2026_05_23.json"

REQUIRED_CHANNELS: tuple[str, ...] = (
    "d2_breakdown",
    "preionization_state",
    "insulator_flashover",
    "sheath_liftoff",
    "early_circuit_handoff",
    "same_scope_material_geometry",
    "startup_bvp_payload",
    "uncertainty_budget",
    "review_certificate",
)

ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)

ALLOWED_CHANNEL_STATUSES: frozenset[str] = frozenset(
    {"blocked", "candidate", "cross_scope_candidate"}
)


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    issue = {"rule": rule, "message": message}
    issue.update(detail)
    return issue


def _resolve_source_path(source_path: str, repo_root: Path) -> Path:
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _iter_source_refs(matrix: dict[str, Any]):
    for row_index, row in enumerate(matrix.get("channels", []) or []):
        if not isinstance(row, dict):
            continue
        channel = row.get("channel", f"channels[{row_index}]")
        for obs_index, observable in enumerate(row.get("observables", []) or []):
            if not isinstance(observable, dict):
                continue
            observable_name = observable.get("name", f"observables[{obs_index}]")
            for ref_index, ref in enumerate(observable.get("source_refs", []) or []):
                yield channel, observable_name, ref_index, ref


def _validate_required_channels(matrix: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    channels = matrix.get("channels")
    if not isinstance(channels, list):
        issues.append(_issue("channels_not_list", "matrix.channels must be a list"))
        return

    seen: dict[str, int] = {}
    for index, row in enumerate(channels):
        if not isinstance(row, dict):
            issues.append(_issue("channel_row_not_object", "channel row must be an object", index=index))
            continue
        channel = row.get("channel")
        if not isinstance(channel, str) or not channel:
            issues.append(_issue("channel_name_missing", "channel row has no channel name", index=index))
            continue
        seen[channel] = seen.get(channel, 0) + 1
        status = row.get("status")
        if status == "accepted":
            issues.append(
                _issue(
                    "accepted_row_forbidden_in_ss16",
                    "SS16 startup evidence closure is non-promoting until source, uncertainty, payload, and review are complete",
                    channel=channel,
                )
            )
        elif status not in ALLOWED_CHANNEL_STATUSES:
            issues.append(
                _issue(
                    "channel_status_invalid",
                    "channel status must be blocked/candidate/cross_scope_candidate",
                    channel=channel,
                    status=status,
                )
            )
        if status != "accepted" and not row.get("blocked_reason"):
            issues.append(
                _issue(
                    "blocked_reason_missing",
                    "non-accepted channels require an explicit blocked_reason",
                    channel=channel,
                )
            )

    ordered = [row.get("channel") for row in channels if isinstance(row, dict)]
    if ordered != list(REQUIRED_CHANNELS):
        issues.append(
            _issue(
                "required_channels_wrong_order_or_missing",
                "SS16 channels must match the required startup evidence order",
                expected=list(REQUIRED_CHANNELS),
                actual=ordered,
            )
        )
    for channel, count in seen.items():
        if count > 1:
            issues.append(_issue("duplicate_required_channel", f"channel {channel!r} appears {count} times", channel=channel, count=count))
        if channel not in REQUIRED_CHANNELS:
            issues.append(_issue("unexpected_channel", f"channel {channel!r} is not in the SS16 required set", channel=channel))


def _validate_acceptance_boundary(matrix: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    boundary = matrix.get("acceptance_boundary")
    if not isinstance(boundary, dict):
        issues.append(_issue("acceptance_boundary_missing", "acceptance_boundary must be an object"))
        return
    for flag in ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            issues.append(
                _issue(
                    "acceptance_flag_not_false",
                    f"acceptance_boundary.{flag} must be false",
                    flag=flag,
                    value=boundary.get(flag),
                )
            )


def _validate_runtime_bridge(matrix: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    bridge = matrix.get("runtime_bridge")
    if not isinstance(bridge, dict):
        issues.append(_issue("runtime_bridge_missing", "runtime_bridge must be an object"))
        return
    for flag in ACCEPTANCE_FLAGS:
        if bridge.get(flag) is not False:
            issues.append(
                _issue(
                    "runtime_bridge_promotes_acceptance",
                    f"runtime_bridge.{flag} must be false",
                    flag=flag,
                    value=bridge.get(flag),
                )
            )
    blockers = bridge.get("blocking_channels")
    if not isinstance(blockers, list) or "startup_bvp_payload" not in blockers:
        issues.append(_issue("runtime_bridge_missing_startup_payload_blocker", "runtime bridge must keep startup_bvp_payload blocking"))


def _validate_source_refs(matrix: dict[str, Any], repo_root: Path, issues: list[dict[str, Any]]) -> None:
    checked = 0
    for channel, observable, ref_index, ref in _iter_source_refs(matrix):
        if not isinstance(ref, dict):
            issues.append(_issue("source_ref_not_object", "source ref must be an object", channel=channel, observable=observable, ref_index=ref_index))
            continue
        source_path = ref.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            issues.append(_issue("source_ref_path_missing", "source ref must include source_path", channel=channel, observable=observable, ref_index=ref_index))
            continue
        resolved = _resolve_source_path(source_path, repo_root)
        if not resolved.is_relative_to(repo_root):
            issues.append(_issue("source_ref_outside_repo", "source ref must resolve inside the repository", channel=channel, observable=observable, source_path=source_path))
            continue
        knowledge_root = (repo_root / "KnowledgeReference").resolve(strict=False)
        if not resolved.is_relative_to(knowledge_root):
            issues.append(
                _issue(
                    "source_ref_not_knowledge_reference",
                    "source refs must cite local KnowledgeReference extracted text, not repo docs/tests or retrieval output",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                )
            )
            continue
        if resolved.suffix.lower() == ".pdf":
            issues.append(_issue("source_ref_pdf_not_line_validated", "PDF refs are not line-validated; cite extracted text", channel=channel, observable=observable, source_path=source_path))
            continue
        if not resolved.exists():
            issues.append(_issue("source_ref_missing", "source ref path does not exist", channel=channel, observable=observable, source_path=source_path))
            continue
        line_start = ref.get("line_start")
        line_end = ref.get("line_end")
        if not isinstance(line_start, int) or not isinstance(line_end, int) or line_start < 1 or line_end < line_start:
            issues.append(_issue("source_ref_line_range_invalid", "source ref line_start/line_end must be valid", channel=channel, observable=observable, source_path=source_path, line_start=line_start, line_end=line_end))
            continue
        if line_end - line_start + 1 > 24:
            issues.append(
                _issue(
                    "source_ref_line_window_too_wide",
                    "source ref line windows must stay narrow enough for quote review (24 lines max)",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
        lines = resolved.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line_end > len(lines):
            issues.append(_issue("source_ref_line_range_invalid", "source ref line range exceeds source line count", channel=channel, observable=observable, source_path=source_path, line_start=line_start, line_end=line_end, line_count=len(lines)))
            continue
        extracted = " ".join(line.strip() for line in lines[line_start - 1 : line_end])
        if ref.get("quote") != extracted:
            issues.append(
                _issue(
                    "source_ref_quote_mismatch",
                    "source ref quote must exactly match the cited line window",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
        checked += 1
    if checked < 8:
        issues.append(_issue("too_few_source_refs", "SS16 packet should expose at least eight line-cited source refs", checked=checked))


def validate_matrix(matrix_path: Path, repo_root: Path | None = None) -> list[dict[str, Any]]:
    repo_root = (repo_root or ROOT).resolve()
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [_issue("matrix_unreadable", "matrix JSON could not be loaded", error=str(exc))]
    if not isinstance(matrix, dict):
        return [_issue("matrix_not_object", "matrix root must be a JSON object")]

    issues: list[dict[str, Any]] = []
    _validate_required_channels(matrix, issues)
    _validate_acceptance_boundary(matrix, issues)
    _validate_runtime_bridge(matrix, issues)
    _validate_source_refs(matrix, repo_root, issues)
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the SS16 startup BVP evidence packet matrix.")
    parser.add_argument("matrix", nargs="?", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="emit JSON issues")
    args = parser.parse_args(argv)
    issues = validate_matrix(args.matrix, args.repo_root)
    if args.json:
        print(json.dumps({"ok": not issues, "issues": issues}, indent=2))
    elif issues:
        for issue in issues:
            print(f"{issue['rule']}: {issue['message']}")
    else:
        print(f"OK: {args.matrix}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
