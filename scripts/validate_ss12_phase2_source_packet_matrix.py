#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_EXTRACTED_2026_05_22.json"

REQUIRED_CHANNELS: tuple[str, ...] = (
    "geometry",
    "bank_circuit",
    "gas_fill",
    "current_waveform",
    "startup",
    "density_history",
    "em_field_history",
    "temperature_or_distribution_history",
    "neutron_scalar_yield",
    "neutron_timing",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_response",
    "uncertainty_budget",
    "review_certificate",
)

ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)

REVIEWED_ACCEPTANCE_STATUSES: frozenset[str] = frozenset({"reviewed", "accepted"})


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    issue = {"rule": rule, "message": message}
    issue.update(detail)
    return issue


def _iter_source_refs(matrix: dict[str, Any]):
    for channel_index, row in enumerate(matrix.get("channels", [])):
        if not isinstance(row, dict):
            continue
        channel = row.get("channel", f"channels[{channel_index}]")
        for observable_index, observable in enumerate(row.get("observables", []) or []):
            if not isinstance(observable, dict):
                continue
            observable_name = observable.get("name", f"observables[{observable_index}]")
            for ref_index, ref in enumerate(observable.get("source_refs", []) or []):
                yield channel, observable_name, ref_index, ref


def _resolve_source_path(source_path: str, repo_root: Path) -> Path:
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _validate_required_channels(matrix: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    channels = matrix.get("channels")
    if not isinstance(channels, list):
        issues.append(_issue("channels_not_list", "matrix.channels must be a list"))
        return

    seen: dict[str, int] = {}
    for index, row in enumerate(channels):
        if not isinstance(row, dict):
            issues.append(
                _issue("channel_row_not_object", "channel row must be an object", index=index)
            )
            continue
        channel = row.get("channel")
        if not isinstance(channel, str) or not channel:
            issues.append(
                _issue("channel_name_missing", "channel row has no channel name", index=index)
            )
            continue
        seen[channel] = seen.get(channel, 0) + 1

    for channel in REQUIRED_CHANNELS:
        if channel not in seen:
            issues.append(
                _issue(
                    "missing_required_channel",
                    f"required channel {channel!r} is absent",
                    channel=channel,
                )
            )

    for channel, count in seen.items():
        if count > 1:
            issues.append(
                _issue(
                    "duplicate_required_channel",
                    f"channel {channel!r} appears {count} times",
                    channel=channel,
                    count=count,
                )
            )
        if channel not in REQUIRED_CHANNELS:
            issues.append(
                _issue(
                    "unexpected_channel",
                    f"channel {channel!r} is not in the SS12 P1 Phase 2 required set",
                    channel=channel,
                )
            )


def _validate_acceptance_boundary(
    matrix: dict[str, Any], issues: list[dict[str, Any]]
) -> None:
    boundary = matrix.get("acceptance_boundary")
    if not isinstance(boundary, dict):
        issues.append(
            _issue("acceptance_boundary_missing", "acceptance_boundary must be an object")
        )
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


def _validate_accepted_rows(matrix: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    channels = matrix.get("channels", [])
    if not isinstance(channels, list):
        return

    for index, row in enumerate(channels):
        if not isinstance(row, dict) or row.get("status") != "accepted":
            continue
        channel = row.get("channel", f"channels[{index}]")
        issues.append(
            _issue(
                "accepted_row_forbidden_in_phase2",
                "Phase 2 extraction matrices are non-promoting; accepted rows are forbidden",
                channel=channel,
            )
        )
        if row.get("scope_match") != "same_scope":
            issues.append(
                _issue(
                    "accepted_row_not_same_scope",
                    "accepted channel must have scope_match='same_scope'",
                    channel=channel,
                    scope_match=row.get("scope_match"),
                )
            )

        observables = row.get("observables")
        if not isinstance(observables, list) or not observables:
            issues.append(
                _issue(
                    "accepted_row_missing_source_refs",
                    "accepted channel must carry at least one observable source ref",
                    channel=channel,
                )
            )
            continue

        row_refs = 0
        for observable_index, observable in enumerate(observables):
            if not isinstance(observable, dict):
                issues.append(
                    _issue(
                        "accepted_observable_not_object",
                        "accepted channel observable must be an object",
                        channel=channel,
                        observable_index=observable_index,
                    )
                )
                continue
            refs = observable.get("source_refs")
            if not isinstance(refs, list) or not refs:
                issues.append(
                    _issue(
                        "accepted_row_missing_source_refs",
                        "accepted observable must carry source refs",
                        channel=channel,
                        observable=observable.get("name", f"observables[{observable_index}]"),
                    )
                )
                continue
            for ref_index, ref in enumerate(refs):
                row_refs += 1
                if not isinstance(ref, dict):
                    issues.append(
                        _issue(
                            "accepted_source_ref_not_object",
                            "accepted source ref must be an object",
                            channel=channel,
                            ref_index=ref_index,
                        )
                    )
                    continue
                review_status = ref.get("review_status")
                if review_status not in REVIEWED_ACCEPTANCE_STATUSES:
                    issues.append(
                        _issue(
                            "accepted_source_ref_not_reviewed",
                            "accepted source refs must be reviewed or accepted",
                            channel=channel,
                            observable=observable.get("name"),
                            review_status=review_status,
                        )
                    )
        if row_refs == 0:
            issues.append(
                _issue(
                    "accepted_row_missing_source_refs",
                    "accepted channel has no usable source refs",
                    channel=channel,
                )
            )


def _validate_source_refs(
    matrix: dict[str, Any], repo_root: Path, issues: list[dict[str, Any]]
) -> None:
    for channel, observable, ref_index, ref in _iter_source_refs(matrix):
        if not isinstance(ref, dict):
            issues.append(
                _issue(
                    "source_ref_not_object",
                    "source ref must be an object",
                    channel=channel,
                    observable=observable,
                    ref_index=ref_index,
                )
            )
            continue

        source_path = ref.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            issues.append(
                _issue(
                    "source_ref_path_missing",
                    "source ref must include source_path",
                    channel=channel,
                    observable=observable,
                    ref_index=ref_index,
                )
            )
            continue

        resolved = _resolve_source_path(source_path, repo_root)
        if not resolved.is_relative_to(repo_root):
            issues.append(
                _issue(
                    "source_ref_outside_repo",
                    "source ref must resolve inside the repository",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                )
            )
            continue
        if resolved.suffix.lower() == ".pdf":
            issues.append(
                _issue(
                    "source_ref_pdf_not_line_validated",
                    "PDF refs are not read by this validator; use extracted text refs",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                )
            )
            continue
        if not resolved.exists():
            issues.append(
                _issue(
                    "source_ref_missing",
                    "source ref path does not exist",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                )
            )
            continue

        line_start = ref.get("line_start")
        line_end = ref.get("line_end")
        if (
            not isinstance(line_start, int)
            or not isinstance(line_end, int)
            or line_start < 1
            or line_end < line_start
        ):
            issues.append(
                _issue(
                    "source_ref_line_range_invalid",
                    "source ref line_start/line_end must be a valid 1-based range",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
            continue

        try:
            line_count = len(resolved.read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError as exc:
            issues.append(
                _issue(
                    "source_ref_unreadable",
                    "source ref could not be read",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                    error=str(exc),
                )
            )
            continue
        if line_end > line_count:
            issues.append(
                _issue(
                    "source_ref_line_range_invalid",
                    "source ref line range exceeds source line count",
                    channel=channel,
                    observable=observable,
                    source_path=source_path,
                    line_start=line_start,
                    line_end=line_end,
                    line_count=line_count,
                )
            )


def validate_matrix(matrix_path: Path, repo_root: Path | None = None) -> list[dict[str, Any]]:
    repo_root = (repo_root or ROOT).resolve()
    matrix_path = matrix_path.resolve()
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [_issue("matrix_unreadable", "matrix JSON could not be loaded", error=str(exc))]
    if not isinstance(matrix, dict):
        return [_issue("matrix_not_object", "matrix root must be a JSON object")]

    issues: list[dict[str, Any]] = []
    _validate_required_channels(matrix, issues)
    _validate_acceptance_boundary(matrix, issues)
    _validate_accepted_rows(matrix, issues)
    _validate_source_refs(matrix, repo_root, issues)
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the SS12 P1 Phase 2 source-packet matrix."
    )
    parser.add_argument("matrix", nargs="?", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="emit JSON issues")
    args = parser.parse_args(argv)

    issues = validate_matrix(args.matrix, args.repo_root)
    if args.json:
        print(json.dumps({"ok": not issues, "issues": issues}, indent=2, sort_keys=True))
    elif issues:
        for issue in issues:
            print(
                f"FAIL {issue['rule']}: {issue['message']} "
                f"{json.dumps({k: v for k, v in issue.items() if k not in {'rule', 'message'}}, sort_keys=True)}"
            )
    else:
        print(f"PASS {args.matrix}: 0 source-packet matrix issues")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
