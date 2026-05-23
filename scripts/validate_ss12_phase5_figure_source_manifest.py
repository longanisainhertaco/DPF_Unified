#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json"

ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_figure_claim",
    "accepted_observable_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "id",
    "channel",
    "source_path",
    "line_start",
    "line_end",
    "figure_id",
    "scope_classification",
    "extraction_priority",
    "review_state",
    "status",
)
REQUIRED_CHANNELS: tuple[str, ...] = (
    "current_waveform",
    "density_history",
    "em_field_history",
    "neutron_timing_or_spectrum",
)
FORBIDDEN_ACCEPTED_STATES: frozenset[str] = frozenset(
    {"accepted", "reviewed_as_accepted", "same_source_accepted"}
)
ROW_ACCEPTANCE_FLAGS: tuple[str, ...] = ACCEPTANCE_FLAGS


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    issue = {"rule": rule, "message": message}
    issue.update(detail)
    return issue


def _resolve_source_path(source_path: str, repo_root: Path) -> Path:
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalized_state(value: object) -> str:
    return str(value or "").strip().lower()


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    repo_root = repo_root.resolve()

    if manifest.get("manifest_id") != "ss12_p1_phase5_figure_source_manifest":
        issues.append(
            _issue(
                "manifest_id_invalid",
                "manifest_id must be ss12_p1_phase5_figure_source_manifest",
                value=manifest.get("manifest_id"),
            )
        )
    if manifest.get("validation_scope") != "pf1000_full_energy_27_to_40_kv":
        issues.append(
            _issue(
                "validation_scope_invalid",
                "validation_scope must be pf1000_full_energy_27_to_40_kv",
                value=manifest.get("validation_scope"),
            )
        )

    _validate_acceptance_boundary(manifest, issues)
    _validate_figure_sources(manifest, repo_root, issues)
    return issues


def _validate_acceptance_boundary(
    manifest: dict[str, Any], issues: list[dict[str, Any]]
) -> None:
    boundary = manifest.get("acceptance_boundary")
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


def _validate_figure_sources(
    manifest: dict[str, Any], repo_root: Path, issues: list[dict[str, Any]]
) -> None:
    rows = manifest.get("figure_sources")
    if not isinstance(rows, list):
        issues.append(_issue("figure_sources_not_list", "figure_sources must be a list"))
        return

    seen_channels: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(_issue("figure_source_not_object", "row must be an object", index=index))
            continue
        row_id = str(row.get("id", f"figure_sources[{index}]"))
        channel = row.get("channel")
        if isinstance(channel, str):
            seen_channels.add(channel)
        _validate_row_fields(row, row_id, issues)
        _validate_row_acceptance_boundary(row, row_id, issues)
        _validate_row_source_ref(row, row_id, repo_root, issues)

    for channel in REQUIRED_CHANNELS:
        if channel not in seen_channels:
            issues.append(
                _issue(
                    "required_figure_channel_missing",
                    f"required Phase 5 figure channel {channel!r} is missing",
                    channel=channel,
                )
            )


def _validate_row_fields(
    row: dict[str, Any], row_id: str, issues: list[dict[str, Any]]
) -> None:
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            issues.append(
                _issue(
                    "figure_source_required_field_missing",
                    f"figure source row is missing {field}",
                    row_id=row_id,
                    field=field,
                )
            )


def _validate_row_acceptance_boundary(
    row: dict[str, Any], row_id: str, issues: list[dict[str, Any]]
) -> None:
    for field in ("status", "review_state", "scope_classification"):
        value = _normalized_state(row.get(field))
        if value in FORBIDDEN_ACCEPTED_STATES:
            issues.append(
                _issue(
                    "figure_source_accepted_status_forbidden",
                    "Phase 5 figure sources are candidates only; accepted states are forbidden",
                    row_id=row_id,
                    field=field,
                    value=row.get(field),
                )
            )
    for flag in ROW_ACCEPTANCE_FLAGS:
        if row.get(flag) is not False:
            issues.append(
                _issue(
                    "figure_source_acceptance_flag_not_false",
                    f"figure source {flag} must be false",
                    row_id=row_id,
                    flag=flag,
                    value=row.get(flag),
                )
            )


def _validate_row_source_ref(
    row: dict[str, Any], row_id: str, repo_root: Path, issues: list[dict[str, Any]]
) -> None:
    if row.get("status") == "blocked_missing_line_citable_source":
        if row.get("source_path") is not None:
            issues.append(
                _issue(
                    "blocked_figure_source_has_source_path",
                    "blocked_missing_line_citable_source rows must not pretend to have a source path",
                    row_id=row_id,
                )
            )
        return

    source_path = row.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        issues.append(
            _issue(
                "figure_source_path_missing",
                "candidate figure source row must include source_path",
                row_id=row_id,
            )
        )
        return

    resolved = _resolve_source_path(source_path, repo_root)
    if not _is_relative_to(resolved, repo_root):
        issues.append(
            _issue(
                "figure_source_outside_repo",
                "figure source path must resolve under repo root",
                row_id=row_id,
                source_path=source_path,
                resolved=str(resolved),
            )
        )
        return
    if not resolved.exists():
        issues.append(
            _issue(
                "figure_source_path_missing_on_disk",
                "figure source path does not exist",
                row_id=row_id,
                source_path=source_path,
            )
        )
        return

    line_start = row.get("line_start")
    line_end = row.get("line_end")
    lines = resolved.read_text(errors="ignore").splitlines()
    if not (
        isinstance(line_start, int)
        and isinstance(line_end, int)
        and 1 <= line_start <= line_end <= len(lines)
    ):
        issues.append(
            _issue(
                "figure_source_line_range_invalid",
                "line_start/line_end must define an existing source range",
                row_id=row_id,
                line_start=line_start,
                line_end=line_end,
                total_lines=len(lines),
            )
        )
        return

    tokens = row.get("evidence_tokens")
    if not isinstance(tokens, list) or not tokens:
        issues.append(
            _issue(
                "figure_source_evidence_tokens_missing",
                "figure source row must include evidence_tokens",
                row_id=row_id,
            )
        )
        return
    excerpt = " ".join(lines[line_start - 1 : line_end]).lower()
    if not any(str(token).lower() in excerpt for token in tokens):
        issues.append(
            _issue(
                "figure_source_evidence_tokens_not_found",
                "at least one evidence token must appear in the cited line range",
                row_id=row_id,
                tokens=tokens,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    issues = validate_manifest(manifest, args.repo_root)
    summary = {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "manifest": str(args.manifest),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
