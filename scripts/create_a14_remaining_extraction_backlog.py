"""Create a current A14 remaining-extraction backlog from local artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = REPO_ROOT / "docs" / "A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.json"
OUTPUT_MD = REPO_ROOT / "docs" / "A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.md"

CROP_REVIEW_PATH = "docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.json"
TABLE_PACKET_PATH = (
    "KnowledgeReference/digitization/a14-2026-05-11-table-draft-packets.json"
)
SPRINGHAM_MONO_PACKET_PATH = (
    "KnowledgeReference/digitization/"
    "a14-2026-05-11-springham-fig5-monoenergetic-draft-packet.json"
)
SPRINGHAM_GAUSSIAN_PACKET_PATH = (
    "KnowledgeReference/digitization/"
    "a14-2026-05-11-springham-fig5-gaussian-curves-draft-packet.json"
)
KLIR_FIG2_PACKET_PATH = (
    "KnowledgeReference/digitization/"
    "a14-2026-05-11-klir-fig2-timing-response-draft-packet.json"
)
CIKHARDTOVA_BLOCKER_PATH = (
    "docs/A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.json"
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str) -> Any:
    return json.loads((REPO_ROOT / path).read_text())


def build_backlog() -> dict[str, Any]:
    crop_review = _load_json(CROP_REVIEW_PATH)
    table_bundle = _load_json(TABLE_PACKET_PATH)
    cikhardtova_blocker = _load_json(CIKHARDTOVA_BLOCKER_PATH)

    packets_by_crop: dict[str, list[dict[str, str]]] = {}
    for packet in table_bundle["packets"]:
        packets_by_crop.setdefault(packet["crop_image_path"], []).append(
            {
                "task_id": packet["task_id"],
                "packet_path": TABLE_PACKET_PATH,
                "packet_kind": "table_draft_in_bundle",
            }
        )
    for packet_path in (
        SPRINGHAM_MONO_PACKET_PATH,
        SPRINGHAM_GAUSSIAN_PACKET_PATH,
        KLIR_FIG2_PACKET_PATH,
    ):
        packet = _load_json(packet_path)
        packets_by_crop.setdefault(packet["figure_image_path"], []).append(
            {
                "task_id": packet["task_id"],
                "packet_path": packet_path,
                "packet_kind": "figure_digitization_draft",
            }
        )

    blocker_by_crop = {
        cikhardtova_blocker["figure_image_path"]: {
            "task_id": cikhardtova_blocker["task_id"],
            "blocker_path": CIKHARDTOVA_BLOCKER_PATH,
            "draft_extraction_status": cikhardtova_blocker[
                "draft_extraction_status"
            ],
        }
    }

    backlog_items = []
    for entry in crop_review["review_entries"]:
        crop_path = entry["crop_path"]
        packets = packets_by_crop.get(crop_path, [])
        blocker = blocker_by_crop.get(crop_path)
        if packets:
            extraction_status = "reviewable_draft_packet_exists"
            next_action = "submit draft packet for independent review or correct it"
        elif blocker:
            extraction_status = "extraction_blocked"
            next_action = "perform manual or vector-assisted curve separation"
        elif entry["boundary_review_status"] == "manual_review_required":
            extraction_status = "manual_review_required"
            next_action = "complete visual/manual review before numeric extraction"
        elif entry["boundary_review_status"] == "boundary_ready_for_draft_extraction":
            extraction_status = "ready_not_started"
            next_action = "create source-bound draft digitization packet"
        elif entry["boundary_review_status"] == "draft_extracted_review_blocked":
            extraction_status = "reviewable_draft_packet_expected"
            next_action = "confirm draft packet linkage"
        else:
            extraction_status = "unclassified"
            next_action = "review crop status"
        backlog_items.append(
            {
                "source_slug": entry["source_slug"],
                "figure_id": entry["figure_id"],
                "extraction_kind": entry["extraction_kind"],
                "crop_path": crop_path,
                "crop_sha256": entry["crop_sha256"],
                "crop_boundary_status": entry["boundary_review_status"],
                "extraction_status": extraction_status,
                "accepted_for_validation": False,
                "draft_packets": packets,
                "blocker": blocker,
                "next_action": next_action,
            }
        )

    status_counts = Counter(item["extraction_status"] for item in backlog_items)
    return {
        "model_role": "a14_remaining_extraction_backlog",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_crop_review_path": CROP_REVIEW_PATH,
        "source_crop_review_sha256": sha256_file(REPO_ROOT / CROP_REVIEW_PATH),
        "accepted_for_validation_count": 0,
        "total_crop_count": len(backlog_items),
        "reviewable_draft_packet_count": sum(
            len(item["draft_packets"]) for item in backlog_items
        ),
        "distinct_reviewable_crop_count": sum(
            1 for item in backlog_items if item["draft_packets"]
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "backlog_items": backlog_items,
    }


def _markdown(backlog: dict[str, Any]) -> str:
    lines = [
        "# A14 Remaining Extraction Backlog",
        "",
        f"Generated UTC: `{backlog['generated_utc']}`",
        "",
        "This report is generated from the current A14 crop review, draft "
        "packets, and blocker reports. It is not validation evidence.",
        "",
        "## Summary",
        "",
        f"- Total crop candidates: {backlog['total_crop_count']}",
        f"- Reviewable draft packets: {backlog['reviewable_draft_packet_count']}",
        f"- Distinct crops with reviewable drafts: {backlog['distinct_reviewable_crop_count']}",
        f"- Accepted for validation: {backlog['accepted_for_validation_count']}",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(
        f"- `{status}`: {count}"
        for status, count in backlog["status_counts"].items()
    )
    lines.extend(["", "## Open Items", ""])
    for item in backlog["backlog_items"]:
        if item["extraction_status"] in {
            "ready_not_started",
            "manual_review_required",
            "extraction_blocked",
        }:
            lines.append(
                f"- `{item['source_slug']}` `{item['figure_id']}`: "
                f"`{item['extraction_status']}`; {item['next_action']}."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    backlog = build_backlog()
    OUTPUT_JSON.write_text(json.dumps(backlog, indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(_markdown(backlog))
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "total_crop_count": backlog["total_crop_count"],
                "reviewable_draft_packet_count": backlog[
                    "reviewable_draft_packet_count"
                ],
                "accepted_for_validation_count": backlog[
                    "accepted_for_validation_count"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
