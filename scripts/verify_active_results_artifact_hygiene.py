#!/usr/bin/env python3
"""Audit active result artifacts for same-scope namespace violations.

This is a read-only gate for the results artifact policy.  It does not modify
any artifact and does not promote any source, target, or simulation claim.

SS12-P0 (finding SS11-A4) upgrade
---------------------------------
The SS11 linter scanned only two flat strings
(``same_scope_3d_validation_packet``, ``llnl_like_180ka_axisymmetric_hybrid_pic``).
That coverage was too narrow: it could not enforce the intended *structure*,
namely that architecture / cross-scope evidence must never appear under
same-scope source fields.

This linter now parses each active (non-archive) ``results/**/*.json`` file as
JSON and walks the object tree by *key chain* (the ordered sequence of dict keys
from the document root to a scalar leaf; list indices are positions, not keys,
so they are not part of the key chain).  Namespace rules are then enforced over
the key chains:

  Rule 1 — ``slug_under_same_scope``
      The SS11 hybrid-PIC source slugs are FORBIDDEN under any key chain that
      contains a key whose name includes ``same_scope``.  Forbidden slugs:
        * ``llnl_like_180ka_axisymmetric_hybrid_pic``
        * ``fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield``
        * ``hybrid_pic_architecture_order_of_magnitude_other_scope``
      The same evidence outside a ``same_scope`` key chain is allowed.

  Rule 2 — ``scope_token_under_same_scope_source``
      The tokens ``other_scope`` / ``wrong_scope`` are FORBIDDEN inside any
      scalar value under a key chain that contains a key whose name includes
      ``same_scope_source``.

  Rules 1 and 2 are enforced over both scalar leaf VALUES and dict KEY NAMES.
  A forbidden slug or token carried by a key name -- for example an
  ``other_scope_source_groups`` key nested under a ``same_scope_source`` key --
  is reported as ``forbidden_key_name_under_same_scope`` /
  ``forbidden_key_name_under_same_scope_source``.

  Rule 3 — approved architecture / cross-scope context keys
      The enforced safety property is "never under same-scope evidence keys",
      not "only under context keys".  Architecture and cross-scope evidence is
      FORBIDDEN under ``same_scope`` key chains (Rules 1 and 2); it MAY appear
      in ordinary non-``same_scope`` source fields — for example a closure or
      power-port ``source`` attribution that cites a hybrid-PIC paper.  The
      explicitly named non-acceptance context keys — any key ending in
      ``_context_sources`` (``architecture_or_schema_context_sources``,
      ``cross_scope_context_sources``) or a key named ``source_scope_context``
      — are the canonical, recommended home for relocated cross-scope context,
      but they are not the only permitted location.  The *same* evidence
      relocated under a ``same_scope`` / ``same_scope_source`` key still fails.

Archive policy:
  Any path component that matches the glob ``archive_*`` is considered an
  explicitly archived stale artifact and is excluded from the active scan.
  Only results/**/*.json files whose full path contains no ``archive_*``
  component are subject to the hygiene gate.

A malformed (non-JSON) active file is reported as a ``malformed_json`` issue;
it does not crash the scan.

Exit codes:
  0 — all active artifacts are clean (no namespace violations, no malformed
      files)
  1 — one or more active artifacts violate a namespace rule or are malformed,
      and --strict or --check is set
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# SS11 hybrid-PIC source slugs.  Forbidden under any ``same_scope`` key chain.
HYBRID_PIC_SLUGS: tuple[str, ...] = (
    "llnl_like_180ka_axisymmetric_hybrid_pic",
    "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield",
    "hybrid_pic_architecture_order_of_magnitude_other_scope",
)

# Scope tokens forbidden inside any ``same_scope_source`` key chain.
WRONG_SCOPE_TOKENS: tuple[str, ...] = (
    "other_scope",
    "wrong_scope",
)

# Key name that always denotes an approved cross-scope context container.
SOURCE_SCOPE_CONTEXT_KEY = "source_scope_context"
# Any key name ending in this suffix is an approved context container, e.g.
# ``architecture_or_schema_context_sources`` / ``cross_scope_context_sources``.
CONTEXT_KEY_SUFFIX = "_context_sources"


def _is_archive_path(path: Path) -> bool:
    """Return True when any component of *path* starts with 'archive_'."""
    return any(part.startswith("archive_") for part in path.parts)


def _is_context_key(key: str) -> bool:
    """Return True when *key* names an approved architecture/context container."""
    key_lower = key.lower()
    return key_lower == SOURCE_SCOPE_CONTEXT_KEY or key_lower.endswith(
        CONTEXT_KEY_SUFFIX
    )


def _walk_json(node: Any, key_chain: tuple[str, ...]):
    """Yield ``(key_chain, scalar)`` for every scalar leaf in *node*.

    ``key_chain`` is the tuple of dict keys from the document root to the leaf.
    List elements are positions rather than named keys, so list indices do not
    extend the key chain — the namespace rules are defined over key *names*.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_json(value, key_chain + (str(key),))
    elif isinstance(node, list):
        for element in node:
            yield from _walk_json(element, key_chain)
    else:
        yield key_chain, node


def _walk_dict_keys(node: Any, key_chain: tuple[str, ...]):
    """Yield ``(parent_key_chain, key)`` for every dict key in *node*.

    Unlike :func:`_walk_json`, which stops at scalar leaves, this visits every
    dict KEY name so the namespace rules can be enforced over key names and not
    only over scalar values.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            key_str = str(key)
            yield key_chain, key_str
            yield from _walk_dict_keys(value, key_chain + (key_str,))
    elif isinstance(node, list):
        for element in node:
            yield from _walk_dict_keys(element, key_chain)


def _scan_json_object(
    data: Any, rel_file: str, issues: list[dict[str, Any]]
) -> None:
    """Apply the three namespace rules to a parsed JSON object."""
    for key_chain, leaf in _walk_json(data, ()):
        chain_lower = [key.lower() for key in key_chain]
        under_same_scope = any("same_scope" in key for key in chain_lower)
        under_same_scope_source = any(
            "same_scope_source" in key for key in chain_lower
        )
        value_text = str(leaf)
        dotted = ".".join(key_chain)

        # Rule 1 — hybrid-PIC slug under a same_scope key chain.
        if under_same_scope:
            for slug in HYBRID_PIC_SLUGS:
                if slug in value_text:
                    issues.append({
                        "file": rel_file,
                        "rule": "slug_under_same_scope",
                        "key_path": dotted,
                        "violation": slug,
                        "value": value_text,
                    })

        # Rule 2 — other_scope/wrong_scope token under a same_scope_source chain.
        if under_same_scope_source:
            for token in WRONG_SCOPE_TOKENS:
                if token in value_text.lower():
                    issues.append({
                        "file": rel_file,
                        "rule": "scope_token_under_same_scope_source",
                        "key_path": dotted,
                        "violation": token,
                        "value": value_text,
                    })

    # Rules 1b / 2b -- a forbidden slug or token in a dict KEY NAME, not only a
    # scalar value.  SS12-P0 review (HIGH): a forbidden token such as
    # ``other_scope_source_groups`` can live in a dict key name nested under a
    # ``same_scope_source`` key, which the scalar-leaf scan above never reads.
    for parent_chain, key in _walk_dict_keys(data, ()):
        ancestors_lower = [name.lower() for name in parent_chain]
        if not any("same_scope" in name for name in ancestors_lower):
            continue
        under_same_scope_source = any(
            "same_scope_source" in name for name in ancestors_lower
        )
        key_lower = key.lower()
        dotted = ".".join(parent_chain + (key,))
        for slug in HYBRID_PIC_SLUGS:
            if slug in key_lower:
                issues.append({
                    "file": rel_file,
                    "rule": "forbidden_key_name_under_same_scope",
                    "key_path": dotted,
                    "violation": slug,
                    "value": key,
                })
        if under_same_scope_source:
            for token in WRONG_SCOPE_TOKENS:
                if token in key_lower:
                    issues.append({
                        "file": rel_file,
                        "rule": "forbidden_key_name_under_same_scope_source",
                        "key_path": dotted,
                        "violation": token,
                        "value": key,
                    })


def scan_active_results(repo_root: Path) -> list[dict[str, Any]]:
    """Scan non-archive results/**/*.json for same-scope namespace violations.

    Each active (non-archive) JSON file is parsed and its object tree is walked
    by key chain.  Architecture / cross-scope evidence (the SS11 hybrid-PIC
    slugs, and ``other_scope`` / ``wrong_scope`` tokens) is FORBIDDEN under
    ``same_scope`` / ``same_scope_source`` key chains; it may otherwise appear
    in ordinary non-``same_scope`` source fields.  The approved
    ``*_context_sources`` / ``source_scope_context`` keys are the recommended
    home for relocated cross-scope context, not the only permitted location.

    Returns a list of issue dicts.  Namespace violations carry the keys:
      - ``file``: str, path relative to *repo_root*
      - ``rule``: str, ``slug_under_same_scope`` or
        ``scope_token_under_same_scope_source``
      - ``key_path``: str, dotted key chain from the document root to the leaf
      - ``violation``: str, the forbidden slug or token found
      - ``value``: str, the offending scalar leaf value
    A malformed (non-JSON) active file produces an issue with the keys:
      - ``file``: str, path relative to *repo_root*
      - ``rule``: ``malformed_json``
      - ``error``: str, the JSON decode error message
    """
    results_dir = repo_root / "results"
    if not results_dir.is_dir():
        return []

    issues: list[dict[str, Any]] = []
    for json_path in sorted(results_dir.rglob("*.json")):
        rel_path = json_path.relative_to(repo_root)
        if _is_archive_path(rel_path):
            continue
        rel_file = str(rel_path)
        try:
            raw = json_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            issues.append({
                "file": rel_file,
                "rule": "malformed_json",
                "error": f"unreadable: {exc}",
            })
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append({
                "file": rel_file,
                "rule": "malformed_json",
                "error": str(exc),
            })
            continue
        _scan_json_object(data, rel_file, issues)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit active results/ JSON artifacts for same-scope namespace "
            "violations.  Excludes archive_* directories."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any active artifact violates a namespace rule.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Read-only verification mode: renders the report and fails if any "
            "namespace violation or malformed file is found in a non-archive "
            "artifact.  Use in CI."
        ),
    )
    args = parser.parse_args()

    issues = scan_active_results(ROOT)
    clean = len(issues) == 0

    payload: dict[str, Any] = {
        "scope": "active_results_artifact_hygiene",
        "authority_policy": (
            "results/ JSON artifacts outside archive_* directories are walked "
            "by key chain; the SS11 hybrid-PIC source slugs are forbidden under "
            "any 'same_scope' key chain, 'other_scope'/'wrong_scope' tokens are "
            "forbidden under any 'same_scope_source' key chain (over both "
            "scalar values and dict key names); architecture or cross-scope "
            "evidence may otherwise appear in ordinary non-same_scope source "
            "fields, with the approved context keys (a key ending in "
            "'_context_sources' or named 'source_scope_context') the "
            "recommended home for relocated cross-scope context; stale "
            "artifacts are relocated (not rewritten) to archive_* dirs"
        ),
        "clean": clean,
        "active_hit_count": len(issues),
        "hybrid_pic_slugs": list(HYBRID_PIC_SLUGS),
        "wrong_scope_tokens": list(WRONG_SCOPE_TOKENS),
        "approved_context_keys": {
            "exact": [SOURCE_SCOPE_CONTEXT_KEY],
            "suffix": [CONTEXT_KEY_SUFFIX],
        },
        "issues": issues,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))

    if (args.strict or args.check) and not clean:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
