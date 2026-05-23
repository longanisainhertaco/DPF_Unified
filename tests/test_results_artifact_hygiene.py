"""Tests for the active results artifact same-scope namespace hygiene audit.

SS12-P0 (finding SS11-A4): the linter was upgraded from flat-string scanning to
structured JSON key-chain scanning.  These tests verify that:

  1. The live repo's active (non-archive) result artifacts are clean — no
     same-scope namespace violation.
  2. A temp active JSON with a hybrid-PIC source slug under a ``same_scope_source``
     key is flagged.
  3. A temp active JSON with an ``other_scope`` / ``wrong_scope`` value under a
     ``same_scope_source`` key is flagged.
  4. A temp active JSON with the SAME architecture evidence under an approved
     ``*_context_sources`` context key is NOT flagged.
  5. The CLI authority-policy JSON records current-behavior P1-0 semantics.
  6. Files under an ``archive_*`` directory are excluded from the scan.
  7. A malformed (non-JSON) active file is reported, not crashed on.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the scan function from the audit script
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_mod = importlib.import_module("verify_active_results_artifact_hygiene")
scan_active_results = _mod.scan_active_results
audit_main = _mod.main

ROOT = Path(__file__).resolve().parents[1]

# The SS11 hybrid-PIC source slug used as the canonical architecture evidence
# in the namespace fixtures below.
_HYBRID_PIC_SLUG = "llnl_like_180ka_axisymmetric_hybrid_pic"


def _write_results_json(results_dir: Path, name: str, payload: dict) -> Path:
    """Write *payload* as JSON to ``results_dir/name`` and return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / name
    target.write_text(json.dumps(payload, indent=2))
    return target


# ---------------------------------------------------------------------------
# Positive test: live repo results/ must be clean
# ---------------------------------------------------------------------------


class TestLiveRepoResultsClean:
    """The live results/ directory must contain no namespace violations."""

    def test_no_active_namespace_violations(self) -> None:
        issues = scan_active_results(ROOT)
        if issues:
            detail = "\n".join(
                f"  {i['file']}: rule={i.get('rule')!r} "
                f"key_path={i.get('key_path')!r} "
                f"violation={i.get('violation') or i.get('error')!r}"
                for i in issues
            )
            pytest.fail(
                f"Found {len(issues)} active result artifact namespace "
                f"violation(s):\n{detail}"
            )


# ---------------------------------------------------------------------------
# Negative test: hybrid-PIC slug under a same_scope key chain must be flagged
# ---------------------------------------------------------------------------


class TestSlugUnderSameScopeIsFlagged:
    """A hybrid-PIC slug under a same_scope_source key must produce an issue."""

    def test_slug_under_same_scope_source_is_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        payload = {
            "same_scope_source": {
                "selected_source_references": [
                    {"path": f"KnowledgeReference/{_HYBRID_PIC_SLUG}.md"},
                ],
            },
        }
        _write_results_json(
            results_dir, "probe_same_scope_slug_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        matching = [i for i in issues if i.get("rule") == "slug_under_same_scope"]
        assert matching, (
            f"Expected a slug_under_same_scope issue; got {issues}"
        )
        issue = matching[0]
        assert issue["file"] == "results/probe_same_scope_slug_2026_05_21.json"
        assert issue["violation"] == _HYBRID_PIC_SLUG
        assert "same_scope_source" in issue["key_path"]

    def test_slug_nested_deep_under_same_scope_is_flagged(
        self, tmp_path: Path
    ) -> None:
        # The same_scope key may be any ancestor, not just the top level.
        results_dir = tmp_path / "results"
        payload = {
            "candidate_results": [
                {
                    "same_scope_validation_packet": {
                        "evidence": {
                            "source": (
                                "fully-electromagnetic-hybrid-pic-fluid-dpf-"
                                "neutron-yield-acb71fa9.md"
                            ),
                        },
                    },
                },
            ],
        }
        _write_results_json(
            results_dir, "probe_nested_same_scope_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        matching = [i for i in issues if i.get("rule") == "slug_under_same_scope"]
        assert matching, (
            f"Expected a slug_under_same_scope issue for the nested slug; "
            f"got {issues}"
        )
        assert matching[0]["violation"] == (
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield"
        )


# ---------------------------------------------------------------------------
# Negative test: other_scope/wrong_scope token under same_scope_source flagged
# ---------------------------------------------------------------------------


class TestScopeTokenUnderSameScopeSourceIsFlagged:
    """other_scope / wrong_scope under a same_scope_source key must be flagged."""

    def test_other_scope_token_under_same_scope_source_is_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        payload = {
            "same_scope_source": {
                "context_group": "hybrid_pic_architecture_other_scope_evidence",
            },
        }
        _write_results_json(
            results_dir, "probe_other_scope_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        matching = [
            i
            for i in issues
            if i.get("rule") == "scope_token_under_same_scope_source"
        ]
        assert matching, (
            f"Expected a scope_token_under_same_scope_source issue; got {issues}"
        )
        assert matching[0]["violation"] == "other_scope"
        assert matching[0]["file"] == "results/probe_other_scope_2026_05_21.json"

    def test_wrong_scope_token_under_same_scope_source_is_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        payload = {
            "telemetry": {
                "same_scope_source_packet": {
                    "status": "carried_wrong_scope_group",
                },
            },
        }
        _write_results_json(
            results_dir, "probe_wrong_scope_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        matching = [
            i
            for i in issues
            if i.get("rule") == "scope_token_under_same_scope_source"
        ]
        assert matching, (
            f"Expected a scope_token_under_same_scope_source issue; got {issues}"
        )
        assert matching[0]["violation"] == "wrong_scope"


# ---------------------------------------------------------------------------
# Negative test: a forbidden token in a KEY NAME under a same_scope chain
# (SS12-P0 review HIGH) must be flagged — the scalar-value scan alone misses it
# ---------------------------------------------------------------------------


class TestForbiddenKeyNameUnderSameScopeIsFlagged:
    """A forbidden slug/token in a dict KEY name under same_scope is flagged."""

    def test_other_scope_key_name_under_same_scope_source_is_flagged(
        self, tmp_path: Path
    ) -> None:
        # The forbidden token lives in the KEY name other_scope_source_groups,
        # not in any scalar value — the SS11-era value-only scan missed this.
        results_dir = tmp_path / "results"
        payload = {
            "telemetry": {
                "same_scope_source": {
                    "status": "blocked_same_scope_source_packet_not_available",
                    "other_scope_source_groups": [
                        {"name": "pf1000_interferometry_density_campaign"},
                    ],
                },
            },
        }
        _write_results_json(
            results_dir, "probe_key_name_other_scope_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        matching = [
            i
            for i in issues
            if i.get("rule") == "forbidden_key_name_under_same_scope_source"
        ]
        assert matching, (
            f"Expected a forbidden_key_name_under_same_scope_source issue for "
            f"the other_scope_source_groups key; got {issues}"
        )
        assert matching[0]["violation"] == "other_scope"
        assert matching[0]["value"] == "other_scope_source_groups"
        assert "same_scope_source" in matching[0]["key_path"]

    def test_slug_key_name_under_same_scope_is_flagged(
        self, tmp_path: Path
    ) -> None:
        # A hybrid-PIC slug carried by a dict KEY name under a same_scope chain.
        results_dir = tmp_path / "results"
        payload = {
            "same_scope_validation_packet": {
                f"{_HYBRID_PIC_SLUG}_packet": {"status": "present"},
            },
        }
        _write_results_json(
            results_dir, "probe_key_name_slug_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        matching = [
            i
            for i in issues
            if i.get("rule") == "forbidden_key_name_under_same_scope"
        ]
        assert matching, (
            f"Expected a forbidden_key_name_under_same_scope issue; got {issues}"
        )
        assert matching[0]["violation"] == _HYBRID_PIC_SLUG


# ---------------------------------------------------------------------------
# Positive test: same evidence under an approved context key must NOT be flagged
# ---------------------------------------------------------------------------


class TestArchitectureEvidenceUnderContextKeyIsAllowed:
    """The same architecture evidence is allowed under an approved context key."""

    def test_slug_under_architecture_context_key_is_not_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        payload = {
            "architecture_or_schema_context_sources": [
                {"path": f"KnowledgeReference/{_HYBRID_PIC_SLUG}.md"},
            ],
        }
        _write_results_json(
            results_dir, "probe_arch_context_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "Architecture evidence under architecture_or_schema_context_sources "
            f"must be allowed; got issues: {issues}"
        )

    def test_slug_under_cross_scope_context_key_is_not_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        payload = {
            "cross_scope_context_sources": {
                "other_scope_reference": (
                    "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield.md"
                ),
            },
        }
        _write_results_json(
            results_dir, "probe_cross_scope_context_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "Architecture/cross-scope evidence under cross_scope_context_sources "
            f"must be allowed; got issues: {issues}"
        )

    def test_slug_outside_any_same_scope_chain_is_not_flagged(
        self, tmp_path: Path
    ) -> None:
        # Mirrors the live repo: the slug appears in ordinary physics-domain
        # source fields with no same_scope ancestor key — that is allowed.
        results_dir = tmp_path / "results"
        payload = {
            "last_step": {
                "electron_energy": {
                    "closure_validity": {
                        "electron_fluid_domain": {
                            "source": (
                                "KnowledgeReference/fully-electromagnetic-"
                                "hybrid-pic-fluid-dpf-neutron-yield.md"
                            ),
                        },
                    },
                },
            },
        }
        _write_results_json(
            results_dir, "probe_plain_source_2026_05_21.json", payload
        )

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "A hybrid-PIC slug in a plain source field outside any same_scope "
            f"key chain must be allowed; got issues: {issues}"
        )


# ---------------------------------------------------------------------------
# Policy JSON test: the CLI output must state the current-behavior contract
# ---------------------------------------------------------------------------


class TestAuthorityPolicyJson:
    """The machine-readable policy output documents the P1-0 decision."""

    def test_policy_json_records_current_behavior_and_protected_chains(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify_active_results_artifact_hygiene.py"])
        exit_code = audit_main()

        assert exit_code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["ordinary_non_same_scope_source_fields"] == "allowed"
        assert report["protected_key_chains"] == [
            "same_scope",
            "same_scope_source",
        ]
        assert report["approved_context_keys"]["exact"] == ["source_scope_context"]
        assert report["approved_context_keys"]["suffix"] == ["_context_sources"]
        assert "may otherwise appear in ordinary non-same_scope source fields" in (
            report["authority_policy"]
        )
        assert "forbidden under any 'same_scope' key chain" in (
            report["authority_policy"]
        )


# ---------------------------------------------------------------------------
# Archive exclusion test: file under archive_* must be silently ignored
# ---------------------------------------------------------------------------


class TestArchiveFileIsIgnored:
    """A namespace violation inside an archive_* directory must be ignored."""

    def test_archive_dir_namespace_violation_not_flagged(
        self, tmp_path: Path
    ) -> None:
        archive_dir = (
            tmp_path / "results" / "archive_stale_pre_ss11_2026_05_21"
        )
        payload = {
            "same_scope_source": {
                "path": f"KnowledgeReference/{_HYBRID_PIC_SLUG}.md",
                "context_group": "other_scope_evidence",
            },
        }
        _write_results_json(
            archive_dir, "experimental_old_artifact_2026_05_16.json", payload
        )

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            f"Files under archive_* must be excluded from the active scan; "
            f"got issues: {issues}"
        )

    def test_nested_archive_dir_violation_not_flagged(
        self, tmp_path: Path
    ) -> None:
        nested_archive = (
            tmp_path
            / "results"
            / "family"
            / "archive_stale_pre_ssr_2026_05_18"
        )
        payload = {"same_scope_source": {"slug": _HYBRID_PIC_SLUG}}
        _write_results_json(nested_archive, "some_old_probe.json", payload)

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "Nested archive_* files must be excluded from the scan"
        )

    def test_clean_active_file_alongside_archive_is_not_flagged(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        archive_dir = results_dir / "archive_stale_pre_ss11_2026_05_21"
        # Archive file (namespace violation — must be ignored).
        _write_results_json(
            archive_dir,
            "old.json",
            {"same_scope_source": {"slug": _HYBRID_PIC_SLUG}},
        )
        # Clean active file (no namespace violation — must produce no issues).
        _write_results_json(
            results_dir,
            "clean_active_probe.json",
            {"status": "ok", "result": "clean"},
        )

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "A clean active file alongside an archived file must produce no "
            f"issues; got: {issues}"
        )


# ---------------------------------------------------------------------------
# Malformed-file test: a non-JSON active file is reported, not crashed on
# ---------------------------------------------------------------------------


class TestMalformedActiveFileIsReported:
    """A non-JSON active file is reported as malformed_json without crashing."""

    def test_malformed_active_json_is_reported(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        broken = results_dir / "broken_active_2026_05_21.json"
        broken.write_text("{ this is not valid json ]")

        issues = scan_active_results(tmp_path)
        malformed = [i for i in issues if i.get("rule") == "malformed_json"]
        assert malformed, (
            f"Expected a malformed_json issue for the broken file; got {issues}"
        )
        assert malformed[0]["file"] == "results/broken_active_2026_05_21.json"
        assert "error" in malformed[0]

    def test_malformed_archive_json_is_ignored(self, tmp_path: Path) -> None:
        archive_dir = (
            tmp_path / "results" / "archive_stale_pre_ss11_2026_05_21"
        )
        archive_dir.mkdir(parents=True)
        broken = archive_dir / "broken_archived.json"
        broken.write_text("{ not json")

        issues = scan_active_results(tmp_path)
        assert issues == [], (
            "A malformed file under archive_* must be excluded from the scan"
        )
