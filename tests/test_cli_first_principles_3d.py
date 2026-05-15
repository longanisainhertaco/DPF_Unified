"""CLI coverage for the package-native 3-D first-principles runner."""

from __future__ import annotations

import json

from click.testing import CliRunner


def test_first_principles_3d_default_deck_writes_json(tmp_path) -> None:
    from dpf.cli.main import cli

    output = tmp_path / "first_principles_3d.json"

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["tool"] == "dpf first-principles-3d"
    assert payload["runner"] == "dpf.fields.HybridPIC3DSimulator"
    assert payload["command_status"] == (
        "package_native_first_principles_3d_engineering_run"
    )
    assert payload["deck"]["source"] == "built_in"
    assert payload["deck"]["name"] == (
        "pf1000_akel_16kv_1p2torr_shot_12581_engineering_candidate"
    )
    assert payload["deck"]["device_name"] == (
        "PF-1000/Akel shot 12581 engineering candidate"
    )
    assert payload["scientific_status"] == "engineering_candidate_not_validation"
    assert payload["validation_packet"]["same_scope_source_status"] == (
        "blocked_same_scope_source_packet_not_available"
    )
    assert payload["simulation"]["status"] == (
        "candidate_engineering_3d_hybrid_pic_simulation"
    )
    assert "Package-native 3-D first-principles engineering candidate" in result.output


def test_first_principles_3d_reads_deck_json_and_prints_json(tmp_path) -> None:
    from dpf.cli.main import cli

    deck = tmp_path / "deck.json"
    deck.write_text(json.dumps({
        "name": "tiny_test_deck",
        "steps": 1,
        "grid_shape": [4, 4, 4],
        "dt_s": 1.0e-13,
        "apply_circuit_boundary": False,
    }))

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--deck",
            str(deck),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tool"] == "dpf first-principles-3d"
    assert payload["deck"]["source"] == str(deck)
    assert payload["deck"]["name"] == "tiny_test_deck"
    assert payload["deck"]["steps"] == 1
    assert payload["deck"]["grid_shape"] == [4, 4, 4]
    assert payload["deck"]["apply_circuit_boundary"] is False
    assert payload["n_steps"] == 1
