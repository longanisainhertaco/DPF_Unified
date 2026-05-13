from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_probe_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mlx_pf1000_probe.py"
    spec = importlib.util.spec_from_file_location("run_mlx_pf1000_probe_policy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_mlx_probe_policy_is_engineering_regression_only() -> None:
    module = _load_probe_script()

    assert module._ENDURANCE_POLICY == {
        "lane": "endurance_regression",
        "scientific_status": "non_scientific",
        "source_status": "s1_s2_source_closure_blocked",
        "opt_in_env": "DPF_MLX_RUN_ENDURANCE",
    }


def test_pytest_mlx_probe_declares_source_blocked_policy() -> None:
    path = Path(__file__).resolve().parent / "test_mlx_pf1000_probe.py"
    source = path.read_text()

    assert '"scientific_status": "non_scientific"' in source
    assert '"source_status": "s1_s2_source_closure_blocked"' in source
    assert "Scientific PF-1000 gates stay" in source
