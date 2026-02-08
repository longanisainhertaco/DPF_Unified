Run inverse design to find DPF configurations matching target outputs.

Given $ARGUMENTS (a targets JSON file and checkpoint path), use WALRUS surrogate + optimization to find device parameters that achieve desired plasma performance.

Steps:
1. Load targets and constraints from JSON
2. Load WALRUS surrogate model
3. Run Bayesian (optuna) or evolutionary (scipy) optimization
4. Report best parameters and score

Key files:
- src/dpf/ai/inverse_design.py — InverseDesigner, InverseResult
- src/dpf/ai/surrogate.py — DPFSurrogate (objective evaluator)
- src/dpf/cli/main.py — `inverse` CLI command

Use `dpf inverse <targets.json> --checkpoint model.pt --method bayesian --n-trials 100` for CLI usage.

Targets JSON format:
```json
{
    "targets": {"max_Te": 1e7, "neutron_yield": 1e8},
    "constraints": {"max_current": 500000},
    "parameter_ranges": {"circuit.V0": [10000, 50000], "circuit.C": [1e-6, 100e-6]}
}
```
