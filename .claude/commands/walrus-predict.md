Run WALRUS surrogate prediction for a DPF configuration.

Given $ARGUMENTS (a config file and checkpoint path), run a WALRUS surrogate model to predict DPF plasma evolution.

Steps:
1. Load the DPF config and WALRUS checkpoint
2. Run a short physics simulation to generate initial history states
3. Execute surrogate rollout for the specified number of steps
4. Report prediction results and timing

Key files:
- src/dpf/ai/surrogate.py — DPFSurrogate class
- src/dpf/ai/hybrid_engine.py — HybridEngine (physics + surrogate)
- src/dpf/ai/confidence.py — EnsemblePredictor for uncertainty
- src/dpf/cli/main.py — `predict` CLI command

Use `dpf predict <config> --checkpoint model.pt --steps 100 --device cpu` for CLI usage.
