Benchmark AI surrogate performance against full physics simulation.

Given $ARGUMENTS (a config file, checkpoint, and comparison parameters), compare WALRUS surrogate speed and accuracy against the full DPF physics engine.

Steps:
1. Run full physics simulation for reference trajectory
2. Run surrogate rollout from same initial conditions
3. Compute per-field L2 divergence and timing comparison
4. Use InstabilityDetector to identify divergence onset
5. Report speedup factor, accuracy metrics, and instability events

Key files:
- src/dpf/ai/surrogate.py — DPFSurrogate
- src/dpf/ai/instability_detector.py — InstabilityDetector
- src/dpf/ai/confidence.py — EnsemblePredictor (uncertainty)
- src/dpf/ai/hybrid_engine.py — HybridEngine (combined approach)

Metrics reported:
- Wall-clock speedup (surrogate vs physics)
- Per-field normalized L2 error over time
- Instability onset step and severity
- Confidence/OOD scores from ensemble
