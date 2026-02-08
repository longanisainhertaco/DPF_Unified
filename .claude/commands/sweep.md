Run a parameter sweep to generate WALRUS training data.

Given $ARGUMENTS (a sweep config JSON path or parameter description), generate a batch of DPF simulations with Latin Hypercube sampling across the specified parameter ranges.

Steps:
1. Parse the sweep config (base_config, parameter_ranges, n_samples)
2. Create a BatchRunner from dpf.ai.batch_runner
3. Run the sweep with parallel workers
4. Report results: n_success/n_total, output directory, any failures

Key files:
- src/dpf/ai/batch_runner.py — BatchRunner, ParameterRange
- src/dpf/ai/well_exporter.py — exports each run to Well format
- src/dpf/cli/main.py — `sweep` CLI command

Use `dpf sweep <sweep_config.json> -o sweep_output -w 4` for CLI usage.
