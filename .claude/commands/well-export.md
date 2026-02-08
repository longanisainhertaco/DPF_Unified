Export DPF simulation data to Well format for WALRUS training.

Given $ARGUMENTS (a config file path, or use a default preset), run a DPF simulation and export the trajectory to Well HDF5 format suitable for WALRUS fine-tuning.

Steps:
1. Load the config file and create a SimulationEngine
2. Use WellExporter from dpf.ai.well_exporter to accumulate field snapshots
3. Run the simulation, capturing snapshots every N steps (default 10)
4. Finalize the Well file and report statistics (n_snapshots, file size, fields exported)

Key files:
- src/dpf/ai/well_exporter.py — WellExporter class
- src/dpf/ai/field_mapping.py — DPF↔Well field mapping
- src/dpf/cli/main.py — `export-well` CLI command

Use `dpf export-well <config> -o output.h5 --field-interval 10` for CLI usage.
