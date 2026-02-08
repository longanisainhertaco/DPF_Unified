Validate a Well-format training dataset for WALRUS.

Given $ARGUMENTS (a directory path containing Well HDF5 files), validate the dataset for correctness, completeness, and quality.

Steps:
1. Use DatasetValidator from dpf.ai.dataset_validator
2. Validate each .h5 file: schema, NaN/Inf checks, energy conservation, field statistics
3. Generate a summary report with pass/fail counts and warnings

Key files:
- src/dpf/ai/dataset_validator.py — DatasetValidator, ValidationResult
- src/dpf/cli/main.py — `validate-dataset` CLI command

Use `dpf validate-dataset <directory>` for CLI usage.
