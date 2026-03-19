# Contributing to DPF-Unified

## Setup

```bash
git clone https://github.com/longanisainhertaco/DPF_Unified.git
cd DPF_Unified
pip install -e ".[dev]"
```

## Running Tests

```bash
# Fast suite (< 2 min)
pytest tests/ -x -q -m "not slow"

# Full suite including slow tests
pytest tests/ -x -q
```

## Code Style

- `ruff check src/ tests/` must pass
- Type hints on all public functions
- NumPy-style docstrings
- 100-char line length
- Conventional commits: feat:, fix:, refactor:, test:, docs:, chore:

## Adding a Device Preset

1. Add the preset dict to `src/dpf/presets.py`
2. Add published experimental data to `src/dpf/validation/experimental.py`
3. Run validation and add a test

## Reporting Bugs

Open a GitHub Issue with: what you expected, what happened, steps to reproduce, device preset and backend used.

## Physics Changes

Before modifying any physics model:
1. Write the governing equation in the PR description
2. Trace all term signs
3. Grep all callers of any property you mutate
4. Run the validation suite before and after
