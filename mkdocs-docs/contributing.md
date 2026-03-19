# Contributing

See [CONTRIBUTING.md](https://github.com/anthonyzamora/dpf-unified/blob/main/CONTRIBUTING.md) for full guidelines.

## Development Setup

```bash
git clone https://github.com/anthonyzamora/dpf-unified
cd dpf-unified
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -x -q                    # fast suite
pytest tests/ -x -q -m slow           # slow suite (30+ min)
pytest tests/ --cov=src/dpf           # with coverage
```

## Code Style

- Line length: 100 chars (`ruff check src/ tests/`)
- Physics names exempt from naming conventions: `Te, Ti, B, rho, eta`
- NumPy-style docstrings on all public functions

## Commit Convention

```
feat: add WENO5-Z reconstruction
fix: clamp velocity after snowplow handoff
refactor: split cylindrical MHD into sub-modules
test: add Brio-Wu shock convergence test
docs: update API reference for SimulationConfig
```

## Building Docs

```bash
pip install "dpf-unified[docs]"
mkdocs serve        # local preview at http://localhost:8000
mkdocs build        # build static site to site/
```
