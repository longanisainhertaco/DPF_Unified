PYTHON = /opt/homebrew/Cellar/python@3.11/3.11.11/bin/python3.11

.PHONY: test test-fast test-slow test-slow-parallel test-all test-metal test-dashboard lint

test:  ## Run non-slow tests
	$(PYTHON) -m pytest tests/ -x -q -m "not slow"

test-fast:  ## Run non-slow tests in parallel
	$(PYTHON) -m pytest tests/ -x -q -m "not slow" -n auto

test-slow:  ## Run slow tests only (serial)
	$(PYTHON) -m pytest tests/ -x -q -m slow

test-slow-parallel:  ## Run slow tests in parallel (4 workers)
	$(PYTHON) -m pytest tests/ -x -q -m slow -n 4 --dist loadgroup

test-all:  ## Run all tests in parallel
	$(PYTHON) -m pytest tests/ -q -n auto

test-metal:  ## Run Metal GPU tests only
	$(PYTHON) -m pytest tests/test_metal_production.py tests/test_phase_o_physics_accuracy.py tests/test_phase_n_cross_backend.py -x -q -m slow

test-dashboard:  ## Launch visual test dashboard
	$(PYTHON) tools/test_dashboard.py

lint:  ## Run linter
	$(PYTHON) -m ruff check src/ tests/
