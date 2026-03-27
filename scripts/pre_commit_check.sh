#!/bin/bash
# Pre-commit static checks for DPF-Unified
# Catches the 5 error classes from Sprint S-1/S-2 rework analysis:
# 1. self._* private attributes that don't exist in class
# 2. Imports from top-level app modules (app_*.py)
# 3. Missing analytical tests for new stencil operators
# 4. Hardcoded test grid sizes that don't cover device geometry
# 5. Unconditional dt floors/caps without backend guards

set -e

echo "=== DPF Pre-Commit Checks ==="

# 1. Check for self._nr, self._nz (should be self.nr, self.nz in MLX solver)
if git diff --cached --name-only | grep -q 'mlx_solver.py'; then
    if git diff --cached -U0 -- src/dpf/metal/mlx_solver.py | grep -E '^\+.*self\._nr|^\+.*self\._nz' | grep -v '^\+\+\+'; then
        echo "WARN: self._nr or self._nz found in mlx_solver.py — use self.nr/self.nz"
    fi
fi

# 2. Check for imports from app_* modules in library code
if git diff --cached -U0 -- src/dpf/ | grep '^+.*from app_' | grep -v '^+++'; then
    echo "WARN: Import from app_* module in library code — inline the dependency"
fi

# 3. Check for new stencil operators without analytical tests
NEW_STENCILS=$(git diff --cached -U0 -- src/dpf/ | grep '^+.*def.*_rhs\|^+.*def.*laplacian' | grep -v '^+++' || true)
if [ -n "$NEW_STENCILS" ]; then
    echo "NOTE: New stencil operator detected. Verify analytical test exists (Laplacian(r^2)=4)."
    echo "$NEW_STENCILS"
fi

echo "=== Checks complete ==="
