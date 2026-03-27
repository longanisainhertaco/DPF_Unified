#!/bin/bash
# DPF-Unified Pre-Commit Quality Gate
# Sprint S-3 Task 1.1 — BLOCKS commits that fail any check.
#
# Install: ln -sf ../../scripts/pre_commit_check.sh .git/hooks/pre-commit
#
# Checks (derived from Sprint S-1/S-2 rework analysis — each caught a real bug):
# 1. Inline physics constants (must use constants.py)
# 2. Imports from top-level app_* modules in library code
# 3. New stencil operators without analytical tests
# 4. Module tests pass for changed files
# 5. No hardcoded _C_BORIS_SQ, P_FLOOR, RHO_FLOOR outside constants.py

set -e

ERRORS=0
WARNINGS=0

fail() {
    echo "FAIL: $1"
    ERRORS=$((ERRORS + 1))
}

warn() {
    echo "WARN: $1"
    WARNINGS=$((WARNINGS + 1))
}

echo "=== DPF Pre-Commit Quality Gate ==="

# Get list of staged Python files in src/dpf/metal/
STAGED_METAL=$(git diff --cached --name-only --diff-filter=ACM | grep '^src/dpf/metal/.*\.py$' || true)
STAGED_TESTS=$(git diff --cached --name-only --diff-filter=ACM | grep '^tests/.*\.py$' || true)
STAGED_ALL=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

# ── Check 1: Inline physics constants ────────────────────────────
# After Task 1.2, all constants must come from constants.py
if [ -n "$STAGED_METAL" ]; then
    for f in $STAGED_METAL; do
        if [ "$f" = "src/dpf/metal/constants.py" ]; then
            continue
        fi
        # Check for inline _C_BORIS_SQ definitions (not imports)
        if git diff --cached -U0 -- "$f" | grep '^+' | grep -v '^+++' | grep -q '_C_BORIS_SQ\s*=\s*[0-9]'; then
            fail "$f: Inline _C_BORIS_SQ definition. Import from dpf.metal.constants instead."
        fi
        # Check for inline P_FLOOR/RHO_FLOOR definitions (not imports)
        if git diff --cached -U0 -- "$f" | grep '^+' | grep -v '^+++' | grep -qE '(P_FLOOR|RHO_FLOOR)\s*[:=]\s*1e-12'; then
            fail "$f: Inline P_FLOOR/RHO_FLOOR definition. Import from dpf.metal.constants instead."
        fi
        # Check for inline MU_0 definitions
        if git diff --cached -U0 -- "$f" | grep '^+' | grep -v '^+++' | grep -qE 'MU_0\s*[:=]\s*4.*pi.*1e-7'; then
            fail "$f: Inline MU_0 definition. Import from dpf.metal.constants instead."
        fi
    done
fi

# ── Check 2: No imports from app_* in library code ──────────────
if [ -n "$STAGED_METAL" ] || echo "$STAGED_ALL" | grep -q '^src/dpf/'; then
    OFFENDING=$(git diff --cached -U0 -- src/dpf/ | grep '^+' | grep -v '^+++' | grep 'from app_' || true)
    if [ -n "$OFFENDING" ]; then
        fail "Import from app_* module in library code. Inline the dependency."
        echo "  $OFFENDING"
    fi
fi

# ── Check 3: New stencil operators need analytical tests ─────────
NEW_STENCILS=$(git diff --cached -U0 -- src/dpf/ | grep '^+' | grep -v '^+++' | grep -E 'def [a-z_]*_rhs\b|def [a-z_]*laplacian\b|def [a-z_]*diffusion\b' || true)
if [ -n "$NEW_STENCILS" ]; then
    # Check if corresponding test file has analytical verification
    HAS_ANALYTICAL=$(git diff --cached -U0 -- tests/ | grep '^+' | grep -v '^+++' | grep -c 'analytical\|Laplacian.*==\|expect.*4\.0\|expect.*0\.0' || true)
    if [ "$HAS_ANALYTICAL" -eq 0 ]; then
        warn "New stencil operator detected but no analytical test found in staged changes."
        echo "  Stencil: $NEW_STENCILS"
        echo "  Required: test with Laplacian(r^2)=4, Laplacian(uniform)=0"
    fi
fi

# ── Check 4: Run tests for changed modules ──────────────────────
if [ -n "$STAGED_METAL" ] || [ -n "$STAGED_TESTS" ]; then
    # Find test files matching changed source files
    TEST_FILES=""
    for f in $STAGED_METAL; do
        basename=$(basename "$f" .py)
        test_file="tests/test_${basename}.py"
        if [ -f "$test_file" ]; then
            TEST_FILES="$TEST_FILES $test_file"
        fi
    done
    # Also run any staged test files
    for f in $STAGED_TESTS; do
        if [ -f "$f" ]; then
            TEST_FILES="$TEST_FILES $f"
        fi
    done

    if [ -n "$TEST_FILES" ]; then
        echo "Running module tests: $TEST_FILES"
        if ! python3 -m pytest $TEST_FILES -x -q --tb=line -m "not slow" 2>&1 | tail -3; then
            fail "Module tests failed. Fix before committing."
        fi
    fi
fi

# ── Check 5: Constants consistency ───────────────────────────────
if echo "$STAGED_ALL" | grep -q 'constants.py\|mlx_kernels.py'; then
    echo "Constants or kernels changed — running consistency test..."
    if ! python3 -m pytest tests/test_constants_consistency.py -x -q --tb=short 2>&1 | tail -3; then
        fail "Constants consistency test failed. Metal shader values may not match Python."
    fi
fi

# ── Summary ──────────────────────────────────────────────────────
echo ""
if [ $ERRORS -gt 0 ]; then
    echo "BLOCKED: $ERRORS error(s), $WARNINGS warning(s). Fix errors before committing."
    exit 1
fi

if [ $WARNINGS -gt 0 ]; then
    echo "PASSED with $WARNINGS warning(s)."
else
    echo "PASSED: All checks clean."
fi
exit 0
