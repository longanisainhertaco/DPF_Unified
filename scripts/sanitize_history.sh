#!/bin/bash
# Sanitize git history: remove TB/engineer content from all commits.
# Run AFTER backing up and AFTER any in-flight processes complete.
#
# Usage: bash scripts/sanitize_history.sh

set -e

echo "=== DPF-Unified History Sanitization ==="
echo "This will rewrite ALL git history to remove engineer/TB content."
echo ""

# Files/dirs to remove from history
git filter-repo \
    --path-glob 'docs/engineer-*' \
    --path-glob 'docs/tb-*' \
    --path-glob 'docs/*TB*' \
    --path-glob 'docs/*operator_training*' \
    --path-glob 'docs/*risk_register*' \
    --path-glob 'docs/*pre_shot*' \
    --path-glob 'docs/*Verus*' \
    --path-glob 'docs/*Orthus*' \
    --path-glob 'training/weapon-corpus/*' \
    --path-glob 'static/dpf_shot_*.wav' \
    --invert-paths \
    --force

echo ""
echo "History rewritten. Next steps:"
echo "  1. Verify: git log --oneline | head -20"
echo "  2. Re-add remotes: git remote add origin git@github.com:longanisainhertaco/DPF_Unified.git"
echo "  3. Force push: git push origin main --force"
echo ""
echo "Done."
