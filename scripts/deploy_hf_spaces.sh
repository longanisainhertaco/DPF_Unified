#!/usr/bin/env bash
set -euo pipefail

# Deploy DPF-Unified to HuggingFace Spaces
# Usage: bash scripts/deploy_hf_spaces.sh [space-name]
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login  (or set HF_TOKEN env var)

SPACE_NAME="${1:-dpf-unified}"
HF_USER=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null) || {
    echo "Not logged in to HuggingFace. Run:"
    echo "  python3 -c \"from huggingface_hub import login; login()\""
    echo "  or set HF_TOKEN environment variable"
    exit 1
}

REPO_ID="${HF_USER}/${SPACE_NAME}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR=$(mktemp -d)

echo "Deploying to https://huggingface.co/spaces/${REPO_ID}"

# Create the Space repo if it doesn't exist
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.repo_info(repo_id='${REPO_ID}', repo_type='space')
    print('Space exists, updating...')
except Exception:
    api.create_repo(repo_id='${REPO_ID}', repo_type='space', space_sdk='gradio', private=False)
    print('Space created.')
"

# Copy files needed for the Space
cp "$REPO_ROOT"/app*.py "$DEPLOY_DIR/"
cp "$REPO_ROOT/requirements.txt" "$DEPLOY_DIR/"
cp -r "$REPO_ROOT/src" "$DEPLOY_DIR/"

# Create Space README with metadata
cat > "$DEPLOY_DIR/README.md" << 'SPACE_README'
---
title: DPF-Unified Simulator
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# DPF-Unified — Dense Plasma Focus Simulator

Multi-fidelity simulation with Lee model (0D), Metal GPU MHD, and Athena++ C++ backends.
8 device presets from 1.85 kJ to 2 MJ. Auto-calibration, parameter sweeps, 3D playback.

[GitHub](https://github.com/longanisainhertaco/DPF_Unified)
SPACE_README

# Upload to HF
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='${DEPLOY_DIR}',
    repo_id='${REPO_ID}',
    repo_type='space',
    commit_message='Deploy DPF-Unified v1.0',
)
print('Deployed! https://huggingface.co/spaces/${REPO_ID}')
"

rm -rf "$DEPLOY_DIR"
echo "Done. Space will build at: https://huggingface.co/spaces/${REPO_ID}"
