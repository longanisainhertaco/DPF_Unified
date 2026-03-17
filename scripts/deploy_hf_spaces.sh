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
VERSION=$(git -C "$REPO_ROOT" describe --tags --always 2>/dev/null || echo "dev")

echo "Deploying ${VERSION} to https://huggingface.co/spaces/${REPO_ID}"

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

# Copy app files
cp "$REPO_ROOT"/app*.py "$DEPLOY_DIR/"

# Copy src/ tree excluding heavy files (WALRUS checkpoints, __pycache__, binaries)
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='*.so' \
    --exclude='*.pt' --exclude='*.bin' --exclude='*.h5' --exclude='*.npz' \
    --exclude='models/' --exclude='*.metal' --exclude='*.o' --exclude='*.a' \
    --exclude='athena_wrapper/cpp/' \
    "$REPO_ROOT/src/" "$DEPLOY_DIR/src/"

# Requirements: pin versions that work on HF Spaces (CPU-only, no torch/Metal)
cat > "$DEPLOY_DIR/requirements.txt" << 'REQS'
numpy>=1.24,<2.0
scipy>=1.10
pydantic>=2.0
numba>=0.57
h5py>=3.8
plotly>=5.15
REQS

# Create Space README with metadata
cat > "$DEPLOY_DIR/README.md" << SPACE_README
---
title: DPF-Unified Simulator
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
---

# DPF-Unified — Dense Plasma Focus Simulator (${VERSION})

Multi-fidelity MHD simulation: Lee model (0D), Metal GPU MHD, Athena++ C++.
8 device presets from 400 J to 1 MJ. Advanced physics modules (FLD, sheath BC,
ablation, Nernst advection, CR ionization). Parameter sweeps, 3D playback,
reconnection diagnostics, plasmoid detection, beam-ion tracking.

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
    commit_message='Deploy DPF-Unified ${VERSION}',
)
print('Deployed! https://huggingface.co/spaces/${REPO_ID}')
"

rm -rf "$DEPLOY_DIR"
echo "Done. Space will build at: https://huggingface.co/spaces/${REPO_ID}"
