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

# Copy app files (all app_*.py modules required by app.py)
cp "$REPO_ROOT"/app*.py "$DEPLOY_DIR/"

# Copy src/ tree excluding:
#   - __pycache__, bytecode, compiled extensions
#   - WALRUS/ML model checkpoints (too large: .pt, .bin, .h5, .npz)
#   - Apple Metal GPU shader source (.metal files — Linux HF Spaces has no Metal)
#   - Athena++ C++ pybind11 bindings (requires compiled binary not available on HF Spaces)
#   - AthenaK C++ sources
#   - Build artifacts
# The Python Metal solver (src/dpf/metal/*.py) IS included — it falls back to CPU via PyTorch.
# The Lee model, hybrid, python, metal_plm/weno5/3d backends all work CPU-only.
rsync -a \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.so' \
    --exclude='*.pt' --exclude='*.bin' --exclude='*.h5' --exclude='*.npz' \
    --exclude='models/' \
    --exclude='*.metal' \
    --exclude='*.o' --exclude='*.a' \
    --exclude='athena_wrapper/cpp/' \
    --exclude='athenak_wrapper/' \
    "$REPO_ROOT/src/" "$DEPLOY_DIR/src/"

# Requirements for HF Spaces (CPU-only Linux environment)
# - gradio: required (was missing in original script)
# - torch (CPU): required for MetalMHDSolver CPU fallback (metal_plm, metal_weno5, hybrid)
# - click: required by dpf CLI entry points imported by app modules
# - tqdm: used by engine runner
# - numba: Python MHD engine acceleration
# Note: torch CPU wheel is ~200 MB; HF Spaces caches it between builds.
cat > "$DEPLOY_DIR/requirements.txt" << 'REQS'
gradio>=5.0,<7.0
numpy>=1.24,<2.0
scipy>=1.10
pydantic>=2.0
numba>=0.57
h5py>=3.8
plotly>=5.15
click>=8.0
tqdm>=4.60
torch>=2.4 --index-url https://download.pytorch.org/whl/cpu
REQS

# Create Space README with HF Spaces YAML header (required for Space to build correctly)
# sdk_version must match the gradio version in requirements.txt
cat > "$DEPLOY_DIR/README.md" << SPACE_README
---
title: DPF-Unified Simulator
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: false
license: mit
tags:
  - physics
  - plasma
  - simulation
  - mhd
  - fusion
  - dense-plasma-focus
  - neutron
  - education
short_description: "Free Dense Plasma Focus simulator — Lee model to MHD"
variables:
  DPF_DEFAULT_PRESET: tutorial
---

# DPF-Unified — Dense Plasma Focus Simulator (${VERSION})

Interactive web simulator for Dense Plasma Focus (DPF) devices.
Multi-fidelity MHD simulation: Lee model (0D, validated against 7+ published devices),
hybrid Lee+MHD, and full 2D/3D MHD backends. 8 device presets from 3 kJ to 1 MJ.

**Start with the Tutorial Device preset** — small, fast, and built for learning.

## Features
- Lee model (0D): validated current waveforms, D-D neutron yield, < 1 second
- Hybrid Lee+MHD: validated waveforms + spatially resolved plasma compression
- 2D MHD (PLM): full density/pressure/B-field maps on CPU
- 2D MHD (WENO5+HLLD+float64): publication-quality 5th-order solver
- 3D MHD: kink instability, filamentation, azimuthal asymmetries
- Physics Narrative: step-by-step LaTeX derivation for every run
- Parameter sweeps, calibration, CSV export, comparison mode

## Backends Available on HF Spaces (CPU-only)
| Backend | Method | Runtime |
|---------|--------|---------|
| Lee (Quick) | 0D snowplow ODEs | < 1 sec |
| Hybrid (Standard) | Lee + 2D MHD CPU | 3-30 sec |
| Detailed (2D MHD PLM) | PyTorch CPU | 15-90 sec |
| High Accuracy (WENO5) | PyTorch float64 CPU | 30-120 sec |
| 3D MHD | PyTorch CPU | 2-15 min |

*Note: Apple Metal GPU acceleration is disabled on HF Spaces (Linux CPU-only). PyTorch CPU fallback is used automatically.*

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
