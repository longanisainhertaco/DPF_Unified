#!/usr/bin/env bash
# Phase K — Development mode launcher.
#
# Starts the Python DPF backend and the Vite dev server concurrently.
# The Electron main process spawns the backend automatically, so this
# script is for running the renderer independently during development.
#
# Usage:
#   bash gui/scripts/dev.sh
#   # or from gui/:
#   npm run dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$GUI_DIR")"

cd "$GUI_DIR"

echo "=== DPF Desktop GUI — Development Mode ==="
echo "  Project root: $PROJECT_ROOT"
echo "  GUI dir:      $GUI_DIR"
echo ""

# Check that dpf CLI is available
if ! command -v dpf &>/dev/null; then
  echo "[WARNING] 'dpf' CLI not found in PATH."
  echo "  Install with: pip install -e \"$PROJECT_ROOT[dev]\""
  echo "  Starting Vite dev server only (no Python backend)..."
  echo ""
  npx vite --config vite.config.ts
  exit 0
fi

# Start Python backend + Vite + Electron concurrently
echo "Starting Python backend (port 8765) + Vite dev server + Electron..."
echo ""

npx concurrently \
  --names "backend,vite,electron" \
  --prefix-colors "yellow,green,blue" \
  --kill-others \
  "dpf serve --port 8765" \
  "npx vite --config vite.config.ts" \
  "sleep 3 && NODE_ENV=development npx electron dist/main/index.js"
