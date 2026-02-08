#!/bin/bash
# Setup AthenaK submodule for DPF Unified
# Usage: bash scripts/setup_athenak.sh
set -e

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"

echo "=== DPF Unified: AthenaK Setup ==="

# Step 1: Add submodule if not already present
if [ ! -d "external/athenak/.git" ] && [ ! -f "external/athenak/.git" ]; then
    echo "Adding AthenaK as git submodule..."
    git submodule add --depth=1 https://github.com/IAS-Astrophysics/athenak.git external/athenak
else
    echo "AthenaK submodule already present."
fi

# Step 2: Initialize Kokkos submodule inside AthenaK
cd external/athenak
if [ ! -f "kokkos/CMakeLists.txt" ]; then
    echo "Initializing Kokkos submodule..."
    git submodule update --init --depth=1
else
    echo "Kokkos submodule already initialized."
fi

echo "=== AthenaK setup complete ==="
echo "Next step: run scripts/build_athenak.sh to compile"
