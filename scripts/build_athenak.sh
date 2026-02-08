#!/bin/bash
# Build AthenaK for DPF Unified
#
# Usage:
#   bash scripts/build_athenak.sh              # Auto-detect (prefer OpenMP)
#   bash scripts/build_athenak.sh serial       # Force Serial backend
#   bash scripts/build_athenak.sh openmp       # Force OpenMP backend
#   bash scripts/build_athenak.sh blast        # Build blast problem generator
#
# Requires: CMake >= 3.16, C++17 compiler
# OpenMP requires: Homebrew LLVM (brew install llvm)
set -e

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATHENAK_DIR="$PROJ_ROOT/external/athenak"
BUILD_MODE="${1:-auto}"

if [ ! -f "$ATHENAK_DIR/CMakeLists.txt" ]; then
    echo "ERROR: AthenaK not found at $ATHENAK_DIR"
    echo "Run: bash scripts/setup_athenak.sh"
    exit 1
fi

if [ ! -f "$ATHENAK_DIR/kokkos/CMakeLists.txt" ]; then
    echo "ERROR: Kokkos submodule not initialized"
    echo "Run: cd $ATHENAK_DIR && git submodule update --init --depth=1"
    exit 1
fi

cd "$ATHENAK_DIR"

# Detect platform and compiler
SYSROOT="$(xcrun --show-sdk-path 2>/dev/null || echo "")"
BREW_LLVM="/opt/homebrew/opt/llvm/bin/clang++"
NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

# Determine Kokkos backend
HAS_OPENMP=false
if [ -x "$BREW_LLVM" ]; then
    HAS_OPENMP=true
fi

case "$BUILD_MODE" in
    serial)
        USE_OPENMP=false
        ;;
    openmp)
        if [ "$HAS_OPENMP" = false ]; then
            echo "ERROR: OpenMP requires Homebrew LLVM: brew install llvm"
            exit 1
        fi
        USE_OPENMP=true
        ;;
    blast)
        # Build blast problem generator (custom pgen)
        PROBLEM_FLAG="-D PROBLEM=blast"
        USE_OPENMP=$HAS_OPENMP
        ;;
    auto|*)
        USE_OPENMP=$HAS_OPENMP
        ;;
esac

# Set up CMake options
CMAKE_OPTS="-D Kokkos_ARCH_ARMV81=On"
BUILD_DIR="build_serial"

if [ "$USE_OPENMP" = true ]; then
    CMAKE_OPTS="$CMAKE_OPTS -D CMAKE_CXX_COMPILER=$BREW_LLVM"
    CMAKE_OPTS="$CMAKE_OPTS -D Kokkos_ENABLE_OPENMP=On"
    if [ -n "$SYSROOT" ]; then
        CMAKE_OPTS="$CMAKE_OPTS -D CMAKE_CXX_FLAGS=--sysroot=$SYSROOT"
    fi
    BUILD_DIR="build_omp"
fi

if [ -n "$PROBLEM_FLAG" ]; then
    CMAKE_OPTS="$CMAKE_OPTS $PROBLEM_FLAG"
    BUILD_DIR="${BUILD_DIR}_blast"
fi

echo "=== Building AthenaK ==="
echo "  Source:   $ATHENAK_DIR"
echo "  Build:    $BUILD_DIR"
echo "  OpenMP:   $USE_OPENMP"
echo "  Problem:  ${PROBLEM_FLAG:-built-in (runtime pgen_name)}"
echo "  Parallel: $NPROC cores"
echo ""

# Configure
cmake -B "$BUILD_DIR" $CMAKE_OPTS 2>&1 | tail -5

# Build
cmake --build "$BUILD_DIR" -j"$NPROC" 2>&1 | tail -5

# Verify binary
BINARY="$BUILD_DIR/src/athena"
if [ -x "$BINARY" ]; then
    echo ""
    echo "=== Build successful ==="
    echo "  Binary: $ATHENAK_DIR/$BINARY"
    echo "  Size:   $(du -h "$BINARY" | cut -f1)"

    # Create symlink in bin/ for easy access
    mkdir -p "$ATHENAK_DIR/bin"
    if [ -n "$PROBLEM_FLAG" ]; then
        ln -sf "../$BINARY" "$ATHENAK_DIR/bin/athenak_blast"
        echo "  Symlink: bin/athenak_blast"
    else
        ln -sf "../$BINARY" "$ATHENAK_DIR/bin/athenak"
        echo "  Symlink: bin/athenak"
    fi
else
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi
