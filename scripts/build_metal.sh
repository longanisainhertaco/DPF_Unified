#!/bin/bash
set -e

# Directory paths
KERNEL_DIR="src/dpf/metal/kernels"
BUILD_DIR="src/dpf/metal/build"

# Ensure build directory exists
mkdir -p "$BUILD_DIR"

echo "Compiling Metal kernels from $KERNEL_DIR to $BUILD_DIR..."

# Compile each .metal file to .air
AIR_FILES=""
for metal_file in "$KERNEL_DIR"/*.metal; do
    filename=$(basename -- "$metal_file")
    name="${filename%.*}"
    air_file="$BUILD_DIR/$name.air"
    
    echo "  Compiling $filename..."
    xcrun -sdk macosx metal -c "$metal_file" -o "$air_file"
    AIR_FILES="$AIR_FILES $air_file"
done

# Link all .air files to .metallib
echo "Linking to default.metallib..."
xcrun -sdk macosx metallib $AIR_FILES -o "$BUILD_DIR/default.metallib"

echo "Done."
