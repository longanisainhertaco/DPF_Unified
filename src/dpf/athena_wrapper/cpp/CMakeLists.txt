cmake_minimum_required(VERSION 3.18)
project(_athena_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Find dependencies
# ============================================================

# pybind11
find_package(pybind11 REQUIRED)

# HDF5 (optional but recommended)
find_package(HDF5 COMPONENTS C)

# ============================================================
# Athena++ source configuration
# ============================================================

# ATHENA_ROOT should point to external/athena
if(NOT DEFINED ATHENA_ROOT)
    set(ATHENA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../external/athena")
endif()

# Verify Athena++ exists
if(NOT EXISTS "${ATHENA_ROOT}/src/athena.hpp")
    message(FATAL_ERROR
        "Athena++ not found at ${ATHENA_ROOT}. "
        "Set -DATHENA_ROOT=<path-to-external/athena>")
endif()

message(STATUS "Athena++ root: ${ATHENA_ROOT}")

# Athena++ must be configured first (generates defs.hpp)
if(NOT EXISTS "${ATHENA_ROOT}/src/defs.hpp")
    message(FATAL_ERROR
        "Athena++ not configured. Run:\n"
        "  cd ${ATHENA_ROOT}\n"
        "  python3 configure.py --prob=magnoh --coord=cylindrical -b "
        "--flux=hlld --hdf5_path=/opt/homebrew/opt/hdf5\n"
        "  make -j8")
endif()

# ============================================================
# Collect Athena++ object files
# ============================================================

# Rather than recompiling all of Athena++, we link against
# the pre-compiled object files from `make`.
# This avoids duplicating the complex Athena++ build system.
file(GLOB ATHENA_OBJECTS "${ATHENA_ROOT}/obj/*.o")

if(NOT ATHENA_OBJECTS)
    message(FATAL_ERROR
        "No Athena++ object files found in ${ATHENA_ROOT}/obj/. "
        "Build Athena++ first with: cd ${ATHENA_ROOT} && make -j8")
endif()

# Filter out main.o — we don't want Athena++'s main()
list(FILTER ATHENA_OBJECTS EXCLUDE REGEX "main\\.o$")

message(STATUS "Found ${CMAKE_LISTS_LENGTH} Athena++ objects (excluding main.o)")
list(LENGTH ATHENA_OBJECTS N_OBJ)
message(STATUS "Athena++ object files: ${N_OBJ}")

# ============================================================
# Build the pybind11 module
# ============================================================

pybind11_add_module(_athena_core
    athena_bindings.cpp
)

# Link Athena++ objects
target_link_libraries(_athena_core PRIVATE
    ${ATHENA_OBJECTS}
)

# Include Athena++ headers
target_include_directories(_athena_core PRIVATE
    "${ATHENA_ROOT}/src"
)

# HDF5 support (matches Athena++ build)
# Note: HDF5OUTPUT is already defined in Athena++ defs.hpp, don't redefine
if(HDF5_FOUND)
    target_include_directories(_athena_core PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(_athena_core PRIVATE ${HDF5_LIBRARIES})
endif()

# OpenMP support (matches Athena++ -omp flag)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_athena_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compile definitions matching Athena++ configuration
# These are read from defs.hpp at compile time via includes

# Optimization flags
target_compile_options(_athena_core PRIVATE -O3)

# ============================================================
# Install target
# ============================================================

# Install the compiled module into the Python package
install(TARGETS _athena_core
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Also provide a convenience copy target
add_custom_command(TARGET _athena_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:_athena_core>
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    COMMENT "Copying _athena_core to athena_wrapper package"
)
