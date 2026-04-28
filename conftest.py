import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# Allow duplicate OpenMP runtime to coexist (numpy/torch/scipy ship one, the
# Athena++ pybind11 extension links another). Without this, on macOS the
# second libomp init calls abort(), segfaulting the test process when any
# test instantiates AthenaPPSolver in linked mode. Must be set BEFORE
# numpy/torch/scipy import, which is why this lives in the root conftest.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

pytest_plugins = ["dpf.testing.progress"]
