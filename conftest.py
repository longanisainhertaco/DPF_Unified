import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

pytest_plugins = ["dpf.testing.progress"]
