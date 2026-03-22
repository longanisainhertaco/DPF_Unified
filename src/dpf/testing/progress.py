import json
import time
from pathlib import Path

_SIDECAR = Path("/tmp/dpf-test-progress.json")

_state: dict = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "total": 0,
    "current": None,
    "elapsed_s": 0.0,
    "pct": 0.0,
    "last_failure": None,
    "_start": None,
}


def pytest_collection_modifyitems(items):
    _state["total"] = len(items)
    _state["_start"] = time.monotonic()
    _flush()


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return

    _state["current"] = report.nodeid

    if report.passed:
        _state["passed"] += 1
    elif report.failed:
        _state["failed"] += 1
        _state["last_failure"] = report.nodeid
    elif report.skipped:
        _state["skipped"] += 1

    start = _state["_start"] or time.monotonic()
    _state["elapsed_s"] = round(time.monotonic() - start, 2)

    done = _state["passed"] + _state["failed"] + _state["skipped"]
    total = _state["total"] or 1
    _state["pct"] = round(done / total * 100, 1)

    _flush()


def _flush():
    payload = {k: v for k, v in _state.items() if not k.startswith("_")}
    _SIDECAR.write_text(json.dumps(payload))
