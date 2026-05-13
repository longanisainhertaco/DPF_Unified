# Air-Gap Release Gate

Status: fail-closed release gate definition
Date: 2026-05-08

## Required Artifacts

The release is not air-gap-ready until these artifacts exist and are reviewed:

- `pyproject.toml`
- `requirements.txt`
- `gui/package-lock.json`
- `dist/wheelhouse/`
- `dist/wheelhouse/SHA256SUMS`
- `docs/airgap_logs/python-offline-smoke.log`
- `docs/airgap_logs/gui-offline-typecheck.log`

## Offline Commands

Run these in a clean environment with network disabled:

```bash
python3 -m pip install --no-index --find-links dist/wheelhouse '.[dev,server]'
python3 -m pytest tests/test_validation_artifacts.py tests/test_export_scope.py tests/test_server_readiness.py -q
npm --prefix gui ci --offline
npm --prefix gui run typecheck
```

## Gate Behavior

`dpf.release.airgap_gate.airgap_release_gate()` reports `passed=false` until the
wheelhouse, hash manifest, and offline logs are present. This prevents the SRS
from claiming air-gap release readiness from ordinary online CI.

Network-created artifacts must be reviewed for licensing before vendoring.
