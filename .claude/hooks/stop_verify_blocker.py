#!/usr/bin/env python3
"""Stop hook: Check if Claude worked on the blocker during this turn."""
import json, os, sys, subprocess, re

input_data = json.load(sys.stdin)

if input_data.get("stop_hook_active", False):
    sys.exit(0)

blocker_path = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), "CRITICAL_BLOCKER.md")
if not os.path.isfile(blocker_path) or os.path.getsize(blocker_path) == 0:
    sys.exit(0)

try:
    RELEVANT = re.compile(r'(mlx_solver|coupling|test_mhd|test_coupling|extract_scalars|pirt_traverse|floor_telemetry|cfl_diagnostic|reference_data)')
    result = subprocess.run(["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, timeout=5)
    if result.stdout.strip() and RELEVANT.search(result.stdout.strip()):
        sys.exit(0)
except Exception:
    sys.exit(0)

output = {
    "additionalContext": (
        "NOTE: CRITICAL_BLOCKER is active but no blocker-relevant files were modified this turn. "
        "Next turn, prioritize work that directly moves test_mhd_acceptance.py toward passing."
    )
}
json.dump(output, sys.stdout)
sys.exit(0)
