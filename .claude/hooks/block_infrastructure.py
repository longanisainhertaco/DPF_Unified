#!/usr/bin/env python3
"""PreToolUse hook: Block new file creation and infrastructure bash during active blocker."""
import json, os, sys, re

input_data = json.load(sys.stdin)
tool = input_data.get("tool_name", "")
blocker_path = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), "CRITICAL_BLOCKER.md")

if not os.path.isfile(blocker_path) or os.path.getsize(blocker_path) == 0:
    sys.exit(0)

ALLOWED = re.compile(r'(test_|extract_scalars|floor_telemetry|pirt_traverse|CRITICAL_BLOCKER|scalars\.json|SIM_REPORT|handoff|cfl_diagnostic|reference_data)')

if tool == "Write":
    file_path = input_data.get("tool_input", {}).get("file_path", "")
    if not ALLOWED.search(file_path):
        print(f"BLOCKED: CRITICAL_BLOCKER is active. Cannot create '{file_path}'.", file=sys.stderr)
        sys.exit(2)

elif tool == "Bash":
    cmd = input_data.get("tool_input", {}).get("command", "")
    FILE_CREATE = re.compile(r'(cat\s*>|echo\s.*>|tee\s|touch\s+\S+\.py|mkdir\s|>\s*\S+\.py|>\s*\S+\.md)')
    if FILE_CREATE.search(cmd) and not ALLOWED.search(cmd):
        print(f"BLOCKED: CRITICAL_BLOCKER is active. Bash file creation detected.", file=sys.stderr)
        sys.exit(2)

sys.exit(0)
