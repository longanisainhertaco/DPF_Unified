#!/usr/bin/env python3
"""SessionStart hook: Inject CRITICAL_BLOCKER.md into Claude's context."""
import json, os, sys

blocker_path = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), "CRITICAL_BLOCKER.md")

if os.path.isfile(blocker_path) and os.path.getsize(blocker_path) > 0:
    with open(blocker_path) as f:
        content = f.read().strip()
    if len(content) > 9000:
        content = content[:9000] + "\n[TRUNCATED — see CRITICAL_BLOCKER.md for full content]"
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": f"=== ACTIVE BLOCKER ===\n{content}\n=== END BLOCKER ==="
        }
    }
    json.dump(output, sys.stdout)
else:
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "No active blocker. Normal operations permitted."
        }
    }
    json.dump(output, sys.stdout)

sys.exit(0)
