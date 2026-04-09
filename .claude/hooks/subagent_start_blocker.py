#!/usr/bin/env python3
"""SubagentStart hook: Inject blocker context into every subagent."""
import json, os, sys

blocker_path = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), "CRITICAL_BLOCKER.md")

if os.path.isfile(blocker_path) and os.path.getsize(blocker_path) > 0:
    with open(blocker_path) as f:
        content = f.read().strip()[:4000]
    output = {
        "additionalContext": (
            f"CRITICAL BLOCKER IS ACTIVE. You are a subagent. The main session has an active blocker:\n"
            f"{content}\n"
            f"Do NOT create infrastructure, documentation, or feature code. "
            f"Focus only on work that directly advances the blocker fix."
        )
    }
    json.dump(output, sys.stdout)

sys.exit(0)
