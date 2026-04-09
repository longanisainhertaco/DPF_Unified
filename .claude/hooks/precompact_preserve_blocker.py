#!/usr/bin/env python3
"""PreCompact hook: Ensure blocker survives context compaction."""
import json, os, sys

blocker_path = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), "CRITICAL_BLOCKER.md")

if os.path.isfile(blocker_path) and os.path.getsize(blocker_path) > 0:
    with open(blocker_path) as f:
        content = f.read().strip()[:2000]
    output = {
        "additionalContext": (
            "CRITICAL COMPACTION INSTRUCTION: Preserve the following BLOCKER VERBATIM "
            "as the FIRST item in the compacted summary:\n\n" + content
        )
    }
    json.dump(output, sys.stdout)

sys.exit(0)
