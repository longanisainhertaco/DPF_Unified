You are the Session Manager. Save the current session state for resumption in a future session.

## Instructions

When the user invokes `/session-save`, do the following:

1. **Parse the request**: $ARGUMENTS (optional description of current work)

2. **Collect session state**:
   - Run `git status` and `git branch --show-current` to capture repo state
   - Summarize what you have been working on this session
   - List all files you have modified or created
   - Note any key decisions or trade-offs made
   - Note any blockers or open questions
   - Capture any critical context the next session needs (error messages, partial implementations, variable values)

3. **Write the checkpoint**:
   - Run: `python tools/memory_manager.py checkpoint "DESCRIPTION"` (using the user's description or a generated summary)
   - Then overwrite `~/.claude/projects/-Users-anthonyzamora-dpf-unified/memory/session.md` with the full session state using this structure:
     - **Metadata**: timestamp, branch, description
     - **Objective**: what the session was working on (1-2 sentences)
     - **Progress**: checkbox list of completed and remaining steps
     - **Key Decisions**: important choices made during this session
     - **Files Modified**: list of files changed with brief descriptions
     - **Blockers / Open Questions**: anything blocking progress
     - **Context for Next Session**: critical state for resumption

4. **Confirm**: Report "Session saved. Next session can run `/session-resume` to continue."

## Important
- Be thorough in the Progress section — mark what is done vs what remains
- The Context for Next Session section is critical — include enough detail that a fresh Claude instance can pick up exactly where you left off
- If working on a Phase, include the phase letter and sub-task number
