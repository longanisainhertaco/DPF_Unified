You are the Memory Manager. Save a fact or finding to the appropriate topic file.

## Instructions

When the user invokes `/remember`, do the following:

1. **Parse the fact**: $ARGUMENTS

2. **Classify the topic** — determine which memory file this belongs in:
   - Bug-related (new bug, bug fix, retraction) → `memory/bugs.md`
   - Coding pattern, gotcha, test technique → `memory/patterns.md`
   - Phase work, implementation history → `memory/phases.md`
   - Metal GPU, MPS, float32/64, WENO, HLLD → `memory/metal.md`
   - WALRUS, surrogate, AI/ML modules → `memory/walrus.md`
   - PhD debate score, verdict, assessment → `memory/debates.md`
   - General project state (test count, CI gate, score) → `memory/MEMORY.md` (only critical ground truth)
   - Research findings → consider a new topic file if it doesn't fit existing ones

3. **Read the target file** to understand its current structure and avoid duplicates.

4. **Append the fact**:
   - Add it under the appropriate section heading in the target file
   - Include today's date as a tag: `[2026-MM-DD]`
   - Keep the entry concise (1-3 lines)
   - Match the existing formatting style of the file

5. **Validate**: Run `python tools/memory_manager.py validate` to check for conflicts or line count issues.

6. **Confirm**: Report what was saved and in which file.

## Important
- Do NOT add to MEMORY.md unless it is critical ground truth (project state, new bugs). MEMORY.md must stay under 150 lines.
- Check for duplicates before adding — search the target file first.
- If MEMORY.md would exceed 150 lines after the addition, move existing content to a topic file instead.
