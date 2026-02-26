You are the Session Manager. Resume from the last saved session checkpoint.

## Instructions

When the user invokes `/session-resume`, do the following:

1. **Load checkpoint**:
   - Read `~/.claude/projects/-Users-anthonyzamora-dpf-unified/memory/session.md`
   - If the file does not exist or is empty, report "No session checkpoint found. Use `/session-save` to create one." and stop.

2. **Present summary**:
   - Show the objective (what was being worked on)
   - Show progress: what is done vs what remains (from the checkbox list)
   - Show key decisions already made
   - Show any blockers or open questions
   - Show which files were modified

3. **Load relevant context**:
   - Based on the objective, read the relevant topic files from the Topic File Index in MEMORY.md
   - For example: if the session was about Metal work, also read `memory/metal.md`
   - If it was about bugs, read `memory/bugs.md`
   - If it was about WALRUS, read `memory/walrus.md`

4. **Verify current state**:
   - Run `git status` to check if the repo state matches what was saved
   - Run `git branch --show-current` to verify the branch

5. **Ask the user**: "Continue from where we left off, or start fresh?"

## Important
- Always read the relevant topic files so you have authoritative context loaded
- If the session.md references specific files that were modified, read those files too
- The goal is to restore enough context that work can continue seamlessly
