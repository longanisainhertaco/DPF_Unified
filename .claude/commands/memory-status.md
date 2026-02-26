You are the Memory Health Monitor. Report on the memory system's state.

## Instructions

When the user invokes `/memory-status`, do the following:

1. **Run health check**: `python tools/memory_manager.py status`

2. **Run validation**: `python tools/memory_manager.py validate`

3. **Run index**: `python tools/memory_manager.py index`

4. **Report**:
   - File inventory with sizes, line counts, and ages
   - Whether MEMORY.md is under the 150-line limit
   - Whether session.md exists and how fresh it is
   - Any staleness or conflict warnings from validation
   - Total memory footprint across all files

5. **Recommendations**: If any issues found, suggest specific fixes:
   - MEMORY.md over limit → identify content to move to topic files
   - Stale session.md → suggest running `/session-save`
   - Large topic files → suggest splitting
   - TODO/FIXME items → flag for resolution
