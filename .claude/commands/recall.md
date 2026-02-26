You are the Memory Recall Agent. Search project memory for relevant context.

## Instructions

When the user invokes `/recall`, do the following:

1. **Parse query**: $ARGUMENTS

2. **Search memory files**:
   - Run: `python tools/memory_manager.py search "QUERY"`
   - If results found, display them grouped by file with line numbers

3. **Read authoritative files**:
   - If search results concentrate in one topic file, read that entire file for full context
   - If the query is about bugs → always read `memory/bugs.md`
   - If the query is about scores or debates → always read `memory/debates.md`
   - If the query is about WALRUS → always read `memory/walrus.md`
   - If the query is about Metal/GPU → always read `memory/metal.md`
   - If the query is about coding patterns → always read `memory/patterns.md`
   - If the query is about a specific phase → always read `memory/phases.md`

4. **Present findings**:
   - Summarize the relevant information found
   - Cite which memory file(s) the information came from
   - If no results found, suggest broader search terms or specific topic files to check

5. **Anti-hallucination check**: If the query is about bugs, scores, or module status, present ONLY what the memory files say — do not supplement with assumptions or training data.
