# DPF-Unified — Project Context

MHD dense plasma focus simulator. Python + C++ (Athena++/AthenaK).

## MCP Tool Policy

This is a pure computation/simulation project. Do NOT use these MCP tools:
- Canva (design tool — irrelevant)
- Gamma (presentation — irrelevant)
- n8n (workflow automation — irrelevant)
- Chrome browser automation (no web UI)
- iMessages (no messaging needed)
- PowerPoint / Word (no office docs)
- PDF Tools (no PDF processing)
- Claude Preview (no web frontend)

Acceptable MCPs: scheduled-tasks (for long simulations), mcp-registry (discovery only).

## Agent Routing

- Explore agents → `model: "haiku"`
- Code edits with clear specs → `model: "sonnet"`
- Physics reasoning, conservation law verification → `model: "opus"` or default
- Use `dpf-mhd-physicist`, `dpf-engine-architect`, `dpf-validation-engineer` agents for domain work

## Known Bugs (always check)

See MEMORY.md "Known DPF Bugs" section before touching any physics code.
