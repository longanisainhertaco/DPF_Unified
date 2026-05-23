# DPF Unified Durable OpenAI + Anthropic Continuous Workflow

Date: 2026-05-22
Board: `dpf-unified`
Repo: `/Users/anthonyzamora/dpf-unified`

## Purpose

This workflow converts the bounded interactive chat loop into a durable Hermes-native worker system using:

- Option B elements: independent Hermes profiles/workers for implementation, review, and orchestration.
- Option C elements: cron-backed watchdog and status/orchestration jobs.
- Option D: Kanban as the durable multi-agent task graph and audit trail.

## Profiles

- `dpfcodex`
  - Provider: `openai-codex`
  - Model: `gpt-5.5`
  - Role: implementation worker. Writes tests first, implements phase artifacts, runs verification, and keeps acceptance fail-closed.

- `dpfclaude`
  - Provider: `anthropic`
  - Model: `claude-sonnet-4`
  - Role: independent read-only reviewer. Reviews security, logic, physics/source grounding, and acceptance-gate leaks.
  - Hermes-native Anthropic API credentials are not configured, but Claude Code CLI is authenticated via Claude Max. Durable bridge job `2deb8a57132a` handles `dpfclaude` review tasks with Claude Code when Hermes-native Anthropic credentials are unavailable.

- `dpforch`
  - Provider: `openai-codex`
  - Model: `gpt-5.5`
  - Role: Kanban orchestrator. Routes next-phase work and creates follow-on tasks instead of implementing directly.

## Kanban board

Board slug: `dpf-unified`
Display name: `DPF Unified Continuous Workflow`
Default workdir: `/Users/anthonyzamora/dpf-unified`

Initial graph:

1. `t_cbb2b24c` — `dpfcodex`
   - Phase 6-C implement power-port certification scaffold.
   - Status at creation/dispatch: running.

2. `t_80dc9e54` — `dpfclaude`
   - Anthropic independent review of Phase 6-C.
   - Parent: `t_cbb2b24c`.

3. `t_5343c2cd` — `dpfcodex`
   - Fix/reverify after Anthropic review.
   - Parent: `t_80dc9e54`.

4. `t_67b414ec` — `dpforch`
   - Orchestrate next phase after Phase 6-C.
   - Parent: `t_5343c2cd`.

5. `t_1c431c40` — `default`, blocked
   - Operator action: configure Anthropic credentials for `dpfclaude`.

## Cron jobs

1. `6db11b254b1f` — DPF Unified Kanban dispatch watchdog
   - Schedule: every 5 minutes
   - Mode: script-only/no-agent
   - Script: `~/.hermes/scripts/dpf_unified_kanban_dispatch_watchdog.py`
   - Behavior: runs `hermes kanban --board dpf-unified dispatch` and stays silent unless dispatch/status fails.

2. `d2b062c0ae23` — DPF Unified continuous orchestrator/status
   - Schedule: every 30 minutes
   - Model/provider: `openai-codex` / `gpt-5.5`
   - Script: `~/.hermes/scripts/dpf_unified_status_snapshot.py`
   - Workdir: `/Users/anthonyzamora/dpf-unified`
   - Behavior: reports board/repo/acceptance-scan status, dispatches once, and creates next-phase Kanban tasks only if the board is empty and work remains.

3. `2deb8a57132a` — DPF Unified Claude Code review bridge
   - Schedule: every 7 minutes
   - Mode: script-only/no-agent
   - Script: `~/.hermes/scripts/dpf_claude_code_kanban_bridge.py`
   - Behavior: handles ready/running/blocked `dpfclaude` review tasks using authenticated Claude Code CLI / Claude Max. It reclaims/unblocks review tasks if needed, runs read-only Claude review, completes on `PASS:`, and blocks on `REQUEST_CHANGES:` or errors.

## Guardrails embedded in worker tasks

- Do not touch active external corpus jobs, Hermes gateway, Ollama, or MLX servers.
- No destructive cleanup.
- Preserve reversibility.
- Tests-first for new validators/code.
- Run focused pytest, integrated pytest, ruff, static scan, and acceptance-promotion scan.
- Write Evaluate/Learn/Continue reports for each completed phase.
- Keep all `accepted_*`, `promotes_acceptance`, and `can_support_first_principles_acceptance` flags false unless the full certificate stack is complete and explicitly reviewed.

## Operator commands

List board:

```bash
hermes kanban --board dpf-unified list
```

Show task:

```bash
hermes kanban --board dpf-unified show t_cbb2b24c
```

Tail task events:

```bash
hermes kanban --board dpf-unified tail t_cbb2b24c
```

Show worker log:

```bash
hermes kanban --board dpf-unified log t_cbb2b24c
```

Show cron jobs:

```bash
hermes cron list
```

Pause watchdog/orchestrator:

```bash
hermes cron pause 6db11b254b1f
hermes cron pause d2b062c0ae23
```

Resume:

```bash
hermes cron resume 6db11b254b1f
hermes cron resume d2b062c0ae23
```

Claude Code bridge status:

```bash
hermes cron run 2deb8a57132a
```

After configuring Hermes-native Anthropic credentials, optional direct profile test:

```bash
hermes -p dpfclaude chat -q 'Connectivity check: reply DPF_CLAUDE_OK' --provider anthropic --model claude-sonnet-4
```
