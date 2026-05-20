# Codex-Claude Dual Audit Automation

Date: 2026-05-20

Script: `scripts/run_codex_claude_dual_audit.py`

Purpose: automate the review loop between Codex and the current local Claude
Code instance without granting Claude authority to edit files or promote
physics claims.

## Operating Rule

Codex remains the source-grounded auditor for this repository. Claude is a
second reviewer that receives a timestamped evidence packet and returns an
independent critique. No source, code, or physics acceptance state changes until
Codex audits the Claude output against `KnowledgeReference/`, the tests, and the
acceptance ledgers.

## One-Shot Evidence Packet

```bash
.venv312/bin/python scripts/run_codex_claude_dual_audit.py
```

This writes:

- `docs/dual_agent_audits/<timestamp>_codex_evidence.json`
- `docs/dual_agent_audits/<timestamp>_claude_prompt.md`

It runs the focused Sprint 5 / ledger pytest suite unless `--skip-tests` is
passed.

## One-Shot Claude Review

```bash
.venv312/bin/python scripts/run_codex_claude_dual_audit.py \
  --run-claude \
  --continue-claude \
  --max-budget-usd 1.00
```

`--continue-claude` resumes the most recent Claude Code conversation for this
working directory. Claude is invoked with:

- `--print`
- `--output-format json`
- `--permission-mode dontAsk`
- `--tools ""`

That means Claude receives the evidence packet but is not given edit tools.
The full evidence JSON is written to disk; the prompt sent to Claude is
compacted so large dirty worktree listings do not consume the entire budget.

If the current Claude teammate session needs read-only memory/file access, use
an explicit read-only allowlist:

```bash
.venv312/bin/python scripts/run_codex_claude_dual_audit.py \
  --run-claude \
  --continue-claude \
  --claude-tools Read,Grep,Glob,mcp__qmd__get \
  --max-budget-usd 1.00
```

Do not add edit or shell tools to this command for audit mode.

Additional outputs:

- `docs/dual_agent_audits/<timestamp>_claude_response.json`
- `docs/dual_agent_audits/<timestamp>_claude_response.md`

## Repeating Audit Loop

```bash
.venv312/bin/python scripts/run_codex_claude_dual_audit.py \
  --run-claude \
  --continue-claude \
  --repeat-minutes 30 \
  --iterations 6 \
  --max-budget-usd 1.00
```

Use this only when the other team is actively landing work. The script records
every iteration separately.

## Guardrails

- Dirty worktree output is included in the evidence packet and must be treated
  as a release/audit blocker unless the changed paths are explicitly waived.
- Claude output is advisory only.
- The script does not push, commit, modify acceptance ledgers, or run full
  physics promotion.
- Runtime acceptance still requires source packet, code consumption, numerical
  acceptance, same-scope comparator, and certificate gates at the same commit.
