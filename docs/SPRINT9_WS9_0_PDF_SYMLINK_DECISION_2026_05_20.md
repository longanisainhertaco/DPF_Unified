# Sprint 9 WS9-0 — PDF Symlink Typechange Decision

**Date:** 2026-05-20
**Workstream:** WS9-0 (Worktree and Audit Gate Cleanup)
**Defect resolved:** P1-3 — `git_status_clean` gate failing on 145 PDF typechanges
**Author:** Agent W (Sprint 9)

---

## 1. Observed State

`git status --short` reports 145 lines with ` T ` (unstaged typechange) status:
files that git originally tracked as regular files have been converted to
symlinks pointing into `~/PDFs/` (the user's external PDF corpus on the host
machine).

### Count

```
145 typechange entries (all ` T ` / unstaged typechange)
```

### Affected Directories

| Directory | Count |
|---|---:|
| `downloaded_books_papers/Research Papers/` | 143 |
| `downloaded_books_papers/papers/` | 1 |
| `tmp/pdfs/` | 1 |

### Where the Symlinks Point

Verified via `ls -la` on a sample:

```
downloaded_books_papers/Research Papers/2026-05-11-user-ingest/buneman1959.pdf
  -> /Users/anthonyzamora/PDFs/01_DPF_Dense_Plasma_Focus/archive/buneman-1959-dissipation-currents-ionized.pdf

tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.pdf
  -> /Users/anthonyzamora/PDFs/01_DPF_Dense_Plasma_Focus/auxiliary/sawsorheoh_ocr.pdf
```

All 145 entries are PDF reference files redirected to the user's personal PDF
corpus at `~/PDFs/`. They are **not source code** and contain **no simulator
logic**.

---

## 2. Options Considered

The controlling audit doc (`CODEX_SUPER_SPRINT8_AUDIT_AND_SUPER_SPRINT9_INSTRUCTIONS_2026_05_20.md`,
§P1-3) offered three options:

| Option | Description | Verdict |
|---|---|---|
| **Option 1** — Normalization commit | `git add` + `git commit` the 145 symlinks into git history | Rejected: symlinks point outside the repo (`~/PDFs/`), making the repo non-portable and checkout non-reproducible on any machine without that exact path layout |
| **Option 2** — Remove from repo | `git rm` the symlinks; rely on KnowledgeReference only | Rejected: removes the corpus navigation layer without Anthony's explicit sign-off; data would still exist in `~/PDFs/` but the repo-side structure would be destroyed |
| **Option 3** — Narrow documented exception | Gate classifies known PDF ` T ` lines as excused churn, passes, and reports them explicitly | **Chosen** |

---

## 3. Decision: Option 3 — Narrow Documented Audit Exception

### Rationale

1. **No data loss.** The symlinks are navigation aids; the actual PDFs remain
   intact in `~/PDFs/`. Removing or committing them requires Anthony to make a
   storage policy decision outside this sprint's scope.
2. **Non-portable commit is worse than an exception.** Committing
   host-machine-absolute symlinks would break every other checkout silently —
   a harder defect than a documented exception.
3. **Honest reporting is preserved.** The gate passes but its `note` field
   explicitly names the count and directories so no reader mistakes this for a
   truly clean worktree.

### What This Exception Matches (Exact Rule)

A git status line is excused **if and only if** all three conditions hold:

1. The XY porcelain code is exactly ` T ` (space-T-space — unstaged typechange, not staged `T ` or any other code).
2. The path begins with `downloaded_books_papers/` **or** `tmp/pdfs/`.
3. The file extension is `.pdf` (implied by the corpus directories — no non-PDF files live there).

### What This Exception Does NOT Match

Anything not satisfying all three conditions fails the gate normally:

- ` M ` modified source files
- ` D ` deleted files
- `??` untracked files
- ` A ` staged additions
- ` T ` typechanges **outside** `downloaded_books_papers/` or `tmp/pdfs/`
- Staged typechange `T ` lines (different XY code)

---

## 4. Implementation

**File modified:** `scripts/run_codex_periodic_audit.py`

Two functions added at module level (after the `ROOT`/`DEFAULT_LOG_ROOT`
constants):

- `_is_excused_pdf_typechange(line: str) -> bool` — matches exactly the
  narrow rule above.
- `_classify_git_status_lines(dirty_lines) -> (excused, real_dirty)` — splits
  dirty lines into excused and gate-failing buckets.

The `_run_gate` function's `require_clean_status` block now calls
`_classify_git_status_lines`; only `real_dirty` causes `ok = False`. If only
excused lines are present, the gate passes with an explicit `note`:

```
APPROVED EXCEPTION: 145 PDF-symlink typechange(s) in known external-storage
dirs excused (Sprint 9 WS9-0 decision). Dirs: `downloaded_books_papers/`,
`tmp/pdfs/`
```

The note is visible in both `summary.json` and `summary.md` so no downstream
reader is misled.

---

## 5. Test Coverage

`tests/test_git_status_clean_exception.py` — unit tests for the narrow
exception classifier:

- Accepts PDF ` T ` lines under `downloaded_books_papers/` and `tmp/pdfs/`
- Rejects ` T ` outside those directories
- Rejects ` M ` modified lines
- Rejects ` D ` deleted lines
- Rejects `??` untracked lines
- Rejects staged typechange `T ` lines
