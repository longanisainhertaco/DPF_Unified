# SRS Traceability Tooling

Date: 2026-05-09

## Decision

Use Doorstop as the first repository-native traceability tool for DPF-Unified SRS work. Doorstop stores requirements and test-like items as version-controlled files, validates trace links, and can publish reports without moving the project away from Git.

Sphinx-Needs remains a later option if the project moves its SRS and V&V documentation into a Sphinx documentation build. The current docs stack is MkDocs, so Doorstop is the lower-friction first step.

## Installed/Configured Surface

- Repo optional dependency: `dpf-unified[traceability]`
- First external tool: `doorstop>=3.1`
- Active environment status: installed and verified as `Doorstop v3.1` using
  the `doorstop` console script.
- Staged RTM export:
  - `scripts/export_srs_traceability.py`
  - `docs/SRS_TRACEABILITY_MATRIX.json`
  - `docs/SRS_TRACEABILITY_MATRIX.csv`
- Codex environment skills created:
  - `dpf-validation`
  - `srs-traceability`
- Supporting curated Codex skills installed:
  - `pdf`
  - `playwright`
  - `security-best-practices`
  - `security-threat-model`
  - `security-ownership-map`

Restart Codex before expecting newly installed skills to auto-trigger in a fresh session.

## Initial Use

Install the optional dependency in the project environment:

```bash
python3 -m pip install -e ".[traceability]"
```

Run Doorstop help before initializing documents:

```bash
doorstop --help
```

Use the `doorstop` console script. The installed package does not expose a
`python3 -m doorstop` module entrypoint.

Do not create a Doorstop requirements tree until the SRS ID scheme is accepted.
The first candidate stable-ID table is `docs/DPF_REQUIREMENTS_BASELINE.md`.
Until Doorstop is installed in the active environment, export and validate the
candidate table with:

```bash
python3 scripts/export_srs_traceability.py
```

The generated JSON/CSV files are staged traceability artifacts, not final
Doorstop validation.

Current environment note: `python3 -m pip check` reports unrelated global
environment dependency conflicts after the editable install, including
`letta` requiring `typer<0.10.0` while the active environment now has
`typer 0.25.1`. Use a dedicated virtual environment before treating a release
or air-gap build as clean.
The planned prefixes are:

- `DPF-SYS`
- `DPF-FUNC`
- `DPF-PHYS`
- `DPF-VV`
- `DPF-DATA`
- `DPF-SEC`
- `DPF-OPS`
- `DPF-UI`
- `DPF-REL`

## Guardrails

- Do not mark draft, blocked, or cross-scope evidence as accepted.
- Do not label scaffolded interfaces as implemented requirements.
- Every requirement must have a verification method.
- Every closed requirement must link to acceptance evidence.
- Scientific validation requirements must link to local `KnowledgeReference/` evidence or remain blocked.
- Findings docs remain the active plan until the Doorstop tree exists and is validated.

## Next Step

Review `docs/DPF_REQUIREMENTS_BASELINE.md` and
`docs/SRS_TRACEABILITY_MATRIX.json`, resolve any ID/status changes, then
initialize Doorstop documents from the accepted table or staged export.
