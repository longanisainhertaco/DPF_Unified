# ADR: Compute Authority and Result Classification

Date: 2026-05-08
Status: accepted for implementation

## Context

The supplied SRS template describes a T0/T2-style authority split where a
reference backend can support validation claims and accelerated backends are
preview-oriented unless promoted by evidence. DPF-Unified currently has Python,
Athena/AthenaK, Metal, MLX, hybrid, and auto-dispatch paths, but it did not
have a formal result-authority model.

## Decision

DPF-Unified will use explicit result classification labels instead of implying
authority from the backend name alone.

Labels:

- `Reference`: may support a validation claim only when linked evidence is
  accepted and same-scope.
- `Preview`: useful engineering or accelerated output; cannot support
  validation claims.
- `Derived Diagnostic`: post-processed or helper output that depends on another
  classified result.
- `Exploratory`: research or tuning output outside release claims.
- `Superseded`: replaced by newer evidence or result.
- `Invalid`: failed, malformed, or explicitly rejected output.

Backend authority defaults:

- `python`, `athena`, and `athenak` are reference candidates, not automatic
  reference results.
- `metal`, `mlx`, `hybrid`, and unresolved `auto` outputs default to `Preview`
  unless a future accepted validation rule promotes a specific result scope.

## Consequences

- No result can become `Reference` only because it came from a preferred
  backend.
- Draft digitization and `blocked_by_review` evidence cannot produce a passing
  validation certificate.
- Run manifests and validation certificates must carry classification and
  validation status.
