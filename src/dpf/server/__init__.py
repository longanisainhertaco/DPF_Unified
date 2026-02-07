"""DPF WebSocket + REST API server for Unity GUI integration.

Provides:
- REST endpoints for simulation lifecycle (create, start, pause, resume, stop)
- WebSocket streaming of per-step scalar diagnostics
- Binary field data transfer on demand
- Configuration validation and preset management
"""

from __future__ import annotations
