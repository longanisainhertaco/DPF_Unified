"""Local-first security posture checks for release governance."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HARDWARE_CONTROL_MODULES = frozenset(
    {
        "RPi",
        "gpiozero",
        "labjack",
        "minimalmodbus",
        "modbus_tk",
        "nidaqmx",
        "ophyd",
        "pydaqmx",
        "pymeasure",
        "pyserial",
        "pyusb",
        "pyvisa",
        "serial",
        "smbus",
        "spidev",
        "usb",
        "visa",
    }
)

LOCAL_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
REMOTE_URL_RE = re.compile(r"https?://[^\s'\"()<>]+")
LOCAL_URL_PREFIXES = (
    "http://localhost",
    "https://localhost",
    "ws://localhost",
    "wss://localhost",
    "http://127.0.0.1",
    "https://127.0.0.1",
    "ws://127.0.0.1",
    "wss://127.0.0.1",
)


@dataclass(frozen=True)
class HardwareControlFinding:
    """Static finding for a direct hardware-control import."""

    path: str
    line: int
    module: str
    statement: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeAIMutationFinding:
    """Static finding for runtime AI access to active simulation mutation paths."""

    path: str
    line: int
    pattern: str
    statement: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RendererExternalAssetFinding:
    """Static finding for renderer references to non-local HTTP assets."""

    path: str
    line: int
    url: str
    statement: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalFirstPolicy:
    """Release policy defaults for the current local-first product posture."""

    default_bind_host: str = "127.0.0.1"
    public_share_default: bool = False
    wildcard_cors_requires_explicit_env: bool = True
    hardware_control_allowed: bool = False
    runtime_ai_mutation_allowed: bool = False
    audit_log_required: bool = True
    artifact_classification_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_python_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            for candidate in path.rglob("*.py"):
                if any(part.startswith(".") for part in candidate.parts):
                    continue
                yield candidate


def _iter_renderer_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    suffixes = {".css", ".html", ".js", ".jsx", ".ts", ".tsx"}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix in suffixes:
            yield path
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if not candidate.is_file() or candidate.suffix not in suffixes:
                    continue
                if any(part.startswith(".") for part in candidate.parts):
                    continue
                yield candidate


def _statement_at(source_lines: list[str], line_no: int) -> str:
    if line_no <= 0 or line_no > len(source_lines):
        return ""
    return source_lines[line_no - 1].strip()


def _module_root(module: str) -> str:
    return module.split(".", 1)[0]


def scan_hardware_control_imports(paths: Iterable[str | Path]) -> list[HardwareControlFinding]:
    """Scan Python files for direct imports of hardware-control libraries."""

    findings: list[HardwareControlFinding] = []
    for path in _iter_python_files(paths):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        source_lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = _module_root(alias.name)
                    if module in HARDWARE_CONTROL_MODULES:
                        findings.append(
                            HardwareControlFinding(
                                path=str(path),
                                line=node.lineno,
                                module=module,
                                statement=_statement_at(source_lines, node.lineno),
                            )
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = _module_root(node.module)
                if module in HARDWARE_CONTROL_MODULES:
                    findings.append(
                        HardwareControlFinding(
                            path=str(path),
                            line=node.lineno,
                            module=module,
                            statement=_statement_at(source_lines, node.lineno),
                        )
                    )
    return findings


RUNTIME_AI_MUTATION_PATTERNS = (
    "_simulations",
    "SimulationManager",
    "SimulationEngine(",
    "create_engine(",
    ".start(",
    ".pause(",
    ".resume(",
    ".stop(",
    ".write_text(",
    ".write_bytes(",
)


def scan_runtime_ai_mutation_boundaries(
    paths: Iterable[str | Path],
) -> list[RuntimeAIMutationFinding]:
    """Scan runtime AI entrypoints for active simulation mutation access."""

    findings: list[RuntimeAIMutationFinding] = []
    for path in _iter_python_files(paths):
        try:
            source_lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_no, line in enumerate(source_lines, start=1):
            stripped = line.strip()
            for pattern in RUNTIME_AI_MUTATION_PATTERNS:
                if pattern in stripped:
                    findings.append(
                        RuntimeAIMutationFinding(
                            path=str(path),
                            line=line_no,
                            pattern=pattern,
                            statement=stripped,
                        )
                    )
    return findings


def scan_renderer_external_assets(
    paths: Iterable[str | Path],
) -> list[RendererExternalAssetFinding]:
    """Scan renderer files for non-local HTTP(S) asset references."""

    findings: list[RendererExternalAssetFinding] = []
    for path in _iter_renderer_files(paths):
        try:
            source_lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_no, line in enumerate(source_lines, start=1):
            for match in REMOTE_URL_RE.finditer(line):
                url = match.group(0)
                if url.startswith(LOCAL_URL_PREFIXES):
                    continue
                findings.append(
                    RendererExternalAssetFinding(
                        path=str(path),
                        line=line_no,
                        url=url,
                        statement=line.strip(),
                    )
                )
    return findings


def _control(control_id: str, status: str, evidence: Any) -> dict[str, Any]:
    return {"id": control_id, "status": status, "evidence": evidence}


def local_first_security_audit(
    project_root: str | Path,
    *,
    policy: LocalFirstPolicy | None = None,
) -> dict[str, Any]:
    """Return an auditable local-first security-control snapshot."""

    root = Path(project_root)
    active_policy = policy or LocalFirstPolicy()
    hardware_findings = scan_hardware_control_imports([root / "src" / "dpf"])
    runtime_ai_findings = scan_runtime_ai_mutation_boundaries(
        [
            root / "src" / "dpf" / "ai" / "realtime_server.py",
            root / "src" / "dpf" / "ai" / "chat_router.py",
        ]
    )
    renderer_external_asset_findings = scan_renderer_external_assets(
        [root / "gui" / "src" / "renderer"]
    )

    controls = [
        _control(
            "DPF-SEC-001",
            "passed" if not hardware_findings and not active_policy.hardware_control_allowed else "failed",
            {
                "hardware_control_allowed": active_policy.hardware_control_allowed,
                "findings": [finding.as_dict() for finding in hardware_findings],
            },
        ),
        _control(
            "DPF-SEC-002",
            "passed"
            if active_policy.default_bind_host in LOCAL_BIND_HOSTS
            and not active_policy.public_share_default
            and active_policy.wildcard_cors_requires_explicit_env
            else "failed",
            {
                "default_bind_host": active_policy.default_bind_host,
                "public_share_default": active_policy.public_share_default,
                "wildcard_cors_requires_explicit_env": (
                    active_policy.wildcard_cors_requires_explicit_env
                ),
            },
        ),
        _control(
            "DPF-SEC-003",
            "passed"
            if not active_policy.runtime_ai_mutation_allowed and not runtime_ai_findings
            else "failed",
            {
                "runtime_ai_mutation_allowed": active_policy.runtime_ai_mutation_allowed,
                "findings": [finding.as_dict() for finding in runtime_ai_findings],
            },
        ),
        _control(
            "DPF-SEC-004",
            "passed" if active_policy.artifact_classification_required else "failed",
            {"artifact_classification_required": active_policy.artifact_classification_required},
        ),
        _control(
            "DPF-SEC-005",
            "passed" if not renderer_external_asset_findings else "failed",
            {
                "renderer_external_assets_allowed": False,
                "findings": [
                    finding.as_dict() for finding in renderer_external_asset_findings
                ],
            },
        ),
    ]

    return {
        "policy": active_policy.as_dict(),
        "controls": controls,
        "passed": all(control["status"] == "passed" for control in controls),
    }
