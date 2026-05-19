"""Backend capability diagnostics for requested physics options."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class BackendFeatureDiagnostic:
    """Diagnostic for a requested feature with backend-specific behavior."""

    backend: str
    feature: str
    severity: Severity
    behavior: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def backend_feature_diagnostics(config: Any, backend: str) -> list[BackendFeatureDiagnostic]:
    """Return explicit diagnostics for requested backend/physics combinations."""

    fc = config.fluid
    rad = config.radiation
    sheath = config.sheath
    diagnostics: list[BackendFeatureDiagnostic] = []

    def add(feature: str, severity: Severity, behavior: str, message: str) -> None:
        diagnostics.append(
            BackendFeatureDiagnostic(
                backend=backend,
                feature=feature,
                severity=severity,
                behavior=behavior,
                message=message,
            )
        )

    if backend in ("athena", "athenak", "hybrid"):
        skipped = {
            "Braginskii viscosity": bool(fc.enable_viscosity or fc.full_braginskii_viscosity),
            "Nernst effect": bool(fc.enable_nernst),
            "anisotropic thermal conduction": bool(fc.enable_anisotropic_conduction),
            "radiation transport (bremsstrahlung/line)": bool(
                rad.bremsstrahlung_enabled or rad.line_radiation_enabled
            ),
            "sheath boundary conditions": bool(sheath.enabled),
            "RKL2 super time-stepping": fc.diffusion_method == "sts",
            "implicit diffusion (ADI)": fc.diffusion_method == "implicit",
        }
        for feature, requested in skipped.items():
            if requested:
                add(
                    feature,
                    "warning",
                    "skipped",
                    (
                        f"{feature} is requested but is not applied by the "
                        f"{backend} fast path"
                    ),
                )

    if backend in ("metal", "mlx") and fc.diffusion_method in ("sts", "implicit"):
        add(
            f"{fc.diffusion_method} diffusion",
            "info",
            "explicit_fallback",
            (
                f"{backend} currently uses explicit diffusion when "
                f"diffusion_method={fc.diffusion_method!r} is requested"
            ),
        )

    if backend in ("metal", "mlx"):
        backend_owned = {
            "Nernst effect": bool(fc.enable_nernst),
            "anisotropic thermal conduction": bool(fc.enable_anisotropic_conduction),
            "Braginskii viscosity": bool(fc.enable_viscosity or fc.full_braginskii_viscosity),
            "bremsstrahlung radiation": bool(rad.bremsstrahlung_enabled),
        }
        for feature, requested in backend_owned.items():
            if requested:
                add(
                    feature,
                    "info",
                    "backend_owned",
                    (
                        f"{feature} is requested and owned by the {backend} "
                        "solver path; Python operator-split ownership is skipped "
                        "to avoid double application"
                    ),
                )
        if rad.line_radiation_enabled:
            add(
                "line radiation",
                "info",
                "python_operator_owned",
                (
                    "Line radiation is requested and remains owned by the "
                    "Python operator-split path for this backend unless a "
                    "backend-native source packet is explicitly wired"
                ),
            )

    return diagnostics
