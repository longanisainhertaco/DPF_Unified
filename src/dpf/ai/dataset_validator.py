"""Well format HDF5 dataset validation.

Validates HDF5 files exported in the Well format for ML training,
checking schema compliance, numerical validity, energy conservation,
and field statistics.

References:
    The Well data format: https://github.com/PolymathicAI/the_well
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available; dataset validation disabled")


@dataclass
class ValidationResult:
    """Result of validating a single HDF5 dataset file.

    Attributes:
        valid: Overall validity status.
        n_trajectories: Number of trajectories in dataset.
        n_timesteps: Number of timesteps per trajectory.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
        field_stats: Statistics for each field (mean, std, min, max, n_nan, n_inf).
        energy_drift: Maximum absolute energy conservation error.
    """

    valid: bool = True
    n_trajectories: int = 0
    n_timesteps: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    energy_drift: float = 0.0


class DatasetValidator:
    """Validator for Well-format HDF5 dataset files.

    Performs comprehensive validation including:
    - Schema compliance (required groups and attributes)
    - Numerical validity (NaN/Inf detection)
    - Energy conservation monitoring
    - Field-level statistics

    Args:
        energy_drift_threshold: Maximum acceptable energy drift (fraction).
    """

    def __init__(
        self,
        energy_drift_threshold: float = 0.05,
        *,
        strict: bool = False,
        saturation_abs_threshold: float = 1.0e30,
    ) -> None:
        self.energy_drift_threshold = energy_drift_threshold
        self.strict = strict
        self.saturation_abs_threshold = saturation_abs_threshold

    def validate_file(self, path: str | Path) -> ValidationResult:
        """Run all validation checks on a single Well HDF5 file.

        Args:
            path: Path to HDF5 file to validate.

        Returns:
            ValidationResult containing validation status and diagnostics.
        """
        if not HAS_H5PY:
            return ValidationResult(
                valid=False,
                errors=["h5py not available; cannot validate HDF5 files"],
            )

        path = Path(path)
        if not path.exists():
            return ValidationResult(
                valid=False,
                errors=[f"File not found: {path}"],
            )

        result = ValidationResult()

        # Check schema
        schema_errors = self.check_well_schema(path)
        result.errors.extend(schema_errors)

        # Check for NaN/Inf
        nan_inf_fields = self.check_nan_inf(path)
        if nan_inf_fields:
            result.errors.extend([f"NaN/Inf detected in field: {f}" for f in nan_inf_fields])

        # Compute field statistics
        result.field_stats = self.compute_field_statistics(path)

        if self.strict:
            result.errors.extend(self.check_strict_integrity(path))

        # Check energy conservation. NaN means the check was skipped
        # (h5py missing, field absent, or read error -- see
        # check_energy_conservation docstring); record a distinct warning
        # so reports surface "not measured" rather than silently passing.
        result.energy_drift = self.check_energy_conservation(path)
        if np.isnan(result.energy_drift):
            result.warnings.append(
                "Energy-conservation check skipped (no field or read error); "
                "drift unknown -- do not interpret missing value as zero."
            )
        elif result.energy_drift > self.energy_drift_threshold:
            result.warnings.append(
                f"Energy drift {result.energy_drift:.2%} exceeds threshold "
                f"{self.energy_drift_threshold:.2%}"
            )

        # Extract trajectory/timestep counts
        try:
            with h5py.File(path, "r") as f:
                if "n_trajectories" in f.attrs:
                    result.n_trajectories = int(f.attrs["n_trajectories"])
                if "t0_fields" in f:
                    # Shape is (n_traj, n_steps, ...)
                    shape = list(f["t0_fields"].values())[0].shape
                    result.n_trajectories = shape[0]
                    result.n_timesteps = shape[1]
                elif "t1_fields" in f:
                    shape = list(f["t1_fields"].values())[0].shape
                    result.n_trajectories = shape[0]
                    result.n_timesteps = shape[1]
        except Exception as e:
            result.errors.append(f"Failed to read dimensions: {e}")

        # Set overall validity
        result.valid = len(result.errors) == 0

        return result

    def validate_directory(self, directory: str | Path) -> dict[str, ValidationResult]:
        """Find and validate all HDF5 files in a directory.

        Args:
            directory: Path to directory containing HDF5 files.

        Returns:
            Dictionary mapping filename to ValidationResult.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Validation directory not found: {directory}")

        results: dict[str, ValidationResult] = {}
        for path in directory.rglob("*.h5"):
            results[path.name] = self.validate_file(path)
        for path in directory.rglob("*.hdf5"):
            results[path.name] = self.validate_file(path)

        return results

    def check_nan_inf(self, path: str | Path) -> list[str]:
        """Check for NaN/Inf values in all field datasets.

        Args:
            path: Path to HDF5 file.

        Returns:
            List of field names containing NaN or Inf values.
        """
        if not HAS_H5PY:
            return []

        bad_fields: list[str] = []

        try:
            with h5py.File(path, "r") as f:
                # Check t0_fields (scalars)
                if "t0_fields" in f:
                    for field_name, dataset in f["t0_fields"].items():
                        data = dataset[...]
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            bad_fields.append(f"t0_fields/{field_name}")

                # Check t1_fields (vectors)
                if "t1_fields" in f:
                    for field_name, dataset in f["t1_fields"].items():
                        data = dataset[...]
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            bad_fields.append(f"t1_fields/{field_name}")

                # Check non-spatial scalar histories such as current/voltage.
                if "scalars" in f:
                    for scalar_name, dataset in f["scalars"].items():
                        data = dataset[...]
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            bad_fields.append(f"scalars/{scalar_name}")

        except Exception as e:
            logger.error(f"Failed to check NaN/Inf in {path}: {e}")

        return bad_fields

    def check_strict_integrity(self, path: str | Path) -> list[str]:
        """Run fail-closed checks for training-data provenance and integrity."""
        if not HAS_H5PY:
            return ["h5py not available"]

        errors: list[str] = []

        try:
            with h5py.File(path, "r") as f:
                if "scalars" not in f or "energy_conservation" not in f["scalars"]:
                    errors.append(
                        "Strict validation requires scalars/energy_conservation"
                    )

                if "dimensions" not in f or "time" not in f["dimensions"]:
                    errors.append("Strict validation requires dimensions/time")
                elif np.any(np.diff(f["dimensions/time"][...]) <= 0.0):
                    errors.append("Strict validation requires strictly increasing time")

                grid_type = f.attrs.get("grid_type")
                if grid_type not in ("cartesian", "cylindrical"):
                    errors.append(
                        "Strict validation requires root grid_type "
                        "'cartesian' or 'cylindrical'"
                    )
                elif "dimensions" in f:
                    dims = f["dimensions"]
                    dim_names = set(dims.keys())
                    if grid_type == "cartesian":
                        missing = {"x", "y", "z"} - dim_names
                        if missing:
                            errors.append(
                                "Strict cartesian grid missing dimensions: "
                                + ", ".join(sorted(missing))
                            )
                    if grid_type == "cylindrical":
                        missing = {"r", "z"} - dim_names
                        if missing:
                            errors.append(
                                "Strict cylindrical grid missing dimensions: "
                                + ", ".join(sorted(missing))
                            )
                        if "x" in dim_names or "y" in dim_names:
                            errors.append(
                                "Strict cylindrical grid must not be labeled "
                                "with cartesian x/y dimensions"
                            )

                if "validation_status" not in f.attrs or "result_label" not in f.attrs:
                    errors.append(
                        "Strict validation requires validation_status and "
                        "result_label root attributes"
                    )
                if "source_status" not in f.attrs and "artifact_classification" not in f.attrs:
                    errors.append(
                        "Strict validation requires source_status or "
                        "artifact_classification root metadata"
                    )

                for group_name in ("t0_fields", "t1_fields", "scalars"):
                    if group_name not in f:
                        continue
                    for dataset_name, dataset in f[group_name].items():
                        if bool(dataset.attrs.get("nonfinite_sanitized", False)):
                            errors.append(
                                f"Strict validation rejects sanitized non-finite "
                                f"dataset: {group_name}/{dataset_name}"
                            )
                        data = dataset[...]
                        if np.any(np.abs(data) > self.saturation_abs_threshold):
                            errors.append(
                                f"Strict validation detected saturation-scale values "
                                f"in {group_name}/{dataset_name}"
                            )

                b_dataset = None
                if "t1_fields" in f:
                    if "magnetic_field" in f["t1_fields"]:
                        b_dataset = f["t1_fields/magnetic_field"]
                    elif "B" in f["t1_fields"]:
                        b_dataset = f["t1_fields/B"]
                if b_dataset is None:
                    errors.append("Strict validation requires magnetic field data")
                else:
                    b_data = b_dataset[...]
                    if np.all(np.isfinite(b_data)) and np.allclose(b_data, 0.0):
                        errors.append(
                            "Strict validation rejects all-zero magnetic field data"
                        )

        except Exception as e:
            errors.append(f"Failed strict integrity validation: {e}")

        return errors

    def check_well_schema(self, path: str | Path) -> list[str]:
        """Verify required groups and attributes exist.

        The Well format requires:
        - Either "t0_fields" or "t1_fields" group
        - "dimensions" group
        - "boundary_conditions" group
        - Root attribute "n_trajectories" or "dataset_name"

        Args:
            path: Path to HDF5 file.

        Returns:
            List of error strings (empty if schema is valid).
        """
        if not HAS_H5PY:
            return ["h5py not available"]

        errors: list[str] = []

        try:
            with h5py.File(path, "r") as f:
                # Check required groups
                if "t0_fields" not in f and "t1_fields" not in f:
                    errors.append("Missing required group: t0_fields or t1_fields")

                if "dimensions" not in f:
                    errors.append("Missing required group: dimensions")

                if "boundary_conditions" not in f:
                    errors.append("Missing required group: boundary_conditions")

                # Check required attributes
                if "n_trajectories" not in f.attrs and "dataset_name" not in f.attrs:
                    errors.append(
                        "Missing required root attribute: n_trajectories or dataset_name"
                    )

        except Exception as e:
            errors.append(f"Failed to read HDF5 schema: {e}")

        return errors

    def check_energy_conservation(self, path: str | Path) -> float:
        """Compute maximum energy conservation error.

        Args:
            path: Path to HDF5 file.

        Returns:
            Maximum absolute deviation from unity in ``energy_conservation``.

            Returns ``float('nan')`` (not 0.0) in any case where the check
            could not be performed, distinguishing three skip cases (each
            paired with a logged warning or error so the caller can audit):

            - ``h5py`` not installed -> NaN + warning ``"h5py unavailable"``
            - ``scalars/energy_conservation`` field absent -> NaN + warning
              ``"field absent"``
            - HDF5 read failure -> NaN + ``logger.error`` describing the
              exception

            Callers MUST treat NaN as "check skipped" (NOT
            "perfectly conserved"); ``validate_file`` does this by suppressing
            the threshold warning when the drift is NaN. See P0.6.7.
        """
        if not HAS_H5PY:
            logger.warning(
                "Energy-conservation check skipped: h5py unavailable. "
                f"Returning NaN for {path} (NOT 0.0 -- 'skipped', not 'OK')."
            )
            return float("nan")

        try:
            with h5py.File(path, "r") as f:
                if "scalars" in f and "energy_conservation" in f["scalars"]:
                    energy_cons = f["scalars/energy_conservation"][...]
                    # energy_conservation should be ~1.0
                    drift = np.max(np.abs(1.0 - energy_cons))
                    return float(drift)
                # Field absent -> distinct skip case
                logger.warning(
                    f"Energy-conservation check skipped: field "
                    f"'scalars/energy_conservation' absent in {path}. "
                    "Returning NaN -- dataset does not track energy "
                    "conservation; do NOT interpret as 'conserves energy'."
                )
                return float("nan")
        except Exception as e:
            # Read error -> distinct skip case
            logger.error(
                f"Energy-conservation check failed: HDF5 read error in {path}: {e}. "
                "Returning NaN -- callers must not treat as 'conserves energy'."
            )
            return float("nan")

    def compute_field_statistics(self, path: str | Path) -> dict[str, dict[str, float]]:
        """Compute mean, std, min, max, n_nan, n_inf for all fields.

        Args:
            path: Path to HDF5 file.

        Returns:
            Dictionary mapping field name to statistics dict.
            Example: {"density": {"mean": 1.0, "std": 0.1, "min": 0.5,
                                   "max": 2.0, "n_nan": 0, "n_inf": 0}}
        """
        if not HAS_H5PY:
            return {}

        stats: dict[str, dict[str, float]] = {}

        try:
            with h5py.File(path, "r") as f:
                # Process t0_fields (scalars)
                if "t0_fields" in f:
                    for field_name, dataset in f["t0_fields"].items():
                        data = dataset[...]
                        stats[field_name] = {
                            "mean": float(np.mean(data)),
                            "std": float(np.std(data)),
                            "min": float(np.min(data)),
                            "max": float(np.max(data)),
                            "n_nan": int(np.sum(np.isnan(data))),
                            "n_inf": int(np.sum(np.isinf(data))),
                        }

                # Process t1_fields (vectors)
                if "t1_fields" in f:
                    for field_name, dataset in f["t1_fields"].items():
                        data = dataset[...]
                        stats[field_name] = {
                            "mean": float(np.mean(data)),
                            "std": float(np.std(data)),
                            "min": float(np.min(data)),
                            "max": float(np.max(data)),
                            "n_nan": int(np.sum(np.isnan(data))),
                            "n_inf": int(np.sum(np.isinf(data))),
                        }

        except Exception as e:
            logger.error(f"Failed to compute field statistics in {path}: {e}")

        return stats

    def summary_report(self, results: dict[str, ValidationResult]) -> str:
        """Format a human-readable validation summary.

        Args:
            results: Dictionary mapping filename to ValidationResult.

        Returns:
            Multi-line string report.
        """
        lines = ["Dataset Validation Report", "=" * 50, ""]

        n_valid = sum(1 for r in results.values() if r.valid)
        n_total = len(results)

        for filename, result in sorted(results.items()):
            status = "VALID" if result.valid else "INVALID"
            lines.append(f"{filename}: {status}")
            lines.append(
                f"  Trajectories: {result.n_trajectories}, "
                f"Timesteps: {result.n_timesteps}"
            )
            lines.append(f"  Energy drift: {result.energy_drift:.2%}")

            if result.errors:
                lines.append("  Errors:")
                for error in result.errors:
                    lines.append(f"    - {error}")

            if result.warnings:
                lines.append("  Warnings:")
                for warning in result.warnings:
                    lines.append(f"    - {warning}")

            lines.append("")

        lines.append("=" * 50)
        lines.append(f"Summary: {n_valid}/{n_total} files valid")

        return "\n".join(lines)
