#!/usr/bin/env python3
"""End-to-end test of the WALRUS surrogate model in the DPF project.

This is NOT a unit test with mocks -- it uses the real 4.8GB checkpoint,
real PyTorch, and real WALRUS inference. CPU only.

Steps:
  1. Load real WALRUS checkpoint via DPFSurrogate
  2. Create realistic DPF initial state (16x16x16)
  3. Run predict_next_step() inference
  4. Test parameter_sweep()
  5. Test field_mapping round-trip
  6. Test WellExporter + DatasetValidator
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import traceback

import numpy as np

# Ensure the project is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("walrus_e2e_test")

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models", "walrus-pretrained"
)

RESULTS: dict[str, str] = {}


def record(step: str, status: str, detail: str = "") -> None:
    RESULTS[step] = status
    marker = "PASS" if status == "PASS" else "FAIL"
    msg = f"[{marker}] {step}"
    if detail:
        msg += f" -- {detail}"
    logger.info(msg)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load the real WALRUS checkpoint
# ═══════════════════════════════════════════════════════════════════════

def step1_load_checkpoint() -> object | None:
    """Load DPFSurrogate with the real checkpoint."""
    logger.info("=" * 70)
    logger.info("STEP 1: Loading real WALRUS checkpoint")
    logger.info("=" * 70)

    try:
        from dpf.ai.surrogate import DPFSurrogate
        from dpf.ai import HAS_WALRUS, HAS_TORCH

        logger.info(f"  HAS_TORCH  = {HAS_TORCH}")
        logger.info(f"  HAS_WALRUS = {HAS_WALRUS}")
        logger.info(f"  Checkpoint = {CHECKPOINT_DIR}")

        t0 = time.time()
        surrogate = DPFSurrogate(
            checkpoint_path=CHECKPOINT_DIR,
            device="cpu",
            history_length=4,
        )
        load_time = time.time() - t0

        logger.info(f"  Load time: {load_time:.1f}s")
        logger.info(f"  is_loaded: {surrogate.is_loaded}")
        logger.info(f"  _is_walrus_model: {surrogate._is_walrus_model}")

        if surrogate._is_walrus_model:
            model = surrogate._model
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model class: {type(model).__name__}")
            logger.info(f"  Parameters: {n_params:,} ({n_params/1e9:.2f}B)")
            logger.info(f"  Device: {surrogate.device}")
            logger.info(f"  History length: {surrogate.history_length}")

            if surrogate._walrus_config is not None:
                cfg = surrogate._walrus_config
                logger.info(f"  Hidden dim: {cfg.model.hidden_dim}")
                logger.info(f"  Processor blocks: {cfg.model.processor_blocks}")
                logger.info(f"  Causal in time: {cfg.model.causal_in_time}")
                logger.info(f"  Prediction type: {cfg.trainer.prediction_type}")

            if surrogate._revin is not None:
                logger.info(f"  RevIN: {type(surrogate._revin).__name__}")
            else:
                logger.info("  RevIN: None (degraded)")

            if surrogate._formatter is not None:
                logger.info(f"  Formatter: {type(surrogate._formatter).__name__}")

            if surrogate._dpf_field_indices is not None:
                logger.info(f"  Field indices: {surrogate._dpf_field_indices.tolist()}")

            record("1_load_checkpoint", "PASS",
                   f"{n_params/1e9:.2f}B params, loaded in {load_time:.1f}s")
        else:
            record("1_load_checkpoint", "FAIL",
                   "Model loaded as placeholder dict, not a real WALRUS model")
            return surrogate

        return surrogate

    except Exception:
        tb = traceback.format_exc()
        logger.error(f"STEP 1 FAILED:\n{tb}")
        record("1_load_checkpoint", "FAIL", tb.splitlines()[-1])
        return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Create a realistic DPF state
# ═══════════════════════════════════════════════════════════════════════

def create_dpf_state(grid: tuple[int, int, int] = (16, 16, 16)) -> dict[str, np.ndarray]:
    """Create a physically realistic DPF state.

    Fill gas: deuterium at 1e-4 kg/m3 (~5 Torr)
    Temperature: 1 eV = 11604 K
    Pressure: 100 Pa
    B: small seed field 0.01 T in z
    """
    nx, ny, nz = grid
    state = {
        "rho": np.full(grid, 1e-4, dtype=np.float64),
        "pressure": np.full(grid, 100.0, dtype=np.float64),
        "Te": np.full(grid, 11604.0, dtype=np.float64),
        "Ti": np.full(grid, 11604.0, dtype=np.float64),
        "psi": np.zeros(grid, dtype=np.float64),
        "velocity": np.zeros((3, *grid), dtype=np.float64),
        "B": np.zeros((3, *grid), dtype=np.float64),
    }
    # Small seed B-field in z-direction
    state["B"][2] = 0.01
    return state


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Run predict_next_step()
# ═══════════════════════════════════════════════════════════════════════

def step3_inference(surrogate: object) -> dict[str, np.ndarray] | None:
    """Run one inference step with the real WALRUS model."""
    logger.info("=" * 70)
    logger.info("STEP 3: Running predict_next_step() inference")
    logger.info("=" * 70)

    try:
        # Build history: 4 identical states (history_length=4)
        base_state = create_dpf_state((16, 16, 16))
        history = []
        for i in range(4):
            s = {k: v.copy() for k, v in base_state.items()}
            # Add small perturbation to make states slightly different
            s["rho"] += np.random.normal(0, 1e-7, s["rho"].shape)
            s["pressure"] += np.random.normal(0, 0.01, s["pressure"].shape)
            history.append(s)

        logger.info(f"  History length: {len(history)}")
        logger.info(f"  Grid shape: {history[0]['rho'].shape}")
        logger.info(f"  Input rho range: [{history[-1]['rho'].min():.6e}, {history[-1]['rho'].max():.6e}]")
        logger.info(f"  Input pressure range: [{history[-1]['pressure'].min():.4f}, {history[-1]['pressure'].max():.4f}]")

        t0 = time.time()
        predicted = surrogate.predict_next_step(history)
        inference_time = time.time() - t0

        logger.info(f"  Inference time: {inference_time:.1f}s")

        # Check output structure
        expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        actual_keys = set(predicted.keys())
        logger.info(f"  Output keys: {sorted(actual_keys)}")

        missing_keys = expected_keys - actual_keys
        if missing_keys:
            logger.warning(f"  Missing keys: {missing_keys}")

        # Check shapes
        for key in sorted(actual_keys):
            arr = predicted[key]
            logger.info(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")

        # Check for NaN/Inf
        all_finite = True
        for key, arr in predicted.items():
            n_nan = np.count_nonzero(np.isnan(arr))
            n_inf = np.count_nonzero(np.isinf(arr))
            if n_nan > 0 or n_inf > 0:
                logger.warning(f"  {key}: {n_nan} NaN, {n_inf} Inf values!")
                all_finite = False

        # Physical reasonableness checks
        logger.info("\n  Physical reasonableness:")
        phys_ok = True

        if "rho" in predicted:
            rho_min = predicted["rho"].min()
            rho_max = predicted["rho"].max()
            logger.info(f"    rho: [{rho_min:.6e}, {rho_max:.6e}]")
            if rho_min < 0:
                logger.warning("    rho has negative values (unphysical)")
                phys_ok = False

        if "pressure" in predicted:
            p_min = predicted["pressure"].min()
            p_max = predicted["pressure"].max()
            logger.info(f"    pressure: [{p_min:.6e}, {p_max:.6e}]")
            if p_min < 0:
                logger.warning("    pressure has negative values (unphysical)")
                phys_ok = False

        if "Te" in predicted:
            Te_min = predicted["Te"].min()
            Te_max = predicted["Te"].max()
            logger.info(f"    Te: [{Te_min:.6e}, {Te_max:.6e}]")

        if "Ti" in predicted:
            Ti_min = predicted["Ti"].min()
            Ti_max = predicted["Ti"].max()
            logger.info(f"    Ti: [{Ti_min:.6e}, {Ti_max:.6e}]")

        # State deltas
        logger.info("\n  State deltas (predicted - input):")
        ref = history[-1]
        for key in sorted(actual_keys):
            if key in ref:
                delta = predicted[key] - ref[key]
                delta_abs = np.abs(delta)
                logger.info(
                    f"    {key}: mean_delta={np.mean(delta):.6e}, "
                    f"max_abs_delta={delta_abs.max():.6e}, "
                    f"rms_delta={np.sqrt(np.mean(delta**2)):.6e}"
                )

        # Determine pass/fail
        has_all_keys = not missing_keys
        detail_parts = [f"inference={inference_time:.1f}s"]
        if not all_finite:
            detail_parts.append("HAS NaN/Inf!")
        if not phys_ok:
            detail_parts.append("unphysical values")
        if not has_all_keys:
            detail_parts.append(f"missing keys: {missing_keys}")

        if all_finite and has_all_keys:
            record("3_inference", "PASS", ", ".join(detail_parts))
        else:
            record("3_inference", "FAIL", ", ".join(detail_parts))

        return predicted

    except Exception:
        tb = traceback.format_exc()
        logger.error(f"STEP 3 FAILED:\n{tb}")
        record("3_inference", "FAIL", tb.splitlines()[-1])
        return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Test parameter_sweep()
# ═══════════════════════════════════════════════════════════════════════

def step4_parameter_sweep(surrogate: object) -> None:
    """Test parameter_sweep with 2 configs, 1 step each."""
    logger.info("=" * 70)
    logger.info("STEP 4: Testing parameter_sweep()")
    logger.info("=" * 70)

    try:
        configs = [
            {"rho0": 1e-4, "pressure0": 100.0, "Te0": 11604.0, "Ti0": 11604.0},
            {"rho0": 2e-4, "pressure0": 200.0, "Te0": 23208.0, "Ti0": 23208.0},
        ]

        # parameter_sweep creates 8x8x8 grids which are below WALRUS min of 16x16x16
        # This will likely fail for real WALRUS models, so we test with awareness
        t0 = time.time()
        results = surrogate.parameter_sweep(configs, n_steps=1)
        sweep_time = time.time() - t0

        logger.info(f"  Sweep time: {sweep_time:.1f}s")
        logger.info(f"  Number of results: {len(results)}")

        all_ok = True
        for i, result in enumerate(results):
            if "error" in result:
                logger.warning(f"  Config {i}: ERROR - {result['error']}")
                all_ok = False
            else:
                metrics = result.get("metrics", {})
                logger.info(f"  Config {i}: metrics={metrics}")

        if all_ok:
            record("4_parameter_sweep", "PASS", f"{len(results)} configs in {sweep_time:.1f}s")
        else:
            # Check if failure is due to 8x8x8 grid (expected limitation)
            err_msgs = [r.get("error", "") for r in results if "error" in r]
            if any("size" in str(e).lower() or "shape" in str(e).lower() for e in err_msgs):
                record("4_parameter_sweep", "FAIL",
                       "Grid size too small (8x8x8) for WALRUS min 16x16x16 -- known limitation")
            else:
                record("4_parameter_sweep", "FAIL",
                       f"Errors: {err_msgs}")

    except Exception:
        tb = traceback.format_exc()
        logger.error(f"STEP 4 FAILED:\n{tb}")
        record("4_parameter_sweep", "FAIL", tb.splitlines()[-1])


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Test field_mapping round-trip
# ═══════════════════════════════════════════════════════════════════════

def step5_field_mapping_roundtrip() -> None:
    """Test DPF -> Well -> DPF round-trip fidelity."""
    logger.info("=" * 70)
    logger.info("STEP 5: Testing field_mapping round-trip")
    logger.info("=" * 70)

    try:
        from dpf.ai.field_mapping import (
            dpf_scalar_to_well,
            dpf_vector_to_well,
            well_scalar_to_dpf,
            well_vector_to_dpf,
            validate_state_dict,
        )

        state = create_dpf_state((16, 16, 16))

        # Validate original state
        errors = validate_state_dict(state)
        logger.info(f"  Validation errors for original state: {errors}")

        max_roundtrip_error = 0.0
        all_ok = True

        # Test scalar round-trip
        logger.info("\n  Scalar round-trips:")
        for key in ("rho", "Te", "Ti", "pressure", "psi"):
            original = state[key]
            well_fmt = dpf_scalar_to_well(original)
            reconstructed = well_scalar_to_dpf(well_fmt)

            err = np.max(np.abs(original - reconstructed))
            max_roundtrip_error = max(max_roundtrip_error, err)
            rel_err = err / (np.max(np.abs(original)) + 1e-30)

            logger.info(f"    {key}: max_abs_error={err:.2e}, rel_error={rel_err:.2e}")
            if rel_err > 1e-5:
                logger.warning(f"    {key}: round-trip error too large!")
                all_ok = False

        # Test vector round-trip
        logger.info("\n  Vector round-trips:")
        for key in ("B", "velocity"):
            original = state[key]
            well_fmt = dpf_vector_to_well(original)
            reconstructed = well_vector_to_dpf(well_fmt)

            err = np.max(np.abs(original - reconstructed))
            max_roundtrip_error = max(max_roundtrip_error, err)
            # Avoid division by zero for zero velocity
            denom = np.max(np.abs(original)) + 1e-30
            rel_err = err / denom

            logger.info(f"    {key}: max_abs_error={err:.2e}, rel_error={rel_err:.2e}")
            if err > 1e-5 and rel_err > 1e-5:
                logger.warning(f"    {key}: round-trip error too large!")
                all_ok = False

        logger.info(f"\n  Max round-trip error: {max_roundtrip_error:.2e}")

        if all_ok:
            record("5_field_mapping_roundtrip", "PASS",
                   f"max_error={max_roundtrip_error:.2e}")
        else:
            record("5_field_mapping_roundtrip", "FAIL",
                   f"max_error={max_roundtrip_error:.2e}")

    except Exception:
        tb = traceback.format_exc()
        logger.error(f"STEP 5 FAILED:\n{tb}")
        record("5_field_mapping_roundtrip", "FAIL", tb.splitlines()[-1])


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Test WellExporter + DatasetValidator
# ═══════════════════════════════════════════════════════════════════════

def step6_well_export_validate() -> None:
    """Export a 2-snapshot trajectory to HDF5 and validate."""
    logger.info("=" * 70)
    logger.info("STEP 6: Testing WellExporter + DatasetValidator")
    logger.info("=" * 70)

    try:
        from dpf.ai.well_exporter import WellExporter
        from dpf.ai.dataset_validator import DatasetValidator

        grid = (16, 16, 16)
        state1 = create_dpf_state(grid)
        state2 = create_dpf_state(grid)
        # Perturb state2 slightly to simulate time evolution
        state2["rho"] *= 1.01
        state2["pressure"] *= 0.99
        state2["B"][2] *= 1.005

        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = os.path.join(tmpdir, "dpf_test_trajectory.h5")

            # Export
            logger.info(f"  Exporting to: {hdf5_path}")
            exporter = WellExporter(
                output_path=hdf5_path,
                grid_shape=grid,
                dx=0.001,  # 1mm grid spacing
                dz=0.001,
                geometry="cartesian",
                sim_params={"V0": 25000.0, "C0": 30e-6},
            )

            exporter.add_snapshot(state1, time=0.0, circuit_scalars={"current": 0.0, "voltage": 25000.0})
            exporter.add_snapshot(state2, time=1e-9, circuit_scalars={"current": 1000.0, "voltage": 24500.0})

            result_path = exporter.finalize()
            logger.info(f"  Exported to: {result_path}")
            logger.info(f"  File size: {os.path.getsize(result_path) / 1024:.1f} KB")

            # Validate
            validator = DatasetValidator(energy_drift_threshold=0.05)
            vresult = validator.validate_file(result_path)

            logger.info(f"  Validation result:")
            logger.info(f"    valid: {vresult.valid}")
            logger.info(f"    n_trajectories: {vresult.n_trajectories}")
            logger.info(f"    n_timesteps: {vresult.n_timesteps}")
            logger.info(f"    energy_drift: {vresult.energy_drift:.2%}")
            if vresult.errors:
                logger.info(f"    errors: {vresult.errors}")
            if vresult.warnings:
                logger.info(f"    warnings: {vresult.warnings}")

            # Print field stats
            if vresult.field_stats:
                logger.info(f"    field_stats:")
                for fname, stats in vresult.field_stats.items():
                    logger.info(
                        f"      {fname}: mean={stats['mean']:.6e}, "
                        f"std={stats['std']:.6e}, "
                        f"min={stats['min']:.6e}, max={stats['max']:.6e}, "
                        f"n_nan={stats['n_nan']}, n_inf={stats['n_inf']}"
                    )

            # Also verify HDF5 structure directly
            import h5py
            with h5py.File(result_path, "r") as f:
                logger.info(f"\n  HDF5 structure:")
                logger.info(f"    Root attrs: {dict(f.attrs)}")
                logger.info(f"    Groups: {list(f.keys())}")
                if "t0_fields" in f:
                    logger.info(f"    t0_fields: {list(f['t0_fields'].keys())}")
                    for name, ds in f["t0_fields"].items():
                        logger.info(f"      {name}: shape={ds.shape}, dtype={ds.dtype}")
                if "t1_fields" in f:
                    logger.info(f"    t1_fields: {list(f['t1_fields'].keys())}")
                    for name, ds in f["t1_fields"].items():
                        logger.info(f"      {name}: shape={ds.shape}, dtype={ds.dtype}")
                if "dimensions" in f:
                    logger.info(f"    dimensions: {list(f['dimensions'].keys())}")
                if "boundary_conditions" in f:
                    logger.info(f"    boundary_conditions attrs: {dict(f['boundary_conditions'].attrs)}")
                if "scalars" in f:
                    logger.info(f"    scalars: {list(f['scalars'].keys())}")

            if vresult.valid:
                record("6_well_export_validate", "PASS",
                       f"{vresult.n_timesteps} timesteps, {len(vresult.field_stats)} fields")
            else:
                record("6_well_export_validate", "FAIL",
                       f"Errors: {vresult.errors}")

    except Exception:
        tb = traceback.format_exc()
        logger.error(f"STEP 6 FAILED:\n{tb}")
        record("6_well_export_validate", "FAIL", tb.splitlines()[-1])


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    logger.info("=" * 70)
    logger.info("DPF WALRUS End-to-End Test")
    logger.info("=" * 70)

    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"WALRUS checkpoint: {CHECKPOINT_DIR}")
    logger.info(f"Device: CPU (forced)")
    logger.info("")

    # Step 1: Load checkpoint
    surrogate = step1_load_checkpoint()

    # Step 3: Inference (skip step 2 -- state creation is inline)
    predicted = None
    if surrogate is not None and surrogate.is_loaded:
        predicted = step3_inference(surrogate)
    else:
        record("3_inference", "FAIL", "Surrogate not loaded")

    # Step 4: Parameter sweep
    if surrogate is not None and surrogate.is_loaded:
        step4_parameter_sweep(surrogate)
    else:
        record("4_parameter_sweep", "FAIL", "Surrogate not loaded")

    # Step 5: Field mapping round-trip (independent of model)
    step5_field_mapping_roundtrip()

    # Step 6: Well export + validate (independent of model)
    step6_well_export_validate()

    # ── Final Summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)

    n_pass = 0
    n_fail = 0
    for step, status in sorted(RESULTS.items()):
        marker = "PASS" if status == "PASS" else "FAIL"
        logger.info(f"  [{marker}] {step}")
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

    logger.info("")
    logger.info(f"  Total: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail}")
    logger.info("=" * 70)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
