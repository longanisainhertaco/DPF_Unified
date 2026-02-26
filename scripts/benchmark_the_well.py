#!/usr/bin/env python3
"""Benchmark DPF Metal Backend against Polymathic AI 'The Well' datasets.

This script:
1. Loads an initial state from a Polymathic MHD dataset (local or HuggingFace).
2. Runs the DPF Metal Solver (and optionally CPU solver) on this state.
3. Compares the simulation trajectory against the ground truth in The Well.
4. Reports L2 error and Performance (steps/second).

Usage:
    python scripts/benchmark_the_well.py --dataset_path /path/to/MHD_64.h5
    python scripts/benchmark_the_well.py --mock  # Use generated dummy data

Dependencies:
    h5py, torch, numpy
    (Optional) the_well
"""

import argparse
import logging
import time
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dpf.metal.metal_solver import MetalMHDSolver
from dpf.ai.field_mapping import WELL_TO_DPF_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

def generate_mock_well_data(grid_size=64, n_steps=10):
    """Generate a dummy dataset in memory matching The Well format."""
    logger.info(f"Generating mock data {grid_size}^3 x {n_steps} steps...")
    
    # MHD_64 usually has: density, velocity, magnetic_field, pressure
    # Shape: (n_traj, n_steps, nx, ny, nz, [dims])
    
    data = {}
    nx = ny = nz = grid_size
    
    # 1. Scalar Fields
    for name in ["density", "pressure"]:
        # (1, n_steps, nx, ny, nz)
        arr = np.random.rand(1, n_steps, nx, ny, nz).astype(np.float32) + 0.1
        data[name] = arr
        
    # 2. Vector Fields
    for name in ["velocity", "magnetic_field"]:
        # (1, n_steps, nx, ny, nz, 3)
        arr = np.random.rand(1, n_steps, nx, ny, nz, 3).astype(np.float32)
        data[name] = arr
        
    attrs = {
        "dt": 1e-4,
        "dx": 1.0 / grid_size,
        "grid_type": "cartesian"
    }
    
    return data, attrs

def load_well_data(path):
    import h5py
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
        
    logger.info(f"Loading {path}...")
    f = h5py.File(path, 'r')
    
    # Read relevant fields only (Trajectory 0)
    data = {}
    
    # Groups: t0_fields (scalars), t1_fields (vectors)
    # Note: Structure might vary slightly between Well versions.
    # Assuming polymathic-ai/MHD structure.
    
    # Try flat structure or groups
    def read_dataset(key):
        if key in f: return f[key]
        if "t0_fields" in f and key in f["t0_fields"]: return f["t0_fields"][key]
        if "t1_fields" in f and key in f["t1_fields"]: return f["t1_fields"][key]
        return None

    for well_name in ["density", "pressure", "velocity", "magnetic_field"]:
        dset = read_dataset(well_name)
        if dset is not None:
            # Read all steps for Traj 0
            data[well_name] = dset[0] # (n_steps, ...)
        else:
            logger.warning(f"Field {well_name} not found in dataset")
            
    attrs = dict(f.attrs)
    if "dt" not in attrs and "time" in f:
        t = f["time"][:]
        attrs["dt"] = t[1] - t[0]
        
    return data, attrs

def run_benchmark(data, attrs, device="mps", steps_to_run=10):
    # Extracts
    rho_hist = data.get("density")
    p_hist = data.get("pressure")
    v_hist = data.get("velocity") # (T, X, Y, Z, 3)
    B_hist = data.get("magnetic_field")
    
    if rho_hist is None:
        logger.error("Missing density data")
        return

    n_available = rho_hist.shape[0]
    steps_to_run = min(steps_to_run, n_available - 1)
    
    nx, ny, nz = rho_hist.shape[1:4]
    dx = float(attrs.get("dx", 1.0/nx))
    dt = float(attrs.get("dt", 1e-4))
    
    logger.info(f"Simulating {nx}x{ny}x{nz}, dt={dt:.2e}, device={device}")
    
    # Initialize Solver
    solver = MetalMHDSolver(
        grid_shape=(nx, ny, nz),
        dx=dx,
        gamma=5.0/3.0,
        device=device,
        use_native_metal=True # Trigger our optimization
    )
    
    # Initial State (t=0)
    # Convert to standard layout: Scalars (X,Y,Z), Vectors (3,X,Y,Z)
    
    def prep_scalar(arr):
        return torch.from_numpy(arr[0]).to(device)
        
    def prep_vector(arr):
        # Well: (X,Y,Z,3) -> DPF: (3,X,Y,Z)
        x = np.moveaxis(arr[0], -1, 0)
        return torch.from_numpy(x).to(device)

    rho = prep_scalar(rho_hist)
    p = prep_scalar(p_hist)
    vel = prep_vector(v_hist)
    B = prep_vector(B_hist)
    
    # Run Loop
    t0 = time.time()
    errors = []
    
    for i in range(steps_to_run):
        rho, vel, p, B = solver.step(rho, vel, p, B, dt)
        
        # Validation
        # Get Ground Truth at t=i+1
        rho_gt = torch.from_numpy(rho_hist[i+1]).to(device)
        
        # Relative L2 Error on Density
        diff = torch.norm(rho - rho_gt)
        norm = torch.norm(rho_gt)
        err = float(diff / (norm + 1e-8))
        errors.append(err)
        
        if (i+1) % 10 == 0:
            logger.info(f"Step {i+1}: L2 Error (rho) = {err:.4e}")
            
    wall_time = time.time() - t0
    sps = steps_to_run / wall_time
    
    logger.info(f"Done. {steps_to_run} steps in {wall_time:.2f}s => {sps:.1f} steps/sec")
    logger.info(f"Final L2 Error: {errors[-1]:.4e}")
    
    return sps, errors[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to .h5 file")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--device", type=str, default="mps", help="Compute device")
    parser.add_argument("--steps", type=int, default=50, help="Steps to run")
    
    args = parser.parse_args()
    
    if args.mock:
        data, attrs = generate_mock_well_data()
    elif args.dataset_path:
        data, attrs = load_well_data(args.dataset_path)
    else:
        logger.error("Must specify --dataset_path or --mock")
        sys.exit(1)
        
    run_benchmark(data, attrs, device=args.device, steps_to_run=args.steps)

if __name__ == "__main__":
    main()
