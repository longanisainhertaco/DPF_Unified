#!/usr/bin/env python3
"""Analyze data regimes: Polymathic 'The Well' vs. DPF Simulations.

This script calculates statistical properties (Mean, Std, Min, Max) of
fields in both datasets to guide normalization strategies for Transfer Learning.

Usage:
    python scripts/analyze_regime.py --well_path /path/to/MHD_64.h5 --dpf_path /path/to/dpf_sim.h5
"""

import argparse
import logging
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dpf.ai.well_loader import WellDataset
from dpf.ai.well_exporter import WellExporter # For reading DPF HDF5 if needed, or just use h5py
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze_regime")

def get_stats(loader, n_samples=100):
    """Compute stats for a dataset loader."""
    logger.info(f"Computing stats for {len(loader)} samples (using subset of {n_samples})...")
    
    indices = np.random.choice(len(loader), min(len(loader), n_samples), replace=False)
    
    # Accumulators
    field_data = {}
    
    for idx in indices:
        sample = loader[idx]
        for field, tensor in sample.items():
            if field not in field_data:
                field_data[field] = []
            
            # Subsample pixels to save memory/time
            # Flatten: C, T, X, Y, Z -> C, N
            # WellDataset returns: Tensor
            # If scalar: (T, 1, X, Y, Z)
            # If vector: (T, 3, X, Y, Z)
            
            flat = tensor.view(-1)
            # Take random 1000 points per sample
            if flat.numel() > 1000:
                perm = torch.randperm(flat.numel())[:1000]
                flat = flat[perm]
                
            field_data[field].append(flat)
            
    stats = {}
    for field, chunks in field_data.items():
        if not chunks: continue
        all_data = torch.cat(chunks)
        
        stats[field] = {
            "mean": float(all_data.mean()),
            "std": float(all_data.std()),
            "min": float(all_data.min()),
            "max": float(all_data.max())
        }
        
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--well_path", type=str, required=True, help="Path to The Well MHD .h5 file")
    parser.add_argument("--dpf_path", type=str, required=True, help="Path to DPF simulation .h5 file (WELL format)")
    args = parser.parse_args()
    
    # 1. Load Well Data
    print("--- Loading The Well Data ---")
    well_ds = WellDataset(
        hdf5_paths=[args.well_path],
        fields=["density", "pressure", "magnetic_field", "velocity"],
        normalize=False # Raw data
    )
    well_stats = get_stats(well_ds)
    
    # 2. Load DPF Data
    print("\n--- Loading DPF Simulation Data ---")
    dpf_ds = WellDataset(
        hdf5_paths=[args.dpf_path],
        fields=["density", "pressure", "magnetic_field", "velocity"],
        normalize=False
    )
    dpf_stats = get_stats(dpf_ds)
    
    # 3. Compare
    print("\n" + "="*60)
    print(f"{'Field':<15} | {'Well Mean':<12} | {'DPF Mean':<12} | {'Ratio (DPF/Well)':<15}")
    print("-" * 60)
    
    for field in ["density", "pressure", "B", "velocity"]:
        # field naming might differ: 'magnetic_field' vs 'B'
        # WellLoader maps 'magnetic_field' -> 'B' automatically via WELL_TO_DPF_NAMES
        # So stats keys should be DPF names.
        
        # Check keys
        key = field
        if field == "magnetic_field": key = "B"
        
        if key in well_stats and key in dpf_stats:
            w_mean = well_stats[key]["mean"]
            d_mean = dpf_stats[key]["mean"]
            ratio = d_mean / w_mean if w_mean != 0 else 0.0
            
            print(f"{key:<15} | {w_mean:.2e}   | {d_mean:.2e}   | {ratio:.2e}")
            
    print("="*60)
    print("Analysis Complete. Use these ratios to set normalization constants.")

if __name__ == "__main__":
    main()
