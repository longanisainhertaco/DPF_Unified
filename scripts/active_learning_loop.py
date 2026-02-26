#!/usr/bin/env python3
"""Active Learning Loop for WALRUS.

Runs DPF simulations using the Hybrid Engine (Physics + Surrogate).
If the Surrogate model fails (divergence or unphysical state), the script automatically:
1. Logs the failure as a "Hard Example".
2. Re-runs the FULL simulation using the Physics Engine (Ground Truth).
3. Saves the Ground Truth trajectory in Polymathic WELL format for fine-tuning.

Usage:
    python scripts/active_learning_loop.py --checkpoint path/to/model.pt --n-samples 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dpf.ai.batch_runner import BatchRunner, ParameterRange, FULL_RANGES
from dpf.ai.hybrid_engine import HybridEngine
from dpf.ai.surrogate import DPFSurrogate
from dpf.ai.well_exporter import WellExporter
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("active_learning")

class ActiveLearner(BatchRunner):
    """
    Extends BatchRunner to use HybridEngine and capture hard examples.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.surrogate = None
        
    def _load_surrogate(self):
        if self.surrogate is None:
            logger.info(f"Loading surrogate from {self.checkpoint_path}...")
            self.surrogate = DPFSurrogate(self.checkpoint_path, device="cpu") # Use CPU for inference in loop to save GPU for training/physics? Or MPS?
            # Creating surrogate on CPU to avoid VRAM contention if running physics on GPU
            logger.info("Surrogate loaded.")

    def run_single(self, idx: int, params: dict[str, float]) -> tuple[int, str | None]:
        self._load_surrogate()
        
        try:
            config = self.build_config(params)
            
            # 1. Run Hybrid Simulation
            hybrid = HybridEngine(
                config, 
                self.surrogate, 
                handoff_fraction=0.2,
                validation_interval=20,
                max_l2_divergence=1e-2
            )
            
            logger.info(f"[{idx}] Running Hybrid Simulation...")
            summary = hybrid.run()
            
            # 2. Check for Fallback
            if summary["fallback_to_physics"]:
                logger.warning(f"[{idx}] Surrogate Failed! Fallback triggered. Generating Ground Truth...")
                
                # 3. Generate Ground Truth (Full Physics)
                # Re-run from scratch to ensure clean trajectory
                self._generate_ground_truth(idx, config, params)
                return (idx, "Fallback (Saved)")
                
            else:
                logger.info(f"[{idx}] Surrogate Succeeded. Skipping Ground Truth generation.")
                return (idx, None)
                
        except Exception as e:
            logger.error(f"[{idx}] Failed: {e}")
            return (idx, str(e))

    def _generate_ground_truth(self, idx: int, config: SimulationConfig, params: dict[str, float]):
        """Run full physics simulation and save to WELL format."""
        logger.info(f"[{idx}] Starting Ground Truth Physics Run...")
        
        engine = SimulationEngine(config)
        
        output_path = self.output_dir / f"hard_example_{idx:04d}.h5"
        dz = config.geometry.dz if config.geometry.dz is not None else config.dx
        
        exporter = WellExporter(
            output_path=output_path,
            grid_shape=tuple(config.grid_shape),
            dx=config.dx,
            dz=dz,
            geometry=config.geometry.type,
            sim_params=params
        )
        
        # Initial State
        exporter.add_snapshot(
            state=engine.get_field_snapshot(),
            time=0.0,
            circuit_scalars={"current": engine.circuit.current, "voltage": engine.circuit.voltage}
        )
        
        step_count = 0
        while True:
            res = engine.step()
            step_count += 1
            
            if step_count % self.field_interval == 0:
                exporter.add_snapshot(
                    state=engine.get_field_snapshot(),
                    time=engine.time,
                    circuit_scalars={"current": engine.circuit.current, "voltage": engine.circuit.voltage}
                )
                
            if res.finished:
                break
                
        # Final
        if step_count % self.field_interval != 0:
            exporter.add_snapshot(
                state=engine.get_field_snapshot(),
                time=engine.time,
                circuit_scalars={"current": engine.circuit.current, "voltage": engine.circuit.voltage}
            )
            
        exporter.finalize()
        logger.info(f"[{idx}] Ground Truth Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="WALRUS Active Learning Loop")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to WALRUS checkpoint")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--output-dir", type=str, default="hard_examples", help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    
    args = parser.parse_args()
    
    # Base Config (Standard DPF)
    base_config = SimulationConfig(
        grid_shape=[32, 1, 64],
        dx=5e-4,
        sim_time=2e-6,
        circuit={"type": "rlc", "L": 20e-9, "C": 1e-6, "V0": 10e3, "R": 5e-3}
    )
    
    learner = ActiveLearner(
        checkpoint_path=args.checkpoint,
        base_config=base_config,
        parameter_ranges=FULL_RANGES,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        workers=args.workers,
        field_interval=5
    )
    
    learner.run()

if __name__ == "__main__":
    main()
