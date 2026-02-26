"""Train Time-Stepping Surrogate (SimpleUNet3D) on Well/DPF Data.

Supports pre-training on 'The Well' MHD datasets by mapping available fields
to the DPF 11-channel schema and masking missing fields in the loss.
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dpf.ai.models import SimpleUNet3D
from dpf.ai.well_loader import WellDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_surrogate")

# DPF Channel Order (11 channels)
# rho, Te, Ti, pressure, psi, Bx, By, Bz, vx, vy, vz
DPF_CHANNELS = [
    "density",      # 0
    "Te",           # 1
    "Ti",           # 2
    "pressure",     # 3
    "psi",          # 4
    "B_x",          # 5
    "B_y",          # 6
    "B_z",          # 7
    "v_x",          # 8
    "v_y",          # 9
    "v_z"           # 10
]

def collate_well_to_dpf(batch):
    """
    Collate list of dicts from WellDataset into (B, T, 11, X, Y, Z) tensors.
    Fills missing DPF fields with zeros.
    Returns: x (history), y (target), mask (valid channels)
    """
    # Batch size
    B = len(batch)

    # Get shape from first sample's density (mapped to "rho" by WellDataset)
    # sample['rho']: (T, 1, X, Y, Z)
    ref = batch[0]["rho"]
    T, _, X, Y, Z = ref.shape

    # Output tensor: (B, T, 11, X, Y, Z)
    out = torch.zeros(B, T, 11, X, Y, Z, dtype=torch.float32)
    mask = torch.zeros(11, dtype=torch.float32)  # Global mask (assuming same fields for all)

    for i, sample in enumerate(batch):
        # 0: Density (WellDataset maps "density" -> "rho")
        if "rho" in sample:
            out[i, :, 0] = sample["rho"].squeeze(1)
            mask[0] = 1.0

        # 3: Pressure
        if "pressure" in sample:
            out[i, :, 3] = sample["pressure"].squeeze(1)
            mask[3] = 1.0

        # 5-7: B
        if "B" in sample:
            # Well B is (T, 3, X, Y, Z)
            out[i, :, 5:8] = sample["B"]
            mask[5:8] = 1.0

        # 8-10: V
        if "velocity" in sample:
            out[i, :, 8:11] = sample["velocity"]
            mask[8:11] = 1.0

    # For now, simplistic handle of missing Te, Ti, psi
    # Maybe map Pressure -> Te, Ti?
    # If mask[1] is 0 (missing Te), let's fill it with Pressure * const?
    # Better to leave as 0 and mask loss.

    return out, mask

def train(args):
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 1. Dataset
    logger.info(f"Loading dataset from {args.data_path}")
    ds = WellDataset(
        hdf5_paths=[args.data_path],
        fields=["density", "pressure", "magnetic_field", "velocity"],
        sequence_length=args.history + 1, # Need history + 1 target
        normalize=True # Important for training
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_well_to_dpf,
        num_workers=2
    )

    # 2. Model
    # Input: History * 11 channels
    # Output: 11 channels (next step)
    model = SimpleUNet3D(
        in_channels=11 * args.history,
        out_channels=11
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, (data, mask) in enumerate(loader):
            # data: (B, T, 11, X, Y, Z)
            # T = history + 1

            # Split Input / Target
            # Input: T=0..history-1
            # Target: T=history

            # Example: history=1. T=2. Input=idx 0. Target=idx 1.

            input_seq = data[:, :args.history] # (B, H, 11, ...)
            target = data[:, args.history]     # (B, 11, ...)

            # Flatten history into channels
            # (B, H, 11, X,Y,Z) -> (B, H*11, X,Y,Z)
            B_sz, H, C, X, Y, Z = input_seq.shape
            inputs = input_seq.reshape(B_sz, H*C, X, Y, Z).to(device)
            target = target.to(device)
            mask = mask.to(device) # (11,)

            # Forward
            optimizer.zero_grad()
            pred = model(inputs) # (B, 11, X,Y,Z)

            # Masked Loss
            # Only penalize known fields
            # mask shape (11). pred shape (B, 11, ...).
            # Expand mask
            loss_map = criterion(pred, target)
            masked_loss = (loss_map * mask.view(1, -1, 1, 1, 1)).mean()

            masked_loss.backward()
            optimizer.step()

            train_loss += masked_loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                logger.debug(f"Ep {epoch} B {batch_idx} Loss {masked_loss.item():.4e}")

        avg_loss = train_loss / n_batches
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4e}")

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s")

    # Save
    save_path = Path("src/dpf/ai/surrogate_model.pt")

    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model": "SimpleUNet3D",
            "history_length": args.history,
            "stats": ds.stats # In case we implemented stats sharing
        }
    }

    torch.save(save_dict, save_path)
    logger.info(f"Saved model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")

    args = parser.parse_args()
    train(args)
