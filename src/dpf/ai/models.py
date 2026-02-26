"""Neural Network Architectures for DPF AI Acceleration.

This module defines lightweight neural networks for use when the full WALRUS
library is unavailable or when a task-specific surrogate is needed.
"""

import torch
import torch.nn as nn


class SimplePoissonNet(nn.Module):
    """Simple 3D CNN for solving Poisson Equation: rho -> phi.

    Architecture:
        Conv3d(1, 16) -> ReLU
        Conv3d(16, 32) -> ReLU
        Conv3d(32, 32) -> ReLU
        Conv3d(32, 16) -> ReLU
        Conv3d(16, 1)

    Maintains spatial resolution (padding=SAME).
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, 1, D, H, W)
        return self.encoder(x)

class SimpleUNet3D(nn.Module):
    """Lightweight 3D U-Net for more complex field mappings."""
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self._conv_block(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.center = self._conv_block(base_filters*2, base_filters*4)

        # Decoder
        self.up2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_filters*4, base_filters*2) # concats

        self.up1 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_filters*2, base_filters) # concats

        self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        c = self.center(p2)

        u2 = self.up2(c)
        # Pad if necessary for non-power-of-2? Assuming 64^3 here.
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)

        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)

        return self.final(d1)
