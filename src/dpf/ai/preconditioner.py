"""Neural Preconditioner for Implicit Physics Solvers using WALRUS.

This module defines the `NeuralPreconditioner` class, which uses a pre-trained
WALRUS (or similar Neural Operator) model to predict an approximate solution
to the Poisson equation: nabla^2 phi = -rho / epsilon_0.

This prediction is used as the initial guess for iterative solvers (CG, GMRES),
reducing the number of iterations required for convergence.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try to import WALRUS components if available
try:
    from walrus.models import IsotropicModel
    _WALRUS_AVAILABLE = True
except ImportError:
    logger.warning("WALRUS library not found. NeuralPreconditioner will run in mock mode.")
    _WALRUS_AVAILABLE = False
    IsotropicModel = None

class NeuralPreconditioner:
    """AI-based preconditioner for the Poisson equation.
    
    Attributes:
        model: The loaded PyTorch model (WALRUS/FNO/U-Net).
        device: 'mps' or 'cpu'.
    """

    def __init__(self, model_path: str | None = None, device: str = "mps"):
        """Initialize the preconditioner.
        
        Args:
            model_path: Path to the .pt checkpoint. If None, uses mock mode.
            device: Compute device.
        """
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.model = None

        if _WALRUS_AVAILABLE and model_path:
            try:
                self.model = IsotropicModel.load_from_checkpoint(model_path)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded WALRUS model from {model_path} on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            # Fallback: Check for custom trained surrogate
            from pathlib import Path
            p = Path(__file__).parent / "poisson_net.pt"

            if p.exists():
                try:
                    from dpf.ai.models import SimpleUNet3D
                    checkpoint = torch.load(p, map_location=self.device, weights_only=True)
                    # Handle full checkpoint dict vs state_dict
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        state_dict = checkpoint

                    self.model = SimpleUNet3D(in_channels=1, out_channels=1).to(self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    logger.info(f"Loaded custom SimpleUNet3D from {p} on {self.device}")
                except Exception as e:
                    logger.warning(f"Failed to load custom surrogate: {e}")
                    self.model = None
            else:
                logger.info("Initialized NeuralPreconditioner in MOCK mode (Identity).")

    def predict(self, rho: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Predict potential phi given charge density rho.
        
        Args:
            rho: Input density field (N, N, N).
            
        Returns:
            phi: Predicted potential field (N, N, N).
        """
        is_numpy = isinstance(rho, np.ndarray)

        if is_numpy:
            x_tensor = torch.from_numpy(rho).float().to(self.device)
            # Add batch/channel dims if missing
            # Input to SimpleUNet3D expects (B, C, D, H, W)
            # If input is (D,H,W), make it (1, 1, D, H, W)
            if x_tensor.ndim == 3:
                x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
        else:
            x_tensor = rho
            if x_tensor.ndim == 3:
                x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)

        # Inference
        with torch.no_grad():
            if self.model:
                phi_tensor = self.model(x_tensor)
            else:
                # Mock: Return zeros
                phi_tensor = torch.zeros_like(x_tensor)

        if is_numpy:
            return phi_tensor.squeeze().cpu().numpy()
        return phi_tensor
