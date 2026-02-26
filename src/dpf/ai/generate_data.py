"""Generate training data for WALRUS Neural Preconditioner.

This script runs a series of static Poisson solves using the standard
CPU solver to create a dataset mapping Charge Density (rho) -> Potential (phi).

The dataset is saved in HDF5 format, compatible with Polymathic's "The Well".

Equation:
    nabla^2 phi = -rho / epsilon_0

Method:
    1. Generate randomized charge density distributions (Gaussian blobs, noise).
    2. Solve Poisson equation using Spectral method (FFT) or SOR.
    3. Save pair (rho, phi).
"""

import numpy as np

# Constants
EPSILON_0 = 8.854e-12
N_SAMPLES = 1000 # Keep 1000 or reduce if slow
GRID_SIZE = 64

def generate_random_density(grid_size: int) -> np.ndarray:
    """Generate a randomized 3D charge density field."""
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    rho = np.zeros((grid_size, grid_size, grid_size))

    # Add random Gaussian blobs
    n_blobs = np.random.randint(1, 10)
    for _ in range(n_blobs):
        x0, y0, z0 = np.random.uniform(-0.8, 0.8, 3)
        sigma = np.random.uniform(0.1, 0.3)
        amp = np.random.uniform(-1e-6, 1e-6)  # Coulombs/m^3

        blob = amp * np.exp(-((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2) / (2*sigma**2))
        rho += blob

    return rho

def solve_poisson_fft(rho: np.ndarray, L: float = 2.0) -> np.ndarray:
    """Solve Poisson equation using FFT (Periodic boundary conditions).

    nabla^2 phi = -rho / eps0

    In k-space: -k^2 * phi_k = -rho_k / eps0
    => phi_k = rho_k / (eps0 * k^2)
    """
    nx, ny, nz = rho.shape

    # Fourier transform density
    rho_k = np.fft.fftn(rho)

    # Wave numbers
    kx = np.fft.fftfreq(nx, d=L/nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=L/ny) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=L/nz) * 2 * np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    # Avoid division by zero at k=0 (mean potential is uniform, set to 0)
    K2[0, 0, 0] = 1.0
    rho_k[0, 0, 0] = 0.0 # Enforce charge neutrality (periodic)

    # Solve in k-space
    phi_k = rho_k / (EPSILON_0 * K2)

    # Inverse FFT
    phi = np.real(np.fft.ifftn(phi_k))
    return phi

def main():
    print(f"Generating {N_SAMPLES} samples of Poisson data (Grid {GRID_SIZE}^3)...")

    save_path = "src/dpf/ai/training_data_poisson.npz"

    rhos = []
    phis = []

    for i in range(N_SAMPLES):
        if i % 100 == 0:
            print(f"Generated {i}/{N_SAMPLES}")
        rho = generate_random_density(GRID_SIZE)
        phi = solve_poisson_fft(rho)

        rhos.append(rho.astype('f4'))
        phis.append(phi.astype('f4'))

    print("Saving to .npz...")
    np.savez_compressed(save_path, rho=np.array(rhos), phi=np.array(phis))
    print(f"Saved dataset to {save_path}")

if __name__ == "__main__":
    main()
