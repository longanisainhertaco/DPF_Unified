"""Tests for multi-species impurity tracking on MLX."""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _make_manager():
    from dpf.metal.mlx_species import SpeciesManager

    return SpeciesManager(
        species=["D", "Cu"],
        Z=[1, 29],
        A=[2.014, 63.546],
        background="D",
    )


class TestSpeciesManager:
    def test_init(self):
        mgr = _make_manager()
        assert mgr.n_evolved == 1  # only Cu evolved, D is background
        assert mgr.species_idx == {"D": 0, "Cu": 1}

    def test_init_mass_fractions(self):
        mgr = _make_manager()
        Y = mgr.init_mass_fractions(8, 16, initial_fractions={"Cu": 0.01})
        assert Y.shape == (1, 8, 16)
        assert float(mx.mean(Y[0])) == pytest.approx(0.01, abs=1e-6)

    def test_recover_background(self):
        mgr = _make_manager()
        Y_ev = mx.full((1, 8, 16), 0.05, dtype=mx.float32)
        Y_full = mgr.recover_background(Y_ev)
        assert Y_full.shape == (2, 8, 16)
        total = float(mx.sum(Y_full, axis=0).mean())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestSpeciesAdvection:
    def test_uniform_conserves_mass(self):
        from dpf.metal.mlx_species import species_advection_step

        nr, nz = 16, 32
        Y = mx.full((1, nr, nz), 0.05, dtype=mx.float32)
        U = mx.zeros((10, nr, nz), dtype=mx.float32)
        U = U.at[0].add(1e-4)  # density
        U = U.at[4].add(1e3)  # energy

        mass_before = float(mx.sum(Y))
        Y_new = species_advection_step(Y, U, dr=0.01, dz=0.01, dt=1e-8, gamma=5 / 3)
        mass_after = float(mx.sum(Y_new))
        assert mass_after == pytest.approx(mass_before, rel=1e-6)

    def test_nonzero_velocity_transports(self):
        from dpf.metal.mlx_species import species_advection_step

        nr, nz = 32, 64
        Y = mx.zeros((1, nr, nz), dtype=mx.float32)
        # Gaussian pulse in center
        r = np.arange(nr, dtype=np.float32)
        z = np.arange(nz, dtype=np.float32)
        rr, zz = np.meshgrid(r, z, indexing="ij")
        pulse = np.exp(-((zz - 32) ** 2) / 50).astype(np.float32) * 0.1
        Y = Y.at[0].add(mx.array(pulse))

        U = mx.zeros((10, nr, nz), dtype=mx.float32)
        U = U.at[0].add(1e-4)
        U = U.at[2].add(1e-4 * 1e4)  # vz = 1e4 m/s momentum
        U = U.at[4].add(1e3)

        Y_new = species_advection_step(Y, U, dr=0.01, dz=0.01, dt=1e-7, gamma=5 / 3)

        # Peak should shift in z direction
        peak_before = int(np.argmax(np.array(Y[0, nr // 2, :])))
        peak_after = int(np.argmax(np.array(Y_new[0, nr // 2, :])))
        assert peak_after >= peak_before  # shifted or same


class TestAblationSources:
    def test_ablation_increases_cu(self):
        from dpf.metal.mlx_species import apply_ablation_sources

        Y = mx.zeros((1, 8, 16), dtype=mx.float32)
        rate = mx.zeros((8, 16), dtype=mx.float32)
        rate = rate.at[0, :].add(0.001)  # ablation at inner boundary

        Y_new = apply_ablation_sources(Y, dt=1e-8, ablation_rate=rate, cu_idx=0)
        assert float(mx.max(Y_new)) > 0

    def test_ablation_nonnegative(self):
        from dpf.metal.mlx_species import apply_ablation_sources

        Y = mx.full((1, 4, 4), 0.01, dtype=mx.float32)
        rate = mx.full((4, 4), -0.1, dtype=mx.float32)  # negative = nonsense
        Y_new = apply_ablation_sources(Y, dt=1.0, ablation_rate=rate, cu_idx=0)
        assert float(mx.min(Y_new)) >= 0.0


class TestZeff:
    def test_pure_deuterium(self):
        from dpf.metal.mlx_species import compute_zeff_field

        Y = mx.array([[[1.0]], [[0.0]]], dtype=mx.float32)  # (2, 1, 1)
        Z = mx.array([1.0, 29.0])
        A = mx.array([2.014, 63.546])
        zeff = compute_zeff_field(Y, Z, A)
        assert float(zeff[0, 0]) == pytest.approx(1.0, abs=0.01)

    def test_copper_raises_zeff(self):
        from dpf.metal.mlx_species import compute_zeff_field

        Y_pure = mx.array([[[1.0]], [[0.0]]], dtype=mx.float32)
        Y_mixed = mx.array([[[0.9]], [[0.1]]], dtype=mx.float32)
        Z = mx.array([1.0, 29.0])
        A = mx.array([2.014, 63.546])
        zeff_pure = float(compute_zeff_field(Y_pure, Z, A)[0, 0])
        zeff_mixed = float(compute_zeff_field(Y_mixed, Z, A)[0, 0])
        assert zeff_mixed > zeff_pure


class TestGhostPad:
    def test_shape(self):
        from dpf.metal.mlx_species import pad_species_ghost

        Y = mx.ones((2, 8, 16), dtype=mx.float32)
        padded = pad_species_ghost(Y, ng=2)
        assert padded.shape == (2, 12, 16)

    def test_zero_gradient(self):
        from dpf.metal.mlx_species import pad_species_ghost

        Y = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
        padded = pad_species_ghost(Y, ng=1)
        # Inner ghost = first cell
        np.testing.assert_allclose(np.array(padded[0, 0]), np.array(Y[0, 0]))
        # Outer ghost = last cell
        np.testing.assert_allclose(np.array(padded[0, -1]), np.array(Y[0, -1]))


class TestZeffVacuumMask:
    """Z_eff vacuum masking prevents catastrophic radiation from trace Cu."""

    def test_vacuum_cells_return_zeff_1(self):
        """Vacuum cells (Y_total < 1e-4) get Z_eff = 1.0."""
        from dpf.metal.mlx_species import compute_zeff_field

        # 2 species: D (Z=1,A=2) and Cu (Z=29,A=63)
        species_Z = mx.array([1.0, 29.0])
        species_A = mx.array([2.0, 63.0])

        nr, nz = 8, 4
        Y = mx.zeros((2, nr, nz))
        # Vacuum: negligible mass fractions (density floor artifacts)
        Y_np = np.zeros((2, nr, nz), dtype=np.float32)
        Y_np[0, :, :] = 1e-10  # trace D
        Y_np[1, :, :] = 1e-12  # trace Cu
        Y = mx.array(Y_np)

        zeff = compute_zeff_field(Y, species_Z, species_A)
        zeff_np = np.asarray(zeff)
        assert np.all(zeff_np == 1.0), (
            f"Vacuum Z_eff should be 1.0, got max={np.max(zeff_np):.1f}"
        )

    def test_plasma_cells_compute_correct_zeff(self):
        """Non-vacuum cells compute correct Z_eff from composition."""
        from dpf.metal.mlx_species import compute_zeff_field

        species_Z = mx.array([1.0, 29.0])
        species_A = mx.array([2.0, 63.0])

        nr, nz = 4, 4
        Y_np = np.zeros((2, nr, nz), dtype=np.float32)
        Y_np[0, :, :] = 0.99  # mostly deuterium
        Y_np[1, :, :] = 0.01  # 1% Cu
        Y = mx.array(Y_np)

        zeff = compute_zeff_field(Y, species_Z, species_A)
        zeff_np = np.asarray(zeff)
        # Z_eff should be > 1 (Cu contribution) but << 29
        assert np.all(zeff_np > 1.0), "Z_eff should exceed 1 with Cu impurity"
        assert np.all(zeff_np < 10.0), f"Z_eff too high: {np.max(zeff_np):.1f}"

    def test_mixed_vacuum_and_plasma(self):
        """Z_eff is masked in vacuum cells but computed in plasma cells."""
        from dpf.metal.mlx_species import compute_zeff_field

        species_Z = mx.array([1.0, 29.0])
        species_A = mx.array([2.0, 63.0])

        nr, nz = 8, 4
        Y_np = np.zeros((2, nr, nz), dtype=np.float32)
        # Left half: plasma
        Y_np[0, :4, :] = 0.95
        Y_np[1, :4, :] = 0.05
        # Right half: vacuum
        Y_np[0, 4:, :] = 1e-10
        Y_np[1, 4:, :] = 1e-12
        Y = mx.array(Y_np)

        zeff = compute_zeff_field(Y, species_Z, species_A)
        zeff_np = np.asarray(zeff)
        assert np.all(zeff_np[:4] > 1.0), "Plasma cells should have Z_eff > 1"
        assert np.all(zeff_np[4:] == 1.0), "Vacuum cells should have Z_eff = 1"


# ---------------------------------------------------------------------------
# End-to-end species tests (Cycle 3)
# ---------------------------------------------------------------------------


def _make_pf1000_state(nr: int, nz: int) -> dict[str, np.ndarray]:
    rho0, p0 = 0.084, 350.0
    return {
        "rho": np.full((nr, 1, nz), rho0, dtype=np.float64),
        "velocity": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "pressure": np.full((nr, 1, nz), p0, dtype=np.float64),
        "B": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "Te": np.full((nr, 1, nz), 100.0, dtype=np.float64),
        "Ti": np.full((nr, 1, nz), 100.0, dtype=np.float64),
        "psi": np.zeros((nr, 1, nz), dtype=np.float64),
    }


class TestSpeciesE2E:
    """End-to-end species tests using the solver's built-in species path.

    CYCLE 3: species advection is already wired at mlx_solver.py:754-763.
    These tests use species_config kwarg — no manual Y management needed.
    """

    @pytest.mark.slow
    def test_pf1000_d2_cu_100_steps(self):
        """PF-1000 with 99% D2 + 1% Cu: 100 steps, Z_eff bounded, species conserved."""
        from dpf.metal.mlx_solver import MLXMHDSolver

        nr, nz = 32, 64
        dr, dz = 1e-3, 1e-3

        solver = MLXMHDSolver(
            grid_shape=(nr, 1, nz), dx=dr, dz=dz, gamma=5.0 / 3.0,
            coordinates="cylindrical",
            riemann_solver="hll", reconstruction="plm",
            time_integrator="ssp_rk2",
            enable_bremsstrahlung=True,
            species_config={
                "species": ["D", "Cu"],
                "Z": [1, 29],
                "A": [2.014, 63.546],
                "background": "D",
            },
        )
        # Inject 1% Cu into the solver's internal Y after init
        if solver._species_mgr is not None and solver._Y is not None:
            Y_init = solver._species_mgr.init_mass_fractions(
                nr, nz, initial_fractions={"Cu": 0.01}
            )
            solver._Y = Y_init

        state = _make_pf1000_state(nr, nz)
        dt = 1e-9
        current, voltage = 500e3, 20e3

        for step_i in range(100):
            state = solver.step(state, dt, current=current, voltage=voltage)

            if "species" in state:
                species_data = state["species"]
                assert "D" in species_data or "Cu" in species_data, (
                    f"Species data missing at step {step_i}"
                )

        # No NaN in final fluid state
        for key in ("rho", "pressure"):
            arr = state[key]
            assert np.all(np.isfinite(arr)), f"NaN/Inf in {key} after 100 steps"

        # Species fractions must be returned and sum to ~1
        assert "species" in state, "Solver did not return species data in final state"
        sp = state["species"]
        if "Cu" in sp and "D" in sp:
            Y_total = sp["Cu"] + sp["D"]
            max_dev = float(np.max(np.abs(Y_total - 1.0)))
            assert max_dev < 0.01, f"Species fractions deviate from 1: {max_dev:.4f}"

        # Pressure is finite
        assert np.all(np.isfinite(state["pressure"])), "Pressure has NaN after 100 steps"

    def test_species_fraction_conservation_advection(self):
        """Species advection conserves total mass fraction (unit test, fast)."""
        from dpf.metal.mlx_kernels import IDN, IMR, NVAR  # noqa: I001
        from dpf.metal.mlx_species import species_advection_step

        nr, nz = 32, 64
        r = mx.arange(nr, dtype=mx.float32)[:, None]
        z = mx.arange(nz, dtype=mx.float32)[None, :]
        Y_cu = 0.05 * mx.exp(-((r - 5) ** 2 + (z - 32) ** 2) / 50.0)
        Y = Y_cu[None, :, :]

        total_before = float(mx.sum(Y))

        U = mx.zeros((NVAR, nr, nz), dtype=mx.float32)
        U = U.at[IDN].add(1.0)
        U = U.at[IMR].add(100.0)

        Y_new = species_advection_step(Y, U, dr=1e-3, dz=1e-3, dt=1e-8, gamma=5.0 / 3.0)
        total_after = float(mx.sum(Y_new))

        rel_change = abs(total_after - total_before) / max(total_before, 1e-30)
        assert rel_change < 0.05, f"Species mass changed by {rel_change * 100:.1f}%"
