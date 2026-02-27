"""Bremsstrahlung radiation coupling tests for the Metal MHD solver.

Verifies that the MetalMHDSolver correctly applies bremsstrahlung
radiation cooling as an operator-split energy sink.  Tests cover:

1. Te cooling: electron temperature decreases under bremsstrahlung
2. Pressure coupling: pressure decreases consistent with Te change
3. Energy accounting: radiated energy matches Te × ne × k_B change
4. Implicit stability: no negative Te even for large dt
5. Hot-spot cooling: localized hot region cools faster than ambient
6. Comparison with standalone bremsstrahlung module

References:
    Rybicki & Lightman (1979), Eq. 5.14a.
    NRL Plasma Formulary (2019), p. 58.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402


def _make_uniform_state(
    nx: int = 8,
    rho: float = 1e-3,
    Te: float = 1e7,
    Ti: float = 1e7,
    p: float | None = None,
    B0: float = 0.0,
) -> tuple[dict, float]:
    """Create a uniform plasma state for bremsstrahlung tests.

    Args:
        nx: Grid size (cubic).
        rho: Mass density [kg/m^3].
        Te: Electron temperature [K].
        Ti: Ion temperature [K].
        p: Pressure [Pa].  If None, computed from ideal gas: p = n*k_B*(Te+Ti).
        B0: Background magnetic field in z [T].

    Returns:
        (state_dict, dx).
    """
    dx = 0.01
    k_B = 1.380649e-23
    m_D = 3.34358377e-27

    if p is None:
        ne = rho / m_D
        p = ne * k_B * (Te + Ti)

    state = {
        "rho": np.full((nx, nx, nx), rho, dtype=np.float64),
        "velocity": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "pressure": np.full((nx, nx, nx), p, dtype=np.float64),
        "B": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "Te": np.full((nx, nx, nx), Te, dtype=np.float64),
        "Ti": np.full((nx, nx, nx), Ti, dtype=np.float64),
        "psi": np.zeros((nx, nx, nx), dtype=np.float64),
    }
    if B0 > 0:
        state["B"][2] = B0

    return state, dx


class TestBremsstrahlungCooling:
    """Verify Te decreases and pressure drops from bremsstrahlung."""

    def test_te_decreases(self):
        """Electron temperature must decrease after bremsstrahlung step."""
        nx = 8
        Te0 = 1e7  # 10 MK — DPF-relevant temperature
        state, dx = _make_uniform_state(nx=nx, Te=Te0)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )
        dt = solver.compute_dt(state)
        out = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        Te_new = np.mean(out["Te"])
        assert Te_new < Te0, f"Te should decrease: {Te_new:.6e} >= {Te0:.6e}"

    def test_pressure_decreases(self):
        """Pressure must decrease when Te cools (Ti unchanged)."""
        nx = 8
        Te0 = 1e7
        state, dx = _make_uniform_state(nx=nx, Te=Te0)
        p0 = np.mean(state["pressure"])

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )
        dt = solver.compute_dt(state)
        out = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        p_new = np.mean(out["pressure"])
        assert p_new < p0, f"Pressure should decrease: {p_new:.6e} >= {p0:.6e}"

    def test_ti_unchanged(self):
        """Ion temperature should not be affected by bremsstrahlung."""
        nx = 8
        Ti0 = 5e6
        state, dx = _make_uniform_state(nx=nx, Te=1e7, Ti=Ti0)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )
        dt = solver.compute_dt(state)
        out = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        # Ti is passed through unchanged by bremsstrahlung
        # (MHD step may slightly modify due to flux roundoff, but not brem)
        np.testing.assert_allclose(
            out["Ti"], state["Ti"], rtol=1e-4,
            err_msg="Ti should be unaffected by bremsstrahlung",
        )

    def test_no_bremsstrahlung_when_disabled(self):
        """Te should not change from bremsstrahlung when disabled."""
        nx = 8
        Te0 = 1e7
        state, dx = _make_uniform_state(nx=nx, Te=Te0)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=False,
        )
        dt = solver.compute_dt(state)
        out = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        # Te is passed through when bremsstrahlung is off
        np.testing.assert_allclose(
            out["Te"], state["Te"], rtol=1e-6,
            err_msg="Te should not change when bremsstrahlung is disabled",
        )


class TestBremsstrahlungEnergyAccounting:
    """Verify energy conservation under bremsstrahlung."""

    def test_radiated_energy_positive(self):
        """Total radiated energy must be positive after steps."""
        nx = 8
        state, dx = _make_uniform_state(nx=nx, Te=1e7)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )
        for _ in range(5):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        assert solver.total_radiated_energy > 0, (
            f"Radiated energy should be positive: {solver.total_radiated_energy}"
        )

    def test_pressure_drop_matches_te_drop(self):
        """Pressure drop should be consistent with ne*k_B*(Te_old-Te_new)."""
        nx = 8
        rho = 1e-3
        Te0 = 1e7
        Ti0 = 5e6
        k_B = 1.380649e-23
        m_D = 3.34358377e-27
        ne = rho / m_D

        state, dx = _make_uniform_state(nx=nx, rho=rho, Te=Te0, Ti=Ti0)
        p0 = np.mean(state["pressure"])

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float64",
            use_ct=False,
            enable_bremsstrahlung=True,
        )

        # Use a small dt where MHD flux divergence is negligible
        dt = 1e-12
        out = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        Te_new = np.mean(out["Te"])
        p_new = np.mean(out["pressure"])
        dTe = Te0 - Te_new
        dp_expected = ne * k_B * dTe
        dp_actual = p0 - p_new

        # Tolerance: 10% — MHD flux roundoff can shift pressure slightly
        if dp_expected > 0:
            rel_err = abs(dp_actual - dp_expected) / dp_expected
            assert rel_err < 0.10, (
                f"Pressure drop mismatch: actual={dp_actual:.4e}, "
                f"expected={dp_expected:.4e}, rel_err={rel_err:.2%}"
            )


class TestBremsstrahlungImplicitStability:
    """Verify implicit backward Euler prevents negative Te."""

    def test_large_dt_no_negative_te(self):
        """Even with unrealistically large dt, Te stays positive."""
        nx = 8
        state, dx = _make_uniform_state(nx=nx, Te=1e6)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )

        # Override with large dt (10x the CFL timestep)
        dt_cfl = solver.compute_dt(state)
        dt_large = 10.0 * dt_cfl

        out = solver.step(state, dt=dt_large, current=0.0, voltage=0.0)

        assert np.all(out["Te"] > 0), "Te went negative — implicit scheme failed"
        assert np.all(out["pressure"] > 0), "Pressure went negative"

    def test_cold_plasma_minimal_cooling(self):
        """Cold plasma (Te=1000 K) should have negligible bremsstrahlung."""
        nx = 8
        Te0 = 1000.0  # Room-temperature plasma
        state, dx = _make_uniform_state(nx=nx, Te=Te0, rho=1e-6)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float64",
            use_ct=False,
            enable_bremsstrahlung=True,
        )
        dt = solver.compute_dt(state)
        out = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        # At low Te and low ne, bremsstrahlung power is tiny
        Te_new = np.mean(out["Te"])
        rel_change = abs(Te_new - Te0) / Te0
        assert rel_change < 0.01, (
            f"Cold plasma should have negligible cooling: {rel_change:.4%}"
        )


class TestBremsstrahlungHotSpot:
    """Verify localized hot regions cool faster than ambient."""

    def test_hot_spot_cools_faster(self):
        """A hot spot should lose more Te than the surrounding plasma."""
        nx = 16
        rho = 1e-3
        Te_ambient = 1e6
        Te_hot = 1e8  # 100x hotter
        state, dx = _make_uniform_state(nx=nx, Te=Te_ambient, rho=rho)

        # Place hot spot in center
        cx, cy, cz = nx // 2, nx // 2, nx // 2
        state["Te"][cx-1:cx+1, cy-1:cy+1, cz-1:cz+1] = Te_hot
        # Recompute pressure for hot spot cells
        k_B = 1.380649e-23
        m_D = 3.34358377e-27
        ne = rho / m_D
        state["pressure"] = ne * k_B * (state["Te"] + state["Ti"])

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )

        dt = solver.compute_dt(state)
        out = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        Te_hot_after = np.mean(
            out["Te"][cx-1:cx+1, cy-1:cy+1, cz-1:cz+1]
        )
        Te_ambient_after = np.mean(out["Te"][0, 0, 0])

        dTe_hot = Te_hot - Te_hot_after
        dTe_ambient = Te_ambient - Te_ambient_after

        assert dTe_hot > dTe_ambient, (
            f"Hot spot should cool more: dTe_hot={dTe_hot:.4e}, "
            f"dTe_ambient={dTe_ambient:.4e}"
        )


class TestBremsstrahlungScaling:
    """Verify P_brem ~ ne^2 * sqrt(Te) scaling."""

    def test_density_squared_scaling(self):
        """Doubling density should ~4x the radiated power."""
        nx = 8
        Te0 = 1e7

        powers = []
        for rho in [1e-3, 2e-3]:
            state, dx = _make_uniform_state(nx=nx, rho=rho, Te=Te0)
            solver = MetalMHDSolver(
                grid_shape=(nx, nx, nx),
                dx=dx,
                device="cpu",
                precision="float64",
                use_ct=False,
                enable_bremsstrahlung=True,
            )
            solver.total_radiated_energy = 0.0
            dt = 1e-12  # Very small dt to isolate bremsstrahlung
            solver.step(state, dt=dt, current=0.0, voltage=0.0)
            powers.append(solver.total_radiated_energy)

        # P ~ ne^2, so doubling rho → 4x power
        ratio = powers[1] / powers[0]
        assert 3.0 < ratio < 5.0, (
            f"Expected ~4x power ratio, got {ratio:.2f} "
            f"(P1={powers[0]:.4e}, P2={powers[1]:.4e})"
        )

    def test_temperature_sqrt_scaling(self):
        """Quadrupling Te should ~2x the bremsstrahlung power."""
        nx = 8
        rho = 1e-3

        powers = []
        for Te in [1e7, 4e7]:
            state, dx = _make_uniform_state(nx=nx, rho=rho, Te=Te)
            solver = MetalMHDSolver(
                grid_shape=(nx, nx, nx),
                dx=dx,
                device="cpu",
                precision="float64",
                use_ct=False,
                enable_bremsstrahlung=True,
            )
            solver.total_radiated_energy = 0.0
            dt = 1e-12
            solver.step(state, dt=dt, current=0.0, voltage=0.0)
            powers.append(solver.total_radiated_energy)

        # P ~ sqrt(Te), so 4x Te → 2x power
        ratio = powers[1] / powers[0]
        assert 1.5 < ratio < 2.5, (
            f"Expected ~2x power ratio, got {ratio:.2f} "
            f"(P1={powers[0]:.4e}, P2={powers[1]:.4e})"
        )


class TestBremsstrahlungConsistency:
    """Cross-check with standalone bremsstrahlung module."""

    def test_matches_standalone_module(self):
        """Metal solver bremsstrahlung should match radiation.bremsstrahlung."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        nx = 8
        rho = 1e-3
        Te0 = 1e7
        m_D = 3.34358377e-27
        ne = rho / m_D

        # Standalone module
        ne_arr = np.full((nx, nx, nx), ne)
        Te_arr = np.full((nx, nx, nx), Te0)
        P_standalone = bremsstrahlung_power(ne_arr, Te_arr, Z=1.0, gaunt_factor=1.2)
        P_standalone_total = float(np.sum(P_standalone))

        # Metal solver — measure radiated power from first step
        state, dx = _make_uniform_state(nx=nx, rho=rho, Te=Te0)
        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float64",
            use_ct=False,
            enable_bremsstrahlung=True,
            gaunt_factor=1.2,
            Z_eff=1.0,
        )
        solver.total_radiated_energy = 0.0
        dt = 1e-14  # Tiny dt — explicit limit approximation
        solver.step(state, dt=dt, current=0.0, voltage=0.0)

        # P_metal = total_radiated_energy / (dt * cell_vol)
        cell_vol = dx ** 3
        P_metal_total = solver.total_radiated_energy / (dt * cell_vol)

        # Should match within 5% (implicit vs explicit difference at tiny dt)
        rel_err = abs(P_metal_total - P_standalone_total) / P_standalone_total
        assert rel_err < 0.05, (
            f"Metal P_brem={P_metal_total:.4e} vs standalone={P_standalone_total:.4e}, "
            f"rel_err={rel_err:.2%}"
        )


class TestBremsstrahlungMultiStep:
    """Verify cooling over multiple steps."""

    def test_monotonic_te_decrease(self):
        """Te should monotonically decrease over multiple steps."""
        nx = 8
        state, dx = _make_uniform_state(nx=nx, Te=5e7, rho=1e-3)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )

        Te_history = [np.mean(state["Te"])]
        for _ in range(10):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
            Te_history.append(np.mean(state["Te"]))

        # Check monotonic decrease
        for i in range(1, len(Te_history)):
            assert Te_history[i] <= Te_history[i - 1], (
                f"Te increased at step {i}: "
                f"{Te_history[i]:.4e} > {Te_history[i-1]:.4e}"
            )

    def test_cumulative_radiated_energy(self):
        """Cumulative radiated energy should increase each step."""
        nx = 8
        state, dx = _make_uniform_state(nx=nx, Te=5e7, rho=1e-3)

        solver = MetalMHDSolver(
            grid_shape=(nx, nx, nx),
            dx=dx,
            device="cpu",
            precision="float32",
            use_ct=False,
            enable_bremsstrahlung=True,
        )

        energy_history = [0.0]
        for _ in range(5):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
            energy_history.append(solver.total_radiated_energy)

        for i in range(1, len(energy_history)):
            assert energy_history[i] >= energy_history[i - 1], (
                f"Radiated energy decreased at step {i}"
            )
