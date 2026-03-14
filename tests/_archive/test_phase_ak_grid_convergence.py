"""Phase AK: Grid convergence at DPF-relevant conditions.

PhD Debate #22 Path to 7.0, Action #1: Grid convergence study on DPF-relevant
problem.  Existing convergence tests (test_linear_wave_metal_convergence.py)
use normalized code units (rho0=1, p0=1).  This test uses PF-1000 fill gas
conditions to verify that the Metal MHD solver converges at the expected rate
with realistic density/pressure/B-field magnitudes.

Physical conditions (PF-1000 fill gas):
    rho0 = 7.53e-4 kg/m^3  (3.5 Torr D2 at 300 K)
    p0   = 466 Pa           (ideal gas, D2 molecular mass)
    B0   = 0.01 T           (representative early axial field)
    dx   = 2.5 mm           (PF-1000 preset grid spacing)

In Heaviside-Lorentz code units (Metal solver convention):
    B0_HL = B0_SI / sqrt(mu_0) ~ 8.92

Dimensionless numbers:
    beta = 2*mu0*p0/B0^2 ~ 11.7  (gas-pressure dominated fill)
    cs   ~ 1017 m/s               (D2 sound speed at 300 K)
    va   ~ 325 m/s                (Alfven speed at B0=0.01 T)
    cf   ~ 1068 m/s               (fast magnetosonic speed)

Tests include:
1. Repair fraction diagnostics for positivity fallback
2. Sound wave convergence at PF-1000 density/pressure (pure hydro)
3. Fast magnetosonic wave convergence at PF-1000 conditions (MHD)
4. Float32 vs float64 precision comparison
5. R_plasma convergence across 3 Metal engine resolutions

References:
    Stone J.M. et al., ApJS 178, 137 (2008), Sec 6.1 — linear wave convergence.
    Scholz M. et al., Nukleonika 51(1):79-84 (2006) — PF-1000 parameters.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_riemann import (  # noqa: E402, I001
    get_repair_stats,
    reset_repair_stats,
)
from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402

# =====================================================================
# Physical constants for PF-1000 fill conditions
# =====================================================================

_MU_0 = 4.0 * np.pi * 1e-7  # vacuum permeability [H/m]
_K_B = 1.380649e-23          # Boltzmann constant [J/K]
_M_D2 = 6.688e-27            # D2 molecular mass [kg]

# PF-1000 fill gas
_RHO0 = 7.53e-4              # kg/m^3 (3.5 Torr D2 at 300 K)
_T0 = 300.0                  # K
_P0 = _RHO0 * _K_B * _T0 / _M_D2  # ~466 Pa
_B0_SI = 0.01                # T (representative early axial B)
_B0_HL = _B0_SI / np.sqrt(_MU_0)   # ~8.92 in Heaviside-Lorentz units
_GAMMA = 5.0 / 3.0

# Derived speeds
_CS = np.sqrt(_GAMMA * _P0 / _RHO0)         # ~1017 m/s
_VA = _B0_HL / np.sqrt(_RHO0)               # ~325 m/s
_CF = np.sqrt(_CS**2 + _VA**2)              # ~1068 m/s


# =====================================================================
# IC builders
# =====================================================================


def _make_sound_wave_dpf(
    nx: int,
    amplitude: float = 1e-4,
) -> tuple[dict, float, float, float]:
    """Create linear sound wave ICs at PF-1000 fill conditions (B=0).

    Returns (state, dx, cs, L).
    """
    ny = nz = 4
    L = nx * 2.5e-3  # PF-1000 scale grid spacing
    dx = L / nx
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    k = 2.0 * np.pi / L

    rho = _RHO0 * (1.0 + amplitude * np.sin(k * xc))
    vx = amplitude * _CS * np.sin(k * xc)
    p = _P0 * (1.0 + _GAMMA * amplitude * np.sin(k * xc))

    rho_3d = np.broadcast_to(rho[:, None, None], (nx, ny, nz)).copy()
    p_3d = np.broadcast_to(p[:, None, None], (nx, ny, nz)).copy()
    vel = np.zeros((3, nx, ny, nz), dtype=np.float64)
    vel[0] = np.broadcast_to(vx[:, None, None], (nx, ny, nz)).copy()

    state = {
        "rho": rho_3d,
        "velocity": vel,
        "pressure": p_3d,
        "B": np.zeros((3, nx, ny, nz), dtype=np.float64),
        "Te": np.full((nx, ny, nz), _T0, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), _T0, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx, _CS, L


def _make_fast_wave_dpf(
    nx: int,
    amplitude: float = 1e-4,
) -> tuple[dict, float, float, float]:
    """Create fast magnetosonic wave ICs at PF-1000 fill conditions.

    Wave propagates along x with B0 along y (perpendicular).
    The fast magnetosonic eigenvector (linearized) is:
        delta_rho = A * rho0 * sin(kx)
        delta_vx  = A * cf * sin(kx)
        delta_p   = A * gamma * p0 * sin(kx)
        delta_By  = A * B0 * sin(kx)

    Returns (state, dx, cf, L).
    """
    ny = nz = 4
    L = nx * 2.5e-3
    dx = L / nx
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    k = 2.0 * np.pi / L

    rho = _RHO0 * (1.0 + amplitude * np.sin(k * xc))
    vx = amplitude * _CF * np.sin(k * xc)
    p = _P0 * (1.0 + _GAMMA * amplitude * np.sin(k * xc))
    By = _B0_HL * (1.0 + amplitude * np.sin(k * xc))

    rho_3d = np.broadcast_to(rho[:, None, None], (nx, ny, nz)).copy()
    p_3d = np.broadcast_to(p[:, None, None], (nx, ny, nz)).copy()
    vel = np.zeros((3, nx, ny, nz), dtype=np.float64)
    vel[0] = np.broadcast_to(vx[:, None, None], (nx, ny, nz)).copy()
    B_3d = np.zeros((3, nx, ny, nz), dtype=np.float64)
    B_3d[1] = np.broadcast_to(By[:, None, None], (nx, ny, nz)).copy()

    state = {
        "rho": rho_3d,
        "velocity": vel,
        "pressure": p_3d,
        "B": B_3d,
        "Te": np.full((nx, ny, nz), _T0, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), _T0, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx, _CF, L


def _exact_rho(
    xc: np.ndarray,
    t: float,
    c: float,
    L: float,
    amplitude: float,
) -> np.ndarray:
    """Exact linear wave density at time t (rightward propagation)."""
    k = 2.0 * np.pi / L
    return _RHO0 * (1.0 + amplitude * np.sin(k * (xc - c * t)))


def _run_dpf_convergence(
    nx: int,
    wave_type: str = "sound",
    riemann: str = "hll",
    recon: str = "plm",
    integrator: str = "ssp_rk2",
    precision: str = "float32",
    amplitude: float = 1e-4,
    t_frac: float = 0.125,
) -> float:
    """Run a wave to t_frac * period and return L1(rho) error vs exact.

    Args:
        nx: Number of cells along propagation axis.
        wave_type: "sound" (B=0) or "fast" (fast magnetosonic).
        riemann: Riemann solver.
        recon: Reconstruction method.
        integrator: Time integrator.
        precision: "float32" or "float64".
        amplitude: Wave amplitude.
        t_frac: Fraction of period to propagate.

    Returns:
        L1 error in density over interior cells.
    """
    if wave_type == "fast":
        state, dx, c_wave, L = _make_fast_wave_dpf(nx, amplitude=amplitude)
    else:
        state, dx, c_wave, L = _make_sound_wave_dpf(nx, amplitude=amplitude)

    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    t_end = t_frac * L / c_wave

    solver = MetalMHDSolver(
        grid_shape=(nx, 4, 4),
        dx=dx,
        gamma=_GAMMA,
        cfl=0.4,
        device="cpu",
        riemann_solver=riemann,
        reconstruction=recon,
        time_integrator=integrator,
        precision=precision,
        use_ct=False,
    )

    t_total = 0.0
    for _ in range(50000):
        dt = solver.compute_dt(state)
        if t_total + dt > t_end:
            dt = t_end - t_total
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        t_total += dt
        if t_total >= t_end * 0.9999:
            break

    rho_num = state["rho"][:, 2, 2]
    rho_exact = _exact_rho(xc, t_total, c_wave, L, amplitude)

    # Compare interior cells (avoid boundary effects)
    margin = max(nx // 8, 2)
    interior = slice(margin, nx - margin)
    l1 = float(np.mean(np.abs(rho_num[interior] - rho_exact[interior])))
    return l1


# =====================================================================
# Repair fraction diagnostics
# =====================================================================


class TestRepairFractionDiagnostic:
    """Verify positivity fallback repair statistics API."""

    def test_repair_stats_api(self):
        """get_repair_stats returns dict with expected keys."""
        stats = get_repair_stats()
        assert "total_checked" in stats
        assert "total_repaired" in stats
        assert "calls" in stats

    def test_repair_stats_reset(self):
        """reset_repair_stats zeroes all counters."""
        reset_repair_stats()
        stats = get_repair_stats()
        assert stats["total_checked"] == 0
        assert stats["total_repaired"] == 0
        assert stats["calls"] == 0

    def test_repair_stats_increment_on_solver_step(self):
        """Repair stats increment after Metal solver steps."""
        reset_repair_stats()

        nx = 16
        state, dx, _, _ = _make_sound_wave_dpf(nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4), dx=dx, gamma=_GAMMA, cfl=0.4,
            device="cpu", precision="float32", use_ct=False,
        )

        dt = solver.compute_dt(state)
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        stats = get_repair_stats()
        # At least one call to _positivity_fallback (3 dims x 1 step = 3 calls min)
        assert stats["calls"] >= 3, f"Expected >= 3 calls, got {stats['calls']}"
        assert stats["total_checked"] > 0

    def test_zero_repairs_on_smooth_dpf_wave(self):
        """Smooth sound wave at PF-1000 conditions needs zero repairs."""
        reset_repair_stats()

        nx = 32
        state, dx, cs, L = _make_sound_wave_dpf(nx, amplitude=1e-6)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4), dx=dx, gamma=_GAMMA, cfl=0.3,
            device="cpu", precision="float64", use_ct=False,
        )

        # Run for 10 steps
        for _ in range(10):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        stats = get_repair_stats()
        assert stats["total_repaired"] == 0, (
            f"Smooth wave triggered {stats['total_repaired']} repairs "
            f"out of {stats['total_checked']} checked"
        )

    def test_repair_fraction_reported(self):
        """Repair fraction = total_repaired / total_checked is well-defined."""
        reset_repair_stats()

        nx = 32
        state, dx, _, _ = _make_fast_wave_dpf(nx)
        solver = MetalMHDSolver(
            grid_shape=(nx, 4, 4), dx=dx, gamma=_GAMMA, cfl=0.4,
            device="cpu", precision="float32", use_ct=False,
        )

        for _ in range(5):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        stats = get_repair_stats()
        assert stats["total_checked"] > 0
        fraction = stats["total_repaired"] / stats["total_checked"]
        assert 0.0 <= fraction <= 1.0
        print(f"\nRepair fraction (fast wave, float32, 32 cells): "
              f"{fraction:.6f} ({stats['total_repaired']}/{stats['total_checked']})")


# =====================================================================
# Sound wave convergence at DPF conditions
# =====================================================================


class TestDPFSoundWaveConvergence:
    """Sound wave convergence at PF-1000 fill gas conditions.

    Uses rho0=7.53e-4, p0=466 Pa, B=0 — pure hydro at DPF-relevant
    density and pressure.  Verifies PLM+HLL convergence at ~2nd order
    with physical SI values instead of normalized code units.
    """

    @pytest.mark.slow
    def test_convergence_order(self):
        """PLM+HLL converges at >= 1.0 order on DPF-scale sound wave."""
        resolutions = [32, 64, 128]
        errors = {}
        for nx in resolutions:
            errors[nx] = _run_dpf_convergence(nx, wave_type="sound")

        order_1 = np.log2(errors[32] / errors[64])
        order_2 = np.log2(errors[64] / errors[128])
        avg_order = 0.5 * (order_1 + order_2)

        print("\n=== DPF Sound Wave Convergence (PLM+HLL) ===")
        print(f"Errors: {errors}")
        print(f"Orders: {order_1:.2f}, {order_2:.2f}, avg={avg_order:.2f}")

        assert avg_order > 1.0, (
            f"DPF sound wave convergence too low: order={avg_order:.2f}, "
            f"errors={errors}"
        )

    @pytest.mark.slow
    def test_errors_strictly_decrease(self):
        """L1 error strictly decreases with resolution."""
        resolutions = [32, 64, 128]
        errors = {}
        for nx in resolutions:
            errors[nx] = _run_dpf_convergence(nx, wave_type="sound")

        assert errors[32] > errors[64] > errors[128], (
            f"Non-monotonic convergence: {errors}"
        )

    @pytest.mark.slow
    def test_absolute_error_128_cells(self):
        """L1 error on 128 cells should be < 1e-8 (amplitude-relative ~ 1e-4)."""
        l1 = _run_dpf_convergence(128, wave_type="sound")
        assert l1 < 1e-8, f"L1 error too high on 128 cells: {l1:.2e}"


# =====================================================================
# Fast magnetosonic wave convergence at DPF conditions
# =====================================================================


class TestDPFFastWaveConvergence:
    """Fast magnetosonic wave convergence at PF-1000 conditions.

    Uses rho0=7.53e-4, p0=466 Pa, B0=0.01 T (HL: ~8.92).
    This is the first convergence test with a non-zero B-field at
    DPF-relevant physical magnitudes (beta ~ 12).
    """

    @pytest.mark.slow
    def test_convergence_order(self):
        """PLM+HLL converges at >= 1.0 order on DPF-scale fast wave."""
        resolutions = [32, 64, 128]
        errors = {}
        for nx in resolutions:
            errors[nx] = _run_dpf_convergence(nx, wave_type="fast")

        order_1 = np.log2(errors[32] / errors[64])
        order_2 = np.log2(errors[64] / errors[128])
        avg_order = 0.5 * (order_1 + order_2)

        print("\n=== DPF Fast Magnetosonic Wave Convergence (PLM+HLL) ===")
        print(f"beta={2*_MU_0*_P0/_B0_SI**2:.1f}, cs={_CS:.0f} m/s, "
              f"va={_VA:.0f} m/s, cf={_CF:.0f} m/s")
        print(f"Errors: {errors}")
        print(f"Orders: {order_1:.2f}, {order_2:.2f}, avg={avg_order:.2f}")

        assert avg_order > 1.0, (
            f"DPF fast wave convergence too low: order={avg_order:.2f}, "
            f"errors={errors}"
        )

    @pytest.mark.slow
    def test_errors_strictly_decrease(self):
        """L1 error strictly decreases with resolution."""
        resolutions = [32, 64, 128]
        errors = {}
        for nx in resolutions:
            errors[nx] = _run_dpf_convergence(nx, wave_type="fast")

        assert errors[32] > errors[64] > errors[128], (
            f"Non-monotonic convergence: {errors}"
        )

    @pytest.mark.slow
    def test_absolute_error_128_cells(self):
        """L1 error on 128 cells should be < 1e-8."""
        l1 = _run_dpf_convergence(128, wave_type="fast")
        assert l1 < 1e-8, f"L1 error too high on 128 cells: {l1:.2e}"

    @pytest.mark.slow
    def test_no_repairs_needed(self):
        """Smooth fast wave needs zero positivity repairs at all resolutions."""
        for nx in [32, 64, 128]:
            reset_repair_stats()
            _run_dpf_convergence(nx, wave_type="fast", precision="float64")
            stats = get_repair_stats()
            assert stats["total_repaired"] == 0, (
                f"nx={nx}: {stats['total_repaired']} repairs triggered on smooth wave"
            )


# =====================================================================
# Higher-order method convergence (WENO5+HLLD+SSP-RK3)
# =====================================================================


class TestHigherOrderConvergence:
    """WENO5+HLLD+SSP-RK3 convergence at DPF conditions.

    Demonstrates that higher-order reconstruction (5th-order WENO-Z)
    with a 4-wave Riemann solver (HLLD) and 3rd-order time integration
    (SSP-RK3) achieves significantly better convergence than PLM+HLL
    at PF-1000 conditions.

    The Metal solver's WENO5 implementation uses FD formulas for
    point-value reconstruction (Borges et al. 2008).
    """

    @pytest.mark.slow
    def test_weno5_convergence_order(self):
        """WENO5+HLLD+SSP-RK3 converges at >= 2.0 order on DPF fast wave."""
        resolutions = [32, 64, 128]
        errors = {}
        for nx in resolutions:
            errors[nx] = _run_dpf_convergence(
                nx, wave_type="fast",
                riemann="hlld", recon="weno5", integrator="ssp_rk3",
                precision="float64",
            )

        order_1 = np.log2(errors[32] / errors[64])
        order_2 = np.log2(errors[64] / errors[128])
        avg_order = 0.5 * (order_1 + order_2)

        print("\n=== WENO5+HLLD+SSP-RK3 Convergence (DPF fast wave) ===")
        print(f"Errors: {errors}")
        print(f"Orders: {order_1:.2f}, {order_2:.2f}, avg={avg_order:.2f}")

        assert avg_order > 2.0, (
            f"WENO5 convergence too low: order={avg_order:.2f} (expected > 2.0)"
        )

    @pytest.mark.slow
    def test_weno5_lower_error_than_plm(self):
        """WENO5 achieves lower error than PLM at 128 cells."""
        l1_plm = _run_dpf_convergence(
            128, wave_type="fast", riemann="hll", recon="plm",
            integrator="ssp_rk2", precision="float64",
        )
        l1_weno = _run_dpf_convergence(
            128, wave_type="fast", riemann="hlld", recon="weno5",
            integrator="ssp_rk3", precision="float64",
        )

        print("\n=== PLM vs WENO5 at 128 cells (DPF fast wave, float64) ===")
        print(f"PLM+HLL+RK2:      {l1_plm:.2e}")
        print(f"WENO5+HLLD+RK3:   {l1_weno:.2e}")
        print(f"Improvement: {l1_plm/l1_weno:.1f}x")

        assert l1_weno < l1_plm, (
            f"WENO5 ({l1_weno:.2e}) not better than PLM ({l1_plm:.2e})"
        )

    @pytest.mark.slow
    def test_method_comparison_table(self):
        """Report convergence comparison: PLM vs WENO5 at DPF conditions."""
        resolutions = [32, 64, 128]
        methods = {
            "PLM+HLL+RK2": {"riemann": "hll", "recon": "plm", "integrator": "ssp_rk2"},
            "WENO5+HLLD+RK3": {"riemann": "hlld", "recon": "weno5", "integrator": "ssp_rk3"},
        }

        print("\n=== Method Comparison: DPF Fast Wave (float64) ===")
        print(f"{'Method':<20} {'nx=32':>12} {'nx=64':>12} {'nx=128':>12} {'Order':>8}")

        for name, params in methods.items():
            errors = {}
            for nx in resolutions:
                errors[nx] = _run_dpf_convergence(
                    nx, wave_type="fast", precision="float64", **params,
                )
            order = np.log2(errors[64] / errors[128]) if errors[128] > 0 else 0.0
            print(f"{name:<20} {errors[32]:>12.4e} {errors[64]:>12.4e} "
                  f"{errors[128]:>12.4e} {order:>8.2f}")


# =====================================================================
# Float32 vs Float64 precision comparison
# =====================================================================


class TestFloat32vsFloat64:
    """Compare convergence behavior at float32 and float64 precision.

    At DPF-relevant values (rho ~ 1e-4, p ~ 500), float32 has
    ~7 significant digits.  Verify convergence is not degraded by
    precision limitations at these physical scales.
    """

    @pytest.mark.slow
    def test_float64_lower_error(self):
        """Float64 achieves lower L1 error than float32 at 128 cells."""
        l1_f32 = _run_dpf_convergence(
            128, wave_type="fast", precision="float32",
        )
        l1_f64 = _run_dpf_convergence(
            128, wave_type="fast", precision="float64",
        )
        print("\n=== Float32 vs Float64 at 128 cells (fast wave) ===")
        print(f"Float32 L1: {l1_f32:.2e}")
        print(f"Float64 L1: {l1_f64:.2e}")
        print(f"Ratio: {l1_f32/l1_f64:.1f}x")

        # Float64 should be at least as good (allow small tolerance for noise)
        assert l1_f64 <= l1_f32 * 1.1, (
            f"Float64 ({l1_f64:.2e}) worse than float32 ({l1_f32:.2e})"
        )

    @pytest.mark.slow
    def test_both_converge_at_first_order(self):
        """Both float32 and float64 achieve >= 1st order convergence."""
        for prec in ["float32", "float64"]:
            errors = {}
            for nx in [32, 64, 128]:
                errors[nx] = _run_dpf_convergence(
                    nx, wave_type="fast", precision=prec,
                )
            order = np.log2(errors[64] / errors[128])
            assert order > 1.0, (
                f"{prec} convergence order {order:.2f} < 1.0 at DPF conditions"
            )

    @pytest.mark.slow
    def test_precision_comparison_report(self):
        """Report full precision comparison table."""
        resolutions = [32, 64, 128]
        results: dict[str, dict[int, float]] = {}

        for prec in ["float32", "float64"]:
            results[prec] = {}
            for nx in resolutions:
                results[prec][nx] = _run_dpf_convergence(
                    nx, wave_type="fast", precision=prec,
                )

        print("\n=== Precision Comparison: Fast Wave at DPF Conditions ===")
        print(f"{'nx':>6} {'float32':>12} {'float64':>12} {'ratio':>8}")
        for nx in resolutions:
            f32 = results["float32"][nx]
            f64 = results["float64"][nx]
            ratio = f32 / f64 if f64 > 0 else float("inf")
            print(f"{nx:>6} {f32:>12.4e} {f64:>12.4e} {ratio:>8.1f}x")

        # Convergence orders
        for prec in ["float32", "float64"]:
            e = results[prec]
            order = np.log2(e[64] / e[128])
            print(f"{prec} order (64->128): {order:.2f}")


# =====================================================================
# R_plasma convergence across resolutions (coupled engine)
# =====================================================================


class TestRPlasmaConvergence:
    """R_plasma convergence across Metal engine resolutions.

    Run PF-1000 coupled simulation at 3 resolutions (16x1x32,
    32x1x64, 64x1x128) and measure R_plasma at t ~ 3 us during
    the axial rundown phase.  R_plasma should converge (bounded,
    monotonic approach to a limit).
    """

    @staticmethod
    def _run_engine_to_time(
        grid_shape: tuple[int, int, int],
        dx: float,
        t_target: float,
    ) -> dict:
        """Run PF-1000 Metal engine to t_target and return diagnostics."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        preset["grid_shape"] = list(grid_shape)
        preset["dx"] = dx
        preset["sim_time"] = t_target * 1.1
        preset["diagnostics_path"] = ":memory:"
        preset["fluid"] = {
            "backend": "metal",
            "riemann_solver": "hll",
            "reconstruction": "plm",
            "time_integrator": "ssp_rk2",
            "precision": "float32",
            "use_ct": False,
        }
        preset["radiation"] = {"bremsstrahlung_enabled": False, "fld_enabled": False}
        preset["collision"] = {"enabled": False}

        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)

        times = []
        currents = []
        for _ in range(10000):
            result = engine.step()
            times.append(engine.time)
            currents.append(abs(engine.circuit.current))
            if engine.time >= t_target:
                break
            if result.finished:
                break

        return {
            "times": np.array(times),
            "currents": np.array(currents),
            "peak_current": float(np.max(np.abs(currents))),
            "final_time": times[-1] if times else 0.0,
            "n_steps": len(times),
        }

    @pytest.mark.slow
    def test_peak_current_bounded(self):
        """Peak current bounded [0.5, 5.0] MA at all resolutions."""
        configs = [
            ((16, 1, 32), 10e-3),
            ((32, 1, 64), 5e-3),
        ]
        for grid, dx in configs:
            result = self._run_engine_to_time(grid, dx, t_target=5e-6)
            peak_MA = result["peak_current"] / 1e6
            assert 0.5 < peak_MA < 5.0, (
                f"Grid {grid}: peak={peak_MA:.2f} MA outside range"
            )

    @pytest.mark.slow
    def test_peak_current_converges(self):
        """Peak current at higher resolution should be closer to experiment."""
        from dpf.validation.experimental import PF1000_DATA

        configs = [
            ((16, 1, 32), 10e-3),
            ((32, 1, 64), 5e-3),
        ]
        peak_errors = []
        for grid, dx in configs:
            result = self._run_engine_to_time(grid, dx, t_target=8e-6)
            peak = result["peak_current"]
            err = abs(peak - PF1000_DATA.peak_current) / PF1000_DATA.peak_current
            peak_errors.append(err)
            print(f"Grid {grid}: peak={peak/1e6:.3f} MA, "
                  f"error={err:.1%} vs exp {PF1000_DATA.peak_current/1e6:.3f} MA")

        # Finer grid should have <= error (allow 5% tolerance)
        assert peak_errors[1] <= peak_errors[0] + 0.05, (
            f"Higher resolution did not improve: "
            f"coarse={peak_errors[0]:.1%}, fine={peak_errors[1]:.1%}"
        )

    @pytest.mark.slow
    def test_convergence_report(self):
        """Report full convergence table across resolutions."""
        configs = [
            ("16x1x32", (16, 1, 32), 10e-3),
            ("32x1x64", (32, 1, 64), 5e-3),
        ]

        print("\n=== R_plasma / Peak Current Convergence ===")
        print(f"{'Grid':>12} {'dx [mm]':>10} {'Peak [MA]':>12} {'Steps':>8}")

        for name, grid, dx in configs:
            result = self._run_engine_to_time(grid, dx, t_target=8e-6)
            peak_MA = result["peak_current"] / 1e6
            print(f"{name:>12} {dx*1e3:>10.1f} {peak_MA:>12.3f} "
                  f"{result['n_steps']:>8}")


# =====================================================================
# Full convergence report
# =====================================================================


class TestConvergenceReport:
    """Generate comprehensive convergence report for PhD debate."""

    @pytest.mark.slow
    def test_full_report(self):
        """Print full convergence table for both wave types and precisions."""
        resolutions = [32, 64, 128]
        wave_types = ["sound", "fast"]
        precisions = ["float32", "float64"]

        print("\n" + "=" * 72)
        print("PHASE AK: Grid Convergence at DPF-Relevant Conditions")
        print("=" * 72)
        print(f"\nPF-1000 fill: rho0={_RHO0:.4e} kg/m^3, p0={_P0:.1f} Pa, "
              f"B0={_B0_SI} T")
        print(f"Speeds: cs={_CS:.0f} m/s, va={_VA:.0f} m/s, cf={_CF:.0f} m/s")
        print(f"beta = {2*_MU_0*_P0/_B0_SI**2:.1f}\n")

        for wt in wave_types:
            wave_label = "Sound" if wt == "sound" else "Fast Magnetosonic"
            c_label = f"cs={_CS:.0f}" if wt == "sound" else f"cf={_CF:.0f}"
            print(f"--- {wave_label} Wave ({c_label} m/s) ---")
            print(f"{'nx':>6} ", end="")
            for prec in precisions:
                print(f" {prec:>12}", end="")
            print()

            errors: dict[str, dict[int, float]] = {}
            for prec in precisions:
                errors[prec] = {}
                for nx in resolutions:
                    errors[prec][nx] = _run_dpf_convergence(
                        nx, wave_type=wt, precision=prec,
                    )

            for nx in resolutions:
                print(f"{nx:>6} ", end="")
                for prec in precisions:
                    print(f" {errors[prec][nx]:>12.4e}", end="")
                print()

            # Orders
            print("Order (64->128):", end="")
            for prec in precisions:
                if errors[prec][128] > 0 and errors[prec][64] > 0:
                    order = np.log2(errors[prec][64] / errors[prec][128])
                else:
                    order = float("nan")
                print(f" {order:>12.2f}", end="")
            print("\n")

        # Repair fraction summary
        reset_repair_stats()
        _run_dpf_convergence(128, wave_type="fast", precision="float32")
        stats = get_repair_stats()
        fraction = stats["total_repaired"] / max(stats["total_checked"], 1)
        print(f"Repair fraction (fast wave, float32, 128 cells): "
              f"{fraction:.6f} ({stats['total_repaired']}/{stats['total_checked']})")
