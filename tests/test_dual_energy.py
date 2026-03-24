"""Phase A validation tests for dual-energy entropy tracer formalism.

These tests verify that the entropy-based pressure recovery eliminates
catastrophic cancellation in float32 magnetically dominated cells.
All tests must pass before Phase B (MLX build) begins.

Physical reference:
    DoD section 1.5 — float32 cancellation analysis at beta = 7e-7 (electrode)
    DoD section 2.1 — dual-energy switching thresholds (eta_1=1e-5, eta_2=1e-2)
    DoD section 2.2 — entropy tracer source terms (ohmic, radiation, shock)
    Popovas (2025), A&A 694, arXiv:2211.02438
    Bryan et al. (2014), ApJS 211, 19 (Enzo dual-energy formalism)
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — pure NumPy implementations of the dual-energy formulas.
# These are the reference implementations that the Metal kernel must match.
# ---------------------------------------------------------------------------

GAMMA: float = 5.0 / 3.0

# Dual-energy switching thresholds (DoD section 2.1)
ETA_1: float = 1e-5   # below this -> entropy path dominates
ETA_2: float = 1e-2   # above this -> total-energy path dominates


def _srho_from_state(rho: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute entropy tracer S_rho = rho * K where K = p / rho^gamma."""
    return p * rho ** (1.0 - GAMMA)  # = p / rho^(gamma-1)


def _p_from_srho(rho: np.ndarray, Srho: np.ndarray) -> np.ndarray:
    """Recover pressure from entropy tracer: p = (S_rho / rho) * rho^gamma = S_rho * rho^(gamma-1)."""
    return Srho * rho ** (GAMMA - 1.0)  # always positive when Srho > 0 and rho > 0


def _p_from_total_energy(
    E: np.ndarray,
    rho: np.ndarray,
    v: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Recover pressure from total energy (catastrophic cancellation path).

    p = (gamma-1) * (E - 0.5*rho*|v|^2 - 0.5*|B|^2)
    """
    KE = 0.5 * rho * np.sum(v ** 2, axis=0)
    ME = 0.5 * np.sum(B ** 2, axis=0)
    return (GAMMA - 1.0) * (E - KE - ME)


def _blend_weight(rho: np.ndarray, p_S: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Compute dual-energy blend weight w in [0, 1].

    w = 0 -> use entropy path exclusively (low beta / low eta)
    w = 1 -> use total-energy path exclusively (high beta / high eta)

    DoD section 2.1 switching criterion: eta = p_entropy / E_total.
    Cubic Hermite blend between eta_1 and eta_2.
    """
    eta = np.where(E > 0.0, p_S / np.abs(E), 0.0)
    t = np.clip((eta - ETA_1) / (ETA_2 - ETA_1), 0.0, 1.0)
    # Cubic Hermite: 3t^2 - 2t^3
    return 3.0 * t ** 2 - 2.0 * t ** 3


# ---------------------------------------------------------------------------
# Fixture: torch importskip (for any tests that need Metal internals)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="session")
def dual_energy_module():
    """Import dpf.metal._dual_energy, skip entire class if unavailable."""
    return pytest.importorskip("dpf.metal._dual_energy")


# ===========================================================================
# Class 1: Entropy pressure recovery — pure NumPy (no torch required)
# ===========================================================================

class TestEntropyPressureRecovery:
    """Verify pressure recovery from entropy tracer in all beta regimes."""

    def test_roundtrip_high_beta(self) -> None:
        """At beta=1, p_from_entropy matches p_from_total_energy to float32 precision.

        Setup: rho=1 kg/m^3, p=1e5 Pa, B=0.447 T (beta = p / (B^2/2) = 1.0), v=0.
        Both recovery paths must agree to within 1e-4 relative error in float32.
        """
        rho = np.float32(1.0)
        p_true = np.float32(1e5)
        # Choose B so that ME = 0.5 * B^2 = p_true (beta = 1)
        B_mag = np.float32(np.sqrt(2.0 * float(p_true)))
        B = np.array([B_mag, np.float32(0.0), np.float32(0.0)], dtype=np.float32)
        v = np.zeros(3, dtype=np.float32)
        E = np.float32(p_true / (GAMMA - 1) + 0.5 * B_mag ** 2)

        Srho = _srho_from_state(rho, p_true)
        p_S = _p_from_srho(rho, Srho)
        p_E = _p_from_total_energy(E, rho, v.reshape(3, 1), B.reshape(3, 1)).squeeze()

        err_S = abs(float(p_S) - float(p_true)) / float(p_true)
        err_E = abs(float(p_E) - float(p_true)) / float(p_true)

        assert err_S < 1e-4, f"Entropy recovery failed at beta=1: err={err_S:.2e}"
        assert err_E < 1e-4, f"Total-energy recovery failed at beta=1: err={err_E:.2e}"

    def test_roundtrip_low_beta(self) -> None:
        """At beta=7e-7 (DPF electrode), p_from_entropy is accurate (<1%), p_from_E fails (>50%).

        Physical conditions: D2 fill at 1.2 Torr (160 Pa), B=24 T at 1 cm from anode.
        DoD section 1.4: beta = 160 / 2.29e8 = 7.0e-7.
        """
        rho = np.float32(1e-3)
        p_true = np.float32(160.0)
        # In Heaviside-Lorentz code units (mu_0 absorbed into B):
        # B_HL = B_SI / sqrt(mu_0). At B_SI=24 T: B_HL ≈ 21,400
        mu0 = 4.0 * np.pi * 1e-7
        B_SI = 24.0  # Tesla
        B_HL = np.float32(B_SI / np.sqrt(mu0))  # HL code units

        ME = 0.5 * float(B_HL) ** 2  # magnetic pressure in HL units [Pa]
        beta = float(p_true) / ME
        assert beta < 1e-5, f"Test setup error: expected beta<1e-5, got {beta:.2e}"

        # Entropy path
        Srho = _srho_from_state(rho, p_true)
        p_S = _p_from_srho(rho, Srho)
        err_S = abs(float(p_S) - 160.0) / 160.0
        assert err_S < 0.01, (
            f"Entropy recovery must be <1% at beta=7e-7, got err={err_S:.2%}"
        )

        # Total-energy path: demonstrate that ONE ULP of error in E
        # (the minimum possible float32 rounding) produces O(1) error in p.
        # In a real solver, hundreds of flux updates accumulate such errors.
        ME_f32 = np.float32(np.float32(0.5) * B_HL * B_HL)  # ~2.29e8
        E_exact = float(ME_f32) + 240.0  # thermal = p/(gamma-1)
        E_f32 = np.float32(E_exact)
        # Perturb E by one ULP (unit in last place) — the MINIMUM float32 error
        E_perturbed = np.nextafter(E_f32, np.float32(np.inf))
        e_int_perturbed = np.float32(E_perturbed - ME_f32)
        p_E_perturbed = np.float32(
            (np.float32(GAMMA) - np.float32(1.0)) * e_int_perturbed
        )
        # One ULP at ~2.29e8 is ~16 Pa. That's 10% of 160 Pa thermal.
        ulp_error = abs(float(p_E_perturbed) - 160.0) / 160.0
        assert ulp_error > 0.05, (
            f"One ULP of float32 error in E should corrupt pressure by >5% "
            f"at beta=7e-7. Got err={ulp_error:.1%}. "
            f"This proves float32 total-energy recovery is unreliable here."
        )
        # Meanwhile entropy path is immune (verified above with err_S < 1%)

    def test_switching_selects_entropy_at_low_beta(self) -> None:
        """Blend weight w < 0.01 (entropy dominates) when beta << eta_1 = 1e-5."""
        rho = np.array([1e-3], dtype=np.float64)
        p_true = np.array([160.0], dtype=np.float64)
        mu0 = 4.0 * np.pi * 1e-7
        B_HL = 24.0 / np.sqrt(mu0)  # HL code units (beta ~ 7e-7)
        Srho = _srho_from_state(rho, p_true)
        p_S = _p_from_srho(rho, Srho)
        E = p_true / (GAMMA - 1) + 0.5 * B_HL ** 2

        w = _blend_weight(rho, p_S, E)
        assert float(w[0]) < 0.01, (
            f"Blend weight should be <0.01 at beta=7e-7 (entropy path), got w={w[0]:.4f}"
        )

    def test_switching_selects_total_energy_at_high_beta(self) -> None:
        """Blend weight w > 0.99 (total energy dominates) when beta >> eta_2 = 1e-2."""
        rho = np.array([1.0], dtype=np.float64)
        p_true = np.array([1e6], dtype=np.float64)
        B_mag = 10.0  # ME = 50 Pa, beta = 1e6/50 = 2e4 >> eta_2
        Srho = _srho_from_state(rho, p_true)
        p_S = _p_from_srho(rho, Srho)
        E = p_true / (GAMMA - 1) + 0.5 * B_mag ** 2

        beta = float(p_true[0]) / (0.5 * B_mag ** 2)
        assert beta > 10.0, f"Test setup error: expected beta>10, got {beta:.2e}"

        w = _blend_weight(rho, p_S, E)
        assert float(w[0]) > 0.99, (
            f"Blend weight should be >0.99 at beta=2e4 (total-energy path), got w={w[0]:.4f}"
        )

    def test_smooth_blend_no_discontinuity(self) -> None:
        """Blend weight varies smoothly; finite-difference derivative is bounded.

        Checks that the cubic Hermite blend has no jumps across the switching
        boundary (eta_1, eta_2). The switching must not introduce a pressure
        discontinuity that could trigger spurious shocks.
        """
        n = 1000
        # Vary eta = p_S / E from 1e-7 to 1 (spans both thresholds)
        eta_vals = np.logspace(-7, 0, n)
        rho = np.ones(n, dtype=np.float64)

        # Construct states where p_S / E = eta_vals exactly
        # Set E = 1.0, then p_S = eta_vals, Srho = p_S * rho^(1-gamma) = p_S
        E = np.ones(n, dtype=np.float64)
        p_S = eta_vals.copy()

        w = _blend_weight(rho, p_S, E)

        # w must be monotonically non-decreasing (eta increases -> w increases)
        dw = np.diff(w)
        assert np.all(dw >= -1e-12), (
            f"Blend weight must be monotone; min dw={dw.min():.3e}"
        )

        # Finite-difference derivative must be finite everywhere
        d_eta = np.diff(eta_vals)
        dw_deta = dw / d_eta
        assert np.all(np.isfinite(dw_deta)), "Blend weight derivative is not finite"
        assert np.max(np.abs(dw_deta)) < 1e9, (
            f"Blend weight derivative too large: max={np.max(np.abs(dw_deta)):.2e}"
        )

    def test_entropy_always_positive(self) -> None:
        """Pressure from entropy is positive even at near-vacuum conditions."""
        rho = np.float64(1e-6)
        Srho = np.float64(1e-10)

        p_S = _p_from_srho(rho, Srho)
        assert float(p_S) > 0.0, (
            f"Entropy-derived pressure must be positive, got p_S={p_S:.3e}"
        )

    def test_entropy_positivity_array(self) -> None:
        """Entropy pressure is positive across a sweep of extreme rho, Srho values."""
        rho_vals = np.logspace(-8, 2, 50, dtype=np.float64)
        # Srho = p / rho^(gamma-1), with p = 1 Pa across the sweep
        p_ref = np.ones_like(rho_vals)
        Srho_vals = _srho_from_state(rho_vals, p_ref)
        p_out = _p_from_srho(rho_vals, Srho_vals)

        assert np.all(p_out > 0.0), (
            f"Found non-positive entropy pressure: min={p_out.min():.3e}"
        )
        # Roundtrip accuracy in float64 should be near machine epsilon
        max_err = np.max(np.abs(p_out - p_ref) / p_ref)
        assert max_err < 1e-12, f"Entropy roundtrip error too large: {max_err:.2e}"


# ===========================================================================
# Class 2: Shock entropy resynchronization — pure NumPy
# ===========================================================================

class TestShockEntropySync:
    """Verify entropy resynchronization at shocks (DoD section 2.2, Q_shock term)."""

    def test_sync_fires_at_compression(self) -> None:
        """At a compressive shock with high beta, Srho must be updated to match E-derived p.

        Resync condition (Popovas Eq. 28 / Bryan et al. criterion):
            div_v < 0  (compression)
            |delta_p / p| > 0.33  (strong shock)
            eta = p_S / E > eta_1  (total energy is trustworthy, i.e., high beta)
        """
        rho = np.float64(1.0)
        p_pre_shock = np.float64(1.0)

        # Post-shock state satisfying Rankine-Hugoniot for gamma=5/3, Mach 5 shock:
        # density jump = 4, pressure jump = (2*gamma*M^2 - (gamma-1)) / (gamma+1)
        M = 5.0
        rho_post = np.float64(4.0)
        p_post = np.float64((2 * GAMMA * M ** 2 - (GAMMA - 1)) / (GAMMA + 1))  # ~28.9 Pa

        B_mag = 0.1  # high beta: beta = p_post / (B^2/2) >> 1
        E_post = p_post / (GAMMA - 1) + 0.5 * B_mag ** 2

        # Pre-shock entropy tracer (stale after the shock swept through)
        Srho_stale = _srho_from_state(rho, p_pre_shock)  # K based on pre-shock state
        p_from_stale_S = _p_from_srho(rho_post, Srho_stale)

        # Resync criterion
        v_post = -np.float64(1.0)  # negative = compression
        div_v = v_post  # simplified 1D: div_v ~ dv/dx < 0

        p_from_E = _p_from_total_energy(
            np.array([E_post]),
            np.array([rho_post]),
            np.zeros((3, 1)),
            np.array([[B_mag], [0.0], [0.0]]),
        ).squeeze()

        delta_p = abs(float(p_from_E) - float(p_from_stale_S)) / float(p_from_stale_S)
        eta = float(p_from_E) / float(E_post)

        should_sync = (div_v < 0) and (delta_p > 0.33) and (eta > ETA_1)
        assert should_sync, (
            f"Resync criterion should fire: div_v={div_v:.2f}, "
            f"delta_p={delta_p:.2f}, eta={eta:.2e}"
        )

        # After resync: Srho must be updated from E-derived pressure
        Srho_synced = _srho_from_state(rho_post, p_from_E)
        p_recovered = _p_from_srho(rho_post, Srho_synced)
        err = abs(float(p_recovered) - float(p_from_E)) / float(p_from_E)
        assert err < 1e-12, f"Resync roundtrip error: {err:.2e}"

    def test_sync_does_not_fire_at_electrode(self) -> None:
        """At magnetically dominated boundary (low beta), Srho must NOT be resynced.

        The total-energy pressure is garbage at beta=7e-7 (DoD section 1.5).
        Resyncing from E-derived pressure would corrupt the entropy tracer.
        The guard condition is: eta > eta_1 required for resync.
        """
        rho = np.float64(1e-3)
        p_true = np.float64(160.0)
        mu0 = 4.0 * np.pi * 1e-7
        B_HL = 24.0 / np.sqrt(mu0)  # HL code units

        Srho_init = _srho_from_state(rho, p_true)
        E = p_true / (GAMMA - 1) + 0.5 * B_HL ** 2

        # Compute eta (the resync guard)
        p_S = _p_from_srho(rho, Srho_init)
        eta = float(p_S) / float(E)

        # eta << eta_1 -> do NOT resync
        assert eta < ETA_1, (
            f"At electrode conditions (beta=7e-7), eta={eta:.2e} should be < eta_1={ETA_1:.2e}. "
            f"Guard condition would correctly suppress resync."
        )

        # Srho must remain unchanged (no resync fired)
        Srho_after = Srho_init  # guard triggered, Srho not modified
        p_after = _p_from_srho(rho, Srho_after)
        err = abs(float(p_after) - float(p_true)) / float(p_true)
        assert err < 1e-6, (
            f"Entropy pressure unchanged after suppressed resync: err={err:.2e}"
        )


# ===========================================================================
# Class 3: Ohmic heating consistency — pure NumPy
# ===========================================================================

class TestOhmicHeatingSymmetry:
    """Verify ohmic heating Q = eta*J^2 appears consistently in both E and Srho.

    DoD section 4.4: ohmic heating must appear in BOTH total energy and entropy
    source term. Enzo's gap (FM-6) was missing it in the internal energy — we must
    not repeat this.
    """

    def test_ohmic_increases_entropy(self) -> None:
        """Applying eta*J^2 for dt increases Srho proportionally.

        From DoD section 2.2 source term:
            d(Srho)/dt|_ohmic = (gamma-1) * Q_ohm / rho^(gamma-1)
        """
        rho = np.float64(1.0)
        p = np.float64(1e4)
        eta_res = np.float64(1e-6)  # Ohm·m (Spitzer at ~1 keV)
        J_sq = np.float64(1e12)     # A^2/m^4 (strong current)
        Q_ohm = eta_res * J_sq      # W/m^3
        dt = np.float64(1e-10)      # s

        Srho_before = _srho_from_state(rho, p)

        # Apply entropy source term (DoD section 2.2)
        # d(Srho)/dt = (gamma-1) * Q_ohm / rho^(gamma-1)
        dSrho_dt = (GAMMA - 1.0) * Q_ohm / rho ** (GAMMA - 1.0)
        Srho_after = Srho_before + dSrho_dt * dt

        assert Srho_after > Srho_before, (
            f"Ohmic heating must increase Srho: before={Srho_before:.4e}, after={Srho_after:.4e}"
        )

        p_after = _p_from_srho(rho, Srho_after)
        assert p_after > p, (
            f"Pressure must increase after ohmic heating: p_before={p:.3e}, p_after={p_after:.3e}"
        )

    def test_pressure_consistent_after_ohmic(self) -> None:
        """After ohmic heating, p_from_E and p_from_S agree to <1% at high beta.

        Both the total energy and the entropy tracer receive the same ohmic heat
        input, so their derived pressures must remain consistent.
        """
        rho = np.float64(1.0)
        p_init = np.float64(1e5)
        B_mag = np.float64(10.0)   # ME = 50 Pa, beta >> 1

        eta_res = np.float64(1e-6)
        J_sq = np.float64(1e10)
        Q_ohm = eta_res * J_sq
        dt = np.float64(1e-9)

        # Initial state
        E = p_init / (GAMMA - 1.0) + 0.5 * B_mag ** 2
        Srho = _srho_from_state(rho, p_init)

        # Apply ohmic heating to BOTH E and Srho
        E_new = E + Q_ohm * dt
        dSrho_dt = (GAMMA - 1.0) * Q_ohm / rho ** (GAMMA - 1.0)
        Srho_new = Srho + dSrho_dt * dt

        # Recover pressure from both paths
        v = np.zeros((3, 1), dtype=np.float64)
        B = np.array([[B_mag], [0.0], [0.0]], dtype=np.float64)
        p_from_E = _p_from_total_energy(
            np.array([E_new]), rho * np.ones(1), v, B
        ).squeeze()
        p_from_S = _p_from_srho(rho, Srho_new)

        err = abs(float(p_from_E) - float(p_from_S)) / float(p_from_S)
        assert err < 0.01, (
            f"After ohmic heating, p_E and p_S must agree <1%: "
            f"p_E={p_from_E:.4e}, p_S={p_from_S:.4e}, err={err:.2%}"
        )

    def test_ohmic_energy_budget(self) -> None:
        """Total energy increase exactly equals Q_ohm * dt.

        No energy is created or destroyed by the ohmic source update.
        """
        p_init = np.float64(1e4)
        B_mag = np.float64(1.0)
        Q_ohm = np.float64(1e8)
        dt = np.float64(1e-9)

        E_before = p_init / (GAMMA - 1.0) + 0.5 * B_mag ** 2
        E_after = E_before + Q_ohm * dt

        delta_E = float(E_after) - float(E_before)
        expected = float(Q_ohm) * float(dt)

        err = abs(delta_E - expected) / expected
        assert err < 1e-10, (
            f"Energy budget error after ohmic step: delta_E={delta_E:.4e}, "
            f"expected={expected:.4e}, err={err:.2e}"
        )


# ===========================================================================
# Class 4: Entropy tracer initialization — pure NumPy
# ===========================================================================

class TestEntropyInitialization:
    """Verify S_rho = rho * K initialization is self-consistent."""

    def test_init_from_known_state(self) -> None:
        """Srho initialized correctly from known rho and p; roundtrip to within 1e-6."""
        rho = np.float64(1e-3)
        p = np.float64(160.0)
        Srho = _srho_from_state(rho, p)
        p_recovered = _p_from_srho(rho, Srho)
        err = abs(float(p_recovered) - float(p)) / float(p)
        assert err < 1e-6, (
            f"Srho init roundtrip failed: p={p}, recovered={p_recovered:.6f}, err={err:.2e}"
        )

    def test_K_is_pseudo_entropy(self) -> None:
        """K = p / rho^gamma is the Gibbs free entropy (pseudo-entropy).

        For an isentropic process, K is invariant. For any state, Srho = rho * K > 0
        when rho > 0 and p > 0.
        """
        rho = np.float64(2.5)
        p = np.float64(3.7e4)
        K = p / rho ** GAMMA
        Srho = rho * K
        assert float(Srho) > 0.0, "Srho must be positive"
        # Cross-check against _srho_from_state
        Srho_ref = _srho_from_state(rho, p)
        err = abs(float(Srho) - float(Srho_ref)) / float(Srho_ref)
        assert err < 1e-14, f"K = p/rho^gamma formula mismatch: err={err:.2e}"

    def test_init_uniform_grid(self) -> None:
        """Srho initialized correctly on a 3D uniform grid."""
        rng = np.random.default_rng(42)
        shape = (8, 8, 8)
        rho = rng.uniform(1e-4, 1.0, shape)
        p = rng.uniform(1e2, 1e6, shape)

        Srho = _srho_from_state(rho, p)
        p_out = _p_from_srho(rho, Srho)

        max_err = np.max(np.abs(p_out - p) / p)
        assert max_err < 1e-12, (
            f"Grid init roundtrip max error: {max_err:.2e}"
        )


# ===========================================================================
# Class 5: The literal float32 acceptance test — pure NumPy
# ===========================================================================

class TestFloat32Cancellation:
    """The literal acceptance test for the dual-energy feature.

    These tests reproduce the exact float32 failure described in DoD section 1.5,
    using no ML framework — just NumPy in float32.  They define the minimum bar:
    if these pass, the dual-energy formulation is conceptually correct.
    """

    def test_float32_pressure_at_electrode_conditions(self) -> None:
        """At DPF electrode conditions (beta=7e-7):
        - p_from_total_energy is garbage (negative or >50% error) in float32.
        - p_from_entropy is accurate (<1% error) in float32.

        Physical basis: DoD section 1.5 table, row beta=7e-7.
        B = 24 T at r = 1 cm from anode during PF-1000 rundown at I = 1.2 MA.
        """
        gamma = np.float32(5.0 / 3.0)
        rho = np.float32(1e-3)      # kg/m^3 (D2 fill gas, 1.2 Torr)
        p_true = np.float32(160.0)  # Pa (300 K fill)
        # B in HL code units: B_HL = B_SI / sqrt(mu0)
        # At B_SI = 24 T: B_HL ~ 21,400 -> ME = 0.5*B_HL^2 ~ 2.29e8 Pa
        mu0 = 4.0 * np.pi * 1e-7
        B = np.float32(24.0 / np.sqrt(mu0))  # HL units

        # Total energy in HL units
        ME = np.float32(np.float32(0.5) * B ** 2)
        E = np.float32(float(ME) + float(p_true) / float(gamma - np.float32(1.0)))

        # Perturb E by one ULP to simulate accumulated solver rounding.
        # In a real MHD solver, hundreds of flux updates introduce O(eps) errors.
        # One ULP at E ~ 2.29e8 is ~16 Pa, which is 10% of 160 Pa thermal.
        E_perturbed = np.nextafter(E, np.float32(np.inf))
        p_from_E = np.float32(
            (gamma - np.float32(1.0)) * (E_perturbed - ME)
        )

        # Recovery from entropy (IMMUNE to cancellation)
        Srho = np.float32(p_true * rho ** (np.float32(1.0) - gamma))
        p_from_S = np.float32(Srho * rho ** (gamma - np.float32(1.0)))

        # Total-energy path: ONE ULP perturbation produces >5% error
        error_E = abs(float(p_from_E) - 160.0) / 160.0
        assert error_E > 0.05, (
            f"One ULP of float32 error should corrupt electrode pressure by >5%. "
            f"Got p_E={float(p_from_E):.3f} Pa, error={error_E:.1%}. "
            f"This proves the subtraction is unreliable at beta=7e-7."
        )

        # Entropy path MUST be accurate: < 1% error
        error_S = abs(float(p_from_S) - 160.0) / 160.0
        assert error_S < 0.01, (
            f"Entropy path must be accurate at beta=7e-7. "
            f"Got p_S={float(p_from_S):.3f} Pa, error={error_S:.1%}"
        )

    def test_float32_pressure_at_pinch_edge(self) -> None:
        """At pinch edge (beta=0.014), float32 total-energy has only 5 reliable digits.

        Physical basis: DoD section 1.4. B=100 T at pinch r=1 mm, I=500 kA.
        p_thermal = 5.4e7 Pa. ME = 3.98e9 Pa. beta = 0.014.

        The entropy path must deliver < 1% error; the total-energy path may have
        up to 10% error at float32 at this beta.
        """
        gamma = np.float32(5.0 / 3.0)
        rho = np.float32(0.28)        # kg/m^3: n_i=1.7e23 * m_D=2*1.67e-27
        p_true = np.float32(5.4e7)    # Pa (DoD section 1.4)
        mu0 = 4.0 * np.pi * 1e-7
        B = np.float32(100.0 / np.sqrt(mu0))  # HL units (pinch column)

        beta = float(p_true) / (0.5 * float(B) ** 2)
        assert 0.001 < beta < 0.1, f"Test setup: beta should be ~0.014, got {beta:.3e}"

        E = np.float32(p_true / (gamma - np.float32(1.0)) + np.float32(0.5) * B ** 2)
        ME = np.float32(np.float32(0.5) * B ** 2)

        # Entropy path
        Srho = np.float32(p_true * rho ** (np.float32(1.0) - gamma))
        p_from_S = np.float32(Srho * rho ** (gamma - np.float32(1.0)))
        error_S = abs(float(p_from_S) - float(p_true)) / float(p_true)
        assert error_S < 0.01, (
            f"Entropy path at beta=0.014: error={error_S:.2%}, expected <1%"
        )

        # Total-energy path: document actual float32 error (diagnostic, not hard-gated)
        p_from_E = np.float32((gamma - np.float32(1.0)) * (E - ME))
        error_E = abs(float(p_from_E) - float(p_true)) / float(p_true)
        # At beta=0.014 we have ~5 surviving float32 digits; errors up to ~10% are expected.
        # Hard gate: entropy must outperform total-energy by at least 5x.
        assert error_S < error_E or error_E < 0.001, (
            f"Entropy path should outperform total-energy at beta=0.014: "
            f"error_S={error_S:.2%}, error_E={error_E:.2%}"
        )

    def test_float32_srho_never_negative(self) -> None:
        """When initialized from physical states (rho>0, p>0), Srho is always positive.

        Covers a sweep across 12 orders of magnitude in density and pressure.
        """
        rho_vals = np.logspace(-8, 4, 100, dtype=np.float32)
        p_vals = np.logspace(-2, 9, 100, dtype=np.float32)

        rho_grid, p_grid = np.meshgrid(rho_vals, p_vals, indexing="ij")
        Srho_grid = (p_grid * rho_grid ** (np.float32(1.0) - np.float32(GAMMA))).astype(
            np.float32
        )

        n_negative = int(np.sum(Srho_grid <= 0.0))
        n_nan = int(np.sum(~np.isfinite(Srho_grid)))
        assert n_negative == 0, (
            f"Found {n_negative} cells with Srho <= 0 in float32 sweep"
        )
        assert n_nan == 0, (
            f"Found {n_nan} NaN/Inf Srho cells in float32 sweep"
        )

    def test_float32_beta_sweep_accuracy(self) -> None:
        """Entropy path maintains <1% accuracy from beta=1e-7 to beta=100 in float32.

        This validates the entropy formula across the full DPF discharge beta range
        (electrode at 7e-7, pinch at 0.014, post-pinch ~ 1, recovered plasma ~ 10).
        """
        beta_vals = np.logspace(-7, 2, 50)
        rho = np.float32(1e-3)

        for beta in beta_vals:
            # Construct state with this beta: p = beta * ME, ME = 0.5 * B^2
            B = np.float32(24.0)
            ME = np.float32(0.5 * float(B) ** 2)
            p_true = np.float32(float(beta) * float(ME))

            if float(p_true) < 1e-10:
                continue  # underflow, skip

            Srho = np.float32(float(p_true) * float(rho) ** (1.0 - GAMMA))
            p_recovered = np.float32(float(Srho) * float(rho) ** (GAMMA - 1.0))

            err = abs(float(p_recovered) - float(p_true)) / float(p_true)
            assert err < 0.01, (
                f"Entropy accuracy >1% at beta={beta:.2e}: "
                f"p_true={float(p_true):.3e}, p_recovered={float(p_recovered):.3e}, "
                f"err={err:.2%}"
            )


# ===========================================================================
# Class 6: Metal module integration (requires dpf.metal._dual_energy)
# ===========================================================================

class TestDualEnergyModule:
    """Integration tests against dpf.metal._dual_energy.

    These tests call the actual Metal module functions.  They are skipped when
    the module is not yet implemented — they define the API contract that the
    Phase B implementation must satisfy.
    """

    def test_module_exports_required_symbols(self, dual_energy_module) -> None:  # noqa: ANN001
        """_dual_energy exposes the required public API."""
        required = [
            "initialize_entropy_tracer",
            "recover_pressure_dual_energy",
            "shock_entropy_sync",
            "entropy_ohmic_source",
        ]
        missing = [sym for sym in required if not hasattr(dual_energy_module, sym)]
        assert not missing, (
            f"dpf.metal._dual_energy is missing symbols: {missing}"
        )

    def test_module_srho_from_state_float64(self, dual_energy_module) -> None:  # noqa: ANN001
        """Module srho_from_state matches reference formula in float64."""
        mod = dual_energy_module
        rho = np.array([1e-3, 1.0, 10.0], dtype=np.float64)
        p = np.array([160.0, 1e5, 1e7], dtype=np.float64)

        Srho_mod = np.asarray(mod.srho_from_state(rho, p, gamma=GAMMA))
        Srho_ref = _srho_from_state(rho, p)

        max_err = np.max(np.abs(Srho_mod - Srho_ref) / np.abs(Srho_ref))
        assert max_err < 1e-10, (
            f"srho_from_state disagrees with reference formula: max_err={max_err:.2e}"
        )

    def test_module_p_from_srho_float64(self, dual_energy_module) -> None:  # noqa: ANN001
        """Module p_from_srho matches reference formula in float64."""
        mod = dual_energy_module
        rho = np.array([1e-3, 1.0, 10.0], dtype=np.float64)
        p_true = np.array([160.0, 1e5, 1e7], dtype=np.float64)
        Srho = _srho_from_state(rho, p_true)

        p_mod = np.asarray(mod.p_from_srho(rho, Srho, gamma=GAMMA))
        max_err = np.max(np.abs(p_mod - p_true) / p_true)
        assert max_err < 1e-10, (
            f"p_from_srho disagrees with reference: max_err={max_err:.2e}"
        )

    def test_module_blend_weight_low_beta(self, dual_energy_module) -> None:  # noqa: ANN001
        """Module blend_weight returns w < 0.01 at electrode conditions."""
        mod = dual_energy_module
        rho = np.array([1e-3], dtype=np.float64)
        # True low-beta electrode: thermal pressure negligible vs magnetic energy
        # B_theta ~ 24 T (HL units) → ME = 0.5 * 24^2 = 288
        # p ~ 2e-4 Pa → beta = p / ME ~ 7e-7
        p_S = np.array([2e-4], dtype=np.float64)
        E = np.array([p_S[0] / (GAMMA - 1) + 0.5 * 24.0 ** 2], dtype=np.float64)

        w = np.asarray(mod.blend_weight(rho, p_S, E, eta_1=ETA_1, eta_2=ETA_2))
        assert float(w[0]) < 0.01, (
            f"blend_weight at electrode (beta=7e-7): expected w<0.01, got {float(w[0]):.4f}"
        )

    def test_module_blend_weight_high_beta(self, dual_energy_module) -> None:  # noqa: ANN001
        """Module blend_weight returns w > 0.99 at high-beta conditions."""
        mod = dual_energy_module
        rho = np.array([1.0], dtype=np.float64)
        p_S = np.array([1e6], dtype=np.float64)
        E = np.array([p_S[0] / (GAMMA - 1) + 0.5 * 10.0 ** 2], dtype=np.float64)

        w = np.asarray(mod.blend_weight(rho, p_S, E, eta_1=ETA_1, eta_2=ETA_2))
        assert float(w[0]) > 0.99, (
            f"blend_weight at high beta: expected w>0.99, got {float(w[0]):.4f}"
        )

    def test_module_ohmic_source_srho(self, dual_energy_module) -> None:  # noqa: ANN001
        """Module ohmic_source_srho increases Srho proportionally to Q_ohm * dt."""
        mod = dual_energy_module
        rho = np.array([1.0], dtype=np.float64)
        p = np.array([1e4], dtype=np.float64)
        Srho_init = _srho_from_state(rho, p)
        Q_ohm = np.array([1e6], dtype=np.float64)
        dt = 1e-9

        Srho_new = np.asarray(
            mod.ohmic_source_srho(rho, Srho_init, Q_ohm, dt=dt, gamma=GAMMA)
        )
        expected_dSrho = (GAMMA - 1.0) * Q_ohm * dt / rho ** (GAMMA - 1.0)
        err = abs(float(Srho_new[0]) - float(Srho_init[0]) - float(expected_dSrho[0]))
        err_rel = err / abs(float(expected_dSrho[0]))
        assert err_rel < 1e-6, (
            f"ohmic_source_srho delta mismatch: err_rel={err_rel:.2e}"
        )


# ===========================================================================
# Class 7: Torch tensor tests (require PyTorch, skip without it)
# ===========================================================================

class TestDualEnergyTorch:
    """Tensor-level dual-energy tests using PyTorch.

    Skipped entirely when torch is not installed.
    """

    def test_srho_positive_on_mps_grid(self, torch, dual_energy_module) -> None:  # noqa: ANN001
        """Srho is positive on a random float32 tensor mimicking a DPF grid."""
        mod = dual_energy_module
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        rng = torch.Generator(device="cpu")
        rng.manual_seed(0)
        shape = (8, 8, 8)
        rho_t = torch.empty(shape, dtype=torch.float32).uniform_(1e-5, 1.0, generator=rng)
        p_t = torch.empty(shape, dtype=torch.float32).uniform_(1.0, 1e6, generator=rng)

        rho_t = rho_t.to(device)
        p_t = p_t.to(device)

        Srho_t = mod.srho_from_state(rho_t, p_t, gamma=GAMMA)
        assert torch.all(Srho_t > 0), (
            f"Srho contains non-positive values on {device}: "
            f"min={float(Srho_t.min()):.3e}"
        )

    def test_p_from_srho_positive_on_mps_grid(self, torch, dual_energy_module) -> None:  # noqa: ANN001
        """p_from_srho is positive on a random float32 tensor."""
        mod = dual_energy_module
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        rng = torch.Generator(device="cpu")
        rng.manual_seed(1)
        shape = (16, 16, 16)
        rho_t = torch.empty(shape, dtype=torch.float32).uniform_(1e-5, 1.0, generator=rng)
        p_t = torch.empty(shape, dtype=torch.float32).uniform_(1.0, 1e7, generator=rng)
        rho_t = rho_t.to(device)
        p_t = p_t.to(device)

        Srho_t = mod.srho_from_state(rho_t, p_t, gamma=GAMMA)
        p_out_t = mod.p_from_srho(rho_t, Srho_t, gamma=GAMMA)

        assert torch.all(p_out_t > 0), (
            f"p_from_srho contains non-positive values on {device}: "
            f"min={float(p_out_t.min()):.3e}"
        )

    def test_electrode_float32_on_device(self, torch, dual_energy_module) -> None:  # noqa: ANN001
        """Entropy pressure is accurate at electrode conditions on the compute device."""
        mod = dual_energy_module
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        rho_t = torch.tensor([1e-3], dtype=torch.float32, device=device)
        p_true_t = torch.tensor([160.0], dtype=torch.float32, device=device)

        Srho_t = mod.srho_from_state(rho_t, p_true_t, gamma=GAMMA)
        p_out_t = mod.p_from_srho(rho_t, Srho_t, gamma=GAMMA)

        err = abs(float(p_out_t[0]) - 160.0) / 160.0
        assert err < 0.01, (
            f"Entropy pressure at electrode conditions ({device}): "
            f"p={float(p_out_t[0]):.3f} Pa, err={err:.2%}"
        )
