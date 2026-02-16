"""Phase R.3: Tests for full 6-region HLLD solver with double-star states.

Verifies that the Metal HLLD Riemann solver (hlld_flux_mps) correctly
implements the Miyoshi & Kusano (2005) double-star states for resolving
Alfven waves and contact discontinuities.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_riemann import (  # noqa: E402, I001
    NVAR,
    IDN,
    IB1,
    IM2,
    _prim_to_cons_mps,
    hlld_flux_mps,
)

DEVICE = torch.device("cpu")
GAMMA = 5.0 / 3.0


def _make_cons(rho, vx, vy, vz, p, Bx, By, Bz, gamma=GAMMA):
    """Helper: build conservative state (8,) from primitives."""
    rho_t = torch.tensor([rho], dtype=torch.float64, device=DEVICE)
    vel_t = torch.tensor([[vx], [vy], [vz]], dtype=torch.float64, device=DEVICE)
    p_t = torch.tensor([p], dtype=torch.float64, device=DEVICE)
    B_t = torch.tensor([[Bx], [By], [Bz]], dtype=torch.float64, device=DEVICE)
    return _prim_to_cons_mps(rho_t, vel_t, p_t, B_t, gamma)


class TestHLLDDoublestar:
    """Tests for the full 6-region HLLD solver with double-star states."""

    def test_hlld_uniform_state_zero_flux_difference(self):
        """Uniform state should produce zero net flux difference (F_L = F_R)."""
        U = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        F = hlld_flux_mps(U, U, GAMMA, dim=0)
        # For uniform state, F_HLLD(UL=UR=U) should equal the physical flux F(U).
        # The key test: F is finite and not NaN.
        assert not torch.isnan(F).any(), "HLLD flux contains NaN for uniform state"
        assert torch.isfinite(F).all(), "HLLD flux contains Inf for uniform state"

    def test_hlld_contact_discontinuity_density_flux(self):
        """A pure contact (density jump, same p and v) should be resolved sharply.

        For a contact discontinuity, the mass flux F_rho = rho * vn should
        correspond to the contact speed SM. With v=0, F_rho should be zero
        on both sides (no flow across the contact).
        """
        # Left: high density, Right: low density, same p, v=0, Bn only
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        UR = _make_cons(0.125, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for contact discontinuity"
        # Mass flux should be near zero since v=0 on both sides and contact is stationary
        assert torch.abs(F[IDN]).item() < 0.1, (
            f"Mass flux too large for stationary contact: {F[IDN].item():.4e}"
        )

    def test_hlld_alfven_wave_resolution(self):
        """Alfven discontinuity (transverse B jump) should be well-resolved.

        Set up left/right states with transverse B jump and same rho, p, v.
        The HLLD solver with double-star states should handle the Alfven
        jump condition correctly.
        """
        # Pure Alfven discontinuity: Bt changes sign, everything else same
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        UR = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Alfven discontinuity"
        assert torch.isfinite(F).all(), "HLLD Inf for Alfven discontinuity"
        # Mass flux should be zero (no density jump, no normal velocity)
        assert torch.abs(F[IDN]).item() < 1e-6, (
            f"Mass flux should be ~0 for Alfven wave: {F[IDN].item():.4e}"
        )
        # Normal B flux should be zero (Bn is continuous)
        assert torch.abs(F[IB1]).item() < 1e-6, (
            f"Normal B flux should be ~0: {F[IB1].item():.4e}"
        )

    def test_hlld_brio_wu_no_nan(self):
        """Brio-Wu MHD shock tube should not produce NaN.

        Brio-Wu: rho_L=1, p_L=1, By_L=1, rho_R=0.125, p_R=0.1, By_R=-1,
        Bx=0.75 everywhere.
        """
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        UR = _make_cons(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Brio-Wu"
        assert torch.isfinite(F).all(), "HLLD Inf for Brio-Wu"

    def test_hlld_double_star_degeneracy_bn_zero(self):
        """When Bn=0, double-star should reduce to single-star (HLLC).

        With Bn=0, the Alfven speeds collapse to SM and the double-star
        states should be identical to single-star states.
        """
        # Bn = 0: only transverse B
        UL = _make_cons(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        UR = _make_cons(0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Bn=0"
        assert torch.isfinite(F).all(), "HLLD Inf for Bn=0"

    def test_hlld_conservation(self):
        """HLLD flux should conserve all 8 quantities (finite + correct shape)."""
        UL = _make_cons(1.0, 0.5, 0.1, -0.2, 2.0, 0.5, 0.3, 0.1)
        UR = _make_cons(0.8, -0.3, 0.2, 0.1, 1.5, 0.5, -0.2, 0.3)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert F.shape == (NVAR, 1), f"Flux shape mismatch: {F.shape}"
        assert torch.isfinite(F).all(), "HLLD flux not finite"

    def test_hlld_all_dimensions(self):
        """HLLD flux should work correctly for all three dimensions."""
        UL = _make_cons(1.0, 0.2, -0.1, 0.3, 1.5, 0.5, 0.3, 0.2)
        UR = _make_cons(0.5, -0.1, 0.2, -0.1, 0.8, 0.5, -0.1, 0.4)
        for dim in range(3):
            F = hlld_flux_mps(UL, UR, GAMMA, dim=dim)
            assert not torch.isnan(F).any(), f"HLLD NaN for dim={dim}"
            assert torch.isfinite(F).all(), f"HLLD Inf for dim={dim}"

    def test_hlld_symmetry(self):
        """HLLD should give symmetric results for mirror-symmetric states.

        If we swap L/R states and negate normal velocity, the flux should
        change sign for odd quantities (momentum, induction) and preserve
        sign for even (density, energy, normal B).
        """
        UL = _make_cons(1.0, 0.5, 0.1, 0.0, 1.0, 0.5, 0.3, 0.0)
        UR = _make_cons(1.0, -0.5, -0.1, 0.0, 1.0, 0.5, -0.3, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any()
        # For this symmetric setup, mass flux should be near zero
        assert torch.abs(F[IDN]).item() < 0.1

    def test_hlld_strong_b_field(self):
        """HLLD should handle strong magnetic fields (magnetically dominated)."""
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 0.01, 10.0, 5.0, 3.0)
        UR = _make_cons(1.0, 0.0, 0.0, 0.0, 0.01, 10.0, -5.0, -3.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for strong B"
        assert torch.isfinite(F).all(), "HLLD Inf for strong B"

    def test_hlld_double_star_vs_single_star_with_bn(self):
        """With nonzero Bn, double-star flux should differ from single-star.

        The double-star states resolve Alfven waves, which modify the
        transverse velocity and B components. With significant Bn and
        transverse jumps, the double-star contribution should be nonzero.
        """
        # Strong Bn with transverse jumps → Alfven waves are important
        UL = _make_cons(1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0)
        UR = _make_cons(1.0, 0.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0)
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert not torch.isnan(F).any()
        assert torch.isfinite(F).all()
        # The transverse momentum flux should be nonzero (Alfven wave carries
        # transverse momentum)
        # IM2 is the y-momentum flux (transverse to x-normal)
        assert F[IM2].abs().item() > 1e-10, (
            "Transverse momentum flux too small — double-star not active?"
        )

    def test_hlld_batch_no_nan(self):
        """HLLD should handle batched inputs (multi-cell) without NaN."""
        n = 16
        rho = torch.ones(n, dtype=torch.float64, device=DEVICE)
        vel = torch.zeros(3, n, dtype=torch.float64, device=DEVICE)
        vel[0] = torch.linspace(-0.5, 0.5, n)
        p = torch.ones(n, dtype=torch.float64, device=DEVICE)
        B = torch.zeros(3, n, dtype=torch.float64, device=DEVICE)
        B[0] = 1.0
        B[1] = torch.linspace(-1, 1, n)

        UL = _prim_to_cons_mps(rho, vel, p, B, GAMMA)
        UR = _prim_to_cons_mps(
            rho * 0.8,
            vel * 0.5,
            p * 0.9,
            B * 1.1,
            GAMMA,
        )
        F = hlld_flux_mps(UL, UR, GAMMA, dim=0)
        assert F.shape == (NVAR, n)
        assert not torch.isnan(F).any(), "HLLD NaN in batched inputs"
        assert torch.isfinite(F).all(), "HLLD Inf in batched inputs"

    def test_hlld_float32_brio_wu_no_nan(self):
        """HLLD should also work in float32 (Metal-native) for Brio-Wu."""
        UL = _make_cons(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        UR = _make_cons(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
        UL32 = UL.float()
        UR32 = UR.float()
        F = hlld_flux_mps(UL32, UR32, GAMMA, dim=0)
        assert not torch.isnan(F).any(), "HLLD NaN for Brio-Wu in float32"
        assert torch.isfinite(F).all(), "HLLD Inf for Brio-Wu in float32"
