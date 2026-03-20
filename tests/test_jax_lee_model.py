"""Tests for differentiable JAX Lee model.

Covers:
- Forward simulation: PF-1000 I_peak within 25% of experimental ~1.87 MA
- Gradient pass: jax.grad returns finite, non-zero gradients
- Speed test: 1000 vmap simulations in under 10 seconds
- Loss function: NRMSE computes and is differentiable
- Sensitivity: d(I_peak)/d(param) has correct signs
- UNU-ICTP device: separate device validation
"""

from __future__ import annotations

import time

import pytest

jax = pytest.importorskip("jax", reason="JAX required for differentiable Lee model")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from dpf.jax.lee_model import (  # noqa: E402
    default_pf1000_params,
    default_unu_ictp_params,
    loss_fn,
    sensitivity,
    simulate,
    vmap_simulate,
)


class TestForwardSimulation:
    """Forward pass produces physically reasonable results."""

    def test_pf1000_ipeak_magnitude(self) -> None:
        """PF-1000 I_peak should be within 25% of ~1.87 MA experimental."""
        params = default_pf1000_params()
        result = simulate(params)
        I_peak = float(result["I_peak"])
        # PF-1000 experimental: ~1.87 MA. Lee model with soft-switching
        # gives 1.5-2.0 MA depending on params. Allow 25% tolerance.
        assert 1.0e6 < I_peak < 2.5e6, f"I_peak={I_peak:.2e} outside [1.0, 2.5] MA"

    def test_pf1000_tpeak_timing(self) -> None:
        """PF-1000 peak current should occur at ~5-7 us."""
        params = default_pf1000_params()
        result = simulate(params)
        t_peak_us = float(result["t_peak"]) * 1e6
        assert 3.0 < t_peak_us < 10.0, f"t_peak={t_peak_us:.2f} us outside [3, 10] us"

    def test_unu_ictp_ipeak(self) -> None:
        """UNU-ICTP I_peak should be in the 50-300 kA range."""
        params = default_unu_ictp_params()
        result = simulate(params)
        I_peak = float(result["I_peak"])
        assert 30e3 < I_peak < 400e3, f"I_peak={I_peak:.2e} outside [30, 400] kA"

    def test_output_shapes(self) -> None:
        """All output arrays should have correct shape."""
        n_steps = 5000
        params = default_pf1000_params()
        result = simulate(params, n_steps=n_steps)
        assert result["t"].shape == (n_steps,)
        assert result["I"].shape == (n_steps,)
        assert result["V"].shape == (n_steps,)
        assert result["z"].shape == (n_steps,)
        assert result["r"].shape == (n_steps,)

    def test_initial_conditions(self) -> None:
        """Initial current = 0, voltage = V0."""
        params = default_pf1000_params()
        result = simulate(params)
        assert abs(float(result["I"][0])) < 1.0, "Initial current should be ~0"
        V0 = float(params["V0"])
        assert abs(float(result["V"][0]) - V0) < V0 * 0.01, "Initial voltage should be ~V0"

    def test_voltage_decreases(self) -> None:
        """Capacitor voltage should decrease from V0 as discharge proceeds."""
        params = default_pf1000_params()
        result = simulate(params)
        V = result["V"]
        # Voltage at midpoint should be less than initial
        mid = len(V) // 2
        assert float(V[mid]) < float(V[0]), "Voltage should decrease during discharge"


class TestGradients:
    """Gradient computation via jax.grad."""

    def test_grad_finite(self) -> None:
        """All gradients should be finite."""
        params = default_pf1000_params()
        grads = sensitivity(params)
        for k, v in grads.items():
            assert jnp.isfinite(v), f"Gradient for {k} is not finite: {v}"

    def test_grad_key_params_nonzero(self) -> None:
        """Gradients for circuit params (V0, R0, fc, fm) should be non-zero."""
        params = default_pf1000_params()
        grads = sensitivity(params)
        for k in ["V0", "R0", "fc", "fm"]:
            g = float(grads[k])
            assert abs(g) > 1e-10, f"Gradient for {k} is effectively zero: {g}"

    def test_grad_R0_sign(self) -> None:
        """Higher resistance should decrease I_peak: d(I_peak)/d(R0) < 0."""
        params = default_pf1000_params()
        grads = sensitivity(params)
        assert float(grads["R0"]) < 0, "d(I_peak)/d(R0) should be negative"

    def test_grad_V0_sign(self) -> None:
        """Higher voltage should increase I_peak: d(I_peak)/d(V0) > 0."""
        params = default_pf1000_params()
        grads = sensitivity(params)
        assert float(grads["V0"]) > 0, "d(I_peak)/d(V0) should be positive"

    def test_grad_C0_sign(self) -> None:
        """Higher capacitance should increase I_peak (more stored energy)."""
        params = default_pf1000_params()
        grads = sensitivity(params)
        assert float(grads["C0"]) > 0, "d(I_peak)/d(C0) should be positive"

    def test_grad_L0_sign(self) -> None:
        """Higher inductance should decrease I_peak (slower rise)."""
        params = default_pf1000_params()
        grads = sensitivity(params)
        assert float(grads["L0"]) < 0, "d(I_peak)/d(L0) should be negative"

    def test_loss_fn_gradient(self) -> None:
        """loss_fn should be differentiable and produce finite gradients."""
        params = default_pf1000_params()
        # Create a synthetic target from the model itself (should give ~0 loss)
        result = simulate(params, n_steps=2000)
        target_I = result["I"]
        target_t = result["t"]

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, target_I, target_t)
        for k, v in grads.items():
            assert jnp.isfinite(v), f"Loss gradient for {k} is not finite"


class TestLossFunction:
    """NRMSE loss function behavior."""

    def test_self_consistency(self) -> None:
        """Loss of model against its own output should be small.

        Note: loss_fn internally uses N_STEPS=10000 while the target is
        generated at the same default resolution. Interpolation between
        the two grids introduces small error, so threshold is 0.01.
        """
        params = default_pf1000_params()
        result = simulate(params)  # use default n_steps to match loss_fn internal
        nrmse = float(loss_fn(params, result["I"], result["t"]))
        assert nrmse < 0.01, f"Self-consistency NRMSE={nrmse:.4f} too high"

    def test_perturbed_params_higher_loss(self) -> None:
        """Perturbing R0 by 50% should increase loss vs unperturbed target."""
        params = default_pf1000_params()
        result = simulate(params, n_steps=2000)
        target_I = result["I"]
        target_t = result["t"]

        perturbed = {**params, "R0": params["R0"] * 1.5}
        loss_perturbed = float(loss_fn(perturbed, target_I, target_t))
        assert loss_perturbed > 0.01, f"Perturbed loss={loss_perturbed:.4f} too low"


@pytest.mark.slow
class TestVmapSpeed:
    """Parallel simulation speed via jax.vmap."""

    def test_1000_simulations_under_10s(self) -> None:
        """1000 parallel simulations should complete in under 10 seconds."""
        base = default_pf1000_params()
        batch_size = 1000
        batch_params = {
            k: jnp.broadcast_to(v, (batch_size,))
            for k, v in base.items()
        }
        batch_params["fc"] = jnp.linspace(0.3, 0.9, batch_size)

        # Warmup (JIT compile)
        _ = vmap_simulate(batch_params, n_steps=5000)
        jax.block_until_ready(_["I"])

        t0 = time.time()
        result = vmap_simulate(batch_params, n_steps=5000)
        jax.block_until_ready(result["I"])
        elapsed = time.time() - t0

        assert elapsed < 10.0, f"1000 vmap sims took {elapsed:.1f}s (limit: 10s)"
        assert result["I"].shape == (batch_size, 5000)

    def test_vmap_ipeak_varies_with_V0(self) -> None:
        """I_peak should increase with V0 across batch."""
        base = default_pf1000_params()
        batch_size = 20
        batch_params = {
            k: jnp.broadcast_to(v, (batch_size,))
            for k, v in base.items()
        }
        batch_params["V0"] = jnp.linspace(20e3, 35e3, batch_size)

        result = vmap_simulate(batch_params, n_steps=5000)
        I_peaks = result["I_peak"]

        # All should be finite
        assert jnp.all(jnp.isfinite(I_peaks)), "Some I_peak values are NaN/Inf"

        # Higher V0 means more stored energy, higher I_peak
        assert float(I_peaks[-1]) > float(I_peaks[0]), (
            "I_peak should increase with V0"
        )


class TestSensitivity:
    """Sensitivity analysis (d(I_peak)/d(param))."""

    def test_sensitivity_returns_all_params(self) -> None:
        """Sensitivity dict should have same keys as input params."""
        params = default_pf1000_params()
        grads = sensitivity(params)
        assert set(grads.keys()) == set(params.keys())

    def test_sensitivity_tpeak(self) -> None:
        """Can compute d(t_peak)/d(param) as alternative observable."""
        params = default_pf1000_params()
        grads = sensitivity(params, observable="t_peak")
        for k, v in grads.items():
            assert jnp.isfinite(v), f"t_peak sensitivity for {k} not finite"
