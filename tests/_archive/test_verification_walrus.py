"""T4: WALRUS surrogate verification tests.

End-to-end tests that load the real 4.8GB pretrained WALRUS checkpoint and
run inference through the DPFSurrogate pipeline. Every test is marked
``@pytest.mark.slow`` because model loading takes ~10s and a single
forward pass takes ~58s on CPU.

Prerequisites:
    - ``torch`` and ``walrus`` packages installed
    - Checkpoint directory at ``models/walrus-pretrained/`` containing
      ``walrus.pt`` + ``extended_config.yaml``

Sections:
    T4.1 — Model Loading
    T4.2 — Physical Constraints (positivity, nontriviality)
    T4.3 — Conservation Properties (mass, energy)
    T4.4 — Consistency (determinism, continuity / Lipschitz)
    T4.5 — Physics Sanity (static stability, shock propagation)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "walrus-pretrained"
_CHECKPOINT_PT = _CHECKPOINT_DIR / "walrus.pt"


def _skip_if_no_checkpoint() -> None:
    """Skip current test if the WALRUS checkpoint is not on disk."""
    if not _CHECKPOINT_PT.exists():
        pytest.skip("WALRUS checkpoint not available at models/walrus-pretrained/walrus.pt")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_NX = 16  # Minimum 3-D grid accepted by WALRUS


def make_state(
    nx: int = _NX,
    rho: float = 1e-3,
    pressure: float = 1e5,
    Te: float = 1e6,  # noqa: N803
    Ti: float = 1e6,  # noqa: N803
) -> dict[str, np.ndarray]:
    """Create a synthetic DPF state dict on a uniform 3-D grid.

    Parameters
    ----------
    nx : int
        Grid cells per dimension (``nx x nx x nx``).
    rho, pressure, Te, Ti : float
        Uniform fill values for scalar fields.

    Returns
    -------
    dict[str, np.ndarray]
        State dict with keys: rho, velocity, pressure, B, Te, Ti, psi.
    """
    return {
        "rho": np.full((nx, nx, nx), rho, dtype=np.float64),
        "velocity": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "pressure": np.full((nx, nx, nx), pressure, dtype=np.float64),
        "B": np.zeros((3, nx, nx, nx), dtype=np.float64),
        "Te": np.full((nx, nx, nx), Te, dtype=np.float64),
        "Ti": np.full((nx, nx, nx), Ti, dtype=np.float64),
        "psi": np.zeros((nx, nx, nx), dtype=np.float64),
    }


def make_history(
    n: int = 4, nx: int = _NX, **kwargs: float
) -> list[dict[str, np.ndarray]]:
    """Build a list of ``n`` identical DPF states (convenience wrapper)."""
    return [make_state(nx=nx, **kwargs) for _ in range(n)]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def surrogate():
    """Module-scoped DPFSurrogate backed by the real WALRUS checkpoint.

    Loading the 4.8 GB checkpoint takes several seconds.  Re-using the same
    instance across all tests in the module avoids repeated I/O.
    """
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("walrus")
    _skip_if_no_checkpoint()

    from dpf.ai.surrogate import DPFSurrogate

    return DPFSurrogate(checkpoint_path=_CHECKPOINT_DIR, device="cpu")


@pytest.fixture(scope="module")
def predicted_state(surrogate):
    """Module-scoped prediction from a uniform quiescent input.

    Running inference once and caching the result avoids paying the ~58 s
    forward-pass cost in every test that only inspects the output.
    """
    history = make_history(n=surrogate.history_length)
    return surrogate.predict_next_step(history)


# ===================================================================
# T4.1  Model Loading
# ===================================================================


class TestT41ModelLoading:
    """Verify that the checkpoint loads into a real WALRUS IsotropicModel."""

    @pytest.mark.slow
    def test_walrus_model_loads(self, surrogate):
        """T4.1.1 — _is_walrus_model is True after loading checkpoint."""
        assert surrogate._is_walrus_model is True, (
            "Expected a real WALRUS model (not a placeholder dict).  "
            "Check that the 'walrus' package is installed."
        )

    @pytest.mark.slow
    def test_walrus_model_field_mapping(self, surrogate):
        """T4.1.2 — DPF -> WALRUS batch -> DPF roundtrip preserves shapes."""
        history = make_history(n=surrogate.history_length)
        ref = history[0]

        # Build a WALRUS batch from the history
        batch = surrogate._build_walrus_batch(history)

        # Verify the batch tensor has the expected shape:
        # input_fields: [B=1, T=history_length, H, W, D, C=11]
        inp = batch["input_fields"]
        assert inp.shape[0] == 1, "Batch dimension should be 1"
        assert inp.shape[1] == surrogate.history_length, "Time dim mismatch"
        assert inp.shape[2] == _NX, f"H dim: expected {_NX}"
        assert inp.shape[3] == _NX, f"W dim: expected {_NX}"
        assert inp.shape[4] == _NX, f"D dim: expected {_NX}"
        assert inp.shape[5] == 11, "Channel dim should be 11"

        # Verify scalar fill values survive the DPF -> batch conversion
        inp_np = inp.cpu().numpy()
        # Channel 0 is rho (see _SCALAR_KEYS ordering in surrogate.py)
        rho_channel = inp_np[0, 0, :, :, :, 0]
        np.testing.assert_allclose(
            rho_channel,
            np.float32(ref["rho"].flat[0]),
            rtol=1e-5,
            err_msg="rho channel value mismatch after DPF->batch",
        )


# ===================================================================
# T4.2  Physical Constraints
# ===================================================================


class TestT42PhysicalConstraints:
    """Verify that WALRUS predictions obey basic physical constraints."""

    @pytest.mark.slow
    def test_walrus_density_positive(self, predicted_state):
        """T4.2.1 — Predicted density should be positive everywhere."""
        rho = predicted_state["rho"]
        # Allow a tiny negative margin for floating-point noise
        assert np.all(rho > -1e-10), (
            f"Density has large negative values: min={rho.min():.6e}"
        )
        # Warn if any values are negative but very small
        if np.any(rho < 0):
            neg_frac = np.mean(rho < 0)
            assert neg_frac < 0.01, (
                f"{neg_frac:.1%} of density cells are negative "
                f"(min={rho.min():.6e})"
            )

    @pytest.mark.slow
    def test_walrus_pressure_positive(self, predicted_state):
        """T4.2.2 — Predicted pressure should be positive everywhere."""
        p = predicted_state["pressure"]
        assert np.all(p > -1e-10), (
            f"Pressure has large negative values: min={p.min():.6e}"
        )
        if np.any(p < 0):
            neg_frac = np.mean(p < 0)
            assert neg_frac < 0.01, (
                f"{neg_frac:.1%} of pressure cells are negative "
                f"(min={p.min():.6e})"
            )

    @pytest.mark.slow
    def test_walrus_prediction_nontrivial(self, surrogate, predicted_state):
        """T4.2.3 — Output should differ from the last input state (delta prediction)."""
        last_input = make_state()

        # Check at least one scalar field has changed meaningfully
        any_changed = False
        for key in ("rho", "pressure", "Te", "Ti"):
            diff = np.abs(predicted_state[key] - last_input[key])
            rel_change = np.max(diff) / max(np.max(np.abs(last_input[key])), 1e-30)
            if rel_change > 1e-6:
                any_changed = True
                break

        assert any_changed, (
            "Prediction is identical to input — delta prediction may be "
            "trivially zero. Check RevIN normalisation and model forward pass."
        )


# ===================================================================
# T4.3  Conservation Properties
# ===================================================================


class TestT43Conservation:
    """Verify approximate conservation of integral quantities."""

    @pytest.mark.slow
    def test_walrus_mass_approximately_conserved(self, surrogate, predicted_state):
        """T4.3.1 — Total mass should change less than 10 % per step."""
        input_state = make_state()
        mass_in = np.sum(input_state["rho"])
        mass_out = np.sum(predicted_state["rho"])

        if mass_in > 0:
            rel_change = abs(mass_out - mass_in) / mass_in
            assert rel_change < 0.10, (
                f"Mass changed by {rel_change:.1%} "
                f"(in={mass_in:.6e}, out={mass_out:.6e})"
            )

    @pytest.mark.slow
    def test_walrus_energy_bounded(self, surrogate, predicted_state):
        """T4.3.2 — Total energy should not grow unboundedly.

        Compute thermal + kinetic + magnetic energy and check that
        the output energy is within 10x of the input energy.
        """
        def total_energy(state: dict[str, np.ndarray]) -> float:
            # Thermal: p / (gamma - 1)
            gamma = 5.0 / 3.0
            E_th = np.sum(state["pressure"]) / (gamma - 1.0)

            # Kinetic: 0.5 * rho * v^2
            v2 = np.sum(state["velocity"] ** 2, axis=0)
            E_kin = 0.5 * np.sum(state["rho"] * v2)

            # Magnetic: B^2 / (2 * mu0)
            mu0 = 4.0 * np.pi * 1e-7
            B2 = np.sum(state["B"] ** 2, axis=0)
            E_mag = np.sum(B2) / (2.0 * mu0)

            return float(E_th + E_kin + E_mag)

        input_state = make_state()
        E_in = total_energy(input_state)
        E_out = total_energy(predicted_state)

        # Allow up to 10x growth (generous bound for a pretrained model
        # evaluated on out-of-distribution data)
        assert E_out < 10.0 * max(E_in, 1e-30), (
            f"Energy grew excessively: E_in={E_in:.6e}, E_out={E_out:.6e}"
        )


# ===================================================================
# T4.4  Consistency
# ===================================================================


class TestT44Consistency:
    """Verify determinism and Lipschitz continuity."""

    @pytest.mark.slow
    def test_walrus_deterministic(self, surrogate):
        """T4.4.1 — Same inputs yield identical outputs."""
        history = make_history(n=surrogate.history_length)

        pred_a = surrogate.predict_next_step(history)
        pred_b = surrogate.predict_next_step(history)

        for key in pred_a:
            np.testing.assert_array_equal(
                pred_a[key],
                pred_b[key],
                err_msg=f"Field '{key}' differs between two identical runs",
            )

    @pytest.mark.slow
    def test_walrus_continuous(self, surrogate):
        """T4.4.2 — Small input perturbation produces small output change.

        A Lipschitz-like bound: if the input changes by epsilon,
        the output should change by at most K * epsilon for some
        reasonable constant K.
        """
        history_base = make_history(n=surrogate.history_length)

        # Perturb only the last state's density by 1 %
        history_pert = [
            {k: v.copy() for k, v in s.items()} for s in history_base
        ]
        epsilon = 0.01 * history_pert[-1]["rho"].mean()
        history_pert[-1]["rho"] += epsilon

        pred_base = surrogate.predict_next_step(history_base)
        pred_pert = surrogate.predict_next_step(history_pert)

        # Compute max relative change across all scalar fields
        max_rel_change = 0.0
        for key in ("rho", "pressure", "Te", "Ti", "psi"):
            diff = np.max(np.abs(pred_pert[key] - pred_base[key]))
            scale = max(np.max(np.abs(pred_base[key])), 1e-30)
            max_rel_change = max(max_rel_change, diff / scale)

        # A 1 % density perturbation should not cause > 100 % output change
        K_bound = 100.0
        assert max_rel_change < K_bound, (
            f"Output changed by {max_rel_change:.2e} relative — "
            f"exceeds Lipschitz bound K={K_bound} for 1 % input perturbation"
        )


# ===================================================================
# T4.5  Physics Sanity
# ===================================================================


class TestT45PhysicsSanity:
    """Sanity checks grounded in physical intuition."""

    @pytest.mark.slow
    def test_walrus_static_state_stable(self, surrogate, predicted_state):
        """T4.5.1 — Uniform quiescent state should remain approximately uniform.

        Starting from a spatially uniform state with zero velocity and
        zero magnetic field, the output should not develop large spatial
        gradients (the model should recognise there is no driving force).
        """
        for key in ("rho", "pressure", "Te", "Ti"):
            field = predicted_state[key]
            field_mean = np.mean(field)
            field_std = np.std(field)

            if abs(field_mean) > 1e-30:
                cv = field_std / abs(field_mean)
                # Coefficient of variation should be small for a
                # "do-nothing" input.  Allow up to 50 % for the
                # pretrained model on OOD data.
                assert cv < 0.50, (
                    f"Field '{key}' developed excessive spatial variation: "
                    f"CV={cv:.2%} (mean={field_mean:.6e}, std={field_std:.6e})"
                )

    @pytest.mark.slow
    def test_walrus_shock_propagates(self, surrogate):
        """T4.5.2 — State with a sharp gradient should show the gradient moving.

        Create a simple Sod-like initial condition with a density jump
        at the midplane, run one step, and verify that the density
        profile has changed relative to the input.
        """
        nx = _NX
        mid = nx // 2

        # Build history: all identical with a density jump at x = mid
        def sod_state() -> dict[str, np.ndarray]:
            s = make_state(nx=nx)
            # Left half: high density, right half: low density
            s["rho"][:mid, :, :] = 1.0
            s["rho"][mid:, :, :] = 0.125
            # Corresponding pressure jump
            s["pressure"][:mid, :, :] = 1e5
            s["pressure"][mid:, :, :] = 1e4
            return s

        history = [sod_state() for _ in range(surrogate.history_length)]
        pred = surrogate.predict_next_step(history)

        # The density profile should differ from the initial condition
        input_rho = history[-1]["rho"]
        pred_rho = pred["rho"]

        diff = np.abs(pred_rho - input_rho)
        max_diff = np.max(diff)

        # The gradient should have moved — at least *some* cells should
        # have changed.  We use a very loose criterion because the model
        # is pretrained and this is OOD data.
        assert max_diff > 1e-10, (
            f"Density profile did not change at all after one step "
            f"(max |delta rho| = {max_diff:.6e}).  The model may be "
            f"returning a trivial copy."
        )

    @pytest.mark.slow
    def test_walrus_output_no_nans_or_infs(self, predicted_state):
        """T4.5.3 — Output fields should contain no NaN or Inf values."""
        for key, arr in predicted_state.items():
            assert np.all(np.isfinite(arr)), (
                f"Field '{key}' contains NaN or Inf values: "
                f"NaN count={np.sum(np.isnan(arr))}, "
                f"Inf count={np.sum(np.isinf(arr))}"
            )
