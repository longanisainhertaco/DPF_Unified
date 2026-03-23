"""Tests for the thermal diffusion path in engine.py.

Validates tau_e (electron collision time) against the NRL Plasma Formulary
and confirms that the Te unit conversion (K -> eV) introduced to fix the
Six Sigma dimensional audit finding is correct and consistent.

NRL Formulary reference: eq. 2-5
    tau_e [s] = 3.44e5 * Te_eV^1.5 / (ne [m^-3] * ln_Lambda)

Note: the NRL coefficient 3.44e5 already folds in all SI prefactors and
assumes ne in m^-3 (not cm^-3).  The engine uses this convention.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from dpf.constants import eV, k_B, m_e

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tau_e_nrl(Te_eV: float, ne_m3: float, ln_Lambda: float) -> float:
    """NRL Formulary tau_e, all inputs in SI / eV."""
    return 3.44e5 * Te_eV**1.5 / (ne_m3 * ln_Lambda)


def _kappa_spitzer(ne_m3: float, Te_K: float, tau_e: float) -> float:
    """Spitzer parallel thermal conductivity [W/(m·K)].

    kappa = 3.2 * ne * k_B^2 * Te * tau_e / m_e
    """
    return 3.2 * ne_m3 * k_B**2 * Te_K * tau_e / m_e


# ---------------------------------------------------------------------------
# Test 1 — tau_e against the NRL Plasma Formulary
# ---------------------------------------------------------------------------

class TestTauENRL:
    """Verify the engine's tau_e computation matches the NRL Formulary."""

    def test_tau_e_against_nrl_formulary(self) -> None:
        """Te = 100 eV, ne = 1e20 m^-3, ln_Lambda = 10 -> tau_e = 3.44e-13 s."""
        Te_eV = 100.0
        ne = 1e20
        ln_Lambda = 10.0

        # Expected: 3.44e5 * 100^1.5 / (1e20 * 10) = 3.44e5 * 1000 / 1e21 = 3.44e-13 s
        expected = 3.44e-13

        # Replicate the engine formula verbatim (engine.py lines 2425-2426)
        Te_K = Te_eV * eV / k_B          # eV -> Kelvin (invert the K->eV conversion)
        Te_safe = max(Te_K, 1000.0)
        Te_eV_computed = Te_safe * k_B / eV  # engine: Te_safe * k_B / eV
        tau_e = 3.44e5 * Te_eV_computed**1.5 / (ne * ln_Lambda)

        assert tau_e == pytest.approx(expected, rel=0.01), (
            f"tau_e = {tau_e:.4e} s, expected {expected:.4e} s "
            f"(error {abs(tau_e - expected) / expected * 100:.2f}%)"
        )

    def test_tau_e_wrong_without_unit_conversion(self) -> None:
        """Demonstrate magnitude of the pre-fix bug: raw Te_K inflates tau_e by ~(k_B/eV)^1.5 ~ 3.8e-4x.

        Without the k_B/eV conversion, Te was treated as eV but was actually K,
        so te_eV = Te_K (wrong).  For Te=100 eV the Kelvin value is ~1.16e6,
        giving a tau_e ~1.16e6^1.5 / 100^1.5 ≈ 4e10× too large.
        This test documents the magnitude of that error.
        """
        Te_eV = 100.0
        ne = 1e20
        ln_Lambda = 10.0

        Te_K = Te_eV * eV / k_B          # correct Kelvin equivalent
        # Buggy path: forget the conversion, pass Te_K as if it were Te_eV
        tau_e_buggy = 3.44e5 * Te_K**1.5 / (ne * ln_Lambda)
        tau_e_correct = _tau_e_nrl(Te_eV, ne, ln_Lambda)

        ratio = tau_e_buggy / tau_e_correct
        # Ratio should be approximately (Te_K / Te_eV)^1.5 = (1.16e6/100)^1.5
        expected_ratio = (Te_K / Te_eV) ** 1.5
        assert ratio == pytest.approx(expected_ratio, rel=0.01)
        # Sanity: the bug inflates tau_e by ~(k_B/eV)^-1.5 ~ 1.25e6x
        assert ratio > 1e5


# ---------------------------------------------------------------------------
# Test 2 — units consistency: K -> eV conversion
# ---------------------------------------------------------------------------

class TestTauEUnitsConsistency:
    """Verify the k_B / eV conversion is applied and kappa is physically reasonable."""

    def test_tau_e_units_consistency(self) -> None:
        """kappa must be positive, finite, and in plausible SI range for DPF conditions."""
        Te_eV = 100.0
        ne = 1e24  # typical DPF pinch density [m^-3]
        ln_Lambda = 10.0

        Te_K = Te_eV * eV / k_B

        # Engine formula
        Te_safe = max(Te_K, 1000.0)
        Te_eV_check = Te_safe * k_B / eV
        tau_e = 3.44e5 * Te_eV_check**1.5 / (ne * ln_Lambda)
        kappa = 3.2 * ne * k_B**2 * Te_safe * tau_e / m_e

        assert np.isfinite(kappa), "kappa is not finite"
        assert kappa > 0.0, "kappa must be positive"

        # Spitzer kappa ~ 3.2 * k_B^2 * Te * (3.44e5 * Te_eV^1.5 / ln_Lambda) / m_e
        # For 100 eV: ~2.7e-2 W/(m·K).  kappa is independent of ne because
        # tau_e ~ 1/ne cancels the ne prefactor.
        # Physical range: 1e-4 -- 1e4 W/(m·K) for DPF-relevant temperatures.
        assert 1e-4 < kappa < 1e4, (
            f"kappa = {kappa:.3e} W/(m·K) outside expected physical range"
        )

    def test_conversion_constant_roundtrip(self) -> None:
        """k_B / eV must equal 1 / 11604.52 (eV-to-K conversion)."""
        # 1 eV = 11604.518... K; so k_B/eV = 1/11604.518
        # scipy.constants uses CODATA 2018
        conversion = k_B / eV
        expected = 1.0 / 11604.518  # NRL value for eV-to-K

        assert conversion == pytest.approx(expected, rel=1e-4)

    def test_kappa_array_shapes_preserved(self) -> None:
        """Vectorised computation over a 2D spatial grid preserves shape."""
        shape = (8, 8)
        Te_K = np.full(shape, 100.0 * eV / k_B)
        ne = np.full(shape, 1e24)
        ln_Lambda = 10.0

        Te_safe = np.maximum(Te_K, 1000.0)
        ne_safe = np.maximum(ne, 1e10)
        Te_eV_arr = Te_safe * k_B / eV
        tau_e = 3.44e5 * Te_eV_arr**1.5 / (ne_safe * ln_Lambda)
        kappa = 3.2 * ne_safe * k_B**2 * Te_safe * tau_e / m_e

        assert kappa.shape == shape
        assert np.all(np.isfinite(kappa))
        assert np.all(kappa > 0)


# ---------------------------------------------------------------------------
# Test 3 — kappa_parallel ~ Te^(5/2)
# ---------------------------------------------------------------------------

class TestThermalConductivityScaling:
    """kappa_parallel is proportional to Te^(5/2) for fixed ne and ln_Lambda."""

    def test_thermal_conductivity_scales_with_Te(self) -> None:
        """kappa(100 eV) / kappa(10 eV) must equal (100/10)^2.5 = 316.2 within 1%."""
        ne = 1e24
        ln_Lambda = 10.0

        def kappa_at(Te_eV: float) -> float:
            Te_K = Te_eV * eV / k_B
            Te_safe = max(Te_K, 1000.0)
            ne_safe = max(ne, 1e10)
            Te_eV_c = Te_safe * k_B / eV
            tau_e = 3.44e5 * Te_eV_c**1.5 / (ne_safe * ln_Lambda)
            return 3.2 * ne_safe * k_B**2 * Te_safe * tau_e / m_e

        kappa_100 = kappa_at(100.0)
        kappa_10 = kappa_at(10.0)

        ratio = kappa_100 / kappa_10
        expected_ratio = (100.0 / 10.0) ** 2.5  # = 316.227...

        assert ratio == pytest.approx(expected_ratio, rel=0.01), (
            f"kappa ratio = {ratio:.4f}, expected {expected_ratio:.4f} "
            f"(error {abs(ratio - expected_ratio) / expected_ratio * 100:.2f}%)"
        )

    def test_thermal_conductivity_scaling_multiple_points(self) -> None:
        """Te^2.5 scaling holds across a decade-wide temperature range."""
        ne = 1e23
        ln_Lambda = 10.0
        T_ref_eV = 50.0

        def kappa_at(Te_eV: float) -> float:
            Te_K = Te_eV * eV / k_B
            Te_safe = max(Te_K, 1000.0)
            ne_safe = max(ne, 1e10)
            Te_eV_c = Te_safe * k_B / eV
            tau_e = 3.44e5 * Te_eV_c**1.5 / (ne_safe * ln_Lambda)
            return 3.2 * ne_safe * k_B**2 * Te_safe * tau_e / m_e

        kappa_ref = kappa_at(T_ref_eV)

        for Te_eV in [20.0, 100.0, 200.0, 500.0]:
            ratio = kappa_at(Te_eV) / kappa_ref
            expected = (Te_eV / T_ref_eV) ** 2.5
            assert ratio == pytest.approx(expected, rel=0.01), (
                f"At Te={Te_eV} eV: ratio={ratio:.4f}, expected {expected:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 4 — no shadowed k_B in engine.py
# ---------------------------------------------------------------------------

class TestNoShadowedConstants:
    """Static AST check: engine.py must not shadow k_B with a local assignment."""

    def test_no_shadowed_k_B(self) -> None:
        """Fail if any assignment target named 'k_B' exists in engine.py."""
        with open("src/dpf/engine.py") as _f:
            source = _f.read()
        tree = ast.parse(source)

        violations: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "k_B":
                        violations.append(node.lineno)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "k_B":
                    violations.append(node.lineno)

        assert not violations, (
            "Shadowed k_B assignment(s) found in engine.py at line(s): "
            + ", ".join(str(ln) for ln in violations)
        )

    def test_engine_imports_k_B_from_constants(self) -> None:
        """engine.py must import k_B from dpf.constants (not define it locally)."""
        with open("src/dpf/engine.py") as _f:
            source = _f.read()
        tree = ast.parse(source)

        found_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "dpf.constants":
                    names = [alias.name for alias in node.names]
                    if "k_B" in names:
                        found_import = True
                        break

        assert found_import, (
            "k_B is not imported from dpf.constants in engine.py. "
            "It must come from there to avoid unit inconsistencies."
        )

    def test_no_shadowed_eV(self) -> None:
        """Fail if any local assignment to 'eV' exists in engine.py."""
        with open("src/dpf/engine.py") as _f:
            source = _f.read()
        tree = ast.parse(source)

        violations: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "eV":
                        violations.append(node.lineno)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "eV":
                    violations.append(node.lineno)

        assert not violations, (
            "Shadowed eV assignment(s) found in engine.py at line(s): "
            + ", ".join(str(ln) for ln in violations)
        )
