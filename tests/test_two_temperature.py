"""Tests for the two-temperature electron energy evolution module.

Validates:
    1. Uniform state preservation (no source terms -> e_e unchanged)
    2. Equilibration: Te=100eV, Ti=10eV -> converge to common T
    3. Ohmic heating: Q_ohm > 0 increases Te only
    4. Energy conservation: e_e + e_i = e_total at all times
    5. Radiation cooling: Q_rad > 0 decreases Te
    6. Roundtrip Te <-> e_e conversions
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann as k_B

from dpf.constants import eV, m_d
from dpf.fluid.two_temperature import (
    compute_equilibration_source,
    compute_ohmic_heating,
    compute_radiation_loss,
    electron_energy_from_temperature,
    electron_energy_rhs,
    initialize_electron_energy,
    ion_temperature_from_total,
    step_electron_energy,
    temperature_from_electron_energy,
)

# --- Fixtures ---

GRID_SHAPE = (16, 16)
DX = 0.01  # 1 cm
ION_MASS = m_d
Z = 1.0
GAMMA = 5.0 / 3.0


def _uniform_state(
    Te_K: float = 1.16e7,  # ~1 keV
    Ti_K: float = 1.16e6,  # ~0.1 keV
    n_e: float = 1e25,
    vx: float = 0.0,
) -> dict:
    """Build a uniform plasma state for testing."""
    Te = np.full(GRID_SHAPE, Te_K)
    Ti = np.full(GRID_SHAPE, Ti_K)
    ne = np.full(GRID_SHAPE, n_e)
    ni = ne / Z
    rho = ni * ION_MASS
    velocity = np.zeros((3, *GRID_SHAPE))
    velocity[0] = vx
    eta = np.full(GRID_SHAPE, 1e-6)  # Ohm*m
    J_sq = np.zeros(GRID_SHAPE)
    rho_e_e = electron_energy_from_temperature(Te, ne)
    p_total = ne * k_B * Te + ni * k_B * Ti
    e_total = p_total / (GAMMA - 1.0)
    return {
        "Te": Te, "Ti": Ti, "n_e": ne, "n_i": ni, "rho": rho,
        "velocity": velocity, "eta": eta, "J_sq": J_sq,
        "rho_e_e": rho_e_e, "p_total": p_total, "e_total": e_total,
    }


# --- Conversion roundtrip tests ---

class TestConversions:
    def test_roundtrip_Te_to_energy_and_back(self) -> None:
        Te = np.array([1e6, 1e7, 1e8])
        n_e = np.array([1e24, 1e25, 1e26])
        rho_e_e = electron_energy_from_temperature(Te, n_e)
        Te_recovered = temperature_from_electron_energy(rho_e_e, n_e)
        np.testing.assert_allclose(Te_recovered, Te, rtol=1e-14)

    def test_energy_density_positive(self) -> None:
        Te = np.array([1.0, 1e4, 1e8])
        n_e = np.array([1e20, 1e24, 1e26])
        rho_e_e = electron_energy_from_temperature(Te, n_e)
        assert np.all(rho_e_e > 0)

    def test_temperature_floor(self) -> None:
        rho_e_e = np.array([0.0, -1.0, 1e-30])
        n_e = np.array([1e25, 1e25, 1e25])
        Te = temperature_from_electron_energy(rho_e_e, n_e, Te_floor=100.0)
        assert np.all(Te >= 100.0)

    def test_ion_temperature_from_total(self) -> None:
        Te = 1e7
        Ti = 5e6
        n_e = 1e25
        n_i = n_e / Z
        e_total_val = 1.5 * (n_e * k_B * Te + n_i * k_B * Ti)
        rho_e_e_val = 1.5 * n_e * k_B * Te
        e_total = np.full(GRID_SHAPE, e_total_val)
        rho_e_e = np.full(GRID_SHAPE, rho_e_e_val)
        n_i_arr = np.full(GRID_SHAPE, n_i)
        Ti_recovered = ion_temperature_from_total(e_total, rho_e_e, n_i_arr)
        np.testing.assert_allclose(Ti_recovered, Ti, rtol=1e-12)


# --- Uniform state preservation ---

class TestUniformPreservation:
    def test_no_sources_preserves_energy(self) -> None:
        """With zero velocity, zero J, zero radiation -> e_e unchanged."""
        s = _uniform_state(vx=0.0)
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0
        # Set Te = Ti to eliminate equilibration
        s["Ti"][:] = s["Te"]
        rho_e_e_new, Te_new, Ti_new = step_electron_energy(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=1e-9,
            Z=Z, gaunt_factor=0.0,  # disable radiation
        )
        # Equilibration with Te=Ti should be identity
        np.testing.assert_allclose(
            Te_new, s["Te"], rtol=1e-10,
            err_msg="Te changed with no source terms",
        )

    def test_uniform_velocity_no_change(self) -> None:
        """Uniform velocity has div(v)=0 -> no compressional work."""
        s = _uniform_state(vx=1e5)
        s["Ti"][:] = s["Te"]
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0
        rho_e_e_new, Te_new, _ = step_electron_energy(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=1e-9,
            Z=Z, gaunt_factor=0.0,
        )
        np.testing.assert_allclose(Te_new, s["Te"], rtol=1e-10)


# --- Equilibration tests ---

class TestEquilibration:
    def test_hot_electrons_cool_toward_ions(self) -> None:
        """Te >> Ti: electrons should lose energy to ions."""
        Te_init = 100 * eV / k_B   # 100 eV in K
        Ti_init = 10 * eV / k_B    # 10 eV in K
        s = _uniform_state(Te_K=Te_init, Ti_K=Ti_init)
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0

        _, Te_new, Ti_new = step_electron_energy(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=1e-9,
            Z=Z, gaunt_factor=0.0,
        )
        assert np.all(Te_new < s["Te"]), "Te should decrease when Te > Ti"
        assert np.all(Ti_new > s["Ti"]), "Ti should increase when Te > Ti"

    def test_equilibration_conserves_weighted_temperature(self) -> None:
        """Z*Te + Ti should be approximately conserved during equilibration."""
        Te_init = 100 * eV / k_B
        Ti_init = 10 * eV / k_B
        s = _uniform_state(Te_K=Te_init, Ti_K=Ti_init)
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0

        T_weighted_before = Z * s["Te"] + s["Ti"]

        _, Te_new, Ti_new = step_electron_energy(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=1e-9,
            Z=Z, gaunt_factor=0.0,
        )
        T_weighted_after = Z * Te_new + Ti_new
        np.testing.assert_allclose(
            T_weighted_after, T_weighted_before, rtol=1e-6,
            err_msg="Weighted temperature Z*Te + Ti not conserved",
        )

    def test_long_equilibration_converges(self) -> None:
        """Many small steps should converge Te -> Ti -> T_eq."""
        Te_init = 1000 * eV / k_B
        Ti_init = 10 * eV / k_B
        s = _uniform_state(Te_K=Te_init, Ti_K=Ti_init)
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0

        Te = s["Te"].copy()
        Ti = s["Ti"].copy()
        rho_e_e = s["rho_e_e"].copy()
        dt = 1e-10

        for _ in range(10000):
            rho_e_e, Te, Ti = step_electron_energy(
                rho_e_e=rho_e_e, rho=s["rho"], velocity=s["velocity"],
                eta=s["eta"], J_sq=s["J_sq"], Te=Te, Ti=Ti,
                n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=dt,
                Z=Z, gaunt_factor=0.0,
            )

        T_eq_expected = (Z * Te_init + Ti_init) / (Z + 1.0)
        np.testing.assert_allclose(
            Te[0, 0], T_eq_expected, rtol=0.05,
            err_msg="Te did not converge to equilibrium",
        )
        np.testing.assert_allclose(
            Ti[0, 0], T_eq_expected, rtol=0.05,
            err_msg="Ti did not converge to equilibrium",
        )


# --- Ohmic heating tests ---

class TestOhmicHeating:
    def test_ohmic_increases_electron_energy(self) -> None:
        """Ohmic heating should increase Te, not Ti."""
        s = _uniform_state()
        s["Ti"][:] = s["Te"]  # equal temps -> no equilibration
        s["J_sq"][:] = 1e14  # 10^7 A/m^2 squared
        s["eta"][:] = 1e-5

        _, Te_new, Ti_new = step_electron_energy(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=1e-9,
            Z=Z, gaunt_factor=0.0,
        )
        assert np.all(Te_new > s["Te"]), "Ohmic heating should increase Te"
        # Ti changes only through equilibration (Te > Ti after heating)
        # With Te=Ti initially and Ohmic heating making Te>Ti,
        # equilibration will push Ti up slightly
        dTi = np.mean(Ti_new - s["Ti"])
        dTe = np.mean(Te_new - s["Te"])
        assert dTe > 10 * abs(dTi), (
            f"Ohmic heating should primarily heat electrons: dTe={dTe:.2e}, dTi={dTi:.2e}"
        )

    def test_ohmic_heating_magnitude(self) -> None:
        """Check Q_ohm = eta * J^2 dimensionally."""
        eta = np.array([1e-5])
        J_sq = np.array([1e14])
        Q = compute_ohmic_heating(eta, J_sq)
        expected = 1e-5 * 1e14  # = 1e9 W/m^3
        np.testing.assert_allclose(Q, expected, rtol=1e-14)


# --- Radiation cooling tests ---

class TestRadiationCooling:
    def test_radiation_cools_electrons(self) -> None:
        """Bremsstrahlung should decrease Te."""
        s = _uniform_state(Te_K=1.16e8)  # 10 keV
        s["Ti"][:] = s["Te"]
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0

        _, Te_new, _ = step_electron_energy(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=1e-9,
            Z=Z, gaunt_factor=1.2,
        )
        assert np.all(Te_new < s["Te"]), "Radiation should cool electrons"

    def test_radiation_loss_positive(self) -> None:
        """Radiation loss rate should always be non-negative."""
        Te = np.array([1e5, 1e7, 1e9])
        n_e = np.array([1e23, 1e25, 1e27])
        Q_rad = compute_radiation_loss(Te, n_e)
        assert np.all(Q_rad >= 0)


# --- Energy conservation tests ---

class TestEnergyConservation:
    def test_total_energy_budget(self) -> None:
        """e_e + e_i should equal e_total when only equilibration acts."""
        Te_init = 50 * eV / k_B
        Ti_init = 5 * eV / k_B
        n_e = 1e25
        s = _uniform_state(Te_K=Te_init, Ti_K=Ti_init, n_e=n_e)
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0

        e_total_before = 1.5 * (s["n_e"] * k_B * s["Te"] + s["n_i"] * k_B * s["Ti"])

        rho_e_e_new, Te_new, Ti_new = step_electron_energy(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=1e-9,
            Z=Z, gaunt_factor=0.0,
        )
        e_e_after = rho_e_e_new
        e_i_after = 1.5 * s["n_i"] * k_B * Ti_new
        e_total_after = e_e_after + e_i_after

        np.testing.assert_allclose(
            e_total_after, e_total_before, rtol=1e-6,
            err_msg="Total internal energy not conserved during equilibration",
        )

    def test_ion_energy_derived_correctly(self) -> None:
        """Ion energy = total - electron energy."""
        Te = 1e7
        Ti = 5e6
        n_e = 1e25
        n_i = n_e / Z
        e_e = 1.5 * n_e * k_B * Te
        e_i = 1.5 * n_i * k_B * Ti
        e_total = e_e + e_i

        e_total_arr = np.full(GRID_SHAPE, e_total)
        rho_e_e_arr = np.full(GRID_SHAPE, e_e)
        n_i_arr = np.full(GRID_SHAPE, n_i)

        Ti_recovered = ion_temperature_from_total(e_total_arr, rho_e_e_arr, n_i_arr)
        np.testing.assert_allclose(Ti_recovered, Ti, rtol=1e-12)


# --- Equilibration source term tests ---

class TestEquilibrationSource:
    def test_positive_when_Ti_gt_Te(self) -> None:
        """Q_ei > 0 when Ti > Te (energy flows to electrons)."""
        Te = np.array([1e6])
        Ti = np.array([1e7])
        n_e = np.array([1e25])
        Q = compute_equilibration_source(Te, Ti, n_e, Z)
        assert Q[0] > 0, "Q_ei should be positive when Ti > Te"

    def test_negative_when_Te_gt_Ti(self) -> None:
        """Q_ei < 0 when Te > Ti (energy flows from electrons)."""
        Te = np.array([1e7])
        Ti = np.array([1e6])
        n_e = np.array([1e25])
        Q = compute_equilibration_source(Te, Ti, n_e, Z)
        assert Q[0] < 0, "Q_ei should be negative when Te > Ti"

    def test_zero_at_equilibrium(self) -> None:
        """Q_ei = 0 when Te = Ti."""
        Te = np.array([1e7])
        Ti = np.array([1e7])
        n_e = np.array([1e25])
        Q = compute_equilibration_source(Te, Ti, n_e, Z)
        np.testing.assert_allclose(Q, 0.0, atol=1e-10)


# --- Initialize from existing state ---

class TestInitialization:
    def test_initialize_from_temperature(self) -> None:
        """initialize_electron_energy should produce correct e_e."""
        Te = np.full(GRID_SHAPE, 1e7)
        Ti = np.full(GRID_SHAPE, 5e6)
        n_e = 1e25
        n_i = n_e / Z
        rho = np.full(GRID_SHAPE, n_i * ION_MASS)
        p = np.full(GRID_SHAPE, (n_e * k_B * 1e7 + n_i * k_B * 5e6))

        rho_e_e = initialize_electron_energy(Te, Ti, p, rho, ION_MASS, Z)
        expected = 1.5 * n_e * k_B * 1e7
        np.testing.assert_allclose(rho_e_e, expected, rtol=1e-12)

    def test_initialize_roundtrip(self) -> None:
        """Initialize -> recover Te -> should match."""
        Te_orig = np.full(GRID_SHAPE, 5e7)
        n_i = 1e25
        rho = np.full(GRID_SHAPE, n_i * ION_MASS)
        Ti = np.full(GRID_SHAPE, 1e6)
        p = np.full(GRID_SHAPE, Z * n_i * k_B * 5e7 + n_i * k_B * 1e6)

        rho_e_e = initialize_electron_energy(Te_orig, Ti, p, rho, ION_MASS, Z)
        n_e = Z * rho / ION_MASS
        Te_back = temperature_from_electron_energy(rho_e_e, n_e)
        np.testing.assert_allclose(Te_back, Te_orig, rtol=1e-12)


# --- RHS function tests ---

class TestElectronEnergyRHS:
    def test_rhs_zero_for_quiescent_equal_T(self) -> None:
        """No sources, no velocity, Te=Ti -> RHS ~ 0."""
        s = _uniform_state()
        s["Ti"][:] = s["Te"]
        s["J_sq"][:] = 0.0
        s["eta"][:] = 0.0

        rhs = electron_energy_rhs(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, Z=Z, gaunt_factor=0.0,
        )
        np.testing.assert_allclose(rhs, 0.0, atol=1e-10)

    def test_rhs_positive_with_ohmic_heating(self) -> None:
        """Ohmic heating should produce positive RHS."""
        s = _uniform_state()
        s["Ti"][:] = s["Te"]
        s["J_sq"][:] = 1e14
        s["eta"][:] = 1e-5

        rhs = electron_energy_rhs(
            rho_e_e=s["rho_e_e"], rho=s["rho"], velocity=s["velocity"],
            eta=s["eta"], J_sq=s["J_sq"], Te=s["Te"], Ti=s["Ti"],
            n_e=s["n_e"], n_i=s["n_i"], dx=DX, Z=Z, gaunt_factor=0.0,
        )
        assert np.all(rhs > 0), "Ohmic heating should give positive RHS"
