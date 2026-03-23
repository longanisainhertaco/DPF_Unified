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
        assert np.all(np.isfinite(rho_e_e)), "Intermediate energy density must be finite"

    def test_energy_density_positive(self) -> None:
        Te = np.array([1.0, 1e4, 1e8])
        n_e = np.array([1e20, 1e24, 1e26])
        rho_e_e = electron_energy_from_temperature(Te, n_e)
        assert np.all(rho_e_e > 0)
        # Energy scales as n_e * Te: highest-T, highest-n case should dominate
        assert rho_e_e[2] > rho_e_e[0], "Higher Te*n_e should give higher energy density"

    def test_temperature_floor(self) -> None:
        rho_e_e = np.array([0.0, -1.0, 1e-30])
        n_e = np.array([1e25, 1e25, 1e25])
        Te = temperature_from_electron_energy(rho_e_e, n_e, Te_floor=100.0)
        assert np.all(Te >= 100.0)
        assert np.all(np.isfinite(Te)), "Temperature floor must produce finite values"

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
        assert np.all(Ti_recovered > 0), "Recovered Ti must be positive"


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
        assert np.all(np.isfinite(rho_e_e_new)), "rho_e_e must remain finite"
        assert np.all(Te_new > 0), "Te must stay positive with no source terms"

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
        assert np.all(np.isfinite(rho_e_e_new)), "rho_e_e must be finite after uniform-velocity step"


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
        assert np.all(Te_new > 0) and np.all(Ti_new > 0), "Both temperatures must remain positive"

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
        assert np.all(Q > 0), "Ohmic heating must be positive for J > 0"


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
        assert np.all(Te_new > 0), "Te must remain positive after radiation cooling"

    def test_radiation_loss_positive(self) -> None:
        """Radiation loss rate should always be non-negative."""
        Te = np.array([1e5, 1e7, 1e9])
        n_e = np.array([1e23, 1e25, 1e27])
        Q_rad = compute_radiation_loss(Te, n_e)
        assert np.all(Q_rad >= 0)
        # Radiation scales as n_e^2 * Te^{1/2}: highest-density, highest-T case dominates
        assert np.all(np.isfinite(Q_rad)), "Radiation loss must be finite across all temperatures"


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
        assert np.all(rho_e_e_new > 0), "Electron energy density must remain positive"

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
        # Ti should be strictly less than total thermal temperature (electrons take some share)
        T_total_mean = e_total / (1.5 * (n_e + n_i) * k_B)
        assert float(np.mean(Ti_recovered)) < T_total_mean * 1.001, (
            "Ion temperature should not exceed mean total temperature"
        )


# --- Equilibration source term tests ---

class TestEquilibrationSource:
    def test_positive_when_Ti_gt_Te(self) -> None:
        """Q_ei > 0 when Ti > Te (energy flows to electrons)."""
        Te = np.array([1e6])
        Ti = np.array([1e7])
        n_e = np.array([1e25])
        Q = compute_equilibration_source(Te, Ti, n_e, Z)
        assert Q[0] > 0, "Q_ei should be positive when Ti > Te"
        assert np.isfinite(Q[0]), "Q_ei must be finite"

    def test_negative_when_Te_gt_Ti(self) -> None:
        """Q_ei < 0 when Te > Ti (energy flows from electrons)."""
        Te = np.array([1e7])
        Ti = np.array([1e6])
        n_e = np.array([1e25])
        Q = compute_equilibration_source(Te, Ti, n_e, Z)
        assert Q[0] < 0, "Q_ei should be negative when Te > Ti"
        assert np.isfinite(Q[0]), "Q_ei must be finite when Te > Ti"

    def test_zero_at_equilibrium(self) -> None:
        """Q_ei = 0 when Te = Ti."""
        Te = np.array([1e7])
        Ti = np.array([1e7])
        n_e = np.array([1e25])
        Q = compute_equilibration_source(Te, Ti, n_e, Z)
        np.testing.assert_allclose(Q, 0.0, atol=1e-10)
        assert np.isfinite(Q[0]), "Q_ei at equilibrium must be finite (not NaN)"


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
        assert np.all(rho_e_e > 0), "Initialized electron energy must be positive"

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
        assert rho_e_e.shape == GRID_SHAPE, "Initialized e_e must have grid shape"


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
        assert rhs.shape == GRID_SHAPE, "RHS must have same shape as grid"

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
        # RHS should equal Q_ohm = eta * J^2 = 1e-5 * 1e14 = 1e9 W/m^3 for uniform case
        expected_Q_ohm = 1e-5 * 1e14  # 1e9 W/m^3
        np.testing.assert_allclose(
            np.mean(rhs), expected_Q_ohm, rtol=0.01,
            err_msg="RHS magnitude should match Q_ohm = eta*J^2 when Te=Ti (no equilibration)",
        )


# --- Engine integration tests ---

class TestEngineIntegration:
    """Verify 2T flows through the full SimulationEngine."""

    def test_engine_init_creates_e_electron(self) -> None:
        """Engine with two_temperature=True creates e_electron in state."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        preset["grid_shape"] = [8, 1, 16]
        preset["dx"] = 0.02
        preset["sim_time"] = 1e-7
        preset["diagnostics_path"] = ":memory:"
        preset["fluid"] = {"two_temperature": True}

        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)

        assert "e_electron" in engine.state
        assert engine.state["e_electron"].shape == (8, 1, 16)
        assert np.all(engine.state["e_electron"] > 0)

    def test_engine_init_without_2t_no_e_electron(self) -> None:
        """Engine without two_temperature has no e_electron."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        preset["grid_shape"] = [8, 1, 16]
        preset["dx"] = 0.02
        preset["sim_time"] = 1e-7
        preset["diagnostics_path"] = ":memory:"

        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)

        assert "e_electron" not in engine.state
        # Mandatory state fields must still be present
        assert "rho" in engine.state and "pressure" in engine.state, (
            "Core state fields must exist even without 2T"
        )

    def test_e_electron_survives_step(self) -> None:
        """e_electron persists through engine step."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.presets import get_preset

        preset = get_preset("pf1000")
        preset["grid_shape"] = [8, 1, 16]
        preset["dx"] = 0.02
        preset["sim_time"] = 1e-7
        preset["diagnostics_path"] = ":memory:"
        preset["fluid"] = {"two_temperature": True}

        config = SimulationConfig(**preset)
        engine = SimulationEngine(config)
        ee_before = engine.state["e_electron"].copy()

        engine.step()

        assert "e_electron" in engine.state
        # At T=300K with no current, e_electron should barely change
        np.testing.assert_allclose(
            engine.state["e_electron"], ee_before, rtol=0.1,
            err_msg="e_electron changed dramatically in one low-current step",
        )


# --- e_electron advection tests (cylindrical solver) ---

class TestElectronEnergyAdvection:
    """Verify e_electron is advected correctly through the Riemann solver."""

    def _make_solver_and_state(
        self, nr: int = 32, nz: int = 64, dr: float = 0.001, dz: float = 0.001
    ):
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver
        solver = CylindricalMHDSolver(
            nr=nr, nz=nz, dr=dr, dz=dz,
            gamma=5.0 / 3.0, cfl=0.3,
            time_integrator="ssp_rk3",
            conservative_energy=True,
            use_godunov_flux=True,
        )
        rho0 = 1e-4  # kg/m^3 (typical DPF fill)
        p0 = 100.0   # Pa
        state = {
            "rho": np.full((nr, 1, nz), rho0),
            "velocity": np.zeros((3, nr, 1, nz)),
            "pressure": np.full((nr, 1, nz), p0),
            "B": np.zeros((3, nr, 1, nz)),
            "Te": np.full((nr, 1, nz), 1e4),
            "Ti": np.full((nr, 1, nz), 1e4),
            "psi": np.zeros((nr, 1, nz)),
        }
        return solver, state

    def test_uniform_ee_spatially_uniform_after_step(self) -> None:
        """Uniform e_electron should remain spatially uniform after one step.

        The absolute value may change due to 2T source terms (radiation,
        equilibration), but the spatial pattern should remain uniform since
        uniform advection has zero flux divergence.
        """
        solver, state = self._make_solver_and_state()
        ee_val = 1e8  # J/m^3 (high enough that radiation is fractionally small)
        state["e_electron"] = np.full_like(state["rho"], ee_val)
        state["velocity"][2] = 1e4  # 10 km/s axial

        dt = solver._compute_dt(state)
        result = solver.step(state, dt, current=0.0, voltage=0.0)

        assert "e_electron" in result
        ee_out = result["e_electron"][:, 0, :]
        # Check spatial uniformity: std/mean should be small
        rel_std = np.std(ee_out) / np.mean(ee_out)
        assert rel_std < 0.05, (
            f"e_electron is not spatially uniform: rel_std={rel_std:.4f}"
        )

    def test_ee_stays_nonnegative(self) -> None:
        """e_electron must never go negative (positivity preservation)."""
        solver, state = self._make_solver_and_state()
        nr, nz = 32, 64
        # Step function in z: high in left half, zero in right half
        ee = np.zeros((nr, 1, nz))
        ee[:, :, :nz // 2] = 1e6
        state["e_electron"] = ee
        # Strong axial flow to advect the step
        state["velocity"][2] = 5e4

        dt = solver._compute_dt(state)
        for _ in range(5):
            state = solver.step(state, dt, current=0.0, voltage=0.0)
            dt = solver._compute_dt(state)

        assert np.all(state["e_electron"] >= 0.0), (
            f"e_electron went negative: min={np.min(state['e_electron']):.3e}"
        )
        assert np.all(np.isfinite(state["e_electron"])), (
            "e_electron must be finite (no NaN/Inf after advection)"
        )

    def test_step_function_advects_in_z(self) -> None:
        """A step-function profile in z should move with the flow."""
        nr, nz = 16, 64
        dz = 0.001
        solver, state = self._make_solver_and_state(nr=nr, nz=nz, dz=dz)
        # Step function: high for z < 0.5*L, zero above
        ee = np.zeros((nr, 1, nz))
        midz = nz // 2
        ee[:, :, :midz] = 1e6
        state["e_electron"] = ee
        # Positive axial velocity
        vz = 1e4  # m/s
        state["velocity"][2] = vz

        # Advect for several steps
        dt = solver._compute_dt(state)
        n_steps = 10
        for _ in range(n_steps):
            state = solver.step(state, dt, current=0.0, voltage=0.0)
            dt = solver._compute_dt(state)

        ee_final = state["e_electron"][:, 0, :]
        assert np.all(np.isfinite(ee_final)), "e_electron must be finite after advection steps"
        # The centroid of the ee distribution should have shifted right
        z_coords = np.arange(nz) * dz
        ee_profile = np.mean(ee_final, axis=0)  # average over r
        total_ee = np.sum(ee_profile)
        assert total_ee > 0, "Total e_electron must remain positive after advection"
        centroid = np.sum(ee_profile * z_coords) / total_ee
        initial_centroid = np.sum(z_coords[:midz]) / midz
        assert centroid > initial_centroid, (
            f"ee centroid did not move right: initial={initial_centroid:.4f}, "
            f"final={centroid:.4f}"
        )

    def test_total_ee_conserved(self) -> None:
        """Total electron energy should be approximately conserved during advection."""
        nr, nz = 16, 64
        solver, state = self._make_solver_and_state(nr=nr, nz=nz)
        # Gaussian-like profile in z (avoids boundary losses)
        z_idx = np.arange(nz)
        z_center = nz // 2
        sigma = nz / 8
        profile = np.exp(-0.5 * ((z_idx - z_center) / sigma) ** 2)
        ee = np.zeros((nr, 1, nz))
        for i in range(nr):
            ee[i, 0, :] = 1e6 * profile
        state["e_electron"] = ee
        state["velocity"][2] = 5e3  # slow flow to stay in domain

        # Cell volumes: 2*pi*r*dr*dz
        dr = 0.001
        dz = 0.001
        r_centers = (np.arange(nr) + 0.5) * dr
        cell_vol = 2 * np.pi * r_centers[:, None] * dr * dz  # (nr, nz)
        total_before = np.sum(ee[:, 0, :] * cell_vol)

        dt = solver._compute_dt(state)
        for _ in range(5):
            state = solver.step(state, dt, current=0.0, voltage=0.0)
            dt = solver._compute_dt(state)

        total_after = np.sum(state["e_electron"][:, 0, :] * cell_vol)
        # Allow 20% for boundary losses + radiation cooling (2T source step
        # applies bremsstrahlung with gaunt_factor=1.2 inside solver.step)
        rel_change = abs(total_after - total_before) / total_before
        assert rel_change < 0.20, (
            f"Total e_electron changed by {rel_change*100:.1f}% (should be <20%)"
        )
        assert total_after > 0, "Total e_electron must remain positive after advection"


# --- Production 2T validation (PF-1000 pinch conditions) ---

class TestProduction2T:
    """Validate 2T model produces physically correct Te/Ti separation."""

    def test_pinch_conditions_separate_temperatures(self) -> None:
        """At pinch conditions (high J, low density), Te should exceed Ti.

        Physics: Ohmic heating (eta * J^2) heats electrons directly.
        Use low density (1e23) so Q_ohm / e_e is significant, and moderate
        initial T so equilibration doesn't dominate.
        Q_ohm = eta * J^2 = 1e-4 * 1e18 = 1e14 W/m^3
        Initial e_e = 1.5 * 1e23 * 1.38e-23 * 1e5 ~ 2e5 J/m^3
        Over 10 ns: delta_e = 1e14 * 1e-8 = 1e6 >> e_e_init
        """
        Te_K = 1e5   # ~10 eV initial
        Ti_K = 1e5
        n_e = 1e23   # low density for fast heating
        s = _uniform_state(Te_K=Te_K, Ti_K=Ti_K, n_e=n_e)
        s["J_sq"][:] = 1e18  # (10^9)^2 — strong pinch J
        s["eta"][:] = 1e-4   # higher eta at lower T

        Te = s["Te"].copy()
        Ti = s["Ti"].copy()
        rho_e_e = s["rho_e_e"].copy()
        dt = 1e-11  # small for stability

        for _ in range(500):  # 5 ns total
            rho_e_e, Te, Ti = step_electron_energy(
                rho_e_e=rho_e_e, rho=s["rho"], velocity=s["velocity"],
                eta=s["eta"], J_sq=s["J_sq"], Te=Te, Ti=Ti,
                n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=dt,
                Z=Z, gaunt_factor=0.0,  # no radiation to isolate heating
            )

        Te_mean = float(np.mean(Te))
        Ti_mean = float(np.mean(Ti))

        ratio = Te_mean / Ti_mean
        assert ratio > 1.1, (
            f"Te/Ti = {ratio:.4f}, expected > 1.1. "
            f"Te = {Te_mean*k_B/eV:.1f} eV, Ti = {Ti_mean*k_B/eV:.1f} eV"
        )
        assert np.all(Te > 0) and np.all(Ti > 0), "Temperatures must remain positive during pinch heating"

    def test_strong_ohmic_produces_te_gt_2x_ti(self) -> None:
        """Under very strong Ohmic heating at low density, Te >> Ti.

        This is the Must-Have DoD criterion: Te > 2*Ti at pinch stagnation.
        Use n_e = 1e21 (low density) so equilibration time >> heating time.
        tau_eq ~ Te^{3/2} / n_e: at n_e=1e21, tau_eq ~ microseconds.
        Q_ohm = eta * J^2 = 1e-3 * 1e18 = 1e15 W/m^3.
        e_e_init = 1.5 * 1e21 * 1.38e-23 * 1e5 ~ 2e3 J/m^3.
        Over 1 ns: delta_e = 1e6 >> e_e_init.
        """
        Te_K = 1e5   # ~10 eV
        Ti_K = 1e5
        n_e = 1e21   # very low density — equilibration slow
        s = _uniform_state(Te_K=Te_K, Ti_K=Ti_K, n_e=n_e)
        s["J_sq"][:] = 1e18  # (10^9)^2
        s["eta"][:] = 1e-3   # high resistivity

        Te = s["Te"].copy()
        Ti = s["Ti"].copy()
        rho_e_e = s["rho_e_e"].copy()
        dt = 1e-12  # small for stability at extreme Q_ohm

        for _ in range(500):  # 0.5 ns total
            rho_e_e, Te, Ti = step_electron_energy(
                rho_e_e=rho_e_e, rho=s["rho"], velocity=s["velocity"],
                eta=s["eta"], J_sq=s["J_sq"], Te=Te, Ti=Ti,
                n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=dt,
                Z=Z, gaunt_factor=0.0,
            )

        Te_mean = float(np.mean(Te))
        Ti_mean = float(np.mean(Ti))
        ratio = Te_mean / Ti_mean

        assert ratio > 2.0, (
            f"Te/Ti = {ratio:.2f}, need > 2.0. "
            f"Te = {Te_mean*k_B/eV:.1f} eV, Ti = {Ti_mean*k_B/eV:.1f} eV"
        )
        assert np.all(Te > 0) and np.all(Ti > 0), "Temperatures must stay positive"

    def test_radiation_limits_te(self) -> None:
        """Bremsstrahlung radiation should cap Te growth at high temperatures.

        At high Te, radiation loss ~ Te^{1/2} * n_e^2 competes with Ohmic heating.
        The equilibrium Te is where Q_ohm = Q_rad.
        """
        Te_K = 1e7
        Ti_K = 1e7
        n_e = 1e25
        s = _uniform_state(Te_K=Te_K, Ti_K=Ti_K, n_e=n_e)
        s["J_sq"][:] = 1e16
        s["eta"][:] = 1e-5

        # Run WITH radiation
        Te_rad = s["Te"].copy()
        Ti_rad = s["Ti"].copy()
        ee_rad = s["rho_e_e"].copy()

        # Run WITHOUT radiation
        Te_norad = s["Te"].copy()
        Ti_norad = s["Ti"].copy()
        ee_norad = s["rho_e_e"].copy()

        dt = 1e-10
        for _ in range(100):
            ee_rad, Te_rad, Ti_rad = step_electron_energy(
                rho_e_e=ee_rad, rho=s["rho"], velocity=s["velocity"],
                eta=s["eta"], J_sq=s["J_sq"], Te=Te_rad, Ti=Ti_rad,
                n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=dt,
                Z=Z, gaunt_factor=1.2,  # radiation ON
            )
            ee_norad, Te_norad, Ti_norad = step_electron_energy(
                rho_e_e=ee_norad, rho=s["rho"], velocity=s["velocity"],
                eta=s["eta"], J_sq=s["J_sq"], Te=Te_norad, Ti=Ti_norad,
                n_e=s["n_e"], n_i=s["n_i"], dx=DX, dt=dt,
                Z=Z, gaunt_factor=0.0,  # radiation OFF
            )

        assert np.mean(Te_rad) < np.mean(Te_norad), (
            "Radiation should limit Te compared to no-radiation case"
        )
        assert np.all(Te_rad > 0) and np.all(Ti_rad > 0), (
            "Temperatures must remain positive even with radiation enabled"
        )


class TestTwoTempWithConduction:
    """Regression: Braginskii conduction must NOT be disabled in 2T mode.

    Bug history: engine.py line 2125 had `and not _two_t` which silently
    disabled anisotropic conduction when two_temperature was enabled.
    The 2T module (step_electron_energy) handles Ohmic heating, radiation,
    and equilibration — but NOT conduction. Conduction must apply to Te
    through the engine's Braginskii operator.
    """

    def test_conduction_guard_allows_2t(self):
        """Verify the Braginskii conduction guard does not exclude 2T mode."""
        import inspect

        import pytest

        from dpf.engine import SimulationEngine

        source = inspect.getsource(SimulationEngine._apply_collision_radiation)
        for line in source.split("\n"):
            if "enable_anisotropic_conduction" in line:
                assert "not _two_t" not in line, (
                    f"Conduction guard must not exclude 2T mode: {line.strip()}"
                )
                break
        else:
            pytest.skip("enable_anisotropic_conduction guard not found")

    def test_conduction_modifies_te_in_2t_mode(self):
        """With a Te gradient and B-field, conduction should smooth Te in 2T."""
        from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction

        nx, ny, nz = 8, 8, 32
        dx = 0.001
        # Create Te with a sharp gradient along z (B-field direction)
        Te = np.ones((nx, ny, nz)) * 1e6  # 1M Kelvin ~86 eV
        Te[:, :, nz // 4 : 3 * nz // 4] = 1e7  # hot region along z

        # Strong axial B-field (conduction along z)
        B = np.zeros((3, nx, ny, nz))
        B[2, :, :, :] = 1.0  # Bz = 1 Tesla

        ne = np.ones((nx, ny, nz)) * 1e24  # m^-3
        dt = 1e-7  # longer dt for observable effect

        Te_after = anisotropic_thermal_conduction(
            Te, B, ne, dt, dx, dx, dx, Z_eff=1.0,
        )

        # Conduction should smooth the gradient
        Te_range_before = np.max(Te) - np.min(Te)
        Te_range_after = np.max(Te_after) - np.min(Te_after)
        assert Te_range_after < Te_range_before, (
            f"Conduction should reduce Te range: before={Te_range_before:.2e}, "
            f"after={Te_range_after:.2e}"
        )
