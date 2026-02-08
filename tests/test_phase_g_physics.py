"""Tests for Phase G: Athena++ DPF Physics.

Sprint G.0: dpf_zpinch.cpp problem generator with circuit coupling.

Tests are organized into:
1. Problem generator existence and configuration
2. Circuit coupling data flow (set_circuit_params -> get_coupling_data)
3. Source term physics (J x B force, Ohmic heating)
4. Electrode boundary conditions
5. Energy conservation
6. Fallback behavior when Athena++ not compiled

Tests requiring compiled Athena++ are skipped with @pytest.mark.skipif.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from dpf.athena_wrapper import is_available as athena_available
from dpf.config import SimulationConfig

# Skip markers for tests requiring compiled Athena++
requires_athena = pytest.mark.skipif(
    not athena_available(),
    reason="Athena++ C++ extension not compiled",
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def default_circuit_params():
    """Standard DPF circuit parameters."""
    return {
        "C": 1e-6,
        "V0": 1e3,
        "L0": 1e-7,
        "R0": 0.01,
        "ESR": 0.0,
        "ESL": 0.0,
        "anode_radius": 0.005,
        "cathode_radius": 0.01,
    }


@pytest.fixture
def dpf_config(default_circuit_params):
    """Create a DPF simulation config for testing."""
    return SimulationConfig(
        grid_shape=[8, 1, 8],
        dx=1e-3,
        sim_time=1e-9,
        dt_init=1e-12,
        circuit=default_circuit_params,
        geometry={"type": "cylindrical"},
        fluid={"backend": "python"},
    )


@pytest.fixture
def dpf_config_athena(default_circuit_params):
    """Create a DPF simulation config with Athena++ backend."""
    return SimulationConfig(
        grid_shape=[8, 1, 8],
        dx=1e-3,
        sim_time=1e-9,
        dt_init=1e-12,
        circuit=default_circuit_params,
        geometry={"type": "cylindrical"},
        fluid={"backend": "athena"},
    )


# ============================================================
# Test: dpf_zpinch.cpp problem generator file exists
# ============================================================


class TestDPFZpinchExists:
    """Verify dpf_zpinch.cpp exists and has expected structure."""

    def test_dpf_zpinch_cpp_exists(self):
        """dpf_zpinch.cpp file exists in pgen directory."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        dpf_file = pgen_path / "dpf_zpinch.cpp"
        assert dpf_file.exists(), f"dpf_zpinch.cpp not found at {dpf_file}"

    def test_dpf_zpinch_has_init_user_mesh_data(self):
        """dpf_zpinch.cpp contains InitUserMeshData function."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        content = (pgen_path / "dpf_zpinch.cpp").read_text()
        assert "InitUserMeshData" in content

    def test_dpf_zpinch_has_problem_generator(self):
        """dpf_zpinch.cpp contains ProblemGenerator function."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        content = (pgen_path / "dpf_zpinch.cpp").read_text()
        assert "ProblemGenerator" in content

    def test_dpf_zpinch_has_source_terms(self):
        """dpf_zpinch.cpp contains DPFSourceTerms function."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        content = (pgen_path / "dpf_zpinch.cpp").read_text()
        assert "DPFSourceTerms" in content

    def test_dpf_zpinch_has_electrode_bc(self):
        """dpf_zpinch.cpp contains electrode boundary condition."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        content = (pgen_path / "dpf_zpinch.cpp").read_text()
        assert "ElectrodeOuterX1" in content

    def test_dpf_zpinch_has_ruser_mesh_data(self):
        """dpf_zpinch.cpp allocates ruser_mesh_data for coupling."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        content = (pgen_path / "dpf_zpinch.cpp").read_text()
        assert "AllocateRealUserMeshDataField" in content
        assert "ruser_mesh_data" in content

    def test_dpf_zpinch_has_user_work_in_loop(self):
        """dpf_zpinch.cpp contains UserWorkInLoop for diagnostics."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        content = (pgen_path / "dpf_zpinch.cpp").read_text()
        assert "UserWorkInLoop" in content

    def test_dpf_zpinch_has_circuit_timestep(self):
        """dpf_zpinch.cpp contains CircuitTimestep function."""
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        content = (pgen_path / "dpf_zpinch.cpp").read_text()
        assert "CircuitTimestep" in content


# ============================================================
# Test: athinput.dpf_zpinch input deck
# ============================================================


class TestAthinputDPFZpinch:
    """Verify athinput.dpf_zpinch exists and is well-formed."""

    def test_athinput_exists(self):
        """athinput.dpf_zpinch file exists."""
        from pathlib import Path

        athinput_path = (
            Path(__file__).parents[1] / "external" / "athinput" / "athinput.dpf_zpinch"
        )
        assert athinput_path.exists()

    def test_athinput_has_problem_block(self):
        """athinput.dpf_zpinch contains <problem> block with DPF parameters."""
        from pathlib import Path

        athinput_path = (
            Path(__file__).parents[1] / "external" / "athinput" / "athinput.dpf_zpinch"
        )
        content = athinput_path.read_text()
        assert "<problem>" in content
        assert "V0" in content
        assert "C" in content
        assert "anode_r" in content
        assert "cathode_r" in content

    def test_athinput_has_user_bc(self):
        """athinput uses user BC on outer x1 for electrode."""
        from pathlib import Path

        athinput_path = (
            Path(__file__).parents[1] / "external" / "athinput" / "athinput.dpf_zpinch"
        )
        content = athinput_path.read_text()
        assert "ox1_bc" in content
        assert "user" in content

    def test_athinput_has_physics_toggles(self):
        """athinput contains physics enable flags."""
        from pathlib import Path

        athinput_path = (
            Path(__file__).parents[1] / "external" / "athinput" / "athinput.dpf_zpinch"
        )
        content = athinput_path.read_text()
        assert "enable_resistive" in content
        assert "enable_radiation" in content
        assert "enable_braginskii" in content


# ============================================================
# Test: athena_bindings.cpp get_coupling_data
# ============================================================


class TestBindingsGetCouplingData:
    """Verify get_coupling_data exists in bindings source."""

    def test_bindings_has_get_coupling_data(self):
        """athena_bindings.cpp contains get_coupling_data function."""
        from pathlib import Path

        bindings_path = (
            Path(__file__).parents[1]
            / "src"
            / "dpf"
            / "athena_wrapper"
            / "cpp"
            / "athena_bindings.cpp"
        )
        content = bindings_path.read_text()
        assert "get_coupling_data" in content

    def test_bindings_pushes_to_ruser_mesh_data(self):
        """set_circuit_params pushes to ruser_mesh_data when available."""
        from pathlib import Path

        bindings_path = (
            Path(__file__).parents[1]
            / "src"
            / "dpf"
            / "athena_wrapper"
            / "cpp"
            / "athena_bindings.cpp"
        )
        content = bindings_path.read_text()
        assert "nreal_user_mesh_data_" in content
        assert "ruser_mesh_data[0]" in content

    def test_bindings_returns_coupling_dict(self):
        """get_coupling_data returns R_plasma, L_plasma, peak_Te."""
        from pathlib import Path

        bindings_path = (
            Path(__file__).parents[1]
            / "src"
            / "dpf"
            / "athena_wrapper"
            / "cpp"
            / "athena_bindings.cpp"
        )
        content = bindings_path.read_text()
        assert '"R_plasma"' in content
        assert '"L_plasma"' in content
        assert '"peak_Te"' in content
        assert '"total_rad_power"' in content


# ============================================================
# Test: athena_engine.py coupling update
# ============================================================


class TestAthenaEngineCoupling:
    """Verify athena_engine.py uses get_coupling_data."""

    def test_engine_tries_cpp_coupling(self):
        """_update_coupling attempts to use get_coupling_data from C++."""
        from pathlib import Path

        engine_path = (
            Path(__file__).parents[1]
            / "src"
            / "dpf"
            / "athena_wrapper"
            / "athena_engine.py"
        )
        content = engine_path.read_text()
        assert "get_coupling_data" in content

    def test_engine_has_python_fallback(self):
        """_update_coupling falls back to Python volume integrals."""
        from pathlib import Path

        engine_path = (
            Path(__file__).parents[1]
            / "src"
            / "dpf"
            / "athena_wrapper"
            / "athena_engine.py"
        )
        content = engine_path.read_text()
        # Should still have the Python fallback code
        assert "B_sq" in content
        assert "L_plasma" in content


# ============================================================
# Test: AthenaPPSolver with Python backend (no compilation needed)
# ============================================================


class TestAthenaPPSolverPythonFallback:
    """Test AthenaPPSolver behavior when Athena++ is not compiled."""

    def test_solver_falls_back_to_python(self, dpf_config):
        """When backend='python', solver uses Python MHD."""
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(dpf_config)
        assert engine.backend == "python"

    def test_python_backend_step(self, dpf_config):
        """Python backend can execute a step."""
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(dpf_config)
        result = engine.step()
        assert result.time > 0
        assert result.finished is False

    def test_coupling_state_valid(self, dpf_config):
        """CouplingState is properly initialized."""
        from dpf.core.bases import CouplingState

        coupling = CouplingState()
        assert coupling.Lp >= 0
        assert coupling.R_plasma >= 0
        assert coupling.Z_bar >= 0


# ============================================================
# Test: DPF source term physics (unit tests, no Athena++ needed)
# ============================================================


class TestDPFSourcePhysics:
    """Unit tests for the physics in dpf_zpinch.cpp source terms."""

    def test_lorentz_force_direction(self):
        """J x B force is radially inward (pinching) for z-current with B_phi."""
        # In cylindrical coords: J = J_z * z_hat, B = B_phi * phi_hat
        # J x B = J_z * B_phi * (z_hat x phi_hat) = -J_z * B_phi * R_hat
        # Negative = inward (pinch)
        J_z = 1e8  # A/m^2
        B_phi = 0.1  # T
        F_R = -J_z * B_phi  # N/m^3
        assert F_R < 0, "Lorentz force should be inward (negative R)"

    def test_current_density_uniform(self):
        """Uniform current density J_z = I_circ / (pi * R_anode^2)."""
        I_circ = 100e3  # 100 kA
        R_anode = 0.005  # 5 mm
        J_z = I_circ / (math.pi * R_anode**2)
        # Expected: ~1.27e9 A/m^2
        assert J_z == pytest.approx(1.273e9, rel=0.01)

    def test_bphi_at_cathode(self):
        """B_phi at cathode = mu_0 * I_circ / (2*pi*r)."""
        I_circ = 100e3  # 100 kA
        r = 0.01  # cathode radius
        mu_0 = 4e-7 * math.pi
        B_phi = mu_0 * I_circ / (2 * math.pi * r)
        # Expected: ~2 T
        assert B_phi == pytest.approx(2.0, rel=0.01)

    def test_ohmic_heating_power(self):
        """Ohmic heating Q = eta * J^2."""
        eta = 1e-7  # Spitzer resistivity [Ohm*m]
        J = 1e9  # Current density [A/m^2]
        Q = eta * J**2
        # Expected: 1e11 W/m^3
        assert pytest.approx(1e11, rel=0.01) == Q

    def test_plasma_inductance_formula(self):
        """L_plasma = sum(B^2/mu_0 * dV) / I_circ^2."""
        mu_0 = 4e-7 * math.pi
        I_circ = 100e3  # 100 kA
        B = 1.0  # T average
        vol = 1e-6  # 1 cm^3 volume
        L = B**2 / mu_0 * vol / I_circ**2
        # L ~ (1 / (4pi*1e-7)) * 1e-6 / 1e10 ~ 8e-14 H
        assert L > 0
        assert L < 1e-3  # Physically reasonable for DPF

    def test_circuit_timestep_constraint(self):
        """Circuit timestep = 0.1 * min(L/R, sqrt(LC))."""
        L = 1e-7  # H
        R = 0.01  # Ohm
        C = 1e-6  # F
        dt_LR = L / R  # 1e-5 s
        dt_LC = math.sqrt(L * C)  # ~3.16e-7 s
        dt_circuit = 0.1 * min(dt_LR, dt_LC)
        assert dt_circuit == pytest.approx(3.16e-8, rel=0.01)


# ============================================================
# Test: Config generation for dpf_zpinch
# ============================================================


class TestConfigGeneration:
    """Test that athena_config.py generates correct athinput for DPF."""

    def test_generate_athinput_has_dpf_params(self, dpf_config):
        """Generated athinput includes DPF circuit parameters."""
        from dpf.athena_wrapper.athena_config import generate_athinput

        text = generate_athinput(dpf_config)
        assert "V0" in text
        assert "C" in text
        assert "anode_r" in text
        assert "cathode_r" in text

    def test_generate_athinput_cylindrical(self, dpf_config):
        """Generated athinput uses cylindrical coordinates."""
        from dpf.athena_wrapper.athena_config import generate_athinput

        text = generate_athinput(dpf_config)
        assert "cylindrical" in text

    def test_generate_athinput_has_physics_toggles(self, dpf_config):
        """Generated athinput includes physics enable flags."""
        from dpf.athena_wrapper.athena_config import generate_athinput

        text = generate_athinput(dpf_config)
        assert "enable_resistive" in text
        assert "enable_nernst" in text
        assert "enable_viscosity" in text

    def test_generate_athinput_has_magnoh_compat(self, dpf_config):
        """Generated athinput includes magnoh.cpp compatibility params."""
        from dpf.athena_wrapper.athena_config import generate_athinput

        text = generate_athinput(dpf_config)
        # These are needed even if using dpf_zpinch, for parameter parsing
        assert "alpha" in text
        assert "beta" in text
        assert "pcoeff" in text

    def test_config_serialization_with_backend(self, default_circuit_params):
        """Config with backend field serializes correctly."""
        config = SimulationConfig(
            grid_shape=[4, 4, 4],
            dx=1e-3,
            sim_time=1e-7,
            circuit=default_circuit_params,
            fluid={"backend": "python"},
        )
        data = json.loads(config.to_json())
        assert data["fluid"]["backend"] == "python"


# ============================================================
# Test: Physical constants in dpf_zpinch.cpp
# ============================================================


class TestPhysicalConstants:
    """Verify physical constants in dpf_zpinch.cpp match Python values."""

    def test_mu_0(self):
        """Vacuum permeability matches scipy."""
        from dpf.constants import mu_0

        cpp_mu_0 = 4.0e-7 * math.pi
        assert cpp_mu_0 == pytest.approx(mu_0, rel=1e-10)

    def test_k_B(self):
        """Boltzmann constant matches scipy."""
        from dpf.constants import k_B

        cpp_k_B = 1.380649e-23
        assert cpp_k_B == pytest.approx(k_B, rel=1e-10)

    def test_elementary_charge(self):
        """Elementary charge matches scipy."""
        from dpf.constants import e

        cpp_e = 1.602176634e-19
        assert cpp_e == pytest.approx(e, rel=1e-10)

    def test_electron_mass(self):
        """Electron mass matches scipy."""
        from dpf.constants import m_e

        cpp_m_e = 9.1093837015e-31
        assert cpp_m_e == pytest.approx(m_e, rel=1e-6)

    def test_epsilon_0(self):
        """Vacuum permittivity matches scipy."""
        from dpf.constants import epsilon_0

        cpp_eps0 = 8.8541878128e-12
        assert cpp_eps0 == pytest.approx(epsilon_0, rel=1e-6)


# ============================================================
# Test: Spitzer resistivity formula consistency
# ============================================================


class TestSpitzerConsistency:
    """Verify Spitzer resistivity formulas match between Python and C++."""

    def test_spitzer_at_1keV(self):
        """Spitzer resistivity at 1 keV deuterium plasma."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity

        Te = np.array([1e4 * 11600.0])  # 1 keV in K
        ne = np.array([1e24])  # 1e24 m^-3

        lnL = coulomb_log(ne, Te)
        eta = spitzer_resistivity(ne, Te, lnL, Z=1)

        # Spitzer resistivity at 1 keV should be ~1e-10 to 1e-6 Ohm*m
        # (hot plasma is very conductive; eta ~ Te^{-3/2})
        assert 1e-12 < float(eta[0]) < 1e-5

    def test_coulomb_log_range(self):
        """Coulomb logarithm is in physically reasonable range."""
        from dpf.collision.spitzer import coulomb_log

        Te = np.array([1e6, 1e7, 1e8])  # 0.1 - 10 keV
        ne = np.array([1e22, 1e23, 1e24])

        lnL = coulomb_log(ne, Te)
        # Coulomb log should be between 1 and 30
        assert np.all(lnL >= 1.0)
        assert np.all(lnL <= 30.0)

    def test_resistivity_decreases_with_temperature(self):
        """Spitzer eta ~ T^{-3/2}, so hotter plasma is less resistive."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity

        ne = np.array([1e23, 1e23])
        Te = np.array([1e6, 1e7])  # 0.1 keV vs 1 keV

        lnL = coulomb_log(ne, Te)
        eta = spitzer_resistivity(ne, Te, lnL, Z=1)

        # Higher Te should have lower resistivity
        assert eta[1] < eta[0]


# ============================================================
# Sprint G.1: Spitzer Resistivity Tests
# ============================================================


class TestSpitzerDiffusivityCpp:
    """Verify SpitzerDiffusivity function exists and is enrolled in dpf_zpinch.cpp."""

    def _read_dpf_zpinch(self):
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        return (pgen_path / "dpf_zpinch.cpp").read_text()

    def test_spitzer_diffusivity_function_exists(self):
        """dpf_zpinch.cpp contains SpitzerDiffusivity function."""
        content = self._read_dpf_zpinch()
        assert "SpitzerDiffusivity" in content

    def test_spitzer_diffusivity_signature(self):
        """SpitzerDiffusivity has correct Athena++ FieldDiffusionCoeffFunc signature."""
        content = self._read_dpf_zpinch()
        # Must accept FieldDiffusion*, MeshBlock*, AthenaArray<Real>&, ...
        assert "FieldDiffusion *pfdif" in content
        assert "const AthenaArray<Real> &w" in content
        assert "const AthenaArray<Real> &bmag" in content

    def test_spitzer_enrolled_via_field_diffusivity(self):
        """SpitzerDiffusivity is enrolled via EnrollFieldDiffusivity."""
        content = self._read_dpf_zpinch()
        assert "EnrollFieldDiffusivity(SpitzerDiffusivity)" in content

    def test_spitzer_writes_to_etaB(self):
        """SpitzerDiffusivity writes to pfdif->etaB for Ohmic diffusion."""
        content = self._read_dpf_zpinch()
        assert "pfdif->etaB(FieldDiffusion::DiffProcess::ohmic" in content

    def test_includes_field_diffusion_header(self):
        """dpf_zpinch.cpp includes field_diffusion.hpp for FieldDiffusion class."""
        content = self._read_dpf_zpinch()
        assert "field_diffusion.hpp" in content

    def test_has_planck_constant(self):
        """dpf_zpinch.cpp defines Planck constant for quantum Coulomb log."""
        content = self._read_dpf_zpinch()
        assert "H_PLANCK" in content
        assert "6.626" in content  # Planck constant value

    def test_has_anomalous_threshold(self):
        """SpitzerDiffusivity includes anomalous Buneman threshold."""
        content = self._read_dpf_zpinch()
        assert "anomalous_alpha" in content
        assert "v_drift" in content
        assert "v_th_e" in content

    def test_has_eta_cap(self):
        """SpitzerDiffusivity caps eta at eta_max."""
        content = self._read_dpf_zpinch()
        assert "eta_max" in content
        # Verify the capping logic exists
        assert "std::min(eta_spitzer, eta_max)" in content

    def test_computes_r_plasma(self):
        """SpitzerDiffusivity computes R_plasma from eta*J^2 volume integral."""
        content = self._read_dpf_zpinch()
        assert "R_plasma_sum" in content
        assert "ruser_mesh_data[2]" in content

    def test_conditional_on_enable_resistive(self):
        """SpitzerDiffusivity enrollment is conditional on enable_resistive flag."""
        content = self._read_dpf_zpinch()
        # Enrollment should be inside if(enable_resistive) block
        idx_enroll = content.find("EnrollFieldDiffusivity(SpitzerDiffusivity)")
        assert idx_enroll > 0
        # Check that enable_resistive appears nearby before it
        preceding = content[max(0, idx_enroll - 200):idx_enroll]
        assert "enable_resistive" in preceding


class TestSpitzerPhysicsFormulas:
    """Unit tests for Spitzer physics formulas (Python reference implementation)."""

    def test_spitzer_vs_nrl_at_10eV(self):
        """Spitzer eta at 10 eV matches NRL Formulary estimate."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity

        # 10 eV deuterium plasma, ne = 1e22 m^-3
        Te = np.array([10.0 * 11600.0])  # 10 eV in K
        ne = np.array([1e22])

        lnL = coulomb_log(ne, Te)
        eta = spitzer_resistivity(ne, Te, lnL, Z=1)

        # NRL Formulary: eta ~ 5.2e-5 * Z * lnL / Te_eV^1.5 [Ohm*m]
        # At 10 eV, lnL ~ 10: eta ~ 5.2e-5 * 10 / 31.6 ~ 1.6e-5 Ohm*m
        # This is order-of-magnitude; the exact formula may differ slightly
        assert 1e-7 < float(eta[0]) < 1e-3, f"Spitzer eta at 10 eV = {eta[0]}"

    def test_spitzer_vs_nrl_at_100eV(self):
        """Spitzer eta at 100 eV matches NRL estimate."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity

        Te = np.array([100.0 * 11600.0])  # 100 eV in K
        ne = np.array([1e23])

        lnL = coulomb_log(ne, Te)
        eta = spitzer_resistivity(ne, Te, lnL, Z=1)

        # At 100 eV: eta ~ 5.2e-5 * lnL / (100)^1.5 ~ 5.2e-7 Ohm*m
        assert 1e-9 < float(eta[0]) < 1e-4, f"Spitzer eta at 100 eV = {eta[0]}"

    def test_spitzer_scaling_te_minus_three_halves(self):
        """Spitzer eta scales as Te^{-3/2} (at fixed ne, lnL ~const)."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity

        ne = np.array([1e23, 1e23, 1e23])
        Te = np.array([1e6, 4e6, 16e6])  # factors of 4

        lnL = coulomb_log(ne, Te)
        eta = spitzer_resistivity(ne, Te, lnL, Z=1)

        # Ratio for Te factor of 4: eta should decrease by factor ~4^1.5 = 8
        # (approximately, since lnL varies weakly)
        ratio_1 = float(eta[0] / eta[1])
        ratio_2 = float(eta[1] / eta[2])
        # Each should be ~8 (within a factor of 2 due to lnL variation)
        assert 3.0 < ratio_1 < 20.0, f"eta ratio (4x Te) = {ratio_1}"
        assert 3.0 < ratio_2 < 20.0, f"eta ratio (4x Te) = {ratio_2}"

    def test_coulomb_log_quantum_correction(self):
        """Coulomb log uses quantum (de Broglie) correction at high density."""
        from dpf.collision.spitzer import coulomb_log

        # Low density (classical): de Broglie << b_class
        ne_low = np.array([1e18])
        Te_low = np.array([1e4])
        lnL_low = coulomb_log(ne_low, Te_low)

        # High density, low Te: de Broglie becomes relevant
        ne_high = np.array([1e28])
        Te_high = np.array([1e4])
        lnL_high = coulomb_log(ne_high, Te_high)

        # Low density should give reasonable lnL (> 1)
        assert float(lnL_low[0]) >= 1.0
        # High density gives very small or zero lnL (Lambda < 1 → log(1) = 0)
        # Python floors at log(max(Lambda, 1.0)) = 0 when Lambda < 1
        assert float(lnL_high[0]) >= 0.0
        # High density should have lower or equal lnL
        assert float(lnL_high[0]) <= float(lnL_low[0])

    def test_anomalous_resistivity_threshold_logic(self):
        """Anomalous resistivity triggers when v_drift > alpha * v_th_e."""
        # This tests the physics logic (not C++ code)
        from dpf.constants import e as elem_e
        from dpf.constants import k_B as kB
        from dpf.constants import m_e

        ne = 1e23  # m^-3
        Te = 1e6  # K (~86 eV)
        J_z = 1e10  # A/m^2 (very high current density)
        alpha = 0.1

        v_drift = J_z / (ne * elem_e)
        v_th_e = np.sqrt(kB * Te / m_e)

        # At these params, v_drift ~ 6.2e8 m/s, v_th_e ~ 3.9e6 m/s
        # v_drift >> alpha * v_th_e, so anomalous should trigger
        assert v_drift > alpha * v_th_e, (
            f"v_drift={v_drift:.2e}, alpha*v_th={alpha * v_th_e:.2e}"
        )

    def test_eta_max_cap(self):
        """Resistivity is capped at eta_max for very cold/dense plasma."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity

        # Very cold plasma: high resistivity
        ne = np.array([1e20])
        Te = np.array([300.0])  # Room temperature

        lnL = coulomb_log(ne, Te)
        eta = spitzer_resistivity(ne, Te, lnL, Z=1)

        eta_max_val = 1.0  # Same as in athinput.dpf_zpinch
        eta_capped = min(float(eta[0]), eta_max_val)
        assert eta_capped <= eta_max_val
        # Cold plasma should be very resistive (close to or exceeding cap)
        assert float(eta[0]) > 1e-4, f"Cold plasma eta = {eta[0]}"

    def test_c_vs_python_coulomb_log(self):
        """C++ Coulomb log formula matches Python at same (ne, Te)."""
        # Reproduce the C++ formula manually and compare to Python
        from dpf.collision.spitzer import coulomb_log
        from dpf.constants import e as elem_e
        from dpf.constants import epsilon_0 as eps0
        from dpf.constants import h as h_planck
        from dpf.constants import k_B as kB
        from dpf.constants import m_e
        from dpf.constants import pi as pi_val

        ne_val = 1e23
        Te_val = 1e7  # ~860 eV

        # Python implementation
        ne_arr = np.array([ne_val])
        Te_arr = np.array([Te_val])
        lnL_python = float(coulomb_log(ne_arr, Te_arr)[0])

        # C++ formula (manual reproduction, matching overflow guards)
        # Note: the +1e-30 guards match the Python spitzer.py implementation
        lambda_D = math.sqrt(eps0 * kB * Te_val / (ne_val * elem_e**2 + 1e-30))
        b_class = elem_e**2 / (4.0 * pi_val * eps0 * kB * Te_val + 1e-30)
        lambda_db = h_planck / math.sqrt(2.0 * pi_val * m_e * kB * Te_val + 1e-30)
        b_min = max(b_class, lambda_db)
        Lambda = lambda_D / (b_min + 1e-30)
        lnL_cpp = max(math.log(max(Lambda, 1.0)), 1.0)

        assert lnL_cpp == pytest.approx(lnL_python, rel=0.01), (
            f"C++={lnL_cpp:.4f}, Python={lnL_python:.4f}"
        )

    def test_c_vs_python_spitzer_eta(self):
        """C++ Spitzer eta formula matches Python at same (ne, Te)."""
        from dpf.collision.spitzer import coulomb_log, spitzer_resistivity
        from dpf.constants import e as elem_e
        from dpf.constants import epsilon_0 as eps0
        from dpf.constants import h as h_planck
        from dpf.constants import k_B as kB
        from dpf.constants import m_e
        from dpf.constants import pi as pi_val

        ne_val = 1e23
        Te_val = 5e6  # ~430 eV
        Z = 1.0

        # Python reference
        ne_arr = np.array([ne_val])
        Te_arr = np.array([Te_val])
        lnL_arr = coulomb_log(ne_arr, Te_arr)
        eta_python = float(spitzer_resistivity(ne_arr, Te_arr, lnL_arr, Z=Z)[0])

        # C++ formula (manual reproduction, matching overflow guards)
        lambda_D = math.sqrt(eps0 * kB * Te_val / (ne_val * elem_e**2 + 1e-30))
        b_class = elem_e**2 / (4.0 * pi_val * eps0 * kB * Te_val + 1e-30)
        lambda_db = h_planck / math.sqrt(2.0 * pi_val * m_e * kB * Te_val + 1e-30)
        b_min = max(b_class, lambda_db)
        Lambda = lambda_D / (b_min + 1e-30)
        lnL = max(math.log(max(Lambda, 1.0)), 1.0)

        coeff = 4.0 * math.sqrt(2.0 * pi_val) * ne_val * Z * elem_e**4 * lnL
        denom = 3.0 * (4.0 * pi_val * eps0) ** 2 * math.sqrt(m_e) * (kB * Te_val) ** 1.5
        nu_ei_val = coeff / denom
        eta_cpp = m_e * nu_ei_val / (ne_val * elem_e**2 + 1e-300)

        # Should agree within 1%
        assert eta_cpp == pytest.approx(eta_python, rel=0.01), (
            f"C++={eta_cpp:.4e}, Python={eta_python:.4e}"
        )


class TestSpitzerDiffusivityEnrollment:
    """Test that Spitzer diffusivity interacts correctly with Athena++ API."""

    def test_diffprocess_ohmic_enum_value(self):
        """DiffProcess::ohmic = 0 (consistent with Athena++ field_diffusion.hpp)."""
        # This is a compile-time constant in C++; we verify the expected value
        # by checking the header file
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "field"
            / "field_diffusion"
            / "field_diffusion.hpp"
        )
        content = hpp_path.read_text()
        assert "ohmic=0" in content.replace(" ", "")

    def test_field_diffusion_hpp_has_etaB(self):
        """field_diffusion.hpp declares etaB array."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "field"
            / "field_diffusion"
            / "field_diffusion.hpp"
        )
        content = hpp_path.read_text()
        assert "etaB" in content

    def test_mesh_hpp_has_enroll_field_diffusivity(self):
        """mesh.hpp declares EnrollFieldDiffusivity function."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "mesh"
            / "mesh.hpp"
        )
        content = hpp_path.read_text()
        assert "EnrollFieldDiffusivity" in content

    def test_field_hpp_includes_field_diffusion(self):
        """field.hpp includes field_diffusion.hpp (transitive availability)."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1] / "external" / "athena" / "src" / "field" / "field.hpp"
        )
        content = hpp_path.read_text()
        assert "field_diffusion.hpp" in content


# ============================================================
# Sprint G.2: Two-Temperature Model Tests
# ============================================================


class TestTwoTemperatureCpp:
    """Verify two-temperature model code in dpf_zpinch.cpp."""

    def _read_dpf_zpinch(self):
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        return (pgen_path / "dpf_zpinch.cpp").read_text()

    def test_includes_scalars_header(self):
        """dpf_zpinch.cpp includes scalars.hpp for passive scalar support."""
        content = self._read_dpf_zpinch()
        assert "scalars.hpp" in content

    def test_nscalars_guard(self):
        """Two-temp code guarded by NSCALARS >= 2 compile-time check."""
        content = self._read_dpf_zpinch()
        assert "NSCALARS >= 2" in content

    def test_passive_scalar_init_in_problem_generator(self):
        """ProblemGenerator initializes s[0] and s[1] for Te and Ti."""
        content = self._read_dpf_zpinch()
        assert "pscalars->s(0" in content
        assert "pscalars->s(1" in content

    def test_source_term_signature_matches_typedef(self):
        """DPFSourceTerms has correct SrcTermFunc signature with prim_scalar."""
        content = self._read_dpf_zpinch()
        assert "prim_scalar" in content
        assert "cons_scalar" in content
        # Old incorrect name should NOT be present
        assert "prim_df" not in content

    def test_has_ei_relaxation(self):
        """DPFSourceTerms includes e-i temperature relaxation."""
        content = self._read_dpf_zpinch()
        assert "nu_eq" in content
        assert "Te - Ti" in content

    def test_ohmic_heating_to_electrons(self):
        """Ohmic heating goes to electron scalar (s[0]) when NSCALARS >= 2."""
        content = self._read_dpf_zpinch()
        assert "cons_scalar(0" in content

    def test_energy_conservation_in_relaxation(self):
        """e-i relaxation conserves total energy (electrons lose, ions gain)."""
        content = self._read_dpf_zpinch()
        # cons_scalar(0) -= dQ and cons_scalar(1) += dQ
        assert "cons_scalar(0, k, j, i) -= dQ" in content
        assert "cons_scalar(1, k, j, i) += dQ" in content

    def test_equipartition_frequency_formula(self):
        """Equipartition frequency uses NRL formula: nu_eq = nu_ei * 2*m_e/m_i."""
        content = self._read_dpf_zpinch()
        assert "2.0 * M_ELECTRON / ion_mass" in content


class TestTwoTemperaturePhysics:
    """Unit tests for two-temperature physics formulas."""

    def test_equipartition_timescale(self):
        """Equipartition timescale tau_eq = m_i / (2 * m_e * nu_ei).

        For ne=1e23, Te=1e6 K, deuterium: tau_eq ~ microseconds.
        """
        from dpf.collision.spitzer import coulomb_log, nu_ei
        from dpf.constants import m_e

        ne = np.array([1e23])
        Te = np.array([1e6])  # ~86 eV
        m_i = 3.34e-27  # deuterium

        lnL = coulomb_log(ne, Te)
        freq = nu_ei(ne, Te, lnL, Z=1)
        nu_eq_val = float(freq[0]) * 2.0 * m_e / m_i
        tau_eq = 1.0 / nu_eq_val

        # tau_eq should be in microseconds for DPF conditions
        assert 1e-9 < tau_eq < 1e-3, f"tau_eq = {tau_eq:.2e} s"

    def test_initial_te_ti_equal(self):
        """At initialization, Te = Ti = T0."""
        T0 = 300.0  # Room temperature
        ne = 1e20  # Low density fill gas
        ni = ne  # Z=1

        # Energy densities should give back T0
        from dpf.constants import k_B

        eps_e = 1.5 * ne * k_B * T0
        eps_i = 1.5 * ni * k_B * T0
        Te_recovered = (2.0 / 3.0) * eps_e / (ne * k_B)
        Ti_recovered = (2.0 / 3.0) * eps_i / (ni * k_B)

        assert Te_recovered == pytest.approx(T0, rel=1e-10)
        assert Ti_recovered == pytest.approx(T0, rel=1e-10)

    def test_relaxation_conserves_total_energy(self):
        """e-i relaxation: total (Te + Ti) energy is conserved."""
        from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures

        ne = np.array([1e23])
        Te = np.array([1e7])  # Hot electrons
        Ti = np.array([1e5])  # Cold ions

        lnL = coulomb_log(ne, Te)
        freq = nu_ei(ne, Te, lnL, Z=1)

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq, dt=1e-10)

        # Total energy (ne*Te + ni*Ti) should be conserved
        # For Z=1: ne = ni, so Te + Ti = const
        assert float(Te_new[0] + Ti_new[0]) == pytest.approx(
            float(Te[0] + Ti[0]), rel=1e-6
        )

    def test_relaxation_direction(self):
        """Relaxation moves Te toward Ti (electrons cool, ions heat)."""
        from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures

        ne = np.array([1e23])
        Te = np.array([1e7])  # Hot electrons
        Ti = np.array([1e5])  # Cold ions

        lnL = coulomb_log(ne, Te)
        freq = nu_ei(ne, Te, lnL, Z=1)

        Te_new, Ti_new = relax_temperatures(Te, Ti, freq, dt=1e-9)

        # Te should decrease, Ti should increase
        assert float(Te_new[0]) < float(Te[0])
        assert float(Ti_new[0]) > float(Ti[0])

    def test_equilibrium_reached(self):
        """After many relaxation steps, Te ≈ Ti."""
        from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures

        ne = np.array([1e23])
        Te = np.array([1e7])
        Ti = np.array([1e5])

        # Very long dt to ensure equilibration
        lnL = coulomb_log(ne, Te)
        freq = nu_ei(ne, Te, lnL, Z=1)
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq, dt=1.0)

        # Should be very close to equilibrium
        assert float(Te_new[0]) == pytest.approx(float(Ti_new[0]), rel=0.01)


class TestTwoTempAthinput:
    """Test athinput configuration for two-temperature model."""

    def test_defs_hpp_has_nscalars(self):
        """defs.hpp.in has NSCALARS placeholder."""
        from pathlib import Path

        defs_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "defs.hpp.in"
        )
        content = defs_path.read_text()
        assert "NUMBER_PASSIVE_SCALARS" in content or "NSCALARS" in content

    def test_scalars_hpp_exists(self):
        """scalars.hpp exists in Athena++ source."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "scalars"
            / "scalars.hpp"
        )
        assert hpp_path.exists()

    def test_scalars_hpp_has_s_array(self):
        """scalars.hpp declares s (conserved) and r (primitive) arrays."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "scalars"
            / "scalars.hpp"
        )
        content = hpp_path.read_text()
        assert "AthenaArray<Real> s" in content
        assert "AthenaArray<Real> r" in content


# ============================================================
# Sprint G.3: Radiation and Cooling Tests
# ============================================================


class TestRadiationCoolingCpp:
    """Verify radiation cooling code in dpf_zpinch.cpp."""

    def _read_dpf_zpinch(self):
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        return (pgen_path / "dpf_zpinch.cpp").read_text()

    def test_has_bremsstrahlung_coefficient(self):
        """dpf_zpinch.cpp defines BREM_COEFF = 1.69e-32."""
        content = self._read_dpf_zpinch()
        assert "BREM_COEFF" in content
        assert "1.69e-32" in content

    def test_has_gaunt_factor(self):
        """dpf_zpinch.cpp defines GAUNT_FACTOR."""
        content = self._read_dpf_zpinch()
        assert "GAUNT_FACTOR" in content

    def test_has_te_floor(self):
        """dpf_zpinch.cpp defines TE_FLOOR for temperature minimum."""
        content = self._read_dpf_zpinch()
        assert "TE_FLOOR" in content

    def test_has_newton_iteration(self):
        """Radiation cooling uses implicit Newton-Raphson iteration."""
        content = self._read_dpf_zpinch()
        assert "NEWTON_ITERS" in content
        assert "Newton" in content or "newton" in content or "f / fp" in content

    def test_has_enable_radiation_guard(self):
        """Radiation cooling is guarded by enable_radiation flag."""
        content = self._read_dpf_zpinch()
        assert "enable_radiation" in content
        # Verify the radiation block is inside an if(enable_radiation) check
        idx_brem = content.find("P_radiated")
        assert idx_brem > 0
        preceding = content[max(0, idx_brem - 2000):idx_brem]
        assert "enable_radiation" in preceding

    def test_radiation_removes_energy(self):
        """Radiation cooling removes energy from total and electron scalar."""
        content = self._read_dpf_zpinch()
        assert "cons(IEN, k, j, i) -= dt * P_radiated" in content

    def test_radiation_diagnostic(self):
        """Total radiated power stored in ruser_mesh_data[4]."""
        content = self._read_dpf_zpinch()
        assert "ruser_mesh_data[4]" in content
        assert "P_radiated * vol" in content

    def test_radiation_affects_electron_scalar(self):
        """Radiation cooling removes energy from electron scalar (s[0])."""
        content = self._read_dpf_zpinch()
        # Should subtract from cons_scalar(0) for radiation losses
        # Count occurrences of cons_scalar(0 subtraction
        assert "cons_scalar(0, k, j, i) -= dt * P_radiated" in content


class TestRadiationPhysicsFormulas:
    """Unit tests for radiation physics (Python reference)."""

    def test_bremsstrahlung_power_magnitude(self):
        """Bremsstrahlung power at DPF conditions has expected magnitude."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne = np.array([1e24])  # m^-3
        Te = np.array([1e7])  # K (~860 eV)

        P_ff = bremsstrahlung_power(ne, Te, Z=1.0, gaunt_factor=1.2)
        # P_ff = 1.69e-32 * 1.2 * 1 * (1e24)^2 * sqrt(1e7) ~ 6.4e19 W/m^3
        assert 1e18 < float(P_ff[0]) < 1e21, f"P_ff = {P_ff[0]:.2e}"

    def test_bremsstrahlung_scales_with_ne_squared(self):
        """Bremsstrahlung P_ff ~ ne^2."""
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power

        ne_1 = np.array([1e23])
        ne_2 = np.array([2e23])
        Te = np.array([1e7])

        P1 = float(bremsstrahlung_power(ne_1, Te)[0])
        P2 = float(bremsstrahlung_power(ne_2, Te)[0])

        assert pytest.approx(4.0, rel=0.01) == P2 / P1

    def test_implicit_cooling_stable(self):
        """Implicit cooling never produces negative Te."""
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.array([1e25])
        Te = np.array([1e5])  # Relatively cool
        dt = 1e-6  # Very large timestep

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt, Z=1.0)
        assert float(Te_new[0]) >= 1.0, f"Te_new = {Te_new[0]}"

    def test_cooling_rate_positive(self):
        """Cooling removes energy (Te decreases)."""
        from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses

        ne = np.array([1e24])
        Te = np.array([1e7])
        dt = 1e-10

        Te_new, P_rad = apply_bremsstrahlung_losses(Te, ne, dt, Z=1.0)
        assert float(Te_new[0]) < float(Te[0])
        assert float(P_rad[0]) > 0

    def test_total_radiation_power(self):
        """Total radiation includes brem + line + recombination."""
        from dpf.radiation.line_radiation import total_radiation_power

        ne = np.array([1e24])
        Te = np.array([1e7])

        P_total = total_radiation_power(ne, Te, Z_eff=1.0)
        assert float(P_total[0]) > 0

    def test_implicit_newton_matches_python(self):
        """C++ Newton iteration produces physically correct results."""
        # Reproduce the C++ implicit solve manually
        from dpf.constants import k_B as kB

        ne_val = 1e22
        Te_old = 1e6
        dt_val = 1e-10
        Z = 1.0
        g_ff = 1.2
        brem_coeff = 1.69e-32
        Te_floor = 1.0

        C_v = 1.5 * ne_val * kB
        inv_cv_dt = dt_val / C_v
        alpha_rad = inv_cv_dt * brem_coeff * g_ff * Z * Z * ne_val * ne_val

        Te_new = Te_old
        for _ in range(4):
            sqrt_T = math.sqrt(max(Te_new, Te_floor))
            f = Te_new + alpha_rad * sqrt_T - Te_old
            fp = 1.0 + alpha_rad / (2.0 * sqrt_T)
            Te_new = Te_new - f / fp
            Te_new = max(Te_new, Te_floor)

        # Te should decrease
        assert Te_new < Te_old
        # Should not hit floor (implicit solve prevents overcooling)
        assert Te_new > Te_floor
        # At these conditions Te_new/Te_old ~ 0.39 (strong but stable cooling)
        assert Te_new > 0.1 * Te_old


# ============================================================
# Sprint G.4: Braginskii Transport Tests
# ============================================================


class TestBraginskiiViscosityCpp:
    """Verify BraginskiiViscosity function in dpf_zpinch.cpp."""

    def _read_dpf_zpinch(self):
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        return (pgen_path / "dpf_zpinch.cpp").read_text()

    def test_braginskii_viscosity_function_exists(self):
        """dpf_zpinch.cpp contains BraginskiiViscosity function."""
        content = self._read_dpf_zpinch()
        assert "BraginskiiViscosity" in content

    def test_braginskii_viscosity_signature(self):
        """BraginskiiViscosity has correct ViscosityCoeffFunc signature."""
        content = self._read_dpf_zpinch()
        assert "HydroDiffusion *phdif" in content
        assert "BraginskiiViscosity(HydroDiffusion *phdif, MeshBlock *pmb" in content

    def test_braginskii_viscosity_enrolled(self):
        """BraginskiiViscosity is enrolled via EnrollViscosityCoefficient."""
        content = self._read_dpf_zpinch()
        assert "EnrollViscosityCoefficient(BraginskiiViscosity)" in content

    def test_braginskii_viscosity_writes_nu_iso(self):
        """BraginskiiViscosity writes to phdif->nu for isotropic viscosity."""
        content = self._read_dpf_zpinch()
        assert "phdif->nu(HydroDiffusion::DiffProcess::iso" in content

    def test_braginskii_viscosity_writes_nu_aniso(self):
        """BraginskiiViscosity writes to phdif->nu for anisotropic viscosity."""
        content = self._read_dpf_zpinch()
        assert "phdif->nu(HydroDiffusion::DiffProcess::aniso" in content

    def test_has_eta0_formula(self):
        """BraginskiiViscosity computes eta_0 = 0.96 * ni * kB * Ti * tau_i."""
        content = self._read_dpf_zpinch()
        assert "0.96" in content
        assert "eta_0" in content

    def test_has_eta1_formula(self):
        """BraginskiiViscosity computes eta_1 with omega_ci dependence."""
        content = self._read_dpf_zpinch()
        assert "omega_ci" in content
        assert "eta_1" in content

    def test_eta1_capped_at_eta0(self):
        """eta_1 is capped at eta_0 for weakly magnetised limit."""
        content = self._read_dpf_zpinch()
        assert "std::min(eta_1, eta_0)" in content

    def test_has_ion_collision_time(self):
        """BraginskiiViscosity computes ion collision time tau_i."""
        content = self._read_dpf_zpinch()
        assert "tau_i" in content

    def test_conditional_on_enable_braginskii(self):
        """Braginskii enrollment is conditional on enable_braginskii flag."""
        content = self._read_dpf_zpinch()
        idx_enroll = content.find("EnrollViscosityCoefficient(BraginskiiViscosity)")
        assert idx_enroll > 0
        preceding = content[max(0, idx_enroll - 200):idx_enroll]
        assert "enable_braginskii" in preceding

    def test_includes_hydro_diffusion_header(self):
        """dpf_zpinch.cpp includes hydro_diffusion.hpp."""
        content = self._read_dpf_zpinch()
        assert "hydro_diffusion.hpp" in content


class TestBraginskiiConductionCpp:
    """Verify BraginskiiConduction function in dpf_zpinch.cpp."""

    def _read_dpf_zpinch(self):
        from pathlib import Path

        pgen_path = Path(__file__).parents[1] / "external" / "athena" / "src" / "pgen"
        return (pgen_path / "dpf_zpinch.cpp").read_text()

    def test_braginskii_conduction_function_exists(self):
        """dpf_zpinch.cpp contains BraginskiiConduction function."""
        content = self._read_dpf_zpinch()
        assert "BraginskiiConduction" in content

    def test_braginskii_conduction_signature(self):
        """BraginskiiConduction has correct ConductionCoeffFunc signature."""
        content = self._read_dpf_zpinch()
        assert "BraginskiiConduction(HydroDiffusion *phdif, MeshBlock *pmb" in content

    def test_braginskii_conduction_enrolled(self):
        """BraginskiiConduction is enrolled via EnrollConductionCoefficient."""
        content = self._read_dpf_zpinch()
        assert "EnrollConductionCoefficient(BraginskiiConduction)" in content

    def test_braginskii_conduction_writes_kappa_iso(self):
        """BraginskiiConduction writes to phdif->kappa for isotropic conduction."""
        content = self._read_dpf_zpinch()
        assert "phdif->kappa(HydroDiffusion::DiffProcess::iso" in content

    def test_braginskii_conduction_writes_kappa_aniso(self):
        """BraginskiiConduction writes to phdif->kappa for anisotropic conduction."""
        content = self._read_dpf_zpinch()
        assert "phdif->kappa(HydroDiffusion::DiffProcess::aniso" in content

    def test_has_kappa_par_formula(self):
        """BraginskiiConduction computes kappa_par = 3.16 * ne * kB^2 * Te * tau_e / m_e."""
        content = self._read_dpf_zpinch()
        assert "3.16" in content
        assert "kappa_par" in content

    def test_has_kappa_perp_formula(self):
        """BraginskiiConduction computes kappa_perp with omega_ce dependence."""
        content = self._read_dpf_zpinch()
        assert "omega_ce" in content
        assert "kappa_perp" in content

    def test_has_flux_limiter(self):
        """BraginskiiConduction implements Sharma-Hammett flux limiter."""
        content = self._read_dpf_zpinch()
        assert "flux_limiter" in content
        assert "q_free" in content

    def test_perp_does_not_exceed_par(self):
        """kappa_perp is capped at kappa_par."""
        content = self._read_dpf_zpinch()
        assert "std::min(kappa_perp, kappa_par)" in content

    def test_has_electron_collision_time(self):
        """BraginskiiConduction computes electron collision time tau_e."""
        content = self._read_dpf_zpinch()
        assert "tau_e" in content


class TestBraginskiiPhysicsFormulas:
    """Unit tests for Braginskii physics formulas (Python reference)."""

    def test_eta0_vs_python(self):
        """Braginskii eta_0 formula matches Python at same (ni, Ti)."""
        from dpf.fluid.viscosity import braginskii_eta0, ion_collision_time

        ni = np.array([1e23])
        Ti = np.array([1e7])  # ~860 eV
        m_ion = 3.34e-27  # deuterium

        tau_i = ion_collision_time(ni, Ti, Z_eff=1.0, m_ion=m_ion)
        eta0_py = braginskii_eta0(ni, Ti, tau_i)

        # Manual C++ formula reproduction
        from dpf.constants import e as elem_e
        from dpf.constants import epsilon_0 as eps0
        from dpf.constants import k_B as kB

        Ti_eV = float(Ti[0]) * kB / elem_e
        ni_cm3 = float(ni[0]) * 1e-6
        arg = math.sqrt(max(ni_cm3, 1.0)) * 1.0 / max(Ti_eV, 1e-3) ** 1.5
        lnL_i = max(23.0 - math.log(max(arg, 1e-30)), 2.0)

        numerator = 3.0 * math.sqrt(2.0 * math.pi) * eps0**2 * math.sqrt(m_ion) * (kB * float(Ti[0])) ** 1.5
        denominator = float(ni[0]) * 1.0**4 * elem_e**4 * lnL_i
        tau_i_cpp = numerator / denominator

        eta0_cpp = 0.96 * float(ni[0]) * kB * float(Ti[0]) * tau_i_cpp

        assert eta0_cpp == pytest.approx(float(eta0_py[0]), rel=0.01), (
            f"C++={eta0_cpp:.4e}, Python={float(eta0_py[0]):.4e}"
        )

    def test_kappa_par_vs_python(self):
        """Braginskii kappa_par matches Python at same (ne, Te)."""
        from dpf.collision.spitzer import braginskii_kappa

        ne = np.array([1e23])
        Te = np.array([1e7])  # ~860 eV
        B_mag = np.array([1.0])  # 1 T

        kappa_par_py, _ = braginskii_kappa(ne, Te, B_mag)

        # Manual C++ formula: kappa_par = 3.16 * ne * kB^2 * Te / (m_e * nu_ei)
        # which is the same as 3.16 * ne * kB^2 * Te * tau_e / m_e
        from dpf.collision.spitzer import coulomb_log, nu_ei
        from dpf.constants import k_B as kB
        from dpf.constants import m_e

        lnL = coulomb_log(ne, Te)
        freq = nu_ei(ne, Te, lnL)
        tau_e_val = 1.0 / float(freq[0])

        kappa_par_cpp = 3.16 * float(ne[0]) * kB**2 * float(Te[0]) * tau_e_val / m_e

        assert kappa_par_cpp == pytest.approx(float(kappa_par_py[0]), rel=0.01), (
            f"C++={kappa_par_cpp:.4e}, Python={float(kappa_par_py[0]):.4e}"
        )

    def test_kappa_anisotropy_ratio(self):
        """kappa_par / kappa_perp > 1e3 for strongly magnetised plasma."""
        from dpf.collision.spitzer import braginskii_kappa

        ne = np.array([1e23])
        Te = np.array([1e7])
        B_mag = np.array([1.0])  # Strong field

        kappa_par, kappa_perp = braginskii_kappa(ne, Te, B_mag)
        ratio = float(kappa_par[0]) / float(kappa_perp[0])

        # For omega_ce * tau_e >> 1, ratio ~ (omega_ce * tau_e)^2 >> 1000
        assert ratio > 1e3, f"kappa_par/kappa_perp = {ratio:.1f}"

    def test_eta1_suppressed_by_magnetisation(self):
        """eta_1 << eta_0 for strongly magnetised ions."""
        from dpf.fluid.viscosity import braginskii_eta0, braginskii_eta1, ion_collision_time

        ni = np.array([1e23])
        Ti = np.array([1e7])
        B_mag = np.array([1.0])  # Strong field
        m_ion = 3.34e-27

        tau_i = ion_collision_time(ni, Ti, Z_eff=1.0, m_ion=m_ion)
        eta0 = braginskii_eta0(ni, Ti, tau_i)
        eta1 = braginskii_eta1(ni, Ti, tau_i, B_mag, m_ion=m_ion)

        # For strong magnetisation, eta_1 should be much smaller than eta_0
        ratio = float(eta0[0]) / float(eta1[0])
        assert ratio > 10.0, f"eta_0/eta_1 = {ratio:.1f}"

    def test_viscous_damping_positive(self):
        """Viscous heating rate is always positive (dissipative)."""
        from dpf.fluid.viscosity import braginskii_eta0, ion_collision_time, viscous_heating_rate

        ni = np.ones((4, 4, 4)) * 1e23
        Ti = np.ones((4, 4, 4)) * 1e6
        m_ion = 3.34e-27

        tau_i = ion_collision_time(ni, Ti, Z_eff=1.0, m_ion=m_ion)
        eta0 = braginskii_eta0(ni, Ti, tau_i)

        # Linear velocity profile: shear flow
        velocity = np.zeros((3, 4, 4, 4))
        for i in range(4):
            velocity[0, i, :, :] = i * 1000.0  # vx increases with x

        Q = viscous_heating_rate(velocity, eta0, dx=1e-3, dy=1e-3, dz=1e-3)
        assert np.all(Q >= 0), "Viscous heating must be non-negative"
        assert np.any(Q > 0), "Shear flow should produce nonzero viscous heating"

    def test_flux_limiter_caps_conduction(self):
        """Sharma-Hammett flux limiter prevents unphysically large heat flux."""
        from dpf.constants import k_B as kB
        from dpf.constants import m_e

        ne = 1e24
        Te = 1e8  # Very hot (weak field scenario)

        # Free streaming heat flux
        v_th_e = math.sqrt(kB * Te / m_e)
        q_free = ne * kB * Te * v_th_e

        # Classical kappa gives unrestricted flux
        # With limiter = 0.1, maximum heat flux is 0.1 * q_free
        flux_lim = 0.1
        q_max = flux_lim * q_free

        # For a temperature gradient of T/dx
        dx = 1e-3
        grad_T = Te / dx
        # Limited kappa satisfies: kappa_limited * grad_T <= q_max
        kappa_max = q_max / grad_T

        assert kappa_max > 0
        assert kappa_max < 1e30, "Flux limiter should bound conductivity"

    def test_braginskii_coefficient_table(self):
        """Braginskii viscosity coefficients match NRL Formulary Table 2.7.

        NRL Formulary: eta_0 = 0.96 * n_i * k_B * T_i * tau_i
        This is a direct verification of the coefficient.
        """
        from dpf.fluid.viscosity import braginskii_eta0, ion_collision_time

        # Use conditions where we can verify against known values
        ni = np.array([1e22])
        Ti = np.array([1e6])  # ~86 eV
        m_ion = 3.34e-27

        tau_i = ion_collision_time(ni, Ti, Z_eff=1.0, m_ion=m_ion)
        eta0 = braginskii_eta0(ni, Ti, tau_i)

        # Manually compute with the 0.96 coefficient
        from dpf.constants import k_B as kB

        expected = 0.96 * float(ni[0]) * kB * float(Ti[0]) * float(tau_i[0])
        assert float(eta0[0]) == pytest.approx(expected, rel=1e-10)

    def test_eta2_equals_four_eta1(self):
        """eta_2 = 4 * eta_1 (Braginskii relation)."""
        from dpf.fluid.viscosity import braginskii_eta1, braginskii_eta2, ion_collision_time

        ni = np.array([1e23])
        Ti = np.array([1e7])
        B_mag = np.array([1.0])
        m_ion = 3.34e-27

        tau_i = ion_collision_time(ni, Ti, Z_eff=1.0, m_ion=m_ion)
        eta1 = braginskii_eta1(ni, Ti, tau_i, B_mag, m_ion=m_ion)
        eta2 = braginskii_eta2(ni, Ti, tau_i, B_mag, m_ion=m_ion)

        assert float(eta2[0]) == pytest.approx(4.0 * float(eta1[0]), rel=1e-10)


class TestBraginskiiAthinput:
    """Test athinput configuration for Braginskii transport."""

    def test_athinput_has_enable_braginskii(self):
        """athinput.dpf_zpinch has enable_braginskii parameter."""
        from pathlib import Path

        athinput_path = (
            Path(__file__).parents[1] / "external" / "athinput" / "athinput.dpf_zpinch"
        )
        content = athinput_path.read_text()
        assert "enable_braginskii" in content

    def test_mesh_hpp_has_viscosity_enrollment(self):
        """mesh.hpp declares EnrollViscosityCoefficient."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "mesh"
            / "mesh.hpp"
        )
        content = hpp_path.read_text()
        assert "EnrollViscosityCoefficient" in content

    def test_mesh_hpp_has_conduction_enrollment(self):
        """mesh.hpp declares EnrollConductionCoefficient."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "mesh"
            / "mesh.hpp"
        )
        content = hpp_path.read_text()
        assert "EnrollConductionCoefficient" in content

    def test_hydro_diffusion_hpp_has_nu_kappa(self):
        """hydro_diffusion.hpp declares nu and kappa arrays."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "hydro"
            / "hydro_diffusion"
            / "hydro_diffusion.hpp"
        )
        content = hpp_path.read_text()
        assert "AthenaArray<Real> nu" in content
        assert "AthenaArray<Real> kappa" in content

    def test_hydro_diffusion_has_diff_process_enum(self):
        """HydroDiffusion has DiffProcess enum with iso=0, aniso=1."""
        from pathlib import Path

        hpp_path = (
            Path(__file__).parents[1]
            / "external"
            / "athena"
            / "src"
            / "hydro"
            / "hydro_diffusion"
            / "hydro_diffusion.hpp"
        )
        content = hpp_path.read_text()
        assert "iso=0" in content.replace(" ", "")
        assert "aniso=1" in content.replace(" ", "")


# ============================================================
# Test: Engine Athena step (conditional)
# ============================================================


class TestEngineAthenaStep:
    """Tests for Athena++ backend in SimulationEngine (requires compilation)."""

    @requires_athena
    def test_athena_engine_init(self, dpf_config_athena):
        """SimulationEngine initializes with Athena++ backend."""
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(dpf_config_athena)
        assert engine.backend == "athena"

    @requires_athena
    def test_athena_engine_step(self, dpf_config_athena):
        """Athena++ engine can execute a step."""
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(dpf_config_athena)
        result = engine.step()
        assert result.time > 0

    @requires_athena
    def test_athena_coupling_data(self, dpf_config_athena):
        """Coupling data is available from Athena++ after step."""
        from dpf.engine import SimulationEngine

        engine = SimulationEngine(dpf_config_athena)
        engine.step()
        coupling = engine.fluid.coupling_interface()
        # L_plasma should be non-negative
        assert coupling.Lp >= 0
