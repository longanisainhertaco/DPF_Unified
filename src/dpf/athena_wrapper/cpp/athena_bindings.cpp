/**
 * @file athena_bindings.cpp
 * @brief pybind11 bindings for Athena++ MHD solver.
 *
 * Exposes the core Athena++ Mesh/MeshBlock functionality to Python,
 * enabling zero-copy NumPy array access to primitive variables and
 * magnetic fields.
 *
 * Build with CMakeLists.txt in this directory:
 *   mkdir build && cd build
 *   cmake .. -DATHENA_ROOT=<path>
 *   make -j8
 *
 * The resulting _athena_core.cpython-*.so module provides:
 *   - init_from_file(path) -> mesh_handle
 *   - init_from_string(text) -> mesh_handle
 *   - execute_cycle(handle, dt) -> None
 *   - get_primitive_data(handle) -> dict of numpy arrays
 *   - get_dt(handle) -> float
 *   - get_time(handle) -> float
 *   - set_circuit_params(handle, current, voltage) -> None
 *   - get_num_meshblocks(handle) -> int
 *   - finalize(handle) -> None
 *
 * @note This file uses Athena++ internal headers. It must be compiled
 *       with the same flags used to build Athena++ (see CMakeLists.txt).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs/outputs.hpp"
#include "outputs/io_wrapper.hpp"
#include "task_list/task_list.hpp"
#include "hydro/hydro.hpp"
#include "field/field.hpp"
#include "coordinates/coordinates.hpp"

namespace py = pybind11;

/**
 * @brief Container holding the Athena++ simulation state.
 *
 * Owns the ParameterInput, Mesh, TaskList, and Outputs objects.
 * Lifetime is managed by Python via shared_ptr.
 */
struct AthenaState {
    std::unique_ptr<ParameterInput> pinput;
    std::unique_ptr<Mesh> pmesh;
    std::unique_ptr<TimeIntegratorTaskList> ptlist;
    std::unique_ptr<Outputs> pouts;

    // DPF circuit coupling parameters (set from Python each step)
    Real circuit_current = 0.0;
    Real circuit_voltage = 0.0;

    ~AthenaState() {
        // Ensure proper cleanup order
        pouts.reset();
        ptlist.reset();
        pmesh.reset();
        pinput.reset();
    }
};

using AthenaHandle = std::shared_ptr<AthenaState>;

/**
 * @brief Initialize Athena++ from an athinput file path.
 *
 * @param filepath Path to the athinput file.
 * @return Handle to the initialized Athena++ state.
 */
AthenaHandle init_from_file(const std::string& filepath) {
    auto state = std::make_shared<AthenaState>();

    // Create ParameterInput from file
    state->pinput = std::make_unique<ParameterInput>();
    IOWrapper infile;
    infile.Open(filepath.c_str(), IOWrapper::FileMode::read);
    state->pinput->LoadFromFile(infile);
    infile.Close();

    // Create Mesh
    state->pmesh = std::make_unique<Mesh>(state->pinput.get());

    // Create task list
    state->ptlist = std::make_unique<TimeIntegratorTaskList>(
        state->pinput.get(), state->pmesh.get());

    // Initialize (runs problem generator)
    state->pmesh->Initialize(0, state->pinput.get());

    // Create outputs
    state->pouts = std::make_unique<Outputs>(
        state->pmesh.get(), state->pinput.get());

    return state;
}

/**
 * @brief Initialize Athena++ from an athinput string.
 *
 * @param athinput_text Complete athinput file content as string.
 * @return Handle to the initialized Athena++ state.
 */
AthenaHandle init_from_string(const std::string& athinput_text) {
    auto state = std::make_shared<AthenaState>();

    // Create ParameterInput from string stream
    state->pinput = std::make_unique<ParameterInput>();
    std::istringstream iss(athinput_text);
    state->pinput->LoadFromStream(iss);

    // Create Mesh
    state->pmesh = std::make_unique<Mesh>(state->pinput.get());

    // Create task list
    state->ptlist = std::make_unique<TimeIntegratorTaskList>(
        state->pinput.get(), state->pmesh.get());

    // Initialize (runs problem generator)
    state->pmesh->Initialize(0, state->pinput.get());

    // Create outputs
    state->pouts = std::make_unique<Outputs>(
        state->pmesh.get(), state->pinput.get());

    return state;
}

/**
 * @brief Execute one integration cycle with given timestep.
 *
 * @param handle Athena++ state handle.
 * @param dt Timestep to use (overrides Athena++ internal dt).
 */
void execute_cycle(AthenaHandle handle, double dt) {
    Mesh* pmesh = handle->pmesh.get();
    TimeIntegratorTaskList* ptlist = handle->ptlist.get();

    // Set timestep
    pmesh->dt = static_cast<Real>(dt);

    // Execute all stages of the time integrator
    for (int stage = 1; stage <= ptlist->nstages; ++stage) {
        ptlist->DoTaskListOneStage(pmesh, stage);
    }

    // User work in loop (diagnostics, source terms)
    pmesh->UserWorkInLoop();

    // Advance time and cycle
    pmesh->ncycle++;
    pmesh->time += pmesh->dt;
    pmesh->NewTimeStep();
}

/**
 * @brief Get primitive variable data from all mesh blocks.
 *
 * Returns a Python dict with:
 *   "rho":      numpy array of density
 *   "velocity": numpy array of velocities (3 components)
 *   "pressure": numpy array of pressure
 *   "B":        numpy array of cell-centered B-field (3 components)
 *
 * For single-block runs, shapes are (nk, nj, ni).
 * For multi-block, data is concatenated.
 *
 * @param handle Athena++ state handle.
 * @return Python dict of numpy arrays.
 */
py::dict get_primitive_data(AthenaHandle handle) {
    Mesh* pmesh = handle->pmesh.get();
    py::dict result;

    // Count total cells across all blocks
    int total_cells = 0;
    int ni = 0, nj = 0, nk = 0;
    int nblocks = pmesh->nblocal;

    // Get dimensions from first block
    MeshBlock* pmb = pmesh->my_blocks(0);
    ni = pmb->ie - pmb->is + 1;
    nj = pmb->je - pmb->js + 1;
    nk = pmb->ke - pmb->ks + 1;
    total_cells = nblocks * nk * nj * ni;

    // Allocate output arrays
    // Shape: (nblocks, nk, nj, ni) — Python will reshape
    auto rho = py::array_t<double>({nblocks, nk, nj, ni});
    auto vx  = py::array_t<double>({nblocks, nk, nj, ni});
    auto vy  = py::array_t<double>({nblocks, nk, nj, ni});
    auto vz  = py::array_t<double>({nblocks, nk, nj, ni});
    auto prs = py::array_t<double>({nblocks, nk, nj, ni});
    auto Bx  = py::array_t<double>({nblocks, nk, nj, ni});
    auto By  = py::array_t<double>({nblocks, nk, nj, ni});
    auto Bz  = py::array_t<double>({nblocks, nk, nj, ni});

    auto r_rho = rho.mutable_unchecked<4>();
    auto r_vx  = vx.mutable_unchecked<4>();
    auto r_vy  = vy.mutable_unchecked<4>();
    auto r_vz  = vz.mutable_unchecked<4>();
    auto r_prs = prs.mutable_unchecked<4>();
    auto r_Bx  = Bx.mutable_unchecked<4>();
    auto r_By  = By.mutable_unchecked<4>();
    auto r_Bz  = Bz.mutable_unchecked<4>();

    // Copy data from each mesh block
    for (int b = 0; b < nblocks; ++b) {
        MeshBlock* pmb = pmesh->my_blocks(b);

        // Ensure cell-centered B is up to date
#if MAGNETIC_FIELDS_ENABLED
        pmb->pfield->CalculateCellCenteredField(
            pmb->pfield->b, pmb->pfield->bcc, pmb->pcoord,
            pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);
#endif

        for (int k = pmb->ks; k <= pmb->ke; ++k) {
            for (int j = pmb->js; j <= pmb->je; ++j) {
                for (int i = pmb->is; i <= pmb->ie; ++i) {
                    int kk = k - pmb->ks;
                    int jj = j - pmb->js;
                    int ii = i - pmb->is;

                    // Primitive variables: w(IDN,k,j,i), w(IVX,k,j,i), etc.
                    r_rho(b, kk, jj, ii) = static_cast<double>(pmb->phydro->w(IDN, k, j, i));
                    r_vx(b, kk, jj, ii)  = static_cast<double>(pmb->phydro->w(IVX, k, j, i));
                    r_vy(b, kk, jj, ii)  = static_cast<double>(pmb->phydro->w(IVY, k, j, i));
                    r_vz(b, kk, jj, ii)  = static_cast<double>(pmb->phydro->w(IVZ, k, j, i));
                    r_prs(b, kk, jj, ii) = static_cast<double>(pmb->phydro->w(IPR, k, j, i));

#if MAGNETIC_FIELDS_ENABLED
                    r_Bx(b, kk, jj, ii) = static_cast<double>(pmb->pfield->bcc(IB1, k, j, i));
                    r_By(b, kk, jj, ii) = static_cast<double>(pmb->pfield->bcc(IB2, k, j, i));
                    r_Bz(b, kk, jj, ii) = static_cast<double>(pmb->pfield->bcc(IB3, k, j, i));
#else
                    r_Bx(b, kk, jj, ii) = 0.0;
                    r_By(b, kk, jj, ii) = 0.0;
                    r_Bz(b, kk, jj, ii) = 0.0;
#endif
                }
            }
        }
    }

    // Pack velocity and B into (3, nblocks, nk, nj, ni) arrays
    auto velocity = py::array_t<double>({3, nblocks, nk, nj, ni});
    auto B_field  = py::array_t<double>({3, nblocks, nk, nj, ni});

    auto r_vel = velocity.mutable_unchecked<5>();
    auto r_B   = B_field.mutable_unchecked<5>();

    for (int b = 0; b < nblocks; ++b) {
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    r_vel(0, b, k, j, i) = r_vx(b, k, j, i);
                    r_vel(1, b, k, j, i) = r_vy(b, k, j, i);
                    r_vel(2, b, k, j, i) = r_vz(b, k, j, i);
                    r_B(0, b, k, j, i)   = r_Bx(b, k, j, i);
                    r_B(1, b, k, j, i)   = r_By(b, k, j, i);
                    r_B(2, b, k, j, i)   = r_Bz(b, k, j, i);
                }
            }
        }
    }

    result["rho"]      = rho;
    result["velocity"] = velocity;
    result["pressure"] = prs;
    result["B"]        = B_field;

    return result;
}

/**
 * @brief Get current CFL-limited timestep.
 */
double get_dt(AthenaHandle handle) {
    return static_cast<double>(handle->pmesh->dt);
}

/**
 * @brief Get current simulation time.
 */
double get_time(AthenaHandle handle) {
    return static_cast<double>(handle->pmesh->time);
}

/**
 * @brief Set circuit parameters for DPF coupling.
 *
 * These values are accessible from the problem generator via
 * ruser_mesh_data.
 */
void set_circuit_params(AthenaHandle handle, double current, double voltage) {
    handle->circuit_current = static_cast<Real>(current);
    handle->circuit_voltage = static_cast<Real>(voltage);

    // Push circuit parameters into ruser_mesh_data if the problem generator
    // allocated it (dpf_zpinch.cpp does; magnoh.cpp does not).
    Mesh* pmesh = handle->pmesh.get();
    if (pmesh != nullptr && pmesh->nreal_user_mesh_data_ >= 2) {
        pmesh->ruser_mesh_data[0](0) = static_cast<Real>(current);
        pmesh->ruser_mesh_data[1](0) = static_cast<Real>(voltage);
    }
}

/**
 * @brief Get coupling data computed by dpf_zpinch.cpp UserWorkInLoop.
 *
 * Returns a Python dict with diagnostics stored in ruser_mesh_data:
 *   R_plasma:       Plasma resistance [Ohm]
 *   L_plasma:       Plasma inductance [H]
 *   total_rad_power: Total radiated power [W]
 *   peak_Te:        Peak electron temperature [K]
 *
 * Returns empty values if ruser_mesh_data is not allocated (e.g. magnoh.cpp).
 */
py::dict get_coupling_data(AthenaHandle handle) {
    py::dict result;
    Mesh* pmesh = handle->pmesh.get();

    if (pmesh != nullptr && pmesh->nreal_user_mesh_data_ >= 6) {
        result["R_plasma"]        = static_cast<double>(pmesh->ruser_mesh_data[2](0));
        result["L_plasma"]        = static_cast<double>(pmesh->ruser_mesh_data[3](0));
        result["total_rad_power"] = static_cast<double>(pmesh->ruser_mesh_data[4](0));
        result["peak_Te"]         = static_cast<double>(pmesh->ruser_mesh_data[5](0));
    } else {
        result["R_plasma"]        = 0.0;
        result["L_plasma"]        = 0.0;
        result["total_rad_power"] = 0.0;
        result["peak_Te"]         = 0.0;
    }

    return result;
}

/**
 * @brief Get number of local mesh blocks.
 */
int get_num_meshblocks(AthenaHandle handle) {
    return handle->pmesh->nblocal;
}

/**
 * @brief Get current cycle number.
 */
int get_cycle(AthenaHandle handle) {
    return handle->pmesh->ncycle;
}

/**
 * @brief Clean up Athena++ state.
 */
void finalize(AthenaHandle handle) {
    handle->pouts.reset();
    handle->ptlist.reset();
    handle->pmesh.reset();
    handle->pinput.reset();
}

// ============================================================
// Python module definition
// ============================================================

PYBIND11_MODULE(_athena_core, m) {
    m.doc() = "Athena++ MHD solver bindings for DPF Unified";

    // Opaque handle type
    py::class_<AthenaState, std::shared_ptr<AthenaState>>(m, "AthenaState")
        .def_readonly("circuit_current", &AthenaState::circuit_current)
        .def_readonly("circuit_voltage", &AthenaState::circuit_voltage);

    // Initialization
    m.def("init_from_file", &init_from_file,
          py::arg("filepath"),
          "Initialize Athena++ from an athinput file path.");

    m.def("init_from_string", &init_from_string,
          py::arg("athinput_text"),
          "Initialize Athena++ from an athinput string.");

    // Simulation control
    m.def("execute_cycle", &execute_cycle,
          py::arg("handle"), py::arg("dt"),
          "Execute one integration cycle with given timestep.");

    // Data access
    m.def("get_primitive_data", &get_primitive_data,
          py::arg("handle"),
          "Get primitive variable arrays from all mesh blocks.");

    m.def("get_dt", &get_dt,
          py::arg("handle"),
          "Get current CFL-limited timestep.");

    m.def("get_time", &get_time,
          py::arg("handle"),
          "Get current simulation time.");

    m.def("get_cycle", &get_cycle,
          py::arg("handle"),
          "Get current cycle number.");

    // Circuit coupling
    m.def("set_circuit_params", &set_circuit_params,
          py::arg("handle"), py::arg("current"), py::arg("voltage"),
          "Set circuit current and voltage for DPF coupling.");

    m.def("get_coupling_data", &get_coupling_data,
          py::arg("handle"),
          "Get coupling diagnostics (R_plasma, L_plasma, peak_Te) from ruser_mesh_data.");

    // Mesh info
    m.def("get_num_meshblocks", &get_num_meshblocks,
          py::arg("handle"),
          "Get number of local mesh blocks.");

    // Cleanup
    m.def("finalize", &finalize,
          py::arg("handle"),
          "Clean up Athena++ state and free memory.");
}
