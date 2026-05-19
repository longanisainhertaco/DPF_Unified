import numpy as np

from dpf.constants import e as ELEMENTARY_CHARGE  # noqa: N812
from dpf.constants import m_d as DEUTERON_MASS_KG  # noqa: N812
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.ionization_transport import (
    DeuteriumIonizationTransport,
    apply_ionization_particle_source,
    ionization_transport_candidate_evidence,
    nrl_ground_state_ionization_rate,
    nrl_three_body_recombination_rate,
)
from dpf.fields.maxwell_3d import Maxwell3DGrid


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(3, 3, 3), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def test_nrl_rate_helpers_are_positive_and_temperature_sensitive() -> None:
    cold = nrl_ground_state_ionization_rate(np.array([1.0]), 13.6)[0]
    hot = nrl_ground_state_ionization_rate(np.array([40.0]), 13.6)[0]
    three_body_cold = nrl_three_body_recombination_rate(np.array([1.0]))[0]
    three_body_hot = nrl_three_body_recombination_rate(np.array([40.0]))[0]

    assert hot > cold >= 0.0
    assert three_body_cold > three_body_hot > 0.0


def test_deuterium_ionization_transport_advances_candidate_charge_state() -> None:
    grid = _grid()
    closure = DeuteriumIonizationTransport(grid)
    state = closure.initialize(
        total_deuterium_density_m3=1.0e20,
        ionization_fraction=0.01,
    )

    next_state, telemetry = closure.step(
        state,
        electron_temperature_K=np.full(grid.shape, 4.0e5),
        dt_s=1.0e-10,
    )
    evidence = ionization_transport_candidate_evidence(telemetry)

    assert telemetry.status == "candidate_deuterium_charge_state_transport"
    assert telemetry.can_support_first_principles_acceptance is False
    assert evidence["status"] == "candidate"
    assert evidence["can_support_first_principles_acceptance"] is False
    assert float(np.mean(next_state.mean_charge_state)) > float(
        np.mean(state.mean_charge_state)
    )
    np.testing.assert_allclose(
        next_state.neutral_density_m3 + next_state.ion_density_m3,
        state.neutral_density_m3 + state.ion_density_m3,
    )


def test_deuterium_ionization_transport_bounds_density_changes() -> None:
    grid = _grid()
    closure = DeuteriumIonizationTransport(grid)
    state = closure.initialize(
        total_deuterium_density_m3=1.0e20,
        ionization_fraction=1.0e-4,
    )

    next_state, telemetry = closure.step(
        state,
        electron_temperature_K=np.full(grid.shape, 1.0e7),
        dt_s=1.0,
    )

    assert np.all(next_state.neutral_density_m3 >= 0.0)
    assert np.all(next_state.ion_density_m3 >= 0.0)
    assert np.all(next_state.mean_charge_state <= 1.0)
    assert telemetry.max_limited_density_change_m3 <= 1.0e20


def test_ionization_particle_source_creates_macroparticle_weight() -> None:
    grid = _grid()
    closure = DeuteriumIonizationTransport(grid)
    previous = closure.initialize(
        total_deuterium_density_m3=1.0e20,
        ionization_fraction=0.0,
    )
    next_state = closure.initialize(
        total_deuterium_density_m3=1.0e20,
        ionization_fraction=0.1,
    )
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=1.0e-13,
        use_esirkepov=False,
        use_binary_collisions=False,
    )

    telemetry = apply_ionization_particle_source(
        pic,
        grid,
        previous_state=previous,
        next_state=next_state,
        ion_mass_kg=DEUTERON_MASS_KG,
        ion_charge_C=ELEMENTARY_CHARGE,
    )

    assert telemetry.status == "candidate_ionization_pic_particle_source"
    assert telemetry.macro_particles_created == np.prod(grid.shape)
    assert telemetry.physical_ions_created > 0.0
    assert pic.species[0].n_particles() == np.prod(grid.shape)
    np.testing.assert_allclose(
        np.sum(pic.species[0].weights),
        np.sum(next_state.ion_density_m3 - previous.ion_density_m3)
        * grid.dx
        * grid.dy
        * grid.dz,
    )
