"""Readiness gate for a true 3-D hybrid PIC-fluid DPF core.

This module converts the local arXiv:2604.09032v1 KnowledgeReference source
into explicit first-principles architecture requirements.  It is a gate, not a
model implementation: current runs only pass when they attach reviewed evidence
for every required 3-D hybrid PIC-fluid capability.
"""

from __future__ import annotations

from collections.abc import Mapping

HYBRID_PIC_3D_SOURCE = (
    "KnowledgeReference/"
    "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
)

_ACCEPTED_STATUSES = {"accepted", "validated"}

HYBRID_PIC_3D_CAPABILITIES: tuple[dict[str, object], ...] = (
    {
        "id": "full_maxwell_vacuum_plasma_fields",
        "required_for": "Evolve electromagnetic fields in plasma and vacuum.",
        "source_lines": "150-173, 186-207, 609-618, 1243-1247",
        "current_repo_hooks": [
            "src/dpf/fields/maxwell_3d.py",
            "src/dpf/fluid/cylindrical_mhd.py",
            "src/dpf/validation/circuit_field_coupling.py",
        ],
        "gap": "A 3-D Yee/FDTD Maxwell field component now exists as engineering verification, but accepted DPF runs still lack the full plasma/electron/current closure and validation packet.",
    },
    {
        "id": "kinetic_ion_pic_push_deposition",
        "required_for": "Track ion distributions, beams, and high-energy tails.",
        "source_lines": "210-236, 246-311, 633-639",
        "current_repo_hooks": [
            "src/dpf/experimental/pic/hybrid.py",
            "src/dpf/fields/hybrid_loop.py",
            "src/dpf/fields/pic_coupling.py",
            "src/dpf/kinetic/manager.py",
        ],
        "gap": "HybridPIC can push/deposit 3-D particles and a candidate loop now feeds deposited ion current into the field-current stepper, but this is not yet accepted self-consistent field/current authority for DPF runs.",
    },
    {
        "id": "electron_fluid_generalized_ohm_solver",
        "required_for": "Close electron-fluid current with resistive, pressure-gradient, and Hall terms.",
        "source_lines": "200-209, 325-408, 1107-1185",
        "current_repo_hooks": [
            "src/dpf/fields/ohm_solver.py",
            "src/dpf/metal/mlx_solver.py",
            "src/dpf/diagnostics/regime_classifier.py",
        ],
        "gap": "A 3-D algebraic Ohm-Ampere solver now exists as engineering verification, but it is not yet integrated into the Yee/PIC loop or accepted with electron-energy and same-scope validation packets.",
    },
    {
        "id": "current_predictor_corrector",
        "required_for": "Avoid stale-current ion pushes during long coupled runs.",
        "source_lines": "431-532, 561-562",
        "current_repo_hooks": [
            "src/dpf/fields/predictor_corrector.py",
            "src/dpf/fields/hybrid_stepper.py",
            "src/dpf/fields/hybrid_loop.py",
        ],
        "gap": "A current extrapolation/end-step Ohm correction primitive exists and the candidate stepper/loop can request end-step correction, but no accepted provisional ion-push/rebuild loop is wired into DPF runs.",
    },
    {
        "id": "source_ordered_time_loop",
        "required_for": "Execute the source main loop ordering rather than only isolated components.",
        "source_lines": "246-315, 428-535",
        "current_repo_hooks": [
            "src/dpf/fields/hybrid_loop.py",
            "src/dpf/fields/hybrid_stepper.py",
            "src/dpf/experimental/pic/hybrid.py",
        ],
        "gap": "The candidate loop can now run a source-ordered position/current/field/Eq.7 velocity-update sequence, but accepted Te/Ti rebuild, predictor-corrector particle rebuild, long-run stability, and same-scope validation remain missing.",
    },
    {
        "id": "gauss_law_or_marder_control",
        "required_for": "Control divergence/continuity error introduced by fluid electrons.",
        "source_lines": "410-425, 1067-1073",
        "current_repo_hooks": [
            "src/dpf/fields/marder.py",
            "src/dpf/fields/hybrid_stepper.py",
            "src/dpf/fields/hybrid_loop.py",
            "src/dpf/experimental/pic/hybrid.py",
        ],
        "gap": "A candidate Marder/Gauss-law correction exists and can be applied through the candidate stepper/loop, but no accepted 3-D Maxwell nondominance or divergence-control packet is attached to DPF runs.",
    },
    {
        "id": "plasma_vacuum_conductivity_blending",
        "required_for": "Transition between conducting plasma and vacuum without suppressing full EM modes.",
        "source_lines": "563-606, 1050-1066",
        "current_repo_hooks": [
            "src/dpf/fields/conductivity.py",
            "src/dpf/fluid/cylindrical_mhd.py",
            "src/dpf/metal/mlx_transport.py",
        ],
        "gap": "A source-derived plasma-vacuum conductivity blend exists as engineering verification, but it is not yet integrated into the Maxwell/Ohm loop or accepted by DPF sensitivity evidence.",
    },
    {
        "id": "pml_conductor_particle_boundaries",
        "required_for": "Represent open EM boundaries, conductors, and particle absorption/deletion.",
        "source_lines": "613-619, 625-628",
        "current_repo_hooks": [
            "src/dpf/fields/maxwell_3d.py",
            "src/dpf/fields/particle_boundaries.py",
            "src/dpf/fluid/cylindrical_mhd.py",
            "src/dpf/geometry",
        ],
        "gap": "The Maxwell field slice has engineering conductor/PML semantics and candidate particle absorption can delete particles in conductor/PML regions, but accepted DPF geometry still lacks electrode and same-scope boundary-validation packets.",
    },
    {
        "id": "external_circuit_magnetic_boundary",
        "required_for": "Drive the injection-port magnetic boundary from the external RLC circuit current.",
        "source_lines": "740-792",
        "current_repo_hooks": [
            "src/dpf/fields/circuit_boundary.py",
            "src/dpf/fluid/cylindrical_mhd.py",
            "src/dpf/validation/circuit_field_coupling.py",
        ],
        "gap": "A candidate RLC current step and Cartesian azimuthal-B boundary projection exist, but accepted first-principles runs still lack UDPF magnetic-flux closure, true injection-port geometry, and same-scope circuit validation.",
    },
    {
        "id": "ion_collision_operator",
        "required_for": "Advance kinetic ions with collisional scattering appropriate to DPF plasma.",
        "source_lines": "310-311",
        "current_repo_hooks": [
            "src/dpf/experimental/pic/hybrid.py",
            "src/dpf/fields/hybrid_loop.py",
        ],
        "gap": "Nanbu/Perez-like kernels exist and the candidate 3-D loop now reports collision telemetry, but they are not yet verified in an accepted DPF 3-D hybrid loop.",
    },
    {
        "id": "true_3d_dimensionality",
        "required_for": "Capture m=1 kink, higher azimuthal modes, and fragmentation.",
        "source_lines": "1215-1225, 1274-1278",
        "current_repo_hooks": [
            "src/dpf/fields/hybrid_simulator.py",
            "src/dpf/metal/mlx_engine.py",
            "src/dpf/experimental/pic/hybrid.py",
        ],
        "gap": "A candidate Cartesian 3-D hybrid simulator driver now advances the 3-D loop for multiple steps, but first-principles acceptance remains Python cylindrical until 3-D instability, backend, and same-scope validation packets exist.",
    },
    {
        "id": "separate_electron_energy_closure",
        "required_for": "Make pressure-gradient/Hall neutron-yield predictions quantitative.",
        "source_lines": "1074-1097, 1226-1240, 1267-1278",
        "current_repo_hooks": [
            "src/dpf/fields/electron_energy.py",
            "src/dpf/fluid/cylindrical_mhd.py",
            "src/dpf/fluid/two_temperature.py",
        ],
        "gap": "A candidate 3-D electron-energy source wrapper now exists around the repo two-temperature scaffold, but accepted Te authority still lacks source-closed heat-flux/collisional coupling, loop integration, diagnostics, and same-scope validation.",
    },
    {
        "id": "kinetic_ion_neutron_yield_history",
        "required_for": "Compute time-resolved D-D yield from resolved ion distributions.",
        "source_lines": "952-963, 1083-1089, 1259-1266",
        "current_repo_hooks": [
            "src/dpf/fields/kinetic_yield.py",
            "src/dpf/diagnostics/pic_yield.py",
            "src/dpf/fields/hybrid_loop.py",
            "src/dpf/validation/first_principles_mhd.py",
        ],
        "gap": "Candidate PIC ion yield-history telemetry can now be accumulated from loop particles, but kinetic/hybrid neutron authority and same-scope detector/UQ packets remain blocked.",
    },
    {
        "id": "same_scope_3d_validation_packet",
        "required_for": "Prevent a 2-D/order-of-magnitude paper comparison from becoming simulator validation.",
        "source_lines": "942-951, 974-991, 1215-1225, 1259-1266",
        "current_repo_hooks": [
            "src/dpf/fields/source_geometry.py",
            "docs/VALIDATED_PHYSICS_PIPELINE_PLAN.md",
            "src/dpf/validation/kr_targets.py",
        ],
        "gap": "A typed architecture geometry packet exists as candidate context, but a same-scope 3-D validation packet is still missing.",
        # SS11-3 (audit S10-A3): this capability's identity IS the missing
        # same-scope validation packet.  Its source is the (absent) same-scope
        # 3-D validation packet -- NOT the other-scope LLNL-like hybrid-PIC
        # architecture paper.  Attaching HYBRID_PIC_3D_SOURCE here would put an
        # other-scope architecture source inside a ``same_scope``-named
        # capability subtree, blurring scope context.
        "capability_source": (
            "missing_same_scope_3d_validation_packet_no_accepted_source"
        ),
    },
)

HYBRID_PIC_3D_CAPABILITY_IDS = tuple(
    str(item["id"]) for item in HYBRID_PIC_3D_CAPABILITIES
)


def _accepted_evidence(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status") or "").strip().lower()
    return value.get("passed") is True and status in _ACCEPTED_STATUSES


def hybrid_pic_3d_readiness_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Return fail-closed readiness for the 3-D hybrid PIC-fluid finish line."""
    evidence = result.get("hybrid_pic_3d_evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}

    capability_status: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    satisfied: list[str] = []
    for capability in HYBRID_PIC_3D_CAPABILITIES:
        capability_id = str(capability["id"])
        record = evidence.get(capability_id)
        accepted = _accepted_evidence(record)
        if accepted:
            satisfied.append(capability_id)
            state = "accepted"
        else:
            missing.append(capability_id)
            state = "missing_or_unaccepted"
        # SS11-3 (audit S10-A3): a ``same_scope``-named capability subtree must
        # not carry the LLNL-like hybrid-PIC architecture path as its source.
        capability_source = str(
            capability.get("capability_source") or HYBRID_PIC_3D_SOURCE
        )
        capability_status[capability_id] = {
            "status": state,
            "accepted": accepted,
            "source": capability_source,
            "source_lines": capability["source_lines"],
            "required_for": capability["required_for"],
            "current_repo_hooks": capability["current_repo_hooks"],
            "gap": capability["gap"],
        }

    dimensionality = str(
        result.get("geometry_dimensionality")
        or result.get("dimensionality")
        or ""
    ).strip().lower()
    explicit_3d = dimensionality in {"3d", "three_dimensional", "cartesian_3d"}
    if not explicit_3d:
        missing.append("explicit_3d_geometry")

    accepted = not missing
    return {
        "status": "accepted" if accepted else "blocked",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_status": "source_ingested_target_extraction_needed",
        "can_support_first_principles_acceptance": accepted,
        "geometry_dimensionality": dimensionality or "unset",
        "satisfied_capabilities": sorted(set(satisfied)),
        "missing_capabilities": sorted(set(missing)),
        "capabilities": capability_status,
        "validity_notes": {
            "source_boundary": (
                "arXiv:2604.09032v1 is local source authority for architecture "
                "requirements only. Its sheath benchmark, geometry, cross-section "
                "fit, and neutron yield are not accepted validation targets until "
                "typed same-scope target packets pass review."
            ),
            "current_scope": (
                "The present first-principles path remains a 2-D/cylindrical MHD "
                "engineering candidate until all 3-D hybrid PIC-fluid capabilities "
                "above are implemented, verified, and validated."
            ),
        },
    }
