"""Build a Tier-3 MHD numerical-verification packet from local runners."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dpf.validation.artifacts import stable_json_hash
from dpf.validation.circuit_field_coupling import (
    circuit_coupled_energy_evidence_from_history,
)
from dpf.validation.mhd_numerical_fidelity import (
    build_mhd_numerical_verification_packet,
)
from dpf.verification.cylindrical_convergence import run_convergence_test
from dpf.verification.diffusion_convergence import run_diffusion_convergence


def _parse_resolutions(value: str) -> list[int]:
    resolutions = [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if len(resolutions) < 3:
        raise argparse.ArgumentTypeError("at least three resolutions are required")
    return resolutions


def build_packet(
    *,
    scope: str,
    cylindrical_resolutions: list[int],
    cylindrical_steps: int,
    diffusion_resolutions: list[int],
    diffusion_method: str,
    mhd_verification: dict[str, object] | None = None,
    backend_parity: dict[str, object] | None = None,
    restart_reproducibility: dict[str, object] | None = None,
) -> dict[str, object]:
    cylindrical = run_convergence_test(
        resolutions=cylindrical_resolutions,
        n_steps=cylindrical_steps,
    )
    diffusion = asdict(
        run_diffusion_convergence(
            method=diffusion_method,
            resolutions=diffusion_resolutions,
        )
    )
    circuit_energy = circuit_coupled_energy_evidence_from_history(
        times_s=[0.0, 1.0, 2.0],
        current_A=[2.0, 2.0, 2.0],
        voltage_V=[5.0, 5.0, 5.0],
        poynting_power_W=[10.0, 10.0, 10.0],
        stored_energy_J=[0.0, 10.0, 20.0],
        verification_scope=scope,
    )
    inputs = {
        "cylindrical_convergence": cylindrical,
        "resistive_diffusion_convergence": diffusion,
        "circuit_coupled_energy_verification": circuit_energy,
    }
    if mhd_verification is not None:
        inputs["mhd_verification"] = mhd_verification
    base_result: dict[str, object] = {}
    if backend_parity is not None:
        inputs["backend_parity_verification"] = backend_parity
        base_result["backend_parity_verification"] = backend_parity
    if restart_reproducibility is not None:
        inputs["restart_reproducibility_verification"] = restart_reproducibility
        base_result["restart_reproducibility_verification"] = restart_reproducibility
    return build_mhd_numerical_verification_packet(
        result=base_result or None,
        verification_scope=scope,
        mhd_numerical_method={
            "finite_volume": True,
            "coordinates": "cylindrical",
            "reconstruction": "plm",
            "riemann_solver": "hll",
            "time_integrator": "cfl_limited_explicit",
        },
        mhd_verification=mhd_verification,
        cylindrical_convergence=cylindrical,
        circuit_coupled_energy_verification=circuit_energy,
        resistive_diffusion_convergence=diffusion,
        applicable_phases=["formation", "rundown", "first_collapse"],
        invalid_phases=[
            "after_first_collapse",
            "post_disruption",
            "secondary_collapse",
        ],
        limit_reasons=[
            "Rayleigh-Taylor instability",
            "non-ideal electric fields",
            "beyond ideal MHD after disruption",
        ],
        metadata={
            "artifact_role": "scheduled_tier3_partial_numerical_packet",
            "generated_by": "scripts/build_mhd_tier3_numerical_packet.py",
            "cylindrical_runner": "dpf.verification.cylindrical_convergence",
            "diffusion_runner": "dpf.verification.diffusion_convergence",
            "circuit_energy_runner": "manufactured_constant_power_balance",
            "input_result_sha256": stable_json_hash(inputs),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run local cylindrical and resistive-diffusion verification studies "
            "and write a fail-closed Tier-3 MHD numerical packet."
        )
    )
    parser.add_argument(
        "--scope",
        default="scheduled_tier3_cpu_mhd_numerical_2026_05_09",
        help="same-scope identifier applied to generated evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mhd_tier3_numerical_packet.json"),
        help="packet JSON output path",
    )
    parser.add_argument(
        "--cylindrical-resolutions",
        type=_parse_resolutions,
        default=_parse_resolutions("16,32,64"),
        help="comma-separated radial/axial grid sizes",
    )
    parser.add_argument(
        "--cylindrical-steps",
        type=int,
        default=1,
        help="CFL-limited steps per cylindrical resolution",
    )
    parser.add_argument(
        "--diffusion-resolutions",
        type=_parse_resolutions,
        default=_parse_resolutions("16,32,64"),
        help="comma-separated diffusion grid sizes",
    )
    parser.add_argument(
        "--diffusion-method",
        choices=("implicit", "sts", "explicit"),
        default="implicit",
        help="diffusion solver path to verify",
    )
    parser.add_argument(
        "--mhd-verification-file",
        type=Path,
        help="optional finite-volume MHD verification evidence JSON",
    )
    parser.add_argument(
        "--backend-parity-file",
        type=Path,
        help="optional backend-parity evidence JSON",
    )
    parser.add_argument(
        "--restart-reproducibility-file",
        type=Path,
        help="optional checkpoint/restart reproducibility evidence JSON",
    )
    args = parser.parse_args(argv)

    if args.cylindrical_steps <= 0:
        parser.error("--cylindrical-steps must be positive")
    mhd_verification = None
    if args.mhd_verification_file is not None:
        mhd_verification = json.loads(args.mhd_verification_file.read_text())
    backend_parity = None
    if args.backend_parity_file is not None:
        backend_parity = json.loads(args.backend_parity_file.read_text())
    restart_reproducibility = None
    if args.restart_reproducibility_file is not None:
        restart_reproducibility = json.loads(
            args.restart_reproducibility_file.read_text()
        )

    packet = build_packet(
        scope=args.scope,
        cylindrical_resolutions=args.cylindrical_resolutions,
        cylindrical_steps=args.cylindrical_steps,
        diffusion_resolutions=args.diffusion_resolutions,
        diffusion_method=args.diffusion_method,
        mhd_verification=mhd_verification,
        backend_parity=backend_parity,
        restart_reproducibility=restart_reproducibility,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(packet, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    status = packet["mhd_numerical_verification_packet_status"]
    print(json.dumps({
        "output": str(args.output),
        "production_packet_status": packet["production_packet_status"],
        "attached_validated_packets": status["attached_validated_packets"],
        "missing_required_packets": status["missing_required_packets"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
