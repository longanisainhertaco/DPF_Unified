"""KnowledgeReference corpus inventory and review-status helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

_DPF_FILENAME_MARKERS = (
    "dense-plasma-focus",
    "plasma-focus",
    "pf1000",
    "pf-1000",
    "mjolnir",
    "dpf",
    "focus",
)

_DPF_CONTENT_MARKERS = (
    "dense plasma focus",
    "plasma focus",
    "pf-1000",
    "pf1000",
    "pf 1000",
    "mjolnir",
    "mather-type",
    "filippov",
)

_TARGET_CATEGORY_MARKERS = {
    "circuit_waveform": (
        "current waveform",
        "current trace",
        "current dip",
        "rogowski",
        "peak current",
    ),
    "phase_timing": (
        "axial phase",
        "radial phase",
        "pinch duration",
        "pinch time",
        "rundown",
        "run-down",
    ),
    "spatial_density": (
        "density profile",
        "electron density",
        "interferogram",
        "interferometer",
        "interferometry",
    ),
    "spatial_magnetic_or_em": (
        "magnetic field",
        "b-field",
        "probe",
        "electric field",
        "poloidal",
    ),
    "spatial_temperature": (
        "electron temperature",
        "ion temperature",
        "pinch temperature",
        "temperature",
        "kev",
    ),
    "neutron_validation": (
        "neutron yield",
        "neutron",
        "anisotropy",
        "spectrum",
        "activation",
    ),
    "uncertainty": (
        "uncertainty",
        "error bar",
        "standard deviation",
        "shot-to-shot",
        "shot to shot",
    ),
}

_EXPLICIT_REVIEW_DECISIONS = (
    {
        "source": "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md",
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-"
            "z-pinch-8.md"
        ),
        "reason": (
            "Same Schmidt/Tang/Welch fully kinetic DPF paper as the accepted "
            "Phys. Rev. Lett. manuscript; keep one coded target and count this "
            "local copy as review-closed by duplicate decision."
        ),
    },
    {
        "source": "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch-9.md",
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-"
            "z-pinch-8.md"
        ),
        "reason": (
            "Duplicate accepted-manuscript local copy of the Schmidt/Tang/Welch "
            "fully kinetic DPF paper; target extraction is represented by the "
            "canonical -8 source."
        ),
    },
    {
        "source": "KnowledgeReference/modification-and-numerical-modelling-of-dense-plasma-focus.md",
        "status": "insufficient_extractable_validation_data",
        "canonical_source": "",
        "reason": (
            "The local markdown extraction contains useful abstract, "
            "introduction, table-caption, and scaling context, but the "
            "Experimental System, Numerical Modelling, Results and Discussion, "
            "and Conclusion sections are empty page stubs in this text. Result "
            "values are present only as figure-list captions, so this file is "
            "not a reliable line-referenced validation target source without "
            "re-ingesting the original thesis PDF."
        ),
    },
    {
        "source": "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md",
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
            "dense-plasma-focus-z-pinch-5.md"
        ),
        "reason": (
            "Same Goyon 2025 MA-class MJOLNIR neutron-generation paper as the "
            "canonical -5 source already represented by coded timing, "
            "temperature, and detector-response targets."
        ),
    },
    {
        "source": (
            "KnowledgeReference/paper-open-access-dense-plasma-focus-from-"
            "alternative-fusion-source-to-versatile-high-energy.md"
        ),
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/paper-open-access-dense-plasma-focus-from-"
            "alternative-fusion-source-to-versatile-high-energy-4.md"
        ),
        "reason": (
            "Header/PDF-name variant of the Rawat 2015 DPF review. The "
            "canonical -4 local source carries the coded generic operating-"
            "envelope target."
        ),
    },
    {
        "source": "KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md",
        "status": "duplicate",
        "canonical_source": "KnowledgeReference/goyon-2022-mjolnir-high-low.md",
        "reason": (
            "LLNL report/extraction variant of the same Schmidt/Goyon 2022 "
            "MJOLNIR high/low-performing discharge paper already represented "
            "by the coded parasitic-current target from the canonical local "
            "Phys. Plasmas manuscript extraction."
        ),
    },
    {
        "source": (
            "KnowledgeReference/measurement-of-electric-flux-emission-a-new-"
            "diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v-4.md"
        ),
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/measurement-of-electric-flux-emission-a-new-"
            "diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v.md"
        ),
        "reason": (
            "Header/PDF-name variant of the same Blagoev/Yordanov/Auluck "
            "2025 electric-flux DPF diagnostic paper. The unsuffixed local "
            "source has the cleaner title and carries the coded electric-flux "
            "formation-symmetry diagnostic target."
        ),
    },
    {
        "source": "KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus-5.md",
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md"
        ),
        "reason": (
            "Header/PDF-name variant of the same Auluck 2024 Physics of "
            "Plasmas poloidal magnetic-field letter. The unsuffixed local "
            "source carries the coded poloidal-field/dynamo target."
        ),
    },
    {
        "source": (
            "KnowledgeReference/double-3-mj-dense-plasma-focus-for-"
            "thermonuclear-drive-inertial-confinement-fusion-5.md"
        ),
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/2025-double-3mj-dense-plasma-focus-"
            "thermonuclear-icf.md"
        ),
        "reason": (
            "Header/PDF-name variant of the same Kiai 2025 Scientific "
            "Reports double 3 MJ DPF/ICF theoretical concept paper. The "
            "2025 local extraction is richer and carries the coded "
            "double-DPF/ICF concept and validation-roadmap target."
        ),
    },
    {
        "source": (
            "KnowledgeReference/double-3-mj-dense-plasma-focus-for-"
            "thermonuclear-drive-inertial-confinement-fusion.md"
        ),
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/2025-double-3mj-dense-plasma-focus-"
            "thermonuclear-icf.md"
        ),
        "reason": (
            "Duplicate extraction of the same Kiai 2025 Scientific Reports "
            "double 3 MJ DPF/ICF theoretical concept paper. The 2025 local "
            "source carries the coded concept target and the duplicated file "
            "does not add independent validation data."
        ),
    },
    {
        "source": "KnowledgeReference/auluck-2022-dpf-theory-part1.md",
        "status": "insufficient_extractable_validation_data",
        "canonical_source": "",
        "reason": (
            "The local markdown extraction for this 74-page Auluck theory "
            "source contains only the final references page. It lists related "
            "papers on current sheath structure, radiation transport, neutron "
            "signals, density diagnostics, and poloidal magnetic flux, but it "
            "does not expose the body text, equations, figures, or tables "
            "needed for a line-referenced KR target."
        ),
    },
    {
        "source": "KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md",
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus.md"
        ),
        "reason": (
            "Header/PDF-name variant of the same Chinese FOI two-dimensional "
            "DPF simulation source. The unsuffixed local extraction has the "
            "cleaner title and carries the coded FOI 2D MHD parameter-sweep "
            "target."
        ),
    },
    {
        "source": (
            "KnowledgeReference/building-a-sci-fi-themed-dense-plasma-focus-"
            "simulation-front-end-in-unity.md"
        ),
        "status": "non_scientific_frontend_guide",
        "canonical_source": "",
        "reason": (
            "This local markdown is a Unity front-end and visualization "
            "tutorial. It discusses URP, VFX Graph, UI controls, raymarching, "
            "data ingestion, and WebSocket display plumbing, but it does not "
            "provide verified DPF physics equations, experimental validation "
            "targets, or diagnostic measurements for the KR-only scientific "
            "model review."
        ),
    },
    {
        "source": (
            "KnowledgeReference/2023-correction-to-focus-fusion-overview-of-"
            "progress-towards-p-b11-fusion-with-the.md"
        ),
        "status": "correction_only",
        "canonical_source": (
            "KnowledgeReference/focus-fusion-overview-of-progress-towards-"
            "p-b11-fusion-with-the-dense-plasma-focus.md"
        ),
        "reason": (
            "One-page correction notice for the Lerner 2023 Focus Fusion "
            "paper. It corrects the abstract's highest n-tau-T product to "
            "3.4e20 keV-s/m3; that corrected value is already represented by "
            "the canonical coded FF-1 plasmoid/p-B11 context target, and the "
            "correction notice does not add independent validation data."
        ),
    },
    {
        "source": (
            "KnowledgeReference/dimensions-and-lifetime-of-the-plasma-focus-"
            "pinch-plasma-science-ieee-transactions-on-2.md"
        ),
        "status": "insufficient_extractable_validation_data",
        "canonical_source": "",
        "reason": (
            "The local markdown extraction contains only the title/source "
            "header and page stub for DimLifePF96. No body text, equations, "
            "figures, tables, diagnostics, or numerical pinch dimensions/"
            "lifetimes are available for KR-only target extraction."
        ),
    },
    {
        "source": (
            "KnowledgeReference/dpf-bi-rrt-an-improved-path-planning-"
            "algorithm-for-complex-3d-environments-with-adaptive-sampling.md"
        ),
        "status": "non_dpf_acronym_collision",
        "canonical_source": "",
        "reason": (
            "In this IEEE Access path-planning paper, DPF means Dual "
            "Potential Field in the DPF-Bi-RRT* algorithm for autonomous "
            "aerial vehicle navigation. The source is unrelated to Dense "
            "Plasma Focus physics and provides no scientific validation data "
            "for this project."
        ),
    },
    {
        "source": (
            "KnowledgeReference/optimization-and-development-of-a-dense-"
            "plasma-focus-simulator.md"
        ),
        "status": "non_scientific_software_performance_summary",
        "canonical_source": "",
        "reason": (
            "Two-page software/HPC architecture and performance summary for "
            "a DPF simulator. It lists GUI, solver, ML-control, visualization, "
            "Metal GPU acceleration, CPU utilization, memory, and FPS claims, "
            "but provides no verified DPF physics equations, experimental "
            "diagnostics, calibration data, or validation targets."
        ),
    },
    {
        "source": "KnowledgeReference/deuterium-hybrid-x-pinch-driven-by-small-dense-pla.md",
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/deuterium-hybrid-x-pinch-driven-by-small-"
            "dense-plasma-focus-2.md"
        ),
        "reason": (
            "Title/truncated-filename variant of the deuterium hybrid X-pinch "
            "paper already represented by the coded PFZ-200 hybrid X-pinch "
            "particle-source target from the cleaner -2 extraction."
        ),
    },
    {
        "source": (
            "KnowledgeReference/experimental-results-and-analysis-of-plasma-"
            "dynamics-and-radiation-output-of-the-100-kv-dense.md"
        ),
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-"
            "radiation-output-of-a-100-kv-plasma-focus-device.md"
        ),
        "reason": (
            "Header/PDF-name variant of the Damideh 2025 FAETON-I high-voltage "
            "plasma-focus source already represented by the coded FAETON-I "
            "target."
        ),
    },
    {
        "source": (
            "KnowledgeReference/experimental-results-and-analysis-of-plasma-"
            "dynamics-and-radiation-output-of-the-100-kv-dense-5.md"
        ),
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-"
            "radiation-output-of-a-100-kv-plasma-focus-device.md"
        ),
        "reason": (
            "Duplicate Damideh 2025 FAETON-I extraction. The canonical local "
            "source carries the coded high-voltage DPF target and this copy "
            "does not add independent validation observables."
        ),
    },
    {
        "source": (
            "KnowledgeReference/2024-a-hybrid-kinetic-simulation-tool-for-"
            "non-thermal-warm-x-ray-z-pinch-sources-wit.md"
        ),
        "status": "non_dpf_reference_only",
        "canonical_source": "",
        "reason": (
            "Warm x-ray Z-pinch hybrid-kinetic source paper. The only strong "
            "DPF hit is a dense-plasma-focus reference in the bibliography, "
            "not a DPF experiment, equation set, or validation target."
        ),
    },
    {
        "source": (
            "KnowledgeReference/a-comprehensive-analytical-model-of-the-"
            "dynamic-z-pinch.md"
        ),
        "status": "non_dpf_general_z_pinch_model",
        "canonical_source": "",
        "reason": (
            "General dynamic Z-pinch analytical-model paper with plasma-focus "
            "citations and context. It is not a DPF machine dataset and does "
            "not provide same-scope circuit, phase, spatial, neutron, or "
            "uncertainty targets for this DPF validator."
        ),
    },
    {
        "source": "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md",
        "status": "non_dpf_reference_only",
        "canonical_source": "",
        "reason": (
            "Pulsed-power ideal-MHD paper whose DPF hits occur in references "
            "to HAWK dense plasma focus and general DPF literature. It does "
            "not add DPF validation observables beyond existing HAWK/3D-MHD "
            "target context."
        ),
    },
    {
        "source": (
            "KnowledgeReference/experimental-investigation-of-plasma-electrode-"
            "interactions-on-the-zap-hd-sheared-flow-stabilized-z.md"
        ),
        "status": "non_dpf_reference_only",
        "canonical_source": "",
        "reason": (
            "Zap-HD sheared-flow-stabilized Z-pinch electrode-interaction "
            "paper. The DPF hit is a citation to a plasma-focus material "
            "bombardment paper, not DPF machine validation data."
        ),
    },
    {
        "source": (
            "KnowledgeReference/experimental-investigation-of-plasma-electrode-"
            "interactions-on-the-zap-hd-sheared-flow-stabilized-z-5.md"
        ),
        "status": "duplicate_non_dpf_reference_only",
        "canonical_source": (
            "KnowledgeReference/experimental-investigation-of-plasma-electrode-"
            "interactions-on-the-zap-hd-sheared-flow-stabilized-z.md"
        ),
        "reason": (
            "Duplicate Zap-HD extraction. It is not a DPF validation target "
            "source and contains only a DPF-related reference hit."
        ),
    },
    {
        "source": (
            "KnowledgeReference/formation-and-dynamics-of-z-pinch-plasma-in-a-"
            "coaxial-plasma-gun.md"
        ),
        "status": "non_dpf_reference_only",
        "canonical_source": "",
        "reason": (
            "Coaxial-plasma-gun Z-pinch paper. MJOLNIR and plasma-focus hits "
            "appear in bibliography/context references, not as extractable "
            "DPF simulation-validation observables."
        ),
    },
    {
        "source": (
            "KnowledgeReference/optical-measurements-of-plasma-dynamics-in-"
            "carbon-fiber-z-pinches.md"
        ),
        "status": "non_dpf_reference_only",
        "canonical_source": "",
        "reason": (
            "MAGPIE carbon-fiber Z-pinch optical-diagnostics source. Plasma "
            "focus appears only as a related pulsed-plasma class, not as a DPF "
            "machine dataset for this project."
        ),
    },
    {
        "source": "KnowledgeReference/plasma-physics-and-controlled.md",
        "status": "false_positive_content_marker",
        "canonical_source": "",
        "reason": (
            "Introductory plasma-physics textbook hit is the ordinary phrase "
            "that plasma focuses a wave, not Dense Plasma Focus. It is not a "
            "DPF source for machine validation."
        ),
    },
    {
        "source": (
            "KnowledgeReference/lagrangian-formulation-of-the-snowplow-model-"
            "and-operating-point-for-z-pinch-devices-miguel.md"
        ),
        "status": "non_dpf_general_z_pinch_model",
        "canonical_source": "",
        "reason": (
            "General Lagrangian snowplow/Z-pinch formulation. It references "
            "plasma focus as an application class but does not provide a DPF "
            "device/shot validation packet."
        ),
    },
    {
        "source": (
            "KnowledgeReference/scaling-law-for-discharges-in-z-pinch-devices-"
            "miguel-crdenas-alejandro-nettle-and-leandro-nez.md"
        ),
        "status": "non_dpf_general_z_pinch_model",
        "canonical_source": "",
        "reason": (
            "General Z-pinch scaling-law paper with plasma-focus references. "
            "It is useful background for snowplow-style scaling, but it does "
            "not add DPF-specific measured observables for validation."
        ),
    },
    {
        "source": (
            "KnowledgeReference/snowplow-model-predictions-for-plasma-"
            "temperature-in-z-pinch-discharges-miguel-cardenas-alejandro.md"
        ),
        "status": "non_dpf_general_z_pinch_model",
        "canonical_source": "",
        "reason": (
            "General snowplow temperature-prediction paper for Z-pinch "
            "discharges. Plasma focus is mentioned as an applicability class, "
            "but no DPF same-scope validation data are provided."
        ),
    },
    {
        "source": (
            "KnowledgeReference/see-discussions-stats-and-author-profiles-for-"
            "this-publication-at-4.md"
        ),
        "status": "non_dpf_reference_only",
        "canonical_source": "",
        "reason": (
            "Plasma modelling/numerical simulation source with a plasma-focus-"
            "like reference hit near the bibliography. It does not expose DPF "
            "machine observables or validation data for extraction."
        ),
    },
    {
        "source": (
            "KnowledgeReference/powerlaps-innovative-education-training-in-"
            "high-power-laser-plasmas-plasma-physics-theory-and.md"
        ),
        "status": "educational_lab_manual_not_validation",
        "canonical_source": "",
        "reason": (
            "Educational high-power-laser/plasma laboratory manual including "
            "a miniature plasma-focus exercise. It is not a verified research "
            "paper/book dataset for predictive DPF machine validation."
        ),
    },
    {
        "source": "KnowledgeReference/reference-images/INDEX.md",
        "status": "non_scientific_image_index",
        "canonical_source": "",
        "reason": (
            "Reference-image index listing visual assets and provenance hints. "
            "It does not contain scientific equations, diagnostics, or numeric "
            "validation targets."
        ),
    },
    {
        "source": (
            "KnowledgeReference/investigation-of-optical-properties-and-"
            "chemical-structure-of-nd2o3-nanoparticles-deposited-on-nax.md"
        ),
        "status": "application_materials_not_machine_validation",
        "canonical_source": "",
        "reason": (
            "Application paper on nanoparticle deposition using a plasma-focus "
            "device. It characterizes deposited material, not DPF machine "
            "waveform, phase, spatial plasma, neutron, or uncertainty targets."
        ),
    },
    {
        "source": (
            "KnowledgeReference/investigation-on-the-dynamics-of-z-pinch-in-"
            "discha.md"
        ),
        "status": "non_dpf_reference_only",
        "canonical_source": "",
        "reason": (
            "Discharge-produced Z-pinch light-source paper. Dense-plasma-focus "
            "and plasma-focus terms appear as related light-source references, "
            "not as a DPF validation dataset."
        ),
    },
    {
        "source": "KnowledgeReference/usimindepth-release-301-tech-x-corporation.md",
        "status": "software_manual_not_validation",
        "canonical_source": "",
        "reason": (
            "USim software manual containing a Dense Plasma Focus example. It "
            "documents software setup rather than verified DPF experimental "
            "measurements or peer-reviewed validation targets."
        ),
    },
    {
        "source": "KnowledgeReference/auluck-2022-filamentation.md",
        "status": "qualitative_theory_context_not_validation_target",
        "canonical_source": "",
        "reason": (
            "Auluck filamentation letter is scientifically relevant but "
            "qualitative/conceptual for this validator. It argues that "
            "filamentation may be a native current-distribution feature and "
            "discusses PF-1000 observations, but it does not provide digitized "
            "field, density, timing, neutron, or uncertainty targets."
        ),
    },
    {
        "source": "KnowledgeReference/auluck-2022-poloidal-flux-emission.md",
        "status": "covered_by_poloidal_field_target_context",
        "canonical_source": (
            "KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md"
        ),
        "reason": (
            "Theory letter on poloidal flux emission. Its dynamo/GV-surface "
            "context is represented in the coded Auluck poloidal-field target "
            "from the later local source; this earlier letter does not add a "
            "calibrated PMFE waveform or uncertainty table."
        ),
    },
    {
        "source": "KnowledgeReference/auluck-2023-poloidal-flux-survey.md",
        "status": "partial_diagnostic_survey_not_closure_target",
        "canonical_source": (
            "KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md"
        ),
        "reason": (
            "Exploratory UNU-ICTP PMFE survey is useful magnetic-diagnostic "
            "context, but the current product still needs calibrated, "
            "phase-resolved, same-scope magnetic-field data. The coded "
            "poloidal-field target keeps this area partial rather than using "
            "survey traces as full tier-4 validation."
        ),
    },
    {
        "source": "KnowledgeReference/date-of-current-version-september-26-2019.md",
        "status": "partial_uhf_diagnostic_context_not_closure_target",
        "canonical_source": "",
        "reason": (
            "Orellana 2019 antenna/inductive-sensor paper provides remote UHF "
            "diagnostic context for a hundred-joule plasma-focus accelerator. "
            "It does not supply digitized waveform targets or same-scope "
            "plasma/neutron validation for the end-to-end DPF simulator."
        ),
    },
    {
        "source": "KnowledgeReference/date-of-current-version-september-26-2019-4.md",
        "status": "duplicate",
        "canonical_source": (
            "KnowledgeReference/date-of-current-version-september-26-2019.md"
        ),
        "reason": (
            "Duplicate local extraction of the Orellana 2019 UHF antenna/"
            "inductive-sensor plasma-focus diagnostic paper."
        ),
    },
    {
        "source": "KnowledgeReference/energies.md",
        "status": "lee_model_review_context_covered_by_targets",
        "canonical_source": (
            "KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-"
            "s-lee-and-s-h-saw-part-1-basic-course.md"
        ),
        "reason": (
            "Lee/Saw Energies 2010 review of numerical experiments is useful "
            "for model scope and scaling context, but the active coded Lee "
            "phase/model targets use the more detailed local Lee course and "
            "Lee RADPF theory sources. This review does not add same-scope "
            "experimental validation data."
        ),
    },
    {
        "source": (
            "KnowledgeReference/evolution-of-a-pinch-column-during-the-"
            "acceleration-of-fast-electrons-and-deuterons-in-a-plasma.md"
        ),
        "status": "partial_pf1000_late_pinch_context_not_closure_target",
        "canonical_source": "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md",
        "reason": (
            "PF-1000 late-pinch/current-structure paper is relevant to "
            "filaments, plasmoids, HXR/neutron timing, and magnetic-energy "
            "transfer, but its extractable content remains qualitative or "
            "figure-based for this validator. It does not provide the digitized "
            "same-scope current, neutron, magnetic-field, temperature, and "
            "uncertainty series needed to close the PF-1000 packet."
        ),
    },
    {
        "source": (
            "KnowledgeReference/fused-silica-activation-cherenkov-detector-"
            "for-pulsed-d-t-fusion-yields.md"
        ),
        "status": "detector_calibration_context_not_dpf_machine_validation",
        "canonical_source": "",
        "reason": (
            "Fused-silica activation Cherenkov detector paper provides useful "
            "D-T detector calibration and a MJOLNIR D-D insensitivity check. "
            "It is detector-response context, not a self-consistent DPF "
            "machine validation target."
        ),
    },
    {
        "source": (
            "KnowledgeReference/hard-x-ray-emission-detection-using-deep-"
            "learning-analysis-of-the-radiated-uhf-electromagnetic.md"
        ),
        "status": "xray_ml_diagnostic_context_not_physics_validation",
        "canonical_source": "",
        "reason": (
            "Hard-x-ray/UHF deep-learning diagnostic paper for a 205 J hydrogen "
            "plasma focus. It supports diagnostic correlation research, but "
            "not a physics-model validation target for DPF current, phase, "
            "spatial plasma state, neutron production, or uncertainty."
        ),
    },
    {
        "source": (
            "KnowledgeReference/investigation-of-the-optical-spectra-emitted-"
            "from-plasma-streams-is-of-primary-importance-for.md"
        ),
        "status": "partial_dpf1000u_stream_spectroscopy_not_closure_target",
        "canonical_source": "",
        "reason": (
            "DPF-1000U optical-emission-spectroscopy paper measures free "
            "plasma-stream spectra with and without gas puffing. It is useful "
            "application/stream-diagnostic context but not a same-scope pinch "
            "core validation packet for the simulator."
        ),
    },
    {
        "source": "KnowledgeReference/january-1995-2.md",
        "status": "legacy_thesis_context_not_current_closure_target",
        "canonical_source": "",
        "reason": (
            "Large legacy Serban thesis contains extensive DPF background and "
            "experimental figures/tables. It is not promoted here because the "
            "current closure path requires line-referenced, same-scope, "
            "digitized targets; the local extraction is long OCR text with "
            "many figure-dependent values that need a separate dedicated "
            "digitization pass before use."
        ),
    },
    {
        "source": "KnowledgeReference/neutron-scaling-laws-from-numerical-experiments.md",
        "status": "lee_scaling_context_covered_by_findings",
        "canonical_source": "KnowledgeReference/lee_radpf_theory.md",
        "reason": (
            "Short Lee/Saw neutron-scaling note supports the scaling-law caveat "
            "already recorded in findings. It is not a same-scope validation "
            "target and does not provide measured waveform/neutron histories."
        ),
    },
    {
        "source": "KnowledgeReference/online.md",
        "status": "small_dpf_hollow_anode_yield_context_not_closure_target",
        "canonical_source": "",
        "reason": (
            "Small 2 kJ DPF hollow-anode study reports neutron-yield trends "
            "and copper sputter context. It is useful design context but does "
            "not close the simulator's same-scope high-fidelity validation "
            "requirements."
        ),
    },
    {
        "source": (
            "KnowledgeReference/panda-fes-portable-and-adaptable-neutron-"
            "diagnostics-for-advancing-fusion-energy-science.md"
        ),
        "status": "detector_calibration_context_not_dpf_machine_validation",
        "canonical_source": "",
        "reason": (
            "PANDA-FES activation-detector paper uses MJOLNIR for detector "
            "cross-calibration and uncertainty context. It does not provide "
            "DPF plasma-state or neutron-production validation data for a "
            "self-consistent machine simulation."
        ),
    },
    {
        "source": "KnowledgeReference/plasma-physics-and-technology-1211-9-2025.md",
        "status": "lee_code_pf1000_radiation_model_context_not_validation",
        "canonical_source": "",
        "reason": (
            "Lee-code nitrogen radiation study for PF1000 is numerical-model "
            "context, not an experimental validation packet with measured "
            "same-scope waveform, spatial state, radiation response, and "
            "uncertainty."
        ),
    },
    {
        "source": "KnowledgeReference/preliminary-measurements-of-alpha-particles-produc.md",
        "status": "pb11_alpha_application_context_not_dpf_machine_validation",
        "canonical_source": "",
        "reason": (
            "2026 preliminary p-B11 alpha-measurement paper on PF-360 and "
            "DPF-1000U is relevant to aneutronic application diagnostics, but "
            "it does not validate this project's DPF machine model end to end."
        ),
    },
    {
        "source": (
            "KnowledgeReference/study-of-the-plasma-pinch-and-ion-beam-"
            "properties-versus-the-nitrogen-gas-pressure-using-the-lee.md"
        ),
        "status": "lee_code_ion_beam_model_context_not_validation",
        "canonical_source": "",
        "reason": (
            "Lee-code study of PF400/APF nitrogen pinch and ion-beam "
            "properties is model-output context. It does not add experimental "
            "same-scope validation observables for the end-to-end simulator."
        ),
    },
    {
        "source": (
            "KnowledgeReference/the-code-uses-a-phenomenological-mechanism-"
            "for-beam-target-production-of-fusion-.md"
        ),
        "status": "fragmentary_lee_beam_target_context",
        "canonical_source": "KnowledgeReference/lee_radpf_theory.md",
        "reason": (
            "Short fragment on the Lee phenomenological beam-target neutron "
            "mechanism. The more complete Lee RADPF theory source is already "
            "used for model-scope and beam-target caveats."
        ),
    },
    {
        "source": (
            "KnowledgeReference/this-work-was-performed-under-the-auspices-"
            "of-the-us-department-of-energy-by-lawrence-livermore.md"
        ),
        "status": "presentation_context_covered_by_mjolnir_targets",
        "canonical_source": (
            "KnowledgeReference/ieee-trans-plas-sci-paper-first-experiments-"
            "and-radiographs-on-the-megajoule-neutron-imaging.md"
        ),
        "reason": (
            "LLNL MJOLNIR design presentation contains useful narrative and "
            "tables, but the peer-reviewed MJOLNIR first-experiments and "
            "high/low-yield sources now carry the coded campaign targets."
        ),
    },
    {
        "source": (
            "KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-"
            "dpf-neutron-yield-acb71fa9.md"
        ),
        "status": "source_ingested_target_extraction_needed",
        "canonical_source": "",
        "reason": (
            "User-validated arXiv:2604.09032v1 DPF hybrid PIC-fluid source "
            "is now locally ingested with PDF hash and text parity metadata. "
            "It is first-principles-relevant source authority for model "
            "architecture review, but its geometry, benchmarks, cross-section "
            "fit, and neutron-yield numbers are not accepted validation "
            "targets until typed same-scope target packets are extracted and "
            "reviewed."
        ),
    },
)


def _resolve_kr_root(root: str | Path = "KnowledgeReference") -> Path:
    path = Path(root)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        candidate = base / path
        if candidate.is_dir():
            return candidate
    return path


def _relative_kr_path(path: Path, root: Path) -> str:
    try:
        return f"KnowledgeReference/{path.relative_to(root).as_posix()}"
    except ValueError:
        return path.as_posix()


def _is_dpf_named_md(path: Path) -> bool:
    if path.suffix.lower() != ".md":
        return False
    name = path.name.lower()
    return any(marker in name for marker in _DPF_FILENAME_MARKERS)


def _is_dpf_content_md(path: Path) -> bool:
    if path.suffix.lower() != ".md":
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return any(marker in text for marker in _DPF_CONTENT_MARKERS)


def kr_corpus_inventory(
    *,
    root: str | Path = "KnowledgeReference",
    include_files: bool = False,
) -> dict[str, object]:
    """Return a local inventory of the KnowledgeReference corpus."""
    kr_root = _resolve_kr_root(root)
    files = sorted(path for path in kr_root.rglob("*") if path.is_file())
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    md_files = [path for path in files if path.suffix.lower() == ".md"]
    json_files = [path for path in files if path.suffix.lower() == ".json"]
    dpf_named_md_files = [path for path in md_files if _is_dpf_named_md(path)]
    dpf_content_md_files = [path for path in md_files if _is_dpf_content_md(path)]
    dpf_relevant_md_files = sorted({
        *dpf_named_md_files,
        *dpf_content_md_files,
    })

    inventory: dict[str, object] = {
        "passed": bool(files),
        "model_role": "kr_corpus_inventory",
        "root": kr_root.as_posix(),
        "total_files": len(files),
        "md_files": len(md_files),
        "json_files": len(json_files),
        "dpf_named_md_files": len(dpf_named_md_files),
        "dpf_content_md_files": len(dpf_content_md_files),
        "dpf_relevant_md_files": len(dpf_relevant_md_files),
        "extension_counts": dict(sorted(extension_counts.items())),
        "dpf_filename_markers": list(_DPF_FILENAME_MARKERS),
        "dpf_content_markers": list(_DPF_CONTENT_MARKERS),
        "validity_notes": {
            "scope": (
                "DPF relevance is filename-marker or strong content-marker "
                "based. This inventory does not prove that every relevant fact "
                "inside the corpus has been reviewed or extracted."
            ),
        },
    }
    if include_files:
        inventory["files"] = [_relative_kr_path(path, kr_root) for path in files]
        inventory["dpf_named_md_file_list"] = [
            _relative_kr_path(path, kr_root) for path in dpf_named_md_files
        ]
        inventory["dpf_content_md_file_list"] = [
            _relative_kr_path(path, kr_root) for path in dpf_content_md_files
        ]
        inventory["dpf_relevant_md_file_list"] = [
            _relative_kr_path(path, kr_root) for path in dpf_relevant_md_files
        ]
    return inventory


def _target_source_set(manifest: Sequence[Mapping[str, object]]) -> set[str]:
    return {
        str(target.get("source", ""))
        for target in manifest
        if str(target.get("source", "")).startswith("KnowledgeReference/")
    }


def kr_corpus_review_decisions() -> list[dict[str, object]]:
    """Return explicit non-target review decisions for local KR sources."""
    return [dict(record) for record in _EXPLICIT_REVIEW_DECISIONS]


def kr_corpus_review_status(
    *,
    root: str | Path = "KnowledgeReference",
) -> dict[str, object]:
    """Report how much of the local KR corpus is represented by coded targets."""
    from dpf.validation.kr_targets import (
        kr_validation_same_scope_target_report,
        kr_validation_target_coverage_report,
        kr_validation_target_manifest,
    )

    inventory = kr_corpus_inventory(root=root, include_files=True)
    manifest = kr_validation_target_manifest()
    target_sources = _target_source_set(manifest)
    review_decisions = kr_corpus_review_decisions()
    decision_sources = {
        str(record.get("source", ""))
        for record in review_decisions
        if str(record.get("source", "")).startswith("KnowledgeReference/")
    }
    dpf_named_files = set(inventory.get("dpf_named_md_file_list", []))
    dpf_relevant_files = set(inventory.get("dpf_relevant_md_file_list", []))
    reviewed_dpf_named = sorted(dpf_named_files & (target_sources | decision_sources))
    unreviewed_dpf_named = sorted(dpf_named_files - set(reviewed_dpf_named))
    reviewed_dpf_relevant = sorted(
        dpf_relevant_files & (target_sources | decision_sources)
    )
    unreviewed_dpf_relevant = sorted(
        dpf_relevant_files - set(reviewed_dpf_relevant)
    )
    coverage = kr_validation_target_coverage_report()
    same_scope = kr_validation_same_scope_target_report()

    dpf_total = int(inventory.get("dpf_named_md_files", 0))
    dpf_relevant_total = int(inventory.get("dpf_relevant_md_files", 0))
    reviewed_count = len(reviewed_dpf_named)
    reviewed_fraction = reviewed_count / dpf_total if dpf_total else 0.0
    reviewed_relevant_count = len(reviewed_dpf_relevant)
    reviewed_relevant_fraction = (
        reviewed_relevant_count / dpf_relevant_total
        if dpf_relevant_total else 0.0
    )
    missing_or_partial_groups = coverage.get("missing_or_partial_groups", [])
    if unreviewed_dpf_relevant:
        next_ratcheting_steps = [
            "Review each unreviewed DPF-relevant markdown source for "
            "extractable device, shot, waveform, phase, spatial, neutron, and "
            "uncertainty data.",
            "Promote any extractable records into typed KR targets with source "
            "line ranges and semantic markers.",
            "Stop source review only after the unreviewed DPF-relevant list is "
            "empty; filename-only closure is no longer sufficient.",
        ]
    else:
        next_ratcheting_steps = [
            "DPF-relevant KnowledgeReference markdown review is complete; do "
            "not look for more source files before closing validation gaps.",
            "Close remaining target coverage blockers: "
            f"{', '.join(str(group) for group in missing_or_partial_groups)}.",
            "Promote one same-scope validation packet by adding KR-backed "
            "circuit, phase, spatial, neutron, and uncertainty evidence for a "
            "single device/shot/scope, or keep readiness blocked when KR lacks "
            "those observables.",
        ]

    return {
        "passed": (
            not unreviewed_dpf_relevant
            and coverage.get("passed") is True
            and same_scope.get("passed") is True
        ),
        "model_role": "kr_corpus_review_status",
        "corpus_counts": {
            "total_files": inventory.get("total_files", 0),
            "md_files": inventory.get("md_files", 0),
            "json_files": inventory.get("json_files", 0),
            "dpf_named_md_files": dpf_total,
            "dpf_content_md_files": inventory.get("dpf_content_md_files", 0),
            "dpf_relevant_md_files": dpf_relevant_total,
        },
        "coded_target_count": len(manifest),
        "unique_coded_target_source_count": len(target_sources),
        "reviewed_dpf_named_md_files": reviewed_count,
        "reviewed_dpf_named_md_fraction": reviewed_fraction,
        "reviewed_dpf_relevant_md_files": reviewed_relevant_count,
        "reviewed_dpf_relevant_md_fraction": reviewed_relevant_fraction,
        "reviewed_by_coded_target_files": len(dpf_named_files & target_sources),
        "reviewed_by_explicit_decision_files": len(dpf_named_files & decision_sources),
        "reviewed_relevant_by_coded_target_files": len(
            dpf_relevant_files & target_sources
        ),
        "reviewed_relevant_by_explicit_decision_files": len(
            dpf_relevant_files & decision_sources
        ),
        "explicit_review_decisions": review_decisions,
        "unreviewed_dpf_named_md_files": unreviewed_dpf_named,
        "unreviewed_dpf_relevant_md_files": unreviewed_dpf_relevant,
        "target_coverage_passed": coverage.get("passed") is True,
        "target_missing_or_partial_groups": missing_or_partial_groups,
        "same_scope_passed": same_scope.get("passed") is True,
        "best_available_scope": same_scope.get("best_available_scope"),
        "next_ratcheting_steps": next_ratcheting_steps,
        "validity_notes": {
            "review_definition": (
                "A file counts as reviewed here only when it contributes to a "
                "coded KR validation target or has an explicit review decision "
                "with a reason. Human reading without either artifact is "
                "intentionally not counted as closure."
            ),
        },
    }


def _line_hits_for_markers(
    path: Path,
    markers: Sequence[str],
    *,
    max_hits: int,
) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return hits
    lowered_markers = tuple(marker.lower() for marker in markers)
    for line_number, line in enumerate(lines, start=1):
        line_lower = line.lower()
        matched = [marker for marker in lowered_markers if marker in line_lower]
        if not matched:
            continue
        hits.append({
            "line": line_number,
            "matched_markers": matched,
            "text": line.strip()[:240],
        })
        if len(hits) >= max_hits:
            break
    return hits


def kr_unreviewed_dpf_source_triage(
    *,
    root: str | Path = "KnowledgeReference",
    max_hits_per_category: int = 3,
) -> dict[str, object]:
    """Triage unreviewed DPF-relevant KR markdown files by observable keywords."""
    status = kr_corpus_review_status(root=root)
    kr_root = _resolve_kr_root(root)
    records: list[dict[str, object]] = []
    category_counts = {category: 0 for category in _TARGET_CATEGORY_MARKERS}

    for source in status.get("unreviewed_dpf_relevant_md_files", []):
        source_path = Path(str(source))
        if not source_path.is_absolute():
            source_path = kr_root / source_path.relative_to("KnowledgeReference")
        categories: dict[str, list[dict[str, object]]] = {}
        for category, markers in _TARGET_CATEGORY_MARKERS.items():
            hits = _line_hits_for_markers(
                source_path,
                markers,
                max_hits=max_hits_per_category,
            )
            if hits:
                categories[category] = hits
                category_counts[category] += 1
        records.append({
            "source": str(source),
            "candidate_categories": sorted(categories),
            "line_hits": categories,
            "needs_human_review": True,
        })

    records.sort(
        key=lambda record: (
            -len(record["candidate_categories"]),  # type: ignore[arg-type]
            str(record["source"]),
        )
    )
    return {
        "passed": not records,
        "model_role": "kr_unreviewed_dpf_source_triage",
        "unreviewed_dpf_relevant_md_files": len(records),
        "category_counts": category_counts,
        "records": records,
        "validity_notes": {
            "scope": (
                "This is a keyword triage queue, not a scientific extraction. "
                "A category hit means the source should be reviewed for usable "
                "line-referenced validation targets."
            ),
        },
    }
