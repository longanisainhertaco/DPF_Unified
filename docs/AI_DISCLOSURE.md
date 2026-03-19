# AI Usage Disclosure

This document describes the role of artificial intelligence tools in the
development of DPF-Unified, in compliance with the Journal of Open Source
Software (JOSS) AI usage disclosure requirements.

## AI Tools Used

**Claude Code (Anthropic Claude, Opus model)** was used extensively throughout
the development of DPF-Unified from 2024 to 2026. Claude Code is an
AI-powered coding assistant that operates within a terminal environment,
reading and writing source files, running tests, and executing shell commands
under human supervision.

## Scope of AI Contribution

### Areas where AI contributed substantially

- **Test scaffolding and test generation**: The majority of the 4,183-test
  suite was drafted by Claude Code, including unit tests, integration tests,
  convergence studies, and cross-backend parity checks. Test designs were
  specified by the human developer; AI generated the boilerplate and assertion
  logic.

- **UI and web interface code**: The Gradio web interface (`app.py`,
  `app_engine.py`, `app_mhd.py`, `app_plots.py`, `app_narrative.py`,
  `app_anim.py`, `app_compare.py`) was largely AI-generated from functional
  specifications.

- **Documentation**: README, architecture documents, V&V reports, and inline
  docstrings were drafted by AI and edited by the human developer.

- **CI/CD configuration**: GitHub Actions workflows, Docker configuration,
  and linting setup.

- **Boilerplate and infrastructure**: CLI entry points, FastAPI server
  endpoints, WebSocket streaming, Pydantic configuration models, and HDF5 I/O
  routines.

- **Code refactoring**: File splitting (enforcing a 400-line limit),
  import cleanup, type annotation, and style enforcement.

- **Literature search and synthesis**: AI assisted in identifying relevant
  publications, extracting experimental parameters from papers, and comparing
  results across devices. A research database of 731 papers was compiled with
  AI assistance.

### Areas where human expertise was essential

- **Physics model selection and design**: The choice of governing equations
  (resistive MHD + Lee model snowplow + circuit coupling), the decision to use
  cylindrical (r,z) axisymmetric geometry, and the selection of transport
  coefficient formulations (Spitzer with GMS Coulomb logarithm, Braginskii
  anisotropic transport) were made by the human developer (Anthony Zamora)
  based on domain knowledge of dense plasma focus physics.

- **Validation interpretation**: Determining whether simulation-experiment
  agreement is physically meaningful required human judgment. For example,
  identifying that the early-time current rise mismatch is caused by the
  absence of an insulator flashover model (a known Lee model limitation), or
  recognizing that NX2 "experimental" data in the literature is likely RADPF
  model output rather than measurement.

- **Device preset calibration**: Calibrating Lee model parameters (fc, fm,
  fmr) for each device required understanding the coupled nature of circuit
  and plasma parameters. The 24-shot PF-1000 statistical validation against
  Akel et al. (2021) required identifying a 6.43 mOhm parasitic resistance
  offset -- a subtle experimental detail that AI could not infer from the
  published data alone.

- **Numerical method selection**: Choosing WENO5-Z over WENO-JS, HLLD over
  HLL, and SSP-RK3 over lower-order integrators involved trade-off analysis
  between accuracy, stability, and computational cost on Apple Silicon
  hardware.

- **Bug diagnosis in physics code**: Critical bugs -- such as a factor-of-2
  error in the coaxial inductance coefficient, a missing current fraction
  (fc) in the snowplow model, and an SI-to-Heaviside-Lorentz unit conversion
  error in the Metal solver -- were identified through physics reasoning, not
  automated testing. These bugs produced plausible but incorrect results that
  passed all existing tests.

- **Research direction and architecture decisions**: The tri-engine
  architecture (Python + Athena++ + AthenaK), the decision to implement a
  0D snowplow model before attempting full 2D MHD coupling, and the strategy
  of validating against statistical multi-shot data rather than single
  waveforms were human decisions.

## Verification Process

All physics implementations generated or modified with AI assistance were
verified through multiple independent checks:

1. **Equation tracing**: Every physics module traces its governing equations
   to published references (cited in docstrings). Sign conventions and unit
   conversions were manually verified against the original papers.

2. **Analytical benchmarks**: Sod shock tube, Brio-Wu MHD shock tube,
   resistive diffusion convergence, and Orszag-Tang vortex results were
   compared against published solutions.

3. **Experimental validation**: Current waveforms were compared against
   published data for 6 devices (PF-1000, UNU-ICTP, MJOLNIR, FAETON-I,
   PF-400J, POSEIDON). The PF-1000 was validated statistically across 24
   shots (Akel et al. 2021) with 1.27% mean I_peak error and Pearson
   r = 0.9899.

4. **Automated test suite**: 4,183 tests (pytest) covering unit physics,
   shock tubes, convergence studies, cross-backend parity, energy
   conservation, and integration tests. Tests run in CI on every commit.

5. **Independent review panels**: The codebase was assessed by simulated
   expert panels ("PhD Debate" sessions) at multiple milestones, identifying
   bugs and physics gaps that were subsequently addressed.

## Statement of Responsibility

Anthony Zamora is the sole author and takes full responsibility for the
scientific correctness of all physics implementations, validation claims, and
published results. AI tools were used as productivity aids; all physics
decisions and interpretations are human-authored.
