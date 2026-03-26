# Thomson Scattering Fitting Robustness Investigation

**Date**: 2026-03-26
**Concern**: `fit_te_ne_v` using `scipy.optimize.curve_fit` may be fragile in the
collective regime (alpha > 2) where the Salpeter spectrum has multi-peak structure.

## Test Setup

Synthetic Salpeter spectrum at PF-1000 pinch conditions:
- ne = 1e25 m^-3, Te = 300 eV, Ti = 300 eV, v_bulk = 0
- lambda_0 = 1064 nm, theta = 90 deg
- **alpha = 2.94** (firmly collective)
- 5% Gaussian noise (S/N = 20 at peak)
- 2000 wavelength bins over +/-50 nm

Script: `scripts/test_thomson_fitting_robustness.py`

## Results

### Method 1: Raw curve_fit (as specified in design doc)

| Initial Guess | ne error | Te error | Converged | Time |
|---------------|----------|----------|-----------|------|
| Good (2x off) | 50% | 33% | NO | 0.00s |
| Bad (100x low) | 99% | 83% | NO | 0.00s |
| Very bad (100x high) | 9900% | 567% | NO | 0.00s |
| Wrong regime | 100% | 97% | NO | 0.00s |

**0/4 converged.** The optimizer takes zero steps (0.00s runtime). It returns the
initial guess unchanged every time.

**Root cause**: The Jacobian is catastrophically ill-conditioned. The spectrum S has
values ~1e-13 s/rad, while ne ~1e25 m^-3. The partial derivative dS/dne ~ 1e-38 is
below the TRF algorithm's gradient tolerance, so it declares convergence at the
initial guess immediately. This is not a local minimum problem -- the optimizer
literally cannot see the gradient.

### Method 2: Log-scaled curve_fit (log10(ne) parameter)

**0/4 converged.** Log transform fixes the ne scaling, but the spectrum values
(~1e-13) are still too small. The absolute residual is ~1e-24, and its gradient
is below machine precision thresholds for TRF.

### Method 3: Normalized spectrum + log10(ne)

| Initial Guess | ne error | Te error | Ti error | Converged |
|---------------|----------|----------|----------|-----------|
| Good (2x off) | 1.0% | 0.8% | 0.1% | YES |
| Bad (100x low) | 1.9% | 1567% | 767% | NO |
| Very bad (100x high) | 1.0% | 0.8% | 0.1% | YES |

**2/3 converged.** Normalizing spectrum to peak=1 gives the optimizer real gradients.
Works when initial guess is within ~1 order of magnitude on Te/Ti. Fails when
starting from Te=50 eV (stuck in local minimum where ion feature width matches
a different Te/ne combination).

### Method 4: Differential Evolution (global optimizer)

| ne error | Te error | Ti error | Converged | Time | Func Evals |
|----------|----------|----------|-----------|------|------------|
| 0.9% | 0.8% | 0.1% | YES | 0.6s | 5845 |

**Robust.** Finds the global minimum regardless of initial conditions. The
`polish=True` option runs L-BFGS-B after DE for refinement. 0.6s is acceptable
for a diagnostic post-processing step (not in any hot loop).

### Method 5: Two-stage (grid search + curve_fit)

Grid search (12,500 evaluations, 1.0s) found ne=2.6e25, Te=268, Ti=350 --
close but not converged. The subsequent curve_fit from this starting point
did NOT refine further (stuck at the grid point). The grid was too coarse on
ne, and the ne-Te degeneracy means many grid points have similar residuals.

## Failure Modes Identified

### 1. Jacobian Conditioning (CRITICAL)

The raw spectrum has values ~1e-13 s/rad. When combined with ne ~ 1e25 m^-3
as a fit parameter, the Jacobian elements span 38 orders of magnitude. TRF's
finite-difference Jacobian approximation cannot resolve this. The optimizer
concludes it is already at a stationary point and returns p0 unchanged.

**Fix**: Always normalize the spectrum to peak=1 before fitting, and fit
log10(ne) instead of ne.

### 2. ne-Te Degeneracy

In the collective regime, alpha = 1/(k * lambda_D) depends on both ne and Te
through lambda_D = sqrt(eps0 * kB * Te / (ne * e^2)). The spectral shape
is primarily controlled by alpha, not ne or Te independently. This creates
a valley in parameter space where many (ne, Te) pairs produce similar spectra.

**Fix**: Use the ion acoustic peak location to constrain Te independently of ne
(peak position depends on sqrt(Te/mi), not ne). Then ne is determined from
total spectral power.

### 3. Te-Ti Coupling

When Te = Ti (as in our test), the ion and electron features overlap significantly.
The optimizer can trade Te for Ti freely within a range. With Ti != Te, the ion
feature narrowing provides an independent Ti constraint.

**Fix**: Accept that Te-Ti separation requires the ion feature to be resolved.
Report fit uncertainty from the covariance matrix.

### 4. v_bulk-Noise Aliasing

Differential evolution found v_bulk = 274,000 m/s (true: 0) because noise-induced
spectral asymmetry can be partially explained by a Doppler shift. This is a real
degeneracy at S/N = 20.

**Fix**: If v_bulk is not expected to be large, constrain its bounds more tightly
(e.g., +/- 1e5 m/s for DPF conditions). Or fix v_bulk = 0 if flow measurement
is not the goal.

## Recommended Implementation for `fit_te_ne_v()`

Replace the current `scipy.optimize.curve_fit` design with a two-phase approach:

```python
def fit_te_ne_v(wavelength_grid, spectrum, ...):
    """Robust Thomson spectrum fitting.

    Phase 1: Differential evolution (global search, ~0.5s)
    Phase 2: curve_fit refinement from DE result (local polish)
    """
    # Normalize spectrum
    S_peak = np.max(spectrum)
    S_normed = spectrum / S_peak

    # Fit in transformed space: [log10(ne), Te_eV, Ti_eV, v_bulk]
    def cost(params):
        log_ne, Te, Ti, v = params
        model = _forward_model(wavelength_grid, 10**log_ne, Te, Ti, v, ...)
        return np.sum(((model / S_peak) - S_normed)**2)

    # Phase 1: DE global search
    bounds = [(20, 28), (10, 5000), (10, 5000), (-1e6, 1e6)]
    de_result = differential_evolution(cost, bounds, seed=0,
                                        maxiter=300, polish=True)

    # Phase 2: Local refinement (optional, DE polish usually sufficient)
    log_ne, Te, Ti, v = de_result.x

    # Compute alpha for regime reporting
    ...

    return {
        "Te_eV": Te, "ne_m3": 10**log_ne, "Ti_eV": Ti,
        "v_flow_ms": v, "alpha": alpha,
        "chi2_dof": de_result.fun / (len(spectrum) - 4),
        "converged": de_result.success,
    }
```

### Why Differential Evolution

1. **No initial guess required** -- eliminates the fragility entirely
2. **Handles multi-modal landscapes** -- mutation/crossover explores the full parameter space
3. **0.6s runtime** -- acceptable for post-processing (Thomson spectra are analyzed offline)
4. **polish=True** gives local refinement for free (L-BFGS-B after DE)
5. **scipy built-in** -- no new dependencies

### Alternative: Feature Detection + curve_fit

For real-time applications where 0.6s is too slow:

1. Detect ion acoustic peaks via `scipy.signal.find_peaks` on smoothed spectrum
2. Peak location -> Te estimate: `Te ~ m_i * (omega_peak/k)^2 / (4*kB)` for Z=1, Ti~Te
3. Total power integral -> ne estimate (calibrated against forward model)
4. Feed estimates to normalized+log curve_fit

This approach is ~10ms but requires the ion feature to be resolvable (alpha > 1.5)
and well above noise. Not recommended as the primary method.

### Changes to Design Doc

1. Replace `curve_fit` with `differential_evolution` in `fit_te_ne_v()` specification
2. Add `chi2_dof` to return dict as fit quality metric
3. Add `Ti_eV` bounds parameter to constrain Te-Ti if prior knowledge available
4. Document that v_bulk recovery requires S/N > 50 or independent measurement
5. Add ~30 LOC to the estimate (DE is simpler code but needs bounds setup)

## References

- Real Thomson scattering analysis codes (e.g., OMEGA laser facility, NIF) use
  Bayesian MCMC (emcee) or nested sampling for similar multi-parameter Salpeter
  fits. DE is the minimum viable robust approach.
- Sheffield (2011) Ch. 7 discusses fitting procedures and warns about ne-Te
  degeneracy in the collective regime.
- The Salpeter sum rule (integral of S = 1/k) provides an independent ne
  constraint that could be added as a penalty term.
