"""Thomson scattering curve_fit robustness test for collective regime.

Tests whether scipy.optimize.curve_fit can reliably recover plasma parameters
from a synthetic Salpeter spectrum at PF-1000 pinch conditions (alpha ~ 2.9).

The concern: multi-peak spectra (ion acoustic + electron features) create a
rugged chi-squared landscape where gradient-based optimizers (Levenberg-Marquardt)
can get trapped in local minima. Initial guess becomes critical.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks
from scipy.special import wofz

# numpy compat
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ---------- Physical constants (SI) ----------
e_charge = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
k_B = 1.380649e-23
m_e = 9.1093837015e-31
m_D = 3.3435837724e-27
c = 2.99792458e8
eV = e_charge


def spectral_density_salpeter(
    omega: np.ndarray, k: float, ne: float, Te_eV: float,
    Ti_eV: float | None = None, m_i: float = m_D, Z_ion: int = 1,
) -> np.ndarray:
    """Full Salpeter spectral density S(k, omega) via Faddeeva function."""
    if Ti_eV is None:
        Ti_eV = Te_eV
    Te = Te_eV * eV / k_B
    Ti = Ti_eV * eV / k_B
    v_th_e = np.sqrt(2 * k_B * Te / m_e)
    v_th_i = np.sqrt(2 * k_B * Ti / m_i)

    lambda_De = np.sqrt(epsilon_0 * k_B * Te / (ne * e_charge**2))
    lambda_Di = np.sqrt(epsilon_0 * k_B * Ti / (ne * Z_ion * e_charge**2))
    alpha_e = 1.0 / (k * lambda_De)
    alpha_i = 1.0 / (k * lambda_Di)

    zeta_e = omega / (k * v_th_e)
    zeta_i = omega / (k * v_th_i)

    Z_e = 1j * np.sqrt(np.pi) * wofz(zeta_e)
    Z_i = 1j * np.sqrt(np.pi) * wofz(zeta_i)

    chi_e = -alpha_e**2 * (1.0 + zeta_e * Z_e)
    chi_i = -alpha_i**2 * (1.0 + zeta_i * Z_i)
    epsilon_d = 1.0 + chi_e + chi_i

    f_e = np.exp(-zeta_e**2) / (v_th_e * np.sqrt(np.pi))
    f_i = np.exp(-zeta_i**2) / (v_th_i * np.sqrt(np.pi))

    S_e = (2 * np.pi / k) * np.abs(1 - chi_e / epsilon_d) ** 2 * f_e
    S_i = (2 * np.pi / k) * np.abs(chi_e / epsilon_d) ** 2 * f_i * Z_ion

    return np.real(S_e + S_i)


def wavelength_to_omega(wl: np.ndarray, lambda0: float) -> np.ndarray:
    return 2 * np.pi * c * (1.0 / wl - 1.0 / lambda0)


# ---------- Test parameters: PF-1000 pinch ----------
NE_TRUE = 1e25
TE_TRUE = 300.0
TI_TRUE = 300.0
V_TRUE = 0.0
LAMBDA0 = 1064e-9
THETA = np.pi / 2
K_SCAT = (4 * np.pi / LAMBDA0) * np.sin(THETA / 2)

lambda_D = np.sqrt(epsilon_0 * k_B * TE_TRUE * eV / k_B / (NE_TRUE * e_charge**2))
alpha = 1.0 / (K_SCAT * lambda_D)

print("=== Thomson Scattering Fitting Robustness Test ===")
print(f"PF-1000 pinch: ne={NE_TRUE:.1e} m^-3, Te={TE_TRUE} eV, Ti={TI_TRUE} eV")
print(f"k = {K_SCAT:.3e} m^-1, lambda_D = {lambda_D:.3e} m, alpha = {alpha:.2f}")
print()

# ---------- Generate synthetic spectrum ----------
wl_grid = np.linspace(LAMBDA0 - 50e-9, LAMBDA0 + 50e-9, 2000)
omega_grid = wavelength_to_omega(wl_grid, LAMBDA0)
S_true = spectral_density_salpeter(omega_grid, K_SCAT, NE_TRUE, TE_TRUE, TI_TRUE)

rng = np.random.default_rng(42)
noise_level = 0.05 * np.max(S_true)
S_noisy = S_true + rng.normal(0, noise_level, size=S_true.shape)
S_noisy = np.maximum(S_noisy, 0)

print(f"Spectrum: max={np.max(S_true):.3e}, noise_level={noise_level:.3e}")
print(f"S/N ratio at peak: {np.max(S_true)/noise_level:.1f}")
print()


# ============================================================
# Method 1: NAIVE curve_fit (raw parameters, no scaling)
# This is what the design doc specifies
# ============================================================

def forward_raw(wl, ne, Te_eV, Ti_eV, v_bulk):
    """Forward model with raw physical parameters."""
    doppler = LAMBDA0 * v_bulk * K_SCAT / (2 * np.pi * c)
    omega = wavelength_to_omega(wl - doppler, LAMBDA0)
    return spectral_density_salpeter(omega, K_SCAT, ne, Te_eV, Ti_eV)


def try_raw_curvefit(label, p0, bounds):
    print(f"--- {label} ---")
    print(f"  p0: ne={p0[0]:.1e}, Te={p0[1]:.0f}, Ti={p0[2]:.0f}, v={p0[3]:.0f}")
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            popt, pcov = curve_fit(
                forward_raw, wl_grid, S_noisy,
                p0=p0, bounds=bounds, maxfev=20000, method="trf",
                x_scale="jac",  # let TRF scale by Jacobian
            )
        dt = time.perf_counter() - t0
        ne_f, Te_f, Ti_f, v_f = popt
        ne_err = abs(ne_f - NE_TRUE) / NE_TRUE * 100
        Te_err = abs(Te_f - TE_TRUE) / TE_TRUE * 100
        Ti_err = abs(Ti_f - TI_TRUE) / TI_TRUE * 100
        res = np.sum((forward_raw(wl_grid, *popt) - S_noisy) ** 2)
        res_true = np.sum((S_true - S_noisy) ** 2)
        converged = ne_err < 20 and Te_err < 20 and Ti_err < 20
        print(f"  Result: ne={ne_f:.2e} ({ne_err:.1f}%), Te={Te_f:.1f} ({Te_err:.1f}%), "
              f"Ti={Ti_f:.1f} ({Ti_err:.1f}%), v={v_f:.0f}")
        print(f"  Residual ratio: {res/res_true:.2f}, Time: {dt:.2f}s, OK: {'YES' if converged else 'NO'}")
        warns = []
        if warns:
            print(f"  Warnings: {warns[0][:80]}")
        return {"converged": converged, "ne_err": ne_err, "Te_err": Te_err, "Ti_err": Ti_err, "time": dt, "residual": res}
    except Exception as ex:
        dt = time.perf_counter() - t0
        print(f"  FAILED: {type(ex).__name__}: {str(ex)[:100]}")
        return {"converged": False, "time": dt, "error": str(ex)}


BOUNDS_RAW = ([1e20, 10, 10, -1e7], [1e27, 5000, 5000, 1e7])

print("=" * 70)
print("METHOD 1: Raw curve_fit (naive, as in design doc)")
print("=" * 70)
r1_good = try_raw_curvefit("Good guess (2x off)", [5e24, 200, 200, 0], BOUNDS_RAW)
print()
r1_bad = try_raw_curvefit("Bad guess (100x off ne)", [1e23, 50, 50, 0], BOUNDS_RAW)
print()
r1_vbad = try_raw_curvefit("Very bad (100x high ne)", [1e27, 2000, 2000, 0], BOUNDS_RAW)
print()
r1_wrong = try_raw_curvefit("Wrong regime (non-collective)", [1e21, 10, 10, 0], BOUNDS_RAW)
print()


# ============================================================
# Method 2: Log-scaled curve_fit (fit log10(ne) instead of ne)
# Fixes the Jacobian conditioning problem
# ============================================================

def forward_log(wl, log10_ne, Te_eV, Ti_eV, v_bulk):
    """Forward model with log10(ne) for better conditioning."""
    ne = 10**log10_ne
    doppler = LAMBDA0 * v_bulk * K_SCAT / (2 * np.pi * c)
    omega = wavelength_to_omega(wl - doppler, LAMBDA0)
    return spectral_density_salpeter(omega, K_SCAT, ne, Te_eV, Ti_eV)


def try_log_curvefit(label, p0, bounds):
    print(f"--- {label} ---")
    print(f"  p0: log10(ne)={p0[0]:.1f} (ne={10**p0[0]:.1e}), Te={p0[1]:.0f}, Ti={p0[2]:.0f}")
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            popt, pcov = curve_fit(
                forward_log, wl_grid, S_noisy,
                p0=p0, bounds=bounds, maxfev=20000, method="trf",
            )
        dt = time.perf_counter() - t0
        lne_f, Te_f, Ti_f, v_f = popt
        ne_f = 10**lne_f
        ne_err = abs(ne_f - NE_TRUE) / NE_TRUE * 100
        Te_err = abs(Te_f - TE_TRUE) / TE_TRUE * 100
        Ti_err = abs(Ti_f - TI_TRUE) / TI_TRUE * 100
        res = np.sum((forward_log(wl_grid, *popt) - S_noisy) ** 2)
        res_true = np.sum((S_true - S_noisy) ** 2)
        converged = ne_err < 20 and Te_err < 20 and Ti_err < 20
        print(f"  Result: ne={ne_f:.2e} ({ne_err:.1f}%), Te={Te_f:.1f} ({Te_err:.1f}%), "
              f"Ti={Ti_f:.1f} ({Ti_err:.1f}%), v={v_f:.0f}")
        print(f"  Residual ratio: {res/res_true:.2f}, Time: {dt:.2f}s, OK: {'YES' if converged else 'NO'}")
        warns = []
        if warns:
            print(f"  Warnings: {warns[0][:80]}")
        return {"converged": converged, "ne_err": ne_err, "Te_err": Te_err, "Ti_err": Ti_err, "time": dt}
    except Exception as ex:
        dt = time.perf_counter() - t0
        print(f"  FAILED: {type(ex).__name__}: {str(ex)[:100]}")
        return {"converged": False, "time": dt, "error": str(ex)}


BOUNDS_LOG = ([20, 10, 10, -1e7], [27, 5000, 5000, 1e7])

print("=" * 70)
print("METHOD 2: Log-scaled curve_fit (log10(ne), same TRF)")
print("=" * 70)
r2_good = try_log_curvefit("Good guess", [np.log10(5e24), 200, 200, 0], BOUNDS_LOG)
print()
r2_bad = try_log_curvefit("Bad guess", [23, 50, 50, 0], BOUNDS_LOG)
print()
r2_vbad = try_log_curvefit("Very bad", [27, 2000, 2000, 0], BOUNDS_LOG)
print()
r2_wrong = try_log_curvefit("Wrong regime", [21, 10, 10, 0], BOUNDS_LOG)
print()


# ============================================================
# Method 3: Differential Evolution (global optimizer)
# ============================================================

print("=" * 70)
print("METHOD 3: Differential Evolution (global)")
print("=" * 70)

def de_cost(params):
    log_ne, Te, Ti, v = params
    ne = 10**log_ne
    try:
        omega = wavelength_to_omega(wl_grid, LAMBDA0)
        model = spectral_density_salpeter(omega, K_SCAT, ne, Te, Ti)
        return np.sum((model - S_noisy) ** 2)
    except Exception:
        return 1e30

t0 = time.perf_counter()
de_bounds = [(22, 27), (10, 3000), (10, 3000), (-1e6, 1e6)]
de_res = differential_evolution(
    de_cost, de_bounds, seed=42, maxiter=500, tol=1e-10,
    workers=1, polish=True, popsize=20, mutation=(0.5, 1.5), recombination=0.9,
)
dt_de = time.perf_counter() - t0
ne_de = 10**de_res.x[0]
Te_de, Ti_de, v_de = de_res.x[1], de_res.x[2], de_res.x[3]
ne_err_de = abs(ne_de - NE_TRUE) / NE_TRUE * 100
Te_err_de = abs(Te_de - TE_TRUE) / TE_TRUE * 100
Ti_err_de = abs(Ti_de - TI_TRUE) / TI_TRUE * 100
conv_de = ne_err_de < 20 and Te_err_de < 20 and Ti_err_de < 20
print(f"  Result: ne={ne_de:.2e} ({ne_err_de:.1f}%), Te={Te_de:.1f} ({Te_err_de:.1f}%), "
      f"Ti={Ti_de:.1f} ({Ti_err_de:.1f}%), v={v_de:.0f}")
print(f"  Iters: {de_res.nit}, Evals: {de_res.nfev}, Time: {dt_de:.1f}s, OK: {'YES' if conv_de else 'NO'}")
print()


# ============================================================
# Method 4: Two-stage (coarse grid + log curve_fit)
# ============================================================

print("=" * 70)
print("METHOD 4: Two-stage (grid search + log curve_fit)")
print("=" * 70)
t0 = time.perf_counter()

ne_grid = np.logspace(22, 27, 20)
Te_test = np.linspace(20, 2000, 25)
Ti_test = np.linspace(20, 2000, 25)
omega_g = wavelength_to_omega(wl_grid, LAMBDA0)

best_cost = np.inf
best_params = None

for ne_g in ne_grid:
    for Te_g in Te_test:
        for Ti_g in Ti_test:
            try:
                model = spectral_density_salpeter(omega_g, K_SCAT, ne_g, Te_g, Ti_g)
                cost = np.sum((model - S_noisy) ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_params = (ne_g, Te_g, Ti_g)
            except Exception:
                continue

dt_grid = time.perf_counter() - t0
ne_g, Te_g, Ti_g = best_params
print(f"  Grid: {20*25*25} evals in {dt_grid:.1f}s")
print(f"  Best: ne={ne_g:.1e}, Te={Te_g:.0f}, Ti={Ti_g:.0f}")

# Refine with log curve_fit
r4 = try_log_curvefit(
    "Grid -> log curve_fit",
    [np.log10(ne_g), Te_g, Ti_g, 0.0], BOUNDS_LOG,
)
dt_total_4 = time.perf_counter() - t0
print(f"  Total time: {dt_total_4:.1f}s")
print()


# ============================================================
# Method 5: Feature detection + log curve_fit
# ============================================================

print("=" * 70)
print("METHOD 5: Feature detection + log curve_fit")
print("=" * 70)
t0 = time.perf_counter()

# Find ion acoustic peaks
center_idx = np.argmin(np.abs(wl_grid - LAMBDA0))
right_half = S_noisy[center_idx:]
wl_right = wl_grid[center_idx:]

# Smooth
kernel = np.ones(30) / 30
smooth = np.convolve(right_half, kernel, mode="same")
peaks, props = find_peaks(smooth, height=0.05 * np.max(smooth), prominence=0.02 * np.max(smooth), distance=20)

if len(peaks) > 0:
    peak_wl = wl_right[peaks[0]]
    dlambda_peak = peak_wl - LAMBDA0
    print(f"  Ion acoustic peak at dlambda = {dlambda_peak*1e9:.2f} nm")

    # Ion acoustic speed: c_s = sqrt(Z*kB*Te/mi + 3*kB*Ti/mi)
    # Peak at omega_ia = k * c_s, so dlambda ~ lambda0^2 * c_s * k / (2*pi*c)
    omega_peak = abs(wavelength_to_omega(np.array([peak_wl]), LAMBDA0)[0])
    c_s = omega_peak / K_SCAT
    # Assuming Ti ~ Te and Z=1: c_s^2 ~ 4*kB*T/mi
    T_est_eV = m_D * c_s**2 / (4 * k_B / eV)
    print(f"  Estimated T from peak: {T_est_eV:.0f} eV")

    # ne from total spectral power (rough scaling: S ~ ne for fixed shape)
    total_power = np.trapz(S_noisy, wl_grid)
    # Use the grid-detected Te to compute a reference spectrum
    ref_spec = spectral_density_salpeter(omega_g, K_SCAT, 1e25, T_est_eV, T_est_eV)
    ref_power = np.trapz(ref_spec, wl_grid)
    ne_est = 1e25 * total_power / max(ref_power, 1e-30)
    ne_est = np.clip(ne_est, 1e20, 1e28)
    print(f"  Estimated ne: {ne_est:.1e}")
else:
    print("  No peaks detected -- falling back to broad guess")
    T_est_eV = 200.0
    ne_est = 1e24

r5 = try_log_curvefit(
    "Feature -> log curve_fit",
    [np.log10(ne_est), T_est_eV, T_est_eV, 0.0], BOUNDS_LOG,
)
dt_feat = time.perf_counter() - t0
print(f"  Total time: {dt_feat:.2f}s")
print()


# ============================================================
# Method 6: Normalized residual curve_fit
# Scale spectrum to [0,1] to normalize the cost landscape
# ============================================================

print("=" * 70)
print("METHOD 6: Normalized spectrum + log curve_fit")
print("=" * 70)

S_scale = np.max(S_noisy)
S_normed = S_noisy / S_scale

def forward_normed(wl, log10_ne, Te_eV, Ti_eV, v_bulk):
    ne = 10**log10_ne
    omega = wavelength_to_omega(wl, LAMBDA0)
    S = spectral_density_salpeter(omega, K_SCAT, ne, Te_eV, Ti_eV)
    return S / S_scale

def try_normed_curvefit(label, p0, bounds):
    print(f"--- {label} ---")
    print(f"  p0: log10(ne)={p0[0]:.1f}, Te={p0[1]:.0f}, Ti={p0[2]:.0f}")
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            popt, pcov = curve_fit(
                forward_normed, wl_grid, S_normed,
                p0=p0, bounds=bounds, maxfev=20000, method="trf",
            )
        dt = time.perf_counter() - t0
        lne_f, Te_f, Ti_f, v_f = popt
        ne_f = 10**lne_f
        ne_err = abs(ne_f - NE_TRUE) / NE_TRUE * 100
        Te_err = abs(Te_f - TE_TRUE) / TE_TRUE * 100
        Ti_err = abs(Ti_f - TI_TRUE) / TI_TRUE * 100
        converged = ne_err < 20 and Te_err < 20 and Ti_err < 20
        print(f"  Result: ne={ne_f:.2e} ({ne_err:.1f}%), Te={Te_f:.1f} ({Te_err:.1f}%), "
              f"Ti={Ti_f:.1f} ({Ti_err:.1f}%), v={v_f:.0f}")
        print(f"  Time: {dt:.2f}s, OK: {'YES' if converged else 'NO'}")
        return {"converged": converged, "ne_err": ne_err, "Te_err": Te_err, "Ti_err": Ti_err, "time": dt}
    except Exception as ex:
        dt = time.perf_counter() - t0
        print(f"  FAILED: {type(ex).__name__}: {str(ex)[:100]}")
        return {"converged": False, "time": dt, "error": str(ex)}

r6_good = try_normed_curvefit("Good guess", [np.log10(5e24), 200, 200, 0], BOUNDS_LOG)
print()
r6_bad = try_normed_curvefit("Bad guess", [23, 50, 50, 0], BOUNDS_LOG)
print()
r6_vbad = try_normed_curvefit("Very bad", [27, 2000, 2000, 0], BOUNDS_LOG)
print()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print()
print(f"{'Method':<50} {'Good':<8} {'Bad':<8} {'VBad':<8} {'Wrong':<8}")
print("-" * 82)

def yn(r):
    if isinstance(r, dict) and r.get("converged"):
        return "YES"
    return "NO"

print(f"{'1. Raw curve_fit (design doc spec)':<50} {yn(r1_good):<8} {yn(r1_bad):<8} {yn(r1_vbad):<8} {yn(r1_wrong):<8}")
print(f"{'2. Log-scaled curve_fit':<50} {yn(r2_good):<8} {yn(r2_bad):<8} {yn(r2_vbad):<8} {yn(r2_wrong):<8}")
print(f"{'3. Differential Evolution':<50} {'---':<8} {'---':<8} {'---':<8} {yn({'converged': conv_de}):<8}")
print(f"{'4. Grid search + log curve_fit':<50} {'---':<8} {'---':<8} {'---':<8} {yn(r4):<8}")
print(f"{'5. Feature detection + log curve_fit':<50} {'---':<8} {'---':<8} {'---':<8} {yn(r5):<8}")
print(f"{'6. Normalized + log curve_fit (good)':<50} {yn(r6_good):<8} {yn(r6_bad):<8} {yn(r6_vbad):<8} {'---':<8}")

print()
# Diagnosis
raw_ok = sum(1 for r in [r1_good, r1_bad, r1_vbad, r1_wrong] if r.get("converged"))
log_ok = sum(1 for r in [r2_good, r2_bad, r2_vbad, r2_wrong] if r.get("converged"))
print(f"Raw curve_fit: {raw_ok}/4 converged")
print(f"Log curve_fit: {log_ok}/4 converged")
print(f"Differential Evolution: {'YES' if conv_de else 'NO'}")
print(f"Two-stage (grid+log): {'YES' if r4.get('converged') else 'NO'} ({dt_total_4:.1f}s)")
print(f"Feature+log: {'YES' if r5.get('converged') else 'NO'} ({dt_feat:.1f}s)")

print()
if raw_ok < 3:
    print("CONCLUSION: Raw curve_fit is FRAGILE in the collective regime (alpha > 2).")
    print("ROOT CAUSE: The chi-squared landscape has:")
    print("  1. Extreme parameter scale mismatch (ne~1e25 vs Te~300)")
    print("  2. Multiple local minima from ion acoustic + electron features")
    print("  3. ne-Te degeneracy (different ne/Te combos produce similar alpha)")
    print()
    print("RECOMMENDED FIX for fit_te_ne_v():")
    print("  1. ALWAYS use log10(ne) parameterization (not raw ne)")
    print("  2. Normalize spectrum to peak=1 before fitting")
    print("  3. For alpha > 1: use feature detection to set initial guess")
    print("     - Ion acoustic peak location -> Te estimate")
    print("     - Total spectral power -> ne estimate")
    print("  4. For production: two-stage (coarse grid + log curve_fit)")
    print("     or differential_evolution with polish=True")
    print("  5. Return chi2/dof as a fit quality metric")
