"""
Phase 3 - GP detrending (celerite2, rotation kernel).

Instead of sliding a filter window through the data (Phase 2's approach),
this models the star's variability directly as a smooth, structured signal
using a Gaussian Process with a "rotation" kernel -- built specifically to
capture the quasi-periodic brightness modulation caused by starspots
rotating in and out of view. This is the direct rival method we're
comparing against Phase 2's wotan output.

Same rules as Phase 2, for a fair comparison:
  - same outlier clipping (wotan.slide_clip)
  - same data-driven transit mask for WASP-18 (deep enough to flag directly)
  - no mask yet for Pi Mensae (transit too shallow -- Phase 4 will fix this)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import minimize
import celerite2
from celerite2 import terms
import wotan

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_transit_mask(flux, sigma=4):
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) * 1.4826
    raw_mask = flux < (med - sigma * mad)
    return np.convolve(raw_mask.astype(float), np.ones(5), mode="same") > 0


def estimate_rotation_period(t, flux, min_p=0.3, max_p=27):
    ls = LombScargle(t, flux)
    freq, power = ls.autopower(minimum_frequency=1 / max_p, maximum_frequency=1 / min_p)
    return 1 / freq[np.argmax(power)]


def neg_log_like(params, t, y, yerr):
    sigma, period, Q0, dQ, f, jitter = params
    if not (0 < f < 1 and sigma > 0 and period > 0 and Q0 > 0 and dQ > 0 and jitter >= 0):
        return 1e10
    try:
        kernel = terms.RotationTerm(sigma=sigma, period=period, Q0=Q0, dQ=dQ, f=f)
        gp = celerite2.GaussianProcess(kernel, mean=1.0)
        gp.compute(t, diag=yerr ** 2 + jitter ** 2, check_sorted=False)
        return -gp.log_likelihood(y)
    except Exception:
        return 1e10


def gp_detrend(name, build_mask=False, mask_sigma=4):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    t, flux, ferr = df["time"].values, df["flux"].values, df["flux_err"].values

    # Step 1: same outlier clipping as Phase 2
    clipped = wotan.slide_clip(t, flux, window_length=1.0, low=5, high=5, method="mad")
    good = ~np.isnan(clipped)
    t, flux, ferr = t[good], flux[good], ferr[good]

    # Step 2: same data-driven transit mask as Phase 2 (WASP-18 only)
    mask = build_transit_mask(flux, sigma=mask_sigma) if build_mask else np.zeros(len(flux), dtype=bool)
    fit_t, fit_flux, fit_ferr = t[~mask], flux[~mask], ferr[~mask]

    print(f"\n--- {name} ---")
    guess_period = estimate_rotation_period(fit_t, fit_flux)
    print(f"  Lomb-Scargle rotation period guess: {guess_period:.2f} d")

    x0 = [np.std(fit_flux), guess_period, 1.0, 1.0, 0.5, np.median(fit_ferr)]
    bounds = [(1e-6, 0.05), (0.1, 30), (1e-2, 50), (1e-2, 50), (0.02, 0.98), (0, 0.01)]

    t_start = time.time()
    soln = minimize(neg_log_like, x0, args=(fit_t, fit_flux, fit_ferr),
                     bounds=bounds, method="L-BFGS-B")
    fit_time = time.time() - t_start

    sigma, period, Q0, dQ, f, jitter = soln.x
    print(f"  Fitted period: {period:.3f} d, sigma: {sigma*1e6:.0f} ppm, jitter: {jitter*1e6:.0f} ppm")
    print(f"  GP fit took {fit_time:.1f} s ({len(fit_t)} points)")

    kernel = terms.RotationTerm(sigma=sigma, period=period, Q0=Q0, dQ=dQ, f=f)
    gp = celerite2.GaussianProcess(kernel, mean=1.0)
    gp.compute(fit_t, diag=fit_ferr ** 2 + jitter ** 2, check_sorted=False)

    # Predict the smooth trend at EVERY original time point (including masked
    # transits), conditioned only on the out-of-transit training points.
    trend = gp.predict(fit_flux, t=t, include_mean=True)
    flat_flux = flux - trend + 1.0

    out_of_transit = ~mask
    raw_std = np.std(flux[out_of_transit])
    flat_std = np.std(flat_flux[out_of_transit])
    print(f"  Out-of-transit scatter: {raw_std*1e6:.0f} ppm (raw) -> {flat_std*1e6:.0f} ppm (GP detrended)")

    return dict(t=t, flux=flux, ferr=ferr, flat=flat_flux, trend=trend,
                mask=mask, fit_time=fit_time, period=period)


results = {}
results["WASP-18"] = gp_detrend("WASP-18", build_mask=True, mask_sigma=4)
results["Pi_Mensae"] = gp_detrend("Pi_Mensae", build_mask=False)

# ---- Plot: GP trend + flattened, zoomed, same layout as Phase 2 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col, (name, zoom_days) in enumerate([("WASP-18", 10), ("Pi_Mensae", 14)]):
    r = results[name]
    t0 = r["t"].min()
    m = (r["t"] > t0) & (r["t"] < t0 + zoom_days)

    axes[0, col].plot(r["t"][m], r["flux"][m], ".", color="black", ms=2, label="raw")
    axes[0, col].plot(r["t"][m], r["trend"][m], "-", color="teal", lw=1.5, label="GP mean")
    if r["mask"] is not None and r["mask"].any():
        mm = m & r["mask"]
        axes[0, col].plot(r["t"][mm], r["flux"][mm], ".", color="orange", ms=3, label="masked (in-transit)")
    axes[0, col].set_title(f"{name}: raw + GP-fitted stellar trend")
    axes[0, col].legend(fontsize=8)
    axes[0, col].set_xlabel("Time [BTJD]"); axes[0, col].set_ylabel("Relative flux")

    axes[1, col].plot(r["t"][m], r["flat"][m], ".", color="black", ms=2)
    axes[1, col].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[1, col].set_title(f"{name}: after GP detrending (flattened)")
    axes[1, col].set_xlabel("Time [BTJD]"); axes[1, col].set_ylabel("Relative flux")

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "phase3_gp_detrending.png")
plt.savefig(out_path, dpi=140)
print(f"\nSaved {out_path}")

for name, r in results.items():
    out_csv = os.path.join(DATA_DIR, f"{name}_flattened_gp.csv")
    pd.DataFrame({"time": r["t"], "flux_raw": r["flux"], "flux_err": r["ferr"],
                  "trend_gp": r["trend"], "flux_flat_gp": r["flat"]}).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
