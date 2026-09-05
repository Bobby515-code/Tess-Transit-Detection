"""
Phase 2 - Conventional detrending (wotan, biweight).

Two-step process per target:
  1. Clip obvious outliers (flares, instrument artifacts) with a sliding
     window BEFORE fitting anything -- otherwise a single flare can drag
     the fitted trend toward it.
  2. Fit and remove the slow stellar-variability trend with wotan's
     biweight filter.

For WASP-18, the transit is deep enough to flag with a simple data-driven
threshold, so we mask it out before fitting -- this stops the filter from
partially "eating" the transit signal it's supposed to preserve.

For Pi Mensae, the transit is far too shallow to find this way (that's the
whole point of the project) -- so for now we detrend WITHOUT a mask. Once
Phase 4 (transit search) finds a real candidate period, we'll come back and
redo this step with a proper mask. This mirrors how real detrending
pipelines are actually built: iteratively, not in one pass.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wotan

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def detrend_target(name, window_length, build_mask=False, mask_sigma=4):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    t, flux, ferr = df["time"].values, df["flux"].values, df["flux_err"].values

    # Step 1: outlier clipping (catches things like Pi Mensae's flares)
    clipped = wotan.slide_clip(t, flux, window_length=1.0, low=5, high=5, method="mad")
    good = ~np.isnan(clipped)
    n_clipped = int((~good).sum())
    t, flux, ferr = t[good], flux[good], ferr[good]

    # Step 2 (optional): data-driven transit mask for obviously-deep transits
    mask = None
    if build_mask:
        med = np.median(flux)
        mad = np.median(np.abs(flux - med)) * 1.4826  # robust sigma estimate
        raw_mask = flux < (med - mask_sigma * mad)
        # widen slightly to also catch ingress/egress, not just the deepest point
        mask = np.convolve(raw_mask.astype(float), np.ones(5), mode="same") > 0

    flat_flux, trend = wotan.flatten(
        t, flux, window_length=window_length, method="biweight",
        mask=mask, return_trend=True, break_tolerance=1.0, return_nsplines=False,
    )

    out_of_transit = ~mask if mask is not None else np.ones(len(flux), dtype=bool)
    raw_std = np.std(flux[out_of_transit])
    flat_std = np.nanstd(flat_flux[out_of_transit])

    print(f"\n--- {name} ---")
    print(f"  Clipped {n_clipped} outlier point(s)")
    if mask is not None:
        print(f"  Flagged {mask.sum()} point(s) as in-transit (excluded from trend fit)")
    print(f"  Out-of-transit scatter: {raw_std*1e6:.0f} ppm (raw) -> {flat_std*1e6:.0f} ppm (detrended)")

    return dict(t=t, flux=flux, ferr=ferr, flat=flat_flux, trend=trend, mask=mask)


results = {}
results["WASP-18"] = detrend_target("WASP-18", window_length=0.5, build_mask=True, mask_sigma=4)
results["Pi_Mensae"] = detrend_target("Pi_Mensae", window_length=0.5, build_mask=False)

# ---- Plot: raw+trend and flattened, zoomed, for both targets ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col, (name, zoom_days) in enumerate([("WASP-18", 10), ("Pi_Mensae", 14)]):
    r = results[name]
    t0 = r["t"].min()
    m = (r["t"] > t0) & (r["t"] < t0 + zoom_days)

    axes[0, col].plot(r["t"][m], r["flux"][m], ".", color="black", ms=2, label="raw")
    axes[0, col].plot(r["t"][m], r["trend"][m], "-", color="crimson", lw=1.5, label="fitted trend")
    if r["mask"] is not None:
        mm = m & r["mask"]
        axes[0, col].plot(r["t"][mm], r["flux"][mm], ".", color="orange", ms=3, label="masked (in-transit)")
    axes[0, col].set_title(f"{name}: raw + fitted stellar trend")
    axes[0, col].legend(fontsize=8)
    axes[0, col].set_xlabel("Time [BTJD]"); axes[0, col].set_ylabel("Relative flux")

    axes[1, col].plot(r["t"][m], r["flat"][m], ".", color="black", ms=2)
    axes[1, col].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[1, col].set_title(f"{name}: after detrending (flattened)")
    axes[1, col].set_xlabel("Time [BTJD]"); axes[1, col].set_ylabel("Relative flux")

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "phase2_conventional_detrending.png")
plt.savefig(out_path, dpi=140)
print(f"\nSaved {out_path}")

# Save flattened light curves for Phase 3/4 to use
for name, r in results.items():
    out_csv = os.path.join(DATA_DIR, f"{name}_flattened.csv")
    pd.DataFrame({"time": r["t"], "flux_raw": r["flux"], "flux_err": r["ferr"],
                  "trend": r["trend"], "flux_flat": r["flat"]}).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
