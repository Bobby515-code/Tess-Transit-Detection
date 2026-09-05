"""
Phase 4 - Transit search (Box Least Squares).

Runs a BLIND period search -- no hardcoded periods -- on all four detrended
light curves from Phase 2 and Phase 3 (conventional vs GP, WASP-18 vs
Pi Mensae), and compares what each recovers against real published values.

A note on the search grid: a naive search checks for aliasing across the
ENTIRE observing baseline, which explodes in size for a sub-day period
against WASP-18's 7-year baseline (millions of grid points -> out of
memory). We use a fixed, manually-sized log-spaced grid instead -- a
standard practical compromise, at the cost of not being mathematically
guaranteed alias-free at the very finest resolution.

Published reference values (for comparison only -- NOT used in the search):
  WASP-18 b : P = 0.94145252 d   (Sodickson et al. 2026, arXiv:2606.02473)
  Pi Men c  : P = 6.25 d          (Huang et al. 2018, discovery paper)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LITERATURE_PERIOD = {"WASP-18": 0.94145252, "Pi_Mensae": 6.25}

PERIOD_GRID = np.geomspace(0.4, 15, 20000)
DURATION_GRID = np.linspace(0.02, 0.25, 5)


def search(name, method_label, flux_col):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_flattened{'_gp' if method_label=='GP' else ''}.csv"))
    t, flux, ferr = df["time"].values, df[flux_col].values, df["flux_err"].values

    model = BoxLeastSquares(t, flux, dy=ferr)
    result = model.power(PERIOD_GRID, DURATION_GRID, method="fast")
    best = np.argmax(result.power)

    found = dict(
        period=result.period[best], t0=result.transit_time[best],
        duration=result.duration[best], depth=result.depth[best],
        snr=result.depth_snr[best],
    )
    lit_p = LITERATURE_PERIOD[name]
    pct_diff = 100 * abs(found["period"] - lit_p) / lit_p

    print(f"\n--- {name} ({method_label}-detrended) ---")
    print(f"  Found period: {found['period']:.6f} d   (literature: {lit_p} d, {pct_diff:.4f}% off)")
    print(f"  Depth: {found['depth']*1e6:.0f} ppm   SNR: {found['snr']:.1f}")

    return dict(t=t, flux=flux, result=result, best=best, found=found, pct_diff=pct_diff)


runs = {}
runs[("WASP-18", "Conventional")] = search("WASP-18", "Conventional", "flux_flat")
runs[("WASP-18", "GP")] = search("WASP-18", "GP", "flux_flat_gp")
runs[("Pi_Mensae", "Conventional")] = search("Pi_Mensae", "Conventional", "flux_flat")
runs[("Pi_Mensae", "GP")] = search("Pi_Mensae", "GP", "flux_flat_gp")

# ---- Periodograms: 2x2, literature period marked ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, target in enumerate(["WASP-18", "Pi_Mensae"]):
    for j, method in enumerate(["Conventional", "GP"]):
        r = runs[(target, method)]
        ax = axes[i, j]
        ax.plot(r["result"].period, r["result"].power, lw=0.6, color="black")
        ax.axvline(LITERATURE_PERIOD[target], color="crimson", ls="--", lw=1,
                   label=f"literature P={LITERATURE_PERIOD[target]}d")
        ax.axvline(r["found"]["period"], color="teal", ls=":", lw=1.2,
                   label=f"found P={r['found']['period']:.4f}d")
        ax.set_xscale("log")
        ax.set_title(f"{target} -- {method}")
        ax.set_xlabel("Period [days]"); ax.set_ylabel("BLS power")
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase4_periodograms.png"), dpi=140)
print(f"\nSaved {os.path.join(RESULTS_DIR, 'phase4_periodograms.png')}")

# ---- Phase-folded views at each run's OWN found period ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, target in enumerate(["WASP-18", "Pi_Mensae"]):
    for j, method in enumerate(["Conventional", "GP"]):
        r = runs[(target, method)]
        p, t0 = r["found"]["period"], r["found"]["t0"]
        phase = ((r["t"] - t0 + 0.5 * p) % p) - 0.5 * p
        ax = axes[i, j]
        ax.plot(phase * 24, r["flux"], ".", color="gray", ms=1.5, alpha=0.3)
        # binned curve for visibility
        bins = np.linspace(-0.5 * p * 24, 0.5 * p * 24, 60)
        idx = np.digitize(phase * 24, bins)
        bin_means = [r["flux"][idx == k].mean() for k in range(1, len(bins)) if np.any(idx == k)]
        bin_centers = [(bins[k-1]+bins[k])/2 for k in range(1, len(bins)) if np.any(idx == k)]
        ax.plot(bin_centers, bin_means, "-", color="crimson", lw=1.5)
        ax.set_xlim(-r["found"]["duration"]*24*3, r["found"]["duration"]*24*3)
        ax.set_title(f"{target} -- {method} (folded @ found P, SNR={r['found']['snr']:.1f})")
        ax.set_xlabel("Hours from mid-transit"); ax.set_ylabel("Relative flux")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase4_phase_folded.png"), dpi=140)
print(f"Saved {os.path.join(RESULTS_DIR, 'phase4_phase_folded.png')}")

print("\n=== SUMMARY ===")
print(f"{'Target':<12}{'Method':<14}{'Found P (d)':<14}{'% off lit.':<12}{'Depth (ppm)':<14}{'SNR':<8}")
for (target, method), r in runs.items():
    f = r["found"]
    print(f"{target:<12}{method:<14}{f['period']:<14.5f}{r['pct_diff']:<12.4f}{f['depth']*1e6:<14.0f}{f['snr']:<8.1f}")
