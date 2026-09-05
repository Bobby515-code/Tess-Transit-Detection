"""
Phase 4b - TLS (Transit Least Squares) on Pi Mensae.

Plain BLS (transit_search.py) couldn't distinguish Pi Men c's real ~6.25-day
signal from its own bias toward long periods. TLS fits an actual physical
transit shape (not just a box) and reports a properly-normalized detection
statistic (SDE) specifically designed to avoid that bias -- this checks
whether that actually helps recover the real planet.

Data is binned to 10-minute cadence first: TLS fits a full transit template
per trial period/duration, which is far more expensive per grid point than
BLS's simple box.

IMPORTANT (Windows): TLS uses multiprocessing internally to parallelize the
search across CPU cores. On Windows, new worker processes are created by
re-importing this file from scratch (unlike Linux/Mac, which just forks the
running process) -- so everything that actually RUNS must live inside the
`if __name__ == "__main__":` guard below, or each spawned worker re-triggers
the whole search itself, spawning more workers, recursively.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

LITERATURE_PERIOD = 6.25  # Huang et al. 2018 discovery paper


def bin_lightcurve(t, flux, bin_minutes=10):
    bin_width = bin_minutes / 1440
    bins = np.arange(t.min(), t.max() + bin_width, bin_width)
    idx = np.digitize(t, bins)
    bt, bf = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() > 0:
            bt.append(t[m].mean())
            bf.append(flux[m].mean())
    return np.array(bt), np.array(bf)


def run_tls(name, col, suffix):
    df = pd.read_csv(os.path.join(DATA_DIR, f"Pi_Mensae_flattened{suffix}.csv"))
    t, flux = df["time"].values, df[col].values
    good = np.isfinite(flux)
    bt, bf = bin_lightcurve(t[good], flux[good], bin_minutes=10)

    model = transitleastsquares(bt, bf, verbose=False)
    results = model.power(period_min=1, period_max=10, n_transits_min=2,
                           oversampling_factor=1, duration_grid_step=1.4,
                           show_progress_bar=False)

    print(f"\n--- Pi Mensae ({name}, TLS) ---")
    print(f"  Best period: {results.period:.4f} d  (literature: {LITERATURE_PERIOD} d)")
    print(f"  SDE: {results.SDE:.2f}   depth: {(1-results.depth)*1e6:.0f} ppm")
    print(f"  Odd/even depth mismatch: {results.odd_even_mismatch:.2f} sigma "
          f"(large values suggest a false positive, not a real planet)")
    print(f"  Transits found: {results.distinct_transit_count}, "
          f"empty (no data): {results.empty_transit_count}")

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_conv = run_tls("Conventional", "flux_flat", "")
    results_gp = run_tls("GP", "flux_flat_gp", "_gp")

    # ---- SDE periodograms ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, results, label in zip(axes, [results_conv, results_gp], ["Conventional", "GP"]):
        ax.plot(results.periods, results.power, lw=0.7, color="black")
        ax.axvline(LITERATURE_PERIOD, color="crimson", ls="--", lw=1, label=f"literature P={LITERATURE_PERIOD}d")
        ax.axvline(results.period, color="teal", ls=":", lw=1.2, label=f"found P={results.period:.3f}d")
        ax.set_title(f"Pi Mensae -- {label} (TLS, SDE={results.SDE:.1f})")
        ax.set_xlabel("Period [days]"); ax.set_ylabel("SDE")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "phase4b_tls_periodograms.png")
    plt.savefig(out1, dpi=140)
    print(f"\nSaved {out1}")

    # ---- Phase-folded at TLS's own best-fit model ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, results, label in zip(axes, [results_conv, results_gp], ["Conventional", "GP"]):
        ax.plot(results.folded_phase, results.folded_y, ".", color="gray", ms=2, alpha=0.4)
        order = np.argsort(results.model_folded_phase)
        ax.plot(np.array(results.model_folded_phase)[order], np.array(results.model_folded_model)[order],
                 "-", color="crimson", lw=1.5)
        ax.set_title(f"Pi Mensae -- {label} folded @ TLS best P={results.period:.3f}d")
        ax.set_xlabel("Phase"); ax.set_ylabel("Relative flux")
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "phase4b_tls_folded.png")
    plt.savefig(out2, dpi=140)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
