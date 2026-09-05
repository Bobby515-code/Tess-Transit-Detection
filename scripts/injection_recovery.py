"""
Phase 5 - Injection-recovery.

Injects synthetic transits (real Mandel & Agol shape via batman, not a toy
box) into Pi Mensae's REAL raw light curve at a grid of depths and periods,
then runs the FULL blind pipeline on each -- outlier clipping, detrending
(no mask, since in a real discovery we wouldn't know where to mask yet),
then a BLS search -- exactly mirroring how we actually analyzed Pi Mensae
in Phases 2-4. This turns "we couldn't detect the real planet" into a
proper measured completeness curve for each detrending method.

A trial counts as RECOVERED if the search's single best period matches the
injected period to within 1%. This is intentionally strict (real pipelines
often also credit harmonics/aliases as partial recoveries) -- documented
here as a known simplification.

Geometry is fixed (a/Rs=15, inc=89 deg, quadratic limb darkening) across all
trials so only depth and period vary -- not meant to be a precise match to
Pi Mensae's real stellar parameters, just a consistent yardstick for
comparing the two detrending methods against each other.
"""

import os
import time
import numpy as np
import pandas as pd
import batman
import wotan
import celerite2
from celerite2 import terms
from astropy.timeseries import BoxLeastSquares, LombScargle
from scipy.optimize import minimize

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DEPTHS_PPM = [100, 200, 400, 700, 1200, 2000]
PERIODS = [1.7, 2.9, 4.3, 6.9, 8.5]  # original grid (avoids the ~9.48d and ~13.7d systematics found earlier)
RNG_SEED = 42

# Extension (added after initial review flagged n=5/depth as statistically thin):
# 5 additional periods, doubling resolution to n=10 recoveries per depth point.
# Uses a SEPARATE documented seed rather than regenerating the original grid,
# so the original 60 trials remain exactly reproducible from this file.
EXTRA_PERIODS = [2.3, 3.7, 5.1, 6.1, 7.7]
EXTRA_SEED = 43


def load_base_lightcurve():
    df = pd.read_csv(os.path.join(DATA_DIR, "Pi_Mensae.csv"))
    t, flux, ferr = df["time"].values, df["flux"].values, df["flux_err"].values
    clipped = wotan.slide_clip(t, flux, window_length=1.0, low=5, high=5, method="mad")
    good = ~np.isnan(clipped)
    return t[good], flux[good], ferr[good]


def inject(t, flux, period, t0, depth_ppm):
    rp_rs = np.sqrt(depth_ppm * 1e-6)
    params = batman.TransitParams()
    params.t0, params.per, params.rp = t0, period, rp_rs
    params.a, params.inc, params.ecc, params.w = 15.0, 89.0, 0.0, 90.0
    params.u, params.limb_dark = [0.3, 0.25], "quadratic"
    model = batman.TransitModel(params, t).light_curve(params)
    measured_depth_ppm = (1 - model.min()) * 1e6
    return flux * model, measured_depth_ppm


def detrend_wotan(t, flux, ferr):
    flat, _ = wotan.flatten(t, flux, window_length=0.5, method="biweight",
                             return_trend=True, break_tolerance=1.0)
    return flat


def detrend_gp(t, flux, ferr):
    guess_period = 1 / LombScargle(t, flux).autopower(minimum_frequency=1/27, maximum_frequency=1/0.3)[0][
        np.argmax(LombScargle(t, flux).autopower(minimum_frequency=1/27, maximum_frequency=1/0.3)[1])]

    def neg_log_like(p, t, y, yerr):
        sigma, period, Q0, dQ, f, jitter = p
        if not (0 < f < 1 and sigma > 0 and period > 0 and Q0 > 0 and dQ > 0 and jitter >= 0):
            return 1e10
        try:
            kernel = terms.RotationTerm(sigma=sigma, period=period, Q0=Q0, dQ=dQ, f=f)
            gp = celerite2.GaussianProcess(kernel, mean=1.0)
            gp.compute(t, diag=yerr**2 + jitter**2, check_sorted=False)
            return -gp.log_likelihood(y)
        except Exception:
            return 1e10

    x0 = [np.std(flux), guess_period, 1.0, 1.0, 0.5, np.median(ferr)]
    bounds = [(1e-6, 0.05), (0.1, 30), (1e-2, 50), (1e-2, 50), (0.02, 0.98), (0, 0.01)]
    soln = minimize(neg_log_like, x0, args=(t, flux, ferr), bounds=bounds, method="L-BFGS-B")
    sigma, period, Q0, dQ, f, jitter = soln.x
    kernel = terms.RotationTerm(sigma=sigma, period=period, Q0=Q0, dQ=dQ, f=f)
    gp = celerite2.GaussianProcess(kernel, mean=1.0)
    gp.compute(t, diag=ferr**2 + jitter**2, check_sorted=False)
    trend = gp.predict(flux, t=t, include_mean=True)
    return flux - trend + 1.0


def search_bls(t, flux, ferr):
    periods = np.geomspace(0.4, 10, 4000)  # coarser than Phase 4 -- speed matters, we run this ~60 times
    durations = np.linspace(0.02, 0.25, 3)
    model = BoxLeastSquares(t, flux, dy=ferr)
    result = model.power(periods, durations, method="fast")
    best = np.argmax(result.power)
    return result.period[best]


def main(depths=None, periods=None, seed=None, append=False):
    """depths/periods: optional subsets (for splitting a long grid across
    multiple runs). seed: RNG seed for injection phase, defaults to
    RNG_SEED. append: append to the results CSV instead of overwriting."""
    depths = DEPTHS_PPM if depths is None else depths
    periods = PERIODS if periods is None else periods
    seed = RNG_SEED if seed is None else seed
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    t_base, flux_base, ferr_base = load_base_lightcurve()
    print(f"Base light curve: {len(t_base)} points (real Pi Mensae, outlier-clipped)")

    rows = []
    total = len(depths) * len(periods) * 2
    n = 0
    t_start = time.time()

    for depth_ppm in depths:
        for period in periods:
            t0 = t_base.min() + rng.uniform(0, period)  # random injection phase
            flux_inj, measured_depth = inject(t_base, flux_base, period, t0, depth_ppm)

            for method in ["wotan", "GP"]:
                n += 1
                if method == "wotan":
                    flat = detrend_wotan(t_base, flux_inj, ferr_base)
                else:
                    flat = detrend_gp(t_base, flux_inj, ferr_base)

                found_period = search_bls(t_base, flat, ferr_base)
                recovered = abs(found_period - period) / period < 0.01

                rows.append(dict(depth_ppm=measured_depth, period=period, method=method,
                                  found_period=found_period, recovered=recovered))
                elapsed = time.time() - t_start
                print(f"  [{n}/{total}] depth={depth_ppm}ppm P={period}d {method}: "
                      f"found={found_period:.3f}d recovered={recovered}  ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_DIR, "injection_recovery_results.csv")
    if append and os.path.exists(out_csv):
        df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)
    print(f"\nSaved (batch of {len(rows)} rows) -> {out_csv}")
    print(f"Batch time: {time.time()-t_start:.0f}s")
    return df


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion -- more reliable than
    a naive normal-approximation interval at small n or near 0%/100%,
    both of which apply here."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize():
    import matplotlib.pyplot as plt
    out_csv = os.path.join(DATA_DIR, "injection_recovery_results.csv")
    df = pd.read_csv(out_csv)

    # Group by the INTENDED target depth, not the raw measured value -- the
    # measured depth jitters by a fraction of a ppm depending on exactly
    # where the transit minimum lands on the real cadence grid, which
    # otherwise fragments the groupby into near-duplicate rows.
    df["target_depth_ppm"] = df["depth_ppm"].apply(lambda x: min(DEPTHS_PPM, key=lambda t: abs(t - x)))

    print(f"\n=== COMPLETENESS ({len(df)} total trials, recovery fraction across periods, per depth) ===")
    grouped = df.groupby(["method", "target_depth_ppm"])["recovered"].agg(["sum", "count"])
    grouped["rate"] = grouped["sum"] / grouped["count"]
    grouped["ci_lo"], grouped["ci_hi"] = zip(*grouped.apply(lambda r: wilson_ci(r["sum"], r["count"]), axis=1))
    print(grouped)
    grouped.to_csv(os.path.join(DATA_DIR, "completeness_summary.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color in [("wotan", "crimson"), ("GP", "teal")]:
        sub = grouped.xs(method, level="method").sort_index()
        rate = sub["rate"] * 100
        lo = rate - sub["ci_lo"] * 100
        hi = sub["ci_hi"] * 100 - rate
        label = "Conventional (wotan)" if method == "wotan" else "GP (celerite2)"
        ax.errorbar(sub.index, rate, yerr=[lo, hi], fmt="o-", color=color, label=label,
                    lw=2, ms=8, capsize=4, elinewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Injected transit depth [ppm]"); ax.set_ylabel("Recovery rate [%]")
    ax.set_ylim(-8, 108)
    n_per_point = grouped["count"].iloc[0]
    ax.set_title(f"Injection-recovery completeness: conventional vs GP detrending\n"
                 f"(n={n_per_point} periods/depth, error bars = 95% Wilson score CI)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "phase5_completeness_curve.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")
    return grouped


if __name__ == "__main__":
    main(periods=PERIODS, seed=RNG_SEED, append=False)
    main(periods=EXTRA_PERIODS, seed=EXTRA_SEED, append=True)
    summarize()
