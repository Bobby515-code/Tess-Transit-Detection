"""
Phase 0 — Synthetic stand-in for real TESS data.

Generates a light curve with the same *shape* of realism as real TESS data:
  - correct 2-minute cadence
  - a ~1 day data-downlink gap mid-sector (real TESS sectors have this)
  - correlated "red" noise mimicking stellar rotation/spot variability
    (an Ornstein-Uhlenbeck process -- same family of behavior a GP with a
    rotation kernel is designed to model, without hard-coding celerite2 yet)
  - white photometric noise
  - a real Mandel & Agol transit shape via `batman`, injected at known
    truth values so we can check recovery later

Output has the identical schema as download_data.py's output
(time, flux, flux_err) so every later phase works on either real or
synthetic data without modification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import batman

rng = np.random.default_rng(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- 1. Time array: one TESS sector, 2-min cadence, with the downlink gap --
CADENCE_DAYS = 2.0 / (24 * 60)          # 2 minutes
SECTOR_LENGTH_DAYS = 25.0
t = np.arange(0, SECTOR_LENGTH_DAYS, CADENCE_DAYS)
gap = (t > 12.0) & (t < 13.0)           # ~1 day downlink gap mid-sector
t = t[~gap]

# ---- 2. Injected transit (truth values -- WASP-18 b-like hot Jupiter) ------
TRUTH = dict(
    per=0.94124,     # days
    rp=0.0965,       # Rp/Rs
    a=3.57,           # a/Rs
    inc=86.0,        # degrees
    ecc=0.0,
    w=90.0,
    u=[0.30, 0.25],  # quadratic limb darkening
    t0=0.30,
)

params = batman.TransitParams()
params.t0, params.per, params.rp = TRUTH["t0"], TRUTH["per"], TRUTH["rp"]
params.a, params.inc, params.ecc = TRUTH["a"], TRUTH["inc"], TRUTH["ecc"]
params.w, params.limb_dark, params.u = TRUTH["w"], "quadratic", TRUTH["u"]

m = batman.TransitModel(params, t)
transit_flux = m.light_curve(params)     # ~1.0 out of transit, dips at transit

# ---- 3. Correlated "red" noise (Ornstein-Uhlenbeck: mimics starspot/rotation
#         variability -- the kind of signal a GP rotation kernel targets) ----
def ou_process(t, tau_days, amplitude, rng):
    dt = np.diff(t, prepend=t[0])
    x = np.zeros_like(t)
    for i in range(1, len(t)):
        theta = dt[i] / tau_days
        x[i] = x[i - 1] * np.exp(-theta) + amplitude * np.sqrt(1 - np.exp(-2 * theta)) * rng.normal()
    return x

red_noise = ou_process(t, tau_days=3.0, amplitude=0.0015, rng=rng)  # ~1500 ppm rotation signal

# ---- 4. White photometric noise (bright star, ~200 ppm) --------------------
white_noise = rng.normal(0, 0.0002, size=t.size)

# ---- 5. Combine ------------------------------------------------------------
flux = transit_flux * (1 + red_noise) + white_noise
flux_err = np.full_like(flux, 0.0002)

# ---- Save in the same schema real data will use -----------------------------
import pandas as pd
df = pd.DataFrame({"time": t, "flux": flux, "flux_err": flux_err})
out_csv = os.path.join(DATA_DIR, "SIM_WASP18-like.csv")
df.to_csv(out_csv, index=False)

# ---- Plot -------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 7))

axes[0].plot(t, flux, ".", color="black", ms=1.5)
axes[0].set_xlabel("Time [days]")
axes[0].set_ylabel("Relative flux")
axes[0].set_title("Simulated TESS-like light curve (transit + stellar variability + noise)")

phase = ((t - TRUTH["t0"] + 0.5 * TRUTH["per"]) % TRUTH["per"]) - 0.5 * TRUTH["per"]
order = np.argsort(phase)
axes[1].plot(phase[order] * 24, flux[order], ".", color="gray", ms=2, alpha=0.4, label="raw (folded)")
axes[1].set_xlim(-3, 3)
axes[1].set_xlabel("Hours from transit center")
axes[1].set_ylabel("Relative flux")
axes[1].set_title(f"Phase-folded on truth period = {TRUTH['per']:.5f} d (transit still buried in variability)")
axes[1].legend()

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "phase0_simulated_lightcurve.png")
plt.savefig(plot_path, dpi=140)

print(f"Generated {len(t)} cadences over {SECTOR_LENGTH_DAYS:.0f} days (with downlink gap).")
print(f"Injected transit truth: {TRUTH}")
print(f"Saved data  -> {out_csv}")
print(f"Saved plot  -> {plot_path}")
