"""
Phase 0 - Real data acquisition.

RUN THIS ON YOUR OWN MACHINE, not in a restricted sandbox: it needs to reach
MAST (archive.stsci.edu / mast.stsci.edu), which is where TESS data actually
lives.

Usage:
    python download_data.py
"""

import os
import lightkurve as lk
import numpy as np
import pandas as pd

# MAST's server gives up after 600s by default -- too short for targets with
# many sectors (like Pi Mensae, which TESS observes almost continuously).
# Give it much more room.
from astroquery.mast import Conf as MastConf
MastConf.timeout = 1800  # 30 minutes

# ---- Target list -----------------------------------------------------------
TARGETS = {
    "WASP-18": {"tic": "TIC 100100827", "note": "hot Jupiter, deep/short transit", "max_sectors": 6},
    "Pi Mensae": {"tic": "TIC 261136679", "note": "shallow, longer period, many sectors (CVZ)", "max_sectors": 6},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_target(name, tic_id, max_sectors=None):
    print(f"\n--- {name} ({tic_id}) ---")

    # exptime="short" = 1-2 minute cadence only. This skips the duplicate
    # 20-second "fast" cadence products that recent sectors also have --
    # same data at coarser time resolution, half the files to download.
    search = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC", exptime="short")
    if len(search) == 0:
        print("  No SPOC light curves found for this target.")
        return None

    sectors = list(dict.fromkeys(search.table["mission"]))
    print(f"  Found {len(search)} product(s) across sectors: {sectors}")

    if max_sectors is not None and len(search) > max_sectors:
        print(f"  Limiting to first {max_sectors} sector(s) for now "
              f"(raise max_sectors in the TARGETS dict to get more).")
        search = search[:max_sectors]

    # Download one sector at a time instead of all-at-once: if one sector is
    # slow or fails, the rest still succeed instead of losing everything.
    lcs = []
    for i, row in enumerate(search):
        try:
            lc = row.download()
            lcs.append(lc)
            print(f"  Downloaded sector {i + 1}/{len(search)}")
        except Exception as e:
            print(f"  Skipped one sector due to {type(e).__name__}: {e}")

    if not lcs:
        print("  Nothing downloaded successfully.")
        return None

    lc = lk.LightCurveCollection(lcs).stitch()
    lc = lc.remove_nans()

    # Build the CSV from the underlying arrays directly. lc.to_pandas() puts
    # time in the DataFrame's index rather than a normal column, which is
    # what caused the earlier KeyError -- this sidesteps that entirely.
    df = pd.DataFrame({
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value,
    })
    out_csv = os.path.join(DATA_DIR, f"{name.replace(' ', '_')}.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved {len(df)} points -> {out_csv}")
    return lc


if __name__ == "__main__":
    for name, meta in TARGETS.items():
        try:
            download_target(name, meta["tic"], max_sectors=meta.get("max_sectors"))
        except Exception as e:
            print(f"  FAILED for {name}: {type(e).__name__}: {e}")

    print("\nDone. Each target's light curve is cached in data/<name>.csv")
    print("so you don't need to re-download every time you re-run later phases.")
