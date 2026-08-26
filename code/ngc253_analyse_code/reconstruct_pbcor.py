"""One-shot helper: reconstruct pbcor = nonpbcor / PB for NGC 253.

NGC 253 work_dir only has the non-pbcor Band 3 cube + a 2D PB (mfs) image.
Phase Y `_step3` pipeline needs both pbcor and nonpbcor.

Output: NGC253_H40a_pbcor.fits (~7.4 GB, same shape as nonpbcor)

Run once:
    /opt/anaconda3/envs/casa_env/bin/python reconstruct_pbcor.py

Notes:
- PB is 2D (720×720) — broadcast across all 3838 channels (frequency
  dependence of PB across 97–99 GHz is <1%, negligible for Band 3 analysis).
- Pixels with PB < 0.2 → NaN in pbcor (1/small blows up).
"""
from pathlib import Path
import numpy as np
from astropy.io import fits

WORKDIR = Path("/Volumes/HouAstro/master/master_thesis/work_dir/NGC253")
NONPBCOR = WORKDIR / "NGC253_H40a_spw31_Xf05_v1.fits"
PB_2D = WORKDIR / "NGC253_spw31_pb_mfs.fits"
PBCOR_OUT = WORKDIR / "NGC253_H40a_pbcor.fits"

if PBCOR_OUT.exists():
    print(f"{PBCOR_OUT.name} already exists ({PBCOR_OUT.stat().st_size/1e9:.1f} GB), skipping")
else:
    print(f"Loading non-pbcor cube {NONPBCOR.name} ...")
    with fits.open(NONPBCOR) as h:
        nonpbcor_data = h[0].data.astype(np.float32)  # float32 saves disk vs float64
        nonpbcor_hdr = h[0].header.copy()
    while nonpbcor_data.ndim > 3:
        nonpbcor_data = nonpbcor_data[0]

    print(f"Loading PB {PB_2D.name} ...")
    with fits.open(PB_2D) as h:
        pb_data = h[0].data.astype(np.float32)
    while pb_data.ndim > 2:
        pb_data = pb_data[0]

    print(f"Broadcasting PB (2D {pb_data.shape}) to cube shape {nonpbcor_data.shape} ...")
    # Divide each channel by PB; mask PB<0.2 as NaN (would blow up)
    pb_safe = np.where(pb_data < 0.2, np.nan, pb_data)
    print(f"Computing pbcor = nonpbcor / PB (in memory) ...")
    pbcor_data = nonpbcor_data / pb_safe[np.newaxis, :, :]

    print(f"Writing {PBCOR_OUT.name} ...")
    fits.PrimaryHDU(data=pbcor_data, header=nonpbcor_hdr).writeto(PBCOR_OUT, overwrite=True)
    print(f"Done. {PBCOR_OUT.name} = {PBCOR_OUT.stat().st_size/1e9:.1f} GB")
