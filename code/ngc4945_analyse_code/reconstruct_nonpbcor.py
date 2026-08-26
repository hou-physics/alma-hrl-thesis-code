"""One-shot helper: reconstruct nonpbcor = pbcor × PB for CO and H30α.

The existing NGC 4945 work_dir has pbcor + PB ("flux") cubes but no separate
nonpbcor files (CASA's .image output wasn't kept). Phase Y `_step3` pipeline
needs both pbcor (for flux integration) and nonpbcor (for mask + noise).

Run once:
    /opt/anaconda3/envs/casa_env/bin/python reconstruct_nonpbcor.py

Outputs two new FITS files in work_dir/NGC4945/:
    NGC4945_CO_nonpbcor.fits
    NGC4945_H30a_nonpbcor.fits
"""
from pathlib import Path
from astropy.io import fits

WORKDIR = Path("/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945")

pairs = [
    ("CO",   WORKDIR / "NGC4945_CO_pbcor.fits",
             WORKDIR / "NGC4945_CO_flux.fits",
             WORKDIR / "NGC4945_CO_nonpbcor.fits"),
    ("H30a", WORKDIR / "NGC4945_H30a_spw1_v1_contsub.fits",
             WORKDIR / "NGC4945_H30a_spw1_flux.fits",
             WORKDIR / "NGC4945_H30a_nonpbcor.fits"),
]

for label, pbcor_path, pb_path, out_path in pairs:
    if out_path.exists():
        print(f"{label}: {out_path.name} already exists, skipping")
        continue
    with fits.open(pbcor_path) as h_pbcor, fits.open(pb_path) as h_pb:
        data = h_pbcor[0].data * h_pb[0].data
        fits.PrimaryHDU(data=data, header=h_pbcor[0].header).writeto(out_path)
    print(f"{label}: wrote {out_path.name}  shape={data.shape}")
