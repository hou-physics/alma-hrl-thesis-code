"""Pre-flight header verification for NGC 7793 HRL + CO cubes.

Verify:
  - HRL observed freq 231.727 GHz is inside HRL cube freq range
  - CO observed freq  230.366 GHz is inside CO  cube freq range
  - Basic WCS / beam info for both
"""
from astropy.io import fits

Z = 0.000749
H30A_REST = 231.900928
CO21_REST = 230.538

targets = [
    ("HRL pbcor", H30A_REST,
     "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pbcor.fits"),
    ("CO pbcor", CO21_REST,
     "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_CO21_pbcor.fits"),
]

for label, rest_ghz, path in targets:
    with fits.open(path, memmap=True) as hdul:
        h = hdul[0].header
        naxis3 = h["NAXIS3"]
        crval3 = h["CRVAL3"]
        cdelt3 = h["CDELT3"]
        crpix3 = h["CRPIX3"]
        ctype3 = h["CTYPE3"]
        cunit3 = h.get("CUNIT3", "Hz")
        restfrq = h.get("RESTFRQ", h.get("RESTFREQ", None))

        f1 = crval3 + (1 - crpix3) * cdelt3
        fN = crval3 + (naxis3 - crpix3) * cdelt3
        fmin, fmax = min(f1, fN), max(f1, fN)

        obs_hz = (rest_ghz / (1.0 + Z)) * 1e9
        in_range = fmin <= obs_hz <= fmax

        print(f"--- {label} ---")
        print(f"  path: {path}")
        print(f"  NAXIS1/2/3: {h['NAXIS1']} x {h['NAXIS2']} x {naxis3}")
        print(f"  CTYPE3={ctype3}  CUNIT3={cunit3}  CDELT3={cdelt3:.4e}")
        print(f"  freq range: {fmin/1e9:.5f} - {fmax/1e9:.5f} GHz")
        print(f"  RESTFRQ:    {restfrq/1e9 if restfrq else None} GHz")
        print(f"  CRVAL1,CRVAL2 (deg): {h['CRVAL1']:.6f}, {h['CRVAL2']:.6f}")
        print(f"  CDELT1,CDELT2 (deg): {h['CDELT1']:.4e}, {h['CDELT2']:.4e}")
        print(f"  BMAJ/BMIN/BPA: {h.get('BMAJ')}, {h.get('BMIN')}, {h.get('BPA')}")
        print(f"  target obs freq: {obs_hz/1e9:.5f} GHz  -->  IN RANGE: {in_range}")
        print()
