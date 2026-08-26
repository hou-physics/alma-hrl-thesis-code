"""Query NED for NGC 7793 coordinates + report ALMA pointing offset.

Also inspects HRL PB fits.gz to pick a noise region with PB>0.3 far from source.
"""
import gzip
import shutil
import tempfile
import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.ipac.ned import Ned

print("Querying NED for NGC 7793...")
tbl = Ned.query_object("NGC 7793")
ra_ned = float(tbl["RA"][0])
dec_ned = float(tbl["DEC"][0])
print(f"NED NGC 7793 coords: RA={ra_ned:.6f} deg   Dec={dec_ned:.6f} deg")
c_ned = SkyCoord(ra_ned * u.deg, dec_ned * u.deg)
print(f"  -> {c_ned.to_string('hmsdms')}")
print()

# ALMA pointing center (from pbcor header CRVAL)
pbcor_path = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pbcor.fits"
with fits.open(pbcor_path, memmap=True) as hdul:
    h = hdul[0].header
    ra_alma = float(h["CRVAL1"])
    dec_alma = float(h["CRVAL2"])
c_alma = SkyCoord(ra_alma * u.deg, dec_alma * u.deg)
sep = c_ned.separation(c_alma).arcsec
print(f"ALMA pointing (CRVAL): RA={ra_alma:.6f} deg  Dec={dec_alma:.6f} deg")
print(f"  -> {c_alma.to_string('hmsdms')}")
print(f"Offset NED_nucleus <-> ALMA pointing: {sep:.2f} arcsec")
print()

# Read the PB cube, compute average PB, find a "clean" off-source spot with PB>0.3
pb_gz = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pb.fits.gz"
print("Inspecting PB cube (decompressing transiently to a temp file)...")
with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
    tmp_path = tf.name
try:
    with gzip.open(pb_gz, "rb") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    with fits.open(tmp_path, memmap=True) as hdul:
        pb_h = hdul[0].header
        pb_data = hdul[0].data
        shape = pb_data.shape
        print(f"  PB shape: {shape}")
        # Use middle channel
        if len(shape) == 4:
            mid = shape[1] // 2
            pb2d = pb_data[0, mid]
        elif len(shape) == 3:
            mid = shape[0] // 2
            pb2d = pb_data[mid]
        else:
            pb2d = pb_data
        # Simple inspection
        nx = pb2d.shape[1]; ny = pb2d.shape[0]
        print(f"  pb slice shape: ({ny},{nx})")
        print(f"  pb [center]: {pb2d[ny//2, nx//2]}")
        # Use astropy WCS to convert
        from astropy.wcs import WCS
        wcs_full = WCS(pb_h)
        wcs2 = wcs_full.celestial
        # NED pixel coords
        xp, yp = wcs2.world_to_pixel(c_ned)
        print(f"  NED nucleus pixel in PB cube: x={float(xp):.1f}, y={float(yp):.1f}  PB={pb2d[int(yp), int(xp)]:.3f}")
        xa, ya = wcs2.world_to_pixel(c_alma)
        print(f"  ALMA pointing pixel in PB cube: x={float(xa):.1f}, y={float(ya):.1f}  PB={pb2d[int(ya), int(xa)]:.3f}")

        # Suggest a noise region: a square offset 40-50 arcsec from NED, where PB>0.3 and far from galaxy
        # Try several offsets and pick one with highest PB
        offsets_arcsec = [
            ( 40,   0), (-40,   0), (  0,  40), (  0, -40),
            ( 30,  30), (-30,  30), ( 30, -30), (-30, -30),
            ( 50,  0), (-50,  0), ( 0,  50), ( 0, -50),
            ( 40, 40), (-40,  40), ( 40, -40), (-40, -40),
            ( 60, 60), (-60,  60), ( 60, -60), (-60, -60),
            ( 80,  0), (-80,  0), ( 0,  80), ( 0, -80),
            ( 60,  0), (-60,  0), ( 0,  60), ( 0, -60),
            ( 70, 40), (-70,  40), ( 70, -40), (-70, -40),
        ]
        candidates = []
        for dra, ddec in offsets_arcsec:
            c_test = SkyCoord(c_ned.ra + (dra / 3600.0 / np.cos(c_ned.dec.rad)) * u.deg,
                              c_ned.dec + (ddec / 3600.0) * u.deg)
            xpt, ypt = wcs2.world_to_pixel(c_test)
            xi, yi = int(xpt), int(ypt)
            if 0 <= xi < nx and 0 <= yi < ny:
                # sample a 20 arcsec box (half = 10 arcsec = ~360 pix... wait CDELT 2.78e-5 deg = 0.1 arcsec)
                # CDELT 2.78e-5 deg = 0.1 arcsec/pix
                half_pix = int(round(10.0 / 0.1))
                x0 = max(0, xi - half_pix); x1 = min(nx, xi + half_pix)
                y0 = max(0, yi - half_pix); y1 = min(ny, yi + half_pix)
                patch = pb2d[y0:y1, x0:x1]
                pb_min = float(np.nanmin(patch))
                pb_mean = float(np.nanmean(patch))
                candidates.append((pb_mean, pb_min, dra, ddec, c_test.ra.deg, c_test.dec.deg))
        candidates.sort(reverse=True)
        print("\nCandidate noise-region centers (offset_arcsec_from_NED -> PB mean/min over 20 arcsec box):")
        for pbm, pbn, dra, ddec, r, d in candidates[:8]:
            print(f"  dRA={dra:+4d}  dDec={ddec:+4d}  RA={r:.6f}  Dec={d:.6f}  PB_mean={pbm:.3f}  PB_min={pbn:.3f}")
finally:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
