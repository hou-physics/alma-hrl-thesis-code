"""Compute CO(2-1) moment-0 to find a clean noise region away from galaxy emission.

Looks at the CO non-pbcor cube around the line-observed channels and picks
candidate noise-region centers where the CO mom0 is ~0 and PB > 0.3.
"""
import gzip
import shutil
import tempfile
import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

Z = 0.000749
CO_REST_GHZ = 230.538

# NED nucleus
RA_NED = 359.457210
DEC_NED = -32.590990

co_nonpbcor = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_CO21_nonpbcor.fits"
hrl_pb_gz = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pb.fits.gz"

print("Computing CO(2-1) moment-0 around the observed line freq...")
with fits.open(co_nonpbcor, memmap=True) as hdul:
    h = hdul[0].header
    data = hdul[0].data
    shape = data.shape
    print(f"  CO cube shape: {shape}")
    naxis3 = h["NAXIS3"]
    crval3 = h["CRVAL3"]; cdelt3 = h["CDELT3"]; crpix3 = h["CRPIX3"]
    freqs = crval3 + (np.arange(naxis3) + 1 - crpix3) * cdelt3
    obs_hz = (CO_REST_GHZ / (1 + Z)) * 1e9
    # CDELT3 ~ 0.977 MHz = ~1.27 km/s at 230 GHz, need window +-300 km/s = +-230 MHz
    # Use channels within +-80 km/s of line center for mom0 focus on bright peak
    c = 299792.458
    vel = (obs_hz - freqs) / obs_hz * c  # km/s
    mask_line = np.abs(vel) < 150.0
    print(f"  number of line channels (|v|<150 km/s): {mask_line.sum()}")
    if len(shape) == 4:
        sub = data[0, mask_line, :, :]
    else:
        sub = data[mask_line, :, :]
    mom0 = np.nansum(sub, axis=0) * (cdelt3 / 1e9)  # in units of Jy/beam * GHz (arbitrary units ok)
    print(f"  mom0 shape: {mom0.shape}  median={np.nanmedian(mom0):.4f}  max={np.nanmax(mom0):.4f}")

    wcs2 = WCS(h).celestial
    ny, nx = mom0.shape

    # Get noise level in mom0 from off-source corners
    sig = np.nanstd(mom0[50:250, 50:250])
    print(f"  mom0 noise (corner estimate) ~ {sig:.4f}")

# Open PB cube
print("\nOpening HRL PB cube...")
with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
    tmp_path = tf.name
try:
    with gzip.open(hrl_pb_gz, "rb") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    with fits.open(tmp_path, memmap=True) as hdul:
        pb_h = hdul[0].header
        pb_data = hdul[0].data
        pbshape = pb_data.shape
        if len(pbshape) == 4:
            pb2d = pb_data[0, pbshape[1] // 2]
        elif len(pbshape) == 3:
            pb2d = pb_data[pbshape[0] // 2]
        else:
            pb2d = pb_data
        wcs2_pb = WCS(pb_h).celestial

        c_ned = SkyCoord(RA_NED * u.deg, DEC_NED * u.deg)

        # Test candidate centers on a grid
        candidates = []
        for dra in range(-90, 95, 10):
            for ddec in range(-90, 95, 10):
                if abs(dra) < 30 and abs(ddec) < 30:
                    continue
                c_test = SkyCoord(c_ned.ra + (dra / 3600.0 / np.cos(c_ned.dec.rad)) * u.deg,
                                  c_ned.dec + (ddec / 3600.0) * u.deg)
                # coord in CO mom0 pixel
                xc, yc = wcs2.world_to_pixel(c_test)
                xcp, ycp = wcs2_pb.world_to_pixel(c_test)
                xi, yi = int(xc), int(yc)
                xip, yip = int(xcp), int(ycp)
                if not (0 <= xi < nx and 0 <= yi < ny): continue
                if not (0 <= xip < pb2d.shape[1] and 0 <= yip < pb2d.shape[0]): continue
                hp = int(round(10.0 / 0.1))  # 10 arcsec half-box
                y0, y1 = max(0, yi-hp), min(ny, yi+hp)
                x0, x1 = max(0, xi-hp), min(nx, xi+hp)
                patch_mom = mom0[y0:y1, x0:x1]
                y0p, y1p = max(0, yip-hp), min(pb2d.shape[0], yip+hp)
                x0p, x1p = max(0, xip-hp), min(pb2d.shape[1], xip+hp)
                patch_pb = pb2d[y0p:y1p, x0p:x1p]
                mom_mean = float(np.nanmean(np.abs(patch_mom)))
                pb_min = float(np.nanmin(patch_pb))
                pb_mean = float(np.nanmean(patch_pb))
                # Only keep candidates with PB > 0.3
                if pb_min < 0.3: continue
                candidates.append((mom_mean, pb_min, pb_mean, dra, ddec,
                                   c_test.ra.deg, c_test.dec.deg))
        # Sort: prefer low mom0, high PB
        candidates.sort(key=lambda r: (r[0], -r[2]))
        print("\nTop 10 candidates (lowest |CO mom0| within 20 arcsec box, PB_min>=0.3):")
        print(f"{'|mom0|':>10} {'PB_min':>7} {'PB_mean':>8}  dRA  dDec   RA           Dec")
        for mom_mean, pbn, pbm, dra, ddec, r, d in candidates[:15]:
            print(f"{mom_mean:>10.4f} {pbn:>7.3f} {pbm:>8.3f}  {dra:+4d} {ddec:+4d}  {r:.6f}  {d:.6f}")
finally:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
