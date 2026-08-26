##############################################################################
# Step -1 (Plan B, modern-pipeline FITS): M83 / NGC 5236 download
#
# Pivoted here after Plan A (2013.1.01161.S scenario-C re-imaging) was blocked
# by CASA-version incompatibility: scriptForCalibration.py hard-asserts CASA
# 4.3.1 and uses Python-2 syntax; our casa_env has CASA 6.6.6.17. See
# work_dir/NGC5236_reimaging/stage_log.md for details.
#
# This alternative uses project 2015.1.00121.S, mousid uid://A001/X2fe/X34
# (12-m compact, 1.07" beam, 6.46 mJy/bm) — same observing parameter class
# as Plan A's X327, but delivered with modern ARI-L pre-calibrated FITS so
# we skip calibration entirely and go straight to Phase Y.
#
# Download just the 2 science SPWs we need (H30α + CO21), skipping continuum
# SPW2/3. Total ~43 GB (full mousid is 85 GB).
#
# SPW mapping (from earlier probe):
#   spw0_230247MHz -> CO21 (obs 230.147 GHz)
#   spw1_231818MHz -> H30a (obs 231.507 GHz)
#   spw2_244516MHz -> continuum, skip
#   spw3_246191MHz -> continuum, skip
##############################################################################

import re
from pathlib import Path
from astroquery.alma import Alma

MOUSID = "uid://A001/X2fe/X34"
# Match only spw0 + spw1, pbcor + pb.fits.gz. Skip spw2/spw3 continuum SPWs.
REGEX = r"spw[01]_[0-9]+MHz\..*\.cube\.I\.(pbcor\.fits|pb\.fits\.gz)$"
DEST = Path("/Volumes/HouAstro/master/master_thesis/work_dir/NGC5236")
DEST.mkdir(parents=True, exist_ok=True)

print(f"Probing data info for {MOUSID} ...")
alma = Alma()
info = alma.get_data_info(MOUSID, expand_tarfiles=True)

matched = [(str(r["access_url"]), int(r["content_length"]) if r["content_length"] else 0)
           for r in info if re.search(REGEX, str(r["access_url"]))]
matched_urls = [u for u, _ in matched]
total_gb = sum(sz for _, sz in matched) / 1e9

urls_txt = DEST / "download_urls.txt"
with open(urls_txt, "w") as f:
    for u in matched_urls:
        f.write(f"{u}\n")
print(f"Fallback URL list: {urls_txt}")

print(f"\nMatched {len(matched_urls)} URLs, total {total_gb:.2f} GB.")
for u, sz in matched:
    print(f"  {sz/1e9:6.2f} GB  {u.rsplit('/', 1)[-1][:80]}")
print(f"\nDownloading to {DEST} ...")

alma.download_files(
    matched_urls,
    savedir=str(DEST),
    cache=True,
    continuation=True,
)

# ------------------------------------------------------------------
# Post-processing: reconstruct non-pbcor = pbcor × pb (Phase Y expects both)
# ------------------------------------------------------------------
print("\nReconstructing non-pbcor cubes = pbcor × pb ...")
import gzip
import shutil
from astropy.io import fits

for pbcor_path in sorted(DEST.glob("*_sci.*.cube.I.pbcor.fits")):
    stem = pbcor_path.name.replace(".cube.I.pbcor.fits", "")
    pb_gz_path = DEST / f"{stem}.cube.I.pb.fits.gz"
    pb_path = DEST / f"{stem}.cube.I.pb.fits"
    nonpbcor_path = DEST / f"{stem}.cube.I.nonpbcor.fits"
    if not pb_gz_path.exists():
        print(f"  SKIP {pbcor_path.name}: matching pb file not found")
        continue
    if nonpbcor_path.exists():
        print(f"  SKIP {nonpbcor_path.name}: already exists")
        continue
    if not pb_path.exists():
        print(f"  decompressing {pb_gz_path.name} ...")
        with gzip.open(pb_gz_path, "rb") as src, open(pb_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    with fits.open(pbcor_path, memmap=True) as h_pbcor, fits.open(pb_path, memmap=True) as h_pb:
        pbcor_data = h_pbcor[0].data
        pb_data = h_pb[0].data
        header = h_pbcor[0].header.copy()
        shape = pbcor_data.shape
        shdu = fits.StreamingHDU(str(nonpbcor_path), header)
        try:
            if len(shape) == 4:
                for i in range(shape[1]):
                    shdu.write(pbcor_data[0, i] * pb_data[0, i])
            elif len(shape) == 3:
                for i in range(shape[0]):
                    shdu.write(pbcor_data[i] * pb_data[i])
            else:
                raise ValueError(f"Unexpected cube shape {shape}")
        finally:
            shdu.close()
    print(f"  wrote {nonpbcor_path.name}")
    pb_path.unlink()
    print(f"  cleaned up {pb_path.name} (kept compressed .gz)")

# ------------------------------------------------------------------
# Create canonical symlinks NGC5236_{H30a,CO21}_{pbcor,nonpbcor,pb.fits.gz}
# ------------------------------------------------------------------
import numpy as np

Z = 0.0017
GALAXY_SAFE = "NGC5236"
LINE_TABLE = [("H30a", 231.90092784), ("CO21", 230.538)]


def _spw_freq_range_hz(fpath):
    with fits.open(fpath) as h:
        hdr = h[0].header
        f0 = float(hdr["CRVAL3"])
        df = float(hdr["CDELT3"])
        n = int(hdr["NAXIS3"])
        crpix = float(hdr["CRPIX3"])
        freqs = f0 + (np.arange(n) + 1 - crpix) * df
    return float(min(freqs[0], freqs[-1])), float(max(freqs[0], freqs[-1]))


print(f"\nDetecting lines in each SPW (z={Z}) ...")
mapping = []
for pbcor_path in sorted(DEST.glob("*_sci.*.cube.I.pbcor.fits")):
    f_min, f_max = _spw_freq_range_hz(pbcor_path)
    matched_line = None
    for line_name, rest_ghz in LINE_TABLE:
        obs_hz = (rest_ghz / (1.0 + Z)) * 1e9
        if f_min <= obs_hz <= f_max:
            matched_line = line_name
            break
    if not matched_line:
        print(f"  SKIP {pbcor_path.name}: no configured line in range")
        continue
    stem = pbcor_path.name.replace(".cube.I.pbcor.fits", "")
    for old_suffix, new_suffix in [
        (".cube.I.pbcor.fits", "_pbcor.fits"),
        (".cube.I.nonpbcor.fits", "_nonpbcor.fits"),
        (".cube.I.pb.fits.gz", "_pb.fits.gz"),
    ]:
        old = DEST / f"{stem}{old_suffix}"
        new = DEST / f"{GALAXY_SAFE}_{matched_line}{new_suffix}"
        if not old.exists():
            continue
        if new.is_symlink():
            if new.resolve() == old.resolve():
                continue
            new.unlink()
        elif new.exists():
            print(f"  SKIP {new.name}: non-symlink file exists, refusing overwrite")
            continue
        new.symlink_to(old.name)
        mapping.append((new.name, old.name))
        print(f"  [{matched_line}] {new.name} -> {old.name[:70]}...")

mapping_path = DEST / "file_mapping.txt"
with open(mapping_path, "w") as f:
    f.write(f"# Generated by step-1_download_plan_b.py\n")
    f.write(f"# Galaxy: {GALAXY_SAFE}  z={Z}  Plan B mousid={MOUSID}\n")
    f.write(f"# short_name  ->  original_name\n")
    for short, orig in mapping:
        f.write(f"{short}  ->  {orig}\n")
print(f"\nFile mapping saved: {mapping_path}")

print("\nFinal files:")
for p in sorted(DEST.iterdir()):
    if p.is_file() or p.is_symlink():
        try:
            sz = p.stat().st_size / 1e9
            print(f"  {sz:6.2f} GB  {p.name}")
        except Exception:
            print(f"  (link?)  {p.name}")
