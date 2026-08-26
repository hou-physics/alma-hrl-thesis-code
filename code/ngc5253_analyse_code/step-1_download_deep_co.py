##############################################################################
# Step -1 (deep tier, CO leg): NGC 5253 X1a spw0 CO(2-1)-only download
#
# Discovered 2026-04-25 evening: X1a project's spw0 covers CO(2-1) at
# 230.228 GHz, exactly matching NGC 5253's redshifted CO(2-1) (230.215 GHz,
# 13 MHz / 17 km/s offset — well within the SPW). Earlier alma-query missed
# this because it defaulted to the line-name match in the project's primary
# tuning (spw1 = H30α); spw0 was downloaded only as continuum sideband.
#
# With spw0 (CO21) and spw1 (H30α) both from X1a at 0.19″ beam, we can re-do
# NGC 5253 deep analysis as a SAME-MOUSID 0.19″ run instead of cross-mousid
# (X1a HRL + X1c CO at 0.91″ common beam). No common-beam convolution
# needed → mask matched to the compact ~0.3-0.5″ central starburst →
# expected S/N 12-16 (vs current 8.01).
#
# Mousid: uid://A001/X136/X1a (12-m long-baseline, 0.19" res)
# This script downloads spw0 only (~24 GB: 16 GB pbcor + 7 GB pb.gz).
# Lands in work_dir/NGC5253_deep/ alongside H30α, creates canonical
# NGC5253_CO21_*.fits symlinks for Phase Y consumption.
##############################################################################

import re
from pathlib import Path
from astroquery.alma import Alma

MOUSID = "uid://A001/X136/X1a"
# Match only spw0 (both science and calibrator cubes in that SPW)
REGEX = r"spw0_[0-9]+MHz\..*\.cube\.I\.(pbcor\.fits|pb\.fits\.gz)$"
DEST = Path("/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep")
DEST.mkdir(parents=True, exist_ok=True)

print(f"Probing data info for {MOUSID} (spw0 / CO21 subset) ...")
alma = Alma()
info = alma.get_data_info(MOUSID, expand_tarfiles=True)

matched = [(str(r["access_url"]), int(r["content_length"]) if r["content_length"] else 0)
           for r in info if re.search(REGEX, str(r["access_url"]))]
matched_urls = [u for u, _ in matched]
total_gb = sum(sz for _, sz in matched) / 1e9

urls_txt = DEST / "download_urls_co.txt"
with open(urls_txt, "w") as f:
    for u in matched_urls:
        f.write(f"{u}\n")

print(f"Matched {len(matched_urls)} URLs, total {total_gb:.2f} GB.")
for u, sz in matched:
    print(f"  {sz/1e9:6.2f} GB  {u.rsplit('/', 1)[-1][:80]}")
print(f"\nFallback URL list: {urls_txt}")
print(f"Downloading to {DEST} ...")

alma.download_files(
    matched_urls,
    savedir=str(DEST),
    cache=True,
    continuation=True,
)

# ------------------------------------------------------------------
# Post-process: reconstruct non-pbcor + create canonical CO21 symlinks
# ------------------------------------------------------------------
print("\nReconstructing CO21 non-pbcor = pbcor × pb ...")
import gzip
import shutil
from astropy.io import fits

for pbcor_path in sorted(DEST.glob("*NGC5253_sci.spw0_*.cube.I.pbcor.fits")):
    stem = pbcor_path.name.replace(".cube.I.pbcor.fits", "")
    pb_gz_path = DEST / f"{stem}.cube.I.pb.fits.gz"
    pb_path = DEST / f"{stem}.cube.I.pb.fits"
    nonpbcor_path = DEST / f"{stem}.cube.I.nonpbcor.fits"
    if not pb_gz_path.exists():
        print(f"  SKIP {pbcor_path.name}: pb.fits.gz not found")
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
                raise ValueError(f"Unexpected shape {shape}")
        finally:
            shdu.close()
    print(f"  wrote {nonpbcor_path.name}")
    pb_path.unlink()

# Canonical symlinks for CO21
print("\nCreating canonical symlinks for CO21 ...")
GALAXY = "NGC5253"
sci_stem = None
for p in sorted(DEST.glob("*NGC5253_sci.spw0_*.cube.I.pbcor.fits")):
    sci_stem = p.name.replace(".cube.I.pbcor.fits", "")
    break

if sci_stem:
    for old_suffix, new_suffix in [
        (".cube.I.pbcor.fits", "_pbcor.fits"),
        (".cube.I.nonpbcor.fits", "_nonpbcor.fits"),
        (".cube.I.pb.fits.gz", "_pb.fits.gz"),
    ]:
        old = DEST / f"{sci_stem}{old_suffix}"
        new = DEST / f"{GALAXY}_CO21{new_suffix}"
        if not old.exists():
            continue
        if new.is_symlink():
            if new.resolve() == old.resolve():
                continue
            new.unlink()
        elif new.exists():
            continue
        new.symlink_to(old.name)
        print(f"  [CO21] {new.name} -> {old.name[:70]}...")
else:
    print("  ERROR: could not find NGC5253 science spw0 pbcor file")

print("\nDone. Final files in DEST:")
for p in sorted(DEST.iterdir()):
    if p.is_file() or p.is_symlink():
        try:
            sz = p.stat().st_size / 1e9
            print(f"  {sz:6.2f} GB  {p.name}")
        except Exception:
            print(f"  (link)  {p.name}")

print("\nNext step: write step3_analyze_deep_native.py with HRL = X1a spw1,")
print("CO = X1a spw0 (both 0.19″ same-mousid → no common-beam convolution).")
print("Expected: S/N 12-16, flux ~0.8 Jy·km/s, mask matched to compact ~0.5″ source.")
