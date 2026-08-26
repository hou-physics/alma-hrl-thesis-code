##############################################################################
# Step -1 (deep tier): NGC 5253 X1a spw1 H30α-only download
#
# Option A path after X1c (0.91″, 2.05 mJy/bm) gave marginal S/N 3.31 under
# the footprint-constrained pipeline (2026-04-24). X1a is the extended 12m
# deep tier that Bendo+2017 used to get the 18σ / 0.86 Jy·km/s detection.
# Full X1a mousid is 76 GB; we download only the spw1 (H30α) subset ~25.6 GB
# because X1c's CO21 spw0 already provides the strong-tracer cube. Phase Y
# dual-cube pipeline natively supports HRL and CO from different mousids
# (common-beam convolution + WCS reprojection handle the tier mismatch).
#
# Mousid: uid://A001/X136/X1a (12-m long-baseline, 0.19" res, 0.92 mJy/bm)
# Only spw1 (H30α tuning), skip spw0/spw2/continuum (saved: 50 GB).
##############################################################################

import re
from pathlib import Path
from astroquery.alma import Alma

MOUSID = "uid://A001/X136/X1a"
# Match only spw1 (both science and calibrator cubes in that SPW)
REGEX = r"spw1_[0-9]+MHz\..*\.cube\.I\.(pbcor\.fits|pb\.fits\.gz)$"
DEST = Path("/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep")
DEST.mkdir(parents=True, exist_ok=True)

print(f"Probing data info for {MOUSID} (spw1 subset) ...")
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
# Post-process: reconstruct non-pbcor for Phase Y + create canonical symlinks
# (H30α only — CO21 stays in work_dir/NGC5253/ from X1c).
# ------------------------------------------------------------------
print("\nReconstructing non-pbcor = pbcor × pb ...")
import gzip
import shutil
from astropy.io import fits

for pbcor_path in sorted(DEST.glob("*NGC5253_sci*.cube.I.pbcor.fits")):
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

# Canonical symlinks for H30α only
print("\nCreating canonical symlinks for H30α ...")
GALAXY = "NGC5253"
sci_stem = None
for p in sorted(DEST.glob("*NGC5253_sci.spw1_*.cube.I.pbcor.fits")):
    sci_stem = p.name.replace(".cube.I.pbcor.fits", "")
    break

if sci_stem:
    for old_suffix, new_suffix in [
        (".cube.I.pbcor.fits", "_pbcor.fits"),
        (".cube.I.nonpbcor.fits", "_nonpbcor.fits"),
        (".cube.I.pb.fits.gz", "_pb.fits.gz"),
    ]:
        old = DEST / f"{sci_stem}{old_suffix}"
        new = DEST / f"{GALAXY}_H30a{new_suffix}"
        if not old.exists():
            continue
        if new.is_symlink():
            if new.resolve() == old.resolve():
                continue
            new.unlink()
        elif new.exists():
            continue
        new.symlink_to(old.name)
        print(f"  [H30a] {new.name} -> {old.name[:70]}...")
else:
    print("  ERROR: could not find NGC5253 science spw1 pbcor file")

print("\nDone. Final files:")
for p in sorted(DEST.iterdir()):
    if p.is_file() or p.is_symlink():
        try:
            sz = p.stat().st_size / 1e9
            print(f"  {sz:6.2f} GB  {p.name}")
        except Exception:
            print(f"  (link)  {p.name}")

print("\nNext step: write step3_analyze_deep.py pointing HRL to NGC5253_deep/ and")
print("CO to work_dir/NGC5253/ (X1c CO21). Run Phase Y to confirm or reject the")
print("~0.3 Jy·km/s weak-signal hypothesis at Bendo-equivalent sensitivity.")
