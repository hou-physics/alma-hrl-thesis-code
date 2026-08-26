##############################################################################
# Step -1 (scenario C, ASDM re-imaging): M83 / NGC 5236 data download
#
# Target: Bendo-equivalent quality for H30α detection by combining
#   - X327: 12-m compact config (1.05" beam, 14 GB ASDM)
#   - X329: 7-m ACA (5.00" beam, 3 GB ASDM)
# After calibration + concat + tclean, final image should have ~1" beam with
# short-spacings recovered (better than either X1c or X1e alone in the
# NGC 5253 test). See INSTRUCTIONS_reimaging.md in this directory.
#
# Per mousid we download:
#   - README.txt (~0 GB)
#   - _001_of_001.tar (delivery package with scripts; may include pre-cal MS)
#   - _auxiliary.tar (scriptForPI + calibration aux)
#   - .asdm.sdm.tar (raw visibility data)
# We SKIP _external_ari_l_*.tar (77 GB per X327, just the reduced FITS cubes
# that we're rebuilding from scratch).
#
# Total download: ~23 GB. Destination: work_dir/NGC5236_reimaging/{mousid_tag}/
#
# Generated: 2026-04-24 for M83 Bendo-equivalent re-imaging.
# astroquery 0.4.11, astropy, python 3.10.20, CASA 6.6.6.17 (casa_env).
##############################################################################

import re
from pathlib import Path
from astroquery.alma import Alma

MOUSIDS = {
    "X327": "uid://A001/X12f/X327",   # 12-m compact, 1.05"
    "X329": "uid://A001/X12f/X329",   # 7-m ACA, 5.00"
}
# Match delivery + auxiliary + ASDM tars + README. Exclude external_ari_l (FITS cubes).
INCLUDE_RE = re.compile(r"(_\d+_of_\d+\.tar|_auxiliary\.tar|\.asdm\.sdm\.tar|README\.txt)$")
EXCLUDE_RE = re.compile(r"_external_ari_l_")

ROOT = Path("/Volumes/HouAstro/master/master_thesis/work_dir/NGC5236_reimaging")
ROOT.mkdir(parents=True, exist_ok=True)

alma = Alma()

for tag, mous in MOUSIDS.items():
    dest = ROOT / tag
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {tag}  {mous} ===")
    print(f"Destination: {dest}")
    info = alma.get_data_info(mous, expand_tarfiles=False)

    urls, total = [], 0
    for row in info:
        url = str(row["access_url"])
        if not INCLUDE_RE.search(url) or EXCLUDE_RE.search(url):
            continue
        sz = int(row["content_length"]) if row["content_length"] else 0
        urls.append(url)
        total += sz

    if not urls:
        print(f"  WARNING: no files matched for {mous}; skipping")
        continue

    urls_txt = dest / "download_urls.txt"
    with open(urls_txt, "w") as f:
        for u in urls:
            f.write(f"{u}\n")
    print(f"Wrote fallback URL list: {urls_txt}  ({len(urls)} files, {total/1e9:.2f} GB)")
    print(f"  (if astroquery download fails, run:  wget -c -i {urls_txt.name}  inside {tag}/)")

    print(f"Downloading {len(urls)} files to {dest} ...")
    alma.download_files(
        urls,
        savedir=str(dest),
        cache=True,
        continuation=True,
    )

    # Report final contents
    print(f"\nFiles downloaded to {dest}:")
    for p in sorted(dest.iterdir()):
        if p.is_file():
            print(f"  {p.stat().st_size/1e9:6.2f} GB  {p.name}")

print()
print("=" * 70)
print("Download complete. Next steps are in INSTRUCTIONS_reimaging.md:")
print("  1. Extract each .tar (including .asdm.sdm.tar)")
print("  2. Inspect delivery packages for pre-calibrated MS — if present, skip step 3")
print("  3. Run scriptForPI.py (if no pre-cal MS) to calibrate ASDM → MS")
print("  4. concat X327.ms + X329.ms → M83_concat.ms")
print("  5. uvcontsub (adapted from ngc3628_analyse_code/step1_uvcontsub.py)")
print("  6. tclean (adapted from step2_imaging.py) → H30a and CO21 cubes as FITS")
print("  7. Symlink FITS to canonical names, run /analyze-galaxy NGC5236")
print("=" * 70)
