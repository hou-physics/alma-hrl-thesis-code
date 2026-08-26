"""Clean redundant scratch + legacy files from work_dir/.

KEEP:
  - {GAL}_{line}_{pbcor,nonpbcor,pb.fits.gz} symlinks (used by /analyze-galaxy)
  - member.uid___...spw{N}.cube.I.{pbcor,nonpbcor,pb.fits.gz} for SPWs that
    are mapped to a line (H30a / CO21 / etc.)
  - download_urls.txt, file_mapping.txt
  - results/{GAL}/_plot_cache.pkl (NOT in work_dir; under results/)

DELETE:
  - *.conv_b*.fits           (step3 common-beam scratch, regenerable)
  - *.pbmask_*.fits          (step3 PB-cut scratch, regenerable)
  - *.pb.fits AND *_pb.fits  (load_cube decompression scratch, .gz twin must exist)
  - ._*                      (macOS finder sidecar)
  - tclean.last, *.last, casa-*.log
  - *_v[1-4].fits            (legacy versioned)
  - *_results_v*.txt         (legacy result txt)
  - *pre_footprint*          (pre-2026-04-24 backup)
  - *.bak*                   (any backup)
  - *_spwunknown_* symlinks + their target FITS  (unused SPW cubes)
  - *_spw[0-9]_* symlinks NOT tagged as a known line + their targets

Re-running analyze() rebuilds conv_b + pbmask scratch on demand
(~5-30 min/galaxy depending on cube size).
"""
from pathlib import Path
import os

WORK = Path("/Volumes/HouAstro/master/master_thesis/work_dir")
KNOWN_LINES = ("H30a", "H29a", "H40a", "H41a", "H42a", "CO21", "CO10", "CO32",
               "HCN10", "HCN32", "HCN43", "HCO+10", "HCO+32", "HCO+43")


def collect_used_targets(galaxy_dir):
    used = set()
    for ext in ("*.fits", "*.fits.gz"):
        for sym in galaxy_dir.glob(ext):
            if sym.is_symlink():
                try:
                    target = (galaxy_dir / os.readlink(sym)).resolve()
                    if target.exists():
                        used.add(target)
                except OSError:
                    pass
    return used


def main(dry_run: bool = False):
    deleted_n = 0
    deleted_b = 0
    galaxies = sorted([d for d in WORK.iterdir() if d.is_dir()])
    print(f"{'DRY RUN' if dry_run else 'EXECUTING'} on {len(galaxies)} galaxies")

    for gd in galaxies:
        # 1. Pure scratch: macOS sidecars, CASA logs, conv_b, pbmask, plot_cache (none in work_dir)
        for pat in ("._*", "*.last", "casa-*.log", "tclean.last",
                    "*.conv_b*.fits", "*.pbmask_*.fits"):
            for f in gd.glob(pat):
                if f.is_symlink():
                    continue
                if f.is_file():
                    sz = f.stat().st_size
                    if not dry_run:
                        f.unlink()
                    deleted_n += 1
                    deleted_b += sz

        # 2. Decompressed PB scratch where .gz twin exists (BOTH glob patterns
        # required: '*.pb.fits' for member.uid___*.cube.I.pb.fits style;
        # '*_pb.fits' for {GAL}_{line}_pb.fits style — fnmatch '.' is literal).
        seen = set()
        for pat in ("*.pb.fits", "*_pb.fits"):
            for pb in gd.glob(pat):
                if pb in seen or pb.is_symlink():
                    continue
                seen.add(pb)
                if (pb.parent / (pb.name + ".gz")).exists():
                    sz = pb.stat().st_size
                    if not dry_run:
                        pb.unlink()
                    deleted_n += 1
                    deleted_b += sz

        # 3. spwunknown / numeric-spw symlinks + their target FITS
        targets_to_drop = set()
        for sym in list(gd.glob("*spwunknown_*")) + list(gd.glob("*_spw[0-9]*_*.fits*")):
            if not sym.is_symlink():
                continue
            # Skip if symlink is for a known line (e.g. NGC4945_H30a_spw1_v1.fits, etc.)
            if any(f"_{line}_" in sym.name for line in KNOWN_LINES):
                continue
            try:
                target = (gd / os.readlink(sym)).resolve()
                if target.exists():
                    targets_to_drop.add(target)
            except OSError:
                pass
            if not dry_run:
                sym.unlink()
            deleted_n += 1
        for t in targets_to_drop:
            if t.exists() and not t.is_symlink():
                sz = t.stat().st_size
                if not dry_run:
                    t.unlink()
                deleted_n += 1
                deleted_b += sz

        # 4. Legacy / backup
        for pat in ("*_v[1-4].fits", "*_results_v*.txt",
                    "*pre_footprint*", "*.bak", "*.bak.*"):
            for f in gd.glob(pat):
                if f.is_symlink() or not f.is_file():
                    continue
                sz = f.stat().st_size
                if not dry_run:
                    f.unlink()
                deleted_n += 1
                deleted_b += sz

        # 5. Unused-SPW cubes whose pbcor/nonpbcor/pb.fits.gz isn't symlinked
        used = collect_used_targets(gd)
        for f in gd.glob("member.uid___*.cube.I.*.fits*"):
            if f.is_symlink():
                continue
            if f.resolve() in used:
                continue
            if any(s in f.name for s in (".cube.I.pbcor.fits",
                                          ".cube.I.nonpbcor.fits",
                                          ".cube.I.pb.fits.gz",
                                          ".cube.I.pb.fits")):
                sz = f.stat().st_size
                if not dry_run:
                    f.unlink()
                deleted_n += 1
                deleted_b += sz

    print(f"\n{'Would delete' if dry_run else 'Deleted'}: "
          f"{deleted_n} files, {deleted_b/1e9:.2f} GB")


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    main(dry_run=dry)
