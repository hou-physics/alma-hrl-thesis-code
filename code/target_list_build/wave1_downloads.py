"""Wave-1 download kit generator (16 confirmed targets, 2026-08-03).

Per target: expand the qualifying observations' file lists, keep ONLY
`*.cube.I.pbcor.fits` + `*.cube.I.pb.fits.gz`, and write into
master_thesis/work_dir/{CANON}/ :
  - download_urls.txt      (one URL per line)
  - step-1_download.sh     (curl loop, resumable: -C -)
plus a master runner  work_dir/_wave1/wave1_download_all.sh  and a manifest.

Downloads are NOT executed here (user runs locally per large-file policy).
Run: conda run -n casa_env python -u target_list_build/wave1_downloads.py
"""
from __future__ import annotations

import csv
import re
import stat
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave1_shortlist import good_obs_uids, fnum        # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_new_sample")
WORK = Path("/Volumes/HouAstro/master/master_thesis/work_dir")
PRODUCT_RE = re.compile(r"\.cube\.I\.pbcor\.fits$|\.cube\.I\.pb\.fits\.gz$")

# confirmed wave-1 (user, 2026-08-03), scout-name → canonical folder name
WAVE1 = {
    "IRAS F05189-2524": "IRASF05189-2524",
    "IRAS 19542+1110": "IRAS19542+1110",
    "IRAS 09022-3615": "IRAS09022-3615",
    "IRAS F14378-3651": "IRASF14378-3651",
    "IRAS F19297-0406": "IRASF19297-0406",
    "IRAS 07251-0248": "IRAS07251-0248",
    "NGC 1808": "NGC1808",
    "IRAS F12112+0305": "IRASF12112+0305",
    "NGC 2369": "NGC2369",
    "IRAS F14348-1447": "IRASF14348-1447",
    "NGC 7552": "NGC7552",
    "NGC 5128": "NGC5128",
    "NGC 7469": "NGC7469",
    "NGC 6814": "NGC6814",
    "NGC 0055": "NGC55",
    "NGC 3256": "NGC3256",
}


def main():
    warnings.filterwarnings("ignore")
    from astroquery.alma import Alma
    scout = {r["name"]: r for r in csv.DictReader(open(OUT / "alma_scout.csv"))}
    (WORK / "_wave1").mkdir(exist_ok=True)
    manifest = []
    master_lines = ["#!/bin/bash", "# Wave-1 master downloader — run locally.",
                    "set -e"]
    for scout_name, canon in WAVE1.items():
        r = scout[scout_name]
        z = fnum(r["z"])
        uids, _ = good_obs_uids(fnum(r["ra"]), fnum(r["dec"]), z)
        urls, tot = [], 0.0
        for uid in sorted(uids):
            info = Alma.get_data_info(uid, expand_tarfiles=True)
            for row in info:
                url = str(row["access_url"])
                if PRODUCT_RE.search(url):
                    urls.append(url)
                    s = fnum(row["content_length"])
                    tot += s if np.isfinite(s) else 0
        tdir = WORK / canon
        tdir.mkdir(exist_ok=True)
        (tdir / "download_urls.txt").write_text("\n".join(urls) + "\n")
        sh = tdir / "step-1_download.sh"
        sh.write_text(
            "#!/bin/bash\n"
            f"# {canon} — wave-1 selective product download "
            f"({len(urls)} files, ~{tot/1e9:.1f} GB)\n"
            f"cd \"$(dirname \"$0\")\"\n"
            "while read -r u; do\n"
            "  [ -z \"$u\" ] && continue\n"
            "  echo \"### $u\"\n"
            "  curl -L -O -C - --retry 5 --retry-delay 10 \"$u\"\n"
            "done < download_urls.txt\n")
        sh.chmod(sh.stat().st_mode | stat.S_IEXEC)
        master_lines.append(f"bash '{sh}'")
        manifest.append(dict(target=canon, scout_name=scout_name,
                             z=z, n_uids=len(uids), n_files=len(urls),
                             GB=round(tot / 1e9, 1),
                             hrl=r["hrl_lines"], tracers=r["tracers"]))
        print(f"  {canon:<18} {len(urls):>3} files  {tot/1e9:>6.1f} GB",
              flush=True)

    master = WORK / "_wave1" / "wave1_download_all.sh"
    master.write_text("\n".join(master_lines) + "\n")
    master.chmod(master.stat().st_mode | stat.S_IEXEC)
    with open(WORK / "_wave1" / "wave1_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)
    tot = sum(m["GB"] for m in manifest)
    print(f"\n{len(manifest)} targets, {tot:.0f} GB total", flush=True)
    print(f"master runner: {master}", flush=True)


if __name__ == "__main__":
    main()
