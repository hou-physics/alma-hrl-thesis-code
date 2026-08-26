"""Wave-1 onboarding: turn downloaded product cubes into pipeline configs.

Per target (work_dir/{CANON}/, verified complete):
 1. read every *.cube.I.pbcor.fits header → frequency range → which lines
    (HRL H30α/H40α + tracers) fall inside at the target redshift;
 2. UNIFORM pairing rule (logged, no per-source judgment):
    prefer a member OUS containing BOTH an HRL cube and a tracer cube;
    among candidates pick the member whose HRL beam is finest within
    [0.3″, 8″]; tracer priority CO21 > CO10 > CS21 > HCO+10 > HCN10 >
    13CO21; if no member has both → cross-member pairing (flagged);
 3. canonical symlinks {CANON}_{line}_pbcor.fits (+ matching .pb.fits.gz);
 4. extract a 2-D primary-beam plane (mid-channel) for mom-0 flattening
    (non-pbcor cubes are not downloaded; stage 1 uses the pbcor×PB-plane
    flattening precedented by NGC 253);
 5. append rows to work_dir/_wave1/wave1_configs.csv (consumed by
    uniform_batch_configs.build_table).

Run: conda run -n casa_env python -u target_list_build/wave1_onboard.py
"""
from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

WORK = Path("/Volumes/HouAstro/master/master_thesis/work_dir")
OUT_CSV = WORK / "_wave1" / "wave1_configs.csv"

HRL_GHZ = {"H30a": 231.900928, "H40a": 99.022952}
TRACER_PRIORITY = ["CO21", "CO10", "CS21", "HCO+10", "HCN10", "13CO21"]
TRACER_GHZ = {"CO21": 230.538, "CO10": 115.271202, "CS21": 97.980953,
              "HCO+10": 89.188523, "HCN10": 88.631847, "13CO21": 220.398684}
BEAM_OK = (0.3, 8.0)
EDGE_GHZ = 0.05


def cube_info(path):
    hdr = fits.getheader(path)
    n = hdr["NAXIS3"]
    f0 = hdr["CRVAL3"] + (1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    f1 = f0 + (n - 1) * hdr["CDELT3"]
    lo, hi = sorted((f0 / 1e9, f1 / 1e9))
    bmaj = float(hdr.get("BMAJ", 0)) * 3600
    return lo, hi, bmaj


def lines_in(lo, hi, z, table):
    return [k for k, nu in table.items()
            if lo + EDGE_GHZ <= nu / (1 + z) <= hi - EDGE_GHZ]


def member_of(fname):
    return ".".join(fname.split(".")[:2])   # 'member.uid___A001_Xxxx_Xxx'


def onboard(canon, z):
    tdir = WORK / canon
    cubes = sorted(tdir.glob("member.*.cube.I.pbcor.fits"))
    # ARI-L tarballs include calibrator-field images (J****_ph / _bp) —
    # quasar pointings, not the galaxy. Science target cubes only.
    cubes = [p for p in cubes
             if not any(t in p.name for t in ("_ph.", "_bp.", "_chk."))]
    inv = []
    for p in cubes:
        try:
            lo, hi, bmaj = cube_info(p)
        except Exception as e:
            print(f"    header fail {p.name}: {str(e)[:40]}", flush=True)
            continue
        inv.append(dict(path=p, member=member_of(p.name), lo=lo, hi=hi,
                        bmaj=bmaj,
                        hrl=lines_in(lo, hi, z, HRL_GHZ),
                        tr=lines_in(lo, hi, z, TRACER_GHZ)))
    # members with both HRL and tracer
    members = {}
    for c in inv:
        members.setdefault(c["member"], []).append(c)
    paired = []
    for m, cs in members.items():
        hrls = [c for c in cs if c["hrl"] and BEAM_OK[0] <= c["bmaj"] <= BEAM_OK[1]]
        trs = [c for c in cs if c["tr"]]
        if hrls and trs:
            best_h = min(hrls, key=lambda c: c["bmaj"])
            paired.append((m, best_h, trs))
    flag = ""
    if paired:
        m, hcube, trs = min(paired, key=lambda t: t[1]["bmaj"])
    else:
        flag = "cross-member"
        hrls = [c for c in inv if c["hrl"] and BEAM_OK[0] <= c["bmaj"] <= BEAM_OK[1]]
        trs = [c for c in inv if c["tr"]]
        if not hrls or not trs:
            return None, "NO-VALID-PAIR"
        hcube = min(hrls, key=lambda c: c["bmaj"])
    # tracer choice by priority, then beam closest to HRL beam
    best_t, best_key = None, None
    for c in trs:
        for t in c["tr"]:
            k = (TRACER_PRIORITY.index(t) if t in TRACER_PRIORITY else 99,
                 abs(c["bmaj"] - hcube["bmaj"]))
            if best_key is None or k < best_key:
                best_key, best_t = k, (c, t)
    tcube, tracer = best_t
    hrl_line = hcube["hrl"][0] if len(hcube["hrl"]) == 1 else \
        ("H30a" if "H30a" in hcube["hrl"] else hcube["hrl"][0])
    same = tcube["path"] == hcube["path"]

    def canonical_link(line, src):
        dst = tdir / f"{canon}_{line}_pbcor.fits"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.name)
        pb_src = tdir / src.name.replace(".cube.I.pbcor.fits",
                                         ".cube.I.pb.fits.gz")
        pdst = tdir / f"{canon}_{line}_pb.fits.gz"
        if pb_src.exists():
            if pdst.is_symlink() or pdst.exists():
                pdst.unlink()
            pdst.symlink_to(pb_src.name)
        return dst, (pb_src if pb_src.exists() else None)

    hdst, _ = canonical_link(hrl_line, hcube["path"])
    tdst, tpb = canonical_link(tracer, tcube["path"])

    # PB plane for mom-0 flattening (mid channel of the tracer PB cube)
    pbplane = ""
    if tpb is not None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))
            from cube_io import load_cube
            b = load_cube(str(tpb))
            plane = np.asarray(b.data[b.data.shape[0] // 2], dtype=np.float32)
            w2 = WCS(b.header).celestial
            out = tdir / f"{canon}_{tracer}_pbplane.fits"
            fits.PrimaryHDU(plane, header=w2.to_header()).writeto(
                out, overwrite=True)
            pbplane = str(out)
            b.data = None
            sidecar = Path(str(tpb)[:-3])      # '.fits.gz' → '.fits'
            if sidecar.exists():
                os.remove(sidecar)             # reclaim disk
        except Exception as e:
            flag += f" pbplane-fail({str(e)[:30]})"
    else:
        flag += " no-pb"

    return dict(galaxy=canon.lower(), z=z, weak_line=hrl_line,
                strong_line=tracer, strong_path=str(tdst),
                strong_kind="pbcor+pbmult" if pbplane else "pbcor",
                hrl_path=str(hdst), window_kms=300.0, pb_mult=pbplane,
                hrl_beam=round(hcube["bmaj"], 3),
                tracer_beam=round(tcube["bmaj"], 3),
                same_cube=same, member=hcube["member"][-20:],
                flag=flag.strip()), None


def main():
    import sys
    warnings.filterwarnings("ignore")
    only = {a.upper() for a in sys.argv[1:]}     # optional re-run subset
    man = list(csv.DictReader(open(WORK / "_wave1" / "wave1_manifest.csv")))
    rows = []
    if only and OUT_CSV.exists():                # keep untouched targets
        rows = [r for r in csv.DictReader(open(OUT_CSV))
                if r["galaxy"].upper() not in only]
    for m in man:
        canon = m["target"]
        if only and canon.upper() not in only:
            continue
        z = float(m["z"])
        print(f"=== {canon} (z={z}) ===", flush=True)
        try:
            row, err = onboard(canon, z)
        except Exception as e:
            import traceback; traceback.print_exc()
            row, err = None, f"ERROR {e}"
        if row is None:
            print(f"  SKIP: {err}", flush=True)
            continue
        rows.append(row)
        print(f"  {row['weak_line']} ({row['hrl_beam']}\") + "
              f"{row['strong_line']} ({row['tracer_beam']}\")  "
              f"same_cube={row['same_cube']}  {row['flag'] or 'ok'}",
              flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} configs → {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
