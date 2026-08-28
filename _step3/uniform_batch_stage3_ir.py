"""Uniform batch — STAGE 3 (mainline C): infrared sizes + coverage + global TIR.

Per galaxy (all 31 stage-1 entries, incl. B-mosaic for corroboration):
  1. center + pixel scale + observing freq from the strong-cube header;
     ALMA FoV two ways: analytic 12-m primary-beam FWHM (single pointings)
     and the empirical mapped-extent diameter from the stage-1 finite mask;
  2. WISE W3 (12um) + W4 (22um) cutouts via SkyView (cached FITS);
  3. background-subtracted curve of growth centred on the ALMA pointing →
     R50 / R90 (+ "≤PSF" flag when unresolved, W4 PSF 12″) — sizes and
     coverage are RATIOS, immune to zero-point calibration;
  4. coverage: W4 flux fraction inside the PB FWHM aperture; criterion
     PB_FWHM ≥ 2×R90 → "covered" (Frank's 90%-IR-size rule);
  5. IRAS PSC/FSC fluxes via VizieR → TIR (Sanders & Mirabel 1996) →
     SFR (Kennicutt & Evans 2012). Distances cz/H0=70 PROVISIONAL
     (blueshifted/very-nearby sources flagged NEEDS-LIT-DISTANCE).
     ⚠ 2026-08-22 (decisions (5)): PSC/FSC underestimates extended nearby
     galaxies by up to 0.2 dex vs RBGS total photometry. After ANY rerun
     of this stage, uniform_batch_rbgs_flux_patch.py MUST be re-applied
     (stage 4 refuses to plot without its iras_src column).
  6. overlay thumbnail: W4 image + R90 circle + ALMA PB circle.

Calibration caveat (flagged, not silent): WISE cutout absolute zero points
are NOT verified — instrumental totals recorded for later verification;
the TIR axis uses calibrated IRAS catalog fluxes instead.

Run: conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_stage3_ir.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from uniform_batch_configs import build_table              # noqa: E402

# astroquery's HTTP cache stores EMPTY responses from service outages and
# replays them forever (same query → same cache key, even across processes;
# poisoned Cen A + 8 WISE fetches on 2026-08-04). Our own caches (wise/*.fits,
# *_strong_mom0.npy) already dedupe the expensive fetches — disable theirs.
try:
    from astroquery import cache_conf
    cache_conf.cache_active = False
except Exception:
    pass

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
WISE_DIR = OUT / "wise"
THUMB_DIR = OUT / "ir_thumbs"
CSV_PATH = OUT / "stage3_ir.csv"
C_KMS = 299792.458
H0 = 70.0
W4_PSF_ARCSEC = 12.0
CSV_FIELDS = ["galaxy", "final_class", "ra", "dec", "band_freq_ghz",
              "pb_fwhm_arcsec", "mapped_diam_arcsec",
              "w3_R50", "w3_R90", "w4_R50", "w4_R90", "w4_unresolved",
              "frac_in_pb_w4", "cov_corr", "band_used", "coverage", "cog_flag",
              "iras_f12", "iras_f25", "iras_f60", "iras_f100",
              "D_Mpc_cz", "dist_flag", "logL_TIR_Lsun", "SFR_TIR",
              "note"]


def stage1_class():
    with open(OUT / "stage1_masks.csv") as f:
        return {r["galaxy"]: r["final_class"] for r in csv.DictReader(f)}


DISH_M = {"ngc4826": 7.0}   # ACA 7m-only dataset (6.4″ beam); all others 12m


def alma_geometry(row):
    """center coord, obs freq, pixel scale, mapped-extent diameter."""
    strong = row["strong_path"] if (row["strong_path"] and
                                    Path(row["strong_path"]).exists()) \
        else row["hrl_path"]
    hdr = fits.getheader(strong)
    ra, dec = float(hdr["CRVAL1"]), float(hdr["CRVAL2"])
    freq_ghz = float(hdr["CRVAL3"]) / 1e9
    pix_arcsec = abs(float(hdr["CDELT1"])) * 3600.0
    dish = DISH_M.get(row["galaxy"], 12.0)
    pb_fwhm = 1.13 * (C_KMS * 1e3 / (freq_ghz * 1e9)) / dish * 206265.0
    mom0 = np.load(OUT / f"{row['galaxy']}_strong_mom0.npy", mmap_mode="r")
    n_fin = int(np.isfinite(mom0).sum())
    mapped_diam = 2.0 * np.sqrt(n_fin / np.pi) * pix_arcsec
    return ra, dec, freq_ghz, pb_fwhm, mapped_diam


def fetch_wise(name, ra, dec, width_deg):
    """SkyView cutouts, cached. Returns {band: path or None}."""
    from astroquery.skyview import SkyView
    out = {}
    for band, key in [("w3", "WISE 12"), ("w4", "WISE 22")]:
        dest = WISE_DIR / f"{name}_{band}.fits"
        if dest.exists():
            out[band] = dest
            continue
        out[band] = None
        for attempt in range(3):        # SkyView intermittently returns []
            try:
                imgs = SkyView.get_images(
                    position=SkyCoord(ra * u.deg, dec * u.deg),
                    survey=[key], width=width_deg * u.deg,
                    height=width_deg * u.deg)
                imgs[0].writeto(dest, overwrite=True)
                out[band] = dest
                break
            except Exception as e:
                print(f"    {band} fetch attempt {attempt+1} failed: {e}",
                      flush=True)
                import time
                time.sleep(15)
    return out


def curve_of_growth(path, ra, dec, pb_fwhm):
    """Background-subtracted CoG centred on (ra, dec).
    Returns dict(R50, R90, total_instr, frac_in_pb, flag)."""
    with fits.open(path) as hdul:
        img = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
    ny, nx = img.shape
    pix = abs(float(hdr["CDELT1"])) * 3600.0
    from astropy.wcs import WCS
    w = WCS(hdr)
    xc, yc = w.world_to_pixel_values(ra, dec)
    yy, xx = np.indices(img.shape)
    r_arcsec = np.hypot(yy - yc, xx - xc) * pix
    half = min(nx, ny) / 2.0 * pix

    bg_zone = (r_arcsec > 0.70 * half) & (r_arcsec < 0.95 * half) \
        & np.isfinite(img)
    bg = float(np.nanmedian(img[bg_zone]))
    work = img - bg

    # surface-brightness truncation: integrate only out to R_edge where the
    # azimuthal-mean SB falls below 2× the annular-scatter of the background
    # zone (kills the PSF-wing + residual-gradient CoG tail creep that
    # inflated R90 ~4-5× on compact sources in the first pass)
    edges = np.arange(0.0, 0.70 * half, 4.0)
    ann_sb = np.array([np.nanmean(work[(r_arcsec >= r0) & (r_arcsec < r1)])
                       for r0, r1 in zip(edges[:-1], edges[1:])])
    bg_edges = np.arange(0.70 * half, 0.95 * half, 4.0)
    bg_sb = np.array([np.nanmean(work[(r_arcsec >= r0) & (r_arcsec < r1)])
                      for r0, r1 in zip(bg_edges[:-1], bg_edges[1:])])
    sb_noise = float(np.nanstd(bg_sb)) if len(bg_sb) > 3 else \
        float(np.nanstd(ann_sb[-5:]))
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    below = (ann_sb < 2.0 * sb_noise) & (r_mid > 15.0)
    r_edge = 0.70 * half
    flag = "CoG-not-converged"
    for i in range(len(below) - 1):
        if below[i] and below[i + 1]:
            r_edge = r_mid[i]
            flag = ""
            break

    radii = np.arange(4.0, r_edge + 4.0, 4.0)
    cog = np.array([np.nansum(work[r_arcsec <= r]) for r in radii])
    total = cog[-1]
    if total <= 0:
        return dict(R50=np.nan, R90=np.nan, total=total,
                    frac_in_pb=np.nan, flag="no-positive-flux")
    R50 = float(np.interp(0.5 * total, cog, radii))
    R90 = float(np.interp(0.9 * total, cog, radii))
    f_pb = float(np.nansum(work[r_arcsec <= pb_fwhm / 2.0]) / total)
    return dict(R50=R50, R90=R90, total=total, frac_in_pb=f_pb, flag=flag)


PSF_STANDARD = "iras17208-0014"   # D=183 Mpc → guaranteed point source at W4 res
_psf_cache = {}


def psf_ee(band):
    """Empirical enclosed-energy curve of the Atlas-coadd PSF, measured with
    the IDENTICAL pipeline on the in-sample point-source standard.
    Returns (radii_arcsec, EE) interpolation arrays."""
    if band in _psf_cache:
        return _psf_cache[band]
    path = WISE_DIR / f"{PSF_STANDARD}_{band}.fits"
    with fits.open(path) as hdul:
        img = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
    ny, nx = img.shape
    pix = abs(float(hdr["CDELT1"])) * 3600.0
    cy, cx = ny // 2, nx // 2
    sub = img[cy - 15:cy + 15, cx - 15:cx + 15]
    dy, dx = np.unravel_index(np.nanargmax(sub), sub.shape)
    cy, cx = cy - 15 + dy, cx - 15 + dx
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx) * pix
    bg = float(np.nanmedian(img[(r > 250) & (r < 350)]))
    work = img - bg
    radii = np.arange(3.0, 120.0, 3.0)
    cog = np.array([np.nansum(work[r <= rr]) for rr in radii])
    ee = cog / cog[-1]
    _psf_cache[band] = (radii, ee)
    return radii, ee


def iras_fluxes(ra, dec):
    """VizieR PSC then FSC with whole-pass retry: an overloaded CDS can
    answer HTTP 200 with an EMPTY table (no exception), so retrying only
    on exceptions is not enough (Cen A failed 3× this way 2026-08-04)."""
    import time
    for attempt in range(3):
        out = _iras_fluxes_once(ra, dec)
        if np.isfinite(out[2]):
            return out
        if attempt < 2:
            print(f"    IRAS all-NaN (pass {attempt+1}) — retrying in 20 s",
                  flush=True)
            time.sleep(20)
    return out


def _iras_fluxes_once(ra, dec):
    """VizieR PSC (II/125) then FSC (II/156A); returns f12/f25/f60/f100 [Jy]."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["**"], row_limit=5)
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    for cat in ["II/125/main", "II/156A/fsc"]:
        res = None
        for attempt in range(3):        # transient VizieR outages
            try:
                res = v.query_region(coord, radius=2.0 * u.arcmin,
                                     catalog=cat)
                break
            except Exception as e:
                print(f"    VizieR {cat} attempt {attempt+1} failed: {e}",
                      flush=True)
                import time
                time.sleep(15)
        if res is None:
            continue
        if not res:
            continue
        t = res[0]
        cols = {c.lower(): c for c in t.colnames}
        def get(k):
            c = cols.get(k)
            try:
                return float(t[c][0]) if c else np.nan
            except Exception:
                return np.nan
        f12, f25 = get("fnu_12"), get("fnu_25")
        f60, f100 = get("fnu_60"), get("fnu_100")
        if np.isfinite(f60):
            return f12, f25, f60, f100
    return np.nan, np.nan, np.nan, np.nan


def tir_sfr(f12, f25, f60, f100, z):
    """Sanders & Mirabel 1996 F_IR(8-1000um) + KE12 SFR; cz/H0 distance."""
    if not np.isfinite(f60):
        return np.nan, np.nan, np.nan, "no-iras"
    if z < 0.0008:
        return np.nan, np.nan, np.nan, "NEEDS-LIT-DISTANCE"
    D_mpc = C_KMS * z / H0
    f12_ = f12 if np.isfinite(f12) else 0.0
    f25_ = f25 if np.isfinite(f25) else 0.0
    f100_ = f100 if np.isfinite(f100) else 0.0
    fir_wm2 = 1.8e-14 * (13.48 * f12_ + 5.16 * f25_ + 2.58 * f60 + f100_)
    D_m = D_mpc * 3.0857e22
    L_w = 4 * np.pi * D_m ** 2 * fir_wm2
    logL_sun = np.log10(L_w / 3.828e26)
    log_sfr = np.log10(L_w * 1e7) - 43.41          # KE12, L in erg/s
    return D_mpc, logL_sun, 10 ** log_sfr, "cz/H0-provisional"


def thumbnail(name, wise_path, ra, dec, pb_fwhm, R90, cls, cov):
    with fits.open(wise_path) as hdul:
        img = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
    from astropy.wcs import WCS
    w = WCS(hdr)
    xc, yc = w.world_to_pixel_values(ra, dec)
    pix = abs(float(hdr["CDELT1"])) * 3600.0
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    vmin, vmax = np.nanpercentile(img, [5, 99.5])
    ax.imshow(img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    # deepskyblue dashed vs lime solid, BOTH with black stroke outline:
    # the stroke keeps lines visible over the white-saturated core AND the
    # black sky; hue + linestyle double-coding covers colorblind readers
    import matplotlib.patheffects as pe
    for rad, cc, ls, lab in [(pb_fwhm / 2.0, "deepskyblue", "-", "ALMA PB"),
                             (R90, "lime", "-", "R90(22um)")]:
        if np.isfinite(rad):
            circ = plt.Circle((xc, yc), rad / pix, fill=False,
                              color=cc, lw=2.0, linestyle=ls, label=lab)
            circ.set_path_effects(
                [pe.withStroke(linewidth=3.8, foreground="black")])
            ax.add_patch(circ)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{name} [{cls}]  W4 22um  {cov}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(THUMB_DIR / f"{name}.png", dpi=110)
    plt.close()


def process(row, cls):
    name = row["galaxy"]
    print(f"\n=== {name} ({cls}) ===", flush=True)
    ra, dec, freq, pb_fwhm, mapped = alma_geometry(row)
    width = 0.4 if cls.startswith("B") else 0.25
    print(f"  RA={ra:.4f} Dec={dec:.4f}  ν={freq:.1f} GHz  "
          f"PB={pb_fwhm:.0f}\"  mapped={mapped:.0f}\"", flush=True)
    paths = fetch_wise(name, ra, dec, width)

    res = {b: dict(R50=np.nan, R90=np.nan, total=np.nan,
                   frac_in_pb=np.nan, flag="no-image")
           for b in ("w3", "w4")}
    for b in ("w3", "w4"):
        if paths.get(b):
            res[b] = curve_of_growth(paths[b], ra, dec, pb_fwhm)
    # PSF-relative coverage: the Atlas-coadd W4 PSF has R50 ≈ 17″ + broad
    # wings, so absolute R90 / raw frac_in_PB are PSF-dominated for compact
    # sources. Calibrate against the in-sample point-source standard.
    band_used = "w4" if paths.get("w4") else ("w3" if paths.get("w3") else None)
    w4 = res[band_used] if band_used else res["w4"]
    unres, cov_corr = False, np.nan
    if band_used and np.isfinite(w4["R50"]):
        radii, ee = psf_ee(band_used)
        r50_psf = float(np.interp(0.5, ee, radii))
        unres = w4["R50"] <= 1.3 * r50_psf
        ee_psf_at_pb = float(np.interp(pb_fwhm / 2.0, radii, ee))
        if ee_psf_at_pb > 0:
            cov_corr = w4["frac_in_pb"] / ee_psf_at_pb
        coverage = ("covered-unresolved" if unres else
                    "covered" if cov_corr >= 0.9 else "partial")
    else:
        coverage = "unknown"
    print(f"  {band_used or 'w4'}: R50={w4['R50']:.0f}\" R90={w4['R90']:.0f}\" "
          f"frac_in_PB={w4['frac_in_pb']:.2f} cov_corr={cov_corr:.2f} "
          f"→ {coverage}  {w4['flag']}", flush=True)

    f12, f25, f60, f100 = iras_fluxes(ra, dec)
    z = float(row["z"])
    D_mpc, logL, sfr, dflag = tir_sfr(f12, f25, f60, f100, z)
    if np.isfinite(logL):
        print(f"  IRAS f60={f60:.1f} Jy → logL_TIR={logL:.2f} Lsun, "
              f"SFR={sfr:.2f} Msun/yr [{dflag}]", flush=True)
    else:
        print(f"  IRAS/TIR: {dflag}", flush=True)

    if paths.get("w4"):
        try:
            thumbnail(name, paths["w4"], ra, dec, pb_fwhm, w4["R90"],
                      cls, coverage)
        except Exception as e:
            print(f"    thumbnail failed: {e}", flush=True)

    return dict(
        galaxy=name, final_class=cls, ra=round(ra, 5), dec=round(dec, 5),
        band_freq_ghz=round(freq, 2), pb_fwhm_arcsec=round(pb_fwhm, 1),
        mapped_diam_arcsec=round(mapped, 1),
        w3_R50=round(res["w3"]["R50"], 1), w3_R90=round(res["w3"]["R90"], 1),
        w4_R50=round(w4["R50"], 1), w4_R90=round(w4["R90"], 1),
        w4_unresolved=unres,
        frac_in_pb_w4=round(w4["frac_in_pb"], 3)
        if np.isfinite(w4["frac_in_pb"]) else "",
        cov_corr=round(cov_corr, 3) if np.isfinite(cov_corr) else "",
        band_used=band_used or "", coverage=coverage, cog_flag=w4["flag"],
        iras_f12=round(f12, 2) if np.isfinite(f12) else "",
        iras_f25=round(f25, 2) if np.isfinite(f25) else "",
        iras_f60=round(f60, 2) if np.isfinite(f60) else "",
        iras_f100=round(f100, 2) if np.isfinite(f100) else "",
        D_Mpc_cz=round(D_mpc, 1) if np.isfinite(D_mpc) else "",
        dist_flag=dflag,
        logL_TIR_Lsun=round(logL, 2) if np.isfinite(logL) else "",
        SFR_TIR=round(sfr, 2) if np.isfinite(sfr) else "",
        note="")


def main():
    warnings.filterwarnings("ignore")
    for d in (WISE_DIR, THUMB_DIR):
        d.mkdir(exist_ok=True)
    classes = stage1_class()
    rows = build_table()
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            done = {r["galaxy"] for r in csv.DictReader(f)}
        print(f"resuming — {len(done)} done", flush=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for row in rows:
            if row["galaxy"] in done or row["galaxy"] not in classes:
                continue
            try:
                out = process(row, classes[row["galaxy"]])
            except Exception as e:
                import traceback; traceback.print_exc()
                out = dict(galaxy=row["galaxy"],
                           final_class=classes.get(row["galaxy"], ""),
                           note=f"ERROR: {e}")
            w.writerow({k: out.get(k, "") for k in CSV_FIELDS})
            f.flush()
    print(f"\nstage 3 complete → {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
