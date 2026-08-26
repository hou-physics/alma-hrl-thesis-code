"""
Clean 2x2 curve-of-growth panel for the group-meeting slide.

Four typical detections: NGC 4945, NGC 253, NGC 3628, He 2-10.
For each galaxy we grow the integration aperture (dilation-CoG, dense) from
the scan-best seed and plot enclosed flux F vs aperture size (N_beam). A real,
localized line rises then flattens onto a plateau. Simplified style: no verdict
labels, no scan-argmax markers -- just the honest curve + shaded plateau zone.

Reuses the CoG math from cog_clean_detection_poc.py. Not part of the
production pipeline; a figure generator for the slide only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402

C_KMS = 299792.458
OUT = Path("/Volumes/HouAstro/master/group_meeting_slides/assets/cog_detection.png")
LINE_REST_HZ = {"H30a": 231.900928e9, "H40a": 99.022952e9}
N_DILATION = 45


@dataclass
class Cfg:
    name: str          # display name
    cube: str
    z: float
    line: str
    W_kms: float
    sf: int
    sthresh: float
    sra: float
    sdec: float
    ssize: float       # RA size arcsec
    sdecsize: float    # Dec size arcsec (== ssize if square)
    nra: float
    ndec: float
    nsize: float
    xcap: float = 0.0   # truncate curve at this N_beam (0 = keep all)


GAL = [
    Cfg("NGC 4945", "/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945/NGC4945_H30a_spw1_v1_contsub.fits",
        0.00188, "H30a", 315.0, 2, 8.6,
        196.36260, -49.46942, 17.0, 19.0, 196.36901, -49.46442, 7.0),
    Cfg("NGC 253", "/Volumes/HouAstro/master/master_thesis/work_dir/NGC253/NGC253_H40a_pbcor.fits",
        0.00080, "H40a", 180.0, 3, 32.0,
        11.88808, -25.28838, 31.2, 31.2, 11.87983, -25.28024, 12.8),
    Cfg("NGC 3628", "/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628/NGC3628_H30a_pbcor.fits",
        0.002772, "H30a", 345.0, 1, 30.3,
        170.0708, 13.5888, 40.0, 12.0, 170.0707, 13.5944, 15.0, xcap=42.0),
    Cfg("He 2-10", "/Volumes/HouAstro/master/master_thesis/work_dir/He2-10/He2-10_H40a_pbcor.fits",
        0.002912, "H40a", 75.0, 1, 5.7,
        129.06329, -26.40935, 16.0, 16.0, 129.05833, -26.40491, 11.0, xcap=22.0),
]


def chan_v(hdr, rest_obs):
    n = hdr["NAXIS3"]
    f = hdr["CRVAL3"] + (np.arange(n) - (hdr["CRPIX3"] - 1)) * hdr["CDELT3"]
    return (rest_obs - f) / rest_obs * C_KMS


def beam_fwhm_pix(hdr):
    return hdr["BMAJ"] / abs(hdr["CDELT1"])


def pix_per_beam(hdr):
    p = abs(hdr["CDELT1"])
    return (np.pi / (4.0 * np.log(2.0))) * (hdr["BMAJ"] / p) * (hdr["BMIN"] / p)


def smooth(img, fwhm):
    if fwhm <= 0:
        return img
    sig = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return convolve_fft(img, Gaussian2DKernel(sig), normalize_kernel=True,
                        nan_treatment="fill", boundary="fill")


def box(wcs2, ny, nx, ra, dec, ra_size, dec_size):
    xc, yc = wcs2.world_to_pixel_values(ra, dec)
    ps = abs(wcs2.wcs.cdelt[0]) * 3600.0
    hx = (ra_size / 2.0) / ps
    hy = (dec_size / 2.0) / ps
    m = np.zeros((ny, nx), dtype=bool)
    y0, y1 = max(0, int(yc - hy)), min(ny, int(yc + hy))
    x0, x1 = max(0, int(xc - hx)), min(nx, int(xc + hx))
    m[y0:y1, x0:x1] = True
    return m


def cog(c: Cfg):
    print(f"=== {c.name} ===")
    b = load_cube(c.cube)
    cube, hdr = b.data, b.header
    _, ny, nx = cube.shape
    rest_obs = LINE_REST_HZ[c.line] / (1.0 + c.z)
    v = chan_v(hdr, rest_obs)
    dv = float(np.abs(np.median(np.diff(v))))
    ppb = pix_per_beam(hdr)
    bf = beam_fwhm_pix(hdr)
    wcs2 = WCS(hdr).celestial

    sig = box(wcs2, ny, nx, c.sra, c.sdec, c.ssize, c.sdecsize)
    noi = box(wcs2, ny, nx, c.nra, c.ndec, c.nsize, c.nsize)

    inl = np.abs(v) <= c.W_kms / 2.0
    mom0 = np.nansum(cube[inl, :, :], axis=0) * dv
    ms = smooth(mom0, c.sf * bf)
    sigma = float(np.nanstd(ms[noi]))

    seed = (ms > c.sthresh * sigma) & sig
    if not seed.any():
        seed = (ms > 5.0 * sigma) & sig
    cur = seed.copy()
    Nb = [seed.sum() / ppb]
    F = [float(np.nansum(mom0[seed]) / ppb)]
    for _ in range(N_DILATION):
        cur = ndimage.binary_dilation(cur, iterations=1) & sig
        Nb.append(cur.sum() / ppb)
        F.append(float(np.nansum(mom0[cur]) / ppb))
    b.data = None  # type: ignore
    Nb, F = np.array(Nb), np.array(F)
    if c.xcap > 0:
        keep = Nb <= c.xcap
        Nb, F = Nb[keep], F[keep]
    print(f"  Nbeam {Nb[0]:.1f} -> {Nb[-1]:.1f}, F {F[0]:.3f} -> max {F.max():.3f}")
    return c.name, c.line, Nb, F


def main():
    res = [cog(c) for c in GAL]
    from matplotlib import rcParams
    rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (name, line, Nb, F) in zip(axes.ravel(), res):
        Fmax = float(np.nanmax(F))
        ax.axhspan(0.95 * Fmax, Fmax, color="#1a6b3c", alpha=0.12,
                   label="plateau (≥95% of max)")
        ax.plot(Nb, F, "o-", color="#1a6b3c", ms=4, lw=1.8)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("aperture size  $N_\\mathrm{beam}$")
        ax.set_ylabel("enclosed flux  $F$  (Jy·km/s)")
        lab = line.replace("a", r"$\alpha$")
        ax.set_title(f"{name}  ({lab})", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Curve of growth — real detections rise, then plateau",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
