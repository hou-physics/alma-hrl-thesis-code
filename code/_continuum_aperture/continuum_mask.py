"""Build an aperture mask for the continuum-aperture methodology.

Two modes:

1. `mode="continuum_3sigma"` (Toma's spec): aperture = pixels where line-free
   continuum > sigma_thresh × per-pixel rms. REQUIRES a non-continuum-subtracted
   cube. Many ALMA Large Program products are continuum-subtracted, in which
   case this mode returns an empty mask — fall back to mode 2.

2. `mode="circular"`: fixed circular aperture at (ra_deg, dec_deg) of given
   radius_arcsec. Method-agnostic to continuum subtraction. Use when the
   nuclear position is known (NED, ALMA continuum image, etc.). For literature
   comparison this matches Toma's "X arcsec aperture" descriptions.

3. `mode="auto"`: try continuum_3sigma first; if aperture < beam_area_pix,
   fall back to circular at given (ra, dec, radius).

Public API: `build_aperture_mask(cube_path, mode=..., ...) -> ApertureResult`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import label as cc_label
import astropy.units as u

C_KMS = 299792.458


def _load_pb_2d(pb_path: str) -> np.ndarray:
    """Load a PB FITS into a 2D array (ny, nx).

    Cube-shape PBs (3D/4D) are reduced to the central channel. astropy can
    only memmap uncompressed FITS, so for *.fits.gz we prefer the *.fits
    twin if present (project convention from CLAUDE.md IGNORE rules) — this
    is critical for ~40 GB PB cubes where full decompression OOMs the
    process. For 2D-shape compressed PBs the in-RAM cost is negligible.
    """
    p = Path(pb_path)
    if p.suffix == ".gz":
        twin = p.with_suffix("")  # drops .gz, leaves .fits
        if twin.exists():
            pb_path = str(twin)
    use_memmap = not pb_path.endswith(".gz")
    with fits.open(pb_path, memmap=use_memmap) as hdul:
        data = hdul[0].data
        if data.ndim == 4:
            data = data[0]
        if data.ndim == 3:
            data = data[data.shape[0] // 2]
        return np.array(data, dtype=np.float32)


@dataclass
class ApertureResult:
    mask_2d: np.ndarray            # bool, (Ny, Nx)
    mode_used: str                 # "continuum_3sigma" or "circular"
    continuum_2d: np.ndarray | None  # only set in continuum mode
    sigma_2d: np.ndarray | None
    sigma_thresh: float | None
    n_line_free_chan: int | None
    radius_arcsec: float | None    # only set in circular mode
    cube_path: str
    line_centers_kms: tuple[float, ...]
    line_window_kms: float
    npix: int


# Backward-compat alias
ContinuumMaskResult = ApertureResult


def build_circular_mask(
    cube_path: str | Path,
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    pb_path: str | Path | None = None,
    pb_cutoff: float = 0.5,
    line_centers_kms: tuple[float, ...] = (0.0,),
    line_window_kms: float = 300.0,
) -> ApertureResult:
    """Fixed circular aperture at sky position (ra, dec) of given radius."""
    cube_path = str(cube_path)
    with fits.open(cube_path, memmap=True) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        if data.ndim == 4:
            data = data[0]
        nchan, ny, nx = data.shape
        wcs = WCS(hdr).celestial
        try:
            pix_arcsec = abs(float(hdr['CDELT1'])) * 3600
        except KeyError:
            pix_arcsec = abs(float(hdr['CD1_1'])) * 3600

    sky = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame='icrs')
    x_pix, y_pix = wcs.world_to_pixel(sky)
    cx, cy = float(x_pix), float(y_pix)

    yy, xx = np.indices((ny, nx))
    rad_pix = radius_arcsec / pix_arcsec
    aperture = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad_pix ** 2

    # Apply PB cut if provided
    if pb_path is not None:
        pb_data = _load_pb_2d(str(pb_path))
        aperture &= np.isfinite(pb_data) & (pb_data > pb_cutoff)

    return ApertureResult(
        mask_2d=aperture,
        mode_used="circular",
        continuum_2d=None,
        sigma_2d=None,
        sigma_thresh=None,
        n_line_free_chan=None,
        radius_arcsec=radius_arcsec,
        cube_path=cube_path,
        line_centers_kms=line_centers_kms,
        line_window_kms=line_window_kms,
        npix=int(aperture.sum()),
    )


def build_continuum_mask(
    cube_path: str | Path,
    rest_freq_hz: float,            # primary line rest freq (for velocity ref)
    z: float,
    line_centers_kms: tuple[float, ...] = (0.0,),
    line_window_kms: float = 300.0,
    sigma_thresh: float = 3.0,
    signal_box_pix: tuple[int, int, int, int] | None = None,
    pb_cutoff: float = 0.5,
    pb_path: str | Path | None = None,
    keep_connected_only: bool = True,
) -> ApertureResult:
    """Compute continuum image + per-pixel rms from line-free channels;
    threshold to define an aperture mask.

    `signal_box_pix = (x1, y1, x2, y2)`: only return mask pixels inside this
    box AND connected to a component overlapping the box. None = use full
    field but still apply pb_cutoff.

    `pb_path`: optional `.pb.fits[.gz]` to enforce PB > pb_cutoff. If None,
    no PB constraint is applied.
    """
    cube_path = str(cube_path)
    print(f"  [continuum] open #1 (header+freq)", flush=True)
    with fits.open(cube_path, memmap=True) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        if data.ndim == 4:
            data = data[0]
        nchan, ny, nx = data.shape
        print(f"  [continuum] cube shape: nchan={nchan}, ny={ny}, nx={nx}", flush=True)
        wcs = WCS(hdr)
        freq_hz = wcs.spectral.array_index_to_world_values(np.arange(nchan)).astype(float)
    print(f"  [continuum] freq_hz built ({nchan} channels)", flush=True)

    # Build line-free channel mask (in velocity space relative to rest_freq)
    f_rest_obs = rest_freq_hz / (1.0 + z)
    vel_kms = -C_KMS * (freq_hz - f_rest_obs) / f_rest_obs
    chan_in_line = np.zeros(nchan, dtype=bool)
    for vc in line_centers_kms:
        chan_in_line |= np.abs(vel_kms - vc) <= line_window_kms
    chan_line_free = ~chan_in_line
    n_lf = int(chan_line_free.sum())
    if n_lf < 50:
        raise RuntimeError(
            f"Only {n_lf} line-free channels — increase cube extent or"
            " narrow line_window_kms"
        )

    # Compute continuum + per-pixel σ across line-free channels.
    # Per-channel sequential reads avoid OS page-cache buildup that comes
    # with fancy-indexed chunked reads on 40 GB cubes. NGC 3627 base config
    # OOM-killed at chunk=64 AND chunk=16 (fancy-indexing into a memmap'd
    # 40 GB file forces the kernel to pin a wide working set of pages).
    # Sequential scan lets pages drop naturally behind the read head.
    import sys as _sys
    n_acc = 0
    sum_acc = np.zeros((ny, nx), dtype=np.float64)
    sumsq_acc = np.zeros((ny, nx), dtype=np.float64)
    n_lf_processed = 0
    n_lf_total = int(chan_line_free.sum())
    print(f"  [continuum] starting per-channel scan, {n_lf_total} line-free / {nchan} total chans",
          flush=True)
    with fits.open(cube_path, memmap=True) as hdul:
        cube = hdul[0].data
        if cube.ndim == 4:
            cube = cube[0]
        for c in range(nchan):
            if not chan_line_free[c]:
                continue
            chan = np.asarray(cube[c, :, :], dtype=np.float32)
            finite = np.isfinite(chan)
            np.copyto(chan, 0.0, where=~finite)
            sum_acc += chan  # implicit upcast to float64
            sumsq_acc += chan.astype(np.float64, copy=False) ** 2
            n_acc += finite
            del chan, finite
            n_lf_processed += 1
            if n_lf_processed % 200 == 0:
                print(f"  [continuum] processed {n_lf_processed}/{n_lf_total} line-free chans",
                      flush=True)
    print(f"  [continuum] done: processed {n_lf_processed}/{n_lf_total}", flush=True)
    n_acc = np.maximum(n_acc, 1)
    cont = sum_acc / n_acc
    var = sumsq_acc / n_acc - cont * cont
    var = np.where(var > 0, var, 0)
    sigma = np.sqrt(var)

    # Apply PB cut
    pb_ok = np.ones((ny, nx), dtype=bool)
    if pb_path is not None:
        pb_data = _load_pb_2d(str(pb_path))
        pb_ok = np.isfinite(pb_data) & (pb_data > pb_cutoff)

    # Threshold: continuum > sigma_thresh × per-pixel σ
    # Avoid division-by-zero; if σ=0 (e.g. a flagged pixel), reject
    valid = np.isfinite(cont) & np.isfinite(sigma) & (sigma > 0) & pb_ok
    aperture = valid & (cont > sigma_thresh * sigma)

    # Restrict to signal box
    if signal_box_pix is not None:
        x1, y1, x2, y2 = signal_box_pix
        box_mask = np.zeros_like(aperture)
        box_mask[y1:y2 + 1, x1:x2 + 1] = True
        aperture &= box_mask

    # Keep only connected components overlapping the signal box (avoids
    # picking up disconnected continuum sources elsewhere)
    if keep_connected_only and aperture.any() and signal_box_pix is not None:
        labeled, n_lab = cc_label(aperture, structure=np.ones((3, 3), dtype=int))
        x1, y1, x2, y2 = signal_box_pix
        center = (y1 + y2) // 2, (x1 + x2) // 2
        center_label = labeled[center]
        if center_label > 0:
            aperture = labeled == center_label
        else:
            # No component at the geometric center — keep all components
            # within the box (don't drop everything)
            box_labels = set(labeled[y1:y2 + 1, x1:x2 + 1].ravel()) - {0}
            aperture = np.isin(labeled, list(box_labels))

    return ApertureResult(
        mask_2d=aperture,
        mode_used="continuum_3sigma",
        continuum_2d=cont.astype(np.float32),
        sigma_2d=sigma.astype(np.float32),
        sigma_thresh=sigma_thresh,
        n_line_free_chan=n_lf,
        radius_arcsec=None,
        cube_path=cube_path,
        line_centers_kms=line_centers_kms,
        line_window_kms=line_window_kms,
        npix=int(aperture.sum()),
    )


def build_aperture_mask_auto(
    cube_path: str,
    ra_deg: float,
    dec_deg: float,
    rest_freq_hz: float,
    z: float,
    fallback_radius_arcsec: float,
    line_centers_kms: tuple[float, ...] = (0.0,),
    line_window_kms: float = 300.0,
    sigma_thresh: float = 3.0,
    signal_box_pix: tuple[int, int, int, int] | None = None,
    pb_path: str | None = None,
    pb_cutoff: float = 0.5,
    min_pix_for_continuum_mode: int | None = None,
) -> ApertureResult:
    """Try continuum_3sigma first; fall back to circular if too few pixels.

    `min_pix_for_continuum_mode = beam_area_pix * 2` by default — a continuum
    mask smaller than ~2 beams is not useful for aperture photometry.
    """
    # First try continuum
    cont = build_continuum_mask(
        cube_path,
        rest_freq_hz=rest_freq_hz,
        z=z,
        line_centers_kms=line_centers_kms,
        line_window_kms=line_window_kms,
        sigma_thresh=sigma_thresh,
        signal_box_pix=signal_box_pix,
        pb_path=pb_path,
        pb_cutoff=pb_cutoff,
    )

    # Compute beam area for threshold check
    if min_pix_for_continuum_mode is None:
        with fits.open(cube_path, memmap=True) as hdul:
            hdr = hdul[0].header
        bmaj = float(hdr['BMAJ']) * 3600
        bmin = float(hdr['BMIN']) * 3600
        try:
            pix_a = abs(float(hdr['CDELT1'])) * 3600
        except KeyError:
            pix_a = abs(float(hdr['CD1_1'])) * 3600
        beam_area_pix = np.pi * bmaj * bmin / (4 * np.log(2)) / pix_a ** 2
        min_pix_for_continuum_mode = int(2 * beam_area_pix)

    if cont.npix >= min_pix_for_continuum_mode:
        return cont

    # Fall back to circular
    print(f"  [auto-mask] continuum mode gave {cont.npix} pix < {min_pix_for_continuum_mode} threshold; "
          f"falling back to circular r={fallback_radius_arcsec:.1f}″")
    return build_circular_mask(
        cube_path,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_arcsec=fallback_radius_arcsec,
        pb_path=pb_path,
        pb_cutoff=pb_cutoff,
        line_centers_kms=line_centers_kms,
        line_window_kms=line_window_kms,
    )
