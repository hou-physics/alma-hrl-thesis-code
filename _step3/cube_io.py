"""FITS cube I/O, WCS parsing, beam arithmetic."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class CubeBundle:
    """A loaded FITS cube with metadata parsed out."""
    path: str
    data: np.ndarray
    wcs: WCS
    header: fits.Header
    bmaj_deg: float
    bmin_deg: float
    bpa_deg: float
    crval3_hz: float
    cdelt3_hz: float
    crpix3: float


def load_cube(path: str) -> CubeBundle:
    """Load one FITS cube. Collapses singleton Stokes axis if present.

    Preserves source dtype (usually float32 for ALMA products). Forcing
    float64 doubles memory with no precision benefit at the mJy noise level.

    If `path` ends in `.gz`, decompresses to a sibling `.fits` once
    (`{path}.decompressed.fits`) and memmaps that — gzipped FITS can't be
    memmapped and astropy would otherwise decompress the full cube into RAM
    (catastrophic for ~17 GB PB archives on 16 GB machines).
    """
    path = str(path)
    if path.endswith(".gz"):
        import gzip
        import shutil
        decompressed = Path(path[:-3])  # strip .gz
        if not decompressed.exists():
            print(f"  decompressing {Path(path).name} -> "
                  f"{decompressed.name} (one-time, memmap needs uncompressed FITS)")
            with gzip.open(path, "rb") as src, open(decompressed, "wb") as dst:
                shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        path = str(decompressed)
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data)  # keep source dtype, memmapped
        header = hdul[0].header.copy()
    while data.ndim > 3:
        data = data[0]
    wcs3 = WCS(header)
    return CubeBundle(
        path=str(path),
        data=data,
        wcs=wcs3,
        header=header,
        bmaj_deg=float(header.get("BMAJ", 0.0)),
        bmin_deg=float(header.get("BMIN", 0.0)),
        bpa_deg=float(header.get("BPA", 0.0)),
        crval3_hz=float(header.get("CRVAL3", 0.0)),
        cdelt3_hz=float(header.get("CDELT3", 0.0)),
        crpix3=float(header.get("CRPIX3", 1.0)),
    )


def load_cube_bundle(
    *,
    hrl_pbcor_path: str,
    hrl_nonpbcor_path: str,
    hrl_pb_path: str,
    co_pbcor_path: str,
    co_nonpbcor_path: str,
    co_pb_path: str,
) -> Dict[str, CubeBundle]:
    """Load all 6 cubes needed for the dual-cube pipeline."""
    return {
        "hrl_pbcor": load_cube(hrl_pbcor_path),
        "hrl_nonpbcor": load_cube(hrl_nonpbcor_path),
        "hrl_pb": load_cube(hrl_pb_path),
        "co_pbcor": load_cube(co_pbcor_path),
        "co_nonpbcor": load_cube(co_nonpbcor_path),
        "co_pb": load_cube(co_pb_path),
    }


def apply_pb_cutoff(
    non_pbcor: CubeBundle, pb: CubeBundle, cutoff: float
) -> CubeBundle:
    """Return a new CubeBundle with non_pbcor.data zeroed where pb < cutoff.

    PB may be stored as 2D (y, x), 3D with singleton channel axis (1, y, x),
    or 3D full cube (n_chan, y, x). Streams channel-by-channel to a scratch
    FITS on disk (`{non_pbcor.path}.pbmask_{cutoff:.2f}.fits`) instead of
    allocating a full-cube copy in RAM. Scratch is reused across reruns.

    Per-channel peak RAM is ~2 planes (~25 MB for 1800² float32) instead
    of ~3× cube size (~80 GB for NGC 7793).
    """
    pb_data = np.squeeze(pb.data)
    scratch_path = (
        Path(non_pbcor.path).parent
        / f"{Path(non_pbcor.path).stem}.pbmask_{cutoff:.2f}.fits"
    )
    if not scratch_path.exists():
        header = non_pbcor.header.copy()
        if header.get("NAXIS", 0) == 4:
            header["NAXIS"] = 3
            for key in ("NAXIS4", "CRVAL4", "CRPIX4", "CDELT4",
                        "CTYPE4", "CUNIT4"):
                header.pop(key, None)
        header["NAXIS1"] = non_pbcor.data.shape[2]
        header["NAXIS2"] = non_pbcor.data.shape[1]
        header["NAXIS3"] = non_pbcor.data.shape[0]

        shdu = fits.StreamingHDU(str(scratch_path), header)
        try:
            for ch in range(non_pbcor.data.shape[0]):
                plane = np.array(non_pbcor.data[ch], copy=True)
                if pb_data.ndim == 2:
                    pb_plane = pb_data
                elif pb_data.ndim == 3:
                    pb_plane = pb_data[ch]
                else:
                    raise ValueError(
                        f"Unexpected PB ndim {pb_data.ndim} for cube "
                        f"{non_pbcor.path}"
                    )
                # Match legacy v1 behavior: 0 (not NaN) so sigma_clip sees
                # them as zero-mean signal vs nan → ignored.
                plane[pb_plane < cutoff] = 0.0
                shdu.write(plane.astype(non_pbcor.data.dtype, copy=False))
        finally:
            shdu.close()

    with fits.open(str(scratch_path), memmap=True) as hdul:
        new_data = np.asarray(hdul[0].data)
    return CubeBundle(
        path=str(scratch_path),
        data=new_data,
        wcs=non_pbcor.wcs,
        header=non_pbcor.header,
        bmaj_deg=non_pbcor.bmaj_deg,
        bmin_deg=non_pbcor.bmin_deg,
        bpa_deg=non_pbcor.bpa_deg,
        crval3_hz=non_pbcor.crval3_hz,
        cdelt3_hz=non_pbcor.cdelt3_hz,
        crpix3=non_pbcor.crpix3,
    )


def get_beam_area_pix(cube: CubeBundle) -> float:
    """beam_area_pix = pi * bmaj * bmin / (4 ln 2) / pixel_area."""
    pixel_scale_deg = abs(cube.header.get("CDELT1", cube.header.get("CDELT2", 1.0)))
    bmaj_pix = cube.bmaj_deg / pixel_scale_deg
    bmin_pix = cube.bmin_deg / pixel_scale_deg
    return math.pi * bmaj_pix * bmin_pix / (4 * math.log(2))


C_KMS = 299792.458


def channel_width_kms(cube: CubeBundle, rest_ghz: float) -> float:
    """Convert CDELT3 (Hz) to km/s at a given rest frequency."""
    return C_KMS * abs(cube.cdelt3_hz) / (rest_ghz * 1e9)
