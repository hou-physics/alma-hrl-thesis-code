"""Mask construction + cross-cube transfer (beam convolution + reprojection)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from astropy.wcs import WCS
import astropy.units as u
from radio_beam import Beam
from reproject import reproject_interp
from scipy.ndimage import label

from cube_io import CubeBundle


def smooth(image: np.ndarray, kernel_fwhm_pix: float) -> np.ndarray:
    """Gaussian smoothing on a 2-D image via FFT (O(N² log N) regardless
    of kernel size). Equivalent to direct convolution within float32
    precision (~1e-5 relative); ~10000× faster for sf >> beam."""
    sigma = kernel_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
    kernel = Gaussian2DKernel(sigma)
    return convolve_fft(image, kernel, normalize_kernel=True,
                        nan_treatment="fill", boundary="fill")


def build_mask(
    smoothed: np.ndarray,
    threshold: float,
    signal_box: Tuple[int, int, int, int],
    require_connected: bool,
    galaxy_center_pix: Optional[Tuple[int, int]],
    co_footprint: Optional[np.ndarray] = None,
    components_mode: str = "center",
) -> np.ndarray:
    """Binary mask = (smoothed > threshold) ∩ signal_box ∩ co_footprint.

    If `require_connected`, components_mode controls which connected components
    of the resulting mask are retained:
      "center"  — keep the component containing galaxy_center_pix; fall back
                  to the largest component if center pixel is empty.
      "largest" — keep only the largest component (ignores center).
      "all"     — keep ALL components within the footprint (no filtering).
                  Appropriate for distributed/disk-dominated sources where CO
                  emission spans multiple arms or bar ends.

    `co_footprint` (if provided) is a hard AND-constraint: the mask is
    intersected with this boolean array before the component step.
    """
    mask = np.zeros_like(smoothed, dtype=bool)
    if isinstance(signal_box, np.ndarray) and signal_box.dtype == bool:
        # Adaptive 2D signal mask (e.g. dilated footprint).
        mask = (smoothed > threshold) & signal_box
    else:
        x1, y1, x2, y2 = signal_box
        sub = smoothed[y1:y2 + 1, x1:x2 + 1]
        mask[y1:y2 + 1, x1:x2 + 1] = sub > threshold

    if co_footprint is not None:
        mask = mask & co_footprint

    if require_connected and mask.any() and components_mode != "all":
        labeled, nlabels = label(mask)
        if components_mode == "largest":
            if nlabels > 0:
                sizes = [(labeled == i).sum() for i in range(1, nlabels + 1)]
                mask = labeled == (int(np.argmax(sizes)) + 1)
            else:
                mask = np.zeros_like(smoothed, dtype=bool)
        else:  # "center" (default)
            if galaxy_center_pix is None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            else:
                cx, cy = galaxy_center_pix
            center_label = 0
            if 0 <= cy < labeled.shape[0] and 0 <= cx < labeled.shape[1]:
                center_label = int(labeled[cy, cx])
            if center_label > 0:
                mask = labeled == center_label
            elif nlabels > 0:
                sizes = [(labeled == i).sum() for i in range(1, nlabels + 1)]
                mask = labeled == (int(np.argmax(sizes)) + 1)
            else:
                mask = np.zeros_like(smoothed, dtype=bool)
    return mask


def _cube_beam(cube: CubeBundle) -> Beam:
    return Beam(
        major=cube.bmaj_deg * u.deg,
        minor=cube.bmin_deg * u.deg,
        pa=cube.bpa_deg * u.deg,
    )


def compute_common_beam(*cubes: CubeBundle) -> Beam:
    """Smallest Beam that encloses all input cube beams."""
    beams = [_cube_beam(c) for c in cubes]
    if len(beams) == 1:
        return beams[0]
    from radio_beam import Beams
    b = Beams(beams=beams)
    return b.common_beam()


_IDENTITY_TOL = 0.001


def _beams_equal(a: Beam, b: Beam) -> bool:
    rel_maj = abs(a.major.to("deg").value - b.major.to("deg").value) / b.major.to("deg").value
    rel_min = abs(a.minor.to("deg").value - b.minor.to("deg").value) / b.minor.to("deg").value
    return rel_maj < _IDENTITY_TOL and rel_min < _IDENTITY_TOL


def convolve_to_beam(cube: CubeBundle, target_beam: Beam) -> CubeBundle:
    """Convolve cube data to target_beam, streaming to a FITS scratch file.

    Identity short-circuit on equal beams. Otherwise writes the convolved
    cube to `{cube.path}.conv_<beamhash>.fits` next to the input and
    returns a CubeBundle memmapped onto it. Peak RAM is ~1 channel plane
    (~50 MB for 1700×1700 float32) instead of the full-cube allocation
    (~25 GB for NGC 7793 / NGC 1313), which otherwise OOMs on a 16 GB
    machine. The scratch file is reused across reruns (same target beam
    → same filename) — delete it manually to force reconvolution.
    """
    from pathlib import Path
    from astropy.io import fits
    current = _cube_beam(cube)
    if _beams_equal(current, target_beam):
        return cube
    diff = target_beam.deconvolve(current)
    pixel_scale_val = abs(cube.header.get("CDELT1", cube.header.get("CDELT2", 1.0)))
    pixel_scale = pixel_scale_val * u.deg
    kernel = diff.as_kernel(pixel_scale)

    # Deterministic tempfile name keyed on target beam so reruns reuse scratch.
    beam_tag = (
        f"b{target_beam.major.to('arcsec').value:.3f}x"
        f"{target_beam.minor.to('arcsec').value:.3f}"
        f"pa{target_beam.pa.to('deg').value:+.1f}"
    )
    scratch_path = Path(cube.path).with_suffix(f".conv_{beam_tag}.fits")

    # Prepare a 3-D header for StreamingHDU (original may have a singleton Stokes
    # axis as NAXIS=4; the in-memory data is 3-D after load_cube, so header must
    # match).
    header = cube.header.copy()
    if header.get("NAXIS", 0) == 4:
        header["NAXIS"] = 3
        for key in ("NAXIS4", "CRVAL4", "CRPIX4", "CDELT4",
                    "CTYPE4", "CUNIT4", "PC04_04", "PC4_4"):
            header.pop(key, None)
        for axis in (1, 2, 3):
            for key in (f"PC0{axis}_04", f"PC04_0{axis}",
                        f"PC{axis}_4", f"PC4_{axis}"):
                header.pop(key, None)
    header["BMAJ"] = float(target_beam.major.to("deg").value)
    header["BMIN"] = float(target_beam.minor.to("deg").value)
    header["BPA"] = float(target_beam.pa.to("deg").value)
    header["NAXIS1"] = cube.data.shape[2]
    header["NAXIS2"] = cube.data.shape[1]
    header["NAXIS3"] = cube.data.shape[0]

    if not scratch_path.exists():
        # Choose direct vs FFT convolution based on kernel size (2026-04-25).
        # astropy's `convolve` is O(N² × M²); for cross-mousid common-beam
        # (X1a 0.19″ → 0.91″ gives FWHM ~80 pix, support ~300 pix) this is
        # ~1000× slower than `convolve_fft` (O(N² log N), kernel-size independent).
        # Threshold ~10 pix kernel FWHM: above this, FFT dominates.
        try:
            karr = kernel.array if hasattr(kernel, "array") else np.asarray(kernel)
            kernel_size_pix = max(karr.shape)
        except Exception:
            kernel_size_pix = 0
        use_fft = kernel_size_pix > 20

        # Beam-area scaling for [Jy/beam] convolution (2026-04-25 fix). astropy's
        # `convolve(normalize_kernel=True)` performs pixel-space averaging (kernel
        # sums to 1 in pixel units), which preserves the pixel-summed flux but
        # NOT the per-beam intensity when the beam changes. For [Jy/beam] data
        # convolved to a larger common beam, the convolved image must be rescaled
        # by (new_beam_area / old_beam_area) so total flux in physical units
        # (Jy = sum(Jy/beam) × pix_area / beam_area) is preserved. Bug found
        # 2026-04-25 in NGC 5253 X1a + X1c dual-mousid run: factor was ~56×.
        # Same-mousid runs hide the bug because beam ratio ≈ 1.
        new_area = (target_beam.major.to("arcsec").value *
                    target_beam.minor.to("arcsec").value)
        old_area = (current.major.to("arcsec").value *
                    current.minor.to("arcsec").value)
        beam_area_scale = new_area / old_area

        shdu = fits.StreamingHDU(str(scratch_path), header)
        try:
            for ch in range(cube.data.shape[0]):
                if use_fft:
                    out_plane = convolve_fft(
                        cube.data[ch], kernel,
                        normalize_kernel=True, nan_treatment="fill",
                        allow_huge=True,
                    )
                else:
                    out_plane = convolve(
                        cube.data[ch], kernel,
                        normalize_kernel=True, nan_treatment="fill",
                    )
                out_plane = out_plane * beam_area_scale
                shdu.write(out_plane.astype(cube.data.dtype, copy=False))
        finally:
            shdu.close()

    with fits.open(str(scratch_path), memmap=True) as hdul:
        new_data = np.asarray(hdul[0].data)
    return CubeBundle(
        path=str(scratch_path),
        data=new_data,
        wcs=cube.wcs,
        header=header,
        bmaj_deg=float(target_beam.major.to("deg").value),
        bmin_deg=float(target_beam.minor.to("deg").value),
        bpa_deg=float(target_beam.pa.to("deg").value),
        crval3_hz=cube.crval3_hz,
        cdelt3_hz=cube.cdelt3_hz,
        crpix3=cube.crpix3,
    )


def reproject_mask(
    mask_2d: np.ndarray,
    source_wcs: WCS,
    target_wcs: WCS,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Reproject a 2-D boolean mask using bilinear interp + threshold at 0.5."""
    src = source_wcs.celestial
    tgt = target_wcs.celestial
    out, _ = reproject_interp(
        (mask_2d.astype(float), src), tgt, shape_out=target_shape
    )
    return (out > 0.5) & np.isfinite(out)
