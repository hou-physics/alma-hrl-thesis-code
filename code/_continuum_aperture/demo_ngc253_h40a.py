"""Demo: continuum-aperture methodology on NGC 253 H40α.

Validates the implementation against a known strong detection. Expected:
- Continuum > 3σ aperture should pick up the NGC 253 nuclear region
- Aperture-integrated H40α flux should be in the same ballpark as our
  scan-optimized headline 4.695 Jy·km/s (units now compatible: both Jy·km/s)
- S/N should be high (>20) since this is the brightest control galaxy

If the numbers come out wildly different, the implementation has a bug.
"""
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent))
from continuum_mask import build_aperture_mask_auto
from aperture_spectrum import extract_aperture_spectrum, measure_line


# Inputs
CUBE = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC253/NGC253_H40a_pbcor.fits"
Z = 0.000811
H40A_REST_HZ = 99.022952e9
NUC_RA = 11.88806
NUC_DEC = -25.28823


def signal_box_pix(cube_path, ra_deg, dec_deg, half_size_arcsec=20.0):
    """Pick a (x1, y1, x2, y2) box centered on (ra, dec) of given half-size."""
    with fits.open(cube_path, memmap=True) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr).celestial
        try:
            pix_arcsec = abs(float(hdr['CDELT1'])) * 3600
        except KeyError:
            pix_arcsec = abs(float(hdr['CD1_1'])) * 3600

    sky = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame='icrs')
    x_pix, y_pix = wcs.world_to_pixel(sky)
    half_pix = int(round(half_size_arcsec / pix_arcsec))
    cx, cy = int(round(float(x_pix))), int(round(float(y_pix)))
    return (cx - half_pix, cy - half_pix, cx + half_pix, cy + half_pix), (cx, cy), pix_arcsec


def main():
    print("=" * 70)
    print("Continuum-aperture methodology demo — NGC 253 H40α")
    print("=" * 70)

    # 1. Define signal box (where to look for both continuum and aperture)
    box, (cx, cy), pix_arcsec = signal_box_pix(CUBE, NUC_RA, NUC_DEC, half_size_arcsec=15.0)
    print(f"\n[1] Signal box around nucleus: {box}, pixel scale {pix_arcsec:.3f}″")

    # 2. Build aperture mask (auto: try continuum_3sigma, fall back to circular)
    # Bendo+2015 measured NGC 253 H40α flux in a ~6″ nuclear aperture.
    # Match this for direct literature comparison.
    print("\n[2] Building aperture mask (auto mode: continuum_3σ → circular fallback)...")
    cm = build_aperture_mask_auto(
        CUBE,
        ra_deg=NUC_RA,
        dec_deg=NUC_DEC,
        rest_freq_hz=H40A_REST_HZ,
        z=Z,
        fallback_radius_arcsec=6.0,  # Bendo+2015 nuclear aperture
        line_centers_kms=(0.0,),
        line_window_kms=300.0,
        sigma_thresh=3.0,
        signal_box_pix=box,
    )
    print(f"  Aperture pixels: {cm.npix}, mode={cm.mode_used}")
    if cm.mode_used == "continuum_3sigma":
        print(f"  Continuum peak in box: {cm.continuum_2d[box[1]:box[3], box[0]:box[2]].max()*1e3:.2f} mJy/beam")
        print(f"  Per-pixel σ (median in box): {np.nanmedian(cm.sigma_2d[box[1]:box[3], box[0]:box[2]])*1e3:.3f} mJy/beam")
        print(f"  Line-free channels used: {cm.n_line_free_chan}")
    else:
        print(f"  Fallback radius: {cm.radius_arcsec}″")

    # 3. Extract aperture-integrated spectrum
    print("\n[3] Extracting aperture-integrated 1D spectrum...")
    spec = extract_aperture_spectrum(
        CUBE,
        cm.mask_2d,
        rest_freq_hz=H40A_REST_HZ,
        z=Z,
    )
    print(f"  Aperture pix = {spec.npix_aperture}, beam_area_pix = {spec.beam_area_pix:.2f}")
    print(f"  N_eff_beam = npix / beam_area = {spec.npix_aperture/spec.beam_area_pix:.2f}")
    print(f"  Channel width: {spec.chan_width_kms:.2f} km/s")
    print(f"  σ_chan (off-line, aperture-integrated): {spec.sigma_chan_jy*1e3:.3f} mJy")

    # 4. Toma-style fixed-window measurement (200 km/s)
    print("\n[4] Toma-style 200 km/s fixed-window measurement at v=0:")
    res200 = measure_line(spec, integration_window_kms=200.0, line_center_kms=0.0)
    print(f"  flux        = {res200['flux_jy_kms']:.4f} Jy·km/s")
    print(f"  σ_int       = {res200['sigma_int_jy_kms']:.4f} Jy·km/s")
    print(f"  S/N         = {res200['sn']:.2f}")
    print(f"  n_int_chan  = {res200['n_int_chan']}")

    # 5. Comparison: also try 250 / 300 / 100 km/s windows for sensitivity
    print("\n[5] Width sensitivity (no scan, just diagnostic):")
    for W in [100, 150, 200, 250, 300, 400]:
        try:
            r = measure_line(spec, integration_window_kms=W)
            print(f"  W = {W:3d} km/s → S/N = {r['sn']:6.2f}, flux = {r['flux_jy_kms']:.4f} Jy·km/s")
        except RuntimeError as e:
            print(f"  W = {W:3d} km/s → SKIP ({e})")

    print("\n" + "=" * 70)
    print("Comparison target (scan-optimized headline from results/NGC253/summary.md):")
    print("  S/N = 58.1, flux = 4.695 Jy·km/s")
    print()
    print("Expected: continuum-aperture flux should be SIMILAR ORDER (factor ~2)")
    print("  - if much smaller: aperture is too tight (continuum threshold too high?)")
    print("  - if much larger: aperture too loose, integrating noise")
    print("  - S/N likely lower than scan-optimized (no scan over W; fixed at 200")
    print("    km/s — Toma's reported H40α width was 240 km/s for NGC 253")
    print("    nucleus, so 200 may slightly under-integrate)")
    print("=" * 70)


if __name__ == "__main__":
    main()
