"""Generate a small synthetic ALMA-like cube for unit tests.

Shape: 20 channels × 50 pix × 50 pix.
Contents:
  - Two injected lines: a "strong tracer" (Gaussian peak at ch 6, FWHM 3 chan)
    and a "weak tracer" (Gaussian peak at ch 14, FWHM 3 chan, 1/5 of strong amplitude).
  - Spatial profile: 2D Gaussian centered at (y=25, x=25), FWHM = 4 pixels.
  - White noise σ = 0.0005 Jy/beam.
  - Beam: bmaj=2", bmin=1", BPA=0°. Pixel scale 0.2"/pix.
  - Freq axis: CRVAL3 = 230.0 GHz at CRPIX3=1, CDELT3 = -0.010 GHz (channel width ≈ 13 km/s).
"""
import numpy as np
from astropy.io import fits
from pathlib import Path

OUT = Path(__file__).parent / "mini_cube.fits"

rng = np.random.default_rng(seed=42)
NCHAN, NY, NX = 20, 50, 50
SIGMA_NOISE = 0.0005

# Spatial profile (2D Gaussian, FWHM=4 pix)
yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
fwhm_pix = 4.0
sigma_pix = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
spatial = np.exp(-((yy - 25)**2 + (xx - 25)**2) / (2 * sigma_pix**2))

# Spectral profiles (two Gaussians, FWHM=3 chan)
chan_axis = np.arange(NCHAN)
sigma_chan = 3.0 / (2 * np.sqrt(2 * np.log(2)))
strong_spec = 0.010 * np.exp(-((chan_axis - 6)**2) / (2 * sigma_chan**2))   # 10 mJy peak
weak_spec = 0.002 * np.exp(-((chan_axis - 14)**2) / (2 * sigma_chan**2))     # 2 mJy peak

# Build cube
cube = np.zeros((NCHAN, NY, NX))
for ch in range(NCHAN):
    cube[ch] = spatial * (strong_spec[ch] + weak_spec[ch])
cube += rng.normal(0, SIGMA_NOISE, size=cube.shape)

# Build header
hdr = fits.Header()
hdr['SIMPLE'] = True
hdr['BITPIX'] = -64
hdr['NAXIS'] = 3
hdr['NAXIS1'] = NX
hdr['NAXIS2'] = NY
hdr['NAXIS3'] = NCHAN
hdr['CTYPE1'] = 'RA---SIN'
hdr['CTYPE2'] = 'DEC--SIN'
hdr['CTYPE3'] = 'FREQ'
hdr['CRVAL1'] = 220.0     # deg
hdr['CRVAL2'] = -58.0
hdr['CRVAL3'] = 230.0e9   # Hz
hdr['CRPIX1'] = NX / 2 + 1
hdr['CRPIX2'] = NY / 2 + 1
hdr['CRPIX3'] = 1
hdr['CDELT1'] = -0.2 / 3600.0   # deg (0.2")
hdr['CDELT2'] = 0.2 / 3600.0
hdr['CDELT3'] = -10.0e6         # Hz (negative: freq decreases with ch)
hdr['BUNIT'] = 'Jy/beam'
hdr['BMAJ'] = 2.0 / 3600.0      # deg
hdr['BMIN'] = 1.0 / 3600.0
hdr['BPA'] = 0.0
hdr['RESTFRQ'] = 230.538e9      # CO rest

fits.PrimaryHDU(data=cube.astype(np.float64), header=hdr).writeto(OUT, overwrite=True)
print(f"Wrote {OUT}  shape={cube.shape}  noise_sigma={SIGMA_NOISE}")
