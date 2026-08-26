"""
Didactic figure illustrating the two physical mechanisms (other than
baseline residual) that cause F(scale) in spatial CoG to decline:

  Top row:    CLEAN deconvolution sidelobes — compact source + dirty beam
              pattern leaves negative ring around the source post-CLEAN.
  Bottom row: short-spacing missing flux — extended source filtered by
              interferometer creates negative halo at large radius.

Both show: the underlying truth, what the cube ACTUALLY contains, and
the resulting F(r) curve for circular aperture photometry — F rises,
peaks at the source edge, then drops.

Output: /Volumes/HouAstro/master/result_v2/tmp/_didactic_clean_shortspacing.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cmasher  # noqa: F401
from scipy.special import j1
from scipy.ndimage import gaussian_filter

OUT = "/Volumes/HouAstro/master/result_v2/tmp/_didactic_clean_shortspacing.png"

NPIX = 256
PIX_SCALE = 0.1  # arbitrary arcsec/pix


def airy_psf(npix, fwhm_pix):
    """Approximate dirty beam: Airy disk (j1(r)/r)² with first negative ring."""
    yy, xx = np.indices((npix, npix))
    cx, cy = npix / 2, npix / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # convert FWHM to Airy first-zero radius (FWHM ≈ 1.029 × λ/D, first
    # zero at 1.22 × λ/D → first zero ≈ 1.186 × FWHM)
    k = 3.83171 / (1.186 * fwhm_pix)  # first zero
    arg = k * r
    arg[arg == 0] = 1e-10
    psf = (2 * j1(arg) / arg) ** 2
    # multiply by sign to expose the first sidelobe (real dirty beam has
    # both positive and negative lobes; Airy² is all positive, so we
    # synthesise a more realistic dirty pattern):
    sign = np.where(arg < 3.83171, 1, np.where(arg < 7.0156, -1, 1))
    return psf * sign


def aperture_cog(image, max_r):
    """Concentric circular aperture photometry F(r)."""
    npix = image.shape[0]
    yy, xx = np.indices((npix, npix))
    cx, cy = npix / 2, npix / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rs = np.arange(1, max_r + 1)
    F = np.array([float(image[r <= rr].sum()) for rr in rs])
    return rs, F


# ============== Top row: CLEAN sidelobes ==============
beam_fwhm = 12  # pixels
src_truth = np.zeros((NPIX, NPIX))
src_truth[NPIX // 2, NPIX // 2] = 1.0  # delta source
# imaging: convolve with dirty beam (Airy² with negative sidelobe)
psf_dirty = airy_psf(NPIX, beam_fwhm)
psf_dirty /= psf_dirty.max()
src_imaged = gaussian_filter(src_truth, beam_fwhm / 2.355)  # CLEAN restored
# Add explicit negative first sidelobe ring (residual of imperfect CLEAN)
yy, xx = np.indices((NPIX, NPIX))
cx, cy = NPIX / 2, NPIX / 2
r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
sidelobe_amp = -0.04   # negative ring depth
sidelobe = sidelobe_amp * np.exp(-((r - 1.4 * beam_fwhm) / (0.4 * beam_fwhm)) ** 2)
clean_image = src_imaged + sidelobe

# CoG
rs_clean, F_clean = aperture_cog(clean_image, max_r=80)

# ============== Bottom row: short-spacing missing flux ==============
yy, xx = np.indices((NPIX, NPIX))
src_extended = np.exp(-((xx - NPIX / 2) ** 2 + (yy - NPIX / 2) ** 2) / (2 * 25 ** 2))
src_extended /= src_extended.sum()  # normalize
# interferometer: high-pass filter in Fourier domain (remove central uv plane)
F_full = np.fft.fftshift(np.fft.fft2(src_extended))
ky, kx = np.indices(F_full.shape)
cx2, cy2 = F_full.shape[1] / 2, F_full.shape[0] / 2
kr = np.sqrt((kx - cx2) ** 2 + (ky - cy2) ** 2)
# Remove uv samples within central R_cutoff (smooth taper to avoid ringing)
R_cutoff = 4
high_pass = 1 - np.exp(-((kr / R_cutoff) ** 4))
F_filtered = F_full * high_pass
src_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

# CoG on both
rs_ext, F_truth = aperture_cog(src_extended, max_r=80)
_, F_obs_ext = aperture_cog(src_filtered, max_r=80)

# ============== Figure ==============
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# --- CLEAN row ---
ax = axes[0, 0]
im = ax.imshow(src_truth, origin="lower", cmap="cmr.cosmic")
ax.set_title("Truth: compact point source")
ax.set_xlim(NPIX/2-50, NPIX/2+50); ax.set_ylim(NPIX/2-50, NPIX/2+50)
plt.colorbar(im, ax=ax, fraction=0.045)

ax = axes[0, 1]
vlim = max(abs(clean_image.min()), clean_image.max()) * 0.5
im = ax.imshow(clean_image, origin="lower", cmap="RdBu_r",
               vmin=-vlim, vmax=vlim)
ax.set_title("After CLEAN: source + negative sidelobe ring\n"
             "(arrows mark first-null annulus)")
# arrows annotating the negative ring
for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    rx, ry = 1.4 * beam_fwhm * np.cos(ang), 1.4 * beam_fwhm * np.sin(ang)
    ax.annotate("", xy=(NPIX/2 + rx, NPIX/2 + ry),
                xytext=(NPIX/2 + 0.8 * rx, NPIX/2 + 0.8 * ry),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2))
ax.set_xlim(NPIX/2-50, NPIX/2+50); ax.set_ylim(NPIX/2-50, NPIX/2+50)
plt.colorbar(im, ax=ax, fraction=0.045)

ax = axes[0, 2]
ax.plot(rs_clean, F_clean, "o-", color="C3", markersize=3)
ax.axhline(1.0, color="green", ls="--", lw=1.0, label="true F (=1)")
ax.axhline(0, color="gray", ls="-", lw=0.5)
ax.axvline(1.4 * beam_fwhm, color="orange", ls=":", lw=1.0,
           label=f"negative ring radius ≈ {1.4*beam_fwhm:.0f} pix")
ax.set_xlabel("aperture radius (pix)")
ax.set_ylabel("F enclosed (arb)")
ax.set_title("F(aperture) — peaks at source size, then drops\nbecause aperture crosses the negative ring")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- short-spacing row ---
ax = axes[1, 0]
im = ax.imshow(src_extended, origin="lower", cmap="cmr.cosmic")
ax.set_title("Truth: extended Gaussian source\n(σ=25 pix)")
plt.colorbar(im, ax=ax, fraction=0.045)

ax = axes[1, 1]
vlim = max(abs(src_filtered.min()), src_filtered.max()) * 0.7
im = ax.imshow(src_filtered, origin="lower", cmap="RdBu_r",
               vmin=-vlim, vmax=vlim)
ax.set_title("After interferometer (short-spacing removed):\n"
             "positive core + extended negative halo")
plt.colorbar(im, ax=ax, fraction=0.045)

ax = axes[1, 2]
ax.plot(rs_ext, F_truth, "-", color="green", lw=2, label="F_truth (rises to source total)")
ax.plot(rs_ext, F_obs_ext, "o-", color="C3", markersize=3, label="F_observed (interferometer)")
ax.axhline(0, color="gray", ls="-", lw=0.5)
ax.set_xlabel("aperture radius (pix)")
ax.set_ylabel("F enclosed (arb)")
ax.set_title("F(aperture) — rises early but DECLINES\nbecause aperture sweeps up the negative halo")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle(
    "Two physical causes of F(scale) decline in spatial CoG\n"
    "(both produce the same shape as baseline residual but at different physical scales)",
    fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved {OUT}")
