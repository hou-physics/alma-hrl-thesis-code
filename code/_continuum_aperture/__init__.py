"""Continuum-aperture spectrum methodology — reproduction of Toma+ literature.

This package implements the methodology described in Toma's PhD thesis Ch. 2.1.2:

    "spectra were extracted from apertures surrounding the mm-RL emission.
     [...] In the case of a non-detection, spectra were extracted from regions
     where the continuum emission at the mm-RL frequency was more than three
     times above the background per-pixel rms level."

NOT Toma's own code. Independent reproduction of her methodology spec, written
by us, used to (a) sanity-check our scan-optimized headlines against a
literature-comparable measurement, and (b) populate a fourth diagnostic block
in each galaxy's summary.md.

The fundamental difference from `_step3/` (our scan-optimized pipeline) is:
- Mask defined from CUBE-INTERNAL CONTINUUM (line-free channels > 3σ),
  not from a separately-acquired CO cube.
- Integration window fixed (default 200 km/s for upper limits per Toma),
  not scanned.
- Aperture-integrated 1D spectrum (Jy units), not scan-optimized 2D mask flux.
"""
