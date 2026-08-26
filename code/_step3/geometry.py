"""Sky-coordinate region → pixel box conversion."""
from __future__ import annotations

from typing import Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from config import SkyRegion


def sky_region_to_pixel_box(
    region: SkyRegion,
    wcs: WCS,
    cube_shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[int, int, int, int]:
    """Convert SkyRegion (center + size_arcsec) → (x1, y1, x2, y2) pixel corners.

    If cube_shape is (nchan, ny, nx), result is clipped to [0, nx-1] and
    [0, ny-1]. If the region is entirely outside the cube, raises ValueError.
    """
    wcs2 = wcs.celestial
    center = SkyCoord(region.ra_deg * u.deg, region.dec_deg * u.deg, frame="icrs")
    xc, yc = wcs2.world_to_pixel(center)
    xc = float(xc)
    yc = float(yc)
    if not (np.isfinite(xc) and np.isfinite(yc)):
        raise ValueError(
            f"SkyRegion (ra={region.ra_deg}, dec={region.dec_deg}, "
            f"size={region.size_arcsec}″) is outside cube (shape={cube_shape})"
        )
    cdelt = abs(wcs2.wcs.cdelt[0]) * 3600.0
    half_pix_x = (region.size_arcsec / 2.0) / cdelt
    half_pix_y = (
        (region.dec_size_arcsec / 2.0) / cdelt
        if region.dec_size_arcsec is not None
        else half_pix_x
    )
    x1_raw = int(round(xc - half_pix_x))
    x2_raw = int(round(xc + half_pix_x))
    y1_raw = int(round(yc - half_pix_y))
    y2_raw = int(round(yc + half_pix_y))

    if cube_shape is not None:
        _, ny, nx = cube_shape
        x1 = max(0, x1_raw)
        x2 = min(nx - 1, x2_raw)
        y1 = max(0, y1_raw)
        y2 = min(ny - 1, y2_raw)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(
                f"SkyRegion (ra={region.ra_deg}, dec={region.dec_deg}, "
                f"size={region.size_arcsec}″) is outside cube (shape={cube_shape})"
            )
        return (x1, y1, x2, y2)
    return (x1_raw, y1_raw, x2_raw, y2_raw)
