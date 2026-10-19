"""Detection of rooftop obstructions.

Three complementary cues are combined, all restricted to the target building:

* **Chimneys / dormers** - positive residuals of the measured DSM against the
  regularised facet planes. Compact, tall, small-footprint blobs are chimneys;
  broader but still-raised blobs are dormers.
* **Skylights / roof windows / glass roofs** - bright, low-saturation (glass)
  pixels in the RGB imagery. On the fitted plane these are flush, which is what
  distinguishes them from dormers.

The output is a set of boolean masks plus a summary of counts and areas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .logging_utils import get_logger
from .segmentation import Facet, facets_label_map
from .coords import px_to_xy

log = get_logger(__name__)


@dataclass
class Obstructions:
    chimney: np.ndarray
    dormer: np.ndarray
    skylight: np.ndarray

    @property
    def combined(self) -> np.ndarray:
        return self.chimney | self.dormer | self.skylight

    def counts(self) -> dict:
        def n(m):
            lab, k = ndi.label(m)
            return int(k)
        return {"chimneys": n(self.chimney), "dormers": n(self.dormer),
                "skylights": n(self.skylight)}


def roof_surface(facets: list[Facet], label_map: np.ndarray, dsm: np.ndarray,
                 pixel_size: float = 0.1, mask: np.ndarray | None = None,
                 nearest: bool = True):
    """Evaluate the regularised plane of the (nearest) facet at every pixel."""
    lm = label_map
    if nearest:
        ind = ndi.distance_transform_edt(lm == 0, return_distances=False, return_indices=True)
        lm = lm[tuple(ind)]
    surf = np.full(dsm.shape, np.nan, np.float32)
    for k, f in enumerate(facets):
        m = lm == (k + 1)
        if mask is not None:
            m = m & mask
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        x, y = px_to_xy(ys, xs, pixel_size)
        surf[ys, xs] = f.z_at(x, y)
    return surf


def residual_field(facets, label_map, dsm, mask, pixel_size=0.1):
    """dsm - regularised \'nearest\' facet plane, only defined on the mask."""
    surf = roof_surface(facets, label_map, dsm, pixel_size, mask=mask, nearest=True)
    resid = dsm.astype(np.float32) - surf
    resid[~mask] = np.nan
    return resid


def detect(rgb: np.ndarray, dsm: np.ndarray, facets: list[Facet], mask: np.ndarray,
           pixel_size: float = 0.1, chimney_min_h: float = 1.0,
           dormer_min_h: float = 0.35,
           object_max_area_m2: float = 25.0, min_object_area_m2: float = 0.4,
           glass_v_min: float = 195, glass_sat_max: float = 65,
           glass_min_area_m2: float = 1.5,
           glass_close_px: int = 15) -> Obstructions:
    label_map = facets_label_map(facets, dsm.shape, mask, dsm, pixel_size, thresh=None)
    resid = residual_field(facets, label_map, dsm, mask, pixel_size)

    chimney = np.zeros_like(mask)
    dormer = np.zeros_like(mask)
    raised = mask & np.isfinite(resid) & (resid > dormer_min_h)
    lab, n = ndi.label(raised)
    px_area = pixel_size ** 2
    for i in range(1, n + 1):
        comp = lab == i
        area = comp.sum() * px_area
        if area > object_max_area_m2 or area < min_object_area_m2:
            continue
        peak = float(np.nanmax(resid[comp]))
        if peak >= chimney_min_h:
            chimney |= comp
        else:
            dormer |= comp

    # --- glass (skylights / roof windows / glass roofs) from RGB ---
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6) * 255.0, 0)
    glass = mask & (mx > glass_v_min) & (sat < glass_sat_max)
    glass = ndi.binary_closing(glass, np.ones((glass_close_px, glass_close_px)))
    lab2, n2 = ndi.label(glass)
    skylight = np.zeros_like(mask)
    for i in range(1, n2 + 1):
        comp = lab2 == i
        if comp.sum() * px_area >= glass_min_area_m2:
            skylight |= comp
    skylight &= ~chimney

    counts = {"chimneys": int(ndi.label(chimney)[1]), "dormers": int(ndi.label(dormer)[1]),
              "skylights": int(ndi.label(skylight)[1])}
    log.info("obstructions: %s (chimney %.1f m^2, dormer/non-planar %.1f m^2, skylight %.1f m^2)",
             counts, chimney.sum() * px_area, dormer.sum() * px_area, skylight.sum() * px_area)
    return Obstructions(chimney=chimney, dormer=dormer, skylight=skylight)
