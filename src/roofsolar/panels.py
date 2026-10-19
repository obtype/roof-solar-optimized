"""Solar-panel placement on reconstructed roof facets.

Each roof facet is treated as an independent planar surface. Panels are
flush-mounted (panel plane == facet plane). Placement works in the facet's own
2-D in-plane coordinate system ``(u, v)``:

1. The facet footprint and the obstructions inside it are rasterised in
   ``(u, v)`` and vectorised with contours.
2. The **usable polygon** is the footprint shrunk by an edge/ridge/eave setback,
   minus the obstructions grown by a clearance buffer.
3. Standard-size panels are packed on a rectilinear lattice aligned with the
   facet's strike direction (both panel orientations are tried); a panel is
   accepted only if its rectangle lies entirely inside the usable polygon.

Because everything happens in true in-plane metres, geometric validity (no
overlap with obstructions, no crossing of ridges, real module dimensions) is
enforced exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .logging_utils import get_logger
from .segmentation import Facet, facets_label_map
from .solar import annual_factor
from .coords import px_to_xy

log = get_logger(__name__)


@dataclass
class Panel:
    facet_id: int
    corners_xyz: np.ndarray       # (4, 3) world corners
    center_xyz: np.ndarray
    tilt_deg: float
    azimuth_deg: float
    panel_w: float
    panel_h: float
    irradiance: float             # relative annual beam factor vs. best facet


@dataclass
class LayoutResult:
    panels: list[Panel]
    facet_stats: list[dict]
    skipped_facets: list[dict]


class FacetFrame:
    """Orthonormal in-plane frame (u = strike/horizontal, v = up-slope)."""

    def __init__(self, facet: Facet):
        self.facet = facet
        n = facet.normal()
        u = np.array([-n[1], n[0], 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        v /= np.linalg.norm(v)
        if v[2] < 0:
            v = -v
        self.n = n
        self.u = u
        self.v = v
        cx, cy = facet.centroid
        self.origin = np.array([cx, cy, facet.z_at(cx, cy)])

    def to_uv(self, xyz: np.ndarray) -> np.ndarray:
        d = xyz - self.origin
        return np.c_[d @ self.u, d @ self.v]

    def to_xyz(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(uv)
        xyz = self.origin + uv[:, [0]] * self.u + uv[:, [1]] * self.v
        return xyz


def _rasterise_uv(facet_mask: np.ndarray, obst_mask: np.ndarray, frame: FacetFrame,
                  pixel_size: float = 0.1, res: float = 0.05):
    ys, xs = np.nonzero(facet_mask)
    x, y = px_to_xy(ys, xs, pixel_size)
    z = frame.facet.z_at(x, y)
    xyz = np.c_[x, y, z]
    uv = frame.to_uv(xyz)
    u0, v0 = uv.min(0)
    iu = np.round((uv[:, 0] - u0) / res).astype(int)
    iv = np.round((uv[:, 1] - v0) / res).astype(int)
    nu, nv = iu.max() + 1, iv.max() + 1
    fmask = np.zeros((nv, nu), np.uint8)
    omask = np.zeros((nv, nu), np.uint8)
    fmask[iv, iu] = 1
    omask[iv, iu] = obst_mask[ys, xs].astype(np.uint8)
    # close small gaps, then treat non-planar interior holes as obstacles
    fmask = ndi.binary_closing(fmask, np.ones((3, 3)))
    filled = ndi.binary_fill_holes(fmask)
    holes = filled & ~fmask
    omask = ndi.binary_dilation(omask, iterations=2)
    omask = filled & (omask | holes)
    return filled.astype(np.uint8), omask.astype(np.uint8), (u0, v0), res


def _mask_to_polygons(mask: np.ndarray, origin_uv, res: float, min_area_m2: float):
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) * res * res < min_area_m2:
            continue
        pts = c.reshape(-1, 2).astype(float)
        uv = np.c_[pts[:, 0] * res + origin_uv[0], pts[:, 1] * res + origin_uv[1]]
        p = Polygon(uv)
        if not p.is_valid:
            p = p.buffer(0)
        if p.area > 0:
            polys.append(p)
    return polys


def _pack_polygon(usable: Polygon, pw: float, ph: float, gap: float):
    minx, miny, maxx, maxy = usable.bounds
    pitch_u, pitch_v = pw + gap, ph + gap
    rects = []
    if maxx - minx < pw or maxy - miny < ph:
        return rects
    # split into connected pieces
    from shapely.geometry import MultiPolygon
    pieces = list(usable.geoms) if isinstance(usable, MultiPolygon) else [usable]
    for piece in pieces:
        if piece.is_empty or piece.area < pw * ph:
            continue
        bx0, by0, bx1, by1 = piece.bounds
        best = []
        # try a few lattice offsets to compact the layout
        for ou in (0.0, pitch_u / 2):
            for ov in (0.0, pitch_v / 2):
                cands = []
                v = by0 + ov
                while v + ph <= by1 + 1e-9:
                    u = bx0 + ou
                    while u + pw <= bx1 + 1e-9:
                        r = box(u, v, u + pw, v + ph)
                        if piece.covers(r):
                            cands.append(r)
                        u += pitch_u
                    v += pitch_v
                if len(cands) > len(best):
                    best = cands
        rects.extend(best)
    return rects


def plan(rgb: np.ndarray, dsm: np.ndarray, facets: list[Facet], target_mask: np.ndarray,
         obstructions, pixel_size: float = 0.1, suns: np.ndarray | None = None,
         panel_w: float = 1.72, panel_h: float = 1.13, gap: float = 0.03,
         setback: float = 0.30, clearance: float = 0.30,
         min_slope: float = 10.0, max_slope: float = 60.0,
         min_usable_m2: float | None = None) -> LayoutResult:
    label_map = facets_label_map(facets, dsm.shape, target_mask, dsm, pixel_size, thresh=None)
    obst = obstructions.combined
    # grow obstructions by clearance (in pixels)
    obst_grown = ndi.binary_dilation(obst, ndi.generate_binary_structure(2, 2),
                                     iterations=max(1, int(round(clearance / pixel_size))))
    panels: list[Panel] = []
    stats, skipped = [], []
    if min_usable_m2 is None:
        min_usable_m2 = panel_w * panel_h

    # honest irradiance normalisation: best facet found in this run
    factors = {}
    for f in facets:
        if suns is not None:
            factors[f.id] = annual_factor(f.normal(), suns)
    best_factor = max(factors.values()) if factors else 1.0

    for f in facets:
        fmask = label_map == f.id
        if not fmask.any():
            continue
        if f.slope < min_slope or f.slope > max_slope:
            skipped.append({"facet": f.id, "reason": f"slope {f.slope:.1f} deg outside [{min_slope},{max_slope}]"})
            continue
        frame = FacetFrame(f)
        fmu, omu, origin_uv, res = _rasterise_uv(fmask, obst_grown & fmask, frame, pixel_size)
        fp = _mask_to_polygons(fmu, origin_uv, res, min_area_m2=0.5)
        if not fp:
            skipped.append({"facet": f.id, "reason": "no footprint"})
            continue
        footprint = max(fp, key=lambda p: p.area)
        usable = footprint.buffer(-setback, join_style=2)
        if usable.is_empty:
            skipped.append({"facet": f.id, "reason": "footprint smaller than setback"})
            continue
        obst_polys = _mask_to_polygons(omu, origin_uv, res, min_area_m2=0.2)
        if obst_polys:
            usable = usable.difference(unary_union(obst_polys))
        if usable.is_empty or usable.area < min_usable_m2:
            skipped.append({"facet": f.id, "reason": "usable area below one panel after setbacks/obstructions"})
            continue

        # choose the orientation that fits the most panels
        best_rects, best_pw, best_ph = [], panel_w, panel_h
        for (pw, ph) in ((panel_w, panel_h), (panel_h, panel_w)):
            rects = _pack_polygon(usable, pw, ph, gap)
            if len(rects) > len(best_rects):
                best_rects, best_pw, best_ph = rects, pw, ph

        for r in best_rects:
            corners = np.array([[r.bounds[0], r.bounds[1]], [r.bounds[2], r.bounds[1]],
                                [r.bounds[2], r.bounds[3]], [r.bounds[0], r.bounds[3]]])
            xyz = frame.to_xyz(corners)
            panels.append(Panel(facet_id=f.id, corners_xyz=xyz,
                                center_xyz=xyz.mean(0), tilt_deg=f.slope, azimuth_deg=f.azimuth,
                                panel_w=best_pw, panel_h=best_ph,
                                irradiance=factors.get(f.id, 0.0) / best_factor if best_factor else 0.0))

        stats.append({
            "facet": f.id, "slope_deg": round(f.slope, 1), "azimuth_deg": round(f.azimuth, 1),
            "facet_area_m2": round(f.area_m2, 1), "usable_area_m2": round(usable.area, 1),
            "panels": len(best_rects), "orientation": f"{best_pw}x{best_ph}",
            "rms_m": round(f.rms, 3),
        })
        log.info("facet %d: slope %.1f az %.0f usable %.1f m2 -> %d panels",
                 f.id, f.slope, f.azimuth, usable.area, len(best_rects))

    log.info("placed %d panels across %d facets", len(panels), len(stats))
    return LayoutResult(panels=panels, facet_stats=stats, skipped_facets=skipped)
