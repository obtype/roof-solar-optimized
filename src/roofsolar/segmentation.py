"""Roof-face (planar facet) segmentation.

Algorithm
---------
1. **Sequential RANSAC plane extraction** on DSM samples of the building mask.
   Each round fits the plane with the most inliers (``z = a x + b y + c``) and
   removes them, so a hip roof naturally decomposes into its slope facets.
2. **Assignment + clean-up**: every mask pixel is assigned to its best-fitting
   plane; a median (mode) filter removes salt-and-pepper mislabelling and only
   components above a minimum area survive.
3. **Refinement**: a least-squares plane is re-fitted from the assigned pixels.
4. **Merge**: adjacent facets with almost identical normals are merged.
5. **Regularisation**: facet azimuths are snapped to a small set of dominant
   directions (the building axes), removing angular noise from the smoothed DSM
   while preserving the fitted pitch.

The output is a list of :class:`Facet` objects plus a labelled facet map. Any
raised surface that is *not* explained by a facet plane (dormers, chimneys)
becomes residual and is picked up later as an obstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .coords import px_to_xy
from .logging_utils import get_logger
from .terrain import normal_azimuth_slope, plane_from_normal

log = get_logger(__name__)


@dataclass
class Facet:
    id: int
    a: float
    b: float
    c: float
    azimuth: float
    slope: float
    area_m2: float
    rms: float
    pixel_count: int
    centroid: tuple[float, float]      # local (x east, y north)
    polygon: np.ndarray = field(default=None)   # Nx3 local polygon

    def z_at(self, x, y):
        return self.a * np.asarray(x) + self.b * np.asarray(y) + self.c

    def normal(self):
        n = np.array([-self.a, -self.b, 1.0])
        return n / np.linalg.norm(n)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fit_plane(pts: np.ndarray) -> tuple[float, float, float]:
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    A = np.c_[x, y, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _sample_points(mask, dsm, pixel_size, max_points=30000):
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        ys, xs = ys[idx], xs[idx]
    x, y = px_to_xy(ys, xs, pixel_size)
    return np.c_[x, y, dsm[ys, xs]]


def _ransac_planes(pts, thresh, min_inliers, iters, max_planes, seed=0):
    rng = np.random.default_rng(seed)
    remaining = np.ones(len(pts), bool)
    planes = []
    for _ in range(max_planes):
        idx = np.nonzero(remaining)[0]
        if len(idx) < min_inliers:
            break
        sub = pts[idx]
        best, best_n = None, 0
        for _ in range(iters):
            s = rng.choice(len(idx), 3, replace=False)
            p = sub[s]
            nrm = np.cross(p[1] - p[0], p[2] - p[0])
            if abs(nrm[2]) < 1e-6:
                continue
            a, b = -nrm[0] / nrm[2], -nrm[1] / nrm[2]
            c = p[0, 2] - a * p[0, 0] - b * p[0, 1]
            cnt = int(np.sum(np.abs(sub[:, 2] - (a * sub[:, 0] + b * sub[:, 1] + c)) < thresh))
            if cnt > best_n:
                best_n, best = cnt, (a, b, c)
        if best is None or best_n < min_inliers:
            break
        a, b, c = best
        inl = remaining & (np.abs(pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)) < thresh)
        a, b, c = _fit_plane(pts[inl])
        inl = remaining & (np.abs(pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)) < thresh)
        planes.append((a, b, c, inl.copy()))
        remaining &= ~inl
    return planes


def _keep_large_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes[1:] >= min_px
    return keep[lab]


def _fit_facet(label_pixels_yx, dsm, pixel_size) -> Facet:
    yy, xx = label_pixels_yx
    x, y = px_to_xy(yy, xx, pixel_size)
    p = np.c_[x, y, dsm[yy, xx]]
    a, b, c = _fit_plane(p)
    resid = p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c)
    az, sl = normal_azimuth_slope(a, b)
    return Facet(id=-1, a=a, b=b, c=c, azimuth=az, slope=sl,
                 area_m2=float(len(p) * pixel_size ** 2),
                 rms=float(np.sqrt(np.mean(resid ** 2))),
                 pixel_count=int(len(p)), centroid=(float(p[:, 0].mean()), float(p[:, 1].mean())))


def _normal_angle(f1: Facet, f2: Facet) -> float:
    n1, n2 = f1.normal(), f2.normal()
    return float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1, 1))))


def _merge_facets(facets, label_map, dsm, pixel_size, max_angle=7.0, max_offset=0.25):
    """Merge adjacent facets whose normals (and planes) nearly coincide."""
    active = {f.id: f for f in facets}
    changed = True
    while changed:
        changed = False
        ids = list(active.keys())
        for ii in range(len(ids)):
            i = ids[ii]
            if i not in active:
                continue
            mi = label_map == i
            mi_d = ndi.binary_dilation(mi, iterations=2)
            for jj in range(ii + 1, len(ids)):
                j = ids[jj]
                if j not in active:
                    continue
                if _normal_angle(active[i], active[j]) > max_angle:
                    continue
                mj = label_map == j
                if not np.any(mi_d & mj):
                    continue
                yy, xx = np.nonzero(mi_d & mj)
                x, y = px_to_xy(yy, xx, pixel_size)
                if np.mean(np.abs(active[i].z_at(x, y) - active[j].z_at(x, y))) > max_offset:
                    continue
                label_map[mj] = i
                yy, xx = np.nonzero(label_map == i)
                f = _fit_facet((yy, xx), dsm, pixel_size)
                f.id = i
                active[i] = f
                del active[j]
                changed = True
                mi_d = ndi.binary_dilation(label_map == i, iterations=2)
        ids = list(active.keys())
    return list(active.values())


def segment(dsm: np.ndarray, mask: np.ndarray, pixel_size: float = 0.1,
            thresh: float = 0.10, min_inliers: int = 800, iters: int = 900,
            max_planes: int = 10, min_area_m2: float = 3.0,
            snap_azimuth_tol: float = 12.0, merge_angle: float = 7.0,
            min_roof_slope: float = 0.0) -> list[Facet]:
    pts = _sample_points(mask, dsm, pixel_size)
    planes = _ransac_planes(pts, thresh, min_inliers, iters, max_planes)
    log.info("RANSAC recovered %d raw planes", len(planes))

    ys, xs = np.nonzero(mask)
    z = dsm[ys, xs]
    X, Y = px_to_xy(ys, xs, pixel_size)
    best_res = np.full(len(ys), np.inf)
    best_lab = np.zeros(len(ys), int)
    for i, (a, b, c, _) in enumerate(planes):
        res = np.abs(z - (a * X + b * Y + c))
        upd = res < best_res
        best_res[upd] = res[upd]
        best_lab[upd] = i + 1
    label_map = np.zeros(dsm.shape, int)
    label_map[ys, xs] = np.where(best_res <= thresh, best_lab, 0)

    # clean-up: remove speckle, then drop small components
    label_map = ndi.median_filter(label_map, size=5)
    min_px = max(1, int(round(min_area_m2 / pixel_size ** 2)))

    facets: list[Facet] = []
    for i in range(1, len(planes) + 1):
        m = label_map == i
        if not m.any():
            continue
        m = _keep_large_components(m, min_px)
        removed = (label_map == i) & ~m
        label_map[removed] = 0
        yy, xx = np.nonzero(m)
        if len(yy) * pixel_size ** 2 < min_area_m2:
            label_map[m] = 0
            continue
        f = _fit_facet((yy, xx), dsm, pixel_size)
        f.id = i
        facets.append(f)

    facets = _merge_facets(facets, label_map, dsm, pixel_size, max_angle=merge_angle)
    # drop implausible near-flat planes (ridge caps, artefacts); they are not roof faces
    facets = [f for f in facets if f.slope >= min_roof_slope]
    # relabel merged facets to contiguous ids 1..K (remember old -> new)
    new = np.zeros_like(label_map)
    for k, f in enumerate(facets):
        new[label_map == f.id] = k + 1
        f.id = k + 1
    label_map = new

    facets = _snap_azimuths(facets, snap_azimuth_tol)
    log.info("kept %d facets", len(facets))
    return facets


def _snap_azimuths(facets, tol):
    if not facets:
        return facets
    clusters: list[float] = []
    for az in sorted(f.azimuth for f in facets):
        if not any(min(abs(az - c), 360 - abs(az - c)) < tol for c in clusters):
            clusters.append(float(az))
    log.info("dominant azimuths: %s", [round(c) for c in clusters])
    for f in facets:
        best = min(clusters, key=lambda c: min(abs(f.azimuth - c), 360 - abs(f.azimuth - c)))
        if min(abs(f.azimuth - best), 360 - abs(f.azimuth - best)) < tol:
            f.azimuth = best
            cx, cy = f.centroid
            zc = f.a * cx + f.b * cy + f.c
            a, b = plane_from_normal(f.azimuth, f.slope)
            c = zc - a * cx - b * cy
            f.a, f.b, f.c = a, b, c
    return facets


def facets_label_map(facets, shape, mask, dsm, pixel_size=0.1, thresh=0.10):
    """Assign mask pixels to facets.

    ``thresh=None`` assigns every pixel to its nearest plane (a complete
    piecewise-planar tiling); a numeric threshold leaves poorly explained
    pixels unlabelled.
    """
    ys, xs = np.nonzero(mask)
    z = dsm[ys, xs]
    X, Y = px_to_xy(ys, xs, pixel_size)
    best_res = np.full(len(ys), np.inf)
    best = np.zeros(len(ys), int)
    for i, f in enumerate(facets):
        res = np.abs(z - (f.a * X + f.b * Y + f.c))
        upd = res < best_res
        best_res[upd] = res[upd]
        best[upd] = i + 1
    out = np.zeros(shape, int)
    if thresh is None:
        out[ys, xs] = best
        # compact the regions: mode-like filtering removes thin tendrils
        out = ndi.median_filter(out, size=15)
        # drop tiny islands per label
        for i, f in enumerate(facets):
            m = out == (i + 1)
            if not m.any():
                continue
            m2 = _keep_large_components(m, max(20, int(0.15 / pixel_size ** 2)))
            out[m & ~m2] = 0
    else:
        out[ys, xs] = np.where(best_res <= thresh, best, 0)
    return out


def facet_polygons(label_map, facets, pixel_size=0.1, simplify_frac=0.008):
    import cv2
    polys = {}
    for i, f in enumerate(facets):
        m = (label_map == i + 1).astype(np.uint8)
        if m.sum() < 10:
            continue
        # remove thin tendrils, then smooth the boundary
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        lab, ncc = ndi.label(m)
        if ncc > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            m = (lab == int(np.argmax(sizes))).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        eps = simplify_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            approx = c.reshape(-1, 2)
        x, y = px_to_xy(approx[:, 1], approx[:, 0], pixel_size)
        f.polygon = np.c_[x, y, f.z_at(x, y)]
        polys[i] = f.polygon
    return polys
