"""3-D reconstruction of the regularised roof and mesh export.

The surface is sampled from the **piecewise-planar** facet assignment on a
regular grid. A quad is only emitted when all four corners belong to the *same*
facet, so every patch is exactly planar and no artificial vertical walls are
created between facets. The result is a clean, faceted reconstruction of the
roof rather than the noisy measured DSM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .coords import px_to_xy
from .logging_utils import get_logger
from .segmentation import Facet

log = get_logger(__name__)


def build_roof_mesh(facets: list[Facet], label_map: np.ndarray, mask: np.ndarray = None,
                    pixel_size: float = 0.1, step: int = 3):
    """Sample the piecewise-planar roof on a grid -> (vertices, faces)."""
    if mask is None:
        mask = label_map > 0
    lab = np.where(mask, label_map, 0)
    ys, xs = np.nonzero(lab > 0)
    if len(ys) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    rows = list(range(r0, r1 + 1, step))
    cols = list(range(c0, c1 + 1, step))
    if rows[-1] != r1:
        rows.append(r1)
    if cols[-1] != c1:
        cols.append(c1)

    verts: list[list[float]] = []
    vid: dict[tuple[int, int], int] = {}
    for r in rows:
        for c in cols:
            fid = int(lab[r, c])
            if fid <= 0:
                continue
            f = facets[fid - 1]
            x, y = px_to_xy(np.array([r]), np.array([c]), pixel_size)
            vid[(r, c)] = len(verts)
            verts.append([float(x[0]), float(y[0]), float(f.z_at(x[0], y[0]))])

    faces: list[list[int]] = []
    for i in range(len(rows) - 1):
        r, r2 = rows[i], rows[i + 1]
        for j in range(len(cols) - 1):
            c, c2 = cols[j], cols[j + 1]
            corners = [(r, c), (r2, c), (r2, c2), (r, c2)]
            fids = {int(lab[a, b]) for a, b in corners}
            if len(fids) != 1 or 0 in fids:
                continue
            if not all(k in vid for k in corners):
                continue
            a, b, d, e = (vid[k] for k in corners)
            faces.append([a, b, e])
            faces.append([b, d, e])

    v = np.asarray(verts, float)
    fc = np.asarray(faces, int)
    log.info("roof mesh: %d vertices, %d triangles (grid step %d px)", len(v), len(fc), step)
    return v, fc


def export_obj(path: str | Path, verts: np.ndarray, faces: np.ndarray,
               name: str = "reconstructed roof mesh"):
    path = Path(path)
    with path.open("w") as fh:
        fh.write(f"# {name} (roofsolar)\n")
        for x, y, z in verts:
            fh.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for face in faces:
            fh.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")
    log.info("wrote %s", path)


def export_panel_obj(path: str | Path, panels):
    path = Path(path)
    with path.open("w") as fh:
        fh.write("# Solar panel footprints (roofsolar)\n")
        for p in panels:
            for x, y, z in p.corners_xyz:
                fh.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for i in range(len(panels)):
            b = i * 4
            fh.write(f"f {b + 1} {b + 2} {b + 3} {b + 4}\n")
    log.info("wrote %s", path)
