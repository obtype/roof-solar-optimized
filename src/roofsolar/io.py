"""Raster input/output and georeferencing helpers.

All internal processing happens on the north-up grid of ``dsm_full`` using
pixel coordinates, with metric distances obtained through ``Grid.pixel_size``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Grid:
    """Regular north-up raster grid (affine transform + CRS)."""

    transform: "rasterio.Affine"
    width: int
    height: int
    crs: Optional[object] = None

    @property
    def pixel_size(self) -> float:
        return float(abs(self.transform.a))

    def world_xy(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert (row, col) pixel centres to world (easting, northing)."""
        xs = self.transform.c + (cols + 0.5) * self.transform.a
        ys = self.transform.f + (rows + 0.5) * self.transform.e
        return xs, ys

    def rowcol(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.transform.c) / self.transform.a)
        row = int((y - self.transform.f) / self.transform.e)
        return row, col


def load_raster(path: str | Path):
    """Load first band (or all bands) plus grid metadata."""
    path = Path(path)
    with rasterio.open(path) as ds:
        arr = ds.read()
        grid = Grid(ds.transform, ds.width, ds.height, ds.crs)
        nodata = ds.nodata
    log.info("loaded %s shape=%s nodata=%s", path.name, arr.shape, nodata)
    return arr, grid, nodata


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """(bands, H, W) -> (H, W, bands)."""
    if arr.ndim == 3:
        return np.transpose(arr, (1, 2, 0))
    return arr


def load_scene(data_dir: str | Path) -> dict:
    """Load RGB, full DSM and clipped DSM from ``data_dir``."""
    data_dir = Path(data_dir)
    rgb_arr, rgb_grid, _ = load_raster(data_dir / "rgb.tif")
    dsm_arr, dsm_grid, dsm_nodata = load_raster(data_dir / "dsm_full.tif")
    clip_arr, clip_grid, clip_nodata = load_raster(data_dir / "dsm_clipped.tif")

    rgb = _to_hwc(rgb_arr)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return {
        "rgb": rgb,
        "rgb_grid": rgb_grid,
        "dsm": dsm_arr[0].astype(np.float32),
        "dsm_grid": dsm_grid,
        "dsm_nodata": dsm_nodata,
        "dsm_clipped": clip_arr[0].astype(np.float32),
        "dsm_clipped_grid": clip_grid,
        "dsm_clipped_nodata": clip_nodata,
    }


def clip_roi_mask(serie: dict) -> np.ndarray:
    """Boolean mask of the clipped DSM footprint mapped onto the dsm_full grid.

    The clipped raster covers a rotated rectangular parcel; its valid pixels are
    transferred to the full grid using the affine transforms.
    """
    full: Grid = serie["dsm_grid"]
    cgrid: Grid = serie["dsm_clipped_grid"]
    clip = serie["dsm_clipped"]
    nodata = serie["dsm_clipped_nodata"]

    valid = clip != nodata
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        valid = np.isfinite(clip)
    rows, cols = np.nonzero(valid)
    xs, ys = cgrid.world_xy(rows, cols)
    # top-left world corner of the clipped raster
    x0, y0 = xs.min(), ys.max()
    fr = int(round((full.transform.f - y0) / abs(full.transform.e)))
    fc = int(round((x0 - full.transform.c) / abs(full.transform.a)))

    roi = np.zeros((full.height, full.width), bool)
    h, w = clip.shape
    r1, c1 = min(fr + h, full.height), min(fc + w, full.width)
    r0, c0 = max(fr, 0), max(fc, 0)
    src = valid[r0 - fr:r1 - fr, c0 - fc:c1 - fc]
    roi[r0:r1, c0:c1] = src
    return roi


def fill_nodata(dsm: np.ndarray, nodata, reference: Optional[np.ndarray] = None,
                max_iter: int = 200) -> np.ndarray:
    """Robustly fill invalid DSM cells.

    Strategy: if a co-registered ``reference`` DSM is supplied, sample it where
    the target is invalid; any remaining holes are filled by iterative
    neighbour averaging (a fast, deterministic approximation of Laplace
    inpainting). This avoids importing heavy inpainting packages.
    """
    out = dsm.astype(np.float32).copy()
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        missing = out == np.float32(nodata)
    else:
        missing = ~np.isfinite(out)
    missing |= ~np.isfinite(out)
    n_missing = int(missing.sum())
    if n_missing == 0:
        return out
    log.info("filling %d invalid cells", n_missing)

    if reference is not None:
        ref = reference.astype(np.float32)
        use = missing & np.isfinite(ref)
        out[use] = ref[use]
        missing = missing & ~np.isfinite(out)

    # iterative neighbour averaging for whatever is left
    filled = out.copy()
    seed_valid = ~missing
    if missing.any():
        filled[missing] = np.nan
        for _ in range(max_iter):
            holes = np.isnan(filled)
            if not holes.any():
                break
            neigh = np.zeros_like(filled)
            cnt = np.zeros_like(filled)
            for sh in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                roll = np.roll(np.roll(filled, sh[0], 0), sh[1], 1)
                valid = ~np.isnan(roll)
                if sh[0] == 1:
                    valid[-1, :] = False
                if sh[0] == -1:
                    valid[0, :] = False
                if sh[1] == 1:
                    valid[:, -1] = False
                if sh[1] == -1:
                    valid[:, 0] = False
                neigh[valid] += roll[valid]
                cnt[valid] += 1
            upd = holes & (cnt > 0)
            filled[upd] = neigh[upd] / cnt[upd]
        out = np.where(np.isnan(filled), np.nanmean(filled), filled)
    return out.astype(np.float32)
