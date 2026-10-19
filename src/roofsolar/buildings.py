"""Building detection and target selection.

Buildings are found from the above-ground height field combined with surface
planarity (roofs are planar, vegetation is rough) and a colour vegetation mask.
The target building is the largest connected component that overlaps the
``dsm_clipped`` parcel footprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class BuildingResult:
    mask: np.ndarray                 # full-scene building mask
    labels: np.ndarray               # connected-component labels
    target_id: int                   # label id of the target building
    target_mask: np.ndarray          # boolean mask of the target building


def building_mask(hag: np.ndarray, roughness: np.ndarray, veg: np.ndarray,
                  min_hag: float = 1.7, max_hag: float = 8.5,
                  max_roughness: float = 0.18) -> np.ndarray:
    mask = (hag > min_hag) & (hag < max_hag) & (roughness < max_roughness) & (~veg)
    mask = ndi.binary_opening(mask, np.ones((3, 3)))
    mask = ndi.binary_closing(mask, np.ones((9, 9)))
    mask = ndi.binary_fill_holes(mask)
    return mask


def _planar_fraction(comp: np.ndarray, roughness: np.ndarray, thr: float = 0.12) -> float:
    return float(np.mean(roughness[comp] < thr))


def detect_buildings(mask: np.ndarray, roughness: np.ndarray,
                     min_area_px: int = 400) -> tuple[np.ndarray, np.ndarray]:
    labels, n = ndi.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        if sizes[i] < min_area_px:
            continue
        comp = labels == i
        # reject long thin vegetation ridges: require reasonable planarity
        if _planar_fraction(comp, roughness) < 0.15:
            continue
        keep[i] = True
    labels = np.where(keep[labels], labels, 0)
    return labels, sizes


def select_target(labels: np.ndarray, sizes: np.ndarray, roi: Optional[np.ndarray]) -> int:
    """Largest surviving component overlapping the clipped parcel ROI."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        raise RuntimeError("no building components found")
    best, best_area = int(ids[0]), -1
    for i in ids:
        if roi is not None and not np.any(roi & (labels == i)):
            continue
        if sizes[i] > best_area:
            best, best_area = int(i), int(sizes[i])
    return best


def run(hag: np.ndarray, roughness: np.ndarray, veg: np.ndarray,
        roi: Optional[np.ndarray], min_area_px: int = 400, **mask_kwargs) -> BuildingResult:
    mask = building_mask(hag, roughness, veg, **mask_kwargs)
    labels, sizes = detect_buildings(mask, roughness, min_area_px=min_area_px)
    target_id = select_target(labels, sizes, roi)
    target_mask = labels == target_id
    log.info("target building id=%d area=%d px (%.1f m^2)",
             target_id, int(target_mask.sum()), target_mask.sum() * 0.1 * 0.1)
    return BuildingResult(mask=mask, labels=labels, target_id=target_id, target_mask=target_mask)
