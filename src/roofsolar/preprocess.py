"""Pre-processing: denoising, ground estimation and above-ground height.

The DSM is a smoothed surface, so the pipeline (a) applies a light
edge-preserving denoise, (b) estimates a bare-earth surface with a large-window
low percentile (robust for the gently varying terrain here) and (c) derives the
height-above-ground (HAG) field used to separate buildings from vegetation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def denoise(dsm: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Light Gaussian denoise; the DSM is already smooth so sigma stays small."""
    out = dsm.astype(np.float32)
    finite = np.isfinite(out)
    if not finite.all():
        out = np.where(finite, out, np.nanmean(out[finite]))
    if sigma > 0:
        out = ndi.gaussian_filter(out, sigma)
    return out.astype(np.float32)


def local_roughness(dsm: np.ndarray, window: int = 9) -> np.ndarray:
    """Local standard deviation - high over vegetation, low over flat planes."""
    m = ndi.uniform_filter(dsm, window)
    m2 = ndi.uniform_filter(dsm * dsm, window)
    return np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)


def estimate_ground(dsm: np.ndarray, downsample: int = 10, window: int = 31,
                    percentile: float = 10.0, smooth: float = 2.0) -> np.ndarray:
    """Estimate a bare-earth surface with a robust low percentile.

    The DSM is block-median downsampled (robust against spikes), filtered with a
    low percentile over a large window so buildings and trees are rejected, then
    smoothed and upsampled back to the full grid.
    """
    H, W = dsm.shape
    f = downsample
    hh, ww = H // f, W // f
    block = dsm[:hh * f, :ww * f].reshape(hh, f, ww, f).transpose(0, 2, 1, 3).reshape(hh, ww, f * f)
    ds = np.median(block, axis=2)
    g = ndi.percentile_filter(ds, percentile, size=window)
    g = ndi.median_filter(g, size=3)
    ground = ndi.zoom(g, (H / g.shape[0], W / g.shape[1]), order=1)[:H, :W]
    if ground.shape != (H, W):
        ground = np.pad(ground, ((0, H - ground.shape[0]), (0, W - ground.shape[1])), mode="edge")
    if smooth > 0:
        ground = ndi.gaussian_filter(ground, smooth)
    return ground.astype(np.float32)


def vegetation_mask(rgb: np.ndarray, exg_threshold: float = 18.0) -> np.ndarray:
    """Colour-based vegetation mask (excess-green index)."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    exg = 2 * g - r - b
    return (exg > exg_threshold) & (g > r) & (g > b)


def above_ground(dsm: np.ndarray, ground: np.ndarray) -> np.ndarray:
    return (dsm - ground).astype(np.float32)
