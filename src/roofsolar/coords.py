"""Pixel <-> local metric coordinate conversion.

Local coordinates are a right-handed metric frame with **x pointing east** and
**y pointing north**, which makes azimuth/ENU maths (slope, aspect, sun
vectors) consistent. A north-up raster has rows increasing southwards, hence
the negated y.
"""
from __future__ import annotations

import numpy as np


def px_to_xy(rows, cols, pixel_size: float = 0.1):
    """(row, col) pixel indices -> local (x=east, y=north) metres."""
    return np.asarray(cols, float) * pixel_size, -np.asarray(rows, float) * pixel_size


def xy_to_px(x, y, pixel_size: float = 0.1):
    """local (x, y) metres -> (row, col) pixel indices."""
    return -np.asarray(y, float) / pixel_size, np.asarray(x, float) / pixel_size
