"""Terrain derivatives: slope and aspect (azimuth) from a DSM.

Aspect is reported as a compass azimuth (0 deg = North, 90 = East, increasing
clockwise). With a north-up raster the row index grows southwards, so the
gradient is converted from (row, col) to geographic (north, east) axes before
computing the downslope direction.
"""
from __future__ import annotations

import numpy as np


def gradient(dsm: np.ndarray, pixel_size: float = 0.1):
    """Central-difference gradient in geographic axes.

    Returns (dz_dx_east, dz_dy_north).
    """
    gy_row, gx_col = np.gradient(dsm.astype(np.float64), pixel_size)
    dz_dx = gx_col          # column increases eastwards
    dz_dy = -gy_row         # row increases southwards -> north gradient
    return dz_dx, dz_dy


def slope_aspect(dsm: np.ndarray, pixel_size: float = 0.1):
    """Return (slope_deg, aspect_deg) using Horn's method."""
    # Horn's 3x3 weighted gradient (more robust than plain central diff)
    z = dsm.astype(np.float64)
    zp = np.pad(z, 1, mode="edge")
    # 3x3 neighbourhoods
    a = zp[:-2, :-2]; b = zp[:-2, 1:-1]; c = zp[:-2, 2:]
    d = zp[1:-1, :-2]; e = zp[1:-1, 1:-1]; f = zp[1:-1, 2:]
    g = zp[2:, :-2]; h = zp[2:, 1:-1]; i = zp[2:, 2:]
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * pixel_size)
    # rows increase southwards, so north is (top row) - (bottom row)
    dz_dy_north = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * pixel_size)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy_north)))
    # downslope direction = -grad; azimuth clockwise from north = atan2(E, N)
    aspect = (np.degrees(np.arctan2(-dz_dx, -dz_dy_north))) % 360.0
    return slope.astype(np.float32), aspect.astype(np.float32)


def plane_from_normal(azimuth_deg: float, slope_deg: float):
    """Return (a, b) for a plane z = a*x + b*y with the given slope and azimuth.

    x is east, y is north. The unit normal is n = (-a, -b, 1)/|.|.
    """
    slope = np.radians(slope_deg)
    az = np.radians(azimuth_deg)
    # downslope horizontal direction (east, north)
    d_east = np.sin(az)
    d_north = np.cos(az)
    a = -np.tan(slope) * d_east
    b = -np.tan(slope) * d_north
    return a, b


def normal_azimuth_slope(a: float, b: float):
    """Inverse of :func:`plane_from_normal` for plane z = a*x + b*y."""
    gx, gy = a, b
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    azimuth = (np.degrees(np.arctan2(-gx, -gy))) % 360.0
    return float(azimuth), float(slope)
