"""Simple solar-geometry model.

Used to (a) score roof facets by how much sunlight they receive over a year and
(b) report an energy proxy for the final layout. We integrate a clear-sky beam
model over a set of representative days and daylight hours: for each sun
direction ``s`` (unit vector in ENU) the useful irradiance on a surface with
unit normal ``n`` is proportional to ``max(0, n . s)``.
"""
from __future__ import annotations

import numpy as np


def solar_position(lat_deg: float, day_of_year: int, solar_hour: float) -> np.ndarray:
    """Sun unit vector in ENU (East, North, Up) for a local solar time."""
    lat = np.radians(lat_deg)
    decl = np.radians(23.45) * np.sin(np.radians(360.0 * (284 + day_of_year) / 365.0))
    hour_angle = np.radians(15.0 * (solar_hour - 12.0))

    sin_alt = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt = np.arcsin(sin_alt)
    cos_alt = np.cos(alt)

    # azimuth clockwise from north
    az = np.arctan2(-np.cos(decl) * np.sin(hour_angle),
                    np.sin(decl) * np.cos(lat) - np.cos(decl) * np.sin(lat) * np.cos(hour_angle))
    az = az % (2 * np.pi)

    e = cos_alt * np.sin(az)
    nn = cos_alt * np.cos(az)
    u = sin_alt
    return np.array([e, nn, u])


def sun_vectors(lat_deg: float, n_days: int = 24, step_hours: float = 0.5):
    """Representative set of daytime sun vectors (ENU) for a whole year."""
    days = np.linspace(15, 350, n_days)
    hours = np.arange(5.0, 19.0 + 1e-9, step_hours)
    vecs = []
    for d in days:
        for h in hours:
            s = solar_position(lat_deg, int(d), float(h))
            if s[2] > 0.02:  # above the horizon
                vecs.append(s)
    return np.array(vecs)


def annual_factor(normal: np.ndarray, suns: np.ndarray,
                  horizontal_ref: np.ndarray | None = None) -> float:
    """Sum of ``max(0, n.s)`` over all sun positions (arbitrary energy units)."""
    n = normal / np.linalg.norm(normal)
    inc = np.clip(suns @ n, 0.0, None)
    val = float(inc.sum())
    if horizontal_ref is not None:
        ref = np.clip(suns @ horizontal_ref, 0.0, None).sum()
        if ref > 0:
            val /= ref
    return val
