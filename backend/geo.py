"""Distance helpers. Local equirectangular projection (metres) around the trip centre.
Valid for city-scale extents (~10 km): error well under 0.1%, far better than raw degrees."""
import numpy as np

R_EARTH = 6371008.8


def make_projector(lon0: float, lat0: float):
    c = np.cos(np.radians(lat0))

    def proj(lon, lat):
        x = np.radians(np.asarray(lon, dtype=float) - lon0) * R_EARTH * c
        y = np.radians(np.asarray(lat, dtype=float) - lat0) * R_EARTH
        return x, y

    return proj


def point_to_segments(px, py, ax, ay, bx, by):
    """Distance from one point to many segments A->B (all in metres)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))
