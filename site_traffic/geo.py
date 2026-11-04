"""
Lightweight geometry helpers (numpy only).

Lines are stored as a list of parts; each part is an (N, 2) array of lon/lat.
Shapely / GeoJSON / WKT inputs are converted on the way in, so the core does
not depend on shapely or geopandas (they remain optional in the host system).
"""
from __future__ import annotations

import json
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path

EARTH_R = 6_371_008.8
Parts = List[np.ndarray]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_wkt_line(wkt: str) -> Parts:
    wkt = wkt.strip()
    body = wkt[wkt.index("("):]
    body = body.strip()
    if wkt.upper().startswith("MULTILINESTRING"):
        inner = body[1:-1]
        chunks = [c.strip(" ()") for c in inner.split("),")]
    else:
        chunks = [body.strip("()")]
    parts = []
    for c in chunks:
        pts = [tuple(map(float, p.split()[:2])) for p in c.split(",") if p.strip()]
        parts.append(np.asarray(pts, dtype=float))
    return parts


def to_line_parts(geom) -> Parts:
    """Accept GeoJSON dict/str, WKT, shapely (Multi)LineString, or coords list."""
    if geom is None:
        return []
    if hasattr(geom, "geom_type"):  # shapely
        if geom.geom_type == "LineString":
            return [np.asarray(geom.coords, dtype=float)[:, :2]]
        if geom.geom_type == "MultiLineString":
            return [np.asarray(g.coords, dtype=float)[:, :2] for g in geom.geoms]
        return []
    if isinstance(geom, str):
        s = geom.strip()
        if s.startswith("{"):
            geom = json.loads(s)
        else:
            return _parse_wkt_line(s)
    if isinstance(geom, dict):
        t, c = geom.get("type"), geom.get("coordinates")
        if t == "LineString":
            return [np.asarray(c, dtype=float)[:, :2]]
        if t == "MultiLineString":
            return [np.asarray(p, dtype=float)[:, :2] for p in c]
        if t == "GeometryCollection":   # ST_Intersection can return these
            out: Parts = []
            for g in geom.get("geometries", []):
                out.extend(to_line_parts(g))
            return out
        return []
    arr = np.asarray(geom, dtype=float)
    if arr.ndim == 2:
        return [arr[:, :2]]
    return [np.asarray(p, dtype=float)[:, :2] for p in geom]


def to_polygon_rings(poly) -> List[np.ndarray]:
    """Exterior rings (lon/lat) of a Polygon / MultiPolygon in any common format."""
    if hasattr(poly, "geom_type"):
        if poly.geom_type == "Polygon":
            return [np.asarray(poly.exterior.coords)[:, :2]]
        return [np.asarray(p.exterior.coords)[:, :2] for p in poly.geoms]
    if isinstance(poly, str):
        poly = json.loads(poly)
    if isinstance(poly, dict):
        if poly.get("type") == "Feature":
            poly = poly["geometry"]
        if poly.get("type") == "FeatureCollection":
            poly = poly["features"][0]["geometry"]
        if poly["type"] == "Polygon":
            return [np.asarray(poly["coordinates"][0], dtype=float)[:, :2]]
        if poly["type"] == "MultiPolygon":
            return [np.asarray(p[0], dtype=float)[:, :2] for p in poly["coordinates"]]
        raise ValueError(f"Unsupported polygon type {poly['type']}")
    return [np.asarray(poly, dtype=float)[:, :2]]


def polygon_to_wkt(poly) -> str:
    rings = to_polygon_rings(poly)
    def ring_txt(r):
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[:1]])
        return "(" + ", ".join(f"{x:.7f} {y:.7f}" for x, y in r) + ")"
    if len(rings) == 1:
        return f"POLYGON({ring_txt(rings[0])})"
    return "MULTIPOLYGON(" + ", ".join(f"({ring_txt(r)})" for r in rings) + ")"


# --------------------------------------------------------------------------- #
# Local metric projection (equirectangular around the site; < 0.1 % error
# inside a 15-20 km drive-time polygon)
# --------------------------------------------------------------------------- #
class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = math.radians(1) * EARTH_R * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * EARTH_R

    def fwd(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float)
        return np.column_stack([(lonlat[..., 0] - self.lon0) * self.kx,
                                (lonlat[..., 1] - self.lat0) * self.ky]) if lonlat.ndim == 2 else \
            np.array([(lonlat[0] - self.lon0) * self.kx, (lonlat[1] - self.lat0) * self.ky])

    def inv(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            return np.array([xy[0] / self.kx + self.lon0, xy[1] / self.ky + self.lat0])
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def lonlat_to_webmercator(lonlat: np.ndarray) -> np.ndarray:
    lonlat = np.atleast_2d(np.asarray(lonlat, dtype=float))
    x = np.radians(lonlat[:, 0]) * 6378137.0
    y = np.log(np.tan(np.pi / 4 + np.radians(lonlat[:, 1]) / 2)) * 6378137.0
    return np.column_stack([x, y])


# --------------------------------------------------------------------------- #
# Measurements on projected (metre) coordinates
# --------------------------------------------------------------------------- #
def parts_length_m(parts_xy: Parts) -> float:
    return float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in parts_xy if len(p) > 1))


def nearest_point_on_parts(parts_xy: Parts, pt: np.ndarray) -> Tuple[np.ndarray, float, int, int, float]:
    """Return (nearest_xy, distance, part_idx, seg_idx, t) for point pt."""
    best = (None, math.inf, -1, -1, 0.0)
    for pi, p in enumerate(parts_xy):
        if len(p) < 2:
            continue
        a, b = p[:-1], p[1:]
        ab = b - a
        denom = (ab ** 2).sum(axis=1)
        denom[denom == 0] = 1e-12
        t = np.clip(((pt - a) * ab).sum(axis=1) / denom, 0, 1)
        proj = a + ab * t[:, None]
        d = np.linalg.norm(proj - pt, axis=1)
        i = int(np.argmin(d))
        if d[i] < best[1]:
            best = (proj[i], float(d[i]), pi, i, float(t[i]))
    return best


def split_at(part_xy: np.ndarray, seg_idx: int, t: float, pt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a polyline at a point; returns (start->pt, pt->end)."""
    before = np.vstack([part_xy[: seg_idx + 1], pt])
    after = np.vstack([pt, part_xy[seg_idx + 1:]])
    return before, after


def points_in_polygon(points_lonlat: np.ndarray, rings: List[np.ndarray]) -> np.ndarray:
    pts = np.atleast_2d(points_lonlat)
    inside = np.zeros(len(pts), dtype=bool)
    for r in rings:
        inside |= Path(r).contains_points(pts)
    return inside


def bearing_label(dx: float, dy: float) -> str:
    """Compass label for the direction a flow COMES FROM (vector from site to origin)."""
    ang = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    labels = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return labels[int(((ang + 22.5) % 360) // 45)]


def walk_along(part_xy: np.ndarray, distance_m: float) -> np.ndarray:
    """Point at a given distance from the start of a polyline (clamped)."""
    seg = np.diff(part_xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    if distance_m >= cum[-1]:
        return part_xy[-1].copy()
    i = int(np.searchsorted(cum, distance_m, side="right") - 1)
    t = (distance_m - cum[i]) / max(seg_len[i], 1e-9)
    return part_xy[i] + seg[i] * t
