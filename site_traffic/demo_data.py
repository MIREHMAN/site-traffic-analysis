"""
Synthetic test network loosely modelled on the Anchor & Bay TIA site
(NE corner of S. Egret Bay Blvd / Austin St, League City TX).

Geometry and AADT values are illustrative, NOT real counts. They are sized so the
frontage-road volumes match the TIA's peak-hour counts (Egret Bay ~1,170 two-way
approach vph in the AM peak -> ~15-17k AADT; Austin St ~230 vph -> ~2.5k AADT).
It deliberately includes the messy cases the filter must handle: ramps with
AADT, service/parking aisles, residential streets with no counts, low-count
streets, and a road split into many count segments.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .geo import LocalProjection

SITE_LAT, SITE_LON = 29.4998, -95.0597


def _line(proj, pts_xy):
    return {"type": "LineString", "coordinates": proj.inv(np.asarray(pts_xy, float)).round(7).tolist()}


def _split(pts, n):
    """Split a straight 2-point line into n consecutive pieces."""
    a, b = np.asarray(pts[0], float), np.asarray(pts[1], float)
    return [[a + (b - a) * i / n, a + (b - a) * (i + 1) / n] for i in range(n)]


def make_segments() -> pd.DataFrame:
    proj = LocalProjection(SITE_LON, SITE_LAT)
    rows = []
    sid = [0]

    def add(name, ref, cls, aadt, pts, year=2024, src="TxDOT (synthetic)"):
        sid[0] += 1
        rows.append(dict(segment_id=f"S{sid[0]:03d}", name=name, ref=ref, road_class=cls, aadt=aadt,
                         aadt_year=year if aadt is not None else None,
                         aadt_source=src if aadt is not None else None,
                         geometry=_line(proj, pts)))

    # S. Egret Bay Blvd (FM 270): runs NNW-SSE along the site's west side (~40 m away)
    def egret(y):  # x offset rotates the road ~20 deg like the aerial
        return np.array([-40 - 0.36 * y, y])
    for (a, b), aadt in zip([(-5200, -2300), (-2300, -350), (-350, 1400), (1400, 3200), (3200, 6000)],
                            [13800, 15100, 16200, 17400, 18900]):
        add("S Egret Bay Blvd", "FM 270", "primary", aadt, [egret(a), egret(b)])

    # Austin St: E-W collector along the site's south side (~45 m)
    add("Austin St", None, "tertiary", 1100, [(-1600, -45), (egret(-45)[0], -45)])
    add("Austin St", None, "tertiary", 2600, [(egret(-45)[0], -45), (1500, -45)])
    add("Austin St", None, "tertiary", 1900, [(1500, -45), (2600, -60)])

    # W Main St / FM 518: major E-W arterial ~2.8 km north
    for a, b, aadt in [(-7500, -3500, 26500), (-3500, -1000, 29800), (-1000, 1500, 31200), (1500, 6000, 27400)]:
        add("W Main St", "FM 518", "primary", aadt, [(a, 2800 + 0.05 * a), (b, 2800 + 0.05 * b)])

    # Marina Bay Dr / FM 2094: E-W ~2.4 km south
    for a, b, aadt in [(-4000, -800, 17200), (-800, 2500, 19600), (2500, 6500, 15800)]:
        add("Marina Bay Dr", "FM 2094", "secondary", aadt, [(a, -2400 - 0.08 * a), (b, -2400 - 0.08 * b)])

    # Calder Rd: N-S minor arterial ~3.4 km west
    for a, b, aadt in [(-4500, -800, 8200), (-800, 4200, 9600)]:
        add("Calder Rd", None, "secondary", aadt, [(-3400, a), (-3400, b)])

    # I-45 Gulf Fwy: ~6.4 km west (edge of the polygon) + ramps
    for a, b, aadt in [(-6000, -1500, 158000), (-1500, 2800, 166000), (2800, 7000, 171000)]:
        add("Gulf Fwy", "I 45", 1, aadt, [(-6400, a), (-6400, b)])   # FHWA FC 1 = Interstate
    add("I-45 Exit 23 Ramp", None, "motorway_link", 9100, [(-6400, 2600), (-6150, 2800)])
    add("I-45 Entrance Ramp", None, "motorway_link", 8700, [(-6150, 2800), (-6400, 3050)])

    # Hobbs Rd: N-S collector ~1.6 km east
    for a, b, aadt in [(-2400, 0, 4100), (0, 2800, 5200)]:
        add("Hobbs Rd", None, "tertiary", aadt, [(1600, a), (1600 + 30, b)])

    # Residential streets: mostly no counts, a few low / modest counts
    rng = np.random.default_rng(7)
    for i, y in enumerate(np.arange(-1800, 2400, 300)):
        if abs(y) < 150:
            continue
        aadt = None if i % 3 else float(rng.choice([250, 420, 900, 1300]))
        add(f"Residential St {i}", None, "residential", aadt, [(200, y), (1350, y)])
    for i, x in enumerate([-900, -650, -400]):
        add(f"Bay Colony Ln {i}", None, "residential", None, [(x, 200), (x, 1500)])
    add("Shady Oak Ct", None, "residential", 380, [(260, 120), (520, 120)])   # cul-de-sac next to site

    # Service / parking aisles beside the site
    add(None, None, "service", None, [(-10, 10), (120, 10)])
    add("Shopping Ctr Aisle", None, "parking_aisle", None, [(-20, 260), (180, 260)])
    add("Strip Center Drive", None, "service", 600, [(-60, 420), (150, 430)])

    df = pd.DataFrame(rows)
    return df


def make_drive_time_polygon(minutes: float = 10.0) -> dict:
    """Star-shaped synthetic isochrone: reaches further along the arterials."""
    proj = LocalProjection(SITE_LON, SITE_LAT)
    scale = minutes / 10.0
    pts = []
    for deg in np.arange(0, 360, 5):                 # bearing from north
        th = math.radians(deg)
        r = 4400 + 1000 * abs(math.cos(th)) ** 4 + 250 * math.cos(4 * th)   # longer along Egret Bay (N-S)
        if 245 <= deg <= 295:                        # west: arterial run out to I-45
            r = 7000
        pts.append((r * scale * math.sin(th), r * scale * math.cos(th)))
    pts = np.array(pts)
    ring = proj.inv(np.vstack([pts, pts[:1]]))
    return {"type": "Polygon", "coordinates": [ring.round(7).tolist()]}


def clip_to_polygon_stub(df: pd.DataFrame, polygon: dict) -> pd.DataFrame:
    """Stand-in for the DB's ST_Intersects: keep segments with any vertex inside."""
    from site_traffic.geo import points_in_polygon, to_line_parts, to_polygon_rings
    rings = to_polygon_rings(polygon)
    keep = []
    for g in df["geometry"]:
        pts = np.vstack(to_line_parts(g))
        # densify so long segments crossing the boundary are caught
        dense = np.vstack([np.linspace(pts[i], pts[i + 1], 30) for i in range(len(pts) - 1)])
        keep.append(points_in_polygon(dense, rings).any())
    return df[np.array(keep)].reset_index(drop=True)
