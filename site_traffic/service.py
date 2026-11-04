"""
The single pipeline behind both entry points.

    site_traffic.api    HTTP:  POST|GET /api/test/traffic-data/
    run.py              CLI:   python run.py --lat 29.506 --lng -95.0676 --minutes 5 --vfp 12

Both call `run_site(params)`: segments -> corridors -> trips -> JSON body, and with
`snapshot` also the maps (trip map + one satellite map per top-5 corridor) and the PDF report.

Segment source (first that applies):
  * env TRAFFIC_DEMO=1  -> the synthetic League City network (site_traffic.demo_data)
  * otherwise PostGIS   -> PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD, table
                           TRAFFIC_SCHEMA.TRAFFIC_TABLE (default tezintel_datastore.traffic_counts),
                           columns auto-mapped (override with TRAFFIC_COL_NAME/CLASS/AADT).
Thresholds and constants: site_traffic/config.py (restart after editing; uvicorn --reload does it).
Drive-time polygon: `drive_time_polygon` (GeoJSON / WKT) if given; else a Mapbox isochrone when
MAPBOX_TOKEN is set; else a circle sized from the drive time (reported in `polygon_source`).
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import urllib.request
from typing import Any, Dict, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

try:   # load <project>/.env; real environment variables win
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import TrafficAnalysis, _json_default  # noqa: E402
from .config import DRIVE_TIME_AVG_SPEED_KPH, SitePayload  # noqa: E402
from .db import SegmentTableSchema  # noqa: E402

DEFAULT_OUTPUT_ROOT = os.path.join("output", "traffic_analysis")
REPORT_NAME = "site_traffic_report.pdf"
MILES_PER_M = 1 / 1609.344

# candidate column names, first case-insensitive match wins
CANDIDATES = {
    "geom": ["geom", "geometry", "the_geom", "wkb_geometry", "shape"],
    "segment_id": ["segment_id", "id", "gid", "objectid", "station_id", "count_id", "fid"],
    "name": ["name", "road_name", "street_name", "street", "rd_name", "roadway", "location", "rte_nm"],
    "ref": ["ref", "route", "route_id", "route_name", "rte_nm", "hwy", "highway_ref"],
    "road_class": ["road_class", "func_class", "functional_class", "f_system", "fclass", "highway", "class", "rte_prfx"],
    "aadt": ["aadt", "aadt_cur", "current_aadt", "adt", "volume", "traffic_count", "count"],
    "aadt_year": ["aadt_year", "year", "count_year", "data_year"],
    "aadt_source": ["aadt_source", "source", "data_source"],
    "speed_kph": ["speed_kph", "speed", "speed_limit", "maxspeed"],
}

_lock = threading.Lock()
_state: Dict[str, Any] = {}


class ServiceError(Exception):
    """Bad input (422) or a failed dependency (503); the API maps `status` to the HTTP code."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def output_root() -> str:
    return os.getenv("TRAFFIC_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT)


# --------------------------------------------------------------------------- #
# Segment source
# --------------------------------------------------------------------------- #
def _connect():
    import psycopg2
    return psycopg2.connect(host=os.getenv("PGHOST", "localhost"), port=int(os.getenv("PGPORT", 5433)),
                            dbname=os.getenv("PGDATABASE", "tezintel_testing"), user=os.getenv("PGUSER", "postgres"),
                            password=os.getenv("PGPASSWORD", "postgres"))


def _auto_schema(conn) -> SegmentTableSchema:
    schema = os.getenv("TRAFFIC_SCHEMA", "tezintel_datastore")
    table = os.getenv("TRAFFIC_TABLE", "traffic_counts")
    cols = pd.read_sql("SELECT column_name FROM information_schema.columns "
                       "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
                       conn, params=(schema, table))
    if cols.empty:
        raise RuntimeError(f"Table {schema}.{table} not found")
    geo = pd.read_sql("SELECT f_geometry_column, srid, type FROM geometry_columns "
                      "WHERE f_table_schema=%s AND f_table_name=%s", conn, params=(schema, table))
    names = {c.lower(): c for c in cols["column_name"]}
    pick = {f: next((names[c] for c in cands if c in names), None) for f, cands in CANDIDATES.items()}
    for env, field_ in (("TRAFFIC_COL_NAME", "name"), ("TRAFFIC_COL_CLASS", "road_class"),
                        ("TRAFFIC_COL_AADT", "aadt")):
        if os.getenv(env):
            pick[field_] = os.environ[env]
    srid = 4326
    if len(geo):
        pick["geom"] = geo.iloc[0]["f_geometry_column"]
        if "POINT" in str(geo.iloc[0]["type"]).upper():
            raise RuntimeError("traffic_counts holds POINT stations; the model needs road LINE segments")
        if int(geo.iloc[0]["srid"]) > 0:
            srid = int(geo.iloc[0]["srid"])
    missing = [f for f in ("geom", "segment_id", "aadt") if not pick[f]]
    if missing:
        raise RuntimeError(f"Could not map columns {missing}; available: {list(cols['column_name'])}")
    qualified = f'{schema}."{table}"' if table != table.lower() else f"{schema}.{table}"
    return SegmentTableSchema(table=qualified, geom=pick["geom"], srid=srid, segment_id=pick["segment_id"],
                              name=pick["name"] or pick["segment_id"], ref=pick["ref"],
                              road_class=pick["road_class"], aadt=pick["aadt"], aadt_year=pick["aadt_year"],
                              aadt_source=pick["aadt_source"], speed_kph=pick["speed_kph"])


def _get_analysis() -> TrafficAnalysis:
    with _lock:
        if "analysis" in _state:
            return _state["analysis"]
        if os.getenv("TRAFFIC_DEMO") == "1":
            from .demo_data import clip_to_polygon_stub, make_segments
            segs = make_segments()
            analysis = TrafficAnalysis(segment_source=lambda poly: clip_to_polygon_stub(segs, poly))
        else:
            conn = _connect()
            conn.autocommit = True
            analysis = TrafficAnalysis(conn, schema=_auto_schema(conn))
        _state["analysis"] = analysis
        return analysis


def _reset_connection():
    with _lock:
        _state.pop("analysis", None)


# --------------------------------------------------------------------------- #
# Drive-time polygon
# --------------------------------------------------------------------------- #
def _parse_wkt_polygon(wkt: str) -> dict:
    m = re.match(r"\s*(MULTI)?POLYGON\s*\(\(+(.*?)\)+\s*", wkt, re.I | re.S)
    if not m:
        raise ValueError("Only POLYGON / MULTIPOLYGON WKT is supported")
    ring = m.group(2).split(")")[0]
    pts = [[float(v) for v in p.split()[:2]] for p in ring.split(",") if p.strip()]
    return {"type": "Polygon", "coordinates": [pts]}


def _normalise_polygon(poly):
    if isinstance(poly, str):
        poly = _parse_wkt_polygon(poly) if not poly.strip().startswith("{") else json.loads(poly)
    return poly


def _circle(lat: float, lng: float, minutes: float, avg_kph: float = DRIVE_TIME_AVG_SPEED_KPH) -> dict:
    r = avg_kph * 1000 / 60 * minutes / 1.3                # km/h -> m/min, / circuity
    pts = [[lng + r * math.sin(t) / (111320 * math.cos(math.radians(lat))), lat + r * math.cos(t) / 110540]
           for t in (i * math.pi / 32 for i in range(65))]
    return {"type": "Polygon", "coordinates": [pts]}


def _mapbox_isochrone(lat: float, lng: float, minutes: float) -> Optional[dict]:
    token = os.getenv("MAPBOX_TOKEN")
    if not token:
        return None
    url = (f"https://api.mapbox.com/isochrone/v1/mapbox/driving/{lng},{lat}"
           f"?contours_minutes={int(round(min(max(minutes, 1), 60)))}&polygons=true&access_token={token}")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.load(r)["features"][0]["geometry"]
    except Exception:
        return None


def resolve_polygon(lat, lng, minutes, supplied, avg_kph: float = DRIVE_TIME_AVG_SPEED_KPH):
    if supplied:
        return _normalise_polygon(supplied), "provided"
    iso = _mapbox_isochrone(lat, lng, minutes)
    if iso:
        return iso, "mapbox_isochrone"
    return _circle(lat, lng, minutes, avg_kph), "circle_approximation"


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #
def _first(d: dict, *keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _truthy(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y") if v is not None else False


def _peak_trip_ends(hourly: np.ndarray, start: int, end: int) -> float:
    return float(2 * hourly[start:end].max())              # entering visits x 2 = trip ends (50:50)


def build_response(res, lat, lng, minutes, polygon_source) -> dict:
    trips = res.trips
    max_pot = max((t.corridor.potential for t in trips.corridors), default=0) or 1
    top = [{
        "rank": t.corridor.rank, "corridor_name": t.corridor.label,
        "distance_miles": round(t.corridor.distance_m * MILES_PER_M, 2),
        "distance_display": f"{t.corridor.distance_m * MILES_PER_M:.2f} mi",
        "aadt_cur": t.corridor.aadt,
        "aadt_estimated": t.corridor.aadt_estimated,
        "trip_share": round(t.share_of_site, 4),
        "modeled_trip_share": round(t.share_of_site, 4),
        "relevance_score": round(t.corridor.potential / max_pot, 3),
        "site_visits_per_day": round(t.visits, 1),
        "trip_type": t.trip_type,
        "is_access_road": t.corridor.is_access_road,
        "context_only": t.corridor.context_only,
    } for t in trips.corridors][:max(res.config.top_k, 1) + 1]

    summ = res.summary()
    seg = res.segments
    real = seg[seg["drop_reason"] == ""]
    aadt = real["aadt"].astype(float)
    return {
        "status": "success",
        "site_coordinates": {"latitude": lat, "longitude": lng},
        "drive_time_minutes": minutes,
        "polygon_source": polygon_source,
        "details": {
            "total_segments": int(len(seg)),
            "traffic_segments": int(len(real)),
            "total_traffic": float(aadt.sum()),
            "avg_traffic": round(float(aadt.mean()), 2) if len(aadt) else 0.0,
            "candidate_corridors": summ["candidate_corridors"],
            "drop_reasons": summ["drop_reasons"],
        },
        "top_5_corridors": top,
        "trip_generation_analysis": {
            "summary": {
                "daily_trips": round(summ["site_trip_ends_per_day"]),
                "daily_visits": summ["site_visits_per_day"],
                "am_peak_trips": round(_peak_trip_ends(trips.hourly_visits, 7, 9)),
                "pm_peak_trips": round(_peak_trip_ends(trips.hourly_visits, 16, 18)),
                "binding_constraint": trips.binding_constraint,
            },
            "ite_supply": trips.ite,
            "components": res.to_records(),
            "dayparts": summ["dayparts"],
            "pass_by_scenarios": trips.checks,
        },
        "warnings": res.warnings,
    }


def _snapshot(res, body: dict, polygon_source: str, out_dir: str, site_key: str):
    """Files for the site: data (CSV/JSON/GeoJSON), maps and the PDF report."""
    from . import viz
    from .report import build_pdf
    try:
        files = {**res.save(out_dir), **viz.render_all(res, out_dir)}
        body["snapshot"] = {"status": "saved", "output_dir": os.path.abspath(out_dir),
                            "files": {k: os.path.basename(v) for k, v in files.items()},
                            "basemap": {k: dict(v) for k, v in viz.BASEMAP_STATUS.items()}}
    except Exception as e:
        body["snapshot"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        body["pdf_report"] = {"status": "skipped", "detail": "snapshot rendering failed"}
        return
    try:
        pdf = build_pdf(res, files, os.path.join(out_dir, REPORT_NAME), polygon_source=polygon_source,
                        basemap={k: dict(v) for k, v in viz.BASEMAP_STATUS.items()})
        body["pdf_report"] = {"status": "generated", "filename": REPORT_NAME, "path": os.path.abspath(pdf),
                              "download_url": f"/api/test/traffic-report/{site_key}"}
    except Exception as e:
        body["pdf_report"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}


def run_site(params: dict, out_root: Optional[str] = None) -> dict:
    """The whole pipeline. `params` accepts lat/lng (or site_lat/site_lng), drive_time_minutes, vfp (alias mpds),
    drive_time_polygon, site_id, site_name (the report title), snapshot (maps + PDF). Returns a JSON-safe dict."""
    lat = _first(params, "lat", "site_lat", "latitude")
    lng = _first(params, "lng", "lon", "site_lng", "site_lon", "longitude")
    if lat is None or lng is None:
        raise ServiceError(422, "lat and lng (or site_lat / site_lng) are required")
    try:
        lat, lng = float(lat), float(lng)
        minutes = float(_first(params, "drive_time_minutes", "minutes") or 5)
        vfp = _first(params, "vfp", "mpds", "mpd", "fueling_positions")      # fueling positions (MPDs)
        vfp = int(float(vfp)) if vfp is not None else None
        if vfp is not None and vfp <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ServiceError(422, "lat, lng and drive_time_minutes must be numeric and vfp a positive number")
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ServiceError(422, "lat/lng out of range")
    snapshot = _truthy(params.get("snapshot")) or _truthy(params.get("pdf"))

    try:
        poly, source = resolve_polygon(lat, lng, minutes, params.get("drive_time_polygon"))
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise ServiceError(422, f"Invalid drive_time_polygon: {e}")

    payload = SitePayload(site_lat=lat, site_lon=lng, drive_time_polygon=poly, drive_time_minutes=minutes,
                          vfp=vfp, site_id=params.get("site_id"),
                          site_name=_first(params, "site_name", "name", "title"))
    try:
        res = _get_analysis().run(payload)
    except Exception as e:
        _reset_connection()
        raise ServiceError(503, f"Traffic analysis failed: {type(e).__name__}: {e}")

    body = build_response(res, lat, lng, minutes, source)
    if snapshot:
        site_key = f"site_{lat}_{lng}"
        _snapshot(res, body, source, os.path.join(out_root or output_root(), site_key), site_key)
    return json.loads(json.dumps(body, default=_json_default))
