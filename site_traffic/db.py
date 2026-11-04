"""
Step 1 - fetch every road segment that intersects the drive-time polygon.

Kept separate from the model so the host system can swap the data source
(PostGIS, BigQuery GIS, a file) without touching the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .geo import polygon_to_wkt


@dataclass
class SegmentTableSchema:
    """Map your table's column names onto the fields the model expects."""
    table: str = "public.road_segments"
    geom: str = "geom"                    # LineString / MultiLineString, any SRID
    srid: int = 4326                      # SRID of `geom`
    segment_id: str = "segment_id"
    name: str = "name"
    ref: Optional[str] = "ref"            # route number, e.g. FM 270 / I-45
    road_class: str = "road_class"        # OSM highway tag or FHWA FC 1-7
    aadt: str = "aadt"
    aadt_year: Optional[str] = "aadt_year"
    aadt_source: Optional[str] = "aadt_source"
    speed_kph: Optional[str] = None       # posted/modelled speed if you have it
    oneway: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)   # {output_name: column}


def build_intersect_sql(schema: SegmentTableSchema, clip: bool = True, paramstyle: str = "pyformat") -> str:
    ph = "%(poly_wkt)s" if paramstyle == "pyformat" else ":poly_wkt"
    poly = f"ST_GeomFromText({ph}, 4326)"
    poly_native = poly if schema.srid == 4326 else f"ST_Transform({poly}, {schema.srid})"
    geom_out = (f"ST_Intersection(s.{schema.geom}, {poly_native})" if clip else f"s.{schema.geom}")
    if schema.srid != 4326:
        geom_out = f"ST_Transform({geom_out}, 4326)"

    def col(c, alias):
        return f"s.{c} AS {alias}" if c else f"NULL AS {alias}"

    cols = [
        col(schema.segment_id, "segment_id"), col(schema.name, "name"), col(schema.ref, "ref"),
        col(schema.road_class, "road_class"), col(schema.aadt, "aadt"),
        col(schema.aadt_year, "aadt_year"), col(schema.aadt_source, "aadt_source"),
        col(schema.speed_kph, "speed_kph"), col(schema.oneway, "oneway"),
    ] + [f"s.{c} AS {a}" for a, c in schema.extra.items()]
    return (
        "SELECT " + ", ".join(cols) + f", ST_AsGeoJSON({geom_out}) AS geometry\n"
        f"FROM {schema.table} s\n"
        f"WHERE s.{schema.geom} && {poly_native}\n"          # index-assisted bbox filter
        f"  AND ST_Intersects(s.{schema.geom}, {poly_native})"
    )


def fetch_intersecting_segments(conn, drive_time_polygon, schema: SegmentTableSchema | None = None,
                                clip: bool = True) -> pd.DataFrame:
    """
    Return all segments intersecting the polygon as a DataFrame with a GeoJSON
    `geometry` column (clipped to the polygon when clip=True).

    conn: a DB-API connection (psycopg2/psycopg 3) or a SQLAlchemy engine/connection.
    """
    schema = schema or SegmentTableSchema()
    wkt = polygon_to_wkt(drive_time_polygon)
    is_sqlalchemy = type(conn).__module__.startswith("sqlalchemy")
    if is_sqlalchemy:
        from sqlalchemy import text
        sql = text(build_intersect_sql(schema, clip, paramstyle="named"))
        return pd.read_sql(sql, conn, params={"poly_wkt": wkt})
    sql = build_intersect_sql(schema, clip, paramstyle="pyformat")
    return pd.read_sql(sql, conn, params={"poly_wkt": wkt})
