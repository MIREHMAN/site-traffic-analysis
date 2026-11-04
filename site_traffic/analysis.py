"""
TrafficAnalysis - the layer the host system calls.

    analysis = TrafficAnalysis(segment_source=my_db_conn)          # or a callable
    result   = analysis.run(SitePayload(lat, lon, drive_time_polygon, vfp=12))
    result.to_records()          # Appendix-A style JSON rows
    result.to_geojson()          # selected corridors + approach arrows
    result.save("out/")          # CSV / JSON / GeoJSON

Maps and the PDF report are produced by site_traffic.service (the single entry point).

Each step is also a public method so it can be called (and cached) on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import MAX_CONNECTOR_M, NETWORK_SNAP_M, USE_NETWORK, SitePayload, TripModelConfig
from .db import SegmentTableSchema, fetch_intersecting_segments
from .geo import LocalProjection, points_in_polygon, to_polygon_rings
from .potential import Corridor, aggregate_corridors, score_segments, select_corridors
from .routing import RoadNetwork
from .segments import build_corridors, filter_real_traffic_segments
from .tripgen import SiteTrips, generate_site_trips

SegmentSource = Union[Callable[[object], pd.DataFrame], pd.DataFrame, object]


@dataclass
class TrafficAnalysisResult:
    payload: SitePayload
    config: TripModelConfig
    projection: LocalProjection
    polygon_rings: List[np.ndarray]
    segments: pd.DataFrame                  # every fetched segment + drop_reason + scores
    corridors: List[Corridor]               # every candidate corridor, ranked by potential
    selected: List[Corridor]
    trips: SiteTrips
    warnings: List[str] = field(default_factory=list)

    # ---------------- tabular outputs ---------------- #
    def corridor_table(self) -> pd.DataFrame:
        trips = {t.corridor.corridor_id: t for t in self.trips.corridors}
        rows = []
        for c in self.corridors:
            t = trips.get(c.corridor_id)
            rows.append({
                "rank": c.rank, "selected": c.selected, "context_only": c.context_only,
                "corridor_id": c.corridor_id, "name": c.label, "road_class": c.road_class,
                "aadt": c.aadt, "aadt_range": f"{c.aadt_min:,.0f}-{c.aadt_max:,.0f}",
                "distance_m": round(c.distance_m), "detour_min": round(c.detour_min, 2),
                "decay": round(c.decay, 3), "class_access": c.class_access,
                "effective_exposure": round(c.effective_exposure),
                "potential": round(c.potential), "potential_share": round(c.potential_share, 4),
                "is_site_access_road": c.is_access_road,
                "trip_type": t.trip_type if t else None,
                "site_visits_per_day": round(t.visits, 1) if t else 0.0,
                "site_trip_ends_per_day": round(t.trip_ends, 1) if t else 0.0,
                "share_of_site_trips": round(t.share_of_site, 4) if t else 0.0,
                "capture_of_aadt": round(t.visits / c.aadt, 4) if t else 0.0,
            })
        return pd.DataFrame(rows)

    def to_records(self) -> List[dict]:
        """Output schema from the design doc (Appendix A), extended with trip figures."""
        recs = []
        for t in self.trips.corridors:
            c = t.corridor
            recs.append({
                "corridor_id": c.corridor_id, "rank": c.rank, "name": c.name, "road_reference": c.ref or None,
                "functional_class": c.road_class,
                "aadt": c.aadt, "aadt_source": c.aadt_source, "aadt_year": c.aadt_year,
                "distance_to_site_m": round(c.distance_m, 1),
                "detour_minutes": round(c.detour_min, 2),
                "potential_vehicles_per_day": round(c.potential, 1),
                "modeled_potential_share": round(c.potential_share, 4),
                "site_visits_per_day": round(t.visits, 1),
                "site_trip_ends_per_day": round(t.trip_ends, 1),
                "share_of_site_trips": round(t.share_of_site, 4),
                "trip_type": t.trip_type,
                "approaches": [{"from": a.from_direction, "share": round(a.flow_share, 3),
                                "visits_per_day": round(t.visits * a.flow_share, 1)} for a in c.approaches],
                "hourly_visits": [round(x, 2) for x in t.hourly_visits],
                "is_site_access_road": c.is_access_road,
                "context_only": c.context_only,
                "notes": "Modeled potential from measured AADT; not an observed site count.",
            })
        return recs

    def summary(self) -> dict:
        tr = self.trips
        return {
            "site_id": self.payload.site_id, "site_name": self.payload.site_name,
            "site": [self.payload.site_lat, self.payload.site_lon],
            "segments_fetched": int(len(self.segments)),
            "segments_real_traffic": int((self.segments["drop_reason"] == "").sum()),
            "drop_reasons": self.segments.loc[self.segments["drop_reason"] != "", "drop_reason"]
                                .value_counts().to_dict(),
            "candidate_corridors": len(self.corridors),
            "selected_corridors": [c.label for c in self.selected],
            "selected_potential_coverage": round(sum(c.potential_share for c in self.selected), 4),
            "demand_visits_per_day": round(tr.demand_visits, 1),
            "ite_supply": tr.ite,
            "site_visits_per_day": round(tr.visits, 1),
            "site_trip_ends_per_day": round(tr.trip_ends, 1),
            "binding_constraint": tr.binding_constraint,
            "dayparts": tr.dayparts.round({"visits": 1, "trip_ends": 1}).to_dict(orient="records"),
            "checks": tr.checks,
            "warnings": self.warnings,
        }

    def to_geojson(self) -> dict:
        proj = self.projection
        feats = []
        for t in self.trips.corridors:
            c = t.corridor
            feats.append({"type": "Feature",
                          "properties": {"kind": "corridor", "rank": c.rank, "name": c.label,
                                         "visits_per_day": round(t.visits, 1),
                                         "share": round(t.share_of_site, 4), "trip_type": t.trip_type},
                          "geometry": {"type": "MultiLineString",
                                       "coordinates": [proj.inv(p).round(7).tolist() for p in c.parts_xy]}})
            for a in c.approaches:
                feats.append({"type": "Feature",
                              "properties": {"kind": "approach", "corridor": c.label, "from": a.from_direction,
                                             "visits_per_day": round(t.visits * a.flow_share, 1)},
                              "geometry": {"type": "LineString",
                                           "coordinates": proj.inv(a.path_xy).round(7).tolist()}})
        feats.append({"type": "Feature", "properties": {"kind": "site"},
                      "geometry": {"type": "Point", "coordinates": [self.payload.site_lon, self.payload.site_lat]}})
        return {"type": "FeatureCollection", "features": feats}

    def save(self, out_dir: str) -> Dict[str, str]:
        import os
        os.makedirs(out_dir, exist_ok=True)
        paths = {
            "corridors_csv": os.path.join(out_dir, "corridors.csv"),
            "records_json": os.path.join(out_dir, "selected_corridors.json"),
            "summary_json": os.path.join(out_dir, "summary.json"),
            "geojson": os.path.join(out_dir, "trip_distribution.geojson"),
        }
        self.corridor_table().to_csv(paths["corridors_csv"], index=False)
        with open(paths["records_json"], "w") as f:
            json.dump(self.to_records(), f, indent=2, default=_json_default)
        with open(paths["summary_json"], "w") as f:
            json.dump(self.summary(), f, indent=2, default=_json_default)
        with open(paths["geojson"], "w") as f:
            json.dump(self.to_geojson(), f, default=_json_default)
        return paths


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


class TrafficAnalysis:
    """
    Parameters
    ----------
    segment_source : DB connection / SQLAlchemy engine, a callable(polygon) -> DataFrame,
                     or a DataFrame already holding the intersecting segments.
    schema         : column mapping for the DB table (ignored for callables/DataFrames).
    config         : TripModelConfig (all parameters + their justification).
    use_network    : route each corridor's exit intersection to the site over the fetched road
                     network (default). False = straight-line distance x circuity / connector speed.
    """

    def __init__(self, segment_source: SegmentSource, schema: Optional[SegmentTableSchema] = None,
                 config: Optional[TripModelConfig] = None, use_network: bool = USE_NETWORK,
                 network_snap_m: float = NETWORK_SNAP_M, max_connector_m: float = MAX_CONNECTOR_M):
        self.segment_source = segment_source
        self.schema = schema or SegmentTableSchema()
        self.config = (config or TripModelConfig()).validate()
        self.use_network = use_network
        self.network_snap_m, self.max_connector_m = network_snap_m, max_connector_m

    # Step 1 ------------------------------------------------------------------
    def fetch_segments(self, payload: SitePayload) -> pd.DataFrame:
        src = self.segment_source
        if isinstance(src, pd.DataFrame):
            return src.copy()
        if callable(src):
            return src(payload.drive_time_polygon)
        return fetch_intersecting_segments(src, payload.drive_time_polygon, self.schema)

    # Step 2 ------------------------------------------------------------------
    def prepare_segments(self, raw: pd.DataFrame, payload: SitePayload) -> pd.DataFrame:
        site = (payload.site_lon, payload.site_lat)
        seg = filter_real_traffic_segments(raw, site, self.config)
        keep = seg["drop_reason"] == ""
        seg["corridor_id"] = None
        if keep.any():
            seg.loc[keep, "corridor_id"] = build_corridors(seg[keep], self.config)
        return seg

    # Step 3 ------------------------------------------------------------------
    def score_and_select(self, seg: pd.DataFrame, payload: SitePayload):
        site = (payload.site_lon, payload.site_lat)
        keep = seg["drop_reason"] == ""
        if not keep.any():
            return seg, [], []
        network = RoadNetwork(seg, self.config, self.network_snap_m, self.max_connector_m) if self.use_network else None     # every fetched road, not only counted ones
        scored = score_segments(seg[keep], site, self.config, network)
        seg = seg.join(scored[[c for c in scored.columns if c not in seg.columns]])
        corridors = aggregate_corridors(scored)
        selected = select_corridors(corridors, self.config)
        return seg, corridors, selected

    # Step 4 ------------------------------------------------------------------
    def generate_trips(self, selected: List[Corridor], payload: SitePayload) -> SiteTrips:
        return generate_site_trips(selected, self.config, payload.vfp)

    # All ---------------------------------------------------------------------
    def run(self, payload: SitePayload) -> TrafficAnalysisResult:
        warnings: List[str] = []
        rings = to_polygon_rings(payload.drive_time_polygon)
        if not points_in_polygon(np.array([[payload.site_lon, payload.site_lat]]), rings)[0]:
            warnings.append("Site point lies outside the drive-time polygon.")
        raw = self.fetch_segments(payload)
        seg = self.prepare_segments(raw, payload)
        n_noaadt = int((seg["drop_reason"] == "no measured AADT").sum())
        if n_noaadt:
            warnings.append(f"{n_noaadt} segments have no measured AADT and were excluded (not treated as zero).")
        seg, corridors, selected = self.score_and_select(seg, payload)
        if not selected:
            warnings.append("No corridor with measured traffic reaches the selection threshold.")
        elif len(selected) < self.config.top_k:
            warnings.append(f"Only {len(selected)} corridors pass the "
                            f"{self.config.min_potential_share:.0%} potential threshold; list not padded.")
        est = [c.label for c in selected if c.aadt_estimated]
        if est:
            warnings.append(f"AADT of {', '.join(est)} is below the {self.config.min_aadt:,.0f} minimum (usually a default "
                            "value for uncounted local streets); counted at the minimum because the road touches the site.")
        if not any(c.is_access_road for c in corridors):
            warnings.append(f"No counted road within {self.config.frontage_buffer_m:.0f} m of the site: "
                            "access road unknown, all trips treated as diverted-link.")
        trips = self.generate_trips(selected, payload)
        if payload.vfp is None:
            warnings.append("VFP not supplied - result is market potential, not capped by station size.")
        return TrafficAnalysisResult(payload, self.config, LocalProjection(payload.site_lon, payload.site_lat),
                                     rings, seg, corridors, selected, trips, warnings)
