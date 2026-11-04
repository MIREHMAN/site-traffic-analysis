"""
Step 3 - score every corridor's trip *potential* for the site and pick the top K.

potential_c = AADT_c x direction_factor x class_access_c x exp(-lambda x detour_min_c)

  * AADT              - real measured traffic passing by (exposure)
  * direction_factor  - half the traffic is on the far side and must turn left
                        across the road: 0.5 x (1 + left_turn_factor)
  * class_access      - can that class physically divert (freeway = ramp)
  * detour decay      - drive time from the corridor's exit intersection to the site, out and back

The exit intersection and the route to the site come from the road network (routing.py); without a
connected network the straight-line distance is used. The site's frontage road has zero detour, so its
traffic is pass-by; every other corridor contributes diverted-link trips that decay with detour time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .config import TripModelConfig
from .geo import LocalProjection, bearing_label, nearest_point_on_parts, split_at
from .routing import RoadNetwork

@dataclass
class Approach:
    """One approach arm of a corridor toward the access point (drawn as an arrow)."""
    from_direction: str          # e.g. "north" = traffic coming FROM the north
    path_xy: np.ndarray          # metres, ordered far point -> access point
    length_m: float
    flow_share: float = 0.0      # share of this corridor's site trips


@dataclass
class Corridor:
    corridor_id: str
    name: str
    ref: str
    road_class: str
    aadt: float
    aadt_min: float
    aadt_max: float
    aadt_year: Optional[int]
    aadt_source: Optional[str]
    distance_m: float
    connector_min: float
    detour_min: float
    decay: float
    direction_factor: float
    class_access: float
    effective_exposure: float
    potential: float
    is_access_road: bool
    access_xy: np.ndarray
    rep_segment_id: object
    segment_ids: list
    parts_xy: list
    approaches: List[Approach] = field(default_factory=list)
    potential_share: float = 0.0
    rank: Optional[int] = None
    selected: bool = False
    context_only: bool = False
    route_xy: Optional[np.ndarray] = None      # exit intersection -> site, along the road network
    aadt_estimated: bool = False               # count raised to the minimum on a road touching the site

    @property
    def label(self) -> str:
        if self.name and self.ref and self.ref.upper() not in self.name.upper():
            return f"{self.name} ({self.ref})"
        return self.name or self.ref or str(self.corridor_id)


def score_segments(seg: pd.DataFrame, site_lonlat, cfg: TripModelConfig,
                   network: Optional[RoadNetwork] = None) -> pd.DataFrame:
    """Per-segment exit point, drive time and potential. `seg` = kept rows from filter_real_traffic_segments."""
    site_xy = np.zeros(2)
    dir_factor = 0.5 * (1.0 + cfg.left_turn_factor)
    if network is not None and not network.connect_site(site_xy):
        network = None
    rows = []
    for idx, r in seg.iterrows():
        near_xy, dist, pi, si, t = nearest_point_on_parts(r["parts_xy"], site_xy)
        frontage = dist <= cfg.frontage_buffer_m
        exit_xy, route_xy, conn_min = near_xy, np.vstack([near_xy, site_xy]), \
            dist * cfg.circuity_factor / (cfg.connector_speed_kph * 1000 / 60)       # straight-line fallback
        if not frontage and network is not None:
            rt = network.route_for(idx)
            if rt is not None:
                exit_xy, route_xy, conn_min = rt.exit_xy, rt.path_xy, rt.minutes
                _n, _d, pi, si, t = nearest_point_on_parts(r["parts_xy"], exit_xy)
        detour = 0.0 if frontage else 2.0 * conn_min      # leave, visit, return to route
        decay = math.exp(-cfg.detour_lambda_per_min * detour) if detour <= cfg.max_detour_min else 0.0
        access = cfg.class_access_factor.get(r["road_class_norm"], 1.0)
        eff = float(r["aadt"]) * dir_factor * access
        rows.append(dict(index=idx, access_x=exit_xy[0], access_y=exit_xy[1], distance_m=dist,
                         part_idx=pi, seg_idx=si, seg_t=t, is_frontage=frontage, route_xy=route_xy,
                         connector_min=conn_min, detour_min=detour, decay=decay,
                         direction_factor=dir_factor, class_access=access,
                         effective_exposure=eff, potential=eff * decay))
    sc = pd.DataFrame(rows).set_index("index")
    return seg.join(sc)


def _approaches(row: pd.Series, access_xy: np.ndarray, min_arm_m: float = 50.0) -> List[Approach]:
    part = row["parts_xy"][int(row["part_idx"])]
    before, after = split_at(part, int(row["seg_idx"]), float(row["seg_t"]), access_xy)
    arms = []
    for arm in (before, after[::-1]):       # both ordered far -> access
        seglen = np.linalg.norm(np.diff(arm, axis=0), axis=1)
        L = float(seglen.sum())
        if L < min_arm_m:
            continue
        far = arm[0]
        d = far - access_xy
        arms.append(Approach(bearing_label(d[0], d[1]), arm, L))
    if not arms:  # degenerate: tiny segment - treat as a single approach
        arms.append(Approach("local", np.vstack([part[0], access_xy]), 0.0))
    return arms


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()


def aggregate_corridors(scored: pd.DataFrame) -> List[Corridor]:
    """
    One record per corridor. The corridor's potential is the MAX of its segments
    (the same vehicles flow along the whole road - summing pieces double counts).
    """
    out = []
    for cid, g in scored.groupby("corridor_id", sort=False):
        rep = g.loc[g["potential"].idxmax()]
        access_xy = np.array([rep["access_x"], rep["access_y"]])
        yr = rep.get("aadt_year")
        out.append(Corridor(
            corridor_id=str(cid), name=_s(rep.get("name")), ref=_s(rep.get("ref")),
            road_class=rep["road_class_norm"], aadt=float(rep["aadt"]),
            aadt_min=float(g["aadt"].min()), aadt_max=float(g["aadt"].max()),
            aadt_year=int(yr) if pd.notna(yr) else None,
            aadt_source=rep.get("aadt_source") if pd.notna(rep.get("aadt_source")) else None,
            distance_m=float(g["distance_m"].min()), connector_min=float(rep["connector_min"]),
            detour_min=float(rep["detour_min"]), decay=float(rep["decay"]),
            direction_factor=float(rep["direction_factor"]), class_access=float(rep["class_access"]),
            effective_exposure=float(rep["effective_exposure"]), potential=float(rep["potential"]),
            is_access_road=bool(g["is_frontage"].any()), access_xy=access_xy,
            rep_segment_id=rep["segment_id"], segment_ids=list(g["segment_id"]),
            parts_xy=[p for ps in g["parts_xy"] for p in ps],
            approaches=_approaches(rep, access_xy), route_xy=rep["route_xy"],
            aadt_estimated=bool(rep.get("aadt_estimated", False)),
        ))
    total = sum(c.potential for c in out) or 1.0
    for c in out:
        c.potential_share = c.potential / total
    out.sort(key=lambda c: c.potential, reverse=True)
    return out


def select_corridors(corridors: List[Corridor], cfg: TripModelConfig) -> List[Corridor]:
    """
    Which corridors feed the site:
      1. roads touching the site are its own entrances - always selected (best potential first);
      2. the remaining slots, up to top_k, go to the best-scoring corridors holding at least
         cfg.min_potential_share of the total potential (fewer than K rather than padding).
    Ranks follow potential across the selected set.
    """
    pool = [c for c in corridors if c.potential > 0]                       # already sorted by potential
    access = [c for c in pool if c.is_access_road] if cfg.always_include_access_road else []
    chosen = access[:cfg.top_k]
    for c in pool:
        if len(chosen) >= cfg.top_k:
            break
        if c not in chosen and c.potential_share >= cfg.min_potential_share:
            chosen.append(c)
    chosen.sort(key=lambda c: c.potential, reverse=True)
    for i, c in enumerate(chosen, start=1):
        c.selected, c.rank = True, i
    return chosen
