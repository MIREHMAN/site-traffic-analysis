"""
Step 2 - keep only real traffic segments, then merge them into corridors.
"""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import FHWA_TO_CLASS, TripModelConfig
from .geo import LocalProjection, nearest_point_on_parts, parts_length_m, to_line_parts

_ABBR = {
    "BOULEVARD": "BLVD", "STREET": "ST", "AVENUE": "AVE", "ROAD": "RD", "DRIVE": "DR",
    "PARKWAY": "PKWY", "HIGHWAY": "HWY", "LANE": "LN", "FREEWAY": "FWY", "EXPRESSWAY": "EXPY",
    "SOUTH": "S", "NORTH": "N", "EAST": "E", "WEST": "W", "FARM TO MARKET": "FM",
    "INTERSTATE": "I", "STATE HIGHWAY": "SH", "COUNTY ROAD": "CR",
}


def normalise_road_class(value) -> str:
    """OSM highway tag or FHWA functional-class code -> normalised class string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unclassified"
    if isinstance(value, (int, np.integer)) or (isinstance(value, str) and value.strip().isdigit()):
        return FHWA_TO_CLASS.get(int(value), "unclassified")
    v = str(value).strip().lower().replace(" ", "_")
    aliases = {"interstate": "motorway", "freeway": "motorway", "expressway": "motorway",
               "principal_arterial": "primary", "major_arterial": "primary",
               "minor_arterial": "secondary", "collector": "tertiary", "major_collector": "tertiary",
               "minor_collector": "tertiary", "local": "residential",
               # TxDOT route prefixes (RI-1 roadway inventory)
               "ih": "motorway", "tl": "motorway", "us": "primary", "sh": "primary", "sl": "primary",
               "ua": "primary", "bu": "primary", "bi": "primary", "bs": "primary", "ss": "primary",
               "fm": "secondary", "rm": "secondary", "pa": "secondary", "ps": "secondary",
               "cr": "tertiary", "cs": "residential"}
    return aliases.get(v, v)


def normalise_name(name: Optional[str]) -> str:
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = re.sub(r"^([A-Z]{2}\d{3,4})-[A-Z]G$", r"\1", str(name).strip().upper())   # TxDOT roadbed suffix (-KG/-LG/-RG)
    s = re.sub(r"[.\-,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for long, short in _ABBR.items():
        s = re.sub(rf"\b{long}\b", short, s)
    return s


def filter_real_traffic_segments(df: pd.DataFrame, site_lonlat, cfg: TripModelConfig) -> pd.DataFrame:
    """
    Keep segments that carry real, countable vehicle traffic:
      * measured AADT present and >= cfg.min_aadt (missing AADT is NOT zero - it is excluded
        and reported, never ranked as low traffic). Exception: a road within cfg.frontage_buffer_m of
        the site is the site's own entrance, so a count below the floor (typically a default value for
        uncounted local streets) is raised to cfg.min_aadt and flagged `aadt_estimated`.
      * a drivable, non-ramp, non-service road class
      * a usable geometry longer than cfg.min_segment_length_m after clipping
    Adds: road_class_norm, parts_ll, parts_xy, length_m, aadt_estimated, drop_reason.
    """
    proj = LocalProjection(*site_lonlat)
    out = df.copy()
    out["road_class_norm"] = out["road_class"].map(normalise_road_class)
    out["parts_ll"] = out["geometry"].map(to_line_parts)
    out["parts_xy"] = out["parts_ll"].map(lambda ps: [proj.fwd(p) for p in ps if len(p) >= 2])
    out["length_m"] = out["parts_xy"].map(parts_length_m)
    out["aadt"] = pd.to_numeric(out["aadt"], errors="coerce")

    out["aadt_estimated"] = False
    low = out["aadt"].notna() & (out["aadt"] < cfg.min_aadt) & ~out["road_class_norm"].isin(cfg.excluded_classes)
    for idx in out.index[low]:
        parts = out.at[idx, "parts_xy"]
        if parts and nearest_point_on_parts(parts, np.zeros(2))[1] <= cfg.frontage_buffer_m:
            out.at[idx, "aadt"] = cfg.min_aadt
            out.at[idx, "aadt_estimated"] = True

    reason = pd.Series("", index=out.index, dtype=object)
    # later rules overwrite earlier ones -> most fundamental reason wins
    reason[(out["aadt"].notna()) & (out["aadt"] < cfg.min_aadt)] = f"AADT < {cfg.min_aadt:.0f}"
    reason[out["aadt"].isna()] = "no measured AADT"
    reason[out["road_class_norm"].isin(cfg.excluded_classes)] = "non-traffic class (ramp/service/etc.)"
    reason[out["length_m"] < cfg.min_segment_length_m] = "short/empty geometry"
    out["drop_reason"] = reason
    return out


def build_corridors(seg: pd.DataFrame, cfg: TripModelConfig) -> pd.Series:
    """
    Assign a stable corridor_id to each kept segment.

    Key = route ref if present, else normalised name, else the segment id.
    Same-key pieces that are not within cfg.corridor_gap_m of each other become
    separate corridors (two different "Main St" in different towns never merge).
    """
    keys = []
    for _, r in seg.iterrows():
        ref = normalise_name(r.get("ref"))
        nm = normalise_name(r.get("name"))
        keys.append(ref or nm or f"SEG:{r['segment_id']}")
    seg = seg.assign(_key=keys)

    corridor_ids = pd.Series(index=seg.index, dtype=object)
    for key, grp in seg.groupby("_key", sort=False):
        idx = list(grp.index)
        parent = {i: i for i in idx}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        pts = {i: np.vstack(grp.at[i, "parts_xy"]) for i in idx}
        for a_i, a in enumerate(idx):
            for b in idx[a_i + 1:]:
                if find(a) == find(b):
                    continue
                pa, pb = pts[a], pts[b]
                # cheap bbox gate, then vertex-to-vertex distance
                if (pa[:, 0].min() - pb[:, 0].max() > cfg.corridor_gap_m or
                        pb[:, 0].min() - pa[:, 0].max() > cfg.corridor_gap_m or
                        pa[:, 1].min() - pb[:, 1].max() > cfg.corridor_gap_m or
                        pb[:, 1].min() - pa[:, 1].max() > cfg.corridor_gap_m):
                    continue
                d = np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2))
                if d <= cfg.corridor_gap_m:
                    parent[find(a)] = find(b)
        roots: List = sorted({find(i) for i in idx}, key=lambda r: str(r))
        for i in idx:
            n = roots.index(find(i))
            corridor_ids[i] = key if len(roots) == 1 else f"{key}#{n + 1}"
    return corridor_ids
