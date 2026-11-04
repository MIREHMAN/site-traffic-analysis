"""
Step 4 - site trip generation for the SELECTED corridors only.

Two independent estimates, reconciled:

  Demand side (market):   V_dem = sum_c  min(capture_rate x potential_c,
                                              max_capture_of_aadt x AADT_c)
  Supply side (ITE 945):  V_ite = ITE daily trip ends per VFP x VFP / 2   (if VFP known)

  Site visits/day  V = min(V_dem, V_ite)   - a station cannot serve more than
  its pumps support (ITE), nor more than the passing market supplies (demand).

V is then distributed to corridors in proportion to their capped demand
(Schultz & Adamson "Application 2": only movements that can actually enter or
exit the site), split to approach arms, and spread over the day using the NACS
fill-up time-of-day survey.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import TripModelConfig
from .potential import Corridor


def hourly_profile(cfg: TripModelConfig) -> np.ndarray:
    """24 hourly shares of daily visits (sum = 1). NACS dayparts + within-bucket shape."""
    prof = np.zeros(24)
    scale = 1.0 - cfg.overnight_share
    for start, end, share, name in cfg.dayparts:
        shape = np.asarray(cfg.within_daypart_shape.get(name, [1.0] * (end - start)), dtype=float)
        shape = shape / shape.sum()
        prof[start:end] = share * scale * shape
    covered = {h for s, e, _, _ in cfg.dayparts for h in range(s, e)}
    night = [h for h in range(24) if h not in covered]
    if night:
        prof[night] += cfg.overnight_share / len(night)
    return prof / prof.sum()


def ite_supply(vfp: Optional[int], cfg: TripModelConfig) -> Optional[Dict[str, float]]:
    if not vfp:
        return None
    daily = cfg.ite_daily_trip_ends_per_vfp * vfp
    return {
        "vfp": vfp,
        "daily_trip_ends": daily,
        "daily_visits": daily * cfg.entering_share,
        "am_peak_trip_ends": cfg.ite_am_peak_trip_ends_per_vfp * vfp,
        "pm_peak_trip_ends": cfg.ite_pm_peak_trip_ends_per_vfp * vfp,
        "am_peak_share_of_daily": cfg.ite_am_peak_trip_ends_per_vfp / cfg.ite_daily_trip_ends_per_vfp,
        "pm_peak_share_of_daily": cfg.ite_pm_peak_trip_ends_per_vfp / cfg.ite_daily_trip_ends_per_vfp,
        "passby_am": cfg.ite_passby_am, "passby_pm": cfg.ite_passby_pm,
    }


@dataclass
class CorridorTrips:
    corridor: Corridor
    demand_visits: float          # uncapped by ITE
    visits: float                 # final entering vehicles/day
    share_of_site: float
    trip_type: str                # pass-by | diverted-link
    hourly_visits: np.ndarray
    approach_visits: Dict[str, float]
    capped_by_aadt: bool

    @property
    def trip_ends(self) -> float:
        return 2.0 * self.visits


@dataclass
class SiteTrips:
    corridors: List[CorridorTrips]
    demand_visits: float
    ite: Optional[Dict[str, float]]
    visits: float
    binding_constraint: str
    hourly_visits: np.ndarray
    profile: np.ndarray
    dayparts: pd.DataFrame
    checks: Dict[str, object] = field(default_factory=dict)

    @property
    def trip_ends(self) -> float:
        return 2.0 * self.visits


def generate_site_trips(selected: List[Corridor], cfg: TripModelConfig, vfp: Optional[int] = None) -> SiteTrips:
    prof = hourly_profile(cfg)
    dem, capped = [], []
    for c in selected:
        raw = cfg.capture_rate * c.potential
        cap = cfg.max_capture_of_aadt * c.aadt
        dem.append(min(raw, cap))
        capped.append(raw > cap)
    is_pb = [c.is_access_road for c in selected]
    pb_dem = float(sum(d for d, f in zip(dem, is_pb) if f))
    dv_dem = float(sum(d for d, f in zip(dem, is_pb) if not f))
    p = cfg.passby_share
    # Diverted pool sized from the pass-by market via the ITE / agency pass-by share.
    dv = min(dv_dem, pb_dem * (1 - p) / p) if pb_dem > 0 else dv_dem
    v_dem = pb_dem + dv
    ite = ite_supply(vfp, cfg)

    if ite is None:
        v, binding = v_dem, "demand (no VFP supplied - market potential only)"
    elif ite["daily_visits"] < v_dem:
        v, binding = ite["daily_visits"], f"supply: ITE LU 945 x {vfp} VFP"
    else:
        v, binding = v_dem, "demand: passing traffic on selected corridors"
    scale = v / v_dem if v_dem > 0 else 0.0
    dv_scale = dv / dv_dem if dv_dem > 0 else 0.0

    rows: List[CorridorTrips] = []
    for c, d, cp, f in zip(selected, dem, capped, is_pb):
        visits = d * scale * (1.0 if f else dv_scale)
        arms = [a for a in c.approaches if a.length_m > 0] or c.approaches
        for a in c.approaches:
            a.flow_share = 0.0
        for a in arms:
            a.flow_share = 1.0 / len(arms)   # AADT is two-way: equal from each live arm
        rows.append(CorridorTrips(
            corridor=c, demand_visits=d, visits=visits,
            share_of_site=visits / v if v else 0.0,
            trip_type="pass-by" if c.is_access_road else "diverted-link",
            hourly_visits=visits * prof,
            approach_visits={a.from_direction: visits * a.flow_share for a in c.approaches},
            capped_by_aadt=cp,
        ))

    hourly = np.sum([r.hourly_visits for r in rows], axis=0) if rows else np.zeros(24)
    dp = []
    for s, e, share, name in cfg.dayparts:
        dp.append({"daypart": name, "profile_share": share, "visits": float(hourly[s:e].sum()),
                   "trip_ends": float(2 * hourly[s:e].sum())})
    covered = {h for s, e, _, _ in cfg.dayparts for h in range(s, e)}
    night = [h for h in range(24) if h not in covered]
    dp.append({"daypart": "Overnight 0-6", "profile_share": 0.0, "visits": float(hourly[night].sum()),
               "trip_ends": float(2 * hourly[night].sum())})

    passby = sum(r.visits for r in rows if r.trip_type == "pass-by")
    checks = {
        "passby_demand_visits": pb_dem,
        "diverted_demand_visits_uncapped": dv_dem,
        "model_passby_share": passby / v if v else 0.0,
        "target_passby_share": p,
        "ite_passby_reference": (cfg.ite_passby_am + cfg.ite_passby_pm) / 2,
        "wisdot_passby_range": (0.0, 0.60),
        "model_am_peak_hour_share": float(prof[6:10].max()),
        "model_pm_peak_hour_share": float(prof[15:19].max()),
        "ite_am_peak_hour_share": cfg.ite_am_peak_trip_ends_per_vfp / cfg.ite_daily_trip_ends_per_vfp,
        "ite_pm_peak_hour_share": cfg.ite_pm_peak_trip_ends_per_vfp / cfg.ite_daily_trip_ends_per_vfp,
        "corridors_capped_by_aadt": [r.corridor.label for r in rows if r.capped_by_aadt],
    }
    return SiteTrips(rows, v_dem, ite, v, binding, hourly, prof, pd.DataFrame(dp), checks)
