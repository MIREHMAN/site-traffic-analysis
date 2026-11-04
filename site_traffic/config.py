"""
All thresholds and constants of the algorithm. Edit the values in the first block (plain Python), save,
and restart the service (`uvicorn ... --reload` does that for you; `python run.py` reads it on every run).

How the pieces fit:
  1. SEGMENT FILTER   keep only real traffic segments
  2. CORRIDORS        merge segments into corridors (roads)
  3. ROAD NETWORK     find the intersection where a driver leaves each corridor, and the route to the site
  4. ACCESSIBILITY    drive time -> detour retention; how much of the traffic can reach the site
  5. SELECTION        which corridors are selected
  6. DEMAND / ITE / TIME OF DAY   convert potential into site visits per day and per hour

  potential = AADT x direction factor x road-class access x exp(-DETOUR_LAMBDA_PER_MIN x detour minutes)
  site visits (per corridor) = CAPTURE_RATE x potential, capped at MAX_CAPTURE_OF_AADT x AADT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# =========================================================================================================
#  EDIT HERE
# =========================================================================================================

# ---- 1. SEGMENT FILTER ------------------------------------------------------------------------------------
MIN_AADT = 500                 # vehicles/day. Below this a road is not a fuel market. A road touching the site
                               #   (FRONTAGE_BUFFER_M) is raised to this value instead of being dropped.
MIN_SEGMENT_LENGTH_M = 25      # metres. Shorter pieces (slivers made by clipping to the polygon) are dropped.
EXCLUDED_CLASSES = (           # road classes that never carry site trips (their AADT is already on the parent road)
    "service", "track", "path", "footway", "cycleway", "pedestrian", "steps",
    "parking_aisle", "driveway", "private", "construction", "proposed",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
)
FHWA_TO_CLASS = {              # FHWA functional-class codes -> road classes used below
    1: "motorway", 2: "motorway", 3: "primary", 4: "secondary", 5: "tertiary", 6: "tertiary", 7: "residential",
}

# ---- 2. CORRIDORS -----------------------------------------------------------------------------------------
CORRIDOR_GAP_M = 250           # metres. Same-name pieces further apart than this are separate corridors.

# ---- 3. ROAD NETWORK (exit intersection + route to the site) ----------------------------------------------
USE_NETWORK = True             # False = straight-line distance x CIRCUITY_FACTOR / CONNECTOR_SPEED_KPH
NETWORK_SNAP_M = 15            # metres. Road ends / T-junctions closer than this count as one intersection.
                               #   Raise it if a road that should connect shows no route; lower it if
                               #   unrelated roads (e.g. an overpass) join.
MAX_CONNECTOR_M = 400          # metres. The site must be this close to a road to be routed over the network.

# ---- 4. ACCESSIBILITY: drive time and detour --------------------------------------------------------------
FRONTAGE_BUFFER_M = 75         # metres. A road this close to the site is its frontage: zero detour, pass-by trips.
CIRCUITY_FACTOR = 1.30         # network / straight-line distance (US urban 1.2-1.4); straight-line fallback only.
CONNECTOR_SPEED_KPH = 40       # km/h on the link between a road and the site (and in the straight-line fallback).
DETOUR_LAMBDA_PER_MIN = 0.09   # per minute. Retention = exp(-lambda x detour). 0.09 keeps ~64 % at 5 min, ~41 % at 10.
MAX_DETOUR_MIN = 15            # minutes. Roads needing a longer detour than this contribute nothing.
LEFT_TURN_FACTOR = 0.60        # share (0-1) of opposite-direction traffic willing to cross the road.
                               #   direction factor = 0.5 x (1 + LEFT_TURN_FACTOR)

# Probability that a road class can practically divert into the site (1 = fully, lower = harder).
CLASS_ACCESS_FACTOR = {
    "motorway": 0.35,          # no direct access; drivers must exit
    "trunk": 0.85,
    "primary": 1.0,
    "secondary": 1.0,
    "tertiary": 1.0,
    "unclassified": 1.0,
    "residential": 1.0,
}
# Speed (km/h) used for drive times when a segment has no speed of its own.
CLASS_DEFAULT_SPEED_KPH = {
    "motorway": 100, "trunk": 80, "primary": 65, "secondary": 55, "tertiary": 50, "unclassified": 40, "residential": 40,
}

# ---- 5. SELECTION -----------------------------------------------------------------------------------------
TOP_K = 5                      # most corridors selected / shown.
MIN_POTENTIAL_SHARE = 0.02     # a corridor needs at least this share (0-1) of the total potential to be selected.
ALWAYS_INCLUDE_ACCESS_ROAD = True   # roads touching the site are always selected (they fill slots first).

# ---- 6a. DEMAND: potential -> visits ----------------------------------------------------------------------
CAPTURE_RATE = 0.08            # share of effective passing vehicles that stop.
MAX_CAPTURE_OF_AADT = 0.10     # visits drawn from a road never exceed this share of its AADT.
PASSBY_SHARE = 0.60            # pass-by share of site trips; sizes the diverted pool against the frontage market:
                               #   diverted = min(diverted demand, pass-by x (1 - p) / p)

# ---- 6b. SUPPLY CAP (only when vfp is sent) ---------------------------------------------------------------
ITE_DAILY_TRIP_ENDS_PER_VFP = 345.75   # ITE LU 945 weekday trip ends per fueling position
ITE_AM_PEAK_TRIP_ENDS_PER_VFP = 31.60
ITE_PM_PEAK_TRIP_ENDS_PER_VFP = 26.90
ITE_PASSBY_AM = 0.76                   # ITE pass-by share, AM / PM peak
ITE_PASSBY_PM = 0.75
ENTERING_SHARE = 0.50                  # share of trip ends that are entering (50:50)

# ---- 6c. HOUR OF DAY --------------------------------------------------------------------------------------
OVERNIGHT_SHARE = 0.02         # share of daily visits outside the dayparts below (taken pro rata from them)
DAYPARTS = (                   # (start hour, end hour exclusive, share of daily visits, name); shares sum to 1
    (6, 10, 0.24, "Morning 6-10"),
    (10, 15, 0.36, "Midday 10-15"),
    (15, 19, 0.34, "Afternoon 15-19"),
    (19, 24, 0.06, "Evening 19-24"),
)
WITHIN_DAYPART_SHAPE = {       # hourly shape inside each daypart: one value per hour, summing to 1
    "Morning 6-10": (0.20, 0.28, 0.28, 0.24),
    "Midday 10-15": (0.18, 0.20, 0.22, 0.20, 0.20),
    "Afternoon 15-19": (0.24, 0.27, 0.27, 0.22),
    "Evening 19-24": (0.32, 0.26, 0.20, 0.13, 0.09),
}

# ---- DRIVE-TIME AREA FALLBACK -----------------------------------------------------------------------------
DRIVE_TIME_AVG_SPEED_KPH = 35  # km/h. Used only when no drive_time_polygon is sent and Mapbox is unavailable.

# ---- PDF REPORT ONLY (the model is not affected) ----------------------------------------------------------
REPORT_GROWTH_RATE = 0.025     # projected AADT on the two top-corridor cards: per year ...
REPORT_GROWTH_YEARS = 5        # ... over this many years
CONFIDENCE_HIGH_MAX_AGE_YEARS = 3      # "High" confidence: counted within this many years ...
CONFIDENCE_HIGH_MAX_SPREAD = 1.5       # ... and max/min AADT along the corridor no more than this
CONFIDENCE_MEDIUM_MAX_AGE_YEARS = 6    # "Medium": within this many years ...
CONFIDENCE_MEDIUM_MAX_SPREAD = 2.5     # ... and no more than this spread; anything else is "Low"
PASS_BY_SCENARIOS = (          # rows of the pass-by sensitivity table: (pass-by share, label)
    (0.40, "Low pass-by"),
    (0.60, "Reviewer-accepted maximum (WisDOT)"),
    (0.75, "ITE LU 945 data average"),
)

# =========================================================================================================
#  END OF EDITABLE VALUES
# =========================================================================================================


@dataclass
class TripModelConfig:
    """The values above, as one object handed to the model."""
    # segment filter
    min_aadt: float = MIN_AADT
    min_segment_length_m: float = MIN_SEGMENT_LENGTH_M
    excluded_classes: Sequence[str] = EXCLUDED_CLASSES
    # corridors
    corridor_gap_m: float = CORRIDOR_GAP_M
    # accessibility
    frontage_buffer_m: float = FRONTAGE_BUFFER_M
    circuity_factor: float = CIRCUITY_FACTOR
    connector_speed_kph: float = CONNECTOR_SPEED_KPH
    detour_lambda_per_min: float = DETOUR_LAMBDA_PER_MIN
    max_detour_min: float = MAX_DETOUR_MIN
    left_turn_factor: float = LEFT_TURN_FACTOR
    class_access_factor: Dict[str, float] = field(default_factory=lambda: dict(CLASS_ACCESS_FACTOR))
    class_default_speed_kph: Dict[str, float] = field(default_factory=lambda: dict(CLASS_DEFAULT_SPEED_KPH))
    # selection
    top_k: int = TOP_K
    min_potential_share: float = MIN_POTENTIAL_SHARE
    always_include_access_road: bool = ALWAYS_INCLUDE_ACCESS_ROAD
    # demand
    capture_rate: float = CAPTURE_RATE
    max_capture_of_aadt: float = MAX_CAPTURE_OF_AADT
    passby_share: float = PASSBY_SHARE
    # supply (ITE LU 945)
    ite_daily_trip_ends_per_vfp: float = ITE_DAILY_TRIP_ENDS_PER_VFP
    ite_am_peak_trip_ends_per_vfp: float = ITE_AM_PEAK_TRIP_ENDS_PER_VFP
    ite_pm_peak_trip_ends_per_vfp: float = ITE_PM_PEAK_TRIP_ENDS_PER_VFP
    ite_passby_am: float = ITE_PASSBY_AM
    ite_passby_pm: float = ITE_PASSBY_PM
    entering_share: float = ENTERING_SHARE
    # time of day
    overnight_share: float = OVERNIGHT_SHARE
    dayparts: Tuple[Tuple[int, int, float, str], ...] = DAYPARTS
    within_daypart_shape: Dict[str, Tuple[float, ...]] = field(default_factory=lambda: dict(WITHIN_DAYPART_SHAPE))

    def validate(self) -> "TripModelConfig":
        """Reject nonsense values with a message that names the constant to fix."""
        checks = [
            (self.min_aadt >= 0 and self.min_segment_length_m >= 0, "MIN_AADT and MIN_SEGMENT_LENGTH_M must be >= 0"),
            (0 < self.capture_rate <= self.max_capture_of_aadt <= 1, "need 0 < CAPTURE_RATE <= MAX_CAPTURE_OF_AADT <= 1"),
            (0 <= self.left_turn_factor <= 1, "LEFT_TURN_FACTOR must be between 0 and 1"),
            (0 < self.passby_share < 1, "PASSBY_SHARE must be between 0 and 1 (exclusive)"),
            (self.detour_lambda_per_min >= 0 and self.max_detour_min > 0, "DETOUR_LAMBDA_PER_MIN >= 0 and MAX_DETOUR_MIN > 0"),
            (self.top_k >= 1, "TOP_K must be >= 1"),
            (0 <= self.min_potential_share < 1, "MIN_POTENTIAL_SHARE must be in [0, 1)"),
            (abs(sum(d[2] for d in self.dayparts) - 1.0) < 1e-6, "DAYPARTS shares must sum to 1"),
            (0 <= self.overnight_share < 1, "OVERNIGHT_SHARE must be in [0, 1)"),
            (self.frontage_buffer_m >= 0 and self.connector_speed_kph > 0 and self.circuity_factor >= 1,
             "FRONTAGE_BUFFER_M >= 0, CONNECTOR_SPEED_KPH > 0, CIRCUITY_FACTOR >= 1"),
            (all(name in self.within_daypart_shape and len(self.within_daypart_shape[name]) == e - s
                 and abs(sum(self.within_daypart_shape[name]) - 1.0) < 1e-6 for s, e, _sh, name in self.dayparts),
             "WITHIN_DAYPART_SHAPE needs, for every daypart, one value per hour summing to 1"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(f"config.py: {msg}")
        return self


@dataclass
class NetworkSettings:
    """Road-network routing (routing.py)."""
    use_network: bool = USE_NETWORK
    snap_m: float = NETWORK_SNAP_M
    max_connector_m: float = MAX_CONNECTOR_M


@dataclass
class ReportSettings:
    """Constants of the PDF report (derived figures only)."""
    growth_rate: float = REPORT_GROWTH_RATE
    growth_years: int = REPORT_GROWTH_YEARS
    pass_by_scenarios: List[Tuple[float, str]] = field(default_factory=lambda: list(PASS_BY_SCENARIOS))
    confidence_high_max_age_years: int = CONFIDENCE_HIGH_MAX_AGE_YEARS
    confidence_high_max_spread: float = CONFIDENCE_HIGH_MAX_SPREAD
    confidence_medium_max_age_years: int = CONFIDENCE_MEDIUM_MAX_AGE_YEARS
    confidence_medium_max_spread: float = CONFIDENCE_MEDIUM_MAX_SPREAD


@dataclass
class SitePayload:
    """Input from the host system."""
    site_lat: float
    site_lon: float
    drive_time_polygon: object            # GeoJSON dict | shapely Polygon | [(lon,lat),...]
    drive_time_minutes: Optional[float] = None
    vfp: Optional[int] = None             # vehicle fueling positions, if known
    site_id: Optional[str] = None
    site_name: Optional[str] = None
