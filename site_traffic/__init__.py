"""Gas-station site-trip potential layer: real-traffic corridors -> top-K -> trip generation."""
from .analysis import TrafficAnalysis, TrafficAnalysisResult
from .config import SitePayload, TripModelConfig
from .db import SegmentTableSchema, build_intersect_sql, fetch_intersecting_segments
from .tripgen import hourly_profile, ite_supply

__all__ = ["TrafficAnalysis", "TrafficAnalysisResult", "SitePayload", "TripModelConfig",
           "SegmentTableSchema", "fetch_intersecting_segments", "build_intersect_sql",
           "hourly_profile", "ite_supply"]
__version__ = "0.1.0"
