import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from site_traffic import (SegmentTableSchema, SitePayload, TrafficAnalysis, TripModelConfig,  # noqa: E402
                          build_intersect_sql, fetch_intersecting_segments, hourly_profile, ite_supply)
from site_traffic.geo import LocalProjection  # noqa: E402

LAT, LON = 29.5, -95.06
PROJ = LocalProjection(LON, LAT)


def line(*pts):
    return {"type": "LineString", "coordinates": PROJ.inv(np.asarray(pts, float)).tolist()}


def square(r=6000):
    ring = PROJ.inv(np.array([(-r, -r), (r, -r), (r, r), (-r, r), (-r, -r)], float))
    return {"type": "Polygon", "coordinates": [ring.tolist()]}


def seg(sid, name, cls, aadt, *pts, ref=None):
    return dict(segment_id=sid, name=name, ref=ref, road_class=cls, aadt=aadt,
                aadt_year=2024, aadt_source="test", geometry=line(*pts))


def run(rows, vfp=None, cfg=None, **kw):
    a = TrafficAnalysis(pd.DataFrame(rows), config=cfg, **kw)
    return a.run(SitePayload(LAT, LON, square(), drive_time_minutes=10, vfp=vfp))


# ---------------------------------------------------------------- references
def test_ite_supply_reproduces_anchor_bay_tia():
    s = ite_supply(10, TripModelConfig())          # TIA Table 7: 10 VFP -> 3,458 / 316 / 269
    assert round(s["daily_trip_ends"]) == 3458
    assert round(s["am_peak_trip_ends"]) == 316
    assert round(s["pm_peak_trip_ends"]) == 269


def test_hourly_profile_preserves_nacs_dayparts():
    cfg = TripModelConfig()
    p = hourly_profile(cfg)
    assert p.sum() == pytest.approx(1.0)
    assert p[6:10].sum() == pytest.approx(0.24 * (1 - cfg.overnight_share))
    assert p[10:15].sum() == pytest.approx(0.36 * (1 - cfg.overnight_share))
    assert p[0:6].sum() == pytest.approx(cfg.overnight_share)


def test_detour_lambda_matches_nacs_survey_points():
    lam = TripModelConfig().detour_lambda_per_min
    assert math.exp(-lam * 5) == pytest.approx(0.66, abs=0.03)   # NACS 2014, 5 min
    assert math.exp(-lam * 10) == pytest.approx(0.39, abs=0.03)  # NACS 2014, 10 min


# ---------------------------------------------------------------- filtering
def test_only_real_traffic_segments_are_kept():
    rows = [
        seg("a", "Main St", "primary", 20000, (40, -3000), (40, 3000)),
        seg("b", "No Count Ln", "residential", None, (100, 0), (900, 0)),
        seg("c", "Low Ct", "residential", 200, (100, 50), (900, 50)),
        seg("d", "Ramp", "motorway_link", 9000, (500, 500), (900, 900)),
        seg("e", "Lot", "service", 800, (60, 10), (200, 10)),
    ]
    r = run(rows)
    reasons = r.segments.set_index("segment_id")["drop_reason"]
    assert reasons["a"] == ""
    assert reasons["b"] == "no measured AADT"            # missing is NOT zero
    assert reasons["c"].startswith("AADT <")
    assert "ramp" in reasons["d"] and "ramp" in reasons["e"]
    assert [c.corridor_id for c in r.corridors] == ["MAIN ST"]
    assert any("no measured AADT" in w for w in r.warnings)


def test_fhwa_codes_are_understood():
    r = run([seg("a", "Gulf Fwy", 1, 150000, (-2000, -3000), (-2000, 3000), ref="I-45"),
             seg("b", "Main", 3, 15000, (40, -3000), (40, 3000))])
    classes = {c.label: c.road_class for c in r.corridors}
    assert classes["Gulf Fwy (I-45)"] == "motorway"


# ---------------------------------------------------------------- corridors
def test_fragments_merge_into_one_corridor_and_take_one_slot():
    rows = [seg(f"m{i}", "Egret Bay Blvd", "primary", 15000 + i, (40, y), (40, y + 1000))
            for i, y in enumerate(range(-4000, 4000, 1000))]
    rows += [seg("x", "Other Rd", "secondary", 9000, (-1500, -3000), (-1500, 3000))]
    r = run(rows)
    ids = [c.corridor_id for c in r.selected]
    assert ids.count("EGRET BAY BLVD") == 1
    assert len(r.corridors) == 2


def test_same_name_far_apart_stays_separate():
    rows = [seg("a", "Main St", "primary", 15000, (40, -500), (40, 500)),
            seg("b", "Main St", "primary", 15000, (4000, -500), (4000, 500))]
    r = run(rows)
    assert len(r.corridors) == 2


def test_name_normalisation_merges_spelling_variants():
    rows = [seg("a", "South Egret Bay Boulevard", "primary", 15000, (40, -1000), (40, 0)),
            seg("b", "S. Egret Bay Blvd", "primary", 16000, (40, 0), (40, 1000))]
    assert len(run(rows).corridors) == 1


# ---------------------------------------------------------------- selection
def test_busy_road_further_away_beats_quiet_road_next_door():
    rows = [seg("loc", "Cul De Sac", "residential", 800, (60, -200), (60, 200)),
            seg("art", "Big Arterial", "primary", 30000, (-1500, -4000), (-1500, 4000))]
    r = run(rows)
    assert r.corridors[0].label == "Big Arterial"
    # ... and the cul-de-sac is still shown as the access road
    assert any(c.is_access_road and c.label == "Cul De Sac" for c in r.selected)


def test_detour_beyond_limit_contributes_nothing():
    rows = [seg("a", "Front Rd", "primary", 15000, (40, -3000), (40, 3000)),
            seg("f", "Far Fwy", "motorway", 200000, (-5800, -5000), (-5800, 5000))]
    r = run(rows)
    far = [c for c in r.corridors if c.label == "Far Fwy"][0]
    assert far.detour_min > r.config.max_detour_min and far.potential == 0
    assert not far.selected


def test_threshold_returns_fewer_than_k_rather_than_padding():
    rows = [seg("a", "Front Rd", "primary", 20000, (40, -3000), (40, 3000)),
            seg("b", "Tiny Rd", "tertiary", 600, (-3000, -3000), (-3000, 3000))]
    r = run(rows)
    ranked = r.selected
    assert len(ranked) == 1
    assert any("not padded" in w for w in r.warnings)


def test_top_k_respected():
    rows = [seg(f"r{i}", f"Road {i}", "secondary", 10000, (-500 * i - 300, -3000), (-500 * i - 300, 3000))
            for i in range(9)]
    r = run(rows, cfg=TripModelConfig(top_k=3, min_potential_share=0.0))
    assert len(r.selected) == 3


# ---------------------------------------------------------------- trip generation
def _two_road_rows():
    return [seg("a", "Front Rd", "primary", 16000, (40, -3000), (40, 3000)),
            seg("b", "Cross Rd", "primary", 25000, (-3000, 1500), (3000, 1500)),
            seg("c", "Far Rd", "secondary", 18000, (-3000, -2000), (3000, -2000))]


def test_passby_share_matches_target_when_diverted_market_is_ample():
    r = run(_two_road_rows())
    assert r.trips.checks["model_passby_share"] == pytest.approx(0.60, abs=1e-6)


def test_small_station_is_capped_by_ite_supply():
    r = run(_two_road_rows(), vfp=2)
    assert r.trips.binding_constraint.startswith("supply")
    assert r.trips.visits == pytest.approx(ite_supply(2, TripModelConfig())["daily_visits"])


def test_large_station_is_capped_by_market():
    r = run(_two_road_rows(), vfp=40)
    assert r.trips.binding_constraint.startswith("demand")
    assert r.trips.visits < ite_supply(40, TripModelConfig())["daily_visits"]


def test_visits_never_exceed_capture_cap_of_aadt():
    cfg = TripModelConfig(capture_rate=0.10, max_capture_of_aadt=0.10)
    r = run(_two_road_rows(), cfg=cfg)
    for t in r.trips.corridors:
        assert t.visits <= cfg.max_capture_of_aadt * t.corridor.aadt + 1e-6


def test_trip_bookkeeping_is_consistent():
    r = run(_two_road_rows(), vfp=12)
    tr = r.trips
    assert sum(t.visits for t in tr.corridors) == pytest.approx(tr.visits)
    assert tr.hourly_visits.sum() == pytest.approx(tr.visits)
    assert tr.trip_ends == pytest.approx(2 * tr.visits)
    for t in tr.corridors:
        assert sum(t.approach_visits.values()) == pytest.approx(t.visits)


def test_exit_point_and_route_follow_the_network():
    """A side street joins the cross road: the exit is at that junction and the route reaches the site."""
    rows = [seg("fr", "Frontage Rd", "secondary", 9000, (-600, 0), (600, 0)),
            seg("cr", "Cross Rd", "primary", 20000, (1500, -1200), (1500, 1200)),
            seg("link", "Link St", "residential", 300, (1500, 0), (600, 0))]       # joins Cross Rd to Frontage Rd
    r = run(rows)
    cross = [c for c in r.corridors if c.label == "Cross Rd"][0]
    assert np.allclose(cross.access_xy, (1500, 0), atol=10)                       # exit = the junction, not the nearest point
    assert cross.route_xy is not None and np.allclose(cross.route_xy[-1], (0, 0), atol=1)
    assert cross.detour_min == pytest.approx(2 * cross.connector_min)


def test_without_network_falls_back_to_straight_line():
    r = run(_two_road_rows(), use_network=False)
    cross = [c for c in r.corridors if c.label == "Cross Rd"][0]
    assert cross.detour_min > 0 and len(cross.route_xy) == 2


def test_empty_area_does_not_crash():
    r = run([seg("b", "No Count Ln", "residential", None, (100, 0), (900, 0))])
    assert r.selected == [] and r.trips.visits == 0
    assert r.to_records() == []


# ---------------------------------------------------------------- DB layer
def test_sql_uses_index_friendly_intersects_and_transforms_srid():
    sql = build_intersect_sql(SegmentTableSchema(table="t.roads", srid=3857))
    assert "ST_Intersects" in sql and "&&" in sql
    assert "ST_Transform(ST_GeomFromText(%(poly_wkt)s, 4326), 3857)" in sql
    assert "ST_AsGeoJSON(ST_Transform(ST_Intersection" in sql


def test_fetch_passes_polygon_wkt(monkeypatch):
    captured = {}

    def fake_read_sql(sql, conn, params=None):
        captured["sql"], captured["params"] = sql, params
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    fetch_intersecting_segments(object(), square(1000))
    assert captured["params"]["poly_wkt"].startswith("POLYGON((")


# ---------------------------------------------------------------- end to end
@pytest.fixture
def demo_service(monkeypatch, tmp_path):
    from site_traffic import service
    monkeypatch.setenv("TRAFFIC_DEMO", "1")
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    monkeypatch.setattr(service, "_state", {})
    monkeypatch.setattr(service, "load_dotenv", lambda *a, **k: None, raising=False)
    return service


def test_service_json_only(demo_service):
    from site_traffic.demo_data import SITE_LAT, SITE_LON
    body = demo_service.run_site({"lat": SITE_LAT, "lng": SITE_LON, "drive_time_minutes": 10, "vfp": 12})
    assert body["status"] == "success" and "snapshot" not in body
    assert body["top_5_corridors"][0]["corridor_name"].startswith("S Egret Bay Blvd")


def test_service_validates_input(demo_service):
    with pytest.raises(demo_service.ServiceError) as e:
        demo_service.run_site({"lat": "x", "lng": 1})
    assert e.value.status == 422


def test_snapshot_writes_maps_and_pdf(demo_service, tmp_path):
    from site_traffic.demo_data import SITE_LAT, SITE_LON
    body = demo_service.run_site({"lat": SITE_LAT, "lng": SITE_LON, "drive_time_minutes": 10, "vfp": 12,
                                  "snapshot": True}, out_root=str(tmp_path))
    out = tmp_path / f"site_{SITE_LAT}_{SITE_LON}"
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert any(n.startswith("satellite_01") for n in pngs)
    assert "trip_distribution.png" in pngs
    assert len([n for n in pngs if n.startswith("satellite_")]) == 5       # top 5 only, nothing else
    assert "corridor_overview.png" in pngs
    assert len(pngs) == 7                                                   # no overview / detail maps / charts
    assert (out / "selected_corridors.json").exists()
    assert body["pdf_report"]["status"] == "generated" and (out / "site_traffic_report.pdf").stat().st_size > 10_000
    assert body["snapshot"]["basemap"]["satellite"]["status"] == "not_requested"


def test_runner_cli(demo_service, tmp_path, capsys):
    import run
    assert run.main(["--demo", "--minutes", "10", "--vfp", "12", "--out", str(tmp_path)]) == 0
    assert "visits/day" in capsys.readouterr().out


def test_api_routes_use_the_service(demo_service):
    from fastapi.testclient import TestClient
    from site_traffic.api import app
    from site_traffic.demo_data import SITE_LAT, SITE_LON
    c = TestClient(app)
    r = c.get(f"/api/test/traffic-data/?lat={SITE_LAT}&lng={SITE_LON}&drive_time_minutes=10")
    assert r.status_code == 200 and r.json()["status"] == "success"
    assert c.post("/api/test/traffic-data/", json={}).status_code == 422


def test_vfp_aliases_and_site_name_title(demo_service, tmp_path):
    from site_traffic.demo_data import SITE_LAT, SITE_LON
    a = demo_service.run_site({"lat": SITE_LAT, "lng": SITE_LON, "drive_time_minutes": 10, "mpds": "12"})
    b = demo_service.run_site({"lat": SITE_LAT, "lng": SITE_LON, "drive_time_minutes": 10, "vfp": 12})
    assert a["trip_generation_analysis"]["ite_supply"]["vfp"] == 12 == b["trip_generation_analysis"]["ite_supply"]["vfp"]
    with pytest.raises(demo_service.ServiceError):
        demo_service.run_site({"lat": SITE_LAT, "lng": SITE_LON, "vfp": 0})


def test_pdf_html_has_title_and_filled_template(demo_service, tmp_path):
    from site_traffic import report, viz
    from site_traffic.demo_data import SITE_LAT, SITE_LON
    demo_service.run_site({"lat": SITE_LAT, "lng": SITE_LON, "drive_time_minutes": 10, "vfp": 12,
                           "site_name": "Anchor & Bay Fuel Site", "snapshot": True}, out_root=str(tmp_path))
    res = demo_service._get_analysis().run(
        SitePayload(SITE_LAT, SITE_LON, __import__("site_traffic.demo_data", fromlist=["x"]).make_drive_time_polygon(10),
                    drive_time_minutes=10, vfp=12, site_name="Anchor & Bay Fuel Site"))
    html = report.build_html(res, {})
    assert "Anchor &amp; Bay Fuel Site" in html and "report-title" in html
    assert "{" not in html.split("</style>")[1]                      # every placeholder was filled


def test_low_count_road_touching_the_site_is_selected_but_far_ones_are_not():
    """A road at the site with a default count (301) is the site's entrance: kept at the floor and selected."""
    rows = [seg("big", "Big Rd", "primary", 30000, (40, 400), (40, 3000)),
            seg("gr", "Green Rd", "residential", 301, (-1500, 20), (200, 20)),        # touches the site
            seg("far", "Far Rd", "residential", 301, (-1500, 900), (200, 900))]       # same count, 900 m away
    r = run(rows)
    green = [c for c in r.selected if c.label == "Green Rd"]
    assert green and green[0].aadt == r.config.min_aadt and green[0].aadt_estimated and green[0].is_access_road
    assert not any(c.label == "Far Rd" for c in r.corridors)
    assert any("touches the site" in w for w in r.warnings)


def test_config_py_defaults_are_valid_and_nonsense_is_rejected():
    from site_traffic import config
    cfg = TripModelConfig().validate()
    assert cfg.min_aadt == config.MIN_AADT and cfg.top_k == config.TOP_K
    for bad in (dict(top_k=0), dict(passby_share=1.5), dict(left_turn_factor=-1), dict(min_potential_share=2)):
        with pytest.raises(ValueError, match="config.py"):
            TripModelConfig(**bad).validate()


def test_changing_a_config_value_changes_the_result():
    rows = _two_road_rows()
    few = run(rows, cfg=TripModelConfig(top_k=1, min_potential_share=0.0))
    many = run(rows, cfg=TripModelConfig(top_k=5, min_potential_share=0.0))
    assert len(few.selected) == 1 and len(many.selected) > 1
