"""
CLI entry point (same pipeline as the API: site_traffic.service.run_site).

    python run.py --lat 29.5060 --lng -95.0676 --minutes 5 --vfp 12
    python run.py --lat 29.5060 --lng -95.0676 --polygon isochrone.geojson --name "Site A"
    python run.py --demo                    # synthetic League City network, no database

Writes the maps, data files and PDF report to <out>/site_<lat>_<lng>/ and prints a short summary.
Database and Mapbox settings come from .env (see .env.example); every threshold and constant of the
algorithm is in site_traffic/config.py.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lng", type=float)
    ap.add_argument("--minutes", type=float, default=5, help="drive time (default 5)")
    ap.add_argument("--vfp", type=int, help="vehicle fueling positions (MPDs), if known")
    ap.add_argument("--polygon", help="GeoJSON / WKT file with the drive-time polygon (else Mapbox isochrone / circle)")
    ap.add_argument("--name", help="site name for titles")
    ap.add_argument("--out", help="output root (default: TRAFFIC_OUTPUT_DIR or output/traffic_analysis)")
    ap.add_argument("--no-files", action="store_true", help="JSON only: skip maps and PDF")
    ap.add_argument("--demo", action="store_true", help="use the synthetic League City network")
    a = ap.parse_args(argv)

    if a.demo:
        os.environ["TRAFFIC_DEMO"] = "1"
        from site_traffic.demo_data import SITE_LAT, SITE_LON
        a.lat = SITE_LAT if a.lat is None else a.lat
        a.lng = SITE_LON if a.lng is None else a.lng
    if a.lat is None or a.lng is None:
        ap.error("--lat and --lng are required (or use --demo)")

    from site_traffic.service import ServiceError, run_site
    params = {"lat": a.lat, "lng": a.lng, "drive_time_minutes": a.minutes, "vfp": a.vfp, "site_name": a.name,
              "snapshot": not a.no_files}
    if a.polygon:
        with open(a.polygon, encoding="utf-8") as f:
            params["drive_time_polygon"] = f.read()
    try:
        body = run_site(params, out_root=a.out)
    except ServiceError as e:
        print(f"error {e.status}: {e.message}", file=sys.stderr)
        return 1

    d, t = body["details"], body["trip_generation_analysis"]["summary"]
    print(f"{d['traffic_segments']} of {d['total_segments']} segments carry measured traffic; "
          f"{t['daily_visits']:,.0f} visits/day, {t['daily_trips']:,} trip ends/day")
    for c in body["top_5_corridors"]:
        print(f"  {c['rank'] or 'A'}. {c['corridor_name']:<34} AADT {c['aadt_cur']:>8,.0f}  "
              f"{c['site_visits_per_day']:>6,.0f} visits/day  {c['trip_share']:.0%}  {c['trip_type']}")
    for w in body["warnings"]:
        print("  warning:", w)
    if "snapshot" in body:
        print(json.dumps({k: body[k] for k in ("snapshot", "pdf_report") if k in body}, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
