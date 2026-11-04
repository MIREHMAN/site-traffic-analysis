# site_traffic

Gas-station site-trip potential. Given a site point and a drive-time polygon it:

1. fetches the road segments that intersect the polygon (PostGIS),
2. keeps only real traffic segments (measured AADT, drivable, no ramps or service roads),
3. ranks corridors by trip potential and selects the top K (default 5),
4. generates site trips for those corridors: visits/day, trip ends, pass-by vs diverted, approach direction, hour of day,
5. writes one **trip map** (Mapbox streets), one **satellite map per top-5 corridor** (Mapbox satellite), data files and a **PDF report**.

## Configuration

Every threshold and constant of the algorithm is a plain value at the top of **`site_traffic/config.py`** (commented, in the block marked EDIT HERE).
Edit, save, restart (`uvicorn site_traffic.api:app --reload` restarts for you; `python run.py` reads it on every run). Nonsense values are rejected with the name of the constant to fix.

## Two entry points, one pipeline (`site_traffic/service.py`)

```bash
pip install -r requirements.txt
cp .env.example .env                       # database + MAPBOX_TOKEN

# API
uvicorn site_traffic.api:app --port 8000   # POST|GET /api/test/traffic-data/?snapshot=true

# CLI runner
python run.py --lat 29.5060 --lng -95.0676 --minutes 5 --vfp 12 --name "Site A"
python run.py --demo                       # synthetic League City network, no database
python -m pytest -q tests
```

`snapshot=true` (or a CLI run without `--no-files`) writes to `output/traffic_analysis/site_<lat>_<lng>/`:
`trip_distribution.png`, `satellite_01..05_*.png`, `site_traffic_report.pdf`, `corridors.csv`,
`selected_corridors.json`, `summary.json`, `trip_distribution.geojson`. Without `MAPBOX_TOKEN` the maps fall back to the
gray road network (the satellite pages then say imagery is unavailable) and the response reports the basemap status.

See **METHODOLOGY.md** for the model, every parameter's source, and the calibration plan.

| Module | Role |
|---|---|
| `service.py` | the pipeline: input validation, DB source, drive-time polygon, response, snapshot |
| `api.py` / `run.py` | thin HTTP and CLI entry points over `service.run_site` |
| `db.py`, `segments.py`, `potential.py`, `tripgen.py` | fetch, real-traffic filter, corridor potential and selection, trip generation |
| `analysis.py` | `TrafficAnalysis` and `TrafficAnalysisResult` (records, GeoJSON, CSV) |
| `viz.py` | the trip map and the per-corridor satellite maps |
| `report.py` + `templates/` | PDF report (WeasyPrint, TezIntel HTML/CSS template) |
| `config.py` | **all editable thresholds and constants**, and the dataclasses that carry them |
| `routing.py` | exit intersection and route from each corridor to the site |
| `demo_data.py` | synthetic network for demos and tests |
