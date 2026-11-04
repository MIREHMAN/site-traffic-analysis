# Gas-Station Site Trip Potential — Method & Guidance

**What it answers:** *Of the vehicles already driving on the counted roads inside a
site's drive-time area, how many could stop at this fuel site each weekday, from
which roads and approach directions, and at what times of day?*

It is a **modeled potential** built from measured AADT. It is not an observed
count or a formal TIA. It replaces "nearest road wins" with "the roads whose
traffic can realistically reach the site win."

---

## 1. Pipeline

```
payload (site lat/lon, drive-time polygon, optional VFP)
   │
   ├─ 1. fetch_intersecting_segments()      PostGIS ST_Intersects (+ clip)       db.py
   ├─ 2. filter_real_traffic_segments()     measured AADT, drivable, non-ramp    segments.py
   │     build_corridors()                  merge fragments by ref/name + gap
   ├─ 3. score_segments() → aggregate_corridors() → select_corridors()         potential.py
   │     potential = AADT × direction × class access × detour decay
   │     top-K, τ threshold, access road kept as context
   ├─ 4. generate_site_trips()              ONLY on selected corridors          tripgen.py
   │     demand (capture) vs supply (ITE 945 × VFP); pass-by/diverted split;
   │     approach arrows; NACS hourly profile
   └─ 5. render / save                      CSV, JSON, GeoJSON (analysis.py); trip + satellite maps (viz.py); PDF (report.py)
```

`TrafficAnalysis.run(payload)` does all of it. Each step is also a public method
(`fetch_segments`, `prepare_segments`, `score_and_select`, `generate_trips`), so
your system can cache step 1 or swap any stage.

---

## 2. The model

### 2.1 "Real traffic" filter
A segment is kept only if **all** of these hold:
* It has a measured AADT ≥ `min_aadt` (500). **Missing AADT is excluded and reported, never treated as 0.**
* Its class is not service, parking, driveway, track, path or a `*_link` ramp. Ramp volumes are already inside the parent freeway's count.
* It is ≥ 25 m long after clipping.

### 2.2 Corridor potential (ranking)

```
potential_c = AADT_c × D × A_class × exp(−λ · detour_c)        (0 if detour_c > 15 min)
detour_c    = 0                         if the road is within 75 m (site frontage)
            = 2 × one-way connector time  otherwise (leave route, visit, rejoin)
D           = 0.5 × (1 + L)             half the traffic must turn left across the road
```

A corridor's potential is the **max** over its segments, not the sum, because the
same vehicles travel the whole road.

Selection rules:
* Top-K (5) by potential.
* τ = 5 % of total area potential. **If fewer pass, return fewer rather than pad.**
* One slot per corridor.
* The site access road is always shown. If it did not earn a rank it is flagged `context_only`.

### 2.3 Trip generation (selected corridors only)

```
demand_c  = min(capture × potential_c,  10 % × AADT_c)
pass-by   = Σ demand on frontage corridors
diverted  = min(Σ demand on other corridors,  pass-by × (1−p)/p)
V_market  = pass-by + diverted
V_ITE     = 345.75 × VFP / 2                      (entering vehicles/day, if VFP known)
V         = min(V_market, V_ITE)                  binding constraint is reported
```

* `V` is distributed to corridors in proportion to their demand. Within a corridor it is split evenly across the live approach arms (AADT is two-way). These splits drive the arrows.
* Trip ends = 2 × V (50:50 in/out, ITE).
* Hourly visits = V × NACS time-of-day profile.

---

## 3. Parameter justification

The percentages are **derived** from the sources, not copied. Each one is a
config field, so you can recalibrate it.

| Parameter | Value | Source | How the number is derived |
|---|---|---|---|
| Detour decay λ | 0.09 /min | NACS 2014: 66 % would drive 5 min, 39 % 10 min for 5¢/gal. NACS 2025: 54 % for 10 min | −ln(0.66)/5 = 0.083 and −ln(0.39)/10 = 0.094. Survey answers include a price incentive, so we take the stricter curve. This also confirms the design doc's provisional β = 0.10. |
| Max detour | 15 min | NACS surveys stop at 10 min | Retention is < 26 % beyond 15 min, and closer competitors (not modeled) win. |
| Left-turn factor L | 0.60 | NACS 2025: 52–67 % would turn left across a busy street | Midpoint of that range → direction factor D = 0.80. |
| Capture rate | 0.08 | TIA (ITE 945, 10 VFP): 1,314 pass-by veh/day ÷ ~15.6–16.9k adjacent AADT ≈ 8 % of raw AADT. NACS fact sheet: ~1,100 customers/day per fuel store ≈ 5.5–7 % of a 15–20k arterial | 0.08 of *effective* (direction-adjusted) exposure sits between the two anchors. |
| Max capture of AADT | 10 % | Upper bound from the TIA anchor | Guards against tiny-AADT / large-station combinations. |
| Pass-by share p | 0.60 | ITE 945 ≈ 75 % (TIA: AM 76, PM 75). WisDOT accepts 0–60 % for LU 944/945 | Uses the agency-defensible cap. It sizes the diverted pool so far roads are not over-credited when competitor data is missing. |
| ITE supply | 345.75 / 31.60 / 26.90 trip ends per VFP (daily / AM / PM) | ITE TGM LU 945 as applied in the Anchor & Bay TIA (Table 6). ITE 11th Ed. merged LU 853/960 into 945 and added VFP sub-categories | Reproduces the TIA's 3,458 / 316 / 269 for 10 VFP (unit-tested). |
| Time of day | 24 / 36 / 34 / 6 % (6–10, 10–15, 15–19, 19–24) | NACS Jan 2020 consumer survey (U.S. Drivers infographic) | 2 % moved to overnight for 24-h stores. Commute-shaped within each daypart. NACS 2024 shows the morning share rising to 27 %. |
| Class access | motorway 0.35, trunk 0.85, others 1.0 | Judgement. WisDOT counts a site as a "freeway location" only within ½ mile of an interchange | **Calibrate first.** It has the weakest evidence. |
| Circuity, connector speed | 1.30, 40 km/h | Typical U.S. urban network circuity is 1.2–1.4 | Used only when no `travel_time_fn` is supplied. |
| τ, top-K, frontage buffer | 5 %, 5, 75 m | Design doc (Site Road Traffic Relevance) | Starting hypotheses. Run a sensitivity test. |

**Distribution logic.** Schultz & Adamson (ITE Journal, Oct 2021) found that
distributing site trips by the movements that can actually enter or exit the site
("Application 2") was the most realistic of the options tested. Our distribution
is proportional to detour-weighted, accessible exposure, which is the same idea
computed from AADT instead of turning counts.

**Built-in sanity checks** (`summary()["checks"]`):
* Model pass-by share vs ITE ~75 % and the WisDOT 0–60 % range.
* Model peak-hour share vs ITE (AM 9.1 %, PM 7.8 % of daily). On the synthetic test the model gives 6.6 % AM and 9.0 % PM. NACS fill-up timing is flatter in the morning and heavier in the afternoon than ITE count peaks, so reconcile this when calibrating.
* Corridors that hit the 10 %-of-AADT cap.
* Comparison with the NACS average of ~1,100 customers/day per store.

---

## 4. Integrating with your system

```python
from site_traffic import TrafficAnalysis, SitePayload, SegmentTableSchema, TripModelConfig

schema = SegmentTableSchema(
    table="traffic.aadt_segments", geom="geom", srid=4326,
    segment_id="id", name="road_name", ref="route_id", road_class="func_class",  # OSM tag or FHWA 1-7
    aadt="aadt", aadt_year="year", aadt_source="source")

analysis = TrafficAnalysis(segment_source=engine_or_conn, schema=schema,
                           config=TripModelConfig(top_k=5),
                           travel_time_fn=None)   # plug in OSRM/Valhalla/your isochrone engine here

result = analysis.run(SitePayload(site_lat=29.4998, site_lon=-95.0597,
                                  drive_time_polygon=geojson_polygon,
                                  drive_time_minutes=10, vfp=12))

result.to_records()     # per-corridor JSON (design-doc Appendix A schema + trips, hourly, approaches)
result.to_geojson()     # corridors + approach arrows + site → drop straight onto your map layer
result.summary()        # totals, binding constraint, dayparts, checks, warnings
result.save("out/")   # CSV / JSON / GeoJSON ; maps + PDF come from python run.py or the API
result.save("out/")     # CSV / JSON / GeoJSON
```

* **Dependencies:** numpy, pandas, matplotlib. Optional: SQLAlchemy or psycopg for the DB, contextily for basemap tiles. Shapely and geopandas geometries are accepted but not required.
* **Travel time:** pass `travel_time_fn(segment_row, access_lonlat, site_lonlat) -> minutes` to use real network times. This is the biggest accuracy upgrade available, because it handles barriers, rivers and interchange spacing.
* **Performance:** step 1 uses `&&` and `ST_Intersects` (GiST index). Corridor merging is O(n²) per road name, which is fine for the few hundred segments in a 10–15-minute polygon.

---

## 5. Calibration plan (before production)

1. Collect daily transaction or vehicle counts from 15–30 existing stations of known VFP, across urban, suburban, rural and freeway-interchange locations.
2. Run the model with each station's VFP **unset**, then regress observed visits on modeled demand. That slope recalibrates `capture_rate`.
3. Where pump or loyalty data shows customer origin (card ZIP codes, telematics), fit `detour_lambda_per_min` and `passby_share` directly.
4. Fit `class_access_factor["motorway"]` on the freeway-interchange subset.
5. Re-check the τ and K sensitivity on the design doc's test sites: highway 2 km away, cul-de-sac, barrier, repeated fragments, missing AADT.

## 6. Known limits

* **No competitor stations.** Adding competitor locations (closest-facility share) is the natural next layer. Until then, `passby_share` caps the diverted pool.
* **Straight-line × circuity** connector time unless `travel_time_fn` is supplied.
* **Two-way AADT** is split evenly across approach arms. Directional counts can replace this if available.
* **Weekday only.** Weekend and seasonal factors (e.g. WisDOT's tourist-area rule, summer AADT ≥ 150 % of the rest of the year) are not applied.

### Sources
* ITE Journal, Oct 2021 — "Updates to the Trip Generation Manual, 11th Edition"; Schultz & Adamson, "Study Area Trip Distribution Method"; Basha, "Trip Generation Estimation Methodology".
* ACE Traffic Engineering, *Anchor & Bay Mixed Use Development TIA*, League City TX, March 2026 (Tables 6–7, Exhibits 7–9).
* NACS U.S. Drivers infographic (Jan 2020 survey): https://www.convenience.org/Topics/Fuels/Documents/USDrivers_infographic.pdf
* NACS Magazine, May 2024 — "A Parting of the Clouds": https://www.nacsmagazine.com/issues/may-2024/a-parting-of-the-clouds
* NACS Magazine, Sep 2025 — "Navigating the Customer's Path to the Pump": https://www.nacsmagazine.com/Issues/September-2025/Navigating-the-Customers-Path-to-the-Pump
* NACS 2014 survey via Fuels Market News: https://fuelsmarketnews.com/nacs-gas-customers-evolve-shop-price/
* NACS industry fact sheet: https://www.nacsshow.com/About/ForTheMedia/Documents/FactSheet_ConvenienceStoreIndustry
* WisDOT pass-by guidance: https://www.wisconsindot.gov/dtsdManuals/traffic-ops/manuals-and-standards/pass-by-trips.pdf
