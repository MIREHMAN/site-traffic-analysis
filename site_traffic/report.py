"""
PDF report for a TrafficAnalysisResult, rendered with WeasyPrint from the TezIntel traffic template
(templates/traffic_pdf_template.html + .css, A4 portrait).

  1   Executive overview   title, two top-corridor cards, ranked corridors table
  2   Overview map        every corridor with measured traffic, top corridors highlighted
  2b  Trip map            potential site trips by approach road
  2c  Details             KPIs, key findings, corridor detail table, trip distribution by approach
  3   Trip generation     peak strip, trip-generation table, pass-by sensitivity
  4+  Satellite maps      one per top corridor with its key figures, two per page
  end Candidates, appendix  top-10 candidate corridors; route-selection criteria, parameters, data quality, warnings

The PNGs come from viz.render_all(); pass its returned dict as `files`. Nothing here recomputes the model:
every figure is read from the result, and the few derived numbers (net trips, pass-by scenarios, projected AADT)
are plain arithmetic on those figures, with their constants in site_traffic/config.py (REPORT_* and CONFIDENCE_*).
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from .config import ReportSettings
from .viz import CATEGORICAL

TEMPLATE_DIR = Path(__file__).parent / "templates"
ACCESS_GREY = "#52514e"


def _n(v, d=0) -> str:
    return f"{v:,.{d}f}"


def _e(v) -> str:
    return escape(str(v))


def _rs(result) -> ReportSettings:
    return getattr(result, "report_settings", None) or ReportSettings()


def _logo_svg() -> str:
    """The official TezIntel logo (templates/tezintel_logo.svg); TEZINTEL_LOGO_SVG points at another file."""
    path = os.getenv("TEZINTEL_LOGO_SVG") or str(TEMPLATE_DIR / "tezintel_logo.svg")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _chip(c) -> str:
    if c.context_only or not c.rank:
        return f'<span class="rank-chip" style="background:{ACCESS_GREY}">A</span>'
    return f'<span class="rank-chip" style="background:{CATEGORICAL[(c.rank - 1) % len(CATEGORICAL)]}">{c.rank}</span>'


def _confidence(c, rs: ReportSettings, today: Optional[_dt.date] = None) -> str:
    """High: counted recently and AADT steady along the corridor; Medium: older / more variable; else Low
    (thresholds in ReportSettings)."""
    year = (today or _dt.date.today()).year
    spread = (c.aadt_max / c.aadt_min) if c.aadt_min else 99.0
    age = year - c.aadt_year if c.aadt_year else 99
    if age <= rs.confidence_high_max_age_years and spread <= rs.confidence_high_max_spread:
        return "High"
    if age <= rs.confidence_medium_max_age_years and spread <= rs.confidence_medium_max_spread:
        return "Medium"
    return "Low"


def _card(header, subtitle, body_html, extra_style: str = "") -> str:
    style = f' style="{extra_style}"' if extra_style else ""
    return (f'<div class="traffic-table-card"{style}><div class="traffic-table-heading">'
            f'<div class="traffic-table-title">{header}</div><div class="traffic-table-subtitle">{subtitle}</div></div>'
            f'{body_html}</div>')


def _rows_html(rows: List[List[str]], aligns: List[str]) -> str:
    return "".join("<tr>" + "".join(f'<td style="text-align:{aligns[i]};">{cell}</td>' for i, cell in enumerate(r)) + "</tr>"
                   for r in rows)


def _table(head: List[tuple], rows: List[List[str]]) -> str:
    """head: [(label, width %, align)]"""
    th = "".join(f'<th style="width:{w}%; text-align:{a};">{label}</th>' for label, w, a in head)
    return (f'<table class="traffic-counts-table"><thead><tr>{th}</tr></thead>'
            f'<tbody>{_rows_html(rows, [a for _, _, a in head])}</tbody></table>')


def _peaks(result):
    """(source label, AM trip ends/hour, PM trip ends/hour) for the site."""
    c, vfp, tr = result.config, result.payload.vfp, result.trips
    if vfp:
        return (f"ITE LU 945 x {vfp} VFP", round(c.ite_am_peak_trip_ends_per_vfp * vfp),
                round(c.ite_pm_peak_trip_ends_per_vfp * vfp))
    return ("modeled hourly profile", round(2 * float(tr.hourly_visits[6:10].max())),
            round(2 * float(tr.hourly_visits[15:19].max())))


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
def _cards(result) -> str:
    rs = _rs(result)
    ranked = [t for t in result.trips.corridors if t.corridor.rank][:2]
    notes = ["High visibility corridor", "Strategic secondary access"]
    out = []
    for i, t in enumerate(ranked):
        c = t.corridor
        proj = c.aadt * (1 + rs.growth_rate) ** rs.growth_years
        kind = "pass-by on the site frontage" if t.trip_type == "pass-by" else f"diverted, {c.detour_min:.1f} min detour"
        out.append(
            f'<div class="traffic-card{" blue" if i else ""}"><div class="card-location">{_e(c.label)}</div>'
            f'<div class="card-sublabel">{_e(c.road_class.title())} corridor &bull; {_e(kind)}</div>'
            f'<div class="metric-label-small">Current AADT</div><div class="metric-value-large">{_n(c.aadt)}'
            f'{"*" if c.aadt_estimated else ""}</div>'
            f'<div class="card-divider"></div>'
            f'<div class="metric-label-small">Projected AADT (+{rs.growth_rate:.1%}/yr, {rs.growth_years} years)</div>'
            f'<div class="metric-value-large small">{_n(proj)}</div>'
            f'<div class="card-footer-note">{_n(t.visits)} site visits/day &bull; {t.share_of_site:.0%} of site trips'
            f' &bull; {notes[i]}</div></div>')
    return "".join(out)


def _criteria(result, s) -> str:
    cfg, rs = result.config, _rs(result)
    rows = [
        ("Real traffic only", f"Measured AADT &ge; {_n(cfg.min_aadt)}; no ramps, service roads or unmeasured segments "
                              f"({s['segments_real_traffic']:,} of {s['segments_fetched']:,} segments kept). A road touching the site "
                              f"(within {cfg.frontage_buffer_m:g} m) with a lower count is counted at the minimum."),
        ("Exit and detour", "Each corridor's exit is the intersection on that road with the shortest drive time to the site "
                            "(road network); detour = that time out and back."),
        ("Corridor potential", "AADT &times; detour retention &times; direction factor &times; road-class access."),
        ("Detour retention", f"exp(&minus;{cfg.detour_lambda_per_min:g} &times; detour minutes), cut-off {cfg.max_detour_min:g} min; "
                             f"opposite-direction left-turn factor {cfg.left_turn_factor:.0%}."),
        ("Selection", f"Roads touching the site are always selected; the remaining slots (up to {cfg.top_k}) go to the corridors "
                      f"holding at least {cfg.min_potential_share:.0%} of total potential."),
        ("Confidence", f"High: counted within {rs.confidence_high_max_age_years} years and AADT steady along the corridor; "
                       f"Medium: within {rs.confidence_medium_max_age_years} years; Low: older or variable."),
    ]
    return _card("Route Selection Criteria", "How routes were selected",
                 _table([("Criterion", 22, "left"), ("Rule", 78, "left")], [[f"<b>{a}</b>", b] for a, b in rows]))


def _corridor_rows(result, limit: int = 10) -> str:
    rs = _rs(result)
    shares = {t.corridor.corridor_id: t.share_of_site for t in result.trips.corridors}
    shown = result.corridors[:limit]                                   # best potential first
    max_pot = max((c.potential for c in shown), default=0) or 1
    rows = []
    for c in shown:
        conf = _confidence(c, rs)
        nseg = len(c.segment_ids)
        kind = ("pass-by" if c.is_access_road else "diverted-link") if c.selected else "not selected"
        rows.append([_chip(c) if c.rank else '<span class="rank-chip" style="background:#cbd5e1">&ndash;</span>',
                     f'<b>{_e(c.label)}</b><span class="segment-note">{nseg} joined segment{"s" if nseg != 1 else ""}'
                     f' &bull; {kind}</span>',
                     _e(c.road_class.title()), _n(c.aadt) + ("*" if c.aadt_estimated else ""),
                     f"{shares[c.corridor_id]:.1%}" if c.corridor_id in shares else "&ndash;",
                     f"{c.potential / max_pot:.2f}", f'<span class="confidence-{conf.lower()}">{conf}</span>'])
    return _rows_html(rows, ["center", "left", "center", "right", "right", "right", "center"])


def _trip_gen_rows(result) -> str:
    cfg, tr, vfp = result.config, result.trips, result.payload.vfp
    rows = []
    if vfp and tr.ite:
        gross = tr.ite["daily_trip_ends"]
        pb = (cfg.ite_passby_am + cfg.ite_passby_pm) / 2
        rows.append(["Convenience Store / Gas Station (ITE LU 945)", f"{vfp} VFP", _n(gross), f"{pb:.0%}",
                     _n(gross * (1 - pb)), _n(tr.ite["am_peak_trip_ends"]), _n(tr.ite["pm_peak_trip_ends"])])
    pb = tr.checks.get("model_passby_share", cfg.passby_share)
    am, pm = round(2 * float(tr.hourly_visits[6:10].max())), round(2 * float(tr.hourly_visits[15:19].max()))
    rows.append(["Modeled site trips (selected corridors)", "&ndash;", _n(tr.trip_ends), f"{pb:.0%}",
                 _n(tr.trip_ends * (1 - pb)), _n(am), _n(pm)])
    return _rows_html(rows, ["left", "center", "right", "center", "right", "right", "right"])


def _pass_by_block(result) -> str:
    tr, model = result.trips, result.trips.checks.get("model_passby_share", result.config.passby_share)
    _src, am, pm = _peaks(result)
    rows = []
    for share, label in _rs(result).pass_by_scenarios:
        mark = " <b>(model)</b>" if abs(share - model) < 0.005 else ""
        rows.append([f"{_e(label)}{mark}", f"{share:.0%}", _n(tr.trip_ends * share), _n(tr.trip_ends * (1 - share)),
                     _n(am * (1 - share)), _n(pm * (1 - share))])
    head = [("Scenario", 36, "left"), ("Pass-by", 10, "center"), ("Pass-by trips/day", 15, "right"),
            ("New trips/day", 15, "right"), ("New AM peak", 12, "right"), ("New PM peak", 12, "right")]
    return _card("Pass-By Scenario Sensitivity", "Pass-by share changes how many trips are new to the road network",
                 _table(head, rows), "margin-top: 8px;")


def _header(date_str: str) -> str:
    return (f'<div class="header"><div class="header-left"><div class="logo">{_logo_svg()}</div>'
            f'<p class="report-date">Report generated on {date_str}</p></div><div class="header-right">'
            f'<a href="https://www.tezintel.com" class="website-link">www.tezintel.com</a></div></div>')


def _company(site_name: str, line1: str, p, big: bool = False) -> str:
    fuel = f" &bull; Vehicle fueling positions: {p.vfp}" if p.vfp else ""
    return (f'<div class="company-info"><div class="report-title{"" if big else " small"}">{_e(site_name)}</div>'
            f'<div class="company-name">Commercial Site Traffic &amp; Arterial Analysis</div><div class="company-details">'
            f'<div>{line1}</div><div>Site Location: Lat {p.site_lat:.5f}, Lng {p.site_lon:.5f}{fuel}</div></div></div>')


def _map_block(path: Optional[str], alt: str) -> str:
    if not path:
        return '<p class="note-line">Map not available.</p>'
    return f'<div class="map-viewport"><img class="map-image" src="{Path(path).resolve().as_uri()}" alt="{_e(alt)}"></div>'


def _page(header: str, company: str, title: str, subtitle: str, body: str) -> str:
    return ('<div class="page-break"></div>' + header + company + '<section class="traffic-section">'
            f'<h2 class="section-title">{title}</h2><p class="section-subtitle">{subtitle}</p>{body}</section>')


def _findings(result) -> List[str]:
    tr, p = result.trips, result.payload
    out = []
    if tr.corridors:
        top = max(tr.corridors, key=lambda t: t.visits)
        out.append(f"<b>{_e(top.corridor.label)}</b> is the largest source: {_n(top.visits)} visits/day "
                   f"({top.share_of_site:.0%} of site trips), {_e(top.trip_type)}.")
    pb = sum(t.visits for t in tr.corridors if t.trip_type == "pass-by")
    if tr.visits:
        out.append(f"About <b>{pb / tr.visits:.0%}</b> of site visits are pass-by (drivers already on the frontage road); "
                   f"the rest are diverted from roads up to {result.config.max_detour_min:g} min away.")
    am, pm = tr.hourly_visits[6:10].max(), tr.hourly_visits[15:19].max()
    out.append(f"Busiest hours: <b>{_n(am)}</b> visits (AM peak) and <b>{_n(pm)}</b> visits (PM peak).")
    out.append(f"Binding constraint: {_e(tr.binding_constraint)}"
               + ("" if p.vfp else " &mdash; no fueling-position count was given, so this is market potential, not a cap."))
    return out


def _kpi_strip(result, s) -> str:
    tr = result.trips
    items = [(_n(tr.visits), "vehicles/day entering the site"), (_n(tr.trip_ends), "trip ends/day (in + out)"),
             (_n(tr.demand_visits), "market demand, visits/day"),
             (_n(s["segments_real_traffic"]), f"traffic segments of {_n(s['segments_fetched'])} fetched"),
             (f"{s['selected_potential_coverage']:.0%}", "of corridor potential covered")]
    boxes = "".join(
        f'<div style="float:left; width:18.4%; margin-right:{"0" if i == len(items) - 1 else "2%"}; background:#F8FAFC; '
        f'border:1px solid #E2E8F0; border-radius:6px; padding:6px 8px; box-sizing:border-box;">'
        f'<div style="font-size:15px; font-weight:700; color:#0F172A;">{v}</div>'
        f'<div style="font-size:8px; color:#64748B;">{label}</div></div>' for i, (v, label) in enumerate(items))
    return f'<div style="margin-bottom:8px; clear:both; overflow:hidden;">{boxes}</div>'


def _details_page(result, s, header, company) -> str:
    tr = result.trips
    drows = []
    for t in tr.corridors:
        c = t.corridor
        drows.append([_chip(c), f"<b>{_e(c.label)}</b>", _e(c.road_class.title()),
                      _n(c.aadt) + ("*" if c.aadt_estimated else ""),
                      f"{c.distance_m / 1609.344:.2f}", "frontage" if c.is_access_road else f"{c.detour_min:.1f}",
                      f"{c.decay:.0%}", _e(t.trip_type), _n(t.visits), f"{t.share_of_site:.0%}"])
    detail = _table([("#", 5, "center"), ("Corridor", 24, "left"), ("Class", 10, "center"), ("AADT", 9, "right"),
                     ("Dist (mi)", 8, "right"), ("Detour (min)", 10, "right"), ("Retention", 9, "right"),
                     ("Trip type", 11, "center"), ("Visits/day", 8, "right"), ("Share", 6, "right")], drows)
    arows = []
    for t in tr.corridors:
        for a in t.corridor.approaches:
            if t.visits * a.flow_share > 0:
                arows.append([_chip(t.corridor), _e(t.corridor.label), _e(a.from_direction.capitalize()), _e(t.trip_type),
                              _n(t.visits * a.flow_share), f"{t.share_of_site * a.flow_share:.1%}"])
    appr = _table([("#", 6, "center"), ("Corridor", 38, "left"), ("From", 14, "left"), ("Trip type", 16, "center"),
                   ("Visits/day", 14, "right"), ("Share of site trips", 12, "right")], arows)
    finds = "".join(f'<li style="margin-bottom:3px;">{f}</li>' for f in _findings(result))
    body = (_kpi_strip(result, s)
            + _card("Key Findings", "What drives the site's trips",
                    f'<ul style="margin:6px 0 6px 0; padding-left:22px; font-size:9.5px; color:#334155;">{finds}</ul>')
            + _card("Selected Corridors &mdash; Detail", "Measured traffic, distance, detour and modeled site visits", detail,
                    "margin-top: 8px;")
            + _card("Trip Distribution by Approach", "The figures behind the trip map", appr, "margin-top: 8px;"))
    return _page(header, company, "Corridor Details &amp; Trip Distribution",
                 "Every selected corridor with the figures behind the maps", body)


def _facts(t) -> str:
    c = t.corridor
    am, pm = t.hourly_visits[6:10].max(), t.hourly_visits[15:19].max()
    appr = " &bull; ".join(f"from {_e(a.from_direction)} {_n(t.visits * a.flow_share)} ({a.flow_share:.0%})"
                           for a in c.approaches if t.visits * a.flow_share > 0)
    rows = [[f"AADT {_n(c.aadt)}{'*' if c.aadt_estimated else ''}" + (f" ({c.aadt_year})" if c.aadt_year else ""),
             f"Along corridor {_n(c.aadt_min)}&ndash;{_n(c.aadt_max)}", f"Distance {c.distance_m:,.0f} m",
             "Detour: frontage" if c.is_access_road else f"Detour {c.detour_min:.1f} min ({c.decay:.0%} retained)"],
            [f"Potential {_n(c.potential)} veh/day ({c.potential_share:.0%})", f"Visits/day {_n(t.visits)} ({t.visits / c.aadt:.1%} of AADT)",
             f"Peak hour AM {_n(am)} / PM {_n(pm)}", appr or "&ndash;"]]
    cells = "".join("<tr>" + "".join(f'<td style="font-size:8px; padding:3px 6px;">{x}</td>' for x in r) + "</tr>" for r in rows)
    return f'<table class="traffic-counts-table" style="margin-top:3px;"><tbody>{cells}</tbody></table>'


def _satellite_pages(result, files, header, company) -> str:
    items = []
    for t in result.trips.corridors:
        c = t.corridor
        key = f"satellite_{c.rank:02d}" if c.rank else None
        if key and files.get(key):
            items.append(
                f'<div class="sat-block"><div class="corridor-card-title">{_chip(c)} {_e(c.label)} &mdash; AADT {_n(c.aadt)}'
                f'{"*" if c.aadt_estimated else ""} &bull; {_n(t.visits)} site visits/day &bull; {t.share_of_site:.0%} of site trips'
                f' &bull; {_e(t.trip_type)}</div>'
                f'<div class="map-viewport"><img class="sat-map" src="{Path(files[key]).resolve().as_uri()}" '
                f'alt="{_e(c.label)}"></div>{_facts(t)}</div>')
    pages = []
    for i in range(0, len(items), 2):
        pages.append('<div class="page-break"></div>' + header + company
                     + '<section class="traffic-section"><h2 class="section-title">Satellite Route Focus Maps</h2>'
                       '<p class="section-subtitle">The highlighted road is the corridor; arrows show the direction of approach</p>'
                     + "".join(items[i:i + 2]) + "</section>")
    return "".join(pages)


def _candidates_page(result, header, company) -> str:
    cfg = result.config
    rows = []
    for _, r in result.corridor_table().head(10).iterrows():
        rank = r["rank"]
        rows.append([f"{int(rank)}" if rank == rank and rank else "&ndash;", _e(r["name"]), _e(str(r["road_class"]).title()),
                     _n(r["aadt"]), _n(r["distance_m"]), f"{r['detour_min']:.1f}", f"{r['decay']:.0%}",
                     f"{r['potential_share']:.1%}", "yes" if r["selected"] else "", _n(r["site_visits_per_day"])])
    table = _table([("Rank", 6, "center"), ("Corridor", 27, "left"), ("Class", 10, "center"), ("AADT", 9, "right"),
                    ("Dist (m)", 8, "right"), ("Detour", 8, "right"), ("Retention", 9, "right"),
                    ("Potential", 9, "right"), ("Selected", 7, "center"), ("Visits/day", 7, "right")], rows)
    note = (f'<p class="note-line">Roads touching the site are always selected; other corridors are selected when they hold at '
            f'least {cfg.min_potential_share:.0%} of total potential (up to {cfg.top_k} in all).</p>')
    return _page(header, company, "Candidate Corridors",
                 "The 10 best-scoring corridors and why the others were not selected",
                 note + _card("Candidate Corridors (top 10 by potential)", "Ranked by modeled trip potential", table))


def _notes_page(result, s, header, company, polygon_source, basemap) -> str:
    cfg, tr, p = result.config, result.trips, result.payload
    params = [["Capture rate", f"{cfg.capture_rate:.0%} of effective passing vehicles", "ITE LU 945 trip generation and average-store customer counts"],
              ["Pass-by share", f"{cfg.passby_share:.0%}", "WisDOT accepts 0&ndash;60 %; ITE data ~75 %"],
              ["Detour decay", f"exp(&minus;{cfg.detour_lambda_per_min:g} &times; min), cut-off {cfg.max_detour_min:g} min",
               "consumer fuel-stop surveys"],
              ["Left-turn factor", f"{cfg.left_turn_factor:.0%}", "consumer surveys: 52&ndash;67 % would cross traffic"],
              ["Minimum AADT", _n(cfg.min_aadt), "below this a road is not a fuel market"],
              ["ITE supply", f"{cfg.ite_daily_trip_ends_per_vfp:g} trip ends per VFP per weekday", "ITE Trip Generation, LU 945"],
              ["Time of day", "weekday fill-up timing by daypart, commute-peak shape", "consumer survey, ITE hourly distribution"]]
    ptable = _table([("Parameter", 20, "left"), ("Value", 42, "left"), ("Basis", 38, "left")], params)
    bm = basemap or {}

    def base(kind, label):
        b = bm.get(kind) or {}
        if b.get("status") == "ok":
            return f"{label}: Mapbox {_e(b.get('style'))} &copy; Mapbox &copy; OpenStreetMap."
        why = f" Mapbox error: {_e(b['error'])}" if b.get("error") else " (no MAPBOX_TOKEN or not requested)"
        return f"{label}: road network only.{why}"
    dropped = ", ".join(f"{_e(k)}: {v:,}" for k, v in s["drop_reasons"].items()) or "none"
    area = {"mapbox_isochrone": "Mapbox isochrone", "circle_approximation": "an approximate circle, not a true drive-time isochrone",
            "provided": "a supplied polygon"}.get(polygon_source or "provided", _e(polygon_source))
    ite = tr.ite or {}
    bullets = [
        f"<b>Traffic data:</b> measured AADT per road segment; {s['segments_real_traffic']:,} of {s['segments_fetched']:,} "
        f"fetched segments kept. Excluded &mdash; {dropped}.",
        "<b>Supply:</b> ITE Land Use 945 trip generation"
        + (f" &mdash; {_n(ite.get('daily_visits', 0))} visits/day for {p.vfp} fueling positions." if p.vfp else "."),
        f"<b>Pass-by share modeled:</b> {tr.checks.get('model_passby_share', 0):.0%} "
        f"(target {tr.checks.get('target_passby_share', 0):.0%}; ITE reference ~75 %).",
        f"<b>Drive-time area:</b> {area}.",
        "<b>Basemaps:</b> " + base("trip_map", "Overview and trip maps") + " " + base("satellite", "Satellite pages"),
    ]
    lis = "".join(f'<li style="margin-bottom:3px;">{b}</li>' for b in bullets)
    warn = ""
    if result.warnings:
        wl = "".join(f'<li style="margin-bottom:3px;">{_e(w)}</li>' for w in result.warnings)
        warn = _card("Warnings", "Conditions that affect how to read this report",
                     f'<ul style="margin:6px 0; padding-left:22px; font-size:9.5px; color:#334155;">{wl}</ul>', "margin-top: 8px;")
    body = (_criteria(result, s)
            + _card("Model Parameters", "Starting values to calibrate against real site counts; all editable in site_traffic/config.py",
                    ptable, "margin-top: 8px;")
            + _card("Data &amp; Basemaps", "Where the numbers and imagery come from",
                    f'<ul style="margin:6px 0; padding-left:22px; font-size:9.5px; color:#334155;">{lis}</ul>', "margin-top: 8px;")
            + warn
            + '<p class="note-line" style="margin-top:10px;">Modeled potential from measured AADT &mdash; not an observed '
              'count or a formal traffic impact analysis.</p>')
    return _page(header, company, "Appendix: Method, Data &amp; Notes",
                 "Route-selection criteria, parameters with their sources, data quality and basemap status", body)


def _fill(template: str, values: Dict[str, str]) -> str:
    """Replace {name} tokens in one pass, so braces inside the substituted CSS are never re-scanned."""
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)


def build_html(result, files: Dict[str, str], polygon_source: Optional[str] = None,
               basemap: Optional[dict] = None) -> str:
    p, s = result.payload, result.summary()
    site_name = p.site_name or f"Site {p.site_lat:.5f}, {p.site_lon:.5f}"
    minutes = f"{p.drive_time_minutes:g}" if p.drive_time_minutes else "n/a"
    area = {"mapbox_isochrone": "Mapbox isochrone", "circle_approximation": "approximate circle",
            "provided": "supplied polygon"}.get(polygon_source or "provided", _e(polygon_source))
    header = _header(f"{_dt.date.today():%d %B %Y}")
    _src, am, pm = _peaks(result)
    with open(TEMPLATE_DIR / "traffic_pdf_template.css", encoding="utf-8") as f:
        css = f.read()
    with open(TEMPLATE_DIR / "traffic_pdf_template.html", encoding="utf-8") as f:
        template = f.read()
    estimated = any(t.corridor.aadt_estimated for t in result.trips.corridors)
    values = {
        "css_content": css, "site_name": _e(site_name), "header_html": header, "minutes": minutes,
        "company_info_1": _company(site_name, f"Study Area: {minutes}-Minute Drive-Time Polygon ({area})", p, big=True),
        "company_info_2": _company(site_name, f"Study Area: {minutes}-Minute Drive-Time &bull; Network Boundary &amp; Corridor Tracks", p),
        "company_info_2b": _company(site_name, "Potential site trips by approach road", p),
        "company_info_3": _company(site_name, "Trip Generation &amp; Pass-By Scenarios", p),
        "traffic_cards": _cards(result), "table_rows_html": _corridor_rows(result),
        "aadt_note": ('<p class="note-line">* Count below the %s minimum (usually a default value for uncounted local streets) '
                      'on a road that touches the site; counted at the minimum.</p>' % _n(result.config.min_aadt)
                      if estimated else ""),
        "overview_map_block": _map_block(files.get("corridor_overview"), "Corridor overview"),
        "trip_map_block": _map_block(files.get("trip_distribution"), "Trip distribution"),
        "am_peak_vph": _n(am), "am_in_vph": _n(round(am / 2)), "am_out_vph": _n(am - round(am / 2)),
        "pm_peak_vph": _n(pm), "pm_in_vph": _n(round(pm / 2)), "pm_out_vph": _n(pm - round(pm / 2)),
        "trip_gen_table_rows": _trip_gen_rows(result), "pass_by_scenarios_block": _pass_by_block(result),
        "details_page_html": _details_page(result, s, header,
                                           _company(site_name, "Selected corridors, detail and trip distribution", p)),
        "satellite_pages_html": _satellite_pages(
            result, files, header,
            _company(site_name, "Satellite route focus &bull; highlighted road = corridor", p)),
        "closing_pages_html": (
            _candidates_page(result, header, _company(site_name, "Candidate corridors and selection", p))
            + _notes_page(result, s, header, _company(site_name, "Appendix: route selection, method, data and notes", p),
                          polygon_source, basemap)),
    }
    return _fill(template, values)


def _quiet_fontconfig() -> None:
    """On Windows the GTK runtime ships no fontconfig file, so Pango prints "Cannot load default config file".
    Point it at a minimal one (system font folders) before WeasyPrint loads Pango. An existing setting wins.
    The GTK DLLs use the old msvcrt environment, which os.environ does not touch, so set it there too."""
    if os.name != "nt" or os.environ.get("FONTCONFIG_FILE"):
        return
    conf = str(TEMPLATE_DIR / "fonts.conf")
    os.environ["FONTCONFIG_FILE"] = conf
    try:
        import ctypes
        ctypes.cdll.msvcrt._putenv(f"FONTCONFIG_FILE={conf}".encode("mbcs"))
    except Exception:
        pass


def build_pdf(result, files: Dict[str, str], pdf_path: str, polygon_source: Optional[str] = None,
              basemap: Optional[dict] = None) -> str:
    _quiet_fontconfig()
    from weasyprint import HTML
    out_dir = os.path.dirname(os.path.abspath(pdf_path))
    os.makedirs(out_dir, exist_ok=True)
    html = build_html(result, files, polygon_source, basemap)
    HTML(string=html, base_url=out_dir).write_pdf(pdf_path, optimize_images=True, jpeg_quality=85, dpi=170)
    return pdf_path
