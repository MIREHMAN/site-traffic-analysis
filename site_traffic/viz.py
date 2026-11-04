"""
The two map types of the report (matplotlib; basemaps from Mapbox).

  corridor_overview.png     every traffic corridor, top 5 highlighted, name tags only
  trip_distribution.png     all selected corridors, arrows sized by site visits, figures panel
  satellite_<rank>.png      one satellite map per top-5 corridor (Mapbox satellite-v9)

Basemaps: the Mapbox Static Images API when MAPBOX_TOKEN is set - streets-v12 (Mapbox street layout) under the trip map
(MAPBOX_STYLE), satellite-v9 under the corridor maps (MAPBOX_SATELLITE_STYLE). Without a token,
or when Mapbox fails, the gray road network is drawn instead and the status is reported.
"""
from __future__ import annotations

import io
import math
import os
import re
import urllib.request
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.patches import Polygon as MplPolygon

from .geo import lonlat_to_webmercator, walk_along

# ---- palette (validated default reference palette, light mode) ------------- #
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CONTEXT_ROAD = "#d6d5ce"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ACCESS_CONTEXT = "#52514e"
SITE_COLOR = "#e34948"
ALL_SEGMENTS_GRAY = "#b9c0c8"
HIGHLIGHT_BLUE = "#2a78d6"
STREETS_STYLE = "mapbox/streets-v12"
SATELLITE_STYLE = "mapbox/satellite-v9"
HALO = [pe.withStroke(linewidth=2.5, foreground="#0b0b0b")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


class _Map:
    """Local-metre -> Web Mercator transform shared by all map functions."""
    def __init__(self, result):
        self.r = result
        self.proj = result.projection
        site_ll = np.array([[result.payload.site_lon, result.payload.site_lat]])
        self.site = lonlat_to_webmercator(site_ll)[0]

    def xy(self, local_xy: np.ndarray) -> np.ndarray:
        return lonlat_to_webmercator(self.proj.inv(np.atleast_2d(local_xy)))

    def ll(self, lonlat: np.ndarray) -> np.ndarray:
        return lonlat_to_webmercator(lonlat)


def _corridor_color(result, corridor) -> str:
    if corridor.context_only:
        return ACCESS_CONTEXT
    return CATEGORICAL[(corridor.rank - 1) % len(CATEGORICAL)]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


# basemap outcome per map type, read by the service/report:
#   {"trip_map": {...}, "satellite": {...}}  each {"style", "status": ok|failed|not_requested, "error"}
BASEMAP_STATUS: Dict[str, Dict[str, object]] = {}
_basemap_cache: Dict[str, np.ndarray] = {}
_MERC_R = 6378137.0


def mapbox_token() -> Optional[str]:
    return os.getenv("MAPBOX_TOKEN") or None


def _style(kind: str) -> str:
    return os.getenv("MAPBOX_SATELLITE_STYLE", SATELLITE_STYLE) if kind == "satellite" \
        else os.getenv("MAPBOX_STYLE", STREETS_STYLE)


def _fetch_mapbox_image(ext, style: str) -> np.ndarray:
    """Static image covering exactly `ext` (x0, x1, y0, y1 in Web Mercator metres)."""
    from PIL import Image
    x0, x1, y0, y1 = ext
    w_m, h_m = x1 - x0, y1 - y0
    longest = max(w_m, h_m)
    W = max(int(round(1280 * w_m / longest)), 16)
    H = max(int(round(1280 * h_m / longest)), 16)
    mpp = longest / 1280                                       # metres per logical pixel
    zoom = min(max(math.log2(2 * math.pi * _MERC_R / (512 * mpp)), 0), 19.5)   # 512 px world tiles
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lon = math.degrees(cx / _MERC_R)
    lat = math.degrees(2 * math.atan(math.exp(cy / _MERC_R)) - math.pi / 2)
    url = (f"https://api.mapbox.com/styles/v1/{style}/static/{lon:.6f},{lat:.6f},{zoom:.3f},0/{W}x{H}@2x"
           f"?access_token={mapbox_token()}&attribution=false&logo=false")
    key = f"{style}|{lon:.5f}|{lat:.5f}|{zoom:.3f}|{W}x{H}"
    if key not in _basemap_cache:
        with urllib.request.urlopen(url, timeout=30) as r:
            _basemap_cache[key] = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
    return _basemap_cache[key]


def _add_mapbox_basemap(ax, ext, kind: str = "trip_map") -> bool:
    style = _style("satellite" if kind == "satellite" else "trip_map")
    status = BASEMAP_STATUS.setdefault(kind, {})
    try:
        img = _fetch_mapbox_image(ext, style)
    except Exception as e:                       # never leak the token in the message
        msg = re.sub(r"access_token=[^&\s'\"]+", "access_token=***", f"{type(e).__name__}: {e}")
        status.update(style=style, status="failed", error=msg)
        return False
    ax.imshow(img, extent=(ext[0], ext[1], ext[2], ext[3]), origin="upper", zorder=0, interpolation="bilinear")
    credit = "© Mapbox © OpenStreetMap © Maxar" if kind == "satellite" else "© Mapbox © OpenStreetMap"
    ax.text(0.995, 0.005, credit, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=INK2,
            zorder=30, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.7))
    if status.get("status") != "failed":
        status.update(style=style, status="ok", error=None)
    return True


def _style_map(ax, m: _Map, extent=None, basemap=False, kind: str = "trip_map", light_ink: bool = False):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if basemap and extent is not None and mapbox_token():
        _add_mapbox_basemap(ax, extent, kind)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    _north_arrow(ax, light_ink)
    _scale_bar(ax, m, light_ink)


def _north_arrow(ax, light_ink: bool = False):
    col, fx = ("white", HALO) if light_ink else (INK2, None)
    ax.annotate("N", xy=(0.965, 0.95), xytext=(0.965, 0.87), xycoords="axes fraction",
                ha="center", va="center", fontsize=9, color=col, fontweight="bold", path_effects=fx, zorder=40,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, path_effects=fx))


def _scale_bar(ax, m: _Map, light_ink: bool = False):
    col, fx = ("white", HALO) if light_ink else (INK2, None)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    k = 1 / math.cos(math.radians(m.r.payload.site_lat))     # mercator stretch
    span_m = (x1 - x0) / k
    nice = [50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]
    L = min(nice, key=lambda v: abs(v - span_m / 5))
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.04 * (y1 - y0)
    ax.plot([bx, bx + L * k], [by, by], color=col, lw=2, solid_capstyle="butt", zorder=40, path_effects=fx)
    ax.text(bx + L * k / 2, by + 0.012 * (y1 - y0), f"{L/1000:g} km" if L >= 1000 else f"{L} m",
            ha="center", va="bottom", fontsize=7, color=col, zorder=40, path_effects=fx)


def _extent_from_points(pts: np.ndarray, pad_frac=0.08, min_span=600.0):
    xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin, min_span) * (1 + 2 * pad_frac)
    return (cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2)


# ---- labels that never overlap -------------------------------------------------------------------------
def _queue_label(ax, anchor, txt, color, prio=1, fontsize=7.5, bold=False, ec=None, **kw):
    """Register a label; _flush_labels places all of them without overlaps (and without covering markers)."""
    if not hasattr(ax, "_lblq"):
        ax._lblq = []
    ax._lblq.append(dict(anchor=np.asarray(anchor, float), txt=txt, color=color, prio=prio, fs=fontsize, bold=bold,
                         ec=ec or color, **kw))


def _mark_obstacle(ax, xy, r_px=11):
    if not hasattr(ax, "_obst"):
        ax._obst = []
    ax._obst.append((np.asarray(xy, float).ravel()[:2], r_px))


def _overlap_area(a: Bbox, b: Bbox) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def _flush_labels(ax):
    """Place queued labels: try the anchor, then rings of offsets around it, keeping the first free spot
    (inside the map, clear of the legend, markers and earlier labels); a thin leader joins a moved label."""
    queue = sorted(getattr(ax, "_lblq", []), key=lambda r: r["prio"])
    if not queue:
        return
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pad = 3.0
    taken = []
    leg = ax.get_legend()
    if leg is not None:
        taken.append(leg.get_window_extent(rend).expanded(1.0, 1.0))
    for xy, r in getattr(ax, "_obst", []):
        d = ax.transData.transform(xy)
        taken.append(Bbox.from_extents(d[0] - r, d[1] - r, d[0] + r, d[1] + r))
    frame = ax.get_window_extent(rend)
    inner = Bbox.from_extents(frame.x0 + 4, frame.y0 + 4, frame.x1 - 4, frame.y1 - 4)
    offsets = [(0, 0)] + [(1.7 * r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
                          for r in (24, 40, 58, 80, 106, 136) for a in range(0, 360, 45)]
    for r in queue:
        kw = dict(ha="center", va="center", fontsize=r["fs"], color=INK, zorder=52,
                  fontweight="bold" if r["bold"] else "normal",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=r["ec"], lw=1.0, alpha=0.96),
                  arrowprops=dict(arrowstyle="-", color=r["color"], lw=0.9, shrinkA=0, shrinkB=0))
        t = ax.annotate(r["txt"], r["anchor"], xytext=(0, 0), textcoords="offset points", **kw)
        best, best_cost = (0, 0), math.inf
        for dx, dy in offsets:
            t.xyann = (dx, dy)
            bb = t.get_window_extent(rend)
            bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
            cost = sum(_overlap_area(bb, o) for o in taken)
            if bb.x0 < inner.x0 or bb.x1 > inner.x1 or bb.y0 < inner.y0 or bb.y1 > inner.y1:
                cost += 5000.0
            cost += 0.01 * math.hypot(dx, dy)                  # prefer staying near the anchor
            if cost < best_cost:
                best, best_cost = (dx, dy), cost
            if cost < 1.0:
                break
        t.xyann = best
        bb = t.get_window_extent(rend)
        taken.append(Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad))


def _draw_all_segments(ax, m, color=ALL_SEGMENTS_GRAY, lw=1.3, alpha=0.9, zorder=4):
    """Every fetched road segment (counted or not) as a light-gray line, under the highlighted corridors."""
    for parts in m.r.segments["parts_ll"]:
        for p in parts:
            if len(p) >= 2:
                xy = m.ll(p)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)


def _draw_polygon(ax, m: _Map, fill=True):
    for ring in m.r.polygon_rings:
        xy = m.ll(ring)
        if fill:
            ax.add_patch(MplPolygon(xy, closed=True, facecolor="#2a78d6", alpha=0.04, edgecolor="none", zorder=1))
        ax.plot(xy[:, 0], xy[:, 1], color="#2a78d6", lw=1.2, ls=(0, (5, 3)), alpha=0.7, zorder=2)


def _draw_context_roads(ax, m: _Map, kept_only=False, color=CONTEXT_ROAD, lw=0.9, zorder=3):
    seg = m.r.segments
    if kept_only:
        seg = seg[seg["drop_reason"] == ""]
    for parts in seg["parts_ll"]:
        for p in parts:
            if len(p) >= 2:
                xy = m.ll(p)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=zorder)


def _draw_site(ax, m: _Map, label=True):
    ax.scatter([m.site[0]], [m.site[1]], marker="*", s=260, color=SITE_COLOR,
               edgecolor="white", linewidth=1.2, zorder=30)
    _mark_obstacle(ax, m.site, 12)
    if label:
        _queue_label(ax, m.site, m.r.payload.site_name or "Site", INK, prio=0, fontsize=8, bold=True, ec="#888888")


def _offset_path(xy: np.ndarray, offset: float) -> np.ndarray:
    """Shift a polyline sideways (right-hand side of travel) so arrows sit beside the road."""
    if len(xy) < 2 or offset == 0:
        return xy
    d = np.gradient(xy, axis=0)
    n = np.column_stack([d[:, 1], -d[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return xy + n * offset


def _draw_approach_arrow(ax, m: _Map, approach, corridor, visits, share_of_site, color, lw,
                         arrow_len_m, offset, label=True, zorder=25):
    # last `arrow_len_m` metres of the arm (far -> access)
    arm = approach.path_xy
    seglen = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    L = float(seglen.sum())
    start_d = max(0.0, L - arrow_len_m)
    # resample tail section
    ds = np.linspace(start_d, L, 25)
    tail = np.array([walk_along(arm, d) for d in ds])
    xy = m.xy(tail)
    xy = _offset_path(xy, offset)
    ax.plot(xy[:-2, 0], xy[:-2, 1], color=color, lw=lw, solid_capstyle="round", zorder=zorder, alpha=0.95)
    ax.annotate("", xy=xy[-1], xytext=xy[-3],
                arrowprops=dict(arrowstyle=f"-|>,head_length={0.35 + lw/25:.2f},head_width={0.2 + lw/40:.2f}",
                                color=color, lw=lw * 0.6, shrinkA=0, shrinkB=0), zorder=zorder + 1)
    if label:
        _queue_label(ax, xy[0], f"from {approach.from_direction}\n{visits:,.0f} veh/day · {share_of_site:.0%}", color,
                     prio=2 + (1.0 - min(visits / 1000.0, 0.99)))


def _draw_connector(ax, m: _Map, corridor, color, show_label=True):
    """Exit intersection -> site, drawn along the real route; the arrowhead is at the site."""
    a = m.xy(corridor.access_xy)[0]
    if corridor.is_access_road:
        if show_label:
            ax.annotate("site frontage · pass-by", a, xytext=(10, 14), textcoords="offset points",
                        fontsize=7, color=INK2, zorder=28,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        return
    route = getattr(corridor, "route_xy", None)
    xy = m.xy(route) if route is not None and len(route) >= 2 else np.vstack([a, m.site])
    ax.plot(xy[:-1, 0], xy[:-1, 1], color=color, lw=2.2, ls=(0, (4, 2.5)), solid_capstyle="round", zorder=24,
            path_effects=[pe.withStroke(linewidth=4.2, foreground="white", alpha=0.85)])
    ax.annotate("", xy=xy[-1], xytext=xy[-2] if len(xy) > 1 else a, zorder=25,
                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35", color=color, lw=2.2,
                                shrinkA=0, shrinkB=4))
    ax.scatter([a[0]], [a[1]], s=44, color="white", edgecolor=color, lw=2, zorder=29)       # the exit intersection
    _mark_obstacle(ax, a, 9)
    if show_label:
        mid = xy[len(xy) // 2]
        ax.annotate(f"exit · +{corridor.detour_min:.1f} min detour", mid, xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color=INK2, zorder=28,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


# ============================================================================ #
# 1. Site trip distribution (all selected corridors)
# ============================================================================ #
def plot_trip_distribution(result, path: str, basemap=False):
    """Full-page map: Mapbox streets layout, thin arrows beside the roads. All figures are in the PDF tables."""
    m = _Map(result)
    fig_w, fig_h, map_h = 9.5, 10.6, 0.945
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, map_h])
    trips = result.trips
    vmax = max([t.visits for t in trips.corridors] or [1])
    pts = [m.site[None, :]] + [m.xy(t.corridor.access_xy) for t in trips.corridors]
    ext = _fit_extent(_extent_from_points(np.vstack(pts), 0.30, min_span=1600), fig_w / (fig_h * map_h))
    span = ext[1] - ext[0]
    drawn = bool(basemap and mapbox_token() and _add_mapbox_basemap(ax, ext, "trip_map"))
    _draw_polygon(ax, m, fill=not drawn)
    _draw_all_segments(ax, m)                       # every fetched segment, light gray, under the highlights
    for t in trips.corridors:
        c = t.corridor; col = _corridor_color(result, c)
        for p in c.parts_xy:                        # thin highlight so the original road stays visible
            xy = m.xy(p)
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.0, alpha=0.6, solid_capstyle="round", zorder=8)
    for t in trips.corridors:
        c = t.corridor; col = _corridor_color(result, c)
        for a in c.approaches:
            v = t.visits * a.flow_share
            if v <= 0:
                continue
            _draw_approach_arrow(ax, m, a, c, v, v / trips.visits if trips.visits else 0, col, 1.2 + 2.6 * v / vmax,
                                 arrow_len_m=min(a.length_m, span * 0.075), offset=span * 0.010, label=True)
        _draw_connector(ax, m, c, col, show_label=False)
    _draw_site(ax, m)
    _style_map(ax, m, ext, basemap=False)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    handles = [Line2D([], [], color=_corridor_color(result, t.corridor), lw=3,
                      label=f"{t.corridor.rank or 'A'}  {t.corridor.label}  ({t.visits:,.0f}/day)")
               for t in trips.corridors]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, framealpha=0.93, facecolor="white",
              edgecolor=GRID, title="Corridor (site visits/day)", title_fontsize=8.5).set_zorder(50)
    _flush_labels(ax)
    fig.text(0.012, 0.982, "Potential site trips by approach road", fontsize=14, fontweight="bold", color=INK, va="center")
    fig.text(0.012, 0.963, "Arrow width is proportional to vehicles/day entering the site from that approach "
                           "(pass-by + diverted-link).", fontsize=9, color=INK2, va="center")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


# ============================================================================ #
# 2. One satellite map per top-5 corridor
# ============================================================================ #
def _fit_extent(ext, aspect: float):
    """Grow (x0, x1, y0, y1) to a width/height ratio so the imagery fills the axes."""
    cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
    w, h = ext[1] - ext[0], ext[3] - ext[2]
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


def plot_corridor_satellite(result, ct, path: str, basemap: Optional[bool] = None):
    """Satellite map of one corridor: the road, its approach arrows, the site and a figures card."""
    c = ct.corridor
    m = _Map(result)
    col = HIGHLIGHT_BLUE
    if basemap is None:
        basemap = bool(mapbox_token())
    BASEMAP_STATUS.setdefault("satellite", {"style": _style("satellite"),
                                            "status": "pending" if basemap else "not_requested", "error": None})
    fig_w, fig_h, map_h = 11.0, 5.0, 0.89
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, map_h])

    pts = [m.site[None, :], m.xy(c.access_xy)]
    for a in c.approaches:                                   # the road near the site, not its whole length
        L = float(np.linalg.norm(np.diff(a.path_xy, axis=0), axis=1).sum())
        pts.append(m.xy(np.array([walk_along(a.path_xy, d) for d in np.linspace(max(L - 700, 0), L, 12)])))
    ext = _fit_extent(_extent_from_points(np.vstack(pts), 0.15, min_span=700), fig_w / (fig_h * map_h))
    span = ext[1] - ext[0]
    drawn = basemap and mapbox_token() and _add_mapbox_basemap(ax, ext, "satellite")
    if not drawn:
        ax.set_facecolor("#2b2f33")
        _draw_context_roads(ax, m, color="#6b7075", lw=0.8)
        ax.text(0.5, 0.5, "Satellite imagery unavailable" + ("" if mapbox_token() else " (no MAPBOX_TOKEN)"),
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#9aa0a6", zorder=2)

    _draw_all_segments(ax, m, color="#f3f4f6" if drawn else ALL_SEGMENTS_GRAY, lw=1.0, alpha=0.55)
    for p in c.parts_xy:
        xy = m.xy(p)
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=3.4, alpha=0.95, solid_capstyle="round", zorder=8,
                path_effects=[pe.withStroke(linewidth=6.0, foreground="white", alpha=0.9)])
    for a in c.approaches:
        v = ct.visits * a.flow_share
        if v > 0:
            _draw_approach_arrow(ax, m, a, c, v, v / result.trips.visits if result.trips.visits else 0, col,
                                 lw=1.8 + 3.0 * a.flow_share, arrow_len_m=min(a.length_m, span * 0.14),
                                 offset=span * 0.012, label=False)
    _draw_connector(ax, m, c, col, show_label=False)
    a_xy = m.xy(c.access_xy)[0]
    ax.scatter([a_xy[0]], [a_xy[1]], s=46, color="white", edgecolor=col, lw=2, zorder=29)
    _draw_site(ax, m)
    _style_map(ax, m, ext, basemap=False, light_ink=bool(drawn))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    _flush_labels(ax)

    rank = f"#{c.rank}" if c.rank else "Site access road"
    fig.text(0.012, 0.96, f"{rank}  {c.label}", fontsize=14, fontweight="bold", color=INK, va="center")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


# ============================================================================ #
# 3. Overview: every traffic corridor in the drive-time area, top 5 highlighted with name tags
# ============================================================================ #
def plot_corridor_overview(result, path: str, basemap: Optional[bool] = None):
    m = _Map(result)
    if basemap is None:
        basemap = bool(mapbox_token())
    BASEMAP_STATUS.setdefault("overview", {"style": _style("trip_map"),
                                           "status": "pending" if basemap else "not_requested", "error": None})
    fig_w, fig_h, map_h = 9.5, 10.6, 0.945
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, map_h])
    poly = np.vstack([m.ll(r) for r in result.polygon_rings])
    ext = _fit_extent(_extent_from_points(poly, 0.03), fig_w / (fig_h * map_h))
    drawn = bool(basemap and mapbox_token() and _add_mapbox_basemap(ax, ext, "overview"))
    for ring in result.polygon_rings:
        xy = m.ll(ring)
        ax.plot(xy[:, 0], xy[:, 1], color="#2a78d6", lw=1.2, ls=(0, (5, 3)), alpha=0.8, zorder=2)
    top = [c for c in result.corridors if c.rank and c.selected]
    top_ids = {c.corridor_id for c in top}
    _draw_all_segments(ax, m, lw=1.6, alpha=0.95, zorder=5)     # every fetched segment, light gray
    for c in top:
        col = _corridor_color(result, c)
        for p in c.parts_xy:
            xy = m.xy(p)
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=4.2, alpha=0.95, solid_capstyle="round", zorder=8,
                    path_effects=[pe.withStroke(linewidth=7.0, foreground="white", alpha=0.9)])
    placed = []
    span = ext[1] - ext[0]
    for c in top:                                             # name tag on the longest visible stretch
        best = (0.0, None, 0.0)
        for p in c.parts_xy:
            xy = m.xy(p)
            for a, b in zip(xy[:-1], xy[1:]):
                mid = (a + b) / 2
                if not (ext[0] + 0.05 * span < mid[0] < ext[1] - 0.05 * span and
                        ext[2] + 0.05 * span < mid[1] < ext[3] - 0.05 * span):
                    continue
                if any(np.linalg.norm(mid - q) < 0.09 * span for q in placed):
                    continue
                L = float(np.linalg.norm(b - a))
                if L > best[0]:
                    best = (L, mid, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
        if best[1] is None:
            continue
        placed.append(best[1])
        ang = best[2] - 180 if best[2] > 90 else best[2] + 180 if best[2] < -90 else best[2]
        ax.text(best[1][0], best[1][1], c.label, rotation=ang, rotation_mode="anchor", ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK, zorder=30,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_corridor_color(result, c), lw=2.0, alpha=0.96))
    _draw_site(ax, m, label=False)
    _style_map(ax, m, ext, basemap=False)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    if not drawn:
        _draw_context_roads(ax, m, color=CONTEXT_ROAD, lw=0.7, zorder=3)
    ttl = result.payload.drive_time_minutes
    fig.text(0.012, 0.982, f"Traffic corridors within the {ttl:g}-minute drive-time area" if ttl
             else "Traffic corridors within the drive-time area", fontsize=14, fontweight="bold", color=INK, va="center")
    fig.text(0.012, 0.963, f"All {len(result.segments)} fetched road segments in light gray; "
                           f"the top {len(top)} corridors are highlighted.", fontsize=9, color=INK2, va="center")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def render_all(result, out_dir: str, basemap: Optional[bool] = None) -> Dict[str, str]:
    """corridor_overview.png, trip_distribution.png and satellite_<rank>.png for each top-5 corridor.
    basemap=None -> use Mapbox whenever MAPBOX_TOKEN is set."""
    os.makedirs(out_dir, exist_ok=True)
    if basemap is None:
        basemap = bool(mapbox_token())
    BASEMAP_STATUS.clear()
    for kind in ("trip_map", "satellite", "overview"):
        BASEMAP_STATUS[kind] = {"style": _style(kind), "error": None,
                                "status": "pending" if basemap else "not_requested"}
    out = {"corridor_overview": plot_corridor_overview(result, os.path.join(out_dir, "corridor_overview.png"), basemap),
           "trip_distribution": plot_trip_distribution(result, os.path.join(out_dir, "trip_distribution.png"), basemap)}
    for ct in result.trips.corridors:
        if ct.corridor.rank:                                  # top-K only; the context-only access road is excluded
            out[f"satellite_{ct.corridor.rank:02d}"] = plot_corridor_satellite(
                result, ct, os.path.join(out_dir, f"satellite_{ct.corridor.rank:02d}_{_slug(ct.corridor.label)}.png"),
                basemap)
    for kind, st in BASEMAP_STATUS.items():                    # untouched "pending" -> nothing needed the basemap
        if st["status"] == "pending":
            st["status"] = "ok" if basemap and mapbox_token() else "not_requested"
    return out
