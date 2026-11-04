"""
Where does a driver leave each road to reach the site?

All fetched road segments (including low-AADT local streets) form a small road network. One Dijkstra run
from the site gives the shortest route *distance* from every network node to the site. For a counted corridor
the *exit point* is its network node with the shortest route to the site, i.e. the intersection where a driver
would leave that road, and the route from there to the site is drawn along the real streets. Drive time is
the time along that shortest-distance route (road-class speeds), so the detour follows the geometry we have;
a routing service can replace this later.

If the site cannot be connected to the network, callers fall back to the straight-line estimate.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import TripModelConfig
from .geo import nearest_point_on_parts


@dataclass
class Route:
    exit_xy: np.ndarray          # intersection where the driver leaves the corridor (local metres)
    minutes: float               # one-way drive time exit -> site, along the shortest route
    meters: float                # length of that route
    path_xy: np.ndarray          # polyline exit -> site along the network


def _cum(p: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def _s_of(part: np.ndarray, pt: np.ndarray) -> float:
    near, d, _pi, si, t = nearest_point_on_parts([part], pt)
    c = _cum(part)
    return float(c[si] + t * (c[si + 1] - c[si]))


def _slice(part: np.ndarray, s0: float, s1: float, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    c = _cum(part)
    inner = part[(c > s0 + 1e-6) & (c < s1 - 1e-6)]
    return np.vstack([p0[None, :], inner, p1[None, :]])


class RoadNetwork:
    def __init__(self, seg: pd.DataFrame, cfg: TripModelConfig, snap_m: float = 15.0,
                 max_connector_m: float = 400.0):
        self.cfg, self.snap, self.max_connector_m = cfg, snap_m, max_connector_m
        self.centers: List[np.ndarray] = []
        self.edges: List[Tuple[int, int, np.ndarray, float, object]] = []   # (a, b, geom a->b, minutes, seg index)
        self.adj: Dict[int, List[int]] = {}
        self.node_segs: Dict[int, Set[object]] = {}
        parts = []
        for idx, r in seg.iterrows():
            speed = r.get("speed_kph")
            if speed is None or pd.isna(speed):
                speed = cfg.class_default_speed_kph.get(r["road_class_norm"], 40.0)
            for p in r["parts_xy"]:
                p = np.asarray(p, float)
                if len(p) >= 2 and _cum(p)[-1] > 1.0:
                    parts.append((idx, p, max(float(speed), 15.0)))
        self._build(parts)

    # -- graph construction ------------------------------------------------------------------------------
    def _node(self, pt: np.ndarray) -> int:
        if self.centers:
            d = np.linalg.norm(np.asarray(self.centers) - pt, axis=1)
            k = int(np.argmin(d))
            if d[k] <= self.snap:
                return k
        self.centers.append(np.asarray(pt, float))
        return len(self.centers) - 1

    def _build(self, parts):
        marks: List[List[np.ndarray]] = [[p[0], p[-1]] for _i, p, _s in parts]
        # T-junctions: an endpoint that touches the interior of another road splits that road there
        for a, (_ia, pa, _sa) in enumerate(parts):
            for end in (pa[0], pa[-1]):
                for b, (_ib, pb, _sb) in enumerate(parts):
                    if a == b:
                        continue
                    near, d, *_ = nearest_point_on_parts([pb], end)
                    if d <= self.snap and np.linalg.norm(near - pb[0]) > self.snap and np.linalg.norm(near - pb[-1]) > self.snap:
                        marks[b].append(near)
        for (idx, p, speed), pts in zip(parts, marks):
            ordered = sorted({(round(_s_of(p, q), 1), self._node(q)) for q in pts})
            dedup: List[Tuple[float, int]] = []
            for s, k in ordered:
                if not dedup or dedup[-1][1] != k:
                    dedup.append((s, k))
            for (s0, k0), (s1, k1) in zip(dedup[:-1], dedup[1:]):
                if k0 == k1 or s1 - s0 < 1.0:
                    continue
                geom = _slice(p, s0, s1, self.centers[k0], self.centers[k1])
                self._add_edge(k0, k1, geom, (s1 - s0) / (speed * 1000 / 60), idx)

    def _add_edge(self, a, b, geom, minutes, idx):
        self.edges.append((a, b, geom, minutes, idx))
        e = len(self.edges) - 1
        self.adj.setdefault(a, []).append(e)
        self.adj.setdefault(b, []).append(e)
        for n in (a, b):
            self.node_segs.setdefault(n, set()).add(idx)

    # -- site -> everywhere ------------------------------------------------------------------------------
    def connect_site(self, site_xy: np.ndarray) -> bool:
        """Attach the site to the nearest road point and run Dijkstra (on route length) from it."""
        best = (math.inf, None, None, 0.0)
        for e, (a, b, geom, minutes, _idx) in enumerate(self.edges):
            near, d, _pi, si, t = nearest_point_on_parts([geom], site_xy)
            if d < best[0]:
                best = (d, e, near, _s_of(geom, near))
        if best[1] is None or best[0] > self.max_connector_m:
            self.dist = None
            return False
        d, e, near, s = best
        a, b, geom, minutes, _idx = self.edges[e]
        L = _cum(geom)[-1]
        conn_min = d / (self.cfg.connector_speed_kph * 1000 / 60)
        self.site_node = -1
        self.site_xy = np.asarray(site_xy, float)
        self.site_links = [       # (node, polyline node -> site, minutes)
            (a, np.vstack([_slice(geom, 0, s, geom[0], near), site_xy[None, :]]), minutes * s / L + conn_min),
            (b, np.vstack([_slice(geom, s, L, near, geom[-1])[::-1], site_xy[None, :]]),
             minutes * (L - s) / L + conn_min),
        ]
        dist: Dict[int, float] = {}                         # metres along the route
        mins: Dict[int, float] = {}                         # minutes along the same route
        prev: Dict[int, Tuple[int, object]] = {}
        pq: List[Tuple[float, int]] = []
        for node, geom_link, t in self.site_links:          # geom_link runs node -> site
            length = float(_cum(geom_link)[-1])
            if length < dist.get(node, math.inf):
                dist[node], mins[node] = length, t
                prev[node] = (self.site_node, geom_link)
                heapq.heappush(pq, (length, node))
        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > dist.get(u, math.inf):
                continue
            for ei in self.adj.get(u, []):
                a2, b2, g2, m2, _ = self.edges[ei]
                v = b2 if u == a2 else a2
                nd = d_u + float(_cum(g2)[-1])
                if nd < dist.get(v, math.inf):
                    dist[v], mins[v] = nd, mins[u] + m2
                    prev[v] = (u, g2[::-1] if u == a2 else g2)       # geometry runs v -> u
                    heapq.heappush(pq, (nd, v))
        self.dist, self.mins, self.prev = dist, mins, prev
        return True

    def route_for(self, seg_index) -> Optional[Route]:
        """Best exit of the road segment `seg_index`: its node with the shortest route to the site."""
        if getattr(self, "dist", None) is None:
            return None
        nodes = [n for n, segs in self.node_segs.items() if seg_index in segs and n in self.dist]
        if not nodes:
            return None
        n = min(nodes, key=lambda k: self.dist[k])
        path = [self.centers[n]]
        cur = n
        while cur != self.site_node:
            nxt, geom = self.prev[cur]
            path.append(np.asarray(geom)[1:])               # geom runs cur -> nxt; drop the repeated start
            cur = nxt
        pts = np.vstack([path[0][None, :]] + [p for p in path[1:]])
        return Route(self.centers[n].copy(), float(self.mins[n]), float(self.dist[n]), pts)
