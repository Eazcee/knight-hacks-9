"""Geo map: where you are, where you walked, and how to get back somewhere.

Position comes from the phone's GPS (lat/lng/accuracy sent with each frame). When
the fix is poor (indoors) the server dead-reckons from step counts along the
heading. The walk graph builds itself: a node every ~8 m along the path you
actually walked, edges carry cost = distance walked x (1 + crowd), and an edge
can be blocked (dashboard tap, voice "blocked", or a printed QR sign). Routes
are A* over that graph; when there's no walked path to the goal yet, "compass
mode" points straight at it. Directions are relative to your heading (glasses
IMU / phone compass, or the direction you're walking from GPS).

Coordinates: lat/lng everywhere; x/y are local meters (east, north) from the first fix.
"""
import heapq
import math
import os
import time
from collections import deque

import cv2

NODE_SPACING_M = float(os.environ.get("BEACON_NODE_M", 8.0))   # a new place every ~8 m walked
SNAP_M = 6.0                 # within this of a place = you're at it
GOOD_ACC_M = float(os.environ.get("BEACON_GOOD_ACC", 15.0))     # GPS accuracy (m) we trust to build the map
STEP_M = 0.72                # meters per step (dead reckoning)
CROWD_SCALE = 4.0            # faces per frame that count as "crowded" (cost x2)
ARRIVE_M = 6.0               # within this of a waypoint = reached it
OFFGRAPH_M = 40.0            # start/goal connect to the walked map if a place is this close
JUMP_M = 15.0                # a per-frame move bigger than this is a GPS glitch, not walking
MOVE_WINDOW_S = 4.0          # "am I moving" window (without step counts)
MOVE_MIN_M = 2.5
POS_ALPHA = 0.6              # smoothing of GPS fixes
WALK_MPS = 1.2               # for ETAs
SENSOR_HEADING_TTL = 3.0     # after this long without a sensor heading, use the GPS course


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lng1, lat2, lng2):
    """Compass bearing from point 1 to point 2 (0 = north, clockwise)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_latlng(lat, lng, dist_m, bearing):
    """Point dist_m away from (lat,lng) in direction bearing."""
    dy = dist_m * math.cos(math.radians(bearing))
    dx = dist_m * math.sin(math.radians(bearing))
    return lat + dy / 110540.0, lng + dx / (111320.0 * math.cos(math.radians(lat)))


def compass8(bearing):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((bearing % 360) + 22.5) // 45) % 8]


def turn_word(rel):
    """rel = bearing to target minus heading, in -180..180."""
    a = abs(rel)
    if a < 25:
        return "straight"
    if a > 150:
        return "turn around"
    side = "left" if rel < 0 else "right"
    return ("sharp " if a > 110 else "" if a > 50 else "slight ") + side


class GeoMap:
    def __init__(self, db):
        self.db = db
        self.qr = cv2.QRCodeDetector()
        self.lat = self.lng = None
        self.acc = None
        self.fix_ts = None            # last trusted GPS fix
        self.quality = "none"         # none | good | rough | dr
        self.heading = 0.0            # 0 = north, clockwise
        self.heading_known = False
        self.heading_src = None       # sensor | gps
        self.sensor_heading_ts = 0.0
        self.origin = None            # (lat, lng) of the local meters frame
        self.x = self.y = 0.0
        self.current_node = None
        self.dist_since_node = 0.0
        self.faces_since_node = []
        self.away = 0                 # consecutive frames far from every place
        self.last_steps = None
        self.step_ts = 0.0
        self.hist = deque()           # (ts, x, y) recent positions, for "am I moving"
        self.trail = deque(maxlen=800)
        self.course_ref = None
        self.route = None
        self.arrived = None           # {"name", "ts", "person_id", "found"}
        self.last_img = None
        self.last_ts = None
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self):
        nodes = [n for n in self.db.nodes() if n.get("lat") is not None]
        if nodes:
            self.origin = (nodes[0]["lat"], nodes[0]["lng"])
            for n in nodes:   # recompute local meters in case the origin moved
                x, y = self._xy(n["lat"], n["lng"])
                if abs(x - (n["x"] or 0)) > 0.5 or abs(y - (n["y"] or 0)) > 0.5:
                    self.db.update_node(n["id"], x=x, y=y)

    # ---- coordinates ---------------------------------------------------
    def _xy(self, lat, lng):
        if self.origin is None:
            self.origin = (lat, lng)
        lat0, lng0 = self.origin
        return ((lng - lng0) * 111320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 110540.0)

    def _set_position(self, lat, lng, ts):
        self.lat, self.lng = lat, lng
        self.x, self.y = self._xy(lat, lng)
        if not self.trail or haversine_m(self.trail[-1][1], self.trail[-1][2], lat, lng) >= 1.0 \
                or ts - self.trail[-1][0] > 5:
            self.trail.append((ts, lat, lng))

    def project(self, dist_m, rel_deg):
        """Where something dist_m ahead of you (rel_deg right of your heading) is on the map."""
        if self.lat is None:
            return None, None
        if not self.heading_known:
            return self.lat, self.lng
        return offset_latlng(self.lat, self.lng, dist_m, self.heading + rel_deg)

    # ---- main update ---------------------------------------------------
    def observe(self, img_bgr, heading_deg=None, steps=None, lat=None, lng=None, acc=None, ts=None, faces=0):
        ts = ts or time.time()
        events = []
        self.last_img = img_bgr
        if heading_deg is not None:
            self.heading, self.heading_known, self.heading_src = float(heading_deg) % 360, True, "sensor"
            self.sensor_heading_ts = ts

        step_delta = 0
        if steps is not None:
            if self.last_steps is None or steps < self.last_steps:
                self.last_steps = steps
            step_delta = int(steps) - int(self.last_steps)
            self.last_steps = steps
            if step_delta > 0:
                self.step_ts = ts

        moved = 0.0
        trusted = False
        if lat is not None and lng is not None:
            acc = float(acc) if acc is not None else 10.0
            trusted = acc <= GOOD_ACC_M
            if trusted:
                if self.lat is None or self.quality != "good":
                    nlat, nlng = lat, lng
                else:
                    nlat = self.lat + POS_ALPHA * (lat - self.lat)
                    nlng = self.lng + POS_ALPHA * (lng - self.lng)
                d = haversine_m(self.lat, self.lng, nlat, nlng) if self.lat is not None else 0.0
                moved = d if d <= JUMP_M else 0.0
                self._set_position(nlat, nlng, ts)
                self.acc, self.fix_ts, self.quality = acc, ts, "good"
                # heading from the direction of travel, unless a sensor heading is fresh
                if self.course_ref is None:
                    self.course_ref = (nlat, nlng)
                elif haversine_m(self.course_ref[0], self.course_ref[1], nlat, nlng) >= 3.0:
                    course = bearing_deg(self.course_ref[0], self.course_ref[1], nlat, nlng)
                    self.course_ref = (nlat, nlng)
                    if ts - self.sensor_heading_ts > SENSOR_HEADING_TTL:
                        self.heading, self.heading_known, self.heading_src = course, True, "gps"
            elif self.lat is None:
                self._set_position(lat, lng, ts)
                self.acc, self.quality = acc, "rough"
            elif self._dr_live(ts):
                pass   # indoors with steps + a sensor heading: dead reckoning beats a 30 m Wi-Fi guess
            elif self.quality != "good" or ts - (self.fix_ts or 0) > 20:
                # nothing better for a while: drift gently toward the rough fix
                self._set_position(self.lat + 0.2 * (lat - self.lat), self.lng + 0.2 * (lng - self.lng), ts)
                self.acc, self.quality = acc, "rough"

        if not trusted and step_delta > 0 and self.lat is not None and self.heading_known:
            dist = step_delta * STEP_M
            nlat, nlng = offset_latlng(self.lat, self.lng, dist, self.heading)
            self._set_position(nlat, nlng, ts)
            moved = dist
            if self.quality != "dr":
                self.acc = min(self.acc or 10.0, 8.0)   # starting to dead-reckon: trust it about this much
            self.quality = "dr"
            self.acc = (self.acc or 8.0) + dist * 0.05   # drift grows as we walk blind (~5 %)
        elif not trusted and self.quality == "dr" and self._dr_live(ts):
            pass   # standing still indoors: keep the dead-reckoned position

        # movement gate: GPS jitter while standing still must not count as walking
        self.hist.append((ts, self.x, self.y))
        while self.hist and ts - self.hist[0][0] > MOVE_WINDOW_S:
            self.hist.popleft()
        if steps is not None:
            moving = ts - self.step_ts < 3.0
        else:
            moving = math.hypot(self.x - self.hist[0][1], self.y - self.hist[0][2]) >= MOVE_MIN_M
        if moving and self.lat is not None:
            self.dist_since_node += min(moved, 5.0)
        self.faces_since_node.append(faces)
        self.last_ts = ts

        # printed QR signs (optional): "place:Name" names a place here, "blocked" blocks the path ahead
        self.qr_tick = getattr(self, "qr_tick", 0) + 1
        text = ""
        if self.qr_tick % 4 == 0:   # QR signs are held up for seconds; checking every 4th frame is plenty
            try:
                small = img_bgr if img_bgr.shape[1] <= 720 else cv2.resize(img_bgr, (720, int(img_bgr.shape[0] * 720 / img_bgr.shape[1])))
                text, _, _ = self.qr.detectAndDecode(small)
            except cv2.error:
                text = ""
        if text:
            text = text.strip()
            if text.lower().startswith("place:") and self.lat is not None:
                r = self.mark_place(text[6:].strip())
                if r:
                    events.append({"type": "new_place", **r})
            elif text.lower() == "blocked":
                ev = self.block_ahead()
                if ev:
                    events.append(ev)

        # the map builds itself from trusted positions
        if self.lat is not None and self.quality in ("good", "dr"):
            nid, nd = self._nearest_node(self.lat, self.lng)
            if nid is not None and nd <= SNAP_M:
                self.away = 0
                if nid != self.current_node:
                    self._arrive(nid, events)
                else:
                    self.dist_since_node = 0.0
                    self.faces_since_node = []
            elif nid is None or nd >= NODE_SPACING_M:
                self.away += 1   # two frames in a row, so one GPS glitch doesn't make a place
                if self.current_node is None or (self.away >= 2 and moving):
                    self._create_node(None, events)
                    self.away = 0
            else:
                self.away = 0

        # route following
        self._follow(events)
        return self._result(events)

    def _dr_live(self, ts):
        """Dead reckoning is usable: a sensor heading and steps were seen recently."""
        return (self.heading_src == "sensor" and ts - self.sensor_heading_ts < 10
                and self.step_ts and ts - self.step_ts < 30)

    def _nearest_node(self, lat, lng, nodes=None, max_m=None):
        best, bd = None, float("inf")
        for n in (nodes or self.db.nodes()):
            if n.get("lat") is None:
                continue
            d = haversine_m(lat, lng, n["lat"], n["lng"])
            if d < bd:
                best, bd = n["id"], d
        if max_m is not None and bd > max_m:
            return None, bd
        return best, bd

    def _edge_len(self, a_id, b_id):
        a, b = self.db.node(a_id), self.db.node(b_id)
        straight = max(1.0, haversine_m(a["lat"], a["lng"], b["lat"], b["lng"]))
        walked = self.dist_since_node + SNAP_M   # path length counts from the edge of the last place
        return round(max(straight, min(walked, 3.0 * straight)), 1)

    def _crowd(self):
        return (sum(self.faces_since_node) / max(1, len(self.faces_since_node))) / CROWD_SCALE

    def _create_node(self, name, events=None):
        nid = self.db.add_node(name, self.x, self.y, lat=self.lat, lng=self.lng)
        if self.current_node is not None:
            self.db.add_edge(self.current_node, nid, self._edge_len(self.current_node, nid), self._crowd())
        self.current_node = nid
        self.dist_since_node = 0.0
        self.faces_since_node = []
        if events is not None:
            events.append({"type": "new_place", "node_id": nid, "name": self.node_name(nid)})
        return nid

    def _arrive(self, nid, events):
        if self.current_node is not None and self.current_node != nid:
            self.db.add_edge(self.current_node, nid, self._edge_len(self.current_node, nid), self._crowd())
        self.current_node = nid
        self.dist_since_node = 0.0
        self.faces_since_node = []
        self.db.bump_visits(nid)
        events.append({"type": "arrived", "node_id": nid, "name": self.node_name(nid)})

    def node_name(self, nid):
        n = self.db.node(nid) if not isinstance(nid, dict) else nid
        if not n:
            return None
        return n["name"] or f"Place {n['id']}"

    def place_name(self):
        """Only places you named are worth showing on the glasses."""
        if self.current_node is None:
            return None
        n = self.db.node(self.current_node)
        return n["name"] if n and n["name"] else None

    def _result(self, events):
        return {
            "node_id": self.current_node,
            "place": self.place_name(),
            "lat": self.lat, "lng": self.lng, "acc": self.acc, "quality": self.quality,
            "x": round(self.x, 2), "y": round(self.y, 2),
            "heading": round(self.heading, 1) if self.heading_known else None,
            "heading_src": self.heading_src if self.heading_known else None,
            "events": events,
            "next": self.next_step(),
        }

    # ---- labels / edits ------------------------------------------------
    def mark_place(self, name=None):
        """A named place right here (voice: 'mark place X' / 'call this place X')."""
        if self.lat is None:
            return None
        name = (name or "").strip() or None
        nid, nd = self._nearest_node(self.lat, self.lng)
        if nid is not None and nd <= SNAP_M:
            if name:
                self.db.update_node(nid, name=name)
            self.current_node = nid
        else:
            nid = self._create_node(name)
        return {"node_id": nid, "name": self.node_name(nid)}

    def label(self, nid, name):
        self.db.update_node(nid, name=(name or "").strip() or None)

    def block(self, a, b, blocked=True):
        self.db.set_edge(a, b, blocked=1 if blocked else 0)
        if self.route:
            self._replan()

    def block_ahead(self):
        """'Blocked' (voice / QR sign): block the stretch of the route you're about to walk."""
        if not self.route or self.route["mode"] != "graph":
            return None
        wps = self.route["waypoints"]
        i = self.route["i"]
        if i >= len(wps) or wps[i].get("node") is None:
            return None
        prev = wps[i - 1]["node"] if i > 0 and wps[i - 1].get("node") is not None else self.route.get("start_node")
        if prev is None or prev == wps[i]["node"]:
            return None
        self.db.set_edge(prev, wps[i]["node"], blocked=1)
        self._replan()
        return {"type": "blocked", "a": prev, "b": wps[i]["node"], "rerouted": bool(self.route and self.route["mode"] == "graph")}

    # ---- routing -------------------------------------------------------
    def graph(self):
        g = {}
        for e in self.db.edges():
            if e["blocked"]:
                continue
            g.setdefault(e["a"], []).append((e["b"], e["dist"] * (1.0 + (e["crowd"] or 0))))
        return g

    def astar(self, start, goal, nodes):
        if start not in nodes or goal not in nodes:
            return None, 0.0
        g = self.graph()

        def h(n):
            return haversine_m(nodes[n]["lat"], nodes[n]["lng"], nodes[goal]["lat"], nodes[goal]["lng"])

        open_set = [(h(start), 0.0, start)]
        came, cost = {}, {start: 0.0}
        while open_set:
            f, c, n = heapq.heappop(open_set)
            if n == goal:
                path = [n]
                while n in came:
                    n = came[n]
                    path.append(n)
                return path[::-1], cost[goal]
            if c > cost.get(n, float("inf")):
                continue
            for m, w in g.get(n, []):
                nc = c + w
                if nc < cost.get(m, float("inf")):
                    cost[m], came[m] = nc, n
                    heapq.heappush(open_set, (nc + h(m), nc, m))
        return None, 0.0

    def plan(self, goal_lat, goal_lng, name=None, person_id=None, photo=None, goal_node=None):
        """Route from here to a point. Uses the walked map when both ends are near it,
        otherwise compass mode (straight-line bearing)."""
        if self.lat is None:
            return {"ok": False, "reason": "no GPS position yet"}
        nodes = {n["id"]: n for n in self.db.nodes() if n.get("lat") is not None}
        start, _ = self._nearest_node(self.lat, self.lng, nodes.values(), OFFGRAPH_M)
        if goal_node not in nodes:
            goal_node, _ = self._nearest_node(goal_lat, goal_lng, nodes.values(), OFFGRAPH_M)
        path, cost = self.astar(start, goal_node, nodes) if (start is not None and goal_node is not None) else (None, 0.0)
        straight = haversine_m(self.lat, self.lng, goal_lat, goal_lng)
        goal = {"lat": goal_lat, "lng": goal_lng, "name": name or (self.node_name(goal_node) if goal_node else "destination"),
                "person_id": person_id, "photo": photo, "node": goal_node}
        if path:
            wps = [{"lat": nodes[n]["lat"], "lng": nodes[n]["lng"], "name": self.node_name(nodes[n]), "node": n} for n in path]
            if len(wps) > 1 and haversine_m(self.lat, self.lng, wps[0]["lat"], wps[0]["lng"]) <= SNAP_M:
                wps = wps[1:]
            gn = nodes[goal_node]
            if haversine_m(goal_lat, goal_lng, gn["lat"], gn["lng"]) > 3.0:
                cost += haversine_m(goal_lat, goal_lng, gn["lat"], gn["lng"])
                wps.append({"lat": goal_lat, "lng": goal_lng, "name": goal["name"], "node": None})
            cost += haversine_m(self.lat, self.lng, wps[0]["lat"], wps[0]["lng"]) if wps and wps[0].get("node") != start else 0.0
            mode = "graph"
        else:
            wps = [{"lat": goal_lat, "lng": goal_lng, "name": goal["name"], "node": None}]
            path, cost, mode = [], straight, "compass"
        self.route = {"mode": mode, "path": path, "waypoints": wps, "i": 0, "goal": goal, "cost_m": round(cost, 1),
                      "start_node": start, "started": time.time()}
        self.arrived = None
        return self.route_public()

    def _replan(self):
        g = self.route["goal"]
        self.plan(g["lat"], g["lng"], name=g["name"], person_id=g["person_id"], photo=g["photo"], goal_node=g["node"])

    def cancel(self):
        self.route, self.arrived = None, None

    def found(self, person_id):
        """The camera recognized the person we're walking to: that's the arrival."""
        if self.route and self.route["goal"].get("person_id") == person_id:
            self.arrived = {"name": self.route["goal"]["name"], "ts": time.time(), "person_id": person_id, "found": True}
            self.route = None
            return True
        a = self.arrived
        if a and a.get("person_id") == person_id and not a.get("found") and time.time() - a["ts"] < 120:
            a.update(found=True, ts=time.time())
            return True
        return False

    def _follow(self, events):
        """Advance waypoints as you reach them."""
        r = self.route
        if not r or self.lat is None:
            return
        wps = r["waypoints"]
        while r["i"] < len(wps):
            w = wps[r["i"]]
            d = haversine_m(self.lat, self.lng, w["lat"], w["lng"])
            passed = False
            if r["i"] + 1 < len(wps):
                nxt = wps[r["i"] + 1]
                leg = haversine_m(w["lat"], w["lng"], nxt["lat"], nxt["lng"])
                passed = haversine_m(self.lat, self.lng, nxt["lat"], nxt["lng"]) < leg - 2.0 and d < 2 * ARRIVE_M
            if d <= ARRIVE_M or passed:
                r["i"] += 1
                if w.get("node") is not None and w["node"] != self.current_node:
                    pass   # the map update above already handles arriving at nodes
            else:
                break
        if r["i"] >= len(wps):
            g = r["goal"]
            self.arrived = {"name": g["name"], "ts": time.time(), "person_id": g.get("person_id"), "found": False}
            self.route = None
            events.append({"type": "arrived_goal", "name": g["name"]})

    def remaining_m(self):
        r = self.route
        if not r or self.lat is None:
            return 0.0
        pts = [(self.lat, self.lng)] + [(w["lat"], w["lng"]) for w in r["waypoints"][r["i"]:]]
        return sum(haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(pts, pts[1:]))

    def next_step(self):
        """The one instruction for the glasses right now."""
        r = self.route
        if not r or self.lat is None or r["i"] >= len(r["waypoints"]):
            return None
        w = r["waypoints"][r["i"]]
        g = r["goal"]
        d = haversine_m(self.lat, self.lng, w["lat"], w["lng"])
        b = bearing_deg(self.lat, self.lng, w["lat"], w["lng"])
        rem = self.remaining_m()
        out = {"to": w.get("node"), "name": w["name"], "goal_name": g["name"], "dist_m": round(d),
               "remaining_m": round(rem), "eta_s": round(rem / WALK_MPS), "bearing": round(b),
               "compass": compass8(b), "mode": r["mode"], "remaining": len(r["waypoints"]) - r["i"],
               "person_id": g.get("person_id"), "photo": g.get("photo")}
        if self.heading_known:
            rel = (b - self.heading + 540) % 360 - 180
            out["rel"] = round(rel)
            out["turn"] = turn_word(rel)
        else:
            out["rel"], out["turn"] = None, None
        return out

    def route_public(self):
        if not self.route:
            return {"ok": False, "reason": "no route"}
        r = self.route
        return {"ok": True, "mode": r["mode"], "path": r["path"], "cost_m": r["cost_m"],
                "remaining_m": round(self.remaining_m()), "waypoints": r["waypoints"][r["i"]:],
                "target": r["goal"], "next": self.next_step()}

    # ---- dashboard -----------------------------------------------------
    def state(self):
        nodes = [{**n, "name": self.node_name(n), "named": bool(n["name"])} for n in self.db.nodes()]
        return {
            "nodes": nodes,
            "edges": self.db.edges(),
            "position": {"lat": self.lat, "lng": self.lng, "acc": self.acc, "quality": self.quality,
                         "heading": self.heading if self.heading_known else None, "heading_src": self.heading_src,
                         "x": self.x, "y": self.y},
            "trail": [[round(t[1], 6), round(t[2], 6)] for t in list(self.trail)[-400:]],
            "current_node": self.current_node,
            "route": self.route_public() if self.route else None,
            "arrived": self.arrived if self.arrived and time.time() - self.arrived["ts"] < 20 else None,
        }
