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

from odometry import TurnTracker

NODE_SPACING_M = float(os.environ.get("BEACON_NODE_M", 8.0))   # a new place every ~8 m walked
SNAP_M = 6.0                 # within this of a place = you're at it
GOOD_ACC_M = float(os.environ.get("BEACON_GOOD_ACC", 15.0))     # GPS accuracy (m) we trust to build the map
DR_BEATS_ACC_M = float(os.environ.get("BEACON_DR_BEATS", 6.0))  # while walking on steps + compass, only a fix this good overrides it
STEP_M = 0.72                # meters per step (dead reckoning)
FLOOR_HPA = float(os.environ.get("BEACON_FLOOR_HPA", 0.40))   # air pressure drop per floor up (~3.3-4 m; ~8.3 m per hPa)
AUTO_INDOOR_S = 15.0         # walking this long without a good GPS fix = indoors (camera turns + steps)
OUTDOOR_FIXES = 3            # this many good fixes (<= 5 m) in a row = outdoors again
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
        self.turns = TurnTracker()
        self.indoors_forced = False   # the Indoors button / voice: GPS ignored until switched off
        self.auto_indoor = False      # no good GPS fix for a while while walking = indoors, by itself
        self.good_fix_ts = 0.0        # last fix good enough to trust over dead reckoning
        self.good_streak = 0          # consecutive fixes <= 5 m (to come back outdoors)
        self.moving_flag = False      # the phone's accelerometer: walking or not (sent as moving=0/1)
        self.moving_ts = 0.0
        self.step_hist = deque(maxlen=40)   # (ts, cumulative steps) for walking speed
        self.gps_speed = 0.0
        self._last_good = None
        self.last_turn = 0.0          # committed camera turn on the last frame (deg), for snapshots
        self.compass_offset = 0.0     # phone compass -> direction of travel; learned outdoors from GPS (a phone in a pocket sits any way)
        self.floor = 1                # which floor you're on (voice / barometer); each floor is its own map layer
        self.floor_hpa = None         # barometer reading at the reference floor (_base_floor)
        self._base_floor = 1
        self._hpa = None              # smoothed pressure
        self.floor_change_ts = 0.0
        self.stairs_ts = 0.0          # last time you actually went up/down (not a correction)
        self.compass_raw = None
        self.last_moved = 0.0         # meters moved on the last frame (for street-view shots)
        self._dr_log = deque(maxlen=900)   # (ts, meters, heading) of every dead-reckoned move
        self._node_ts = {}            # node id -> when it was created (to move a place made mid-turn)
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

    def latlng_of(self, x, y):
        """Local meters -> lat/lng (inverse of _xy)."""
        lat0, lng0 = self.origin if self.origin else (self.lat or 0.0, self.lng or 0.0)
        return lat0 + y / 110540.0, lng0 + x / (111320.0 * math.cos(math.radians(lat0)))

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
    @property
    def indoors(self):
        return self.indoors_forced or self.auto_indoor

    def walking(self, ts):
        """(walking?, speed m/s). The phone's accelerometer flag if it sends one, else the step counter,
        else (no phone sensors at all) moving GPS fixes."""
        steps_speed = None
        if len(self.step_hist) >= 2:
            t0, s0 = next(((t, s) for t, s in self.step_hist if ts - t <= 5.0), self.step_hist[-1])
            t1, s1 = self.step_hist[-1]
            if t1 - t0 >= 1.0:
                steps_speed = (s1 - s0) * STEP_M / (t1 - t0)
        if self.moving_ts and ts - self.moving_ts < 3.0:
            return self.moving_flag, max(0.0, steps_speed or 1.3)
        if self.step_ts:
            return ts - self.step_ts < 1.8, max(0.0, steps_speed or 1.3)
        return self.gps_speed > 0.5 and self.quality == "good", self.gps_speed

    def set_floor(self, floor, events=None, link=True, rebase=True):
        """Change floor. link: you really went up/down (a stairs place joins the floors, the photos are
        the stairwell). rebase: it came from you, not the barometer, so the barometer takes it as the truth."""
        floor = int(floor)
        if rebase and self._hpa is not None:
            self.floor_hpa, self._base_floor = self._hpa, floor
        if floor == self.floor:
            return None
        old = self.floor
        self.floor = floor
        self.floor_change_ts = self.last_ts or time.time()
        if link:
            self.stairs_ts = self.floor_change_ts
        ev = {"type": "floor", "floor": floor, "from": old, "since": self.floor_change_ts - 15}   # the climb itself came before it registered
        # the stairs/elevator: a new place on the new floor, joined to where you left the old one, so routes can cross floors
        if link and self.lat is not None and self.current_node is not None:
            prev = self.current_node
            nid = self.db.add_node(None, self.x, self.y, lat=self.lat, lng=self.lng, floor=floor)
            self._node_ts[nid] = self.floor_change_ts
            self.db.add_edge(prev, nid, 8.0, 0.0)
            self.db.update_node(nid, name=f"Stairs (floor {floor})")
            pn = self.db.node(prev)
            if pn and not pn["name"]:
                self.db.update_node(prev, name=f"Stairs (floor {old})")
            self.current_node = nid
            self.dist_since_node = 0.0
            ev["node_id"] = nid
        if events is not None:
            events.append(ev)
        return ev

    def pressure(self, hpa, ts, events):
        """Barometer: air pressure drops ~0.4 hPa per floor up. floor = reference floor + (reference
        pressure - now) / 0.4, with hysteresis (a landing halfway doesn't flicker) and the reference
        following slow changes (weather) while you stay on a floor. Works on stairs and in elevators,
        several floors at once too. After a break (no readings for a minute) the floor is updated
        without inventing stairs."""
        hpa = float(hpa)
        self.last_ts = ts
        prev_ts = getattr(self, "_hpa_ts", None)
        gap = prev_ts is not None and ts - prev_ts > 60
        self._hpa = hpa if (self._hpa is None or gap) else 0.8 * self._hpa + 0.2 * hpa
        self._hpa_ts = ts
        if self.floor_hpa is None:
            self.floor_hpa, self._base_floor = self._hpa, self.floor
            return
        est = self._base_floor + (self.floor_hpa - self._hpa) / FLOOR_HPA
        if abs(est - self.floor) >= 0.7 and (gap or ts - self.floor_change_ts > 3):
            self.set_floor(int(round(est)), events, link=not gap, rebase=False)
        elif not gap and prev_ts is not None and abs(est - self.floor) < 0.35:
            target = self._hpa + (self.floor - self._base_floor) * FLOOR_HPA     # the reference that makes est == floor
            self.floor_hpa += (target - self.floor_hpa) * min(1.0, (ts - prev_ts) / 300.0)

    def observe(self, img_bgr, heading_deg=None, steps=None, lat=None, lng=None, acc=None, ts=None, faces=0, indoors=None,
                pressure=None, floor=None, moving=None):
        ts = ts or time.time()
        events = []
        self.last_img = img_bgr
        self.last_ts = ts
        if indoors is not None:
            self.indoors_forced = bool(indoors)
        if moving is not None:
            self.moving_flag, self.moving_ts = bool(moving), ts
        if steps is not None:
            self.step_hist.append((ts, int(steps)))
        if floor is not None:
            self.set_floor(floor, events)
        if pressure is not None:
            self.pressure(pressure, ts, events)
        # indoors by itself: walking for a while with no fix we'd trust; outdoors again after a few good ones
        if lat is not None and lng is not None and acc is not None:
            a = float(acc)
            self.good_streak = self.good_streak + 1 if a <= 5.0 else 0
            if a <= DR_BEATS_ACC_M:
                if self._last_good and ts > self._last_good[0]:
                    v = haversine_m(self._last_good[1], self._last_good[2], lat, lng) / (ts - self._last_good[0])
                    self.gps_speed = 0.7 * self.gps_speed + 0.3 * min(v, 4.0)
                self._last_good = (ts, lat, lng)
                self.good_fix_ts = ts
        walking_now, _ = self.walking(ts)
        if not self.auto_indoor and self.lat is not None and walking_now and ts - self.good_fix_ts > AUTO_INDOOR_S:
            self.auto_indoor = True
            events.append({"type": "indoors", "auto": True})
        elif self.auto_indoor and self.good_streak >= OUTDOOR_FIXES:
            self.auto_indoor = False
            events.append({"type": "outdoors", "auto": True})
            if self.floor != 1:
                self.set_floor(1, events, link=False)      # outside = ground floor; re-baseline the barometer
        if heading_deg is not None:
            self.compass_raw = float(heading_deg) % 360
            heading_deg = (self.compass_raw + self.compass_offset) % 360
            self.turns.compass(ts, heading_deg)
            self.sensor_heading_ts = ts
        turn, turn_mid = self.turns.update(img_bgr, ts)
        self.last_turn = turn
        if self.indoors:
            # indoors: the compass only sets the starting direction; from then on the camera's turns move it
            if not self.heading_known:
                self.heading = float(heading_deg) % 360 if heading_deg is not None else 0.0
                self.heading_known, self.heading_src = True, "camera"
            if turn:
                events.append(self._commit_turn(turn, turn_mid, ts))
            if self.lat is not None:
                lat = lng = None          # a phone fix only ever seeds the very first position indoors
            elif lat is not None:
                acc = max(float(acc or 30.0), 30.0)   # ...and counts as rough, never as a map-building fix
        elif heading_deg is not None:
            self.heading, self.heading_known, self.heading_src = float(heading_deg) % 360, True, "sensor"

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
            if trusted and acc > DR_BEATS_ACC_M and self._dr_live(ts) and self.lat is not None:
                trusted = False   # indoors phones claim 10-15 m from Wi-Fi and jump around; steps + compass are steadier
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
                    elif self.compass_raw is not None and acc <= 8:
                        # the phone's compass vs the way we actually moved: learn how the phone is sitting
                        want = (course - self.compass_raw + 540) % 360 - 180
                        self._offset_n = getattr(self, "_offset_n", 0) + 1
                        w = 0.6 if self._offset_n <= 3 else 0.25      # quick to learn, then steady
                        self.compass_offset = (self.compass_offset + w * ((want - self.compass_offset + 540) % 360 - 180)) % 360
                        self.heading = (self.compass_raw + self.compass_offset) % 360
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
            self._dr_log.append((ts, dist, self.heading))
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
        self.last_moved = min(moved, 5.0) if (moving and self.lat is not None) else 0.0
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

    def _commit_turn(self, turn, mid_ts, ts):
        """The camera is sure you turned, a few seconds ago. Turn the heading, and re-walk the steps
        taken since the turn along the new heading (they were credited to the old one meanwhile)."""
        old = self.heading
        redo = [(t, d) for t, d, h in self._dr_log if t >= mid_ts] if mid_ts else []
        back = sum(d for _, d in redo)
        if back and self.lat is not None:
            lat, lng = offset_latlng(self.lat, self.lng, back, old + 180)   # walk back to the corner
            self.lat, self.lng = lat, lng
        pivot = self._xy(self.lat, self.lng) if self.lat is not None else (self.x, self.y)
        self.heading = (old + turn) % 360
        self.heading_src = "camera"
        self.turns.compass_hist.clear()
        if back and self.lat is not None:
            lat, lng = offset_latlng(self.lat, self.lng, back, self.heading)   # ...and on along the new way
            while self.trail and self.trail[-1][0] >= mid_ts:
                self.trail.pop()
            self._set_position(lat, lng, ts)
            for i in range(len(self._dr_log)):
                t, d, h = self._dr_log[i]
                if t >= mid_ts:
                    self._dr_log[i] = (t, d, self.heading)
            nid = self.current_node
            if nid is not None and self._node_ts.get(nid, 0) >= mid_ts:   # a place made mid-turn: it's here now
                self.db.update_node(nid, x=self.x, y=self.y, lat=self.lat, lng=self.lng)
                self.dist_since_node = 0.0
        return {"type": "turn", "deg": turn, "heading": round(self.heading), "since": mid_ts,
                "pivot": [round(pivot[0], 2), round(pivot[1], 2)], "moved_node": self.current_node if back else None}

    def _dr_live(self, ts):
        """Dead reckoning is usable: a heading (compass, or indoors the camera) and steps were seen recently."""
        if not (self.heading_known and self.step_ts and ts - self.step_ts < 30):
            return False
        if self.indoors:
            return True
        return self.heading_src == "sensor" and ts - self.sensor_heading_ts < 10

    def _nearest_node(self, lat, lng, nodes=None, max_m=None, floor=None):
        best, bd = None, float("inf")
        floor = self.floor if floor is None else floor
        for n in (nodes or self.db.nodes()):
            if n.get("lat") is None or (floor is not False and (n.get("floor") or 1) != floor):
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
        nid = self.db.add_node(name, self.x, self.y, lat=self.lat, lng=self.lng, floor=self.floor)
        self._node_ts[nid] = self.last_ts or time.time()
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
            "floor": self.floor,
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
            goal_node, _ = self._nearest_node(goal_lat, goal_lng, nodes.values(), OFFGRAPH_M, floor=False)
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
                         "x": self.x, "y": self.y, "indoors": self.indoors, "indoors_auto": self.auto_indoor and not self.indoors_forced,
                         "compass_offset": round(self.compass_offset), "floor": self.floor, "barometer": self._hpa is not None,
                         "walking": self.walking(self.last_ts or time.time())[0]},
            "trail": [[round(t[1], 6), round(t[2], 6)] for t in list(self.trail)[-400:]],
            "current_node": self.current_node,
            "route": self.route_public() if self.route else None,
            "arrived": self.arrived if self.arrived and time.time() - self.arrived["ts"] < 20 else None,
        }
