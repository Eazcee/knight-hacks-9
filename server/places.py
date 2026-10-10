"""Place map: the world builds itself as you walk.

Every few meters the camera snapshots a keyframe and that becomes a "place"
(a node). Consecutive places are linked by edges whose cost is the distance you
actually walked between them (times a crowd factor). When the camera later
recognizes a keyframe it has seen before, you are "back at" that place and your
position snaps to it (loop closure). A* over this graph gives routes with costs.

Position between places comes from dead reckoning: step count from the phone
pedometer (or elapsed time as a fallback) along the heading from the glasses
IMU. It drifts, and loop closures fix the drift.

Printed QR codes are optional anchors:
    place:<name>   -> a guaranteed, named place
    blocked        -> blocks the edge you're about to walk on the active route
"""
import heapq
import math
import os
import time

import cv2
import numpy as np

STEP_M = 0.72            # meters per step
WALK_MPS = 1.0           # assumed speed when we only have time, not steps
# Guessing distance from time alone turns every head turn into "walking" and
# spams places. Only use it for video replays (BEACON_TIME_DR=1). Live, the map
# grows from real step counts, QR anchors, and "mark place" (voice/dashboard).
TIME_DR = os.environ.get("BEACON_TIME_DR", "0") == "1"
NODE_SPACING_M = 6.0     # create a new place after walking this far
MIN_FEATURES = 60        # a keyframe needs texture to be recognizable later
MIN_INLIERS = 15         # ORB inliers needed to say "I'm back here"
THUMB_MATCH = 0.93       # near-identical thumbnail = same place (cheap fallback)
CROWD_SCALE = 4.0        # faces per frame that count as "crowded"


def _thumb(gray):
    t = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 1e-6 else t


class PlaceMap:
    def __init__(self, db, keyframes_dir="data/keyframes"):
        self.db = db
        self.keyframes_dir = keyframes_dir
        os.makedirs(keyframes_dir, exist_ok=True)
        self.orb = cv2.ORB_create(nfeatures=700)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.qr = cv2.QRCodeDetector()
        self.desc = {}          # node_id -> {"thumb", "kp", "des"}
        self.x = self.y = 0.0
        self.heading = 0.0      # degrees, 0 = up on the map, clockwise
        self.heading_known = False
        self.current_node = None
        self.prev_node = None
        self.dist_since_node = 0.0
        self.faces_since_node = []
        self.last_steps = None
        self.last_ts = None
        self.prev_thumb = None
        self.last_keyframe_thumb = None
        self.route = None       # active route as list of node ids
        self.route_goal = None
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self):
        for n in self.db.nodes():
            if n["keyframe"] and os.path.exists(n["keyframe"]):
                img = cv2.imread(n["keyframe"])
                if img is not None:
                    self.desc[n["id"]] = self._describe(img)
        nodes = self.db.nodes()
        if nodes:
            last = nodes[-1]
            self.current_node = last["id"]
            self.x, self.y = last["x"], last["y"]

    def _describe(self, img_bgr):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        if w > 480:
            g = cv2.resize(g, (480, int(h * 480 / w)))
        kp, des = self.orb.detectAndCompute(g, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        return {"thumb": _thumb(g), "pts": pts, "des": des}

    # ---- recognition ---------------------------------------------------
    def recognize(self, d):
        """Match a frame descriptor against all known places. Returns (node_id, score)."""
        if not self.desc:
            return None, 0
        ids = list(self.desc.keys())
        thumbs = np.stack([self.desc[i]["thumb"] for i in ids])
        sims = thumbs @ d["thumb"]
        order = np.argsort(-sims)[:6]
        best_id, best_score = None, 0
        for k in order:
            nid, tsim = ids[k], float(sims[k])
            if tsim < 0.45:
                break
            inl = self._inliers(d, self.desc[nid])
            score = inl + (20 if tsim >= THUMB_MATCH else 0)
            if score > best_score:
                best_id, best_score = nid, score
        if best_score >= MIN_INLIERS:
            return best_id, best_score
        return None, best_score

    def _inliers(self, a, b):
        if a["des"] is None or b["des"] is None or len(a["des"]) < 8 or len(b["des"]) < 8:
            return 0
        good = []
        for m in self.bf.knnMatch(a["des"], b["des"], k=2):
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good.append(m[0])
        if len(good) < 8:
            return len(good) // 2
        src = a["pts"][[m.queryIdx for m in good]]
        dst = b["pts"][[m.trainIdx for m in good]]
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        return int(mask.sum()) if mask is not None else 0

    # ---- main update ---------------------------------------------------
    def observe(self, img_bgr, heading_deg=None, steps=None, ts=None, faces=0):
        ts = ts or time.time()
        events = []
        d = self._describe(img_bgr)

        # 1. dead reckoning
        if heading_deg is not None:
            self.heading = float(heading_deg) % 360
            self.heading_known = True
        if steps is not None:
            if self.last_steps is None:
                self.last_steps = steps
            delta = max(0, int(steps) - int(self.last_steps))
            self.last_steps = steps
            dist = delta * STEP_M
        elif TIME_DR:
            dt = (ts - self.last_ts) if self.last_ts else 0.0
            moving = self.prev_thumb is None or float(self.prev_thumb @ d["thumb"]) < 0.965
            dist = min(dt * WALK_MPS, 3.0) if moving else 0.0
        else:
            dist = 0.0
        self.last_ts = ts
        self.prev_thumb = d["thumb"]
        self.last_img = img_bgr
        self.last_desc = d
        if dist > 0:
            r = math.radians(self.heading)
            self.x += dist * math.sin(r)
            self.y += dist * math.cos(r)
            self.dist_since_node += dist
        self.faces_since_node.append(faces)

        # 2. QR anchors
        try:
            text, _, _ = self.qr.detectAndDecode(img_bgr)
        except cv2.error:
            text = ""
        if text:
            text = text.strip()
            if text.lower().startswith("place:"):
                name = text[6:].strip()
                n = self.db.node_by_qr(name)
                if n:
                    if n["id"] != self.current_node:
                        self._arrive(n["id"], events, how="qr")
                else:
                    self._create_node(name, d, img_bgr, qr=name, events=events)
                return self._result(events, d)
            if text.lower() == "blocked":
                ev = self.block_ahead()
                if ev:
                    events.append(ev)

        # 3. visual place recognition
        nid, score = self.recognize(d)
        if nid is not None:
            if nid != self.current_node:
                self._arrive(nid, events, how="visual", score=score)
            else:
                # still here: pull drifted position back onto the place
                n = self.db.node(nid)
                self.x, self.y = 0.7 * self.x + 0.3 * n["x"], 0.7 * self.y + 0.3 * n["y"]
                self.dist_since_node *= 0.5
            return self._result(events, d)

        # 4. new place?
        far_enough = self.dist_since_node >= NODE_SPACING_M or self.current_node is None
        textured = d["des"] is not None and len(d["des"]) >= MIN_FEATURES
        different = self.last_keyframe_thumb is None or float(self.last_keyframe_thumb @ d["thumb"]) < 0.90
        if far_enough and textured and different:
            self._create_node(None, d, img_bgr, events=events)
        return self._result(events, d)

    def _create_node(self, name, d, img_bgr, qr=None, events=None):
        nid_guess = (max(self.desc.keys()) + 1) if self.desc else 1
        name = name or f"Place {nid_guess}"
        path = os.path.join(self.keyframes_dir, f"{int(time.time()*1000)}.jpg")
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (640, int(h * 640 / w))) if w > 640 else img_bgr
        cv2.imwrite(path, small)
        nid = self.db.add_node(name, self.x, self.y, keyframe=path, qr=qr)
        self.desc[nid] = d
        self.last_keyframe_thumb = d["thumb"]
        if self.current_node is not None:
            self.db.add_edge(self.current_node, nid, max(1.0, self.dist_since_node), self._crowd())
        self.prev_node, self.current_node = self.current_node, nid
        self.dist_since_node = 0.0
        self.faces_since_node = []
        if events is not None:
            events.append({"type": "new_place", "node_id": nid, "name": name})
        self._advance_route(nid)
        return nid

    def _arrive(self, nid, events, how="visual", score=0):
        n = self.db.node(nid)
        if self.current_node is not None and self.current_node != nid:
            self.db.add_edge(self.current_node, nid, max(1.0, self.dist_since_node), self._crowd())
        self.prev_node, self.current_node = self.current_node, nid
        self.x, self.y = n["x"], n["y"]              # loop closure: snap position
        self.dist_since_node = 0.0
        self.faces_since_node = []
        self.last_keyframe_thumb = self.desc[nid]["thumb"]
        self.db.bump_visits(nid)
        events.append({"type": "arrived", "node_id": nid, "name": n["name"], "how": how, "score": score})
        self._advance_route(nid)

    def _crowd(self):
        return (sum(self.faces_since_node) / max(1, len(self.faces_since_node))) / CROWD_SCALE

    def _result(self, events, d):
        cur = self.db.node(self.current_node) if self.current_node else None
        return {
            "node_id": self.current_node,
            "place": cur["name"] if cur else None,
            "x": round(self.x, 2), "y": round(self.y, 2),
            "heading": round(self.heading, 1) if self.heading_known else None,
            "dist_since_place": round(self.dist_since_node, 1),
            "features": 0 if d["des"] is None else int(len(d["des"])),
            "events": events,
            "next": self.next_step(),
        }

    # ---- labels / edits ------------------------------------------------
    def mark_place(self, name=None, dist_m=None):
        """Create a place right here from the last camera frame (voice: 'mark place X').
        Without step counts we don't know how far you walked, so the link to the
        previous place gets dist_m (default 5 m) unless dead reckoning measured it."""
        if getattr(self, "last_img", None) is None:
            return None
        if dist_m is not None or self.dist_since_node < 0.5:
            self.dist_since_node = dist_m if dist_m is not None else 5.0
        events = []
        nid = self._create_node(name, self.last_desc, self.last_img, events=events)
        return {"node_id": nid, "name": self.db.node(nid)["name"]}

    def label(self, nid, name):
        self.db.update_node(nid, name=name)

    def block(self, a, b, blocked=True):
        self.db.set_edge(a, b, blocked=1 if blocked else 0)
        if self.route and self.route_goal is not None:
            self.plan(self.route_goal)

    def block_ahead(self):
        """Camera saw a 'blocked' sign: block the edge we're about to take."""
        if not self.route or self.current_node not in self.route:
            return None
        i = self.route.index(self.current_node)
        if i + 1 >= len(self.route):
            return None
        a, b = self.route[i], self.route[i + 1]
        self.block(a, b, True)
        return {"type": "blocked", "a": a, "b": b, "rerouted": bool(self.route)}

    # ---- routing -------------------------------------------------------
    def graph(self):
        g = {}
        for e in self.db.edges():
            if e["blocked"]:
                continue
            g.setdefault(e["a"], []).append((e["b"], e["dist"] * (1.0 + e["crowd"])))
        return g

    def astar(self, start, goal):
        nodes = {n["id"]: n for n in self.db.nodes()}
        if start not in nodes or goal not in nodes:
            return None, 0.0
        g = self.graph()

        def h(n):
            a, b = nodes[n], nodes[goal]
            return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

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

    def plan(self, goal):
        if self.current_node is None:
            return {"ok": False, "reason": "no current place yet"}
        path, total = self.astar(self.current_node, goal)
        if not path:
            self.route, self.route_goal = None, goal
            return {"ok": False, "reason": "no route (blocked?)", "goal": goal}
        self.route, self.route_goal = path, goal
        return {"ok": True, "path": path, "cost_m": round(total, 1), "steps": self.directions(path)}

    def _advance_route(self, nid):
        if self.route and nid in self.route:
            if nid == self.route[-1]:
                self.route, self.route_goal = None, None

    def directions(self, path):
        nodes = {n["id"]: n for n in self.db.nodes()}
        out = []
        prev_bearing = self.heading if self.heading_known else None
        for a, b in zip(path, path[1:]):
            na, nb = nodes[a], nodes[b]
            dx, dy = nb["x"] - na["x"], nb["y"] - na["y"]
            dist = math.hypot(dx, dy)
            bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
            turn = ""
            if prev_bearing is not None and dist > 0.5:
                rel = (bearing - prev_bearing + 540) % 360 - 180
                turn = "straight" if abs(rel) < 30 else ("left" if rel < 0 else "right")
            prev_bearing = bearing if dist > 0.5 else prev_bearing
            out.append({"to": b, "name": nb["name"], "dist_m": round(dist, 1), "turn": turn})
        return out

    def next_step(self):
        if not self.route or self.current_node not in self.route:
            return None
        i = self.route.index(self.current_node)
        if i + 1 >= len(self.route):
            return {"arrived": True}
        d = self.directions(self.route[i:i + 2])[0]
        d["remaining"] = len(self.route) - 1 - i
        return d

    # ---- dashboard -----------------------------------------------------
    def state(self):
        return {
            "nodes": self.db.nodes(),
            "edges": self.db.edges(),
            "position": {"x": self.x, "y": self.y, "heading": self.heading if self.heading_known else None},
            "current_node": self.current_node,
            "route": self.route,
            "route_goal": self.route_goal,
        }
