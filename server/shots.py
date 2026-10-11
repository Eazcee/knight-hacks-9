"""Street view: a photo every few meters of walking, with where you stood and which way you
faced. Step through them on the dashboard like Street View; the mini-map shows where each
one was taken. Nothing 3D: just pictures, pinned along the path.
"""
import json
import os
import time

import cv2

SHOT_M = 3.0          # one photo per this many meters walked
SHOT_W = 480          # stored width (portrait frames: 480x853, ~40 KB)
MAX_SHOTS = 3000


class Shots:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.index = os.path.join(path, "index.json")
        self.items = []
        self.rev = 0
        self.dist = 0.0           # meters walked since the last shot
        self.last_heading = None
        try:
            with open(self.index, encoding="utf-8") as f:
                self.items = json.load(f)
        except Exception:
            self.items = []

    def _save(self):
        tmp = self.index + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.items, f)
        os.replace(tmp, self.index)
        self.rev += 1

    def walked(self, meters):
        self.dist += meters

    def due(self, heading, force=False):
        """Time for a photo? After SHOT_M meters, or right after a turn (the new view), or on demand."""
        if force or self.dist >= SHOT_M:
            return True
        return self.last_heading is not None and heading is not None and abs((heading - self.last_heading + 540) % 360 - 180) >= 40

    def add(self, img_bgr, x, y, lat, lng, heading, ts, node_id=None, people=(), quality=None):
        sid = int(ts * 1000)
        h, w = img_bgr.shape[:2]
        if w > SHOT_W:
            img_bgr = cv2.resize(img_bgr, (SHOT_W, int(h * SHOT_W / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        with open(os.path.join(self.path, f"{sid}.jpg"), "wb") as f:
            f.write(buf.tobytes())
        item = {"id": sid, "ts": ts, "x": round(float(x), 2), "y": round(float(y), 2), "lat": lat, "lng": lng,
                "heading": None if heading is None else round(float(heading)), "node": node_id,
                "people": list(people)[:6], "q": quality}
        self.items.append(item)
        if len(self.items) > MAX_SHOTS:
            old = self.items.pop(0)
            self._rm(old["id"])
        self.dist = 0.0
        self.last_heading = heading
        self._save()
        return item

    def _rm(self, sid):
        try:
            os.remove(os.path.join(self.path, f"{sid}.jpg"))
        except OSError:
            pass

    def file(self, sid):
        p = os.path.join(self.path, f"{int(sid)}.jpg")
        return p if os.path.exists(p) else None

    def clear(self):
        for it in self.items:
            self._rm(it["id"])
        self.items = []
        self.dist, self.last_heading = 0.0, None
        self._save()

    def rotate_since(self, since_ts, pivot, deg, to_latlng):
        """A turn was recognised late: swing the photos taken since around the corner, onto the new heading."""
        import math
        a = math.radians(deg)
        px, py = pivot
        changed = False
        for it in self.items:
            if it["ts"] < since_ts:
                continue
            dx, dy = it["x"] - px, it["y"] - py
            # heading clockwise from north: rotating the walk by +deg clockwise
            nx = px + dx * math.cos(a) + dy * math.sin(a)
            ny = py - dx * math.sin(a) + dy * math.cos(a)
            it["x"], it["y"] = round(nx, 2), round(ny, 2)
            it["lat"], it["lng"] = to_latlng(nx, ny)
            if it.get("heading") is not None:
                it["heading"] = round((it["heading"] + deg) % 360)
            changed = True
        if changed:
            self.last_heading = None if self.last_heading is None else (self.last_heading + deg) % 360
            self._save()
        return changed

    def near(self, x, y, max_m=6.0):
        best, bd = None, max_m
        for it in self.items:
            d = ((it["x"] - x) ** 2 + (it["y"] - y) ** 2) ** 0.5
            if d < bd:
                best, bd = it, d
        return best
