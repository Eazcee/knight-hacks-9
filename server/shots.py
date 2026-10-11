"""Street view of everywhere you walked: photos while you walk (and only then), pinned to
where you stood, which way you faced and which floor you were on. No 3D, no edge detection:
just pictures along the path, like Google's Street View, plus the people in them.

Capture rules
  * walking (the phone's accelerometer / step counter says so): one photo every ~0.8 m,
    the sharpest frame since the last one (walking frames blur; this keeps the clear ones)
  * not walking: nothing, however much you look around
  * anyone comes into view (after a minute out of view): a "moment" photo right away, walking
    or not, so the timeline has everyone you saw, every time you saw them
  * every face is kept: recognized people with their id + box (named or "Unknown"), faces too far
    to identify with a box ("far"). Nothing is blurred.

Walks: photos within 25 m and 10 min of the previous one continue the same walk; otherwise
a new walk starts. Walks that cross are joined by the viewer (junction arrows).
"""
import functools
import json
import math
import os
import random
import threading

import cv2
import numpy as np

SPACING_M = 0.8       # target distance between photos while walking
MIN_S, MAX_S = 0.35, 1.2
SHOT_W = 540          # stored width (portrait frames: 540x960, ~50-70 KB)
MAX_SHOTS = 8000
WALK_GAP_S = 600      # a break longer than this starts a new walk
WALK_JUMP_M = 25.0    # ...so does a jump this far from the last photo
MOMENT_GAP_S = 60     # a person counts as "came into view" after this long out of view


def _locked(fn):
    """The pipeline adds photos while the dashboard / backfill reads and rewrites the list: one at a time."""
    @functools.wraps(fn)
    def w(self, *a, **k):
        with self.lock:
            return fn(self, *a, **k)
    return w


def sharpness(img):
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (160, max(16, int(h * 160 / w))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _blur(img, box):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    pw, ph = (x2 - x1) * 0.2, (y2 - y1) * 0.25
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR), (0, 0), 3)


class Shots:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        os.makedirs(path, exist_ok=True)
        self.index = os.path.join(path, "index.jsonl")
        self.items = []
        self.epoch = random.randint(1, 1 << 30)   # changes whenever existing photos change (the dashboard refetches all)
        self.rev = 0
        self.cand = None           # sharpest frame since the last photo: (sharpness, img, people, far, meta)
        self.last_ts = 0.0
        self.in_view = {}          # person id -> last time in view (for "moments")
        self.describe = None       # set by the server: (people, far, W, H) -> (shown, far_boxes, blur_boxes)
        self.last_moment_ts = 0.0
        self.enc_start = {}        # person id -> when they came into view this time (after a minute away)
        self.last_captured = {}    # person id -> time of the last photo they're in
        self.paused = False
        self._load()

    # ---- storage -----------------------------------------------------
    def _load(self):
        items = []
        legacy = os.path.join(self.path, "index.json")
        try:
            if os.path.exists(self.index):
                with open(self.index, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except ValueError:
                                pass
            elif os.path.exists(legacy):
                with open(legacy, encoding="utf-8") as f:
                    items = json.load(f)
        except Exception:
            items = []
        seen, out = set(), []
        for it in items:            # later lines win (edits are appended by rewrite)
            seen.add(it["id"])
        byid = {it["id"]: it for it in items}
        out = sorted(byid.values(), key=lambda it: it["id"])
        for it in out:
            it.setdefault("walk", 1)
            it.setdefault("floor", 1)
            it.setdefault("kind", "walk")
            it["people"] = [p for p in it.get("people", []) if isinstance(p, dict)]
        self.items = out
        if os.path.exists(legacy) and not os.path.exists(self.index):
            self._save_all()

    def _append(self, it):
        with open(self.index, "a", encoding="utf-8") as f:
            f.write(json.dumps(it) + "\n")
        self.rev += 1

    def _save_all(self):
        tmp = self.index + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it) + "\n")
        os.replace(tmp, self.index)
        self.epoch = random.randint(1, 1 << 30)
        self.rev += 1

    def file(self, sid):
        p = os.path.join(self.path, f"{int(sid)}.jpg")
        return p if os.path.exists(p) else None

    def _rm(self, sid):
        try:
            os.remove(os.path.join(self.path, f"{int(sid)}.jpg"))
        except OSError:
            pass

    # ---- capture -----------------------------------------------------
    def offer(self, img, people, far, meta, ts, walking, speed):
        """Every frame while the pipeline runs. Keeps the sharpest frame; commits one every ~0.8 m of walking."""
        if self.paused or not walking:
            self.cand = None
            return None
        s = sharpness(img)
        if self.cand is None or s > self.cand[0]:
            self.cand = (s, img, list(people), list(far), dict(meta))
        interval = min(MAX_S, max(MIN_S, SPACING_M / max(speed or 1.3, 0.3)))
        if ts - self.last_ts < interval:
            return None
        _, cimg, cpeople, cfar, cmeta = self.cand
        self.cand = None
        return self.add(cimg, cpeople, cfar, cmeta, cmeta.get("ts", ts), "walk")

    @_locked
    def add(self, img, people, far, meta, ts, kind):
        H, W = img.shape[:2]
        desc = self.describe(people, far, W, H) if self.describe else ([], [], [])
        shown, far_boxes, blur = desc if len(desc) == 3 else (desc[0], [], desc[1])
        img = img.copy()
        for b in blur:
            _blur(img, b)
        if W > SHOT_W:
            img = cv2.resize(img, (SHOT_W, int(H * SHOT_W / W)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        sid = int(ts * 1000)
        while any(it["id"] == sid for it in self.items[-5:]):
            sid += 1
        with open(os.path.join(self.path, f"{sid}.jpg"), "wb") as f:
            f.write(buf.tobytes())
        last = self.items[-1] if self.items else None
        walk = last["walk"] if last else 0
        far_jump = (last is not None and meta.get("lat") is not None and last.get("lat") is not None
                    and math.hypot(meta["x"] - last["x"], meta["y"] - last["y"]) > WALK_JUMP_M)
        if last is None or ts - last["ts"] > WALK_GAP_S or far_jump:
            walk += 1
        it = {"id": sid, "ts": round(ts, 3), "x": round(float(meta.get("x") or 0.0), 2), "y": round(float(meta.get("y") or 0.0), 2),
              "lat": meta.get("lat"), "lng": meta.get("lng"),
              "heading": None if meta.get("heading") is None else round(float(meta["heading"])),
              "floor": int(meta.get("floor") or 1), "stairs": bool(meta.get("stairs")), "node": meta.get("node"),
              "walk": walk, "kind": kind, "q": meta.get("q"), "people": shown, "far": far_boxes}
        self.items.append(it)
        self._append(it)
        self.last_ts = ts
        for p in shown:
            self.last_captured[p["id"]] = ts
        if len(self.items) > MAX_SHOTS:
            old = self.items.pop(0)
            self._rm(old["id"])
        return it

    @_locked
    def add_seen(self, img, people, meta, ts, src):
        """A photo from the sighting log (a moment you saw someone and no street-view photo has them):
        the camera frame that was saved with the sighting, slotted into the day at its time."""
        H, W = img.shape[:2]
        if W > SHOT_W:
            img = cv2.resize(img, (SHOT_W, int(H * SHOT_W / W)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        sid = int(ts * 1000)
        ids = {it["id"] for it in self.items}
        while sid in ids:
            sid += 1
        with open(os.path.join(self.path, f"{sid}.jpg"), "wb") as f:
            f.write(buf.tobytes())
        it = {"id": sid, "ts": round(ts, 3), "x": round(float(meta.get("x") or 0.0), 2), "y": round(float(meta.get("y") or 0.0), 2),
              "lat": meta.get("lat"), "lng": meta.get("lng"), "heading": None, "floor": 1, "stairs": False,
              "node": meta.get("node"), "walk": 0, "kind": "seen", "q": None, "people": people, "far": [], "src": src}
        self.items.append(it)
        return it

    @_locked
    def finish_backfill(self):
        self.items.sort(key=lambda it: it["id"])
        self._save_all()

    # ---- people changed on the dashboard: keep the photos in step ------
    @_locked
    def rename_person(self, src, dst, name=None, known=None, only=None):
        """`src` turned out to be `dst` (a merge): every box of src now says dst. Returns what changed (for undo)."""
        changed = []
        for it in self.items:
            if only is not None and it["id"] not in only:
                continue
            for p in it["people"]:
                if p["id"] == src and (only is None or p["box"] in only[it["id"]]):
                    changed.append((it["id"], p["box"]))
                    p["id"] = dst
                    if name is not None:
                        p["name"] = name or None
                    if known is not None:
                        p["known"] = bool(known)
        if changed:
            self._save_all()
        return changed

    @_locked
    def drop_person(self, pid):
        """A person was deleted: their boxes come off the photos (the photos stay). Returns them, for undo."""
        removed = []
        for it in self.items:
            keep = []
            for p in it["people"]:
                (removed.append((it["id"], dict(p))) if p["id"] == pid else keep.append(p))
            it["people"] = keep
        if removed:
            self._save_all()
        return removed

    @_locked
    def restore_entries(self, removed):
        byid = {it["id"]: it for it in self.items}
        for sid, p in removed:
            if sid in byid:
                byid[sid]["people"].append(p)
        if removed:
            self._save_all()

    # ---- edits -------------------------------------------------------
    @_locked
    def rotate_since(self, since_ts, pivot, deg, to_latlng):
        """A turn was recognised a moment late: swing the photos taken since around the corner."""
        a = math.radians(deg)
        px, py = pivot
        changed = False
        for it in self.items:
            if it["ts"] < since_ts:
                continue
            dx, dy = it["x"] - px, it["y"] - py
            nx = px + dx * math.cos(a) + dy * math.sin(a)
            ny = py - dx * math.sin(a) + dy * math.cos(a)
            it["x"], it["y"] = round(nx, 2), round(ny, 2)
            it["lat"], it["lng"] = to_latlng(nx, ny)
            if it.get("heading") is not None:
                it["heading"] = round((it["heading"] + deg) % 360)
            changed = True
        if changed:
            self._save_all()
        return changed

    @_locked
    def mark_stairs(self, since_ts):
        """A floor change was confirmed: the photos of the last ~20 s were the stairwell / elevator."""
        changed = False
        for it in reversed(self.items):
            if it["ts"] < since_ts:
                break
            if not it.get("stairs"):
                it["stairs"] = True
                changed = True
        if changed:
            self._save_all()

    @_locked
    def delete(self, sid):
        n = len(self.items)
        self.items = [it for it in self.items if it["id"] != int(sid)]
        if len(self.items) != n:
            self._rm(sid)
            self._save_all()
            return True
        return False

    @_locked
    def clear(self):
        for it in self.items:
            self._rm(it["id"])
        self.items = []
        self.cand = None
        self.last_ts = 0.0
        self.in_view.clear()
        self.enc_start.clear()
        self.last_captured.clear()
        self.last_moment_ts = 0.0
        self._save_all()

    # ---- lookups -----------------------------------------------------
    @_locked
    def since(self, after_id):
        import bisect
        ids = [it["id"] for it in self.items]
        return self.items[bisect.bisect_right(ids, int(after_id)):]

    @_locked
    def at_time(self, ts, max_s=120, person=None):
        best, bd = None, max_s
        for it in self.items:
            if person and not any(p["id"] == person for p in it["people"]):
                continue
            d = abs(it["ts"] - ts)
            if d < bd:
                best, bd = it, d
        return best

    @_locked
    def moments(self, pid):
        return [it for it in self.items if any(p["id"] == pid for p in it["people"])]
