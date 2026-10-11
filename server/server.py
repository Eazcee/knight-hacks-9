"""Beacon server: faces + training sets + notes + sightings + alert watchlist +
self-building map + A* routing.

Run:   uvicorn server:app --host 0.0.0.0 --port 8000
Then:  open http://localhost:8000 for the dashboard.

The phone app (or webcam_client.py) POSTs JPEG frames to /frame and renders the
returned `display` card on the glasses.
"""
import json
import math
import os
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from db import DB
import faces as faces_mod
from faces import FaceEngine
from geo import GeoMap, haversine_m

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BEACON_DATA", "data")
RETENTION_H = float(os.environ.get("BEACON_RETENTION_H", 24))   # forget un-annotated strangers after this
AUTO_ENROLL = os.environ.get("BEACON_AUTO_ENROLL", "1") == "1"
SIGHTING_GAP_S = 15                                              # log a sighting at most this often per person+place
DIRS = {k: os.path.join(DATA, k) for k in ("crops", "frames", "keyframes")}
SETTINGS_PATH = os.path.join(DATA, "settings.json")
DEFAULT_SETTINGS = {"convo_seconds": 30, "convo_enabled": True, "google_maps_key": "", "group_radius_m": 30,
                    "walk_min_per_mile": 16.5, "indoors": False, "blur_strangers": True, "photos_paused": False}
CARD_HOLD_S = 4.0            # keep a person's card on the glasses this long after they leave view (no flicker)
SESSION_GAP_S = 20 * 60      # without GPS, meetings more than 20 min apart are separate groups
CONVO_GAP_S = 8              # a conversation survives looking away for this long
CAM_HFOV_DEG = 75.0          # glasses camera horizontal field of view (approx.)
FACE_W_M = 0.15              # average face width, for distance from face size

os.makedirs(DATA, exist_ok=True)
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)
db = DB(os.path.join(DATA, "beacon.db"))
db.prune_anonymous(RETENTION_H * 3600)
faces = FaceEngine(db, crops_dir=DIRS["crops"], auto_enroll=AUTO_ENROLL)
geo = GeoMap(db)

# depth from the single camera: off by default (BEACON_DEPTH=1: distances to people + "obstacle ahead" reroutes)
depth = None
if os.environ.get("BEACON_DEPTH", "0") == "1":
    try:
        from depth import DepthEngine
        _prov, _ctx = faces_mod._providers()
        depth = DepthEngine(os.path.join(DATA, "models"), _prov, _ctx)
    except Exception as _e:
        print("depth disabled:", _e)
AUTO_BLOCK = os.environ.get("BEACON_AUTO_BLOCK", "1") == "1"
from shots import Shots  # noqa: E402
shots = Shots(os.path.join(DATA, "shots"))
OBSTACLE_COOLDOWN_S = 20
last_cloud = None
last_depth = {}
last_obstacle = {"ts": 0.0, "rerouted": False, "shown": False}


def _load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)


settings = _load_settings()
presence = {}                # person_id -> {"start": ts, "last": ts} continuous time in view

app = FastAPI(title="Beacon")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

QUIET = {"level": "quiet", "title": "Beacon ready", "lines": []}
last_people = []            # people seen in the most recent frame
last_display = dict(QUIET)
last_live = None            # latest frame as JPEG with boxes drawn, for the dashboard
recent_alerts = deque(maxlen=50)
reports = []                # alerts the wearer chose to report (location sent)
last_sighting_log = {}      # (pid, node_id) -> ts
frame_count = 0
dropped = 0
hold = {"until": 0.0, "display": None}
card_hold = {"card": None, "ts": 0.0}
undo_log = {}                # token -> merge snapshot
last_shown = {"card": None, "ts": 0.0}   # last non-empty card, for the dashboard's glasses preview   # keep a scan result on the glasses for a few seconds
last_scan = None
pipeline_lock = threading.Lock()   # one frame at a time; extra frames are dropped, not queued
timing = deque(maxlen=60)          # (face seconds, rest seconds, wall time) of recent frames


# ---------------------------------------------------------------- helpers
def _decode(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return img


async def _image_from_request(request: Request):
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        up = form.get("image")
        if up is None:
            raise HTTPException(400, "multipart field 'image' missing")
        return _decode(await up.read())
    if ctype.startswith("image/") or ctype == "application/octet-stream":
        return _decode(await request.body())
    raise HTTPException(400, "send a JPEG body (Content-Type: image/jpeg) or multipart field 'image'")


def _ago(ts):
    if not ts:
        return None
    s = max(0, time.time() - ts)
    if s < 60:
        return f"{int(s)}s ago"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 48 * 3600:
        return f"{s / 3600:.1f} h ago"
    return f"{int(s // 86400)} days ago"


def _url(path):
    """data/crops/x.jpg (or data\\crops\\x.jpg on Windows) -> /img/crops/x.jpg"""
    if not path:
        return None
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[-2] not in DIRS:
        return None
    return f"/img/{parts[-2]}/{parts[-1]}"


_nodes_cache = {"t": 0.0, "nodes": []}
POI_PATH = os.path.join(DATA, "poi_cache.json")
try:
    with open(POI_PATH, encoding="utf-8") as _f:
        poi_cache = json.load(_f)
except (OSError, ValueError):
    poi_cache = {}
poi_state = {"pending": set(), "status": None, "retry_at": 0.0}
POI_MONTHLY_CAP = int(os.environ.get("BEACON_POI_CAP", 3000))   # stay well under Google's free 5,000/month
SKIP_TYPES = {"route", "street_address", "intersection", "plus_code", "political", "locality", "postal_code"}


def _poi(lat, lng):
    """Name of the building / business you're at (e.g. 'McDonald's', 'Harris Engineering Center'),
    from Google Places Nearby Search with the saved Maps key. Cached per ~11 m cell; looked up in
    the background, so the first call returns None and later calls return the name."""
    key = settings.get("google_maps_key")
    if lat is None or not key:
        return None
    # one lookup per ~45 m square (the search covers 60 m around it), cached on disk forever
    g = 0.0004
    clat, clng = round(lat / g) * g, round(lng / g) * g
    k = f"{clat:.4f},{clng:.4f}"
    if k in poi_cache:
        return poi_cache[k] or None
    month = time.strftime("%Y-%m")
    used = poi_cache.get("_calls", {}).get(month, 0)
    if used >= POI_MONTHLY_CAP:
        poi_state["status"] = f"monthly cap reached ({used} lookups)"
        return None
    if k not in poi_state["pending"] and time.time() >= poi_state["retry_at"] and len(poi_state["pending"]) < 4:
        poi_state["pending"].add(k)
        poi_cache.setdefault("_calls", {})[month] = used + 1
        threading.Thread(target=_poi_fetch, args=(k, clat, clng, key), daemon=True).start()
    return None


def _poi_fetch(k, lat, lng, key):
    import urllib.error
    import urllib.request
    body = json.dumps({"maxResultCount": 8, "rankPreference": "DISTANCE",
                       "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": 60.0}}})
    req = urllib.request.Request("https://places.googleapis.com/v1/places:searchNearby", data=body.encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Goog-Api-Key": key,
                                          "X-Goog-FieldMask": "places.displayName,places.types,places.location"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        name = ""
        for pl in data.get("places", []):
            if set(pl.get("types", [])) & SKIP_TYPES:
                continue
            name = (pl.get("displayName") or {}).get("text", "")
            if name:
                break
        poi_cache[k] = name
        poi_state["status"] = "ok"
        with open(POI_PATH, "w", encoding="utf-8") as f:
            json.dump(poi_cache, f)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="ignore")[:300]
        poi_state["status"] = ("Places API (New) isn't enabled for this key" if e.code == 403 else f"Places error {e.code}") + ": " + msg
        poi_state["retry_at"] = time.time() + 60
    except Exception as e:
        poi_state["status"] = f"Places lookup failed: {e}"
        poi_state["retry_at"] = time.time() + 30
    finally:
        poi_state["pending"].discard(k)


def _place_near(lat, lng, radius=40.0):
    """Name of the closest place you named, within radius meters."""
    if lat is None:
        return None
    best, bd = None, radius
    if time.time() - _nodes_cache["t"] > 2:
        _nodes_cache.update(t=time.time(), nodes=[n for n in db.nodes() if n.get("name") and n.get("lat") is not None])
    for n in _nodes_cache["nodes"]:
        if n.get("name") and n.get("lat") is not None:
            d = haversine_m(lat, lng, n["lat"], n["lng"])
            if d <= bd:
                best, bd = n["name"], d
    return best or _poi(lat, lng)   # a place you named wins; else the building you're at


def _meet(pid, row=None):
    """When and where you first met someone (where you talked, if you did)."""
    row = row or db.person(pid) or {}
    first = db.first_sighting(pid)
    if row.get("met") and row.get("met_lat") is not None:
        lat, lng, ts = row["met_lat"], row["met_lng"], row.get("met_at")
    else:
        loc = db.first_sighting(pid, located=True)
        lat, lng = (loc["lat"], loc["lng"]) if loc else (None, None)
        ts = first["ts"] if first else row.get("created_at")
    place = (first or {}).get("place") or _place_near(lat, lng)
    return {"ts": ts, "ago": _ago(ts), "place": place, "lat": lat, "lng": lng}


def _people_card(people):
    """What the glasses show when people are in view. One person: who they are, your note,
    where you met. Several: their names. Strangers: whether you've seen them before."""
    seen = sorted([p for p in people if p["confirmed"] and not p.get("possible")], key=lambda p: -p.get("px", 0))
    if not seen:
        return None
    named = [p for p in seen if p["name"]]
    others = len(seen) - 1
    if len(named) == 1:
        p = named[0]
        row = db.person(p["person_id"]) or {}
        m = _meet(p["person_id"], row)
        title = p["name"] + (f" · {row['category']}" if row.get("category") else "")
        lines = []
        if p["notes"]:
            lines.append(p["notes"][:70])
        when = ("talked " if row.get("met") else "met ") + (m["ago"] or "")
        lines.append(when + (f" · {m['place']}" if m["place"] else ""))
        if p.get("tier") == "far":
            lines.append(f"{p['dir']} · far away")
        elif others:
            lines.append(f"+{others} other{'s' if others > 1 else ''} in view")
        return {"level": "info", "title": title, "lines": lines[:3]}
    if named:
        lines = [p["name"] + (f" · {p['notes'][:40]}" if p["notes"] else "") for p in named[:3]]
        rest = len(seen) - min(3, len(named))
        if rest and len(lines) < 3:
            lines.append(f"+{rest} more")
        return {"level": "info", "title": f"{len(seen)} people", "lines": lines}
    p = seen[0]
    m = _meet(p["person_id"])
    fresh = p["new"] or not m["ts"] or time.time() - m["ts"] < 60
    lines = [] if fresh else [f"first seen {m['ago']}" + (f" · {m['place']}" if m["place"] else "")]
    lines.append("say 'this is …' to name them")
    if others:
        lines.append(f"{len(seen)} people in view")
    return {"level": "info", "title": "New face" if fresh else "Seen before", "lines": lines}


def _build_display(people, place, alerts, next_step, train):
    if alerts:
        a = alerts[0]
        return {"level": "alert", "title": f"ALERT: {a['watch_label'] or a['name'] or 'watchlist match'}",
                "lines": [f"match {int(a['sim'] * 100)}% · {a.get('dir', 'ahead')} · {place or 'here'}",
                          "say 'report' to send location"]}
    possible = [p for p in people if p.get("possible")]
    if possible and not (train and not train["done"]):
        p = max(possible, key=lambda q: q["sim"])
        lines = [f"{p['dir']} · too far to be sure", "get closer to confirm"]
        if next_step and not next_step.get("arrived"):
            lines.append(f"{(next_step.get('turn') or 'head').upper()} → {next_step['name']} · {next_step['dist_m']} m")
        return {"level": "info", "title": f"Possible: {p['watch_label'] or p['name'] or 'alert match'}", "lines": lines}
    if train and not train["done"]:
        return {"level": "info", "title": f"Training: {train['name'] or train['person_id']}",
                "lines": [f"{train['added']}/{train['goal']} photos · turn your head slowly",
                          f"{train['seconds_left']}s left"]}
    arrived = geo.arrived if geo.arrived and time.time() - geo.arrived["ts"] < 20 else None
    if arrived:
        if arrived.get("found"):
            return {"level": "info", "title": f"FOUND {arrived['name']}", "lines": ["they're right here"]}
        return {"level": "info", "title": f"Arrived · {arrived['name']}",
                "lines": ["look around for them"] if arrived.get("person_id") else []}
    card = _people_card(people)
    if next_step:
        extra = [f"search {len(search['checked'])}/{len(search['buildings'])} · {search['name']}"] if search else []
        return _nav_card(next_step, extra + ([card["title"]] if card else []))
    if search:
        left = len(search["buildings"]) - len(search["checked"])
        return {"level": "info", "title": f"Search · {search['name']}",
                "lines": [f"{left} places left" if left else "every place checked",
                          "say 'next' for the next building" if left else "say 'stop search' to end"]}
    if card:
        return card
    return {"level": "quiet", "title": "", "lines": []}


def _nav_card(n, lines):
    """Turn-by-turn for the glasses: one instruction, the distance, what's left."""
    warn = None
    if time.time() - last_obstacle.get("wall", 0) < 6 and last_obstacle.get("shown"):
        warn = "BLOCKED AHEAD · rerouting" if last_obstacle["rerouted"] else "obstacle ahead · go around"
    if n["turn"]:
        first = f"{n['turn'].upper()} · {n['dist_m']} m"
    else:
        first = f"HEAD {n['compass']} · {n['dist_m']} m · walk so I get your heading"
    if n["mode"] == "compass":
        first += " · direct"
    rest = f"{n['remaining_m']} m left · {max(1, round(n['eta_s'] / 60))} min"
    if n["remaining"] > 1:
        rest += f" · {n['remaining']} turns"
    out = ([warn] if warn else []) + [first, rest] + lines[:1]
    return {"level": "info", "title": f"→ {n['goal_name']}", "lines": out[:3]}


def _person_pos(p, W):
    """Rough map position of a face: distance from its size, angle from where it is in the frame."""
    f = W / (2 * np.tan(np.radians(CAM_HFOV_DEG / 2)))
    dist = float(np.clip(f * FACE_W_M / max(8, p.get("px") or 8), 0.8, 8.0))
    if p.get("dist_m"):
        dist = float(np.clip(p["dist_m"], 0.5, 30.0))   # measured by the depth net
    cx = (p["box"][0] + p["box"][2]) / 2
    rel = float(np.degrees(np.arctan2(cx - W / 2, f)))
    return geo.project(dist, rel)


def _track_conversations(people, ts):
    """Someone who stays in view for convo_seconds (looking away briefly is fine) = you talked."""
    seen = set()
    for p in people:
        if p.get("possible") or not p["confirmed"] or p.get("tier") != "near":
            continue
        pid = p["person_id"]
        seen.add(pid)
        st = presence.get(pid)
        if not st or ts - st["last"] > CONVO_GAP_S:
            st = presence[pid] = {"start": ts, "last": ts}
        st["last"] = ts
        p["in_view_s"] = round(ts - st["start"])
        if settings["convo_enabled"] and ts - st["start"] >= settings["convo_seconds"]:
            row = db.person(pid)
            if row and not row.get("met"):
                db.update_person(pid, met=1, met_at=ts, met_lat=geo.lat, met_lng=geo.lng)
                p["met_now"] = True
    for pid in [k for k, v in presence.items() if ts - v["last"] > 600 and k not in seen]:
        del presence[pid]


def _person_public(p):
    """A person row for the UI, with image URLs and last place."""
    ls = db.last_sighting(p["id"])
    loc = ls if ls and ls.get("lat") is not None else db.last_located_sighting(p["id"])
    meet = _meet(p["id"], p)
    best = p.get("best_crop") or p.get("crop")
    st = presence.get(p["id"])
    return {
        "id": p["id"], "name": p["name"], "notes": p["notes"], "is_watch": bool(p["is_watch"]),
        "watch_label": p["watch_label"], "photo": _url(best), "n_train": p.get("n_train"),
        "n_seen": p.get("n_seen"), "last_seen": p["last_seen"], "last_seen_ago": _ago(p["last_seen"]),
        "last_place": ls["place"] if ls else None,
        "last_seen_place": (ls["place"] if ls and ls.get("place") else None) or (_place_near(loc["lat"], loc["lng"]) if loc else None),
        "lat": loc["lat"] if loc else None, "lng": loc["lng"] if loc else None,
        "met": bool(p.get("met")), "met_at": p.get("met_at"), "category": p.get("category"),
        "met_place": meet["place"], "first_seen_ago": meet["ago"], "first_seen": meet["ts"],
        "meet_lat": meet["lat"], "meet_lng": meet["lng"],
        "in_view_s": round(st["last"] - st["start"]) if st and time.time() - st["last"] < CONVO_GAP_S else 0,
    }


def _sighting_public(s):
    box = json.loads(s["box"]) if s.get("box") else None
    return {"id": s["id"], "person_id": s["person_id"], "person_name": s.get("person_name"),
            "ts": s["ts"], "ago": _ago(s["ts"]), "place": s.get("place"), "sim": s["sim"],
            "quality": s.get("quality"), "photo": _url(s["crop"]), "frame": _url(s.get("frame")), "box": box,
            "lat": s.get("lat"), "lng": s.get("lng")}


# ---------------------------------------------------------------- frames
@app.post("/frame")
async def frame(request: Request, heading: float | None = None, steps: int | None = None,
                lat: float | None = None, lng: float | None = None, acc: float | None = None,
                ts: float | None = None, pressure: float | None = None, floor: int | None = None,
                moving: int | None = None):
    """Main pipeline. Body = JPEG (Content-Type image/jpeg) or multipart 'image'.
    Optional query params: lat, lng, acc (GPS, meters), heading (deg, 0=north, clockwise),
    steps (cumulative pedometer), moving (0/1, the phone's accelerometer: walking right now),
    pressure (hPa, the phone's barometer: floors), floor (explicit), ts."""
    global dropped
    img = await _image_from_request(request)
    ts = ts or time.time()
    if not pipeline_lock.acquire(blocking=False):
        dropped += 1   # still busy with the previous frame: drop this one, keep latency low
        return {"dropped": True, "people": [], "alerts": [], "place": None, "display": last_display}
    try:
        return await run_in_threadpool(_pipeline, img, heading, steps, lat, lng, acc, ts, pressure, floor, moving)
    finally:
        pipeline_lock.release()


def _record(people, img, place, ts, store_scale=1.0, force=False):
    """Log sightings (with the full frame + face box + map position) and collect alerts."""
    node_id = place["node_id"]
    h, w = img.shape[:2]
    alerts = []
    frame_path = None
    for p in people:
        if p.get("possible") or not (p["confirmed"] or p.get("tier") == "near"):
            continue   # "possible" / unconfirmed far matches are never logged as sightings
        key = (p["person_id"], node_id)
        if force or ts - last_sighting_log.get(key, 0) > SIGHTING_GAP_S:
            if frame_path is None:   # one full frame per camera frame, shared by everyone in it
                frame_path = os.path.join(DIRS["frames"], f"f_{int(ts * 1000)}.jpg")
                stored = img if store_scale >= 1 else cv2.resize(
                    img, (int(w * store_scale), int(h * store_scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(frame_path, stored, [cv2.IMWRITE_JPEG_QUALITY, 88])
            crop = faces._save_crop(img, p["box"], f"s_{p['person_id']}_{int(ts * 1000)}")
            box = [int(v * store_scale) for v in p["box"]] + [int(w * store_scale), int(h * store_scale)]
            plat, plng = _person_pos(p, w)
            db.add_sighting(p["person_id"], ts, node_id, geo.x, geo.y, p["sim"], crop,
                            frame=frame_path, box=json.dumps(box), quality=p["quality"], lat=plat, lng=plng)
            last_sighting_log[key] = ts
        if p["is_watch"] and p["confirmed"]:
            a = {**p, "ts": ts, "node_id": node_id, "place": place["place"], "x": geo.x, "y": geo.y,
                 "lat": geo.lat, "lng": geo.lng}
            alerts.append(a)
            recent_alerts.appendleft(a)
    return alerts


def _draw_live(img, people, far, scale=1.0):
    """Frame with labeled boxes for the dashboard. Grey boxes = faces too small/unclear to name."""
    global last_live
    live = img if scale >= 1 else cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                                             interpolation=cv2.INTER_AREA)
    live = live.copy()
    th = max(1, int(round(live.shape[1] / 500)))
    fs = 0.45 * max(1.0, live.shape[1] / 700)

    def bx(b):
        return [int(v * scale) for v in b]

    for f in far:
        x1, y1, x2, y2 = bx(f["box"])
        cv2.rectangle(live, (x1, y1), (x2, y2), (150, 150, 150), 1)
        cv2.putText(live, f"? {f['px']}px", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (170, 170, 170), 1)
    for p in people:
        x1, y1, x2, y2 = bx(p["box"])
        if p.get("possible"):
            color = (0, 165, 255)
        elif p["is_watch"]:
            color = (60, 60, 230)
        elif p["training"]:
            color = (40, 200, 255)
        else:
            color = (90, 200, 90) if p.get("tier", "near") == "near" else (200, 170, 60)
        cv2.rectangle(live, (x1, y1), (x2, y2), color, th + (0 if p["confirmed"] else -1) or 1)
        label = ("possible " if p.get("possible") else "") + (p["name"] or p["watch_label"] or p["person_id"])
        label += f" {int(p['sim'] * 100)}% {p.get('px', '')}px" + (f" {p['dist_m']}m" if p.get("dist_m") else "")
        cv2.putText(live, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)
    ok, buf = cv2.imencode(".jpg", live, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        last_live = buf.tobytes()


def _pipeline(img, heading, steps, lat, lng, acc, ts, pressure=None, floor=None, moving=None):
    global last_people, last_display, frame_count
    frame_count += 1
    t0 = time.perf_counter()
    people = faces.process_frame(img, ts=ts)
    t1 = time.perf_counter()
    obstacle = None
    if depth is not None:
        obstacle = _depth_step(img, people, ts)
    place = geo.observe(img, heading_deg=heading, steps=steps, lat=lat, lng=lng, acc=acc, ts=ts,
                        faces=len(people) + len(faces.last_far), indoors=settings.get("indoors", False),
                        pressure=pressure, floor=floor, moving=moving)
    if obstacle and obstacle["blocked"]:
        _maybe_auto_block(place, ts)
    for ev in place["events"]:
        if ev["type"] == "turn":
            shots.rotate_since(ev["since"], ev["pivot"], ev["deg"], geo.latlng_of)
        elif ev["type"] == "floor" and ev.get("node_id") is not None:
            shots.mark_stairs(ev["since"])
    _maybe_shot(img, people, place, ts)
    _track_conversations(people, ts)
    for p in people:   # walking to someone and the camera sees them: that's the arrival
        if p["confirmed"] and not p.get("possible") and geo.found(p["person_id"]):
            place["next"] = None
        if search and p["person_id"] == search["person_id"] and p["confirmed"] and not p.get("possible"):
            geo.cancel()   # found them: the alert card takes over
            place["next"] = None
    if search and any(e["type"] == "arrived_goal" for e in place["events"]) and search["target"]:
        _search_check(search["target"])
    _search_tick()
    if search and place["next"] is None and geo.route is None and search["target"] is None:
        _search_next()
    place["next"] = geo.next_step()
    alerts = _record(people, img, place, ts)
    _draw_live(img, people, faces.last_far)
    last_people = people
    last_display = _build_display(people, place["place"], alerts, place["next"], faces.train_state())
    if last_display["level"] == "info" and not place["next"] and people:
        card_hold.update(card=last_display, ts=time.time())
    elif last_display["level"] == "quiet" and card_hold["card"] and time.time() - card_hold["ts"] < CARD_HOLD_S:
        last_display = card_hold["card"]   # they just looked away: don't blink the card off and on
    if not alerts and time.time() < hold["until"] and hold["display"]:
        last_display = hold["display"]   # a scan result stays up for a few seconds
    if last_display["level"] != "quiet":
        last_shown.update(card=last_display, ts=time.time())
    t2 = time.perf_counter()
    timing.append((t1 - t0, t2 - t1, time.time()))
    return {"people": people, "place": place, "alerts": alerts, "display": last_display,
            "far_unknown": len(faces.last_far), "nav": place["next"], "ms": round((t2 - t0) * 1000)}


def _timing():
    """How fast the server keeps up: per-frame ms, and frames/s actually processed."""
    rows = list(timing)
    if not rows:
        return {"device": faces.device}
    face_ms = sorted(r[0] * 1000 for r in rows)
    total = sorted((r[0] + r[1]) * 1000 for r in rows)
    span = rows[-1][2] - rows[0][2]
    return {"device": faces.device, "faces_ms": round(face_ms[len(face_ms) // 2]),
            "frame_ms": round(total[len(total) // 2]), "frame_ms_p90": round(total[int(len(total) * 0.9) - 1]) if len(total) >= 10 else None,
            "fps": round((len(rows) - 1) / span, 1) if span > 0 else None, "dropped": dropped}


# ---------------------------------------------------------------- scan (12 MP photo)
def _depth_step(img, people, ts):
    """Depth map -> real distances to people, obstacle check, point cloud for the dashboard."""
    global last_cloud, last_depth
    H, W = img.shape[:2]
    try:
        rel, color = depth.infer(img)
    except Exception as e:
        last_depth = {"error": str(e)[:120]}
        return None
    near_faces = [p for p in people if p.get("tier", "near") == "near"]
    if not depth.calibrate(rel, near_faces, W, H) or not near_faces:
        depth.calibrate_floor(rel)
    dm = depth.metric(rel)
    faces_3d = []
    for p in people:
        d = depth.distance_to(rel, p["box"], W, H)
        if d is not None:
            p["dist_m"] = round(d, 1)
            faces_3d.append({"name": p["name"] or p["person_id"], "xyz": depth.face_3d(p["box"], d, W, H),
                             "dist_m": round(d, 1), "is_watch": p["is_watch"], "met": bool((db.person(p["person_id"]) or {}).get("met"))})
    ob = depth.obstacle(dm, people + faces.last_far, W, H)
    last_cloud = depth.cloud(dm, color, W, H, ts, faces_3d)
    last_depth = {"device": depth.device, "ms": round(depth.ms), "scale": None if depth.scale is None else round(depth.scale, 1),
                  "calibrated": depth.scale is not None, "scale_src": depth.scale_src, "obstacle": ob, "faces": faces_3d, "ts": ts,
                  "center_m": round(float(np.median(dm[dm.shape[0] // 3: 2 * dm.shape[0] // 3, dm.shape[1] // 3: 2 * dm.shape[1] // 3])), 1)}
    return ob


def _shot_people(people, far, W, H):
    """Who goes on a street-view photo: known people (named, talked to, on the watchlist) get a box and
    their name; everyone else's face is blurred in the stored photo (setting blur_strangers)."""
    shown, blur = [], []
    for p in people:
        x1, y1, x2, y2 = p["box"][:4]
        rec = db.person(p["person_id"]) or {}
        known = p["confirmed"] and (p["name"] or p["is_watch"] or rec.get("met"))
        if known or (p["is_watch"] and p.get("possible")):
            shown.append({"id": p["person_id"], "name": p["name"] or p["watch_label"] or None,
                          "box": [round(x1 / W, 4), round(y1 / H, 4), round(x2 / W, 4), round(y2 / H, 4)],
                          "watch": bool(p["is_watch"]), "met": bool(rec.get("met")), "possible": bool(p.get("possible"))})
        else:
            blur.append(p["box"])
    blur += [f["box"] for f in far]
    return shown, (blur if settings.get("blur_strangers", True) else [])


shots.describe = _shot_people


def _maybe_shot(img, people, place, ts, force=False):
    """Street view: photos only while walking (~one per 0.8 m, the sharpest frame of each stretch); a
    'moment' photo whenever a known person comes into view; on demand (button / voice 'photo')."""
    shots.paused = bool(settings.get("photos_paused"))
    walking, speed = geo.walking(ts)
    meta = {"ts": ts, "x": geo.x, "y": geo.y, "lat": geo.lat, "lng": geo.lng,
            "heading": geo.heading if geo.heading_known else None, "floor": geo.floor,
            "stairs": ts - geo.stairs_ts < 5, "node": geo.current_node, "q": geo.quality}
    kind = "manual" if force else None
    for p in people:
        if not p["confirmed"] and not (p["is_watch"] and p.get("possible")):
            continue
        pid = p["person_id"]
        prev = shots.in_view.get(pid)
        shots.in_view[pid] = ts
        if prev is not None and ts - prev < (15 if p["is_watch"] else 60):
            continue
        rec = db.person(pid) or {}
        if p["is_watch"]:
            kind = "alert"
        elif (p["name"] or rec.get("met")) and kind is None:
            kind = "person"
    it = None
    if kind and (force or not shots.paused):
        it = shots.add(img, people, faces.last_far, meta, ts, kind)
    elif geo.lat is not None:
        it = shots.offer(img, people, faces.last_far, meta, ts, walking, speed)
    if it:
        place.setdefault("events", []).append({"type": "shot", "id": it["id"], "kind": it["kind"]})
    return it


def _maybe_auto_block(place, ts):
    """Something is in the way while we're following a route toward it: block that stretch, reroute."""
    nxt = place.get("next")
    if not AUTO_BLOCK or not geo.route or not nxt or nxt.get("rel") is None or abs(nxt["rel"]) > 45:
        return
    if ts - last_obstacle["ts"] < OBSTACLE_COOLDOWN_S:
        return
    ev = geo.block_ahead()
    last_obstacle.update(ts=ts, wall=time.time(), rerouted=bool(ev and ev.get("rerouted")), shown=True, edge=ev)
    place["events"].append({"type": "obstacle", "rerouted": last_obstacle["rerouted"]})
    place["next"] = geo.next_step()


def _current_place():
    return {"node_id": geo.current_node, "place": geo.place_name(), "next": geo.next_step()}


def _scan_display(people, alerts, n_far):
    lines = []
    for a in alerts:
        lines.append(f"ALERT {a['watch_label'] or a['name'] or ''} · {a['dir']}")
    for p in people:
        if p.get("possible"):
            lines.append(f"possible {p['watch_label'] or p['name'] or 'alert'} · {p['dir']}")
    named = [p for p in people if p["confirmed"] and not p["is_watch"] and (p["name"] or p["notes"])]
    for p in named[:3]:
        lines.append(f"{p['name'] or 'Unknown'} · {p['dir']}" + (f" · {p['notes'][:30]}" if p["notes"] else ""))
    unnamed = [p for p in people if p["confirmed"] and not p["is_watch"] and not (p["name"] or p["notes"])]
    new = len([p for p in unnamed if p["new"]])
    if unnamed:
        lines.append(f"{len(unnamed)} unknown" + (f" · {new} new" if new else ""))
    if n_far:
        lines.append(f"{n_far} too far / unclear")
    n = len(people) + n_far
    return {"level": "alert" if alerts else "info",
            "title": f"Scan · {n} face{'s' if n != 1 else ''}" if not alerts else f"ALERT: {alerts[0]['watch_label'] or 'match'}",
            "lines": lines[:4] or ["nobody found"]}


def _scan(img, ts):
    global last_people, last_display, last_scan
    t0 = time.time()
    people = faces.process_frame(img, ts=ts, tiles=True)
    far = faces.last_far
    place = _current_place()
    H, W = img.shape[:2]
    scale = min(1.0, 2048 / max(H, W))
    alerts = _record(people, img, place, ts, store_scale=scale, force=True)
    _draw_live(img, people, far, scale=scale)
    _track_conversations(people, ts)
    last_people = people
    last_display = _scan_display(people, alerts, len(far))
    hold.update(until=time.time() + 8, display=last_display)
    last_scan = {"ts": ts, "faces": len(people) + len(far), "named": len([p for p in people if p["name"]]),
                 "far_unknown": len(far), "size": [W, H], "ms": int((time.time() - t0) * 1000)}
    return {"people": people, "alerts": alerts, "display": last_display, "place": place, **last_scan}


@app.post("/scan")
async def scan(request: Request):
    """Scan the room: body = one big photo (e.g. the glasses' 12 MP photo capture).
    It's searched in full-resolution tiles, so faces far beyond the video range are
    found. Every face is matched; results + a display card come back."""
    img = await _image_from_request(request)
    ts = time.time()
    got = await run_in_threadpool(pipeline_lock.acquire, True, 20)
    if not got:
        raise HTTPException(503, "busy, try again")
    try:
        return await run_in_threadpool(_scan, img, ts)
    finally:
        pipeline_lock.release()


@app.get("/shots")
def list_shots(after: int = 0, epoch: int = 0):
    """Street view photos, oldest first: where (x/y local meters, lat/lng), facing, floor, walk, kind
    (walk / person / alert / manual), and the known people in each with their box (0-1 of the photo).
    Incremental: pass back `epoch` and the last id you have as `after`; when existing photos changed
    (a turn re-placed some, one was deleted) the epoch moves and you get everything again."""
    full = epoch != shots.epoch
    items = shots.items if full else shots.since(after)
    named = {n["id"]: n["name"] for n in db.nodes() if n["name"]}
    return {"epoch": shots.epoch, "full": full, "n": len(shots.items),
            "shots": [{**it, "place": named.get(it.get("node"))} for it in items]}


@app.get("/shots/at")
def shot_at(ts: float, person: str | None = None):
    """The photo nearest a moment in time (e.g. a sighting), with that person in it if possible."""
    it = (shots.at_time(ts, person=person) if person else None) or shots.at_time(ts)
    if not it:
        raise HTTPException(404, "no photo near that time")
    return it


@app.get("/people/{pid}/moments")
def person_moments(pid: str):
    """Every street-view photo this person is in, newest first."""
    return list(reversed(shots.moments(pid)))


@app.post("/floor")
async def set_floor(request: Request):
    """{floor, moved}: moved=true (default) = you just went up/down: a stairs place joins the floors and the
    last photos are the stairwell; moved=false = a correction ("I'm on floor 3")."""
    body = await request.json()
    ev = geo.set_floor(int(body.get("floor", 1)), link=bool(body.get("moved", True)))
    if ev and ev.get("node_id") is not None:
        shots.mark_stairs(ev["since"])
    return {"ok": True, "floor": geo.floor, "event": ev}


@app.get("/shots/{sid}.jpg")
def shot_jpg(sid: int):
    p = shots.file(sid)
    if not p:
        raise HTTPException(404, "no such photo")
    return Response(open(p, "rb").read(), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.delete("/shots/{sid}")
def shot_delete(sid: int):
    if not shots.delete(sid):
        raise HTTPException(404, "no such photo")
    return {"ok": True, "n": len(shots.items)}


@app.post("/shots/take")
def shot_take():
    """A photo right here, right now (dashboard button / voice 'photo')."""
    if geo.last_img is None:
        raise HTTPException(409, "no camera frame yet")
    it = _maybe_shot(geo.last_img, last_people, {"events": []}, time.time(), force=True)
    return {"ok": bool(it), "shot": it, "n": len(shots.items)}


@app.post("/shots/clear")
def shots_clear():
    shots.clear()
    return {"ok": True}


@app.post("/shots/pause")
async def shots_pause(request: Request):
    body = await request.json()
    settings["photos_paused"] = bool(body.get("paused", True))
    shots.paused = settings["photos_paused"]
    _save_settings()
    return {"ok": True, "paused": settings["photos_paused"]}


@app.get("/cloud.bin")
def cloud_bin():
    """Latest 3D point cloud of what the glasses see (see depth.py for the layout)."""
    if not last_cloud:
        raise HTTPException(404, "no depth yet")
    return Response(last_cloud, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/live.jpg")
def live():
    if not last_live:
        raise HTTPException(404, "no frames yet")
    return Response(last_live, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/report")
def report():
    """'Report' the most recent alert: freeze its location + time for responders."""
    if not recent_alerts:
        raise HTTPException(404, "no alert to report")
    a = recent_alerts[0]
    rep = {"person_id": a["person_id"], "label": a["watch_label"], "ts": a["ts"], "ago": _ago(a["ts"]),
           "place": a["place"], "x": a["x"], "y": a["y"], "lat": a.get("lat"), "lng": a.get("lng"),
           "maps": f"https://maps.google.com/?q={a['lat']:.6f},{a['lng']:.6f}" if a.get("lat") else None,
           "reported_at": time.time()}
    reports.append(rep)
    where = a["place"] or (f"{a['lat']:.5f}, {a['lng']:.5f}" if a.get("lat") else "current position")
    return {"ok": True, "say": f"Reported: {a['watch_label'] or a['person_id']} at {where}", "report": rep}


# ---------------------------------------------------------------- people
@app.get("/people")
def list_people():
    return [_person_public(p) for p in db.people()]


@app.get("/people/{pid}")
def get_person(pid: str):
    p = db.person(pid)
    if not p:
        raise HTTPException(404)
    p["sightings"] = db.sightings(pid)
    return p


@app.get("/people/{pid}/photos")
def person_photos(pid: str):
    """Everything about one person: their training photos (what recognition compares
    against) and every sighting photo with the full camera frame."""
    p = db.person(pid)
    if not p:
        raise HTTPException(404)
    row = next((r for r in db.people() if r["id"] == pid), p)
    training = [{"id": e["id"], "photo": _url(e["crop"]), "quality": e["quality"], "source": e["source"],
                 "created_at": e["created_at"], "ago": _ago(e["created_at"])} for e in db.embeddings_for(pid)]
    sightings = [_sighting_public(s) for s in db.sightings(pid, limit=60)]
    return {"person": _person_public(row), "training": training, "sightings": sightings,
            "max_training": 16}


@app.post("/people/{pid}")
async def update_person(pid: str, request: Request):
    body = await request.json()
    if not db.person(pid):
        raise HTTPException(404)
    fields = {}
    if "name" in body:
        fields["name"] = (body["name"] or "").strip() or None
    if "notes" in body:
        fields["notes"] = body["notes"] or ""
    if "append_note" in body:
        old = db.person(pid)["notes"] or ""
        fields["notes"] = (old + " · " if old else "") + body["append_note"]
    db.update_person(pid, **fields)
    return db.person(pid)


@app.delete("/people/{pid}")
def delete_person(pid: str):
    snap = db.person_snapshot(pid)
    db.delete_person(pid)
    faces._reload_gallery()
    if not snap:
        return {"ok": True}
    token = uuid.uuid4().hex[:10]
    undo_log[token] = {"kind": "delete", **snap}
    while len(undo_log) > 20:
        undo_log.pop(next(iter(undo_log)))
    return {"ok": True, "undo": token, "name": snap["person"]["name"] or "Unknown"}


@app.post("/people/{pid}/merge")
async def merge_person(pid: str, request: Request):
    """This person is actually `into`: move all their photos and sightings there."""
    body = await request.json()
    into = body.get("into")
    snap = db.merge_people(pid, into)
    if not snap:
        raise HTTPException(400, "can't merge those two")
    faces._reload_gallery()
    token = uuid.uuid4().hex[:10]
    undo_log[token] = snap
    while len(undo_log) > 20:
        undo_log.pop(next(iter(undo_log)))
    return {"ok": True, "person_id": into, "undo": token,
            "src_name": snap["src"]["name"] or "Unknown", "dst_name": snap["dst"]["name"] or "Unknown"}


@app.post("/undo/{token}")
def undo(token: str):
    """Undo a merge (the dashboard's Undo button)."""
    snap = undo_log.pop(token, None)
    if snap and snap.get("kind") == "delete":
        if not db.restore_person(snap):
            raise HTTPException(404, "nothing to undo")
        faces._reload_gallery()
        return {"ok": True, "person_id": snap["person"]["id"]}
    if not snap or not db.undo_merge(snap):
        raise HTTPException(404, "nothing to undo")
    faces._reload_gallery()
    return {"ok": True, "person_id": snap["src"]["id"]}


@app.get("/people/{pid}/sightings")
def person_sightings(pid: str):
    return db.sightings(pid)


@app.delete("/train_photos/{eid}")
def delete_train_photo(eid: int):
    """Remove a bad or wrong photo from someone's training set."""
    if not db.embedding(eid):
        raise HTTPException(404)
    faces._remove_from_gallery(eid)
    return {"ok": True}


@app.post("/train_photos/{eid}/move")
async def move_train_photo(eid: int, request: Request):
    """This training photo belongs to someone else (or a new person)."""
    body = await request.json()
    e = db.embedding(eid)
    if not e:
        raise HTTPException(404)
    to = body.get("to")
    if to == "new":
        to = "p_" + uuid.uuid4().hex[:6]
        db.add_person(to, crop=e["crop"])
    elif not db.person(to):
        raise HTTPException(404, "no such person")
    db._run("UPDATE embeddings SET person_id=? WHERE rowid=?", (to, eid))
    faces._reload_gallery()
    return {"ok": True, "person_id": to}


@app.post("/sightings/{sid}/train")
def train_from_sighting(sid: int):
    """Add a sighting photo to that person's training set."""
    s = db.sighting(sid)
    if not s or not s["crop"] or not os.path.exists(s["crop"]):
        raise HTTPException(404)
    img = cv2.imread(s["crop"])
    f = faces.largest_face(img) if img is not None else None
    if not f:
        raise HTTPException(422, "couldn't find the face in that photo")
    faces._add_to_gallery(s["person_id"], f["emb"], img, f["box"], max(f["quality"], 0.4), "manual")
    return {"ok": True}


@app.post("/sightings/{sid}/move")
async def move_sighting(sid: int, request: Request):
    """This sighting was actually someone else (or a new person)."""
    body = await request.json()
    s = db.sighting(sid)
    if not s:
        raise HTTPException(404)
    to = body.get("to")
    if to == "new":
        to = "p_" + uuid.uuid4().hex[:6]
        db.add_person(to, crop=s["crop"])
    elif not db.person(to):
        raise HTTPException(404, "no such person")
    db._run("UPDATE sightings SET person_id=? WHERE id=?", (to, sid))
    return {"ok": True, "person_id": to}


@app.delete("/sightings/{sid}")
def delete_sighting(sid: int):
    db.delete_sighting(sid)
    return {"ok": True}


@app.get("/suggestions")
def suggestions():
    """Pairs of people who might be the same person (recognition split them)."""
    out = []
    pmap = {p["id"]: p for p in db.people()}
    for sgt in faces.suggest_merges():
        a, b = pmap.get(sgt["a"]), pmap.get(sgt["b"])
        if not a or not b:
            continue
        out.append({"sim": sgt["sim"], "a": _person_public(a), "b": _person_public(b)})
    return out


@app.post("/suggestions/dismiss")
async def dismiss_suggestion(request: Request):
    body = await request.json()
    faces.dismiss_suggestion(body["a"], body["b"])
    return {"ok": True}


@app.post("/enroll")
async def enroll(image: UploadFile = File(...), name: str = Form(None), notes: str = Form("")):
    """Enroll from a photo (e.g. a selfie). If the face is already known, the name/notes
    attach to the existing person."""
    img = _decode(await image.read())
    r = faces.enroll_image(img, name=name, notes=notes)
    if not r:
        raise HTTPException(422, "no face found")
    return r


# ---------------------------------------------------------------- train mode
@app.post("/train/start")
async def train_start(request: Request):
    """Collect a person's best angles: they look at the glasses and slowly turn their
    head. Body: {person_id} or {name} (creates the person if new), seconds, goal."""
    body = await request.json()
    pid = body.get("person_id")
    name = (body.get("name") or "").strip()
    if not pid and name:
        p = db.find_person_by_name(name)
        if p:
            pid = p["id"]
        else:
            pid = "p_" + uuid.uuid4().hex[:6]
            db.add_person(pid, name=name.title())
    if not pid:
        pid = "p_" + uuid.uuid4().hex[:6]
        db.add_person(pid)
    if not db.person(pid):
        raise HTTPException(404, "no such person")
    return faces.start_training(pid, seconds=int(body.get("seconds", 20)), goal=int(body.get("goal", 10)))


@app.post("/train/stop")
def train_stop():
    return faces.stop_training() or {"ok": True}


@app.get("/train")
def train_status():
    return faces.train_state()


# ---------------------------------------------------------------- watchlist (alerts)
@app.post("/watch")
async def watch(image: UploadFile = File(...), name: str = Form(...), age: str = Form(""),
                description: str = Form(""), label: str = Form("")):
    """Load an AMBER-style alert: the person's name, photo, and optionally age and description.
    Returns whether we've ALREADY seen them, where and when, plus a route to that last sighting.
    From now on, seeing them triggers an alert on the glasses."""
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "the person's name is required")
    label = label.strip() or f"AMBER Alert · {name}" + (f", {age.strip()}" if age.strip() else "")
    img = _decode(await image.read())
    r = faces.enroll_image(img, name=name, notes=description.strip(), is_watch=1, watch_label=label)
    if not r:
        raise HTTPException(422, "no face found in alert photo")
    pid = r["person_id"]
    seen = db.sightings(pid, limit=20)
    for s in seen:
        s["ago"] = _ago(s["ts"])
        s["photo"] = _url(s["crop"])
    route = _route_to_person(pid) if seen else None
    return {"person_id": pid, "seen_before": r["matched_existing"], "sim": r["sim"], "sightings": seen,
            "last": seen[0] if seen else None, "route": route}


@app.post("/people/{pid}/flag")
def flag_person(pid: str, label: str = "ALERT"):
    """Mark an existing person as an alert target (no photo needed)."""
    if not db.person(pid):
        raise HTTPException(404)
    db.update_person(pid, is_watch=1, watch_label=label)
    return {"ok": True}


@app.delete("/watch/{pid}")
def unwatch(pid: str):
    db.update_person(pid, is_watch=0, watch_label="")
    return {"ok": True}


@app.get("/alerts")
def alerts():
    return [{k: v for k, v in a.items() if k != "box"} | {"ago": _ago(a["ts"])} for a in recent_alerts]


# ---------------------------------------------------------------- map & routing
def _pins(public=None):
    """Everyone with a map position: photo, red (seen) / green (talked) ring."""
    in_view = {p["person_id"] for p in last_people if p["confirmed"] and not p.get("possible")}
    out = []
    for pp in (public if public is not None else [_person_public(p) for p in db.people()]):
        if pp["lat"] is None:
            continue
        out.append({**pp, "in_view": pp["id"] in in_view})
    return out


_sugg_cache = {"t": 0.0, "n": 0}


def _n_suggestions():
    """Merge suggestions are O(people^2): recompute at most every 10 s for the dashboard."""
    if time.time() - _sugg_cache["t"] > 10:
        _sugg_cache.update(t=time.time(), n=len(faces.suggest_merges()))
    return _sugg_cache["n"]


def _route_to_person(pid):
    p = db.person(pid)
    if not p:
        raise HTTPException(404, "no such person")
    ls = db.last_located_sighting(pid)
    if not ls:
        raise HTTPException(404, "never seen them at a known position")
    row = next((r for r in db.people() if r["id"] == pid), p)
    photo = _url(row.get("best_crop") or row.get("crop"))
    return geo.plan(ls["lat"], ls["lng"], name=p["name"] or "that person", person_id=pid, photo=photo)


def _route_to_node(nid):
    n = db.node(nid)
    if not n or n.get("lat") is None:
        raise HTTPException(404, "no such place")
    return geo.plan(n["lat"], n["lng"], name=geo.node_name(n), goal_node=n["id"])


@app.get("/map")
def get_map():
    return {**geo.state(), "pins": _pins()}


@app.post("/map/clear")
def clear_map():
    """Forget all places and the walked map (keeps people and their photos)."""
    db.clear_map()
    geo.__init__(db)
    geo.indoors_forced = bool(settings.get("indoors", False))
    shots.clear()
    last_sighting_log.clear()
    return {"ok": True}


@app.post("/places/mark")
async def mark_place(request: Request):
    """Name the place you're standing at."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    r = geo.mark_place((body.get("name") or "").strip().title() or None)
    if not r:
        raise HTTPException(409, "no GPS position yet")
    return r


@app.post("/nodes/{nid}")
async def label_node(nid: int, request: Request):
    body = await request.json()
    geo.label(nid, body.get("name"))
    n = db.node(nid)
    return {**n, "name": geo.node_name(n)} if n else {}


@app.post("/edges/{a}/{b}")
async def set_edge(a: int, b: int, request: Request):
    body = await request.json()
    geo.block(a, b, bool(body.get("blocked", True)))
    return {"ok": True, "route": geo.route_public() if geo.route else None}


@app.get("/route")
def route(to: int | None = None, to_person: str | None = None, to_name: str | None = None,
          lat: float | None = None, lng: float | None = None, name: str | None = None):
    """Plan a route to a place id, a place name, a person's last sighting, or any lat/lng."""
    if to_person:
        return _route_to_person(to_person)
    if to_name:
        n = db.node_by_name(to_name)
        if not n:
            raise HTTPException(404, "no place with that name")
        return _route_to_node(n["id"])
    if to is not None:
        return _route_to_node(to)
    if lat is not None and lng is not None:
        return geo.plan(lat, lng, name=name or "pin")
    raise HTTPException(400, "give to, to_person, to_name or lat&lng")


@app.delete("/route")
def cancel_route():
    geo.cancel()
    return {"ok": True}


# ---------------------------------------------------------------- categories
@app.get("/categories")
def get_categories():
    return db.categories()


@app.post("/categories")
async def add_category(request: Request):
    name = ((await request.json()).get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    db.add_category(name)
    return db.categories()


@app.post("/categories/rename")
async def rename_category(request: Request):
    body = await request.json()
    old, new = (body.get("old") or "").strip(), (body.get("new") or "").strip()
    if not old or not new:
        raise HTTPException(400, "old and new required")
    db.rename_category(old, new)
    return db.categories()


@app.delete("/categories/{name}")
def delete_category(name: str):
    """Remove a category; its people go back to Unsorted."""
    db.delete_category(name)
    return db.categories()


@app.post("/people/{pid}/category")
async def set_category(pid: str, request: Request):
    """Put someone in a category (drag and drop). {category: null} = Unsorted."""
    if not db.person(pid):
        raise HTTPException(404)
    cat = ((await request.json()).get("category") or "").strip() or None
    if cat:
        db.add_category(cat)
    db.update_person(pid, category=cat)
    return {"ok": True, "category": cat}


# ---------------------------------------------------------------- groups: who you met where
def _groups(public=None):
    """People grouped by where you met them (within group_radius_m), or, without GPS,
    by when (meetings more than 20 minutes apart are separate groups)."""
    R = float(settings.get("group_radius_m", 30))
    public = public if public is not None else [_person_public(p) for p in db.people()]
    located, timed = [], []
    for pp in public:
        item = {"id": pp["id"], "name": pp["name"], "photo": pp["photo"], "met": pp["met"],
                "category": pp["category"], "is_watch": pp["is_watch"], "ts": pp["first_seen"] or 0}
        if pp["meet_lat"] is not None:
            located.append((pp["meet_lat"], pp["meet_lng"], item))
        else:
            timed.append(item)
    groups = []
    for lat, lng, item in sorted(located, key=lambda t: t[2]["ts"]):
        g = next((g for g in groups if haversine_m(g["lat"], g["lng"], lat, lng) <= R), None)
        if g is None:
            g = {"kind": "place", "lat": lat, "lng": lng, "people": []}
            groups.append(g)
        g["people"].append(item)
        n = len(g["people"])
        g["lat"] += (lat - g["lat"]) / n
        g["lng"] += (lng - g["lng"]) / n
    k = 0
    for g in groups:
        k += 1
        g["name"] = _place_near(g["lat"], g["lng"], R) or f"Area {k}"
        g["named"] = not g["name"].startswith("Area ")
    session = None
    for item in sorted(timed, key=lambda i: i["ts"]):
        if session is None or item["ts"] - session["end"] > SESSION_GAP_S:
            session = {"kind": "time", "lat": None, "lng": None, "people": [], "start": item["ts"], "end": item["ts"], "named": False}
            groups.append(session)
        session["people"].append(item)
        session["end"] = item["ts"]
    for g in groups:
        ts = [i["ts"] for i in g["people"]]
        g["start"], g["end"] = min(ts), max(ts)
        g["ago"] = _ago(g["end"])
        if g["kind"] == "time":
            a, b = time.localtime(g["start"]), time.localtime(g["end"])
            g["name"] = time.strftime("%a %I:%M %p", a).replace(" 0", " ") + \
                ("" if g["end"] - g["start"] < 120 else time.strftime(" – %I:%M %p", b).replace(" 0", " "))
        g["people"].sort(key=lambda i: (not i["met"], -(i["ts"] or 0)))
        g["id"] = f"{g['kind']}-{int(g['start'])}"
    return sorted(groups, key=lambda g: -g["end"])


@app.get("/groups")
def get_groups():
    return _groups()


@app.post("/groups/name")
async def name_group(request: Request):
    """Name a place group: labels the nearest place on the map (or creates one there)."""
    body = await request.json()
    lat, lng, name = body.get("lat"), body.get("lng"), (body.get("name") or "").strip()
    if lat is None or not name:
        raise HTTPException(400, "lat, lng and name required")
    nid, d = geo._nearest_node(lat, lng)
    R = float(settings.get("group_radius_m", 30))
    if nid is not None and d <= R:
        geo.label(nid, name)
    else:
        x, y = geo._xy(lat, lng)
        nid = db.add_node(name, x, y, lat=lat, lng=lng)
    _nodes_cache["t"] = 0.0
    return {"ok": True, "node_id": nid, "name": name}


# ---------------------------------------------------------------- search area (AMBER)
SEARCH_SPEED_MPS = 1.5        # brisk walk, ~3.4 mph: how far someone can get on foot
SEARCH_MAX_M = 3200.0         # cap the search circle at 2 miles
SEARCH_MIN_M = 150.0
SEARCH_MAX_CALLS = 16         # Places lookups per search area (each returns up to 20 places)
VISIT_M = 25.0                # walking this close to a building marks it checked
search = None                 # the active search, see _start_search


# "Main buildings": places someone would walk into and that have a real name (campus buildings,
# libraries, malls, stations, parks...), not every food truck and ATM.
MAIN_TYPES = ["university", "school", "primary_school", "secondary_school", "library", "hospital",
              "shopping_mall", "department_store", "supermarket", "stadium", "museum", "church",
              "city_hall", "police", "fire_station", "transit_station", "bus_station", "train_station",
              "gym", "park", "movie_theater", "amusement_park", "zoo", "aquarium", "hotel", "lodging",
              "restaurant", "cafe", "fast_food_restaurant", "convenience_store", "store"]
_types_ok = {"v": True}   # if Google rejects a type name, fall back to no filter


def _places_nearby(lat, lng, radius, n=20, types=None):
    """Google Places Nearby Search (New): places within radius m. Raises on HTTP errors."""
    import urllib.error
    import urllib.request
    key = settings.get("google_maps_key")
    if not key:
        raise HTTPException(409, "add a Google Maps key (with Places API (New) enabled) to search buildings")
    q = {"maxResultCount": n, "rankPreference": "DISTANCE",
         "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": float(radius)}}}
    if types and _types_ok["v"]:
        q["includedTypes"] = types
    body = json.dumps(q)
    req = urllib.request.Request("https://places.googleapis.com/v1/places:searchNearby", data=body.encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Goog-Api-Key": key,
                                          "X-Goog-FieldMask": "places.id,places.displayName,places.types,places.location,places.primaryType"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 400 and "includedTypes" in q:   # a type name Google doesn't accept
            _types_ok["v"] = False
            return _places_nearby(lat, lng, radius, n)
        raise
    month = time.strftime("%Y-%m")
    poi_cache.setdefault("_calls", {})[month] = poi_cache.get("_calls", {}).get(month, 0) + 1
    out = []
    for pl in data.get("places", []):
        types = set(pl.get("types", []))
        if types & SKIP_TYPES or not pl.get("location"):
            continue
        out.append({"id": pl.get("id"), "name": (pl.get("displayName") or {}).get("text", "?"),
                    "lat": pl["location"]["latitude"], "lng": pl["location"]["longitude"],
                    "type": (pl.get("primaryType") or next(iter(types), "")).replace("_", " ")})
    return out


def _buildings_in(lat, lng, radius, main=True):
    """Every place Google knows inside the circle: tile it into smaller circles (each lookup
    returns at most 20), dedupe, and keep the ones really inside."""
    tile_r = max(120.0, radius / 2.2)
    step = tile_r * 1.5
    k = int(math.ceil(radius / step))
    centers = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            dx, dy = i * step, j * step
            if math.hypot(dx, dy) <= radius + tile_r * 0.5:
                centers.append((math.hypot(dx, dy), dx, dy))
    centers.sort()
    found = {}
    for _, dx, dy in centers[:SEARCH_MAX_CALLS]:
        clat = lat + dy / 110540.0
        clng = lng + dx / (111320.0 * math.cos(math.radians(lat)))
        for pl in _places_nearby(clat, clng, tile_r, types=MAIN_TYPES if main else None):
            d = haversine_m(lat, lng, pl["lat"], pl["lng"])
            if d <= radius and pl["id"] not in found:
                found[pl["id"]] = {**pl, "dist": round(d)}
    return list(found.values())


def _two_opt(seq, start):
    """Shorten a walking order (open path from `start`) by reversing segments that cross."""
    if len(seq) < 4:
        return seq
    pts = [start] + [(b["lat"], b["lng"]) for b in seq]
    d = lambda a, b: haversine_m(a[0], a[1], b[0], b[1])  # noqa: E731
    idx = list(range(1, len(pts)))
    improved, rounds = True, 0
    while improved and rounds < 30:
        improved, rounds = False, rounds + 1
        path = [0] + idx
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                a, b = pts[path[i - 1]], pts[path[i]]
                c = pts[path[j]]
                e = pts[path[j + 1]] if j + 1 < len(path) else None
                before = d(a, b) + (d(c, e) if e else 0)
                after = d(a, c) + (d(b, e) if e else 0)
                if after < before - 0.5:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True
        idx = path[1:]
    return [seq[k - 1] for k in idx]


def _ring_w(radius):
    return max(150.0, radius / 6.0)   # bigger circles search in wider rings


PRIORITY_W = 1.2   # how much "closer to where they were last seen" outweighs "closer to walk to"


def _next_cost(cur, b):
    """Cost of searching building b next, standing at cur: the walk there plus a penalty for being
    far from where they were last seen. Close-to-last-seen first, without zig-zagging across the map."""
    return haversine_m(cur[0], cur[1], b["lat"], b["lng"]) + PRIORITY_W * b["dist"]


def _order_buildings(blds, center, here):
    """Search order: repeatedly take the cheapest next building (walk + distance from last seen)."""
    left, order, cur = blds[:], [], here or center
    while left:
        nxt = min(left, key=lambda b: _next_cost(cur, b))
        left.remove(nxt)
        order.append(nxt)
        cur = (nxt["lat"], nxt["lng"])
    for i, b in enumerate(order):
        b["order"] = i + 1
    return order


def _start_search(pid, minutes=None, speed=None, main=True):
    global search
    p = db.person(pid)
    if not p:
        raise HTTPException(404, "no such person")
    ls = db.last_located_sighting(pid)
    if not ls:
        raise HTTPException(404, "they were never seen at a known map position")
    speed = float(speed or 1609.34 / (60.0 * float(settings.get("walk_min_per_mile", 16.5))))
    elapsed = (minutes * 60.0) if minutes else max(0.0, time.time() - ls["ts"])
    radius = max(SEARCH_MIN_M, min(SEARCH_MAX_M, speed * elapsed))
    try:
        blds = _buildings_in(ls["lat"], ls["lng"], radius, main=main)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"building lookup failed: {e}")
    here = (geo.lat, geo.lng) if geo.lat is not None else None
    search = {"person_id": pid, "name": p["name"] or "them", "label": p["watch_label"] or p["name"] or "",
              "lat": ls["lat"], "lng": ls["lng"], "seen_ts": ls["ts"], "place": ls.get("place") or _place_near(ls["lat"], ls["lng"]),
              "speed": speed, "minutes": minutes, "radius": radius, "started": time.time(),
              "buildings": _order_buildings(blds, (ls["lat"], ls["lng"]), here), "checked": set(), "target": None}
    return _search_public()


def _search_next(route=True):
    """Route to the next building nobody has checked yet."""
    if not search:
        return None
    left = [b for b in search["buildings"] if b["id"] not in search["checked"]]
    if not left:
        search["target"] = None
        return None
    here = (geo.lat, geo.lng) if geo.lat is not None else (search["lat"], search["lng"])
    nxt = min(left, key=lambda b: _next_cost(here, b))
    search["target"] = nxt["id"]
    if route and geo.lat is not None:
        geo.plan(nxt["lat"], nxt["lng"], name=nxt["name"])
    return nxt


def _search_check(bid):
    if search and bid:
        search["checked"].add(bid)


def _search_tick():
    """Each frame: walking up to a building marks it checked; arriving at the target moves on."""
    if not search or geo.lat is None:
        return
    for b in search["buildings"]:
        if b["id"] not in search["checked"] and haversine_m(geo.lat, geo.lng, b["lat"], b["lng"]) <= VISIT_M:
            search["checked"].add(b["id"])
    if search["target"] and search["target"] in search["checked"]:
        _search_next()


def _search_public():
    if not search:
        return None
    s = search
    reach = s["speed"] * (time.time() - s["seen_ts"]) if not s["minutes"] else s["radius"]
    tgt = next((b for b in s["buildings"] if b["id"] == s["target"]), None)
    return {"person_id": s["person_id"], "name": s["name"], "label": s["label"], "lat": s["lat"], "lng": s["lng"],
            "place": s["place"], "seen_ago": _ago(s["seen_ts"]), "radius_m": round(s["radius"]),
            "reach_now_m": round(min(SEARCH_MAX_M, max(SEARCH_MIN_M, reach))), "speed": s["speed"],
            "buildings": [{**b, "checked": b["id"] in s["checked"], "next": b["id"] == s["target"]} for b in s["buildings"]],
            "n": len(s["buildings"]), "n_checked": len(s["checked"]), "target": tgt}


@app.post("/search/start/{pid}")
async def start_search(pid: str, request: Request):
    """Search area for an alert target: everywhere they could have walked since they were last
    seen (speed x time, or {minutes}), every building in it, in search order, and a route to the first."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    r = _start_search(pid, body.get("minutes"), body.get("speed"), main=body.get("main", True))
    _search_next()
    return _search_public()


@app.get("/search")
def get_search():
    return _search_public() or {}


@app.post("/search/check")
async def search_check(request: Request):
    """Mark a building searched ({id}, or the current target) and route to the next one."""
    if not search:
        raise HTTPException(404, "no active search")
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    bid = body.get("id") or search["target"]
    if body.get("unchecked"):
        search["checked"].discard(bid)
    else:
        _search_check(bid)
    if bid == search["target"] or not search["target"]:
        _search_next()
    return _search_public()


@app.post("/search/go")
async def search_go(request: Request):
    """Route to a specific building ({id}) or the next one."""
    if not search:
        raise HTTPException(404, "no active search")
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    b = next((b for b in search["buildings"] if b["id"] == body.get("id")), None)
    if b:
        search["target"] = b["id"]
        if geo.lat is not None:
            geo.plan(b["lat"], b["lng"], name=b["name"])
    else:
        _search_next()
    return _search_public()


@app.delete("/search")
def end_search():
    global search
    search = None
    geo.cancel()
    return {"ok": True}


def _watchlist(public):
    """Alert targets with what the glasses know: in view now / last seen where / never."""
    in_view = {p["person_id"]: p for p in last_people if p["is_watch"]}
    out = []
    for pp in public:
        if not pp["is_watch"]:
            continue
        v = in_view.get(pp["id"])
        status = ("in_view" if v and v["confirmed"] and not v.get("possible") else
                  "possible" if v else "seen" if pp["n_seen"] else "not_seen")
        out.append({**pp, "status": status, "dir": v["dir"] if v else None})
    return out


# ---------------------------------------------------------------- settings / met
def _save_settings():
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f)


@app.get("/settings")
def get_settings():
    return settings


@app.post("/settings")
async def set_settings(request: Request):
    body = await request.json()
    if "convo_seconds" in body:
        settings["convo_seconds"] = max(3, int(body["convo_seconds"]))
    if "convo_enabled" in body:
        settings["convo_enabled"] = bool(body["convo_enabled"])
    if "walk_min_per_mile" in body:
        settings["walk_min_per_mile"] = max(6.0, min(60.0, float(body["walk_min_per_mile"])))
    if "group_radius_m" in body:
        settings["group_radius_m"] = max(5, min(500, float(body["group_radius_m"])))
    if "google_maps_key" in body:
        settings["google_maps_key"] = (body["google_maps_key"] or "").strip()
    if "indoors" in body:
        settings["indoors"] = bool(body["indoors"])
        geo.indoors_forced = settings["indoors"]
        if not settings["indoors"]:
            geo.auto_indoor = False
            geo.good_fix_ts = time.time()
    if "blur_strangers" in body:
        settings["blur_strangers"] = bool(body["blur_strangers"])
    if "photos_paused" in body:
        settings["photos_paused"] = bool(body["photos_paused"])
        shots.paused = settings["photos_paused"]
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f)
    return settings


@app.post("/people/{pid}/met")
async def set_met(pid: str, request: Request):
    """Mark by hand that you talked to someone (green pin), or undo it."""
    body = await request.json()
    if not db.person(pid):
        raise HTTPException(404)
    met = bool(body.get("met", True))
    db.update_person(pid, met=1 if met else 0, met_at=time.time() if met else None,
                     met_lat=geo.lat if met else None, met_lng=geo.lng if met else None)
    return {"ok": True, "met": met}


# ---------------------------------------------------------------- voice commands
@app.post("/command")
async def command(request: Request):
    """Pipe speech-to-text here. Understands:
       'this is <name>'            name the person in front of you
       'note <text>'               attach a note to the person in front of you
       'train' / 'train <name>'    train mode on the person in front of you
       'mark place [<name>]'       create a place right here
       'call this place <name>'    label the current place
       'take me to <place>'        route to a place
       'find <name>'               route to that person's last sighting
       'where did I see <name>'    answer with place + time
       'report'                    report the latest alert
       'cancel'                    drop the route / stop training"""
    body = await request.json()
    text = (body.get("text") or "").strip().lower().rstrip(".!?")
    if not text:
        raise HTTPException(400, "text required")

    def _say_route(r):
        if not r.get("ok"):
            return r.get("reason", "No route")
        n = r.get("next") or {}
        how = "direct" if r["mode"] == "compass" else "along the way you walked"
        return f"{r['remaining_m']} meters {how}. " + (f"{n['turn']}" if n.get("turn") else f"Head {n.get('compass', '')}")

    def front_person():
        cands = [p for p in last_people if not p.get("possible")]
        if not cands:
            return None
        return max(cands, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))

    if text.startswith("this is "):
        p = front_person()
        if not p:
            return {"ok": False, "say": "I don't see anyone"}
        db.update_person(p["person_id"], name=text[8:].strip().title())
        return {"ok": True, "say": f"Got it, {text[8:].strip().title()}"}
    if text.startswith("note "):
        p = front_person()
        if not p:
            return {"ok": False, "say": "I don't see anyone"}
        old = db.person(p["person_id"])["notes"] or ""
        db.update_person(p["person_id"], notes=(old + " · " if old else "") + text[5:].strip())
        return {"ok": True, "say": "Noted"}
    if text == "train" or text.startswith("train "):
        name = text[6:].strip()
        if name:
            p = db.find_person_by_name(name)
            pid = p["id"] if p else None
            if not pid:
                pid = "p_" + uuid.uuid4().hex[:6]
                db.add_person(pid, name=name.title())
        else:
            fp = front_person()
            if not fp:
                return {"ok": False, "say": "Look at the person first"}
            pid = fp["person_id"]
        st = faces.start_training(pid)
        return {"ok": True, "say": f"Training {st['name'] or 'this person'}: turn your head slowly", "train": st}
    if text.startswith("mark place") or text.startswith("call this place "):
        name = text[16:] if text.startswith("call this place ") else text[10:]
        r = geo.mark_place(name.strip().title() or None)
        return {"ok": bool(r), "say": f"Marked {r['name']}" if r else "No GPS position yet"}
    if text.startswith("take me to ") or text.startswith("navigate to ") or text.startswith("go to "):
        target = text.split(" to ", 1)[1].strip()
        n = db.node_by_name(target)
        if n:
            r = _route_to_node(n["id"])
        else:
            p = db.find_person_by_name(target)
            if not p:
                return {"ok": False, "say": "I don't know that place or person"}
            try:
                r = _route_to_person(p["id"])
            except HTTPException as e:
                return {"ok": False, "say": e.detail}
        return {"ok": r["ok"], "say": _say_route(r), "route": r}
    if text in ("blocked", "path blocked", "this way is blocked", "it's blocked", "its blocked", "road blocked"):
        ev = geo.block_ahead()
        if not ev:
            return {"ok": False, "say": "No route to block"}
        return {"ok": True, "say": "Blocked. " + ("Rerouting" if ev["rerouted"] else "No other way, going direct"),
                "route": geo.route_public() if geo.route else None}
    for prefix in ("where did i see ", "where did i meet "):
        if text.startswith(prefix):
            p = db.find_person_by_name(text[len(prefix):].strip())
            ls = db.last_sighting(p["id"]) if p else None
            if not ls:
                return {"ok": False, "say": "I haven't seen them"}
            return {"ok": True, "say": f"{p['name']}: {ls['place'] or 'unknown place'}, {_ago(ls['ts'])}", "sighting": ls}
    if text.startswith("find "):
        p = db.find_person_by_name(text[5:].strip())
        if not p:
            return {"ok": False, "say": "I don't know them"}
        ls = db.last_located_sighting(p["id"])
        if not ls:
            return {"ok": False, "say": "Never seen them at a known position"}
        r = _route_to_person(p["id"])
        return {"ok": r["ok"], "say": f"Last seen {ls['place'] + ' ' if ls['place'] else ''}{_ago(ls['ts'])}. " + _say_route(r), "route": r}
    if text in ("next", "next building", "searched", "checked", "clear", "nothing here", "not here"):
        if not search:
            return {"ok": False, "say": "No search running"}
        if text != "next" and text != "next building":
            _search_check(search["target"])
        b = _search_next()
        return {"ok": bool(b), "say": f"Next: {b['name']}, {b['dist']} meters from where they were seen" if b else "Every place is checked"}
    if text in ("stop search", "end search", "cancel search"):
        end_search()
        return {"ok": True, "say": "Search ended"}
    if text in ("indoors", "going inside", "inside", "we're inside"):
        settings["indoors"] = True
        geo.indoors_forced = True
        _save_settings()
        return {"ok": True, "say": "Indoors: steps and the camera from here"}
    if text in ("outdoors", "going outside", "outside", "we're outside"):
        settings["indoors"] = False
        geo.indoors_forced = False
        geo.auto_indoor = False
        geo.good_fix_ts = time.time()
        _save_settings()
        return {"ok": True, "say": "Outdoors: GPS again"}
    if text in ("pause photos", "stop photos", "privacy", "stop recording"):
        settings["photos_paused"] = True
        shots.paused = True
        _save_settings()
        return {"ok": True, "say": "Photos paused"}
    if text in ("resume photos", "start photos", "start recording"):
        settings["photos_paused"] = False
        shots.paused = False
        _save_settings()
        return {"ok": True, "say": "Photos on"}
    if text in ("upstairs", "going upstairs", "going up", "up a floor"):
        geo.set_floor(geo.floor + 1)
        return {"ok": True, "say": f"Floor {geo.floor}"}
    if text in ("downstairs", "going downstairs", "going down", "down a floor"):
        geo.set_floor(geo.floor - 1)
        return {"ok": True, "say": f"Floor {geo.floor}"}
    if text.startswith("floor ") and text[6:].strip().lstrip("-").isdigit():
        geo.set_floor(int(text[6:].strip()))
        return {"ok": True, "say": f"Floor {geo.floor}"}
    if text in ("photo", "take a photo", "street view", "picture", "snapshot", "take a snapshot"):
        try:
            r = shot_take()
            return {"ok": True, "say": f"Photo saved, {r['n']} along the walk"}
        except HTTPException as e:
            return {"ok": False, "say": e.detail}
    if text in ("scan", "scan the room", "scan room", "scan the crowd", "look around", "who's here", "who is here"):
        # the server can't take the photo: the app sees action=scan, captures a photo, POSTs it to /scan
        return {"ok": True, "action": "scan", "say": "Scanning, hold still"}
    if text in ("cancel", "stop", "cancel route", "stop navigation"):
        geo.cancel()
        faces.stop_training()
        return {"ok": True, "say": "Cancelled"}
    if text in ("report", "report it", "send location"):
        if not recent_alerts:
            return {"ok": False, "say": "No alert to report"}
        return report()
    return {"ok": False, "say": "Didn't catch that"}


# ---------------------------------------------------------------- misc
@app.get("/state")
def state():
    public = list_people()
    return {
        "people": public,
        "categories": db.categories(),
        "groups": _groups(public),
        "watchlist": _watchlist(public),
        "search": _search_public(),
        "last_shown": {**last_shown, "ago": _ago(last_shown["ts"]) if last_shown["ts"] else None},
        "sightings": [_sighting_public(s) for s in db.sightings(limit=30)],
        "map": {**geo.state(), "pins": _pins(public), "near": _poi(geo.lat, geo.lng),
                "poi_status": poi_state["status"] if settings.get("google_maps_key") else None},
        "settings": settings,
        "alerts": alerts()[:10],
        "display": last_display,
        "last_people": [{k: v for k, v in p.items() if k != "box"} for p in last_people],
        "reports": reports[-10:],
        "train": faces.train_state(),
        "n_suggestions": _n_suggestions(),
        "frames": frame_count,
        "dropped": dropped,
        "has_live": last_live is not None,
        "has_cloud": last_cloud is not None,
        "n_shots": len(shots.items), "shots_rev": shots.rev, "shots_epoch": shots.epoch,
        "photos_paused": bool(settings.get("photos_paused")),
        "indoors": geo.indoors, "indoors_forced": bool(settings.get("indoors", False)), "floor": geo.floor,
        "depth": last_depth if depth is not None else None,
        "frame_size": faces.last_size,
        "timing": _timing(),
        "far_unknown": len(faces.last_far),
        "last_scan": last_scan,
        "range": {"min_face_px": faces_mod.MIN_FACE_PX, "near_px": faces_mod.NEAR_PX,
                  "enroll_px": faces_mod.ENROLL_PX, "det": faces_mod.DET_SIZE},
    }


@app.get("/img/{kind}/{name}")
def image(kind: str, name: str):
    if kind not in DIRS or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404)
    path = os.path.join(DIRS[kind], name)
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "max-age=3600"})


@app.get("/crop/{pid}.jpg")
def crop(pid: str):
    p = db.person(pid)
    if not p or not p["crop"] or not os.path.exists(p["crop"]):
        raise HTTPException(404)
    return FileResponse(p["crop"])


@app.get("/keyframe/{nid}.jpg")
def keyframe(nid: int):
    n = db.node(nid)
    if not n or not n["keyframe"] or not os.path.exists(n["keyframe"]):
        raise HTTPException(404)
    return FileResponse(n["keyframe"])


@app.post("/reset")
def reset():
    """Wipe everything (demo reset / 'delete all' button)."""
    global last_people, last_display, last_live, last_scan
    last_scan = None
    db.wipe()
    faces._reload_gallery()
    faces.stop_training()
    geo.__init__(db)
    presence.clear()
    last_people, last_display, last_live = [], dict(QUIET), None
    hold.update(until=0.0, display=None)
    card_hold.update(card=None, ts=0.0)
    last_shown.update(card=None, ts=0.0)
    globals()["search"] = None
    shots.clear()
    geo.indoors_forced = bool(settings.get("indoors", False))
    faces.last_far = []
    recent_alerts.clear()
    reports.clear()
    last_sighting_log.clear()
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True, "frames": frame_count, "people": len(db.people()), "places": len(db.nodes())}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as f:
        return f.read()
