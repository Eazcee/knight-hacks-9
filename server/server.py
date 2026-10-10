"""Beacon server: faces + training sets + notes + sightings + alert watchlist +
self-building map + A* routing.

Run:   uvicorn server:app --host 0.0.0.0 --port 8000
Then:  open http://localhost:8000 for the dashboard.

The phone app (or webcam_client.py) POSTs JPEG frames to /frame and renders the
returned `display` card on the glasses.
"""
import json
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

from db import DB
import faces as faces_mod
from faces import FaceEngine
from places import PlaceMap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BEACON_DATA", "data")
RETENTION_H = float(os.environ.get("BEACON_RETENTION_H", 24))   # forget un-annotated strangers after this
AUTO_ENROLL = os.environ.get("BEACON_AUTO_ENROLL", "1") == "1"
SIGHTING_GAP_S = 15                                              # log a sighting at most this often per person+place
DIRS = {k: os.path.join(DATA, k) for k in ("crops", "frames", "keyframes")}

os.makedirs(DATA, exist_ok=True)
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)
db = DB(os.path.join(DATA, "beacon.db"))
db.prune_anonymous(RETENTION_H * 3600)
faces = FaceEngine(db, crops_dir=DIRS["crops"], auto_enroll=AUTO_ENROLL)
places = PlaceMap(db, keyframes_dir=DIRS["keyframes"])

app = FastAPI(title="Beacon")

QUIET = {"level": "quiet", "title": "Beacon ready", "lines": []}
last_people = []            # people seen in the most recent frame
last_display = dict(QUIET)
last_live = None            # latest frame as JPEG with boxes drawn, for the dashboard
recent_alerts = deque(maxlen=50)
reports = []                # alerts the wearer chose to report (location sent)
last_sighting_log = {}      # (pid, node_id) -> ts
frame_count = 0
dropped = 0
hold = {"until": 0.0, "display": None}   # keep a scan result on the glasses for a few seconds
last_scan = None
pipeline_lock = threading.Lock()   # one frame at a time; extra frames are dropped, not queued


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
    return f"{s / 3600:.1f} h ago"


def _url(path):
    """data/crops/x.jpg (or data\\crops\\x.jpg on Windows) -> /img/crops/x.jpg"""
    if not path:
        return None
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[-2] not in DIRS:
        return None
    return f"/img/{parts[-2]}/{parts[-1]}"


def _person_card(p):
    """One glanceable line per person for the glasses."""
    line = p["name"] or "Unknown"
    if p["notes"]:
        line += " · " + p["notes"][:60]
    return line


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
    lines = []
    seen = [p for p in people if p["confirmed"] and not p.get("possible")]
    known = sorted([p for p in seen if p["name"] or p["notes"]], key=lambda p: -p.get("px", 0))
    for p in known[:3]:
        lines.append(_person_card(p) + (f" · {p['dir']}" if p.get("tier") == "far" else ""))
    strangers = len([p for p in seen if not (p["name"] or p["notes"])])
    extra = len(known) - 3
    if strangers > 0 or extra > 0:
        new = len([p for p in seen if p["new"]])
        parts = ([f"+{extra} more"] if extra > 0 else []) + ([f"{strangers} unknown"] if strangers else [])
        lines.append(" · ".join(parts) + (f" · {new} new" if new else ""))
    if next_step and not next_step.get("arrived"):
        turn = next_step.get("turn") or "head"
        lines.insert(0, f"{turn.upper()} → {next_step['name']} · {next_step['dist_m']} m · {next_step['remaining']} left")
        return {"level": "info", "title": "Route", "lines": lines}
    if next_step and next_step.get("arrived"):
        return {"level": "info", "title": "Arrived", "lines": lines}
    if lines:
        return {"level": "info", "title": place or "", "lines": lines}
    return {"level": "quiet", "title": place or "", "lines": []}


def _person_public(p):
    """A person row for the UI, with image URLs and last place."""
    ls = db.last_sighting(p["id"])
    best = p.get("best_crop") or p.get("crop")
    return {
        "id": p["id"], "name": p["name"], "notes": p["notes"], "is_watch": bool(p["is_watch"]),
        "watch_label": p["watch_label"], "photo": _url(best), "n_train": p.get("n_train"),
        "n_seen": p.get("n_seen"), "last_seen": p["last_seen"], "last_seen_ago": _ago(p["last_seen"]),
        "last_place": ls["place"] if ls else None,
    }


def _sighting_public(s):
    box = json.loads(s["box"]) if s.get("box") else None
    return {"id": s["id"], "person_id": s["person_id"], "person_name": s.get("person_name"),
            "ts": s["ts"], "ago": _ago(s["ts"]), "place": s.get("place"), "sim": s["sim"],
            "quality": s.get("quality"), "photo": _url(s["crop"]), "frame": _url(s.get("frame")), "box": box}


# ---------------------------------------------------------------- frames
@app.post("/frame")
async def frame(request: Request, heading: float | None = None, steps: int | None = None,
                ts: float | None = None):
    """Main pipeline. Body = JPEG (Content-Type image/jpeg) or multipart 'image'.
    Optional query params: heading (deg, 0=north), steps (cumulative pedometer), ts."""
    global dropped
    img = await _image_from_request(request)
    ts = ts or time.time()
    if not pipeline_lock.acquire(blocking=False):
        dropped += 1   # still busy with the previous frame: drop this one, keep latency low
        return {"dropped": True, "people": [], "alerts": [], "place": None, "display": last_display}
    try:
        return await run_in_threadpool(_pipeline, img, heading, steps, ts)
    finally:
        pipeline_lock.release()


def _record(people, img, place, ts, store_scale=1.0, force=False):
    """Log sightings (with the full frame + face box) and collect alerts."""
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
            db.add_sighting(p["person_id"], ts, node_id, places.x, places.y, p["sim"], crop,
                            frame=frame_path, box=json.dumps(box), quality=p["quality"])
            last_sighting_log[key] = ts
        if p["is_watch"] and p["confirmed"]:
            a = {**p, "ts": ts, "node_id": node_id, "place": place["place"], "x": places.x, "y": places.y}
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
        label += f" {int(p['sim'] * 100)}% {p.get('px', '')}px"
        cv2.putText(live, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)
    ok, buf = cv2.imencode(".jpg", live, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        last_live = buf.tobytes()


def _pipeline(img, heading, steps, ts):
    global last_people, last_display, frame_count
    frame_count += 1
    people = faces.process_frame(img, ts=ts)
    place = places.observe(img, heading_deg=heading, steps=steps, ts=ts,
                           faces=len(people) + len(faces.last_far))
    alerts = _record(people, img, place, ts)
    _draw_live(img, people, faces.last_far)
    last_people = people
    last_display = _build_display(people, place["place"], alerts, place["next"], faces.train_state())
    if not alerts and time.time() < hold["until"] and hold["display"]:
        last_display = hold["display"]   # a scan result stays up for a few seconds
    return {"people": people, "place": place, "alerts": alerts, "display": last_display,
            "far_unknown": len(faces.last_far)}


# ---------------------------------------------------------------- scan (12 MP photo)
def _current_place():
    cur = db.node(places.current_node) if places.current_node else None
    return {"node_id": places.current_node, "place": cur["name"] if cur else None, "next": places.next_step()}


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
    rep = {"person_id": a["person_id"], "label": a["watch_label"], "ts": a["ts"],
           "place": a["place"], "x": a["x"], "y": a["y"], "reported_at": time.time()}
    reports.append(rep)
    return {"ok": True, "say": f"Reported: {a['watch_label'] or a['person_id']} at {a['place'] or 'current position'}", "report": rep}


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
    db.delete_person(pid)
    faces._reload_gallery()
    return {"ok": True}


@app.post("/people/{pid}/merge")
async def merge_person(pid: str, request: Request):
    """This person is actually `into`: move all their photos and sightings there."""
    body = await request.json()
    into = body.get("into")
    if not db.merge_people(pid, into):
        raise HTTPException(400, "can't merge those two")
    faces._reload_gallery()
    return {"ok": True, "person_id": into}


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
async def watch(image: UploadFile = File(...), label: str = Form("ALERT"), name: str = Form(None)):
    """Load an alert photo. Returns whether we've ALREADY seen this person, where and when,
    plus a route to that last sighting. From now on, seeing them triggers an alert."""
    img = _decode(await image.read())
    r = faces.enroll_image(img, name=name, is_watch=1, watch_label=label)
    if not r:
        raise HTTPException(422, "no face found in alert photo")
    pid = r["person_id"]
    seen = db.sightings(pid, limit=20)
    for s in seen:
        s["ago"] = _ago(s["ts"])
        s["photo"] = _url(s["crop"])
    route = None
    if seen and seen[0]["node_id"]:
        route = places.plan(seen[0]["node_id"])
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


# ---------------------------------------------------------------- places & routing
@app.get("/map")
def get_map():
    return places.state()


@app.post("/map/clear")
def clear_map():
    """Forget all places (keeps people and their photos)."""
    db.clear_map()
    places.__init__(db, keyframes_dir=DIRS["keyframes"])
    last_sighting_log.clear()
    return {"ok": True}


@app.post("/places/mark")
async def mark_place(request: Request):
    """Create a place right here from the latest camera frame."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    r = places.mark_place((body.get("name") or "").strip().title() or None, body.get("dist_m"))
    if not r:
        raise HTTPException(409, "no camera frames yet")
    return r


@app.post("/nodes/{nid}")
async def label_node(nid: int, request: Request):
    body = await request.json()
    places.label(nid, body["name"])
    return db.node(nid)


@app.post("/edges/{a}/{b}")
async def set_edge(a: int, b: int, request: Request):
    body = await request.json()
    places.block(a, b, bool(body.get("blocked", True)))
    return {"ok": True, "route": places.route}


@app.get("/route")
def route(to: int | None = None, to_person: str | None = None, to_name: str | None = None):
    """Plan a route to a place id, a place name, or a person's last sighting."""
    if to_person:
        ls = db.last_sighting(to_person)
        if not ls or not ls["node_id"]:
            raise HTTPException(404, "never seen that person at a known place")
        to = ls["node_id"]
    if to_name:
        n = db.node_by_name(to_name)
        if not n:
            raise HTTPException(404, "no place with that name")
        to = n["id"]
    if to is None:
        raise HTTPException(400, "give to, to_person or to_name")
    return places.plan(to)


@app.delete("/route")
def cancel_route():
    places.route, places.route_goal = None, None
    return {"ok": True}


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
    if text.startswith("mark place"):
        r = places.mark_place(text[10:].strip().title() or None)
        return {"ok": bool(r), "say": f"Marked {r['name']}" if r else "No camera yet"}
    if text.startswith("call this place "):
        if places.current_node is None:
            return {"ok": False, "say": "No place yet"}
        places.label(places.current_node, text[16:].strip().title())
        return {"ok": True, "say": "Labeled"}
    if text.startswith("take me to "):
        n = db.node_by_name(text[11:].strip())
        if not n:
            return {"ok": False, "say": "I don't know that place"}
        r = places.plan(n["id"])
        return {"ok": r["ok"], "say": f"Routing, {r.get('cost_m', 0)} meters" if r["ok"] else r["reason"], "route": r}
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
        ls = db.last_sighting(p["id"])
        if not ls or not ls["node_id"]:
            return {"ok": False, "say": "Never seen them at a known place"}
        r = places.plan(ls["node_id"])
        return {"ok": r["ok"], "say": f"Last seen {ls['place']} {_ago(ls['ts'])}. " + (f"Routing, {r['cost_m']} meters" if r["ok"] else r["reason"]), "route": r}
    if text in ("scan", "scan the room", "scan room", "scan the crowd", "look around", "who's here", "who is here"):
        # the server can't take the photo: the app sees action=scan, captures a photo, POSTs it to /scan
        return {"ok": True, "action": "scan", "say": "Scanning, hold still"}
    if text in ("cancel", "stop"):
        places.route, places.route_goal = None, None
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
    return {
        "people": list_people()[:60],
        "sightings": [_sighting_public(s) for s in db.sightings(limit=30)],
        "map": places.state(),
        "alerts": alerts()[:10],
        "display": last_display,
        "last_people": [{k: v for k, v in p.items() if k != "box"} for p in last_people],
        "reports": reports[-10:],
        "train": faces.train_state(),
        "n_suggestions": len(faces.suggest_merges()),
        "frames": frame_count,
        "dropped": dropped,
        "has_live": last_live is not None,
        "frame_size": faces.last_size,
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
    places.__init__(db, keyframes_dir=DIRS["keyframes"])
    last_people, last_display, last_live = [], dict(QUIET), None
    hold.update(until=0.0, display=None)
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
