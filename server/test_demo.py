"""End-to-end test of the Beacon server with no hardware.

Simulates a GPS walk: north from A to B (meet a group there, talk to them), east
to C, back to A. The map builds itself from the walk. Then an alert photo of
someone met at B arrives: the server says "seen before", where, and routes back
there over the walked map; a "blocked" command reroutes; walking there and
seeing her face ends the route. Also checks compass mode, dead reckoning,
conversation detection (green pins) and the settings.

  python test_demo.py
"""
import math
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("BEACON_DATA", "data_test")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

c = TestClient(server.app)
c.post("/reset")
c.post("/settings", json={"convo_seconds": 30, "convo_enabled": True})

from skimage import data  # noqa: E402
import insightface.data as idata  # noqa: E402

LAT0, LNG0 = 28.6024, -81.2001          # UCF


def ll(x, y):
    """local meters (east, north) -> lat, lng"""
    return LAT0 + y / 110540.0, LNG0 + x / (111320.0 * math.cos(math.radians(LAT0)))


def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def jpg(img):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return b.tobytes()


T = {"now": 1_000_000.0}


def frame(img, x=None, y=None, acc=4.0, heading=None, steps=None, dt=1.0):
    T["now"] += dt
    params = {"ts": T["now"]}
    if x is not None:
        params["lat"], params["lng"] = ll(x, y)
        params["acc"] = acc
    if heading is not None:
        params["heading"] = heading
    if steps is not None:
        params["steps"] = steps
    r = c.post("/frame", content=jpg(img), headers={"Content-Type": "image/jpeg"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def walk(img, x0, y0, x1, y1, step=1.3, noise=1.0):
    """Frames every second along a straight line, with GPS noise."""
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    rng = np.random.default_rng(0)
    r = None
    for i in range(1, n + 1):
        f = i / n
        r = frame(img, x0 + (x1 - x0) * f + rng.normal(0, noise), y0 + (y1 - y0) * f + rng.normal(0, noise))
    return r


scene = bgr(data.astronaut())
group = idata.get_image("t1")  # 6 faces, BGR
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


t0 = time.time()
# --- A: first GPS fix
r = frame(scene, 0, 0)
check(r["place"]["quality"] == "good" and any(e["type"] == "new_place" for e in r["place"]["events"]), "first GPS fix creates place 1 (A)")
nA = r["place"]["node_id"]

# --- walk north 24 m to B
r = walk(scene, 0, 0, 0, 24)
m = c.get("/map").json()
check(3 <= len(m["nodes"]) <= 5, f"24 m walk: {len(m['nodes'])} places (one every ~8 m)")
check(len(m["edges"]) >= 2 * (len(m["nodes"]) - 1), "consecutive places are linked")
check(r["place"]["heading"] is not None and abs(r["place"]["heading"]) < 20, f"heading from GPS course: {r['place']['heading']}° (north)")
r = c.post("/places/mark", json={"name": "coffee station"}).json()
nB = r["node_id"]
check(r["name"] == "Coffee Station" and nB != nA, "named the place here: Coffee Station")

# --- meet people at B; talk to them for 30+ s
r = frame(group, 0, 24)
check(len(r["people"]) >= 4, f"group photo: {len(r['people'])} strangers auto-enrolled at B")
pid0 = r["people"][0]["person_id"]
c.post(f"/people/{pid0}", json={"name": "Maya", "append_note": "judge, AI track, likes Rust"})
r = frame(group, 0, 24)
check(any(p["person_id"] == pid0 and p["name"] == "Maya" for p in r["people"]), "second look: Maya recognized with her note")
check(any("Maya" in l for l in r["display"]["lines"]), "glasses display shows Maya's card")
p = c.get("/people").json()
me = [q for q in p if q["id"] == pid0][0]
blat, blng = ll(0, 24)
check(me["lat"] and server.haversine_m(me["lat"], me["lng"], blat, blng) < 10 and not me["met"],
      f"Maya is on the map near B ({server.haversine_m(me['lat'], me['lng'], blat, blng):.1f} m off), red (seen, not talked)")
for _ in range(10):
    r = frame(group, 0, 24, dt=2.5)       # 25 s more in view
check(not [q for q in c.get("/people").json() if q["id"] == pid0][0]["met"], "25 s in view: still red")
r = frame(group, 0, 24, dt=6)
mp = [q for q in c.get("/map").json()["pins"] if q["id"] == pid0][0]
check(mp["met"] and any(q.get("met_now") for q in r["people"]), "30 s in view: Maya's pin turns green (talked)")
c.post("/settings", json={"convo_seconds": 45, "convo_enabled": False})
check(c.get("/settings").json()["convo_enabled"] is False and c.get("/state").json()["settings"]["convo_seconds"] == 45, "conversation setting is toggleable")
c.post("/settings", json={"convo_seconds": 30, "convo_enabled": True})

# --- walk east to C, then back to A along the diagonal
walk(scene, 0, 24, 24, 24)
r = walk(scene, 24, 24, 0, 0)
m = c.get("/map").json()
check(r["place"]["node_id"] == nA, f"back at A: snapped to place 1 ({len(m['nodes'])} places on the map)")
check(len(m["trail"]) > 30 and m["position"]["lat"], "walked trail recorded")

# --- alert: a photo of Maya arrives. Seen before? Where? Route there.
faces = server.faces.detect(group, min_px=20)
f = [f for f in faces if server.faces.match(f["emb"])[0] == pid0][0]
x1, y1, x2, y2 = f["box"]
w, h = x2 - x1, y2 - y1
alert_photo = cv2.flip(group[max(0, y1 - h):y2 + h, max(0, x1 - w):x2 + w], 1)
r = c.post("/watch", files={"image": ("alert.jpg", jpg(alert_photo), "image/jpeg")},
           data={"label": "AMBER Alert · Maya · 7"}).json()
check(r["seen_before"] and r["person_id"] == pid0, "alert photo matched someone we already saw")
check(r["last"] and r["last"]["place"] == "Coffee Station", f"last seen at {r['last']['place'] if r['last'] else None}")
route = r["route"]
check(route["ok"] and route["mode"] == "graph" and route["path"][0] == nA and nB in route["path"],
      f"route to her last sighting over the walked map: {route.get('path')} ({route.get('remaining_m')} m)")
direct_cost = route["cost_m"]
r = frame(scene, 0, 0, heading=0)
check(r["display"]["title"] == "→ Maya" and r["display"]["lines"][0].startswith("STRAIGHT"), f"glasses nav card: {r['display']['title']} / {r['display']['lines']}")
check(r["nav"]["turn"] == "straight" and 20 <= r["nav"]["remaining_m"] <= 35, f"nav: {r['nav']['turn']} {r['nav']['dist_m']} m, {r['nav']['remaining_m']} m left")
r = frame(scene, 0, 0, heading=90)
check(r["nav"]["turn"] == "left", f"facing east, B is north: turn {r['nav']['turn']}")

# --- voice
r = c.post("/command", json={"text": "where did I see Maya"}).json()
check(r["ok"] and "Coffee Station" in r["say"], f"voice: {r['say']}")
r = c.post("/command", json={"text": "find maya"}).json()
check(r["ok"] and "meters" in r["say"], f"voice find: {r['say']}")

# --- "blocked": the first stretch of the route is closed -> reroute the long way (via C)
r = c.post("/command", json={"text": "blocked"}).json()
route2 = c.get("/map").json()["route"]
check(r["ok"] and route2 and route2["mode"] == "graph" and route2["path"] != route["path"] and route2["cost_m"] > direct_cost,
      f"blocked -> rerouted: {route2['path'] if route2 else None} ({route2['cost_m'] if route2 else 0} m vs {direct_cost} m)")
check(any(e["blocked"] for e in c.get("/map").json()["edges"]), "blocked edge is marked on the map")

# --- walk the route; seeing her face ends it
wps = route2["waypoints"]
px, py = 0, 0
for wp in wps[:-2]:
    tx = (wp["lng"] - LNG0) * 111320.0 * math.cos(math.radians(LAT0))
    ty = (wp["lat"] - LAT0) * 110540.0
    walk(scene, px, py, tx, ty, noise=0.5)
    px, py = tx, ty
r = frame(scene, px, py)
check(r["nav"] and r["nav"]["remaining"] <= 2, f"waypoints advance as we walk: {r['nav']['remaining']} left, {r['nav']['remaining_m']} m")
r = frame(group, 0, 24)
check(r["nav"] is None and server.geo.arrived and server.geo.arrived["found"] and r["alerts"] and r["display"]["level"] == "alert",
      f"she's in view: route ends (found) + {r['display']['title']}")
r = c.post("/command", json={"text": "report"}).json()
check(r["ok"] and r["report"]["lat"], f"voice report has GPS: {r['say']}")

# --- compass mode: somewhere we never walked
tlat, tlng = ll(120, -60)
r = c.get("/route", params={"lat": tlat, "lng": tlng, "name": "parking"}).json()
check(r["ok"] and r["mode"] == "compass" and 125 <= r["remaining_m"] <= 145, f"off the walked map: compass mode, {r['remaining_m']} m direct")
r = frame(scene, 0, 24, heading=0)
check(r["display"]["title"] == "→ parking" and "direct" in r["display"]["lines"][0], f"compass card: {r['display']['lines'][0]}")
c.delete("/route")

# --- indoors: bad GPS, steps + heading keep the position moving
r = frame(scene, 0, 24, acc=45, heading=90, steps=0)
r = frame(scene, 0, 24, acc=45, heading=90, steps=10)
dx = (r["place"]["lng"] - blng) * 111320.0 * math.cos(math.radians(LAT0))
check(r["place"]["quality"] == "dr" and 5 < dx < 9, f"rough GPS: 10 steps east moved the position {dx:.1f} m by dead reckoning")

# --- manual 'talked' toggle, people pins
c.post(f"/people/{pid0}/met", json={"met": False})
check(not [q for q in c.get("/people").json() if q["id"] == pid0][0]["met"], "talked flag can be undone by hand")
st = c.get("/state").json()
check(st["frames"] >= 40 and len(st["sightings"]) >= 4 and st["map"]["pins"], f"state: {st['frames']} frames, {len(st['sightings'])} sightings, {len(st['map']['pins'])} pins")
print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
