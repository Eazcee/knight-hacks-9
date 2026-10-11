"""AMBER search area: reach circle, buildings in it (Google Places mocked), search order,
routing building to building, auto-check on arrival, voice, and finding them.

  python test_search.py
"""
import math
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("BEACON_DATA", "data_test_search")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import insightface.data as idata  # noqa: E402
from skimage import data as skd  # noqa: E402

c = TestClient(server.app)
c.post("/reset")
c.post("/map/clear")
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


LAT0, LNG0 = 28.6024, -81.2001


def ll(x, y):
    return LAT0 + y / 110540.0, LNG0 + x / (111320.0 * math.cos(math.radians(LAT0)))


def xy(lat, lng):
    return (lng - LNG0) * 111320.0 * math.cos(math.radians(LAT0)), (lat - LAT0) * 110540.0


# fake Google: 60 buildings scattered within 1.6 km of the start, plus a road
rng = np.random.default_rng(3)
FAKE = []
for i in range(60):
    r, a = 1600 * math.sqrt(rng.random()), rng.random() * 2 * math.pi
    la, ln = ll(r * math.cos(a), r * math.sin(a))
    FAKE.append({"id": f"b{i}", "name": f"Building {i}", "lat": la, "lng": ln, "type": "university"})
calls = []


def fake_nearby(lat, lng, radius, n=20):
    calls.append(radius)
    hits = sorted((b for b in FAKE if server.haversine_m(lat, lng, b["lat"], b["lng"]) <= radius),
                  key=lambda b: server.haversine_m(lat, lng, b["lat"], b["lng"]))
    return [dict(b) for b in hits[:n]]


real_nearby = server._places_nearby
server._places_nearby = fake_nearby
server.settings["google_maps_key"] = "test"

group = idata.get_image("t1")
scene = cv2.cvtColor(skd.coffee(), cv2.COLOR_RGB2BGR)
T = {"now": time.time() - 900}


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def frame(img, x, y, dt=1.0, **kw):
    T["now"] += dt
    lat, lng = ll(x, y)
    r = c.post("/frame", content=jpg(img), headers={"Content-Type": "image/jpeg"},
               params={"ts": T["now"], "lat": lat, "lng": lng, "acc": 4, **kw})
    assert r.status_code == 200, r.text
    return r.json()


t0 = time.time()
# she's seen at the start point 10+ minutes ago, then put on the watchlist
r = frame(group, 0, 0)
pid = sorted(r["people"], key=lambda p: p["box"][0])[2]["person_id"]
c.post(f"/people/{pid}", json={"name": "Maya"})
c.post(f"/people/{pid}/flag", params={"label": "AMBER Alert · Maya, 7"})
frame(scene, 0, 0, dt=60)

# search area for a 10-minute head start: 1.5 m/s x 600 s = 900 m
z = c.post(f"/search/start/{pid}", json={"minutes": 10}).json()
inside = [b for b in FAKE if server.haversine_m(z["lat"], z["lng"], b["lat"], b["lng"]) <= 900]
check(z["radius_m"] == 900, f"search circle: {z['radius_m']} m for 10 min on foot")
check(z["n"] == len(inside) and z["n"] > 5, f"every building inside the circle: {z['n']} (truth {len(inside)}) in {len(calls)} lookups")
d = [b["dist"] for b in z["buildings"]]
rings = [x // 150 for x in d]
check(rings == sorted(rings), "search order goes outward from where she was last seen (150 m rings)")
check(z["target"] and z["target"]["id"] == z["buildings"][0]["id"] or z["target"]["dist"] < 150, f"first target: {z['target']['name']} ({z['target']['dist']} m)")
r = frame(scene, 0, 0, heading=0)
check(r["display"]["title"] == "→ " + z["target"]["name"] and any("search 0/" in l for l in r["display"]["lines"]),
      f"glasses: {r['display']['title']} / {r['display']['lines']}")

# walk to the target: it gets checked, the next one is routed automatically
tx, ty = xy(z["target"]["lat"], z["target"]["lng"])
first = z["target"]["id"]
n = 12
for i in range(1, n + 1):
    r = frame(scene, tx * i / n, ty * i / n)
z2 = c.get("/search").json()
check(first in [b["id"] for b in z2["buildings"] if b["checked"]], "arriving at the building checks it off")
check(z2["target"] and z2["target"]["id"] != first and r["display"]["title"].startswith("→ "), f"…and routes to the next: {z2['target']['name'] if z2['target'] else None}")

# voice: nothing here -> next
before = z2["n_checked"]
v = c.post("/command", json={"text": "nothing here"}).json()
z3 = c.get("/search").json()
check(v["ok"] and z3["n_checked"] == before + 1 and "Next:" in v["say"], f"voice: {v['say']}")

# dashboard: pick a building by hand, uncheck one
far = z3["buildings"][-1]
z4 = c.post("/search/go", json={"id": far["id"]}).json()
check(z4["target"]["id"] == far["id"] and c.get("/map").json()["route"]["target"]["name"] == far["name"], f"go to a chosen building: {far['name']}")
z5 = c.post("/search/check", json={"id": first, "unchecked": True}).json()
check(first not in [b["id"] for b in z5["buildings"] if b["checked"]], "a building can be un-checked")

# no key / never seen
server.settings["google_maps_key"] = ""
server._places_nearby = real_nearby
keep = server.search
check(c.post(f"/search/start/{pid}").status_code == 409, "no Google key: clear error")
server.search = keep
server._places_nearby = fake_nearby
server.settings["google_maps_key"] = "test"

# she shows up: route stops, alert fires
r = frame(group, tx, ty)
check(r["alerts"] and r["display"]["level"] == "alert" and r["nav"] is None, f"found her: {r['display']['title']}")
st = c.get("/state").json()
check(st["search"] and st["search"]["n"] == z["n"], "search in dashboard state")
c.delete("/search")
check(c.get("/state").json()["search"] is None and c.get("/map").json()["route"] is None, "end search")

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
