"""Depth from the camera: distances to people, point cloud, obstacle -> automatic reroute.

  python test_depth.py
"""
import math
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("BEACON_DATA", "data_test_depth")
os.environ.setdefault("BEACON_DEPTH", "1")      # depth is off by default now; this test turns it on
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import insightface.data as idata  # noqa: E402
from skimage import data as skd  # noqa: E402

c = TestClient(server.app)
c.post("/reset")
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


LAT0, LNG0 = 28.6024, -81.2001


def ll(x, y):
    return LAT0 + y / 110540.0, LNG0 + x / (111320.0 * math.cos(math.radians(LAT0)))


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


T = {"now": 1_000_000.0}


def frame(img, x=None, y=None, dt=1.0, **kw):
    T["now"] += dt
    params = {"ts": T["now"], **kw}
    if x is not None:
        params["lat"], params["lng"] = ll(x, y)
        params["acc"] = 4
    r = c.post("/frame", content=jpg(img), headers={"Content-Type": "image/jpeg"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def walk(img, x0, y0, x1, y1, step=1.3):
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    rng = np.random.default_rng(0)
    r = None
    for i in range(1, n + 1):
        f = i / n
        r = frame(img, x0 + (x1 - x0) * f + rng.normal(0, 0.8), y0 + (y1 - y0) * f + rng.normal(0, 0.8))
    return r


t0 = time.time()
check(server.depth is not None, f"depth engine loaded on {server.depth.device if server.depth else '-'}")
group = idata.get_image("t1")
scene = cv2.cvtColor(skd.coffee(), cv2.COLOR_RGB2BGR)

# 1. faces -> distances, scale, cloud
r = frame(group, 0, 0)
dists = [p.get("dist_m") for p in r["people"]]
check(all(d for d in dists) and all(0.5 <= d <= 8 for d in dists), f"every face has a measured distance: {dists}")
st = c.get("/state").json()
d = st["depth"]
check(d["calibrated"] and d["scale"] and len(d["faces"]) == len(r["people"]), f"depth: {d['device']} {d['ms']} ms, scale from faces, {len(d['faces'])} faces in 3D")
check(all(len(f["xyz"]) == 3 and abs(f["xyz"][2] - f["dist_m"]) < 0.3 for f in d["faces"]), "faces placed in 3D at their distance")
b = c.get("/cloud.bin")
check(b.status_code == 200 and b.headers["content-type"].startswith("application/octet-stream"), "point cloud served")
buf = b.content
hdr = np.frombuffer(buf[:16], np.uint32)
n, cw, ch = int(hdr[1]), int(hdr[2]), int(hdr[3])
off = 16 + 4 + 8
xyz = np.frombuffer(buf[off:off + n * 12], np.float32).reshape(n, 3)
rgb = np.frombuffer(buf[off + n * 12:off + n * 15], np.uint8).reshape(n, 3)
check(hdr[0] == 0xBEAC0001 and n > 1000 and len(buf) == off + n * 15, f"cloud layout ok: {n} points ({cw}x{ch})")
check(0.2 <= float(np.median(xyz[:, 2])) <= 60 and rgb.max() > 0, f"cloud in meters (median depth {np.median(xyz[:, 2]):.1f} m) with colors")

# 2. obstacle: a route is active, we walk toward the next place, and a wall fills the view
walk(scene, 0, 0, 0, 30)
c.post("/places/mark", json={"name": "Library"})
walk(scene, 0, 30, 30, 30)
walk(scene, 30, 30, 30, 0)
walk(scene, 30, 0, 0, 0)
r = c.get("/route", params={"to_name": "Library"}).json()
path0 = r["path"]
check(r["ok"] and r["mode"] == "graph", f"route to the Library over the walked map: {path0}")
wall = np.full((1280, 720, 3), 90, np.uint8)
cv2.rectangle(wall, (0, 0), (720, 1280), (60, 60, 70), -1)
noise = np.random.default_rng(1).integers(0, 30, (1280, 720, 3), dtype=np.uint8)
wall = cv2.add(wall, noise)
# fake the depth net for determinism: a wall 1 m ahead across the middle of the view, the ceiling/sky above it far away
real_infer = server.depth.infer
fake_rel = np.full((128, 72), 40.0, np.float32)
fake_rel[:30] = 5.0              # top of the view: 8 m
fake_rel[115:] = 20.0            # floor at the bottom: 2 m (that's what keeps the scale without faces)
server.depth.infer = lambda img: (fake_rel.copy(), np.zeros((128, 72, 3), np.uint8))
server.depth.scale = 40.0 * 1.0   # 1 m
server.depth.face_ts = time.time()
res, evs = None, []
for i in range(4):
    res = frame(wall, 0.5, 1 + i, heading=0)
    evs += [e for e in res["place"]["events"] if e["type"] == "obstacle"]
check(len(evs) == 1, f"wall ahead for 3 frames -> one obstacle event: {evs}")
m = c.get("/map").json()
check(any(e["blocked"] for e in m["edges"]), "the stretch ahead is blocked on the map")
check(m["route"] and m["route"]["path"] != path0 and m["route"]["path"][-1] == path0[-1], f"rerouted the other way: {m['route']['path']}")
check(res["display"]["lines"][0].startswith("BLOCKED AHEAD"), f"glasses: {res['display']['lines'][0]}")
# no re-block within the cooldown
blocked_n = sum(1 for e in m["edges"] if e["blocked"])
for i in range(4):
    res = frame(wall, 0.5, 5 + i, heading=0)
check(sum(1 for e in c.get("/map").json()["edges"] if e["blocked"]) == blocked_n, "cooldown: not blocking again right away")
server.depth.infer = real_infer

# 2b. a frame where *everything* reads near (the net's scale is off) must not block anything
server.depth.near_streak = 0
server.depth.infer = lambda img: (np.full((128, 72), 40.0, np.float32), np.zeros((128, 72, 3), np.uint8))
server.depth.face_ts = time.time()
for i in range(4):
    frame(wall, 0.5, 9 + i, heading=0)
ob = c.get("/state").json()["depth"]["obstacle"]
check(not ob["blocked"], f"whole frame 'near' (scale off) is not an obstacle (near={ob['near']}, blocked={ob['blocked']})")
server.depth.infer = real_infer

# 3. faces close up are not obstacles
server.depth.near_streak = 0
r = frame(cv2.resize(group, (720, 498)), 0.5, 9, heading=0)
ob = c.get("/state").json()["depth"]["obstacle"]
check(not ob["blocked"] and ob["people"], f"a face in front is a person, not a wall (people={ob['people']})")

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
