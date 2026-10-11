"""Indoors, for real: GPS ignored, distance from steps, direction from the camera (turns), a
photo every ~3 m, and the street-view endpoints.

  python test_street.py
"""
import math
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("BEACON_DATA", "data_test_street")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata"))
import server  # noqa: E402
import geo as geo_mod  # noqa: E402
from synth import hallway  # noqa: E402

c = TestClient(server.app)
c.post("/reset")
c.post("/settings", json={"indoors": False})
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


LAT0, LNG0 = 28.6024, -81.2001
STEP = geo_mod.STEP_M
rng = np.random.default_rng(5)


def ll(x, y):
    return LAT0 + y / 110540.0, LNG0 + x / (111320.0 * math.cos(math.radians(LAT0)))


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


T = {"now": 1_000_000.0, "steps": 0, "i": 0, "truth": [0.0, 0.0], "dir": 90.0}
FPS = 7.0


def frame(yaw=0.0, step=True, compass=None, gps=True, length=20.0, dt=1.0 / FPS):
    """One processed frame. yaw = where the camera looks relative to the walking direction.
    Walking: a step every 4 frames (1.75 steps/s, 1.3 m/s). GPS: the phone's indoor guess, 10 m acc, up to 8 m off."""
    T["now"] += dt
    T["i"] += 1
    params = {"ts": T["now"]}
    if step and T["i"] % 4 == 0:
        T["steps"] += 1
        T["truth"][0] += STEP * math.sin(math.radians(T["dir"]))
        T["truth"][1] += STEP * math.cos(math.radians(T["dir"]))
    params["steps"] = T["steps"]
    if compass is not None:
        params["heading"] = compass % 360
    if gps and (T["i"] % 5 == 0 or T["i"] == 1):
        params["lat"], params["lng"] = ll(T["truth"][0] + rng.normal(0, 5), T["truth"][1] + rng.normal(0, 5))
        params["acc"] = 10
    img = hallway(seed=T["i"] % 3, length=length, yaw=yaw)
    r = c.post("/frame", content=jpg(img), headers={"Content-Type": "image/jpeg"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def walk(seconds, **kw):
    out = None
    for _ in range(int(seconds * FPS)):
        out = frame(**kw)
    return out


def walk_m(meters, **kw):
    out = None
    for _ in range(int(round(meters / STEP * 4))):
        out = frame(**kw)
    return out


t0 = time.time()
# 1. indoors on; the phone's first fix seeds the position; the compass (90 = east) sets the start direction
c.post("/settings", json={"indoors": True})
r = frame(compass=90, step=False)
r = walk(2, compass=90 + rng.normal(0, 8))
check(r["place"]["quality"] in ("dr", "rough") and r["place"]["heading"] == 90 and r["place"]["heading_src"] == "camera",
      f"indoors: compass only sets the start ({r['place']['heading']}°), then the camera owns the heading ({r['place']['heading_src']})")

# 2. 20 m straight east with the phone's GPS jumping ±8 m and the compass wobbling ±8°: a straight line
for _ in range(int(round(20 / STEP * 4))):
    frame(compass=90 + rng.normal(0, 8), length=max(4, 24 - T["truth"][0]))
p = c.get("/state").json()["map"]["position"]
check(abs(p["x"] - T["truth"][0]) < 1.0 and abs(p["y"]) < 0.5 and p["quality"] == "dr",
      f"20 m straight: you're at ({p['x']:.1f}, {p['y']:.1f}), truth ({T['truth'][0]:.1f}, 0) — GPS jumps ignored, line straight")
st = c.get("/state").json()
nodes = st["map"]["nodes"]
ys = [n["y"] for n in nodes]
check(2 <= len(nodes) <= 4 and max(abs(v) for v in ys) < 1.0, f"{len(nodes)} places, all on the line (|y| < 1 m)")

# 3. a glance: head turns 50° left for a second and back, the phone doesn't turn — not a turn
for i in range(7):
    frame(yaw=-7 * (i + 1), compass=90)
for i in range(7):
    frame(yaw=-50, compass=90)
for i in range(7):
    frame(yaw=-50 + 7 * (i + 1), compass=90)
r = walk(2.5, compass=90)
check(r["place"]["heading"] == 90, f"a glance to the side is not a turn (heading still {r['place']['heading']}°)")

# 4. a real corner: the camera turns 90° left over 1.5 s and stays, the phone turns too
x_corner = c.get("/state").json()["map"]["position"]["x"]
for i in range(10):
    if i == 5:
        T["dir"] = 0.0      # the body comes round mid-turn
    frame(yaw=-9 * (i + 1), compass=90 - 9 * (i + 1))
r = walk(2.5, yaw=-90, compass=0 + rng.normal(0, 8))
check(r["place"]["heading"] == 0 and any(e["type"] == "turn" for e in r["place"]["events"]) or r["place"]["heading"] == 0,
      f"turned the corner: heading {r['place']['heading']}° (camera said {r['place']['heading_src']})")
# ...then 10 m north
walk_m(10, yaw=-90, compass=0 + rng.normal(0, 8))
p = c.get("/state").json()["map"]["position"]
check(abs(p["x"] - T["truth"][0]) < 2.0 and abs(p["y"] - T["truth"][1]) < 2.0,
      f"10 m after the corner: ({p['x']:.1f}, {p['y']:.1f}), truth ({T['truth'][0]:.1f}, {T['truth'][1]:.1f}) — an L, like the hallway")
nodes = c.get("/state").json()["map"]["nodes"]
stray = [n for n in nodes if n["x"] > T["truth"][0] + 3]
check(not stray, f"no stray place left east of the corner ({len(stray)})")

# 5. street view: a photo every ~3 m, each with position + heading; served; who was there
sv = c.get("/shots").json()
walked = math.hypot(*T["truth"]) + 4
check(walked / 4.5 <= sv["n"] <= walked / 2.0, f"{sv['n']} photos along {walked:.0f} m (one every ~3 m)")
hs = [s["heading"] for s in sv["shots"]]
check(hs[0] == 90 and hs[-1] == 0 and all(s["x"] is not None for s in sv["shots"]), f"photos know where they were taken and which way: {hs[0]}° … {hs[-1]}°")
east = [s for s in sv["shots"] if s["heading"] == 90]
north = [s for s in sv["shots"] if s["heading"] == 0]
check(all(abs(s["y"]) < 1.5 for s in east) and all(abs(s["x"] - T["truth"][0]) < 2.5 for s in north),
      f"photos taken mid-turn were moved onto the new hallway ({len(east)} east, {len(north)} north)")
d = [math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for a, b in zip(sv["shots"], sv["shots"][1:])]
check(all(v <= 5 for v in d), f"photos spaced evenly (max gap {max(d):.1f} m)")
im = c.get(f"/shots/{sv['shots'][0]['id']}.jpg")
check(im.status_code == 200 and im.headers["content-type"] == "image/jpeg" and len(im.content) > 5000, f"photo served ({len(im.content) // 1024} KB)")
r = c.post("/shots/take").json()
check(r["ok"] and c.get("/shots").json()["n"] == sv["n"] + 1, "photo on demand (button / voice 'photo')")
v = c.post("/command", json={"text": "photo"}).json()
check(v["ok"] and "Photo saved" in v["say"], f"voice: {v['say']}")
check(c.get("/state").json()["n_shots"] == sv["n"] + 2 and c.get("/state").json()["indoors"], "dashboard state has the count and the indoors flag")

# 6. outdoors again: a good GPS fix is trusted
v = c.post("/command", json={"text": "outdoors"}).json()
T["now"] += 1
r = c.post("/frame", content=jpg(hallway(0)), headers={"Content-Type": "image/jpeg"},
           params={"ts": T["now"], "lat": ll(0, 0)[0], "lng": ll(0, 0)[1], "acc": 4, "steps": T["steps"]}).json()
check(v["ok"] and r["place"]["quality"] == "good" and not c.get("/state").json()["indoors"], f"voice 'outdoors': {v['say']} → GPS trusted again")
c.post("/map/clear")
check(c.get("/shots").json()["n"] == 0, "clear map clears the photos too")

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
