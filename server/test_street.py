"""Street view of your walks + indoor positioning: GPS ignored indoors, distance from steps,
direction from the camera (turns); photos only while walking (~every 0.8 m, the sharpest frame),
moments when a known person appears (strangers blurred), floors from the barometer, indoors
detected by itself, and the incremental /shots API the dashboard uses.

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


T = {"now": 1_000_000.0, "steps": 0, "i": 0, "truth": [0.0, 0.0], "dir": 90.0, "path": 0.0}
FPS = 7.0


def frame(yaw=0.0, step=True, compass=None, gps=True, length=20.0, dt=1.0 / FPS, pressure=None, img=None, moving=None, acc=10):
    """One processed frame. yaw = where the camera looks relative to the walking direction.
    Walking: a step every 4 frames (1.75 steps/s, 1.3 m/s). GPS: the phone's indoor guess, 10 m acc, up to 8 m off."""
    T["now"] += dt
    T["i"] += 1
    params = {"ts": T["now"]}
    if step and T["i"] % 4 == 0:
        T["steps"] += 1
        T["truth"][0] += STEP * math.sin(math.radians(T["dir"]))
        T["truth"][1] += STEP * math.cos(math.radians(T["dir"]))
        T["path"] += STEP
    params["steps"] = T["steps"]
    if compass is not None:
        params["heading"] = compass % 360
    if pressure is not None:
        params["pressure"] = pressure
    if moving is not None:
        params["moving"] = int(moving)
    if gps and (T["i"] % 5 == 0 or T["i"] == 1):
        noise = 5 if acc > 6 else 1
        params["lat"], params["lng"] = ll(T["truth"][0] + rng.normal(0, noise), T["truth"][1] + rng.normal(0, noise))
        params["acc"] = acc
    if img is None:
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

# 5. street view: photos only while walking, ~every 0.8 m, each with position + facing; served; incremental
sv = c.get("/shots").json()
walked = T["path"]
check(walked / 1.1 <= sv["n"] <= walked / 0.5, f"{sv['n']} photos along {walked:.0f} m walked (one every ~0.6-0.8 m)")
d = [math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for a, b in zip(sv["shots"], sv["shots"][1:])]
ds = sorted(d)
check(ds[len(ds) // 2] <= 1.0 and ds[int(len(ds) * 0.95)] <= 1.6, f"evenly spaced: median {ds[len(ds) // 2]:.2f} m, 95% within {ds[int(len(ds) * 0.95)]:.1f} m (the corner itself can skip a few m)")
n_before = sv["n"]
for _ in range(14):       # stopping: the step counter lags, so a photo or two can still come in the first 2 s...
    frame(step=False, compass=90 + 60 * math.sin(T["i"] / 3.0), yaw=-90 + 30 * math.sin(T["i"] / 4.0))
n_stop = c.get("/shots").json()["n"]
for _ in range(56):       # ...then sitting 8 s, looking around (compass swings, head turns): none
    frame(step=False, compass=90 + 60 * math.sin(T["i"] / 3.0), yaw=-90 + 30 * math.sin(T["i"] / 4.0))
check(n_stop - n_before <= 2 and c.get("/shots").json()["n"] == n_stop, f"stopped: {n_stop - n_before} photo(s) while the step counter caught up, then none while sitting and looking around")
walk(2, yaw=-90, compass=0)
check(c.get("/shots").json()["n"] >= n_before + 2, "walking again: photos again")
# the phone's accelerometer flag beats the step counter (it's instant; step counters lag)
n1 = c.get("/shots").json()["n"]
for _ in range(14):
    frame(step=True, moving=0, yaw=-90, compass=0)
check(c.get("/shots").json()["n"] == n1, "moving=0 from the phone (standing, shuffling): no photos even if a step ticks")
for _ in range(14):
    frame(step=False, moving=1, yaw=-90, compass=0)
check(c.get("/shots").json()["n"] >= n1 + 2, "moving=1 (walking, step counter lagging): photos")
sv = c.get("/shots").json()
hs = [s["heading"] for s in sv["shots"]]
check(hs[0] == 90 and hs[-1] == 0 and all(s["x"] is not None and s["lat"] is not None for s in sv["shots"]),
      f"photos know where they were taken (x/y and lat/lng) and which way: {hs[0]}° … {hs[-1]}°")
east = [s for s in sv["shots"] if s["heading"] == 90]
north = [s for s in sv["shots"] if s["heading"] == 0]
check(all(abs(s["y"]) < 1.5 for s in east) and all(abs(s["x"] - T["truth"][0]) < 2.5 for s in north),
      f"photos taken mid-turn were moved onto the new hallway ({len(east)} east, {len(north)} north)")
check(len({s["walk"] for s in sv["shots"]}) == 1 and all(s["kind"] == "walk" for s in sv["shots"]), "one walk so far")
im = c.get(f"/shots/{sv['shots'][0]['id']}.jpg")
check(im.status_code == 200 and im.headers["content-type"] == "image/jpeg" and len(im.content) > 5000, f"photo served ({len(im.content) // 1024} KB)")
ep, last = sv["epoch"], sv["shots"][-1]["id"]
walk(1.5, yaw=-90, compass=0)
inc = c.get("/shots", params={"epoch": ep, "after": last}).json()
check(not inc["full"] and inc["shots"] and all(s["id"] > last for s in inc["shots"]) and inc["n"] == sv["n"] + len(inc["shots"]),
      f"incremental: only the {len(inc['shots'])} new photos come back")
r = c.post("/shots/take").json()
check(r["ok"] and r["shot"]["kind"] == "manual", "photo on demand (button / voice 'photo')")
v = c.post("/command", json={"text": "photo"}).json()
check(v["ok"] and "Photo saved" in v["say"], f"voice: {v['say']}")
dl = c.delete(f"/shots/{r['shot']['id']}")
again = c.get("/shots", params={"epoch": ep, "after": last}).json()
check(dl.status_code == 200 and again["full"] and r["shot"]["id"] not in [s["id"] for s in again["shots"]], "delete a photo: gone, and the dashboard is told to refetch")
v = c.post("/command", json={"text": "pause photos"}).json()
n2 = c.get("/shots").json()["n"]
walk(2, yaw=-90, compass=0)
check(v["ok"] and c.get("/shots").json()["n"] == n2 and c.get("/state").json()["photos_paused"], "voice 'pause photos': walking takes none")
c.post("/command", json={"text": "resume photos"})

# 5b. the sharpest frame of each stretch is the one kept (walking blurs frames)
import shots as shots_mod  # noqa: E402
sh = shots_mod.Shots(os.path.join(server.DATA, "shots_unit"))
sh.clear()
sharp = hallway(seed=0)
blurry = cv2.GaussianBlur(sharp, (0, 0), 6)
t_ = 5000.0
for k in range(6):
    sh.offer(sharp if k == 3 else blurry, [], [], {"x": k * 0.2, "y": 0, "lat": 1.0, "lng": 1.0, "ts": t_ + k * 0.15}, t_ + k * 0.15, True, 1.3)
it = sh.offer(blurry, [], [], {"x": 9, "y": 0, "lat": 1.0, "lng": 1.0, "ts": t_ + 1.0}, t_ + 1.0, True, 1.3)
kept = cv2.imread(sh.file(sh.items[-1]["id"])) if sh.items else None
check(kept is not None and shots_mod.sharpness(kept) > 3 * shots_mod.sharpness(cv2.resize(blurry, (kept.shape[1], kept.shape[0]))),
      "of a blurry stretch, the one sharp frame is the photo kept")
sh.clear()

# 5c. anyone comes into view: a moment photo right away (walking or not); everyone boxed (named or "Unknown"), nothing blurred
import insightface.data as idata  # noqa: E402
group = idata.get_image("t1")
n_first = c.get("/shots").json()["n"]
r = frame(img=group, step=False, compass=0)
ppl = sorted([p for p in r["people"] if p["confirmed"]], key=lambda p: p["box"][0])
first = c.get("/shots").json()["shots"][n_first:]
check(any(s["kind"] == "person" and len(s["people"]) >= 4 for s in first), f"strangers coming into view (standing still) → a photo right away, all of them in it ({[len(s['people']) for s in first]})")
check(len(ppl) >= 4, f"{len(ppl)} faces in the group photo")
named, stranger = ppl[1], ppl[3]
c.post(f"/people/{named['person_id']}", json={"name": "Dana"})
n3 = c.get("/shots").json()["n"]
for _ in range(3):
    r = frame(img=group, step=False, compass=0)
T["now"] += 70     # Dana out of view for a minute, then back
r = frame(img=group, step=False, compass=0)
sv = c.get("/shots").json()
moms = [s for s in sv["shots"][n3:] if s["kind"] == "person"]
names = [p["name"] for p in moms[0]["people"]] if moms else []
check(len(moms) == 1 and "Dana" in names and names.count(None) >= 3 and all("box" in p and "known" in p for p in moms[0]["people"]),
      f"back into view a minute later → one more photo: Dana named, the others as Unknown ({names})")
m = moms[0] if moms else sv["shots"][-1]
dp = [p for p in m["people"] if p["name"] == "Dana"]
b = dp[0]["box"] if dp else [0, 0, 0, 0]
check(0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1 and abs((b[0] + b[2]) / 2 - (named["box"][0] + named["box"][2]) / 2 / group.shape[1]) < 0.02,
      f"her box is where her face is (0-1 of the photo): {b}")
photo = cv2.imread(server.shots.file(m["id"]))
H, W = photo.shape[:2]
gs = cv2.resize(group, (W, H))


def region_detail(img, box):
    x1, y1, x2, y2 = [int(v) for v in (box[0] * W / group.shape[1], box[1] * H / group.shape[0], box[2] * W / group.shape[1], box[3] * H / group.shape[0])]
    return float(cv2.Laplacian(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


check(region_detail(photo, stranger["box"]) > 0.5 * region_detail(gs, stranger["box"]) and region_detail(photo, named["box"]) > 0.5 * region_detail(gs, named["box"]),
      f"nobody is blurred: the stranger's face keeps its detail ({region_detail(photo, stranger['box']):.0f} vs {region_detail(gs, stranger['box']):.0f} original)")
check(any(p["id"] == stranger["person_id"] and p["name"] is None and not p["known"] for p in m["people"]), "the stranger is in the photo's people list as an unknown person (so the timeline can list them)")
mo = c.get(f"/people/{named['person_id']}/moments").json()
check(mo and mo[0]["id"] == m["id"], "Dana's card → her moments (the street view where you saw her)")
at = c.get("/shots/at", params={"ts": m["ts"] + 5, "person": named["person_id"]}).json()
check(at["id"] == m["id"], "a sighting time → the photo with her in it")

# 5d. floors: the barometer sees ~0.4 hPa less per floor up; the stairs join the floors; voice too
f0 = c.get("/state").json()["map"]["position"]["floor"]
for i in range(20):
    frame(compass=0, pressure=1013.0, yaw=-90)
for i in range(40):
    frame(compass=0, pressure=1013.0 - 0.42 * min(1.0, i / 25), yaw=-90)
p = c.get("/state").json()["map"]
check(p["position"]["floor"] == f0 + 1 and p["position"]["barometer"], f"barometer: pressure dropped 0.42 hPa → floor {p['position']['floor']} (was {f0})")
stairs = [n for n in p["nodes"] if str(n.get("name") or "").startswith("Stairs")]
check(len(stairs) >= 1 and any((n.get("floor") or 1) == f0 + 1 for n in stairs), f"a stairs place on the new floor, joined to the old one: {[n['name'] for n in stairs]}")
walk_m(6, compass=0, pressure=1013.0 - 0.42, yaw=-90)
sv2 = c.get("/shots").json()
check(sv2["shots"][-1]["floor"] == f0 + 1 and any(s.get("stairs") for s in sv2["shots"]),
      f"photos carry the floor; the ones on the stairs are marked ({sum(1 for s in sv2['shots'] if s.get('stairs'))} stairs photos)")
up = [n for n in c.get("/state").json()["map"]["nodes"] if (n.get("floor") or 1) == f0 + 1]
check(len(up) >= 2, f"the new floor gets its own places ({len(up)}), separate from the floor below at the same spot")
v = c.post("/command", json={"text": "downstairs"}).json()
check(v["ok"] and c.get("/state").json()["map"]["position"]["floor"] == f0, f"voice: {v['say']}")
n_st = sum(1 for s in c.get("/shots").json()["shots"] if s.get("stairs"))
T["now"] += 20     # (the "downstairs" just now was real stairs: photos within 15 s of it are stairs photos)
fx = c.post("/floor", json={"floor": 3, "moved": False}).json()
walk(2, compass=0, yaw=-90)
sv_fx = c.get("/shots").json()["shots"]
check(fx["floor"] == 3 and sv_fx[-1]["floor"] == 3 and not sv_fx[-1].get("stairs") and fx["event"].get("node_id") is None,
      "a floor correction from the dashboard (\"I'm on 3\") is not a staircase: no stairs place, photos not marked")
c.post("/floor", json={"floor": f0, "moved": False})
g2 = geo_mod.GeoMap(server.db)                         # weather: 0.3 hPa over 20 minutes is not a floor
for k in range(1200):
    g2.pressure(1013.0 - 0.3 * k / 1200, 9000.0 + k, [])
check(g2.floor == 1, f"slow weather drift (0.3 hPa over 20 min) doesn't change the floor (floor {g2.floor})")
ev = []
for k in range(30):
    g2.pressure(1012.7 - 0.45 * min(1, k / 15), 10300.0 + k, ev)
check(g2.floor == 2, "…but the same barometer then catches a real floor in 15 s")
g3 = geo_mod.GeoMap(server.db)                         # an elevator: 3 floors up in 8 s, then 2 down
for k in range(40):
    g3.pressure(1013.0 - 1.2 * min(1, max(0, k - 10) / 8), 20000.0 + k * 0.5, [])
up3 = g3.floor
for k in range(40):
    g3.pressure(1011.8 + 0.8 * min(1, max(0, k - 10) / 6), 20020.0 + k * 0.5, [])
check(up3 == 4 and g3.floor == 2, f"elevator: 3 floors up → floor {up3}, 2 down → floor {g3.floor}")
g4 = geo_mod.GeoMap(server.db)                         # standing on a landing halfway up, the sensor jittering
nrng = np.random.default_rng(1)
flips = 0
for k in range(300):
    f_before = g4.floor
    g4.pressure(1013.0 - (0.0 if k < 20 else 0.2) + nrng.normal(0, 0.03), 30000.0 + k * 0.5, [])
    flips += g4.floor != f_before
check(flips == 0, f"halfway up the stairs (on the landing) the floor doesn't flicker ({flips} changes)")
st0 = g2.stairs_ts
g2.pressure(1012.69, 10300.0 + 30 + 900, [])          # 15 min later (glasses were off): back on the ground floor, same weather
check(g2.floor == 1 and g2.stairs_ts == st0, "after a break the floor is updated, but no stairs are invented")

# 5e. a long break starts a new walk
T["now"] += 700
walk(2, compass=0, yaw=-90)
sv3 = c.get("/shots").json()
check(len({s["walk"] for s in sv3["shots"]}) >= 2, f"a 10-minute break starts a new walk (walks {sorted({s['walk'] for s in sv3['shots']})})")

# 6. outdoors again: a good GPS fix is trusted
v = c.post("/command", json={"text": "outdoors"}).json()
T["now"] += 1
r = c.post("/frame", content=jpg(hallway(0)), headers={"Content-Type": "image/jpeg"},
           params={"ts": T["now"], "lat": ll(0, 0)[0], "lng": ll(0, 0)[1], "acc": 4, "steps": T["steps"]}).json()
check(v["ok"] and r["place"]["quality"] == "good" and not c.get("/state").json()["indoors"], f"voice 'outdoors': {v['say']} → GPS trusted again")

# 7. outdoors, phone sideways in a pocket: its compass reads 90° off; walking north with good GPS teaches the server the offset
for i in range(1, 16):
    T["now"] += 1
    r = c.post("/frame", content=jpg(hallway(i % 3)), headers={"Content-Type": "image/jpeg"},
               params={"ts": T["now"], "lat": ll(0, i * 1.3)[0], "lng": ll(0, i * 1.3)[1], "acc": 4, "steps": T["steps"], "heading": 90}).json()
p = c.get("/state").json()["map"]["position"]
check(p["heading_src"] == "sensor" and (abs(p["heading"]) < 15 or abs(p["heading"] - 360) < 15) and abs(((p["compass_offset"] + 90 + 180) % 360) - 180) < 15,
      f"phone sideways in a pocket: compass 90° but walking north → heading {round(p['heading'])}° (learned offset {p['compass_offset']}°)")
c.post("/map/clear")
check(c.get("/shots").json()["n"] == 0, "clear map clears the photos too")

# 8. indoors by itself: walking with only 10 m fixes for 15 s → indoors (camera turns + steps); good fixes → outdoors
c.post("/settings", json={"indoors": False})
T["truth"], T["dir"] = [0.0, 0.0], 90.0
for _ in range(10):
    frame(acc=4, compass=90)
p = c.get("/state").json()["map"]["position"]
check(not p["indoors"], "good GPS: outdoors")
evs = []
for _ in range(int(20 * FPS)):
    evs += frame(acc=12, compass=90)["place"]["events"]
p = c.get("/state").json()["map"]["position"]
check(p["indoors"] and p["indoors_auto"] and any(e["type"] == "indoors" for e in evs), "15 s of walking on 12 m fixes → indoors by itself")
evs = []
for _ in range(int(3 * FPS)):
    evs += frame(acc=4, compass=90)["place"]["events"]
p = c.get("/state").json()["map"]["position"]
check(not p["indoors"] and any(e["type"] == "outdoors" for e in evs), "three good fixes → outdoors again")

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
