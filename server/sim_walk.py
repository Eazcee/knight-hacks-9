"""Fake GPS walk for the dashboard (no glasses needed).

  python sim_walk.py                 # walks a 40 x 45 m loop at UCF, meets a group, routes back
  python sim_walk.py --lat 28.6 --lng -81.2 --server http://127.0.0.1:8000

Watch http://localhost:8000: the map builds itself, pins appear, a route is drawn,
'blocked' reroutes. Use it to rehearse the demo and to tune the map without walking.
"""
import argparse
import math
import time

import cv2
import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8000")
ap.add_argument("--lat", type=float, default=28.6024)
ap.add_argument("--lng", type=float, default=-81.2001)
ap.add_argument("--fast", action="store_true", help="no real-time delays")
ap.add_argument("--reset", action="store_true", help="wipe the server first")
args = ap.parse_args()
B = args.server
LAT0, LNG0 = args.lat, args.lng

from skimage import data  # noqa: E402
import insightface.data as idata  # noqa: E402

scene = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)
group = idata.get_image("t1")
T = [time.time() - (0 if not args.fast else 600)]


def ll(x, y):
    return LAT0 + y / 110540.0, LNG0 + x / (111320.0 * math.cos(math.radians(LAT0)))


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def frame(img, x, y, dt=1.0, **kw):
    if not args.fast:
        time.sleep(dt * 0.4)
    T[0] += dt
    lat, lng = ll(x, y)
    r = requests.post(B + "/frame", data=jpg(img), headers={"Content-Type": "image/jpeg"},
                      params={"ts": T[0], "lat": lat, "lng": lng, "acc": 4, **kw}, timeout=30).json()
    d = r.get("display") or {}
    print(f"  ({x:5.1f},{y:5.1f}) {d.get('title','')!s:18} {' · '.join(d.get('lines', []))}")
    return r


def walk(img, x0, y0, x1, y1, step=1.3, noise=0.8, seed=1):
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    rng = np.random.default_rng(seed)
    r = None
    for i in range(1, n + 1):
        f = i / n
        r = frame(img, x0 + (x1 - x0) * f + rng.normal(0, noise), y0 + (y1 - y0) * f + rng.normal(0, noise))
    return r


if args.reset:
    requests.post(B + "/reset")
print("walk north 40 m")
frame(scene, 0, 0)
walk(scene, 0, 0, 0, 40)
requests.post(B + "/places/mark", json={"name": "coffee station"})
print("meet a group at the coffee station, talk for ~35 s")
r = frame(group, 0, 40)
pids = [p["person_id"] for p in sorted(r["people"], key=lambda p: p["box"][0])]
for pid, n in zip(pids, ["Ana", "Ben", "Cy", "Dee", "Eli", "Fay"]):
    requests.post(B + f"/people/{pid}", json={"name": n})
for _ in range(14):
    frame(group, 0, 40, dt=2.5)
print("walk east 45 m, south 40 m, back west to the start")
walk(scene, 0, 40, 45, 40)
requests.post(B + "/places/mark", json={"name": "judges table"})
walk(scene, 45, 40, 45, 0)
walk(scene, 45, 0, 0, 0)
print("alert for Cy -> route back to where he was seen")
requests.post(B + f"/people/{pids[2]}/flag", params={"label": "AMBER Alert · Cy"})
r = requests.get(B + "/route", params={"to_person": pids[2]}).json()
print("  route:", r.get("mode"), r.get("path"), r.get("remaining_m"), "m")
frame(scene, 0, 0, heading=0)
if not args.fast:
    time.sleep(3)
print("'blocked' -> reroute")
print(" ", requests.post(B + "/command", json={"text": "blocked"}).json().get("say"))
frame(scene, 0, 0, heading=0)
print("done: open the dashboard. Say/send 'cancel' to drop the route.")
