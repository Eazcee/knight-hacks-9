"""Indoors: no usable GPS, so position comes from the glasses' compass + the phone's step
counter (dead reckoning). Every new place gets a 3D snapshot of the depth cloud, turned
the way the wearer faced and pinned where they stood: together they're a rough 3D model
of the hallway.

  python test_world.py
"""
import math
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("BEACON_DATA", "data_test_world")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import geo as geo_mod  # noqa: E402
import world as world_mod  # noqa: E402
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
STEP = geo_mod.STEP_M


def ll(x, y):
    return LAT0 + y / 110540.0, LNG0 + x / (111320.0 * math.cos(math.radians(LAT0)))


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


T = {"now": 1_000_000.0, "steps": 0}


def frame(img, dt=1.0, fix=None, acc=45, **kw):
    """One frame from the glasses. fix=(x, y) sends a rough phone fix (acc m); no fix = none this frame."""
    T["now"] += dt
    params = {"ts": T["now"], **kw}
    if fix is not None:
        params["lat"], params["lng"] = ll(*fix)
        params["acc"] = acc
    r = c.post("/frame", content=jpg(img), headers={"Content-Type": "image/jpeg"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def walk(img, heading, meters, per_frame=2):
    """Walk `meters` facing `heading`, `per_frame` steps per frame, with the phone's rough fix every 3rd frame."""
    n = max(1, int(round(meters / (per_frame * STEP))))
    out, evs = None, []
    for i in range(n):
        T["steps"] += per_frame
        fix = (0, 0) if i % 3 == 0 else None
        out = frame(img, heading=heading, steps=T["steps"], fix=fix)
        evs += out["place"]["events"]
    return out, evs


t0 = time.time()
check(server.depth is not None, f"depth engine loaded on {server.depth.device if server.depth else '-'}")
hall = cv2.cvtColor(skd.coffee(), cv2.COLOR_RGB2BGR)
hall = cv2.resize(hall, (720, 1280))
group = idata.get_image("t1")

# 1. only a rough fix (45 m): a position, but no map yet (a 45 m guess shouldn't draw places)
r = frame(group, fix=(0, 0), heading=90, steps=0)
check(r["place"]["quality"] == "rough" and r["place"]["node_id"] is None, f"rough fix alone: position known, no place made ({r['place']['quality']})")
check(r["place"]["heading"] == 90 and r["place"]["heading_src"] == "sensor", "compass heading from the glasses")

# 2. steps + heading: dead reckoning takes over, the first place appears and gets a snapshot
r, evs = walk(hall, 90, 1.5)
check(r["place"]["quality"] == "dr" and r["place"]["node_id"] is not None, f"steps + compass -> dead reckoning, first place ({r['place']['quality']})")
check(any(e["type"] == "snapshot" for e in evs), f"new place -> 3D snapshot: {[e['type'] for e in evs]}")
st = c.get("/state").json()
check(st["world_stamps"] == 1, f"dashboard knows: {st['world_stamps']} snapshot")

# 3. walk 24 m east down the hallway: ~3 places, each with a snapshot, position = steps x 0.72 m
r, evs = walk(hall, 90, 24)
wj = c.get("/world.json").json()
you = wj["you"]
expect = T["steps"] * STEP
check(abs(you["x"] - expect) < 1.5 and abs(you["y"]) < 1.5 and you["quality"] == "dr",
      f"after {T['steps']} steps east you're at x={you['x']} m (steps x {STEP} = {expect:.1f}), y={you['y']}")
check(len(wj["nodes"]) >= 3 and len(wj["stamps"]) == len(wj["nodes"]), f"{len(wj['nodes'])} places down the hallway, {len(wj['stamps'])} snapshots")
check(all(abs(s["heading"] - 90) < 1 for s in wj["stamps"]), "snapshots remember which way you faced")
xs = sorted(n["x"] for n in wj["nodes"])
gaps = [b - a for a, b in zip(xs, xs[1:])]
check(gaps and min(gaps) >= 6 and max(gaps) <= 14, f"places spaced along the walk: gaps {[round(g, 1) for g in gaps]} m")
check(len(wj["edges"]) == len(wj["nodes"]) - 1, f"{len(wj['edges'])} hallway segments connect them")

# 4. the phone's rough fix (30 m off) must not yank you around while dead reckoning works
before = c.get("/world.json").json()["you"]
r = frame(hall, fix=(30, 30), acc=40, heading=90, steps=T["steps"])
after = r["place"]
check(abs(after["x"] - before["x"]) < 0.5 and abs(after["y"] - before["y"]) < 0.5 and after["quality"] == "dr",
      f"a 30 m-off Wi-Fi fix is ignored indoors (you stayed at {after['x']}, {after['y']})")

# 5. turn left (north) and walk 10 m: y grows, x doesn't
x0 = after["x"]
r, evs = walk(hall, 0, 10)
p = r["place"]
check(abs(p["x"] - x0) < 1.0 and 8 <= p["y"] <= 12, f"turned north: x stays {p['x']}, y={p['y']} m")
check(any(e["type"] == "snapshot" for e in evs), "the corner got its own snapshot")

# 6. the binary world: one surface grid per snapshot, every point in front of where it was taken
buf = c.get("/world.bin").content
hdr = np.frombuffer(buf[:16], np.uint32)
wj = c.get("/world.json").json()
ns, n_total = int(hdr[1]), int(hdr[2])
off, grids = 16, []
for _ in range(ns):
    node, gw, gh = (int(v) for v in np.frombuffer(buf[off:off + 12], np.uint32))
    off += 16
    nv = gw * gh
    xyz = np.frombuffer(buf[off:off + nv * 12], np.float32).reshape(gh, gw, 3)
    off += nv * 12
    dep = np.frombuffer(buf[off:off + nv * 4], np.float32).reshape(gh, gw)
    off += nv * 4
    rgb = np.frombuffer(buf[off:off + nv * 3], np.uint8).reshape(gh, gw, 3)
    off += nv * 3 + (-(nv * 3) % 4)
    grids.append({"node": node, "xyz": xyz, "dep": dep, "rgb": rgb})
n_valid = sum(int(np.isfinite(g["xyz"][..., 0]).sum()) for g in grids)
check(hdr[0] == 0xBEAC0003 and ns == len(wj["stamps"]) and off == len(buf) and n_total == n_valid == wj["n_points"],
      f"world.bin: {ns} snapshot grids of {grids[0]['xyz'].shape[1]}x{grids[0]['xyz'].shape[0]}, {n_total} points")
allz = np.concatenate([g["xyz"][..., 2][np.isfinite(g["xyz"][..., 2])] for g in grids])
check(0.5 < float(np.median(allz)) < 3.0 and max(g["rgb"].max() for g in grids) > 0, f"points at eye height-ish (median z {np.median(allz):.1f} m) with colors")
check(all(np.isfinite(g["dep"]).all() and (g["dep"] > 0).all() for g in grids), "every cell keeps its camera depth (for drawing surfaces, not dots)")
s0 = sorted(wj["stamps"], key=lambda s: s["ts"])[0]
g0 = [g for g in grids if g["node"] == s0["node"]][0]
p0 = g0["xyz"][np.isfinite(g0["xyz"][..., 0])]
ahead = float((p0[:, 0] > s0["x"] + 0.1).mean())
check(ahead > 0.95, f"first snapshot (facing east at x={s0['x']}): {ahead * 100:.0f}% of its points lie east of it")
spread = float(np.ptp(p0[:, 1]))
check(2 < spread < 60, f"…and spread sideways across the hallway ({spread:.1f} m wide)")
rev0 = wj["rev"]

# 7. snapshot on demand (button / voice) replaces the one at this place; clear empties the world
nid = p["node_id"]
old_ts = [s for s in wj["stamps"] if s["node"] == nid][0]["ts"]
r = c.post("/world/snapshot").json()
wj2 = c.get("/world.json").json()
new_ts = [s for s in wj2["stamps"] if s["node"] == nid][0]["ts"]
check(r["ok"] and r["node_id"] == nid and new_ts > old_ts and len(wj2["stamps"]) == len(wj["stamps"]), "snapshot button re-takes the snapshot here")
v = c.post("/command", json={"text": "snapshot"}).json()
check(v["ok"] and "Snapshot saved" in v["say"], f"voice: {v['say']}")
# a fresh server would reload the same world from disk
w2 = world_mod.World(server.world.path)
check(len(w2.stamps) == len(wj2["stamps"]) and w2.blob() == c.get("/world.bin").content, "snapshots persist on disk")
c.post("/world/clear")
check(c.get("/state").json()["world_stamps"] == 0 and c.get("/world.json").json()["n_points"] == 0, "clear: world empty")
check(np.frombuffer(c.get("/world.bin").content[:16], np.uint32)[1] == 0 and c.get("/world.json").json()["rev"] > rev0, "…and world.bin is empty too (revision moved on, so the dashboard refetches)")
check(c.get("/world.json").json()["nodes"], "the map (places + paths) is kept, only the 3D is gone")

# 8. GPS comes back good outdoors: the fix wins again
r = frame(hall, fix=(p["x"], p["y"] + 2), acc=5, heading=0, steps=T["steps"])
check(r["place"]["quality"] == "good", "a good GPS fix takes over from dead reckoning")

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
