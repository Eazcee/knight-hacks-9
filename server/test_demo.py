"""End-to-end test of the Beacon server with no hardware.

Simulates a walk through three places, meets a group of people at the second
place, walks on, comes back to the start (loop closure), then loads an "alert"
photo of someone met earlier: the server should say "seen before", where, and
route back there. Then blocks an edge and reroutes.

  python test_demo.py
"""
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

from skimage import data  # noqa: E402
import insightface.data as idata  # noqa: E402


def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def jpg(img):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return b.tobytes()


def frame(img, heading=None, steps=None):
    params = {}
    if heading is not None:
        params["heading"] = heading
    if steps is not None:
        params["steps"] = steps
    r = c.post("/frame", content=jpg(img), headers={"Content-Type": "image/jpeg"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def shifted(img, scale=0.92, dx=15, dy=10):
    h, w = img.shape[:2]
    M = np.float32([[scale, 0, dx], [0, scale, dy]])
    return cv2.warpAffine(img, M, (w, h))


A = bgr(data.astronaut())     # place A keyframe
B = bgr(data.coffee())        # place B
C = bgr(data.chelsea())       # place C
group = idata.get_image("t1")  # 6 faces, BGR

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


t0 = time.time()
# --- walk: A -> north 10 steps -> B -> east 10 steps -> C
r = frame(A, heading=0, steps=0)
check(any(e["type"] == "new_place" for e in r["place"]["events"]), "first frame creates place 1 (A)")
nA = r["place"]["node_id"]
r = frame(B, heading=0, steps=10)
check(any(e["type"] == "new_place" for e in r["place"]["events"]), "after 10 steps a new place is created (B)")
nB = r["place"]["node_id"]
c.post(f"/nodes/{nB}", json={"name": "Coffee Station"})

# meet people at B
r = frame(group, heading=0, steps=10)
check(len(r["people"]) >= 4, f"group photo: {len(r['people'])} strangers auto-enrolled at B")
pid0 = r["people"][0]["person_id"]
c.post(f"/people/{pid0}", json={"name": "Maya", "append_note": "judge, AI track, likes Rust"})
r = frame(group, heading=0, steps=10)
check(any(p["person_id"] == pid0 and p["name"] == "Maya" for p in r["people"]), "second look: Maya recognized with her note")
check(any("Maya" in l for l in r["display"]["lines"]), "glasses display shows Maya's card")

r = frame(C, heading=90, steps=20)
check(any(e["type"] == "new_place" for e in r["place"]["events"]), "walk east: place C created")
nC = r["place"]["node_id"]

# --- come back to A: dead reckoning says we're somewhere southwest, the camera says we're at A
r = frame(shifted(A), heading=225, steps=34)
ev = [e for e in r["place"]["events"] if e["type"] == "arrived"]
check(bool(ev) and ev[0]["node_id"] == nA, f"back at A: visual loop closure ({ev[0]['score'] if ev else 0} inliers)")
pos = r["place"]
check(abs(pos["x"]) < 0.1 and abs(pos["y"]) < 0.1, "position snapped to A on loop closure")

m = c.get("/map").json()
check(len(m["nodes"]) == 3, f"map has 3 places ({len(m['nodes'])})")
edges = {(e["a"], e["b"]) for e in m["edges"]}
check((nA, nB) in edges and (nB, nC) in edges and (nC, nA) in edges, "edges A-B, B-C, C-A exist")

# --- alert: a photo of Maya arrives (AMBER-style). Have we seen her? Where? Route there.
x1, y1, x2, y2 = [int(v) for v in r["people"][0]["box"]] if r["people"] else (0, 0, 0, 0)
faces = server.faces.detect(group, min_px=20)
f = [f for f in faces if server.faces.match(f["emb"])[0] == pid0][0]
x1, y1, x2, y2 = f["box"]
w, h = x2 - x1, y2 - y1
alert_photo = group[max(0, y1 - h):y2 + h, max(0, x1 - w):x2 + w]
alert_photo = cv2.flip(alert_photo, 1)  # a different photo of her
r = c.post("/watch", files={"image": ("alert.jpg", jpg(alert_photo), "image/jpeg")},
           data={"label": "AMBER Alert · Maya · 7"}).json()
check(r["seen_before"] and r["person_id"] == pid0, "alert photo matched someone we already saw")
check(r["last"] and r["last"]["place"] == "Coffee Station", f"last seen at {r['last']['place'] if r['last'] else None}")
check(r["route"]["ok"] and r["route"]["path"] == [nA, nB], f"route to last sighting: {r['route'].get('path')} ({r['route'].get('cost_m')} m)")

# --- voice commands (we're at A, she was last seen at B)
r = c.post("/command", json={"text": "where did I see Maya"}).json()
check(r["ok"] and "Coffee Station" in r["say"], f"voice: {r['say']}")
r = c.post("/command", json={"text": "find maya"}).json()
check(r["ok"] and "Routing" in r["say"], f"voice find: {r['say']}")

# --- block A-B and reroute
r = c.post(f"/edges/{nA}/{nB}", json={"blocked": True}).json()
route = c.get("/route", params={"to": nB}).json()
check(route["ok"] and route["path"] == [nA, nC, nB], f"blocked A-B, rerouted via C: {route.get('path')} ({route.get('cost_m')} m)")

# --- live alert: she appears in front of the glasses now
r = frame(group, heading=0, steps=34)
check(len(r["alerts"]) >= 1 and r["display"]["level"] == "alert", f"live match raises alert: {r['display']['title']}")
r = c.post("/command", json={"text": "report"}).json()
check(r["ok"] and r["report"]["place"], f"voice report: {r['say']}")

st = c.get("/state").json()
check(st["frames"] >= 7 and len(st["sightings"]) >= 4, f"state: {st['frames']} frames, {len(st['sightings'])} sightings logged")
print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
