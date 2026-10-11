"""Range tests: far faces, "possible" alerts, and Scan on a 12 MP photo.

  python test_range.py

Distance is simulated by shrinking a group photo inside a 720x1280 frame (what the
glasses send on HIGH). A face half as far away is twice as many pixels.
"""
import os
import sys
import time

import cv2
import numpy as np

test_database_url = os.environ.get("BEACON_TEST_DATABASE_URL")
if not test_database_url:
    raise SystemExit("Set BEACON_TEST_DATABASE_URL to a disposable Neon branch; never use production")
os.environ["DATABASE_URL"] = test_database_url
os.environ.setdefault("BEACON_DATA", "data_test_range")
from fastapi.testclient import TestClient  # noqa: E402

import faces as faces_mod  # noqa: E402
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


group = idata.get_image("t1")
GH, GW = group.shape[:2]
bg = cv2.cvtColor(skd.coffee(), cv2.COLOR_RGB2BGR)
ref = server.faces.detect(group, min_px=20)
med_face = float(np.median([f["px"] for f in ref]))


def frame_with_faces(face_px, FW=720, FH=1280, at=(0.5, 0.5)):
    """The group photo shrunk so faces are ~face_px, pasted into a frame of FWxFH."""
    s = face_px / med_face
    g = cv2.resize(group, (int(GW * s), int(GH * s)), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(bg, (FW, FH))
    x0 = int(FW * at[0] - g.shape[1] / 2)
    y0 = int(FH * at[1] - g.shape[0] / 2)
    canvas[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
    ok, b = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return b.tobytes()


def post(path, body, **params):
    r = c.post(path, content=body, headers={"Content-Type": "image/jpeg"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


t0 = time.time()

# 1. meet everyone up close, give them names
r = post("/frame", frame_with_faces(med_face, FW=1400, FH=1400))
check(len(r["people"]) == 6 and all(p["new"] for p in r["people"]), f"up close ({med_face:.0f} px faces): {len(r['people'])} people enrolled")
names = ["Ana", "Ben", "Cy", "Dee", "Eli", "Fay"]
pid_by_x = sorted(r["people"], key=lambda p: p["box"][0])
for p, n in zip(pid_by_x, names):
    c.post(f"/people/{p['person_id']}", json={"name": n})
n_people = len(c.get("/people").json())

# 2. step back: faces ~30 px = "far" tier. Known people only, need two sightings to confirm
r1 = post("/frame", frame_with_faces(30))
far = [p for p in r1["people"] if p["tier"] == "far"]
check(len(far) >= 5, f"30 px faces: {len(far)}/6 matched to known people (far tier)")
check(all(p["name"] in names for p in far), "far matches carry the right names")
right = sum(1 for p, q in zip(sorted(far, key=lambda p: p["box"][0]), [x for x in pid_by_x if x["person_id"] in {f['person_id'] for f in far}]) if p["person_id"] == q["person_id"])
check(right == len(far), f"every far match is the same person as up close ({right}/{len(far)})")
check(len(c.get("/people").json()) == n_people, "far faces never create new people")
r2 = post("/frame", frame_with_faces(30))
check(all(p["confirmed"] for p in r2["people"] if p["tier"] == "far"), "second far sighting confirms them")
check(r2["display"]["title"].endswith("people") and any(n in " ".join(r2["display"]["lines"]) for n in names), f"glasses card names far people: {r2['display']['title']} / {r2['display']['lines']}")

# 3. even further: faces ~22 px still matched; ~12 px ignored (too small)
r = post("/frame", frame_with_faces(22))
check(len([p for p in r["people"] if p["name"]]) >= 3, f"22 px faces: {len([p for p in r['people'] if p['name']])}/6 still named")
r = post("/frame", frame_with_faces(12))
check(len(r["people"]) == 0, "12 px faces: nobody named (below the floor), nobody created")
check(len(c.get("/people").json()) == n_people, f"still {n_people} people after all the far frames")

# 4. "possible" alert: a far face that's only vaguely like an alert target
target = pid_by_x[2]["person_id"]
c.post(f"/people/{target}/flag", params={"label": "AMBER Alert · Cy"})
n_alerts = len(c.get("/alerts").json())
r = post("/frame", frame_with_faces(30))
r = post("/frame", frame_with_faces(30))
tp = [p for p in r["people"] if p["person_id"] == target]
check(bool(tp) and tp[0]["possible"] and not r["alerts"], "alert target far away (clear match): still only 'possible', no real alert")
faces_mod.FAR_MATCH = 0.99          # pretend the far face is too blurry for a firm match
r = post("/frame", frame_with_faces(26))
faces_mod.FAR_MATCH = 0.40
poss = [p for p in r["people"] if p["possible"]]
check(bool(poss) and poss[0]["person_id"] == target, "blurry far face -> 'possible' match to the alert target")
check(r["display"]["title"].startswith("Possible") and "closer" in " ".join(r["display"]["lines"]), f"glasses: {r['display']['title']} / {r['display']['lines']}")
check(len(c.get("/alerts").json()) == n_alerts and not r["alerts"], "a 'possible' never fires a real alert")
r = post("/frame", frame_with_faces(med_face, FW=1400, FH=1400))
check(any(a["person_id"] == target for a in r["alerts"]), f"walk up -> confirmed: {r['display']['title']} ({r['display']['lines'][0]})")

# 5. Scan: a 12 MP photo where the group is far away (faces ~40 px of 3024x4032)
t = time.time()
big = frame_with_faces(40, FW=3024, FH=4032, at=(0.62, 0.45))
r = post("/scan", big)
named = [p for p in r["people"] if p["name"]]
check(len(named) >= 5, f"scan 3024x4032: {len(named)}/6 named in {r['ms']} ms")
check(any(a["person_id"] == target for a in r["alerts"]), "scan finds the alert target")
check(r["display"]["title"].startswith("ALERT") and len(r["display"]["lines"]) >= 2, f"scan card: {r['display']['title']} / {r['display']['lines']}")
check(all(p["dir"] in ("left", "ahead", "right") for p in r["people"]), "every face has a direction")
same_frame_only = post("/frame", frame_with_faces(40 * 720 / 3024, at=(0.62, 0.45)))
check(len([p for p in same_frame_only["people"] if p["name"]]) < len(named),
      f"same scene as a video frame: only {len([p for p in same_frame_only['people'] if p['name']])} named (scan sees further)")
sg = c.get(f"/people/{target}/photos").json()["sightings"][0]
check(sg["box"][4] <= 2048 and c.get(sg["frame"]).status_code == 200, "scan photo stored (downsized) with the face box")
check(len(c.get("/people").json()) == n_people, "scan didn't duplicate anyone")

# 6. voice + state
r = c.post("/command", json={"text": "scan"}).json()
check(r.get("action") == "scan", "voice 'scan' tells the app to take a photo")
st = c.get("/state").json()
check(st["frame_size"] and st["range"]["min_face_px"] == faces_mod.MIN_FACE_PX and st["last_scan"], f"state: frame {st['frame_size']}, range {st['range']}")
check(c.get("/live.jpg").status_code == 200, "live view after scan")

# 7. walk-by: a blurry stranger passing in 2 frames still gets enrolled (but not from one frame)
c.post("/reset")
def blurred(px, k):
    img = cv2.imdecode(np.frombuffer(frame_with_faces(px, FW=1280, FH=1280), np.uint8), 1)
    kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0 / k   # horizontal motion blur
    return cv2.imencode(".jpg", cv2.filter2D(img, -1, kern), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
r1 = post("/frame", blurred(64, 9))
q = sorted(f["quality"] for f in server.faces.detect(cv2.imdecode(np.frombuffer(blurred(64, 9), np.uint8), 1)))
n1 = len(c.get("/people").json())
r2 = post("/frame", blurred(66, 9))
n2 = len(c.get("/people").json())
check(n1 <= 2 and n2 >= 4, f"walk-by (motion blur, quality {q[0]:.2f}-{q[-1]:.2f}): {n1} enrolled after 1 frame, {n2} after 2")
r3 = post("/frame", blurred(64, 9))
check(len(c.get("/people").json()) == n2, "third blurry frame: recognized, no duplicates")
st = c.get("/state").json()
check(st["timing"]["device"] and st["timing"]["frame_ms"] > 0, f"timing: {st['timing']}")

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
