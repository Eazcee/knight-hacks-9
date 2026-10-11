"""Tests for photo storage, training sets, merging and the no-spam map.

  python test_training.py
"""
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("BEACON_DATA", "data_test_training")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

c = TestClient(server.app)
c.post("/reset")
c.post("/map/clear")

from skimage import data  # noqa: E402
import insightface.data as idata  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


def jpg(img):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return b.tobytes()


def frame(img, **params):
    r = c.post("/frame", content=jpg(img), headers={"Content-Type": "image/jpeg"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def shifted(img, dx, dy, scale=1.0):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[scale, 0, dx], [0, scale, dy]]), (w, h))


group = idata.get_image("t1")
scene = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
t0 = time.time()

# 1. no GPS -> no places, even while "looking around"
for i in range(8):
    frame(shifted(scene, 12 * i, -6 * i, 1 - 0.02 * i))
m = c.get("/map").json()
check(len(m["nodes"]) == 0 and m["position"]["lat"] is None, "8 head-turn frames without GPS: no places, no position")

# 2. people get photo URLs and training photos with quality scores
r = frame(group)
check(len(r["people"]) >= 4, f"group photo: {len(r['people'])} people enrolled")
check(all("quality" in p for p in r["people"]), "every face has a quality score")
people = c.get("/people").json()
p0 = people[0]
check(p0["photo"] and c.get(p0["photo"]).status_code == 200, f"person photo served at {p0['photo']}")
check(all(p["n_train"] >= 1 for p in people), "everyone has at least one training photo")

pid = r["people"][0]["person_id"]
ph = c.get(f"/people/{pid}/photos").json()
check(len(ph["training"]) >= 1 and ph["training"][0]["photo"], "training photos listed with images")
check(len(ph["sightings"]) >= 1 and ph["sightings"][0]["frame"] and ph["sightings"][0]["box"],
      "sightings carry the full camera frame + face box")
check(c.get(ph["sightings"][0]["frame"]).status_code == 200, "full frame image served")
check(c.get("/live.jpg").status_code == 200, "live preview served")

# 3. train mode: show one face from different angles, it collects varied photos
f = [f for f in server.faces.detect(group, min_px=20) if server.faces.match(f["emb"])[0] == pid][0]
x1, y1, x2, y2 = f["box"]
w, h = x2 - x1, y2 - y1
solo = group[max(0, y1 - h):y2 + h, max(0, x1 - w):x2 + w]
solo = cv2.resize(solo, (solo.shape[1] * 2, solo.shape[0] * 2))
before = len(ph["training"])
c.post("/train/start", json={"person_id": pid, "seconds": 30, "goal": 6})


def aug(img, k):
    hh, ww = img.shape[:2]
    if k % 4 == 0:
        return cv2.flip(img, 1)
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), [-14, 10, 18, -8, 6, -18, 14][k % 7], 1.0 - 0.03 * (k % 3))
    out = cv2.warpAffine(img, M, (ww, hh))
    return np.clip(out.astype(np.int16) + (k * 9 - 30), 0, 255).astype(np.uint8)


# augmented copies of ONE photo are near-duplicates, so by default they're rejected
# (real head turns differ more). First prove the duplicate filter, then relax it.
r = frame(aug(solo, 1))
time.sleep(0.4)
check(c.get("/train").json()["added"] == 0, "near-duplicate photos are not added to training")
import faces as faces_mod  # noqa: E402
faces_mod.TRAIN_MODE_MAX_SIM = 0.995
for k in range(14):
    r = frame(aug(solo, k))
    time.sleep(0.4)
faces_mod.TRAIN_MODE_MAX_SIM = 0.88
st = c.get("/train").json()
after = c.get(f"/people/{pid}/photos").json()["training"]
trained = [t for t in after if t["source"] == "train"]
check(st["added"] >= 3 and len(trained) == st["added"], f"train mode collected {st['added']} new angles")
check(r["display"]["title"].startswith("Training") or st["done"], f"glasses card during training: {r['display']['title']}")
c.post("/train/stop")

# 4. remove a bad photo
n_before = len(after)
c.delete(f"/train_photos/{after[-1]['id']}")
n_after = len(c.get(f"/people/{pid}/photos").json()["training"])
check(n_after == n_before - 1, f"removed a training photo ({n_before} -> {n_after})")
check(server.faces.match(f["emb"])[0] == pid, "still recognized after removing a photo")

# 5. split + suggestion + merge
tp = c.get(f"/people/{pid}/photos").json()["training"]
moved = c.post(f"/train_photos/{tp[0]['id']}/move", json={"to": "new"}).json()
new_pid = moved["person_id"]
sugg = c.get("/suggestions").json()
pair = [s for s in sugg if {s["a"]["id"], s["b"]["id"]} == {pid, new_pid}]
check(bool(pair), f"split person shows up as a merge suggestion ({pair[0]['sim'] if pair else '-'})")
c.post(f"/people/{new_pid}/merge", json={"into": pid})
check(c.get(f"/people/{new_pid}/photos").status_code == 404, "merged person is gone")
check(len(c.get(f"/people/{pid}/photos").json()["training"]) == len(tp), "all photos back on one person")

# 6. promote a sighting photo, and move a wrong sighting to a new person
s = c.get(f"/people/{pid}/photos").json()["sightings"][0]
r1 = c.post(f"/sightings/{s['id']}/train")
check(r1.status_code == 200, "added a sighting photo to training")
r2 = c.post(f"/sightings/{s['id']}/move", json={"to": "new"}).json()
check(r2["person_id"] != pid, "moved a wrong sighting to a new person")

# 7. alert flag, voice train, mark place, clear map, dashboard
c.post(f"/people/{pid}/flag", params={"label": "AMBER Alert · Test"})
check(c.get(f"/people/{pid}").json()["is_watch"] == 1, "flagged as alert without a photo")
c.post(f"/people/{pid}", json={"name": "Maya"})
r = c.post("/command", json={"text": "train maya"}).json()
check(r["ok"] and r["train"]["person_id"] == pid, f"voice: {r['say']}")
c.post("/train/stop")
check(c.post("/places/mark", json={"name": "x"}).status_code == 409, "can't mark a place without GPS")
frame(scene, lat=28.6024, lng=-81.2001, acc=5)
r = c.post("/places/mark", json={"name": "food court"}).json()
check(r["name"] == "Food Court" and len(c.get("/map").json()["nodes"]) == 1, "marked a place by hand at the GPS position")
c.post("/map/clear")
check(len(c.get("/map").json()["nodes"]) == 0 and len(c.get("/people").json()) >= 4, "cleared map, kept people")
html = c.get("/")
check(html.status_code == 200 and "BEACON" in html.text, "dashboard served")
st = c.get("/state").json()
check("train" in st and "n_suggestions" in st and st["people"][0]["photo"], "state has photos, training, suggestions")

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
sys.exit(1 if fails else 0)
