"""Face engine: detect faces, score their quality, turn them into 512-d ArcFace
embeddings ("face fingerprints"), match them against everyone seen so far, and
build each person's training set.

How the "categories" work:
  * Every person is a category. Their training set is up to MAX_EMB_PER_PERSON
    fingerprints, each with the photo it came from.
  * A new face is compared with every fingerprint of every person; the best
    person wins if the similarity is above MATCH_THRESHOLD.
  * Strangers become a new anonymous person automatically (if the photo is good).
  * Good new angles of a known person are added to their set automatically.
  * Train mode: point the glasses at one person for ~20 s and their set fills
    with their best, most varied angles.
  * Mistakes are fixed by people, not code: merge two people, remove a bad photo.

Uses InsightFace (buffalo_l). On first run it downloads ~280 MB of models to
~/.insightface/models.
"""
import os
import threading
import time
import uuid

import cv2
import numpy as np

MATCH_THRESHOLD = float(os.environ.get("BEACON_MATCH", 0.45))      # cosine sim to call it a match
NEW_PERSON_THRESHOLD = float(os.environ.get("BEACON_NEW", 0.32))   # below this = definitely a stranger
# Range. Face size in pixels (shorter side of the box) is what limits range, not the
# camera: a face half as far away is twice as many pixels. Three tiers:
#   >= ENROLL_PX : can become a new person
#   >= NEAR_PX   : full recognition, learns new angles
#   >= MIN_FACE_PX: "far" -- only matched to people we already know, with stricter rules
MIN_FACE_PX = int(os.environ.get("BEACON_MIN_FACE", 18))           # smallest face we look at
NEAR_PX = int(os.environ.get("BEACON_NEAR_PX", 40))
ENROLL_PX = int(os.environ.get("BEACON_ENROLL_PX", 56))
FAR_MATCH = float(os.environ.get("BEACON_FAR_MATCH", 0.40))        # far faces: similarity needed ...
FAR_MARGIN = 0.10             # ... the best person must beat the runner-up by this much ...
FAR_CONFIRM_S = 6.0           # ... and appear twice within this many seconds (or score >= FAR_SURE)
FAR_SURE = 0.60
POSSIBLE_ALERT = float(os.environ.get("BEACON_POSSIBLE", 0.32))    # far face vaguely like an alert target: "get closer"
ALERT_NEEDS_NEAR = os.environ.get("BEACON_FAR_ALERTS", "0") != "1"  # real alerts only from close faces (or Scan)
MIN_DET_SCORE = 0.60
MAX_EMB_PER_PERSON = 16       # training photos kept per person
NEW_MIN_QUALITY = 0.45        # a stranger needs a decent photo before we create a person
TRAIN_MIN_QUALITY = 0.45      # automatic additions to a training set
TRAIN_MODE_MIN_QUALITY = 0.30 # train mode is supervised, so we accept a bit more
SUGGEST_MERGE = 0.45          # cross-similarity above this = "might be the same person"
AUTO_MAX_SIM = 0.80           # auto-learning skips photos this similar to one we already have
TRAIN_MODE_MAX_SIM = 0.88     # train mode is a bit more generous (slow head turns change little per frame)
# Walk-by enrollment: a passer-by is often blurry in every frame. If the same unknown face
# shows up in 2+ frames within a few seconds, enroll from the best of them with a lower bar.
WALKBY_MIN_QUALITY = float(os.environ.get("BEACON_WALKBY_Q", 0.32))
WALKBY_MIN_PX = int(os.environ.get("BEACON_WALKBY_PX", 46))
WALKBY_WINDOW_S = 4.0
WALKBY_SAME = 0.45            # two blurry looks at the same stranger usually score above this


def _providers():
    """GPU if the laptop has one (CUDA for NVIDIA, DirectML for any Windows GPU), else CPU.
    BEACON_PROVIDERS=cpu forces CPU."""
    if os.environ.get("BEACON_PROVIDERS", "").lower() == "cpu":
        return ["CPUExecutionProvider"], -1
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"], -1
    for gpu in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if gpu in avail:
            return [gpu, "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1


MODEL = os.environ.get("BEACON_MODEL", "buffalo_l")   # buffalo_s = ~4x faster on CPU, a bit less accurate
DET_SIZE = int(os.environ.get("BEACON_DET", 640))                  # detector input, long side. smaller = faster, shorter range
SCAN_TILE = int(os.environ.get("BEACON_SCAN_TILE", 1600))          # /scan: big photos are searched in tiles this size


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def face_quality(img, box, kps, det):
    """0..1 score: big, sharp, front-facing, confidently detected faces score high."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    s_size = float(np.clip((min(w, h) - 40) / 80.0, 0, 1))
    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0, {}
    g = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    s_sharp = float(np.clip((sharp - 30) / 220.0, 0, 1))
    s_front = 0.5
    if kps is not None and len(kps) >= 3:
        le, re, nose = kps[0], kps[1], kps[2]
        eye_d = max(1.0, abs(re[0] - le[0]))
        yaw = abs(nose[0] - (le[0] + re[0]) / 2) / eye_d
        s_front = float(np.clip(1 - yaw / 0.45, 0, 1))
    s_det = float(np.clip((det - 0.6) / 0.3, 0, 1))
    q = 0.35 * s_size + 0.30 * s_sharp + 0.25 * s_front + 0.10 * s_det
    return round(q, 3), {"size": round(s_size, 2), "sharp": round(s_sharp, 2),
                         "front": round(s_front, 2), "det": round(s_det, 2)}


class FaceEngine:
    def __init__(self, db, crops_dir="data/crops", auto_enroll=True):
        from insightface.app import FaceAnalysis  # imported here so the server starts fast

        self.db = db
        self.crops_dir = crops_dir
        os.makedirs(crops_dir, exist_ok=True)
        self.auto_enroll = auto_enroll
        self.lock = threading.Lock()
        # embeddings from different models can't be compared: don't mix them in one database
        mpath = os.path.join(os.path.dirname(crops_dir) or ".", "model.txt")
        saved = open(mpath).read().strip() if os.path.exists(mpath) else None
        if saved and saved != MODEL and len(db.embeddings()[0]):
            raise SystemExit(f"This database was built with {saved}, but BEACON_MODEL={MODEL}. "
                             f"Use {saved}, or delete the data folder to start fresh with {MODEL}.")
        with open(mpath, "w") as fh:
            fh.write(MODEL)
        providers, ctx = _providers()
        self.device = "GPU (" + providers[0].replace("ExecutionProvider", "") + ")" if ctx >= 0 else "CPU"
        try:
            self.app = FaceAnalysis(name=MODEL, providers=providers, allowed_modules=["detection", "recognition"])
            self.app.prepare(ctx_id=ctx, det_size=(DET_SIZE, DET_SIZE))
        except Exception as e:   # GPU provider present but broken: fall back to CPU
            print("GPU init failed, using CPU:", e)
            self.device = "CPU"
            self.app = FaceAnalysis(name=MODEL, providers=["CPUExecutionProvider"],
                                    allowed_modules=["detection", "recognition"])
            self.app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))
        self.device += f" · {MODEL}"
        print("Beacon face engine on", self.device)
        self.pending = []       # walk-by candidates: unknown faces seen recently
        self._det_cur = (DET_SIZE, DET_SIZE)
        self._reload_gallery()
        self.recent = {}        # person_id -> last match time (for smoothing)
        self.last_auto_add = {} # person_id -> last time we auto-added a training photo
        self.train = None       # active train-mode state
        self.dismissed = set()  # merge suggestions the user said "not the same"
        self.last_far = []      # faces in the last frame that were too small/unclear to name
        self.last_size = None   # (w, h) of the last frame

    # ---- gallery -------------------------------------------------------
    def _reload_gallery(self):
        self.gal_ids, self.gal_mat, self.gal_eids = self.db.embeddings()
        self.gal_ids = list(self.gal_ids)
        self.gal_eids = list(self.gal_eids)

    def _add_to_gallery(self, pid, emb, img=None, box=None, quality=0.0, source="auto"):
        crop = None
        if img is not None and box is not None:
            crop = self._save_crop(img, box, f"t_{pid}_{uuid.uuid4().hex[:8]}")
            p = self.db.person(pid)
            if p is not None and not p.get("crop"):
                self.db.update_person(pid, crop=crop)
        eid = self.db.add_embedding(pid, emb, crop=crop, quality=quality, source=source)
        self.gal_ids.append(pid)
        self.gal_eids.append(eid)
        self.gal_mat = np.vstack([self.gal_mat, emb[None, :]]) if len(self.gal_mat) else emb[None, :]
        return eid

    def _remove_from_gallery(self, eid):
        self.db.delete_embedding(eid)
        self._reload_gallery()

    def match(self, emb):
        """Returns (person_id, similarity) of the best match, or (None, best_sim)."""
        if len(self.gal_mat) == 0:
            return None, 0.0
        sims = self.gal_mat @ emb
        best = {}
        for pid, s in zip(self.gal_ids, sims):
            if s > best.get(pid, -1):
                best[pid] = float(s)
        pid, s = max(best.items(), key=lambda kv: kv[1])
        return (pid if s >= MATCH_THRESHOLD else None), s

    def match_detail(self, emb):
        """(best person, their similarity, runner-up similarity) without any threshold."""
        if len(self.gal_mat) == 0:
            return None, 0.0, 0.0
        sims = self.gal_mat @ emb
        best = {}
        for pid, s in zip(self.gal_ids, sims):
            if s > best.get(pid, -1):
                best[pid] = float(s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return ranked[0][0], ranked[0][1], second

    def _watch_ids(self):
        return [r["id"] for r in self.db._run("SELECT id FROM people WHERE is_watch=1", fetch="all")]

    def best_sim_to(self, emb, pid):
        idx = [i for i, p in enumerate(self.gal_ids) if p == pid]
        if not idx:
            return 0.0
        return float(np.max(self.gal_mat[idx] @ emb))

    def _maybe_learn(self, pid, f, img, ts, source="auto", min_q=TRAIN_MIN_QUALITY, max_sim=None):
        """Add this face to pid's training set if it's good and adds a new angle.
        When the set is full, replace the weakest automatic photo if this one is better."""
        if f["quality"] < min_q:
            return False
        own = self.best_sim_to(f["emb"], pid)
        if own >= (AUTO_MAX_SIM if max_sim is None else max_sim):              # near-duplicate of a photo we already have
            return False
        if self.db.count_embeddings(pid) >= MAX_EMB_PER_PERSON:
            worst = self.db.worst_embedding(pid)
            if not worst or f["quality"] <= (worst["quality"] or 0) + 0.10:
                return False
            if worst["source"] != "auto" and source == "auto":
                return False
            self._remove_from_gallery(worst["id"])
        self._add_to_gallery(pid, f["emb"], img, f["box"], f["quality"], source)
        return True

    # ---- enrollment ----------------------------------------------------
    def enroll_image(self, img_bgr, name=None, notes="", is_watch=0, watch_label=""):
        """Enroll the largest face in an image. If that face already belongs to a known
        person, we attach the name/notes/watch flag to THAT person instead of making a
        duplicate, and return matched_existing=True. That's the "have I seen them?" moment."""
        f = self.largest_face(img_bgr)
        if not f:
            return None
        pid, sim = self.match(f["emb"])
        matched_existing = pid is not None
        if not matched_existing:
            pid = "p_" + uuid.uuid4().hex[:6]
            self.db.add_person(pid, name=name, notes=notes, is_watch=is_watch, watch_label=watch_label)
        else:
            fields = {}
            if name:
                fields["name"] = name
            if notes:
                fields["notes"] = notes
            if is_watch:
                fields["is_watch"] = 1
                fields["watch_label"] = watch_label
            self.db.update_person(pid, **fields)
        self._add_to_gallery(pid, f["emb"], img_bgr, f["box"], max(f["quality"], 0.5), "photo")
        return {"person_id": pid, "matched_existing": matched_existing, "sim": sim}

    def largest_face(self, img_bgr):
        faces = self.detect(img_bgr, min_px=20)
        return max(faces, key=lambda f: f["area"]) if faces else None

    # ---- train mode ----------------------------------------------------
    def start_training(self, pid, seconds=20, goal=10):
        self.train = {"person_id": pid, "until": time.time() + seconds, "goal": goal,
                      "added": 0, "frames": 0, "started": time.time(), "last_add": 0.0}
        return self.train_state()

    def stop_training(self):
        st = self.train_state()
        self.train = None
        return st

    def train_state(self):
        t = self.train
        if not t:
            return None
        if time.time() > t["until"] or t["added"] >= t["goal"]:
            t["done"] = True
        p = self.db.person(t["person_id"]) or {}
        return {"person_id": t["person_id"], "name": p.get("name"), "added": t["added"],
                "goal": t["goal"], "seconds_left": max(0, round(t["until"] - time.time())),
                "done": bool(t.get("done"))}

    # ---- per-frame pipeline -------------------------------------------
    @staticmethod
    def _det_size_for(h, w, long_side):
        """Detector input that keeps the frame's shape: a portrait 720x1280 frame gets a
        352x640 detector instead of being squeezed into a 640x640 square."""
        short = max(128, int(round(long_side * min(h, w) / max(h, w) / 32.0)) * 32)
        return (short, long_side) if h >= w else (long_side, short)

    def _run_app(self, img, long_side=DET_SIZE):
        size = self._det_size_for(img.shape[0], img.shape[1], long_side)
        with self.lock:
            if size != self._det_cur:
                d = self.app.det_model
                d.input_size = size
                if hasattr(d, "input_sizes"):      # insightface >= 2.x keeps a list of sizes
                    d.input_sizes = [size]
                self._det_cur = size
            raw = self.app.get(img)
        return [{"bbox": np.asarray(f.bbox, dtype=np.float32), "det": float(f.det_score),
                 "kps": None if getattr(f, "kps", None) is None else np.asarray(f.kps, dtype=np.float32),
                 "emb": f.normed_embedding.astype(np.float32)}
                for f in raw if getattr(f, "normed_embedding", None) is not None]

    def _detect_tiled(self, img):
        """Big photos (e.g. a 12 MP capture): one pass on the whole image for close faces,
        then overlapping tiles at full resolution so faces far away are still found."""
        H, W = img.shape[:2]
        found = [(f, float("inf")) for f in self._run_app(img)]
        T = SCAN_TILE
        if max(H, W) > DET_SIZE * 1.5:
            O = int(T * 0.2)

            def starts(dim):
                if dim <= T:
                    return [0], dim
                n = int(np.ceil((dim - O) / (T - O)))
                return [int(v) for v in np.linspace(0, dim - T, n)], T

            xs, tw = starts(W)
            ys, th = starts(H)
            if len(xs) * len(ys) > 1:
                for y0 in ys:
                    for x0 in xs:
                        tile = img[y0:y0 + th, x0:x0 + tw]
                        for f in self._run_app(tile):
                            x1, y1, x2, y2 = f["bbox"]
                            # faces cut by an inner tile edge are found whole by a neighbor tile
                            edge = min(x1 if x0 > 0 else 1e9, y1 if y0 > 0 else 1e9,
                                       (tw - x2) if x0 + tw < W else 1e9, (th - y2) if y0 + th < H else 1e9)
                            if edge < 3:
                                continue
                            f["bbox"] = f["bbox"] + np.float32([x0, y0, x0, y0])
                            if f["kps"] is not None:
                                f["kps"] = f["kps"] + np.float32([x0, y0])
                            found.append((f, edge))
        # the same face found twice (whole image + tile, or two overlapping tiles): keep one
        found.sort(key=lambda t: (-t[0]["det"]))
        keep = []
        for f, _ in found:
            if all(_iou(f["bbox"], k["bbox"]) < 0.3 for k in keep):
                keep.append(f)
        return keep

    def detect(self, img_bgr, min_px=MIN_FACE_PX, tiles=False):
        raw = self._detect_tiled(img_bgr) if tiles else self._run_app(img_bgr)
        out = []
        for f in raw:
            x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
            w, h = x2 - x1, y2 - y1
            if min(w, h) < min_px or f["det"] < MIN_DET_SCORE:
                continue
            box = [max(0, x1), max(0, y1), x2, y2]
            q, parts = face_quality(img_bgr, box, f["kps"], f["det"])
            out.append({
                "box": box,
                "px": int(min(w, h)),
                "area": w * h,
                "det": f["det"],
                "emb": f["emb"],
                "quality": q,
                "q_parts": parts,
            })
        return out

    def process_frame(self, img_bgr, ts=None, tiles=False):
        """Detect + match every face in a frame. Strangers with a good photo get
        auto-enrolled with an anonymous id so the next sighting recognizes them.
        Small (far) faces are only matched to people we already know.
        Returns a list of people in the frame; unnamed far faces go to self.last_far."""
        ts = ts or time.time()
        H, W = img_bgr.shape[:2]
        self.last_size = (W, H)
        faces = self.detect(img_bgr, tiles=tiles)
        results, far_unknown = [], []
        watch = None

        # train mode: the largest (close enough) face in view belongs to the person being trained
        train_face = None
        st = self.train_state()
        near_faces = [f for f in faces if f["px"] >= NEAR_PX]
        if st and not st["done"] and near_faces:
            train_face = max(near_faces, key=lambda f: f["area"])
            self.train["frames"] += 1

        for f in faces:
            new = learned = possible = False
            near = f["px"] >= NEAR_PX
            if near:
                pid, sim = self.match(f["emb"])
                if f is train_face:
                    target = self.train["person_id"]
                    strongly_other = pid is not None and pid != target and sim >= 0.55
                    if not strongly_other and ts - self.train["last_add"] >= 0.35:
                        if self._maybe_learn(target, f, img_bgr, ts, source="train",
                                             min_q=TRAIN_MODE_MIN_QUALITY, max_sim=TRAIN_MODE_MAX_SIM):
                            self.train["added"] += 1
                            self.train["last_add"] = ts
                            learned = True
                    if not strongly_other:
                        pid, sim = target, max(sim if pid == target else 0.0, self.best_sim_to(f["emb"], target))
                if pid is None:
                    best = None
                    if self.auto_enroll and sim < NEW_PERSON_THRESHOLD:
                        if f["px"] >= ENROLL_PX and f["quality"] >= NEW_MIN_QUALITY:
                            best = (f, img_bgr, f["box"])
                        else:
                            best = self._walkby(f, img_bgr, ts)
                    if best is not None:
                        bf, bimg, bbox = best
                        pid = "p_" + uuid.uuid4().hex[:6]
                        self.db.add_person(pid)
                        self._add_to_gallery(pid, bf["emb"], bimg, bbox, bf["quality"], "auto")
                        new = True
                        sim = 1.0   # this photo IS their reference now
                    else:
                        far_unknown.append({"box": f["box"], "px": f["px"]})
                        continue  # ambiguous or low quality: skip this face this frame
                elif not learned and f is not train_face:
                    # learn new angles of known people over time, at most one photo / 2 s
                    if ts - self.last_auto_add.get(pid, 0) >= 2.0 and sim >= MATCH_THRESHOLD:
                        if self._maybe_learn(pid, f, img_bgr, ts):
                            self.last_auto_add[pid] = ts
                            learned = True
                # smoothing: a weak match needs to repeat before we announce it
                confirmed = sim >= 0.55 or new or f is train_face or (ts - self.recent.get(pid, 0)) < 3.0
            else:
                # far: never creates people or learns; must be clearly one known person
                pid, sim, second = self.match_detail(f["emb"])
                if pid is None or sim < FAR_MATCH or sim - second < FAR_MARGIN:
                    if watch is None:
                        watch = self._watch_ids()
                    cand = max(((w, self.best_sim_to(f["emb"], w)) for w in watch),
                               key=lambda t: t[1], default=(None, 0.0))
                    if cand[0] is None or cand[1] < POSSIBLE_ALERT:
                        far_unknown.append({"box": f["box"], "px": f["px"]})
                        continue
                    pid, sim, possible = cand[0], cand[1], True
                elif ALERT_NEEDS_NEAR and pid in (watch if watch is not None else (watch := self._watch_ids())):
                    possible = True   # an alert is serious: from far away it's only ever "possible, get closer"
                confirmed = not possible and (sim >= FAR_SURE or (ts - self.recent.get(pid, 0)) < FAR_CONFIRM_S)
            if not possible:
                self.recent[pid] = ts
            person = self.db.person(pid) or {}
            if confirmed:
                self.db.touch_person(pid, ts)
            cx = (f["box"][0] + f["box"][2]) / 2 / max(1, W)
            results.append({
                "person_id": pid,
                "name": person.get("name"),
                "notes": person.get("notes", ""),
                "is_watch": bool(person.get("is_watch")),
                "watch_label": person.get("watch_label", ""),
                "sim": round(float(sim), 3),
                "quality": f["quality"],
                "box": f["box"],
                "px": f["px"],
                "tier": "near" if near else "far",
                "dir": "left" if cx < 0.36 else ("right" if cx > 0.64 else "ahead"),
                "new": new,
                "learned": learned,
                "training": f is train_face,
                "confirmed": bool(confirmed),
                "possible": possible,
            })
        self.last_far = far_unknown
        return results

    def _walkby(self, f, img, ts):
        """Remember an unknown face that isn't good enough to enroll yet. If the same face was
        seen in another frame in the last few seconds, enroll from the best look at them."""
        self.pending = [c for c in self.pending if ts - c["last"] <= WALKBY_WINDOW_S]
        if f["px"] < 30:
            return None
        x1, y1, x2, y2 = f["box"]
        h, w = img.shape[:2]
        mx, my = int((x2 - x1) * 0.6), int((y2 - y1) * 0.6)
        X1, Y1, X2, Y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
        look = {"f": f, "img": img[Y1:Y2, X1:X2].copy(), "box": [x1 - X1, y1 - Y1, x2 - X1, y2 - Y1]}
        match = None
        for c in self.pending:
            if float(c["emb"] @ f["emb"]) >= WALKBY_SAME:
                match = c
                break
        if match is None:
            self.pending.append({"emb": f["emb"], "hits": 1, "last": ts, "best": look})
            return None
        match["hits"] += 1
        match["last"] = ts
        if f["quality"] > match["best"]["f"]["quality"]:
            match["best"] = look
        b = match["best"]
        if b["f"]["quality"] >= WALKBY_MIN_QUALITY and b["f"]["px"] >= WALKBY_MIN_PX:
            self.pending = [c for c in self.pending if c is not match]
            return b["f"], b["img"], b["box"]
        return None

    # ---- curation helpers ----------------------------------------------
    def suggest_merges(self, limit=10):
        """Pairs of people whose photos look alike enough that they may be one person."""
        if len(self.gal_mat) < 2:
            return []
        groups = {}
        for i, pid in enumerate(self.gal_ids):
            groups.setdefault(pid, []).append(i)
        pids = list(groups)
        names = {r["id"]: (r["name"] or "").strip().lower() for r in self.db._run("SELECT id, name FROM people", fetch="all")}
        sims = self.gal_mat @ self.gal_mat.T
        out = []
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                key = tuple(sorted((pids[i], pids[j])))
                if key in self.dismissed:
                    continue
                na, nb = names.get(pids[i]), names.get(pids[j])
                if na and nb and na != nb:
                    continue   # you named them differently: they're different people
                s = float(sims[np.ix_(groups[pids[i]], groups[pids[j]])].max())
                if s >= SUGGEST_MERGE:
                    out.append({"a": key[0], "b": key[1], "sim": round(s, 3)})
        return sorted(out, key=lambda d: -d["sim"])[:limit]

    def dismiss_suggestion(self, a, b):
        self.dismissed.add(tuple(sorted((a, b))))

    # ---- utils ---------------------------------------------------------
    def _save_crop(self, img, box, name):
        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        mx, my = int((x2 - x1) * 0.4), int((y2 - y1) * 0.4)
        crop = img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        path = os.path.join(self.crops_dir, f"{name}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path
