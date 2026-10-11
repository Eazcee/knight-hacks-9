"""Depth from the glasses' single camera: a neural net estimates how far away every
pixel is (MiDaS small, ~66 MB, downloaded on first start). The net only knows
*relative* depth, so real meters come from the one thing we do know the size of:
a human face (~15 cm wide). Faces set the scale; the depth map gives everything
else the same scale.

What it feeds:
  * a live 3D point cloud of what the wearer sees (dashboard panel)
  * real distances to people (better pins on the map)
  * obstacles: a wall / closed gate / crowd filling the path ahead blocks that
    stretch of the map and the route re-plans, no sign or voice needed
"""
import os
import time
import urllib.request

import cv2
import numpy as np

MODEL_URL = os.environ.get("BEACON_DEPTH_URL", "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx")
MODEL_FILE = os.environ.get("BEACON_DEPTH_MODEL", "")      # any MiDaS / Depth-Anything style ONNX (ImageNet-normalized RGB in, inverse depth out)
HFOV_DEG = 75.0               # glasses camera horizontal field of view (approx.)
FACE_W_M = 0.15
CLOUD_W = 72                  # point cloud columns (rows follow the frame's aspect)
NEAR_M = 1.8                  # "obstacle" = most of the path band closer than this
OBSTACLE_FRAMES = 3           # ...for this many frames in a row
EYE_M = 1.6                   # eye height: with a level head the floor at the bottom of the frame is where the rays from here hit it
FACE_SCALE_TTL = 10.0         # a face sets the scale; after this many seconds the floor keeps it
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthEngine:
    def __init__(self, models_dir, providers=("CPUExecutionProvider",), ctx=-1):
        import onnxruntime as ort
        os.makedirs(models_dir, exist_ok=True)
        path = MODEL_FILE or os.path.join(models_dir, os.path.basename(MODEL_URL))
        if not os.path.exists(path):
            print("Downloading depth model (~66 MB)...")
            urllib.request.urlretrieve(MODEL_URL, path + ".part")
            os.replace(path + ".part", path)
        try:
            self.sess = ort.InferenceSession(path, providers=list(providers))
            self.device = "GPU" if ctx >= 0 and "CPU" not in providers[0] else "CPU"
        except Exception as e:
            print("depth: GPU init failed, using CPU:", e)
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.device = "CPU"
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        shp = inp.shape
        self.in_h = int(shp[2]) if isinstance(shp[2], int) else 256
        self.in_w = int(shp[3]) if isinstance(shp[3], int) else 256
        self.scale = None             # meters * rel  (set by faces, kept by the floor)
        self.scale_ts = 0.0
        self.face_ts = 0.0
        self.scale_src = None         # faces | floor
        self.shift = 0.0              # the net's output = scale/meters + shift (MiDaS is affine in inverse depth); the floor fit finds it
        self.near_streak = 0
        self.last = None              # last result dict
        self.ms = 0.0
        print(f"Beacon depth on {self.device} ({self.in_w}x{self.in_h})")

    # ---- inference ---------------------------------------------------
    def infer(self, img_bgr):
        """Relative inverse depth (bigger = closer) at a reduced resolution, plus the frame colors there."""
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(img_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.in_name: x})[0]
        rel = np.asarray(out, np.float32).reshape(out.shape[-2], out.shape[-1])
        cw = CLOUD_W
        ch = max(8, int(round(cw * h / w)))
        rel = cv2.resize(rel, (cw, ch), interpolation=cv2.INTER_AREA)
        rel = np.maximum(rel, 1e-3)
        color = cv2.cvtColor(cv2.resize(img_bgr, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.ms = (time.perf_counter() - t0) * 1000
        return rel, color

    # ---- scale from faces ----------------------------------------------
    def calibrate(self, rel, faces, W, H):
        """A face ~15 cm wide at `px` pixels is f*0.15/px meters away: that pins the scale."""
        f = W / (2 * np.tan(np.radians(HFOV_DEG / 2)))
        ks = []
        for fc in faces:
            x1, y1, x2, y2 = fc["box"]
            px = max(8, min(x2 - x1, y2 - y1))
            d_face = float(np.clip(f * FACE_W_M / px, 0.4, 12.0))
            r = self._rel_in_box(rel, (x1, y1, x2, y2), W, H)
            if r is not None:
                ks.append(d_face * max(r - self.shift, 1e-3))
        if ks:
            k = float(np.median(ks))
            self.scale = k if self.scale is None else 0.7 * self.scale + 0.3 * k
            self.scale_ts = self.face_ts = time.time()
            self.scale_src = "faces"
        return self.scale

    def calibrate_floor(self, rel):
        """No face around: the floor keeps the scale. The net's output is only relative *per frame*
        (inverse depth up to a scale), so without this every frame gets a different idea of a meter.
        With a level head, row v of the frame looks down at a known angle, so the floor there is a
        known distance away (eye height 1.6 m): the net's output at the bottom rows pins the scale."""
        ch, cw = rel.shape
        f = (cw / 2) / np.tan(np.radians(HFOV_DEG / 2))
        r0 = int(ch * 0.72)
        down = (np.arange(r0, ch, dtype=np.float32) - ch / 2 + 0.5) / f
        if r0 >= ch or down.min() <= 0:
            return self.scale
        q = down / EYE_M                                          # 1 / floor distance, per row
        r = np.median(rel[r0:, int(cw * 0.35):int(cw * 0.65)], axis=1)
        # scale only: fitting the shift as well (MiDaS is affine in inverse depth) was tried and made
        # things worse; the floor rows are too short a lever to extrapolate a shift to the end of a hall
        n = max(2, len(r) // 3)
        k, b = float(np.median(r[-n:] / q[-n:])), 0.0
        if self.scale is None:
            self.scale, self.shift = k, b
        elif time.time() - self.face_ts > FACE_SCALE_TTL:
            self.scale = 0.8 * self.scale + 0.2 * k
            self.shift = 0.8 * self.shift + 0.2 * b
        else:
            return self.scale
        self.scale_ts = time.time()
        self.scale_src = "floor"
        return self.scale

    def _rel_in_box(self, rel, box, W, H):
        ch, cw = rel.shape
        x1, y1, x2, y2 = box
        bx1, by1 = int(x1 / W * cw), int(y1 / H * ch)
        bx2, by2 = int(np.ceil(x2 / W * cw)), int(np.ceil(y2 / H * ch))
        # central part of the face (skip hair/background at the edges)
        mx, my = max(1, (bx2 - bx1) // 4), max(1, (by2 - by1) // 4)
        sub = rel[max(0, by1 + my):min(ch, by2 - my), max(0, bx1 + mx):min(cw, bx2 - mx)]
        if sub.size == 0:
            sub = rel[max(0, by1):min(ch, by2 + 1), max(0, bx1):min(cw, bx2 + 1)]
        if sub.size == 0:
            return None
        return float(np.median(sub))

    def metric(self, rel):
        """Meters per cell. Without any face yet, assume the middle of the scene is ~4 m away."""
        k = self.scale
        if k is None:
            return np.clip(4.0 * float(np.median(rel)) / rel, 0.2, 60.0)
        d = rel - self.shift
        return np.where(d > 1e-6, np.clip(k / np.maximum(d, 1e-6), 0.2, 60.0), 60.0).astype(np.float32)

    def distance_to(self, rel, box, W, H):
        r = self._rel_in_box(rel, box, W, H)
        if r is None:
            return None
        if self.scale is None:
            return float(np.clip(4.0 * float(np.median(rel)) / r, 0.2, 60.0))
        return float(np.clip(self.scale / max(r - self.shift, 1e-6), 0.2, 60.0))

    # ---- obstacles -----------------------------------------------------
    def obstacle(self, depth_m, faces, W, H):
        """Is the path straight ahead blocked? The band at chest height across the middle of the
        view: if most of it is closer than NEAR_M and it isn't a person's face, something's in the way."""
        ch, cw = depth_m.shape
        y0, y1 = int(ch * 0.35), int(ch * 0.70)
        x0, x1 = int(cw * 0.30), int(cw * 0.70)
        band = depth_m[y0:y1, x0:x1]
        # a person in the path is a person, not a wall: any face overlapping the band (or just
        # above it, where a close person's head is) means "people", never "blocked"
        for fc in faces:
            fx1, fy1, fx2, fy2 = fc["box"]
            bx1, by1 = fx1 / W * cw, fy1 / H * ch
            bx2, by2 = fx2 / W * cw, fy2 / H * ch
            if bx2 >= x0 and bx1 <= x1 and by2 >= y0 - (y1 - y0) * 0.5 and by1 <= y1:
                self.near_streak = 0
                return {"near": 0.0, "blocked": False, "people": True}
        near = float((band < NEAR_M).mean())
        # the scale is a guess at best: only believe "near" when the band is clearly closer than the
        # top of the view (ceiling / sky / the far end of the hall), not when the whole frame reads near
        top = depth_m[: max(1, int(ch * 0.2)), x0:x1]
        standout = float(np.median(band)) < 0.6 * float(np.median(top))
        self.near_streak = self.near_streak + 1 if (near >= 0.6 and standout) else 0
        return {"near": round(near, 2), "blocked": self.near_streak >= OBSTACLE_FRAMES, "people": False}

    # ---- point cloud ---------------------------------------------------
    def cloud(self, depth_m, color, W, H, ts, faces_3d=None):
        """Binary point cloud for the dashboard. Camera frame: x right, y up, z forward (meters).
        Layout: uint32 magic(0xBEAC0001), n, cw, ch; float32 scale; float64 ts; float32 xyz[n*3]; uint8 rgb[n*3]."""
        ch, cw = depth_m.shape
        f = (cw / 2) / np.tan(np.radians(HFOV_DEG / 2))
        us, vs = np.meshgrid(np.arange(cw, dtype=np.float32), np.arange(ch, dtype=np.float32))
        z = depth_m.astype(np.float32)
        x = (us - cw / 2 + 0.5) / f * z
        y = -(vs - ch / 2 + 0.5) / f * z
        keep = z < 40.0
        xyz = np.stack([x[keep], y[keep], z[keep]], axis=1).astype(np.float32)
        rgb = color[keep].astype(np.uint8)
        head = np.array([0xBEAC0001, len(xyz), cw, ch], np.uint32).tobytes()
        head += np.array([self.scale or 0.0], np.float32).tobytes() + np.array([ts], np.float64).tobytes()
        return head + xyz.tobytes() + rgb.tobytes()

    @staticmethod
    def face_3d(box, dist, W, H):
        """Where a face is in the camera frame, in meters (x right, y up, z forward)."""
        f = W / (2 * np.tan(np.radians(HFOV_DEG / 2)))
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return [round(float((cx - W / 2) / f * dist), 2), round(float(-(cy - H / 2) / f * dist), 2), round(float(dist), 2)]
