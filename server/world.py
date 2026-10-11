"""The 3D world: snapshots of the depth cloud, one per place, turned to the heading the
wearer had and pinned where they stood. Laid out together they're a rough 3D model of
everywhere you walked, outdoors (GPS) or indoors (steps + the glasses' compass).

Each snapshot keeps its grid (rows x cols of the depth map), so the dashboard can draw it
as a surface (triangles between neighbouring cells) instead of loose dots: a hallway looks
like a hallway.

World frame = the map's local meters: x east, y north, z up (eye height 1.6 m above ground).
"""
import glob
import json
import math
import os

import numpy as np

EYE_M = 1.6
STAMP_MAX_M = 8.0         # keep the near field only: the net's far field is a smear, and the next place's snapshot covers it
U_MIN, U_MAX = -1.2, 4.0  # floor to ceiling (meters above the ground, with slack for the camera's unknown vertical FOV); the rest is sky / the net guessing
STAMP_STRIDE = 2          # keep every 2nd cell of the depth map (72x128 -> 36x64 vertices)
MAX_STAMPS = 400
MAGIC = 0xBEAC0003
WALL_U = (0.4, 2.0)       # wall evidence: cells at body height...
WALL_Z = (0.7, 7.0)       # ...0.7-7 m ahead...
WALL_X = (0.35, 3.0)      # ...0.35-3 m to the side (farther than 3 m = open space, not a hallway wall)


def corridor(xc, yc, z, color):
    """What a hallway looks like from here, measured from the depth map (camera frame:
    xc right, yc up, z forward, meters): the distance to the wall on each side, the walls'
    and floor's colors, how far each wall runs and how far it's open straight ahead.
    A side wall is a band of cells at body height whose sideways distance hardly changes
    while their depth does; a wall you're facing is the opposite and doesn't count."""
    u = yc + EYE_M
    band = (u > WALL_U[0]) & (u < WALL_U[1]) & (z > WALL_Z[0]) & (z < WALL_Z[1])
    out = {}
    for side, sel in (("l", band & (xc < -WALL_X[0])), ("r", band & (xc > WALL_X[0]))):
        if sel.sum() < 12:
            continue
        ax, zz = np.abs(xc[sel]), z[sel]
        half = float(np.median(ax))
        iqr_x = float(np.subtract(*np.percentile(ax, [75, 25])))
        iqr_z = float(np.subtract(*np.percentile(zz, [75, 25])))
        if half > WALL_X[1] or iqr_z < 0.5 or iqr_x > 0.5 * iqr_z:
            continue   # open space, or a wall in front of you rather than beside you
        near = sel & (np.abs(np.abs(xc) - half) < 0.4)
        out[side] = round(half, 2)
        out[side + "_len"] = round(float(np.percentile(z[near], 97)) if near.any() else float(np.percentile(zz, 97)), 1)
        out[side + "_rgb"] = [int(v) for v in np.median(color[near if near.any() else sel], axis=0)]
    floor = (u < 0.25) & (z < WALL_Z[1])
    if floor.sum() >= 12:
        out["floor_rgb"] = [int(v) for v in np.median(color[floor], axis=0)]
    ch, cw = z.shape
    mid = z[int(ch * 0.35):int(ch * 0.65), int(cw * 0.4):int(cw * 0.6)]
    out["ahead"] = round(float(min(np.median(mid), 20.0)), 1)
    return out


class World:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.stamps = {}      # node_id -> dict(xyz[gh,gw,3] (NaN = no point), rgb[gh,gw,3], depth[gh,gw], x, y, heading, ts)
        self.rev = 0          # bumps on every change (the dashboard refetches when it moves)
        self._blob = None
        for f in sorted(glob.glob(os.path.join(path, "n*.npz"))):
            try:
                d = np.load(f)
                if d["xyz"].ndim != 3:
                    continue
                nid = int(os.path.basename(f)[1:-4])
                self.stamps[nid] = {"xyz": d["xyz"], "rgb": d["rgb"], "depth": d["depth"], "x": float(d["x"]), "y": float(d["y"]),
                                    "heading": float(d["heading"]), "ts": float(d["ts"]),
                                    "walls": json.loads(str(d["walls"])) if "walls" in d else {}}
            except Exception:
                pass

    def has(self, nid):
        return nid in self.stamps

    def photo_path(self, nid):
        return os.path.join(self.path, f"n{nid}.jpg")

    def add(self, nid, depth_m, color, x, y, heading, ts, photo=None, info=None):
        """Freeze this frame's cloud at place `nid`, standing at (x, y) facing `heading`.
        `photo` = the camera frame as JPEG bytes (shown above the place in the 3D view)."""
        ch, cw = depth_m.shape
        f = (cw / 2) / math.tan(math.radians(75.0 / 2))
        us, vs = np.meshgrid(np.arange(cw, dtype=np.float32), np.arange(ch, dtype=np.float32))
        z = depth_m.astype(np.float32)
        xc = (us - cw / 2 + 0.5) / f * z          # camera: right
        yc = -(vs - ch / 2 + 0.5) / f * z         # camera: up
        walls = corridor(xc, yc, z, color)
        walls.update(info or {})
        s = STAMP_STRIDE
        xc, yc, z = xc[::s, ::s], yc[::s, ::s], z[::s, ::s]
        h = math.radians(heading)
        e = xc * math.cos(h) + z * math.sin(h) + x
        n = -xc * math.sin(h) + z * math.cos(h) + y
        u = yc + EYE_M
        xyz = np.stack([e, n, u], axis=2).astype(np.float32)
        xyz[(z >= STAMP_MAX_M) | (u < U_MIN) | (u > U_MAX)] = np.nan
        rgb = np.ascontiguousarray(color[::s, ::s]).astype(np.uint8)
        depth = np.ascontiguousarray(z).astype(np.float32)
        self.stamps[nid] = {"xyz": xyz, "rgb": rgb, "depth": depth, "x": x, "y": y, "heading": heading, "ts": ts, "walls": walls}
        if photo:
            with open(self.photo_path(nid), "wb") as fh:
                fh.write(photo)
        if len(self.stamps) > MAX_STAMPS:
            old = min(self.stamps, key=lambda k: self.stamps[k]["ts"])
            self.remove(old)
        np.savez_compressed(os.path.join(self.path, f"n{nid}.npz"), xyz=xyz, rgb=rgb, depth=depth, x=x, y=y, heading=heading, ts=ts,
                            walls=np.array(json.dumps(walls)))
        self._blob = None
        self.rev += 1
        return int(np.isfinite(xyz[..., 0]).sum())

    def move(self, nid, x, y, heading):
        """A place was repositioned (a turn recognised late): carry its snapshot along, re-turned."""
        s = self.stamps.get(nid)
        if not s:
            return
        dh = math.radians(heading - s["heading"])
        e, n, u = s["xyz"][..., 0] - s["x"], s["xyz"][..., 1] - s["y"], s["xyz"][..., 2]
        s["xyz"] = np.stack([e * math.cos(dh) + n * math.sin(dh) + x, -e * math.sin(dh) + n * math.cos(dh) + y, u], axis=2).astype(np.float32)
        s["x"], s["y"], s["heading"] = x, y, heading
        np.savez_compressed(os.path.join(self.path, f"n{nid}.npz"), xyz=s["xyz"], rgb=s["rgb"], depth=s["depth"], x=x, y=y,
                            heading=heading, ts=s["ts"], walls=np.array(json.dumps(s.get("walls") or {})))
        self._blob = None
        self.rev += 1

    def remove(self, nid):
        self.stamps.pop(nid, None)
        for f in (os.path.join(self.path, f"n{nid}.npz"), self.photo_path(nid)):
            try:
                os.remove(f)
            except OSError:
                pass
        self._blob = None
        self.rev += 1

    def clear(self):
        for nid in list(self.stamps):
            self.remove(nid)

    def n_points(self):
        return int(sum(np.isfinite(s["xyz"][..., 0]).sum() for s in self.stamps.values()))

    def blob(self):
        """All snapshots as one binary, grids kept so the viewer can draw surfaces.
        uint32 magic(0xBEAC0003), n_stamps, n_points, 0; then per snapshot:
          uint32 node, gw, gh, 0; float32 xyz[gh*gw*3] (NaN = no point); float32 depth[gh*gw]; uint8 rgb[gh*gw*3] padded to 4 bytes."""
        if self._blob is None:
            parts = [np.array([MAGIC, len(self.stamps), self.n_points(), 0], np.uint32).tobytes()]
            for nid, s in self.stamps.items():
                gh, gw = s["depth"].shape
                parts.append(np.array([nid, gw, gh, 0], np.uint32).tobytes())
                xyz = s["xyz"].copy()   # older snapshots were saved with a longer reach: trim them the same way
                xyz[(s["depth"] >= STAMP_MAX_M) | (xyz[..., 2] < U_MIN) | (xyz[..., 2] > U_MAX)] = np.nan
                parts.append(xyz.tobytes())
                parts.append(s["depth"].tobytes())
                rgb = s["rgb"].tobytes()
                parts.append(rgb + b"\0" * (-len(rgb) % 4))
            self._blob = b"".join(parts)
        return self._blob

    def meta(self):
        return [{"node": k, "x": round(s["x"], 1), "y": round(s["y"], 1), "heading": round(s["heading"]),
                 "n": int(np.isfinite(s["xyz"][..., 0]).sum()), "ts": s["ts"], "walls": s.get("walls") or {},
                 "photo": f"/world/photo/{k}.jpg?{int(s['ts'])}" if os.path.exists(self.photo_path(k)) else None}
                for k, s in self.stamps.items()]
