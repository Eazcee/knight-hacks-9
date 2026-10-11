"""Synthetic hallway frames for the tests (portrait 720x1280, 75° HFOV): a corridor with doors, lights, floor
lines and wall texture, rendered from any yaw (camera turned right by `yaw` degrees), with a side opening."""
import math
import os
import sys
import time

import cv2
import numpy as np

W, H = 720, 1280
F = (W / 2) / math.tan(math.radians(37.5))
CX, CY = W / 2, H / 2


YAW = {"deg": 0.0}   # camera turned right by this much (clockwise from above)
NEAR = 0.15


def cam(X, Y, Z):
    t = math.radians(YAW["deg"])
    return (X * math.cos(t) - Z * math.sin(t), Y, X * math.sin(t) + Z * math.cos(t))


def proj(X, Y, Z):
    """3D corridor point -> pixel (None if behind the camera). Polygons go through poly(), which clips."""
    xc, yc, zc = cam(X, Y, Z)
    if zc < NEAR:
        return None
    return int(round(CX + F * xc / zc)), int(round(CY - F * yc / zc))


def poly(img, pts, color):
    """pts = 3D corridor points; clipped against the near plane, then projected and filled."""
    pts = [cam(*p) for p in pts]
    out = []
    n = len(pts)
    for i in range(n):          # Sutherland-Hodgman against z >= NEAR
        a, b = pts[i], pts[(i + 1) % n]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), NEAR))
    if len(out) < 3:
        return
    px = [(int(round(CX + F * x / z)), int(round(CY - F * y / z))) for x, y, z in out]
    cv2.fillPoly(img, [np.array(px, np.int32)], color)


def hallway(seed=0, length=22.0, wall_l=(150, 140, 125), wall_r=(140, 150, 135), turn=None, yaw=0.0):
    YAW["deg"] = yaw
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    zn, zf = 0.6, length
    floor_y, ceil_y, hw = -1.6, 1.1, 1.3
    # end wall
    poly(img, [(-hw, floor_y, zf), (hw, floor_y, zf), (hw, ceil_y, zf), (-hw, ceil_y, zf)], (70, 70, 75))
    # fog-shaded slabs along z
    zs = np.geomspace(zn, zf, 60)
    for z0, z1 in zip(zs, zs[1:]):
        k = max(0.25, 1 - (z0 / zf) ** 0.8)
        poly(img, [(-hw, floor_y, z0), (hw, floor_y, z0), (hw, floor_y, z1), (-hw, floor_y, z1)], tuple(int(c * k) for c in (120, 115, 110)))
        poly(img, [(-hw, ceil_y, z0), (hw, ceil_y, z0), (hw, ceil_y, z1), (-hw, ceil_y, z1)], tuple(int(c * k) for c in (210, 210, 205)))
        poly(img, [(-hw, floor_y, z0), (-hw, ceil_y, z0), (-hw, ceil_y, z1), (-hw, floor_y, z1)], tuple(int(c * k) for c in wall_l))
        poly(img, [(hw, floor_y, z0), (hw, ceil_y, z0), (hw, ceil_y, z1), (hw, floor_y, z1)], tuple(int(c * k) for c in wall_r))
    # wall texture anchored to the world (baseboard, a handrail, small marks), so frames track each other like a real wall
    for side in (-1, 1):
        x = side * hw
        for yy in (floor_y + 0.1, floor_y + 0.9):
            for z0, z1 in zip(zs[::3], zs[3::3]):
                pa, pb = proj(x, yy, z0), proj(x, yy, z1)
                if pa and pb:
                    cv2.line(img, pa, pb, (70, 60, 50), 2)
        for z in np.arange(1.0, zf, 0.7):
            for yy in (floor_y + 1.4, floor_y + 2.3):
                p = proj(x, yy, z)
                if p:
                    cv2.circle(img, p, max(1, int(6 / max(0.5, z) * 2)), (40, 40, 40), -1)
    # floor tile lines, doors, lights
    for z in np.arange(1.0, zf, 1.0):
        pa, pb = proj(-hw, floor_y, z), proj(hw, floor_y, z)
        if pa and pb:
            cv2.line(img, pa, pb, (90, 85, 80), 1)
    for z in np.arange(2.0, zf - 1, 3.5):
        for side in (-1, 1):
            x = side * hw
            k = max(0.3, 1 - (z / zf) ** 0.8)
            col = tuple(int(c * k) for c in (60, 90, 140))
            poly(img, [(x, floor_y, z), (x, floor_y + 2.1, z), (x, floor_y + 2.1, z + 0.95), (x, floor_y, z + 0.95)], col)
    for z in np.arange(1.5, zf, 3.0):
        k = max(0.4, 1 - (z / zf) ** 0.8)
        poly(img, [(-0.3, ceil_y - 0.02, z), (0.3, ceil_y - 0.02, z), (0.3, ceil_y - 0.02, z + 1.2), (-0.3, ceil_y - 0.02, z + 1.2)], tuple(int(c * k) for c in (255, 250, 235)))
    if turn:   # an opening in one wall some meters ahead (a side corridor)
        side, z = turn
        x = side * hw
        poly(img, [(x, floor_y, z), (x, ceil_y, z), (x, ceil_y, z + 2.2), (x, floor_y, z + 2.2)], (40, 40, 45))
    noise = rng.integers(0, 18, (H, W, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return cv2.GaussianBlur(img, (3, 3), 0)
