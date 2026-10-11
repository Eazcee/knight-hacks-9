"""Turns from the camera. Indoors a compass is bent by steel and GPS is a guess, but the
camera knows when you turn: the whole scene slides sideways. Walking straight it doesn't
(things expand from the middle instead). So heading indoors = where you started + the turns
the camera saw, each snapped to 45° (hallways meet at right angles): a straight hall comes
out straight, a corner comes out as a corner.

A glance (head turns, body doesn't) is told apart from a real turn in two ways: the new
direction has to hold for a couple of seconds, and if the phone's compass is available the
phone (in your hand or pocket, following the body) has to have turned too.
"""
import math
from collections import deque

import cv2
import numpy as np

SMALL_W = 240                 # frame width the flow runs at
HFOV_DEG = 75.0
TURN_MIN_DEG = 35.0           # less than this is a wobble, not a turn
SNAP_DEG = 45.0
HOLD_S = 2.0                  # the new heading must hold this long before the turn counts
ROT_FRAME_DEG = 1.0           # per-frame rotation above this = "turning"
COMPASS_AGREE_DEG = 25.0      # if a compass is around, it must have moved at least this much, same way
MIN_POINTS = 15


class TurnTracker:
    def __init__(self):
        self.prev = None
        self.prev_ts = None
        self.acc = 0.0
        self.acc_start_ts = None
        self.last_rot_ts = None
        self.compass_hist = deque(maxlen=300)   # (ts, deg)
        self.last_dyaw = 0.0
        self.f = (SMALL_W / 2) / math.tan(math.radians(HFOV_DEG / 2))

    def compass(self, ts, deg):
        self.compass_hist.append((ts, float(deg) % 360))

    def _compass_at(self, ts):
        best = None
        for t, d in self.compass_hist:
            if t <= ts + 0.5:
                best = d
        return best

    def update(self, img_bgr, ts):
        """Feed every processed frame. Returns (turn_deg, mid_ts): the committed turn (clockwise +, usually 0)
        and when it happened (the middle of the rotation), so steps taken since can be re-attributed."""
        h, w = img_bgr.shape[:2]
        sh = max(32, int(round(h * SMALL_W / w)))
        g = cv2.cvtColor(cv2.resize(img_bgr, (SMALL_W, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        dyaw = 0.0
        if self.prev is not None and self.prev.shape == g.shape and (self.prev_ts is None or ts - self.prev_ts < 2.0):
            # features in the far field (middle rows, inner columns): rotation shows there, walking hardly does
            y0, y1 = int(sh * 0.15), int(sh * 0.7)
            x0, x1 = int(SMALL_W * 0.1), int(SMALL_W * 0.9)
            pts = cv2.goodFeaturesToTrack(self.prev[y0:y1, x0:x1], 200, 0.01, 6)
            if pts is None or len(pts) < MIN_POINTS:            # blank wall close up: take anything the frame has
                y0, x0 = 0, 0
                pts = cv2.goodFeaturesToTrack(self.prev, 200, 0.005, 6)
            if pts is not None and len(pts) >= MIN_POINTS:
                pts = pts.reshape(-1, 2) + np.array([x0, y0], np.float32)
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, pts.reshape(-1, 1, 2), None, winSize=(25, 25), maxLevel=4)
                ok = st.reshape(-1) == 1
                if ok.sum() >= MIN_POINTS:
                    d = nxt.reshape(-1, 2)[ok] - pts[ok]
                    d = d[np.abs(d[:, 1]) < 20]          # sideways motion only
                    if len(d) >= MIN_POINTS:
                        dx = float(np.median(d[:, 0]))
                        dyaw = -math.degrees(math.atan(dx / self.f))   # scene slides left = you turned right
        self.prev, self.prev_ts, self.last_dyaw = g, ts, dyaw

        if abs(dyaw) >= ROT_FRAME_DEG:
            if self.acc_start_ts is None:
                self.acc_start_ts = ts
            self.last_rot_ts = ts
        self.acc += dyaw
        self.acc = max(-270.0, min(270.0, self.acc))
        if self.acc_start_ts is None or self.last_rot_ts is None or ts - self.last_rot_ts < HOLD_S:
            return 0.0, None
        # the rotation stopped HOLD_S ago: was it a turn?
        turn = 0.0
        if abs(self.acc) >= TURN_MIN_DEG:
            c0, c1 = self._compass_at(self.acc_start_ts - 0.5), self._compass_at(ts)
            agree = True
            amount = self.acc
            if c0 is not None and c1 is not None and self.compass_hist and ts - self.compass_hist[-1][0] < 5:
                dc = (c1 - c0 + 540) % 360 - 180
                agree = abs(dc) >= COMPASS_AGREE_DEG and (dc > 0) == (self.acc > 0)
                if agree and abs(dc) > abs(self.acc) + 30:
                    amount = dc            # a fast spin loses the camera track for a moment: the compass has the full swing
            if agree:
                turn = round(amount / SNAP_DEG) * SNAP_DEG
        mid = (self.acc_start_ts + self.last_rot_ts) / 2
        self.acc, self.acc_start_ts, self.last_rot_ts = 0.0, None, None
        return float(turn), mid

    def reset(self):
        self.prev = None
        self.acc, self.acc_start_ts, self.last_rot_ts = 0.0, None, None
