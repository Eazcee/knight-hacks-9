"""Stream frames to the Beacon server without the glasses.

  python webcam_client.py                      # laptop webcam
  python webcam_client.py --video walk.mp4     # replay a video recorded on the Ray-Bans
  python webcam_client.py --server http://192.168.1.20:8000 --fps 3 --show
  python webcam_client.py --lat 28.6024 --lng -81.2001   # fake GPS so faces get pins on the map

Replaying a video you recorded with the glasses' normal capture is the fastest
way to build a map of the venue before the live SDK path works.
"""
import argparse
import time

import cv2
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8000")
ap.add_argument("--video", default=None, help="video file instead of webcam")
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--fps", type=float, default=2.0, help="frames per second to send")
ap.add_argument("--show", action="store_true", help="draw boxes in a window")
ap.add_argument("--lat", type=float, default=None, help="fake GPS position so people land on the map")
ap.add_argument("--lng", type=float, default=None)
ap.add_argument("--width", type=int, default=0, help="downscale before sending (0 = send as is)")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video if args.video else args.cam)
if not cap.isOpened():
    raise SystemExit("could not open camera/video")
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
skip = max(1, int(round(src_fps / args.fps))) if args.video else 1
i = 0
last_send = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    if args.video and i % skip:
        continue
    if not args.video and time.time() - last_send < 1.0 / args.fps:
        if args.show:
            cv2.imshow("beacon", frame)
            if cv2.waitKey(1) == 27:
                break
        continue
    last_send = time.time()
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (args.width, int(h * args.width / w))) if args.width and w > args.width else frame
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    params = {"lat": args.lat, "lng": args.lng, "acc": 5} if args.lat is not None else {}
    try:
        r = requests.post(f"{args.server}/frame", data=jpg.tobytes(), params=params,
                          headers={"Content-Type": "image/jpeg"}, timeout=10).json()
    except Exception as e:
        print("server error:", e)
        continue
    d = r["display"]
    who = ", ".join((p["name"] or p["person_id"]) + f" {p['sim']:.2f}" for p in r["people"])
    ev = " ".join(e["type"] + ":" + str(e.get("name", "")) for e in r["place"]["events"])
    print(f"[{d['level']:5}] {d['title']} | {' / '.join(d['lines'])} | place={r['place']['place']} {ev} | people: {who}")
    if args.show:
        for p in r["people"]:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(small, (x1, y1), (x2, y2), (0, 0, 255) if p["is_watch"] else (0, 255, 0), 2)
            cv2.putText(small, p["name"] or p["person_id"], (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(small, f"{d['title']} {' / '.join(d['lines'])}"[:80], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("beacon", small)
        if cv2.waitKey(1) == 27:
            break
cap.release()
