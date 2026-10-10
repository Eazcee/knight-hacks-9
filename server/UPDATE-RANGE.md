# Update: longer range + Scan

## What changed (server)
- Detector keeps the frame's shape and runs bigger (352×640 for a portrait frame instead of a squeezed 416×416 square).
- Range tiers by face size: ≥56 px can become a new person, ≥40 px full recognition, **18–40 px "far"** = only matched to people already known (one person must clearly win, and it has to repeat within 6 s).
- Alert targets are never confirmed from far away: a far match → glasses say **"Possible: … get closer to confirm"** (never a real alert, never logged). The real alert fires up close or from a Scan.
- **`POST /scan`**: send one big photo (the glasses' 12 MP capture). It's searched in full-resolution tiles, every face matched, card held on the glasses ~8 s. Voice "scan" → `/command` returns `action: "scan"`.
- Dashboard: every live box shows its pixel size, grey boxes = too far/unclear, header shows the frame size the server receives, "Scan a photo" button to test Scan with any photo.
- Alert card and Scan say where: left / ahead / right.

This zip also contains everything from the previous update (beacon-update.zip). If you never installed that one, this replaces it.

## Paste this to your terminal Claude
Download beacon-range-update.zip first (it lands in Downloads), then paste:

```
Two parts. Part A is the server, Part B is the phone app. Finish whatever step you're on first.

PART A — update the Beacon server from C:\Users\josep\Downloads\beacon-range-update.zip
1. Stop the running server (the background uvicorn on port 8000).
2. If you changed any of these files in C:\Users\josep\AndroidStudioProjects\beacon yourself, save a copy first: server.py, faces.py, db.py, places.py, dashboard.html, README.md.
3. Unzip over C:\Users\josep\AndroidStudioProjects\beacon (overwrite, add test_range.py and UPDATE-RANGE.md). Don't touch .venv or data.
4. Re-apply any of your own changes from step 2.
5. Run test_demo.py, test_training.py and test_range.py with the venv's python. All three must say ALL PASSED.
6. Start the server again in the background on 0.0.0.0:8000; make sure adb reverse tcp:8000 tcp:8000 is still on.

PART B — longer range in the BeaconApp Android app (C:\Users\josep\AndroidStudioProjects\BeaconApp)
1. Camera stream resolution: switch from MEDIUM (504x896) to HIGH (720x1280). Keep the frame rate low (7 fps is fine, we only upload ~2-3 fps). Use the Device Access Toolkit's own quality/frame-rate settings; check the SDK docs or the CameraAccess sample for the exact names.
2. Stop shrinking frames before upload. Send each frame at its native size as JPEG quality ~80. Keep the existing throttle (~every 400 ms) and the "one request in flight" behavior.
3. Log the real frame size you get (bitmap width x height) to logcat once every ~10 s. The SDK may lower resolution when Bluetooth bandwidth is tight; I want to know what actually arrives. The dashboard header also shows "frame WxH".
4. If HIGH makes the stream stutter or uploads take > 500 ms, fall back to MEDIUM but still don't shrink.
5. Add SCAN:
   - A "Scan" button on the phone screen, plus a glasses input if one is free (touchpad tap/swipe; the Neural Band doesn't work yet), plus voice: if POST /command returns "action": "scan", run a scan.
   - Scan = capture ONE full-resolution photo from the glasses with the toolkit's photo capture (see how the CameraAccess sample's capture-photo button does it), POST the JPEG bytes to http://127.0.0.1:8000/scan with Content-Type image/jpeg and a 30 s timeout, then render the returned "display" card on the glasses exactly like /frame cards.
   - Show "Scanning… hold still" on the glasses while it runs.
   - Check the SDK docs / release notes for whether photo capture works while the video stream is running on Ray-Ban Display. If it can't, do: pause frame uploads → stop stream → capture → restart stream → resume. Make sure the stream always comes back even if capture fails.
   - Log the photo size and how long the scan took.
6. Build, install on the phone, test: open http://localhost:8000, press Scan, confirm a "Scan · N faces" card appears on the glasses and in the dashboard's "On the glasses" panel.
7. Tell me: frame size shown in the dashboard header, the scan photo size, scan time, and anything that didn't work.
```

## Then measure it (2 minutes, worth a slide)
Have a named friend stand at 3, 6, 10, 15, 20 ft. At each distance read the face's **px** on the dashboard live view and whether their name shows (blue-ish box = far tier). Then press Scan at the distance where video stops naming them.
