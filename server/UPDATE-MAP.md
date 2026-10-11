# Update: navigation on a satellite map

## What changed (server + dashboard)
- The old camera-keyframe map is gone. The map is now **GPS** (the phone sends `lat/lng/acc` with every frame).
- Dashboard shows a **satellite map with labels** (Esri tiles, no API key). Paste a Google Maps JavaScript API key in the box under the map to switch to Google's hybrid view. Everyone seen is a **pin with their clearest photo** where you saw them: **red** ring = seen, **green** = you talked (in view for 30 s; threshold and on/off in the map bar). Click a pin → note, Navigate, Open.
- The map builds itself as you walk (a place every ~8 m, paths with cost = distance × crowd). Tap a path to block it; "blocked" by voice or QR sign does the same; routes reroute.
- **Navigate to a person** (pin, person page, voice "take me to dana" / "find dana", or the alert photo): A* over the walked paths, or compass mode (straight line) if you never walked there. The glasses show `→ Dana / LEFT · 24 m / 120 m left · 2 min`. The route ends with **FOUND** the moment the camera recognizes them.
- Indoors with poor GPS the server dead-reckons from step counts if the app sends `steps` + `heading`.
- `sim_walk.py` fakes a GPS walk so you can rehearse the demo on the dashboard without walking.
- Includes everything from the earlier updates (range, photo pages, training).

## Paste this to your terminal Claude
Download beacon-map-update.zip first (it lands in Downloads), then paste:

```
Two parts: server, then the phone app. Finish whatever step you're on first.

PART A — update the Beacon server from C:\Users\josep\Downloads\beacon-map-update.zip
1. Stop the running server on port 8000.
2. Unzip over C:\Users\josep\AndroidStudioProjects\beacon, overwriting files. It adds geo.py, sim_walk.py, static\leaflet\, test files and UPDATE-MAP.md. Delete places.py (it is no longer used). Don't touch .venv or data.
3. Run test_demo.py, test_training.py and test_range.py with the venv's python. All three must say ALL PASSED.
4. Start the server again in the background on 0.0.0.0:8000 and make sure adb reverse tcp:8000 tcp:8000 is on.
5. Run "python sim_walk.py --reset --fast" once and tell me what the dashboard map at http://localhost:8000 shows (satellite tiles, a green route, photo pins).

PART B — send GPS from the BeaconApp Android app (C:\Users\josep\AndroidStudioProjects\BeaconApp)
1. Back up the app first (git commit, or copy the folder to BeaconApp-backup).
2. Add location: dependency com.google.android.gms:play-services-location, permissions ACCESS_FINE_LOCATION + ACCESS_COARSE_LOCATION in the manifest, runtime permission request on launch. Use FusedLocationProviderClient with PRIORITY_HIGH_ACCURACY and a 1000 ms interval while streaming. Keep the latest Location in memory.
3. On every frame upload, add query parameters to the existing POST /frame URL: lat, lng, acc (location.accuracy in meters). If there is no fix yet, send the frame without them.
4. Steps: register a TYPE_STEP_COUNTER sensor listener (permission ACTIVITY_RECOGNITION, request at runtime) and send the cumulative count as steps=<n> with each frame. If the phone has no step sensor, skip it.
5. Heading: if the Device Access Toolkit's Motion capability gives an orientation that is referenced to north (check the SDK docs; if it's only relative/gyro, don't use it), convert it to compass degrees (0 = north, clockwise) and send heading=<deg>. Otherwise send no heading; the server works out the heading from the direction of walking.
6. Nothing changes on the glasses UI: navigation arrives as the same "display" card that is already rendered (title + lines). Make sure a title like "→ Dana" and 3 lines fit; shorten lines if they wrap.
7. Build, install on the phone, then walk outside for a minute with the app streaming and tell me: the accuracy shown in the dashboard map header (±N m), whether "heading" shows "sensor" or "gps", and whether places appear on the map as you walk.

Don't edit the server files yourself (server.py, geo.py, faces.py, db.py, dashboard.html); tell me if something looks wrong there. If the SDK doesn't match what I describe, follow the SDK docs and tell me what you did differently.
```

## Try it in 2 minutes once it's installed
1. Go outside with the glasses streaming. Walk 30 m, turn, walk 30 m. The map draws your path and makes places.
2. Look at a friend for 30 s: their pin turns green. Click the pin, write a note.
3. Walk 50 m away. On the dashboard click their pin → Navigate (or say "take me to <name>"). Follow the glasses. When you see them, the glasses say FOUND.
4. Say "blocked" while on a route: it reroutes the other way if you've walked one, otherwise goes direct.
