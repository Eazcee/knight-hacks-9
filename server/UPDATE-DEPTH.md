# Update: 3D from the camera

## What it adds
- **3D view** panel on the dashboard: a live point cloud of what the glasses see, faces labeled with their distance. Drag to orbit, wheel to zoom.
- **Real distances** to people (a depth net + face size for scale), so pins land where people stand.
- **Automatic blocking:** while following a route, a wall / closed gate / anything filling the path ahead within 1.8 m blocks that stretch of the map and the route re-plans. People in the way don't count as walls.
- First start downloads the depth model (~66 MB, from GitHub) into data\models. Runs on the GPU next to the face model; falls back to CPU by itself.

## Paste this to your terminal Claude
Download beacon-depth-update.zip first, then paste:

```
Unzip C:\Users\josep\Downloads\beacon-depth-update.zip into the beacon server folder (overwrite server.py, dashboard.html, README.md; add depth.py, test_depth.py and static\three\). Restart the server and watch its startup log: it should print "Downloading depth model" once, then "Beacon depth on GPU" (or CPU). Run test_depth.py and test_demo.py (both ALL PASSED). While the glasses stream, tell me the "3D view" header text (device, ms) and the fps chip. Then push to knight-hacks-9 (data\models must stay out of git; server/data/ is already ignored).
If the depth model crashes the server, start it with the environment variable BEACON_DEPTH=0 and tell me the error.
```

Then Ctrl+F5 on the dashboard.

## To revert
git checkout -- server/server.py server/dashboard.html  (then restart the server), or set BEACON_DEPTH=0.
