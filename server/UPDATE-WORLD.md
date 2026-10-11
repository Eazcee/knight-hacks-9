# Update: the 3D world, indoors too (branch depth-3d)

## What it adds
- **World** mode in the 3D panel rebuilds the hallways you walked: at each place the map makes (every ~8 m) the depth net measures how far the walls are on each side, and the dashboard draws floor + walls along your path from that, with the photo from each spot above it, people standing where they were seen, and you (blue arrow). `▶ Walk it` flies through at eye height; `Raw depth` shows the unprocessed surfaces; `📸 Snapshot` / voice `snapshot` takes one right here; `⤢ Bigger` makes the panel full width.
- **Indoors:** the server now dead-reckons from the phone's **step counter** + the **glasses' compass heading** while GPS is rough, and ignores the 30–50 m Wi-Fi fixes that would yank you around. The first good GPS fix outside takes over again. Dashboard header shows `steps + compass` while it's in that mode.
- Snapshots live in `data\world\` (one small file per place; cleared with the map).

The one thing the app must add: send `heading=<degrees>` with every frame, from the glasses' compass. Without it the server has no idea which way you walk indoors (outdoors it still gets the heading from GPS).

## Paste this to your terminal Claude
Download beacon-world-update.zip first, then paste:

```
We're on branch depth-3d. Unzip C:\Users\josep\Downloads\beacon-world-update.zip into the beacon server folder (overwrite server.py, geo.py, depth.py, faces.py, db.py, dashboard.html, README.md, test_depth.py and static\three\; add world.py, test_world.py, UPDATE-WORLD.md). Restart the server, run test_world.py and test_depth.py (both ALL PASSED), then commit on depth-3d.

Then the app, one feature: send the glasses' compass heading with every frame.
1. The BirdSpotter sample this app was built from has GlassesAim.kt with bearingFromMotion(sample): a tilt-compensated compass bearing (degrees clockwise from magnetic north) from the glasses' Motion capability (magneticField + acceleration). Reuse that function (copy it into our app if it isn't already there).
2. Where we set up the glasses session, add the Motion capability at 5 Hz (session.addMotion(MotionConfiguration(samplingRate = MotionSamplingRate.HZ_5)) and start it, the same way the sample does), and in its sample callback compute bearingFromMotion(sample). Smooth it over the last ~1 s with a circular mean (average sin/cos of the angles, then atan2), and keep the latest value in a variable. Skip samples where magneticField is null.
3. In the frame upload, add heading=<that value, 0-360> as a query parameter alongside lat/lng/acc/steps. Keep sending steps (TYPE_STEP_COUNTER cumulative count) exactly as now.
4. Build, install, then with the glasses streaming: stand still and turn slowly in a circle; tell me what the dashboard map header shows for heading (it should say "sensor" and the number should follow you around the circle, north = 0). Then walk down a hallway indoors ~20 m and back: tell me whether the header says "steps + compass", whether places appear on the map every ~8 m, and what the 3D view's World button shows (how many snapshots, and whether it looks like the hallway).
5. Push depth-3d (data\ stays out of git).
```

Then Ctrl+F5 on the dashboard, click **World** in the 3D panel.

## Trying it without walking
`python test_world.py` simulates a hallway walk (steps + compass, rough GPS) and checks the snapshots. For a look at the dashboard: run the server, then in another terminal `python sim_walk.py` (outdoor GPS walk) → the World view fills with snapshots as places are made (the camera frames of the simulation are test images, so the surfaces are odd shapes — the real hallway is what the glasses will give you).

## What to expect indoors
- Position drift grows ~5 % of the distance walked (20 m → ~1 m). Fine for a hallway, a corner, a room; don't expect it to survive a whole building.
- Steel and electronics bend a compass 10–30° in some spots; the hallway may bend a bit with it. Walking steadily and keeping the glasses level helps.
- Steps come from the phone in your pocket; if the phone has no step sensor, there is no indoor positioning (the app skips it) — the World still works outdoors from GPS.

## To revert
git checkout main (the safe demo build). Or on depth-3d: git checkout -- server/ and rebuild the app from the previous commit.
