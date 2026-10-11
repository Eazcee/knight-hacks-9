# Update: Street view (the 3D is gone)

## What changes
- **3D removed**: no point clouds, no 3D world, no depth model running (BEACON_DEPTH=1 brings back distance-to-people + obstacle reroutes if you ever want them; no UI).
- **Street view panel** (full width, under the live camera): photos of everywhere you walked, walk through them like Google Street View.
  - Photos **only while walking** (phone accelerometer), ~1 per 0.8 m, sharpest frame of each stretch. Sitting = none.
  - **Known people** (named / talked to / watchlist): a photo the moment they come into view, them boxed + named. **Strangers blurred.**
  - ↑/→ forward, ↓/← back (hold = keep walking), space = play. Ground arrows: ahead / behind; **green = another walk crosses here**, click to switch.
  - Time bar to jump to any moment (green ticks = people). Mini-map of the floor, people on this walk, list of walks.
  - **Big map**: your walks as blue lines, click one → street view there. Yellow pegman = the photo you're on. 📍 Navigate here = glasses guide you there.
  - Person card 📷 / map pin "📷 Street view" / sightings in a person's drawer → the photo where you saw them.
  - **Floors** from the phone's barometer (stairs + elevators); no barometer → voice "floor 2", "upstairs", "downstairs".
  - **Indoors detected by itself** (no good GPS for 15 s while walking) → steps + camera turns; GPS again outside.

## Paste this to your terminal Claude
Download beacon-street-update.zip first, then paste:

```
Make a new branch street-view off depth-3d (depth-3d keeps the 3D version). Unzip C:\Users\josep\Downloads\beacon-street-update.zip into the beacon server folder, overwriting everything. Then delete these, they're not used any more: world.py, test_world.py, the static\three folder, and the server's data\world folder. Restart the server and run test_street.py (ALL PASSED; it takes ~2.5 min). Commit and push street-view (data\ stays out of git).

Then two small app changes (send with every frame, next to lat/lng/acc/steps/heading):
1. moving=1 or moving=0: is the phone being carried by someone walking right now. Use the phone's accelerometer (TYPE_ACCELEROMETER): keep the last 1.5 s of |acceleration| samples; walking if their standard deviation is above 0.7 m/s², and once walking, stopped only when it drops below 0.35 (hysteresis). Also count a TYPE_STEP_DETECTOR event in the last 1.5 s as walking if the phone has that sensor. Show "Walking: yes/no" on the status screen.
2. pressure=<hPa>: the phone's barometer (TYPE_PRESSURE), averaged over the last 1 s. If the phone has no barometer, don't send it, and show "Barometer: none (floors by voice)" on the status screen.
Keep the glasses' Motion/compass off. Build, install. Then test: sit for 30 s (the dashboard's Street view header should say "stands still = no photos" and the photo count shouldn't move), walk 20 m (count goes up ~25, header shows "● recording"), and if you can, walk up one flight of stairs and tell me what floor the header shows.
```

Then Ctrl+F5 the dashboard.

## Notes
- Indoors positions come from steps + camera turns, anchored to where you came in. Walks from different days line up within a few meters indoors, so crossing arrows show up where walks pass within 4 m.
- Floor height assumed ~0.4 hPa (~3.5 m); `BEACON_FLOOR_HPA=0.45` for taller floors.
- Storage: ~50 KB a photo, ~60 MB per hour of walking.

## To revert
git checkout depth-3d (3D version) or main (safe demo build).
