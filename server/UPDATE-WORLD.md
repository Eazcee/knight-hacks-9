# Update: Street view + real indoor positioning (branch depth-3d)

## What it adds
- **Indoors** button on the map (voice `indoors` / `outdoors`): GPS ignored completely. Position = step counter along a heading the **camera** keeps: a corner shows up as the whole frame sliding sideways; walking straight doesn't. Glances are ignored (the new direction must hold ~2 s and the phone must have turned too). Turns snap to 45°, so a straight hallway stays straight. Test: 20 m, a glance, a 90° corner, 10 m — exact to 0.1 m with GPS jumping ±8 m and the compass wobbling.
- **Street view** mode in the 3D panel: a photo every ~3 m (and after every turn; 📷 / voice `photo` any time), pinned where it was taken. ‹ › / ← → to walk through, ▶ Play, mini-map with every photo's spot, places, people and you; click to jump. Who was in view is on each photo.
- World (3D boxes) stays as a third mode. New files: `odometry.py`, `shots.py`, `test_street.py`, `testdata/synth.py` (a synthetic hallway the tests walk through).

## Paste this to your terminal Claude
Download beacon-world-update.zip first, then paste:

```
On branch depth-3d: unzip C:\Users\josep\Downloads\beacon-world-update.zip into the beacon server folder, overwriting everything (new files: odometry.py, shots.py, test_street.py, testdata\). Restart the server, run test_street.py and test_world.py (both ALL PASSED), commit and push depth-3d.
In the app, nothing changes: keep sending steps and the phone's compass heading with every frame.
```

## How to walk it
1. Ctrl+F5 the dashboard. On the map press **Indoors** (it turns green, header says "indoors · steps + camera").
2. Map → **Clear map** once (the old GPS-scattered places are gone; people stay).
3. Walk the hallway at a steady pace. Turn corners like you mean it (the body turns, not just the head). A photo appears every ~3 m; a place every ~8 m.
4. 3D panel → **Street view**: step through with ‹ ›. Mini-map = the hallway as walked.
5. Going outside: press Indoors again (or say "outdoors") and GPS takes over.

## Honest limits
- Steps: 0.72 m each. Tall or short strides drift the length ±10 %.
- Compass only sets the starting direction; if it's 20° off, the whole indoor track is rotated 20° on the map. Indoors that's fine.
- Turns under ~35° aren't turns (a slow bend in a corridor is drawn straight).
- Spinning fast in place (180° in under a second) can lose the camera; the phone's compass fills in if it's sending.

## To revert
git checkout main (the safe demo build).
