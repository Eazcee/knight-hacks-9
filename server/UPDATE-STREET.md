# Update: Street view v4

## What changes
- **Everyone, every time:** the side list now comes from your full sighting history, not just street-view photos. Kevin and Nagel (seen at ~4:40 AM, before people got photos) appear, with the camera frame from that moment slotted into your day. From now on every encounter is guaranteed a photo (a busy moment can no longer skip someone).
- **▶ Play** (bottom left of the viewer), speed button 1× / 2× / 4×, space bar plays/pauses.
- **Sorting people updates it live:** name an Unknown → every photo and list entry shows the name; merge an Unknown into someone → their entries become that person; delete someone → their boxes come off; Undo puts it back.

## Paste this to your terminal Claude
Download beacon-street-v4.zip first, then paste:

```
On branch street-view: unzip C:\Users\josep\Downloads\beacon-street-v4.zip into the beacon server folder, overwriting everything. Restart the server (on startup it adds your past sightings to the street view; watch the log for errors) and run test_street.py and test_training.py (both ALL PASSED). Commit and push street-view.
```

Then Ctrl+F5 the dashboard and open the street view: Kevin and Nagel should be in the list.
