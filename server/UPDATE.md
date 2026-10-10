# Update: photo pages, training sets, merge, train mode, better dashboard

## What changed
- New dashboard: live camera view, people grid with big photos, click a person for their page,
  full-screen photo viewer (with the full camera frame and the face boxed), merge suggestions,
  train mode, alert photo upload, fixed map.
- Every face gets a quality score; only good faces create people or train them.
- Each person keeps up to 16 training photos (varied angles). You can remove bad ones,
  add good sightings, move wrong ones, and merge people who got split.
- Train mode: point the glasses at someone for 20 s while they turn their head.
- The map no longer makes a new place every time you turn your head (it waits for real step
  counts, QR anchors, or "Mark place here").

Files: server.py, faces.py, db.py, places.py, dashboard.html (new), test_training.py (new), README.md.
The database upgrades itself on start; existing people and photos are kept.

## Paste this to your terminal Claude
Download beacon-update.zip first (it lands in Downloads), then paste:

```
Update the Beacon server from C:\Users\josep\Downloads\beacon-update.zip:
1. Stop the running server (the background uvicorn on port 8000).
2. Check whether you changed any of these files in C:\Users\josep\AndroidStudioProjects\beacon since setup: server.py, faces.py, db.py, places.py, README.md. If you did, save a copy of your version first.
3. Unzip the update over C:\Users\josep\AndroidStudioProjects\beacon (overwrite those files, add dashboard.html and test_training.py). Do NOT touch .venv or data.
4. If you had your own changes in step 2, re-apply them on top of the new files.
5. Run test_training.py and test_demo.py with the venv's python and tell me if both say ALL PASSED.
6. Start the server again in the background on 0.0.0.0:8000, and make sure the phone's adb reverse is still on.
Then continue with whatever step of PROMPT.md you were on.
```

After it restarts: open http://localhost:8000 and press **Clear map** once (the old map has the
12 junk places from head turns).
