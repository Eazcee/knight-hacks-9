# Beacon — faces you've seen, pinned on a map that built itself

**One line:** the glasses remember every face you pass and where you were when you passed it; when an alert comes in, they tell you if you've already seen that person, and walk you back there over the paths you actually walked.

Two engines, one record:
- **Faces** (InsightFace/ArcFace): detect → embed → match against everyone seen so far. Strangers get an anonymous id automatically, so the *next* sighting recognizes them. Add names and notes by voice. Load an alert photo → "seen before: Coffee Station, 12 min ago".
- **Map** (GPS + self-building walk graph + A*): the phone's GPS comes with every frame. Everyone you see gets a **pin** with their clearest photo at the spot you saw them: **red** = seen, **green** = you talked (in view for 30 s, setting). A place is created every ~8 m you walk; edges carry **cost = distance walked × (1 + crowd)** and can be **blocked** (tap, voice "blocked", QR sign). Routes are A* over that; off the walked map it's **compass mode** (bearing + distance). The glasses show turn-by-turn, and the route ends the moment the camera recognizes the person you're walking to.

The bridge: every sighting is stored with its map position. "Where did I see her" and "take me there" are the same row in the database.

### How people are grouped ("training sets")
- Every person is a category with up to **16 training photos** (face fingerprints + the photo each came from). New faces are compared against all of them; best person above `BEACON_MATCH` wins.
- **Quality score** (0–100%) on every face: size, sharpness, how front-facing, detection confidence. Only good faces create new people or get added to a training set; near-duplicates are skipped so the set covers *different* angles.
- **Train mode** (dashboard "Train", voice `train` / `train dana`): for ~20 s the largest face in view is added to that person whenever it's a new angle. Have them slowly turn their head.
- **You fix the mistakes:** click a person on the dashboard → see every training photo and sighting (click for the full camera frame with the face boxed) → remove bad photos, "+ Train" good sightings, move a wrong photo to another/new person, or merge two people. The dashboard suggests merges when two people look alike.

### Range (how far away it recognizes people)
Range is set by **how many pixels a face covers**, not by the camera itself: a face twice as far away is half as many pixels. Three tiers (shorter side of the face box):
- **≥ 56 px** (`BEACON_ENROLL_PX`): can become a new person (still needs a sharp, fairly frontal photo).
- **≥ 40 px** (`BEACON_NEAR_PX`): full recognition; learns new angles.
- **≥ 18 px** (`BEACON_MIN_FACE`): **far** — only matched to people you already know, and only if one person clearly wins (`BEACON_FAR_MATCH`, margin 0.10) and it repeats within 6 s. Alert targets are never confirmed from far away: a far face that matches one (even clearly) shows **"Possible: … get closer"** on the glasses; the real alert fires only at ≥ 40 px or from a Scan (`BEACON_FAR_ALERTS=1` to allow far alerts).

To get more pixels: stream **HIGH (720×1280)** and upload frames **without shrinking them**; for long range, **Scan**: the app takes one 12 MP photo and posts it to `/scan`, which searches it in full-resolution tiles (several times the video range). The dashboard live view labels every face with its pixel size, grey boxes are faces seen but too small/unclear to name, and the header shows the frame size the server is getting.

---

### Map & navigation (how it works)
- **Position** = the phone's GPS (`lat`, `lng`, `acc` on every `/frame`). Fixes better than 15 m build the map; worse fixes (indoors) don't, and if the app also sends `steps` + `heading` the server dead-reckons (0.72 m/step along the heading) until GPS is good again. Outdoors expect 3–5 m; indoors 10–30 m (pins land on the right building, not the right room — demo the navigation outside).
- **Heading** = `heading` from the app if it sends one (glasses IMU / phone compass), else the direction you're walking from GPS. Turn-by-turn needs a heading; standing still with GPS-only heading the card says "HEAD NE · 40 m · walk so I get your heading".
- **Pins**: a person's pin = where they were last seen (projected a couple of meters ahead of you from their face size and position in the frame). Click a pin: note, Navigate, Open. Pins at the same spot fan out in a ring.
- **Talked** (green): someone in view (close range) for `convo_seconds` (default 30, gaps up to 8 s allowed). Toggle/threshold in the map bar or `POST /settings`; mark by hand on the person page or `POST /people/{id}/met`.
- **Routes**: `/route?to_person=` (their last sighting), `?to_name=` / `?to=` (a place), `?lat=&lng=` (map click). Glasses card: `→ Dana / LEFT · 24 m / 120 m left · 2 min · 3 turns`. Reaching a waypoint advances; seeing the person's face ends the route with **FOUND**. "blocked" (voice, QR, or tapping a path on the map) blocks the stretch ahead and reroutes; if there's no other walked path it goes direct.
- **Satellite map**: Esri World Imagery + labels by default (free, no key; the laptop needs internet for tiles; Leaflet is served locally). For Google's hybrid view, paste a **Maps JavaScript API key** into the "Google Maps key" box under the map (saved in `data/settings.json`); the dashboard reloads on Google Maps and falls back to Esri if the key is rejected.

## 1. Run it (5 minutes)

```bash
pip install insightface onnxruntime opencv-python fastapi "uvicorn[standard]" python-multipart numpy requests scikit-image httpx
uvicorn server:app --host 0.0.0.0 --port 8000
```
First start downloads the face models (~280 MB, from GitHub). Then open **http://localhost:8000** for the dashboard.

Test everything with no hardware:
```bash
python test_demo.py          # simulated walk + people + alert + reroute (all should PASS)
python test_training.py      # photos, quality, train mode, merge, remove, map clear (all should PASS)
python test_range.py         # far faces, "possible" alerts, Scan on a 12 MP photo (all should PASS)
python sim_walk.py --reset   # fake GPS walk: watch the map build, pins appear, a route draw and reroute
python webcam_client.py --show --lat 28.6024 --lng -81.2001   # laptop webcam → server, faces get pins
```
`sim_walk.py` is the rehearsal tool: run it against the live server and the dashboard shows exactly what the demo will look like, no walking required.

Speed: ~1.3 s/frame on a weak 2-core box; expect 150–400 ms on a laptop. Send 2–3 fps. For more speed set `BEACON_DET=416` (smaller detector, shorter range). A 12 MP Scan takes a few seconds.

## 2. API (what the phone app calls)

| Call | What it does |
|---|---|
| `POST /frame?lat=&lng=&acc=&heading=&steps=` body = JPEG (`Content-Type: image/jpeg`) | The whole pipeline. Returns `people`, `place` (position, quality, events), `alerts`, `nav` (the current turn, if a route is active) and a ready-to-render `display` card `{level, title, lines}` for the glasses. `lat/lng/acc` = phone GPS (meters); `heading` (0 = north, clockwise) and `steps` (cumulative pedometer) are optional. Without GPS nothing goes on the map. |
| `POST /scan` body = JPEG (any size, e.g. a 12 MP photo) | **Scan the room.** Tiled full-res search; every face matched (with `dir` left/ahead/right and `px`); returns `display` (held on the glasses ~8 s). Voice `scan` makes `/command` return `action: "scan"` so the app knows to take the photo. |
| `POST /people/{id}` JSON `{name, notes, append_note}` | Name / annotate someone. |
| `POST /enroll` multipart `image`, `name`, `notes` | Enroll from a photo (selfie). If the face is already known, the info attaches to that person. |
| `POST /watch` multipart `image`, `label` | **The alert.** Returns `seen_before`, the sightings (place + time), and a `route` to the last one. From now on, seeing them raises `alerts` and a red display card. |
| `GET /route?to_person=<id>` / `?to=<place_id>` / `?to_name=<place>` / `?lat=&lng=` | Plan a route (A* over walked paths with costs, or compass mode). Returns `mode`, `path`, `waypoints`, `remaining_m`, `next` (turn, dist, bearing), `target` (name, photo). `DELETE /route` cancels. |
| `GET /settings` · `POST /settings` `{convo_seconds, convo_enabled}` | The "talked" rule (green pins). |
| `POST /people/{id}/met` `{met: true\|false}` | Mark "we talked" by hand. |
| `GET /map` | Nodes (lat/lng), edges (dist, crowd, blocked), position, trail, active route, `pins` (people with photo + red/green). |
| `POST /nodes/{id}` `{name}` · `POST /edges/{a}/{b}` `{blocked:true}` | Label a place / block a hallway (reroutes if a route is active). |
| `GET /people/{id}/photos` | Training photos (with quality, source) + sightings (crop, full frame, face box). |
| `DELETE /train_photos/{eid}` · `POST /train_photos/{eid}/move` `{to: id\|"new"}` | Remove a bad training photo / move it to the right person. |
| `POST /sightings/{sid}/train` · `POST /sightings/{sid}/move` `{to}` | Add a sighting to training / reassign a wrong sighting. |
| `POST /people/{id}/merge` `{into}` · `GET /suggestions` | Merge two people / "might be the same person" pairs. |
| `POST /train/start` `{person_id \| name, seconds, goal}` · `POST /train/stop` · `GET /train` | Train mode. The glasses card shows progress. |
| `POST /people/{id}/flag?label=` | Mark a known person as an alert target (no photo needed). |
| `POST /places/mark` `{name}` · `POST /nodes/{id}` `{name}` · `POST /edges/{a}/{b}` `{blocked}` · `POST /map/clear` | Name the place you're at / rename a place / block a path / forget the map (people stay). |
| `GET /live.jpg` | Latest camera frame with labeled boxes (dashboard live view). |
| `POST /report` | "Report" the latest alert: freezes who/where/when for responders (voice: `report`). |
| `POST /command` `{text}` | Voice: `this is Dana`, `note judges the AI track`, `mark place coffee station` / `call this place …`, `take me to coffee station` / `take me to dana`, `find dana`, `where did I see dana`, `blocked`, `train` / `train dana`, `scan`, `report`, `cancel`. Returns `say` for the display. |
| `GET /state` · `GET /map` · `GET /people` · `GET /alerts` | Dashboard feeds. `POST /reset` wipes everything ("delete all"). |

Printed QR codes are optional: text `place:Coffee Station` names the place you're at; text `blocked` held up while following a route blocks the stretch ahead and reroutes (same as saying "blocked"). Make them with any QR generator, print big (≥ 15 cm).

Env flags: `BEACON_MODEL=buffalo_l` (`buffalo_s` = ~8x faster on CPU, same close-range accuracy, shorter far range; needs its own data folder, e.g. `BEACON_DATA=data_fast`), `BEACON_PROVIDERS=cpu` (force CPU), `BEACON_NODE_M=8` (place spacing), `BEACON_GOOD_ACC=15` (GPS accuracy trusted to build the map), `BEACON_DET=640` (detector size, long side; 416 = faster, shorter range), `BEACON_MIN_FACE=18` / `BEACON_NEAR_PX=40` / `BEACON_ENROLL_PX=56` / `BEACON_FAR_MATCH=0.40` (range tiers, above), `BEACON_SCAN_TILE=1600`, `BEACON_MATCH=0.45` (match threshold), `BEACON_AUTO_ENROLL=0` (only recognize enrolled people), `BEACON_RETENTION_H=24` (forget un-annotated strangers after N hours).

## 3. Glasses side (the part only you can test)

Setup first (do this before anything else): toolkit 1.0.1 needs Meta AI app ≥ V292 and glasses firmware ≥ V128 (reboot the glasses after the V128 update or camera capture breaks). Glasses battery > 10%. Dev Mode (Meta AI app → Settings → App Info → tap App Version 5×), then the install button to put the Device Access Toolkit on the glasses (wear them, keep the app open). In Dev Mode the App ID can be `0`/placeholder; only one third-party app can be registered at a time.

Start from Meta's samples in `facebook/meta-wearables-dat-ios` or `-android` (`samples/CameraAccess` + `samples/DisplayAccess`; the Connect session's **BirdSpotter** sample combines camera, motion, speech and display in one app if it's published). You need three additions:

**a) Frames → server.** In the camera stream's frame callback, every ~400 ms convert the frame to JPEG at ~640 px wide and POST it:

```swift
// iOS (Swift) — call from the stream frame callback, throttled to ~2–3 fps
func upload(_ image: UIImage, heading: Double?, steps: Int?) {
    guard let jpg = image.jpegData(compressionQuality: 0.7) else { return }
    var comps = URLComponents(string: "http://LAPTOP_IP:8000/frame")!
    comps.queryItems = [heading.map { URLQueryItem(name: "heading", value: "\($0)") },
                        steps.map { URLQueryItem(name: "steps", value: "\($0)") }].compactMap { $0 }
    var req = URLRequest(url: comps.url!); req.httpMethod = "POST"
    req.setValue("image/jpeg", forHTTPHeaderField: "Content-Type"); req.httpBody = jpg
    URLSession.shared.dataTask(with: req) { data, _, _ in
        guard let data, let r = try? JSONDecoder().decode(FrameResponse.self, from: data) else { return }
        DispatchQueue.main.async { self.render(r.display) }   // → glasses display
    }.resume()
}
```
```kotlin
// Android (Kotlin) — OkHttp
fun upload(bmp: Bitmap, heading: Double?, steps: Int?) {
    val out = ByteArrayOutputStream(); bmp.compress(Bitmap.CompressFormat.JPEG, 70, out)
    val url = "http://LAPTOP_IP:8000/frame".toHttpUrl().newBuilder().apply {
        heading?.let { addQueryParameter("heading", it.toString()) }
        steps?.let { addQueryParameter("steps", it.toString()) } }.build()
    val req = Request.Builder().url(url).post(out.toByteArray().toRequestBody("image/jpeg".toMediaType())).build()
    client.newCall(req).enqueue(object : Callback {
        override fun onResponse(call: Call, resp: Response) { render(parseDisplay(resp.body!!.string())) }
        override fun onFailure(call: Call, e: IOException) {} })
}
```
Phone and laptop on the same Wi-Fi (hackathon Wi-Fi often blocks device-to-device: use a phone hotspot).

**b) Display card.** Render `display.title` as a heading `Text` and `display.lines` as body `Text` in a `FlexBox` column; if `level == "alert"` prefix a warning `Icon`. Each `send()`/`sendContent()` replaces the whole screen, so send the full card every time it changes (not every frame). The display is 600×600; keep it to a title + 3 lines.

**c) GPS (required for the map) and sensors.**
- Position: the phone's fused location at 1 Hz, high accuracy → `lat`, `lng`, `acc` on every `/frame`.
- Heading (optional): glasses Motion capability yaw if it's referenced to north, else the phone's rotation-vector azimuth, else send nothing and the server uses the direction you walk.
- Steps (optional, helps indoors): Android `TYPE_STEP_COUNTER` → cumulative count → `steps`.
- Voice: the Speech capability's final transcriptions → `POST /command` → show `say` on the display.

Navigation needs no new glasses UI: the turn-by-turn is just another `display` card.

## 4. Plan

1. Phone app sends `lat/lng/acc` (+ `steps`, `heading` if easy). Walk a loop outside the venue: watch the map build on the dashboard, pins appear where you met people.
2. Rehearse with `sim_walk.py` until the story is tight. Print a `blocked` QR sign (or just say "blocked").
3. Record a backup video of the whole outdoor run in case Wi-Fi or GPS dies on stage.
4. Slides with numbers: faces enrolled, recognition hit rate and range (px → feet), places built per 100 m, reroute time after a block.

## 5. The 4-minute demo

1. Wearer walks outside for 60 s. On the big screen the satellite map draws the path: places appear every few meters, edges show walking costs.
2. Meets the judges. "This is Dana." "Note: judges the AI track." Talks for 30 s: Dana's pin turns green. Looks away, looks back: Dana's card pops on the glasses.
3. One judge wanders off. Five minutes later the "alert" photo is loaded (`/watch`). Screen: **seen before · Coffee Station · 5 min ago**, route drawn over the walked paths. Wearer follows the turns on the glasses; a teammate holds the `blocked` sign in the aisle (or wearer says "blocked"); the route flips to the other way around; wearer reaches the judge; glasses say **FOUND** and flash the alert; "report" sends the GPS location (a maps link).
4. "Delete all." Everything's gone.

Say on stage: strangers are kept as anonymous ids only, un-annotated ones expire, alerts match only loaded photos, and there's a one-tap wipe. That's the difference between this and the Harvard project.

## 6. Files
- `server.py` — API · `dashboard.html` — the big-screen dashboard (Leaflet in `static/`) · `faces.py` — recognition + training sets · `geo.py` — GPS, walk graph, A*, turn-by-turn · `db.py` — SQLite
- `sim_walk.py` — fake GPS walk for rehearsals · `webcam_client.py` — frames from webcam/video · `test_demo.py`, `test_training.py`, `test_range.py` — all should say `ALL PASSED`
- `data/` — database, face crops, frames, settings (created on first run)
