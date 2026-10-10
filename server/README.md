# Beacon — faces you've seen, on a map the camera built

**One line:** the glasses remember every face you pass and where you were when you passed it; when an alert comes in, they tell you if you've already seen that person, and route you back there over a map that built itself as you walked.

Two engines, one record:
- **Faces** (InsightFace/ArcFace): detect → embed → match against everyone seen so far. Strangers get an anonymous id automatically, so the *next* sighting recognizes them. Add names and notes by voice. Load an alert photo → "seen before: Coffee Station, 12 min ago".
- **Places** (self-building map + A*): every ~6 m of walking the camera snapshots a keyframe = a place. Places link into a graph with edge cost = distance actually walked × crowd factor. Seeing a known keyframe again = loop closure (position snaps). Routes are A* over that graph; a blocked edge reroutes live.

The bridge: every sighting is stored with the place it happened at. "Where did I see her" and "take me there" are the same row in the database.

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
python webcam_client.py --show         # your laptop webcam → server, boxes on screen
python webcam_client.py --video walk.mp4 --fps 3   # replay a video recorded on the Ray-Bans
```
**That last one is your secret weapon:** record a normal video with the glasses while walking the venue, AirDrop it, replay it through the server. The map builds itself and you can tune everything *before* the live SDK path works.

Speed: ~1.3 s/frame on a weak 2-core box; expect 150–400 ms on a laptop. Send 2–3 fps. For more speed set `BEACON_DET=416` (smaller detector, shorter range). A 12 MP Scan takes a few seconds.

## 2. API (what the phone app calls)

| Call | What it does |
|---|---|
| `POST /frame?heading=<deg>&steps=<n>` body = JPEG (`Content-Type: image/jpeg`) | The whole pipeline. Returns `people`, `place`, `alerts`, and a ready-to-render `display` card `{level, title, lines}` for the glasses. `heading` (0 = north, clockwise) and `steps` (cumulative pedometer count) are optional. Without `steps` the map only grows from QR anchors and "mark place" (set `BEACON_TIME_DR=1` to estimate distance from time, e.g. for video replays). |
| `POST /scan` body = JPEG (any size, e.g. a 12 MP photo) | **Scan the room.** Tiled full-res search; every face matched (with `dir` left/ahead/right and `px`); returns `display` (held on the glasses ~8 s). Voice `scan` makes `/command` return `action: "scan"` so the app knows to take the photo. |
| `POST /people/{id}` JSON `{name, notes, append_note}` | Name / annotate someone. |
| `POST /enroll` multipart `image`, `name`, `notes` | Enroll from a photo (selfie). If the face is already known, the info attaches to that person. |
| `POST /watch` multipart `image`, `label` | **The alert.** Returns `seen_before`, the sightings (place + time), and a `route` to the last one. From now on, seeing them raises `alerts` and a red display card. |
| `GET /route?to=<place_id>` / `?to_person=<id>` / `?to_name=<place>` | Plan a route (A*, costs). `DELETE /route` cancels. |
| `POST /nodes/{id}` `{name}` · `POST /edges/{a}/{b}` `{blocked:true}` | Label a place / block a hallway (reroutes if a route is active). |
| `GET /people/{id}/photos` | Training photos (with quality, source) + sightings (crop, full frame, face box). |
| `DELETE /train_photos/{eid}` · `POST /train_photos/{eid}/move` `{to: id\|"new"}` | Remove a bad training photo / move it to the right person. |
| `POST /sightings/{sid}/train` · `POST /sightings/{sid}/move` `{to}` | Add a sighting to training / reassign a wrong sighting. |
| `POST /people/{id}/merge` `{into}` · `GET /suggestions` | Merge two people / "might be the same person" pairs. |
| `POST /train/start` `{person_id \| name, seconds, goal}` · `POST /train/stop` · `GET /train` | Train mode. The glasses card shows progress. |
| `POST /people/{id}/flag?label=` | Mark a known person as an alert target (no photo needed). |
| `POST /places/mark` `{name}` · `POST /map/clear` | Create a place here / forget all places (people stay). |
| `GET /live.jpg` | Latest camera frame with labeled boxes (dashboard live view). |
| `POST /report` | "Report" the latest alert: freezes who/where/when for responders (voice: `report`). |
| `POST /command` `{text}` | Voice: `this is Dana`, `note judges the AI track`, `call this place coffee station`, `take me to coffee station`, `find dana`, `where did I see dana`, `train` / `train dana`, `mark place food court`, `report`, `cancel`. Returns `say` for the display. |
| `GET /state` · `GET /map` · `GET /people` · `GET /alerts` | Dashboard feeds. `POST /reset` wipes everything ("delete all"). |

Printed QR codes are optional anchors: text `place:Coffee Station` = a guaranteed named place; text `blocked` held up while following a route = blocks the edge you're about to walk, and the route reroutes. Make them with any QR generator, print big (≥ 15 cm).

Env flags: `BEACON_DET=640` (detector size, long side; 416 = faster, shorter range), `BEACON_MIN_FACE=18` / `BEACON_NEAR_PX=40` / `BEACON_ENROLL_PX=56` / `BEACON_FAR_MATCH=0.40` (range tiers, above), `BEACON_SCAN_TILE=1600`, `BEACON_MATCH=0.45` (match threshold), `BEACON_AUTO_ENROLL=0` (only recognize enrolled people), `BEACON_RETENTION_H=24` (forget un-annotated strangers after N hours).

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

**c) Sensors (optional but makes the map real).**
- Heading: Motion capability at 10 Hz → `orientation` quaternion → yaw in degrees → `heading`. It can be null on some devices; fall back to the phone's compass.
- Steps: iOS `CMPedometer` / Android `TYPE_STEP_COUNTER` → cumulative count → `steps`.
- Voice: the Speech capability's final transcriptions → `POST /command` → show `say` on the display.

Without b/c you still have a working demo: the server estimates distance from time and the dashboard shows everything.

## 4. Plan for the rest of the weekend

**Now → 3 pm:** server running on the laptop, webcam client, name two teammates, see the cards. Record a 2-minute video walking the venue with the glasses, replay it, watch the map build. Tune `NODE_SPACING_M` / `MIN_INLIERS` in `places.py` until revisits are recognized reliably (tape 4–5 `place:` QR codes at key spots as insurance).

**3 → 8 pm (split):** (1) phone app: camera stream → `/frame` → display card. (2) dashboard polish + alert flow + "report" button. (3) print QR anchors + `blocked` sign, write the demo script, collect numbers.

**8 pm → midnight:** heading + steps into `/frame`; voice commands; full alert run-through; blocked-sign reroute; record a backup video of the whole demo in case Wi-Fi dies on stage.

**Sunday morning:** rehearse 3×, slides with numbers: faces enrolled, recognition hit rate across N people at 1–2 m, place-recognition rate on revisits, reroute time after a block.

## 5. The 4-minute demo

1. Wearer walks the hall for 60 s. On the big screen the map draws itself: places appear, edges show walking costs.
2. Meets the judges. "This is Dana." "Note: judges the AI track." Looks away, looks back: Dana's card pops on the glasses.
3. One judge wanders off. Five minutes later the "alert" photo is loaded (`/watch`). Screen: **seen before · Coffee Station · 5 min ago**, route drawn. Wearer follows the arrows; a teammate holds the `blocked` sign in the aisle; the route flips; wearer reaches the judge; glasses flash the alert; "report" sends the location.
4. "Delete all." Everything's gone.

Say on stage: strangers are kept as anonymous ids only, un-annotated ones expire, alerts match only loaded photos, and there's a one-tap wipe. That's the difference between this and the Harvard project.

## 6. Files
- `server.py` — API · `dashboard.html` — the big-screen dashboard · `faces.py` — recognition + training sets · `places.py` — map, loop closure, A* · `db.py` — SQLite
- `webcam_client.py` — frames from webcam/video · `test_demo.py` — end-to-end test (`ALL PASSED` in ~7 s)
- `data/` — database, face crops, keyframes (created on first run)
