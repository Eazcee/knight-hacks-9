# Beacon — architecture, features, frontend, build plan

**Pitch:** Smart glasses that remember every face you pass and *where* you passed it, on a map the camera built as you walked. When an alert comes in (AMBER / Silver / missing person), the glasses tell you if you've already seen that person, where and when, and route you back there over that map, re-routing around anything the camera sees blocking the way.

**Why the two systems belong together:** a sighting is a face *plus* a place. Recognition without the map answers "who"; the map without recognition is Google Maps. Together: "you passed him 12 minutes ago at the coffee station, here's the way, the east aisle is blocked, go around."

---

## 1. Architecture

```
┌──────────────────────── Meta Ray-Ban Display ────────────────────────┐
│ camera stream · IMU (orientation) · mics (on-glasses ASR) · 600×600  │
│ display · Neural Band gestures · speaker                              │
└──────────────┬──────────────────────────────▲────────────────────────┘
        Bluetooth (Device Access Toolkit 1.0.1)│ display cards (FlexBox/Text/Icon)
┌──────────────▼──────────────────────────────┴────────────────────────┐
│ PHONE COMPANION APP (Swift or Kotlin)                                 │
│ • DAT session: camera stream, motion, speech, display, inputs         │
│ • throttles frames to 2–3 fps, JPEG 640px → POST /frame               │
│ • heading (IMU yaw) + steps (pedometer) as query params               │
│ • final speech transcripts → POST /command                            │
│ • renders `display` card from server; Neural Band: swipe/tap actions  │
│ • small phone UI: connect, live mirror, alerts, people, settings      │
└──────────────┬──────────────────────────────▲────────────────────────┘
        Wi-Fi / phone hotspot (HTTP JSON)       │
┌──────────────▼──────────────────────────────┴────────────────────────┐
│ LAPTOP SERVER (Python, FastAPI) — exists, tested                      │
│ faces.py   detect → ArcFace embed → match → auto-enroll strangers     │
│ places.py  keyframes → place graph → loop closure → A* with costs     │
│ server.py  /frame pipeline, people+notes, sightings, /watch alerts,   │
│            /route, /command, /report, /state, dashboard               │
│ db.py      SQLite: people, embeddings, sightings, nodes, edges        │
└──────────────┬───────────────────────────────────────────────────────┘
               │ /state (1 Hz) · /map · /people · /alerts
┌──────────────▼───────────────────────────────────────────────────────┐
│ WEB DASHBOARD (big screen for judges)                                 │
│ 2D map now → 3D map (Three.js) · glasses mirror · people · timeline   │
│ alert composer · block hallway · replay · stats · delete all          │
└──────────────────────────────────────────────────────────────────────┘
```

Nothing runs *on* the glasses. Every capability (camera, display, mic, IMU, Neural Band) is reached from the phone app through the Device Access Toolkit over Bluetooth. The laptop does all the AI. The phone is a relay plus the glasses UI.

### Latency budget (glass-to-glass)
| hop | time |
|---|---|
| camera frame glasses → phone (BT stream) | ~100–200 ms |
| JPEG encode + Wi-Fi POST | ~50–100 ms |
| server: faces (~150–400 ms on a laptop) + places (~30 ms) | ~200–450 ms |
| display card phone → glasses (BT) | ~100–300 ms |
| **total** | **~0.5–1.0 s** → send 2–3 fps, drop frames while busy (server already does this) |

### Data model (SQLite)
- `people` (id, name, notes, is_watch, watch_label, crop, last_seen) + `embeddings` (several 512-d vectors per person, grows as new angles are seen)
- `sightings` (person, ts, node, x, y, sim, crop) ← **the bridge table**
- `nodes` (places: name, x, y, keyframe, qr, visits) + `edges` (a, b, dist walked, crowd, blocked)

---

## 2. Features

### A. On the glasses (display cards, 600×600, title + ≤3 lines)
| card | when | content |
|---|---|---|
| Idle | nothing to say | place name · "3 people" · scanning dot |
| Person | known face confirmed | **Dana** · judges the AI track · last met: Coffee Station |
| Stranger | new face auto-enrolled | "Unknown · say *this is …*" |
| Route | route active | ↰ **LEFT** → Coffee Station · 12 m · 2 stops left (DAT `Icon` for the arrow) |
| Arrived | reached goal | "Arrived · Coffee Station" |
| Alert loaded | after `/watch` | "Seen before · Coffee Station · 12 min ago · say *find* to route" |
| **ALERT** (red) | watchlist face in view | **AMBER Alert · Leo · 7** · match 91% · "tap to report" |
| Toast | after a voice command | "Noted" / "Routing, 40 m" / "Reported" |

Rules: send a card only when it changes (not every frame). Alert beats route beats people beats place. Keep the L0 root view = Idle so Back never kills the session by accident.

### B. Voice (on-glasses ASR → `/command`)
`this is <name>` · `note <text>` · `call this place <name>` · `take me to <place>` · `find <name>` · `where did I see <name>` · `report` · `cancel` · `pause` / `resume` (stop scanning — privacy pause; implement as a phone-side flag that stops uploads). Launch with **"Hey Meta, open Beacon"** (DAT voice invocation).

### C. Neural Band / touchpad
swipe left/right = cycle person cards when several people are in view · double-tap = confirm/report · long-pinch = pause scanning · swipe down = dismiss card.

### D. Phone app screens
1. **Connect** — glasses status, server URL (laptop IP:8000), start/stop session, fps slider.
2. **Live** — mirror of the current glasses card, small camera thumbnail, latency/fps/dropped counters.
3. **Alerts** — pick a photo from the camera roll (or camera), label it ("AMBER Alert · Leo · 7"), send to `/watch`; shows *seen before / where / when*, a **Route** button and a big **Report** button. This screen is the demo's emotional beat; make it good.
4. **People** — list with crops, names, notes; edit; delete; "go" (route to last sighting).
5. **Map** — a web view of the dashboard (don't rebuild it natively).
6. **Settings** — auto-enroll strangers on/off, retention hours, detector size, delete all.

### E. Dashboard (big screen)
- **Map**: 2D now; **3D viewer** (Three.js): keyframe thumbnails floating at each place, edges as lines labeled with cost, blocked edges red, route glowing, wearer marker with heading, sighting pins, click a place to label/route/block. Optional venue mesh underneath (see §3).
- **Glasses mirror** (what the wearer sees), **people** panel, **timeline**, **alerts** with report button, **alert composer** (upload photo → `/watch`), **close hallway** (block edge), **replay** slider (re-play sightings over time), **stats** (frames, dropped, latency, people, places), **delete all**.

### F. Server (exists) + next
Have: everything in README §2. Next: `/pause` flag · `/replay?from&to` · sightings heatmap endpoint · multi-camera (a phone propped at a doorway posts to `/frame?source=door1` and becomes a fixed node) · export CSV.

### G. Stretch (only if everything above works)
- Body/clothing re-ID for long-range matches (InsightFace 2.1 `PersonAnalysis`: face + body ReID against registered people).
- Predict where the alerted person is *now*: simulate them walking the graph from the last sighting at 1.2 m/s, show a spreading probability over places, route to the best intercept (this is the "simulation" judges will love).
- Second wearer or doorway phone camera merging sightings.

---

## 3. The 3D model question (honest answer)

Live 3D reconstruction from the glasses (Tesla-style occupancy from a monocular ~2 fps stream over Bluetooth, no depth, no calibration) is not a weekend job. What you *can* have, in order of effort:

1. **Have now — topological map:** places as nodes, walked distance as edge cost, loop closure by visual place recognition, QR anchors. This is what routing runs on. Keep it as the source of truth no matter what.
2. **Today (2–3 h) — 3D viewer:** render that graph in Three.js: nodes at (x, y, 0), keyframe thumbnails as billboards, route and sightings in 3D, orbit camera, fog-of-war "explored" disc around visited nodes. It *looks* like a world model and it's honest.
3. **Tonight (if someone has a LiDAR iPhone or 30 min of laptop time) — real venue mesh:** scan the demo area with Polycam / Scaniverse (LiDAR) or run COLMAP/OpenSfM on the glasses video → GLB mesh or point cloud → load under the graph in the Three.js scene. Align by clicking the 3 QR-anchor spots in the mesh (similarity transform from graph XY to mesh coords). Now the place graph, sightings and route sit inside a real 3D model of the venue. Pathfinding still runs on the graph.
4. **Not this weekend:** live monocular SLAM / occupancy from the glasses stream.

Pitch line: "The map builds itself from what the camera sees; the 3D model is the venue scan we align it to."

---

## 4. Build plan with the glasses in hand (now → Sunday)

**Gate (first 60–90 min, one person, do nothing else):** Meta AI app ≥ V292, glasses firmware ≥ V128 (reboot after the update), battery > 10%, Dev Mode (Settings → App Info → tap App Version 5×), install the toolkit on the glasses. Run Meta's **CameraAccess** sample → frames on the phone. Run **DisplayAccess** sample → text on the glasses. Until both work, nothing else matters.

**Then, in parallel:**
- **Phone (owner A):** merge the two samples; throttle + upload frames; render `display`; then speech → `/command`; then motion yaw → `heading`, pedometer → `steps`; then Neural Band actions; then the Alerts screen.
- **Server/map (owner B):** record a 2-min walk video on the glasses, replay with `webcam_client.py --video`, tune `NODE_SPACING_M`, `MIN_INLIERS`; print 5 `place:` QR codes + 2 `blocked` signs; add `/pause`, replay endpoint.
- **Dashboard (owner C):** 3D viewer; alert composer; glasses mirror; stats; replay slider.
- **Pitch (owner D, or B after tuning):** numbers (recognition hit rate on N people at 1–2 m, place-recognition rate on revisits, reroute time), slides, backup video of a full run, QR anchors taped.

**Sunday morning:** rehearse 3×; a second full backup video; charge everything.

---

## 5. Demo script (4 min)
1. Wearer walks the hall 60 s; the 3D map draws itself on screen.
2. Meets judges: "this is Dana" · "note judges the AI track". Looks away, back: Dana's card on the glasses.
3. A judge wanders off. Minutes later, the alert photo is loaded on the phone: **seen before · Coffee Station · 5 min ago**, route on the big screen. Wearer follows the arrow; a teammate holds the `blocked` sign in the aisle; route flips; wearer reaches the judge; glasses flash red; "report" → location frozen on screen.
4. "Delete all."

## 6. Risks → fallbacks
- Venue Wi-Fi blocks phone↔laptop → phone hotspot (test it first).
- BT stream drops → phone app auto-reconnects; server is stateless per frame.
- Place recognition misses under stage lights → QR anchors at every demo spot.
- Faces too small from a distance → walk closer; min face 40 px; stretch: body ReID.
- Everything dies → backup video + live dashboard replay from the recorded DB.

## 7. Guardrails (say them on stage)
Strangers are anonymous ids only; un-annotated ones expire (`BEACON_RETENTION_H`); alerts match only photos you loaded; scanning can be paused; one tap wipes everything. No web lookups, ever.
