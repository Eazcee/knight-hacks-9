# Beacon

Meta Ray-Ban Display glasses that remember faces you pass and where you passed them, raise an
alert when a missing person you've already seen comes back into view, and route you back over a
map the camera built as you walked.

- [`server/`](server/) — Python FastAPI server: face recognition (InsightFace), sightings,
  alert watchlist, self-building place graph with A* routing, voice commands, web dashboard.
  Setup and API: [`server/README.md`](server/README.md).
- [`android/`](android/) — Beacon phone app (Kotlin, Meta Wearables Device Access Toolkit 1.0.1):
  streams the glasses camera to the server and renders the server's card on the glasses display.
  Open the folder in Android Studio or run `gradlew.bat installDebug`.

During development the phone talks to the laptop over USB: run `adb reverse tcp:8000 tcp:8000`
after every replug. Over a hotspot, set the server URL in the app instead.
