# Update: Street view, simplified

## What changes (v3: no face blur, everyone in the timeline)
- **No blur.** Every face is boxed: named people by name, everyone else as "Unknown" (name them later and every photo updates), faces too far to identify with a dashed box.
- The side list has **everyone** you saw, every time, unknown people included; anyone coming into view gets a photo right then (crowds: at most one every 3 s).

## From v2
- Street view opens **full screen** with **✕ Exit** (Esc too). The dashboard shows a strip of your latest photos; click one, ⤢ Open, a person's 📷, or the path on the map.
- Inside: **one continuous day**, **Previous / Next** (← → keys, hold to keep going), time bar. No walks, no arrows, no mini-map, no extra buttons.
- Right side: **everyone you saw, every time you saw them** (repeats included), with their face from that moment; click to jump there.
- Map: no indoors / floor buttons or text. **Today's path** is a thick white-outlined blue line with a green start dot; **🧭 Today's path** zooms to it; click the line to open that spot.
- The ⏸ Pause photos button is gone, and photos are un-paused whenever the server starts (they were paused in your screenshot).

## Paste this to your terminal Claude
Download beacon-street-v3.zip first, then paste:

```
On branch street-view: unzip C:\Users\josep\Downloads\beacon-street-v3.zip into the beacon server folder, overwriting everything. Restart the server and run test_street.py (ALL PASSED, ~2.5 min). Commit and push street-view.
```

Then Ctrl+F5 the dashboard.
