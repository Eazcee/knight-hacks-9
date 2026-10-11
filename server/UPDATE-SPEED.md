# Update: catch people walking by

## What changed (server)
- Uses the laptop's GPU automatically if it has one (NVIDIA via CUDA, or any Windows GPU via DirectML). Usually 3-10x faster than CPU.
- Walk-by enrollment: a passer-by who is blurry in every frame still gets saved if their face shows up in 2+ frames within 4 s (the sharpest look is kept). One blurry frame alone never creates a person.
- Faster bookkeeping: database commits no longer wait on the disk each time; QR codes are checked every 4th frame instead of every frame.
- Dashboard header shows how fast the server keeps up: "CPU 3.1 fps · 290 ms" (or GPU). Walk-bys need about 5 fps or more.

## Paste this to your terminal Claude
Download beacon-speed-update.zip first, then paste:

```
Speed-up for catching people walking by. Three parts.

PART A — server: unzip C:\Users\josep\Downloads\beacon-speed-update.zip over C:\Users\josep\AndroidStudioProjects\beacon (overwrite; keep the repo's existing .gitignore). Stop the server, run test_demo.py, test_training.py and test_range.py (all ALL PASSED), start the server again on 0.0.0.0:8000.

PART B — GPU: check what GPU this laptop has (Get-CimInstance Win32_VideoController | Select Name). Then in the server's venv:
 - NVIDIA GPU: pip uninstall -y onnxruntime, then pip install onnxruntime-gpu. If CUDA/cuDNN libraries are missing and it falls back to CPU, undo it and use the DirectML option instead.
 - Otherwise (Intel/AMD, or NVIDIA without CUDA working): pip uninstall -y onnxruntime, then pip install onnxruntime-directml.
 Restart the server. Its startup log prints "Beacon face engine on GPU (...)" or "on CPU". Tell me which, and if GPU fails, put plain onnxruntime back so it still runs on CPU.

PART C — app (BeaconApp): stop throttling uploads to a fixed ~400 ms. Instead: as soon as the previous /frame response comes back, upload the newest frame available (never queue old frames; one request in flight; at least 80 ms between uploads). JPEG quality 75. Log the round-trip time per upload every 20 frames. Build and install.

Then tell me what the dashboard header chip shows (device, fps, ms) while streaming with one person in view.
```

## What to expect
| Server speed | Frames of someone walking past (in view ~1.5 s) |
|---|---|
| 1 fps (slow CPU) | 1-2 → often missed |
| 3 fps | 4-5 → usually caught |
| 8+ fps (GPU) | 10+ → caught, sharpest frame kept |

Also helps: light. Indoors at night the glasses' camera uses a slow shutter, so moving faces smear. Brighter rooms = sharper faces.
