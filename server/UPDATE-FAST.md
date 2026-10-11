# Update: fast mode

On a laptop CPU the full face model takes too long per frame, so people walking by get 1-2 frames or none.
`buffalo_s` is a smaller model: about 8x faster on CPU (1.6 s -> 0.2 s per frame on my test machine).
Close-range recognition is the same; far-away faces (under ~40 px) are recognized less often.
A GPU, if the laptop has one, gets the speed without the trade-off.

## Paste this to your terminal Claude
Download beacon-fast-update.zip first, then paste:

```
Speed fix, server only.
1. Stop the server. Unzip C:\Users\josep\Downloads\beacon-fast-update.zip over C:\Users\josep\AndroidStudioProjects\beacon (overwrite; keep the repo's .gitignore).
2. Start the server once as before and open http://localhost:8000 while streaming. Read the header chip that shows device · model, fps and ms. Tell me the numbers.
3. If it says CPU and fps is under 4: stop the server and start it in fast mode with these environment variables:
   BEACON_MODEL=buffalo_s
   BEACON_DATA=data_fast
   (PowerShell: $env:BEACON_MODEL="buffalo_s"; $env:BEACON_DATA="data_fast"; then the usual uvicorn command.)
   The first start downloads the small model (~125 MB). People saved with the big model live in "data" and stay there; fast mode starts an empty list in "data_fast", so re-meet/re-name people.
4. Tell me the new fps and ms from the header chip.
Also check the app side: log the /frame round-trip time and how many frames per second the glasses stream actually delivers to the app. If the glasses deliver fewer than ~5 fps at HIGH resolution, try MEDIUM to see if the stream itself is the bottleneck.
```
