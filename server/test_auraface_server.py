"""Real-model smoke test for AuraFace and the Beacon FastAPI pipeline.

This intentionally does not mock InsightFace, ONNX Runtime, or the HTTP layer. On
a fresh machine it may download fal/AuraFace-v1 before running.

    python test_auraface_server.py
"""
import os
import sys
import time
from pathlib import Path

import cv2

# Set these before importing server: it constructs the DB and FaceEngine at import time.
test_database_url = os.environ.get("BEACON_TEST_DATABASE_URL")
if not test_database_url:
    raise SystemExit("Set BEACON_TEST_DATABASE_URL to a disposable Neon branch; never use production")
os.environ["DATABASE_URL"] = test_database_url
os.environ["BEACON_MODEL"] = "auraface"
os.environ["BEACON_DATA"] = "data_test_auraface_server"

from fastapi.testclient import TestClient  # noqa: E402
import insightface.data as insightface_data  # noqa: E402

import faces as faces_module  # noqa: E402
import server  # noqa: E402


failures = 0


def check(condition, message):
    global failures
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures += 1


def jpeg(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise AssertionError("OpenCV could not encode the integration-test frame")
    return encoded.tobytes()


started = time.time()
client = TestClient(server.app)
reset = client.post("/reset")
check(reset.status_code == 200, "Beacon test data resets through the API")

model_dir = Path(faces_module.MODEL_ROOT) / "models" / "auraface"
required_models = ("scrfd_10g_bnkps.onnx", "glintr100.onnx")
check(faces_module.MODEL == "auraface", "server selected the AuraFace model")
check(
    faces_module.HF_MODEL_REPOS.get("auraface") == "fal/AuraFace-v1",
    "AuraFace resolves to the expected Hugging Face repository",
)
check(
    all((model_dir / filename).is_file() for filename in required_models),
    "AuraFace detector and recognizer ONNX files are installed",
)
check(
    server.faces.device.endswith("auraface"),
    f"FaceEngine initialized AuraFace ({server.faces.device})",
)

group = insightface_data.get_image("t1")
detected = server.faces.detect(group, min_px=20)
check(len(detected) >= 4, f"AuraFace detects faces directly ({len(detected)} found)")
check(
    bool(detected) and all(face["emb"].shape == (512,) for face in detected),
    "AuraFace produces a 512-value recognition embedding for every detected face",
)

body = jpeg(group)
headers = {"Content-Type": "image/jpeg"}
first_response = client.post("/frame", content=body, headers=headers, params={"ts": 1_000_000.0})
check(first_response.status_code == 200, "FastAPI /frame accepts a JPEG")
first = first_response.json() if first_response.status_code == 200 else {}
first_people = first.get("people", [])
check(len(first_people) >= 4, f"/frame processes AuraFace results ({len(first_people)} people)")

second_response = client.post("/frame", content=body, headers=headers, params={"ts": 1_000_001.0})
check(second_response.status_code == 200, "FastAPI /frame processes a second JPEG")
second = second_response.json() if second_response.status_code == 200 else {}
first_ids = {person["person_id"] for person in first_people}
second_ids = {person["person_id"] for person in second.get("people", [])}
check(
    bool(first_ids) and first_ids.issubset(second_ids),
    "the server recognizes the same AuraFace identities on the next frame",
)

state_response = client.get("/state")
check(state_response.status_code == 200, "FastAPI /state remains healthy after inference")
state = state_response.json() if state_response.status_code == 200 else {}
timing = state.get("timing", {})
check(state.get("frames") == 2, "server recorded both processed frames")
check(timing.get("frame_ms", 0) > 0, "server reports real end-to-end inference timing")
check(timing.get("device", "").endswith("auraface"), "server state reports AuraFace as the active model")
check(len(state.get("people", [])) >= 4, "AuraFace identities are persisted by the server")

elapsed = time.time() - started
print(f"\n{'ALL PASSED' if not failures else str(failures) + ' FAILED'} in {elapsed:.1f}s")
sys.exit(1 if failures else 0)
