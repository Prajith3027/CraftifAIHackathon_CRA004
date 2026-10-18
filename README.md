# ContextGuard AI
*From object detection to explainable situational intelligence.*

A laptop **software prototype** for context-aware zone-intrusion monitoring: person detection → tracking →
foot-point-in-polygon test → a **temporally validated, explainable hazard-event engine** → incident log,
analytics, CSV/JSON export.

> **Honest status.** This is a standalone Python application. It is **not** a CraftifAI PipeGen/FirmGen
> deliverable, has no TensorRT engine, and has run on no edge hardware. See *CraftifAI integration status*.
> Alerts are **rule-triggered**, not confirmed safety incidents; nothing here is safety-certified.

## Problem
Detecting a person does not mean the person is in danger. A useful system must consider *where* they are,
*for how long*, and *how certain* the detection is — and say why it alerted.

## Solution & features
- Webcam, uploaded video, or a clearly-labelled **SIMULATED** synthetic scenario
- Person detection (OpenCV HOG built-in; optional YOLO via `ultralytics`), boxes + confidence
- Editable polygon zone (normalised coords); foot-point (bottom-centre of box) membership
- **Engine**: confidence threshold, persistence, hysteresis/grace, dedup-by-merge; every alert *and* suppression has a reason
- Baseline (alert immediately) vs proposed run **side-by-side on identical observations**
- SQLite incident history, CSV/JSON export, evaluation module (precision/recall only with ground truth)
- 7-page dark dashboard incl. System Status that separates ACTUAL / SIMULATED / NOT COMPLETED

## Install & run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Quick offline demo: **Live Monitoring → Synthetic demo (SIMULATED) → Start monitoring**.

## Tests
```bash
python -m pytest -q          # or, without pytest:  python -m unittest discover -s tests -v
```
Regenerate synthetic data + evaluation: `python scripts_generate_synthetic.py`

## Models / downloads
HOG detector needs **no download**. YOLO is optional: `pip install ultralytics`, then place weights at
`models/yolov8n.pt` (or let Ultralytics download on first use — needs internet). Detector `auto` tries YOLO then falls back to HOG.

## Camera permissions
Allow camera access for your terminal/Python (macOS: Settings → Privacy → Camera; Windows: Privacy → Camera).
Use **Test camera** on System Status. Close other apps using the camera.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Camera 0 unavailable" | permissions / other app / try index 1 |
| "Cannot decode video" | re-encode to H.264 mp4 |
| HOG misses people | use YOLO; larger people in frame; lower confidence threshold |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Streamlit port busy | `streamlit run app.py --server.port 8502` |

## Limitations
- HOG is weak (misses, false positives); its "confidence" is an **uncalibrated pseudo-score**, not a probability.
- Foot-point from bbox bottom is approximate (occlusion, camera angle, no homography).
- Simple IoU tracker; ID switches possible. Zone-level dedup can merge two different people arriving back-to-back within the cooldown.
- Persistence adds alert delay (≈ the persistence threshold) — a safety trade-off.
- Synthetic evaluation validates logic only. **No real-world precision/recall has been measured.**
- Dashboard was verified with a stubbed Streamlit harness; see `docs/TESTING.md`.

## CraftifAI integration status — **NOT COMPLETED**
No PipeGen run, Pipeline IR, TensorRT engine, speed report, or edge deployment exists. A proposed requirement
spec and event schema are in `docs/PIPEGEN_REQUIREMENT_SPEC.md` and `src/pipegen_adapter.py` (unvalidated).
**Remaining:** NVIDIA GPU/Jetson; install PipeGen from the portal; follow its documented workflow
(Requirement → Pipeline IR → Model → Compiler → Run); capture real speed report + JSON/MQTT output into
`evidence/pipegen/`. Details: `docs/HACKATHON_COMPLIANCE.md`.

## Future work
YOLO + ByteTrack, camera-homography ground-plane zones, multi-zone, PipeGen/TensorRT port, MQTT output, closed-loop actuator (bonus track), real annotated dataset.
