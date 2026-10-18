# Hackathon compliance (truthful)
Official portal: https://developer.craftifai.com/dashboard?tab=downloads — **not accessible during build (no internet); no PipeGen docs were read, so no CLI/API is claimed.**

| Requirement | Status | Evidence | Remaining work |
|---|---|---|---|
| A: Edge-vision pipeline via Requirement → Pipeline IR → Model → Compiler → Run | **NOT STARTED** | Proposed spec only (`PIPEGEN_REQUIREMENT_SPEC.md`) | Install PipeGen; follow documented workflow |
| A: Compile ≥1 model to TensorRT engine | **BLOCKED** | none | NVIDIA GPU/Jetson + PipeGen compiler |
| A: Run live on video/USB/CSI/RTSP on NVIDIA GPU/Jetson | **BLOCKED** (software path PARTIAL: runs on laptop CPU, not NVIDIA) | `app.py`, tests | NVIDIA hardware |
| A: Actionable outputs (JSON/MQTT/alerts/analytics) | **PARTIAL** | JSON/CSV incident export, alerts, analytics in app (not from PipeGen) | Emit from PipeGen run |
| A: CraftifAI tool integral to product | **NOT STARTED** | none | Rebuild pipeline in PipeGen |
| B: FirmGen firmware from frozen PRD | **NOT STARTED** | — | Out of scope for chosen project |
| B: Flash ESP32/STM32, physical behaviour, traceability report | **BLOCKED** | — | Hardware + FirmGen |
| Closed-loop bonus | **NOT STARTED** | — | Perception event → physical actuator |
| Standalone software prototype (own goal) | **COMPLETE** (tested as stated in TESTING.md) | repo | — |
| Synthetic dataset clearly labelled | **COMPLETE** | `data/synthetic_dataset.json` | — |
| Real-world precision/recall | **NOT STARTED** | none (no annotated real video) | annotate real clips |

No mandatory hardware requirement is marked COMPLETE. As it stands this project **does not qualify** for Track A or B.
