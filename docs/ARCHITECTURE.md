# Architecture
```mermaid
flowchart LR
  A[Webcam / Video file / SIMULATED synthetic] --> B[Detector<br/>HOG | YOLO | Synthetic replay]
  B --> C[IoU Tracker]
  C --> D[Zone foot-point test]
  D --> E1[Engine: BASELINE]
  D --> E2[Engine: PROPOSED<br/>threshold + persistence + hysteresis + dedup]
  E1 --> F[(SQLite incident store)]
  E2 --> F
  F --> G[Streamlit dashboard]
  F --> H[CSV / JSON export]
  F --> I[Evaluation module]
  F -.proposed, unvalidated.-> J[PipeGen-style event JSON]
```
| Module | Role |
|---|---|
| `src/event_engine.py` | UI/GPU-independent engine; deterministic given timestamps |
| `src/zone_manager.py` | polygon validation, point-in-polygon, foot-point |
| `src/detector.py`, `tracking.py` | detection + IDs |
| `src/pipeline.py` | wires everything; annotates frames; records latency |
| `src/incident_store.py` | SQLite persistence + retention |
| `src/evaluation.py` | metrics; refuses to compute P/R without ground truth |
| `src/synthetic.py` | SIMULATED scenarios with explicit ground-truth policy |
| `src/system_status.py` | ACTUAL vs SIMULATED vs NOT COMPLETED reporting |
| `app.py` | dashboard |

Engine state per track: `IDLE → CANDIDATE → ALERT(active) → RESOLVED`, or `CANDIDATE → SUPPRESSED(transient)`;
low-confidence in-zone streaks → `SUPPRESSED(low_confidence)`.
