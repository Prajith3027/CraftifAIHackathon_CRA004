# Proposed PipeGen requirement specification (DRAFT — NOT executed, NOT validated)
**Product:** zone-intrusion monitor. **Input:** video file / USB / RTSP. **Intended stages:**
1. Decode → 2. person detection (TensorRT-compiled detector) → 3. tracking → 4. zone foot-point test →
5. persistence/hysteresis rule engine → 6. JSON/MQTT event emission.
**Acceptance (proposed):** person entering the zone for ≥ N s emits one event with fields below; transient grazes emit none; per-frame latency and FPS reported from the real speed report.

Event schema (`contextguard.event.v1-proposed`): `schema, event_id, event_type, timestamp_utc, stream_time_s, confidence, duration_s, zone, track_id, reason, is_confirmed_incident (always false), source_type`.
Implemented in `src/pipegen_adapter.py` (unit-tested for shape only).

**Required to complete:** NVIDIA GPU/Jetson; PipeGen from the portal; its documented Requirement/Pipeline-IR/Model/Compiler/Run steps; real engine + speed report + output logs saved in `evidence/pipegen/`.
