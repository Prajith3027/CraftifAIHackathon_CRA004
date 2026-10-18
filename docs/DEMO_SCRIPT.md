# 3-minute demo
1. **0:00** Problem: "Person detected ≠ person in danger." Show System Status: ACTUAL vs SIMULATED vs NOT COMPLETED (be upfront: laptop prototype, PipeGen not done).
2. **0:30** Configuration: show zone, confidence 0.5, persistence 2.0 s.
3. **0:50** Live Monitoring → *Synthetic demo (SIMULATED)* → Start. Narrate: graze at ~15 s is suppressed (proposed) but alerts (baseline); flickering intrusion yields 1 alert vs several.
4. **1:50** Alert feed: read one explanation (thresholds, duration, gaps).
5. **2:10** Event Analytics: baseline vs proposed table; run synthetic benchmark; state it validates logic only.
6. **2:35** Incident Log + Export CSV/JSON. Upload `data/synthetic_ground_truth.json` on Export to show precision/recall appear only with ground truth.
7. **2:50** Limitations: low-confidence miss, delay cost, no real-world metrics, no TensorRT/edge yet; next step PipeGen on Jetson.
