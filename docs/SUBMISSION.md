# Submission materials
**Abstract.** ContextGuard AI is a laptop prototype that turns person detections into explainable, temporally validated zone-intrusion
alerts. A configurable engine (confidence gate, persistence, hysteresis, dedup) is compared against an immediate-alert baseline on identical input, with every alert and suppression explained.

**Problem.** Object detectors report objects, not situations. Naive "person in zone" rules flicker with detector noise and fire on brief grazes, causing alert fatigue, and give no reason.

**Solution.** Pipeline: video → detector → tracker → foot-point zone test → engine → SQLite log → dashboard/export/evaluation.

**Innovation statement.** None of the components is novel; the contribution is a configurable, testable engine + evaluation harness that quantifies the trade-off (fewer false/fragmented alerts vs added delay). Verified only on synthetic data.

**Stack.** Python, OpenCV (HOG), optional Ultralytics YOLO, NumPy/Pandas, SQLite, Streamlit, Plotly, unittest/pytest — chosen for zero-cost, laptop-only reliability.

**Testing/evaluation.** 44 tests executed, all passed (unittest). Synthetic: proposed 2 alerts / precision 1.0 / recall 0.667 vs baseline 7 / 0.857 / 0.667; delay +2.2 s. Not real-world evidence.

**Limitations.** See README. **Future scope.** YOLO+ByteTrack, ground-plane zones, PipeGen→TensorRT on Jetson/NVIDIA, MQTT, actuator closed loop, real annotated data.

## Final checklist
- [x] Prototype runs; tests executed (44 OK)
- [x] Docs, compliance table, synthetic labelled
- [ ] Verify `streamlit run app.py` in a real browser on your laptop
- [ ] Try real video/webcam; (optional) install YOLO
- [ ] Access PipeGen portal; NVIDIA GPU/Jetson; compile TensorRT; real run + speed report in `evidence/pipegen/`
- [ ] Record demo video; submit repo/ZIP; state PipeGen status honestly
