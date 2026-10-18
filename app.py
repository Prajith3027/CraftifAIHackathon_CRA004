"""ContextGuard AI - dashboard. Run:  streamlit run app.py"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src import exporters
from src.config import ConfigError, load_config, save_user_config, validate_config, USER_PATH
from src.detector import SyntheticDetector, build_detector
from src.evaluation import load_ground_truth, run_scenario, summarize
from src.incident_store import IncidentStore
from src.pipeline import MonitoringSession, run_source
from src.pipegen_adapter import to_actionable_event
from src.synthetic import SyntheticSource, mixed_scenario
from src.system_status import get_system_status
from src.video_source import SourceError, VideoFileSource, WebcamSource, probe_camera

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ContextGuard AI", page_icon="⚠️", layout="wide")

CSS = """
<style>
:root{--panel:#1a1f26;--line:#2a313b;--muted:#8b95a3;--haz:#f2c200;--bad:#e5484d;--ok:#3fb27f;--warn:#f5a524}
html,body,[class*="css"]{font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif}
.block-container{padding-top:1.2rem;max-width:1400px}
.tape{height:8px;border-radius:2px;margin-bottom:10px;
 background:repeating-linear-gradient(135deg,#f2c200 0 12px,#12151a 12px 24px)}
.brand{font-size:1.7rem;font-weight:700;letter-spacing:.01em;margin:0}
.sub{color:var(--muted);margin:0 0 .8rem 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.k{color:var(--muted);font-size:.78rem}.v{font-size:1.7rem;font-weight:600;line-height:1.2}
.chip{display:inline-block;padding:1px 9px;border-radius:3px;font-size:.78rem;font-weight:600;border:1px solid}
.ACTUAL{color:var(--ok);border-color:var(--ok)}.SIMULATED{color:var(--warn);border-color:var(--warn)}
.UNAVAILABLE{color:var(--muted);border-color:var(--muted)}.NOTCOMPLETED{color:var(--bad);border-color:var(--bad)}
.alert{border-left:4px solid var(--bad);background:#2a1a1d;padding:7px 10px;margin:4px 0;border-radius:3px;font-size:.88rem}
.supp{border-left:4px solid var(--warn);background:#2a2418;padding:7px 10px;margin:4px 0;border-radius:3px;font-size:.88rem}
.info{border-left:4px solid var(--line);background:#181c22;padding:7px 10px;margin:4px 0;border-radius:3px;font-size:.88rem}
.banner{border:1px dashed var(--warn);color:var(--warn);padding:6px 10px;border-radius:4px;font-size:.85rem}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="tape"></div><p class="brand">ContextGuard AI</p>'
            '<p class="sub">From object detection to explainable situational intelligence '
            '&nbsp;·&nbsp; research prototype, not safety-certified</p>', unsafe_allow_html=True)


# ------------------------------------------------------------------ state helpers
def get_cfg():
    if "cfg" not in st.session_state:
        try:
            st.session_state.cfg = load_config()
        except ConfigError as e:
            st.error(f"Configuration problem: {e}. Falling back to defaults.")
            st.session_state.cfg = load_config(use_user=False)
    return st.session_state.cfg


@st.cache_resource(show_spinner="Loading person detector…")
def get_detector(pref, yolo_path, width):
    return build_detector(pref, yolo_path, width)


@st.cache_resource
def get_store():
    return IncidentStore()


def card(label, value, note=""):
    st.markdown(f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div>'
                f'<div class="k">{note}</div></div>', unsafe_allow_html=True)


def chip(state):
    return f'<span class="chip {state.replace(" ", "")}">{state}</span>'


def chart_bar(df, x, y, color=None, title=""):
    if df is None or df.empty:
        st.info("No data yet.")
        return
    try:
        import plotly.express as px
        fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title, template="plotly_dark")
        fig.update_layout(paper_bgcolor="#12151a", plot_bgcolor="#1a1f26", margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption(title)
        st.bar_chart(df.pivot_table(index=x, columns=color, values=y, aggfunc="sum") if color else df.set_index(x)[y])


def to_df(recs):
    df = pd.DataFrame(recs)
    return df


cfg = get_cfg()
store = get_store()
pages = ["Overview", "Live Monitoring", "Event Analytics", "Incident Log", "Configuration",
         "Export", "System Status"]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Alerts are **rule-triggered**, not confirmed incidents. "
                   "Synthetic input is always labelled SIMULATED.")

# ------------------------------------------------------------------ Overview
if page == "Overview":
    prop = store.list(method="proposed")
    alerts = [r for r in prop if r["category"] == "rule_alert"]
    active = [r for r in alerts if r["status"] == "active"]
    today = datetime.now(timezone.utc).date().isoformat()
    today_n = sum(1 for r in alerts if str(r["detected_at"]).startswith(today))
    cand = len(alerts) + sum(int(r["merged_reentries"] or 0) for r in alerts) + \
        sum(1 for r in prop if r["category"] == "suppressed_transient")
    runs = store.runs()
    frames = sum(r["frames_processed"] or 0 for r in runs)
    avg = (sum((r["avg_latency_ms"] or 0) * (r["frames_processed"] or 0) for r in runs) / frames) if frames else None
    c = st.columns(5)
    with c[0]: card("Candidate events (proposed)", cand, "all runs in store")
    with c[1]: card("Active alerts", len(active), "status = active")
    with c[2]: card("Alerts today (UTC)", today_n, "rule-triggered")
    with c[3]: card("Avg processing time", f"{avg:.1f} ms" if avg is not None else "—", "per processed frame")
    with c[4]:
        card("Monitoring status", "Idle", f"last run: {runs[-1]['run_id']}" if runs else "no runs yet")
    st.write("")
    if not runs:
        st.info("No monitoring runs recorded yet. Open **Live Monitoring** and run the "
                "**Synthetic demo (SIMULATED)** or upload a video.")
    else:
        st.subheader("Recent runs")
        st.dataframe(pd.DataFrame(runs).drop(columns=["config_json"]).tail(10), use_container_width=True)
        st.subheader("Latest incidents")
        st.dataframe(pd.DataFrame(prop).tail(8)[["detected_at", "incident_id", "category", "confidence",
                                                  "duration_s", "status", "source_type"]], use_container_width=True)

# ------------------------------------------------------------------ Live monitoring
elif page == "Live Monitoring":
    det, det_msg = get_detector(cfg["detector"], cfg["yolo_model_path"], cfg["process_width"])
    st.caption(f"Detector: {det_msg}")
    mode = st.radio("Input", ["Synthetic demo (SIMULATED)", "Uploaded video", "Webcam"], horizontal=True)
    up = idx = None
    max_s = 60
    if mode == "Uploaded video":
        up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    elif mode == "Webcam":
        idx = st.number_input("Camera index", 0, 10, 0)
        max_s = st.slider("Max seconds to monitor", 10, 600, 60)
        if st.button("Test camera"):
            ok, msg = probe_camera(int(idx))
            (st.success if ok else st.error)(msg)
        st.caption("Webcam monitoring ends after the limit or when you click anything else.")
    else:
        st.markdown('<div class="banner">SIMULATED INPUT: scripted detections on a blank canvas — '
                    'not a camera feed and not real perception.</div>', unsafe_allow_html=True)
    ui_every = st.slider("Refresh display every N processed frames", 1, 10, 2)
    if st.button("▶ Start monitoring", type="primary"):
        try:
            if mode == "Synthetic demo (SIMULATED)":
                sc = mixed_scenario()
                source, detector, stype, name = SyntheticSource(sc), SyntheticDetector(sc.frames), "synthetic", sc.name
                stride = 1
            else:
                if det is None:
                    raise SourceError(f"No person detector available: {det_msg}")
                detector, stride = det, cfg["frame_stride"]
                if mode == "Uploaded video":
                    if up is None:
                        raise SourceError("Please upload a video file first.")
                    d = ROOT / "data" / "uploads"
                    d.mkdir(parents=True, exist_ok=True)
                    path = d / Path(up.name).name
                    path.write_bytes(up.getbuffer())
                    source, stype, name = VideoFileSource(str(path)), "video_file", up.name
                else:
                    source, stype, name = WebcamSource(int(idx)), "webcam", f"camera {int(idx)}"
            session = MonitoringSession(cfg, detector, stype, store, name)
        except (SourceError, ConfigError, ValueError) as e:
            st.error(str(e))
        else:
            left, right = st.columns([3, 2])
            frame_ph, stat_ph = left.empty(), left.empty()
            right.markdown("**Explainable alert feed**")
            feed_ph = right.empty()
            feed = []
            n = 0
            try:
                for r in run_source(source, session, stride, max_seconds=max_s if mode == "Webcam" else None):
                    n += 1
                    for ev in r.events:
                        if ev.kind == "CANDIDATE_STARTED":
                            continue
                        css = "alert" if ev.kind in ("ALERT_RAISED", "ALERT_MERGED") else \
                            "supp" if ev.kind == "SUPPRESSED" else "info"
                        tag = "baseline" if ev.incident_id and "-B" in ev.incident_id else "proposed"
                        feed.insert(0, f'<div class="{css}"><b>{ev.kind}</b> t={ev.ts:.1f}s [{tag}] {ev.message}</div>')
                    if n % ui_every == 0:
                        frame_ph.image(r.annotated[:, :, ::-1], use_container_width=True)
                        stat_ph.markdown(f"State: **{r.state}** · frame {r.frame_idx} · "
                                         f"{r.latency_ms:.1f} ms (detect {r.detect_ms:.1f} ms) · "
                                         f"active alerts {r.n_active}")
                        feed_ph.markdown("".join(feed[:12]) or "_No events yet_", unsafe_allow_html=True)
            except Exception as e:  # keep the UI alive on decoder/detector failures
                st.error(f"Monitoring stopped: {type(e).__name__}: {e}")
            summ = session.finish()
            feed_ph.markdown("".join(feed[:12]) or "_No events_", unsafe_allow_html=True)
            st.success(f"Run {summ['run_id']} finished: {summ['frames_processed']} frames, "
                       f"avg {summ['avg_latency_ms'] or 0:.1f} ms/frame, {summ['fps'] or 0:.1f} FPS")
            st.json(summ["counters"])

# ------------------------------------------------------------------ Analytics
elif page == "Event Analytics":
    runs = store.runs()
    allrec = store.list()
    if not allrec:
        st.info("No recorded events yet. Run a source in Live Monitoring first.")
    else:
        df = to_df(allrec)
        run_ids = list(df["run_id"].unique())
        sel = st.selectbox("Run", run_ids, index=len(run_ids) - 1)
        d = df[df["run_id"] == sel].copy()
        d["bucket_s"] = (d["start_ts"].astype(float) // 10 * 10).astype(int)
        al = d[d["category"] == "rule_alert"]
        st.subheader("Alert frequency over stream time (10 s buckets)")
        chart_bar(al.groupby(["bucket_s", "method"]).size().reset_index(name="alerts"),
                  "bucket_s", "alerts", "method")
        st.subheader("Intrusions by duration category (proposed method, alerts + transients)")
        p = d[(d["method"] == "proposed") & d["category"].isin(["rule_alert", "suppressed_transient"])].copy()
        if p.empty:
            st.info("No intrusions in this run.")
        else:
            p["duration"] = pd.cut(p["duration_s"].astype(float), [-1, 2, 5, 10, 1e9],
                                   labels=["<2 s", "2–5 s", "5–10 s", ">10 s"])
            chart_bar(p.groupby(["duration", "category"], observed=True).size().reset_index(name="count"),
                      "duration", "count", "category")
        st.subheader("Baseline vs proposed (this run, identical observations)")
        cmp = []
        for m in ("baseline", "proposed"):
            s = summarize([r for r in allrec if r["run_id"] == sel and r["method"] == m])
            cmp.append({"method": m, **{k: s[k] for k in ("candidate_events", "emitted_alerts",
                        "suppressed_transient", "suppressed_low_confidence", "merged_reentries")}})
        cdf = pd.DataFrame(cmp)
        st.dataframe(cdf, use_container_width=True)
        chart_bar(cdf.melt("method", var_name="metric", value_name="count"), "metric", "count", "method")
        st.caption("Precision/recall need ground truth — upload it on the Export page.")
    st.subheader("Processing performance")
    if runs:
        rdf = pd.DataFrame(runs)[["run_id", "source_type", "detector", "frames_processed",
                                  "avg_latency_ms", "p95_latency_ms", "avg_detect_ms", "fps"]]
        st.dataframe(rdf, use_container_width=True)
        st.line_chart(rdf.set_index("run_id")[["avg_latency_ms", "p95_latency_ms"]])
    else:
        st.info("No runs yet.")
    st.subheader("Rule-engine benchmark on SYNTHETIC data")
    st.caption("Ground truth here is defined by our own policy (stay ≥ 2 s in zone). This validates the "
               "engine logic only; it is not evidence of real-world accuracy.")
    if st.button("Run synthetic benchmark"):
        st.session_state.synth_eval = run_scenario(mixed_scenario(), cfg)
    ev = st.session_state.get("synth_eval")
    if ev:
        rows = [{"method": m, **{k: v for k, v in x["metrics"].items() if k not in ("note",)}}
                for m, x in ev["methods"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.json(ev["performance"])

# ------------------------------------------------------------------ Incident log
elif page == "Incident Log":
    recs = store.list()
    if not recs:
        st.info("The incident log is empty.")
    else:
        df = to_df(recs)
        c = st.columns(4)
        m = c[0].multiselect("Method", sorted(df["method"].unique()), default=["proposed"])
        cat = c[1].multiselect("Category", sorted(df["category"].unique()), default=list(df["category"].unique()))
        stt = c[2].multiselect("Status", sorted(df["status"].unique()), default=list(df["status"].unique()))
        src = c[3].multiselect("Source", sorted(df["source_type"].unique()), default=list(df["source_type"].unique()))
        f = df[df["method"].isin(m) & df["category"].isin(cat) & df["status"].isin(stt) & df["source_type"].isin(src)]
        st.dataframe(f[["detected_at", "incident_id", "category", "confidence", "duration_s", "status",
                        "source_type", "reason"]], use_container_width=True, height=420)
        st.caption("status: active / resolved (alert) or suppressed. Every row is a rule outcome, "
                   "not a verified incident.")
    with st.expander("Danger zone"):
        if st.checkbox("I understand this permanently deletes all incidents and run history") and st.button("Clear all"):
            store.clear()
            st.rerun()

# ------------------------------------------------------------------ Configuration
elif page == "Configuration":
    with st.form("cfgform"):
        c1, c2 = st.columns(2)
        thr = c1.slider("Confidence threshold", 0.0, 1.0, float(cfg["confidence_threshold"]), 0.05)
        pers = c1.slider("Intrusion persistence threshold (s)", 0.0, 30.0, float(cfg["persistence_s"]), 0.1)
        grace = c1.slider("Exit grace / hysteresis (s)", 0.0, 10.0, float(cfg["exit_grace_s"]), 0.1)
        cool = c1.slider("Dedup cooldown (s)", 0.0, 60.0, float(cfg["dedup_cooldown_s"]), 0.5)
        stride = c2.slider("Process every Nth frame (video/webcam)", 1, 30, int(cfg["frame_stride"]))
        days = c2.number_input("Retention: max days", 1, 3650, int(cfg["retention"]["max_days"]))
        recs_max = c2.number_input("Retention: max records", 10, 1_000_000, int(cfg["retention"]["max_records"]))
        det = c2.selectbox("Detector", ["auto", "hog", "yolo"], ["auto", "hog", "yolo"].index(cfg["detector"]))
        zname = st.text_input("Zone name", cfg["zone"]["name"])
        ztxt = st.text_area("Zone polygon — one 'x,y' per line, normalised 0–1 (min 3 points)",
                            "\n".join(f"{x},{y}" for x, y in cfg["zone"]["polygon"]), height=130)
        saved = st.form_submit_button("Save configuration", type="primary")
    if saved:
        try:
            poly = [[float(v) for v in ln.split(",")] for ln in ztxt.strip().splitlines() if ln.strip()]
            new = json.loads(json.dumps(cfg))
            new.update(confidence_threshold=thr, persistence_s=pers, exit_grace_s=grace,
                       dedup_cooldown_s=cool, frame_stride=stride, detector=det)
            new["retention"] = {"max_days": int(days), "max_records": int(recs_max)}
            new["zone"] = {"name": zname, "polygon": poly}
            save_user_config(new)
            st.session_state.cfg = new
            st.success("Saved to config/user_config.json")
        except (ConfigError, ValueError) as e:
            st.error(f"Not saved: {e}")
    if st.button("Reset to defaults"):
        if USER_PATH.exists():
            USER_PATH.unlink()
        st.session_state.pop("cfg", None)
        st.rerun()
    # zone preview
    import cv2
    import numpy as np
    try:
        from src.zone_manager import Zone
        z = Zone(cfg["zone"]["name"], cfg["zone"]["polygon"])
        img = np.full((270, 480, 3), (26, 21, 18), np.uint8)
        pts = np.array(z.to_pixels(480, 270), np.int32)
        cv2.fillPoly(img, [pts], (60, 60, 140))
        cv2.polylines(img, [pts], True, (77, 77, 229), 2)
        st.image(img[:, :, ::-1], caption="Current zone preview (normalised coordinates)", width=480)
    except Exception as e:
        st.error(f"Zone preview unavailable: {e}")

# ------------------------------------------------------------------ Export
elif page == "Export":
    recs = store.list()
    method = st.selectbox("Method to export", ["all", "proposed", "baseline"])
    sel = recs if method == "all" else [r for r in recs if r["method"] == method]
    st.write(f"{len(sel)} real recorded records (synthetic runs carry source_type = synthetic).")
    c = st.columns(3)
    c[0].download_button("Download CSV", exporters.incidents_to_csv(sel), "incidents.csv", "text/csv")
    c[1].download_button("Download JSON", exporters.incidents_to_json(sel), "incidents.json", "application/json")
    if c[2].button("Also save copies to outputs/"):
        exporters.write_text(ROOT / "outputs" / "incidents.csv", exporters.incidents_to_csv(sel))
        exporters.write_text(ROOT / "outputs" / "incidents.json", exporters.incidents_to_json(sel))
        st.success("Saved to outputs/")
    st.subheader("Evaluation summary")
    runs = sorted({r["run_id"] for r in recs})
    if not runs:
        st.info("No runs to evaluate.")
    else:
        rid = st.selectbox("Run", runs, index=len(runs) - 1)
        gt_file = st.file_uploader("Ground-truth JSON: [{\"start\": s, \"end\": s}, …] in video seconds",
                                   type=["json"])
        gt = None
        if gt_file:
            try:
                gt = load_ground_truth(gt_file.getvalue().decode())
            except ValueError as e:
                st.error(str(e))
        rr = next((r for r in store.runs() if r["run_id"] == rid), None)
        out = {"run": rid, "methods": {}}
        for m in ("baseline", "proposed"):
            out["methods"][m] = summarize([r for r in recs if r["run_id"] == rid and r["method"] == m], gt,
                                          rr["duration_s"] if rr else None)
        if rr:
            out["performance"] = {k: rr[k] for k in ("frames_processed", "avg_latency_ms", "p95_latency_ms", "fps")}
        st.json(out)
        st.download_button("Download evaluation summary (JSON)", exporters.evaluation_to_json(out),
                           "evaluation_summary.json", "application/json")
    st.subheader("Proposed PipeGen-style actionable events (schema NOT validated against PipeGen)")
    pa = [to_actionable_event(r) for r in recs if r["method"] == "proposed" and r["category"] == "rule_alert"]
    st.download_button("Download actionable events JSON", json.dumps(pa, indent=2), "actionable_events.json")

# ------------------------------------------------------------------ System status
else:
    if st.button("Test camera 0"):
        st.session_state.cam = probe_camera(0)
    det, det_msg = get_detector(cfg["detector"], cfg["yolo_model_path"], cfg["process_width"])
    rows = get_system_status(det, det_msg, st.session_state.get("cam"), cfg)
    for r in rows:
        st.markdown(f'{chip(r["state"])} &nbsp; **{r["component"]}** — {r["detail"]}', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**ACTUAL** = running here · **SIMULATED** = scripted test data · **UNAVAILABLE** = missing · "
                "**NOT COMPLETED** = official hackathon requirement not yet met (see docs/HACKATHON_COMPLIANCE.md).")
