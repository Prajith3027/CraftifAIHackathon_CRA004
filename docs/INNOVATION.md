# Innovation statement
**Not new:** object detection, polygon zones, temporal filtering, explainable messages.
**Our contribution (engineering, measurable):** a *single configurable engine* that combines confidence gating,
persistence validation, hysteresis over missing/uncertain/outside observations, and merge-based deduplication, with a
recorded reason for **every alert and every suppression**, plus a built-in baseline-vs-proposed evaluation on identical input.

## Existing approach (baseline)
Alert on the first confident in-zone detection. Detector dropouts and boundary jitter fragment one intrusion
into many alerts; brief grazes alert instantly.

## Proposed
- **Persistence** (`persistence_s`): alert only after continuous presence.
- **Hysteresis** (`exit_grace_s`): gaps from missing, low-confidence or momentarily-outside observations don't end the episode.
- **Confidence gate**: below-threshold in-zone observations never raise alerts; long streaks are logged as *possible missed intrusion*.
- **Dedup**: re-entry within `dedup_cooldown_s` of a resolved alert is merged into the same incident.

## Why it may reduce transient alerts
Spurious/brief events are short-lived; requiring persistence trades detection delay for fewer false alerts. Hysteresis
prevents one long event being counted many times.

## Explainability
Each incident carries a plain-language `reason` (thresholds, measured duration/confidence, gap counts, merge/suppression cause)
and a disclaimer that it is rule-triggered, not a confirmed incident.

## Evaluation methodology
Same observation stream → both engines. Metrics: candidates, alerts, suppressions, event-level precision/recall (alert overlaps
a ground-truth interval), alerts per detected GT event, median alert delay, false alerts/min, latency, FPS. Without ground truth,
P/R/FA-per-min are **not computed**.

## Measured result (SYNTHETIC only — our own scenario and annotation policy)
| | baseline | proposed |
|---|---|---|
| emitted alerts | 7 | 2 |
| false alerts (1 graze) | 1 | 0 |
| precision | 0.857 | 1.000 |
| recall | 0.667 | 0.667 |
| alerts per detected GT event | 3.0 | 1.0 |
| median alert delay | 0.0 s | 2.2 s |
| false alerts/min | 1.0 | 0.0 |

Both methods **miss** the low-confidence true intrusion (recall 0.667). The proposed method pays ≈ persistence delay.
Because the scenario and ground-truth policy were written by us, this demonstrates logic, **not** real-world superiority.

## Limitations / research directions
Real annotated video; parameter sweeps (ROC-like persistence curves); calibrated detector scores; ground-plane zones; learned vs rule gating.
