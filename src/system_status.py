"""Honest system status: every component is labelled ACTUAL, SIMULATED, UNAVAILABLE or NOT COMPLETED."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def pipegen_evidence_files() -> List[str]:
    d = ROOT / "evidence" / "pipegen"
    return sorted(p.name for p in d.glob("*") if p.is_file() and p.name != ".gitkeep") if d.exists() else []


def get_system_status(detector=None, detector_msg: str = "", camera: Optional[tuple] = None,
                      cfg: Optional[dict] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def add(component, state, detail):
        rows.append({"component": component, "state": state, "detail": detail})

    add("OpenCV", "ACTUAL" if _has("cv2") else "UNAVAILABLE",
        "image/video I/O and HOG detector" if _has("cv2") else "pip install opencv-python")
    if detector is None:
        add("Person detector", "UNAVAILABLE", detector_msg or "not loaded")
    else:
        add("Person detector", "ACTUAL", f"{detector.name}: {detector_msg}")
    yolo_path = ROOT / (cfg or {}).get("yolo_model_path", "models/yolov8n.pt")
    add("YOLO (optional)", "ACTUAL" if _has("ultralytics") and yolo_path.exists() else "UNAVAILABLE",
        f"ultralytics {'installed' if _has('ultralytics') else 'not installed'}; weights "
        f"{'found' if yolo_path.exists() else 'not found'} at {yolo_path.relative_to(ROOT)}")
    if camera is None:
        add("Camera", "UNAVAILABLE", "not probed yet - press 'Test camera' in Live Monitoring/System Status")
    else:
        add("Camera", "ACTUAL" if camera[0] else "UNAVAILABLE", camera[1])
    gpu = _nvidia_smi()
    add("NVIDIA GPU", "ACTUAL" if gpu else "UNAVAILABLE", gpu or "nvidia-smi not found: CPU-only inference")
    add("TensorRT", "ACTUAL" if _has("tensorrt") else "UNAVAILABLE",
        "python package present (no engine compiled by this project)" if _has("tensorrt")
        else "not installed; no TensorRT engine exists in this project")
    ev = pipegen_evidence_files()
    add("CraftifAI PipeGen", "NOT COMPLETED",
        "No PipeGen run, Pipeline IR, compiled engine or speed report has been produced by this project."
        + (f" Files found in evidence/pipegen/ (unverified): {', '.join(ev)}" if ev else
           " evidence/pipegen/ is empty."))
    add("Edge hardware (Jetson/ESP32/STM32)", "NOT COMPLETED", "No physical hardware connected or flashed.")
    add("Synthetic scenario generator", "SIMULATED", "Deterministic simulated detections; never a real camera feed.")
    return rows
