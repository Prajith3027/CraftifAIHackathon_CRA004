"""Person detectors. All implement detect(frame, frame_idx) -> list[Detection].

* HOGPersonDetector  - OpenCV's built-in HOG+SVM people detector. No download needed.
                       Low accuracy vs. YOLO; 'confidence' is a pseudo-score (see below).
* YoloPersonDetector - Ultralytics YOLO (optional dependency + weights file/download).
* SyntheticDetector  - replays pre-defined detections (SIMULATED, for offline demos/tests).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox: BBox  # (x1, y1, x2, y2) in pixels
    confidence: float
    label: str = "person"


class DetectorUnavailable(RuntimeError):
    pass


class HOGPersonDetector:
    name = "opencv-hog"
    simulated = False

    def __init__(self, process_width: int = 480):
        try:
            import cv2
        except ImportError as e:  # pragma: no cover
            raise DetectorUnavailable("OpenCV (cv2) is not installed") from e
        self._cv2 = cv2
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.process_width = process_width

    def detect(self, frame, frame_idx: int = 0) -> List[Detection]:
        cv2 = self._cv2
        h, w = frame.shape[:2]
        scale = 1.0
        img = frame
        if w > self.process_width:
            scale = self.process_width / float(w)
            img = cv2.resize(frame, (self.process_width, int(h * scale)))
        rects, weights = self._hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        out: List[Detection] = []
        for (x, y, rw, rh), wt in zip(rects, weights):
            # HOG SVM margin is not a probability. Map to (0,1) monotonically; NOT calibrated.
            conf = 1.0 - math.exp(-max(float(wt), 0.0))
            out.append(Detection((x / scale, y / scale, (x + rw) / scale, (y + rh) / scale), conf))
        return out


class YoloPersonDetector:
    name = "ultralytics-yolo"
    simulated = False

    def __init__(self, model_path: str = "models/yolov8n.pt", min_conf: float = 0.1):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorUnavailable("ultralytics is not installed (pip install ultralytics)") from e
        p = Path(model_path)
        try:
            # If the file is missing Ultralytics tries to download it (needs internet).
            self._model = YOLO(str(p) if p.exists() else p.name)
        except Exception as e:
            raise DetectorUnavailable(f"Could not load YOLO weights '{model_path}': {e}") from e
        self.min_conf = min_conf

    def detect(self, frame, frame_idx: int = 0) -> List[Detection]:
        res = self._model.predict(frame, classes=[0], conf=self.min_conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append(Detection((x1, y1, x2, y2), float(b.conf[0])))
        return out


class SyntheticDetector:
    """Replays per-frame detections. Output is SIMULATED, never real perception."""
    name = "synthetic-replay"
    simulated = True

    def __init__(self, frames: Sequence[Sequence[Detection]]):
        self._frames = frames

    def detect(self, frame=None, frame_idx: int = 0) -> List[Detection]:
        if 0 <= frame_idx < len(self._frames):
            return list(self._frames[frame_idx])
        return []


def build_detector(preference: str = "auto", yolo_path: str = "models/yolov8n.pt",
                   process_width: int = 480):
    """Return (detector_or_None, status_message). Never raises."""
    errors = []
    if preference in ("auto", "yolo"):
        try:
            d = YoloPersonDetector(yolo_path)
            return d, f"YOLO loaded ({yolo_path})"
        except DetectorUnavailable as e:
            errors.append(f"YOLO unavailable: {e}")
            if preference == "yolo":
                return None, errors[0]
    try:
        d = HOGPersonDetector(process_width)
        msg = "OpenCV HOG person detector loaded (built-in, lower accuracy)"
        if errors:
            msg += " | " + errors[0]
        return d, msg
    except DetectorUnavailable as e:
        errors.append(str(e))
        return None, " | ".join(errors)
