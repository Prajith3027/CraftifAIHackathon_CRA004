"""Frame sources with explicit error handling. Yields (frame_bgr, timestamp_s, frame_idx)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Tuple


class SourceError(RuntimeError):
    """Camera/video cannot be opened or read."""


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise SourceError("OpenCV (cv2) is not installed: pip install opencv-python") from e


class VideoFileSource:
    kind = "video_file"

    def __init__(self, path: str):
        cv2 = _cv2()
        p = Path(path)
        if not p.exists():
            raise SourceError(f"Video file not found: {path}")
        self.cap = cv2.VideoCapture(str(p))
        if not self.cap.isOpened():
            raise SourceError(f"Cannot decode video (unsupported codec or corrupt file): {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1e-3 else 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = n if n > 0 else None

    def frames(self) -> Iterator[Tuple[object, float, int]]:
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame, i / self.fps, i       # timestamp = video time (deterministic)
            i += 1

    def release(self):
        self.cap.release()


class WebcamSource:
    kind = "webcam"

    def __init__(self, index: int = 0):
        cv2 = _cv2()
        self.cap = cv2.VideoCapture(int(index))
        if not self.cap.isOpened():
            raise SourceError(f"Camera {index} unavailable. Check it is connected, not used by another "
                              f"app, and that this process has camera permission (OS privacy settings).")
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise SourceError(f"Camera {index} opened but returned no frames.")
        self.height, self.width = frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.total_frames = None
        self._first = frame

    def frames(self) -> Iterator[Tuple[object, float, int]]:
        t0 = time.monotonic()
        i = 0
        frame = self._first
        while True:
            yield frame, time.monotonic() - t0, i      # wall-clock timestamp
            i += 1
            ok, frame = self.cap.read()
            if not ok:
                break

    def release(self):
        self.cap.release()


def probe_camera(index: int = 0) -> Tuple[bool, str]:
    """Try to open a camera once. Returns (ok, message)."""
    try:
        s = WebcamSource(index)
    except SourceError as e:
        return False, str(e)
    msg = f"Camera {index} connected ({s.width}x{s.height})"
    s.release()
    return True, msg
