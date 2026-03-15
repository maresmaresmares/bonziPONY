"""Webcam capture — detects camera on init and grabs JPEG frames."""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Camera:
    """Wraps OpenCV webcam capture. Call grab() to get a JPEG frame as bytes."""

    def __init__(self, device_index: int = 0) -> None:
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            raise ImportError("opencv-python is required for vision: pip install opencv-python")

        self._index = device_index
        self._cap = None
        self.available = self._probe()

    def _probe(self) -> bool:
        """Try opening the camera to confirm it exists."""
        cap = self._cv2.VideoCapture(self._index)
        ok = cap.isOpened()
        cap.release()
        if ok:
            logger.info("Camera detected at index %d.", self._index)
        else:
            logger.warning("No camera found at index %d — vision disabled.", self._index)
        return ok

    def grab(self) -> Optional[bytes]:
        """Capture one frame and return it as JPEG bytes, or None on failure."""
        if not self.available:
            return None

        cap = self._cv2.VideoCapture(self._index)
        try:
            if not cap.isOpened():
                logger.warning("Camera unavailable for capture.")
                return None

            # Discard first few frames — cameras often return dark/stale frames immediately
            for _ in range(3):
                cap.read()

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Camera returned empty frame.")
                return None

            ret, buf = self._cv2.imencode(".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                logger.warning("JPEG encode failed.")
                return None

            return buf.tobytes()
        finally:
            cap.release()
