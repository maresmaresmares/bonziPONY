"""Screenshot capture — grabs the primary monitor and returns JPEG bytes."""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Captures screenshots of the primary monitor via mss + Pillow."""

    def __init__(self, max_width: int = 1280) -> None:
        try:
            import mss  # noqa: F401
            self._mss_mod = mss
        except ImportError:
            raise ImportError("mss is required for screen capture: pip install mss")

        self._max_width = max_width
        self._available = True
        logger.info("ScreenCapture ready (max_width=%d).", max_width)

    @property
    def available(self) -> bool:
        return self._available

    def grab_pil(self):
        """Capture the primary monitor and return a PIL Image, or None on failure."""
        if not self._available:
            return None

        try:
            from PIL import Image

            with self._mss_mod.mss() as sct:
                monitor = sct.monitors[1]
                raw = sct.grab(monitor)

            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

            if img.width > self._max_width:
                ratio = self._max_width / img.width
                new_h = int(img.height * ratio)
                img = img.resize((self._max_width, new_h), Image.LANCZOS)

            return img

        except Exception as exc:
            logger.warning("Screen capture failed: %s", exc)
            return None

    def grab(self) -> Optional[bytes]:
        """Capture the primary monitor and return JPEG bytes, or None on failure."""
        img = self.grab_pil()
        if img is None:
            return None
        try:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=60)
            return buf.getvalue()
        except Exception as exc:
            logger.warning("JPEG encode failed: %s", exc)
            return None
