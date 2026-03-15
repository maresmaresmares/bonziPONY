"""Loads GIF sprite animations and cycles frames as QPixmaps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)


@dataclass
class SpriteAnimation:
    """Holds all frames + per-frame delays for one GIF."""

    frames: list[QPixmap] = field(default_factory=list)
    delays: list[int] = field(default_factory=list)  # ms per frame


def _pil_to_qpixmap(pil_image: Image.Image, scale: float) -> QPixmap:
    """Convert a Pillow RGBA image to a scaled QPixmap."""
    rgba = pil_image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, 4 * rgba.width, QImage.Format_RGBA8888)
    # Must copy because data buffer will be garbage collected
    qimg = qimg.copy()
    pixmap = QPixmap.fromImage(qimg)
    if scale != 1.0:
        new_w = int(pixmap.width() * scale)
        new_h = int(pixmap.height() * scale)
        pixmap = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap


def _extract_gif_frames(gif_path: Path, scale: float) -> SpriteAnimation:
    """Extract all frames from a GIF with correct disposal handling.

    Desktop Ponies GIFs have no disposal method set and each frame is
    full-size, so each frame is rendered onto a fresh transparent canvas
    to avoid ghosting (solitaire card effect).
    """
    anim = SpriteAnimation()
    try:
        img = Image.open(gif_path)
    except Exception as exc:
        logger.warning("Failed to open GIF %s: %s", gif_path, exc)
        return anim

    n_frames = getattr(img, "n_frames", 1)

    for i in range(n_frames):
        try:
            img.seek(i)
        except EOFError:
            break

        # Fresh canvas each frame — prevents ghosting from transparent
        # pixels failing to overwrite previous frame's opaque pixels
        canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
        frame = img.convert("RGBA")
        canvas.paste(frame, (0, 0), frame)

        pixmap = _pil_to_qpixmap(canvas, scale)
        anim.frames.append(pixmap)

        delay = img.info.get("duration", 100)
        if delay <= 0:
            delay = 100
        anim.delays.append(delay)

    if not anim.frames:
        logger.warning("No frames extracted from %s", gif_path)

    return anim


class SpriteManager:
    """Loads and caches sprite animations from a pony directory."""

    # Keyword patterns used to auto-map behavior names → canonical sprite names
    _KEYWORD_MAP = [
        (("stand", "idle"),       "stand"),
        (("trot", "walk"),        "walk"),
        (("fly",),                "fly"),
        (("hover",),             "hover"),
        (("sleep",),             "sleep"),
        (("dizzy",),             "dizzy"),
        (("drag",),              "drag"),
        (("gallop", "dash", "zoom", "run"), "dash"),
        (("gala",),              "gala"),
    ]

    def __init__(self, pony_dir: Path, scale: float = 2.0) -> None:
        self.pony_dir = pony_dir
        self.scale = scale
        # Cache: "name_right" / "name_left" -> SpriteAnimation
        self._cache: Dict[str, SpriteAnimation] = {}
        # Canonical sprite name -> (right_gif, left_gif), built dynamically
        self._sprite_map: Dict[str, tuple[str, str]] = {}

    # ── Dynamic sprite map builder ─────────────────────────────────────────

    def build_sprite_map(self, behavior_manager) -> None:
        """Scan parsed behaviors by keyword and populate ``_sprite_map``.

        Call this after ``behavior_manager.parse()`` to map canonical names
        (stand, walk, fly, …) to the actual GIF filenames for the loaded pony.
        """
        self._sprite_map.clear()
        assigned: set[str] = set()

        for beh in behavior_manager.behaviors.values():
            name_lower = beh.name.lower()
            for keywords, canonical in self._KEYWORD_MAP:
                if canonical in assigned:
                    continue
                if any(kw in name_lower for kw in keywords):
                    self._sprite_map[canonical] = (beh.right_image, beh.left_image)
                    assigned.add(canonical)
                    break

        # Fallback: ensure "stand" always exists
        if "stand" not in self._sprite_map and behavior_manager.behaviors:
            first = next(iter(behavior_manager.behaviors.values()))
            self._sprite_map["stand"] = (first.right_image, first.left_image)

        logger.info(
            "Built sprite map for %s: %s",
            self.pony_dir.name,
            list(self._sprite_map.keys()),
        )

    def load_animation(self, name: str, gif_filename: str) -> SpriteAnimation:
        """Load a single GIF animation by filename, caching by name."""
        if name in self._cache:
            return self._cache[name]

        gif_path = self.pony_dir / gif_filename
        if not gif_path.exists():
            logger.warning("Sprite GIF not found: %s", gif_path)
            self._cache[name] = SpriteAnimation()
            return self._cache[name]

        anim = _extract_gif_frames(gif_path, self.scale)
        self._cache[name] = anim
        logger.debug("Loaded sprite %s (%d frames)", name, len(anim.frames))
        return anim

    def get_animation(self, name: str, facing_right: bool = True) -> SpriteAnimation:
        """Get a cached animation by name and direction.

        Falls back to ``"stand"`` if *name* isn't in the sprite map so that
        pony-specific animations (e.g. Rainbow Dash's "salute") degrade
        gracefully for other characters.
        """
        key = f"{name}_{'right' if facing_right else 'left'}"
        if key in self._cache:
            return self._cache[key]

        # Try to load from the dynamic sprite map
        if name in self._sprite_map:
            right_gif, left_gif = self._sprite_map[name]
            gif = right_gif if facing_right else left_gif
            return self.load_animation(key, gif)

        # Fallback to "stand" so missing pony-specific anims don't break
        if name != "stand" and "stand" in self._sprite_map:
            logger.debug("Animation '%s' not mapped — falling back to stand", name)
            return self.get_animation("stand", facing_right)

        logger.warning("Unknown animation: %s (no fallback available)", name)
        return SpriteAnimation()

    def get_by_gif(self, gif_filename: str) -> SpriteAnimation:
        """Load/get animation directly by GIF filename (as specified in pony.ini)."""
        if gif_filename in self._cache:
            return self._cache[gif_filename]
        return self.load_animation(gif_filename, gif_filename)

    def preload_all(self) -> None:
        """Load all mapped sprites at startup."""
        for name, (right_gif, left_gif) in self._sprite_map.items():
            self.load_animation(f"{name}_right", right_gif)
            self.load_animation(f"{name}_left", left_gif)
        logger.info("Preloaded %d sprite animations", len(self._cache))
