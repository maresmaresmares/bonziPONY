"""
Acknowledgement player.

Randomly picks a WAV/MP3 from a per-character folder and plays it immediately
so the user gets instant feedback that the pony heard them — before the slower
STT stage.

Sounds are looked up in:  acknowledgement/assets/{preset_slug}/
If no folder or no files exist for the active character, ack is silently skipped.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"


class AcknowledgementPlayer:
    """Plays a random acknowledgement sound for the active character."""

    def __init__(self, assets_dir: Path | None = None) -> None:
        self._root = assets_dir or ASSETS_DIR
        self._wavs: list[Path] = []
        self._current_slug: str = ""

    def set_character(self, slug: str) -> None:
        """Switch to a character's acknowledgement sounds.

        Looks in assets/{slug}/ for .wav and .mp3 files.
        Backward compat: if slug is 'rainbow_dash' and no subdir exists,
        falls back to root assets/ directory.
        """
        self._current_slug = slug
        char_dir = self._root / slug

        if char_dir.is_dir():
            self._wavs = sorted(
                list(char_dir.glob("*.wav")) + list(char_dir.glob("*.mp3"))
            )
        elif slug == "rainbow_dash":
            # Backward compat: existing files in root assets/ are Dash's
            self._wavs = sorted(
                list(self._root.glob("*.wav")) + list(self._root.glob("*.mp3"))
            )
        else:
            self._wavs = []

        if self._wavs:
            logger.debug("Loaded %d ack sounds for '%s'.", len(self._wavs), slug)
        else:
            logger.debug("No ack sounds for '%s' — acknowledgements disabled.", slug)

    def play(self) -> None:
        """Play a random acknowledgement sound. Blocks until done."""
        if not self._wavs:
            return

        chosen = random.choice(self._wavs)
        logger.debug("Playing acknowledgement: %s", chosen.name)

        try:
            import sounddevice as sd
            import soundfile as sf

            data, samplerate = sf.read(str(chosen), dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
        except Exception as exc:
            logger.warning("Could not play acknowledgement sound %s: %s", chosen.name, exc)

    def get_assets_dir(self) -> Path:
        """Return the directory where ack sounds for the current character should go."""
        if self._current_slug:
            return self._root / self._current_slug
        return self._root
