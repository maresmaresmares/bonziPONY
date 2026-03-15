"""Moondream — lightweight local vision model for cheap screen descriptions."""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPT = (
    "Describe what's on this computer screen concisely in 2-3 sentences. "
    "Focus on which applications or windows are open, what content is displayed, "
    "and any notable text or activity. Ignore any small animated sprite overlay."
)


class MoondreamDescriber:
    """Lazy-loaded Moondream2 vision model for local screen descriptions."""

    def __init__(self, use_gpu: bool = False) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if use_gpu else "cpu"
        self._available = True

    def _load(self) -> bool:
        """Lazy-load model on first use."""
        if self._model is not None:
            return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "vikhyatk/moondream2"
            logger.info("Loading Moondream2 (%s)...", self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True,
            ).to(self._device).eval()
            logger.info("Moondream2 ready on %s.", self._device)
            return True
        except Exception as exc:
            logger.warning("Moondream2 failed to load: %s", exc)
            self._available = False
            return False

    @property
    def available(self) -> bool:
        return self._available

    def describe(self, jpeg_bytes: bytes) -> Optional[str]:
        """Describe a screenshot using Moondream. Returns text or None."""
        if not self._available:
            return None
        if not self._load():
            return None
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            enc_image = self._model.encode_image(img)
            result = self._model.answer_question(enc_image, _PROMPT, self._tokenizer)
            return result.strip() if result else None
        except Exception as exc:
            logger.warning("Moondream describe failed: %s", exc)
            return None
