"""
Speaker verification — only transcribe audio that matches the enrolled user's voice.

Uses resemblyzer to create a voice embedding (fingerprint) from an enrollment sample,
then compares each captured audio segment against it. Non-matching audio (YouTube,
other people, TV) is discarded before hitting Whisper, saving compute and preventing
false transcriptions.

Two-layer filtering:
  1. Energy gate — audio from speakers is quieter than direct mic speech. Reject
     audio with RMS energy well below the user's enrollment baseline.
  2. Speaker verification — cosine similarity between voice embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROFILE_DIR = Path(__file__).parent.parent / "voice_profile"
_PROFILE_FILE = _PROFILE_DIR / "user_profile.npz"      # new format (embedding + energy)
_LEGACY_EMBEDDING = _PROFILE_DIR / "user_embedding.npy" # old format (embedding only)
SAMPLE_RATE = 16000

# Similarity threshold — higher = stricter (0.0 to 1.0)
# resemblyzer's GE2E model typically gives same-speaker similarity of 0.55-0.80.
# 0.45 is a reasonable floor — the energy gate does the heavy lifting for
# rejecting speaker audio. This catches obviously-different voices.
DEFAULT_THRESHOLD = 0.45

# Energy gate — reject audio with RMS below this fraction of enrollment RMS.
# Speakers playing YouTube/TV are typically much quieter than direct mic speech.
ENERGY_RATIO_MIN = 0.25


class VoiceFilter:
    """Compares captured audio against an enrolled voice profile."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._encoder = None  # lazy-loaded
        self._user_embedding: Optional[np.ndarray] = None
        self._user_rms: float = 0.0  # RMS energy baseline from enrollment
        self._load_profile()

    @property
    def enrolled(self) -> bool:
        """True if a voice profile exists."""
        return self._user_embedding is not None

    def _get_encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            logger.info("VoiceEncoder loaded.")
        return self._encoder

    def _load_profile(self) -> None:
        """Load saved voice profile (new .npz or legacy .npy)."""
        if _PROFILE_FILE.exists():
            try:
                data = np.load(str(_PROFILE_FILE))
                self._user_embedding = data["embedding"]
                self._user_rms = float(data.get("rms_energy", 0.0))
                logger.info("Voice profile loaded (threshold=%.2f, rms=%.4f)",
                            self._threshold, self._user_rms)
            except Exception as exc:
                logger.warning("Failed to load voice profile: %s", exc)
                self._user_embedding = None
        elif _LEGACY_EMBEDDING.exists():
            # Migrate from old format
            try:
                self._user_embedding = np.load(str(_LEGACY_EMBEDDING))
                self._user_rms = 0.0  # unknown — energy gate disabled until re-enrollment
                logger.info("Loaded legacy voice profile (no energy baseline — re-enroll for best results).")
            except Exception as exc:
                logger.warning("Failed to load legacy voice profile: %s", exc)
                self._user_embedding = None

    def enroll(self, audio_f32: np.ndarray) -> bool:
        """
        Enroll the user's voice from a float32 audio sample (16 kHz).

        Computes a single voice embedding from the full audio sample and saves
        the RMS energy baseline for the energy gate.
        The audio should be 5-15 seconds of the user speaking naturally.
        Returns True on success.
        """
        try:
            from resemblyzer import preprocess_wav
            encoder = self._get_encoder()

            # Preprocess and compute embedding (resemblyzer normalizes internally)
            processed = preprocess_wav(audio_f32, source_sr=SAMPLE_RATE)
            embedding = encoder.embed_utterance(processed)

            # Compute RMS energy baseline from the original audio
            rms_energy = float(np.sqrt(np.mean(audio_f32 ** 2)))

            # Save to disk
            _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(str(_PROFILE_FILE), embedding=embedding, rms_energy=rms_energy)

            # Clean up legacy file if it exists
            if _LEGACY_EMBEDDING.exists():
                try:
                    _LEGACY_EMBEDDING.unlink()
                except Exception:
                    pass

            self._user_embedding = embedding
            self._user_rms = rms_energy

            logger.info("Voice profile enrolled: %d-dim embedding, RMS=%.4f.",
                        len(embedding), rms_energy)
            return True
        except Exception as exc:
            logger.error("Voice enrollment failed: %s", exc)
            return False

    def is_user(self, audio_f32: np.ndarray) -> bool:
        """
        Check if the given audio (float32, 16 kHz) matches the enrolled user.

        Two-layer check:
          1. Energy gate — reject audio much quieter than enrollment baseline
             (catches YouTube/TV from speakers which is quieter than direct mic speech)
          2. Speaker embedding similarity — reject voices that don't match

        Returns True if no profile is enrolled or audio passes both checks.
        Returns False if audio fails either check.
        """
        if self._user_embedding is None:
            return True  # No enrollment = no filtering

        if len(audio_f32) < SAMPLE_RATE * 0.5:
            # Less than 0.5s of audio — too short to verify, allow it
            return True

        try:
            # ── Layer 1: Energy gate ──────────────────────────────────────
            rms = float(np.sqrt(np.mean(audio_f32 ** 2)))

            if self._user_rms > 0 and rms < self._user_rms * ENERGY_RATIO_MIN:
                logger.info("Voice rejected — energy too low: RMS=%.4f vs baseline=%.4f (ratio=%.2f, min=%.2f). "
                           "Likely speakers, not direct mic.",
                           rms, self._user_rms, rms / self._user_rms, ENERGY_RATIO_MIN)
                return False

            # ── Layer 2: Speaker embedding similarity ─────────────────────
            from resemblyzer import preprocess_wav
            encoder = self._get_encoder()

            processed = preprocess_wav(audio_f32, source_sr=SAMPLE_RATE)
            if len(processed) < SAMPLE_RATE * 0.3:
                return True  # Too short after preprocessing

            embedding = encoder.embed_utterance(processed)

            # Cosine similarity
            similarity = float(np.dot(self._user_embedding, embedding) / (
                np.linalg.norm(self._user_embedding) * np.linalg.norm(embedding)
            ))

            logger.info("Voice check — similarity: %.3f (threshold: %.3f), RMS: %.4f (baseline: %.4f)",
                        similarity, self._threshold, rms, self._user_rms)
            print(f"[Voice] similarity={similarity:.3f} threshold={self._threshold:.3f} | "
                  f"RMS={rms:.4f} baseline={self._user_rms:.4f} ratio={rms/self._user_rms:.2f}"
                  if self._user_rms > 0 else
                  f"[Voice] similarity={similarity:.3f} threshold={self._threshold:.3f} | RMS={rms:.4f}")

            if similarity >= self._threshold:
                return True
            else:
                logger.info("Voice rejected — similarity %.3f < threshold %.3f (not the user)",
                           similarity, self._threshold)
                return False

        except Exception as exc:
            logger.warning("Voice verification failed: %s — allowing audio", exc)
            return True  # On error, don't block

    def delete_profile(self) -> None:
        """Delete the saved voice profile."""
        for f in (_PROFILE_FILE, _LEGACY_EMBEDDING):
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass
        self._user_embedding = None
        self._user_rms = 0.0
        logger.info("Voice profile deleted.")
