"""Persist brief session summaries between runs."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MEMORY_FILE = Path(__file__).parent.parent / "memory" / "sessions.txt"
MAX_SESSIONS = 3  # how many past sessions to inject into the next prompt


def save_summary(summary: str) -> None:
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n{summary.strip()}\n")
    logger.info("Session summary saved to %s", MEMORY_FILE)


def load_recent() -> Optional[str]:
    if not MEMORY_FILE.exists():
        return None
    text = MEMORY_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return None
    sessions = re.split(r"\n(?=\[[\d-]+ [\d:]+\])", text)
    recent = [s.strip() for s in sessions if s.strip()][-MAX_SESSIONS:]
    return "\n".join(recent) if recent else None
