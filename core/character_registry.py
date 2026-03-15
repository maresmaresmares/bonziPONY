"""Central character registry — scans Ponies/ directory at startup.

Single source of truth for mapping between directory names, display names,
and slugs for all 311+ Desktop Ponies characters.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PRESETS_DIR = Path(__file__).parent.parent / "presets"


@dataclass
class CharacterInfo:
    dir_name: str            # Exact directory name: "Changeling (Lv2) #1"
    display_name: str        # Same as dir_name (unique, human-readable)
    slug: str                # "changeling_lv2_1"
    categories: list[str]    # From pony.ini: ["non-ponies", "mares"]
    has_custom_preset: bool   # True if presets/{slug}.txt exists


# ── Module-level registry ────────────────────────────────────────────────

_characters: Dict[str, CharacterInfo] = {}   # slug → CharacterInfo
_dir_to_slug: Dict[str, str] = {}            # dir_name → slug


def slugify(name: str) -> str:
    """Convert a directory name to a slug.

    "Rainbow Dash"           → "rainbow_dash"
    "Soarin'"                → "soarin"
    "Changeling (Lv2) #1"   → "changeling_lv2_1"
    "Rarity's Father"       → "raritys_father"
    "PP Rarity"              → "pp_rarity"
    """
    s = name.lower()
    s = s.replace("'", "").replace(".", "").replace("-", " ")
    s = re.sub(r"[()#]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    return s


def _parse_categories(line: str) -> list[str]:
    """Parse a Categories line from pony.ini using CSV parsing."""
    reader = csv.reader(io.StringIO(line))
    for row in reader:
        # First field is "Categories", rest are the categories
        return [c.strip().lower() for c in row[1:] if c.strip()]
    return []


def scan_ponies(ponies_root: Path) -> None:
    """Scan all subdirectories in Ponies/ and build the registry.

    Called once at startup.
    """
    global _characters, _dir_to_slug
    _characters.clear()
    _dir_to_slug.clear()

    if not ponies_root.is_dir():
        logger.warning("Ponies directory not found: %s", ponies_root)
        return

    count = 0
    for entry in sorted(ponies_root.iterdir()):
        if not entry.is_dir():
            continue

        pony_ini = entry / "pony.ini"
        if not pony_ini.exists():
            continue

        dir_name = entry.name
        categories: list[str] = []

        try:
            with pony_ini.open("r", encoding="utf-8", errors="replace") as f:
                for i, raw_line in enumerate(f):
                    if i >= 2:
                        break
                    line = raw_line.strip().lstrip("\ufeff")
                    if line.startswith("Categories"):
                        categories = _parse_categories(line)
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", pony_ini, exc)

        slug = slugify(dir_name)
        has_preset = (_PRESETS_DIR / f"{slug}.txt").exists()

        info = CharacterInfo(
            dir_name=dir_name,
            display_name=dir_name,
            slug=slug,
            categories=categories,
            has_custom_preset=has_preset,
        )

        _characters[slug] = info
        _dir_to_slug[dir_name] = slug
        count += 1

    logger.info("Character registry: scanned %d characters (%d with custom presets)",
                count, sum(1 for c in _characters.values() if c.has_custom_preset))


def get_all_characters() -> List[CharacterInfo]:
    """Return all characters sorted alphabetically by display name."""
    return sorted(_characters.values(), key=lambda c: c.display_name.lower())


def get_character(slug: str) -> Optional[CharacterInfo]:
    """Look up a character by slug."""
    return _characters.get(slug)


def slug_to_dir_name(slug: str) -> str:
    """Return the exact directory name for a slug.

    Falls back to the old .replace("_", " ").title() if slug is not found.
    """
    info = _characters.get(slug)
    if info:
        return info.dir_name
    # Fallback for unknown slugs
    return slug.replace("_", " ").title()


def get_display_name(slug: str) -> str:
    """Return display name for a slug (same as dir_name)."""
    info = _characters.get(slug)
    if info:
        return info.display_name
    return slug.replace("_", " ").title()
