"""Parses pony.ini behavior/effect definitions and manages behavior selection."""

from __future__ import annotations

import csv
import io
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MovementType(Enum):
    NONE = "None"
    HORIZONTAL_ONLY = "Horizontal_Only"
    VERTICAL_ONLY = "Vertical_Only"
    DIAGONAL_VERTICAL = "Diagonal_Vertical"
    DIAGONAL_HORIZONTAL = "Diagonal_horizontal"
    ALL = "All"
    MOUSEOVER = "MouseOver"
    SLEEP = "Sleep"
    DRAGGED = "Dragged"


@dataclass
class BehaviorDef:
    name: str
    probability: float
    max_duration: float  # seconds
    min_duration: float  # seconds
    speed: float  # pixels per tick
    right_image: str
    left_image: str
    movement: MovementType
    linked_behavior: str = ""
    start_speech: str = ""
    end_speech: str = ""
    skip_normally: bool = False
    follow_target: str = ""
    effects: List[str] = field(default_factory=list)


@dataclass
class EffectDef:
    name: str
    behavior_name: str
    right_image: str
    left_image: str
    duration: float  # seconds, 0 = until behavior ends
    delay: float  # seconds before effect starts
    right_placement: str = "Center"
    right_centering: str = "Center"
    left_placement: str = "Center"
    left_centering: str = "Center"
    follow: bool = False
    dont_repeat: bool = False


def _parse_csv_line(line: str) -> list[str]:
    """Parse a single CSV line handling quoted fields with commas."""
    reader = csv.reader(io.StringIO(line))
    for row in reader:
        return row
    return []


def _parse_movement(value: str) -> MovementType:
    """Parse movement type string to enum, with fallback."""
    value = value.strip()
    for mt in MovementType:
        if mt.value.lower() == value.lower():
            return mt
    return MovementType.NONE


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return default


class BehaviorManager:
    """Parses pony.ini and manages weighted random behavior selection."""

    def __init__(self, pony_ini_path: Path) -> None:
        self.pony_ini_path = pony_ini_path
        self.behaviors: Dict[str, BehaviorDef] = {}
        self.effects: Dict[str, EffectDef] = {}
        self._selectable: List[BehaviorDef] = []  # non-skip behaviors with probability > 0

    def parse(self) -> None:
        """Parse the pony.ini file."""
        if not self.pony_ini_path.exists():
            logger.error("pony.ini not found: %s", self.pony_ini_path)
            return

        # Map behavior names to their effects (populated after parsing effects)
        behavior_effects: Dict[str, List[str]] = {}

        with self.pony_ini_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                fields = _parse_csv_line(line)
                if not fields:
                    continue

                record_type = fields[0].strip()

                if record_type == "Behavior" and len(fields) >= 9:
                    self._parse_behavior(fields)
                elif record_type == "Effect" and len(fields) >= 12:
                    self._parse_effect(fields, behavior_effects)

        # Assign effects to behaviors
        for beh_name, effect_names in behavior_effects.items():
            if beh_name in self.behaviors:
                self.behaviors[beh_name].effects = effect_names

        # Build selectable list (non-skip, probability > 0)
        self._selectable = [
            b for b in self.behaviors.values()
            if not b.skip_normally and b.probability > 0
        ]

        logger.info(
            "Parsed %d behaviors (%d selectable), %d effects",
            len(self.behaviors),
            len(self._selectable),
            len(self.effects),
        )

    def _parse_behavior(self, fields: list[str]) -> None:
        """Parse a Behavior line from pony.ini."""
        try:
            name = fields[1].strip()
            beh = BehaviorDef(
                name=name,
                probability=_parse_float(fields[2]),
                max_duration=_parse_float(fields[3]),
                min_duration=_parse_float(fields[4]),
                speed=_parse_float(fields[5]),
                right_image=fields[6].strip(),
                left_image=fields[7].strip(),
                movement=_parse_movement(fields[8]),
                linked_behavior=fields[9].strip() if len(fields) > 9 else "",
                start_speech=fields[10].strip() if len(fields) > 10 else "",
                end_speech=fields[11].strip() if len(fields) > 11 else "",
                skip_normally=_parse_bool(fields[12]) if len(fields) > 12 else False,
                follow_target=fields[15].strip() if len(fields) > 15 else "",
            )
            self.behaviors[name] = beh
        except Exception as exc:
            logger.debug("Failed to parse behavior line: %s", exc)

    def _parse_effect(self, fields: list[str], behavior_effects: Dict[str, List[str]]) -> None:
        """Parse an Effect line from pony.ini."""
        try:
            name = fields[1].strip()
            behavior_name = fields[2].strip()
            eff = EffectDef(
                name=name,
                behavior_name=behavior_name,
                right_image=fields[3].strip(),
                left_image=fields[4].strip(),
                duration=_parse_float(fields[5]),
                delay=_parse_float(fields[6]),
                right_placement=fields[7].strip() if len(fields) > 7 else "Center",
                right_centering=fields[8].strip() if len(fields) > 8 else "Center",
                left_placement=fields[9].strip() if len(fields) > 9 else "Center",
                left_centering=fields[10].strip() if len(fields) > 10 else "Center",
                follow=_parse_bool(fields[11]) if len(fields) > 11 else False,
                dont_repeat=_parse_bool(fields[12]) if len(fields) > 12 else False,
            )
            self.effects[name] = eff

            # Track which behavior triggers this effect
            if behavior_name not in behavior_effects:
                behavior_effects[behavior_name] = []
            behavior_effects[behavior_name].append(name)
        except Exception as exc:
            logger.debug("Failed to parse effect line: %s", exc)

    def pick_behavior(self) -> BehaviorDef:
        """Pick a random behavior weighted by probability."""
        if not self._selectable:
            # Fallback: return a basic stand behavior
            return BehaviorDef(
                name="stand",
                probability=1.0,
                max_duration=10,
                min_duration=5,
                speed=0,
                right_image="",
                left_image="",
                movement=MovementType.NONE,
            )

        weights = [b.probability for b in self._selectable]
        return random.choices(self._selectable, weights=weights, k=1)[0]

    def get_behavior(self, name: str) -> Optional[BehaviorDef]:
        """Get a behavior by name."""
        return self.behaviors.get(name)

    def get_effects_for(self, behavior_name: str) -> list[EffectDef]:
        """Get all effects triggered by a behavior."""
        return [e for e in self.effects.values() if e.behavior_name == behavior_name]

    def get_linked(self, behavior: BehaviorDef) -> Optional[BehaviorDef]:
        """Get the linked behavior if one exists."""
        if behavior.linked_behavior:
            return self.behaviors.get(behavior.linked_behavior)
        return None
