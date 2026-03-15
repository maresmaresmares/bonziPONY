"""Handles rendering visual effects (Sonic Rainboom) as overlay sprites."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from PyQt5.QtGui import QPixmap

from desktop_pet.behavior_manager import BehaviorManager, EffectDef
from desktop_pet.sprite_manager import SpriteAnimation, SpriteManager

logger = logging.getLogger(__name__)


@dataclass
class ActiveEffect:
    """A currently-playing effect instance."""

    effect_def: EffectDef
    animation: SpriteAnimation
    frame_index: int = 0
    x: int = 0
    y: int = 0
    start_time: float = 0.0
    last_frame_time: float = 0.0
    origin_x: int = 0  # Where the effect was spawned
    origin_y: int = 0
    facing_right: bool = True


class EffectRenderer:
    """Manages active visual effects that overlay the main sprite."""

    def __init__(self, sprite_manager: SpriteManager, behavior_manager: BehaviorManager) -> None:
        self.sprite_manager = sprite_manager
        self.behavior_manager = behavior_manager
        self.active_effects: List[ActiveEffect] = []

    def trigger_effects(
        self,
        behavior_name: str,
        facing_right: bool,
        sprite_x: int,
        sprite_y: int,
        sprite_w: int,
        sprite_h: int,
    ) -> None:
        """Start all effects associated with a behavior."""
        effect_defs = self.behavior_manager.get_effects_for(behavior_name)
        if not effect_defs:
            return

        now = time.monotonic()
        for edef in effect_defs:
            # Load the effect animation
            img_file = edef.right_image if facing_right else edef.left_image
            anim_key = f"effect_{edef.name}_{'right' if facing_right else 'left'}"
            anim = self.sprite_manager.load_animation(anim_key, img_file)
            if not anim.frames:
                continue

            # Calculate initial position based on placement
            ex, ey = self._calc_position(
                edef, facing_right, sprite_x, sprite_y, sprite_w, sprite_h, anim
            )

            effect = ActiveEffect(
                effect_def=edef,
                animation=anim,
                frame_index=0,
                x=ex,
                y=ey,
                start_time=now + edef.delay,
                last_frame_time=now + edef.delay,
                origin_x=sprite_x + sprite_w // 2,
                origin_y=sprite_y + sprite_h // 2,
                facing_right=facing_right,
            )
            self.active_effects.append(effect)
            logger.debug("Triggered effect '%s' for behavior '%s'", edef.name, behavior_name)

    def tick(
        self,
        sprite_x: int,
        sprite_y: int,
        sprite_w: int,
        sprite_h: int,
    ) -> List[Tuple[QPixmap, int, int]]:
        """Update effects and return list of (pixmap, x, y) to render."""
        now = time.monotonic()
        results: List[Tuple[QPixmap, int, int]] = []
        still_active: List[ActiveEffect] = []

        for eff in self.active_effects:
            # Skip if delay hasn't elapsed yet
            if now < eff.start_time:
                still_active.append(eff)
                continue

            # Check if duration expired (0 = infinite, lasts until cleared)
            if eff.effect_def.duration > 0:
                elapsed = now - eff.start_time
                if elapsed > eff.effect_def.duration:
                    continue  # Expired, don't keep

            anim = eff.animation
            if not anim.frames:
                continue

            # Advance frame based on delay
            frame_delay = anim.delays[eff.frame_index] / 1000.0
            if now - eff.last_frame_time >= frame_delay:
                eff.frame_index = (eff.frame_index + 1) % len(anim.frames)
                eff.last_frame_time = now

            # Update position if the effect follows the sprite
            if eff.effect_def.follow:
                eff.x, eff.y = self._calc_position(
                    eff.effect_def,
                    eff.facing_right,
                    sprite_x,
                    sprite_y,
                    sprite_w,
                    sprite_h,
                    anim,
                )

            pixmap = anim.frames[eff.frame_index]
            results.append((pixmap, eff.x, eff.y))
            still_active.append(eff)

        self.active_effects = still_active
        return results

    def clear(self) -> None:
        """Remove all active effects."""
        self.active_effects.clear()

    def _calc_position(
        self,
        edef: EffectDef,
        facing_right: bool,
        sprite_x: int,
        sprite_y: int,
        sprite_w: int,
        sprite_h: int,
        anim: SpriteAnimation,
    ) -> Tuple[int, int]:
        """Calculate effect position based on placement and centering rules."""
        if not anim.frames:
            return sprite_x, sprite_y

        eff_w = anim.frames[0].width()
        eff_h = anim.frames[0].height()

        placement = edef.right_placement if facing_right else edef.left_placement
        centering = edef.right_centering if facing_right else edef.left_centering

        # Placement: where on the sprite the effect anchors
        if placement == "Center":
            anchor_x = sprite_x + sprite_w // 2
            anchor_y = sprite_y + sprite_h // 2
        elif placement == "Right":
            anchor_x = sprite_x + sprite_w
            anchor_y = sprite_y + sprite_h // 2
        elif placement == "Left":
            anchor_x = sprite_x
            anchor_y = sprite_y + sprite_h // 2
        elif placement == "Top":
            anchor_x = sprite_x + sprite_w // 2
            anchor_y = sprite_y
        elif placement == "Bottom":
            anchor_x = sprite_x + sprite_w // 2
            anchor_y = sprite_y + sprite_h
        else:
            anchor_x = sprite_x + sprite_w // 2
            anchor_y = sprite_y + sprite_h // 2

        # Centering: which part of the effect aligns to the anchor
        if centering == "Center":
            ex = anchor_x - eff_w // 2
            ey = anchor_y - eff_h // 2
        elif centering == "Right":
            ex = anchor_x - eff_w
            ey = anchor_y - eff_h // 2
        elif centering == "Left":
            ex = anchor_x
            ey = anchor_y - eff_h // 2
        elif centering == "Top":
            ex = anchor_x - eff_w // 2
            ey = anchor_y
        elif centering == "Bottom":
            ex = anchor_x - eff_w // 2
            ey = anchor_y - eff_h
        else:
            ex = anchor_x - eff_w // 2
            ey = anchor_y - eff_h // 2

        return ex, ey
