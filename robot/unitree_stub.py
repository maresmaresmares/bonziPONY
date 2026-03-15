"""
Unitree robot controller stub.

Logs every requested action. No hardware required.
Intended for development and testing before the real SDK is available.

Future integration notes:
  1. Install: pip install unitree_sdk2_python
     (https://github.com/unitreerobotics/unitree_sdk2_python)
  2. Create robot/unitree_go2.py subclassing RobotController
  3. Set robot.controller: "unitree_go2" in config.yaml
  4. No other code changes needed — factory pattern isolates the swap.
"""

from __future__ import annotations

import logging

from robot.actions import RobotAction
from robot.base import RobotController

logger = logging.getLogger(__name__)

# Map actions to human-readable descriptions for stub logging
_ACTION_DESCRIPTIONS = {
    RobotAction.WALK_FORWARD:  "Walking forward",
    RobotAction.WALK_BACKWARD: "Walking backward",
    RobotAction.SIT:           "Sitting down",
    RobotAction.STAND:         "Standing up",
    RobotAction.WAVE:          "Waving a foreleg",
    RobotAction.SHAKE:         "Shaking body",
    RobotAction.SPIN:          "Spinning",
}


class UnitreeStub(RobotController):
    """Stub: logs actions only, no real hardware interaction."""

    def execute(self, action: RobotAction) -> None:
        description = _ACTION_DESCRIPTIONS.get(action, str(action))
        logger.info("[ROBOT STUB] %s (%s)", description, action.name)
        print(f"[Robot] {description}")

    def shutdown(self) -> None:
        logger.info("[ROBOT STUB] Shutting down.")


def get_controller(config) -> RobotController:
    """Factory for robot controllers. Extend this when adding real hardware."""
    from core.config_loader import RobotConfig
    cfg: RobotConfig = config.robot

    controller_name = cfg.controller.lower()

    if controller_name == "stub":
        return UnitreeStub()

    # Future: elif controller_name == "unitree_go2":
    #     from robot.unitree_go2 import UnitreeGo2Controller
    #     return UnitreeGo2Controller(config)

    raise ValueError(
        f"Unknown robot controller: '{controller_name}'. "
        "Supported: stub"
    )
