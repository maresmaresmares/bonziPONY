"""Robot action enum — must match the [ACTION:XYZ] tags in the LLM prompt."""

from enum import Enum, auto


class RobotAction(Enum):
    WALK_FORWARD = auto()
    WALK_BACKWARD = auto()
    SIT = auto()
    STAND = auto()
    WAVE = auto()
    SHAKE = auto()
    SPIN = auto()
    # Desktop control actions
    CLOSE_WINDOW = auto()
    MINIMIZE_WINDOW = auto()
    MAXIMIZE_WINDOW = auto()
    SNAP_WINDOW_LEFT = auto()
    SNAP_WINDOW_RIGHT = auto()
    VOLUME_UP = auto()
    VOLUME_DOWN = auto()
    VOLUME_MUTE = auto()
    SCREENSHOT = auto()
