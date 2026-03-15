"""Abstract robot controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from robot.actions import RobotAction


class RobotController(ABC):
    """Base class for all robot backends."""

    @abstractmethod
    def execute(self, action: RobotAction) -> None:
        """Execute a robot action."""

    def shutdown(self) -> None:
        """Optional cleanup on shutdown."""
