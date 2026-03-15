"""Desktop automation — executes named preset actions and parameterized commands."""

from __future__ import annotations

import logging
import subprocess
import time
import webbrowser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config_loader import DesktopControlConfig
    from llm.response_parser import DesktopCommand

from robot.actions import RobotAction

logger = logging.getLogger(__name__)

# Named actions that operate on the foreground window
_WINDOW_ACTIONS = {
    RobotAction.CLOSE_WINDOW,
    RobotAction.MINIMIZE_WINDOW,
    RobotAction.MAXIMIZE_WINDOW,
    RobotAction.SNAP_WINDOW_LEFT,
    RobotAction.SNAP_WINDOW_RIGHT,
}

_VOLUME_ACTIONS = {
    RobotAction.VOLUME_UP,
    RobotAction.VOLUME_DOWN,
    RobotAction.VOLUME_MUTE,
}

# Default allowlist for OPEN command
_DEFAULT_ALLOWED_APPS = ["notepad", "calculator", "calc", "explorer", "chrome", "firefox", "mspaint"]

# Hotkeys that must never be sent
_BLOCKED_HOTKEYS = {"ctrl+alt+delete", "ctrl+alt+del"}


class DesktopController:
    """Handles both named preset actions and parameterized [DESKTOP:...] commands."""

    def __init__(self, config: DesktopControlConfig, pet_hwnd: int = 0) -> None:
        import pyautogui
        pyautogui.FAILSAFE = True  # Mouse to corner (0,0) aborts

        self._config = config
        self._pet_hwnd = pet_hwnd
        self._pyautogui = pyautogui
        self._last_command_time = 0.0
        self._cooldown = 0.5  # seconds between desktop commands

        # Build sets from config
        self._allowed_apps = set(
            app.lower() for app in (config.allowed_apps or _DEFAULT_ALLOWED_APPS)
        )
        self._blocked_hotkeys = set(
            hk.lower().replace(":", "+") for hk in (config.blocked_hotkeys or [])
        ) | _BLOCKED_HOTKEYS

        logger.info("DesktopController ready (pet_hwnd=%d).", pet_hwnd)

    def _get_foreground_hwnd(self) -> int:
        """Return the HWND of the foreground window (Windows only)."""
        try:
            import win32gui
            return win32gui.GetForegroundWindow()
        except ImportError:
            logger.warning("win32gui not available — window actions disabled.")
            return 0

    def _is_pet_window(self, hwnd: int) -> bool:
        """Check if the given HWND is the pet window itself."""
        return self._pet_hwnd != 0 and hwnd == self._pet_hwnd

    def _enforce_cooldown(self) -> None:
        """Wait if we're within cooldown period of last command."""
        elapsed = time.monotonic() - self._last_command_time
        if elapsed < self._cooldown:
            time.sleep(self._cooldown - elapsed)
        self._last_command_time = time.monotonic()

    # ── Named preset actions ────────────────────────────────────────────────

    def execute_action(self, action: RobotAction) -> None:
        """Execute a named desktop action (no parameters, operates on foreground window)."""
        if action in _WINDOW_ACTIONS:
            self._execute_window_action(action)
        elif action in _VOLUME_ACTIONS:
            self._execute_volume_action(action)
        else:
            logger.debug("DesktopController ignoring non-desktop action: %s", action)

    def _execute_window_action(self, action: RobotAction) -> None:
        try:
            import win32gui
            import win32con
        except ImportError:
            logger.warning("pywin32 not installed — window actions unavailable.")
            return

        self._enforce_cooldown()
        hwnd = self._get_foreground_hwnd()
        if hwnd == 0:
            logger.warning("No foreground window found.")
            return
        if self._is_pet_window(hwnd):
            logger.info("Skipping window action — foreground is pet window.")
            return

        try:
            if action == RobotAction.CLOSE_WINDOW:
                logger.info("Closing window HWND=%d", hwnd)
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

            elif action == RobotAction.MINIMIZE_WINDOW:
                logger.info("Minimizing window HWND=%d", hwnd)
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

            elif action == RobotAction.MAXIMIZE_WINDOW:
                logger.info("Maximizing window HWND=%d", hwnd)
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

            elif action == RobotAction.SNAP_WINDOW_LEFT:
                screen_w = self._pyautogui.size()[0]
                screen_h = self._pyautogui.size()[1]
                logger.info("Snapping window left HWND=%d", hwnd)
                win32gui.MoveWindow(hwnd, 0, 0, screen_w // 2, screen_h, True)

            elif action == RobotAction.SNAP_WINDOW_RIGHT:
                screen_w = self._pyautogui.size()[0]
                screen_h = self._pyautogui.size()[1]
                logger.info("Snapping window right HWND=%d", hwnd)
                win32gui.MoveWindow(hwnd, screen_w // 2, 0, screen_w // 2, screen_h, True)

        except Exception as exc:
            logger.warning("Window action %s failed: %s", action, exc)

    def _execute_volume_action(self, action: RobotAction) -> None:
        self._enforce_cooldown()
        try:
            if action == RobotAction.VOLUME_UP:
                logger.info("Volume up")
                self._pyautogui.press("volumeup")
            elif action == RobotAction.VOLUME_DOWN:
                logger.info("Volume down")
                self._pyautogui.press("volumedown")
            elif action == RobotAction.VOLUME_MUTE:
                logger.info("Volume mute")
                self._pyautogui.press("volumemute")
        except Exception as exc:
            logger.warning("Volume action %s failed: %s", action, exc)

    # ── Targeted window actions (by title) ─────────────────────────────────

    def close_window_by_title(self, title_substring: str) -> bool:
        """Close the first window whose title contains the substring. Returns True if found."""
        hwnd = self._find_window_by_title(title_substring)
        if hwnd is None:
            logger.info("No window found matching %r to close.", title_substring)
            return False
        if self._is_pet_window(hwnd):
            logger.info("Skipping close — matched window is pet window.")
            return False
        try:
            import win32gui
            import win32con
            logger.info("Closing window %r (HWND=%d)", title_substring, hwnd)
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        except Exception as exc:
            logger.warning("close_window_by_title failed: %s", exc)
            return False

    def minimize_window_by_title(self, title_substring: str) -> bool:
        """Minimize the first window whose title contains the substring. Returns True if found."""
        hwnd = self._find_window_by_title(title_substring)
        if hwnd is None:
            logger.info("No window found matching %r to minimize.", title_substring)
            return False
        if self._is_pet_window(hwnd):
            logger.info("Skipping minimize — matched window is pet window.")
            return False
        try:
            import win32gui
            import win32con
            logger.info("Minimizing window %r (HWND=%d)", title_substring, hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
        except Exception as exc:
            logger.warning("minimize_window_by_title failed: %s", exc)
            return False

    def minimize_all_windows(self) -> int:
        """Minimize every visible window except the pet. Returns count minimized."""
        try:
            import win32gui
            import win32con
        except ImportError:
            return 0

        minimized = 0

        def _callback(hwnd, _extra):
            nonlocal minimized
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if self._is_pet_window(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or not title.strip():
                return True
            try:
                if not win32gui.IsIconic(hwnd):  # not already minimized
                    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                    minimized += 1
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass

        logger.info("Minimized %d windows.", minimized)
        return minimized

    def _is_prominent(self, hwnd: int) -> bool:
        """Check if a window is maximized or covers a significant portion of the screen."""
        try:
            import win32gui
            if win32gui.IsZoomed(hwnd):
                return True
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            screen_w, screen_h = self._pyautogui.size()
            return w > 0 and h > 0 and (w * h) >= (screen_w * screen_h) * 0.4
        except Exception:
            return False

    def _find_window_by_title(self, title_substring: str) -> int | None:
        """Find the first visible window whose title contains the substring (case-insensitive)."""
        try:
            import win32gui
        except ImportError:
            return None

        target = title_substring.lower()
        result = [None]

        def _callback(hwnd: int, _extra) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if title and target in title.lower():
                result[0] = hwnd
                return False  # Stop enumeration
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass  # EnumWindows raises when callback returns False — that's our "found" signal

        return result[0]

    # ── Parameterized commands ──────────────────────────────────────────────

    def execute_command(self, cmd: DesktopCommand) -> None:
        """Execute a parameterized [DESKTOP:...] command."""
        self._enforce_cooldown()
        command = cmd.command.upper()

        try:
            if command == "CLICK":
                self._cmd_click(cmd.args)
            elif command == "TYPE":
                self._cmd_type(cmd.args)
            elif command == "HOTKEY":
                self._cmd_hotkey(cmd.args)
            elif command == "OPEN":
                self._cmd_open(cmd.args)
            elif command == "BROWSE":
                self._cmd_browse(cmd.args)
            elif command == "SCROLL":
                self._cmd_scroll(cmd.args)
            else:
                logger.warning("Unknown desktop command: %s", command)
        except Exception as exc:
            logger.warning("Desktop command %s failed: %s", command, exc)

    def _cmd_click(self, args: list[str]) -> None:
        if not self._config.click_enabled:
            logger.info("Click disabled by config.")
            return
        if len(args) < 2:
            logger.warning("CLICK requires x and y args.")
            return

        x, y = int(args[0]), int(args[1])
        screen_w, screen_h = self._pyautogui.size()

        # Bounds check
        x = max(0, min(x, screen_w - 1))
        y = max(0, min(y, screen_h - 1))

        logger.info("Click at (%d, %d)", x, y)
        self._pyautogui.click(x, y)

    def _cmd_type(self, args: list[str]) -> None:
        if not self._config.type_enabled:
            logger.info("Type disabled by config.")
            return
        if not args:
            logger.warning("TYPE requires text arg.")
            return

        text = ":".join(args)  # Rejoin in case text contained colons
        # 200-char limit for safety
        if len(text) > 200:
            text = text[:200]
            logger.warning("TYPE text truncated to 200 chars.")

        logger.info("Typing: %r", text)
        self._pyautogui.write(text, interval=0.02)

    def _cmd_hotkey(self, args: list[str]) -> None:
        if not args:
            logger.warning("HOTKEY requires key args.")
            return

        # Check blocklist
        combo = "+".join(a.lower() for a in args)
        if combo in self._blocked_hotkeys:
            logger.warning("Blocked hotkey: %s", combo)
            return

        logger.info("Hotkey: %s", "+".join(args))
        self._pyautogui.hotkey(*args)

    def _cmd_open(self, args: list[str]) -> None:
        if not args:
            logger.warning("OPEN requires app name arg.")
            return

        app_name = args[0].lower().strip()
        if app_name not in self._allowed_apps:
            logger.warning("App not in allowlist: %s (allowed: %s)", app_name, self._allowed_apps)
            return

        logger.info("Opening app: %s", app_name)
        try:
            subprocess.Popen(app_name, shell=True)
        except Exception as exc:
            logger.warning("Failed to open %s: %s", app_name, exc)

    def _cmd_browse(self, args: list[str]) -> None:
        if not args:
            logger.warning("BROWSE requires a URL or site name.")
            return

        raw = ":".join(args).strip()  # Rejoin in case URL contained colons

        # Figure out the URL
        if "://" in raw:
            url = raw  # Already a full URL
        elif "." in raw:
            # Looks like a domain — add https://
            url = f"https://{raw}"
        else:
            # Bare name like "youtube" or "4chan" — guess the .com domain
            url = f"https://www.{raw}.com"

        logger.info("Opening URL: %s", url)
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Failed to open URL %s: %s", url, exc)

    def _cmd_scroll(self, args: list[str]) -> None:
        if not args:
            logger.warning("SCROLL requires amount arg.")
            return

        amount = int(args[0])
        # Clamp scroll amount
        amount = max(-20, min(20, amount))

        logger.info("Scroll: %d", amount)
        self._pyautogui.scroll(amount)

    # ── Advanced actions (called by agent loop) ──────────────────────────────

    def pause_media(self) -> None:
        """Press the media play/pause key to pause/toggle media playback."""
        self._enforce_cooldown()
        try:
            logger.info("Pausing/toggling media playback")
            self._pyautogui.press("playpause")
        except Exception as exc:
            logger.warning("pause_media failed: %s", exc)

    def shake_window(self, hwnd: int = 0, duration: float = 5.0, intensity: int = 15) -> None:
        """Rapidly vibrate a window to get the user's attention — like an alarm clock."""
        try:
            import win32gui
        except ImportError:
            return

        if hwnd == 0:
            hwnd = self._get_foreground_hwnd()
        if hwnd == 0 or self._is_pet_window(hwnd):
            return
        if not self._is_prominent(hwnd):
            logger.info("Skipping shake — window HWND=%d is not maximized/prominent.", hwnd)
            return

        try:
            import win32gui
            rect = win32gui.GetWindowRect(hwnd)
            orig_x, orig_y = rect[0], rect[1]
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            import random as _rand
            end_time = time.monotonic() + duration
            while time.monotonic() < end_time:
                dx = _rand.randint(-intensity, intensity)
                dy = _rand.randint(-intensity, intensity)
                win32gui.MoveWindow(hwnd, orig_x + dx, orig_y + dy, width, height, True)
                time.sleep(0.03)

            # Restore original position
            win32gui.MoveWindow(hwnd, orig_x, orig_y, width, height, True)
            logger.info("Shook window HWND=%d for %.1fs", hwnd, duration)
        except Exception as exc:
            logger.warning("shake_window failed: %s", exc)

    def shake_all_windows(self, duration: float = 8.0, intensity: int = 12) -> None:
        """Shake maximized / prominent windows — earthquake mode for high urgency."""
        try:
            import win32gui
        except ImportError:
            return

        screen_w, screen_h = self._pyautogui.size()
        screen_area = screen_w * screen_h
        hwnds_and_rects = []

        def _callback(hwnd, _extra):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if self._is_pet_window(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or not title.strip():
                return True
            try:
                rect = win32gui.GetWindowRect(hwnd)
                # Only shake maximized or large windows (>40% of screen)
                if win32gui.IsZoomed(hwnd):
                    hwnds_and_rects.append((hwnd, rect))
                else:
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    if w > 0 and h > 0 and (w * h) >= screen_area * 0.4:
                        hwnds_and_rects.append((hwnd, rect))
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass

        if not hwnds_and_rects:
            return

        import random as _rand
        end_time = time.monotonic() + duration
        try:
            while time.monotonic() < end_time:
                dx = _rand.randint(-intensity, intensity)
                dy = _rand.randint(-intensity, intensity)
                for hwnd, rect in hwnds_and_rects:
                    try:
                        w = rect[2] - rect[0]
                        h = rect[3] - rect[1]
                        win32gui.MoveWindow(hwnd, rect[0] + dx, rect[1] + dy, w, h, True)
                    except Exception:
                        pass
                time.sleep(0.03)

            # Restore all
            for hwnd, rect in hwnds_and_rects:
                try:
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    win32gui.MoveWindow(hwnd, rect[0], rect[1], w, h, True)
                except Exception:
                    pass
            logger.info("Shook %d windows for %.1fs", len(hwnds_and_rects), duration)
        except Exception as exc:
            logger.warning("shake_all_windows failed: %s", exc)

    def mess_with_mouse(self, duration: float = 6.0, jitter: int = 80) -> None:
        """Jitter the mouse around chaotically — for high urgency nagging."""
        import random as _rand
        try:
            start_x, start_y = self._pyautogui.position()
            screen_w, screen_h = self._pyautogui.size()
            end_time = time.monotonic() + duration

            logger.info("Messing with mouse for %.1fs", duration)
            while time.monotonic() < end_time:
                dx = _rand.randint(-jitter, jitter)
                dy = _rand.randint(-jitter, jitter)
                new_x = max(5, min(screen_w - 5, start_x + dx))
                new_y = max(5, min(screen_h - 5, start_y + dy))
                self._pyautogui.moveTo(new_x, new_y, duration=0.05)
                time.sleep(0.05)

            # Return mouse to roughly where it was
            self._pyautogui.moveTo(start_x, start_y, duration=0.1)
        except Exception as exc:
            logger.warning("mess_with_mouse failed: %s", exc)

    def shake_window_by_title(self, title_substring: str, duration: float = 5.0, intensity: int = 15) -> bool:
        """Shake the first window matching the title. Returns True if found and prominent."""
        hwnd = self._find_window_by_title(title_substring)
        if hwnd is None:
            return False
        if self._is_pet_window(hwnd):
            return False
        if not self._is_prominent(hwnd):
            logger.info("Skipping shake — window %r is not maximized/prominent.", title_substring)
            return False
        self.shake_window(hwnd=hwnd, duration=duration, intensity=intensity)
        return True
