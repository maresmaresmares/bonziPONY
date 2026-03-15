"""Win32 multi-monitor helpers using ctypes (no extra dependencies)."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Win32 constants
MONITOR_DEFAULTTONEAREST = 2
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class MonitorRect(NamedTuple):
    left: int
    top: int
    width: int
    height: int
    right: int
    bottom: int


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.DWORD),
        ("rcMonitor", ctypes.wintypes.RECT),
        ("rcWork", ctypes.wintypes.RECT),
        ("dwFlags", ctypes.wintypes.DWORD),
    ]


def _rect_to_monitor_rect(rect: ctypes.wintypes.RECT) -> MonitorRect:
    return MonitorRect(
        left=rect.left,
        top=rect.top,
        width=rect.right - rect.left,
        height=rect.bottom - rect.top,
        right=rect.right,
        bottom=rect.bottom,
    )


def _get_monitor_info(hmon) -> tuple[MonitorRect, MonitorRect] | None:
    """Return (work_area, screen_rect) for a monitor handle, or None on failure."""
    info = _MONITORINFO()
    info.cbSize = ctypes.sizeof(_MONITORINFO)
    if ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
        return _rect_to_monitor_rect(info.rcWork), _rect_to_monitor_rect(info.rcMonitor)
    return None


def get_monitor_rect_for_point(x: int, y: int) -> MonitorRect:
    """Get the work-area rect of the monitor containing point (x, y)."""
    hmon = ctypes.windll.user32.MonitorFromPoint(
        ctypes.wintypes.POINT(x, y), MONITOR_DEFAULTTONEAREST
    )
    result = _get_monitor_info(hmon)
    if result:
        return result[0]
    # Fallback: primary monitor via GetSystemMetrics
    w = ctypes.windll.user32.GetSystemMetrics(0)
    h = ctypes.windll.user32.GetSystemMetrics(1)
    return MonitorRect(0, 0, w, h, w, h)


def get_monitor_rect_for_hwnd(hwnd: int) -> MonitorRect:
    """Get the work-area rect of the monitor the window is on."""
    hmon = ctypes.windll.user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    result = _get_monitor_info(hmon)
    if result:
        return result[0]
    w = ctypes.windll.user32.GetSystemMetrics(0)
    h = ctypes.windll.user32.GetSystemMetrics(1)
    return MonitorRect(0, 0, w, h, w, h)


def get_monitor_screen_rect_for_hwnd(hwnd: int) -> MonitorRect:
    """Get the full screen rect (not work area) of the monitor the window is on."""
    hmon = ctypes.windll.user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    result = _get_monitor_info(hmon)
    if result:
        return result[1]
    w = ctypes.windll.user32.GetSystemMetrics(0)
    h = ctypes.windll.user32.GetSystemMetrics(1)
    return MonitorRect(0, 0, w, h, w, h)


def get_virtual_desktop_rect() -> MonitorRect:
    """Get the bounding box of all monitors (virtual desktop)."""
    left = ctypes.windll.user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = ctypes.windll.user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = ctypes.windll.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = ctypes.windll.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return MonitorRect(left, top, width, height, left + width, top + height)
