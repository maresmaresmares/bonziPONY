"""Read Windows clipboard history (Win+V) and current clipboard contents.

Windows 10/11 stores clipboard history via the WinRT API. We query it through
a PowerShell subprocess since Python has no first-class WinRT bindings in the
base install. Falls back to win32clipboard (current item only) if history is
unavailable or the feature is disabled.
"""

from __future__ import annotations

import logging
import subprocess
import textwrap
from typing import List

logger = logging.getLogger(__name__)

# Max characters per clipboard item before truncation
_ITEM_MAX_CHARS = 500
# Max items to return from history
_HISTORY_MAX_ITEMS = 10

# PowerShell script that reads Windows clipboard history via WinRT.
# Returns items separated by ---ITEM--- markers.
# Falls back gracefully if clipboard history is disabled or WinRT fails.
_PS_HISTORY_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$null = [System.Reflection.Assembly]::LoadWithPartialName('System.Runtime.WindowsRuntime') 2>$null
try {
    $null = [Windows.ApplicationModel.DataTransfer.Clipboard, Windows.ApplicationModel.DataTransfer, ContentType=WindowsRuntime]
    $methods = [System.WindowsRuntimeSystemExtensions].GetMethods()
    $asTask = ($methods | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -like 'IAsyncOperation*'
    })[0]
    $histType = [Windows.ApplicationModel.DataTransfer.ClipboardHistoryItemsResult, Windows.ApplicationModel.DataTransfer, ContentType=WindowsRuntime]
    $op = [Windows.ApplicationModel.DataTransfer.Clipboard]::GetHistoryItemsAsync()
    $task = $asTask.MakeGenericMethod($histType).Invoke($null, @($op))
    if (-not $task.Wait(3000)) { throw "timeout" }
    $textFmt = [Windows.ApplicationModel.DataTransfer.StandardDataFormats, Windows.ApplicationModel.DataTransfer, ContentType=WindowsRuntime]
    $textAsTask = $asTask.MakeGenericMethod([string])
    $count = 0
    $out = [System.Collections.Generic.List[string]]::new()
    foreach ($item in $task.Result.Items) {
        if ($count -ge """ + str(_HISTORY_MAX_ITEMS) + r""") { break }
        if ($item.Content.Contains($textFmt::Text)) {
            try {
                $textOp = $item.Content.GetTextAsync()
                $textTask = $textAsTask.Invoke($null, @($textOp))
                if (-not $textTask.Wait(2000)) { continue }
                $text = $textTask.Result
                if ($text -and $text.Trim()) {
                    $trimmed = $text.Trim()
                    $max = [Math]::Min($trimmed.Length, """ + str(_ITEM_MAX_CHARS) + r""")
                    $out.Add("---ITEM---")
                    $out.Add($trimmed.Substring(0, $max))
                    $count++
                }
            } catch {}
        }
    }
    $out -join "`n"
} catch {
    try { Get-Clipboard -ErrorAction SilentlyContinue } catch {}
}
"""


def _get_clipboard_win32() -> str:
    """Get current clipboard text via win32clipboard (fallback)."""
    try:
        import win32clipboard
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT) or ""
            elif win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_TEXT):
                data = win32clipboard.GetClipboardData(win32clipboard.CF_TEXT)
                return data.decode("utf-8", errors="replace") if data else ""
        finally:
            win32clipboard.CloseClipboard()
    except Exception as exc:
        logger.debug("win32clipboard fallback failed: %s", exc)
    return ""


def get_clipboard_history() -> List[str]:
    """Return clipboard history items, most recent first.

    Uses the Windows clipboard history API (Win+V feature). Returns at most
    _HISTORY_MAX_ITEMS strings. Each string is truncated to _ITEM_MAX_CHARS.
    Falls back to [current_clipboard_item] if history is unavailable.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-NoProfile", "-Command", _PS_HISTORY_SCRIPT],
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = (result.stdout or "").strip()
        if output and "---ITEM---" in output:
            parts = output.split("---ITEM---")
            items = [p.strip() for p in parts if p.strip()]
            if items:
                return items
        # PowerShell returned something but no markers — it's the fallback Get-Clipboard line
        if output:
            return [output[:_ITEM_MAX_CHARS]]
    except Exception as exc:
        logger.debug("PowerShell clipboard history failed: %s", exc)

    # Last resort: current item via win32clipboard
    current = _get_clipboard_win32()
    if current.strip():
        return [current[:_ITEM_MAX_CHARS]]
    return []


def format_clipboard_for_llm(items: List[str]) -> str:
    """Format clipboard history into a readable block for LLM context."""
    if not items:
        return "CLIPBOARD HISTORY: (empty or clipboard history disabled)"

    lines = [f"CLIPBOARD HISTORY ({len(items)} item{'s' if len(items) != 1 else ''}):"]
    for i, item in enumerate(items, 1):
        # Single-line preview label
        preview = item.replace("\n", " ").replace("\r", "")
        if len(preview) > 60:
            preview = preview[:57] + "..."
        lines.append(f"[{i}] {preview}")
        # Full content if it has newlines or is long
        if "\n" in item or len(item) > 60:
            lines.append(textwrap.indent(item, "    "))
    return "\n".join(lines)
