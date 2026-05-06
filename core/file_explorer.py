"""File system tree walker for the agent's EXPLORE_FILES command.

Produces a compact, LLM-readable tree using numbered folder references:
  [N]    = folder #N (path shown relative to root)
  [>N]   = item inside folder N (file or empty marker)
  [>N][M]= sub-folder M which lives inside folder N

Example output:
  FILE TREE: C:\\Users\\alice  (depth 2, 14 entries)
  [1] Documents/
  [>1] notes.txt
  [>1][2] projects/
  [>2] main.py
  [>2] utils.py
  [3] Desktop/
  [>3] game.lnk
  [4] Downloads/  (contents not shown — depth limit)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Folders that are almost always noise
_SKIP_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", ".eggs",
    "$RECYCLE.BIN", "System Volume Information", "WindowsApps",
}

# Extensions to hide from the listing (system junk)
_SKIP_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".sys", ".lnk"}

_MAX_ENTRIES = 120  # hard cap to avoid flooding context


def explore(path: str, depth: int = 2) -> str:
    """Walk *path* up to *depth* levels deep and return the tree string."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"EXPLORE_FILES: path not found — {path}"
    if root.is_file():
        return f"EXPLORE_FILES: {root.name} is a file, not a directory"

    lines: list[str] = []
    folder_counter = [0]          # mutable counter shared across recursion
    entry_count = [0]             # total entries emitted

    def _walk(directory: Path, parent_num: Optional[int], current_depth: int) -> None:
        if entry_count[0] >= _MAX_ENTRIES:
            return

        # Sort: dirs first, then files, both alphabetically (case-insensitive)
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            prefix = f"[>{parent_num}]" if parent_num is not None else ""
            lines.append(f"{prefix} (permission denied)")
            entry_count[0] += 1
            return

        for entry in entries:
            if entry_count[0] >= _MAX_ENTRIES:
                lines.append("  ... (truncated — too many entries)")
                return

            name = entry.name

            # Skip hidden files/dirs (dot-prefixed on Unix, hidden on Windows)
            if name.startswith("."):
                continue

            if entry.is_dir():
                if name in _SKIP_DIRS:
                    continue
                folder_counter[0] += 1
                this_num = folder_counter[0]

                parent_prefix = f"[>{parent_num}]" if parent_num is not None else ""
                lines.append(f"{parent_prefix}[{this_num}] {name}/")
                entry_count[0] += 1

                if current_depth < depth:
                    _walk(entry, this_num, current_depth + 1)
                else:
                    # Depth limit — note contents hidden
                    lines.append(f"[>{this_num}] (depth limit — not expanded)")
                    entry_count[0] += 1

            elif entry.is_file():
                if entry.suffix.lower() in _SKIP_EXTS:
                    continue
                parent_prefix = f"[>{parent_num}]" if parent_num is not None else ""
                lines.append(f"{parent_prefix} {name}")
                entry_count[0] += 1

    _walk(root, None, 1)

    header = f"FILE TREE: {root}  (depth {depth}, {entry_count[0]} entries)"
    if not lines:
        return f"{header}\n  (empty directory)"
    return header + "\n" + "\n".join(lines)


# ── File reader ───────────────────────────────────────────────────────────────

_READ_MAX_CHARS = 8_000
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm", ".css",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".sh", ".bat", ".ps1", ".c", ".cpp", ".h", ".java", ".go", ".rs",
    ".xml", ".svg", ".sql", ".r", ".rb", ".php",
}


def read_file(path: str) -> str:
    """Read a text file and return its contents, truncated if large.

    Returns a formatted string suitable for LLM context injection.
    Refuses binary files, enforces an 8 000-char cap.
    """
    target = Path(path).expanduser().resolve()

    if not target.exists():
        return f"READ_FILE: not found — {path}"
    if target.is_dir():
        return f"READ_FILE: {target.name} is a directory — use EXPLORE_FILES"
    if target.stat().st_size == 0:
        return f"READ_FILE: {target.name} is empty"
    if target.stat().st_size > 2_000_000:
        return f"READ_FILE: {target.name} is too large ({target.stat().st_size // 1024} KB) — refusing"

    # Warn (but still try) if extension isn't in the known-text list
    ext_note = ""
    if target.suffix.lower() not in _TEXT_EXTENSIONS and target.suffix:
        ext_note = f" (unusual extension '{target.suffix}' — may be garbled)"

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = target.read_text(encoding=encoding, errors="replace")
            break
        except Exception:
            continue
    else:
        return f"READ_FILE: could not decode {target.name}"

    truncated = ""
    if len(text) > _READ_MAX_CHARS:
        text = text[:_READ_MAX_CHARS]
        truncated = f"\n... (truncated at {_READ_MAX_CHARS} chars)"

    return f"FILE CONTENTS: {target}{ext_note}\n{text}{truncated}"
