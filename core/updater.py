"""Self-updater — checks GitHub for new commits and pulls updates."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/maresmaresmares/bonziPONY.git"
REPO_API = "https://api.github.com/repos/maresmaresmares/bonziPONY"


def _git(*args: str, cwd: Optional[str] = None) -> Tuple[int, str]:
    """Run a git command and return (returncode, stdout)."""
    repo_root = cwd or str(Path(__file__).resolve().parent.parent)
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout.strip()
    except FileNotFoundError:
        return -1, "git not found"
    except subprocess.TimeoutExpired:
        return -1, "git command timed out"
    except Exception as e:
        return -1, str(e)


def get_local_head() -> Optional[str]:
    """Get the local HEAD commit hash."""
    rc, out = _git("rev-parse", "HEAD")
    return out if rc == 0 else None


def get_remote_head() -> Optional[str]:
    """Get the remote HEAD commit hash via ls-remote (no fetch needed)."""
    rc, out = _git("ls-remote", "origin", "HEAD")
    if rc == 0 and out:
        # Format: "<hash>\tHEAD"
        return out.split()[0]
    return None


def check_for_updates() -> Tuple[bool, str, List[str]]:
    """Check if updates are available.

    Returns:
        (has_updates, status_message, new_commit_summaries)
    """
    local = get_local_head()
    if not local:
        return False, "Could not determine local version (not a git repo?).", []

    remote = get_remote_head()
    if not remote:
        return False, "Could not reach GitHub. Check your internet connection.", []

    if local == remote:
        return False, "You're already on the latest version!", []

    # Fetch to get the new commits locally so we can list them
    rc, _ = _git("fetch", "origin", "master")
    if rc != 0:
        # Still report that updates are available even if fetch fails
        return True, "Updates available, but fetch failed. Try again.", []

    # Get list of new commits
    rc, log_out = _git("log", "--oneline", f"{local}..origin/master")
    commits = log_out.splitlines() if rc == 0 and log_out else []

    count = len(commits)
    msg = f"{count} new update{'s' if count != 1 else ''} available!"
    return True, msg, commits


def pull_updates() -> Tuple[bool, str]:
    """Pull latest changes from GitHub.

    Returns:
        (success, message)
    """
    # Stash any local changes to config/data files
    rc, status = _git("status", "--porcelain")
    has_local_changes = rc == 0 and bool(status)

    if has_local_changes:
        _git("stash", "push", "-m", "auto-stash before update")

    # Pull from origin
    rc, out = _git("pull", "origin", "master", "--ff-only")
    if rc != 0:
        # Try rebase as fallback
        rc, out = _git("pull", "origin", "master", "--rebase")
        if rc != 0:
            if has_local_changes:
                _git("stash", "pop")
            return False, f"Update failed:\n{out}\n\nYou may need to update manually."

    # Pop stashed changes
    if has_local_changes:
        rc_pop, pop_out = _git("stash", "pop")
        if rc_pop != 0:
            logger.warning("Stash pop failed: %s", pop_out)

    return True, "Update successful!"


def install_new_requirements() -> Tuple[bool, str]:
    """Run pip install -r requirements.txt in case dependencies changed."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    req_file = os.path.join(repo_root, "requirements.txt")
    if not os.path.exists(req_file):
        return True, "No requirements.txt found."

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return True, "Dependencies updated."
        return False, f"pip install failed:\n{result.stderr[:500]}"
    except Exception as e:
        return False, f"Could not install dependencies: {e}"


def restart_application() -> None:
    """Restart the application by re-launching main.py."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    main_py = os.path.join(repo_root, "main.py")

    # Build the command to relaunch
    args = [sys.executable, main_py] + sys.argv[1:]

    logger.info("Restarting application: %s", " ".join(args))

    # Use Popen to start the new process, then exit the current one.
    # CREATE_NEW_PROCESS_GROUP so the new process survives our exit.
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(args, cwd=repo_root, **kwargs)

    # Exit current process
    os._exit(0)
