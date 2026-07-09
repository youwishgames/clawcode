"""Per-session read/staleness registry shared by every file-mutating tool.

Why this lives in the tool layer and not the agent loop
------------------------------------------------------
The guard originally sat in ``Agent._collect_single_tool_events``, which
covers calls the agent loop dispatches — but NOT callers that invoke the
tools directly. ``execute_code`` does exactly that (its sandbox
``write_file`` / ``patch`` helpers call ``WriteTool.run`` under the hood),
so it drove straight past the checkpoint. Placing the registry here means
every caller inherits it: the agent loop, ``execute_code``, and anything
added later.

The contract (mirrors Claude Code's Edit/Write semantics):

* Modifying an EXISTING file requires that the session has read it
  (``view`` / ``batch_view``, or written it) and that it has not changed on
  disk since. Creating a new file is exempt.
* Successful writes re-register the file, so an agent's own consecutive
  edits pass.
* ``bash`` remains an unguarded path by design — it is permission-gated
  per call instead (execute-kind requests never get session-scoped grants).

Escape hatch: ``settings.require_read_before_edit = false``.
"""

from __future__ import annotations

import threading
from pathlib import Path

#: "session_id|abs_path" -> mtime observed when the file was last read
#: or written by this session. Module-level: tools are constructed per
#: runtime bundle, but sessions outlive individual tool instances.
_READ_STATE: dict[str, float] = {}
_LOCK = threading.Lock()


def _key(session_id: str, path: Path) -> str:
    return f"{session_id}|{path}"


def _current_mtime(path: Path) -> float | None:
    try:
        if not path.is_file():
            return None
        return path.stat().st_mtime
    except OSError:
        return None


def require_read_before_edit() -> bool:
    """Whether the guard is enabled (settings-driven, fail-safe to True)."""
    try:
        from ...config import get_settings

        return bool(getattr(get_settings(), "require_read_before_edit", True))
    except Exception:
        return True


def record_read(session_id: str, path: Path) -> None:
    """Register a file the session just read (or wrote) with its mtime."""
    mtime = _current_mtime(path)
    if mtime is None:
        return
    with _LOCK:
        _READ_STATE[_key(session_id, path)] = mtime


def check_editable(session_id: str, path: Path) -> str | None:
    """Return an error message if this file may not be modified, else None.

    Creating a new file is always allowed. Modifying an existing one
    requires a prior read this session, unchanged on disk since.
    """
    if not require_read_before_edit():
        return None

    mtime = _current_mtime(path)
    if mtime is None:
        # New file (or unstattable) — creation is exempt.
        return None

    with _LOCK:
        seen = _READ_STATE.get(_key(session_id, path))

    if seen is None:
        return (
            f"Error: {path} has not been read this session. "
            "Use the view tool to read it before modifying it."
        )
    if seen != mtime:
        return (
            f"Error: {path} changed on disk since it was last read. "
            "Re-read it with the view tool before modifying it."
        )
    return None


def forget_session(session_id: str) -> None:
    """Drop a session's registry (called when a session ends)."""
    prefix = f"{session_id}|"
    with _LOCK:
        for k in [k for k in _READ_STATE if k.startswith(prefix)]:
            del _READ_STATE[k]


def _reset_for_tests() -> None:
    with _LOCK:
        _READ_STATE.clear()
