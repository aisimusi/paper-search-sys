from __future__ import annotations

"""Cross-platform helpers shared by the pipeline and backends.

The first version of scholar-daily assumed Linux (``fcntl.flock`` and
``os.killpg``). This module keeps the same semantics on Windows without
pulling in third-party dependencies:

- ``FileLock`` uses ``msvcrt.locking`` byte-range locks on Windows and
  ``fcntl.flock`` elsewhere.
- ``terminate_process`` uses ``terminate``/``kill`` on Windows and the
  process group (``killpg``) elsewhere.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Optional

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class FileLock:
    """Advisory exclusive lock held for the object's lifetime.

    ``acquire`` raises ``RuntimeError`` when another process already holds the
    lock, matching the original "已有每日任务在运行" contract.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: Optional[BinaryIO] = None

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("已有每日任务在运行") from exc
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def terminate_process(process: subprocess.Popen) -> None:
    """Terminate a subprocess, escalating to kill after a short grace period.

    On POSIX the child is expected to have been started with
    ``start_new_session=True`` so its process group id equals its pid; the
    whole group is signalled. On Windows ``start_new_session`` is ignored, so
    only the immediate process is terminated first.
    """
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        process.terminate()
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait()
