"""Shared AVH checkpoint path resolution for subprocess/sandbox compatibility."""

from __future__ import annotations

import os
import shutil
import tempfile


def get_readable_ckpt_path(
    path: str,
    tmp_name: str = "AVH-Align_AV1M.pt",
    *,
    force_tmp: bool = False,
) -> str:
    """
    Return a path readable by subprocesses.

    If reading the original raises PermissionError, copy under /tmp.
    If force_tmp is True, always mirror to /tmp (avoids sandbox permission issues).
    """
    tmp = os.path.join(tempfile.gettempdir(), tmp_name)
    try:
        with open(path, "rb") as f:
            f.read(1)
    except PermissionError:
        shutil.copy2(path, tmp)
        return tmp

    if force_tmp:
        if not os.path.isfile(tmp) or os.path.getsize(tmp) != os.path.getsize(path):
            shutil.copy2(path, tmp)
        return tmp

    return path
