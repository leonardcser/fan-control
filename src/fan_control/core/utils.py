"""Utility functions for fan control."""

import os
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple


def get_original_user() -> Optional[Tuple[int, int]]:
    """
    Get the UID and GID of the user who invoked sudo.

    Returns:
        Tuple of (uid, gid) if running under sudo, None otherwise
    """
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")

    if sudo_uid and sudo_gid:
        return (int(sudo_uid), int(sudo_gid))
    return None


@contextmanager
def drop_privileges() -> Iterator[None]:
    """
    Context manager to temporarily drop root privileges to the original user.

    Use this when creating files/directories to ensure they're owned by the
    user who invoked sudo, not root.

    Example:
        with drop_privileges():
            Path("output.csv").write_text("data")
    """
    user_info = get_original_user()

    if user_info is None:
        # Not running under sudo, nothing to do
        yield
        return

    uid, gid = user_info

    # Save current effective IDs
    saved_euid = os.geteuid()
    saved_egid = os.getegid()

    try:
        # Drop privileges (order matters: group first, then user)
        os.setegid(gid)
        os.seteuid(uid)
        yield
    finally:
        # Restore root privileges (reverse order: user first, then group)
        os.seteuid(saved_euid)
        os.setegid(saved_egid)
