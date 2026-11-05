"""Version definition for GEB."""

from importlib.metadata import version
from pathlib import Path

__version__: str = version("GEB")


def get_git_commit() -> str | None:
    """Return the git commit of the GEB source code, if GEB runs from a git checkout.

    The commit is read directly from the ``.git`` folder next to the ``geb`` package, so neither
    the git executable nor a subprocess is needed.

    Notes:
        Returns None when GEB is not in a git checkout (e.g. installed with pip), and for ref
        formats that are not plain files (e.g. the optional reftable format), so that this never
        fails. Supports branch checkouts (loose or packed refs), detached HEADs and git worktrees.

    Returns:
        The full commit hash of the checked-out GEB source, or None if unavailable.
    """
    git_path: Path = Path(__file__).resolve().parent.parent / ".git"
    try:
        if git_path.is_file():
            # worktree: ".git" is a file pointing to the worktree's git directory
            git_dir: Path = (
                git_path.parent / git_path.read_text().split("gitdir:", 1)[1].strip()
            )
        elif git_path.is_dir():
            git_dir = git_path
        else:
            return None
        # refs are shared between worktrees and live in the common git directory
        common_dir: Path = git_dir
        if (git_dir / "commondir").is_file():
            common_dir = git_dir / (git_dir / "commondir").read_text().strip()

        head: str = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head if _is_commit_hash(head) else None  # detached HEAD
        ref: str = head.split(":", 1)[1].strip()
        if (common_dir / ref).is_file():
            commit: str = (common_dir / ref).read_text().strip()
            return commit if _is_commit_hash(commit) else None
        if (common_dir / "packed-refs").is_file():
            for line in (common_dir / "packed-refs").read_text().splitlines():
                parts: list[str] = line.split()
                if len(parts) == 2 and parts[1] == ref and _is_commit_hash(parts[0]):
                    return parts[0]
    except OSError, IndexError, UnicodeDecodeError:
        return None
    return None


def _is_commit_hash(value: str) -> bool:
    """Check whether a string is a full git commit hash (SHA-1 or SHA-256).

    Args:
        value: The string to check.

    Returns:
        True if the string is 40 or 64 hexadecimal characters.
    """
    return len(value) in (40, 64) and all(c in "0123456789abcdef" for c in value)
