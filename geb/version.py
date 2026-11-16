"""Version definition for GEB."""

from importlib.metadata import version
from pathlib import Path

__version__: str = version("GEB")


def get_git_commit() -> str | None:
    """Return the git commit of the GEB source code, if GEB runs from a git checkout.

    The commit is read directly from the ``.git`` folder next to the ``geb`` package, so neither
    the git executable nor a subprocess is needed. Supports branch checkouts (loose or packed
    refs), detached HEADs and git worktrees.

    Returns:
        The full commit hash of the checked-out GEB source, or None when GEB is not in a git
        checkout (e.g. installed with pip).

    Raises:
        ValueError: If GEB is in a git checkout but its HEAD cannot be resolved to a commit.
    """
    git_path: Path = Path(__file__).resolve().parent.parent / ".git"
    if not git_path.exists():
        return None

    if git_path.is_file():
        # worktree: ".git" is a file pointing to the worktree's git directory
        git_dir: Path = (
            git_path.parent / git_path.read_text().removeprefix("gitdir:").strip()
        )
    else:
        git_dir = git_path
    # refs are shared between worktrees and live in the common git directory
    common_dir: Path = git_dir
    if (git_dir / "commondir").is_file():
        common_dir = git_dir / (git_dir / "commondir").read_text().strip()

    head: str = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head  # detached HEAD contains the commit itself
    ref: str = head.removeprefix("ref:").strip()
    if (common_dir / ref).is_file():
        return (common_dir / ref).read_text().strip()
    if (common_dir / "packed-refs").is_file():
        for line in (common_dir / "packed-refs").read_text().splitlines():
            parts: list[str] = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    raise ValueError(
        f"Cannot resolve git HEAD ({ref}) of the GEB checkout in {git_dir}."
    )
