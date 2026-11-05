"""Tests for GEB version information."""

from pathlib import Path

import pytest

import geb.version
from geb.version import get_git_commit

COMMIT: str = "0123456789abcdef0123456789abcdef01234567"


def _fake_package(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make get_git_commit look for ``.git`` next to a fake ``geb`` package in ``root``.

    Args:
        root: Directory that plays the role of the GEB source checkout.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The ``root`` directory.
    """
    (root / "geb").mkdir(parents=True)
    monkeypatch.setattr(geb.version, "__file__", str(root / "geb" / "version.py"))
    return root


def test_get_git_commit_real_checkout() -> None:
    """In a git checkout of GEB, the full commit hash is returned."""
    if not (Path(geb.version.__file__).resolve().parent.parent / ".git").exists():
        pytest.skip("GEB is not installed from a git checkout")
    commit: str | None = get_git_commit()
    assert commit is not None and len(commit) == 40


def test_get_git_commit_loose_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch checkout with a loose ref file."""
    root: Path = _fake_package(tmp_path, monkeypatch)
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    assert get_git_commit() == COMMIT


def test_get_git_commit_packed_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch whose ref only exists in packed-refs."""
    root: Path = _fake_package(tmp_path, monkeypatch)
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{COMMIT} refs/heads/main\n"
    )
    assert get_git_commit() == COMMIT


def test_get_git_commit_detached_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detached HEAD contains the commit itself."""
    root: Path = _fake_package(tmp_path, monkeypatch)
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text(COMMIT + "\n")
    assert get_git_commit() == COMMIT


def test_get_git_commit_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git worktree: ``.git`` is a file and refs live in the common directory."""
    main_git: Path = tmp_path / "main" / ".git"
    (main_git / "refs" / "heads").mkdir(parents=True)
    (main_git / "refs" / "heads" / "feature").write_text(COMMIT + "\n")
    worktree_git: Path = main_git / "worktrees" / "wt"
    worktree_git.mkdir(parents=True)
    (worktree_git / "HEAD").write_text("ref: refs/heads/feature\n")
    (worktree_git / "commondir").write_text("../..\n")
    root: Path = _fake_package(tmp_path / "wt", monkeypatch)
    (root / ".git").write_text(f"gitdir: {worktree_git}\n")
    assert get_git_commit() == COMMIT


def test_get_git_commit_not_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``.git`` next to the package (e.g. installed with pip): None, no error."""
    _fake_package(tmp_path, monkeypatch)
    assert get_git_commit() is None


def test_get_git_commit_unreadable_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ref that cannot be resolved from plain files (e.g. reftable): None, no error."""
    root: Path = _fake_package(tmp_path, monkeypatch)
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/.invalid\n")
    assert get_git_commit() is None
