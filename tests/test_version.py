"""Tests for GEB version information."""

import shutil
import subprocess
from pathlib import Path

import pytest

import geb.version
from geb.version import get_git_commit


def test_get_git_commit_matches_git() -> None:
    """In a git checkout, the commit read from .git equals `git rev-parse HEAD`."""
    repository: Path = Path(geb.version.__file__).resolve().parent.parent
    if not (repository / ".git").exists() or shutil.which("git") is None:
        pytest.skip("GEB is not installed from a git checkout")
    expected: str = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True
    ).stdout.strip()
    assert get_git_commit() == expected


def test_get_git_commit_not_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without .git next to the package (e.g. installed with pip), None is returned."""
    (tmp_path / "geb").mkdir()
    monkeypatch.setattr(geb.version, "__file__", str(tmp_path / "geb" / "version.py"))
    assert get_git_commit() is None


def test_get_git_commit_packed_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch whose ref only exists in packed-refs (git packs refs over time)."""
    commit: str = "0123456789abcdef0123456789abcdef01234567"
    (tmp_path / "geb").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "packed-refs").write_text(f"{commit} refs/heads/main\n")
    monkeypatch.setattr(geb.version, "__file__", str(tmp_path / "geb" / "version.py"))
    assert get_git_commit() == commit
