"""Tests for MERIT Hydro data downloading workflows."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from geb.build.workflows.merit_hydro import (
    _group_tiles_by_package,
    _package_name,
    _tile_filename,
    _tiles_for_bbox,
    download,
)


def test_tile_filename_and_package() -> None:
    """Test tile filename and package name generation.

    Verifies that tile filenames and package names are correctly
    generated for different latitude/longitude coordinates.
    """
    assert _tile_filename(30, -120, "elv") == "n30w120_elv.tif"
    assert _tile_filename(-5, 10, "dir") == "s05e010_dir.tif"
    assert _package_name(30, -120, "elv") == "elv_n30w120.tar"
    assert _package_name(59, 29, "elv") == "elv_n30e000.tar"


@pytest.mark.parametrize(
    "bbox, expected_first",
    [
        ((-121.0, -114.9, 29.9, 35.1), (30, -120)),
        ((9.1, 14.9, -5.0, 0.0), (-5, 10)),
    ],
)
def test_tiles_for_bbox(
    bbox: tuple[float, float, float, float], expected_first: tuple[int, int]
) -> None:
    """Test tile selection for bounding box.

    Verifies that tiles are correctly selected for a given
    bounding box and are properly aligned to 5-degree grid.
    """
    tiles = _tiles_for_bbox(*bbox)
    assert expected_first in tiles
    # All tiles should be aligned to 5 degrees
    for lat, lon in tiles:
        assert lat % 5 == 0 and lon % 5 == 0


def test_group_tiles_by_package() -> None:
    """Test grouping tiles by package.

    Verifies that tiles are correctly grouped into their
    corresponding 30x30-degree packages.
    """
    tiles = [(30, -120), (35, -115), (30, -115)]
    groups = _group_tiles_by_package(tiles, "elv")
    # All of these are in n30w120 package
    assert set(groups.keys()) == {"elv_n30w120.tar"}
    assert set(groups["elv_n30w120.tar"]) == set(tiles)


def test_download_handles_missing_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that download handles missing package gracefully."""

    # Mock requests
    class MockResp:
        def __init__(self, status_code: int = 200) -> None:
            self.status_code = status_code
            self.raw = MagicMock()

        def close(self) -> None:
            pass

    class MockSession:
        def head(
            self,
            url: str,
            auth: str | None = None,
            allow_redirects: bool = True,
            timeout: int = 60,
        ) -> "MockResp":
            return MockResp(status_code=404)  # package does not exist (ocean)

        def get(
            self,
            url: str,
            auth: str | None = None,
            stream: bool = True,
            timeout: int = 60,
        ) -> "MockResp":
            return MockResp(status_code=404)

    # Ensure auth env vars are present
    monkeypatch.setenv("MERIT_USERNAME", "user")
    monkeypatch.setenv("MERIT_PASSWORD", "pass")

    # Should not raise when package is missing (404) — just skip
    download(
        xmin=-121,
        xmax=-119,
        ymin=30,
        ymax=31,
        variable="elv",
        out_dir=tmp_path,
        session=MockSession(),  # type: ignore[arg-type]
        request_timeout_s=1,
    )
