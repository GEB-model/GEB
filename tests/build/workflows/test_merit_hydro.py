from unittest.mock import MagicMock

import pytest

from geb.build.workflows.merit_hydro import (
    _group_tiles_by_package,
    _package_name,
    _tile_filename,
    _tiles_for_bbox,
    download,
)


def test_tile_filename_and_package():
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
def test_tiles_for_bbox(bbox, expected_first):
    tiles = _tiles_for_bbox(*bbox)
    assert expected_first in tiles
    # All tiles should be aligned to 5 degrees
    for lat, lon in tiles:
        assert lat % 5 == 0 and lon % 5 == 0


def test_group_tiles_by_package():
    tiles = [(30, -120), (35, -115), (30, -115)]
    groups = _group_tiles_by_package(tiles, "elv")
    # All of these are in n30w120 package
    assert set(groups.keys()) == {"elv_n30w120.tar"}
    assert set(groups["elv_n30w120.tar"]) == set(tiles)


def test_download_handles_missing_package(monkeypatch, tmp_path):
    # Mock requests
    class MockResp:
        def __init__(self, status_code: int = 200) -> None:
            self.status_code = status_code
            self.raw = MagicMock()

        def close(self):
            pass

    class MockSession:
        def head(self, url, auth=None, allow_redirects: bool = True, timeout: int = 60):
            return MockResp(status_code=404)  # package does not exist (ocean)

        def get(self, url, auth=None, stream: bool = True, timeout: int = 60):
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
