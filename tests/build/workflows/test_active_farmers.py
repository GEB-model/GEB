"""Test consistent removal and reindexing of farmers outside the active grid."""

import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from geb.agents.crop_farmers import CropFarmers
from geb.build import GEBModel, version_updates
from geb.build.workflows.farmers import retain_active_farmers


@pytest.mark.parametrize("mask_all", [False, True])
def test_retain_active_farmers(mask_all: bool) -> None:
    """Retain partial farms, remove masked farms, and compact IDs.

    Args:
        mask_all: Whether all model cells are masked.
    """
    farms: xr.DataArray = xr.DataArray([[0, 1, 1, 1], [0, 2, -1, 1]], dims=("y", "x"))
    mask: xr.DataArray = xr.DataArray([[mask_all, True]], dims=("y", "x"))
    updated: xr.DataArray
    retained: np.ndarray
    updated, retained = retain_active_farmers(farms, mask, 3)
    np.testing.assert_array_equal(retained, [] if mask_all else [0, 1, 2])
    assert (updated.values[:, 2:] == -1).all()
    if mask_all:
        assert (updated.values == -1).all()
    else:
        np.testing.assert_array_equal(updated.values[:, :2], farms.values[:, :2])


@pytest.mark.parametrize("invalid_attribute_rows", [False, True])
def test_repair_farmer_arrays(tmp_path: Path, invalid_attribute_rows: bool) -> None:
    """Keep multidimensional farmer attributes aligned and make repair idempotent.

    Args:
        tmp_path: Directory for the miniature model inputs.
        invalid_attribute_rows: Whether one array has the wrong number of farmer rows.
    """
    builder: GEBModel = GEBModel(logger=logging.getLogger(__name__), root=tmp_path)
    builder.files = builder.read_or_create_file_library()
    farms: xr.DataArray = xr.DataArray(
        [[0, 2, 1, 1], [0, 2, -1, 1]],
        dims=("y", "x"),
        coords={"y": [0.0, -0.01], "x": [0.0, 0.01, 0.02, 0.03]},
        attrs={"_FillValue": -1},
    ).rio.write_crs(4326)
    builder.grid = xr.Dataset({"mask": xr.DataArray([[False, True]], dims=("y", "x"))})
    builder.subgrid = xr.Dataset(
        {"mask": xr.zeros_like(farms, dtype=bool), "agents/farmers/farms": farms}
    )
    builder.set_array(np.array([10, 11, 12]), name="agents/farmers/region_id")
    builder.set_array(
        np.arange(4).reshape(2, 2)
        if invalid_attribute_rows
        else np.arange(6).reshape(3, 2),
        name="agents/farmers/adaptations",
    )
    if invalid_attribute_rows:
        with pytest.raises(ValueError, match="does not have 3 rows"):
            builder.remove_inactive_farmers()
        np.testing.assert_array_equal(
            builder.array["agents/farmers/region_id"], [10, 11, 12]
        )
        xr.testing.assert_identical(
            builder.subgrid["agents/farmers/farms"],
            farms.rename("agents/farmers/farms"),
        )
        return
    assert builder.remove_inactive_farmers() == 1
    np.testing.assert_array_equal(builder.array["agents/farmers/region_id"], [10, 12])
    np.testing.assert_array_equal(
        builder.array["agents/farmers/adaptations"], [[0, 1], [4, 5]]
    )
    np.testing.assert_array_equal(
        builder.subgrid["agents/farmers/farms"].values, [[0, 1, -1, -1], [0, 1, -1, -1]]
    )
    assert builder.remove_inactive_farmers() == 0


@pytest.mark.parametrize("invalid_ids", [True, False])
def test_elevation_validation(
    monkeypatch: pytest.MonkeyPatch, invalid_ids: bool
) -> None:
    """Compute valid farmer means or identify farmers missing active fields.

    Args:
        monkeypatch: Fixture for supplying a small elevation raster.
        invalid_ids: Whether the active fields omit one farmer ID.
    """
    monkeypatch.setattr(
        "geb.agents.crop_farmers.read_grid",
        Mock(return_value=np.array([[10.0, 20.0], [30.0, 40.0]])),
    )
    farmer: Mock = Mock()
    farmer.var.n = 3 if invalid_ids else 2
    farmer.var.max_n = 4
    farmer.model.files = {"subgrid": {"landsurface/elevation": "unused"}}
    farmer.HRU.decompress.return_value = (
        np.array([[0, 2], [0, -1]]) if invalid_ids else np.array([[0, 1], [0, -1]])
    )
    if invalid_ids:
        with pytest.raises(ValueError, match=r"1 farmers.*IDs: \[1\]"):
            CropFarmers.get_farmer_elevation(farmer)
    else:
        np.testing.assert_array_equal(
            CropFarmers.get_farmer_elevation(farmer).data, [20.0, 20.0]
        )


def test_farmer_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repair farmers when migrating an existing b36 input to b37.

    Args:
        monkeypatch: Fixture for isolating the current package version.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b37")
    builder: Mock = Mock()
    with pytest.raises(RuntimeError, match="Rerun spinup"):
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b36",
            build_model=builder,
            methods={},
            logger=logging.getLogger(__name__),
        )
    builder.remove_inactive_farmers.assert_called_once_with()
    builder.set_version.assert_called_with("1.0.0b37")
