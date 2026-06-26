"""Tests for the river burning workflow in the GEB package."""

from typing import Literal

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString

from geb.hazards.floods.workflows.bathymetry import burn_rivers
from tests.testconfig import output_folder

output_folder_river_burning = output_folder / "river_burning"
output_folder_river_burning.mkdir(exist_ok=True, parents=True)


def create_dummy_data(
    crs: int = 32631,
) -> tuple[xr.DataArray, xr.DataArray, gpd.GeoDataFrame]:
    """Generates synthetic terrain and river networks.

    Args:
        crs: Coordinate reference system (EPSG code) for the output data.

    Returns:
        synthetic elevation DataArray, manning DataArray, and GeoDataFrame of rivers.
    """
    res = 0.25
    # Increased terrain space from 200 to 300 to comfortably fit the new test cases
    x_m = np.arange(0, 300, res)
    y_m = np.arange(0, 300, res)
    X_m, Y_m = np.meshgrid(x_m, y_m)

    # Regional slope + localized hill (computed in meters for consistency)
    regional_slope = 28.0 - 0.05 * X_m - 0.03 * Y_m
    gauss_hill = 5.0 * np.exp(-(((X_m - 130) ** 2 + (Y_m - 100) ** 2) / (2 * 30**2)))
    elevation_matrix = (regional_slope + gauss_hill).astype(np.float32)
    manning_matrix = (0.03 + 0.005 * (elevation_matrix > 12.0)).astype(np.float32)

    # Base line geometries in local meter space
    rivers = [
        LineString([(5, 5), (180, 5)]),  # 0
        LineString(  # 1
            np.column_stack(
                (
                    np.linspace(5, 180, 200),
                    25 + 5 * np.sin(np.linspace(0, 2 * np.pi, 200)),
                )
            )
        ),
        LineString(  # 2
            np.column_stack(
                (
                    np.linspace(5, 190, 200),
                    50 + 15 * np.sin(np.linspace(0, 18 * np.pi, 200)),
                )
            )
        ),
        LineString(  # 3
            np.column_stack(
                (
                    5 + 180 * np.linspace(0, 1, 300),
                    75
                    + 10
                    * np.sin(np.linspace(0, 6 * np.pi, 300))
                    * np.cos(np.linspace(0, 3 * np.pi, 300)),
                )
            )
        ),
        LineString(  # 4
            np.column_stack(
                (
                    np.linspace(5, 190, 200),
                    90
                    + 5 * np.sin((np.linspace(5, 190, 200) - 5) / 3)
                    + 2 * np.cos((np.linspace(5, 190, 200) - 5) / 1.5),
                )
            )
        ),
        LineString(  # 5
            np.column_stack(
                (
                    50 + 12 * np.cos(np.linspace(0, np.pi, 200)),
                    15 + 15 * np.sin(np.linspace(0, np.pi, 200)),
                )
            )
        ),
        LineString([(5, 120), (45, 145), (85, 115), (135, 145), (195, 120)]),  # 7
        # Horseshoe confluence network
        LineString(  # 6
            [(10, 190)]
            + list(
                zip(
                    40 + 15 * np.cos(np.linspace(np.pi, 2 * np.pi, 50)),
                    190 + 12 * np.sin(np.linspace(np.pi, 2 * np.pi, 50)),
                )
            )
            + [(55, 190), (70, 170)]
        ),
        LineString([(10, 145), (40, 155), (70, 170)]),  # 8
        LineString([(100, 195), (110, 180), (120, 170)]),  # 9
        LineString([(70, 170), (120, 170), (195, 165)]),  # 10
        # Explicit depth anomaly testing pairs (11 flows directly into 12)
        LineString([(10, 250), (150, 250)]),  # 11: Deeper Upstream River
        LineString([(150, 250), (285, 250)]),  # 12: Shallower Downstream River
    ]

    gdf_riv = gpd.GeoDataFrame(
        {
            "width": [
                0.01,
                5.0,
                10.0,
                5.0,
                5.0,
                6.0,
                4.0,
                3.5,
                3.0,
                2.5,
                8.0,
                6.0,
                9.0,
            ],
            "depth": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 1.8, 1.5, 1.2, 4.0, 5.5, 2.0],
            "manning": [
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                0.035,
                0.03,
                0.03,
                0.03,
                0.022,
                0.03,
                0.03,
            ],
            "downstream_ID": [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                10,
                10,
                10,
                -1,
                12,  # 11 points to 12
                -1,  # 12 is the outlet out of the domain
            ],
            "shreve_stream_order": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2],
        },
        geometry=rivers,
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        crs=32631,
    )

    if crs == 4326:
        # Spatial scaling constants centered around Amsterdam
        lon_origin, lat_origin = 4.890, 52.370
        lat_deg_per_m = 1.0 / 111320.0
        lon_deg_per_m = 1.0 / (111320.0 * np.cos(np.radians(lat_origin)))

        # Create a matching transformation for the vector data
        # instead of to_crs (which uses the UTM origin), we use the same linear shift
        from shapely.affinity import affine_transform

        # [a, b, d, e, xoff, yoff] -> x' = ax + by + xoff, y' = dx + ey + yoff
        # (x_m, y_m) -> (lon_origin + x_m * lon_deg_per_m, lat_origin + y_m * lat_deg_per_m)
        matrix = [lon_deg_per_m, 0, 0, lat_deg_per_m, lon_origin, lat_origin]
        gdf_riv.geometry = gdf_riv.geometry.map(lambda g: affine_transform(g, matrix))
        gdf_riv.crs = "EPSG:4326"

        x_coords = lon_origin + x_m * lon_deg_per_m
        y_coords = lat_origin + y_m * lat_deg_per_m
    else:
        x_coords = x_m
        y_coords = y_m

    da_elv = xr.DataArray(
        elevation_matrix,
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
        name="elv",
    )
    da_elv = da_elv.rio.write_crs(crs)
    da_elv = da_elv.rio.set_nodata(np.nan)

    da_man = xr.DataArray(
        manning_matrix,
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
        name="manning",
    )
    da_man = da_man.rio.write_crs(crs)
    da_man = da_man.rio.set_nodata(np.nan)

    return da_elv, da_man, gdf_riv


def add_river_obstacles(
    da_elv: xr.DataArray, gdf_riv: gpd.GeoDataFrame
) -> xr.DataArray:
    """Adds local morphological anomalies safely scaled to coordinate space units.

    Args:
        da_elv: Elevation DataArray to modify.
        gdf_riv: GeoDataFrame of rivers to use for obstacle placement.

    Returns:
        Modified elevation DataArray with added obstacles.
    """
    da_elv = da_elv.copy()
    X, Y = np.meshgrid(da_elv.x.values, da_elv.y.values)

    obstacle_specs = [
        (1, 0.40, 1.5, 2.0),
        (2, 0.55, 2.0, 1.5),
        (7, 0.70, 1.0, 2.5),
        (10, 0.35, 1.8, 2.0),
    ]

    # Convert metric sigma parameter into coordinate increments
    lat_origin = 52.370
    deg_per_m_y = 1.0 / 111320.0
    deg_per_m_x = 1.0 / (111320.0 * np.cos(np.radians(lat_origin)))

    for riv_idx, alpha, height, sigma in obstacle_specs:
        p = gdf_riv.geometry.iloc[riv_idx].interpolate(alpha, normalized=True)

        if da_elv.rio.crs.is_geographic:
            sigma_x = sigma * deg_per_m_x
            sigma_y = sigma * deg_per_m_y
        else:
            sigma_x = sigma
            sigma_y = sigma

        bump = height * np.exp(
            -(((X - p.x) / sigma_x) ** 2 + ((Y - p.y) / sigma_y) ** 2) / 2.0
        )
        da_elv.values += bump.astype(np.float32)

    return da_elv


def plot_result(
    da_elv: xr.DataArray,
    da_elv_out: xr.DataArray,
    da_man: xr.DataArray,
    da_man_out: xr.DataArray,
    gdf_riv: gpd.GeoDataFrame,
    title: str,
    filename: str,
) -> None:
    """Plots the original and burned elevation data along with river geometries.

    Args:
        da_elv: Original elevation DataArray.
        da_elv_out: Burned elevation DataArray.
        da_man: Original manning DataArray.
        da_man_out: Burned manning DataArray.
        gdf_riv: GeoDataFrame of rivers.
        title: Title for the plot.
        filename: Filename to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    da_elv.plot(ax=axes[0][0], cmap="terrain")  # ty:ignore[missing-argument]
    axes[0][0].set_title("Original Elevation")
    gdf_riv.plot(ax=axes[0][0], color="blue", linewidth=1, alpha=0.5)

    da_elv_out.plot(ax=axes[1][0], cmap="terrain")  # ty:ignore[missing-argument]
    axes[1][0].set_title("Burned Elevation")
    gdf_riv.plot(ax=axes[1][0], color="blue", linewidth=1, alpha=0.5)

    da_man.plot(ax=axes[0][1], cmap="viridis")  # ty:ignore[missing-argument]
    axes[0][1].set_title("Original Manning")
    gdf_riv.plot(ax=axes[0][1], color="red", linewidth=1, alpha=0.5)

    da_man_out.plot(ax=axes[1][1], cmap="viridis")  # ty:ignore[missing-argument]
    axes[1][1].set_title("Burned Manning")
    gdf_riv.plot(ax=axes[1][1], color="red", linewidth=1, alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_folder_river_burning / filename, dpi=300)
    plt.close()


@pytest.mark.parametrize("crs", [32631, 4326])
@pytest.mark.parametrize("with_obstacles", [False, True])
def test_burn_rivers(crs: int, with_obstacles: bool) -> None:
    """Tests the river burning workflow with and without obstacles in different CRS.

    Args:
        crs: Coordinate reference system (EPSG code) for the test.
        with_obstacles: Whether to add obstacles to the terrain before burning.
    """
    da_elv, da_man, gdf_riv = create_dummy_data(crs=crs)

    if with_obstacles:
        da_elv = add_river_obstacles(da_elv, gdf_riv)

    da_elv_out, da_man_out = burn_rivers(
        elevation_grid=da_elv,
        manning_grid=da_man,
        rivers=gdf_riv,
        fill_first=True,
    )

    # Structural validations
    assert da_elv_out.shape == da_elv.shape
    assert not np.isnan(da_elv_out.values).all()

    crs_lbl: Literal["projected", "geographic"] = (
        "projected" if crs == 32631 else "geographic"
    )
    obs_lbl: Literal["with_obstacles", "no_obstacles"] = (
        "with_obstacles" if with_obstacles else "no_obstacles"
    )

    # Check if burning actually happened (burned values should be significantly lower than original)
    diff = da_elv.values - da_elv_out.values
    max_diff = np.nanmax(diff)

    assert max_diff > 0.5, (
        f"No significant burning detected for {crs_lbl} CRS. Max diff: {max_diff}"
    )

    plot_result(
        da_elv,
        da_elv_out,
        da_man,
        da_man_out,
        gdf_riv,
        f"burn_river_monotonic ({crs_lbl} - {obs_lbl})",
        f"custom_burn_{crs_lbl}_{obs_lbl}.png",
    )
