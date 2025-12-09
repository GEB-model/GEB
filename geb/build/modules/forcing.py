"""Forcing data processing and plotting methods for GEB."""

from __future__ import annotations

import re
import tempfile
from datetime import date, datetime, timedelta
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import imageio.v2 as imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xclim.indices as xci
from dateutil.relativedelta import relativedelta
from matplotlib.colors import ListedColormap
from zarr.codecs.numcodecs import FixedScaleOffset

from geb.build.data_catalog.base import Adapter
from geb.build.methods import build_method
from geb.workflows.raster import resample_like

from ...workflows.io import calculate_scaling, write_zarr

if TYPE_CHECKING:
    from geb.build import GEBModel


def plot_forcing(self: GEBModel, da: xr.DataArray, name: str) -> None:
    """Plot forcing data with a temporal (timeline) plot and a spatial plot.

    Args:
        self: The class instance.
        da: The xarray DataArray containing the forcing data. Must have dimensions 'time',
        name: The name of the variable being plotted, used for titles and filenames.
    """
    fig, axes = plt.subplots(
        4, 1, figsize=(20, 10), gridspec_kw={"hspace": 0.5}
    )  # Create 4 subplots stacked vertically

    mask = self.grid["mask"]  # get the GEB grid
    data = (
        (da * ~mask).sum(dim=("y", "x")) / (~mask).sum()
    ).compute()  # Area-weighted average
    assert not np.isnan(data.values).any(), (
        "data contains NaN values"
    )  # ensure no NaNs in data

    plot_timeline(da, data, name, axes[0])  # Plot the entire timeline on the first axis

    for i in range(0, 3):  # plot the first three years on separate axes
        year = data.time[0].dt.year + i  # get the year to plot
        year_data = data.sel(
            time=data.time.dt.year == year
        )  # select data for that year
        if year_data.size > 0:  # only plot if there is data for that year
            plot_timeline(
                da,  # original data
                data.sel(time=da.time.dt.year == year),  # data for that year
                f"{name} - {year.item()}",  # title
                axes[i + 1],  # axis to plot on
            )

    fp = self.report_dir / (
        name + "_timeline.png"
    )  # file path for saving the timeline plot
    fp.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    plt.savefig(fp)  # save the timeline plot
    plt.close(fig)  # close the figure to free memory

    spatial_data = da.mean(dim="time")  # mean over time for spatial plot

    spatial_data.plot()  # plot the spatial data

    plt.title(name)  # title
    plt.xlabel("Longitude")  # x-axis label
    plt.ylabel("Latitude")  # y-axis label

    spatial_fp: Path = self.report_dir / (
        name + "_spatial.png"
    )  # file path for saving the spatial plot
    plt.savefig(spatial_fp)  # save the spatial plot
    plt.close()  # close the plot to free memory


def plot_forecasts(geb_build_model: GEBModel, da: xr.DataArray, name: str) -> None:
    """Plot forecast data with a temporal (timeline) plot and a spatial plot.

    Handles only ensemble forecasts for now. Makes a spatial plot for every single ensemble member.

    Args:
        geb_build_model: The class instance.
        da: The xarray DataArray containing the forecast data. Must have dimensions 'time', 'y', 'x', and 'member'.
        name: The name of the variable being plotted, used for titles and filenames.
    """
    # pre-processing of plotting data
    mask = geb_build_model.grid["mask"]  # get the GEB grid
    da_plot = da.copy()  # make a copy to avoid modifying the original data
    # Convert data to mm/hour if it's precipitation
    if "pr" in name.lower() and "kg m-2 s-1" in da_plot.attrs.get("units", ""):
        da_plot = da_plot * 3600  # convert to mm/hour
        ylabel = "mm/hour"  # set y-axis label
    else:
        da_plot = da_plot.copy()  # no conversion
        ylabel = da_plot.attrs.get("units", "")

    n_members: int = da.sizes["member"]  # number of ensemble members

    # Timeline plot
    fig, ax_time = plt.subplots(1, 1, figsize=(12, 9))  # Create temporal plot

    colors = plt.cm.viridis(np.linspace(0, 1, n_members))  # Distinct colors for members

    spatial_average = (
        (da_plot * ~mask).sum(dim=("y", "x")) / (~mask).sum()
    ).compute()  # Area-weighted average (only over the catchment area)

    ensemble_data = []  # Store ensemble member data
    for i, member in enumerate(spatial_average.member):  # Iterate over ensemble members
        member_avg = spatial_average.sel(member=member)  # Select member data
        ensemble_data.append(member_avg)  # Collect for ensemble mean

        ax_time.plot(  # plot member line
            member_avg.time,
            member_avg,
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
            label=f"Member {member.values}",
        )

    # Calculate ensemble mean and add to plot
    ensemble_mean = sum(ensemble_data) / len(ensemble_data)  # ensemble mean
    ax_time.plot(
        ensemble_mean.time,
        ensemble_mean,
        "k-",
        linewidth=3,
        label="Ensemble Mean",
    )  # plot ensemble mean
    ax_time.legend(
        bbox_to_anchor=(0.5, -0.2), loc="center", ncol=5, fontsize=8
    )  # legend
    ax_time.set_xlabel("Time")  # x-axis label
    ax_time.set_ylabel(ylabel)  # y-axis label
    ax_time.set_title(f"{name} - Ensemble Forecast Timeline")  # title
    ax_time.grid(True, alpha=0.3)  # light grid

    fp = geb_build_model.report_dir / (name + "_ensemble_timeline.png")  # File path
    fp.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    plt.tight_layout()  # tight layout
    plt.savefig(fp, dpi=300, bbox_inches="tight")  # save figure
    plt.close(fig)  # close figure to free memory

    # Spatial plot (max over time))
    n_cols = min(6, n_members)  # Changed from 4 to 6 columns
    n_rows = (n_members + n_cols - 1) // n_cols  # Calculate rows needed

    # Create figure with cartopy projection
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )  # Create subplots with cartopy projection
    plt.subplots_adjust(
        hspace=0.2, wspace=0.2, bottom=0.05, left=0.05, right=0.85
    )  # Tighter spacing

    custom_cmap = plt.cm.Blues  # Use simple Blues colormap
    da_plot_max_over_time = da_plot.max(dim="time")  # max over time for color scale
    for i, member in enumerate(
        da_plot_max_over_time.member
    ):  # Iterate over ensemble members
        ax = axes.flatten()[i]  # Select subplot
        spatial_data = da_plot_max_over_time.sel(member=member)  # Select member data
        if "pr" in name.lower() and "kg m-2 s-1" in da_plot_max_over_time.attrs.get(
            "units", ""
        ):  # Convert spatial data to mm/hour if it's precipitation
            cbar_label = "mm/hour"  # units for colorbar
            vmin = 0  # minimum value for color scale
            vmax = 30  # maximum value for color scale, set to 30 mm/hour for better visualization
        else:
            cbar_label = da_plot_max_over_time.attrs.get(
                "units", ""
            )  # units for colorbar
            vmin = np.min(
                ensemble_data
            )  # min/max over all members for consistent color scale
            vmax = np.max(
                ensemble_data
            )  # min/max over all members for consistent color scale

        im = ax.pcolormesh(  # Plot spatial data
            spatial_data.x,
            spatial_data.y,
            spatial_data,
            cmap=custom_cmap,  # Use custom colormap
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_title(f"Member {member.values}")  # Title for each subplot
        ax.set_xlabel("Longitude")  # Longitude label
        ax.set_ylabel("Latitude")  # Latitude label
        # gridlines
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.2, color="gray", alpha=0.5
        )  # changed linewidth from 0.5 to 0.2
        gl.top_labels = False  # Remove top labels
        gl.right_labels = False  # Remove right labels
        gl.xlabel_style = {"size": 8}  # smaller font size
        gl.ylabel_style = {"size": 8}  # smaller font size
        # add coastlines and borders
        ax.add_feature(
            cfeature.COASTLINE, linewidth=0.5, color="black"
        )  # add coastlines
        ax.add_feature(
            cfeature.BORDERS, linewidth=0.5, color="gray"
        )  # add country borders

        # Add region shapefile boundary with thick line
        geb_build_model.geom["mask"].boundary.plot(
            ax=ax, color="red", linewidth=3, transform=ccrs.PlateCarree()
        )

    fig.subplots_adjust(right=0.85)  # make space for colorbar
    cbar_ax = fig.add_axes((0.87, 0.15, 0.03, 0.7))  # (left, bottom, width, height)
    cbar = fig.colorbar(  # add colorbar
        im,
        cax=cbar_ax,
    )
    cbar.set_label(cbar_label)  # label for colorbar
    fig.suptitle(
        f"{name} - Ensemble Spatial Distribution (Max over Time)", y=0.99
    )  # Overall title
    spatial_fp: Path = geb_build_model.report_dir / (
        name + "_ensemble_spatial.png"
    )  # File path
    plt.savefig(spatial_fp, dpi=300, bbox_inches="tight")  # Save figure
    plt.close(fig)  # Close figure to free memory


def plot_gif(
    report_dir: Path,
    geom_mask_boundary: Any,  # The boundary geometry for plotting catchment outline
    da: xr.DataArray,
    name: str,
    interpolation: str = "none",
    accumulated: bool = False,
) -> None:
    """Create a GIF animation of the data over time.

    Args:
        report_dir: Directory path where output files should be saved.
        geom_mask_boundary: Geometry boundary for the catchment area overlay.
        da: The xarray DataArray containing the data to animate. Must have dimensions 'time', 'y', and 'x'.
        name: The name of the variable being animated, used for titles and filenames.
        interpolation: The interpolation method to use for displaying the data. Default is 'none'. Interpolation can be set to 'bicubic', etc. for smoother interpolation.
        accumulated: Whether to plot accumulated precipitation (True) or instantaneous (False). Default is False.
    """
    # Use xarray's quantile function to calculate percentiles across the ensemble dimension
    ensemble_dim = None
    # Pattern to match YYYYMMDDT000000 format
    date_pattern = r"(\d{4})(\d{2})(\d{2})T\d{6}"
    match = re.search(date_pattern, name)
    if match:
        year, month, day = match.groups()
        forecast_date = f"{year}-{month}-{day}"
    else:
        forecast_date = ""

    for dim in da.dims:
        if (
            "member" in dim.lower()
            or "ensemble" in dim.lower()
            or "perturbation" in dim.lower()
        ):
            ensemble_dim = dim

    percentiles = [25, 50, 75, 90, 95]  # Define desired percentiles
    # Convert chosen percentiles to decimals to use xr.quantile
    percentiles_decimal = [
        p / 100 for p in percentiles
    ]  # Convert to 0-1 range for xarray

    # Calculate all percentiles at once with the quantile function over the ensemble dimension
    ensemble_percentiles_xr = da.quantile(
        percentiles_decimal, dim=ensemble_dim, keep_attrs=True
    )

    # Rename the 'quantile' dimension to 'percentile' and assign percentile values
    ensemble_percentiles_xr = ensemble_percentiles_xr.rename({"quantile": "percentile"})
    ensemble_percentiles_xr = ensemble_percentiles_xr.assign_coords(
        percentile=percentiles
    )

    # Add metadata
    ensemble_percentiles_xr.attrs.update(
        {
            "long_name": f"{name} ensemble percentiles",
            "description": f"Percentiles ({percentiles}) computed across ensemble members",
            "source_variable": name,
            "percentiles": percentiles,
            "ensemble_members": f"Derived from {da.sizes[ensemble_dim] if ensemble_dim else 1} members",
            "computation_method": "xarray.quantile across ensemble dimension",
        }
    )
    da_plot = (
        ensemble_percentiles_xr.copy()
    )  # make a copy to avoid modifying the original data

    # Convert data to mm/hour if it's precipitation
    if "pr" in name.lower() and "kg m-2 s-1" in da_plot.attrs.get("units", ""):
        da_plot = da_plot * 3600  # convert to mm/hour
        # Handle accumulated precipitation if specified (e.g. colormap, ylabel)
        if accumulated:
            da_plot = da_plot.cumsum(dim="time")  # convert to accumulated precipitation
            ylabel = "mm"  # set y-axis label
            name += "_accumulated"
            viridis = cm.get_cmap("viridis")
            viridis_colors = viridis(
                np.linspace(0, 1, 25)
            )  # The more colors, the smoother the gradient but more movement in cbar during animation
            light_blue = np.array([0.7, 0.9, 1.0, 1.0])  # RGBA: light blue
            orange_colors = np.array(
                [
                    [1.0, 0.8, 0.0, 1.0],  # Yellow-orange
                    [1.0, 0.6, 0.0, 1.0],  # Orange
                    [1.0, 0.4, 0.0, 1.0],  # Dark orange
                    [0.9, 0.3, 0.0, 1.0],  # Red-orange
                    [0.8, 0.2, 0.0, 1.0],  # Dark red-orange
                ]
            )
            custom_colors = np.vstack(
                [light_blue.reshape(1, -1), viridis_colors, orange_colors]
            )
            custom_cmap = ListedColormap(custom_colors)
        else:
            ylabel = "mm/hour"  # set y-axis label
            viridis = cm.get_cmap("viridis")
            viridis_colors = viridis(
                np.linspace(0, 1, 20)
            )  # The more colors, the smoother the gradient but more movement in cbar during animation
            light_blue = np.array([0.7, 0.9, 1.0, 1.0])  # RGBA
            custom_colors = np.vstack([light_blue.reshape(1, -1), viridis_colors])
            custom_cmap = ListedColormap(custom_colors)
    else:
        da_plot = da_plot.copy()  # no conversion
        ylabel = da_plot.attrs.get("units", "")

    # Settings for the imshow plot
    vmin = float(da_plot.min())
    vmax = float(da_plot.max())

    data_extent = [
        da_plot.x.min().values,
        da_plot.x.max().values,
        da_plot.y.min().values,
        da_plot.y.max().values,
    ]

    y_coords = da_plot.y.values
    if y_coords[0] < y_coords[-1]:  # Y is increasing
        origin = "lower"
        print("Using origin='lower' - Y coordinates increase")
    else:  # Y is decreasing
        origin = "upper"
        print("Using origin='upper' - Y coordinates decrease")

    # Generating Animation frames
    frames = []
    times = da_plot["time"].values

    for i, t in enumerate(times):
        fig, axes = plt.subplots(
            1,
            len(percentiles),
            figsize=(25, 5),
            constrained_layout=True,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        for j, p in enumerate(percentiles):
            ax = axes[j]
            im = ax.imshow(
                da_plot.sel(percentile=p, time=t).values,
                extent=data_extent,
                cmap=custom_cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.6,
                origin=origin,
                aspect="auto",
                zorder=2,
                interpolation=interpolation,
            )

            geom_mask_boundary.boundary.plot(
                ax=ax,
                color="black",
                linewidth=1.5,
                alpha=0.8,
                zorder=1,
                label="Catchment Boundary",
            )

            try:
                ctx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zorder=0,
                )

            except Exception as e:
                print(f"Warning: Could not add basemap: {e}")

            ax.set_title(f"{p}th percentile", fontsize=14)
            ax.tick_params(labelsize=12)

            if j == 0:
                ax.legend(loc="upper left", fontsize=12, framealpha=0.8)

            if j == len(axes) // 2:
                ax.set_xlabel("Longitude", fontsize=12)
            else:
                ax.set_xlabel("")

            if j == 0:
                ax.set_ylabel("Latitude", fontsize=12)
            else:
                ax.set_ylabel("")

        cbar = fig.colorbar(
            im, ax=axes, orientation="vertical", fraction=0.4, pad=0.01, shrink=0.8
        )
        cbar.set_label(f"{ylabel}", fontsize=12)

        if hasattr(t, "astype"):
            # For numpy datetime64
            time_str = str(t)[:19].replace("T", " ")  # YYYY-MM-DD HH:MM:SS
        else:
            time_str = str(t)

        fig.suptitle(
            f"Forecast initialization {forecast_date}- {time_str}", fontsize=16, y=1
        )

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    # Saving GIF
    gif_fp = report_dir / f"{name}_animation.gif"  # File path for GIF
    imageio.mimsave(gif_fp, frames, fps=5)


def _plot_data(geb_build_model: GEBModel, da: xr.DataArray, name: str) -> None:
    """Plot data using appropriate method based on data type.

    Uses plot_forecasts if 'forecast' is in the name, otherwise uses plot_forcing.

    Args:
        geb_build_model: The class instance.
        da: Data to plot.
        name: Name for the plots and file outputs.
    """
    if "forecast" in name.lower():
        plot_forecasts(geb_build_model, da, name)  # plot forecasts
    else:
        plot_forcing(geb_build_model, da, name)  # plot historical forcing data


def plot_timeline(
    da: xr.DataArray, data: xr.DataArray, name: str, ax: plt.Axes
) -> None:
    """Plot a timeline of the data.

    Args:
        da: the original xarray DataArray containing the data to plot.
        data: the data to plot, should be a 1D xarray DataArray with a time dimension.
        name: the name of the data, used for the plot title.
        ax: the matplotlib axes to plot on.
    """
    ax.plot(data.time, data)
    ax.set_xlabel("Time")
    if "units" in da.attrs:
        ax.set_ylabel(da.attrs["units"])
    ax.set_xlim(data.time[0], data.time[-1])
    ax.set_ylim(data.min(), data.max() * 1.1)
    significant_digits: int = 6
    ax.set_title(
        f"{name} - mean: {data.mean().item():.{significant_digits}f} - min: {data.min().item():.{significant_digits}f} - max: {data.max().item():.{significant_digits}f}"
    )


def get_chunk_size(da: xr.DataArray, target: float | int = 1e8) -> int:
    """Calculate the optimal chunk size for the given xarray DataArray based on the target size.

    Args:
        da: The xarray DataArray for which to calculate the chunk size.
        target: The target size in bytes. Default is 1e8 (100 MB).

    Returns:
        The calculated chunk size in bytes.
    """
    spatial_size = da.x.size * da.y.size
    # Include member dimension if it exists
    if "member" in da.dims:
        spatial_size *= da.member.size
    return int(target / (da.dtype.itemsize * spatial_size))


class Forcing:
    """Contains methods to download and process climate forcing data for GEB."""

    def __init__(self) -> None:
        """Initialize the Forcing class."""
        pass

    def set_xy_attrs(self, da: xr.DataArray) -> None:
        """Set CF-compliant attributes for the x and y coordinates of a DataArray."""
        da.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
        da.y.attrs = {"long_name": "latitude", "units": "degrees_north"}

    def set_pr_kg_per_m2_per_s(
        self,
        da: xr.DataArray,
        name: str = "climate/pr_kg_per_m2_per_s",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Sets the Precipitation DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the precipitation values efficiently.

        Args:
            da: The xarray DataArray containing the precipitation data.
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with precipitation data.
        """
        da.attrs = {
            "standard_name": "precipitation_flux",
            "long_name": "Precipitation",
            "units": "kg m-2 s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        # maximum rainfall in one hour was 304.8 mm in 1956 in Holt, Missouri, USA
        # https://www.guinnessworldrecords.com/world-records/737965-greatest-rainfall-in-one-hour
        # we take a wide margin of 500 mm/h
        # this function is currently daily, so the hourly value should be save
        max_value: float = 500 / 3600  # convert to kg/m2/s
        precision: float = 0.01 / 3600  # 0.01 mm in kg/m2/s

        offset: int = 0
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, max_value, offset=offset, precision=precision
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        if "forecasts" in name.lower():
            # Check if GIF files already exist before creating them
            gif_fp_regular = self.report_dir / f"{name.replace('/', '_')}_animation.gif"
            gif_fp_accumulated = (
                self.report_dir / f"{name.replace('/', '_')}_animation_accumulated.gif"
            )
            if not gif_fp_regular.exists():
                plot_gif(
                    self.report_dir, self.geom["mask"], da, name, accumulated=False
                )
                self.logger.info(f"Creating a GIF animation: {gif_fp_regular.name}")
            else:
                self.logger.info(
                    f"GIF file {gif_fp_regular.name} already exists, skipping regular animation creation"
                )

            if not gif_fp_accumulated.exists():
                plot_gif(self.report_dir, self.geom["mask"], da, name, accumulated=True)
                self.logger.info(f"Creating a GIF animation: {gif_fp_accumulated.name}")
            else:
                self.logger.info(
                    f"GIF file {gif_fp_accumulated.name} already exists, skipping accumulated animation creation"
                )
        return da

    def set_rsds_W_per_m2(
        self,
        da: xr.DataArray,
        name: str = "climate/rsds_W_per_m2",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Sets the Surface Downwelling Shortwave Radiation DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the radiation values efficiently.

        Args:
            da: The xarray DataArray containing the shortwave radiation data.
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with shortwave radiation data.
        """
        da.attrs = {
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "long_name": "Surface Downwelling Shortwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        return da

    def set_rlds_W_per_m2(
        self,
        da: xr.DataArray,
        name: str = "climate/rlds_W_per_m2",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Sets the Surface Downwelling Longwave Radiation DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the radiation values efficiently.

        Args:
            da: The xarray DataArray containing the longwave radiation data.
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with longwave radiation data.
        """
        da.attrs = {
            "standard_name": "surface_downwelling_longwave_flux_in_air",
            "long_name": "Surface Downwelling Longwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        return da

    def set_tas_2m_K(
        self,
        da: xr.DataArray,
        name: str = "climate/tas_2m_K",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Sets the Near-Surface Air Temperature DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the temperature values efficiently.

        Args:
            da: The xarray DataArray containing the air temperature data.
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with air temperature data.
        """
        da.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C = 273.15
        offset = -15 - K_to_C  # average temperature on earth
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )

        _plot_data(self, da, name)
        return da

    def set_dewpoint_tas_2m_K(
        self,
        da: xr.DataArray,
        name: str = "climate/dewpoint_tas_2m_K",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Sets the Near-Surface Dewpoint Temperature DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the dewpoint temperature values efficiently.

        Args:
            da: The xarray DataArray containing the dewpoint temperature data.
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with dewpoint temperature data.
        """
        da.attrs = {
            "standard_name": "air_temperature_dow_point",
            "long_name": "Hourly Near-Surface Dewpoint Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C: float = 273.15
        offset: float = -15 - K_to_C  # average temperature on earth
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        return da

    def set_ps_pascal(
        self,
        da: xr.DataArray,
        name: str = "climate/ps_pascal",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Sets the Surface Air Pressure DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the pressure values efficiently.

        Args:
            da: The xarray DataArray containing the surface air pressure data.
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with surface air pressure data.
        """
        da.attrs = {
            "standard_name": "surface_air_pressure",
            "long_name": "Surface Air Pressure",
            "units": "Pa",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset: int = -100_000
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 30_000, 120_000, offset=offset, precision=10
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        return da

    def set_wind_10m_m_per_s(
        self,
        da: xr.DataArray,
        direction: str,
        name: str = "climate/wind_{direction}10m_m_per_s",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Sets the Near-Surface Wind Speed DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the wind speed values efficiently.

        Args:
            da: The xarray DataArray containing the wind speed data.
            direction: The wind direction component (e.g., 'u' or 'v').
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with wind speed data.
        """
        name: str = name.format(direction=direction)
        da.attrs = {
            "standard_name": "wind_speed",
            "long_name": "Near-Surface Wind Speed",
            "units": "m s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        # wind can be both positive and negative
        # we assume a maximum wind speed of 120 m/s (432 km/h), which is a stronger
        # than the strongest wind gust ever recorded on earth (113 m/s)
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -120, 120, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        return da

    def set_SPEI(
        self, da: xr.DataArray, name: str = "climate/SPEI", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        """Sets the Standard Precipitation Evapotranspiration Index (SPEI) DataArray with appropriate attributes and scaling.

        Uses scaling and rounding to store the SPEI values efficiently.

        Args:
            da: The xarray DataArray containing the SPEI data.
            name: The name to assign to the DataArray in the model.
            *args: Additional positional arguments to pass to the set_other method.
            **kwargs: Additional keyword arguments to pass to the set_other method.

        Returns:
            The processed xarray DataArray with SPEI data.
        """
        da.attrs = {
            "units": "-",
            "long_name": "Standard Precipitation Evapotranspiration Index",
            "name": "spei",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        # this range corresponds to probabilities of lower than 0.001 and higher than 0.999
        # which should be considered non-significant
        min_SPEI = -3.09
        max_SPEI = 3.09
        da = da.clip(min=min_SPEI, max=max_SPEI)

        offset = 0
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, min_SPEI, max_SPEI, offset=offset, precision=0.001
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def setup_forcing_ERA5(self) -> None:
        """Sets up the ERA5 forcing data for GEB.

        Sets:
            The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.
        """
        era5_store: Adapter = self.new_data_catalog.fetch("era5")
        era5_loader: partial = partial(
            era5_store.read,
            start_date=self.start_date - relativedelta(years=1),
            end_date=self.end_date,
            bounds=self.grid["mask"].rio.bounds(recalc=True),
        )

        pr_hourly: xr.DataArray = era5_loader(variable="tp")
        pr_hourly: xr.DataArray = pr_hourly * (
            1000 / 3600
        )  # convert from m/hr to kg/m2/s

        # ensure no negative values for precipitation, which may arise due to float precision
        pr_hourly: xr.DataArray = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
        pr_hourly: xr.DataArray = self.set_pr_kg_per_m2_per_s(pr_hourly)

        tas: xr.DataArray = era5_loader("t2m")
        self.set_tas_2m_K(tas)

        dew_point_tas: xr.DataArray = era5_loader("d2m")
        self.set_dewpoint_tas_2m_K(dew_point_tas)

        rsds: xr.DataArray = (
            era5_loader("ssrd") / 3600  # convert from J/m2/(per timestep) to W/m2
        )  # surface_solar_radiation_downwards
        self.set_rsds_W_per_m2(rsds)

        # surface_thermal_radiation_downwards
        rlds: xr.DataArray = (
            era5_loader("strd") / 3600
        )  # convert from J/m2/(per timestep) to W/m2
        self.set_rlds_W_per_m2(rlds)

        pressure: xr.DataArray = era5_loader("sp")
        self.set_ps_pascal(pressure)

        u_wind: xr.DataArray = era5_loader("u10")
        self.set_wind_10m_m_per_s(u_wind, direction="u")

        v_wind: xr.DataArray = era5_loader("v10")
        self.set_wind_10m_m_per_s(v_wind, direction="v")

        elevation_forcing: xr.DataArray = self.get_elevation_forcing(pr_hourly)
        self.set_other(
            elevation_forcing,
            name="climate/elevation_forcing",
        )

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_forcing(
        self,
        forcing: str = "ERA5",
    ) -> None:
        """Sets up the forcing data for GEB.

        Args:
            forcing: The data source to use for the forcing data. Currently only ERA5 is supported.

        Sets:
            The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.

        Raises:
            ValueError: If an unknown data source is specified.
        """
        if forcing == "ISIMIP":
            raise NotImplementedError(
                "ISIMIP forcing is not supported anymore. We switched fully to hourly forcing data."
            )
        elif forcing == "ERA5":
            self.setup_forcing_ERA5()
        elif forcing == "CMIP":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(f"Unknown data source: {forcing}, supported are 'ERA5'")

    @build_method(depends_on=["setup_forcing"])
    def setup_SPEI(
        self,
        calibration_period_start: date = date(1981, 1, 1),
        calibration_period_end: date = date(2010, 1, 1),
        window_months: int = 12,
    ) -> None:
        """Sets up the Standardized Precipitation Evapotranspiration Index (SPEI).

        Note that due to the sliding window, the SPEI data will be shorter than the original data. When
        a sliding window of 12 months is used, the SPEI data will be shorter by 11 months.

        Also sets up the Generalized Extreme Value (GEV) parameters for the SPEI data, being
        the c shape (ξ), loc location (μ), and scale (σ) parameters.

        The chunks for the climate data are optimized for reading the data in xy-direction. However,
        for the SPEI calculation, the data is needs to be read in time direction. Therefore, we
        create an intermediate temporary file of the water balance wher chunks are in an intermediate
        size between the xy and time chunks.

        Args:
            calibration_period_start: The start time of the reSPEI data in ISO 8601 format (YYYY-MM-DD).
            calibration_period_end: The end time of the SPEI data in ISO 8601 format (YYYY-MM-DD). Endtime is exclusive.
            window_months: The window size in months for the SPEI calculation. Default is 12 months.

        Raises:
            ValueError: If the input data do not have the same coordinates.
        """
        assert window_months <= 12, (
            "window_months must be less than or equal to 12 (otherwise we run out of climate data)"
        )
        assert window_months >= 1, (
            "window_months must be greater than or equal to 1 (otherwise we have no sliding window)"
        )

        # assert input data have the same coordinates
        tasmin_2m_K = self.other["climate/tas_2m_K"].resample(time="D").min()
        tasmax_2m_K = self.other["climate/tas_2m_K"].resample(time="D").max()
        pr_kg_per_m2_per_s = (
            self.other["climate/pr_kg_per_m2_per_s"].resample(time="D").mean()
        )

        assert np.array_equal(self.other["climate/pr_kg_per_m2_per_s"].x, tasmin_2m_K.x)
        assert np.array_equal(self.other["climate/pr_kg_per_m2_per_s"].y, tasmin_2m_K.y)

        assert calibration_period_start < calibration_period_end, (
            f"Start date {calibration_period_start} must be earlier than end date {calibration_period_end}."
        )

        if not self.other[
            "climate/pr_kg_per_m2_per_s"
        ].time.min().dt.date <= calibration_period_start and self.other[
            "climate/pr_kg_per_m2_per_s"
        ].time.max().dt.date >= calibration_period_end - timedelta(days=1):
            forcing_start_date = (
                self.other["climate/pr_kg_per_m2_per_s"].time.min().dt.date.item()
            )
            forcing_end_date = (
                self.other["climate/pr_kg_per_m2_per_s"].time.max().dt.date.item()
            )
            raise ValueError(
                f"water data does not cover the entire calibration period, forcing data covers from {forcing_start_date} to {forcing_end_date}, "
                f"while requested calibration period is from {calibration_period_start} to {calibration_period_end}"
            )

        pet = xci.potential_evapotranspiration(
            tasmin=tasmin_2m_K,
            tasmax=tasmax_2m_K,
            # hurs=self.other["climate/hurs"],
            # rsds=self.other["climate/rsds"],
            # rlds=self.other["climate/rlds"],
            # rsus=self.full_like(
            #     self.other["climate/rsds"],
            #     fill_value=0,
            #     nodata=np.nan,
            #     attrs=self.other["climate/rsds"].attrs,
            # ),
            # rlus=self.full_like(
            #     self.other["climate/rsds"],
            #     fill_value=0,
            #     nodata=np.nan,
            #     attrs=self.other["climate/rsds"].attrs,
            # ),
            # sfcWind=self.other["climate/sfcwind"],
            method="BR65",
        ).astype(np.float32)

        # Compute the potential evapotranspiration
        water_budget = xci.water_budget(pr=pr_kg_per_m2_per_s, evspsblpot=pet)

        water_budget = water_budget.resample(time="MS").mean(keep_attrs=True)
        water_budget.attrs["_FillValue"] = np.nan

        temp_xy_chunk_size = 50

        with tempfile.TemporaryDirectory() as tmp_water_budget_folder:
            tmp_water_budget_file = (
                Path(tmp_water_budget_folder) / "tmp_water_budget_file.zarr"
            )
            self.logger.info("Exporting temporary water budget to zarr")
            water_budget = write_zarr(
                water_budget,
                tmp_water_budget_file,
                crs=4326,
                x_chunksize=temp_xy_chunk_size,
                y_chunksize=temp_xy_chunk_size,
                time_chunksize=50,
                time_chunks_per_shard=None,
            ).chunk({"time": -1})  # for the SPEI calculation time must not be chunked

            # We set freq to None, so that the input frequency is used (no recalculating)
            # this means that we can calculate SPEI much more efficiently, as it is not
            # rechunked in the xclim package

            # The log-logistic distribution used in SPEI has three parameters: scale, shape, and location.
            # In practice, fixing the location (floc) to 0 simplifies the fitting process and often
            # provides satisfactory results.The fitting is then effectively done with two parameters,
            # also reducing the risk of overfitting, especially with limited data.
            # When empirical data suggest that the climatic water balance values are significantly shifted,
            # a non-zero floc may better fit the distribution. However, this is not typical in routine applications.
            SPEI = xci.standardized_precipitation_evapotranspiration_index(
                wb=water_budget,
                cal_start=calibration_period_start.strftime("%Y-%m-%d"),
                cal_end=calibration_period_end.strftime("%Y-%m-%d"),
                freq=None,
                window=window_months,
                dist="fisk",  # log-logistic distribution
                method="APP",  # approximative method
                fitkwargs={
                    "floc": water_budget.min().compute().item()
                },  # location parameter, assures that the distribution is always positive
            ).astype(np.float32)

            # remove all nan values as a result of the sliding window
            SPEI: xr.DataArray = SPEI.isel(
                time=slice(window_months - 1, None)
            ).compute()

            with tempfile.TemporaryDirectory() as tmp_spei_folder:
                tmp_spei_file = Path(tmp_spei_folder) / "tmp_spei_file.zarr"
                self.logger.info("Calculating SPEI and exporting to temporary file...")
                SPEI.attrs = {
                    "_FillValue": np.nan,
                }
                SPEI: xr.DataArray = write_zarr(
                    SPEI,
                    tmp_spei_file,
                    x_chunksize=temp_xy_chunk_size,
                    y_chunksize=temp_xy_chunk_size,
                    time_chunksize=10,
                    time_chunks_per_shard=None,
                    crs=4326,
                )

                self.set_SPEI(SPEI)

                self.logger.info("calculating GEV parameters...")

                # Group the data by year and find the maximum monthly sum for each year
                SPEI_yearly_min = SPEI.groupby("time.year").min(dim="time", skipna=True)
                SPEI_yearly_min = (
                    SPEI_yearly_min.rename({"year": "time"})
                    .chunk({"time": -1})
                    .compute()
                )

                GEV = xci.stats.fit(SPEI_yearly_min, dist="genextreme").compute()

                self.set_other(
                    GEV.sel(dparams="c").astype(np.float32), name="climate/gev_c"
                )
                self.set_other(
                    GEV.sel(dparams="loc").astype(np.float32), name="climate/gev_loc"
                )
                self.set_other(
                    GEV.sel(dparams="scale").astype(np.float32),
                    name="climate/gev_scale",
                )

    @build_method(depends_on=["setup_forcing"])
    def setup_pr_GEV(self) -> None:
        """Sets up the Generalized Extreme Value (GEV) parameters for the precipitation data.

        Sets the c shape (ξ), loc location (μ), and scale (σ) parameters.
        """
        pr: xr.DataArray = (
            self.other["climate/pr_kg_per_m2_per_s"] * 3600
        )  # convert to mm/hour
        pr_monthly: xr.DataArray = pr.resample(time="M").sum(dim="time", skipna=True)

        pr_yearly_max = (
            pr_monthly.groupby("time.year")
            .max(dim="time", skipna=True)
            .rename({"year": "time"})
            .chunk({"time": -1})
            .compute()
        )

        gev_pr = xci.stats.fit(pr_yearly_max, dist="genextreme").compute()

        self.set_other(
            gev_pr.sel(dparams="c").astype(np.float32), name="climate/pr_gev_c"
        )
        self.set_other(
            gev_pr.sel(dparams="loc").astype(np.float32), name="climate/pr_gev_loc"
        )
        self.set_other(
            gev_pr.sel(dparams="scale").astype(np.float32),
            name="climate/pr_gev_scale",
        )

    def get_elevation_forcing(self, forcing_grid: xr.DataArray) -> xr.DataArray:
        """Gets elevation maps for both the normal grid (target of resampling) and the forcing grid.

        Args:
            forcing_grid: grid of the forcing data

        Returns:
            elevation data for the forcing grid and the normal grid
        """
        xmin, ymin, xmax, ymax = forcing_grid.rio.bounds(recalc=True)

        buffer: float = 0.5
        xmin: float = xmin - buffer
        ymin: float = ymin - buffer
        xmax: float = xmax + buffer
        ymax: float = ymax + buffer

        elevation: xr.DataArray = (
            self.new_data_catalog.fetch(
                "fabdem", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, prefix="forcing"
            )
            .read(prefix="forcing")
            .compute()
        )

        # FABDEM has nodata values in the ocean, for which we can assume an elevation of 0 m
        elevation = xr.where(~np.isnan(elevation), elevation, 0, keep_attrs=True)

        target: xr.DataArray = forcing_grid.isel(time=0).drop_vars("time")

        elevation_forcing: xr.DataArray = resample_like(
            elevation, target, method="bilinear"
        )
        elevation_forcing: xr.DataArray = elevation_forcing.chunk({"x": -1, "y": -1})
        elevation_forcing: xr.DataArray = elevation_forcing.rio.write_crs(4326)

        return elevation_forcing

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_CO2_concentration(self) -> None:
        """Aquires the CO2 concentration data for the specified SSP in ppm."""
        df: pd.DataFrame = self.new_data_catalog.fetch("isimip_co2").read(
            scenario=self.ISIMIP_ssp
        )
        df: pd.DataFrame = df[
            (df.index >= self.start_date.year) & (df.index <= self.end_date.year)
        ]
        self.set_table(df, name="climate/CO2_ppm")

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_forecasts(
        self,
        forecast_start: date | datetime,
        forecast_end: date | datetime,
        forecast_provider: str,
        forecast_model: str,
        forecast_resolution: float,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        n_ensemble_members: int,
    ) -> None:
        """Sets up forecast data for the model based on configuration.

        Args:
            forecast_variable: List of ECMWF parameter codes to download (see ECMWF documentation).
            forecast_start: The forecast initialization time (date or datetime).
            forecast_end: The forecast end time (date or datetime).
            forecast_provider: The forecast data provider to use (default: "ECMWF").
            forecast_model: The ECMWF forecast model to use (probabilistic_forecast or control_forecast).
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep_hours: The forecast timestep in hours.
            n_ensemble_members: The number of ensemble members to download.
        """
        if (
            forecast_provider == "ECMWF"
        ):  # Check if ECMWF is the selected forecast provider
            self.setup_forecasts_ECMWF(  # Call ECMWF-specific setup method
                forecast_start,  # Pass forecast start date
                forecast_end,  # Pass forecast end date
                forecast_model,  # Pass forecast model type
                forecast_resolution,  # Pass spatial resolution
                forecast_horizon,  # Pass forecast horizon in hours
                forecast_timestep_hours,  # Pass timestep interval
                n_ensemble_members,  # Pass number of ensemble members
            )

    def setup_forecasts_ECMWF(
        self,
        forecast_start: date | datetime,
        forecast_end: date | datetime,
        forecast_model: str,
        forecast_resolution: float,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        n_ensemble_members: int = 50,
    ) -> None:
        """Sets up the folder structure for ECMWF forecast data.

        Args:
            forecast_start: The forecast initialization time (date or datetime).
            forecast_end: The forecast end time (date or datetime).
            forecast_model: The ECMWF forecast model to use (probabilistic_forecast or control_forecast).
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep_hours: The forecast timestep in hours.
            n_ensemble_members: The number of ensemble members to download (default: 50).
        """
        MARS_codes: dict[str, float] = {  # Complete set of weather variables
            "tp": 228.128,  # total precipitation
            "t2m": 167.128,  # 2 metre temperature
            "d2m": 168.128,  # 2 metre dewpoint temperature
            "ssrd": 169.128,  # surface shortwave solar radiation downwards
            "strd": 175.128,  # surface longwave radiation downwards
            "sp": 134.128,  # surface pressure
            "u10": 165.128,  # 10 metre u-component of wind
            "v10": 166.128,  # 10 metre v-component of wind
        }

        forecast_issue_dates = pd.date_range(  # Create pandas date range
            start=forecast_start,  # Start from forecast start date
            end=forecast_end,  # End at forecast end date
            freq="24H",  # Daily frequency (24-hour intervals)
        )

        self.logger.info(f"Processing {forecast_model} ECMWF forecasts...")

        ECMWF_forecasts_store = self.new_data_catalog.fetch(
            "ecmwf_forecasts",
            forecast_variables=list(MARS_codes.values()),
            bounds=self.bounds,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            forecast_model=forecast_model,  # Use current model type
            forecast_resolution=forecast_resolution,
            forecast_horizon=forecast_horizon,  # Forecast horizon in hours
            forecast_timestep_hours=forecast_timestep_hours,  # Temporal resolution in hours
            n_ensemble_members=n_ensemble_members,  # Number of ensemble members
        )

        for (
            forecast_issue_date
        ) in forecast_issue_dates:  # # Process each forecast issue date separately
            forecast_issue_date_str = forecast_issue_date.strftime(
                "%Y%m%dT%H%M%S"
            )  # Format date for filenames

            self.logger.info(f"Processing forecast issued at {forecast_issue_date}...")

            ECMWF_forecast = ECMWF_forecasts_store.read(
                bounds=self.bounds,
                forecast_issue_date=forecast_issue_date,
                forecast_model=forecast_model,
                forecast_resolution=forecast_resolution,
                forecast_horizon=forecast_horizon,
                forecast_timestep_hours=forecast_timestep_hours,
                reproject_like=self.other["climate/pr_kg_per_m2_per_s"],
            )  # Reproject to grid of other climate data

            # Create name based on forecast_model for consistent file structure
            if forecast_model == "both_control_and_probabilistic":
                base_name = (
                    f"forecasts/ECMWF/merged_control_ensemble/{forecast_issue_date_str}"
                )
            else:
                base_name = (
                    f"forecasts/ECMWF/{forecast_model}/{forecast_issue_date_str}"
                )

            # Extract and process hourly precipitation data
            pr = ECMWF_forecast["tp"].rename(
                "precipitation"
            )  # Get total precipitation variable
            pr = pr.where(
                pr >= 0, 0
            )  # Handle negative values (caused by floating-point precision issues) by setting them to zero
            self.set_pr_kg_per_m2_per_s(
                pr,
                name=f"{base_name}/pr_kg_per_m2_per_s_{forecast_issue_date_str}",  # Use date-specific filename
            )

            tas = ECMWF_forecast["t2m"].rename("tas")  # Extract 2-meter temperature
            self.set_tas_2m_K(
                tas, name=f"{base_name}/tas_2m_K_{forecast_issue_date_str}"
            )

            dew_point_tas = ECMWF_forecast["d2m"].rename(
                "dew_point_tas"
            )  # Extract dewpoint temperature
            self.set_dewpoint_tas_2m_K(
                dew_point_tas,
                name=f"{base_name}/dewpoint_tas_2m_K_{forecast_issue_date_str}",
            )

            rsds = ECMWF_forecast["ssrd"].rename("rsds")  # Extract shortwave radiation
            self.set_rsds_W_per_m2(
                rsds,
                name=f"{base_name}/rsds_W_per_m2_{forecast_issue_date_str}",
            )

            # Process surface longwave (thermal) radiation downwards
            rlds = ECMWF_forecast["strd"].rename("rlds")  # Extract longwave radiation
            self.set_rlds_W_per_m2(
                rlds,
                name=f"{base_name}/rlds_W_per_m2_{forecast_issue_date_str}",
            )

            pressure = ECMWF_forecast["sp"].rename("ps")  # Extract surface pressure
            self.set_ps_pascal(
                pressure,
                name=f"{base_name}/ps_pascal_{forecast_issue_date_str}",
            )

            u_wind = ECMWF_forecast["u10"].rename(
                "u10"
            )  # Extract u-component of wind at 10m
            self.set_wind_10m_m_per_s(
                u_wind,
                direction="u",
                name=f"{base_name}/wind_u10m_m_per_s_{forecast_issue_date_str}",
            )

            v_wind = ECMWF_forecast["v10"].rename(
                "v10"
            )  # Extract v-component of wind at 10m
            self.set_wind_10m_m_per_s(
                v_wind,
                direction="v",
                name=f"{base_name}/wind_v10m_m_per_s_{forecast_issue_date_str}",
            )
