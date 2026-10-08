"""Module for building groundwater related datasets for GEB."""

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from affine import Affine

from geb.build.methods import build_method
from geb.workflows.io import get_window
from geb.workflows.raster import (
    convert_nodata,
    full_like,
    interpolate_na_2d,
    pad_cells,
    rasterize_like,
    resample_like,
)

from .base import BuildModelBase


class GroundWater(BuildModelBase):
    """Contains all build methods for the groundwater for GEB."""

    def __init__(self) -> None:
        """Initialize the GroundWater class."""
        pass

    @build_method(required=False)
    def setup_karst(self) -> None:
        """Use WOKAM to estimate how much of each grid cell is karst.

        Notes:
            WOKAM gives ranges, not exact fractions. Use their
            midpoints: 0.825 for continuous rocks, 0.4 for discontinuous rocks,
            and 0.65 for mixed rocks. Inactive grid cells are NaN.

        Returns:
            None; saves ``groundwater/karst_fraction`` (0–1).

        Raises:
            ValueError: If a coordinate reference system is missing, the grid
                is rotated, or WOKAM has unknown rock classes.
        """  # noqa: DOC202
        polygons: gpd.GeoDataFrame = self.data_catalog.fetch("wokam").read()
        mask: xr.DataArray = self.grid["mask"]
        if polygons.crs is None or mask.rio.crs is None:
            raise ValueError("WOKAM polygons and the model grid must have a CRS.")
        if not polygons["rock_type"].isin([1, 2, 3, 4, 5]).all():
            raise ValueError("WOKAM contains unknown rock classes.")
        transform: Affine = mask.rio.transform(recalc=True)
        if transform.b != 0 or transform.d != 0:
            raise ValueError("Karst coverage requires an unrotated model grid.")
        coverage: xr.DataArray = full_like(mask, 0.0, nodata=np.nan, dtype=np.float32)
        coverage.values[mask.values] = np.nan
        coverage.attrs["units"] = "1"
        coverage.attrs["source"] = "WHYMAP WOKAM, BGR, IAH, KIT, UNESCO, 2017"
        polygons = polygons.to_crs(mask.rio.crs)
        bounds: tuple[float, float, float, float] = mask.rio.bounds(recalc=True)
        polygons = polygons.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]].copy()
        if not polygons.empty:
            # Merge polygons of the same rock type to avoid counting overlap twice.
            polygons = polygons.dissolve(by="rock_type").to_crs(6933)
            rows: np.ndarray
            columns: np.ndarray
            rows, columns = np.where(~mask.values)
            cell_x: np.ndarray = transform.c + columns * transform.a
            cell_y: np.ndarray = transform.f + rows * transform.e
            # Calculate areas in an equal-area projection, rather than degrees.
            cells: gpd.GeoSeries = gpd.GeoSeries(
                shapely.box(
                    np.minimum(cell_x, cell_x + transform.a),
                    np.minimum(cell_y, cell_y + transform.e),
                    np.maximum(cell_x, cell_x + transform.a),
                    np.maximum(cell_y, cell_y + transform.e),
                ),
                crs=mask.rio.crs,
            ).to_crs(6933)
            cell_indices: np.ndarray
            polygon_indices: np.ndarray
            cell_indices, polygon_indices = polygons.sindex.query(
                cells, predicate="intersects"
            )
            # WOKAM codes: 1/3 continuous, 2/5 discontinuous, 4 mixed rocks.
            rock_fraction: np.ndarray = polygons.index.map(
                {1: 0.825, 2: 0.4, 3: 0.825, 4: 0.65, 5: 0.4}
            ).to_numpy(dtype=np.float64)
            overlap_area_m2: np.ndarray = shapely.area(
                shapely.intersection(
                    cells.array[cell_indices], polygons.geometry.array[polygon_indices]
                )
            )
            karst_area_m2: np.ndarray = np.bincount(
                cell_indices,
                weights=overlap_area_m2 * rock_fraction[polygon_indices],
                minlength=len(cells),
            )
            coverage.values[rows, columns] = np.clip(
                karst_area_m2 / cells.area.to_numpy(), 0.0, 1.0
            ).astype(np.float32)
        self.set_grid(coverage, name="groundwater/karst_fraction")

    @build_method(depends_on=["setup_elevation"], required=True)
    def setup_groundwater(
        self,
        minimum_thickness_confined_layer: int | float = 50,
        maximum_thickness_confined_layer: int | float = 1000,
        force_one_layer: bool = True,
    ) -> None:
        """Sets up the MODFLOW grid for GEB.

        This code is adopted from the GLOBGM model (https://github.com/UU-Hydro/GLOBGM). Also see ThirdPartyNotices.txt.

        More about GLOBGM: https://doi.org/10.5194/gmd-17-275-2024
        More about Fan: https://doi.org/10.1126/science.1229881

        Args:
            minimum_thickness_confined_layer: The minimum thickness of the confined layer in meters. Default is 50.
            maximum_thickness_confined_layer: The maximum thickness of the confined layer in meters. Default is 1000.
            force_one_layer: If True, the model will be forced to use only one layer. Default is True.
        """
        aquifer_top_elevation = convert_nodata(
            self.grid["landsurface/elevation_m"], new_nodata=np.nan
        )

        # Create an extended template grid (one cell larger in each direction)
        # to store groundwater heads and aquifer properties for all cells surrounding the model domain,
        # which are used as boundary conditions in the groundwater model.
        extended_template: xr.DataArray = pad_cells(aquifer_top_elevation, 1, 1, 1, 1)
        extended_bounds: tuple[float, float, float, float] = (
            extended_template.rio.bounds(recalc=True)
        )

        # Determine outside boundary cells surrounding the active model domain on the extended grid.
        # An inactive cell on the extended grid is an external boundary cell if any of its 4 orthogonal neighbors is active.
        is_active_model: np.ndarray = ~np.isnan(aquifer_top_elevation.values)
        extended_active: np.ndarray = np.zeros(
            (extended_template.shape[0], extended_template.shape[1]), dtype=bool
        )
        extended_active[1:-1, 1:-1] = is_active_model

        touches_active: np.ndarray = np.zeros_like(extended_active, dtype=bool)
        touches_active[:-1, :] |= extended_active[1:, :]
        touches_active[1:, :] |= extended_active[:-1, :]
        touches_active[:, :-1] |= extended_active[:, 1:]
        touches_active[:, 1:] |= extended_active[:, :-1]

        boundary_mask_data: np.ndarray = ~extended_active & touches_active
        extended_boundary_mask: xr.DataArray = xr.DataArray(
            boundary_mask_data,
            coords=extended_template.coords,
            dims=extended_template.dims,
            name="boundary_mask",
        )
        extended_boundary_mask.attrs["_FillValue"] = None
        self.set_other(extended_boundary_mask, name="groundwater/boundary_mask")

        # Load total thickness over the extended bounds
        total_thickness = self.data_catalog.fetch(
            "total_groundwater_thickness_globgm"
        ).read()
        total_thickness = total_thickness.isel(
            get_window(total_thickness.x, total_thickness.y, extended_bounds, buffer=2)
        )

        total_thickness = np.clip(
            total_thickness,
            minimum_thickness_confined_layer,
            maximum_thickness_confined_layer,
        )

        confining_layer = self.data_catalog.fetch(
            "thickness_confining_layer_globgm"
        ).read()
        confining_layer = confining_layer.isel(
            get_window(confining_layer.x, confining_layer.y, extended_bounds, buffer=2)
        )

        if not (confining_layer == 0).all() and not force_one_layer:  # two-layer-model
            two_layers = True
        else:
            two_layers = False

        if two_layers:
            # make sure that total thickness is at least 50 m thicker than confining layer
            total_thickness = np.maximum(
                total_thickness, confining_layer + minimum_thickness_confined_layer
            )
            # thickness of layer 2 is based on the predefined confiningLayerThickness
            relative_bottom_top_layer = -confining_layer
            # make sure that the minimum thickness of layer 2 is at least 0.1 m
            thickness_top_layer = np.maximum(0.1, -relative_bottom_top_layer)
            relative_bottom_top_layer = -thickness_top_layer
            # thickness of layer 1 is at least 5.0 m
            thickness_bottom_layer = np.maximum(
                5.0, total_thickness - thickness_top_layer
            )
            relative_bottom_bottom_layer = (
                relative_bottom_top_layer - thickness_bottom_layer
            )

            relative_layer_boundary_elevation = xr.concat(
                [
                    self.full_like(
                        relative_bottom_bottom_layer, fill_value=0, nodata=np.nan
                    ),
                    relative_bottom_top_layer,
                    relative_bottom_bottom_layer,
                ],
                dim=xr.Variable("boundary", np.array([0, 1, 2], dtype=np.int32)),
                compat="equals",
            )
        else:
            relative_bottom_bottom_layer = -total_thickness
            relative_layer_boundary_elevation = xr.concat(
                [
                    self.full_like(
                        relative_bottom_bottom_layer, fill_value=0, nodata=np.nan
                    ),
                    relative_bottom_bottom_layer,
                ],
                dim=xr.Variable(
                    "boundary",
                    np.array([0, 1], dtype=np.int32),
                ),
                compat="equals",
            )

        extended_relative_boundaries: xr.DataArray = resample_like(
            relative_layer_boundary_elevation,
            extended_template,
            method="bilinear",
        )

        # load hydraulic conductivity over the extended bounds
        hydraulic_conductivity = self.data_catalog.fetch(
            "hydraulic_conductivity_globgm"
        ).read()
        hydraulic_conductivity = hydraulic_conductivity.isel(
            get_window(
                hydraulic_conductivity.x,
                hydraulic_conductivity.y,
                extended_bounds,
                buffer=2,
            )
        )

        # because hydraulic conductivity is log-normally distributed, we interpolate the log values
        # after log transformation and then back-transform after interpolation
        raw_k_log: xr.DataArray = np.log(hydraulic_conductivity)
        extended_k_log: xr.DataArray = resample_like(
            raw_k_log,
            extended_template,
            method="bilinear",
        )
        extended_k: xr.DataArray = np.exp(extended_k_log)  # ty:ignore[invalid-assignment]

        if two_layers:
            extended_k = xr.concat(
                [extended_k, extended_k],
                dim=xr.Variable("layer", ["upper", "lower"]),
                compat="equals",
            )
        else:
            extended_k = extended_k.expand_dims(layer=["upper"])

        extended_k.attrs["_FillValue"] = np.nan
        self.set_other(
            extended_k,
            name="groundwater/boundary_hydraulic_conductivity",
        )

        # load specific yield
        specific_yield = self.data_catalog.fetch("specific_yield_aquifer_globgm").read()
        specific_yield = specific_yield.isel(
            get_window(specific_yield.x, specific_yield.y, self.bounds, buffer=2)
        )

        specific_yield = resample_like(
            specific_yield,
            aquifer_top_elevation,
            method="bilinear",
        )

        if two_layers:
            specific_yield = xr.concat(
                [specific_yield, specific_yield],
                dim=xr.Variable("layer", ["upper", "lower"]),
                compat="equals",
            )
        else:
            specific_yield = specific_yield.expand_dims(layer=["upper"])
        self.set_grid(specific_yield, name="groundwater/specific_yield")

        why_map: gpd.GeoDataFrame = self.data_catalog.fetch("why_map").read()
        why_map = why_map[
            why_map["HYGEO2"] != 88
        ]  # remove areas under continuous ice cover
        why_map["aquifer_classification"] = why_map["HYGEO2"] // 10

        why_map_grid: xr.DataArray = rasterize_like(
            why_map,
            column="aquifer_classification",
            raster=aquifer_top_elevation,
            dtype=np.int16,
            nodata=-1,
            all_touched=False,
        )
        why_map_grid = interpolate_na_2d(why_map_grid)
        self.set_grid(why_map_grid, name="groundwater/why_map")

        # the GLOBGM DEM has a slight offset, which we fix here before loading it
        reference_globgm_map = self.data_catalog.fetch("head_upper_layer_globgm").read()

        dem_globgm = self.data_catalog.fetch("dem_globgm").read()
        dem_globgm = dem_globgm.assign_coords(
            x=reference_globgm_map.x.values,
            y=reference_globgm_map.y.values,
        )
        dem_globgm = dem_globgm.isel(
            get_window(dem_globgm.x, dem_globgm.y, extended_bounds, buffer=2)
        )
        dem = convert_nodata(self.grid["landsurface/elevation_m"], new_nodata=np.nan)

        # heads
        head_upper_layer = self.data_catalog.fetch("head_upper_layer_globgm").read()
        head_upper_layer = head_upper_layer.isel(
            get_window(
                head_upper_layer.x, head_upper_layer.y, extended_bounds, buffer=2
            ),
        )
        head_upper_layer = convert_nodata(head_upper_layer, new_nodata=np.nan)

        # Resample raw GLOBGM heads to the extended grid so surrounding boundary cells have values
        extended_head_upper: xr.DataArray = resample_like(
            head_upper_layer,
            extended_template,
            method="bilinear",
        )
        # Ocean/offshore cells in GLOBGM are NaN; assign 0 m sea level
        extended_head_upper = xr.where(
            np.isnan(extended_head_upper), 0.0, extended_head_upper
        )

        relative_head_upper_layer = head_upper_layer - dem_globgm
        relative_head_upper_layer = resample_like(
            relative_head_upper_layer, aquifer_top_elevation, method="bilinear"
        )
        head_upper_layer_resampled = dem + relative_head_upper_layer

        head_lower_layer = self.data_catalog.fetch("head_lower_layer_globgm").read()
        head_lower_layer = head_lower_layer.isel(
            get_window(
                head_lower_layer.x, head_lower_layer.y, extended_bounds, buffer=2
            ),
        )
        head_lower_layer = convert_nodata(head_lower_layer, new_nodata=np.nan)

        extended_head_lower: xr.DataArray = resample_like(
            head_lower_layer,
            extended_template,
            method="bilinear",
        )
        extended_head_lower = xr.where(
            np.isnan(extended_head_lower), 0.0, extended_head_lower
        )

        relative_head_lower_layer = head_lower_layer - dem_globgm
        relative_head_lower_layer = resample_like(
            relative_head_lower_layer, aquifer_top_elevation, method="bilinear"
        )

        # TODO: Make sure head in lower layer is not lower than topography, but why is this needed?
        relative_layer_boundary_elevation_inner = extended_relative_boundaries.isel(
            x=slice(1, -1), y=slice(1, -1)
        )
        relative_head_lower_layer = xr.where(
            relative_head_lower_layer
            < relative_layer_boundary_elevation_inner.isel(boundary=-1),
            relative_layer_boundary_elevation_inner.isel(boundary=-1),
            relative_head_lower_layer,
        )
        head_lower_layer_resampled = dem + relative_head_lower_layer

        # Splice high-resolution DEM-corrected heads into interior of extended heads,
        # preserving the direct GLOBGM heads (and 0 m sea level boundaries) for all surrounding boundary cells.
        valid_inner_upper: np.ndarray = ~np.isnan(head_upper_layer_resampled.values)
        valid_inner_lower: np.ndarray = ~np.isnan(head_lower_layer_resampled.values)
        upper_vals: np.ndarray = extended_head_upper.values.copy()
        lower_vals: np.ndarray = extended_head_lower.values.copy()
        upper_vals[1:-1, 1:-1][valid_inner_upper] = head_upper_layer_resampled.values[
            valid_inner_upper
        ]
        lower_vals[1:-1, 1:-1][valid_inner_lower] = head_lower_layer_resampled.values[
            valid_inner_lower
        ]

        extended_head_upper = xr.DataArray(
            upper_vals.astype(np.float32),
            coords=extended_template.coords,
            dims=extended_template.dims,
        )
        extended_head_lower = xr.DataArray(
            lower_vals.astype(np.float32),
            coords=extended_template.coords,
            dims=extended_template.dims,
        )

        if two_layers:
            boundary_heads: xr.DataArray = xr.concat(
                [extended_head_upper, extended_head_lower],
                dim=xr.Variable("layer", ["upper", "lower"]),
                compat="equals",
            )
        else:
            boundary_heads = extended_head_lower.expand_dims(layer=["upper"])

        boundary_heads.attrs["_FillValue"] = np.nan
        self.set_other(boundary_heads, name="groundwater/boundary_heads")

        # Extended top elevation (DEM) and layer boundary elevations:
        # For interior cells: use aquifer_top_elevation (from high-res DEM)
        # For boundary cells: set 1.0 m above constant boundary head
        extended_top_values = np.full(
            (extended_template.shape[0], extended_template.shape[1]),
            np.nan,
            dtype=np.float32,
        )
        extended_top_values[1:-1, 1:-1] = aquifer_top_elevation.values
        extended_top_values[boundary_mask_data] = (
            boundary_heads.isel(layer=0).values[boundary_mask_data] + 1.0
        )
        extended_top = xr.DataArray(
            extended_top_values,
            coords=extended_template.coords,
            dims=extended_template.dims,
        )
        boundary_layer_boundary_elevation = extended_relative_boundaries + extended_top
        boundary_layer_boundary_elevation.attrs["_FillValue"] = np.nan
        self.set_other(
            boundary_layer_boundary_elevation,
            name="groundwater/boundary_layer_boundary_elevation",
        )
