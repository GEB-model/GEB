import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import xarray as xr
from hydromt.exceptions import NoDataException
from pyflwdir import FlwdirRaster
from scipy.ndimage import value_indices

from geb.hydrology.lakes_reservoirs import LAKE, RESERVOIR

from ..workflows.hydrography import (
    create_river_raster_from_river_lines,
    get_downstream_subbasins,
    get_rivers,
    get_subbasins_geometry,
    get_SWORD_river_widths,
    get_SWORD_translation_IDs_and_lenghts,
    get_upstream_subbasin_ids,
)


class Hydrography:
    def __init__(self):
        pass

    def setup_mannings(self) -> None:
        """
        Sets up the Manning's coefficient for the model.

        Notes
        -----
        This method sets up the Manning's coefficient for the model by calculating the coefficient based on the cell area
        and topography of the grid. It first calculates the upstream area of each cell in the grid using the
        `routing/upstream_area` attribute of the grid. It then calculates the coefficient using the formula:

            C = 0.025 + 0.015 * (2 * A / U) + 0.030 * (Z / 2000)

        where C is the Manning's coefficient, A is the cell area, U is the upstream area, and Z is the elevation of the cell.

        The resulting Manning's coefficient is then set as the `routing/mannings` attribute of the grid using the
        `set_grid()` method.
        """
        self.logger.info("Setting up Manning's coefficient")
        a = (2 * self.grid["cell_area"]) / self.grid["routing/upstream_area"]
        a = xr.where(a < 1, a, 1, keep_attrs=True)
        b = self.grid["routing/outflow_elevation"] / 2000
        b = xr.where(b < 1, b, 1, keep_attrs=True)

        mannings = 0.025 + 0.015 * a + 0.030 * b
        mannings.attrs["_FillValue"] = np.nan

        self.set_grid(mannings, "routing/mannings")

    def set_routing_subbasins(self, river_graph, sink_subbasin_ids):
        # always make a list of the subbasin ids, such that the function always gets the same type of input
        if not isinstance(sink_subbasin_ids, (list, set)):
            sink_subbasin_ids = [sink_subbasin_ids]

        subbasin_ids = get_upstream_subbasin_ids(river_graph, sink_subbasin_ids)
        subbasin_ids.update(sink_subbasin_ids)

        downstream_subbasins = get_downstream_subbasins(river_graph, sink_subbasin_ids)
        subbasin_ids.update(downstream_subbasins)

        subbasins = get_subbasins_geometry(self.data_catalog, subbasin_ids)
        subbasins["is_downstream_outflow_subbasin"] = pd.Series(
            True, index=downstream_subbasins
        ).reindex(subbasins.index, fill_value=False)

        subbasins["associated_upstream_basins"] = pd.Series(
            downstream_subbasins.values(), index=downstream_subbasins
        ).reindex(subbasins.index, fill_value=[])

        self.set_geoms(subbasins, name="routing/subbasins")

    def setup_hydrography(self):
        original_d8_elevation = self.other["original_d8_elevation"]
        original_d8_ldd = self.other["original_d8_flow_directions"]
        original_d8_ldd_data = original_d8_ldd.values

        flow_raster_original = pyflwdir.from_array(
            original_d8_ldd_data,
            ftype="d8",
            transform=original_d8_ldd.rio.transform(recalc=True),
            latlon=True,  # hydrography is specified in latlon
            mask=original_d8_ldd_data
            != original_d8_ldd.attrs[
                "_FillValue"
            ],  # this mask is True within study area
        )

        original_upstream_area = self.full_like(
            original_d8_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        original_upstream_area_data = flow_raster_original.upstream_area(
            unit="m2"
        ).astype(np.float32)
        original_upstream_area_data[original_upstream_area_data == -9999.0] = np.nan
        original_upstream_area.data = original_upstream_area_data
        self.set_other(original_upstream_area, name="original_d8_upstream_area")

        elevation_coarsened = original_d8_elevation.coarsen(
            x=self.ldd_scale_factor,
            y=self.ldd_scale_factor,
            boundary="exact",
            coord_func="mean",
        )

        # elevation (we only set this later, because it has to be done after setting the mask)
        elevation = elevation_coarsened.mean()
        elevation = self.snap_to_grid(elevation, self.grid["mask"])

        self.set_grid(elevation, name="landsurface/elevation")

        elevation_std = elevation_coarsened.std()
        elevation_std = self.snap_to_grid(elevation_std, self.grid["mask"])
        self.set_grid(
            elevation_std,
            name="landsurface/elevation_standard_deviation",
        )

        # outflow elevation
        outflow_elevation = elevation_coarsened.min()
        outflow_elevation = self.snap_to_grid(outflow_elevation, self.grid["mask"])
        self.set_grid(outflow_elevation, name="routing/outflow_elevation")

        slope = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        slope.raster.set_nodata(np.nan)
        slope_data = pyflwdir.dem.slope(
            elevation.values,
            nodata=np.nan,
            latlon=True,
            transform=elevation.rio.transform(recalc=True),
        )
        # set slope to zero on the mask boundary
        slope_data[np.isnan(slope_data) & (~self.grid["mask"].data)] = 0
        slope.data = slope_data
        self.set_grid(slope, name="landsurface/slope")

        flow_raster_idxs_ds = self.grid["flow_raster_idxs_ds"].compute()
        flow_raster = FlwdirRaster(
            flow_raster_idxs_ds.values.ravel(),
            shape=flow_raster_idxs_ds.shape,
            transform=flow_raster_idxs_ds.rio.transform(recalc=True),
            ftype="d8",
            latlon=True,
        )

        # flow direction
        ldd = self.full_like(
            outflow_elevation, fill_value=255, nodata=255, dtype=np.uint8
        )
        ldd.data = flow_raster.to_array(ftype="ldd")
        self.set_grid(ldd, name="routing/ldd")

        # upstream area
        upstream_area = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        upstream_area_data = flow_raster.upstream_area(unit="m2").astype(np.float32)
        upstream_area_data[upstream_area_data == -9999.0] = np.nan
        upstream_area.data = upstream_area_data
        self.set_grid(upstream_area, name="routing/upstream_area")

        # river length
        river_length = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_length_data = flow_raster_original.subgrid_rivlen(
            self.grid["idxs_outflow"].values, unit="m", direction="down"
        )
        river_length_data[river_length_data == -9999.0] = np.nan
        river_length.data = river_length_data
        self.set_grid(river_length, name="routing/river_length")

        # river slope
        river_slope = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_slope_data = flow_raster_original.subgrid_rivslp(
            self.grid["idxs_outflow"].values, original_d8_elevation
        )
        river_slope_data[river_slope_data == -9999.0] = np.nan
        river_slope.data = river_slope_data
        self.set_grid(
            river_slope,
            name="routing/river_slope",
        )

        # river width
        subbasin_ids = self.geoms["routing/subbasins"].index

        self.logger.info("Retrieving river data")
        rivers = get_rivers(self.data_catalog, subbasin_ids)

        self.logger.info("Processing river data")
        # remove all rivers that are both shorter than 1 km and have no upstream river
        rivers = rivers[~((rivers["lengthkm"] < 1) & (rivers["maxup"] == 0))]

        rivers = rivers.join(
            self.geoms["routing/subbasins"][
                ["is_downstream_outflow_subbasin", "associated_upstream_basins"]
            ],
            how="left",
        )

        COMID_IDs_raster_data = create_river_raster_from_river_lines(
            rivers, self.grid["idxs_outflow"], original_d8_elevation
        )

        assert set(
            np.unique(COMID_IDs_raster_data[COMID_IDs_raster_data != -1])
        ) == set(np.unique(COMID_IDs_raster_data[COMID_IDs_raster_data != -1]))

        # Derive the xy coordinates of the river network. Here the coordinates
        # are the PIXEL coordinates for the coarse drainage network.
        rivers["hydrography_xy"] = [[]] * len(rivers)
        rivers["hydrography_upstream_area_m2"] = [[]] * len(rivers)
        xy_per_river_segment = value_indices(COMID_IDs_raster_data, ignore_value=-1)
        for COMID, (ys, xs) in xy_per_river_segment.items():
            upstream_area = upstream_area_data[ys, xs]
            up_to_downstream_ids = np.argsort(upstream_area)
            upstream_area_sorted = upstream_area[up_to_downstream_ids]
            ys = ys[up_to_downstream_ids]
            xs = xs[up_to_downstream_ids]
            rivers.at[COMID, "hydrography_xy"] = list(zip(xs, ys))
            rivers.at[COMID, "hydrography_upstream_area_m2"] = (
                upstream_area_sorted.tolist()
            )

        COMID_IDs_raster = self.full_like(
            outflow_elevation, fill_value=-1, nodata=-1, dtype=np.int32
        )
        COMID_IDs_raster.data = COMID_IDs_raster_data
        self.set_grid(COMID_IDs_raster, name="routing/river_ids")

        SWORD_reach_IDs, SWORD_reach_lengths = get_SWORD_translation_IDs_and_lenghts(
            self.data_catalog, rivers
        )

        SWORD_river_widths = get_SWORD_river_widths(self.data_catalog, SWORD_reach_IDs)
        MINIMUM_RIVER_WIDTH = 3.0
        rivers["width"] = np.nansum(
            SWORD_river_widths * SWORD_reach_lengths, axis=0
        ) / np.nansum(SWORD_reach_lengths, axis=0)
        # ensure that all rivers with a SWORD ID have a width
        assert (~np.isnan(rivers["width"][(SWORD_reach_IDs != -1).any(axis=0)])).all()

        # set initial width guess where width is not available from SWORD
        rivers.loc[rivers["width"].isnull(), "width"] = (
            rivers[rivers["width"].isnull()]["uparea"] / 10
        )
        rivers["width"] = rivers["width"].clip(lower=float(MINIMUM_RIVER_WIDTH))

        self.set_geoms(rivers, name="routing/rivers")

        river_width_data = np.vectorize(
            lambda ID: rivers["width"].to_dict().get(ID, float(MINIMUM_RIVER_WIDTH))
        )(COMID_IDs_raster).astype(np.float32)

        river_width = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_width.data = river_width_data
        self.set_grid(river_width, name="routing/river_width")

    def setup_waterbodies(
        self,
        command_areas=None,
        custom_reservoir_capacity=None,
    ):
        """
        Sets up the waterbodies for GEB.

        Notes
        -----
        This method sets up the waterbodies for GEB. It first retrieves the waterbody data from the
        specified data catalog and sets it as a geometry in the model. It then rasterizes the waterbody data onto the model
        grid and the subgrid using the `rasterize` method of the `raster` object. The resulting grids are set as attributes
        of the model with names of the form 'waterbodies/{grid_name}'.

        The method also retrieves the reservoir command area data from the data catalog and calculates the area of each
        command area that falls within the model region. The `waterbody_id` key is used to do the matching between these
        databases. The relative area of each command area within the model region is calculated and set as a column in
        the waterbody data. The method sets all lakes with a command area to be reservoirs and updates the waterbody data
        with any custom reservoir capacity data from the data catalog.

        TODO: Make the reservoir command area data optional.

        The resulting waterbody data is set as a table in the model with the name 'waterbodies/waterbody_data'.
        """
        self.logger.info("Setting up waterbodies")
        dtypes = {
            "waterbody_id": np.int32,
            "waterbody_type": np.int32,
            "volume_total": np.float64,
            "average_discharge": np.float64,
            "average_area": np.float64,
        }
        try:
            waterbodies = self.data_catalog.get_geodataframe(
                "hydro_lakes",
                geom=self.region,
                predicate="intersects",
                variables=[
                    "waterbody_id",
                    "waterbody_type",
                    "volume_total",
                    "average_discharge",
                    "average_area",
                ],
            )
            waterbodies = waterbodies.astype(dtypes)
            hydrolakes_to_geb = {
                1: LAKE,
                2: RESERVOIR,
            }
            waterbodies["waterbody_type"] = waterbodies["waterbody_type"].map(
                hydrolakes_to_geb
            )
        except NoDataException:
            self.logger.info(
                "No water bodies found in domain, skipping water bodies setup"
            )
            waterbodies = gpd.GeoDataFrame(
                columns=[
                    "waterbody_id",
                    "waterbody_type",
                    "volume_total",
                    "average_discharge",
                    "average_area",
                    "geometry",
                ],
                crs=4326,
            )
            waterbodies = waterbodies.astype(dtypes)
            water_body_id = xr.zeros_like(self.grid["mask"], dtype=np.int32)
            sub_water_body_id = xr.zeros_like(self.subgrid["mask"], dtype=np.int32)
        else:
            water_body_id = self.grid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=-1,
                all_touched=True,
                dtype=np.int32,
            )
            sub_water_body_id = self.subgrid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=-1,
                all_touched=True,
                dtype=np.int32,
            )

        water_body_id.attrs["_FillValue"] = -1
        sub_water_body_id.attrs["_FillValue"] = -1

        self.set_grid(water_body_id, name="waterbodies/water_body_id")
        self.set_subgrid(sub_water_body_id, name="waterbodies/sub_water_body_id")

        waterbodies["volume_flood"] = waterbodies["volume_total"]

        if command_areas:
            command_areas = self.data_catalog.get_geodataframe(
                command_areas, geom=self.region, predicate="intersects"
            )
            command_areas = command_areas[
                ~command_areas["waterbody_id"].isnull()
            ].reset_index(drop=True)
            command_areas["waterbody_id"] = command_areas["waterbody_id"].astype(
                np.int32
            )

            self.set_grid(
                self.grid.raster.rasterize(
                    command_areas,
                    col_name="waterbody_id",
                    nodata=-1,
                    all_touched=True,
                    dtype=np.int32,
                ),
                name="waterbodies/command_area",
            )
            self.set_subgrid(
                self.subgrid.raster.rasterize(
                    command_areas,
                    col_name="waterbody_id",
                    nodata=-1,
                    all_touched=True,
                    dtype=np.int32,
                ),
                name="waterbodies/subcommand_areas",
            )

            # set all lakes with command area to reservoir
            waterbodies.loc[
                waterbodies.index.isin(command_areas["waterbody_id"]), "waterbody_type"
            ] = RESERVOIR
        else:
            command_areas = self.full_like(
                self.grid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )
            subcommand_areas = self.full_like(
                self.subgrid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )

            self.set_grid(command_areas, name="waterbodies/command_area")
            self.set_subgrid(subcommand_areas, name="waterbodies/subcommand_areas")

        if custom_reservoir_capacity:
            custom_reservoir_capacity = self.data_catalog.get_dataframe(
                custom_reservoir_capacity
            )
            custom_reservoir_capacity = custom_reservoir_capacity[
                custom_reservoir_capacity.index != -1
            ]

            waterbodies.set_index("waterbody_id", inplace=True)
            waterbodies.update(custom_reservoir_capacity)
            waterbodies.reset_index(inplace=True)

        # spatial dimension is not required anymore, so drop it.
        waterbodies = waterbodies.drop("geometry", axis=1)

        assert "waterbody_id" in waterbodies.columns, "waterbody_id is required"
        assert "waterbody_type" in waterbodies.columns, "waterbody_type is required"
        assert "volume_total" in waterbodies.columns, "volume_total is required"
        assert "average_discharge" in waterbodies.columns, (
            "average_discharge is required"
        )
        assert "average_area" in waterbodies.columns, "average_area is required"
        self.set_table(waterbodies, name="waterbodies/waterbody_data")
