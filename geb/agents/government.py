# -*- coding: utf-8 -*-
import rioxarray
from .general import AgentBaseClass
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray
import numpy as np
from rasterio.features import rasterize
from rasterio.features import shapes
from shapely.geometry import shape
import numpy as np
from ..hydrology.landcover import (
    FOREST,
    GRASSLAND_LIKE,
    PADDY_IRRIGATED,
    NON_PADDY_IRRIGATED,
)

from ..hydrology.soil import estimate_soil_properties
from geb.HRUs import load_grid


class Government(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.HRU = model.data.HRU
        self.grid = model.data.grid
        self.config = (
            self.model.config["agent_settings"]["government"]
            if "government" in self.model.config["agent_settings"]
            else {}
        )
        self.ratio_farmers_to_provide_subsidies_per_year = 0.05
        if self.model.spinup:
            self.spinup()

        AgentBaseClass.__init__(self)

    def spinup(self) -> None:
        self.var = self.model.store.create_bucket("agents.government.var")

    def provide_subsidies(self) -> None:
        if "subsidies" not in self.config:
            return None
        if self.model.current_timestep == 1:
            for region in self.agents.crop_farmers.borewell_cost_1[1].keys():
                self.agents.crop_farmers.borewell_cost_1[1][region] = [
                    0.5 * x for x in self.agents.crop_farmers.borewell_cost_1[1][region]
                ]
                self.agents.crop_farmers.borewell_cost_2[1][region] = [
                    0.5 * x for x in self.agents.crop_farmers.borewell_cost_2[1][region]
                ]

        return

    def request_flood_cushions(self, reservoirIDs):
        pass

    def set_irrigation_limit(self) -> None:
        if "irrigation_limit" not in self.config:
            return None
        irrigation_limit = self.config["irrigation_limit"]
        if irrigation_limit["per"] == "capita":
            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                self.agents.crop_farmers.var.household_size * irrigation_limit["limit"]
            )
        elif irrigation_limit["per"] == "area":  # limit per m2 of field
            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                self.agents.crop_farmers.field_size_per_farmer
                * irrigation_limit["limit"]
            )
        else:
            raise NotImplementedError(
                "Only 'capita' is implemented for irrigation limit"
            )
        if "min" in irrigation_limit:
            self.agents.crop_farmers.irrigation_limit_m3[
                self.agents.crop_farmers.irrigation_limit_m3 < irrigation_limit["min"]
            ] = irrigation_limit["min"]

    def reforestation(self) -> None:
        if not self.config.get("reforestation", False):
            return None
        if self.model.current_timestep == 1:
            print("running the reforestation scenario")
            # load reforestation map
            # forest_path = "/scistor/ivm/vbl220/PhD/reclassified_landcover_geul_new4.nc"

            # Open dataset and explicitly select `esa_worldcover`
            to_forest = xr.open_dataset(self.model.files["forcing"]["hydrodynamics/esa_worldcover_reforestation_scenario"])["esa_worldcover"]

            # Convert to a spatially-aware raster dataset
            to_forest = to_forest.rio.write_crs("EPSG:28992").squeeze()

            # If needed, reproject to match other layers (e.g., EPSG:4326)
            to_forest = to_forest.rio.reproject("EPSG:4326")

            # Ensure values are in correct range and type
            data = np.clip(to_forest.values, 0, 255).astype(np.uint8)

            # Debug information
            print(f"Shape: {to_forest.shape}, Dtype: {to_forest.dtype}")
            print(f"Min: {data.min()}, Max: {data.max()}")

            y_coords = to_forest.coords["y"].values
            x_coords = to_forest.coords["x"].values

            transform = rasterio.transform.from_origin(
                x_coords[0],
                y_coords[0],
                abs(x_coords[1] - x_coords[0]),
                abs(y_coords[1] - y_coords[0]),
            )

            mask = data == 10
            shapes_gen = shapes(data, mask=mask, transform=transform)

            polygons = []
            for geom, value in shapes_gen:
                if value == 10:
                    polygons.append(shape(geom))

            forest = gpd.GeoDataFrame(
                {"value": [10] * len(polygons), "geometry": polygons}
            )
            forest.set_crs(epsg=4326, inplace=True)

            output_vector_path = "/scistor/ivm/vbl220/PhD/forestation_vectorized.gpkg"
            forest.to_file(output_vector_path)

            # then we rasterize this file to match the HRUs
            forest = rasterize(
                [(shape(geom), 1) for geom in forest.geometry],
                out_shape=self.model.data.HRU.shape,
                transform=self.model.data.HRU.transform,
                fill=False,
                dtype="uint8",  # bool is not supported, so we use uint8 and convert to bool
            ).astype(bool)
            # do not create forests outside the study area
            forest[self.model.data.HRU.mask] = False
            # only create forests in grassland or agricultural areas
            forest[
                ~np.isin(
                    self.model.data.HRU.decompress(
                        self.model.data.HRU.var.land_use_type
                    ),
                    [GRASSLAND_LIKE, PADDY_IRRIGATED, NON_PADDY_IRRIGATED],
                )
            ] = False

            import matplotlib.pyplot as plt

            plt.imshow(forest)
            plt.savefig("forest.png")

            new_forest_HRUs = np.unique(
                self.model.data.HRU.var.unmerged_HRU_indices[forest]
            )
            # set the land use type to forest
            self.model.data.HRU.var.land_use_type[new_forest_HRUs] = FOREST

            # get the farmers corresponding to the new forest HRUs
            farmers_with_land_converted_to_forest = np.unique(
                self.model.data.HRU.var.land_owners[new_forest_HRUs]
            )
            farmers_with_land_converted_to_forest = (
                farmers_with_land_converted_to_forest
            )[farmers_with_land_converted_to_forest != -1]

            print(farmers_with_land_converted_to_forest)

            HRUs_removed_farmers = self.model.agents.crop_farmers.remove_agents(
                farmers_with_land_converted_to_forest, new_land_use_type=FOREST
            )

            new_forest_HRUs = np.unique(
                np.concatenate([new_forest_HRUs, HRUs_removed_farmers])
            )
           
            print("loading soil parameter input files")
          
            self.HRU.var.soil_organic_carbon = self.HRU.compress(
                load_grid(
                    self.model.files["subgrid"]["soil/forestation_soil_organic_carbon"],
                    layer=None,
                ),
                method="mean",
            )
          
            self.HRU.var.bulk_density = self.HRU.compress(
                load_grid(
                    self.model.files["subgrid"]["soil/forestation_bulk_density"],
                    layer=None,
                ),
                method="mean",
            )
         
            print("changing soil parameters")
            # Estimate soil properties
            estimate_soil_properties(
                self,
                soil_layer_height=self.HRU.var.soil_layer_height,
                soil_organic_carbon=self.HRU.var.soil_organic_carbon,
                bulk_density=self.HRU.var.bulk_density,
                sand=self.HRU.var.sand,
                clay=self.HRU.var.clay,
                silt=self.HRU.var.silt,
            )
            print("soil parameters should be changed")

    def cropland_to_grassland(self) -> None:
        if not self.config.get("cropland_to_grassland", False):
            return None
        if self.model.current_timestep == 1:
            print("running the cropland to grassland conversion scenario")

            # Load scenario 
            # new_grassland_path = "/scistor/ivm/vbl220/PhD/reclassified_landcover_geul_cropland_conversion.nc"

            # Open dataset and explicitly select `esa_worldcover`
            to_grasland = xr.open_dataset(self.model.files["forcing"]["hydrodynamics/esa_worldcover_cropland_scenario"], engine="netcdf4")["lulc"]

            # Convert to a spatially-aware raster dataset
            to_grasland = to_grasland.rio.write_crs("EPSG:28992").squeeze()
            to_grasland = to_grasland.rio.reproject("EPSG:4326")

            # Ensure values are in correct range and type
            data = np.clip(to_grasland.values, 0, 255).astype(np.uint8)

            # Debug information
            print(f"Shape: {to_grasland.shape}, Dtype: {to_grasland.dtype}")
            print(f"Min: {data.min()}, Max: {data.max()}")


            y_coords = to_grasland.coords["y"].values
            x_coords = to_grasland.coords["x"].values

            transform = rasterio.transform.from_origin(
                x_coords[0],
                y_coords[0],
                abs(x_coords[1] - x_coords[0]),
                abs(y_coords[1] - y_coords[0]),
            )

            mask = data == 30
            shapes_gen = shapes(data, mask=mask, transform=transform)

            polygons = []
            for geom, value in shapes_gen:
                if value == 30:
                    polygons.append(shape(geom))

            grassland = gpd.GeoDataFrame(
                {"value": [30] * len(polygons), "geometry": polygons}
            )
            grassland.set_crs(epsg=4326, inplace=True)

            output_vector_path = "/scistor/ivm/vbl220/PhD/grassland_vectorized.gpkg"
            grassland.to_file(output_vector_path)

            # then we rasterize this file to match the HRUs
            grassland = rasterize(
                [(shape(geom), 1) for geom in grassland.geometry],
                out_shape=self.model.data.HRU.shape,
                transform=self.model.data.HRU.transform,
                fill=False,
                dtype="uint8",  # bool is not supported, so we use uint8 and convert to bool
            ).astype(bool)
            # do not create forests outside the study area
            grassland[self.model.data.HRU.mask] = False
            # only create forests in grassland or agricultural areas
            grassland[
                ~np.isin(
                    self.model.data.HRU.decompress(
                        self.model.data.HRU.var.land_use_type
                    ),
                    [GRASSLAND_LIKE, PADDY_IRRIGATED, NON_PADDY_IRRIGATED],
                )
            ] = False

            import matplotlib.pyplot as plt

            plt.imshow(grassland)
            plt.savefig("grassland.png")

            # new_grassland_HRUs = np.unique(
            #     self.model.data.HRU.var.unmerged_HRU_indices[GRASSLAND_LIKE]
            # )
            # print(new_grassland_HRUs)
            # self.model.data.HRU.var.land_use_type[new_grassland_HRUs] = GRASSLAND_LIKE
            # print(self.model.data.HRU.var.land_use_type)
            # farmers_with_land_converted_to_grassland = np.unique(
            #     self.model.data.HRU.var.land_owners[new_grassland_HRUs]
            # )
            # print(farmers_with_land_converted_to_grassland)
            # farmers_with_land_converted_to_grassland = (
            #     farmers_with_land_converted_to_grassland
            # )[farmers_with_land_converted_to_grassland != -1]

            # print(farmers_with_land_converted_to_grassland)

            # HRUs_removed_farmers = self.model.agents.crop_farmers.remove_agents(
            #     farmers_with_land_converted_to_grassland, new_land_use_type=GRASSLAND_LIKE
            # )

            # new_grassland_HRUs = np.unique(
            #     np.concatenate([new_grassland_HRUs, HRUs_removed_farmers])
            # )
           
            print("loading soil parameter input files")
          
            self.HRU.var.soil_organic_carbon = self.HRU.compress(
                load_grid(
                    self.model.files["subgrid"]["soil/grassland_soil_organic_carbon"],
                    layer=None,
                ),
                method="mean",
            )
          
            self.HRU.var.bulk_density = self.HRU.compress(
                load_grid(
                    self.model.files["subgrid"]["soil/grassland_bulk_density"],
                    layer=None,
                ),
                method="mean",
            )
         
            print("changing soil parameters")
            # Estimate soil properties
            estimate_soil_properties(
                self,
                soil_layer_height=self.HRU.var.soil_layer_height,
                soil_organic_carbon=self.HRU.var.soil_organic_carbon,
                bulk_density=self.HRU.var.bulk_density,
                sand=self.HRU.var.sand,
                clay=self.HRU.var.clay,
                silt=self.HRU.var.silt,
            )
            print("soil parameters should be changed")

    def step(self) -> None:
        """This function is run each timestep."""
        self.set_irrigation_limit()
        self.provide_subsidies()
        self.reforestation()
        self.cropland_to_grassland()
