# -*- coding: utf-8 -*-
import rioxarray
from .general import AgentBaseClass


class Government(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["government"]
            if "government" in self.model.config["agent_settings"]
            else {}
        )
        self.ratio_farmers_to_provide_subsidies_per_year = 0.05

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
            import geopandas as gpd
            import rasterio
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

            self.var = self.model.data.HRU

            # load reforestation map
            forest_path = "/scistor/ivm/vbl220/PhD/reclassified_landuse_only_belgium.nc"
            to_forest = rioxarray.open_rasterio(forest_path, masked=True)
            # data = to_forest["esa_worldcover"].values
            data = to_forest.values
            data = data.astype(np.uint8)

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
                    self.model.data.HRU.decompress(self.var.land_use_type),
                    [GRASSLAND_LIKE, PADDY_IRRIGATED, NON_PADDY_IRRIGATED],
                )
            ] = False

            import matplotlib.pyplot as plt

            plt.imshow(forest)
            plt.savefig("forest.png")

            new_forest_HRUs = np.unique(
                self.model.data.HRU.unmerged_HRU_indices[forest]
            )

            # set the land use type to forest
            self.var.land_use_type[new_forest_HRUs] = FOREST

            # get the farmers corresponding to the new forest HRUs
            farmers_with_land_converted_to_forest = np.unique(
                self.model.data.HRU.land_owners[new_forest_HRUs]
            )
            farmers_with_land_converted_to_forest = (
                farmers_with_land_converted_to_forest
            )[farmers_with_land_converted_to_forest != -1]

            HRUs_removed_farmers = self.model.agents.crop_farmers.remove_agents(
                farmers_with_land_converted_to_forest, new_land_use_type=FOREST
            )

            new_forest_HRUs = np.unique(
                np.concatenate([new_forest_HRUs, HRUs_removed_farmers])
            )

    def step(self) -> None:
        """This function is run each timestep."""
        self.set_irrigation_limit()
        self.provide_subsidies()
        self.reforestation()
