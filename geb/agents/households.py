import numpy as np
import geopandas as gpd
import calendar
from .general import downscale_volume, AgentBaseClass
from ..store import DynamicArray
from ..hydrology.landcover import (
    SEALED,
    FOREST,
)
import pandas as pd
import os
from os.path import join
from damagescanner.core import object_scanner
import json
import rioxarray
from rasterio.features import shapes
from shapely.geometry import shape


def from_landuse_raster_to_polygon(mask, transform, crs):
    """
    Convert raster data into separate GeoDataFrames for specified land use values.

    Parameters:
    - landuse: An xarray DataArray or similar with land use data and 'x' and 'y' coordinates.
    - values_to_extract: List of integer values to extract (e.g., [0, 1] for forest and agriculture).

    Returns:
    - Geodataframe
    """

    shapes_gen = shapes(mask.astype(np.uint8), mask=mask, transform=transform)

    polygons = []
    for geom, _ in shapes_gen:
        polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)

    return gdf


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.HRU = model.data.HRU
        self.grid = model.data.grid
        self.agents = agents
        self.reduncancy = reduncancy
        self.config = (
            self.model.config["agent_settings"]["households"]
            if "households" in self.model.config["agent_settings"]
            else {}
        )

        if self.model.spinup:
            self.spinup()

    def spinup(self):
        self.var = self.model.store.create_bucket("agents.households.var")

        # Load buildings
        self.var.buildings = gpd.read_file(
            self.model.files["geoms"]["assets/buildings"]
        )
        self.var.buildings["object_type"] = "building_structure"
        self.var.buildings_centroid = gpd.GeoDataFrame(
            geometry=self.var.buildings.centroid
        )
        self.var.buildings_centroid["object_type"] = "building_content"

        # Load roads
        self.var.roads = gpd.read_file(
            self.model.files["geoms"]["assets/roads"]
        ).rename(columns={"highway": "object_type"})

        # Load rail
        self.var.rail = gpd.read_file(self.model.files["geoms"]["assets/rails"])
        self.var.rail["object_type"] = "rail"

        # Load maximum damages
        with open(
            self.model.files["dict"][
                "damage_parameters/flood/rail/main/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_rail = float(json.load(f)["maximum_damage"])
        self.var.rail["maximum_damage"] = self.var.max_dam_rail

        self.var.max_dam_road = {}
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/maximum_damage"),
            (
                "unclassified",
                "damage_parameters/flood/road/unclassified/maximum_damage",
            ),
            ("tertiary", "damage_parameters/flood/road/tertiary/maximum_damage"),
            ("primary", "damage_parameters/flood/road/primary/maximum_damage"),
            (
                "primary_link",
                "damage_parameters/flood/road/primary_link/maximum_damage",
            ),
            ("secondary", "damage_parameters/flood/road/secondary/maximum_damage"),
            (
                "secondary_link",
                "damage_parameters/flood/road/secondary_link/maximum_damage",
            ),
            ("motorway", "damage_parameters/flood/road/motorway/maximum_damage"),
            (
                "motorway_link",
                "damage_parameters/flood/road/motorway_link/maximum_damage",
            ),
            ("trunk", "damage_parameters/flood/road/trunk/maximum_damage"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/maximum_damage"),
        ]

        for road_type, path in road_types:
            with open(self.model.files["dict"][path], "r") as f:
                max_damage = json.load(f)
            self.var.max_dam_road[road_type] = max_damage["maximum_damage"]

        self.var.roads["maximum_damage"] = self.var.roads["object_type"].map(
            self.var.max_dam_road
        )

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/land_use/forest/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_forest = float(json.load(f)["maximum_damage"])

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/land_use/agriculture/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_agriculture = float(json.load(f)["maximum_damage"])

        # Load vulnerability curves
        road_curves = []
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/curve"),
            ("unclassified", "damage_parameters/flood/road/unclassified/curve"),
            ("tertiary", "damage_parameters/flood/road/tertiary/curve"),
            ("primary", "damage_parameters/flood/road/primary/curve"),
            ("primary_link", "damage_parameters/flood/road/primary_link/curve"),
            ("secondary", "damage_parameters/flood/road/secondary/curve"),
            ("secondary_link", "damage_parameters/flood/road/secondary_link/curve"),
            ("motorway", "damage_parameters/flood/road/motorway/curve"),
            ("motorway_link", "damage_parameters/flood/road/motorway_link/curve"),
            ("trunk", "damage_parameters/flood/road/trunk/curve"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/curve"),
        ]

        severity_column = None
        for road_type, path in road_types:
            df = pd.read_parquet(self.model.files["table"][path])

            if severity_column is None:
                severity_column = df["severity"]

            df = df.rename(columns={"damage_ratio": road_type})

            road_curves.append(df[[road_type]])

        self.var.road_curves = pd.concat([severity_column] + road_curves, axis=1)
        self.var.road_curves.set_index("severity", inplace=True)

        self.var.forest_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/land_use/forest/curve"]
        )
        self.var.forest_curve.set_index("severity", inplace=True)
        self.var.forest_curve = self.var.forest_curve.rename(
            columns={"damage_ratio": "forest"}
        )
        self.var.agriculture_curve = pd.read_parquet(
            self.model.files["table"][
                "damage_parameters/flood/land_use/agriculture/curve"
            ]
        )
        self.var.agriculture_curve.set_index("severity", inplace=True)
        self.var.agriculture_curve = self.var.agriculture_curve.rename(
            columns={"damage_ratio": "agriculture"}
        )

        self.var.rail_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/rail/main/curve"]
        )
        self.var.rail_curve.set_index("severity", inplace=True)
        self.var.rail_curve = self.var.rail_curve.rename(
            columns={"damage_ratio": "rail"}
        )

        super().__init__()

        water_demand, efficiency = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_efficiency = efficiency

        locations = np.load(self.model.files["binary"]["agents/households/locations"])[
            "data"
        ]
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)

        self.var.locations = DynamicArray(locations, max_n=self.max_n)

        sizes = np.load(self.model.files["binary"]["agents/households/sizes"])["data"]
        self.var.sizes = DynamicArray(sizes, max_n=self.max_n)

    def flood(self, flood_map, model_root, simulation_root, return_period=None):

        print("Measure:", self.config.get("measure"))

        # Get vulnerability curves based on adaptation scenario
        if self.config.get("measure") not in ["dry_proofing", "wet_proofing"]:
            print("were going for the normal curves ")
            self.var.buildings_structure_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/normal/structure/curve"
                ]
            )
            self.var.buildings_structure_curve.set_index("severity", inplace=True)
            self.var.buildings_structure_curve = (
                self.var.buildings_structure_curve.rename(
                    columns={"damage_ratio": "building_structure"}
                )
            )

            self.var.buildings_content_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/normal/content/curve"
                ]
            )
            self.var.buildings_content_curve.set_index("severity", inplace=True)
            self.var.buildings_content_curve = self.var.buildings_content_curve.rename(
                columns={"damage_ratio": "building_content"}
            )
            with open(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/normal/structure/maximum_damage"
                ],
                "r",
            ) as f:
                self.var.max_dam_buildings_structure = json.load(f)
            self.var.max_dam_buildings_structure = float(
                self.var.max_dam_buildings_structure["maximum_damage"]
            )
            self.var.buildings["maximum_damage"] = self.var.max_dam_buildings_structure
            with open(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/normal/content/maximum_damage"
                ],
                "r",
            ) as f:
                self.var.max_dam_buildings_content = json.load(f)
            self.var.max_dam_buildings_content = float(
                self.var.max_dam_buildings_content["maximum_damage"]
            )
            self.var.buildings_centroid["maximum_damage"] = (
                self.var.max_dam_buildings_content
            )

        if self.config.get("measure") == "dry_proofing":
            print("turning on dryproofing")
            self.var.buildings_structure_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/dry_proofing/structure/curve"
                ]
            )
            self.var.buildings_structure_curve.set_index("severity", inplace=True)
            self.var.buildings_structure_curve = (
                self.var.buildings_structure_curve.rename(
                    columns={"damage_ratio": "building_structure"}
                )
            )

            self.var.buildings_content_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/dry_proofing/content/curve"
                ]
            )
            self.var.buildings_content_curve.set_index("severity", inplace=True)
            self.var.buildings_content_curve = self.var.buildings_content_curve.rename(
                columns={"damage_ratio": "building_content"}
            )
            with open(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/dry_proofing/structure/maximum_damage"
                ],
                "r",
            ) as f:
                self.var.max_dam_buildings_structure = json.load(f)
            self.var.max_dam_buildings_structure = float(
                self.var.max_dam_buildings_structure["maximum_damage"]
            )
            self.var.buildings["maximum_damage"] = self.var.max_dam_buildings_structure
            with open(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/dry_proofing/content/maximum_damage"
                ],
                "r",
            ) as f:
                self.var.max_dam_buildings_content = json.load(f)
            self.var.max_dam_buildings_content = float(
                self.var.max_dam_buildings_content["maximum_damage"]
            )
            self.var.buildings_centroid["maximum_damage"] = (
                self.var.max_dam_buildings_content
            )

        if self.config.get("measure") == "wet_proofing":
            print("turning on wetproofing")
            self.var.buildings_structure_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/wet_proofing/structure/curve"
                ]
            )
            self.var.buildings_structure_curve.set_index("severity", inplace=True)
            self.var.buildings_structure_curve = (
                self.var.buildings_structure_curve.rename(
                    columns={"damage_ratio": "building_structure"}
                )
            )

            self.var.buildings_content_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/wet_proofing/content/curve"
                ]
            )
            self.var.buildings_content_curve.set_index("severity", inplace=True)
            self.var.buildings_content_curve = self.var.buildings_content_curve.rename(
                columns={"damage_ratio": "building_content"}
            )

            with open(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/wet_proofing/structure/maximum_damage"
                ],
                "r",
            ) as f:
                self.var.max_dam_buildings_structure = json.load(f)
            self.var.max_dam_buildings_structure = float(
                self.var.max_dam_buildings_structure["maximum_damage"]
            )
            self.var.buildings["maximum_damage"] = self.var.max_dam_buildings_structure
            with open(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/wet_proofing/content/maximum_damage"
                ],
                "r",
            ) as f:
                self.var.max_dam_buildings_content = json.load(f)
            self.var.max_dam_buildings_content = float(
                self.var.max_dam_buildings_content["maximum_damage"]
            )
            self.var.buildings_centroid["maximum_damage"] = (
                self.var.max_dam_buildings_content
            )

        custom_flood_map = self.config.get("hazards", {}).get("floods", {}).get("custom_flood_map")
        if custom_flood_map:
            flood_path = custom_flood_map
            flood_map = rioxarray.open_rasterio(flood_path)
            flood_map = flood_map.rio.write_crs(28992)

        elif return_period is not None:
            flood_path = join(simulation_root, f"hmax RP {int(return_period)}.tif")
            flood_map = rioxarray.open_rasterio(flood_path)

        else:
            flood_path = join(simulation_root, "hmax.tif")
            flood_map = rioxarray.open_rasterio(flood_path)

        print(f"using this flood map: {flood_path}")

        # Remove rivers from flood map
        rivers_path = join(model_root, "rivers.gpkg")
        rivers = gpd.read_file(rivers_path)
        rivers.set_crs(epsg=4326, inplace=True)
        rivers_projected = rivers.to_crs(flood_map.rio.crs)
        rivers_projected["geometry"] = rivers_projected.buffer(
            rivers_projected["rivwth"] / 2
        )
        rivers_mask = flood_map.raster.geometry_mask(
            gdf=rivers_projected, all_touched=True
        )

        # Load landuse and make turn into polygons
        agriculture = from_landuse_raster_to_polygon(
            self.model.data.HRU.decompress(self.model.data.HRU.var.land_owners != -1),
            self.model.data.HRU.transform,
            self.model.crs,
        )
        agriculture["object_type"] = "agriculture"
        agriculture["maximum_damage"] = self.var.max_dam_agriculture

        agriculture = agriculture.to_crs(flood_map.rio.crs)

        # Load landuse and make turn into polygons
        forest = from_landuse_raster_to_polygon(
            self.model.data.HRU.decompress(
                self.model.data.HRU.var.land_use_type == FOREST
            ),
            self.model.data.HRU.transform,
            self.model.crs,
        )
        forest["object_type"] = "forest"
        forest["maximum_damage"] = self.var.max_dam_forest

        forest = forest.to_crs(flood_map.rio.crs)

        flood_map = flood_map.where(~rivers_mask)
        flood_map = flood_map.fillna(0)
        flood_map = flood_map.where(flood_map != 0, np.nan)

        # Clip the flood map to the region for which we want to know the damages
        region_path = "/scistor/ivm/vbl220/PhD/damages_region.gpkg"
        region = gpd.read_file(region_path)
        region_projected = region.to_crs(flood_map.rio.crs)
        flood_map_clipped = flood_map.rio.clip(
            region_projected.geometry, region_projected.crs
        )

        def compute_damages_by_country(assets, curve, category_name, model_root, return_period=None):
            assets = assets.to_crs(flood_map_clipped.rio.crs)

            # Check for multiple geometry types
            geometry_types = assets.geometry.geom_type.unique()
            print(f"Geometry types in {category_name}: {geometry_types}")

            if "MultiPolygon" in geometry_types:
                assets = assets.explode(index_parts=False).reset_index(drop=True)

            # If only one geometry type, proceed normally
            if len(geometry_types) == 1:
                damages = object_scanner(
                    objects=assets, hazard=flood_map_clipped, curves=curve
                )
                assets["damages"] = damages
                exposed_assets_count = (assets["damages"] > 0).sum()
                print(assets)
                print(f"exposed assets for {category_name} are: {exposed_assets_count}")
                total_damages = damages.sum()
                print(f"damages to {category_name} are: {total_damages}")

                if return_period is not None:
                    filename = f"damages_{category_name}_RP{int(return_period)}.gpkg"
                else:
                    filename = f"damages_{category_name}.gpkg"

                file_path = join(model_root, filename)
                print(file_path)
                
                # Save to GeoPackage
                assets.to_file(file_path, driver="GPKG")
                print(f"Saved assets to {file_path}")

                split_assets = gpd.overlay(
                    assets,
                    gdf_filtered_countries,
                    how="intersection",
                    keep_geom_type=True,
                )
                unmatched_assets = split_assets[split_assets["COUNTRY"].isnull()]

                for country in selection_countries:
                    country_assets = split_assets[split_assets["COUNTRY"] == country]
                    if not country_assets.empty:
                        country_assets = country_assets.to_crs(
                            flood_map_clipped.rio.crs
                        )
                        country_damages = object_scanner(
                            objects=country_assets,
                            hazard=flood_map_clipped,
                            curves=curve,
                        ).sum()
                        print(
                            f"damages to {category_name} ({country}): {country_damages}"
                        )

                return total_damages

            # If multiple geometry types, split and process each separately
            if len(geometry_types) > 1:
                total_damages = 0
                for geom_type in geometry_types:
                    print(f"Processing geometry type: {geom_type}")
                    subset = assets[assets.geometry.geom_type == geom_type]

                    # Process subset as usual
                    damages = object_scanner(
                        objects=subset, hazard=flood_map_clipped, curves=curve
                    )
                    subset_damages = damages.sum()
                    total_damages += subset_damages
                    print(
                        f"damages to {category_name} ({geom_type}) are: {subset_damages}"
                    )

                    # Perform overlay for this subset
                    split_assets = gpd.overlay(
                        subset, gdf_filtered_countries, how="intersection"
                    )
                    unmatched_assets = split_assets[split_assets["COUNTRY"].isnull()]
                    print(f"Unmatched assets for {geom_type}: {unmatched_assets}")

                    for country in selection_countries:
                        country_assets = split_assets[
                            split_assets["COUNTRY"] == country
                        ]
                        if not country_assets.empty:
                            country_assets = country_assets.to_crs(
                                flood_map_clipped.rio.crs
                            )
                            country_damages = object_scanner(
                                objects=country_assets,
                                hazard=flood_map_clipped,
                                curves=curve,
                            ).sum()
                            print(
                                f"damages to {category_name} ({country}, {geom_type}): {country_damages}"
                            )
                print(
                    f"Total damages to {category_name} (all geometry types): {total_damages}"
                )
                return total_damages

        # Filter countries
        all_countries = gpd.read_file("/scistor/ivm/vbl220/PhD/Europe_merged.shp")
        selection_countries = ["Netherlands", "Belgium", "Germany"]
        gdf_filtered_countries = all_countries[
            all_countries["COUNTRY"].isin(selection_countries)
        ]
        if self.var.buildings.crs != flood_map.rio.crs:
            gdf_filtered_countries = gdf_filtered_countries.to_crs(flood_map.rio.crs)

        # Compute damages for each category
        total_damages_agriculture = compute_damages_by_country(
            agriculture, self.var.agriculture_curve, "agriculture", model_root, return_period
        )
        total_damages_forest = compute_damages_by_country(
            forest, self.var.forest_curve, "forest", model_root, return_period
        )
        total_damage_structure = compute_damages_by_country(
            self.var.buildings, self.var.buildings_structure_curve, "building structure", model_root, return_period
        )
        total_damages_content = compute_damages_by_country(
            self.var.buildings_centroid,
            self.var.buildings_content_curve,
            "building content",
            model_root,
            return_period
        )
        total_damages_roads = compute_damages_by_country(
            self.var.roads, self.var.road_curves, "roads", model_root, return_period
        )
        total_damages_rail = compute_damages_by_country(
            self.var.rail, self.var.rail_curve, "rail", model_root, return_period
        )

        # Calculate total flood damages
        total_flood_damages = (
            total_damage_structure
            + total_damages_content
            + total_damages_roads
            + total_damages_rail
            + total_damages_forest
            + total_damages_agriculture
        )
        print(f"the total flood damages are: {total_flood_damages}")
        return total_flood_damages

    def update_water_demand(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        downscale_mask = self.HRU.var.land_use_type != SEALED
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
        water_demand = (
            self.model.domestic_water_demand_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).domestic_water_demand
            * 1_000_000
            / days_in_year
        )
        water_demand = (
            water_demand.rio.set_crs(4326).rio.reproject(
                4326,
                shape=self.model.data.grid.shape,
                transform=self.model.data.grid.transform,
            )
            / (water_demand.rio.transform().a / self.model.data.grid.transform.a) ** 2
        )
        water_demand = downscale_volume(
            water_demand.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.var.land_use_ratio,
        )
        water_demand = self.model.data.HRU.M3toM(water_demand)

        water_consumption = (
            self.model.domestic_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).domestic_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = (
            water_consumption.rio.set_crs(4326).rio.reproject(
                4326,
                shape=self.model.data.grid.shape,
                transform=self.model.data.grid.transform,
            )
            / (water_consumption.rio.transform().a / self.model.data.grid.transform.a)
            ** 2
        )
        water_consumption = downscale_volume(
            water_consumption.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.var.land_use_ratio,
        )
        water_consumption = self.model.data.HRU.M3toM(water_consumption)

        efficiency = np.divide(
            water_consumption,
            water_demand,
            out=np.zeros_like(water_consumption, dtype=float),
            where=water_demand != 0,
        )

        efficiency = self.model.data.to_grid(HRU_data=efficiency, fn="max")

        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        self.var.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.domestic_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.var.current_water_demand = water_demand
            self.var.current_efficiency = efficiency

        assert (
            self.model.current_time - self.var.last_water_demand_update
        ).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the household water demand datasets."
        )
        return self.var.current_water_demand, self.var.current_efficiency

    def step(self) -> None:
        return None

    @property
    def n(self):
        return self.locations.shape[0]
