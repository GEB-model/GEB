import numpy as np
import geopandas as gpd
import pyproj
import calendar
from .general import AgentArray, downscale_volume
from . import AgentBaseClass


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.reduncancy = reduncancy

        super().__init__()

        water_demand, efficiency = self.update_water_demand()
        self.current_water_demand = water_demand
        self.current_efficiency = efficiency

    def initiate(self) -> None:
        locations = np.load(
            self.model.model_structure["binary"]["agents/households/locations"]
        )["data"]
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)

        self.locations = AgentArray(locations, max_n=self.max_n)

        sizes = np.load(
            self.model.model_structure["binary"]["agents/households/sizes"]
        )["data"]
        self.sizes = AgentArray(sizes, max_n=self.max_n)

        self.flood_depth = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=0, dtype=np.float32
        )
        self.risk_perception = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=1, dtype=np.float32
        )

        self.buildings = gpd.read_file(
            self.model.model_structure["geoms"]["assets/buildings"]
        )

    def flood(self, flood_map):
        self.flood_depth.fill(0)  # Reset flood depth for all households

        import matplotlib.pyplot as plt

        plt.figure()

        flood_map.plot()
        plt.savefig("flood.png")

        transformer = pyproj.Transformer.from_crs(
            4326, flood_map.raster.crs, always_xy=True
        )
        x, y = transformer.transform(self.locations[:, 0], self.locations[:, 1])

        forward_transform = flood_map.raster.transform
        backward_transform = ~forward_transform

        pixel_x, pixel_y = backward_transform * (x, y)
        pixel_x = pixel_x.astype(int)  # TODO: Should I add 0.5?
        pixel_y = pixel_y.astype(int)  # TODO: Should I add 0.5?

        # Create a mask that includes only the pixels inside the grid
        mask = (
            (pixel_x >= 0)
            & (pixel_x < flood_map.shape[1])
            & (pixel_y >= 0)
            & (pixel_y < flood_map.shape[0])
        )

        flood_depth_per_household = flood_map.values[pixel_y[mask], pixel_x[mask]]
        self.flood_depth[mask] = flood_depth_per_household > 0

        self.risk_perception[(self.flood_depth > 0)] *= 10

        print("mean risk perception", self.risk_perception.mean())

        return None

    def update_water_demand(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        downscale_mask = self.model.data.HRU.land_use_type != 4
        if self.model.use_gpu:
            downscale_mask = downscale_mask.get()
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
        water_demand = (
            self.model.domestic_water_demand_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).domestic_water_demand
            * 1_000_000
            / days_in_year
        )
        water_demand = downscale_volume(
            self.model.domestic_water_demand_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_demand = cp.array(water_demand)
        water_demand = self.model.data.HRU.M3toM(water_demand)

        water_consumption = (
            self.model.domestic_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).domestic_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = downscale_volume(
            self.model.domestic_water_consumption_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_consumption = cp.array(water_consumption)
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
        self.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.domestic_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.current_water_demand = water_demand
            self.current_efficiency = efficiency

        assert (self.model.current_time - self.last_water_demand_update).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the household water demand datasets."
        )
        return self.current_water_demand, self.current_efficiency

    def step(self) -> None:
        self.risk_perception *= self.risk_perception
        return None

    @property
    def n(self):
        return self.locations.shape[0]
