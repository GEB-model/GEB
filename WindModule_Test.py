###MODIFIED LOAD_FLOOD_MAPs -> LOAD_WIND_MAPs

# this considers the 50 and 100 return period for windstoms 
# create new folder in the geul base directory called "wind_maps"
# and copy the wind maps there (maps in tif format)

def load_wind_maps(self):
    """
    Load wind maps for the 50 and 100 return periods.
    """
    self.windstorm_return_periods = np.array(
        self.model.config["hazards"]["windstorm"]["return_periods"]
        ).astype(int
    )

    windstorm_maps ={}
    windstorm_path = self.model.output_folder_root / "wind_maps"
    for return_period in self.windstorm_return_periods:
        file_path = windstorm_path / f"{return_period}.tif" #depends on de name of the file
        ds = xr.open_dataarray(file_path, engine ="rasterio")

        #Reproject to match flood map CRS if needed (check if this is necessary)
        if ds.rio.crs != self.flood_maps["crs"]:
            ds = ds.rio.reproject(self.flood_maps["crs"])

            windstorm_maps[return_period] = ds

    windstorm_maps["crs"] = self.flood_maps["crs"]
    windstorm_maps["gdal_geotransform"] = (
        windstorm_maps[return_period].rio.transform().to_gdal()
    )

    self.windstorm_maps = windstorm_maps

    ## this might need and additon in the configuration file like:

    """hazards":{
        "windstorm": {
            "return_periods": [50, 100],
            "folder": "wind_maps"
        }
    }"""

    # this function is called in the __init__ function in households


def load_windstorm_damage_curves(self):
###MODIFIED load_damage_curves() -> load_windstorm_damage_curves()
    """ 
    The function loads the damage curves for windstorms need to look for a windstorm damage function.
    We could use Koks & Haer., 2020 similar to what they did in the CLIMAAX handbook. (Ted WCR)
    For the purpose of building this code we will focus only on concrete building type when selecting the damage curves.
    """
    wind_road_curves = []
    # This path should be created with the windstorm vulnerability curves by Koks & Haer., 2020
    wind_road_types = [
        ("residential", "damage_parameters/windstorm/road/residential/curve"),
    ]

    wind_severity_column = None
    for road_type, path in wind_road_types:
        df = pd.read_parquet(self.model.files["table"][path])

        if wind_severity_column is None:
            wind_severity_column = df["wind_severity"]
        
        df = df.rename(columns = {"wind_damage_ratio": road_type})
        wind_road_curves.append(df[[road_type]])

    self.var.wind_road_curves = pd.concat([wind_severity_column] + wind_road_curves, axis=1)
    self.var.wind_road_curves.set_index("wind_severity", inplace=True)

    # Use only the residential concrete building type for now
    self.var.wind_buildings_structure_curve = pd.read_parquet(
        self.model.files["table"]["damage_parameters/windstorm/buildings/concrete/curve"]
    )
    
def create_wind_damage_interpolators(self):
    """
    Create interpolators for windstorm damage curves.
    For now only concrete and unprotected buildings, no adaptation measures are considered.
    """
    self. windstorm_building_curve_interpolator = interpolate.interp1d(
        x=self.wind_building_curve.index,
        y=self.wind_building_curve["residential"],
        bounds_error=False,
        #fill_value = "extrapolate
    )

def calculate_building_windstorm_damages(self):
    """ 
    This function caclulates the windstorm damages for the households in the model.
    It iterates over the return periods and calculates the damages for each household.  """
    wind_damages = np.zeros((self.windstorm_return_periods.size, self.n), np.float32)

    for i, return_period in enumerate(self.windstorm_return_periods):
        wind_map: xr.DataArray = self.windstorm_maps[return_period]

        # All buildings are affected (no spatial filtering needed) - should we consider the probability here?
        buildings: gpd.GeodataFrame = self.buildings.copy().to_crs(
            self.windstorm_maps["crs"]
        )

        #Calculate the damages for each building
        wind_damage_output: pd.DataFrame = pd.Series = VectorScanner(
            feature_file=buildings.rename(
                # Is the maximum damage link to the effect of windstorms or to the general reconstruction value?
                columns={"maximum_wind_damage_m2":"maximum_wind_damage"}
            ),
            hazard_file=wind_map,
            # check continuity of buildings_structure_windstorm_curve
            curve_path=self.buildings_structure_windstorm_curve,
            gridded=False,
        )
        # maybe i need to change the name of the column to "damage", where does this name come from?
        total_wind_damage_structure = wind_damage_output["wind_damage"].sum()
        print(f"Windstorm damages to building structure rp{return_period} are: {total_wind_damage_structure}")

        # Save the damages to the dataframe
        wind_damage_output = wind_damage_output[["osm_id", "osm_way_id", "wind_damage"]]

        #Assign damages to agents
        for _, row in wind_damage_output.iterrows():
            wind_damage = row["wind_damage"]
            if row["osm_id"] is not None:
                osm_id = int(row["osm_id"])
                idx_agents_in_building = np.where(self.var.osm_id == osm_id)[0]
                wind_damages[i, idx_agents_in_building] = wind_damage
                
            else: 
                osm_way_id = int(row["osm_way_id"])
                idx_agents_in_building_way = np.where(
                    self.var.osm_way_id == osm_way_id
                )[0]
                wind_damages[i, idx_agents_in_building_way] = wind_damage

        return wind_damages    
                



def windstorm(self, windstorm_maps: xr.DataArray) -> float:
###MODIFIED flood() -> windstorm()
    """
    Applies damage to agents based on windstorm intensity and curves.
    Right now, it only considers concrete non-adapted residential buildings.
    """

    windstorm_maps: xr.DataArray = windstorm_maps.compute()
    #windstorm_maps = windstorm_maps.chunk({"x": 100, "y": 1000})

    buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(flood_map.rio.crs)

    # household is directly link to protected buildings so not sure how to add this at this point
    # it will definitely get added once i account for the adaptation measures.
    #household_points: gpd.GeoDataFrame = self.var.household_points.copy().to_crs(
    #    flood_map.rio.crs          
    #)

    # For now, we assume all buildings are residential and concrete
    buildings["object_type"] = "residential" 
    # Since i am not considering the adaptation measures, i am only considering the maximum damage
    buildings["maximum_damage"] = self.var.max_dam_buildings_structure

    # Create the folder to save damages maps if it doesn't exist
    damage_folder : Path = self.model.output_folder / "damages_maps"
    damage_folder.mkdir(parents=True, exist_ok=True)

    # Compute damges for buildings without adaptation measures
    wind_damages_buildings_structure = VectorScanner(
        feature_file=buildings.rename(columns={"maximum_wind_damage_m2": "maximum_wind_damage"}),
        hazard_file=windstorm_maps,
        curve_path=self.wind_building_structure_curve,
        gridded=False,
    )

    total_wind_damage_structure = damages_buildings_structure["wind_damage"].sum()
    print(f"Windstorm damages to building structure are: {total_wind_damage_structure}")

    # Save the wind damages to a file
    filename: str = "wind_damages_buildings_structure.gpkg"
    wind_damages_buildings_structure.to_file(damage_folder / filename, driver="GPKG")


    return total_wind_damage_structure
