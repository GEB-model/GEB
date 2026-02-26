"""Contains class for early warning functions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterstats import zonal_stats
from workflows.households.early_warning import EarlyWarningModule

from ..workflows.io import read_zarr, write_zarr


class EarlyWarningModule:
    """This module contains functions to implement an early warning system for floods based on the forecasted flood maps and damage maps.

    It includes functions to create flood probability maps, implement warning strategies, communicate warnings to households, and simulate household response decisions.
    """

    def __init__(self, households):
        self.households = households

    def assign_households_to_postal_codes(self) -> None:
        """This function associates the household points with their postal codes to get the correct geometry for the warning function."""
        households = self.households.var.household_points.copy()

        # Associate households with their postal codes to use it later in the warning function
        postal_codes: gpd.GeoDataFrame = gpd.read_parquet(
            self.households.model.files["geom"]["postal_codes"]
        )
        postal_codes["postcode"] = postal_codes["postcode"].astype(str)
        households.to_crs(
            postal_codes.crs, inplace=True
        )  # Change to the same CRS as the postal codes

        households_with_postal_codes = gpd.sjoin(
            households,
            postal_codes[["postcode", "geometry"]],
            how="left",
            predicate="intersects",
        )

        # Drop columns that are not needed
        households_with_postal_codes.drop(columns=["index_right"], inplace=True)

        households_with_postal_codes.to_parquet(
            self.households.model.output_folder
            / "household_points_w_postal_codes.geoparquet"
        )

        self.households.var.household_points = households_with_postal_codes

        print(
            f"{len(households_with_postal_codes[households_with_postal_codes['postcode'].notnull()])} households assigned to {households_with_postal_codes['postcode'].nunique()} postal codes."
        )

    def load_ensemble_flood_maps(self, date_time: datetime) -> xr.DataArray:
        """Loads the flood maps for all ensemble members for a specific forecast date time.

        Args:
            date_time: The forecast date time for which to load the flood maps.
        Returns:
            A dataarray containing the flood maps from all ensemble members stacked along a new "member" dimension for the specified forecast time.
        """
        # Get number of members
        # open the flood maps folder to see the number of members
        flood_forecast_folder = (
            self.households.model.output_folder
            / "flood_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
        )
        n_ensemble_members = sum(1 for _ in flood_forecast_folder.glob("member_*"))
        print(f"Loading flood maps for {n_ensemble_members} ensemble members")

        # Load all the flood maps in an ensemble per each day
        flood_start_time = self.households.model.config["hazards"]["floods"]["events"][
            0
        ]["start_time"]
        flood_end_time = self.households.model.config["hazards"]["floods"]["events"][0][
            "end_time"
        ]

        members = []  # Every time has its own ensemble
        for member in range(
            1, n_ensemble_members + 1
        ):  # Define the number of members here
            file_path = (
                flood_forecast_folder
                / f"member_{member}"
                / f"{flood_start_time.strftime(format='%Y%m%dT%H%M%S')} - {flood_end_time.strftime(format='%Y%m%dT%H%M%S')}.zarr"
            )

            members.append(read_zarr(file_path))

        # Concatenate the members for each time (This stacks the list of dataarrays along a new "member" dimension)
        ensemble_flood_maps = xr.concat(members, dim="member")

        return ensemble_flood_maps

    def load_ensemble_damage_maps(self, date_time: datetime) -> pd.DataFrame:
        """Loads the damage maps for all ensemble members and aggregates them into a single dataframe. Work in standby for now.

        Args:
            date_time: The forecast date time for which to load the damage maps.
        Returns:
            A dataframe containing the aggregated damage maps for all ensemble members.
        """
        # Get number of members
        # open the damage maps folder to see the number of members
        damage_forecast_folder = (
            self.households.model.output_folder
            / "damage_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
        )
        n_ensemble_members = sum(1 for _ in damage_forecast_folder.glob("member_*"))
        print(f"Loading damage maps for {n_ensemble_members} ensemble members.")

        damage_maps = []
        # Load the damage maps for each ensemble member
        for member in range(1, n_ensemble_members + 1):
            file_path = (
                damage_forecast_folder
                / f"damage_map_buildings_content_{date_time.isoformat().replace(':', '').replace('-', '')}_member{member}.gpkg"
            )
            damage_map = gpd.read_file(file_path)

            # Add member and building_id columns
            damage_map["member"] = member
            damage_map["building_id"] = damage_map.index + 1

            # Aggregate all damage maps
            damage_maps.append(damage_map)

        # Concatenate all damage maps into a single dataframe
        ensemble_damage_maps = pd.concat(damage_maps)

        return ensemble_damage_maps

    def create_flood_probability_maps(
        self, date_time: datetime, strategy: int = 1, exceedance: bool = False
    ) -> dict[tuple[datetime, int], xr.DataArray]:
        """Creates flood probability maps based on the ensemble of flood maps for different warning strategies.

        Args:
            date_time: The forecast date time for which to create the probability maps.
            strategy: Identifier of the warning strategy (1 for water level ranges with measures,
                2 for energy substations, 3 for vulnerable/emergency facilities).
            exceedance: Whether to calculate flood exceedance probability maps (instead of regular probability maps).

        Returns:
            Probability maps for each water level range OR Probability exceedance maps for each lower bound of the water level ranges
            for that forecast initialization date.
        """
        # Load the ensemble of flood maps for that specific date time
        ensemble_flood_maps = self.load_ensemble_flood_maps(date_time=date_time)
        crs = self.households.model.config["hazards"]["floods"]["crs"]
        # TODO: need to think on how to load all the flood maps for each date time, instead of for each strategy/and to create flood prob exceedance maps

        if exceedance:
            # Create output folder for exceedance probability maps
            prob_folder = (
                self.households.model.output_folder
                / "flood_prob_exceedance_maps"
                / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            )
        else:
            # Create output folder for regular probability maps
            prob_folder = (
                self.households.model.output_folder
                / "flood_prob_maps"
                / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            )

        # Define water level ranges based on the chosen strategy (check the arguments for more info)
        if strategy == 1:
            # Water level ranges associated to specific measures (based on "impact" scale)
            ranges = []
            if exceedance:
                for key, value in self.households.var.wlranges_and_measures.items():
                    ranges.append((key, value["min"], None))
            else:
                for key, value in self.households.var.wlranges_and_measures.items():
                    ranges.append((key, value["min"], value["max"]))
        elif strategy == 2:
            # Water level range for energy substations (based on critical hit > 30 cm of flood)
            # TODO: need to make it not hard coded
            ranges = [(1, 0.3, None)]

        else:
            # Water level range for vulnerable and emergency facilities (flooded or not)
            # TODO: need to make it not hard coded
            ranges = [(1, 0.05, None)]

        probability_maps = {}
        # Loop over water level ranges to calculate probability maps
        for range_id, wl_min, wl_max in ranges:
            daily_ensemble = ensemble_flood_maps
            if wl_max is not None:
                condition = (daily_ensemble >= wl_min) & (daily_ensemble <= wl_max)
            else:
                condition = daily_ensemble >= wl_min
            probability = condition.sum(dim="member") / condition.sizes["member"]

            # Save probability map as a zarr file
            if exceedance:
                file_name = (
                    f"prob_exceedance_map_range{range_id}_strategy{strategy}.zarr"
                )
            else:
                file_name = f"prob_map_range{range_id}_strategy{strategy}.zarr"
            file_path = prob_folder / file_name
            file_path.mkdir(parents=True, exist_ok=True)

            # The y axis is flipped when writing to zarr, so fixing it here so it can be used for zonal stats later
            # TODO: need do it in a more elegant way
            if probability.y.values[0] < probability.y.values[-1]:
                print("flipping y axis so it can be used for zonal stats later")
                probability = probability.sortby("y", ascending=False)

            probability = probability.astype(np.float32)
            probability = probability.rio.write_nodata(np.nan)
            probability = write_zarr(da=probability, path=file_path, crs=crs)

            probability_maps[(date_time, range_id)] = probability

        return probability_maps
        # Right now I am not using this for anything, but maybe useful later to replace the file loading

    def create_damage_probability_maps(self, date_time: datetime) -> None:
        """Creates an object-based (buildings) probability map based on the ensemble of damage maps. Work in standby for now.

        Args:
            date_time: The forecast date time for which to create the damage probability maps.
        """
        crs = self.households.model.config["hazards"]["floods"]["crs"]

        damage_prob_maps_folder = (
            self.households.model.output_folder
            / "damage_prob_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
        )
        damage_prob_maps_folder.mkdir(parents=True, exist_ok=True)

        # Damage ranges for the probability map
        damage_ranges = [
            (1, 0, 1),
            (2, 100, 15000),
            (3, 15000, 30000),
            (4, 30000, 45000),
            (5, 45000, 60000),
            (6, 60000, None),
        ]
        # TODO: need to create a json dictionary with the right damage ranges

        # Load the ensemble of damage maps
        damage_ensemble = self.load_ensemble_damage_maps(date_time)

        # Get buildings geometry from one ensemble member (needed for the merges later)
        building_geometry = damage_ensemble[damage_ensemble["member"] == 1][
            ["building_id", "geometry"]
        ].copy()

        # Total number of ensemble members
        n_members = damage_ensemble["member"].nunique()

        # Loop over damage ranges and calculate probabilities
        for range_id, min_d, max_d in damage_ranges:
            if max_d is not None:
                condition = damage_ensemble["damages"].between(
                    min_d, max_d, inclusive="left"
                )
            else:
                condition = damage_ensemble["damages"] >= min_d

            # Count how many times each building falls in the range
            range_counts = (
                damage_ensemble[condition]
                .groupby("building_id")
                .size()
                .rename("count")
                .reset_index()
            )

            # Include all buildings and calculate the probability for each building in the range
            range_counts = pd.merge(
                building_geometry[["building_id"]],
                range_counts,
                on="building_id",
                how="left",
            )
            range_counts["count"] = range_counts["count"].fillna(0)
            range_counts["range_id"] = range_id
            range_counts["probability"] = range_counts["count"] / n_members

            damage_probability_map = pd.merge(
                range_counts, building_geometry, on="building_id", how="left"
            )
            damage_probability_map = gpd.GeoDataFrame(
                damage_probability_map, geometry="geometry", crs=crs
            )

            output_path = (
                damage_prob_maps_folder
                / f"damage_prob_map_range{range_id}_forecast{date_time.isoformat().replace(':', '').replace('-', '')}.geoparquet"
            )
            damage_probability_map.to_parquet(output_path)

    def water_level_warning_strategy(
        self,
        date_time: datetime,
        prob_threshold: float = 0.6,
        area_threshold: float = 0.1,
        strategy_id: int = 1,
    ) -> None:
        """Implements the water level warning strategy based on flood probability maps.

        Args:
            date_time: The forecast date time for which to implement the warning strategy.
            prob_threshold: The probability threshold above which a warning is issued.
            area_threshold: The area threshold (percentage of area) above which a warning is issued.
            strategy_id: Identifier of the warning strategy (1 for water level ranges with measures).
        """
        # Get the range ids and initialize the warning_log
        range_ids = list(self.households.var.wlranges_and_measures.keys())
        warning_log = []

        # Create probability maps
        self.create_flood_probability_maps(strategy=strategy_id, date_time=date_time)

        # Load households and postal codes
        households = self.households.var.household_points.copy()
        postal_codes = self.households.postal_codes.copy()
        # Maybe load this as a global var (?) instead of loading it each time

        for range_id in range_ids:
            # Build path to probability map
            prob_map = Path(
                self.households.model.output_folder
                / "prob_maps"
                / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
                / f"prob_map_range{range_id}_strategy{strategy_id}.zarr"
            )

            # Open the probability map
            prob_map = read_zarr(prob_map)
            affine = prob_map.rio.transform()
            prob_map = np.asarray(prob_map.values)

            # Get the pixel values for each postal code
            stats = zonal_stats(
                postal_codes,
                prob_map,
                affine=affine,
                raster_out=True,
                all_touched=True,
                nodata=np.nan,
            )

            # Iterate through each postal code and check how many pixels exceed the threshold
            for i, postalcode in enumerate(stats):
                pixel_values = postalcode["mini_raster_array"]
                postal_code = postal_codes.iloc[i]["postcode"]

                # Only get the values that are within the postal code
                postalcode_pixels = pixel_values[~pixel_values.mask]

                # Calculate the number of pixels with flood prob above the threshold
                n_above = np.sum(postalcode_pixels >= prob_threshold)

                # Calculate the total number of pixels within the postal code
                n_total = len(postalcode_pixels)

                if n_total == 0:
                    print(f"No valid pixels found for postal code {postal_code}")
                    continue
                else:
                    percentage = n_above / n_total

                if percentage >= area_threshold:
                    print(
                        f"Warning issued to postal code {postal_code} on {date_time.strftime('%d-%m-%Y T%H:%M:%S')} for range {range_id}: {percentage:.0%} of pixels > {prob_threshold}"
                    )

                    # Filter the affected households based on the postal code
                    affected_households = households[
                        households["postcode"] == postal_code
                    ]

                    # Get the measures and evacuation flag from the json dictionary to use in the warning communication function
                    measures = self.households.var.wlranges_and_measures[range_id][
                        "measure"
                    ]
                    evacuate = self.households.var.wlranges_and_measures[range_id][
                        "evacuate"
                    ]

                    # Communicate the warning to the target households
                    # This function should return the number of households that were warned
                    n_warned_households = self.households.warning_communication(
                        target_households=affected_households,
                        measures=measures,
                        evacuate=evacuate,
                        trigger="water_levels",
                    )

                    warning_log.append(
                        {
                            "date_time": date_time.isoformat(),
                            "postcode": postal_code,
                            "range": range_id,
                            "n_affected_households": len(affected_households),
                            "n_warned_households": n_warned_households,
                            "percentage_above_threshold": f"{percentage:.2f}",
                        }
                    )

        # Save the warning log to a csv file
        warnings_folder = self.households.model.output_folder / "warning_logs"
        warnings_folder.mkdir(exist_ok=True, parents=True)
        path = (
            warnings_folder
            / f"warning_log_water_levels_{date_time.isoformat().replace(':', '').replace('-', '')}.csv"
        )
        pd.DataFrame(warning_log).to_csv(path, index=False)

    def warning_communication(
        self,
        target_households: gpd.GeoDataFrame,
        measures: set[str],
        evacuate: bool,
        trigger: str,
        communication_efficiency: float = 0.35,
        lt_threshold_evacuation: int = 48,
    ) -> int:
        """Send warnings to a subset of target households according to communication_efficiency.

        Makes sure to send a single message and allows only upward escalation:
        none (0) -> measures (1) -> evacuate (2)
        Evacuation warning only sent if evacuate is True and lead_time <= 48h.

        Args:
            target_households: GeoDataFrame of household points targeted by the warning.
            measures: Set of recommended protective measures to communicate (strings).
            evacuate: Whether evacuation should be advised for this warning.
            trigger: Identifier of the trigger that initiated the warning.
            communication_efficiency: Fraction (0-1) of targeted households that can be reached.
            lt_threshold_evacuation: Lead time threshold (hours) below which evacuation warnings
                are allowed.

        Returns:
            int: Number of households that were successfully warned.
        """
        print("Running the warning communication for households...")

        # Get the number of target households
        n_target_households = len(target_households)
        if n_target_households == 0:
            print("No target households to warn.")
            return 0

        # Based on communication efficiency, get household indices of those who can receive the warning (randomly selected)
        n_feasible_warnings = int(n_target_households * communication_efficiency)
        if n_feasible_warnings == 0:
            print(
                "Based on the communication efficiency, no warning will reach households."
            )
            return 0

        rng = np.random.default_rng(42)  # Using a fixed seed for reproducibility
        position_indices = rng.choice(
            n_target_households, n_feasible_warnings, replace=False
        )
        selected_households = target_households.iloc[position_indices]

        # Get lead time in hours
        lead_time = self.compute_lead_time()
        print(f"Lead time for warning: {lead_time:.0f} hours")

        # Helper function to pick the measures to recommend based on lead time
        def pick_recommendations(
            available_measures: set[str], evacuate: bool, lead_time: float
        ) -> list[str]:
            """
            Decide which recommendations to include in the warning based on the lead time available.

            Args:
                available_measures (set[str]): Set of possible in-place measures to recommend.
                evacuate (bool): Whether evacuation is requested.
                lead_time (float): Lead time available in hours.

            Returns:
                chosen (list[str]): List of measures to recommend.
            """
            # Get implementation times for measures
            implementation_times = self.households.var.implementation_times

            # Sort the measures by their implementation times and alphabetically
            sorted_inplace_measures_per_time = sorted(
                available_measures,
                key=lambda measure: (implementation_times[measure], measure),
            )

            # Get evacuation time
            evac_time = implementation_times["evacuate"]

            # Non-evacuation case
            # If all measures fit in the lead time, return all
            if not evacuate:
                total = sum(
                    implementation_times[measure]
                    for measure in sorted_inplace_measures_per_time
                )
                if lead_time >= total:
                    return sorted_inplace_measures_per_time

                else:
                    # If not, pick as many measures as possible within the lead time
                    chosen_measures = []
                    used_time = 0
                    for measure in sorted_inplace_measures_per_time:
                        imp_time = implementation_times[measure]
                        if used_time + imp_time <= lead_time:
                            chosen_measures.append(measure)
                            used_time += imp_time
                    return chosen_measures
            else:
                # Evacuation case
                # If there is no time for evacuation, nothing to recommend
                if evac_time > lead_time:
                    return []
                else:
                    # If there is time for evacuation, check if any in-place measures fit as well
                    chosen_measures = ["evacuate"]
                    used_time = evac_time
                    for measure in sorted_inplace_measures_per_time:
                        imp_time = implementation_times[measure]
                        if used_time + imp_time <= lead_time:
                            chosen_measures.append(measure)
                            used_time += imp_time

                    return chosen_measures

        # Pick the measures to recommend based on lead time
        recommended_measures = pick_recommendations(measures, evacuate, lead_time)

        # Policy check for evacuation (only recommend it if lead time <= threshold)
        evac_feasible = "evacuate" in recommended_measures
        evac_policy = lead_time <= lt_threshold_evacuation
        if evac_feasible and not evac_policy:
            recommended_measures.remove("evacuate")
            evac_feasible = False

        # Determine the desired level of warning (0 = no warning, 1 = measures, 2 = evacuate)
        desired_level = 2 if evac_feasible else (1 if recommended_measures else 0)

        if desired_level == 0:
            return 0

        # Get possible measures and triggers (this is a list of all possible measures, not the recommended measures given by the warning strategies)
        # Used only to make sure the indices are correct when updating the arrays
        # (self.households.var.recommended_measures, self.households.var.warning_trigger)
        possible_measures_to_recommend = self.households.var.possible_measures
        possible_warning_triggers = self.households.var.possible_warning_triggers

        n_warned_households = 0
        for household_id in selected_households.index:
            # Skip already evacuated
            if self.households.var.evacuated[household_id] == 1:
                continue

            # Get current warning level
            current_level = int(self.households.var.warning_level[household_id])

            # Only send a warning if the desired level is higher than current level
            if desired_level > current_level:
                # For every measure in the recommended measures, get the corresponding index and set the right column to True to store it
                for measure in recommended_measures:
                    measure_idx = possible_measures_to_recommend.index(measure)
                    self.households.var.recommended_measures[
                        household_id, measure_idx
                    ] = True
            else:
                continue

            # Mark warning level and reached status
            self.households.var.warning_level[household_id] = desired_level
            self.households.var.warning_reached[household_id] = 1

            # Mark the trigger
            trigger_idx = possible_warning_triggers.index(trigger)
            self.households.var.warning_trigger[household_id, trigger_idx] = True
            n_warned_households += 1

        print(f"Warning targeted to reach {n_target_households} households")
        print(f"Warning reached {n_warned_households} households")

        return n_warned_households

    def compute_lead_time(self) -> float:
        """Compute lead time in hours based on forecast start time and current model time.

        Returns:
            float: Lead time in hours.
        """
        flood_event_start_time = self.households.model.config["hazards"]["floods"][
            "events"
        ][0]["start_time"]
        current_time = self.households.model.current_time
        lead_time = (flood_event_start_time - current_time).total_seconds() / 3600
        lead_time = max(lead_time, 0)  # Ensure non-negative lead time
        return lead_time

    def household_decision_making(
        self, date_time: datetime, responsive_ratio: float = 0.7
    ) -> None:
        """Simulate household emergency response decisions based on warnings and lead time.

        Args:
            date_time: The forecast date time for which to run the decision-making.
            responsive_ratio: Threshold for household responsiveness (0-1). Households with
                response_probability below this value are considered responsive and will
                act on warnings.
        """
        print("Running emergency response decision-making for households...")

        # Get lead time in hours
        lead_time = self.compute_lead_time()

        # Filter households that did not evacuate, were warned and are responsive
        not_evacuated_ids = self.households.var.evacuated == 0
        warned_ids = self.households.var.warning_reached == 1
        responsive_ids = self.households.var.response_probability < responsive_ratio

        # Combine the filters and apply it to household_points
        eligible_ids = np.asarray(warned_ids & not_evacuated_ids & responsive_ids)
        eligible_households = self.households.var.household_points.loc[eligible_ids]

        # Get the list of possible actions for eligible households
        # (this is a list of all possible actions, not the recommended actions given by the warning strategies)
        possible_actions = self.households.var.possible_measures
        evac_idx = possible_actions.index("evacuate")

        # Initialize an empty list to log actions taken
        actions_log = []

        # For every eligible household, do the recommended measures in the communicated warning
        for household_id in eligible_households.index:
            # For measure in recommended measures, change it to true in the actions_taken array
            self.households.var.actions_taken[household_id] = (
                self.households.var.recommended_measures[household_id].copy()
            )

            # If evacuation is among the actions taken, mark household as evacuated
            if self.households.var.actions_taken[household_id, evac_idx]:
                self.households.var.evacuated[household_id] = 1

            # Log the actions taken
            actions = []
            for i, action in enumerate(possible_actions):
                if self.households.var.actions_taken[household_id, i]:
                    actions.append(action)

            actions_log.append(
                {
                    "lead_time": lead_time,
                    "date_time": date_time.isoformat(),
                    "postal_code": self.households.var.household_points.loc[
                        household_id, "postcode"
                    ],
                    "household_id": household_id,
                    "actions": actions,
                }
            )

        # Save actions log
        actions_log_folder = self.households.model.output_folder / "actions_logs"
        actions_log_folder.mkdir(exist_ok=True, parents=True)
        path = (
            actions_log_folder
            / f"actions_log_{date_time.isoformat().replace(':', '').replace('-', '')}.csv"
        )
        pd.DataFrame(actions_log).to_csv(path, index=False)

    def update_households_geodataframe_w_warning_variables(
        self, date_time: datetime
    ) -> None:
        """This function merges the global variables related to warnings to the households geodataframe for visualization purposes.

        Args:
            date_time: The forecast date time for which to update the households geodataframe.
        """
        household_points: gpd.GeoDataFrame = self.households.var.household_points.copy()

        action_maps_folder: Path = self.households.model.output_folder / "action_maps"
        action_maps_folder.mkdir(parents=True, exist_ok=True)

        global_vars = [
            "warning_reached",
            "warning_level",
            "response_probability",
            "evacuated",
            "recommended_measures",
            "warning_trigger",
            "actions_taken",
        ]

        # make sure household points and global variables have the same length
        for global_var in global_vars:
            global_array = getattr(self.households.var, global_var)
            assert len(household_points) == global_array.shape[0], (
                f"The size of household points and {global_var} do not match"
            )

        # add columns in the household points geodataframe
        for name in [
            "warning_reached",
            "warning_level",
            "response_probability",
            "evacuated",
        ]:
            household_points[name] = getattr(self.households.var, name)

        warning_triggers = self.households.var.possible_warning_triggers
        for i, _ in enumerate(warning_triggers):
            if warning_triggers[i] == "water_levels":
                household_points["trig_w_levels"] = self.households.var.warning_trigger[
                    :, i
                ]
            if warning_triggers[i] == "critical_infrastructure":
                household_points["trig_crit_infra"] = (
                    self.households.var.warning_trigger[:, i]
                )

        possible_measures_to_recommend = self.households.var.possible_measures
        for i, measure in enumerate(possible_measures_to_recommend):
            if measure == "sandbags":
                household_points["recom_sandbags"] = (
                    self.households.var.recommended_measures[:, i]
                )
            if measure == "elevate possessions":
                household_points["recom_elev_possessions"] = (
                    self.households.var.recommended_measures[:, i]
                )
            if measure == "evacuate":
                household_points["recom_evacuate"] = (
                    self.households.var.recommended_measures[:, i]
                )

        possible_actions = self.households.var.possible_measures
        for i, action in enumerate(possible_actions):
            if action == "sandbags":
                household_points["sandbags"] = self.households.var.actions_taken[:, i]
            if action == "elevate possessions":
                household_points["elevated_possessions"] = (
                    self.households.var.actions_taken[:, i]
                )

        household_points.to_parquet(
            self.households.model.output_folder
            / "action_maps"
            / f"households_with_warning_parameters_{date_time.isoformat().replace(':', '').replace('-', '')}.geoparquet"
        )

    def step(self) -> None:
        """Step function for early warning module."""
        self.households.create_flood_probability_maps(
            date_time=self.households.model.__format__current_time,
            strategy=1,
            exceedance=True,
        )
        self.households.water_level_warning_strategy(
            date_time=self.households.model.current_time
        )
        self.households.critical_infrastructure_warning_strategy(
            date_time=self.households.model.current_time
        )
        self.households.household_decision_making(
            date_time=self.households.model.current_time
        )
        self.households.update_households_geodataframe_w_warning_variables(
            date_time=self.households.model.current_time
        )
