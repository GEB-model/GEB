import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xclim.indices as xci
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, K2Score
from pgmpy.factors.discrete import State
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from scipy.stats import chi2_contingency, norm

from geb.agents.crop_farmers import (
    FIELD_EXPANSION_ADAPTATION,
    INDEX_INSURANCE_ADAPTATION,
    IRRIGATION_EFFICIENCY_ADAPTATION,
    PERSONAL_INSURANCE_ADAPTATION,
    PR_INSURANCE_ADAPTATION,
    SURFACE_IRRIGATION_EQUIPMENT,
    WELL_ADAPTATION,
)
from geb.build.methods import build_method

from ...workflows.io import open_zarr
from .. import GEBModel
from ..workflows.general import repeat_grid


class Survey:
    def __init__(self) -> None:
        self.mappers = {}

    def learn_structure(self, max_indegree=3):
        print("Estimating network structure")
        est = HillClimbSearch(data=self.samples)
        self.structure = est.estimate(
            scoring_method=K2Score(data=self.samples),
            max_indegree=max_indegree,
            max_iter=int(1e4),
            epsilon=1e-8,
            show_progress=True,
        )

    def estimate_parameters(self, plot=False, save=False):
        print("Learning network parameters")
        self.model = BayesianNetwork(self.structure)
        self.model.fit(
            self.samples,
            estimator=BayesianEstimator,
            prior_type="K2",
        )
        self.model.get_cpds()

        edge_params = {}
        for edge in self.model.edges():
            # get correlation between two variables
            cross_tab = pd.crosstab(self.samples[edge[0]], self.samples[edge[1]])
            chi_stat = chi2_contingency(cross_tab)[0]
            N: int = len(self.samples)
            minimum_dimension = min(cross_tab.shape) - 1

            # Cramer’s V value
            cramers_v_value = np.sqrt((chi_stat / N) / minimum_dimension)

            edge_params[edge] = {"label": round(cramers_v_value, 2)}

        if plot or save:
            self.model.to_daft(
                "circular",
                edge_params=edge_params,
                pgm_params={"grid_unit": 10},
                latex=False,
            ).render()
            if save:
                plt.savefig(save)
            if plot:
                plt.show()

    def fix_naming(self, samples):
        # replace all spaces in column names with underscores, otherwise pgmpy will throw an error when saving/loading model
        samples.columns = (
            samples.columns.str.lower()  # lowercase everything
            .str.replace(r"\s+", "_", regex=True)  # spaces → underscores
            .str.replace(r"[?&:()]", "", regex=True)  # drop ?, &, :, (, )
            .str.replace(r"_+", "_", regex=True)  # collapse multiple underscores
            .str.strip("_")  # trim leading/trailing underscores
        )
        # assert all column names are valid with values -0-9A-Z_a-z
        assert all(samples.columns.str.match(r"^[0-9A-Za-z_]+$"))
        # replace all spaces in dataframe data with underscores, otherwise pgmpy will throw an error when saving/loading model
        samples = samples.replace(" ", "_", regex=True)
        return samples

    def save(self, path):
        print("Saving model")
        self.model.save(str(path))

    def read(self, path):
        print("Loading model")
        self.model = BayesianNetwork().load(str(path), n_jobs=1)

    def create_mapper(
        self,
        variable,
        mean,
        std,
        nan_value=-1,
        plot=False,
        save=False,
        distribution="normal",
        invert=False,
    ):
        assert distribution == "normal", "Only normal distribution is implemented"
        values = self.get(variable).values
        values = values[values != nan_value]

        # Get all unique values. There may be gaps in the values, so we need to fill them in
        unique_values, unique_counts = np.unique(values, return_counts=True)
        # ... which we do using arange from the first to the last value
        values_intervals = np.zeros(
            unique_values[-1] + 1 - unique_values[0], dtype=np.int32
        )
        # then we set the counts of the unique values at the correct indices
        values_intervals[unique_values - unique_values[0]] = unique_counts

        values_intervals = values_intervals / values.size

        assert values_intervals.sum() == 1
        values_intervals_cum = np.insert(np.cumsum(values_intervals), 0, 0.01)
        assert values_intervals_cum[1] > 0.01
        assert values_intervals_cum[-2] < 0.99
        values_intervals_cum[-1] = 0.99
        values_sd_values = norm.ppf(values_intervals_cum, loc=mean, scale=std)

        if plot or save:
            _, ax = plt.subplots()
            x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
            y = norm.pdf(x, mean, std)
            ax.plot(x, y, color="black")
            for left_sd, right_sd in zip(values_sd_values[:-1], values_sd_values[1:]):
                assert left_sd < right_sd
                ax.fill_between(x, y, where=(x >= left_sd) & (x < right_sd))
            if plot:
                plt.show()
            if save:
                plt.savefig(save)
        self.mappers[variable] = {
            "sd_bins": values_sd_values,
            "mean": mean,
            "std": std,
            "min": values.min(),
            "max": values.max(),
            "invert": invert,
        }

    def apply_mapper(self, variable, values):
        values_ = []
        for value in values:
            assert variable in self.mappers, (
                f"Mapper for variable {variable} does not exist"
            )
            mapper = self.mappers[variable]
            bin = value - mapper["min"]
            range_ = mapper["sd_bins"][bin : bin + 2]

            prob_left, prob_right = norm.cdf(
                range_, loc=mapper["mean"], scale=mapper["std"]
            )
            random_prob = np.random.uniform(prob_left, prob_right)
            s = norm.ppf(random_prob, loc=mapper["mean"], scale=mapper["std"])
            assert s >= range_[0] and s <= range_[1]
            if mapper["invert"]:
                value_ = mapper["mean"] - s * mapper["std"]
            else:
                value_ = mapper["mean"] + s * mapper["std"]
            values_.append(value_)
        return values_

    def bin(self, data, question):
        values = self.bins[question]
        assert len(values["bins"]) == len(values["labels"]) + 1, (
            "Bin bounds must be one longer than labels"
        )
        return pd.cut(
            data,
            bins=values["bins"],
            labels=values["labels"],
        )

    def sample(
        self,
        n: int,
        evidence: list = [],
        evidence_columns: list = None,
        method: str = "rejection",
        show_progress: str = True,
    ):
        """Sample from the Bayesian network.

        Args:
            n: number of samples to generate
            evidence: list of evidence values (i.e., all samples will have these values ...)
            evidence_columns: list of evidence column names (i.e., ... for these columns)
            method: sampling method, only 'rejection' is implemented.
            show_progress: whether to show progress bar
        """
        assert method == "rejection", "Only rejection sampling is implemented"
        if show_progress:
            print("Generating samples")
        sampler = BayesianModelSampling(self.model)
        # if no evidence this is equalivalent to forward sampling
        if evidence:
            assert evidence_columns, (
                "If evidence is given, evidence_columns must be given as well"
            )
            assert len(evidence) == len(evidence_columns), (
                "Number of evidence values must match number of evidence columns"
            )
            for state, evidence_column in zip(evidence, evidence_columns):
                assert state in self.model.states[evidence_column], (
                    f"State {state} is not a valid state for variable {evidence_column}"
                )
            evidence = [
                State(var=evidence_column, state=state)
                for evidence_column, state in zip(evidence_columns, evidence)
            ]
        sample = sampler.rejection_sample(
            evidence=evidence, size=n, show_progress=show_progress
        )
        return sample

    @property
    def variables(self):
        return self.survey.columns.tolist()

    def get(self, question):
        return self.samples[question]


class FarmerSurvey(Survey):
    def __init__(self):
        super().__init__()
        # self.password = password
        self.bins = {
            "What is your age?": {
                "bins": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "labels": [
                    "0-10",
                    "10-20",
                    "20-30",
                    "30-40",
                    "40-50",
                    "50-60",
                    "60-70",
                    "70-80",
                    "80-90",
                    ">90",
                ],
            },
            "perceived self efficacy": {
                "bins": [1, 1.5, 2.5, 3.5, 4.5, 5],
                "labels": [1, 2, 3, 4, 5],
            },
            "perceived effectivity": {
                "bins": [1, 1.5, 2.5, 3.5, 4.5, 5],
                "labels": [1, 2, 3, 4, 5],
            },
            "How large is the area you grow crops on in hectares?": {
                "bins": [0, 1, 2, 4, np.inf],
                "labels": [
                    "0-1",
                    "1-2",
                    "2-4",
                    ">4",
                ],  # after size class groups in agricultural census
            },
            "How many years of education did you complete?": {
                "bins": [-np.inf, 0, 5, 10, 12, np.inf],
                "labels": [
                    "none",
                    "5th_standard",
                    "matriculation",
                    "higher_secondary",
                    "graduate",
                ],
            },
        }
        self.renames = {
            "What is your age?": "age",
            "How many years of education did you complete?": "education",
            "How large is the area you grow crops on in hectares?": "field_size",
            "How do you see yourself? Are you generally a person who is fully prepared to take risks or do you try to avoid taking risks?": "risk_aversion",
            "Some people live day by day and do not plan some years ahead in making financial decision for their household, how similar do you feel you are to those people?": "discount_rate",
        }

    def load_survey(self, path):
        # Read the survey data
        with zipfile.ZipFile(path) as zf:
            with zf.open(
                "survey_results_cleaned.xlsx"  # , pwd=self.password
            ) as excel_file:
                df = pd.read_excel(excel_file)
        return df

    def parse(self, path):
        self.survey = self.load_survey(path)
        self.survey["perceived self efficacy"] = self.survey[
            [
                column
                for column in self.survey.columns
                if column.startswith("Ability - ")
            ]
        ].mean(axis=1)

        self.survey["perceived effectivity"] = self.survey[
            [
                column
                for column in self.survey.columns
                if column.startswith("Effectivity - ")
            ]
        ].mean(axis=1)
        self.survey["How large is the area you grow crops on in hectares?"] = (
            self.survey["How large is the area you grow crops on in acres?"] * 0.404686
        )

        self.samples = self.survey[
            [
                # 'What is your gender?',
                "How many years of education did you complete?",
                # 'Savings',
                # 'Loans',
                "What is your age?",
                "Which sources do you use for irrigation?",
                "How large is the area you grow crops on in hectares?",
                "Are you planning to adopt any additional drought adaptation measures in the coming five years?",
                # "In which section of the survey area does the surveyee live?",
                "Which crops did you grow during the last Kharif season?",
                "perceived self efficacy",
                "perceived effectivity",
                "How do you see yourself? Are you generally a person who is fully prepared to take risks or do you try to avoid taking risks?",
                "Some people live day by day and do not plan some years ahead in making financial decision for their household, how similar do you feel you are to those people?",
            ]
        ]

        for question in self.bins.keys():
            self.samples[question] = self.bin(self.samples[question], question)

        # remove where data is -1
        self.samples = self.samples[(self.samples != -1).all(1)]
        # remove where data is NaN
        self.samples = self.samples.dropna()

        self.samples = self.samples.rename(columns=self.renames)

        self.fix_naming()

        return self.samples


class IHDSSurvey(Survey):
    def __init__(self):
        super().__init__()
        self.bins = {
            "age": {
                "bins": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "labels": [
                    "0-10",
                    "10-20",
                    "20-30",
                    "30-40",
                    "40-50",
                    "50-60",
                    "60-70",
                    "70-80",
                    "80-90",
                    ">90",
                ],
            },
            "area owned & cultivated (hectare)": {
                "bins": [0, 1, 2, 4, np.inf],
                "labels": [
                    "0-1",
                    "1-2",
                    "2-4",
                    ">4",
                ],  # after size class groups in agricultural census
            },
            "Education": {
                "bins": [-np.inf, 0, 5, 10, 12, np.inf],
                "labels": [
                    "none",
                    "5th_standard",
                    "matriculation",
                    "higher_secondary",
                    "graduate",
                ],
            },
            "Monthly consumption per capita Rs": {
                "bins": [50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 10000],
                "labels": [
                    "50-100",
                    "100-250",
                    "250-500",
                    "500-1000",
                    "1000-2000",
                    "2000-3000",
                    "3000-4000",
                    "4000-5000",
                    "5000-10000",
                ],
            },
        }
        self.renames = {
            "age": "age",
            "Education": "education",
            "area owned & cultivated (hectare)": "field_size",
            "Monthly consumption per capita Rs": "monthly_consumption_per_capita",
        }

    def load_survey(self, path):
        df = pd.read_csv(path)
        return df

    def parse(self, path):
        self.survey = self.load_survey(path)
        self.samples = self.survey[
            [
                "age",
                "Education",
                "area owned & cultivated (hectare)",
                "Monthly consumption per capita Rs",
            ]
        ]
        self.samples = self.samples[
            self.samples["Monthly consumption per capita Rs"] > 50
        ]  # below 50 is unrealisicaly low
        self.samples = self.samples[
            self.samples["Monthly consumption per capita Rs"] < 10000
        ]  # too little very rich people to be representative. Need specific study to adequately say something about them
        # drop all samples where at least one value is negative
        self.samples = self.samples[(self.samples >= 0).all(1)]
        for question in self.bins.keys():
            self.samples[question] = self.bin(self.samples[question], question)
        self.samples = self.samples.rename(columns=self.renames)
        self.fix_naming()
        return self.samples

    def build_crop_calendar_pivots(self, path, regions, size_labels):
        # 1. Load full survey
        self.survey = self.load_survey(path)

        # 2. Pick only the columns you care about for crops
        cols = [
            "State code",
            "District code",
            "area owned & cultivated (local unit)",
            "Kharif: Crop: Name",
            "Kharif: Crop: Irrigation",
            "Rabi: Crop: Name",
            "Rabi: Crop: Irrigation",
            "Summer: Crop: Name",
            "Summer: Crop: Irrigation",
        ]
        samples = self.survey[cols].copy()

        # Keep Karnataka and Maharasthra
        states_codes = samples["State code"]
        samples = samples[(states_codes == 27) | (states_codes == 29)]
        state_map = {27: "MH", 29: "KA"}

        samples["state_name"] = samples["State code"].map(state_map)

        # 6. Bin any of these columns that appear in self.bins
        for question in self.bins:
            if question in samples.columns:
                samples[question] = self.bin(samples[question], question)

        # 7. Apply your renames and naming fixes
        samples = samples.rename(columns=self.renames)
        self.samples_crops = self.fix_naming(samples)

        # Remap the crops
        crops_new = {
            "Fallow": -1,
            "Bajra": 0,
            "Groundnut": 1,
            "Jowar": 2,
            "Paddy": 3,
            "Sugarcane": 4,
            "Wheat": 5,
            "Gram": 6,
            "Maize": 7,
            "Moong": 8,
            "Ragi": 9,
            "Sunflower": 10,
            "Tur": 11,
        }
        raw2std = {
            # ── exact matches ───────────────────────────
            "Bajra": "Bajra",
            "Groundnut": "Groundnut",
            "Jowar": "Jowar",
            "Rice/_Paddy": "Paddy",
            "Sugarcane": "Sugarcane",
            "Wheat": "Wheat",
            "Gram": "Gram",
            "Maize": "Maize",
            "Moong": "Moong",
            "Ragi": "Ragi",
            "Sunflower": "Sunflower",
            "Tur_(arhar)": "Tur",
            "None": "Fallow",
            # ── cereals grouped with their nearest cousin ─
            "Barley": "Wheat",
            "Other_cereals": "Wheat",
            # ── pulses ───────────────────────────────────
            "Kulthi": "Gram",
            "Masur": "Gram",
            "Moth": "Moong",
            "Urad": "Moong",
            "Other_pulses": "Gram",
            # ── oilseeds, fibre, etc. ────────────────────
            "Safflower": "Sunflower",
            "Soyabean": "Groundnut",
            "Cotton": "Sugarcane",
            "Jute": "Sugarcane",
            # ── fodder/green forage ──────────────────────
            "Fodder": "Maize",
            # ── spices / fruit / veg / misc cash crops ───
            "Bananas": "Sugarcane",
            "Apples,_etc.": "Sugarcane",
            "Citrus_fruit": "Sugarcane",
            "Grapes": "Sugarcane",
            "Potatoes": "Sugarcane",
            "Onion": "Sugarcane",
            "Chilis": "Sugarcane",
            "Ginger": "Sugarcane",
            "Other_spice": "Sugarcane",
            "Other_veg": "Sugarcane",
            "Other_fruits": "Sugarcane",
            "Other_nonfood": "Sugarcane",
        }

        crop_cols = ["kharif_crop_name", "rabi_crop_name", "summer_crop_name"]

        for col in crop_cols:
            self.samples_crops[col].fillna("None", inplace=True)
            std_col = f"{col}_std"
            self.samples_crops[std_col] = (
                self.samples_crops[col].map(raw2std).fillna("Fallow")
            )
            code_col = f"{col}_code"
            self.samples_crops[code_col] = self.samples_crops[std_col].map(crops_new)

        irr_cols = [
            "kharif_crop_irrigation",
            "rabi_crop_irrigation",
            "summer_crop_irrigation",
        ]
        for col in irr_cols:
            self.samples_crops[col].fillna("No", inplace=True)

        # ── drop any plot whose rabi or summer crop is Sugarcane ────────────────
        mask_sc = (self.samples_crops["rabi_crop_name_std"] == "Sugarcane") | (
            self.samples_crops["summer_crop_name_std"] == "Sugarcane"
        )
        self.samples_crops = self.samples_crops.loc[~mask_sc]

        size_m2 = self.samples_crops["area_owned_cultivated_local_unit"] * 10_000
        bins = [
            0,
            5_000,
            10_000,
            20_000,
            30_000,
            40_000,
            50_000,
            75_000,
            100_000,
            200_000,
            np.inf,
        ]

        self.samples_crops["size_class"] = pd.cut(
            size_m2, bins=bins, labels=size_labels, right=True, include_lowest=True
        )

        self.samples_crops["crop_calendar"] = (
            "["
            + self.samples_crops["kharif_crop_name_code"].astype(int).astype(str)
            + ","
            + self.samples_crops["rabi_crop_name_code"].astype(int).astype(str)
            + ","
            + self.samples_crops["summer_crop_name_code"].astype(int).astype(str)
            + "]"
        )

        mask_irrigated = (
            (self.samples_crops["kharif_crop_irrigation"] == "Yes")
            | (self.samples_crops["rabi_crop_irrigation"] == "Yes")
            | (self.samples_crops["summer_crop_irrigation"] == "Yes")
        )

        df_rainfed = self.samples_crops.loc[~mask_irrigated].copy()
        df_irrigated = self.samples_crops.loc[mask_irrigated].copy()

        for _df in (df_rainfed, df_irrigated):
            _df["size_class"] = pd.Categorical(
                _df["size_class"], categories=size_labels, ordered=True
            )

        def make_pivot(data):
            return data.pivot_table(
                index=["state_name", "size_class"],
                columns="crop_calendar",
                aggfunc="size",
                fill_value=0,
                observed=False,
            ).astype(int)

        crop_cal_per_district_rainfed = make_pivot(df_rainfed)
        crop_cal_per_district_irrigated = make_pivot(df_irrigated)

        def _fill_empty_size_classes(block):
            empty = block.eq(0).all(axis=1)
            out = block.copy()
            out.loc[empty] = np.nan
            out = out.ffill().bfill()
            return out.astype(int)

        def fill_empty_rows(pivot):
            return pivot.groupby(level=0, group_keys=False).apply(
                _fill_empty_size_classes
            )

        crop_cal_per_district_rainfed = fill_empty_rows(crop_cal_per_district_rainfed)
        crop_cal_per_district_irrigated = fill_empty_rows(
            crop_cal_per_district_irrigated
        )

        return crop_cal_per_district_rainfed, crop_cal_per_district_irrigated


class fairSTREAMModel(GEBModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_farm_size(self):
        farms = self.subgrid["agents/farmers/farms"]
        farm_ids, farm_size_n_cells = np.unique(farms, return_counts=True)
        farm_size_n_cells = farm_size_n_cells[farm_ids != -1]
        farm_ids = farm_ids[farm_ids != -1]

        mean_cell_size = self.subgrid["cell_area"].mean().compute().item()
        farm_size_m2 = farm_size_n_cells * mean_cell_size
        return farm_size_m2

    @build_method(depends_on=["setup_forcing"])
    def setup_pr_GEV(self):
        pr: xr.DataArray = open_zarr(
            Path("input/other/climate/pr.zarr"),
        ) * (24 * 3600)
        pr_monthly: xr.DataArray = pr.resample(time="M").sum(dim="time", skipna=True)

        pr_yearly_max = (
            pr_monthly.groupby("time.year")
            .max(dim="time", skipna=True)
            .rename({"year": "time"})
            .chunk({"time": -1})
            .compute()
        )

        gev_pr = xci.stats.fit(pr_yearly_max, dist="genextreme").compute()

        self.set_grid(
            gev_pr.sel(dparams="c").astype(np.float32), name="climate/pr_gev_c"
        )
        self.set_grid(
            gev_pr.sel(dparams="loc").astype(np.float32), name="climate/pr_gev_loc"
        )
        self.set_grid(
            gev_pr.sel(dparams="scale").astype(np.float32),
            name="climate/pr_gev_scale",
        )

    @build_method(depends_on=["setup_create_farms", "setup_regions_and_land_use"])
    def setup_farmer_crop_calendar(
        self,
        seasons,
        crop_variables,
        irrigation_status_per_tehsil_fn,
        crop_data_per_tehsil_fn,
    ):
        n_farmers = self.array["agents/farmers/id"].size
        farms = self.subgrid["agents/farmers/farms"]

        # Set all farmers within command areas to canal irrigation
        adaptations = np.full(
            (
                n_farmers,
                max(
                    [
                        SURFACE_IRRIGATION_EQUIPMENT,
                        WELL_ADAPTATION,
                        IRRIGATION_EFFICIENCY_ADAPTATION,
                        FIELD_EXPANSION_ADAPTATION,
                        PERSONAL_INSURANCE_ADAPTATION,
                        INDEX_INSURANCE_ADAPTATION,
                        PR_INSURANCE_ADAPTATION,
                    ]
                )
                + 1,
            ),
            -1,
            dtype=np.int32,
        )

        command_areas = self.subgrid["waterbodies/subcommand_areas"]
        canal_irrigated_farms = np.unique(farms.where(command_areas != -1, -1))
        canal_irrigated_farms = canal_irrigated_farms[canal_irrigated_farms != -1]
        adaptations[canal_irrigated_farms, SURFACE_IRRIGATION_EQUIPMENT] = 1

        # Set all farmers within cells with rivers to canal irrigation

        def get_rivers(da, axis, **kwargs):
            from geb.hydrology.landcover import OPEN_WATER

            return np.any(da == OPEN_WATER, axis=axis)

        grid_cells_with_river = (
            self.subgrid["landsurface/land_use_classes"]
            .coarsen(
                x=self.subgrid_factor,
                y=self.subgrid_factor,
            )
            .reduce(get_rivers)
        )
        subgrid_cells_with_river = repeat_grid(
            grid_cells_with_river.values, self.subgrid_factor
        )

        canal_irrigated_farms = np.unique(farms.where(subgrid_cells_with_river, -1))
        canal_irrigated_farms = canal_irrigated_farms[canal_irrigated_farms != -1]
        adaptations[canal_irrigated_farms, SURFACE_IRRIGATION_EQUIPMENT] = 1

        groundwater_depth = self.grid["landsurface/elevation"] - self.grid[
            "groundwater/heads"
        ].sel(layer="upper")
        groundwater_depth_subgrid = repeat_grid(
            groundwater_depth.values, self.subgrid_factor
        )

        farms_values = farms.values.ravel()
        farms_mask = np.where(farms_values != -1)[0]
        farms_values_masked = farms_values[farms_mask]

        groundwater_depth_per_farm = np.bincount(
            farms_values_masked, weights=groundwater_depth_subgrid.ravel()[farms_mask]
        ) / np.bincount(farms_values_masked)
        assert not np.isnan(groundwater_depth_per_farm).any()

        # # well probability is set such that the farmers with the deepest groundwater have the lowest probability
        # farmer_well_probability = 1 - (
        #     groundwater_depth_per_farm - groundwater_depth_per_farm.min()
        # ) / (groundwater_depth_per_farm.max() - groundwater_depth_per_farm.min())

        farm_sizes = self.get_farm_size()
        assert farm_sizes.size == n_farmers

        # irrigated_area = (
        #     (irrigation_source == irrigation_sources["canal"]) * farm_sizes
        # ).sum()

        # target_irrigated_area_ratio = 0.9

        # remaining_irrigated_area = (
        #     farm_sizes.sum() * target_irrigated_area_ratio - irrigated_area
        # )

        # ordered_well_indices = np.arange(n_farmers)[
        #     np.argsort(farmer_well_probability)[::-1]
        # ]
        # cumulative_farm_area = np.cumsum(farm_sizes[ordered_well_indices])
        # farmers_with_well = ordered_well_indices[
        #     cumulative_farm_area <= remaining_irrigated_area
        # ]
        # irrigation_source[farmers_with_well] = irrigation_sources["well"]

        # irrigated_area = (
        #     (irrigation_source != irrigation_sources["no"]) * farm_sizes
        # ).sum()

        regions = self.geoms["regions"]

        irrigation_status_per_tehsil = pd.read_excel(irrigation_status_per_tehsil_fn)
        irrigation_status_per_tehsil["size_class"] = irrigation_status_per_tehsil[
            "size_class"
        ].map(
            {
                "Below 0.5": 0,
                "0.5-1.0": 1,
                "1.0-2.0": 2,
                "2.0-3.0": 3,
                "3.0-4.0": 4,
                "4.0-5.0": 5,
                "5.0-7.5": 6,
                "7.5-10.0": 7,
                "10.0-20.0": 8,
                "20.0 & ABOVE": 9,
            }
        )
        irrigation_status_per_tehsil["state_name"] = irrigation_status_per_tehsil[
            "state_name"
        ].ffill()
        irrigation_status_per_tehsil["district_n"] = irrigation_status_per_tehsil[
            "district_n"
        ].ffill()
        irrigation_status_per_tehsil["sub_dist_1"] = irrigation_status_per_tehsil[
            "sub_dist_1"
        ].ffill()

        def match_region(row, regions):
            region_id = regions.loc[
                (regions["state_name"] == row["state_name"])
                & (regions["district_n"] == row["district_n"])
                & (regions["sub_dist_1"] == row["sub_dist_1"]),
            ]["region_id"]  # .item()
            if region_id.size == 0:
                return -1
            else:
                return region_id.item()

        # assign region_id to crop data
        irrigation_status_per_tehsil["region_id"] = irrigation_status_per_tehsil.apply(
            lambda row: match_region(row, regions),
            axis=1,
        )
        irrigation_status_per_tehsil = irrigation_status_per_tehsil[
            irrigation_status_per_tehsil["region_id"] != -1
        ]
        irrigation_status_per_tehsil = irrigation_status_per_tehsil.drop(
            ["state_name", "district_n", "sub_dist_1"], axis=1
        )

        irrigation_status_per_tehsil = irrigation_status_per_tehsil.set_index(
            ["region_id", "size_class"]
        )

        irrigation_status_per_tehsil["well_ratio"] = (
            irrigation_status_per_tehsil["well_n_holdings"]
            + irrigation_status_per_tehsil["tubewell_n_holdings"]
        ) / (
            irrigation_status_per_tehsil["canals_n_holdings"]
            + irrigation_status_per_tehsil["tank_n_holdings"]
            + irrigation_status_per_tehsil["well_n_holdings"]
            + irrigation_status_per_tehsil["tubewell_n_holdings"]
            + irrigation_status_per_tehsil["other_n_holdings"]
            + irrigation_status_per_tehsil["no_irrigation_n_holdings"]
        )

        farm_size_class = np.zeros(n_farmers, dtype=np.int32)
        farm_size_class[farm_sizes > 5000] = 1
        farm_size_class[farm_sizes > 10000] = 2
        farm_size_class[farm_sizes > 20000] = 3
        farm_size_class[farm_sizes > 30000] = 4
        farm_size_class[farm_sizes > 40000] = 5
        farm_size_class[farm_sizes > 50000] = 6
        farm_size_class[farm_sizes > 75000] = 7
        farm_size_class[farm_sizes > 100000] = 8
        farm_size_class[farm_sizes > 200000] = 9

        region_id = self.array["agents/farmers/region_id"]

        region_ids = np.unique(region_id)
        size_classes = np.unique(farm_size_class)
        rate_array = (
            np.array([16.5, 11.5, 10.0, 7.75, 6.5, 6.5, 6.5, 5.0, 3.0, 3.0]) / 100
        )
        interest_rates = np.full(n_farmers, 0.05, dtype=np.float32)

        WELL_DEPTH_THRESHOLD = 80

        for region_id_class in region_ids:
            for size_class in size_classes:
                interest_rate_per_agent = rate_array[size_class]
                agent_subset = np.where(
                    (region_id_class == region_id) & (size_class == farm_size_class)
                )[0]
                if agent_subset.size == 0:
                    continue
                target_well_ratio = irrigation_status_per_tehsil.loc[
                    (region_id_class, size_class), "well_ratio"
                ]

                groundwater_depth_subset = groundwater_depth_per_farm[agent_subset]

                well_probability = np.maximum(
                    1 - (groundwater_depth_subset / WELL_DEPTH_THRESHOLD), 0
                )

                well_irrigated_agents = np.random.choice(
                    agent_subset,
                    int(target_well_ratio * len(agent_subset)),
                    replace=False,
                    p=well_probability / well_probability.sum(),
                )

                adaptations[well_irrigated_agents, WELL_ADAPTATION] = 1
                interest_rates[agent_subset] = interest_rate_per_agent

                # not_yet_irrigated_agents = np.where(
                #     adaptations[agent_subset, SURFACE_IRRIGATION_EQUIPMENT] == -1
                # )[0]

                # if not_yet_irrigated_agents.size == 0:
                #     continue

                # groundwater_depth_subset = groundwater_depth_per_farm[agent_subset][
                #     not_yet_irrigated_agents
                # ]
                # if (groundwater_depth_subset > WELL_DEPTH_THRESHOLD).all():
                #     continue

                # well_probability = np.maximum(
                #     1 - (groundwater_depth_subset / WELL_DEPTH_THRESHOLD), 0
                # )

                # well_irrigated_agents = np.random.choice(
                #     not_yet_irrigated_agents,
                #     int(target_well_ratio * len(not_yet_irrigated_agents)),
                #     replace=False,
                #     p=well_probability / well_probability.sum(),
                # )

                # adaptations[agent_subset[well_irrigated_agents], WELL_ADAPTATION] = 1

        crop_data_per_tehsil = pd.read_excel(crop_data_per_tehsil_fn)
        crop_data_per_tehsil = crop_data_per_tehsil[
            [
                c
                for c in crop_data_per_tehsil.columns
                if "_area" not in c and "_total" not in c
            ]
        ]
        crop_data_per_tehsil["state_name"] = crop_data_per_tehsil["state_name"].ffill()
        crop_data_per_tehsil["district_n"] = crop_data_per_tehsil["district_n"].ffill()
        crop_data_per_tehsil["sub_dist_1"] = crop_data_per_tehsil["sub_dist_1"].ffill()

        # assign region_id to crop data
        crop_data_per_tehsil["region_id"] = crop_data_per_tehsil.apply(
            lambda row: match_region(row, regions),
            axis=1,
        )
        crop_data_per_tehsil = crop_data_per_tehsil[
            crop_data_per_tehsil["region_id"] != -1
        ]
        crop_data_per_tehsil = crop_data_per_tehsil.drop(
            ["state_name", "district_n", "sub_dist_1"], axis=1
        )

        # create multi-level index using region id as the first level
        crop_data_per_tehsil = crop_data_per_tehsil.set_index(
            ["region_id", "size_class"]
        )

        crop_data_per_tehsil = crop_data_per_tehsil[
            [c for c in crop_data_per_tehsil.columns if "Cotton" not in c]
        ]
        crop_data_per_tehsil.columns = [
            c.replace("_holdings", "").replace("Tur (Arhar)", "Tur")
            for c in crop_data_per_tehsil.columns
        ]
        crop_data_per_tehsil_irrigated = crop_data_per_tehsil[
            [c for c in crop_data_per_tehsil.columns if "rain" not in c]
        ]
        crop_data_per_tehsil_irrigated.columns = [
            c.replace("_irr", "") for c in crop_data_per_tehsil_irrigated.columns
        ]
        crop_data_per_tehsil_rainfed = crop_data_per_tehsil[
            [c for c in crop_data_per_tehsil.columns if "irr" not in c]
        ]
        crop_data_per_tehsil_rainfed.columns = [
            c.replace("_rain", "") for c in crop_data_per_tehsil_rainfed.columns
        ]

        crop_name_to_ID = {
            crop["name"]: int(ID)
            for ID, crop in self.dict["crops/crop_data"]["data"].items()
        }

        # process crop calendars
        crop_calendar_per_farmer = np.full((n_farmers, 3, 4), -1, dtype=np.int32)
        crop_calendar_rotation_years = np.full(n_farmers, 1, dtype=np.int32)
        # crop_ids = list(crop_variables.keys())
        size_labels = [
            "Below 0.5",
            "0.5-1.0",
            "1.0-2.0",
            "2.0-3.0",
            "3.0-4.0",
            "4.0-5.0",
            "5.0-7.5",
            "7.5-10.0",
            "10.0-20.0",
            "20.0 & ABOVE",
        ]
        bayesian_net_folder = Path(self.root).parent / "preprocessing" / "bayesian_net"
        bayesian_net_folder.mkdir(exist_ok=True, parents=True)
        IHDS_survey_table = IHDSSurvey()
        crop_cal_per_district_rainfed, crop_cal_per_district_irrigated = (
            IHDS_survey_table.build_crop_calendar_pivots(
                path=Path("data") / "IHDS_I.csv",
                regions=regions,
                size_labels=size_labels,
            )
        )

        size_edges = np.array(
            [
                5_000,
                10_000,
                20_000,
                30_000,
                40_000,
                50_000,
                75_000,
                100_000,
                200_000,
                np.inf,
            ]
        )
        size_labels = np.array(
            [
                "Below 0.5",
                "0.5-1.0",
                "1.0-2.0",
                "2.0-3.0",
                "3.0-4.0",
                "4.0-5.0",
                "5.0-7.5",
                "7.5-10.0",
                "10.0-20.0",
                "20.0 & ABOVE",
            ]
        )
        n_sizes = len(size_labels)

        crop_data_tbl = {
            True: crop_data_per_tehsil_irrigated,
            False: crop_data_per_tehsil_rainfed,
        }
        calendar_tbl = {
            True: crop_cal_per_district_irrigated,
            False: crop_cal_per_district_rainfed,
        }

        size_mid = n_sizes // 2
        sugarcane_id = crop_name_to_ID["Sugarcane"]

        # ---- main loop ----------------------------------------------------------
        for idx in range(n_farmers):
            farmer_crop_calendar = crop_calendar_per_farmer[idx]

            is_irrigated = adaptations[
                idx, (SURFACE_IRRIGATION_EQUIPMENT, WELL_ADAPTATION)
            ].any()
            state_name = regions["state_name"][region_id[idx]]
            farmer_region_id = region_id[idx]

            farm_size = farm_sizes[idx]
            size_pos = np.searchsorted(size_edges, farm_size, side="right")
            size_class = size_labels[size_pos]

            crop_data_df = crop_data_tbl[is_irrigated]
            crop_calendar_df = calendar_tbl[is_irrigated]

            crop_data = crop_data_df.loc[farmer_region_id, size_class]
            if crop_data.sum() == 0:
                direction = 1 if size_pos < size_mid else -1  # search up or down first

                for step in range(1, n_sizes):
                    sc = size_labels[(size_pos + direction * step) % n_sizes]
                    candidate = crop_data_df.loc[farmer_region_id, sc]
                    if candidate.sum() > 0:
                        crop_data = candidate
                        break

                if crop_data.sum() == 0:
                    fallback_df = crop_data_per_tehsil_rainfed
                    for step in range(n_sizes):
                        sc = size_labels[(size_pos + direction * step) % n_sizes]
                        candidate = fallback_df.loc[farmer_region_id, sc]
                        if candidate.sum() > 0:
                            crop_data = candidate
                            break

            farmer_main_crop_id = crop_name_to_ID[
                np.random.choice(crop_data.index, p=crop_data / crop_data.sum())
            ]

            # ---------- choose rotation ----------
            if farmer_main_crop_id == sugarcane_id:
                crop_per_season = np.array([-1, -1, sugarcane_id])
            else:
                direction = 1 if size_pos < size_mid else -1
                pat = f"[{farmer_main_crop_id},"
                hit_found = False

                for step in range(n_sizes):
                    sc = size_labels[(size_pos + direction * step) % n_sizes]
                    row = crop_calendar_df.loc[state_name, sc]
                    subset = row[row.index.str.startswith(pat)]
                    if subset.sum() > 0:
                        rotation_str = np.random.choice(
                            subset.index, p=subset / subset.sum()
                        )
                        crop_per_season = np.fromstring(
                            rotation_str.strip("[]"), sep=",", dtype=int
                        )
                        hit_found = True
                        break

                if not hit_found:
                    crop_per_season = np.array([farmer_main_crop_id, -1, -1])

            # ---------- write rotation to farmer calendar ----------
            for season_idx, season_crop in enumerate(crop_per_season):
                if season_crop == -1:
                    continue
                duration = crop_variables[season_crop][
                    f"season_#{season_idx + 1}_duration"
                ]
                year_idx = 1 if duration > 365 else 0
                if year_idx:
                    crop_calendar_rotation_years[idx] = 2
                farmer_crop_calendar[season_idx] = [
                    season_crop,
                    seasons[f"season_#{season_idx + 1}_start"] - 1,
                    duration,
                    year_idx,
                ]

        self.set_array(adaptations, name="agents/farmers/adaptations")
        self.set_array(interest_rates, name="agents/farmers/interest_rate")
        self.set_array(crop_calendar_per_farmer, name="agents/farmers/crop_calendar")
        self.set_array(
            crop_calendar_rotation_years,
            name="agents/farmers/crop_calendar_rotation_years",
        )

    @build_method(depends_on=["setup_create_farms", "setup_cell_area"])
    def setup_farmer_characteristics(
        self,
        risk_aversion_mean,
        risk_aversion_std,
        discount_rate_mean,
        discount_rate_std,
        interest_rate,
        overwrite_bayesian_network=False,
    ):
        def normalize(array):
            return (array - np.min(array)) / (np.max(array) - np.min(array))

        education_levels = self.array["agents/farmers/education_level"]
        household_head_age = self.array["agents/farmers/age_household_head"]

        # Calculate intention factor based on age and education
        # Intention factor scales negatively with age and positively with education level
        intention_factor_raw = normalize(education_levels) - normalize(
            household_head_age
        )

        # Adjust the intention factor to center it around a mean of 0.5
        # The total intention of age, education and neighbor effects can scale to 1
        intention_factor = np.interp(intention_factor_raw, [-1, 1], [0.25, 0.75])

        self.set_array(intention_factor, name="agents/farmers/intention_factor")

        bayesian_net_folder = Path(self.root).parent / "preprocessing" / "bayesian_net"
        bayesian_net_folder.mkdir(exist_ok=True, parents=True)

        # IHDS_survey = IHDSSurvey()
        # IHDS_survey.parse_crops(path=Path("data") / "IHDS_I.csv")
        # save_path = bayesian_net_folder / "IHDS.bif"
        # if not save_path.exists() or overwrite_bayesian_network:
        #     IHDS_survey.learn_structure()
        #     IHDS_survey.estimate_parameters(
        #         plot=False, save=bayesian_net_folder / "IHDS.png"
        #     )
        #     IHDS_survey.save(save_path)
        # else:
        #     IHDS_survey.read(save_path)

        farmer_survey = FarmerSurvey()
        farmer_survey.parse(path=Path("data") / "survey_results_cleaned.zip")
        # save_path = bayesian_net_folder / "farmer_survey.bif"
        # if not save_path.exists() or overwrite_bayesian_network:
        #     farmer_survey.learn_structure()
        #     farmer_survey.estimate_parameters(
        #         plot=False, save=bayesian_net_folder / "farmer_survey.png"
        #     )
        #     farmer_survey.save(save_path)
        # else:
        #     farmer_survey.read(save_path)

        farmer_survey.learn_structure()
        farmer_survey.estimate_parameters(
            plot=False, save=bayesian_net_folder / "farmer_survey.png"
        )

        farmer_survey.create_mapper(
            "risk_aversion",
            mean=risk_aversion_mean,
            std=risk_aversion_std,
            nan_value=-1,
            save=bayesian_net_folder / "risk_aversion_mapper.png",
            invert=True,
        )

        # High values (5) means very short term planning, low values (1) means very long term planning
        farmer_survey.create_mapper(
            "discount_rate",
            mean=discount_rate_mean,
            std=discount_rate_std,
            nan_value=-1,
            save=bayesian_net_folder / "discount_rate_mapper.png",
            invert=True,
        )

        # n_farmers = self.array["agents/farmers/id"].size

        farms = self.subgrid["agents/farmers/farms"]
        farm_ids, farm_size_n_cells = np.unique(farms, return_counts=True)
        farm_size_n_cells = farm_size_n_cells[farm_ids != -1]
        farm_ids = farm_ids[farm_ids != -1]

        mean_cell_size = self.subgrid["cell_area"].mean()
        farm_size_m2 = farm_size_n_cells * mean_cell_size.compute().item()
        farm_size_bins = farmer_survey.bin(
            farm_size_m2 / 10_000,
            "How large is the area you grow crops on in hectares?",
        )
        groups, group_inverse, group_counts = np.unique(
            farm_size_bins, return_inverse=True, return_counts=True
        )

        # TODO: should this be float32?
        perceived_effectivity = np.full_like(farm_size_m2, -1, dtype=np.int32)
        risk_aversion_raw = np.full_like(farm_size_m2, -1, dtype=np.int32)
        discount_rate_raw = np.full_like(farm_size_m2, -1, dtype=np.int32)

        for group_count, (group, group_size) in enumerate(zip(groups, group_counts)):
            group_mask = group_inverse == group_count

            farmer_survey_samples = farmer_survey.sample(
                n=group_size,
                evidence=[group],
                evidence_columns=["field_size"],
                show_progress=False,
            )

            perceived_effectivity[group_mask] = farmer_survey_samples[
                "perceived_effectivity"
            ].values.astype(
                int
            )  # use array to avoid mathing on index, convert to int to make sure it is an integer
            risk_aversion_raw[group_mask] = farmer_survey_samples[
                "risk_aversion"
            ].astype(int)
            discount_rate_raw[group_mask] = farmer_survey_samples[
                "discount_rate"
            ].astype(int)

            # IHDS_survey_samples = IHDS_survey.sample(
            #     n=group_size,
            #     evidence=[group],
            #     evidence_columns=["field_size"],
            #     show_progress=False,
            # )

        risk_aversion = np.array(
            farmer_survey.apply_mapper("risk_aversion", risk_aversion_raw)
        )
        discount_rate = np.array(
            farmer_survey.apply_mapper("discount_rate", discount_rate_raw)
        )

        self.set_array(
            perceived_effectivity, name="agents/farmers/perceived_effectivity"
        )
        self.set_array(risk_aversion, name="agents/farmers/risk_aversion")
        self.set_array(discount_rate, name="agents/farmers/discount_rate")
