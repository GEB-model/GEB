from ..geb import GEBModel as GEBModel

import numpy as np
import geopandas as gpd
import zipfile
import gzip
from pathlib import Path
import pandas as pd
from honeybees.library.raster import pixels_to_coords
import matplotlib.pyplot as plt

from scipy.stats import norm
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

from pgmpy.estimators import HillClimbSearch

from scipy.stats import chi2_contingency


class Survey:
    def __init__(self) -> None:
        self.mappers = {}

    def learn_structure(self, max_indegree=3):
        print("Estimating network structure")
        est = HillClimbSearch(data=self.samples)
        self.model = est.estimate(
            scoring_method=K2Score(data=self.samples),
            max_indegree=max_indegree,
            max_iter=int(1e4),
            epsilon=1e-8,
        )

    def estimate_parameters(self, plot=False, save=False):
        print("Learning network parameters")
        self.model = BayesianNetwork(self.model)
        self.model.fit(
            self.samples,
            estimator=BayesianEstimator,
            prior_type="dirichlet",
            pseudo_counts=0.1,
        )
        self.model.get_cpds()

        edge_params = {}
        for edge in self.model.edges():
            # get correlation between two variables
            cross_tab = pd.crosstab(self.samples[edge[0]], self.samples[edge[1]])
            chi_stat = chi2_contingency(cross_tab)[0]
            N = len(self.samples)
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

    def fix_naming(self):
        # replace all spaces in column names with underscores, otherwise pgmpy will throw an error when saving/loading model
        self.samples.columns = self.samples.columns.str.replace(" ", "_")
        # remove all ?
        self.samples.columns = self.samples.columns.str.replace("?", "")
        # assert all column names are valid with values -0-9A-Z_a-z
        assert all(self.samples.columns.str.match(r"^[0-9A-Za-z_]+$"))
        # replace all spaces in dataframe data with underscores, otherwise pgmpy will throw an error when saving/loading model
        self.samples = self.samples.replace(" ", "_", regex=True)

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
            assert (
                variable in self.mappers
            ), f"Mapper for variable {variable} does not exist"
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
        assert (
            len(values["bins"]) == len(values["labels"]) + 1
        ), "Bin bounds must be one longer than labels"
        return pd.cut(
            data,
            bins=values["bins"],
            labels=values["labels"],
        )

    def sample(
        self,
        n,
        evidence=[],
        evidence_columns=None,
        method="rejection",
        show_progress=True,
    ):
        """
        Args:
            n (int): number of samples to generate
            evidence (list): list of evidence values (i.e., all samples will have these values ...)
            evidence_columns (list): list of evidence column names (i.e., ... for these columns)
            method (str): sampling method, only 'rejection' is implemented
        """
        assert method == "rejection", "Only rejection sampling is implemented"
        if show_progress:
            print("Generating samples")
        sampler = BayesianModelSampling(self.model)
        # if no evidence this is equalivalent to forward sampling
        if evidence:
            assert (
                evidence_columns
            ), "If evidence is given, evidence_columns must be given as well"
            assert len(evidence) == len(
                evidence_columns
            ), "Number of evidence values must match number of evidence columns"
            for state, evidence_column in zip(evidence, evidence_columns):
                assert (
                    state in self.model.states[evidence_column]
                ), f"State {state} is not a valid state for variable {evidence_column}"
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
                # "Which crops did you grow during the last Kharif season?",
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


class fairSTREAMModel(GEBModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_census_characteristics(
        self,
        maximum_age=85,
    ):
        n_farmers = self.binary["agents/farmers/id"].size
        farms = self.subgrid["agents/farmers/farms"]

        # get farmer locations
        vertical_index = (
            np.arange(farms.shape[0])
            .repeat(farms.shape[1])
            .reshape(farms.shape)[farms != -1]
        )
        horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(
            farms.shape
        )[farms != -1]
        farms_flattened = farms.values[farms.values != -1]
        pixels = np.zeros((n_farmers, 2), dtype=np.int32)
        pixels[:, 0] = np.round(
            np.bincount(farms_flattened, horizontal_index)
            / np.bincount(farms_flattened)
        ).astype(int)
        pixels[:, 1] = np.round(
            np.bincount(farms_flattened, vertical_index) / np.bincount(farms_flattened)
        ).astype(int)

        locations = pixels_to_coords(pixels + 0.5, farms.raster.transform.to_gdal())
        locations = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(locations[:, 0], locations[:, 1]),
            crs="EPSG:4326",
        )  # convert locations to geodataframe

        # GLOPOP-S uses the GDL regions. So we need to get the GDL region for each farmer using their location
        GDL_regions = self.data_catalog.get_geodataframe(
            "GDL_regions_v4", geom=self.region, variables=["GDLcode"]
        )
        GDL_region_per_farmer = gpd.sjoin(
            locations, GDL_regions, how="left", predicate="within"
        )

        # ensure that each farmer has a region
        assert GDL_region_per_farmer["GDLcode"].notna().all()

        # Load GLOPOP-S data. This is a binary file and has no proper loading in hydromt. So we use the data catalog to get the path and format the path with the regions and load it with NumPy
        GLOPOP_S = self.data_catalog.get_source("GLOPOP-S")

        GLOPOP_S_attribute_names = [
            "economic_class",
            "settlement_type_rural",
            "farmer",
            "age_class",
            "gender",
            "education_level",
            "household_type",
            "household_ID",
            "relation_to_household_head",
            "household_size_category",
        ]
        # Get list of unique GDL codes from farmer dataframe
        GDL_region_per_farmer["household_size"] = np.full(
            len(GDL_region_per_farmer), -1, dtype=np.int32
        )
        GDL_region_per_farmer["age_household_head"] = np.full(
            len(GDL_region_per_farmer), -1, dtype=np.int32
        )
        GDL_region_per_farmer["education_level"] = np.full(
            len(GDL_region_per_farmer), -1, dtype=np.int32
        )
        for GDL_region, farmers_GDL_region in GDL_region_per_farmer.groupby("GDLcode"):
            with gzip.open(GLOPOP_S.path.format(region=GDL_region), "rb") as f:
                GLOPOP_S_region = np.frombuffer(f.read(), dtype=np.int32)

            n_people = GLOPOP_S_region.size // len(GLOPOP_S_attribute_names)
            GLOPOP_S_region = pd.DataFrame(
                np.reshape(
                    GLOPOP_S_region, (len(GLOPOP_S_attribute_names), n_people)
                ).transpose(),
                columns=GLOPOP_S_attribute_names,
            ).drop(
                ["economic_class", "settlement_type_rural", "household_size_category"],
                axis=1,
            )
            # select farmers only
            GLOPOP_S_region = GLOPOP_S_region[GLOPOP_S_region["farmer"] == 1].drop(
                "farmer", axis=1
            )

            # shuffle GLOPOP-S data to avoid biases in that regard
            GLOPOP_S_household_IDs = GLOPOP_S_region["household_ID"].unique()
            np.random.shuffle(GLOPOP_S_household_IDs)  # shuffle array in-place
            GLOPOP_S_region = (
                GLOPOP_S_region.set_index("household_ID")
                .loc[GLOPOP_S_household_IDs]
                .reset_index()
            )

            # Select a sample of farmers from the database. Because the households were
            # shuflled there is no need to pick random households, we can just take the first n_farmers.
            # If there are not enough farmers in the region, we need to upsample the data. In this case
            # we will just take the same farmers multiple times starting from the top.
            GLOPOP_S_household_IDs = GLOPOP_S_region["household_ID"].values

            # first we mask out all consecutive duplicates
            mask = np.concatenate(
                ([True], GLOPOP_S_household_IDs[1:] != GLOPOP_S_household_IDs[:-1])
            )
            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[mask]

            GLOPOP_S_region_sampled = []
            if GLOPOP_S_household_IDs.size < len(farmers_GDL_region):
                n_repetitions = len(farmers_GDL_region) // GLOPOP_S_household_IDs.size
                max_household_ID = GLOPOP_S_household_IDs.max()
                for i in range(n_repetitions):
                    GLOPOP_S_region_copy = GLOPOP_S_region.copy()
                    # increase the household ID to avoid duplicate household IDs. Using (i + 1) so that the original household IDs are not changed
                    # so that they can be used in the final "topping up" below.
                    GLOPOP_S_region_copy["household_ID"] = GLOPOP_S_region_copy[
                        "household_ID"
                    ] + ((i + 1) * max_household_ID)
                    GLOPOP_S_region_sampled.append(GLOPOP_S_region_copy)
                requested_farmers = (
                    len(farmers_GDL_region) % GLOPOP_S_household_IDs.size
                )
            else:
                requested_farmers = len(farmers_GDL_region)

            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[:requested_farmers]
            GLOPOP_S_region_sampled.append(
                GLOPOP_S_region[
                    GLOPOP_S_region["household_ID"].isin(GLOPOP_S_household_IDs)
                ]
            )

            GLOPOP_S_region_sampled = pd.concat(
                GLOPOP_S_region_sampled, ignore_index=True
            )
            assert GLOPOP_S_region_sampled["household_ID"].unique().size == len(
                farmers_GDL_region
            )

            households_region = GLOPOP_S_region_sampled.groupby("household_ID")
            # select only household heads
            household_heads = households_region.apply(
                lambda x: x[x["relation_to_household_head"] == 1]
            )
            assert len(household_heads) == len(farmers_GDL_region)

            # age
            household_heads["age"] = np.full(len(household_heads), -1, dtype=np.int32)
            age_class_to_age = {
                1: (0, 16),
                2: (16, 26),
                3: (26, 36),
                4: (36, 46),
                5: (46, 56),
                6: (56, 66),
                7: (66, maximum_age + 1),
            }  # exclusive
            for age_class, age_range in age_class_to_age.items():
                household_heads_age_class = household_heads[
                    household_heads["age_class"] == age_class
                ]
                household_heads.loc[household_heads_age_class.index, "age"] = (
                    np.random.randint(
                        age_range[0],
                        age_range[1],
                        size=len(household_heads_age_class),
                        dtype=GDL_region_per_farmer["age_household_head"].dtype,
                    )
                )
            GDL_region_per_farmer.loc[
                farmers_GDL_region.index, "age_household_head"
            ] = household_heads["age"].values

            # education level
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "education_level"] = (
                household_heads["education_level"].values
            )

            # household size
            household_sizes_region = households_region.size().values.astype(np.int32)
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "household_size"] = (
                household_sizes_region
            )

        # assert none of the household sizes are placeholder value -1
        assert (GDL_region_per_farmer["household_size"] != -1).all()
        assert (GDL_region_per_farmer["age_household_head"] != -1).all()
        assert (GDL_region_per_farmer["education_level"] != -1).all()

        self.set_binary(
            GDL_region_per_farmer["household_size"].values,
            name="agents/farmers/household_size",
        )
        self.set_binary(
            GDL_region_per_farmer["age_household_head"].values,
            name="agents/farmers/age_household_head",
        )
        self.set_binary(
            GDL_region_per_farmer["education_level"].values,
            name="agents/farmers/education_level",
        )

    def setup_farmer_cropping(
        self,
        well_irrigated_ratio,
        seasons,
        crop_variables,
    ):
        n_farmers = self.binary["agents/farmers/id"].size
        farms = self.subgrid["agents/farmers/farms"]

        # process irrigation sources
        irrigation_sources = self.dict["agents/farmers/irrigation_sources"]
        irrigation_source = np.full(n_farmers, irrigation_sources["no"], dtype=np.int32)

        if "routing/lakesreservoirs/subcommand_areas" in self.subgrid:
            command_areas = self.subgrid["routing/lakesreservoirs/subcommand_areas"]
            canal_irrigated_farms = np.unique(farms.where(command_areas != -1, -1))
            canal_irrigated_farms = canal_irrigated_farms[canal_irrigated_farms != -1]
            irrigation_source[canal_irrigated_farms] = irrigation_sources["canal"]

        well_irrigated_farms = np.random.choice(
            [0, 1],
            size=n_farmers,
            replace=True,
            p=[1 - well_irrigated_ratio, well_irrigated_ratio],
        ).astype(bool)
        irrigation_source[
            (well_irrigated_farms) & (irrigation_source == irrigation_sources["no"])
        ] = irrigation_sources["well"]

        self.set_binary(irrigation_source, name="agents/farmers/irrigation_source")

        # process crop calendars
        crop_calendar_per_farmer = np.full((n_farmers, 3, 4), -1, dtype=np.int32)
        crop_calendar_rotation_years = np.full(n_farmers, 1, dtype=np.int32)
        crop_ids = list(crop_variables.keys())

        for idx in range(n_farmers):
            farmer_crop_calendar = crop_calendar_per_farmer[idx]
            farmer_irrigation_source = irrigation_source[idx]

            if farmer_irrigation_source in (
                irrigation_sources["well"],
                irrigation_sources["canal"],
            ):
                n_crops = 1 if np.random.random() < 0.2 else 2
            else:
                n_crops = 1 if np.random.random() < 0.8 else 2

            crops = np.random.choice(crop_ids, size=n_crops)
            for season_idx, crop in enumerate(crops):
                duration = crop_variables[crop][f"season_#{season_idx+1}_duration"]
                if duration > 365:
                    year_index = 1
                    crop_calendar_rotation_years[idx] = 2
                else:
                    year_index = 0
                farmer_crop_calendar[season_idx] = [
                    crop,
                    seasons[f"season_#{season_idx+1}_start"] - 1,
                    crop_variables[crop][f"season_#{season_idx+1}_duration"],
                    year_index,
                ]

        self.set_binary(crop_calendar_per_farmer, name="agents/farmers/crop_calendar")
        self.set_binary(
            crop_calendar_rotation_years,
            name="agents/farmers/crop_calendar_rotation_years",
        )

    def setup_farmer_characteristics(
        self,
        risk_aversion_mean,
        risk_aversion_std,
        discount_rate_mean,
        discount_rate_std,
        interest_rate,
        overwrite_bayesian_network=False,
    ):
        bayesian_net_folder = Path(self.root).parent / "preprocessing" / "bayesian_net"
        bayesian_net_folder.mkdir(exist_ok=True, parents=True)

        IHDS_survey = IHDSSurvey()
        IHDS_survey.parse(path=Path("data") / "IHDS_I.csv")
        save_path = bayesian_net_folder / "IHDS.bif"
        if not save_path.exists() or overwrite_bayesian_network:
            IHDS_survey.learn_structure()
            IHDS_survey.estimate_parameters(
                plot=False, save=bayesian_net_folder / "IHDS.png"
            )
            IHDS_survey.save(save_path)
        else:
            IHDS_survey.read(save_path)

        farmer_survey = FarmerSurvey()
        farmer_survey.parse(path=Path("data") / "survey_results_cleaned.zip")
        save_path = bayesian_net_folder / "farmer_survey.bif"
        if not save_path.exists() or overwrite_bayesian_network:
            farmer_survey.learn_structure()
            farmer_survey.estimate_parameters(
                plot=False, save=bayesian_net_folder / "farmer_survey.png"
            )
            farmer_survey.save(save_path)
        else:
            farmer_survey.read(save_path)

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

        n_farmers = self.binary["agents/farmers/id"].size

        # household_head_age = self.binary["agents/farmers/age_household_head"]
        farms = self.subgrid["agents/farmers/farms"]
        farm_ids, farm_size_n_cells = np.unique(farms, return_counts=True)
        farm_size_n_cells = farm_size_n_cells[farm_ids != -1]
        farm_ids = farm_ids[farm_ids != -1]

        mean_cell_size = self.subgrid["areamaps/sub_cell_area"].mean()
        farm_size_m2 = farm_size_n_cells * mean_cell_size.item()
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

        risk_aversion = farmer_survey.apply_mapper("risk_aversion", risk_aversion_raw)
        discount_rate = farmer_survey.apply_mapper("discount_rate", discount_rate_raw)

        self.set_binary(
            perceived_effectivity, name="agents/farmers/perceived_effectivity"
        )
        self.set_binary(risk_aversion, name="agents/farmers/risk_aversion")
        self.set_binary(discount_rate, name="agents/farmers/discount_rate")

        interest_rate = np.full(n_farmers, interest_rate, dtype=np.float32)
        self.set_binary(interest_rate, name="agents/farmers/interest_rate")
