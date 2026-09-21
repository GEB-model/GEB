"""Run trained crop-calendar decisions during the operational GEB simulation.

The module loads the temporal encoder and random forest only after spin-up, rebuilds
the recurrent history needed by the first operational decision from spin-up reporters,
replays prescribed historical decision boundaries without making ML choices during
spin-up, maintains compact daily and SPEI state, schedules farmer decisions, and
installs predicted calendars. Bootstrap I/O is chunked so warm-state reconstruction
does not require one Zarr read per variable per simulated day.
"""

from __future__ import annotations

import calendar
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from geb.geb_types import (
    Array,
    ArrayDatetime64,
    ArrayFloat32,
    ArrayInt32,
    ArrayInt64,
    ThreeDArrayFloat32,
    TwoDArray,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayInt32,
)

from ..data import DateIndex, load_economic_data
from ..workflows.io import read_array

if TYPE_CHECKING:
    from .crop_farmers import CropFarmers


# These names mirror the deployed feature contract written by
# machine_learning/main/crop_prediction.py. Land-surface values are aggregated
# where they are calculated and handed to CropFarmers once per completed model
# day. Farmer-owned values are added here before the joint daily encoder runs.
ML_LAND_SURFACE_DAILY_FEATURES = (
    "soil_temperature_layer_0_C_agents",
    "topsoil_temperature_C_agents",
    "profile_soil_temperature_C_agents",
    "top_soil_frozen_fraction_agents",
    "deep_soil_temperature_C_agents",
    "runoff_m_agents",
    "actual_evapotranspiration_m_agents",
    "potential_evapotranspiration_m_agents",
    "transpiration_m_agents",
    "topsoil_relative_available_water_agents",
    "profile_relative_available_water_agents",
    "crop_factor_agents",
    "root_depth_m_agents",
    "crop_sub_stage_agents",
    "interception_capacity_m_agents",
    "leaf_area_index_agents",
    "precipitation_m_daily_agents",
    "tas_2m_C_daily_mean_agents",
    "tas_2m_C_daily_min_agents",
    "tas_2m_C_daily_max_agents",
    "dewpoint_tas_2m_C_agents",
    "wind_speed_10m_m_per_s_agents",
    "downward_shortwave_radiation_MJ_m2_daily_agents",
    "reference_evapotranspiration_grass_m_agents",
    "actual_irrigation_consumption_m_agents",
)
ML_FARMER_DAILY_FEATURES = (
    "groundwater_depth",
    "cumulative_water_deficit_current_day",
    "risk_perception",
    "crop_decision_spei",
)
ML_FARMER_DAILY_REPORTS = ML_FARMER_DAILY_FEATURES + (
    "crop_decision_active_year_index",
    "crop_decision_yield_ratio",
)
ML_ECONOMIC_INPUTS = {
    "inflation_rates": "socioeconomics/inflation_rates",
    "interest_rates": "socioeconomics/interest_rates",
    "price_ratio": "socioeconomics/price_ratio",
    "LCU_per_USD": "socioeconomics/LCU_per_USD",
    "wholesale_price_index": "socioeconomics/wholesale_price_index",
    "commodity_price_index": (
        "socioeconomics/commodity_price_indices/commodity_price_index"
    ),
    "energy": "socioeconomics/commodity_price_indices/energy",
    "agriculture": "socioeconomics/commodity_price_indices/agriculture",
    "food": "socioeconomics/commodity_price_indices/food",
    "oils_and_meals": "socioeconomics/commodity_price_indices/oils_and_meals",
    "grains": "socioeconomics/commodity_price_indices/grains",
    "other_food": "socioeconomics/commodity_price_indices/other_food",
    "fertilizers": "socioeconomics/commodity_price_indices/fertilizers",
}
ML_REGIONAL_FARMER_INPUTS = {
    "household_size": "agents/farmers/household_size",
    "age_household_head": "agents/farmers/age_household_head",
    "education_level": "agents/farmers/education_level",
    "risk_aversion": "agents/farmers/risk_aversion",
    "risk_aversion_gains": "agents/farmers/risk_aversion_gains",
    "risk_aversion_losses": "agents/farmers/risk_aversion_losses",
    "intention_factor": "agents/farmers/intention_factor",
    "discount_rate": "agents/farmers/discount_rate",
}

ML_SUPPORTED_DECISION_AGENT_FEATURES = {
    "crop_decision_yield_ratio",
    "crop_1",
    "crop_1_start_date",
    "crop_1_duration",
}
ML_SUPPORTED_DECISION_TIME_FEATURES = {
    "decision_day_of_year",
    "decision_day_of_year_sin",
    "decision_day_of_year_cos",
}
ML_SUPPORTED_REGIONAL_CROP_FEATURES = {"crop_prices"}
ML_REQUIRED_VARIABLE_CONFIG_GROUPS = {
    "daily_agent",
    "decision_agent",
    "decision_time",
    "economic",
    "regional_crop",
    "regional_farmer",
    "static_agent",
}


def _ml_transform_specs(
    variable_name: str,
    settings: dict[str, Any],
) -> list[tuple[str, int, str]]:
    """Build lag and rolling transform specifications in training order.

    Returns:
        Ordered ``(kind, value, feature_name)`` transform specifications.
    """
    specs = [
        ("lag", int(lag), f"{variable_name}__lag_{int(lag)}")
        for lag in sorted(settings.get("lags", [0]))
    ]
    specs.extend(
        (
            "rolling",
            int(window),
            f"{variable_name}__rolling_mean_{int(window)}y",
        )
        for window in sorted(settings.get("rolling", []))
    )
    return specs


def _ml_raw_tabular_feature_names(
    variable_config: dict[str, dict[str, dict[str, Any]]],
    crop_ids: ArrayInt64,
) -> list[str]:
    """Reconstruct raw tabular feature names in the training order.

    Returns:
        Feature names in the same pre-imputation order used during training.
    """
    names: list[str] = []
    for name, settings in variable_config["decision_agent"].items():
        if settings.get("enabled", True):
            names.extend(
                feature_name
                for _, _, feature_name in _ml_transform_specs(name, settings)
            )
    names.extend(
        name
        for name, settings in variable_config["decision_time"].items()
        if settings.get("enabled", True)
    )
    for name, settings in variable_config["economic"].items():
        if settings.get("enabled", True):
            names.extend(
                feature_name
                for _, _, feature_name in _ml_transform_specs(name, settings)
            )
    for name, settings in variable_config["regional_crop"].items():
        if not settings.get("enabled", True):
            continue
        for _, _, transform_name in _ml_transform_specs(name, settings):
            names.extend(
                f"{transform_name}__crop_{int(crop_id)}" for crop_id in crop_ids
            )
    for name, settings in variable_config["regional_farmer"].items():
        if settings.get("enabled", True):
            names.extend(
                f"{name}__regional_{statistic}"
                for statistic in settings.get("statistics", ["mean", "std"])
            )
    names.extend(
        name
        for name, settings in variable_config["static_agent"].items()
        if settings.get("enabled", True)
    )
    return names


class DecisionModuleML:
    """Run the final continuous-state crop-choice model inside operational GEB.

    The persistent lag-0 daily branch is updated continuously with one retained
    decision-aligned ten-day block plus the GRU hidden state. Complete blocks are
    committed immediately; the final partial block is zero-padded and committed
    when a crop decision is evaluated. The carried hidden state is decayed with
    the trained 200-day half-life before every recurrent update. Decision-relative
    lagged/rolling SPEI is reconstructed from a compact monthly cache and encoded
    only for farmers whose decision is due.
    """

    FINAL_BUNDLE_FORMAT_VERSION = 6
    FINAL_MEMORY_HALF_LIFE_DAYS = 200.0

    def __init__(self, farmers: CropFarmers, config: dict[str, Any]) -> None:
        """Load the final trained model and initialize compact operational state.

        Args:
            farmers: Crop-farmer agent instance whose restored state and inputs are used
                by the operational crop-choice model.
            config: Machine-learning runtime configuration, including the trained-model
                directory and optional execution settings such as device and batch size.

        Raises:
            FileNotFoundError: If the trained-model directory is incomplete or required
                deployment/checkpoint files cannot be found.
            KeyError: If required bundle/checkpoint metadata or feature mappings are
                missing.
            NotImplementedError: If the deployed feature configuration enables an online
                predictor that this runtime does not implement.
            RuntimeError: If initialization occurs during spin-up, restored farmer state
                is incomplete, or the requested execution device is unavailable.
            TypeError: If the deployment bundle, checkpoint, or saved objects have an
                unexpected type or structure.
            ValueError: If the trained-model contract, feature layout, architecture,
                calendar state, or runtime dimensions are inconsistent with this final
                decision-aligned implementation.
        """
        self.farmers = farmers
        self.model = farmers.model
        self.config = config
        initialization_timer = time.perf_counter()

        if self.model.in_spinup:
            raise RuntimeError(
                "DecisionModuleML is operational only and cannot be constructed "
                "during spin-up."
            )

        import joblib
        import torch
        from torch import nn

        self.torch = torch

        required_restored_state = (
            "n",
            "crop_calendar",
            "crop_calendar_base_array",
            "crop_calendar_years",
            "crop_calendar_active_year_index",
        )
        missing_restored_state = [
            name for name in required_restored_state if not hasattr(farmers.var, name)
        ]
        if missing_restored_state:
            raise RuntimeError(
                "DecisionModuleML was constructed before CropFarmers spin-up state "
                f"was restored. Missing Bucket entries: {missing_restored_state}."
            )
        if farmers.var.crop_calendar_base_array.ndim != 4:
            raise ValueError(
                "Machine-learning crop switching requires the multi-year HRL crop "
                "calendar and agents/farmers/crop_calendar_years."
            )

        model_directory_value = config.get("model_directory")
        if not model_directory_value:
            raise ValueError(
                "Machine-learning crop switching requires "
                "expected_utility.crop_switching.machine_learning.model_directory."
            )
        self.model_directory = Path(model_directory_value).expanduser()
        deployment_path = self.model_directory / "geb_crop_choice_bundle.joblib"
        checkpoint_path = self.model_directory / "temporal_autoencoder.pt"
        if not deployment_path.exists():
            raise FileNotFoundError(
                "The final crop-choice runtime requires geb_crop_choice_bundle.joblib; "
                f"missing {deployment_path}."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing temporal crop-choice checkpoint {checkpoint_path}."
            )

        artifact_timer = time.perf_counter()
        self.bundle = joblib.load(deployment_path)
        if not isinstance(self.bundle, Mapping):
            raise TypeError(
                "The crop-choice deployment bundle must be a mapping, found "
                f"{type(self.bundle).__name__}."
            )
        if (
            int(self.bundle.get("format_version", -1))
            != self.FINAL_BUNDLE_FORMAT_VERSION
        ):
            raise ValueError(
                "DecisionModuleML requires final deployment bundle format "
                f"{self.FINAL_BUNDLE_FORMAT_VERSION}; found "
                f"{self.bundle.get('format_version')!r}."
            )
        if (
            self.bundle.get("persistent_state_training")
            != "decision_aligned_carried_state"
        ):
            raise ValueError(
                "The deployed model was not trained with carried decision-aligned state."
            )
        if (
            self.bundle.get("decision_timing_mode")
            != "earliest_subregion_candidate_planting"
        ):
            raise ValueError(
                "The deployed model was not trained with "
                "earliest_subregion_candidate_planting."
            )
        memory_half_life = float(self.bundle.get("memory_half_life_days", np.nan))
        if not np.isclose(memory_half_life, self.FINAL_MEMORY_HALF_LIFE_DAYS):
            raise ValueError(
                "The final GEB crop-choice runtime requires a 200-day persistent-state "
                f"half-life; found {memory_half_life!r}."
            )
        self.memory_half_life_days = memory_half_life
        expected_decay_rule = "multiply_carried_hidden_by_2**(-elapsed_days/half_life)"
        if self.bundle.get("memory_decay_rule") != expected_decay_rule:
            raise ValueError("The deployment bundle memory-decay rule is unsupported.")
        self.model.logger.info(
            "DecisionModuleML timing: deployment artifact load=%.2f s.",
            time.perf_counter() - artifact_timer,
        )

        self.prediction_model = (
            str(
                config.get(
                    "prediction_model",
                    self.bundle.get(
                        "default_prediction_model", "hierarchical_random_forest"
                    ),
                )
            )
            .strip()
            .lower()
        )
        # The format-6 deployment bundle contains a crop-first hierarchical RF.
        # Keep the old configuration names as aliases so existing GEB configs do not
        # need to change solely because the training artifact became hierarchical.
        prediction_model_aliases = {
            "random_forest": "hierarchical_random_forest",
            "random_forest_switch_gate": "hierarchical_random_forest_switch_gate",
        }
        self.prediction_model = prediction_model_aliases.get(
            self.prediction_model, self.prediction_model
        )
        if self.prediction_model not in {
            "hierarchical_random_forest",
            "hierarchical_random_forest_switch_gate",
        }:
            raise ValueError(
                "machine_learning.prediction_model must be "
                "'hierarchical_random_forest' or "
                "'hierarchical_random_forest_switch_gate', found "
                f"{self.prediction_model!r}."
            )

        self.selection_mode = (
            str(config.get("selection_mode", "argmax")).strip().lower()
        )
        if self.selection_mode not in {"argmax", "stochastic"}:
            raise ValueError(
                "machine_learning.selection_mode must be 'argmax' or 'stochastic', "
                f"found {self.selection_mode!r}."
            )
        self.selection_random_seed = int(
            config.get("random_seed", self.bundle.get("model_random_seed", 44))
        )
        self.selection_rng = np.random.default_rng(self.selection_random_seed)

        self.variable_config = self.bundle["variable_config"]
        if not isinstance(self.variable_config, Mapping):
            raise TypeError("The deployed variable_config must be a mapping.")
        missing_config_groups = sorted(
            ML_REQUIRED_VARIABLE_CONFIG_GROUPS.difference(self.variable_config)
        )
        if missing_config_groups:
            raise KeyError(
                "The deployed crop-choice variable configuration is missing required "
                f"groups: {missing_config_groups}."
            )

        enabled_decision_agent = {
            name
            for name, settings in self.variable_config["decision_agent"].items()
            if settings.get("enabled", True)
        }
        unsupported_decision_agent = sorted(
            enabled_decision_agent - ML_SUPPORTED_DECISION_AGENT_FEATURES
        )
        if unsupported_decision_agent:
            raise NotImplementedError(
                "DecisionModuleML does not implement these decision-agent features: "
                f"{unsupported_decision_agent}."
            )
        enabled_decision_time = {
            name
            for name, settings in self.variable_config["decision_time"].items()
            if settings.get("enabled", True)
        }
        unsupported_decision_time = sorted(
            enabled_decision_time - ML_SUPPORTED_DECISION_TIME_FEATURES
        )
        if unsupported_decision_time:
            raise NotImplementedError(
                "DecisionModuleML does not implement these decision-time features: "
                f"{unsupported_decision_time}."
            )
        enabled_regional_crop = {
            name
            for name, settings in self.variable_config["regional_crop"].items()
            if settings.get("enabled", True)
        }
        unsupported_regional_crop = sorted(
            enabled_regional_crop - ML_SUPPORTED_REGIONAL_CROP_FEATURES
        )
        if unsupported_regional_crop:
            raise NotImplementedError(
                "DecisionModuleML does not implement these regional-crop features: "
                f"{unsupported_regional_crop}."
            )
        unsupported_economic = sorted(
            name
            for name, settings in self.variable_config["economic"].items()
            if settings.get("enabled", True) and name not in ML_ECONOMIC_INPUTS
        )
        if unsupported_economic:
            raise NotImplementedError(
                "DecisionModuleML has no online economic input mapping for "
                f"{unsupported_economic}."
            )
        unsupported_regional_farmer = sorted(
            name
            for name, settings in self.variable_config["regional_farmer"].items()
            if settings.get("enabled", True) and name not in ML_REGIONAL_FARMER_INPUTS
        )
        if unsupported_regional_farmer:
            raise NotImplementedError(
                "DecisionModuleML has no online regional-farmer input mapping for "
                f"{unsupported_regional_farmer}."
            )

        self.days_per_year = int(self.bundle["days_per_year"])
        self.sequence_end_offset_days = int(
            self.bundle["daily_sequence_end_offset_days"]
        )
        if self.days_per_year != 366 or self.sequence_end_offset_days != 1:
            raise ValueError(
                "The final runtime requires a 366-day first-decision history ending "
                "one day before the decision."
            )

        self.raw_daily_names = [
            name
            for name, settings in self.variable_config["daily_agent"].items()
            if settings.get("enabled", True)
        ]
        self.raw_daily_index = {
            name: index for index, name in enumerate(self.raw_daily_names)
        }
        non_streamable_daily = [
            name
            for name, settings in self.variable_config["daily_agent"].items()
            if settings.get("enabled", True)
            and name != "crop_decision_spei"
            and (
                list(settings.get("lags", [0])) != [0]
                or bool(settings.get("rolling", []))
            )
        ]
        if non_streamable_daily:
            raise ValueError(
                "Only crop_decision_spei may use decision-relative daily transforms; "
                f"found {non_streamable_daily}."
            )

        spei_settings = self.variable_config["daily_agent"].get(
            "crop_decision_spei", {}
        )
        spei_enabled = bool(spei_settings and spei_settings.get("enabled", True))
        if spei_enabled:
            spei_offsets = [int(value) for value in spei_settings.get("lags", [0])]
            for window in spei_settings.get("rolling", []):
                spei_offsets.extend(range(int(window)))
            if 0 not in spei_offsets:
                raise ValueError("The daily SPEI contract must include lag 0.")
            self.maximum_spei_years_back = max(spei_offsets, default=0)
        else:
            self.maximum_spei_years_back = 0
        configured_spei_months = int(
            self.bundle.get("online_temporal_state", {}).get(
                "spei_month_cache_months", 0
            )
        )
        # Recomputing a full 366-day decision branch at decision time requires
        # roughly one extra year beyond the largest calendar lag. For lag 2 this
        # is 40 months, rather than the old 28-month streaming cache.
        self.spei_month_capacity = max(
            configured_spei_months,
            12 * (self.maximum_spei_years_back + 1) + 4,
        )

        supported_daily = set(ML_LAND_SURFACE_DAILY_FEATURES) | set(
            ML_FARMER_DAILY_FEATURES
        )
        unsupported_daily = sorted(set(self.raw_daily_names) - supported_daily)
        if unsupported_daily:
            raise KeyError(
                "The deployed model contains daily variables GEB does not capture: "
                f"{unsupported_daily}."
            )
        self.land_surface_daily_names = tuple(
            name
            for name in self.raw_daily_names
            if name in ML_LAND_SURFACE_DAILY_FEATURES
        )

        self.daily_feature_names = list(self.bundle["daily_feature_names"])
        self.raw_tabular_feature_names = list(self.bundle["raw_tabular_feature_names"])
        if len(set(self.daily_feature_names)) != len(self.daily_feature_names):
            raise ValueError("Saved daily feature names contain duplicates.")
        if len(set(self.raw_tabular_feature_names)) != len(
            self.raw_tabular_feature_names
        ):
            raise ValueError("Saved raw tabular feature names contain duplicates.")
        configured_daily_feature_names = [
            feature_name
            for name, settings in self.variable_config["daily_agent"].items()
            if settings.get("enabled", True)
            for _, _, feature_name in _ml_transform_specs(name, settings)
        ]
        if configured_daily_feature_names != self.daily_feature_names:
            raise ValueError(
                "The deployed daily feature-name contract differs from variable_config."
            )
        configured_raw_tabular_feature_names = _ml_raw_tabular_feature_names(
            self.variable_config,
            farmers.var.crop_data.index.to_numpy(dtype=np.int64),
        )
        if configured_raw_tabular_feature_names != self.raw_tabular_feature_names:
            raise ValueError(
                "The deployed raw tabular feature-name contract differs from variable_config."
            )

        self.daily_mean = np.asarray(
            self.bundle["daily_standardizer_mean"], dtype=np.float32
        )
        self.daily_scale = np.asarray(
            self.bundle["daily_standardizer_scale"], dtype=np.float32
        )
        if self.daily_mean.shape != (len(self.daily_feature_names),) or (
            self.daily_scale.shape != self.daily_mean.shape
        ):
            raise ValueError("Daily standardizer and daily feature names disagree.")
        if (
            np.any(~np.isfinite(self.daily_mean))
            or np.any(~np.isfinite(self.daily_scale))
            or np.any(self.daily_scale <= 0.0)
        ):
            raise ValueError(
                "Daily standardizer moments must be finite and scales positive."
            )

        hierarchical_state = self.bundle.get("hierarchical_random_forest")
        if not isinstance(hierarchical_state, Mapping):
            raise TypeError(
                "The format-6 deployment bundle must contain a "
                "hierarchical_random_forest mapping."
            )
        required_hierarchical_keys = {
            "crop_forest",
            "conditional_forests",
            "conditional_class_ids",
            "crop_values",
            "class_crop_values",
            "n_calendar_classes",
        }
        missing_hierarchical_keys = sorted(
            required_hierarchical_keys.difference(hierarchical_state)
        )
        if missing_hierarchical_keys:
            raise KeyError(
                "The deployed hierarchical RF is missing keys: "
                f"{missing_hierarchical_keys}."
            )
        self.crop_forest = hierarchical_state["crop_forest"]
        self.conditional_forests = {
            int(crop): forest
            for crop, forest in hierarchical_state["conditional_forests"].items()
        }
        self.conditional_class_ids = {
            int(crop): np.asarray(class_ids, dtype=np.int64)
            for crop, class_ids in hierarchical_state["conditional_class_ids"].items()
        }
        self.hierarchical_crop_values = np.asarray(
            hierarchical_state["crop_values"], dtype=np.int64
        ).reshape(-1)
        self.class_crop_values = np.asarray(
            hierarchical_state["class_crop_values"], dtype=np.int64
        ).reshape(-1)
        self.n_calendar_classes = int(hierarchical_state["n_calendar_classes"])
        self.tabular_imputer = self.bundle["tabular_imputer"]
        target_columns = tuple(self.bundle.get("target_columns", ()))
        expected_target_columns = ("crop_1", "crop_1_start_date", "crop_1_duration")
        if target_columns and target_columns != expected_target_columns:
            raise ValueError(
                f"Unsupported target-column contract {target_columns}; expected "
                f"{expected_target_columns}."
            )
        raw_target_classes = np.asarray(self.bundle["target_encoder_classes"])
        if raw_target_classes.ndim != 2 or raw_target_classes.shape[1] != 3:
            raise ValueError("Target calendar classes must have shape (class, 3).")
        if not np.all(np.isfinite(raw_target_classes)) or not np.all(
            np.isclose(raw_target_classes, np.rint(raw_target_classes))
        ):
            raise ValueError("Target calendar classes must contain finite integers.")
        self.target_classes = np.rint(raw_target_classes).astype(np.int32)
        if np.unique(self.target_classes, axis=0).shape[0] != len(self.target_classes):
            raise ValueError("Target calendar classes contain duplicate rows.")
        if self.n_calendar_classes != len(self.target_classes):
            raise ValueError(
                "Hierarchical RF calendar-class count disagrees with target vocabulary."
            )
        if self.class_crop_values.shape != (len(self.target_classes),):
            raise ValueError(
                "Hierarchical RF class_crop_values must contain one crop per calendar class."
            )
        if not np.array_equal(
            self.class_crop_values, self.target_classes[:, 0].astype(np.int64)
        ):
            raise ValueError(
                "Hierarchical RF class_crop_values disagree with target calendar crops."
            )
        if (
            self.hierarchical_crop_values.ndim != 1
            or self.hierarchical_crop_values.size == 0
        ):
            raise ValueError(
                "Hierarchical RF crop_values must be a non-empty 1D array."
            )
        if (
            np.unique(self.hierarchical_crop_values).size
            != self.hierarchical_crop_values.size
        ):
            raise ValueError("Hierarchical RF crop_values contain duplicates.")
        for crop in self.hierarchical_crop_values:
            crop = int(crop)
            if (
                crop not in self.conditional_forests
                or crop not in self.conditional_class_ids
            ):
                raise KeyError(
                    f"Hierarchical RF is missing conditional state for crop {crop}."
                )
            expected_class_ids = np.flatnonzero(self.class_crop_values == crop).astype(
                np.int64
            )
            if not np.array_equal(
                np.sort(self.conditional_class_ids[crop]), expected_class_ids
            ):
                raise ValueError(
                    f"Conditional calendar-class IDs disagree for crop {crop}."
                )

        imputer_input_width = getattr(self.tabular_imputer, "n_features_in_", None)
        if imputer_input_width is not None and int(imputer_input_width) != len(
            self.raw_tabular_feature_names
        ):
            raise ValueError(
                "Tabular imputer input width disagrees with raw feature names."
            )
        if hasattr(self.tabular_imputer, "get_feature_names_out"):
            imputer_feature_names = [
                str(name)
                for name in self.tabular_imputer.get_feature_names_out(
                    self.raw_tabular_feature_names
                )
            ]
            if imputer_feature_names != list(self.bundle["rf_tabular_feature_names"]):
                raise ValueError(
                    "Tabular imputer output names disagree with the saved RF contract."
                )

        encoder_timer = time.perf_counter()
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise TypeError("Temporal crop-choice checkpoint is missing a state_dict.")
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("Temporal checkpoint state_dict is not a mapping.")
        dimensions = dict(checkpoint.get("model_dimensions", {}))

        n_daily = len(self.daily_feature_names)
        continuous_value_indices = tuple(
            index
            for index, name in enumerate(self.daily_feature_names)
            if name.endswith("__lag_0")
        )
        decision_value_indices = tuple(
            index for index in range(n_daily) if index not in continuous_value_indices
        )
        seasonal_indices = (2 * n_daily, 2 * n_daily + 1)
        continuous_sequence_indices = (
            continuous_value_indices
            + tuple(n_daily + index for index in continuous_value_indices)
            + seasonal_indices
        )
        decision_sequence_indices = (
            decision_value_indices
            + tuple(n_daily + index for index in decision_value_indices)
            + seasonal_indices
        )
        if (
            tuple(dimensions.get("continuous_sequence_indices", ()))
            != continuous_sequence_indices
        ):
            raise ValueError(
                "Checkpoint continuous feature partition differs from runtime."
            )
        if (
            tuple(dimensions.get("decision_sequence_indices", ()))
            != decision_sequence_indices
        ):
            raise ValueError(
                "Checkpoint decision-history feature partition differs from runtime."
            )
        self.continuous_value_indices = np.asarray(
            continuous_value_indices, dtype=np.int64
        )
        self.decision_value_indices = np.asarray(decision_value_indices, dtype=np.int64)
        self.continuous_feature_names = [
            self.daily_feature_names[index] for index in continuous_value_indices
        ]
        self.continuous_raw_names = []
        for feature_name in self.continuous_feature_names:
            if not feature_name.endswith("__lag_0"):
                raise ValueError(
                    f"Persistent feature {feature_name!r} is not a lag-0 channel."
                )
            raw_name = feature_name[: -len("__lag_0")]
            if raw_name not in self.raw_daily_index:
                raise KeyError(
                    f"No raw online input for persistent feature {feature_name!r}."
                )
            self.continuous_raw_names.append(raw_name)
        self.decision_feature_names = [
            self.daily_feature_names[index] for index in decision_value_indices
        ]

        window_days = int(dimensions["window_days"])
        window_stride_days = int(dimensions["window_stride_days"])
        if window_days != 10 or window_stride_days != 10:
            raise ValueError(
                "The final runtime requires non-overlapping 10-day blocks."
            )
        continuous_input_channels = int(dimensions["continuous_input_channels"])
        decision_input_channels = int(dimensions["decision_history_input_channels"])
        if continuous_input_channels != len(continuous_sequence_indices):
            raise ValueError("Continuous checkpoint input width is inconsistent.")
        if decision_input_channels != len(decision_sequence_indices):
            raise ValueError("Decision-history checkpoint input width is inconsistent.")
        block_embedding_dim = int(dimensions["block_embedding_dim"])
        gru_hidden_dim = int(dimensions["gru_hidden_dim"])
        gru_num_layers = int(dimensions["gru_num_layers"])
        latent_dim = int(dimensions["latent_dim"])
        decision_embedding_dim = int(
            dimensions["decision_history_window_embedding_dim"]
        )
        decision_hidden_dim = int(dimensions["decision_history_hidden_dim"])
        decision_latent_dim = int(dimensions["decision_history_latent_dim"])

        def conv_channels(prefix: str) -> int:
            key = f"{prefix}.network.0.weight"
            if key not in state_dict:
                raise KeyError(f"Temporal checkpoint is missing {key!r}.")
            return int(state_dict[key].shape[0])

        class TemporalWindowEncoder(nn.Module):
            def __init__(
                self,
                input_channels: int,
                convolution_channels: int,
                embedding_dim: int,
            ) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        convolution_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv1d(
                        convolution_channels,
                        convolution_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Flatten(),
                    nn.Linear(convolution_channels * window_days, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                )

            def forward(self, windows: Any) -> Any:
                return self.network(windows.transpose(1, 2))

        class OnlineTemporalEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_encoder = TemporalWindowEncoder(
                    continuous_input_channels,
                    conv_channels("window_encoder"),
                    block_embedding_dim,
                )
                self.temporal_encoder = nn.GRU(
                    input_size=block_embedding_dim,
                    hidden_size=gru_hidden_dim,
                    num_layers=gru_num_layers,
                    batch_first=True,
                    dropout=0.15 if gru_num_layers > 1 else 0.0,
                )
                self.continuous_latent_projection = nn.Sequential(
                    nn.Linear(gru_hidden_dim, latent_dim),
                    nn.GELU(),
                    nn.LayerNorm(latent_dim),
                )
                self.decision_history_window_encoder = TemporalWindowEncoder(
                    decision_input_channels,
                    conv_channels("decision_history_window_encoder"),
                    decision_embedding_dim,
                )
                self.decision_history_temporal_encoder = nn.GRU(
                    input_size=decision_embedding_dim,
                    hidden_size=decision_hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )
                self.decision_history_projection = nn.Sequential(
                    nn.Linear(decision_hidden_dim, decision_latent_dim),
                    nn.GELU(),
                    nn.LayerNorm(decision_latent_dim),
                )
                self.latent_fusion = nn.Sequential(
                    nn.Linear(latent_dim + decision_latent_dim, latent_dim),
                    nn.GELU(),
                    nn.LayerNorm(latent_dim),
                )

            def decision_history_latent(self, sequence: Any) -> Any:
                n_days = int(sequence.shape[1])
                n_windows = (
                    int(np.ceil(max(n_days - window_days, 0) / window_stride_days)) + 1
                )
                padded_days = (n_windows - 1) * window_stride_days + window_days
                if padded_days > n_days:
                    sequence = nn.functional.pad(
                        sequence,
                        (0, 0, 0, padded_days - n_days),
                        value=0.0,
                    )
                windows = (
                    sequence.unfold(
                        dimension=1,
                        size=window_days,
                        step=window_stride_days,
                    )
                    .permute(0, 1, 3, 2)
                    .contiguous()
                )
                flat = windows.reshape(
                    windows.shape[0] * windows.shape[1],
                    window_days,
                    windows.shape[3],
                )
                embeddings = self.decision_history_window_encoder(flat).reshape(
                    windows.shape[0], windows.shape[1], decision_embedding_dim
                )
                _, hidden = self.decision_history_temporal_encoder(embeddings)
                return self.decision_history_projection(hidden[-1])

        self.encoder = OnlineTemporalEncoder()
        encoder_prefixes = (
            "window_encoder.",
            "temporal_encoder.",
            "continuous_latent_projection.",
            "decision_history_window_encoder.",
            "decision_history_temporal_encoder.",
            "decision_history_projection.",
            "latent_fusion.",
        )
        encoder_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(encoder_prefixes)
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        del encoder_state, state_dict, checkpoint

        online_state = self.bundle.get("online_temporal_state", {})
        expected_online_values = {
            "mode": "continuous_recurrent_state",
            "raw_window_days": window_days,
            "regular_update_stride_days": window_stride_days,
            "stored_block_count": 0,
            "persistent_hidden_layers": gru_num_layers,
            "persistent_hidden_dim": gru_hidden_dim,
            "decision_history_recomputed_only_when_due": True,
            "decision_commits_partial_block": True,
            "decision_tail_uses_temporary_hidden_copy": False,
            "memory_decay_applies_to": "persistent_lag0_daily_gru_state_only",
        }
        for key, expected in expected_online_values.items():
            if online_state.get(key) != expected:
                raise ValueError(
                    f"Deployment online_temporal_state[{key!r}]={online_state.get(key)!r}; "
                    f"expected {expected!r}."
                )
        if not np.isclose(
            float(online_state.get("memory_half_life_days", np.nan)),
            self.FINAL_MEMORY_HALF_LIFE_DAYS,
        ):
            raise ValueError("online_temporal_state has the wrong memory half-life.")
        if (
            list(online_state.get("decision_conditioned_daily_features", []))
            != self.decision_feature_names
        ):
            raise ValueError(
                "Deployment decision-conditioned feature names differ from checkpoint partition."
            )

        requested_device = str(config.get("device", "auto")).strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"DecisionModuleML requested {requested_device!r}, but CUDA is unavailable."
            )
        self.device = torch.device(requested_device)
        self.encoder.to(self.device).eval()
        self.window_days = window_days
        self.window_stride_days = window_stride_days
        self.continuous_input_channels = continuous_input_channels
        self.decision_history_input_channels = decision_input_channels
        self.block_embedding_dim = block_embedding_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.latent_dim = latent_dim
        self.batch_size = max(1, int(config.get("batch_size", 2048)))
        self.daily_state_batch_size = max(
            self.batch_size, int(config.get("daily_state_batch_size", 65536))
        )
        # Spin-up replay is primarily I/O-bound. Read a long contiguous period from
        # every reporter in one operation, with an independent memory cap below. A
        # one-year requested chunk substantially reduces Zarr call overhead for small
        # and medium models, while the byte-based limiter automatically shortens it for
        # large Europe models.
        self.spinup_daily_chunk_days = max(
            1, int(config.get("spinup_daily_chunk_days", 365))
        )
        self.spinup_bootstrap_max_buffer_mb = max(
            64.0, float(config.get("spinup_bootstrap_max_buffer_mb", 4096.0))
        )
        # During block-oriented warm replay, encode this many ten-day windows at most
        # in one CNN call. This bounds temporary standardized/input tensors separately
        # from the reporter read buffer.
        self.spinup_encoder_window_batch_size = max(
            1, int(config.get("spinup_encoder_window_batch_size", 16384))
        )
        # Candidate planting days depend only on prescribed calendars, region, and
        # target year. Warm historical replay asks for the same combinations often.
        self._candidate_start_days_cache: dict[tuple[int, int], ArrayInt32] = {}
        # Deterministic calendar argmax. Legacy top_k/random_seed settings no
        # longer affect action selection; planting feasibility is retained.

        latent_feature_names = list(self.bundle.get("latent_feature_names", []))
        if latent_feature_names and len(latent_feature_names) != latent_dim:
            raise ValueError(
                "Saved latent feature names disagree with checkpoint width."
            )
        rf_tabular_feature_names = list(self.bundle["rf_tabular_feature_names"])
        expected_rf_width = latent_dim + len(rf_tabular_feature_names)
        crop_rf_input_width = getattr(self.crop_forest, "n_features_in_", None)
        if crop_rf_input_width is None or int(crop_rf_input_width) != expected_rf_width:
            raise ValueError(
                "Crop-stage RF input width disagrees with latent/tabular contract: "
                f"forest={crop_rf_input_width}, expected={expected_rf_width}."
            )
        crop_rf_classes = np.asarray(getattr(self.crop_forest, "classes_", []))
        if crop_rf_classes.ndim != 1 or crop_rf_classes.size == 0:
            raise ValueError(
                "The deployed crop-stage RF has no one-dimensional classes_."
            )
        if not np.all(np.isfinite(crop_rf_classes)) or not np.all(
            np.isclose(crop_rf_classes, np.rint(crop_rf_classes))
        ):
            raise ValueError("Crop-stage RF class labels must be finite integers.")
        crop_rf_classes = np.rint(crop_rf_classes).astype(np.int64)
        if not np.all(np.isin(crop_rf_classes, self.hierarchical_crop_values)):
            raise ValueError("Crop-stage RF classes fall outside saved crop_values.")
        for crop, conditional_forest in self.conditional_forests.items():
            class_ids = self.conditional_class_ids[crop]
            if conditional_forest is None:
                if class_ids.size != 1:
                    raise ValueError(
                        f"Crop {crop} has no conditional forest but {class_ids.size} "
                        "calendar classes."
                    )
                continue
            conditional_width = getattr(conditional_forest, "n_features_in_", None)
            if conditional_width is None or int(conditional_width) != expected_rf_width:
                raise ValueError(
                    f"Conditional RF for crop {crop} has feature width "
                    f"{conditional_width}; expected {expected_rf_width}."
                )
            conditional_classes = np.asarray(
                getattr(conditional_forest, "classes_", []), dtype=np.int64
            )
            if conditional_classes.ndim != 1 or conditional_classes.size == 0:
                raise ValueError(
                    f"Conditional RF for crop {crop} has invalid classes_."
                )
            if not np.all(np.isin(conditional_classes, class_ids)):
                raise ValueError(
                    f"Conditional RF classes for crop {crop} fall outside its saved "
                    "calendar-class IDs."
                )
        self.model.logger.info(
            "DecisionModuleML timing: checkpoint validation and encoder load=%.2f s.",
            time.perf_counter() - encoder_timer,
        )

        runtime_state_timer = time.perf_counter()
        self.n_farmers = int(farmers.var.n)
        if self.n_farmers < 0:
            raise ValueError("CropFarmers.var.n cannot be negative.")
        base_years = np.asarray(farmers.var.crop_calendar_years, dtype=np.int32)
        if base_years.ndim != 1 or base_years.size == 0:
            raise ValueError("crop_calendar_years must be a non-empty 1D array.")
        if np.unique(base_years).size != base_years.size or np.any(
            np.diff(base_years) <= 0
        ):
            raise ValueError(
                "crop_calendar_years must be unique and strictly increasing."
            )
        base_calendar_shape = farmers.var.crop_calendar_base_array.shape
        if (
            base_calendar_shape[0] != base_years.size
            or base_calendar_shape[1] < self.n_farmers
            or base_calendar_shape[2] < 1
            or base_calendar_shape[3] < 4
        ):
            raise ValueError("crop_calendar_base_array has incompatible dimensions.")
        runtime_calendar_shape = farmers.var.crop_calendar.shape
        if (
            len(runtime_calendar_shape) != 3
            or runtime_calendar_shape[0] < self.n_farmers
            or runtime_calendar_shape[1] < 1
            or runtime_calendar_shape[2] < 4
        ):
            raise ValueError(
                "CropFarmers.var.crop_calendar has incompatible dimensions."
            )
        region_ids = np.asarray(farmers.var.region_id).reshape(-1)
        if region_ids.shape != (self.n_farmers,) or not np.all(np.isfinite(region_ids)):
            raise ValueError(
                "ML crop switching requires one finite region_id per farmer."
            )
        if not np.issubdtype(region_ids.dtype, np.integer):
            if not np.all(region_ids == np.floor(region_ids)):
                raise ValueError("region_id values must be integral.")
        if np.any(region_ids < 0):
            raise ValueError("region_id values must be non-negative.")
        rotation_years = np.asarray(farmers.var.crop_calendar_rotation_years).reshape(
            -1
        )
        if rotation_years.shape != (self.n_farmers,) or np.any(rotation_years <= 0):
            raise ValueError(
                "Each farmer requires a positive crop-calendar rotation length."
            )

        # Final persistent-state runtime: one GRU hidden state plus one raw
        # 10-day lag-0 block. No annual array of block embeddings is retained.
        self.continuous_hidden = np.zeros(
            (self.gru_num_layers, self.n_farmers, self.gru_hidden_dim),
            dtype=np.float32,
        )
        self.continuous_raw_window = np.full(
            (self.n_farmers, self.window_days, len(self.continuous_raw_names)),
            np.nan,
            dtype=np.float32,
        )
        self.continuous_block_start_date = np.full(
            self.n_farmers, np.datetime64("NaT", "D"), dtype="datetime64[D]"
        )
        self.continuous_window_days = np.zeros(self.n_farmers, dtype=np.uint8)
        self.continuous_interval_days = np.zeros(self.n_farmers, dtype=np.int32)
        self.continuous_started = np.zeros(self.n_farmers, dtype=bool)
        self.first_decision_done = np.zeros(self.n_farmers, dtype=bool)
        self.first_state_start_date = np.full(
            self.n_farmers, np.datetime64("NaT", "D"), dtype="datetime64[D]"
        )
        self.last_decision_date = np.full(
            self.n_farmers, np.datetime64("NaT", "D"), dtype="datetime64[D]"
        )

        self.spei_month_values = np.full(
            (self.n_farmers, self.spei_month_capacity),
            np.nan,
            dtype=np.float32,
        )
        self.pending_target_year = np.full(self.n_farmers, -1, dtype=np.int32)
        self.pending_due_date = np.full(
            self.n_farmers, np.datetime64("NaT", "D"), dtype="datetime64[D]"
        )
        self.awaiting_decision = np.zeros(self.n_farmers, dtype=bool)
        self.calendar_is_predicted = np.zeros(self.n_farmers, dtype=bool)
        self.completed_prescribed_before_run = np.zeros(self.n_farmers, dtype=bool)
        self.yield_recorded_year = np.full(self.n_farmers, -1, dtype=np.int32)

        calendar_history_depth = 1
        for name in ("crop_1", "crop_1_start_date", "crop_1_duration"):
            settings = self.variable_config["decision_agent"].get(name, {})
            if settings.get("enabled", True):
                lags = [int(value) for value in settings.get("lags", [0])]
                if lags:
                    calendar_history_depth = max(calendar_history_depth, max(lags) + 1)
        self.calendar_history = np.full(
            (self.n_farmers, calendar_history_depth, 3),
            np.nan,
            dtype=np.float32,
        )
        yield_settings = self.variable_config["decision_agent"].get(
            "crop_decision_yield_ratio", {}
        )
        yield_history_depth = 1
        if yield_settings.get("enabled", True):
            yield_lags = [int(value) for value in yield_settings.get("lags", [0])]
            yield_rolling = [int(value) for value in yield_settings.get("rolling", [])]
            if yield_lags:
                yield_history_depth = max(yield_history_depth, max(yield_lags) + 1)
            if yield_rolling:
                yield_history_depth = max(yield_history_depth, max(yield_rolling))
        self.yield_history = np.full(
            (self.n_farmers, yield_history_depth), np.nan, dtype=np.float32
        )

        self.store_runtime_diagnostics = bool(
            config.get("store_runtime_diagnostics", False)
        )
        if self.store_runtime_diagnostics:
            self.latent: TwoDArrayFloat32 | None = np.full(
                (self.n_farmers, self.latent_dim), np.nan, dtype=np.float32
            )
            self.predicted_calendar: TwoDArrayInt32 | None = np.full(
                (self.n_farmers, 3), -1, dtype=np.int32
            )
            self.prediction_probability: ArrayFloat32 | None = np.full(
                self.n_farmers, np.nan, dtype=np.float32
            )
        else:
            self.latent = None
            self.predicted_calendar = None
            self.prediction_probability = None

        self.last_daily_date: np.datetime64 | None = None
        self.latest_spei_month_ordinal: int | None = None
        self.dynamic_history_start = np.datetime64(
            f"{int(base_years.min()):04d}-01-01", "D"
        )

        active_indices = np.asarray(
            farmers.var.crop_calendar_active_year_index
        ).reshape(-1)
        if active_indices.shape != (self.n_farmers,):
            raise ValueError(
                "crop_calendar_active_year_index must contain one value per farmer."
            )
        if np.any(active_indices < 0) or np.any(active_indices >= len(base_years)):
            raise ValueError("Invalid active prescribed crop-calendar year index.")
        self.calendar_year = np.asarray(base_years[active_indices], dtype=np.int32)
        farmer_indices = np.arange(self.n_farmers, dtype=np.int64)
        for lag in range(self.calendar_history.shape[1]):
            source_indices = active_indices - lag
            valid = source_indices >= 0
            if np.any(valid):
                self.calendar_history[valid, lag, :] = (
                    farmers.var.crop_calendar_base_array[
                        source_indices[valid], farmer_indices[valid], 0, :3
                    ]
                )
        self.model.logger.info(
            "DecisionModuleML timing: compact recurrent-state allocation/history "
            "setup=%.2f s for %s farmers.",
            time.perf_counter() - runtime_state_timer,
            self.n_farmers,
        )

        supporting_inputs_timer = time.perf_counter()
        self.economic_data = {
            name: load_economic_data(self.model.files["dict"][ML_ECONOMIC_INPUTS[name]])
            for name, settings in self.variable_config["economic"].items()
            if settings.get("enabled", True)
        }
        self.regional_farmer_values = self._build_regional_farmer_values()
        self.model.logger.info(
            "DecisionModuleML timing: economic/regional supporting inputs=%.2f s.",
            time.perf_counter() - supporting_inputs_timer,
        )

        run_start = pd.Timestamp(self.model.config["general"]["start_time"])
        initial_schedule_timer = time.perf_counter()
        # The restored farmer state is authoritative at the spin-up/run boundary.
        # A farmer may already be in the run-start calendar year, so preparing all
        # farmers for ``run_start.year`` can point backwards. The first ML target is
        # always the year immediately following that farmer's restored calendar.
        self.schedule(
            np.arange(self.n_farmers, dtype=np.int64),
            target_years=(self.calendar_year + 1).astype(np.int32, copy=False),
            _bootstrap=True,
        )
        self.model.logger.info(
            "DecisionModuleML timing: initial decision scheduling=%.2f s.",
            time.perf_counter() - initial_schedule_timer,
        )
        self._bootstrap_from_spinup_reports(run_start)

        completed = np.flatnonzero(self.completed_prescribed_before_run).astype(
            np.int64
        )
        if completed.size:
            self.schedule(completed, activate=True)
            self.model.logger.info(
                "Activated %s initial ML decision(s) whose final prescribed calendar "
                "completed during spin-up.",
                completed.size,
            )

        self.model.logger.info(
            "Loaded final online crop-choice model from %s: persistent state=%sx%s, "
            "decision-history channels=%s, memory half-life=%g days, prediction=%s, "
            "selection=%s.",
            self.model_directory,
            self.gru_num_layers,
            self.gru_hidden_dim,
            self.decision_history_input_channels,
            self.memory_half_life_days,
            self.prediction_model,
            self.selection_mode,
        )
        self.model.logger.info(
            "DecisionModuleML timing: total initialization=%.2f s.",
            time.perf_counter() - initialization_timer,
        )

    def _spinup_report_directory(self) -> Path:
        """Locate the completed spin-up report directory.

        Returns:
            Directory containing the reporter Zarr stores used for ML bootstrap.
        """
        configured = self.config.get("spinup_report_directory")
        if configured:
            return Path(configured).expanduser()

        output_folder = Path(self.model.config["general"]["output_folder"])
        if output_folder.is_absolute():
            return output_folder / "spinup" / "report"
        return Path.cwd() / output_folder / "spinup" / "report"

    def _reporter_name(self, variable_name: str) -> str:
        """Map an ML input variable to the GEB reporter that stores it.

        Returns:
            Reporter name used below the spin-up report directory.
        """
        overrides = self.config.get("spinup_reporter_overrides", {})
        if variable_name in overrides:
            return str(overrides[variable_name])
        if variable_name in ML_LAND_SURFACE_DAILY_FEATURES:
            return "hydrology.landsurface"
        return "agents.crop_farmers"

    def _open_spinup_report(
        self,
        variable_name: str,
        *,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> tuple[Any, Any, str]:
        """Open and validate one farmer-aligned spin-up Zarr reporter lazily.

        The dataset remains lazy so bootstrap reads only the required dates instead of
        materializing the complete reporter in memory.

        Returns:
            The open dataset, farmer-aligned data array, and farmer dimension name.

        Raises:
            FileNotFoundError: If the requested reporter does not exist.
            ValueError: If the reporter has an unexpected variable or dimension layout.
        """
        import xarray as xr

        zarr_path = (
            self._spinup_report_directory()
            / self._reporter_name(variable_name)
            / f"{variable_name}.zarr"
        )
        if not zarr_path.exists():
            raise FileNotFoundError(
                "DecisionModuleML requires the completed spin-up reporter "
                f"{variable_name!r}, but {zarr_path} does not exist."
            )

        dataset = xr.open_dataset(
            zarr_path,
            engine="zarr",
            chunks=None,
            consolidated=False,
            mask_and_scale=False,
        )
        if variable_name in dataset.data_vars:
            data = dataset[variable_name]
        elif len(dataset.data_vars) == 1:
            data = dataset[next(iter(dataset.data_vars))].rename(variable_name)
        else:
            dataset.close()
            raise ValueError(
                f"Expected one variable in {zarr_path}, found "
                f"{list(dataset.data_vars)}."
            )

        if "time" not in data.dims:
            dataset.close()
            raise ValueError(
                f"Spin-up reporter {variable_name!r} has no time dimension."
            )
        agent_dimensions = [dimension for dimension in data.dims if dimension != "time"]
        if len(agent_dimensions) != 1:
            dataset.close()
            raise ValueError(
                f"Spin-up reporter {variable_name!r} must have one farmer "
                f"dimension besides time, found {data.dims}."
            )
        agent_dimension = agent_dimensions[0]
        if int(data.sizes[agent_dimension]) != self.n_farmers:
            dataset.close()
            raise ValueError(
                f"Spin-up reporter {variable_name!r} has "
                f"{data.sizes[agent_dimension]} farmers, expected {self.n_farmers}."
            )

        if start_time is not None or end_time is not None:
            data = data.sel(
                time=slice(
                    str(start_time) if start_time is not None else None,
                    str(end_time) if end_time is not None else None,
                )
            )
        data = data.transpose("time", agent_dimension)
        return dataset, data, agent_dimension

    def _month_ordinal(self, timestamp: pd.Timestamp) -> int:
        """Convert a timestamp to an absolute zero-based month number.

        Returns:
            Integer month ordinal used by the circular SPEI cache.
        """
        return int(timestamp.year * 12 + timestamp.month - 1)

    def _bootstrap_yield_history(self, run_start: pd.Timestamp) -> None:
        """Reconstruct harvest AND completed-fallow yield-event history.

        The reporters are scanned backwards in bounded chunks so only the newest yield
        observations required by the trained lag configuration are retained.
        Completed fallow calendars are then merged as NaN events by calendar
        index, preserving lag/rolling positions without imputing a crop yield.

        Raises:
            ValueError: If the sparse decision and yield reporters are misaligned or
                contain invalid calendar indices.
        """
        if "crop_decision_yield_ratio" not in self.variable_config["decision_agent"]:
            return
        if not self.variable_config["decision_agent"]["crop_decision_yield_ratio"].get(
            "enabled", True
        ):
            return

        yield_bootstrap_timer = time.perf_counter()
        final_spinup_day = run_start - pd.Timedelta(days=1)
        history_start = pd.Timestamp(
            self.dynamic_history_start.astype("datetime64[ns]")
        )
        active_dataset, active_data, _ = self._open_spinup_report(
            "crop_decision_active_year_index",
            start_time=history_start,
            end_time=final_spinup_day,
        )
        yield_dataset, yield_data, _ = self._open_spinup_report(
            "crop_decision_yield_ratio",
            start_time=history_start,
            end_time=final_spinup_day,
        )
        try:
            active_dates = pd.DatetimeIndex(np.asarray(active_data["time"].values))
            yield_dates = pd.DatetimeIndex(np.asarray(yield_data["time"].values))
            if not active_dates.equals(yield_dates):
                raise ValueError(
                    "Spin-up active-calendar-index and yield-ratio reporters are "
                    "not time aligned."
                )

            depth = self.yield_history.shape[1]
            history_indices = np.full((self.n_farmers, depth), -1, dtype=np.int32)
            event_count = np.zeros(self.n_farmers, dtype=np.int16)
            chunk_days = max(1, int(self.config.get("spinup_history_chunk_days", 32)))
            stop = len(active_dates)
            crop_calendar_years = np.asarray(
                self.farmers.var.crop_calendar_years, dtype=np.int32
            )
            current_active_indices = np.asarray(
                self.farmers.var.crop_calendar_active_year_index
            )

            while stop > 0 and np.any(event_count < depth):
                start = max(0, stop - chunk_days)
                active_block = np.asarray(
                    active_data.isel(time=slice(start, stop)).values
                )
                yield_block = np.asarray(
                    yield_data.isel(time=slice(start, stop)).values,
                    dtype=np.float32,
                )
                for row in range(active_block.shape[0] - 1, -1, -1):
                    active_indices = np.asarray(active_block[row])
                    selected = np.flatnonzero(
                        (active_indices >= 0) & (event_count < depth)
                    )
                    if selected.size == 0:
                        continue
                    selected_indices = active_indices[selected]
                    valid = selected_indices < crop_calendar_years.size
                    if not np.all(valid):
                        invalid = selected[~valid][:20]
                        raise ValueError(
                            "Spin-up decision reporter contains invalid active "
                            f"calendar indices for farmers {invalid.tolist()}."
                        )
                    slots = event_count[selected]
                    self.yield_history[selected, slots] = yield_block[row, selected]
                    history_indices[selected, slots] = selected_indices
                    newest = slots == 0
                    if np.any(newest):
                        newest_farmers = selected[newest]
                        newest_active_indices = selected_indices[newest]
                        self.yield_recorded_year[newest_farmers] = crop_calendar_years[
                            newest_active_indices
                        ]
                        # During the final prescribed HRL year, CropFarmers records
                        # the decision event before trying to advance. Without an ML
                        # module present in spin-up there is no next calendar to
                        # install, so the active base index remains unchanged. Match
                        # that final sparse event to the restored active index to
                        # identify decisions that must start in the operational run.
                        completed_final = (
                            newest_active_indices
                            == current_active_indices[newest_farmers]
                        )
                        if np.any(completed_final):
                            self.completed_prescribed_before_run[
                                newest_farmers[completed_final]
                            ] = True
                    event_count[selected] += 1
                stop = start

            # The sparse spin-up reporter contains harvests only. Offline
            # training synthesizes a decision for each all--1 fallow calendar;
            # reconstruct those missing slots from prescribed calendars here.
            # Retaining the newest `depth` harvests first is sufficient: merging
            # fallow events can only push existing harvests farther back.
            inserted_fallow_events = 0
            all_farmers = np.arange(self.n_farmers, dtype=np.int64)
            for calendar_index, calendar_year in enumerate(crop_calendar_years):
                completion = pd.Timestamp(year=int(calendar_year) + 1, month=1, day=1)
                if completion > run_start.normalize():
                    continue  # Never invent an event for an unfinished year.
                eligible = all_farmers[current_active_indices >= calendar_index]
                for start in range(0, eligible.size, self.batch_size):
                    batch = eligible[start : start + self.batch_size]
                    calendars = self.farmers.var.crop_calendar_base_array[
                        calendar_index, batch, :, :
                    ]
                    fallow = np.all(calendars == -1, axis=(1, 2))
                    selected = batch[fallow]
                    if selected.size == 0:
                        continue
                    # Also tolerate a future reporter that explicitly records
                    # fallow, without inserting the same event twice.
                    duplicate = np.any(
                        history_indices[selected] == calendar_index, axis=1
                    )
                    selected = selected[~duplicate]
                    if selected.size == 0:
                        continue
                    merged_indices = np.column_stack(
                        (
                            history_indices[selected],
                            np.full(selected.size, calendar_index, dtype=np.int32),
                        )
                    )
                    merged_yields = np.column_stack(
                        (
                            self.yield_history[selected],
                            np.full(selected.size, np.nan, dtype=np.float32),
                        )
                    )
                    order = np.argsort(-merged_indices, axis=1, kind="stable")[
                        :, :depth
                    ]
                    history_indices[selected] = np.take_along_axis(
                        merged_indices, order, axis=1
                    )
                    self.yield_history[selected] = np.take_along_axis(
                        merged_yields, order, axis=1
                    )
                    inserted_fallow_events += selected.size
            newest_indices = history_indices[:, 0]
            recorded = newest_indices >= 0
            self.yield_recorded_year[recorded] = crop_calendar_years[
                newest_indices[recorded]
            ]
            self.model.logger.info(
                "Merged %s completed-fallow yield events into spin-up history; "
                "NaN events retain their lag positions.",
                inserted_fallow_events,
            )
        finally:
            active_dataset.close()
            yield_dataset.close()

        self.model.logger.info(
            "DecisionModuleML timing: spin-up yield-history bootstrap=%.2f s.",
            time.perf_counter() - yield_bootstrap_timer,
        )

    def _prescribed_decision_dates(
        self,
        farmers: ArrayInt64,
        target_years: ArrayInt32,
        *,
        warn_fallback: bool,
    ) -> ArrayDatetime64:
        """Compute trained decision dates from prescribed HRL calendars only.

        This helper is used by warm recurrent-state bootstrap. It deliberately does
        not mutate scheduling state and never invokes the RF. Historical prescribed
        calendars supply only the previous-harvest constraint and the regional
        candidate planting-day vocabulary needed to reproduce the offline decision
        timing rule.

        Args:
            farmers: Farmer indices whose historical decision dates are requested.
            target_years: Prescribed target calendar year for every farmer.
            warn_fallback: Whether to log the normal no-candidate fallback. Historical
                warm replay normally disables these warnings to avoid log spam.

        Returns:
            One absolute ``datetime64[D]`` decision date per farmer.

        Raises:
            ValueError: If ``farmers`` and ``target_years`` are not aligned 1D
                arrays, or if a requested target year has no preceding prescribed
                calendar year from which to derive the harvest constraint.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        target_years = np.asarray(target_years, dtype=np.int32)
        if farmers.shape != target_years.shape or farmers.ndim != 1:
            raise ValueError("farmers and target_years must be aligned 1D arrays.")
        if farmers.size == 0:
            return np.empty(0, dtype="datetime64[D]")

        source_years = target_years - 1
        base_years = np.asarray(self.farmers.var.crop_calendar_years, dtype=np.int32)
        source_indices = np.searchsorted(base_years, source_years)
        valid_source = source_indices < base_years.size
        valid_source &= (
            base_years[np.minimum(source_indices, base_years.size - 1)] == source_years
        )
        if not np.all(valid_source):
            missing_years = np.unique(source_years[~valid_source])
            raise ValueError(
                "Historical recurrent-state replay requires the preceding prescribed "
                f"calendar year. Missing years: {missing_years.tolist()}."
            )

        source_calendars = self.farmers.var.crop_calendar_base_array[
            source_indices, farmers, :, :
        ]
        active_indices = np.asarray(
            self.farmers.var.crop_calendar_active_year_index[farmers], dtype=np.int32
        )
        rotation_years = np.asarray(
            self.farmers.var.crop_calendar_rotation_years[farmers], dtype=np.int32
        )
        source_rotation_indices = (
            np.asarray(
                self.farmers.var.current_crop_calendar_rotation_year_index[farmers],
                dtype=np.int32,
            )
            + source_indices
            - active_indices
        ) % rotation_years

        valid_source_crops = (
            (source_calendars[:, :, 0] >= 0)
            & (source_calendars[:, :, 1] >= 0)
            & (source_calendars[:, :, 2] > 0)
            & (source_calendars[:, :, 3] == source_rotation_indices[:, None])
        )
        harvest_offsets = np.where(
            valid_source_crops,
            source_calendars[:, :, 1] + source_calendars[:, :, 2],
            -1,
        ).max(axis=1)
        has_harvest_constraint = harvest_offsets >= 0
        source_year_starts = (source_years - 1970).astype("datetime64[Y]")
        harvest_dates = source_year_starts + np.maximum(harvest_offsets, 0).astype(
            "timedelta64[D]"
        )

        region_ids = np.asarray(self.farmers.var.region_id[farmers], dtype=np.int64)
        due_dates = np.full(farmers.size, np.datetime64("NaT", "D"))
        for pair in np.unique(np.column_stack((region_ids, target_years)), axis=0):
            region_id, target_year = int(pair[0]), int(pair[1])
            positions = np.flatnonzero(
                (region_ids == region_id) & (target_years == target_year)
            )
            start_days = self._candidate_start_days(region_id, target_year)
            year_start = np.datetime64(f"{target_year:04d}-01-01", "D")
            days_in_year = 366 if calendar.isleap(target_year) else 365
            candidates = (
                year_start + start_days[(start_days >= 0) & (start_days < days_in_year)]
            )
            for position in positions:
                if candidates.size:
                    if not has_harvest_constraint[position]:
                        due_dates[position] = candidates[0]
                    else:
                        feasible = candidates[candidates >= harvest_dates[position]]
                        if feasible.size:
                            due_dates[position] = feasible[0]
                if np.isnat(due_dates[position]):
                    due_dates[position] = (
                        harvest_dates[position]
                        if has_harvest_constraint[position]
                        else year_start
                    )
                    if warn_fallback:
                        target_end = np.datetime64(f"{target_year + 1:04d}-01-01", "D")
                        outside = not (year_start <= due_dates[position] < target_end)
                        self.model.logger.warning(
                            "No feasible historical candidate planting date for farmer %s "
                            "(region %s, target %s); using %s%s.",
                            int(farmers[position]),
                            region_id,
                            target_year,
                            due_dates[position],
                            " outside the nominal target year" if outside else "",
                        )
        return due_dates

    def _historical_state_boundaries(
        self,
        run_start: pd.Timestamp,
    ) -> dict[np.datetime64, ArrayInt64]:
        """Build prescribed historical decision boundaries for warm GRU replay.

        The offline encoder carries recurrent state across decision events. Operational
        GEB intentionally does not make ML choices during spin-up, so initialization
        reconstructs those *state boundaries* afterwards from prescribed calendars.
        No crop calendar is predicted or installed here.

        Args:
            run_start: First timestamp of the operational run. Historical boundaries
                on or after this day are excluded from warm replay.

        Returns:
            Mapping from absolute decision day to the farmers crossing a recurrent
            state boundary on that day. Only boundaries strictly before ``run_start``
            are returned.
        """
        years = np.asarray(self.farmers.var.crop_calendar_years, dtype=np.int32)
        if years.size < 2:
            return {}

        boundary_lists: dict[np.datetime64, list[ArrayInt64]] = {}
        earliest_boundary = np.full(
            self.n_farmers, np.datetime64("NaT", "D"), dtype="datetime64[D]"
        )
        all_farmers = np.arange(self.n_farmers, dtype=np.int64)
        run_start_day = np.datetime64(run_start.date(), "D")

        # A prescribed decision for target year Y exists only for farmers whose
        # restored prescribed state has reached at least Y. The first base year has no
        # preceding prescribed calendar and therefore cannot define this timing rule.
        for target_year in years[1:]:
            eligible = all_farmers[self.calendar_year >= int(target_year)]
            if eligible.size == 0:
                continue
            target_years = np.full(eligible.size, int(target_year), dtype=np.int32)
            due_dates = self._prescribed_decision_dates(
                eligible, target_years, warn_fallback=False
            )
            historical = due_dates < run_start_day
            if not np.any(historical):
                continue
            eligible = eligible[historical]
            due_dates = due_dates[historical]
            for day in np.unique(due_dates):
                selected = eligible[due_dates == day]
                boundary_lists.setdefault(day, []).append(selected)
            missing = np.isnat(earliest_boundary[eligible])
            earliest_boundary[eligible[missing]] = due_dates[missing]
            existing = ~missing
            if np.any(existing):
                farmers_existing = eligible[existing]
                earliest_boundary[farmers_existing] = np.minimum(
                    earliest_boundary[farmers_existing], due_dates[existing]
                )

        has_history = ~np.isnat(earliest_boundary)
        if np.any(has_history):
            warm_starts = earliest_boundary[has_history] - np.timedelta64(
                self.days_per_year, "D"
            )
            farmers = all_farmers[has_history]
            current = self.first_state_start_date[farmers]
            replace = np.isnat(current) | (warm_starts < current)
            self.first_state_start_date[farmers[replace]] = warm_starts[replace]

        boundaries: dict[np.datetime64, ArrayInt64] = {}
        for day, arrays in boundary_lists.items():
            boundaries[day] = np.unique(np.concatenate(arrays)).astype(
                np.int64, copy=False
            )
        if boundaries:
            self.model.logger.info(
                "Prepared %s historical recurrent-state boundary day(s) spanning %s "
                "through %s for warm spin-up replay.",
                len(boundaries),
                min(boundaries),
                max(boundaries),
            )
        return boundaries

    def _commit_historical_state_boundary(
        self,
        farmers: ArrayInt64,
        boundary_day: np.datetime64,
    ) -> None:
        """Commit a prescribed historical decision boundary without making an ML choice.

        Args:
            farmers: Farmer indices crossing the historical recurrent-state boundary.
            boundary_day: Absolute day of the prescribed historical decision boundary.

        Raises:
            RuntimeError: If committing the boundary leaves an incomplete recurrent
                block for any affected farmer.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        if farmers.size == 0:
            return
        active = farmers[self.continuous_started[farmers]]
        if active.size == 0:
            return

        partial_counts = self.continuous_window_days[active].astype(np.int64)
        for elapsed_days in np.unique(partial_counts[partial_counts > 0]):
            selected = active[partial_counts == elapsed_days]
            self._commit_continuous_blocks(selected, int(elapsed_days))
        if np.any(self.continuous_window_days[active] != 0):
            raise RuntimeError(
                "Historical recurrent-state boundary left an uncommitted partial block."
            )

        # Mark the recurrent state as warm. This flag is about the state trajectory,
        # not whether a historical RF prediction was made: no RF is evaluated here.
        self.first_decision_done[active] = True
        self.last_decision_date[active] = boundary_day
        self.continuous_interval_days[active] = 0

    def _bootstrap_write_partial_days(
        self,
        farmers: ArrayInt64,
        position: int,
        dates: pd.DatetimeIndex,
        blocks: dict[str, TwoDArray],
        row_start: int,
        row_stop: int,
    ) -> None:
        """Append a contiguous bootstrap slice to retained raw recurrent blocks.

        This is the small residual path used only when a replay segment begins or ends
        inside a ten-day encoder block. Complete ten-day windows are handled directly
        by :meth:`_bootstrap_commit_full_windows` instead of being replayed day by day.

        Args:
            farmers: Farmer indices sharing the same current position in their raw block.
            position: Zero-based write position in the retained ten-day block.
            dates: Full date index for the currently loaded reporter chunk.
            blocks: Farmer-aligned reporter values for the currently loaded chunk.
            row_start: Inclusive row offset in ``dates``/``blocks``.
            row_stop: Exclusive row offset in ``dates``/``blocks``.

        Raises:
            KeyError: If a required persistent lag-0 predictor is absent from ``blocks``.
            ValueError: If the requested slice does not fit in the ten-day retained block.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        position = int(position)
        row_start = int(row_start)
        row_stop = int(row_stop)
        n_days = row_stop - row_start
        if farmers.size == 0 or n_days == 0:
            return
        if position < 0 or position + n_days > self.window_days:
            raise ValueError(
                "Bootstrap partial slice does not fit in the retained block."
            )

        first_day = np.datetime64(dates[row_start].date(), "D")
        if position == 0:
            self.continuous_block_start_date[farmers] = first_day

        dynamic_rows = (
            dates[row_start:row_stop].values.astype("datetime64[D]")
            < self.dynamic_history_start
        )
        for column, raw_name in enumerate(self.continuous_raw_names):
            if raw_name not in blocks:
                raise KeyError(
                    f"Missing bootstrap persistent daily variable {raw_name!r}."
                )
            values = np.asarray(
                blocks[raw_name][row_start:row_stop, :][:, farmers],
                dtype=np.float32,
            ).T
            settings = self.variable_config["daily_agent"][raw_name]
            if settings.get("history_scope", "dynamic") == "dynamic" and np.any(
                dynamic_rows
            ):
                values = values.copy()
                values[:, dynamic_rows] = np.nan
            self.continuous_raw_window[
                farmers, position : position + n_days, column
            ] = values
        self.continuous_window_days[farmers] = np.uint8(position + n_days)

    def _bootstrap_commit_full_windows(
        self,
        farmers: ArrayInt64,
        dates: pd.DatetimeIndex,
        blocks: dict[str, TwoDArray],
        row_start: int,
        n_blocks: int,
    ) -> None:
        """Commit consecutive complete ten-day windows without daily Python replay.

        All CNN windows for one farmer batch are encoded together. The resulting block
        embeddings are then advanced through the GRU sequentially on the selected
        device, applying the trained ten-day hidden-state decay before each recurrent
        update. Hidden state crosses the CPU/device boundary only once per farmer batch.

        Args:
            farmers: Farmer indices whose retained raw block is currently empty.
            dates: Full date index for the currently loaded reporter chunk.
            blocks: Farmer-aligned reporter values for the currently loaded chunk.
            row_start: First reporter row belonging to the first complete window.
            n_blocks: Number of consecutive complete ten-day windows to commit.

        Raises:
            KeyError: If a required persistent lag-0 predictor is absent from ``blocks``.
            RuntimeError: If selected farmers already retain a partial recurrent block.
            ValueError: If ``n_blocks`` is negative or the requested windows exceed the
                loaded reporter chunk.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        row_start = int(row_start)
        n_blocks = int(n_blocks)
        if farmers.size == 0 or n_blocks == 0:
            return
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative.")
        if np.any(self.continuous_window_days[farmers] != 0):
            raise RuntimeError(
                "Direct bootstrap window commits require an empty retained raw block."
            )

        n_days = n_blocks * self.window_days
        row_stop = row_start + n_days
        if row_start < 0 or row_stop > len(dates):
            raise ValueError(
                "Requested bootstrap windows exceed the loaded date block."
            )

        date_slice = dates[row_start:row_stop]
        dynamic_rows = (
            date_slice.values.astype("datetime64[D]") < self.dynamic_history_start
        )
        sin_day, cos_day = self._canonical_seasonal_coordinates(
            date_slice.values.astype("datetime64[ns]")
        )
        sin_day = sin_day.reshape(n_blocks, self.window_days)
        cos_day = cos_day.reshape(n_blocks, self.window_days)
        means = self.daily_mean[self.continuous_value_indices]
        scales = self.daily_scale[self.continuous_value_indices]
        n_values = len(self.continuous_raw_names)
        decay = float(
            np.float32(2.0 ** (-float(self.window_days) / self.memory_half_life_days))
        )

        # Bound temporary encoder tensors by number of flattened ten-day windows, not
        # just by farmer count. This remains safe when a long reporter chunk contains
        # many complete windows per farmer.
        farmer_batch_size = max(
            1,
            min(
                self.daily_state_batch_size,
                self.spinup_encoder_window_batch_size // max(1, n_blocks),
            ),
        )
        for start in range(0, farmers.size, farmer_batch_size):
            batch = farmers[start : start + farmer_batch_size]
            raw = np.empty(
                (batch.size, n_blocks, self.window_days, n_values),
                dtype=np.float32,
            )
            for column, raw_name in enumerate(self.continuous_raw_names):
                if raw_name not in blocks:
                    raise KeyError(
                        f"Missing bootstrap persistent daily variable {raw_name!r}."
                    )
                values = np.asarray(
                    blocks[raw_name][row_start:row_stop, :][:, batch],
                    dtype=np.float32,
                ).T
                if self.variable_config["daily_agent"][raw_name].get(
                    "history_scope", "dynamic"
                ) == "dynamic" and np.any(dynamic_rows):
                    values = values.copy()
                    values[:, dynamic_rows] = np.nan
                raw[:, :, :, column] = values.reshape(
                    batch.size, n_blocks, self.window_days
                )

            observed = np.isfinite(raw)
            standardized = (raw - means[None, None, None, :]) / scales[
                None, None, None, :
            ]
            standardized[~observed] = 0.0
            sequence = np.zeros(
                (
                    batch.size,
                    n_blocks,
                    self.window_days,
                    self.continuous_input_channels,
                ),
                dtype=np.float32,
            )
            sequence[..., :n_values] = standardized
            sequence[..., n_values : 2 * n_values] = observed
            sequence[..., -2] = sin_day[None, :, :]
            sequence[..., -1] = cos_day[None, :, :]

            flat = sequence.reshape(
                batch.size * n_blocks,
                self.window_days,
                self.continuous_input_channels,
            )
            window = self.torch.from_numpy(flat).to(self.device)
            hidden = self.torch.from_numpy(self.continuous_hidden[:, batch, :]).to(
                self.device
            )
            with self.torch.inference_mode():
                embeddings = self.encoder.window_encoder(window).reshape(
                    batch.size, n_blocks, self.block_embedding_dim
                )
                for block_index in range(n_blocks):
                    hidden = hidden * decay
                    _, hidden = self.encoder.temporal_encoder(
                        embeddings[:, block_index : block_index + 1, :], hidden
                    )
            self.continuous_hidden[:, batch, :] = (
                hidden.cpu().numpy().astype(np.float32, copy=False)
            )

    def _bootstrap_replay_segment(
        self,
        dates: pd.DatetimeIndex,
        blocks: dict[str, TwoDArray],
        row_start: int,
        row_stop: int,
    ) -> int:
        """Replay one event-free date segment in recurrent ten-day units.

        Farmer start dates and historical decision boundaries are handled outside this
        helper, so the active farmer set is constant throughout the segment. Existing
        partial blocks are completed first, all middle full windows are encoded in bulk,
        and only the final residual days are retained as a partial block.

        Args:
            dates: Full date index for the currently loaded reporter chunk.
            blocks: Farmer-aligned reporter values for the currently loaded chunk.
            row_start: Inclusive segment row offset.
            row_stop: Exclusive segment row offset.

        Returns:
            Number of complete ten-day farmer-windows committed through the bulk path.

        Raises:
            RuntimeError: If replay is non-consecutive with the preceding bootstrap
                segment or an invalid retained block position is encountered.
        """
        row_start = int(row_start)
        row_stop = int(row_stop)
        if row_stop <= row_start:
            return 0
        first_day = np.datetime64(dates[row_start].date(), "D")
        last_day = np.datetime64(dates[row_stop - 1].date(), "D")
        if self.last_daily_date is not None:
            expected = self.last_daily_date + np.timedelta64(1, "D")
            if first_day != expected:
                raise RuntimeError(
                    "Bootstrap crop-choice inputs must be replayed on consecutive days; "
                    f"received {first_day} after {self.last_daily_date}."
                )

        active = np.flatnonzero(self.continuous_started).astype(np.int64)
        n_days = row_stop - row_start
        if active.size == 0:
            self.last_daily_date = last_day
            return 0

        positions = self.continuous_window_days[active].astype(np.int64)
        if np.any(positions >= self.window_days):
            raise RuntimeError("Persistent raw block contains an invalid day position.")

        full_window_count = 0
        for position in np.unique(positions):
            group = active[positions == position]
            offset = 0
            current_position = int(position)

            if current_position > 0:
                needed = self.window_days - current_position
                take = min(needed, n_days)
                self._bootstrap_write_partial_days(
                    group,
                    current_position,
                    dates,
                    blocks,
                    row_start,
                    row_start + take,
                )
                offset += take
                if current_position + take < self.window_days:
                    continue
                self._commit_continuous_blocks(group, self.window_days)

            remaining = n_days - offset
            n_full = remaining // self.window_days
            if n_full:
                self._bootstrap_commit_full_windows(
                    group,
                    dates,
                    blocks,
                    row_start + offset,
                    n_full,
                )
                full_window_count += int(group.size) * int(n_full)
                offset += n_full * self.window_days

            tail = n_days - offset
            if tail:
                self._bootstrap_write_partial_days(
                    group,
                    0,
                    dates,
                    blocks,
                    row_start + offset,
                    row_stop,
                )

        self.continuous_interval_days[active] += np.int32(n_days)
        self.last_daily_date = last_day
        return full_window_count

    def _bootstrap_daily_state(self, run_start: pd.Timestamp) -> None:
        """Warm-replay spin-up days into the persistent recurrent state used online.

        Historical prescribed decision dates are replayed only as recurrent-state
        boundaries. ML remains operational-only: the random forest and switch gate are
        not evaluated and prescribed spin-up crop calendars are never replaced.

        Reporter I/O is performed in large memory-bounded chunks. Within each loaded
        chunk, replay is split only at farmer state starts and historical decision
        boundaries. Event-free spans are then processed in recurrent ten-day units:
        complete windows are encoded in bulk and only residual partial windows use the
        retained raw-block path. This preserves the exact decision-aligned semantics
        while avoiding one Python state update per farmer-day.

        Args:
            run_start: First timestamp of the operational run. Reporter values through
                the preceding day are replayed chronologically.

        Raises:
            RuntimeError: If a recurrent bootstrap start is encountered after its exact
                configured day.
            ValueError: If a required spin-up reporter does not cover the complete
                bootstrap period or if reporter time axes cannot be aligned to the
                requested replay chunk.
        """
        historical_boundaries = self._historical_state_boundaries(run_start)
        valid_start_mask = ~np.isnat(self.first_state_start_date)
        valid_starts = self.first_state_start_date[valid_start_mask]
        if valid_starts.size == 0:
            return
        first_day = pd.Timestamp(valid_starts.min())
        last_day = run_start - pd.Timedelta(days=1)
        if first_day > last_day:
            return

        # Pre-index recurrent-state starts once. Sorting avoids repeatedly scanning the
        # full farmer population for every distinct historical date.
        start_farmers = np.flatnonzero(valid_start_mask).astype(np.int64)
        start_days = self.first_state_start_date[start_farmers]
        order = np.argsort(start_days, kind="stable")
        start_farmers = start_farmers[order]
        start_days = start_days[order]
        start_events: dict[np.datetime64, ArrayInt64] = {}
        if start_days.size:
            split_points = np.flatnonzero(start_days[1:] != start_days[:-1]) + 1
            farmer_groups = np.split(start_farmers, split_points)
            day_groups = np.split(start_days, split_points)
            for day_group, farmer_group in zip(day_groups, farmer_groups):
                start_events[day_group[0]] = farmer_group

        event_days = np.asarray(
            sorted(set(start_events) | set(historical_boundaries)),
            dtype="datetime64[D]",
        )

        timer = time.perf_counter()
        opened: dict[str, tuple[Any, Any, pd.DatetimeIndex]] = {}
        spei_history_dataset: Any | None = None
        bulk_window_count = 0
        try:
            # SPEI needs a longer view for decision-relative lag/rolling history. Open
            # that store once and reuse a sliced view for the lag-0 daily replay.
            spei_history_data: Any | None = None
            spei_history_dates: pd.DatetimeIndex | None = None
            oldest_spei = (
                first_day
                - pd.DateOffset(years=self.maximum_spei_years_back)
                - pd.Timedelta(days=400)
            )
            if "crop_decision_spei" in self.raw_daily_index:
                spei_history_dataset, spei_history_data, _ = self._open_spinup_report(
                    "crop_decision_spei", start_time=oldest_spei, end_time=last_day
                )
                spei_history_dates = pd.DatetimeIndex(
                    np.asarray(spei_history_data["time"].values)
                )
                daily_spei = spei_history_data.sel(
                    time=slice(str(first_day), str(last_day))
                )
                daily_spei_dates = pd.DatetimeIndex(
                    np.asarray(daily_spei["time"].values)
                )
                opened["crop_decision_spei"] = (
                    spei_history_dataset,
                    daily_spei,
                    daily_spei_dates,
                )

            for name in self.raw_daily_names:
                if name == "crop_decision_spei":
                    continue
                dataset, data, _ = self._open_spinup_report(
                    name, start_time=first_day, end_time=last_day
                )
                dates = pd.DatetimeIndex(np.asarray(data["time"].values))
                if dates.empty or dates[-1].normalize() < last_day.normalize():
                    dataset.close()
                    raise ValueError(
                        f"Spin-up reporter {name!r} does not cover {last_day.date()}."
                    )
                opened[name] = (dataset, data, dates)

            if "crop_decision_spei" in opened:
                _, _, dates = opened["crop_decision_spei"]
                if dates.empty or dates[-1].normalize() < last_day.normalize():
                    raise ValueError(
                        "Spin-up reporter 'crop_decision_spei' does not cover "
                        f"{last_day.date()}."
                    )

            spei_position_by_ordinal: dict[int, int] = {}
            if spei_history_dates is not None:
                for position, date in enumerate(spei_history_dates):
                    ordinal = self._month_ordinal(date) + (int(date.day) > 15)
                    spei_position_by_ordinal.setdefault(ordinal, position)

            def advance_spei_cache(timestamp: pd.Timestamp) -> None:
                if spei_history_data is None:
                    return
                desired = self._month_ordinal(timestamp) + (int(timestamp.day) > 15)
                if self.latest_spei_month_ordinal is None:
                    first_ordinal = desired - self.spei_month_capacity + 1
                else:
                    first_ordinal = self.latest_spei_month_ordinal + 1
                for ordinal in range(first_ordinal, desired + 1):
                    position = spei_position_by_ordinal.get(ordinal)
                    if position is None:
                        continue
                    values = np.asarray(
                        spei_history_data.isel(time=position).values,
                        dtype=np.float32,
                    ).reshape(-1)
                    self.spei_month_values[:, ordinal % self.spei_month_capacity] = (
                        values
                    )
                    self.latest_spei_month_ordinal = ordinal
                if (
                    self.latest_spei_month_ordinal is None
                    or self.latest_spei_month_ordinal < desired
                ):
                    self.latest_spei_month_ordinal = desired

            full_dates = pd.date_range(first_day, last_day, freq="D")
            requested_chunk_days = self.spinup_daily_chunk_days
            max_buffer_mb = self.spinup_bootstrap_max_buffer_mb
            bytes_per_day = max(1, self.n_farmers * len(self.raw_daily_names) * 4)
            memory_limited_days = max(
                1, int((max_buffer_mb * 1024.0 * 1024.0) // bytes_per_day)
            )
            chunk_days = min(requested_chunk_days, memory_limited_days)
            self.model.logger.info(
                "Warm spin-up replay spans %s day(s); reporter chunk=%s day(s) "
                "(requested=%s, buffer budget=%.0f MiB); recurrent replay uses "
                "%s-day blocks.",
                len(full_dates),
                chunk_days,
                requested_chunk_days,
                max_buffer_mb,
                self.window_days,
            )

            def aligned_block(
                data: Any,
                dates: pd.DatetimeIndex,
                chunk_dates: pd.DatetimeIndex,
            ) -> TwoDArray:
                target64 = chunk_dates.values.astype("datetime64[ns]")
                positions = dates.searchsorted(chunk_dates)
                valid = positions < len(dates)
                if np.any(valid):
                    valid_indices = np.flatnonzero(valid)
                    valid_positions = positions[valid_indices]
                    matches = (
                        dates.values[valid_positions].astype("datetime64[ns]")
                        == target64[valid_indices]
                    )
                    valid[valid_indices[~matches]] = False
                output = np.full(
                    (len(chunk_dates), self.n_farmers), np.nan, dtype=np.float32
                )
                if not np.any(valid):
                    return output
                selected_positions = positions[valid]
                lo = int(selected_positions.min())
                hi = int(selected_positions.max()) + 1
                raw = np.asarray(data.isel(time=slice(lo, hi)).values, dtype=np.float32)
                output[np.flatnonzero(valid)] = raw[selected_positions - lo]
                return output

            for chunk_start in range(0, len(full_dates), chunk_days):
                chunk_stop = min(chunk_start + chunk_days, len(full_dates))
                chunk_dates = full_dates[chunk_start:chunk_stop]
                blocks = {
                    name: aligned_block(data, dates, chunk_dates)
                    for name, (_, data, dates) in opened.items()
                }
                chunk_first = np.datetime64(chunk_dates[0].date(), "D")
                chunk_last = np.datetime64(chunk_dates[-1].date(), "D")
                left = int(np.searchsorted(event_days, chunk_first, side="left"))
                right = int(np.searchsorted(event_days, chunk_last, side="right"))
                chunk_events = event_days[left:right]
                segment_days = np.unique(
                    np.concatenate(
                        (np.asarray([chunk_first], dtype="datetime64[D]"), chunk_events)
                    )
                )

                for segment_index, segment_day in enumerate(segment_days):
                    boundary_farmers = historical_boundaries.get(segment_day)
                    if boundary_farmers is not None:
                        self._commit_historical_state_boundary(
                            boundary_farmers, segment_day
                        )

                    starting = start_events.get(segment_day)
                    if starting is not None:
                        late = self.continuous_started[starting]
                        if np.any(late):
                            affected = starting[late][:20]
                            raise RuntimeError(
                                "A recurrent crop-choice bootstrap state was already "
                                "active on its configured first day for farmers "
                                f"{affected.tolist()}."
                            )
                        self.continuous_started[starting] = True

                    if segment_index + 1 < len(segment_days):
                        next_day = segment_days[segment_index + 1]
                    else:
                        next_day = chunk_last + np.timedelta64(1, "D")
                    local_start = int((segment_day - chunk_first).astype(np.int64))
                    local_stop = int((next_day - chunk_first).astype(np.int64))
                    bulk_window_count += self._bootstrap_replay_segment(
                        chunk_dates, blocks, local_start, local_stop
                    )

            # Historical warm replay never evaluates decision-time SPEI features. The
            # final circular cache therefore only needs the state it would have after
            # the final replayed day; populate all retained monthly ordinals in one go.
            advance_spei_cache(last_day)
        finally:
            # crop_decision_spei reuses spei_history_dataset and must be closed once.
            closed_ids: set[int] = set()
            for dataset, _, _ in opened.values():
                if id(dataset) not in closed_ids:
                    dataset.close()
                    closed_ids.add(id(dataset))
            if (
                spei_history_dataset is not None
                and id(spei_history_dataset) not in closed_ids
            ):
                spei_history_dataset.close()

        self.model.logger.info(
            "DecisionModuleML timing: warm spin-up persistent-state bootstrap=%.2f s "
            "(%s complete ten-day farmer-window(s) bulk committed).",
            time.perf_counter() - timer,
            bulk_window_count,
        )

    def _bootstrap_from_spinup_reports(
        self,
        run_start: pd.Timestamp,
    ) -> None:
        """Reconstruct history and initialize the first persistent neural state.

        Args:
            run_start: First timestamp of the operational run.

        Raises:
            FileNotFoundError: If the completed spin-up report directory is missing.
        """
        timer = time.perf_counter()
        report_directory = self._spinup_report_directory()
        if not report_directory.exists():
            raise FileNotFoundError(
                "DecisionModuleML is enabled, but the completed spin-up report "
                f"directory does not exist: {report_directory}."
            )
        self._bootstrap_yield_history(run_start)
        self._bootstrap_daily_state(run_start)
        self.model.logger.info(
            "Initialized DecisionModuleML from completed spin-up reporters in %s (%.2f s).",
            report_directory,
            time.perf_counter() - timer,
        )

    @staticmethod
    def _canonical_seasonal_coordinates(
        dates: ArrayDatetime64,
    ) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Return the fixed leap-year seasonal coordinates used in training.

        Args:
            dates: One-dimensional array of daily timestamps.

        Returns:
            Tuple containing float32 sine and cosine seasonal coordinates for each date.
        """
        date_index = pd.DatetimeIndex(np.asarray(dates, dtype="datetime64[ns]"))
        slots = np.asarray(
            [
                datetime(2000, int(month), int(day)).timetuple().tm_yday - 1
                for month, day in zip(date_index.month, date_index.day)
            ],
            dtype=np.float32,
        )
        angle = 2.0 * np.pi * slots / 366.0
        return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)

    def _continuous_block_inputs(
        self,
        farmers: ArrayInt64,
        elapsed_days: int,
    ) -> ThreeDArrayFloat32:
        """Transform retained lag-0 days into one padded encoder block.

        Raw daily values are retained instead of already-expanded value-and-mask rows,
        which approximately halves the dominant ten-day block memory. A final partial
        decision-aligned block is padded with all-zero rows exactly as in training.

        Args:
            farmers: Farmer indices whose current raw block is transformed.
            elapsed_days: Number of real observed days represented by the block. Must be
                between one and ``window_days`` inclusive.

        Returns:
            Float32 encoder input with shape
            ``(n_farmers, window_days, continuous_input_channels)``.

        Raises:
            RuntimeError: If a selected farmer has no recorded start date for the current
                persistent block.
            ValueError: If ``elapsed_days`` falls outside the valid block length.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        elapsed_days = int(elapsed_days)
        if elapsed_days <= 0 or elapsed_days > self.window_days:
            raise ValueError("elapsed_days must be within the ten-day block.")
        raw = np.asarray(self.continuous_raw_window[farmers], dtype=np.float32)
        values = raw[:, :elapsed_days, :].copy()
        observed = np.isfinite(values)
        means = self.daily_mean[self.continuous_value_indices]
        scales = self.daily_scale[self.continuous_value_indices]
        values = (values - means[None, None, :]) / scales[None, None, :]
        values[~observed] = 0.0

        sequence = np.zeros(
            (farmers.size, self.window_days, self.continuous_input_channels),
            dtype=np.float32,
        )
        n_values = values.shape[2]
        sequence[:, :elapsed_days, :n_values] = values
        sequence[:, :elapsed_days, n_values : 2 * n_values] = observed

        starts = self.continuous_block_start_date[farmers]
        if np.any(np.isnat(starts)):
            raise RuntimeError("A persistent block has no recorded start date.")
        date_matrix = starts[:, None] + np.arange(elapsed_days).astype("timedelta64[D]")
        flat_dates = date_matrix.reshape(-1)
        sin_day, cos_day = self._canonical_seasonal_coordinates(flat_dates)
        sequence[:, :elapsed_days, -2] = sin_day.reshape(farmers.size, elapsed_days)
        sequence[:, :elapsed_days, -1] = cos_day.reshape(farmers.size, elapsed_days)
        return sequence

    def _commit_continuous_blocks(
        self,
        farmers: ArrayInt64,
        elapsed_days: int,
    ) -> None:
        """Commit one full or padded decision-aligned block to persistent GRU state.

        The carried hidden state is decayed using the number of real elapsed days before
        the block embedding is passed through the GRU. For a partial decision block, the
        CNN input is padded to ten days while the decay still uses only the real days.

        Args:
            farmers: Farmer indices whose current block is committed.
            elapsed_days: Number of real observed days represented by the block.

        Raises:
            RuntimeError: If the retained per-farmer block-day count does not match the
                requested commit length.
            ValueError: If ``elapsed_days`` is outside the valid block length.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        if farmers.size == 0:
            return
        elapsed_days = int(elapsed_days)
        if elapsed_days <= 0 or elapsed_days > self.window_days:
            raise ValueError(
                f"A recurrent block must represent 1..{self.window_days} real days."
            )
        expected_counts = self.continuous_window_days[farmers]
        if np.any(expected_counts != elapsed_days):
            raise RuntimeError(
                "Persistent block day counts do not match the requested commit length."
            )
        decay = np.float32(2.0 ** (-float(elapsed_days) / self.memory_half_life_days))
        # Recurrent-only updates are substantially lighter than complete RF
        # inference. Use the larger state batch during both bootstrap and live block
        # commits to reduce GPU/CPU launch overhead.
        for start in range(0, farmers.size, self.daily_state_batch_size):
            batch = farmers[start : start + self.daily_state_batch_size]
            block_inputs = self._continuous_block_inputs(batch, elapsed_days)
            window = self.torch.from_numpy(block_inputs).to(self.device)
            hidden = self.torch.from_numpy(self.continuous_hidden[:, batch, :]).to(
                self.device
            )
            hidden = hidden * float(decay)
            with self.torch.inference_mode():
                embeddings = self.encoder.window_encoder(window)
                _, updated = self.encoder.temporal_encoder(
                    embeddings[:, None, :], hidden
                )
            self.continuous_hidden[:, batch, :] = (
                updated.cpu().numpy().astype(np.float32, copy=False)
            )
        self.continuous_raw_window[farmers] = np.nan
        self.continuous_block_start_date[farmers] = np.datetime64("NaT", "D")
        self.continuous_window_days[farmers] = 0

    def capture_daily(
        self,
        timestamp: datetime,
        land_surface_values: dict[str, Array],
    ) -> None:
        """Capture one completed live day and update persistent neural state.

        Args:
            timestamp: Timestamp of the completed model day.
            land_surface_values: Farmer-aligned land-surface predictor arrays for the day.

        Raises:
            RuntimeError: If the operational farmer population size differs from the
                population used to initialize this runtime.
        """
        if int(self.farmers.var.n) != self.n_farmers:
            raise RuntimeError(
                "DecisionModuleML requires a fixed farmer population during a run."
            )
        farmer_values = {
            "groundwater_depth": np.asarray(
                self.farmers.groundwater_depth, dtype=np.float32
            ),
            "cumulative_water_deficit_current_day": np.asarray(
                self.farmers.var.cumulative_water_deficit_current_day,
                dtype=np.float32,
            ),
            "risk_perception": np.asarray(
                self.farmers.var.risk_perception, dtype=np.float32
            ),
            "crop_decision_spei": np.asarray(
                self.farmers.var.crop_decision_spei, dtype=np.float32
            ),
        }
        self._capture_daily_values(timestamp, {**land_surface_values, **farmer_values})

    def _capture_daily_values(
        self,
        timestamp: datetime,
        values_by_name: dict[str, Array],
    ) -> None:
        """Validate one day and append it to every active persistent state stream.

        The same path is used by live simulation days and spin-up replay. Complete
        ten-day blocks are committed immediately; an incomplete block remains raw until
        either it reaches ten days or a crop decision commits it.

        Args:
            timestamp: Timestamp represented by ``values_by_name``.
            values_by_name: Mapping from required raw daily feature name to a
                farmer-aligned one-dimensional array.

        Raises:
            KeyError: If one or more required online daily predictor arrays are missing.
            RuntimeError: If daily captures are non-consecutive or a persistent block has
                an inconsistent internal day count.
            ValueError: If a daily predictor is not aligned with the farmer population.
        """
        current_date = np.datetime64(timestamp.date(), "D")
        if self.last_daily_date is not None:
            expected = self.last_daily_date + np.timedelta64(1, "D")
            if current_date == self.last_daily_date:
                return
            if current_date != expected:
                raise RuntimeError(
                    "Online crop-choice inputs must be captured on consecutive days; "
                    f"received {current_date} after {self.last_daily_date}."
                )
        missing = sorted(set(self.raw_daily_names) - set(values_by_name))
        if missing:
            raise KeyError(f"Missing online daily crop-choice values: {missing}.")

        current_values: dict[str, Array] = {}
        for name in self.raw_daily_names:
            values = np.asarray(values_by_name[name], dtype=np.float32).reshape(-1)
            if values.shape != (self.n_farmers,):
                raise ValueError(
                    f"Daily crop-choice variable {name!r} has shape {values.shape}; "
                    f"expected {(self.n_farmers,)}."
                )
            settings = self.variable_config["daily_agent"][name]
            history_scope = settings.get("history_scope", "dynamic")
            if history_scope == "dynamic" and current_date < self.dynamic_history_start:
                values = np.full_like(values, np.nan)
            current_values[name] = values
        can_start = (
            ~self.continuous_started
            & ~np.isnat(self.first_state_start_date)
            & (current_date >= self.first_state_start_date)
        )
        missed_start = can_start & (current_date > self.first_state_start_date)
        if np.any(missed_start):
            affected = np.flatnonzero(missed_start)[:20]
            raise RuntimeError(
                "A first persistent crop-choice state started after its exact "
                f"366-day bootstrap date. Affected farmers: {affected.tolist()}."
            )
        self.continuous_started[can_start] = True
        active = np.flatnonzero(self.continuous_started).astype(np.int64)
        if active.size == 0:
            self.last_daily_date = current_date
            return

        # Fill the retained raw block in bounded chunks. Avoid materializing an
        # n_farmers x n_daily_features matrix on every simulation day.
        for start in range(0, active.size, self.daily_state_batch_size):
            batch = active[start : start + self.daily_state_batch_size]
            positions = self.continuous_window_days[batch].astype(np.int64)
            if np.any(positions >= self.window_days):
                raise RuntimeError(
                    "Persistent raw block contains an invalid day position."
                )
            new_block = positions == 0
            if np.any(new_block):
                self.continuous_block_start_date[batch[new_block]] = current_date
            for column, raw_name in enumerate(self.continuous_raw_names):
                self.continuous_raw_window[batch, positions, column] = current_values[
                    raw_name
                ][batch]
            self.continuous_window_days[batch] += 1
            self.continuous_interval_days[batch] += 1

        complete = active[self.continuous_window_days[active] == self.window_days]
        if complete.size:
            self._commit_continuous_blocks(complete, self.window_days)
        self.last_daily_date = current_date

    def _spei_values_for_dates(
        self,
        farmers: ArrayInt64,
        dates: ArrayDatetime64,
    ) -> TwoDArrayFloat32:
        """Read nearest-month SPEI values from the compact circular cache.

        Args:
            farmers: Farmer indices for which SPEI is requested.
            dates: Dates whose nearest-month SPEI values are requested.

        Returns:
            Float32 matrix with shape ``(n_farmers, n_dates)``. Dates outside the cache
            coverage remain ``NaN`` and are handled by the trained observation masks.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        date_index = pd.DatetimeIndex(np.asarray(dates, dtype="datetime64[ns]"))
        ordinals = (
            date_index.year.to_numpy(dtype=np.int64) * 12
            + date_index.month.to_numpy(dtype=np.int64)
            - 1
            + (date_index.day.to_numpy(dtype=np.int64) > 15)
        )
        output = np.full((farmers.size, len(ordinals)), np.nan, dtype=np.float32)
        if self.latest_spei_month_ordinal is None:
            return output
        valid = (ordinals <= self.latest_spei_month_ordinal) & (
            ordinals > self.latest_spei_month_ordinal - self.spei_month_capacity
        )
        if np.any(valid):
            output[:, valid] = self.spei_month_values[
                farmers[:, None],
                (ordinals[valid] % self.spei_month_capacity)[None, :],
            ]
        return output

    def _decision_history_sequence(
        self,
        farmers: ArrayInt64,
        predictor_timestamp: datetime,
    ) -> ThreeDArrayFloat32:
        """Rebuild the 366-day decision-relative SPEI branch for due farmers.

        Args:
            farmers: Farmer indices whose crop decision is being evaluated.
            predictor_timestamp: Decision timestamp. The sequence ends one day before
                this timestamp, matching the training contract.

        Returns:
            Float32 decision-history encoder sequence in the exact feature order stored
            by the deployment bundle.

        Raises:
            ValueError: If the reconstructed decision-history feature order or width
                differs from the trained deployment contract.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        predictor = pd.Timestamp(predictor_timestamp)
        histories: dict[int, TwoDArrayFloat32] = {}
        for years_back in range(self.maximum_spei_years_back + 1):
            shifted = predictor - pd.DateOffset(years=years_back)
            dates = pd.date_range(
                end=shifted - pd.Timedelta(days=1),
                periods=self.days_per_year,
                freq="D",
            ).to_numpy(dtype="datetime64[D]")
            histories[years_back] = self._spei_values_for_dates(farmers, dates)

        transformed_columns: list[TwoDArrayFloat32] = []
        names: list[str] = []
        spei_settings = self.variable_config["daily_agent"]["crop_decision_spei"]
        for kind, value, feature_name in _ml_transform_specs(
            "crop_decision_spei", spei_settings
        ):
            if kind == "lag" and value == 0:
                continue
            if kind == "lag":
                transformed = histories[value]
            else:
                stack = np.stack([histories[offset] for offset in range(value)], axis=0)
                count = np.isfinite(stack).sum(axis=0)
                transformed = np.divide(
                    np.nansum(stack, axis=0),
                    count,
                    out=np.full_like(stack[0], np.nan, dtype=np.float32),
                    where=count > 0,
                )
            transformed_columns.append(np.asarray(transformed, dtype=np.float32))
            names.append(feature_name)
        if names != self.decision_feature_names:
            raise ValueError(
                "Online decision-history feature order differs from training: "
                f"online={names}, trained={self.decision_feature_names}."
            )
        values = np.stack(transformed_columns, axis=2)
        observed = np.isfinite(values)
        means = self.daily_mean[self.decision_value_indices]
        scales = self.daily_scale[self.decision_value_indices]
        values = (values - means[None, None, :]) / scales[None, None, :]
        values[~observed] = 0.0

        base_dates = pd.date_range(
            end=predictor - pd.Timedelta(days=1),
            periods=self.days_per_year,
            freq="D",
        ).to_numpy(dtype="datetime64[D]")
        sin_day, cos_day = self._canonical_seasonal_coordinates(base_dates)
        sequence = np.empty(
            (farmers.size, self.days_per_year, self.decision_history_input_channels),
            dtype=np.float32,
        )
        n_values = values.shape[2]
        sequence[:, :, :n_values] = values
        sequence[:, :, n_values : 2 * n_values] = observed
        sequence[:, :, -2] = sin_day[None, :]
        sequence[:, :, -1] = cos_day[None, :]
        return sequence

    def _build_regional_farmer_values(self) -> dict[str, ArrayFloat32]:
        """Compute configured regional farmer statistics and map them to farmers.

        Returns:
            Mapping from deployed regional-statistic feature name to farmer values.

        Raises:
            ValueError: If a regional-farmer input is not aligned with the farmer array.
        """
        timer = time.perf_counter()
        output: dict[str, ArrayFloat32] = {}
        region_ids = np.asarray(self.farmers.var.region_id)
        for name, settings in self.variable_config["regional_farmer"].items():
            if not settings.get("enabled", True):
                continue
            values = np.asarray(
                read_array(self.model.files["array"][ML_REGIONAL_FARMER_INPUTS[name]])
            ).reshape(-1)
            if values.size != region_ids.size:
                raise ValueError(
                    f"Regional-farmer input {name!r} is not farmer-aligned."
                )
            for statistic in settings.get("statistics", ["mean", "std"]):
                mapped = np.full(region_ids.size, np.nan, dtype=np.float32)
                for region_id in np.unique(region_ids):
                    mask = region_ids == region_id
                    finite_values = values[mask & np.isfinite(values)]
                    if finite_values.size:
                        mapped[mask] = (
                            finite_values.mean()
                            if statistic == "mean"
                            else finite_values.std(ddof=0)
                        )
                output[f"{name}__regional_{statistic}"] = mapped
        self.model.logger.info(
            "DecisionModuleML timing: regional-farmer summary preparation=%.2f s.",
            time.perf_counter() - timer,
        )
        return output

    def _asof_index(self, date_index: DateIndex, timestamp: datetime) -> int | None:
        """Find the latest observation available at or before a timestamp.

        Returns:
            Positional index of the latest observation, or ``None`` if none is available.
        """
        dates = np.asarray(date_index.dates, dtype="datetime64[ns]")
        position = int(
            np.searchsorted(dates, np.datetime64(timestamp), side="right") - 1
        )
        return position if position >= 0 else None

    def _economic_values(
        self,
        name: str,
        farmers: ArrayInt64,
        timestamp: datetime,
    ) -> ArrayFloat32:
        """Map one regional economic variable to the selected farmers.

        Returns:
            Farmer-aligned float32 values using the latest observation available by date.
        """
        date_index, values_by_region = self.economic_data[name]
        position = self._asof_index(date_index, timestamp)
        output = np.full(farmers.size, np.nan, dtype=np.float32)
        if position is None:
            return output
        region_ids = np.asarray(self.farmers.var.region_id[farmers], dtype=np.int64)
        for region_id in np.unique(region_ids):
            output[region_ids == region_id] = values_by_region[int(region_id)][position]
        return output

    def _crop_price_values(
        self,
        farmers: ArrayInt64,
        timestamp: datetime,
    ) -> TwoDArrayFloat32:
        """Map crop prices available at a timestamp to the selected farmers.

        Returns:
            Farmer-by-crop float32 price matrix.
        """
        date_index, values = self.farmers.crop_prices
        n_crops = len(self.farmers.var.crop_data)
        output = np.full((farmers.size, n_crops), np.nan, dtype=np.float32)
        if date_index is None:
            static_values = np.asarray(values, dtype=np.float32)
            if static_values.ndim == 2:
                output[:] = static_values[
                    np.asarray(self.farmers.var.region_id[farmers], dtype=np.int64)
                ]
            else:
                output[:] = static_values
            return output
        position = self._asof_index(date_index, timestamp)
        if position is not None:
            output[:] = np.asarray(values[position], dtype=np.float32)[
                np.asarray(self.farmers.var.region_id[farmers], dtype=np.int64)
            ]
        return output

    def _static_value(self, name: str, farmers: ArrayInt64) -> ArrayFloat32:
        """Read one static deployed feature for the selected farmers.

        Returns:
            Farmer-aligned float32 feature values.

        Raises:
            ValueError: If the source is not a one-dimensional farmer-aligned array.
        """
        if name == "field_size_per_farmer":
            values = self.farmers.field_size_per_farmer
        elif name == "elevation":
            values = self.farmers.var.elevation
        elif name == "slope":
            values = self.farmers.var.slope
        elif name == "region_id":
            values = self.farmers.var.region_id
        elif name == "locations_x":
            values = self.farmers.var.locations[:, 0]
        elif name == "locations_y":
            values = self.farmers.var.locations[:, 1]
        else:
            values = getattr(self.farmers.HRU.var, name)
        values_array = np.asarray(values, dtype=np.float32)
        if values_array.ndim != 1 or values_array.shape[0] < self.n_farmers:
            raise ValueError(
                f"Static ML feature {name!r} is not a one-dimensional "
                "farmer-aligned array."
            )
        return values_array[farmers]

    def _tabular_values(
        self,
        farmers: ArrayInt64,
        predictor_timestamp: datetime,
    ) -> TwoDArrayFloat32:
        """Build and impute the RF tabular features for one decision batch.

        Recreates decision-history, timing, economic, crop-price, regional, and static
        features in the exact order stored in the deployment bundle.

        Returns:
            Imputed float32 RF tabular matrix in training feature order.

        Raises:
            KeyError: If a required deployed feature cannot be constructed.
            ValueError: If a transform is unsupported or the final feature matrix differs
                from the fitted training contract.
        """
        feature_values: dict[str, Array] = {}
        for name, settings in self.variable_config["decision_agent"].items():
            if not settings.get("enabled", True):
                continue
            for kind, value, feature_name in _ml_transform_specs(name, settings):
                if name == "crop_decision_yield_ratio":
                    history = self.yield_history[farmers]
                    if kind == "lag":
                        transformed = history[:, value]
                    else:
                        selected = history[:, :value]
                        count = np.isfinite(selected).sum(axis=1)
                        transformed = np.divide(
                            np.nansum(selected, axis=1),
                            count,
                            out=np.full(farmers.size, np.nan, dtype=np.float32),
                            where=count > 0,
                        )
                else:
                    if kind != "lag":
                        raise ValueError("Crop-calendar features only support lags.")
                    component = (
                        "crop_1",
                        "crop_1_start_date",
                        "crop_1_duration",
                    ).index(name)
                    transformed = self.calendar_history[farmers, value, component]
                feature_values[feature_name] = np.asarray(transformed, dtype=np.float32)

        day_of_year = predictor_timestamp.timetuple().tm_yday
        days_in_year = 366 if calendar.isleap(predictor_timestamp.year) else 365
        angle = 2.0 * np.pi * (day_of_year - 1) / days_in_year
        decision_values = {
            "decision_day_of_year": float(day_of_year),
            "decision_day_of_year_sin": float(np.sin(angle)),
            "decision_day_of_year_cos": float(np.cos(angle)),
        }
        for name, settings in self.variable_config["decision_time"].items():
            if settings.get("enabled", True):
                feature_values[name] = np.full(
                    farmers.size, decision_values[name], dtype=np.float32
                )

        for name, settings in self.variable_config["economic"].items():
            if not settings.get("enabled", True):
                continue
            for kind, value, feature_name in _ml_transform_specs(name, settings):
                if kind == "lag":
                    transformed = self._economic_values(
                        name,
                        farmers,
                        predictor_timestamp - pd.DateOffset(years=value),
                    )
                else:
                    values = [
                        self._economic_values(
                            name,
                            farmers,
                            predictor_timestamp - pd.DateOffset(years=lag),
                        )
                        for lag in range(value)
                    ]
                    stacked = np.stack(values)
                    count = np.isfinite(stacked).sum(axis=0)
                    transformed = np.divide(
                        np.nansum(stacked, axis=0),
                        count,
                        out=np.full(farmers.size, np.nan, dtype=np.float32),
                        where=count > 0,
                    )
                feature_values[feature_name] = transformed

        for name, settings in self.variable_config["regional_crop"].items():
            if not settings.get("enabled", True):
                continue
            for kind, value, transform_name in _ml_transform_specs(name, settings):
                if kind == "lag":
                    transformed = self._crop_price_values(
                        farmers,
                        predictor_timestamp - pd.DateOffset(years=value),
                    )
                else:
                    values = [
                        self._crop_price_values(
                            farmers,
                            predictor_timestamp - pd.DateOffset(years=lag),
                        )
                        for lag in range(value)
                    ]
                    stacked = np.stack(values)
                    count = np.isfinite(stacked).sum(axis=0)
                    transformed = np.divide(
                        np.nansum(stacked, axis=0),
                        count,
                        out=np.full_like(stacked[0], np.nan),
                        where=count > 0,
                    )
                for crop_position, crop_id in enumerate(
                    self.farmers.var.crop_data.index
                ):
                    feature_values[f"{transform_name}__crop_{int(crop_id)}"] = (
                        transformed[:, crop_position]
                    )

        for name, values in self.regional_farmer_values.items():
            feature_values[name] = values[farmers]
        for name, settings in self.variable_config["static_agent"].items():
            if settings.get("enabled", True):
                feature_values[name] = self._static_value(name, farmers)

        missing = [
            name
            for name in self.raw_tabular_feature_names
            if name not in feature_values
        ]
        if missing:
            raise KeyError(f"Online tabular features are missing {missing}.")
        raw = np.column_stack(
            [feature_values[name] for name in self.raw_tabular_feature_names]
        ).astype(np.float32, copy=False)
        transformed = self.tabular_imputer.transform(raw).astype(np.float32, copy=False)
        expected_names = list(self.bundle["rf_tabular_feature_names"])
        if transformed.shape != (farmers.size, len(expected_names)):
            raise ValueError(
                "Online imputed RF feature shape differs from training: "
                f"found {transformed.shape}, expected "
                f"{(farmers.size, len(expected_names))}."
            )
        if np.any(~np.isfinite(transformed)):
            raise ValueError(
                "The fitted tabular imputer returned non-finite online RF inputs."
            )
        return transformed

    def note_completed_yield(
        self,
        farmers: ArrayInt64,
        values: ArrayFloat32,
    ) -> None:
        """Record one completed-calendar yield observation per farmer.

        The newest observation is inserted at lag zero and older lags are shifted only
        when that calendar year has not already been recorded.

        Raises:
            IndexError: If any farmer index is outside the runtime population.
            ValueError: If farmer/value shapes are invalid or farmer indices are duplicated.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        if farmers.ndim != 1 or values.shape != farmers.shape:
            raise ValueError(
                "Completed-yield updates require one value per one-dimensional "
                "farmer index."
            )
        if np.any((farmers < 0) | (farmers >= self.n_farmers)):
            raise IndexError("Completed-yield farmer indices are out of range.")
        if np.unique(farmers).size != farmers.size:
            raise ValueError("Completed-yield farmer indices must be unique.")
        years = self.calendar_year[farmers]
        fresh = self.yield_recorded_year[farmers] != years
        if not np.any(fresh):
            return
        selected = farmers[fresh]
        if self.yield_history.shape[1] > 1:
            self.yield_history[selected, 1:] = self.yield_history[selected, :-1]
        self.yield_history[selected, 0] = values[fresh]
        self.yield_recorded_year[selected] = years[fresh]

    def note_observed_calendar_advance(
        self,
        farmers: ArrayInt64,
        active_indices: ArrayInt32,
    ) -> None:
        """Update ML calendar history after a prescribed HRL calendar advances.

        This keeps training-style lag features synchronized while prescribed calendars
        are still authoritative before ML predictions take over.

        Raises:
            IndexError: If farmer or active-calendar indices are outside valid bounds.
            ValueError: If input shapes are invalid or farmer indices are duplicated.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        active_indices = np.asarray(active_indices, dtype=np.int64)
        if farmers.ndim != 1 or active_indices.shape != farmers.shape:
            raise ValueError(
                "Observed-calendar advancement requires one active index per "
                "one-dimensional farmer index."
            )
        if np.any((farmers < 0) | (farmers >= self.n_farmers)):
            raise IndexError("Observed-calendar farmer indices are out of range.")
        if np.unique(farmers).size != farmers.size:
            raise ValueError("Observed-calendar farmer indices must be unique.")
        if np.any(
            (active_indices < 0)
            | (active_indices >= len(self.farmers.var.crop_calendar_years))
        ):
            raise IndexError("Observed-calendar active indices are out of range.")
        new_calendars = self.farmers.var.crop_calendar_base_array[
            active_indices, farmers, 0, :3
        ]
        if self.calendar_history.shape[1] > 1:
            self.calendar_history[farmers, 1:] = self.calendar_history[farmers, :-1]
        self.calendar_history[farmers, 0] = new_calendars
        self.calendar_is_predicted[farmers] = False
        self.calendar_year[farmers] = self.farmers.var.crop_calendar_years[
            active_indices
        ]

    def _candidate_start_days(self, region_id: int, target_year: int) -> ArrayInt32:
        """Collect historical main-crop planting days for a region and target year.

        Calendar years are processed one at a time to avoid materializing a large
        ``years x farmers`` calendar subset.

        Args:
            region_id: Farmer subregion whose historical planting dates are queried.
            target_year: Target calendar year; only earlier prescribed years contribute.

        Returns:
            Sorted unique planting-day indices observed strictly before the target year.
        """
        cache_key = (int(region_id), int(target_year))
        cached = self._candidate_start_days_cache.get(cache_key)
        if cached is not None:
            return cached

        years = np.asarray(self.farmers.var.crop_calendar_years)
        region_farmers = np.flatnonzero(
            np.asarray(self.farmers.var.region_id) == region_id
        )
        year_indices = np.flatnonzero(years < target_year)
        if region_farmers.size == 0 or year_indices.size == 0:
            result = np.empty(0, dtype=np.int32)
            self._candidate_start_days_cache[cache_key] = result
            return result
        # Only the unique planting days are needed here. Process one calendar
        # year at a time so large subregions do not create a temporary
        # (n_years x n_region_farmers x 3) calendar array.
        start_days: list[ArrayInt32] = []
        for year_index in year_indices:
            calendars = self.farmers.var.crop_calendar_base_array[
                year_index, region_farmers, 0, :3
            ]
            valid = (
                (calendars[:, 0] >= 0) & (calendars[:, 1] >= 0) & (calendars[:, 2] > 0)
            )
            if np.any(valid):
                start_days.append(np.unique(calendars[valid, 1]))
        if not start_days:
            result = np.empty(0, dtype=np.int32)
        else:
            result = np.unique(np.concatenate(start_days)).astype(np.int32, copy=False)
        self._candidate_start_days_cache[cache_key] = result
        return result

    def schedule(
        self,
        farmers: ArrayInt64,
        *,
        target_years: ArrayInt32 | None = None,
        activate: bool = False,
        _bootstrap: bool = False,
    ) -> None:
        """Schedule decision dates without resetting the continuously carried state.

        Scheduling determines only the crop-decision due date and activation status; it
        never starts or resets neural history. The persistent daily state continues to
        update independently of future decision dates.

        Args:
            farmers: One-dimensional array of farmer indices to schedule or activate.
            target_years: Optional target calendar year for each farmer. When omitted,
                the year following each farmer's current calendar year is used.
            activate: Whether the prepared decision should become active immediately.
            _bootstrap: Internal flag used while preparing first operational decisions
                from prescribed spin-up calendars.

        Raises:
            IndexError: If any farmer index lies outside the runtime population.
            RuntimeError: If activation conflicts with the prepared target year or occurs
                after the trained decision date.
            ValueError: If farmer/target-year inputs are malformed, duplicated, or the
                required preceding prescribed calendar year is unavailable.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        if farmers.ndim != 1:
            raise ValueError("farmers must be a one-dimensional index array.")
        if farmers.size == 0:
            return
        if np.any((farmers < 0) | (farmers >= self.n_farmers)):
            raise IndexError("Scheduled farmer indices are out of range.")
        if np.unique(farmers).size != farmers.size:
            raise ValueError("Scheduled farmer indices must be unique.")

        schedule_timer = time.perf_counter()
        requested_farmer_count = farmers.size
        if target_years is not None:
            target_years = np.asarray(target_years, dtype=np.int32)
            if target_years.shape != farmers.shape:
                raise ValueError("target_years must have one value per farmer.")

        already_scheduled = self.pending_target_year[farmers] >= 0
        if activate and np.any(already_scheduled):
            activated = farmers[already_scheduled]
            expected_years = self.calendar_year[activated] + 1
            if np.any(self.pending_target_year[activated] != expected_years):
                raise RuntimeError(
                    "The prepared ML decision does not follow the completed crop year."
                )
            today = np.datetime64(self.model.current_time.date(), "D")
            overdue = self.pending_due_date[activated] < today
            if np.any(overdue):
                raise RuntimeError(
                    "An ML decision was activated after its trained decision date for "
                    f"farmers {activated[overdue][:20].tolist()}."
                )
            self.awaiting_decision[activated] = True
            self.farmers.var.crop_calendar[activated] = -1

        available = ~already_scheduled
        if target_years is not None:
            target_years = target_years[available]
        farmers = farmers[available]
        if farmers.size == 0:
            self.model.logger.info(
                "DecisionModuleML timing: schedule=%.2f s for %s requested farmer(s) "
                "(activation only).",
                time.perf_counter() - schedule_timer,
                requested_farmer_count,
            )
            return

        current_date = np.datetime64(self.model.current_time.date(), "D")
        if target_years is None:
            source_years = self.calendar_year[farmers]
            target_years = source_years + 1
            source_calendars = np.asarray(
                self.farmers.var.crop_calendar[farmers], dtype=np.int32
            )
            source_rotation_indices = np.asarray(
                self.farmers.var.current_crop_calendar_rotation_year_index[farmers],
                dtype=np.int32,
            )
        else:
            source_years = target_years - 1
            base_years = np.asarray(
                self.farmers.var.crop_calendar_years, dtype=np.int32
            )
            source_indices = np.searchsorted(base_years, source_years)
            valid_source = source_indices < base_years.size
            valid_source &= (
                base_years[np.minimum(source_indices, base_years.size - 1)]
                == source_years
            )
            if not np.all(valid_source):
                missing_years = np.unique(source_years[~valid_source])
                raise ValueError(
                    "Preparing the first runtime ML decision requires its preceding "
                    f"HRL calendar year. Missing years: {missing_years.tolist()}."
                )
            source_calendars = self.farmers.var.crop_calendar_base_array[
                source_indices, farmers, :, :
            ]
            active_indices = np.asarray(
                self.farmers.var.crop_calendar_active_year_index[farmers],
                dtype=np.int32,
            )
            rotation_years = np.asarray(
                self.farmers.var.crop_calendar_rotation_years[farmers], dtype=np.int32
            )
            source_rotation_indices = (
                np.asarray(
                    self.farmers.var.current_crop_calendar_rotation_year_index[farmers],
                    dtype=np.int32,
                )
                + source_indices
                - active_indices
            ) % rotation_years

        valid_source_crops = (
            (source_calendars[:, :, 0] >= 0)
            & (source_calendars[:, :, 1] >= 0)
            & (source_calendars[:, :, 2] > 0)
            & (source_calendars[:, :, 3] == source_rotation_indices[:, None])
        )
        harvest_offsets = np.where(
            valid_source_crops,
            source_calendars[:, :, 1] + source_calendars[:, :, 2],
            -1,
        ).max(axis=1)
        has_harvest_constraint = harvest_offsets >= 0
        source_year_starts = (source_years - 1970).astype("datetime64[Y]")
        harvest_dates = source_year_starts + np.maximum(harvest_offsets, 0).astype(
            "timedelta64[D]"
        )

        region_ids = np.asarray(self.farmers.var.region_id[farmers], dtype=np.int64)
        due_dates = np.full(farmers.size, np.datetime64("NaT", "D"))
        for pair in np.unique(np.column_stack((region_ids, target_years)), axis=0):
            region_id, target_year = int(pair[0]), int(pair[1])
            positions = np.flatnonzero(
                (region_ids == region_id) & (target_years == target_year)
            )
            start_days = self._candidate_start_days(region_id, target_year)
            year_start = np.datetime64(f"{target_year:04d}-01-01", "D")
            days_in_year = 366 if calendar.isleap(target_year) else 365
            candidates = year_start + start_days[start_days < days_in_year]
            for position in positions:
                if candidates.size:
                    if not has_harvest_constraint[position]:
                        due_dates[position] = candidates[0]
                    else:
                        feasible = candidates[candidates >= harvest_dates[position]]
                        if feasible.size:
                            due_dates[position] = feasible[0]
                if np.isnat(due_dates[position]):
                    due_dates[position] = (
                        harvest_dates[position]
                        if has_harvest_constraint[position]
                        else year_start
                    )
                    target_end = np.datetime64(f"{target_year + 1:04d}-01-01", "D")
                    outside = not (year_start <= due_dates[position] < target_end)
                    self.model.logger.warning(
                        "No feasible historical candidate planting date for farmer %s "
                        "(region %s, target %s); using %s%s.",
                        int(farmers[position]),
                        region_id,
                        target_year,
                        due_dates[position],
                        " outside the nominal target year" if outside else "",
                    )

        self.pending_target_year[farmers] = target_years
        self.pending_due_date[farmers] = due_dates
        self.awaiting_decision[farmers] = activate
        if activate:
            if np.any(due_dates < current_date):
                affected = farmers[due_dates < current_date][:20]
                raise RuntimeError(
                    "A newly activated ML decision date is already in the past for "
                    f"farmers {affected.tolist()}."
                )
            self.farmers.var.crop_calendar[farmers] = -1

        first_pending = (
            ~self.first_decision_done[farmers] & ~self.continuous_started[farmers]
        )
        if np.any(first_pending):
            selected = farmers[first_pending]
            starts = due_dates[first_pending] - np.timedelta64(self.days_per_year, "D")
            existing = self.first_state_start_date[selected]
            conflict = ~np.isnat(existing) & (existing != starts)
            if np.any(conflict):
                raise RuntimeError(
                    "The first persistent-state bootstrap date changed after scheduling."
                )
            self.first_state_start_date[selected] = starts
            if not _bootstrap and np.any(starts < current_date):
                affected = selected[starts < current_date][:20]
                raise RuntimeError(
                    "A first persistent state was scheduled after its bootstrap history "
                    f"had already begun for farmers {affected.tolist()}."
                )

        self.model.logger.info(
            "Prepared %s ML decision date(s) for %s through %s%s in %.2f s%s.",
            farmers.size,
            str(due_dates.min()),
            str(due_dates.max()),
            " and activated them" if activate else "",
            time.perf_counter() - schedule_timer,
            " (bootstrap)" if _bootstrap else "",
        )

    def _calendar_feasibility_mask(
        self,
        predictor_timestamp: datetime,
        target_years: ArrayInt32,
    ) -> TwoDArrayBool:
        """Return absolute-date calendar feasibility for each farmer and class.

        A cultivated class is feasible only when its planting-day index exists in that
        farmer's nominal target year and the resulting absolute planting date is on or
        after the actual decision timestamp. Fallow classes remain feasible. This avoids
        accepting leap-day index 365 in a non-leap target year and avoids comparing only
        day-of-year values across different calendar years.

        Args:
            predictor_timestamp: Absolute timestamp at which the crop decision is made.
            target_years: Pending target calendar year for every prediction row.

        Returns:
            Boolean array with shape ``(n_rows, n_classes)`` indicating whether each
            calendar class is feasible for the corresponding target year and decision
            date.
        """
        target_years = np.asarray(target_years, dtype=np.int32).reshape(-1)
        feasible = np.zeros((target_years.size, len(self.target_classes)), dtype=bool)
        fallow = self.target_classes[:, 0] < 0
        feasible[:, fallow] = True
        cultivated = ~fallow
        planting_days = self.target_classes[:, 1].astype(np.int64, copy=False)
        decision_day = np.datetime64(predictor_timestamp.date(), "D")

        for target_year in np.unique(target_years):
            rows = target_years == target_year
            year_start = np.datetime64(f"{int(target_year):04d}-01-01", "D")
            year_end = np.datetime64(f"{int(target_year) + 1:04d}-01-01", "D")
            days_in_year = 366 if calendar.isleap(int(target_year)) else 365
            valid_day = (
                cultivated & (planting_days >= 0) & (planting_days < days_in_year)
            )
            if decision_day >= year_end:
                class_mask = np.zeros(len(self.target_classes), dtype=bool)
            else:
                minimum_day = int((decision_day - year_start).astype(np.int64))
                class_mask = valid_day & (planting_days >= minimum_day)
            class_mask[fallow] = True
            feasible[rows] = class_mask
        return feasible

    def _hierarchical_calendar_probabilities(
        self,
        rf_input: TwoDArrayFloat32,
    ) -> TwoDArrayFloat64:
        """Reconstruct joint calendar probabilities from the crop-first RF hierarchy.

        This mirrors ``HierarchicalRandomForestModel.predict_proba`` in training:
        ``P(calendar) = P(crop_1) * P(calendar | crop_1)``.
        """
        features = np.asarray(rf_input, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("Hierarchical RF input must be two-dimensional.")

        local_crop_probabilities = np.asarray(
            self.crop_forest.predict_proba(features), dtype=np.float64
        )
        crop_forest_classes = np.asarray(self.crop_forest.classes_, dtype=np.int64)
        if local_crop_probabilities.shape != (
            features.shape[0],
            crop_forest_classes.size,
        ):
            raise ValueError(
                "Crop-stage RF probability output has an unexpected shape."
            )
        crop_lookup = {
            int(crop): position
            for position, crop in enumerate(self.hierarchical_crop_values)
        }
        crop_probabilities = np.zeros(
            (features.shape[0], self.hierarchical_crop_values.size), dtype=np.float64
        )
        for local_index, crop in enumerate(crop_forest_classes):
            crop_probabilities[:, crop_lookup[int(crop)]] = local_crop_probabilities[
                :, local_index
            ]

        probabilities = np.zeros(
            (features.shape[0], self.n_calendar_classes), dtype=np.float64
        )
        for crop_position, crop_value in enumerate(self.hierarchical_crop_values):
            crop = int(crop_value)
            class_ids = self.conditional_class_ids[crop]
            crop_mass = crop_probabilities[:, crop_position]
            conditional_forest = self.conditional_forests[crop]

            if conditional_forest is None:
                probabilities[:, int(class_ids[0])] = crop_mass
                continue

            local = np.asarray(
                conditional_forest.predict_proba(features), dtype=np.float64
            )
            local_classes = np.asarray(conditional_forest.classes_, dtype=np.int64)
            if local.shape != (features.shape[0], local_classes.size):
                raise ValueError(
                    f"Conditional RF probability output for crop {crop} has an "
                    "unexpected shape."
                )
            for local_index, class_id in enumerate(local_classes):
                probabilities[:, int(class_id)] = crop_mass * local[:, local_index]

        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError(
                "Hierarchical RF probabilities must be finite and non-negative."
            )
        row_sums = probabilities.sum(axis=1)
        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
            raise RuntimeError("Hierarchical RF produced a zero-probability row.")
        probabilities /= row_sums[:, None]
        return probabilities

    def _apply_crop_persistence_gate(
        self,
        probabilities: TwoDArrayFloat64,
        current_crop_ids: ArrayInt32,
        feasible_classes: TwoDArrayBool,
    ) -> TwoDArrayFloat64:
        """Apply the calibrated persistence gate to first-stage crop probability mass."""
        if self.prediction_model != "hierarchical_random_forest_switch_gate":
            return probabilities

        switch_gate = self.bundle.get("switch_gate")
        if not switch_gate or not switch_gate.get("enabled", False):
            raise ValueError(
                "hierarchical_random_forest_switch_gate was requested, but no enabled "
                "crop gate is stored in the deployment bundle."
            )
        if switch_gate.get("gate_definition") not in {None, "first_stage_crop_1"}:
            raise ValueError("The deployment bundle contains an unsupported crop gate.")

        threshold = float(switch_gate["selected_threshold"])
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Switch-gate threshold must be between zero and one.")

        gated = np.asarray(probabilities, dtype=np.float64).copy()
        current_crop_ids = np.asarray(current_crop_ids, dtype=np.int64).reshape(-1)
        if current_crop_ids.shape != (gated.shape[0],):
            raise ValueError(
                "current_crop_ids must contain one crop per probability row."
            )
        if feasible_classes.shape != gated.shape:
            raise ValueError("feasible_classes must match calendar probability shape.")

        # Match training: the gate is available only for a currently cultivated crop.
        # Operationally, do not enforce persistence when no same-crop calendar remains
        # feasible at today's absolute decision date.
        for row_index, crop_id in enumerate(current_crop_ids):
            if crop_id < 0:
                continue
            same_crop = self.class_crop_values == int(crop_id)
            if not np.any(same_crop):
                continue
            same_probability = float(gated[row_index, same_crop].sum())
            switch_probability = float(np.clip(1.0 - same_probability, 0.0, 1.0))
            same_crop_feasible_mass = float(
                gated[row_index, same_crop & feasible_classes[row_index]].sum()
            )
            if switch_probability < threshold and same_crop_feasible_mass > 0.0:
                gated[row_index, ~same_crop] = 0.0
                total = float(gated[row_index].sum())
                if not np.isfinite(total) or total <= 0.0:
                    raise RuntimeError(
                        "The crop persistence gate removed all calendar probability mass."
                    )
                gated[row_index] /= total
        return gated

    def _select_calendar_classes(
        self,
        probabilities: TwoDArrayFloat64,
        predictor_timestamp: datetime,
        target_years: ArrayInt32,
    ) -> tuple[ArrayInt64, ArrayFloat32]:
        """Select feasible calendars using hierarchical argmax or stochastic draws.

        ``argmax`` preserves the trained hierarchy: first choose the main crop from
        marginal crop probability, then choose the highest-probability feasible calendar
        conditional on that crop. ``stochastic`` draws from the feasible post-gate joint
        distribution, which is equivalent to drawing a crop from its marginal and then
        drawing a calendar conditional on that crop.

        Args:
            probabilities: Class probabilities with shape ``(batch, n_classes)``.
            predictor_timestamp: Absolute timestamp at which the crop decision is made.
            target_years: Pending target calendar year for every probability row.

        Returns:
            Tuple containing the selected target-class index for every row and the
            corresponding pre-feasibility-renormalization class probability.

        Raises:
            ValueError: If probability dimensions, target-year dimensions, probability
                values, or row sums are invalid.
            RuntimeError: If the random forest assigns no positive probability to any
                absolute-date feasible calendar for one or more rows.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(
            self.target_classes
        ):
            raise ValueError(
                "Calendar probabilities must have shape (batch, n_classes)."
            )
        target_years = np.asarray(target_years, dtype=np.int32).reshape(-1)
        if target_years.shape != (probabilities.shape[0],):
            raise ValueError("target_years must contain one year per probability row.")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("Calendar probabilities must be finite and non-negative.")
        if np.any(probabilities.sum(axis=1) <= 0.0):
            raise ValueError("Calendar probabilities must have positive row sums.")

        feasible = self._calendar_feasibility_mask(predictor_timestamp, target_years)
        action_probabilities = np.where(feasible, probabilities, 0.0)
        row_sums = action_probabilities.sum(axis=1)
        valid = np.isfinite(row_sums) & (row_sums > 0.0)
        if not np.all(valid):
            bad = np.flatnonzero(~valid)[:20]
            raise RuntimeError(
                "The hierarchical RF assigns no positive probability to an absolute-date "
                f"feasible calendar on {predictor_timestamp:%Y-%m-%d}; batch rows "
                f"{bad.tolist()}, target years {target_years[bad].tolist()}."
            )

        if self.selection_mode == "stochastic":
            normalized = action_probabilities / row_sums[:, None]
            selected = np.empty(probabilities.shape[0], dtype=np.int64)
            for row_index, row in enumerate(normalized):
                selected[row_index] = int(
                    self.selection_rng.choice(len(self.target_classes), p=row)
                )
        else:
            # Crop-first argmax. Marginalize only feasible calendar probability mass so
            # an operationally impossible crop cannot win the first stage.
            crop_probabilities = np.column_stack(
                [
                    action_probabilities[:, self.class_crop_values == crop].sum(axis=1)
                    for crop in self.hierarchical_crop_values
                ]
            )
            selected_crops = self.hierarchical_crop_values[
                np.argmax(crop_probabilities, axis=1)
            ]
            selected = np.empty(probabilities.shape[0], dtype=np.int64)
            for crop in self.hierarchical_crop_values:
                rows = np.flatnonzero(selected_crops == crop)
                if rows.size == 0:
                    continue
                class_ids = np.flatnonzero(self.class_crop_values == crop)
                local = action_probabilities[np.ix_(rows, class_ids)]
                selected[rows] = class_ids[np.argmax(local, axis=1)]

        selected_probability = probabilities[np.arange(selected.size), selected]
        return selected, selected_probability.astype(np.float32)

    def farmers_due_for_earliest_subregion_candidate_planting(
        self,
        timestamp: datetime,
    ) -> ArrayInt64:
        """Select farmers whose prepared historical-subregion decision is due.

        Returns:
            Farmer indices with an active decision whose due date is today or earlier.
        """
        today = np.datetime64(timestamp.date(), "D")
        return np.flatnonzero(
            (self.pending_target_year >= 0)
            & self.awaiting_decision
            & ~np.isnat(self.pending_due_date)
            & (self.pending_due_date <= today)
        ).astype(np.int64)

    def run_due(self) -> ArrayInt64:
        """Evaluate and install every crop decision due on the current model day.

        Before prediction, any incomplete decision-aligned block is padded and committed
        using its real elapsed-day count for memory decay. The carried-state latent,
        decision-relative SPEI latent, and tabular predictors are then combined exactly
        as during training before RF/gate inference and calendar installation.

        Returns:
            Farmer indices for which a crop decision was processed. Returns an empty
            array during spin-up or when no decision is due.

        Raises:
            RuntimeError: If a due decision is overdue, its persistent state is not
                initialized consistently, or decision-aligned block/state invariants are
                violated.
            ValueError: If online RF features/probabilities or calendar-class selection
                violate the trained deployment contract.
        """
        if self.model.in_spinup:
            return np.empty(0, dtype=np.int64)
        due = self.farmers_due_for_earliest_subregion_candidate_planting(
            self.model.current_time
        )
        if due.size == 0:
            return due

        today = np.datetime64(self.model.current_time.date(), "D")
        overdue = self.pending_due_date[due] < today
        if np.any(overdue):
            raise RuntimeError(
                "A crop-choice decision is overdue; evaluating it now would include "
                "daily state after the trained decision timestamp. Affected farmers: "
                f"{due[overdue][:20].tolist()}."
            )
        if np.any(self.pending_due_date[due] != today):
            raise RuntimeError(
                "Due-farmer selection contains a non-today decision date."
            )
        if not np.all(self.continuous_started[due]):
            raise RuntimeError(
                "A due farmer has no initialized persistent neural state."
            )

        first = ~self.first_decision_done[due]
        if np.any(first):
            first_farmers = due[first]
            invalid = self.continuous_interval_days[first_farmers] != self.days_per_year
            if np.any(invalid):
                raise RuntimeError(
                    "First crop decisions must contain exactly 366 persistent daily "
                    "observations; affected farmers: "
                    f"{first_farmers[invalid][:20].tolist()}."
                )
        repeated = self.first_decision_done[due]
        if np.any(repeated):
            repeated_farmers = due[repeated]
            expected_days = (today - self.last_decision_date[repeated_farmers]).astype(
                np.int64
            )
            invalid = self.continuous_interval_days[repeated_farmers] != expected_days
            if np.any(invalid):
                raise RuntimeError(
                    "Carried persistent-state day count differs from the elapsed "
                    "decision interval for farmers "
                    f"{repeated_farmers[invalid][:20].tolist()}."
                )

        # Decision-aligned semantics: commit the final incomplete block permanently,
        # using its actual number of elapsed days for the 200-day memory decay.
        partial_counts = self.continuous_window_days[due].astype(np.int64)
        for elapsed_days in np.unique(partial_counts[partial_counts > 0]):
            selected = due[partial_counts == elapsed_days]
            self._commit_continuous_blocks(selected, int(elapsed_days))
        if np.any(self.continuous_window_days[due] != 0):
            raise RuntimeError(
                "Decision-aligned partial blocks were not fully committed."
            )

        run_due_timer = time.perf_counter()
        encoder_seconds = 0.0
        tabular_seconds = 0.0
        forest_seconds = 0.0
        selection_seconds = 0.0
        predicted_calendars = np.empty((due.size, 3), dtype=np.int32)
        prediction_probabilities = np.empty(due.size, dtype=np.float32)

        for start in range(0, due.size, self.batch_size):
            stop = min(start + self.batch_size, due.size)
            batch_farmers = due[start:stop]
            phase_timer = time.perf_counter()
            decision_sequence = self._decision_history_sequence(
                batch_farmers, self.model.current_time
            )
            with self.torch.inference_mode():
                hidden = self.torch.from_numpy(
                    self.continuous_hidden[:, batch_farmers, :]
                ).to(self.device)
                continuous_latent = self.encoder.continuous_latent_projection(
                    hidden[-1]
                )
                decision_latent = self.encoder.decision_history_latent(
                    self.torch.from_numpy(decision_sequence).to(self.device)
                )
                latent = self.encoder.latent_fusion(
                    self.torch.cat((continuous_latent, decision_latent), dim=1)
                )
                latent_np = latent.cpu().numpy().astype(np.float32, copy=False)
            encoder_seconds += time.perf_counter() - phase_timer

            phase_timer = time.perf_counter()
            tabular = self._tabular_values(batch_farmers, self.model.current_time)
            tabular_seconds += time.perf_counter() - phase_timer

            phase_timer = time.perf_counter()
            rf_input = np.concatenate((latent_np, tabular), axis=1)
            if rf_input.shape[1] != int(self.crop_forest.n_features_in_):
                raise ValueError(
                    "Online hierarchical RF feature width differs from training: "
                    f"{rf_input.shape[1]} versus {self.crop_forest.n_features_in_}."
                )
            probabilities = self._hierarchical_calendar_probabilities(rf_input)
            batch_target_years = self.pending_target_year[batch_farmers]
            feasible_classes = self._calendar_feasibility_mask(
                self.model.current_time, batch_target_years
            )
            probabilities = self._apply_crop_persistence_gate(
                probabilities,
                self.calendar_history[batch_farmers, 0, 0].astype(np.int32),
                feasible_classes,
            )
            forest_seconds += time.perf_counter() - phase_timer

            phase_timer = time.perf_counter()
            selected, selected_probability = self._select_calendar_classes(
                probabilities,
                self.model.current_time,
                self.pending_target_year[batch_farmers],
            )
            selection_seconds += time.perf_counter() - phase_timer
            predicted_calendars[start:stop] = self.target_classes[selected]
            prediction_probabilities[start:stop] = selected_probability
            if self.latent is not None:
                self.latent[batch_farmers] = latent_np

        install_timer = time.perf_counter()
        runtime_calendars = np.full(
            (due.size, *self.farmers.var.crop_calendar.shape[1:]),
            -1,
            dtype=np.int32,
        )
        cultivated = predicted_calendars[:, 0] >= 0
        runtime_calendars[cultivated, 0, :3] = predicted_calendars[cultivated]
        rotation_indices = np.asarray(
            self.farmers.var.current_crop_calendar_rotation_year_index[due],
            dtype=np.int32,
        )
        runtime_calendars[cultivated, 0, 3] = rotation_indices[cultivated]
        self.farmers.var.crop_calendar[due] = runtime_calendars
        if self.predicted_calendar is not None:
            self.predicted_calendar[due] = predicted_calendars
        if self.prediction_probability is not None:
            self.prediction_probability[due] = prediction_probabilities
        if self.calendar_history.shape[1] > 1:
            self.calendar_history[due, 1:] = self.calendar_history[due, :-1]
        self.calendar_history[due, 0] = predicted_calendars
        self.calendar_is_predicted[due] = True
        self.calendar_year[due] = self.pending_target_year[due]

        self.first_decision_done[due] = True
        self.last_decision_date[due] = today
        self.continuous_interval_days[due] = 0
        # Block count is already zero after the partial-block commit. This is the
        # decision boundary from which the next local ten-day cadence begins.
        self.pending_target_year[due] = -1
        self.pending_due_date[due] = np.datetime64("NaT", "D")
        self.awaiting_decision[due] = False
        install_seconds = time.perf_counter() - install_timer

        self.model.logger.info(
            "Installed %s ML-predicted crop calendar(s) on %s.",
            due.size,
            self.model.current_time.date(),
        )
        schedule_timer = time.perf_counter()
        self.schedule(due)
        next_schedule_seconds = time.perf_counter() - schedule_timer
        self.model.logger.info(
            "DecisionModuleML timing on %s for %s decision(s): encoder=%.2f s, "
            "tabular=%.2f s, forest/gate=%.2f s, argmax=%.2f s, install=%.2f s, "
            "next-schedule=%.2f s, total=%.2f s.",
            self.model.current_time.date(),
            due.size,
            encoder_seconds,
            tabular_seconds,
            forest_seconds,
            selection_seconds,
            install_seconds,
            next_schedule_seconds,
            time.perf_counter() - run_due_timer,
        )
        return due
