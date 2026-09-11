"""Run trained crop-calendar decisions during the operational GEB simulation.

The module loads the temporal encoder and random forest only after spin-up, rebuilds
the history needed by the first decision from spin-up reporters, maintains compact
daily and SPEI state, schedules farmer decisions, and installs predicted calendars.
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
    "daily_regional_crop",
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
    crop_ids: np.ndarray,
) -> list[str]:
    """Reconstruct raw tabular feature names for legacy model outputs.

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


class _LegacyDailyStandardizer:
    """Unpickling shim for artifacts created while the trainer was __main__."""


class _LegacyCalendarLabelEncoder:
    """Unpickling shim for artifacts created while the trainer was __main__."""


class DecisionModuleML:
    """Run the trained temporal-encoder plus RF crop-choice pipeline inside GEB.

    GEB retains at most one ten-day daily-input block. Completed blocks are
    immediately compressed by the convolutional encoder and the resulting
    block embeddings are retained until the decision. The GRU then converts
    those embeddings into the yearly latent vector used by the random forest.
    Calendar-shifted SPEI channels use a compact monthly cache rather than a
    multi-year daily farmer array.
    """

    def __init__(self, farmers: CropFarmers, config: dict[str, Any]) -> None:
        """Initialize the operational crop-choice ML runtime.

        Loads and validates the deployment artifacts, rebuilds the temporal encoder,
        allocates compact farmer state, and reconstructs the history needed for the
        first operational decision from completed spin-up reporters.

        Raises:
            RuntimeError: If initialization occurs during spin-up, before restored state,
                or when a requested runtime device is unavailable.
            FileNotFoundError: If required model artifacts or input files are missing.
            ValueError: If configuration, feature contracts, saved arrays, or model
                dimensions are inconsistent.
            TypeError: If deployment or checkpoint objects have unexpected types.
            KeyError: If required model metadata or model entries are missing.
            NotImplementedError: If the trained model enables unsupported online features.
            AttributeError: If the fitted random forest lacks required sklearn metadata.
        """
        self.farmers = farmers
        self.model = farmers.model
        self.config = config
        initialization_timer = time.perf_counter()

        if self.model.in_spinup:
            raise RuntimeError(
                "DecisionModuleML is an operational module and must only be "
                "constructed after spin-up has completed."
            )

        # Heavy ML dependencies are imported only for the operational run. Spin-up
        # therefore has no dependency on a trained model or its deployment stack.
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
                "DecisionModuleML was constructed before the CropFarmers spin-up "
                "state was restored. Missing Bucket entries: "
                f"{missing_restored_state}. Initialization must occur from the "
                "post-restore operational hooks in CropFarmers."
            )

        if farmers.var.crop_calendar_base_array.ndim != 4:
            raise ValueError(
                "Machine-learning crop switching requires the multi-year HRL "
                "crop calendar and agents/farmers/crop_calendar_years."
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
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing temporal crop-choice checkpoint {checkpoint_path}."
            )

        artifact_timer = time.perf_counter()
        if deployment_path.exists():
            self.bundle = joblib.load(deployment_path)
        else:
            # Backward-compatible loading for completed runs made before the
            # dedicated GEB bundle was added. Bare shims are sufficient because
            # only the standardizer moments and plain class vocabulary are used.
            import json
            import sys
            from types import ModuleType

            configuration_path = (
                self.model_directory / "machine_learning_configuration.json"
            )
            sklearn_path = self.model_directory / "sklearn_models.joblib"
            if not configuration_path.exists() or not sklearn_path.exists():
                raise FileNotFoundError(
                    "The crop-choice directory has neither the new deployment "
                    "bundle nor the legacy configuration/sklearn model pair."
                )
            main_module = sys.modules["__main__"]
            setattr(main_module, "DailyStandardizer", _LegacyDailyStandardizer)
            setattr(main_module, "CalendarLabelEncoder", _LegacyCalendarLabelEncoder)
            for module_name in (
                "crop_prediction",
                "machine_learning.main.crop_prediction",
            ):
                if module_name not in sys.modules:
                    shim_module = ModuleType(module_name)
                    shim_module.DailyStandardizer = _LegacyDailyStandardizer
                    shim_module.CalendarLabelEncoder = _LegacyCalendarLabelEncoder
                    sys.modules[module_name] = shim_module
            legacy = joblib.load(sklearn_path)
            with configuration_path.open("r", encoding="utf-8") as file:
                configuration = json.load(file)
            variable_config = configuration["variable_config"]
            raw_tabular_feature_names = configuration.get(
                "raw_tabular_feature_names"
            ) or _ml_raw_tabular_feature_names(
                variable_config,
                farmers.var.crop_data.index.to_numpy(dtype=np.int64),
            )
            daily_standardizer = legacy["daily_standardizer"]
            target_encoder_classes = legacy.get("target_encoder_classes")
            if target_encoder_classes is None:
                target_encoder = legacy.get("target_encoder")
                if target_encoder is None or not hasattr(target_encoder, "classes_"):
                    raise KeyError(
                        "Legacy sklearn_models.joblib contains neither "
                        "'target_encoder_classes' nor a target encoder with classes_."
                    )
                target_encoder_classes = target_encoder.classes_
            self.bundle = {
                "format_version": 1,
                "default_prediction_model": "random_forest",
                "random_forest": legacy["random_forest"],
                "tabular_imputer": legacy["tabular_imputer"],
                "daily_standardizer_mean": daily_standardizer.mean_,
                "daily_standardizer_scale": daily_standardizer.scale_,
                "target_encoder_classes": target_encoder_classes,
                "daily_feature_names": configuration["daily_feature_names"],
                "raw_tabular_feature_names": raw_tabular_feature_names,
                "rf_tabular_feature_names": configuration["rf_tabular_feature_names"],
                "latent_feature_names": configuration["latent_feature_names"],
                "target_columns": configuration["target_columns"],
                "variable_config": variable_config,
                "decision_timing_mode": configuration["decision_timing"]["mode"],
                "days_per_year": configuration["training_settings"]["days_per_year"],
                "daily_sequence_end_offset_days": configuration[
                    "daily_sequence_end_offset_days"
                ],
                "model_random_seed": configuration["model_random_seed"],
                "switch_gate": legacy.get("switch_gate"),
            }
            self.model.logger.warning(
                "Loaded legacy crop-choice outputs from %s. Future training runs "
                "will write geb_crop_choice_bundle.joblib directly.",
                self.model_directory,
            )
        self.model.logger.info(
            "DecisionModuleML timing: deployment artifact load=%.2f s.",
            time.perf_counter() - artifact_timer,
        )
        if not isinstance(self.bundle, Mapping):
            raise TypeError(
                "The crop-choice deployment bundle must be a mapping, found "
                f"{type(self.bundle).__name__}."
            )
        if int(self.bundle.get("format_version", -1)) != 1:
            raise ValueError("Unsupported GEB crop-choice bundle format version.")
        if (
            self.bundle.get("decision_timing_mode")
            != "earliest_subregion_candidate_planting"
        ):
            raise ValueError(
                "The deployed model was not trained with "
                "earliest_subregion_candidate_planting."
            )

        self.prediction_model = (
            str(
                config.get(
                    "prediction_model",
                    self.bundle.get("default_prediction_model", "random_forest"),
                )
            )
            .strip()
            .lower()
        )
        if self.prediction_model not in {
            "random_forest",
            "random_forest_switch_gate",
        }:
            raise ValueError(
                "machine_learning.prediction_model must be 'random_forest' or "
                f"'random_forest_switch_gate', found {self.prediction_model!r}."
            )

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

        enabled_regional_daily = [
            name
            for name, settings in self.variable_config["daily_regional_crop"].items()
            if settings.get("enabled", True)
        ]
        if enabled_regional_daily:
            raise NotImplementedError(
                "The first GEB integration supports the selected baseline with "
                "daily regional-crop channels disabled. Enabled channels: "
                f"{enabled_regional_daily}."
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
                "DecisionModuleML does not yet implement these enabled "
                f"decision-agent features: {unsupported_decision_agent}."
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
                "DecisionModuleML does not yet implement these enabled decision-time "
                f"features: {unsupported_decision_time}."
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
                "DecisionModuleML does not yet implement these enabled regional-crop "
                f"features: {unsupported_regional_crop}."
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
                "The first GEB integration requires the trained 366-day window "
                "ending one day before the decision."
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
                "The deployed model contains daily lag or rolling channels that "
                "cannot satisfy the ten-day raw-retention contract. Retrain with "
                "lag 0 and no rolling transform for non-SPEI variables: "
                f"{non_streamable_daily}."
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
        configured_spei_months = self.bundle.get("online_temporal_state", {}).get(
            "spei_month_cache_months", 0
        )
        self.spei_month_capacity = max(
            int(configured_spei_months),
            12 * self.maximum_spei_years_back + 4,
        )
        supported_daily = set(ML_LAND_SURFACE_DAILY_FEATURES) | set(
            ML_FARMER_DAILY_FEATURES
        )
        unsupported = sorted(set(self.raw_daily_names) - supported_daily)
        if unsupported:
            raise KeyError(
                "The deployed model contains daily variables that GEB does not "
                f"yet capture online: {unsupported}."
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
                "The deployed daily feature-name contract differs from the saved "
                "variable configuration. "
                f"Configured={configured_daily_feature_names}; "
                f"saved={self.daily_feature_names}."
            )

        configured_raw_tabular_feature_names = _ml_raw_tabular_feature_names(
            self.variable_config,
            farmers.var.crop_data.index.to_numpy(dtype=np.int64),
        )
        if configured_raw_tabular_feature_names != self.raw_tabular_feature_names:
            raise ValueError(
                "The deployed raw tabular feature-name contract differs from the "
                "saved variable configuration. "
                f"Configured={configured_raw_tabular_feature_names}; "
                f"saved={self.raw_tabular_feature_names}."
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
                "Daily standardizer moments must be finite and scales must be "
                "strictly positive."
            )

        self.random_forest = self.bundle["random_forest"]
        self.tabular_imputer = self.bundle["tabular_imputer"]

        target_columns = tuple(self.bundle.get("target_columns", ()))
        expected_target_columns = ("crop_1", "crop_1_start_date", "crop_1_duration")
        if target_columns and target_columns != expected_target_columns:
            raise ValueError(
                "The deployed target-column contract is unsupported: "
                f"{target_columns}; expected {expected_target_columns}."
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

        imputer_input_width = getattr(self.tabular_imputer, "n_features_in_", None)
        if imputer_input_width is not None and int(imputer_input_width) != len(
            self.raw_tabular_feature_names
        ):
            raise ValueError(
                "Tabular imputer input width disagrees with the saved raw feature "
                f"names: {imputer_input_width} versus "
                f"{len(self.raw_tabular_feature_names)}."
            )
        if hasattr(self.tabular_imputer, "get_feature_names_out"):
            imputer_feature_names = [
                str(name)
                for name in self.tabular_imputer.get_feature_names_out(
                    self.raw_tabular_feature_names
                )
            ]
            saved_rf_tabular_feature_names = list(
                self.bundle["rf_tabular_feature_names"]
            )
            if imputer_feature_names != saved_rf_tabular_feature_names:
                raise ValueError(
                    "Tabular imputer output names disagree with the saved RF "
                    "feature contract. "
                    f"Imputer={imputer_feature_names}; "
                    f"saved={saved_rf_tabular_feature_names}."
                )

        encoder_timer = time.perf_counter()
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # PyTorch releases predating ``weights_only`` do not accept the keyword.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Temporal crop-choice checkpoint {checkpoint_path} is not a mapping."
            )
        if "state_dict" not in checkpoint:
            raise KeyError(
                f"Temporal crop-choice checkpoint {checkpoint_path} has no state_dict."
            )
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("Temporal checkpoint state_dict is not a mapping.")
        dimensions = dict(checkpoint.get("model_dimensions", {}))

        # Prefer saved architecture metadata. Legacy checkpoints may lack it, so
        # inference is kept as a lazy fallback instead of assuming layer numbers.
        def checkpoint_dimension(
            name: str,
            fallback: Any,
        ) -> int:
            """Read a saved model dimension, inferring it only for legacy checkpoints.

            The fallback is a callable so it is evaluated only when metadata is absent.
            This avoids touching architecture keys that may not exist in newer checkpoints.

            Returns:
                The saved or inferred integer dimension.
            """
            if name in dimensions:
                return int(dimensions[name])
            value = fallback()
            self.model.logger.warning(
                "Temporal checkpoint lacks model_dimensions[%r]; inferred %s.",
                name,
                value,
            )
            return int(value)

        first_conv_key = "window_encoder.network.0.weight"
        second_conv_key = "window_encoder.network.2.weight"
        projection_key = "window_encoder.network.5.weight"
        projection_norm_key = "window_encoder.network.6.weight"
        gru_input_key = "temporal_encoder.weight_ih_l0"
        gru_hidden_key = "temporal_encoder.weight_hh_l0"
        latent_projection_key = "latent_projection.0.weight"
        # Validate the exact encoder pieces used online before allocating the large
        # farmer-state arrays; failures here then stay cheap and easy to diagnose.
        required_checkpoint_keys = (
            first_conv_key,
            second_conv_key,
            projection_key,
            projection_norm_key,
            gru_input_key,
            gru_hidden_key,
            latent_projection_key,
        )
        missing_checkpoint_keys = [
            key for key in required_checkpoint_keys if key not in state_dict
        ]
        if missing_checkpoint_keys:
            raise KeyError(
                "Temporal checkpoint is incompatible with the deployed encoder "
                f"architecture; missing state_dict keys: {missing_checkpoint_keys}."
            )

        expected_input_channels = 2 * len(self.daily_feature_names) + 2
        checkpoint_input_channels = int(state_dict[first_conv_key].shape[1])
        metadata_input_channels = checkpoint_dimension(
            "input_channels",
            lambda: checkpoint_input_channels,
        )
        if metadata_input_channels != checkpoint_input_channels:
            raise ValueError(
                "Temporal checkpoint input-channel metadata disagrees with its "
                f"state_dict: {metadata_input_channels} versus "
                f"{checkpoint_input_channels}."
            )
        if checkpoint_input_channels != expected_input_channels:
            raise ValueError(
                "Temporal checkpoint input width does not match deployed daily "
                f"feature names: checkpoint={checkpoint_input_channels}, "
                f"expected={expected_input_channels}."
            )

        window_days = checkpoint_dimension(
            "window_days",
            lambda: int(
                state_dict[projection_key].shape[1]
                // state_dict[first_conv_key].shape[0]
            ),
        )
        window_stride_days = checkpoint_dimension(
            "window_stride_days",
            lambda: window_days,
        )
        if window_days != 10 or window_stride_days != window_days:
            raise ValueError(
                "The online block encoder requires the trained non-overlapping "
                "10-day windows (window_days=window_stride_days=10)."
            )

        convolution_channels = int(state_dict[first_conv_key].shape[0])
        first_conv_shape = tuple(state_dict[first_conv_key].shape)
        second_conv_shape = tuple(state_dict[second_conv_key].shape)
        if len(first_conv_shape) != 3 or len(second_conv_shape) != 3:
            raise ValueError(
                "Temporal window convolution weights must be three-dimensional."
            )
        if first_conv_shape[2] != 3 or second_conv_shape[2] != 3:
            raise ValueError(
                "The deployed temporal window encoder requires kernel_size=3 for "
                f"both convolutions; found {first_conv_shape} and "
                f"{second_conv_shape}."
            )
        if (
            second_conv_shape[0] != convolution_channels
            or second_conv_shape[1] != convolution_channels
        ):
            raise ValueError(
                "Temporal checkpoint convolution widths are internally inconsistent: "
                f"first={first_conv_shape}, second={second_conv_shape}."
            )

        block_embedding_dim = checkpoint_dimension(
            "block_embedding_dim",
            lambda: int(state_dict[projection_key].shape[0]),
        )
        projection_shape = tuple(state_dict[projection_key].shape)
        if projection_shape != (
            block_embedding_dim,
            convolution_channels * window_days,
        ):
            raise ValueError(
                "Temporal window projection shape disagrees with checkpoint "
                f"dimensions: found {projection_shape}, expected "
                f"{(block_embedding_dim, convolution_channels * window_days)}."
            )
        if tuple(state_dict[projection_norm_key].shape) != (block_embedding_dim,):
            raise ValueError(
                "Temporal window LayerNorm width does not match block embedding "
                f"dimension {block_embedding_dim}."
            )

        gru_hidden_dim = checkpoint_dimension(
            "gru_hidden_dim",
            lambda: int(state_dict[gru_hidden_key].shape[1]),
        )
        inferred_gru_layers = sum(
            key.startswith("temporal_encoder.weight_ih_l")
            and not key.endswith("_reverse")
            for key in state_dict
        )
        gru_num_layers = checkpoint_dimension(
            "gru_num_layers",
            lambda: inferred_gru_layers,
        )
        if gru_num_layers <= 0:
            raise ValueError("Temporal checkpoint contains no GRU layers.")
        if int(state_dict[gru_input_key].shape[1]) != block_embedding_dim:
            raise ValueError(
                "GRU input width does not match the window embedding dimension."
            )
        if int(state_dict[gru_hidden_key].shape[1]) != gru_hidden_dim:
            raise ValueError(
                "GRU hidden-state width disagrees with checkpoint metadata."
            )
        for layer_index in range(gru_num_layers):
            layer_keys = {
                prefix: f"temporal_encoder.{prefix}{layer_index}"
                for prefix in (
                    "weight_ih_l",
                    "weight_hh_l",
                    "bias_ih_l",
                    "bias_hh_l",
                )
            }
            for key in layer_keys.values():
                if key not in state_dict:
                    raise KeyError(
                        "Temporal checkpoint GRU metadata requests "
                        f"{gru_num_layers} layers, but {key!r} is missing."
                    )
            expected_layer_input = (
                block_embedding_dim if layer_index == 0 else gru_hidden_dim
            )
            expected_ih_shape = (3 * gru_hidden_dim, expected_layer_input)
            expected_hh_shape = (3 * gru_hidden_dim, gru_hidden_dim)
            expected_bias_shape = (3 * gru_hidden_dim,)
            actual_ih_shape = tuple(state_dict[layer_keys["weight_ih_l"]].shape)
            actual_hh_shape = tuple(state_dict[layer_keys["weight_hh_l"]].shape)
            actual_bias_ih_shape = tuple(state_dict[layer_keys["bias_ih_l"]].shape)
            actual_bias_hh_shape = tuple(state_dict[layer_keys["bias_hh_l"]].shape)
            if actual_ih_shape != expected_ih_shape:
                raise ValueError(
                    f"GRU layer {layer_index} input-weight shape is "
                    f"{actual_ih_shape}; expected {expected_ih_shape}."
                )
            if actual_hh_shape != expected_hh_shape:
                raise ValueError(
                    f"GRU layer {layer_index} recurrent-weight shape is "
                    f"{actual_hh_shape}; expected {expected_hh_shape}."
                )
            if (
                actual_bias_ih_shape != expected_bias_shape
                or actual_bias_hh_shape != expected_bias_shape
            ):
                raise ValueError(
                    f"GRU layer {layer_index} bias shapes are "
                    f"{actual_bias_ih_shape}/{actual_bias_hh_shape}; expected "
                    f"{expected_bias_shape}."
                )

        latent_dim = checkpoint_dimension(
            "latent_dim",
            lambda: int(state_dict[latent_projection_key].shape[0]),
        )
        latent_projection_shape = tuple(state_dict[latent_projection_key].shape)
        if latent_projection_shape != (latent_dim, gru_hidden_dim):
            raise ValueError(
                "Latent projection shape disagrees with checkpoint dimensions: "
                f"found {latent_projection_shape}, expected "
                f"{(latent_dim, gru_hidden_dim)}."
            )

        if (
            "days_per_year" in dimensions
            and int(dimensions["days_per_year"]) != self.days_per_year
        ):
            raise ValueError(
                "Temporal checkpoint and deployment bundle disagree on days_per_year: "
                f"{dimensions['days_per_year']} versus {self.days_per_year}."
            )

        expected_block_count = int(np.ceil(self.days_per_year / window_days))
        if (
            "n_windows" in dimensions
            and int(dimensions["n_windows"]) != expected_block_count
        ):
            raise ValueError(
                "Temporal checkpoint n_windows disagrees with the online block count: "
                f"{dimensions['n_windows']} versus {expected_block_count}."
            )

        online_state = self.bundle.get("online_temporal_state", {})
        if online_state:
            if (
                int(online_state.get("raw_window_days", window_days)) != window_days
                or int(online_state.get("stored_block_count", expected_block_count))
                != expected_block_count
                or int(online_state.get("block_embedding_dim", block_embedding_dim))
                != block_embedding_dim
            ):
                raise ValueError(
                    "Deployment bundle online_temporal_state disagrees with the "
                    "temporal checkpoint dimensions."
                )

        latent_feature_names = list(self.bundle.get("latent_feature_names", []))
        if latent_feature_names and len(latent_feature_names) != latent_dim:
            raise ValueError(
                "Saved latent feature names disagree with the temporal checkpoint: "
                f"{len(latent_feature_names)} versus {latent_dim}."
            )

        rf_tabular_feature_names = list(self.bundle["rf_tabular_feature_names"])
        expected_rf_width = latent_dim + len(rf_tabular_feature_names)
        rf_input_width = getattr(self.random_forest, "n_features_in_", None)
        if rf_input_width is None:
            raise AttributeError(
                "The deployed random forest has no n_features_in_; it does not look "
                "like a fitted sklearn classifier."
            )
        if int(rf_input_width) != expected_rf_width:
            raise ValueError(
                "Random-forest input width disagrees with the saved latent/tabular "
                f"feature contract: forest={rf_input_width}, expected={expected_rf_width}."
            )

        rf_classes = np.asarray(getattr(self.random_forest, "classes_", []))
        if rf_classes.ndim != 1 or rf_classes.size == 0:
            raise ValueError(
                "The deployed random forest has no one-dimensional classes_."
            )
        if not np.all(np.isfinite(rf_classes)) or not np.all(
            np.isclose(rf_classes, np.rint(rf_classes))
        ):
            raise ValueError("Random-forest class labels must be finite integers.")
        rf_classes = np.rint(rf_classes).astype(np.int64)
        if np.any(rf_classes < 0) or np.any(rf_classes >= len(self.target_classes)):
            raise ValueError(
                "Random-forest class labels fall outside the target-calendar class "
                f"range 0..{len(self.target_classes) - 1}: {rf_classes.tolist()}."
            )

        class TemporalWindowEncoder(nn.Module):
            def __init__(self) -> None:
                """Build the convolutional encoder for one fixed-length daily block."""
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv1d(
                        expected_input_channels,
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
                    nn.Linear(
                        convolution_channels * window_days,
                        block_embedding_dim,
                    ),
                    nn.LayerNorm(block_embedding_dim),
                )

            def forward(self, windows: Any) -> Any:
                """Encode a batch of daily windows into block embeddings.

                Returns:
                    One embedding per input window.
                """
                return self.network(windows.transpose(1, 2))

        class TemporalEncoder(nn.Module):
            def __init__(self) -> None:
                """Build the window encoder, GRU, and final latent projection."""
                super().__init__()
                self.window_encoder = TemporalWindowEncoder()
                self.temporal_encoder = nn.GRU(
                    input_size=block_embedding_dim,
                    hidden_size=gru_hidden_dim,
                    num_layers=gru_num_layers,
                    batch_first=True,
                )
                self.latent_projection = nn.Sequential(
                    nn.Linear(gru_hidden_dim, latent_dim),
                    nn.GELU(),
                    nn.LayerNorm(latent_dim),
                )

            def encode_window(self, window: Any) -> Any:
                """Encode one batch of daily windows.

                Returns:
                    Block-level embeddings for the supplied windows.
                """
                return self.window_encoder(window)

            def encode_blocks(self, blocks: Any) -> Any:
                """Encode a sequence of block embeddings into yearly latent vectors.

                Returns:
                    One latent crop-choice representation per farmer.
                """
                _, hidden = self.temporal_encoder(blocks)
                return self.latent_projection(hidden[-1])

        self.encoder = TemporalEncoder()
        encoder_state = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(
                ("window_encoder.", "temporal_encoder.", "latent_projection.")
            )
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        # The checkpoint also contains decoder/head tensors that are not used by
        # the online RF runtime. Release the loaded checkpoint before allocating
        # the large farmer-level runtime arrays below.
        del encoder_state, state_dict, checkpoint

        requested_device = str(config.get("device", "auto")).strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"DecisionModuleML requested device {requested_device!r}, but CUDA "
                "is not available in this process."
            )
        self.device = torch.device(requested_device)
        self.encoder.to(self.device).eval()
        self.latent_dim = latent_dim
        self.window_days = window_days
        self.input_channels = expected_input_channels
        self.block_embedding_dim = block_embedding_dim
        self.block_count = int(np.ceil(self.days_per_year / self.window_days))
        self.batch_size = max(1, int(config.get("batch_size", 2048)))
        self.top_k = max(1, int(config.get("top_k", 3)))
        self.rng = np.random.default_rng(
            int(config.get("random_seed", self.bundle.get("model_random_seed", 44)))
        )
        self.model.logger.info(
            "DecisionModuleML timing: checkpoint validation and encoder load=%.2f s.",
            time.perf_counter() - encoder_timer,
        )

        # ML deployment state is deliberately created only here, after spin-up.
        # Nothing in this block is part of the CropFarmers spin-up Bucket/state.
        runtime_state_timer = time.perf_counter()
        self.n_farmers = int(farmers.var.n)
        if self.n_farmers < 0:
            raise ValueError("CropFarmers.var.n cannot be negative.")

        base_years = np.asarray(farmers.var.crop_calendar_years, dtype=np.int32)
        if base_years.ndim != 1 or base_years.size == 0:
            raise ValueError(
                "ML crop switching requires a non-empty one-dimensional "
                "crop_calendar_years array."
            )
        if np.unique(base_years).size != base_years.size or np.any(
            np.diff(base_years) <= 0
        ):
            raise ValueError(
                "crop_calendar_years must be unique and strictly increasing; "
                "DecisionModuleML uses searchsorted for first-run source calendars."
            )
        base_calendar_shape = farmers.var.crop_calendar_base_array.shape
        if (
            base_calendar_shape[0] != base_years.size
            or base_calendar_shape[1] < self.n_farmers
            or base_calendar_shape[2] < 1
            or base_calendar_shape[3] < 4
        ):
            raise ValueError(
                "crop_calendar_base_array dimensions disagree with the required "
                "(year, farmer, crop-slot, crop/start/duration/rotation) contract."
            )

        runtime_calendar_shape = farmers.var.crop_calendar.shape
        if (
            len(runtime_calendar_shape) != 3
            or runtime_calendar_shape[0] < self.n_farmers
            or runtime_calendar_shape[1] < 1
            or runtime_calendar_shape[2] < 4
        ):
            raise ValueError(
                "CropFarmers.var.crop_calendar does not have the required "
                "(farmer, crop-slot, crop/start/duration/rotation) shape."
            )

        region_ids = np.asarray(farmers.var.region_id).reshape(-1)
        if region_ids.shape != (self.n_farmers,) or not np.all(np.isfinite(region_ids)):
            raise ValueError(
                "ML crop switching requires one finite region_id per farmer."
            )
        if np.issubdtype(region_ids.dtype, np.integer):
            integral_region_ids = True
        else:
            integral_region_ids = np.all(region_ids == np.floor(region_ids))
        if not integral_region_ids or np.any(region_ids < 0):
            raise ValueError(
                "ML crop switching requires non-negative integral region_id values."
            )

        rotation_years = np.asarray(farmers.var.crop_calendar_rotation_years).reshape(
            -1
        )
        if rotation_years.shape != (self.n_farmers,) or np.any(rotation_years <= 0):
            raise ValueError(
                "ML crop switching requires a positive crop-calendar rotation length "
                "for every farmer."
            )

        self.recent_daily = np.full(
            (self.n_farmers, self.window_days, len(self.raw_daily_names)),
            np.nan,
            dtype=np.float32,
        )
        self.spei_month_values = np.full(
            (self.n_farmers, self.spei_month_capacity),
            np.nan,
            dtype=np.float32,
        )
        self.stream_window = np.zeros(
            (self.n_farmers, self.window_days, self.input_channels),
            dtype=np.float32,
        )
        self.stream_blocks = np.zeros(
            (self.n_farmers, self.block_count, self.block_embedding_dim),
            dtype=np.float32,
        )
        self.stream_block_count = np.zeros(self.n_farmers, dtype=np.int16)
        self.stream_ready = np.zeros(self.n_farmers, dtype=bool)
        self.stream_next_day_index = np.zeros(self.n_farmers, dtype=np.int16)
        self.pending_target_year = np.full(self.n_farmers, -1, dtype=np.int32)
        self.pending_due_date = np.full(
            self.n_farmers, np.datetime64("NaT", "D"), dtype="datetime64[D]"
        )
        # The stream refers to the full per-farmer rolling temporal input sequence for
        # a ML crop decision. Here typically 366 days.
        self.stream_start_date = np.full_like(
            self.pending_due_date, np.datetime64("NaT", "D")
        )
        self.awaiting_decision = np.zeros(self.n_farmers, dtype=bool)
        self.calendar_is_predicted = np.zeros(self.n_farmers, dtype=bool)
        # Farmers whose final prescribed calendar already completed during spin-up
        # have no harvest event left to activate the prepared ML decision in the
        # operational run. This flag is reconstructed from the sparse spin-up
        # decision reporter below.
        self.completed_prescribed_before_run = np.zeros(self.n_farmers, dtype=bool)
        self.yield_recorded_year = np.full(self.n_farmers, -1, dtype=np.int32)

        calendar_history_depth = 1
        for name in ("crop_1", "crop_1_start_date", "crop_1_duration"):
            settings = self.variable_config["decision_agent"].get(name, {})
            if not settings.get("enabled", True):
                continue
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
            (self.n_farmers, yield_history_depth),
            np.nan,
            dtype=np.float32,
        )

        # Optional runtime diagnostics. These are not spin-up variables and are not
        # required by GEB's saved state.
        # These arrays are diagnostics only and can be sizeable for large Europe
        # models (especially n_farmers x latent_dim). Keep them off by default.
        self.store_runtime_diagnostics = bool(
            config.get("store_runtime_diagnostics", False)
        )
        if self.store_runtime_diagnostics:
            self.latent: np.ndarray | None = np.full(
                (self.n_farmers, self.latent_dim), np.nan, dtype=np.float32
            )
            self.predicted_calendar: np.ndarray | None = np.full(
                (self.n_farmers, 3), -1, dtype=np.int32
            )
            self.prediction_probability: np.ndarray | None = np.full(
                self.n_farmers, np.nan, dtype=np.float32
            )
        else:
            self.latent = None
            self.predicted_calendar = None
            self.prediction_probability = None

        self.recent_daily_dates = np.full(
            self.window_days,
            np.datetime64("NaT", "D"),
            dtype="datetime64[D]",
        )
        self.recent_daily_position = 0
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
        if np.any(active_indices < 0) or np.any(
            active_indices >= len(farmers.var.crop_calendar_years)
        ):
            raise ValueError(
                "ML crop switching requires a valid active prescribed crop-calendar "
                "year for every farmer at the end of spin-up."
            )
        self.calendar_year = np.asarray(
            farmers.var.crop_calendar_years[active_indices], dtype=np.int32
        )
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
            "DecisionModuleML timing: farmer runtime-state allocation/history setup="
            "%.2f s for %s farmers.",
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

        # The first operational decisions are prepared from the final prescribed
        # crop year. Their 366-day neural histories are then reconstructed from
        # the already completed spin-up reporters; no trained model is needed
        # while spin-up itself is running.
        run_start = pd.Timestamp(self.model.config["general"]["start_time"])
        run_start_year = int(run_start.year)
        initial_schedule_timer = time.perf_counter()
        self.schedule(
            np.arange(self.n_farmers, dtype=np.int64),
            target_years=np.full(self.n_farmers, run_start_year, dtype=np.int32),
            _bootstrap=True,
        )
        self.model.logger.info(
            "DecisionModuleML timing: initial decision-stream scheduling=%.2f s.",
            time.perf_counter() - initial_schedule_timer,
        )
        self._bootstrap_from_spinup_reports(run_start)

        # A farmer that completed the final prescribed calendar during spin-up
        # will not harvest it again after `geb run` starts. Activate its already
        # prepared decision now. Farmers whose final calendar is still active
        # (including late crops and fallow years) remain inactive until the normal
        # CropFarmers advancement event occurs during the operational run.
        completed = np.flatnonzero(self.completed_prescribed_before_run).astype(
            np.int64
        )
        if completed.size:
            self.schedule(completed, activate=True)
            self.model.logger.info(
                "Activated %s initial ML decision stream(s) whose final prescribed "
                "calendar completed during spin-up.",
                completed.size,
            )

        self.model.logger.info(
            "Loaded online crop-choice model from %s: %s daily channels, %s "
            "raw tabular features, %s latent features and %s calendar classes; "
            "prediction model=%s.",
            self.model_directory,
            len(self.daily_feature_names),
            len(self.raw_tabular_feature_names),
            self.latent_dim,
            len(self.target_classes),
            self.prediction_model,
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
            chunks={},
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

    @staticmethod
    def _month_ordinal(timestamp: pd.Timestamp) -> int:
        """Convert a timestamp to an absolute zero-based month number.

        Returns:
            Integer month ordinal used by the circular SPEI cache.
        """
        return int(timestamp.year * 12 + timestamp.month - 1)

    def _bootstrap_yield_history(self, run_start: pd.Timestamp) -> None:
        """Reconstruct recent decision-event yield lags from spin-up reporters.

        The reporters are scanned backwards in bounded chunks so only the newest yield
        observations required by the trained lag configuration are retained.

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
        finally:
            active_dataset.close()
            yield_dataset.close()

        self.model.logger.info(
            "DecisionModuleML timing: spin-up yield-history bootstrap=%.2f s.",
            time.perf_counter() - yield_bootstrap_timer,
        )

    def _bootstrap_daily_streams(self, run_start: pd.Timestamp) -> None:
        """Replay pre-run days needed by the first operational encoder windows.

        Daily reporters are read one date at a time. Calendar-shifted SPEI history is
        rebuilt into the compact monthly cache while those dates are replayed.

        Raises:
            ValueError: If a required daily reporter does not cover the bootstrap period.
        """
        valid_starts = self.stream_start_date[~np.isnat(self.stream_start_date)]
        if valid_starts.size == 0:
            return

        first_day = pd.Timestamp(valid_starts.min())
        last_day = run_start - pd.Timedelta(days=1)
        if first_day > last_day:
            return

        daily_bootstrap_timer = time.perf_counter()
        bootstrap_day_count = (
            int((last_day.normalize() - first_day.normalize()).days) + 1
        )
        self.model.logger.info(
            "Reconstructing ML daily history from spin-up reporters: %s days, "
            "%s daily inputs, %s farmers.",
            bootstrap_day_count,
            len(self.raw_daily_names),
            self.n_farmers,
        )

        opened: dict[str, tuple[Any, Any, pd.DatetimeIndex]] = {}
        spei_history_dataset: Any | None = None
        try:
            for name in self.raw_daily_names:
                dataset, data, _ = self._open_spinup_report(
                    name,
                    start_time=first_day,
                    end_time=last_day,
                )
                dates = pd.DatetimeIndex(np.asarray(data["time"].values))
                if dates.empty or dates[-1].normalize() < last_day.normalize():
                    dataset.close()
                    raise ValueError(
                        f"Spin-up reporter {name!r} does not cover the final "
                        f"pre-run day {last_day.date()}."
                    )
                opened[name] = (dataset, data, dates)

            # SPEI lags are calendar-shifted by whole years. Reconstruct the
            # compact monthly cache chronologically while the daily stream is
            # replayed, rather than filling it with the final months up front.
            spei_position_by_ordinal: dict[int, int] = {}
            spei_history_data: Any | None = None
            if "crop_decision_spei" in self.raw_daily_index:
                oldest_spei = first_day - pd.DateOffset(
                    years=self.maximum_spei_years_back
                )
                oldest_spei -= pd.Timedelta(days=40)
                spei_history_dataset, spei_history_data, _ = self._open_spinup_report(
                    "crop_decision_spei",
                    start_time=oldest_spei,
                    end_time=last_day,
                )
                spei_dates = pd.DatetimeIndex(
                    np.asarray(spei_history_data["time"].values)
                )
                for position, date in enumerate(spei_dates):
                    ordinal = self._month_ordinal(date)
                    if int(date.day) > 15:
                        ordinal += 1
                    spei_position_by_ordinal.setdefault(ordinal, position)

            def advance_spei_cache(timestamp: pd.Timestamp) -> None:
                """Advance the compact SPEI cache to the current replay date."""
                if spei_history_data is None:
                    return
                desired = self._month_ordinal(timestamp)
                if int(timestamp.day) > 15:
                    desired += 1

                # On the first replay day, seed only the months that can still be
                # addressed by the circular cache. Afterwards add newly exposed
                # nearest-month values one month at a time.
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

                # If some old months were unavailable, still let the cache's
                # validity window advance with the model date. Missing entries
                # remain NaN and are represented by the encoder observation mask.
                if (
                    self.latest_spei_month_ordinal is None
                    or self.latest_spei_month_ordinal < desired
                ):
                    self.latest_spei_month_ordinal = desired

            full_dates = pd.date_range(first_day, last_day, freq="D")
            missing_template = np.full(self.n_farmers, np.nan, dtype=np.float32)

            # Read only one farmer vector per variable at a time. This keeps
            # bootstrap memory bounded even for large Europe models.
            for timestamp in full_dates:
                advance_spei_cache(timestamp)
                values_by_name: dict[str, np.ndarray] = {}
                timestamp64 = np.datetime64(timestamp.date(), "ns")
                for name, (_, data, dates) in opened.items():
                    position = int(dates.searchsorted(timestamp))
                    if (
                        position < len(dates)
                        and dates[position].to_datetime64() == timestamp64
                    ):
                        values = np.asarray(
                            data.isel(time=position).values,
                            dtype=np.float32,
                        ).reshape(-1)
                    else:
                        values = missing_template
                    values_by_name[name] = values
                self._capture_daily_values(
                    timestamp.to_pydatetime(),
                    values_by_name,
                )
        finally:
            for dataset, _, _ in opened.values():
                dataset.close()
            if spei_history_dataset is not None:
                spei_history_dataset.close()

        self.model.logger.info(
            "DecisionModuleML timing: spin-up daily-stream bootstrap=%.2f s (%s days).",
            time.perf_counter() - daily_bootstrap_timer,
            bootstrap_day_count,
        )

    def _bootstrap_from_spinup_reports(self, run_start: pd.Timestamp) -> None:
        """Reconstruct all history required before the first operational decision.

        Raises:
            FileNotFoundError: If the completed spin-up report directory is missing.
        """
        bootstrap_timer = time.perf_counter()
        report_directory = self._spinup_report_directory()
        if not report_directory.exists():
            raise FileNotFoundError(
                "DecisionModuleML is enabled, but the completed spin-up report "
                f"directory does not exist: {report_directory}."
            )

        self._bootstrap_yield_history(run_start)
        self._bootstrap_daily_streams(run_start)

        self.model.logger.info(
            "Initialized DecisionModuleML from completed spin-up reporters in %s.",
            report_directory,
        )
        self.model.logger.info(
            "DecisionModuleML timing: total spin-up reporter bootstrap=%.2f s.",
            time.perf_counter() - bootstrap_timer,
        )

    def _build_regional_farmer_values(self) -> dict[str, np.ndarray]:
        """Compute configured regional farmer statistics and map them to farmers.

        Returns:
            Mapping from deployed regional-statistic feature name to farmer values.

        Raises:
            ValueError: If a regional-farmer input is not aligned with the farmer array.
        """
        timer = time.perf_counter()
        output: dict[str, np.ndarray] = {}
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

    def _sequence_days(
        self,
        farmers: np.ndarray,
        dates: np.ndarray,
        raw_values: np.ndarray,
    ) -> np.ndarray:
        """Build standardized temporal-encoder rows for selected farmer-days.

        Non-SPEI variables use the retained raw day. SPEI lag and rolling features are
        reconstructed from the compact monthly cache using the same decision-relative
        calendar shifts as training, then observation masks and seasonal channels are
        added.

        Returns:
            Float32 encoder rows with values, observation masks, and seasonal features.

        Raises:
            AssertionError: If a non-SPEI daily feature uses an unsupported transform.
            ValueError: If input shapes or the generated feature order differ from training.
        """
        farmers = np.asarray(farmers, dtype=np.int64)
        dates = np.asarray(dates, dtype="datetime64[D]")
        raw_values = np.asarray(raw_values, dtype=np.float32)
        if dates.shape != farmers.shape or raw_values.shape != (
            farmers.size,
            len(self.raw_daily_names),
        ):
            raise ValueError("Invalid daily input shapes for the temporal encoder.")

        day_indices = (dates - self.stream_start_date[farmers]).astype(np.int64)
        # SPEI lags are relative to each farmer's decision date, not the current
        # simulation year. Rebuild those shifted dates from the monthly cache.
        spei_by_offset: dict[int, np.ndarray] = {}
        if "crop_decision_spei" in self.raw_daily_index:
            spei_by_offset[0] = raw_values[
                :, self.raw_daily_index["crop_decision_spei"]
            ]
            for years_back in range(1, self.maximum_spei_years_back + 1):
                shifted_due = pd.DatetimeIndex(
                    self.pending_due_date[farmers].astype("datetime64[ns]")
                ) - pd.DateOffset(years=years_back)
                requested_dates = (
                    shifted_due.to_numpy(dtype="datetime64[D]")
                    - np.timedelta64(self.days_per_year, "D")
                    + day_indices.astype("timedelta64[D]")
                )
                requested = pd.DatetimeIndex(requested_dates.astype("datetime64[ns]"))
                month_ordinals = (
                    requested.year.to_numpy(dtype=np.int64) * 12
                    + requested.month.to_numpy(dtype=np.int64)
                    - 1
                    + (requested.day.to_numpy(dtype=np.int64) > 15)
                )
                transformed = np.full(farmers.size, np.nan, dtype=np.float32)
                if self.latest_spei_month_ordinal is not None:
                    valid = (month_ordinals <= self.latest_spei_month_ordinal) & (
                        month_ordinals
                        > self.latest_spei_month_ordinal - self.spei_month_capacity
                    )
                    if np.any(valid):
                        transformed[valid] = self.spei_month_values[
                            farmers[valid],
                            month_ordinals[valid] % self.spei_month_capacity,
                        ]
                spei_by_offset[years_back] = transformed

        columns: list[np.ndarray] = []
        names: list[str] = []
        for name, settings in self.variable_config["daily_agent"].items():
            if not settings.get("enabled", True):
                continue
            for kind, value, feature_name in _ml_transform_specs(name, settings):
                if name == "crop_decision_spei":
                    if kind == "lag":
                        transformed = spei_by_offset[value]
                    else:
                        stacked = np.stack(
                            [spei_by_offset[offset] for offset in range(value)]
                        )
                        count = np.isfinite(stacked).sum(axis=0)
                        transformed = np.divide(
                            np.nansum(stacked, axis=0),
                            count,
                            out=np.full(farmers.size, np.nan, dtype=np.float32),
                            where=count > 0,
                        )
                else:
                    if kind != "lag" or value != 0:
                        raise AssertionError(
                            "Non-SPEI daily deployment features must use lag 0."
                        )
                    transformed = raw_values[:, self.raw_daily_index[name]]
                columns.append(np.asarray(transformed, dtype=np.float32))
                names.append(feature_name)
        if names != self.daily_feature_names:
            raise ValueError(
                "Online daily feature order differs from the training bundle. "
                f"Online={names}; trained={self.daily_feature_names}."
            )

        values = np.column_stack(columns).astype(np.float32, copy=False)
        observed = np.isfinite(values)
        values -= self.daily_mean[None, :]
        values /= self.daily_scale[None, :]
        values[~observed] = 0.0

        date_index = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
        # Use leap year 2000 as a fixed seasonal calendar so Feb 29 has its own
        # position and seasonal channels are comparable across simulation years.
        canonical_slots = np.asarray(
            [
                datetime(2000, int(month), int(day)).timetuple().tm_yday - 1
                for month, day in zip(date_index.month, date_index.day)
            ],
            dtype=np.float32,
        )
        angle = 2.0 * np.pi * canonical_slots / self.days_per_year

        # Build the final encoder row directly. np.concatenate previously created
        # an extra float32 copy of the complete observation-mask matrix.
        feature_count = len(self.daily_feature_names)
        sequence = np.empty((farmers.size, self.input_channels), dtype=np.float32)
        sequence[:, :feature_count] = values
        sequence[:, feature_count : 2 * feature_count] = observed
        sequence[:, -2] = np.sin(angle)
        sequence[:, -1] = np.cos(angle)
        return sequence

    def capture_daily(
        self,
        timestamp: datetime,
        land_surface_values: dict[str, np.ndarray],
    ) -> None:
        """Capture one live model day for the temporal crop-choice streams.

        Combines land-surface inputs with farmer-owned daily variables and forwards the
        complete farmer-aligned day to the shared streaming encoder logic.

        Raises:
            RuntimeError: If the farmer population changes during the operational run.
        """
        if int(self.farmers.var.n) != self.n_farmers:
            raise RuntimeError(
                "DecisionModuleML currently requires a fixed farmer population "
                "during an operational run."
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
        self._capture_daily_values(
            timestamp,
            {**land_surface_values, **farmer_values},
        )

    def _capture_daily_values(
        self,
        timestamp: datetime,
        values_by_name: dict[str, np.ndarray],
    ) -> None:
        """Validate, buffer, and encode one complete farmer-aligned raw day.

        The same path is used for live simulation days and spin-up replay. Completed
        10-day blocks are encoded immediately so raw history does not accumulate.

        Raises:
            KeyError: If a required daily input is missing.
            RuntimeError: If days are non-consecutive or stream blocks are out of order.
            ValueError: If a daily input is not aligned with the farmer population.
        """
        current_date = np.datetime64(timestamp.date(), "D")
        if self.last_daily_date is not None:
            expected = self.last_daily_date + np.timedelta64(1, "D")
            if current_date == self.last_daily_date:
                return
            if current_date != expected:
                raise RuntimeError(
                    "Online crop-choice inputs must be captured on consecutive "
                    f"days; received {current_date} after {self.last_daily_date}."
                )

        missing = sorted(set(self.raw_daily_names) - set(values_by_name))
        if missing:
            raise KeyError(f"Missing online daily crop-choice values: {missing}.")

        current_values: dict[str, np.ndarray] = {}
        for name in self.raw_daily_names:
            values = np.asarray(values_by_name[name], dtype=np.float32).reshape(-1)
            if values.shape != (self.n_farmers,):
                raise ValueError(
                    f"Daily crop-choice variable {name!r} has shape "
                    f"{values.shape}; expected {(self.n_farmers,)}."
                )
            settings = self.variable_config["daily_agent"][name]
            history_scope = settings.get("history_scope", "dynamic")
            if history_scope == "dynamic" and current_date < self.dynamic_history_start:
                values = np.full_like(values, np.nan)
            current_values[name] = values

        raw_values = np.column_stack(
            [current_values[name] for name in self.raw_daily_names]
        ).astype(np.float32, copy=False)

        recent_position = self.recent_daily_position
        self.recent_daily[:, recent_position, :] = raw_values
        self.recent_daily_dates[recent_position] = current_date
        self.recent_daily_position = (recent_position + 1) % self.window_days

        active = np.flatnonzero(
            (self.pending_target_year >= 0)
            & ~np.isnat(self.stream_start_date)
            & (current_date >= self.stream_start_date)
            & (current_date < self.pending_due_date)
            & (~self.stream_ready)
        ).astype(np.int64)

        if active.size == 0:
            self.last_daily_date = current_date
            return

        day_indices = (current_date - self.stream_start_date[active]).astype(np.int64)
        if np.any(day_indices != self.stream_next_day_index[active]):
            invalid = active[day_indices != self.stream_next_day_index[active]][:20]
            raise RuntimeError(
                "A crop-choice encoder stream was scheduled after its exact "
                "366-day input window had started without the required historical "
                f"backfill. Affected farmers: {invalid.tolist()}."
            )

        sequence_day = self._sequence_days(
            active,
            np.full(active.size, current_date, dtype="datetime64[D]"),
            raw_values[active],
        )
        positions = day_indices % self.window_days
        self.stream_window[active, positions, :] = sequence_day
        self.stream_next_day_index[active] += 1

        complete_mask = ((day_indices + 1) % self.window_days == 0) | (
            day_indices == self.days_per_year - 1
        )
        complete = active[complete_mask]
        complete_day_indices = day_indices[complete_mask]
        for start in range(0, complete.size, self.batch_size):
            batch = complete[start : start + self.batch_size]
            batch_days = complete_day_indices[start : start + self.batch_size]
            block_indices = batch_days // self.window_days
            if np.any(self.stream_block_count[batch] != block_indices):
                raise RuntimeError("Crop-choice blocks were not encoded in order.")
            with self.torch.inference_mode():
                embeddings = self.encoder.encode_window(
                    self.torch.from_numpy(self.stream_window[batch]).to(self.device)
                )
            self.stream_blocks[batch, block_indices, :] = (
                embeddings.cpu().numpy().astype(np.float32, copy=False)
            )
            self.stream_block_count[batch] += 1
            final = batch_days == self.days_per_year - 1
            self.stream_ready[batch[final]] = True
        self.stream_window[complete] = 0.0

        self.last_daily_date = current_date

    @staticmethod
    def _asof_index(date_index: DateIndex, timestamp: datetime) -> int | None:
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
        farmers: np.ndarray,
        timestamp: datetime,
    ) -> np.ndarray:
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
        farmers: np.ndarray,
        timestamp: datetime,
    ) -> np.ndarray:
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

    def _static_value(self, name: str, farmers: np.ndarray) -> np.ndarray:
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
        farmers: np.ndarray,
        predictor_timestamp: datetime,
    ) -> np.ndarray:
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
        feature_values: dict[str, np.ndarray] = {}
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
        farmers: np.ndarray,
        values: np.ndarray,
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
        farmers: np.ndarray,
        active_indices: np.ndarray,
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

    def _candidate_start_days(self, region_id: int, target_year: int) -> np.ndarray:
        """Collect historical main-crop planting days for a region and target year.

        Calendar years are processed one at a time to avoid materializing a large
        ``years x farmers`` calendar subset.

        Returns:
            Sorted unique planting-day indices observed strictly before the target year.
        """
        years = np.asarray(self.farmers.var.crop_calendar_years)
        region_farmers = np.flatnonzero(
            np.asarray(self.farmers.var.region_id) == region_id
        )
        year_indices = np.flatnonzero(years < target_year)
        if region_farmers.size == 0 or year_indices.size == 0:
            return np.empty(0, dtype=np.int32)
        # Only the unique planting days are needed here. Process one calendar
        # year at a time so large subregions do not create a temporary
        # (n_years x n_region_farmers x 3) calendar array.
        start_days: list[np.ndarray] = []
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
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(start_days)).astype(np.int32, copy=False)

    def schedule(
        self,
        farmers: np.ndarray,
        *,
        target_years: np.ndarray | None = None,
        activate: bool = False,
        _bootstrap: bool = False,
    ) -> None:
        """Prepare and optionally activate ML decision streams for selected farmers.

        For each farmer, the decision date is the earliest historically observed
        subregion planting date that is still feasible after the previous harvest. The
        preceding 366-day encoder window is then prepared or backfilled as needed.

        Raises:
            IndexError: If a farmer index is outside the runtime population.
            RuntimeError: If activation conflicts with the prepared target year or if
                required recent history is no longer available for backfill.
            ValueError: If farmer or target-year arrays have invalid shapes or duplicate
                farmer indices.
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

        # Streams are often prepared before the preceding calendar finishes. When
        # harvest occurs, activation should reuse that prepared stream, not reset it.
        already_scheduled = self.pending_target_year[farmers] >= 0
        if activate and np.any(already_scheduled):
            activated = farmers[already_scheduled]
            expected_years = self.calendar_year[activated] + 1
            if np.any(self.pending_target_year[activated] != expected_years):
                raise RuntimeError(
                    "The prepared ML decision does not follow the completed "
                    "crop-calendar year."
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
                "(activation only; no new streams).",
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
                    "Preparing the first run-time ML decision requires its "
                    "preceding HRL calendar year. Missing years: "
                    f"{missing_years.tolist()}."
                )
            source_calendars = self.farmers.var.crop_calendar_base_array[
                source_indices, farmers, :, :
            ]
            active_indices = np.asarray(
                self.farmers.var.crop_calendar_active_year_index[farmers],
                dtype=np.int32,
            )
            rotation_years = np.asarray(
                self.farmers.var.crop_calendar_rotation_years[farmers],
                dtype=np.int32,
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
            & (source_calendars[:, :, 3] == source_rotation_indices[:, np.newaxis])
        )
        harvest_offsets = np.where(
            valid_source_crops,
            source_calendars[:, :, 1] + source_calendars[:, :, 2],
            -1,
        ).max(axis=1)
        has_harvest_constraint = harvest_offsets >= 0
        # Avoid a Python list of one formatted date string/object per farmer.
        # NumPy datetime64[Y] uses 1970 as its integer epoch.
        source_year_starts = (source_years - 1970).astype("datetime64[Y]")
        harvest_dates = source_year_starts + np.maximum(harvest_offsets, 0).astype(
            "timedelta64[D]"
        )

        region_ids = np.asarray(self.farmers.var.region_id[farmers], dtype=np.int64)
        due_dates = np.full(farmers.size, np.datetime64("NaT", "D"))
        for pair in np.unique(np.column_stack((region_ids, target_years)), axis=0):
            region_id, target_year = (int(pair[0]), int(pair[1]))
            positions = np.flatnonzero(
                (region_ids == region_id) & (target_years == target_year)
            )
            # Training used the earliest historical subregion planting date that
            # remains feasible after harvest; reproduce that rule exactly online.
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
                    self.model.logger.warning(
                        "No feasible historical candidate planting date for "
                        "farmer %s (region %s, target %s); using %s.",
                        int(farmers[position]),
                        region_id,
                        target_year,
                        due_dates[position],
                    )

        self.pending_target_year[farmers] = target_years
        self.pending_due_date[farmers] = due_dates
        self.stream_start_date[farmers] = due_dates - np.timedelta64(
            self.days_per_year, "D"
        )
        self.stream_window[farmers] = 0.0
        self.stream_blocks[farmers] = 0.0
        self.stream_block_count[farmers] = 0
        self.stream_ready[farmers] = False
        self.stream_next_day_index[farmers] = 0
        self.awaiting_decision[farmers] = activate
        if activate:
            self.farmers.var.crop_calendar[farmers] = -1

        if _bootstrap:
            self.model.logger.info(
                "Prepared %s initial ML crop decision stream(s) for spin-up "
                "reporter bootstrap in %.2f s.",
                farmers.size,
                time.perf_counter() - schedule_timer,
            )
            return

        # A new annual stream can start up to ten days before the preceding
        # decision. Backfill those few days from the one retained recent block.
        elapsed = (current_date - self.stream_start_date[farmers]).astype(np.int64)
        late = elapsed > 0
        if np.any(elapsed[late] > self.window_days):
            affected = farmers[elapsed > self.window_days][:20]
            raise RuntimeError(
                "A crop-choice stream was scheduled more than ten days after "
                "its decision-relative input window began; raw daily history is "
                f"intentionally not retained. Affected farmers: {affected.tolist()}."
            )
        if np.any(late):
            for offset in range(int(elapsed[late].max())):
                positions = np.flatnonzero(elapsed > offset)
                requested_dates = self.stream_start_date[
                    farmers[positions]
                ] + np.timedelta64(offset, "D")
                for requested_date in np.unique(requested_dates):
                    recent_slots = np.flatnonzero(
                        self.recent_daily_dates == requested_date
                    )
                    if recent_slots.size != 1:
                        raise RuntimeError(
                            "The ten-day recent block cannot backfill required "
                            f"crop-choice date {requested_date}."
                        )
                    selected_positions = positions[requested_dates == requested_date]
                    selected_farmers = farmers[selected_positions]
                    sequence_days = self._sequence_days(
                        selected_farmers,
                        np.full(
                            selected_farmers.size,
                            requested_date,
                            dtype="datetime64[D]",
                        ),
                        np.asarray(
                            self.recent_daily[
                                selected_farmers,
                                recent_slots[0],
                                :,
                            ],
                            dtype=np.float32,
                        ),
                    )
                    self.stream_window[
                        selected_farmers, offset % self.window_days, :
                    ] = sequence_days
                    self.stream_next_day_index[selected_farmers] += 1

            completed_first_block = farmers[elapsed == self.window_days]
            if completed_first_block.size:
                for start in range(0, completed_first_block.size, self.batch_size):
                    batch = completed_first_block[start : start + self.batch_size]
                    with self.torch.inference_mode():
                        embeddings = self.encoder.encode_window(
                            self.torch.from_numpy(self.stream_window[batch]).to(
                                self.device
                            )
                        )
                    self.stream_blocks[batch, 0, :] = (
                        embeddings.cpu().numpy().astype(np.float32, copy=False)
                    )
                    self.stream_block_count[batch] = 1
                self.stream_window[completed_first_block] = 0.0

        self.model.logger.info(
            "Prepared %s ML crop decision stream(s) for %s through %s%s in %.2f s.",
            farmers.size,
            str(due_dates.min()),
            str(due_dates.max()),
            " and activated them" if activate else "",
            time.perf_counter() - schedule_timer,
        )

    def _sample_calendar_classes(
        self,
        probabilities: np.ndarray,
        predictor_timestamp: datetime,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply planting feasibility and sample calendar classes from RF probabilities.

        Only still-feasible classes retain probability mass. Sampling is restricted to
        the configured top-k positive classes and renormalized within that set.

        Returns:
            Selected class indices and their original RF probabilities.

        Raises:
            RuntimeError: If no trained class has positive feasible probability mass.
            ValueError: If probability shapes or values are invalid or cannot normalize.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(
            self.target_classes
        ):
            raise ValueError(
                "Calendar probabilities must have shape "
                f"(batch, {len(self.target_classes)})."
            )
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("Calendar probabilities must be finite and non-negative.")

        feasible = (self.target_classes[:, 0] < 0) | (
            self.target_classes[:, 1] >= predictor_timestamp.timetuple().tm_yday - 1
        )
        feasible_indices = np.flatnonzero(feasible)
        if feasible_indices.size == 0:
            raise RuntimeError(
                "No trained crop-calendar class is feasible on the decision day."
            )

        selected_classes = np.empty(probabilities.shape[0], dtype=np.int64)
        selected_probability = np.empty(probabilities.shape[0], dtype=np.float32)
        for row_index, row in enumerate(probabilities):
            if row.sum() <= 0.0:
                raise ValueError(
                    f"RF probabilities sum to zero for batch row {row_index}."
                )

            action_probabilities = row.copy()
            action_probabilities[~feasible] = 0.0
            positive = np.flatnonzero(action_probabilities > 0.0)
            if positive.size == 0:
                raise RuntimeError(
                    "The random forest assigns no positive probability to any "
                    "calendar that is still feasible on "
                    f"{predictor_timestamp:%Y-%m-%d} (batch row {row_index})."
                )
            else:
                top_count = min(self.top_k, positive.size)
                if top_count == positive.size:
                    top = positive
                else:
                    positive_probabilities = action_probabilities[positive]
                    top_local = np.argpartition(positive_probabilities, -top_count)[
                        -top_count:
                    ]
                    top = positive[top_local]
                top_probabilities = action_probabilities[top]
                top_probability_sum = float(top_probabilities.sum())
                if not np.isfinite(top_probability_sum) or top_probability_sum <= 0.0:
                    raise ValueError(
                        "Top-k calendar probabilities cannot be normalized."
                    )
                top_probabilities = top_probabilities / top_probability_sum
                selected = int(self.rng.choice(top, p=top_probabilities))
            selected_classes[row_index] = selected
            selected_probability[row_index] = row[selected]
        return selected_classes, selected_probability

    def farmers_due_for_earliest_subregion_candidate_planting(
        self,
        timestamp: datetime,
    ) -> np.ndarray:
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

    def run_due(self) -> np.ndarray:
        """Predict and install calendars for all farmers whose ML decision is due.

        The temporal blocks are compressed to latent vectors, combined with imputed
        RF tabular inputs, optionally filtered by the switch gate, sampled under
        planting-date feasibility, and written back to the runtime crop calendar.

        Returns:
            Farmer indices for which an ML decision was processed.

        Raises:
            RuntimeError: If a due temporal stream is incomplete or the switch gate removes
                all usable probability mass.
            ValueError: If RF inputs or probability outputs violate the trained contract.
        """
        if self.model.in_spinup:
            # Daily input histories are intentionally collected during spin-up,
            # but prescribed calendars must remain the only decision mechanism.
            return np.empty(0, dtype=np.int64)

        due = self.farmers_due_for_earliest_subregion_candidate_planting(
            self.model.current_time
        )
        if due.size == 0:
            return due
        run_due_timer = time.perf_counter()
        encoder_seconds = 0.0
        tabular_seconds = 0.0
        forest_seconds = 0.0
        selection_seconds = 0.0
        if not np.all(self.stream_ready[due]):
            incomplete = due[~self.stream_ready[due]][:20]
            raise RuntimeError(
                "An ML crop decision became due before all "
                f"{self.block_count} encoded blocks were complete. Affected "
                f"farmers: {incomplete.tolist()}."
            )
        if np.any(self.stream_block_count[due] != self.block_count):
            raise RuntimeError("A ready crop-choice stream has an invalid block count.")

        predicted_calendars = np.empty((due.size, 3), dtype=np.int32)
        prediction_probabilities = np.empty(due.size, dtype=np.float32)
        for start in range(0, due.size, self.batch_size):
            stop = min(start + self.batch_size, due.size)
            batch_farmers = due[start:stop]
            phase_timer = time.perf_counter()
            with self.torch.inference_mode():
                latent = (
                    self.encoder.encode_blocks(
                        self.torch.from_numpy(self.stream_blocks[batch_farmers]).to(
                            self.device
                        )
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            encoder_seconds += time.perf_counter() - phase_timer

            phase_timer = time.perf_counter()
            tabular = self._tabular_values(batch_farmers, self.model.current_time)
            tabular_seconds += time.perf_counter() - phase_timer

            phase_timer = time.perf_counter()
            # The RF was fitted on [temporal latent | imputed tabular] in this exact
            # order, so keep the concatenation contract explicit here.
            rf_input = np.concatenate((latent, tabular), axis=1)
            if rf_input.shape[1] != int(self.random_forest.n_features_in_):
                raise ValueError(
                    "Online RF feature width differs from training: "
                    f"{rf_input.shape[1]} versus "
                    f"{self.random_forest.n_features_in_}."
                )
            raw_probabilities = np.asarray(
                self.random_forest.predict_proba(rf_input), dtype=np.float64
            )
            rf_classes = np.asarray(self.random_forest.classes_, dtype=np.int64)
            expected_probability_shape = (batch_farmers.size, rf_classes.size)
            if raw_probabilities.shape != expected_probability_shape:
                raise ValueError(
                    "Random-forest probability output has shape "
                    f"{raw_probabilities.shape}; expected "
                    f"{expected_probability_shape}."
                )
            if np.any(~np.isfinite(raw_probabilities)) or np.any(
                raw_probabilities < 0.0
            ):
                raise ValueError(
                    "Random-forest probabilities must be finite and non-negative."
                )
            row_sums = raw_probabilities.sum(axis=1)
            if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
                raise ValueError(
                    "Random-forest probability rows must have positive mass."
                )
            probabilities = np.zeros(
                (batch_farmers.size, len(self.target_classes)), dtype=np.float64
            )
            probabilities[:, rf_classes] = raw_probabilities

            switch_gate = self.bundle.get("switch_gate")
            if self.prediction_model == "random_forest_switch_gate":
                if not switch_gate or not switch_gate.get("enabled", False):
                    raise ValueError(
                        "prediction_model='random_forest_switch_gate' was requested, "
                        "but the deployed crop-choice bundle has no enabled switch gate."
                    )
                threshold = float(switch_gate["selected_threshold"])
                current_crop = self.calendar_history[batch_farmers, 0, 0]
                decision_day_index = self.model.current_time.timetuple().tm_yday - 1
                feasible_classes = (self.target_classes[:, 0] < 0) | (
                    self.target_classes[:, 1] >= decision_day_index
                )
                for row_index, crop_id in enumerate(current_crop):
                    same_crop = self.target_classes[:, 0] == int(crop_id)
                    switch_probability = 1.0 - probabilities[row_index, same_crop].sum()
                    same_crop_feasible_mass = probabilities[
                        row_index, same_crop & feasible_classes
                    ].sum()
                    # Feasibility has priority over the persistence gate. If every
                    # same-crop calendar supported by the RF has already missed its
                    # planting day, do not gate away the feasible switching classes.
                    if (
                        same_crop.any()
                        and switch_probability < threshold
                        and same_crop_feasible_mass > 0.0
                    ):
                        probabilities[row_index, ~same_crop] = 0.0
                        total = probabilities[row_index].sum()
                        if total <= 0.0 or not np.isfinite(total):
                            raise RuntimeError(
                                "The switch gate removed all crop-calendar "
                                "probability mass."
                            )
                        probabilities[row_index] /= total

            forest_seconds += time.perf_counter() - phase_timer
            phase_timer = time.perf_counter()
            selected, selected_probability = self._sample_calendar_classes(
                probabilities, self.model.current_time
            )
            selection_seconds += time.perf_counter() - phase_timer
            predicted_calendars[start:stop] = self.target_classes[selected]
            prediction_probabilities[start:stop] = selected_probability
            if self.latent is not None:
                self.latent[batch_farmers] = latent

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
        self.pending_target_year[due] = -1
        self.pending_due_date[due] = np.datetime64("NaT", "D")
        self.stream_start_date[due] = np.datetime64("NaT", "D")
        self.stream_ready[due] = False
        self.awaiting_decision[due] = False
        install_seconds = time.perf_counter() - install_timer
        self.model.logger.info(
            "Installed %s ML-predicted crop calendar(s) on %s.",
            due.size,
            self.model.current_time.date(),
        )
        # The selected calendar makes its final harvest date known immediately,
        # so prepare the next decision-relative stream before its first raw day.
        schedule_timer = time.perf_counter()
        self.schedule(due)
        next_schedule_seconds = time.perf_counter() - schedule_timer
        self.model.logger.info(
            "DecisionModuleML timing on %s for %s decision(s): encoder=%.2f s, "
            "tabular=%.2f s, forest/gate=%.2f s, sampling=%.2f s, install=%.2f s, "
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
