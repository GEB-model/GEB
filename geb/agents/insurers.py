"""This module contains the insurers agent class for GEB."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geb.geb_types import TwoDArrayFloat32
from geb.workflows.raster import sample_from_map

from ..store import Bucket, DynamicArray
from .crop_farmers import (
    INDEX_INSURANCE_ADAPTATION,
    PR_INSURANCE_ADAPTATION,
    TRADITIONAL_INSURANCE_ADAPTATION,
    shift_and_reset_matrix,
)
from .general import AgentBaseClass
from .workflows.crop_farmers import (
    compute_premiums_and_best_contracts_numba,
)

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel

logger = logging.getLogger(__name__)


class InsurersVariables(Bucket):
    """Variables for the Insurers agent."""

    insured_yearly_income: DynamicArray
    insurance_duration: int
    GEV_pr_parameters: DynamicArray
    avg_income_per_agent: npt.NDArray[np.floating]
    adjusted_yearly_income_insured: DynamicArray


class Insurers(AgentBaseClass):
    """This class is used to simulate the insurers.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    var: InsurersVariables

    def __init__(self, model: GEBModel, agents: Agents) -> None:
        """Initialize the insurers agent.

        Args:
            model: The GEB model.
            agents: The class that includes all agent types (allowing easier communication between agents).
        """
        super().__init__(model)
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["insurers"]
            if "insurers" in self.model.config["agent_settings"]
            else {}
        )
        if self.model.simulate_hydrology:
            self.HRU = model.hydrology.HRU
            self.grid = model.hydrology.grid

        self.hydrological_year_start_month = self.model.config["general"][
            "hydrological_year_start_month"
        ]
        self.traditional_insurance_adaptation_active = (
            not self.config["traditional_insurance"]["ruleset"] == "no-adaptation"
        )
        self.index_insurance_adaptation_active = (
            not self.config["index_insurance"]["ruleset"] == "no-adaptation"
        )
        self.pr_insurance_adaptation_active = (
            not self.config["pr_insurance"]["ruleset"] == "no-adaptation"
        )

        self.traditional_loading_rate = self.config["traditional_insurance"][
            "loading_rate"
        ]
        self.index_loading_rate = self.config["index_insurance"]["loading_rate"]
        self.pr_loading_rate = self.config["pr_insurance"]["loading_rate"]

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Name of the module.

        Returns:
            The name of the module.
        """
        return "agents.insurers"

    def spinup(self) -> None:
        """This function is called during model spinup."""
        self.var.insured_yearly_income = DynamicArray(
            n=self.agents.crop_farmers.var.n,
            max_n=self.agents.crop_farmers.var.max_n,
            extra_dims=(self.agents.crop_farmers.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )

        self.var.insurance_duration = self.config["duration"]

        self.var.GEV_pr_parameters = DynamicArray(
            n=self.agents.crop_farmers.var.n,
            max_n=self.agents.crop_farmers.var.max_n,
            extra_dims=(3,),
            extra_dims_names=("gev_parameters",),
            dtype=np.float32,
            fill_value=np.nan,
        )

        if (
            self.pr_insurance_adaptation_active
            or self.traditional_insurance_adaptation_active
            or self.index_insurance_adaptation_active
        ):
            for i, varname in enumerate(["pr_gev_c", "pr_gev_loc", "pr_gev_scale"]):
                GEV_pr_grid = getattr(self.grid, varname)
                self.var.GEV_pr_parameters[:, i] = sample_from_map(
                    GEV_pr_grid.values,
                    self.agents.crop_farmers.var.locations.data,
                    GEV_pr_grid.rio.transform().to_gdal(),
                )
                assert not np.isnan(self.var.GEV_pr_parameters[:, i]).any(), (
                    f"{i} contains NaN values"
                )  # ensure no NaNs in data

    def government_premium_cap_india(self) -> np.ndarray:
        """Compute per-farmer government premium cap in India based on income and crop mix.

        Farmers are grouped by well status. If all farmers in a group have
        sugarcane (``crop_calendar[..., -1, 0] == 11``), the cap is 5% of mean
        income per m²; otherwise 2%. Caps are then scaled by each farmer's field
        size.

        Returns:
            Premium cap per farmer.
        """
        year_income_m2 = (
            self.agents.crop_farmers.var.yearly_income[:, 0]
            / self.agents.crop_farmers.field_size_per_farmer
        )

        group_indices, n_groups = self.agents.crop_farmers.create_unique_groups(
            self.agents.crop_farmers.well_irrigated,
        )
        group_mean_cap = np.zeros(n_groups, dtype=float)
        for group_idx in range(n_groups):
            agent_indices = np.where(group_indices == group_idx)[0]
            sugarcane_check = np.all(
                self.agents.crop_farmers.var.crop_calendar[agent_indices, -1, 0] == 11
            )
            if sugarcane_check:
                group_mean_cap[group_idx] = (
                    np.mean(year_income_m2[agent_indices]) * 0.05
                )
            else:
                group_mean_cap[group_idx] = (
                    np.mean(year_income_m2[agent_indices]) * 0.02
                )

        agent_caps = (
            group_mean_cap[group_indices]
            * self.agents.crop_farmers.field_size_per_farmer
        )

        return agent_caps

    def potential_insured_loss(self) -> np.ndarray:
        """Compute potential insured loss per farmer-year.

        Masks unfilled years (all-zero income), computes each farmer's average
        income over filled years, and sets the potential insured loss as the
        positive difference between that average and the realized income.

        Returns:
            Array shaped like ``yearly_income`` with per farmer-year potential
                insured losses (``float32``). Masked years remain zero.
        """
        # Calculating traditional pure premiums and Bühlmann-Straub parameters to get the credibility premium
        # Mask out unfilled years
        mask_columns = np.all(self.agents.crop_farmers.var.yearly_income == 0, axis=0)

        # Apply the mask to data
        income_masked = self.agents.crop_farmers.var.yearly_income.data[
            :, ~mask_columns
        ]
        n_agents, n_years = income_masked.shape

        # Calculate traditional loss
        self.var.avg_income_per_agent = np.nanmean(income_masked, axis=1)

        potential_insured_loss = np.zeros_like(
            self.agents.crop_farmers.var.yearly_income, dtype=np.float32
        )

        potential_insured_loss[:, ~mask_columns] = np.maximum(
            self.var.avg_income_per_agent[..., None] - income_masked, 0
        )

        return potential_insured_loss

    def premium_traditional_insurance(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        government_premium_cap: npt.NDArray[np.floating],
        masked_income: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute capped traditional insurance premiums via Bühlmann–Straub credibility.

        Uses historical insured losses and realized income to estimate expected loss
        per farmer under traditional insurance, applies the credibility-based
        premium calculation, and caps premiums at the government premium cap.

        Args:
            potential_insured_loss: Potential insured loss per farmer-year.
            government_premium_cap: Maximum premium allowed per farmer.
            masked_income: Historical realized income for valid years.

        Returns:
            Capped traditional insurance premium per farmer.
        """
        # Calculating traditional pure premiums and Bühlmann-Straub parameters to get the credibility premium
        # Calculate traditional loss
        agent_pure_premiums_m2 = (
            np.mean(potential_insured_loss, axis=1)
            / self.agents.crop_farmers.field_size_per_farmer
        )

        group_indices, n_groups = self.agents.crop_farmers.create_unique_groups(
            self.agents.crop_farmers.well_irrigated,
        )

        years_observed = np.sum(~np.isnan(masked_income), axis=1)
        # Initialize arrays for coefficients and R²
        group_mean_premiums = np.zeros(n_groups, dtype=float)
        for group_idx in range(n_groups):
            agent_indices = np.where(group_indices == group_idx)[0]
            group_mean_premiums[group_idx] = np.mean(
                agent_pure_premiums_m2[agent_indices]
            )

        sample_var_per_agent = np.var(potential_insured_loss, axis=1, ddof=1)
        valid_for_within = years_observed > 1

        within_variance = np.sum(
            (years_observed[valid_for_within] - 1)
            * sample_var_per_agent[valid_for_within]
        ) / np.sum(years_observed[valid_for_within] - 1)
        between_variance = np.var(agent_pure_premiums_m2, ddof=1)
        credibility_param_K = (
            within_variance / between_variance if between_variance > 0 else np.inf
        )

        # Classical Bühlmann–Straub: Z = n / (n + K)
        credibility_weights = years_observed / (years_observed + credibility_param_K)
        credibility_premiums_m2 = (
            credibility_weights * agent_pure_premiums_m2
            + (1 - credibility_weights) * group_mean_premiums[group_indices]
        )
        # Return to traditional prices and add loading factor
        traditional_premium = (
            credibility_premiums_m2
            * self.agents.crop_farmers.field_size_per_farmer
            * self.traditional_loading_rate
        )

        return np.minimum(government_premium_cap, traditional_premium)

    def insured_payouts_traditional(
        self,
        insured_farmers_mask: DynamicArray,
        masked_income: npt.NDArray[np.floating],
        historic_window_years: np.int8 = 7,
    ) -> npt.NDArray[np.floating]:
        """Compute historical traditional-insurance payouts and update state.

        Derives traditional insurance payouts for insured farmers from historical
        income, adds the payout for the current year to insured income, and returns
        the historical payout matrix.

        Args:
            insured_farmers_mask: Boolean mask indicating which farmers are insured
                with traditional insurance.
            masked_income: Historical realized income for valid years.
            historic_window_years: Historical window. 7 is used based on the PFMBY scheme
                in India.

        Returns:
            historical traditional-insurance payouts per farmer-year.
        """
        cumsum = np.cumsum(masked_income, axis=1, dtype=float)

        n_agents, T = masked_income.shape
        years = np.arange(T)
        thr_m = np.empty_like(masked_income, dtype=float)

        first7 = years < historic_window_years
        thr_m[:, first7] = cumsum[:, first7] / (years[first7] + 1)

        if T > historic_window_years:
            window_sum = (
                cumsum[:, historic_window_years:] - cumsum[:, :-historic_window_years]
            )
            thr_m[:, historic_window_years:] = window_sum / historic_window_years

        thr_m = thr_m[:, ::-1]

        threshold_full = np.zeros_like(masked_income, dtype=float)
        threshold_full[:, :] = thr_m

        insured_losses = np.maximum(threshold_full - masked_income, 0)

        self.var.insured_yearly_income[insured_farmers_mask, 0] += insured_losses[
            insured_farmers_mask, 0
        ]

        return insured_losses

    def premium_index_insurance(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        history: npt.NDArray[np.floating],
        gev_params: npt.NDArray[np.floating],
        strike_vals: npt.NDArray[np.floating],
        exit_vals: npt.NDArray[np.floating],
        rate_vals: npt.NDArray[np.floating],
        government_premium_cap: npt.NDArray[np.floating],
        loading_rate: np.float32,
        valid_years: npt.NDArray[np.bool_],
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """Select an index-insurance contract and compute capped premiums.

        Builds candidate contracts from strike, exit, and rate values, evaluates
        basis risk using past losses, selects the best contract per farmer, applies
        a loading factor to premiums, and caps premiums at the government premium
        cap.

        Args:
            potential_insured_loss: Potential insured loss per farmer-year; shape
                matches the historical income arrays.
            history: Historical index values per farmer-year, such as SPEI or
                precipitation, aligned with ``potential_insured_loss``.
            gev_params: Fitted GEV parameters per farmer for the insured index.
            strike_vals: Candidate strike levels.
            exit_vals: Candidate exit levels.
            rate_vals: Candidate rate-on-line values.
            government_premium_cap: Maximum premium allowed per farmer.
            loading_rate: Multiplicative loading factor applied to the actuarially
                estimated premium.
            valid_years: Boolean mask selecting years with valid historical income
                and index data.

        Returns:
            Best strike, best exit, best rate, and capped premium per farmer.
        """
        # Make a series of candidate insurance contracts and find the optimal contract
        # with the least basis risk considering past losses
        potential_insured_loss_masked = potential_insured_loss[:, valid_years]
        history_masked = history[:, valid_years]

        (
            best_strike_idx,
            best_exit_idx,
            best_rate_idx,
            best_rmse,
            best_prem,
        ) = compute_premiums_and_best_contracts_numba(
            gev_params,
            history_masked,
            potential_insured_loss_masked,
            strike_vals,
            exit_vals,
            rate_vals,
            n_sims=100,
            seed=42,
        )

        best_strike = strike_vals[best_strike_idx]
        best_exit = exit_vals[best_exit_idx]
        best_rate = rate_vals[best_rate_idx]
        best_premiums = best_prem * loading_rate  # add loading factor

        return (
            best_strike,
            best_exit,
            best_rate,
            np.minimum(best_premiums, government_premium_cap),
        )

    def estimate_index_insurance_candidate_space(
        self,
        yearly_spei: npt.NDArray[np.floating],
        masked_income: npt.NDArray[np.floating],
        masked_potential_income: npt.NDArray[np.floating],
        valid_years: npt.NDArray[np.bool_],
        *,
        strike_step: float = 0.2,
        exit_step: float = 0.2,
        n_strike_vals: int = 13,
        n_exit_vals: int = 9,
        n_rate_vals: int = 10,
        strike_half_width: float = 0.6,
        exit_half_width: float = 0.6,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Estimate simple group-level candidate spaces for strike, exit, and rate.

        The method uses as few fixed assumptions as possible:
        - positive-loss years are years where potential income exceeds actual income
        - strike is the upper quartile of SPEI in positive-loss years
        - exit is the lower quartile of SPEI in positive-loss years
        - rate is the median monetary loss in drought-loss years

        This provides a simple first estimate of the candidate parameter space that
        can be passed directly into the contract search.

        Args:
            yearly_spei: Historical yearly SPEI per farmer-year.
            masked_income: Historical realized income per farmer-year.
            masked_potential_income: Historical potential income per farmer-year.
            valid_years: Boolean mask indicating valid historical years.
            strike_step: Step size for strike candidate grid.
            exit_step: Step size for exit candidate grid.
            n_strike_vals: Number of strike candidates.
            n_exit_vals: Number of exit candidates.
            n_rate_vals: Number of rate candidates.
            strike_half_width: Half-width around the strike center.
            exit_half_width: Half-width around the exit center.

        Returns:
            Candidate arrays for strike, exit, and rate.

        Raises:
            ValueError: If the masked SPEI, income, and potential-income arrays do
                not have matching shapes.
        """
        print("Estimating index insurance candidate parameters..")
        spei = np.asarray(yearly_spei[:, valid_years], dtype=np.float64)
        income = np.asarray(masked_income, dtype=np.float64)
        potential_income = np.asarray(masked_potential_income, dtype=np.float64)

        if spei.shape != income.shape or income.shape != potential_income.shape:
            raise ValueError(
                "Shapes must match after masking valid years: "
                f"SPEI {spei.shape}, income {income.shape}, "
                f"potential income {potential_income.shape}."
            )

        loss = np.maximum(potential_income - income, 0.0)

        valid_mask = (
            np.isfinite(spei)
            & np.isfinite(loss)
            & np.isfinite(income)
            & np.isfinite(potential_income)
            & (potential_income > 0.0)
        )

        spei_flat = spei[valid_mask]
        loss_flat = loss[valid_mask]

        if spei_flat.size == 0:
            strike_center = -1.0
            exit_center = -2.0
            rate_center = 100.0
        else:
            positive_loss_mask = loss_flat > 0.0

            if np.any(positive_loss_mask):
                spei_loss = spei_flat[positive_loss_mask]
                loss_pos = loss_flat[positive_loss_mask]

                # Contract thresholds from the distribution of SPEI in loss years
                strike_center = float(np.quantile(spei_loss, 0.75))
                exit_center = float(np.quantile(spei_loss, 0.25))

                # Prefer losses in years that are actually drought-like
                drought_loss_mask = spei_loss <= strike_center
                if np.any(drought_loss_mask):
                    rate_center = float(np.median(loss_pos[drought_loss_mask]))
                else:
                    rate_center = float(np.median(loss_pos))
            else:
                # Fallback: use the overall SPEI distribution if no losses are observed
                strike_center = float(np.quantile(spei_flat, 0.33))
                exit_center = float(np.quantile(spei_flat, 0.10))
                rate_center = 100.0

        min_gap = max(strike_step, exit_step, 0.2)
        if exit_center >= strike_center - min_gap:
            exit_center = strike_center - min_gap

        strike_vals = self._build_candidate_grid(
            strike_center,
            strike_half_width,
            strike_step,
            n_strike_vals,
        )
        exit_vals = self._build_candidate_grid(
            exit_center,
            exit_half_width,
            exit_step,
            n_exit_vals,
        )

        exit_vals = exit_vals[exit_vals < np.max(strike_vals) - min_gap]
        if exit_vals.size == 0:
            exit_vals = np.array([strike_center - min_gap], dtype=np.float64)

        # Keep rate space simple and scale it around the observed loss magnitude
        rate_center = max(rate_center, 1.0)
        rate_low = max(1.0, rate_center * 0.5)
        rate_high = max(rate_low * 1.5, rate_center * 2.0)

        rate_vals = np.geomspace(rate_low, rate_high, n_rate_vals, dtype=np.float64)

        return strike_vals, exit_vals, rate_vals

    def estimate_pr_insurance_candidate_space(
        self,
        yearly_pr: npt.NDArray[np.floating],
        masked_income: npt.NDArray[np.floating],
        masked_potential_income: npt.NDArray[np.floating],
        valid_years: npt.NDArray[np.bool_],
        *,
        strike_step: float = 50.0,
        exit_step: float = 50.0,
        n_strike_vals: int = 13,
        n_exit_vals: int = 9,
        n_rate_vals: int = 10,
        strike_half_width: float = 150.0,
        exit_half_width: float = 150.0,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Estimate simple group-level candidate spaces for precipitation insurance.

        The method uses as few fixed assumptions as possible:
        - positive-loss years are years where potential income exceeds actual income
        - strike is the upper quartile of precipitation in positive-loss years
        - exit is the lower quartile of precipitation in positive-loss years
        - rate is the median monetary loss in dry-loss years

        This provides a simple first estimate of the candidate parameter space that
        can be passed directly into the existing contract search.

        Args:
            yearly_pr: Historical yearly precipitation totals per farmer-year.
            masked_income: Historical realized income per farmer-year.
            masked_potential_income: Historical potential income per farmer-year.
            valid_years: Boolean mask indicating valid historical years.
            strike_step: Step size for strike candidate grid in precipitation units.
            exit_step: Step size for exit candidate grid in precipitation units.
            n_strike_vals: Number of strike candidates.
            n_exit_vals: Number of exit candidates.
            n_rate_vals: Number of rate candidates.
            strike_half_width: Half-width around the strike center.
            exit_half_width: Half-width around the exit center.

        Returns:
            Candidate arrays for strike, exit, and rate.

        Raises:
            ValueError: If the masked precipitation, income, and potential-income
                arrays do not have matching shapes.
        """
        print("Estimating precipitation insurance candidate parameters..")
        pr = np.asarray(yearly_pr[:, valid_years], dtype=np.float64)
        income = np.asarray(masked_income, dtype=np.float64)
        potential_income = np.asarray(masked_potential_income, dtype=np.float64)

        if pr.shape != income.shape or income.shape != potential_income.shape:
            raise ValueError(
                "Shapes must match after masking valid years: "
                f"precipitation {pr.shape}, income {income.shape}, "
                f"potential income {potential_income.shape}."
            )

        loss = np.maximum(potential_income - income, 0.0)

        valid_mask = (
            np.isfinite(pr)
            & np.isfinite(loss)
            & np.isfinite(income)
            & np.isfinite(potential_income)
            & (potential_income > 0.0)
        )

        pr_flat = pr[valid_mask]
        loss_flat = loss[valid_mask]

        if pr_flat.size == 0:
            strike_center = 1000.0
            exit_center = 500.0
            rate_center = 100.0
        else:
            positive_loss_mask = loss_flat > 0.0

            if np.any(positive_loss_mask):
                pr_loss = pr_flat[positive_loss_mask]
                loss_pos = loss_flat[positive_loss_mask]

                strike_center = float(np.quantile(pr_loss, 0.75))
                exit_center = float(np.quantile(pr_loss, 0.25))

                dry_loss_mask = pr_loss <= strike_center
                if np.any(dry_loss_mask):
                    rate_center = float(np.median(loss_pos[dry_loss_mask]))
                else:
                    rate_center = float(np.median(loss_pos))
            else:
                strike_center = float(np.quantile(pr_flat, 0.33))
                exit_center = float(np.quantile(pr_flat, 0.10))
                rate_center = 100.0

        min_gap = max(strike_step, exit_step, 50.0)
        if exit_center >= strike_center - min_gap:
            exit_center = strike_center - min_gap

        strike_vals = self._build_candidate_grid(
            strike_center,
            strike_half_width,
            strike_step,
            n_strike_vals,
        )
        exit_vals = self._build_candidate_grid(
            exit_center,
            exit_half_width,
            exit_step,
            n_exit_vals,
        )

        exit_vals = exit_vals[exit_vals < np.max(strike_vals) - min_gap]
        if exit_vals.size == 0:
            exit_vals = np.array([strike_center - min_gap], dtype=np.float64)

        rate_center = max(rate_center, 1.0)
        rate_low = max(1.0, rate_center * 0.5)
        rate_high = max(rate_low * 1.5, rate_center * 2.0)

        rate_vals = np.geomspace(rate_low, rate_high, n_rate_vals, dtype=np.float64)

        return strike_vals, exit_vals, rate_vals

    def _build_candidate_grid(
        self,
        center: float,
        half_width: float,
        step: float,
        n_vals: int,
    ) -> npt.NDArray[np.float64]:
        """Build a descending grid around a center and snap it to a fixed step.

        Args:
            center: Central value around which the candidate grid is built.
            half_width: Distance from the center to the upper and lower edge of the
                initial grid before snapping.
            step: Step size to which values are snapped.
            n_vals: Number of candidate values to return.

        Returns:
           Descending candidate grid.
        """
        vals = center + np.linspace(half_width, -half_width, n_vals, dtype=np.float64)
        vals = np.round(vals / step) * step
        vals = np.unique(vals)[::-1]

        if vals.size < n_vals:
            start = np.max(vals) if vals.size > 0 else center + half_width
            vals = start - step * np.arange(n_vals, dtype=np.float64)

        return vals.astype(np.float64)

    def insured_payouts_index(
        self,
        strike: npt.NDArray[np.floating],
        exit: npt.NDArray[np.floating],
        rate: npt.NDArray[np.floating],
        insured_farmers_mask: DynamicArray,
        index_nr: int,
        valid_years: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        """Compute index-insurance payouts historically and update state.

        Uses strike, exit, and per-farmer rate values to derive payouts from the
        historical insured index, updates ``insured_yearly_income`` for insured
        farmers, and records payout events in ``payout_mask`` at ``index_nr``.

        Args:
            strike: Strike level per farmer.
            exit: Exit level per farmer, which must be less than or equal to strike.
            rate: Rate-on-line per farmer.
            insured_farmers_mask: Boolean mask indicating which farmers are insured.
            index_nr: Column in ``payout_mask`` corresponding to this insurance
                product.
            valid_years: Boolean mask selecting years with valid historical income
                and index data.

        Returns:
            Per farmer-year payouts shaped like the historical income array over all years.
        """
        # Determine what the index insurance would have paid out in the past
        spei_hist = self.agents.crop_farmers.var.yearly_SPEI.data[:, valid_years]

        denom = strike - exit
        shortfall = strike[:, None] - spei_hist
        # (no payout if rainfall ≥ strike)
        shortfall = np.clip(shortfall, 0.0, None)
        # (full payout once exit is breached)
        shortfall = np.minimum(shortfall, denom[:, None])
        # convert to fraction of maximum shortfall
        ratio = shortfall / denom[:, None]
        # scale by each agent’s rate
        payouts = ratio * rate[:, None]

        potential_insured_loss = np.zeros_like(spei_hist, dtype=np.float32)
        potential_insured_loss[:, :] = payouts

        self.var.insured_yearly_income[insured_farmers_mask, 0] += (
            potential_insured_loss[insured_farmers_mask, 0]
        )

        return potential_insured_loss

    def insured_yields(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        valid_years: npt.NDArray[np.bool_],
        masked_income: npt.NDArray[np.floating],
        masked_potential_income: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute insured yield-index relation given that the agent had insurance.

        Adds potential insured losses to historical income, converts the result to
        a yield ratio relative to potential income, clips that ratio to ``[0, 1]``,
        and derives the insured yield relation using the groupwise linear relation.

        Args:
            potential_insured_loss: Potential insured loss per farmer-year.
            valid_years: Boolean mask selecting years with valid historical income
                and index data.
            masked_income: Historical realized income for valid years.
            masked_potential_income: Historical potential income for valid years.

        Returns:
            Insured yield-index relation per farmer-year.
        """
        insured_yearly_income = masked_income + potential_insured_loss

        insured_yearly_yield_ratio = insured_yearly_income / masked_potential_income

        insured_yearly_yield_ratio = np.clip(insured_yearly_yield_ratio, 0, 1)

        insured_yield_probability_relation = (
            self.agents.crop_farmers.calculate_yield_spei_relation_group_lin(
                insured_yearly_yield_ratio,
                self.agents.crop_farmers.var.yearly_SPEI_probability.data[
                    :, valid_years
                ],
            )
        )
        return insured_yield_probability_relation

    def inflation_correction_export(self) -> None:
        """Create inflation-adjust exports.

        Computes inflation-adjusted views of yearly income, and premiums for export.
        """
        self.var.adjusted_yearly_income_insured = (
            self.agents.crop_farmers.var.yearly_income
            / self.agents.crop_farmers.var.cumulative_inflation[..., None]
        )

    def insurance_premiums(self) -> tuple[npt.NDArray[np.floating], TwoDArrayFloat32]:
        """Compute premiums and insured yield-probability relations.

        Prepares common insurance inputs, including potential insured loss, optional
        premium caps, and the subset of historical years with non-zero potential
        income for all farmers. It then dispatches to the active insurance product
        and returns that product's premium and insured yield-probability relation.

        Returns:
            Premium per farmer and the corresponding insured
                yield-probability relation for the active insurance product.

        Raises:
            ValueError: If no insurance adaptation is active.
        """
        farmer_yield_probability_relation = (
            self.agents.crop_farmers.farmer_yield_probability_relation
        )
        farmer_yield_probability_relation_budget_cap = (
            self.agents.crop_farmers.farmer_yield_probability_relation_budget_cap
        )

        # Set the base insured income of this year as the yearly income.
        # Later, insured losses will be added to this.
        self.var.insured_yearly_income[:, 0] = (
            self.agents.crop_farmers.var.yearly_income[:, 0].copy()
        )

        potential_insured_loss = self.potential_insured_loss()

        if self.config["government_premium_cap"]:
            government_premium_cap = self.government_premium_cap_india()
        else:
            government_premium_cap = np.full(self.agents.crop_farmers.var.n, np.inf)

        valid_years = ~(self.agents.crop_farmers.var.yearly_potential_income == 0).any(
            axis=0
        )
        masked_income = self.agents.crop_farmers.var.yearly_income.data[:, valid_years]
        masked_potential_income = (
            self.agents.crop_farmers.var.yearly_potential_income.data[:, valid_years]
        )

        if self.traditional_insurance_adaptation_active:
            (
                traditional_premium,
                farmer_yield_probability_relation_insured_traditional,
            ) = self.traditional_insurance(
                potential_insured_loss,
                government_premium_cap,
                farmer_yield_probability_relation,
                farmer_yield_probability_relation_budget_cap,
                valid_years,
                masked_income,
                masked_potential_income,
            )
            return (
                traditional_premium,
                farmer_yield_probability_relation_insured_traditional,
            )
        if self.index_insurance_adaptation_active:
            index_premium, farmer_yield_probability_relation_insured_index = (
                self.index_insurance(
                    potential_insured_loss,
                    government_premium_cap,
                    farmer_yield_probability_relation_budget_cap,
                    valid_years,
                    masked_income,
                    masked_potential_income,
                )
            )
            return index_premium, farmer_yield_probability_relation_insured_index
        if self.pr_insurance_adaptation_active:
            pr_premium, farmer_yield_probability_relation_insured_pr = (
                self.pr_insurance(
                    potential_insured_loss,
                    government_premium_cap,
                    farmer_yield_probability_relation_budget_cap,
                    valid_years,
                    masked_income,
                    masked_potential_income,
                )
            )
            return pr_premium, farmer_yield_probability_relation_insured_pr

        msg = "No insurance adaptation is active."
        raise ValueError(msg)

    def traditional_insurance(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        government_premium_cap: npt.NDArray[np.floating],
        farmer_yield_probability_relation: npt.NDArray[np.floating],
        farmer_yield_probability_relation_budget_cap: npt.NDArray[np.floating],
        valid_years: npt.NDArray[np.bool_],
        masked_income: npt.NDArray[np.floating],
        masked_potential_income: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], TwoDArrayFloat32]:
        """Apply traditional insurance and update insured yield relations.

        Computes premiums for traditional insurance, determines historical payouts
        for insured farmers, recalculates the insured yield-probability relation,
        and writes the insured relation back to the standard and budget-capped
        relation arrays for insured farmers only.

        Args:
            potential_insured_loss: Potential insured loss per farmer-year.
            government_premium_cap: Maximum premium allowed per farmer.
            farmer_yield_probability_relation: Baseline farmer yield-probability
                relation to update for insured farmers.
            farmer_yield_probability_relation_budget_cap: Budget-capped
                farmer yield-probability relation to update for insured farmers.
            valid_years: Boolean mask selecting years with valid income histories.
            masked_income: Historical realized income for valid years.
            masked_potential_income: Historical potential income for valid years.

        Returns:
            Premium per farmer and the insured yield-probability relation under
                traditional insurance.
        """
        # Determine the potential (past and current) indemnity payments and
        # recalculate the probability-yield relation.
        traditional_premium = self.premium_traditional_insurance(
            potential_insured_loss,
            government_premium_cap,
            masked_income,
        )

        traditional_insured_farmers_mask = (
            self.agents.crop_farmers.var.adaptations[
                :, TRADITIONAL_INSURANCE_ADAPTATION
            ]
            > 0
        )

        # Add the insured loss to the income of this year's insured farmers.
        potential_insured_loss_traditional = self.insured_payouts_traditional(
            traditional_insured_farmers_mask,
            masked_income,
        )

        farmer_yield_probability_relation_insured_traditional = self.insured_yields(
            potential_insured_loss_traditional,
            valid_years,
            masked_income,
            masked_potential_income,
        )

        farmer_yield_probability_relation[traditional_insured_farmers_mask, :] = (
            farmer_yield_probability_relation_insured_traditional[
                traditional_insured_farmers_mask, :
            ]
        )
        farmer_yield_probability_relation_budget_cap[
            traditional_insured_farmers_mask, :
        ] = farmer_yield_probability_relation_insured_traditional[
            traditional_insured_farmers_mask, :
        ]

        return (
            traditional_premium,
            farmer_yield_probability_relation_insured_traditional,
        )

    def index_insurance(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        government_premium_cap: npt.NDArray[np.floating],
        farmer_yield_probability_relation_budget_cap: npt.NDArray[np.floating],
        valid_years: npt.NDArray[np.bool_],
        masked_income: npt.NDArray[np.floating],
        masked_potential_income: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], TwoDArrayFloat32]:
        """Apply SPEI-based index insurance and update insured yield relations.

        Estimates candidate strike, exit, and rate values from the historical SPEI
        and income data, selects the best-fitting contract per farmer, computes
        historical index payouts for insured farmers, and updates the budget-capped
        yield-probability relation for those farmers.

        Args:
            potential_insured_loss: Potential insured loss per farmer-year.
            government_premium_cap: Maximum premium allowed per farmer.
            farmer_yield_probability_relation_budget_cap: Budget-capped
                farmer yield-probability relation to update for insured farmers.
            valid_years: Boolean mask selecting years with valid income histories.
            masked_income: Historical realized income for valid years.
            masked_potential_income: Historical potential income for valid years.

        Returns:
            Premium per farmer and the insured yield-probability relation under
                SPEI-based index insurance.
        """
        gev_params = self.agents.crop_farmers.var.GEV_parameters.data
        strike_vals, exit_vals, rate_vals = (
            self.estimate_index_insurance_candidate_space(
                yearly_spei=self.agents.crop_farmers.var.yearly_SPEI.data,
                masked_income=masked_income,
                masked_potential_income=masked_potential_income,
                valid_years=valid_years,
            )
        )

        # Calculate the best strike, exit, and rate for the chosen contract.
        strike, exit, rate, index_premium = self.premium_index_insurance(
            potential_insured_loss=potential_insured_loss,
            history=self.agents.crop_farmers.var.yearly_SPEI.data,
            gev_params=gev_params,
            strike_vals=strike_vals,
            exit_vals=exit_vals,
            rate_vals=rate_vals,
            government_premium_cap=government_premium_cap,
            loading_rate=self.index_loading_rate,
            valid_years=valid_years,
        )

        index_insured_farmers_mask = (
            self.agents.crop_farmers.var.adaptations[:, INDEX_INSURANCE_ADAPTATION] > 0
        )
        potential_insured_loss_index = self.insured_payouts_index(
            strike,
            exit,
            rate,
            index_insured_farmers_mask,
            INDEX_INSURANCE_ADAPTATION,
            valid_years,
        )
        farmer_yield_probability_relation_insured_index = self.insured_yields(
            potential_insured_loss_index,
            valid_years,
            masked_income,
            masked_potential_income,
        )

        farmer_yield_probability_relation_budget_cap[index_insured_farmers_mask, :] = (
            farmer_yield_probability_relation_insured_index[
                index_insured_farmers_mask, :
            ]
        )

        return index_premium, farmer_yield_probability_relation_insured_index

    def pr_insurance(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        government_premium_cap: npt.NDArray[np.floating],
        farmer_yield_probability_relation_budget_cap: npt.NDArray[np.floating],
        valid_years: npt.NDArray[np.bool_],
        masked_income: npt.NDArray[np.floating],
        masked_potential_income: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], TwoDArrayFloat32]:
        """Apply precipitation index insurance and update insured yield relations.

        Estimates candidate strike, exit, and rate values from the historical
        precipitation and income data, selects the best-fitting contract per
        farmer, computes historical precipitation-index payouts for insured
        farmers, and updates the budget-capped yield-probability relation for
        those farmers.

        Args:
            potential_insured_loss: Potential insured loss per farmer-year.
            government_premium_cap: Maximum premium allowed per farmer.
            farmer_yield_probability_relation_budget_cap: Budget-capped
                farmer yield-probability relation to update for insured farmers.
            valid_years: Boolean mask selecting years with valid income histories.
            masked_income: Historical realized income for valid years.
            masked_potential_income: Historical potential income for valid years.

        Returns:
            Premium per farmer and the insured yield-probability relation under
                precipitation index insurance.
        """
        gev_params = self.var.GEV_pr_parameters.data
        strike_vals, exit_vals, rate_vals = self.estimate_pr_insurance_candidate_space(
            yearly_pr=self.agents.crop_farmers.var.yearly_pr.data,
            masked_income=masked_income,
            masked_potential_income=masked_potential_income,
            valid_years=valid_years,
        )

        # Calculate the best strike, exit, and rate for the chosen contract.
        strike, exit, rate, pr_premium = self.premium_index_insurance(
            potential_insured_loss=potential_insured_loss,
            history=self.agents.crop_farmers.var.yearly_pr.data,
            gev_params=gev_params,
            strike_vals=strike_vals,
            exit_vals=exit_vals,
            rate_vals=rate_vals,
            government_premium_cap=government_premium_cap,
            loading_rate=self.pr_loading_rate,
            valid_years=valid_years,
        )

        pr_insured_farmers_mask = (
            self.agents.crop_farmers.var.adaptations[:, PR_INSURANCE_ADAPTATION] > 0
        )
        potential_insured_loss_pr = self.insured_payouts_index(
            strike,
            exit,
            rate,
            pr_insured_farmers_mask,
            PR_INSURANCE_ADAPTATION,
            valid_years,
        )
        farmer_yield_probability_relation_insured_pr = self.insured_yields(
            potential_insured_loss_pr,
            valid_years,
            masked_income,
            masked_potential_income,
        )

        farmer_yield_probability_relation_budget_cap[pr_insured_farmers_mask, :] = (
            farmer_yield_probability_relation_insured_pr[pr_insured_farmers_mask, :]
        )

        return pr_premium, farmer_yield_probability_relation_insured_pr

    def step(self) -> None:
        """This function is run each timestep."""
        # Yearly actions
        if (
            self.model.current_time.month == self.hydrological_year_start_month
            and self.model.current_time.day == 1
        ):
            self.inflation_correction_export()
            shift_and_reset_matrix(self.var.insured_yearly_income)

        self.report(locals())
