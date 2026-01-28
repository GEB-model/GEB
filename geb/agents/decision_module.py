"""This module contains the DecisionModule class for handling decision-making processes in the GEB model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange


class DecisionModule:
    """This class implements the decision module for drought adaptation."""

    @staticmethod
    @njit(cache=True)
    def IterateThroughFloods(
        NPV_summed: np.ndarray,
        n_events: int,
        discount_rate: float,
        max_T: int,
        wealth: np.ndarray,
        income: np.ndarray,
        amenity_value: np.ndarray,
        expected_damages: np.ndarray,
    ) -> np.ndarray:
        """This function calculates the NPV of each flood event.

        Args:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent
            n_events: number of events (droughts and floods) included in model run
            discount_rate: time discounting rate
            max_T: time horizon (same for each agent)
            wealth: array containing the wealth of each household
            income: array containing the income of each household
            amenity_value: array containing the amenity value of each household
            expected_damages: array containing the expected damages of each household under all events i

        Returns:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent
        """
        NPV_t0 = (wealth + income + amenity_value).astype(
            np.float32
        )  # no flooding in t=0

        # Iterate through all floods
        for i, index in enumerate(np.arange(1, n_events + 3)):
            # Check if we are in the last iterations
            if i < n_events:
                NPV_event_i = wealth + income + amenity_value - expected_damages[i]
                NPV_event_i = (NPV_event_i).astype(np.float32)

            # if in the last two iterations do not subtract damages (probs of no
            # flood event)
            elif i >= n_events:
                NPV_event_i = wealth + income + amenity_value
                NPV_event_i = NPV_event_i.astype(np.float32)

            # Calculate time discounted NPVs
            t_arr = np.arange(1, max_T, dtype=np.float32)
            discounts = 1 / (1 + discount_rate) ** t_arr
            NPV_tx = np.sum(discounts) * NPV_event_i

            # Add NPV at t0 (which is not discounted)
            NPV_tx += NPV_t0

            # Store result
            NPV_summed[index] = NPV_tx

        # Store NPV at p=0 for bounds in integration
        NPV_summed[0] = NPV_summed[1]
        return NPV_summed

    @staticmethod
    @njit(cache=True)
    def IterateThroughDroughts(
        NPV_summed: np.ndarray,
        n_events: int,
        discount_rate: float,
        max_T: int,
        total_profits: np.ndarray,
        profits_no_event: np.ndarray,
    ) -> np.ndarray:
        """This function calculates the NPV of each drought event.

        Args:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent
            n_events: number of events (droughts and floods) included in model run
            discount_rate: time discounting rate
            max_T: time horizon (same for each agent)
            total_profits: array containing the total profits of each household under all events i
            profits_no_event: array containing the profits of each household under the no flood event

        Returns:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent
        """
        # Iterate through all droughts
        for i, index in enumerate(np.arange(1, n_events + 3)):
            # Check if we are in the last iterations
            if i < n_events:
                NPV_event_i = total_profits[i]
                NPV_event_i = (NPV_event_i).astype(np.float32)

            # if in the last two iterations do not subtract damages (probs of no event)
            elif i >= n_events:
                NPV_event_i = profits_no_event
                NPV_event_i = NPV_event_i.astype(np.float32)

            # iterate over NPVs for each year in the time horizon and apply time
            # discounting
            NPV_t0 = (profits_no_event).astype(np.float32)  # no flooding in t=0

            # Calculate time discounted NPVs
            t_arr = np.arange(1, max_T, dtype=np.float32)

            discounts = 1 / (1 + np.reshape(discount_rate, (-1, 1))) ** t_arr
            NPV_tx = np.sum(discounts, axis=1) * NPV_event_i

            # Add NPV at t0 (which is not discounted)
            NPV_tx += NPV_t0

            # Store result
            NPV_summed[index] = NPV_tx

        # Store NPV at p=0 for bounds in integration
        NPV_summed[0] = NPV_summed[1]
        return NPV_summed

    @staticmethod
    @njit(cache=True)
    def IterateThroughWindstorm(
        NPV_summed: np.ndarray,
        n_windstorms: int,
        wealth: np.ndarray,
        income: np.ndarray,
        amenity_value: np.ndarray,
        max_T: int,
        expected_damages: np.ndarray,
        r: float,
    ) -> np.ndarray:
        """This function calculates the NPV of each windstorm event.

        Args:
            NPV_summed: Array containing the summed time discounted NPV for each windstorm event i for each agent
            n_windstorms: number of windstorm events included in model run
            wealth: array containing the wealth of each household
            income: array containing the income of each household
            amenity_value: array containing the amenity value of each household
            max_T: time horizon (same for each agent)
            expected_damages: array containing the expected damages of each household under all windstorm events i
            n_agents: number of household agents in the current admin unit
            r: time discounting rate"

        Returns:
            NPV_summed_w: Array containing the summed time discounted NPV for each windstorm event i for each agent
        """
        # No windstorm in t=0
        NPV_t0 = (wealth + income + amenity_value).astype(np.float32)

        # Iterate through all windstorms events
        for i, index in enumerate(np.arange(1, n_windstorms + 3)):
            # Check if we are in the last iterations
            if i < n_windstorms:
                NPV_wind_i = wealth + income + amenity_value - expected_damages[i]
                NPV_wind_i = NPV_wind_i.astype(np.float32)
            # if in the last two windstorms do not subtract damages (probs of no event)
            elif i >= n_windstorms:
                # No damages for the last two 'no event' cases
                NPV_wind_i = wealth + income + amenity_value
                NPV_wind_i = NPV_wind_i.astype(np.float32)

            # Discounted future NPVs
            t_arr = np.arange(1, max_T, dtype=np.float32)
            discounts = 1 / (1 + r) ** t_arr
            NPV_tx = np.sum(discounts) * NPV_wind_i

            # Add NPV at t0 (which is not discounted)
            NPV_tx += NPV_t0

            # Store result
            NPV_summed[index] = NPV_tx

        # Store NPVat p=0 for bounds in integration
        NPV_summed[0] = NPV_summed[1]

        return NPV_summed

    @staticmethod
    @njit(cache=True)
    def IterateThroughMultiHazard(
        NPV_summed: np.ndarray,
        n_windstorms: int,
        n_floods: int,
        wealth: np.ndarray,
        income: np.ndarray,
        amenity_value: np.ndarray,
        max_T: int,
        expected_damages_flood: np.ndarray,
        expected_damages_wind: np.ndarray,
        r: float,
    ) -> np.ndarray:
        """This function calculates the NPV of each multi-hazard event.

        Args:
            NPV_summed: Array containing the summed time discounted NPV for each multi-hazard event i for each agent
            n_windstorms: number of windstorm events included in model run
            n_floods: number of flood events included in model run
            wealth: array containing the wealth of each household
            income: array containing the income of each household
            amenity_value: array containing the amenity value of each household
            max_T: time horizon (same for each agent)
            expected_damages_flood: array containing the expected damages of each household under all flood events i
            expected_damages_wind: array containing the expected damages of each household under all windstorm events i
            r: time discounting rate"

        Returns:
            NPV_summed_mh: Array containing the summed time discounted NPV for each multi-hazard event i for each agent
        """
        n_agents = wealth.shape[0]
        n_events = n_floods + n_windstorms
        expected_damages_total = np.zeros((n_events, n_agents), dtype=np.float32)

        # Combine expected damages from floods and windstorms
        for i in range(n_events):
            if i < n_floods:
                expected_damages_total[i] = expected_damages_flood[i]
            else:
                expected_damages_total[i] = expected_damages_wind[i - n_floods]

        # No windstorm in t=0
        NPV_t0 = (wealth + income + amenity_value).astype(
            np.float32
        )  # no hazard in t=0

        # Iterate through all hazard evenets
        for i, index in enumerate(np.arange(1, n_events + 3)):
            # Check if we are in the last iterations
            if i < n_events:
                NPV_hazards_i = (
                    wealth + income + amenity_value - expected_damages_total[i]
                )
                NPV_hazards_i = (NPV_hazards_i).astype(np.float32)

            # if in the last two iterations do not substract damages (probs of no event)
            elif i >= n_events:
                NPV_hazards_i = wealth + income + amenity_value
                NPV_hazards_i = NPV_hazards_i.astype(np.float32)

            # Calculate time discounted NPVs
            t_arr = np.arange(1, max_T, dtype=np.float32)
            discounts = 1 / (1 + r) ** t_arr
            NPV_tx = np.sum(discounts) * NPV_hazards_i

            # Add NPV at t0 (which is not discounted)
            NPV_tx += NPV_t0

            # Store results
            NPV_summed[index] = NPV_tx

        # Store NPV at p=0 for bounds in itegration
        NPV_summed[0] = NPV_summed[1]
        return NPV_summed

    def IterateThroughEvents(
        self,
        n_events: int,
        n_agents: int,
        discount_rate: float,
        max_T: int,
        wealth: np.ndarray | None = None,
        income: np.ndarray | None = None,
        amenity_value: np.ndarray | None = None,
        expected_damages: np.ndarray | None = None,
        expected_damages_wind: np.ndarray | None = None,
        expected_damages_flood: np.ndarray | None = None,
        total_profits: np.ndarray | None = None,
        profits_no_event: np.ndarray | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        """This function iterates through each (no)flood event i (see manuscript for details).

        It calculates the time discounted NPV of each event i:

        Args:
            n_events: number of events (droughts and floods) included in model run
            discount_rate: time discounting rate
            n_agents: number of household agents in the current admin unit
            wealth: array containing the wealth of each household
            income: array containing the income of each household
            amenity_value: array containing the amenity value of each household'
            max_T: time horizon (same for each agent)
            expected_damages: array containing the expected damages of each household under all events i
            total_profits: array containing the total profits of each household under all events i
            profits_no_event: array containing the profits of each household under the no flood event
            mode: either 'flood' or 'drought', indicating which type of event to iterate through

        Returns:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent

        Raises:
            ValueError: If `mode` is not 'flood' or 'drought'.
        """
        # Allocate array
        NPV_summed = np.full((n_events + 3, n_agents), -1, dtype=np.float32)

        if mode == "flood":
            NPV_summed = self.IterateThroughFloods(
                NPV_summed=NPV_summed,
                n_events=n_events,
                discount_rate=discount_rate,
                max_T=max_T,
                wealth=wealth,
                income=income,
                amenity_value=amenity_value,
                expected_damages=expected_damages,
            )

        elif mode == "drought":
            NPV_summed = self.IterateThroughDroughts(
                NPV_summed=NPV_summed,
                n_events=n_events,
                discount_rate=discount_rate,
                max_T=max_T,
                total_profits=total_profits,
                profits_no_event=profits_no_event,
            )

        elif mode == "windstorm":
            NPV_summed = self.IterateThroughWindstorm(
                NPV_summed=NPV_summed,
                n_windstorms=n_events,
                r=discount_rate,
                max_T=max_T,
                wealth=wealth,
                income=income,
                amenity_value=amenity_value,
                expected_damages=expected_damages,
            )

        elif mode == "multi_hazard":
            NPV_summed = self.IterateThroughMultiHazard(
                NPV_summed=NPV_summed,
                n_floods=n_events,
                n_windstorms=n_events,
                r=discount_rate,
                max_T=max_T,
                wealth=wealth,
                income=income,
                amenity_value=amenity_value,
                expected_damages_flood=expected_damages_flood,
                expected_damages_wind=expected_damages_wind,
            )
        else:
            raise ValueError(
                "Mode must be either 'flood' 'drought' 'windstorm' or 'multi_hazard'"
            )

        return NPV_summed

    def calcEU_do_nothing_drought(
        self,
        n_agents: int,
        risk_perception: np.ndarray,
        total_profits: np.ndarray,
        profits_no_event: np.ndarray,
        p_droughts: np.ndarray,
        T: np.ndarray,
        discount_rate: float,
        sigma: float,
        subjective: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Returns:
            EU_do_nothing_array: array containing the time discounted subjective utility of doing nothing for each agent.
        """
        # Ensure p floods is in increasing order
        indices = np.argsort(p_droughts)
        total_profits = total_profits[indices]
        p_droughts = np.sort(p_droughts)

        # Preallocate arrays
        n_floods, n_agents = total_profits.shape

        # Prepare arrays
        max_T = int(np.max(T))

        # Part njit, iterate through floods
        n_agents = int(n_agents)
        NPV_summed = self.IterateThroughEvents(
            n_events=n_floods,
            total_profits=total_profits,
            profits_no_event=profits_no_event,
            max_T=max_T,
            n_agents=n_agents,
            discount_rate=discount_rate,
            mode="drought",
        )

        # Filter out negative NPVs
        NPV_summed = np.maximum(1, NPV_summed)

        # Calculate expected utility
        ## NPV_Summed here is the wealth and income minus the expected damages of a certain probabilty event
        EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

        p_all_events = np.full((p_droughts.size + 3, n_agents), -1, dtype=np.float32)

        # calculate perceived risk
        perc_risk = p_droughts.repeat(n_agents).reshape(p_droughts.size, n_agents)

        # If
        if subjective:
            perc_risk *= risk_perception

        p_all_events[1:-2, :] = perc_risk

        # Cap percieved probability at 0.998. People cannot percieve any flood
        # event to occur more than once per year
        if np.max(p_all_events > 0.998):
            p_all_events[np.where(p_all_events > 0.998)] = 0.998

        # Add lasts p to complete x axis to 1 for trapezoid function (integrate
        # domain [0,1])
        p_all_events[-2, :] = p_all_events[-3, :] + 0.001
        p_all_events[-1, :] = 1

        # Add 0 to ensure we integrate [0, 1]
        p_all_events[0, :] = 0

        # Use composite trapezoidal rule integrate EU over event probability
        ## Here all
        y = EU_store
        x = p_all_events
        EU_do_nothing_array = np.trapezoid(y=y, x=x, axis=0)

        # People who already adapted cannot adapt, changed to a condition in the function that calls this, need the SEUT of doing nothing of those that have adapted
        # EU_do_nothing_array[np.where(adapted == 1)] = -np.inf

        return EU_do_nothing_array

    @staticmethod
    @njit(cache=True, parallel=True)
    def calcEU_adapt_drought_numba(
        expenditure_cap: np.ndarray,
        loan_duration: int,
        n_agents: int,
        sigma: np.ndarray,
        profits_no_event: np.ndarray,
        total_profits_adaptation: np.ndarray,
        profits_no_event_adaptation: np.ndarray,
        p_droughts: np.ndarray,
        risk_perception: np.ndarray,
        adaptation_costs: np.ndarray,
        total_annual_costs: np.ndarray,
        time_adapted: np.ndarray,
        adapted: np.ndarray,
        T: np.ndarray,
        discount_rate: np.ndarray,
        extra_constraint: np.ndarray,
        total_profits: np.ndarray,
    ) -> np.ndarray:
        """This function calculates the discounted subjective utility for adapting for each agent.

        We take into account the current adaptation status of each agent, so that agents also consider the number of years of remaining loan payment.

        Args:
            expenditure_cap: expenditure cap for dry flood proofing investments.
            loan_duration: loan duration of the dry flood proofing investment.
            n_agents: number of agents present in the current floodplain.
            sigma: array of risk aversion settings for the agents.
            profits_no_event: array containing the profits of each agent without any drought events.
            total_profits_adaptation: 2D array containing the total profits of adaptation for each drought event and agent.
            profits_no_event_adaptation: array containing the profits of each agent without any drought events after adaptation.
            p_droughts: array containing the probabilities of each drought event.
            risk_perception: array containing the risk perception of each agent.
            adaptation_costs: array of annual implementation costs of the adaptation.
            total_annual_costs: array containing total annual costs for each agent.
            time_adapted: array containing the time each agent has been paying off their adaptation loan.
            adapted: array containing the adaptation status of each agent (1 = adapted, 0 = not adapted).
            T: array containing the decision horizon of each agent.
            discount_rate: array of time discounting factors for each agent.
            extra_constraint: array of boolean values representing extra constraints for each agent.
            total_profits: array containing total profits for each agent.

        Returns:
            EU_adapt: array containing the time-discounted subjective utility of adapting for each agent.
        """
        # Preallocate arrays
        EU_adapt = np.full(n_agents, -np.inf, dtype=np.float32)

        # Ensure p_droughts is in increasing order
        indices = np.argsort(p_droughts)
        total_profits_adaptation = total_profits_adaptation[indices]
        p_droughts = p_droughts[indices]

        # Identify agents able to afford the adaptation and that have not yet adapted
        unconstrained_mask = (expenditure_cap) & (~adapted) & (extra_constraint)

        # Iterate only through agents who can afford to adapt
        unconstrained_indices = np.where(unconstrained_mask)[0]

        for idx in prange(unconstrained_indices.size):  # ty: ignore[not-iterable]
            i = unconstrained_indices[idx]

            # Loan payment years remaining
            payment_remainder = max(loan_duration - time_adapted[i], 0)

            # Decision horizon for the agent
            T_i = T[i]
            t_agent = np.arange(T_i, dtype=np.int32)

            # NPV under no drought event after adaptation
            NPV_adapt_no_flood = np.full(
                T_i, profits_no_event_adaptation[i], dtype=np.float32
            )

            # Subtract adaptation costs during payment period
            for t in range(T_i):
                if t < payment_remainder:
                    NPV_adapt_no_flood[t] -= adaptation_costs[i]

            # Ensure NPVs are at least a small positive number to prevent NaNs
            NPV_adapt_no_flood = np.maximum(NPV_adapt_no_flood, 1e-6)

            # Time-discounted NPVs
            discount_factors = (1 + discount_rate[i]) ** t_agent
            NPV_adapt_no_flood_discounted = NPV_adapt_no_flood / discount_factors
            NPV_adapt_no_flood_summed = np.sum(NPV_adapt_no_flood_discounted)

            # Apply utility function to NPVs
            NPV_adapt_no_flood_summed = max(
                NPV_adapt_no_flood_summed, 1e-6
            )  # Ensure positive
            EU_adapt_no_flood = (NPV_adapt_no_flood_summed ** (1 - sigma[i])) / (
                1 - sigma[i]
            )

            # Calculate NPVs for each drought event
            n_events = p_droughts.size
            NPV_adapt = np.empty((n_events, T_i), dtype=np.float32)
            total_profits_adaptation_i = total_profits_adaptation[:, i]

            for j in range(n_events):
                NPV_event = np.full(
                    T_i, total_profits_adaptation_i[j], dtype=np.float32
                )
                for t in range(T_i):
                    if t < payment_remainder:
                        NPV_event[t] -= adaptation_costs[i]
                NPV_event = np.maximum(NPV_event, 1e-6)  # Ensure positive
                NPV_event_discounted = NPV_event / discount_factors
                NPV_adapt[j, :] = NPV_event_discounted

            NPV_adapt_summed = np.sum(NPV_adapt, axis=1)
            NPV_adapt_summed = np.maximum(NPV_adapt_summed, 1e-6)  # Ensure positive

            # Calculate expected utilities for each event
            EU_adapt_flood = (NPV_adapt_summed ** (1 - sigma[i])) / (1 - sigma[i])

            # Prepare arrays for integration
            EU_adapt_dict = np.zeros(n_events + 3, dtype=np.float32)
            p_all_events = np.zeros(n_events + 3, dtype=np.float32)

            EU_adapt_dict[1 : n_events + 1] = EU_adapt_flood
            EU_adapt_dict[n_events + 1 : n_events + 3] = EU_adapt_no_flood
            EU_adapt_dict[0] = EU_adapt_flood[0]

            # Adjust for perceived risk
            p_all_events[1 : n_events + 1] = risk_perception[i] * p_droughts
            p_all_events[n_events + 1] = (
                p_all_events[n_events] + 0.001
            )  # Small increment
            p_all_events[n_events + 2] = 1.0  # Ensure total probability sums to 1
            p_all_events[0] = 0.0

            # Integrate EU over probabilities using trapezoidal rule
            EU_adapt[i] = np.trapezoid(EU_adapt_dict, p_all_events)

            # Ensure no NaNs
            if np.isnan(EU_adapt[i]):
                EU_adapt[i] = -np.inf

        return EU_adapt

    def calcEU_adapt_drought(self, **kwargs: Any) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Returns:
            EU_do_nothing_array: array containing the time discounted subjective utility of doing nothing for each agent.
        """
        assert kwargs["adapted"].dtype == bool
        return self.calcEU_adapt_drought_numba(**kwargs)

    def calcEU_adapt_flood(
        self,
        geom_id: int | str,
        n_agents: int,
        wealth: np.ndarray,
        income: np.ndarray,
        expendature_cap: float,
        amenity_value: np.ndarray,
        amenity_weight: float | int,
        risk_perception: np.ndarray,
        expected_damages_adapt: np.ndarray,
        adaptation_costs: np.ndarray,
        time_adapted: np.ndarray,
        loan_duration: int,
        p_floods: np.ndarray,
        T: np.ndarray | int | float,
        r: float,
        sigma: float,
        **kwargs: dict,
    ) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Returns:
            EU_do_nothing_array: array containing the time discounted subjective utility of doing nothing for each agent.
        """
        # weigh amenities
        amenity_value = amenity_value * amenity_weight

        # Ensure p floods is in increasing order
        indices = np.argsort(p_floods)
        expected_damages_adapt = expected_damages_adapt[indices]
        p_floods = np.sort(p_floods)

        # Preallocate arrays
        n_floods, n_agents = expected_damages_adapt.shape
        p_all_events = np.full((p_floods.size + 3, n_agents), -1, dtype=np.float32)

        # calculate perceived risk
        perc_risk = p_floods.repeat(n_agents).reshape(p_floods.size, n_agents)
        perc_risk *= risk_perception
        p_all_events[1:-2, :] = perc_risk

        # Cap percieved probability at 0.998. People cannot percieve any flood
        # event to occur more than once per year
        if np.max(p_all_events > 0.998):
            p_all_events[np.where(p_all_events > 0.998)] = 0.998

        # Add lasts p to complete x axis to 1 for trapezoid function (integrate
        # domain [0,1])
        p_all_events[-2, :] = p_all_events[-3, :] + 0.001
        p_all_events[-1, :] = 1

        # Add 0 to ensure we integrate [0, 1]
        p_all_events[0, :] = 0

        # Prepare arrays
        max_T = int(np.max(T))

        # Part njit, iterate through events
        n_agents = int(n_agents)
        NPV_summed = self.IterateThroughEvents(
            n_events=n_floods,
            n_agents=n_agents,
            discount_rate=r,
            wealth=wealth,
            income=income,
            amenity_value=amenity_value,
            max_T=max_T,
            expected_damages=expected_damages_adapt,
            mode="flood",
        )

        # some usefull attributes for cost calculations
        # get time discounted adaptation cost for each agent based on time since adaptation
        discounts = 1 / (1 + r) ** np.arange(loan_duration)

        # create cost array for years payment left
        years = np.arange(loan_duration + 1, dtype=np.int32)
        cost_array = np.full((n_agents, years.size), -1, np.float32)
        for i, year in enumerate(years):
            cost_array[:, i] = np.sum(discounts[:year]) * adaptation_costs

        loan_left = (loan_duration - time_adapted).astype(np.int32)
        loan_left = np.maximum(loan_left, 0)  # cap duration at 0
        loan_left = np.minimum(loan_left, T)  # cap duration at decision horizon

        # take the calculated adaptation cost based on loan duration left for agents
        axis_0 = np.array(np.arange(n_agents))
        # create index to sample from cost array
        time_discounted_adaptation_cost = cost_array[axis_0, loan_left]

        # subtract from NPV array
        NPV_summed -= time_discounted_adaptation_cost

        # Filter out negative NPVs
        NPV_summed = np.maximum(1, NPV_summed)
        if (NPV_summed == 1).any():
            n_negative = np.sum(NPV_summed == 1)
            print(
                f"[calcEU_adapt] Warning, {n_negative} negative NPVs encountered in {geom_id} ({np.round(n_negative / NPV_summed.size * 100, 2)}%)"
            )

        # Calculate expected utility
        if sigma == 1:
            EU_store = np.log(NPV_summed)
        else:
            EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

        # Use composite trapezoidal rule integrate EU over event probability
        y = EU_store
        x = p_all_events
        EU_adapt_array = np.trapezoid(y=y, x=x, axis=0)

        # set EU of adapt to -np.inf for thuse unable to afford it [maybe move this earlier to reduce array sizes]
        constrained = np.where(income * expendature_cap <= adaptation_costs)
        EU_adapt_array[constrained] = -np.inf
        # EU_adapt_array *= self.error_terms_stay
        return EU_adapt_array

    def calcEU_do_nothing_flood(
        self,
        geom_id: str | int,
        n_agents: int,
        wealth: np.ndarray,
        income: np.ndarray,
        amenity_value: np.ndarray,
        amenity_weight: np.ndarray | float,
        risk_perception: np.ndarray,
        expected_damages: np.ndarray,
        adapted: np.ndarray,
        p_floods: np.ndarray,
        T: np.ndarray | float | int,
        r: float,
        sigma: float,
        **kwargs: dict,
    ) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Returns:
            EU_do_nothing_array: array containing the time discounted subjective utility of doing nothing for each agent.
        """
        # weigh amenities
        amenity_value = amenity_value * amenity_weight

        # Ensure p floods is in increasing order
        indices = np.argsort(p_floods)
        expected_damages = expected_damages[indices]
        p_floods = np.sort(p_floods)

        # Preallocate arrays
        n_floods, n_agents = expected_damages.shape
        p_all_events = np.full((p_floods.size + 3, n_agents), -1, dtype=np.float32)

        # calculate perceived risk
        perc_risk = p_floods.repeat(n_agents).reshape(p_floods.size, n_agents)
        perc_risk *= risk_perception
        p_all_events[1:-2, :] = perc_risk

        # Cap percieved probability at 0.998. People cannot percieve any flood
        # event to occur more than once per year
        if np.max(p_all_events > 0.998):
            p_all_events[np.where(p_all_events > 0.998)] = 0.998

        # Add lasts p to complete x axis to 1 for trapezoid function (integrate
        # domain [0,1])
        p_all_events[-2, :] = p_all_events[-3, :] + 0.001
        p_all_events[-1, :] = 1

        # Add 0 to ensure we integrate [0, 1]
        p_all_events[0, :] = 0

        # Prepare arrays
        max_T = int(np.max(T))

        # Part njit, iterate through events
        n_agents = int(n_agents)
        NPV_summed = self.IterateThroughEvents(
            n_events=n_floods,
            n_agents=n_agents,
            discount_rate=r,
            wealth=wealth,
            income=income,
            amenity_value=amenity_value,
            max_T=max_T,
            expected_damages=expected_damages,
            mode="flood",
        )

        # Filter out negative NPVs
        NPV_summed = np.maximum(1, NPV_summed)

        if (NPV_summed == 1).any():
            n_negative = np.sum(NPV_summed == 1)
            print(
                f"[calcEU_do_nothing] Warning, {n_negative} negative NPVs encountered in {geom_id} ({np.round(n_negative / NPV_summed.size * 100, 2)}%)"
            )

        # Calculate expected utility
        if sigma == 1:
            EU_store = np.log(NPV_summed)
        else:
            EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

        # Use composite trapezoidal rule integrate EU over event probability
        y = EU_store
        x = p_all_events
        EU_do_nothing_array = np.trapezoid(y=y, x=x, axis=0)

        # People who already adapted cannot not adapt
        adapted_mask = np.asarray(adapted).astype(bool)
        EU_do_nothing_array[adapted_mask] = -np.inf

        return EU_do_nothing_array

    def calcEU_shutters_windstorm(
        self,
        geom_id,
        n_agents: int,
        wealth: np.ndarray,
        income: np.ndarray,
        expendature_cap,
        amenity_value: np.ndarray,
        amenity_weight,
        risk_perception: np.ndarray,
        expected_damages_adapt: np.ndarray,
        adaptation_costs: np.ndarray,
        time_adapted,
        loan_duration,
        p_windstorm: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: float,
        **kwargs,
    ) -> np.ndarray:
        """_This function calculates the subjective utility of implementing window shutters.

        Returns:
            EU_shutters_array: array containing the time discounted subjective utility of installing window shutters for each agent.
        """
        # weight amenities
        amenity_value = amenity_value * amenity_weight

        # ensure probabilites are in increasing order
        indices = np.argsort(p_windstorm)
        expected_damages_adapt = expected_damages_adapt[indices]
        p_windstorm = np.sort(p_windstorm)

        # Preallocate arrays
        n_windstorms, n_agents = expected_damages_adapt.shape
        p_all_windstorms = np.full(
            (p_windstorm.size + 3, n_agents), -1, dtype=np.float32
        )

        # calculate perceived risk
        perc_risk = p_windstorm.repeat(n_agents).reshape(p_windstorm.size, n_agents)
        perc_risk *= risk_perception
        p_all_windstorms[1:-2, :] = perc_risk

        # Cap perceived probability at 0.998. People cannot percieve any flood event
        # to occur more than once per year
        if np.max(p_all_windstorms > 0.998):
            p_all_windstorms[np.where(p_all_windstorms > 0.998)] = 0.998

        # Add lasts p to complete x axis to 1 for trapezoid function
        p_all_windstorms[-2, :] = p_all_windstorms[-3, :] + 0.001
        p_all_windstorms[-1, :] = 1
        p_all_windstorms[0, :] = 0

        # Prepare arrays
        max_T = np.int32(np.max(T))

        n_agents = np.int32(n_agents)
        NPV_summed = self.IterateThroughEvents(
            n_events=n_windstorms,
            n_agents=n_agents,
            discount_rate=r,
            wealth=wealth,
            income=income,
            amenity_value=amenity_value,
            max_T=max_T,
            expected_damages=expected_damages_adapt,
            mode="windstorm",
        )

        # ---- Adaptation cost logic ----
        # For testing we assume a one-time payment with no loan for shutters installation
        discounts = 1 / (1 + r) ** np.arange(1)  # only year 0
        years = np.arange(1, dtype=np.int32)
        cost_array = np.full((n_agents, years.size), -1, np.float32)
        for i, year in enumerate(years):
            cost_array[:, i] = np.sum(discounts[: year + 1]) * adaptation_costs

        # right now we are considering no loan but this could be change/adapted in the future
        loan_left = (loan_duration - time_adapted).astype(np.int32)  # 0 for no loan
        loan_left = np.maximum(loan_left, 0)
        loan_left - np.minimum(loan_left, T)

        axis_0 = np.array(np.arange(n_agents))
        time_discounted_adaptation_cost = adaptation_costs  # one-time payment

        # substract from NPV array
        NPV_summed -= time_discounted_adaptation_cost

        # Filter negative NPVs
        NPV_summed = np.maximum(1, NPV_summed)
        if (NPV_summed == 1).any():
            n_negative = np.sum(NPV_summed == 1)
            print(
                f"[CalcEU_adapt_windsotrm] Warning, {n_negative} negative NPVs in {geom_id}"
            )

        # Calculate expected utility
        if sigma == 1:
            EU_store = np.log(NPV_summed)
        else:
            EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

        # Use composite trapezoidal rule to integrate EU over event probability
        y = EU_store
        x = p_all_windstorms
        EU_shutters_array = np.trapezoid(y=y, x=x, axis=0)

        # Constrained affordability -> set EU of adapt to -np.inf for those unable to afford it
        constrained = np.where(income * expendature_cap <= adaptation_costs)
        EU_shutters_array[constrained] = -np.inf

        return EU_shutters_array

    def calcEU_do_nothing_w(
        self,
        geom_id,
        n_agents: int,
        wealth: np.ndarray,
        income: np.ndarray,
        amenity_value: np.ndarray,
        amenity_weight,
        risk_perception: np.ndarray,
        expected_damages: np.ndarray,
        adapted: np.ndarray,
        p_windstorm: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: float,
        **kwargs,
    ) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Returns:
            EU_do_nothing_array: array containing the time discounted subjective utility of doing nothing for each agent.
        """
        # weigh amenities
        amenity_value = amenity_value * amenity_weight

        # Ensure p floods is in increasing order
        indices = np.argsort(p_windstorm)
        expected_damages = expected_damages[indices]
        p_windstorm = np.sort(p_windstorm)

        # Preallocate arrays
        n_windstorm, n_agents = expected_damages.shape
        p_all_events = np.full((p_windstorm.size + 3, n_agents), -1, dtype=np.float32)

        # calculate perceived risk
        perc_risk = p_windstorm.repeat(n_agents).reshape(p_windstorm.size, n_agents)
        perc_risk *= risk_perception
        p_all_events[1:-2, :] = perc_risk

        # Cap percieved probability at 0.998. People cannot percieve any flood
        # event to occur more than once per year
        if np.max(p_all_events > 0.998):
            p_all_events[np.where(p_all_events > 0.998)] = 0.998

        # Add lasts p to complete x axis to 1 for trapezoid function (integrate
        # domain [0,1])
        p_all_events[-2, :] = p_all_events[-3, :] + 0.001
        p_all_events[-1, :] = 1

        # Add 0 to ensure we integrate [0, 1]
        p_all_events[0, :] = 0

        # Prepare arrays
        max_T = np.int32(np.max(T))

        n_agents = np.int32(n_agents)
        NPV_summed = self.IterateThroughEvents(
            n_events=n_windstorm,
            n_agents=n_agents,
            discount_rate=r,
            wealth=wealth,
            income=income,
            amenity_value=amenity_value,
            max_T=max_T,
            expected_damages=expected_damages,
            mode="windstorm",
        )

        # Filter out negative NPVs
        NPV_summed = np.maximum(1, NPV_summed)

        if (NPV_summed == 1).any():
            n_negative = np.sum(NPV_summed == 1)
            print(
                f"[calcEU_do_nothing_w] Warning, {n_negative} negative NPVs encountered in {geom_id} ({np.round(n_negative / NPV_summed.size * 100, 2)}%)"
            )

        # Calculate expected utility
        if sigma == 1:
            EU_store = np.log(NPV_summed)
        else:
            EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

        # Use composite trapezoidal rule integrate EU over event probability
        y = EU_store
        x = p_all_events
        EU_do_nothing_w_array = np.trapezoid(y=y, x=x, axis=0)

        # People who already adapted cannot not adapt
        # adapted_mask = np.asarray(adapted).astype(bool)
        EU_do_nothing_w_array[np.where(adapted == 1)] = -np.inf

        # EU_do_nothing_array *= self.error_terms_stay
        return EU_do_nothing_w_array

    # def calcEU_do_nothing_multirisk(
    #     self,
    #     n_agents: int,
    #     wealth: np.ndarray,
    #     income: np.ndarray,
    #     expenditure_cap: float,
    #     amenity_value: np.ndarray,
    #     amenity_weight: float,
    #     risk_perception: np.ndarray,
    #     expected_damages_flood: np.ndarray,
    #     expected_damages_wind: np.ndarray,
    #     p_flood: np.ndarray,
    #     p_wind: np.ndarray,
    #     T: np.ndarray,
    #     r: float,
    #     sigma: float,
    #     operating_insurer: float = 0.3,
    #     public_reinsurer: float = 0.0,
    # ):
    #     """
    #     Expected Utility for households choosing not to adapt but potentially buying multirisk insurance.

    #     Args:
    #         n_agents (int): household parameters
    #         wealth (np.ndarray): household paramters
    #         income (np.ndarray): household parameters
    #         expenditure_cap (float): household parameters
    #         amenity_value (np.ndarray): household parameters
    #         amenity_weight (float): household parameters
    #         risk_perception (np.ndarray): subjective perception of risk per household
    #         expected_damages_flood (np.ndarray): arrays of expected damges per flood event per household
    #         expected_damages_wind (np.ndarray): arrays of expected damges per wind event per household
    #         p_flood (np.ndarray): exceedance probabilities of flood events
    #         p_wind (np.ndarray): exceedance probabilities of wind events
    #         T (np.ndarray): decision horizon
    #         r (float): discount rate
    #         sigma (float): risk aversion parameter
    #         operating_insurer (float, optional): insurer margin/cost
    #         public_reinsurer (float, optional): fraction of damage covered by public scheme

    #         Returns:
    #         EU_do_nothing_multirisk (np.ndarray): expected utility of not adapting but potentially buying insurance
    #     """
    #     premium, premium_public, premium_private = self.Insurance_premium_PPP(
    #         expected_damages_flood=expected_damages_flood,
    #         p_flood=p_flood,
    #         expected_damages_wind=expected_damages_wind,
    #         p_wind=p_wind,
    #         operating_insurer=operating_insurer,
    #         method="household",
    #         public_reinsurer=public_reinsurer,
    #     )
    #     # # Integrate subjective risk perception
    #     # p_flood_adjusted = np.clip(
    #     #     p_flood * risk_perception, 0, 0.998
    #     # )  # what does the 0.998 do here?
    #     # p_wind_adjusted = np.clip(p_wind * risk_perception, 0, 0.998)

    #     # Compute dicsounted costs of premium over household decision horizon
    #     max_T = np.max(T)
    #     discounts = 1.0 / (1.0 + r) ** np.arange(1, max_T + 1)
    #     cost_array = np.full((max_T + 1,), -1, dtype=np.float32)
    #     for i, year in enumerate(range(max_T + 1)):
    #         cost_array[i] = np.sum(discounts[:year] * premium.mean())

    #     # Substract discounted premium from wealth + amenity (why not income?)
    #     amenity_adjusted = amenity_value * amenity_weight
    #     NPV = wealth + amenity_adjusted - np.take(cost_array, T)

    #     # Ensure NPV is positive for utility calculation
    #     NPV = np.maximum(1, NPV)

    #     # Apply CRRA utility (what is CRRA?)
    #     if sigma == 1:
    #         EU_array = np.log(NPV)
    #     else:
    #         EU_array = (NPV ** (1 - sigma)) / (1 - sigma)

    #     # Constraints: if household cannot afford premium, set EU to -inf
    #     constrained = np.where(income * expenditure_cap <= premium)
    #     EU_array[constrained] = -np.inf

    #     return EU_array, premium, premium_private, premium_public

    # def calcEU_no_insure(
    #     self,
    #     n_agents: int,
    #     wealth: np.ndarray,
    #     income: np.ndarray,
    #     expenditure_cap: float,
    #     amenity_value: np.ndarray,
    #     amenity_weight: float,
    #     risk_perception: np.ndarray,
    #     expected_damages_flood: np.ndarray,
    #     expected_damages_wind: np.ndarray,
    #     p_flood: np.ndarray,
    #     p_wind: np.ndarray,
    #     T: np.ndarray,
    #     r: float,
    #     sigma: float,
    #     **kwargs,
    # ):
    #     """
    #     Expected Utility for households choosing not to buy insurance.

    #     Args:
    #         n_agents (int): household parameters
    #         wealth (np.ndarray): household paramters
    #         income (np.ndarray): household parameters
    #         expenditure_cap (float): household parameters
    #         amenity_value (np.ndarray): household parameters
    #         amenity_weight (float): household parameters
    #         risk_perception (np.ndarray): subjective perception of risk per household
    #         expected_damages_flood (np.ndarray): arrays of expected damges per flood event per household
    #         expected_damages_wind (np.ndarray): arrays of expected damges per wind event per household
    #         p_flood (np.ndarray): exceedance probabilities of flood events
    #         p_wind (np.ndarray): exceedance probabilities of wind events
    #         T (np.ndarray): decision horizon
    #         r (float): discount rate
    #         sigma (float): risk aversion parameter
    #         Returns:
    #         EU_no_insure_array (np.ndarray): expected utility of not buying insurance
    #     """
    #     # Weight amenities
    #     amenity_value = amenity_value * amenity_weight

    #     # Ensure probability arrays are 1-D float
    #     p_flood = np.asarray(p_flood, dtype=np.float32).reshape(-1)
    #     p_wind = np.asarray(p_wind, dtype=np.float32).reshape(-1)

    #     flood_order = np.argsort(p_flood)
    #     wind_order = np.argsort(p_wind)

    #     p_flood = p_flood[flood_order]
    #     p_wind = p_wind[wind_order]

    #     damage_flood = expected_damages_flood[flood_order, :].astype(np.float32)
    #     damage_wind = expected_damages_wind[wind_order, :].astype(np.float32)

    #     n_f = int(p_flood.size)
    #     n_w = int(p_wind.size)
    #     n_common = min(n_f, n_w)

    #     p_flood = p_flood[:n_common]
    #     p_wind = p_wind[:n_common]
    #     damage_flood = damage_flood[:n_common, :]
    #     damage_wind = damage_wind[:n_common, :]

    #     # Combine events into one list (floods then windstorms)
    #     p_all = np.concatenate([p_flood, p_wind], axis=0)
    #     n_events_total = p_all.size
    #     p_all_events = np.full((n_events_total + 3, n_agents), -1, dtype=np.float32)

    #     # perceived risk per event per agent
    #     risk_perception_matrix = np.asarray(risk_perception, dtype=np.float32).reshape(
    #         -1
    #     )
    #     if risk_perception_matrix.shape != (n_agents,):
    #         raise AssertionError(
    #             "risk_perception must be of shape (n_agents,), got {risk_perception_matrix.shape}"
    #         )

    #     perc_risk = (
    #         p_all.repeat(n_agents).reshape(n_events_total, n_agents)
    #         * risk_perception_matrix
    #     )
    #     perc_risk = np.minimum(perc_risk, np.float32(0.998))  # Cap at 0.998
    #     p_all_events[1:-2, :] = perc_risk

    #     # Complete x-axis for trapezoid integration
    #     p_all_events[-2, :] = p_all_events[-3, :] + np.float32(0.001)
    #     p_all_events[-1, :] = np.float32(1.0)
    #     p_all_events[0, :] = np.float32(0.0)

    #     # Iterate trhough events to get NPV sums
    #     max_T = int(np.max(T))
    #     NPV_summed = self.IterateThroughEvents(
    #         n_events=n_common,
    #         n_agents=n_agents,
    #         discount_rate=r,
    #         wealth=wealth,
    #         income=income,
    #         amenity_value=amenity_value,
    #         max_T=max_T,
    #         expected_damages_flood=damage_flood,
    #         expected_damages_wind=damage_wind,
    #         mode="multi_hazard",
    #     )

    #     NPV_summed = np.maximum(np.float32(1.0), NPV_summed)
    #     if sigma == 1:
    #         EU_store = np.log(NPV_summed)
    #     else:
    #         EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

    #     x = np.full((n_common + 3, n_agents), -1, dtype=np.float32)
    #     perc_risk_single = (
    #         p_flood.repeat(n_agents).reshape(n_common, n_agents)
    #         * risk_perception_matrix
    #     )
    #     perc_risk_single = np.minimum(
    #         perc_risk_single, np.float32(0.998)
    #     )  # Cap at 0.998
    #     x[1:-2, :] = perc_risk_single
    #     x[-2, :] = x[-3, :] + np.float32(0.001)
    #     x[-1, :] = np.float32(1.0)
    #     x[0, :] = np.float32(0.0)

    #     EU_do_nothing_multirisk = np.trapezoid(y=EU_store, x=x, axis=0)

    #     return EU_do_nothing_multirisk
    def calcEU_no_insure(
        self,
        n_agents: int,
        wealth: np.ndarray,
        income: np.ndarray,
        expenditure_cap: float,
        amenity_value: np.ndarray,
        amenity_weight: float,
        risk_perception: np.ndarray,
        expected_damages_flood: np.ndarray,
        expected_damages_wind: np.ndarray,
        p_flood: np.ndarray,
        p_wind: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: float,
        **kwargs,
    ):
        """
        Expected Utility for households choosing not to buy insurance (multi-hazard baseline),
        built from the existing single-hazard do-nothing EU functions (which use IterateThroughEvents).

        This avoids IterateThroughEvents(mode="multi_hazard"), which is currently unsafe (Numba crash).
        """
        # 1) compute single-hazard EUs using existing functions (these use IterateThroughEvents internally)
        EU_flood = self.calcEU_do_nothing_flood(
            geom_id=kwargs.get("geom_id", "NoID"),
            n_agents=n_agents,
            wealth=wealth,
            income=income,
            amenity_value=amenity_value,
            amenity_weight=amenity_weight,
            risk_perception=risk_perception,
            expected_damages=expected_damages_flood,
            adapted=np.zeros(n_agents, dtype=bool),
            p_floods=p_flood,
            T=T,
            r=r,
            sigma=sigma,
        )

        EU_wind = self.calcEU_do_nothing_w(
            geom_id=kwargs.get("geom_id", "NoID"),
            n_agents=n_agents,
            wealth=wealth,
            income=income,
            expendature_cap=expenditure_cap,
            amenity_value=amenity_value,
            amenity_weight=amenity_weight,
            risk_perception=risk_perception,
            expected_damages=expected_damages_wind,
            adapted=np.zeros(n_agents, dtype=bool),
            p_windstorm=p_wind,
            T=T,
            r=r,
            sigma=sigma,
        )

        # 2) compute U0 (no-event utility) in the same wealth convention
        amenity_adjusted = np.asarray(amenity_value, dtype=np.float32) * np.float32(
            amenity_weight
        )
        W0 = np.maximum(
            np.float32(1.0),
            wealth.astype(np.float32) + income.astype(np.float32) + amenity_adjusted,
        )

        if sigma == 1:
            U0 = np.log(W0)
        else:
            s = np.float32(sigma)
            U0 = (W0 ** (1 - s)) / (1 - s)

        # 3) perceived total hazard probabilities (consistent with other code: cap at 0.998 per event)
        rp = np.asarray(risk_perception, dtype=np.float32).reshape(-1)
        p_f = np.asarray(p_flood, dtype=np.float32).reshape(-1)
        p_w = np.asarray(p_wind, dtype=np.float32).reshape(-1)

        P_f = np.sum(np.minimum(p_f[:, None] * rp[None, :], np.float32(0.998)), axis=0)
        P_w = np.sum(np.minimum(p_w[:, None] * rp[None, :], np.float32(0.998)), axis=0)
        P_none = np.maximum(np.float32(0.0), np.float32(1.0) - P_f - P_w)

        eps = np.float32(1e-8)

        # 4) back out "event-conditional EU" for each hazard from the single-hazard mixture
        EU_flood_event = (EU_flood - (np.float32(1.0) - P_f) * U0) / np.maximum(
            P_f, eps
        )
        EU_wind_event = (EU_wind - (np.float32(1.0) - P_w) * U0) / np.maximum(P_w, eps)

        # 5) combine into one multi-hazard EU without double counting no-event utility
        EU_no_insure_array = P_f * EU_flood_event + P_w * EU_wind_event + P_none * U0

        return np.asarray(EU_no_insure_array, dtype=np.float32).reshape(-1)

    # def calcEU_insure_multirisk_residual_CATNAT(
    #     self,
    #     geom_id,
    #     n_agents: int,
    #     wealth: np.ndarray,
    #     income: np.ndarray,
    #     expenditure_cap: float,
    #     amenity_value: np.ndarray,
    #     amenity_weight: float,
    #     risk_perception: np.ndarray,
    #     expected_damages_flood: np.ndarray,
    #     expected_damages_floodadapted: np.ndarray,
    #     expected_damages_wind: np.ndarray,
    #     expected_damages_windadapted: np.ndarray,
    #     p_flood: np.ndarray,
    #     p_wind: np.ndarray,
    #     time_adapted,
    #     loan_duration,
    #     T: np.ndarray,
    #     r: float,
    #     sigma: float,
    #     deductible: float = 0.1,
    #     operating_insurer: float = 0.3,
    #     public_reinsurer: float = 0.0,
    #     adapted_floodproofing: np.ndarray = None,
    #     adapted_windshutters: np.ndarray = None,
    #     **kwargs,
    # ):
    #     """
    #     Expected Utility for households choosing to buy multi-hazard insurance with residual damages after adaptation.

    #     Args:
    #         geom_id: geographic identifier
    #         n_agents (int): household parameters
    #         wealth (np.ndarray): household paramters
    #         income (np.ndarray): household parameters
    #         expenditure_cap (float): household parameters
    #         amenity_value (np.ndarray): household parameters
    #         amenity_weight (float): household parameters
    #         risk_perception (np.ndarray): subjective perception of risk per household
    #         expected_damages_flood (np.ndarray): arrays of expected damges per flood event
    #         expected_damages_floodadapted (np.ndarray): arrays of expected damges per flood event after adaptation
    #         expected_damages_wind (np.ndarray): arrays of expected damges per wind event
    #         expected_damages_windadapted (np.ndarray): arrays of expected damges per wind event after adaptation
    #         p_flood (np.ndarray): exceedance probabilities of flood events
    #         p_wind (np.ndarray): exceedance probabilities of wind events
    #         time_adapted: time since adaptation was implemented
    #         loan_duration: duration of loan taken to finance adaptation
    #         T (np.ndarray): decision horizon
    #         r (float): discount rate
    #         sigma (float): risk aversion parameter
    #         deductible (float, optional): insurance deductible (fraction of damage not covered by insurance)
    #         operating_insurer (float, optional): insurer margin/cost
    #         public_reinsurer (float, optional): fraction of damage covered by public scheme
    #         adapted_floodproofing (np.ndarray, optional): array indicating which agents have floodproofing adaptation
    #         adapted_windshutters (np.ndarray, optional): array indicating which agents have windshutter adaptation

    #         Returns:
    #         EU_insure_array (np.ndarray): expected utility of buying multirisk insurance
    #         premium (np.ndarray): insurance premium per household
    #         premium_public (np.ndarray): public share of insurance premium per household
    #         premium_private (np.ndarray): private share of insurance premium per household
    #     """
    #     if adapted_floodproofing is not None:
    #         idx_flood = np.where(adapted_floodproofing == 1)[0]
    #         expected_damages_flood[:, idx_flood] = expected_damages_floodadapted[
    #             :, idx_flood
    #         ]

    #     if adapted_windshutters is not None:
    #         idx_wind = np.where(adapted_windshutters == 1)[0]
    #         expected_damages_wind[:, idx_wind] = expected_damages_windadapted[
    #             :, idx_wind
    #         ]

    #     expected_damages_flood_residual = expected_damages_flood * deductible
    #     expected_damages_wind_residual = expected_damages_wind * deductible

    #     expected_flood_insured = expected_damages_flood * (1 - deductible)
    #     expected_wind_insured = expected_damages_wind * (1 - deductible)

    #     # Sort damages by probability
    #     flood_order = np.argsort(p_flood)
    #     wind_order = np.argsort(p_wind)
    #     expected_damages_flood_residual = expected_damages_flood_residual[
    #         flood_order, :
    #     ]
    #     expected_damages_wind_residual = expected_damages_wind_residual[wind_order, :]
    #     p_flood_sorted = np.sort(p_flood)
    #     p_wind_sorted = np.sort(p_wind)

    #     # Compute premiums with PPP (check order of arrays)
    #     # premium, premium_public, premium_private = self.Insurance_premium_PPP(
    #     #     expected_damages_flood=expected_flood_insured,
    #     #     p_flood=p_flood_sorted,
    #     #     expected_damages_wind=expected_wind_insured,
    #     #     p_wind=p_wind_sorted,
    #     #     operating_insurer=operating_insurer,
    #     #     public_reinsurer=public_reinsurer,
    #     # )

    #     premium, premium_reinsured, premium_private = self.Insurance_premium_CATNAT(
    #         expected_damages_flood=expected_flood_insured,
    #         p_flood=p_flood_sorted,
    #         expected_damages_wind=expected_wind_insured,
    #         p_wind=p_wind_sorted,
    #         operating_insurer=operating_insurer,
    #     )

    #     # Compute NPV of wealth minus discounted premiums over decision horizon
    #     discounts = 1.0 / (1.0 + r) ** np.arange(1, np.max(T) + 1)
    #     cost_array = np.full(np.max(T) + 1, -1, dtype=np.float32)
    #     for i, year in enumerate(range(np.max(T) + 1)):
    #         cost_array[year] = np.sum(discounts[:year] * premium.mean())

    #     amenity_adjusted = amenity_value * amenity_weight
    #     NPV = wealth + amenity_adjusted - np.take(cost_array, T)

    #     NPV = np.maximum(1, NPV)
    #     if sigma == 1:
    #         EU_insure_array = np.log(NPV)
    #     else:
    #         EU_insure_array = (NPV ** (1 - sigma)) / (1 - sigma)

    #     # Constraint: cannot afford premium
    #     constrained = np.where(income * expenditure_cap <= premium)
    #     EU_insure_array[constrained] = -np.inf

    #     return (
    #         EU_insure_array,
    #         premium,
    #         premium_private,
    #         premium_reinsured,
    #     )  # premium_public

    def calcEU_insure_multirisk_residual_CATNAT(
        self,
        geom_id,
        n_agents: int,
        wealth: np.ndarray,
        income: np.ndarray,
        expenditure_cap: float,
        amenity_value: np.ndarray,
        amenity_weight: float,
        risk_perception: np.ndarray,
        expected_damages_flood: np.ndarray,
        expected_damages_floodadapted: np.ndarray,
        expected_damages_wind: np.ndarray,
        expected_damages_windadapted: np.ndarray,
        p_flood: np.ndarray,
        p_wind: np.ndarray,
        time_adapted,
        loan_duration,
        T: np.ndarray,
        r: float,
        sigma: float,
        deductible: float = 0.1,
        operating_insurer: float = 0.3,
        public_reinsurer: float = 0.5,
        adapted_floodproofing: np.ndarray = None,
        adapted_windshutters: np.ndarray = None,
        **kwargs,
    ):
        """
        Expected Utility for households choosing to buy multi-hazard insurance with residual damages after adaptation.

        Args:
            geom_id (_type_): _description_
            n_agents (int): _description_
            wealth (np.ndarray): _description_
            income (np.ndarray): _description_
            expenditure_cap (float): _description_
            amenity_value (np.ndarray): _description_
            amenity_weight (float): _description_
            risk_perception (np.ndarray): _description_
            expected_damages_flood (np.ndarray): _description_
            expected_damages_floodadapted (np.ndarray): _description_
            expected_damages_wind (np.ndarray): _description_
            expected_damages_windadapted (np.ndarray): _description_
            p_flood (np.ndarray): _description_
            p_wind (np.ndarray): _description_
            time_adapted (_type_): _description_
            loan_duration (_type_): _description_
            T (np.ndarray): _description_
            r (float): _description_
            sigma (float): _description_
            deductible (float, optional): _description_. Defaults to 0.1.
            operating_insurer (float, optional): _description_. Defaults to 0.3.
            adapted_floodproofing (np.ndarray, optional): _description_. Defaults to None.
            adapted_windshutters (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            EU_insure_array (_type_): _description_
            premium (_type_): _description_
            premium_private (_type_): _description_
            premium_reinsured (_type_): _description_
        """
        damages_flood = expected_damages_flood.copy()
        damages_wind = expected_damages_wind.copy()

        if adapted_floodproofing is not None:
            idx_flood = np.where(adapted_floodproofing == 1)[0]
            damages_flood[:, idx_flood] = expected_damages_floodadapted[:, idx_flood]

        if adapted_windshutters is not None:
            idx_wind = np.where(adapted_windshutters == 1)[0]
            damages_wind[:, idx_wind] = expected_damages_windadapted[:, idx_wind]

        # Sort by probabilities (keep p as 1-D vectors)
        flood_order = np.argsort(p_flood)
        wind_order = np.argsort(p_wind)

        p_flood = np.asarray(p_flood).reshape(-1).astype(np.float32)[flood_order]
        p_wind = np.asarray(p_wind).reshape(-1).astype(np.float32)[wind_order]

        damages_flood = damages_flood[flood_order, :].astype(np.float32)
        damages_wind = damages_wind[wind_order, :].astype(np.float32)

        # insurance loss split
        damages_flood_residual = damages_flood * np.float32(deductible)
        damages_wind_residual = damages_wind * np.float32(deductible)

        damages_flood_insured = damages_flood * np.float32(1.0 - deductible)
        damages_wind_insured = damages_wind * np.float32(1.0 - deductible)

        # CAT-NAT premium
        premium, premium_reinsured, premium_private = self.Insurance_premium_CATNAT(
            expected_damages_flood=damages_flood_insured,
            p_flood=p_flood,
            expected_damages_wind=damages_wind_insured,
            p_wind=p_wind,
            operating_insurer=operating_insurer,
        )

        # Compute premiums with PPP (check order of arrays)
        # premium, premium_public, premium_private = self.Insurance_premium_PPP(
        #     expected_damages_flood=damages_flood_insured,
        #     p_flood=p_flood,
        #     expected_damages_wind=damages_wind_insured,
        #     p_wind=p_wind,
        #     operating_insurer=operating_insurer,
        #     public_reinsurer=public_reinsurer,
        # )

        # --- perceived probabilities (risk perception) ---
        rp = np.asarray(risk_perception, dtype=np.float32).reshape(-1)
        # if rp.shape != (n_agents,):
        #     raise AssertionError(f"risk_perception must be (n_agents,), got {rp.shape}")

        # Per-event perceived probabilities, capped
        p_flood_perc = np.minimum(p_flood[:, None] * rp[None, :], np.float32(0.998))
        p_wind_perc = np.minimum(p_wind[:, None] * rp[None, :], np.float32(0.998))

        # --- wealth baseline after paying premium (no income added here, consistent with your other flood EU functions) ---
        amenity_adjusted = np.asarray(amenity_value, dtype=np.float32) * np.float32(
            amenity_weight
        )
        W0 = np.maximum(
            np.float32(1.0),
            wealth.astype(np.float32) + amenity_adjusted - premium.astype(np.float32),
        )

        # --- utilities for each event-state ---
        # Flood event i: lose flood residual only
        W_iflood = np.maximum(np.float32(1.0), W0[None, :] - damages_flood_residual)
        # Wind event j: lose wind residual only
        W_iwind = np.maximum(np.float32(1.0), W0[None, :] - damages_wind_residual)

        if sigma == 1:
            U0 = np.log(W0)
            U_iflood = np.log(W_iflood)
            U_iwind = np.log(W_iwind)
        else:
            s = np.float32(sigma)
            U0 = (W0 ** (1 - s)) / (1 - s)
            U_iflood = (W_iflood ** (1 - s)) / (1 - s)
            U_iwind = (W_iwind ** (1 - s)) / (1 - s)

        # --- expected utility across combined hazard state-space (no double counting) ---
        # NOTE: This treats events as mutually exclusive states within a year.
        # It matches your discrete-event setup and avoids EU_flood + EU_wind inflation.
        sum_pf = np.sum(p_flood_perc, axis=0)  # (n_agents,)
        sum_pw = np.sum(p_wind_perc, axis=0)  # (n_agents,)
        p_noevent = np.maximum(np.float32(0.0), np.float32(1.0) - sum_pf - sum_pw)

        EU_insure_array = (
            np.sum(p_flood_perc * U_iflood, axis=0)
            + np.sum(p_wind_perc * U_iwind, axis=0)
            + p_noevent * U0
        )

        # Affordability constraint
        constrained = np.where(income * expenditure_cap <= premium)
        EU_insure_array[constrained] = -np.inf

        # Normalize shapes and add defensive checks so downstream code does not
        # accidentally sum across events instead of agents (which would lead to
        # counts > n_agents).
        EU_insure_array = np.asarray(EU_insure_array).reshape(-1)

        # Ensure premium outputs are arrays of length n_agents. If a scalar was
        # returned, broadcast to per-agent arrays to avoid shape mismatches.
        def _to_agent_array(x):
            arr = np.asarray(x)
            if arr.ndim == 0:
                return np.full(n_agents, float(arr), dtype=np.float32)
            return arr.reshape(-1).astype(np.float32)

        premium = _to_agent_array(premium)
        premium_private = _to_agent_array(premium_private)
        premium_reinsured = _to_agent_array(premium_reinsured)
        # premium_public = _to_agent_array(premium_public)

        return (
            EU_insure_array,
            premium,
            premium_private,
            premium_reinsured,
            # premium_public,
        )

    def Insurance_premium_PPP(
        self,
        expected_damages_flood: np.ndarray,
        p_flood: np.ndarray,
        expected_damages_wind: np.ndarray,
        p_wind: np.ndarray,
        operating_insurer: float = 0.3,
        method: str = "household",
        public_reinsurer: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute insurance premium under a public-private partnership (PPP) scheme.

        Args:
            expected_damages_flood (np.ndarray): _description_
            p_flood (np.ndarray): _description_
            expected_damages_wind (np.ndarray): _description_
            p_wind (np.ndarray): _description_
            adapted_floodproofing (np.ndarray, optional): _description_. Defaults to None.
            adapted_windshutters (np.ndarray, optional): _description_. Defaults to None.
            deductible (float, optional): _description_. Defaults to 0.1.
            operating_insurer (float, optional): _description_. Defaults to 0.3.
            method (str, optional): _description_. Defaults to "household".
            public_reinsurer (float, optional): _description_. Defaults to 0.0.

        Returns:
            premium_total: total premium per household
            premium_public: premium paid by public sector
            premium_private: premium paid by private insurer
        """

        def calc_EAD(
            damages: np.ndarray, probabilities: np.ndarray, method="household"
        ) -> np.ndarray:
            """
            Calculate Expected Annual Damages (EAD) with trapezoidal integration.

            Args:
                damages (np.ndarray): _description_
                probabilities (np.ndarray): _description_
                method (str, optional): _description_. Defaults to "household".

            Returns:
                EAD_
            """
            idx = np.argsort(probabilities)
            probabilities = probabilities[idx]
            damages = damages[idx, :]

            if method == "regional":
                # Sum across households first
                damages = damages.sum(axis=1, keepdims=True)

            return np.trapezoid(damages, x=probabilities, axis=0)

        # Calculate EADs (revisit what does it line mean)
        EAD_flood = (
            calc_EAD(expected_damages_flood, p_flood, method=method)
            if expected_damages_flood.size
            else np.zeros(expected_damages_wind.shape[1])
        )
        EAD_wind = (
            calc_EAD(expected_damages_wind, p_wind, method=method)
            if expected_damages_wind.size
            else np.zeros(expected_damages_flood.shape[1])
        )

        EAD_total = EAD_flood + EAD_wind

        # Base premium with operating_insurer factor
        premium = EAD_total * (1.0 + operating_insurer)

        # Split premium between public and private sector contributions
        if public_reinsurer > 0:
            premium_public = premium * public_reinsurer
            premium_private = premium * (1 - public_reinsurer)
        else:
            premium_public = np.zeros_like(premium)
            premium_private = premium

        return premium, premium_public, premium_private

    def Insurance_premium_CATNAT(
        self,
        expected_damages_flood: np.ndarray,
        p_flood: np.ndarray,
        expected_damages_wind: np.ndarray,
        p_wind: np.ndarray,
        operating_insurer: float = 0.3,
        catnat_surcharge: float = 0.20,
        reinsurance_share: float = 0.5,
        method: str = "household",
        pooled: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Insurance premium calculation including CATNAT scheme.

        Args:
            expected_damages_flood (np.ndarray): _description_
            p_flood (np.ndarray): _description_
            expected_damages_wind (np.ndarray): _description_
            p_wind (np.ndarray): _description_
            operating_insurer (float, optional): _description_. Defaults to 0.3.
            catnat_surcharge (float, optional): _description_. Defaults to 0.20.
            reinsurance_share (float, optional): _description_. Defaults to 0.5.
            method (str, optional): _description_. Defaults to "household".
            pooled (bool, optional): EAD represented as a mean and boradcasted to all households.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """

        def calc_EAD(
            damages: np.ndarray, probabilities: np.ndarray, method: str = "household"
        ) -> np.ndarray:
            idx = np.argsort(probabilities)
            probabilities = probabilities[idx]
            damages = damages[idx, :]

            if method == "regional":
                damages = damages.sum(axis=1, keepdims=True)
            return np.trapezoid(damages, x=probabilities, axis=0)

        EAD_flood = (
            calc_EAD(expected_damages_flood, p_flood, method)
            if expected_damages_flood.size
            else np.zeros(expected_damages_wind.shape[1])
        )

        EAD_wind = (
            calc_EAD(expected_damages_wind, p_wind, method)
            if expected_damages_wind.size
            else np.zeros(expected_damages_flood.shape[1])
        )

        EAD_total = EAD_flood + EAD_wind

        if pooled:
            ead_mean = float(np.mean(EAD_total))
            EAD_total = np.full_like(EAD_total, ead_mean, dtype=np.float32)

        premium_risk = EAD_total * (1.0 + operating_insurer)

        premium_catnat = catnat_surcharge * premium_risk
        premium_total = premium_risk + premium_catnat

        # upstreaam split
        premium_reinsured = premium_catnat * reinsurance_share
        premium_retained = premium_total - premium_reinsured

        return premium_total, premium_reinsured, premium_retained
