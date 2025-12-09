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

        else:
            raise ValueError("Mode must be either 'flood' or 'drought'")

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
        T: np.ndarray,
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
        amenity_weight: np.ndarray,
        risk_perception: np.ndarray,
        expected_damages: np.ndarray,
        adapted: np.ndarray,
        p_floods: np.ndarray,
        T: np.ndarray,
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
        EU_do_nothing_array[np.where(adapted == 1)] = -np.inf

        return EU_do_nothing_array
