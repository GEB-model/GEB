import numpy as np
from numba import njit, prange


class DecisionModule:
    def __init__(self, agents) -> None:
        self.agents = agents

    @staticmethod
    @njit(cache=True)
    def IterateThroughFlood(
        n_floods: int,
        total_profits: np.ndarray,
        profits_no_event: np.ndarray,
        max_T: int,
        n_agents: int,
        discount_rate: np.ndarray,
    ) -> np.ndarray:
        """This function iterates through each (no)flood event i (see manuscript for details). It calculates the time discounted NPV of each event i:

        Args:
            n_floods: number of flood maps included in model run
            wealth: array containing the wealth of each household
            income: array containing the income of each household
            max_T: time horizon (same for each agent)
            expected_damages: array containing the expected damages of each household under all events i
            n_agents: number of household agents in the current admin unit
            r: time discounting rate

        Returns:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent
        """

        # Allocate array
        NPV_summed = np.full((n_floods + 3, n_agents), -1, dtype=np.float32)

        # Iterate through all floods
        for i, index in enumerate(np.arange(1, n_floods + 3)):
            # Check if we are in the last iterations
            if i < n_floods:
                NPV_flood_i = total_profits[i]
                NPV_flood_i = (NPV_flood_i).astype(np.float32)

            # if in the last two iterations do not subtract damages (probs of no event)
            elif i >= n_floods:
                NPV_flood_i = profits_no_event
                NPV_flood_i = NPV_flood_i.astype(np.float32)

            # iterate over NPVs for each year in the time horizon and apply time
            # discounting
            NPV_t0 = (profits_no_event).astype(np.float32)  # no flooding in t=0

            # Calculate time discounted NPVs
            t_arr = np.arange(1, max_T, dtype=np.float32)

            discounts = 1 / (1 + np.reshape(discount_rate, (-1, 1))) ** t_arr
            NPV_tx = np.sum(discounts, axis=1) * NPV_flood_i

            # Add NPV at t0 (which is not discounted)
            NPV_tx += NPV_t0

            # Store result
            NPV_summed[index] = NPV_tx

        # Store NPV at p=0 for bounds in integration
        NPV_summed[0] = NPV_summed[1]

        return NPV_summed

    def calcEU_do_nothing(
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
        **kwargs,
    ) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Args:
            n_agents: number of agents in the floodplain.
            wealth: array containing the wealth of each household.
            income: array containing the income of each household.
            risk_perception: array containing the risk perception of each household (see manuscript for details).
            expected_damages: array expected damages for each flood event for each agent under no implementation of dry flood proofing.
            adapted: array containing the adaptation status of each agent (1 = adapted, 0 = not adapted).
            p_droughts: array containing the exceedance probabilities of each flood event included in the analysis.
            T: array containing the decision horizon of each agent.
            r: time discounting factor.
            sigma: risk aversion setting.

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
        max_T = np.int32(np.max(T))

        # Part njit, iterate through floods
        n_agents = np.int32(n_agents)
        NPV_summed = self.IterateThroughFlood(
            n_floods,
            total_profits,
            profits_no_event,
            max_T,
            n_agents,
            discount_rate,
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
        EU_do_nothing_array = np.trapz(y=y, x=x, axis=0)

        # People who already adapted cannot adapt, changed to a condition in the function that calls this, need the SEUT of doing nothing of those that have adapted
        # EU_do_nothing_array[np.where(adapted == 1)] = -np.inf

        return EU_do_nothing_array

    @staticmethod
    @njit(cache=True, parallel=True)
    def calcEU_adapt(
        expenditure_cap: float,
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
        unconstrained_mask = (
            (profits_no_event * expenditure_cap > total_annual_costs)
            & (adapted == 0)
            & extra_constraint
        )

        # Iterate only through agents who can afford to adapt
        unconstrained_indices = np.where(unconstrained_mask)[0]

        for idx in prange(unconstrained_indices.size):
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
            EU_adapt[i] = np.trapz(EU_adapt_dict, p_all_events)

            # Ensure no NaNs
            if np.isnan(EU_adapt[i]):
                EU_adapt[i] = -np.inf

        return EU_adapt

    @staticmethod
    def calcEU_adapt_vectorized(
        *,
        expenditure_cap: float,
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
        """Vectorized version of the calcEU_adapt function without @njit."""
        # Preallocate arrays
        EU_adapt = np.full(n_agents, -np.inf, dtype=np.float32)

        # Ensure p_droughts is in increasing order
        indices = np.argsort(p_droughts)
        total_profits_adaptation = total_profits_adaptation[indices]
        p_droughts = p_droughts[indices]

        if adapted is not None:
            # Identify agents able to afford the adaptation and that have not yet adapted
            unconstrained_mask = (
                (profits_no_event * expenditure_cap > total_annual_costs)
                & (adapted == 0)
                & extra_constraint
            )
        else:
            unconstrained_mask = extra_constraint

        # Proceed with unconstrained agents
        if np.any(unconstrained_mask):
            # Extract data for unconstrained agents
            idx = np.where(unconstrained_mask)[0]
            n_unconstrained = idx.size

            # Variables per agent
            sigma_uc = sigma[idx]
            discount_rate_uc = discount_rate[idx]
            T_uc = T[idx]
            payment_remainder = np.maximum(loan_duration - time_adapted[idx], 0)
            profits_no_event_adaptation_uc = profits_no_event_adaptation[idx]
            adaptation_costs_uc = adaptation_costs[idx]
            risk_perception_uc = risk_perception[idx]

            # Max decision horizon
            max_T = np.max(T_uc)
            t = np.arange(max_T, dtype=np.int32)

            # Time arrays
            t_agents = np.tile(
                t, (n_unconstrained, 1)
            )  # Shape: (n_unconstrained, max_T)
            mask_t = t_agents < T_uc[:, np.newaxis]  # Shape: (n_unconstrained, max_T)

            # Payment mask
            payment_mask = (
                t_agents < payment_remainder[:, np.newaxis]
            )  # Shape: (n_unconstrained, max_T)

            # NPV under no drought event
            NPV_adapt_no_flood = np.full(
                (n_unconstrained, max_T),
                profits_no_event_adaptation_uc[:, np.newaxis],
                dtype=np.float32,
            )
            NPV_adapt_no_flood -= adaptation_costs_uc[:, np.newaxis] * payment_mask
            NPV_adapt_no_flood = np.maximum(NPV_adapt_no_flood, 1e-6)
            NPV_adapt_no_flood /= (1 + discount_rate_uc[:, np.newaxis]) ** t_agents
            NPV_adapt_no_flood *= mask_t
            NPV_adapt_no_flood_summed = np.sum(NPV_adapt_no_flood, axis=1)
            NPV_adapt_no_flood_summed = np.maximum(NPV_adapt_no_flood_summed, 1e-6)
            EU_adapt_no_flood = (NPV_adapt_no_flood_summed ** (1 - sigma_uc)) / (
                1 - sigma_uc
            )

            # NPV outcomes for each drought event
            n_events = p_droughts.size
            total_profits_adaptation_uc = total_profits_adaptation[:, idx]

            # Expand arrays to match dimensions
            NPV_adapt = np.repeat(
                total_profits_adaptation_uc.T[:, :, np.newaxis],
                max_T,
                axis=2,
            ).astype(np.float32)  # Shape: (n_unconstrained, n_events, max_T)

            # Adjust for adaptation costs during payment period
            payment_mask_expanded = payment_mask[
                :, np.newaxis, :
            ]  # Shape: (n_unconstrained, 1, max_T)
            NPV_adapt -= (
                adaptation_costs_uc[:, np.newaxis, np.newaxis] * payment_mask_expanded
            )

            # Ensure NPVs are at least a small positive number before discounting
            NPV_adapt = np.maximum(NPV_adapt, 1e-6)

            # Discounting
            t_agents_expanded = t_agents[
                :, np.newaxis, :
            ]  # Shape: (n_unconstrained, 1, max_T)
            NPV_adapt /= (
                1 + discount_rate_uc[:, np.newaxis, np.newaxis]
            ) ** t_agents_expanded

            # Apply decision horizon mask
            mask_t_expanded = mask_t[:, np.newaxis, :]
            NPV_adapt *= mask_t_expanded

            # Sum over time
            NPV_adapt_summed = np.sum(NPV_adapt, axis=2)
            NPV_adapt_summed = np.maximum(NPV_adapt_summed, 1e-6)

            # Calculate expected utilities
            EU_adapt_flood = (NPV_adapt_summed ** (1 - sigma_uc[:, np.newaxis])) / (
                1 - sigma_uc[:, np.newaxis]
            )

            # Prepare for integration
            EU_adapt_dict = np.zeros((n_unconstrained, n_events + 3), dtype=np.float32)
            EU_adapt_dict[:, 1 : n_events + 1] = EU_adapt_flood
            EU_adapt_dict[:, n_events + 1 : n_events + 3] = EU_adapt_no_flood[
                :, np.newaxis
            ]
            EU_adapt_dict[:, 0] = EU_adapt_flood[:, 0]

            p_all_events = np.zeros((n_unconstrained, n_events + 3), dtype=np.float32)
            p_all_events[:, 1 : n_events + 1] = (
                risk_perception_uc[:, np.newaxis] * p_droughts[np.newaxis, :]
            )
            p_all_events[:, n_events + 1] = p_all_events[:, n_events] + 0.001
            p_all_events[:, n_events + 2] = 1.0
            p_all_events[:, 0] = 0.0

            # Integrate EU over probabilities using trapezoidal rule
            EU_adapt_uc = np.trapz(EU_adapt_dict, p_all_events, axis=1)

            # Assign back to main array
            EU_adapt[idx] = EU_adapt_uc

            # Handle NaNs if any
            EU_adapt[np.isnan(EU_adapt)] = -np.inf

        return EU_adapt
