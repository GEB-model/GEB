import numpy as np
from numba.core.decorators import njit
from scipy import interpolate


@njit
def fast_intersect(a, b):
    # Assuming both arrays are sorted
    i, j = 0, 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return np.array(result)


class DecisionModule:
    def __init__(self, agents, model) -> None:
        self.agents = agents
        self.model = model
        # income wealth ratio
        perc = np.array([0, 20, 40, 60, 80, 100])  # percentile in income distribution
        ratio = np.array([0, 1.06, 4.14, 4.19, 5.24, 6])  # wealth in relation to income
        income_wealth_ratio_function = interpolate.interp1d(x=perc, y=ratio)
        self.income_wealth_ratio = income_wealth_ratio_function(np.arange(101))

        # get value for error terms from model settings
        if hasattr(self.model, "settings"):
            self.error_interval = self.model.settings["decisions"]["error_interval"]
        else:
            self.error_interval = 0

    def sample_error_terms(self, n_agents, regions_select):
        """This function samples error terms for the expected utility of migration and the expected utility of adaptation."""
        self.error_terms_stay = self.model.random_module.random_state.uniform(
            (1 - self.error_interval), (1 + self.error_interval), size=n_agents
        )
        self.error_terms_migrate = self.model.random_module.random_state.uniform(
            (1 - self.error_interval),
            (1 + self.error_interval),
            size=(regions_select.size, n_agents),
        )

    @staticmethod
    @njit(cache=True)
    def IterateThroughFlood(
        n_floods: int,
        wealth: np.ndarray,
        income: np.ndarray,
        amenity_value: np.ndarray,
        max_T: int,
        expected_damages: np.ndarray,
        n_agents: int,
        r: float,
    ) -> np.ndarray:
        """This function iterates through each (no)flood event i (see manuscript for details).

        It calculates the time discounted NPV of each event i:

        Args:
            n_floods: number of flood maps included in model run
            wealth: array containing the wealth of each household
            income: array containing the income of each household
            amenity_value: array containing the amenity value of each household'
            max_T: time horizon (same for each agent)
            expected_damages: array containing the expected damages of each household under all events i
            n_agents: number of household agents in the current admin unit
            r: time discounting rate

        Returns:
            NPV_summed: Array containing the summed time discounted NPV for each event i for each agent
        """
        # Allocate array
        NPV_summed = np.full((n_floods + 3, n_agents), -1, dtype=np.float32)

        NPV_t0 = (wealth + income + amenity_value).astype(
            np.float32
        )  # no flooding in t=0

        # Iterate through all floods
        for i, index in enumerate(np.arange(1, n_floods + 3)):
            # Check if we are in the last iterations
            if i < n_floods:
                NPV_flood_i = wealth + income + amenity_value - expected_damages[i]
                NPV_flood_i = (NPV_flood_i).astype(np.float32)

            # if in the last two iterations do not subtract damages (probs of no
            # flood event)
            elif i >= n_floods:
                NPV_flood_i = wealth + income + amenity_value
                NPV_flood_i = NPV_flood_i.astype(np.float32)

            # Calculate time discounted NPVs
            t_arr = np.arange(1, max_T, dtype=np.float32)
            discounts = 1 / (1 + r) ** t_arr
            NPV_tx = np.sum(discounts) * NPV_flood_i

            # Add NPV at t0 (which is not discounted)
            NPV_tx += NPV_t0

            # Store result
            NPV_summed[index] = NPV_tx

        # Store NPV at p=0 for bounds in integration
        NPV_summed[0] = NPV_summed[1]

        return NPV_summed

    def calcEU_adapt(
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
        p_floods: np.ndarray,
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
        max_T = np.int32(np.max(T))

        # Part njit, iterate through floods
        n_agents = np.int32(n_agents)
        NPV_summed = self.IterateThroughFlood(
            n_floods,
            wealth,
            income,
            amenity_value,
            max_T,
            expected_damages_adapt,
            n_agents,
            r,
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

    def calcEU_insure(
        self,
        geom_id,
        n_agents: int,
        wealth: np.ndarray,
        income: np.ndarray,
        expendature_cap,
        amenity_value: np.ndarray,
        amenity_weight,
        risk_perception: np.ndarray,
        expected_damages: np.ndarray,
        premium: np.ndarray,
        p_floods: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: float,
        deductable=0.1,
        **kwargs,
    ) -> np.ndarray:
        """This function calculates the time discounted subjective utility of not undertaking any action.

        Returns:
            EU_do_nothing_array: array containing the time discounted subjective utility of doing nothing for each agent.
        """
        # multiply damages by deductable
        expected_damages_insured = expected_damages * deductable

        # weigh amenities
        amenity_value = amenity_value * amenity_weight

        # Ensure p floods is in increasing order
        indices = np.argsort(p_floods)
        expected_damages_insured = expected_damages_insured[indices]
        p_floods = np.sort(p_floods)

        # Preallocate arrays
        n_floods, n_agents = expected_damages_insured.shape
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
        max_T = np.int32(np.max(T))

        # Part njit, iterate through floods
        n_agents = np.int32(n_agents)
        NPV_summed = self.IterateThroughFlood(
            n_floods,
            wealth,
            income,
            amenity_value,
            max_T,
            expected_damages_insured,
            n_agents,
            r,
        )

        # some usefull attributes for cost calculations
        # get time discounted adaptation cost for each agent based on time since adaptation
        discounts = 1 / (1 + r) ** np.arange(max_T)

        # create cost array for years payment left
        years = np.arange(max_T + 1)
        cost_array = np.full(years.size, -1, np.float32)
        for i, year in enumerate(years):
            cost_array[i] = np.sum(discounts[:year] * premium.mean())

        # take the calculated adaptation cost based on decision horizon agents
        time_discounted_adaptation_cost = np.take(cost_array, T)

        # subtract from NPV array
        NPV_summed -= time_discounted_adaptation_cost

        # Filter out negative NPVs
        NPV_summed = np.maximum(1, NPV_summed)
        if (NPV_summed == 1).any():
            n_negative = np.sum(NPV_summed == 1)
            print(
                f"[calcEU_insure] Warning, {n_negative} negative NPVs encountered in {geom_id} ({np.round(n_negative / NPV_summed.size * 100, 2)}%)"
            )

        # Calculate expected utility
        if sigma == 1:
            EU_store = np.log(NPV_summed)
        else:
            EU_store = (NPV_summed ** (1 - sigma)) / (1 - sigma)

        # Use composite trapezoidal rule integrate EU over event probability
        y = EU_store
        x = p_all_events
        EU_insure_array = np.trapezoid(y=y, x=x, axis=0)

        # set EU of adapt to -np.inf for thuse unable to afford it [maybe move this earlier to reduce array sizes]
        constrained = np.where(income * expendature_cap <= premium)
        EU_insure_array[constrained] = -np.inf
        EU_insure_array *= self.error_terms_stay

        return EU_insure_array

    def calcEU_do_nothing(
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
        p_floods: np.ndarray,
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
        max_T = np.int32(np.max(T))

        # Part njit, iterate through floods
        n_agents = np.int32(n_agents)
        NPV_summed = self.IterateThroughFlood(
            n_floods,
            wealth,
            income,
            amenity_value,
            max_T,
            expected_damages,
            n_agents,
            r,
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

        # EU_do_nothing_array *= self.error_terms_stay
        return EU_do_nothing_array

    def match_amenity_premium(
        self,
        allocation_mask,
        damage_factor_region,
        risk_perception,
        admin_idx,
        cells_to_assess,
        amenity_premium_cells_region,
        map_amenity_premium_region,
        current_amenity_premium,
        unique_current_premiums,
        counts,
    ):
        # iterate over unique current premiums households
        for premium, count in zip(unique_current_premiums, counts):
            # get cells that have the same or a lower premium
            cells_with_similar_premium = map_amenity_premium_region[premium]

            if cells_with_similar_premium.size > 0:
                # only consider cells that are in the allocation mask (this might be slow)
                cells_within_mask = fast_intersect(
                    allocation_mask, cells_with_similar_premium
                )

                if self.model.args.run_with_checks:
                    # check if fast intersect works as it should
                    cells_within_mask_1 = np.intersect1d(
                        allocation_mask, cells_with_similar_premium
                    )
                    assert all(cells_within_mask == cells_within_mask_1)

                if cells_within_mask.size > 0:
                    cells_with_similar_premium = cells_within_mask
                    agent_idx = np.where(
                        current_amenity_premium == premium
                    )  # indices of the agents with the premium we are iterating over
                    sample = self.model.random_module.random_state.choice(
                        cells_with_similar_premium, size=(5, count)
                    )  # sample 5 random cells
                    premium_sample = amenity_premium_cells_region[
                        sample
                    ]  # get the amenity premium of these cells
                    risk_sample = damage_factor_region[
                        sample
                    ]  # get the risk in these cells
                    risk_sample *= risk_perception[
                        agent_idx
                    ]  # adjust for risk perceptions
                    # take best cells
                    best_cells = (premium_sample - risk_sample).argmax(
                        axis=0
                    )  # subtract from amenity premium to determine best locations
                    cells_to_assess[agent_idx] = sample[
                        best_cells, np.arange(sample.shape[1])
                    ]
            else:
                # if no matching premiums are found return -1. Will be used as 'flag'
                cells_to_assess = np.full(
                    cells_to_assess.size, -1, cells_to_assess.dtype
                )

        return cells_to_assess

    def fill_regions(
        self,
        n_agents,
        admin_idx,
        regions_select: np.ndarray,
        income_distribution_regions: np.ndarray,
        amenity_premium_regions,
        snapshot_damages_cells,
        property_value_nodes,
        amenity_weight,
        current_amenity_premium,
        income_percentile: np.ndarray,
        expected_income_agents: np.ndarray,
        expected_wealth_agents,
        expected_amenity_value_agents,
        expected_ead_agents,
        risk_perception,
        cells_assessed,
    ):
        # frac_coastal_cells=np.full(len(amenity_value_regions), 1)
        for i, region in enumerate(regions_select):
            # Sort income
            sorted_income = income_distribution_regions[
                region
            ]  # should be sorted after creation
            # Derive indices from percentiless
            income_indices = np.floor(
                income_percentile * 0.01 * len(sorted_income)
            ).astype(np.int32)

            # Extract based on indices
            expected_income_agents[i] = sorted_income[income_indices]

            # calculate expected wealth agents
            expected_wealth_agents[i] = (
                np.take(self.income_wealth_ratio, income_percentile)
                * expected_income_agents[i]
            )

            # sample amenity premiums
            assert (
                amenity_premium_regions[region].shape
                == snapshot_damages_cells[region].shape
            )
            # iterate to find matching amenity premiums in destination
            cells_to_assess = np.full(n_agents, -1, np.int32)
            if (
                region in self.agents.regions.amenity_premium_indices.keys()
                and amenity_premium_regions[region].size > 1
            ):
                # get allocation mask to filter out full cells
                allocation_mask = self.agents.regions.all_households[
                    region
                ].mask_household_allocation

                # get indices of amenity values destination region
                amenity_indices_destination = (
                    self.agents.regions.amenity_premium_indices[region]
                )

                # get unique bins
                unique_current_amenity_bins, counts = np.unique(
                    current_amenity_premium, return_counts=True
                )

                # get percieved risk
                damage_factor_region = snapshot_damages_cells[region]

                cells_to_assess = self.match_amenity_premium(
                    allocation_mask=allocation_mask,
                    admin_idx=admin_idx,
                    cells_to_assess=cells_to_assess,
                    damage_factor_region=damage_factor_region,
                    risk_perception=risk_perception,
                    amenity_premium_cells_region=amenity_premium_regions[region],
                    map_amenity_premium_region=amenity_indices_destination,
                    current_amenity_premium=current_amenity_premium,
                    unique_current_premiums=unique_current_amenity_bins,
                    counts=counts,
                )

                # get amenity premium
                amenity_premium_sampled_cells = amenity_premium_regions[region][
                    cells_to_assess
                ]

                # no amenity premium to -np.inf if no matching premiums are found (inhibit move to this region)
                amenity_premium_sampled_cells[cells_to_assess == -1] = 0

                # get ead corresponding with cell
                ead_sampled_cells = snapshot_damages_cells[region][cells_to_assess]
                expected_amenity_value = (
                    amenity_premium_sampled_cells
                    * amenity_weight
                    * expected_wealth_agents[i]
                )

                # do some checks
                # print(f'amenity value exceeds risk for {(expected_amenity_value>ead_sampled_cells).sum()} agents ({(expected_amenity_value>ead_sampled_cells).sum()/ expected_amenity_value.size * 100}%)')
                assert all(current_amenity_premium >= amenity_premium_sampled_cells)
                cells_assessed[region] = cells_to_assess

            else:  # migration to inland node
                amenity_premium_sampled_cells = 0
                ead_sampled_cells = np.zeros(n_agents)
                expected_amenity_value = np.zeros(n_agents)

            expected_amenity_value_agents[i] = expected_amenity_value
            expected_ead_agents[i] = ead_sampled_cells
        return amenity_premium_sampled_cells

    def fill_NPVs(
        self,
        admin_idx,
        regions_select: np.ndarray,
        average_flood_risk_regions,
        risk_perception,
        expected_wealth_agents: np.ndarray,
        expected_income_agents: np.ndarray,
        expected_amenity_value_agents: np.ndarray,
        expected_ead_agents,
        n_agents: int,
        distance: np.ndarray,
        max_T: int,
        r: float,
        t_arr: np.ndarray,
        sigma: float,
        Cmax: int,
        cost_shape: float,
        error_terms: np.ndarray,
        cells_assessed,
    ):
        # Preallocate arrays
        NPV_summed = np.full((regions_select.size, n_agents), -1, dtype=np.float32)

        # Fill NPV arrays for migration to each region
        for i, region in enumerate(regions_select):
            # Add wealth and expected incomes. Fill values for each year t in max_T
            NPV_t0 = (
                expected_wealth_agents[i, :]
                + expected_income_agents[i, :]
                + expected_amenity_value_agents[i, :]
            ).astype(np.float32)

            # subtract percieved flood risk region
            percieved_flood_risk = expected_ead_agents[i, :] * risk_perception
            NPV_t0 -= percieved_flood_risk

            # # Apply and sum time discounting
            t_arr = np.arange(max_T, dtype=np.float32)
            discounts = 1 / (1 + r) ** t_arr
            NPV_region_discounted = np.sum(discounts) * NPV_t0

            # Subtract migration costs (these are not time discounted, occur at
            # t=0) s
            # set a max to distance of 5000 km to prevent underflow
            distance_to_dest = np.minimum(distance[region], 5_000)
            y = Cmax / (1 + np.exp(-cost_shape * distance_to_dest))
            NPV_region_discounted = NPV_region_discounted - y

            # Filter out negative NPVs
            NPV_region_discounted[NPV_region_discounted < 0] = -np.inf

            # account for agents not finding housing within range of current amenity premium if destination is coastal
            if self.agents.regions.ids[region].endswith("flood_plain"):
                NPV_region_discounted[cells_assessed[region] == -1] = -np.inf

            # Store time discounted values
            NPV_summed[i, :] = NPV_region_discounted

        # Calculate expected utility
        if sigma == 1:
            NPV_summed[NPV_summed <= 0] = 1
            EU_regions = np.log(NPV_summed)
            EU_regions[
                NPV_summed < 0
            ] = -np.inf  # just to make sure nobody moves to negative NPVs
            assert (EU_regions != -1).any()  # check if all filled
        else:
            EU_regions = NPV_summed ** (1 - sigma) / (1 - sigma)

        # process error terms
        EU_regions *= error_terms

        # set EU to -np.inf for those who could not find suitable housing

        return EU_regions

    def EU_migrate(
        self,
        property_value_nodes,
        current_amenity_premium,
        average_ead_snapshot,
        snapshot_damages_cells,
        risk_perception,
        admin_idx,
        geom_id,
        regions_select: np.ndarray,
        n_agents: int,
        sigma: float,
        wealth: np.ndarray,
        income_distribution_regions: np.ndarray,
        income_percentile: np.ndarray,
        amenity_premium_regions: np.ndarray,
        amenity_weight,
        distance: np.ndarray,
        Cmax: int,
        cost_shape: float,
        T: np.ndarray,
        r: float,
    ):
        """This function calculates the subjective expected utilty of migration and the region for which utility is higher.

        The function loops through each agent, processing their individual characteristics.

        Returns:
            EU_migr_MAX_return: array containing the maximum expected utility of migration of each agent.
            ID_migr_MAX_return: array containing the region IDs for which the utility of migration is highest for each agent.

        """
        # Preallocate arrays
        # EU_migr = np.full(len(regions_select), -1, dtype=np.float32)
        max_T = np.int32(np.max(T))
        # NPV_discounted = np.full(regions_select.size, -1, np.float32)

        # Preallocate decision horizons
        t_arr = np.arange(0, np.max(T), dtype=np.int32)
        regions_select = np.sort(regions_select)

        # Fill array with expected income based on position in income distribution
        expected_income_agents = np.full(
            (regions_select.size, income_percentile.size), -1, dtype=np.int32
        )
        expected_amenity_value_agents = np.full(
            (regions_select.size, income_percentile.size), -1, dtype=np.int32
        )
        expected_wealth_agents = np.full(
            (regions_select.size, income_percentile.size), -1, dtype=np.int32
        )
        expected_ead_agents = np.full(
            (regions_select.size, income_percentile.size), -1, dtype=np.int32
        )
        cells_assessed = np.full(
            (self.model.agents.regions.n, income_percentile.size), -1, dtype=np.int32
        )  # now this array is a bit large, could also work with relative indices

        self.fill_regions(
            n_agents=n_agents,
            admin_idx=admin_idx,
            regions_select=regions_select,
            amenity_premium_regions=amenity_premium_regions,
            snapshot_damages_cells=snapshot_damages_cells,
            property_value_nodes=property_value_nodes,
            income_distribution_regions=income_distribution_regions,
            income_percentile=income_percentile,
            current_amenity_premium=current_amenity_premium,
            amenity_weight=amenity_weight,
            expected_amenity_value_agents=expected_amenity_value_agents,
            expected_income_agents=expected_income_agents,
            expected_wealth_agents=expected_wealth_agents,
            expected_ead_agents=expected_ead_agents,
            risk_perception=risk_perception,
            cells_assessed=cells_assessed,
        )

        EU_regions = self.fill_NPVs(
            admin_idx=admin_idx,
            average_flood_risk_regions=average_ead_snapshot,
            regions_select=regions_select,
            risk_perception=risk_perception,
            expected_wealth_agents=expected_wealth_agents,
            expected_income_agents=expected_income_agents,
            expected_amenity_value_agents=expected_amenity_value_agents,
            expected_ead_agents=expected_ead_agents,
            n_agents=n_agents,
            distance=distance,
            max_T=max_T,
            r=r,
            t_arr=t_arr,
            sigma=sigma,
            Cmax=Cmax,
            cost_shape=cost_shape,
            error_terms=self.error_terms_migrate,
            cells_assessed=cells_assessed,
        )

        EU_migr_MAX_return, ID_migr_MAX_return = self.allocate_destinations(
            EU_regions, regions_select, n_agents
        )

        return EU_migr_MAX_return, ID_migr_MAX_return, cells_assessed

    def allocate_destinations(self, EU_regions, regions_select, n_agents, n_choices=10):
        """This function allocates agents to regions based on the highest expected utility of migration.

        It checks if the number of household agents does not exceed a percentage of the current populaton, if it does it sets the expected utility of migration to -np.inf for all other agents.
        This is done to prevent agents from migrating to a region that is already full.

        Args:
            EU_regions: array containing the expected utility of migration for each agent in each region.
            regions_select: array containing the region IDs of the regions that are selected for migration.
            n_agents: integer containing the number of agents in the simulation.
            n_choices: integer containing the number of regions to consider for migration.

        Returns:
            EU_migr_MAX_return: array containing the maximum expected utility of migration of each agent.
            ID_migr_MAX_return: array containing the region IDs for which the utility of migration is highest for each agent.
        """
        # n choices
        n_choices = np.min([n_choices, regions_select.size])

        # create array with first and second migration choice
        EU_migr_MAX_return = np.full((n_choices, n_agents), -1, np.float32)
        ID_migr_MAX_return = np.full((n_choices, n_agents), -1, np.int32)

        # Loop over each choice
        for i in range(n_choices):
            # Set the first and subsequent best migration choices
            EU_migr_MAX_return[i, :] = EU_regions.max(axis=0)
            region_indices = EU_regions.argmax(axis=0)
            ID_migr_MAX_return[i, :] = np.take(regions_select, region_indices)

            # set EU to -np.inf and get next best if not the last choice
            if i < n_choices - 1:
                EU_regions[region_indices, np.arange(n_agents)] = -np.inf
        return EU_migr_MAX_return, ID_migr_MAX_return
