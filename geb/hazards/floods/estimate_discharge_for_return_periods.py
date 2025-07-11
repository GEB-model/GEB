from .io import export_rivers
from .sfincs_utils import (
    assign_return_periods,
    create_hourly_hydrograph,
    get_discharge_by_river,
    get_representative_river_points,
)


def estimate_discharge_for_return_periods(
    model_root,
    discharge,
    rivers,
    rising_limb_hours=72,
    return_periods=[2, 5, 10, 20, 50, 100, 250, 500, 1000],
):
    recession_limb_hours = rising_limb_hours

    # here we only select the rivers that have an upstream forcing point
    rivers_with_forcing_point = rivers[~rivers["is_downstream_outflow_subbasin"]]

    river_representative_points = []
    for ID in rivers_with_forcing_point.index:
        river_representative_points.append(
            get_representative_river_points(ID, rivers_with_forcing_point)
        )

    discharge_by_river, _, _ = get_discharge_by_river(
        rivers_with_forcing_point.index,
        river_representative_points,
        discharge=discharge,
    )
    rivers_with_forcing_point = assign_return_periods(
        rivers_with_forcing_point, discharge_by_river, return_periods=return_periods
    )

    for return_period in return_periods:
        rivers_with_forcing_point[f"hydrograph_{return_period}"] = None

    for river_idx in rivers_with_forcing_point.index:
        for return_period in return_periods:
            discharge_for_return_period = rivers_with_forcing_point.at[
                river_idx, f"Q_{return_period}"
            ]
            hydrograph = create_hourly_hydrograph(
                discharge_for_return_period,
                rising_limb_hours,
                recession_limb_hours,
            )
            hydrograph = {
                time.isoformat(): Q.item() for time, Q in hydrograph.iterrows()
            }
            rivers_with_forcing_point.at[river_idx, f"hydrograph_{return_period}"] = (
                hydrograph
            )

    export_rivers(model_root, rivers_with_forcing_point, postfix="_return_periods")
