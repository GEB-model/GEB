from .io import export_rivers
from .sfincs_utils import (
    assign_return_periods,
    create_hourly_hydrograph,
    get_discharge_by_point,
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

    rivers_with_forcing_point_ = rivers_with_forcing_point[
        rivers_with_forcing_point["hydrography_xy"].apply(len) > 0
    ]
    if len(rivers_with_forcing_point_) < len(rivers_with_forcing_point):
        print('WARNING: REMOVED SMALL RIVERS, TEMPORARY "FIX"')
        rivers_with_forcing_point = rivers_with_forcing_point_.copy()

    xs, ys = [], []
    for _, river in rivers_with_forcing_point.iterrows():
        xy = river["hydrography_xy"][0]  # get most upstream point
        xs.append(xy[0])
        ys.append(xy[1])

    discharge_series = get_discharge_by_point(
        xs=xs,
        ys=ys,
        discharge=discharge,
    )
    rivers_with_forcing_point = assign_return_periods(
        rivers_with_forcing_point, discharge_series, return_periods=return_periods
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
