import numpy as np
import pandas as pd

from geb.agents.modules.flood_risk import FloodRiskModule


def test_apply_dry_flood_proofing_to_curve_threshold_step() -> None:
    unprotected: pd.Series = pd.Series(
        [0.0, 0.2, 0.5, 0.5, 0.9],
        index=pd.Index([0.0, 0.5, 1.0, 1.01, 2.0], name="depth"),
        name="building_unprotected",
        dtype=float,
    )

    out: pd.Series = FloodRiskModule.apply_dry_flood_proofing_to_curve(
        unprotected, reduction_factor=0.15, threshold_depth_m=1.0
    )

    expected: pd.Series = pd.Series(
        [0.0, 0.03, 0.075, 0.5, 0.9],
        index=unprotected.index,
        name=unprotected.name,
        dtype=float,
    )
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)


# CBA tests disabled; retained for possible future use.
# def test_compute_cba_metrics_for_return_period_shapes_and_totals() -> None:
#     baseline: np.ndarray = np.array([100.0, 200.0, 0.0], dtype=np.float64)
#     flood_proofed: np.ndarray = np.array([40.0, 150.0, 0.0], dtype=np.float64)
#     adapted_mask: np.ndarray = np.array([True, False, True], dtype=bool)
#     costs_upfront: np.ndarray = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)
#
#     household_df, summary_df = FloodRiskModule.compute_cba_metrics_for_return_period(
#         damages_baseline_usd_event=baseline,
#         damages_flood_proofed_usd_event=flood_proofed,
#         adapted_mask=adapted_mask,
#         adaptation_costs_usd_upfront=costs_upfront,
#         return_period_years=100.0,
#         loan_duration_years=20,
#     )
#
#     assert household_df.shape[0] == 3
#     assert summary_df.shape[0] == 1
#
#     np.testing.assert_allclose(
#         household_df["damage_with_measure_usd_event"].to_numpy(),
#         np.array([40.0, 200.0, 0.0], dtype=np.float64),
#     )
#     np.testing.assert_allclose(
#         household_df["avoided_damage_usd_event"].to_numpy(),
#         np.array([60.0, 0.0, 0.0], dtype=np.float64),
#     )
#
#     assert float(summary_df.loc[0, "avoided_damage_usd_event_total"]) == 60.0
#     assert float(summary_df.loc[0, "cost_usd_upfront_total"]) == 2000.0
#     assert float(summary_df.loc[0, "net_benefit_usd_event_total"]) == -1940.0
#
#
# def test_compute_cba_metrics_for_return_period_rejects_bad_params() -> None:
#     baseline: np.ndarray = np.array([1.0], dtype=np.float64)
#     flood_proofed: np.ndarray = np.array([0.5], dtype=np.float64)
#     adapted_mask: np.ndarray = np.array([True], dtype=bool)
#     costs_upfront: np.ndarray = np.array([1.0], dtype=np.float64)
#
#     try:
#         FloodRiskModule.compute_cba_metrics_for_return_period(
#             damages_baseline_usd_event=baseline,
#             damages_flood_proofed_usd_event=flood_proofed,
#             adapted_mask=adapted_mask,
#             adaptation_costs_usd_upfront=costs_upfront,
#             return_period_years=0.0,
#             loan_duration_years=20,
#         )
#     except ValueError:
#         pass
#     else:
#         raise AssertionError("Expected ValueError for return_period_years=0.0")
