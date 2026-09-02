"""Regression tests for HRL-to-MIRCA crop-calendar construction.

These tests cover the data-quality and temporal-feasibility boundaries used by
``Europe.setup_farmer_crop_calendar_from_HRL``. They intentionally use small
synthetic calendars so failures identify calendar logic rather than requiring
the full Europe data catalog. Strict ``xfail`` cases document suspected defects
without changing production behavior; an eventual fix turns them into an XPASS
failure until the marker is deliberately removed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geb.build.custom_models.europe import (
    _MIRCACalendarCandidatePool,
    _build_mirca_calendar_candidate_pool,
    _calendar_timing_offsets,
    _chunk_slices_without_singletons,
    _crop_area_fit_scores,
    _farmer_area_array_from_farmer_table,
    _fix_365_in_crop_calendar,
    _lexicographic_prefix_best_predecessors,
    _preferred_mirca_calendar_path_if_feasible,
    _prepare_mirca_irrigation_fraction_lookup,
    _prepare_mirca_calendar_source_for_parsing,
    _solve_mirca_calendar_path,
    _weighted_candidate_ranks,
    check_crop_calendar,
    check_crop_calendar_sequence,
)
from geb.build.workflows.crop_calendars import parse_MIRCA_crop_calendar
from geb.build.workflows.farmers import (
    TargetFarm,
    _allocate_lowder_target_counts,
    _count_farm_components_numba,
    _crop_sequence_similarity_numba,
    _largest_remainder_round,
    _switch_timing_similarity_numba,
    _target_cell_counts_from_areas,
    ensure_complete_sequence_in_selected_cells,
    grow_farms_from_raster_cells,
    select_cultivated_cells_by_area,
)


def _compact_calendar(
    crop_id: int,
    planting_day: int,
    duration: int,
    rotation_year: int = 0,
) -> np.ndarray:
    """Create one compact three-row calendar."""
    calendar = np.full((3, 4), -1, dtype=np.int32)
    calendar[0] = [crop_id, planting_day, duration, rotation_year]
    return calendar


def _full_mirca_calendar(
    crop_id: int,
    *,
    is_irrigated: bool,
    planting_day: int,
    duration: int,
    rotation_year: int = 0,
) -> np.ndarray:
    """Create one five-column MIRCA source calendar."""
    calendar = np.full((3, 5), -1, dtype=np.int32)
    calendar[0] = [
        crop_id,
        int(is_irrigated),
        planting_day,
        duration,
        rotation_year,
    ]
    return calendar


def _candidate_pool(
    candidates: list[tuple[int, int, int, int]],
    probabilities: list[float] | None = None,
) -> _MIRCACalendarCandidatePool:
    """Build a candidate pool from crop, planting, duration, and tier tuples."""
    calendars = np.stack(
        [
            _compact_calendar(crop_id, planting_day, duration)
            for crop_id, planting_day, duration, _ in candidates
        ]
    )
    tiers = np.asarray([candidate[3] for candidate in candidates], dtype=np.int8)
    if probabilities is None:
        probability_values = np.empty(len(candidates), dtype=np.float64)
        for tier in np.unique(tiers):
            tier_mask = tiers == tier
            probability_values[tier_mask] = 1.0 / np.count_nonzero(tier_mask)
    else:
        probability_values = np.asarray(probabilities, dtype=np.float64)
    planting = np.asarray(
        [candidate[1] for candidate in candidates],
        dtype=np.int64,
    )
    duration = np.asarray(
        [candidate[2] for candidate in candidates],
        dtype=np.int64,
    )
    return _MIRCACalendarCandidatePool(
        calendars=calendars,
        probabilities=probability_values,
        fallback_tiers=tiers,
        source_units=np.ones(len(candidates), dtype=np.int32),
        earliest_planting=planting,
        latest_harvest=planting + duration,
    )


def _raw_calendar_source(
    *,
    unit_codes: list[object],
    crops: list[object],
) -> pd.DataFrame:
    """Create a raw MIRCA-OS table using the schema received by Europe."""
    n_rows = len(unit_codes)
    assert len(crops) == n_rows
    return pd.DataFrame(
        {
            "unit_code": unit_codes,
            "Crop": crops,
            "Growing_area": np.full(n_rows, 10.0),
            "Planting_Month": np.full(n_rows, 3),
            "Maturity_Month": np.full(n_rows, 8),
        }
    )


def test_prepare_mirca_source_prevents_crop_class_int_casting_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing/unknown raw crops never become NaN crop classes in the parser."""
    source = _raw_calendar_source(
        unit_codes=[1, 1, 1, 1, 2],
        crops=["Wheat", np.nan, "Unsupported crop", "Rice2", np.nan],
    )
    logger = logging.getLogger("test_prepare_mirca_source")

    with caplog.at_level(logging.WARNING):
        prepared = _prepare_mirca_calendar_source_for_parsing(
            source,
            mirca_units=[1],
            source_name="rainfed",
            logger=logger,
        )

    # crop_class is derived by parse_MIRCA_crop_calendar; it is not a raw column.
    assert "crop_class" not in prepared.columns
    assert prepared["Crop"].tolist() == ["Wheat", "Rice2"]
    assert "Discarding 2 rainfed MIRCA-OS calendar row(s)" in caplog.text

    # This call previously failed at crop_class.astype(np.int64). Both retained
    # names map to finite integer classes, so the parser now completes.
    parsed = parse_MIRCA_crop_calendar(
        {},
        prepared,
        [1],
        is_irrigated=False,
    )
    assert sorted(int(entry[1][0, 0]) for entry in parsed[1]) == [0, 2]


def test_prepare_mirca_source_accepts_raw_schema_without_crop_class() -> None:
    """A normal raw source is accepted without requiring a derived column."""
    source = _raw_calendar_source(
        unit_codes=[1, 1],
        crops=["Wheat1", "Others annual3"],
    )

    prepared = _prepare_mirca_calendar_source_for_parsing(
        source,
        mirca_units=[1],
        source_name="rainfed",
        logger=logging.getLogger("test_raw_mirca_schema"),
    )

    pd.testing.assert_frame_equal(prepared.reset_index(drop=True), source)


def test_prepare_mirca_source_ignores_invalid_out_of_scope_crop() -> None:
    """Bad crop values outside the selected units do not contaminate output."""
    source = _raw_calendar_source(
        unit_codes=[1, 2, np.nan, "not-a-unit"],
        crops=["Barley", np.nan, "Rye", "Millet"],
    )

    prepared = _prepare_mirca_calendar_source_for_parsing(
        source,
        mirca_units=[1],
        source_name="irrigated",
        logger=logging.getLogger("test_out_of_scope_mirca_rows"),
    )

    assert prepared[["unit_code", "Crop"]].to_dict("records") == [
        {"unit_code": 1, "Crop": "Barley"}
    ]


@pytest.mark.parametrize("missing_column", ["unit_code", "Crop"])
def test_prepare_mirca_source_requires_identifier_columns(
    missing_column: str,
) -> None:
    """A changed source schema fails with an explicit column diagnostic."""
    columns = {"unit_code": [1], "Crop": ["Wheat"]}
    del columns[missing_column]

    with pytest.raises(ValueError, match=missing_column):
        _prepare_mirca_calendar_source_for_parsing(
            pd.DataFrame(columns),
            mirca_units=[1],
            source_name="rainfed",
            logger=logging.getLogger("test_missing_mirca_column"),
        )


def test_prepare_mirca_source_requires_dataframe() -> None:
    """Non-tabular catalog output is rejected before parsing."""
    with pytest.raises(TypeError, match="pandas DataFrame"):
        _prepare_mirca_calendar_source_for_parsing(
            np.array([[1, 2]]),  # type: ignore[arg-type]
            mirca_units=[1],
            source_name="rainfed",
            logger=logging.getLogger("test_non_dataframe_mirca_source"),
        )


def test_fix_365_clamps_only_crop_duration() -> None:
    """A documented 365-day duration is made compatible with the model year."""
    full_calendar = _full_mirca_calendar(
        1,
        is_irrigated=False,
        planting_day=100,
        duration=365,
    )

    adjusted = _fix_365_in_crop_calendar({1: [(10.0, full_calendar)]})

    assert adjusted[1][0][1][0, 3] == 364


def test_fix_365_rejects_365_in_another_column() -> None:
    """A 365 value outside duration indicates a malformed calendar."""
    malformed = _full_mirca_calendar(
        1,
        is_irrigated=False,
        planting_day=365,
        duration=100,
    )

    with pytest.raises(ValueError, match="Found 365 outside column 3"):
        _fix_365_in_crop_calendar({1: [(10.0, malformed)]})


def test_calendar_timing_offsets_include_rotation_year() -> None:
    """Compact rotation years shift both planting and harvest by 365 days."""
    calendars = _compact_calendar(1, 10, 20, rotation_year=1)[None, :, :]

    active, earliest, latest = _calendar_timing_offsets(calendars)

    np.testing.assert_array_equal(active, [True])
    np.testing.assert_array_equal(earliest, [375])
    np.testing.assert_array_equal(latest, [395])


@pytest.mark.parametrize(
    "calendar_row",
    [
        [1, -1, 20, 0],
        [1, 10, -1, 0],
        [1, 10, 20, -1],
    ],
)
def test_calendar_timing_offsets_reject_negative_active_timing(
    calendar_row: list[int],
) -> None:
    """Negative timing is allowed only in inactive sentinel rows."""
    calendars = np.full((1, 3, 4), -1, dtype=np.int32)
    calendars[0, 0] = calendar_row

    with pytest.raises(ValueError, match="non-negative"):
        _calendar_timing_offsets(calendars)


def test_calendar_sequence_allows_same_day_harvest_and_planting() -> None:
    """Harvest on 1 January may be followed by planting on that same day."""
    stack = np.stack(
        [
            _compact_calendar(1, 300, 65),
            _compact_calendar(2, 0, 100),
        ]
    )[:, None, :, :]

    summary = check_crop_calendar_sequence(stack, [2017, 2018])

    assert summary == {
        "checked_transitions": 1,
        "same_day_transitions": 1,
        "minimum_gap_days": 0,
    }


def test_calendar_sequence_rejects_planting_before_previous_harvest() -> None:
    """A next-year calendar cannot overlap the preceding assigned crop."""
    stack = np.stack(
        [
            _compact_calendar(1, 300, 66),
            _compact_calendar(2, 0, 100),
        ]
    )[:, None, :, :]

    with pytest.raises(AssertionError, match="sequence violation"):
        check_crop_calendar_sequence(stack, [2017, 2018])


def test_fallow_year_preserves_last_harvest_constraint() -> None:
    """Fallow does not erase a long crop whose harvest reaches a later year."""
    stack = np.stack(
        [
            _compact_calendar(1, 300, 500),
            np.full((3, 4), -1, dtype=np.int32),
            _compact_calendar(2, 0, 100),
        ]
    )[:, None, :, :]

    with pytest.raises(AssertionError, match="sequence violation"):
        check_crop_calendar_sequence(stack, [2017, 2018, 2019])


def test_weighted_candidate_ranks_lazy_matches_complete_order() -> None:
    """Lazy local ranking is identical and leaves unused fallbacks unranked."""
    pool = _candidate_pool(
        [
            (1, 100, 100, 0),
            (1, 110, 100, 0),
            (1, 120, 100, 1),
            (1, 130, 100, 2),
            (1, 140, 100, 2),
        ],
        probabilities=[0.7, 0.3, 1.0, 0.4, 0.6],
    )
    arguments = {
        "random_seed": 42,
        "farmer_id": 17,
        "lookup_unit": 8,
        "main_crop": 1,
        "is_irrigated": False,
    }

    complete = _weighted_candidate_ranks(pool, **arguments)
    lazy = np.full(pool.probabilities.size, -1, dtype=np.int32)
    _weighted_candidate_ranks(
        pool,
        **arguments,
        ranks=lazy,
        maximum_fallback_tier=0,
    )

    np.testing.assert_array_equal(lazy[pool.fallback_tiers == 0], complete[:2])
    assert np.all(lazy[pool.fallback_tiers > 0] == -1)

    _weighted_candidate_ranks(
        pool,
        **arguments,
        ranks=lazy,
        maximum_fallback_tier=2,
    )
    np.testing.assert_array_equal(lazy, complete)


def test_preferred_path_fast_path_matches_dynamic_programming() -> None:
    """A feasible independently preferred sequence is the exact DP optimum."""
    first_pool = _candidate_pool([(1, 100, 100, 0), (1, 110, 120, 0)])
    second_pool = _candidate_pool([(2, 90, 100, 0), (2, 120, 100, 0)])
    ranks = [
        np.array([1, 0], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
    ]
    active_years = np.array([0, 1], dtype=np.int32)
    years = np.array([2017, 2018], dtype=np.int32)

    preferred, checks = _preferred_mirca_calendar_path_if_feasible(
        farmer_pools=[first_pool, second_pool],
        farmer_ranks=ranks,
        active_year_indices=active_years,
        years=years,
    )
    solved, failure_position, _, _ = _solve_mirca_calendar_path(
        farmer_pools=[first_pool, second_pool],
        farmer_ranks=ranks,
        active_year_indices=active_years,
        years=years,
        maximum_allowed_tier=0,
    )

    assert checks == 1
    assert failure_position is None
    np.testing.assert_array_equal(preferred, solved)


def test_dynamic_programming_uses_fallback_to_escape_temporal_dead_end() -> None:
    """A broader tier can replace an earlier late-harvest calendar."""
    first_pool = _candidate_pool(
        [
            (1, 0, 500, 0),
            (1, 0, 100, 1),
        ]
    )
    second_pool = _candidate_pool([(2, 20, 100, 0)])
    ranks = [
        np.array([0, 0], dtype=np.int32),
        np.array([0], dtype=np.int32),
    ]
    active_years = np.array([0, 1], dtype=np.int32)
    years = np.array([2017, 2018], dtype=np.int32)

    local_only, failure_position, _, _ = _solve_mirca_calendar_path(
        farmer_pools=[first_pool, second_pool],
        farmer_ranks=ranks,
        active_year_indices=active_years,
        years=years,
        maximum_allowed_tier=0,
    )
    with_fallback, _, _, _ = _solve_mirca_calendar_path(
        farmer_pools=[first_pool, second_pool],
        farmer_ranks=ranks,
        active_year_indices=active_years,
        years=years,
        maximum_allowed_tier=1,
    )

    assert local_only is None
    assert failure_position == 1
    np.testing.assert_array_equal(with_fallback, [1, 0])


def test_candidate_pool_collapses_duplicate_calendars_within_tier() -> None:
    """Duplicate unit calendars do not multiply the path-search state space."""
    local = _full_mirca_calendar(
        1,
        is_irrigated=False,
        planting_day=100,
        duration=120,
    )
    local_irrigated = _full_mirca_calendar(
        1,
        is_irrigated=True,
        planting_day=105,
        duration=115,
    )
    other_unit = _full_mirca_calendar(
        1,
        is_irrigated=False,
        planting_day=90,
        duration=100,
    )
    calendar_source = {
        7: [
            (2.0, local.copy()),
            (3.0, local.copy()),
            (4.0, local_irrigated),
        ],
        8: [(5.0, other_unit)],
    }

    pool = _build_mirca_calendar_candidate_pool(
        calendar_source,
        lookup_unit=7,
        main_crop=1,
        is_irrigated=False,
    )

    assert pool.calendars.shape == (3, 3, 4)
    np.testing.assert_array_equal(pool.fallback_tiers, [0, 1, 2])
    np.testing.assert_allclose(pool.probabilities, [1.0, 1.0, 1.0])


def test_candidate_pool_rejects_malformed_calendar_shape() -> None:
    """Unexpected MIRCA row counts fail before entering path selection."""
    malformed = np.full((2, 5), -1, dtype=np.int32)
    malformed[0] = [1, 0, 100, 120, 0]

    with pytest.raises(ValueError, match=r"shape \(3, 4\)"):
        _build_mirca_calendar_candidate_pool(
            {7: [(1.0, malformed)]},
            lookup_unit=7,
            main_crop=1,
            is_irrigated=False,
        )


def test_lexicographic_prefix_predecessors_match_brute_force() -> None:
    """The accelerated DP prefix minima retain the exact cost ordering."""
    sorted_positions = np.array([2, 0, 3, 1], dtype=np.int32)
    maximum_tier = np.array([0, 2, 1, 1], dtype=np.int16)
    tier_sum = np.array([1, 0, 0, 0], dtype=np.int32)
    rank_sum = np.array([2, 0, 1, 1], dtype=np.int32)
    candidate_tier = 1

    actual = _lexicographic_prefix_best_predecessors(
        sorted_previous_positions=sorted_positions,
        maximum_tier=maximum_tier,
        tier_sum=tier_sum,
        rank_sum=rank_sum,
        candidate_tier=candidate_tier,
    )
    expected = np.empty(sorted_positions.size, dtype=np.int32)
    for prefix_end in range(sorted_positions.size):
        prefix = sorted_positions[: prefix_end + 1]
        expected[prefix_end] = min(
            (int(position) for position in prefix),
            key=lambda position: (
                max(int(maximum_tier[position]), candidate_tier),
                int(tier_sum[position]),
                int(rank_sum[position]),
                position,
            ),
        )

    np.testing.assert_array_equal(actual, expected)


def test_irrigation_fraction_lookup_normalizes_nan_once() -> None:
    """NaN MIRCA fractions become zero and cell totals remain aligned."""
    rainfed = xr.DataArray(
        np.array([[[1.0, np.nan]], [[3.0, 4.0]]]),
        dims=("crop", "y", "x"),
    )
    irrigated = xr.DataArray(
        np.array([[[1.0, 2.0]], [[1.0, np.nan]]]),
        dims=("crop", "y", "x"),
    )

    lookup = _prepare_mirca_irrigation_fraction_lookup(rainfed, irrigated)

    np.testing.assert_array_equal(
        lookup.rainfed_values,
        np.array([[1.0, 0.0], [3.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        lookup.irrigated_values,
        np.array([[1.0, 2.0], [1.0, 0.0]]),
    )
    np.testing.assert_array_equal(lookup.total_rainfed_by_cell, [4.0, 4.0])
    np.testing.assert_array_equal(lookup.total_irrigated_by_cell, [2.0, 2.0])


def test_irrigation_fraction_lookup_rejects_misaligned_shapes() -> None:
    """Rainfed and irrigated stacks must describe the same MIRCA grid."""
    rainfed = xr.DataArray(
        np.ones((2, 1, 2)),
        dims=("crop", "y", "x"),
    )
    irrigated = xr.DataArray(
        np.ones((2, 1, 3)),
        dims=("crop", "y", "x"),
    )

    with pytest.raises(ValueError, match="same shape"):
        _prepare_mirca_irrigation_fraction_lookup(rainfed, irrigated)


@pytest.mark.parametrize(
    ("length", "chunk_size", "expected_bounds"),
    [
        (0, 4, []),
        (1, 4, [(0, 1)]),
        (8, 4, [(0, 4), (4, 8)]),
        (9, 4, [(0, 4), (4, 9)]),
    ],
)
def test_chunk_slices_never_leave_a_singleton_tail(
    length: int,
    chunk_size: int,
    expected_bounds: list[tuple[int, int]],
) -> None:
    """A one-cell raster tail is folded into its preceding chunk."""
    slices = _chunk_slices_without_singletons(length, chunk_size)
    assert [(item.start, item.stop) for item in slices] == expected_bounds


def test_crop_area_fit_scores_exclude_fallow_from_fit() -> None:
    """Negative crop sentinels cannot improve or worsen positive-crop fit."""
    diagnostics = pd.DataFrame(
        {
            "crop_code": [-1, 1, 2],
            "source_area_m2": [1_000.0, 100.0, 0.0],
            "assigned_area_m2": [0.0, 80.0, 20.0],
        }
    )

    scores = _crop_area_fit_scores(diagnostics)

    assert scores["source_area_m2"] == pytest.approx(100.0)
    assert scores["assigned_area_m2"] == pytest.approx(100.0)
    assert scores["total_area_fit_score"] == pytest.approx(100.0)
    assert scores["crop_share_fit_score"] == pytest.approx(80.0)
    assert scores["crop_area_fit_score"] == pytest.approx(80.0)


def test_farmer_area_lookup_requires_every_compact_farmer() -> None:
    """Missing compact farmer IDs are rejected instead of receiving NaN area."""
    farmers = pd.DataFrame(
        {
            "farmer_id": [0, 2],
            "area_m2": [10.0, 30.0],
        }
    )

    with pytest.raises(ValueError, match=r"Missing examples: \[1\]"):
        _farmer_area_array_from_farmer_table(farmers, n_farmers=3)


def test_crop_sequence_similarity_distinguishes_fallow_and_missing() -> None:
    """Missing years are excluded while fallow matches receive reduced weight."""
    first = np.array([1, -1, -2, 2], dtype=np.int32)
    second = np.array([1, -1, 3, 4], dtype=np.int32)

    similarity = _crop_sequence_similarity_numba(
        first,
        second,
        missing_value=-2,
        fallow_value=-1,
        min_valid_overlap=2,
        fallow_match_weight=0.35,
    )

    # Three comparable years, weighted matches 1 + 0.35, and 3/4 coverage.
    assert similarity == pytest.approx((1.35 / 3.0) * 0.75)
    assert (
        _crop_sequence_similarity_numba(
            first,
            second,
            missing_value=-2,
            fallow_value=-1,
            min_valid_overlap=4,
        )
        == 0.0
    )


def test_switch_timing_similarity_covers_switch_and_no_switch_cases() -> None:
    """Switch timing uses Jaccard overlap and a neutral constant-sequence score."""
    first = np.array([1, 2, 2, 3], dtype=np.int32)
    second = np.array([1, 2, 3, 3], dtype=np.int32)
    constant = np.array([4, 4, 4], dtype=np.int32)

    assert _switch_timing_similarity_numba(first, second, -2) == pytest.approx(1 / 3)
    assert _switch_timing_similarity_numba(constant, constant, -2) == 0.5
    assert (
        _switch_timing_similarity_numba(
            np.array([-2, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            -2,
        )
        == 0.0
    )


def test_select_cultivated_cells_is_deterministic_and_meets_area_target() -> None:
    """Equal scores use flat-index order and selection never undershoots area."""
    scores = np.array([[3.0, 2.0, 2.0, np.inf]])
    eligible = np.ones_like(scores, dtype=bool)
    areas = np.array([[1.0, 2.0, 5.0, 100.0]])

    selected = select_cultivated_cells_by_area(
        scores,
        eligible,
        areas,
        target_area_m2=4.0,
    )

    np.testing.assert_array_equal(selected, [[True, True, True, False]])
    assert float(areas[selected].sum()) >= 4.0


def test_target_cell_counts_are_positive_exact_and_stable_on_ties() -> None:
    """Continuous equal-area targets deterministically exhaust raster cells."""
    targets = [
        TargetFarm(target_area_m2=100.0, size_class=f"class-{index}")
        for index in range(3)
    ]

    ordered, counts = _target_cell_counts_from_areas(targets, n_cells=5)

    assert [target.size_class for target in ordered] == [
        "class-0",
        "class-1",
        "class-2",
    ]
    np.testing.assert_array_equal(counts, [2, 2, 1])
    assert np.all(counts > 0)
    assert int(counts.sum()) == 5


def test_component_count_uses_four_connectivity_and_compact_ids() -> None:
    """Diagonal contacts remain separate parcels; edge contacts are joined."""
    farms = np.array(
        [
            [0, -1, 0],
            [-1, -1, -1],
            [1, 1, 0],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(_count_farm_components_numba(farms, 2), [3, 1])


def test_largest_remainder_round_is_exact_and_stable_on_ties() -> None:
    """Equal remainders favor lower indices and preserve the requested total."""
    rounded = _largest_remainder_round(np.array([1.0, 1.0, 1.0]), target_sum=2)

    np.testing.assert_array_equal(rounded, [1, 1, 0])
    assert rounded.dtype == np.int64
    assert int(rounded.sum()) == 2


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, -1.0]),
        np.array([1.0, np.nan]),
        np.array([[1.0, 2.0]]),
    ],
)
def test_largest_remainder_round_rejects_invalid_weights(values: np.ndarray) -> None:
    """Invalid Lowder count weights fail before integer allocation."""
    with pytest.raises(ValueError):
        _largest_remainder_round(values, target_sum=2)


def test_lowder_count_allocation_retains_largest_area_classes_when_capped() -> None:
    """A hard two-farm cap retains the two most important area classes."""
    counts = _allocate_lowder_target_counts(
        expected_n_farms=np.array([100.0, 1.0, 1.0]),
        target_bin_area_m2=np.array([1.0, 100.0, 50.0]),
        target_sum=2,
    )

    np.testing.assert_array_equal(counts, [0, 1, 1])


def test_lowder_count_allocation_represents_all_supported_classes() -> None:
    """Every supported class gets a farm before residual counts are allocated."""
    counts = _allocate_lowder_target_counts(
        expected_n_farms=np.array([100.0, 1.0, 1.0]),
        target_bin_area_m2=np.array([1.0, 100.0, 50.0]),
        target_sum=4,
    )

    np.testing.assert_array_equal(counts, [2, 1, 1])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Compact calendars store duration in column 2, but the current validator "
        "compares rotation year in column 3. Production behavior is intentionally "
        "unchanged pending a separate fix decision."
    ),
)
def test_known_issue_calendar_validation_compares_planting_and_duration() -> None:
    """Duplicate planting-duration pairs should be rejected across crop rows."""
    calendars = np.full((1, 3, 4), -1, dtype=np.int32)
    calendars[0, 0] = [1, 100, 80, 0]
    calendars[0, 1] = [2, 100, 80, 1]

    with pytest.raises(AssertionError, match="Duplicate active"):
        check_crop_calendar(calendars)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Duplicate farmer IDs currently overwrite earlier area rows. Production "
        "behavior is intentionally unchanged pending a separate validation change."
    ),
)
def test_known_issue_farmer_area_lookup_rejects_duplicate_ids() -> None:
    """One and only one area row should exist for every compact farmer ID."""
    farmers = pd.DataFrame(
        {
            "farmer_id": [0, 0, 1],
            "area_m2": [10.0, 20.0, 30.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        _farmer_area_array_from_farmer_table(farmers, n_farmers=2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Sequence repair validates the inserted candidate area but not existing "
        "selected-cell areas. Production behavior is intentionally unchanged."
    ),
)
def test_known_issue_sequence_repair_rejects_nonfinite_selected_area() -> None:
    """A selected agricultural domain must not retain non-finite cell area."""
    with pytest.raises(ValueError, match="finite"):
        ensure_complete_sequence_in_selected_cells(
            selected_mask=np.array([[True, False]]),
            complete_sequence_mask=np.array([[False, True]]),
            selection_score=np.array([[1.0, 0.5]]),
            cell_area_m2=np.array([[np.nan, 1.0]]),
            target_area_m2=1.0,
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Farm-growth weights are normalized when their sum is positive, even if "
        "an individual weight is negative. Production behavior is unchanged."
    ),
)
def test_known_issue_farm_growth_rejects_negative_individual_weight() -> None:
    """Every farm-growth score component should have a non-negative weight."""
    with pytest.raises(ValueError, match="non-negative"):
        grow_farms_from_raster_cells(
            cultivated_mask=np.array([[True]], dtype=bool),
            crop_sequences=np.array([[[1]], [[1]]], dtype=np.int32),
            cell_area_m2=np.array([[1.0]]),
            target_farms=[TargetFarm(target_area_m2=1.0, size_class="test")],
            distance_weight=-1.0,
            crop_sequence_weight=2.0,
            switch_timing_weight=0.0,
        )
