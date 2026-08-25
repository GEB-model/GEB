"""Tests for the current raster-based farmer workflows.

The HRL/Lowder tests in this module intentionally follow the current
``setup_create_farms_from_HRL_lowder`` architecture:

1. select a static agricultural domain on the model raster;
2. keep HRL ``-2`` as outside/missing and convert CTY ``0`` to fallow ``-1`` only
   inside that selected domain;
3. grow Lowder-guided farms directly from model-grid cells; and
4. assign complete original multi-year crop sequences jointly.

Older field-boundary, dominant-field-crop, and CPSCT-based farm-construction tests were
removed because those steps are no longer part of ``setup_create_farms_from_HRL_lowder``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from geb.build.data_catalog import DataCatalog
from geb.build.workflows.farmers import (
    TargetFarm,
    assign_farmer_sequences_to_area_targets,
    create_farms_numba,
    create_lowder_target_farm_areas,
    ensure_complete_sequence_in_selected_cells,
    grow_farms_from_raster_cells,
    relax_lowder_targets_for_sequence_fit,
    select_cultivated_cells_by_area,
)

from ...testconfig import IN_GITHUB_ACTIONS


LOWDER_SIZE_CLASS_BOUNDARIES_M2: dict[str, tuple[float, float]] = {
    "< 1 Ha": (0.0, 10_000.0),
    "1 - 2 Ha": (10_000.0, 20_000.0),
    "2 - 5 Ha": (20_000.0, 50_000.0),
    "5 - 10 Ha": (50_000.0, 100_000.0),
    "10 - 20 Ha": (100_000.0, 200_000.0),
    "20 - 50 Ha": (200_000.0, 500_000.0),
    "50 - 100 Ha": (500_000.0, 1_000_000.0),
    "100 - 200 Ha": (1_000_000.0, 2_000_000.0),
    "200 - 500 Ha": (2_000_000.0, 5_000_000.0),
    "500 - 1000 Ha": (5_000_000.0, 10_000_000.0),
    "> 1000 Ha": (10_000_000.0, np.inf),
}

_HRL_FALLOW = -1
_HRL_MISSING = -2
_HRL_NO_CROPLAND = 0


def _positive_crop_area_targets(
    crop_sequences: np.ndarray,
    cell_area_m2: np.ndarray,
    active_mask: np.ndarray,
) -> list[dict[int, float]]:
    """Build simple positive-crop area targets for synthetic sequence tests."""
    targets: list[dict[int, float]] = []
    for year_values in crop_sequences:
        year_targets: dict[int, float] = {}
        active_values = year_values[active_mask]
        active_areas = cell_area_m2[active_mask]
        for crop_code in np.unique(active_values[active_values > 0]):
            year_targets[int(crop_code)] = float(
                active_areas[active_values == crop_code].sum()
            )
        targets.append(year_targets)
    return targets


def _apply_selected_hrl_domain_semantics(
    crop_sequences: np.ndarray,
    cultivated_mask: np.ndarray,
) -> np.ndarray:
    """Apply the same 0/-1/-2 semantics as the Europe Lowder setup."""
    crop_sequences = np.asarray(crop_sequences, dtype=np.int32).copy()
    selected_3d = cultivated_mask[None, :, :]
    crop_sequences[selected_3d & (crop_sequences == _HRL_NO_CROPLAND)] = _HRL_FALLOW
    crop_sequences[:, ~cultivated_mask] = _HRL_MISSING
    return crop_sequences


def _complete_original_sequence_mask(
    crop_sequences: np.ndarray,
    cultivated_mask: np.ndarray,
) -> np.ndarray:
    """Mirror the pre-assignment completeness guard in the Europe workflow."""
    return (
        cultivated_mask
        & ~np.any(crop_sequences == _HRL_MISSING, axis=0)
        & np.any(crop_sequences > 0, axis=0)
    )


def test_create_farms_numba_no_farms() -> None:
    """When there are no farms and no cultivated land, the result is all -1."""
    cultivated_land = np.zeros((3, 3), dtype=np.int32)
    ids = np.array([], dtype=np.int32)
    farm_sizes = np.array([], dtype=np.int32)

    np.random.seed(0)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    assert farms.shape == cultivated_land.shape
    assert np.all(farms == -1)


def test_create_farms_numba_some_farmers() -> None:
    """Allocate a small set of farms across a contiguous cultivated block."""
    cultivated_land = np.zeros((4, 5), dtype=np.int32)
    cultivated_land[0:2, 0:3] = 1
    ids = np.array([1, 2], dtype=np.int32)
    farm_sizes = np.array([2, 4], dtype=np.int32)

    np.random.seed(42)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    assert farms.shape == cultivated_land.shape
    assert np.all(farms[cultivated_land == 0] == -1)
    assigned_ids = np.unique(farms[cultivated_land == 1])
    assigned_ids = assigned_ids[assigned_ids != -1]
    assert set(assigned_ids.tolist()) == {1, 2}
    assert int(np.count_nonzero(farms == 1)) == 2
    assert int(np.count_nonzero(farms == 2)) == 4


def test_create_farms_numba_single_farmer_single_cell() -> None:
    """Single farmer owning exactly one cultivated cell."""
    cultivated_land = np.zeros((2, 2), dtype=np.int32)
    cultivated_land[1, 1] = 1
    ids = np.array([99], dtype=np.int32)
    farm_sizes = np.array([1], dtype=np.int32)

    np.random.seed(123)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    assert farms[1, 1] == 99
    other = np.ones_like(cultivated_land, dtype=bool)
    other[1, 1] = False
    assert np.all(farms[other] == -1)


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Downloads the Lowder workbook and iterates over all ISO3 groups.",
)
def test_create_lowder_target_farms_for_all_lowder_regions() -> None:
    """Current Lowder target creation must work for every available ISO3 group."""
    logger = logging.getLogger("test_create_lowder_target_farms_for_all_lowder_regions")
    lowder_farm_sizes = (
        DataCatalog(logger=logger).fetch("lowder_farm_size_distribution").read()
    )
    cultivated_area_m2 = 10_000_000.0
    mean_cell_area_m2 = 2_500.0
    n_available_cells = int(cultivated_area_m2 / mean_cell_area_m2)

    processed_iso3: list[str] = []
    for iso3, region_farm_sizes in tqdm(
        lowder_farm_sizes.groupby("ISO3"), desc="Processing ISO3 groups"
    ):
        region_farm_sizes = region_farm_sizes.drop(
            ["Country", "Census Year", "Total"], axis=1
        )
        assert len(region_farm_sizes) == 2, (
            f"Expected two Lowder rows for {iso3}, found {len(region_farm_sizes)}."
        )

        targets = create_lowder_target_farm_areas(
            region_farm_sizes=region_farm_sizes,
            size_class_boundaries=LOWDER_SIZE_CLASS_BOUNDARIES_M2,
            cultivated_area_m2=cultivated_area_m2,
            iso3=str(iso3),
            logger=logger,
            random_seed=42,
            minimum_cells_per_farm=1.0,
            mean_cell_area_m2=mean_cell_area_m2,
        )

        assert targets, f"Expected at least one target farm for {iso3}."
        assert len(targets) <= n_available_cells
        assert all(target.target_area_m2 > 0.0 for target in targets)
        assert all(
            target.size_class in LOWDER_SIZE_CLASS_BOUNDARIES_M2 for target in targets
        )
        assert np.isclose(
            sum(target.target_area_m2 for target in targets),
            cultivated_area_m2,
            rtol=1e-10,
            atol=1e-3,
        )
        processed_iso3.append(str(iso3))

    assert processed_iso3


def test_create_lowder_targets_respects_hard_cell_count_cap() -> None:
    """A strong Lowder count reduction must never leave more farms than cells."""
    region_farm_sizes = pd.DataFrame(
        [
            {
                "Holdings/ agricultural area": "Holdings",
                "ISO3": "TST",
                "< 1 Ha": 10_000.0,
            },
            {
                "Holdings/ agricultural area": "Agricultural area (Ha)",
                "ISO3": "TST",
                "< 1 Ha": 100.0,
            },
        ]
    )
    logger = logging.getLogger(
        "test_create_lowder_targets_respects_hard_cell_count_cap"
    )

    # Database statistics imply ~100 farms after regional area scaling, but four
    # 2,500 m² cells can physically support only four farms. The old rounding helper
    # reduced the count by only one and would therefore fail later during farm growth.
    targets = create_lowder_target_farm_areas(
        region_farm_sizes=region_farm_sizes,
        size_class_boundaries={"< 1 Ha": (0.0, 10_000.0)},
        cultivated_area_m2=10_000.0,
        iso3="TST",
        logger=logger,
        random_seed=42,
        minimum_cells_per_farm=1.0,
        mean_cell_area_m2=2_500.0,
    )

    assert len(targets) == 4
    assert np.isclose(
        sum(target.target_area_m2 for target in targets),
        10_000.0,
        rtol=0.0,
        atol=1e-6,
    )


def test_relax_lowder_targets_preserves_area_and_cell_capacity() -> None:
    """Target splitting may add farms but must preserve area and remain feasible."""
    targets = [
        TargetFarm(target_area_m2=40_000.0, size_class="large"),
        TargetFarm(target_area_m2=20_000.0, size_class="medium"),
        TargetFarm(target_area_m2=10_000.0, size_class="small"),
    ]
    relaxed = relax_lowder_targets_for_sequence_fit(
        targets,
        extra_farm_fraction=2 / 3,
        n_available_cells=20,
        mean_cell_area_m2=2_500.0,
        minimum_cells_per_farm=1.0,
    )

    assert len(relaxed) == 5
    assert len(relaxed) <= 20
    assert all(target.target_area_m2 > 0.0 for target in relaxed)
    assert np.isclose(
        sum(target.target_area_m2 for target in relaxed),
        sum(target.target_area_m2 for target in targets),
        rtol=0.0,
        atol=1e-6,
    )


def test_sequence_assignment_reproduces_no_complete_original_sequence_error() -> None:
    """Reproduce the historical exception with the current HRL sentinel semantics.

    Every active pixel has at least one positive crop observation, so it can look
    agriculturally relevant in individual years. However, every pixel also contains
    ``-2`` in at least one requested year. Therefore there is no *complete original*
    multi-year sequence from which the regional fallback pool can be constructed.
    """
    farm_values = np.array([[0, 0], [1, 1]], dtype=np.int32)
    crop_sequences = np.array(
        [
            [[1110, _HRL_MISSING], [1120, _HRL_MISSING]],
            [[_HRL_MISSING, 1110], [_HRL_MISSING, 1120]],
            [[1130, 1130], [1130, 1130]],
        ],
        dtype=np.int32,
    )
    cell_area_m2 = np.full((2, 2), 2_500.0, dtype=np.float64)
    farmer_areas_m2 = np.array([5_000.0, 5_000.0], dtype=np.float64)
    targets = _positive_crop_area_targets(
        crop_sequences, cell_area_m2, farm_values >= 0
    )

    with pytest.raises(
        ValueError,
        match="No complete original crop sequence is available in the selected region",
    ):
        assign_farmer_sequences_to_area_targets(
            farm_values=farm_values,
            crop_sequences=crop_sequences,
            cell_area_m2=cell_area_m2,
            farmer_areas_m2=farmer_areas_m2,
            target_crop_areas_per_year=targets,
            missing_code=_HRL_MISSING,
            fallow_code=_HRL_FALLOW,
        )


def test_static_selection_repairs_excluded_complete_original_sequence() -> None:
    """Selection must retain a regional complete sequence instead of skipping."""
    raw_crop_sequences = np.array(
        [
            [[1110, 1110, 1110]],
            [[1120, _HRL_MISSING, _HRL_NO_CROPLAND]],
            [[_HRL_MISSING, 1130, _HRL_NO_CROPLAND]],
        ],
        dtype=np.int32,
    )
    cell_area_m2 = np.full((1, 3), 2_500.0, dtype=np.float64)
    region_mask = np.ones((1, 3), dtype=bool)
    valid_count = np.count_nonzero(raw_crop_sequences > 0, axis=0)
    eligible_mask = region_mask & (valid_count > 0)
    valid_frequency = valid_count / float(raw_crop_sequences.shape[0])
    mean_fraction = np.ones((1, 3), dtype=np.float64)
    selection_score = 0.80 * valid_frequency + 0.20 * mean_fraction

    regional_complete = (
        eligible_mask
        & ~np.any(raw_crop_sequences == _HRL_MISSING, axis=0)
        & np.any(raw_crop_sequences > 0, axis=0)
    )
    assert regional_complete.tolist() == [[False, False, True]]

    cultivated_mask = select_cultivated_cells_by_area(
        selection_score,
        eligible_mask,
        cell_area_m2,
        target_area_m2=5_000.0,
    )
    assert cultivated_mask.tolist() == [[True, True, False]]

    repaired, changed, used_extra_cell = ensure_complete_sequence_in_selected_cells(
        cultivated_mask,
        regional_complete,
        selection_score,
        cell_area_m2,
        target_area_m2=5_000.0,
    )
    assert changed
    assert not used_extra_cell
    assert repaired.tolist() == [[True, False, True]]
    assert np.isclose(cell_area_m2[repaired].sum(), 5_000.0)

    selected_sequences = _apply_selected_hrl_domain_semantics(
        raw_crop_sequences, repaired
    )
    complete_selected = _complete_original_sequence_mask(selected_sequences, repaired)
    assert complete_selected.tolist() == [[False, False, True]]

    # The repaired domain must now be usable by the same sequence helper that raised
    # the historical exception before production gained this repair.
    farm_values = np.array([[0, -1, 1]], dtype=np.int32)
    farmer_areas_m2 = np.array([2_500.0, 2_500.0], dtype=np.float64)
    targets = _positive_crop_area_targets(selected_sequences, cell_area_m2, repaired)
    assigned, quality, _ = assign_farmer_sequences_to_area_targets(
        farm_values=farm_values,
        crop_sequences=selected_sequences,
        cell_area_m2=cell_area_m2,
        farmer_areas_m2=farmer_areas_m2,
        target_crop_areas_per_year=targets,
        missing_code=_HRL_MISSING,
        fallow_code=_HRL_FALLOW,
    )
    assert not np.any(assigned == _HRL_MISSING)
    assert quality["crop_sequence_is_original"].all()


def test_complete_sequence_repair_adds_cell_when_swap_breaks_area_target() -> None:
    """Unequal cell areas may require one extra complete-sequence cell."""
    selected = np.array([[True, False]], dtype=bool)
    complete = np.array([[False, True]], dtype=bool)
    scores = np.array([[1.0, 0.5]], dtype=np.float64)
    cell_area_m2 = np.array([[5_000.0, 1_000.0]], dtype=np.float64)

    repaired, changed, used_extra_cell = ensure_complete_sequence_in_selected_cells(
        selected,
        complete,
        scores,
        cell_area_m2,
        target_area_m2=5_000.0,
    )
    assert changed
    assert used_extra_cell
    assert repaired.tolist() == [[True, True]]
    assert np.isclose(cell_area_m2[repaired].sum(), 6_000.0)


def test_complete_sequence_repair_is_noop_when_already_represented() -> None:
    """Normal regions must retain the original static selection exactly."""
    selected = np.array([[True, True, False]], dtype=bool)
    complete = np.array([[False, True, True]], dtype=bool)
    scores = np.array([[1.0, 0.9, 0.8]], dtype=np.float64)
    cell_area_m2 = np.full((1, 3), 2_500.0, dtype=np.float64)

    repaired, changed, used_extra_cell = ensure_complete_sequence_in_selected_cells(
        selected,
        complete,
        scores,
        cell_area_m2,
        target_area_m2=5_000.0,
    )
    np.testing.assert_array_equal(repaired, selected)
    assert not changed
    assert not used_extra_cell


def test_sequence_assignment_preserves_multiple_local_original_sequences() -> None:
    """An exactly representable local sequence mix should need no fallback."""
    farm_values = np.array(
        [
            [0, 0, 1, 1],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    sequences_by_farmer = np.array(
        [
            [1110, 1120, 1130],
            [1110, 1120, 1140],
            [1150, _HRL_FALLOW, 1130],
            [1150, _HRL_FALLOW, 1140],
        ],
        dtype=np.int32,
    )
    crop_sequences = np.empty((3, 2, 4), dtype=np.int32)
    for farmer_id, sequence in enumerate(sequences_by_farmer):
        crop_sequences[:, farm_values == farmer_id] = sequence[:, None]

    cell_area_m2 = np.full((2, 4), 2_500.0, dtype=np.float64)
    farmer_areas_m2 = np.full(4, 5_000.0, dtype=np.float64)
    targets = _positive_crop_area_targets(
        crop_sequences, cell_area_m2, farm_values >= 0
    )

    assigned, quality, diagnostics = assign_farmer_sequences_to_area_targets(
        farm_values=farm_values,
        crop_sequences=crop_sequences,
        cell_area_m2=cell_area_m2,
        farmer_areas_m2=farmer_areas_m2,
        target_crop_areas_per_year=targets,
        missing_code=_HRL_MISSING,
        fallow_code=_HRL_FALLOW,
    )

    np.testing.assert_array_equal(assigned, sequences_by_farmer)
    assert np.all(quality["crop_sequence_quality_flag"].to_numpy() == 2)
    assert quality["crop_sequence_is_local_dominant"].all()
    assert quality["crop_sequence_is_original"].all()
    assert not diagnostics["regional_fallback_stage_used"].any()
    assert np.allclose(diagnostics["local_only_fit_score_pct"], 100.0)


def test_current_lowder_raster_pipeline_uses_regional_complete_sequence_fallback() -> (
    None
):
    """Exercise the current selection -> farm growth -> joint sequence pipeline.

    Only one selected pixel has a complete original sequence. Other pixels are
    agriculturally active but have a missing year. The workflow is nevertheless
    valid: the complete pixel seeds the regional sequence pool and farms without a
    local complete sequence receive that original regional fallback.
    """
    raw_crop_sequences = np.empty((3, 3, 4), dtype=np.int32)
    raw_crop_sequences[0] = 1110
    raw_crop_sequences[1] = 1120
    raw_crop_sequences[2] = 1130

    # The sole complete sequence deliberately contains genuine no-cropland in the
    # middle year; setup converts this 0 to fallow -1, which remains valid.
    raw_crop_sequences[:, 0, 0] = np.array(
        [1110, _HRL_NO_CROPLAND, 1130], dtype=np.int32
    )

    # Every other cell gets exactly one missing year, cycling the position so all
    # years still retain plenty of positive crop observations.
    flat = raw_crop_sequences.reshape(3, -1)
    for flat_index in range(1, flat.shape[1]):
        flat[(flat_index - 1) % 3, flat_index] = _HRL_MISSING

    cell_area_m2 = np.full((3, 4), 2_500.0, dtype=np.float64)
    valid_count = np.count_nonzero(raw_crop_sequences > 0, axis=0)
    eligible_mask = valid_count > 0
    selection_score = valid_count / 3.0 + 0.1
    cultivated_mask = select_cultivated_cells_by_area(
        selection_score,
        eligible_mask,
        cell_area_m2,
        target_area_m2=float(cell_area_m2.sum()),
    )
    assert cultivated_mask.all()

    crop_sequences = _apply_selected_hrl_domain_semantics(
        raw_crop_sequences, cultivated_mask
    )
    complete_mask = _complete_original_sequence_mask(crop_sequences, cultivated_mask)
    assert int(complete_mask.sum()) == 1
    np.testing.assert_array_equal(
        crop_sequences[:, 0, 0],
        np.array([1110, _HRL_FALLOW, 1130], dtype=np.int32),
    )

    target_farms = [
        TargetFarm(target_area_m2=7_500.0, size_class="synthetic") for _ in range(4)
    ]
    farm_values, farmers = grow_farms_from_raster_cells(
        cultivated_mask=cultivated_mask,
        crop_sequences=crop_sequences,
        cell_area_m2=cell_area_m2,
        target_farms=target_farms,
        random_seed=42,
    )
    assert ((farm_values >= 0) == cultivated_mask).all()
    assert len(farmers) == 4

    farmer_areas_m2 = farmers["area_m2"].to_numpy(dtype=np.float64)
    targets = _positive_crop_area_targets(crop_sequences, cell_area_m2, cultivated_mask)
    assigned, quality, diagnostics = assign_farmer_sequences_to_area_targets(
        farm_values=farm_values,
        crop_sequences=crop_sequences,
        cell_area_m2=cell_area_m2,
        farmer_areas_m2=farmer_areas_m2,
        target_crop_areas_per_year=targets,
        missing_code=_HRL_MISSING,
        fallow_code=_HRL_FALLOW,
    )

    expected_sequence = np.array([1110, _HRL_FALLOW, 1130], dtype=np.int32)
    assert assigned.shape == (len(farmers), 3)
    assert not np.any(assigned == _HRL_MISSING)
    assert np.all(assigned == expected_sequence[None, :])
    assert quality["crop_sequence_is_original"].all()
    # At most one farm can own the sole locally complete source cell; the others must
    # therefore exercise the regional-fallback path (quality flag 0).
    assert (
        int(np.count_nonzero(quality["crop_sequence_quality_flag"].to_numpy() == 0))
        >= 1
    )
    assert not diagnostics.empty


def test_grow_farms_from_raster_cells_handles_disconnected_patches() -> None:
    """A farm may continue in a second parcel when a connected patch is exhausted."""
    cultivated_mask = np.array(
        [
            [True, True, False, False, True, True],
            [True, True, False, False, True, True],
        ],
        dtype=bool,
    )
    cell_area_m2 = np.full(cultivated_mask.shape, 2_500.0, dtype=np.float64)
    crop_sequences = np.stack(
        [
            np.where(cultivated_mask, 1110, _HRL_MISSING),
            np.where(cultivated_mask, 1120, _HRL_MISSING),
            np.where(cultivated_mask, 1130, _HRL_MISSING),
        ]
    ).astype(np.int32)
    target_farms = [TargetFarm(target_area_m2=20_000.0, size_class="synthetic")]

    farms, farmers = grow_farms_from_raster_cells(
        cultivated_mask,
        crop_sequences,
        cell_area_m2,
        target_farms,
        random_seed=7,
        jump_candidate_sample=8,
        max_jump_distance_m=10_000.0,
    )

    assert ((farms >= 0) == cultivated_mask).all()
    assert len(farmers) == 1
    assert int(farmers.loc[0, "n_cells"]) == int(cultivated_mask.sum())
    assert int(farmers.loc[0, "n_fields"]) == 2
    assert np.isclose(float(farmers.loc[0, "area_m2"]), 20_000.0)


def test_grow_farms_rejects_more_targets_than_selected_cells() -> None:
    """Every target farm needs at least one selected model cell."""
    cultivated_mask = np.array([[True, True]], dtype=bool)
    crop_sequences = np.array([[[1110, 1110]], [[1120, 1120]]], dtype=np.int32)
    cell_area_m2 = np.full((1, 2), 2_500.0, dtype=np.float64)
    target_farms = [
        TargetFarm(target_area_m2=1_000.0, size_class="synthetic") for _ in range(3)
    ]

    with pytest.raises(ValueError, match="Cannot create 3 farms from only 2"):
        grow_farms_from_raster_cells(
            cultivated_mask,
            crop_sequences,
            cell_area_m2,
            target_farms,
        )


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Exercises the Numba raster farm pipeline."
)
def test_current_lowder_raster_pipeline_is_deterministic_for_fixed_seed() -> None:
    """Farm geometry and sequence assignment must be stable for a fixed seed."""
    cultivated_mask = np.ones((4, 4), dtype=bool)
    cell_area_m2 = np.full((4, 4), 2_500.0, dtype=np.float64)
    crop_sequences = np.stack(
        [
            np.full((4, 4), 1110, dtype=np.int32),
            np.full((4, 4), 1120, dtype=np.int32),
            np.full((4, 4), 1130, dtype=np.int32),
        ]
    )
    target_farms = [
        TargetFarm(target_area_m2=10_000.0, size_class="synthetic") for _ in range(4)
    ]

    first_farms, first_table = grow_farms_from_raster_cells(
        cultivated_mask,
        crop_sequences,
        cell_area_m2,
        target_farms,
        random_seed=123,
    )
    second_farms, second_table = grow_farms_from_raster_cells(
        cultivated_mask,
        crop_sequences,
        cell_area_m2,
        target_farms,
        random_seed=123,
    )

    np.testing.assert_array_equal(first_farms, second_farms)
    np.testing.assert_allclose(
        first_table["area_m2"].to_numpy(), second_table["area_m2"].to_numpy()
    )
