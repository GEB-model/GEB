"""Tests for evapotranspiration functions in GEB."""

import math

import numpy as np

from geb.hydrology.evapotranspiration import (
    calculate_bare_soil_evaporation,
    calculate_transpiration,
    evapotranspirate,
    get_critical_soil_moisture_content,
    get_fraction_easily_available_soil_water,
    get_root_mass_ratios,
    get_root_ratios,
    get_transpiration_factor,
    get_transpiration_factor_per_layer,
)


def test_get_root_ratios() -> None:
    """Test calculation of root ratios for soil layers.

    Verifies that root ratios are correctly calculated based on
    root depth and soil layer heights.
    """
    soil_layer_height_m = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    root_ratios = get_root_ratios(
        root_depth_m=np.float32(0.5), soil_layer_height_m=soil_layer_height_m
    )
    np.testing.assert_almost_equal(
        root_ratios, np.array([1.0, 1.0, 2 / 3], dtype=np.float32)
    )

    root_ratios = get_root_ratios(
        root_depth_m=np.float32(0.2), soil_layer_height_m=soil_layer_height_m
    )
    np.testing.assert_almost_equal(
        root_ratios, np.array([1.0, 0.5, 0.0], dtype=np.float32)
    )

    root_ratios = get_root_ratios(
        root_depth_m=np.float32(0.1), soil_layer_height_m=soil_layer_height_m
    )
    np.testing.assert_almost_equal(
        root_ratios, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    root_ratios = get_root_ratios(
        root_depth_m=np.float32(0.05), soil_layer_height_m=soil_layer_height_m
    )
    np.testing.assert_almost_equal(
        root_ratios, np.array([0.5, 0.0, 0.0], dtype=np.float32)
    )


def test_get_root_mass_ratios() -> None:
    """Test calculation of root mass ratios for soil layers.

    Verifies that root mass ratios are correctly calculated
    assuming a triangular root distribution.
    """
    soil_layer_height_m = np.array([1, 1, 1], dtype=np.float32)

    root_depth_m = np.float32(0.5)
    root_ratios = get_root_ratios(
        root_depth_m=root_depth_m, soil_layer_height_m=soil_layer_height_m
    )
    root_mass_ratios = get_root_mass_ratios(
        root_depth_m=root_depth_m,
        root_ratios=root_ratios,
        soil_layer_height_m=soil_layer_height_m,
    )
    assert math.isclose(root_mass_ratios.sum(), 1.0)
    np.testing.assert_almost_equal(
        root_mass_ratios, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    root_depth_m = np.float32(1.0)
    root_ratios = get_root_ratios(
        root_depth_m=root_depth_m, soil_layer_height_m=soil_layer_height_m
    )
    root_mass_ratios = get_root_mass_ratios(
        root_depth_m=root_depth_m,
        root_ratios=root_ratios,
        soil_layer_height_m=soil_layer_height_m,
    )
    assert math.isclose(root_mass_ratios.sum(), 1.0)
    np.testing.assert_almost_equal(
        root_mass_ratios, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    root_depth_m = np.float32(2.0)
    root_ratios = get_root_ratios(
        root_depth_m=root_depth_m, soil_layer_height_m=soil_layer_height_m
    )
    root_mass_ratios = get_root_mass_ratios(
        root_depth_m=root_depth_m,
        root_ratios=root_ratios,
        soil_layer_height_m=soil_layer_height_m,
    )
    assert math.isclose(root_mass_ratios.sum(), 1.0)
    np.testing.assert_almost_equal(
        root_mass_ratios, np.array([0.75, 0.25, 0], dtype=np.float32)
    )

    soil_layer_height_m = np.array([0.5, 1, 1], dtype=np.float32)
    root_depth_m = np.float32(2.0)
    root_ratios = get_root_ratios(
        root_depth_m=root_depth_m, soil_layer_height_m=soil_layer_height_m
    )
    root_mass_ratios = get_root_mass_ratios(
        root_depth_m=root_depth_m,
        root_ratios=root_ratios,
        soil_layer_height_m=soil_layer_height_m,
    )
    assert math.isclose(root_mass_ratios.sum(), 1.0)
    np.testing.assert_almost_equal(
        root_mass_ratios, np.array([0.4375, 0.5, 0.0625], dtype=np.float32)
    )


def test_get_transpiration_factor_per_layer() -> None:
    """Test calculation of transpiration factor per soil layer.

    Verifies that transpiration factors are correctly calculated
    for each soil layer based on available water and root ratios.
    """
    soil_layer_height_m = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2),
        w_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=False,
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)
    np.testing.assert_equal(
        transpiration_factor_per_layer,
        np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2),
        w_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)
    np.testing.assert_equal(
        transpiration_factor_per_layer,
        np.array([0.75, 0.25, 0.0, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=False,
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)
    np.testing.assert_equal(
        transpiration_factor_per_layer,
        np.array([0.4, 0.4, 0.2, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)
    np.testing.assert_equal(
        transpiration_factor_per_layer,
        np.array([0.64, 0.32, 0.04, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=np.array([0.05, 1.0, 1.0, 1.0], dtype=np.float32),
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.03, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.04, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.01, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=False,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.02, 0.4, 0.4, 0.18], dtype=np.float32),
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0, abs_tol=1e-6)

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=np.array([0.05, 1.0, 1.0, 1.0], dtype=np.float32),
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.03, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.04, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.01, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.0396, 0.624, 0.304, 0.0324], dtype=np.float32),
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.15, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=False,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.0, 1 / 3 * 2, 1 / 3, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.15, 0.3, 0.3, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.0, 0.8888889, 0.1111111, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.15, 0.15, 0.15, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=False,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.15, 0.15, 0.15, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.15, 0.15, 0.15, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.16, 0.16, 0.16, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=False,
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)
    np.testing.assert_equal(
        transpiration_factor_per_layer,
        np.array([1.0 / 2.5, 1.0 / 2.5, 0.5 / 2.5, 0.0], dtype=np.float32),
    )  # at p of 1, we can still extract water as we like

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.16, 0.16, 0.16, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)
    np.testing.assert_equal(
        transpiration_factor_per_layer,
        np.array([0.64, 0.32, 0.04, 0.0], dtype=np.float32),
    )  # at p of 1, we can still extract water as we like

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.16, 0.16, 0.16, 0.16], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(0.5),
        correct_root_mass=False,
    )

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.16, 0.16, 0.16, 0.3], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(1),
        correct_root_mass=True,
    )
    assert math.isclose(transpiration_factor_per_layer.sum(), 1.0)
    np.testing.assert_equal(
        transpiration_factor_per_layer,
        np.array([0.64, 0.32, 0.04, 0.0], dtype=np.float32),
    )  # at p of 1, we can still extract water as we like

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.16, 0.16, 0.16, 0.16], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(0.5),
        correct_root_mass=False,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array(
            [0.1333333 / 2.5, 0.1333333 / 2.5, 0.066666 / 2.5, 0.0], dtype=np.float32
        ),
        decimal=4,
    )  # but at lower p, this becomes much more difficult
    assert transpiration_factor_per_layer.sum() < 1.0

    transpiration_factor_per_layer = get_transpiration_factor_per_layer(
        soil_layer_height_m=soil_layer_height_m,
        effective_root_depth_m=np.float32(2.5),
        w_m=np.array([0.16, 0.16, 0.16, 0.16], dtype=np.float32),
        wfc_m=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        wwp_m=np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32),
        p=np.float32(0.5),
        correct_root_mass=True,
    )
    np.testing.assert_almost_equal(
        transpiration_factor_per_layer,
        np.array([0.0853, 0.0427, 0.0053, 0.0], dtype=np.float32),
        decimal=4,
    )  # but at lower p, this becomes much more difficult
    assert transpiration_factor_per_layer.sum() < 1.0


def test_get_transpiration_factor() -> None:
    """Test calculation of transpiration factor.

    Verifies that transpiration factor is correctly calculated
    based on soil water content and critical moisture levels.
    """
    critical_soil_moisture_content = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    wwp = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15])  # wilting point
    w = np.array([0.3, 0.2, 0.175, 0.15, 0.1, 0.0])

    transpiration_factor = np.zeros_like(w)
    for i in range(len(w)):
        transpiration_factor[i] = get_transpiration_factor(
            w[i], wwp[i], critical_soil_moisture_content[i]
        )

    np.testing.assert_almost_equal(
        transpiration_factor,
        np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),
    )


def test_get_fraction_easily_available_soil_water() -> None:
    """Test calculation of fraction of easily available soil water.

    Verifies that the fraction of easily available soil water
    is correctly calculated based on crop group and evapotranspiration.
    """
    potential_evapotranspiratios = (
        np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) / 100
    )  # cm/day to m/day
    p5s_test = np.array(
        [
            0.94339623,
            0.82644628,
            0.73529412,
            0.66225166,
            0.60240964,
            0.55248619,
            0.51020408,
            0.47393365,
            0.44247788,
        ]
    )
    p1s_test = np.array(
        [
            0.44339623,
            0.35144628,
            0.28529412,
            0.23725166,
            0.20240964,
            0.17748619,
            0.16020408,
            0.14893365,
            0.14247788,
        ]
    )
    for potential_evapotranspiration, p5_test, p1_test in zip(
        potential_evapotranspiratios, p5s_test, p1s_test, strict=True
    ):
        p5 = get_fraction_easily_available_soil_water(
            crop_group_number=5,
            potential_evapotranspiration_m=potential_evapotranspiration,
        )

        assert math.isclose(p5, p5_test, rel_tol=1e-6)

        p1 = get_fraction_easily_available_soil_water(
            crop_group_number=1,
            potential_evapotranspiration_m=potential_evapotranspiration,
        )

        assert math.isclose(p1, p1_test, rel_tol=1e-6)


def test_get_critical_soil_moisture_content() -> None:
    """Test calculation of critical soil moisture content.

    Verifies that critical soil moisture content is correctly
    calculated based on field capacity, wilting point, and p factor.
    """
    p = np.array([0.3, 0.7, 1.0, 0.0])
    wfc = np.array([0.35, 0.35, 0.35, 0.35])  # field capacity
    wwp = np.array([0.15, 0.15, 0.15, 0.15])  # wilting point

    critical_soil_moisture_content = get_critical_soil_moisture_content(
        p=p, wfc_m=wfc, wwp_m=wwp
    )
    assert np.array_equal(critical_soil_moisture_content, [0.29, 0.21, 0.15, 0.35])


def test_evapotranspirate() -> None:
    """Test the scalar evapotranspirate function."""
    from geb.hydrology.landcovers import NON_PADDY_IRRIGATED

    # Test data for a single cell
    soil_is_frozen = False
    wwp_cell = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    wfc_cell = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    wres_cell = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    soil_layer_height_cell = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    land_use_type = NON_PADDY_IRRIGATED
    root_depth = np.float32(0.3)
    crop_map = 0  # Some crop
    natural_crop_groups = np.float32(3.0)
    potential_transpiration = np.float32(0.002)
    potential_bare_soil_evaporation = np.float32(0.001)
    potential_evapotranspiration = np.float32(0.003)
    frost_index = np.float32(0.0)
    crop_group_number_per_group = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    w_cell = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32
    )  # Above field capacity
    topwater = np.float32(0.0)
    open_water_evaporation = np.float32(0.0)
    minimum_effective_root_depth = np.float32(0.1)

    transpiration, evaporation, topwater_m = evapotranspirate(
        soil_is_frozen=soil_is_frozen,
        wwp_m=wwp_cell,
        wfc_m=wfc_cell,
        wres_m=wres_cell,
        soil_layer_height_m=soil_layer_height_cell,
        land_use_type=land_use_type,
        root_depth_m=root_depth,
        crop_map=crop_map,
        natural_crop_groups=natural_crop_groups,
        potential_transpiration_m=potential_transpiration,
        potential_bare_soil_evaporation_m=potential_bare_soil_evaporation,
        potential_evapotranspiration_m=potential_evapotranspiration,
        frost_index=frost_index,
        crop_group_number_per_group=crop_group_number_per_group,
        w_m=w_cell,
        topwater_m=topwater,
        open_water_evaporation_m=open_water_evaporation,
        minimum_effective_root_depth_m=minimum_effective_root_depth,
        time_step_hours_h=np.float32(24),  # Daily time step for test
    )

    # Basic checks
    assert isinstance(transpiration, (float, np.float32))
    assert isinstance(evaporation, (float, np.float32))
    assert transpiration >= 0
    assert evaporation >= 0
    assert transpiration <= potential_transpiration
    assert evaporation <= potential_bare_soil_evaporation


def test_calculate_transpiration() -> None:
    """Test the calculate_transpiration function."""
    from geb.hydrology.landcovers import NON_PADDY_IRRIGATED

    # Test data for a single cell
    soil_is_frozen = False
    wwp_cell = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    wfc_cell = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    wres_cell = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    soil_layer_height_cell = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    land_use_type = NON_PADDY_IRRIGATED
    root_depth = np.float32(0.3)
    crop_map = 0  # Some crop
    natural_crop_groups = np.float32(3.0)
    potential_transpiration = np.float32(0.002)
    potential_evapotranspiration = np.float32(0.003)
    frost_index = np.float32(0.0)
    crop_group_number_per_group = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    w_cell = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32
    )  # Above field capacity
    topwater = np.float32(0.0)
    minimum_effective_root_depth = np.float32(0.1)

    transpiration, topwater_m = calculate_transpiration(
        soil_is_frozen=soil_is_frozen,
        wwp_m=wwp_cell,
        wfc_m=wfc_cell,
        wres_m=wres_cell,
        soil_layer_height_m=soil_layer_height_cell,
        land_use_type=land_use_type,
        root_depth_m=root_depth,
        crop_map=crop_map,
        natural_crop_groups=natural_crop_groups,
        potential_transpiration_m=potential_transpiration,
        potential_evapotranspiration_m=potential_evapotranspiration,
        crop_group_number_per_group=crop_group_number_per_group,
        w_m=w_cell,
        topwater_m=topwater,
        minimum_effective_root_depth_m=minimum_effective_root_depth,
        time_step_hours_h=np.float32(24),  # Daily time step for test
    )

    # Basic checks
    assert isinstance(transpiration, (float, np.float32))
    assert transpiration >= 0
    assert transpiration <= potential_transpiration


def test_calculate_bare_soil_evaporation() -> None:
    """Test the calculate_bare_soil_evaporation function."""
    from geb.hydrology.landcovers import NON_PADDY_IRRIGATED

    # Test data for a single cell
    soil_is_frozen = False
    land_use_type = NON_PADDY_IRRIGATED
    potential_bare_soil_evaporation = np.float32(0.001)
    open_water_evaporation = np.float32(0.0)
    w_cell = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    wres_cell = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)

    evaporation = calculate_bare_soil_evaporation(
        soil_is_frozen=soil_is_frozen,
        land_use_type=land_use_type,
        potential_bare_soil_evaporation_m=potential_bare_soil_evaporation,
        open_water_evaporation_m=open_water_evaporation,
        w_m=w_cell,
        wres_m=wres_cell,
    )

    # Basic checks
    assert isinstance(evaporation, (float, np.float32))
    assert evaporation >= 0
    assert evaporation <= potential_bare_soil_evaporation

    # Test with frozen soil - should return 0
    evaporation_frozen = calculate_bare_soil_evaporation(
        soil_is_frozen=True,
        land_use_type=land_use_type,
        potential_bare_soil_evaporation_m=potential_bare_soil_evaporation,
        open_water_evaporation_m=open_water_evaporation,
        w_m=w_cell,
        wres_m=wres_cell,
    )
    assert evaporation_frozen == 0.0


def test_get_fraction_easily_available_soil_water_time_steps() -> None:
    """Test get_fraction_easily_available_soil_water with different time steps."""
    # Test that equivalent daily/hourly rates give same results
    daily_et = 0.005  # 5mm/day
    hourly_et = daily_et / 24  # equivalent hourly rate

    p_daily = get_fraction_easily_available_soil_water(
        crop_group_number=5,
        potential_evapotranspiration_m=daily_et,
        time_step_hours_h=24,
    )
    p_hourly = get_fraction_easily_available_soil_water(
        crop_group_number=5,
        potential_evapotranspiration_m=hourly_et,
        time_step_hours_h=1,
    )

    assert math.isclose(p_daily, p_hourly, rel_tol=1e-6)

    # Test with different crop groups
    for crop_group in [1, 2, 3, 4, 5]:
        p_daily = get_fraction_easily_available_soil_water(
            crop_group_number=crop_group,
            potential_evapotranspiration_m=daily_et,
            time_step_hours_h=24,
        )
        p_hourly = get_fraction_easily_available_soil_water(
            crop_group_number=crop_group,
            potential_evapotranspiration_m=hourly_et,
            time_step_hours_h=1,
        )
        assert math.isclose(p_daily, p_hourly, rel_tol=1e-6)


def test_get_fraction_easily_available_soil_water_edge_cases() -> None:
    """Test edge cases for get_fraction_easily_available_soil_water."""
    # Very low ET should give high p values
    p_low = get_fraction_easily_available_soil_water(
        crop_group_number=5, potential_evapotranspiration_m=0.00001, time_step_hours_h=1
    )
    assert p_low > 0.9  # Should be close to 1

    # Very high ET should give low p values
    p_high = get_fraction_easily_available_soil_water(
        crop_group_number=5, potential_evapotranspiration_m=0.01, time_step_hours_h=1
    )
    assert p_high < 0.1  # Should be close to 0

    # Test crop group 1 and 2 special cases
    et_test = 0.001
    p1 = get_fraction_easily_available_soil_water(
        crop_group_number=1,
        potential_evapotranspiration_m=et_test,
        time_step_hours_h=24,
    )
    p2 = get_fraction_easily_available_soil_water(
        crop_group_number=2,
        potential_evapotranspiration_m=et_test,
        time_step_hours_h=24,
    )
    p3 = get_fraction_easily_available_soil_water(
        crop_group_number=3,
        potential_evapotranspiration_m=et_test,
        time_step_hours_h=24,
    )

    # Crop groups 1 and 2 should have different behavior due to correction
    assert p1 != p3 or p2 != p3  # At least one should be different


def test_calculate_transpiration_frozen_soil() -> None:
    """Test calculate_transpiration with frozen soil."""
    wwp = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    wfc = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    wres = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    soil_layer_height = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    transpiration_frozen, topwater_m = calculate_transpiration(
        soil_is_frozen=True,  # Frozen soil
        wwp_m=wwp,
        wfc_m=wfc,
        wres_m=wres,
        soil_layer_height_m=soil_layer_height,
        land_use_type=1,
        root_depth_m=0.3,
        crop_map=0,
        natural_crop_groups=3.0,
        potential_transpiration_m=0.002,
        potential_evapotranspiration_m=0.003,
        crop_group_number_per_group=np.array([3.0, 4.0, 5.0], dtype=np.float32),
        w_m=w.copy(),
        topwater_m=0.0,
        minimum_effective_root_depth_m=0.1,
        time_step_hours_h=24,
    )

    assert transpiration_frozen == 0.0


def test_calculate_transpiration_paddy_irrigation() -> None:
    """Test calculate_transpiration with paddy irrigation."""
    from geb.hydrology.landcovers import PADDY_IRRIGATED

    wwp = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    wfc = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    wres = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    soil_layer_height = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    # Test with topwater available
    transpiration_paddy, topwater_m = calculate_transpiration(
        soil_is_frozen=False,
        wwp_m=wwp,
        wfc_m=wfc,
        wres_m=wres,
        soil_layer_height_m=soil_layer_height,
        land_use_type=PADDY_IRRIGATED,
        root_depth_m=0.3,
        crop_map=0,
        natural_crop_groups=3.0,
        potential_transpiration_m=0.002,
        potential_evapotranspiration_m=0.003,
        crop_group_number_per_group=np.array([3.0, 4.0, 5.0], dtype=np.float32),
        w_m=w.copy(),
        topwater_m=0.005,  # Topwater available
        minimum_effective_root_depth_m=0.1,
        time_step_hours_h=24,
    )

    # Should use topwater first
    assert transpiration_paddy > 0.0


def test_calculate_bare_soil_evaporation_paddy() -> None:
    """Test calculate_bare_soil_evaporation with paddy irrigation."""
    from geb.hydrology.landcovers import PADDY_IRRIGATED

    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    wres = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)

    evaporation_paddy = calculate_bare_soil_evaporation(
        soil_is_frozen=False,
        land_use_type=PADDY_IRRIGATED,
        potential_bare_soil_evaporation_m=0.001,
        open_water_evaporation_m=0.0,
        w_m=w,
        wres_m=wres,
    )

    # Should be 0 for paddy irrigation
    assert evaporation_paddy == 0.0


def test_calculate_bare_soil_evaporation_open_water() -> None:
    """Test calculate_bare_soil_evaporation with open water evaporation."""
    from geb.hydrology.landcovers import NON_PADDY_IRRIGATED

    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    wres = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)

    # Test with open water evaporation reducing bare soil evaporation
    evaporation_with_open_water = calculate_bare_soil_evaporation(
        soil_is_frozen=False,
        land_use_type=NON_PADDY_IRRIGATED,
        potential_bare_soil_evaporation_m=0.001,
        open_water_evaporation_m=0.0005,  # Half of potential
        w_m=w,
        wres_m=wres,
    )

    evaporation_no_open_water = calculate_bare_soil_evaporation(
        soil_is_frozen=False,
        land_use_type=NON_PADDY_IRRIGATED,
        potential_bare_soil_evaporation_m=0.001,
        open_water_evaporation_m=0.0,
        w_m=w,
        wres_m=wres,
    )

    # Should be less with open water evaporation
    assert evaporation_with_open_water < evaporation_no_open_water


def test_evapotranspirate_different_time_steps() -> None:
    """Test evapotranspirate with different time steps."""
    from geb.hydrology.landcovers import NON_PADDY_IRRIGATED

    # Test data
    wwp = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    wfc = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    wres = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    soil_layer_height = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    # Daily totals
    daily_trans = 0.002
    daily_evap = 0.001
    daily_et = 0.003

    # Equivalent hourly rates
    hourly_trans = daily_trans / 24
    hourly_evap = daily_evap / 24
    hourly_et = daily_et / 24

    # Test daily
    trans_daily, evap_daily, topwater_m = evapotranspirate(
        soil_is_frozen=False,
        wwp_m=wwp,
        wfc_m=wfc,
        wres_m=wres,
        soil_layer_height_m=soil_layer_height,
        land_use_type=NON_PADDY_IRRIGATED,
        root_depth_m=0.3,
        crop_map=0,
        natural_crop_groups=3.0,
        potential_transpiration_m=daily_trans,
        potential_bare_soil_evaporation_m=daily_evap,
        potential_evapotranspiration_m=daily_et,
        frost_index=0.0,
        crop_group_number_per_group=np.array([3.0, 4.0, 5.0], dtype=np.float32),
        w_m=w.copy(),
        topwater_m=0.0,
        open_water_evaporation_m=0.0,
        minimum_effective_root_depth_m=0.1,
        time_step_hours_h=24,
    )

    # Test hourly (24 steps of 1 hour each)
    total_trans_hourly = 0.0
    total_evap_hourly = 0.0

    for _ in range(24):
        trans_step, evap_step, topwater_m = evapotranspirate(
            soil_is_frozen=False,
            wwp_m=wwp,
            wfc_m=wfc,
            wres_m=wres,
            soil_layer_height_m=soil_layer_height,
            land_use_type=NON_PADDY_IRRIGATED,
            root_depth_m=0.3,
            crop_map=0,
            natural_crop_groups=3.0,
            potential_transpiration_m=hourly_trans,
            potential_bare_soil_evaporation_m=hourly_evap,
            potential_evapotranspiration_m=hourly_et,
            frost_index=0.0,
            crop_group_number_per_group=np.array([3.0, 4.0, 5.0], dtype=np.float32),
            w_m=w.copy(),
            topwater_m=0.0,
            open_water_evaporation_m=0.0,
            minimum_effective_root_depth_m=0.1,
            time_step_hours_h=1,
        )
        total_trans_hourly += trans_step
        total_evap_hourly += evap_step

    # Should be approximately equal (allowing for numerical differences)
    assert abs(total_trans_hourly - trans_daily) < 1e-6
    assert abs(total_evap_hourly - evap_daily) < 1e-6


def test_get_transpiration_factor_edge_cases() -> None:
    """Test get_transpiration_factor with edge cases."""
    # Test when denominator is zero
    factor1 = get_transpiration_factor(0.1, 0.1, 0.1)  # w = wwp = wcrit
    assert factor1 == 0.0  # No available water

    factor2 = get_transpiration_factor(0.2, 0.1, 0.1)  # w > wwp, but wwp = wcrit
    assert factor2 == 1.0  # All water available

    # Test normal cases
    factor3 = get_transpiration_factor(0.15, 0.1, 0.2)  # w between wwp and wcrit
    assert 0.0 <= factor3 <= 1.0

    # Test bounds
    factor4 = get_transpiration_factor(0.05, 0.1, 0.2)  # w < wwp
    assert factor4 == 0.0

    factor5 = get_transpiration_factor(0.25, 0.1, 0.2)  # w > wcrit
    assert factor5 == 1.0


def test_get_critical_soil_moisture_content_bounds() -> None:
    """Test get_critical_soil_moisture_content with boundary p values."""
    wfc = np.array([0.25, 0.25], dtype=np.float32)
    wwp = np.array([0.1, 0.1], dtype=np.float32)

    # p = 1: critical = wwp
    critical_p1 = get_critical_soil_moisture_content(np.array([1.0, 1.0]), wfc, wwp)
    np.testing.assert_array_almost_equal(critical_p1, wwp)

    # p = 0: critical = wfc
    critical_p0 = get_critical_soil_moisture_content(np.array([0.0, 0.0]), wfc, wwp)
    np.testing.assert_array_almost_equal(critical_p0, wfc)

    # p = 0.5: critical = midpoint
    expected_mid = (wfc + wwp) / 2
    critical_p0_5 = get_critical_soil_moisture_content(np.array([0.5, 0.5]), wfc, wwp)
    np.testing.assert_array_almost_equal(critical_p0_5, expected_mid)
