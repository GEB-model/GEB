"""Regression coverage for discharge characteristic analysis and plotting."""

import logging
from pathlib import Path
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from geb.evaluate.workflows import discharge_characteristics as characteristics


@pytest.fixture
def station_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create scores and attributes with one unmatched station.

    Returns:
        Scores in stored units and GRDC-Caravan catchment attributes.
    """
    scores: pd.DataFrame = pd.DataFrame({"station_ID": [1, 2, 3, 4]})
    for target in characteristics.KGE_COMPONENT_TARGETS:
        scores[target.column] = [0.1, 0.3, 0.6, 0.9]
    attributes: pd.DataFrame = pd.DataFrame(
        {"gauge_id": ["GRDC_1", "GRDC_2", "GRDC_3"]}
    )
    for characteristic in characteristics.SCREENING_CHARACTERISTICS:
        if characteristic.column == "upstream_area_GEB":
            scores[characteristic.column] = [1e6, 2e6, 3e6, 4e6]
        else:
            attributes[characteristic.column] = [1.0, 2.0, 3.0]
    return scores, attributes


def test_analysis_units_and_associations(
    station_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Preserve matching, unit conversion, and all component associations.

    Args:
        station_tables: Synthetic scores and catchment attributes.
    """
    scores, attributes = station_tables
    original: pd.DataFrame = scores.copy(deep=True)
    enriched: pd.DataFrame = characteristics.enrich_discharge_evaluation(
        scores, attributes
    )
    analysis: pd.DataFrame = characteristics.prepare_kge_characteristic_analysis(
        enriched
    )
    assert len(analysis) == 3
    np.testing.assert_allclose(analysis["upstream_area_GEB"], [1, 2, 3])
    np.testing.assert_allclose(analysis["tmp_dc_syr"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(analysis["low_prec_freq"], [100, 200, 300])
    associations: pd.DataFrame = characteristics.calculate_kge_component_associations(
        analysis
    )
    assert len(associations) == 128
    np.testing.assert_allclose(associations["spearman_rho"], 1.0)
    assert associations["n"].eq(3).all()
    analysis["ele_mt_sav"] = 1.0
    associations = characteristics.calculate_kge_component_associations(analysis)
    assert (
        associations.loc[associations["variable"] == "ele_mt_sav", "spearman_rho"]
        .isna()
        .all()
    )
    pd.testing.assert_frame_equal(scores, original)
    with pytest.raises(ValueError, match="duplicate gauge_id"):
        characteristics.enrich_discharge_evaluation(
            scores, pd.concat([attributes, attributes])
        )


@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("logarithmic", [False, True])
def test_relationship_curve(summary: bool, logarithmic: bool) -> None:
    """Keep a linear relationship exact in both display modes.

    Args:
        summary: Whether to include the bootstrap interval.
        logarithmic: Whether x is displayed on a logarithmic axis.
    """
    model_x: np.ndarray = np.linspace(1.0, 2.0, 30)
    table: pd.DataFrame = pd.DataFrame(
        {
            "x": 10**model_x if logarithmic else model_x,
            "KGE_daily": model_x * 0.4,
        }
    )
    figure, axis = plt.subplots()
    try:
        assert characteristics._plot_relationship_panel(
            axis,
            table,
            characteristics.Characteristic("x", "X (–)", logarithmic_x=logarithmic),
            1.0,
            (0.0, 1.0),
            np.random.default_rng(42) if summary else None,
        )
        curve_data: np.ndarray = np.asarray(axis.lines[0].get_xydata())
        curve_x: np.ndarray = curve_data[:, 0]
        curve_y: np.ndarray = curve_data[:, 1]
        np.testing.assert_allclose(
            curve_y, (np.log10(curve_x) if logarithmic else curve_x) * 0.4
        )
        assert len(axis.collections) == (2 if summary else 1)
        assert axis.get_xscale() == ("log" if logarithmic else "linear")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("values", [np.arange(9), np.ones(12), -np.arange(12)])
def test_unplottable_relationship(values: np.ndarray) -> None:
    """Hide panels with too few pairs, constant values, or nonpositive log data.

    Args:
        values: Characteristic values for an unusable relationship.
    """
    figure, axis = plt.subplots()
    try:
        assert not characteristics._plot_relationship_panel(
            axis,
            pd.DataFrame({"x": values, "KGE_daily": values}),
            characteristics.Characteristic("x", "X (–)", logarithmic_x=True),
            1.0,
            (0.0, 1.0),
        )
        assert not axis.axison
    finally:
        plt.close(figure)


@pytest.mark.parametrize("export", [False, True])
def test_workflow_exports_and_closes_figures(
    station_tables: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export: bool,
) -> None:
    """Check workflow exports, period suffixes, and figure cleanup.

    Args:
        station_tables: Synthetic station inputs.
        tmp_path: Temporary output directory.
        monkeypatch: Fixture replacing expensive figure exports.
        export: Whether the workflow should write outputs.
    """
    figures: list[plt.Figure] = [plt.figure() for _ in range(3)]
    for name, figure in zip(
        (
            "create_characteristic_correlation_matrix",
            "create_kge_characteristic_summary",
            "create_kge_characteristic_scatterplots",
        ),
        figures,
        strict=True,
    ):
        monkeypatch.setattr(characteristics, name, Mock(return_value=figure))
    output_folder: Path = tmp_path / "explanations"
    characteristics.plot_discharge_characteristics(
        *station_tables,
        output_folder,
        logging.getLogger(__name__),
        "_2000_2005",
        export,
    )
    assert not any(plt.fignum_exists(figure.number) for figure in figures)
    assert output_folder.exists() == export
    if export:
        assert (
            len(
                pd.read_csv(
                    output_folder / "discharge_kge_component_associations_2000_2005.csv"
                )
            )
            == 128
        )


def test_workflow_without_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip analysis when no evaluated station matches the attributes.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Fixture recording analysis calls.
    """
    prepare: Mock = Mock()
    monkeypatch.setattr(characteristics, "prepare_kge_characteristic_analysis", prepare)
    characteristics.plot_discharge_characteristics(
        pd.DataFrame({"station_ID": [1]}),
        pd.DataFrame({"gauge_id": ["GRDC_2"]}),
        tmp_path,
        logging.getLogger(__name__),
        export=False,
    )
    prepare.assert_not_called()
