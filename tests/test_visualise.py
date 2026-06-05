"""Tests for visualisation workflows."""

import matplotlib.pyplot as plt

from geb.workflows.visualise import plot_sunburst

from .testconfig import output_folder


def test_plot_sunburst_simple() -> None:
    """Test plot_sunburst with a simple hierarchy."""
    hierarchy = {
        "in": {"rain": 100, "snow": 20},
        "out": {
            "evapotranspiration": {
                "transpiration": 50,
                "bare soil evaporation": 10,
                "open water evaporation": 5,
            },
            "water demand": 30,
            "river outflow": 20,
        },
        "storage change": 5,
    }

    fig = plot_sunburst(hierarchy, title="Test Water Circle")
    assert isinstance(fig, plt.Figure)

    # Check if we can save it
    output_path = output_folder / "test_sunburst_simple.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    assert output_path.exists()
    plt.close(fig)


def test_plot_sunburst_empty() -> None:
    """Test plot_sunburst with minimal data."""
    hierarchy = {"in": 1, "out": 1}
    fig = plot_sunburst(hierarchy)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
