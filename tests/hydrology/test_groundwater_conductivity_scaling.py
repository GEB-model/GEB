"""Check calibration consistency at the internal/boundary groundwater interface."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import geb.hydrology.groundwater as groundwater_module
from geb.hydrology.groundwater import GroundWater


@pytest.mark.parametrize('multiplier', [0.5, 1.0, 2.0, 10.0])
def test_groundwater_conductivity_scales_across_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, multiplier: float
) -> None:
    """Apply calibration consistently on both sides of a constant-head boundary.

    Args:
        monkeypatch: Fixture replacing file reads and the MODFLOW constructor.
        tmp_path: Temporary simulation directory.
        multiplier: Dimensionless groundwater conductivity calibration factor.

    Returns:
        None.

    Raises:
        AssertionError: If the same aquifer has inconsistent calibration across the boundary.
    """  # noqa: DOC202, DOC502 - Document test outcomes and propagated assertions.
    conductivity_m_per_day: np.ndarray = np.full((1, 3, 3), 2.0, dtype=np.float64)
    boundaries_m: np.ndarray = np.stack([np.ones((3, 3)), np.zeros((3, 3))])
    heads_m: np.ndarray = np.full((1, 3, 3), 0.5, dtype=np.float64)
    read_array: MagicMock = MagicMock(side_effect=[
        SimpleNamespace(values=conductivity_m_per_day), SimpleNamespace(values=boundaries_m),
        SimpleNamespace(values=heads_m), SimpleNamespace(values=heads_m),
        SimpleNamespace(values=np.ones((3, 3), dtype=bool)),
        SimpleNamespace(values=boundaries_m), SimpleNamespace(values=conductivity_m_per_day),
    ])
    constructor: MagicMock = MagicMock()
    monkeypatch.setattr(groundwater_module, 'read_zarr', read_array)
    monkeypatch.setattr(groundwater_module, 'ModFlowSimulation', constructor)
    model: MagicMock = MagicMock()
    model.in_spinup = True
    model.simulation_root_spinup = tmp_path
    model.bin_folder = tmp_path
    model.config = {'parameters': {'groundwater_hydraulic_conductivity_multiplier': multiplier}}
    model.files = {'other': {key: key for key in [
        'groundwater/boundary_hydraulic_conductivity', 'groundwater/boundary_layer_boundary_elevation',
        'groundwater/boundary_heads', 'groundwater/boundary_mask',
    ]}, 'grid': {'groundwater/specific_yield': 'unused', 'landsurface/elevation_m': 'unused'}}
    hydrology: MagicMock = MagicMock()
    hydrology.grid.var = SimpleNamespace()
    hydrology.grid.compress.side_effect = lambda data: data.reshape(data.shape[0], -1)
    hydrology.grid.load3d.return_value = np.array([[0.2]], dtype=np.float32)
    hydrology.grid.load2d.return_value = np.array([1], dtype=np.float32)
    hydrology.grid.full_compressed.return_value = np.zeros(1, dtype=np.float32)
    model.hydrology = hydrology
    groundwater: GroundWater = GroundWater(model, hydrology)
    groundwater.initalize_modflow_model()
    np.testing.assert_allclose(constructor.call_args.kwargs['hydraulic_conductivity'], 2 * multiplier)
    np.testing.assert_allclose(constructor.call_args.kwargs['boundary_hydraulic_conductivity'], 2 * multiplier)
    # Calibration must not mutate the raw input values and compound on later loads.
    np.testing.assert_array_equal(conductivity_m_per_day, np.full((1, 3, 3), 2.0))
