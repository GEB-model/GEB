"""Tests for SSP and RCP configuration."""

import logging
from pathlib import Path

import pytest

from geb.build import GEBModel

SSPS = ("ssp1", "ssp2", "ssp3", "ssp4", "ssp5")
RCPS = ("19", "26", "34", "45", "60", "70", "85")


@pytest.fixture
def model(tmp_path: Path) -> GEBModel:
    """Create a model instance for scenario tests.

    Returns:
        A model instance configured for testing.
    """
    model = GEBModel(logger=logging.getLogger("test"), root=tmp_path)
    model.files = {"dict": {}}
    return model


@pytest.mark.parametrize("ssp", SSPS)
@pytest.mark.parametrize("rcp", RCPS)
def test_explicit_ssp_rcp_pair_is_preserved(
    model: GEBModel, ssp: str, rcp: str
) -> None:
    """Preserve explicitly configured SSP/RCP pairs."""
    model.set_ssp(ssp=ssp, rcp=rcp)
    assert model.ssp == ssp
    assert model.rcp == rcp


@pytest.mark.parametrize(
    ("ssp", "expected_rcp"),
    [
        ("ssp1", "26"),
        ("ssp2", "45"),
        ("ssp3", "70"),
        ("ssp4", "34"),
        ("ssp5", "85"),
    ],
)
def test_rcp_is_inferred_from_ssp(model: GEBModel, ssp: str, expected_rcp: str) -> None:
    """Infer the default RCP from the configured SSP."""
    model.set_ssp(ssp=ssp)
    assert model.ssp == ssp
    assert model.rcp == expected_rcp


def test_defaults_when_scenarios_are_unset(model: GEBModel) -> None:
    """Use default scenarios when neither is configured."""
    assert model.ssp == "ssp3"
    assert model.rcp == "70"

    model.set_ssp(ssp=model.ssp)
    assert model.ssp == "ssp3"
    assert model.rcp == "70"


@pytest.mark.parametrize("ssp", ["invalid", "ssp6"])
def test_set_ssp_rejects_unsupported_ssp(model: GEBModel, ssp: str) -> None:
    """Reject unsupported SSP values."""
    with pytest.raises(ValueError, match="not supported"):
        model.set_ssp(ssp=ssp, rcp="70")


@pytest.mark.parametrize("rcp", ["invalid", "100"])
def test_set_ssp_rejects_unsupported_rcp(model: GEBModel, rcp: str) -> None:
    """Reject unsupported RCP values."""
    with pytest.raises(ValueError, match="not supported"):
        model.set_ssp(ssp="ssp3", rcp=rcp)


@pytest.mark.parametrize(
    ("rcp", "expected"),
    [
        ("26", 126),
        ("34", 434),
        ("45", 245),
        ("60", 460),
        ("70", 370),
        ("85", 585),
    ],
)
def test_cmip_identifier_depends_on_rcp(
    model: GEBModel, rcp: str, expected: int
) -> None:
    """Build the CMIP identifier from the RCP."""
    model.set_ssp(ssp="ssp4", rcp=rcp)
    assert model.ssp_rcp == expected
