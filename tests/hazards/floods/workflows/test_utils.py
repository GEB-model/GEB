"""Tests for the SFINCS workflow utilities."""

from pathlib import Path

from geb.hazards.floods.workflows.utils import is_sfincs_simulation_finished

FINISHED_LOG: str = """
---------- Starting simulation -----------
 100% complete,     0.0 s remaining ...

---------- Simulation finished -----------

 Total time             :     13.393

---------- Closing off SFINCS -----------
"""

INTERRUPTED_LOG: str = """
---------- Starting simulation -----------
  45% complete,     6.2 s remaining ...
"""


def test_is_sfincs_simulation_finished(tmp_path: Path) -> None:
    """Test that only simulations with map output and a complete log are finished."""
    # missing folder
    assert not is_sfincs_simulation_finished(tmp_path / "does_not_exist")

    # log complete, but no map output
    (tmp_path / "sfincs.log").write_text(FINISHED_LOG)
    assert not is_sfincs_simulation_finished(tmp_path)

    # map output and complete log
    (tmp_path / "sfincs_map.nc").write_bytes(b"")
    assert is_sfincs_simulation_finished(tmp_path)

    # map output, but simulation was interrupted
    (tmp_path / "sfincs.log").write_text(INTERRUPTED_LOG)
    assert not is_sfincs_simulation_finished(tmp_path)

    # map output, but no log
    (tmp_path / "sfincs.log").unlink()
    assert not is_sfincs_simulation_finished(tmp_path)
