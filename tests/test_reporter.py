"""Tests for the Reporter module utilities in GEB."""

import datetime
from pathlib import Path

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from geb.reporter import create_time_array


def _to_datetimes(time_array: np.ndarray) -> list[datetime.datetime]:
    """Convert an int64 Unix-seconds array back to a list of datetime objects.

    Returns:
        List of datetime objects corresponding to each Unix-second timestamp.
    """
    return [
        datetime.datetime.fromtimestamp(int(t), tz=datetime.UTC).replace(tzinfo=None)
        for t in time_array
    ]


class TestDailyFrequency:
    """Tests for every-day reporting frequency."""

    def test_daily_no_substeps(self) -> None:
        """Verify that a 10-day range produces 10 daily entries."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 10)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}  # defaults to daily

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        assert len(dts) == 10
        assert dts[0] == start
        assert dts[-1] == end
        # Check uniform 1-day spacing
        for i in range(1, len(dts)):
            assert dts[i] - dts[i - 1] == datetime.timedelta(days=1)

    def test_daily_explicit_conf(self) -> None:
        """Verify explicit daily conf gives the same result as the default."""
        start = datetime.datetime(2020, 3, 1)
        end = datetime.datetime(2020, 3, 5)
        timestep = datetime.timedelta(days=1)
        conf_default: dict = {}
        conf_explicit: dict = {"frequency": {"every": "day"}}

        result_default = create_time_array(start, end, timestep, conf_default)
        result_explicit = create_time_array(start, end, timestep, conf_explicit)

        np.testing.assert_array_equal(result_default, result_explicit)

    def test_daily_single_day(self) -> None:
        """Verify that start == end yields exactly one entry."""
        start = datetime.datetime(2021, 6, 15)
        end = datetime.datetime(2021, 6, 15)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        result = create_time_array(start, end, timestep, conf)

        assert len(result) == 1
        assert _to_datetimes(result)[0] == start

    def test_daily_with_substeps(self) -> None:
        """Verify that substeps subdivide each day evenly at interval midpoints.

        With timestep=1 day and substeps=4 (6h interval), each day should contribute 4
        midpoint entries at 3h, 9h, 15h, and 21h.
        """
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 3)
        timestep = datetime.timedelta(days=1)
        conf: dict = {"frequency": {"every": "day"}}
        substeps = 4

        result = create_time_array(start, end, timestep, conf, substeps=substeps)

        dts = _to_datetimes(result)
        # 3 days × 4 substeps = 12 entries
        assert len(dts) == 12
        assert dts[0] == datetime.datetime(2020, 1, 1, 3, 0, 0)
        assert dts[1] == datetime.datetime(2020, 1, 1, 9, 0, 0)
        assert dts[2] == datetime.datetime(2020, 1, 1, 15, 0, 0)
        assert dts[3] == datetime.datetime(2020, 1, 1, 21, 0, 0)
        assert dts[4] == datetime.datetime(2020, 1, 2, 3, 0, 0)
        assert dts[-1] == datetime.datetime(2020, 1, 3, 21, 0, 0)

    def test_daily_with_hourly_substeps_midpoint(self) -> None:
        """Verify that 24 hourly substeps per day are centered at HH:30:00."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 1)
        timestep = datetime.timedelta(days=1)
        conf: dict = {"frequency": {"every": "day"}}

        result = create_time_array(start, end, timestep, conf, substeps=24)

        dts = _to_datetimes(result)
        assert len(dts) == 24
        assert dts[0] == datetime.datetime(2020, 1, 1, 0, 30, 0)
        assert dts[1] == datetime.datetime(2020, 1, 1, 1, 30, 0)
        assert dts[23] == datetime.datetime(2020, 1, 1, 23, 30, 0)

    def test_daily_with_relativedelta_timestep(self) -> None:
        """Verify that a relativedelta(days=1) timestep produces correct daily entries."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 5)
        timestep = relativedelta(days=1)
        conf: dict = {}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        assert len(dts) == 5
        assert dts[0] == start
        assert dts[-1] == end


class TestMonthlyFrequency:
    """Tests for every-month reporting frequency."""

    def test_monthly_first_of_month(self) -> None:
        """Verify that the first of each month is captured across a 3-month span."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 3, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "month", "day": 1}}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        assert len(dts) == 3
        assert dts[0] == datetime.datetime(2020, 1, 1)
        assert dts[1] == datetime.datetime(2020, 2, 1)
        assert dts[2] == datetime.datetime(2020, 3, 1)

    def test_monthly_mid_month(self) -> None:
        """Verify mid-month day is captured for each month in range."""
        start = datetime.datetime(2020, 1, 15)
        end = datetime.datetime(2020, 4, 20)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "month", "day": 15}}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        assert len(dts) == 4
        expected = [
            datetime.datetime(2020, 1, 15),
            datetime.datetime(2020, 2, 15),
            datetime.datetime(2020, 3, 15),
            datetime.datetime(2020, 4, 15),
        ]
        assert dts == expected

    def test_monthly_day_31_skips_short_months(self) -> None:
        """Verify that day-31 entries are skipped for months with fewer than 31 days."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 4, 30)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "month", "day": 31}}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        # Only January (31 days) and March (31 days) qualify; February and April do not
        assert len(dts) == 2
        assert dts[0] == datetime.datetime(2020, 1, 31)
        assert dts[1] == datetime.datetime(2020, 3, 31)

    def test_monthly_substeps_raises(self) -> None:
        """Verify that substeps with monthly frequency raises ValueError."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 3, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "month", "day": 1}}

        with pytest.raises(ValueError, match="Substeps not supported for monthly"):
            create_time_array(start, end, timestep, conf, substeps=4)


class TestYearlyFrequency:
    """Tests for every-year reporting frequency."""

    def test_yearly_basic(self) -> None:
        """Verify that a specific month/day is captured once per year."""
        start = datetime.datetime(2018, 1, 1)
        end = datetime.datetime(2021, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "year", "month": 6, "day": 15}}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        assert len(dts) == 4
        for year, dt in zip(range(2018, 2022), dts):
            assert dt == datetime.datetime(year, 6, 15)

    def test_yearly_partial_range(self) -> None:
        """Verify that dates outside [start, end] are excluded."""
        start = datetime.datetime(2020, 7, 1)
        end = datetime.datetime(2022, 5, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "year", "month": 6, "day": 15}}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        # 2020-06-15 is before start; 2022-06-15 is after end
        assert len(dts) == 1
        assert dts[0] == datetime.datetime(2021, 6, 15)

    def test_yearly_leap_day_skipped_in_non_leap_year(self) -> None:
        """Verify that Feb-29 is skipped in non-leap years."""
        start = datetime.datetime(2019, 1, 1)
        end = datetime.datetime(2024, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "year", "month": 2, "day": 29}}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        # Only 2020 and 2024 are leap years in the range
        assert len(dts) == 2
        assert dts[0] == datetime.datetime(2020, 2, 29)
        assert dts[1] == datetime.datetime(2024, 2, 29)

    def test_yearly_substeps_raises(self) -> None:
        """Verify that substeps with yearly frequency raises ValueError."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2022, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "year", "month": 6, "day": 15}}

        with pytest.raises(ValueError, match="Substeps not supported for yearly"):
            create_time_array(start, end, timestep, conf, substeps=4)


class TestInitialFinalFrequency:
    """Tests for 'initial' and 'final' reporting frequencies."""

    def test_initial_frequency(self) -> None:
        """Verify that initial frequency yields exactly the start datetime."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": "initial"}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        assert len(dts) == 1
        assert dts[0] == start

    def test_final_frequency(self) -> None:
        """Verify that final frequency yields exactly the end datetime."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": "final"}

        result = create_time_array(start, end, timestep, conf)

        dts = _to_datetimes(result)
        assert len(dts) == 1
        assert dts[0] == end

    def test_initial_substeps_raises(self) -> None:
        """Verify that substeps with initial frequency raises ValueError."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": "initial"}

        with pytest.raises(ValueError, match="Substeps not supported for initial"):
            create_time_array(start, end, timestep, conf, substeps=4)

    def test_final_substeps_raises(self) -> None:
        """Verify that substeps with final frequency raises ValueError."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": "final"}

        with pytest.raises(ValueError, match="Substeps not supported for final"):
            create_time_array(start, end, timestep, conf, substeps=4)


class TestOutputArrayProperties:
    """Tests for return value type and encoding."""

    def test_return_type_is_int64(self) -> None:
        """Verify that the returned array has dtype int64."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 5)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        result = create_time_array(start, end, timestep, conf)

        assert result.dtype == np.int64

    def test_values_are_unix_seconds(self) -> None:
        """Verify that the values are Unix timestamps in seconds."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 3)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        result = create_time_array(start, end, timestep, conf)

        expected_unix = [
            int(np.datetime64("2020-01-01", "s").astype(np.int64)),
            int(np.datetime64("2020-01-02", "s").astype(np.int64)),
            int(np.datetime64("2020-01-03", "s").astype(np.int64)),
        ]
        np.testing.assert_array_equal(result, expected_unix)

    def test_invalid_frequency_raises(self) -> None:
        """Verify that an unrecognised frequency string raises ValueError."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": "weekly"}

        with pytest.raises(ValueError, match="Frequency weekly not recognized"):
            create_time_array(start, end, timestep, conf)

    def test_invalid_every_value_raises(self) -> None:
        """Verify that an unrecognised 'every' value raises ValueError."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 12, 31)
        timestep = datetime.timedelta(days=1)
        conf = {"frequency": {"every": "week"}}

        with pytest.raises(ValueError, match="not recognized"):
            create_time_array(start, end, timestep, conf)


class TestCachingAndReadOnly:
    """Tests for result caching and the read-only guarantee."""

    def test_repeated_call_returns_same_values(self) -> None:
        """Verify that two calls with identical arguments return identical arrays."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 10)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        first = create_time_array(start, end, timestep, conf)
        second = create_time_array(start, end, timestep, conf)

        np.testing.assert_array_equal(first, second)

    def test_repeated_call_returns_view_not_same_object(self) -> None:
        """Verify that each call returns a distinct view, not the cached array itself.

        If the same object were returned, a writeable flag set on it by one
        caller would affect every subsequent caller.
        """
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 5)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        first = create_time_array(start, end, timestep, conf)
        second = create_time_array(start, end, timestep, conf)

        assert first is not second

    def test_returned_array_is_readonly(self) -> None:
        """Verify that the returned array (and cached base) are not writeable."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 5)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        result = create_time_array(start, end, timestep, conf)

        assert not result.flags.writeable

    def test_write_to_returned_array_raises(self) -> None:
        """Verify that attempting to write to the returned array raises ValueError."""
        start = datetime.datetime(2020, 1, 1)
        end = datetime.datetime(2020, 1, 5)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        result = create_time_array(start, end, timestep, conf)

        with pytest.raises(ValueError, match="read-only"):
            result[0] = 0

    def test_different_inputs_not_confused(self) -> None:
        """Verify that calls with different inputs return different results."""
        start = datetime.datetime(2020, 1, 1)
        timestep = datetime.timedelta(days=1)
        conf: dict = {}

        result_5 = create_time_array(
            start, datetime.datetime(2020, 1, 5), timestep, conf
        )
        result_10 = create_time_array(
            start, datetime.datetime(2020, 1, 10), timestep, conf
        )

        assert len(result_5) == 5
        assert len(result_10) == 10


class TestSpecialExportersSingleFile:
    """Tests verifying single parquet file export for special exporters and error handling."""

    def test_discharge_stations_export_individual_files(self, tmp_path: Path) -> None:
        """Verify that discharge validation stations export as individual files with attributes preserved."""
        from unittest.mock import MagicMock

        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        from geb.reporter import Reporter

        report_dir: Path = tmp_path / "report"
        model: MagicMock = MagicMock()
        model.config = {
            "report": {
                "_config": {"compression_level": 1, "chunk_target_size_bytes": 1000000},
                "_discharge_stations": True,
            }
        }
        model.mode = "w"
        model.simulate_hydrology = False
        stations_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
            {
                "snapped_grid_pixel_xy": [[10, 20], [30, 40]],
                "snapped_grid_pixel_lonlat": [[5.1, 52.1], [5.2, 52.2]],
                "GEB_upstream_area_from_grid": [1000.0, 2000.0],
            },
            index=[101, 102],
            geometry=[Point(5.1, 52.1), Point(5.2, 52.2)],
        )
        stations_file: Path = tmp_path / "stations.geoparquet"
        stations_gdf.to_parquet(stations_file)
        model.files = {"geom": {"discharge/discharge_snapped_locations": stations_file}}

        reporter: Reporter = Reporter(model, report_dir, clean=True)
        routing_reporters: dict = reporter.variables_to_report["hydrology.routing"]
        assert "discharge_hourly_m3_per_s_101" in routing_reporters
        assert "_group" not in routing_reporters["discharge_hourly_m3_per_s_101"]

        time_array: np.ndarray = np.array([1609459200, 1609462800], dtype=np.int64)
        routing_reporters["discharge_hourly_m3_per_s_101"]["_time_array"] = time_array
        routing_reporters["discharge_hourly_m3_per_s_101"]["_data_array"] = np.array(
            [10.5, 11.0], dtype=np.float32
        )
        routing_reporters["discharge_hourly_m3_per_s_102"]["_time_array"] = time_array
        routing_reporters["discharge_hourly_m3_per_s_102"]["_data_array"] = np.array(
            [20.5, 21.0], dtype=np.float32
        )

        reporter.finalize()

        file_101: Path = (
            report_dir / "hydrology.routing" / "discharge_hourly_m3_per_s_101.parquet"
        )
        file_102: Path = (
            report_dir / "hydrology.routing" / "discharge_hourly_m3_per_s_102.parquet"
        )
        assert file_101.exists()
        assert file_102.exists()

        df_101: pd.DataFrame = pd.read_parquet(file_101)
        assert list(df_101.columns) == ["discharge_hourly_m3_per_s_101"]
        assert df_101.attrs["station_location"]["pixel_xy"] == [10, 20]

    def test_grouped_variables_export_single_file(self, tmp_path: Path) -> None:
        """Verify that variables with _group export into a single consolidated parquet file."""
        from unittest.mock import MagicMock

        import pandas as pd

        from geb.reporter import Reporter

        report_dir: Path = tmp_path / "report"
        model: MagicMock = MagicMock()
        model.config = {
            "report": {
                "_config": {"compression_level": 1, "chunk_target_size_bytes": 1000000},
                "hydrology.routing": {
                    "retention_basin_discharge_m3_per_s_1": {
                        "varname": "grid.var.discharge_m3_s_per_substep",
                        "type": "grid",
                        "substeps": 24,
                        "_group": "retention_basin_discharge_m3_per_s",
                        "_group_key": "1",
                    },
                    "retention_basin_discharge_m3_per_s_2": {
                        "varname": "grid.var.discharge_m3_s_per_substep",
                        "type": "grid",
                        "substeps": 24,
                        "_group": "retention_basin_discharge_m3_per_s",
                        "_group_key": "2",
                    },
                },
            }
        }
        model.mode = "w"
        model.simulate_hydrology = False
        model.files = {}

        reporter: Reporter = Reporter(model, report_dir, clean=True)
        routing_reporters: dict = reporter.variables_to_report["hydrology.routing"]

        time_array: np.ndarray = np.array([1609459200, 1609462800], dtype=np.int64)
        routing_reporters["retention_basin_discharge_m3_per_s_1"]["_time_array"] = (
            time_array
        )
        routing_reporters["retention_basin_discharge_m3_per_s_1"]["_data_array"] = (
            np.array([1.0, 2.0], dtype=np.float32)
        )
        routing_reporters["retention_basin_discharge_m3_per_s_2"]["_time_array"] = (
            time_array
        )
        routing_reporters["retention_basin_discharge_m3_per_s_2"]["_data_array"] = (
            np.array([3.0, 4.0], dtype=np.float32)
        )

        reporter.finalize()

        output_file: Path = (
            report_dir
            / "hydrology.routing"
            / "retention_basin_discharge_m3_per_s.parquet"
        )
        assert output_file.exists()
        assert not (
            report_dir
            / "hydrology.routing"
            / "retention_basin_discharge_m3_per_s_1.parquet"
        ).exists()

        df: pd.DataFrame = pd.read_parquet(output_file)
        assert list(df.columns) == ["1", "2"]
        assert len(df) == 2

    def test_grouped_variables_with_extra_attributes_raises(
        self, tmp_path: Path
    ) -> None:
        """Verify that having extra_attributes on a grouped variable raises a clear ValueError."""
        from unittest.mock import MagicMock

        from geb.reporter import Reporter

        report_dir: Path = tmp_path / "report"
        model: MagicMock = MagicMock()
        model.config = {
            "report": {
                "_config": {"compression_level": 1, "chunk_target_size_bytes": 1000000},
                "hydrology.routing": {
                    "test_var": {
                        "varname": "var.test",
                        "type": "grid",
                        "_group": "test_group",
                        "_group_key": "k1",
                        "extra_attributes": {"key": "value"},
                    }
                },
            }
        }
        model.mode = "w"
        model.simulate_hydrology = False
        model.files = {}

        reporter: Reporter = Reporter(model, report_dir, clean=True)
        rep = reporter.variables_to_report["hydrology.routing"]["test_var"]
        rep["_time_array"] = np.array([1609459200], dtype=np.int64)
        rep["_data_array"] = np.array([1.0], dtype=np.float32)

        with pytest.raises(
            ValueError,
            match="Exporting grouped variables to a single parquet file does not support extra_attributes",
        ):
            reporter.finalize()

    def test_grouped_variables_missing_group_key_raises(self, tmp_path: Path) -> None:
        """Verify that a grouped variable without _group_key raises KeyError."""
        from unittest.mock import MagicMock

        from geb.reporter import Reporter

        report_dir: Path = tmp_path / "report"
        model: MagicMock = MagicMock()
        model.config = {
            "report": {
                "_config": {"compression_level": 1, "chunk_target_size_bytes": 1000000},
                "hydrology.routing": {
                    "test_var": {
                        "varname": "var.test",
                        "type": "grid",
                        "_group": "test_group",
                    }
                },
            }
        }
        model.mode = "w"
        model.simulate_hydrology = False
        model.files = {}

        reporter: Reporter = Reporter(model, report_dir, clean=True)
        rep = reporter.variables_to_report["hydrology.routing"]["test_var"]
        rep["_time_array"] = np.array([1609459200], dtype=np.int64)
        rep["_data_array"] = np.array([1.0], dtype=np.float32)

        with pytest.raises(
            KeyError,
            match="missing required '_group_key'",
        ):
            reporter.finalize()
