"""Tests for the Reporter module utilities in GEB."""

import datetime

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
        """Verify that substeps subdivide each day evenly.

        With timestep=1 day and substeps=4, each day should contribute 4
        entries at 0h, 6h, 12h, and 18h.
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
        assert dts[0] == datetime.datetime(2020, 1, 1, 0, 0, 0)
        assert dts[1] == datetime.datetime(2020, 1, 1, 6, 0, 0)
        assert dts[2] == datetime.datetime(2020, 1, 1, 12, 0, 0)
        assert dts[3] == datetime.datetime(2020, 1, 1, 18, 0, 0)
        assert dts[4] == datetime.datetime(2020, 1, 2, 0, 0, 0)

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
