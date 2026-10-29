"""Tests for the Destination Earth ERA5 data adapter and retry filesystem."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from geb.build.data_catalog.destination_earth import (
    DestinationEarthFileSystem,
)


def test_destination_earth_filesystem_retry_success() -> None:
    """Test that DestinationEarthFileSystem retries on transient payload errors and succeeds.

    Simulates transient connection drops (ClientPayloadError) on initial chunk reads,
    verifying that the filesystem retries and returns the expected byte payload.
    """

    async def run_test() -> None:
        mock_logger: MagicMock = MagicMock()
        fs: DestinationEarthFileSystem = DestinationEarthFileSystem(
            asynchronous=True, logger=mock_logger
        )
        expected_payload: bytes = b"chunk_data_bytes"

        attempts: list[int] = []

        async def mock_super_cat_file(
            url: str,
            start: int | None = None,
            end: int | None = None,
            **kwargs: Any,
        ) -> bytes:
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise aiohttp.ClientPayloadError(
                    "Response payload is not completed: Not enough data"
                )
            return expected_payload

        with (
            patch(
                "fsspec.implementations.http.HTTPFileSystem._cat_file",
                side_effect=mock_super_cat_file,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result: bytes = await fs._cat_file("https://destine.eu/chunk/0.0.0")

        assert result == expected_payload
        assert len(attempts) == 3
        assert mock_sleep.call_count == 2
        assert mock_logger.warning.call_count == 2

    asyncio.run(run_test())


def test_destination_earth_filesystem_retry_exhausted() -> None:
    """Test that DestinationEarthFileSystem raises the error after exhausting attempts.

    Verifies that when persistent network failures occur, the exception is raised
    after the configured maximum attempts.
    """

    async def run_test() -> None:
        mock_logger: MagicMock = MagicMock()
        fs: DestinationEarthFileSystem = DestinationEarthFileSystem(
            asynchronous=True, logger=mock_logger
        )

        async def mock_super_cat_file(
            url: str,
            start: int | None = None,
            end: int | None = None,
            **kwargs: Any,
        ) -> bytes:
            raise aiohttp.ClientPayloadError("Permanent connection failure")

        with (
            patch(
                "fsspec.implementations.http.HTTPFileSystem._cat_file",
                side_effect=mock_super_cat_file,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(aiohttp.ClientPayloadError),
        ):
            await fs._cat_file("https://destine.eu/chunk/0.0.0")
        assert mock_logger.error.call_count == 1

    asyncio.run(run_test())
