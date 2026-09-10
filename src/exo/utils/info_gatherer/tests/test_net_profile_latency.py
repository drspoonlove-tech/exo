from collections.abc import Iterator

import httpx
import pytest

from exo.shared.types.common import NodeId
from exo.utils.info_gatherer.net_profile import check_reachability


def _clock(values: list[float]) -> Iterator[float]:
    yield from values


@pytest.mark.anyio
async def test_check_reachability_records_successful_round_trip() -> None:
    expected = NodeId("node-b")
    ticks = _clock([10.0, 10.015])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=str(expected))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sample = await check_reachability(
            "10.0.0.2",
            expected,
            client,
            52415,
            clock=lambda: next(ticks),
        )

    assert sample is not None
    assert sample.ip_address == "10.0.0.2"
    assert sample.node_id == expected
    assert sample.latency_ms == 15.0


@pytest.mark.anyio
async def test_check_reachability_rejects_unexpected_node() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="someone-else")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sample = await check_reachability(
            "10.0.0.2",
            NodeId("node-b"),
            client,
            52415,
            clock=lambda: 1.0,
        )

    assert sample is None
