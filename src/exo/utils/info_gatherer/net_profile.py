import time
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from typing import final

import anyio
import httpx
from anyio import create_task_group
from loguru import logger

from exo.shared.link_profile import elapsed_milliseconds
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.utils.channels import Sender, channel

REACHABILITY_ATTEMPTS = 3


@final
@dataclass(frozen=True, slots=True)
class ReachabilitySample:
    ip_address: str
    node_id: NodeId
    latency_ms: float


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    client: httpx.AsyncClient,
    api_port: int,
    clock: Callable[[], float] = time.monotonic,
) -> ReachabilitySample | None:
    """Probe ``/node_id`` and return a timed sample when identity matches."""
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:{api_port}/node_id"
    else:
        url = f"http://{target_ip}:{api_port}/node_id"

    remote_node_id = None
    last_error = None
    latency_ms: float | None = None

    for _ in range(REACHABILITY_ATTEMPTS):
        try:
            started = clock()
            response = await client.get(url)
            finished = clock()
            if response.status_code != 200:
                await anyio.sleep(1)
                continue

            body = response.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            latency_ms = elapsed_milliseconds(started, finished)
            break

        # expected failure cases
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
        ):
            await anyio.sleep(1)

        # other failures should be logged on last attempt
        except httpx.HTTPError as e:
            last_error = e
            await anyio.sleep(1)

    if last_error is not None:
        logger.warning(
            f"connect error {type(last_error).__name__} from {target_ip} after {REACHABILITY_ATTEMPTS} attempts; treating as down"
        )

    if remote_node_id is None or latency_ms is None:
        return None

    if remote_node_id != expected_node_id:
        logger.debug(
            f"Discovered node with unexpected node_id; "
            f"ip={target_ip}, expected_node_id={expected_node_id}, "
            f"remote_node_id={remote_node_id}"
        )
        return None

    return ReachabilitySample(
        ip_address=target_ip,
        node_id=remote_node_id,
        latency_ms=latency_ms,
    )


async def check_reachable(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    api_port: int,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncGenerator[ReachabilitySample, None]:
    """Yield reachability samples as probes complete."""

    send, recv = channel[ReachabilitySample]()

    # these are intentionally httpx's defaults so we can tune them later
    timeout = httpx.Timeout(timeout=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async def _probe(
        target_ip: str,
        expected_node_id: NodeId,
        client: httpx.AsyncClient,
        send: Sender[ReachabilitySample],
    ) -> None:
        async with send:
            sample = await check_reachability(
                target_ip, expected_node_id, client, api_port, clock
            )
            if sample is not None:
                await send.send(sample)

    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits, verify=False) as client,
        create_task_group() as tg,
    ):
        for node_id in topology.list_nodes():
            if node_id not in node_network:
                continue
            if node_id == self_node_id:
                continue
            for iface in node_network[node_id].interfaces:
                tg.start_soon(_probe, iface.ip_address, node_id, client, send.clone())
        send.close()

        with recv:
            async for item in recv:
                yield item
