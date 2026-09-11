from collections.abc import Mapping
from typing import Protocol, final

from exo.shared.types.profiling import (
    NetworkInterfaceUtilization,
    NodeNetworkInfo,
    NodeNetworkUtilization,
)
from exo.shared.types.topology import Connection, RDMAConnection, SocketConnection
from exo.utils.pydantic_ext import FrozenModel

MINIMUM_ELAPSED_SECONDS = 1e-3


class ByteCounterSnapshot(Protocol):
    @property
    def bytes_sent(self) -> int: ...

    @property
    def bytes_recv(self) -> int: ...


@final
class InterfaceByteCounters(FrozenModel):
    name: str
    bytes_sent: int = 0
    bytes_recv: int = 0


def is_loopback_interface_name(name: str) -> bool:
    lowered = name.lower()
    if lowered == "lo":
        return True
    return lowered.startswith("lo") and lowered[2:].isdigit()


def interface_byte_counters_from_per_nic(
    per_nic: Mapping[str, ByteCounterSnapshot],
) -> dict[str, InterfaceByteCounters]:
    return {
        name: InterfaceByteCounters(
            name=name,
            bytes_sent=snapshot.bytes_sent,
            bytes_recv=snapshot.bytes_recv,
        )
        for name, snapshot in per_nic.items()
        if not is_loopback_interface_name(name)
    }


def utilization_from_counter_delta(
    previous: InterfaceByteCounters | None,
    current: InterfaceByteCounters,
    elapsed_seconds: float,
) -> NetworkInterfaceUtilization | None:
    if previous is None or previous.name != current.name:
        return None
    if elapsed_seconds < MINIMUM_ELAPSED_SECONDS:
        return None
    sent_delta = current.bytes_sent - previous.bytes_sent
    recv_delta = current.bytes_recv - previous.bytes_recv
    if sent_delta < 0 or recv_delta < 0:
        return None
    return NetworkInterfaceUtilization(
        name=current.name,
        bytes_sent_per_sec=sent_delta / elapsed_seconds,
        bytes_recv_per_sec=recv_delta / elapsed_seconds,
    )


@final
class NetworkUtilizationSampler:
    """Stateful differencer: previous counters + clock → live rates."""

    def __init__(self) -> None:
        self._previous: Mapping[str, InterfaceByteCounters] | None = None
        self._previous_monotonic: float | None = None

    def observe(
        self,
        counters: Mapping[str, InterfaceByteCounters],
        *,
        monotonic_seconds: float,
    ) -> tuple[NetworkInterfaceUtilization, ...] | None:
        previous = self._previous
        previous_at = self._previous_monotonic
        self._previous = counters
        self._previous_monotonic = monotonic_seconds
        if previous is None or previous_at is None:
            return None
        elapsed_seconds = monotonic_seconds - previous_at
        return tuple(
            utilization
            for name, current in counters.items()
            if (
                utilization := utilization_from_counter_delta(
                    previous.get(name), current, elapsed_seconds
                )
            )
            is not None
        )


def utilization_sample_from_observation(
    sampler: NetworkUtilizationSampler,
    counters: Mapping[str, InterfaceByteCounters],
    *,
    monotonic_seconds: float,
) -> NodeNetworkUtilization | None:
    interfaces = sampler.observe(counters, monotonic_seconds=monotonic_seconds)
    if interfaces is None:
        return None
    return NodeNetworkUtilization(interfaces=interfaces)


def utilization_for_named_interface(
    utilization: NodeNetworkUtilization | None,
    interface_name: str,
) -> NetworkInterfaceUtilization | None:
    if utilization is None:
        return None
    for interface in utilization.interfaces:
        if interface.name == interface_name:
            return interface
    return None


def interface_name_owning_address(
    network: NodeNetworkInfo | None, ip_address: str
) -> str | None:
    if network is None:
        return None
    for interface in network.interfaces:
        if interface.ip_address == ip_address:
            return interface.name
    return None


def source_interface_name_for_connection(connection: Connection) -> str | None:
    match connection.edge:
        case RDMAConnection():
            return connection.edge.source_rdma_iface
        case SocketConnection():
            return None


def sink_interface_name_for_connection(
    connection: Connection, sink_network: NodeNetworkInfo | None
) -> str | None:
    match connection.edge:
        case RDMAConnection():
            return connection.edge.sink_rdma_iface
        case SocketConnection():
            return interface_name_owning_address(
                sink_network, connection.edge.sink_multiaddr.ip_address
            )


def utilization_for_connection(
    connection: Connection,
    *,
    source_utilization: NodeNetworkUtilization | None,
    sink_utilization: NodeNetworkUtilization | None,
    sink_network: NodeNetworkInfo | None = None,
) -> NetworkInterfaceUtilization | None:
    """Attribute a topology edge to the NIC we can identify honestly.

    RDMA edges name both NICs; we report the source NIC's live rate.
    Socket edges only identify the sink address, so we report the sink NIC
    that owns that address. Source routing is not guessed.
    """
    source_name = source_interface_name_for_connection(connection)
    if source_name is not None:
        source_rate = utilization_for_named_interface(
            source_utilization, source_name
        )
        if source_rate is not None:
            return source_rate
    sink_name = sink_interface_name_for_connection(connection, sink_network)
    if sink_name is None:
        return None
    return utilization_for_named_interface(sink_utilization, sink_name)
