from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Protocol, final

from exo.master.placement_utils import (
    NON_NEIGHBOR_PLACEHOLDER_IP_ADDRESS,
    SELF_BIND_IP_ADDRESS,
    find_ip_prioritised,
    get_mlx_jaccl_coordinators,
    get_mlx_ring_hosts_by_node,
    interface_priority_score,
)
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.events import InstanceReplacedAtomically
from exo.shared.types.profiling import InterfaceType, NodeNetworkInfo
from exo.shared.types.topology import Cycle, SocketConnection
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)

_NON_DATA_PLANE_IP_ADDRESSES = frozenset(
    {SELF_BIND_IP_ADDRESS, NON_NEIGHBOR_PLACEHOLDER_IP_ADDRESS}
)


class Clock(Protocol):
    def now(self) -> datetime: ...


@final
class SystemClock:
    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


def cycle_from_instance(instance: Instance) -> Cycle:
    assignments = instance.shard_assignments
    ranked_nodes = sorted(
        assignments.node_to_runner.items(),
        key=lambda item: assignments.runner_to_shard[item[1]].device_rank,
    )
    return Cycle(node_ids=[node_id for node_id, _ in ranked_nodes])


def interface_type_for_ip_address(
    ip_address: str,
    node_network: NodeNetworkInfo,
) -> InterfaceType:
    for interface in node_network.interfaces:
        if interface.ip_address == ip_address:
            return interface.interface_type
    return "unknown"


def reachable_socket_ip_addresses(
    source: NodeId,
    sink: NodeId,
    topology: Topology,
) -> set[str]:
    return {
        connection.sink_multiaddr.ip_address
        for connection in topology.get_all_connections_between(source, sink)
        if isinstance(connection, SocketConnection)
    }


def instance_with_preferred_connections(
    instance: Instance,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> Instance | None:
    """Return a same-id instance on a better path, or None if no upgrade.

    Does not raise. Incomplete topology (a neighbour with no socket path)
    is treated as "cannot upgrade yet" and returns None.
    """
    match instance:
        case MlxRingInstance():
            return _preferred_ring_instance(instance, topology, node_network)
        case MlxJacclInstance():
            return _preferred_jaccl_instance(instance, topology, node_network)


def connection_upgrade_events(
    instances: Mapping[InstanceId, Instance],
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> Sequence[InstanceReplacedAtomically]:
    """Immediate replacements with no stability delay. Used by tests and one-shot checks."""
    events: list[InstanceReplacedAtomically] = []
    for instance in instances.values():
        replacement = instance_with_preferred_connections(
            instance, topology, node_network
        )
        if replacement is not None:
            events.append(InstanceReplacedAtomically(instance=replacement))
    return events


@final
class ConnectionUpgradeScheduler:
    """Hold a candidate replacement until it stays preferred for ``stable_for``.

    Thunderbolt interfaces bounce while enumerating. Replacing on the first
    edge create would restart runners onto a path that then disappears.
    """

    def __init__(
        self,
        clock: Clock,
        stable_for: timedelta = timedelta(seconds=2),
    ) -> None:
        self._clock = clock
        self._stable_for = stable_for
        self._pending: dict[InstanceId, tuple[Instance, datetime]] = {}

    def events(
        self,
        instances: Mapping[InstanceId, Instance],
        topology: Topology,
        node_network: Mapping[NodeId, NodeNetworkInfo],
    ) -> list[InstanceReplacedAtomically]:
        now = self._clock.now()
        replacements: list[InstanceReplacedAtomically] = []

        for instance in instances.values():
            replacement = instance_with_preferred_connections(
                instance, topology, node_network
            )
            if replacement is None:
                self._pending.pop(instance.instance_id, None)
                continue

            previous = self._pending.get(instance.instance_id)
            if previous is None or previous[0] != replacement:
                self._pending[instance.instance_id] = (replacement, now)
                continue

            if now - previous[1] >= self._stable_for:
                replacements.append(InstanceReplacedAtomically(instance=replacement))
                self._pending.pop(instance.instance_id, None)

        return replacements


def _preferred_ring_instance(
    instance: MlxRingInstance,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> MlxRingInstance | None:
    selected_cycle = cycle_from_instance(instance)
    if len(selected_cycle) <= 1:
        return None

    cycle_digraph = topology.get_subgraph_from_nodes(selected_cycle.node_ids)
    if not _ring_neighbors_are_reachable(selected_cycle, cycle_digraph, node_network):
        return None

    new_hosts = get_mlx_ring_hosts_by_node(
        selected_cycle=selected_cycle,
        cycle_digraph=cycle_digraph,
        ephemeral_port=instance.ephemeral_port,
        node_network=node_network,
    )
    if new_hosts == instance.hosts_by_node:
        return None
    if not _host_lists_justify_replace(
        current=instance.hosts_by_node,
        candidate=new_hosts,
        selected_cycle=selected_cycle,
        topology=cycle_digraph,
        node_network=node_network,
        ring=True,
    ):
        return None
    return instance.model_copy(update={"hosts_by_node": new_hosts})


def _preferred_jaccl_instance(
    instance: MlxJacclInstance,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> MlxJacclInstance | None:
    selected_cycle = cycle_from_instance(instance)
    if len(selected_cycle) <= 1:
        return None

    coordinator_node_id = _rank_zero_node(instance)
    coordinator_port = _coordinator_port(instance)
    if coordinator_node_id is None or coordinator_port is None:
        return None

    cycle_digraph = topology.get_subgraph_from_nodes(selected_cycle.node_ids)
    if not _coordinator_is_reachable(
        coordinator_node_id, selected_cycle, cycle_digraph, node_network
    ):
        return None

    new_coordinators = get_mlx_jaccl_coordinators(
        coordinator=coordinator_node_id,
        coordinator_port=coordinator_port,
        cycle_digraph=cycle_digraph,
        node_network=node_network,
    )
    if new_coordinators == instance.jaccl_coordinators:
        return None
    if not _coordinators_justify_replace(
        current=instance.jaccl_coordinators,
        candidate=new_coordinators,
        coordinator_node_id=coordinator_node_id,
        topology=cycle_digraph,
        node_network=node_network,
    ):
        return None
    return instance.model_copy(update={"jaccl_coordinators": new_coordinators})


def _ring_neighbors_are_reachable(
    selected_cycle: Cycle,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> bool:
    world_size = len(selected_cycle)
    for rank, node_id in enumerate(selected_cycle):
        for neighbor_rank in ((rank - 1) % world_size, (rank + 1) % world_size):
            if neighbor_rank == rank:
                continue
            other_node_id = selected_cycle.node_ids[neighbor_rank]
            if (
                find_ip_prioritised(
                    node_id, other_node_id, topology, node_network, ring=True
                )
                is None
            ):
                return False
    return True


def _coordinator_is_reachable(
    coordinator_node_id: NodeId,
    selected_cycle: Cycle,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> bool:
    for node_id in selected_cycle:
        if node_id == coordinator_node_id:
            continue
        if (
            find_ip_prioritised(
                node_id, coordinator_node_id, topology, node_network, ring=False
            )
            is None
        ):
            return False
    return True


def _rank_zero_node(instance: Instance) -> NodeId | None:
    for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        if shard.device_rank == 0:
            return node_id
    return None


def _coordinator_port(instance: MlxJacclInstance) -> int | None:
    first_address = next(iter(instance.jaccl_coordinators.values()), None)
    if first_address is None or ":" not in first_address:
        return None
    port_text = first_address.rsplit(":", 1)[1]
    if not port_text.isdigit():
        return None
    return int(port_text)


def _host_lists_justify_replace(
    current: Mapping[NodeId, list[Host]],
    candidate: Mapping[NodeId, list[Host]],
    selected_cycle: Cycle,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    *,
    ring: bool,
) -> bool:
    improved = False
    for owner_node_id in selected_cycle:
        current_hosts = current.get(owner_node_id, [])
        candidate_hosts = candidate.get(owner_node_id, [])
        if len(current_hosts) != len(candidate_hosts):
            return True
        for index, (current_host, candidate_host) in enumerate(
            zip(current_hosts, candidate_hosts, strict=True)
        ):
            peer_node_id = selected_cycle.node_ids[index]
            if (
                current_host.ip in _NON_DATA_PLANE_IP_ADDRESSES
                and candidate_host.ip in _NON_DATA_PLANE_IP_ADDRESSES
            ):
                continue
            if current_host == candidate_host:
                continue

            peer_network = node_network.get(peer_node_id, NodeNetworkInfo())
            current_score = interface_priority_score(
                interface_type_for_ip_address(current_host.ip, peer_network),
                ring=ring,
            )
            candidate_score = interface_priority_score(
                interface_type_for_ip_address(candidate_host.ip, peer_network),
                ring=ring,
            )
            current_still_reachable = current_host.ip in reachable_socket_ip_addresses(
                owner_node_id, peer_node_id, topology
            )
            if not current_still_reachable:
                improved = True
                continue
            if candidate_score < current_score:
                improved = True
            elif candidate_score > current_score:
                return False
    return improved


def _coordinators_justify_replace(
    current: Mapping[NodeId, str],
    candidate: Mapping[NodeId, str],
    coordinator_node_id: NodeId,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> bool:
    improved = False
    coordinator_network = node_network.get(coordinator_node_id, NodeNetworkInfo())
    for node_id, current_address in current.items():
        candidate_address = candidate.get(node_id)
        if candidate_address is None or candidate_address == current_address:
            continue
        current_ip_address = current_address.rsplit(":", 1)[0]
        candidate_ip_address = candidate_address.rsplit(":", 1)[0]
        if (
            current_ip_address in _NON_DATA_PLANE_IP_ADDRESSES
            and candidate_ip_address in _NON_DATA_PLANE_IP_ADDRESSES
        ):
            continue

        current_score = interface_priority_score(
            interface_type_for_ip_address(current_ip_address, coordinator_network),
            ring=False,
        )
        candidate_score = interface_priority_score(
            interface_type_for_ip_address(candidate_ip_address, coordinator_network),
            ring=False,
        )
        current_still_reachable = current_ip_address in reachable_socket_ip_addresses(
            node_id, coordinator_node_id, topology
        )
        if not current_still_reachable:
            improved = True
            continue
        if candidate_score < current_score:
            improved = True
        elif candidate_score > current_score:
            return False
    return improved


__all__ = [
    "Clock",
    "ConnectionUpgradeScheduler",
    "SystemClock",
    "connection_upgrade_events",
    "cycle_from_instance",
    "instance_with_preferred_connections",
    "interface_type_for_ip_address",
    "reachable_socket_ip_addresses",
]
