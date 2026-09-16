from datetime import datetime, timedelta, timezone
from typing import final

from exo.master.connection_upgrade import (
    ConnectionUpgradeScheduler,
    connection_upgrade_events,
    instance_with_preferred_connections,
)
from exo.master.placement_utils import interface_priority_score
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import Host, NodeId
from exo.shared.types.events import InstanceReplacedAtomically
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    InterfaceType,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import (
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata

NODE_A = NodeId("node-a")
NODE_B = NodeId("node-b")
INSTANCE_ID = InstanceId("instance-upgrade")
RUNNER_A = RunnerId("runner-a")
RUNNER_B = RunnerId("runner-b")
EPHEMERAL_PORT = 50123

WIFI_A = "10.0.0.1"
WIFI_B = "10.0.0.2"
ETHERNET_A = "10.1.0.1"
ETHERNET_B = "10.1.0.2"
THUNDERBOLT_A = "169.254.1.1"
THUNDERBOLT_B = "169.254.1.2"
ALT_ETHERNET_B = "10.1.0.22"


@final
class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now = self._now + delta


def _model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=4,
        hidden_size=32,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _shard(device_rank: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_model_card(),
        device_rank=device_rank,
        world_size=2,
        start_layer=device_rank * 2,
        end_layer=(device_rank + 1) * 2,
        n_layers=4,
    )


def _assignments() -> ShardAssignments:
    return ShardAssignments(
        model_id=ModelId("test-model"),
        runner_to_shard={RUNNER_A: _shard(0), RUNNER_B: _shard(1)},
        node_to_runner={NODE_A: RUNNER_A, NODE_B: RUNNER_B},
    )


def _socket(ip_address: str) -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address=f"/ip4/{ip_address}/tcp/52415")
    )


def _link(source: NodeId, sink: NodeId, sink_ip_address: str) -> Connection:
    return Connection(source=source, sink=sink, edge=_socket(sink_ip_address))


def _add_pair(topology: Topology, source_ip: str, sink_ip: str) -> None:
    topology.add_connection(_link(NODE_A, NODE_B, sink_ip))
    topology.add_connection(_link(NODE_B, NODE_A, source_ip))


def _interface(
    name: str, ip_address: str, interface_type: InterfaceType
) -> NetworkInterfaceInfo:
    return NetworkInterfaceInfo(
        name=name, ip_address=ip_address, interface_type=interface_type
    )


def _node_network(*interfaces: NetworkInterfaceInfo) -> NodeNetworkInfo:
    return NodeNetworkInfo(interfaces=interfaces)


def _full_network() -> dict[NodeId, NodeNetworkInfo]:
    return {
        NODE_A: _node_network(
            _interface("en0", WIFI_A, "wifi"),
            _interface("en1", ETHERNET_A, "ethernet"),
            _interface("en2", THUNDERBOLT_A, "thunderbolt"),
        ),
        NODE_B: _node_network(
            _interface("en0", WIFI_B, "wifi"),
            _interface("en1", ETHERNET_B, "ethernet"),
            _interface("en2", THUNDERBOLT_B, "thunderbolt"),
            _interface("en3", ALT_ETHERNET_B, "ethernet"),
        ),
    }


def _hosts(neighbor_a: str, neighbor_b: str) -> dict[NodeId, list[Host]]:
    return {
        NODE_A: [
            Host(ip="0.0.0.0", port=EPHEMERAL_PORT),
            Host(ip=neighbor_b, port=EPHEMERAL_PORT),
        ],
        NODE_B: [
            Host(ip=neighbor_a, port=EPHEMERAL_PORT),
            Host(ip="0.0.0.0", port=EPHEMERAL_PORT),
        ],
    }


def _ring_instance(neighbor_a: str, neighbor_b: str) -> MlxRingInstance:
    return MlxRingInstance(
        instance_id=INSTANCE_ID,
        shard_assignments=_assignments(),
        hosts_by_node=_hosts(neighbor_a, neighbor_b),
        ephemeral_port=EPHEMERAL_PORT,
    )


def _jaccl_instance(coordinator_from_b: str) -> MlxJacclInstance:
    return MlxJacclInstance(
        instance_id=INSTANCE_ID,
        shard_assignments=_assignments(),
        jaccl_devices=[[None, "rdma_en3"], ["rdma_en3", None]],
        jaccl_coordinators={
            NODE_A: "0.0.0.0:6000",
            NODE_B: f"{coordinator_from_b}:6000",
        },
    )


def test_ring_priority_ranks_thunderbolt_above_ethernet_and_wifi() -> None:
    assert interface_priority_score("thunderbolt", ring=True) < (
        interface_priority_score("ethernet", ring=True)
    )
    assert interface_priority_score("ethernet", ring=True) < (
        interface_priority_score("wifi", ring=True)
    )


def test_wifi_instance_switches_to_thunderbolt_when_it_appears() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, WIFI_A, WIFI_B)
    _add_pair(topology, THUNDERBOLT_A, THUNDERBOLT_B)

    instance = _ring_instance(WIFI_A, WIFI_B)
    replacement = instance_with_preferred_connections(
        instance, topology, _full_network()
    )

    assert replacement is not None
    assert replacement.instance_id == INSTANCE_ID
    assert isinstance(replacement, MlxRingInstance)
    assert replacement.hosts_by_node == _hosts(THUNDERBOLT_A, THUNDERBOLT_B)
    assert replacement.shard_assignments == instance.shard_assignments
    assert replacement.ephemeral_port == EPHEMERAL_PORT


def test_already_on_thunderbolt_does_not_replace() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, WIFI_A, WIFI_B)
    _add_pair(topology, THUNDERBOLT_A, THUNDERBOLT_B)

    instance = _ring_instance(THUNDERBOLT_A, THUNDERBOLT_B)
    assert (
        instance_with_preferred_connections(instance, topology, _full_network()) is None
    )


def test_same_priority_ip_change_does_not_replace() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, ETHERNET_A, ALT_ETHERNET_B)
    _add_pair(topology, ETHERNET_A, ETHERNET_B)

    instance = _ring_instance(ETHERNET_A, ETHERNET_B)
    assert (
        instance_with_preferred_connections(instance, topology, _full_network()) is None
    )


def test_dead_path_fails_over_to_remaining_link() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, WIFI_A, WIFI_B)

    instance = _ring_instance(ETHERNET_A, ETHERNET_B)
    replacement = instance_with_preferred_connections(
        instance, topology, _full_network()
    )

    assert replacement is not None
    assert isinstance(replacement, MlxRingInstance)
    assert replacement.hosts_by_node == _hosts(WIFI_A, WIFI_B)


def test_incomplete_topology_does_not_replace() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    instance = _ring_instance(WIFI_A, WIFI_B)
    assert (
        instance_with_preferred_connections(instance, topology, _full_network()) is None
    )


def test_single_node_instance_does_not_replace() -> None:
    node = NodeId("solo")
    runner = RunnerId("solo-runner")
    instance = MlxRingInstance(
        instance_id=INSTANCE_ID,
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=_model_card(),
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=4,
                    n_layers=4,
                )
            },
            node_to_runner={node: runner},
        ),
        hosts_by_node={node: [Host(ip="0.0.0.0", port=EPHEMERAL_PORT)]},
        ephemeral_port=EPHEMERAL_PORT,
    )
    topology = Topology()
    topology.add_node(node)
    assert instance_with_preferred_connections(instance, topology, {}) is None


def test_jaccl_coordinator_switches_wifi_to_ethernet() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, WIFI_A, WIFI_B)
    _add_pair(topology, ETHERNET_A, ETHERNET_B)

    instance = _jaccl_instance(WIFI_A)
    replacement = instance_with_preferred_connections(
        instance, topology, _full_network()
    )

    assert replacement is not None
    assert isinstance(replacement, MlxJacclInstance)
    assert replacement.instance_id == INSTANCE_ID
    assert replacement.jaccl_coordinators[NODE_B] == f"{ETHERNET_A}:6000"
    assert replacement.jaccl_devices == instance.jaccl_devices


def test_connection_upgrade_events_emit_atomic_replace() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, WIFI_A, WIFI_B)
    _add_pair(topology, THUNDERBOLT_A, THUNDERBOLT_B)

    instance = _ring_instance(WIFI_A, WIFI_B)
    events = connection_upgrade_events(
        {INSTANCE_ID: instance}, topology, _full_network()
    )

    assert len(events) == 1
    assert isinstance(events[0], InstanceReplacedAtomically)
    assert events[0].instance.instance_id == INSTANCE_ID


def test_scheduler_waits_for_stable_window_with_fake_clock() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, WIFI_A, WIFI_B)
    _add_pair(topology, THUNDERBOLT_A, THUNDERBOLT_B)

    instance = _ring_instance(WIFI_A, WIFI_B)
    clock = FakeClock()
    scheduler = ConnectionUpgradeScheduler(clock=clock, stable_for=timedelta(seconds=2))

    assert scheduler.events({INSTANCE_ID: instance}, topology, _full_network()) == []

    clock.advance(timedelta(seconds=1))
    assert scheduler.events({INSTANCE_ID: instance}, topology, _full_network()) == []

    clock.advance(timedelta(seconds=1))
    events = scheduler.events({INSTANCE_ID: instance}, topology, _full_network())
    assert len(events) == 1
    assert isinstance(events[0], InstanceReplacedAtomically)
    assert isinstance(events[0].instance, MlxRingInstance)
    assert events[0].instance.hosts_by_node == _hosts(THUNDERBOLT_A, THUNDERBOLT_B)


def test_scheduler_resets_when_candidate_changes() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    _add_pair(topology, WIFI_A, WIFI_B)
    _add_pair(topology, ETHERNET_A, ETHERNET_B)

    instance = _ring_instance(WIFI_A, WIFI_B)
    clock = FakeClock()
    scheduler = ConnectionUpgradeScheduler(clock=clock, stable_for=timedelta(seconds=2))

    assert scheduler.events({INSTANCE_ID: instance}, topology, _full_network()) == []

    clock.advance(timedelta(seconds=1))
    _add_pair(topology, THUNDERBOLT_A, THUNDERBOLT_B)
    assert scheduler.events({INSTANCE_ID: instance}, topology, _full_network()) == []

    clock.advance(timedelta(seconds=1))
    assert scheduler.events({INSTANCE_ID: instance}, topology, _full_network()) == []

    clock.advance(timedelta(seconds=1))
    events = scheduler.events({INSTANCE_ID: instance}, topology, _full_network())
    assert len(events) == 1
    assert isinstance(events[0].instance, MlxRingInstance)
    assert events[0].instance.hosts_by_node == _hosts(THUNDERBOLT_A, THUNDERBOLT_B)


def test_scheduler_clears_pending_when_upgrade_disappears() -> None:
    topology = Topology()
    topology.add_node(NODE_A)
    topology.add_node(NODE_B)
    wifi = _link(NODE_A, NODE_B, WIFI_B)
    wifi_back = _link(NODE_B, NODE_A, WIFI_A)
    thunderbolt = _link(NODE_A, NODE_B, THUNDERBOLT_B)
    thunderbolt_back = _link(NODE_B, NODE_A, THUNDERBOLT_A)
    topology.add_connection(wifi)
    topology.add_connection(wifi_back)
    topology.add_connection(thunderbolt)
    topology.add_connection(thunderbolt_back)

    instance = _ring_instance(WIFI_A, WIFI_B)
    clock = FakeClock()
    scheduler = ConnectionUpgradeScheduler(clock=clock, stable_for=timedelta(seconds=2))
    assert scheduler.events({INSTANCE_ID: instance}, topology, _full_network()) == []

    topology.remove_connection(thunderbolt)
    topology.remove_connection(thunderbolt_back)
    clock.advance(timedelta(seconds=2))
    assert scheduler.events({INSTANCE_ID: instance}, topology, _full_network()) == []
