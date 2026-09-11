from types import SimpleNamespace

from exo.shared.network_utilization import (
    InterfaceByteCounters,
    NetworkUtilizationSampler,
    interface_byte_counters_from_per_nic,
    is_loopback_interface_name,
    read_per_interface_byte_counters,
    sink_interface_name_for_connection,
    source_interface_name_for_connection,
    utilization_for_connection,
    utilization_from_counter_delta,
    utilization_sample_from_observation,
)
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NetworkInterfaceUtilization,
    NodeNetworkInfo,
    NodeNetworkUtilization,
)
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, RDMAConnection, SocketConnection


def test_loopback_names_are_filtered() -> None:
    assert is_loopback_interface_name("lo")
    assert is_loopback_interface_name("lo0")
    assert is_loopback_interface_name("LO1")
    assert not is_loopback_interface_name("eth0")
    assert not is_loopback_interface_name("en0")
    assert not is_loopback_interface_name("docker0")


def test_counters_from_per_nic_skip_loopback() -> None:
    snapshot = interface_byte_counters_from_per_nic(
        {
            "lo": SimpleNamespace(bytes_sent=10, bytes_recv=10),
            "eth0": SimpleNamespace(bytes_sent=100, bytes_recv=40),
        }
    )
    assert set(snapshot) == {"eth0"}
    assert snapshot["eth0"] == InterfaceByteCounters(
        name="eth0", bytes_sent=100, bytes_recv=40
    )


def test_delta_computes_bytes_per_second() -> None:
    previous = InterfaceByteCounters(name="eth0", bytes_sent=1_000, bytes_recv=200)
    current = InterfaceByteCounters(name="eth0", bytes_sent=3_500, bytes_recv=1_200)
    utilization = utilization_from_counter_delta(previous, current, 2.0)
    assert utilization == NetworkInterfaceUtilization(
        name="eth0",
        bytes_sent_per_sec=1_250.0,
        bytes_recv_per_sec=500.0,
    )


def test_delta_rejects_reset_and_tiny_windows() -> None:
    previous = InterfaceByteCounters(name="eth0", bytes_sent=5_000, bytes_recv=200)
    reset = InterfaceByteCounters(name="eth0", bytes_sent=10, bytes_recv=200)
    assert utilization_from_counter_delta(previous, reset, 1.0) is None
    current = InterfaceByteCounters(name="eth0", bytes_sent=6_000, bytes_recv=300)
    assert utilization_from_counter_delta(previous, current, 0.0) is None
    assert utilization_from_counter_delta(None, current, 1.0) is None


def test_sampler_needs_two_ticks_then_reports_rates() -> None:
    sampler = NetworkUtilizationSampler()
    first = utilization_sample_from_observation(
        sampler,
        {"eth0": InterfaceByteCounters(name="eth0", bytes_sent=100, bytes_recv=50)},
        monotonic_seconds=10.0,
    )
    assert first is None

    second = utilization_sample_from_observation(
        sampler,
        {"eth0": InterfaceByteCounters(name="eth0", bytes_sent=1_100, bytes_recv=250)},
        monotonic_seconds=11.0,
    )
    assert second is not None
    assert list(second.interfaces) == [
        NetworkInterfaceUtilization(
            name="eth0",
            bytes_sent_per_sec=1_000.0,
            bytes_recv_per_sec=200.0,
        )
    ]


def test_rdma_connection_uses_source_nic_rate() -> None:
    connection = Connection(
        source=NodeId("node-a"),
        sink=NodeId("node-b"),
        edge=RDMAConnection(source_rdma_iface="en2", sink_rdma_iface="en5"),
    )
    assert source_interface_name_for_connection(connection) == "en2"
    utilization = utilization_for_connection(
        connection,
        source_utilization=NodeNetworkUtilization(
            interfaces=[
                NetworkInterfaceUtilization(
                    name="en2",
                    bytes_sent_per_sec=40_000_000.0,
                    bytes_recv_per_sec=1_000.0,
                )
            ]
        ),
        sink_utilization=None,
    )
    assert utilization is not None
    assert utilization.name == "en2"
    assert utilization.bytes_sent_per_sec == 40_000_000.0


def test_socket_connection_uses_sink_nic_that_owns_the_address() -> None:
    connection = Connection(
        source=NodeId("node-a"),
        sink=NodeId("node-b"),
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/10.0.0.8/tcp/52415")
        ),
    )
    assert source_interface_name_for_connection(connection) is None
    sink_network = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(
                name="eth0", ip_address="10.0.0.8", interface_type="ethernet"
            )
        ]
    )
    assert sink_interface_name_for_connection(connection, sink_network) == "eth0"
    utilization = utilization_for_connection(
        connection,
        source_utilization=None,
        sink_utilization=NodeNetworkUtilization(
            interfaces=[
                NetworkInterfaceUtilization(
                    name="eth0",
                    bytes_sent_per_sec=100.0,
                    bytes_recv_per_sec=9_000.0,
                )
            ]
        ),
        sink_network=sink_network,
    )
    assert utilization is not None
    assert utilization.bytes_recv_per_sec == 9_000.0


def test_state_json_uses_camel_case_overlay_field() -> None:
    node_id = NodeId("node-a")
    state = State(
        node_network_utilization={
            node_id: NodeNetworkUtilization(
                interfaces=[
                    NetworkInterfaceUtilization(
                        name="eth0",
                        bytes_sent_per_sec=12.5,
                        bytes_recv_per_sec=4.0,
                    )
                ]
            )
        }
    )
    dumped = state.model_dump(by_alias=True)
    overlay = dumped["nodeNetworkUtilization"]
    assert overlay
    reported = next(iter(overlay.values()))
    assert reported["interfaces"][0]["bytesSentPerSec"] == 12.5


def test_read_real_psutil_counters_include_a_non_loopback_nic() -> None:
    snapshot = read_per_interface_byte_counters()
    assert snapshot
    assert all(not is_loopback_interface_name(name) for name in snapshot)
    for counters in snapshot.values():
        assert counters.bytes_sent >= 0
        assert counters.bytes_recv >= 0
