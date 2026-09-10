from exo.shared.link_profile import (
    advertised_thunderbolt_link_speed,
    build_link_profiles,
    compose_link_profile,
    elapsed_milliseconds,
    estimate_bandwidth_bps,
    interface_for_ip,
    nic_speed_mbps_or_none,
    parse_advertised_link_speed_bps,
)
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    LinkProfile,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeThunderboltInfo,
)
from exo.shared.types.state import State
from exo.shared.types.thunderbolt import ThunderboltIdentifier


def test_nic_speed_mbps_or_none_treats_zero_as_unknown() -> None:
    assert nic_speed_mbps_or_none(0) is None
    assert nic_speed_mbps_or_none(-1) is None
    assert nic_speed_mbps_or_none(1000) == 1000


def test_elapsed_milliseconds_clamps_negative() -> None:
    assert elapsed_milliseconds(1.0, 1.0125) == 12.5
    assert elapsed_milliseconds(2.0, 1.0) == 0.0


def test_parse_advertised_link_speed_bps() -> None:
    assert parse_advertised_link_speed_bps("") is None
    assert parse_advertised_link_speed_bps("not a speed") is None
    assert parse_advertised_link_speed_bps("40 Gb/s") == 40_000_000_000
    assert parse_advertised_link_speed_bps("10Gbps") == 10_000_000_000
    assert parse_advertised_link_speed_bps("1000 Mb/s") == 1_000_000_000
    assert parse_advertised_link_speed_bps("600") == 600_000_000


def test_estimate_bandwidth_prefers_nic_speed() -> None:
    bits, source = estimate_bandwidth_bps(
        nic_speed_mbps=25000,
        interface_type="wifi",
        advertised_link_speed="40 Gb/s",
    )
    assert bits == 25_000_000_000
    assert source == "nic_speed"


def test_estimate_bandwidth_uses_advertised_then_type() -> None:
    bits, source = estimate_bandwidth_bps(
        nic_speed_mbps=None,
        interface_type="thunderbolt",
        advertised_link_speed="40 Gb/s",
    )
    assert bits == 40_000_000_000
    assert source == "thunderbolt_link"

    bits, source = estimate_bandwidth_bps(
        nic_speed_mbps=None,
        interface_type="ethernet",
        advertised_link_speed="",
    )
    assert bits == 1_000_000_000
    assert source == "interface_type"

    bits, source = estimate_bandwidth_bps(
        nic_speed_mbps=None,
        interface_type="unknown",
        advertised_link_speed="",
    )
    assert bits is None
    assert source == "interface_type"


def test_compose_and_lookup_link_profile() -> None:
    remote = NodeId("node-b")
    interface = NetworkInterfaceInfo(
        name="en0",
        ip_address="10.0.0.2",
        interface_type="ethernet",
        nic_speed_mbps=1000,
    )
    network = NodeNetworkInfo(interfaces=[interface])
    assert interface_for_ip(network, "10.0.0.2") == interface
    assert interface_for_ip(network, "10.0.0.9") is None

    profile = compose_link_profile(
        remote_node_id=remote,
        remote_ip="10.0.0.2",
        latency_ms=1.5,
        interface=interface,
    )
    assert profile.latency_ms == 1.5
    assert profile.bandwidth_bps == 1_000_000_000
    assert profile.bandwidth_source == "nic_speed"


def test_build_link_profiles_from_samples_and_rdma() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    node_c = NodeId("node-c")
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="10.0.0.2",
                    interface_type="ethernet",
                    nic_speed_mbps=10000,
                )
            ]
        )
    }
    node_thunderbolt = {
        node_c: NodeThunderboltInfo(
            interfaces=[
                ThunderboltIdentifier(
                    rdma_interface="rdma_en2",
                    domain_uuid="uuid-c",
                    link_speed="40 Gb/s",
                )
            ]
        )
    }

    profiles = build_link_profiles(
        samples=[(node_b, "10.0.0.2", 2.25)],
        node_network=node_network,
        node_thunderbolt=node_thunderbolt,
        rdma_sinks=(node_c, node_b),
    )
    by_remote = {profile.remote_node_id: profile for profile in profiles}
    assert node_a not in by_remote
    assert by_remote[node_b].latency_ms == 2.25
    assert by_remote[node_b].bandwidth_bps == 10_000_000_000
    assert by_remote[node_c].latency_ms is None
    assert by_remote[node_c].bandwidth_bps == 40_000_000_000
    assert by_remote[node_c].bandwidth_source == "thunderbolt_link"
    assert advertised_thunderbolt_link_speed(node_thunderbolt[node_c]) == "40 Gb/s"


def test_link_profile_state_json_uses_camel_case() -> None:
    source = NodeId("node-a")
    remote = NodeId("node-b")
    profile = LinkProfile(
        remote_node_id=remote,
        remote_ip="10.0.0.2",
        latency_ms=2.5,
        bandwidth_bps=1_000_000_000,
        bandwidth_source="nic_speed",
    )
    state = State(node_link_profiles={source: (profile,)})
    dumped = state.model_dump(by_alias=True)
    assert "nodeLinkProfiles" in dumped
    restored = State.model_validate(dumped)
    assert restored.node_link_profiles[source][0].latency_ms == 2.5
    assert restored.node_link_profiles[source][0].bandwidth_bps == 1_000_000_000
