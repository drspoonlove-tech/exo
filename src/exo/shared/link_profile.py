import re
from collections.abc import Mapping, Sequence
from typing import Final

from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    BandwidthSource,
    InterfaceType,
    LinkProfile,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeThunderboltInfo,
)

_LINK_SPEED_PATTERN: Final = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>tb/?s|gb/?s|mb/?s|kb/?s|tbps|gbps|mbps|kbps)?",
    re.IGNORECASE,
)

_UNIT_TO_BITS: Final[Mapping[str, int]] = {
    "kbps": 1_000,
    "kbs": 1_000,
    "mbps": 1_000_000,
    "mbs": 1_000_000,
    "gbps": 1_000_000_000,
    "gbs": 1_000_000_000,
    "tbps": 1_000_000_000_000,
    "tbs": 1_000_000_000_000,
}

INTERFACE_TYPE_BANDWIDTH_BPS: Final[Mapping[InterfaceType, int | None]] = {
    "thunderbolt": 40_000_000_000,
    "ethernet": 1_000_000_000,
    "maybe_ethernet": 1_000_000_000,
    "wifi": 600_000_000,
    "unknown": None,
}


def nic_speed_mbps_or_none(reported_mbps: int) -> int | None:
    """psutil reports ``0`` when the NIC speed is unknown."""
    return reported_mbps if reported_mbps > 0 else None


def elapsed_milliseconds(start_seconds: float, end_seconds: float) -> float:
    return max(0.0, (end_seconds - start_seconds) * 1000.0)


def parse_advertised_link_speed_bps(advertised: str) -> int | None:
    """Parse strings such as ``40 Gb/s``, ``10Gbps``, or ``1000 Mb/s``."""
    text = advertised.strip()
    if not text:
        return None
    match = _LINK_SPEED_PATTERN.search(text)
    if match is None:
        return None
    value = float(match.group("value"))
    raw_unit = match.group("unit")
    unit = (raw_unit or "mbps").lower().replace("/", "")
    multiplier = _UNIT_TO_BITS.get(unit)
    if multiplier is None:
        return None
    bits = int(value * multiplier)
    return bits if bits > 0 else None


def estimate_bandwidth_bps(
    *,
    nic_speed_mbps: int | None,
    interface_type: InterfaceType,
    advertised_link_speed: str = "",
) -> tuple[int | None, BandwidthSource]:
    """Prefer measured NIC speed, then an advertised link string, then type."""
    if nic_speed_mbps is not None and nic_speed_mbps > 0:
        return nic_speed_mbps * 1_000_000, "nic_speed"
    parsed = parse_advertised_link_speed_bps(advertised_link_speed)
    if parsed is not None:
        return parsed, "thunderbolt_link"
    return INTERFACE_TYPE_BANDWIDTH_BPS[interface_type], "interface_type"


def advertised_thunderbolt_link_speed(
    thunderbolt: NodeThunderboltInfo | None,
) -> str:
    if thunderbolt is None:
        return ""
    for identifier in thunderbolt.interfaces:
        if identifier.link_speed:
            return identifier.link_speed
    return ""


def interface_for_ip(
    network: NodeNetworkInfo | None, ip_address: str
) -> NetworkInterfaceInfo | None:
    if network is None:
        return None
    for interface in network.interfaces:
        if interface.ip_address == ip_address:
            return interface
    return None


def compose_link_profile(
    *,
    remote_node_id: NodeId,
    remote_ip: str,
    latency_ms: float | None,
    interface: NetworkInterfaceInfo | None,
    advertised_link_speed: str = "",
    interface_type: InterfaceType | None = None,
) -> LinkProfile:
    resolved_type: InterfaceType = (
        interface.interface_type
        if interface is not None
        else (interface_type or "unknown")
    )
    nic_speed = interface.nic_speed_mbps if interface is not None else None
    bandwidth_bps, bandwidth_source = estimate_bandwidth_bps(
        nic_speed_mbps=nic_speed,
        interface_type=resolved_type,
        advertised_link_speed=advertised_link_speed,
    )
    return LinkProfile(
        remote_node_id=remote_node_id,
        remote_ip=remote_ip,
        latency_ms=latency_ms,
        bandwidth_bps=bandwidth_bps,
        bandwidth_source=bandwidth_source,
    )


def build_link_profiles(
    *,
    samples: Sequence[tuple[NodeId, str, float]],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_thunderbolt: Mapping[NodeId, NodeThunderboltInfo],
    rdma_sinks: Sequence[NodeId] = (),
) -> tuple[LinkProfile, ...]:
    """Compose directed link profiles from HTTP RTT samples and RDMA sinks.

    ``samples`` is ``(remote_node_id, remote_ip, latency_ms)``.
    RDMA sinks get bandwidth-only profiles (no HTTP RTT).
    """
    profiles: dict[tuple[NodeId, str], LinkProfile] = {}
    for remote_node_id, remote_ip, latency_ms in samples:
        interface = interface_for_ip(node_network.get(remote_node_id), remote_ip)
        advertised = (
            advertised_thunderbolt_link_speed(node_thunderbolt.get(remote_node_id))
            if interface is None or interface.interface_type == "thunderbolt"
            else ""
        )
        profiles[(remote_node_id, remote_ip)] = compose_link_profile(
            remote_node_id=remote_node_id,
            remote_ip=remote_ip,
            latency_ms=latency_ms,
            interface=interface,
            advertised_link_speed=advertised,
        )

    for sink_node_id in rdma_sinks:
        key = (sink_node_id, "")
        if key in profiles:
            continue
        if any(remote_id == sink_node_id for remote_id, _ip in profiles):
            continue
        profiles[key] = compose_link_profile(
            remote_node_id=sink_node_id,
            remote_ip="",
            latency_ms=None,
            interface=None,
            advertised_link_speed=advertised_thunderbolt_link_speed(
                node_thunderbolt.get(sink_node_id)
            ),
            interface_type="thunderbolt",
        )

    return tuple(profiles.values())
