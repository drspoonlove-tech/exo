from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, final

from exo.shared.types.profiling import InterfaceType

ConnectionTypeLabel = Literal[
    "TB5", "Thunderbolt", "Ethernet", "Wi-Fi", "RDMA", "Unknown"
]

_THUNDERBOLT_5_SPEED_MARKERS: Final[tuple[str, ...]] = (
    "80 gb",
    "80gb",
    "120 gb",
    "120gb",
    "tb5",
    "thunderbolt 5",
    "thunderbolt5",
)


@final
@dataclass(frozen=True)
class InterfaceClassificationHints:
    hardware_port_name: str | None = None
    sysfs_wireless: bool = False
    sysfs_device_path: str | None = None


def _normalized(value: str | None) -> str:
    return (value or "").strip().lower()


def _hardware_port_type(hardware_port_name: str | None) -> InterfaceType | None:
    port_name = _normalized(hardware_port_name)
    if not port_name:
        return None
    if "thunderbolt" in port_name or port_name.startswith("tb5"):
        return "thunderbolt"
    if "wi-fi" in port_name or "wifi" in port_name or "airport" in port_name:
        return "wifi"
    if "ethernet" in port_name or port_name.endswith(" lan") or port_name == "lan":
        return "ethernet"
    return None


def _sysfs_type(
    *, sysfs_wireless: bool, sysfs_device_path: str | None
) -> InterfaceType | None:
    if sysfs_wireless:
        return "wifi"
    device_path = _normalized(sysfs_device_path)
    if "thunderbolt" in device_path:
        return "thunderbolt"
    return None


def _interface_name_type(interface_name: str) -> InterfaceType:
    name = _normalized(interface_name)
    if not name:
        return "unknown"

    if name.startswith(("docker", "br-", "veth", "cni", "flannel", "calico", "weave")):
        return "unknown"
    if "bridge" in name:
        return "unknown"
    if name.startswith("lo"):
        return "unknown"
    if name.startswith(("tun", "tap", "vtun", "utun", "gif", "stf", "awdl", "llw")):
        return "unknown"

    if name.startswith(("tb", "nx", "ten")):
        return "thunderbolt"
    if name.startswith(("wlan", "wifi", "wl")):
        return "wifi"
    if name.startswith(("eth", "enp", "ens", "eno")):
        return "ethernet"
    if name in {"en0", "en1"}:
        return "wifi"
    if name.startswith("en"):
        return "maybe_ethernet"
    return "unknown"


def classify_interface_type(
    interface_name: str,
    hints: InterfaceClassificationHints | None = None,
) -> InterfaceType:
    """Classify a NIC from a hardware-port name, sysfs hints, or the iface name.

    Hardware-port names (macOS ``networksetup``) win, then sysfs, then the
    iface-name heuristics from classic exo ``helpers.py``.
    """
    resolved_hints = hints or InterfaceClassificationHints()
    return (
        _hardware_port_type(resolved_hints.hardware_port_name)
        or _sysfs_type(
            sysfs_wireless=resolved_hints.sysfs_wireless,
            sysfs_device_path=resolved_hints.sysfs_device_path,
        )
        or _interface_name_type(interface_name)
    )


def parse_networksetup_hardware_ports(output: str) -> dict[str, InterfaceType]:
    """Parse ``networksetup -listallhardwareports`` into device → type."""
    types: dict[str, InterfaceType] = {}
    current_port = ""
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Hardware Port:"):
            current_port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and current_port:
            device = line.split(":", 1)[1].strip()
            types[device] = classify_interface_type(
                device,
                InterfaceClassificationHints(hardware_port_name=current_port),
            )
            current_port = ""
    return types


def collect_sysfs_classification_hints(
    interface_name: str, *, sysfs_class_net: Path = Path("/sys/class/net")
) -> InterfaceClassificationHints:
    """Read optional Linux sysfs hints. Missing paths yield empty hints."""
    interface_directory = sysfs_class_net / interface_name
    if not interface_directory.exists():
        return InterfaceClassificationHints()

    wireless_present = (interface_directory / "wireless").exists()
    device_path = interface_directory / "device"
    resolved_device_path: str | None = None
    if device_path.exists() or device_path.is_symlink():
        try:
            resolved_device_path = str(device_path.resolve())
        except OSError:
            resolved_device_path = str(device_path)

    return InterfaceClassificationHints(
        sysfs_wireless=wireless_present,
        sysfs_device_path=resolved_device_path,
    )


def is_thunderbolt_5_link_speed(link_speed: str | None) -> bool:
    normalized = _normalized(link_speed)
    return any(marker in normalized for marker in _THUNDERBOLT_5_SPEED_MARKERS)


def connection_type_label(
    interface_type: InterfaceType | Literal["rdma"],
    *,
    hardware_port_name: str | None = None,
    thunderbolt_link_speed: str | None = None,
) -> ConnectionTypeLabel:
    """Human-readable edge/interface label for the topology UI."""
    if interface_type == "rdma":
        return "RDMA"
    if interface_type == "wifi":
        return "Wi-Fi"
    if interface_type in {"ethernet", "maybe_ethernet"}:
        return "Ethernet"
    if interface_type == "thunderbolt":
        if is_thunderbolt_5_link_speed(
            thunderbolt_link_speed
        ) or is_thunderbolt_5_link_speed(hardware_port_name):
            return "TB5"
        return "Thunderbolt"
    return "Unknown"
