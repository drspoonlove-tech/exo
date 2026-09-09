from pathlib import Path

from exo.utils.info_gatherer.interface_classification import (
    InterfaceClassificationHints,
    classify_interface_type,
    collect_sysfs_classification_hints,
    connection_type_label,
    parse_networksetup_hardware_ports,
)

NETWORKSETUP_SAMPLE = """
Hardware Port: Ethernet
Device: en0
Ethernet Address: 11:22:33:44:55:66

Hardware Port: Wi-Fi
Device: en1
Ethernet Address: aa:bb:cc:dd:ee:ff

Hardware Port: Thunderbolt 1
Device: en2
Ethernet Address: 01:23:45:67:89:ab

Hardware Port: Thunderbolt 5
Device: en3
Ethernet Address: fe:dc:ba:98:76:54

Hardware Port: iPhone USB
Device: en7
Ethernet Address: 00:11:22:33:44:55
"""


def test_networksetup_keeps_thunderbolt_on_en_ports() -> None:
    types = parse_networksetup_hardware_ports(NETWORKSETUP_SAMPLE)
    assert types["en0"] == "ethernet"
    assert types["en1"] == "wifi"
    assert types["en2"] == "thunderbolt"
    assert types["en3"] == "thunderbolt"
    assert types["en7"] == "maybe_ethernet"


def test_classify_by_interface_name() -> None:
    assert classify_interface_type("wlan0") == "wifi"
    assert classify_interface_type("wlp3s0") == "wifi"
    assert classify_interface_type("eth0") == "ethernet"
    assert classify_interface_type("enp0s3") == "ethernet"
    assert classify_interface_type("ens1") == "ethernet"
    assert classify_interface_type("tb0") == "thunderbolt"
    assert classify_interface_type("en0") == "wifi"
    assert classify_interface_type("en2") == "maybe_ethernet"
    assert classify_interface_type("lo") == "unknown"
    assert classify_interface_type("docker0") == "unknown"
    assert classify_interface_type("utun0") == "unknown"


def test_hardware_port_wins_over_name_heuristic() -> None:
    assert (
        classify_interface_type(
            "en2",
            InterfaceClassificationHints(hardware_port_name="Thunderbolt 4"),
        )
        == "thunderbolt"
    )
    assert (
        classify_interface_type(
            "en0",
            InterfaceClassificationHints(hardware_port_name="Wi-Fi"),
        )
        == "wifi"
    )


def test_sysfs_wireless_and_thunderbolt_device_path() -> None:
    assert (
        classify_interface_type(
            "enp3s0",
            InterfaceClassificationHints(sysfs_wireless=True),
        )
        == "wifi"
    )
    assert (
        classify_interface_type(
            "enp1s0",
            InterfaceClassificationHints(
                sysfs_device_path="/sys/devices/pci0000:00/0000:00:1c.0/domain0/0-1/thunderbolt/0-1"
            ),
        )
        == "thunderbolt"
    )


def test_collect_sysfs_hints_from_synthetic_tree(tmp_path: Path) -> None:
    wifi = tmp_path / "wlp3s0"
    (wifi / "wireless").mkdir(parents=True)
    (wifi / "device").symlink_to("/sys/devices/pci0000:00/0000:00:1c.4")

    thunderbolt = tmp_path / "enp2s0"
    thunderbolt.mkdir()
    (thunderbolt / "device").symlink_to(
        "/sys/devices/pci0000:00/domain0/0-1/thunderbolt/0-1"
    )

    missing = collect_sysfs_classification_hints("nope", sysfs_class_net=tmp_path)
    assert missing.sysfs_wireless is False
    assert missing.sysfs_device_path is None

    wifi_hints = collect_sysfs_classification_hints("wlp3s0", sysfs_class_net=tmp_path)
    assert wifi_hints.sysfs_wireless is True
    assert classify_interface_type("wlp3s0", wifi_hints) == "wifi"

    thunderbolt_hints = collect_sysfs_classification_hints(
        "enp2s0", sysfs_class_net=tmp_path
    )
    assert thunderbolt_hints.sysfs_device_path is not None
    assert "thunderbolt" in thunderbolt_hints.sysfs_device_path
    assert classify_interface_type("enp2s0", thunderbolt_hints) == "thunderbolt"


def test_connection_type_labels() -> None:
    assert connection_type_label("rdma") == "RDMA"
    assert connection_type_label("wifi") == "Wi-Fi"
    assert connection_type_label("ethernet") == "Ethernet"
    assert connection_type_label("maybe_ethernet") == "Ethernet"
    assert connection_type_label("unknown") == "Unknown"
    assert connection_type_label("thunderbolt") == "Thunderbolt"
    assert (
        connection_type_label("thunderbolt", hardware_port_name="Thunderbolt 5")
        == "TB5"
    )
    assert (
        connection_type_label("thunderbolt", thunderbolt_link_speed="80 Gb/s") == "TB5"
    )
    assert (
        connection_type_label("thunderbolt", thunderbolt_link_speed="40 Gb/s")
        == "Thunderbolt"
    )
