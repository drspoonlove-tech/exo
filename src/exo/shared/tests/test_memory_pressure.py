from pathlib import Path

import pytest

from exo.shared.apply import apply_node_gathered_info
from exo.shared.memory_pressure import (
    compute_memory_pressure,
    parse_linux_memory_pressure_stall_average_10,
    parse_macos_memory_pressure_level,
    read_linux_memory_pressure_stall_average_10,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeGatheredInfo
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.utils.info_gatherer.macmon import MacmonMetrics, RawMacmonMetrics


def test_scarcity_uses_available_not_used() -> None:
    assert (
        compute_memory_pressure(ram_total_bytes=16_000, ram_available_bytes=4_000)
        == 0.75
    )


def test_zero_total_is_full_pressure() -> None:
    assert compute_memory_pressure(ram_total_bytes=0, ram_available_bytes=0) == 1.0


def test_available_above_total_is_no_pressure() -> None:
    assert compute_memory_pressure(ram_total_bytes=10, ram_available_bytes=20) == 0.0


def test_linux_stall_raises_pressure_above_scarcity() -> None:
    assert (
        compute_memory_pressure(
            ram_total_bytes=100,
            ram_available_bytes=50,
            linux_pressure_stall_average_10=80.0,
        )
        == 0.8
    )


def test_linux_stall_does_not_lower_scarcity() -> None:
    assert (
        compute_memory_pressure(
            ram_total_bytes=100,
            ram_available_bytes=10,
            linux_pressure_stall_average_10=5.0,
        )
        == 0.9
    )


def test_macos_urgent_raises_pressure_above_scarcity() -> None:
    assert (
        compute_memory_pressure(
            ram_total_bytes=100,
            ram_available_bytes=70,
            macos_memory_pressure_level=2,
        )
        == 0.8
    )


def test_macos_normal_does_not_raise_scarcity() -> None:
    assert (
        compute_memory_pressure(
            ram_total_bytes=100,
            ram_available_bytes=25,
            macos_memory_pressure_level=0,
        )
        == 0.75
    )


def test_unknown_macos_level_is_ignored() -> None:
    assert (
        compute_memory_pressure(
            ram_total_bytes=100,
            ram_available_bytes=40,
            macos_memory_pressure_level=99,
        )
        == 0.6
    )


def test_parse_linux_stall_average_10() -> None:
    contents = (
        "some avg10=12.50 avg60=1.00 avg300=0.50 total=123\n"
        "full avg10=0.10 avg60=0.05 avg300=0.01 total=4\n"
    )
    assert parse_linux_memory_pressure_stall_average_10(contents) == 12.5


def test_parse_linux_stall_missing_some_line() -> None:
    assert (
        parse_linux_memory_pressure_stall_average_10(
            "full avg10=9.00 avg60=0.00 avg300=0.00 total=0\n"
        )
        is None
    )


def test_parse_linux_stall_invalid_avg10() -> None:
    assert (
        parse_linux_memory_pressure_stall_average_10(
            "some avg10=not-a-number avg60=0.00 avg300=0.00 total=0\n"
        )
        is None
    )


def test_parse_macos_pressure_level_sysctl_form() -> None:
    assert (
        parse_macos_memory_pressure_level("kern.memorystatus_vm_pressure_level: 1\n")
        == 1
    )


def test_parse_macos_pressure_level_bare_integer() -> None:
    assert parse_macos_memory_pressure_level("4") == 4


def test_parse_macos_pressure_level_invalid() -> None:
    assert parse_macos_memory_pressure_level("not-an-int") is None


def test_read_linux_stall_from_file(tmp_path: Path) -> None:
    pressure_file = tmp_path / "memory"
    pressure_file.write_text(
        "some avg10=3.25 avg60=0.00 avg300=0.00 total=1\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    )
    assert (
        read_linux_memory_pressure_stall_average_10(pressure_file=pressure_file) == 3.25
    )


def test_read_linux_stall_missing_file(tmp_path: Path) -> None:
    assert (
        read_linux_memory_pressure_stall_average_10(pressure_file=tmp_path / "missing")
        is None
    )


def test_memory_usage_from_bytes_fills_pressure() -> None:
    usage = MemoryUsage.from_bytes(
        ram_total=1000,
        ram_available=250,
        swap_total=0,
        swap_available=0,
    )
    assert usage.memory_pressure == 0.75


def test_memory_usage_from_bytes_uses_platform_signals() -> None:
    usage = MemoryUsage.from_bytes(
        ram_total=1000,
        ram_available=800,
        swap_total=0,
        swap_available=0,
        linux_pressure_stall_average_10=90.0,
    )
    assert usage.memory_pressure == 0.9


def test_memory_usage_legacy_payload_without_pressure_is_filled() -> None:
    usage = MemoryUsage.model_validate(
        {
            "ramTotal": {"inBytes": 100},
            "ramAvailable": {"inBytes": 25},
            "swapTotal": {"inBytes": 0},
            "swapAvailable": {"inBytes": 0},
        }
    )
    assert usage.memory_pressure == 0.75
    dumped = usage.model_dump(by_alias=True)
    assert dumped["memoryPressure"] == 0.75


def test_from_psutil_uses_available_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    class _VirtualMemory:
        total = 1000
        available = 400
        used = 900

    class _SwapMemory:
        total = 0
        free = 0

    monkeypatch.setattr(
        "exo.shared.types.profiling.psutil.virtual_memory",
        lambda: _VirtualMemory(),
    )
    monkeypatch.setattr(
        "exo.shared.types.profiling.psutil.swap_memory",
        lambda: _SwapMemory(),
    )
    usage = MemoryUsage.from_psutil(override_memory=None)
    assert usage.ram_available.in_bytes == 400
    assert usage.memory_pressure == 0.6


def test_from_psutil_override_memory_and_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _VirtualMemory:
        total = 2000
        available = 1800
        used = 200

    class _SwapMemory:
        total = 0
        free = 0

    monkeypatch.setattr(
        "exo.shared.types.profiling.psutil.virtual_memory",
        lambda: _VirtualMemory(),
    )
    monkeypatch.setattr(
        "exo.shared.types.profiling.psutil.swap_memory",
        lambda: _SwapMemory(),
    )
    usage = MemoryUsage.from_psutil(
        override_memory=200,
        linux_pressure_stall_average_10=10.0,
    )
    assert usage.ram_available.in_bytes == 200
    assert usage.memory_pressure == 0.9


def _macmon_raw(*, ram_total: int, ram_usage: int) -> RawMacmonMetrics:
    return RawMacmonMetrics.model_validate(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "temp": {"cpu_temp_avg": 40.0, "gpu_temp_avg": 41.0},
            "memory": {
                "ram_total": ram_total,
                "ram_usage": ram_usage,
                "swap_total": 1000,
                "swap_usage": 0,
            },
            "ecpu_usage": [1000, 0.1],
            "pcpu_usage": [2000, 0.2],
            "gpu_usage": [800, 0.3],
            "all_power": 10.0,
            "ane_power": 1.0,
            "cpu_power": 5.0,
            "gpu_power": 3.0,
            "gpu_ram_power": 0.5,
            "ram_power": 0.5,
            "sys_power": 20.0,
        }
    )


def test_macmon_fallback_without_available_uses_total_minus_used() -> None:
    metrics = MacmonMetrics.from_raw(_macmon_raw(ram_total=16_000, ram_usage=12_000))
    assert metrics.memory.ram_available.in_bytes == 4_000
    assert metrics.memory.memory_pressure == 0.75


def test_macmon_prefers_reclaimable_available_over_used() -> None:
    metrics = MacmonMetrics.from_raw(
        _macmon_raw(ram_total=16_000, ram_usage=15_000),
        ram_available=8_000,
    )
    assert metrics.memory.ram_available.in_bytes == 8_000
    assert metrics.memory.memory_pressure == 0.5


def test_macmon_macos_level_can_exceed_scarcity() -> None:
    metrics = MacmonMetrics.from_raw(
        _macmon_raw(ram_total=16_000, ram_usage=4_000),
        ram_available=12_000,
        macos_memory_pressure_level=4,
    )
    assert metrics.memory.ram_available.in_bytes == 12_000
    assert metrics.memory.memory_pressure == 1.0


def test_apply_stores_memory_pressure_on_node() -> None:
    node_id = NodeId()
    usage = MemoryUsage.from_bytes(
        ram_total=1000,
        ram_available=200,
        swap_total=0,
        swap_available=0,
        linux_pressure_stall_average_10=85.0,
    )
    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=node_id,
            when="2026-01-01T00:00:00+00:00",
            info=usage,
        ),
        State(),
    )
    stored = state.node_memory[node_id]
    assert stored.ram_available.in_bytes == 200
    assert stored.memory_pressure == 0.85
