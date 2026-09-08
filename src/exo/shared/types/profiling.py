import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import psutil
from pydantic import Field, model_validator

from exo.shared.memory_pressure import compute_memory_pressure
from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel


def _memory_field_to_bytes(value: object) -> int | None:
    if isinstance(value, Memory):
        return value.in_bytes
    if isinstance(value, dict):
        in_bytes = value.get("in_bytes", value.get("inBytes"))
        if isinstance(in_bytes, int):
            return in_bytes
    return None


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory
    memory_pressure: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _default_memory_pressure(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "memory_pressure" in data or "memoryPressure" in data:
            return data
        ram_total_bytes = _memory_field_to_bytes(
            data.get("ram_total", data.get("ramTotal"))
        )
        ram_available_bytes = _memory_field_to_bytes(
            data.get("ram_available", data.get("ramAvailable"))
        )
        if ram_total_bytes is None or ram_available_bytes is None:
            return data
        return {
            **data,
            "memory_pressure": compute_memory_pressure(
                ram_total_bytes=ram_total_bytes,
                ram_available_bytes=ram_available_bytes,
            ),
        }

    @classmethod
    def from_bytes(
        cls,
        *,
        ram_total: int,
        ram_available: int,
        swap_total: int,
        swap_available: int,
        linux_pressure_stall_average_10: float | None = None,
        macos_memory_pressure_level: int | None = None,
        memory_pressure: float | None = None,
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            memory_pressure=(
                memory_pressure
                if memory_pressure is not None
                else compute_memory_pressure(
                    ram_total_bytes=ram_total,
                    ram_available_bytes=ram_available,
                    linux_pressure_stall_average_10=linux_pressure_stall_average_10,
                    macos_memory_pressure_level=macos_memory_pressure_level,
                )
            ),
        )

    @classmethod
    def from_psutil(
        cls,
        *,
        override_memory: int | None,
        linux_pressure_stall_average_10: float | None = None,
        macos_memory_pressure_level: int | None = None,
    ) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
            linux_pressure_stall_average_10=linux_pressure_stall_average_10,
            macos_memory_pressure_level=macos_memory_pressure_level,
        )


class DiskUsage(FrozenModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(FrozenModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(FrozenModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(FrozenModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"


class NodeNetworkInfo(FrozenModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(FrozenModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(FrozenModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(FrozenModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
