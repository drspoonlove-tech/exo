import shutil
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self, final

import anyio

from exo.shared.types.profiling import (
    IbvDevicePort,
    IbvDeviceSummary,
    NodeIbvDevinfoStatus,
)
from exo.utils.pydantic_ext import TaggedModel

IBV_DEVINFO_BINARY = "ibv_devinfo"
SYSFS_INFINIBAND_CLASS_PATH = Path("/sys/class/infiniband")
_USABLE_PORT_STATES = frozenset({"PORT_ACTIVE", "PORT_ACTIVE_DEFER"})


@final
@dataclass(frozen=True)
class IbvDevinfoCommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


type LocateBinary = Callable[[str], str | None]
type RunIbvDevinfo = Callable[[str], Awaitable[IbvDevinfoCommandResult]]


def claimed_rdma_device_names_from_sysfs(
    infiniband_class_path: Path = SYSFS_INFINIBAND_CLASS_PATH,
) -> tuple[str, ...]:
    """Device names advertised by sysfs, if the InfiniBand class exists."""
    if not infiniband_class_path.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in infiniband_class_path.iterdir()
            if entry.is_dir() or entry.is_symlink()
        )
    )


def ibv_port_is_usable(state: str) -> bool:
    return state in _USABLE_PORT_STATES


def parse_ibv_devinfo_output(output: str) -> tuple[IbvDeviceSummary, ...]:
    """Parse ``ibv_devinfo`` text into device/port summaries.

    Expected input is the default (non-verbose) ``ibv_devinfo`` listing:
    ``hca_id`` blocks with nested ``port:`` / ``state:`` / ``link_layer:`` lines.
    """
    devices: list[IbvDeviceSummary] = []
    current_name: str | None = None
    current_transport = ""
    current_ports: list[IbvDevicePort] = []
    current_port_number: int | None = None
    current_port_state = "UNKNOWN"
    current_link_layer = ""

    def flush_port() -> None:
        nonlocal current_port_number, current_port_state, current_link_layer
        if current_port_number is not None:
            current_ports.append(
                IbvDevicePort(
                    port_number=current_port_number,
                    state=current_port_state,
                    link_layer=current_link_layer,
                )
            )
        current_port_number = None
        current_port_state = "UNKNOWN"
        current_link_layer = ""

    def flush_device() -> None:
        nonlocal current_name, current_transport, current_ports
        flush_port()
        if current_name is not None:
            devices.append(
                IbvDeviceSummary(
                    name=current_name,
                    transport=current_transport,
                    ports=tuple(current_ports),
                )
            )
        current_name = None
        current_transport = ""
        current_ports = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("hca_id:"):
            flush_device()
            current_name = line.split(":", 1)[1].strip()
            continue
        if current_name is None:
            continue
        if line.startswith("transport:"):
            current_transport = line.split(":", 1)[1].strip()
            continue
        if line.startswith("port:"):
            flush_port()
            port_token = line.split(":", 1)[1].strip().split()[0]
            current_port_number = int(port_token)
            continue
        if line.startswith("state:"):
            current_port_state = line.split(":", 1)[1].strip().split()[0]
            continue
        if line.startswith("link_layer:"):
            current_link_layer = line.split(":", 1)[1].strip()

    flush_device()
    return tuple(devices)


def evaluate_ibv_devinfo(
    *,
    binary_path: str | None,
    command_result: IbvDevinfoCommandResult | None,
    claimed_device_names: Sequence[str] = (),
) -> NodeIbvDevinfoStatus:
    """Pure pass/fail evaluation of an ``ibv_devinfo`` invocation.

    ``command_result`` is ``None`` when the binary could not be executed
    (missing, timeout, or OS error). Callers that catch those conditions
    should pass ``None`` rather than inventing a process result.
    """
    if binary_path is None:
        return NodeIbvDevinfoStatus(
            ok=False,
            failure="ibv_devinfo not found in PATH",
            missing_claimed_devices=tuple(claimed_device_names),
        )
    if command_result is None:
        return NodeIbvDevinfoStatus(
            ok=False,
            failure="ibv_devinfo failed to run",
            missing_claimed_devices=tuple(claimed_device_names),
        )

    stdout_text = command_result.stdout.decode("utf-8", errors="replace")
    stderr_text = command_result.stderr.decode("utf-8", errors="replace").strip()
    devices = parse_ibv_devinfo_output(stdout_text)
    device_names = {device.name for device in devices}
    missing_claimed = tuple(
        name for name in claimed_device_names if name not in device_names
    )

    failures: list[str] = []
    if not devices:
        if command_result.returncode != 0:
            detail = stderr_text or f"exit {command_result.returncode}"
            failures.append(f"ibv_devinfo reported no RDMA devices ({detail})")
        else:
            failures.append("ibv_devinfo reported no RDMA devices")
    if missing_claimed:
        failures.append(
            "claimed RDMA devices not present in ibv_devinfo: "
            + ", ".join(missing_claimed)
        )
    for device in devices:
        if claimed_device_names and device.name not in claimed_device_names:
            continue
        if not device.ports:
            failures.append(f"RDMA device {device.name} has no ports")
            continue
        if not any(ibv_port_is_usable(port.state) for port in device.ports):
            port_desc = ", ".join(
                f"port {port.port_number}={port.state}" for port in device.ports
            )
            failures.append(
                f"RDMA device {device.name} has no usable port ({port_desc})"
            )

    return NodeIbvDevinfoStatus(
        ok=len(failures) == 0,
        failure="; ".join(failures) if failures else None,
        devices=devices,
        missing_claimed_devices=missing_claimed,
    )


async def run_ibv_devinfo_command(binary_path: str) -> IbvDevinfoCommandResult:
    """Execute ``ibv_devinfo``. Timeouts and OS errors propagate to the caller."""
    with anyio.fail_after(5):
        proc = await anyio.run_process([binary_path], check=False)
    return IbvDevinfoCommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


class IbvDevinfoStatus(TaggedModel):
    """Gathered ``ibv_devinfo`` validation for a node."""

    status: NodeIbvDevinfoStatus

    @classmethod
    async def gather(
        cls,
        *,
        claimed_device_names: Sequence[str] = (),
        locate_binary: LocateBinary | None = None,
        run_ibv_devinfo: RunIbvDevinfo | None = None,
    ) -> Self:
        """Run ``ibv_devinfo`` and record whether claimed devices are usable.

        ``FileNotFoundError``, ``TimeoutError``, and ``OSError`` from the
        runner are converted into a failed status here so the gatherer can
        surface them instead of crashing the monitor loop.
        """
        resolve_binary = shutil.which if locate_binary is None else locate_binary
        runner = run_ibv_devinfo_command if run_ibv_devinfo is None else run_ibv_devinfo
        binary_path = resolve_binary(IBV_DEVINFO_BINARY)
        if binary_path is None:
            return cls(
                status=evaluate_ibv_devinfo(
                    binary_path=None,
                    command_result=None,
                    claimed_device_names=claimed_device_names,
                )
            )
        try:
            command_result = await runner(binary_path)
        except (TimeoutError, OSError):
            return cls(
                status=evaluate_ibv_devinfo(
                    binary_path=binary_path,
                    command_result=None,
                    claimed_device_names=claimed_device_names,
                )
            )
        return cls(
            status=evaluate_ibv_devinfo(
                binary_path=binary_path,
                command_result=command_result,
                claimed_device_names=claimed_device_names,
            )
        )
