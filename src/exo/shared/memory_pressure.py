import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

LINUX_MEMORY_PRESSURE_FILE: Final[Path] = Path("/proc/pressure/memory")

MACOS_MEMORY_PRESSURE_LEVEL_TO_RATIO: Final[Mapping[int, float]] = {
    0: 0.0,  # normal
    1: 0.5,  # warning
    2: 0.8,  # urgent
    4: 1.0,  # critical
}


def compute_memory_pressure(
    *,
    ram_total_bytes: int,
    ram_available_bytes: int,
    linux_pressure_stall_average_10: float | None = None,
    macos_memory_pressure_level: int | None = None,
) -> float:
    """Return a 0.0–1.0 pressure ratio from available RAM plus platform signals.

    Scarcity is ``1 - available/total`` using reclaimable memory (MemAvailable /
    psutil ``available``), not ``used/total``. Linux stall ``avg10`` and macOS
    ``kern.memorystatus_vm_pressure_level`` can raise the result when the
    machine is thrashing even if some pages still look free.
    """
    if ram_total_bytes <= 0:
        scarcity = 1.0
    else:
        bounded_available = min(max(ram_available_bytes, 0), ram_total_bytes)
        scarcity = 1.0 - (bounded_available / ram_total_bytes)

    candidates = [scarcity]

    if linux_pressure_stall_average_10 is not None:
        candidates.append(
            min(max(linux_pressure_stall_average_10 / 100.0, 0.0), 1.0)
        )

    if macos_memory_pressure_level is not None:
        mapped = MACOS_MEMORY_PRESSURE_LEVEL_TO_RATIO.get(
            macos_memory_pressure_level
        )
        if mapped is not None:
            candidates.append(mapped)

    return max(candidates)


def parse_linux_memory_pressure_stall_average_10(contents: str) -> float | None:
    """Parse ``some avg10`` from ``/proc/pressure/memory`` text.

    Returns ``None`` when the ``some`` line or ``avg10`` token is missing or
    not a number. Callers treat ``None`` as “no stall signal”.
    """
    for line in contents.splitlines():
        if not line.startswith("some "):
            continue
        for token in line.split():
            if token.startswith("avg10="):
                try:
                    return float(token.removeprefix("avg10="))
                except ValueError:
                    return None
    return None


def parse_macos_memory_pressure_level(sysctl_output: str) -> int | None:
    """Parse ``sysctl kern.memorystatus_vm_pressure_level`` stdout.

    Accepts ``kern.memorystatus_vm_pressure_level: 1`` or a bare integer.
    Returns ``None`` when the value is not an integer. Callers treat ``None``
    as “no macOS pressure level”.
    """
    stripped = sysctl_output.strip()
    if ":" in stripped:
        stripped = stripped.split(":", 1)[1].strip()
    try:
        return int(stripped)
    except ValueError:
        return None


def read_linux_memory_pressure_stall_average_10(
    *,
    pressure_file: Path = LINUX_MEMORY_PRESSURE_FILE,
) -> float | None:
    """Read Linux memory stall ``avg10``, or ``None`` if the file is unavailable.

    ``OSError`` (missing file, permission) is handled here so gatherers can
    call this on any platform without a local try/except.
    """
    try:
        contents = pressure_file.read_text()
    except OSError:
        return None
    return parse_linux_memory_pressure_stall_average_10(contents)


def read_macos_memory_pressure_level() -> int | None:
    """Read macOS ``kern.memorystatus_vm_pressure_level``, or ``None``.

    Missing ``sysctl``, non-Darwin hosts, timeouts, and non-zero exits are
    handled here so gatherers can call this without a local try/except.
    """
    if sys.platform != "darwin":
        return None

    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, TimeoutError):
        return None
    if result.returncode != 0:
        return None
    return parse_macos_memory_pressure_level(result.stdout)
