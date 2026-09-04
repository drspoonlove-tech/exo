import logging
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final, Literal

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

_MAX_LOG_ARCHIVES = 5

type LogLevelName = Literal[
    "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
]

_LOG_LEVEL_NAMES: Final[tuple[LogLevelName, ...]] = (
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
)


def _as_log_level(level_name: str) -> LogLevelName:
    normalized = level_name.strip().upper()
    for candidate in _LOG_LEVEL_NAMES:
        if normalized == candidate:
            return candidate
    raise ValueError(
        f"Invalid log level {level_name!r}: expected one of {', '.join(_LOG_LEVEL_NAMES)}"
    )


def parse_log_filters(spec: str) -> dict[str, LogLevelName]:
    """Parse ``exo.download=DEBUG,httpx=WARNING`` into a logger-name → level map.

    Raises:
        ValueError: if an entry is missing ``=``, the logger name is empty, or
            the level is unknown. Handled at CLI parse time so a typo fails
            fast instead of being ignored.
    """
    filters: dict[str, LogLevelName] = {}
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid log filter {entry!r}: expected LOGGER=LEVEL")
        logger_name, level_name = entry.split("=", 1)
        logger_name = logger_name.strip()
        if not logger_name:
            raise ValueError(f"Invalid log filter {entry!r}: logger name is empty")
        filters[logger_name] = _as_log_level(level_name)
    return filters


def console_default_level(verbosity: int) -> LogLevelName:
    if verbosity > 0:
        return "DEBUG"
    if verbosity < 0:
        return "WARNING"
    return "INFO"


def _logger_filter_dict(
    default_level: LogLevelName,
    logger_levels: Mapping[str, LogLevelName] | None,
) -> dict[str | None, str | int | bool]:
    filters: dict[str | None, str | int | bool] = {"": default_level}
    if logger_levels is not None:
        for logger_name, level_name in logger_levels.items():
            filters[logger_name] = level_name
    return filters


def console_filter_levels(
    verbosity: int,
    logger_levels: Mapping[str, LogLevelName] | None = None,
) -> dict[str | None, str | int | bool]:
    """Loguru filter dict for the console sink.

    Unmatched loggers use the verbosity default; ``logger_levels`` overrides
    specific name prefixes (``exo.download`` matches ``exo.download.coordinator``).
    """
    return _logger_filter_dict(console_default_level(verbosity), logger_levels)


def file_filter_levels(
    logger_levels: Mapping[str, LogLevelName] | None = None,
) -> dict[str | None, str | int | bool]:
    """Loguru filter dict for the file sink. Default is always DEBUG."""
    return _logger_filter_dict("DEBUG", logger_levels)


def _zstd_compress(filepath: str) -> None:
    source = Path(filepath)
    dest = source.with_suffix(source.suffix + ".zst")
    cctx = zstandard.ZstdCompressor()
    with open(source, "rb") as f_in, open(dest, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    source.unlink()


def _once_then_never() -> Iterator[bool]:
    yield True
    while True:
        yield False


class InterceptLogger(HypercornLogger):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.error_logger
        self.error_logger.handlers = [_InterceptHandler()]


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=3, exception=record.exc_info).log(level, record.getMessage())


def logger_setup(
    log_file: Path | None,
    verbosity: int = 0,
    logger_levels: Mapping[str, LogLevelName] | None = None,
):
    """Set up logging for this process - formatting, file handles, verbosity and output.

    Console default is INFO (WARNING if ``verbosity < 0``, DEBUG if ``verbosity > 0``).
    The file sink is always DEBUG so disk keeps more detail than the console.
    Both sinks take a loguru filter dict so ``logger_levels`` can raise or lower
    a logger-name prefix independently of those defaults. Sink ``level`` is
    DEBUG because loguru applies ``level`` before ``filter``.
    """

    logging.getLogger("exo_rs").setLevel(logging.INFO)
    logging.getLogger("networking").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.remove()

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    if verbosity == 0:
        console_format = "[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>"
    else:
        console_format = (
            "[ {time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | "
            "{name}:{function}:{line} ] <level>{message}</level>"
        )
    logger.add(
        sys.__stderr__,  # type: ignore
        format=console_format,
        level="DEBUG",
        filter=console_filter_levels(verbosity, logger_levels),
        colorize=True,
        enqueue=True,
    )
    if log_file:
        rotate_once = _once_then_never()
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level="DEBUG",
            filter=file_filter_levels(logger_levels),
            colorize=False,
            enqueue=True,
            rotation=lambda _, __: next(rotate_once),
            retention=_MAX_LOG_ARCHIVES,
            compression=_zstd_compress,
        )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()


""" --- TODO: Capture MLX Log output:
import contextlib
import sys
from loguru import logger

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass

logger.remove()
logger.add(sys.__stdout__)

stream = StreamToLogger()
with contextlib.redirect_stdout(stream):
    print("Standard output is sent to added handlers.")
"""
