from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from loguru import logger

from exo.shared.logging import (
    console_default_level,
    console_filter_levels,
    file_filter_levels,
    parse_log_filters,
)

if TYPE_CHECKING:
    from loguru import Record


def _log_as(logger_name: str, level: str, message: str) -> None:
    def set_name(record: Record) -> None:
        record["name"] = logger_name

    logger.patch(set_name).log(level, message)


def test_parse_log_filters_empty() -> None:
    assert parse_log_filters("") == {}
    assert parse_log_filters("  , , ") == {}


def test_parse_log_filters_single_and_multiple() -> None:
    assert parse_log_filters("exo.download=DEBUG") == {"exo.download": "DEBUG"}
    assert parse_log_filters("exo.download=DEBUG,httpx=WARNING") == {
        "exo.download": "DEBUG",
        "httpx": "WARNING",
    }


def test_parse_log_filters_whitespace_and_case() -> None:
    assert parse_log_filters(" exo.download = debug , httpx = warning ") == {
        "exo.download": "DEBUG",
        "httpx": "WARNING",
    }


@pytest.mark.parametrize(
    "spec",
    [
        "exo.download",
        "=DEBUG",
        "exo.download=NOPE",
        "exo.download=INFO,broken",
    ],
)
def test_parse_log_filters_rejects_invalid(spec: str) -> None:
    with pytest.raises(ValueError, match="Invalid log"):
        parse_log_filters(spec)


def test_console_default_follows_verbosity() -> None:
    assert console_default_level(-1) == "WARNING"
    assert console_default_level(0) == "INFO"
    assert console_default_level(1) == "DEBUG"


def test_file_filter_defaults_to_debug() -> None:
    assert file_filter_levels() == {"": "DEBUG"}
    assert file_filter_levels({"httpx": "WARNING"}) == {
        "": "DEBUG",
        "httpx": "WARNING",
    }


def test_module_debug_passes_while_other_module_stays_info() -> None:
    filter_levels = console_filter_levels(0, {"exo.download": "DEBUG"})
    captured: list[str] = []

    def sink(message: str) -> None:
        captured.append(message.rstrip("\n"))

    handler_id = logger.add(
        sink,
        level="DEBUG",
        filter=filter_levels,
        format="{name}|{level}|{message}",
        colorize=False,
        enqueue=False,
    )
    try:
        _log_as("exo.download.coordinator", "DEBUG", "download debug")
        _log_as("exo.master", "DEBUG", "master debug")
        _log_as("exo.master", "INFO", "master info")
    finally:
        logger.remove(handler_id)

    assert "exo.download.coordinator|DEBUG|download debug" in captured
    assert "exo.master|DEBUG|master debug" not in captured
    assert "exo.master|INFO|master info" in captured


def test_verbosity_debug_lets_all_modules_through() -> None:
    filter_levels = console_filter_levels(1)
    captured: list[str] = []

    def sink(message: str) -> None:
        captured.append(message.rstrip("\n"))

    handler_id = logger.add(
        sink,
        level="DEBUG",
        filter=filter_levels,
        format="{name}|{level}|{message}",
        colorize=False,
        enqueue=False,
    )
    try:
        _log_as("exo.download", "DEBUG", "download debug")
        _log_as("exo.master", "DEBUG", "master debug")
    finally:
        logger.remove(handler_id)

    assert "exo.download|DEBUG|download debug" in captured
    assert "exo.master|DEBUG|master debug" in captured
