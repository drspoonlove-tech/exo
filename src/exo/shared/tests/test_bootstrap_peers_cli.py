"""CLI honesty for the temporarily removed bootstrap-peers feature."""

import os
from collections.abc import Iterator
from unittest import mock

import pytest

from exo.main import BOOTSTRAP_PEERS_REMOVED_MESSAGE, Args

BOOTSTRAP_PEER_MULTIADDR: str = "/ip4/127.0.0.1/tcp/4001/p2p/12D3KooWExample"


@pytest.fixture
def without_bootstrap_peers_environment() -> Iterator[None]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "EXO_BOOTSTRAP_PEERS"
    }
    with mock.patch.dict(os.environ, environment, clear=True):
        yield


def test_help_does_not_advertise_bootstrap_peers_as_live(
    capsys: pytest.CaptureFixture[str],
    without_bootstrap_peers_environment: None,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        Args.parse(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out.casefold()
    assert "dial on startup" not in help_text
    assert "removed" in help_text
    assert "--bootstrap-peers" in help_text


def test_bootstrap_peers_flag_fails_as_removed(
    without_bootstrap_peers_environment: None,
) -> None:
    with pytest.raises(ValueError, match="temporarily removed") as error_info:
        Args.parse(["--bootstrap-peers", BOOTSTRAP_PEER_MULTIADDR])
    assert str(error_info.value) == BOOTSTRAP_PEERS_REMOVED_MESSAGE


def test_bootstrap_peers_environment_fails_as_removed() -> None:
    with (
        mock.patch.dict(
            os.environ, {"EXO_BOOTSTRAP_PEERS": BOOTSTRAP_PEER_MULTIADDR}
        ),
        pytest.raises(ValueError, match="temporarily removed") as error_info,
    ):
        Args.parse([])
    assert str(error_info.value) == BOOTSTRAP_PEERS_REMOVED_MESSAGE
