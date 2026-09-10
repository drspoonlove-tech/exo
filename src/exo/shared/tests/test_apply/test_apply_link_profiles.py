from datetime import datetime, timezone

from exo.shared.apply import apply_node_gathered_info, apply_node_timed_out
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeGatheredInfo, NodeTimedOut
from exo.shared.types.profiling import LinkProfile
from exo.shared.types.state import State
from exo.utils.info_gatherer.info_gatherer import NodeLinkProfiles


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_apply_node_link_profiles_replaces_per_node() -> None:
    source = NodeId("node-a")
    remote = NodeId("node-b")
    first = LinkProfile(
        remote_node_id=remote,
        remote_ip="10.0.0.2",
        latency_ms=4.0,
        bandwidth_bps=1_000_000_000,
        bandwidth_source="interface_type",
    )
    second = first.model_copy(update={"latency_ms": 1.25})

    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=source,
            when=_now(),
            info=NodeLinkProfiles(profiles=[first]),
        ),
        State(),
    )
    assert state.node_link_profiles[source] == (first,)

    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=source,
            when=_now(),
            info=NodeLinkProfiles(profiles=[second]),
        ),
        state,
    )
    assert state.node_link_profiles[source] == (second,)


def test_apply_node_timed_out_drops_link_profiles() -> None:
    source = NodeId("node-a")
    other = NodeId("node-b")
    profile = LinkProfile(
        remote_node_id=other,
        remote_ip="10.0.0.2",
        latency_ms=3.0,
        bandwidth_bps=1_000_000_000,
    )
    other_profile = LinkProfile(
        remote_node_id=source,
        remote_ip="10.0.0.1",
        latency_ms=3.5,
        bandwidth_bps=1_000_000_000,
    )
    state = State(
        node_link_profiles={
            source: (profile,),
            other: (other_profile,),
        }
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=source), state)
    assert source not in new_state.node_link_profiles
    assert new_state.node_link_profiles[other] == (other_profile,)
