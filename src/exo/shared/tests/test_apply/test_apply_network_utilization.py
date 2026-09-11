from datetime import datetime, timezone

from exo.shared.apply import apply_node_gathered_info, apply_node_timed_out
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeGatheredInfo, NodeTimedOut
from exo.shared.types.profiling import (
    NetworkInterfaceUtilization,
    NodeNetworkUtilization,
)
from exo.shared.types.state import State
from exo.utils.info_gatherer.info_gatherer import NodeNetworkUtilizationSample


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_apply_node_network_utilization_replaces_per_node() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    state = State(
        node_network_utilization={
            node_b: NodeNetworkUtilization(
                interfaces=[
                    NetworkInterfaceUtilization(
                        name="en0",
                        bytes_sent_per_sec=1.0,
                        bytes_recv_per_sec=2.0,
                    )
                ]
            )
        }
    )
    event = NodeGatheredInfo(
        node_id=node_a,
        when=_now(),
        info=NodeNetworkUtilizationSample(
            interfaces=[
                NetworkInterfaceUtilization(
                    name="eth0",
                    bytes_sent_per_sec=1_250_000.0,
                    bytes_recv_per_sec=400_000.0,
                )
            ]
        ),
    )

    new_state = apply_node_gathered_info(event, state)

    assert node_b in new_state.node_network_utilization
    reported = new_state.node_network_utilization[node_a]
    assert len(reported.interfaces) == 1
    assert reported.interfaces[0].name == "eth0"
    assert reported.interfaces[0].bytes_sent_per_sec == 1_250_000.0


def test_apply_node_timed_out_drops_network_utilization() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    state = State(
        node_network_utilization={
            node_a: NodeNetworkUtilization(
                interfaces=[
                    NetworkInterfaceUtilization(
                        name="eth0",
                        bytes_sent_per_sec=10.0,
                        bytes_recv_per_sec=5.0,
                    )
                ]
            ),
            node_b: NodeNetworkUtilization(
                interfaces=[
                    NetworkInterfaceUtilization(
                        name="eth1",
                        bytes_sent_per_sec=3.0,
                        bytes_recv_per_sec=1.0,
                    )
                ]
            ),
        }
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=node_a), state)

    assert node_a not in new_state.node_network_utilization
    assert node_b in new_state.node_network_utilization
